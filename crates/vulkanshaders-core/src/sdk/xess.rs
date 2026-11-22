// SPDX-License-Identifier: GPL-3.0-only
// Additional permissions under GPLv3 Section 7 apply to optional proprietary vendor SDK
// runtime loading; see LICENSE-EXCEPTION.md in the repository root.

//! Clean-room Intel XeSS Vulkan runtime dynamic loader (`libxess.so` / `libxess.dll`).
//!
//! Binds the exact C-ABI structs and entry points from `xess.h` / `xess_vk.h` via `libloading`
//! (`xessVKCreateContext`, `xessVKInit`, `xessVKExecute`, `xessDestroyContext`, plus query and
//! Vulkan extension-requirement helpers).

use ash::vk;
use libloading::Library;
use std::{
    ffi::{CStr, c_char, c_void},
    fmt,
    path::{Path, PathBuf},
    ptr,
};
use thiserror::Error;

/// Opaque XeSS context handle (`xess_context_handle_t`).
#[repr(C)]
pub struct XessContextHandle {
    _private: [u8; 0],
}

pub type XessContextPtr = *mut XessContextHandle;

#[derive(Debug, Error)]
pub enum XessError {
    #[error("failed to load Intel XeSS library ({0})")]
    LibraryLoad(String),
    #[error("missing required XeSS symbol `{0}`: {1}")]
    MissingSymbol(&'static str, String),
    #[error("XeSS API call `{call}` failed with {result}")]
    ApiFailure {
        call: &'static str,
        result: XessResult,
    },
    #[error("null XeSS context handle")]
    NullContext,
}

/// `xess_version_t` (8 bytes, align 2).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct XessVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
    pub reserved: u16,
}

impl fmt::Display for XessVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// `xess_2d_t` / `xess_coord_t` (8 bytes, align 4).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Xess2d {
    pub x: u32,
    pub y: u32,
}

/// `xess_quality_settings_t` (`int32_t`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum XessQualitySettings {
    UltraPerformance = 100,
    Performance = 101,
    Balanced = 102,
    Quality = 103,
    UltraQuality = 104,
    UltraQualityPlus = 105,
    AntiAliasing = 106,
}

/// `xess_init_flags_t` bitflags.
pub mod init_flags {
    pub const NONE: u32 = 0;
    pub const HIGH_RES_MV: u32 = 1 << 0;
    pub const INVERTED_DEPTH: u32 = 1 << 1;
    pub const EXPOSURE_SCALE_TEXTURE: u32 = 1 << 2;
    pub const RESPONSIVE_PIXEL_MASK: u32 = 1 << 3;
    pub const USE_NDC_VELOCITY: u32 = 1 << 4;
    pub const EXTERNAL_DESCRIPTOR_HEAP: u32 = 1 << 5;
    pub const LDR_INPUT_COLOR: u32 = 1 << 6;
    pub const JITTERED_MV: u32 = 1 << 7;
    pub const ENABLE_AUTOEXPOSURE: u32 = 1 << 8;
}

/// `xess_result_t` (`int32_t`).
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct XessResult(pub i32);

impl XessResult {
    pub const WARNING_NONEXISTING_FOLDER: Self = Self(1);
    pub const WARNING_OLD_DRIVER: Self = Self(2);
    pub const SUCCESS: Self = Self(0);
    pub const ERROR_UNSUPPORTED_DEVICE: Self = Self(-1);
    pub const ERROR_UNSUPPORTED_DRIVER: Self = Self(-2);
    pub const ERROR_UNINITIALIZED: Self = Self(-3);
    pub const ERROR_INVALID_ARGUMENT: Self = Self(-4);
    pub const ERROR_DEVICE_OUT_OF_MEMORY: Self = Self(-5);
    pub const ERROR_DEVICE: Self = Self(-6);
    pub const ERROR_NOT_IMPLEMENTED: Self = Self(-7);
    pub const ERROR_INVALID_CONTEXT: Self = Self(-8);
    pub const ERROR_OPERATION_IN_PROGRESS: Self = Self(-9);
    pub const ERROR_UNSUPPORTED: Self = Self(-10);
    pub const ERROR_CANT_LOAD_LIBRARY: Self = Self(-11);
    pub const ERROR_WRONG_CALL_ORDER: Self = Self(-12);
    pub const ERROR_UNKNOWN: Self = Self(-1000);

    #[inline]
    pub const fn is_success(self) -> bool {
        self.0 >= 0
    }
}

impl fmt::Debug for XessResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "XessResult({})", self.0)
    }
}

impl fmt::Display for XessResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self.0 {
            0 => "XESS_RESULT_SUCCESS",
            1 => "XESS_RESULT_WARNING_NONEXISTING_FOLDER",
            2 => "XESS_RESULT_WARNING_OLD_DRIVER",
            -1 => "XESS_RESULT_ERROR_UNSUPPORTED_DEVICE",
            -2 => "XESS_RESULT_ERROR_UNSUPPORTED_DRIVER",
            -3 => "XESS_RESULT_ERROR_UNINITIALIZED",
            -4 => "XESS_RESULT_ERROR_INVALID_ARGUMENT",
            -5 => "XESS_RESULT_ERROR_DEVICE_OUT_OF_MEMORY",
            -6 => "XESS_RESULT_ERROR_DEVICE",
            -7 => "XESS_RESULT_ERROR_NOT_IMPLEMENTED",
            -8 => "XESS_RESULT_ERROR_INVALID_CONTEXT",
            -9 => "XESS_RESULT_ERROR_OPERATION_IN_PROGRESS",
            -10 => "XESS_RESULT_ERROR_UNSUPPORTED",
            -11 => "XESS_RESULT_ERROR_CANT_LOAD_LIBRARY",
            -12 => "XESS_RESULT_ERROR_WRONG_CALL_ORDER",
            _ => "XESS_RESULT_ERROR_UNKNOWN",
        };
        write!(f, "{label} ({})", self.0)
    }
}

/// `xess_properties_t` (24 bytes, align 8).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct XessProperties {
    pub required_descriptor_count: u32,
    pub temp_buffer_heap_size: u64,
    pub temp_texture_heap_size: u64,
}

/// `xess_vk_image_view_info` (48 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct XessVkImageViewInfo {
    pub image_view: vk::ImageView,
    pub image: vk::Image,
    pub subresource_range: vk::ImageSubresourceRange,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
}

/// `xess_vk_init_params_t` (64 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct XessVkInitParams {
    pub output_resolution: Xess2d,
    pub quality_setting: XessQualitySettings,
    pub init_flags: u32,
    pub creation_node_mask: u32,
    pub visible_node_mask: u32,
    pub temp_buffer_heap: vk::DeviceMemory,
    pub buffer_heap_offset: u64,
    pub temp_texture_heap: vk::DeviceMemory,
    pub texture_heap_offset: u64,
    pub pipeline_cache: vk::PipelineCache,
}

impl Default for XessVkInitParams {
    fn default() -> Self {
        Self {
            output_resolution: Xess2d::default(),
            quality_setting: XessQualitySettings::Quality,
            init_flags: init_flags::NONE,
            creation_node_mask: 0,
            visible_node_mask: 0,
            temp_buffer_heap: vk::DeviceMemory::null(),
            buffer_heap_offset: 0,
            temp_texture_heap: vk::DeviceMemory::null(),
            texture_heap_offset: 0,
            pipeline_cache: vk::PipelineCache::null(),
        }
    }
}

/// `xess_vk_execute_params_t` (360 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct XessVkExecuteParams {
    pub color_texture: XessVkImageViewInfo,
    pub velocity_texture: XessVkImageViewInfo,
    pub depth_texture: XessVkImageViewInfo,
    pub exposure_scale_texture: XessVkImageViewInfo,
    pub responsive_pixel_mask_texture: XessVkImageViewInfo,
    pub output_texture: XessVkImageViewInfo,
    pub jitter_offset_x: f32,
    pub jitter_offset_y: f32,
    pub exposure_scale: f32,
    pub reset_history: u32,
    pub input_width: u32,
    pub input_height: u32,
    pub input_color_base: Xess2d,
    pub input_motion_vector_base: Xess2d,
    pub input_depth_base: Xess2d,
    pub input_responsive_mask_base: Xess2d,
    pub reserved0: Xess2d,
    pub output_color_base: Xess2d,
}

impl Default for XessVkExecuteParams {
    fn default() -> Self {
        Self {
            color_texture: XessVkImageViewInfo::default(),
            velocity_texture: XessVkImageViewInfo::default(),
            depth_texture: XessVkImageViewInfo::default(),
            exposure_scale_texture: XessVkImageViewInfo::default(),
            responsive_pixel_mask_texture: XessVkImageViewInfo::default(),
            output_texture: XessVkImageViewInfo::default(),
            jitter_offset_x: 0.0,
            jitter_offset_y: 0.0,
            exposure_scale: 1.0,
            reset_history: 0,
            input_width: 0,
            input_height: 0,
            input_color_base: Xess2d::default(),
            input_motion_vector_base: Xess2d::default(),
            input_depth_base: Xess2d::default(),
            input_responsive_mask_base: Xess2d::default(),
            reserved0: Xess2d::default(),
            output_color_base: Xess2d::default(),
        }
    }
}

// XeSS C-ABI entry points
pub type PfnXessGetVersion = unsafe extern "C" fn(p_version: *mut XessVersion) -> XessResult;

pub type PfnXessVkGetRequiredInstanceExtensions = unsafe extern "C" fn(
    instance_extensions_count: *mut u32,
    instance_extensions: *mut *const *const c_char,
    min_vk_api_version: *mut u32,
) -> XessResult;

pub type PfnXessVkGetRequiredDeviceExtensions = unsafe extern "C" fn(
    instance: vk::Instance,
    physical_device: vk::PhysicalDevice,
    device_extensions_count: *mut u32,
    device_extensions: *mut *const *const c_char,
) -> XessResult;

pub type PfnXessVkGetRequiredDeviceFeatures = unsafe extern "C" fn(
    instance: vk::Instance,
    physical_device: vk::PhysicalDevice,
    features: *mut *mut c_void,
) -> XessResult;

pub type PfnXessVkCreateContext = unsafe extern "C" fn(
    instance: vk::Instance,
    physical_device: vk::PhysicalDevice,
    device: vk::Device,
    ph_context: *mut XessContextPtr,
) -> XessResult;

pub type PfnXessVkBuildPipelines = unsafe extern "C" fn(
    h_context: XessContextPtr,
    pipeline_cache: vk::PipelineCache,
    blocking: bool,
    init_flags: u32,
) -> XessResult;

pub type PfnXessVkInit = unsafe extern "C" fn(
    h_context: XessContextPtr,
    p_init_params: *const XessVkInitParams,
) -> XessResult;

pub type PfnXessVkExecute = unsafe extern "C" fn(
    h_context: XessContextPtr,
    command_buffer: vk::CommandBuffer,
    p_exec_params: *const XessVkExecuteParams,
) -> XessResult;

pub type PfnXessGetInputResolution = unsafe extern "C" fn(
    h_context: XessContextPtr,
    p_output_resolution: *const Xess2d,
    quality_settings: XessQualitySettings,
    p_input_resolution: *mut Xess2d,
) -> XessResult;

pub type PfnXessGetOptimalInputResolution = unsafe extern "C" fn(
    h_context: XessContextPtr,
    p_output_resolution: *const Xess2d,
    quality_settings: XessQualitySettings,
    p_optimal: *mut Xess2d,
    p_min: *mut Xess2d,
    p_max: *mut Xess2d,
) -> XessResult;

pub type PfnXessGetProperties = unsafe extern "C" fn(
    h_context: XessContextPtr,
    p_output_resolution: *const Xess2d,
    p_binding_properties: *mut XessProperties,
) -> XessResult;

pub type PfnXessSetJitterScale =
    unsafe extern "C" fn(h_context: XessContextPtr, x: f32, y: f32) -> XessResult;

pub type PfnXessSetVelocityScale =
    unsafe extern "C" fn(h_context: XessContextPtr, x: f32, y: f32) -> XessResult;

pub type PfnXessDestroyContext = unsafe extern "C" fn(h_context: XessContextPtr) -> XessResult;

/// Dynamically loaded Intel XeSS Vulkan runtime library.
pub struct XessLibrary {
    _library: Library,
    pub loaded_path: PathBuf,
    pub get_version: PfnXessGetVersion,
    pub vk_get_required_instance_extensions: Option<PfnXessVkGetRequiredInstanceExtensions>,
    pub vk_get_required_device_extensions: Option<PfnXessVkGetRequiredDeviceExtensions>,
    pub vk_get_required_device_features: Option<PfnXessVkGetRequiredDeviceFeatures>,
    pub vk_create_context: PfnXessVkCreateContext,
    pub vk_build_pipelines: Option<PfnXessVkBuildPipelines>,
    pub vk_init: PfnXessVkInit,
    pub vk_execute: PfnXessVkExecute,
    pub get_input_resolution: Option<PfnXessGetInputResolution>,
    pub get_optimal_input_resolution: Option<PfnXessGetOptimalInputResolution>,
    pub get_properties: Option<PfnXessGetProperties>,
    pub set_jitter_scale: Option<PfnXessSetJitterScale>,
    pub set_velocity_scale: Option<PfnXessSetVelocityScale>,
    pub destroy_context: PfnXessDestroyContext,
}

impl XessLibrary {
    pub fn candidate_names() -> &'static [&'static str] {
        #[cfg(target_os = "windows")]
        {
            &["libxess.dll", "xess.dll"]
        }
        #[cfg(target_os = "macos")]
        {
            &[]
        }
        #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
        {
            &["libxess.so", "libxess.so.1"]
        }
    }

    pub fn load(search_dirs: &[&Path]) -> Result<Self, XessError> {
        #[cfg(target_os = "macos")]
        {
            let _ = search_dirs;
            return Err(XessError::LibraryLoad(
                "Intel XeSS is not supported on macOS".to_string(),
            ));
        }
        let mut candidates: Vec<PathBuf> = Vec::new();
        for dir in search_dirs {
            for name in Self::candidate_names() {
                let p = dir.join(name);
                if p.exists() {
                    candidates.push(p);
                }
            }
        }
        for name in Self::candidate_names() {
            candidates.push(PathBuf::from(name));
        }

        let mut last_err = String::from("no candidate paths attempted");
        for path in candidates {
            match unsafe { Library::new(&path) } {
                Ok(lib) => return Self::from_library(lib, path),
                Err(e) => last_err = format!("{}: {e}", path.display()),
            }
        }

        Err(XessError::LibraryLoad(last_err))
    }

    fn from_library(lib: Library, loaded_path: PathBuf) -> Result<Self, XessError> {
        unsafe {
            let get_version = *lib
                .get::<PfnXessGetVersion>(b"xessGetVersion\0")
                .map_err(|e| XessError::MissingSymbol("xessGetVersion", e.to_string()))?;
            let vk_get_required_instance_extensions = lib
                .get::<PfnXessVkGetRequiredInstanceExtensions>(
                    b"xessVKGetRequiredInstanceExtensions\0",
                )
                .ok()
                .map(|s| *s);
            let vk_get_required_device_extensions = lib
                .get::<PfnXessVkGetRequiredDeviceExtensions>(b"xessVKGetRequiredDeviceExtensions\0")
                .ok()
                .map(|s| *s);
            let vk_get_required_device_features = lib
                .get::<PfnXessVkGetRequiredDeviceFeatures>(b"xessVKGetRequiredDeviceFeatures\0")
                .ok()
                .map(|s| *s);
            let vk_create_context = *lib
                .get::<PfnXessVkCreateContext>(b"xessVKCreateContext\0")
                .map_err(|e| XessError::MissingSymbol("xessVKCreateContext", e.to_string()))?;
            let vk_build_pipelines = lib
                .get::<PfnXessVkBuildPipelines>(b"xessVKBuildPipelines\0")
                .ok()
                .map(|s| *s);
            let vk_init = *lib
                .get::<PfnXessVkInit>(b"xessVKInit\0")
                .map_err(|e| XessError::MissingSymbol("xessVKInit", e.to_string()))?;
            let vk_execute = *lib
                .get::<PfnXessVkExecute>(b"xessVKExecute\0")
                .map_err(|e| XessError::MissingSymbol("xessVKExecute", e.to_string()))?;
            let get_input_resolution = lib
                .get::<PfnXessGetInputResolution>(b"xessGetInputResolution\0")
                .ok()
                .map(|s| *s);
            let get_optimal_input_resolution = lib
                .get::<PfnXessGetOptimalInputResolution>(b"xessGetOptimalInputResolution\0")
                .ok()
                .map(|s| *s);
            let get_properties = lib
                .get::<PfnXessGetProperties>(b"xessGetProperties\0")
                .ok()
                .map(|s| *s);
            let set_jitter_scale = lib
                .get::<PfnXessSetJitterScale>(b"xessSetJitterScale\0")
                .ok()
                .map(|s| *s);
            let set_velocity_scale = lib
                .get::<PfnXessSetVelocityScale>(b"xessSetVelocityScale\0")
                .ok()
                .map(|s| *s);
            let destroy_context = *lib
                .get::<PfnXessDestroyContext>(b"xessDestroyContext\0")
                .map_err(|e| XessError::MissingSymbol("xessDestroyContext", e.to_string()))?;

            Ok(Self {
                _library: lib,
                loaded_path,
                get_version,
                vk_get_required_instance_extensions,
                vk_get_required_device_extensions,
                vk_get_required_device_features,
                vk_create_context,
                vk_build_pipelines,
                vk_init,
                vk_execute,
                get_input_resolution,
                get_optimal_input_resolution,
                get_properties,
                set_jitter_scale,
                set_velocity_scale,
                destroy_context,
            })
        }
    }

    pub fn version(&self) -> Result<XessVersion, XessError> {
        let mut v = XessVersion::default();
        let res = unsafe { (self.get_version)(&mut v) };
        check_xess("xessGetVersion", res)?;
        Ok(v)
    }

    pub fn required_instance_extensions(&self) -> Result<(Vec<String>, u32), XessError> {
        let Some(func) = self.vk_get_required_instance_extensions else {
            return Ok((Vec::new(), 0));
        };
        let mut count = 0u32;
        let mut ptr_arr: *const *const c_char = ptr::null();
        let mut min_vk = 0u32;
        let res = unsafe { func(&mut count, &mut ptr_arr, &mut min_vk) };
        check_xess("xessVKGetRequiredInstanceExtensions", res)?;
        Ok((extract_cstr_list(ptr_arr, count), min_vk))
    }

    pub fn required_device_extensions(
        &self,
        instance: vk::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Vec<String>, XessError> {
        let Some(func) = self.vk_get_required_device_extensions else {
            return Ok(Vec::new());
        };
        let mut count = 0u32;
        let mut ptr_arr: *const *const c_char = ptr::null();
        let res = unsafe { func(instance, physical_device, &mut count, &mut ptr_arr) };
        check_xess("xessVKGetRequiredDeviceExtensions", res)?;
        Ok(extract_cstr_list(ptr_arr, count))
    }
}

/// Owned Intel XeSS Vulkan context (`xessVKCreateContext` -> `xessVKInit` -> `xessVKExecute`).
pub struct XessContext {
    lib: std::sync::Arc<XessLibrary>,
    handle: XessContextPtr,
}

impl XessContext {
    /// Creates an XeSS Vulkan context bound to `instance`/`physical_device`/`device`.
    ///
    /// # Safety
    /// All passed Vulkan handles must be valid for the lifetime of this context.
    pub unsafe fn create(
        lib: std::sync::Arc<XessLibrary>,
        instance: vk::Instance,
        physical_device: vk::PhysicalDevice,
        device: vk::Device,
    ) -> Result<Self, XessError> {
        let mut handle: XessContextPtr = ptr::null_mut();
        let res =
            unsafe { (lib.vk_create_context)(instance, physical_device, device, &mut handle) };
        check_xess("xessVKCreateContext", res)?;
        if handle.is_null() {
            return Err(XessError::NullContext);
        }
        Ok(Self { lib, handle })
    }

    /// Initializes XeSS Vulkan kernels and internal heaps (`xessVKInit`).
    pub fn init(&mut self, params: &XessVkInitParams) -> Result<(), XessError> {
        let res = unsafe { (self.lib.vk_init)(self.handle, params) };
        check_xess("xessVKInit", res)
    }

    /// Records XeSS upscaling compute commands into `cmd`.
    ///
    /// # Safety
    /// `cmd` must be open and recording, and all image views in `params` must be valid.
    pub unsafe fn execute(
        &mut self,
        cmd: vk::CommandBuffer,
        params: &XessVkExecuteParams,
    ) -> Result<(), XessError> {
        let res = unsafe { (self.lib.vk_execute)(self.handle, cmd, params) };
        check_xess("xessVKExecute", res)
    }

    pub fn get_input_resolution(
        &self,
        output_resolution: Xess2d,
        quality: XessQualitySettings,
    ) -> Result<Xess2d, XessError> {
        let Some(func) = self.lib.get_input_resolution else {
            return Err(XessError::MissingSymbol(
                "xessGetInputResolution",
                "not exported".to_string(),
            ));
        };
        let mut out = Xess2d::default();
        let res = unsafe { func(self.handle, &output_resolution, quality, &mut out) };
        check_xess("xessGetInputResolution", res)?;
        Ok(out)
    }
}

impl Drop for XessContext {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = (self.lib.destroy_context)(self.handle);
            }
            self.handle = ptr::null_mut();
        }
    }
}

fn check_xess(call: &'static str, result: XessResult) -> Result<(), XessError> {
    if result.is_success() {
        Ok(())
    } else {
        Err(XessError::ApiFailure { call, result })
    }
}

fn extract_cstr_list(ptr_arr: *const *const c_char, count: u32) -> Vec<String> {
    if ptr_arr.is_null() || count == 0 {
        return Vec::new();
    }
    let slice = unsafe { std::slice::from_raw_parts(ptr_arr, count as usize) };
    slice
        .iter()
        .filter(|&&p| !p.is_null())
        .filter_map(|&p| {
            unsafe { CStr::from_ptr(p) }
                .to_str()
                .ok()
                .map(str::to_owned)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn verify_xess_c_abi_layouts() {
        assert_eq!(size_of::<XessVersion>(), 8);
        assert_eq!(align_of::<XessVersion>(), 2);
        assert_eq!(size_of::<XessProperties>(), 24);
        assert_eq!(align_of::<XessProperties>(), 8);
        assert_eq!(size_of::<XessVkImageViewInfo>(), 48);
        assert_eq!(align_of::<XessVkImageViewInfo>(), 8);
        assert_eq!(size_of::<XessVkInitParams>(), 64);
        assert_eq!(align_of::<XessVkInitParams>(), 8);
        assert_eq!(size_of::<XessVkExecuteParams>(), 360);
        assert_eq!(align_of::<XessVkExecuteParams>(), 8);
    }

    #[test]
    fn xess_missing_library_reports_honest_error() {
        let missing = Path::new("/nonexistent/vulkanturbo/natives");
        let err = XessLibrary::load(&[missing])
            .err()
            .expect("must fail cleanly");
        assert!(err.to_string().contains("XeSS"));
    }
}
