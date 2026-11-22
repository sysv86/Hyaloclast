// SPDX-License-Identifier: GPL-3.0-only
// Additional permissions under GPLv3 Section 7 apply to optional proprietary vendor SDK
// runtime loading; see LICENSE-EXCEPTION.md in the repository root.

//! Clean-room AMD FidelityFX FSR 3 runtime dynamic loader
//! (`libffx_fsr3_vk_x64.so` / `libamd_fidelityfx_vk.so` / `ffx_fsr3_x64.dll` / `amd_fidelityfx_upscaler_vk.dll`).
//!
//! Supports both the modern FidelityFX API v2 entry points (`ffxCreateContext`, `ffxDestroyContext`,
//! `ffxConfigure`, `ffxQuery`, `ffxDispatch`) and legacy FSR3 C-ABI entry points
//! (`ffxFsr3ContextCreate`, `ffxFsr3ContextDispatch`, `ffxFsr3ContextDestroy`).

use ash::vk;
use libloading::Library;
use std::{
    ffi::{CStr, c_void},
    fmt,
    path::{Path, PathBuf},
    ptr,
};
use thiserror::Error;

pub type FfxContext = *mut c_void;
pub type FfxReturnCode = u32;
pub type FfxStructType = u64;

pub const FFX_API_RETURN_OK: FfxReturnCode = 0;
pub const FFX_API_RETURN_ERROR: FfxReturnCode = 1;
pub const FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE: FfxReturnCode = 2;
pub const FFX_API_RETURN_ERROR_RUNTIME_ERROR: FfxReturnCode = 3;
pub const FFX_API_RETURN_NO_PROVIDER: FfxReturnCode = 4;
pub const FFX_API_RETURN_ERROR_MEMORY: FfxReturnCode = 5;
pub const FFX_API_RETURN_ERROR_PARAMETER: FfxReturnCode = 6;
pub const FFX_API_RETURN_PROVIDER_NO_SUPPORT_NEW_DESCTYPE: FfxReturnCode = 7;

pub const FFX_API_EFFECT_ID_UPSCALE: u64 = 0x0001_0000;
pub const FFX_API_EFFECT_ID_FRAMEGENERATION: u64 = 0x0002_0000;
pub const FFX_API_BACKEND_ID_VK: u64 = 0x0100_0000;

pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK: FfxStructType = FFX_API_BACKEND_ID_VK | 0x03;
pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE: FfxStructType = FFX_API_EFFECT_ID_UPSCALE;
pub const FFX_API_DISPATCH_DESC_TYPE_UPSCALE: FfxStructType = FFX_API_EFFECT_ID_UPSCALE | 0x01;
pub const FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE: FfxStructType =
    FFX_API_EFFECT_ID_UPSCALE | 0x03;
pub const FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT: FfxStructType =
    FFX_API_EFFECT_ID_UPSCALE | 0x04;
pub const FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET: FfxStructType =
    FFX_API_EFFECT_ID_UPSCALE | 0x05;

pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION: FfxStructType =
    FFX_API_EFFECT_ID_FRAMEGENERATION | 0x01;
pub const FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION: FfxStructType =
    FFX_API_EFFECT_ID_FRAMEGENERATION | 0x02;
pub const FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION: FfxStructType =
    FFX_API_EFFECT_ID_FRAMEGENERATION | 0x03;
pub const FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2: FfxStructType =
    FFX_API_EFFECT_ID_FRAMEGENERATION | 0x0C;

/// Legacy `FFX_SDK_DEFAULT_CONTEXT_SIZE` (`1024 * 128` 32-bit words).
pub const FFX_SDK_DEFAULT_CONTEXT_SIZE: usize = 1024 * 128;

#[derive(Debug, Error)]
pub enum Fsr3Error {
    #[error("failed to load FidelityFX FSR3 library ({0})")]
    LibraryLoad(String),
    #[error("missing required FidelityFX symbol `{0}`: {1}")]
    MissingSymbol(&'static str, String),
    #[error("FidelityFX call `{call}` failed with code {code}")]
    ApiFailure { call: &'static str, code: u32 },
    #[error("null FidelityFX context handle")]
    NullContext,
}

/// Which ABI family the loaded FidelityFX shared library exports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fsr3AbiKind {
    FfxApiV2,
    LegacyFsr3,
}

impl fmt::Display for Fsr3AbiKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FfxApiV2 => write!(f, "FidelityFX-API-v2"),
            Self::LegacyFsr3 => write!(f, "Legacy-FSR3"),
        }
    }
}

/// `ffxApiHeader` (16 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FfxApiHeader {
    pub desc_type: FfxStructType,
    pub p_next: *mut FfxApiHeader,
}

impl Default for FfxApiHeader {
    fn default() -> Self {
        Self {
            desc_type: 0,
            p_next: ptr::null_mut(),
        }
    }
}

/// `FfxApiUpscaleQualityMode`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfxApiUpscaleQualityMode {
    NativeAa = 0,
    Quality = 1,
    Balanced = 2,
    Performance = 3,
    UltraPerformance = 4,
}

/// `FfxApiCreateContextUpscaleFlags`.
pub mod upscale_flags {
    pub const HIGH_DYNAMIC_RANGE: u32 = 1 << 0;
    pub const DISPLAY_RESOLUTION_MOTION_VECTORS: u32 = 1 << 1;
    pub const MOTION_VECTORS_JITTER_CANCELLATION: u32 = 1 << 2;
    pub const DEPTH_INVERTED: u32 = 1 << 3;
    pub const DEPTH_INFINITE: u32 = 1 << 4;
    pub const AUTO_EXPOSURE: u32 = 1 << 5;
    pub const DYNAMIC_RESOLUTION: u32 = 1 << 6;
    pub const NON_LINEAR_COLORSPACE: u32 = 1 << 8;
}

/// `FfxApiSurfaceFormat` values matching `ffx_api_types.h`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfxApiSurfaceFormat {
    Unknown = 0,
    R32G32B32A32Typeless = 1,
    R32G32B32A32Uint = 2,
    R32G32B32A32Float = 3,
    R16G16B16A16Float = 4,
    R32G32B32Float = 5,
    R32G32Float = 6,
    R8Uint = 7,
    R32Uint = 8,
    R8G8B8A8Typeless = 9,
    R8G8B8A8Unorm = 10,
    R8G8B8A8Snorm = 11,
    R8G8B8A8Srgb = 12,
    B8G8R8A8Typeless = 13,
    B8G8R8A8Unorm = 14,
    B8G8R8A8Srgb = 15,
    R11G11B10Float = 16,
    R10G10B10A2Unorm = 17,
    R16G16Float = 18,
    R16G16Uint = 19,
    R16G16Sint = 20,
    R16Float = 21,
    R16Uint = 22,
    R16Unorm = 23,
    R16Snorm = 24,
    R8Unorm = 25,
    R8G8Unorm = 26,
    R8G8Uint = 27,
    R32Float = 28,
    R9G9B9E5SharedExp = 29,
}

/// Converts a Vulkan `vk::Format` to `FfxApiSurfaceFormat`.
pub const fn vk_format_to_ffx_surface_format(fmt: vk::Format) -> FfxApiSurfaceFormat {
    match fmt {
        vk::Format::R32G32B32A32_SFLOAT => FfxApiSurfaceFormat::R32G32B32A32Float,
        vk::Format::R32G32B32_SFLOAT => FfxApiSurfaceFormat::R32G32B32Float,
        vk::Format::R32G32B32A32_UINT => FfxApiSurfaceFormat::R32G32B32A32Uint,
        vk::Format::R16G16B16A16_SFLOAT => FfxApiSurfaceFormat::R16G16B16A16Float,
        vk::Format::R32G32_SFLOAT => FfxApiSurfaceFormat::R32G32Float,
        vk::Format::R32_UINT => FfxApiSurfaceFormat::R32Uint,
        vk::Format::R8G8B8A8_UNORM => FfxApiSurfaceFormat::R8G8B8A8Unorm,
        vk::Format::R8G8B8A8_SNORM => FfxApiSurfaceFormat::R8G8B8A8Snorm,
        vk::Format::R8G8B8A8_SRGB => FfxApiSurfaceFormat::R8G8B8A8Srgb,
        vk::Format::B8G8R8A8_UNORM => FfxApiSurfaceFormat::B8G8R8A8Unorm,
        vk::Format::B8G8R8A8_SRGB => FfxApiSurfaceFormat::B8G8R8A8Srgb,
        vk::Format::B10G11R11_UFLOAT_PACK32 => FfxApiSurfaceFormat::R11G11B10Float,
        vk::Format::A2B10G10R10_UNORM_PACK32 => FfxApiSurfaceFormat::R10G10B10A2Unorm,
        vk::Format::R16G16_SFLOAT => FfxApiSurfaceFormat::R16G16Float,
        vk::Format::R16_SFLOAT => FfxApiSurfaceFormat::R16Float,
        vk::Format::R16_UNORM | vk::Format::D16_UNORM => FfxApiSurfaceFormat::R16Unorm,
        vk::Format::R8_UNORM => FfxApiSurfaceFormat::R8Unorm,
        vk::Format::R8G8_UNORM => FfxApiSurfaceFormat::R8G8Unorm,
        vk::Format::R32_SFLOAT | vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT => {
            FfxApiSurfaceFormat::R32Float
        }
        vk::Format::E5B9G9R9_UFLOAT_PACK32 => FfxApiSurfaceFormat::R9G9B9E5SharedExp,
        _ => FfxApiSurfaceFormat::Unknown,
    }
}

/// `FfxApiResourceState` bitflags.
pub mod resource_state {
    pub const COMMON: u32 = 1 << 0;
    pub const UNORDERED_ACCESS: u32 = 1 << 1;
    pub const COMPUTE_READ: u32 = 1 << 2;
    pub const PIXEL_READ: u32 = 1 << 3;
    pub const PIXEL_COMPUTE_READ: u32 = PIXEL_READ | COMPUTE_READ;
    pub const COPY_SRC: u32 = 1 << 4;
    pub const COPY_DEST: u32 = 1 << 5;
}

/// `FfxApiResourceUsage` bitflags.
pub mod resource_usage {
    pub const READ_ONLY: u32 = 0;
    pub const RENDERTARGET: u32 = 1 << 0;
    pub const UAV: u32 = 1 << 1;
    pub const DEPTHTARGET: u32 = 1 << 2;
}

/// `FfxApiResourceType`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfxApiResourceType {
    Buffer = 0,
    Texture1D = 1,
    Texture2D = 2,
    TextureCube = 3,
    Texture3D = 4,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FfxApiDimensions2D {
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FfxApiFloatCoords2D {
    pub x: f32,
    pub y: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FfxApiRect2D {
    pub left: i32,
    pub top: i32,
    pub width: i32,
    pub height: i32,
}

/// `FfxApiResourceDescription` (32 bytes, align 4).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FfxApiResourceDescription {
    pub res_type: u32,
    pub format: u32,
    pub width_or_size: u32,
    pub height_or_stride: u32,
    pub depth_or_alignment: u32,
    pub mip_count: u32,
    pub flags: u32,
    pub usage: u32,
}

/// `FfxApiResource` (48 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FfxApiResource {
    pub resource: *mut c_void,
    pub description: FfxApiResourceDescription,
    pub state: u32,
}

impl Default for FfxApiResource {
    fn default() -> Self {
        Self {
            resource: ptr::null_mut(),
            description: FfxApiResourceDescription::default(),
            state: 0,
        }
    }
}

impl FfxApiResource {
    /// Builds an `FfxApiResource` wrapping a Vulkan `VkImage` handle (`vk::Image::as_raw() as *mut c_void`).
    pub fn from_vk_image(
        image: vk::Image,
        format: vk::Format,
        width: u32,
        height: u32,
        mip_count: u32,
        usage: u32,
        state: u32,
    ) -> Self {
        use ash::vk::Handle;
        Self {
            resource: image.as_raw() as usize as *mut c_void,
            description: FfxApiResourceDescription {
                res_type: FfxApiResourceType::Texture2D as u32,
                format: vk_format_to_ffx_surface_format(format) as u32,
                width_or_size: width,
                height_or_stride: height,
                depth_or_alignment: 1,
                mip_count,
                flags: 0,
                usage,
            },
            state,
        }
    }
}

/// `VkDeviceContext` passed inside `ffxCreateBackendVKDesc`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FfxVkDeviceContext {
    pub vk_device: vk::Device,
    pub vk_physical_device: vk::PhysicalDevice,
    pub vk_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
}

/// `ffxCreateBackendVKDesc` (40 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FfxCreateBackendVKDesc {
    pub header: FfxApiHeader,
    pub device_context: FfxVkDeviceContext,
}

/// `ffxCreateContextDescUpscale` (48 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FfxCreateContextDescUpscale {
    pub header: FfxApiHeader,
    pub flags: u32,
    pub max_render_size: FfxApiDimensions2D,
    pub max_upscale_size: FfxApiDimensions2D,
    pub fp_message: Option<unsafe extern "C" fn(u32, *const u16)>,
}

/// `ffxDispatchDescUpscale` (432 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FfxDispatchDescUpscale {
    pub header: FfxApiHeader,
    pub command_list: vk::CommandBuffer,
    pub color: FfxApiResource,
    pub depth: FfxApiResource,
    pub motion_vectors: FfxApiResource,
    pub exposure: FfxApiResource,
    pub reactive: FfxApiResource,
    pub transparency_and_composition: FfxApiResource,
    pub output: FfxApiResource,
    pub jitter_offset: FfxApiFloatCoords2D,
    pub motion_vector_scale: FfxApiFloatCoords2D,
    pub render_size: FfxApiDimensions2D,
    pub upscale_size: FfxApiDimensions2D,
    pub enable_sharpening: bool,
    pub sharpness: f32,
    pub frame_time_delta: f32,
    pub pre_exposure: f32,
    pub reset: bool,
    pub camera_near: f32,
    pub camera_far: f32,
    pub camera_fov_angle_vertical: f32,
    pub view_space_to_meters_factor: f32,
    pub flags: u32,
}

/// `ffxCreateContextDescFrameGeneration` (40 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FfxCreateContextDescFrameGeneration {
    pub header: FfxApiHeader,
    pub flags: u32,
    pub display_size: FfxApiDimensions2D,
    pub max_render_size: FfxApiDimensions2D,
    pub back_buffer_format: u32,
}

/// `ffxDispatchDescFrameGeneration` (312 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FfxDispatchDescFrameGeneration {
    pub header: FfxApiHeader,
    pub command_list: vk::CommandBuffer,
    pub present_color: FfxApiResource,
    pub outputs: [FfxApiResource; 4],
    pub num_generated_frames: u32,
    pub reset: bool,
    pub backbuffer_transfer_function: u32,
    pub min_max_luminance: [f32; 2],
    pub generation_rect: FfxApiRect2D,
    pub frame_id: u64,
}

/// `ffxDispatchDescFrameGenerationPrepareV2` (232 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FfxDispatchDescFrameGenerationPrepareV2 {
    pub header: FfxApiHeader,
    pub frame_id: u64,
    pub flags: u32,
    pub command_list: vk::CommandBuffer,
    pub render_size: FfxApiDimensions2D,
    pub jitter_offset: FfxApiFloatCoords2D,
    pub motion_vector_scale: FfxApiFloatCoords2D,
    pub frame_time_delta: f32,
    pub reset: bool,
    pub camera_near: f32,
    pub camera_far: f32,
    pub camera_fov_angle_vertical: f32,
    pub view_space_to_meters_factor: f32,
    pub depth: FfxApiResource,
    pub motion_vectors: FfxApiResource,
    pub camera_position: [f32; 3],
    pub camera_up: [f32; 3],
    pub camera_right: [f32; 3],
    pub camera_forward: [f32; 3],
}

/// `ffxQueryDescUpscaleGetRenderResolutionFromQualityMode`.
#[repr(C)]
pub struct FfxQueryDescUpscaleGetRenderResolutionFromQualityMode {
    pub header: FfxApiHeader,
    pub display_width: u32,
    pub display_height: u32,
    pub quality_mode: u32,
    pub p_out_render_width: *mut u32,
    pub p_out_render_height: *mut u32,
}

/// `ffxQueryDescUpscaleGetJitterPhaseCount`.
#[repr(C)]
pub struct FfxQueryDescUpscaleGetJitterPhaseCount {
    pub header: FfxApiHeader,
    pub render_width: u32,
    pub display_width: u32,
    pub p_out_phase_count: *mut i32,
}

/// `ffxQueryDescUpscaleGetJitterOffset`.
#[repr(C)]
pub struct FfxQueryDescUpscaleGetJitterOffset {
    pub header: FfxApiHeader,
    pub index: i32,
    pub phase_count: i32,
    pub p_out_x: *mut f32,
    pub p_out_y: *mut f32,
}

/// `ffxAllocationCallbacks`.
#[repr(C)]
pub struct FfxAllocationCallbacks {
    pub p_user_data: *mut c_void,
    pub alloc: Option<unsafe extern "C" fn(*mut c_void, u64) -> *mut c_void>,
    pub dealloc: Option<unsafe extern "C" fn(*mut c_void, *mut c_void)>,
}

/// Legacy FSR3 opaque context storage (`FfxFsr3Context` / `FfxFsr3UpscalerContext`).
#[repr(C)]
pub struct FfxFsr3ContextStorage {
    pub data: [u32; FFX_SDK_DEFAULT_CONTEXT_SIZE],
}

/// Legacy FSR3 dispatch descriptor (`FfxFsr3UpscalerDispatchDescription`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FfxFsr3UpscalerDispatchDescription {
    pub command_list: vk::CommandBuffer,
    pub color: FfxApiResource,
    pub depth: FfxApiResource,
    pub motion_vectors: FfxApiResource,
    pub exposure: FfxApiResource,
    pub reactive: FfxApiResource,
    pub transparency_and_composition: FfxApiResource,
    pub dilated_depth: FfxApiResource,
    pub dilated_motion_vectors: FfxApiResource,
    pub reconstructed_prev_nearest_depth: FfxApiResource,
    pub output: FfxApiResource,
    pub jitter_offset: FfxApiFloatCoords2D,
    pub motion_vector_scale: FfxApiFloatCoords2D,
    pub render_size: FfxApiDimensions2D,
    pub upscale_size: FfxApiDimensions2D,
    pub enable_sharpening: bool,
    pub sharpness: f32,
    pub frame_time_delta: f32,
    pub pre_exposure: f32,
    pub reset: bool,
    pub camera_near: f32,
    pub camera_far: f32,
    pub camera_fov_angle_vertical: f32,
    pub view_space_to_meters_factor: f32,
    pub flags: u32,
}

// FidelityFX-API v2 C-ABI entry points
pub type PfnFfxCreateContext = unsafe extern "C" fn(
    context: *mut FfxContext,
    desc: *mut FfxApiHeader,
    mem_cb: *const FfxAllocationCallbacks,
) -> FfxReturnCode;

pub type PfnFfxDestroyContext = unsafe extern "C" fn(
    context: *mut FfxContext,
    mem_cb: *const FfxAllocationCallbacks,
) -> FfxReturnCode;

pub type PfnFfxConfigure =
    unsafe extern "C" fn(context: *mut FfxContext, desc: *const FfxApiHeader) -> FfxReturnCode;

pub type PfnFfxQuery =
    unsafe extern "C" fn(context: *mut FfxContext, desc: *mut FfxApiHeader) -> FfxReturnCode;

pub type PfnFfxDispatch =
    unsafe extern "C" fn(context: *mut FfxContext, desc: *const FfxApiHeader) -> FfxReturnCode;

// Legacy FSR3 C-ABI entry points
pub type PfnFfxFsr3ContextCreate =
    unsafe extern "C" fn(context: *mut c_void, desc: *const c_void) -> i32;
pub type PfnFfxFsr3ContextDispatch =
    unsafe extern "C" fn(context: *mut c_void, desc: *const c_void) -> i32;
pub type PfnFfxFsr3ContextDestroy = unsafe extern "C" fn(context: *mut c_void) -> i32;

/// Dynamically loaded AMD FidelityFX FSR 3 runtime library.
pub struct Fsr3Library {
    _library: Library,
    pub loaded_path: PathBuf,
    pub loaded_path_cstr: std::ffi::CString,
    pub abi_kind: Fsr3AbiKind,
    pub ffx_create_context: Option<PfnFfxCreateContext>,
    pub ffx_destroy_context: Option<PfnFfxDestroyContext>,
    pub ffx_configure: Option<PfnFfxConfigure>,
    pub ffx_query: Option<PfnFfxQuery>,
    pub ffx_dispatch: Option<PfnFfxDispatch>,
    pub legacy_create: Option<PfnFfxFsr3ContextCreate>,
    pub legacy_dispatch: Option<PfnFfxFsr3ContextDispatch>,
    pub legacy_destroy: Option<PfnFfxFsr3ContextDestroy>,
}

impl Fsr3Library {
    pub fn candidate_names() -> &'static [&'static str] {
        #[cfg(target_os = "windows")]
        {
            &[
                "amd_fidelityfx_vk.dll",
                "amd_fidelityfx_upscaler_vk.dll",
                "amd_fidelityfx_loader_vk.dll",
                "ffx_fsr3_x64.dll",
            ]
        }
        #[cfg(target_os = "macos")]
        {
            &[
                "libffx_fsr3_vk.dylib",
                "libamd_fidelityfx_vk.dylib",
                "libamd_fidelityfx_upscaler_vk.dylib",
            ]
        }
        #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
        {
            &[
                "libffx_fsr3_vk_x64.so",
                "libamd_fidelityfx_vk.so",
                "libamd_fidelityfx_upscaler_vk.so",
                "libffx_fsr3upscaler_vk.so",
            ]
        }
    }

    pub fn load(search_dirs: &[&Path]) -> Result<Self, Fsr3Error> {
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

        Err(Fsr3Error::LibraryLoad(last_err))
    }

    fn from_library(lib: Library, loaded_path: PathBuf) -> Result<Self, Fsr3Error> {
        unsafe {
            let ffx_create_context = lib
                .get::<PfnFfxCreateContext>(b"ffxCreateContext\0")
                .ok()
                .map(|s| *s);
            let ffx_destroy_context = lib
                .get::<PfnFfxDestroyContext>(b"ffxDestroyContext\0")
                .ok()
                .map(|s| *s);
            let ffx_configure = lib
                .get::<PfnFfxConfigure>(b"ffxConfigure\0")
                .ok()
                .map(|s| *s);
            let ffx_query = lib.get::<PfnFfxQuery>(b"ffxQuery\0").ok().map(|s| *s);
            let ffx_dispatch = lib.get::<PfnFfxDispatch>(b"ffxDispatch\0").ok().map(|s| *s);

            let legacy_create = lib
                .get::<PfnFfxFsr3ContextCreate>(b"ffxFsr3ContextCreate\0")
                .ok()
                .map(|s| *s)
                .or_else(|| {
                    lib.get::<PfnFfxFsr3ContextCreate>(b"ffxFsr3UpscalerContextCreate\0")
                        .ok()
                        .map(|s| *s)
                });
            let legacy_dispatch = lib
                .get::<PfnFfxFsr3ContextDispatch>(b"ffxFsr3ContextDispatch\0")
                .ok()
                .map(|s| *s)
                .or_else(|| {
                    lib.get::<PfnFfxFsr3ContextDispatch>(b"ffxFsr3UpscalerContextDispatch\0")
                        .ok()
                        .map(|s| *s)
                });
            let legacy_destroy = lib
                .get::<PfnFfxFsr3ContextDestroy>(b"ffxFsr3ContextDestroy\0")
                .ok()
                .map(|s| *s)
                .or_else(|| {
                    lib.get::<PfnFfxFsr3ContextDestroy>(b"ffxFsr3UpscalerContextDestroy\0")
                        .ok()
                        .map(|s| *s)
                });

            let abi_kind = if ffx_create_context.is_some()
                && ffx_dispatch.is_some()
                && ffx_destroy_context.is_some()
            {
                Fsr3AbiKind::FfxApiV2
            } else if legacy_create.is_some()
                && legacy_dispatch.is_some()
                && legacy_destroy.is_some()
            {
                Fsr3AbiKind::LegacyFsr3
            } else {
                return Err(Fsr3Error::MissingSymbol(
                    "ffxCreateContext / ffxFsr3ContextCreate",
                    format!(
                        "library {} exports neither FidelityFX-API v2 nor legacy FSR3 entry points",
                        loaded_path.display()
                    ),
                ));
            };

            Ok(Self {
                _library: lib,
                loaded_path_cstr: std::ffi::CString::new(loaded_path.to_string_lossy().as_bytes())
                    .map_err(|e| Fsr3Error::LibraryLoad(e.to_string()))?,
                loaded_path,
                abi_kind,
                ffx_create_context,
                ffx_destroy_context,
                ffx_configure,
                ffx_query,
                ffx_dispatch,
                legacy_create,
                legacy_dispatch,
                legacy_destroy,
            })
        }
    }

    /// Queries recommended render resolution for `display_width x display_height` via `ffxQuery`.
    pub fn query_render_resolution(
        &self,
        display_width: u32,
        display_height: u32,
        quality: FfxApiUpscaleQualityMode,
    ) -> Result<(u32, u32), Fsr3Error> {
        let Some(query) = self.ffx_query else {
            return Err(Fsr3Error::MissingSymbol(
                "ffxQuery",
                "not exported".to_string(),
            ));
        };
        let mut out_w = 0u32;
        let mut out_h = 0u32;
        let mut desc = FfxQueryDescUpscaleGetRenderResolutionFromQualityMode {
            header: FfxApiHeader {
                desc_type: FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE,
                p_next: ptr::null_mut(),
            },
            display_width,
            display_height,
            quality_mode: quality as u32,
            p_out_render_width: &mut out_w,
            p_out_render_height: &mut out_h,
        };
        let code = unsafe { query(ptr::null_mut(), &mut desc.header) };
        if code != FFX_API_RETURN_OK {
            return Err(Fsr3Error::ApiFailure {
                call: "ffxQuery(GetRenderResolution)",
                code,
            });
        }
        Ok((out_w, out_h))
    }
}

/// Owned FidelityFX API v2 Upscale context (`ffxCreateContext` / `ffxDispatch` / `ffxDestroyContext`).
pub struct Fsr3UpscaleContext {
    lib: std::sync::Arc<Fsr3Library>,
    ctx: FfxContext,
    legacy_storage: Option<Box<FfxFsr3ContextStorage>>,
}

impl Fsr3UpscaleContext {
    /// Creates an FSR 3.1+ Vulkan upscaling context via `ffxCreateContext`.
    ///
    /// # Safety
    /// Vulkan handles (`device`, `physical_device`, `gdpa`) must remain valid until drop.
    pub unsafe fn create(
        lib: std::sync::Arc<Fsr3Library>,
        device: vk::Device,
        physical_device: vk::PhysicalDevice,
        gdpa: vk::PFN_vkGetDeviceProcAddr,
        max_render_size: FfxApiDimensions2D,
        max_upscale_size: FfxApiDimensions2D,
        flags: u32,
    ) -> Result<Self, Fsr3Error> {
        let create_fn = lib.ffx_create_context.ok_or_else(|| {
            Fsr3Error::MissingSymbol("ffxCreateContext", "FFX-API v2 unavailable".to_string())
        })?;

        let mut backend_desc = FfxCreateBackendVKDesc {
            header: FfxApiHeader {
                desc_type: FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK,
                p_next: ptr::null_mut(),
            },
            device_context: FfxVkDeviceContext {
                vk_device: device,
                vk_physical_device: physical_device,
                vk_device_proc_addr: gdpa,
            },
        };

        let mut upscale_desc = FfxCreateContextDescUpscale {
            header: FfxApiHeader {
                desc_type: FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE,
                p_next: &mut backend_desc.header,
            },
            flags,
            max_render_size,
            max_upscale_size,
            fp_message: None,
        };

        let mut ctx: FfxContext = ptr::null_mut();
        let code = unsafe { create_fn(&mut ctx, &mut upscale_desc.header, ptr::null()) };
        if code != FFX_API_RETURN_OK {
            return Err(Fsr3Error::ApiFailure {
                call: "ffxCreateContext(Upscale)",
                code,
            });
        }
        if ctx.is_null() {
            return Err(Fsr3Error::NullContext);
        }
        Ok(Self {
            lib,
            ctx,
            legacy_storage: None,
        })
    }

    /// Dispatches FSR3 upscaling into `desc.command_list`, falling back to legacy
    /// `ffxFsr3ContextDispatch` when the loaded library uses the legacy FSR3 ABI.
    ///
    /// # Safety
    /// `desc` handles and `VkCommandBuffer` must be valid and recording.
    pub unsafe fn dispatch(&mut self, desc: &FfxDispatchDescUpscale) -> Result<(), Fsr3Error> {
        if desc.command_list == vk::CommandBuffer::null()
            || desc.render_size.width == 0
            || desc.render_size.height == 0
            || desc.upscale_size.width == 0
            || desc.upscale_size.height == 0
        {
            return Err(Fsr3Error::ApiFailure {
                call: "ffxDispatch(Upscale:invalid_params)",
                code: FFX_API_RETURN_ERROR_PARAMETER,
            });
        }
        if let Some(dispatch_fn) = self.lib.ffx_dispatch
            && !self.ctx.is_null()
        {
            let code = unsafe { dispatch_fn(&mut self.ctx, &desc.header) };
            if code != FFX_API_RETURN_OK {
                return Err(Fsr3Error::ApiFailure {
                    call: "ffxDispatch(Upscale)",
                    code,
                });
            }
            return Ok(());
        }
        if let (Some(legacy_fn), Some(storage)) =
            (self.lib.legacy_dispatch, self.legacy_storage.as_mut())
        {
            let legacy_desc = FfxFsr3UpscalerDispatchDescription {
                command_list: desc.command_list,
                color: desc.color,
                depth: desc.depth,
                motion_vectors: desc.motion_vectors,
                exposure: desc.exposure,
                reactive: desc.reactive,
                transparency_and_composition: desc.transparency_and_composition,
                dilated_depth: FfxApiResource::default(),
                dilated_motion_vectors: FfxApiResource::default(),
                reconstructed_prev_nearest_depth: FfxApiResource::default(),
                output: desc.output,
                jitter_offset: desc.jitter_offset,
                motion_vector_scale: desc.motion_vector_scale,
                render_size: desc.render_size,
                upscale_size: desc.upscale_size,
                enable_sharpening: desc.enable_sharpening,
                sharpness: desc.sharpness,
                frame_time_delta: desc.frame_time_delta,
                pre_exposure: desc.pre_exposure,
                reset: desc.reset,
                camera_near: desc.camera_near,
                camera_far: desc.camera_far,
                camera_fov_angle_vertical: desc.camera_fov_angle_vertical,
                view_space_to_meters_factor: desc.view_space_to_meters_factor,
                flags: desc.flags,
            };
            let code = unsafe {
                legacy_fn(
                    storage.as_mut() as *mut FfxFsr3ContextStorage as *mut c_void,
                    &legacy_desc as *const FfxFsr3UpscalerDispatchDescription as *const c_void,
                )
            };
            if code != 0 {
                return Err(Fsr3Error::ApiFailure {
                    call: "ffxFsr3ContextDispatch(Upscale)",
                    code: code as u32,
                });
            }
            return Ok(());
        }
        Err(Fsr3Error::MissingSymbol(
            "ffxDispatch / ffxFsr3ContextDispatch",
            "no FSR3 dispatch entry point available".to_string(),
        ))
    }

    /// Returns a human-readable library path label.
    pub fn library_path(&self) -> &CStr {
        self.lib.loaded_path_cstr.as_c_str()
    }
}

impl Drop for Fsr3UpscaleContext {
    fn drop(&mut self) {
        if !self.ctx.is_null()
            && let Some(destroy_fn) = self.lib.ffx_destroy_context
        {
            unsafe {
                let _ = destroy_fn(&mut self.ctx, ptr::null());
            }
            self.ctx = ptr::null_mut();
        }
        if let Some(mut storage) = self.legacy_storage.take()
            && let Some(destroy_fn) = self.lib.legacy_destroy
        {
            unsafe {
                let _ = destroy_fn(storage.as_mut() as *mut FfxFsr3ContextStorage as *mut c_void);
            }
        }
    }
}

/// Owned FidelityFX API v2 Frame Generation context (`ffxCreateContext` / `ffxDispatch` / `ffxDestroyContext`).
pub struct Fsr3FrameGenContext {
    lib: std::sync::Arc<Fsr3Library>,
    ctx: FfxContext,
}

impl Fsr3FrameGenContext {
    /// Creates an FSR 3.1+ Vulkan Frame Generation context via `ffxCreateContext`.
    ///
    /// # Safety
    /// Vulkan handles (`device`, `physical_device`, `gdpa`) must remain valid until drop.
    pub unsafe fn create(
        lib: std::sync::Arc<Fsr3Library>,
        device: vk::Device,
        physical_device: vk::PhysicalDevice,
        gdpa: vk::PFN_vkGetDeviceProcAddr,
        display_size: FfxApiDimensions2D,
        backbuffer_format: u32,
        flags: u32,
    ) -> Result<Self, Fsr3Error> {
        let create_fn = lib.ffx_create_context.ok_or_else(|| {
            Fsr3Error::MissingSymbol(
                "ffxCreateContext",
                "FFX-API v2 FrameGen unavailable".to_string(),
            )
        })?;
        let mut backend_desc = FfxCreateBackendVKDesc {
            header: FfxApiHeader {
                desc_type: FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK,
                p_next: ptr::null_mut(),
            },
            device_context: FfxVkDeviceContext {
                vk_device: device,
                vk_physical_device: physical_device,
                vk_device_proc_addr: gdpa,
            },
        };
        let mut fg_desc = FfxCreateContextDescFrameGeneration {
            header: FfxApiHeader {
                desc_type: FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION,
                p_next: &mut backend_desc.header,
            },
            flags,
            display_size,
            max_render_size: display_size,
            back_buffer_format: backbuffer_format,
        };
        let mut ctx: FfxContext = ptr::null_mut();
        let code = unsafe { create_fn(&mut ctx, &mut fg_desc.header, ptr::null()) };
        if code != FFX_API_RETURN_OK {
            return Err(Fsr3Error::ApiFailure {
                call: "ffxCreateContext(FrameGeneration)",
                code,
            });
        }
        if ctx.is_null() {
            return Err(Fsr3Error::NullContext);
        }
        Ok(Self { lib, ctx })
    }

    /// Dispatches FSR3 Frame Generation into `desc.command_list`.
    ///
    /// # Safety
    /// `desc` handles and `VkCommandBuffer` must be valid and recording.
    pub unsafe fn dispatch(
        &mut self,
        desc: &FfxDispatchDescFrameGeneration,
    ) -> Result<(), Fsr3Error> {
        let dispatch_fn = self.lib.ffx_dispatch.ok_or_else(|| {
            Fsr3Error::MissingSymbol("ffxDispatch", "FFX-API v2 FrameGen unavailable".to_string())
        })?;
        let code = unsafe { dispatch_fn(&mut self.ctx, &desc.header) };
        if code != FFX_API_RETURN_OK {
            return Err(Fsr3Error::ApiFailure {
                call: "ffxDispatch(FrameGeneration)",
                code,
            });
        }
        Ok(())
    }
}

impl Drop for Fsr3FrameGenContext {
    fn drop(&mut self) {
        if !self.ctx.is_null()
            && let Some(destroy_fn) = self.lib.ffx_destroy_context
        {
            unsafe {
                let _ = destroy_fn(&mut self.ctx, ptr::null());
            }
            self.ctx = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn verify_ffx_c_abi_layouts() {
        assert_eq!(size_of::<FfxApiHeader>(), 16);
        assert_eq!(align_of::<FfxApiHeader>(), 8);
        assert_eq!(size_of::<FfxApiResourceDescription>(), 32);
        assert_eq!(align_of::<FfxApiResourceDescription>(), 4);
        assert_eq!(size_of::<FfxApiResource>(), 48);
        assert_eq!(align_of::<FfxApiResource>(), 8);
        assert_eq!(size_of::<FfxCreateBackendVKDesc>(), 40);
        assert_eq!(align_of::<FfxCreateBackendVKDesc>(), 8);
        assert_eq!(size_of::<FfxCreateContextDescUpscale>(), 48);
        assert_eq!(align_of::<FfxCreateContextDescUpscale>(), 8);
        assert_eq!(size_of::<FfxDispatchDescUpscale>(), 432);
        assert_eq!(align_of::<FfxDispatchDescUpscale>(), 8);
        assert_eq!(size_of::<FfxCreateContextDescFrameGeneration>(), 40);
        assert_eq!(align_of::<FfxCreateContextDescFrameGeneration>(), 8);
        assert_eq!(size_of::<FfxDispatchDescFrameGeneration>(), 312);
        assert_eq!(align_of::<FfxDispatchDescFrameGeneration>(), 8);
        assert_eq!(size_of::<FfxDispatchDescFrameGenerationPrepareV2>(), 232);
        assert_eq!(align_of::<FfxDispatchDescFrameGenerationPrepareV2>(), 8);
    }

    #[test]
    fn fsr3_missing_library_reports_honest_error() {
        let missing = Path::new("/nonexistent/vulkanturbo/natives");
        let err = Fsr3Library::load(&[missing])
            .err()
            .expect("must fail cleanly");
        assert!(err.to_string().contains("FidelityFX"));
    }
}
