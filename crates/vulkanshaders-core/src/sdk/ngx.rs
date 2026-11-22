// SPDX-License-Identifier: GPL-3.0-only
// Additional permissions under GPLv3 Section 7 apply to optional proprietary vendor SDK
// runtime loading; see LICENSE-EXCEPTION.md in the repository root.

//! Clean-room NVIDIA NGX / DLSS Vulkan runtime dynamic loader (`libnvidia-ngx.so.1` / `_nvngx.dll`).
//!
//! Loads the driver-shipped NGX Core library via `libloading` at runtime, wraps the C++
//! `NVSDK_NGX_Parameter` vtable interface, constructs `NVSDK_NGX_Resource_VK` descriptors, and
//! exposes DLSS Super Resolution (`SuperSampling`), DLSS Ray Reconstruction (`RayReconstruction`),
//! and DLSS Frame Generation (`FrameGeneration`) feature lifecycles without requiring static vendor
//! SDK linkage at compile time.

use ash::vk;
use libloading::Library;
use std::{
    ffi::{CStr, CString, c_char, c_void},
    fmt,
    path::{Path, PathBuf},
    ptr,
};
use thiserror::Error;

/// NGX SDK API version constant (`NVSDK_NGX_VERSION_API_MACRO` = `0x0000015`).
pub const NVSDK_NGX_VERSION_API: u32 = 0x0000_0015;

/// Default NVIDIA NGX application ID used when no custom NVIDIA AppID is assigned.
pub const DEFAULT_NGX_APP_ID: u64 = 0x5654_5552_424F; // "VTURBO"

/// Maximum `NVSDK_NGX_GBufferType` slots (`NVSDK_NGX_GBUFFERTYPE_NUM`).
pub const NVSDK_NGX_GBUFFER_TYPE_NUM: usize = 17;

/// Platform `wchar_t` code unit (`u32` on Unix/Linux UTF-32, `u16` on Windows UTF-16).
#[cfg(windows)]
pub type PlatformWchar = u16;
#[cfg(not(windows))]
pub type PlatformWchar = u32;

/// Encodes a Rust `&str` as a NUL-terminated platform `wchar_t` buffer.
pub fn to_platform_wchar(s: &str) -> Vec<PlatformWchar> {
    #[cfg(windows)]
    {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }
    #[cfg(not(windows))]
    {
        s.chars()
            .map(|c| c as u32)
            .chain(std::iter::once(0))
            .collect()
    }
}

/// Errors returned while loading or invoking NVIDIA NGX.
#[derive(Debug, Error)]
pub enum NgxError {
    #[error("failed to load NGX Core library ({0})")]
    LibraryLoad(String),
    #[error("missing required NGX symbol `{0}`: {1}")]
    MissingSymbol(&'static str, String),
    #[error("NGX API call `{call}` failed with result {result}")]
    ApiFailure {
        call: &'static str,
        result: NvSdkNgxResult,
    },
    #[error("invalid interior NUL byte in parameter key `{0}`")]
    InvalidString(String),
    #[error("null NGX parameter or handle pointer")]
    NullPointer,
}

/// `NVSDK_NGX_Result` C-ABI status code (`uint32_t`).
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NvSdkNgxResult(pub u32);

impl NvSdkNgxResult {
    pub const SUCCESS: Self = Self(0x0000_0001);
    pub const FAIL: Self = Self(0xBAD0_0000);
    pub const FAIL_FEATURE_NOT_SUPPORTED: Self = Self(0xBAD0_0001);
    pub const FAIL_PLATFORM_ERROR: Self = Self(0xBAD0_0002);
    pub const FAIL_FEATURE_ALREADY_EXISTS: Self = Self(0xBAD0_0003);
    pub const FAIL_FEATURE_NOT_FOUND: Self = Self(0xBAD0_0004);
    pub const FAIL_INVALID_PARAMETER: Self = Self(0xBAD0_0005);
    pub const FAIL_SCRATCH_BUFFER_TOO_SMALL: Self = Self(0xBAD0_0006);
    pub const FAIL_NOT_INITIALIZED: Self = Self(0xBAD0_0007);
    pub const FAIL_UNSUPPORTED_INPUT_FORMAT: Self = Self(0xBAD0_0008);
    pub const FAIL_RW_FLAG_MISSING: Self = Self(0xBAD0_0009);
    pub const FAIL_MISSING_INPUT: Self = Self(0xBAD0_000A);
    pub const FAIL_UNABLE_TO_INITIALIZE_FEATURE: Self = Self(0xBAD0_000B);
    pub const FAIL_OUT_OF_DATE: Self = Self(0xBAD0_000C);
    pub const FAIL_OUT_OF_GPU_MEMORY: Self = Self(0xBAD0_000D);
    pub const FAIL_UNSUPPORTED_FORMAT: Self = Self(0xBAD0_000E);
    pub const FAIL_UNABLE_TO_WRITE_TO_APP_DATA_PATH: Self = Self(0xBAD0_000F);
    pub const FAIL_UNSUPPORTED_PARAMETER: Self = Self(0xBAD0_0010);
    pub const FAIL_DENIED: Self = Self(0xBAD0_0011);
    pub const FAIL_NOT_IMPLEMENTED: Self = Self(0xBAD0_0012);

    /// Matches `NVSDK_NGX_SUCCEED(value)` macro (`(value & 0xFFF00000) != 0xBAD00000`).
    #[inline]
    pub const fn is_success(self) -> bool {
        (self.0 & 0xFFF0_0000) != Self::FAIL.0
    }

    pub const fn name(self) -> &'static str {
        match self.0 {
            0x0000_0001 => "NVSDK_NGX_Result_Success",
            0xBAD0_0000 => "NVSDK_NGX_Result_Fail",
            0xBAD0_0001 => "NVSDK_NGX_Result_FAIL_FeatureNotSupported",
            0xBAD0_0002 => "NVSDK_NGX_Result_FAIL_PlatformError",
            0xBAD0_0003 => "NVSDK_NGX_Result_FAIL_FeatureAlreadyExists",
            0xBAD0_0004 => "NVSDK_NGX_Result_FAIL_FeatureNotFound",
            0xBAD0_0005 => "NVSDK_NGX_Result_FAIL_InvalidParameter",
            0xBAD0_0006 => "NVSDK_NGX_Result_FAIL_ScratchBufferTooSmall",
            0xBAD0_0007 => "NVSDK_NGX_Result_FAIL_NotInitialized",
            0xBAD0_0008 => "NVSDK_NGX_Result_FAIL_UnsupportedInputFormat",
            0xBAD0_0009 => "NVSDK_NGX_Result_FAIL_RWFlagMissing",
            0xBAD0_000A => "NVSDK_NGX_Result_FAIL_MissingInput",
            0xBAD0_000B => "NVSDK_NGX_Result_FAIL_UnableToInitializeFeature",
            0xBAD0_000C => "NVSDK_NGX_Result_FAIL_OutOfDate",
            0xBAD0_000D => "NVSDK_NGX_Result_FAIL_OutOfGPUMemory",
            0xBAD0_000E => "NVSDK_NGX_Result_FAIL_UnsupportedFormat",
            0xBAD0_000F => "NVSDK_NGX_Result_FAIL_UnableToWriteToAppDataPath",
            0xBAD0_0010 => "NVSDK_NGX_Result_FAIL_UnsupportedParameter",
            0xBAD0_0011 => "NVSDK_NGX_Result_FAIL_Denied",
            0xBAD0_0012 => "NVSDK_NGX_Result_FAIL_NotImplemented",
            _ => "NVSDK_NGX_Result_Unknown",
        }
    }
}

impl fmt::Debug for NvSdkNgxResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (0x{:08X})", self.name(), self.0)
    }
}

impl fmt::Display for NvSdkNgxResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (0x{:08X})", self.name(), self.0)
    }
}

/// `NVSDK_NGX_Feature` identifiers (`uint32_t`).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NvSdkNgxFeature {
    SuperSampling = 1,
    InPainting = 2,
    ImageSuperResolution = 3,
    SlowMotion = 4,
    VideoSuperResolution = 5,
    ImageSignalProcessing = 9,
    DeepResolve = 10,
    FrameGeneration = 11,
    DeepDvc = 12,
    RayReconstruction = 13,
}

/// `NVSDK_NGX_PerfQuality_Value` modes.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NvSdkNgxPerfQuality {
    MaxPerf = 0,
    Balanced = 1,
    MaxQuality = 2,
    UltraPerformance = 3,
    UltraQuality = 4,
    Dlaa = 5,
}

/// `NVSDK_NGX_DLSS_Feature_Flags` bitmask.
pub mod dlss_flags {
    pub const NONE: i32 = 0;
    pub const IS_HDR: i32 = 1 << 0;
    pub const MV_LOW_RES: i32 = 1 << 1;
    pub const MV_JITTERED: i32 = 1 << 2;
    pub const DEPTH_INVERTED: i32 = 1 << 3;
    pub const AUTO_EXPOSURE: i32 = 1 << 6;
    pub const ALPHA_UPSCALING: i32 = 1 << 7;

    /// Computes canonical `NVSDK_NGX_DLSS_Feature_Flags` for Hyaloclast:
    /// `IS_HDR | MV_LOW_RES`, plus `DEPTH_INVERTED` when not in Tier 3 (Reverse-Z raster depth).
    pub const fn for_pipeline(is_tier3: bool) -> i32 {
        let mut flags = IS_HDR | MV_LOW_RES;
        if !is_tier3 {
            flags |= DEPTH_INVERTED;
        }
        flags
    }
}

/// `NVSDK_NGX_DLSS_Hint_Render_Preset` values including DLSS 4 Transformer presets (J/K).
pub mod dlss_render_preset {
    pub const DEFAULT: u32 = 0;
    pub const PRESET_A: u32 = 1;
    pub const PRESET_B: u32 = 2;
    pub const PRESET_C: u32 = 3;
    pub const PRESET_D: u32 = 4;
    pub const PRESET_E: u32 = 5;
    pub const PRESET_F: u32 = 6;
    pub const PRESET_G: u32 = 7;
    pub const PRESET_J: u32 = 10;
    pub const PRESET_K: u32 = 11;
}

/// Applies optional DLSS 4 Transformer render preset hints to an `NVSDK_NGX_Parameter` block.
/// Strictly non-fatal: older NGX runtimes safely ignore unrecognized hint keys.
pub fn apply_dlss4_transformer_preset_hints(params: &mut NvSdkNgxParameter, preset: u32) {
    params.set_u32(params::DLSS_HINT_RENDER_PRESET_DLAA, preset);
    params.set_u32(params::DLSS_HINT_RENDER_PRESET_QUALITY, preset);
    params.set_u32(params::DLSS_HINT_RENDER_PRESET_BALANCED, preset);
    params.set_u32(params::DLSS_HINT_RENDER_PRESET_PERFORMANCE, preset);
    params.set_u32(params::DLSS_HINT_RENDER_PRESET_ULTRA_PERFORMANCE, preset);
    params.set_u32(params::DLSS_HINT_RENDER_PRESET_ULTRA_QUALITY, preset);
}

/// `NVSDK_NGX_EngineType`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvSdkNgxEngineType {
    Custom = 0,
    Unreal = 1,
    Unity = 2,
    Omniverse = 3,
}

/// `NVSDK_NGX_Application_Identifier_Type`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvSdkNgxApplicationIdentifierType {
    ApplicationId = 0,
    ProjectId = 1,
}

/// `NVSDK_NGX_Logging_Level`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvSdkNgxLoggingLevel {
    Off = 0,
    On = 1,
    Verbose = 2,
}

/// `NVSDK_NGX_Feature_Support_Result` bitflags (`0` means supported).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvSdkNgxFeatureSupportResult(pub u32);

impl NvSdkNgxFeatureSupportResult {
    pub const SUPPORTED: u32 = 0;
    pub const CHECK_NOT_PRESENT: u32 = 1;
    pub const DRIVER_VERSION_UNSUPPORTED: u32 = 2;
    pub const ADAPTER_UNSUPPORTED: u32 = 4;
    pub const OS_VERSION_BELOW_MINIMUM: u32 = 8;
    pub const NOT_IMPLEMENTED: u32 = 16;

    #[inline]
    pub const fn is_supported(self) -> bool {
        self.0 == Self::SUPPORTED
    }
}

/// Opaque NGX feature handle (`NVSDK_NGX_Handle`).
#[repr(C)]
pub struct NvSdkNgxHandle {
    _private: [u8; 0],
}

/// `NVSDK_NGX_PathListInfo`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvSdkNgxPathListInfo {
    pub path: *const *const PlatformWchar,
    pub length: u32,
}

/// `NVSDK_NGX_AppLogCallback` C-ABI callback signature.
pub type PfnNvSdkNgxAppLogCallback = Option<
    unsafe extern "C" fn(
        message: *const c_char,
        logging_level: NvSdkNgxLoggingLevel,
        source_component: u32,
    ),
>;

/// `NVSDK_NGX_LoggingInfo`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvSdkNgxLoggingInfo {
    pub logging_callback: PfnNvSdkNgxAppLogCallback,
    pub minimum_logging_level: NvSdkNgxLoggingLevel,
    pub disable_other_logging_sinks: bool,
}

/// `NVSDK_NGX_FeatureCommonInfo` (40 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvSdkNgxFeatureCommonInfo {
    pub path_list_info: NvSdkNgxPathListInfo,
    pub internal_data: *mut c_void,
    pub logging_info: NvSdkNgxLoggingInfo,
}

/// `NVSDK_NGX_ProjectIdDescription`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvSdkNgxProjectIdDescription {
    pub project_id: *const c_char,
    pub engine_type: NvSdkNgxEngineType,
    pub engine_version: *const c_char,
}

/// Union payload of `NVSDK_NGX_Application_Identifier`.
#[repr(C)]
#[derive(Clone, Copy)]
pub union NvSdkNgxApplicationIdentifierUnion {
    pub project_desc: NvSdkNgxProjectIdDescription,
    pub application_id: u64,
}

/// `NVSDK_NGX_Application_Identifier`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvSdkNgxApplicationIdentifier {
    pub identifier_type: NvSdkNgxApplicationIdentifierType,
    pub v: NvSdkNgxApplicationIdentifierUnion,
}

/// `NVSDK_NGX_FeatureDiscoveryInfo` (56 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvSdkNgxFeatureDiscoveryInfo {
    pub sdk_version: u32,
    pub feature_id: NvSdkNgxFeature,
    pub identifier: NvSdkNgxApplicationIdentifier,
    pub application_data_path: *const PlatformWchar,
    pub feature_info: *const NvSdkNgxFeatureCommonInfo,
}

/// `NVSDK_NGX_FeatureRequirement` (264 bytes, align 4).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvSdkNgxFeatureRequirement {
    pub feature_supported: NvSdkNgxFeatureSupportResult,
    pub min_hw_architecture: u32,
    pub min_os_version: [c_char; 255],
}

impl Default for NvSdkNgxFeatureRequirement {
    fn default() -> Self {
        Self {
            feature_supported: NvSdkNgxFeatureSupportResult(
                NvSdkNgxFeatureSupportResult::CHECK_NOT_PRESENT,
            ),
            min_hw_architecture: 0,
            min_os_version: [0; 255],
        }
    }
}

/// `NVSDK_NGX_Resource_VK_Type`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvSdkNgxResourceVkType {
    ImageView = 0,
    Buffer = 1,
}

/// `NVSDK_NGX_ImageViewInfo_VK` (48 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvSdkNgxImageViewInfoVk {
    pub image_view: vk::ImageView,
    pub image: vk::Image,
    pub subresource_range: vk::ImageSubresourceRange,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
}

/// `NVSDK_NGX_BufferInfo_VK` (16 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvSdkNgxBufferInfoVk {
    pub buffer: vk::Buffer,
    pub size_in_bytes: u32,
}

/// Union payload of `NVSDK_NGX_Resource_VK`.
#[repr(C)]
#[derive(Clone, Copy)]
pub union NvSdkNgxResourceUnionVk {
    pub image_view_info: NvSdkNgxImageViewInfoVk,
    pub buffer_info: NvSdkNgxBufferInfoVk,
}

/// `NVSDK_NGX_Resource_VK` (56 bytes, align 8; `Type` at offset 48, `ReadWrite` at offset 52).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvSdkNgxResourceVk {
    pub resource: NvSdkNgxResourceUnionVk,
    pub resource_type: NvSdkNgxResourceVkType,
    pub read_write: bool,
}

impl NvSdkNgxResourceVk {
    /// Constructs a Vulkan `VkImageView` resource descriptor matching `NVSDK_NGX_Create_ImageView_Resource_VK`.
    pub fn from_image_view(
        image_view: vk::ImageView,
        image: vk::Image,
        subresource_range: vk::ImageSubresourceRange,
        format: vk::Format,
        width: u32,
        height: u32,
        read_write: bool,
    ) -> Self {
        Self {
            resource: NvSdkNgxResourceUnionVk {
                image_view_info: NvSdkNgxImageViewInfoVk {
                    image_view,
                    image,
                    subresource_range,
                    format,
                    width,
                    height,
                },
            },
            resource_type: NvSdkNgxResourceVkType::ImageView,
            read_write,
        }
    }

    /// Constructs a Vulkan `VkBuffer` resource descriptor matching `NVSDK_NGX_Create_Buffer_Resource_VK`.
    pub fn from_buffer(buffer: vk::Buffer, size_in_bytes: u32, read_write: bool) -> Self {
        Self {
            resource: NvSdkNgxResourceUnionVk {
                buffer_info: NvSdkNgxBufferInfoVk {
                    buffer,
                    size_in_bytes,
                },
            },
            resource_type: NvSdkNgxResourceVkType::Buffer,
            read_write,
        }
    }
}

/// `NVSDK_NGX_Parameter` C++ virtual table layout.
///
/// On Linux/Unix (Itanium C++ ABI), overloaded virtual methods appear in exact declaration order.
/// On Windows (MSVC x64 C++ ABI), overloaded virtual methods within each overload group are
/// ordered in reverse declaration order.
#[cfg(not(target_os = "windows"))]
#[repr(C)]
pub struct NvSdkNgxParameterVTable {
    pub set_ull: unsafe extern "C" fn(*mut NvSdkNgxParameter, *const c_char, u64),
    pub set_f32: unsafe extern "C" fn(*mut NvSdkNgxParameter, *const c_char, f32),
    pub set_f64: unsafe extern "C" fn(*mut NvSdkNgxParameter, *const c_char, f64),
    pub set_u32: unsafe extern "C" fn(*mut NvSdkNgxParameter, *const c_char, u32),
    pub set_i32: unsafe extern "C" fn(*mut NvSdkNgxParameter, *const c_char, i32),
    pub set_d3d11: unsafe extern "C" fn(*mut NvSdkNgxParameter, *const c_char, *mut c_void),
    pub set_d3d12: unsafe extern "C" fn(*mut NvSdkNgxParameter, *const c_char, *mut c_void),
    pub set_void_ptr: unsafe extern "C" fn(*mut NvSdkNgxParameter, *const c_char, *mut c_void),

    pub get_ull:
        unsafe extern "C" fn(*const NvSdkNgxParameter, *const c_char, *mut u64) -> NvSdkNgxResult,
    pub get_f32:
        unsafe extern "C" fn(*const NvSdkNgxParameter, *const c_char, *mut f32) -> NvSdkNgxResult,
    pub get_f64:
        unsafe extern "C" fn(*const NvSdkNgxParameter, *const c_char, *mut f64) -> NvSdkNgxResult,
    pub get_u32:
        unsafe extern "C" fn(*const NvSdkNgxParameter, *const c_char, *mut u32) -> NvSdkNgxResult,
    pub get_i32:
        unsafe extern "C" fn(*const NvSdkNgxParameter, *const c_char, *mut i32) -> NvSdkNgxResult,
    pub get_d3d11: unsafe extern "C" fn(
        *const NvSdkNgxParameter,
        *const c_char,
        *mut *mut c_void,
    ) -> NvSdkNgxResult,
    pub get_d3d12: unsafe extern "C" fn(
        *const NvSdkNgxParameter,
        *const c_char,
        *mut *mut c_void,
    ) -> NvSdkNgxResult,
    pub get_void_ptr: unsafe extern "C" fn(
        *const NvSdkNgxParameter,
        *const c_char,
        *mut *mut c_void,
    ) -> NvSdkNgxResult,

    pub reset: unsafe extern "C" fn(*mut NvSdkNgxParameter),
}

#[cfg(target_os = "windows")]
#[repr(C)]
pub struct NvSdkNgxParameterVTable {
    pub set_void_ptr: unsafe extern "C" fn(*mut NvSdkNgxParameter, *const c_char, *mut c_void),
    pub set_d3d12: unsafe extern "C" fn(*mut NvSdkNgxParameter, *const c_char, *mut c_void),
    pub set_d3d11: unsafe extern "C" fn(*mut NvSdkNgxParameter, *const c_char, *mut c_void),
    pub set_i32: unsafe extern "C" fn(*mut NvSdkNgxParameter, *const c_char, i32),
    pub set_u32: unsafe extern "C" fn(*mut NvSdkNgxParameter, *const c_char, u32),
    pub set_f64: unsafe extern "C" fn(*mut NvSdkNgxParameter, *const c_char, f64),
    pub set_f32: unsafe extern "C" fn(*mut NvSdkNgxParameter, *const c_char, f32),
    pub set_ull: unsafe extern "C" fn(*mut NvSdkNgxParameter, *const c_char, u64),

    pub get_void_ptr: unsafe extern "C" fn(
        *const NvSdkNgxParameter,
        *const c_char,
        *mut *mut c_void,
    ) -> NvSdkNgxResult,
    pub get_d3d12: unsafe extern "C" fn(
        *const NvSdkNgxParameter,
        *const c_char,
        *mut *mut c_void,
    ) -> NvSdkNgxResult,
    pub get_d3d11: unsafe extern "C" fn(
        *const NvSdkNgxParameter,
        *const c_char,
        *mut *mut c_void,
    ) -> NvSdkNgxResult,
    pub get_i32:
        unsafe extern "C" fn(*const NvSdkNgxParameter, *const c_char, *mut i32) -> NvSdkNgxResult,
    pub get_u32:
        unsafe extern "C" fn(*const NvSdkNgxParameter, *const c_char, *mut u32) -> NvSdkNgxResult,
    pub get_f64:
        unsafe extern "C" fn(*const NvSdkNgxParameter, *const c_char, *mut f64) -> NvSdkNgxResult,
    pub get_f32:
        unsafe extern "C" fn(*const NvSdkNgxParameter, *const c_char, *mut f32) -> NvSdkNgxResult,
    pub get_ull:
        unsafe extern "C" fn(*const NvSdkNgxParameter, *const c_char, *mut u64) -> NvSdkNgxResult,

    pub reset: unsafe extern "C" fn(*mut NvSdkNgxParameter),
}

/// C-ABI representation of an `NVSDK_NGX_Parameter` object (`vptr` at offset 0).
#[repr(C)]
pub struct NvSdkNgxParameter {
    pub vtable: *const NvSdkNgxParameterVTable,
}

impl NvSdkNgxParameter {
    #[inline]
    unsafe fn vt(&self) -> &NvSdkNgxParameterVTable {
        unsafe { &*self.vtable }
    }

    pub fn set_u64(&mut self, name: &CStr, value: u64) {
        unsafe { (self.vt().set_ull)(self, name.as_ptr(), value) };
    }

    pub fn set_f32(&mut self, name: &CStr, value: f32) {
        unsafe { (self.vt().set_f32)(self, name.as_ptr(), value) };
    }

    pub fn set_f64(&mut self, name: &CStr, value: f64) {
        unsafe { (self.vt().set_f64)(self, name.as_ptr(), value) };
    }

    pub fn set_u32(&mut self, name: &CStr, value: u32) {
        unsafe { (self.vt().set_u32)(self, name.as_ptr(), value) };
    }

    pub fn set_i32(&mut self, name: &CStr, value: i32) {
        unsafe { (self.vt().set_i32)(self, name.as_ptr(), value) };
    }

    /// Sets an opaque `void*` parameter (such as `*mut NvSdkNgxResourceVk`).
    ///
    /// # Safety
    /// Any pointer passed must remain valid for the duration of the NGX call reading it.
    pub unsafe fn set_void_ptr(&mut self, name: &CStr, value: *mut c_void) {
        unsafe { (self.vt().set_void_ptr)(self, name.as_ptr(), value) };
    }

    pub fn get_u64(&self, name: &CStr) -> Result<u64, NvSdkNgxResult> {
        let mut out = 0u64;
        let res = unsafe { (self.vt().get_ull)(self, name.as_ptr(), &mut out) };
        if res.is_success() { Ok(out) } else { Err(res) }
    }

    pub fn get_f32(&self, name: &CStr) -> Result<f32, NvSdkNgxResult> {
        let mut out = 0.0f32;
        let res = unsafe { (self.vt().get_f32)(self, name.as_ptr(), &mut out) };
        if res.is_success() { Ok(out) } else { Err(res) }
    }

    pub fn get_u32(&self, name: &CStr) -> Result<u32, NvSdkNgxResult> {
        let mut out = 0u32;
        let res = unsafe { (self.vt().get_u32)(self, name.as_ptr(), &mut out) };
        if res.is_success() { Ok(out) } else { Err(res) }
    }

    pub fn get_i32(&self, name: &CStr) -> Result<i32, NvSdkNgxResult> {
        let mut out = 0i32;
        let res = unsafe { (self.vt().get_i32)(self, name.as_ptr(), &mut out) };
        if res.is_success() { Ok(out) } else { Err(res) }
    }

    pub fn get_void_ptr(&self, name: &CStr) -> Result<*mut c_void, NvSdkNgxResult> {
        let mut out = ptr::null_mut();
        let res = unsafe { (self.vt().get_void_ptr)(self, name.as_ptr(), &mut out) };
        if res.is_success() { Ok(out) } else { Err(res) }
    }

    pub fn reset(&mut self) {
        unsafe { (self.vt().reset)(self) };
    }
}

/// Canonical NGX parameter key strings (`nvsdk_ngx_defs.h`).
pub mod params {
    use std::ffi::CStr;

    pub const SUPERSAMPLING_AVAILABLE: &CStr = c"SuperSampling.Available";
    pub const SUPERSAMPLING_NEEDS_UPDATED_DRIVER: &CStr = c"SuperSampling.NeedsUpdatedDriver";
    pub const SUPERSAMPLING_MIN_DRIVER_MAJOR: &CStr = c"SuperSampling.MinDriverVersionMajor";
    pub const SUPERSAMPLING_MIN_DRIVER_MINOR: &CStr = c"SuperSampling.MinDriverVersionMinor";
    pub const SUPERSAMPLING_FEATURE_INIT_RESULT: &CStr = c"SuperSampling.FeatureInitResult";

    pub const SUPERSAMPLING_DENOISING_AVAILABLE: &CStr = c"SuperSamplingDenoising.Available";
    pub const SUPERSAMPLING_DENOISING_INIT_RESULT: &CStr =
        c"SuperSamplingDenoising.FeatureInitResult";

    pub const FRAME_INTERPOLATION_AVAILABLE: &CStr = c"FrameInterpolation.Available";
    pub const FRAME_INTERPOLATION_NEEDS_UPDATED_DRIVER: &CStr =
        c"FrameInterpolation.NeedsUpdatedDriver";
    pub const FRAME_INTERPOLATION_MIN_DRIVER_MAJOR: &CStr =
        c"FrameInterpolation.MinDriverVersionMajor";
    pub const FRAME_INTERPOLATION_FEATURE_INIT_RESULT: &CStr =
        c"FrameInterpolation.FeatureInitResult";
    pub const FRAME_GENERATION_AVAILABLE: &CStr = c"FrameGeneration.Available";
    pub const FRAME_GENERATION_FEATURE_INIT_RESULT: &CStr = c"FrameGeneration.FeatureInitResult";

    pub const CREATION_NODE_MASK: &CStr = c"CreationNodeMask";
    pub const VISIBILITY_NODE_MASK: &CStr = c"VisibilityNodeMask";
    pub const WIDTH: &CStr = c"Width";
    pub const HEIGHT: &CStr = c"Height";
    pub const OUT_WIDTH: &CStr = c"OutWidth";
    pub const OUT_HEIGHT: &CStr = c"OutHeight";
    pub const PERF_QUALITY_VALUE: &CStr = c"PerfQualityValue";
    pub const DLSS_FEATURE_CREATE_FLAGS: &CStr = c"DLSS.Feature.Create.Flags";
    pub const DLSS_ENABLE_OUTPUT_SUBRECTS: &CStr = c"DLSS.Enable.Output.Subrects";
    pub const DLSS_HINT_RENDER_PRESET_DLAA: &CStr = c"DLSS.Hint.Render.Preset.DLAA";
    pub const DLSS_HINT_RENDER_PRESET_QUALITY: &CStr = c"DLSS.Hint.Render.Preset.Quality";
    pub const DLSS_HINT_RENDER_PRESET_BALANCED: &CStr = c"DLSS.Hint.Render.Preset.Balanced";
    pub const DLSS_HINT_RENDER_PRESET_PERFORMANCE: &CStr = c"DLSS.Hint.Render.Preset.Performance";
    pub const DLSS_HINT_RENDER_PRESET_ULTRA_PERFORMANCE: &CStr =
        c"DLSS.Hint.Render.Preset.UltraPerformance";
    pub const DLSS_HINT_RENDER_PRESET_ULTRA_QUALITY: &CStr =
        c"DLSS.Hint.Render.Preset.UltraQuality";

    pub const SCRATCH: &CStr = c"Scratch";
    pub const SCRATCH_SIZE_IN_BYTES: &CStr = c"Scratch.SizeInBytes";
    pub const COLOR: &CStr = c"Color";
    pub const OUTPUT: &CStr = c"Output";
    pub const DEPTH: &CStr = c"Depth";
    pub const MOTION_VECTORS: &CStr = c"MotionVectors";
    pub const TRANSPARENCY_MASK: &CStr = c"TransparencyMask";
    pub const EXPOSURE_TEXTURE: &CStr = c"ExposureTexture";
    pub const BIAS_CURRENT_COLOR_MASK: &CStr = c"DLSS.Input.Bias.Current.Color.Mask";

    pub const JITTER_OFFSET_X: &CStr = c"Jitter.Offset.X";
    pub const JITTER_OFFSET_Y: &CStr = c"Jitter.Offset.Y";
    pub const MV_SCALE_X: &CStr = c"MV.Scale.X";
    pub const MV_SCALE_Y: &CStr = c"MV.Scale.Y";
    pub const SHARPNESS: &CStr = c"Sharpness";
    pub const RESET: &CStr = c"Reset";
    pub const FRAME_TIME_DELTA_MS: &CStr = c"FrameTimeDeltaInMsec";
    pub const PRE_EXPOSURE: &CStr = c"DLSS.Pre.Exposure";
    pub const EXPOSURE_SCALE: &CStr = c"DLSS.Exposure.Scale";
    pub const RENDER_SUBRECT_WIDTH: &CStr = c"DLSS.Render.Subrect.Dimensions.Width";
    pub const RENDER_SUBRECT_HEIGHT: &CStr = c"DLSS.Render.Subrect.Dimensions.Height";

    // DLSS Ray Reconstruction (DLSS-RR) G-Buffer parameters
    pub const DLSS_DENOISE_MODE: &CStr = c"DLSS.Denoise.Mode";
    pub const DLSS_ROUGHNESS_MODE: &CStr = c"DLSS.Roughness.Mode";
    pub const DLSS_USE_HW_DEPTH: &CStr = c"DLSS.Use.HW.Depth";
    pub const DLSS_SUBRECT_WIDTH: &CStr = c"DLSS.Render.Subrect.Dimensions.Width";
    pub const DLSS_SUBRECT_HEIGHT: &CStr = c"DLSS.Render.Subrect.Dimensions.Height";
    pub const DLSS_PRE_EXPOSURE: &CStr = c"DLSS.Pre.Exposure";
    pub const DLSS_EXPOSURE_SCALE: &CStr = c"DLSS.Exposure.Scale";
    pub const DLSS_WORLD_TO_VIEW: &CStr = c"WorldToViewMatrix";
    pub const DLSS_VIEW_TO_CLIP: &CStr = c"ViewToClipMatrix";
    pub const GBUFFER_ALBEDO: &CStr = c"GBuffer.Albedo";
    pub const GBUFFER_DIFFUSE_ALBEDO: &CStr = c"DLSS.Input.DiffuseAlbedo";
    pub const GBUFFER_SPECULAR_ALBEDO: &CStr = c"DLSS.Input.SpecularAlbedo";
    pub const GBUFFER_NORMALS: &CStr = c"GBuffer.Normals";
    pub const GBUFFER_ROUGHNESS: &CStr = c"GBuffer.Roughness";
    pub const GBUFFER_METALLIC: &CStr = c"GBuffer.Metallic";
    pub const GBUFFER_SPECULAR_MVEC: &CStr = c"GBuffer.SpecularMvec";
    pub const RAY_TRACING_HIT_DISTANCE: &CStr = c"RayTracingHitDistance";

    // DLSS Frame Generation parameters
    pub const FI_COLOR1: &CStr = c"Color1";
    pub const FI_COLOR2: &CStr = c"Color2";
    pub const FI_OUTPUT1: &CStr = c"Output1";
    pub const FI_MV1: &CStr = c"MotionVectors1";
    pub const FI_DEPTH1: &CStr = c"Depth1";
    pub const FI_MODE: &CStr = c"FIMode";

    // Snippet path / logging override parameters
    pub const FREE_MEM_ON_RELEASE_FEATURE: &CStr = c"FreeMemOnReleaseFeature";
}

// NGX Vulkan Core C-ABI function pointer types (`NGX_SNIPPET_BUILD` interface exported by
// `libnvidia-ngx.so.1` and `_nvngx.dll`).
pub type PfnNvSdkNgxVulkanInit = unsafe extern "C" fn(
    app_id: u64,
    app_data_path: *const PlatformWchar,
    instance: vk::Instance,
    pd: vk::PhysicalDevice,
    device: vk::Device,
    sdk_version: u32,
) -> NvSdkNgxResult;

pub type PfnNvSdkNgxVulkanInitExt = unsafe extern "C" fn(
    app_id: u64,
    app_data_path: *const PlatformWchar,
    instance: vk::Instance,
    pd: vk::PhysicalDevice,
    device: vk::Device,
    sdk_version: u32,
    parameters: *const NvSdkNgxParameter,
) -> NvSdkNgxResult;

pub type PfnNvSdkNgxVulkanInitExt2 = unsafe extern "C" fn(
    app_id: u64,
    app_data_path: *const PlatformWchar,
    instance: vk::Instance,
    pd: vk::PhysicalDevice,
    device: vk::Device,
    gipa: vk::PFN_vkGetInstanceProcAddr,
    gdpa: vk::PFN_vkGetDeviceProcAddr,
    sdk_version: u32,
    parameters: *const NvSdkNgxParameter,
) -> NvSdkNgxResult;

#[cfg(target_os = "linux")]
pub type PfnNvSdkNgxVulkanShutdown1 =
    unsafe extern "C" fn(device: vk::Device, remaining_contexts: *mut u32) -> NvSdkNgxResult;

#[cfg(not(target_os = "linux"))]
pub type PfnNvSdkNgxVulkanShutdown1 = unsafe extern "C" fn(device: vk::Device) -> NvSdkNgxResult;

pub type PfnNvSdkNgxVulkanGetCapabilityParameters =
    unsafe extern "C" fn(out_parameters: *mut *mut NvSdkNgxParameter) -> NvSdkNgxResult;

pub type PfnNvSdkNgxVulkanAllocateParameters =
    unsafe extern "C" fn(out_parameters: *mut *mut NvSdkNgxParameter) -> NvSdkNgxResult;

pub type PfnNvSdkNgxVulkanDestroyParameters =
    unsafe extern "C" fn(in_parameters: *mut NvSdkNgxParameter) -> NvSdkNgxResult;

pub type PfnNvSdkNgxVulkanGetScratchBufferSize = unsafe extern "C" fn(
    feature_id: NvSdkNgxFeature,
    in_parameters: *const NvSdkNgxParameter,
    out_size_in_bytes: *mut usize,
) -> NvSdkNgxResult;

pub type PfnNvSdkNgxVulkanCreateFeature1 = unsafe extern "C" fn(
    device: vk::Device,
    cmd_list: vk::CommandBuffer,
    feature_id: NvSdkNgxFeature,
    in_parameters: *const NvSdkNgxParameter,
    out_handle: *mut *mut NvSdkNgxHandle,
) -> NvSdkNgxResult;

pub type PfnNvSdkNgxVulkanCreateFeature = unsafe extern "C" fn(
    cmd_list: vk::CommandBuffer,
    feature_id: NvSdkNgxFeature,
    in_parameters: *const NvSdkNgxParameter,
    out_handle: *mut *mut NvSdkNgxHandle,
) -> NvSdkNgxResult;

pub type PfnNvSdkNgxProgressCallbackC =
    Option<unsafe extern "C" fn(current_progress: f32, out_should_cancel: *mut bool)>;

pub type PfnNvSdkNgxVulkanEvaluateFeature = unsafe extern "C" fn(
    cmd_list: vk::CommandBuffer,
    feature_handle: *const NvSdkNgxHandle,
    in_parameters: *const NvSdkNgxParameter,
    callback: PfnNvSdkNgxProgressCallbackC,
) -> NvSdkNgxResult;

pub type PfnNvSdkNgxVulkanReleaseFeature =
    unsafe extern "C" fn(handle: *mut NvSdkNgxHandle) -> NvSdkNgxResult;

pub type PfnNvSdkNgxVulkanGetFeatureInstanceExtensionRequirements =
    unsafe extern "C" fn(
        feature_discovery_info: *const NvSdkNgxFeatureDiscoveryInfo,
        out_extension_count: *mut u32,
        out_extension_properties: *mut *mut vk::ExtensionProperties,
    ) -> NvSdkNgxResult;

pub type PfnNvSdkNgxVulkanGetFeatureDeviceExtensionRequirements =
    unsafe extern "C" fn(
        instance: vk::Instance,
        physical_device: vk::PhysicalDevice,
        feature_discovery_info: *const NvSdkNgxFeatureDiscoveryInfo,
        out_extension_count: *mut u32,
        out_extension_properties: *mut *mut vk::ExtensionProperties,
    ) -> NvSdkNgxResult;

pub type PfnNvSdkNgxVulkanGetFeatureRequirements = unsafe extern "C" fn(
    instance: vk::Instance,
    physical_device: vk::PhysicalDevice,
    feature_discovery_info: *const NvSdkNgxFeatureDiscoveryInfo,
    out_supported: *mut NvSdkNgxFeatureRequirement,
) -> NvSdkNgxResult;

/// Dynamically loaded NGX Vulkan Core symbol dispatch table.
pub struct NgxVulkanLibrary {
    _library: Library,
    pub loaded_path: PathBuf,
    pub init_ext2: PfnNvSdkNgxVulkanInitExt2,
    pub init_ext: Option<PfnNvSdkNgxVulkanInitExt>,
    pub init: Option<PfnNvSdkNgxVulkanInit>,
    pub shutdown1: PfnNvSdkNgxVulkanShutdown1,
    pub get_capability_parameters: PfnNvSdkNgxVulkanGetCapabilityParameters,
    pub allocate_parameters: PfnNvSdkNgxVulkanAllocateParameters,
    pub destroy_parameters: PfnNvSdkNgxVulkanDestroyParameters,
    pub get_scratch_buffer_size: PfnNvSdkNgxVulkanGetScratchBufferSize,
    pub create_feature1: PfnNvSdkNgxVulkanCreateFeature1,
    pub create_feature: Option<PfnNvSdkNgxVulkanCreateFeature>,
    pub evaluate_feature: PfnNvSdkNgxVulkanEvaluateFeature,
    pub release_feature: PfnNvSdkNgxVulkanReleaseFeature,
    pub get_feature_instance_ext_reqs:
        Option<PfnNvSdkNgxVulkanGetFeatureInstanceExtensionRequirements>,
    pub get_feature_device_ext_reqs: Option<PfnNvSdkNgxVulkanGetFeatureDeviceExtensionRequirements>,
    pub get_feature_requirements: Option<PfnNvSdkNgxVulkanGetFeatureRequirements>,
}

impl NgxVulkanLibrary {
    /// Candidate library filenames searched across system paths and optional `search_dirs`.
    pub fn candidate_names() -> &'static [&'static str] {
        #[cfg(target_os = "windows")]
        {
            &["_nvngx.dll", "nvngx.dll"]
        }
        #[cfg(target_os = "macos")]
        {
            &[]
        }
        #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
        {
            &["libnvidia-ngx.so.1", "libnvidia-ngx.so"]
        }
    }

    /// Attempts to load `libnvidia-ngx.so.1` / `_nvngx.dll` from `search_dirs` or system paths.
    pub fn load(search_dirs: &[&Path]) -> Result<Self, NgxError> {
        #[cfg(target_os = "macos")]
        {
            let _ = search_dirs;
            return Err(NgxError::LibraryLoad(
                "NVIDIA NGX / DLSS is not supported on macOS".to_string(),
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
        #[cfg(target_os = "windows")]
        {
            for dir in super::windows_system_driver_dirs() {
                for name in Self::candidate_names() {
                    let p = dir.join(name);
                    if p.exists() {
                        candidates.push(p);
                    }
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

        Err(NgxError::LibraryLoad(last_err))
    }

    fn from_library(lib: Library, loaded_path: PathBuf) -> Result<Self, NgxError> {
        unsafe {
            let init_ext2 = *lib
                .get::<PfnNvSdkNgxVulkanInitExt2>(b"NVSDK_NGX_VULKAN_Init_Ext2\0")
                .map_err(|e| {
                    NgxError::MissingSymbol("NVSDK_NGX_VULKAN_Init_Ext2", e.to_string())
                })?;
            let init_ext = lib
                .get::<PfnNvSdkNgxVulkanInitExt>(b"NVSDK_NGX_VULKAN_Init_Ext\0")
                .ok()
                .map(|s| *s);
            let init = lib
                .get::<PfnNvSdkNgxVulkanInit>(b"NVSDK_NGX_VULKAN_Init\0")
                .ok()
                .map(|s| *s);
            let shutdown1 = *lib
                .get::<PfnNvSdkNgxVulkanShutdown1>(b"NVSDK_NGX_VULKAN_Shutdown1\0")
                .map_err(|e| {
                    NgxError::MissingSymbol("NVSDK_NGX_VULKAN_Shutdown1", e.to_string())
                })?;
            let get_capability_parameters = *lib
                .get::<PfnNvSdkNgxVulkanGetCapabilityParameters>(
                    b"NVSDK_NGX_VULKAN_GetCapabilityParameters\0",
                )
                .map_err(|e| {
                    NgxError::MissingSymbol(
                        "NVSDK_NGX_VULKAN_GetCapabilityParameters",
                        e.to_string(),
                    )
                })?;
            let allocate_parameters = *lib
                .get::<PfnNvSdkNgxVulkanAllocateParameters>(
                    b"NVSDK_NGX_VULKAN_AllocateParameters\0",
                )
                .map_err(|e| {
                    NgxError::MissingSymbol("NVSDK_NGX_VULKAN_AllocateParameters", e.to_string())
                })?;
            let destroy_parameters = *lib
                .get::<PfnNvSdkNgxVulkanDestroyParameters>(b"NVSDK_NGX_VULKAN_DestroyParameters\0")
                .map_err(|e| {
                    NgxError::MissingSymbol("NVSDK_NGX_VULKAN_DestroyParameters", e.to_string())
                })?;
            let get_scratch_buffer_size = *lib
                .get::<PfnNvSdkNgxVulkanGetScratchBufferSize>(
                    b"NVSDK_NGX_VULKAN_GetScratchBufferSize\0",
                )
                .map_err(|e| {
                    NgxError::MissingSymbol("NVSDK_NGX_VULKAN_GetScratchBufferSize", e.to_string())
                })?;
            let create_feature1 = *lib
                .get::<PfnNvSdkNgxVulkanCreateFeature1>(b"NVSDK_NGX_VULKAN_CreateFeature1\0")
                .map_err(|e| {
                    NgxError::MissingSymbol("NVSDK_NGX_VULKAN_CreateFeature1", e.to_string())
                })?;
            let create_feature = lib
                .get::<PfnNvSdkNgxVulkanCreateFeature>(b"NVSDK_NGX_VULKAN_CreateFeature\0")
                .ok()
                .map(|s| *s);
            let evaluate_feature = match lib
                .get::<PfnNvSdkNgxVulkanEvaluateFeature>(b"NVSDK_NGX_VULKAN_EvaluateFeature\0")
            {
                Ok(sym) => *sym,
                Err(_) => *lib
                    .get::<PfnNvSdkNgxVulkanEvaluateFeature>(
                        b"NVSDK_NGX_VULKAN_EvaluateFeature_C\0",
                    )
                    .map_err(|e| {
                        NgxError::MissingSymbol("NVSDK_NGX_VULKAN_EvaluateFeature", e.to_string())
                    })?,
            };
            let release_feature = *lib
                .get::<PfnNvSdkNgxVulkanReleaseFeature>(b"NVSDK_NGX_VULKAN_ReleaseFeature\0")
                .map_err(|e| {
                    NgxError::MissingSymbol("NVSDK_NGX_VULKAN_ReleaseFeature", e.to_string())
                })?;
            let get_feature_instance_ext_reqs = lib
                .get::<PfnNvSdkNgxVulkanGetFeatureInstanceExtensionRequirements>(
                    b"NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements\0",
                )
                .ok()
                .map(|s| *s);
            let get_feature_device_ext_reqs = lib
                .get::<PfnNvSdkNgxVulkanGetFeatureDeviceExtensionRequirements>(
                    b"NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements\0",
                )
                .ok()
                .map(|s| *s);
            let get_feature_requirements = lib
                .get::<PfnNvSdkNgxVulkanGetFeatureRequirements>(
                    b"NVSDK_NGX_VULKAN_GetFeatureRequirements\0",
                )
                .ok()
                .map(|s| *s);

            Ok(Self {
                _library: lib,
                loaded_path,
                init_ext2,
                init_ext,
                init,
                shutdown1,
                get_capability_parameters,
                allocate_parameters,
                destroy_parameters,
                get_scratch_buffer_size,
                create_feature1,
                create_feature,
                evaluate_feature,
                release_feature,
                get_feature_instance_ext_reqs,
                get_feature_device_ext_reqs,
                get_feature_requirements,
            })
        }
    }

    /// Queries Vulkan instance extension names required by `feature` before `vkCreateInstance`.
    pub fn query_instance_extensions(
        &self,
        feature: NvSdkNgxFeature,
        app_id: u64,
        app_data_path: &str,
        snippet_dirs: &[&Path],
    ) -> Result<Vec<String>, NgxError> {
        let Some(func) = self.get_feature_instance_ext_reqs else {
            return Ok(Vec::new());
        };
        let (_app_w, _dir_ws, _dir_ptrs, common, discovery) =
            build_discovery_info(feature, app_id, app_data_path, snippet_dirs);
        let mut count = 0u32;
        let mut props_ptr: *mut vk::ExtensionProperties = ptr::null_mut();
        let res = unsafe { func(&discovery, &mut count, &mut props_ptr) };
        let _keep_alive = &common;
        check_ngx(
            "NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements",
            res,
        )?;
        Ok(extract_extension_names(props_ptr, count))
    }

    /// Queries Vulkan device extension names required by `feature` before `vkCreateDevice`.
    pub fn query_device_extensions(
        &self,
        instance: vk::Instance,
        physical_device: vk::PhysicalDevice,
        feature: NvSdkNgxFeature,
        app_id: u64,
        app_data_path: &str,
        snippet_dirs: &[&Path],
    ) -> Result<Vec<String>, NgxError> {
        let Some(func) = self.get_feature_device_ext_reqs else {
            return Ok(Vec::new());
        };
        let (_app_w, _dir_ws, _dir_ptrs, common, discovery) =
            build_discovery_info(feature, app_id, app_data_path, snippet_dirs);
        let mut count = 0u32;
        let mut props_ptr: *mut vk::ExtensionProperties = ptr::null_mut();
        let res = unsafe {
            func(
                instance,
                physical_device,
                &discovery,
                &mut count,
                &mut props_ptr,
            )
        };
        let _keep_alive = &common;
        check_ngx(
            "NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements",
            res,
        )?;
        Ok(extract_extension_names(props_ptr, count))
    }

    /// Queries hardware/driver feature requirements without needing a full `NVSDK_NGX_VULKAN_Init`.
    pub fn query_feature_requirements(
        &self,
        instance: vk::Instance,
        physical_device: vk::PhysicalDevice,
        feature: NvSdkNgxFeature,
        app_id: u64,
        app_data_path: &str,
        snippet_dirs: &[&Path],
    ) -> Result<NvSdkNgxFeatureRequirement, NgxError> {
        let Some(func) = self.get_feature_requirements else {
            return Err(NgxError::MissingSymbol(
                "NVSDK_NGX_VULKAN_GetFeatureRequirements",
                "symbol not exported".to_string(),
            ));
        };
        let (_app_w, _dir_ws, _dir_ptrs, common, discovery) =
            build_discovery_info(feature, app_id, app_data_path, snippet_dirs);
        let mut req = NvSdkNgxFeatureRequirement::default();
        let res = unsafe { func(instance, physical_device, &discovery, &mut req) };
        let _keep_alive = &common;
        check_ngx("NVSDK_NGX_VULKAN_GetFeatureRequirements", res)?;
        Ok(req)
    }
}

/// Active NGX Vulkan runtime bound to a specific `VkDevice`.
pub struct NgxVulkanContext {
    lib: std::sync::Arc<NgxVulkanLibrary>,
    device: vk::Device,
    initialized: bool,
}

impl NgxVulkanContext {
    unsafe extern "C" fn log_native_message(
        message: *const c_char,
        _level: NvSdkNgxLoggingLevel,
        _source: u32,
    ) {
        if !message.is_null() {
            let message = unsafe { CStr::from_ptr(message) }.to_string_lossy();
            eprintln!("[NGX] {}", message.trim_end());
        }
    }

    /// Initializes NGX on `instance` / `physical_device` / `device` via `NVSDK_NGX_VULKAN_Init_Ext2`.
    ///
    /// # Safety
    /// Vulkan handles and `gipa`/`gdpa` function pointers must remain valid until `shutdown()` or drop.
    pub unsafe fn init(
        lib: std::sync::Arc<NgxVulkanLibrary>,
        app_id: u64,
        app_data_path: &str,
        instance: vk::Instance,
        physical_device: vk::PhysicalDevice,
        device: vk::Device,
        gipa: vk::PFN_vkGetInstanceProcAddr,
        gdpa: vk::PFN_vkGetDeviceProcAddr,
    ) -> Result<Self, NgxError> {
        unsafe {
            Self::init_with_snippet_dirs(
                lib,
                app_id,
                app_data_path,
                &[],
                instance,
                physical_device,
                device,
                gipa,
                gdpa,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn init_with_snippet_dirs(
        lib: std::sync::Arc<NgxVulkanLibrary>,
        app_id: u64,
        app_data_path: &str,
        snippet_dirs: &[PathBuf],
        instance: vk::Instance,
        physical_device: vk::PhysicalDevice,
        device: vk::Device,
        gipa: vk::PFN_vkGetInstanceProcAddr,
        gdpa: vk::PFN_vkGetDeviceProcAddr,
    ) -> Result<Self, NgxError> {
        let app_data_w = to_platform_wchar(app_data_path);
        let mut owned_wpaths: Vec<Vec<PlatformWchar>> = snippet_dirs
            .iter()
            .filter_map(|p| p.to_str())
            .map(to_platform_wchar)
            .collect();
        owned_wpaths.push(app_data_w.clone());
        let path_ptrs: Vec<*const PlatformWchar> =
            owned_wpaths.iter().map(|w| w.as_ptr()).collect();
        let debug_logging = std::env::var_os("VS_CORE_NGX_DEBUG").is_some();
        let common_info = NvSdkNgxFeatureCommonInfo {
            path_list_info: NvSdkNgxPathListInfo {
                path: path_ptrs.as_ptr(),
                length: path_ptrs.len() as u32,
            },
            internal_data: ptr::null_mut(),
            logging_info: NvSdkNgxLoggingInfo {
                logging_callback: debug_logging.then_some(Self::log_native_message),
                minimum_logging_level: if debug_logging {
                    NvSdkNgxLoggingLevel::Verbose
                } else {
                    NvSdkNgxLoggingLevel::Off
                },
                disable_other_logging_sinks: false,
            },
        };
        let res = unsafe {
            (lib.init_ext2)(
                app_id,
                app_data_w.as_ptr(),
                instance,
                physical_device,
                device,
                gipa,
                gdpa,
                NVSDK_NGX_VERSION_API,
                (&common_info as *const NvSdkNgxFeatureCommonInfo).cast(),
            )
        };
        check_ngx("NVSDK_NGX_VULKAN_Init_Ext2", res)?;
        Ok(Self {
            lib,
            device,
            initialized: true,
        })
    }

    /// Allocates a new empty NGX parameter map (`NVSDK_NGX_VULKAN_AllocateParameters`).
    pub fn allocate_parameters(&self) -> Result<NgxOwnedParameters, NgxError> {
        let mut ptr_out: *mut NvSdkNgxParameter = ptr::null_mut();
        let res = unsafe { (self.lib.allocate_parameters)(&mut ptr_out) };
        check_ngx("NVSDK_NGX_VULKAN_AllocateParameters", res)?;
        if ptr_out.is_null() {
            return Err(NgxError::NullPointer);
        }
        Ok(NgxOwnedParameters {
            ptr: ptr_out,
            destroy_fn: self.lib.destroy_parameters,
        })
    }

    /// Allocates an NGX parameter map pre-populated with capability flags
    /// (`NVSDK_NGX_VULKAN_GetCapabilityParameters`).
    pub fn get_capability_parameters(&self) -> Result<NgxOwnedParameters, NgxError> {
        let mut ptr_out: *mut NvSdkNgxParameter = ptr::null_mut();
        let res = unsafe { (self.lib.get_capability_parameters)(&mut ptr_out) };
        check_ngx("NVSDK_NGX_VULKAN_GetCapabilityParameters", res)?;
        if ptr_out.is_null() {
            return Err(NgxError::NullPointer);
        }
        Ok(NgxOwnedParameters {
            ptr: ptr_out,
            destroy_fn: self.lib.destroy_parameters,
        })
    }

    /// Queries the scratch buffer byte size needed for `feature`.
    pub fn get_scratch_buffer_size(
        &self,
        feature: NvSdkNgxFeature,
        params: &NvSdkNgxParameter,
    ) -> Result<usize, NgxError> {
        let mut size_bytes = 0usize;
        let res = unsafe { (self.lib.get_scratch_buffer_size)(feature, params, &mut size_bytes) };
        check_ngx("NVSDK_NGX_VULKAN_GetScratchBufferSize", res)?;
        Ok(size_bytes)
    }

    /// Creates an NGX feature (`SuperSampling`, `RayReconstruction`, or `FrameGeneration`).
    ///
    /// # Safety
    /// `cmd` must be an open, recording `VkCommandBuffer` belonging to `self.device`.
    pub unsafe fn create_feature(
        &self,
        cmd: vk::CommandBuffer,
        feature: NvSdkNgxFeature,
        params: &NvSdkNgxParameter,
    ) -> Result<*mut NvSdkNgxHandle, NgxError> {
        let mut handle: *mut NvSdkNgxHandle = ptr::null_mut();
        let res =
            unsafe { (self.lib.create_feature1)(self.device, cmd, feature, params, &mut handle) };
        check_ngx("NVSDK_NGX_VULKAN_CreateFeature1", res)?;
        if handle.is_null() {
            return Err(NgxError::NullPointer);
        }
        Ok(handle)
    }

    /// Evaluates an active NGX feature on `cmd`.
    ///
    /// # Safety
    /// `cmd` must be recording, `handle` must be alive, and all resources in `params` must be in
    /// valid Vulkan image layouts for NGX consumption.
    pub unsafe fn evaluate_feature(
        &self,
        cmd: vk::CommandBuffer,
        handle: *const NvSdkNgxHandle,
        params: &NvSdkNgxParameter,
    ) -> Result<(), NgxError> {
        if handle.is_null() {
            return Err(NgxError::NullPointer);
        }
        let res = unsafe { (self.lib.evaluate_feature)(cmd, handle, params, None) };
        check_ngx("NVSDK_NGX_VULKAN_EvaluateFeature", res)
    }

    /// Releases a previously created NGX feature handle (`NVSDK_NGX_VULKAN_ReleaseFeature`).
    ///
    /// # Safety
    /// All GPU commands referencing `handle` must have finished executing.
    pub unsafe fn release_feature(&self, handle: *mut NvSdkNgxHandle) -> Result<(), NgxError> {
        if handle.is_null() {
            return Ok(());
        }
        let res = unsafe { (self.lib.release_feature)(handle) };
        check_ngx("NVSDK_NGX_VULKAN_ReleaseFeature", res)
    }

    /// Creates a DLSS Super Resolution (`SuperSampling`) feature handle.
    ///
    /// # Safety
    /// `cmd` must be recording on `self.device`.
    pub unsafe fn create_dlss_sr(
        &self,
        cmd: vk::CommandBuffer,
        params: &mut NvSdkNgxParameter,
        render_width: u32,
        render_height: u32,
        output_width: u32,
        output_height: u32,
        quality: NvSdkNgxPerfQuality,
        flags: i32,
    ) -> Result<*mut NvSdkNgxHandle, NgxError> {
        params.set_u32(params::CREATION_NODE_MASK, 1);
        params.set_u32(params::VISIBILITY_NODE_MASK, 1);
        params.set_u32(params::WIDTH, render_width);
        params.set_u32(params::HEIGHT, render_height);
        params.set_u32(params::OUT_WIDTH, output_width);
        params.set_u32(params::OUT_HEIGHT, output_height);
        params.set_i32(params::PERF_QUALITY_VALUE, quality as i32);
        params.set_i32(params::DLSS_FEATURE_CREATE_FLAGS, flags);
        params.set_i32(params::DLSS_ENABLE_OUTPUT_SUBRECTS, 0);
        unsafe { self.create_feature(cmd, NvSdkNgxFeature::SuperSampling, params) }
    }

    /// Creates a DLSS Ray Reconstruction (`RayReconstruction`) feature handle.
    ///
    /// # Safety
    /// `cmd` must be recording on `self.device`.
    pub unsafe fn create_dlss_rr(
        &self,
        cmd: vk::CommandBuffer,
        params: &mut NvSdkNgxParameter,
        render_width: u32,
        render_height: u32,
        output_width: u32,
        output_height: u32,
        quality: NvSdkNgxPerfQuality,
        flags: i32,
    ) -> Result<*mut NvSdkNgxHandle, NgxError> {
        params.set_u32(params::CREATION_NODE_MASK, 1);
        params.set_u32(params::VISIBILITY_NODE_MASK, 1);
        params.set_u32(params::WIDTH, render_width);
        params.set_u32(params::HEIGHT, render_height);
        params.set_u32(params::OUT_WIDTH, output_width);
        params.set_u32(params::OUT_HEIGHT, output_height);
        params.set_i32(params::PERF_QUALITY_VALUE, quality as i32);
        params.set_i32(params::DLSS_FEATURE_CREATE_FLAGS, flags);
        params.set_i32(params::DLSS_ENABLE_OUTPUT_SUBRECTS, 0);
        params.set_i32(params::DLSS_DENOISE_MODE, 1); // DLUnified
        params.set_u32(params::DLSS_ROUGHNESS_MODE, 0); // Separate roughness input
        params.set_u32(
            params::DLSS_USE_HW_DEPTH,
            u32::from(flags & dlss_flags::DEPTH_INVERTED != 0),
        );
        unsafe { self.create_feature(cmd, NvSdkNgxFeature::RayReconstruction, params) }
    }

    /// Creates a DLSS Frame Generation (`FrameGeneration`) feature handle.
    ///
    /// # Safety
    /// `cmd` must be recording on `self.device`.
    pub unsafe fn create_dlss_fg(
        &self,
        cmd: vk::CommandBuffer,
        params: &mut NvSdkNgxParameter,
        width: u32,
        height: u32,
    ) -> Result<*mut NvSdkNgxHandle, NgxError> {
        params.set_u32(params::CREATION_NODE_MASK, 1);
        params.set_u32(params::VISIBILITY_NODE_MASK, 1);
        params.set_u32(params::WIDTH, width);
        params.set_u32(params::HEIGHT, height);
        params.set_u32(params::OUT_WIDTH, width);
        params.set_u32(params::OUT_HEIGHT, height);
        unsafe { self.create_feature(cmd, NvSdkNgxFeature::FrameGeneration, params) }
    }

    /// Shuts down NGX for `self.device` (`NVSDK_NGX_VULKAN_Shutdown1`).
    pub fn shutdown(&mut self) -> Result<(), NgxError> {
        if !self.initialized {
            return Ok(());
        }
        self.initialized = false;
        // The Linux driver export writes a context count through a second argument
        // that is absent from the public SDK wrapper declaration.
        #[cfg(target_os = "linux")]
        let res = {
            let mut remaining_contexts = 0;
            unsafe { (self.lib.shutdown1)(self.device, &mut remaining_contexts) }
        };
        #[cfg(not(target_os = "linux"))]
        let res = unsafe { (self.lib.shutdown1)(self.device) };
        check_ngx("NVSDK_NGX_VULKAN_Shutdown1", res)
    }
}

impl Drop for NgxVulkanContext {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

/// RAII wrapper around an NGX-allocated `NVSDK_NGX_Parameter` instance.
pub struct NgxOwnedParameters {
    ptr: *mut NvSdkNgxParameter,
    destroy_fn: PfnNvSdkNgxVulkanDestroyParameters,
}

impl NgxOwnedParameters {
    pub fn as_ref(&self) -> &NvSdkNgxParameter {
        unsafe { &*self.ptr }
    }

    pub fn as_mut(&mut self) -> &mut NvSdkNgxParameter {
        unsafe { &mut *self.ptr }
    }

    /// Helper setting a `String` parameter key by name (allocating a `CString`).
    pub fn set_u32_named(&mut self, key: &str, value: u32) -> Result<(), NgxError> {
        let ckey = CString::new(key).map_err(|_| NgxError::InvalidString(key.to_string()))?;
        self.as_mut().set_u32(&ckey, value);
        Ok(())
    }
}

impl Drop for NgxOwnedParameters {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = (self.destroy_fn)(self.ptr);
            }
            self.ptr = ptr::null_mut();
        }
    }
}

fn check_ngx(call: &'static str, result: NvSdkNgxResult) -> Result<(), NgxError> {
    if result.is_success() {
        Ok(())
    } else {
        Err(NgxError::ApiFailure { call, result })
    }
}

#[allow(clippy::type_complexity)]
fn build_discovery_info(
    feature: NvSdkNgxFeature,
    app_id: u64,
    app_data_path: &str,
    snippet_dirs: &[&Path],
) -> (
    Vec<PlatformWchar>,
    Vec<Vec<PlatformWchar>>,
    Vec<*const PlatformWchar>,
    Box<NvSdkNgxFeatureCommonInfo>,
    NvSdkNgxFeatureDiscoveryInfo,
) {
    let app_w = to_platform_wchar(app_data_path);
    let dir_ws: Vec<Vec<PlatformWchar>> = snippet_dirs
        .iter()
        .map(|p| to_platform_wchar(&p.to_string_lossy()))
        .collect();
    let dir_ptrs: Vec<*const PlatformWchar> = dir_ws.iter().map(|w| w.as_ptr()).collect();

    let common = Box::new(NvSdkNgxFeatureCommonInfo {
        path_list_info: NvSdkNgxPathListInfo {
            path: if dir_ptrs.is_empty() {
                ptr::null()
            } else {
                dir_ptrs.as_ptr()
            },
            length: dir_ptrs.len() as u32,
        },
        internal_data: ptr::null_mut(),
        logging_info: NvSdkNgxLoggingInfo {
            logging_callback: None,
            minimum_logging_level: NvSdkNgxLoggingLevel::Off,
            disable_other_logging_sinks: false,
        },
    });

    let discovery = NvSdkNgxFeatureDiscoveryInfo {
        sdk_version: NVSDK_NGX_VERSION_API,
        feature_id: feature,
        identifier: NvSdkNgxApplicationIdentifier {
            identifier_type: NvSdkNgxApplicationIdentifierType::ApplicationId,
            v: NvSdkNgxApplicationIdentifierUnion {
                application_id: app_id,
            },
        },
        application_data_path: app_w.as_ptr(),
        feature_info: &*common as *const NvSdkNgxFeatureCommonInfo,
    };

    (app_w, dir_ws, dir_ptrs, common, discovery)
}

fn extract_extension_names(props_ptr: *mut vk::ExtensionProperties, count: u32) -> Vec<String> {
    if props_ptr.is_null() || count == 0 {
        return Vec::new();
    }
    let slice = unsafe { std::slice::from_raw_parts(props_ptr, count as usize) };
    slice
        .iter()
        .filter_map(|p| {
            let cstr = unsafe { CStr::from_ptr(p.extension_name.as_ptr()) };
            cstr.to_str().ok().map(str::to_owned)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn verify_ngx_c_abi_layouts() {
        assert_eq!(size_of::<NvSdkNgxImageViewInfoVk>(), 48);
        assert_eq!(align_of::<NvSdkNgxImageViewInfoVk>(), 8);
        assert_eq!(size_of::<NvSdkNgxBufferInfoVk>(), 16);
        assert_eq!(align_of::<NvSdkNgxBufferInfoVk>(), 8);
        assert_eq!(size_of::<NvSdkNgxResourceVk>(), 56);
        assert_eq!(align_of::<NvSdkNgxResourceVk>(), 8);
        assert_eq!(offset_of!(NvSdkNgxResourceVk, resource_type), 48);
        assert_eq!(offset_of!(NvSdkNgxResourceVk, read_write), 52);
        assert_eq!(size_of::<NvSdkNgxFeatureCommonInfo>(), 40);
        assert_eq!(align_of::<NvSdkNgxFeatureCommonInfo>(), 8);
        assert_eq!(size_of::<NvSdkNgxFeatureDiscoveryInfo>(), 56);
        assert_eq!(align_of::<NvSdkNgxFeatureDiscoveryInfo>(), 8);
        assert_eq!(size_of::<NvSdkNgxFeatureRequirement>(), 264);
        assert_eq!(align_of::<NvSdkNgxFeatureRequirement>(), 4);
        assert_eq!(size_of::<NvSdkNgxParameter>(), 8);
    }

    #[test]
    fn construct_ngx_vulkan_resource_descriptors() {
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let res = NvSdkNgxResourceVk::from_image_view(
            vk::ImageView::default(),
            vk::Image::default(),
            range,
            vk::Format::R16G16B16A16_SFLOAT,
            1920,
            1080,
            true,
        );
        assert_eq!(res.resource_type, NvSdkNgxResourceVkType::ImageView);
        assert!(res.read_write);
        unsafe {
            assert_eq!(res.resource.image_view_info.width, 1920);
            assert_eq!(res.resource.image_view_info.height, 1080);
            assert_eq!(
                res.resource.image_view_info.format,
                vk::Format::R16G16B16A16_SFLOAT
            );
        }
    }

    #[test]
    fn resolve_system_ngx_symbols_if_present() {
        let res = NgxVulkanLibrary::load(&[]);
        if Path::new("/usr/lib/libnvidia-ngx.so.1").exists() {
            let lib = res.expect("system libnvidia-ngx.so.1 must load and resolve all symbols");
            assert!(!lib.loaded_path.as_os_str().is_empty());
            assert!(lib.get_feature_instance_ext_reqs.is_some());
            assert!(lib.get_feature_device_ext_reqs.is_some());
            assert!(lib.get_feature_requirements.is_some());
        } else if let Err(err) = res {
            let msg = err.to_string();
            assert!(msg.contains("NGX Core"), "unexpected error: {msg}");
        }
    }
}
