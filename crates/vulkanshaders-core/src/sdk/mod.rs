// SPDX-License-Identifier: GPL-3.0-only
// Additional permissions under GPLv3 Section 7 apply to optional proprietary vendor SDK
// runtime loading; see LICENSE-EXCEPTION.md in the repository root.

//! Clean-room runtime dynamic loader & capability manager for external vendor rendering SDKs
//! (NVIDIA NGX/DLSS, NVIDIA NRD, AMD FidelityFX FSR 3, and Intel XeSS).
//!
//! All vendor runtimes are discovered and bound at runtime via `libloading` so `vulkanshaders-core`
//! carries zero hard link-time dependencies on proprietary vendor SDK libraries and degrades
//! gracefully when a given vendor runtime is unavailable on the host system.

pub mod fsr3;
pub mod ngx;
pub mod nrd;
pub mod xess;

use crate::gpu::retire::RetireQueue;
use ash::vk::{self, Handle};
use fsr3::{
    FfxApiDimensions2D, FfxApiFloatCoords2D, FfxApiHeader, FfxApiResource, FfxDispatchDescUpscale,
    Fsr3AbiKind, Fsr3Library, Fsr3UpscaleContext, resource_state, resource_usage,
};
use ngx::{
    DEFAULT_NGX_APP_ID, NgxError, NgxVulkanContext, NgxVulkanLibrary, NvSdkNgxFeature,
    NvSdkNgxHandle, NvSdkNgxPerfQuality, NvSdkNgxResourceVk, params as ngx_params,
};
use nrd::{
    NrdCommonSettings, NrdDenoiser, NrdDenoiserDesc, NrdInstance, NrdLibrary, NrdReblurSettings,
    NrdRelaxSettings, NrdUserTextureSet, NrdVulkanDispatcher,
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};
use tracing::{info, warn};
use xess::{
    Xess2d, XessContext, XessLibrary, XessQualitySettings, XessVersion, XessVkExecuteParams,
    XessVkImageViewInfo, XessVkInitParams,
};

/// Resources and parameters handed to a vendor upscaler/denoiser `evaluate` call. Image views are
/// supplied alongside their backing images because FSR3 consumes `VkImage` handles while XeSS
/// consumes `VkImageView` handles and NRD consumes its own resource descriptors.
#[derive(Debug, Clone, Copy)]
pub struct SdkDispatchParams {
    pub command_buffer: vk::CommandBuffer,
    pub input_image: vk::Image,
    pub input_view: vk::ImageView,
    pub output_image: vk::Image,
    pub output_view: vk::ImageView,
    pub depth_image: vk::Image,
    pub depth_format: vk::Format,
    pub depth_view: vk::ImageView,
    pub motion_image: vk::Image,
    pub motion_view: vk::ImageView,
    pub width: u32,
    pub height: u32,
    pub out_width: u32,
    pub out_height: u32,
    pub jitter: [f32; 2],
    pub frame_time_delta: f32,
    /// 0 = off, 1 = TSR, 2 = FSR3, 3 = DLSS-SR, 4 = XeSS (matches `UpscalerMode.ordinal`).
    pub upscaler_mode: u32,
    /// 0 = off, 1 = SVGF, 2 = NRD (matches `DenoiserMode.ordinal`).
    pub denoiser_mode: u32,
    /// 0 = off, 1 = optical flow, 2 = FSR3-FG, 3 = DLSS-FG (matches `FrameGenMode.ordinal`).
    pub framegen_mode: u32,
    pub reset: bool,
    pub hdr_enabled: bool,
    pub sun_dir: [f32; 3],
    pub nrd_in_diff: (vk::Image, vk::ImageView),
    pub nrd_in_spec: (vk::Image, vk::ImageView),
    pub nrd_out_spec: (vk::Image, vk::ImageView),
    pub nrd_normal_rough: (vk::Image, vk::ImageView),
    pub nrd_viewz: (vk::Image, vk::ImageView),
    pub rt_hit_dist: (vk::Image, vk::ImageView),
    pub dlss_rr_output_image: vk::Image,
    pub dlss_rr_output_view: vk::ImageView,
    pub is_tier3: bool,
}

impl Default for SdkDispatchParams {
    fn default() -> Self {
        Self {
            command_buffer: vk::CommandBuffer::null(),
            input_image: vk::Image::null(),
            input_view: vk::ImageView::null(),
            output_image: vk::Image::null(),
            output_view: vk::ImageView::null(),
            depth_image: vk::Image::null(),
            depth_format: vk::Format::UNDEFINED,
            depth_view: vk::ImageView::null(),
            motion_image: vk::Image::null(),
            motion_view: vk::ImageView::null(),
            width: 0,
            height: 0,
            out_width: 0,
            out_height: 0,
            jitter: [0.0, 0.0],
            frame_time_delta: 0.016,
            upscaler_mode: 0,
            denoiser_mode: 0,
            framegen_mode: 0,
            reset: false,
            hdr_enabled: false,
            sun_dir: [0.0, 1.0, 0.0],
            nrd_in_diff: (vk::Image::null(), vk::ImageView::null()),
            nrd_in_spec: (vk::Image::null(), vk::ImageView::null()),
            nrd_out_spec: (vk::Image::null(), vk::ImageView::null()),
            nrd_normal_rough: (vk::Image::null(), vk::ImageView::null()),
            nrd_viewz: (vk::Image::null(), vk::ImageView::null()),
            rt_hit_dist: (vk::Image::null(), vk::ImageView::null()),
            dlss_rr_output_image: vk::Image::null(),
            dlss_rr_output_view: vk::ImageView::null(),
            is_tier3: false,
        }
    }
}

/// Optional extended camera/matrix state for vendor SDK dispatches (NRD, FSR3, DLSS).
#[derive(Debug, Clone, Copy)]
pub struct SdkCameraMatrices {
    /// Row-major 4x4 matrices as produced by `WorldUboData` (the newer native core's shape).
    pub view_to_clip: [[f32; 4]; 4],
    pub view_to_clip_prev: [[f32; 4]; 4],
    pub world_to_view: [[f32; 4]; 4],
    pub world_to_view_prev: [[f32; 4]; 4],
    pub camera_near: f32,
    pub camera_far: f32,
    /// Vertical field of view in radians (newer core field name).
    pub camera_fov_y_radians: f32,
    pub depth_inverted: bool,
    pub frame_index: u32,
    pub jitter_prev: [f32; 2],
}

impl SdkCameraMatrices {
    /// Flattens a row-major `[[f32; 4]; 4]` into the `[f32; 16]` layout expected by vendor APIs.
    pub fn flatten(m: [[f32; 4]; 4]) -> [f32; 16] {
        let mut out = [0.0f32; 16];
        for (r, row) in m.iter().enumerate() {
            out[r * 4..r * 4 + 4].copy_from_slice(row);
        }
        out
    }
}

impl Default for SdkCameraMatrices {
    fn default() -> Self {
        let common = NrdCommonSettings::default();
        let to_2d = |m: [f32; 16]| -> [[f32; 4]; 4] {
            [
                [m[0], m[1], m[2], m[3]],
                [m[4], m[5], m[6], m[7]],
                [m[8], m[9], m[10], m[11]],
                [m[12], m[13], m[14], m[15]],
            ]
        };
        Self {
            view_to_clip: to_2d(common.view_to_clip_matrix),
            view_to_clip_prev: to_2d(common.view_to_clip_matrix_prev),
            world_to_view: to_2d(common.world_to_view_matrix),
            world_to_view_prev: to_2d(common.world_to_view_matrix_prev),
            camera_near: 0.05,
            camera_far: 512.0,
            camera_fov_y_radians: 1.22173, // 70 deg in radians
            depth_inverted: true,          // RenderPearl uses Reverse-Z in Tier 0..2
            frame_index: 0,
            jitter_prev: [0.0, 0.0],
        }
    }
}

/// Effective upscaler mode resolved after checking runtime SDK availability and platform
/// support (DLSS-SR -> FSR3 -> XeSS -> TSR fallback chain).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectiveUpscaler {
    Off,
    Tsr,
    Fsr3,
    DlssSr,
    XeSS,
}

impl EffectiveUpscaler {
    pub const fn to_mode_ordinal(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::Tsr => 1,
            Self::Fsr3 => 2,
            Self::DlssSr => 3,
            Self::XeSS => 4,
        }
    }
}

/// Pure helper resolving `requested_mode` (`0=off, 1=TSR, 2=FSR3, 3=DLSS-SR, 4=XeSS`)
/// against `caps` and `is_macos`, returning the effective upscaler and an optional fallback
/// reason string when the requested backend is unavailable.
pub fn resolve_upscaler_fallback_for_caps(
    caps: &RuntimeSdkCapabilities,
    requested_mode: u32,
    is_macos: bool,
) -> (EffectiveUpscaler, Option<String>) {
    match requested_mode {
        0 => (EffectiveUpscaler::Off, None),
        1 => (EffectiveUpscaler::Tsr, None),
        3 => {
            if !is_macos && caps.dlss_sr_available {
                (EffectiveUpscaler::DlssSr, None)
            } else if caps.fsr3_loaded {
                (
                    EffectiveUpscaler::Fsr3,
                    Some("DLSS-SR unavailable; falling back to AMD FSR 3".to_string()),
                )
            } else if !is_macos && caps.xess_loaded {
                (
                    EffectiveUpscaler::XeSS,
                    Some("DLSS-SR unavailable; falling back to Intel XeSS".to_string()),
                )
            } else {
                (
                    EffectiveUpscaler::Tsr,
                    Some("DLSS-SR unavailable; falling back to built-in TSR".to_string()),
                )
            }
        }
        2 => {
            if caps.fsr3_loaded {
                (EffectiveUpscaler::Fsr3, None)
            } else if !is_macos && caps.xess_loaded {
                (
                    EffectiveUpscaler::XeSS,
                    Some("FSR 3 unavailable; falling back to Intel XeSS".to_string()),
                )
            } else {
                (
                    EffectiveUpscaler::Tsr,
                    Some("FSR 3 unavailable; falling back to built-in TSR".to_string()),
                )
            }
        }
        4 => {
            if !is_macos && caps.xess_loaded {
                (EffectiveUpscaler::XeSS, None)
            } else if caps.fsr3_loaded {
                (
                    EffectiveUpscaler::Fsr3,
                    Some("XeSS unavailable; falling back to AMD FSR 3".to_string()),
                )
            } else {
                (
                    EffectiveUpscaler::Tsr,
                    Some("XeSS unavailable; falling back to built-in TSR".to_string()),
                )
            }
        }
        _ => (
            EffectiveUpscaler::Tsr,
            Some("Unknown upscaler mode; using TSR".to_string()),
        ),
    }
}

/// Upscaler backend selected for the active viewport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpscalerKind {
    None,
    DlssSr,
    Fsr3,
    XeSS,
}

/// Denoiser backend selected for the active viewport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenoiserKind {
    None,
    DlssRr,
    NrdReblur,
    NrdRelax,
}

/// Frame generation backend selected for the active viewport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameGenKind {
    None,
    DlssFg,
    Fsr3Fg,
}

/// Resolution pair used to detect resize transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ViewportExtent {
    pub render_width: u32,
    pub render_height: u32,
    pub output_width: u32,
    pub output_height: u32,
}

/// Probed runtime availability and NGX capability flags across all vendor SDKs.
#[derive(Debug, Clone, Default)]
pub struct RuntimeSdkCapabilities {
    pub ngx_core_loaded: bool,
    pub ngx_core_path: Option<PathBuf>,
    pub dlss_sr_binary_present: bool,
    pub dlss_rr_binary_present: bool,
    pub dlss_fg_binary_present: bool,

    pub ngx_initialized: bool,
    pub dlss_sr_available: bool,
    pub dlss_rr_available: bool,
    pub dlss_fg_available: bool,
    pub dlss_sr_needs_driver_update: bool,
    pub dlss_fg_needs_driver_update: bool,
    pub dlss_sr_min_driver_major: u32,
    pub dlss_sr_min_driver_minor: u32,

    pub fsr3_loaded: bool,
    pub fsr3_path: Option<PathBuf>,
    pub fsr3_abi: Option<Fsr3AbiKind>,

    pub xess_loaded: bool,
    pub xess_path: Option<PathBuf>,
    pub xess_version: Option<XessVersion>,

    pub nrd_loaded: bool,
    pub nrd_path: Option<PathBuf>,
    pub nrd_version: Option<(u8, u8, u8)>,
}

impl RuntimeSdkCapabilities {
    /// Returns a concise single-line summary for diagnostics and telemetry overlays.
    pub fn summary(&self) -> String {
        format!(
            "NGX(core={}, init={}, SR={}, RR={}, FG={}) | NRD={} | FSR3={} | XeSS={}",
            self.ngx_core_loaded,
            self.ngx_initialized,
            self.dlss_sr_available,
            self.dlss_rr_available,
            self.dlss_fg_available,
            self.nrd_loaded,
            self.fsr3_loaded,
            self.xess_loaded,
        )
    }
}

/// Coordinates runtime discovery, NGX capability querying, and active upscaler/denoiser/frame-gen
/// context lifetimes across viewport resizes and shutdown.
pub struct RuntimeSdkManager {
    pub search_dirs: Vec<PathBuf>,
    pub capabilities: RuntimeSdkCapabilities,
    pub gpu_arch: crate::gpu::GpuArchitecture,
    pub(crate) framegen_failed_mode: u32,

    ngx_lib: Option<Arc<NgxVulkanLibrary>>,
    ngx_ctx: Option<NgxVulkanContext>,

    fsr3_lib: Option<Arc<Fsr3Library>>,
    xess_lib: Option<Arc<XessLibrary>>,
    nrd_lib: Option<Arc<NrdLibrary>>,

    active_dlss_sr: Option<*mut NvSdkNgxHandle>,
    active_dlss_rr: Option<*mut NvSdkNgxHandle>,
    dlss_rr_configuration: Option<(ViewportExtent, NvSdkNgxPerfQuality, i32)>,
    dlss_rr_init_error: Option<String>,
    active_dlss_fg: Option<*mut NvSdkNgxHandle>,
    active_fsr3_upscale: Option<Fsr3UpscaleContext>,
    active_fsr3_fg: Option<fsr3::Fsr3FrameGenContext>,
    active_xess: Option<XessContext>,
    active_nrd: Option<NrdInstance>,
    active_nrd_dispatcher: Option<NrdVulkanDispatcher>,

    pub gbuffer_albedo: Option<(vk::Image, vk::ImageView)>,
    pub gbuffer_albedo_format: vk::Format,
    pub gbuffer_normal: Option<(vk::Image, vk::ImageView)>,
    pub gbuffer_material: Option<(vk::Image, vk::ImageView)>,
    pub prev_color: Option<(vk::Image, vk::ImageView)>,
    /// Dedicated `.a -> .r` swizzled `VkImageView` over `gbuffer_normal` (`gNormalRoughness`) so
    /// DLSS-RR `GBUFFER_ROUGHNESS` reads the actual roughness channel from `RT1.a` instead of
    /// reading metallic from `gbuffer_material` (`RT2.r`).
    pub gbuffer_roughness_swizzled_view: Option<(ash::Device, vk::Image, vk::ImageView)>,
    /// Dedicated `.a -> .r` swizzled `VkImageView` over `nrd_normal_rough` (produced by
    /// `denoiser_prepare.comp` with clean roughness in `[0.02..1.0]` without the `+1.0` water flag).
    pub nrd_roughness_swizzled_view: Option<(ash::Device, vk::Image, vk::ImageView)>,
    pub camera_matrices: SdkCameraMatrices,

    active_upscaler: UpscalerKind,
    active_denoiser: DenoiserKind,
    active_framegen: FrameGenKind,
    upscaler_extent: ViewportExtent,
    denoiser_extent: ViewportExtent,
    framegen_extent: ViewportExtent,
    pending_upscaler_reset: bool,
    pending_denoiser_reset: bool,
    pending_framegen_reset: bool,

    /// Last real NGX init result (`Ok` / error text). `None` means init was never attempted.
    pub last_init_result: Option<String>,
    /// Last real vendor evaluate result. `None` means no evaluate was ever attempted.
    pub last_evaluate_result: Option<String>,
    pub last_denoiser_status: Option<String>,
}

// NGX feature handles are synchronized externally by the Vulkan queue / frame lifecycle.
unsafe impl Send for RuntimeSdkManager {}

/// Wrapper allowing vendor SDK context handles (externally synchronized by RetireQueue) to be
/// deferred across frame slots.
struct SendContext<T>(T);
unsafe impl<T> Send for SendContext<T> {}

impl RuntimeSdkManager {
    /// Probes all vendor dynamic libraries in `search_dirs` and system library paths.
    /// When `libNRD.so` is present, immediately creates and configures the default `NrdInstance`
    /// (`ReblurDiffuseSpecular`, id `1`) so NRD is immediately ready for capability queries and
    /// pipeline creation.
    pub fn probe(search_dirs: Vec<PathBuf>) -> Self {
        let dir_refs: Vec<&Path> = search_dirs.iter().map(PathBuf::as_path).collect();
        let mut caps = RuntimeSdkCapabilities::default();

        let ngx_lib = match NgxVulkanLibrary::load(&dir_refs) {
            Ok(lib) => {
                caps.ngx_core_loaded = true;
                caps.ngx_core_path = Some(lib.loaded_path.clone());
                info!(path = %lib.loaded_path.display(), "Loaded NVIDIA NGX Core runtime");
                Some(Arc::new(lib))
            }
            Err(e) => {
                warn!("NVIDIA NGX Core unavailable: {e}");
                None
            }
        };

        caps.dlss_sr_binary_present =
            has_snippet_binary(&dir_refs, &["libnvidia-ngx-dlss", "nvngx_dlss"]);
        caps.dlss_rr_binary_present = has_snippet_binary(
            &dir_refs,
            &[
                "libnvidia-ngx-dlssd",
                "libnvidia-ngx-builtin-dlssd",
                "nvngx_dlssd",
            ],
        );
        caps.dlss_fg_binary_present =
            has_snippet_binary(&dir_refs, &["libnvidia-ngx-dlssg", "nvngx_dlssg"]);

        let fsr3_lib = match Fsr3Library::load(&dir_refs) {
            Ok(lib) => {
                caps.fsr3_loaded = true;
                caps.fsr3_path = Some(lib.loaded_path.clone());
                caps.fsr3_abi = Some(lib.abi_kind);
                info!(path = %lib.loaded_path.display(), abi = %lib.abi_kind, "Loaded AMD FidelityFX FSR 3 runtime");
                Some(Arc::new(lib))
            }
            Err(e) => {
                info!("AMD FidelityFX FSR 3 runtime not loaded: {e}");
                None
            }
        };

        let xess_lib = match XessLibrary::load(&dir_refs) {
            Ok(lib) => {
                caps.xess_loaded = true;
                caps.xess_path = Some(lib.loaded_path.clone());
                caps.xess_version = lib.version().ok();
                info!(path = %lib.loaded_path.display(), "Loaded Intel XeSS runtime");
                Some(Arc::new(lib))
            }
            Err(e) => {
                info!("Intel XeSS runtime not loaded: {e}");
                None
            }
        };

        let mut active_nrd = None;
        let mut active_denoiser = DenoiserKind::None;
        let nrd_lib = match NrdLibrary::load(&dir_refs) {
            Ok(lib) => {
                caps.nrd_loaded = true;
                caps.nrd_path = Some(lib.loaded_path.clone());
                if let Some(desc) = lib.library_desc() {
                    caps.nrd_version =
                        Some((desc.version_major, desc.version_minor, desc.version_build));
                }
                info!(path = %lib.loaded_path.display(), version = ?caps.nrd_version, "Loaded NVIDIA NRD runtime");
                let arc_lib = Arc::new(lib);
                if let Ok(mut inst) = NrdInstance::create(
                    arc_lib.clone(),
                    &[NrdDenoiserDesc {
                        identifier: 1,
                        denoiser: NrdDenoiser::ReblurDiffuseSpecular,
                    }],
                ) {
                    let _ = inst.set_reblur_settings(1, &NrdReblurSettings::default());
                    active_nrd = Some(inst);
                    active_denoiser = DenoiserKind::NrdReblur;
                }
                Some(arc_lib)
            }
            Err(e) => {
                info!("NVIDIA NRD runtime not loaded: {e}");
                None
            }
        };

        Self {
            search_dirs,
            capabilities: caps,
            gpu_arch: crate::gpu::GpuArchitecture::Unknown,
            framegen_failed_mode: 0,
            ngx_lib,
            ngx_ctx: None,
            fsr3_lib,
            xess_lib,
            nrd_lib,
            active_dlss_sr: None,
            active_dlss_rr: None,
            dlss_rr_configuration: None,
            dlss_rr_init_error: None,
            active_dlss_fg: None,
            active_fsr3_upscale: None,
            active_fsr3_fg: None,
            active_xess: None,
            active_nrd,
            active_nrd_dispatcher: None,
            gbuffer_albedo: None,
            gbuffer_albedo_format: vk::Format::R8G8B8A8_UNORM,
            gbuffer_normal: None,
            gbuffer_material: None,
            prev_color: None,
            gbuffer_roughness_swizzled_view: None,
            nrd_roughness_swizzled_view: None,
            camera_matrices: SdkCameraMatrices::default(),
            active_upscaler: UpscalerKind::None,
            active_denoiser,
            active_framegen: FrameGenKind::None,
            upscaler_extent: ViewportExtent::default(),
            denoiser_extent: ViewportExtent::default(),
            framegen_extent: ViewportExtent::default(),
            pending_upscaler_reset: true,
            pending_denoiser_reset: true,
            pending_framegen_reset: true,
            last_init_result: None,
            last_evaluate_result: None,
            last_denoiser_status: None,
        }
    }

    /// Resolves the best available upscaler for `requested_mode` with automatic fallback
    /// (`DLSS-SR -> FSR3 -> XeSS -> TSR`) and safe macOS handling.
    pub fn resolve_upscaler_fallback(
        &self,
        requested_mode: u32,
    ) -> (EffectiveUpscaler, Option<String>) {
        resolve_upscaler_fallback_for_caps(
            &self.capabilities,
            requested_mode,
            cfg!(target_os = "macos"),
        )
    }

    /// Propagates a temporal history reset (teleport, dimension change, camera cut, or
    /// upscaler/denoiser mode switch) to NRD, DLSS-SR/RR/FG, FSR3, and XeSS.
    pub fn reset_temporal_history(&mut self) {
        self.pending_upscaler_reset = true;
        self.pending_denoiser_reset = true;
        self.pending_framegen_reset = true;
        if let Some(ref mut nrd) = self.active_nrd {
            nrd.request_history_reset();
        }
    }

    /// Returns `(upscaler_reset_pending, denoiser_reset_pending, framegen_reset_pending)`.
    pub fn pending_temporal_resets(&self) -> (bool, bool, bool) {
        (
            self.pending_upscaler_reset,
            self.pending_denoiser_reset,
            self.pending_framegen_reset,
        )
    }

    pub fn clear_dlss_rr_init_error(&mut self) {
        if self.dlss_rr_init_error.take().is_some() {
            self.dlss_rr_configuration = None;
        }
    }

    pub fn report_denoiser_status(&mut self, status: &str) {
        if self.last_denoiser_status.as_deref() == Some(status) {
            return;
        }
        eprintln!("[Hyaloclast] Denoiser: {status}");
        if status.ends_with("using SVGF") {
            crate::crash::push_native_error("WARN", "sdk::denoiser", status);
        }
        self.last_denoiser_status = Some(status.to_owned());
    }

    /// Updates camera projection and view matrices for NRD, FSR3, and DLSS evaluations.
    pub fn set_camera_matrices(&mut self, matrices: SdkCameraMatrices) {
        self.camera_matrices = matrices;
    }

    /// Records the active GPU architecture so DLSS 4 Transformer render presets are enabled on
    /// Ada Lovelace and Blackwell GPUs.
    pub fn set_gpu_arch(&mut self, arch: crate::gpu::GpuArchitecture) {
        self.gpu_arch = arch;
    }

    /// Supplies auxiliary G-Buffer and history image/view pairs consumed by DLSS-RR (`GBUFFER_*`),
    /// NRD (`IN_NORMAL_ROUGHNESS`), and DLSS-FG (`FI_COLOR1`).
    pub fn set_gbuffer_resources(
        &mut self,
        albedo: (vk::Image, vk::ImageView),
        normal: (vk::Image, vk::ImageView),
        material: (vk::Image, vk::ImageView),
        prev_color: (vk::Image, vk::ImageView),
    ) {
        self.gbuffer_albedo = Some(albedo);
        self.gbuffer_normal = Some(normal);
        self.gbuffer_material = Some(material);
        self.prev_color = Some(prev_color);
    }

    /// Variant of `set_gbuffer_resources` that also creates or updates the `.a -> .r` swizzled
    /// `VkImageView` over `normal.0` (`gNormalRoughness`) so DLSS-RR `GBUFFER_ROUGHNESS` reads
    /// the roughness channel from `gNormalRoughness.a` rather than binding `gbuffer_material`.
    #[allow(clippy::too_many_arguments)]
    pub fn set_gbuffer_resources_with_device(
        &mut self,
        device: Option<&ash::Device>,
        albedo: (vk::Image, vk::ImageView),
        albedo_format: vk::Format,
        normal: (vk::Image, vk::ImageView),
        material: (vk::Image, vk::ImageView),
        prev_color: (vk::Image, vk::ImageView),
        retire: Option<&RetireQueue>,
    ) {
        self.set_gbuffer_resources(albedo, normal, material, prev_color);
        self.gbuffer_albedo_format = if albedo_format != vk::Format::UNDEFINED {
            albedo_format
        } else {
            vk::Format::R8G8B8A8_UNORM
        };
        // The swizzled roughness view is an optional optimisation; without a device handle the
        // base resources are still registered honestly (no fabricated view).
        let Some(device) = device else {
            return;
        };
        if normal.0 != vk::Image::null() {
            let needs_recreate = self
                .gbuffer_roughness_swizzled_view
                .as_ref()
                .is_none_or(|(_, img, _)| *img != normal.0);
            if needs_recreate {
                if let Some((old_dev, _, old_view)) = self.gbuffer_roughness_swizzled_view.take()
                    && old_view != vk::ImageView::null()
                {
                    if let Some(rq) = retire {
                        rq.retire_later(move || unsafe {
                            old_dev.destroy_image_view(old_view, None);
                        });
                    } else {
                        unsafe { old_dev.destroy_image_view(old_view, None) };
                    }
                }
                let view_info = vk::ImageViewCreateInfo::default()
                    .image(normal.0)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R16G16B16A16_SFLOAT)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::A,
                        g: vk::ComponentSwizzle::A,
                        b: vk::ComponentSwizzle::A,
                        a: vk::ComponentSwizzle::ONE,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                if let Ok(swizzled) = unsafe { device.create_image_view(&view_info, None) } {
                    self.gbuffer_roughness_swizzled_view =
                        Some((device.clone(), normal.0, swizzled));
                }
            }
        }
    }

    /// Initializes NGX against the active Vulkan device and queries live feature capabilities.
    ///
    /// # Safety
    /// `instance`, `physical_device`, `device`, `gipa`, and `gdpa` must be valid Vulkan handles.
    pub unsafe fn initialize_ngx(
        &mut self,
        instance: vk::Instance,
        physical_device: vk::PhysicalDevice,
        device: vk::Device,
        gipa: vk::PFN_vkGetInstanceProcAddr,
        gdpa: vk::PFN_vkGetDeviceProcAddr,
        app_data_path: &str,
    ) -> Result<(), NgxError> {
        let Some(lib) = self.ngx_lib.clone() else {
            let err = NgxError::LibraryLoad("NGX Core not loaded".to_string());
            self.last_init_result = Some(err.to_string());
            return Err(err);
        };

        let ctx = match unsafe {
            NgxVulkanContext::init_with_snippet_dirs(
                lib.clone(),
                DEFAULT_NGX_APP_ID,
                app_data_path,
                &self.search_dirs,
                instance,
                physical_device,
                device,
                gipa,
                gdpa,
            )
        } {
            Ok(ctx) => ctx,
            Err(err) => {
                self.last_init_result = Some(err.to_string());
                return Err(err);
            }
        };

        let cap_params = ctx.get_capability_parameters()?;
        let p = cap_params.as_ref();

        self.capabilities.ngx_initialized = true;
        self.capabilities.dlss_sr_available =
            p.get_i32(ngx_params::SUPERSAMPLING_AVAILABLE).unwrap_or(0) != 0;
        self.capabilities.dlss_sr_needs_driver_update = p
            .get_i32(ngx_params::SUPERSAMPLING_NEEDS_UPDATED_DRIVER)
            .unwrap_or(0)
            != 0;
        self.capabilities.dlss_sr_min_driver_major = p
            .get_u32(ngx_params::SUPERSAMPLING_MIN_DRIVER_MAJOR)
            .unwrap_or(0);
        self.capabilities.dlss_sr_min_driver_minor = p
            .get_u32(ngx_params::SUPERSAMPLING_MIN_DRIVER_MINOR)
            .unwrap_or(0);

        let fi_avail = p
            .get_i32(ngx_params::FRAME_INTERPOLATION_AVAILABLE)
            .unwrap_or(0)
            != 0;
        let fg_avail = p
            .get_i32(ngx_params::FRAME_GENERATION_AVAILABLE)
            .unwrap_or(0)
            != 0;
        self.capabilities.dlss_fg_available = fi_avail || fg_avail;
        self.capabilities.dlss_fg_needs_driver_update = p
            .get_i32(ngx_params::FRAME_INTERPOLATION_NEEDS_UPDATED_DRIVER)
            .unwrap_or(0)
            != 0;

        let rr_param = p
            .get_i32(ngx_params::SUPERSAMPLING_DENOISING_AVAILABLE)
            .unwrap_or(0)
            != 0;
        let dir_refs: Vec<&Path> = self.search_dirs.iter().map(PathBuf::as_path).collect();
        let rr_req = lib
            .query_feature_requirements(
                instance,
                physical_device,
                NvSdkNgxFeature::RayReconstruction,
                DEFAULT_NGX_APP_ID,
                app_data_path,
                &dir_refs,
            )
            .map(|r| r.feature_supported.is_supported())
            .unwrap_or(false);
        self.capabilities.dlss_rr_available = rr_param || rr_req;

        self.ngx_ctx = Some(ctx);
        self.last_init_result = Some("ok".to_string());
        info!(summary = %self.capabilities.summary(), "NGX capability probe complete");
        Ok(())
    }

    /// Honest, human-readable reason why a vendor feature is unavailable. Uses only real probe
    /// outcomes (library load, symbol resolution, NGX capability query) — never guesses.
    pub fn unavailable_reason(&self, feature: &str) -> String {
        let c = &self.capabilities;
        let ngx_primary = NgxVulkanLibrary::candidate_names()
            .first()
            .copied()
            .unwrap_or("libnvidia-ngx.so.1");
        let fsr3_primary = Fsr3Library::candidate_names().join(" / ");
        let xess_primary = XessLibrary::candidate_names()
            .first()
            .copied()
            .unwrap_or("libxess.so");
        let nrd_primary = NrdLibrary::candidate_names()
            .first()
            .copied()
            .unwrap_or("libNRD.so");
        let ngx_reason = || match self.last_init_result.as_deref() {
            None => match &c.ngx_core_path {
                None => format!("{ngx_primary} not found in system or natives search paths"),
                Some(p) => format!("NGX loaded from {} but Init was not attempted", p.display()),
            },
            Some("ok") => "NGX capability query reported feature unavailable".to_string(),
            Some(r) => format!("NGX init failed: {r}"),
        };
        match feature {
            "rtAs" | "rayQuery" | "rtPipeline" | "bindless" | "reflex" => {
                "Vulkan device feature/extension not supported".to_string()
            }
            "hdr" => {
                "No HDR surface format probed via vkGetPhysicalDeviceSurfaceFormatsKHR".to_string()
            }
            "dlssSr" => {
                if c.dlss_sr_available {
                    "available".into()
                } else {
                    ngx_reason()
                }
            }
            "dlssRr" => {
                if c.dlss_rr_available {
                    "available".into()
                } else {
                    ngx_reason()
                }
            }
            "dlssFg" => {
                if c.dlss_fg_available {
                    "available".into()
                } else {
                    ngx_reason()
                }
            }
            "fsr3" => match &c.fsr3_path {
                None => format!("{fsr3_primary} not found or required ffx* symbols missing"),
                Some(_) if self.active_fsr3_upscale.is_some() => "available".to_string(),
                Some(p) => format!("loaded {} (ready for context creation)", p.display()),
            },
            "xess" => match &c.xess_path {
                None => format!("{xess_primary} not found or required xess* symbols missing"),
                Some(_) if self.active_xess.is_some() => "available".to_string(),
                Some(p) => format!("loaded {} (ready for context creation)", p.display()),
            },
            "nrd" => match &c.nrd_path {
                None => format!("{nrd_primary} not found or required NRD symbols missing"),
                Some(_) if self.active_nrd.is_some() => "available".to_string(),
                Some(p) => format!("loaded {} but denoiser creation failed", p.display()),
            },
            _ => "unknown feature".to_string(),
        }
    }

    /// True only when the feature really resolved and initialised; false otherwise.
    pub fn capability_available(&self, feature: &str) -> bool {
        let c = &self.capabilities;
        match feature {
            "dlssSr" => c.dlss_sr_available,
            "dlssRr" => c.dlss_rr_available,
            "dlssFg" => c.dlss_fg_available,
            "fsr3" => c.fsr3_loaded,
            "xess" => c.xess_loaded,
            "nrd" => self.is_nrd_available(),
            _ => false,
        }
    }

    pub fn library_path(&self, feature: &str) -> Option<&Path> {
        let c = &self.capabilities;
        match feature {
            "dlssSr" | "dlssRr" | "dlssFg" => c.ngx_core_path.as_deref(),
            "fsr3" => c.fsr3_path.as_deref(),
            "xess" => c.xess_path.as_deref(),
            "nrd" => c.nrd_path.as_deref(),
            _ => None,
        }
    }

    pub fn ngx_library(&self) -> Option<&Arc<NgxVulkanLibrary>> {
        self.ngx_lib.as_ref()
    }

    pub fn ngx_context(&self) -> Option<&NgxVulkanContext> {
        self.ngx_ctx.as_ref()
    }

    pub fn fsr3_library(&self) -> Option<&Arc<Fsr3Library>> {
        self.fsr3_lib.as_ref()
    }

    pub fn xess_library(&self) -> Option<&Arc<XessLibrary>> {
        self.xess_lib.as_ref()
    }

    pub fn nrd_library(&self) -> Option<&Arc<NrdLibrary>> {
        self.nrd_lib.as_ref()
    }

    /// True when the NVIDIA NRD runtime was successfully loaded and a denoiser instance is live.
    pub fn is_nrd_available(&self) -> bool {
        self.capabilities.nrd_loaded && self.active_nrd.is_some()
    }

    /// True when a DLSS-SR feature handle is live (NGX initialised and `configure_upscaler` called).
    pub fn is_dlss_sr_available(&self) -> bool {
        self.capabilities.dlss_sr_available && self.active_dlss_sr.is_some()
    }

    pub fn is_dlss_rr_available(&self) -> bool {
        self.capabilities.dlss_rr_available && self.active_dlss_rr.is_some()
    }

    pub fn is_dlss_fg_available(&self) -> bool {
        self.capabilities.dlss_fg_available && self.active_dlss_fg.is_some()
    }

    pub fn is_fsr3_available(&self) -> bool {
        self.capabilities.fsr3_loaded && self.active_fsr3_upscale.is_some()
    }

    pub fn is_xess_available(&self) -> bool {
        self.capabilities.xess_loaded && self.active_xess.is_some()
    }

    pub fn is_fsr3_fg_available(&self) -> bool {
        self.capabilities.fsr3_loaded && self.active_fsr3_fg.is_some()
    }

    pub fn upscaler_extent(&self) -> ViewportExtent {
        self.upscaler_extent
    }

    pub fn denoiser_extent(&self) -> ViewportExtent {
        self.denoiser_extent
    }

    pub fn framegen_extent(&self) -> ViewportExtent {
        self.framegen_extent
    }

    /// Records AMD FidelityFX FSR3 upscaling compute commands into `p.command_buffer` via
    /// `ffxDispatch` (or the legacy `ffxFsr3ContextDispatch` entry point when selected).
    ///
    /// # Safety
    /// `p.command_buffer` must be open and recording; every image/view in `p` must stay alive until
    /// the submitted command buffer completes.
    pub unsafe fn evaluate_fsr3(&mut self, p: &SdkDispatchParams) -> Result<(), String> {
        let ctx = self
            .active_fsr3_upscale
            .as_mut()
            .ok_or_else(|| "FSR3 context not configured".to_string())?;
        let cam = self.camera_matrices;
        let color = FfxApiResource::from_vk_image(
            p.input_image,
            vk::Format::R16G16B16A16_SFLOAT,
            p.width,
            p.height,
            1,
            resource_usage::READ_ONLY,
            resource_state::COMMON | resource_state::COMPUTE_READ,
        );
        let depth_fmt = if p.depth_format != vk::Format::UNDEFINED {
            p.depth_format
        } else {
            vk::Format::R32_SFLOAT
        };
        let depth_usage = if Self::depth_aspect_for_format(depth_fmt) == vk::ImageAspectFlags::DEPTH
        {
            resource_usage::DEPTHTARGET
        } else {
            resource_usage::READ_ONLY
        };
        let depth = FfxApiResource::from_vk_image(
            p.depth_image,
            depth_fmt,
            p.width,
            p.height,
            1,
            depth_usage,
            resource_state::COMMON | resource_state::COMPUTE_READ,
        );
        let motion = FfxApiResource::from_vk_image(
            p.motion_image,
            vk::Format::R16G16_SFLOAT,
            p.width,
            p.height,
            1,
            resource_usage::READ_ONLY,
            resource_state::COMMON | resource_state::COMPUTE_READ,
        );
        let output = FfxApiResource::from_vk_image(
            p.output_image,
            vk::Format::R16G16B16A16_SFLOAT,
            p.out_width,
            p.out_height,
            1,
            resource_usage::UAV,
            resource_state::COMMON | resource_state::UNORDERED_ACCESS,
        );
        let mut dispatch_flags =
            fsr3::upscale_flags::HIGH_DYNAMIC_RANGE | fsr3::upscale_flags::AUTO_EXPOSURE;
        if cam.depth_inverted {
            dispatch_flags |= fsr3::upscale_flags::DEPTH_INVERTED;
        }
        let desc = FfxDispatchDescUpscale {
            header: FfxApiHeader {
                desc_type: fsr3::FFX_API_DISPATCH_DESC_TYPE_UPSCALE,
                p_next: std::ptr::null_mut(),
            },
            command_list: p.command_buffer,
            color,
            depth,
            motion_vectors: motion,
            exposure: FfxApiResource::default(),
            reactive: FfxApiResource::default(),
            transparency_and_composition: FfxApiResource::default(),
            output,
            jitter_offset: FfxApiFloatCoords2D {
                x: p.jitter[0],
                y: p.jitter[1],
            },
            // Convert normalized UV motion delta `(curNdc - prevNdc) * 0.5` into pixel space with Y-down
            motion_vector_scale: FfxApiFloatCoords2D {
                x: p.width.max(1) as f32,
                y: -(p.height.max(1) as f32),
            },
            render_size: FfxApiDimensions2D {
                width: p.width,
                height: p.height,
            },
            upscale_size: FfxApiDimensions2D {
                width: p.out_width,
                height: p.out_height,
            },
            enable_sharpening: true,
            sharpness: 0.35,
            frame_time_delta: (p.frame_time_delta * 1000.0).max(1.0),
            pre_exposure: 1.0,
            reset: p.reset || std::mem::take(&mut self.pending_upscaler_reset),
            camera_near: if cam.depth_inverted {
                cam.camera_far.max(16.0)
            } else {
                cam.camera_near.max(0.01)
            },
            camera_far: if cam.depth_inverted {
                cam.camera_near.max(0.01)
            } else {
                cam.camera_far.max(16.0)
            },
            camera_fov_angle_vertical: cam.camera_fov_y_radians.clamp(0.2, 2.8),
            view_space_to_meters_factor: 1.0,
            flags: dispatch_flags,
        };
        let result = unsafe { ctx.dispatch(&desc) };
        self.last_evaluate_result = Some(match &result {
            Ok(_) => "FSR3: ok".to_string(),
            Err(e) => format!("FSR3: {e}"),
        });
        result.map_err(|e| e.to_string())
    }

    /// Records Intel XeSS upscaling compute commands into `p.command_buffer` via `xessVKExecute`.
    ///
    /// # Safety
    /// `p.command_buffer` must be open and recording; image views must stay alive until completion.
    #[inline]
    fn depth_aspect_for_format(fmt: vk::Format) -> vk::ImageAspectFlags {
        match fmt {
            vk::Format::D16_UNORM
            | vk::Format::X8_D24_UNORM_PACK32
            | vk::Format::D32_SFLOAT
            | vk::Format::D16_UNORM_S8_UINT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT => vk::ImageAspectFlags::DEPTH,
            _ => vk::ImageAspectFlags::COLOR,
        }
    }

    /// # Safety
    /// `p` must reference live images and a command buffer recording on the configured XeSS device.
    pub unsafe fn evaluate_xess(&mut self, p: &SdkDispatchParams) -> Result<(), String> {
        let ctx = self
            .active_xess
            .as_mut()
            .ok_or_else(|| "XeSS context not configured".to_string())?;
        let depth_fmt = if p.depth_format != vk::Format::UNDEFINED {
            p.depth_format
        } else {
            vk::Format::R32_SFLOAT
        };
        let view_info =
            |image: vk::Image, view: vk::ImageView, format: vk::Format, w, h| XessVkImageViewInfo {
                image_view: view,
                image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: Self::depth_aspect_for_format(format),
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                format,
                width: w,
                height: h,
            };
        let params = XessVkExecuteParams {
            color_texture: view_info(
                p.input_image,
                p.input_view,
                vk::Format::R16G16B16A16_SFLOAT,
                p.width,
                p.height,
            ),
            velocity_texture: view_info(
                p.motion_image,
                p.motion_view,
                vk::Format::R16G16_SFLOAT,
                p.width,
                p.height,
            ),
            depth_texture: view_info(p.depth_image, p.depth_view, depth_fmt, p.width, p.height),
            exposure_scale_texture: view_info(
                vk::Image::null(),
                vk::ImageView::null(),
                vk::Format::UNDEFINED,
                1,
                1,
            ),
            responsive_pixel_mask_texture: view_info(
                vk::Image::null(),
                vk::ImageView::null(),
                vk::Format::UNDEFINED,
                1,
                1,
            ),
            output_texture: view_info(
                p.output_image,
                p.output_view,
                vk::Format::R16G16B16A16_SFLOAT,
                p.out_width,
                p.out_height,
            ),
            jitter_offset_x: p.jitter[0],
            jitter_offset_y: p.jitter[1],
            exposure_scale: 1.0,
            reset_history: u32::from(p.reset || std::mem::take(&mut self.pending_upscaler_reset)),
            input_width: p.width,
            input_height: p.height,
            input_color_base: Xess2d::default(),
            input_motion_vector_base: Xess2d::default(),
            input_depth_base: Xess2d::default(),
            input_responsive_mask_base: Xess2d::default(),
            reserved0: Xess2d::default(),
            output_color_base: Xess2d::default(),
        };
        let result = unsafe { ctx.execute(p.command_buffer, &params) };
        self.last_evaluate_result = Some(match &result {
            Ok(_) => "XeSS: ok".to_string(),
            Err(e) => format!("XeSS: {e}"),
        });
        result.map_err(|e| e.to_string())
    }

    /// Records NVIDIA DLSS-SR compute commands into `p.command_buffer` via
    /// `NVSDK_NGX_VULKAN_EvaluateFeature`, passing real colour/depth/motion views and pixel-scaled MV.
    ///
    /// # Safety
    /// NGX must have been initialised and `configure_upscaler(DlssSr, ..)` called first.
    pub unsafe fn evaluate_dlss_sr(&mut self, p: &SdkDispatchParams) -> Result<(), String> {
        let handle = self
            .active_dlss_sr
            .ok_or_else(|| "DLSS-SR feature not created".to_string())?;
        let ctx = self
            .ngx_ctx
            .as_ref()
            .ok_or_else(|| "NGX context not initialized".to_string())?;
        let mut params = ctx.allocate_parameters().map_err(|e| e.to_string())?;
        let subrange = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let depth_subrange = vk::ImageSubresourceRange {
            aspect_mask: Self::depth_aspect_for_format(p.depth_format),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let color = NvSdkNgxResourceVk::from_image_view(
            p.input_view,
            p.input_image,
            subrange,
            vk::Format::R16G16B16A16_SFLOAT,
            p.width,
            p.height,
            false,
        );
        let depth_fmt = if p.depth_format != vk::Format::UNDEFINED {
            p.depth_format
        } else {
            vk::Format::R32_SFLOAT
        };
        let depth = NvSdkNgxResourceVk::from_image_view(
            p.depth_view,
            p.depth_image,
            depth_subrange,
            depth_fmt,
            p.width,
            p.height,
            false,
        );
        let motion = NvSdkNgxResourceVk::from_image_view(
            p.motion_view,
            p.motion_image,
            subrange,
            vk::Format::R16G16_SFLOAT,
            p.width,
            p.height,
            false,
        );
        let output = NvSdkNgxResourceVk::from_image_view(
            p.output_view,
            p.output_image,
            subrange,
            vk::Format::R16G16B16A16_SFLOAT,
            p.out_width,
            p.out_height,
            true,
        );
        {
            let pset = params.as_mut();
            unsafe {
                pset.set_void_ptr(
                    ngx_params::COLOR,
                    &color as *const _ as *mut std::ffi::c_void,
                );
                pset.set_void_ptr(
                    ngx_params::DEPTH,
                    &depth as *const _ as *mut std::ffi::c_void,
                );
                pset.set_void_ptr(
                    ngx_params::MOTION_VECTORS,
                    &motion as *const _ as *mut std::ffi::c_void,
                );
                pset.set_void_ptr(
                    ngx_params::OUTPUT,
                    &output as *const _ as *mut std::ffi::c_void,
                );
            }
            pset.set_f32(ngx_params::JITTER_OFFSET_X, p.jitter[0]);
            pset.set_f32(ngx_params::JITTER_OFFSET_Y, p.jitter[1]);
            pset.set_f32(ngx_params::MV_SCALE_X, p.width.max(1) as f32);
            pset.set_f32(ngx_params::MV_SCALE_Y, -(p.height.max(1) as f32));
            pset.set_f32(ngx_params::FRAME_TIME_DELTA_MS, p.frame_time_delta * 1000.0);
            let effective_reset = p.reset || std::mem::take(&mut self.pending_upscaler_reset);
            pset.set_i32(ngx_params::RESET, i32::from(effective_reset));
        }
        let result = unsafe { ctx.evaluate_feature(p.command_buffer, handle, params.as_ref()) };
        self.last_evaluate_result = Some(match &result {
            Ok(_) => "DLSS-SR: ok".to_string(),
            Err(e) => format!("DLSS-SR: {e}"),
        });
        result.map_err(|e| e.to_string())
    }

    /// Records NVIDIA DLSS-RR (Ray Reconstruction) compute commands via
    /// `NVSDK_NGX_VULKAN_EvaluateFeature`, binding colour, output, depth, motion vectors, and all
    /// required G-Buffer guides (`GBUFFER_ALBEDO`, `GBUFFER_DIFFUSE_ALBEDO`, `GBUFFER_NORMALS`,
    /// `GBUFFER_ROUGHNESS` from `gNormalRoughness.a` swizzled view, `GBUFFER_METALLIC` from
    /// `gMaterialMer.r`, and `RAY_TRACING_HIT_DISTANCE`).
    ///
    /// # Safety
    /// NGX must be initialised and `configure_denoiser(DlssRr, ..)` called first.
    pub unsafe fn evaluate_dlss_rr(&mut self, p: &SdkDispatchParams) -> Result<(), String> {
        unsafe { self.evaluate_dlss_rr_with_retire(p, None, None) }
    }

    /// Variant of [`Self::evaluate_dlss_rr`] that accepts a Vulkan device and [`RetireQueue`] so
    /// the `.a -> .r` swizzled roughness view over `p.nrd_normal_rough` can be cached and safely
    /// retired when the image handle changes.
    ///
    /// # Safety
    /// NGX must be initialised and `configure_denoiser(DlssRr, ..)` called first.
    pub unsafe fn evaluate_dlss_rr_with_retire(
        &mut self,
        p: &SdkDispatchParams,
        device: Option<&ash::Device>,
        retire: Option<&RetireQueue>,
    ) -> Result<(), String> {
        if p.nrd_normal_rough.0 != vk::Image::null() {
            let needs_recreate = self
                .nrd_roughness_swizzled_view
                .as_ref()
                .is_none_or(|(_, img, _)| *img != p.nrd_normal_rough.0);
            if needs_recreate {
                let dev_opt = device.cloned().or_else(|| {
                    self.gbuffer_roughness_swizzled_view
                        .as_ref()
                        .map(|(d, _, _)| d.clone())
                });
                if let Some(dev) = dev_opt {
                    if let Some((old_dev, _, old_view)) = self.nrd_roughness_swizzled_view.take()
                        && old_view != vk::ImageView::null()
                    {
                        if let Some(rq) = retire {
                            rq.retire_later(move || unsafe {
                                old_dev.destroy_image_view(old_view, None);
                            });
                        } else {
                            unsafe { old_dev.destroy_image_view(old_view, None) };
                        }
                    }
                    let view_info = vk::ImageViewCreateInfo::default()
                        .image(p.nrd_normal_rough.0)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(vk::Format::R16G16B16A16_SFLOAT)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::A,
                            g: vk::ComponentSwizzle::A,
                            b: vk::ComponentSwizzle::A,
                            a: vk::ComponentSwizzle::ONE,
                        })
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        });
                    if let Ok(swizzled) = unsafe { dev.create_image_view(&view_info, None) } {
                        self.nrd_roughness_swizzled_view =
                            Some((dev, p.nrd_normal_rough.0, swizzled));
                    }
                }
            }
        }
        let handle = self
            .active_dlss_rr
            .ok_or_else(|| "DLSS-RR feature not created".to_string())?;
        let ctx = self
            .ngx_ctx
            .as_ref()
            .ok_or_else(|| "NGX context not initialized".to_string())?;
        let mut params = ctx.allocate_parameters().map_err(|e| e.to_string())?;
        let subrange = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let depth_subrange = vk::ImageSubresourceRange {
            aspect_mask: Self::depth_aspect_for_format(p.depth_format),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let color = NvSdkNgxResourceVk::from_image_view(
            p.input_view,
            p.input_image,
            subrange,
            vk::Format::R16G16B16A16_SFLOAT,
            p.width,
            p.height,
            false,
        );
        let depth_fmt = if p.depth_format != vk::Format::UNDEFINED {
            p.depth_format
        } else {
            vk::Format::R32_SFLOAT
        };
        let depth = NvSdkNgxResourceVk::from_image_view(
            p.depth_view,
            p.depth_image,
            depth_subrange,
            depth_fmt,
            p.width,
            p.height,
            false,
        );
        let motion = NvSdkNgxResourceVk::from_image_view(
            p.motion_view,
            p.motion_image,
            subrange,
            vk::Format::R16G16_SFLOAT,
            p.width,
            p.height,
            false,
        );
        let (out_img, out_view) = if p.dlss_rr_output_image != vk::Image::null()
            && p.dlss_rr_output_view != vk::ImageView::null()
        {
            (p.dlss_rr_output_image, p.dlss_rr_output_view)
        } else {
            (p.output_image, p.output_view)
        };
        let output = NvSdkNgxResourceVk::from_image_view(
            out_view,
            out_img,
            subrange,
            vk::Format::R16G16B16A16_SFLOAT,
            p.out_width,
            p.out_height,
            true,
        );
        let (alb_img, alb_view) = p.nrd_in_diff;
        let (spec_img, spec_view) = p.nrd_in_spec;
        if alb_view == vk::ImageView::null() || spec_view == vk::ImageView::null() {
            return Err("DLSS-RR albedo guides missing".into());
        }
        // Prefer unpacked [-1..1] normals from denoiser_prepare (`nrd_normal_rough`) when available.
        let (nrm_img, nrm_view) = if p.nrd_normal_rough.0 != vk::Image::null()
            && p.nrd_normal_rough.1 != vk::ImageView::null()
        {
            p.nrd_normal_rough
        } else {
            self.gbuffer_normal.unwrap_or((p.input_image, p.input_view))
        };
        // Prefer `.a -> .r` swizzled view over `p.nrd_normal_rough` (clean roughness in [0.02..1.0]
        // without the +1.0 water flag); otherwise fall back to `gbuffer_normal`'s swizzled view.
        let (rough_img, rough_view) = if p.nrd_normal_rough.0 != vk::Image::null() {
            if let Some((_, img, view)) = self
                .nrd_roughness_swizzled_view
                .as_ref()
                .filter(|(_, img, _)| *img == p.nrd_normal_rough.0)
            {
                (*img, *view)
            } else {
                let (fallback_img, raw_nrm_view) =
                    self.gbuffer_normal.unwrap_or((nrm_img, nrm_view));
                let fallback_view = self
                    .gbuffer_roughness_swizzled_view
                    .as_ref()
                    .map(|(_, _, v)| *v)
                    .unwrap_or(raw_nrm_view);
                (fallback_img, fallback_view)
            }
        } else {
            let (fallback_img, raw_nrm_view) = self.gbuffer_normal.unwrap_or((nrm_img, nrm_view));
            let fallback_view = self
                .gbuffer_roughness_swizzled_view
                .as_ref()
                .map(|(_, _, v)| *v)
                .unwrap_or(raw_nrm_view);
            (fallback_img, fallback_view)
        };

        let gbuf_albedo = NvSdkNgxResourceVk::from_image_view(
            alb_view,
            alb_img,
            subrange,
            vk::Format::R16G16B16A16_SFLOAT,
            p.width,
            p.height,
            false,
        );
        let gbuf_normal = NvSdkNgxResourceVk::from_image_view(
            nrm_view,
            nrm_img,
            subrange,
            vk::Format::R16G16B16A16_SFLOAT,
            p.width,
            p.height,
            false,
        );
        let gbuf_roughness = NvSdkNgxResourceVk::from_image_view(
            rough_view,
            rough_img,
            subrange,
            vk::Format::R16G16B16A16_SFLOAT,
            p.width,
            p.height,
            false,
        );
        let specular_albedo = NvSdkNgxResourceVk::from_image_view(
            spec_view,
            spec_img,
            subrange,
            vk::Format::R16G16B16A16_SFLOAT,
            p.width,
            p.height,
            false,
        );
        let hit_dist_res =
            if p.rt_hit_dist.0 != vk::Image::null() && p.rt_hit_dist.1 != vk::ImageView::null() {
                NvSdkNgxResourceVk::from_image_view(
                    p.rt_hit_dist.1,
                    p.rt_hit_dist.0,
                    subrange,
                    vk::Format::R16_SFLOAT,
                    p.width,
                    p.height,
                    false,
                )
            } else {
                depth
            };
        let world_to_view = SdkCameraMatrices::flatten(self.camera_matrices.world_to_view);
        let view_to_clip = SdkCameraMatrices::flatten(self.camera_matrices.view_to_clip);
        {
            let pset = params.as_mut();
            unsafe {
                pset.set_void_ptr(
                    ngx_params::DLSS_WORLD_TO_VIEW,
                    world_to_view.as_ptr().cast_mut().cast(),
                );
                pset.set_void_ptr(
                    ngx_params::DLSS_VIEW_TO_CLIP,
                    view_to_clip.as_ptr().cast_mut().cast(),
                );
                pset.set_void_ptr(
                    ngx_params::COLOR,
                    &color as *const _ as *mut std::ffi::c_void,
                );
                pset.set_void_ptr(
                    ngx_params::OUTPUT,
                    &output as *const _ as *mut std::ffi::c_void,
                );
                pset.set_void_ptr(
                    ngx_params::DEPTH,
                    &depth as *const _ as *mut std::ffi::c_void,
                );
                pset.set_void_ptr(
                    ngx_params::MOTION_VECTORS,
                    &motion as *const _ as *mut std::ffi::c_void,
                );
                pset.set_void_ptr(
                    ngx_params::GBUFFER_ALBEDO,
                    &gbuf_albedo as *const _ as *mut std::ffi::c_void,
                );
                pset.set_void_ptr(
                    ngx_params::GBUFFER_DIFFUSE_ALBEDO,
                    &gbuf_albedo as *const _ as *mut std::ffi::c_void,
                );
                pset.set_void_ptr(
                    ngx_params::GBUFFER_NORMALS,
                    &gbuf_normal as *const _ as *mut std::ffi::c_void,
                );
                pset.set_void_ptr(
                    ngx_params::GBUFFER_ROUGHNESS,
                    &gbuf_roughness as *const _ as *mut std::ffi::c_void,
                );
                pset.set_void_ptr(
                    ngx_params::GBUFFER_SPECULAR_ALBEDO,
                    &specular_albedo as *const _ as *mut std::ffi::c_void,
                );
                pset.set_void_ptr(
                    ngx_params::RAY_TRACING_HIT_DISTANCE,
                    &hit_dist_res as *const _ as *mut std::ffi::c_void,
                );
            }
            pset.set_u32(ngx_params::DLSS_SUBRECT_WIDTH, p.width);
            pset.set_u32(ngx_params::DLSS_SUBRECT_HEIGHT, p.height);
            pset.set_f32(ngx_params::DLSS_PRE_EXPOSURE, 1.0);
            pset.set_f32(ngx_params::DLSS_EXPOSURE_SCALE, 1.0);
            pset.set_f32(ngx_params::JITTER_OFFSET_X, p.jitter[0]);
            pset.set_f32(ngx_params::JITTER_OFFSET_Y, p.jitter[1]);
            pset.set_f32(ngx_params::MV_SCALE_X, p.width.max(1) as f32);
            pset.set_f32(ngx_params::MV_SCALE_Y, -(p.height.max(1) as f32));
            pset.set_f32(ngx_params::FRAME_TIME_DELTA_MS, p.frame_time_delta * 1000.0);
            let effective_reset = p.reset || std::mem::take(&mut self.pending_denoiser_reset);
            pset.set_i32(ngx_params::RESET, i32::from(effective_reset));
        }
        let result = unsafe { ctx.evaluate_feature(p.command_buffer, handle, params.as_ref()) };
        self.last_evaluate_result = Some(match &result {
            Ok(_) => "DLSS-RR: ok".to_string(),
            Err(e) => format!("DLSS-RR: {e}"),
        });
        result.map_err(|e| e.to_string())
    }

    /// Records NVIDIA DLSS-FG (Frame Generation) compute commands via
    /// `NVSDK_NGX_VULKAN_EvaluateFeature`, binding both current and frame-interpolation (`FI_*`)
    /// colour, depth, motion vector, and output descriptors.
    ///
    /// # Safety
    /// NGX must be initialised and `configure_framegen(DlssFg, ..)` called first.
    pub unsafe fn evaluate_dlss_fg(&mut self, p: &SdkDispatchParams) -> Result<(), String> {
        let handle = self
            .active_dlss_fg
            .ok_or_else(|| "DLSS-FG feature not created".to_string())?;
        let ctx = self
            .ngx_ctx
            .as_ref()
            .ok_or_else(|| "NGX context not initialized".to_string())?;
        let mut params = ctx.allocate_parameters().map_err(|e| e.to_string())?;
        let subrange = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let depth_subrange = vk::ImageSubresourceRange {
            aspect_mask: Self::depth_aspect_for_format(p.depth_format),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let color_fmt = if p.hdr_enabled {
            vk::Format::R16G16B16A16_SFLOAT
        } else {
            vk::Format::R8G8B8A8_UNORM
        };
        let depth_fmt = if p.depth_format != vk::Format::UNDEFINED {
            p.depth_format
        } else {
            vk::Format::R32_SFLOAT
        };
        let color = NvSdkNgxResourceVk::from_image_view(
            p.input_view,
            p.input_image,
            subrange,
            color_fmt,
            p.width,
            p.height,
            false,
        );
        let (prev_img, prev_view) = self.prev_color.unwrap_or((p.input_image, p.input_view));
        let prev_color = NvSdkNgxResourceVk::from_image_view(
            prev_view, prev_img, subrange, color_fmt, p.width, p.height, false,
        );
        let depth = NvSdkNgxResourceVk::from_image_view(
            p.depth_view,
            p.depth_image,
            depth_subrange,
            depth_fmt,
            p.width,
            p.height,
            false,
        );
        let motion = NvSdkNgxResourceVk::from_image_view(
            p.motion_view,
            p.motion_image,
            subrange,
            vk::Format::R16G16_SFLOAT,
            p.width,
            p.height,
            false,
        );
        let output = NvSdkNgxResourceVk::from_image_view(
            p.output_view,
            p.output_image,
            subrange,
            color_fmt,
            p.out_width,
            p.out_height,
            true,
        );
        {
            let pset = params.as_mut();
            unsafe {
                pset.set_void_ptr(
                    ngx_params::COLOR,
                    &color as *const _ as *mut std::ffi::c_void,
                );
                pset.set_void_ptr(
                    ngx_params::OUTPUT,
                    &output as *const _ as *mut std::ffi::c_void,
                );
                pset.set_void_ptr(
                    ngx_params::DEPTH,
                    &depth as *const _ as *mut std::ffi::c_void,
                );
                pset.set_void_ptr(
                    ngx_params::MOTION_VECTORS,
                    &motion as *const _ as *mut std::ffi::c_void,
                );
                pset.set_void_ptr(
                    ngx_params::FI_COLOR1,
                    &prev_color as *const _ as *mut std::ffi::c_void,
                );
                pset.set_void_ptr(
                    ngx_params::FI_OUTPUT1,
                    &output as *const _ as *mut std::ffi::c_void,
                );
                pset.set_void_ptr(
                    ngx_params::FI_MV1,
                    &motion as *const _ as *mut std::ffi::c_void,
                );
                pset.set_void_ptr(
                    ngx_params::FI_DEPTH1,
                    &depth as *const _ as *mut std::ffi::c_void,
                );
            }
            pset.set_f32(ngx_params::JITTER_OFFSET_X, p.jitter[0]);
            pset.set_f32(ngx_params::JITTER_OFFSET_Y, p.jitter[1]);
            pset.set_f32(ngx_params::FRAME_TIME_DELTA_MS, p.frame_time_delta * 1000.0);
            let effective_reset = p.reset || std::mem::take(&mut self.pending_framegen_reset);
            pset.set_i32(ngx_params::RESET, i32::from(effective_reset));
        }
        let result = unsafe { ctx.evaluate_feature(p.command_buffer, handle, params.as_ref()) };
        self.last_evaluate_result = Some(match &result {
            Ok(_) => "DLSS-FG: ok".to_string(),
            Err(e) => format!("DLSS-FG: {e}"),
        });
        result.map_err(|e| e.to_string())
    }

    /// Compiles NRD's embedded SPIR-V compute shaders into real Vulkan compute pipelines so
    /// `evaluate_nrd` can bind and dispatch them onto `p.command_buffer`.
    ///
    /// # Safety
    /// `device` must be a valid Vulkan device handle.
    pub unsafe fn configure_nrd_dispatcher(&mut self, device: &ash::Device) -> Result<(), String> {
        let Some(ref inst) = self.active_nrd else {
            return Err("NRD instance not configured".to_string());
        };
        let inst_desc = inst.instance_desc().map_err(|e| e.to_string())?;
        let lib_desc = self.nrd_lib.as_ref().and_then(|l| l.library_desc());
        let dispatcher = unsafe { NrdVulkanDispatcher::create(device, inst_desc, lib_desc)? };
        self.active_nrd_dispatcher = Some(dispatcher);
        Ok(())
    }

    /// Ensures NRD's GPU `PermanentPool` and `TransientPool` textures and constant buffer are
    /// allocated for `width x height`.
    ///
    /// # Safety
    /// `instance` and `physical_device` must be valid Vulkan handles.
    pub unsafe fn ensure_nrd_pools(
        &mut self,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        width: u32,
        height: u32,
        retire: Option<&RetireQueue>,
    ) {
        if let Some(ref mut disp) = self.active_nrd_dispatcher {
            unsafe {
                disp.ensure_pools(instance, physical_device, width, height, retire);
            }
        }
    }

    /// Drives NVIDIA NRD: applies real `CommonSettings` (with non-zero projection/view matrices,
    /// UV motion vectors and jitter), obtains the library's actual compute dispatch
    /// descriptors via `GetComputeDispatches`, updates descriptor sets/barriers, and records the
    /// NRD compute pipelines onto `p.command_buffer`.
    ///
    /// # Safety
    /// `p.command_buffer` must be open and recording; denoiser must be configured first.
    pub unsafe fn evaluate_nrd(
        &mut self,
        p: &SdkDispatchParams,
        user_layouts: &mut HashMap<vk::Image, vk::ImageLayout>,
    ) -> Result<usize, String> {
        let inst = self
            .active_nrd
            .as_mut()
            .ok_or_else(|| "NRD denoiser not configured".to_string())?;
        let cam = self.camera_matrices;
        let mut common = NrdCommonSettings::default();
        common.view_to_clip_matrix = SdkCameraMatrices::flatten(cam.view_to_clip);
        common.view_to_clip_matrix_prev = SdkCameraMatrices::flatten(cam.view_to_clip_prev);
        common.world_to_view_matrix = SdkCameraMatrices::flatten(cam.world_to_view);
        common.world_to_view_matrix_prev = SdkCameraMatrices::flatten(cam.world_to_view_prev);
        // The G-buffer stores current UV minus previous UV.
        common.motion_vector_scale = [-1.0, -1.0, 0.0];
        common.denoising_range = 65536.0;
        common.resource_size = [
            p.width.min(u16::MAX as u32) as u16,
            p.height.min(u16::MAX as u32) as u16,
        ];
        common.resource_size_prev = common.resource_size;
        common.rect_size = common.resource_size;
        common.rect_size_prev = common.resource_size;
        common.camera_jitter = p.jitter;
        common.camera_jitter_prev = cam.jitter_prev;
        common.time_delta_between_frames = (p.frame_time_delta * 1000.0).max(1.0);
        common.frame_index = cam.frame_index;
        let effective_reset = p.reset
            || std::mem::take(&mut self.pending_denoiser_reset)
            || inst.take_pending_history_reset();
        common.accumulation_mode = if effective_reset {
            nrd::NrdAccumulationMode::ClearAndRestart
        } else {
            nrd::NrdAccumulationMode::Continue
        };
        inst.set_common_settings(&common)
            .map_err(|e| e.to_string())?;

        let pick_tex = |pair: (vk::Image, vk::ImageView), fallback: (vk::Image, vk::ImageView)| {
            if pair.0 != vk::Image::null() && pair.1 != vk::ImageView::null() {
                pair
            } else {
                fallback
            }
        };
        let has_nrd_viewz =
            p.nrd_viewz.0 != vk::Image::null() && p.nrd_viewz.1 != vk::ImageView::null();
        let in_viewz = if has_nrd_viewz {
            p.nrd_viewz
        } else {
            (p.depth_image, p.depth_view)
        };
        let in_viewz_format = if has_nrd_viewz {
            vk::Format::R32_SFLOAT
        } else if p.depth_format != vk::Format::UNDEFINED {
            p.depth_format
        } else {
            vk::Format::R32_SFLOAT
        };
        let user_textures = NrdUserTextureSet {
            in_mv: (p.motion_image, p.motion_view),
            in_normal_roughness: pick_tex(
                p.nrd_normal_rough,
                self.gbuffer_normal.unwrap_or((p.input_image, p.input_view)),
            ),
            in_viewz,
            in_diff_radiance_hitdist: pick_tex(p.nrd_in_diff, (p.input_image, p.input_view)),
            in_spec_radiance_hitdist: pick_tex(p.nrd_in_spec, (p.input_image, p.input_view)),
            out_diff_radiance_hitdist: (p.output_image, p.output_view),
            out_spec_radiance_hitdist: if p.nrd_out_spec.0 != vk::Image::null()
                && p.nrd_out_spec.1 != vk::ImageView::null()
            {
                p.nrd_out_spec
            } else {
                (vk::Image::null(), vk::ImageView::null())
            },
            in_viewz_format,
        };

        let outcome = match inst.get_compute_dispatches(&[1]) {
            Ok(dispatches) => {
                let count = dispatches.len();
                if let Some(ref mut dispatcher) = self.active_nrd_dispatcher {
                    let recorded = unsafe {
                        dispatcher.record_dispatches_with_resources(
                            p.command_buffer,
                            dispatches,
                            &user_textures,
                            user_layouts,
                        )
                    };
                    if recorded == 0 || recorded != count {
                        Err(format!(
                            "NRD dispatcher recorded {recorded} of {count} dispatches"
                        ))
                    } else {
                        Ok(recorded)
                    }
                } else {
                    Err("NRD Vulkan dispatcher not configured".to_string())
                }
            }
            Err(e) => Err(e.to_string()),
        };
        self.last_evaluate_result = Some(match &outcome {
            Ok(n) => format!("NRD: ok ({n} dispatches)"),
            Err(e) => format!("NRD: {e}"),
        });
        outcome
    }

    /// Ensures the active upscaler context matches `(kind, extent)`, tearing down any stale
    /// context on mode switch or viewport resize before creating the replacement.
    ///
    /// # Safety
    /// Must be called when no GPU work referencing previous handles is in flight; `cmd` must be a
    /// recording `VkCommandBuffer` if NGX creation is invoked.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn configure_upscaler(
        &mut self,
        kind: UpscalerKind,
        extent: ViewportExtent,
        cmd: vk::CommandBuffer,
        instance: vk::Instance,
        physical_device: vk::PhysicalDevice,
        device: vk::Device,
        gdpa: vk::PFN_vkGetDeviceProcAddr,
        dlss_quality: NvSdkNgxPerfQuality,
        dlss_flags: i32,
        retire: Option<&RetireQueue>,
        ash_device: Option<&ash::Device>,
    ) -> Result<(), String> {
        if self.active_upscaler == kind && self.upscaler_extent == extent {
            return Ok(());
        }

        unsafe { self.release_upscaler_contexts(retire, ash_device) };
        self.active_upscaler = UpscalerKind::None;

        match kind {
            UpscalerKind::None => {}
            UpscalerKind::DlssSr => {
                let ctx = self
                    .ngx_ctx
                    .as_ref()
                    .ok_or_else(|| "NGX context not initialized".to_string())?;
                let mut params = ctx.allocate_parameters().map_err(|e| e.to_string())?;
                let effective_flags = if dlss_flags != 0 {
                    dlss_flags
                } else {
                    ngx::dlss_flags::for_pipeline(!self.camera_matrices.depth_inverted)
                };
                if self.gpu_arch.is_ada_or_newer() {
                    ngx::apply_dlss4_transformer_preset_hints(
                        params.as_mut(),
                        ngx::dlss_render_preset::PRESET_K,
                    );
                }
                let handle = unsafe {
                    ctx.create_dlss_sr(
                        cmd,
                        params.as_mut(),
                        extent.render_width,
                        extent.render_height,
                        extent.output_width,
                        extent.output_height,
                        dlss_quality,
                        effective_flags,
                    )
                }
                .map_err(|e| e.to_string())?;
                self.active_dlss_sr = Some(handle);
            }
            UpscalerKind::Fsr3 => {
                let lib = self
                    .fsr3_lib
                    .clone()
                    .ok_or_else(|| "FSR3 library not loaded".to_string())?;
                let mut flags =
                    fsr3::upscale_flags::HIGH_DYNAMIC_RANGE | fsr3::upscale_flags::AUTO_EXPOSURE;
                if self.camera_matrices.depth_inverted {
                    flags |= fsr3::upscale_flags::DEPTH_INVERTED;
                }
                let ctx = unsafe {
                    Fsr3UpscaleContext::create(
                        lib,
                        device,
                        physical_device,
                        gdpa,
                        FfxApiDimensions2D {
                            width: extent.render_width,
                            height: extent.render_height,
                        },
                        FfxApiDimensions2D {
                            width: extent.output_width,
                            height: extent.output_height,
                        },
                        flags,
                    )
                }
                .map_err(|e| e.to_string())?;
                self.active_fsr3_upscale = Some(ctx);
            }
            UpscalerKind::XeSS => {
                let lib = self
                    .xess_lib
                    .clone()
                    .ok_or_else(|| "XeSS library not loaded".to_string())?;
                let mut ctx =
                    unsafe { XessContext::create(lib, instance, physical_device, device) }
                        .map_err(|e| e.to_string())?;
                let mut xess_flags = xess::init_flags::HIGH_RES_MV
                    | xess::init_flags::USE_NDC_VELOCITY
                    | xess::init_flags::ENABLE_AUTOEXPOSURE;
                if self.camera_matrices.depth_inverted {
                    xess_flags |= xess::init_flags::INVERTED_DEPTH;
                }
                let xess_quality = match dlss_quality {
                    NvSdkNgxPerfQuality::UltraPerformance => XessQualitySettings::UltraPerformance,
                    NvSdkNgxPerfQuality::MaxPerf => XessQualitySettings::Performance,
                    NvSdkNgxPerfQuality::Balanced => XessQualitySettings::Balanced,
                    NvSdkNgxPerfQuality::MaxQuality => XessQualitySettings::Quality,
                    NvSdkNgxPerfQuality::UltraQuality => XessQualitySettings::UltraQuality,
                    NvSdkNgxPerfQuality::Dlaa => XessQualitySettings::AntiAliasing,
                };
                let init_params = XessVkInitParams {
                    output_resolution: xess::Xess2d {
                        x: extent.output_width,
                        y: extent.output_height,
                    },
                    quality_setting: xess_quality,
                    init_flags: xess_flags,
                    ..XessVkInitParams::default()
                };
                ctx.init(&init_params).map_err(|e| e.to_string())?;
                self.active_xess = Some(ctx);
            }
        }

        self.active_upscaler = kind;
        self.upscaler_extent = extent;
        self.pending_upscaler_reset = true;
        Ok(())
    }

    /// Ensures the active denoiser context matches `kind`.
    ///
    /// # Safety
    /// Previous GPU work referencing an existing DLSS-RR handle must be complete before switching.
    pub unsafe fn configure_denoiser(
        &mut self,
        kind: DenoiserKind,
        extent: ViewportExtent,
        cmd: vk::CommandBuffer,
        dlss_quality: NvSdkNgxPerfQuality,
        dlss_flags: i32,
        retire: Option<&RetireQueue>,
    ) -> Result<(), String> {
        let rr_configuration = (extent, dlss_quality, dlss_flags);
        if kind == DenoiserKind::DlssRr
            && self.dlss_rr_configuration == Some(rr_configuration)
            && let Some(error) = &self.dlss_rr_init_error
        {
            return Err(error.clone());
        }
        if self.active_denoiser == kind
            && self.denoiser_extent == extent
            && (kind != DenoiserKind::DlssRr
                || self.dlss_rr_configuration == Some(rr_configuration))
            && (!matches!(kind, DenoiserKind::NrdReblur | DenoiserKind::NrdRelax)
                || self.active_nrd.is_some())
        {
            return Ok(());
        }

        unsafe { self.release_denoiser_contexts(retire) };
        self.active_denoiser = DenoiserKind::None;

        match kind {
            DenoiserKind::None => {}
            DenoiserKind::DlssRr => {
                let result = (|| {
                    let ctx = self
                        .ngx_ctx
                        .as_ref()
                        .ok_or_else(|| "NGX context not initialized".to_string())?;
                    let mut params = ctx.allocate_parameters().map_err(|e| e.to_string())?;
                    let effective_flags = if dlss_flags != 0 {
                        dlss_flags
                    } else {
                        ngx::dlss_flags::for_pipeline(!self.camera_matrices.depth_inverted)
                    };
                    if self.gpu_arch.is_ada_or_newer() {
                        ngx::apply_dlss4_transformer_preset_hints(
                            params.as_mut(),
                            ngx::dlss_render_preset::PRESET_K,
                        );
                    }
                    let handle = unsafe {
                        ctx.create_dlss_rr(
                            cmd,
                            params.as_mut(),
                            extent.render_width,
                            extent.render_height,
                            extent.output_width,
                            extent.output_height,
                            dlss_quality,
                            effective_flags,
                        )
                    }
                    .map_err(|e| e.to_string())?;
                    Ok::<_, String>(handle)
                })();
                self.dlss_rr_configuration = Some(rr_configuration);
                match result {
                    Ok(handle) => self.active_dlss_rr = Some(handle),
                    Err(error) => {
                        self.dlss_rr_init_error = Some(error.clone());
                        return Err(error);
                    }
                }
            }
            DenoiserKind::NrdReblur => {
                let lib = self
                    .nrd_lib
                    .clone()
                    .ok_or_else(|| "NRD library not loaded".to_string())?;
                let mut inst = NrdInstance::create(
                    lib,
                    &[NrdDenoiserDesc {
                        identifier: 1,
                        denoiser: NrdDenoiser::ReblurDiffuseSpecular,
                    }],
                )
                .map_err(|e| e.to_string())?;
                inst.set_reblur_settings(1, &NrdReblurSettings::default())
                    .map_err(|e| e.to_string())?;
                self.active_nrd = Some(inst);
            }
            DenoiserKind::NrdRelax => {
                let lib = self
                    .nrd_lib
                    .clone()
                    .ok_or_else(|| "NRD library not loaded".to_string())?;
                let mut inst = NrdInstance::create(
                    lib,
                    &[NrdDenoiserDesc {
                        identifier: 1,
                        denoiser: NrdDenoiser::RelaxDiffuseSpecular,
                    }],
                )
                .map_err(|e| e.to_string())?;
                inst.set_relax_settings(1, &NrdRelaxSettings::default())
                    .map_err(|e| e.to_string())?;
                self.active_nrd = Some(inst);
            }
        }

        self.active_denoiser = kind;
        self.denoiser_extent = extent;
        self.pending_denoiser_reset = true;
        Ok(())
    }

    /// Ensures the active frame-generation context matches `(kind, extent)`.
    ///
    /// # Safety
    /// Previous GPU work referencing an existing DLSS-FG handle must be complete before switching.
    /// Configures FSR3 Frame Generation with explicit Vulkan device handles.
    ///
    /// # Safety
    /// Vulkan handles must remain valid while the FSR3-FG context is active.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn configure_fsr3_framegen(
        &mut self,
        extent: ViewportExtent,
        device: vk::Device,
        physical_device: vk::PhysicalDevice,
        gdpa: vk::PFN_vkGetDeviceProcAddr,
        backbuffer_format: vk::Format,
        retire: Option<&RetireQueue>,
        ash_device: Option<&ash::Device>,
    ) -> Result<(), String> {
        if self.active_framegen == FrameGenKind::Fsr3Fg
            && self.framegen_extent == extent
            && self.active_fsr3_fg.is_some()
        {
            return Ok(());
        }
        unsafe { self.release_framegen_contexts(retire, ash_device) };
        self.active_framegen = FrameGenKind::None;
        let lib = self
            .fsr3_lib
            .clone()
            .ok_or_else(|| "FSR3 library not loaded".to_string())?;
        let ffx_fmt = match backbuffer_format {
            vk::Format::R16G16B16A16_SFLOAT => fsr3::FfxApiSurfaceFormat::R16G16B16A16Float as u32,
            vk::Format::A2B10G10R10_UNORM_PACK32 => {
                fsr3::FfxApiSurfaceFormat::R10G10B10A2Unorm as u32
            }
            _ => fsr3::FfxApiSurfaceFormat::R8G8B8A8Unorm as u32,
        };
        let ctx = unsafe {
            fsr3::Fsr3FrameGenContext::create(
                lib,
                device,
                physical_device,
                gdpa,
                fsr3::FfxApiDimensions2D {
                    width: extent.output_width.max(1),
                    height: extent.output_height.max(1),
                },
                ffx_fmt,
                0,
            )
        }
        .map_err(|e| e.to_string())?;
        self.active_fsr3_fg = Some(ctx);
        self.active_framegen = FrameGenKind::Fsr3Fg;
        self.framegen_extent = extent;
        self.pending_framegen_reset = true;
        Ok(())
    }

    /// Evaluates AMD FSR 3.1+ Frame Generation into `p.output_image`.
    ///
    /// # Safety
    /// `p.command_buffer` must be recording and `configure_fsr3_framegen` called first.
    pub unsafe fn evaluate_fsr3_fg(&mut self, p: &SdkDispatchParams) -> Result<(), String> {
        let fg = self
            .active_fsr3_fg
            .as_mut()
            .ok_or_else(|| "FSR3-FG context not configured".to_string())?;
        let make_res =
            |img: vk::Image, w: u32, h: u32, fmt: u32, state: u32| fsr3::FfxApiResource {
                resource: img.as_raw() as usize as *mut std::ffi::c_void,
                description: fsr3::FfxApiResourceDescription {
                    res_type: fsr3::FfxApiResourceType::Texture2D as u32,
                    format: fmt,
                    width_or_size: w.max(1),
                    height_or_stride: h.max(1),
                    depth_or_alignment: 1,
                    mip_count: 1,
                    flags: 0,
                    usage: 0,
                },
                state,
            };
        let desc = fsr3::FfxDispatchDescFrameGeneration {
            header: fsr3::FfxApiHeader {
                desc_type: fsr3::FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION,
                p_next: std::ptr::null_mut(),
            },
            command_list: p.command_buffer,
            present_color: make_res(
                p.input_image,
                p.width,
                p.height,
                if p.hdr_enabled {
                    fsr3::FfxApiSurfaceFormat::R16G16B16A16Float as u32
                } else {
                    fsr3::FfxApiSurfaceFormat::R8G8B8A8Unorm as u32
                },
                fsr3::resource_state::COMPUTE_READ,
            ),
            outputs: [
                make_res(
                    p.output_image,
                    p.width,
                    p.height,
                    if p.hdr_enabled {
                        fsr3::FfxApiSurfaceFormat::R16G16B16A16Float as u32
                    } else {
                        fsr3::FfxApiSurfaceFormat::R8G8B8A8Unorm as u32
                    },
                    fsr3::resource_state::UNORDERED_ACCESS,
                ),
                fsr3::FfxApiResource::default(),
                fsr3::FfxApiResource::default(),
                fsr3::FfxApiResource::default(),
            ],
            num_generated_frames: 1,
            reset: p.reset || std::mem::take(&mut self.pending_framegen_reset),
            backbuffer_transfer_function: 0,
            min_max_luminance: [0.0, 1000.0],
            generation_rect: fsr3::FfxApiRect2D {
                left: 0,
                top: 0,
                width: p.width as i32,
                height: p.height as i32,
            },
            frame_id: self.camera_matrices.frame_index as u64,
        };
        let res = unsafe { fg.dispatch(&desc) }.map_err(|e| e.to_string());
        self.last_evaluate_result = Some(match &res {
            Ok(()) => "FSR3-FG: ok".to_string(),
            Err(e) => format!("FSR3-FG: {e}"),
        });
        res
    }

    /// # Safety
    /// Supplied Vulkan objects must remain live, and the device must be idle while contexts are reconfigured.
    pub unsafe fn configure_framegen(
        &mut self,
        kind: FrameGenKind,
        extent: ViewportExtent,
        cmd: vk::CommandBuffer,
        retire: Option<&RetireQueue>,
        ash_device: Option<&ash::Device>,
    ) -> Result<(), String> {
        if self.active_framegen == kind && self.framegen_extent == extent {
            return Ok(());
        }

        unsafe { self.release_framegen_contexts(retire, ash_device) };
        self.active_framegen = FrameGenKind::None;

        if kind == FrameGenKind::DlssFg {
            let ctx = self
                .ngx_ctx
                .as_ref()
                .ok_or_else(|| "NGX context not initialized".to_string())?;
            let mut params = ctx.allocate_parameters().map_err(|e| e.to_string())?;
            let handle = unsafe {
                ctx.create_dlss_fg(
                    cmd,
                    params.as_mut(),
                    extent.output_width,
                    extent.output_height,
                )
            }
            .map_err(|e| e.to_string())?;
            self.active_dlss_fg = Some(handle);
        }

        self.active_framegen = kind;
        self.framegen_extent = extent;
        self.pending_framegen_reset = true;
        Ok(())
    }

    unsafe fn release_upscaler_contexts(
        &mut self,
        retire: Option<&RetireQueue>,
        ash_device: Option<&ash::Device>,
    ) {
        let dlss_sr = self.active_dlss_sr.take();
        let fsr3 = self.active_fsr3_upscale.take();
        let xess = self.active_xess.take();
        if dlss_sr.is_none() && fsr3.is_none() && xess.is_none() {
            return;
        }
        if let Some(queue) = retire {
            if let Some(handle) = dlss_sr {
                if let Some(lib) = self.ngx_lib.clone() {
                    let address = handle as usize;
                    queue.retire_later(move || unsafe {
                        (lib.release_feature)(address as *mut NvSdkNgxHandle);
                    });
                } else if let Some(ctx) = &self.ngx_ctx {
                    let _ = unsafe { ctx.release_feature(handle) };
                }
            }
            if let Some(fsr3_ctx) = fsr3 {
                let wrapped = SendContext(fsr3_ctx);
                queue.retire_later(move || drop(wrapped));
            }
            if let Some(xess_ctx) = xess {
                let wrapped = SendContext(xess_ctx);
                queue.retire_later(move || drop(wrapped));
            }
        } else {
            if let Some(dev) = ash_device {
                crate::gpu::with_queue_lock(|| unsafe {
                    let _ = dev.device_wait_idle();
                });
            }
            if let (Some(ctx), Some(handle)) = (&self.ngx_ctx, dlss_sr) {
                let _ = unsafe { ctx.release_feature(handle) };
            }
            drop(fsr3);
            drop(xess);
        }
    }

    unsafe fn release_framegen_contexts(
        &mut self,
        retire: Option<&RetireQueue>,
        ash_device: Option<&ash::Device>,
    ) {
        let dlss_fg = self.active_dlss_fg.take();
        let fsr3_fg = self.active_fsr3_fg.take();
        if dlss_fg.is_none() && fsr3_fg.is_none() {
            return;
        }
        if let Some(queue) = retire {
            if let Some(handle) = dlss_fg {
                if let Some(lib) = self.ngx_lib.clone() {
                    let address = handle as usize;
                    queue.retire_later(move || unsafe {
                        (lib.release_feature)(address as *mut NvSdkNgxHandle);
                    });
                } else if let Some(ctx) = &self.ngx_ctx {
                    let _ = unsafe { ctx.release_feature(handle) };
                }
            }
            if let Some(fg_ctx) = fsr3_fg {
                let wrapped = SendContext(fg_ctx);
                queue.retire_later(move || drop(wrapped));
            }
        } else {
            if let Some(dev) = ash_device {
                crate::gpu::with_queue_lock(|| unsafe {
                    let _ = dev.device_wait_idle();
                });
            }
            if let (Some(ctx), Some(handle)) = (&self.ngx_ctx, dlss_fg) {
                let _ = unsafe { ctx.release_feature(handle) };
            }
            drop(fsr3_fg);
        }
    }

    /// # Safety
    /// No in-flight command may reference a context unless `retire` defers its resource destruction.
    pub unsafe fn release_denoiser_contexts(&mut self, retire: Option<&RetireQueue>) {
        self.dlss_rr_configuration = None;
        self.dlss_rr_init_error = None;
        if let Some(handle) = self.active_dlss_rr.take() {
            if let (Some(queue), Some(lib)) = (retire, self.ngx_lib.clone()) {
                let address = handle as usize;
                queue.retire_later(move || unsafe {
                    (lib.release_feature)(address as *mut NvSdkNgxHandle);
                });
            } else if let Some(ctx) = &self.ngx_ctx {
                let _ = unsafe { ctx.release_feature(handle) };
            }
        }
        if let Some(mut disp) = self.active_nrd_dispatcher.take() {
            unsafe { disp.destroy_with_retire(retire) };
        }
        self.active_nrd = None;
    }

    /// Releases all active SDK contexts and shuts down NGX cleanly.
    ///
    /// # Safety
    /// Device must be idle before calling teardown.
    pub unsafe fn teardown(&mut self) {
        if let Some((dev, _, view)) = self.gbuffer_roughness_swizzled_view.take()
            && view != vk::ImageView::null()
        {
            unsafe { dev.destroy_image_view(view, None) };
        }
        if let Some((dev, _, view)) = self.nrd_roughness_swizzled_view.take()
            && view != vk::ImageView::null()
        {
            unsafe { dev.destroy_image_view(view, None) };
        }
        unsafe { self.release_upscaler_contexts(None, None) };
        unsafe { self.release_denoiser_contexts(None) };
        unsafe { self.release_framegen_contexts(None, None) };
        if let Some(mut ctx) = self.ngx_ctx.take() {
            let _ = ctx.shutdown();
        }
        self.active_upscaler = UpscalerKind::None;
        self.active_denoiser = DenoiserKind::None;
        self.active_framegen = FrameGenKind::None;
        self.upscaler_extent = ViewportExtent::default();
        self.denoiser_extent = ViewportExtent::default();
        self.framegen_extent = ViewportExtent::default();
        self.capabilities.ngx_initialized = false;
    }
}

impl Drop for RuntimeSdkManager {
    fn drop(&mut self) {
        unsafe { self.teardown() };
    }
}

/// Returns Windows system and NVIDIA DriverStore directories where driver-installed NGX binaries live.
pub(crate) fn windows_system_driver_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let Some(sys_root) = std::env::var_os("SystemRoot") else {
        return dirs;
    };
    let sys32 = PathBuf::from(sys_root).join("System32");
    dirs.push(sys32.clone());
    let repo = sys32.join("DriverStore").join("FileRepository");
    if let Ok(entries) = std::fs::read_dir(repo) {
        let mut nv_dirs: Vec<(std::time::SystemTime, PathBuf)> = entries
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_ascii_lowercase();
                if name.starts_with("nv_dispi.inf_amd64_") || name.starts_with("nvmdi.inf_amd64_") {
                    let modified = e
                        .metadata()
                        .and_then(|m| m.modified())
                        .unwrap_or(std::time::UNIX_EPOCH);
                    Some((modified, e.path()))
                } else {
                    None
                }
            })
            .collect();
        nv_dirs.sort_by_key(|entry| std::cmp::Reverse(entry.0));
        for (_, path) in nv_dirs {
            dirs.push(path);
        }
    }
    dirs
}

fn has_snippet_binary(search_dirs: &[&Path], prefixes: &[&str]) -> bool {
    let mut dirs: Vec<PathBuf> = search_dirs.iter().map(|p| p.to_path_buf()).collect();
    for sys_dir in [
        "/usr/lib",
        "/usr/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/local/lib",
        "/opt/homebrew/lib",
    ] {
        dirs.push(PathBuf::from(sys_dir));
    }
    dirs.extend(windows_system_driver_dirs());

    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_lower = name.to_string_lossy().to_ascii_lowercase();
            if prefixes.iter().any(|prefix| {
                name_lower.starts_with(&prefix.to_ascii_lowercase())
                    && (name_lower.contains(".so")
                        || name_lower.ends_with(".dll")
                        || name_lower.ends_with(".dylib"))
            }) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_runtime_sdks_and_staged_natives() {
        let natives_dir = std::env::var_os("VS_NATIVES_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("natives"));
        let dirs = if natives_dir.exists() {
            vec![natives_dir.clone()]
        } else {
            Vec::new()
        };
        let mgr = RuntimeSdkManager::probe(dirs);
        let summary = mgr.capabilities.summary();
        assert!(summary.contains("NGX("));

        if natives_dir.join("libnvidia-ngx-dlss.so.310.9.16").exists() {
            assert!(mgr.capabilities.dlss_sr_binary_present);
            assert!(mgr.capabilities.dlss_rr_binary_present);
            assert!(mgr.capabilities.dlss_fg_binary_present);
        }
        if natives_dir.join("libNRD.so").exists() {
            assert!(mgr.capabilities.nrd_loaded);
            assert!(mgr.is_nrd_available());
            assert_eq!(mgr.unavailable_reason("nrd"), "available");
            assert_eq!(mgr.capabilities.nrd_version, Some((4, 18, 0)));
        }
    }
}
