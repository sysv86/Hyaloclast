// SPDX-License-Identifier: GPL-3.0-only
// Additional permissions under GPLv3 Section 7 apply to optional proprietary vendor SDK
// runtime loading; see LICENSE-EXCEPTION.md in the repository root.

//! Clean-room AMD FSR 4.1.2 / FidelityFX SDK 2.4.0 Vulkan contract simulator.
//!
//! Enforces SDK 2.4.0 lifecycle contracts without physical AMD hardware:
//! - Backend descriptor validation (`0x0200_0003` Vulkan SDK 2.4.0 vs `0x0100_0003` Xbox trap)
//! - Version discovery (`GET_VERSIONS = 4`), pinning (`OVERRIDE_VERSION = 5`), provider query (`6`)
//! - 4-network FSR 4.1.2 device selection (`FSR4` FP8, `FSR4-i8`, `FSR4-i8-2`, `FSR4L-i8`, `FSR 3.1.6`)
//! - 15-frame maximum in-flight cap per context
//! - Live `VkImageLayout` verification against `MockIcdState` (`SHADER_READ_ONLY_OPTIMAL` inputs, `GENERAL` output)
//! - Dispatch parameter invariants (jitter, motion vector scale, frame delta, reverse-Z depth, FOV)

use crate::mock_icd::{
    MockIcdGuard, MockIcdState, mock_vk_get_device_proc_addr, reclaim_mock_physical_device,
    validate_pnext_chain,
};
use crate::profiles::{Fsr4Network, GpuProfile};
use ash::vk::{self, Handle};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use std::ptr;
use std::sync::Arc;
use vulkanshaders_core::sdk::fsr3::{
    FFX_API_RETURN_ERROR_PARAMETER, FFX_API_RETURN_ERROR_RUNTIME_ERROR,
    FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE, FFX_API_RETURN_NO_PROVIDER, FFX_API_RETURN_OK,
    FfxApiDimensions2D, FfxApiFloatCoords2D, FfxApiHeader, FfxApiResource, FfxCreateBackendVKDesc,
    FfxVkDeviceContext, resource_state, resource_usage, upscale_flags,
};

/// FidelityFX SDK 2.4.0 backend IDs (`ffx_api.h`).
pub const FFX_SDK24_BACKEND_ID_DX12: u64 = 0x0000_0000;
pub const FFX_SDK24_BACKEND_ID_XBOX: u64 = 0x0100_0000;
pub const FFX_SDK24_BACKEND_ID_VK: u64 = 0x0200_0000;

/// FidelityFX SDK 2.4.0 Vulkan backend creation descriptor type (`BACKEND_ID_VK | 0x03`).
pub const FFX_SDK24_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK: u64 = FFX_SDK24_BACKEND_ID_VK | 0x03;

/// FidelityFX SDK 2.4.0 core query/override descriptor types (`ffx_api.h`).
pub const FFX_API_QUERY_DESC_TYPE_GET_VERSIONS: u64 = 4;
pub const FFX_API_DESC_TYPE_OVERRIDE_VERSION: u64 = 5;
pub const FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION: u64 = 6;
pub const FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG: u64 = 7;

/// FidelityFX SDK 2.4.0 upscaler descriptor types (`ffx_upscale.h`).
pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE: u64 = 0x0001_0000;
pub const FFX_API_DISPATCH_DESC_TYPE_UPSCALE: u64 = 0x0001_0001;
pub const FFX_API_CONFIGURE_DESC_TYPE_UPSCALE_KEYVALUE: u64 = 0x0001_0007;
pub const FFX_API_QUERY_DESC_TYPE_UPSCALE_GET_RESOURCE_REQUIREMENTS: u64 = 0x0001_000A;
pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION: u64 = 0x0001_000C;

/// Maximum frames allowed in flight concurrently on a single FSR 4.1.2 context.
pub const FSR4_MAX_IN_FLIGHT_FRAMES: usize = 15;

/// Platform-correct `wchar_t` for `ffxApiMessage` callbacks (`u16` on Windows, `u32` on Unix).
#[cfg(windows)]
pub type FfxWchar = u16;
#[cfg(not(windows))]
pub type FfxWchar = u32;

/// Clean-room SDK 2.4.0 `ffxOverrideVersion` struct (`type = 5`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FfxOverrideVersionDesc {
    pub header: FfxApiHeader,
    pub version_id: u64,
}

/// Clean-room SDK 2.4.0 `ffxCreateContextDescUpscaleVersion` struct.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FfxCreateContextDescUpscaleVersion {
    pub header: FfxApiHeader,
    pub version: u64,
}

/// Clean-room SDK 2.4.0 `ffxDispatchDescUpscale` using `u8` booleans (never raw Rust `bool`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FfxSdk24DispatchDescUpscale {
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
    pub enable_sharpening: u8,
    pub sharpness: f32,
    pub frame_time_delta: f32,
    pub pre_exposure: f32,
    pub reset: u8,
    pub camera_near: f32,
    pub camera_far: f32,
    pub camera_fov_angle_vertical: f32,
    pub view_space_to_meters_factor: f32,
    pub flags: u32,
}

/// Active simulated FSR 4.1.2 / 3.1.6 context state.
pub struct SimulatedFsrContext {
    pub selected_network: Fsr4Network,
    pub max_render_size: FfxApiDimensions2D,
    pub max_upscale_size: FfxApiDimensions2D,
    pub flags: u32,
    pub last_render_size: Option<FfxApiDimensions2D>,
    pub in_flight_frames: VecDeque<u64>,
    pub total_dispatches: u64,
}

/// Clean-room stateful FidelityFX SDK 2.4.0 Vulkan provider simulator.
pub struct MockFfxProvider {
    icd: Arc<MockIcdState>,
    override_network: Option<String>,
    context: Option<SimulatedFsrContext>,
}

impl MockFfxProvider {
    pub fn new(icd: Arc<MockIcdState>, override_network: Option<String>) -> Self {
        Self {
            icd,
            override_network,
            context: None,
        }
    }

    /// Simulates `ffxQuery(GET_VERSIONS)` returning available provider version IDs and labels.
    pub fn query_versions(&self) -> Vec<(u64, String)> {
        let mut versions = Vec::new();
        if let Ok(net) = self.icd.profile.select_fsr4_network(self.override_network.as_deref())
            && net.is_ml_fsr4()
        {
            versions.push((
                net.version_id(),
                format!("4.1.2 ({})", net.amd_name()),
            ));
        }
        versions.push((Fsr4Network::Fsr316Fallback.version_id(), "3.1.6".to_string()));
        versions
    }

    /// Simulates `ffxCreateContext` with strict SDK 2.4.0 `pNext` chain and backend checks.
    ///
    /// # Safety
    /// `head` and every chained `p_next` pointer must point to valid `FfxApiHeader`-prefixed structs.
    pub unsafe fn create_context(
        &mut self,
        head: *const FfxApiHeader,
        flags: u32,
        max_render_size: FfxApiDimensions2D,
        max_upscale_size: FfxApiDimensions2D,
    ) -> Result<Fsr4Network, (u32, String)> {
        if head.is_null() {
            return Err((
                FFX_API_RETURN_ERROR_PARAMETER,
                "null FfxApiHeader in ffxCreateContext".to_string(),
            ));
        }
        if max_render_size.width == 0
            || max_render_size.height == 0
            || max_upscale_size.width == 0
            || max_upscale_size.height == 0
            || max_render_size.width > max_upscale_size.width
            || max_render_size.height > max_upscale_size.height
        {
            return Err((
                FFX_API_RETURN_ERROR_PARAMETER,
                format!(
                    "invalid render/upscale dimensions: {}x{} -> {}x{}",
                    max_render_size.width,
                    max_render_size.height,
                    max_upscale_size.width,
                    max_upscale_size.height
                ),
            ));
        }

        // Walk FFX pNext chain with cycle guard.
        let mut visited = [0usize; 32];
        let mut depth = 0usize;
        let mut cur = head;
        let mut saw_vk_backend = false;
        let mut saw_upscale_version = false;
        let mut pinned_version: Option<u64> = None;

        while !cur.is_null() {
            if depth >= visited.len() {
                return Err((
                    FFX_API_RETURN_ERROR_PARAMETER,
                    "FFX pNext chain exceeded depth 32".to_string(),
                ));
            }
            let addr = cur as usize;
            if visited[..depth].contains(&addr) {
                return Err((
                    FFX_API_RETURN_ERROR_PARAMETER,
                    format!("FFX pNext cycle detected at {addr:#x}"),
                ));
            }
            visited[depth] = addr;
            depth += 1;

            let hdr = unsafe { &*cur };
            match hdr.desc_type {
                FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE => {}
                FFX_SDK24_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK => {
                    let vk_desc = unsafe { &*(cur as *const FfxCreateBackendVKDesc) };
                    if vk_desc.device_context.vk_device == vk::Device::null()
                        || vk_desc.device_context.vk_physical_device == vk::PhysicalDevice::null()
                    {
                        return Err((
                            FFX_API_RETURN_ERROR_PARAMETER,
                            "FfxCreateBackendVKDesc has null VkDevice or VkPhysicalDevice".to_string(),
                        ));
                    }
                    // Verify vkGetDeviceProcAddr actually resolves core Vulkan commands.
                    let probe = unsafe {
                        (vk_desc.device_context.vk_device_proc_addr)(
                            vk_desc.device_context.vk_device,
                            c"vkCmdPipelineBarrier2".as_ptr(),
                        )
                    };
                    if probe.is_none() {
                        return Err((
                            FFX_API_RETURN_ERROR_RUNTIME_ERROR,
                            "vkGetDeviceProcAddr failed to resolve vkCmdPipelineBarrier2".to_string(),
                        ));
                    }
                    saw_vk_backend = true;
                }
                desc if (desc & 0xFF00_0000) == FFX_SDK24_BACKEND_ID_XBOX => {
                    return Err((
                        FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE,
                        format!(
                            "rejected Xbox backend desc_type {desc:#010x} (expected Vulkan SDK 2.4.0 {FFX_SDK24_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK:#010x})"
                        ),
                    ));
                }
                FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION => {
                    saw_upscale_version = true;
                }
                FFX_API_DESC_TYPE_OVERRIDE_VERSION => {
                    let ov = unsafe { &*(cur as *const FfxOverrideVersionDesc) };
                    pinned_version = Some(ov.version_id);
                }
                other => {
                    return Err((
                        FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE,
                        format!("unknown FFX create desc_type {other:#010x}"),
                    ));
                }
            }
            cur = hdr.p_next;
        }

        if !saw_vk_backend {
            return Err((
                FFX_API_RETURN_NO_PROVIDER,
                "missing FfxCreateBackendVKDesc in pNext chain".to_string(),
            ));
        }
        if !saw_upscale_version {
            return Err((
                FFX_API_RETURN_ERROR_PARAMETER,
                "missing mandatory FfxCreateContextDescUpscaleVersion in SDK 2.4.0 pNext chain"
                    .to_string(),
            ));
        }

        let mut network = self
            .icd
            .profile
            .select_fsr4_network(self.override_network.as_deref())
            .map_err(|e| (FFX_API_RETURN_NO_PROVIDER, e.to_string()))?;

        if let Some(pin) = pinned_version {
            let available = self.query_versions();
            if !available.iter().any(|(id, _)| *id == pin) {
                return Err((
                    FFX_API_RETURN_NO_PROVIDER,
                    format!("requested override version {pin:#x} unavailable on {}", self.icd.profile.display_name),
                ));
            }
            if pin == Fsr4Network::Fsr316Fallback.version_id() {
                network = Fsr4Network::Fsr316Fallback;
            }
        }

        self.context = Some(SimulatedFsrContext {
            selected_network: network,
            max_render_size,
            max_upscale_size,
            flags,
            last_render_size: None,
            in_flight_frames: VecDeque::new(),
            total_dispatches: 0,
        });

        Ok(network)
    }

    /// Retires the oldest in-flight frame (simulating GPU fence completion).
    pub fn retire_oldest_frame(&mut self) -> bool {
        if let Some(ctx) = &mut self.context {
            ctx.in_flight_frames.pop_front().is_some()
        } else {
            false
        }
    }

    /// Simulates `ffxDispatch` with full layout, parameter, and in-flight frame validation.
    pub fn dispatch(&mut self, desc: &FfxSdk24DispatchDescUpscale) -> Result<u32, (u32, String)> {
        let Some(ctx) = &mut self.context else {
            return Err((
                FFX_API_RETURN_ERROR_PARAMETER,
                "ffxDispatch called without active context".to_string(),
            ));
        };

        if desc.header.desc_type != FFX_API_DISPATCH_DESC_TYPE_UPSCALE {
            return Err((
                FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE,
                format!("unexpected dispatch desc_type {:#x}", desc.header.desc_type),
            ));
        }
        if desc.command_list == vk::CommandBuffer::null() {
            return Err((
                FFX_API_RETURN_ERROR_PARAMETER,
                "null VkCommandBuffer in ffxDispatch".to_string(),
            ));
        }
        if ctx.in_flight_frames.len() >= FSR4_MAX_IN_FLIGHT_FRAMES {
            return Err((
                FFX_API_RETURN_ERROR_RUNTIME_ERROR,
                format!(
                    "exceeded FSR 4.1.2 max in-flight frame limit ({FSR4_MAX_IN_FLIGHT_FRAMES})"
                ),
            ));
        }

        // Dimensions check
        if desc.render_size.width == 0
            || desc.render_size.height == 0
            || desc.render_size.width > ctx.max_render_size.width
            || desc.render_size.height > ctx.max_render_size.height
            || desc.upscale_size.width == 0
            || desc.upscale_size.height == 0
            || desc.upscale_size.width > ctx.max_upscale_size.width
            || desc.upscale_size.height > ctx.max_upscale_size.height
        {
            return Err((
                FFX_API_RETURN_ERROR_PARAMETER,
                "render_size or upscale_size out of context bounds".to_string(),
            ));
        }

        // Dynamic resolution change without reset when DYNAMIC_RESOLUTION flag is absent
        if let Some(prev) = ctx.last_render_size
            && (prev.width != desc.render_size.width || prev.height != desc.render_size.height)
            && (ctx.flags & upscale_flags::DYNAMIC_RESOLUTION) == 0
            && desc.reset == 0
        {
            return Err((
                FFX_API_RETURN_ERROR_PARAMETER,
                "render_size changed without reset=1 or DYNAMIC_RESOLUTION flag".to_string(),
            ));
        }

        // Scalar parameter sanity
        if !desc.jitter_offset.x.is_finite()
            || !desc.jitter_offset.y.is_finite()
            || desc.jitter_offset.x.abs() > 0.5
            || desc.jitter_offset.y.abs() > 0.5
        {
            return Err((
                FFX_API_RETURN_ERROR_PARAMETER,
                format!(
                    "jitter_offset ({}, {}) outside [-0.5, 0.5] or non-finite",
                    desc.jitter_offset.x, desc.jitter_offset.y
                ),
            ));
        }
        if !desc.motion_vector_scale.x.is_finite()
            || !desc.motion_vector_scale.y.is_finite()
            || desc.motion_vector_scale.x == 0.0
            || desc.motion_vector_scale.y == 0.0
        {
            return Err((
                FFX_API_RETURN_ERROR_PARAMETER,
                "motion_vector_scale must be finite and non-zero".to_string(),
            ));
        }
        if !desc.frame_time_delta.is_finite() || desc.frame_time_delta <= 0.0 {
            return Err((
                FFX_API_RETURN_ERROR_PARAMETER,
                format!("invalid frame_time_delta {}", desc.frame_time_delta),
            ));
        }
        if !desc.camera_fov_angle_vertical.is_finite()
            || desc.camera_fov_angle_vertical <= 0.0
            || desc.camera_fov_angle_vertical >= std::f32::consts::PI
        {
            return Err((
                FFX_API_RETURN_ERROR_PARAMETER,
                format!(
                    "invalid camera_fov_angle_vertical {}",
                    desc.camera_fov_angle_vertical
                ),
            ));
        }
        let depth_inverted = (ctx.flags & upscale_flags::DEPTH_INVERTED) != 0;
        let depth_infinite = (ctx.flags & upscale_flags::DEPTH_INFINITE) != 0;
        if depth_inverted {
            if !depth_infinite && !(desc.camera_near > desc.camera_far && desc.camera_far >= 0.0) {
                return Err((
                    FFX_API_RETURN_ERROR_PARAMETER,
                    format!(
                        "DEPTH_INVERTED requires camera_near ({}) > camera_far ({})",
                        desc.camera_near, desc.camera_far
                    ),
                ));
            }
        } else if !(desc.camera_near > 0.0 && desc.camera_far > desc.camera_near) {
            return Err((
                FFX_API_RETURN_ERROR_PARAMETER,
                format!(
                    "standard depth requires 0 < camera_near ({}) < camera_far ({})",
                    desc.camera_near, desc.camera_far
                ),
            ));
        }

        // Validate live VkImage layouts against MockIcdState
        let required_inputs = [
            ("color", desc.color),
            ("depth", desc.depth),
            ("motion_vectors", desc.motion_vectors),
        ];
        for (label, res) in required_inputs {
            if res.resource.is_null() {
                return Err((
                    FFX_API_RETURN_ERROR_PARAMETER,
                    format!("required input `{label}` has null VkImage handle"),
                ));
            }
            let img = vk::Image::from_raw(res.resource as usize as u64);
            match self.icd.query_image_layout(img) {
                Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => {}
                Some(other) => {
                    return Err((
                        FFX_API_RETURN_ERROR_RUNTIME_ERROR,
                        format!(
                            "input `{label}` in layout {other:?}, expected SHADER_READ_ONLY_OPTIMAL"
                        ),
                    ));
                }
                None => {
                    return Err((
                        FFX_API_RETURN_ERROR_PARAMETER,
                        format!("input `{label}` VkImage handle not tracked by MockIcd"),
                    ));
                }
            }
        }

        if desc.output.resource.is_null() {
            return Err((
                FFX_API_RETURN_ERROR_PARAMETER,
                "required `output` has null VkImage handle".to_string(),
            ));
        }
        let out_img = vk::Image::from_raw(desc.output.resource as usize as u64);
        match self.icd.query_image_layout(out_img) {
            Some(vk::ImageLayout::GENERAL) => {}
            Some(other) => {
                return Err((
                    FFX_API_RETURN_ERROR_RUNTIME_ERROR,
                    format!("output image in layout {other:?}, expected GENERAL"),
                ));
            }
            None => {
                return Err((
                    FFX_API_RETURN_ERROR_PARAMETER,
                    "output VkImage handle not tracked by MockIcd".to_string(),
                ));
            }
        }

        ctx.total_dispatches += 1;
        ctx.last_render_size = Some(desc.render_size);
        ctx.in_flight_frames.push_back(ctx.total_dispatches);
        Ok(FFX_API_RETURN_OK)
    }

    pub fn destroy_context(&mut self) -> u32 {
        self.context = None;
        FFX_API_RETURN_OK
    }
}

/// Deterministic Halton(2, 3) sub-pixel jitter in `[-0.5, 0.5]`.
pub fn halton_jitter(frame_index: u32, phase_count: u32) -> (f32, f32) {
    let idx = (frame_index % phase_count.max(1)) + 1;
    let h2 = halton_base(idx, 2) - 0.5;
    let h3 = halton_base(idx, 3) - 0.5;
    (h2, h3)
}

fn halton_base(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0f32;
    let mut r = 0.0f32;
    let bf = base as f32;
    while index > 0 {
        f /= bf;
        r += f * (index % base) as f32;
        index /= base;
    }
    r
}

/// Computes recommended jitter phase count (`ceil(8 * (display / render)^2)`).
pub fn fsr_jitter_phase_count(render_width: u32, display_width: u32) -> u32 {
    if render_width == 0 {
        return 8;
    }
    let ratio = (display_width as f32) / (render_width as f32);
    (8.0 * ratio * ratio).ceil().max(8.0) as u32
}

/// Structured result of simulating the full FSR 4.1.2 / Mock Vulkan ICD lifecycle on a GPU profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub gpu_id: String,
    pub display_name: String,
    pub driver_model: String,
    pub selected_network: String,
    pub available_versions: Vec<String>,
    pub dispatches_succeeded: u32,
    pub in_flight_cap_enforced: bool,
    pub xbox_backend_bug_rejected: bool,
    pub layout_violation_caught: bool,
    pub leak_free: bool,
    pub passed: bool,
    pub notes: Vec<String>,
}

/// Executes a complete hardware-free simulation on `profile`:
/// 1. Boots a thread-isolated Mock Vulkan ICD with `profile`.
/// 2. Creates `VkInstance`, `VkDevice` (with profile-supported extensions), 4 `VkImage`s, and a `VkCommandBuffer`.
/// 3. Verifies that the Xbox backend ID (`0x0100_0003`) is rejected cleanly.
/// 4. Creates a valid SDK 2.4.0 FSR 4.1.2/3.1.6 context with `FfxCreateContextDescUpscaleVersion`.
/// 5. Verifies that dispatching with `UNDEFINED` layouts fails, then transitions via `vkCmdPipelineBarrier2`.
/// 6. Executes 15 valid dispatches, verifies the 16th in-flight dispatch is rejected, retires a frame, and succeeds on frame 16.
/// 7. Destroys all resources and verifies zero handle leaks and zero mock ICD violations.
pub fn run_profile_simulation(
    profile: &GpuProfile,
    fsr_override: Option<&str>,
) -> SimulationResult {
    let (guard, entry) = MockIcdGuard::install(profile.clone());
    let mut notes = Vec::new();
    let mut provider = MockFfxProvider::new(guard.state().clone(), fsr_override.map(str::to_string));
    let available_versions: Vec<String> = provider
        .query_versions()
        .into_iter()
        .map(|(id, name)| format!("{name} ({id:#x})"))
        .collect();

    let mut selected_network = String::from("none");
    let mut dispatches_succeeded = 0u32;
    let mut in_flight_cap_enforced = false;
    let mut xbox_backend_bug_rejected = false;
    let mut layout_violation_caught = false;

    unsafe {
        let app = vk::ApplicationInfo::default().api_version(profile.api_version);
        let inst_ci = vk::InstanceCreateInfo::default().application_info(&app);
        let instance = match entry.create_instance(&inst_ci, None) {
            Ok(i) => i,
            Err(e) => {
                notes.push(format!("vkCreateInstance failed: {e:?}"));
                return SimulationResult {
                    gpu_id: profile.id.slug().to_string(),
                    display_name: profile.display_name.to_string(),
                    driver_model: profile.driver.as_str().to_string(),
                    selected_network,
                    available_versions,
                    dispatches_succeeded,
                    in_flight_cap_enforced,
                    xbox_backend_bug_rejected,
                    layout_violation_caught,
                    leak_free: false,
                    passed: false,
                    notes,
                };
            }
        };

        let phys = instance.enumerate_physical_devices().unwrap()[0];
        let mut dev_ext_ptrs: Vec<*const std::ffi::c_char> = Vec::new();
        let mut dev_ext_cstrs: Vec<std::ffi::CString> = Vec::new();
        for ext in [
            "VK_KHR_swapchain",
            "VK_KHR_synchronization2",
            "VK_KHR_cooperative_matrix",
            "VK_EXT_shader_float8",
            "VK_KHR_compute_shader_derivatives",
        ] {
            if profile.has_extension(ext) {
                let cs = std::ffi::CString::new(ext).unwrap();
                dev_ext_ptrs.push(cs.as_ptr());
                dev_ext_cstrs.push(cs);
            }
        }
        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(0)
            .queue_priorities(&prio)];
        let dev_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qci)
            .enabled_extension_names(&dev_ext_ptrs);
        let device = instance.create_device(phys, &dev_ci, None).unwrap();
        let _keep_alive = dev_ext_cstrs;

        // Validate pNext helper with empty chain
        let _ = validate_pnext_chain(ptr::null(), Some(profile));

        // 1. Negative check: ensure Xbox backend ID 0x0100_0003 is rejected
        let mut bad_backend = FfxCreateBackendVKDesc {
            header: FfxApiHeader {
                desc_type: FFX_SDK24_BACKEND_ID_XBOX | 0x03,
                p_next: ptr::null_mut(),
            },
            device_context: FfxVkDeviceContext {
                vk_device: device.handle(),
                vk_physical_device: phys,
                vk_device_proc_addr: mock_vk_get_device_proc_addr,
            },
        };
        let bad_upscale = FfxApiHeader {
            desc_type: FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE,
            p_next: &mut bad_backend.header,
        };
        if let Err((code, _)) = provider.create_context(
            &bad_upscale,
            upscale_flags::HIGH_DYNAMIC_RANGE | upscale_flags::DEPTH_INVERTED,
            FfxApiDimensions2D {
                width: 1280,
                height: 720,
            },
            FfxApiDimensions2D {
                width: 1920,
                height: 1080,
            },
        ) && code == FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE
        {
            xbox_backend_bug_rejected = true;
        }

        // 2. Positive SDK 2.4.0 context creation
        let mut ver_desc = FfxCreateContextDescUpscaleVersion {
            header: FfxApiHeader {
                desc_type: FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION,
                p_next: ptr::null_mut(),
            },
            version: 0x0004_0001_0002,
        };
        let mut vk_backend = FfxCreateBackendVKDesc {
            header: FfxApiHeader {
                desc_type: FFX_SDK24_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK,
                p_next: &mut ver_desc.header,
            },
            device_context: FfxVkDeviceContext {
                vk_device: device.handle(),
                vk_physical_device: phys,
                vk_device_proc_addr: mock_vk_get_device_proc_addr,
            },
        };
        let upscale_hdr = FfxApiHeader {
            desc_type: FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE,
            p_next: &mut vk_backend.header,
        };

        let render_dim = FfxApiDimensions2D {
            width: 1280,
            height: 720,
        };
        let output_dim = FfxApiDimensions2D {
            width: 1920,
            height: 1080,
        };
        match provider.create_context(
            &upscale_hdr,
            upscale_flags::HIGH_DYNAMIC_RANGE | upscale_flags::DEPTH_INVERTED,
            render_dim,
            output_dim,
        ) {
            Ok(net) => selected_network = net.amd_name().to_string(),
            Err((code, msg)) => notes.push(format!("create_context failed ({code}): {msg}")),
        }

        // Allocate Vulkan images & command buffer
        let mk_img = |fmt: vk::Format, w: u32, h: u32, usage: vk::ImageUsageFlags| {
            let ci = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(fmt)
                .extent(vk::Extent3D {
                    width: w,
                    height: h,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .usage(usage)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            device.create_image(&ci, None).unwrap()
        };

        let color = mk_img(
            vk::Format::R16G16B16A16_SFLOAT,
            1280,
            720,
            vk::ImageUsageFlags::SAMPLED,
        );
        let depth = mk_img(
            vk::Format::D32_SFLOAT,
            1280,
            720,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        );
        let mv = mk_img(
            vk::Format::R16G16_SFLOAT,
            1280,
            720,
            vk::ImageUsageFlags::SAMPLED,
        );
        let output = mk_img(
            vk::Format::R16G16B16A16_SFLOAT,
            1920,
            1080,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
        );

        let pool = device
            .create_command_pool(&vk::CommandPoolCreateInfo::default(), None)
            .unwrap();
        let cbs = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .unwrap();
        let cb = cbs[0];

        let mk_dispatch = |frame: u32, reset: u8| {
            let phase = fsr_jitter_phase_count(1280, 1920);
            let (jx, jy) = halton_jitter(frame, phase);
            FfxSdk24DispatchDescUpscale {
                header: FfxApiHeader {
                    desc_type: FFX_API_DISPATCH_DESC_TYPE_UPSCALE,
                    p_next: ptr::null_mut(),
                },
                command_list: cb,
                color: FfxApiResource::from_vk_image(
                    color,
                    vk::Format::R16G16B16A16_SFLOAT,
                    1280,
                    720,
                    1,
                    resource_usage::READ_ONLY,
                    resource_state::COMPUTE_READ,
                ),
                depth: FfxApiResource::from_vk_image(
                    depth,
                    vk::Format::D32_SFLOAT,
                    1280,
                    720,
                    1,
                    resource_usage::READ_ONLY,
                    resource_state::COMPUTE_READ,
                ),
                motion_vectors: FfxApiResource::from_vk_image(
                    mv,
                    vk::Format::R16G16_SFLOAT,
                    1280,
                    720,
                    1,
                    resource_usage::READ_ONLY,
                    resource_state::COMPUTE_READ,
                ),
                exposure: FfxApiResource::default(),
                reactive: FfxApiResource::default(),
                transparency_and_composition: FfxApiResource::default(),
                output: FfxApiResource::from_vk_image(
                    output,
                    vk::Format::R16G16B16A16_SFLOAT,
                    1920,
                    1080,
                    1,
                    resource_usage::UAV,
                    resource_state::UNORDERED_ACCESS,
                ),
                jitter_offset: FfxApiFloatCoords2D { x: jx, y: jy },
                motion_vector_scale: FfxApiFloatCoords2D {
                    x: -1280.0,
                    y: -720.0,
                },
                render_size: render_dim,
                upscale_size: output_dim,
                enable_sharpening: 1,
                sharpness: 0.4,
                frame_time_delta: 16.666,
                pre_exposure: 1.0,
                reset,
                camera_near: 512.0,
                camera_far: 0.05,
                camera_fov_angle_vertical: 1.22173,
                view_space_to_meters_factor: 1.0,
                flags: 0,
            }
        };

        // 3. Negative layout check: inputs are still UNDEFINED
        if provider.dispatch(&mk_dispatch(0, 1)).is_err() {
            layout_violation_caught = true;
        }

        // 4. Transition layouts properly via vkCmdPipelineBarrier2
        let barriers = [
            vk::ImageMemoryBarrier2::default()
                .image(color)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            vk::ImageMemoryBarrier2::default()
                .image(depth)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            vk::ImageMemoryBarrier2::default()
                .image(mv)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            vk::ImageMemoryBarrier2::default()
                .image(output)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL),
        ];
        device.cmd_pipeline_barrier2(
            cb,
            &vk::DependencyInfo::default().image_memory_barriers(&barriers),
        );

        // 5. Fill 15 in-flight frames
        for frame in 0..FSR4_MAX_IN_FLIGHT_FRAMES as u32 {
            match provider.dispatch(&mk_dispatch(frame, u8::from(frame == 0))) {
                Ok(_) => dispatches_succeeded += 1,
                Err((code, msg)) => notes.push(format!("frame {frame} failed ({code}): {msg}")),
            }
        }

        // 6. Verify 16th in-flight dispatch is rejected, then retire 1 frame and succeed
        if provider.dispatch(&mk_dispatch(15, 0)).is_err() {
            in_flight_cap_enforced = true;
        }
        provider.retire_oldest_frame();
        if provider.dispatch(&mk_dispatch(15, 0)).is_ok() {
            dispatches_succeeded += 1;
        }

        provider.destroy_context();

        // 7. Clean teardown
        device.free_command_buffers(pool, &cbs);
        device.destroy_command_pool(pool, None);
        device.destroy_image(color, None);
        device.destroy_image(depth, None);
        device.destroy_image(mv, None);
        device.destroy_image(output, None);
        device.destroy_device(None);
        reclaim_mock_physical_device(phys);
        instance.destroy_instance(None);
    }

    let leaks = guard.state().leak_report();
    let leak_free = leaks.is_clean();
    if !leak_free {
        notes.push(format!("MockIcd leak/violation report: {leaks:?}"));
    }

    let passed = selected_network != "none"
        && dispatches_succeeded == 16
        && in_flight_cap_enforced
        && xbox_backend_bug_rejected
        && layout_violation_caught
        && leak_free;



    SimulationResult {
        gpu_id: profile.id.slug().to_string(),
        display_name: profile.display_name.to_string(),
        driver_model: profile.driver.as_str().to_string(),
        selected_network,
        available_versions,
        dispatches_succeeded,
        in_flight_cap_enforced,
        xbox_backend_bug_rejected,
        layout_violation_caught,
        leak_free,
        passed,
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiles::{DriverModel, GpuId};

    #[test]
    fn full_fsr4_simulation_passes_across_all_16_gpu_os_profiles() {
        for profile in GpuProfile::all_matrix() {
            let res = run_profile_simulation(&profile, None);
            assert!(
                res.passed,
                "simulation failed for {} ({}): {:?}",
                profile.display_name, profile.driver, res.notes
            );
        }
    }

    #[test]
    fn rx9070xt_selects_fp8_and_rtx3050m_selects_fsr316() {
        let rdna4 = run_profile_simulation(
            &GpuProfile::build(GpuId::Rx9070Xt, DriverModel::Linux),
            None,
        );
        assert_eq!(rdna4.selected_network, "FSR4");

        let ampere = run_profile_simulation(
            &GpuProfile::build(GpuId::Rtx3050Mobile, DriverModel::Linux),
            None,
        );
        assert_eq!(ampere.selected_network, "FSR 3.1.6");
    }
}
