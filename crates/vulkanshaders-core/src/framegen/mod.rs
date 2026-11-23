pub mod present_thread;

use crate::gpu::{VulkanContext, with_queue_lock};
use ash::vk::{self, Handle};
use present_thread::{PresentJob, PresentThread};
use std::{
    ffi::c_void,
    ptr,
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum FrameGenBackendMode {
    Off = 0,
    OpticalFlowCompute = 1,
    Fsr3FrameGen = 2,
    DlssFrameGen = 3,
}

impl FrameGenBackendMode {
    pub const fn from_u32(val: u32) -> Self {
        match val {
            1 => Self::OpticalFlowCompute,
            2 => Self::Fsr3FrameGen,
            3 => Self::DlssFrameGen,
            _ => Self::Off,
        }
    }
}

/// `VK_NV_low_latency2` structures and function pointer types (`VK_STRUCTURE_TYPE_LATENCY_*_NV`).
#[repr(C)]
pub struct VkLatencySleepModeInfoNV {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub low_latency_mode: vk::Bool32,
    pub low_latency_boost: vk::Bool32,
    pub minimum_interval_us: u32,
}

#[repr(C)]
pub struct VkLatencySleepInfoNV {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub signal_semaphore: vk::Semaphore,
    pub value: u64,
}

#[repr(C)]
pub struct VkSetLatencyMarkerInfoNV {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub present_id: u64,
    pub marker: i32,
}

type PfnVkSetLatencySleepModeNV = unsafe extern "system" fn(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    p_sleep_mode_info: *const VkLatencySleepModeInfoNV,
) -> vk::Result;

type PfnVkLatencySleepNV = unsafe extern "system" fn(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    p_sleep_info: *const VkLatencySleepInfoNV,
) -> vk::Result;

type PfnVkSetLatencyMarkerNV = unsafe extern "system" fn(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    p_latency_marker_info: *const VkSetLatencyMarkerInfoNV,
);

const VK_STRUCTURE_TYPE_LATENCY_SLEEP_MODE_INFO_NV: i32 = 1_000_505_000;
const VK_STRUCTURE_TYPE_LATENCY_SLEEP_INFO_NV: i32 = 1_000_505_001;
const VK_STRUCTURE_TYPE_SET_LATENCY_MARKER_INFO_NV: i32 = 1_000_505_002;

pub const VK_LATENCY_MARKER_SIMULATION_START_NV: i32 = 0;
pub const VK_LATENCY_MARKER_SIMULATION_END_NV: i32 = 1;
pub const VK_LATENCY_MARKER_RENDERSUBMIT_START_NV: i32 = 2;
pub const VK_LATENCY_MARKER_RENDERSUBMIT_END_NV: i32 = 3;
pub const VK_LATENCY_MARKER_PRESENT_START_NV: i32 = 4;
pub const VK_LATENCY_MARKER_PRESENT_END_NV: i32 = 5;
pub const VK_LATENCY_MARKER_INPUT_SAMPLE_NV: i32 = 6;

/// Manages `VK_NV_low_latency2` (NVIDIA Reflex) swapchain state, frame pacing markers,
/// UI-aware frame generation (`HudlessColor` + `UiColor` + `MotionVectors` + `LinearDepth`),
/// HDR scRGB / HDR10 swapchain format selection, and the asynchronous `PresentThread`.
pub struct FrameGenController {
    pub mode: FrameGenBackendMode,
    pub hdr_enabled: bool,
    pub reflex_enabled: bool,
    /// Armed only after a recent healthy frame/present; disarmed on swapchain recreate, errors, or >500ms stalls.
    reflex_sleep_armed: bool,
    pub active_swapchain: u64,
    pub active_swapchain_epoch: u64,
    pub swapchain_images: Vec<u64>,
    pub interpolated_image: u64,
    /// Internal `RES_SWAPCHAIN_OUT` RGBA16F image produced by `hdr_compose.comp`; presented directly
    /// on HDR displays so no sRGB blit path is used.
    pub hdr_output_image: u64,
    pub hdr_output_valid_this_frame: bool,
    pub swapchain_width: u32,
    pub swapchain_height: u32,
    pub current_frame_id: u64,
    pub latency_sleep_semaphore: vk::Semaphore,
    pub latency_sleep_value: u64,
    pub present_thread: PresentThread,
    pub pending_present_job: Option<PresentJob>,
    last_present_timestamp: Option<Instant>,
    smoothed_frame_delta: Duration,
    pfn_set_sleep_mode: Option<PfnVkSetLatencySleepModeNV>,
    pfn_latency_sleep: Option<PfnVkLatencySleepNV>,
    pfn_set_marker: Option<PfnVkSetLatencyMarkerNV>,
}

impl Default for FrameGenController {
    fn default() -> Self {
        let present_thread = PresentThread::new();
        let active_swapchain_epoch = present_thread.current_epoch();
        Self {
            mode: FrameGenBackendMode::Off,
            hdr_enabled: false,
            reflex_enabled: false,
            reflex_sleep_armed: false,
            active_swapchain: 0,
            active_swapchain_epoch,
            swapchain_images: Vec::new(),
            interpolated_image: 0,
            hdr_output_image: 0,
            hdr_output_valid_this_frame: false,
            swapchain_width: 0,
            swapchain_height: 0,
            current_frame_id: 0,
            latency_sleep_semaphore: vk::Semaphore::null(),
            latency_sleep_value: 0,
            present_thread,
            pending_present_job: None,
            last_present_timestamp: None,
            smoothed_frame_delta: Duration::from_micros(16_667),
            pfn_set_sleep_mode: None,
            pfn_latency_sleep: None,
            pfn_set_marker: None,
        }
    }
}

impl FrameGenController {
    /// Dynamically loads `VK_NV_low_latency2` device entrypoints and allocates the timeline
    /// semaphore required by `vkLatencySleepNV`.
    pub fn init_reflex(&mut self, ctx: &VulkanContext) {
        if !ctx.capabilities.low_latency2_nv {
            self.reflex_enabled = false;
            return;
        }
        unsafe {
            let sleep_mode_name = c"vkSetLatencySleepModeNV";
            let sleep_name = c"vkLatencySleepNV";
            let marker_name = c"vkSetLatencyMarkerNV";
            let f1 = ctx
                .instance
                .get_device_proc_addr(ctx.device.handle(), sleep_mode_name.as_ptr());
            let f2 = ctx
                .instance
                .get_device_proc_addr(ctx.device.handle(), sleep_name.as_ptr());
            let f3 = ctx
                .instance
                .get_device_proc_addr(ctx.device.handle(), marker_name.as_ptr());
            if let (Some(p1), Some(p3)) = (f1, f3) {
                self.pfn_set_sleep_mode = Some(std::mem::transmute::<
                    vk::PFN_vkVoidFunction,
                    PfnVkSetLatencySleepModeNV,
                >(Some(p1)));
                self.pfn_set_marker = Some(std::mem::transmute::<
                    vk::PFN_vkVoidFunction,
                    PfnVkSetLatencyMarkerNV,
                >(Some(p3)));
                if let Some(p2) = f2 {
                    self.pfn_latency_sleep = Some(std::mem::transmute::<
                        vk::PFN_vkVoidFunction,
                        PfnVkLatencySleepNV,
                    >(Some(p2)));
                }

                // Allocate a timeline semaphore for vkLatencySleepNV
                let mut type_info = vk::SemaphoreTypeCreateInfo::default()
                    .semaphore_type(vk::SemaphoreType::TIMELINE)
                    .initial_value(0);
                let sem_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
                if let Ok(sem) = ctx.device.create_semaphore(&sem_info, None) {
                    self.latency_sleep_semaphore = sem;
                }
                self.reflex_enabled = true;
            }
        }
    }

    /// Resets HDR output validity and FrameGen history when leaving a world (e.g. returning to
    /// the main menu) so stale world frames are never presented onto an HDR/FG swapchain.
    pub fn on_world_unloaded(&mut self) {
        self.pending_present_job = None;
        self.active_swapchain_epoch = self.present_thread.advance_swapchain_epoch();
        self.present_thread.wait_idle();
        self.hdr_output_image = 0;
        self.hdr_output_valid_this_frame = false;
        self.interpolated_image = 0;
        self.last_present_timestamp = None;
        self.reflex_sleep_armed = false;
    }

    /// Updates the active swapchain handle together with its actual pixel extent `(width, height)`.
    pub fn set_swapchain_info(
        &mut self,
        ctx: Option<&VulkanContext>,
        swapchain_raw: u64,
        width: u32,
        height: u32,
    ) {
        self.set_swapchain(ctx, swapchain_raw);
        if swapchain_raw != 0 {
            self.swapchain_width = width;
            self.swapchain_height = height;
        } else {
            self.swapchain_width = 0;
            self.swapchain_height = 0;
        }
    }

    pub fn set_swapchain(&mut self, ctx: Option<&VulkanContext>, swapchain_raw: u64) {
        self.pending_present_job = None;
        self.hdr_output_valid_this_frame = false;
        self.reflex_sleep_armed = false;
        self.last_present_timestamp = None;
        // Advance the swapchain epoch BEFORE waiting so any sleeping PresentThread wakes immediately.
        self.active_swapchain_epoch = self.present_thread.advance_swapchain_epoch();
        self.present_thread.wait_idle();
        if swapchain_raw == 0 {
            self.present_thread.flush_gpu_resources();
            self.active_swapchain_epoch = self.present_thread.current_epoch();
            self.swapchain_width = 0;
            self.swapchain_height = 0;
        }
        self.active_swapchain = swapchain_raw;
        self.swapchain_images.clear();

        if swapchain_raw != 0
            && let Some(c) = ctx
        {
            let swapchain_ext = ash::khr::swapchain::Device::new(&c.instance, &c.device);
            let sc = vk::SwapchainKHR::from_raw(swapchain_raw);
            if let Ok(imgs) = unsafe { swapchain_ext.get_swapchain_images(sc) } {
                self.swapchain_images = imgs.into_iter().map(|i| i.as_raw()).collect();
            }

            if self.reflex_enabled
                && let Some(set_mode) = self.pfn_set_sleep_mode
            {
                let info = VkLatencySleepModeInfoNV {
                    s_type: VK_STRUCTURE_TYPE_LATENCY_SLEEP_MODE_INFO_NV,
                    p_next: ptr::null(),
                    low_latency_mode: vk::TRUE,
                    low_latency_boost: vk::TRUE,
                    minimum_interval_us: 0,
                };
                unsafe {
                    let _ = set_mode(c.device.handle(), sc, &info);
                }
            }
        }
    }

    /// Invoked at the start of the client frame before input polling:
    /// increments the monotonic `present_id`, calls `vkLatencySleepNV`, and emits
    /// `VK_LATENCY_MARKER_SIMULATION_START_NV` + `VK_LATENCY_MARKER_INPUT_SAMPLE_NV`.
    pub fn begin_frame_sleep_and_input_sample(&mut self, ctx: &VulkanContext) {
        self.current_frame_id = self.current_frame_id.wrapping_add(1);
        if self.active_swapchain == 0 || !self.reflex_enabled {
            return;
        }
        let recent_present = self
            .last_present_timestamp
            .is_some_and(|t| t.elapsed() < Duration::from_millis(500));
        if self.reflex_sleep_armed
            && recent_present
            && self.latency_sleep_semaphore != vk::Semaphore::null()
        {
            self.latency_sleep_value = self.latency_sleep_value.wrapping_add(1);
            if !self.sleep_before_input(ctx, self.latency_sleep_semaphore, self.latency_sleep_value)
            {
                self.reflex_sleep_armed = false;
            }
        } else {
            self.reflex_sleep_armed = false;
        }
        self.set_reflex_marker(ctx, VK_LATENCY_MARKER_SIMULATION_START_NV);
        self.set_reflex_marker(ctx, VK_LATENCY_MARKER_INPUT_SAMPLE_NV);
    }

    pub fn sleep_before_input(&self, ctx: &VulkanContext, sem: vk::Semaphore, value: u64) -> bool {
        if self.active_swapchain == 0 || !self.reflex_enabled || sem == vk::Semaphore::null() {
            return false;
        }
        if let Some(sleep_fn) = self.pfn_latency_sleep {
            let info = VkLatencySleepInfoNV {
                s_type: VK_STRUCTURE_TYPE_LATENCY_SLEEP_INFO_NV,
                p_next: ptr::null(),
                signal_semaphore: sem,
                value,
            };
            unsafe {
                let res = sleep_fn(
                    ctx.device.handle(),
                    vk::SwapchainKHR::from_raw(self.active_swapchain),
                    &info,
                );
                if res != vk::Result::SUCCESS {
                    return false;
                }
                let sems = [sem];
                let vals = [value];
                let wait_info = vk::SemaphoreWaitInfo::default()
                    .semaphores(&sems)
                    .values(&vals);
                let _ = ctx.device.wait_semaphores(&wait_info, 15_000_000);
            }
        }
        true
    }

    pub fn set_reflex_marker(&self, ctx: &VulkanContext, marker_id: i32) {
        if self.active_swapchain == 0 || !self.reflex_enabled {
            return;
        }
        if let Some(set_marker) = self.pfn_set_marker {
            let marker = VkSetLatencyMarkerInfoNV {
                s_type: VK_STRUCTURE_TYPE_SET_LATENCY_MARKER_INFO_NV,
                p_next: ptr::null(),
                present_id: self.current_frame_id.max(1),
                marker: marker_id,
            };
            unsafe {
                set_marker(
                    ctx.device.handle(),
                    vk::SwapchainKHR::from_raw(self.active_swapchain),
                    &marker,
                );
            }
        }
    }

    pub fn intercept_acquire_next_image(
        &mut self,
        ctx: &VulkanContext,
        swapchain_raw: u64,
        timeout_ns: u64,
        semaphore_raw: u64,
        fence_raw: u64,
    ) -> i32 {
        if swapchain_raw != self.active_swapchain {
            self.active_swapchain_epoch = self.present_thread.advance_swapchain_epoch();
        }
        if self.mode == FrameGenBackendMode::Off
            || self.active_swapchain == 0
            || swapchain_raw != self.active_swapchain
            || self.present_thread.current_epoch() != self.active_swapchain_epoch
        {
            return -1;
        }
        let swapchain_ext = ash::khr::swapchain::Device::new(&ctx.instance, &ctx.device);
        let sc = vk::SwapchainKHR::from_raw(swapchain_raw);
        let sem = vk::Semaphore::from_raw(semaphore_raw);
        let fence = vk::Fence::from_raw(fence_raw);
        match unsafe {
            with_queue_lock(|| swapchain_ext.acquire_next_image(sc, timeout_ns, sem, fence))
        } {
            Ok((idx, _suboptimal)) => idx as i32,
            Err(_) => {
                self.pending_present_job = None;
                self.reflex_sleep_armed = false;
                self.active_swapchain_epoch = self.present_thread.advance_swapchain_epoch();
                -1
            }
        }
    }

    pub fn intercept_queue_present(
        &mut self,
        ctx: &VulkanContext,
        queue_raw: u64,
        swapchain_raw: u64,
        image_index: u32,
        wait_semaphore_raw: u64,
    ) -> bool {
        if self.mode == FrameGenBackendMode::Off
            || self.active_swapchain == 0
            || swapchain_raw != self.active_swapchain
            || queue_raw == 0
        {
            self.pending_present_job = None;
            return false;
        }
        if self.present_thread.current_epoch() != self.active_swapchain_epoch {
            self.pending_present_job = None;
            return false;
        }
        self.set_reflex_marker(ctx, VK_LATENCY_MARKER_PRESENT_START_NV);

        let swapchain_ext = ash::khr::swapchain::Device::new(&ctx.instance, &ctx.device);
        let queue = vk::Queue::from_raw(queue_raw);
        let swapchains = [vk::SwapchainKHR::from_raw(swapchain_raw)];
        let indices = [image_index];
        let wait_sems = [vk::Semaphore::from_raw(wait_semaphore_raw)];

        let mut present_info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .image_indices(&indices);
        if wait_semaphore_raw != 0 {
            present_info = present_info.wait_semaphores(&wait_sems);
        }

        let res = with_queue_lock(|| unsafe { swapchain_ext.queue_present(queue, &present_info) });
        self.set_reflex_marker(ctx, VK_LATENCY_MARKER_PRESENT_END_NV);

        match res {
            Ok(suboptimal) => {
                if suboptimal {
                    self.pending_present_job = None;
                    self.reflex_sleep_armed = false;
                } else {
                    self.reflex_sleep_armed = true;
                    if let Some(mut job) = self.pending_present_job.take()
                        && job.epoch == self.present_thread.current_epoch()
                    {
                        job.start_time = Instant::now();
                        self.present_thread.submit_job(job);
                    }
                }
                true
            }
            Err(_) => {
                // Swapchain is out-of-date; advance epoch and let RenderPearl recreate it.
                self.pending_present_job = None;
                self.reflex_sleep_armed = false;
                self.active_swapchain_epoch = self.present_thread.advance_swapchain_epoch();
                false
            }
        }
    }

    pub fn wait_present_idle(&self) {
        self.present_thread.wait_idle();
    }

    pub fn record_frame_pacing(&mut self, ctx: Option<&VulkanContext>) -> Duration {
        if self.current_frame_id == 0 {
            self.current_frame_id = 1;
        }
        if let Some(c) = ctx {
            self.set_reflex_marker(c, VK_LATENCY_MARKER_RENDERSUBMIT_START_NV);
        }

        let now = Instant::now();
        if let Some(prev) = self.last_present_timestamp {
            let dt = now.saturating_duration_since(prev);
            let prev_ns = self.smoothed_frame_delta.as_nanos() as u64;
            let cur_ns = dt.as_nanos().min(100_000_000) as u64;
            let blended = (prev_ns * 7 + cur_ns) / 8;
            self.smoothed_frame_delta = Duration::from_nanos(blended);
        }
        self.last_present_timestamp = Some(now);
        self.reflex_sleep_armed = true;
        self.smoothed_frame_delta / 2
    }

    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_and_intercept_blit(
        &mut self,
        ctx: Option<&VulkanContext>,
        cmd_raw: u64,
        src_image: u64,
        src_layout: i32,
        dst_image: u64,
        dst_layout: i32,
        width: u32,
        height: u32,
    ) -> bool {
        let half_period = self.record_frame_pacing(ctx);
        if self.mode == FrameGenBackendMode::Off && !self.hdr_enabled {
            return false;
        }
        let Some(c) = ctx else {
            return false;
        };
        if cmd_raw == 0 || src_image == 0 || dst_image == 0 {
            return false;
        }

        // Ensure any previous background interpolated present has completed before touching the swapchain
        self.present_thread.wait_idle();

        // Never raw-blit sRGB into an active HDR scRGB/HDR10 swapchain and never reuse a stale
        // `hdr_output_image` from a previous frame/world.
        if self.hdr_enabled && (!self.hdr_output_valid_this_frame || self.hdr_output_image == 0) {
            self.pending_present_job = None;
            return false;
        }

        // For the main frame blit into the acquired swapchain image:
        // - When HDR is enabled, blit from `hdr_output_image` (`RES_SWAPCHAIN_OUT`, produced by
        //   `hdr_compose.comp` in `GENERAL` layout).
        // - Otherwise, blit the real frame `src_image` (`mainColorTexture` with GUI, already in
        //   `src_layout = TRANSFER_SRC_OPTIMAL`), while staging the interpolated frame
        //   (`interpolated_image`) in `pending_present_job` to be presented by `PresentThread`
        //   after the main frame's `vkQueuePresentKHR` completes!
        let present_src = if self.hdr_enabled {
            self.hdr_output_valid_this_frame = false;
            self.hdr_output_image
        } else {
            src_image
        };
        let dst_w = if self.swapchain_width > 0 {
            self.swapchain_width
        } else {
            width
        };
        let dst_h = if self.swapchain_height > 0 {
            self.swapchain_height
        } else {
            height
        };
        let cmd = vk::CommandBuffer::from_raw(cmd_raw);
        let subresource = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let region = [vk::ImageBlit {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_offsets: [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: width as i32,
                    y: height as i32,
                    z: 1,
                },
            ],
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            dst_offsets: [
                vk::Offset3D {
                    x: 0,
                    y: dst_h as i32,
                    z: 0,
                },
                vk::Offset3D {
                    x: dst_w as i32,
                    y: 0,
                    z: 1,
                },
            ],
        }];

        unsafe {
            if present_src != src_image {
                let pre = [vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::BLIT)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(vk::Image::from_raw(present_src))
                    .subresource_range(subresource)];
                let dep = vk::DependencyInfo::default().image_memory_barriers(&pre);
                c.sync2_ext.cmd_pipeline_barrier2(cmd, &dep);
            }

            c.device.cmd_blit_image(
                cmd,
                vk::Image::from_raw(present_src),
                if present_src != src_image {
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                } else {
                    vk::ImageLayout::from_raw(src_layout)
                },
                vk::Image::from_raw(dst_image),
                vk::ImageLayout::from_raw(dst_layout),
                &region,
                vk::Filter::LINEAR,
            );

            if present_src != src_image {
                let post = [vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::BLIT)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(vk::Image::from_raw(present_src))
                    .subresource_range(subresource)];
                let dep = vk::DependencyInfo::default().image_memory_barriers(&post);
                c.sync2_ext.cmd_pipeline_barrier2(cmd, &dep);
            }
        }

        self.set_reflex_marker(c, VK_LATENCY_MARKER_RENDERSUBMIT_END_NV);

        // Stage the PresentThread job so it is launched in `intercept_queue_present` AFTER the main
        // frame's command buffer has been submitted and presented.
        let cur_epoch = self.present_thread.current_epoch();
        if self.mode != FrameGenBackendMode::Off
            && self.active_swapchain != 0
            && !self.swapchain_images.is_empty()
            && self.interpolated_image != 0
            && cur_epoch == self.active_swapchain_epoch
        {
            self.pending_present_job = Some(PresentJob {
                epoch: cur_epoch,
                device_raw: c.device.handle().as_raw(),
                instance_raw: c.instance.handle().as_raw(),
                queue_raw: c.graphics_queue.as_raw(),
                queue_family_index: c.graphics_queue_family_index,
                swapchain_raw: self.active_swapchain,
                swapchain_images: self.swapchain_images.clone(),
                generated_images: vec![self.interpolated_image],
                width,
                height,
                swapchain_width: dst_w,
                swapchain_height: dst_h,
                start_time: Instant::now(),
                frame_spacing: half_period,
            });
        } else {
            self.pending_present_job = None;
        }

        true
    }

    pub fn destroy(&mut self, ctx: &VulkanContext) {
        self.pending_present_job = None;
        self.present_thread.stop();
        if self.latency_sleep_semaphore != vk::Semaphore::null() {
            unsafe {
                ctx.device
                    .destroy_semaphore(self.latency_sleep_semaphore, None);
            }
            self.latency_sleep_semaphore = vk::Semaphore::null();
        }
    }
}
