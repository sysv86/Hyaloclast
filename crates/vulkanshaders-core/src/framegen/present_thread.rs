use crate::gpu::{device_merger, with_queue_lock};
use ash::{
    Device, Instance,
    khr::{swapchain, synchronization2},
    vk::{self, Handle},
};
use parking_lot::{Condvar, Mutex};
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const SPIN_TAIL: Duration = Duration::from_micros(1500);
const FRAMES_IN_FLIGHT: usize = 3;

#[derive(Debug, Clone)]
pub struct PresentJob {
    pub epoch: u64,
    pub device_raw: u64,
    pub instance_raw: u64,
    pub queue_raw: u64,
    pub queue_family_index: u32,
    pub swapchain_raw: u64,
    pub swapchain_images: Vec<u64>,
    pub generated_images: Vec<u64>,
    pub width: u32,
    pub height: u32,
    pub swapchain_width: u32,
    pub swapchain_height: u32,
    pub start_time: Instant,
    pub frame_spacing: Duration,
}

#[derive(Debug, Default)]
struct SharedPresentState {
    pending_job: Option<PresentJob>,
    busy: bool,
    stopping: bool,
    flush_gpu_resources: bool,
    last_status: i32,
    presented_count: u64,
}

struct PersistentPresentSlot {
    cmd_pool: vk::CommandPool,
    cmd_buf: vk::CommandBuffer,
    acquire_sem: vk::Semaphore,
    render_sem: vk::Semaphore,
    in_flight_fence: vk::Fence,
    submitted_once: bool,
}

struct WorkerGpuCache {
    instance_raw: u64,
    device_raw: u64,
    queue_family_index: u32,
    instance: Instance,
    device: Device,
    swapchain_ext: swapchain::Device,
    sync2_ext: synchronization2::Device,
    slots: Vec<PersistentPresentSlot>,
    slot_cursor: usize,
}

impl WorkerGpuCache {
    unsafe fn create(
        instance_raw: u64,
        device_raw: u64,
        queue_family_index: u32,
    ) -> Result<Self, vk::Result> {
        let entry = device_merger::load_vulkan_entry();
        let instance =
            unsafe { Instance::load(entry.static_fn(), vk::Instance::from_raw(instance_raw)) };
        let device = unsafe { Device::load(instance.fp_v1_0(), vk::Device::from_raw(device_raw)) };
        let swapchain_ext = swapchain::Device::new(&instance, &device);
        let sync2_ext = synchronization2::Device::new(&instance, &device);

        let mut slots = Vec::with_capacity(FRAMES_IN_FLIGHT);
        let sem_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(
                vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )
            .queue_family_index(queue_family_index);

        let cleanup_slots = |dev: &Device, created: Vec<PersistentPresentSlot>| unsafe {
            for slot in created {
                if slot.cmd_pool != vk::CommandPool::null() {
                    dev.destroy_command_pool(slot.cmd_pool, None);
                }
                if slot.in_flight_fence != vk::Fence::null() {
                    dev.destroy_fence(slot.in_flight_fence, None);
                }
                if slot.render_sem != vk::Semaphore::null() {
                    dev.destroy_semaphore(slot.render_sem, None);
                }
                if slot.acquire_sem != vk::Semaphore::null() {
                    dev.destroy_semaphore(slot.acquire_sem, None);
                }
            }
        };

        for _ in 0..FRAMES_IN_FLIGHT {
            let cmd_pool = match unsafe { device.create_command_pool(&pool_info, None) } {
                Ok(p) => p,
                Err(e) => {
                    cleanup_slots(&device, slots);
                    return Err(e);
                }
            };
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmds = match unsafe { device.allocate_command_buffers(&alloc_info) } {
                Ok(c) => c,
                Err(e) => {
                    unsafe { device.destroy_command_pool(cmd_pool, None) };
                    cleanup_slots(&device, slots);
                    return Err(e);
                }
            };
            let acquire_sem = match unsafe { device.create_semaphore(&sem_info, None) } {
                Ok(s) => s,
                Err(e) => {
                    unsafe { device.destroy_command_pool(cmd_pool, None) };
                    cleanup_slots(&device, slots);
                    return Err(e);
                }
            };
            let render_sem = match unsafe { device.create_semaphore(&sem_info, None) } {
                Ok(s) => s,
                Err(e) => {
                    unsafe {
                        device.destroy_semaphore(acquire_sem, None);
                        device.destroy_command_pool(cmd_pool, None);
                    }
                    cleanup_slots(&device, slots);
                    return Err(e);
                }
            };
            let in_flight_fence = match unsafe { device.create_fence(&fence_info, None) } {
                Ok(f) => f,
                Err(e) => {
                    unsafe {
                        device.destroy_semaphore(render_sem, None);
                        device.destroy_semaphore(acquire_sem, None);
                        device.destroy_command_pool(cmd_pool, None);
                    }
                    cleanup_slots(&device, slots);
                    return Err(e);
                }
            };
            slots.push(PersistentPresentSlot {
                cmd_pool,
                cmd_buf: cmds[0],
                acquire_sem,
                render_sem,
                in_flight_fence,
                submitted_once: false,
            });
        }

        Ok(Self {
            instance_raw,
            device_raw,
            queue_family_index,
            instance,
            device,
            swapchain_ext,
            sync2_ext,
            slots,
            slot_cursor: 0,
        })
    }

    unsafe fn destroy(&mut self) {
        let _ = self.instance.handle();
        for slot in self.slots.drain(..) {
            unsafe {
                if slot.submitted_once && slot.in_flight_fence != vk::Fence::null() {
                    let _ = self
                        .device
                        .wait_for_fences(&[slot.in_flight_fence], true, 100_000_000);
                }
                if slot.cmd_pool != vk::CommandPool::null() {
                    self.device.destroy_command_pool(slot.cmd_pool, None);
                }
                if slot.in_flight_fence != vk::Fence::null() {
                    self.device.destroy_fence(slot.in_flight_fence, None);
                }
                if slot.render_sem != vk::Semaphore::null() {
                    self.device.destroy_semaphore(slot.render_sem, None);
                }
                if slot.acquire_sem != vk::Semaphore::null() {
                    self.device.destroy_semaphore(slot.acquire_sem, None);
                }
            }
        }
    }
}

/// Asynchronous background present thread (`PresentThread`) for Frame Generation.
///
/// Reuses persistent per-frame-in-flight `VkCommandPool`, `VkCommandBuffer`, `VkSemaphore`s,
/// and `VkFence`s, uses the real `graphics_queue_family_index` from `VulkanContext`,
/// synchronizes swapchain recreation/resize via a lock-free `swapchain_epoch` atomic counter so
/// `VK_SUBOPTIMAL_KHR` / `VK_ERROR_OUT_OF_DATE_KHR` under Wayland/FIFO immediately aborts stale
/// jobs without racing `RenderPearl`, and serializes all queue operations via `gpu::with_queue_lock`.
pub struct PresentThread {
    state: Arc<(Mutex<SharedPresentState>, Condvar, Condvar)>,
    swapchain_epoch: Arc<AtomicU64>,
    worker: Option<JoinHandle<()>>,
}

impl Default for PresentThread {
    fn default() -> Self {
        Self::new()
    }
}

impl PresentThread {
    pub fn new() -> Self {
        Self {
            state: Arc::new((
                Mutex::new(SharedPresentState::default()),
                Condvar::new(),
                Condvar::new(),
            )),
            swapchain_epoch: Arc::new(AtomicU64::new(1)),
            worker: None,
        }
    }

    /// Returns the current swapchain epoch token.
    pub fn current_epoch(&self) -> u64 {
        self.swapchain_epoch.load(Ordering::Acquire)
    }

    /// Atomically advances the swapchain epoch and drops any queued present job so an in-flight
    /// background sleep wakes and aborts before touching the old `VkSwapchainKHR`.
    pub fn advance_swapchain_epoch(&self) -> u64 {
        let next = self.swapchain_epoch.fetch_add(1, Ordering::AcqRel) + 1;
        let (lock, wake_cv, _idle_cv) = &*self.state;
        {
            let mut guard = lock.lock();
            guard.pending_job = None;
        }
        wake_cv.notify_all();
        next
    }

    pub fn last_status(&self) -> i32 {
        let (lock, _, _) = &*self.state;
        lock.lock().last_status
    }

    pub fn wait_idle(&self) {
        let (lock, _wake_cv, idle_cv) = &*self.state;
        let mut guard = lock.lock();
        let timeout = Duration::from_secs(2);
        while guard.pending_job.is_some() || guard.busy {
            if idle_cv.wait_for(&mut guard, timeout).timed_out() {
                tracing::warn!(
                    busy = guard.busy,
                    has_pending_job = guard.pending_job.is_some(),
                    "present_thread wait_idle timed out after {:?}; present may be stalled",
                    timeout,
                );
            }
        }
    }

    pub fn flush_gpu_resources(&self) {
        self.advance_swapchain_epoch();
        let (lock, wake_cv, idle_cv) = &*self.state;
        {
            let mut guard = lock.lock();
            guard.pending_job = None;
            guard.flush_gpu_resources = true;
        }
        wake_cv.notify_one();
        let mut guard = lock.lock();
        while (guard.flush_gpu_resources || guard.busy) && self.worker.is_some() {
            idle_cv.wait_for(&mut guard, Duration::from_millis(50));
            if self.worker.is_none() {
                guard.flush_gpu_resources = false;
                guard.busy = false;
                break;
            }
        }
    }

    /// Returns `false` on macOS/MoltenVK where secondary-thread swapchain present is unsafe.
    pub const fn is_supported_on_platform() -> bool {
        !cfg!(target_os = "macos")
    }

    pub fn submit_job(&mut self, job: PresentJob) {
        if !Self::is_supported_on_platform()
            || crate::crash::global_device_lost_guard().is_rt_vsp_disabled()
        {
            return;
        }
        if self.worker.is_none() {
            let state_clone = Arc::clone(&self.state);
            let epoch_clone = Arc::clone(&self.swapchain_epoch);
            if let Ok(handle) = thread::Builder::new()
                .name("hyaloclast-present".to_owned())
                .spawn(move || Self::worker_loop(state_clone, epoch_clone))
            {
                self.worker = Some(handle);
            } else {
                return;
            }
        }
        let (lock, wake_cv, _idle_cv) = &*self.state;
        {
            let mut guard = lock.lock();
            guard.pending_job = Some(job);
        }
        wake_cv.notify_one();
    }

    pub fn stop(&mut self) {
        self.advance_swapchain_epoch();
        if let Some(handle) = self.worker.take() {
            let (lock, wake_cv, idle_cv) = &*self.state;
            {
                let mut guard = lock.lock();
                guard.stopping = true;
                guard.pending_job = None;
            }
            wake_cv.notify_one();
            let _ = handle.join();
            {
                let mut guard = lock.lock();
                guard.busy = false;
            }
            idle_cv.notify_all();
        }
    }

    fn sleep_until_precise_or_epoch_changed(
        target: Instant,
        epoch_ref: &AtomicU64,
        expected_epoch: u64,
    ) -> bool {
        if epoch_ref.load(Ordering::Acquire) != expected_epoch {
            return false;
        }
        let now = Instant::now();
        if now >= target {
            return epoch_ref.load(Ordering::Acquire) == expected_epoch;
        }
        if let Some(coarse) = target.checked_sub(SPIN_TAIL) {
            while Instant::now() < coarse {
                if epoch_ref.load(Ordering::Acquire) != expected_epoch {
                    return false;
                }
                let rem = coarse.saturating_duration_since(Instant::now());
                thread::sleep(rem.min(Duration::from_micros(500)));
            }
        }
        while Instant::now() < target {
            if epoch_ref.load(Ordering::Acquire) != expected_epoch {
                return false;
            }
            thread::yield_now();
        }
        epoch_ref.load(Ordering::Acquire) == expected_epoch
    }

    fn worker_loop(
        state: Arc<(Mutex<SharedPresentState>, Condvar, Condvar)>,
        swapchain_epoch: Arc<AtomicU64>,
    ) {
        let (lock, wake_cv, idle_cv) = &*state;
        let mut gpu_cache: Option<WorkerGpuCache> = None;

        loop {
            let (job, should_flush) = {
                let mut guard = lock.lock();
                while guard.pending_job.is_none() && !guard.flush_gpu_resources && !guard.stopping {
                    wake_cv.wait(&mut guard);
                }
                if guard.stopping {
                    guard.busy = false;
                    drop(guard);
                    if let Some(mut cache) = gpu_cache.take() {
                        with_queue_lock(|| unsafe {
                            cache.destroy();
                        });
                    }
                    return;
                }
                let do_flush = guard.flush_gpu_resources;
                guard.flush_gpu_resources = false;
                let taken = guard.pending_job.take();
                guard.busy = taken.is_some() || do_flush;
                (taken, do_flush)
            };

            if should_flush {
                if let Some(mut cache) = gpu_cache.take() {
                    with_queue_lock(|| unsafe {
                        cache.destroy();
                    });
                }
                if job.is_none() {
                    let mut guard = lock.lock();
                    guard.busy = false;
                }
                idle_cv.notify_all();
            }

            let Some(job) = job else {
                continue;
            };

            let mut status = 0i32;
            let mut count = 0u64;
            for (idx, &src_img_raw) in job.generated_images.iter().enumerate() {
                if swapchain_epoch.load(Ordering::Acquire) != job.epoch {
                    status = -2;
                    break;
                }
                let due = job.start_time + job.frame_spacing.saturating_mul((idx + 1) as u32);
                if !Self::sleep_until_precise_or_epoch_changed(due, &swapchain_epoch, job.epoch) {
                    status = -2;
                    break;
                }

                let res = with_queue_lock(|| {
                    if swapchain_epoch.load(Ordering::Acquire) != job.epoch {
                        return -2;
                    }
                    Self::present_single_frame(&mut gpu_cache, &job, src_img_raw)
                });
                if res != 0 {
                    // On VK_SUBOPTIMAL_KHR (1) or VK_ERROR_OUT_OF_DATE_KHR (-1), advance the
                    // swapchain epoch so no further background presents race RenderPearl's resize.
                    if res == 1 || res == -1 {
                        swapchain_epoch.fetch_add(1, Ordering::AcqRel);
                    }
                    status = res;
                    break;
                }
                count += 1;
            }

            {
                let mut guard = lock.lock();
                guard.busy = false;
                guard.last_status = status;
                guard.presented_count += count;
            }
            idle_cv.notify_all();
        }
    }

    fn present_single_frame(
        gpu_cache: &mut Option<WorkerGpuCache>,
        job: &PresentJob,
        src_img_raw: u64,
    ) -> i32 {
        if job.swapchain_raw == 0
            || job.device_raw == 0
            || job.instance_raw == 0
            || job.queue_raw == 0
            || src_img_raw == 0
        {
            return -1;
        }

        let needs_recreate = gpu_cache.as_ref().is_none_or(|c| {
            c.device_raw != job.device_raw
                || c.instance_raw != job.instance_raw
                || c.queue_family_index != job.queue_family_index
        });

        if needs_recreate {
            if let Some(mut old) = gpu_cache.take() {
                unsafe {
                    old.destroy();
                }
            }
            match unsafe {
                WorkerGpuCache::create(job.instance_raw, job.device_raw, job.queue_family_index)
            } {
                Ok(new_cache) => *gpu_cache = Some(new_cache),
                Err(_) => return -1,
            }
        }

        let cache = gpu_cache.as_mut().unwrap();
        let slot_idx = cache.slot_cursor % cache.slots.len();
        cache.slot_cursor = cache.slot_cursor.wrapping_add(1);
        let slot = &mut cache.slots[slot_idx];

        let swapchain_khr = vk::SwapchainKHR::from_raw(job.swapchain_raw);
        let queue = vk::Queue::from_raw(job.queue_raw);
        let src_image = vk::Image::from_raw(src_img_raw);

        unsafe {
            // 1. Wait for this slot's previous submission (from 3 frames ago) and reset its command pool.
            // Do NOT reset in_flight_fence until acquire succeeds and we are about to vkQueueSubmit.
            if let Err(err) =
                cache
                    .device
                    .wait_for_fences(&[slot.in_flight_fence], true, 100_000_000)
            {
                crate::crash::global_device_lost_guard()
                    .record_vk_result("PresentThread::wait_for_fences", err);
                return -1;
            }
            if cache
                .device
                .reset_command_pool(slot.cmd_pool, vk::CommandPoolResetFlags::empty())
                .is_err()
            {
                return -1;
            }

            // 2. Acquire next swapchain image using a bounded 8 ms timeout so Wayland/FIFO stalls
            // or resize transitions never block the main render thread's queue lock.
            let acquire_res = cache.swapchain_ext.acquire_next_image(
                swapchain_khr,
                8_000_000,
                slot.acquire_sem,
                vk::Fence::null(),
            );
            let (image_index, suboptimal) = match acquire_res {
                Ok(pair) => pair,
                Err(err) => {
                    crate::crash::global_device_lost_guard()
                        .record_vk_result("PresentThread::acquire_next_image", err);
                    return -1;
                }
            };

            let Some(&dst_img_raw) = job.swapchain_images.get(image_index as usize) else {
                return -1;
            };
            let dst_image = vk::Image::from_raw(dst_img_raw);
            let cmd = slot.cmd_buf;

            // 3. Record barrier + blit + present barrier into the persistent command buffer
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            let _ = cache.device.begin_command_buffer(cmd, &begin_info);

            let subresource = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };

            let pre_barriers = [
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .src_access_mask(vk::AccessFlags2::empty())
                    .dst_stage_mask(vk::PipelineStageFlags2::BLIT)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(dst_image)
                    .subresource_range(subresource),
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::BLIT)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(src_image)
                    .subresource_range(subresource),
            ];
            let pre_dep = vk::DependencyInfo::default().image_memory_barriers(&pre_barriers);
            cache.sync2_ext.cmd_pipeline_barrier2(cmd, &pre_dep);

            let src_w = job.width.max(1);
            let src_h = job.height.max(1);
            let dst_w = if job.swapchain_width > 0 {
                job.swapchain_width
            } else {
                src_w
            }
            .max(1);
            let dst_h = if job.swapchain_height > 0 {
                job.swapchain_height
            } else {
                src_h
            }
            .max(1);
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
                        x: src_w as i32,
                        y: src_h as i32,
                        z: 1,
                    },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                // RenderPearl.VulkanGpuSurface presents with a Y-flip: the destination blit range
                // runs from (0, swapchain_height) down to (swapchain_width, 0).
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
            cache.device.cmd_blit_image(
                cmd,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &region,
                vk::Filter::LINEAR,
            );

            let post_barriers = [
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::BLIT)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                    .dst_access_mask(vk::AccessFlags2::empty())
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .image(dst_image)
                    .subresource_range(subresource),
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::BLIT)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(src_image)
                    .subresource_range(subresource),
            ];
            let post_dep = vk::DependencyInfo::default().image_memory_barriers(&post_barriers);
            cache.sync2_ext.cmd_pipeline_barrier2(cmd, &post_dep);

            let _ = cache.device.end_command_buffer(cmd);

            // 4. Submit and Present using the persistent slot semaphores and fence
            let wait_sems = [slot.acquire_sem];
            let wait_stages = [vk::PipelineStageFlags::TRANSFER];
            let signal_sems = [slot.render_sem];
            let submit_cmds = [cmd];
            let submit_info = [vk::SubmitInfo::default()
                .wait_semaphores(&wait_sems)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&submit_cmds)
                .signal_semaphores(&signal_sems)];

            if cache.device.reset_fences(&[slot.in_flight_fence]).is_err() {
                return -1;
            }
            if let Err(err) = cache
                .device
                .queue_submit(queue, &submit_info, slot.in_flight_fence)
            {
                crate::crash::global_device_lost_guard()
                    .record_vk_result("PresentThread::queue_submit", err);
                // Recreate the unsignaled fence as SIGNALED and replace acquire_sem so
                // subsequent frames do not stall on a 100ms fence timeout.
                slot.submitted_once = false;
                let fence_info =
                    vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
                let sem_info = vk::SemaphoreCreateInfo::default();
                let new_fence = cache.device.create_fence(&fence_info, None);
                let new_sem = cache.device.create_semaphore(&sem_info, None);
                if let (Ok(f), Ok(s)) = (new_fence, new_sem) {
                    cache.device.destroy_fence(slot.in_flight_fence, None);
                    cache.device.destroy_semaphore(slot.acquire_sem, None);
                    slot.in_flight_fence = f;
                    slot.acquire_sem = s;
                } else {
                    if let Ok(f) = new_fence {
                        cache.device.destroy_fence(f, None);
                    }
                    if let Ok(s) = new_sem {
                        cache.device.destroy_semaphore(s, None);
                    }
                    // Force cache recreation on the next frame.
                    cache.device_raw = 0;
                }
                return -1;
            }
            slot.submitted_once = true;

            let swapchains = [swapchain_khr];
            let image_indices = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_sems)
                .swapchains(&swapchains)
                .image_indices(&image_indices);
            let present_res = cache.swapchain_ext.queue_present(queue, &present_info);

            match present_res {
                Ok(present_suboptimal) if suboptimal || present_suboptimal => 1,
                Ok(_) => 0,
                Err(err) => {
                    crate::crash::global_device_lost_guard()
                        .record_vk_result("PresentThread::queue_present", err);
                    -1
                }
            }
        }
    }
}

impl Drop for PresentThread {
    fn drop(&mut self) {
        self.stop();
    }
}
