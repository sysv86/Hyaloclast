use crate::gpu::buffer::{GpuBufferAllocation, GpuMemoryAllocator, PooledImageAllocation};
use ash::vk::{self, Handle};
use gpu_allocator::vulkan::Allocation;
use parking_lot::Mutex;

type RetireClosure = Box<dyn FnOnce() + Send + 'static>;

struct TimedRetireItem {
    timeline_value: u64,
    task: RetireClosure,
}

struct RetireInner {
    slots: Vec<Vec<RetireClosure>>,
    slot_fences: Vec<vk::Fence>,
    slot_timeline_values: Vec<u64>,
    timed_queue: Vec<TimedRetireItem>,
    orphans: Vec<RetireClosure>,
    open_slot: usize,
    last_submitted_timeline: u64,
    completed_timeline_value: u64,
    timeline_semaphore: vk::Semaphore,
    configured: bool,
}

/// Deferred GPU resource destruction queue (`RetireQueue`) backed by GPU timeline semaphores
/// (`VK_KHR_timeline_semaphore`) and per-slot `VkFence` checks.
///
/// Guarantees no `VkAccelerationStructureKHR`, scratch `VkBuffer`, `VkImageView`, or descriptor
/// slot dropped on frame `N` is destroyed until the GPU has genuinely completed executing the
/// submission referencing that resource.
pub struct RetireQueue {
    inner: Mutex<RetireInner>,
}

impl Default for RetireQueue {
    fn default() -> Self {
        Self::new(3)
    }
}

impl RetireQueue {
    pub fn new(frames_in_flight: usize) -> Self {
        let count = frames_in_flight.max(1);
        let mut slots = Vec::with_capacity(count);
        slots.resize_with(count, Vec::new);
        Self {
            inner: Mutex::new(RetireInner {
                slots,
                slot_fences: vec![vk::Fence::null(); count],
                slot_timeline_values: vec![0; count],
                timed_queue: Vec::new(),
                orphans: Vec::new(),
                open_slot: 0,
                last_submitted_timeline: 0,
                completed_timeline_value: 0,
                timeline_semaphore: vk::Semaphore::null(),
                configured: true,
            }),
        }
    }

    pub fn configure(&self, frames_in_flight: usize) {
        let mut guard = self.inner.lock();
        let inner = &mut *guard;
        for slot in &mut inner.slots {
            inner.orphans.append(slot);
        }
        let count = frames_in_flight.max(1);
        guard.slots.clear();
        guard.slots.resize_with(count, Vec::new);
        guard.slot_fences.resize(count, vk::Fence::null());
        guard.slot_timeline_values.resize(count, 0);
        guard.open_slot = 0;
        guard.configured = true;
    }

    /// Binds a Vulkan timeline semaphore (`RenderPearl`'s `VulkanCommandEncoder.submitSemaphore`
    /// or an engine-owned timeline semaphore) so `poll_gpu_completion` can query the hardware
    /// counter via `vkGetSemaphoreCounterValue`.
    pub fn bind_timeline_semaphore(&self, semaphore_raw: u64) {
        let mut guard = self.inner.lock();
        guard.timeline_semaphore = vk::Semaphore::from_raw(semaphore_raw);
    }

    /// Associates a `VkFence` with `slot_index` so `open_slot_verified` can verify GPU completion.
    pub fn set_slot_fence(&self, slot_index: usize, fence: vk::Fence) {
        let mut guard = self.inner.lock();
        if !guard.slot_fences.is_empty() {
            let idx = slot_index % guard.slot_fences.len();
            guard.slot_fences[idx] = fence;
        }
    }

    /// Records the timeline value submitted for the currently open frame slot.
    pub fn record_submission(&self, timeline_value: u64) {
        if timeline_value == 0 || timeline_value == u64::MAX {
            return;
        }
        let mut guard = self.inner.lock();
        guard.last_submitted_timeline = guard.last_submitted_timeline.max(timeline_value);
        let idx = guard.open_slot;
        guard.slot_timeline_values[idx] = guard.slot_timeline_values[idx].max(timeline_value);
    }

    /// Notifies the retire queue that the GPU timeline semaphore has reached `completed_value`
    /// and immediately drains all timed destructors up to that ticket.
    pub fn notify_timeline_completed(&self, completed_value: u64) {
        self.poll_gpu_timeline(None, Some(completed_value));
    }

    /// Schedules `destroy_fn` to execute once the currently open frame slot (and its GPU timeline
    /// ticket) completes on the GPU.
    pub fn retire_later<F>(&self, destroy_fn: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let mut guard = self.inner.lock();
        if guard.timeline_semaphore != vk::Semaphore::null() {
            let target_timeline = guard
                .last_submitted_timeline
                .max(guard.completed_timeline_value)
                .saturating_add(1);
            guard.timed_queue.push(TimedRetireItem {
                timeline_value: target_timeline,
                task: Box::new(destroy_fn),
            });
            return;
        }
        if !guard.configured || guard.slots.is_empty() {
            guard.orphans.push(Box::new(destroy_fn));
            return;
        }
        let idx = guard.open_slot % guard.slots.len();
        guard.slots[idx].push(Box::new(destroy_fn));
    }

    /// Schedules `destroy_fn` to execute once the GPU timeline semaphore reaches `timeline_value`.
    pub fn retire_at_timeline<F>(&self, timeline_value: u64, destroy_fn: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let mut guard = self.inner.lock();
        guard.timed_queue.push(TimedRetireItem {
            timeline_value,
            task: Box::new(destroy_fn),
        });
    }

    /// Queries the bound timeline semaphore (or uses `completed_value_override`) and executes
    /// all timed destructors whose `timeline_value <= completed`.
    pub fn poll_gpu_timeline(&self, device: Option<&ash::Device>, completed_override: Option<u64>) {
        let ready: Vec<RetireClosure> = {
            let mut guard = self.inner.lock();
            if let Some(val) = completed_override {
                guard.completed_timeline_value = guard.completed_timeline_value.max(val);
            } else if let Some(dev) = device
                && guard.timeline_semaphore != vk::Semaphore::null()
                && let Ok(hw_val) =
                    unsafe { dev.get_semaphore_counter_value(guard.timeline_semaphore) }
            {
                guard.completed_timeline_value = guard.completed_timeline_value.max(hw_val);
            }

            let completed = guard.completed_timeline_value;
            let mut drained = Vec::new();
            let mut i = 0;
            while i < guard.timed_queue.len() {
                if guard.timed_queue[i].timeline_value <= completed {
                    drained.push(guard.timed_queue.swap_remove(i).task);
                } else {
                    i += 1;
                }
            }
            drained
        };

        for task in ready {
            task();
        }
    }

    /// Advances to `slot_index`, checking any bound `VkFence` or timeline semaphore on `device`
    /// before executing deferred destructors queued during the previous cycle of that slot.
    pub fn open_slot_with_device(&self, slot_index: usize, device: Option<&ash::Device>) {
        self.poll_gpu_timeline(device, None);

        let ready = {
            let mut guard = self.inner.lock();
            if !guard.configured || guard.slots.is_empty() {
                return;
            }
            let idx = slot_index % guard.slots.len();
            guard.open_slot = idx;
            if guard.slots[idx].is_empty() {
                return;
            }

            if let Some(dev) = device {
                let fence = guard.slot_fences[idx];
                if fence != vk::Fence::null() && unsafe { dev.get_fence_status(fence) } != Ok(true)
                {
                    return;
                }
                if guard.timeline_semaphore != vk::Semaphore::null()
                    && guard.slot_timeline_values[idx] > guard.completed_timeline_value
                {
                    return;
                }
            }

            std::mem::take(&mut guard.slots[idx])
        };

        for task in ready {
            task();
        }
    }

    /// Advances to `slot_index` and executes all deferred destructors queued for that slot.
    pub fn open_slot(&self, slot_index: usize) {
        self.open_slot_with_device(slot_index, None);
    }

    /// Convenience helper: defers destruction of a `VkImageView` until the current frame slot
    /// completes on the GPU.
    pub fn retire_image_view(&self, device: ash::Device, view: vk::ImageView) {
        if view == vk::ImageView::null() {
            return;
        }
        self.retire_later(move || unsafe {
            device.destroy_image_view(view, None);
        });
    }

    /// Convenience helper: defers destruction of a `VkImage` + optional `VkDeviceMemory` +
    /// optional `VkImageView` + optional `VkSampler` until the current frame slot completes on GPU.
    pub fn retire_image_resources(
        &self,
        device: ash::Device,
        image: vk::Image,
        memory: vk::DeviceMemory,
        view: vk::ImageView,
        sampler: vk::Sampler,
    ) {
        self.retire_later(move || unsafe {
            if sampler != vk::Sampler::null() {
                device.destroy_sampler(sampler, None);
            }
            if view != vk::ImageView::null() {
                device.destroy_image_view(view, None);
            }
            if image != vk::Image::null() {
                device.destroy_image(image, None);
            }
            if memory != vk::DeviceMemory::null() {
                device.free_memory(memory, None);
            }
        });
    }

    /// Convenience helper: defers destruction of a `VkBuffer` + `VkDeviceMemory` until the
    /// current frame slot completes on the GPU.
    pub fn retire_buffer_resources(
        &self,
        device: ash::Device,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
    ) {
        if buffer == vk::Buffer::null() && memory == vk::DeviceMemory::null() {
            return;
        }
        self.retire_later(move || unsafe {
            if buffer != vk::Buffer::null() {
                device.destroy_buffer(buffer, None);
            }
            if memory != vk::DeviceMemory::null() {
                device.free_memory(memory, None);
            }
        });
    }

    /// Defers destruction of a [`GpuBufferAllocation`] (whether backed by `gpu-allocator` or
    /// standalone `VkDeviceMemory`) until the current frame slot completes on the GPU.
    pub fn retire_gpu_buffer(&self, device: ash::Device, mut buffer: GpuBufferAllocation) {
        if buffer.buffer == vk::Buffer::null() && buffer.memory == vk::DeviceMemory::null() {
            return;
        }
        self.retire_later(move || unsafe {
            buffer.destroy(&device);
        });
    }

    /// Defers destruction of a [`PooledImageAllocation`] (destroying its `VkImage` and returning
    /// its `Allocation` to `gpu-allocator`) until the current frame slot completes on the GPU.
    pub fn retire_pooled_image(&self, device: ash::Device, mut image: PooledImageAllocation) {
        self.retire_later(move || unsafe {
            image.destroy(&device);
        });
    }

    /// Defers returning a raw `gpu_allocator::vulkan::Allocation` to `allocator` until the
    /// current frame slot completes on the GPU.
    pub fn retire_allocation(&self, allocator: GpuMemoryAllocator, allocation: Allocation) {
        self.retire_later(move || {
            let _ = allocator.free(allocation);
        });
    }

    /// Returns whether either a timeline semaphore or any per-slot fence is currently bound.
    pub fn has_gpu_sync(&self) -> bool {
        let guard = self.inner.lock();
        guard.timeline_semaphore != vk::Semaphore::null()
            || guard.slot_fences.iter().any(|&f| f != vk::Fence::null())
    }

    /// Drains all pending destructors across all frame slots (called after `vkDeviceWaitIdle` on shutdown).
    pub fn drain_all(&self) {
        loop {
            let ready = {
                let mut guard = self.inner.lock();
                let mut batch = std::mem::take(&mut guard.orphans);
                for slot in &mut guard.slots {
                    batch.append(slot);
                }
                for item in guard.timed_queue.drain(..) {
                    batch.push(item.task);
                }
                batch
            };
            if ready.is_empty() {
                break;
            }
            for task in ready {
                task();
            }
        }
    }
}
