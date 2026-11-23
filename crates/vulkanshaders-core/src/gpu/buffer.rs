use crate::gpu::VulkanContext;
use ash::vk::{self, Handle};
use gpu_allocator::{
    AllocationError, AllocationSizes, AllocatorDebugSettings, MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
};
use parking_lot::Mutex;
use std::{
    collections::HashMap,
    ptr,
    sync::{Arc, OnceLock},
};

/// Shared, thread-safe wrapper around `gpu_allocator::vulkan::Allocator` (`gpu-allocator 0.28`).
///
/// Pools standalone `VkBuffer` and `VkImage` device memory allocations into larger driver blocks
/// so long sessions and heavy world streaming never exhaust `maxMemoryAllocationCount` (4096).
#[derive(Clone)]
pub struct GpuMemoryAllocator {
    inner: Arc<Mutex<Allocator>>,
    device_raw: u64,
}

impl std::fmt::Debug for GpuMemoryAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuMemoryAllocator")
            .field("device_raw", &self.device_raw)
            .finish()
    }
}

impl GpuMemoryAllocator {
    /// Creates a new `gpu-allocator` Vulkan allocator bound to `(instance, physical_device, device)`.
    pub fn new(
        instance: ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: ash::Device,
        buffer_device_address: bool,
        vram_bytes: u64,
    ) -> Result<Self, AllocationError> {
        let device_raw = device.handle().as_raw();
        const MB: u64 = 1024 * 1024;
        const GB: u64 = 1024 * MB;
        let allocation_sizes = if vram_bytes <= 8 * GB {
            AllocationSizes::new(32 * MB, 16 * MB)
                .with_max_device_memblock_size(64 * MB)
                .with_max_host_memblock_size(32 * MB)
        } else {
            AllocationSizes::default()
        };
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance,
            device,
            physical_device,
            debug_settings: AllocatorDebugSettings::default(),
            buffer_device_address,
            allocation_sizes,
        })?;
        Ok(Self {
            inner: Arc::new(Mutex::new(allocator)),
            device_raw,
        })
    }

    /// Creates a `GpuMemoryAllocator` directly from an active [`VulkanContext`].
    pub fn from_context(ctx: &VulkanContext) -> Result<Self, AllocationError> {
        Self::new(
            ctx.instance.clone(),
            ctx.physical_device,
            ctx.device.clone(),
            ctx.capabilities.buffer_device_address,
            ctx.capabilities.vram_bytes,
        )
    }

    /// Returns the underlying `Arc<Mutex<Allocator>>` for advanced callers.
    pub fn inner(&self) -> &Arc<Mutex<Allocator>> {
        &self.inner
    }

    /// Allocates a sub-allocated memory block from the underlying `gpu-allocator` pool.
    pub fn allocate(&self, desc: &AllocationCreateDesc<'_>) -> Result<Allocation, AllocationError> {
        self.inner.lock().allocate(desc)
    }

    /// Returns a pooled `Allocation` back to the underlying `gpu-allocator` pool.
    pub fn free(&self, allocation: Allocation) -> Result<(), AllocationError> {
        self.inner.lock().free(allocation)
    }

    /// Maps Vulkan `vk::MemoryPropertyFlags` to `gpu_allocator::MemoryLocation`.
    pub const fn memory_location_for_flags(flags: vk::MemoryPropertyFlags) -> MemoryLocation {
        if flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            if flags.contains(vk::MemoryPropertyFlags::HOST_CACHED) {
                MemoryLocation::GpuToCpu
            } else {
                MemoryLocation::CpuToGpu
            }
        } else {
            MemoryLocation::GpuOnly
        }
    }

    /// Allocates a `VkImage` backed by `gpu-allocator` pooled memory and binds it immediately.
    pub fn allocate_image(
        &self,
        device: &ash::Device,
        create_info: &vk::ImageCreateInfo<'_>,
        location: MemoryLocation,
        name: &str,
    ) -> Result<PooledImageAllocation, vk::Result> {
        let image = unsafe { device.create_image(create_info, None)? };
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let linear = create_info.tiling == vk::ImageTiling::LINEAR;
        let allocation = match self.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location,
            linear,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(alloc) => alloc,
            Err(_) => {
                unsafe { device.destroy_image(image, None) };
                return Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
            }
        };

        let memory = unsafe { allocation.memory() };
        let offset = allocation.offset();
        let size = allocation.size();
        if let Err(err) = unsafe { device.bind_image_memory(image, memory, offset) } {
            let _ = self.free(allocation);
            unsafe { device.destroy_image(image, None) };
            return Err(err);
        }

        Ok(PooledImageAllocation {
            image,
            memory,
            offset,
            size,
            allocation: Some(allocation),
            allocator: Some(self.clone()),
            device: Some(device.clone()),
        })
    }
}

/// Owned `VkImage` backed by a `gpu-allocator` pooled [`Allocation`].
///
/// Calling [`Self::destroy`] destroys the `VkImage` and returns the `Allocation` to the shared
/// `GpuMemoryAllocator`. Dropping without calling `destroy` also safely releases both handles.
pub struct PooledImageAllocation {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
    allocation: Option<Allocation>,
    allocator: Option<GpuMemoryAllocator>,
    device: Option<ash::Device>,
}

impl std::fmt::Debug for PooledImageAllocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledImageAllocation")
            .field("image", &self.image)
            .field("memory", &self.memory)
            .field("offset", &self.offset)
            .field("size", &self.size)
            .finish()
    }
}

unsafe impl Send for PooledImageAllocation {}
unsafe impl Sync for PooledImageAllocation {}

impl PooledImageAllocation {
    /// Destroys the `VkImage` and returns its pooled [`Allocation`] to `gpu-allocator`.
    ///
    /// # Safety
    /// Must only be called when no GPU command buffer referencing `self.image` is in flight.
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        if self.image != vk::Image::null() {
            unsafe { device.destroy_image(self.image, None) };
            self.image = vk::Image::null();
        }
        if let (Some(allocator), Some(allocation)) = (self.allocator.take(), self.allocation.take())
        {
            let _ = allocator.free(allocation);
        }
        self.memory = vk::DeviceMemory::null();
        self.offset = 0;
        self.size = 0;
        self.device = None;
    }
}

impl Drop for PooledImageAllocation {
    fn drop(&mut self) {
        if let Some(device) = self.device.take() {
            unsafe { self.destroy(&device) };
        } else if let (Some(allocator), Some(allocation)) =
            (self.allocator.take(), self.allocation.take())
        {
            let _ = allocator.free(allocation);
        }
    }
}

/// Registry mapping `(device_raw, buffer_raw)` -> `(GpuMemoryAllocator, Allocation)` for
/// `GpuBufferAllocation` instances allocated through `gpu-allocator`. This preserves
/// `GpuBufferAllocation`'s exact public field layout while guaranteeing clean, leak-free
/// deallocation back into `gpu-allocator` inside [`GpuBufferAllocation::destroy`].
type PooledBufferKey = (u64, u64);
type PooledBufferValue = (GpuMemoryAllocator, Allocation);
type PooledBufferRegistry = Mutex<HashMap<PooledBufferKey, PooledBufferValue>>;

static POOLED_BUFFER_REGISTRY: OnceLock<PooledBufferRegistry> = OnceLock::new();

fn pooled_buffer_registry() -> &'static PooledBufferRegistry {
    POOLED_BUFFER_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn register_pooled_buffer(
    device_raw: u64,
    buffer: vk::Buffer,
    allocator: GpuMemoryAllocator,
    allocation: Allocation,
) {
    if buffer != vk::Buffer::null() {
        pooled_buffer_registry()
            .lock()
            .insert((device_raw, buffer.as_raw()), (allocator, allocation));
    }
}

fn take_pooled_buffer(
    device_raw: u64,
    buffer: vk::Buffer,
) -> Option<(GpuMemoryAllocator, Allocation)> {
    if buffer == vk::Buffer::null() {
        return None;
    }
    pooled_buffer_registry()
        .lock()
        .remove(&(device_raw, buffer.as_raw()))
}

/// Returns the number of live `GpuBufferAllocation` handles currently backed by `gpu-allocator`.
pub fn active_pooled_buffer_count() -> usize {
    pooled_buffer_registry().lock().len()
}

/// Native Vulkan buffer + device memory allocation managed via `gpu-allocator` (when available on
/// `VulkanContext`) with a standalone `vkAllocateMemory` fallback for legacy callers, used for
/// Acceleration Structure storage slabs, scratch buffers, SBT tables, SHaRC hash grids,
/// ReSTIR DI reservoirs, and dynamic entity geometry.
#[derive(Debug)]
pub struct GpuBufferAllocation {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    pub device_address: vk::DeviceAddress,
    pub mapped_ptr: *mut u8,
}

unsafe impl Send for GpuBufferAllocation {}
unsafe impl Sync for GpuBufferAllocation {}

impl GpuBufferAllocation {
    pub const fn align_up(value: vk::DeviceSize, alignment: vk::DeviceSize) -> vk::DeviceSize {
        if alignment <= 1 {
            value
        } else {
            (value + alignment - 1) & !(alignment - 1)
        }
    }

    /// Allocates a `VkBuffer` and binds memory through `ctx.memory_allocator` (`gpu-allocator`)
    /// when available, falling back to a dedicated `vkAllocateMemory` block only if no pooled
    /// allocator is present or if pooled allocation fails.
    pub fn allocate(
        ctx: &VulkanContext,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
    ) -> Result<Self, vk::Result> {
        if let Some(ref allocator) = ctx.memory_allocator {
            match Self::allocate_pooled(
                ctx,
                allocator,
                size,
                usage,
                memory_properties,
                "hyaloclast_buffer",
            ) {
                Ok(buf) => return Ok(buf),
                Err(_) => {
                    // Fall through to legacy standalone allocation only if gpu-allocator cannot
                    // satisfy unusual memory requirements on this driver.
                }
            }
        }
        Self::allocate_standalone(ctx, size, usage, memory_properties)
    }

    /// Allocates a `VkBuffer` explicitly through a [`GpuMemoryAllocator`] (`gpu-allocator 0.28`),
    /// avoiding per-buffer `vkAllocateMemory` calls and `maxMemoryAllocationCount` exhaustion.
    pub fn allocate_pooled(
        ctx: &VulkanContext,
        allocator: &GpuMemoryAllocator,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
        name: &str,
    ) -> Result<Self, vk::Result> {
        let alloc_size = size.max(64);
        let create_info = vk::BufferCreateInfo::default()
            .size(alloc_size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { ctx.device.create_buffer(&create_info, None)? };
        let mem_reqs = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
        let location = GpuMemoryAllocator::memory_location_for_flags(memory_properties);

        let allocation = match allocator.allocate(&AllocationCreateDesc {
            name,
            requirements: mem_reqs,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(alloc) => alloc,
            Err(_) => {
                unsafe { ctx.device.destroy_buffer(buffer, None) };
                return Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
            }
        };

        let memory = unsafe { allocation.memory() };
        let offset = allocation.offset();
        if let Err(err) = unsafe { ctx.device.bind_buffer_memory(buffer, memory, offset) } {
            let _ = allocator.free(allocation);
            unsafe { ctx.device.destroy_buffer(buffer, None) };
            return Err(err);
        }

        let mapped_ptr = allocation
            .mapped_ptr()
            .map(|p| p.as_ptr().cast::<u8>())
            .unwrap_or(ptr::null_mut());

        let needs_bda = usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            && ctx.capabilities.buffer_device_address;
        let device_address = if needs_bda {
            unsafe { ctx.buffer_device_address(buffer) }
        } else {
            0
        };

        register_pooled_buffer(
            ctx.device.handle().as_raw(),
            buffer,
            allocator.clone(),
            allocation,
        );

        Ok(Self {
            buffer,
            memory,
            size: alloc_size,
            device_address,
            mapped_ptr,
        })
    }

    /// Convenience helper to allocate a pooled `VkImage` through `ctx.memory_allocator`.
    pub fn allocate_image(
        ctx: &VulkanContext,
        create_info: &vk::ImageCreateInfo<'_>,
        location: MemoryLocation,
        name: &str,
    ) -> Result<PooledImageAllocation, vk::Result> {
        let Some(ref allocator) = ctx.memory_allocator else {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        };
        allocator.allocate_image(&ctx.device, create_info, location, name)
    }

    /// Legacy fallback path: allocates a `VkBuffer` with a dedicated `vkAllocateMemory` block.
    fn allocate_standalone(
        ctx: &VulkanContext,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
    ) -> Result<Self, vk::Result> {
        let alloc_size = size.max(64);
        let create_info = vk::BufferCreateInfo::default()
            .size(alloc_size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { ctx.device.create_buffer(&create_info, None)? };
        let mem_reqs = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };

        let Some(memory_type_index) =
            Self::find_memory_type(ctx, mem_reqs.memory_type_bits, memory_properties)
        else {
            unsafe {
                ctx.device.destroy_buffer(buffer, None);
            }
            return Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        };

        let needs_bda = usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            && ctx.capabilities.buffer_device_address;
        let mut flags_info =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);

        let mut alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(memory_type_index);
        if needs_bda {
            alloc_info = alloc_info.push_next(&mut flags_info);
        }

        let memory = match unsafe { ctx.device.allocate_memory(&alloc_info, None) } {
            Ok(mem) => mem,
            Err(err) => {
                unsafe {
                    ctx.device.destroy_buffer(buffer, None);
                }
                return Err(err);
            }
        };

        if let Err(err) = unsafe { ctx.device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                ctx.device.free_memory(memory, None);
                ctx.device.destroy_buffer(buffer, None);
            }
            return Err(err);
        }

        let mapped_ptr = if memory_properties.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            unsafe {
                ctx.device
                    .map_memory(memory, 0, alloc_size, vk::MemoryMapFlags::empty())
                    .map(|p| p as *mut u8)
                    .unwrap_or(ptr::null_mut())
            }
        } else {
            ptr::null_mut()
        };

        let device_address = if needs_bda {
            unsafe { ctx.buffer_device_address(buffer) }
        } else {
            0
        };

        Ok(Self {
            buffer,
            memory,
            size: alloc_size,
            device_address,
            mapped_ptr,
        })
    }

    /// Allocates device-local storage for `VkAccelerationStructureKHR` (`256`-byte aligned).
    pub fn allocate_as_storage(
        ctx: &VulkanContext,
        size: vk::DeviceSize,
    ) -> Result<Self, vk::Result> {
        let aligned = Self::align_up(size.max(256), 256);
        Self::allocate(
            ctx,
            aligned,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
    }

    /// Allocates device-local scratch memory for `vkCmdBuildAccelerationStructuresKHR`.
    pub fn allocate_as_scratch(
        ctx: &VulkanContext,
        size: vk::DeviceSize,
    ) -> Result<Self, vk::Result> {
        let aligned = Self::align_up(size.max(256), ctx.min_scratch_alignment);
        Self::allocate(
            ctx,
            aligned,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
    }

    /// Allocates a device-local buffer with optional `SHADER_DEVICE_ADDRESS`.
    pub fn allocate_device_local_bda(
        ctx: &VulkanContext,
        size: vk::DeviceSize,
        extra_usage: vk::BufferUsageFlags,
    ) -> Result<Self, vk::Result> {
        let usage = if ctx.capabilities.buffer_device_address {
            extra_usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
        } else {
            extra_usage
        };
        Self::allocate(ctx, size, usage, vk::MemoryPropertyFlags::DEVICE_LOCAL)
    }

    /// Allocates host-visible, persistently mapped BDA buffer (for TLAS instances, SBT records, or dynamic entities).
    pub fn allocate_host_visible_bda(
        ctx: &VulkanContext,
        size: vk::DeviceSize,
        extra_usage: vk::BufferUsageFlags,
    ) -> Result<Self, vk::Result> {
        let usage = if ctx.capabilities.buffer_device_address {
            extra_usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
        } else {
            extra_usage
        };
        Self::allocate(
            ctx,
            size,
            usage,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }

    /// Copies `bytes` into the mapped buffer at `byte_offset`.
    pub fn write_bytes(&mut self, byte_offset: usize, bytes: &[u8]) -> bool {
        if self.mapped_ptr.is_null() || byte_offset + bytes.len() > self.size as usize {
            return false;
        }
        unsafe {
            ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.mapped_ptr.add(byte_offset),
                bytes.len(),
            );
        }
        true
    }

    /// Destroys the buffer and frees its backing memory immediately (returning pooled allocations
    /// cleanly to `gpu-allocator` without unmapping or double-freeing, or freeing standalone
    /// `VkDeviceMemory` for legacy allocations).
    ///
    /// # Safety
    /// Must only be called when no GPU command buffer referencing `self.buffer` is in flight
    /// (typically via `RetireQueue::retire_later`).
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        if let Some((allocator, allocation)) =
            take_pooled_buffer(device.handle().as_raw(), self.buffer)
        {
            if self.buffer != vk::Buffer::null() {
                unsafe {
                    device.destroy_buffer(self.buffer, None);
                }
                self.buffer = vk::Buffer::null();
            }
            let _ = allocator.free(allocation);
            self.memory = vk::DeviceMemory::null();
            self.mapped_ptr = ptr::null_mut();
            self.size = 0;
            self.device_address = 0;
            return;
        }

        if !self.mapped_ptr.is_null() && self.memory != vk::DeviceMemory::null() {
            unsafe {
                device.unmap_memory(self.memory);
            }
            self.mapped_ptr = ptr::null_mut();
        }
        if self.buffer != vk::Buffer::null() {
            unsafe {
                device.destroy_buffer(self.buffer, None);
            }
            self.buffer = vk::Buffer::null();
        }
        if self.memory != vk::DeviceMemory::null() {
            unsafe {
                device.free_memory(self.memory, None);
            }
            self.memory = vk::DeviceMemory::null();
        }
        self.size = 0;
        self.device_address = 0;
    }

    fn find_memory_type(
        ctx: &VulkanContext,
        type_filter: u32,
        required: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        let mem_props = unsafe {
            ctx.instance
                .get_physical_device_memory_properties(ctx.physical_device)
        };
        for i in 0..mem_props.memory_type_count {
            let is_supported = (type_filter & (1 << i)) != 0;
            let has_flags = mem_props.memory_types[i as usize]
                .property_flags
                .contains(required);
            if is_supported && has_flags {
                return Some(i);
            }
        }
        None
    }
}

#[derive(Debug, Clone, Copy)]
struct FreeSpan {
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
}

#[derive(Debug)]
struct SlabBlock {
    slab_id: u32,
    allocation: GpuBufferAllocation,
    free_spans: Vec<FreeSpan>,
}

/// Sub-allocated region inside a `SlabArenaAllocator` slab (or a standalone oversized allocation).
#[derive(Debug)]
pub struct SlabSubAllocation {
    pub buffer: vk::Buffer,
    pub offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
    pub device_address: vk::DeviceAddress,
    pub slab_id: Option<u32>,
    pub standalone: Option<GpuBufferAllocation>,
}

/// Coalescing slab sub-allocator for BLAS/TLAS storage and dynamic wind vertex buffers.
///
/// Sub-allocates `256`-byte aligned slices out of 64 MB `DEVICE_LOCAL` BDA slabs so thousands
/// of chunk section BLASes share a small number of `VkDeviceMemory` allocations and never hit
/// `VkPhysicalDeviceLimits::maxMemoryAllocationCount` (`4096`).
#[derive(Clone)]
pub struct SlabArenaAllocator {
    inner: Arc<Mutex<SlabArenaInner>>,
}

struct SlabArenaInner {
    slab_size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    next_slab_id: u32,
    slabs: Vec<SlabBlock>,
}

impl Default for SlabArenaAllocator {
    fn default() -> Self {
        Self::new_as_storage_pool()
    }
}

impl SlabArenaAllocator {
    pub const DEFAULT_SLAB_BYTES: vk::DeviceSize = 64 * 1024 * 1024; // 64 MB per slab

    pub fn new(slab_size: vk::DeviceSize, usage: vk::BufferUsageFlags) -> Self {
        Self {
            inner: Arc::new(Mutex::new(SlabArenaInner {
                slab_size: slab_size.max(4 * 1024 * 1024),
                usage,
                next_slab_id: 1,
                slabs: Vec::new(),
            })),
        }
    }

    pub fn new_as_storage_pool() -> Self {
        Self::new(
            Self::DEFAULT_SLAB_BYTES,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )
    }

    /// Sub-allocates an aligned region of `size` bytes from an existing 64 MB slab, or allocates
    /// a new slab if no existing slab has a contiguous span large enough.
    pub fn allocate_slice(
        &self,
        ctx: &VulkanContext,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
    ) -> Result<SlabSubAllocation, vk::Result> {
        let align = alignment.max(256);
        let req_size = GpuBufferAllocation::align_up(size.max(256), align);

        let mut guard = self.inner.lock();
        if guard.slabs.is_empty() && ctx.capabilities.vram_bytes <= 8 * 1024 * 1024 * 1024 {
            guard.slab_size = guard.slab_size.min(16 * 1024 * 1024);
        }
        if req_size > guard.slab_size / 2 {
            // Oversized allocation gets its own dedicated buffer
            let usage = guard.usage;
            drop(guard);
            let alloc = GpuBufferAllocation::allocate(
                ctx,
                req_size,
                usage,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            return Ok(SlabSubAllocation {
                buffer: alloc.buffer,
                offset: 0,
                size: req_size,
                device_address: alloc.device_address,
                slab_id: None,
                standalone: Some(alloc),
            });
        }

        // First-fit search across existing slabs
        for slab in &mut guard.slabs {
            for idx in 0..slab.free_spans.len() {
                let span = slab.free_spans[idx];
                let aligned_off = GpuBufferAllocation::align_up(span.offset, align);
                let pad = aligned_off - span.offset;
                if span.size >= pad + req_size {
                    let used_end = aligned_off + req_size;
                    let span_end = span.offset + span.size;
                    slab.free_spans.remove(idx);
                    if pad > 0 {
                        slab.free_spans.insert(
                            idx,
                            FreeSpan {
                                offset: span.offset,
                                size: pad,
                            },
                        );
                    }
                    if span_end > used_end {
                        let insert_pos = if pad > 0 { idx + 1 } else { idx };
                        slab.free_spans.insert(
                            insert_pos,
                            FreeSpan {
                                offset: used_end,
                                size: span_end - used_end,
                            },
                        );
                    }
                    return Ok(SlabSubAllocation {
                        buffer: slab.allocation.buffer,
                        offset: aligned_off,
                        size: req_size,
                        device_address: slab.allocation.device_address + aligned_off,
                        slab_id: Some(slab.slab_id),
                        standalone: None,
                    });
                }
            }
        }

        // Allocate a new 64 MB slab
        let slab_size = guard.slab_size;
        let usage = guard.usage;
        let slab_id = guard.next_slab_id;
        guard.next_slab_id += 1;

        let alloc = GpuBufferAllocation::allocate(
            ctx,
            slab_size,
            usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let buf = alloc.buffer;
        let bda = alloc.device_address;
        let mut free_spans = Vec::with_capacity(64);
        if slab_size > req_size {
            free_spans.push(FreeSpan {
                offset: req_size,
                size: slab_size - req_size,
            });
        }
        guard.slabs.push(SlabBlock {
            slab_id,
            allocation: alloc,
            free_spans,
        });

        Ok(SlabSubAllocation {
            buffer: buf,
            offset: 0,
            size: req_size,
            device_address: bda,
            slab_id: Some(slab_id),
            standalone: None,
        })
    }

    /// Returns a sub-allocated slice back to its parent slab and coalesces adjacent free spans,
    /// or destroys the standalone buffer if oversized.
    ///
    /// # Safety
    /// Must only be called after the GPU has finished all work referencing `sub`.
    pub unsafe fn free_slice(&self, device: &ash::Device, mut sub: SlabSubAllocation) {
        if let Some(mut standalone) = sub.standalone.take() {
            unsafe {
                standalone.destroy(device);
            }
            return;
        }
        let Some(slab_id) = sub.slab_id.take() else {
            return;
        };

        let mut guard = self.inner.lock();
        let Some(slab) = guard.slabs.iter_mut().find(|s| s.slab_id == slab_id) else {
            return;
        };

        let pos = slab.free_spans.partition_point(|s| s.offset < sub.offset);
        slab.free_spans.insert(
            pos,
            FreeSpan {
                offset: sub.offset,
                size: sub.size,
            },
        );

        // Coalesce with next span
        if pos + 1 < slab.free_spans.len() {
            let cur = slab.free_spans[pos];
            let next = slab.free_spans[pos + 1];
            if cur.offset + cur.size == next.offset {
                slab.free_spans[pos].size += next.size;
                slab.free_spans.remove(pos + 1);
            }
        }
        // Coalesce with previous span
        if pos > 0 {
            let prev = slab.free_spans[pos - 1];
            let cur = slab.free_spans[pos];
            if prev.offset + prev.size == cur.offset {
                slab.free_spans[pos - 1].size += cur.size;
                slab.free_spans.remove(pos);
            }
        }

        // Reclaim completely empty 64 MB slabs when more than one slab is resident
        // (keeping at most 1 warm empty slab to avoid allocation thrashing).
        let empty_count = guard
            .slabs
            .iter()
            .filter(|s| {
                s.free_spans.len() == 1
                    && s.free_spans[0].offset == 0
                    && s.free_spans[0].size == s.allocation.size
            })
            .count();
        if empty_count > 1
            && guard.slabs.len() > 1
            && let Some(empty_idx) = guard.slabs.iter().position(|s| {
                s.slab_id == slab_id
                    && s.free_spans.len() == 1
                    && s.free_spans[0].offset == 0
                    && s.free_spans[0].size == s.allocation.size
            })
        {
            let mut reclaimed = guard.slabs.swap_remove(empty_idx);
            drop(guard);
            unsafe {
                reclaimed.allocation.destroy(device);
            }
        }
    }

    /// Destroys all underlying `VkBuffer` and `VkDeviceMemory` slabs on shutdown.
    ///
    /// # Safety
    /// The `VkDevice` must be idle (`vkDeviceWaitIdle`).
    pub unsafe fn destroy_all(&self, device: &ash::Device) {
        let mut guard = self.inner.lock();
        for mut slab in guard.slabs.drain(..) {
            unsafe {
                slab.allocation.destroy(device);
            }
        }
    }
}
