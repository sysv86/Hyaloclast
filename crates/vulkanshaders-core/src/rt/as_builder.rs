use crate::gpu::{
    VulkanContext,
    buffer::{GpuBufferAllocation, SlabArenaAllocator, SlabSubAllocation},
    retire::RetireQueue,
};
use ash::vk;

/// Required byte alignment (`256`) for `VkAccelerationStructureKHR` offsets inside storage buffers.
pub const AS_STORAGE_ALIGNMENT: vk::DeviceSize = 256;

/// Owns a created `VkAccelerationStructureKHR` (`BOTTOM_LEVEL_KHR` or `TOP_LEVEL_KHR`),
/// its 64-bit GPU `VkDeviceAddress` (`vkGetAccelerationStructureDeviceAddressKHR`), and its
/// backing sub-allocated slice (`SlabSubAllocation`) inside a 64 MB `DEVICE_LOCAL` slab.
#[derive(Debug)]
pub struct AccelerationStructureAllocation {
    pub handle: vk::AccelerationStructureKHR,
    pub device_address: vk::DeviceAddress,
    pub size: vk::DeviceSize,
    pub build_scratch_size: vk::DeviceSize,
    pub update_scratch_size: vk::DeviceSize,
    pub storage: SlabSubAllocation,
}

impl AccelerationStructureAllocation {
    /// Queries driver build sizes via `vkGetAccelerationStructureBuildSizesKHR`, sub-allocates
    /// `256`-byte aligned AS storage from `slab_pool` (avoiding `maxMemoryAllocationCount` limits),
    /// creates the `VkAccelerationStructureKHR` at `storage.offset`, and fetches its hardware
    /// `VkDeviceAddress`.
    pub fn create_sized(
        ctx: &VulkanContext,
        slab_pool: &SlabArenaAllocator,
        as_type: vk::AccelerationStructureTypeKHR,
        build_info: &vk::AccelerationStructureBuildGeometryInfoKHR<'_>,
        max_primitive_counts: &[u32],
    ) -> Result<Self, vk::Result> {
        let Some(ref as_ext) = ctx.accel_struct_ext else {
            return Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT);
        };

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            as_ext.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                build_info,
                max_primitive_counts,
                &mut size_info,
            );
        }

        let as_size = size_info.acceleration_structure_size.max(256);
        let storage = slab_pool.allocate_slice(ctx, as_size, AS_STORAGE_ALIGNMENT)?;

        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(storage.buffer)
            .offset(storage.offset)
            .size(as_size)
            .ty(as_type);

        let handle = match unsafe { as_ext.create_acceleration_structure(&create_info, None) } {
            Ok(h) => h,
            Err(err) => {
                unsafe {
                    slab_pool.free_slice(&ctx.device, storage);
                }
                return Err(err);
            }
        };

        let addr_info =
            vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(handle);
        let device_address =
            unsafe { as_ext.get_acceleration_structure_device_address(&addr_info) };

        Ok(Self {
            handle,
            device_address,
            size: as_size,
            build_scratch_size: size_info.build_scratch_size.max(256),
            update_scratch_size: size_info.update_scratch_size.max(256),
            storage,
        })
    }

    /// Creates an acceleration structure of an exact byte size (used for compacted BLAS copies).
    pub fn create_with_exact_size(
        ctx: &VulkanContext,
        slab_pool: &SlabArenaAllocator,
        as_type: vk::AccelerationStructureTypeKHR,
        exact_size: vk::DeviceSize,
    ) -> Result<Self, vk::Result> {
        let Some(ref as_ext) = ctx.accel_struct_ext else {
            return Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT);
        };
        let as_size = exact_size.max(256);
        let storage = slab_pool.allocate_slice(ctx, as_size, AS_STORAGE_ALIGNMENT)?;

        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(storage.buffer)
            .offset(storage.offset)
            .size(as_size)
            .ty(as_type);

        let handle = match unsafe { as_ext.create_acceleration_structure(&create_info, None) } {
            Ok(h) => h,
            Err(err) => {
                unsafe {
                    slab_pool.free_slice(&ctx.device, storage);
                }
                return Err(err);
            }
        };

        let addr_info =
            vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(handle);
        let device_address =
            unsafe { as_ext.get_acceleration_structure_device_address(&addr_info) };

        Ok(Self {
            handle,
            device_address,
            size: as_size,
            build_scratch_size: 256,
            update_scratch_size: 256,
            storage,
        })
    }

    /// Retires both the `VkAccelerationStructureKHR` handle and its sub-allocated slab slice
    /// together through `RetireQueue` so the storage slice is never reused before the AS handle
    /// is destroyed and in-flight GPU work completes.
    pub fn retire(
        mut self,
        ctx: &VulkanContext,
        slab_pool: &SlabArenaAllocator,
        retire_queue: &RetireQueue,
    ) {
        let device = ctx.device.clone();
        let as_ext = ctx.accel_struct_ext.clone();
        let pool = slab_pool.clone();
        let handle = self.handle;
        self.handle = vk::AccelerationStructureKHR::null();

        retire_queue.retire_later(move || unsafe {
            if let Some(ext) = as_ext
                && handle != vk::AccelerationStructureKHR::null()
            {
                ext.destroy_acceleration_structure(handle, None);
            }
            pool.free_slice(&device, self.storage);
        });
    }
}

/// Persistent GPU Top-Level Acceleration Structure (`TLAS`) builder for one frame slot.
///
/// Reuses its `HOST_VISIBLE` instance buffer (`VkAccelerationStructureInstanceKHR[]`),
/// `DEVICE_LOCAL` AS storage, and scratch buffer across frames, growing only when instance
/// capacity increases (`1.5x` geometric growth).
pub struct TlasFrameBuilder {
    pub instance_buffer: Option<GpuBufferAllocation>,
    pub instance_capacity: usize,
    pub tlas_built_capacity: usize,
    pub scratch_buffer: Option<GpuBufferAllocation>,
    pub tlas: Option<AccelerationStructureAllocation>,
}

impl Default for TlasFrameBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TlasFrameBuilder {
    pub const fn new() -> Self {
        Self {
            instance_buffer: None,
            instance_capacity: 0,
            tlas_built_capacity: 0,
            scratch_buffer: None,
            tlas: None,
        }
    }

    /// Uploads `instances` to mapped BDA memory and records `vkCmdBuildAccelerationStructuresKHR`
    /// onto `cmd`, followed by an explicit `VK_KHR_synchronization2` barrier making the TLAS ready
    /// for `VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT`.
    pub fn record_build(
        &mut self,
        ctx: &VulkanContext,
        slab_pool: &SlabArenaAllocator,
        retire_queue: &RetireQueue,
        cmd: vk::CommandBuffer,
        instances: &[vk::AccelerationStructureInstanceKHR],
    ) -> Result<vk::DeviceAddress, vk::Result> {
        let Some(ref as_ext) = ctx.accel_struct_ext else {
            return Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT);
        };

        let count = instances.len();
        let instance_stride = std::mem::size_of::<vk::AccelerationStructureInstanceKHR>();
        let needed_bytes = count * instance_stride;

        let capacity_grew = self.instance_buffer.is_none() || self.instance_capacity < count;
        if capacity_grew {
            let new_cap = count + count / 2 + 64;
            let new_buf = GpuBufferAllocation::allocate_host_visible_bda(
                ctx,
                (new_cap * instance_stride) as vk::DeviceSize,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            )?;
            if let Some(mut old) = self.instance_buffer.take() {
                let dev = ctx.device.clone();
                retire_queue.retire_later(move || unsafe {
                    old.destroy(&dev);
                });
            }
            self.instance_buffer = Some(new_buf);
            self.instance_capacity = new_cap;
        }

        let inst_buf = self.instance_buffer.as_mut().unwrap();
        let raw_slice =
            unsafe { std::slice::from_raw_parts(instances.as_ptr() as *const u8, needed_bytes) };
        inst_buf.write_bytes(0, raw_slice);

        // Prior BLAS build -> TLAS build input barrier (host writes before vkQueueSubmit are
        // implicitly visible to submitted command buffers per Vulkan synchronization rules).
        unsafe {
            ctx.cmd_memory_barrier2(
                cmd,
                vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
                vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
            );
        }

        let instances_data = vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: inst_buf.device_address,
            });
        let geometry = [vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: instances_data,
            })];

        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometry);

        let cap_u32 = [self.instance_capacity as u32];
        let need_new_tlas = match self.tlas {
            None => true,
            Some(ref existing)
                if capacity_grew || self.tlas_built_capacity < self.instance_capacity =>
            {
                let mut size_check = vk::AccelerationStructureBuildSizesInfoKHR::default();
                unsafe {
                    as_ext.get_acceleration_structure_build_sizes(
                        vk::AccelerationStructureBuildTypeKHR::DEVICE,
                        &build_info,
                        &cap_u32,
                        &mut size_check,
                    );
                }
                existing.size < size_check.acceleration_structure_size
            }
            Some(_) => false,
        };

        if need_new_tlas {
            let new_tlas = AccelerationStructureAllocation::create_sized(
                ctx,
                slab_pool,
                vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                &build_info,
                &cap_u32,
            )?;
            if let Some(old_tlas) = self.tlas.take() {
                old_tlas.retire(ctx, slab_pool, retire_queue);
            }
            self.tlas = Some(new_tlas);
            self.tlas_built_capacity = self.instance_capacity;
        } else if self.tlas_built_capacity < self.instance_capacity {
            self.tlas_built_capacity = self.instance_capacity;
        }

        let tlas_ref = self.tlas.as_ref().unwrap();
        let needed_scratch = tlas_ref.build_scratch_size;
        if self
            .scratch_buffer
            .as_ref()
            .is_none_or(|s| s.size < needed_scratch)
        {
            let new_scratch = GpuBufferAllocation::allocate_as_scratch(ctx, needed_scratch)?;
            if let Some(mut old_s) = self.scratch_buffer.take() {
                let dev = ctx.device.clone();
                retire_queue.retire_later(move || unsafe {
                    old_s.destroy(&dev);
                });
            }
            self.scratch_buffer = Some(new_scratch);
        }

        let scratch_addr = self.scratch_buffer.as_ref().unwrap().device_address;
        build_info.dst_acceleration_structure = tlas_ref.handle;
        build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: scratch_addr,
        };

        let range = [vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(count as u32)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0)];
        let range_slices: [&[vk::AccelerationStructureBuildRangeInfoKHR]; 1] = [&range];

        unsafe {
            as_ext.cmd_build_acceleration_structures(cmd, &[build_info], &range_slices);
            let mut dst_stages =
                vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER;
            if ctx.capabilities.ray_tracing_pipeline {
                dst_stages |= vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR;
            }
            ctx.cmd_memory_barrier2(
                cmd,
                vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
                dst_stages,
                vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
            );
        }

        Ok(tlas_ref.device_address)
    }

    pub fn handle(&self) -> vk::AccelerationStructureKHR {
        self.tlas
            .as_ref()
            .map(|t| t.handle)
            .unwrap_or(vk::AccelerationStructureKHR::null())
    }

    pub fn destroy(
        &mut self,
        ctx: &VulkanContext,
        slab_pool: &SlabArenaAllocator,
        retire_queue: &RetireQueue,
    ) {
        if let Some(tlas) = self.tlas.take() {
            tlas.retire(ctx, slab_pool, retire_queue);
        }
        self.tlas_built_capacity = 0;
        if let Some(mut inst) = self.instance_buffer.take() {
            let dev = ctx.device.clone();
            retire_queue.retire_later(move || unsafe {
                inst.destroy(&dev);
            });
        }
        if let Some(mut scratch) = self.scratch_buffer.take() {
            let dev = ctx.device.clone();
            retire_queue.retire_later(move || unsafe {
                scratch.destroy(&dev);
            });
        }
    }
}
