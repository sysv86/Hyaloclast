use crate::gpu::{VulkanContext, buffer::GpuBufferAllocation, retire::RetireQueue};
use ash::vk;

/// Shader Binding Table (`SBT`) for `VK_KHR_ray_tracing_pipeline` (`Tier 3`).
///
/// Computes aligned `raygenRegion`, `missRegion`, `hitRegion`, and `callableRegion`
/// (`VkStridedDeviceAddressRegionKHR`) from `VkPhysicalDeviceRayTracingPipelinePropertiesKHR`
/// (`shaderGroupHandleSize`, `shaderGroupHandleAlignment`, `shaderGroupBaseAlignment`),
/// fetches group handles via `vkGetRayTracingShaderGroupHandlesKHR`, and dispatches
/// `vkCmdTraceRaysKHR`.
pub struct ShaderBindingTable {
    pub fixed_buffer: GpuBufferAllocation,
    pub hit_buffer: Option<GpuBufferAllocation>,
    pub raygen_region: vk::StridedDeviceAddressRegionKHR,
    pub miss_region: vk::StridedDeviceAddressRegionKHR,
    pub hit_region: vk::StridedDeviceAddressRegionKHR,
    pub callable_region: vk::StridedDeviceAddressRegionKHR,
}

impl ShaderBindingTable {
    /// Hit group 0: opaque geometry (`rchit` only).
    pub const HIT_GROUP_OPAQUE: u32 = 0;
    /// Hit group 1: alpha-cutout / translucent geometry (`rchit + rahit`).
    pub const HIT_GROUP_CUTOUT: u32 = 1;
    /// Hit group 1: translucent / fluid geometry (`rchit + rahit`).
    pub const HIT_GROUP_TRANSLUCENT: u32 = Self::HIT_GROUP_CUTOUT;

    const fn align_up_u32(val: u32, align: u32) -> u32 {
        if align <= 1 {
            val
        } else {
            (val + align - 1) / align * align
        }
    }

    fn pack_hit_handles(
        raw_handles: &[u8],
        handle_size: u32,
        handle_stride: u32,
        miss_count: u32,
        hit_group_count: u32,
        hit_group_indices: &[u32],
    ) -> Vec<u8> {
        let hs = handle_size as usize;
        let stride = handle_stride as usize;
        let mut packed = vec![0u8; hit_group_indices.len() * stride];
        for (slot, &group_idx) in hit_group_indices.iter().enumerate() {
            let clamped = if hit_group_count > 0 {
                group_idx.min(hit_group_count - 1)
            } else {
                0
            };
            let src_off = ((1 + miss_count + clamped) as usize) * hs;
            let dst_off = slot * stride;
            if src_off + hs <= raw_handles.len() && dst_off + hs <= packed.len() {
                packed[dst_off..dst_off + hs].copy_from_slice(&raw_handles[src_off..src_off + hs]);
            }
        }
        packed
    }

    pub fn build_for_pipeline(
        ctx: &VulkanContext,
        rt_pipeline: vk::Pipeline,
        miss_count: u32,
        hit_group_count: u32,
    ) -> Result<Self, vk::Result> {
        let Some(ref rt_ext) = ctx.rt_pipeline_ext else {
            return Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT);
        };

        let props = ctx.rt_properties;
        let handle_size = props.shader_group_handle_size;
        let handle_stride = Self::align_up_u32(handle_size, props.shader_group_handle_alignment);
        let base_alignment = props.shader_group_base_alignment.max(handle_stride);

        let total_groups = 1 + miss_count + hit_group_count;
        let raw_handles = unsafe {
            rt_ext.get_ray_tracing_shader_group_handles(
                rt_pipeline,
                0,
                total_groups,
                (total_groups * handle_size) as usize,
            )?
        };

        // Raygen record at offset 0; miss records start at the next baseAlignment boundary
        let miss_offset = Self::align_up_u32(handle_stride, base_alignment);
        let fixed_bytes = (miss_offset + handle_stride * miss_count.max(1)) as usize;
        let mut fixed_data = vec![0u8; fixed_bytes];

        // Copy raygen handle (group 0)
        fixed_data[..handle_size as usize].copy_from_slice(&raw_handles[..handle_size as usize]);

        // Copy miss handles (groups 1..=miss_count)
        for i in 0..miss_count {
            let src_start = ((1 + i) * handle_size) as usize;
            let dst_start = (miss_offset + i * handle_stride) as usize;
            fixed_data[dst_start..dst_start + handle_size as usize]
                .copy_from_slice(&raw_handles[src_start..src_start + handle_size as usize]);
        }

        let mut fixed_buffer = GpuBufferAllocation::allocate_host_visible_bda(
            ctx,
            fixed_bytes as vk::DeviceSize,
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
        )?;
        fixed_buffer.write_bytes(0, &fixed_data);

        let base_addr = fixed_buffer.device_address;
        let raygen_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(base_addr)
            .stride(u64::from(handle_stride))
            .size(u64::from(handle_stride));
        let miss_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(base_addr + u64::from(miss_offset))
            .stride(u64::from(handle_stride))
            .size(u64::from(handle_stride * miss_count));

        // Seed hit_buffer with identity handles [0..hit_group_count] so hit_region is valid
        // for both opaque (0) and cutout (1) hit groups immediately after pipeline creation.
        let (hit_buffer, hit_region) = if hit_group_count > 0 {
            let default_hits: Vec<u32> = (0..hit_group_count).collect();
            let packed = Self::pack_hit_handles(
                &raw_handles,
                handle_size,
                handle_stride,
                miss_count,
                hit_group_count,
                &default_hits,
            );
            let mut buf = match GpuBufferAllocation::allocate_host_visible_bda(
                ctx,
                packed.len().max(base_alignment as usize) as vk::DeviceSize,
                vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
            ) {
                Ok(b) => b,
                Err(err) => {
                    unsafe {
                        fixed_buffer.destroy(&ctx.device);
                    }
                    return Err(err);
                }
            };
            buf.write_bytes(0, &packed);
            let region = vk::StridedDeviceAddressRegionKHR::default()
                .device_address(buf.device_address)
                .stride(u64::from(handle_stride))
                .size(packed.len() as vk::DeviceSize);
            (Some(buf), region)
        } else {
            (None, vk::StridedDeviceAddressRegionKHR::default())
        };

        Ok(Self {
            fixed_buffer,
            hit_buffer,
            raygen_region,
            miss_region,
            hit_region,
            callable_region: vk::StridedDeviceAddressRegionKHR::default(),
        })
    }

    /// Dispatches `vkCmdTraceRaysKHR` using this SBT's regions.
    ///
    /// # Safety
    /// `cmd` must have the matching ray tracing pipeline and descriptor sets bound.
    pub unsafe fn cmd_trace_rays(
        &self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        width: u32,
        height: u32,
    ) {
        if let Some(ref rt_ext) = ctx.rt_pipeline_ext {
            unsafe {
                rt_ext.cmd_trace_rays(
                    cmd,
                    &self.raygen_region,
                    &self.miss_region,
                    &self.hit_region,
                    &self.callable_region,
                    width,
                    height,
                    1,
                );
            }
        }
    }

    pub fn destroy(&mut self, ctx: &VulkanContext, retire_queue: &RetireQueue) {
        let dev = ctx.device.clone();
        let mut fixed = std::mem::replace(
            &mut self.fixed_buffer,
            GpuBufferAllocation {
                buffer: vk::Buffer::null(),
                memory: vk::DeviceMemory::null(),
                size: 0,
                device_address: 0,
                mapped_ptr: std::ptr::null_mut(),
            },
        );
        let hit = self.hit_buffer.take();
        retire_queue.retire_later(move || unsafe {
            fixed.destroy(&dev);
            if let Some(mut h) = hit {
                h.destroy(&dev);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_hit_handles_selects_opaque_and_cutout_groups() {
        // 1 raygen + 1 miss + 2 hit groups, handle_size = 32, handle_stride = 64
        let handle_size = 32u32;
        let handle_stride = 64u32;
        let miss_count = 1u32;
        let hit_group_count = 2u32;

        let mut raw = vec![0u8; (4 * handle_size) as usize];
        // group 0 (raygen) = 0xAA, group 1 (miss) = 0xBB, hit 0 (opaque) = 0x11, hit 1 (cutout) = 0x22
        raw[0..32].fill(0xAA);
        raw[32..64].fill(0xBB);
        raw[64..96].fill(0x11);
        raw[96..128].fill(0x22);

        let packed = ShaderBindingTable::pack_hit_handles(
            &raw,
            handle_size,
            handle_stride,
            miss_count,
            hit_group_count,
            &[
                ShaderBindingTable::HIT_GROUP_OPAQUE,
                ShaderBindingTable::HIT_GROUP_CUTOUT,
                99, // clamps to hit_group_count - 1 (cutout)
            ],
        );

        assert_eq!(packed.len(), 3 * 64);
        assert!(packed[0..32].iter().all(|&b| b == 0x11));
        assert!(packed[32..64].iter().all(|&b| b == 0x00));
        assert!(packed[64..96].iter().all(|&b| b == 0x22));
        assert!(packed[128..160].iter().all(|&b| b == 0x22));
    }
}
