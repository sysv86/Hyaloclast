use crate::gpu::{VulkanContext, buffer::GpuBufferAllocation, retire::RetireQueue};
use ash::vk;
use bytemuck::{Pod, Zeroable};

/// Default hash table capacity (`2^20 = 1,048,576` world-space cells) for SHaRC.
pub const SHARC_DEFAULT_CAPACITY: u32 = 1 << 20;

/// Matches `struct SharcConfig` in `sharc_resolve.comp` field-for-field (`112` bytes, `std430`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SharcConfigUniform {
    pub hash_entries_address: [u32; 2],
    pub lock_address: [u32; 2],
    pub accumulation_address: [u32; 2],
    pub resolved_address: [u32; 2],
    pub camera_position: [f32; 4],
    pub camera_position_prev: [f32; 4],
    pub scene_scale: f32,
    pub radiance_scale: f32,
    pub accumulation_frame_num: u32,
    pub stale_frame_num_max: u32,
    pub capacity: u32,
    pub frame_index: u32,
    pub enable_anti_firefly_filter: u32,
    pub use_lock_buffer: u32,
    pub debug_mode: u32,
    pub update_downsample_factor: u32,
    pub has_crumbling: u32,
    pub _padding: u32,
}

const _: () = assert!(std::mem::size_of::<SharcConfigUniform>() == 112);

impl SharcConfigUniform {
    pub const fn pack_u64(addr: u64) -> [u32; 2] {
        [addr as u32, (addr >> 32) as u32]
    }
}

/// Manages the GPU storage buffers and per-frame parameters for the Spatial Hash Radiance Cache
/// (**SHaRC**) used by Tier 2 (`RayQuery`) and Tier 3 (`Full Path Tracing`) multi-bounce GI.
pub struct SharcRuntime {
    pub capacity: u32,
    pub hash_entries_buf: Option<GpuBufferAllocation>,
    pub lock_buf: Option<GpuBufferAllocation>,
    pub accumulation_buf: Option<GpuBufferAllocation>,
    pub resolved_buf: Option<GpuBufferAllocation>,
    pub config_ubo_slots: [Option<GpuBufferAllocation>; 3],
    pub prev_camera_pos: [f32; 3],
    pub initialized: bool,
    pub needs_clear: bool,
}

/// Alias exposed for Tier 2 / Tier 3 pipeline descriptor binding.
pub type SharcState = SharcRuntime;

impl Default for SharcRuntime {
    fn default() -> Self {
        Self::new(SHARC_DEFAULT_CAPACITY)
    }
}

impl SharcRuntime {
    /// Returns the SHaRC update downsample factor scaled by the active GPU tier.
    pub const fn downsample_factor_for_tier(tier: crate::gpu::GpuTierClass) -> u32 {
        match tier {
            crate::gpu::GpuTierClass::Enthusiast => 2,
            crate::gpu::GpuTierClass::High => 3,
            crate::gpu::GpuTierClass::Low | crate::gpu::GpuTierClass::Mid => 4,
        }
    }

    pub const fn new(capacity: u32) -> Self {
        Self {
            capacity,
            hash_entries_buf: None,
            lock_buf: None,
            accumulation_buf: None,
            resolved_buf: None,
            config_ubo_slots: [None, None, None],
            prev_camera_pos: [0.0; 3],
            initialized: false,
            needs_clear: true,
        }
    }

    pub fn ensure_allocated(&mut self, ctx: &VulkanContext) -> Result<(), vk::Result> {
        if self.initialized {
            return Ok(());
        }
        if !ctx.capabilities.buffer_device_address {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }

        let cap = u64::from(self.capacity);
        let storage_flags = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_DST;

        let mut hash_entries_buf = GpuBufferAllocation::allocate(
            ctx,
            cap * 8, // uint64_t per entry
            storage_flags,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        // lock_buf is unused when use_lock_buffer == 0; avoid wasting 4 MB VRAM.
        let mut accumulation_buf = match GpuBufferAllocation::allocate(
            ctx,
            cap * 16, // SharcAccumulationData (uvec4 = 16B)
            storage_flags,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ) {
            Ok(b) => b,
            Err(err) => {
                unsafe {
                    hash_entries_buf.destroy(&ctx.device);
                }
                return Err(err);
            }
        };
        let resolved_buf = match GpuBufferAllocation::allocate(
            ctx,
            cap * 32, // SharcResolvedVoxel (vec3 + float + 4x uint = 32B)
            storage_flags,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ) {
            Ok(b) => b,
            Err(err) => {
                unsafe {
                    accumulation_buf.destroy(&ctx.device);
                    hash_entries_buf.destroy(&ctx.device);
                }
                return Err(err);
            }
        };
        self.hash_entries_buf = Some(hash_entries_buf);
        self.lock_buf = None;
        self.accumulation_buf = Some(accumulation_buf);
        self.resolved_buf = Some(resolved_buf);
        self.initialized = true;
        self.needs_clear = true;
        Ok(())
    }

    /// Marks the SHaRC hash tables for a full GPU clear (`vkCmdFillBuffer`) on the next frame
    /// (called on world load, dimension change, or large camera teleport).
    pub fn request_clear(&mut self) {
        self.needs_clear = true;
    }

    /// Records `vkCmdFillBuffer(..., 0)` on all 4 SHaRC storage buffers when `needs_clear` is set,
    /// followed by a `TRANSFER_WRITE -> COMPUTE_SHADER | RAY_TRACING_SHADER_KHR` barrier.
    pub fn record_clear_if_needed(&mut self, ctx: &VulkanContext, cmd: vk::CommandBuffer) {
        if !self.initialized || !self.needs_clear {
            return;
        }
        unsafe {
            for buf_opt in [
                &self.hash_entries_buf,
                &self.lock_buf,
                &self.accumulation_buf,
                &self.resolved_buf,
            ] {
                if let Some(alloc) = buf_opt
                    && alloc.buffer != vk::Buffer::null()
                {
                    ctx.device
                        .cmd_fill_buffer(cmd, alloc.buffer, 0, vk::WHOLE_SIZE, 0);
                }
            }
            ctx.cmd_memory_barrier2(
                cmd,
                vk::PipelineStageFlags2::ALL_TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER
                    | if ctx.capabilities.ray_tracing_pipeline {
                        vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
                    } else {
                        vk::PipelineStageFlags2::empty()
                    },
                vk::AccessFlags2::from_raw(
                    vk::AccessFlags2::SHADER_STORAGE_READ.as_raw()
                        | vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw(),
                ),
            );
        }
        self.needs_clear = false;
    }

    /// Updates and uploads `SharcConfigUniform` for the current frame. Automatically requests
    /// a table clear if the camera teleported more than `128.0` blocks since the previous frame.
    pub fn update_frame_config(
        &mut self,
        ctx: &VulkanContext,
        frame_slot: usize,
        camera_pos: [f32; 3],
        frame_index: u32,
        has_crumbling: bool,
    ) -> Result<(), vk::Result> {
        let dx = camera_pos[0] - self.prev_camera_pos[0];
        let dy = camera_pos[1] - self.prev_camera_pos[1];
        let dz = camera_pos[2] - self.prev_camera_pos[2];
        if frame_index == 0 || (dx * dx + dy * dy + dz * dz) > 128.0 * 128.0 {
            self.needs_clear = true;
        }

        let hash_addr = Self::buffer_bda(&self.hash_entries_buf);
        let lock_addr = Self::buffer_bda(&self.lock_buf);
        let accum_addr = Self::buffer_bda(&self.accumulation_buf);
        let resolved_addr = Self::buffer_bda(&self.resolved_buf);

        let cfg = SharcConfigUniform {
            hash_entries_address: SharcConfigUniform::pack_u64(hash_addr),
            lock_address: SharcConfigUniform::pack_u64(lock_addr),
            accumulation_address: SharcConfigUniform::pack_u64(accum_addr),
            resolved_address: SharcConfigUniform::pack_u64(resolved_addr),
            camera_position: [camera_pos[0], camera_pos[1], camera_pos[2], 1.0],
            camera_position_prev: [
                self.prev_camera_pos[0],
                self.prev_camera_pos[1],
                self.prev_camera_pos[2],
                1.0,
            ],
            scene_scale: 1.0,
            radiance_scale: 1024.0,
            accumulation_frame_num: 64,
            stale_frame_num_max: 128,
            capacity: if self.initialized { self.capacity } else { 0 },
            frame_index,
            enable_anti_firefly_filter: 1,
            use_lock_buffer: 0,
            debug_mode: 0,
            update_downsample_factor: Self::downsample_factor_for_tier(
                ctx.capabilities.gpu_tier_class,
            ),
            has_crumbling: u32::from(has_crumbling),
            _padding: 0,
        };

        let slot = &mut self.config_ubo_slots[frame_slot % 3];
        if slot.is_none() {
            *slot = Some(GpuBufferAllocation::allocate_host_visible_bda(
                ctx,
                std::mem::size_of::<SharcConfigUniform>() as vk::DeviceSize,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?);
        }
        slot.as_mut()
            .unwrap()
            .write_bytes(0, bytemuck::bytes_of(&cfg));
        self.prev_camera_pos = camera_pos;
        Ok(())
    }

    fn buffer_and_size(buf: &Option<GpuBufferAllocation>) -> (vk::Buffer, u64) {
        buf.as_ref()
            .map(|b| (b.buffer, b.size))
            .unwrap_or((vk::Buffer::null(), 0))
    }

    fn buffer_bda(buf: &Option<GpuBufferAllocation>) -> vk::DeviceAddress {
        buf.as_ref().map(|b| b.device_address).unwrap_or(0)
    }

    pub fn config_buffer(&self, frame_slot: usize) -> (vk::Buffer, u64) {
        Self::buffer_and_size(&self.config_ubo_slots[frame_slot % 3])
    }

    pub fn accumulation_buffer(&self) -> (vk::Buffer, u64) {
        Self::buffer_and_size(&self.accumulation_buf)
    }

    pub fn accumulation_buffer_bda(&self) -> vk::DeviceAddress {
        Self::buffer_bda(&self.accumulation_buf)
    }

    pub fn resolved_buffer(&self) -> (vk::Buffer, u64) {
        Self::buffer_and_size(&self.resolved_buf)
    }

    pub fn resolved_buffer_bda(&self) -> vk::DeviceAddress {
        Self::buffer_bda(&self.resolved_buf)
    }

    pub fn hash_entries_buffer(&self) -> (vk::Buffer, u64) {
        Self::buffer_and_size(&self.hash_entries_buf)
    }

    pub fn hash_entries_buffer_bda(&self) -> vk::DeviceAddress {
        Self::buffer_bda(&self.hash_entries_buf)
    }

    pub fn lock_buffer(&self) -> (vk::Buffer, u64) {
        Self::buffer_and_size(&self.lock_buf)
    }

    pub fn lock_buffer_bda(&self) -> vk::DeviceAddress {
        Self::buffer_bda(&self.lock_buf)
    }

    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn destroy(&mut self, ctx: &VulkanContext, retire_queue: &RetireQueue) {
        let dev = ctx.device.clone();
        let bufs = [
            self.hash_entries_buf.take(),
            self.lock_buf.take(),
            self.accumulation_buf.take(),
            self.resolved_buf.take(),
        ];
        let config_slots = std::mem::take(&mut self.config_ubo_slots);
        self.initialized = false;
        self.needs_clear = true;
        retire_queue.retire_later(move || unsafe {
            for b in bufs.into_iter().chain(config_slots).flatten() {
                let mut b = b;
                b.destroy(&dev);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sharc_state_unallocated_accessors_return_null_and_zero() {
        let state = SharcState::default();
        assert_eq!(state.capacity(), SHARC_DEFAULT_CAPACITY);
        assert_eq!(state.config_buffer(0), (vk::Buffer::null(), 0));
        assert_eq!(state.accumulation_buffer(), (vk::Buffer::null(), 0));
        assert_eq!(state.resolved_buffer(), (vk::Buffer::null(), 0));
        assert_eq!(state.hash_entries_buffer(), (vk::Buffer::null(), 0));
        assert_eq!(state.lock_buffer(), (vk::Buffer::null(), 0));
        assert_eq!(state.accumulation_buffer_bda(), 0);
        assert_eq!(state.resolved_buffer_bda(), 0);
        assert_eq!(state.hash_entries_buffer_bda(), 0);
        assert_eq!(state.lock_buffer_bda(), 0);
    }
}
