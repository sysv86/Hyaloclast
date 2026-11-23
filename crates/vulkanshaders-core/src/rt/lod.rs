use crate::{
    gpu::{
        VulkanContext,
        buffer::{GpuBufferAllocation, SlabArenaAllocator},
        retire::RetireQueue,
    },
    rt::{InstanceMetaGpu, as_builder::AccelerationStructureAllocation, sbt::ShaderBindingTable},
};
use ash::vk::{self, Handle};
use std::collections::{HashMap, HashSet, VecDeque};

/// Distant Horizons (LOD) section geometry descriptor registered via `vs_core_register_lod_slice`.
#[derive(Debug)]
pub struct LodSliceDescriptor {
    pub lod_key_id: u64,
    pub lod_level: u32,
    pub origin_blocks: [i32; 3],
    pub vk_buffer: u64,
    pub byte_offset: u64,
    pub byte_size: u64,
    pub vertex_stride: u32,
    pub vertex_count: u32,
    /// Distant Horizons LOD level scale in blocks (geometry is positioned at `origin + local * scale`).
    pub scale_blocks: f32,
    local_horizontal_bounds: Option<[[f32; 2]; 2]>,
    pub index_buffer: u64,
    pub index_count: u32,
    pub albedo_color_abgr: u32,
    pub material_flags: u32,
    pub cached_vertex_bda: u64,
    pub cached_index_bda: u64,
    pub owned_vertex_buf: Option<GpuBufferAllocation>,
    pub owned_index_buf: Option<GpuBufferAllocation>,
    pub blas: Option<AccelerationStructureAllocation>,
    pub dirty_blas_build: bool,
}

impl LodSliceDescriptor {
    pub fn compute_origin_rel(&self, camera_pos: [f64; 3]) -> [f32; 4] {
        [
            (f64::from(self.origin_blocks[0]) - camera_pos[0]) as f32,
            (f64::from(self.origin_blocks[1]) - camera_pos[1]) as f32,
            (f64::from(self.origin_blocks[2]) - camera_pos[2]) as f32,
            f32::from_bits(self.lod_level),
        ]
    }

    pub fn compute_tlas_transform(&self, camera_pos: [f64; 3]) -> vk::TransformMatrixKHR {
        let rel = self.compute_origin_rel(camera_pos);
        let s = if self.scale_blocks.is_finite() && self.scale_blocks > 0.0 {
            self.scale_blocks
        } else {
            1.0
        };
        vk::TransformMatrixKHR {
            matrix: [
                s, 0.0, 0.0, rel[0], 0.0, s, 0.0, rel[1], 0.0, 0.0, s, rel[2],
            ],
        }
    }
}

fn horizontal_bounds_distance_squared(origin: [f32; 2], scale: f32, bounds: [[f32; 2]; 2]) -> f32 {
    let [min, max] = bounds;
    let x = (origin[0] + min[0] * scale)
        .max(-origin[0] - max[0] * scale)
        .max(0.0);
    let z = (origin[1] + min[1] * scale)
        .max(-origin[1] - max[1] * scale)
        .max(0.0);
    x * x + z * z
}

/// Manages Distant Horizons (LOD) Bottom-Level Acceleration Structures (`BLAS`) and injects them
/// into the scene TLAS with instance mask `0x04` and `layer_ordinal = 4` so Tier 2/3 rays trace
/// seamlessly into distant terrain beyond the vanilla chunk render distance.
#[derive(Default)]
pub struct LodBlasManager {
    pub lod_slices: HashMap<u64, LodSliceDescriptor>,
    pub pending_builds: VecDeque<u64>,
    pub queued_builds: HashSet<u64>,
    pub min_lod_distance_blocks: f32,
    /// Hysteresis band (in blocks) applied when transitioning slices across `min_lod_distance_blocks`
    /// to prevent cracks, overlap, or frame-to-frame popping at the vanilla-to-DH boundary.
    pub hysteresis_blocks: f32,
    /// VRAM pressure scale factor (`0.25..=1.0`) applied to `min_lod_distance_blocks`.
    pub vram_cutoff_scale: f32,
    /// Tracks which LOD slices were injected into TLAS on the previous frame for hysteresis.
    pub visible_lod_keys: HashSet<u64>,
    /// Master enable flag; setting to `false` cleanly clears DH slices when DH is disabled mid-session.
    pub enabled: bool,
}

impl LodBlasManager {
    pub const LOD_INSTANCE_MASK: u8 = 0x04;
    pub const LOD_LAYER_ORDINAL: u32 = 4;
    pub const DEFAULT_HYSTERESIS_BLOCKS: f32 = 16.0;

    /// Returns the default per-frame LOD BLAS build budget scaled by the active GPU tier.
    pub const fn default_builds_per_frame_for_tier(tier: crate::gpu::GpuTierClass) -> usize {
        match tier {
            crate::gpu::GpuTierClass::Enthusiast => 48,
            crate::gpu::GpuTierClass::High => 32,
            crate::gpu::GpuTierClass::Low | crate::gpu::GpuTierClass::Mid => 16,
        }
    }

    pub fn new() -> Self {
        Self {
            lod_slices: HashMap::with_capacity(512),
            pending_builds: VecDeque::with_capacity(128),
            queued_builds: HashSet::with_capacity(128),
            min_lod_distance_blocks: 128.0,
            hysteresis_blocks: Self::DEFAULT_HYSTERESIS_BLOCKS,
            vram_cutoff_scale: 1.0,
            visible_lod_keys: HashSet::with_capacity(512),
            enabled: true,
        }
    }

    /// Sets the minimum horizontal distance (in blocks) before Distant Horizons LOD instances
    /// are injected into the TLAS, preventing overlap with vanilla chunk geometry.
    pub fn set_lod_cutoff_distance(&mut self, blocks: f32) {
        if blocks.is_finite() {
            self.min_lod_distance_blocks = blocks.clamp(16.0, 2048.0);
        }
    }

    /// Sets the hysteresis band width (in blocks) used to prevent boundary cracks/flicker.
    pub fn set_hysteresis_blocks(&mut self, blocks: f32) {
        if blocks.is_finite() {
            self.hysteresis_blocks = blocks.clamp(0.0, 64.0);
        }
    }

    /// Sets the VRAM pressure cutoff scale (`0.25..=1.0`) applied to `min_lod_distance_blocks`.
    pub fn set_vram_cutoff_scale(&mut self, scale: f32) {
        if scale.is_finite() {
            self.vram_cutoff_scale = scale.clamp(0.25, 1.0);
        }
    }

    /// Returns the active LOD cutoff distance after applying VRAM pressure scaling.
    pub fn effective_cutoff_distance(&self) -> f32 {
        (self.min_lod_distance_blocks * self.vram_cutoff_scale).clamp(16.0, 2048.0)
    }

    /// Enables or disables Distant Horizons LOD injection. Disabling mid-session immediately
    /// clears and retires all resident LOD BLASes so no stale geometry remains in TLAS.
    pub fn set_enabled(
        &mut self,
        enabled: bool,
        ctx: Option<&VulkanContext>,
        slab_pool: &SlabArenaAllocator,
        retire_queue: Option<&RetireQueue>,
    ) {
        self.enabled = enabled;
        if !enabled {
            self.clear(ctx, slab_pool, retire_queue);
        }
    }

    /// Evaluates whether a slice at horizontal squared distance `horiz_dist_sq` should be visible
    /// in TLAS given its previous-frame visibility `was_visible` and the active hysteresis band.
    pub fn should_include_with_hysteresis(&self, horiz_dist_sq: f32, was_visible: bool) -> bool {
        let cutoff = self.effective_cutoff_distance();
        if cutoff <= 0.0 {
            return true;
        }
        let threshold = if was_visible {
            (cutoff - self.hysteresis_blocks).max(8.0)
        } else {
            cutoff
        };
        horiz_dist_sq >= threshold * threshold
    }

    fn retire_slice_resources(
        old: LodSliceDescriptor,
        ctx: Option<&VulkanContext>,
        slab_pool: &SlabArenaAllocator,
        retire_queue: Option<&RetireQueue>,
    ) {
        if let Some(c) = ctx {
            if let Some(rq) = retire_queue {
                if let Some(blas) = old.blas {
                    blas.retire(c, slab_pool, rq);
                }
                let vbuf = old.owned_vertex_buf;
                let ibuf = old.owned_index_buf;
                if vbuf.is_some() || ibuf.is_some() {
                    let dev = c.device.clone();
                    rq.retire_later(move || unsafe {
                        if let Some(mut v) = vbuf {
                            v.destroy(&dev);
                        }
                        if let Some(mut i) = ibuf {
                            i.destroy(&dev);
                        }
                    });
                }
            } else {
                unsafe {
                    if let Some(blas) = old.blas {
                        if let Some(ref ext) = c.accel_struct_ext
                            && blas.handle != vk::AccelerationStructureKHR::null()
                        {
                            ext.destroy_acceleration_structure(blas.handle, None);
                        }
                        slab_pool.free_slice(&c.device, blas.storage);
                    }
                    if let Some(mut v) = old.owned_vertex_buf {
                        v.destroy(&c.device);
                    }
                    if let Some(mut i) = old.owned_index_buf {
                        i.destroy(&c.device);
                    }
                }
            }
        }
    }

    fn enqueue_build(&mut self, lod_key_id: u64) {
        if self.queued_builds.insert(lod_key_id) {
            self.pending_builds.push_back(lod_key_id);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn register_lod_slice(
        &mut self,
        lod_key_id: u64,
        lod_level: u32,
        origin_blocks: [i32; 3],
        scale_blocks: f32,
        vk_buffer: u64,
        byte_offset: u64,
        byte_size: u64,
        vertex_stride: u32,
        vertex_count: u32,
        index_buffer: u64,
        index_count: u32,
        albedo_color_abgr: u32,
        ctx: Option<&VulkanContext>,
        slab_pool: &SlabArenaAllocator,
        retire_queue: Option<&RetireQueue>,
    ) {
        if !self.enabled {
            return;
        }
        if let Some(old) = self.lod_slices.remove(&lod_key_id) {
            Self::retire_slice_resources(old, ctx, slab_pool, retire_queue);
        }
        let stride = vertex_stride.max(28);
        let v_count = if vertex_count > 0 {
            vertex_count
        } else {
            (byte_size / u64::from(stride)) as u32
        };
        self.lod_slices.insert(
            lod_key_id,
            LodSliceDescriptor {
                lod_key_id,
                lod_level,
                origin_blocks,
                vk_buffer,
                byte_offset,
                byte_size,
                vertex_stride: stride,
                vertex_count: v_count,
                scale_blocks: if scale_blocks.is_finite() && scale_blocks > 0.0 {
                    scale_blocks
                } else {
                    1.0
                },
                local_horizontal_bounds: None,
                index_buffer,
                index_count,
                albedo_color_abgr,
                // Newer FFI supplies an ABGR colour instead of legacy material flags; LOD slices
                // are non-emissive terrain, so no material bits are set here.
                material_flags: 0,
                cached_vertex_bda: ctx
                    .filter(|_| vk_buffer != 0)
                    .map(|c| {
                        let base =
                            unsafe { c.buffer_device_address(vk::Buffer::from_raw(vk_buffer)) };
                        if base != 0 { base + byte_offset } else { 0 }
                    })
                    .unwrap_or(0),
                cached_index_bda: ctx
                    .filter(|_| index_buffer != 0)
                    .map(|c| unsafe { c.buffer_device_address(vk::Buffer::from_raw(index_buffer)) })
                    .unwrap_or(0),
                owned_vertex_buf: None,
                owned_index_buf: None,
                blas: None,
                dirty_blas_build: true,
            },
        );
        self.enqueue_build(lod_key_id);
    }

    /// Uploads a CPU-unpacked 28-byte/vertex LOD mesh (`pos: [f32; 3]`, `color_abgr: u32`,
    /// `uv: [f32; 2]`, `light_normal: u32`) and `u32` index buffer into owned BDA-enabled
    /// `VkBuffer`s for Distant Horizons integration.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_lod_mesh_cpu(
        &mut self,
        lod_key_id: u64,
        lod_level: u32,
        origin_blocks: [i32; 3],
        scale_blocks: f32,
        vertex_bytes: &[u8],
        vertex_count: u32,
        index_bytes: &[u8],
        index_count: u32,
        material_flags: u32,
        ctx: &VulkanContext,
        slab_pool: &SlabArenaAllocator,
        retire_queue: &RetireQueue,
    ) -> bool {
        if !self.enabled {
            return false;
        }
        let has_indices = !index_bytes.is_empty() && index_count > 0;
        let min_vertex_bytes = (vertex_count as usize).saturating_mul(28);
        if has_indices {
            let min_index_bytes = (index_count as usize).saturating_mul(4);
            if vertex_count < 3
                || index_count < 3
                || vertex_bytes.len() < min_vertex_bytes
                || index_bytes.len() < min_index_bytes
            {
                return false;
            }
        } else if vertex_count < 4 || vertex_bytes.len() < min_vertex_bytes {
            return false;
        }
        let mut horizontal_bounds = [[f32::INFINITY; 2], [f32::NEG_INFINITY; 2]];
        for vertex in vertex_bytes[..min_vertex_bytes].as_chunks::<28>().0 {
            let x = f32::from_ne_bytes(vertex[0..4].try_into().unwrap());
            let z = f32::from_ne_bytes(vertex[8..12].try_into().unwrap());
            for (axis, value) in [x, z].into_iter().enumerate() {
                if !value.is_finite() {
                    return false;
                }
                horizontal_bounds[0][axis] = horizontal_bounds[0][axis].min(value);
                horizontal_bounds[1][axis] = horizontal_bounds[1][axis].max(value);
            }
        }
        let Ok(mut vbuf) = GpuBufferAllocation::allocate_host_visible_bda(
            ctx,
            vertex_bytes.len() as vk::DeviceSize,
            vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::STORAGE_BUFFER,
        ) else {
            return false;
        };
        vbuf.write_bytes(0, vertex_bytes);

        let (index_buffer, stored_index_count, cached_index_bda, owned_index_buf) = if has_indices {
            let Ok(mut ibuf) = GpuBufferAllocation::allocate_host_visible_bda(
                ctx,
                index_bytes.len() as vk::DeviceSize,
                vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
            ) else {
                unsafe {
                    vbuf.destroy(&ctx.device);
                }
                return false;
            };
            ibuf.write_bytes(0, index_bytes);
            let ibuf_raw = ibuf.buffer.as_raw();
            let ibuf_bda = ibuf.device_address;
            (ibuf_raw, index_count, ibuf_bda, Some(ibuf))
        } else {
            (0, 0, 0, None)
        };

        if let Some(old) = self.lod_slices.remove(&lod_key_id) {
            Self::retire_slice_resources(old, Some(ctx), slab_pool, Some(retire_queue));
        }

        let vk_buffer = vbuf.buffer.as_raw();
        let cached_vertex_bda = vbuf.device_address;

        self.lod_slices.insert(
            lod_key_id,
            LodSliceDescriptor {
                lod_key_id,
                lod_level,
                origin_blocks,
                vk_buffer,
                byte_offset: 0,
                byte_size: vertex_bytes.len() as u64,
                vertex_stride: 28,
                vertex_count,
                scale_blocks: if scale_blocks.is_finite() && scale_blocks > 0.0 {
                    scale_blocks
                } else {
                    1.0
                },
                local_horizontal_bounds: Some(horizontal_bounds),
                index_buffer,
                index_count: stored_index_count,
                albedo_color_abgr: 0xFFFF_FFFF,
                material_flags,
                cached_vertex_bda,
                cached_index_bda,
                owned_vertex_buf: Some(vbuf),
                owned_index_buf,
                blas: None,
                dirty_blas_build: true,
            },
        );
        self.enqueue_build(lod_key_id);
        true
    }

    pub fn unregister_lod_slice(
        &mut self,
        lod_key_id: u64,
        ctx: Option<&VulkanContext>,
        slab_pool: &SlabArenaAllocator,
        retire_queue: Option<&RetireQueue>,
    ) {
        self.queued_builds.remove(&lod_key_id);
        self.visible_lod_keys.remove(&lod_key_id);
        if let Some(old) = self.lod_slices.remove(&lod_key_id) {
            Self::retire_slice_resources(old, ctx, slab_pool, retire_queue);
        }
    }

    pub fn clear(
        &mut self,
        ctx: Option<&VulkanContext>,
        slab_pool: &SlabArenaAllocator,
        retire_queue: Option<&RetireQueue>,
    ) {
        self.pending_builds.clear();
        self.queued_builds.clear();
        self.visible_lod_keys.clear();
        for (_, old) in self.lod_slices.drain() {
            Self::retire_slice_resources(old, ctx, slab_pool, retire_queue);
        }
    }

    /// Builds up to `max_builds` pending LOD BLASes per frame using `quad_index_bda`, growing
    /// `blas_scratch` on demand if an individual LOD BLAS exceeds the current scratch capacity.
    #[allow(clippy::too_many_arguments)]
    pub fn record_lod_blas_builds(
        &mut self,
        ctx: &VulkanContext,
        slab_pool: &SlabArenaAllocator,
        retire_queue: &RetireQueue,
        cmd: vk::CommandBuffer,
        quad_index_bda: vk::DeviceAddress,
        blas_scratch: &mut Option<GpuBufferAllocation>,
        max_builds: usize,
    ) -> Result<usize, vk::Result> {
        if !self.enabled {
            self.pending_builds.clear();
            self.queued_builds.clear();
            return Ok(0);
        }
        let Some(ref as_ext) = ctx.accel_struct_ext else {
            return Ok(0);
        };
        if self.pending_builds.is_empty() {
            return Ok(0);
        }
        if blas_scratch.is_none() {
            *blas_scratch = Some(GpuBufferAllocation::allocate_as_scratch(
                ctx,
                2 * 1024 * 1024,
            )?);
        }
        let mut scratch_capacity = blas_scratch.as_ref().map(|s| s.size).unwrap_or(0);
        struct PendingLodBuild {
            key: u64,
            geometry: [vk::AccelerationStructureGeometryKHR<'static>; 1],
            range: [vk::AccelerationStructureBuildRangeInfoKHR; 1],
            new_blas: AccelerationStructureAllocation,
            scratch_offset: u64,
        }

        let mut batch: Vec<PendingLodBuild> = Vec::with_capacity(max_builds);
        let mut scratch_cursor: u64 = 0;

        while batch.len() < max_builds {
            let Some(key) = self.pending_builds.pop_front() else {
                break;
            };
            self.queued_builds.remove(&key);
            let Some(slice) = self.lod_slices.get_mut(&key) else {
                continue;
            };
            if !slice.dirty_blas_build || slice.vk_buffer == 0 || slice.vertex_count < 3 {
                continue;
            }
            if slice.cached_vertex_bda == 0 {
                let base_bda =
                    unsafe { ctx.buffer_device_address(vk::Buffer::from_raw(slice.vk_buffer)) };
                if base_bda != 0 {
                    slice.cached_vertex_bda = base_bda + slice.byte_offset;
                }
            }
            if slice.cached_vertex_bda == 0 {
                continue;
            }
            if slice.index_buffer != 0 && slice.cached_index_bda == 0 {
                slice.cached_index_bda =
                    unsafe { ctx.buffer_device_address(vk::Buffer::from_raw(slice.index_buffer)) };
            }
            let (active_index_bda, primitive_count) =
                if slice.cached_index_bda != 0 && slice.index_count >= 3 {
                    (slice.cached_index_bda, slice.index_count / 3)
                } else {
                    if slice.vertex_count < 4 || quad_index_bda == 0 {
                        continue;
                    }
                    (quad_index_bda, (slice.vertex_count / 4) * 2)
                };
            if primitive_count == 0 {
                continue;
            }

            let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .vertex_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: slice.cached_vertex_bda,
                })
                .vertex_stride(u64::from(slice.vertex_stride))
                .max_vertex(slice.vertex_count - 1)
                .index_type(vk::IndexType::UINT32)
                .index_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: active_index_bda,
                });
            let geom_flags = if (slice.material_flags & 2) != 0 {
                vk::GeometryFlagsKHR::empty()
            } else {
                vk::GeometryFlagsKHR::OPAQUE
            };
            let geometry = [vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .flags(geom_flags)
                .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })];
            let probe_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(&geometry);
            let max_prims = [primitive_count];
            let new_blas = match AccelerationStructureAllocation::create_sized(
                ctx,
                slab_pool,
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                &probe_info,
                &max_prims,
            ) {
                Ok(b) => b,
                Err(err) => {
                    for item in batch {
                        item.new_blas.retire(ctx, slab_pool, retire_queue);
                    }
                    return Err(err);
                }
            };

            let align = ctx.min_scratch_alignment.max(256);
            let aligned_scratch =
                GpuBufferAllocation::align_up(new_blas.build_scratch_size.max(align), align);
            if scratch_cursor + aligned_scratch > scratch_capacity {
                if batch.is_empty() {
                    let alloc_size = (aligned_scratch + aligned_scratch / 2).max(2 * 1024 * 1024);
                    match GpuBufferAllocation::allocate_as_scratch(ctx, alloc_size) {
                        Ok(new_scratch) => {
                            if let Some(mut old_s) = blas_scratch.take() {
                                let dev = ctx.device.clone();
                                retire_queue.retire_later(move || unsafe {
                                    old_s.destroy(&dev);
                                });
                            }
                            scratch_capacity = new_scratch.size;
                            *blas_scratch = Some(new_scratch);
                        }
                        Err(_) => {
                            new_blas.retire(ctx, slab_pool, retire_queue);
                            if self.queued_builds.insert(key) {
                                self.pending_builds.push_back(key);
                            }
                            break;
                        }
                    }
                } else {
                    new_blas.retire(ctx, slab_pool, retire_queue);
                    if self.queued_builds.insert(key) {
                        self.pending_builds.push_front(key);
                    }
                    break;
                }
            }

            let range = [vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(primitive_count)
                .primitive_offset(0)
                .first_vertex(0)
                .transform_offset(0)];
            batch.push(PendingLodBuild {
                key,
                geometry,
                range,
                new_blas,
                scratch_offset: scratch_cursor,
            });
            scratch_cursor += aligned_scratch;
        }

        if batch.is_empty() {
            return Ok(0);
        }
        let scratch_bda = blas_scratch.as_ref().map(|s| s.device_address).unwrap_or(0);
        if scratch_bda == 0 {
            for item in batch {
                item.new_blas.retire(ctx, slab_pool, retire_queue);
            }
            return Ok(0);
        }

        let build_infos: Vec<vk::AccelerationStructureBuildGeometryInfoKHR<'_>> = batch
            .iter()
            .map(|item| {
                vk::AccelerationStructureBuildGeometryInfoKHR::default()
                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                    .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                    .dst_acceleration_structure(item.new_blas.handle)
                    .geometries(&item.geometry)
                    .scratch_data(vk::DeviceOrHostAddressKHR {
                        device_address: scratch_bda + item.scratch_offset,
                    })
            })
            .collect();
        let range_slices: Vec<&[vk::AccelerationStructureBuildRangeInfoKHR]> =
            batch.iter().map(|item| item.range.as_slice()).collect();

        unsafe {
            as_ext.cmd_build_acceleration_structures(cmd, &build_infos, &range_slices);
            ctx.cmd_memory_barrier2(
                cmd,
                vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
                vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::AccessFlags2::from_raw(
                    vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR.as_raw()
                        | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR.as_raw(),
                ),
            );
        }

        let built = batch.len();
        for item in batch {
            if let Some(slice) = self.lod_slices.get_mut(&item.key) {
                if let Some(old) = slice.blas.replace(item.new_blas) {
                    old.retire(ctx, slab_pool, retire_queue);
                }
                slice.dirty_blas_build = false;
            } else {
                item.new_blas.retire(ctx, slab_pool, retire_queue);
            }
        }
        Ok(built)
    }

    /// Appends resident Distant Horizons LOD instances and metadata to the frame's TLAS arrays,
    /// applying hysteresis across `effective_cutoff_distance()` and consistent shadow/GI masks.
    pub fn append_tlas_instances(
        &mut self,
        ctx: &VulkanContext,
        camera_pos: [f64; 3],
        quad_index_bda: vk::DeviceAddress,
        instances: &mut Vec<vk::AccelerationStructureInstanceKHR>,
        metas: &mut Vec<InstanceMetaGpu>,
    ) {
        if !self.enabled || self.lod_slices.is_empty() {
            self.visible_lod_keys.clear();
            return;
        }
        let mut keys: Vec<u64> = self.lod_slices.keys().copied().collect();
        keys.sort_unstable();
        let mut next_visible = HashSet::with_capacity(keys.len());

        for key in keys {
            let Some(slice) = self.lod_slices.get(&key) else {
                continue;
            };
            let Some(ref blas) = slice.blas else {
                continue;
            };
            if blas.device_address == 0 {
                continue;
            }
            let rel = slice.compute_origin_rel(camera_pos);
            let scale = if slice.scale_blocks.is_finite() && slice.scale_blocks > 0.0 {
                slice.scale_blocks
            } else {
                1.0
            };
            // Coarse slices can span hundreds of blocks and enclose the camera.
            let bounds = slice.local_horizontal_bounds.unwrap_or([[32.0; 2]; 2]);
            let horiz_dist_sq = horizontal_bounds_distance_squared([rel[0], rel[2]], scale, bounds);
            let was_visible = self.visible_lod_keys.contains(&key);
            if !self.should_include_with_hysteresis(horiz_dist_sq, was_visible) {
                continue;
            }
            next_visible.insert(key);

            let vertex_bda = if slice.cached_vertex_bda != 0 {
                slice.cached_vertex_bda
            } else if slice.vk_buffer != 0 {
                let base =
                    unsafe { ctx.buffer_device_address(vk::Buffer::from_raw(slice.vk_buffer)) };
                if base != 0 {
                    base + slice.byte_offset
                } else {
                    0
                }
            } else {
                0
            };
            let active_index_bda = if slice.cached_index_bda != 0 && slice.index_count >= 3 {
                slice.cached_index_bda
            } else {
                quad_index_bda
            };
            let idx = (instances.len() as u32) & 0xFF_FFFF;
            let is_translucent = (slice.material_flags & 2) != 0;
            // Consistent shadow/GI participation: bit 2 (0x04) is included in WORLD_RAY_MASK (0xF7)
            // for primary, sun shadow, and GI rays; translucent LODs also set bit 1 (0x02).
            let instance_mask = if is_translucent {
                Self::LOD_INSTANCE_MASK | 0x02
            } else {
                Self::LOD_INSTANCE_MASK
            };
            let (hit_group, inst_flags) = if is_translucent {
                (
                    ShaderBindingTable::HIT_GROUP_TRANSLUCENT,
                    vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
                )
            } else {
                (
                    ShaderBindingTable::HIT_GROUP_OPAQUE,
                    (vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE
                        | vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE)
                        .as_raw() as u8,
                )
            };
            instances.push(vk::AccelerationStructureInstanceKHR {
                transform: slice.compute_tlas_transform(camera_pos),
                instance_custom_index_and_mask: vk::Packed24_8::new(idx, instance_mask),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    hit_group, inst_flags,
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: blas.device_address,
                },
            });
            metas.push(InstanceMetaGpu {
                vertex_bda,
                index_bda: active_index_bda,
                vertex_stride: slice.vertex_stride,
                layer_ordinal: Self::LOD_LAYER_ORDINAL,
                bindless_albedo_slot: 0,
                // Shaders decode LOD solid ABGR color from `meta.bindlessNormalSlot`.
                bindless_normal_slot: slice.albedo_color_abgr,
                bindless_specular_slot: 0,
                material_flags: slice.material_flags,
                _pad: [slice.lod_level, 0],
                region_origin_rel: rel,
            });
        }

        self.visible_lod_keys = next_visible;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coarse_lod_overlap_uses_mesh_edges() {
        let bounds = [[0.0, 0.0], [512.0, 512.0]];
        assert_eq!(
            horizontal_bounds_distance_squared([-240.0, -240.0], 1.0, bounds),
            0.0
        );
        assert_eq!(
            horizontal_bounds_distance_squared([-600.0, -240.0], 1.0, bounds),
            88.0 * 88.0
        );
    }

    #[test]
    fn hysteresis_band_prevents_boundary_flicker() {
        let mut mgr = LodBlasManager::new();
        mgr.set_lod_cutoff_distance(128.0);
        mgr.set_hysteresis_blocks(16.0);

        // At 120 blocks: not entered yet -> excluded; already visible -> retained (>= 112)
        assert!(!mgr.should_include_with_hysteresis(120.0 * 120.0, false));
        assert!(mgr.should_include_with_hysteresis(120.0 * 120.0, true));

        // Inside exit threshold (< 112 blocks) -> dropped even if previously visible
        assert!(!mgr.should_include_with_hysteresis(110.0 * 110.0, true));

        // Beyond enter threshold (>= 128 blocks) -> entered
        assert!(mgr.should_include_with_hysteresis(128.0 * 128.0, false));
    }

    #[test]
    fn vram_cutoff_scale_and_disable_reset_behave_cleanly() {
        let mut mgr = LodBlasManager::new();
        let slab = SlabArenaAllocator::new_as_storage_pool();
        mgr.set_lod_cutoff_distance(200.0);
        mgr.set_vram_cutoff_scale(0.5);
        assert_eq!(mgr.effective_cutoff_distance(), 100.0);

        mgr.register_lod_slice(
            42,
            1,
            [256, 0, 256],
            2.0,
            0,
            0,
            28 * 4,
            28,
            4,
            0,
            0,
            0xFFFF_FFFF,
            None,
            &slab,
            None,
        );
        assert_eq!(mgr.lod_slices.len(), 1);

        mgr.set_enabled(false, None, &slab, None);
        assert!(mgr.lod_slices.is_empty());
        assert!(mgr.pending_builds.is_empty());
        assert!(mgr.visible_lod_keys.is_empty());
    }
}
