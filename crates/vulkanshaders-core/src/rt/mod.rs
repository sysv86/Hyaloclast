pub mod as_builder;
pub mod entities;
pub mod lights;
pub mod lod;
pub mod omm;
pub mod sbt;
pub mod sharc;
pub mod wind;

use crate::{
    gpu::{
        VulkanContext,
        buffer::{GpuBufferAllocation, SlabArenaAllocator, SlabSubAllocation},
        retire::RetireQueue,
    },
    modules::GpuPipelineManager,
};
use as_builder::{AccelerationStructureAllocation, TlasFrameBuilder};
use ash::vk::{self, Handle};
use bytemuck::{Pod, Zeroable};
pub use entities::{
    ARMOR_OVERLAY_NORMAL_BIAS, DynamicEntityBlasBuilder, ENTITY_FLAG_ALPHA_TESTED,
    ENTITY_FLAG_CAST_SHADOW, ENTITY_FLAG_FIRST_PERSON, ENTITY_FLAG_NO_SHADOW,
    ENTITY_FLAG_OVERLAY_BIAS, ENTITY_FLAG_RECEIVE_GI, ENTITY_FLAG_TRANSLUCENT, EntityCategory,
    EntityGeometryKind,
};
pub use lights::{
    DimensionId, EmissiveLightClusterManager, GpuEmissiveLight, GpuEmissiveLightHeader,
    GpuRestirReservoir, WorldEmissiveLightDesc,
};
pub use lod::{LodBlasManager, LodSliceDescriptor};
pub use omm::{
    AtlasAlphaMask, CutoutOmmBuildInput, INSTANCE_FLAG_HAS_OMM, OmmFormat, OmmState,
    OmmTriangleEncoding, OpacityMicromapAllocation, OpacityMicromapBuilder, OpacityMicromapDevice,
};
use sbt::ShaderBindingTable;
pub use sharc::{SharcRuntime, SharcState};
use std::collections::{HashMap, VecDeque};
use wind::{WindJobGpu, WindRefitScheduler};

/// Per-TLAS-instance metadata uploaded to Set 1 binding 1 (`InstanceMetaBuffer`, `std430`),
/// indexed 1-to-1 by `gl_InstanceCustomIndexEXT` / `rayQueryGetIntersectionInstanceCustomIndexEXT`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct InstanceMetaGpu {
    pub vertex_bda: u64,
    pub index_bda: u64,
    pub vertex_stride: u32,
    pub layer_ordinal: u32,
    pub bindless_albedo_slot: u32,
    pub bindless_normal_slot: u32,
    pub bindless_specular_slot: u32,
    pub material_flags: u32,
    pub _pad: [u32; 2],
    pub region_origin_rel: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<InstanceMetaGpu>() == 64);
const _: () = assert!(std::mem::offset_of!(InstanceMetaGpu, region_origin_rel) == 48);

// Keep the mining overlay out of world, lighting, and water rays.
const CRUMBLING_MASK: u8 = 0x08;
// The first-person body participates in lighting and reflections, but not primary visibility.
const FIRST_PERSON_AVATAR_MASK: u8 = 0x10;
const WORLD_MASK: u8 = !CRUMBLING_MASK;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SectionLayerKey {
    pub allocation_key_id: u64,
    pub layer_ordinal: u32,
}

struct WindCandidate {
    dist_sq: f64,
    key: SectionLayerKey,
    displaced_bda: u64,
    rest_bda: u64,
    vertex_count: u32,
    is_region_shifted: bool,
    ox: i32,
    oy: i32,
    oz: i32,
}

#[derive(Debug)]
pub struct SectionSliceDescriptor {
    pub key: SectionLayerKey,
    pub section_pos_packed: i64,
    pub vk_buffer: u64,
    pub byte_offset: u64,
    pub byte_size: u64,
    pub vertex_stride: u32,
    pub vertex_count: u32,
    pub is_region_shifted: bool,
    pub has_foliage_wind: bool,
    pub bindless_albedo_slot: u32,
    pub bindless_normal_slot: u32,
    pub bindless_specular_slot: u32,
    pub material_flags: u32,
    /// Separate device-local buffer holding wind-displaced foliage vertices so `wind_displace.comp`
    /// never mutates the static rest-pose vertices stored in `UberGpuBuffer`.
    pub displaced_vertex_buf: Option<SlabSubAllocation>,
    pub cached_vertex_bda: u64,
    pub blas: Option<AccelerationStructureAllocation>,
    /// Optional hardware Opacity Micromap (`VK_EXT_opacity_micromap`) bound to cutout (`layer_ordinal == 1`) BLASes.
    pub omm: Option<OpacityMicromapAllocation>,
    /// Pending CPU-analyzed cutout OMM data awaiting the next BLAS build.
    pub pending_omm_input: Option<CutoutOmmBuildInput>,
    pub dirty_blas_build: bool,
    pub compacted: bool,
}

impl SectionSliceDescriptor {
    /// Unpacks `SectionPos.asLong()` (`x: 22b [42..63], z: 22b [20..41], y: 20b [0..19]`)
    /// into block-space `SectionOrigin = (sec_x * 16, sec_y * 16, sec_z * 16)`.
    pub const fn unpack_section_origin_blocks(section_pos_packed: i64) -> (i32, i32, i32) {
        let sx = (section_pos_packed >> 42) as i32;
        let sy = ((section_pos_packed << 44) >> 44) as i32;
        let sz = ((section_pos_packed << 22) >> 42) as i32;
        (sx * 16, sy * 16, sz * 16)
    }

    /// Computes the world-space anchor of this slice's vertices.
    ///
    /// When `is_region_shifted` is `true`, `VulkanTurbo` has already baked `(sectionPos & 3) * 16`
    /// into the vertex coordinates inside `UberGpuBuffer`, so the TLAS instance transform MUST
    /// anchor at `RegionOrigin = section_origin & !63` to align bit-for-bit with `terrain_region.vsh`.
    pub const fn world_anchor_blocks(&self) -> (i32, i32, i32) {
        let (ox, oy, oz) = Self::unpack_section_origin_blocks(self.section_pos_packed);
        if self.is_region_shifted {
            (ox & !63, oy & !63, oz & !63)
        } else {
            (ox, oy, oz)
        }
    }

    /// Returns the camera-relative AABB `[min_rel, max_rel]` of this section (`16x16x16` blocks)
    /// for per-cascade frustum culling in `dispatch_csm_shadow`.
    pub fn section_aabb_rel(&self, camera_block_pos: [i32; 3]) -> ([f32; 3], [f32; 3]) {
        let (ox, oy, oz) = Self::unpack_section_origin_blocks(self.section_pos_packed);
        let min_x = ox.wrapping_sub(camera_block_pos[0]) as f32;
        let min_y = oy.wrapping_sub(camera_block_pos[1]) as f32;
        let min_z = oz.wrapping_sub(camera_block_pos[2]) as f32;
        (
            [min_x, min_y, min_z],
            [min_x + 16.0, min_y + 16.0, min_z + 16.0],
        )
    }

    /// Computes the camera-relative region/section origin `[ox - camX, oy - camY, oz - camZ, 0.0]`.
    pub fn compute_region_origin_rel(&self, camera_pos: [f64; 3]) -> [f32; 4] {
        let (ax, ay, az) = self.world_anchor_blocks();
        [
            (ax as f64 - camera_pos[0]) as f32,
            (ay as f64 - camera_pos[1]) as f32,
            (az as f64 - camera_pos[2]) as f32,
            0.0,
        ]
    }

    /// Builds the 3x4 row-major `VkTransformMatrixKHR` relative to `camera_pos`.
    pub fn compute_tlas_transform(&self, camera_pos: [f64; 3]) -> vk::TransformMatrixKHR {
        let rel = self.compute_region_origin_rel(camera_pos);
        vk::TransformMatrixKHR {
            matrix: [
                1.0, 0.0, 0.0, rel[0], // row 0
                0.0, 1.0, 0.0, rel[1], // row 1
                0.0, 0.0, 1.0, rel[2], // row 2
            ],
        }
    }
}

/// Manages Zero-Copy Bottom-Level (`BLAS`) and Top-Level (`TLAS`) Acceleration Structures
/// built over `VulkanTurbo`'s `UberGpuBuffer` slices (`IGNIS_TERRAIN_40B` / `28B`), sub-allocated
/// through a 64 MB `SlabArenaAllocator`, plus dynamic entities (`DynamicEntityBlasBuilder`),
/// Distant Horizons LOD (`LodBlasManager`), non-destructive foliage wind refits (`WindRefitScheduler`),
/// Emissive Light Clustering & ReSTIR DI reservoirs (`EmissiveLightClusterManager`), SHaRC spatial
/// hash cache (`SharcRuntime`), per-frame `InstanceMetaGpu` SSBOs, and SBT tables.
pub struct ZeroCopyBlasManager {
    pub slab_pool: SlabArenaAllocator,
    pub slices: HashMap<SectionLayerKey, SectionSliceDescriptor>,
    pub pending_build_queue: VecDeque<SectionLayerKey>,
    pub fallback_quad_ibo: Option<GpuBufferAllocation>,
    pub fallback_quad_capacity: u32,
    pub blas_scratch: Option<GpuBufferAllocation>,
    pub tlas_slots: [TlasFrameBuilder; 3],
    pub instance_meta_slots: [Option<GpuBufferAllocation>; 3],
    pub instance_meta_counts: [usize; 3],
    pub entity_blas: DynamicEntityBlasBuilder,
    pub lod_blas: LodBlasManager,
    pub emissive_lights: EmissiveLightClusterManager,
    pub wind_scheduler: WindRefitScheduler,
    pub omm_builder: OpacityMicromapBuilder,
    pub sharc: SharcRuntime,
    pub max_builds_per_frame: usize,
    /// When `true` under VRAM pressure, skips new BLAS compaction queries and reduces build bursts.
    pub vram_throttle_compaction: bool,
    /// Scale factor (`0.25..=1.0`) applied to Distant Horizons LOD cutoff distance under VRAM pressure.
    pub vram_lod_cutoff_scale: f32,
    pub last_built_tlas_address: vk::DeviceAddress,
    pub last_built_tlas_handle: vk::AccelerationStructureKHR,
    /// Cached sorted list of resident slice keys (invalidated on register/unregister/clear/build).
    cached_sorted_keys: Vec<SectionLayerKey>,
    cached_terrain_instances: Vec<vk::AccelerationStructureInstanceKHR>,
    cached_terrain_metas: Vec<InstanceMetaGpu>,
    sorted_keys_dirty: bool,
    /// Query pool (`ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR`) and pending compaction batch.
    compaction_query_pool: vk::QueryPool,
    compaction_query_capacity: u32,
    pending_compactions: Vec<(SectionLayerKey, vk::AccelerationStructureKHR, u64)>,
    scratch_compact_handles: Vec<vk::AccelerationStructureKHR>,
    scratch_refit_keys: Vec<SectionLayerKey>,
    scratch_near_candidates: Vec<WindCandidate>,
    scratch_reset_candidates: Vec<(SectionLayerKey, u64, u64, u32)>,
    scratch_instances: Vec<vk::AccelerationStructureInstanceKHR>,
    scratch_instance_metas: Vec<InstanceMetaGpu>,
}

impl Default for ZeroCopyBlasManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ZeroCopyBlasManager {
    pub const WIND_REFIT_RADIUS_BLOCKS: f32 = 48.0;

    /// Returns the default per-frame terrain BLAS build budget scaled by the active GPU tier.
    pub const fn default_blas_builds_for_tier(tier: crate::gpu::GpuTierClass) -> usize {
        match tier {
            crate::gpu::GpuTierClass::Enthusiast => 96,
            crate::gpu::GpuTierClass::High => 64,
            crate::gpu::GpuTierClass::Low | crate::gpu::GpuTierClass::Mid => 32,
        }
    }

    pub fn new() -> Self {
        Self {
            slab_pool: SlabArenaAllocator::new_as_storage_pool(),
            slices: HashMap::with_capacity(2048),
            pending_build_queue: VecDeque::with_capacity(512),
            fallback_quad_ibo: None,
            fallback_quad_capacity: 0,
            blas_scratch: None,
            tlas_slots: [
                TlasFrameBuilder::new(),
                TlasFrameBuilder::new(),
                TlasFrameBuilder::new(),
            ],
            instance_meta_slots: [None, None, None],
            instance_meta_counts: [0; 3],
            entity_blas: DynamicEntityBlasBuilder::new(),
            lod_blas: LodBlasManager::new(),
            emissive_lights: EmissiveLightClusterManager::new(),
            wind_scheduler: WindRefitScheduler::new(),
            omm_builder: OpacityMicromapBuilder::new(),
            sharc: SharcRuntime::default(),
            max_builds_per_frame: 32,
            vram_throttle_compaction: false,
            vram_lod_cutoff_scale: 1.0,
            last_built_tlas_address: 0,
            last_built_tlas_handle: vk::AccelerationStructureKHR::null(),
            cached_sorted_keys: Vec::with_capacity(2048),
            cached_terrain_instances: Vec::with_capacity(2048),
            cached_terrain_metas: Vec::with_capacity(2048),
            sorted_keys_dirty: true,
            compaction_query_pool: vk::QueryPool::null(),
            compaction_query_capacity: 0,
            pending_compactions: Vec::new(),
            scratch_compact_handles: Vec::with_capacity(64),
            scratch_refit_keys: Vec::with_capacity(64),
            scratch_near_candidates: Vec::with_capacity(64),
            scratch_reset_candidates: Vec::with_capacity(64),
            scratch_instances: Vec::with_capacity(2048),
            scratch_instance_metas: Vec::with_capacity(2048),
        }
    }

    /// Returns `(buffer, byte_size)` of the `InstanceMetaGpu` SSBO for `frame_slot` (`0..3`)
    /// so `modules/mod.rs` can bind Set 1 binding 1 (`InstanceMetaBuffer`).
    pub fn instance_meta_buffer_for_slot(&self, frame_slot: usize) -> (vk::Buffer, u64) {
        let slot = frame_slot % 3;
        self.instance_meta_slots[slot]
            .as_ref()
            .map(|b| (b.buffer, b.size))
            .unwrap_or((vk::Buffer::null(), 0))
    }

    /// Returns `(buffer, byte_size)` of the `EmissiveLightSSBO` for `frame_slot` (`0..3`)
    /// bound at Set 1 binding 3.
    pub fn emissive_light_buffer_for_slot(&self, frame_slot: usize) -> (vk::Buffer, u64) {
        self.emissive_lights.light_buffer_for_slot(frame_slot)
    }

    /// Returns `(buffer, byte_size)` of the ReSTIR DI reservoir SSBO bound at Set 1 binding 4.
    pub fn restir_reservoir_buffer(&self) -> (vk::Buffer, u64) {
        self.emissive_lights.reservoir_buffer()
    }

    /// Returns the number of uploaded `InstanceMetaGpu` entries for `frame_slot` (`0..3`).
    pub fn instance_meta_count_for_slot(&self, frame_slot: usize) -> usize {
        self.instance_meta_counts[frame_slot % 3]
    }

    /// Returns the uint32 quad indices consumed by both the BLAS and the hit shaders.
    /// Minecraft's shared index buffer can use uint16, so its address alone is insufficient.
    pub fn ensure_quad_index_bda(
        &mut self,
        ctx: &VulkanContext,
        retire_queue: &RetireQueue,
        max_quads: u32,
    ) -> Result<vk::DeviceAddress, vk::Result> {
        let needed_quads = max_quads.max(16_384);
        if self.fallback_quad_ibo.is_none() || self.fallback_quad_capacity < needed_quads {
            let cap_quads = needed_quads.next_power_of_two();
            let mut indices = Vec::with_capacity((cap_quads as usize) * 6);
            for q in 0..cap_quads {
                let b = q * 4;
                indices.extend_from_slice(&[b, b + 1, b + 2, b + 2, b + 3, b]);
            }
            let byte_slice: &[u8] = bytemuck::cast_slice(&indices);
            let mut ibo = GpuBufferAllocation::allocate_host_visible_bda(
                ctx,
                byte_slice.len() as vk::DeviceSize,
                vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
            )?;
            ibo.write_bytes(0, byte_slice);
            if let Some(mut old_ibo) = self.fallback_quad_ibo.take() {
                let dev = ctx.device.clone();
                retire_queue.retire_later(move || unsafe {
                    old_ibo.destroy(&dev);
                });
            }
            self.fallback_quad_ibo = Some(ibo);
            self.fallback_quad_capacity = cap_quads;
        }

        Ok(self.fallback_quad_ibo.as_ref().unwrap().device_address)
    }

    fn retire_slice_resources(
        old: SectionSliceDescriptor,
        ctx: Option<&VulkanContext>,
        slab_pool: &SlabArenaAllocator,
        omm_builder: &OpacityMicromapBuilder,
        retire_queue: Option<&RetireQueue>,
    ) {
        if let (Some(c), Some(rq)) = (ctx, retire_queue) {
            if let Some(blas) = old.blas {
                blas.retire(c, slab_pool, rq);
            }
            if let Some(omm) = old.omm {
                omm.retire(
                    c,
                    &omm_builder.storage_slab,
                    omm_builder.device.as_ref(),
                    rq,
                );
            }
            if let Some(disp_buf) = old.displaced_vertex_buf {
                let dev = c.device.clone();
                let pool = slab_pool.clone();
                rq.retire_later(move || unsafe {
                    pool.free_slice(&dev, disp_buf);
                });
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn register_slice(
        &mut self,
        allocation_key_id: u64,
        layer_ordinal: u32,
        section_pos_packed: i64,
        vk_buffer: u64,
        byte_offset: u64,
        byte_size: u64,
        vertex_stride: u32,
        vertex_count: u32,
        is_region_shifted: bool,
        ctx: Option<&VulkanContext>,
        retire_queue: Option<&RetireQueue>,
    ) {
        self.register_slice_with_textures(
            allocation_key_id,
            layer_ordinal,
            section_pos_packed,
            vk_buffer,
            byte_offset,
            byte_size,
            vertex_stride,
            vertex_count,
            is_region_shifted,
            0,
            0,
            0,
            0,
            ctx,
            retire_queue,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn register_slice_with_textures(
        &mut self,
        allocation_key_id: u64,
        layer_ordinal: u32,
        section_pos_packed: i64,
        vk_buffer: u64,
        byte_offset: u64,
        byte_size: u64,
        vertex_stride: u32,
        vertex_count: u32,
        is_region_shifted: bool,
        bindless_albedo_slot: u32,
        bindless_normal_slot: u32,
        bindless_specular_slot: u32,
        material_flags: u32,
        ctx: Option<&VulkanContext>,
        retire_queue: Option<&RetireQueue>,
    ) {
        let stride = if vertex_stride >= 20 {
            vertex_stride
        } else {
            40
        };
        let v_count = if vertex_count > 0 {
            vertex_count
        } else {
            (byte_size / u64::from(stride)) as u32
        };

        let key = SectionLayerKey {
            allocation_key_id,
            layer_ordinal,
        };
        if let Some(old) = self.slices.remove(&key) {
            Self::retire_slice_resources(
                old,
                ctx,
                &self.slab_pool,
                &self.omm_builder,
                retire_queue,
            );
        }

        // Bit 8 of material_flags indicates the caller explicitly tracked whether this CUTOUT
        // section contains animated foliage (bit 0 = has foliage). Legacy callers with bit 8 == 0
        // fall back to `layer_ordinal == 1`.
        let has_foliage_wind = if (material_flags & (1 << 8)) != 0 {
            layer_ordinal == 1 && (material_flags & 1) != 0
        } else {
            layer_ordinal == 1
        };
        let cached_vertex_bda = if vk_buffer != 0 {
            ctx.map(|c| {
                let base = unsafe { c.buffer_device_address(vk::Buffer::from_raw(vk_buffer)) };
                if base != 0 { base + byte_offset } else { 0 }
            })
            .unwrap_or(0)
        } else {
            0
        };
        let desc = SectionSliceDescriptor {
            key,
            section_pos_packed,
            vk_buffer,
            byte_offset,
            byte_size,
            vertex_stride: stride,
            vertex_count: v_count,
            is_region_shifted,
            has_foliage_wind,
            bindless_albedo_slot,
            bindless_normal_slot,
            bindless_specular_slot,
            material_flags: material_flags & 0xFF,
            displaced_vertex_buf: None,
            cached_vertex_bda,
            blas: None,
            omm: None,
            pending_omm_input: None,
            dirty_blas_build: true,
            compacted: false,
        };
        self.slices.insert(key, desc);
        self.sorted_keys_dirty = true;
        // Bound queue growth when running non-RT tiers that do not drain pending_build_queue
        if self.pending_build_queue.len() > self.slices.len().saturating_mul(2).max(1024) {
            self.pending_build_queue
                .retain(|k| self.slices.contains_key(k));
        }
        self.pending_build_queue.push_back(key);
    }

    pub fn unregister_slice(
        &mut self,
        allocation_key_id: u64,
        layer_ordinal: u32,
        ctx: Option<&VulkanContext>,
        retire_queue: Option<&RetireQueue>,
    ) {
        let key = SectionLayerKey {
            allocation_key_id,
            layer_ordinal,
        };
        self.wind_scheduler
            .prev_active_keys
            .remove(&(allocation_key_id, layer_ordinal));
        if let Some(removed) = self.slices.remove(&key) {
            self.sorted_keys_dirty = true;
            Self::retire_slice_resources(
                removed,
                ctx,
                &self.slab_pool,
                &self.omm_builder,
                retire_queue,
            );
        }
    }

    /// Retires all resident terrain section BLASes and displaced vertex buffers (called on world
    /// unload / server change so no dangling `VkBuffer` handles or VRAM leaks remain).
    pub fn clear_all_slices(
        &mut self,
        ctx: Option<&VulkanContext>,
        retire_queue: Option<&RetireQueue>,
    ) {
        self.pending_build_queue.clear();
        self.pending_compactions.clear();
        self.cached_sorted_keys.clear();
        self.sorted_keys_dirty = true;
        self.wind_scheduler.prev_active_keys.clear();
        self.wind_scheduler.current_active_keys.clear();
        self.wind_scheduler.active_jobs.clear();
        for (_, old) in self.slices.drain() {
            Self::retire_slice_resources(
                old,
                ctx,
                &self.slab_pool,
                &self.omm_builder,
                retire_queue,
            );
        }
        self.lod_blas.clear(ctx, &self.slab_pool, retire_queue);
        self.last_built_tlas_address = 0;
        self.last_built_tlas_handle = vk::AccelerationStructureKHR::null();
    }

    fn prioritize_nearby_builds(&mut self, camera_pos: [f64; 3]) {
        if self.pending_build_queue.len() <= self.max_builds_per_frame {
            return;
        }
        let camera_section = camera_pos.map(|value| value.floor() as i32 >> 4);
        let slices = &self.slices;
        self.pending_build_queue
            .make_contiguous()
            .select_nth_unstable_by_key(self.max_builds_per_frame, |key| {
                let Some(slice) = slices.get(key) else {
                    return u32::MAX;
                };
                let (x, y, z) =
                    SectionSliceDescriptor::unpack_section_origin_blocks(slice.section_pos_packed);
                (x >> 4).abs_diff(camera_section[0])
                    + (y >> 4).abs_diff(camera_section[1])
                    + (z >> 4).abs_diff(camera_section[2])
            });
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
        retire_queue: Option<&RetireQueue>,
    ) {
        self.lod_blas.register_lod_slice(
            lod_key_id,
            lod_level,
            origin_blocks,
            scale_blocks,
            vk_buffer,
            byte_offset,
            byte_size,
            vertex_stride,
            vertex_count,
            index_buffer,
            index_count,
            albedo_color_abgr,
            ctx,
            &self.slab_pool,
            retire_queue,
        );
    }

    pub fn unregister_lod_slice(
        &mut self,
        lod_key_id: u64,
        ctx: Option<&VulkanContext>,
        retire_queue: Option<&RetireQueue>,
    ) {
        self.lod_blas
            .unregister_lod_slice(lod_key_id, ctx, &self.slab_pool, retire_queue);
    }

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
        retire_queue: &RetireQueue,
    ) -> bool {
        self.lod_blas.upload_lod_mesh_cpu(
            lod_key_id,
            lod_level,
            origin_blocks,
            scale_blocks,
            vertex_bytes,
            vertex_count,
            index_bytes,
            index_count,
            material_flags,
            ctx,
            &self.slab_pool,
            retire_queue,
        )
    }

    pub fn clear_lod_slices(
        &mut self,
        ctx: Option<&VulkanContext>,
        retire_queue: Option<&RetireQueue>,
    ) {
        self.lod_blas.clear(ctx, &self.slab_pool, retire_queue);
    }

    pub fn set_lod_cutoff_distance(&mut self, blocks: f32) {
        self.lod_blas.set_lod_cutoff_distance(blocks);
    }

    /// Enables or disables Distant Horizons LOD injection, cleanly retiring all resident LOD
    /// BLASes when disabled mid-session.
    pub fn set_lod_enabled(
        &mut self,
        enabled: bool,
        ctx: Option<&VulkanContext>,
        retire_queue: Option<&RetireQueue>,
    ) {
        self.lod_blas
            .set_enabled(enabled, ctx, &self.slab_pool, retire_queue);
    }

    /// Responds to `VK_EXT_memory_budget` VRAM pressure snapshots by throttling BLAS compaction
    /// and tightening the Distant Horizons LOD cutoff distance.
    pub fn apply_vram_pressure(&mut self, throttle_compaction: bool, lod_cutoff_scale: f32) {
        self.vram_throttle_compaction = throttle_compaction;
        let clamped_scale = if lod_cutoff_scale.is_finite() {
            lod_cutoff_scale.clamp(0.25, 1.0)
        } else {
            1.0
        };
        self.vram_lod_cutoff_scale = clamped_scale;
        self.lod_blas.set_vram_cutoff_scale(clamped_scale);
    }

    /// Configures the maximum distance (in blocks) for foliage wind compute and BLAS refits.
    pub fn set_wind_max_distance(&mut self, blocks: f32) {
        self.wind_scheduler.set_wind_max_distance(blocks);
    }

    /// Sets the active Minecraft dimension (`0` = Overworld, `1` = Nether, `2` = End) for
    /// dimension-aware emissive light scaling.
    pub fn set_dimension(&mut self, dimension_id: i32) {
        self.emissive_lights.set_dimension(dimension_id);
    }

    /// Queues CPU-analyzed Opacity Micromap data for a cutout (`layer_ordinal == 1`) terrain slice
    /// so its next BLAS build creates and attaches a hardware `VkMicromapEXT`.
    pub fn attach_cutout_omm_data(
        &mut self,
        allocation_key_id: u64,
        layer_ordinal: u32,
        input: CutoutOmmBuildInput,
    ) -> bool {
        if layer_ordinal != 1 {
            return false;
        }
        let key = SectionLayerKey {
            allocation_key_id,
            layer_ordinal,
        };
        let Some(slice) = self.slices.get_mut(&key) else {
            return false;
        };
        slice.pending_omm_input = Some(input);
        slice.dirty_blas_build = true;
        if !self.pending_build_queue.contains(&key) {
            self.pending_build_queue.push_back(key);
        }
        true
    }

    pub fn active_slice_count(&self) -> usize {
        self.slices.len()
    }

    /// Records up to `max_builds_per_frame` real hardware BLAS builds (`vkCmdBuildAccelerationStructuresKHR`),
    /// allocates separate displaced vertex buffers for foliage sections (`has_foliage_wind`),
    /// dispatches `wind_displace.comp` (`rest_address` in `UberGpuBuffer` -> `positions_address` in
    /// `displaced_vertex_buf`), runs in-place BLAS `UPDATE` refits, builds the dynamic Entity BLAS
    /// and Distant Horizons LOD BLAS, uploads the Emissive Light + ReSTIR DI Reservoir SSBOs,
    /// builds the scene TLAS, and initializes/updates the SHaRC spatial hash cache.
    #[allow(clippy::too_many_arguments)]
    pub fn record_frame_rtx(
        &mut self,
        ctx: &VulkanContext,
        retire_queue: &RetireQueue,
        pipelines: Option<&GpuPipelineManager>,
        bindless: Option<&crate::gpu::bindless::BindlessTextureTable>,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
        camera_pos: [f64; 3],
        _prev_camera_pos: [f64; 3],
        frame_index: u32,
        max_blas_builds: usize,
        game_time_sec: f32,
        rain_level: f32,
    ) -> Result<vk::DeviceAddress, vk::Result> {
        if !ctx.capabilities.acceleration_structure || !ctx.capabilities.buffer_device_address {
            self.pending_build_queue.clear();
            return Ok(0);
        }
        self.last_built_tlas_address = 0;
        self.last_built_tlas_handle = vk::AccelerationStructureKHR::null();
        let Some(ref as_ext) = ctx.accel_struct_ext else {
            return Ok(0);
        };

        let tier_default_builds =
            Self::default_blas_builds_for_tier(ctx.capabilities.gpu_tier_class);
        let base_builds = max_blas_builds.max(tier_default_builds).clamp(1, 256);
        self.max_builds_per_frame = if self.vram_throttle_compaction
            || (self.pending_build_queue.len() > 24
                && matches!(
                    ctx.capabilities.gpu_tier_class,
                    crate::gpu::GpuTierClass::Low | crate::gpu::GpuTierClass::Mid
                )) {
            (base_builds / 2).clamp(4, 8)
        } else {
            base_builds
        };

        let mut trace_stages =
            vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER;
        if ctx.capabilities.ray_tracing_pipeline {
            trace_stages |= vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR;
        }
        // TLAS/BLAS work must wait for earlier traces that still read these structures.
        unsafe {
            ctx.cmd_memory_barrier2(
                cmd,
                trace_stages | vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR
                    | vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
                vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR
                    | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
            );
        }
        let has_frame_upload_work = !self.pending_build_queue.is_empty()
            || !self.lod_blas.pending_builds.is_empty()
            || !self.entity_blas.staged_batches.is_empty()
            || !self.wind_scheduler.prev_active_keys.is_empty();
        if has_frame_upload_work {
            unsafe {
                ctx.cmd_memory_barrier2(
                    cmd,
                    vk::PipelineStageFlags2::ALL_TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR
                        | vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR
                        | vk::AccessFlags2::SHADER_READ,
                );
            }
        }

        // 1. Ensure shared quad index buffer BDA is ready (sized for largest terrain slice, LOD slice, or entity quad batch)
        let max_slice_quads = self
            .slices
            .values()
            .map(|s| s.vertex_count / 4)
            .max()
            .unwrap_or(0);
        let max_lod_quads = self
            .lod_blas
            .lod_slices
            .values()
            .map(|s| s.vertex_count / 4)
            .max()
            .unwrap_or(0);
        let max_entity_quads = self.entity_blas.max_quad_count();
        let max_quads = max_slice_quads
            .max(max_lod_quads)
            .max(max_entity_quads)
            .max(32_768);
        let quad_index_bda = self.ensure_quad_index_bda(ctx, retire_queue, max_quads)?;

        // 1b. Complete any ready BLAS compaction queries from previous frames (non-blocking)
        if !self.pending_compactions.is_empty()
            && self.compaction_query_pool != vk::QueryPool::null()
        {
            let count = self.pending_compactions.len();
            let mut compacted_sizes = vec![0u64; count];
            let query_res = unsafe {
                ctx.device.get_query_pool_results(
                    self.compaction_query_pool,
                    0,
                    &mut compacted_sizes,
                    vk::QueryResultFlags::TYPE_64,
                )
            };
            if query_res.is_ok() {
                let pending = std::mem::take(&mut self.pending_compactions);
                let mut compacted_any = false;
                for ((key, src_handle, orig_size), comp_size) in
                    pending.into_iter().zip(compacted_sizes)
                {
                    if comp_size == 0 || comp_size >= orig_size {
                        continue;
                    }
                    let Some(slice) = self.slices.get_mut(&key) else {
                        continue;
                    };
                    if slice.blas.as_ref().map(|b| b.handle) != Some(src_handle) {
                        continue;
                    }
                    if let Ok(compact_blas) =
                        AccelerationStructureAllocation::create_with_exact_size(
                            ctx,
                            &self.slab_pool,
                            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                            comp_size,
                        )
                    {
                        let copy_info = vk::CopyAccelerationStructureInfoKHR::default()
                            .src(src_handle)
                            .dst(compact_blas.handle)
                            .mode(vk::CopyAccelerationStructureModeKHR::COMPACT);
                        unsafe {
                            as_ext.cmd_copy_acceleration_structure(cmd, &copy_info);
                        }
                        if let Some(old_blas) = slice.blas.replace(compact_blas) {
                            old_blas.retire(ctx, &self.slab_pool, retire_queue);
                        }
                        slice.compacted = true;
                        self.sorted_keys_dirty = true;
                        compacted_any = true;
                    }
                }
                if compacted_any {
                    unsafe {
                        ctx.cmd_memory_barrier2(
                            cmd,
                            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                            vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
                            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                            vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
                        );
                    }
                }
            }
        }

        // 2. Drain up to `max_builds_per_frame` pending chunk section BLAS builds in a SINGLE
        //    batched `cmd_build_acceleration_structures` call with bump-allocated scratch slices.
        self.prioritize_nearby_builds(camera_pos);

        struct PreparedChunkBuild {
            key: SectionLayerKey,
            geometry: [vk::AccelerationStructureGeometryKHR<'static>; 1],
            build_flags: vk::BuildAccelerationStructureFlagsKHR,
            range: [vk::AccelerationStructureBuildRangeInfoKHR; 1],
            new_blas: AccelerationStructureAllocation,
            new_omm: Option<OpacityMicromapAllocation>,
            _omm_attachment:
                Option<Box<vk::AccelerationStructureTrianglesOpacityMicromapEXT<'static>>>,
            scratch_offset: u64,
            allow_compaction: bool,
        }

        let mut prepared_builds: Vec<PreparedChunkBuild> =
            Vec::with_capacity(self.max_builds_per_frame);
        let mut total_build_scratch: u64 = 0;
        let mut copied_foliage_buffers = false;

        while prepared_builds.len() < self.max_builds_per_frame {
            let Some(key) = self.pending_build_queue.pop_front() else {
                break;
            };
            let Some(slice) = self.slices.get_mut(&key) else {
                continue;
            };
            if !slice.dirty_blas_build
                || slice.vk_buffer == 0
                || slice.vertex_count < 4
                || slice.vertex_stride < 28
            {
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

            let rest_vertex_bda = slice.cached_vertex_bda;
            let active_vertex_bda = if slice.has_foliage_wind && slice.vertex_stride == 40 {
                let needed_size = u64::from(slice.vertex_count) * u64::from(slice.vertex_stride);
                if slice
                    .displaced_vertex_buf
                    .as_ref()
                    .is_none_or(|b| b.size < needed_size)
                {
                    if let Some(old_disp) = slice.displaced_vertex_buf.take() {
                        let dev = ctx.device.clone();
                        let pool = self.slab_pool.clone();
                        retire_queue.retire_later(move || unsafe {
                            pool.free_slice(&dev, old_disp);
                        });
                    }
                    slice.displaced_vertex_buf =
                        self.slab_pool.allocate_slice(ctx, needed_size, 256).ok();
                }

                if let Some(ref disp_buf) = slice.displaced_vertex_buf {
                    let copy_region = [vk::BufferCopy {
                        src_offset: slice.byte_offset,
                        dst_offset: disp_buf.offset,
                        size: needed_size,
                    }];
                    unsafe {
                        ctx.device.cmd_copy_buffer(
                            cmd,
                            vk::Buffer::from_raw(slice.vk_buffer),
                            disp_buf.buffer,
                            &copy_region,
                        );
                    }
                    copied_foliage_buffers = true;
                    disp_buf.device_address
                } else {
                    rest_vertex_bda
                }
            } else {
                rest_vertex_bda
            };

            let quad_count = slice.vertex_count / 4;
            let primitive_count = quad_count * 2;
            let is_opaque = slice.key.layer_ordinal == 0;

            // Build or reuse hardware Opacity Micromap for cutout (`layer_ordinal == 1`) sections.
            let pending_omm = if slice.key.layer_ordinal == 1 {
                slice.pending_omm_input.take()
            } else {
                None
            };
            let existing_omm = if slice.key.layer_ordinal == 1 && pending_omm.is_none() {
                slice.omm.take()
            } else {
                None
            };
            let mut active_omm: Option<OpacityMicromapAllocation> = existing_omm;
            if let Some(omm_input) = pending_omm
                && ctx.capabilities.supports_opacity_micromap
                && omm_input.triangles.len() == primitive_count as usize
                && let Ok(built_omm) =
                    self.omm_builder
                        .build_cutout_micromap(ctx, retire_queue, cmd, &omm_input)
            {
                active_omm = Some(built_omm);
            }

            let mut triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .vertex_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: active_vertex_bda,
                })
                .vertex_stride(u64::from(slice.vertex_stride))
                .max_vertex(slice.vertex_count - 1)
                .index_type(vk::IndexType::UINT32)
                .index_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: quad_index_bda,
                });

            let omm_attachment = active_omm.as_ref().map(|omm_alloc| {
                let mut attach = vk::AccelerationStructureTrianglesOpacityMicromapEXT::default()
                    .index_type(omm_alloc.index_type)
                    .index_buffer(vk::DeviceOrHostAddressConstKHR {
                        device_address: omm_alloc.index_buffer.device_address,
                    })
                    .index_stride(omm_alloc.index_stride)
                    .base_triangle(0)
                    .micromap(omm_alloc.handle);
                // `omm_alloc.usage_counts` heap buffer stays pinned while `active_omm` lives in `PreparedChunkBuild`.
                attach.usage_counts_count = omm_alloc.usage_counts.len() as u32;
                attach.p_usage_counts = if omm_alloc.usage_counts.is_empty() {
                    std::ptr::null()
                } else {
                    omm_alloc.usage_counts.as_ptr()
                };
                Box::new(attach)
            });
            if let Some(ref attach_box) = omm_attachment {
                triangles.p_next = (attach_box.as_ref()
                    as *const vk::AccelerationStructureTrianglesOpacityMicromapEXT<'_>)
                    .cast::<std::ffi::c_void>();
            }

            let geom_flags = if is_opaque {
                vk::GeometryFlagsKHR::OPAQUE
            } else {
                vk::GeometryFlagsKHR::empty()
            };
            let geometry = [vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .flags(geom_flags)
                .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })];

            let allow_compaction = !slice.has_foliage_wind && !self.vram_throttle_compaction;
            let build_flags = if slice.has_foliage_wind {
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                    | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE
            } else if allow_compaction {
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                    | vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION
            } else {
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
            };

            let probe_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(build_flags)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(&geometry);

            let max_prims = [primitive_count];
            let new_blas = match AccelerationStructureAllocation::create_sized(
                ctx,
                &self.slab_pool,
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                &probe_info,
                &max_prims,
            ) {
                Ok(b) => b,
                Err(err) => {
                    if let Some(omm) = active_omm {
                        omm.retire(
                            ctx,
                            &self.omm_builder.storage_slab,
                            self.omm_builder.device.as_ref(),
                            retire_queue,
                        );
                    }
                    for p in prepared_builds {
                        p.new_blas.retire(ctx, &self.slab_pool, retire_queue);
                        if let Some(omm) = p.new_omm {
                            omm.retire(
                                ctx,
                                &self.omm_builder.storage_slab,
                                self.omm_builder.device.as_ref(),
                                retire_queue,
                            );
                        }
                    }
                    return Err(err);
                }
            };

            let align = ctx.min_scratch_alignment.max(256);
            let aligned_scratch =
                GpuBufferAllocation::align_up(new_blas.build_scratch_size.max(align), align);
            let range = [vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(primitive_count)
                .primitive_offset(0)
                .first_vertex(0)
                .transform_offset(0)];

            prepared_builds.push(PreparedChunkBuild {
                key,
                geometry,
                build_flags,
                range,
                new_blas,
                new_omm: active_omm,
                _omm_attachment: omm_attachment,
                scratch_offset: total_build_scratch,
                allow_compaction,
            });
            total_build_scratch += aligned_scratch;
        }

        if copied_foliage_buffers {
            unsafe {
                ctx.cmd_memory_barrier2(
                    cmd,
                    vk::PipelineStageFlags2::ALL_TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::PipelineStageFlags2::from_raw(
                        vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR.as_raw()
                            | vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw(),
                    ),
                    vk::AccessFlags2::from_raw(
                        vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR.as_raw()
                            | vk::AccessFlags2::SHADER_READ.as_raw(),
                    ),
                );
            }
        }

        if !prepared_builds.is_empty() {
            if self
                .blas_scratch
                .as_ref()
                .is_none_or(|s| s.size < total_build_scratch)
            {
                let alloc_size =
                    (total_build_scratch + total_build_scratch / 2).max(2 * 1024 * 1024);
                let new_scratch = match GpuBufferAllocation::allocate_as_scratch(ctx, alloc_size) {
                    Ok(s) => s,
                    Err(err) => {
                        for p in prepared_builds {
                            p.new_blas.retire(ctx, &self.slab_pool, retire_queue);
                            if let Some(omm) = p.new_omm {
                                omm.retire(
                                    ctx,
                                    &self.omm_builder.storage_slab,
                                    self.omm_builder.device.as_ref(),
                                    retire_queue,
                                );
                            }
                        }
                        return Err(err);
                    }
                };
                if let Some(mut old_s) = self.blas_scratch.take() {
                    let dev = ctx.device.clone();
                    retire_queue.retire_later(move || unsafe {
                        old_s.destroy(&dev);
                    });
                }
                self.blas_scratch = Some(new_scratch);
            }

            let scratch_base_bda = self.blas_scratch.as_ref().unwrap().device_address;
            let build_infos: Vec<vk::AccelerationStructureBuildGeometryInfoKHR<'_>> =
                prepared_builds
                    .iter()
                    .map(|p| {
                        vk::AccelerationStructureBuildGeometryInfoKHR::default()
                            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                            .flags(p.build_flags)
                            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                            .dst_acceleration_structure(p.new_blas.handle)
                            .geometries(&p.geometry)
                            .scratch_data(vk::DeviceOrHostAddressKHR {
                                device_address: scratch_base_bda + p.scratch_offset,
                            })
                    })
                    .collect();
            let range_slices: Vec<&[vk::AccelerationStructureBuildRangeInfoKHR]> =
                prepared_builds.iter().map(|p| p.range.as_slice()).collect();

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

            let mut compact_handles = std::mem::take(&mut self.scratch_compact_handles);
            compact_handles.clear();
            let can_queue_compaction =
                self.pending_compactions.is_empty() && !self.vram_throttle_compaction;
            for p in prepared_builds {
                let handle = p.new_blas.handle;
                let orig_size = p.new_blas.size;
                if let Some(slice) = self.slices.get_mut(&p.key) {
                    if let Some(old_blas) = slice.blas.replace(p.new_blas) {
                        old_blas.retire(ctx, &self.slab_pool, retire_queue);
                    }
                    if let Some(new_omm) = p.new_omm
                        && let Some(old_omm) = slice.omm.replace(new_omm)
                    {
                        old_omm.retire(
                            ctx,
                            &self.omm_builder.storage_slab,
                            self.omm_builder.device.as_ref(),
                            retire_queue,
                        );
                    }
                    slice.dirty_blas_build = false;
                    slice.compacted = false;
                    self.sorted_keys_dirty = true;
                    if can_queue_compaction && p.allow_compaction && compact_handles.len() < 64 {
                        compact_handles.push(handle);
                        self.pending_compactions.push((p.key, handle, orig_size));
                    }
                } else {
                    p.new_blas.retire(ctx, &self.slab_pool, retire_queue);
                    if let Some(omm) = p.new_omm {
                        omm.retire(
                            ctx,
                            &self.omm_builder.storage_slab,
                            self.omm_builder.device.as_ref(),
                            retire_queue,
                        );
                    }
                }
            }

            if !compact_handles.is_empty() {
                let needed_q = compact_handles.len() as u32;
                if self.compaction_query_pool == vk::QueryPool::null()
                    || self.compaction_query_capacity < needed_q
                {
                    if self.compaction_query_pool != vk::QueryPool::null() {
                        let old_pool = self.compaction_query_pool;
                        let dev = ctx.device.clone();
                        retire_queue.retire_later(move || unsafe {
                            dev.destroy_query_pool(old_pool, None);
                        });
                    }
                    let cap = needed_q.max(64);
                    let pool_info = vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR)
                        .query_count(cap);
                    if let Ok(pool) = unsafe { ctx.device.create_query_pool(&pool_info, None) } {
                        self.compaction_query_pool = pool;
                        self.compaction_query_capacity = cap;
                    } else {
                        self.pending_compactions.clear();
                    }
                }
                if self.compaction_query_pool != vk::QueryPool::null()
                    && !self.pending_compactions.is_empty()
                {
                    unsafe {
                        ctx.device.cmd_reset_query_pool(
                            cmd,
                            self.compaction_query_pool,
                            0,
                            needed_q,
                        );
                        as_ext.cmd_write_acceleration_structures_properties(
                            cmd,
                            &compact_handles,
                            vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                            self.compaction_query_pool,
                            0,
                        );
                    }
                }
                self.scratch_compact_handles = compact_handles;
            }
        }

        // 2b. Build pending Distant Horizons LOD BLASes (allocates/grows blas_scratch on demand)
        if !self.lod_blas.pending_builds.is_empty() {
            let base_lod_budget =
                LodBlasManager::default_builds_per_frame_for_tier(ctx.capabilities.gpu_tier_class);
            let lod_budget = if self.vram_throttle_compaction {
                (base_lod_budget / 2).max(4)
            } else {
                base_lod_budget
            };
            let _ = self.lod_blas.record_lod_blas_builds(
                ctx,
                &self.slab_pool,
                retire_queue,
                cmd,
                quad_index_bda,
                &mut self.blas_scratch,
                lod_budget,
            );
        }

        // 3. Collect near-camera foliage wind jobs scaled by GPU tier
        self.wind_scheduler.active_jobs.clear();
        let (max_wind_sections, _) =
            WindRefitScheduler::budget_for_tier(ctx.capabilities.gpu_tier_class);
        let wind_reach_blocks = self
            .wind_scheduler
            .effective_reach_blocks(ctx.capabilities.gpu_tier_class);
        let reach_sq = wind_reach_blocks * wind_reach_blocks;
        let mut current_wind_keys = std::mem::take(&mut self.wind_scheduler.current_active_keys);
        current_wind_keys.clear();
        let mut refit_keys = std::mem::take(&mut self.scratch_refit_keys);
        refit_keys.clear();
        let period = self.wind_scheduler.wind_period_blocks;
        let mut near_candidates = std::mem::take(&mut self.scratch_near_candidates);
        near_candidates.clear();
        let mut reset_candidates = std::mem::take(&mut self.scratch_reset_candidates);
        reset_candidates.clear();

        for (key, slice) in &self.slices {
            if !slice.has_foliage_wind
                || slice.blas.is_none()
                || slice.vk_buffer == 0
                || slice.vertex_stride != 40
            {
                continue;
            }
            let Some(ref disp_buf) = slice.displaced_vertex_buf else {
                continue;
            };
            let rest_bda = if slice.cached_vertex_bda != 0 {
                slice.cached_vertex_bda
            } else {
                let base_bda =
                    unsafe { ctx.buffer_device_address(vk::Buffer::from_raw(slice.vk_buffer)) };
                if base_bda != 0 {
                    base_bda + slice.byte_offset
                } else {
                    0
                }
            };
            if rest_bda == 0 || disp_buf.device_address == 0 {
                continue;
            }
            let displaced_bda = disp_buf.device_address;

            let (ox, oy, oz) =
                SectionSliceDescriptor::unpack_section_origin_blocks(slice.section_pos_packed);
            let cx = (ox as f64 + 8.0) - camera_pos[0];
            let cy = (oy as f64 + 8.0) - camera_pos[1];
            let cz = (oz as f64 + 8.0) - camera_pos[2];
            let dist_sq = cx * cx + cy * cy + cz * cz;
            let raw_key = (key.allocation_key_id, key.layer_ordinal);

            if wind_reach_blocks > 0.0 && dist_sq <= reach_sq {
                near_candidates.push(WindCandidate {
                    dist_sq,
                    key: *key,
                    displaced_bda,
                    rest_bda,
                    vertex_count: slice.vertex_count,
                    is_region_shifted: slice.is_region_shifted,
                    ox,
                    oy,
                    oz,
                });
            } else if self.wind_scheduler.prev_active_keys.contains(&raw_key) {
                reset_candidates.push((*key, displaced_bda, rest_bda, slice.vertex_count));
            }
        }

        // Sort near-camera foliage sections by distance and animate up to max_wind_sections closest.
        near_candidates.sort_by(|a, b| a.dist_sq.total_cmp(&b.dist_sq));
        for (idx, c) in near_candidates.drain(..).enumerate() {
            let raw_key = (c.key.allocation_key_id, c.key.layer_ordinal);
            if idx < max_wind_sections {
                current_wind_keys.insert(raw_key);
                let anchor_x = if c.is_region_shifted {
                    c.ox & !63
                } else {
                    c.ox
                };
                let anchor_y = if c.is_region_shifted {
                    c.oy & !63
                } else {
                    c.oy
                };
                let anchor_z = if c.is_region_shifted {
                    c.oz & !63
                } else {
                    c.oz
                };
                self.wind_scheduler.active_jobs.push(WindJobGpu {
                    positions_address: c.displaced_bda,
                    rest_address: c.rest_bda,
                    vertex_count: c.vertex_count,
                    scale: WindRefitScheduler::distance_attenuation(
                        c.dist_sq.sqrt(),
                        wind_reach_blocks,
                    ),
                    wrapped_x: WindRefitScheduler::floor_mod(anchor_x, period) as f32,
                    wrapped_y: WindRefitScheduler::floor_mod(anchor_y, period) as f32,
                    wrapped_z: WindRefitScheduler::floor_mod(anchor_z, period) as f32,
                    relative_x: (c.ox as f64 - camera_pos[0]) as f32,
                    relative_y: (c.oy as f64 - camera_pos[1]) as f32,
                    relative_z: (c.oz as f64 - camera_pos[2]) as f32,
                });
                refit_keys.push(c.key);
            } else if self.wind_scheduler.prev_active_keys.contains(&raw_key) {
                reset_candidates.push((c.key, c.displaced_bda, c.rest_bda, c.vertex_count));
            }
        }
        for (key, displaced_bda, rest_bda, vertex_count) in reset_candidates.drain(..) {
            self.wind_scheduler.active_jobs.push(WindJobGpu {
                positions_address: displaced_bda,
                rest_address: rest_bda,
                vertex_count,
                scale: 0.0,
                wrapped_x: 0.0,
                wrapped_y: 0.0,
                wrapped_z: 0.0,
                relative_x: 0.0,
                relative_y: 0.0,
                relative_z: 0.0,
            });
            refit_keys.push(key);
        }
        std::mem::swap(
            &mut self.wind_scheduler.prev_active_keys,
            &mut current_wind_keys,
        );
        current_wind_keys.clear();
        self.wind_scheduler.current_active_keys = current_wind_keys;

        // `game_time_sec` and `rain_level` carry the integrated wind phase and the combined
        // rain/thunder storm supplied by the FFI caller; forward both unchanged.
        let wind_phase = game_time_sec;
        let storm = rain_level;
        if let Ok(Some(wind_pc)) = self.wind_scheduler.upload_frame_jobs(
            ctx,
            retire_queue,
            frame_slot,
            wind_phase,
            1.0,
            storm,
        ) && let Some(pipe_mgr) = pipelines
        {
            pipe_mgr.dispatch_wind_displace(
                ctx,
                cmd,
                &wind_pc,
                self.wind_scheduler.max_job_vertex_count,
            );
        }

        // Batch all in-place BLAS UPDATE refits into a single `cmd_build_acceleration_structures` call.
        if !refit_keys.is_empty() {
            struct PreparedRefit {
                handle: vk::AccelerationStructureKHR,
                geometry: [vk::AccelerationStructureGeometryKHR<'static>; 1],
                range: [vk::AccelerationStructureBuildRangeInfoKHR; 1],
                scratch_offset: u64,
            }
            let mut prepared_refits: Vec<PreparedRefit> = Vec::with_capacity(refit_keys.len());
            let mut total_refit_scratch: u64 = 0;

            for key in refit_keys.drain(..) {
                let Some(slice) = self.slices.get(&key) else {
                    continue;
                };
                let (Some(blas), Some(disp_buf)) = (&slice.blas, &slice.displaced_vertex_buf)
                else {
                    continue;
                };
                if slice.vertex_count < 4 || disp_buf.device_address == 0 {
                    continue;
                }
                let primitive_count = (slice.vertex_count / 4) * 2;
                let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                    .vertex_format(vk::Format::R32G32B32_SFLOAT)
                    .vertex_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: disp_buf.device_address,
                    })
                    .vertex_stride(u64::from(slice.vertex_stride))
                    .max_vertex(slice.vertex_count - 1)
                    .index_type(vk::IndexType::UINT32)
                    .index_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: quad_index_bda,
                    });
                let geometry = [vk::AccelerationStructureGeometryKHR::default()
                    .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                    .flags(vk::GeometryFlagsKHR::empty())
                    .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })];
                let range = [vk::AccelerationStructureBuildRangeInfoKHR::default()
                    .primitive_count(primitive_count)
                    .primitive_offset(0)
                    .first_vertex(0)
                    .transform_offset(0)];
                let align = ctx.min_scratch_alignment.max(256);
                let aligned =
                    GpuBufferAllocation::align_up(blas.update_scratch_size.max(align), align);
                prepared_refits.push(PreparedRefit {
                    handle: blas.handle,
                    geometry,
                    range,
                    scratch_offset: total_refit_scratch,
                });
                total_refit_scratch += aligned;
            }

            if !prepared_refits.is_empty() {
                if self
                    .blas_scratch
                    .as_ref()
                    .is_none_or(|s| s.size < total_refit_scratch)
                {
                    let alloc_size =
                        (total_refit_scratch + total_refit_scratch / 2).max(2 * 1024 * 1024);
                    if let Ok(new_scratch) =
                        GpuBufferAllocation::allocate_as_scratch(ctx, alloc_size)
                    {
                        if let Some(mut old_s) = self.blas_scratch.take() {
                            let dev = ctx.device.clone();
                            retire_queue.retire_later(move || unsafe {
                                old_s.destroy(&dev);
                            });
                        }
                        self.blas_scratch = Some(new_scratch);
                    }
                }
                if let Some(ref scratch) = self.blas_scratch
                    && scratch.size >= total_refit_scratch
                {
                    let scratch_base = scratch.device_address;
                    let build_infos: Vec<vk::AccelerationStructureBuildGeometryInfoKHR<'_>> =
                        prepared_refits
                            .iter()
                            .map(|r| {
                                vk::AccelerationStructureBuildGeometryInfoKHR::default()
                                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                                    .flags(
                                        vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                                            | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE,
                                    )
                                    .mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
                                    .src_acceleration_structure(r.handle)
                                    .dst_acceleration_structure(r.handle)
                                    .geometries(&r.geometry)
                                    .scratch_data(vk::DeviceOrHostAddressKHR {
                                        device_address: scratch_base + r.scratch_offset,
                                    })
                            })
                            .collect();
                    let range_slices: Vec<&[vk::AccelerationStructureBuildRangeInfoKHR]> =
                        prepared_refits.iter().map(|r| r.range.as_slice()).collect();
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
                }
            }
        }
        self.scratch_refit_keys = refit_keys;
        self.scratch_near_candidates = near_candidates;
        self.scratch_reset_candidates = reset_candidates;

        // 4. Collect all resident chunk BLAS instances + per-instance metadata + dynamic Entity BLAS + LOD BLAS
        let mut instances = std::mem::take(&mut self.scratch_instances);
        instances.clear();
        let mut instance_metas = std::mem::take(&mut self.scratch_instance_metas);
        instance_metas.clear();
        self.build_terrain_instances_and_meta(
            ctx,
            bindless,
            camera_pos,
            quad_index_bda,
            &mut instances,
            &mut instance_metas,
        );

        let entity_records = self.entity_blas.record_frame_batches(
            ctx,
            &self.slab_pool,
            retire_queue,
            cmd,
            frame_slot,
            quad_index_bda,
        )?;
        let has_crumbling = entity_records
            .iter()
            .any(|record| record.kind == EntityGeometryKind::Crumbling);
        for rec in entity_records {
            let idx = (instances.len() as u32) & 0xFF_FFFF;
            let mut inst = rec.instance;
            let mask = match rec.kind {
                EntityGeometryKind::World => WORLD_MASK & !0x02,
                EntityGeometryKind::Crumbling => CRUMBLING_MASK,
                EntityGeometryKind::FirstPersonAvatar => FIRST_PERSON_AVATAR_MASK,
            };
            inst.instance_custom_index_and_mask = vk::Packed24_8::new(idx, mask);
            let entity_hit_group = if (rec.entity_flags & ENTITY_FLAG_TRANSLUCENT) != 0 {
                ShaderBindingTable::HIT_GROUP_TRANSLUCENT
            } else {
                ShaderBindingTable::HIT_GROUP_CUTOUT
            };
            inst.instance_shader_binding_table_record_offset_and_flags = vk::Packed24_8::new(
                entity_hit_group,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
            );
            instances.push(inst);
            let (live_normal, live_specular) = bindless
                .map(|b| b.companions_for_albedo(rec.bindless_albedo_slot))
                .unwrap_or((0, 0));
            instance_metas.push(InstanceMetaGpu {
                vertex_bda: rec.vertex_bda,
                index_bda: rec.index_bda,
                vertex_stride: rec.vertex_stride,
                layer_ordinal: if rec.kind == EntityGeometryKind::Crumbling {
                    5
                } else {
                    3
                },
                bindless_albedo_slot: rec.bindless_albedo_slot,
                bindless_normal_slot: if live_normal != 0 {
                    live_normal
                } else {
                    rec.bindless_normal_slot
                },
                bindless_specular_slot: if live_specular != 0 {
                    live_specular
                } else {
                    rec.bindless_specular_slot
                },
                material_flags: rec.entity_flags,
                _pad: [rec.category as u32, 0],
                region_origin_rel: [0.0; 4],
            });
        }

        // Append Distant Horizons LOD instances (instance mask 0x04, layer_ordinal = 4)
        self.lod_blas.append_tlas_instances(
            ctx,
            camera_pos,
            quad_index_bda,
            &mut instances,
            &mut instance_metas,
        );

        // 4b. Upload InstanceMetaGpu[] SSBO for the active frame slot
        let slot_idx = frame_slot % 3;
        self.upload_instance_meta_slot(ctx, retire_queue, cmd, slot_idx, &instance_metas)?;

        // 4c. Upload EmissiveLightSSBO + ensure ReSTIR DI reservoirs using actual held light and viewport
        let (vp_w, vp_h) = self.emissive_lights.viewport_dimensions;
        let uses_reservoirs = pipelines.is_some_and(|p| p.rt_uses_reservoirs);
        let sharc_enabled = pipelines.is_some_and(|p| p.rt_sharc_enabled);
        if sharc_enabled || uses_reservoirs {
            let shader_stages = vk::PipelineStageFlags2::COMPUTE_SHADER
                | if ctx.capabilities.ray_tracing_pipeline {
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
                } else {
                    vk::PipelineStageFlags2::empty()
                };
            // Cache resolve and the next reservoir bank both depend on last frame's writes.
            unsafe {
                ctx.cmd_memory_barrier2(
                    cmd,
                    shader_stages,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    shader_stages | vk::PipelineStageFlags2::ALL_TRANSFER,
                    vk::AccessFlags2::SHADER_STORAGE_READ
                        | vk::AccessFlags2::SHADER_STORAGE_WRITE
                        | vk::AccessFlags2::TRANSFER_WRITE,
                );
            }
        }
        self.emissive_lights
            .upload_frame_lights_and_ensure_reservoirs(
                ctx,
                retire_queue,
                cmd,
                slot_idx,
                frame_index,
                camera_pos,
                vp_w,
                vp_h,
                uses_reservoirs,
            )?;

        // 5. Build the frame's Top-Level Acceleration Structure (TLAS)
        let tlas_addr = self.tlas_slots[slot_idx].record_build(
            ctx,
            &self.slab_pool,
            retire_queue,
            cmd,
            &instances,
        )?;
        // The camera uniform remains available to RT packs, including packs that omit SHaRC.
        if sharc_enabled {
            self.sharc.ensure_allocated(ctx)?;
        }
        self.sharc.update_frame_config(
            ctx,
            slot_idx,
            camera_pos.map(|value| value as f32),
            frame_index,
            has_crumbling,
        )?;
        if let Some(pipe_mgr) = pipelines.filter(|_| sharc_enabled) {
            self.sharc.record_clear_if_needed(ctx, cmd);
            pipe_mgr.dispatch_sharc_resolve(
                ctx,
                cmd,
                self.sharc.config_buffer(slot_idx).0,
                slot_idx,
                self.sharc.capacity,
            );
        }

        self.last_built_tlas_address = tlas_addr;
        self.last_built_tlas_handle = if tlas_addr != 0 {
            self.tlas_slots[slot_idx].handle()
        } else {
            vk::AccelerationStructureKHR::null()
        };

        self.scratch_instances = instances;
        self.scratch_instance_metas = instance_metas;

        Ok(tlas_addr)
    }

    fn ordered_resident_slice_keys(&self) -> Vec<SectionLayerKey> {
        let mut keys: Vec<SectionLayerKey> = self
            .slices
            .iter()
            .filter_map(|(&k, s)| {
                let blas = s.blas.as_ref()?;
                (blas.device_address != 0).then_some(k)
            })
            .collect();
        keys.sort_unstable_by_key(|k| (k.allocation_key_id, k.layer_ordinal));
        keys
    }

    fn make_tlas_instance(
        slice: &SectionSliceDescriptor,
        blas_addr: vk::DeviceAddress,
        instance_idx: u32,
        camera_pos: [f64; 3],
    ) -> vk::AccelerationStructureInstanceKHR {
        let transform = slice.compute_tlas_transform(camera_pos);
        let is_opaque = slice.key.layer_ordinal == 0;
        let flags = if is_opaque {
            vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE
        } else {
            vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE
        };
        let masked_idx = instance_idx & 0xFF_FFFF;
        // Reserve mask bit 1 for translucent terrain so its water intersections can be queried
        // separately from the raster G-buffer, which may contain the depth behind the water.
        let mask = if slice.key.layer_ordinal == 2 {
            WORLD_MASK
        } else {
            WORLD_MASK & !0x02
        };
        vk::AccelerationStructureInstanceKHR {
            transform,
            instance_custom_index_and_mask: vk::Packed24_8::new(masked_idx, mask),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                if is_opaque {
                    ShaderBindingTable::HIT_GROUP_OPAQUE
                } else {
                    ShaderBindingTable::HIT_GROUP_CUTOUT
                },
                flags.as_raw() as u8,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: blas_addr,
            },
        }
    }

    fn refresh_sorted_keys(&mut self) {
        if !self.sorted_keys_dirty {
            return;
        }
        self.cached_sorted_keys.clear();
        for (&k, s) in &self.slices {
            if let Some(ref blas) = s.blas
                && blas.device_address != 0
            {
                self.cached_sorted_keys.push(k);
            }
        }
        self.cached_sorted_keys
            .sort_unstable_by_key(|k| (k.allocation_key_id, k.layer_ordinal));
        self.sorted_keys_dirty = false;
    }

    fn build_terrain_instances_and_meta(
        &mut self,
        ctx: &VulkanContext,
        bindless: Option<&crate::gpu::bindless::BindlessTextureTable>,
        camera_pos: [f64; 3],
        quad_index_bda: vk::DeviceAddress,
        instances: &mut Vec<vk::AccelerationStructureInstanceKHR>,
        metas: &mut Vec<InstanceMetaGpu>,
    ) {
        let rebuild_cache = self.sorted_keys_dirty;
        self.refresh_sorted_keys();
        instances.clear();
        metas.clear();

        if rebuild_cache {
            self.cached_terrain_instances.clear();
            self.cached_terrain_metas.clear();
            for (idx, &key) in self.cached_sorted_keys.iter().enumerate() {
                let Some(slice) = self.slices.get(&key) else {
                    continue;
                };
                let Some(ref blas) = slice.blas else {
                    continue;
                };
                if blas.device_address == 0 {
                    continue;
                }

                let vertex_bda = if slice.has_foliage_wind
                    && let Some(ref disp_buf) = slice.displaced_vertex_buf
                    && disp_buf.device_address != 0
                {
                    disp_buf.device_address
                } else if slice.cached_vertex_bda != 0 {
                    slice.cached_vertex_bda
                } else if slice.vk_buffer != 0 {
                    let base_bda =
                        unsafe { ctx.buffer_device_address(vk::Buffer::from_raw(slice.vk_buffer)) };
                    if base_bda != 0 {
                        base_bda + slice.byte_offset
                    } else {
                        0
                    }
                } else {
                    0
                };
                let (live_normal, live_specular) = bindless
                    .map(|table| table.companions_for_albedo(slice.bindless_albedo_slot))
                    .unwrap_or((0, 0));

                self.cached_terrain_instances.push(Self::make_tlas_instance(
                    slice,
                    blas.device_address,
                    idx as u32,
                    camera_pos,
                ));
                self.cached_terrain_metas.push(InstanceMetaGpu {
                    vertex_bda,
                    index_bda: quad_index_bda,
                    vertex_stride: slice.vertex_stride,
                    layer_ordinal: slice.key.layer_ordinal,
                    bindless_albedo_slot: slice.bindless_albedo_slot,
                    bindless_normal_slot: if live_normal != 0 {
                        live_normal
                    } else {
                        slice.bindless_normal_slot
                    },
                    bindless_specular_slot: if live_specular != 0 {
                        live_specular
                    } else {
                        slice.bindless_specular_slot
                    },
                    material_flags: slice.material_flags
                        | if slice.omm.is_some() {
                            INSTANCE_FLAG_HAS_OMM
                        } else {
                            0
                        },
                    _pad: [0; 2],
                    region_origin_rel: slice.compute_region_origin_rel(camera_pos),
                });
            }
        }

        for idx in 0..self.cached_terrain_instances.len() {
            let key = self.cached_sorted_keys[idx];
            let Some(slice) = self.slices.get(&key) else {
                continue;
            };
            let Some(blas) = &slice.blas else {
                continue;
            };
            let mut instance = self.cached_terrain_instances[idx];
            instance.transform = slice.compute_tlas_transform(camera_pos);
            instance.acceleration_structure_reference.device_handle = blas.device_address;
            self.cached_terrain_instances[idx] = instance;

            let meta = &mut self.cached_terrain_metas[idx];
            meta.region_origin_rel = slice.compute_region_origin_rel(camera_pos);
            let (live_normal, live_specular) = bindless
                .map(|table| table.companions_for_albedo(slice.bindless_albedo_slot))
                .unwrap_or((0, 0));
            meta.bindless_normal_slot = if live_normal != 0 {
                live_normal
            } else {
                slice.bindless_normal_slot
            };
            meta.bindless_specular_slot = if live_specular != 0 {
                live_specular
            } else {
                slice.bindless_specular_slot
            };
            instances.push(instance);
            metas.push(*meta);
        }
    }

    fn upload_instance_meta_slot(
        &mut self,
        ctx: &VulkanContext,
        retire_queue: &RetireQueue,
        cmd: vk::CommandBuffer,
        slot_idx: usize,
        metas: &[InstanceMetaGpu],
    ) -> Result<(), vk::Result> {
        self.instance_meta_counts[slot_idx] = metas.len();
        let needed_bytes = std::mem::size_of_val(metas) as vk::DeviceSize;
        if self.instance_meta_slots[slot_idx]
            .as_ref()
            .is_none_or(|b| b.size < needed_bytes)
        {
            let alloc_bytes = (needed_bytes + needed_bytes / 2 + 4096).max(4096);
            let new_buf = GpuBufferAllocation::allocate_host_visible_bda(
                ctx,
                alloc_bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?;
            if let Some(mut old_buf) = self.instance_meta_slots[slot_idx].take() {
                let dev = ctx.device.clone();
                retire_queue.retire_later(move || unsafe {
                    old_buf.destroy(&dev);
                });
            }
            self.instance_meta_slots[slot_idx] = Some(new_buf);
        }

        if let Some(ref mut buf) = self.instance_meta_slots[slot_idx]
            && !metas.is_empty()
        {
            buf.write_bytes(0, bytemuck::cast_slice(metas));
        }

        let _ = cmd;
        // Host writes before vkQueueSubmit are implicitly visible to submitted command buffers.
        Ok(())
    }

    /// Collects `VkAccelerationStructureInstanceKHR` entries for all resident chunk sections,
    /// using `RegionOrigin` (`(section_origin & !63) - camera_pos`) for shifted sections.
    pub fn build_tlas_instances(
        &self,
        camera_pos: [f64; 3],
    ) -> Vec<vk::AccelerationStructureInstanceKHR> {
        let keys = self.ordered_resident_slice_keys();
        let mut instances = Vec::with_capacity(keys.len());
        for key in keys {
            let Some(slice) = self.slices.get(&key) else {
                continue;
            };
            let Some(ref blas) = slice.blas else {
                continue;
            };
            if blas.device_address == 0 {
                continue;
            }
            let idx = instances.len() as u32;
            instances.push(Self::make_tlas_instance(
                slice,
                blas.device_address,
                idx,
                camera_pos,
            ));
        }
        instances
    }

    pub fn shutdown(&mut self, ctx: &VulkanContext, retire_queue: &RetireQueue) {
        for (_, old) in self.slices.drain() {
            Self::retire_slice_resources(
                old,
                Some(ctx),
                &self.slab_pool,
                &self.omm_builder,
                Some(retire_queue),
            );
        }
        self.lod_blas
            .clear(Some(ctx), &self.slab_pool, Some(retire_queue));
        self.emissive_lights.destroy(ctx, retire_queue);
        for tlas in &mut self.tlas_slots {
            tlas.destroy(ctx, &self.slab_pool, retire_queue);
        }
        let meta_slots = std::mem::take(&mut self.instance_meta_slots);
        self.instance_meta_counts = [0; 3];
        if meta_slots.iter().any(Option::is_some) {
            let dev = ctx.device.clone();
            retire_queue.retire_later(move || unsafe {
                for mut buf in meta_slots.into_iter().flatten() {
                    buf.destroy(&dev);
                }
            });
        }
        self.entity_blas.destroy(ctx, &self.slab_pool, retire_queue);
        if let Some(mut scratch) = self.blas_scratch.take() {
            let dev = ctx.device.clone();
            retire_queue.retire_later(move || unsafe {
                scratch.destroy(&dev);
            });
        }
        if let Some(mut ibo) = self.fallback_quad_ibo.take() {
            let dev = ctx.device.clone();
            retire_queue.retire_later(move || unsafe {
                ibo.destroy(&dev);
            });
        }
        if self.compaction_query_pool != vk::QueryPool::null() {
            let pool = self.compaction_query_pool;
            self.compaction_query_pool = vk::QueryPool::null();
            self.compaction_query_capacity = 0;
            self.pending_compactions.clear();
            let dev = ctx.device.clone();
            retire_queue.retire_later(move || unsafe {
                dev.destroy_query_pool(pool, None);
            });
        }
        self.wind_scheduler.destroy(ctx, retire_queue);
        self.omm_builder.destroy(ctx, retire_queue);
        self.sharc.destroy(ctx, retire_queue);

        let dev = ctx.device.clone();
        let pool = self.slab_pool.clone();
        retire_queue.retire_later(move || unsafe {
            pool.destroy_all(&dev);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_meta_gpu_matches_std430_layout() {
        assert_eq!(std::mem::size_of::<InstanceMetaGpu>(), 64);
        assert_eq!(std::mem::offset_of!(InstanceMetaGpu, vertex_bda), 0);
        assert_eq!(std::mem::offset_of!(InstanceMetaGpu, index_bda), 8);
        assert_eq!(std::mem::offset_of!(InstanceMetaGpu, vertex_stride), 16);
        assert_eq!(std::mem::offset_of!(InstanceMetaGpu, layer_ordinal), 20);
        assert_eq!(
            std::mem::offset_of!(InstanceMetaGpu, bindless_albedo_slot),
            24
        );
        assert_eq!(
            std::mem::offset_of!(InstanceMetaGpu, bindless_normal_slot),
            28
        );
        assert_eq!(
            std::mem::offset_of!(InstanceMetaGpu, bindless_specular_slot),
            32
        );
        assert_eq!(std::mem::offset_of!(InstanceMetaGpu, material_flags), 36);
        assert_eq!(std::mem::offset_of!(InstanceMetaGpu, region_origin_rel), 48);
    }

    #[test]
    fn register_slice_with_textures_and_defaults() {
        let mut mgr = ZeroCopyBlasManager::new();
        assert_eq!(
            mgr.instance_meta_buffer_for_slot(0),
            (vk::Buffer::null(), 0)
        );
        assert_eq!(mgr.instance_meta_count_for_slot(0), 0);

        mgr.register_slice(10, 0, 0, 0, 0, 1600, 40, 40, false, None, None);
        mgr.register_slice_with_textures(
            20, 1, 0, 0, 0, 1600, 40, 40, true, 11, 12, 13, 0x3, None, None,
        );

        let opaque = mgr
            .slices
            .get(&SectionLayerKey {
                allocation_key_id: 10,
                layer_ordinal: 0,
            })
            .unwrap();
        assert_eq!(opaque.bindless_albedo_slot, 0);
        assert_eq!(opaque.material_flags, 0);

        let foliage = mgr
            .slices
            .get(&SectionLayerKey {
                allocation_key_id: 20,
                layer_ordinal: 1,
            })
            .unwrap();
        assert_eq!(foliage.bindless_albedo_slot, 11);
        assert_eq!(foliage.bindless_normal_slot, 12);
        assert_eq!(foliage.bindless_specular_slot, 13);
        assert_eq!(foliage.material_flags, 0x3);
        assert!(foliage.has_foliage_wind);
    }

    #[test]
    fn rebuilds_nearby_edited_sections_before_the_streaming_backlog() {
        let mut mgr = ZeroCopyBlasManager::new();
        mgr.max_builds_per_frame = 2;
        for (key, section_x) in [(1, 30_i64), (2, 40), (3, -2), (4, -1)] {
            let section_pos = (section_x & 0x3F_FFFF) << 42;
            mgr.register_slice(key, 0, section_pos, 0, 0, 1600, 40, 40, false, None, None);
        }
        mgr.prioritize_nearby_builds([-0.5, 0.0, 0.0]);
        let mut first_batch: Vec<_> = mgr
            .pending_build_queue
            .iter()
            .take(2)
            .map(|key| key.allocation_key_id)
            .collect();
        first_batch.sort_unstable();
        assert_eq!(first_batch, [3, 4]);
    }

    #[test]
    fn vram_pressure_and_cutout_omm_hooks_update_state() {
        let mut mgr = ZeroCopyBlasManager::new();
        mgr.set_lod_cutoff_distance(200.0);
        mgr.apply_vram_pressure(true, 0.5);
        assert!(mgr.vram_throttle_compaction);
        assert_eq!(mgr.vram_lod_cutoff_scale, 0.5);
        assert_eq!(mgr.lod_blas.effective_cutoff_distance(), 100.0);

        mgr.set_wind_max_distance(32.0);
        assert_eq!(mgr.wind_scheduler.wind_max_distance, 32.0);

        mgr.register_slice(77, 1, 0, 0, 0, 1600, 40, 40, false, None, None);
        let omm_input = CutoutOmmBuildInput {
            triangles: vec![OmmTriangleEncoding::Uniform(
                vk::OpacityMicromapSpecialIndexEXT::FULLY_OPAQUE,
            )],
        };
        assert!(mgr.attach_cutout_omm_data(77, 1, omm_input.clone()));
        assert!(!mgr.attach_cutout_omm_data(77, 0, omm_input));
    }
}
