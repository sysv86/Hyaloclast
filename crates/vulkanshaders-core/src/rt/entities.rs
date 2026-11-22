use crate::{
    gpu::{
        VulkanContext,
        buffer::{GpuBufferAllocation, SlabArenaAllocator},
        retire::RetireQueue,
    },
    rt::as_builder::AccelerationStructureAllocation,
};
use ash::vk;

/// High-level ray-visibility class for dynamic entity batches (FFI-stable).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EntityGeometryKind {
    World,
    Crumbling,
    FirstPersonAvatar,
}

/// Fine-grained Minecraft 26.3 dynamic geometry category for RT material flags, shadow masking,
/// and anti-Z-fighting overlay bias.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EntityCategory {
    Mob = 0,
    Player = 1,
    /// Armor & elytra overlay layers — receives a small outward normal bias to avoid Z-fighting.
    ArmorOverlay = 2,
    /// Block entities (chests, signs, banners, beds).
    BlockEntity = 3,
    /// World particles (smoke, flames, spell effects).
    Particle = 4,
    /// Rain/snow weather curtains (excluded from opaque shadow occlusion).
    Weather = 5,
    /// Third-person or dropped held items.
    HeldItem = 6,
    /// First-person viewmodel hand / held item.
    FirstPersonHand = 7,
    /// Block breaking / crumbling overlay.
    CrumblingOverlay = 8,
}

impl EntityCategory {
    pub const fn from_u32(raw: u32) -> Self {
        match raw {
            0 => Self::Mob,
            1 => Self::Player,
            2 => Self::ArmorOverlay,
            3 => Self::BlockEntity,
            4 => Self::Particle,
            5 => Self::Weather,
            6 => Self::HeldItem,
            7 => Self::FirstPersonHand,
            8 => Self::CrumblingOverlay,
            _ => Self::Mob,
        }
    }

    pub const fn default_for_kind(kind: EntityGeometryKind) -> Self {
        match kind {
            EntityGeometryKind::World => Self::Mob,
            EntityGeometryKind::Crumbling => Self::CrumblingOverlay,
            EntityGeometryKind::FirstPersonAvatar => Self::FirstPersonHand,
        }
    }

    /// Returns the default `EntityGeometryFlags` bitmask for this category.
    pub const fn default_flags(self) -> u32 {
        match self {
            Self::Mob | Self::Player | Self::BlockEntity | Self::HeldItem => {
                ENTITY_FLAG_CAST_SHADOW | ENTITY_FLAG_RECEIVE_GI | ENTITY_FLAG_ALPHA_TESTED
            }
            Self::ArmorOverlay => {
                ENTITY_FLAG_CAST_SHADOW
                    | ENTITY_FLAG_RECEIVE_GI
                    | ENTITY_FLAG_ALPHA_TESTED
                    | ENTITY_FLAG_OVERLAY_BIAS
            }
            Self::Particle => ENTITY_FLAG_ALPHA_TESTED | ENTITY_FLAG_NO_SHADOW,
            Self::Weather => ENTITY_FLAG_TRANSLUCENT | ENTITY_FLAG_NO_SHADOW,
            Self::FirstPersonHand => {
                ENTITY_FLAG_CAST_SHADOW
                    | ENTITY_FLAG_RECEIVE_GI
                    | ENTITY_FLAG_ALPHA_TESTED
                    | ENTITY_FLAG_FIRST_PERSON
            }
            Self::CrumblingOverlay => ENTITY_FLAG_OVERLAY_BIAS | ENTITY_FLAG_NO_SHADOW,
        }
    }
}

/// Entity geometry material/visibility flags stored in `StagedEntityBatch` and propagated to
/// `InstanceMetaGpu::material_flags`.
pub const ENTITY_FLAG_CAST_SHADOW: u32 = 1 << 0;
pub const ENTITY_FLAG_RECEIVE_GI: u32 = 1 << 1;
pub const ENTITY_FLAG_ALPHA_TESTED: u32 = 1 << 2;
pub const ENTITY_FLAG_TRANSLUCENT: u32 = 1 << 3;
pub const ENTITY_FLAG_OVERLAY_BIAS: u32 = 1 << 4;
pub const ENTITY_FLAG_NO_SHADOW: u32 = 1 << 5;
pub const ENTITY_FLAG_FIRST_PERSON: u32 = 1 << 6;

/// Small outward normal bias (in blocks) applied to armor/elytra overlay vertices so they never
/// Z-fight with underlying player or mob skin triangles inside hardware BLASes.
pub const ARMOR_OVERLAY_NORMAL_BIAS: f32 = 0.0025;

/// Staged world feature geometry at a shared vertex layout, category, and material.
#[derive(Debug, Clone)]
pub struct StagedEntityBatch {
    pub vertex_stride: u32,
    pub is_quad: bool,
    pub bindless_albedo_slot: u32,
    pub bindless_normal_slot: u32,
    pub bindless_specular_slot: u32,
    pub kind: EntityGeometryKind,
    pub category: EntityCategory,
    pub entity_flags: u32,
    pub vertex_count: u32,
    pub bytes: Vec<u8>,
}

/// Per-batch BLAS + instance metadata produced by [`DynamicEntityBlasBuilder::record_frame_batches`].
#[derive(Clone, Copy)]
pub struct EntityInstanceRecord {
    pub instance: vk::AccelerationStructureInstanceKHR,
    pub vertex_bda: vk::DeviceAddress,
    pub index_bda: vk::DeviceAddress,
    pub vertex_stride: u32,
    pub bindless_albedo_slot: u32,
    pub bindless_normal_slot: u32,
    pub bindless_specular_slot: u32,
    pub kind: EntityGeometryKind,
    pub category: EntityCategory,
    pub entity_flags: u32,
}

/// Offsets each quad or triangle in `bytes` (stride `dst_stride`, first 12 bytes = `f32x3` pos)
/// outward along its geometric face normal by `bias` blocks to prevent overlay Z-fighting.
fn apply_overlay_normal_bias(bytes: &mut [u8], dst_stride: usize, is_quad: bool, bias: f32) {
    if dst_stride < 12 || !bias.is_finite() || bias == 0.0 {
        return;
    }
    let prim_verts = if is_quad { 4 } else { 3 };
    let prim_bytes = prim_verts * dst_stride;
    for chunk in bytes.chunks_exact_mut(prim_bytes) {
        let read_pos = |buf: &[u8], v: usize| -> [f32; 3] {
            let off = v * dst_stride;
            [
                f32::from_ne_bytes(buf[off..off + 4].try_into().unwrap()),
                f32::from_ne_bytes(buf[off + 4..off + 8].try_into().unwrap()),
                f32::from_ne_bytes(buf[off + 8..off + 12].try_into().unwrap()),
            ]
        };
        let p0 = read_pos(chunk, 0);
        let p1 = read_pos(chunk, 1);
        let p2 = read_pos(chunk, 2);
        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
        if len_sq <= 1e-12 {
            continue;
        }
        let inv = bias / len_sq.sqrt();
        let offset = [n[0] * inv, n[1] * inv, n[2] * inv];
        for v in 0..prim_verts {
            let off = v * dst_stride;
            let p = read_pos(chunk, v);
            chunk[off..off + 4].copy_from_slice(&(p[0] + offset[0]).to_ne_bytes());
            chunk[off + 4..off + 8].copy_from_slice(&(p[1] + offset[1]).to_ne_bytes());
            chunk[off + 8..off + 12].copy_from_slice(&(p[2] + offset[2]).to_ne_bytes());
        }
    }
}

/// Builds per-frame Bottom-Level Acceleration Structures (`BLAS`) for world feature geometry.
///
/// Batches sharing the layout, topology, texture, category, and visibility are coalesced into a
/// single vertex slice inside the frame slot's `vertex_buffers[slot]`, preserving exact per-batch
/// strides and binding each batch's `InstanceMetaGpu` cleanly.
pub struct DynamicEntityBlasBuilder {
    pub vertex_buffers: [Option<GpuBufferAllocation>; 3],
    pub tri_index_buffer: Option<GpuBufferAllocation>,
    pub tri_index_capacity: u32,
    pub blas_slots: [Vec<AccelerationStructureAllocation>; 3],
    pub scratch_buffer: Option<GpuBufferAllocation>,
    pub staged_batches: Vec<StagedEntityBatch>,
    pub staged_vertex_count: u32,
}

impl Default for DynamicEntityBlasBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicEntityBlasBuilder {
    pub const fn new() -> Self {
        Self {
            vertex_buffers: [None, None, None],
            tri_index_buffer: None,
            tri_index_capacity: 0,
            blas_slots: [Vec::new(), Vec::new(), Vec::new()],
            scratch_buffer: None,
            staged_batches: Vec::new(),
            staged_vertex_count: 0,
        }
    }

    pub fn begin_frame(&mut self) {
        self.staged_batches.clear();
        self.staged_vertex_count = 0;
    }

    /// Returns the maximum quad count across staged quad batches (used to size the shared quad IBO).
    pub fn max_quad_count(&self) -> u32 {
        self.staged_batches
            .iter()
            .filter(|b| b.is_quad)
            .map(|b| b.vertex_count / 4)
            .max()
            .unwrap_or(0)
    }

    /// Appends a heterogeneous entity/block-entity geometry batch preserving its native vertex stride
    /// (`28B`, `32B`, `36B`, `40B`, or any 4-byte-aligned stride `>= 24B`), topology (`is_quad` vs
    /// sequential triangles), and bindless texture slots (backward-compatible entrypoint).
    #[allow(clippy::too_many_arguments)]
    pub fn append_geometry_batch(
        &mut self,
        bytes: &[u8],
        vertex_count: u32,
        vertex_stride: u32,
        is_quad: bool,
        bindless_slot: u32,
        bindless_normal_slot: u32,
        bindless_specular_slot: u32,
        kind: EntityGeometryKind,
    ) {
        let category = EntityCategory::default_for_kind(kind);
        self.append_categorized_geometry_batch(
            bytes,
            vertex_count,
            vertex_stride,
            is_quad,
            bindless_slot,
            bindless_normal_slot,
            bindless_specular_slot,
            kind,
            category,
            category.default_flags(),
        );
    }

    /// Appends a categorized MC 26.3 dynamic geometry batch (mobs, players, armor/elytra, block
    /// entities, particles, weather, held items, first-person hand, or crumbling overlay).
    ///
    /// Armor/elytra batches (`EntityCategory::ArmorOverlay` or `ENTITY_FLAG_OVERLAY_BIAS`) receive
    /// a small outward face-normal bias (`ARMOR_OVERLAY_NORMAL_BIAS`) to prevent Z-fighting.
    #[allow(clippy::too_many_arguments)]
    pub fn append_categorized_geometry_batch(
        &mut self,
        bytes: &[u8],
        vertex_count: u32,
        vertex_stride: u32,
        is_quad: bool,
        bindless_slot: u32,
        bindless_normal_slot: u32,
        bindless_specular_slot: u32,
        kind: EntityGeometryKind,
        category: EntityCategory,
        entity_flags: u32,
    ) {
        if vertex_stride < 24 || bytes.is_empty() {
            return;
        }
        let align_verts = if is_quad { 4 } else { 3 };
        let src_stride = vertex_stride as usize;
        let max_complete_verts = (bytes.len() / src_stride) as u32;
        let usable_verts = (vertex_count.min(max_complete_verts) / align_verts) * align_verts;
        if usable_verts < align_verts {
            return;
        }

        // Keep 4-byte alignment so R32G32B32_SFLOAT vertex reads and uint loads stay aligned.
        let dst_stride_u32 = (vertex_stride + 3) & !3;
        let dst_stride = dst_stride_u32 as usize;

        let batch_idx = if let Some(idx) = self.staged_batches.iter().position(|b| {
            b.vertex_stride == dst_stride_u32
                && b.is_quad == is_quad
                && b.bindless_albedo_slot == bindless_slot
                && b.bindless_normal_slot == bindless_normal_slot
                && b.bindless_specular_slot == bindless_specular_slot
                && b.kind == kind
                && b.category == category
                && b.entity_flags == entity_flags
        }) {
            idx
        } else {
            self.staged_batches.push(StagedEntityBatch {
                vertex_stride: dst_stride_u32,
                is_quad,
                bindless_albedo_slot: bindless_slot,
                bindless_normal_slot,
                bindless_specular_slot,
                kind,
                category,
                entity_flags,
                vertex_count: 0,
                bytes: Vec::new(),
            });
            self.staged_batches.len() - 1
        };

        let batch = &mut self.staged_batches[batch_idx];
        let start_offset = batch.bytes.len();
        let byte_len = (usable_verts as usize) * src_stride;
        if dst_stride == src_stride {
            batch.bytes.extend_from_slice(&bytes[..byte_len]);
        } else {
            let pad = dst_stride - src_stride;
            batch.bytes.reserve((usable_verts as usize) * dst_stride);
            for v in 0..(usable_verts as usize) {
                let off = v * src_stride;
                batch.bytes.extend_from_slice(&bytes[off..off + src_stride]);
                batch.bytes.resize(batch.bytes.len() + pad, 0);
            }
        }

        if category == EntityCategory::ArmorOverlay
            || (entity_flags & ENTITY_FLAG_OVERLAY_BIAS) != 0
        {
            apply_overlay_normal_bias(
                &mut batch.bytes[start_offset..],
                dst_stride,
                is_quad,
                ARMOR_OVERLAY_NORMAL_BIAS,
            );
        }

        batch.vertex_count += usable_verts;
        self.staged_vertex_count += usable_verts;
    }

    fn ensure_tri_index_bda(
        &mut self,
        ctx: &VulkanContext,
        retire_queue: &RetireQueue,
        min_vertices: u32,
    ) -> Result<vk::DeviceAddress, vk::Result> {
        let needed = min_vertices.max(16_384);
        if self.tri_index_buffer.is_none() || self.tri_index_capacity < needed {
            let cap = needed.next_power_of_two();
            let indices: Vec<u32> = (0..cap).collect();
            let byte_slice: &[u8] = bytemuck::cast_slice(&indices);
            let mut ibo = GpuBufferAllocation::allocate_host_visible_bda(
                ctx,
                byte_slice.len() as vk::DeviceSize,
                vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
            )?;
            ibo.write_bytes(0, byte_slice);
            if let Some(mut old_ibo) = self.tri_index_buffer.take() {
                let dev = ctx.device.clone();
                retire_queue.retire_later(move || unsafe {
                    old_ibo.destroy(&dev);
                });
            }
            self.tri_index_buffer = Some(ibo);
            self.tri_index_capacity = cap;
        }
        Ok(self.tri_index_buffer.as_ref().unwrap().device_address)
    }

    /// Uploads all staged entity batches into `vertex_buffers[slot]`, records `PREFER_FAST_BUILD`
    /// BLAS builds on `cmd`, and returns per-batch [`EntityInstanceRecord`] entries with exact
    /// `vertex_bda`, `index_bda`, `vertex_stride`, texture slots, and category flags.
    pub fn record_frame_batches(
        &mut self,
        ctx: &VulkanContext,
        slab_pool: &SlabArenaAllocator,
        retire_queue: &RetireQueue,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
        shared_quad_index_bda: vk::DeviceAddress,
    ) -> Result<Vec<EntityInstanceRecord>, vk::Result> {
        let slot = frame_slot % 3;
        for old_blas in self.blas_slots[slot].drain(..) {
            old_blas.retire(ctx, slab_pool, retire_queue);
        }

        if self.staged_batches.is_empty() {
            return Ok(Vec::new());
        }
        let Some(ref as_ext) = ctx.accel_struct_ext else {
            return Ok(Vec::new());
        };

        // Compute 16-byte-aligned vertex offsets inside the frame slot's coalesced vertex buffer.
        let mut batch_offsets = Vec::with_capacity(self.staged_batches.len());
        let mut total_bytes: usize = 0;
        let mut max_tri_verts: u32 = 0;
        for batch in &self.staged_batches {
            let aligned_off = (total_bytes + 15) & !15;
            batch_offsets.push(aligned_off);
            total_bytes = aligned_off + batch.bytes.len();
            if !batch.is_quad {
                max_tri_verts = max_tri_verts.max(batch.vertex_count);
            }
        }
        if total_bytes == 0 {
            return Ok(Vec::new());
        }

        let needed_bytes = total_bytes as vk::DeviceSize;
        if self.vertex_buffers[slot]
            .as_ref()
            .is_none_or(|b| b.size < needed_bytes)
        {
            let alloc_bytes = needed_bytes + needed_bytes / 2 + 65_536;
            let new_vb = GpuBufferAllocation::allocate_host_visible_bda(
                ctx,
                alloc_bytes,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
            )?;
            if let Some(mut old_vb) = self.vertex_buffers[slot].take() {
                let dev = ctx.device.clone();
                retire_queue.retire_later(move || unsafe {
                    old_vb.destroy(&dev);
                });
            }
            self.vertex_buffers[slot] = Some(new_vb);
        }

        let vb_base_bda = {
            let vb = self.vertex_buffers[slot].as_mut().unwrap();
            for (batch, &off) in self.staged_batches.iter().zip(&batch_offsets) {
                vb.write_bytes(off, &batch.bytes);
            }
            vb.device_address
        };

        let tri_index_bda = if max_tri_verts > 0 {
            self.ensure_tri_index_bda(ctx, retire_queue, max_tri_verts)?
        } else {
            0
        };

        let identity = vk::TransformMatrixKHR {
            matrix: [
                1.0, 0.0, 0.0, 0.0, // row 0
                0.0, 1.0, 0.0, 0.0, // row 1
                0.0, 0.0, 1.0, 0.0, // row 2
            ],
        };

        struct PreparedEntityBuild {
            geometry: [vk::AccelerationStructureGeometryKHR<'static>; 1],
            range: [vk::AccelerationStructureBuildRangeInfoKHR; 1],
            new_blas: AccelerationStructureAllocation,
            scratch_offset: u64,
            vertex_bda: vk::DeviceAddress,
            index_bda: vk::DeviceAddress,
            vertex_stride: u32,
            bindless_albedo_slot: u32,
            bindless_normal_slot: u32,
            bindless_specular_slot: u32,
            kind: EntityGeometryKind,
            category: EntityCategory,
            entity_flags: u32,
        }

        let mut prepared: Vec<PreparedEntityBuild> = Vec::with_capacity(self.staged_batches.len());
        let mut total_scratch: u64 = 0;

        for (batch, &off) in self.staged_batches.iter().zip(&batch_offsets) {
            let primitive_count = if batch.is_quad {
                (batch.vertex_count / 4) * 2
            } else {
                batch.vertex_count / 3
            };
            if primitive_count == 0 {
                continue;
            }

            let vertex_bda = vb_base_bda + off as u64;
            let index_bda = if batch.is_quad {
                shared_quad_index_bda
            } else {
                tri_index_bda
            };

            let mut triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .vertex_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: vertex_bda,
                })
                .vertex_stride(u64::from(batch.vertex_stride))
                .max_vertex(batch.vertex_count - 1);

            if index_bda != 0 {
                triangles = triangles.index_type(vk::IndexType::UINT32).index_data(
                    vk::DeviceOrHostAddressConstKHR {
                        device_address: index_bda,
                    },
                );
            } else {
                triangles = triangles.index_type(vk::IndexType::NONE_KHR);
            }

            let geometry = [vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .flags(vk::GeometryFlagsKHR::empty())
                .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })];

            let probe_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD)
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
                    for p in prepared {
                        p.new_blas.retire(ctx, slab_pool, retire_queue);
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

            prepared.push(PreparedEntityBuild {
                geometry,
                range,
                new_blas,
                scratch_offset: total_scratch,
                vertex_bda,
                index_bda,
                vertex_stride: batch.vertex_stride,
                bindless_albedo_slot: batch.bindless_albedo_slot,
                bindless_normal_slot: batch.bindless_normal_slot,
                bindless_specular_slot: batch.bindless_specular_slot,
                kind: batch.kind,
                category: batch.category,
                entity_flags: batch.entity_flags,
            });
            total_scratch += aligned_scratch;
        }

        let mut records = Vec::with_capacity(prepared.len());
        if !prepared.is_empty() {
            if self
                .scratch_buffer
                .as_ref()
                .is_none_or(|s| s.size < total_scratch)
            {
                let alloc_size = (total_scratch + total_scratch / 2).max(256 * 1024);
                let new_scratch = match GpuBufferAllocation::allocate_as_scratch(ctx, alloc_size) {
                    Ok(s) => s,
                    Err(err) => {
                        for p in prepared {
                            p.new_blas.retire(ctx, slab_pool, retire_queue);
                        }
                        return Err(err);
                    }
                };
                if let Some(mut old_s) = self.scratch_buffer.take() {
                    let dev = ctx.device.clone();
                    retire_queue.retire_later(move || unsafe {
                        old_s.destroy(&dev);
                    });
                }
                self.scratch_buffer = Some(new_scratch);
            }

            let scratch_base_bda = self.scratch_buffer.as_ref().unwrap().device_address;
            let build_infos: Vec<vk::AccelerationStructureBuildGeometryInfoKHR<'_>> = prepared
                .iter()
                .map(|p| {
                    vk::AccelerationStructureBuildGeometryInfoKHR::default()
                        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD)
                        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                        .dst_acceleration_structure(p.new_blas.handle)
                        .geometries(&p.geometry)
                        .scratch_data(vk::DeviceOrHostAddressKHR {
                            device_address: scratch_base_bda + p.scratch_offset,
                        })
                })
                .collect();
            let range_slices: Vec<&[vk::AccelerationStructureBuildRangeInfoKHR]> =
                prepared.iter().map(|p| p.range.as_slice()).collect();

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

            for p in prepared {
                let blas_addr = p.new_blas.device_address;
                self.blas_slots[slot].push(p.new_blas);
                records.push(EntityInstanceRecord {
                    instance: vk::AccelerationStructureInstanceKHR {
                        transform: identity,
                        instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xFF),
                        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                            1,
                            vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw()
                                as u8,
                        ),
                        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                            device_handle: blas_addr,
                        },
                    },
                    vertex_bda: p.vertex_bda,
                    index_bda: p.index_bda,
                    vertex_stride: p.vertex_stride,
                    bindless_albedo_slot: p.bindless_albedo_slot,
                    bindless_normal_slot: p.bindless_normal_slot,
                    bindless_specular_slot: p.bindless_specular_slot,
                    kind: p.kind,
                    category: p.category,
                    entity_flags: p.entity_flags,
                });
            }
        }

        Ok(records)
    }

    /// Uploads the frame's dynamic entity vertices, records `PREFER_FAST_BUILD` BLAS builds on `cmd`,
    /// and returns the first `VkAccelerationStructureInstanceKHR` (backward-compatible helper).
    pub fn record_frame_blas(
        &mut self,
        ctx: &VulkanContext,
        slab_pool: &SlabArenaAllocator,
        retire_queue: &RetireQueue,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
        shared_quad_index_bda: vk::DeviceAddress,
    ) -> Result<Option<vk::AccelerationStructureInstanceKHR>, vk::Result> {
        let records = self.record_frame_batches(
            ctx,
            slab_pool,
            retire_queue,
            cmd,
            frame_slot,
            shared_quad_index_bda,
        )?;
        Ok(records.into_iter().next().map(|r| r.instance))
    }

    pub fn destroy(
        &mut self,
        ctx: &VulkanContext,
        slab_pool: &SlabArenaAllocator,
        retire_queue: &RetireQueue,
    ) {
        for slot_blases in &mut self.blas_slots {
            for blas in slot_blases.drain(..) {
                blas.retire(ctx, slab_pool, retire_queue);
            }
        }
        let dev = ctx.device.clone();
        let vbs = std::mem::take(&mut self.vertex_buffers);
        let tri_ibo = self.tri_index_buffer.take();
        let scratch = self.scratch_buffer.take();
        self.tri_index_capacity = 0;
        retire_queue.retire_later(move || unsafe {
            for mut vb in vbs.into_iter().flatten() {
                vb.destroy(&dev);
            }
            if let Some(mut ibo) = tri_ibo {
                ibo.destroy(&dev);
            }
            if let Some(mut s) = scratch {
                s.destroy(&dev);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use EntityGeometryKind::{Crumbling, FirstPersonAvatar, World};

    #[test]
    fn append_geometry_batch_preserves_heterogeneous_strides_and_coalesces_matching_keys() {
        let mut builder = DynamicEntityBlasBuilder::new();
        builder.append_geometry_batch(&[0x11u8; 4 * 28], 4, 28, true, 5, 11, 12, World);
        builder.append_geometry_batch(&[0x22u8; 4 * 32], 4, 32, true, 5, 11, 12, World);
        builder.append_geometry_batch(&[0x33u8; 4 * 36], 4, 36, true, 5, 11, 12, World);
        builder.append_geometry_batch(&[0x44u8; 4 * 40], 4, 40, true, 5, 11, 12, World);
        // Same key as first batch (28B, quad, slots 5/11/12) -> coalesces
        builder.append_geometry_batch(&[0x55u8; 4 * 28], 4, 28, true, 5, 11, 12, World);
        // Non-quad triangle batch (36B, !is_quad, slot 9)
        builder.append_geometry_batch(&[0x66u8; 6 * 36], 6, 36, false, 9, 0, 0, World);

        assert_eq!(builder.staged_batches.len(), 5);
        assert_eq!(builder.staged_batches[0].vertex_stride, 28);
        assert_eq!(builder.staged_batches[0].vertex_count, 8);
        assert_eq!(builder.staged_batches[0].bytes.len(), 8 * 28);
        assert_eq!(builder.staged_batches[1].vertex_stride, 32);
        assert_eq!(builder.staged_batches[2].vertex_stride, 36);
        assert_eq!(builder.staged_batches[3].vertex_stride, 40);
        assert!(!builder.staged_batches[4].is_quad);
        assert_eq!(builder.staged_batches[4].vertex_count, 6);
        assert_eq!(builder.staged_batches[4].bindless_albedo_slot, 9);
        assert_eq!(builder.max_quad_count(), 2);
        assert_eq!(builder.staged_vertex_count, 26);

        builder.begin_frame();
        assert!(builder.staged_batches.is_empty());
        assert_eq!(builder.staged_vertex_count, 0);
    }

    #[test]
    fn keeps_different_ray_visibility_batches_separate() {
        let mut builder = DynamicEntityBlasBuilder::new();
        builder.append_geometry_batch(&[0u8; 4 * 28], 4, 28, true, 5, 0, 0, World);
        builder.append_geometry_batch(&[0u8; 4 * 28], 4, 28, true, 5, 0, 0, Crumbling);
        builder.append_geometry_batch(&[0u8; 4 * 28], 4, 28, true, 5, 0, 0, FirstPersonAvatar);

        assert_eq!(builder.staged_batches.len(), 3);
        assert_eq!(builder.staged_batches[0].kind, World);
        assert_eq!(builder.staged_batches[1].kind, Crumbling);
        assert_eq!(builder.staged_batches[2].kind, FirstPersonAvatar);
    }

    #[test]
    fn armor_overlay_applies_outward_normal_bias_to_prevent_z_fighting() {
        // Quad on the z = 0 plane facing +Z (CCW: (0,0,0), (1,0,0), (1,1,0), (0,1,0))
        let mut verts = Vec::new();
        for pos in [
            [0.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ] {
            for c in pos {
                verts.extend_from_slice(&c.to_ne_bytes());
            }
            verts.extend_from_slice(&[0u8; 16]); // pad to 28B stride
        }
        let mut builder = DynamicEntityBlasBuilder::new();
        builder.append_categorized_geometry_batch(
            &verts,
            4,
            28,
            true,
            7,
            0,
            0,
            World,
            EntityCategory::ArmorOverlay,
            EntityCategory::ArmorOverlay.default_flags(),
        );
        assert_eq!(builder.staged_batches.len(), 1);
        let biased = &builder.staged_batches[0].bytes;
        let z0 = f32::from_ne_bytes(biased[8..12].try_into().unwrap());
        assert!((z0 - ARMOR_OVERLAY_NORMAL_BIAS).abs() < 1e-6);
    }
}
