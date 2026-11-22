use crate::gpu::{VulkanContext, buffer::GpuBufferAllocation, retire::RetireQueue};
use ash::vk;
use bytemuck::{Pod, Zeroable};

pub const MAX_EMISSIVE_LIGHTS: usize = 512;
const HELD_LIGHT_FLAG: u32 = 1;

/// Minecraft dimension identifier used for dimension-aware emissive intensity and ambient scaling.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DimensionId {
    #[default]
    Overworld = 0,
    Nether = 1,
    End = 2,
}

impl DimensionId {
    pub const fn from_i32(raw: i32) -> Self {
        match raw {
            1 => Self::Nether,
            2 => Self::End,
            _ => Self::Overworld,
        }
    }

    /// Returns `(emissive_intensity_scale, ambient_floor_scale)` tuned per dimension:
    /// - Overworld: baseline `(1.0, 1.0)`.
    /// - Nether: boosted warm lava/glowstone/fire emission and cavern ambient `(1.20, 1.35)`.
    /// - End: heightened end-rod/beacon contrast against the dark void `(1.30, 0.85)`.
    pub const fn emissive_and_ambient_scale(self) -> (f32, f32) {
        match self {
            Self::Overworld => (1.0, 1.0),
            Self::Nether => (1.20, 1.35),
            Self::End => (1.30, 0.85),
        }
    }
}

/// 48-byte `std430` GPU representation of an emissive point or area/triangle light source
/// (torches, lanterns, lava, glowstone, froglights, end rods, beacons, or player held light).
/// Matches `struct VsEmissiveLight` in `<vulkanshaders/v1/restir.vsl>`.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuEmissiveLight {
    /// `xyz`: camera-relative world position; `w`: attenuation cutoff radius in blocks.
    pub position_radius: [f32; 4],
    /// `rgb`: linear emissive color; `w`: radiant intensity multiplier.
    pub color_intensity: [f32; 4],
    /// `xyz`: surface normal (`(0,0,0)` for omnidirectional point light); `w`: `f32::from_bits(flags)`.
    pub normal_flags: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<GpuEmissiveLight>() == 48);

/// 16-byte header preceding `GpuEmissiveLight[]` in `EmissiveLightSSBO` (Set 1 binding 3).
/// Matches `uvec4 header; // x = activeLightCount, y = reservoirWidth, z = reservoirHeight, w = frameIndex`
/// in `rayquery_lighting.comp` and `pathtrace.rgen`.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuEmissiveLightHeader {
    pub light_count: u32,
    pub reservoir_width: u32,
    pub reservoir_height: u32,
    pub frame_index: u32,
}

const _: () = assert!(std::mem::size_of::<GpuEmissiveLightHeader>() == 16);

/// 32-byte `std430` ReSTIR DI Spatiotemporal Reservoir per cell (Set 1 binding 4).
/// Matches `struct VsRestirReservoir` in `<vulkanshaders/v1/restir.vsl>`.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuRestirReservoir {
    pub selected_light: u32,
    pub weight_sum: f32,
    pub target_pdf: f32,
    pub m_count: f32,
    pub hit_pos: [f32; 3],
    pub age_flags: u32,
}

const _: () = assert!(std::mem::size_of::<GpuRestirReservoir>() == 32);

/// World-space emissive light descriptor submitted from the client or extracted from emissive chunks.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WorldEmissiveLightDesc {
    pub world_pos: [f32; 3],
    pub radius: f32,
    pub color: [f32; 3],
    pub intensity: f32,
    pub normal: [f32; 3],
    pub flags: u32,
}

const _: () = assert!(std::mem::size_of::<WorldEmissiveLightDesc>() == 48);

/// Manages active emissive point/triangle light sources and ReSTIR DI spatiotemporal reservoir
/// buffers for Tier 2 (`rayquery_lighting.comp`) and Tier 3 (`pathtrace.rgen`).
pub struct EmissiveLightClusterManager {
    pub explicit_lights: Vec<WorldEmissiveLightDesc>,
    pub active_dimension: DimensionId,
    pub light_ssbo_slots: [Option<GpuBufferAllocation>; 3],
    pub reservoir_ssbo: Option<GpuBufferAllocation>,
    pub reservoir_dimensions: (u32, u32),
    pub viewport_dimensions: (u32, u32),
    pub last_uploaded_light_count: [u32; 3],
}

impl Default for EmissiveLightClusterManager {
    fn default() -> Self {
        Self::new()
    }
}

impl EmissiveLightClusterManager {
    pub fn new() -> Self {
        Self {
            explicit_lights: Vec::with_capacity(256),
            active_dimension: DimensionId::Overworld,
            light_ssbo_slots: [None, None, None],
            reservoir_ssbo: None,
            reservoir_dimensions: (0, 0),
            viewport_dimensions: (1920, 1080),
            last_uploaded_light_count: [0; 3],
        }
    }

    /// Sets the active Minecraft dimension (`0` = Overworld, `1` = Nether, `2` = End) for
    /// dimension-aware emissive intensity scaling.
    pub fn set_dimension(&mut self, dimension_id: i32) {
        self.active_dimension = DimensionId::from_i32(dimension_id);
    }

    /// Replaces the explicit world-space emissive light list submitted from Kotlin (`WorldStateCollector`).
    pub fn set_explicit_lights(&mut self, lights: &[WorldEmissiveLightDesc]) {
        self.explicit_lights.clear();
        let cap = lights.len().min(MAX_EMISSIVE_LIGHTS * 2);
        self.explicit_lights.extend_from_slice(&lights[..cap]);
    }

    pub fn set_viewport_dimensions(&mut self, width: u32, height: u32) {
        self.viewport_dimensions = (width.clamp(64, 4096), height.clamp(64, 4096));
    }

    pub fn light_buffer_for_slot(&self, frame_slot: usize) -> (vk::Buffer, u64) {
        let slot = frame_slot % 3;
        self.light_ssbo_slots[slot]
            .as_ref()
            .map(|b| (b.buffer, b.size))
            .unwrap_or((vk::Buffer::null(), 0))
    }

    pub fn reservoir_buffer(&self) -> (vk::Buffer, u64) {
        self.reservoir_ssbo
            .as_ref()
            .map(|b| (b.buffer, b.size))
            .unwrap_or((vk::Buffer::null(), 0))
    }

    /// Returns the full double-buffered ReSTIR DI reservoir SSBO (`(buffer, 0)`).
    /// Shaders index into `curBank = (header.w & 1u) * totalCells` and
    /// `prevBank = ((header.w ^ 1u) & 1u) * totalCells`.
    pub fn reservoir_buffer_for_slot(&self, _frame_slot: usize) -> (vk::Buffer, u64) {
        let (buf, _size) = self.reservoir_buffer();
        (buf, 0)
    }

    /// Returns the emissive light culling distance margin (in blocks) for the active GPU tier.
    pub const fn culling_margin_for_tier(tier: crate::gpu::GpuTierClass) -> f32 {
        match tier {
            crate::gpu::GpuTierClass::Enthusiast => 192.0,
            crate::gpu::GpuTierClass::High => 144.0,
            crate::gpu::GpuTierClass::Low | crate::gpu::GpuTierClass::Mid => 96.0,
        }
    }

    /// Returns the ReSTIR DI reservoir grid dimensions for the active GPU tier.
    /// High/Enthusiast GPUs run full-resolution reservoirs up to 3840x2160; Low/Mid GPUs use half-res.
    pub const fn reservoir_dimensions_for_tier(
        tier: crate::gpu::GpuTierClass,
        viewport_width: u32,
        viewport_height: u32,
    ) -> (u32, u32) {
        let clamped_w = if viewport_width < 64 {
            64
        } else if viewport_width > 4096 {
            4096
        } else {
            viewport_width
        };
        let clamped_h = if viewport_height < 64 {
            64
        } else if viewport_height > 4096 {
            4096
        } else {
            viewport_height
        };
        if tier.is_high_end() && clamped_w <= 3840 && clamped_h <= 2160 {
            (clamped_w, clamped_h)
        } else {
            (clamped_w.div_ceil(2), clamped_h.div_ceil(2))
        }
    }

    /// Converts explicit block lights to camera-relative positions with dimension-aware intensity
    /// scaling. The held source comes first and is submitted in camera-relative world space.
    pub fn build_camera_relative_lights(
        &self,
        camera_pos: [f64; 3],
        culling_margin: f32,
    ) -> Vec<GpuEmissiveLight> {
        let (dim_emissive_scale, _) = self.active_dimension.emissive_and_ambient_scale();
        let mut candidates = Vec::with_capacity(self.explicit_lights.len());
        for desc in &self.explicit_lights {
            let held = desc.flags & HELD_LIGHT_FLAG != 0;
            let [rx, ry, rz] = if held {
                desc.world_pos
            } else {
                std::array::from_fn(|axis| {
                    (f64::from(desc.world_pos[axis]) - camera_pos[axis]) as f32
                })
            };
            if !rx.is_finite() || !ry.is_finite() || !rz.is_finite() {
                continue;
            }
            let dist_sq = rx * rx + ry * ry + rz * rz;
            let radius = if desc.radius.is_finite() {
                desc.radius.clamp(2.0, 48.0)
            } else {
                12.0
            };
            let max_dist = radius + culling_margin;
            if dist_sq > max_dist * max_dist {
                continue;
            }
            let base_intensity = if desc.intensity.is_finite() {
                desc.intensity.clamp(0.05, 16.0)
            } else {
                1.0
            };
            let intensity = (base_intensity * dim_emissive_scale).clamp(0.05, 16.0);
            let score = if held {
                f32::INFINITY
            } else {
                intensity * radius / (dist_sq + 4.0)
            };
            candidates.push((
                score,
                GpuEmissiveLight {
                    position_radius: [rx, ry, rz, radius],
                    color_intensity: [
                        desc.color[0].clamp(0.0, 4.0),
                        desc.color[1].clamp(0.0, 4.0),
                        desc.color[2].clamp(0.0, 4.0),
                        intensity,
                    ],
                    normal_flags: [
                        desc.normal[0],
                        desc.normal[1],
                        desc.normal[2],
                        f32::from_bits(desc.flags),
                    ],
                },
            ));
        }

        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        candidates
            .into_iter()
            .take(MAX_EMISSIVE_LIGHTS)
            .map(|(_, l)| l)
            .collect()
    }

    /// Uploads the active frame's `GpuEmissiveLightHeader` + `GpuEmissiveLight[]` into
    /// `light_ssbo_slots[slot]` and ensures the ReSTIR DI reservoir SSBO is allocated.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_frame_lights_and_ensure_reservoirs(
        &mut self,
        ctx: &VulkanContext,
        retire_queue: &RetireQueue,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
        frame_index: u32,
        camera_pos: [f64; 3],
        viewport_width: u32,
        viewport_height: u32,
        use_reservoirs: bool,
    ) -> Result<(), vk::Result> {
        let slot = frame_slot % 3;
        let culling_margin = Self::culling_margin_for_tier(ctx.capabilities.gpu_tier_class);
        let lights = self.build_camera_relative_lights(camera_pos, culling_margin);
        self.last_uploaded_light_count[slot] = lights.len() as u32;

        let total_light_bytes = (std::mem::size_of::<GpuEmissiveLightHeader>()
            + MAX_EMISSIVE_LIGHTS * std::mem::size_of::<GpuEmissiveLight>())
            as vk::DeviceSize;

        if self.light_ssbo_slots[slot].is_none() {
            let alloc = GpuBufferAllocation::allocate_host_visible_bda(
                ctx,
                total_light_bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?;
            self.light_ssbo_slots[slot] = Some(alloc);
        }

        let (res_w, res_h) = Self::reservoir_dimensions_for_tier(
            ctx.capabilities.gpu_tier_class,
            viewport_width,
            viewport_height,
        );
        let mut shader_stages = vk::PipelineStageFlags2::COMPUTE_SHADER;
        if ctx.capabilities.ray_tracing_pipeline {
            shader_stages |= vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR;
        }
        if use_reservoirs
            && (self.reservoir_ssbo.is_none() || self.reservoir_dimensions != (res_w, res_h))
        {
            // Double-buffered (current + previous frame) reservoir table: 2 * res_w * res_h * 32 bytes
            let cell_count = u64::from(res_w) * u64::from(res_h);
            let res_bytes =
                (cell_count * 2 * (std::mem::size_of::<GpuRestirReservoir>() as u64)).max(65_536);
            if let Ok(new_res) = GpuBufferAllocation::allocate_device_local_bda(
                ctx,
                res_bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            ) {
                if let Some(mut old_res) = self.reservoir_ssbo.take() {
                    let dev = ctx.device.clone();
                    retire_queue.retire_later(move || unsafe {
                        old_res.destroy(&dev);
                    });
                }
                unsafe {
                    ctx.device
                        .cmd_fill_buffer(cmd, new_res.buffer, 0, vk::WHOLE_SIZE, 0);
                    ctx.cmd_memory_barrier2(
                        cmd,
                        vk::PipelineStageFlags2::ALL_TRANSFER,
                        vk::AccessFlags2::TRANSFER_WRITE,
                        shader_stages,
                        vk::AccessFlags2::SHADER_STORAGE_READ
                            | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    );
                }
                self.reservoir_ssbo = Some(new_res);
                self.reservoir_dimensions = (res_w, res_h);
            }
        }

        let header = GpuEmissiveLightHeader {
            light_count: lights.len() as u32,
            reservoir_width: self.reservoir_dimensions.0,
            reservoir_height: self.reservoir_dimensions.1,
            frame_index,
        };

        if let Some(ref mut buf) = self.light_ssbo_slots[slot] {
            buf.write_bytes(0, bytemuck::bytes_of(&header));
            if !lights.is_empty() {
                buf.write_bytes(
                    std::mem::size_of::<GpuEmissiveLightHeader>(),
                    bytemuck::cast_slice(&lights),
                );
            }
        }

        Ok(())
    }

    pub fn destroy(&mut self, ctx: &VulkanContext, retire_queue: &RetireQueue) {
        let slots = std::mem::take(&mut self.light_ssbo_slots);
        let res = self.reservoir_ssbo.take();
        self.reservoir_dimensions = (0, 0);
        let dev = ctx.device.clone();
        retire_queue.retire_later(move || unsafe {
            for mut b in slots.into_iter().flatten() {
                b.destroy(&dev);
            }
            if let Some(mut r) = res {
                r.destroy(&dev);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_scaling_adjusts_emissive_intensity_across_dimensions() {
        let mut mgr = EmissiveLightClusterManager::new();
        mgr.set_explicit_lights(&[WorldEmissiveLightDesc {
            world_pos: [4.0, 64.0, 0.0],
            radius: 14.0,
            color: [1.0, 0.8, 0.4],
            intensity: 2.0,
            normal: [0.0; 3],
            flags: 0,
        }]);

        mgr.set_dimension(0);
        let ow = mgr.build_camera_relative_lights([0.0, 64.0, 0.0], 96.0);
        assert_eq!(ow.len(), 1);
        assert!((ow[0].color_intensity[3] - 2.0).abs() < 1e-5);

        mgr.set_dimension(1);
        let nether = mgr.build_camera_relative_lights([0.0, 64.0, 0.0], 96.0);
        assert!((nether[0].color_intensity[3] - 2.4).abs() < 1e-5);

        mgr.set_dimension(2);
        let end = mgr.build_camera_relative_lights([0.0, 64.0, 0.0], 96.0);
        assert!((end[0].color_intensity[3] - 2.6).abs() < 1e-5);
    }
}
