use crate::gpu::{VulkanContext, buffer::GpuBufferAllocation, retire::RetireQueue};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use std::collections::HashSet;

/// Matches `struct WindJobGpu` in `wind_displace.comp` (`48` bytes, `std430`).
///
/// Crucially:
/// - `positions_address` points to a dedicated dynamic `SlabSubAllocation` buffer owned by
///   the section slice (`SectionSliceDescriptor::displaced_vertex_buf`).
/// - `rest_address` points to the immutable static rest-pose geometry in `VulkanTurbo`'s
///   `UberGpuBuffer` (`base_bda + slice.byte_offset`).
///
/// This guarantees `wind_displace.comp` never overwrites static terrain geometry or accumulates
/// unbounded vertex drift across frames.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WindJobGpu {
    pub positions_address: u64,
    pub rest_address: u64,
    pub vertex_count: u32,
    pub scale: f32,
    pub wrapped_x: f32,
    pub wrapped_y: f32,
    pub wrapped_z: f32,
    pub relative_x: f32,
    pub relative_y: f32,
    pub relative_z: f32,
}

const _: () = assert!(std::mem::size_of::<WindJobGpu>() == 48);

/// Matches `push_constant WindPushConstants` in `wind_displace.comp` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WindPushConstants {
    pub jobs_address: u64,
    pub job_count: u32,
    pub time_sec: f32,
    pub strength: f32,
    pub weather_multiplier: f32,
}

const _: () = assert!(std::mem::size_of::<WindPushConstants>() == 24);

/// Maximum horizontal foliage displacement (in blocks) used to prevent leaf/trunk seams.
pub const MAX_FOLIAGE_DISPLACEMENT_BLOCKS: f32 = 0.14;

/// Fractional block height threshold below which base/stem vertices stay anchored to the trunk or ground.
pub const BASE_ANCHOR_HEIGHT_FRACTION: f32 = 0.08;

/// Computes the vertical anchor weight `[0.0, 1.0]` for a foliage vertex at block-local height `local_y`.
/// Bottom vertices near the block floor (`fract(local_y) <= BASE_ANCHOR_HEIGHT_FRACTION`) return `0.0`
/// so stems and leaf bases never tear away from static trunks or soil blocks.
pub fn compute_vertex_wind_weight(local_y: f32, anchor_bottom: bool) -> f32 {
    if !local_y.is_finite() {
        return 0.0;
    }
    if !anchor_bottom {
        return 1.0;
    }
    let frac_y = local_y.rem_euclid(1.0);
    if frac_y <= BASE_ANCHOR_HEIGHT_FRACTION {
        0.0
    } else {
        ((frac_y - BASE_ANCHOR_HEIGHT_FRACTION) / (1.0 - BASE_ANCHOR_HEIGHT_FRACTION))
            .clamp(0.0, 1.0)
    }
}

/// Clamps horizontal wind displacement `(dx, dz)` with base-vertex anchoring and a maximum
/// radial magnitude (`max_displacement`) to eliminate leaf/trunk seam artifacts.
pub fn clamp_wind_displacement(
    dx: f32,
    dz: f32,
    local_y: f32,
    anchor_bottom: bool,
    max_displacement: f32,
) -> (f32, f32) {
    let weight = compute_vertex_wind_weight(local_y, anchor_bottom);
    if weight <= 0.0 || !max_displacement.is_finite() || max_displacement <= 0.0 {
        return (0.0, 0.0);
    }
    let sx = if dx.is_finite() { dx * weight } else { 0.0 };
    let sz = if dz.is_finite() { dz * weight } else { 0.0 };
    let len_sq = sx * sx + sz * sz;
    let max_sq = max_displacement * max_displacement;
    if len_sq > max_sq && len_sq > 0.0 {
        let inv = max_displacement / len_sq.sqrt();
        (sx * inv, sz * inv)
    } else {
        (sx, sz)
    }
}

/// Schedules near-camera foliage wind compute jobs (`wind_displace.comp`) gated by configurable
/// `wind_max_distance` and tracks sections that exited the wind radius on the previous frame so
/// they are reset back to rest (`scale = 0.0`) before freezing their BLAS.
pub struct WindRefitScheduler {
    pub wind_period_blocks: i32,
    /// User/runtime-configurable maximum wind distance in blocks (`0.0` disables wind refits).
    pub wind_max_distance: f32,
    /// Maximum allowed per-vertex horizontal displacement in blocks.
    pub max_displacement_blocks: f32,
    pub job_buffers: [Option<GpuBufferAllocation>; 3],
    pub active_jobs: Vec<WindJobGpu>,
    pub prev_active_keys: HashSet<(u64, u32)>,
    pub current_active_keys: HashSet<(u64, u32)>,
    pub max_job_vertex_count: u32,
    pub last_push_constants: Option<WindPushConstants>,
}

impl Default for WindRefitScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl WindRefitScheduler {
    pub const WIND_REACH_BLOCKS: f64 = 48.0;
    pub const WIND_PERIOD_BLOCKS: i32 = 1024;

    /// Returns `(max_animated_sections, reach_blocks)` scaled by the active GPU tier.
    pub const fn budget_for_tier(tier: crate::gpu::GpuTierClass) -> (usize, f64) {
        match tier {
            crate::gpu::GpuTierClass::Enthusiast => (64, 96.0),
            crate::gpu::GpuTierClass::High => (48, 72.0),
            crate::gpu::GpuTierClass::Low | crate::gpu::GpuTierClass::Mid => (24, 48.0),
        }
    }

    pub fn new() -> Self {
        Self {
            wind_period_blocks: Self::WIND_PERIOD_BLOCKS,
            wind_max_distance: Self::WIND_REACH_BLOCKS as f32,
            max_displacement_blocks: MAX_FOLIAGE_DISPLACEMENT_BLOCKS,
            job_buffers: [None, None, None],
            active_jobs: Vec::new(),
            prev_active_keys: HashSet::new(),
            current_active_keys: HashSet::new(),
            max_job_vertex_count: 0,
            last_push_constants: None,
        }
    }

    /// Configures the maximum distance (in blocks) at which foliage chunks run wind compute and
    /// BLAS refits. Passing `0.0` cleanly disables wind animation and resets active sections.
    pub fn set_wind_max_distance(&mut self, blocks: f32) {
        if blocks.is_finite() {
            self.wind_max_distance = blocks.clamp(0.0, 256.0);
        }
    }

    /// Returns the effective wind reach in blocks, bounded by both GPU tier budget and
    /// `self.wind_max_distance`.
    pub fn effective_reach_blocks(&self, tier: crate::gpu::GpuTierClass) -> f64 {
        let (_, tier_reach) = Self::budget_for_tier(tier);
        if self.wind_max_distance <= 0.0 {
            0.0
        } else {
            tier_reach.min(f64::from(self.wind_max_distance))
        }
    }

    /// Computes smooth distance attenuation `[0.0, 1.0]` for a section at `dist_blocks` so foliage
    /// gently settles to rest pose before reaching `max_distance` instead of snapping.
    pub fn distance_attenuation(dist_blocks: f64, max_distance: f64) -> f32 {
        if max_distance <= 0.0 || !dist_blocks.is_finite() || dist_blocks >= max_distance {
            return 0.0;
        }
        if dist_blocks <= 0.0 {
            return 1.0;
        }
        let t = (dist_blocks / max_distance).clamp(0.0, 1.0) as f32;
        // Smooth hermite fade over the outer 40% of the wind radius.
        let fade = ((t - 0.60) / 0.40).clamp(0.0, 1.0);
        1.0 - fade * fade * (3.0 - 2.0 * fade)
    }

    pub const fn floor_mod(val: i32, period: i32) -> i32 {
        let r = val % period;
        if r < 0 { r + period } else { r }
    }

    /// Uploads `active_jobs` for `frame_slot` (`0..3`) and returns the `WindPushConstants`
    /// ready for `vkCmdPushConstants` + `vkCmdDispatch`.
    pub fn upload_frame_jobs(
        &mut self,
        ctx: &VulkanContext,
        retire_queue: &RetireQueue,
        frame_slot: usize,
        time_sec: f32,
        strength: f32,
        weather_multiplier: f32,
    ) -> Result<Option<WindPushConstants>, vk::Result> {
        if self.active_jobs.is_empty() {
            self.max_job_vertex_count = 0;
            self.last_push_constants = None;
            return Ok(None);
        }
        let slot = frame_slot % 3;
        let needed_bytes = self.active_jobs.len() * std::mem::size_of::<WindJobGpu>();

        if self.job_buffers[slot]
            .as_ref()
            .is_none_or(|b| (b.size as usize) < needed_bytes)
        {
            let alloc_bytes =
                (needed_bytes + needed_bytes / 2 + 16 * std::mem::size_of::<WindJobGpu>())
                    as vk::DeviceSize;
            let new_buf = GpuBufferAllocation::allocate_host_visible_bda(
                ctx,
                alloc_bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?;
            if let Some(mut old) = self.job_buffers[slot].take() {
                let dev = ctx.device.clone();
                retire_queue.retire_later(move || unsafe {
                    old.destroy(&dev);
                });
            }
            self.job_buffers[slot] = Some(new_buf);
        }

        self.max_job_vertex_count = self
            .active_jobs
            .iter()
            .map(|j| j.vertex_count)
            .max()
            .unwrap_or(0);

        let buf = self.job_buffers[slot].as_mut().unwrap();
        buf.write_bytes(0, bytemuck::cast_slice(&self.active_jobs));

        let clamped_strength = strength.clamp(0.0, 2.0);
        let clamped_weather = weather_multiplier.clamp(0.0, 2.5);
        let pc = WindPushConstants {
            jobs_address: buf.device_address,
            job_count: self.active_jobs.len() as u32,
            time_sec,
            strength: clamped_strength,
            weather_multiplier: clamped_weather,
        };
        self.last_push_constants = Some(pc);
        Ok(Some(pc))
    }

    pub fn destroy(&mut self, ctx: &VulkanContext, retire_queue: &RetireQueue) {
        let dev = ctx.device.clone();
        let bufs = std::mem::take(&mut self.job_buffers);
        self.prev_active_keys.clear();
        self.current_active_keys.clear();
        self.active_jobs.clear();
        self.last_push_constants = None;
        retire_queue.retire_later(move || unsafe {
            for b in bufs.into_iter().flatten() {
                let mut b = b;
                b.destroy(&dev);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuTierClass;

    #[test]
    fn wind_max_distance_gates_effective_reach_and_disables_at_zero() {
        let mut sched = WindRefitScheduler::new();
        assert_eq!(sched.effective_reach_blocks(GpuTierClass::Enthusiast), 48.0);

        sched.set_wind_max_distance(128.0);
        assert_eq!(sched.effective_reach_blocks(GpuTierClass::Enthusiast), 96.0);
        assert_eq!(sched.effective_reach_blocks(GpuTierClass::Low), 48.0);

        sched.set_wind_max_distance(0.0);
        assert_eq!(sched.effective_reach_blocks(GpuTierClass::Enthusiast), 0.0);
    }

    #[test]
    fn distance_attenuation_fades_smoothly_to_zero_at_boundary() {
        assert_eq!(WindRefitScheduler::distance_attenuation(0.0, 48.0), 1.0);
        assert_eq!(WindRefitScheduler::distance_attenuation(20.0, 48.0), 1.0);
        let mid = WindRefitScheduler::distance_attenuation(38.4, 48.0);
        assert!(mid > 0.0 && mid < 1.0);
        assert_eq!(WindRefitScheduler::distance_attenuation(48.0, 48.0), 0.0);
        assert_eq!(WindRefitScheduler::distance_attenuation(64.0, 48.0), 0.0);
    }

    #[test]
    fn base_vertex_anchoring_and_displacement_clamping_prevent_seams() {
        // Bottom vertex on block floor (y = 64.02) stays anchored at 0.0
        assert_eq!(compute_vertex_wind_weight(64.02, true), 0.0);
        assert_eq!(
            clamp_wind_displacement(0.5, -0.5, 64.02, true, MAX_FOLIAGE_DISPLACEMENT_BLOCKS),
            (0.0, 0.0)
        );

        // Top vertex (y = 64.95) gets clamped to MAX_FOLIAGE_DISPLACEMENT_BLOCKS
        let (dx, dz) =
            clamp_wind_displacement(0.5, 0.5, 64.95, true, MAX_FOLIAGE_DISPLACEMENT_BLOCKS);
        let mag = (dx * dx + dz * dz).sqrt();
        assert!((mag - MAX_FOLIAGE_DISPLACEMENT_BLOCKS).abs() < 1e-4);
    }
}
