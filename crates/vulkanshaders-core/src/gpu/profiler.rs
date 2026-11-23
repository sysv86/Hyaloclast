use ash::vk;
use std::{
    collections::VecDeque,
    sync::atomic::{AtomicBool, Ordering},
};

/// Resolved GPU execution duration for a single named render pass.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuPassTiming {
    pub name: String,
    pub duration_ms: f64,
}

#[derive(Debug, Clone)]
struct RecordedPass {
    name: String,
    start_offset: u32,
    end_offset: Option<u32>,
}

#[derive(Debug, Clone, Default)]
struct FrameQuerySlot {
    passes: Vec<RecordedPass>,
    next_query_offset: u32,
    active_in_frame: bool,
}

/// Ring-buffered per-pass Vulkan GPU timestamp profiler (`vk::QueryType::TIMESTAMP`).
///
/// Allocates `frames_in_flight * max_passes_per_frame * 2` timestamp queries in a single
/// `VkQueryPool`, resets each frame's slice on `begin_frame`, writes `TOP_OF_PIPE` /
/// `BOTTOM_OF_PIPE` timestamps around each pass, and resolves completed frames without blocking
/// the CPU via `VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT`.
pub struct GpuTimestampProfiler {
    query_pool: vk::QueryPool,
    timestamp_period_ns: f32,
    frames_in_flight: usize,
    max_passes_per_frame: usize,
    current_frame_slot: usize,
    enabled: AtomicBool,
    slots: Vec<FrameQuerySlot>,
    last_resolved: Vec<GpuPassTiming>,
    recent_frame_totals_ms: VecDeque<f64>,
}

impl GpuTimestampProfiler {
    pub const DEFAULT_FRAMES_IN_FLIGHT: usize = 3;
    pub const DEFAULT_MAX_PASSES: usize = 64;

    /// Creates a live `GpuTimestampProfiler` backed by a `VkQueryPool` (`QueryType::TIMESTAMP`).
    pub fn new(
        device: &ash::Device,
        timestamp_period_ns: f32,
        frames_in_flight: usize,
        max_passes_per_frame: usize,
    ) -> Result<Self, vk::Result> {
        let frames = frames_in_flight.max(1);
        let max_passes = max_passes_per_frame.max(1);
        let total_queries = (frames * max_passes * 2) as u32;
        let create_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(total_queries);
        let query_pool = unsafe { device.create_query_pool(&create_info, None)? };
        let valid_period = timestamp_period_ns.is_finite() && timestamp_period_ns > 0.0;
        Ok(Self {
            query_pool,
            timestamp_period_ns: if valid_period {
                timestamp_period_ns
            } else {
                1.0
            },
            frames_in_flight: frames,
            max_passes_per_frame: max_passes,
            current_frame_slot: 0,
            enabled: AtomicBool::new(valid_period),
            slots: vec![FrameQuerySlot::default(); frames],
            last_resolved: Vec::new(),
            recent_frame_totals_ms: VecDeque::with_capacity(60),
        })
    }

    /// Creates a headless/disabled profiler without a `VkQueryPool` (for disabled mode or tests).
    pub fn new_headless(
        timestamp_period_ns: f32,
        frames_in_flight: usize,
        max_passes_per_frame: usize,
    ) -> Self {
        let frames = frames_in_flight.max(1);
        let max_passes = max_passes_per_frame.max(1);
        let valid_period = timestamp_period_ns.is_finite() && timestamp_period_ns > 0.0;
        Self {
            query_pool: vk::QueryPool::null(),
            timestamp_period_ns: if valid_period {
                timestamp_period_ns
            } else {
                1.0
            },
            frames_in_flight: frames,
            max_passes_per_frame: max_passes,
            current_frame_slot: 0,
            enabled: AtomicBool::new(false),
            slots: vec![FrameQuerySlot::default(); frames],
            last_resolved: Vec::new(),
            recent_frame_totals_ms: VecDeque::with_capacity(60),
        }
    }

    /// Returns whether timestamp recording and resolution are currently enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    /// Toggles GPU timestamp profiling on or off at runtime.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
        if !enabled {
            self.last_resolved.clear();
            self.recent_frame_totals_ms.clear();
        }
    }

    /// Returns the physical device's timestamp period in nanoseconds per tick.
    pub fn timestamp_period_ns(&self) -> f32 {
        self.timestamp_period_ns
    }

    /// Returns the most recently resolved per-pass timings.
    pub fn last_timings(&self) -> &[GpuPassTiming] {
        &self.last_resolved
    }

    /// Returns the total duration in milliseconds across all passes in `last_timings()`.
    pub fn last_total_gpu_ms(&self) -> f64 {
        self.last_resolved.iter().map(|t| t.duration_ms).sum()
    }

    /// Returns a rolling mean after at least 30 completed frame samples, up to the latest 60.
    pub fn average_total_gpu_ms(&self) -> Option<f64> {
        (self.recent_frame_totals_ms.len() >= 30).then(|| {
            self.recent_frame_totals_ms.iter().sum::<f64>()
                / self.recent_frame_totals_ms.len() as f64
        })
    }

    fn record_frame_total(&mut self, timings: &[GpuPassTiming]) {
        if timings.is_empty() {
            return;
        }
        let total_ms = timings.iter().map(|timing| timing.duration_ms).sum::<f64>();
        if total_ms.is_finite() && total_ms >= 0.0 {
            if self.recent_frame_totals_ms.len() == 60 {
                self.recent_frame_totals_ms.pop_front();
            }
            self.recent_frame_totals_ms.push_back(total_ms);
        }
    }

    /// Converts raw GPU timestamp ticks `(start_ticks, end_ticks)` using `timestamp_period_ns`
    /// into milliseconds.
    #[inline]
    pub fn ticks_to_ms(start_ticks: u64, end_ticks: u64, timestamp_period_ns: f32) -> f64 {
        if !timestamp_period_ns.is_finite() || timestamp_period_ns <= 0.0 {
            return 0.0;
        }
        let delta_ticks = end_ticks.saturating_sub(start_ticks);
        (delta_ticks as f64) * f64::from(timestamp_period_ns) / 1_000_000.0
    }

    #[inline]
    fn queries_per_frame(&self) -> u32 {
        (self.max_passes_per_frame * 2) as u32
    }

    #[inline]
    fn frame_base_query(&self, frame_slot: usize) -> u32 {
        ((frame_slot % self.frames_in_flight) as u32) * self.queries_per_frame()
    }

    /// Begins a new profiled frame in `frame_slot`, resetting its slice of the `VkQueryPool`.
    ///
    /// # Safety
    /// `cmd` must be open and recording, and any prior GPU submission using `frame_slot` must
    /// have completed before resetting its queries.
    pub unsafe fn begin_frame(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
    ) {
        let slot_idx = frame_slot % self.frames_in_flight;
        self.current_frame_slot = slot_idx;
        let slot = &mut self.slots[slot_idx];
        slot.passes.clear();
        slot.next_query_offset = 0;
        slot.active_in_frame = false;

        if !self.is_enabled() || self.query_pool == vk::QueryPool::null() {
            return;
        }

        let base = self.frame_base_query(slot_idx);
        let count = self.queries_per_frame();
        unsafe {
            device.cmd_reset_query_pool(cmd, self.query_pool, base, count);
        }
        self.slots[slot_idx].active_in_frame = true;
    }

    /// Records the start timestamp for a named render pass on `cmd`.
    ///
    /// # Safety
    /// `cmd` must be open and recording after [`Self::begin_frame`].
    pub unsafe fn begin_pass(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        name: impl Into<String>,
    ) {
        if !self.is_enabled() || self.query_pool == vk::QueryPool::null() {
            return;
        }
        let slot_idx = self.current_frame_slot;
        let max_queries = self.queries_per_frame();
        let base = self.frame_base_query(slot_idx);
        let slot = &mut self.slots[slot_idx];
        if !slot.active_in_frame || slot.next_query_offset + 2 > max_queries {
            return;
        }
        let start_offset = slot.next_query_offset;
        slot.next_query_offset += 1;
        slot.passes.push(RecordedPass {
            name: name.into(),
            start_offset,
            end_offset: None,
        });
        unsafe {
            device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pool,
                base + start_offset,
            );
        }
    }

    /// Records the end timestamp for a named render pass on `cmd`.
    ///
    /// # Safety
    /// `cmd` must be open and recording after a matching [`Self::begin_pass`].
    pub unsafe fn end_pass(&mut self, device: &ash::Device, cmd: vk::CommandBuffer, name: &str) {
        if !self.is_enabled() || self.query_pool == vk::QueryPool::null() {
            return;
        }
        let slot_idx = self.current_frame_slot;
        let max_queries = self.queries_per_frame();
        let base = self.frame_base_query(slot_idx);
        let slot = &mut self.slots[slot_idx];
        if !slot.active_in_frame || slot.next_query_offset >= max_queries {
            return;
        }
        let pass_index = slot
            .passes
            .iter()
            .rev()
            .position(|p| p.end_offset.is_none() && p.name == name)
            .map(|reverse_index| slot.passes.len() - reverse_index - 1)
            .or_else(|| {
                slot.passes
                    .iter()
                    .rev()
                    .position(|p| p.end_offset.is_none())
                    .map(|reverse_index| slot.passes.len() - reverse_index - 1)
            });
        let Some(pass) = pass_index.and_then(|index| slot.passes.get_mut(index)) else {
            return;
        };
        let end_offset = slot.next_query_offset;
        slot.next_query_offset += 1;
        pass.end_offset = Some(end_offset);
        unsafe {
            device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                base + end_offset,
            );
        }
    }

    /// Resolves GPU timestamps for the completed frame slot (the oldest ring slot, or the
    /// specified slot via [`Self::resolve_frame_slot`]) without blocking on the GPU.
    ///
    /// # Safety
    /// `device` must be the same `ash::Device` that created this profiler.
    pub unsafe fn resolve_completed_frame(&mut self, device: &ash::Device) -> Vec<GpuPassTiming> {
        let completed_slot = (self.current_frame_slot + 1) % self.frames_in_flight;
        unsafe { self.resolve_frame_slot(device, completed_slot) }
    }

    /// Resolves GPU timestamps for an explicit `frame_slot`.
    ///
    /// # Safety
    /// `device` must be the same `ash::Device` that created this profiler.
    pub unsafe fn resolve_frame_slot(
        &mut self,
        device: &ash::Device,
        frame_slot: usize,
    ) -> Vec<GpuPassTiming> {
        if !self.is_enabled() || self.query_pool == vk::QueryPool::null() {
            return Vec::new();
        }
        let slot_idx = frame_slot % self.frames_in_flight;
        let query_count = self.slots[slot_idx].next_query_offset;
        if !self.slots[slot_idx].active_in_frame || query_count == 0 {
            return Vec::new();
        }

        let base = self.frame_base_query(slot_idx);
        // Each query yields [timestamp_u64, availability_u64] (stride = 16 bytes) with WITH_AVAILABILITY.
        let mut raw = vec![[0u64; 2]; query_count as usize];
        let flags = vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY;
        let res = unsafe { device.get_query_pool_results(self.query_pool, base, &mut raw, flags) };
        if let Err(err) = res
            && err != vk::Result::NOT_READY
        {
            return Vec::new();
        }

        let mut timings = Vec::with_capacity(self.slots[slot_idx].passes.len());
        for pass in &self.slots[slot_idx].passes {
            let Some(end_off) = pass.end_offset else {
                continue;
            };
            let s_idx = pass.start_offset as usize;
            let e_idx = end_off as usize;
            if e_idx >= raw.len() || s_idx >= raw.len() {
                continue;
            }
            let [start_ticks, start_avail] = raw[s_idx];
            let [end_ticks, end_avail] = raw[e_idx];
            if start_avail == 0 || end_avail == 0 {
                continue;
            }
            let duration_ms = Self::ticks_to_ms(start_ticks, end_ticks, self.timestamp_period_ns);
            timings.push(GpuPassTiming {
                name: pass.name.clone(),
                duration_ms,
            });
        }

        if !timings.is_empty() {
            self.last_resolved = timings.clone();
            self.record_frame_total(&timings);
        }
        timings
    }

    /// Records synthetic pass queries in headless/unit-test mode and resolves them from raw
    /// `(start_tick, end_tick)` pairs to verify timing conversion and ring-buffer state.
    pub fn resolve_synthetic_passes(
        &mut self,
        frame_slot: usize,
        pass_ticks: &[(&str, u64, u64)],
    ) -> Vec<GpuPassTiming> {
        let slot_idx = frame_slot % self.frames_in_flight;
        self.current_frame_slot = slot_idx;
        let slot = &mut self.slots[slot_idx];
        slot.passes.clear();
        slot.next_query_offset = 0;
        slot.active_in_frame = true;

        let mut timings = Vec::with_capacity(pass_ticks.len().min(self.max_passes_per_frame));
        for &(name, start_tick, end_tick) in pass_ticks.iter().take(self.max_passes_per_frame) {
            let start_offset = slot.next_query_offset;
            let end_offset = start_offset + 1;
            slot.next_query_offset += 2;
            slot.passes.push(RecordedPass {
                name: name.to_string(),
                start_offset,
                end_offset: Some(end_offset),
            });
            timings.push(GpuPassTiming {
                name: name.to_string(),
                duration_ms: Self::ticks_to_ms(start_tick, end_tick, self.timestamp_period_ns),
            });
        }
        self.last_resolved = timings.clone();
        self.record_frame_total(&timings);
        timings
    }

    /// Destroys the underlying `VkQueryPool`.
    ///
    /// # Safety
    /// Must only be called after the GPU has finished all command buffers referencing this pool.
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        if self.query_pool != vk::QueryPool::null() {
            unsafe { device.destroy_query_pool(self.query_pool, None) };
            self.query_pool = vk::QueryPool::null();
        }
        self.enabled.store(false, Ordering::Release);
        for slot in &mut self.slots {
            slot.passes.clear();
            slot.next_query_offset = 0;
            slot.active_in_frame = false;
        }
    }
}
