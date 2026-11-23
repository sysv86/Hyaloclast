//! `vulkanshaders-core` — Native Rust Vulkan 1.3+ shader, ray-tracing, and frame-generation core
//! for **VulkanShaders** (`Hyaloclast` / codename `IGNIS`).
//!
//! Architectural invariants:
//! 1. Shares the exact `VkInstance`, `VkPhysicalDevice`, `VkDevice`, `VkQueue`, and `VmaAllocator`
//!    created by Minecraft 26.3's `RenderPearl` Vulkan backend.
//! 2. Synchronizes all `VkQueue` submissions (`vkQueueSubmit`, `vkQueueSubmit2KHR`, `vkQueuePresentKHR`,
//!    `vkQueueWaitIdle`) across Java (`RenderPearl` + `VulkanTurbo`) and Rust (`PresentThread` + async
//!    BLAS builders) via a reentrant queue mutex (`gpu::queue_lock`).
//! 3. Builds Zero-Copy Bottom-Level Acceleration Structures (`BLAS`) directly over `VulkanTurbo`'s
//!    40-byte (`IGNIS_TERRAIN_40B`) and 28-byte `UberGpuBuffer` slices using `vkGetBufferDeviceAddress`,
//!    placing Top-Level (`TLAS`) instances at `RegionOrigin` (`(section_origin & !63) - camera_pos`)
//!    when `is_region_shifted` is set so ray-traced hits match rasterized G-Buffer geometry bit-for-bit.
//! 4. Executes a Dead-Pass-Culled DAG Render Graph (`graph::RenderGraph`) that completely eliminates
//!    disabled stages (Denoiser, Upscaler, FrameGen, Ray Tracing) with 0 ms GPU/CPU overhead.
//! 5. Each rendering subsystem owns an independent `Mutex`/`RwLock`. The global mutex is held only
//!    for the duration of cloning an `Arc<IgnisEngine>` — never while a background texture upload,
//!    slice registration, or shader compilation is in progress. The render thread therefore never
//!    stalls behind unrelated background work.

pub mod crash;
pub mod ffi;
pub mod framegen;
pub mod gpu;
pub mod graph;
pub mod modules;
pub mod rt;
pub mod sdk;

use framegen::FrameGenController;
use gpu::{
    GpuTimestampProfiler, PlatformTierPolicy, VramBudgetSnapshot, VramPressurePolicy,
    VulkanContext,
    bindless::{BindlessTextureTable, PackTextureTable},
    retire::RetireQueue,
};
use graph::RenderGraph;
use modules::{
    AsyncPipelinePrewarmer, GpuPipelineManager, PersistentPipelineDiskCache, WorldUboData,
};
use parking_lot::{Mutex, RwLock};
use rt::ZeroCopyBlasManager;
use sdk::RuntimeSdkManager;
use std::sync::{
    Arc,
    atomic::{AtomicU32, AtomicU64, Ordering},
};
use vsp_compiler::VspCompiler;

/// Pipeline compilation + mutable GPU-pipeline state. Isolated behind its own lock so that shader
/// recompilation never blocks bindless texture registration, frame generation, or SDK dispatch.
pub struct PipelinesAndCompiler {
    pub pipelines: GpuPipelineManager,
    pub compiler: VspCompiler,
}

/// Engine state after `vs_core_init`. Every subsystem has its own lock; `ctx` is read-mostly and
/// therefore behind an `RwLock`, while monotonically-increasing counters use atomics.
pub struct IgnisEngine {
    pub ctx: RwLock<VulkanContext>,
    pub retire: RetireQueue,
    pub bindless: RwLock<BindlessTextureTable>,
    pub pack_textures: RwLock<PackTextureTable>,
    pub blas_manager: Mutex<ZeroCopyBlasManager>,
    pub pipelines_and_compiler: Mutex<PipelinesAndCompiler>,
    pub framegen: Mutex<FrameGenController>,
    pub sdk_manager: Mutex<RuntimeSdkManager>,
    pub render_graph: Mutex<RenderGraph>,
    pub world_ubo: RwLock<WorldUboData>,
    pub frame_index: AtomicU64,
    pub last_built_tlas_handle: AtomicU64,
    pub framegen_mode: AtomicU32,
    pub timestamp_profiler: Mutex<Option<GpuTimestampProfiler>>,
    pub pipeline_disk_cache: Option<PersistentPipelineDiskCache>,
    pub active_prewarmer: Mutex<Option<AsyncPipelinePrewarmer>>,
    pub last_vram_budget: RwLock<Option<VramBudgetSnapshot>>,
    pub vram_poll_frame_counter: AtomicU64,
    pub last_vram_trim_frame: AtomicU64,
}

/// Alias kept for callers referencing the engine by its crate-level name.
pub type VulkanShadersEngine = IgnisEngine;

impl IgnisEngine {
    /// Number of frame updates between periodic `VK_EXT_memory_budget` VRAM polls.
    pub const VRAM_POLL_INTERVAL_FRAMES: u64 = 30;

    /// Minimum number of frames between consecutive critical-VRAM reservoir/target trims so
    /// memory pressure relief does not thrash allocations every frame.
    pub const VRAM_TRIM_COOLDOWN_FRAMES: u64 = 180;

    /// Returns the current three-frame-in-flight ring slot.
    #[inline]
    pub fn frame_slot(&self) -> usize {
        (self.frame_index.load(Ordering::Acquire) % 3) as usize
    }

    /// Advances the frame counter and returns the (new) ring slot.
    #[inline]
    pub fn next_frame_slot(&self) -> usize {
        let idx = self.frame_index.fetch_add(1, Ordering::AcqRel);
        (idx.wrapping_add(1) % 3) as usize
    }

    /// Queries live VRAM budget every [`Self::VRAM_POLL_INTERVAL_FRAMES`] frames (or immediately
    /// when `force` is `true`), caches the snapshot, and applies the derived [`VramPressurePolicy`]
    /// to `blas_manager` and (under critical pressure, rate-limited by
    /// [`Self::VRAM_TRIM_COOLDOWN_FRAMES`]) trims optional render-graph history/reservoir targets
    /// and resets the SHaRC spatial cache.
    pub fn poll_vram_budget_if_due(&self, force: bool) -> VramBudgetSnapshot {
        let frame_count = self.vram_poll_frame_counter.fetch_add(1, Ordering::AcqRel);
        if !force
            && !frame_count.is_multiple_of(Self::VRAM_POLL_INTERVAL_FRAMES)
            && let Some(cached) = *self.last_vram_budget.read()
        {
            return cached;
        }
        let snapshot = self.ctx.read().query_vram_budget();
        *self.last_vram_budget.write() = Some(snapshot);
        let policy = VramPressurePolicy::for_snapshot(snapshot);

        let should_trim = if policy.trim_reservoirs {
            let last_trim = self.last_vram_trim_frame.load(Ordering::Acquire);
            let elapsed = frame_count.saturating_sub(last_trim);
            if last_trim == 0 || elapsed >= Self::VRAM_TRIM_COOLDOWN_FRAMES {
                self.last_vram_trim_frame
                    .compare_exchange(
                        last_trim,
                        frame_count.max(1),
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
            } else {
                false
            }
        } else {
            false
        };

        {
            let mut blas = self.blas_manager.lock();
            blas.apply_vram_pressure(policy.throttle_blas_compaction, policy.lod_cutoff_scale);
            if should_trim {
                blas.sharc.request_clear();
            }
        }

        if should_trim {
            let ctx = self.ctx.read();
            self.render_graph
                .lock()
                .trim_under_memory_pressure(&ctx, &self.retire);
        }

        snapshot
    }

    /// Evaluates the active platform / MoltenVK render tier ceiling for this Vulkan device.
    pub fn platform_tier_policy(&self) -> PlatformTierPolicy {
        self.ctx.read().platform_tier_policy()
    }

    /// Propagates a temporal history reset across the render graph, SHaRC spatial cache, and
    /// active vendor SDKs (NRD, DLSS-SR/RR/FG, FSR3, XeSS).
    pub fn reset_temporal_history(&self) {
        self.render_graph.lock().reset_temporal_history();
        self.blas_manager.lock().sharc.request_clear();
        self.sdk_manager.lock().reset_temporal_history();
    }
}

/// The global engine slot. The outer `Mutex` is only ever held for a single `Arc::clone`; all
/// substantive work happens through the subsystem-specific locks above.
static ENGINE: Mutex<Option<Arc<IgnisEngine>>> = Mutex::new(None);

/// Returns a cloned handle to the live engine, or `None` before `vs_core_init` / after `vs_core_shutdown`.
pub fn engine() -> Option<Arc<IgnisEngine>> {
    ENGINE.lock().clone()
}

/// Installs a freshly initialised engine, replacing (and dropping) any previous one.
pub fn install_engine(engine: IgnisEngine) {
    *ENGINE.lock() = Some(Arc::new(engine));
}

/// Removes the live engine from the global slot (used by `vs_core_shutdown`).
pub fn take_engine() -> Option<Arc<IgnisEngine>> {
    ENGINE.lock().take()
}
