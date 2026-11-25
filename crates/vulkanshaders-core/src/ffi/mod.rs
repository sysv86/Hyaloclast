//! Java 25 Foreign Function & Memory (`java.lang.foreign.Linker`) C ABI entrypoints.
//!
//! All functions use `#[unsafe(no_mangle)] pub unsafe extern "C" fn` as required by Rust 2024 Edition
//! and match the exact `FunctionDescriptor` signatures in `dev.vulkanshaders.backend.NativeCoreBridge`.
//!
//! Each entrypoint locks only the subsystem it touches. The global engine slot is cloned as an
//! `Arc` and released immediately, so background texture uploads / shader compilation never block
//! the render thread's graph execution, frame pacing, or vendor-SDK dispatch.

use crate::{
    IgnisEngine, PipelinesAndCompiler, engine,
    framegen::{FrameGenBackendMode, FrameGenController},
    gpu::{
        GpuTimestampProfiler, PlatformTierPolicy, VramPressureLevel, VulkanContext,
        bindless::{BindlessTextureTable, PACK_RESOURCE_SLOTS, PackTextureTable},
        device_merger, evaluate_platform_tier_policy, lock_queue,
        retire::RetireQueue,
        unlock_queue,
    },
    graph::{DhFrameUniforms, GraphConfig, RenderGraph},
    install_engine,
    modules::{
        AsyncPipelinePrewarmer, DepthReprojectPushConstants, GpuPipelineManager,
        PersistentPipelineDiskCache, PipelineCacheKeyInput, WorldUboData,
    },
    rt::{
        ZeroCopyBlasManager,
        entities::{EntityCategory, EntityGeometryKind},
        omm::{
            CutoutOmmBuildInput, MAX_OMM_SUBDIV_LEVEL, OmmFormat, OmmTriangleEncoding,
            encoded_byte_size,
        },
    },
    sdk::{
        DenoiserKind, FrameGenKind, RuntimeSdkManager, SdkDispatchParams, UpscalerKind,
        ViewportExtent, ngx::NvSdkNgxPerfQuality,
    },
    take_engine,
};
use ash::vk::{self, Handle};
use parking_lot::Mutex;
use std::{env, fs, path::{Path, PathBuf}, slice, str};
use vsp_compiler::{
    CompiledShaderArtifact, RenderTier, ShaderStageKind, SpirvHookLinker, VspCompiler,
};

/// `vs_core_upload_pack_texture` result: the RGBA data was accepted and queued, and will be
/// uploaded into Set 3 as soon as `vs_core_init` creates the engine. This is a real deferred
/// upload (the bytes are owned by the queue), NOT a simulated success.
pub const VS_CORE_PACK_UPLOAD_DEFERRED: i32 = 2;

struct PendingPackUpload {
    slot: u32,
    width: u32,
    height: u32,
    rgba: Vec<u8>,
    filter_linear: bool,
    wrap_repeat: bool,
}

/// Pack resources are loaded by Kotlin before `VulkanDevice`/`vs_core_init` exists. Queue them
/// here and replay them once the engine (and its Set 3 descriptor table) is live.
static PENDING_PACK_UPLOADS: Mutex<Vec<PendingPackUpload>> = Mutex::new(Vec::new());

struct RuntimeFrameState {
    game_dir: Option<std::path::PathBuf>,
    render_scale: f32,
    prev_view: [[f32; 4]; 4],
    prev_jitter: [f32; 2],
    sdk_frame_index: u32,
    history_reset_pending: bool,
    fg_history_reset_pending: bool,
    last_frame_instant: Option<std::time::Instant>,
    last_frame_dt: f32,
    swapchain_format: u32,
    swapchain_colorspace: u32,
    hdr_peak_nits: f32,
    hdr_paper_white_nits: f32,
    world_graph_executed_this_frame: bool,
    spec_overrides_by_id: std::collections::HashMap<u32, u32>,
}

impl Default for RuntimeFrameState {
    fn default() -> Self {
        Self {
            game_dir: std::env::var_os("VS_GAME_DIR").map(std::path::PathBuf::from),
            render_scale: 1.0,
            prev_view: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            prev_jitter: [0.0, 0.0],
            sdk_frame_index: 0,
            history_reset_pending: true,
            fg_history_reset_pending: true,
            last_frame_instant: None,
            last_frame_dt: 1.0 / 60.0,
            swapchain_format: 0,
            swapchain_colorspace: 0,
            hdr_peak_nits: 1000.0,
            hdr_paper_white_nits: 200.0,
            world_graph_executed_this_frame: false,
            spec_overrides_by_id: std::collections::HashMap::new(),
        }
    }
}

static RUNTIME_FRAME_STATE: std::sync::LazyLock<Mutex<RuntimeFrameState>> =
    std::sync::LazyLock::new(|| Mutex::new(RuntimeFrameState::default()));

fn resolve_game_path(rel: &str) -> std::path::PathBuf {
    let guard = RUNTIME_FRAME_STATE.lock();
    if let Some(ref root) = guard.game_dir {
        root.join(rel)
    } else {
        std::path::PathBuf::from(rel)
    }
}

fn mul_mat4_col_major(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            out[col][row] = a[0][row] * b[col][0]
                + a[1][row] * b[col][1]
                + a[2][row] * b[col][2]
                + a[3][row] * b[col][3];
        }
    }
    out
}

fn invert_rigid_or_affine_4x4(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    // WorldUboData stores column-major inv_view (view_to_world); orthonormal basis transpose + negated translation
    // recovers exact world_to_view without drift.
    let r00 = m[0][0];
    let r01 = m[0][1];
    let r02 = m[0][2];
    let r10 = m[1][0];
    let r11 = m[1][1];
    let r12 = m[1][2];
    let r20 = m[2][0];
    let r21 = m[2][1];
    let r22 = m[2][2];
    let tx = m[3][0];
    let ty = m[3][1];
    let tz = m[3][2];
    let itx = -(r00 * tx + r01 * ty + r02 * tz);
    let ity = -(r10 * tx + r11 * ty + r12 * tz);
    let itz = -(r20 * tx + r21 * ty + r22 * tz);
    [
        [r00, r10, r20, 0.0],
        [r01, r11, r21, 0.0],
        [r02, r12, r22, 0.0],
        [itx, ity, itz, 1.0],
    ]
}

/// Replays every deferred pack-texture upload against the freshly initialised engine.
fn replay_pending_pack_uploads() {
    let pending: Vec<PendingPackUpload> = PENDING_PACK_UPLOADS.lock().drain(..).collect();
    if pending.is_empty() {
        return;
    }
    let Some(engine) = engine() else {
        PENDING_PACK_UPLOADS.lock().extend(pending);
        return;
    };
    let ctx = engine.ctx.read();
    let mut pack = engine.pack_textures.write();
    for upload in pending {
        let rc = unsafe {
            pack.upload_rgba_deferred(
                &ctx,
                Some(&engine.retire),
                upload.slot,
                upload.width,
                upload.height,
                &upload.rgba,
                upload.filter_linear,
                upload.wrap_repeat,
            )
        };
        eprintln!(
            "[vulkanshaders-core] Replayed deferred pack texture slot={} {}x{} -> rc={}",
            upload.slot, upload.width, upload.height, rc
        );
    }
}

/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_set_instance_proc_addr(pfn_raw: u64) {
    device_merger::set_instance_proc_addr(pfn_raw);
}

/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_init(
    vk_instance: u64,
    vk_physical_device: u64,
    vk_device: u64,
    graphics_queue: u64,
    compute_queue: u64,
) -> i32 {
    crate::crash::init_early();
    let ctx = match unsafe {
        VulkanContext::from_raw_handles(
            vk_instance,
            vk_physical_device,
            vk_device,
            graphics_queue,
            compute_queue,
        )
    } {
        Ok(c) => c,
        Err(err) => {
            let msg = format!("VulkanContext::from_raw_handles failed: {err:?}");
            eprintln!("[vulkanshaders-core] ERROR {msg}");
            crate::crash::push_native_error("ERROR", "ffi::vs_core_init", &msg);
            return -1;
        }
    };

    let mut bindless = BindlessTextureTable::new(4096);
    let _ = bindless.init_gpu_resources(&ctx);
    // Defer pack_textures GPU init & pending upload replay until vs_core_set_queue_families
    // provides the real graphics_queue_family_index (never hardcode family 0).
    let pack_textures = PackTextureTable::new();

    let mut framegen = FrameGenController::default();
    framegen.init_reflex(&ctx);

    // Probe real vendor SDK runtimes staged next to the game (NGX / NRD / FSR3 / XeSS) and
    // initialize NGX when running on NVIDIA hardware with NGX Core available.
    let natives_dir_buf = resolve_game_path("natives");
    let natives_dir = natives_dir_buf.as_path();
    let mut sdk_manager = RuntimeSdkManager::probe(vec![natives_dir.to_path_buf()]);
    sdk_manager.set_gpu_arch(ctx.capabilities.gpu_arch);
    // Initialize NGX on a dedicated 8 MiB stack thread: JVM FFM downcall threads default to a
    // small stack (~1 MiB) which can overflow inside libnvidia-ngx.so.1 during snippet discovery.
    let ngx_disabled = matches!(
        std::env::var("VS_CORE_ENABLE_NGX").as_deref(),
        Ok("0") | Ok("false")
    );
    if !ngx_disabled
        && ctx.capabilities.vendor_id == 0x10DE
        && sdk_manager.capabilities.ngx_core_loaded
    {
        let ngx_data_dir_buf = resolve_game_path("config/vulkanshaders/ngx");
        let ngx_data_dir = ngx_data_dir_buf.as_path();
        let _ = std::fs::create_dir_all(ngx_data_dir);
        let inst_raw = ctx.instance.handle().as_raw();
        let pd_raw = ctx.physical_device.as_raw();
        let dev_raw = ctx.device.handle().as_raw();
        let gipa_raw = ctx.entry.static_fn().get_instance_proc_addr as usize;
        let gdpa_raw = ctx.instance.fp_v1_0().get_device_proc_addr as usize;
        let data_dir_str = ngx_data_dir.to_str().unwrap_or(".").to_owned();
        match std::thread::Builder::new()
            .name("vs-ngx-init".to_owned())
            .stack_size(8 * 1024 * 1024)
            .spawn(move || {
                let gipa: vk::PFN_vkGetInstanceProcAddr = unsafe { std::mem::transmute(gipa_raw) };
                let gdpa: vk::PFN_vkGetDeviceProcAddr = unsafe { std::mem::transmute(gdpa_raw) };
                let _ = unsafe {
                    sdk_manager.initialize_ngx(
                        vk::Instance::from_raw(inst_raw),
                        vk::PhysicalDevice::from_raw(pd_raw),
                        vk::Device::from_raw(dev_raw),
                        gipa,
                        gdpa,
                        &data_dir_str,
                    )
                };
                sdk_manager
            }) {
            Ok(handle) => match handle.join() {
                Ok(returned_mgr) => sdk_manager = returned_mgr,
                Err(_) => {
                    sdk_manager = RuntimeSdkManager::probe(vec![natives_dir.to_path_buf()]);
                    sdk_manager.set_gpu_arch(ctx.capabilities.gpu_arch);
                }
            },
            // `spawn` consumes the manager even on failure; re-probe so it remains usable.
            Err(_) => {
                sdk_manager = RuntimeSdkManager::probe(vec![natives_dir.to_path_buf()]);
                sdk_manager.set_gpu_arch(ctx.capabilities.gpu_arch);
            }
        }
    }

    let mut pipelines = GpuPipelineManager::new();
    {
        let rfs = RUNTIME_FRAME_STATE.lock();
        pipelines.hdr_push_constants.peak_brightness_nits = rfs.hdr_peak_nits;
        pipelines.hdr_push_constants.paper_white_nits = rfs.hdr_paper_white_nits;
    }

    let timestamp_profiler = match ctx.create_timestamp_profiler(
        GpuTimestampProfiler::DEFAULT_FRAMES_IN_FLIGHT,
        GpuTimestampProfiler::DEFAULT_MAX_PASSES,
    ) {
        Ok(profiler) => Some(profiler),
        Err(err) => {
            crate::crash::global_device_lost_guard()
                .record_vk_result("ffi::vs_core_init::timestamp_profiler", err);
            None
        }
    };
    let pipeline_disk_cache = Some(PersistentPipelineDiskCache::new(resolve_game_path(
        "config/vulkanshaders/cache",
    )));
    let initial_vram_budget = Some(ctx.query_vram_budget());

    let engine = IgnisEngine {
        ctx: parking_lot::RwLock::new(ctx),
        retire: RetireQueue::new(3),
        bindless: parking_lot::RwLock::new(bindless),
        pack_textures: parking_lot::RwLock::new(pack_textures),
        blas_manager: parking_lot::Mutex::new(ZeroCopyBlasManager::new()),
        pipelines_and_compiler: parking_lot::Mutex::new(PipelinesAndCompiler {
            pipelines,
            compiler: VspCompiler::new(Some(&resolve_game_path(
                "config/vulkanshaders/cache/spirv",
            ))),
        }),
        framegen: parking_lot::Mutex::new(framegen),
        sdk_manager: parking_lot::Mutex::new(sdk_manager),
        render_graph: parking_lot::Mutex::new(RenderGraph::default()),
        world_ubo: parking_lot::RwLock::new(WorldUboData::default()),
        frame_index: std::sync::atomic::AtomicU64::new(0),
        last_built_tlas_handle: std::sync::atomic::AtomicU64::new(0),
        framegen_mode: std::sync::atomic::AtomicU32::new(0),
        timestamp_profiler: parking_lot::Mutex::new(timestamp_profiler),
        pipeline_disk_cache,
        active_prewarmer: parking_lot::Mutex::new(None),
        last_vram_budget: parking_lot::RwLock::new(initial_vram_budget),
        vram_poll_frame_counter: std::sync::atomic::AtomicU64::new(0),
        last_vram_trim_frame: std::sync::atomic::AtomicU64::new(0),
    };
    install_engine(engine);
    0
}

/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_set_queue_families(
    graphics_family_index: i32,
    compute_family_index: i32,
) {
    let Some(engine) = engine() else {
        return;
    };
    engine.ctx.write().set_queue_family_indices(
        graphics_family_index.max(0) as u32,
        compute_family_index.max(0) as u32,
    );
    {
        let ctx = engine.ctx.read();
        let _ = engine.pack_textures.write().init_gpu_resources(&ctx);
    }
    replay_pending_pack_uploads();
}

/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_shutdown() {
    let Some(engine) = take_engine() else {
        return;
    };
    let prev_prewarmer = engine.active_prewarmer.lock().take();
    drop(prev_prewarmer);
    let ctx = engine.ctx.read();
    engine.framegen.lock().destroy(&ctx);
    let wait_res = crate::gpu::with_queue_lock(|| unsafe { ctx.device.device_wait_idle() });
    if let Err(err) = wait_res {
        crate::crash::global_device_lost_guard()
            .record_vk_result("ffi::vs_core_shutdown::device_wait_idle", err);
    }
    if let Some(ref mut profiler) = *engine.timestamp_profiler.lock() {
        unsafe { profiler.destroy(&ctx.device) };
    }
    engine
        .pipelines_and_compiler
        .lock()
        .pipelines
        .destroy(&ctx, &engine.retire);
    engine.blas_manager.lock().shutdown(&ctx, &engine.retire);
    engine.render_graph.lock().destroy(&ctx, &engine.retire);
    engine.bindless.write().destroy(&ctx);
    engine.pack_textures.write().destroy(&ctx);
    engine.retire.drain_all();
    unsafe { engine.sdk_manager.lock().teardown() };
}

/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_create_merged_instance(
    create_info_address: u64,
    allocator_address: u64,
    out_instance_address: u64,
) -> i32 {
    unsafe {
        device_merger::create_merged_instance(
            create_info_address as *const vk::InstanceCreateInfo<'_>,
            allocator_address as *const vk::AllocationCallbacks<'_>,
            out_instance_address as *mut vk::Instance,
        )
    }
}

/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_create_merged_device(
    physical_device_address: u64,
    create_info_address: u64,
    allocator_address: u64,
    out_device_address: u64,
) -> i32 {
    unsafe {
        device_merger::create_merged_device(
            physical_device_address,
            create_info_address as *const vk::DeviceCreateInfo<'_>,
            allocator_address as *const vk::AllocationCallbacks<'_>,
            out_device_address as *mut vk::Device,
        )
    }
}

/// Applies [`PlatformTierPolicy`] (such as macOS / MoltenVK Tier 0/1 clamping) to a capability bitmask.
fn apply_platform_tier_policy_to_caps_mask(mut mask: u32, policy: &PlatformTierPolicy) -> u32 {
    if policy.hardware_rt_blocked || policy.max_render_tier < RenderTier::Tier2RayQuery {
        mask &= !((1 << 0) | (1 << 1) | (1 << 2) | (1 << 13) | (1 << 14) | (1 << 15));
    } else if policy.max_render_tier < RenderTier::Tier3PathTracing {
        mask &= !(1 << 2);
    }
    mask
}

/// Unified capability bitmask matching `vs_core_query_capabilities`:
/// bit0 RT AS, bit1 RayQuery, bit2 RT Pipeline, bit3 Bindless, bit4 Reflex,
/// bit5 DLSS-SR, bit6 DLSS-RR, bit7 DLSS-FG, bit8 FSR3, bit9 XeSS, bit10 NRD, bit11 HDR swapchain,
/// bit12 buffer device address, bit13 SER NV, bit14 SER EXT, bit15 Opacity Micromap,
/// bit16 High-End GPU, bit17 Ada Lovelace or newer.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_get_capabilities_mask(vk_device: u64) -> u32 {
    let caps = if vk_device != 0 {
        device_merger::capabilities_for_device(vk_device)
            .or_else(device_merger::last_merged_capabilities)
            .unwrap_or_default()
    } else {
        device_merger::last_merged_capabilities().unwrap_or_default()
    };
    let mut mask = 0u32;
    if caps.acceleration_structure {
        mask |= 1 << 0;
    }
    if caps.ray_query {
        mask |= 1 << 1;
    }
    if caps.ray_tracing_pipeline {
        mask |= 1 << 2;
    }
    if caps.descriptor_indexing {
        mask |= 1 << 3;
    }
    if caps.low_latency2_nv {
        mask |= 1 << 4;
    }
    if caps.hdr_swapchain_colorspace {
        mask |= 1 << 11;
    }
    if caps.buffer_device_address {
        mask |= 1 << 12;
    }
    if caps.supports_ser_nv {
        mask |= 1 << 13;
    }
    if caps.supports_ser_ext {
        mask |= 1 << 14;
    }
    if caps.supports_opacity_micromap {
        mask |= 1 << 15;
    }
    if caps.gpu_tier_class.is_high_end() {
        mask |= 1 << 16;
    }
    if caps.gpu_arch.is_ada_or_newer() {
        mask |= 1 << 17;
    }
    if let Some(engine) = engine() {
        let sdk = engine.sdk_manager.lock();
        let c = &sdk.capabilities;
        if c.dlss_sr_available {
            mask |= 1 << 5;
        }
        if c.dlss_rr_available {
            mask |= 1 << 6;
        }
        if c.dlss_fg_available {
            mask |= 1 << 7;
        }
        if c.fsr3_loaded {
            mask |= 1 << 8;
        }
        if c.xess_loaded {
            mask |= 1 << 9;
        }
        if c.nrd_loaded {
            mask |= 1 << 10;
        }
        drop(sdk);
        let policy = engine.platform_tier_policy();
        return apply_platform_tier_policy_to_caps_mask(mask, &policy);
    }
    let policy = evaluate_platform_tier_policy(&caps, caps.is_moltenvk, cfg!(target_os = "macos"));
    apply_platform_tier_policy_to_caps_mask(mask, &policy)
}

/// Stable capability bitmask consumed by `dev.vulkanshaders.backend.NativeCoreBridge`.
/// bit0 RT AS, bit1 RayQuery, bit2 RT Pipeline, bit3 Bindless, bit4 Reflex,
/// bit5 DLSS-SR, bit6 DLSS-RR, bit7 DLSS-FG, bit8 FSR3, bit9 XeSS, bit10 NRD, bit11 HDR swapchain,
/// bit12 buffer device address, bit13 SER NV, bit14 SER EXT, bit15 Opacity Micromap,
/// bit16 High-End GPU, bit17 Ada Lovelace or newer.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_query_capabilities() -> u32 {
    let caps = device_merger::last_merged_capabilities().unwrap_or_default();
    let mut mask = 0u32;
    if caps.acceleration_structure {
        mask |= 1 << 0;
    }
    if caps.ray_query {
        mask |= 1 << 1;
    }
    if caps.ray_tracing_pipeline {
        mask |= 1 << 2;
    }
    if caps.descriptor_indexing {
        mask |= 1 << 3;
    }
    if caps.low_latency2_nv {
        mask |= 1 << 4;
    }
    if caps.hdr_swapchain_colorspace {
        mask |= 1 << 11;
    }
    if caps.buffer_device_address {
        mask |= 1 << 12;
    }
    if caps.supports_ser_nv {
        mask |= 1 << 13;
    }
    if caps.supports_ser_ext {
        mask |= 1 << 14;
    }
    if caps.supports_opacity_micromap {
        mask |= 1 << 15;
    }
    if caps.gpu_tier_class.is_high_end() {
        mask |= 1 << 16;
    }
    if caps.gpu_arch.is_ada_or_newer() {
        mask |= 1 << 17;
    }

    if let Some(engine) = engine() {
        let sdk = engine.sdk_manager.lock();
        let c = &sdk.capabilities;
        if c.dlss_sr_available {
            mask |= 1 << 5;
        }
        if c.dlss_rr_available {
            mask |= 1 << 6;
        }
        if c.dlss_fg_available {
            mask |= 1 << 7;
        }
        if c.fsr3_loaded {
            mask |= 1 << 8;
        }
        if c.xess_loaded {
            mask |= 1 << 9;
        }
        if c.nrd_loaded {
            mask |= 1 << 10;
        }
        drop(sdk);
        let policy = engine.platform_tier_policy();
        return apply_platform_tier_policy_to_caps_mask(mask, &policy);
    }
    let policy = evaluate_platform_tier_policy(&caps, caps.is_moltenvk, cfg!(target_os = "macos"));
    apply_platform_tier_policy_to_caps_mask(mask, &policy)
}

/// Returns `1` if `surface` really advertises an HDR format (`R16G16B16A16_SFLOAT` /
/// `A2B10G10R10`) together with `EXTENDED_SRGB_LINEAR_EXT` / `HDR10_ST2084_EXT`, probed via the
/// real `vkGetPhysicalDeviceSurfaceFormatsKHR`. Returns `0` for an unknown surface or SDR-only GPU.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_is_hdr_swapchain_supported(surface_raw: u64) -> i32 {
    i32::from(device_merger::is_hdr_swapchain_supported(surface_raw))
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn json_string_array(items: &[String]) -> String {
    let inner = items
        .iter()
        .map(|s| format!("\"{}\"", json_escape(s)))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{inner}]")
}

/// Writes a UTF-8 JSON capability report into `out_ptr` (at most `max_len` bytes) and returns the
/// number of bytes that *would* be written. Every field is derived from a real Vulkan query or a
/// real vendor-library probe; unavailable features carry a concrete reason for the UI tooltip.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_get_capability_details(out_ptr: u64, max_len: i32) -> i32 {
    let caps = device_merger::last_merged_capabilities().unwrap_or_default();
    let probe = device_merger::probe_info();

    let engine_handle = engine();
    let sdk = engine_handle.as_ref().map(|e| e.sdk_manager.lock());
    let sdk_caps = sdk.as_ref().map(|s| &s.capabilities);
    let fsr3_path = sdk.as_ref().and_then(|s| s.library_path("fsr3"));
    let xess_path = sdk.as_ref().and_then(|s| s.library_path("xess"));
    let nrd_path = sdk.as_ref().and_then(|s| s.library_path("nrd"));
    let ngx_path = sdk.as_ref().and_then(|s| s.library_path("dlssSr"));
    let init_result = sdk.as_ref().and_then(|s| s.last_init_result.clone());
    let eval_result = sdk.as_ref().and_then(|s| s.last_evaluate_result.clone());

    let feature = |name: &str, available: bool, reason: &str, lib_path: Option<&Path>| -> String {
        let symbols = lib_path.is_some();
        format!(
            "\"{name}\":{{\"available\":{},\"reason\":\"{}\",\"libPath\":{},\"symbolsResolved\":{},\"initResult\":{},\"lastEvaluateResult\":{}}}",
            available,
            json_escape(reason),
            lib_path
                .map(|p| format!("\"{}\"", json_escape(&p.to_string_lossy())))
                .unwrap_or_else(|| "null".to_string()),
            symbols,
            init_result
                .as_ref()
                .map(|s| format!("\"{}\"", json_escape(s)))
                .unwrap_or_else(|| "null".to_string()),
            eval_result
                .as_ref()
                .map(|s| format!("\"{}\"", json_escape(s)))
                .unwrap_or_else(|| "null".to_string()),
        )
    };

    let reason_vk = |ok: bool, ext: &str| {
        if ok {
            "supported".to_string()
        } else {
            format!("{ext} unsupported or feature not enabled by the GPU/driver")
        }
    };
    let vendor_reason = |key: &str| -> String {
        sdk.as_ref()
            .map(|s| s.unavailable_reason(key))
            .unwrap_or_else(|| "SDK manager not initialized (vs_core_init not called)".to_string())
    };
    let fsr3_avail = sdk_caps.map(|c| c.fsr3_loaded).unwrap_or(false);
    let xess_avail = sdk_caps.map(|c| c.xess_loaded).unwrap_or(false);
    let nrd_avail = sdk_caps.map(|c| c.nrd_loaded).unwrap_or(false);
    let dlss_sr = sdk_caps.map(|c| c.dlss_sr_available).unwrap_or(false);
    let dlss_rr = sdk_caps.map(|c| c.dlss_rr_available).unwrap_or(false);
    let dlss_fg = sdk_caps.map(|c| c.dlss_fg_available).unwrap_or(false);

    let tier_policy = engine_handle
        .as_ref()
        .map(|e| e.platform_tier_policy())
        .unwrap_or_else(|| {
            evaluate_platform_tier_policy(&caps, caps.is_moltenvk, cfg!(target_os = "macos"))
        });
    let rt_as_avail = caps.acceleration_structure
        && !tier_policy.hardware_rt_blocked
        && tier_policy.max_render_tier >= RenderTier::Tier2RayQuery;
    let ray_query_avail = caps.ray_query
        && !tier_policy.hardware_rt_blocked
        && tier_policy.max_render_tier >= RenderTier::Tier2RayQuery;
    let rt_pipe_avail = caps.ray_tracing_pipeline
        && !tier_policy.hardware_rt_blocked
        && tier_policy.max_render_tier >= RenderTier::Tier3PathTracing;
    let rt_reason = |avail: bool, raw_cap: bool, ext: &str| {
        if avail {
            "supported".to_string()
        } else if raw_cap && tier_policy.hardware_rt_blocked {
            tier_policy.reason.clone()
        } else {
            reason_vk(false, ext)
        }
    };

    let features = [
        feature(
            "rtAs",
            rt_as_avail,
            &rt_reason(
                rt_as_avail,
                caps.acceleration_structure,
                "VK_KHR_acceleration_structure",
            ),
            None,
        ),
        feature(
            "rayQuery",
            ray_query_avail,
            &rt_reason(ray_query_avail, caps.ray_query, "VK_KHR_ray_query"),
            None,
        ),
        feature(
            "rtPipeline",
            rt_pipe_avail,
            &rt_reason(
                rt_pipe_avail,
                caps.ray_tracing_pipeline,
                "VK_KHR_ray_tracing_pipeline",
            ),
            None,
        ),
        feature(
            "bindless",
            caps.descriptor_indexing,
            &reason_vk(caps.descriptor_indexing, "VK_EXT_descriptor_indexing"),
            None,
        ),
        feature(
            "reflex",
            caps.low_latency2_nv,
            &reason_vk(caps.low_latency2_nv, "VK_NV_low_latency2"),
            None,
        ),
        feature("dlssSr", dlss_sr, &vendor_reason("dlssSr"), ngx_path),
        feature("dlssRr", dlss_rr, &vendor_reason("dlssRr"), ngx_path),
        feature("dlssFg", dlss_fg, &vendor_reason("dlssFg"), ngx_path),
        feature("fsr3", fsr3_avail, &vendor_reason("fsr3"), fsr3_path),
        feature("xess", xess_avail, &vendor_reason("xess"), xess_path),
        feature("nrd", nrd_avail, &vendor_reason("nrd"), nrd_path),
        feature(
            "hdr",
            probe.hdr_surface_supported,
            if probe.hdr_surface_supported {
                "HDR format/colorspace confirmed by vkGetPhysicalDeviceSurfaceFormatsKHR"
            } else {
                "No HDR surface probed"
            },
            None,
        ),
        feature(
            "serNv",
            caps.supports_ser_nv,
            &reason_vk(caps.supports_ser_nv, "VK_NV_ray_tracing_invocation_reorder"),
            None,
        ),
        feature(
            "serExt",
            caps.supports_ser_ext,
            &reason_vk(
                caps.supports_ser_ext,
                "VK_EXT_ray_tracing_invocation_reorder",
            ),
            None,
        ),
        feature(
            "opacityMicromap",
            caps.supports_opacity_micromap,
            &reason_vk(caps.supports_opacity_micromap, "VK_EXT_opacity_micromap"),
            None,
        ),
    ]
    .join(",");

    let vram_mb = caps.vram_bytes / (1024 * 1024);
    let raw_dt = RUNTIME_FRAME_STATE.lock().last_frame_dt;
    let last_frame_dt_ms = if raw_dt.is_finite() {
        raw_dt * 1000.0
    } else {
        0.0
    };
    let json = format!(
        "{{\"gpu\":{{\"name\":\"{}\",\"apiVersion\":\"{}\",\"driver\":\"{}\",\"driverVersion\":\"{}\",\"vendorId\":{},\"deviceId\":{},\"deviceType\":\"{}\",\"vramBytes\":{},\"vramMb\":{},\"gpuArch\":\"{}\",\"gpuTierClass\":\"{}\",\"serNv\":{},\"serExt\":{},\"serActive\":{},\"opacityMicromap\":{}}},\"lastFrameDtMs\":{:.3},\"extensions\":{{\"instance\":{},\"device\":{}}},\"hdrSurfaceSupported\":{},\"features\":{{{}}}}}",
        json_escape(&probe.gpu_name),
        json_escape(&probe.api_version),
        json_escape(&probe.driver_name),
        json_escape(&probe.driver_version),
        caps.vendor_id,
        caps.device_id,
        caps.device_type_str(),
        caps.vram_bytes,
        vram_mb,
        caps.gpu_arch.as_str(),
        caps.gpu_tier_class.as_str(),
        caps.supports_ser_nv,
        caps.supports_ser_ext,
        caps.ser_reordering_active,
        caps.supports_opacity_micromap,
        last_frame_dt_ms,
        json_string_array(&probe.instance_extensions),
        json_string_array(&probe.device_extensions),
        probe.hdr_surface_supported,
        features,
    );
    let bytes = json.as_bytes();
    if out_ptr != 0 && max_len > 0 {
        let n = bytes.len().min(max_len as usize);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_ptr as *mut u8, n);
        }
    }
    bytes.len() as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn vs_core_lock_queue() {
    lock_queue();
}

#[unsafe(no_mangle)]
pub extern "C" fn vs_core_unlock_queue() {
    unlock_queue();
}

/// Updates the current frame's Halton sub-pixel jitter offset (in pixels) for TSR and vendor
/// upscalers (`DLSS-SR`, `FSR3`, `XeSS`, `NRD`).
/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_set_jitter(jitter_x: f32, jitter_y: f32) {
    let Some(engine) = engine() else {
        return;
    };
    engine
        .pipelines_and_compiler
        .lock()
        .pipelines
        .tsr_push_constants
        .jitter_offset = [jitter_x, jitter_y];
}

/// Updates the active camera's world-space position so `dispatch_csm_shadow` computes exact
/// camera-relative `RegionOrigin` offsets (`(section_origin & !63) - floor(camera_pos)`).
/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_set_camera_position(cam_x: f64, cam_y: f64, cam_z: f64) {
    let Some(engine) = engine() else {
        return;
    };
    let mut pnc = engine.pipelines_and_compiler.lock();
    pnc.pipelines.camera_pos_world = [cam_x, cam_y, cam_z];
    pnc.pipelines.camera_block_pos = [
        cam_x.floor() as i32,
        cam_y.floor() as i32,
        cam_z.floor() as i32,
    ];
}

/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_update_world_ubo(ubo_address: u64, byte_size: i32) {
    if ubo_address == 0 || byte_size < std::mem::size_of::<WorldUboData>() as i32 {
        return;
    }
    let Some(engine) = engine() else {
        return;
    };
    let src_bytes = unsafe {
        slice::from_raw_parts(
            ubo_address as *const u8,
            std::mem::size_of::<WorldUboData>(),
        )
    };
    let parsed = bytemuck::pod_read_unaligned::<WorldUboData>(src_bytes);
    *engine.world_ubo.write() = parsed;
    let slot = engine.next_frame_slot();
    let _ = engine.poll_vram_budget_if_due(false);
    {
        let ctx = engine.ctx.read();
        engine.retire.open_slot_with_device(slot, Some(&ctx.device));
        engine.blas_manager.lock().entity_blas.begin_frame();
        let mut pnc = engine.pipelines_and_compiler.lock();
        pnc.pipelines.flush_deferred_to_retire(&ctx, &engine.retire);
        if let Err(err) = pnc
            .pipelines
            .upload_world_ubo(&ctx, slot, &engine.world_ubo.read())
        {
            let msg = format!("upload_world_ubo(slot={slot}) failed: {err:?}");
            eprintln!("[vulkanshaders-core] ERROR {msg}");
            crate::crash::push_native_error("ERROR", "ffi::world_ubo", &msg);
            crate::crash::global_device_lost_guard().record_vk_result("ffi::upload_world_ubo", err);
        }
    }
}

/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_reset_temporal_history() {
    let Some(engine) = engine() else {
        return;
    };
    engine.reset_temporal_history();
    {
        let mut rfs = RUNTIME_FRAME_STATE.lock();
        rfs.history_reset_pending = true;
        rfs.fg_history_reset_pending = true;
    }
}

/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_submit_entity_vertices(
    vertex_ptr: u64,
    vertex_count: i32,
    vertex_stride: i32,
    albedo_slot: i32,
    geometry_kind: i32,
) -> i32 {
    if vertex_ptr == 0 || vertex_count < 4 || vertex_stride < 28 || albedo_slot < 0 {
        return 0;
    }
    let Some(engine) = engine() else {
        return 0;
    };
    let Some(total_bytes) = (vertex_count as usize).checked_mul(vertex_stride as usize) else {
        return 0;
    };
    let bytes = unsafe { slice::from_raw_parts(vertex_ptr as *const u8, total_bytes) };
    let kind = match geometry_kind {
        0 => EntityGeometryKind::World,
        1 => EntityGeometryKind::Crumbling,
        2 => EntityGeometryKind::FirstPersonAvatar,
        _ => return 0,
    };
    let (normal_slot, specular_slot) = engine
        .bindless
        .read()
        .companions_for_albedo(albedo_slot as u32);
    engine
        .blas_manager
        .lock()
        .entity_blas
        .append_geometry_batch(
            bytes,
            vertex_count as u32,
            vertex_stride as u32,
            true,
            albedo_slot as u32,
            normal_slot,
            specular_slot,
            kind,
        );
    (vertex_count / 4) * 2
}

/// Links a registered albedo bindless texture slot to its LabPBR normal (`_n`) and specular (`_s`)
/// companion atlas slots so dynamic entities and items sample PBR maps in Tier 2/3 RT.
/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_link_pbr_companions(
    albedo_slot: i32,
    normal_slot: i32,
    specular_slot: i32,
) {
    if albedo_slot < 0 {
        return;
    }
    let Some(engine) = engine() else {
        return;
    };
    engine.bindless.write().link_pbr_companions(
        albedo_slot as u32,
        normal_slot.max(0) as u32,
        specular_slot.max(0) as u32,
    );
}

/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_record_frame_rtx(
    cmd_raw: u64,
    cam_x: f64,
    cam_y: f64,
    cam_z: f64,
    prev_cam_x: f64,
    prev_cam_y: f64,
    prev_cam_z: f64,
    frame_index: i32,
    max_blas_builds: i32,
) -> i64 {
    if cmd_raw == 0 {
        return 0;
    }
    if crate::crash::global_device_lost_guard().is_rt_vsp_disabled() {
        return 0;
    }
    let Some(engine) = engine() else {
        return 0;
    };
    let cmd = vk::CommandBuffer::from_raw(cmd_raw);
    let slot = engine.frame_slot();
    let world = engine.world_ubo.read();
    // Feed the wind scheduler the same time source and storm signal as the raster hook so
    // BLAS displacement matches the G-buffer silhouette. If the producer omitted the
    // integrated phase (legacy 528-byte UBO), fall back to the legacy game clock.
    // The raster hook only sees `gameTimeSec` through its reduced VsHookWorld, so the wind
    // compute path must use the same clock to keep G-buffer and BLAS foliage silhouettes in sync.
    let wind_phase = world.game_time_sec;
    let rain = world.rain.clamp(0.0, 1.0);
    let thunder = world.thunder.clamp(0.0, 1.0);
    let storm = (rain * 0.72 + thunder * 0.55).clamp(0.0, 1.25);
    drop(world);
    let vp_size = engine.render_graph.lock().internal_target_size;
    let ctx = engine.ctx.read();
    let mut blas = engine.blas_manager.lock();
    let mut pnc = engine.pipelines_and_compiler.lock();
    pnc.pipelines.camera_pos_world = [cam_x, cam_y, cam_z];
    pnc.pipelines.camera_block_pos = [
        cam_x.floor() as i32,
        cam_y.floor() as i32,
        cam_z.floor() as i32,
    ];
    if vp_size.0 > 0 && vp_size.1 > 0 {
        blas.emissive_lights
            .set_viewport_dimensions(vp_size.0, vp_size.1);
    }
    let bindless = engine.bindless.read();
    let addr = match blas.record_frame_rtx(
        &ctx,
        &engine.retire,
        Some(&pnc.pipelines),
        Some(&bindless),
        cmd,
        slot,
        [cam_x, cam_y, cam_z],
        [prev_cam_x, prev_cam_y, prev_cam_z],
        frame_index.max(0) as u32,
        max_blas_builds.max(1) as usize,
        wind_phase,
        storm,
    ) {
        Ok(addr) => addr,
        Err(err) => {
            crate::crash::global_device_lost_guard().record_vk_result("ffi::record_frame_rtx", err);
            0
        }
    };
    engine.last_built_tlas_handle.store(
        blas.last_built_tlas_handle.as_raw(),
        std::sync::atomic::Ordering::Release,
    );
    addr as i64
}


/// Loads VSL 2.0 SPIR-V units from a directory and registers them as native pipeline artifacts.
/// Reflection is re-derived from the binary with the engine's rspirv-based linker, so the
/// descriptor layouts and capability gates stay identical to the v1 compiler path.
fn collect_vsl_units(dir: &Path) -> Vec<CompiledShaderArtifact> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut units = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("spv") {
            continue;
        }
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        if bytes.len() < 20 || bytes.len() % 4 != 0 {
            continue;
        }
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        if words[0] != 0x0723_0203 {
            continue;
        }
        let stage_name = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("vsl_unit")
            .to_owned();
        let stage_ext = fs::read_to_string(path.with_extension("stage"))
            .unwrap_or_else(|_| "comp".to_owned());
        let stage_kind = ShaderStageKind::from_extension(stage_ext.trim())
            .unwrap_or(ShaderStageKind::Compute);
        let Ok(reflection) = SpirvHookLinker::reflect_spirv(&words) else {
            continue;
        };
        units.push(CompiledShaderArtifact {
            stage_name,
            stage_kind,
            spirv_words: words,
            content_hash: blake3::hash(&bytes).into(),
            hooks: Vec::new(),
            reflection,
            loaded_from_cache: false,
        });
    }
    let names: Vec<&str> = units.iter().map(|unit| unit.stage_name.as_str()).collect();
    eprintln!(
        "[vulkanshaders-core] VSL 2.0 units from {}: {} ({names:?})",
        dir.display(),
        units.len()
    );
    units
}

/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_compile_active_pack(
    pack_path_utf8_ptr: u64,
    pack_path_utf8_len: i32,
) -> i32 {
    if pack_path_utf8_ptr == 0 || pack_path_utf8_len <= 0 {
        return -1;
    }
    let Some(engine) = engine() else {
        return -1;
    };
    let bytes = unsafe {
        slice::from_raw_parts(pack_path_utf8_ptr as *const u8, pack_path_utf8_len as usize)
    };
    let Ok(path_str) = str::from_utf8(bytes) else {
        return -1;
    };
    let requested_tier = engine.render_graph.lock().config.tier;
    let ctx = engine.ctx.read();
    let tier = ctx.platform_tier_policy().clamp_render_tier(requested_tier);
    let (bindless_layout, pack_layout) = {
        let bl = engine.bindless.read();
        let mut pl = engine.pack_textures.write();
        let _ = pl.init_gpu_resources(&ctx);
        (bl.descriptor_set_layout, pl.descriptor_set_layout)
    };
    let mut pnc = engine.pipelines_and_compiler.lock();
    let PipelinesAndCompiler {
        ref mut pipelines,
        ref mut compiler,
    } = *pnc;
    let pack_path = Path::new(path_str);
    let pack_hash = PersistentPipelineDiskCache::compute_pack_hash(pack_path);
    let (driver_uuid, pipeline_cache_uuid) = PersistentPipelineDiskCache::query_driver_uuids(&ctx);
    let cache_key = PipelineCacheKeyInput {
        pack_hash,
        driver_uuid,
        pipeline_cache_uuid,
        spec_constants: pipelines.active_spec_constants.clone(),
    }
    .compute_key();
    pipelines.ensure_pipeline_cache_for_key(&ctx, cache_key);

    let bundled_units = pack_path.join("vsl-units");
    let vsl_dir = if bundled_units.is_dir() {
        bundled_units
    } else {
        env::var_os("VSL_UNIT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("shaderpacks/vsl-units"))
    };
    pipelines.pending_vsl_artifacts = collect_vsl_units(&vsl_dir);
    let count = pipelines.compile_pack_directory_with_layouts(
        &ctx,
        compiler,
        pack_path,
        tier,
        bindless_layout,
        pack_layout,
        Some(&engine.retire),
    );
    if count > 0 {
        pipelines.save_pipeline_cache_to_disk(&ctx);
        if let Some(ref disk_cache) = engine.pipeline_disk_cache {
            let _ = disk_cache.flush_vk_pipeline_cache(&ctx, pipelines.pipeline_cache, &cache_key);
        }
    }
    drop(pnc);
    if let Ok((manifest, _)) = VspCompiler::resolve_pack_inheritance(pack_path, &[]) {
        engine
            .render_graph
            .lock()
            .register_pack_manifest_graph(&manifest);
    }
    let mut pnc = engine.pipelines_and_compiler.lock();
    if tier == RenderTier::Tier3PathTracing && pnc.pipelines.rt_pipeline.is_none() {
        let msg = pnc
            .pipelines
            .last_compile_error
            .get_or_insert_with(|| {
                "Path Tracing needs a Vulkan ray-tracing pipeline and all four RT shader stages"
                    .to_owned()
            })
            .clone();
        let diag = crate::crash::ShaderCompileDiagnostic::parse_from_compiler_error(path_str, &msg);
        crate::crash::global_device_lost_guard().record_shader_compile_failure(diag);
        return -2;
    }
    if tier == RenderTier::Tier2RayQuery
        && !pnc
            .pipelines
            .compute_pipelines
            .contains_key("rayquery_lighting.comp")
    {
        let msg = pnc
            .pipelines
            .last_compile_error
            .get_or_insert_with(|| {
                "Ray Query needs VK_KHR_ray_query and a rayquery_lighting.comp pipeline".to_owned()
            })
            .clone();
        let diag = crate::crash::ShaderCompileDiagnostic::parse_from_compiler_error(path_str, &msg);
        crate::crash::global_device_lost_guard().record_shader_compile_failure(diag);
        return -2;
    }
    if count == 0
        && let Some(ref msg) = pnc.pipelines.last_compile_error
    {
        let diag = crate::crash::ShaderCompileDiagnostic::parse_from_compiler_error(path_str, msg);
        crate::crash::global_device_lost_guard().record_shader_compile_failure(diag);
        return -2;
    }
    let guard = crate::crash::global_device_lost_guard();
    if guard.snapshot().reason == crate::crash::RecoveryReason::ShaderCompileFailure {
        guard.reset();
    }
    count as i32
}

/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
fn parse_spec_constant_pairs(pairs_ptr: u64, pair_count: i32) -> Vec<(u32, u32)> {
    let mut overrides = Vec::new();
    if pairs_ptr != 0 && pair_count > 0 {
        let raw_u32 =
            unsafe { slice::from_raw_parts(pairs_ptr as *const u32, (pair_count as usize) * 2) };
        for chunk in raw_u32.as_chunks::<2>().0 {
            overrides.push((chunk[0], chunk[1]));
        }
    }
    let mut rfs = RUNTIME_FRAME_STATE.lock();
    rfs.spec_overrides_by_id.clear();
    for &(id, bits) in &overrides {
        rfs.spec_overrides_by_id.insert(id, bits);
    }
    overrides
}

/// Stages specialization constants before `vs_core_compile_pack` so the new pack's pipelines are
/// built once with the final overrides instead of compiling twice.
///
/// # Safety
/// `pairs_ptr` must be null or point to `pair_count * 2` valid `u32` values for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_prepare_spec_constants(pairs_ptr: u64, pair_count: i32) -> i32 {
    let Some(engine) = engine() else {
        return -1;
    };
    let overrides = parse_spec_constant_pairs(pairs_ptr, pair_count);
    let mut pnc = engine.pipelines_and_compiler.lock();
    pnc.pipelines.active_spec_constants = overrides;
    0
}

/// # Safety
/// If `pair_count > 0`, `pairs_ptr` must point to `pair_count` valid `(u32, f32)` specialization constant pairs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_update_spec_constants(pairs_ptr: u64, pair_count: i32) -> i32 {
    let Some(engine) = engine() else {
        return -1;
    };
    let overrides = parse_spec_constant_pairs(pairs_ptr, pair_count);
    let (bindless_layout, pack_layout) = {
        let bl = engine.bindless.read();
        let pl = engine.pack_textures.read();
        (bl.descriptor_set_layout, pl.descriptor_set_layout)
    };
    let ctx = engine.ctx.read();
    let mut pnc = engine.pipelines_and_compiler.lock();
    pnc.pipelines.update_specialization_constants_with_layouts(
        &ctx,
        bindless_layout,
        pack_layout,
        &engine.retire,
        overrides,
    );
    pnc.pipelines.compute_pipelines.len() as i32
}

/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_get_last_compile_error(out_buf_ptr: u64, out_buf_cap: i32) -> i32 {
    let Some(engine) = engine() else {
        return 0;
    };
    let pnc = engine.pipelines_and_compiler.lock();
    let Some(ref err_msg) = pnc.pipelines.last_compile_error else {
        return 0;
    };
    let bytes = err_msg.as_bytes();
    if out_buf_ptr == 0 || out_buf_cap <= 0 {
        return bytes.len() as i32;
    }
    let copy_len = bytes.len().min(out_buf_cap as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_buf_ptr as *mut u8, copy_len);
    }
    copy_len as i32
}

/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_bind_gbuffer_images(
    albedo_img: u64,
    normal_img: u64,
    material_img: u64,
    velocity_img: u64,
    depth_img: u64,
    radiance_img: u64,
) {
    unsafe {
        vs_core_bind_gbuffer_views(
            albedo_img,
            0,
            normal_img,
            0,
            material_img,
            0,
            velocity_img,
            0,
            depth_img,
            0,
            radiance_img,
            0,
        );
    }
}

/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_bind_gbuffer_views(
    albedo_img: u64,
    albedo_view: u64,
    normal_img: u64,
    normal_view: u64,
    material_img: u64,
    material_view: u64,
    velocity_img: u64,
    velocity_view: u64,
    depth_img: u64,
    depth_view: u64,
    radiance_img: u64,
    radiance_view: u64,
) {
    let Some(engine) = engine() else {
        return;
    };
    let ctx = engine.ctx.read();
    let mut graph = engine.render_graph.lock();
    let entries = [
        (RenderGraph::RES_GBUFFER_ALBEDO, albedo_img, albedo_view),
        (RenderGraph::RES_GBUFFER_NORMAL, normal_img, normal_view),
        (
            RenderGraph::RES_GBUFFER_MATERIAL,
            material_img,
            material_view,
        ),
        (
            RenderGraph::RES_GBUFFER_VELOCITY,
            velocity_img,
            velocity_view,
        ),
        (RenderGraph::RES_RAW_RADIANCE, radiance_img, radiance_view),
        (RenderGraph::RES_HUDLESS_COLOR, albedo_img, albedo_view),
        (RenderGraph::RES_MC_MAIN_DEPTH, depth_img, depth_view),
    ];

    for (res_id, img_raw, view_raw) in entries {
        if img_raw == 0 {
            continue;
        }
        let (fmt, aspect) = RenderGraph::default_format_and_aspect(res_id);
        graph.bind_resource_image_detailed(
            Some(&ctx),
            res_id,
            vk::Image::from_raw(img_raw),
            vk::ImageView::from_raw(view_raw),
            vk::Sampler::null(),
            fmt,
            aspect,
        );
    }
}

/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_bind_post_targets(
    hudless_img: u64,
    hudless_view: u64,
    prev_hudless_img: u64,
    prev_hudless_view: u64,
    ui_img: u64,
    ui_view: u64,
    interpolated_img: u64,
    interpolated_view: u64,
) {
    let Some(engine) = engine() else {
        return;
    };
    let ctx = engine.ctx.read();
    let mut graph = engine.render_graph.lock();
    let entries = [
        (RenderGraph::RES_HUDLESS_COLOR, hudless_img, hudless_view),
        (
            RenderGraph::RES_PREV_HUDLESS_COLOR,
            prev_hudless_img,
            prev_hudless_view,
        ),
        (RenderGraph::RES_UI_COLOR, ui_img, ui_view),
        (
            RenderGraph::RES_INTERPOLATED_FRAME,
            interpolated_img,
            interpolated_view,
        ),
    ];

    for (res_id, img_raw, view_raw) in entries {
        if img_raw == 0 {
            continue;
        }
        let (fmt, aspect) = RenderGraph::default_format_and_aspect(res_id);
        graph.bind_resource_image_detailed(
            Some(&ctx),
            res_id,
            vk::Image::from_raw(img_raw),
            vk::ImageView::from_raw(view_raw),
            vk::Sampler::null(),
            fmt,
            aspect,
        );
    }
    if interpolated_img != 0 {
        engine.framegen.lock().interpolated_image = interpolated_img;
    }
}

/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_execute_render_graph(
    cmd_raw: u64,
    width: i32,
    height: i32,
) -> i32 {
    if cmd_raw == 0 {
        return 0;
    }
    if crate::crash::global_device_lost_guard().is_rt_vsp_disabled() {
        return 0;
    }
    let Some(engine) = engine() else {
        return 0;
    };
    let ctx = engine.ctx.read();
    let cmd = vk::CommandBuffer::from_raw(cmd_raw);
    let slot = engine.frame_slot();

    // Lock order is fixed across all entrypoints: blas_manager -> pipelines -> bindless ->
    // pack_textures -> sdk_manager -> render_graph.
    let w = width.max(1) as u32;
    let h = height.max(1) as u32;
    let mut blas = engine.blas_manager.lock();
    blas.emissive_lights.set_viewport_dimensions(w, h);
    let tlas_handle = blas.last_built_tlas_handle;
    let mut pnc = engine.pipelines_and_compiler.lock();
    let bindless = engine.bindless.read();
    let pack_set = engine.pack_textures.read().descriptor_set;
    let mut sdk = engine.sdk_manager.lock();
    let mut graph = engine.render_graph.lock();
    let cfg = graph.config;
    if cfg.tier >= RenderTier::Tier2RayQuery && tlas_handle == vk::AccelerationStructureKHR::null()
    {
        return 0;
    }
    if let Some(ref mut profiler) = *engine.timestamp_profiler.lock()
        && profiler.is_enabled()
    {
        let _ = unsafe { profiler.resolve_completed_frame(&ctx.device) };
        unsafe { profiler.begin_frame(&ctx.device, cmd, slot) };
    }
    let world = *engine.world_ubo.read();
    if cfg.tier < RenderTier::Tier2RayQuery {
        if let Some(ref mut profiler) = *engine.timestamp_profiler.lock() {
            unsafe { profiler.begin_pass(&ctx.device, cmd, "emissive_lights") };
        }
        if let Err(err) = blas
            .emissive_lights
            .upload_frame_lights_and_ensure_reservoirs(
                &ctx,
                &engine.retire,
                cmd,
                slot,
                world.frame_index.max(0) as u32,
                pnc.pipelines.camera_pos_world,
                w,
                h,
                false,
            )
        {
            crate::crash::global_device_lost_guard()
                .record_vk_result("ffi::upload_frame_lights", err);
        }
        if let Some(ref mut profiler) = *engine.timestamp_profiler.lock() {
            unsafe { profiler.end_pass(&ctx.device, cmd, "emissive_lights") };
        }
    }
    let jitter = pnc.pipelines.tsr_push_constants.jitter_offset;

    // Compute real world_to_view (inverse of inv_view), previous frame matrices, real frame_time_delta,
    // real camera_near, render_scale-adjusted (rw, rh), and history reset flag for NRD / FSR3 / DLSS / XeSS.
    let fov_y = {
        let m11 = world.inv_proj[1][1].abs();
        if m11 > 1.0e-5 {
            (2.0 * m11.atan()).clamp(0.2, 2.8)
        } else {
            1.2217305
        }
    };
    let camera_near = {
        let p32 = world.inv_proj[2][3].abs();
        if p32 > 1.0e-5 && p32.is_finite() {
            (1.0 / p32).clamp(0.01, 2.0)
        } else {
            0.05
        }
    };
    let world_to_view = invert_rigid_or_affine_4x4(world.inv_view);
    let (prev_view, prev_jitter, sdk_frame_idx, reset_history, frame_dt, render_scale) = {
        let mut rfs = RUNTIME_FRAME_STATE.lock();
        let now = std::time::Instant::now();
        let dt = rfs
            .last_frame_instant
            .map(|prev| {
                now.duration_since(prev)
                    .as_secs_f32()
                    .clamp(1.0 / 1000.0, 0.1)
            })
            .unwrap_or(1.0 / 60.0);
        rfs.last_frame_instant = Some(now);
        rfs.last_frame_dt = dt;
        let pv = if rfs.history_reset_pending {
            world_to_view
        } else {
            rfs.prev_view
        };
        let pj = if rfs.history_reset_pending {
            jitter
        } else {
            rfs.prev_jitter
        };
        let fi = rfs.sdk_frame_index;
        let rst = rfs.history_reset_pending;
        rfs.prev_view = world_to_view;
        rfs.prev_jitter = jitter;
        rfs.sdk_frame_index = rfs.sdk_frame_index.wrapping_add(1);
        rfs.history_reset_pending = false;
        (pv, pj, fi, rst, dt, rfs.render_scale.clamp(0.33, 1.0))
    };
    // Raster depth and motion guides are rendered at output resolution.
    let effective_scale = if cfg.upscaler_mode != 0 && cfg.tier == RenderTier::Tier3PathTracing {
        render_scale
    } else {
        1.0
    };
    let rw = ((w as f32) * effective_scale).round().max(1.0) as u32;
    let rh = ((h as f32) * effective_scale).round().max(1.0) as u32;
    let prev_target_sizes = (graph.internal_render_size, graph.internal_target_size);
    graph.ensure_internal_targets(&ctx, rw, rh, w, h, &engine.retire);
    if prev_target_sizes != (graph.internal_render_size, graph.internal_target_size) {
        pnc.pipelines.descriptor_sig_cache.clear();
    }
    let ngx_quality = if effective_scale > 0.95 {
        NvSdkNgxPerfQuality::Dlaa
    } else if effective_scale >= 0.63 {
        NvSdkNgxPerfQuality::MaxQuality
    } else if effective_scale >= 0.55 {
        NvSdkNgxPerfQuality::Balanced
    } else if effective_scale >= 0.45 {
        NvSdkNgxPerfQuality::MaxPerf
    } else {
        NvSdkNgxPerfQuality::UltraPerformance
    };
    let dlss_flags =
        crate::sdk::ngx::dlss_flags::for_pipeline(cfg.tier == RenderTier::Tier3PathTracing);

    // Separate pure projection matrix (view_to_clip = cur_view_proj * inv(world_to_view)) so NRD
    // does not double-apply the view transform.
    let view_to_clip = mul_mat4_col_major(world.cur_view_proj, world.inv_view);
    let prev_inv_view = invert_rigid_or_affine_4x4(prev_view);
    let view_to_clip_prev = mul_mat4_col_major(world.prev_view_proj, prev_inv_view);
    sdk.set_camera_matrices(crate::sdk::SdkCameraMatrices {
        view_to_clip,
        view_to_clip_prev,
        world_to_view,
        world_to_view_prev: prev_view,
        camera_near,
        camera_far: world.depth_far.max(64.0),
        camera_fov_y_radians: fov_y,
        depth_inverted: cfg.tier != RenderTier::Tier3PathTracing,
        jitter_prev: prev_jitter,
        frame_index: sdk_frame_idx,
    });

    // Configure vendor SDK contexts on demand and reconfigure automatically on viewport resize.
    let extent = ViewportExtent {
        render_width: rw,
        render_height: rh,
        output_width: w,
        output_height: h,
    };
    let fg_extent = ViewportExtent {
        render_width: w,
        render_height: h,
        output_width: w,
        output_height: h,
    };
    let gdpa = ctx.instance.fp_v1_0().get_device_proc_addr;
    let upscaler_kind = match cfg.upscaler_mode {
        2 if sdk.capabilities.fsr3_loaded => UpscalerKind::Fsr3,
        3 if sdk.capabilities.dlss_sr_available => UpscalerKind::DlssSr,
        4 if sdk.capabilities.xess_loaded => UpscalerKind::XeSS,
        _ => UpscalerKind::None,
    };
    if upscaler_kind != UpscalerKind::None {
        let _ = unsafe {
            sdk.configure_upscaler(
                upscaler_kind,
                extent,
                cmd,
                ctx.instance.handle(),
                ctx.physical_device,
                ctx.device.handle(),
                gdpa,
                ngx_quality,
                dlss_flags,
                Some(&engine.retire),
                Some(&ctx.device),
            )
        };
    }
    if cfg.denoiser_mode != 3 {
        sdk.clear_dlss_rr_init_error();
    }
    let denoiser_kind = match cfg.denoiser_mode {
        2 if cfg.effective_denoiser_active() && sdk.capabilities.nrd_loaded => {
            DenoiserKind::NrdReblur
        }
        3 if cfg.tier >= RenderTier::Tier2RayQuery && sdk.capabilities.dlss_rr_available => {
            DenoiserKind::DlssRr
        }
        _ => DenoiserKind::None,
    };
    if denoiser_kind != DenoiserKind::None {
        let needs_nrd_dispatcher = denoiser_kind == DenoiserKind::NrdReblur
            && (!sdk.is_nrd_available() || sdk.denoiser_extent() != extent);
        let result = unsafe {
            sdk.configure_denoiser(
                denoiser_kind,
                extent,
                cmd,
                ngx_quality,
                dlss_flags,
                Some(&engine.retire),
            )
        };
        if let Err(error) = result {
            if denoiser_kind == DenoiserKind::DlssRr {
                sdk.report_denoiser_status(&format!(
                    "DLSS Ray Reconstruction initialization failed ({error}); using SVGF"
                ));
            }
        } else if needs_nrd_dispatcher {
            let _ = unsafe { sdk.configure_nrd_dispatcher(&ctx.device) };
        }
    } else if cfg.denoiser_mode == 3 {
        let reason = if cfg.tier < RenderTier::Tier2RayQuery {
            "requires Tier 2 or Tier 3".to_owned()
        } else {
            sdk.unavailable_reason("dlssRr")
        };
        sdk.report_denoiser_status(&format!(
            "DLSS Ray Reconstruction unavailable ({reason}); using SVGF"
        ));
    }
    if cfg.framegen_mode == 3 && sdk.capabilities.dlss_fg_available {
        if let Err(error) = unsafe {
            sdk.configure_framegen(
                FrameGenKind::DlssFg,
                fg_extent,
                cmd,
                Some(&engine.retire),
                Some(&ctx.device),
            )
        } {
            tracing::warn!(%error, "DLSS frame generation could not start");
            sdk.framegen_failed_mode = 3;
        }
    } else if cfg.framegen_mode == 2 && sdk.capabilities.fsr3_loaded {
        let bb_fmt = if cfg.hdr_enabled {
            vk::Format::R16G16B16A16_SFLOAT
        } else {
            vk::Format::R8G8B8A8_UNORM
        };
        if let Err(error) = unsafe {
            sdk.configure_fsr3_framegen(
                fg_extent,
                ctx.device.handle(),
                ctx.physical_device,
                gdpa,
                bb_fmt,
                Some(&engine.retire),
                Some(&ctx.device),
            )
        } {
            tracing::warn!(%error, "FSR frame generation could not start");
            sdk.framegen_failed_mode = 2;
        }
    }
    let (albedo_res_id, albedo_format) = if cfg.tier == RenderTier::Tier3PathTracing {
        (RenderGraph::RES_PT_ALBEDO, vk::Format::R16G16B16A16_SFLOAT)
    } else {
        (RenderGraph::RES_GBUFFER_ALBEDO, vk::Format::R8G8B8A8_UNORM)
    };
    sdk.set_gbuffer_resources_with_device(
        Some(&ctx.device),
        (
            graph
                .resource_images
                .get(&albedo_res_id)
                .copied()
                .unwrap_or(vk::Image::null()),
            graph
                .resolve_image_view(albedo_res_id)
                .unwrap_or(vk::ImageView::null()),
        ),
        albedo_format,
        (
            graph
                .resource_images
                .get(&RenderGraph::RES_GBUFFER_NORMAL)
                .copied()
                .unwrap_or(vk::Image::null()),
            graph
                .resolve_image_view(RenderGraph::RES_GBUFFER_NORMAL)
                .unwrap_or(vk::ImageView::null()),
        ),
        (
            graph
                .resource_images
                .get(&RenderGraph::RES_GBUFFER_MATERIAL)
                .copied()
                .unwrap_or(vk::Image::null()),
            graph
                .resolve_image_view(RenderGraph::RES_GBUFFER_MATERIAL)
                .unwrap_or(vk::ImageView::null()),
        ),
        (
            graph
                .resource_images
                .get(&RenderGraph::RES_PREV_HUDLESS_COLOR)
                .copied()
                .unwrap_or(vk::Image::null()),
            graph
                .resolve_image_view(RenderGraph::RES_PREV_HUDLESS_COLOR)
                .unwrap_or(vk::ImageView::null()),
        ),
        Some(&engine.retire),
    );

    let dispatch = SdkDispatchParams {
        command_buffer: cmd,
        input_image: graph
            .resource_images
            .get(&RenderGraph::RES_RAW_RADIANCE)
            .copied()
            .unwrap_or(vk::Image::null()),
        input_view: graph
            .resolve_image_view(RenderGraph::RES_RAW_RADIANCE)
            .unwrap_or(vk::ImageView::null()),
        output_image: graph
            .resource_images
            .get(&RenderGraph::RES_UPSCALED_HDR)
            .copied()
            .unwrap_or(vk::Image::null()),
        output_view: graph
            .resolve_image_view(RenderGraph::RES_UPSCALED_HDR)
            .unwrap_or(vk::ImageView::null()),
        depth_image: graph
            .resource_images
            .get(&if cfg.tier == RenderTier::Tier3PathTracing {
                RenderGraph::RES_LINEAR_DEPTH
            } else {
                RenderGraph::RES_MC_MAIN_DEPTH
            })
            .copied()
            .unwrap_or(vk::Image::null()),
        depth_format: if cfg.tier == RenderTier::Tier3PathTracing {
            vk::Format::R32_SFLOAT
        } else {
            vk::Format::D32_SFLOAT
        },
        depth_view: graph
            .resolve_image_view(if cfg.tier == RenderTier::Tier3PathTracing {
                RenderGraph::RES_LINEAR_DEPTH
            } else {
                RenderGraph::RES_MC_MAIN_DEPTH
            })
            .unwrap_or(vk::ImageView::null()),
        motion_image: graph
            .resource_images
            .get(&RenderGraph::RES_GBUFFER_VELOCITY)
            .copied()
            .unwrap_or(vk::Image::null()),
        motion_view: graph
            .resolve_image_view(RenderGraph::RES_GBUFFER_VELOCITY)
            .unwrap_or(vk::ImageView::null()),
        width: rw,
        height: rh,
        out_width: w,
        out_height: h,
        jitter,
        frame_time_delta: frame_dt,
        upscaler_mode: cfg.upscaler_mode,
        denoiser_mode: cfg.denoiser_mode,
        framegen_mode: cfg.framegen_mode,
        reset: reset_history,
        hdr_enabled: cfg.hdr_enabled,
        sun_dir: world.sun_dir,
        nrd_in_diff: (
            graph
                .resource_images
                .get(&RenderGraph::RES_NRD_IN_DIFF)
                .copied()
                .unwrap_or(vk::Image::null()),
            graph
                .resolve_image_view(RenderGraph::RES_NRD_IN_DIFF)
                .unwrap_or(vk::ImageView::null()),
        ),
        nrd_in_spec: (
            graph
                .resource_images
                .get(&RenderGraph::RES_NRD_IN_SPEC)
                .copied()
                .unwrap_or(vk::Image::null()),
            graph
                .resolve_image_view(RenderGraph::RES_NRD_IN_SPEC)
                .unwrap_or(vk::ImageView::null()),
        ),
        nrd_out_spec: (
            graph
                .resource_images
                .get(&RenderGraph::RES_NRD_OUT_SPEC)
                .copied()
                .unwrap_or(vk::Image::null()),
            graph
                .resolve_image_view(RenderGraph::RES_NRD_OUT_SPEC)
                .unwrap_or(vk::ImageView::null()),
        ),
        nrd_normal_rough: (
            graph
                .resource_images
                .get(&RenderGraph::RES_NRD_NORMAL_ROUGH)
                .copied()
                .unwrap_or(vk::Image::null()),
            graph
                .resolve_image_view(RenderGraph::RES_NRD_NORMAL_ROUGH)
                .unwrap_or(vk::ImageView::null()),
        ),
        nrd_viewz: (
            graph
                .resource_images
                .get(&RenderGraph::RES_NRD_VIEWZ)
                .copied()
                .unwrap_or(vk::Image::null()),
            graph
                .resolve_image_view(RenderGraph::RES_NRD_VIEWZ)
                .unwrap_or(vk::ImageView::null()),
        ),
        rt_hit_dist: (
            graph
                .resource_images
                .get(&RenderGraph::RES_RT_HIT_DIST)
                .copied()
                .unwrap_or(vk::Image::null()),
            graph
                .resolve_image_view(RenderGraph::RES_RT_HIT_DIST)
                .unwrap_or(vk::ImageView::null()),
        ),
        dlss_rr_output_image: graph
            .resource_images
            .get(&RenderGraph::RES_UPSCALED_HDR)
            .copied()
            .unwrap_or(vk::Image::null()),
        dlss_rr_output_view: graph
            .resolve_image_view(RenderGraph::RES_UPSCALED_HDR)
            .unwrap_or(vk::ImageView::null()),
        is_tier3: cfg.tier == RenderTier::Tier3PathTracing,
    };

    let graph_pass_name = match cfg.tier {
        RenderTier::Tier0Raster => "render_graph_tier0",
        RenderTier::Tier1ComputeGi => "render_graph_tier1",
        RenderTier::Tier2RayQuery => "render_graph_tier2",
        RenderTier::Tier3PathTracing => "render_graph_tier3",
    };
    if let Some(ref mut profiler) = *engine.timestamp_profiler.lock() {
        unsafe { profiler.begin_pass(&ctx.device, cmd, graph_pass_name) };
    }
    let result = graph.execute_with_context(
        &ctx,
        cmd,
        &mut pnc.pipelines,
        Some(&bindless),
        pack_set,
        tlas_handle,
        slot,
        w,
        h,
        &blas,
        Some(&mut *sdk),
        &dispatch,
        &engine.retire,
    );
    if let Some(ref mut profiler) = *engine.timestamp_profiler.lock() {
        unsafe { profiler.end_pass(&ctx.device, cmd, graph_pass_name) };
    }
    if result > 0 {
        RUNTIME_FRAME_STATE.lock().world_graph_executed_this_frame = true;
    }
    result as i32
}

/// Executes post-GUI passes (`HdrCompose` and `FrameGenInterpolate`) immediately after
/// `GuiRenderer.render()` completes, binding `ui_img` / `ui_view` (`mainTarget.colorTexture` with
/// HUD) to `RES_UI_COLOR` and copying `RES_POST_OUTPUT` into `RES_PREV_HUDLESS_COLOR` for next
/// frame's temporal history.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_execute_after_gui(
    cmd_raw: u64,
    ui_img: u64,
    ui_view: u64,
    width: i32,
    height: i32,
) -> i32 {
    if cmd_raw == 0 {
        return 0;
    }
    if crate::crash::global_device_lost_guard().is_rt_vsp_disabled() {
        return 0;
    }
    let Some(engine) = engine() else {
        return 0;
    };
    let ctx = engine.ctx.read();
    let cmd = vk::CommandBuffer::from_raw(cmd_raw);
    let slot = engine.frame_slot();
    let tlas_handle = vk::AccelerationStructureKHR::from_raw(
        engine
            .last_built_tlas_handle
            .load(std::sync::atomic::Ordering::Acquire),
    );
    let mut pnc = engine.pipelines_and_compiler.lock();
    let bindless = engine.bindless.read();
    let pack_set = engine.pack_textures.read().descriptor_set;
    let mut sdk = engine.sdk_manager.lock();
    let mut graph = engine.render_graph.lock();
    let w = width.max(1) as u32;
    let h = height.max(1) as u32;
    let cfg = graph.config;
    let world = *engine.world_ubo.read();
    let jitter = pnc.pipelines.tsr_push_constants.jitter_offset;
    let (frame_dt, ui_only) = {
        let mut rfs = RUNTIME_FRAME_STATE.lock();
        let ui_only = !rfs.world_graph_executed_this_frame;
        rfs.world_graph_executed_this_frame = false;
        (rfs.last_frame_dt, ui_only)
    };

    let dispatch = SdkDispatchParams {
        command_buffer: cmd,
        input_image: graph
            .resource_images
            .get(&RenderGraph::RES_POST_OUTPUT)
            .copied()
            .unwrap_or(vk::Image::null()),
        input_view: graph
            .resolve_image_view(RenderGraph::RES_POST_OUTPUT)
            .unwrap_or(vk::ImageView::null()),
        output_image: graph
            .resource_images
            .get(&RenderGraph::RES_INTERPOLATED_FRAME)
            .copied()
            .unwrap_or(vk::Image::null()),
        output_view: graph
            .resolve_image_view(RenderGraph::RES_INTERPOLATED_FRAME)
            .unwrap_or(vk::ImageView::null()),
        depth_image: graph
            .resource_images
            .get(&if cfg.tier == RenderTier::Tier3PathTracing {
                RenderGraph::RES_LINEAR_DEPTH
            } else {
                RenderGraph::RES_MC_MAIN_DEPTH
            })
            .copied()
            .unwrap_or(vk::Image::null()),
        depth_format: if cfg.tier == RenderTier::Tier3PathTracing {
            vk::Format::R32_SFLOAT
        } else {
            vk::Format::D32_SFLOAT
        },
        depth_view: graph
            .resolve_image_view(if cfg.tier == RenderTier::Tier3PathTracing {
                RenderGraph::RES_LINEAR_DEPTH
            } else {
                RenderGraph::RES_MC_MAIN_DEPTH
            })
            .unwrap_or(vk::ImageView::null()),
        motion_image: graph
            .resource_images
            .get(&RenderGraph::RES_GBUFFER_VELOCITY)
            .copied()
            .unwrap_or(vk::Image::null()),
        motion_view: graph
            .resolve_image_view(RenderGraph::RES_GBUFFER_VELOCITY)
            .unwrap_or(vk::ImageView::null()),
        width: w,
        height: h,
        out_width: w,
        out_height: h,
        jitter,
        frame_time_delta: frame_dt,
        upscaler_mode: cfg.upscaler_mode,
        denoiser_mode: cfg.denoiser_mode,
        framegen_mode: cfg.framegen_mode,
        reset: {
            let mut rfs = RUNTIME_FRAME_STATE.lock();
            let fg_rst = rfs.fg_history_reset_pending;
            rfs.fg_history_reset_pending = false;
            fg_rst
        },
        hdr_enabled: cfg.hdr_enabled,
        sun_dir: world.sun_dir,
        is_tier3: cfg.tier == RenderTier::Tier3PathTracing,
        ..Default::default()
    };

    if let Some(ref mut profiler) = *engine.timestamp_profiler.lock() {
        unsafe { profiler.begin_pass(&ctx.device, cmd, "post_gui") };
    }
    let result = graph.execute_after_gui_with_context(
        &ctx,
        cmd,
        &mut pnc.pipelines,
        Some(&bindless),
        pack_set,
        tlas_handle,
        slot,
        w,
        h,
        Some(&mut *sdk),
        &dispatch,
        &engine.retire,
        vk::Image::from_raw(ui_img),
        vk::ImageView::from_raw(ui_view),
        ui_only,
    );
    if let Some(ref mut profiler) = *engine.timestamp_profiler.lock() {
        unsafe { profiler.end_pass(&ctx.device, cmd, "post_gui") };
    }
    let interpolated = if !ui_only && cfg.framegen_mode != 0 && sdk.framegen_failed_mode == 0 {
        graph
            .resource_images
            .get(&RenderGraph::RES_INTERPOLATED_FRAME)
            .map_or(0, |image| image.as_raw())
    } else {
        0
    };
    let mut fg = engine.framegen.lock();
    fg.interpolated_image = interpolated;
    if cfg.hdr_enabled && result > 0 {
        let hdr_img = graph
            .resource_images
            .get(&RenderGraph::RES_SWAPCHAIN_OUT)
            .map(|img| img.as_raw())
            .unwrap_or(0);
        fg.hdr_output_image = hdr_img;
        fg.hdr_output_valid_this_frame = hdr_img != 0;
    } else {
        fg.hdr_output_valid_this_frame = false;
    }
    result as i32
}

/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_register_section_slice(
    key_id: u64,
    layer_ordinal: i32,
    vk_buffer: u64,
    byte_offset: u64,
    byte_size: u64,
) {
    if layer_ordinal < 0 || vk_buffer == 0 || byte_size == 0 {
        return;
    }
    let v_count = (byte_size / 40).min(i32::MAX as u64) as i32;
    unsafe {
        vs_core_register_section_slice_ex(
            key_id,
            layer_ordinal,
            0,
            vk_buffer,
            byte_offset,
            byte_size,
            40,
            v_count,
            1,
        );
    }
}

/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_register_section_slice_ex(
    key_id: u64,
    layer_ordinal: i32,
    section_pos_packed: i64,
    vk_buffer: u64,
    byte_offset: u64,
    byte_size: u64,
    vertex_stride: i32,
    vertex_count: i32,
    is_region_shifted: i32,
) {
    if layer_ordinal < 0 || vk_buffer == 0 || byte_size == 0 || vertex_count < 0 {
        return;
    }
    let Some(engine) = engine() else {
        return;
    };
    let ctx = engine.ctx.read();
    engine.blas_manager.lock().register_slice(
        key_id,
        layer_ordinal as u32,
        section_pos_packed,
        vk_buffer,
        byte_offset,
        byte_size,
        vertex_stride.max(20) as u32,
        vertex_count.max(0) as u32,
        is_region_shifted != 0,
        Some(&ctx),
        Some(&engine.retire),
    );
}

/// Registers a terrain slice in a single atomic call with all geometry metadata (`section_pos_packed`,
/// `vertex_stride`, `vertex_count`, `is_region_shifted`) and LabPBR bindless slots + material flags.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_register_section_slice_with_pbr(
    key_id: u64,
    layer_ordinal: i32,
    section_pos_packed: i64,
    vk_buffer: u64,
    byte_offset: u64,
    byte_size: u64,
    vertex_stride: i32,
    vertex_count: i32,
    is_region_shifted: i32,
    albedo_slot: i32,
    normal_slot: i32,
    specular_slot: i32,
    material_flags: u32,
) {
    if layer_ordinal < 0 || vk_buffer == 0 || byte_size == 0 || vertex_count < 0 {
        return;
    }
    let Some(engine) = engine() else {
        return;
    };
    let ctx = engine.ctx.read();
    engine.blas_manager.lock().register_slice_with_textures(
        key_id,
        layer_ordinal as u32,
        section_pos_packed,
        vk_buffer,
        byte_offset,
        byte_size,
        vertex_stride.max(20) as u32,
        vertex_count.max(0) as u32,
        is_region_shifted != 0,
        albedo_slot.max(0) as u32,
        normal_slot.max(0) as u32,
        specular_slot.max(0) as u32,
        material_flags,
        Some(&ctx),
        Some(&engine.retire),
    );
}

/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_unregister_section_slice(key_id: u64, layer_ordinal: i32) {
    if layer_ordinal < 0 {
        return;
    }
    let Some(engine) = engine() else {
        return;
    };
    let ctx = engine.ctx.read();
    engine.blas_manager.lock().unregister_slice(
        key_id,
        layer_ordinal as u32,
        Some(&ctx),
        Some(&engine.retire),
    );
}

/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_register_texture(
    vk_image: u64,
    width: i32,
    height: i32,
    mip_levels: i32,
) {
    unsafe {
        vs_core_register_texture_ex(vk_image, 0, width, height, mip_levels, 0);
    }
}

/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_register_texture_ex(
    vk_image: u64,
    vk_image_view: u64,
    width: i32,
    height: i32,
    mip_levels: i32,
    pbr_role: i32,
) -> i32 {
    unsafe {
        vs_core_register_texture_with_sampler(
            vk_image,
            vk_image_view,
            0,
            vk::Format::R8G8B8A8_UNORM.as_raw(),
            width,
            height,
            mip_levels,
            pbr_role,
        )
    }
}

/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_register_texture_with_sampler(
    vk_image: u64,
    vk_image_view: u64,
    vk_sampler: u64,
    vk_format: i32,
    width: i32,
    height: i32,
    mip_levels: i32,
    pbr_role: i32,
) -> i32 {
    let Some(engine) = engine() else {
        return -1;
    };
    use crate::gpu::bindless::TexturePbrRole;
    let role = TexturePbrRole::from_u32(pbr_role.max(0) as u32);
    let format = if vk_format > 0 {
        vk::Format::from_raw(vk_format)
    } else {
        vk::Format::R8G8B8A8_UNORM
    };
    let ctx = engine.ctx.read();
    engine
        .bindless
        .write()
        .register_or_update_with_sampler_deferred(
            vk_image,
            vk_image_view,
            vk_sampler,
            format,
            width.max(1) as u32,
            height.max(1) as u32,
            mip_levels.max(1) as u32,
            role,
            Some(&ctx),
            Some(&engine.retire),
        )
        .map(|s| s as i32)
        .unwrap_or(-1)
}

/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_unregister_texture(vk_image: u64) {
    let Some(engine) = engine() else {
        return;
    };
    let ctx = engine.ctx.read();
    engine
        .bindless
        .write()
        .unregister_deferred(vk_image, Some(&ctx), Some(&engine.retire));
}

/// Uploads a shader-pack `[resources.*]` texture into descriptor Set 3 at `slot`.
/// `rgba_ptr` must point to `width * height * 4` RGBA8 bytes.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_upload_pack_texture(
    slot: u32,
    width: i32,
    height: i32,
    rgba_ptr: u64,
    rgba_len: i32,
    filter_linear: i32,
    wrap_repeat: i32,
) -> i32 {
    if slot >= PACK_RESOURCE_SLOTS || width <= 0 || height <= 0 || rgba_ptr == 0 {
        return -1;
    }
    let expected = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(4);
    if rgba_len < 0 || (rgba_len as usize) < expected {
        return -2;
    }
    let bytes = unsafe { slice::from_raw_parts(rgba_ptr as *const u8, expected) };
    let Some(engine) = engine() else {
        PENDING_PACK_UPLOADS.lock().push(PendingPackUpload {
            slot,
            width: width as u32,
            height: height as u32,
            rgba: bytes.to_vec(),
            filter_linear: filter_linear != 0,
            wrap_repeat: wrap_repeat != 0,
        });
        eprintln!(
            "[vulkanshaders-core] Pack texture slot={slot} {width}x{height} deferred (engine not initialized); will replay at vs_core_init"
        );
        return VS_CORE_PACK_UPLOAD_DEFERRED;
    };
    let ctx = engine.ctx.read();
    unsafe {
        engine.pack_textures.write().upload_rgba_deferred(
            &ctx,
            Some(&engine.retire),
            slot,
            width as u32,
            height as u32,
            bytes,
            filter_linear != 0,
            wrap_repeat != 0,
        )
    }
}

/// Writes the raw `VkDescriptorSetLayout` handles that G-Buffer shaders must append to their
/// `VkPipelineLayout`: `[bindless (Set 2), pack resources (Set 3)]`. Returns the number written.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_get_custom_descriptor_set_layouts(
    out_ptr: u64,
    max_count: i32,
) -> i32 {
    if out_ptr == 0 {
        return 0;
    }
    let Some(engine) = engine() else {
        return 0;
    };
    let layouts = {
        let bl = engine.bindless.read();
        let pl = engine.pack_textures.read();
        [
            bl.descriptor_set_layout.as_raw(),
            pl.descriptor_set_layout.as_raw(),
        ]
    };
    let count = layouts.len().min(max_count.max(0) as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(layouts.as_ptr(), out_ptr as *mut u64, count);
    }
    count as i32
}

/// Writes the raw `VkDescriptorSet` handles for the engine's global bindless texture table
/// (Set 2) and pack resources (Set 3) so the JVM can cache and bind them directly without
/// per-draw FFI overhead or mutex locks. Returns `1` on success, `0` if unavailable.
/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_get_custom_descriptor_sets(
    out_bindless_set: *mut u64,
    out_pack_set: *mut u64,
) -> i32 {
    let Some(engine) = engine() else {
        return 0;
    };
    let (bl_raw, pl_raw) = {
        let bl = engine.bindless.read();
        let pl = engine.pack_textures.read();
        (bl.descriptor_set.as_raw(), pl.descriptor_set.as_raw())
    };
    unsafe {
        if !out_bindless_set.is_null() {
            *out_bindless_set = bl_raw;
        }
        if !out_pack_set.is_null() {
            *out_pack_set = pl_raw;
        }
    }
    if bl_raw != 0 || pl_raw != 0 { 1 } else { 0 }
}

/// Binds the engine-owned custom descriptor sets (bindless + pack resources) on `cmd` starting at
/// `first_set`, using `pipeline_layout` and `bind_point` supplied by the caller.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_bind_custom_descriptor_sets(
    cmd_raw: u64,
    pipeline_layout_raw: u64,
    bind_point: i32,
    first_set: i32,
) {
    if cmd_raw == 0 || pipeline_layout_raw == 0 || first_set < 0 {
        return;
    }
    let Some(engine) = engine() else {
        return;
    };
    let bl = engine.bindless.read();
    let pl = engine.pack_textures.read();
    let point = if bind_point == 1 {
        vk::PipelineBindPoint::COMPUTE
    } else {
        vk::PipelineBindPoint::GRAPHICS
    };
    let cmd = vk::CommandBuffer::from_raw(cmd_raw);
    let layout = vk::PipelineLayout::from_raw(pipeline_layout_raw);
    let base_set = first_set as u32;
    let ctx_guard = engine.ctx.read();
    unsafe {
        if bl.descriptor_set != vk::DescriptorSet::null()
            && pl.descriptor_set != vk::DescriptorSet::null()
        {
            let sets = [bl.descriptor_set, pl.descriptor_set];
            ctx_guard
                .device
                .cmd_bind_descriptor_sets(cmd, point, layout, base_set, &sets, &[]);
        } else if bl.descriptor_set != vk::DescriptorSet::null() {
            let sets = [bl.descriptor_set];
            ctx_guard
                .device
                .cmd_bind_descriptor_sets(cmd, point, layout, base_set, &sets, &[]);
        } else if pl.descriptor_set != vk::DescriptorSet::null() {
            let sets = [pl.descriptor_set];
            ctx_guard
                .device
                .cmd_bind_descriptor_sets(cmd, point, layout, base_set + 1, &sets, &[]);
        }
    }
}

/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_can_enable_framegen(mode: i32) -> i32 {
    let Some(engine) = engine() else {
        return 0;
    };
    {
        let fg = engine.framegen.lock();
        if fg.active_swapchain == 0 || fg.swapchain_images.is_empty() {
            return 0;
        }
    }
    let supported = match mode {
        1 => engine
            .pipelines_and_compiler
            .lock()
            .pipelines
            .compute_pipelines
            .contains_key("optical_flow_fg.comp"),
        2 => engine.sdk_manager.lock().capabilities.fsr3_loaded,
        3 => engine.sdk_manager.lock().capabilities.dlss_fg_available,
        _ => false,
    };
    i32::from(supported)
}

/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_take_framegen_failure() -> i32 {
    engine().map_or(0, |engine| {
        std::mem::take(&mut engine.sdk_manager.lock().framegen_failed_mode) as i32
    })
}

/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_configure_pipeline(
    tier_index: i32,
    denoiser_mode: i32,
    upscaler_mode: i32,
    framegen_mode: i32,
    hdr_enabled: i32,
    overlay_pass_enabled: i32,
    ser_enabled: i32,
) -> i32 {
    let Some(engine) = engine() else {
        return -1;
    };
    crate::gpu::set_ser_reordering_enabled(ser_enabled != 0);
    let requested_tier =
        RenderTier::from_index(tier_index.max(0) as u32).unwrap_or(RenderTier::Tier0Raster);
    let policy = engine.platform_tier_policy();
    let tier = policy.clamp_render_tier(requested_tier);
    if tier != requested_tier {
        tracing::warn!(
            requested = ?requested_tier,
            clamped = ?tier,
            reason = %policy.reason,
            "Clamped requested render tier to platform policy ceiling"
        );
    }
    let cfg = GraphConfig {
        tier,
        denoiser_mode: denoiser_mode.max(0) as u32,
        upscaler_mode: upscaler_mode.max(0) as u32,
        framegen_mode: framegen_mode.max(0) as u32,
        hdr_enabled: hdr_enabled != 0,
        overlay_pass_enabled: overlay_pass_enabled != 0,
    };
    {
        let mut graph = engine.render_graph.lock();
        if graph.config != cfg {
            // A tier change rebuilds the whole render graph. Wait for all in-flight GPU work
            // to finish first: otherwise the first submitted frame of the new tier can reference
            // targets that are being replaced, stalling the queue until RenderPearl's 5 s
            // semaphore timeout fires.
            let ctx = engine.ctx.read();
            let wait_res = crate::gpu::with_queue_lock(|| unsafe { ctx.device.device_wait_idle() });
            if let Err(err) = wait_res {
                crate::crash::global_device_lost_guard()
                    .record_vk_result("ffi::configure_pipeline::device_wait_idle", err);
            }
            graph.rebuild(cfg);
        }
    }
    {
        let mut fg = engine.framegen.lock();
        let mode = FrameGenBackendMode::from_u32(cfg.framegen_mode);
        if fg.mode != mode {
            fg.pending_present_job = None;
            fg.interpolated_image = 0;
        }
        fg.mode = mode;
        fg.hdr_enabled = cfg.hdr_enabled;
    }
    engine
        .framegen_mode
        .store(cfg.framegen_mode, std::sync::atomic::Ordering::Release);
    engine
        .pipelines_and_compiler
        .lock()
        .pipelines
        .hdr_push_constants
        .output_mode = if cfg.hdr_enabled {
        let rfs = RUNTIME_FRAME_STATE.lock();
        // VK_FORMAT_A2B10G10R10_UNORM_PACK32 = 64, VK_COLOR_SPACE_HDR10_ST2084_EXT = 1000104008
        if rfs.swapchain_format == 64 || rfs.swapchain_colorspace == 1000104008 {
            2
        } else {
            1
        }
    } else {
        0
    };
    engine.render_graph.lock().compiled_steps.len() as i32
}

/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_set_shadow_config(
    shadow_resolution: i32,
    shadow_distance_chunks: i32,
) {
    let Some(engine) = engine() else {
        return;
    };
    let max_shadow_res = engine.ctx.read().capabilities.max_shadow_resolution();
    let res = (shadow_resolution.max(512) as u32).clamp(512, max_shadow_res);
    let dist = (shadow_distance_chunks.max(4) as u32).clamp(4, 64);
    {
        let mut graph = engine.render_graph.lock();
        if graph.shadow_resolution != res {
            graph.internal_target_size = (0, 0);
        }
        graph.shadow_resolution = res;
        graph.shadow_distance_chunks = dist;
    }
    {
        let mut pnc = engine.pipelines_and_compiler.lock();
        pnc.pipelines.shadow_resolution = res;
        pnc.pipelines.shadow_distance_chunks = dist;
    }
}

/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_intercept_surface_blit(
    cmd_raw: u64,
    src_image: u64,
    src_layout: i32,
    dst_image: u64,
    dst_layout: i32,
    width: i32,
    height: i32,
) -> i32 {
    if cmd_raw == 0 {
        return 0;
    }
    let Some(engine) = engine() else {
        return 0;
    };
    let ctx = engine.ctx.read();
    let intercepted = engine.framegen.lock().evaluate_and_intercept_blit(
        Some(&ctx),
        cmd_raw,
        src_image,
        src_layout,
        dst_image,
        dst_layout,
        width.max(1) as u32,
        height.max(1) as u32,
    );
    i32::from(intercepted)
}

/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_set_overlay_depth_params(
    z_scale: f32,
    z_offset: f32,
    w_scale: f32,
    w_offset: f32,
    src_w: i32,
    src_h: i32,
    dst_w: i32,
    dst_h: i32,
) -> f32 {
    let params = DepthReprojectPushConstants {
        z_scale,
        z_offset,
        w_scale,
        w_offset,
        source_size: [src_w.max(1) as u32, src_h.max(1) as u32],
        target_size: [dst_w.max(1) as u32, dst_h.max(1) as u32],
    };
    if let Some(engine) = engine() {
        engine
            .pipelines_and_compiler
            .lock()
            .pipelines
            .last_overlay_depth_params = Some(params);
    }
    params.evaluate_device_depth(16.0)
}

/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_reflex_set_swapchain(swapchain_raw: u64) {
    let Some(engine) = engine() else {
        return;
    };
    let ctx = engine.ctx.read();
    engine
        .framegen
        .lock()
        .set_swapchain(Some(&ctx), swapchain_raw);
}

/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_reflex_marker(marker_id: i32) {
    let Some(engine) = engine() else {
        return;
    };
    let ctx = engine.ctx.read();
    engine.framegen.lock().set_reflex_marker(&ctx, marker_id);
}

/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_reflex_sleep_and_sample_input() {
    let Some(engine) = engine() else {
        return;
    };
    let ctx = engine.ctx.read();
    engine
        .framegen
        .lock()
        .begin_frame_sleep_and_input_sample(&ctx);
}

/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_intercept_acquire_next_image(
    swapchain_raw: u64,
    timeout_ns: u64,
    semaphore_raw: u64,
    fence_raw: u64,
) -> i32 {
    let Some(engine) = engine() else {
        return -1;
    };
    let ctx = engine.ctx.read();
    engine.framegen.lock().intercept_acquire_next_image(
        &ctx,
        swapchain_raw,
        timeout_ns,
        semaphore_raw,
        fence_raw,
    )
}

/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_intercept_queue_present(
    queue_raw: u64,
    swapchain_raw: u64,
    image_index: i32,
    wait_semaphore_raw: u64,
) -> i32 {
    if image_index < 0 {
        return 0;
    }
    let Some(engine) = engine() else {
        return 0;
    };
    let ctx = engine.ctx.read();
    i32::from(engine.framegen.lock().intercept_queue_present(
        &ctx,
        queue_raw,
        swapchain_raw,
        image_index as u32,
        wait_semaphore_raw,
    ))
}

/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_wait_present_idle() {
    let Some(engine) = engine() else {
        return;
    };
    engine.framegen.lock().wait_present_idle();
}

/// Uploads dynamic block/point/handheld emissive lights for ReSTIR DI and hardware NEE shadow rays.
/// `lights_ptr` points to `light_count` 48-byte `WorldEmissiveLightDesc` records.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_submit_emissive_lights(
    lights_ptr: u64,
    light_count: i32,
    _cam_x: f64,
    _cam_y: f64,
    _cam_z: f64,
) -> i32 {
    let Some(engine) = engine() else {
        return 0;
    };
    let count = light_count.max(0) as usize;
    let slice = if lights_ptr != 0 && count > 0 {
        unsafe {
            slice::from_raw_parts(
                lights_ptr as *const crate::rt::lights::WorldEmissiveLightDesc,
                count,
            )
        }
    } else {
        &[]
    };
    let mut blas = engine.blas_manager.lock();
    blas.emissive_lights.set_explicit_lights(slice);
    slice.len() as i32
}

/// Registers or updates a Distant Horizons LOD terrain mesh slice for ray-traced far-field BLAS/TLAS.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_register_lod_slice(
    lod_key_id: u64,
    level: i32,
    origin_block_x: i32,
    origin_block_y: i32,
    origin_block_z: i32,
    scale_blocks: f32,
    vk_buffer: u64,
    byte_offset: u64,
    byte_size: u64,
    vertex_stride: i32,
    vertex_count: i32,
    index_buffer: u64,
    index_count: i32,
    albedo_color_abgr: u32,
) {
    if vk_buffer == 0
        || byte_size == 0
        || level < 0
        || vertex_stride <= 0
        || vertex_count < 0
        || index_count < 0
    {
        return;
    }
    let Some(engine) = engine() else {
        return;
    };
    let ctx = engine.ctx.read();
    engine.blas_manager.lock().register_lod_slice(
        lod_key_id,
        level as u32,
        [origin_block_x, origin_block_y, origin_block_z],
        scale_blocks,
        vk_buffer,
        byte_offset,
        byte_size,
        vertex_stride.max(28) as u32,
        vertex_count as u32,
        index_buffer,
        index_count as u32,
        albedo_color_abgr,
        Some(&ctx),
        Some(&engine.retire),
    );
}

/// Uploads a CPU-unpacked 28-byte/vertex Distant Horizons LOD mesh (`pos: [f32; 3]`,
/// `color_abgr: u32`, `uv: [f32; 2]`, `light_normal: u32`) and `u32` index buffer into owned
/// BDA-enabled `VkBuffer`s inside `LodBlasManager`. Returns `1` on success, `0` on failure.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_upload_lod_mesh_cpu(
    lod_key_id: u64,
    lod_level: u32,
    origin_x: i32,
    origin_y: i32,
    origin_z: i32,
    scale_blocks: f32,
    vertices_ptr: u64,
    vertex_count: i32,
    indices_ptr: u64,
    index_count: i32,
    material_flags: u32,
) -> i32 {
    let has_indices = indices_ptr != 0 && index_count >= 3;
    if vertices_ptr == 0 {
        return 0;
    }
    if has_indices {
        if vertex_count < 3 {
            return 0;
        }
    } else if indices_ptr != 0 || index_count != 0 || vertex_count < 4 {
        return 0;
    }
    let Some(vertex_bytes_len) = (vertex_count as usize).checked_mul(28) else {
        return 0;
    };
    let index_bytes_len = if has_indices {
        let Some(len) = (index_count as usize).checked_mul(std::mem::size_of::<u32>()) else {
            return 0;
        };
        len
    } else {
        0
    };
    let Some(engine) = engine() else {
        return 0;
    };
    let vertex_bytes =
        unsafe { slice::from_raw_parts(vertices_ptr as *const u8, vertex_bytes_len) };
    let index_bytes: &[u8] = if has_indices {
        unsafe { slice::from_raw_parts(indices_ptr as *const u8, index_bytes_len) }
    } else {
        &[]
    };
    let ctx = engine.ctx.read();
    let ok = engine.blas_manager.lock().upload_lod_mesh_cpu(
        lod_key_id,
        lod_level,
        [origin_x, origin_y, origin_z],
        scale_blocks,
        vertex_bytes,
        vertex_count as u32,
        index_bytes,
        index_count as u32,
        material_flags,
        &ctx,
        &engine.retire,
    );
    i32::from(ok)
}

/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_unregister_lod_slice(lod_key_id: u64) {
    let Some(engine) = engine() else {
        return;
    };
    let ctx = engine.ctx.read();
    engine
        .blas_manager
        .lock()
        .unregister_lod_slice(lod_key_id, Some(&ctx), Some(&engine.retire));
}

/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_clear_lod_slices() {
    let Some(engine) = engine() else {
        return;
    };
    let ctx = engine.ctx.read();
    engine
        .blas_manager
        .lock()
        .clear_lod_slices(Some(&ctx), Some(&engine.retire));
}

/// Updates the minimum horizontal distance (in blocks) before Distant Horizons LOD instances
/// are injected into the ray-tracing TLAS.
/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_set_lod_cutoff_distance(cutoff_blocks: f32) -> i32 {
    let Some(engine) = engine() else {
        return -1;
    };
    engine
        .blas_manager
        .lock()
        .set_lod_cutoff_distance(cutoff_blocks);
    0
}

/// Binds Distant Horizons external G-Buffer image/view handles and updates the 80-byte
/// `DhFrameUniforms` block consumed by `compute_lighting`, `rayquery_lighting`, and `volumetric_fog`.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_bind_dh_targets(
    dh_albedo_image: u64,
    dh_albedo_view: u64,
    dh_normal_image: u64,
    dh_normal_view: u64,
    dh_material_image: u64,
    dh_material_view: u64,
    dh_depth_image: u64,
    dh_depth_view: u64,
    width: u32,
    height: u32,
    inv_view_proj_ptr: *const f32,
    fade_start: f32,
    fade_end: f32,
    dh_far_clip: f32,
    active: u32,
) -> i32 {
    let Some(engine) = engine() else {
        return -1;
    };
    let mut dh_inv_view_proj = [0.0f32; 16];
    if !inv_view_proj_ptr.is_null() {
        let src = unsafe { slice::from_raw_parts(inv_view_proj_ptr, 16) };
        dh_inv_view_proj.copy_from_slice(src);
    }
    let uniforms = DhFrameUniforms {
        dh_inv_view_proj,
        fade_start: if fade_start.is_finite() {
            fade_start
        } else {
            0.0
        },
        fade_end: if fade_end.is_finite() { fade_end } else { 0.0 },
        dh_far_clip: if dh_far_clip.is_finite() {
            dh_far_clip
        } else {
            4096.0
        },
        dh_active: if active != 0 { 1 } else { 0 },
    };
    let ctx = engine.ctx.read();
    engine.render_graph.lock().bind_dh_targets(
        Some(&ctx),
        (dh_albedo_image, dh_albedo_view),
        (dh_normal_image, dh_normal_view),
        (dh_material_image, dh_material_view),
        (dh_depth_image, dh_depth_view),
        width,
        height,
        uniforms,
    );
    0
}

/// Preprocesses a raster G-Buffer GLSL shader (`gbuffer_terrain_region.vsh/.fsh`, `gbuffer_entity.fsh`, etc.)
/// through the unified Rust `vsp-compiler` lexer and preprocessor (`#include`, `@hook`, namespace isolation).
/// Writes the resulting UTF-8 GLSL source into `out_ptr` (up to `out_cap` bytes) and returns the full byte length.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_preprocess_raster_shader(
    source_ptr: u64,
    source_len: i32,
    pack_dir_ptr: u64,
    pack_dir_len: i32,
    stage_is_fragment: i32,
    out_ptr: u64,
    out_cap: i32,
) -> i32 {
    if source_ptr == 0 || source_len <= 0 {
        return -1;
    }
    let src_bytes = unsafe { slice::from_raw_parts(source_ptr as *const u8, source_len as usize) };
    let Ok(src_str) = str::from_utf8(src_bytes) else {
        return -1;
    };
    let pack_dir_opt = if pack_dir_ptr != 0 && pack_dir_len > 0 {
        let dir_bytes =
            unsafe { slice::from_raw_parts(pack_dir_ptr as *const u8, pack_dir_len as usize) };
        str::from_utf8(dir_bytes).ok().map(Path::new)
    } else {
        None
    };

    let tier = engine()
        .map(|e| e.render_graph.lock().config.tier)
        .unwrap_or(RenderTier::Tier0Raster);

    let processed = if let Some(pack_dir) = pack_dir_opt {
        let Ok((manifest, chain_dirs)) = VspCompiler::resolve_pack_inheritance(pack_dir, &[])
        else {
            return -2;
        };
        let mut spec_overrides = std::collections::BTreeMap::new();
        let layout = manifest.build_parameter_layout();
        {
            let rfs = RUNTIME_FRAME_STATE.lock();
            for (key, &const_id) in &layout.specialization_constant_ids {
                if let Some(&raw_bits) = rfs.spec_overrides_by_id.get(&const_id)
                    && let Some(decl) = manifest.settings.get(key)
                {
                    let val = match decl.value_type {
                        vsp_compiler::manifest::SettingValueType::Float => f32::from_bits(raw_bits),
                        vsp_compiler::manifest::SettingValueType::Int => (raw_bits as i32) as f32,
                        vsp_compiler::manifest::SettingValueType::Bool => {
                            if raw_bits != 0 {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        vsp_compiler::manifest::SettingValueType::Enum => (raw_bits as i32) as f32,
                        vsp_compiler::manifest::SettingValueType::ColorRgb => {
                            f32::from_bits(raw_bits)
                        }
                        vsp_compiler::manifest::SettingValueType::Vec3 => f32::from_bits(raw_bits),
                    };
                    spec_overrides.insert(key.clone(), val);
                }
            }
        }
        let chain_refs: Vec<&Path> = chain_dirs.iter().map(|p| p.as_path()).collect();
        match vsp_compiler::Preprocessor::preprocess_raster_source_with_chain(
            src_str,
            &chain_refs,
            &manifest,
            tier as u32,
            stage_is_fragment != 0,
            &spec_overrides,
        ) {
            Ok(out) => out,
            Err(_) => return -2,
        }
    } else {
        src_str.to_owned()
    };

    let out_bytes = processed.as_bytes();
    if out_ptr != 0 && out_cap > 0 {
        let copy_len = out_bytes.len().min(out_cap as usize);
        unsafe {
            std::ptr::copy_nonoverlapping(out_bytes.as_ptr(), out_ptr as *mut u8, copy_len);
        }
    }
    out_bytes.len() as i32
}

/// Clears all resident terrain & LOD BLAS slices (called on world unload / server switch).
/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_clear_section_slices() {
    let Some(engine) = engine() else {
        return;
    };
    let ctx = engine.ctx.read();
    let mut blas = engine.blas_manager.lock();
    blas.clear_all_slices(Some(&ctx), Some(&engine.retire));
    engine
        .last_built_tlas_handle
        .store(0, std::sync::atomic::Ordering::Release);
    engine.framegen.lock().on_world_unloaded();
    {
        let mut rfs = RUNTIME_FRAME_STATE.lock();
        rfs.history_reset_pending = true;
        rfs.fg_history_reset_pending = true;
        rfs.world_graph_executed_this_frame = false;
    }
}

/// Resets per-world HDR / FrameGen / temporal state when returning to the main menu or switching worlds.
/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_on_world_unloaded() {
    let Some(engine) = engine() else {
        return;
    };
    engine.framegen.lock().on_world_unloaded();
    engine.reset_temporal_history();
    {
        let mut rfs = RUNTIME_FRAME_STATE.lock();
        rfs.history_reset_pending = true;
        rfs.fg_history_reset_pending = true;
        rfs.world_graph_executed_this_frame = false;
    }
}

/// Binds RenderPearl's GPU timeline semaphore to `RetireQueue` for real hardware completion checks.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_bind_timeline_semaphore(semaphore_raw: u64) {
    let Some(engine) = engine() else {
        return;
    };
    engine.retire.bind_timeline_semaphore(semaphore_raw);
}

/// Records the GPU timeline value actually submitted for the current frame.
/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_record_submission(timeline_value: u64) {
    let Some(engine) = engine() else {
        return;
    };
    engine.retire.record_submission(timeline_value);
}

/// Notifies `RetireQueue` that GPU timeline semaphore value `completed_value` has finished executing.
/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_notify_timeline_completed(completed_value: u64) {
    let Some(engine) = engine() else {
        return;
    };
    engine.retire.notify_timeline_completed(completed_value);
}

/// Sets internal render resolution scale (`0.5..=1.0`) for TSR / FSR3 / DLSS / XeSS upscalers.
/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_set_render_scale(scale: f32) {
    if scale.is_finite() {
        RUNTIME_FRAME_STATE.lock().render_scale = scale.clamp(0.33, 1.0);
    }
}

/// Configures HDR peak display luminance and UI paper-white luminance (in nits).
/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_set_hdr_params(peak_nits: f32, paper_white_nits: f32) {
    let peak = if peak_nits.is_finite() {
        peak_nits.clamp(250.0, 4000.0)
    } else {
        1000.0
    };
    let paper = if paper_white_nits.is_finite() {
        paper_white_nits.clamp(80.0, 600.0)
    } else {
        200.0
    };
    {
        let mut rfs = RUNTIME_FRAME_STATE.lock();
        rfs.hdr_peak_nits = peak;
        rfs.hdr_paper_white_nits = paper;
    }
    if let Some(engine) = engine() {
        let mut pnc = engine.pipelines_and_compiler.lock();
        pnc.pipelines.hdr_push_constants.peak_brightness_nits = peak;
        pnc.pipelines.hdr_push_constants.paper_white_nits = paper;
    }
}

/// Records the active swapchain format & color space so `hdr_compose.comp` picks scRGB vs HDR10 PQ.
/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_on_swapchain_created(
    format: i32,
    color_space: i32,
    width: i32,
    height: i32,
) {
    let fmt_u32 = format.max(0) as u32;
    let cs_u32 = color_space.max(0) as u32;
    let sw_w = width.max(0) as u32;
    let sw_h = height.max(0) as u32;
    {
        let mut rfs = RUNTIME_FRAME_STATE.lock();
        rfs.swapchain_format = fmt_u32;
        rfs.swapchain_colorspace = cs_u32;
    }
    if let Some(engine) = engine() {
        {
            let mut fg = engine.framegen.lock();
            fg.swapchain_width = sw_w;
            fg.swapchain_height = sw_h;
        }
        let hdr_enabled = engine.render_graph.lock().config.hdr_enabled;
        engine
            .pipelines_and_compiler
            .lock()
            .pipelines
            .hdr_push_constants
            .output_mode = if hdr_enabled {
            // VK_FORMAT_A2B10G10R10_UNORM_PACK32 = 64, VK_COLOR_SPACE_HDR10_ST2084_EXT = 1000104008
            if fmt_u32 == 64 || cs_u32 == 1000104008 {
                2
            } else {
                1
            }
        } else {
            0
        };
    }
}

/// Sets the Minecraft game directory so `natives/` and `config/vulkanshaders/` resolve portably.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_set_game_dir(path_ptr: *const u8, path_len: i32) {
    if path_ptr.is_null() || path_len <= 0 {
        return;
    }
    let bytes = unsafe { slice::from_raw_parts(path_ptr, path_len as usize) };
    if let Ok(s) = str::from_utf8(bytes)
        && !s.is_empty()
    {
        let path = Path::new(s);
        crate::crash::init_early();
        crate::crash::set_game_dir(path);
        RUNTIME_FRAME_STATE.lock().game_dir = Some(path.to_path_buf());
    }
}

/// Returns the number of queued native error/warning entries waiting to be drained.
/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_get_native_error_count() -> u32 {
    crate::crash::pending_error_count()
}

/// Drains queued native errors (`LEVEL|target|message` separated by `\n`) into `out_buf_ptr`.
///
/// If `out_buf_ptr == 0 || out_buf_cap <= 0`, returns the full UTF-8 byte length without clearing
/// the buffer. Otherwise copies up to `out_buf_cap` bytes, clears the buffer, and returns the
/// number of bytes copied.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_drain_native_errors(out_buf_ptr: u64, out_buf_cap: i32) -> i32 {
    if out_buf_ptr == 0 || out_buf_cap <= 0 {
        return crate::crash::peek_native_errors_len() as i32;
    }
    let drained = crate::crash::drain_native_errors();
    let bytes = drained.as_bytes();
    if bytes.is_empty() {
        return 0;
    }
    let copy_len = bytes.len().min(out_buf_cap as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_buf_ptr as *mut u8, copy_len);
    }
    copy_len as i32
}

/// Flags that temporal history must be reset on the next vendor upscaler/denoiser dispatch.
/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_mark_history_reset() {
    {
        let mut rfs = RUNTIME_FRAME_STATE.lock();
        rfs.history_reset_pending = true;
        rfs.fg_history_reset_pending = true;
    }
}

// ---------------------------------------------------------------------------
// Wave 1 Telemetry, Recovery, Async Prewarm, LOD, and Dimension FFI Entrypoints
// ---------------------------------------------------------------------------

/// Writes a UTF-8 string into `out_ptr` with a trailing NUL byte (`max_len` includes NUL),
/// preserving UTF-8 character boundaries. Returns the number of content bytes written (excluding NUL).
unsafe fn write_utf8_nul_terminated(out_ptr: u64, max_len: i32, text: &str) -> i32 {
    if out_ptr == 0 || max_len <= 0 {
        return 0;
    }
    let cap = max_len as usize;
    let max_content = cap.saturating_sub(1);
    let bytes = text.as_bytes();
    let mut copy_len = bytes.len().min(max_content);
    while copy_len > 0 && !text.is_char_boundary(copy_len) {
        copy_len -= 1;
    }
    unsafe {
        let dst = out_ptr as *mut u8;
        if copy_len > 0 {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, copy_len);
        }
        *dst.add(copy_len) = 0;
    }
    copy_len as i32
}

/// Queries the active VRAM usage, driver budget, and pressure level (`0=Normal`, `1=Elevated`, `2=Critical`).
/// Returns `0` on success or `-1` if the engine is not initialized.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_get_vram_budget(
    out_used_bytes_ptr: u64,
    out_budget_bytes_ptr: u64,
    out_pressure_ptr: u64,
) -> i32 {
    let Some(engine) = engine() else {
        return -1;
    };
    let cached = *engine.last_vram_budget.read();
    let snap = match cached {
        Some(s) => s,
        None => engine.poll_vram_budget_if_due(true),
    };
    let pressure_code: i32 = match snap.pressure {
        VramPressureLevel::Normal => 0,
        VramPressureLevel::Elevated => 1,
        VramPressureLevel::Critical => 2,
    };
    unsafe {
        if out_used_bytes_ptr != 0 {
            *(out_used_bytes_ptr as *mut u64) = snap.usage_bytes;
        }
        if out_budget_bytes_ptr != 0 {
            *(out_budget_bytes_ptr as *mut u64) = snap.budget_bytes;
        }
        if out_pressure_ptr != 0 {
            *(out_pressure_ptr as *mut i32) = pressure_code;
        }
    }
    0
}

/// Queries live Ray Tracing acceleration structure counts (resident built BLASes and active TLAS instances).
/// Returns `0` on success or `-1` if the engine is not initialized.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_get_rt_stats(
    out_blas_count_ptr: u64,
    out_tlas_instances_ptr: u64,
) -> i32 {
    let Some(engine) = engine() else {
        return -1;
    };
    let slot = engine.frame_slot();
    let blas = engine.blas_manager.lock();
    let terrain_blas = blas.slices.values().filter(|s| s.blas.is_some()).count();
    let lod_blas = blas
        .lod_blas
        .lod_slices
        .values()
        .filter(|s| s.blas.is_some())
        .count();
    let blas_count = terrain_blas.saturating_add(lod_blas).min(i32::MAX as usize) as i32;
    let tlas_instances = blas
        .instance_meta_count_for_slot(slot)
        .min(i32::MAX as usize) as i32;
    unsafe {
        if out_blas_count_ptr != 0 {
            *(out_blas_count_ptr as *mut i32) = blas_count;
        }
        if out_tlas_instances_ptr != 0 {
            *(out_tlas_instances_ptr as *mut i32) = tlas_instances;
        }
    }
    0
}

/// Enables (`enabled != 0`) or disables (`enabled == 0`) the ring-buffered GPU timestamp profiler.
/// Returns `0` on success or `-1` if the engine or timestamp profiler is not initialized.
/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_set_gpu_profiler_enabled(enabled: i32) -> i32 {
    let Some(engine) = engine() else {
        return -1;
    };
    let mut guard = engine.timestamp_profiler.lock();
    let Some(ref mut profiler) = *guard else {
        return -1;
    };
    profiler.set_enabled(enabled != 0);
    0
}

/// Writes the rolling mean GPU frame time after at least 30 completed timestamp samples.
/// Returns `1` when available, `0` while warming up or disabled, or `-1` for invalid input.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_get_average_gpu_frame_time_ms(out_ms_ptr: u64) -> i32 {
    if out_ms_ptr == 0 {
        return -1;
    }
    let Some(engine) = engine() else {
        return -1;
    };
    let guard = engine.timestamp_profiler.lock();
    let Some(ref profiler) = *guard else {
        return 0;
    };
    if !profiler.is_enabled() {
        return 0;
    }
    let Some(average_ms) = profiler.average_total_gpu_ms() else {
        return 0;
    };
    unsafe {
        (out_ms_ptr as *mut f64).write_unaligned(average_ms);
    }
    1
}

/// Writes the latest resolved GPU pass timings as NUL-terminated UTF-8 `"pass=0.42;pass2=1.15"`
/// into `out_ptr`. Returns bytes written, `0` if disabled/empty, or `-1` on error.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_get_gpu_pass_timings(out_ptr: u64, max_len: i32) -> i32 {
    if out_ptr == 0 || max_len <= 0 {
        return -1;
    }
    let Some(engine) = engine() else {
        return -1;
    };
    let guard = engine.timestamp_profiler.lock();
    let Some(ref profiler) = *guard else {
        return 0;
    };
    if !profiler.is_enabled() {
        return 0;
    }
    let timings = profiler.last_timings();
    if timings.is_empty() {
        return 0;
    }
    let mut formatted = String::new();
    for (idx, timing) in timings.iter().enumerate() {
        if idx > 0 {
            formatted.push(';');
        }
        let ms = if timing.duration_ms.is_finite() {
            timing.duration_ms.max(0.0)
        } else {
            0.0
        };
        let _ =
            std::fmt::Write::write_fmt(&mut formatted, format_args!("{}={:.2}", timing.name, ms));
    }
    unsafe { write_utf8_nul_terminated(out_ptr, max_len, &formatted) }
}

/// Reads the global driver-TDR / shader-failure recovery state. Writes `0`/`1` to `out_disabled_ptr`
/// and a NUL-terminated UTF-8 diagnostic notice to `out_notice_ptr`. Returns bytes written (`>= 0`).
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_get_recovery_state(
    out_disabled_ptr: u64,
    out_notice_ptr: u64,
    max_len: i32,
) -> i32 {
    let guard = crate::crash::global_device_lost_guard();
    let disabled = i32::from(guard.is_rt_vsp_disabled());
    if out_disabled_ptr != 0 {
        unsafe {
            *(out_disabled_ptr as *mut i32) = disabled;
        }
    }
    let notice = guard.format_ffi_notice().unwrap_or_default();
    if out_notice_ptr != 0 && max_len > 0 {
        unsafe { write_utf8_nul_terminated(out_notice_ptr, max_len, &notice) }
    } else {
        0
    }
}

/// Resets `global_device_lost_guard()` to re-enable RT/VSP passes after recovery or pack reload.
/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_reset_recovery_state() -> i32 {
    crate::crash::global_device_lost_guard().reset();
    0
}

/// Starts asynchronous shader pack prewarming on a background `"hyaloclast-prewarm"` thread.
/// Returns `0` on start or `-1` on invalid arguments / uninitialized engine.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_start_pack_prewarm(pack_path_ptr: u64, pack_path_len: i32) -> i32 {
    if pack_path_ptr == 0 || pack_path_len <= 0 {
        return -1;
    }
    let Some(engine) = engine() else {
        return -1;
    };
    let bytes =
        unsafe { slice::from_raw_parts(pack_path_ptr as *const u8, pack_path_len as usize) };
    let Ok(path_str) = str::from_utf8(bytes) else {
        return -1;
    };
    if path_str.is_empty() {
        return -1;
    }
    let requested_tier = engine.render_graph.lock().config.tier;
    let (tier, hw_caps) = {
        let ctx = engine.ctx.read();
        let tier = ctx.platform_tier_policy().clamp_render_tier(requested_tier);
        let hw_caps = vsp_compiler::preprocessor::HardwareShaderCaps {
            ser_ext: crate::gpu::ser_reordering_enabled()
                && ctx.capabilities.supports_ser_ext
                && ctx.capabilities.ser_reordering_active,
            ser_nv: crate::gpu::ser_reordering_enabled()
                && ctx.capabilities.supports_ser_nv
                && ctx.capabilities.ser_reordering_active,
            ada_or_newer: ctx.capabilities.gpu_arch.is_ada_or_newer(),
            high_end_gpu: ctx.capabilities.gpu_tier_class.is_high_end(),
            opacity_micromap: ctx.capabilities.supports_opacity_micromap,
        };
        (tier, hw_caps)
    };
    let cache_dir = Some(resolve_game_path("config/vulkanshaders/cache/spirv"));
    let prewarmer = AsyncPipelinePrewarmer::start_pack_prewarm(
        std::path::PathBuf::from(path_str),
        tier,
        hw_caps,
        cache_dir,
    );
    let prev = engine.active_prewarmer.lock().replace(prewarmer);
    drop(prev);
    0
}

/// Polls progress of the active background pack prewarmer.
/// Returns `0` on success or `-1` if no prewarmer is active (or engine is uninitialized).
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_get_prewarm_progress(
    out_completed_ptr: u64,
    out_total_ptr: u64,
    out_ready_ptr: u64,
) -> i32 {
    let Some(engine) = engine() else {
        return -1;
    };
    let guard = engine.active_prewarmer.lock();
    let Some(ref prewarmer) = *guard else {
        return -1;
    };
    let progress = prewarmer.progress();
    unsafe {
        if out_completed_ptr != 0 {
            *(out_completed_ptr as *mut i32) = progress.completed.min(i32::MAX as usize) as i32;
        }
        if out_total_ptr != 0 {
            *(out_total_ptr as *mut i32) = progress.total.min(i32::MAX as usize) as i32;
        }
        if out_ready_ptr != 0 {
            *(out_ready_ptr as *mut i32) = i32::from(progress.ready);
        }
    }
    0
}

/// Enables (`enabled != 0`) or disables (`enabled == 0`) Distant Horizons LOD BLAS injection,
/// cleanly retiring resident LOD BLASes when disabled. Returns `0` or `-1` if uninitialized.
/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_set_lod_enabled(enabled: i32) -> i32 {
    let Some(engine) = engine() else {
        return -1;
    };
    let ctx = engine.ctx.read();
    engine
        .blas_manager
        .lock()
        .set_lod_enabled(enabled != 0, Some(&ctx), Some(&engine.retire));
    0
}

/// Sets the active Minecraft dimension (`0` = Overworld, `1` = Nether, `2` = End) for
/// dimension-aware emissive scaling and resets temporal history if the dimension changed.
/// Returns `0` on success or `-1` if uninitialized.
/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_set_dimension(dimension_id: i32) -> i32 {
    let Some(engine) = engine() else {
        return -1;
    };
    let changed = {
        let mut blas = engine.blas_manager.lock();
        let prev = blas.emissive_lights.active_dimension;
        blas.set_dimension(dimension_id);
        blas.emissive_lights.active_dimension != prev
    };
    if changed {
        engine.reset_temporal_history();
        let mut rfs = RUNTIME_FRAME_STATE.lock();
        rfs.history_reset_pending = true;
        rfs.fg_history_reset_pending = true;
    }
    0
}

/// Attaches pre-encoded Opacity Micromap (`VK_EXT_opacity_micromap`) bit data to a cutout terrain
/// slice (`layer_ordinal == 1`) so its next BLAS build binds a hardware `VkMicromapEXT`.
/// Returns `0` if attached, or `-1` on invalid arguments / uninitialized engine / missing slice.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_attach_cutout_omm(
    allocation_key_id: u64,
    layer_ordinal: u32,
    triangle_count: u32,
    subdiv_level: u32,
    use_4state: i32,
    encoded_data_ptr: u64,
    encoded_data_len: i32,
) -> i32 {
    if allocation_key_id == 0
        || triangle_count == 0
        || subdiv_level > MAX_OMM_SUBDIV_LEVEL
        || encoded_data_ptr == 0
        || encoded_data_len <= 0
    {
        return -1;
    }
    let Some(engine) = engine() else {
        return -1;
    };
    let format = if use_4state != 0 {
        OmmFormat::FourState
    } else {
        OmmFormat::TwoState
    };
    let per_tri_bytes = encoded_byte_size(subdiv_level, format);
    if per_tri_bytes == 0 {
        return -1;
    }
    let Some(required_bytes) = (triangle_count as usize).checked_mul(per_tri_bytes) else {
        return -1;
    };
    if (encoded_data_len as usize) < required_bytes {
        return -1;
    }
    let raw = unsafe { slice::from_raw_parts(encoded_data_ptr as *const u8, required_bytes) };
    let mut triangles = Vec::with_capacity(triangle_count as usize);
    for tri_bytes in raw.chunks_exact(per_tri_bytes) {
        triangles.push(OmmTriangleEncoding::Micromap {
            subdiv_level: subdiv_level as u16,
            format,
            data: tri_bytes.to_vec(),
        });
    }
    let input = CutoutOmmBuildInput { triangles };
    let attached =
        engine
            .blas_manager
            .lock()
            .attach_cutout_omm_data(allocation_key_id, layer_ordinal, input);
    if attached { 0 } else { -1 }
}

/// Submits a categorized dynamic entity geometry batch (mobs, players, armor/elytra overlays,
/// block entities, particles, weather, held items, first-person hand, or crumbling overlay)
/// for inclusion in the per-frame dynamic Entity BLAS.
/// # Safety
/// The caller must pass live Vulkan handles and valid, correctly sized storage for each address argument.
/// Storage must remain valid until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_submit_entity_vertices_categorized(
    vertex_data_ptr: u64,
    vertex_count: i32,
    vertex_stride: i32,
    is_quads: i32,
    texture_id: i32,
    normal_tex_id: i32,
    specular_tex_id: i32,
    kind: i32,
    category: u32,
    entity_flags: u32,
) {
    if vertex_data_ptr == 0 || vertex_count <= 0 || vertex_stride < 24 {
        return;
    }
    let Some(engine) = engine() else {
        return;
    };
    let Some(total_bytes) = (vertex_count as usize).checked_mul(vertex_stride as usize) else {
        return;
    };
    let bytes = unsafe { slice::from_raw_parts(vertex_data_ptr as *const u8, total_bytes) };
    let geom_kind = match kind {
        0 => EntityGeometryKind::World,
        1 => EntityGeometryKind::Crumbling,
        2 => EntityGeometryKind::FirstPersonAvatar,
        _ => return,
    };
    let entity_cat = EntityCategory::from_u32(category);
    let albedo_slot = texture_id.max(0) as u32;
    let (normal_slot, specular_slot) = if normal_tex_id > 0 || specular_tex_id > 0 {
        (normal_tex_id.max(0) as u32, specular_tex_id.max(0) as u32)
    } else if albedo_slot > 0 {
        engine.bindless.read().companions_for_albedo(albedo_slot)
    } else {
        (0, 0)
    };
    engine
        .blas_manager
        .lock()
        .entity_blas
        .append_categorized_geometry_batch(
            bytes,
            vertex_count as u32,
            vertex_stride as u32,
            is_quads != 0,
            albedo_slot,
            normal_slot,
            specular_slot,
            geom_kind,
            entity_cat,
            entity_flags,
        );
}

/// Handles display/swapchain resize events by resetting temporal history across all subsystems
/// and retiring resize-sensitive internal render-graph targets cleanly.
/// Returns `0` on success or `-1` if the engine is not initialized.
/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_on_display_resized() -> i32 {
    let Some(engine) = engine() else {
        return -1;
    };
    engine.reset_temporal_history();
    {
        let mut rfs = RUNTIME_FRAME_STATE.lock();
        rfs.history_reset_pending = true;
        rfs.fg_history_reset_pending = true;
    }
    engine
        .pipelines_and_compiler
        .lock()
        .pipelines
        .descriptor_sig_cache
        .clear();
    {
        let ctx = engine.ctx.read();
        engine
            .render_graph
            .lock()
            .invalidate_for_resize(&ctx, &engine.retire);
    }
    0
}

/// Handles Minecraft `F3+T` resource reload events by resetting temporal history, clearing
/// cached descriptor signatures, and marking cutout Opacity Micromap slices dirty for rebuild.
/// Returns `0` on success or `-1` if the engine is not initialized.
/// # Safety
/// Calls must follow this entry point’s Vulkan engine lifecycle and must not race core initialization or shutdown.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vs_core_on_resources_reloaded() -> i32 {
    let Some(engine) = engine() else {
        return -1;
    };
    engine.reset_temporal_history();
    {
        let mut rfs = RUNTIME_FRAME_STATE.lock();
        rfs.history_reset_pending = true;
        rfs.fg_history_reset_pending = true;
    }
    engine
        .pipelines_and_compiler
        .lock()
        .pipelines
        .descriptor_sig_cache
        .clear();
    {
        let mut blas = engine.blas_manager.lock();
        let dirty_keys: Vec<_> = blas
            .slices
            .iter_mut()
            .filter_map(|(&key, slice)| {
                if key.layer_ordinal == 1
                    && (slice.omm.is_some() || slice.pending_omm_input.is_some())
                {
                    slice.pending_omm_input = None;
                    slice.dirty_blas_build = true;
                    Some(key)
                } else {
                    None
                }
            })
            .collect();
        for key in dirty_keys {
            if !blas.pending_build_queue.contains(&key) {
                blas.pending_build_queue.push_back(key);
            }
        }
    }
    0
}
