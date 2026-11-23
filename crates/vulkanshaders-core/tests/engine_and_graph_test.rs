use vsp_compiler::{PackManifest, RenderTier};
use vulkanshaders_core::{
    graph::{GraphConfig, PassKind, RenderGraph, ResourceAccess},
    modules::{CsmShadowUBO, DepthReprojectPushConstants, GpuPipelineManager},
    rt::{SectionLayerKey, SectionSliceDescriptor},
};

#[test]
fn tier3_render_graph_has_no_duplicate_hudless_color_writes_and_preserves_temporal_resources() {
    let mut graph = RenderGraph::default();
    graph.rebuild(GraphConfig {
        tier: RenderTier::Tier3PathTracing,
        denoiser_mode: 1,
        upscaler_mode: 1,
        framegen_mode: 1,
        hdr_enabled: false,
        overlay_pass_enabled: true,
    });

    assert!(
        graph
            .compiled_steps
            .iter()
            .any(|s| s.kind == PassKind::FullPathTracing)
    );
    assert!(
        graph
            .compiled_steps
            .iter()
            .any(|s| s.kind == PassKind::DepthReprojectOverlay)
    );
    assert!(
        !graph
            .compiled_steps
            .iter()
            .any(|s| s.kind == PassKind::GbufferMrt)
    );
    assert!(RenderGraph::is_temporal_resource(
        RenderGraph::RES_PREV_HUDLESS_COLOR
    ));
    assert!(RenderGraph::is_temporal_resource(
        RenderGraph::RES_GBUFFER_VELOCITY
    ));
    assert!(RenderGraph::is_temporal_resource(
        RenderGraph::RES_SHARC_HASH_TABLE
    ));
}

#[test]
fn hdr_mode_disables_sdr_tonemap_and_feeds_linear_hdr_into_hdr_compose() {
    let mut graph = RenderGraph::default();
    let cfg = GraphConfig {
        tier: RenderTier::Tier1ComputeGi,
        denoiser_mode: 1,
        upscaler_mode: 1,
        framegen_mode: 1,
        hdr_enabled: true,
        overlay_pass_enabled: true,
    };
    graph.rebuild(cfg);

    assert!(
        !graph
            .compiled_steps
            .iter()
            .any(|s| s.kind == PassKind::PostGradeAndToneMap),
        "SDR PostGradeAndToneMap must be disabled when HDR is enabled"
    );
    assert!(
        graph
            .compiled_steps
            .iter()
            .any(|s| s.kind == PassKind::HdrCompose)
    );
    assert!(
        graph
            .compiled_steps
            .iter()
            .any(|s| s.kind == PassKind::FrameGenInterpolate),
        "HDR and FrameGen must be able to run together"
    );
    assert_eq!(
        GpuPipelineManager::resolve_resource_for_binding(PassKind::HdrCompose, 0, false, &cfg),
        RenderGraph::RES_UPSCALED_HDR
    );
}

#[test]
fn csm_shadow_ubo_uses_column_major_matrices_and_culls_out_of_frustum_sections() {
    let ubo = CsmShadowUBO::from_sun_dir([0.0, 1.0, 0.0], 2048.0);
    // In column-major layout, column 3 (`mat[3]`) holds the translation + w=1.0 component,
    // and rows 0..2 of columns 0..2 hold the basis axes (`mat[col][3] == 0.0` for col 0..2).
    for mat in &ubo.cascade_view_proj {
        assert_eq!(mat[0][3], 0.0);
        assert_eq!(mat[1][3], 0.0);
        assert_eq!(mat[2][3], 0.0);
        assert_eq!(mat[3][3], 1.0);
        // Origin (0,0,0) maps to z_clip == 0.5 inside [0, 1]
        assert!((mat[3][2] - 0.5).abs() < 1.0e-5);
    }
    for &t in &ubo.texel_sizes {
        assert!(t > 0.0 && t < 0.01);
    }
    // Section near origin is inside Cascade 0 (32-block extent)
    assert!(ubo.is_aabb_visible_in_cascade(0, [-8.0, -8.0, -8.0], [8.0, 8.0, 8.0]));
    // Section 400 blocks away in X is culled from Cascade 0 (32-block extent)
    assert!(!ubo.is_aabb_visible_in_cascade(0, [400.0, 0.0, 0.0], [416.0, 16.0, 16.0]));
}

#[test]
fn vsp_graph_v2_splices_custom_passes_and_resolves_named_spirv_bindings() {
    let toml_str = r#"
[pack]
id = "custom-dag-pack"
name = "Custom DAG Pack"
version = "0.2.0"

[capabilities]
tiers = [0, 1, 2, 3]

[pipeline]
mode = "extend"

[targets.custom_bloom]
format = "rgba16f"
scale = 0.5
temporal = true

[[passes]]
id = "custom_bloom_pass"
kind = "compute"
shader = "shaders/compute/bloom.comp"
min_tier = 0
max_tier = 3
before = "post_grade_and_tonemap"
reads = ["raw_radiance"]
writes = ["custom_bloom"]
scale = 0.5
bindings = { b1 = "raw_radiance", b2 = "custom_bloom" }

[[passes]]
id = "custom_post_composite"
kind = "compute"
shader = "shaders/compute/post_tonemap.comp"
min_tier = 0
max_tier = 3
replaces = "post_grade_and_tonemap"
reads = ["raw_radiance", "custom_bloom"]
writes = ["hudless_color", "mc_main_depth"]
"#;
    let manifest = PackManifest::parse_and_validate(toml_str).expect("parse custom DAG manifest");
    let mut graph = RenderGraph::default();
    graph.register_pack_manifest_graph(&manifest);

    // Verify `custom_bloom_pass` is live (because `custom_post_composite` reads `custom_bloom` and writes `hudless_color`)
    assert!(
        graph
            .compiled_steps
            .iter()
            .any(|s| s.name == "custom_bloom_pass"),
        "custom_bloom_pass must be kept live by backward-BFS reachability"
    );
    let bloom_id = graph
        .resolve_resource_by_name("custom_bloom")
        .expect("resolve custom_bloom target");
    assert_eq!(
        GpuPipelineManager::resolve_resource_for_binding_named(
            PassKind::Custom(1),
            2,
            "uCustomBloom",
            true,
            &graph,
        ),
        bloom_id
    );
}

#[test]
fn builtin_froxel_fog_uses_temporal_array_on_every_render_tier() {
    let manifest = PackManifest::parse_and_validate(include_str!(
        "../../../src/main/resources/builtin-packs/hyaloclast-sunlit-vanilla.vsp/pack.toml"
    ))
    .expect("parse built-in pack manifest");
    let mut graph = RenderGraph::default();
    graph.register_pack_manifest_graph(&manifest);

    let fog_target = graph
        .custom_targets
        .get(&RenderGraph::RES_VOLUMETRIC_FOG)
        .expect("register volumetric fog target");
    assert_eq!(fog_target.scale, 0.125);
    assert_eq!(fog_target.array_layers, 16);
    assert!(fog_target.temporal);
    assert_eq!(
        fog_target.history_id,
        Some(RenderGraph::RES_VOLUMETRIC_FOG_HISTORY)
    );

    assert_eq!(
        GpuPipelineManager::resolve_resource_for_binding(
            PassKind::ComputeVolumetricFog,
            1,
            false,
            &graph.config,
        ),
        RenderGraph::RES_VOLUMETRIC_FOG_HISTORY
    );
    assert_eq!(
        GpuPipelineManager::resolve_resource_for_binding(
            PassKind::FullPathTracing,
            8,
            false,
            &graph.config,
        ),
        RenderGraph::RES_VOLUMETRIC_FOG
    );

    let gi_target = graph
        .custom_targets
        .values()
        .find(|target| target.name == "restir_gi_reservoir")
        .expect("register ReSTIR GI reservoir target");
    assert_eq!(gi_target.scale, 0.25);
    assert_eq!(gi_target.array_layers, 3);
    assert!(gi_target.temporal);
    let gi_target_id = gi_target.id;
    let gi_history_id = gi_target.history_id.expect("register GI history target");
    assert_eq!(
        GpuPipelineManager::resolve_resource_for_binding_named(
            PassKind::FullPathTracing,
            9,
            "uRestirGiReservoirHistory",
            false,
            &graph,
        ),
        gi_history_id
    );
    assert_eq!(
        GpuPipelineManager::resolve_resource_for_binding_named(
            PassKind::FullPathTracing,
            10,
            "uRestirGiReservoir",
            true,
            &graph,
        ),
        gi_target_id
    );

    graph.rebuild(GraphConfig {
        tier: RenderTier::Tier3PathTracing,
        ..GraphConfig::default()
    });
    let pathtrace = graph
        .compiled_steps
        .iter()
        .find(|step| step.kind == PassKind::FullPathTracing)
        .expect("compile path tracing pass");
    assert!(
        pathtrace
            .reads
            .contains(&(gi_history_id, ResourceAccess::SampledRead))
    );
    assert!(
        pathtrace
            .writes
            .contains(&(gi_target_id, ResourceAccess::StorageImageWrite))
    );
}

#[test]
fn region_shifted_slice_tlas_transform_matches_64_block_region_origin() {
    let sec_x: i64 = -3;
    let sec_y: i64 = 5;
    let sec_z: i64 = 7;
    let packed = ((sec_x & 0x3F_FFFF) << 42) | ((sec_z & 0x3F_FFFF) << 20) | (sec_y & 0xF_FFFF);

    let slice = SectionSliceDescriptor {
        key: SectionLayerKey {
            allocation_key_id: 1,
            layer_ordinal: 1,
        },
        section_pos_packed: packed,
        vk_buffer: 0,
        byte_offset: 0,
        byte_size: 1600,
        vertex_stride: 40,
        vertex_count: 40,
        is_region_shifted: true,
        has_foliage_wind: true,
        displaced_vertex_buf: None,
        blas: None,
        omm: None,
        pending_omm_input: None,
        dirty_blas_build: true,
        bindless_albedo_slot: 0,
        bindless_normal_slot: 0,
        bindless_specular_slot: 0,
        material_flags: 0,
        cached_vertex_bda: 0,
        compacted: false,
    };

    let (ax, ay, az) = slice.world_anchor_blocks();
    assert_eq!(ax, (-3 * 16) & !63);
    assert_eq!(ay, (5 * 16) & !63);
    assert_eq!(az, (7 * 16) & !63);
}

#[test]
fn evaluates_overlay_depth_reprojection_monotonically() {
    let pc = DepthReprojectPushConstants {
        z_scale: -1.0002,
        z_offset: -0.1000,
        w_scale: -1.0,
        w_offset: 0.0,
        source_size: [1920, 1080],
        target_size: [1920, 1080],
    };
    let near_d = pc.evaluate_device_depth(2.0);
    let far_d = pc.evaluate_device_depth(128.0);
    assert!(near_d < far_d);
    assert!((0.0..=1.0).contains(&near_d));
    assert!((0.0..=1.0).contains(&far_d));
}

#[test]
fn csm_shadow_ubo_with_config_scales_cascades_and_bindless_slot_lookup_is_o1() {
    let ubo_12ch = CsmShadowUBO::from_sun_dir_with_config([0.0, 1.0, 0.0], 4096.0, 192.0);
    assert!((ubo_12ch.split_distances[3] - 192.0).abs() < 1.0e-4);
    assert!((ubo_12ch.texel_sizes[0] - (1.0 / 4096.0)).abs() < 1.0e-7);

    let mut table = vulkanshaders_core::gpu::bindless::BindlessTextureTable::new(64);
    let slot = table
        .register_or_update_with_sampler(
            0x1111,
            0x2222,
            0x3333,
            ash::vk::Format::R8G8B8A8_UNORM,
            1024,
            1024,
            4,
            vulkanshaders_core::gpu::bindless::TexturePbrRole::Albedo,
            None,
        )
        .expect("register bindless slot");
    let (view, sampler) = table.view_sampler_for_slot(slot).expect("O(1) slot lookup");
    assert_eq!(ash::vk::Handle::as_raw(view), 0x2222);
    assert_eq!(ash::vk::Handle::as_raw(sampler), 0x3333);
    table.unregister(0x1111);
    assert!(table.view_sampler_for_slot(slot).is_none());
}

#[test]
fn gpu_architecture_classification_covers_nvidia_amd_and_intel() {
    use vulkanshaders_core::gpu::GpuArchitecture;

    // NVIDIA PCI device IDs
    assert_eq!(
        GpuArchitecture::classify(0x10DE, 0x1E04, "NVIDIA GeForce RTX 2080 Ti"),
        GpuArchitecture::NvidiaTuring
    );
    assert_eq!(
        GpuArchitecture::classify(0x10DE, 0x2204, "NVIDIA GeForce RTX 3090"),
        GpuArchitecture::NvidiaAmpere
    );
    assert_eq!(
        GpuArchitecture::classify(0x10DE, 0x2684, "NVIDIA GeForce RTX 4090"),
        GpuArchitecture::NvidiaAda
    );
    assert_eq!(
        GpuArchitecture::classify(0x10DE, 0x2B85, "NVIDIA GeForce RTX 5090"),
        GpuArchitecture::NvidiaBlackwell
    );
    // NVIDIA name fallback when PCI device_id is unknown/zero
    assert_eq!(
        GpuArchitecture::classify(0x10DE, 0, "NVIDIA GeForce RTX 5080"),
        GpuArchitecture::NvidiaBlackwell
    );
    assert_eq!(
        GpuArchitecture::classify(0x10DE, 0, "NVIDIA GeForce RTX 4070 SUPER"),
        GpuArchitecture::NvidiaAda
    );
    assert_eq!(
        GpuArchitecture::classify(0x10DE, 0, "NVIDIA GeForce RTX 3080"),
        GpuArchitecture::NvidiaAmpere
    );
    assert_eq!(
        GpuArchitecture::classify(0x10DE, 0, "NVIDIA GeForce RTX 2070"),
        GpuArchitecture::NvidiaTuring
    );

    assert!(GpuArchitecture::NvidiaAda.is_ada_or_newer());
    assert!(GpuArchitecture::NvidiaBlackwell.is_ada_or_newer());
    assert!(!GpuArchitecture::NvidiaAmpere.is_ada_or_newer());
    assert!(!GpuArchitecture::NvidiaTuring.is_ada_or_newer());
    assert_eq!(
        GpuArchitecture::NvidiaBlackwell.as_str(),
        "nvidia_blackwell"
    );
    assert_eq!(GpuArchitecture::NvidiaAda.as_str(), "nvidia_ada");

    // AMD PCI device IDs & name fallbacks
    assert_eq!(
        GpuArchitecture::classify(0x1002, 0x73BF, "AMD Radeon RX 6900 XT"),
        GpuArchitecture::AmdRdna2
    );
    assert_eq!(
        GpuArchitecture::classify(0x1002, 0x744C, "AMD Radeon RX 7900 XTX"),
        GpuArchitecture::AmdRdna3
    );
    assert_eq!(
        GpuArchitecture::classify(0x1002, 0x7550, "AMD Radeon RX 9070 XT"),
        GpuArchitecture::AmdRdna4
    );
    assert_eq!(
        GpuArchitecture::classify(0x1002, 0, "AMD Radeon RX 9070"),
        GpuArchitecture::AmdRdna4
    );

    // Intel PCI device IDs & name fallbacks
    assert_eq!(
        GpuArchitecture::classify(0x8086, 0x56A0, "Intel(R) Arc(TM) A770 Graphics"),
        GpuArchitecture::IntelAlchemist
    );
    assert_eq!(
        GpuArchitecture::classify(0x8086, 0xE20B, "Intel(R) Arc(TM) B580 Graphics"),
        GpuArchitecture::IntelBattlemage
    );
    assert_eq!(
        GpuArchitecture::classify(0x8086, 0, "Intel Arc B570"),
        GpuArchitecture::IntelBattlemage
    );
}

#[test]
fn gpu_tier_class_classification_scales_by_arch_vram_and_device_type() {
    use ash::vk::PhysicalDeviceType;
    use vulkanshaders_core::gpu::{GpuArchitecture, GpuTierClass};

    const GIB: u64 = 1024 * 1024 * 1024;

    assert_eq!(
        GpuTierClass::classify(
            GpuArchitecture::NvidiaBlackwell,
            PhysicalDeviceType::DISCRETE_GPU,
            32 * GIB,
            "NVIDIA GeForce RTX 5090",
        ),
        GpuTierClass::Enthusiast
    );
    assert_eq!(
        GpuTierClass::classify(
            GpuArchitecture::NvidiaAda,
            PhysicalDeviceType::DISCRETE_GPU,
            24 * GIB,
            "NVIDIA GeForce RTX 4090",
        ),
        GpuTierClass::Enthusiast
    );
    assert_eq!(
        GpuTierClass::classify(
            GpuArchitecture::AmdRdna4,
            PhysicalDeviceType::DISCRETE_GPU,
            16 * GIB,
            "AMD Radeon RX 9070 XT",
        ),
        GpuTierClass::Enthusiast
    );
    assert_eq!(
        GpuTierClass::classify(
            GpuArchitecture::NvidiaAda,
            PhysicalDeviceType::DISCRETE_GPU,
            12 * GIB,
            "NVIDIA GeForce RTX 4070",
        ),
        GpuTierClass::High
    );
    assert_eq!(
        GpuTierClass::classify(
            GpuArchitecture::NvidiaAmpere,
            PhysicalDeviceType::DISCRETE_GPU,
            10 * GIB,
            "NVIDIA GeForce RTX 3080",
        ),
        GpuTierClass::High
    );
    assert_eq!(
        GpuTierClass::classify(
            GpuArchitecture::NvidiaAda,
            PhysicalDeviceType::DISCRETE_GPU,
            8 * GIB,
            "NVIDIA GeForce RTX 4060",
        ),
        GpuTierClass::Mid
    );
    assert_eq!(
        GpuTierClass::classify(
            GpuArchitecture::NvidiaAmpere,
            PhysicalDeviceType::DISCRETE_GPU,
            6 * GIB,
            "NVIDIA GeForce RTX 3050",
        ),
        GpuTierClass::Low
    );
    assert_eq!(
        GpuTierClass::classify(
            GpuArchitecture::AmdRdna3,
            PhysicalDeviceType::INTEGRATED_GPU,
            16 * GIB,
            "AMD Radeon 780M",
        ),
        GpuTierClass::Low
    );

    assert!(GpuTierClass::Enthusiast.is_high_end());
    assert!(GpuTierClass::High.is_high_end());
    assert!(!GpuTierClass::Mid.is_high_end());
    assert!(!GpuTierClass::Low.is_high_end());
    assert_eq!(GpuTierClass::Enthusiast.as_str(), "enthusiast");
}

#[test]
fn ser_activation_requires_extension_and_hardware_reorder_hint() {
    use vulkanshaders_core::gpu::{
        GpuArchitecture,
        device_merger::{VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER, evaluate_ser_activation},
    };

    // Hardware reorder enabled on Ada/Blackwell via NV or EXT
    assert!(evaluate_ser_activation(
        GpuArchitecture::NvidiaAda,
        true,
        VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER,
        false,
        0,
    ));
    assert!(evaluate_ser_activation(
        GpuArchitecture::NvidiaBlackwell,
        false,
        0,
        true,
        VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER,
    ));
    // Emulated-only SER (hint == 0 / NONE) must NOT activate SER reordering
    assert!(!evaluate_ser_activation(
        GpuArchitecture::NvidiaAmpere,
        true,
        0,
        false,
        0,
    ));
    // Extension absent even if hint were non-zero
    assert!(!evaluate_ser_activation(
        GpuArchitecture::NvidiaAda,
        false,
        VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER,
        false,
        VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER,
    ));
}

#[test]
fn high_end_gpu_budgets_and_shader_caps_scale_by_tier() {
    use vsp_compiler::preprocessor::HardwareShaderCaps;
    use vulkanshaders_core::{
        gpu::{DeviceCapabilities, GpuTierClass},
        rt::{
            EmissiveLightClusterManager, LodBlasManager, SharcRuntime, ZeroCopyBlasManager,
            wind::WindRefitScheduler,
        },
    };

    // BLAS builds/frame
    assert_eq!(
        ZeroCopyBlasManager::default_blas_builds_for_tier(GpuTierClass::Low),
        32
    );
    assert_eq!(
        ZeroCopyBlasManager::default_blas_builds_for_tier(GpuTierClass::Mid),
        32
    );
    assert_eq!(
        ZeroCopyBlasManager::default_blas_builds_for_tier(GpuTierClass::High),
        64
    );
    assert_eq!(
        ZeroCopyBlasManager::default_blas_builds_for_tier(GpuTierClass::Enthusiast),
        96
    );

    // LOD BLAS builds/frame
    assert_eq!(
        LodBlasManager::default_builds_per_frame_for_tier(GpuTierClass::Low),
        16
    );
    assert_eq!(
        LodBlasManager::default_builds_per_frame_for_tier(GpuTierClass::High),
        32
    );
    assert_eq!(
        LodBlasManager::default_builds_per_frame_for_tier(GpuTierClass::Enthusiast),
        48
    );

    // Wind refit sections & reach radius
    assert_eq!(
        WindRefitScheduler::budget_for_tier(GpuTierClass::Mid),
        (24, 48.0)
    );
    assert_eq!(
        WindRefitScheduler::budget_for_tier(GpuTierClass::High),
        (48, 72.0)
    );
    assert_eq!(
        WindRefitScheduler::budget_for_tier(GpuTierClass::Enthusiast),
        (64, 96.0)
    );

    // Emissive light culling margin & ReSTIR reservoir resolution
    assert_eq!(
        EmissiveLightClusterManager::culling_margin_for_tier(GpuTierClass::Low),
        96.0
    );
    assert_eq!(
        EmissiveLightClusterManager::culling_margin_for_tier(GpuTierClass::High),
        144.0
    );
    assert_eq!(
        EmissiveLightClusterManager::culling_margin_for_tier(GpuTierClass::Enthusiast),
        192.0
    );
    assert_eq!(
        EmissiveLightClusterManager::reservoir_dimensions_for_tier(GpuTierClass::Mid, 1920, 1080),
        (960, 540)
    );
    assert_eq!(
        EmissiveLightClusterManager::reservoir_dimensions_for_tier(GpuTierClass::High, 2560, 1440),
        (2560, 1440)
    );
    assert_eq!(
        EmissiveLightClusterManager::reservoir_dimensions_for_tier(
            GpuTierClass::Enthusiast,
            3840,
            2160
        ),
        (3840, 2160)
    );

    // SHaRC update downsample factor
    assert_eq!(
        SharcRuntime::downsample_factor_for_tier(GpuTierClass::Low),
        4
    );
    assert_eq!(
        SharcRuntime::downsample_factor_for_tier(GpuTierClass::High),
        3
    );
    assert_eq!(
        SharcRuntime::downsample_factor_for_tier(GpuTierClass::Enthusiast),
        2
    );

    // Shadow map resolution cap
    let low_caps = DeviceCapabilities {
        gpu_tier_class: GpuTierClass::Mid,
        ..DeviceCapabilities::default()
    };
    let high_caps = DeviceCapabilities {
        gpu_tier_class: GpuTierClass::Enthusiast,
        ..DeviceCapabilities::default()
    };
    assert_eq!(low_caps.max_shadow_resolution(), 4096);
    assert_eq!(high_caps.max_shadow_resolution(), 8192);

    // HardwareShaderCaps defines and cache key bits
    let ada_caps = HardwareShaderCaps {
        ser_ext: false,
        ser_nv: true,
        ada_or_newer: true,
        high_end_gpu: true,
        opacity_micromap: true,
    };
    let defines = ada_caps.to_glsl_defines();
    assert!(!defines.contains("VS_HAS_SER_EXT"));
    assert!(defines.contains("#define VS_HAS_SER_NV 1"));
    assert!(defines.contains("#define VS_GPU_ADA_PLUS 1"));
    assert!(defines.contains("#define VS_HIGH_END_GPU 1"));
    assert!(defines.contains("#define VS_HAS_OPACITY_MICROMAP 1"));
    assert_ne!(
        ada_caps.cache_bits(),
        HardwareShaderCaps::default().cache_bits()
    );
}

#[test]
fn csm_cascades_cover_split_distances_tall_casters_and_texel_snap_world_points() {
    let sun_dir = [0.42f32, 0.84, 0.34];
    let shadow_res = 2048.0f32;
    let max_dist = 192.0f32; // Default 12-chunk shadow distance (dist_scale = 0.5)
    let ubo = CsmShadowUBO::from_sun_dir_with_config(sun_dir, shadow_res, max_dist);

    // Every cascade must cover points at its split distance in X/Z without clipping out of [-1, 1].
    for (c, &split) in ubo.split_distances.iter().enumerate() {
        assert!(
            ubo.is_aabb_visible_in_cascade(c, [split - 1.0, -1.0, -1.0], [split + 1.0, 1.0, 1.0]),
            "Cascade {c} must cover its split distance {split}"
        );
    }

    // A caster upstream of the receiver must fit in Cascade 0's hardware depth range.
    let toward_sun = sun_dir.map(|axis| axis * (200.0 / sun_dir[1]));
    let near = toward_sun.map(|axis| axis - 4.0);
    let far = toward_sun.map(|axis| axis + 4.0);
    assert!(ubo.is_aabb_visible_in_cascade(0, near, far));
    let m = ubo.cascade_view_proj[0];
    for x in [near[0], far[0]] {
        for y in [near[1], far[1]] {
            for z in [near[2], far[2]] {
                let depth = m[0][2] * x + m[1][2] * y + m[2][2] * z + m[3][2];
                assert!((0.0..=1.0).contains(&depth), "Caster depth {depth} clips");
            }
        }
    }

    // Moving camera smoothly must keep a fixed world point aligned to an integer texel delta.
    let cam_a = [100.13f64, 64.0, -42.71];
    let cam_b = [100.41f64, 64.0, -42.39];
    let ubo_a = CsmShadowUBO::from_sun_dir_snapped(sun_dir, shadow_res, max_dist, cam_a);
    let ubo_b = CsmShadowUBO::from_sun_dir_snapped(sun_dir, shadow_res, max_dist, cam_b);
    let world_pt = [105.0f64, 68.0, -39.0];
    for c in 0..4 {
        let project_uv = |ubo: &CsmShadowUBO, cam: [f64; 3]| -> (f64, f64) {
            let p = [
                (world_pt[0] - cam[0]) as f32,
                (world_pt[1] - cam[1]) as f32,
                (world_pt[2] - cam[2]) as f32,
            ];
            let m = ubo.cascade_view_proj[c];
            let cx = m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0];
            let cy = m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1];
            (
                (f64::from(cx) * 0.5 + 0.5) * f64::from(shadow_res),
                (f64::from(cy) * 0.5 + 0.5) * f64::from(shadow_res),
            )
        };
        let (ux_a, uy_a) = project_uv(&ubo_a, cam_a);
        let (ux_b, uy_b) = project_uv(&ubo_b, cam_b);
        let dx = (ux_b - ux_a) - (ux_b - ux_a).round();
        let dy = (uy_b - uy_a) - (uy_b - uy_a).round();
        assert!(
            dx.abs() < 1e-2 && dy.abs() < 1e-2,
            "Cascade {c} shadow texel snapped delta should be near integer (dx={dx}, dy={dy})"
        );
    }
}

#[test]
fn vram_pressure_policy_classifies_normal_elevated_and_critical_budgets() {
    use vulkanshaders_core::gpu::{VramBudgetSnapshot, VramPressureLevel, VramPressurePolicy};

    const GIB: u64 = 1024 * 1024 * 1024;

    let normal = VramBudgetSnapshot::from_usage_and_budget(4 * GIB, 8 * GIB, true);
    assert_eq!(normal.pressure, VramPressureLevel::Normal);
    assert!((normal.usage_ratio - 0.5).abs() < 1e-4);
    let normal_policy = VramPressurePolicy::for_snapshot(normal);
    assert!(!normal_policy.throttle_blas_compaction);
    assert_eq!(normal_policy.lod_cutoff_scale, 1.0);
    assert!(!normal_policy.trim_reservoirs);

    let elevated = VramBudgetSnapshot::from_usage_and_budget(6_400_000_000, 8_000_000_000, true);
    assert_eq!(elevated.pressure, VramPressureLevel::Elevated);
    let elevated_policy = VramPressurePolicy::for_snapshot(elevated);
    assert!(elevated_policy.throttle_blas_compaction);
    assert_eq!(elevated_policy.lod_cutoff_scale, 0.75);
    assert!(!elevated_policy.trim_reservoirs);

    let critical = VramBudgetSnapshot::from_usage_and_budget(7_500_000_000, 8_000_000_000, true);
    assert_eq!(critical.pressure, VramPressureLevel::Critical);
    let critical_policy = VramPressurePolicy::for_snapshot(critical);
    assert!(critical_policy.throttle_blas_compaction);
    assert_eq!(critical_policy.lod_cutoff_scale, 0.50);
    assert!(critical_policy.trim_reservoirs);

    // Fallback with zero budget must not divide by zero
    let zero = VramBudgetSnapshot::from_usage_and_budget(0, 0, false);
    assert_eq!(zero.usage_ratio, 0.0);
    assert_eq!(zero.pressure, VramPressureLevel::Normal);
}

#[test]
fn moltenvk_and_missing_rt_extensions_clamp_render_tier_to_raster() {
    use vulkanshaders_core::gpu::{DeviceCapabilities, evaluate_platform_tier_policy};

    // MoltenVK with Vulkan 1.3 raster features but no hardware RT pipeline
    let molten_raster = DeviceCapabilities {
        portability_subset: true,
        is_moltenvk: true,
        descriptor_indexing: true,
        synchronization2: true,
        dynamic_rendering: true,
        acceleration_structure: false,
        ray_tracing_pipeline: false,
        ..DeviceCapabilities::default()
    };
    let policy = evaluate_platform_tier_policy(&molten_raster, true, true);
    assert!(policy.is_moltenvk);
    assert!(policy.hardware_rt_blocked);
    assert_eq!(policy.max_render_tier, RenderTier::Tier1ComputeGi);
    assert_eq!(
        policy.clamp_render_tier(RenderTier::Tier3PathTracing),
        RenderTier::Tier1ComputeGi
    );
    assert_eq!(
        policy.clamp_render_tier(RenderTier::Tier0Raster),
        RenderTier::Tier0Raster
    );
    assert!(!policy.reason.is_empty());

    // Minimal MoltenVK without dynamic rendering/descriptor indexing clamps to Tier 0
    let molten_minimal = DeviceCapabilities {
        portability_subset: true,
        is_moltenvk: true,
        ..DeviceCapabilities::default()
    };
    let min_policy = evaluate_platform_tier_policy(&molten_minimal, true, true);
    assert_eq!(min_policy.max_render_tier, RenderTier::Tier0Raster);
    assert_eq!(
        min_policy.clamp_render_tier(RenderTier::Tier2RayQuery),
        RenderTier::Tier0Raster
    );

    // Desktop GPU with full RT pipeline allows Tier 3
    let full_rt = DeviceCapabilities {
        buffer_device_address: true,
        descriptor_indexing: true,
        synchronization2: true,
        dynamic_rendering: true,
        acceleration_structure: true,
        ray_query: true,
        ray_tracing_pipeline: true,
        ..DeviceCapabilities::default()
    };
    let rt_policy = evaluate_platform_tier_policy(&full_rt, false, false);
    assert!(!rt_policy.hardware_rt_blocked);
    assert_eq!(rt_policy.max_render_tier, RenderTier::Tier3PathTracing);
    assert_eq!(
        rt_policy.clamp_render_tier(RenderTier::Tier3PathTracing),
        RenderTier::Tier3PathTracing
    );
}

#[test]
fn gpu_timestamp_profiler_converts_ticks_and_tracks_toggle_and_passes() {
    use vulkanshaders_core::gpu::GpuTimestampProfiler;

    // 1 tick = 1 ns -> 2_500_000 ticks = 2.5 ms
    let ms = GpuTimestampProfiler::ticks_to_ms(1_000_000, 3_500_000, 1.0);
    assert!((ms - 2.5).abs() < 1e-6);

    // 1 tick = 10 ns -> 150_000 ticks = 1.5 ms
    let ms_10ns = GpuTimestampProfiler::ticks_to_ms(50_000, 200_000, 10.0);
    assert!((ms_10ns - 1.5).abs() < 1e-6);

    // Underflow and invalid period saturate safely to 0.0
    assert_eq!(GpuTimestampProfiler::ticks_to_ms(500, 100, 1.0), 0.0);
    assert_eq!(GpuTimestampProfiler::ticks_to_ms(0, 1000, 0.0), 0.0);
    assert_eq!(GpuTimestampProfiler::ticks_to_ms(0, 1000, f32::NAN), 0.0);

    let mut profiler = GpuTimestampProfiler::new_headless(1.0, 3, 16);
    assert!(!profiler.is_enabled());
    profiler.set_enabled(true);
    assert!(profiler.is_enabled());

    let timings = profiler.resolve_synthetic_passes(
        0,
        &[
            ("csm_shadow", 1_000_000, 2_250_000),
            ("pathtrace", 2_250_000, 6_750_000),
            ("nrd_denoise", 6_750_000, 8_000_000),
        ],
    );
    assert_eq!(timings.len(), 3);
    assert_eq!(timings[0].name, "csm_shadow");
    assert!((timings[0].duration_ms - 1.25).abs() < 1e-6);
    assert!((timings[1].duration_ms - 4.5).abs() < 1e-6);
    assert!((timings[2].duration_ms - 1.25).abs() < 1e-6);
    assert!((profiler.last_total_gpu_ms() - 7.0).abs() < 1e-6);
    assert_eq!(profiler.average_total_gpu_ms(), None);
    for frame in 0..35 {
        profiler.resolve_synthetic_passes(frame, &[("render", 0, 2_000_000)]);
    }
    assert!((profiler.average_total_gpu_ms().unwrap() - (77.0 / 36.0)).abs() < 1e-6);
    profiler.set_enabled(false);
    assert_eq!(profiler.average_total_gpu_ms(), None);
}

#[test]
fn device_lost_guard_tracks_tdr_and_shader_diagnostics_and_recovers() {
    use vulkanshaders_core::crash::{DeviceLostGuard, RecoveryReason, ShaderCompileDiagnostic};

    let guard = DeviceLostGuard::new();
    assert!(!guard.is_rt_vsp_disabled());
    assert!(guard.format_ffi_notice().is_none());

    // Non-fatal Vulkan results do not trip fallback
    assert!(!guard.record_vk_result("test", ash::vk::Result::SUCCESS));
    assert!(!guard.is_rt_vsp_disabled());

    // VK_ERROR_DEVICE_LOST trips fallback and records notice
    assert!(guard.record_vk_result("queue_submit", ash::vk::Result::ERROR_DEVICE_LOST));
    assert!(guard.is_rt_vsp_disabled());
    let snap = guard.snapshot();
    assert_eq!(snap.reason, RecoveryReason::DeviceLost);
    assert_eq!(snap.device_lost_count, 1);
    assert!(
        guard
            .format_ffi_notice()
            .unwrap()
            .contains("VK_ERROR_DEVICE_LOST")
    );

    // Reset restores normal state
    guard.reset();
    assert!(!guard.is_rt_vsp_disabled());
    assert_eq!(guard.snapshot().reason, RecoveryReason::None);

    // Structured shader compile diagnostic parsing + recording
    let diag = ShaderCompileDiagnostic::parse_from_compiler_error(
        "pathtrace.rgen",
        "ERROR: shaders/rt/pathtrace.rgen:142:8: undeclared identifier 'foo'",
    );
    assert_eq!(diag.file, "shaders/rt/pathtrace.rgen");
    assert_eq!(diag.line, 142);
    assert_eq!(diag.column, 8);
    assert_eq!(diag.message, "undeclared identifier 'foo'");

    guard.record_shader_compile_failure(diag.clone());
    assert!(guard.is_rt_vsp_disabled());
    let snap2 = guard.snapshot();
    assert_eq!(snap2.reason, RecoveryReason::ShaderCompileFailure);
    assert_eq!(snap2.shader_failure_count, 1);
    assert_eq!(snap2.last_shader_diagnostic, Some(diag));
}

#[test]
fn persistent_pipeline_cache_key_is_deterministic_and_roundtrips_blobs() {
    use vulkanshaders_core::modules::{PersistentPipelineDiskCache, PipelineCacheKeyInput};

    let input_a = PipelineCacheKeyInput {
        pack_hash: [0x11; 32],
        driver_uuid: [0x22; 16],
        pipeline_cache_uuid: [0x33; 16],
        spec_constants: vec![(2, 100), (0, 1), (1, 42)],
    };
    let input_b = PipelineCacheKeyInput {
        pack_hash: [0x11; 32],
        driver_uuid: [0x22; 16],
        pipeline_cache_uuid: [0x33; 16],
        spec_constants: vec![(0, 1), (1, 42), (2, 100)],
    };
    let input_diff_driver = PipelineCacheKeyInput {
        driver_uuid: [0x99; 16],
        ..input_a.clone()
    };
    let input_diff_spec = PipelineCacheKeyInput {
        spec_constants: vec![(0, 2), (1, 42), (2, 100)],
        ..input_a.clone()
    };

    assert_eq!(input_a.compute_key(), input_b.compute_key());
    assert_eq!(input_a.compute_hex().len(), 64);
    assert_ne!(input_a.compute_key(), input_diff_driver.compute_key());
    assert_ne!(input_a.compute_key(), input_diff_spec.compute_key());

    let tmp_dir =
        std::env::temp_dir().join(format!("hyaloclast-cache-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    let cache = PersistentPipelineDiskCache::new(tmp_dir.clone());
    let key = input_a.compute_key();

    let fake_vk_blob = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04];
    cache
        .save_pipeline_cache_blob(&key, &fake_vk_blob)
        .expect("save pipeline cache blob");
    assert_eq!(cache.load_pipeline_cache_blob(&key), Some(fake_vk_blob));

    let fake_spv = vec![0x0723_0203u32, 0x0001_0500, 0, 1, 0, 0x0002_0011];
    cache
        .save_spirv_words(&key, "pathtrace.rgen", &fake_spv)
        .expect("save spirv words");
    assert_eq!(
        cache.load_spirv_words(&key, "pathtrace.rgen"),
        Some(fake_spv)
    );

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn async_pipeline_prewarmer_reports_progress_completion_and_cancellation() {
    use std::sync::atomic::Ordering;
    use std::time::Duration;
    use vulkanshaders_core::modules::AsyncPipelinePrewarmer;

    // 1. Successful completion
    let mut prewarmer = AsyncPipelinePrewarmer::start_custom(4, |_cancel, report| {
        for step in 1..=4 {
            report(step, 4);
        }
        Ok(())
    });
    prewarmer.wait();
    let progress = prewarmer.progress();
    assert!(progress.ready);
    assert!(!progress.cancelled);
    assert_eq!(progress.completed, 4);
    assert_eq!(progress.total, 4);
    assert!(progress.last_error.is_none());

    // 2. Cooperative cancellation
    let mut cancellable = AsyncPipelinePrewarmer::start_custom(100, |cancel, report| {
        for step in 0..100 {
            if cancel.load(Ordering::Acquire) {
                return Ok(());
            }
            report(step, 100);
            std::thread::sleep(Duration::from_millis(2));
        }
        Ok(())
    });
    cancellable.cancel();
    cancellable.wait();
    let cancel_progress = cancellable.progress();
    assert!(cancel_progress.cancelled);
    assert!(!cancel_progress.ready);
}

#[test]
fn sdk_upscaler_fallback_chain_and_temporal_reset_propagation() {
    use vulkanshaders_core::sdk::{
        EffectiveUpscaler, RuntimeSdkCapabilities, RuntimeSdkManager,
        resolve_upscaler_fallback_for_caps,
    };

    // DLSS unavailable, FSR3 available -> falls back to FSR3
    let caps_fsr = RuntimeSdkCapabilities {
        fsr3_loaded: true,
        ..RuntimeSdkCapabilities::default()
    };
    let (eff, reason) = resolve_upscaler_fallback_for_caps(&caps_fsr, 3, false);
    assert_eq!(eff, EffectiveUpscaler::Fsr3);
    assert!(reason.is_some());

    // DLSS & FSR3 unavailable, XeSS available -> falls back to XeSS on non-macOS, TSR on macOS
    let caps_xess = RuntimeSdkCapabilities {
        xess_loaded: true,
        ..RuntimeSdkCapabilities::default()
    };
    let (eff_xess, _) = resolve_upscaler_fallback_for_caps(&caps_xess, 3, false);
    assert_eq!(eff_xess, EffectiveUpscaler::XeSS);
    let (eff_macos, _) = resolve_upscaler_fallback_for_caps(&caps_xess, 3, true);
    assert_eq!(eff_macos, EffectiveUpscaler::Tsr);

    // Nothing loaded -> falls back to TSR
    let empty_caps = RuntimeSdkCapabilities::default();
    let (eff_tsr, reason_tsr) = resolve_upscaler_fallback_for_caps(&empty_caps, 3, false);
    assert_eq!(eff_tsr, EffectiveUpscaler::Tsr);
    assert!(reason_tsr.is_some());

    // RuntimeSdkManager::reset_temporal_history flags all three subsystems
    let mut mgr = RuntimeSdkManager::probe(Vec::new());
    mgr.reset_temporal_history();
    assert_eq!(mgr.pending_temporal_resets(), (true, true, true));
}
