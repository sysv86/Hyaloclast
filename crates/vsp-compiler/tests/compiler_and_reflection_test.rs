use std::{collections::BTreeMap, path::Path};
use vsp_compiler::{
    CompileDiagnostic, HardwareShaderCaps, HookStage, PackManifest, PackPresetDecl,
    PipelineGraphMode, ReflectedDescriptorType, RenderTier, SettingValue, ShaderStageKind,
    SourceLineMap, SpirvCacheKey, SpirvDiskCache, SpirvHookLinker, VslPreprocessor, VspCompiler,
    VspError, parse_compile_diagnostics, validate_pack_directory,
};

#[test]
fn isolates_conflicting_helper_names_across_vsl_hooks() {
    let hook_a = r#"
#include <vulkanshaders/v1/common.vsl>
vec3 helper_curve(vec3 x) { return x * 0.5; }
// @hook(stage = "surface", priority = 100)
void apply_hook(inout VsSurface s, in uint flags, in uint bounce) {
    s.albedo = helper_curve(s.albedo);
}
"#;
    let hook_b = r#"
#include <vulkanshaders/v1/common.vsl>
vec3 helper_curve(vec3 x) { return x * 2.0; }
// @hook(stage = "post_grade", priority = 100)
void apply_hook(inout vec3 hdrColor, in vec2 uv, in VsPostContext post) {
    hdrColor = helper_curve(hdrColor);
}
"#;

    let (isolated_a, anns_a) =
        VslPreprocessor::isolate_hook_module(hook_a, "vs_ns_pack_surface_0_");
    let (isolated_b, anns_b) = VslPreprocessor::isolate_hook_module(hook_b, "vs_ns_pack_post_1_");

    assert!(isolated_a.contains("vs_ns_pack_surface_0_helper_curve"));
    assert!(isolated_b.contains("vs_ns_pack_post_1_helper_curve"));
    assert_eq!(anns_a[0].function_name, "vs_ns_pack_surface_0_apply_hook");
    assert_eq!(anns_b[0].function_name, "vs_ns_pack_post_1_apply_hook");
}

#[test]
fn glsl_lexer_ignores_braces_in_comments_and_strings() {
    let tricky_hook = r#"
#include <vulkanshaders/v1/common.vsl>
// Unbalanced brace in line comment: { { {
/* Multi-line block comment with braces: } } { { */
float helper_gain(float x) {
    // Another comment with } inside body
    return x * 1.5;
}
// @hook(stage = "surface", priority = 100)
void tricky_surface(inout VsSurface s, in uint flags, in uint bounce) {
    /* { unmatched brace inside comment */
    s.roughness = clamp(helper_gain(s.roughness), 0.05, 0.95);
}
"#;
    let extracted =
        VslPreprocessor::extract_hook_helper_and_body_lexed(tricky_hook, HookStage::Surface, None)
            .expect("lexed hook extraction must succeed despite comment braces");
    assert!(
        extracted
            .isolated_helper_code
            .contains("float helper_gain(float x)")
    );
    assert!(
        extracted
            .isolated_body
            .contains("s.roughness = clamp(helper_gain(s.roughness), 0.05, 0.95);")
    );
}

#[test]
fn annotated_water_surface_is_not_a_generic_surface_hook() {
    let water_hook = r#"
// @hook(stage = "water_surface", priority = 100)
void water_surface(inout VsSurface s, vec3 worldPos, float waveHeight) {
    s.roughness = worldPos.x;
}
"#;
    assert!(
        VslPreprocessor::extract_hook_helper_and_body_lexed(water_hook, HookStage::Surface, None)
            .is_none()
    );
    assert!(
        VslPreprocessor::extract_hook_helper_and_body_lexed(
            water_hook,
            HookStage::WaterSurface,
            None
        )
        .is_some()
    );
}

#[test]
fn parses_vsp_graph_v2_targets_buffers_passes_and_16_hook_stages() {
    let toml_str = r#"
[pack]
id = "vsp-graph-v2-demo"
name = "VSP Graph v2 Demo"
version = "0.3.0"

[capabilities]
tiers = [0, 1, 2, 3]

[pipeline]
mode = "extend"
gbuffer_MRT_count = 5
modules = ["modules/noise_lib.vsl"]

[targets.custom_bloom]
format = "rgba16f"
scale = 0.5
temporal = true
gbuffer_attachment = 4

[buffers.restir_spatial_grid]
size_bytes = 131072
clear_each_frame = false
persistent = true

[[passes]]
id = "custom_bloom_pass"
kind = "compute"
shader = "shaders/compute/bloom.comp"
min_tier = 0
max_tier = 3
after = "lighting"
reads = ["raw_radiance"]
writes = ["custom_bloom"]
scale = 0.5
bindings = { b1 = "raw_radiance", b2 = "custom_bloom" }

[hooks]
gbuffer_pack = "hooks/gbuffer_pack.vsl"
restir_reservoir = "hooks/restir.vsl"
volumetric_media = "hooks/volumetric.vsl"
"#;
    let manifest = PackManifest::parse_and_validate(toml_str).expect("valid VSP Graph v2 manifest");
    assert_eq!(manifest.pipeline.mode, PipelineGraphMode::Extend);
    assert_eq!(manifest.pipeline.gbuffer_mrt_count, 5);
    assert_eq!(manifest.targets.len(), 1);
    assert_eq!(manifest.buffers.len(), 1);
    assert_eq!(manifest.passes.len(), 1);
    assert_eq!(manifest.passes[0].id, "custom_bloom_pass");
    assert_eq!(manifest.hooks.len(), 3);
    assert_eq!(HookStage::ALL.len(), 16);
}

#[test]
fn compiles_and_reflects_rayquery_bda_and_raytracing_stages_without_noop_stub() {
    let rq_src = r#"#version 460
#extension GL_EXT_ray_query : require
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0) uniform WorldUBO { mat4 vp; } vsWorld;
layout(set = 0, binding = 1) uniform sampler2D gAlbedo;
layout(set = 0, binding = 5, rgba16f) uniform writeonly image2D outHdrRadiance;
layout(set = 1, binding = 0) uniform accelerationStructureEXT topLevelAS;
void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, 0x01u, 0xFFu, vec3(0.0), 0.01, vec3(0.0, 1.0, 0.0), 32.0);
    rayQueryProceedEXT(rq);
    imageStore(outHdrRadiance, px, vec4(1.0));
}
"#;
    let rq_words = VspCompiler::compile_glsl_to_spirv(
        "rayquery_lighting.comp",
        ShaderStageKind::Compute,
        rq_src,
    )
    .expect("compile rayquery_lighting.comp");
    assert!(
        rq_words.len() > 60,
        "SPIR-V module must contain real instructions, not an empty OpReturn stub"
    );

    let rq_refl =
        SpirvHookLinker::reflect_module(&rq_words).expect("reflect rayquery_lighting.comp");
    assert!(rq_refl.uses_ray_query);
    assert_eq!(rq_refl.workgroup_size, [8, 8, 1]);
    assert!(rq_refl.bindings.iter().any(|b| b.set == 1
        && b.binding == 0
        && b.descriptor_type == ReflectedDescriptorType::AccelerationStructure));

    let rgen_src = r#"#version 460
#extension GL_EXT_ray_tracing : require
layout(set = 0, binding = 1, rgba16f) uniform writeonly image2D outRawRadiance;
layout(set = 1, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(location = 0) rayPayloadEXT vec4 hitPayload;
void main() {
    traceRayEXT(topLevelAS, 1u, 0xFFu, 0u, 0u, 0u, vec3(0.0), 0.01, vec3(0.0, 1.0, 0.0), 100.0, 0);
}
"#;
    let rgen_words =
        VspCompiler::compile_glsl_to_spirv("pathtrace.rgen", ShaderStageKind::RayGen, rgen_src)
            .expect("compile pathtrace.rgen");
    let rgen_refl = SpirvHookLinker::reflect_module(&rgen_words).expect("reflect pathtrace.rgen");
    assert!(rgen_refl.uses_ray_tracing_pipeline);
    assert_eq!(rgen_refl.entry_points, vec!["main".to_owned()]);
}

#[test]
fn validates_manifest_and_resolves_specialization_overrides() {
    let toml_str = r#"
[pack]
id = "hyaloclast-default"
name = "Hyaloclast Default"
version = "0.2.0"

[capabilities]
tiers = [0, 1, 2, 3]

[settings.enable_wind]
label = "Foliage Wind"
type = "bool"
default = true
runtime = false
"#;
    let manifest = PackManifest::parse_and_validate(toml_str).expect("valid pack.toml");
    let mut overrides = BTreeMap::new();
    overrides.insert("enable_wind".to_owned(), "false".to_owned());
    let resolved = manifest.resolve_specialization_overrides(&overrides);
    assert_eq!(resolved, vec![(100, 0)]);
}

#[test]
fn invalid_glsl_and_rt_stages_fail_with_glsl_parse_error() {
    let invalid_comp = r#"#version 460
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    undeclared_symbol_should_fail = 42.0;
}
"#;
    let err =
        VspCompiler::compile_glsl_to_spirv("invalid.comp", ShaderStageKind::Compute, invalid_comp)
            .expect_err("invalid compute GLSL must fail");
    assert!(matches!(err, VspError::GlslParse { .. }), "got {err:?}");

    let invalid_rgen = r#"#version 460
#extension GL_EXT_ray_tracing : require
layout(location = 0) rayPayloadEXT vec4 hitPayload;
void main() {
    traceRayEXT(missingTlas, 0u, 0xFFu, 0u, 0u, 0u, vec3(0.0), 0.01, vec3(0.0, 1.0, 0.0), 100.0, 0);
}
"#;
    let rt_err =
        VspCompiler::compile_glsl_to_spirv("invalid.rgen", ShaderStageKind::RayGen, invalid_rgen)
            .expect_err("invalid raygen GLSL must fail with GlslParse");
    assert!(
        matches!(rt_err, VspError::GlslParse { .. }),
        "got {rt_err:?}"
    );

    let invalid_rahit = r#"#version 460
#extension GL_EXT_ray_tracing : require
void main() {
    not_valid_glsl_at_all(123);
}
"#;
    let rahit_err =
        VspCompiler::compile_glsl_to_spirv("invalid.rahit", ShaderStageKind::AnyHit, invalid_rahit)
            .expect_err("invalid any-hit GLSL must fail with GlslParse");
    assert!(
        matches!(rahit_err, VspError::GlslParse { .. }),
        "got {rahit_err:?}"
    );
}

#[test]
fn compiles_and_reflects_any_hit_stage_and_descriptor_sets_0_to_3() {
    let toml_str = r#"
[pack]
id = "rt-test"
name = "RT Test"
version = "0.2.0"

[capabilities]
tiers = [0, 1, 2, 3]

[resources.blue_noise]
path = "textures/blue_noise.png"
"#;
    let manifest = PackManifest::parse_and_validate(toml_str).expect("valid pack.toml");

    let rahit_src = r#"#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_nonuniform_qualifier : require

#include <vulkanshaders/v1/common.vsl>

layout(set = 1, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(set = 1, binding = 1, std430) readonly buffer InstanceMeta {
    uvec4 meta[];
} uInstances;
layout(set = 2, binding = 0) uniform sampler2D uBindlessTextures[];

layout(location = 0) rayPayloadInEXT vec4 hitPayload;
hitAttributeEXT vec2 barycentrics;

void main() {
    uint texIdx = uInstances.meta[gl_InstanceCustomIndexEXT].x;
    vec4 alphaSample = textureLod(uBindlessTextures[nonuniformEXT(texIdx)], barycentrics, 0.0);
    vec4 noise = vsSamplePackTexture_blue_noise(barycentrics + vec2(float(vsWorld.frameIndex) * 0.01));
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT, 0xFFu, vec3(0.0), 0.01, vec3(0.0, 1.0, 0.0), 10.0);
    if (alphaSample.a * noise.r < 0.5 && rayQueryProceedEXT(rq)) {
        ignoreIntersectionEXT;
    }
    hitPayload = vec4(alphaSample.rgb, gl_HitTEXT);
}
"#;

    let compiler = VspCompiler::new(None);
    let artifact = compiler
        .compile_stage(
            "foliage.rahit",
            ShaderStageKind::AnyHit,
            rahit_src,
            &manifest,
            RenderTier::Tier3PathTracing,
            &|_| None,
        )
        .expect("compile foliage.rahit via shaderc");

    assert_eq!(artifact.stage_kind, ShaderStageKind::AnyHit);
    let refl = SpirvHookLinker::reflect_spirv(&artifact.spirv_words)
        .expect("reflect foliage.rahit via reflect_spirv");
    assert!(refl.uses_ray_tracing_pipeline);
    assert_eq!(refl.entry_points, vec!["main".to_owned()]);

    // Set 0: WorldUBO
    assert!(refl.bindings.iter().any(|b| {
        b.set == 0 && b.binding == 0 && b.descriptor_type == ReflectedDescriptorType::UniformBuffer
    }));
    // Set 1: TLAS + InstanceMeta
    assert!(refl.bindings.iter().any(|b| {
        b.set == 1
            && b.binding == 0
            && b.descriptor_type == ReflectedDescriptorType::AccelerationStructure
    }));
    assert!(refl.bindings.iter().any(|b| {
        b.set == 1 && b.binding == 1 && b.descriptor_type == ReflectedDescriptorType::StorageBuffer
    }));
    // Set 2: BindlessTextureTable
    assert!(refl.bindings.iter().any(|b| {
        b.set == 2
            && b.binding == 0
            && b.descriptor_type == ReflectedDescriptorType::CombinedImageSampler
            && b.count == 4096
    }));
    // Set 3: Pack [resources.*]
    assert!(refl.bindings.iter().any(|b| {
        b.set == 3
            && b.binding == 0
            && b.descriptor_type == ReflectedDescriptorType::CombinedImageSampler
    }));
}

#[test]
fn injects_post_grade_hook_into_compute_shader_without_vs_surface() {
    let toml_str = r#"
[pack]
id = "post-only"
name = "Post Only"
version = "0.2.0"

[capabilities]
tiers = [0, 1, 2, 3]

[hooks]
surface = "hooks/surface.vsl"
post_grade = "hooks/post_grade.vsl"

[resources.color_lut]
path = "textures/lut.png"
"#;
    let manifest = PackManifest::parse_and_validate(toml_str).expect("valid pack.toml");

    let surface_hook = r#"#include <vulkanshaders/v1/common.vsl>
// @hook(stage = "surface", priority = 100)
void custom_surface(inout VsSurface s, in uint flags, in uint bounce) {
    s.albedo *= 0.5;
}
"#;
    let post_grade_hook = r#"#include <vulkanshaders/v1/common.vsl>
vec3 helper_tonemap(vec3 c) {
    return c / (c + vec3(1.0));
}
// @hook(stage = "post_grade", priority = 100)
void custom_post_grade(inout vec3 hdrColor, in vec2 uv, in VsPostContext post) {
    hdrColor = helper_tonemap(hdrColor * post.exposure);
}
"#;

    let loader = |path: &str| -> Option<String> {
        match path {
            "hooks/surface.vsl" => Some(surface_hook.to_owned()),
            "hooks/post_grade.vsl" => Some(post_grade_hook.to_owned()),
            _ => None,
        }
    };

    // Compute shader deliberately does NOT declare `struct VsSurface` or include `common.vsl`
    let post_comp_src = r#"#version 460
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

struct VsPostContext {
    float exposure;
    float bloomIntensity;
    uint toneMapMode;
};

layout(set = 0, binding = 0, rgba16f) uniform image2D outColor;

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = (vec2(px) + 0.5) / vec2(imageSize(outColor));
    vec3 hdr = vec3(1.5, 0.8, 0.4);
    VsPostContext post = VsPostContext(1.2, 0.1, 0u);
    vs_dispatch_hook_post_grade(hdr, uv, post);
    vec4 lut = vsSamplePackTexture_color_lut(uv);
    imageStore(outColor, px, vec4(hdr * lut.rgb, 1.0));
}
"#;

    let preprocessor = VslPreprocessor::default();
    let preprocessed = preprocessor
        .preprocess(
            "post_tonemap.comp",
            post_comp_src,
            &manifest,
            RenderTier::Tier0Raster,
            &loader,
        )
        .expect("preprocess post_tonemap.comp without VsSurface");

    assert!(
        preprocessed
            .glsl_source
            .contains("vs_dispatch_hook_post_grade")
    );
    assert!(
        !preprocessed
            .glsl_source
            .contains("vs_dispatch_hook_surface")
    );
    assert!(
        preprocessed
            .glsl_source
            .contains("layout(set = 3, binding = 0) uniform sampler2D uVsPackRes_color_lut;")
    );
    assert_eq!(preprocessed.discovered_hooks.len(), 1);
    assert_eq!(preprocessed.discovered_hooks[0].stage, HookStage::PostGrade);

    let compiler = VspCompiler::new(None);
    let artifact = compiler
        .compile_stage(
            "post_tonemap.comp",
            ShaderStageKind::Compute,
            post_comp_src,
            &manifest,
            RenderTier::Tier0Raster,
            &loader,
        )
        .expect("compile post_tonemap.comp without VsSurface");

    assert!(
        artifact.reflection.bindings.iter().any(|b| {
            b.set == 3
                && b.binding == 0
                && b.descriptor_type == ReflectedDescriptorType::CombinedImageSampler
        }),
        "actual bindings: {:?}",
        artifact.reflection.bindings
    );
}

#[test]
fn compiles_all_builtin_hyaloclast_default_shaders_across_tiers() {
    let pack_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../src/main/resources/builtin-packs/hyaloclast-sunlit-vanilla.vsp");
    let compiler = VspCompiler::new(None);
    for tier in [
        RenderTier::Tier0Raster,
        RenderTier::Tier2RayQuery,
        RenderTier::Tier3PathTracing,
    ] {
        let artifacts = compiler
            .compile_pack_dir(&pack_dir, tier)
            .expect("compile hyaloclast-sunlit-vanilla.vsp");
        assert!(
            artifacts.len() >= 15,
            "expected all 15 shaders in hyaloclast-sunlit-vanilla.vsp, got {}",
            artifacts.len()
        );
    }
}

#[test]
fn compiles_builtin_path_tracing_with_both_ser_extensions() {
    let pack_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../src/main/resources/builtin-packs/hyaloclast-sunlit-vanilla.vsp");
    let compiler = VspCompiler::new(None);
    for (ser_ext, ser_nv) in [(true, false), (false, true)] {
        let caps = HardwareShaderCaps {
            ser_ext,
            ser_nv,
            ada_or_newer: true,
            high_end_gpu: true,
            ..Default::default()
        };
        let result =
            compiler.compile_pack_dir_with_caps(&pack_dir, RenderTier::Tier3PathTracing, caps);
        let artifacts = match result {
            Ok(artifacts) => artifacts,
            Err(error)
                if ser_ext
                    && error
                        .to_string()
                        .contains("GL_EXT_shader_invocation_reorder") =>
            {
                continue;
            }
            Err(error) => panic!("could not compile {caps:?}: {error}"),
        };
        assert!(
            artifacts
                .iter()
                .any(|stage| stage.stage_kind == ShaderStageKind::RayGen)
        );
    }
}

#[test]
fn parses_presets_and_setting_tier_ranges_with_invariant_layout() {
    let toml_str = r#"
[pack]
id = "preset-demo"
name = "Preset Demo"
version = "1.0.0"

[capabilities]
tiers = [0, 1, 2, 3]

[settings.sun_intensity]
label = "Sun Intensity"
type = "float"
default = 2.8
min = 0.5
max = 8.0
runtime = true

[settings.gi_bounce_strength]
label = "GI Bounce"
type = "float"
default = 1.0
min = 0.2
max = 2.0
runtime = true
min_tier = 1
max_tier = 3

[settings.pt_max_bounces]
label = "PT Max Bounces"
type = "int"
default = 2
min = 1
max = 4
runtime = false
min_tier = 3
max_tier = 3

[settings.tone_mapping]
label = "Tone Mapping"
type = "enum"
default = "agx"
options = ["agx", "aces", "pbr_neutral"]
runtime = true

[settings.enable_wind]
label = "Foliage Wind"
type = "bool"
default = true
runtime = false

[presets.low]
label = "Low"
description = "Fast raster profile"
order = 0
min_tier = 0
max_tier = 2
values = { sun_intensity = 2.4, enable_wind = false, tone_mapping = "pbr_neutral" }

[presets.ultra]
label = "Ultra"
description = "Full path tracing profile"
order = 1
min_tier = 2
max_tier = 3

[presets.ultra.values]
sun_intensity = 3.2
gi_bounce_strength = 1.4
pt_max_bounces = 4
enable_wind = true
"#;

    let manifest = PackManifest::parse_and_validate(toml_str).expect("valid manifest with presets");
    assert_eq!(manifest.settings["sun_intensity"].min_tier, 0);
    assert_eq!(manifest.settings["sun_intensity"].max_tier, 3);
    assert_eq!(manifest.settings["gi_bounce_strength"].min_tier, 1);
    assert_eq!(manifest.settings["gi_bounce_strength"].max_tier, 3);
    assert_eq!(manifest.settings["pt_max_bounces"].min_tier, 3);
    assert_eq!(manifest.settings["pt_max_bounces"].max_tier, 3);

    assert_eq!(manifest.presets.len(), 2);
    let low: &PackPresetDecl = &manifest.presets["low"];
    assert_eq!(low.label, "Low");
    assert_eq!(low.description, "Fast raster profile");
    assert_eq!(low.order, 0);
    assert_eq!(low.min_tier, 0);
    assert_eq!(low.max_tier, 2);
    assert_eq!(low.values["enable_wind"], SettingValue::Bool(false));
    assert_eq!(
        low.values["tone_mapping"],
        SettingValue::Str("pbr_neutral".to_owned())
    );

    let ultra = &manifest.presets["ultra"];
    assert_eq!(ultra.order, 1);
    assert_eq!(ultra.min_tier, 2);
    assert_eq!(ultra.max_tier, 3);
    assert_eq!(ultra.values["pt_max_bounces"], SettingValue::Int(4));

    // Layout slot numbering must remain invariant regardless of setting tier ranges.
    let layout = manifest.build_parameter_layout();
    assert_eq!(layout.runtime_ubo_indices["gi_bounce_strength"], 0);
    assert_eq!(layout.runtime_ubo_indices["sun_intensity"], 1);
    assert_eq!(layout.runtime_ubo_indices["tone_mapping"], 2);
    assert_eq!(layout.specialization_constant_ids["enable_wind"], 100);
    assert_eq!(layout.specialization_constant_ids["pt_max_bounces"], 101);
}

#[test]
fn rejects_invalid_presets_and_setting_tier_ranges() {
    let unknown_key = r#"
[pack]
id = "bad-unknown"
name = "Bad Unknown"
[capabilities]
tiers = [0, 1, 2, 3]
[settings.sun_intensity]
label = "Sun"
type = "float"
default = 2.0
min = 0.5
max = 4.0
[presets.low]
label = "Low"
values = { nonexistent = 1.0 }
"#;
    let err = PackManifest::parse_and_validate(unknown_key).expect_err("unknown preset key");
    assert!(
        matches!(err, VspError::ManifestValidation { .. }),
        "got {err:?}"
    );

    let out_of_range = r#"
[pack]
id = "bad-range"
name = "Bad Range"
[capabilities]
tiers = [0, 1, 2, 3]
[settings.pt_max_bounces]
label = "Bounces"
type = "int"
default = 2
min = 1
max = 4
runtime = false
[presets.ultra]
label = "Ultra"
values = { pt_max_bounces = 8 }
"#;
    let err =
        PackManifest::parse_and_validate(out_of_range).expect_err("out of range preset value");
    assert!(
        matches!(err, VspError::ManifestValidation { .. }),
        "got {err:?}"
    );

    let reserved_custom = r#"
[pack]
id = "bad-custom"
name = "Bad Custom"
[capabilities]
tiers = [0, 1, 2, 3]
[settings.enable_wind]
label = "Wind"
type = "bool"
default = true
[presets.Custom]
label = "Custom"
values = { enable_wind = false }
"#;
    let err = PackManifest::parse_and_validate(reserved_custom).expect_err("reserved custom id");
    assert!(
        matches!(err, VspError::ManifestValidation { .. }),
        "got {err:?}"
    );

    let bad_preset_tier = r#"
[pack]
id = "bad-preset-tier"
name = "Bad Preset Tier"
[capabilities]
tiers = [0, 1, 2, 3]
[presets.high]
label = "High"
min_tier = 3
max_tier = 1
"#;
    let err =
        PackManifest::parse_and_validate(bad_preset_tier).expect_err("invalid preset tier range");
    assert!(
        matches!(err, VspError::ManifestValidation { .. }),
        "got {err:?}"
    );

    let bad_setting_tier = r#"
[pack]
id = "bad-setting-tier"
name = "Bad Setting Tier"
[capabilities]
tiers = [0, 1, 2, 3]
[settings.gi_bounce_strength]
label = "GI"
type = "float"
default = 1.0
min_tier = 2
max_tier = 4
"#;
    let err =
        PackManifest::parse_and_validate(bad_setting_tier).expect_err("invalid setting tier range");
    assert!(
        matches!(err, VspError::ManifestValidation { .. }),
        "got {err:?}"
    );

    let bad_enum = r#"
[pack]
id = "bad-enum"
name = "Bad Enum"
[capabilities]
tiers = [0, 1, 2, 3]
[settings.tone_mapping]
label = "Tone Mapping"
type = "enum"
default = "agx"
options = ["agx", "aces"]
[presets.film]
label = "Film"
values = { tone_mapping = "reinhard" }
"#;
    let err = PackManifest::parse_and_validate(bad_enum).expect_err("invalid enum value in preset");
    assert!(
        matches!(err, VspError::ManifestValidation { .. }),
        "got {err:?}"
    );
}

#[test]
fn inherits_and_overrides_presets_across_parent_and_child_manifests() {
    let parent_toml = r#"
[pack]
id = "parent-pack"
name = "Parent"
[capabilities]
tiers = [0, 1, 2, 3]
[settings.sun_intensity]
label = "Sun"
type = "float"
default = 2.8
min = 0.5
max = 8.0
[presets.low]
label = "Low"
order = 0
values = { sun_intensity = 2.0 }
[presets.high]
label = "High"
order = 1
values = { sun_intensity = 3.0 }
"#;
    let child_toml = r#"
[pack]
id = "child-pack"
name = "Child"
extends = "parent-pack"
[capabilities]
tiers = [0, 1, 2, 3]
[settings.sun_intensity]
label = "Sun"
type = "float"
default = 2.8
min = 0.5
max = 8.0
[presets.high]
label = "High Cinema"
order = 1
values = { sun_intensity = 3.5 }
[presets.ultra]
label = "Ultra Cinema"
order = 2
values = { sun_intensity = 4.0 }
"#;
    let parent = PackManifest::parse_and_validate(parent_toml).expect("valid parent");
    let child = PackManifest::parse_and_validate(child_toml).expect("valid child");
    let merged = child.inherit_from(&parent);

    assert_eq!(merged.presets.len(), 3);
    assert_eq!(merged.presets["low"].label, "Low");
    assert_eq!(merged.presets["high"].label, "High Cinema");
    assert_eq!(
        merged.presets["high"].values["sun_intensity"],
        SettingValue::Float(3.5)
    );
    assert_eq!(merged.presets["ultra"].label, "Ultra Cinema");
}

#[test]
fn builtin_hyaloclast_default_manifest_defines_presets_and_tier_ranges() {
    let pack_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../src/main/resources/builtin-packs/hyaloclast-sunlit-vanilla.vsp");
    let manifest = PackManifest::load_from_dir(&pack_dir).expect("load builtin pack.toml");
    assert_eq!(manifest.settings["gi_bounce_strength"].min_tier, 1);
    assert_eq!(manifest.settings["gi_bounce_strength"].max_tier, 3);
    assert_eq!(manifest.settings["pt_max_bounces"].min_tier, 3);
    assert_eq!(manifest.settings["pt_max_bounces"].max_tier, 3);
    assert!(manifest.presets.contains_key("low"));
    assert!(manifest.presets.contains_key("medium"));
    assert!(manifest.presets.contains_key("high"));
    assert!(manifest.presets.contains_key("ultra"));
    assert_eq!(manifest.presets["low"].order, 0);
    assert_eq!(manifest.presets["medium"].order, 1);
    assert_eq!(manifest.presets["high"].order, 2);
    assert_eq!(manifest.presets["ultra"].order, 3);
}

#[test]
fn source_line_map_resolves_vsl_hook_and_include_lines_accurately() {
    let toml_str = r#"
[pack]
id = "line-map-test"
name = "Line Map Test"
version = "0.1.0"

[capabilities]
tiers = [0, 1, 2, 3]

[hooks]
post_grade = "hooks/post_grade.vsl"
"#;
    let manifest = PackManifest::parse_and_validate(toml_str).expect("valid pack.toml");

    let hook_src = "#include <vulkanshaders/v1/common.vsl>\n\
// @hook(stage = \"post_grade\", priority = 100)\n\
void my_grade(inout vec3 hdrColor, in vec2 uv, in VsPostContext post) {\n\
    hdrColor = hdrColor * 1.25;\n\
}\n";

    let comp_src = "#version 460\n\
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;\n\
struct VsPostContext { float exposure; float bloomIntensity; uint toneMapMode; };\n\
layout(set = 0, binding = 0, rgba16f) uniform image2D outColor;\n\
void main() {\n\
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);\n\
    vec3 hdr = vec3(1.0);\n\
    VsPostContext post = VsPostContext(1.0, 0.0, 0u);\n\
    vs_dispatch_hook_post_grade(hdr, vec2(0.5), post);\n\
    imageStore(outColor, px, vec4(hdr, 1.0));\n\
}\n";

    let loader = |path: &str| -> Option<String> {
        if path == "hooks/post_grade.vsl" {
            Some(hook_src.to_owned())
        } else {
            None
        }
    };

    let processed = VslPreprocessor::default()
        .preprocess(
            "grade.comp",
            comp_src,
            &manifest,
            RenderTier::Tier0Raster,
            &loader,
        )
        .expect("preprocess grade.comp");

    assert_eq!(
        processed.glsl_source.lines().count(),
        processed.source_map.len(),
        "SourceLineMap must stay 1:1 with preprocessed GLSL lines"
    );

    let target_line_idx = processed
        .glsl_source
        .lines()
        .position(|l| l.contains("hdrColor = hdrColor * 1.25;"))
        .expect("hook body line must be present in preprocessed output");
    let resolved = processed
        .source_map
        .resolve((target_line_idx + 1) as u32)
        .expect("preprocessed line must resolve in SourceLineMap");
    assert_eq!(resolved.file, "hooks/post_grade.vsl");
    assert_eq!(resolved.line, 4);
}

#[test]
fn compile_diagnostics_parses_shaderc_and_naga_errors_through_source_line_map() {
    let mut map = SourceLineMap::new();
    map.push_line("main.comp", 1);
    map.push_line("hooks/surface.vsl", 17);
    map.push_line("hooks/surface.vsl", 18);

    let shaderc_err =
        "ERROR: main.comp:2: 'undeclared_var' : undeclared identifier\n1 error generated.";
    let diags: Vec<CompileDiagnostic> =
        parse_compile_diagnostics(shaderc_err, "main.comp", Some(&map));
    assert_eq!(diags.len(), 1);
    assert_eq!(diags[0].file, "hooks/surface.vsl");
    assert_eq!(diags[0].line, 17);
    assert_eq!(diags[0].severity, "error");
    assert!(diags[0].message.contains("undeclared_var"));

    let naga_err = "error: unknown variable `foo`\n  ┌─ main.comp:3:9\n";
    let naga_diags = parse_compile_diagnostics(naga_err, "main.comp", Some(&map));
    assert_eq!(naga_diags.len(), 1);
    assert_eq!(naga_diags[0].file, "hooks/surface.vsl");
    assert_eq!(naga_diags[0].line, 18);
    assert_eq!(naga_diags[0].column, Some(9));
    assert_eq!(naga_diags[0].severity, "error");
}

#[test]
fn compile_stage_surfaces_exact_vsl_hook_filename_and_line_on_error() {
    let toml_str = r#"
[pack]
id = "broken-hook-test"
name = "Broken Hook Test"
version = "0.1.0"

[capabilities]
tiers = [0, 1, 2, 3]

[hooks]
post_grade = "hooks/broken_grade.vsl"
"#;
    let manifest = PackManifest::parse_and_validate(toml_str).expect("valid manifest");
    let broken_hook = "// line 1 comment\n\
// @hook(stage = \"post_grade\", priority = 100)\n\
void broken_grade(inout vec3 hdrColor, in vec2 uv, in VsPostContext post) {\n\
    hdrColor = totally_undefined_symbol_xyz * 2.0;\n\
}\n";

    let comp_src = r#"#version 460
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
struct VsPostContext { float exposure; float bloomIntensity; uint toneMapMode; };
layout(set = 0, binding = 0, rgba16f) uniform image2D outColor;
void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    vec3 hdr = vec3(1.0);
    VsPostContext post = VsPostContext(1.0, 0.0, 0u);
    vs_dispatch_hook_post_grade(hdr, vec2(0.5), post);
    imageStore(outColor, px, vec4(hdr, 1.0));
}
"#;

    let loader = |path: &str| -> Option<String> {
        if path == "hooks/broken_grade.vsl" {
            Some(broken_hook.to_owned())
        } else {
            None
        }
    };

    let compiler = VspCompiler::new(None);
    let diags = compiler
        .compile_stage_with_diagnostics(
            "post.comp",
            ShaderStageKind::Compute,
            comp_src,
            &manifest,
            RenderTier::Tier0Raster,
            HardwareShaderCaps::default(),
            &loader,
        )
        .expect_err("broken .vsl hook must fail compilation");

    assert!(
        diags
            .iter()
            .any(|d| d.file == "hooks/broken_grade.vsl" && d.line == 4),
        "expected diagnostic pointing to hooks/broken_grade.vsl:4, got {diags:?}"
    );
}

#[test]
fn spirv_cache_key_is_deterministic_and_sensitive_to_all_inputs() {
    let pack_hash = *blake3::hash(b"pack-v1").as_bytes();
    let driver_a = [0x11u8; 16];
    let driver_b = [0x22u8; 16];

    // Order of defines and spec constants must not change the key.
    let key_1 = SpirvCacheKey::builder(pack_hash, ShaderStageKind::RayGen)
        .entry_point("main")
        .driver_uuid(driver_a)
        .spec_constant(100, 1)
        .spec_constant(101, 42)
        .define("VS_HAS_SER", Some("1".to_owned()))
        .define("VS_HAS_OPACITY_MICROMAP", Some("1".to_owned()))
        .build();

    let key_2 = SpirvCacheKey::builder(pack_hash, ShaderStageKind::RayGen)
        .entry_point("main")
        .driver_uuid(driver_a)
        .spec_constant(101, 42)
        .spec_constant(100, 1)
        .define("VS_HAS_OPACITY_MICROMAP", Some("1".to_owned()))
        .define("VS_HAS_SER", Some("1".to_owned()))
        .build();

    assert_eq!(key_1, key_2);
    assert_eq!(key_1.to_hex().len(), 64);

    // Changing any single component must produce a distinct key.
    let diff_stage = SpirvCacheKey::builder(pack_hash, ShaderStageKind::Compute)
        .entry_point("main")
        .driver_uuid(driver_a)
        .spec_constant(100, 1)
        .spec_constant(101, 42)
        .define("VS_HAS_SER", Some("1".to_owned()))
        .define("VS_HAS_OPACITY_MICROMAP", Some("1".to_owned()))
        .build();
    assert_ne!(key_1, diff_stage);

    let diff_driver = SpirvCacheKey::builder(pack_hash, ShaderStageKind::RayGen)
        .entry_point("main")
        .driver_uuid(driver_b)
        .spec_constant(100, 1)
        .spec_constant(101, 42)
        .define("VS_HAS_SER", Some("1".to_owned()))
        .define("VS_HAS_OPACITY_MICROMAP", Some("1".to_owned()))
        .build();
    assert_ne!(key_1, diff_driver);

    let diff_entry = SpirvCacheKey::builder(pack_hash, ShaderStageKind::RayGen)
        .entry_point("main_ser")
        .driver_uuid(driver_a)
        .spec_constant(100, 1)
        .spec_constant(101, 42)
        .define("VS_HAS_SER", Some("1".to_owned()))
        .define("VS_HAS_OPACITY_MICROMAP", Some("1".to_owned()))
        .build();
    assert_ne!(key_1, diff_entry);

    let diff_spec = SpirvCacheKey::builder(pack_hash, ShaderStageKind::RayGen)
        .entry_point("main")
        .driver_uuid(driver_a)
        .spec_constant(100, 0)
        .spec_constant(101, 42)
        .define("VS_HAS_SER", Some("1".to_owned()))
        .define("VS_HAS_OPACITY_MICROMAP", Some("1".to_owned()))
        .build();
    assert_ne!(key_1, diff_spec);
}

#[test]
fn spirv_disk_cache_roundtrip_and_header_version_validation() {
    let unique_dir = std::env::temp_dir().join(format!(
        "vsp_disk_cache_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let cache = SpirvDiskCache::new(Some(unique_dir.clone()));

    let key = SpirvCacheKey::builder(
        *blake3::hash(b"roundtrip").as_bytes(),
        ShaderStageKind::Compute,
    )
    .driver_uuid([0xAAu8; 16])
    .build();

    // Minimal valid SPIR-V header (magic + version + generator + bound + schema + 2 words)
    let sample_words: Vec<u32> = vec![0x0723_0203, 0x0001_0600, 0, 1, 0, 0x0002_0011, 1];
    assert!(cache.try_load_key(&key).is_none());

    cache.store_key(&key, &sample_words);
    let loaded = cache
        .try_load_key(&key)
        .expect("cached SPIR-V blob must load back");
    assert_eq!(loaded, sample_words);

    // Verify magic header and version bytes on disk
    let cache_file = cache.cache_path_for_key(&key).expect("cache path");
    let mut raw = std::fs::read(&cache_file).expect("read cache file");
    assert_eq!(&raw[0..8], &SpirvDiskCache::CACHE_MAGIC);
    assert_eq!(
        u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]),
        SpirvDiskCache::CACHE_VERSION
    );

    // Bumping version in header must invalidate the cache entry cleanly
    let bad_ver = (SpirvDiskCache::CACHE_VERSION + 1).to_le_bytes();
    raw[8..12].copy_from_slice(&bad_ver);
    std::fs::write(&cache_file, &raw).expect("write mutated version");
    assert!(
        cache.try_load_key(&key).is_none(),
        "mismatched cache version must be rejected"
    );

    let _ = std::fs::remove_dir_all(&unique_dir);
}

#[test]
fn ser_and_omm_defines_and_spirv_1_6_compilation() {
    let ser_ext_caps = HardwareShaderCaps {
        ser_ext: true,
        ser_nv: false,
        ada_or_newer: true,
        high_end_gpu: true,
        opacity_micromap: true,
    };
    assert!(ser_ext_caps.has_ser());
    let defines = ser_ext_caps.to_glsl_defines();
    assert!(defines.contains("#define VS_HAS_SER 1"));
    assert!(defines.contains("#define VS_HAS_SER_EXT 1"));
    assert!(defines.contains("#define VS_HAS_OPACITY_MICROMAP 1"));

    let macro_defs = VspCompiler::hardware_caps_defines(ser_ext_caps);
    assert!(
        macro_defs
            .iter()
            .any(|(k, v)| k == "VS_HAS_SER" && v.as_deref() == Some("1"))
    );
    assert!(
        macro_defs
            .iter()
            .any(|(k, v)| k == "VS_HAS_OPACITY_MICROMAP" && v.as_deref() == Some("1"))
    );

    let ser_rgen = r#"#version 460
#extension GL_EXT_ray_tracing : require
#if defined(VS_HAS_SER) && defined(VS_HAS_SER_EXT)
#extension GL_EXT_shader_invocation_reorder : require
#elif defined(VS_HAS_SER)
#extension GL_NV_shader_invocation_reorder : require
#endif

layout(set = 1, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(location = 0) rayPayloadEXT vec4 hitPayload;

void main() {
#if defined(VS_HAS_SER) && defined(VS_HAS_SER_EXT)
    hitObjectEXT hitObj;
    hitObjectTraceRayEXT(
        hitObj,
        topLevelAS,
        gl_RayFlagsOpaqueEXT,
        0xFFu,
        0u,
        0u,
        0u,
        vec3(0.0),
        0.01,
        vec3(0.0, 1.0, 0.0),
        64.0,
        0
    );
    reorderThreadEXT(hitObj, 1u, 4u);
    hitObjectExecuteShaderEXT(hitObj, 0);
#elif defined(VS_HAS_SER)
    hitObjectNV hitObj;
    hitObjectTraceRayNV(
        hitObj,
        topLevelAS,
        gl_RayFlagsOpaqueEXT,
        0xFFu,
        0u,
        0u,
        0u,
        vec3(0.0),
        0.01,
        vec3(0.0, 1.0, 0.0),
        64.0,
        0
    );
    reorderThreadNV(hitObj, 1u, 4u);
    hitObjectExecuteShaderNV(hitObj, 0);
#else
    traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xFFu, 0u, 0u, 0u, vec3(0.0), 0.01, vec3(0.0, 1.0, 0.0), 64.0, 0);
#endif
}
"#;
    let words = match VspCompiler::compile_glsl_to_spirv_with_caps(
        "ser_test.rgen",
        ShaderStageKind::RayGen,
        ser_rgen,
        ser_ext_caps,
        None,
    ) {
        Ok(words) => words,
        Err(error)
            if error
                .to_string()
                .contains("GL_EXT_shader_invocation_reorder") =>
        {
            let nv_caps = HardwareShaderCaps {
                ser_nv: true,
                opacity_micromap: true,
                ..Default::default()
            };
            VspCompiler::compile_glsl_to_spirv_with_caps(
                "ser_test.rgen",
                ShaderStageKind::RayGen,
                ser_rgen,
                nv_caps,
                None,
            )
            .expect("SER NV raygen shader must compile cleanly to SPIR-V 1.6 with VS_HAS_SER")
        }
        Err(error) => panic!("SER raygen shader failed: {error}"),
    };
    assert!(words.len() > 40);
    assert_eq!(words[0], 0x0723_0203);
}

#[test]
fn manifest_reports_missing_required_fields_and_validates_presets_and_pack_directory() {
    let missing_pack_table = "[capabilities]\ntiers = [0]\n";
    let err = PackManifest::parse_and_validate(missing_pack_table).expect_err("missing [pack]");
    assert!(
        matches!(&err, VspError::ManifestValidation { reason, .. } if reason.contains("[pack]")),
        "got {err:?}"
    );

    let missing_id = "[pack]\nname = \"No Id\"\n";
    let err = PackManifest::parse_and_validate(missing_id).expect_err("missing pack.id");
    assert!(
        matches!(&err, VspError::ManifestValidation { reason, .. } if reason.contains("pack.id")),
        "got {err:?}"
    );

    let missing_name = "[pack]\nid = \"no-name\"\n";
    let err = PackManifest::parse_and_validate(missing_name).expect_err("missing pack.name");
    assert!(
        matches!(&err, VspError::ManifestValidation { reason, .. } if reason.contains("pack.name")),
        "got {err:?}"
    );

    let missing_setting_label = r#"
[pack]
id = "missing-label"
name = "Missing Label"
[settings.exposure]
type = "float"
default = 1.0
"#;
    let err = PackManifest::parse_and_validate(missing_setting_label)
        .expect_err("missing settings.exposure.label");
    assert!(
        matches!(&err, VspError::ManifestValidation { reason, .. } if reason.contains("settings.exposure.label")),
        "got {err:?}"
    );

    // Preset ordering & option override resolution
    let preset_toml = r#"
[pack]
id = "preset-resolution"
name = "Preset Resolution"
[settings.exposure]
label = "Exposure"
type = "float"
default = 1.0
min = 0.5
max = 3.0
[settings.bloom]
label = "Bloom"
type = "bool"
default = false
[presets.ultra]
label = "Ultra"
order = 3
values = { exposure = 1.4, bloom = true }
[presets.low]
label = "Low"
order = 0
values = { exposure = 0.8 }
"#;
    let manifest = PackManifest::parse_and_validate(preset_toml).expect("valid preset manifest");
    let ordered = manifest.ordered_presets();
    assert_eq!(ordered[0].0, "low");
    assert_eq!(ordered[1].0, "ultra");
    let low_vals = manifest
        .resolve_preset_values("Low")
        .expect("case-insensitive preset resolution");
    assert_eq!(low_vals["exposure"], SettingValue::Float(0.8));
    assert_eq!(low_vals["bloom"], SettingValue::Bool(false));

    // Directory validation: missing hook file triggers an error, missing presets triggers a warning
    let tmp_dir = std::env::temp_dir().join(format!(
        "vsp_validate_dir_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmp_dir).expect("create temp pack dir");
    std::fs::write(
        tmp_dir.join("pack.toml"),
        r#"
[pack]
id = "dir-check"
name = "Dir Check"
[hooks]
surface = "hooks/missing_surface.vsl"
[settings.exposure]
label = "Exposure"
type = "float"
default = 1.0
"#,
    )
    .expect("write pack.toml");

    let report = validate_pack_directory(&tmp_dir);
    assert!(!report.is_valid());
    assert!(
        report
            .errors
            .iter()
            .any(|e| e.message.contains("hooks/missing_surface.vsl"))
    );
    assert!(
        report
            .warnings
            .iter()
            .any(|w| w.field.as_deref() == Some("presets"))
    );

    // Create the missing hook file and re-validate
    std::fs::create_dir_all(tmp_dir.join("hooks")).expect("create hooks dir");
    std::fs::write(tmp_dir.join("hooks/missing_surface.vsl"), "// ok\n").expect("write hook");
    let report_ok = validate_pack_directory(&tmp_dir);
    assert!(
        report_ok.is_valid(),
        "unexpected errors: {:?}",
        report_ok.errors
    );

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn pt_max_bounces_specialization_constant_is_not_shadowed_by_fallback_macro() {
    let pack_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../src/main/resources/builtin-packs/relight-vanilla.vsp");
    let compiler = VspCompiler::new();
    let artifacts = compiler
        .compile_pack_dir_with_caps(
            &pack_dir,
            RenderTier::Tier3PathTracing,
            HardwareShaderCaps {
                ray_tracing_pipeline: true,
                ray_query: true,
                high_end_gpu: true,
                ..Default::default()
            },
        )
        .expect("relight-vanilla Tier 3 shaders must compile cleanly");
    let rgen = artifacts
        .iter()
        .find(|a| a.stage == ShaderStageKind::RayGen)
        .expect("relight-vanilla must emit a RayGen artifact");
    assert!(
        !rgen.reflection.spec_constant_ids.is_empty(),
        "pathtrace.rgen must preserve live VkSpecializationInfo constant IDs (pt_max_bounces must not be shadowed by #define)"
    );
}
