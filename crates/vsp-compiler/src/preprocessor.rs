use crate::{
    VspError,
    compiler::ShaderStageKind,
    linker::SpirvHookLinker,
    manifest::{HookStage, PackManifest, RenderTier, SettingValueType},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::Path,
};

/// Convenience facade around [`VslPreprocessor`] used by the native core's G-Buffer shader
/// preprocessing FFI. Resolves `#include` against the pack directory and returns the expanded,
/// hook-injected GLSL source as a plain `String`.
pub struct Preprocessor;

impl Preprocessor {
    /// Preprocesses a standalone GLSL source string with the given pack manifest and tier.
    /// Includes are resolved from `<pack_dir>/<inc>` then `<pack_dir>/include/<inc>`, matching
    /// `VspCompiler::compile_pack_dir`'s loader exactly. Built-in `.vsl` headers are resolved by
    /// `VslPreprocessor` itself.
    pub fn preprocess_source_str(
        source: &str,
        pack_dir: &Path,
        manifest: &PackManifest,
        tier: RenderTier,
    ) -> Result<String, VspError> {
        let loader = |inc: &str| -> Option<String> {
            let direct = pack_dir.join(inc);
            if direct.is_file() {
                return fs::read_to_string(direct).ok();
            }
            fs::read_to_string(pack_dir.join("include").join(inc)).ok()
        };
        let source_name = Self::infer_stage_filename(source);
        let processed =
            VslPreprocessor::default().preprocess(source_name, source, manifest, tier, &loader)?;
        Ok(processed.glsl_source)
    }

    /// Preprocesses a standalone GLSL source string across an inheritance chain of pack directories
    /// (child first, ancestors after).
    pub fn preprocess_source_str_with_chain(
        source: &str,
        chain_dirs: &[std::path::PathBuf],
        manifest: &PackManifest,
        tier: RenderTier,
    ) -> Result<String, VspError> {
        let loader = |inc: &str| -> Option<String> {
            for dir in chain_dirs {
                let direct = dir.join(inc);
                if direct.is_file() {
                    return fs::read_to_string(direct).ok();
                }
                let in_include = dir.join("include").join(inc);
                if in_include.is_file() {
                    return fs::read_to_string(in_include).ok();
                }
            }
            None
        };
        let source_name = Self::infer_stage_filename(source);
        let processed =
            VslPreprocessor::default().preprocess(source_name, source, manifest, tier, &loader)?;
        Ok(processed.glsl_source)
    }

    /// Preprocesses a raster G-Buffer GLSL shader (`.vsh` / `.fsh`) across an inheritance chain
    /// of pack directories:
    /// - Synthesizes `#define VS_PARAM_<KEY> (vsMotion.packParams[idx / 4].<xyzw>)` instead of `vsWorld`.
    /// - Synthesizes inline `#define VS_SPEC_<KEY> <override_or_default>` without `layout(constant_id = ...)`.
    /// - Substitutes `// @hook <stage>` markers with extracted, namespace-isolated pack hook bodies.
    pub fn preprocess_raster_source_with_chain(
        source: &str,
        chain_dirs: &[&Path],
        manifest: &PackManifest,
        tier: u32,
        stage_is_fragment: bool,
        spec_overrides: &BTreeMap<String, f32>,
    ) -> Result<String, String> {
        let loader = |inc: &str| -> Option<String> {
            for dir in chain_dirs {
                let direct = dir.join(inc);
                if direct.is_file() {
                    return fs::read_to_string(direct).ok();
                }
                let in_include = dir.join("include").join(inc);
                if in_include.is_file() {
                    return fs::read_to_string(in_include).ok();
                }
            }
            None
        };
        let render_tier = RenderTier::from_index(tier).unwrap_or(RenderTier::Tier0Raster);
        let source_name = if stage_is_fragment {
            "raster_shader.fsh"
        } else {
            "raster_shader.vsh"
        };
        VslPreprocessor::default()
            .preprocess_raster(
                source_name,
                source,
                manifest,
                render_tier,
                stage_is_fragment,
                spec_overrides,
                &loader,
            )
            .map_err(|e| e.to_string())
    }

    fn infer_stage_filename(source: &str) -> &'static str {
        if source.contains("local_size_x") || source.contains("gl_GlobalInvocationID") {
            "shader.comp"
        } else if source.contains("gl_Position") || source.contains("gl_VertexIndex") {
            "raster_shader.vsh"
        } else {
            "raster_shader.fsh"
        }
    }
}

/// Free-function wrapper for [`Preprocessor::preprocess_raster_source_with_chain`].
pub fn preprocess_raster_source_with_chain(
    source: &str,
    chain_dirs: &[&Path],
    manifest: &PackManifest,
    tier: u32,
    stage_is_fragment: bool,
    spec_overrides: &BTreeMap<String, f32>,
) -> Result<String, String> {
    Preprocessor::preprocess_raster_source_with_chain(
        source,
        chain_dirs,
        manifest,
        tier,
        stage_is_fragment,
        spec_overrides,
    )
}

/// Parsed `@hook(stage = "surface", priority = 100)` annotation attached to a `.vsl` function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookAnnotation {
    pub stage: HookStage,
    pub priority: i32,
    pub function_name: String,
}

/// Hardware capability flags propagated from the active GPU into GLSL preprocessor defines
/// (`VS_HAS_SER_EXT`, `VS_HAS_SER_NV`, `VS_GPU_ADA_PLUS`, `VS_HIGH_END_GPU`,
/// `VS_HAS_OPACITY_MICROMAP`) and folded into the SPIR-V disk cache key.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct HardwareShaderCaps {
    pub ser_ext: bool,
    pub ser_nv: bool,
    pub ada_or_newer: bool,
    pub high_end_gpu: bool,
    pub opacity_micromap: bool,
}

impl HardwareShaderCaps {
    /// Compact bitmask representing all capability flags for cache hashing.
    pub const fn cache_bits(self) -> u8 {
        (self.ser_ext as u8)
            | ((self.ser_nv as u8) << 1)
            | ((self.ada_or_newer as u8) << 2)
            | ((self.high_end_gpu as u8) << 3)
            | ((self.opacity_micromap as u8) << 4)
    }

    /// Returns `true` when either cross-vendor (`VK_EXT_ray_tracing_invocation_reorder`) or
    /// vendor-specific (`VK_NV_ray_tracing_invocation_reorder`) SER is enabled.
    pub const fn has_ser(self) -> bool {
        self.ser_ext || self.ser_nv
    }

    /// Formats the active hardware capability `#define` lines injected immediately after `#version`.
    pub fn to_glsl_defines(self) -> String {
        let mut out = String::new();
        if self.ser_ext {
            out.push_str("#define VS_HAS_SER_EXT 1\n");
        }
        if self.ser_nv {
            out.push_str("#define VS_HAS_SER_NV 1\n");
        }
        if self.has_ser() {
            out.push_str("#define VS_HAS_SER 1\n");
        }
        if self.ada_or_newer {
            out.push_str("#define VS_GPU_ADA_PLUS 1\n");
        }
        if self.high_end_gpu {
            out.push_str("#define VS_HIGH_END_GPU 1\n");
        }
        if self.opacity_micromap {
            out.push_str("#define VS_HAS_OPACITY_MICROMAP 1\n");
        }
        out
    }
}

/// Original `.vsl` / GLSL file path and 1-based line number before preprocessing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceLocation {
    pub file: String,
    pub line: u32,
}

/// Maps 1-based line numbers in a preprocessed GLSL translation unit back to their originating
/// `.vsl` / GLSL file path and 1-based line number.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceLineMap {
    /// Entry at index `i` corresponds to 1-based preprocessed line `i + 1`.
    lines: Vec<SourceLocation>,
}

impl SourceLineMap {
    pub fn new() -> Self {
        Self { lines: Vec::new() }
    }

    pub fn from_single_file(file: &str, source: &str) -> Self {
        let count = source.lines().count();
        let mut lines = Vec::with_capacity(count);
        for idx in 0..count {
            lines.push(SourceLocation {
                file: file.to_owned(),
                line: (idx + 1) as u32,
            });
        }
        Self { lines }
    }

    pub fn push_line(&mut self, file: impl Into<String>, line: u32) {
        self.lines.push(SourceLocation {
            file: file.into(),
            line,
        });
    }

    pub fn push_range(&mut self, file: &str, start_line: u32, count: usize) {
        for offset in 0..count {
            self.lines.push(SourceLocation {
                file: file.to_owned(),
                line: start_line + offset as u32,
            });
        }
    }

    /// Resolves a 1-based preprocessed GLSL line number to its original file and 1-based line.
    pub fn resolve(&self, preprocessed_line: u32) -> Option<&SourceLocation> {
        let idx = preprocessed_line.checked_sub(1)? as usize;
        self.lines.get(idx)
    }

    pub fn len(&self) -> usize {
        self.lines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    pub fn entries(&self) -> &[SourceLocation] {
        &self.lines
    }

    pub fn extend_from(&mut self, other: &Self) {
        self.lines.extend(other.lines.iter().cloned());
    }

    pub fn insert_before_line(&mut self, zero_based_idx: usize, inserted: &[SourceLocation]) {
        let idx = zero_based_idx.min(self.lines.len());
        self.lines.splice(idx..idx, inserted.iter().cloned());
    }
}

#[derive(Debug, Clone)]
pub struct PreprocessedShaderSource {
    pub glsl_source: String,
    pub discovered_hooks: Vec<HookAnnotation>,
    pub content_hash: [u8; 32],
    pub source_map: SourceLineMap,
}

/// Extracted hook function body and its namespace-isolated helper definitions for raster G-Buffer
/// or compute/RT injection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedHookBlock {
    pub stage: HookStage,
    pub priority: i32,
    pub isolated_helper_code: String,
    pub isolated_body: String,
}

/// Preprocesses `.vsl` / `.vsh` / `.fsh` / `.comp` / `.rgen` / `.rchit` / `.rmiss` / `.rahit` files:
/// 1. Expands `#include <...>` directives with cycle/depth guards and unit-wide deduplication.
/// 2. Links shared `[pipeline.modules]` and active pack `.vsl` hook bodies (`manifest.hooks`) with
///    full lexer-driven identifier namespace isolation (`vs_ns_<pack>_<idx>_<ident>`) so helper
///    functions, constants, and hook entrypoints never collide across multiple `.vsl` files or
///    layered packs, ignoring braces/tokens inside `//` or `/* ... */` comments and string literals.
/// 3. Synthesizes `vs_dispatch_hook_<stage>` composite dispatchers via
///    `SpirvHookLinker::build_composite_dispatch_glsl` for all 16 hook stages.
/// 4. Synthesizes `VS_PARAM_<NAME>` macros (reading either `vsWorld.packParams[idx / 4][idx % 4]` or
///    `layout(constant_id = N) const ...`) and named resource accessors (`vsSamplePackTexture_<name>`).
pub struct VslPreprocessor {
    max_include_depth: usize,
}

impl Default for VslPreprocessor {
    fn default() -> Self {
        Self {
            max_include_depth: 12,
        }
    }
}

impl VslPreprocessor {
    pub fn new(max_include_depth: usize) -> Self {
        Self { max_include_depth }
    }

    pub fn preprocess<F>(
        &self,
        source_name: &str,
        raw_source: &str,
        manifest: &PackManifest,
        tier: RenderTier,
        include_loader: &F,
    ) -> Result<PreprocessedShaderSource, VspError>
    where
        F: Fn(&str) -> Option<String>,
    {
        self.preprocess_with_caps(
            source_name,
            raw_source,
            manifest,
            tier,
            HardwareShaderCaps::default(),
            include_loader,
        )
    }

    pub fn preprocess_with_caps<F>(
        &self,
        source_name: &str,
        raw_source: &str,
        manifest: &PackManifest,
        tier: RenderTier,
        hw_caps: HardwareShaderCaps,
        include_loader: &F,
    ) -> Result<PreprocessedShaderSource, VspError>
    where
        F: Fn(&str) -> Option<String>,
    {
        let mut already_included = HashSet::new();
        let (mut expanded, mut source_map) = self.expand_includes_mapped(
            source_name,
            raw_source,
            0,
            &mut already_included,
            include_loader,
            false,
        )?;

        let shader_stage = Path::new(source_name)
            .extension()
            .and_then(|e| e.to_str())
            .and_then(ShaderStageKind::from_extension);

        // Link shared `[pipeline.modules]` and compatible pack `.vsl` hooks with namespace isolation
        let mut discovered_hooks = Self::scan_hook_annotations(&expanded);
        if !source_name.ends_with(".vsl") {
            let has_vsl_context = already_included.contains("vulkanshaders/v1/common.vsl")
                || Self::contains_identifier(&expanded, "vsWorld");

            let mut hook_sources = String::new();
            let mut hook_map = SourceLineMap::new();
            let mut linked_hooks = Vec::new();
            let sanitized_pack = Self::sanitize_ident(&manifest.pack.id);

            let mut base_hook_includes = already_included.clone();
            base_hook_includes.insert("vulkanshaders/v1/common.vsl".to_owned());
            base_hook_includes.insert("vulkanshaders/v1/lighting.vsl".to_owned());
            base_hook_includes.insert("vulkanshaders/v1/restir.vsl".to_owned());

            // 1. Expand any shared `[pipeline.modules]` first
            for mod_path in &manifest.pipeline.modules {
                if let Some(raw_mod) = include_loader(mod_path)
                    && let Ok((exp_mod, exp_mod_map)) = self.expand_includes_mapped(
                        mod_path,
                        &raw_mod,
                        1,
                        &mut base_hook_includes,
                        include_loader,
                        false,
                    )
                {
                    let (stripped_mod, stripped_map) =
                        Self::strip_header_guards_and_includes_mapped(&exp_mod, &exp_mod_map);
                    if Self::hook_body_dependencies_satisfied(&stripped_mod, &expanded) {
                        hook_sources.push_str("// --- shared pipeline module: ");
                        hook_sources.push_str(mod_path);
                        hook_sources.push_str(" ---\n");
                        hook_map.push_line(mod_path, 1);
                        hook_sources.push_str(&stripped_mod);
                        hook_map.extend_from(&stripped_map);
                    }
                }
            }

            struct PreparedHookModule {
                rel_path: String,
                ns_prefix: String,
                isolated_body: String,
                isolated_map: SourceLineMap,
                annotations: Vec<HookAnnotation>,
            }

            let mut prepared_modules = Vec::new();
            for (hook_idx, (stage_key, rel_path)) in manifest.hooks.iter().enumerate() {
                let Some(raw_hook) = include_loader(rel_path) else {
                    continue;
                };
                let mut hook_includes = base_hook_includes.clone();
                let Ok((exp_hook, exp_hook_map)) = self.expand_includes_mapped(
                    rel_path,
                    &raw_hook,
                    1,
                    &mut hook_includes,
                    include_loader,
                    false,
                ) else {
                    continue;
                };
                let ns_prefix = format!(
                    "vs_ns_{}_{}_{}_",
                    sanitized_pack,
                    Self::sanitize_ident(stage_key),
                    hook_idx
                );
                let (isolated_body, isolated_map, annotations) =
                    Self::isolate_hook_module_mapped(&exp_hook, &exp_hook_map, &ns_prefix);
                if annotations.is_empty()
                    || !Self::hook_body_dependencies_satisfied(&isolated_body, &expanded)
                {
                    continue;
                }
                prepared_modules.push(PreparedHookModule {
                    rel_path: rel_path.clone(),
                    ns_prefix,
                    isolated_body,
                    isolated_map,
                    annotations,
                });
            }

            let mut active_stages = Vec::new();
            for &stage in HookStage::ALL {
                let sig_ok = Self::stage_signature_types_present(&expanded, stage);
                if !sig_ok {
                    continue;
                }
                let referenced =
                    Self::contains_identifier(&expanded, stage.dispatch_function_name());
                let stage_ok = Self::is_hook_stage_compatible_with_shader(stage, shader_stage);
                let has_compatible_impl = prepared_modules
                    .iter()
                    .any(|m| m.annotations.iter().any(|a| a.stage == stage));

                if referenced || (stage_ok && (has_vsl_context || has_compatible_impl)) {
                    active_stages.push(stage);
                }
            }

            for module in prepared_modules {
                let module_needed = module
                    .annotations
                    .iter()
                    .any(|ann| active_stages.contains(&ann.stage));
                if !module_needed {
                    continue;
                }
                let mut anns = module.annotations;
                linked_hooks.append(&mut anns);
                hook_sources.push_str("// --- linked hook file: ");
                hook_sources.push_str(&module.rel_path);
                hook_sources.push_str(" (namespace: ");
                hook_sources.push_str(&module.ns_prefix);
                hook_sources.push_str(") ---\n");
                hook_map.push_line(&module.rel_path, 1);
                hook_sources.push_str(&module.isolated_body);
                hook_map.extend_from(&module.isolated_map);
            }

            if !active_stages.is_empty() {
                discovered_hooks.extend(linked_hooks);
                let composite_calls =
                    SpirvHookLinker::from_hook_annotations(&manifest.pack.id, &discovered_hooks);
                hook_sources.push_str("// --- synthesized hook dispatchers ---\n");
                hook_map.push_line("<generated:hooks>", 1);
                for stage in active_stages {
                    let disp =
                        SpirvHookLinker::build_composite_dispatch_glsl(stage, &composite_calls);
                    let disp_lines = disp.lines().count();
                    hook_sources.push_str(&disp);
                    hook_map.push_range("<generated:hooks>", 1, disp_lines);
                }
                expanded = Self::inject_before_main_mapped(
                    &expanded,
                    &mut source_map,
                    &hook_sources,
                    &hook_map,
                );
            }
        }

        let hw_defines = hw_caps.to_glsl_defines();
        let with_hw_caps = if hw_defines.is_empty() {
            expanded
        } else {
            Self::inject_after_version_mapped(
                &expanded,
                &mut source_map,
                &hw_defines,
                "<generated:hw_caps>",
            )
        };
        let generated_prelude =
            Self::synthesize_pack_prelude(manifest, tier, &with_hw_caps, shader_stage);
        let final_glsl = Self::inject_after_version_and_extensions_mapped(
            &with_hw_caps,
            &mut source_map,
            &generated_prelude,
            "<generated:prelude>",
        );

        let mut hasher = blake3::Hasher::new();
        hasher.update(&[tier.as_u32() as u8, hw_caps.cache_bits()]);
        hasher.update(final_glsl.as_bytes());
        let content_hash = *hasher.finalize().as_bytes();

        Ok(PreprocessedShaderSource {
            glsl_source: final_glsl,
            discovered_hooks,
            content_hash,
            source_map,
        })
    }

    /// Preprocesses a raster G-Buffer shader using `vsMotion.packParams` for `VS_PARAM_*`,
    /// inline `#define VS_SPEC_*` for specialization constants (since RenderPearl raster
    /// pipelines do not pass `VkSpecializationInfo`), and `// @hook <stage>` marker injection.
    #[allow(clippy::too_many_arguments)]
    pub fn preprocess_raster<F>(
        &self,
        source_name: &str,
        raw_source: &str,
        manifest: &PackManifest,
        tier: RenderTier,
        stage_is_fragment: bool,
        spec_overrides: &BTreeMap<String, f32>,
        include_loader: &F,
    ) -> Result<String, VspError>
    where
        F: Fn(&str) -> Option<String>,
    {
        let mut already_included = HashSet::new();
        let mut expanded = self.expand_includes_with_options(
            source_name,
            raw_source,
            0,
            &mut already_included,
            include_loader,
            true,
        )?;

        let shader_stage = Some(if stage_is_fragment {
            ShaderStageKind::Fragment
        } else {
            ShaderStageKind::Vertex
        });

        // Discover `// @hook <stage>` markers present in the raster shader.
        let mut marker_stages: Vec<HookStage> = Vec::new();
        for line in expanded.lines() {
            if let Some(stage_name) = line.trim().strip_prefix("// @hook ")
                && let Some(stage) = HookStage::from_name(stage_name.trim())
                && Self::is_hook_stage_compatible_with_shader(stage, shader_stage)
                && !marker_stages.contains(&stage)
            {
                marker_stages.push(stage);
            }
        }

        let mut hook_prelude = String::new();
        let mut applied_stages: Vec<HookStage> = Vec::new();

        if !marker_stages.is_empty() {
            let sanitized_pack = Self::sanitize_ident(&manifest.pack.id);
            let mut base_hook_includes = already_included.clone();
            base_hook_includes.insert("vulkanshaders/v1/common.vsl".to_owned());
            base_hook_includes.insert("vulkanshaders/v1/lighting.vsl".to_owned());
            base_hook_includes.insert("vulkanshaders/v1/restir.vsl".to_owned());

            let mut per_stage_blocks: BTreeMap<String, Vec<ExtractedHookBlock>> = BTreeMap::new();
            for (hook_idx, (stage_key, rel_path)) in manifest.hooks.iter().enumerate() {
                let Some(raw_hook) = include_loader(rel_path) else {
                    continue;
                };
                let mut hook_includes = base_hook_includes.clone();
                let Ok(exp_hook) = self.expand_includes(
                    rel_path,
                    &raw_hook,
                    1,
                    &mut hook_includes,
                    include_loader,
                ) else {
                    continue;
                };
                for &stage in &marker_stages {
                    let ns_prefix = format!(
                        "vs_ns_{}_{}_{}_{}_",
                        sanitized_pack,
                        Self::sanitize_ident(stage_key),
                        stage.as_str(),
                        hook_idx
                    );
                    let Some(block) = Self::extract_hook_helper_and_body_lexed(
                        &exp_hook,
                        stage,
                        Some(&ns_prefix),
                    ) else {
                        continue;
                    };
                    if !Self::hook_body_dependencies_satisfied_raster(
                        &block.isolated_body,
                        &expanded,
                    ) {
                        continue;
                    }
                    per_stage_blocks
                        .entry(stage.as_str().to_owned())
                        .or_default()
                        .push(block);
                }
            }

            if !per_stage_blocks.is_empty() {
                hook_prelude.push_str(
                    "// --- Hyaloclast .vsl raster hook bodies (generated from active pack) ---\n",
                );
                hook_prelude.push_str(
                    "struct VsHookWorld { float gameTimeSec; float rain; float heldLight; float thunder; float eyeInFluid; vec3 sunDir; };\n",
                );
                hook_prelude.push_str("VsHookWorld vsHookWorld;\n");
                hook_prelude.push_str("#define vsWorld vsHookWorld\n");
                // The native wind compute scheduler state is not mirrored in the raster UBOs.
                hook_prelude.push_str("#define VS_WIND_COMPUTE_ACTIVE 0\n");
                if per_stage_blocks.contains_key("water_waves") {
                    hook_prelude.push_str("#define VS_HOOK_WATER_WAVES_BOUND 1\n");
                }
                if !Self::contains_identifier(&expanded, "VsVertexInput") {
                    hook_prelude.push_str("struct VsVertexInput { uint materialFlags; vec3 worldPos; vec2 uv0; vec2 midUv; };\n");
                }
                if !Self::contains_identifier(&expanded, "VsSurface") {
                    hook_prelude.push_str(
                        "struct VsSurface { vec3 albedo; vec3 normal; float roughness; float metallic; float emission; float ao; float subsurface; float f0; };\n",
                    );
                }

                for &stage in &marker_stages {
                    let Some(blocks) = per_stage_blocks.get_mut(stage.as_str()) else {
                        continue;
                    };
                    blocks.sort_by_key(|b| b.priority);
                    applied_stages.push(stage);

                    let mut combined_body = String::new();
                    for block in blocks.iter() {
                        if !block.isolated_helper_code.is_empty() {
                            hook_prelude.push_str(&block.isolated_helper_code);
                            hook_prelude.push('\n');
                        }
                        combined_body.push_str("{\n");
                        combined_body.push_str(&block.isolated_body);
                        combined_body.push_str("\n}\n");
                    }

                    match stage {
                        HookStage::VertexDisplace => {
                            hook_prelude.push_str(&format!(
                                "void vsHookVertexDisplace(inout vec3 localPos, VsVertexInput v) {{\n{combined_body}}}\n"
                            ));
                        }
                        HookStage::WaterWaves => {
                            hook_prelude.push_str(&format!(
                                "void vsHookWaterWaves(inout float strength, inout float speed, inout float scale) {{\n{combined_body}}}\n"
                            ));
                        }
                        HookStage::Surface => {
                            hook_prelude.push_str(&format!(
                                "void vsHookSurface(inout VsSurface s, vec3 worldPos, uint flags, uint bounce) {{\n{combined_body}}}\n"
                            ));
                        }
                        HookStage::GbufferPack => {
                            hook_prelude.push_str(&format!(
                                "void vsHookGbufferPack(in VsSurface s, inout vec4 outAlbedo, inout vec4 outNormalRoughness, inout vec4 outMaterialPbr) {{\n{combined_body}}}\n"
                            ));
                        }
                        other => {
                            let fn_name = format!("vsHookRaster_{}", other.as_str());
                            hook_prelude
                                .push_str(&format!("void {fn_name}() {{\n{combined_body}}}\n"));
                        }
                    }
                }
                hook_prelude.push_str("// --- end .vsl raster hook bodies ---\n");

                for stage in &applied_stages {
                    let marker = format!("// @hook {}", stage.as_str());
                    let call = Self::raster_hook_call(*stage);
                    expanded = expanded.replace(&marker, &call);
                }
            }
        }

        let mut combined_check = String::with_capacity(expanded.len() + hook_prelude.len() + 2);
        combined_check.push_str(&expanded);
        combined_check.push('\n');
        combined_check.push_str(&hook_prelude);

        let generated_prelude =
            Self::synthesize_raster_pack_prelude(manifest, tier, &combined_check, spec_overrides);

        let with_defines = Self::inject_after_version_and_extensions(&expanded, &generated_prelude);
        let final_glsl = if hook_prelude.is_empty() {
            with_defines
        } else {
            Self::inject_raster_hook_prelude(&with_defines, &hook_prelude)
        };
        Ok(final_glsl)
    }

    fn raster_hook_call(stage: HookStage) -> String {
        match stage {
            HookStage::VertexDisplace => {
                "vsHookWorld = VsHookWorld(vsMotion.cameraDeltaTime.w, vsMotion.sunDirRain.w, vsMotion.heldLightData.x, vsMotion.heldLightData.y, vsMotion.heldLightData.z, vsMotion.sunDirRain.xyz);\n    // worldPos uses the section world anchor: Position is region-local, RegionOrigin is absolute block space.\n    vsHookVertexDisplace(localPos, VsVertexInput(vertexHookFlags, Position + vec3(RegionOrigin), UV0, MidUV));".to_owned()
            }
            HookStage::WaterWaves => {
                "vsHookWorld = VsHookWorld(vsMotion.cameraDeltaTime.w, vsMotion.sunDirRain.w, vsMotion.heldLightData.x, vsMotion.heldLightData.y, vsMotion.heldLightData.z, vsMotion.sunDirRain.xyz);\n    float vsWaveStrength = 0.35;\n    float vsWaveSpeed = 1.8;\n    float vsWaveScale = 2.2;\n    vsHookWaterWaves(vsWaveStrength, vsWaveSpeed, vsWaveScale);".to_owned()
            }
            HookStage::Surface => {
                "vsHookWorld = VsHookWorld(vsMotion.cameraDeltaTime.w, vsMotion.sunDirRain.w, vsMotion.heldLightData.x, vsMotion.heldLightData.y, vsMotion.heldLightData.z, vsMotion.sunDirRain.xyz);\n    VsSurface vsHookSurf = VsSurface(pow(max(color.rgb, vec3(0.0)), vec3(2.2)), N, roughness, metallic, emission, 1.0, subsurface, f0);\n    #if defined(VS_SURFACE_WORLD_POS)\n    vec3 vsHookWorldPos = VS_SURFACE_WORLD_POS;\n    #else\n    vec3 vsHookWorldPos = vec3(0.0);\n    #endif\n    vsHookSurface(vsHookSurf, vsHookWorldPos, materialFlags, 0u);\n    N = vsHookSurf.normal;\n    color.rgb = pow(max(vsHookSurf.albedo, vec3(0.0)), vec3(1.0 / 2.2));\n    fragColor.rgb = color.rgb;\n    metallic = vsHookSurf.metallic;\n    roughness = vsHookSurf.roughness;\n    f0 = vsHookSurf.f0;\n    emission = vsHookSurf.emission;\n    subsurface = vsHookSurf.subsurface;".to_owned()
            }
            HookStage::GbufferPack => {
                "vsHookWorld = VsHookWorld(vsMotion.cameraDeltaTime.w, vsMotion.sunDirRain.w, vsMotion.heldLightData.x, vsMotion.heldLightData.y, vsMotion.heldLightData.z, vsMotion.sunDirRain.xyz);\n    VsSurface vsPackSurf = VsSurface(pow(max(color.rgb, vec3(0.0)), vec3(2.2)), N, roughness, metallic, emission, 1.0, subsurface, f0);\n    vsHookGbufferPack(vsPackSurf, fragColor, outNormalRoughness, outMaterialPbr);".to_owned()
            }
            other => format!("vsHookRaster_{}();", other.as_str()),
        }
    }

    fn hook_body_dependencies_satisfied_raster(isolated_body: &str, expanded: &str) -> bool {
        if isolated_body.contains("VS_PARAM_")
            && !(Self::contains_identifier(expanded, "vsMotion")
                || Self::contains_identifier(expanded, "VsMotion"))
        {
            return false;
        }
        const RASTER_UNSUPPORTED_SYMBOLS: &[&str] = &[
            "vsTraceShadow",
            "vsTraceReflection",
            "vsSampleIndirectGI",
            "evaluateCookTorrance",
        ];
        RASTER_UNSUPPORTED_SYMBOLS.iter().all(|sym| {
            !Self::contains_identifier(isolated_body, sym)
                || Self::contains_identifier(expanded, sym)
        })
    }

    fn synthesize_raster_pack_prelude(
        manifest: &PackManifest,
        tier: RenderTier,
        expanded_source: &str,
        spec_overrides: &BTreeMap<String, f32>,
    ) -> String {
        let layout = manifest.build_parameter_layout();
        let mut out = String::with_capacity(512);
        out.push_str("// --- VulkanShaders IGNIS auto-synthesized raster pack prelude ---\n");
        out.push_str(&format!(
            "#define VS_ACTIVE_RENDER_TIER {0}\n#define VS_LIGHTING_MODE {0}\n#define VS_MODE_CLASSIC 0\n#define VS_MODE_COMPUTE 1\n#define VS_MODE_HYBRID 2\n#define VS_MODE_PATH 3\n",
            tier.as_u32()
        ));

        let has_motion_ubo = Self::contains_identifier(expanded_source, "vsMotion")
            || Self::contains_identifier(expanded_source, "VsMotion");
        if has_motion_ubo || expanded_source.contains("VS_PARAM_") {
            const COMPS: [&str; 4] = ["x", "y", "z", "w"];
            for (key, &idx) in &layout.runtime_ubo_indices {
                let macro_name = format!("VS_PARAM_{}", key.to_ascii_uppercase());
                if !has_motion_ubo && !expanded_source.contains(&macro_name) {
                    continue;
                }
                let is_vec3 = manifest
                    .settings
                    .get(key)
                    .is_some_and(|d| d.value_type == SettingValueType::Vec3);
                if is_vec3 {
                    let (v0, c0) = (idx / 4, COMPS[idx % 4]);
                    let (v1, c1) = ((idx + 1) / 4, COMPS[(idx + 1) % 4]);
                    let (v2, c2) = ((idx + 2) / 4, COMPS[(idx + 2) % 4]);
                    out.push_str(&format!(
                        "#define {macro_name} vec3(vsMotion.packParams[{v0}].{c0}, vsMotion.packParams[{v1}].{c1}, vsMotion.packParams[{v2}].{c2})\n"
                    ));
                } else {
                    let vec_idx = idx / 4;
                    let comp = COMPS[idx % 4];
                    out.push_str(&format!(
                        "#define {macro_name} (vsMotion.packParams[{vec_idx}].{comp})\n"
                    ));
                }
            }
        }

        // Inline `#define VS_SPEC_<KEY> <override_or_default>` without `layout(constant_id = ...)`
        // because RenderPearl does not pass `VkSpecializationInfo` to raster pipelines.
        for key in layout.specialization_constant_ids.keys() {
            let const_name = format!("VS_SPEC_{}", key.to_ascii_uppercase());
            let override_val = spec_overrides
                .get(key)
                .or_else(|| spec_overrides.get(&key.to_ascii_uppercase()))
                .copied();
            if let Some(decl) = manifest.settings.get(key) {
                match decl.value_type {
                    SettingValueType::Bool => {
                        let val = override_val.map(|v| v != 0.0).unwrap_or_else(|| {
                            matches!(decl.default, crate::manifest::SettingValue::Bool(true))
                        });
                        out.push_str(&format!("#define {const_name} {val}\n"));
                    }
                    SettingValueType::Int | SettingValueType::Enum => {
                        let val = override_val
                            .map(|v| v.round() as i32)
                            .unwrap_or_else(|| decl.default.to_f32(&decl.options) as i32);
                        out.push_str(&format!("#define {const_name} {val}\n"));
                    }
                    SettingValueType::ColorRgb => {
                        let val = override_val.unwrap_or_else(|| decl.default.to_color_f32());
                        out.push_str(&format!("#define {const_name} {val:.6}\n"));
                    }
                    _ => {
                        let val =
                            override_val.unwrap_or_else(|| decl.default.to_f32(&decl.options));
                        out.push_str(&format!("#define {const_name} {val:.6}\n"));
                    }
                }
            } else if let Some(val) = override_val {
                out.push_str(&format!("#define {const_name} {val:.6}\n"));
            }
        }

        out.push_str("// --- end auto-synthesized raster pack prelude ---\n");
        out
    }

    /// Comment- and string-aware GLSL lexer that extracts the body of the `@hook(stage = ...)`
    /// function along with any preceding helper functions, optionally isolating helper identifiers
    /// with `ns_prefix` so multiple raster/compute hooks never collide.
    pub fn extract_hook_helper_and_body_lexed(
        raw_hook_source: &str,
        target_stage: HookStage,
        ns_prefix: Option<&str>,
    ) -> Option<ExtractedHookBlock> {
        let stripped = Self::strip_header_guards_and_includes(raw_hook_source);
        let annotations = Self::scan_hook_annotations(&stripped);
        let target_ann = annotations
            .iter()
            .find(|a| a.stage == target_stage)
            .cloned()
            .or_else(|| {
                if !annotations.is_empty() {
                    return None;
                }
                // Fallback for unannotated stage-named function
                let stage_str = target_stage.as_str();
                Self::find_function_containing(&stripped, stage_str).map(|fn_name| HookAnnotation {
                    stage: target_stage,
                    priority: 100,
                    function_name: fn_name,
                })
            })?;

        let (helper_raw, body_raw) =
            Self::split_helper_and_function_body_lexed(&stripped, &target_ann.function_name)?;

        if let Some(prefix) = ns_prefix
            && !prefix.is_empty()
        {
            let helper_symbols = Self::collect_top_level_symbols(&helper_raw, &[]);
            if !helper_symbols.is_empty() {
                let mut rename_map = BTreeMap::new();
                for sym in helper_symbols {
                    rename_map.insert(sym.clone(), format!("{prefix}{sym}"));
                }
                return Some(ExtractedHookBlock {
                    stage: target_stage,
                    priority: target_ann.priority,
                    isolated_helper_code: Self::rewrite_identifiers(&helper_raw, &rename_map)
                        .trim()
                        .to_owned(),
                    isolated_body: Self::rewrite_identifiers(&body_raw, &rename_map)
                        .trim_matches(['\n', '\r'])
                        .to_owned(),
                });
            }
        }

        Some(ExtractedHookBlock {
            stage: target_stage,
            priority: target_ann.priority,
            isolated_helper_code: helper_raw.trim().to_owned(),
            isolated_body: body_raw.trim_matches(['\n', '\r']).to_owned(),
        })
    }

    fn find_function_containing(source: &str, stage_substr: &str) -> Option<String> {
        let code_only = Self::mask_comments_and_strings(source);
        let mut brace_depth = 0i32;
        for line in code_only.lines() {
            let trimmed = line.trim();
            if brace_depth == 0
                && !trimmed.is_empty()
                && !trimmed.starts_with('#')
                && let Some(fn_name) = Self::extract_function_name(trimmed)
                && fn_name.contains(stage_substr)
            {
                return Some(fn_name);
            }
            for b in trimmed.bytes() {
                if b == b'{' {
                    brace_depth += 1;
                } else if b == b'}' {
                    brace_depth = (brace_depth - 1).max(0);
                }
            }
        }
        None
    }

    /// Replaces characters inside `// ...`, `/* ... */`, and `"..."` with spaces (preserving
    /// newlines and exact byte offsets) so brace matching and top-level symbol scanning can never
    /// be tricked by braces or keywords inside comments or string literals.
    pub fn mask_comments_and_strings(source: &str) -> String {
        let bytes = source.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0usize;
        while i < bytes.len() {
            if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
                out.push(b' ');
                out.push(b' ');
                i += 2;
                while i < bytes.len() && bytes[i] != b'\n' {
                    out.push(if bytes[i] == b'\r' { b'\r' } else { b' ' });
                    i += 1;
                }
            } else if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
                out.push(b' ');
                out.push(b' ');
                i += 2;
                while i < bytes.len() {
                    if i + 1 < bytes.len() && bytes[i] == b'*' && bytes[i + 1] == b'/' {
                        out.push(b' ');
                        out.push(b' ');
                        i += 2;
                        break;
                    }
                    out.push(if bytes[i] == b'\n' || bytes[i] == b'\r' {
                        bytes[i]
                    } else {
                        b' '
                    });
                    i += 1;
                }
            } else if bytes[i] == b'"' {
                out.push(b' ');
                i += 1;
                while i < bytes.len() {
                    if bytes[i] == b'\\' && i + 1 < bytes.len() {
                        out.push(b' ');
                        out.push(b' ');
                        i += 2;
                        continue;
                    }
                    if bytes[i] == b'"' {
                        out.push(b' ');
                        i += 1;
                        break;
                    }
                    out.push(if bytes[i] == b'\n' || bytes[i] == b'\r' {
                        bytes[i]
                    } else {
                        b' '
                    });
                    i += 1;
                }
            } else {
                out.push(bytes[i]);
                i += 1;
            }
        }
        String::from_utf8(out).unwrap_or_else(|_| source.to_owned())
    }

    fn split_helper_and_function_body_lexed(
        source: &str,
        target_fn_name: &str,
    ) -> Option<(String, String)> {
        let masked = Self::mask_comments_and_strings(source);
        let bytes = masked.as_bytes();
        let needle = target_fn_name.as_bytes();
        let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';

        // Locate top-level definition of `target_fn_name`
        let mut brace_depth = 0i32;
        let mut i = 0usize;
        let mut fn_decl_line_start = 0usize;
        let mut open_brace_pos = None;

        while i < bytes.len() {
            if bytes[i] == b'\n' && brace_depth == 0 {
                fn_decl_line_start = i + 1;
            }
            if brace_depth == 0
                && i + needle.len() <= bytes.len()
                && &bytes[i..i + needle.len()] == needle
                && (i == 0 || !is_ident(bytes[i - 1]))
                && (i + needle.len() == bytes.len() || !is_ident(bytes[i + needle.len()]))
            {
                let mut j = i + needle.len();
                while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                    j += 1;
                }
                if j < bytes.len() && bytes[j] == b'(' {
                    // Advance past parameter list (...) to opening '{'
                    let mut paren_depth = 0i32;
                    while j < bytes.len() {
                        if bytes[j] == b'(' {
                            paren_depth += 1;
                        } else if bytes[j] == b')' {
                            paren_depth -= 1;
                            if paren_depth == 0 {
                                j += 1;
                                break;
                            }
                        }
                        j += 1;
                    }
                    while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                        j += 1;
                    }
                    if j < bytes.len() && bytes[j] == b'{' {
                        open_brace_pos = Some(j);
                        break;
                    }
                }
            }
            if bytes[i] == b'{' {
                brace_depth += 1;
            } else if bytes[i] == b'}' {
                brace_depth = (brace_depth - 1).max(0);
            }
            i += 1;
        }

        let open_pos = open_brace_pos?;
        let mut depth = 0i32;
        let mut close_pos = None;
        for (idx, &b) in bytes.iter().enumerate().skip(open_pos) {
            if b == b'{' {
                depth += 1;
            } else if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    close_pos = Some(idx);
                    break;
                }
            }
        }
        let close = close_pos?;
        let body = source[open_pos + 1..close].to_owned();

        // Filter out `@hook(...)` comment lines from helper prefix
        let mut helper_out = String::new();
        for line in source[..fn_decl_line_start].lines() {
            let t = line.trim();
            if t.starts_with("//") && t.contains("@hook") || t.starts_with("@hook") {
                continue;
            }
            helper_out.push_str(line);
            helper_out.push('\n');
        }
        let has_code = Self::mask_comments_and_strings(&helper_out)
            .chars()
            .any(|c| !c.is_whitespace());
        let final_helper = if has_code { helper_out } else { String::new() };
        Some((final_helper, body))
    }

    fn sanitize_ident(raw: &str) -> String {
        raw.chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect()
    }

    fn contains_identifier(source: &str, ident: &str) -> bool {
        let masked = Self::mask_comments_and_strings(source);
        let bytes = masked.as_bytes();
        let needle = ident.as_bytes();
        if needle.is_empty() || bytes.len() < needle.len() {
            return false;
        }
        let is_ident_byte = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
        let mut start = 0usize;
        while let Some(rel) = masked[start..].find(ident) {
            let pos = start + rel;
            let before_ok = pos == 0 || !is_ident_byte(bytes[pos - 1]);
            let after_pos = pos + needle.len();
            let after_ok = after_pos >= bytes.len() || !is_ident_byte(bytes[after_pos]);
            if before_ok && after_ok {
                return true;
            }
            start = pos + 1;
        }
        false
    }

    fn stage_signature_types_present(expanded: &str, stage: HookStage) -> bool {
        match stage {
            HookStage::VertexDisplace => Self::contains_identifier(expanded, "VsVertexInput"),
            HookStage::GbufferPack
            | HookStage::Surface
            | HookStage::RestirReservoir
            | HookStage::DirectLight
            | HookStage::IndirectGi
            | HookStage::WaterSurface => Self::contains_identifier(expanded, "VsSurface"),
            HookStage::Sky | HookStage::Clouds => Self::contains_identifier(expanded, "VsSkyState"),
            HookStage::PostGrade => Self::contains_identifier(expanded, "VsPostContext"),
            HookStage::EmissiveSample
            | HookStage::BlockLight
            | HookStage::WaterWaves
            | HookStage::VolumetricMedia
            | HookStage::Fog
            | HookStage::AoFilter => true,
        }
    }

    fn is_hook_stage_compatible_with_shader(
        hook_stage: HookStage,
        shader_stage: Option<ShaderStageKind>,
    ) -> bool {
        match shader_stage {
            Some(ShaderStageKind::Vertex) => {
                matches!(
                    hook_stage,
                    HookStage::VertexDisplace | HookStage::WaterWaves
                )
            }
            Some(ShaderStageKind::Fragment) => matches!(
                hook_stage,
                HookStage::GbufferPack
                    | HookStage::Surface
                    | HookStage::EmissiveSample
                    | HookStage::RestirReservoir
                    | HookStage::DirectLight
                    | HookStage::BlockLight
                    | HookStage::IndirectGi
                    | HookStage::WaterWaves
                    | HookStage::WaterSurface
                    | HookStage::Sky
                    | HookStage::Clouds
                    | HookStage::VolumetricMedia
                    | HookStage::Fog
                    | HookStage::AoFilter
            ),
            Some(ShaderStageKind::Compute) => !matches!(
                hook_stage,
                HookStage::VertexDisplace | HookStage::GbufferPack
            ),
            Some(
                ShaderStageKind::RayGen
                | ShaderStageKind::ClosestHit
                | ShaderStageKind::Miss
                | ShaderStageKind::AnyHit,
            ) => matches!(
                hook_stage,
                HookStage::Surface
                    | HookStage::EmissiveSample
                    | HookStage::RestirReservoir
                    | HookStage::DirectLight
                    | HookStage::BlockLight
                    | HookStage::IndirectGi
                    | HookStage::WaterWaves
                    | HookStage::WaterSurface
                    | HookStage::Sky
                    | HookStage::Clouds
                    | HookStage::VolumetricMedia
                    | HookStage::Fog
                    | HookStage::AoFilter
            ),
            None => true,
        }
    }

    fn hook_body_dependencies_satisfied(isolated_body: &str, expanded: &str) -> bool {
        if isolated_body.contains("VS_PARAM_")
            && !(Self::contains_identifier(expanded, "vsWorld")
                || Self::contains_identifier(expanded, "vsMotion")
                || Self::contains_identifier(expanded, "VsMotion"))
        {
            return false;
        }
        const ENGINE_SYMBOLS: &[&str] = &[
            "vsWorld",
            "uSharc",
            "VsVertexInput",
            "VsSurface",
            "VsSkyState",
            "VsPostContext",
            "vsTraceShadow",
            "vsTraceReflection",
            "vsSampleIndirectGI",
            "evaluateCookTorrance",
        ];
        ENGINE_SYMBOLS.iter().all(|sym| {
            !Self::contains_identifier(isolated_body, sym)
                || Self::contains_identifier(expanded, sym)
        })
    }

    /// Strips header guards/directives from a `.vsl` hook file, scans `@hook` annotations,
    /// discovers top-level helper functions & constants defined in that file using the comment-aware
    /// lexer, and rewrites their identifiers with `ns_prefix` so multiple `.vsl` files never collide.
    pub fn isolate_hook_module(
        expanded_hook: &str,
        ns_prefix: &str,
    ) -> (String, Vec<HookAnnotation>) {
        let stripped = Self::strip_header_guards_and_includes(expanded_hook);
        let mut annotations = Self::scan_hook_annotations(&stripped);
        let local_symbols = Self::collect_top_level_symbols(&stripped, &annotations);

        if local_symbols.is_empty() {
            return (stripped, annotations);
        }

        let mut rename_map = BTreeMap::new();
        for sym in local_symbols {
            rename_map.insert(sym.clone(), format!("{ns_prefix}{sym}"));
        }

        for ann in &mut annotations {
            if let Some(renamed) = rename_map.get(&ann.function_name) {
                ann.function_name = renamed.clone();
            }
        }

        let rewritten = Self::rewrite_identifiers(&stripped, &rename_map);
        (rewritten, annotations)
    }

    fn isolate_hook_module_mapped(
        expanded_hook: &str,
        hook_map: &SourceLineMap,
        ns_prefix: &str,
    ) -> (String, SourceLineMap, Vec<HookAnnotation>) {
        let (stripped, stripped_map) =
            Self::strip_header_guards_and_includes_mapped(expanded_hook, hook_map);
        let mut annotations = Self::scan_hook_annotations(&stripped);
        let local_symbols = Self::collect_top_level_symbols(&stripped, &annotations);

        if local_symbols.is_empty() {
            return (stripped, stripped_map, annotations);
        }

        let mut rename_map = BTreeMap::new();
        for sym in local_symbols {
            rename_map.insert(sym.clone(), format!("{ns_prefix}{sym}"));
        }

        for ann in &mut annotations {
            if let Some(renamed) = rename_map.get(&ann.function_name) {
                ann.function_name = renamed.clone();
            }
        }

        let rewritten = Self::rewrite_identifiers(&stripped, &rename_map);
        (rewritten, stripped_map, annotations)
    }

    fn collect_top_level_symbols(source: &str, annotations: &[HookAnnotation]) -> Vec<String> {
        let mut symbols = Vec::new();
        let mut seen = HashSet::new();

        for ann in annotations {
            if seen.insert(ann.function_name.clone()) {
                symbols.push(ann.function_name.clone());
            }
        }

        let masked = Self::mask_comments_and_strings(source);
        let mut brace_depth = 0i32;
        for line in masked.lines() {
            let trimmed = line.trim();
            if brace_depth == 0
                && !trimmed.is_empty()
                && !trimmed.starts_with('#')
                && !trimmed.starts_with("layout")
                && !trimmed.starts_with("uniform")
                && !trimmed.starts_with("buffer")
            {
                if let Some(rest) = trimmed.strip_prefix("struct ") {
                    let struct_name = rest
                        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                        .next()
                        .unwrap_or("");
                    if !struct_name.is_empty()
                        && !struct_name.starts_with("Vs")
                        && !struct_name.starts_with("vs")
                        && !struct_name.starts_with("gl_")
                        && !Self::is_glsl_builtin_or_keyword(struct_name)
                        && seen.insert(struct_name.to_owned())
                    {
                        symbols.push(struct_name.to_owned());
                    }
                } else if trimmed.starts_with("const ") {
                    // Extract global const identifier: `const <type> NAME = ...;` or `const <type> NAME[N] = ...;`
                    let before_eq = trimmed.split('=').next().unwrap_or(trimmed).trim();
                    let before_br = before_eq.split('[').next().unwrap_or(before_eq).trim();
                    let const_name = before_br
                        .split_whitespace()
                        .last()
                        .unwrap_or("")
                        .trim_end_matches(';');
                    if !const_name.is_empty()
                        && const_name
                            .chars()
                            .all(|c| c.is_ascii_alphanumeric() || c == '_')
                        && !const_name.starts_with("VS_")
                        && !const_name.starts_with("vs")
                        && !const_name.starts_with("gl_")
                        && !Self::is_glsl_builtin_or_keyword(const_name)
                        && seen.insert(const_name.to_owned())
                    {
                        symbols.push(const_name.to_owned());
                    }
                } else if let Some(fn_name) = Self::extract_function_name(trimmed)
                    && fn_name != "main"
                    && !fn_name.starts_with("vs")
                    && !fn_name.starts_with("gl_")
                    && !Self::is_glsl_builtin_or_keyword(&fn_name)
                    && seen.insert(fn_name.clone())
                {
                    symbols.push(fn_name);
                }
            }

            for ch in trimmed.chars() {
                if ch == '{' {
                    brace_depth += 1;
                } else if ch == '}' {
                    brace_depth = (brace_depth - 1).max(0);
                }
            }
        }

        symbols
    }

    fn is_glsl_builtin_or_keyword(ident: &str) -> bool {
        matches!(
            ident,
            "void"
                | "bool"
                | "int"
                | "uint"
                | "float"
                | "double"
                | "vec2"
                | "vec3"
                | "vec4"
                | "ivec2"
                | "ivec3"
                | "ivec4"
                | "uvec2"
                | "uvec3"
                | "uvec4"
                | "mat2"
                | "mat3"
                | "mat4"
                | "if"
                | "for"
                | "while"
                | "switch"
                | "return"
                | "clamp"
                | "mix"
                | "smoothstep"
                | "step"
                | "min"
                | "max"
                | "sin"
                | "cos"
                | "tan"
                | "pow"
                | "exp"
                | "log"
                | "sqrt"
                | "abs"
                | "dot"
                | "cross"
                | "normalize"
                | "length"
                | "reflect"
                | "texture"
                | "textureLod"
                | "texelFetch"
                | "imageLoad"
                | "imageStore"
                | "evaluateCookTorrance"
        )
    }

    /// State-machine GLSL lexer that rewrites identifiers in `rename_map` while leaving `//` line
    /// comments, `/* ... */` block comments, `"..."` string literals, `#` preprocessor directives,
    /// and `.member` field accesses completely untouched.
    pub fn rewrite_identifiers(source: &str, rename_map: &BTreeMap<String, String>) -> String {
        if rename_map.is_empty() {
            return source.to_owned();
        }
        let bytes = source.as_bytes();
        let mut out = String::with_capacity(source.len() + rename_map.len() * 32);
        let mut i = 0usize;
        let mut at_line_start = true;

        while i < bytes.len() {
            let b = bytes[i];
            if b == b'\n' {
                out.push('\n');
                at_line_start = true;
                i += 1;
                continue;
            }
            if at_line_start && b.is_ascii_whitespace() {
                out.push(b as char);
                i += 1;
                continue;
            }
            if at_line_start && b == b'#' {
                // Preprocessor directive line
                while i < bytes.len() && bytes[i] != b'\n' {
                    out.push(bytes[i] as char);
                    i += 1;
                }
                continue;
            }
            at_line_start = false;

            // Line comment `// ...`
            if i + 1 < bytes.len() && b == b'/' && bytes[i + 1] == b'/' {
                while i < bytes.len() && bytes[i] != b'\n' {
                    out.push(bytes[i] as char);
                    i += 1;
                }
                continue;
            }
            // Block comment `/* ... */`
            if i + 1 < bytes.len() && b == b'/' && bytes[i + 1] == b'*' {
                out.push('/');
                out.push('*');
                i += 2;
                while i < bytes.len() {
                    if i + 1 < bytes.len() && bytes[i] == b'*' && bytes[i + 1] == b'/' {
                        out.push('*');
                        out.push('/');
                        i += 2;
                        break;
                    }
                    if bytes[i] == b'\n' {
                        at_line_start = true;
                    }
                    out.push(bytes[i] as char);
                    i += 1;
                }
                continue;
            }
            // String literal `"..."`
            if b == b'"' {
                out.push('"');
                i += 1;
                while i < bytes.len() {
                    if bytes[i] == b'\\' && i + 1 < bytes.len() {
                        out.push(bytes[i] as char);
                        out.push(bytes[i + 1] as char);
                        i += 2;
                        continue;
                    }
                    out.push(bytes[i] as char);
                    if bytes[i] == b'"' {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
                continue;
            }

            if b.is_ascii_alphabetic() || b == b'_' {
                let start = i;
                i += 1;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                let token = &source[start..i];
                let preceded_by_dot = start > 0 && bytes[start - 1] == b'.';
                if !preceded_by_dot && let Some(replacement) = rename_map.get(token) {
                    out.push_str(replacement);
                } else {
                    out.push_str(token);
                }
            } else if b < 0x80 {
                out.push(b as char);
                i += 1;
            } else if let Some(ch) = source[i..].chars().next() {
                out.push(ch);
                i += ch.len_utf8();
            } else {
                i += 1;
            }
        }
        if !out.ends_with('\n') {
            out.push('\n');
        }
        out
    }

    fn strip_header_guards_and_includes(source: &str) -> String {
        let mut out = String::with_capacity(source.len());
        let mut include_depth: usize = 0;
        // Track preprocessor `#if`/`#ifdef`/`#ifndef` stack; `true` marks a stripped header guard
        // whose matching `#endif` must also be stripped so no dangling `#endif` remains.
        let mut if_stack: Vec<bool> = Vec::new();
        for line in source.lines() {
            let t = line.trim();
            if t.starts_with("// begin include:") || t.starts_with("// --- begin include") {
                include_depth += 1;
                continue;
            }
            if t.starts_with("// end include:") || t.starts_with("// --- end include") {
                include_depth = include_depth.saturating_sub(1);
                continue;
            }
            if include_depth > 0 {
                continue;
            }
            if t.starts_with("#ifndef VULKANSHADERS_") {
                if_stack.push(true);
                continue;
            }
            if t.starts_with("#if") {
                if_stack.push(false);
                out.push_str(line);
                out.push('\n');
                continue;
            }
            if t.starts_with("#endif") {
                if let Some(was_stripped_guard) = if_stack.pop()
                    && was_stripped_guard
                {
                    continue;
                }
                out.push_str(line);
                out.push('\n');
                continue;
            }
            if t.starts_with("#version")
                || t.starts_with("#extension")
                || t.starts_with("#include")
                || t.starts_with("#define VULKANSHADERS_")
            {
                continue;
            }
            out.push_str(line);
            out.push('\n');
        }
        out
    }

    fn strip_header_guards_and_includes_mapped(
        source: &str,
        source_map: &SourceLineMap,
    ) -> (String, SourceLineMap) {
        let mut out = String::with_capacity(source.len());
        let mut out_map = SourceLineMap::new();
        let mut include_depth: usize = 0;
        let mut if_stack: Vec<bool> = Vec::new();
        for (idx, line) in source.lines().enumerate() {
            let t = line.trim();
            if t.starts_with("// begin include:") || t.starts_with("// --- begin include") {
                include_depth += 1;
                continue;
            }
            if t.starts_with("// end include:") || t.starts_with("// --- end include") {
                include_depth = include_depth.saturating_sub(1);
                continue;
            }
            if include_depth > 0 {
                continue;
            }
            if t.starts_with("#ifndef VULKANSHADERS_") {
                if_stack.push(true);
                continue;
            }
            if t.starts_with("#if") {
                if_stack.push(false);
                out.push_str(line);
                out.push('\n');
                if let Some(loc) = source_map.entries().get(idx) {
                    out_map.push_line(loc.file.clone(), loc.line);
                } else {
                    out_map.push_line("<hook>", (idx + 1) as u32);
                }
                continue;
            }
            if t.starts_with("#endif") {
                if let Some(was_stripped_guard) = if_stack.pop()
                    && was_stripped_guard
                {
                    continue;
                }
                out.push_str(line);
                out.push('\n');
                if let Some(loc) = source_map.entries().get(idx) {
                    out_map.push_line(loc.file.clone(), loc.line);
                } else {
                    out_map.push_line("<hook>", (idx + 1) as u32);
                }
                continue;
            }
            if t.starts_with("#version")
                || t.starts_with("#extension")
                || t.starts_with("#include")
                || t.starts_with("#define VULKANSHADERS_")
            {
                continue;
            }
            out.push_str(line);
            out.push('\n');
            if let Some(loc) = source_map.entries().get(idx) {
                out_map.push_line(loc.file.clone(), loc.line);
            } else {
                out_map.push_line("<hook>", (idx + 1) as u32);
            }
        }
        (out, out_map)
    }

    fn inject_before_main_mapped(
        source: &str,
        source_map: &mut SourceLineMap,
        block_to_inject: &str,
        block_map: &SourceLineMap,
    ) -> String {
        let masked = Self::mask_comments_and_strings(source);
        if let Some(main_pos) = masked
            .find("void main(")
            .or_else(|| masked.find("void main ("))
        {
            let line_start = source[..main_pos].rfind('\n').map(|i| i + 1).unwrap_or(0);
            let line_idx = source[..line_start].bytes().filter(|&b| b == b'\n').count();
            let mut out = String::with_capacity(source.len() + block_to_inject.len() + 16);
            out.push_str(&source[..line_start]);
            out.push_str(block_to_inject);
            if !block_to_inject.is_empty() && !block_to_inject.ends_with('\n') {
                out.push('\n');
            }
            out.push_str(&source[line_start..]);
            source_map.insert_before_line(line_idx, block_map.entries());
            out
        } else {
            let mut out = source.to_owned();
            if !out.is_empty() && !out.ends_with('\n') {
                out.push('\n');
            }
            out.push_str(block_to_inject);
            if !block_to_inject.is_empty() && !block_to_inject.ends_with('\n') {
                out.push('\n');
            }
            source_map.extend_from(block_map);
            out
        }
    }

    fn expand_includes<F>(
        &self,
        source_name: &str,
        source: &str,
        depth: usize,
        already_included: &mut HashSet<String>,
        loader: &F,
    ) -> Result<String, VspError>
    where
        F: Fn(&str) -> Option<String>,
    {
        self.expand_includes_with_options(
            source_name,
            source,
            depth,
            already_included,
            loader,
            false,
        )
    }

    fn expand_includes_mapped<F>(
        &self,
        source_name: &str,
        source: &str,
        depth: usize,
        already_included: &mut HashSet<String>,
        loader: &F,
        preserve_namespaced_includes: bool,
    ) -> Result<(String, SourceLineMap), VspError>
    where
        F: Fn(&str) -> Option<String>,
    {
        if depth > self.max_include_depth {
            return Err(VspError::IncludeDepthExceeded {
                include_path: source_name.to_owned(),
                max_depth: self.max_include_depth,
            });
        }

        let mut out = String::with_capacity(source.len() + 256);
        let mut map = SourceLineMap::new();
        for (idx, line) in source.lines().enumerate() {
            let orig_line = (idx + 1) as u32;
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("#include") {
                let path_token = rest.trim();
                let include_path = path_token
                    .strip_prefix('<')
                    .and_then(|s| s.strip_suffix('>'))
                    .or_else(|| {
                        path_token
                            .strip_prefix('"')
                            .and_then(|s| s.strip_suffix('"'))
                    })
                    .map(str::trim);

                if let Some(inc) = include_path {
                    if preserve_namespaced_includes && inc.contains(':') {
                        out.push_str(line);
                        out.push('\n');
                        map.push_line(source_name, orig_line);
                        continue;
                    }
                    if !already_included.insert(inc.to_owned()) {
                        continue;
                    }
                    let content = loader(inc)
                        .or_else(|| Self::builtin_header(inc).map(str::to_owned))
                        .ok_or_else(|| VspError::UnresolvedInclude {
                            source_file: source_name.to_owned(),
                            include_path: inc.to_owned(),
                        })?;
                    let (nested, nested_map) = self.expand_includes_mapped(
                        inc,
                        &content,
                        depth + 1,
                        already_included,
                        loader,
                        preserve_namespaced_includes,
                    )?;
                    out.push_str("// --- begin include <");
                    out.push_str(inc);
                    out.push_str("> ---\n");
                    map.push_line(source_name, orig_line);
                    out.push_str(&nested);
                    map.extend_from(&nested_map);
                    out.push_str("// --- end include <");
                    out.push_str(inc);
                    out.push_str("> ---\n");
                    map.push_line(source_name, orig_line);
                    continue;
                }
            }
            out.push_str(line);
            out.push('\n');
            map.push_line(source_name, orig_line);
        }
        Ok((out, map))
    }

    fn expand_includes_with_options<F>(
        &self,
        source_name: &str,
        source: &str,
        depth: usize,
        already_included: &mut HashSet<String>,
        loader: &F,
        preserve_namespaced_includes: bool,
    ) -> Result<String, VspError>
    where
        F: Fn(&str) -> Option<String>,
    {
        if depth > self.max_include_depth {
            return Err(VspError::IncludeDepthExceeded {
                include_path: source_name.to_owned(),
                max_depth: self.max_include_depth,
            });
        }

        let mut out = String::with_capacity(source.len() + 256);
        for line in source.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("#include") {
                let path_token = rest.trim();
                let include_path = path_token
                    .strip_prefix('<')
                    .and_then(|s| s.strip_suffix('>'))
                    .or_else(|| {
                        path_token
                            .strip_prefix('"')
                            .and_then(|s| s.strip_suffix('"'))
                    })
                    .map(str::trim);

                if let Some(inc) = include_path {
                    if preserve_namespaced_includes && inc.contains(':') {
                        out.push_str(line);
                        out.push('\n');
                        continue;
                    }
                    if !already_included.insert(inc.to_owned()) {
                        // Already expanded earlier in this translation unit
                        continue;
                    }
                    let content = loader(inc)
                        .or_else(|| Self::builtin_header(inc).map(str::to_owned))
                        .ok_or_else(|| VspError::UnresolvedInclude {
                            source_file: source_name.to_owned(),
                            include_path: inc.to_owned(),
                        })?;
                    let nested = self.expand_includes_with_options(
                        inc,
                        &content,
                        depth + 1,
                        already_included,
                        loader,
                        preserve_namespaced_includes,
                    )?;
                    out.push_str("// --- begin include <");
                    out.push_str(inc);
                    out.push_str("> ---\n");
                    out.push_str(&nested);
                    out.push_str("\n// --- end include <");
                    out.push_str(inc);
                    out.push_str("> ---\n");
                    continue;
                }
            }
            out.push_str(line);
            out.push('\n');
        }
        Ok(out)
    }

    pub fn scan_hook_annotations(source: &str) -> Vec<HookAnnotation> {
        let mut hooks = Vec::new();
        let mut pending: Option<(HookStage, i32)> = None;

        for line in source.lines() {
            let trimmed = line.trim();
            let maybe_attr = trimmed
                .strip_prefix("//")
                .map(str::trim)
                .or(Some(trimmed))
                .and_then(|s| s.strip_prefix("@hook("))
                .and_then(|s| s.strip_suffix(')'));

            if let Some(attr) = maybe_attr {
                let mut stage: Option<HookStage> = None;
                let mut priority: i32 = 100;
                for part in attr.split(',') {
                    let Some((k, v)) = part.split_once('=') else {
                        continue;
                    };
                    let k = k.trim();
                    let v = v.trim().trim_matches('"').trim_matches('\'');
                    if k == "stage" {
                        stage = HookStage::from_name(v);
                    } else if k == "priority"
                        && let Ok(p) = v.parse::<i32>()
                    {
                        priority = p;
                    }
                }
                if let Some(st) = stage {
                    pending = Some((st, priority));
                }
                continue;
            }

            if trimmed.starts_with("//") {
                continue;
            }

            if let Some((stage, priority)) = pending {
                if trimmed.is_empty() || trimmed.starts_with('#') {
                    continue;
                }
                if let Some(fn_name) = Self::extract_function_name(trimmed) {
                    hooks.push(HookAnnotation {
                        stage,
                        priority,
                        function_name: fn_name,
                    });
                    pending = None;
                }
            }
        }
        hooks
    }

    fn extract_function_name(line: &str) -> Option<String> {
        let open_paren = line.find('(')?;
        let prefix = line[..open_paren].trim();
        let mut tokens = prefix.split_whitespace();
        let ret_ty = tokens.next()?;
        let fn_token = tokens.last().unwrap_or(ret_ty);
        if ret_ty == fn_token {
            return None;
        }
        if fn_token
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            Some(fn_token.to_owned())
        } else {
            None
        }
    }

    fn synthesize_pack_prelude(
        manifest: &PackManifest,
        tier: RenderTier,
        expanded_source: &str,
        shader_stage: Option<ShaderStageKind>,
    ) -> String {
        let layout = manifest.build_parameter_layout();
        let mut out = String::with_capacity(512);
        out.push_str("// --- VulkanShaders IGNIS auto-synthesized pack prelude ---\n");
        out.push_str(&format!(
            "#define VS_ACTIVE_RENDER_TIER {0}\n#define VS_LIGHTING_MODE {0}\n#define VS_MODE_CLASSIC 0\n#define VS_MODE_COMPUTE 1\n#define VS_MODE_HYBRID 2\n#define VS_MODE_PATH 3\n",
            tier.as_u32()
        ));

        let has_world_ubo = expanded_source.contains("vsWorld");
        if has_world_ubo || expanded_source.contains("VS_PARAM_") {
            const COMPS: [&str; 4] = ["x", "y", "z", "w"];
            for (key, &idx) in &layout.runtime_ubo_indices {
                let macro_name = format!("VS_PARAM_{}", key.to_ascii_uppercase());
                if !has_world_ubo && !expanded_source.contains(&macro_name) {
                    continue;
                }
                let is_vec3 = manifest
                    .settings
                    .get(key)
                    .is_some_and(|d| d.value_type == SettingValueType::Vec3);
                if is_vec3 {
                    let (v0, c0) = (idx / 4, COMPS[idx % 4]);
                    let (v1, c1) = ((idx + 1) / 4, COMPS[(idx + 1) % 4]);
                    let (v2, c2) = ((idx + 2) / 4, COMPS[(idx + 2) % 4]);
                    out.push_str(&format!(
                        "#define {macro_name} vec3(vsWorld.packParams[{v0}].{c0}, vsWorld.packParams[{v1}].{c1}, vsWorld.packParams[{v2}].{c2})\n"
                    ));
                } else {
                    let vec_idx = idx / 4;
                    let comp = COMPS[idx % 4];
                    out.push_str(&format!(
                        "#define {macro_name} (vsWorld.packParams[{vec_idx}].{comp})\n"
                    ));
                }
            }
        }

        if expanded_source.contains("VS_SPEC_") {
            for (key, &const_id) in &layout.specialization_constant_ids {
                let upper_key = key.to_ascii_uppercase();
                let const_name = format!("VS_SPEC_{upper_key}");
                let guard_macro = format!("VS_HAS_SPEC_{upper_key}");
                if !expanded_source.contains(&const_name) {
                    continue;
                }
                if let Some(decl) = manifest.settings.get(key) {
                    out.push_str(&format!("#define {guard_macro} 1\n"));
                    match decl.value_type {
                        SettingValueType::Bool => {
                            let def =
                                matches!(decl.default, crate::manifest::SettingValue::Bool(true));
                            out.push_str(&format!(
                                "layout(constant_id = {const_id}) const bool {const_name} = {def};\n"
                            ));
                        }
                        SettingValueType::Int | SettingValueType::Enum => {
                            let def = decl.default.to_f32(&decl.options) as i32;
                            out.push_str(&format!(
                                "layout(constant_id = {const_id}) const int {const_name} = {def};\n"
                            ));
                        }
                        SettingValueType::ColorRgb => {
                            let def = decl.default.to_color_f32();
                            out.push_str(&format!(
                                "layout(constant_id = {const_id}) const float {const_name} = {def:.6};\n"
                            ));
                        }
                        _ => {
                            let def = decl.default.to_f32(&decl.options);
                            out.push_str(&format!(
                                "layout(constant_id = {const_id}) const float {const_name} = {def:.6};\n"
                            ));
                        }
                    }
                }
            }
        }

        // Synthesize named pack resource bindings & accessors in descriptor set 3 only when
        // referenced by the shader or its injected hooks so utility stages (wind_displace,
        // sharc_resolve, depth_blit) do not pull unused Set 3 descriptor layouts.
        for (idx, (res_name, _decl)) in manifest.resources.iter().enumerate() {
            let sampler_ident = format!("uVsPackRes_{res_name}");
            let helper_ident = format!("vsSamplePackTexture_{res_name}");
            if !expanded_source.contains(&sampler_ident) && !expanded_source.contains(&helper_ident)
            {
                continue;
            }
            let binding = idx as u32;
            out.push_str(&format!(
                "layout(set = 3, binding = {binding}) uniform sampler2D {sampler_ident};\n"
            ));
            let use_lod = matches!(
                shader_stage,
                Some(
                    ShaderStageKind::Compute
                        | ShaderStageKind::RayGen
                        | ShaderStageKind::ClosestHit
                        | ShaderStageKind::Miss
                        | ShaderStageKind::AnyHit
                )
            );
            if use_lod {
                out.push_str(&format!(
                    "vec4 {helper_ident}(vec2 uv) {{ return textureLod({sampler_ident}, uv, 0.0); }}\n"
                ));
            } else {
                out.push_str(&format!(
                    "vec4 {helper_ident}(vec2 uv) {{ return texture({sampler_ident}, uv); }}\n"
                ));
            }
        }

        out.push_str("// --- end auto-synthesized pack prelude ---\n");
        out
    }

    /// Injects raster `.vsl` hook declarations (`struct VsHookWorld`, `void vsHook*(...)`)
    /// **after** `} vsMotion;` (or right before `void main()`) so hook bodies referencing
    /// `VS_PARAM_*` macros can see the `vsMotion` UBO declaration without GLSL compile errors.
    fn inject_raster_hook_prelude(source: &str, hook_prelude: &str) -> String {
        let lines: Vec<&str> = source.lines().collect();
        let insert_idx = lines
            .iter()
            .position(|line| line.trim().starts_with("} vsMotion;"))
            .map(|idx| idx + 1)
            .or_else(|| {
                lines.iter().position(|line| {
                    let t = line.trim();
                    t.starts_with("void main(") || t.starts_with("void main (")
                })
            });

        if let Some(idx) = insert_idx {
            let mut out = String::with_capacity(source.len() + hook_prelude.len() + 16);
            for (i, line) in lines.iter().enumerate() {
                if i == idx {
                    out.push_str(hook_prelude);
                }
                out.push_str(line);
                out.push('\n');
            }
            if idx >= lines.len() {
                out.push_str(hook_prelude);
            }
            out
        } else {
            Self::inject_after_version_and_extensions(source, hook_prelude)
        }
    }

    fn inject_after_version_and_extensions(source: &str, prelude: &str) -> String {
        let lines: Vec<&str> = source.lines().collect();
        let mut insert_after_idx = 0usize;
        let mut has_version = false;
        let mut if_depth = 0usize;
        let mut saw_ext_in_if = false;

        for (idx, line) in lines.iter().enumerate() {
            let t = line.trim();
            if t.starts_with("#version") {
                has_version = true;
                insert_after_idx = idx + 1;
            } else if t.starts_with("#if") {
                if_depth += 1;
            } else if t.starts_with("#endif") {
                if if_depth > 0 {
                    if_depth -= 1;
                    if if_depth == 0 && saw_ext_in_if {
                        insert_after_idx = idx + 1;
                        saw_ext_in_if = false;
                    }
                }
            } else if t.starts_with("#extension") {
                if if_depth == 0 {
                    insert_after_idx = idx + 1;
                } else {
                    saw_ext_in_if = true;
                }
            }
        }

        let mut out = String::with_capacity(source.len() + prelude.len() + 32);
        if !has_version {
            out.push_str("#version 460\n");
        }
        for (idx, line) in lines.iter().enumerate() {
            if idx == insert_after_idx {
                out.push_str(prelude);
            }
            out.push_str(line);
            out.push('\n');
        }
        if insert_after_idx >= lines.len() {
            out.push_str(prelude);
        }
        out
    }

    fn inject_after_version_mapped(
        source: &str,
        source_map: &mut SourceLineMap,
        defines: &str,
        tag: &str,
    ) -> String {
        let lines: Vec<&str> = source.lines().collect();
        let mut insert_after_idx = 0usize;
        let mut has_version = false;

        for (idx, line) in lines.iter().enumerate() {
            if line.trim().starts_with("#version") {
                has_version = true;
                insert_after_idx = idx + 1;
                break;
            }
        }

        let mut out = String::with_capacity(source.len() + defines.len() + 32);
        let mut new_map = SourceLineMap::new();
        if !has_version {
            out.push_str("#version 460\n");
            new_map.push_line(tag, 1);
        }
        let push_block = |out: &mut String, map: &mut SourceLineMap, block: &str| {
            for (bi, bline) in block.lines().enumerate() {
                out.push_str(bline);
                out.push('\n');
                map.push_line(tag, (bi + 1) as u32);
            }
        };
        for (idx, line) in lines.iter().enumerate() {
            if idx == insert_after_idx {
                push_block(&mut out, &mut new_map, defines);
            }
            out.push_str(line);
            out.push('\n');
            if let Some(loc) = source_map.entries().get(idx) {
                new_map.push_line(loc.file.clone(), loc.line);
            } else {
                new_map.push_line(tag, (idx + 1) as u32);
            }
        }
        if insert_after_idx >= lines.len() {
            push_block(&mut out, &mut new_map, defines);
        }
        *source_map = new_map;
        out
    }

    fn inject_after_version_and_extensions_mapped(
        source: &str,
        source_map: &mut SourceLineMap,
        prelude: &str,
        tag: &str,
    ) -> String {
        let lines: Vec<&str> = source.lines().collect();
        let mut insert_after_idx = 0usize;
        let mut has_version = false;
        let mut if_depth = 0usize;
        let mut saw_ext_in_if = false;

        for (idx, line) in lines.iter().enumerate() {
            let t = line.trim();
            if t.starts_with("#version") {
                has_version = true;
                insert_after_idx = idx + 1;
            } else if t.starts_with("#if") {
                if_depth += 1;
            } else if t.starts_with("#endif") {
                if if_depth > 0 {
                    if_depth -= 1;
                    if if_depth == 0 && saw_ext_in_if {
                        insert_after_idx = idx + 1;
                        saw_ext_in_if = false;
                    }
                }
            } else if t.starts_with("#extension") {
                if if_depth == 0 {
                    insert_after_idx = idx + 1;
                } else {
                    saw_ext_in_if = true;
                }
            }
        }

        let mut out = String::with_capacity(source.len() + prelude.len() + 32);
        let mut new_map = SourceLineMap::new();
        if !has_version {
            out.push_str("#version 460\n");
            new_map.push_line(tag, 1);
        }
        let push_block = |out: &mut String, map: &mut SourceLineMap, block: &str| {
            for (bi, bline) in block.lines().enumerate() {
                out.push_str(bline);
                out.push('\n');
                map.push_line(tag, (bi + 1) as u32);
            }
        };
        for (idx, line) in lines.iter().enumerate() {
            if idx == insert_after_idx {
                push_block(&mut out, &mut new_map, prelude);
            }
            out.push_str(line);
            out.push('\n');
            if let Some(loc) = source_map.entries().get(idx) {
                new_map.push_line(loc.file.clone(), loc.line);
            } else {
                new_map.push_line(tag, (idx + 1) as u32);
            }
        }
        if insert_after_idx >= lines.len() {
            push_block(&mut out, &mut new_map, prelude);
        }
        *source_map = new_map;
        out
    }

    /// Built-in standard headers `<vulkanshaders/v1/common.vsl>`, `<vulkanshaders/v1/lighting.vsl>`,
    /// and `<vulkanshaders/v1/restir.vsl>` matching the exact 528-byte `std140` `WorldUBO` layout.
    pub fn builtin_header(path: &str) -> Option<&'static str> {
        match path {
            "vulkanshaders/v1/common.vsl" => Some(BUILTIN_COMMON_VSL),
            "vulkanshaders/v1/lighting.vsl" => Some(BUILTIN_LIGHTING_VSL),
            "vulkanshaders/v1/restir.vsl" => Some(BUILTIN_RESTIR_VSL),
            _ => None,
        }
    }

    pub fn builtin_headers_map() -> BTreeMap<&'static str, &'static str> {
        let mut map = BTreeMap::new();
        map.insert("vulkanshaders/v1/common.vsl", BUILTIN_COMMON_VSL);
        map.insert("vulkanshaders/v1/lighting.vsl", BUILTIN_LIGHTING_VSL);
        map.insert("vulkanshaders/v1/restir.vsl", BUILTIN_RESTIR_VSL);
        map
    }
}

pub const BUILTIN_COMMON_VSL: &str = r#"#ifndef VULKANSHADERS_V1_COMMON_VSL
#define VULKANSHADERS_V1_COMMON_VSL

struct VsVertexInput {
    vec3 localPos;
    vec4 color;
    vec2 uv0;
    ivec2 uv2;
    vec3 normal;
    vec4 tangent;
    vec2 midUv;
    uint materialFlags;
};

struct VsSurface {
    vec3 albedo;
    vec3 normal;
    float roughness;
    float metallic;
    float emission;
    float ao;
    float subsurface;
    float f0;
};

struct VsSkyState {
    vec3 sunDir;
    float rain;
    float thunder;
    float gameTimeSec;
};

struct VsPostContext {
    float exposure;
    float bloomIntensity;
    uint toneMapMode;
};

// Exact 528-byte std140 layout matching WorldStateCollector.kt and vulkanshaders-core WorldUboData:
//   0..63   : curViewProj (mat4)
//  64..127  : prevViewProj (mat4)
// 128..191  : invProj (mat4)
// 192..255  : invView (mat4)
// 256..271  : cameraPosFrac (vec3) + frameIndex (int)
// 272..287  : cameraDelta (vec3) + gameTimeSec (float)
// 288..303  : sunDir (vec3) + rain (float)
// 304..319  : thunder (float) + eyeInFluid (int) + depthFar (float) + heldLight (float)
// 320..511  : packParams[12] (vec4[12] = 48 floats)
// 512..527  : heldLightPosition (vec4)
layout(std140, set = 0, binding = 0) uniform WorldUBO {
    mat4 curViewProj;
    mat4 prevViewProj;
    mat4 invProj;
    mat4 invView;
    vec3 cameraPosFrac;
    int frameIndex;
    vec3 cameraDelta;
    float gameTimeSec;
    vec3 sunDir;
    float rain;
    float thunder;
    int eyeInFluid;
    float depthFar;
    float heldLight;
    vec4 packParams[12];
    vec4 heldLightPosition;
} vsWorld;

#endif
"#;

pub const BUILTIN_LIGHTING_VSL: &str = r#"#ifndef VULKANSHADERS_V1_LIGHTING_VSL
#define VULKANSHADERS_V1_LIGHTING_VSL

#include <vulkanshaders/v1/common.vsl>

float vsTraceShadow(vec3 worldPos, vec3 normal, vec3 lightDir, float coneRadius) {
    float ndotl = max(dot(normal, lightDir), 0.0);
    if (ndotl <= 0.0) {
        return 0.0;
    }
#if defined(VS_CAP_RAY_QUERY_TLAS)
    rayQueryEXT rq;
    vec3 origin = worldPos + normal * max(0.02, coneRadius * 0.5);
    rayQueryInitializeEXT(
        rq,
        VS_CAP_RAY_QUERY_TLAS,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
        0xF5u,
        origin,
        0.01,
        normalize(lightDir),
        192.0
    );
    while (rayQueryProceedEXT(rq)) {}
    return (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT) ? 1.0 : 0.0;
#elif defined(VS_CAP_CSM_SAMPLER)
    return VS_CAP_CSM_SAMPLER(worldPos, normal, lightDir);
#else
    float shadowBias = clamp(coneRadius * 0.5, 0.01, 0.15);
    return smoothstep(0.0, 0.12 + shadowBias, ndotl);
#endif
}

vec3 vsTraceReflection(vec3 worldPos, vec3 normal, vec3 viewDir, float roughness) {
    vec3 refl = reflect(-viewDir, normal);
#if defined(VS_CAP_SSR_TRACE)
    vec4 ssr = VS_CAP_SSR_TRACE(worldPos, normal, viewDir, roughness);
    if (ssr.a > 0.001) {
        float horizon = clamp(refl.y * 0.5 + 0.5, 0.0, 1.0);
        vec3 skyFallback = mix(vec3(0.14, 0.19, 0.27), vec3(0.42, 0.66, 0.98), horizon);
        return mix(skyFallback, ssr.rgb, clamp(ssr.a, 0.0, 1.0)) * (1.0 - roughness * 0.65);
    }
#endif
    float horizon = clamp(refl.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(vec3(0.18, 0.22, 0.28), vec3(0.42, 0.66, 0.98), horizon) * (1.0 - roughness * 0.75);
}

vec3 vsSampleIndirectGI(vec3 worldPos, vec3 normal, vec3 albedo) {
#if defined(VS_CAP_INDIRECT_GI_SAMPLE)
    return VS_CAP_INDIRECT_GI_SAMPLE(worldPos, normal, albedo);
#else
    float skyFactor = clamp(normal.y * 0.5 + 0.5, 0.15, 1.0);
    return albedo * mix(vec3(0.08, 0.10, 0.14), vec3(0.24, 0.30, 0.40), skyFactor);
#endif
}

vec3 evaluateCookTorrance(
    vec3 albedo,
    vec3 N,
    vec3 V,
    vec3 L,
    float roughness,
    float metallic
) {
    float NoV = max(dot(N, V), 1e-4);
    float NoL = max(dot(N, L), 0.0);
    if (NoL <= 0.0) {
        return vec3(0.0);
    }
    vec3 H = normalize(V + L);
    float NoH = max(dot(N, H), 0.0);
    float VoH = max(dot(V, H), 0.0);

    float a = max(roughness * roughness, 0.002);
    float a2 = a * a;
    float denom = (NoH * NoH * (a2 - 1.0) + 1.0);
    float D = a2 / max(3.14159265 * denom * denom, 1e-6);

    float k = (roughness + 1.0);
    k = (k * k) * 0.125;
    float Gv = NoV / (NoV * (1.0 - k) + k);
    float Gl = NoL / (NoL * (1.0 - k) + k);
    float G = Gv * Gl;

    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = F0 + (vec3(1.0) - F0) * pow(clamp(1.0 - VoH, 0.0, 1.0), 5.0);

    vec3 spec = (D * G * F) / max(4.0 * NoV * NoL, 1e-5);
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diff = kD * albedo * (1.0 / 3.14159265);
    return (diff + spec) * NoL;
}

#endif
"#;

pub const BUILTIN_RESTIR_VSL: &str = r#"#ifndef VULKANSHADERS_V1_RESTIR_VSL
#define VULKANSHADERS_V1_RESTIR_VSL

#include <vulkanshaders/v1/common.vsl>

// Exact 48-byte std430 Emissive Point / Triangle Light representation matching GpuEmissiveLight in Rust
struct VsEmissiveLight {
    vec4 posRadius;      // xyz: camera-relative world position, w: attenuation radius
    vec4 colorIntensity; // rgb: linear emissive color, a: luminous intensity
    vec4 normalFlags;    // xyz: emitter normal (0 for omnidirectional), w: uintBitsToFloat(lightTypeFlags)
};

// Exact 32-byte std430 ReSTIR DI Spatiotemporal Reservoir matching GpuRestirReservoir in Rust
struct VsRestirReservoir {
    uint selectedLight;
    float weightSum;
    float targetPdf;
    float M;
    vec3 hitPos;
    uint ageFlags;
};

void vsRestirInit(out VsRestirReservoir r) {
    r.selectedLight = 0xFFFFFFFFu;
    r.weightSum = 0.0;
    r.targetPdf = 0.0;
    r.M = 0.0;
    r.hitPos = vec3(0.0);
    r.ageFlags = 0u;
}

bool vsRestirUpdate(
    inout VsRestirReservoir r,
    uint lightIdx,
    float targetPdf,
    float sourcePdf,
    float rnd
) {
    float risWeight = max(targetPdf, 0.0) / max(sourcePdf, 1.0e-6);
    r.weightSum += risWeight;
    r.M += 1.0;
    if (r.weightSum > 1.0e-7 && rnd * r.weightSum < risWeight) {
        r.selectedLight = lightIdx;
        r.targetPdf = targetPdf;
        return true;
    }
    return false;
}

void vsRestirCombine(
    inout VsRestirReservoir dst,
    VsRestirReservoir temporal,
    float temporalTargetPdf,
    float maxHistoryM,
    float rnd
) {
    if (temporal.selectedLight == 0xFFFFFFFFu || temporal.M <= 0.0 || temporalTargetPdf <= 0.0) {
        return;
    }
    float clampedM = min(temporal.M, maxHistoryM);
    float avgW = temporal.weightSum / max(temporal.targetPdf * temporal.M, 1.0e-6);
    float w = temporalTargetPdf * clamp(avgW, 0.0, 8.0) * clampedM;
    dst.weightSum += w;
    dst.M += clampedM;
    if (dst.weightSum > 1.0e-7 && rnd * dst.weightSum < w) {
        dst.selectedLight = temporal.selectedLight;
        dst.targetPdf = temporalTargetPdf;
    }
}

float vsRestirFinalizeWeight(VsRestirReservoir r) {
    if (r.selectedLight == 0xFFFFFFFFu || r.targetPdf <= 1.0e-6 || r.M <= 0.0) {
        return 0.0;
    }
    return clamp(r.weightSum / max(r.targetPdf * r.M, 1.0e-6), 0.0, 4.0);
}

float vsEvaluateEmissiveTargetPdf(
    VsEmissiveLight light,
    vec3 surfPosRel,
    vec3 surfNormal,
    vec3 albedo,
    out vec3 outToLight,
    out float outDist,
    out vec3 outUnshadowedContrib
) {
    vec3 delta = light.posRadius.xyz - surfPosRel;
    float distSq = dot(delta, delta);
    outDist = sqrt(max(distSq, 1.0e-4));
    outToLight = delta / outDist;
    float radius = clamp(light.posRadius.w, 0.5, 48.0);
    if (outDist >= radius) {
        outUnshadowedContrib = vec3(0.0);
        return 0.0;
    }
    float ndotl = max(dot(surfNormal, outToLight), 0.0);
    if (ndotl <= 0.0) {
        outUnshadowedContrib = vec3(0.0);
        return 0.0;
    }
    float window = clamp(1.0 - pow(outDist / radius, 4.0), 0.0, 1.0);
    float atten = (window * window) / (distSq + 0.5);
    vec3 li = clamp(light.colorIntensity.rgb, vec3(0.0), vec3(4.0)) * clamp(light.colorIntensity.a, 0.0, 16.0) * atten;
    outUnshadowedContrib = (albedo * (1.0 / 3.14159265)) * li * ndotl;
    return dot(outUnshadowedContrib, vec3(0.2126, 0.7152, 0.0722));
}

#endif
"#;
