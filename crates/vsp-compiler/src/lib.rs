//! Vulkan Shader Package (`.vsp`) compiler, `.vsl` hook preprocessor, SPIR-V reflection, and linker.
//!
//! Responsibilities:
//! 1. Parse and validate declarative `pack.toml` manifests (`[pack]`, `[capabilities]`,
//!    `[pipeline]`, `[hooks]`, `[settings.*]`, `[presets.*]`, `[resources.*]`, `[targets.*]`,
//!    `[buffers.*]`, `[[passes]]`), including `extends` base-pack inheritance, directory
//!    validation (`validate_pack_directory`), and programmable `VSP Graph v2` DAGs.
//! 2. Partition pack settings into `runtime = true` (`WorldUBO.packParams[48]`, 0 ms hot-update)
//!    and `runtime = false` (`layout(constant_id = N)` Vulkan Specialization Constants).
//! 3. Expand `#include <vulkanshaders/v1/...>` headers with deduplication and exact `SourceLineMap`
//!    tracking, namespace-isolate `.vsl` hook helpers and entry functions per `(pack_id, stage)`
//!    via a comment/string-aware GLSL lexer to prevent identifier collisions, synthesize
//!    descriptor-free named resource accessors (`vsSamplePackTexture_<name>`), and inject
//!    capability-scaled lighting, SER (`VS_HAS_SER`), OMM (`VS_HAS_OPACITY_MICROMAP`), and ReSTIR
//!    DI primitives for the active `RenderTier` (Tier 0 Raster CSM -> Tier 1 Compute GI ->
//!    Tier 2 RayQuery -> Tier 3 Full Path Tracing).
//! 4. Compile GLSL 460 `.vsh` / `.fsh` / `.comp` / `.rgen` / `.rchit` / `.rmiss` / `.rahit` / `.vsl`
//!    sources into SPIR-V 1.5/1.6 modules (supporting `GL_EXT_ray_query`, `GL_EXT_buffer_reference2`,
//!    `GL_EXT_ray_tracing`, and `GL_EXT_shader_invocation_reorder`), with structured
//!    `CompileDiagnostic` reporting and a deterministic `blake3` `SpirvCacheKey` + `SpirvDiskCache`.
//! 5. Reflect descriptor sets, bindings, push-constant sizes, specialization constants, and
//!    workgroup dimensions from SPIR-V binaries via `rspirv`, and link multiple active `.vsp` hook
//!    layers across all 16 composable `HookStage`s (`@hook(stage = "...", priority = N)`).

pub mod compiler;
pub mod linker;
pub mod manifest;
pub mod preprocessor;

pub use compiler::{
    CompileDiagnostic, CompiledShaderArtifact, ShaderStageKind, SpirvCacheKey,
    SpirvCacheKeyBuilder, SpirvDiskCache, VspCompiler, format_diagnostics_summary,
    parse_compile_diagnostics,
};
pub use linker::{
    CompositeHookCall, ReflectedDescriptorBinding, ReflectedDescriptorType, SpirvHookLinker,
    SpirvReflection,
};
pub use manifest::{
    HookStage, PackBufferDecl, PackCapabilities, PackHeader, PackManifest, PackParameterLayout,
    PackPassDecl, PackPassKind, PackPipelineConfig, PackPresetDecl, PackResourceDecl,
    PackSettingDecl, PackTargetDecl, PackValidationIssue, PackValidationReport, PipelineGraphMode,
    LightingMode, RenderTier, SettingValue, SettingValueType, ValidationSeverity, validate_pack_directory,
};
pub use preprocessor::{
    ExtractedHookBlock, HardwareShaderCaps, HookAnnotation, PreprocessedShaderSource, Preprocessor,
    SourceLineMap, SourceLocation, VslPreprocessor,
};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum VspError {
    #[error("I/O error while reading shader package at '{path}': {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("Invalid pack.toml manifest: {0}")]
    ManifestParse(#[from] toml::de::Error),

    #[error("Manifest validation failed for pack '{pack_id}': {reason}")]
    ManifestValidation { pack_id: String, reason: String },

    #[error(
        "Exceeded maximum #include recursion depth ({max_depth}) while resolving '{include_path}'"
    )]
    IncludeDepthExceeded {
        include_path: String,
        max_depth: usize,
    },

    #[error("Unresolved #include <{include_path}> in '{source_file}'")]
    UnresolvedInclude {
        source_file: String,
        include_path: String,
    },

    #[error("GLSL frontend parse error in '{stage_name}': {details}")]
    GlslParse { stage_name: String, details: String },

    #[error("IR validation error in '{stage_name}': {details}")]
    IrValidation { stage_name: String, details: String },

    #[error("SPIR-V codegen error in '{stage_name}': {details}")]
    SpirvCodegen { stage_name: String, details: String },

    #[error("SPIR-V binary linker/reflection error: {0}")]
    SpirvLink(String),
}
