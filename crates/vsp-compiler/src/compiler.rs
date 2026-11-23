use crate::{
    VspError,
    linker::{SpirvHookLinker, SpirvReflection},
    manifest::{PackManifest, RenderTier},
    preprocessor::{
        HardwareShaderCaps, HookAnnotation, PreprocessedShaderSource, SourceLineMap,
        VslPreprocessor,
    },
};
use naga::ShaderStage;
#[cfg(not(feature = "shaderc-backend"))]
use naga::{
    back::spv,
    front::glsl,
    valid::{Capabilities, ValidationFlags, Validator},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fmt, fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ShaderStageKind {
    Vertex,
    Fragment,
    Compute,
    RayGen,
    ClosestHit,
    Miss,
    AnyHit,
}

impl ShaderStageKind {
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "vsh" | "vert" => Some(Self::Vertex),
            "fsh" | "frag" => Some(Self::Fragment),
            "comp" => Some(Self::Compute),
            "rgen" => Some(Self::RayGen),
            "rchit" => Some(Self::ClosestHit),
            "rmiss" => Some(Self::Miss),
            "rahit" => Some(Self::AnyHit),
            _ => None,
        }
    }

    pub const fn to_naga_stage(self) -> Option<ShaderStage> {
        match self {
            Self::Vertex => Some(ShaderStage::Vertex),
            Self::Fragment => Some(ShaderStage::Fragment),
            Self::Compute => Some(ShaderStage::Compute),
            Self::RayGen | Self::ClosestHit | Self::Miss | Self::AnyHit => None,
        }
    }

    pub const fn to_execution_model(self) -> spirv::ExecutionModel {
        match self {
            Self::Vertex => spirv::ExecutionModel::Vertex,
            Self::Fragment => spirv::ExecutionModel::Fragment,
            Self::Compute => spirv::ExecutionModel::GLCompute,
            Self::RayGen => spirv::ExecutionModel::RayGenerationKHR,
            Self::ClosestHit => spirv::ExecutionModel::ClosestHitKHR,
            Self::Miss => spirv::ExecutionModel::MissKHR,
            Self::AnyHit => spirv::ExecutionModel::AnyHitKHR,
        }
    }

    pub const fn is_ray_tracing_stage(self) -> bool {
        matches!(
            self,
            Self::RayGen | Self::ClosestHit | Self::Miss | Self::AnyHit
        )
    }
}

/// Structured shader compiler diagnostic mapped back to the original `.vsl` / GLSL file and line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompileDiagnostic {
    pub file: String,
    pub line: u32,
    pub column: Option<u32>,
    pub severity: String,
    pub message: String,
    pub raw: String,
}

impl fmt::Display for CompileDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(col) = self.column {
            write!(
                f,
                "{}:{}:{}: {}: {}",
                self.file, self.line, col, self.severity, self.message
            )
        } else {
            write!(
                f,
                "{}:{}: {}: {}",
                self.file, self.line, self.severity, self.message
            )
        }
    }
}

/// Parses compiler error output from `shaderc` (glslang) or `naga` into structured
/// [`CompileDiagnostic`] entries, mapping preprocessed line numbers back through `source_map`
/// to their originating `.vsl` / GLSL file and line number.
pub fn parse_compile_diagnostics(
    raw_error: &str,
    default_file: &str,
    source_map: Option<&SourceLineMap>,
) -> Vec<CompileDiagnostic> {
    let mut diagnostics = Vec::new();
    let lines: Vec<&str> = raw_error.lines().collect();
    let mut pending_naga_msg: Option<(String, String)> = None;

    for raw_line in &lines {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Skip glslang trailing summary lines
        if trimmed.ends_with("errors generated.")
            || trimmed.ends_with("error generated.")
            || trimmed.eq_ignore_ascii_case("compilation terminated")
            || trimmed.starts_with("ERROR: compilation terminated")
        {
            continue;
        }

        // 1. Naga codespan header: `error: <msg>` followed by `┌─ <file>:<line>:<col>`
        if let Some(rest) = trimmed.strip_prefix("error:") {
            pending_naga_msg = Some(("error".to_owned(), rest.trim().to_owned()));
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("warning:") {
            pending_naga_msg = Some(("warning".to_owned(), rest.trim().to_owned()));
            continue;
        }
        if let Some(loc_spec) = trimmed
            .strip_prefix("┌─")
            .or_else(|| trimmed.strip_prefix("-->"))
        {
            let loc_spec = loc_spec.trim();
            if let Some((parsed_file, parsed_line, parsed_col)) = parse_file_line_col(loc_spec) {
                let (file, line) = resolve_via_map(parsed_file, parsed_line, source_map);
                let (severity, message) = pending_naga_msg
                    .take()
                    .unwrap_or_else(|| ("error".to_owned(), loc_spec.to_owned()));
                diagnostics.push(CompileDiagnostic {
                    file,
                    line,
                    column: parsed_col,
                    severity,
                    message,
                    raw: (*raw_line).to_owned(),
                });
                continue;
            }
        }

        // 2. glslang / shaderc format:
        //    `ERROR: <file_or_0>:<line>: <msg>`
        //    `WARNING: <file_or_0>:<line>: <msg>`
        //    `<file>:<line>[:<col>]: [error|warning:] <msg>`
        if let Some(diag) = parse_glslang_line(trimmed, default_file, source_map) {
            diagnostics.push(diag);
            continue;
        }

        // 3. Naga ParseError debug format containing `line_number: <N>`
        if let Some(pos) = trimmed.find("line_number:") {
            let after = &trimmed[pos + "line_number:".len()..];
            let num_str: String = after
                .trim_start()
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(parsed_line) = num_str.parse::<u32>()
                && parsed_line > 0
            {
                let col = trimmed.find("line_position:").and_then(|cpos| {
                    let cafter = &trimmed[cpos + "line_position:".len()..];
                    let cstr: String = cafter
                        .trim_start()
                        .chars()
                        .take_while(|c| c.is_ascii_digit())
                        .collect();
                    cstr.parse::<u32>().ok()
                });
                let (file, line) = resolve_via_map(default_file, parsed_line, source_map);
                diagnostics.push(CompileDiagnostic {
                    file,
                    line,
                    column: col,
                    severity: "error".to_owned(),
                    message: trimmed.to_owned(),
                    raw: (*raw_line).to_owned(),
                });
            }
        }
    }

    if diagnostics.is_empty() && !raw_error.trim().is_empty() {
        let (file, line) = resolve_via_map(default_file, 1, source_map);
        diagnostics.push(CompileDiagnostic {
            file,
            line,
            column: None,
            severity: "error".to_owned(),
            message: raw_error.trim().to_owned(),
            raw: raw_error.to_owned(),
        });
    }

    diagnostics
}

fn resolve_via_map(
    parsed_file: &str,
    parsed_line: u32,
    source_map: Option<&SourceLineMap>,
) -> (String, u32) {
    if let Some(map) = source_map
        && let Some(loc) = map.resolve(parsed_line)
    {
        return (loc.file.clone(), loc.line);
    }
    (parsed_file.to_owned(), parsed_line)
}

fn parse_file_line_col(spec: &str) -> Option<(&str, u32, Option<u32>)> {
    // Parse trailing `:<col>` and `:<line>` from the right so Windows drive letters work.
    let (head, last) = spec.rsplit_once(':')?;
    let last_num = last.trim().parse::<u32>().ok()?;
    if let Some((file_part, mid)) = head.rsplit_once(':')
        && let Ok(mid_num) = mid.trim().parse::<u32>()
        && mid_num > 0
    {
        return Some((file_part.trim(), mid_num, Some(last_num)));
    }
    if last_num > 0 {
        return Some((head.trim(), last_num, None));
    }
    None
}

fn parse_glslang_line(
    trimmed: &str,
    default_file: &str,
    source_map: Option<&SourceLineMap>,
) -> Option<CompileDiagnostic> {
    // Strip wrapper prefixes emitted by `shaderc::Error::Display` (e.g. "compilation error: ...").
    let unwrapped = trimmed
        .strip_prefix("shader compilation error:")
        .or_else(|| trimmed.strip_prefix("compilation error:"))
        .map(str::trim)
        .unwrap_or(trimmed);

    let (prefix_severity, rest) = if let Some(r) = unwrapped.strip_prefix("ERROR:") {
        (Some("error"), r.trim())
    } else if let Some(r) = unwrapped.strip_prefix("WARNING:") {
        (Some("warning"), r.trim())
    } else if let Some(r) = unwrapped.strip_prefix("NOTE:") {
        (Some("note"), r.trim())
    } else {
        (None, unwrapped)
    };

    // Find `<file>:<line>:` or `<file>:<line>:<col>:`
    let first_colon = rest.find(':')?;
    let file_candidate = rest[..first_colon].trim();
    if file_candidate.is_empty() || file_candidate.contains(' ') {
        return None;
    }
    let after_file = &rest[first_colon + 1..];
    let second_colon = after_file.find(':')?;
    let line_str = after_file[..second_colon].trim();
    let parsed_line = line_str.parse::<u32>().ok()?;
    if parsed_line == 0 {
        return None;
    }

    let mut remainder = after_file[second_colon + 1..].trim();
    let mut column: Option<u32> = None;
    if let Some(third_colon) = remainder.find(':') {
        let maybe_col = remainder[..third_colon].trim();
        if let Ok(col_num) = maybe_col.parse::<u32>() {
            column = Some(col_num);
            remainder = remainder[third_colon + 1..].trim();
        }
    }

    let (severity, message) = if let Some(msg) = remainder.strip_prefix("error:") {
        ("error".to_owned(), msg.trim().to_owned())
    } else if let Some(msg) = remainder.strip_prefix("warning:") {
        ("warning".to_owned(), msg.trim().to_owned())
    } else if let Some(msg) = remainder.strip_prefix("note:") {
        ("note".to_owned(), msg.trim().to_owned())
    } else {
        (
            prefix_severity.unwrap_or("error").to_owned(),
            remainder.to_owned(),
        )
    };

    let effective_file =
        if file_candidate == "0" || file_candidate.chars().all(|c| c.is_ascii_digit()) {
            default_file
        } else {
            file_candidate
        };
    let (file, line) = resolve_via_map(effective_file, parsed_line, source_map);

    Some(CompileDiagnostic {
        file,
        line,
        column,
        severity,
        message,
        raw: trimmed.to_owned(),
    })
}

pub fn format_diagnostics_summary(diagnostics: &[CompileDiagnostic], fallback_raw: &str) -> String {
    if diagnostics.is_empty() {
        return fallback_raw.to_owned();
    }
    diagnostics
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Debug, Clone)]
pub struct CompiledShaderArtifact {
    pub stage_name: String,
    pub stage_kind: ShaderStageKind,
    pub spirv_words: Vec<u32>,
    pub content_hash: [u8; 32],
    pub hooks: Vec<HookAnnotation>,
    pub reflection: SpirvReflection,
    pub loaded_from_cache: bool,
}

/// Deterministic 256-bit `blake3` cache key for compiled SPIR-V binaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SpirvCacheKey([u8; 32]);

impl SpirvCacheKey {
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(&self) -> String {
        let mut hex = String::with_capacity(64);
        for b in self.0 {
            use std::fmt::Write;
            let _ = write!(&mut hex, "{b:02x}");
        }
        hex
    }

    pub fn builder(pack_hash: [u8; 32], shader_stage: ShaderStageKind) -> SpirvCacheKeyBuilder {
        SpirvCacheKeyBuilder::new(pack_hash, shader_stage)
    }
}

impl fmt::Display for SpirvCacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

/// Deterministic builder for [`SpirvCacheKey`] hashing
/// `(pack_hash, shader_stage, entry_point, driver_uuid, spec_constants, defines)` via `blake3`.
#[derive(Debug, Clone)]
pub struct SpirvCacheKeyBuilder {
    pack_hash: [u8; 32],
    shader_stage: ShaderStageKind,
    entry_point: String,
    driver_uuid: [u8; 16],
    spec_constants: BTreeMap<u32, u32>,
    defines: BTreeMap<String, Option<String>>,
    hw_caps_bits: u8,
    debug_info: bool,
}

impl SpirvCacheKeyBuilder {
    const DOMAIN_TAG: &'static [u8] = b"hyaloclast.vsp.spirv_cache_key.v1";

    pub fn new(pack_hash: [u8; 32], shader_stage: ShaderStageKind) -> Self {
        Self {
            pack_hash,
            shader_stage,
            entry_point: "main".to_owned(),
            driver_uuid: [0u8; 16],
            spec_constants: BTreeMap::new(),
            defines: BTreeMap::new(),
            hw_caps_bits: 0,
            debug_info: false,
        }
    }

    pub fn entry_point(mut self, entry: impl Into<String>) -> Self {
        self.entry_point = entry.into();
        self
    }

    pub fn driver_uuid(mut self, uuid: [u8; 16]) -> Self {
        self.driver_uuid = uuid;
        self
    }

    pub fn spec_constant(mut self, id: u32, raw_bits: u32) -> Self {
        self.spec_constants.insert(id, raw_bits);
        self
    }

    pub fn spec_constants<I>(mut self, iter: I) -> Self
    where
        I: IntoIterator<Item = (u32, u32)>,
    {
        for (id, bits) in iter {
            self.spec_constants.insert(id, bits);
        }
        self
    }

    pub fn define(mut self, name: impl Into<String>, value: Option<String>) -> Self {
        self.defines.insert(name.into(), value);
        self
    }

    pub fn defines<I, K, V>(mut self, iter: I) -> Self
    where
        I: IntoIterator<Item = (K, Option<V>)>,
        K: Into<String>,
        V: Into<String>,
    {
        for (k, v) in iter {
            self.defines.insert(k.into(), v.map(Into::into));
        }
        self
    }

    pub fn hardware_caps(mut self, caps: HardwareShaderCaps) -> Self {
        self.hw_caps_bits = caps.cache_bits();
        self
    }

    pub fn debug_info(mut self, enabled: bool) -> Self {
        self.debug_info = enabled;
        self
    }

    pub fn build(&self) -> SpirvCacheKey {
        let mut hasher = blake3::Hasher::new();
        hasher.update(Self::DOMAIN_TAG);
        hasher.update(&self.pack_hash);
        hasher.update(&[
            self.shader_stage as u8,
            self.hw_caps_bits,
            u8::from(self.debug_info),
        ]);
        hasher.update(&(self.entry_point.len() as u32).to_le_bytes());
        hasher.update(self.entry_point.as_bytes());
        hasher.update(&self.driver_uuid);

        hasher.update(&(self.spec_constants.len() as u32).to_le_bytes());
        for (&id, &bits) in &self.spec_constants {
            hasher.update(&id.to_le_bytes());
            hasher.update(&bits.to_le_bytes());
        }

        hasher.update(&(self.defines.len() as u32).to_le_bytes());
        for (k, v) in &self.defines {
            hasher.update(&(k.len() as u32).to_le_bytes());
            hasher.update(k.as_bytes());
            match v {
                Some(val) => {
                    hasher.update(&[1u8]);
                    hasher.update(&(val.len() as u32).to_le_bytes());
                    hasher.update(val.as_bytes());
                }
                None => {
                    hasher.update(&[0u8]);
                }
            }
        }

        SpirvCacheKey(*hasher.finalize().as_bytes())
    }
}

/// Content-addressed SPIR-V binary cache keyed by `blake3` 256-bit keys, stored atomically
/// (temp write + rename) with a versioned magic header.
#[derive(Debug, Clone)]
pub struct SpirvDiskCache {
    root_dir: Option<PathBuf>,
}

impl SpirvDiskCache {
    pub const CACHE_MAGIC: [u8; 8] = *b"VSPCACHE";
    pub const CACHE_VERSION: u32 = 1;
    const HEADER_BYTES: usize = 16; // 8-byte magic + 4-byte version + 4-byte word count

    pub fn new(root_dir: Option<PathBuf>) -> Self {
        if let Some(ref dir) = root_dir {
            let _ = fs::create_dir_all(dir);
        }
        Self { root_dir }
    }

    pub fn root_dir(&self) -> Option<&Path> {
        self.root_dir.as_deref()
    }

    pub fn cache_path_for_key(&self, key: &SpirvCacheKey) -> Option<PathBuf> {
        self.cache_path(key.as_bytes())
    }

    fn cache_path(&self, hash: &[u8; 32]) -> Option<PathBuf> {
        let dir = self.root_dir.as_ref()?;
        let mut hex = String::with_capacity(68);
        for b in hash {
            use std::fmt::Write;
            let _ = write!(&mut hex, "{b:02x}");
        }
        hex.push_str(".spv");
        Some(dir.join(hex))
    }

    pub fn try_load_key(&self, key: &SpirvCacheKey) -> Option<Vec<u32>> {
        self.try_load(key.as_bytes())
    }

    pub fn store_key(&self, key: &SpirvCacheKey, words: &[u32]) {
        self.store(key.as_bytes(), words);
    }

    pub fn try_load(&self, hash: &[u8; 32]) -> Option<Vec<u32>> {
        let path = self.cache_path(hash)?;
        let bytes = fs::read(path).ok()?;
        if bytes.len() < Self::HEADER_BYTES + 20 {
            return None;
        }
        if bytes[0..8] != Self::CACHE_MAGIC {
            return None;
        }
        let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        if version != Self::CACHE_VERSION {
            return None;
        }
        let expected_words =
            u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        let payload = &bytes[Self::HEADER_BYTES..];
        if !payload.len().is_multiple_of(4) || payload.len() != expected_words * 4 {
            return None;
        }
        let words: Vec<u32> = payload
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect();
        if words.first().copied() != Some(spirv::MAGIC_NUMBER) {
            return None;
        }
        Some(words)
    }

    pub fn store(&self, hash: &[u8; 32], words: &[u32]) {
        static TMP_SEQ: AtomicU64 = AtomicU64::new(0);
        let Some(path) = self.cache_path(hash) else {
            return;
        };
        let payload: &[u8] = bytemuck::cast_slice(words);
        let mut blob = Vec::with_capacity(Self::HEADER_BYTES + payload.len());
        blob.extend_from_slice(&Self::CACHE_MAGIC);
        blob.extend_from_slice(&Self::CACHE_VERSION.to_le_bytes());
        blob.extend_from_slice(&(words.len() as u32).to_le_bytes());
        blob.extend_from_slice(payload);

        let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let tmp_name = format!(
            "{}.{pid}.{seq}.tmp",
            path.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("shader")
        );
        let tmp_path = path.with_file_name(tmp_name);
        if fs::write(&tmp_path, &blob).is_ok() {
            if fs::rename(&tmp_path, &path).is_err() {
                let _ = fs::remove_file(&tmp_path);
            }
        } else {
            let _ = fs::remove_file(&tmp_path);
        }
    }
}

/// Compiles GLSL 460 + `.vsl` shader stages (`.vsh`, `.fsh`, `.comp`, `.rgen`, `.rchit`, `.rmiss`, `.rahit`)
/// into SPIR-V 1.5/1.6 modules with full descriptor and push-constant reflection.
pub struct VspCompiler {
    preprocessor: VslPreprocessor,
    cache: SpirvDiskCache,
}

impl VspCompiler {
    pub fn new(cache_dir: Option<&Path>) -> Self {
        Self {
            preprocessor: VslPreprocessor::default(),
            cache: SpirvDiskCache::new(cache_dir.map(Path::to_path_buf)),
        }
    }

    /// Returns the preprocessor macro definitions corresponding to `hw_caps`
    /// (`VS_HAS_SER`, `VS_HAS_SER_EXT`, `VS_HAS_SER_NV`, `VS_GPU_ADA_PLUS`,
    /// `VS_HIGH_END_GPU`, `VS_HAS_OPACITY_MICROMAP`).
    pub fn hardware_caps_defines(hw_caps: HardwareShaderCaps) -> Vec<(String, Option<String>)> {
        let mut defines = Vec::new();
        if hw_caps.ser_ext {
            defines.push(("VS_HAS_SER_EXT".to_owned(), Some("1".to_owned())));
        }
        if hw_caps.ser_nv {
            defines.push(("VS_HAS_SER_NV".to_owned(), Some("1".to_owned())));
        }
        if hw_caps.has_ser() {
            defines.push(("VS_HAS_SER".to_owned(), Some("1".to_owned())));
        }
        if hw_caps.ada_or_newer {
            defines.push(("VS_GPU_ADA_PLUS".to_owned(), Some("1".to_owned())));
        }
        if hw_caps.high_end_gpu {
            defines.push(("VS_HIGH_END_GPU".to_owned(), Some("1".to_owned())));
        }
        if hw_caps.opacity_micromap {
            defines.push(("VS_HAS_OPACITY_MICROMAP".to_owned(), Some("1".to_owned())));
        }
        defines
    }

    /// Parses compiler diagnostics from `raw_error`, resolving line numbers through `source_map`.
    pub fn parse_diagnostics(
        raw_error: &str,
        default_file: &str,
        source_map: Option<&SourceLineMap>,
    ) -> Vec<CompileDiagnostic> {
        parse_compile_diagnostics(raw_error, default_file, source_map)
    }

    fn stage_cache_hash(
        base_hash: &[u8; 32],
        stage_kind: ShaderStageKind,
        hw_caps: HardwareShaderCaps,
    ) -> [u8; 32] {
        let debug_flag = std::env::var_os("VS_SHADER_DEBUG_INFO").is_some();
        *SpirvCacheKey::builder(*base_hash, stage_kind)
            .hardware_caps(hw_caps)
            .debug_info(debug_flag)
            .build()
            .as_bytes()
    }

    pub fn compile_stage<F>(
        &self,
        stage_name: &str,
        stage_kind: ShaderStageKind,
        raw_source: &str,
        manifest: &PackManifest,
        tier: RenderTier,
        include_loader: &F,
    ) -> Result<CompiledShaderArtifact, VspError>
    where
        F: Fn(&str) -> Option<String>,
    {
        self.compile_stage_with_caps(
            stage_name,
            stage_kind,
            raw_source,
            manifest,
            tier,
            HardwareShaderCaps::default(),
            include_loader,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compile_stage_with_caps<F>(
        &self,
        stage_name: &str,
        stage_kind: ShaderStageKind,
        raw_source: &str,
        manifest: &PackManifest,
        tier: RenderTier,
        hw_caps: HardwareShaderCaps,
        include_loader: &F,
    ) -> Result<CompiledShaderArtifact, VspError>
    where
        F: Fn(&str) -> Option<String>,
    {
        let PreprocessedShaderSource {
            glsl_source,
            discovered_hooks,
            content_hash: preprocessed_hash,
            source_map,
        } = self.preprocessor.preprocess_with_caps(
            stage_name,
            raw_source,
            manifest,
            tier,
            hw_caps,
            include_loader,
        )?;

        let content_hash = Self::stage_cache_hash(&preprocessed_hash, stage_kind, hw_caps);

        if let Some(cached_words) = self.cache.try_load(&content_hash)
            && let Ok(reflection) = SpirvHookLinker::reflect_module(&cached_words)
        {
            return Ok(CompiledShaderArtifact {
                stage_name: stage_name.to_owned(),
                stage_kind,
                spirv_words: cached_words,
                content_hash,
                hooks: discovered_hooks,
                reflection,
                loaded_from_cache: true,
            });
        }

        let spirv_words = Self::compile_glsl_to_spirv_with_defines(
            stage_name,
            stage_kind,
            &glsl_source,
            &[],
            Some(&source_map),
        )?;
        let reflection = SpirvHookLinker::reflect_module(&spirv_words)?;
        self.cache.store(&content_hash, &spirv_words);

        Ok(CompiledShaderArtifact {
            stage_name: stage_name.to_owned(),
            stage_kind,
            spirv_words,
            content_hash,
            hooks: discovered_hooks,
            reflection,
            loaded_from_cache: false,
        })
    }

    /// Compiles a shader stage and returns structured [`CompileDiagnostic`] entries mapped back
    /// to original `.vsl` / GLSL file paths and line numbers on failure.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_stage_with_diagnostics<F>(
        &self,
        stage_name: &str,
        stage_kind: ShaderStageKind,
        raw_source: &str,
        manifest: &PackManifest,
        tier: RenderTier,
        hw_caps: HardwareShaderCaps,
        include_loader: &F,
    ) -> Result<CompiledShaderArtifact, Vec<CompileDiagnostic>>
    where
        F: Fn(&str) -> Option<String>,
    {
        self.compile_stage_with_caps(
            stage_name,
            stage_kind,
            raw_source,
            manifest,
            tier,
            hw_caps,
            include_loader,
        )
        .map_err(|err| match &err {
            VspError::GlslParse { details, .. }
            | VspError::IrValidation { details, .. }
            | VspError::SpirvCodegen { details, .. } => {
                parse_compile_diagnostics(details, stage_name, None)
            }
            other => vec![CompileDiagnostic {
                file: stage_name.to_owned(),
                line: 1,
                column: None,
                severity: "error".to_owned(),
                message: other.to_string(),
                raw: other.to_string(),
            }],
        })
    }

    /// Standalone convenience helper for compiling a single GLSL stage with optional specialization patches.
    pub fn compile_glsl_stage(
        &mut self,
        stage_name: &str,
        raw_source: &str,
        stage_kind: ShaderStageKind,
        spec_overrides: &[(u32, u32)],
    ) -> Result<CompiledShaderArtifact, VspError> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(stage_name.as_bytes());
        hasher.update(&[stage_kind as u8]);
        let debug_flag = u8::from(std::env::var_os("VS_SHADER_DEBUG_INFO").is_some());
        hasher.update(&[debug_flag]);
        hasher.update(raw_source.as_bytes());
        let content_hash = *hasher.finalize().as_bytes();

        let (base_words, loaded_from_cache) =
            if let Some(cached) = self.cache.try_load(&content_hash) {
                (cached, true)
            } else {
                let words = Self::compile_glsl_to_spirv(stage_name, stage_kind, raw_source)?;
                self.cache.store(&content_hash, &words);
                (words, false)
            };

        let patched_words = if spec_overrides.is_empty() {
            base_words
        } else {
            SpirvHookLinker::patch_specialization_constants(&base_words, spec_overrides)?
        };
        let reflection = SpirvHookLinker::reflect_module(&patched_words)?;

        Ok(CompiledShaderArtifact {
            stage_name: stage_name.to_owned(),
            stage_kind,
            spirv_words: patched_words,
            content_hash,
            hooks: Vec::new(),
            reflection,
            loaded_from_cache,
        })
    }

    /// Resolves a pack's `extends` inheritance chain (child -> parent -> ... -> root),
    /// returning the merged `PackManifest` and the ordered list of pack directories
    /// (child first, ancestors after) for shader/hook/include fallback lookup.
    pub fn resolve_pack_inheritance(
        pack_dir: &Path,
        extra_search_roots: &[PathBuf],
    ) -> Result<(PackManifest, Vec<PathBuf>), VspError> {
        let mut chain_dirs: Vec<PathBuf> = Vec::new();
        let mut chain_manifests: Vec<PackManifest> = Vec::new();
        let mut visited: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
        let mut cur_dir = pack_dir.to_path_buf();

        let mut resolved_root = false;
        for _ in 0..8 {
            let canonical = fs::canonicalize(&cur_dir).unwrap_or_else(|_| cur_dir.clone());
            if !visited.insert(canonical) {
                let pack_id = chain_manifests
                    .first()
                    .map(|m| m.pack.id.clone())
                    .unwrap_or_else(|| pack_dir.display().to_string());
                return Err(VspError::ManifestValidation {
                    pack_id,
                    reason: format!(
                        "Cyclic pack inheritance detected at '{}'",
                        cur_dir.display()
                    ),
                });
            }
            let manifest_path = cur_dir.join("pack.toml");
            let manifest_raw =
                fs::read_to_string(&manifest_path).map_err(|source| VspError::Io {
                    path: manifest_path.display().to_string(),
                    source,
                })?;
            let manifest = PackManifest::parse_and_validate(&manifest_raw)?;
            let parent_id = manifest.pack.extends.clone();
            chain_dirs.push(cur_dir.clone());
            chain_manifests.push(manifest);

            let Some(parent_id) = parent_id.filter(|s| !s.trim().is_empty()) else {
                resolved_root = true;
                break;
            };
            let mut candidates: Vec<PathBuf> = Vec::new();
            if let Some(parent_root) = cur_dir.parent() {
                candidates.push(parent_root.join(format!("{parent_id}.vsp")));
                candidates.push(parent_root.join(&parent_id));
            }
            for root in extra_search_roots {
                candidates.push(root.join(format!("{parent_id}.vsp")));
                candidates.push(root.join(&parent_id));
            }
            let Some(found) = candidates
                .into_iter()
                .find(|c| c.join("pack.toml").is_file())
            else {
                return Err(VspError::Io {
                    path: format!("extends '{parent_id}' (parent pack.toml not found)"),
                    source: std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("Parent pack '{parent_id}' not found"),
                    ),
                });
            };
            cur_dir = found;
        }
        if !resolved_root {
            let pack_id = chain_manifests
                .first()
                .map(|m| m.pack.id.clone())
                .unwrap_or_else(|| pack_dir.display().to_string());
            return Err(VspError::ManifestValidation {
                pack_id,
                reason: "Pack inheritance depth exceeds maximum limit of 8".to_owned(),
            });
        }

        // Fold from oldest ancestor down to child via `inherit_from`
        let mut iter = chain_manifests.into_iter().rev();
        let mut merged = iter.next().ok_or_else(|| VspError::Io {
            path: pack_dir.display().to_string(),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "Empty pack chain"),
        })?;
        for child in iter {
            merged = child.inherit_from(&merged);
        }
        merged.validate()?;
        Ok((merged, chain_dirs))
    }

    /// Scans and compiles all `.vsh`, `.fsh`, `.comp`, `.rgen`, `.rchit`, `.rmiss`, and `.rahit`
    /// shaders in `shaders/core`, `shaders/compute`, and `shaders/rt` inside a `.vsp` directory
    /// (resolving `extends` inheritance and compiling stages in parallel across CPU cores).
    pub fn compile_pack_dir(
        &self,
        pack_dir: &Path,
        tier: RenderTier,
    ) -> Result<Vec<CompiledShaderArtifact>, VspError> {
        self.compile_pack_dir_with_roots_and_caps(
            pack_dir,
            tier,
            &[],
            HardwareShaderCaps::default(),
        )
    }

    pub fn compile_pack_dir_with_caps(
        &self,
        pack_dir: &Path,
        tier: RenderTier,
        hw_caps: HardwareShaderCaps,
    ) -> Result<Vec<CompiledShaderArtifact>, VspError> {
        self.compile_pack_dir_with_roots_and_caps(pack_dir, tier, &[], hw_caps)
    }

    pub fn compile_pack_dir_with_roots(
        &self,
        pack_dir: &Path,
        tier: RenderTier,
        extra_search_roots: &[PathBuf],
    ) -> Result<Vec<CompiledShaderArtifact>, VspError> {
        self.compile_pack_dir_with_roots_and_caps(
            pack_dir,
            tier,
            extra_search_roots,
            HardwareShaderCaps::default(),
        )
    }

    pub fn compile_pack_dir_with_roots_and_caps(
        &self,
        pack_dir: &Path,
        tier: RenderTier,
        extra_search_roots: &[PathBuf],
        hw_caps: HardwareShaderCaps,
    ) -> Result<Vec<CompiledShaderArtifact>, VspError> {
        let (manifest, chain_dirs) = Self::resolve_pack_inheritance(pack_dir, extra_search_roots)?;

        let loader = |inc: &str| -> Option<String> {
            for dir in &chain_dirs {
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

        // Collect unique shader stages across the inheritance chain (child overrides parent).
        let mut stage_jobs: Vec<(String, ShaderStageKind, String)> = Vec::new();
        let mut seen_stages: std::collections::HashSet<(String, String)> =
            std::collections::HashSet::new();

        for sub in ["core", "compute", "rt"] {
            let mut per_sub: std::collections::BTreeMap<String, (ShaderStageKind, PathBuf)> =
                std::collections::BTreeMap::new();
            for dir in &chain_dirs {
                let sub_dir = dir.join("shaders").join(sub);
                if !sub_dir.is_dir() {
                    continue;
                }
                let entries = fs::read_dir(&sub_dir).map_err(|source| VspError::Io {
                    path: sub_dir.display().to_string(),
                    source,
                })?;
                for entry in entries.flatten() {
                    let path = entry.path();
                    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
                        continue;
                    };
                    let Some(stage_kind) = ShaderStageKind::from_extension(ext) else {
                        continue;
                    };
                    let Some(file_name) = path.file_name().and_then(|s| s.to_str()) else {
                        continue;
                    };
                    if seen_stages.insert((sub.to_owned(), file_name.to_owned())) {
                        per_sub.insert(file_name.to_owned(), (stage_kind, path));
                    }
                }
            }

            for (file_name, (stage_kind, path)) in per_sub {
                let raw_source = fs::read_to_string(&path).map_err(|source| VspError::Io {
                    path: path.display().to_string(),
                    source,
                })?;
                stage_jobs.push((file_name, stage_kind, raw_source));
            }
        }

        if stage_jobs.len() <= 2 {
            let mut artifacts = Vec::with_capacity(stage_jobs.len());
            for (file_name, stage_kind, raw_source) in &stage_jobs {
                artifacts.push(self.compile_stage_with_caps(
                    file_name,
                    *stage_kind,
                    raw_source,
                    &manifest,
                    tier,
                    hw_caps,
                    &loader,
                )?);
            }
            return Ok(artifacts);
        }

        let worker_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .clamp(1, stage_jobs.len().min(16));
        let chunk_size = stage_jobs.len().div_ceil(worker_count);

        let results: Vec<Result<Vec<CompiledShaderArtifact>, VspError>> =
            std::thread::scope(|scope| {
                let mut handles = Vec::new();
                let compiler = self;
                let manifest = &manifest;
                let loader = &loader;
                for chunk in stage_jobs.chunks(chunk_size) {
                    handles.push(scope.spawn(move || {
                        let mut local = Vec::with_capacity(chunk.len());
                        for (file_name, stage_kind, raw_source) in chunk {
                            local.push(compiler.compile_stage_with_caps(
                                file_name,
                                *stage_kind,
                                raw_source,
                                manifest,
                                tier,
                                hw_caps,
                                loader,
                            )?);
                        }
                        Ok(local)
                    }));
                }
                handles
                    .into_iter()
                    .map(|h| {
                        h.join().unwrap_or_else(|_| {
                            Err(VspError::SpirvCodegen {
                                stage_name: "parallel_worker".to_owned(),
                                details: "Shader compilation worker panicked".to_owned(),
                            })
                        })
                    })
                    .collect()
            });

        let mut artifacts = Vec::with_capacity(stage_jobs.len());
        for batch in results {
            artifacts.extend(batch?);
        }
        Ok(artifacts)
    }

    pub fn compile_glsl_to_spirv(
        stage_name: &str,
        stage_kind: ShaderStageKind,
        glsl_source: &str,
    ) -> Result<Vec<u32>, VspError> {
        Self::compile_glsl_to_spirv_with_defines(stage_name, stage_kind, glsl_source, &[], None)
    }

    /// Compiles GLSL 460 directly to SPIR-V 1.6 / Vulkan 1.3 with hardware capability defines
    /// (`VS_HAS_SER`, `VS_HAS_SER_EXT`, `VS_HAS_SER_NV`, `VS_HAS_OPACITY_MICROMAP`, etc.)
    /// and optional [`SourceLineMap`] diagnostic mapping.
    pub fn compile_glsl_to_spirv_with_caps(
        stage_name: &str,
        stage_kind: ShaderStageKind,
        glsl_source: &str,
        hw_caps: HardwareShaderCaps,
        source_map: Option<&SourceLineMap>,
    ) -> Result<Vec<u32>, VspError> {
        let defines = Self::hardware_caps_defines(hw_caps);
        Self::compile_glsl_to_spirv_with_defines(
            stage_name,
            stage_kind,
            glsl_source,
            &defines,
            source_map,
        )
    }

    /// Compiles GLSL 460 to SPIR-V 1.6 / Vulkan 1.3 with explicit macro definitions and optional
    /// [`SourceLineMap`] error mapping.
    pub fn compile_glsl_to_spirv_with_defines(
        stage_name: &str,
        stage_kind: ShaderStageKind,
        glsl_source: &str,
        defines: &[(String, Option<String>)],
        source_map: Option<&SourceLineMap>,
    ) -> Result<Vec<u32>, VspError> {
        #[cfg(feature = "shaderc-backend")]
        {
            Self::compile_via_shaderc(stage_name, stage_kind, glsl_source, defines, source_map)
        }

        #[cfg(not(feature = "shaderc-backend"))]
        {
            Self::compile_via_naga(stage_name, stage_kind, glsl_source, defines, source_map)
        }
    }

    #[cfg(not(feature = "shaderc-backend"))]
    fn compile_via_naga(
        stage_name: &str,
        stage_kind: ShaderStageKind,
        glsl_source: &str,
        defines: &[(String, Option<String>)],
        source_map: Option<&SourceLineMap>,
    ) -> Result<Vec<u32>, VspError> {
        let requires_vulkan_rt_or_bda = stage_kind.is_ray_tracing_stage()
            || glsl_source.contains("GL_EXT_buffer_reference")
            || glsl_source.contains("GL_EXT_ray_query")
            || glsl_source.contains("GL_EXT_ray_tracing")
            || glsl_source.contains("GL_EXT_shader_invocation_reorder")
            || glsl_source.contains("GL_NV_shader_invocation_reorder")
            || glsl_source.contains("GL_EXT_opacity_micromap");

        if requires_vulkan_rt_or_bda {
            return Err(VspError::SpirvCodegen {
                stage_name: stage_name.to_owned(),
                details: "Vulkan ray tracing, SER, OMM, ray query, and buffer device address shaders require the 'shaderc-backend' feature".to_owned(),
            });
        }

        let Some(naga_stage) = stage_kind.to_naga_stage() else {
            return Err(VspError::SpirvCodegen {
                stage_name: stage_name.to_owned(),
                details: format!("Unsupported stage {stage_kind:?} without 'shaderc-backend'"),
            });
        };

        let mut frontend = glsl::Frontend::default();
        let mut options = glsl::Options::from(naga_stage);
        for (name, val) in defines {
            options
                .defines
                .insert(name.clone(), val.clone().unwrap_or_default());
        }

        let module = frontend.parse(&options, glsl_source).map_err(|errors| {
            let raw = format!("{errors:?}");
            let diags = parse_compile_diagnostics(&raw, stage_name, source_map);
            VspError::GlslParse {
                stage_name: stage_name.to_owned(),
                details: format_diagnostics_summary(&diags, &raw),
            }
        })?;

        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        let module_info = validator.validate(&module).map_err(|err| {
            let raw = err.emit_to_string(glsl_source);
            let diags = parse_compile_diagnostics(&raw, stage_name, source_map);
            VspError::IrValidation {
                stage_name: stage_name.to_owned(),
                details: format_diagnostics_summary(&diags, &raw),
            }
        })?;

        let spv_options = spv::Options {
            lang_version: (1, 5),
            flags: spv::WriterFlags::ADJUST_COORDINATE_SPACE | spv::WriterFlags::CLAMP_FRAG_DEPTH,
            ..spv::Options::default()
        };

        spv::write_vec(&module, &module_info, &spv_options, None).map_err(|err| {
            VspError::SpirvCodegen {
                stage_name: stage_name.to_owned(),
                details: err.to_string(),
            }
        })
    }

    #[cfg(feature = "shaderc-backend")]
    fn compile_via_shaderc(
        stage_name: &str,
        stage_kind: ShaderStageKind,
        glsl_source: &str,
        defines: &[(String, Option<String>)],
        source_map: Option<&SourceLineMap>,
    ) -> Result<Vec<u32>, VspError> {
        thread_local! {
            static THREAD_COMPILER: std::cell::RefCell<Option<shaderc::Compiler>> =
                const { std::cell::RefCell::new(None) };
        }
        let mut options = shaderc::CompileOptions::new().map_err(|err| VspError::SpirvCodegen {
            stage_name: stage_name.to_owned(),
            details: format!("Failed to allocate shaderc options: {err}"),
        })?;
        options.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_3 as u32,
        );
        options.set_target_spirv(shaderc::SpirvVersion::V1_6);
        options.set_optimization_level(shaderc::OptimizationLevel::Performance);
        if std::env::var_os("VS_SHADER_DEBUG_INFO").is_some() {
            options.set_generate_debug_info();
        }
        for (name, value) in defines {
            options.add_macro_definition(name, value.as_deref());
        }

        let kind = match stage_kind {
            ShaderStageKind::Vertex => shaderc::ShaderKind::Vertex,
            ShaderStageKind::Fragment => shaderc::ShaderKind::Fragment,
            ShaderStageKind::Compute => shaderc::ShaderKind::Compute,
            ShaderStageKind::RayGen => shaderc::ShaderKind::RayGeneration,
            ShaderStageKind::ClosestHit => shaderc::ShaderKind::ClosestHit,
            ShaderStageKind::Miss => shaderc::ShaderKind::Miss,
            ShaderStageKind::AnyHit => shaderc::ShaderKind::AnyHit,
        };

        THREAD_COMPILER.with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot.is_none() {
                let compiler = shaderc::Compiler::new().map_err(|err| VspError::SpirvCodegen {
                    stage_name: stage_name.to_owned(),
                    details: format!("Failed to initialize shaderc compiler: {err}"),
                })?;
                *slot = Some(compiler);
            }
            let compiler = slot.as_ref().unwrap();
            let artifact = compiler
                .compile_into_spirv(glsl_source, kind, stage_name, "main", Some(&options))
                .map_err(|err| {
                    let raw = err.to_string();
                    let diags = parse_compile_diagnostics(&raw, stage_name, source_map);
                    VspError::GlslParse {
                        stage_name: stage_name.to_owned(),
                        details: format_diagnostics_summary(&diags, &raw),
                    }
                })?;
            Ok(artifact.as_binary().to_vec())
        })
    }
}
