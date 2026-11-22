//! 7-suite cross-verification engine for VulkanShaders.
//!
//! Verifies:
//! 1. Dynamic source symbol extraction & vendor binary export cross-checking (NGX, NRD, FSR3, XeSS)
//! 2. Official C-header ABI parity against VulkanShaders Rust definitions
//! 3. Compile-time & live `#[repr(C)]` struct layout sizes/alignments
//! 4. Rust FFI (`ffi/mod.rs`) <-> Kotlin FFM (`NativeCoreBridge.kt`) 1-to-1 ABI parity
//! 5. Capability bitmask (`CAP_*`) parity across `ffi/mod.rs`, `NativeCoreBridge.kt`, `VulkanDeviceContext.kt`
//! 6. GLSL shader `(set, binding)` uniqueness, compute/RT `texture()` ban, and `CRUMBLING_MASK` exclusion
//! 7. Mixin JSON file existence and `compat.*` `@Pseudo` + `require = 0` guards

use crate::binary_parser::ParsedBinary;
use crate::downloader::SdkKind;
use crate::header_parser::ParsedHeader;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::mem::{align_of, size_of};
use std::path::{Path, PathBuf};
use vulkanshaders_core::sdk::{fsr3, ngx, nrd, xess};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum CheckStatus {
    Pass,
    Warn,
    Fail,
}

impl std::fmt::Display for CheckStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::Warn => write!(f, "WARN"),
            Self::Fail => write!(f, "FAIL"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    pub name: String,
    pub status: CheckStatus,
    pub detail: String,
}

impl CheckResult {
    pub fn pass(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Pass,
            detail: detail.into(),
        }
    }

    pub fn warn(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Warn,
            detail: detail.into(),
        }
    }

    pub fn fail(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Fail,
            detail: detail.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteReport {
    pub suite: String,
    pub status: CheckStatus,
    pub checks: Vec<CheckResult>,
}

impl SuiteReport {
    pub fn from_checks(suite: impl Into<String>, checks: Vec<CheckResult>) -> Self {
        let status = if checks.iter().any(|c| c.status == CheckStatus::Fail) {
            CheckStatus::Fail
        } else if checks.iter().any(|c| c.status == CheckStatus::Warn) {
            CheckStatus::Warn
        } else {
            CheckStatus::Pass
        };
        Self {
            suite: suite.into(),
            status,
            checks,
        }
    }

    pub fn fail_count(&self) -> usize {
        self.checks
            .iter()
            .filter(|c| c.status == CheckStatus::Fail)
            .count()
    }

    pub fn warn_count(&self) -> usize {
        self.checks
            .iter()
            .filter(|c| c.status == CheckStatus::Warn)
            .count()
    }

    pub fn pass_count(&self) -> usize {
        self.checks
            .iter()
            .filter(|c| c.status == CheckStatus::Pass)
            .count()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub repo_root: String,
    pub overall_status: CheckStatus,
    pub total_pass: usize,
    pub total_warn: usize,
    pub total_fail: usize,
    pub suites: Vec<SuiteReport>,
}

impl BenchReport {
    pub fn from_suites(repo_root: &Path, suites: Vec<SuiteReport>) -> Self {
        let total_pass = suites.iter().map(SuiteReport::pass_count).sum();
        let total_warn = suites.iter().map(SuiteReport::warn_count).sum();
        let total_fail = suites.iter().map(SuiteReport::fail_count).sum();
        let overall_status = if total_fail > 0 {
            CheckStatus::Fail
        } else if total_warn > 0 {
            CheckStatus::Warn
        } else {
            CheckStatus::Pass
        };
        Self {
            repo_root: repo_root.display().to_string(),
            overall_status,
            total_pass,
            total_warn,
            total_fail,
            suites,
        }
    }

    pub fn has_failures(&self) -> bool {
        self.total_fail > 0
    }
}

#[derive(Debug, Clone)]
pub struct VerifyOptions<'a> {
    pub repo_root: PathBuf,
    pub binaries_by_sdk: BTreeMap<SdkKind, Vec<&'a ParsedBinary>>,
    pub headers_by_sdk: BTreeMap<SdkKind, Vec<(String, &'a ParsedHeader)>>,
    pub download_requested: bool,
    pub download_errors: BTreeMap<SdkKind, Vec<String>>,
}

impl<'a> VerifyOptions<'a> {
    pub fn offline(repo_root: PathBuf) -> Self {
        Self {
            repo_root,
            binaries_by_sdk: BTreeMap::new(),
            headers_by_sdk: BTreeMap::new(),
            download_requested: false,
            download_errors: BTreeMap::new(),
        }
    }
}

/// Runs all 7 verification suites.
pub fn run_all_suites(opts: &VerifyOptions<'_>) -> BenchReport {
    let suites = vec![
        verify_sdk_binary_symbols(opts),
        verify_c_header_abi_parity(opts),
        verify_repr_c_layouts(),
        verify_ffi_ffm_parity(&opts.repo_root),
        verify_capability_bitmasks(&opts.repo_root),
        verify_shader_contracts(&opts.repo_root),
        verify_mixins_and_compat(&opts.repo_root),
    ];
    BenchReport::from_suites(&opts.repo_root, suites)
}

/// Runs the 5 purely offline source-code verification suites (Suites 3..7).
pub fn run_offline_source_suites(repo_root: &Path) -> BenchReport {
    let suites = vec![
        verify_repr_c_layouts(),
        verify_ffi_ffm_parity(repo_root),
        verify_capability_bitmasks(repo_root),
        verify_shader_contracts(repo_root),
        verify_mixins_and_compat(repo_root),
    ];
    BenchReport::from_suites(repo_root, suites)
}

// ============================================================================
// Suite 1: Dynamic Source Extraction & Binary Export Cross-Check
// ============================================================================

struct CanonicalSdkSymbols {
    sdk: SdkKind,
    label: &'static str,
    source_rel_path: &'static str,
    required: &'static [&'static str],
    optional: &'static [&'static str],
    /// Alternative required groups (e.g., FSR3 Legacy ABI vs FFX API v2, or NGX EvaluateFeature_C).
    alt_required_groups: &'static [&'static [&'static str]],
}

const NGX_REQUIRED: &[&str] = &[
    "NVSDK_NGX_VULKAN_Init_Ext2",
    "NVSDK_NGX_VULKAN_Shutdown1",
    "NVSDK_NGX_VULKAN_GetCapabilityParameters",
    "NVSDK_NGX_VULKAN_AllocateParameters",
    "NVSDK_NGX_VULKAN_DestroyParameters",
    "NVSDK_NGX_VULKAN_GetScratchBufferSize",
    "NVSDK_NGX_VULKAN_CreateFeature1",
    "NVSDK_NGX_VULKAN_EvaluateFeature",
    "NVSDK_NGX_VULKAN_ReleaseFeature",
];

const NGX_OPTIONAL: &[&str] = &[
    "NVSDK_NGX_VULKAN_Init_Ext",
    "NVSDK_NGX_VULKAN_Init",
    "NVSDK_NGX_VULKAN_CreateFeature",
    "NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements",
    "NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements",
    "NVSDK_NGX_VULKAN_GetFeatureRequirements",
];

const NRD_REQUIRED: &[&str] = &[
    "CreateInstance",
    "DestroyInstance",
    "GetLibraryDesc",
    "GetInstanceDesc",
    "SetCommonSettings",
    "SetDenoiserSettings",
    "GetComputeDispatches",
];

const NRD_OPTIONAL: &[&str] = &["GetResourceTypeString", "GetDenoiserString"];

const FSR3_V2_REQUIRED: &[&str] = &["ffxCreateContext", "ffxDestroyContext", "ffxDispatch"];
const FSR3_V2_OPTIONAL: &[&str] = &["ffxConfigure", "ffxQuery"];
const FSR3_LEGACY_REQUIRED_A: &[&str] = &[
    "ffxFsr3ContextCreate",
    "ffxFsr3ContextDispatch",
    "ffxFsr3ContextDestroy",
];
const FSR3_LEGACY_REQUIRED_B: &[&str] = &[
    "ffxFsr3UpscalerContextCreate",
    "ffxFsr3UpscalerContextDispatch",
    "ffxFsr3UpscalerContextDestroy",
];

const XESS_REQUIRED: &[&str] = &[
    "xessGetVersion",
    "xessVKCreateContext",
    "xessVKInit",
    "xessVKExecute",
    "xessDestroyContext",
];

const XESS_OPTIONAL: &[&str] = &[
    "xessVKGetRequiredInstanceExtensions",
    "xessVKGetRequiredDeviceExtensions",
    "xessVKGetRequiredDeviceFeatures",
    "xessVKBuildPipelines",
    "xessGetInputResolution",
    "xessGetOptimalInputResolution",
    "xessGetProperties",
    "xessSetJitterScale",
    "xessSetVelocityScale",
];

fn canonical_sdks() -> Vec<CanonicalSdkSymbols> {
    vec![
        CanonicalSdkSymbols {
            sdk: SdkKind::Ngx,
            label: "NGX",
            source_rel_path: "crates/vulkanshaders-core/src/sdk/ngx.rs",
            required: NGX_REQUIRED,
            optional: NGX_OPTIONAL,
            alt_required_groups: &[],
        },
        CanonicalSdkSymbols {
            sdk: SdkKind::Nrd,
            label: "NRD",
            source_rel_path: "crates/vulkanshaders-core/src/sdk/nrd.rs",
            required: NRD_REQUIRED,
            optional: NRD_OPTIONAL,
            alt_required_groups: &[],
        },
        CanonicalSdkSymbols {
            sdk: SdkKind::Fsr3,
            label: "FSR3",
            source_rel_path: "crates/vulkanshaders-core/src/sdk/fsr3.rs",
            required: FSR3_V2_REQUIRED,
            optional: FSR3_V2_OPTIONAL,
            alt_required_groups: &[FSR3_LEGACY_REQUIRED_A, FSR3_LEGACY_REQUIRED_B],
        },
        CanonicalSdkSymbols {
            sdk: SdkKind::Xess,
            label: "XeSS",
            source_rel_path: "crates/vulkanshaders-core/src/sdk/xess.rs",
            required: XESS_REQUIRED,
            optional: XESS_OPTIONAL,
            alt_required_groups: &[],
        },
    ]
}

/// Dynamically extracts every C symbol name passed to `lib.get(...)`, `load_optional(...)`,
/// `b"...\0"`, or `c"..."` from a Rust SDK loader source file.
pub fn extract_symbols_from_rust_sdk_source(source: &str) -> BTreeSet<String> {
    let cleaned = strip_c_style_comments(source);
    let mut symbols = BTreeSet::new();
    let bytes = cleaned.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        // Match byte-string literals: b"...\0"
        if i + 2 < bytes.len()
            && bytes[i] == b'b'
            && bytes[i + 1] == b'"'
            && (i == 0 || !is_ident_char(bytes[i - 1]))
        {
            if let Some((lit, end)) = parse_quoted_literal(&cleaned, i + 1) {
                if let Some(sym) = lit.strip_suffix("\\0") {
                    if is_c_symbol_ident(sym) {
                        symbols.insert(sym.to_string());
                    }
                }
                i = end;
                continue;
            }
        }
        // Match C-string literals: c"..."
        if i + 2 < bytes.len()
            && bytes[i] == b'c'
            && bytes[i + 1] == b'"'
            && (i == 0 || !is_ident_char(bytes[i - 1]))
        {
            if let Some((lit, end)) = parse_quoted_literal(&cleaned, i + 1) {
                if is_c_symbol_ident(&lit) && lit != "main" {
                    symbols.insert(lit);
                }
                i = end;
                continue;
            }
        }
        i += 1;
    }

    // Also capture string literals directly inside `load_optional(` or `lib.get(` calls.
    for needle in ["lib.get", "load_optional("] {
        let mut search_from = 0;
        while let Some(pos) = cleaned[search_from..].find(needle) {
            let abs = search_from + pos + needle.len();
            let window_end = (abs + 200).min(cleaned.len());
            let window = &cleaned[abs..window_end];
            if let Some(q) = window.find('"') {
                if let Some((lit, _)) = parse_quoted_literal(window, q) {
                    let trimmed = lit.strip_suffix("\\0").unwrap_or(&lit);
                    if is_c_symbol_ident(trimmed) {
                        symbols.insert(trimmed.to_string());
                    }
                }
            }
            search_from = abs;
        }
    }

    symbols
}

pub fn verify_sdk_binary_symbols(opts: &VerifyOptions<'_>) -> SuiteReport {
    let mut checks = Vec::new();

    for spec in canonical_sdks() {
        let src_path = opts.repo_root.join(spec.source_rel_path);
        let source = match fs::read_to_string(&src_path) {
            Ok(s) => s,
            Err(e) => {
                checks.push(CheckResult::fail(
                    format!("{} source extraction", spec.label),
                    format!("failed to read {}: {e}", src_path.display()),
                ));
                continue;
            }
        };

        let extracted = extract_symbols_from_rust_sdk_source(&source);
        let mut missing_in_source = Vec::new();
        for &sym in spec.required.iter().chain(spec.optional.iter()) {
            if !extracted.contains(sym) {
                missing_in_source.push(sym);
            }
        }
        if missing_in_source.is_empty() {
            checks.push(CheckResult::pass(
                format!("{} source symbols", spec.label),
                format!(
                    "extracted {} C symbols from {}; all {} required + {} optional canonical symbols present",
                    extracted.len(),
                    spec.source_rel_path,
                    spec.required.len(),
                    spec.optional.len()
                ),
            ));
        } else {
            checks.push(CheckResult::fail(
                format!("{} source symbols", spec.label),
                format!(
                    "canonical symbols missing from {}: {}",
                    spec.source_rel_path,
                    missing_in_source.join(", ")
                ),
            ));
        }

        // Cross-check against parsed binaries for this SDK.
        let binaries = opts
            .binaries_by_sdk
            .get(&spec.sdk)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let dl_errors = opts
            .download_errors
            .get(&spec.sdk)
            .map(Vec::as_slice)
            .unwrap_or(&[]);

        if binaries.is_empty() {
            if opts.download_requested {
                let err_msg = if dl_errors.is_empty() {
                    "download requested but no binary artifacts were retrieved".to_string()
                } else {
                    format!("download failed: {}", dl_errors.join("; "))
                };
                checks.push(CheckResult::fail(
                    format!("{} binary exports", spec.label),
                    err_msg,
                ));
            } else {
                checks.push(CheckResult::warn(
                    format!("{} binary exports", spec.label),
                    "no local/downloaded binary available in offline mode".to_string(),
                ));
            }
            continue;
        }

        // Union exported symbols across all binaries for this SDK.
        let mut all_exports: BTreeSet<String> = BTreeSet::new();
        for bin in binaries {
            for sym in &bin.exported_symbols {
                all_exports.insert(sym.clone());
            }
        }

        let has_symbol = |sym: &str| -> bool {
            if all_exports.contains(sym) {
                return true;
            }
            // NGX EvaluateFeature allows _C fallback.
            if spec.sdk == SdkKind::Ngx && sym == "NVSDK_NGX_VULKAN_EvaluateFeature" {
                return all_exports.contains("NVSDK_NGX_VULKAN_EvaluateFeature_C");
            }
            false
        };

        let missing_req: Vec<&str> = spec
            .required
            .iter()
            .copied()
            .filter(|s| !has_symbol(s))
            .collect();
        let alt_satisfied = !missing_req.is_empty()
            && spec
                .alt_required_groups
                .iter()
                .any(|grp| grp.iter().all(|s| has_symbol(s)));

        let missing_opt: Vec<&str> = spec
            .optional
            .iter()
            .copied()
            .filter(|s| !has_symbol(s))
            .collect();

        if !missing_req.is_empty() && !alt_satisfied {
            checks.push(CheckResult::fail(
                format!("{} binary exports", spec.label),
                format!(
                    "checked {} binary(ies) ({} exports): missing required symbol(s): {}",
                    binaries.len(),
                    all_exports.len(),
                    missing_req.join(", ")
                ),
            ));
        } else if !missing_opt.is_empty() {
            checks.push(CheckResult::warn(
                format!("{} binary exports", spec.label),
                format!(
                    "checked {} binary(ies) ({} exports): all required symbols exported; optional symbol(s) absent: {}",
                    binaries.len(),
                    all_exports.len(),
                    missing_opt.join(", ")
                ),
            ));
        } else {
            checks.push(CheckResult::pass(
                format!("{} binary exports", spec.label),
                format!(
                    "checked {} binary(ies) ({} exports): all required + optional symbols verified",
                    binaries.len(),
                    all_exports.len()
                ),
            ));
        }
    }

    SuiteReport::from_checks("1. SDK Source & Binary Symbol Verification", checks)
}

// ============================================================================
// Suite 2: Official C-Header ABI Parity Verification
// ============================================================================

/// Extracts all `pub const NAME: &CStr = c"value";` parameter strings from `sdk/ngx.rs`.
pub fn extract_ngx_param_constants(ngx_source: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let Some(start) = ngx_source.find("pub mod params {") else {
        return out;
    };
    let rest = &ngx_source[start..];
    let Some(end) = rest.find("\n}") else {
        return out;
    };
    let block = &rest[..end];
    let cleaned = strip_c_style_comments(block);

    let mut idx = 0;
    while let Some(pos) = cleaned[idx..].find("pub const ") {
        let abs = idx + pos + "pub const ".len();
        let slice = &cleaned[abs..];
        let Some(colon) = slice.find(':') else {
            break;
        };
        let name = slice[..colon].trim().to_string();
        if let Some(semi) = slice.find(';') {
            let stmt = &slice[..semi];
            if let Some(c_pos) = stmt.find("c\"") {
                if let Some((val, _)) = parse_quoted_literal(stmt, c_pos + 1) {
                    out.push((name, val));
                }
            }
            idx = abs + semi + 1;
        } else {
            break;
        }
    }
    out
}

fn parse_c_int_expr(raw: &str) -> Option<i64> {
    let s = raw
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .trim()
        .trim_end_matches(|c: char| matches!(c, 'u' | 'U' | 'l' | 'L'));
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok().map(|v| v as i64)
    } else if let Some((lhs, rhs)) = s.split_once("<<") {
        let a = parse_c_int_expr(lhs)?;
        let b = parse_c_int_expr(rhs)?;
        Some(a << b)
    } else if let Some((lhs, rhs)) = s.split_once('|') {
        let a = parse_c_int_expr(lhs)?;
        let b = parse_c_int_expr(rhs)?;
        Some(a | b)
    } else {
        s.parse::<i64>().ok()
    }
}

fn header_lookup_int(headers: &[(String, &ParsedHeader)], key: &str) -> Option<i64> {
    for (_, hdr) in headers {
        if let Some(&val) = hdr.defines_int.get(key) {
            return i64::try_from(val).ok();
        }
        if let Some(&val) = hdr.enum_values.get(key) {
            return i64::try_from(val).ok();
        }
        let suffix = format!("::{key}");
        for (ek, &ev) in &hdr.enum_values {
            if ek.ends_with(&suffix) {
                return i64::try_from(ev).ok();
            }
        }
    }
    None
}

fn header_has_string_define_value(headers: &[(String, &ParsedHeader)], expected: &str) -> bool {
    let quoted = format!("\"{expected}\"");
    for (_, hdr) in headers {
        for val in hdr.defines_str.values() {
            let trimmed = val.trim();
            if trimmed == expected || trimmed == quoted {
                return true;
            }
        }
    }
    false
}

fn header_has_function(headers: &[(String, &ParsedHeader)], func_name: &str) -> bool {
    headers
        .iter()
        .any(|(_, hdr)| hdr.functions.contains_key(func_name))
}

fn header_has_enum(headers: &[(String, &ParsedHeader)], enum_substr: &str) -> bool {
    headers
        .iter()
        .any(|(_, hdr)| hdr.enum_values.keys().any(|k| k.contains(enum_substr)))
}

pub fn verify_c_header_abi_parity(opts: &VerifyOptions<'_>) -> SuiteReport {
    let mut checks = Vec::new();

    // Always verify NGX param constant count in local Rust source first.
    let ngx_src_path = opts
        .repo_root
        .join("crates/vulkanshaders-core/src/sdk/ngx.rs");
    let ngx_params = fs::read_to_string(&ngx_src_path)
        .map(|s| extract_ngx_param_constants(&s))
        .unwrap_or_default();

    // --- NGX ---
    let ngx_hdrs = opts
        .headers_by_sdk
        .get(&SdkKind::Ngx)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    if ngx_hdrs.is_empty() {
        checks.push(missing_header_check("NGX", opts.download_requested));
    } else {
        let mut mismatches = Vec::new();
        let expected_ints: &[(&str, i64)] = &[
            ("NVSDK_NGX_VERSION_API", 0x15),
            ("NVSDK_NGX_Feature_SuperSampling", 1),
            ("NVSDK_NGX_Feature_FrameGeneration", 11),
            ("NVSDK_NGX_Feature_RayReconstruction", 13),
            ("NVSDK_NGX_PerfQuality_Value_MaxPerf", 0),
            ("NVSDK_NGX_PerfQuality_Value_Balanced", 1),
            ("NVSDK_NGX_PerfQuality_Value_MaxQuality", 2),
            ("NVSDK_NGX_PerfQuality_Value_UltraPerformance", 3),
            ("NVSDK_NGX_PerfQuality_Value_UltraQuality", 4),
            ("NVSDK_NGX_PerfQuality_Value_DLAA", 5),
            ("NVSDK_NGX_Result_Success", 1),
            ("NVSDK_NGX_Result_Fail", 0xBAD0_0000_u32 as i64),
        ];
        for &(k, want) in expected_ints {
            if let Some(got) = header_lookup_int(ngx_hdrs, k) {
                if (got as u32) != (want as u32) {
                    mismatches.push(format!("{k}: header={got:#x} != rust={want:#x}"));
                }
            }
        }
        let mut missing_params = Vec::new();
        for (const_name, param_str) in &ngx_params {
            if !header_has_string_define_value(ngx_hdrs, param_str) {
                missing_params.push(format!("{const_name}(\"{param_str}\")"));
            }
        }
        let mut missing_fns = Vec::new();
        for &f in NGX_REQUIRED {
            if !header_has_function(ngx_hdrs, f) {
                missing_fns.push(f);
            }
        }
        if !mismatches.is_empty() || !missing_fns.is_empty() {
            checks.push(CheckResult::fail(
                "NGX C-header ABI parity",
                format!(
                    "mismatches=[{}], missing_prototypes=[{}]",
                    mismatches.join("; "),
                    missing_fns.join(", ")
                ),
            ));
        } else if !missing_params.is_empty() {
            checks.push(CheckResult::warn(
                "NGX C-header ABI parity",
                format!(
                    "API constants & 9 prototypes matched; {}/{} param strings not found in staged headers: {}",
                    missing_params.len(),
                    ngx_params.len(),
                    missing_params.iter().take(6).cloned().collect::<Vec<_>>().join(", ")
                ),
            ));
        } else {
            checks.push(CheckResult::pass(
                "NGX C-header ABI parity",
                format!(
                    "verified API constants, {} NGX parameter strings, and 9 NVSDK_NGX_VULKAN_* prototypes across {} header(s)",
                    ngx_params.len(),
                    ngx_hdrs.len()
                ),
            ));
        }
    }

    // --- NRD ---
    let nrd_hdrs = opts
        .headers_by_sdk
        .get(&SdkKind::Nrd)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    if nrd_hdrs.is_empty() {
        checks.push(missing_header_check("NRD", opts.download_requested));
    } else {
        let mut mismatches = Vec::new();
        for expected_enum in [
            "Result",
            "Denoiser",
            "ResourceType",
            "NormalEncoding",
            "RoughnessEncoding",
            "AccumulationMode",
        ] {
            if !header_has_enum(nrd_hdrs, expected_enum) {
                mismatches.push(format!("missing enum {expected_enum}"));
            }
        }
        if let Some(v) = header_lookup_int(nrd_hdrs, "IN_MV") {
            if v != 0 {
                mismatches.push(format!("ResourceType::IN_MV={v} != 0"));
            }
        }
        if let Some(v) = header_lookup_int(nrd_hdrs, "PERMANENT_POOL") {
            if v != 31 {
                mismatches.push(format!("ResourceType::PERMANENT_POOL={v} != 31"));
            }
        }
        for &f in NRD_REQUIRED {
            if !header_has_function(nrd_hdrs, f) {
                mismatches.push(format!("missing prototype {f}"));
            }
        }
        if mismatches.is_empty() {
            checks.push(CheckResult::pass(
                "NRD C-header ABI parity",
                format!(
                    "verified NRD enums and 7 C API prototypes across {} header(s)",
                    nrd_hdrs.len()
                ),
            ));
        } else {
            checks.push(CheckResult::fail(
                "NRD C-header ABI parity",
                mismatches.join("; "),
            ));
        }
    }

    // --- FSR3 ---
    let fsr3_hdrs = opts
        .headers_by_sdk
        .get(&SdkKind::Fsr3)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    if fsr3_hdrs.is_empty() {
        checks.push(missing_header_check("FSR3", opts.download_requested));
    } else {
        let mut mismatches = Vec::new();
        let expected_ints: &[(&str, i64)] = &[
            ("FFX_API_RETURN_OK", 0),
            ("FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK", 0x0100_0003),
            ("FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE", 0x0001_0000),
            ("FFX_API_DISPATCH_DESC_TYPE_UPSCALE", 0x0001_0001),
            (
                "FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION",
                0x0002_0001,
            ),
            ("FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION", 0x0002_0003),
        ];
        for &(k, want) in expected_ints {
            if let Some(got) = header_lookup_int(fsr3_hdrs, k) {
                if got != want {
                    mismatches.push(format!("{k}: header={got:#x} != rust={want:#x}"));
                }
            }
        }
        for expected_enum in ["FfxApiSurfaceFormat", "FfxApiUpscaleQualityMode"] {
            if !header_has_enum(fsr3_hdrs, expected_enum) {
                mismatches.push(format!("missing enum {expected_enum}"));
            }
        }
        for &f in FSR3_V2_REQUIRED.iter().chain(FSR3_V2_OPTIONAL.iter()) {
            if !header_has_function(fsr3_hdrs, f) {
                mismatches.push(format!("missing prototype {f}"));
            }
        }
        if mismatches.is_empty() {
            checks.push(CheckResult::pass(
                "FSR3 C-header ABI parity",
                format!(
                    "verified FFX API desc types, enums, and 5 ffx* prototypes across {} header(s)",
                    fsr3_hdrs.len()
                ),
            ));
        } else {
            checks.push(CheckResult::fail(
                "FSR3 C-header ABI parity",
                mismatches.join("; "),
            ));
        }
    }

    // --- XeSS ---
    let xess_hdrs = opts
        .headers_by_sdk
        .get(&SdkKind::Xess)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    if xess_hdrs.is_empty() {
        checks.push(missing_header_check("XeSS", opts.download_requested));
    } else {
        let mut mismatches = Vec::new();
        let expected_ints: &[(&str, i64)] = &[
            ("XESS_RESULT_SUCCESS", 0),
            ("XESS_QUALITY_SETTING_ULTRA_PERFORMANCE", 100),
            ("XESS_QUALITY_SETTING_PERFORMANCE", 101),
            ("XESS_QUALITY_SETTING_BALANCED", 102),
            ("XESS_QUALITY_SETTING_QUALITY", 103),
            ("XESS_QUALITY_SETTING_ULTRA_QUALITY", 104),
            ("XESS_QUALITY_SETTING_ULTRA_QUALITY_PLUS", 105),
            ("XESS_QUALITY_SETTING_AA", 106),
        ];
        for &(k, want) in expected_ints {
            if let Some(got) = header_lookup_int(xess_hdrs, k) {
                if got != want {
                    mismatches.push(format!("{k}: header={got} != rust={want}"));
                }
            }
        }
        for &f in XESS_REQUIRED {
            if !header_has_function(xess_hdrs, f) {
                mismatches.push(format!("missing prototype {f}"));
            }
        }
        if mismatches.is_empty() {
            checks.push(CheckResult::pass(
                "XeSS C-header ABI parity",
                format!(
                    "verified XESS_RESULT_SUCCESS, XESS_QUALITY_SETTING_*, and 5 xess* prototypes across {} header(s)",
                    xess_hdrs.len()
                ),
            ));
        } else {
            checks.push(CheckResult::fail(
                "XeSS C-header ABI parity",
                mismatches.join("; "),
            ));
        }
    }

    SuiteReport::from_checks("2. Official C-Header ABI Parity", checks)
}

fn missing_header_check(label: &str, download_requested: bool) -> CheckResult {
    let name = format!("{label} C-header ABI parity");
    if download_requested {
        CheckResult::fail(
            name,
            "download requested but official C headers were not retrieved",
        )
    } else {
        CheckResult::warn(name, "no official C headers staged locally in offline mode")
    }
}

// ============================================================================
// Suite 3: Compile-Time & Live #[repr(C)] Struct Layout Verification
// ============================================================================

pub fn verify_repr_c_layouts() -> SuiteReport {
    let mut checks = Vec::new();

    let layouts: &[(&str, usize, usize, usize, usize)] = &[
        (
            "ngx::NvSdkNgxImageViewInfoVk",
            size_of::<ngx::NvSdkNgxImageViewInfoVk>(),
            48,
            align_of::<ngx::NvSdkNgxImageViewInfoVk>(),
            8,
        ),
        (
            "ngx::NvSdkNgxBufferInfoVk",
            size_of::<ngx::NvSdkNgxBufferInfoVk>(),
            16,
            align_of::<ngx::NvSdkNgxBufferInfoVk>(),
            8,
        ),
        (
            "ngx::NvSdkNgxResourceVk",
            size_of::<ngx::NvSdkNgxResourceVk>(),
            56,
            align_of::<ngx::NvSdkNgxResourceVk>(),
            8,
        ),
        (
            "nrd::NrdCommonSettings",
            size_of::<nrd::NrdCommonSettings>(),
            452,
            align_of::<nrd::NrdCommonSettings>(),
            4,
        ),
        (
            "nrd::NrdReblurSettings",
            size_of::<nrd::NrdReblurSettings>(),
            120,
            align_of::<nrd::NrdReblurSettings>(),
            4,
        ),
        (
            "nrd::NrdInstanceDesc",
            size_of::<nrd::NrdInstanceDesc>(),
            112,
            align_of::<nrd::NrdInstanceDesc>(),
            8,
        ),
        (
            "fsr3::FfxApiHeader",
            size_of::<fsr3::FfxApiHeader>(),
            16,
            align_of::<fsr3::FfxApiHeader>(),
            8,
        ),
        (
            "fsr3::FfxApiResourceDescription",
            size_of::<fsr3::FfxApiResourceDescription>(),
            32,
            align_of::<fsr3::FfxApiResourceDescription>(),
            4,
        ),
        (
            "fsr3::FfxApiResource",
            size_of::<fsr3::FfxApiResource>(),
            48,
            align_of::<fsr3::FfxApiResource>(),
            8,
        ),
        (
            "fsr3::FfxDispatchDescUpscale",
            size_of::<fsr3::FfxDispatchDescUpscale>(),
            432,
            align_of::<fsr3::FfxDispatchDescUpscale>(),
            8,
        ),
        (
            "xess::XessVersion",
            size_of::<xess::XessVersion>(),
            8,
            align_of::<xess::XessVersion>(),
            2,
        ),
        (
            "xess::Xess2d",
            size_of::<xess::Xess2d>(),
            8,
            align_of::<xess::Xess2d>(),
            4,
        ),
        (
            "xess::XessVkImageViewInfo",
            size_of::<xess::XessVkImageViewInfo>(),
            48,
            align_of::<xess::XessVkImageViewInfo>(),
            8,
        ),
        (
            "xess::XessVkInitParams",
            size_of::<xess::XessVkInitParams>(),
            64,
            align_of::<xess::XessVkInitParams>(),
            8,
        ),
        (
            "xess::XessVkExecuteParams",
            size_of::<xess::XessVkExecuteParams>(),
            360,
            align_of::<xess::XessVkExecuteParams>(),
            8,
        ),
    ];

    for &(name, actual_size, expected_size, actual_align, expected_align) in layouts {
        if actual_size == expected_size && actual_align == expected_align {
            checks.push(CheckResult::pass(
                name,
                format!("size={actual_size}B, align={actual_align}B (matches C ABI)"),
            ));
        } else {
            checks.push(CheckResult::fail(
                name,
                format!(
                    "expected size={expected_size}B align={expected_align}B, got size={actual_size}B align={actual_align}B"
                ),
            ));
        }
    }

    // Verify core compile-time API constants from vulkanshaders-core.
    let const_checks: &[(&str, bool, String)] = &[
        (
            "ngx::NVSDK_NGX_VERSION_API == 0x15",
            ngx::NVSDK_NGX_VERSION_API == 0x15,
            format!("actual={:#x}", ngx::NVSDK_NGX_VERSION_API),
        ),
        (
            "fsr3::FFX_API_RETURN_OK == 0",
            fsr3::FFX_API_RETURN_OK == 0,
            format!("actual={}", fsr3::FFX_API_RETURN_OK),
        ),
        (
            "fsr3::FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK == 0x01000003",
            fsr3::FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK == 0x0100_0003,
            format!(
                "actual={:#x}",
                fsr3::FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK
            ),
        ),
        (
            "fsr3::FFX_API_DISPATCH_DESC_TYPE_UPSCALE == 0x00010001",
            fsr3::FFX_API_DISPATCH_DESC_TYPE_UPSCALE == 0x0001_0001,
            format!("actual={:#x}", fsr3::FFX_API_DISPATCH_DESC_TYPE_UPSCALE),
        ),
        (
            "fsr3::FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION == 0x00020003",
            fsr3::FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION == 0x0002_0003,
            format!(
                "actual={:#x}",
                fsr3::FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION
            ),
        ),
        (
            "xess::XessResult::SUCCESS == 0",
            xess::XessResult::SUCCESS.0 == 0,
            format!("actual={}", xess::XessResult::SUCCESS.0),
        ),
    ];

    for (label, ok, detail) in const_checks {
        if *ok {
            checks.push(CheckResult::pass(*label, detail.clone()));
        } else {
            checks.push(CheckResult::fail(*label, detail.clone()));
        }
    }

    SuiteReport::from_checks("3. #[repr(C)] Struct Layout & ABI Constants", checks)
}

// ============================================================================
// Suite 4: Rust FFI (ffi/mod.rs) <-> Kotlin FFM (NativeCoreBridge.kt) Parity
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
enum AbiType {
    Void,
    Int32,
    Int64,
    Float32,
    Float64,
    Byte,
    Pointer,
    Unknown(String),
}

impl AbiType {
    fn from_rust(raw: &str) -> Self {
        let t = raw.trim();
        if t.is_empty() || t == "()" {
            Self::Void
        } else if matches!(t, "i32" | "u32" | "c_int" | "c_uint") {
            Self::Int32
        } else if matches!(t, "i64" | "u64" | "usize" | "isize") {
            Self::Int64
        } else if t == "f32" {
            Self::Float32
        } else if t == "f64" {
            Self::Float64
        } else if matches!(t, "u8" | "i8" | "bool") {
            Self::Byte
        } else if t.starts_with("*const") || t.starts_with("*mut") {
            Self::Pointer
        } else {
            Self::Unknown(t.to_string())
        }
    }

    fn from_kotlin_layout(raw: &str) -> Self {
        let t = raw.trim();
        match t {
            "void" => Self::Void,
            "ValueLayout.JAVA_INT" => Self::Int32,
            "ValueLayout.JAVA_LONG" => Self::Int64,
            "ValueLayout.JAVA_FLOAT" => Self::Float32,
            "ValueLayout.JAVA_DOUBLE" => Self::Float64,
            "ValueLayout.JAVA_BYTE" | "ValueLayout.JAVA_BOOLEAN" => Self::Byte,
            "ValueLayout.ADDRESS" => Self::Pointer,
            other => Self::Unknown(other.to_string()),
        }
    }

    fn compatible_with_kotlin(&self, kt: &Self) -> bool {
        match (self, kt) {
            (Self::Pointer, Self::Pointer | Self::Int64) => true,
            (Self::Int64, Self::Pointer | Self::Int64) => true,
            (a, b) => a == b,
        }
    }
}

#[derive(Debug, Clone)]
struct FnSig {
    ret: AbiType,
    params: Vec<AbiType>,
}

fn parse_rust_ffi_exports(source: &str) -> BTreeMap<String, FnSig> {
    let cleaned = strip_c_style_comments(source);
    let mut map = BTreeMap::new();
    let needle = "extern \"C\" fn vs_core_";
    let mut search_from = 0;

    while let Some(pos) = cleaned[search_from..].find(needle) {
        let name_start = search_from + pos + "extern \"C\" fn ".len();
        let rest = &cleaned[name_start..];
        let Some(open_paren) = rest.find('(') else {
            break;
        };
        let fn_name = rest[..open_paren].trim().to_string();
        let Some(close_paren_rel) = find_matching_paren(rest, open_paren) else {
            break;
        };
        let params_str = &rest[open_paren + 1..close_paren_rel];
        let after_paren = &rest[close_paren_rel + 1..];
        let Some(brace_rel) = after_paren.find('{') else {
            break;
        };
        let ret_header = after_paren[..brace_rel].trim();
        let ret_ty = if let Some(arrow) = ret_header.find("->") {
            AbiType::from_rust(&ret_header[arrow + 2..])
        } else {
            AbiType::Void
        };

        let mut params = Vec::new();
        for part in params_str.split(',') {
            let p = part.trim();
            if p.is_empty() {
                continue;
            }
            if let Some((_, ty)) = p.split_once(':') {
                params.push(AbiType::from_rust(ty));
            }
        }
        map.insert(
            fn_name,
            FnSig {
                ret: ret_ty,
                params,
            },
        );
        search_from = name_start + close_paren_rel + 1;
    }

    map
}

fn parse_kotlin_ffm_downcalls(source: &str) -> BTreeMap<String, FnSig> {
    let cleaned = strip_c_style_comments(source);
    let mut map = BTreeMap::new();
    let needle = "downcall(";
    let mut search_from = 0;

    while let Some(pos) = cleaned[search_from..].find(needle) {
        let open_paren = search_from + pos + "downcall".len();
        let Some(close_paren) = find_matching_paren(&cleaned, open_paren) else {
            break;
        };
        let call_body = &cleaned[open_paren + 1..close_paren];
        search_from = close_paren + 1;

        let Some(name_q) = call_body.find("\"vs_core_") else {
            continue;
        };
        let Some((fn_name, _)) = parse_quoted_literal(call_body, name_q) else {
            continue;
        };

        let (is_void, fd_pos) = if let Some(p) = call_body.find("FunctionDescriptor.ofVoid(") {
            (true, p + "FunctionDescriptor.ofVoid".len())
        } else if let Some(p) = call_body.find("FunctionDescriptor.of(") {
            (false, p + "FunctionDescriptor.of".len())
        } else {
            continue;
        };

        let Some(fd_close) = find_matching_paren(call_body, fd_pos) else {
            continue;
        };
        let args_str = &call_body[fd_pos + 1..fd_close];
        let layouts: Vec<AbiType> = args_str
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(AbiType::from_kotlin_layout)
            .collect();

        let sig = if is_void {
            FnSig {
                ret: AbiType::Void,
                params: layouts,
            }
        } else if let Some((first, rest)) = layouts.split_first() {
            FnSig {
                ret: first.clone(),
                params: rest.to_vec(),
            }
        } else {
            FnSig {
                ret: AbiType::Unknown("empty FunctionDescriptor.of()".to_string()),
                params: Vec::new(),
            }
        };
        map.insert(fn_name, sig);
    }

    map
}

pub fn verify_ffi_ffm_parity(repo_root: &Path) -> SuiteReport {
    let rust_path = repo_root.join("crates/vulkanshaders-core/src/ffi/mod.rs");
    let kt_path = repo_root.join("src/main/kotlin/dev/vulkanshaders/backend/NativeCoreBridge.kt");

    let rust_src = match fs::read_to_string(&rust_path) {
        Ok(s) => s,
        Err(e) => {
            return SuiteReport::from_checks(
                "4. Rust FFI <-> Kotlin FFM ABI Parity",
                vec![CheckResult::fail(
                    "read ffi/mod.rs",
                    format!("{}: {e}", rust_path.display()),
                )],
            );
        }
    };
    let kt_src = match fs::read_to_string(&kt_path) {
        Ok(s) => s,
        Err(e) => {
            return SuiteReport::from_checks(
                "4. Rust FFI <-> Kotlin FFM ABI Parity",
                vec![CheckResult::fail(
                    "read NativeCoreBridge.kt",
                    format!("{}: {e}", kt_path.display()),
                )],
            );
        }
    };

    let rust_fns = parse_rust_ffi_exports(&rust_src);
    let kt_fns = parse_kotlin_ffm_downcalls(&kt_src);

    let mut checks = Vec::new();
    let all_names: BTreeSet<String> = rust_fns.keys().chain(kt_fns.keys()).cloned().collect();

    if all_names.is_empty() {
        checks.push(CheckResult::fail(
            "FFI symbol discovery",
            "parsed 0 vs_core_* functions",
        ));
        return SuiteReport::from_checks("4. Rust FFI <-> Kotlin FFM ABI Parity", checks);
    }

    let mut mismatches = Vec::new();
    for name in &all_names {
        match (rust_fns.get(name), kt_fns.get(name)) {
            (Some(_), None) => {
                mismatches.push(format!("{name}: present in Rust, missing in Kotlin"))
            }
            (None, Some(_)) => {
                mismatches.push(format!("{name}: present in Kotlin, missing in Rust"))
            }
            (Some(r), Some(k)) => {
                if !r.ret.compatible_with_kotlin(&k.ret) {
                    mismatches.push(format!(
                        "{name}: return mismatch (rust={:?}, kotlin={:?})",
                        r.ret, k.ret
                    ));
                } else if r.params.len() != k.params.len() {
                    mismatches.push(format!(
                        "{name}: arity mismatch (rust={}, kotlin={})",
                        r.params.len(),
                        k.params.len()
                    ));
                } else {
                    for (idx, (rp, kp)) in r.params.iter().zip(k.params.iter()).enumerate() {
                        if !rp.compatible_with_kotlin(kp) {
                            mismatches.push(format!(
                                "{name}: param[{idx}] mismatch (rust={rp:?}, kotlin={kp:?})"
                            ));
                        }
                    }
                }
            }
            (None, None) => {}
        }
    }

    if mismatches.is_empty() {
        checks.push(CheckResult::pass(
            "vs_core_* 1-to-1 ABI parity",
            format!(
                "all {} Rust extern \"C\" exports match Kotlin FFM downcall descriptors (names, return types, arities, and ValueLayouts)",
                all_names.len()
            ),
        ));
    } else {
        for m in mismatches {
            checks.push(CheckResult::fail("vs_core_* ABI mismatch", m));
        }
    }

    SuiteReport::from_checks("4. Rust FFI <-> Kotlin FFM ABI Parity", checks)
}

// ============================================================================
// Suite 5: Capability Bitmask Parity (Rust <-> NativeCoreBridge <-> VulkanDeviceContext)
// ============================================================================

fn extract_rust_cap_bits(ffi_src: &str, fn_name: &str) -> BTreeMap<u32, String> {
    let mut map = BTreeMap::new();
    let Some(start) = ffi_src.find(&format!("fn {fn_name}(")) else {
        return map;
    };
    let rest = &ffi_src[start..];
    let Some(open_brace) = rest.find('{') else {
        return map;
    };
    let Some(close_brace) = find_matching_brace(rest, open_brace) else {
        return map;
    };
    let body = &rest[open_brace + 1..close_brace];
    let lines: Vec<&str> = body.lines().collect();

    for (idx, line) in lines.iter().enumerate() {
        if let Some(pos) = line.find("mask |= 1 << ") {
            let after = &line[pos + "mask |= 1 << ".len()..];
            let bit_str: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(bit) = bit_str.parse::<u32>() {
                // Find the preceding `if ` condition line.
                let cond = (0..=idx)
                    .rev()
                    .find_map(|j| {
                        let l = lines[j].trim();
                        if l.starts_with("if ") {
                            Some(l.to_string())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_default();
                map.insert(bit, cond);
            }
        }
    }
    map
}

fn extract_kotlin_cap_consts(kt_src: &str) -> BTreeMap<String, u32> {
    let mut map = BTreeMap::new();
    for line in kt_src.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("const val CAP_") {
            if let Some((name_part, rhs)) = rest.split_once('=') {
                let const_name =
                    format!("CAP_{}", name_part.split(':').next().unwrap_or("").trim());
                if let Some((_, shift_str)) = rhs.split_once("shl") {
                    let digits: String = shift_str
                        .trim()
                        .chars()
                        .take_while(|c| c.is_ascii_digit())
                        .collect();
                    if let Ok(bit) = digits.parse::<u32>() {
                        map.insert(const_name, bit);
                    }
                }
            }
        }
    }
    map
}

pub fn verify_capability_bitmasks(repo_root: &Path) -> SuiteReport {
    let mut checks = Vec::new();
    let ffi_path = repo_root.join("crates/vulkanshaders-core/src/ffi/mod.rs");
    let bridge_path =
        repo_root.join("src/main/kotlin/dev/vulkanshaders/backend/NativeCoreBridge.kt");
    let ctx_path =
        repo_root.join("src/main/kotlin/dev/vulkanshaders/backend/VulkanDeviceContext.kt");

    let (ffi_src, bridge_src, ctx_src) = match (
        fs::read_to_string(&ffi_path),
        fs::read_to_string(&bridge_path),
        fs::read_to_string(&ctx_path),
    ) {
        (Ok(a), Ok(b), Ok(c)) => (a, b, c),
        (a, b, c) => {
            return SuiteReport::from_checks(
                "5. Capability Bitmask Parity",
                vec![CheckResult::fail(
                    "read capability sources",
                    format!("ffi={}, bridge={}, ctx={}", a.is_ok(), b.is_ok(), c.is_ok()),
                )],
            );
        }
    };

    let rust_query = extract_rust_cap_bits(&ffi_src, "vs_core_query_capabilities");
    let rust_get = extract_rust_cap_bits(&ffi_src, "vs_core_get_capabilities_mask");
    let kt_caps = extract_kotlin_cap_consts(&bridge_src);

    let expected_map: &[(&str, u32, &str)] = &[
        ("CAP_RAY_TRACING_AS", 0, "acceleration_structure"),
        ("CAP_RAY_QUERY", 1, "ray_query"),
        ("CAP_RAY_TRACING_PIPELINE", 2, "ray_tracing_pipeline"),
        ("CAP_BINDLESS", 3, "descriptor_indexing"),
        ("CAP_REFLEX", 4, "low_latency2_nv"),
        ("CAP_DLSS_SR", 5, "dlss_sr_available"),
        ("CAP_DLSS_RR", 6, "dlss_rr_available"),
        ("CAP_DLSS_FG", 7, "dlss_fg_available"),
        ("CAP_FSR3", 8, "fsr3_loaded"),
        ("CAP_XESS", 9, "xess_loaded"),
        ("CAP_NRD", 10, "nrd_loaded"),
        ("CAP_HDR_SWAPCHAIN", 11, "hdr_swapchain_colorspace"),
        ("CAP_BUFFER_DEVICE_ADDRESS", 12, "buffer_device_address"),
    ];

    let mut issues = Vec::new();
    for &(kt_name, bit, rust_field) in expected_map {
        match kt_caps.get(kt_name) {
            Some(&got) if got == bit => {}
            Some(&got) => issues.push(format!("{kt_name}: Kotlin bit={got} != expected {bit}")),
            None => issues.push(format!("{kt_name}: missing in NativeCoreBridge.kt")),
        }
        for (fn_label, rust_bits) in [
            ("vs_core_query_capabilities", &rust_query),
            ("vs_core_get_capabilities_mask", &rust_get),
        ] {
            match rust_bits.get(&bit) {
                Some(cond) if cond.contains(rust_field) => {}
                Some(cond) => issues.push(format!(
                    "{fn_label} bit {bit}: condition '{cond}' does not reference '{rust_field}'"
                )),
                None => issues.push(format!("{fn_label}: missing bit {bit} ({rust_field})")),
            }
        }
    }

    if issues.is_empty() {
        checks.push(CheckResult::pass(
            "Rust <-> NativeCoreBridge CAP_* bit shifts",
            format!(
                "all {} capability bits (0..={}) match across vs_core_query_capabilities, vs_core_get_capabilities_mask, and NativeCoreBridge.kt",
                expected_map.len(),
                expected_map.len() - 1
            ),
        ));
    } else {
        for issue in issues {
            checks.push(CheckResult::fail("CAP_* bitmask mismatch", issue));
        }
    }

    // Verify VulkanDeviceContext.kt uses NativeCoreBridge.CAP_* constants.
    let required_ctx_refs = [
        "NativeCoreBridge.CAP_RAY_TRACING_AS",
        "NativeCoreBridge.CAP_RAY_QUERY",
        "NativeCoreBridge.CAP_RAY_TRACING_PIPELINE",
        "NativeCoreBridge.CAP_BINDLESS",
        "NativeCoreBridge.CAP_REFLEX",
    ];
    let missing_ctx: Vec<&str> = required_ctx_refs
        .iter()
        .copied()
        .filter(|needle| !ctx_src.contains(needle))
        .collect();
    if missing_ctx.is_empty() {
        checks.push(CheckResult::pass(
            "VulkanDeviceContext CAP_* references",
            "uses canonical NativeCoreBridge.CAP_* constants for all 5 device capability flags",
        ));
    } else {
        checks.push(CheckResult::fail(
            "VulkanDeviceContext CAP_* references",
            format!("missing references: {}", missing_ctx.join(", ")),
        ));
    }

    SuiteReport::from_checks("5. Capability Bitmask Parity", checks)
}

// ============================================================================
// Suite 6: Shader Binding & GLSL Static Contract
// ============================================================================

fn collect_files_recursive(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(Result::ok).map(|e| e.path()).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            collect_files_recursive(&path, out);
        } else if path.is_file() {
            out.push(path);
        }
    }
}

fn is_shader_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some(
            "comp"
                | "rgen"
                | "rchit"
                | "rahit"
                | "rmiss"
                | "vert"
                | "frag"
                | "vsh"
                | "fsh"
                | "glsl"
        )
    )
}

fn is_compute_or_rt_shader(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("comp" | "rgen" | "rchit" | "rahit" | "rmiss")
    )
}

fn extract_glsl_set_bindings(clean_src: &str) -> Vec<(u32, u32)> {
    let mut bindings = Vec::new();
    let mut search_from = 0;
    while let Some(pos) = clean_src[search_from..].find("layout") {
        let abs = search_from + pos;
        let before_ok = abs == 0 || !is_ident_char(clean_src.as_bytes()[abs - 1]);
        let after = &clean_src[abs + "layout".len()..];
        let trimmed = after.trim_start();
        if before_ok && trimmed.starts_with('(') {
            let open_offset = abs + "layout".len() + (after.len() - trimmed.len());
            if let Some(close_offset) = find_matching_paren(clean_src, open_offset) {
                let inner = &clean_src[open_offset + 1..close_offset];
                if let (Some(set_val), Some(bind_val)) = (
                    extract_glsl_layout_num(inner, "set"),
                    extract_glsl_layout_num(inner, "binding"),
                ) {
                    bindings.push((set_val, bind_val));
                }
                search_from = close_offset + 1;
                continue;
            }
        }
        search_from = abs + "layout".len();
    }
    bindings
}

fn extract_glsl_layout_num(inner: &str, key: &str) -> Option<u32> {
    for part in inner.split(',') {
        let p = part.trim();
        if let Some((lhs, rhs)) = p.split_once('=') {
            if lhs.trim() == key {
                return rhs.trim().parse::<u32>().ok();
            }
        }
    }
    None
}

fn has_illegal_implicit_lod_texture_call(clean_src: &str) -> bool {
    let bytes = clean_src.as_bytes();
    let mut search_from = 0;
    while let Some(pos) = clean_src[search_from..].find("texture") {
        let abs = search_from + pos;
        let before_ok = abs == 0 || !is_ident_char(bytes[abs - 1]);
        let after_idx = abs + "texture".len();
        let after = &clean_src[after_idx..];
        let trimmed = after.trim_start();
        if before_ok && trimmed.starts_with('(') {
            return true;
        }
        search_from = after_idx;
    }
    false
}

pub fn verify_shader_contracts(repo_root: &Path) -> SuiteReport {
    let mut checks = Vec::new();
    let roots = [
        repo_root.join("src/main/resources/builtin-packs/hyaloclast-sunlit-vanilla.vsp/shaders"),
        repo_root.join("src/main/resources/assets/vulkanshaders/shaders/core"),
    ];

    let mut shader_files = Vec::new();
    for root in &roots {
        collect_files_recursive(root, &mut shader_files);
    }
    shader_files.retain(|p| is_shader_file(p));

    if shader_files.is_empty() {
        checks.push(CheckResult::fail(
            "shader discovery",
            "no GLSL shader files found in builtin-packs or core shaders",
        ));
        return SuiteReport::from_checks("6. Shader Binding & GLSL Static Contract", checks);
    }

    let mut dup_issues = Vec::new();
    let mut tex_issues = Vec::new();
    let mut total_bindings = 0usize;

    for path in &shader_files {
        let rel = path
            .strip_prefix(repo_root)
            .unwrap_or(path)
            .display()
            .to_string();
        let Ok(src) = fs::read_to_string(path) else {
            dup_issues.push(format!("{rel}: unreadable"));
            continue;
        };
        let clean = strip_c_style_comments(&src);
        let bindings = extract_glsl_set_bindings(&clean);
        total_bindings += bindings.len();

        let mut seen = BTreeSet::new();
        for (s, b) in bindings {
            if !seen.insert((s, b)) {
                dup_issues.push(format!("{rel}: duplicate layout(set={s}, binding={b})"));
            }
        }

        if is_compute_or_rt_shader(path) && has_illegal_implicit_lod_texture_call(&clean) {
            tex_issues.push(format!(
                "{rel}: illegal implicit-LOD texture() call in compute/RT shader"
            ));
        }
    }

    if dup_issues.is_empty() {
        checks.push(CheckResult::pass(
            "GLSL (set, binding) uniqueness",
            format!(
                "checked {} shaders ({total_bindings} bindings): 0 duplicate (set, binding) pairs",
                shader_files.len()
            ),
        ));
    } else {
        checks.push(CheckResult::fail(
            "GLSL (set, binding) uniqueness",
            dup_issues.join("; "),
        ));
    }

    if tex_issues.is_empty() {
        checks.push(CheckResult::pass(
            "Compute/RT implicit-LOD texture() ban",
            "0 illegal texture() calls in .comp/.rgen/.rchit/.rahit/.rmiss shaders",
        ));
    } else {
        checks.push(CheckResult::fail(
            "Compute/RT implicit-LOD texture() ban",
            tex_issues.join("; "),
        ));
    }

    // Verify rayquery_lighting.comp excludes CRUMBLING_MASK (uses 0xF7, not 0xFF).
    let rq_path = repo_root
        .join("src/main/resources/builtin-packs/hyaloclast-sunlit-vanilla.vsp/shaders/compute/rayquery_lighting.comp");
    match fs::read_to_string(&rq_path) {
        Ok(src) => {
            let clean = strip_c_style_comments(&src);
            let mut rq_calls = 0usize;
            let mut bad_calls = Vec::new();
            let mut has_f7 = false;
            let mut search_from = 0;
            while let Some(pos) = clean[search_from..].find("rayQueryInitializeEXT") {
                let open = search_from + pos + "rayQueryInitializeEXT".len();
                if let Some(p_open) = clean[open..].find('(') {
                    let abs_open = open + p_open;
                    if let Some(abs_close) = find_matching_paren(&clean, abs_open) {
                        let args = &clean[abs_open + 1..abs_close];
                        rq_calls += 1;
                        let args_upper = args.to_ascii_uppercase();
                        if args_upper.contains("0XFF") {
                            bad_calls.push(format!("call #{rq_calls} passes 0xFF cull mask"));
                        }
                        if args_upper.contains("0XF7") {
                            has_f7 = true;
                        }
                        search_from = abs_close + 1;
                        continue;
                    }
                }
                search_from = open;
            }
            if rq_calls == 0 {
                checks.push(CheckResult::fail(
                    "rayquery_lighting.comp CRUMBLING_MASK",
                    "no rayQueryInitializeEXT calls found",
                ));
            } else if !bad_calls.is_empty() || !has_f7 {
                checks.push(CheckResult::fail(
                    "rayquery_lighting.comp CRUMBLING_MASK",
                    format!("has_0xF7={has_f7}, issues=[{}]", bad_calls.join("; ")),
                ));
            } else {
                checks.push(CheckResult::pass(
                    "rayquery_lighting.comp CRUMBLING_MASK",
                    format!(
                        "all {rq_calls} rayQueryInitializeEXT calls exclude CRUMBLING_MASK (0xF7, no 0xFF)"
                    ),
                ));
            }
        }
        Err(e) => checks.push(CheckResult::fail(
            "rayquery_lighting.comp CRUMBLING_MASK",
            format!("failed to read {}: {e}", rq_path.display()),
        )),
    }

    SuiteReport::from_checks("6. Shader Binding & GLSL Static Contract", checks)
}

// ============================================================================
// Suite 7: Mixin JSON & Compat Guard Verification
// ============================================================================

fn verify_single_mixin_json(
    repo_root: &Path,
    json_path: &Path,
    java_src_root: &Path,
    checks: &mut Vec<CheckResult>,
) {
    let label = json_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("mixins.json");

    let raw = match fs::read_to_string(json_path) {
        Ok(s) => s,
        Err(e) => {
            checks.push(CheckResult::fail(
                format!("{label} parse"),
                format!("failed to read {}: {e}", json_path.display()),
            ));
            return;
        }
    };
    let val: Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            checks.push(CheckResult::fail(
                format!("{label} parse"),
                format!("invalid JSON: {e}"),
            ));
            return;
        }
    };

    let Some(pkg) = val.get("package").and_then(Value::as_str) else {
        checks.push(CheckResult::fail(
            format!("{label} package"),
            "missing \"package\" field",
        ));
        return;
    };
    let pkg_dir = java_src_root.join(pkg.replace('.', "/"));

    let mut entries = Vec::new();
    for key in ["mixins", "client", "server"] {
        if let Some(arr) = val.get(key).and_then(Value::as_array) {
            for item in arr {
                if let Some(s) = item.as_str() {
                    entries.push(s.to_string());
                }
            }
        }
    }

    let mut missing_files = Vec::new();
    let mut compat_issues = Vec::new();
    let mut compat_count = 0usize;

    for entry in &entries {
        let top_level = entry.split('$').next().unwrap_or(entry);
        let rel_java = format!("{}.java", top_level.replace('.', "/"));
        let java_path = pkg_dir.join(&rel_java);

        if !java_path.is_file() {
            missing_files.push(format!(
                "{entry} (expected {})",
                java_path
                    .strip_prefix(repo_root)
                    .unwrap_or(&java_path)
                    .display()
            ));
            continue;
        }

        if entry.starts_with("compat.") {
            compat_count += 1;
            if let Ok(src) = fs::read_to_string(&java_path) {
                let clean = strip_c_style_comments(&src);
                let has_external_targets = clean.contains("@Mixin(") && clean.contains("targets");
                if has_external_targets && !clean.contains("@Pseudo") {
                    compat_issues.push(format!("{entry}: external target mixin missing @Pseudo"));
                }
                // Check every injector annotation has `require = 0`.
                for ann in [
                    "@Inject",
                    "@Redirect",
                    "@ModifyArg",
                    "@ModifyArgs",
                    "@ModifyVariable",
                    "@ModifyConstant",
                ] {
                    let mut from = 0;
                    while let Some(pos) = clean[from..].find(ann) {
                        let abs = from + pos + ann.len();
                        let after = &clean[abs..];
                        let trimmed = after.trim_start();
                        if trimmed.starts_with('(') {
                            let open = abs + (after.len() - trimmed.len());
                            if let Some(close) = find_matching_paren(&clean, open) {
                                let body = &clean[open + 1..close];
                                let compact: String =
                                    body.chars().filter(|c| !c.is_whitespace()).collect();
                                if !compact.contains("require=0") {
                                    compat_issues.push(format!(
                                        "{entry}: {ann} annotation missing require = 0"
                                    ));
                                }
                                from = close + 1;
                                continue;
                            }
                        }
                        from = abs;
                    }
                }
            }
        }
    }

    if missing_files.is_empty() {
        checks.push(CheckResult::pass(
            format!("{label} source files"),
            format!(
                "all {} registered mixin entries resolve to existing .java files",
                entries.len()
            ),
        ));
    } else {
        checks.push(CheckResult::fail(
            format!("{label} source files"),
            format!("missing files: {}", missing_files.join(", ")),
        ));
    }

    if compat_issues.is_empty() {
        checks.push(CheckResult::pass(
            format!("{label} compat.* guards"),
            format!(
                "all {compat_count} compat.* mixins satisfy @Pseudo (for external targets) and require = 0"
            ),
        ));
    } else {
        checks.push(CheckResult::fail(
            format!("{label} compat.* guards"),
            compat_issues.join("; "),
        ));
    }
}

pub fn verify_mixins_and_compat(repo_root: &Path) -> SuiteReport {
    let mut checks = Vec::new();

    let vs_json = repo_root.join("src/main/resources/vulkanshaders.mixins.json");
    let vs_java = repo_root.join("src/main/java");
    verify_single_mixin_json(repo_root, &vs_json, &vs_java, &mut checks);

    // Optional sibling VulkanTurbo mixin JSON if present.
    let vt_candidates = [
        repo_root.join("VulkanTurbo/src/main/resources/vulkanturbo.mixins.json"),
        repo_root.join("../VulkanTurbo/src/main/resources/vulkanturbo.mixins.json"),
    ];
    for vt_json in vt_candidates {
        if vt_json.is_file() {
            if let Some(vt_java) = vt_json
                .parent()
                .and_then(Path::parent)
                .map(|p| p.join("java"))
            {
                verify_single_mixin_json(repo_root, &vt_json, &vt_java, &mut checks);
            }
            break;
        }
    }

    SuiteReport::from_checks("7. Mixin JSON & Compat Guard Verification", checks)
}

// ============================================================================
// Lexing / Comment-Stripping Helpers (Zero External Regex Dependency)
// ============================================================================

fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn is_c_symbol_ident(s: &str) -> bool {
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn strip_c_style_comments(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    let mut in_string = false;

    while i < bytes.len() {
        if in_string {
            let c = input[i..].chars().next().unwrap();
            out.push(c);
            if c == '\\' && i + 1 < bytes.len() {
                i += c.len_utf8();
                let next = input[i..].chars().next().unwrap();
                out.push(next);
                i += next.len_utf8();
                continue;
            }
            if c == '"' {
                in_string = false;
            }
            i += c.len_utf8();
            continue;
        }

        if bytes[i] == b'"' {
            in_string = true;
            out.push('"');
            i += 1;
        } else if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                if bytes[i] == b'\n' {
                    out.push('\n');
                }
                i += 1;
            }
            if i + 1 < bytes.len() {
                i += 2;
            }
        } else {
            let c = input[i..].chars().next().unwrap();
            out.push(c);
            i += c.len_utf8();
        }
    }
    out
}

fn parse_quoted_literal(s: &str, quote_pos: usize) -> Option<(String, usize)> {
    let bytes = s.as_bytes();
    if bytes.get(quote_pos) != Some(&b'"') {
        return None;
    }
    let mut i = quote_pos + 1;
    let mut out = String::new();
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            out.push('\\');
            out.push(bytes[i + 1] as char);
            i += 2;
        } else if bytes[i] == b'"' {
            return Some((out, i + 1));
        } else {
            let c = s[i..].chars().next()?;
            out.push(c);
            i += c.len_utf8();
        }
    }
    None
}

fn find_matching_paren(s: &str, open_pos: usize) -> Option<usize> {
    find_matching_delim(s, open_pos, b'(', b')')
}

fn find_matching_brace(s: &str, open_pos: usize) -> Option<usize> {
    find_matching_delim(s, open_pos, b'{', b'}')
}

fn find_matching_delim(s: &str, open_pos: usize, open: u8, close: u8) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.get(open_pos) != Some(&open) {
        return None;
    }
    let mut depth = 0usize;
    let mut i = open_pos;
    let mut in_string = false;
    while i < bytes.len() {
        if in_string {
            if bytes[i] == b'\\' && i + 1 < bytes.len() {
                i += 2;
                continue;
            }
            if bytes[i] == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if bytes[i] == b'"' {
            in_string = true;
        } else if bytes[i] == open {
            depth += 1;
        } else if bytes[i] == close {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}
