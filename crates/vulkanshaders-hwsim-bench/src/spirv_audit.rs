// SPDX-License-Identifier: GPL-3.0-only
// Additional permissions under GPLv3 Section 7 apply to optional proprietary vendor SDK
// runtime loading; see LICENSE-EXCEPTION.md in the repository root.

//! Pure-Rust SPIR-V module decoder, PE32+/ELF64 embedded shader carver, compressed container
//! detector, and GPU hardware-profile compatibility auditor.
//!
//! Explicitly detects compressed shader payloads (zstd/zlib/lz4) inside vendor binaries and marks
//! them `INCONCLUSIVE_UNINSPECTED` rather than reporting a false-positive zero-error pass.

use crate::error::HwsimError;
use crate::profiles::{CoopComponentType, FeatureFlags, GpuProfile};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::Path;

/// Standard SPIR-V little-endian magic word.
pub const SPIRV_MAGIC: u32 = 0x0723_0203;

/// Core SPIR-V opcodes used by the zero-panic streaming parser.
const OP_EXTENSION: u16 = 10;
const OP_EXECUTION_MODE: u16 = 16;
const OP_CAPABILITY: u16 = 17;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_CONSTANT: u16 = 43;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_VARIABLE: u16 = 59;
const OP_DECORATE: u16 = 71;
const OP_TYPE_COOPERATIVE_MATRIX_KHR: u16 = 4456;

/// SPIR-V capability IDs including `SPV_EXT_float8` and `SPV_KHR_cooperative_matrix`.
pub const CAP_FLOAT16: u32 = 9;
pub const CAP_INT8: u32 = 39;
pub const CAP_STORAGE_BUFFER_16BIT: u32 = 4433;
pub const CAP_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT: u32 = 56;
pub const CAP_GROUP_NON_UNIFORM: u32 = 61;
pub const CAP_FLOAT8_EXT: u32 = 4212;
pub const CAP_FLOAT8_COOP_MATRIX_EXT: u32 = 4213;
pub const CAP_RAY_QUERY_KHR: u32 = 4472;
pub const CAP_RAY_TRACING_KHR: u32 = 4479;
pub const CAP_DOT_PRODUCT_KHR: u32 = 6019;
pub const CAP_COOP_MATRIX_KHR: u32 = 6022;

/// `SPV_EXT_float8` FP encoding literals in `OpTypeFloat` word 3.
pub const FP_ENCODING_FLOAT8_E4M3_EXT: u32 = 4214;
pub const FP_ENCODING_FLOAT8_E5M2_EXT: u32 = 4215;

/// Execution mode IDs.
const EXEC_MODE_LOCAL_SIZE: u32 = 17;
const EXEC_MODE_SUBGROUP_SIZE: u32 = 35;

/// Decoration IDs.
const DEC_BINDING: u32 = 33;
const DEC_DESCRIPTOR_SET: u32 = 34;

/// Compressed container signature found in a vendor binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressedContainerKind {
    Zstd,
    Zlib,
    Lz4Frame,
}

/// Detected compressed payload marker inside a scanned binary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedPayloadHit {
    pub kind: CompressedContainerKind,
    pub section: String,
    pub file_offset: usize,
}

/// Decoded cooperative matrix type usage inside a SPIR-V module.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoopMatrixUsage {
    pub component_type: Option<CoopComponentType>,
    pub rows: Option<u32>,
    pub cols: Option<u32>,
}

/// Descriptor binding extracted from `OpDecorate` + `OpVariable`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DescriptorBinding {
    pub set: u32,
    pub binding: u32,
    pub storage_class: u32,
}

/// Parsed metadata for a single SPIR-V module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpirvModuleAudit {
    pub blake3_hex: String,
    pub source_label: String,
    pub word_count: usize,
    pub spirv_version_major: u8,
    pub spirv_version_minor: u8,
    pub capabilities: BTreeSet<u32>,
    pub extensions: BTreeSet<String>,
    pub uses_fp8_e4m3: bool,
    pub uses_fp8_e5m2: bool,
    pub uses_fp16: bool,
    pub uses_int8: bool,
    pub coop_matrices: Vec<CoopMatrixUsage>,
    pub local_size: Option<[u32; 3]>,
    pub subgroup_size: Option<u32>,
    pub bindings: Vec<DescriptorBinding>,
}

/// Compatibility verdict for one SPIR-V module against one GPU profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpirvProfileCompat {
    pub gpu_id: String,
    pub driver_model: String,
    pub compatible: bool,
    pub blockers: Vec<String>,
}

/// Result of scanning a file (`.spv`, `.dll`, or `.so`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryScanResult {
    pub path: String,
    pub container_format: String,
    pub file_blake3: String,
    pub carved_modules: Vec<SpirvModuleAudit>,
    pub compressed_hits: Vec<CompressedPayloadHit>,
    pub inconclusive_compressed: bool,
}

/// Parses a validated SPIR-V word slice without panicking on unknown/bleeding-edge opcodes.
pub fn parse_spirv_words(source_label: &str, words: &[u32]) -> Result<SpirvModuleAudit, HwsimError> {
    if words.len() < 5 || words[0] != SPIRV_MAGIC {
        return Err(HwsimError::SpirvParse {
            source_label: source_label.to_string(),
            detail: "missing SPIR-V magic header 0x07230203".to_string(),
        });
    }
    let version = words[1];
    let major = ((version >> 16) & 0xFF) as u8;
    let minor = ((version >> 8) & 0xFF) as u8;
    if major != 1 || minor > 6 || (version & 0xFF00_00FF) != 0 {
        return Err(HwsimError::SpirvParse {
            source_label: source_label.to_string(),
            detail: format!("invalid SPIR-V version word {version:#010x}"),
        });
    }
    if words[3] == 0 || words[4] != 0 {
        return Err(HwsimError::SpirvParse {
            source_label: source_label.to_string(),
            detail: format!("invalid SPIR-V bound={} or schema={}", words[3], words[4]),
        });
    }

    let mut capabilities = BTreeSet::new();
    let mut extensions = BTreeSet::new();
    let mut uses_fp8_e4m3 = false;
    let mut uses_fp8_e5m2 = false;
    let mut uses_fp16 = false;
    let mut uses_int8 = false;
    let mut local_size = None;
    let mut subgroup_size = None;

    let mut scalar_types: HashMap<u32, CoopComponentType> = HashMap::new();
    let mut constants_u32: HashMap<u32, u32> = HashMap::new();
    let mut raw_coop: Vec<(u32, u32, u32)> = Vec::new();
    let mut dec_set: HashMap<u32, u32> = HashMap::new();
    let mut dec_binding: HashMap<u32, u32> = HashMap::new();
    let mut vars: Vec<(u32, u32)> = Vec::new();

    let mut i = 5usize;
    while i < words.len() {
        let header = words[i];
        let wc = (header >> 16) as usize;
        let opcode = (header & 0xFFFF) as u16;
        if wc == 0 || i + wc > words.len() {
            return Err(HwsimError::SpirvParse {
                source_label: source_label.to_string(),
                detail: format!("truncated instruction at word {i} (wc={wc}, len={})", words.len()),
            });
        }
        let ops = &words[i + 1..i + wc];
        match opcode {
            OP_CAPABILITY if !ops.is_empty() => {
                capabilities.insert(ops[0]);
                if ops[0] == CAP_FLOAT16 {
                    uses_fp16 = true;
                } else if ops[0] == CAP_INT8 {
                    uses_int8 = true;
                }
            }
            OP_EXTENSION if !ops.is_empty() => {
                if let Some(ext) = decode_spirv_string(ops) {
                    extensions.insert(ext);
                }
            }
            OP_EXECUTION_MODE if ops.len() >= 2 => {
                let mode = ops[1];
                if mode == EXEC_MODE_LOCAL_SIZE && ops.len() >= 5 {
                    local_size = Some([ops[2], ops[3], ops[4]]);
                } else if mode == EXEC_MODE_SUBGROUP_SIZE && ops.len() >= 3 {
                    subgroup_size = Some(ops[2]);
                }
            }
            OP_TYPE_INT if ops.len() >= 3 => {
                let id = ops[0];
                let width = ops[1];
                if width == 8 {
                    uses_int8 = true;
                    scalar_types.insert(id, CoopComponentType::Int8);
                } else if width == 32 {
                    scalar_types.insert(id, CoopComponentType::Int32);
                }
            }
            OP_TYPE_FLOAT if ops.len() >= 2 => {
                let id = ops[0];
                let width = ops[1];
                match width {
                    8 => {
                        let enc = ops.get(2).copied().unwrap_or(0);
                        if enc == FP_ENCODING_FLOAT8_E5M2_EXT {
                            uses_fp8_e5m2 = true;
                            scalar_types.insert(id, CoopComponentType::Float8E5M2);
                        } else {
                            uses_fp8_e4m3 = true;
                            scalar_types.insert(id, CoopComponentType::Float8E4M3);
                        }
                    }
                    16 => {
                        uses_fp16 = true;
                        scalar_types.insert(id, CoopComponentType::Float16);
                    }
                    32 => {
                        scalar_types.insert(id, CoopComponentType::Float32);
                    }
                    _ => {}
                }
            }
            OP_CONSTANT if ops.len() >= 3 => {
                constants_u32.insert(ops[1], ops[2]);
            }
            OP_TYPE_COOPERATIVE_MATRIX_KHR if ops.len() >= 6 => {
                // ResultId=ops[0], ComponentType=ops[1], Scope=ops[2], Rows=ops[3], Cols=ops[4], Use=ops[5]
                raw_coop.push((ops[1], ops[3], ops[4]));
            }
            OP_DECORATE if ops.len() >= 3 => {
                let target = ops[0];
                let dec = ops[1];
                if dec == DEC_DESCRIPTOR_SET {
                    dec_set.insert(target, ops[2]);
                } else if dec == DEC_BINDING {
                    dec_binding.insert(target, ops[2]);
                }
            }
            OP_VARIABLE if ops.len() >= 3 => {
                vars.push((ops[1], ops[2]));
            }
            _ => {}
        }
        i += wc;
    }

    let coop_matrices = raw_coop
        .into_iter()
        .map(|(ty_id, rows_id, cols_id)| CoopMatrixUsage {
            component_type: scalar_types.get(&ty_id).copied(),
            rows: constants_u32.get(&rows_id).copied(),
            cols: constants_u32.get(&cols_id).copied(),
        })
        .collect();

    let mut bindings = Vec::new();
    for (var_id, storage_class) in vars {
        if let (Some(&set), Some(&binding)) = (dec_set.get(&var_id), dec_binding.get(&var_id)) {
            bindings.push(DescriptorBinding {
                set,
                binding,
                storage_class,
            });
        }
    }
    bindings.sort();

    let raw_bytes: &[u8] = bytemuck::cast_slice(words);
    let blake3_hex = blake3::hash(raw_bytes).to_hex().to_string();

    Ok(SpirvModuleAudit {
        blake3_hex,
        source_label: source_label.to_string(),
        word_count: words.len(),
        spirv_version_major: major,
        spirv_version_minor: minor,
        capabilities,
        extensions,
        uses_fp8_e4m3,
        uses_fp8_e5m2,
        uses_fp16,
        uses_int8,
        coop_matrices,
        local_size,
        subgroup_size,
        bindings,
    })
}

fn decode_spirv_string(words: &[u32]) -> Option<String> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for &w in words {
        for b in w.to_le_bytes() {
            if b == 0 {
                return String::from_utf8(bytes).ok();
            }
            bytes.push(b);
        }
    }
    String::from_utf8(bytes).ok()
}

/// Checks whether a parsed SPIR-V module is compatible with `profile`.
pub fn audit_module_against_profile(
    module: &SpirvModuleAudit,
    profile: &GpuProfile,
) -> SpirvProfileCompat {
    let mut blockers = Vec::new();

    if module.capabilities.contains(&CAP_FLOAT8_EXT)
        || module.uses_fp8_e4m3
        || module.uses_fp8_e5m2
        || module.extensions.contains("SPV_EXT_float8")
    {
        if !profile.features.contains(FeatureFlags::SHADER_FLOAT8)
            || !profile.has_extension("VK_EXT_shader_float8")
        {
            blockers.push("requires VK_EXT_shader_float8 (SPV_EXT_float8 / Float8EXT)".to_string());
        }
    }

    if module.capabilities.contains(&CAP_FLOAT8_COOP_MATRIX_EXT)
        && !profile
            .features
            .contains(FeatureFlags::SHADER_FLOAT8_COOP_MATRIX)
    {
        blockers.push("requires shaderFloat8CooperativeMatrix feature".to_string());
    }

    if module.capabilities.contains(&CAP_COOP_MATRIX_KHR)
        || module.extensions.contains("SPV_KHR_cooperative_matrix")
        || !module.coop_matrices.is_empty()
    {
        if !profile.features.contains(FeatureFlags::COOP_MATRIX_KHR)
            || !profile.has_extension("VK_KHR_cooperative_matrix")
        {
            blockers.push("requires VK_KHR_cooperative_matrix".to_string());
        } else {
            for cm in &module.coop_matrices {
                if let (Some(ct), Some(r), Some(c)) = (cm.component_type, cm.rows, cm.cols) {
                    let matched = profile.coop_matrix_shapes.iter().any(|s| {
                        (s.a_type == ct || s.b_type == ct || s.c_type == ct || s.result_type == ct)
                            && (s.m == r || s.n == r || s.k == r)
                            && (s.m == c || s.n == c || s.k == c)
                    });
                    if !matched {
                        blockers.push(format!(
                            "cooperative matrix {r}x{c} ({ct:?}) not supported by {}",
                            profile.display_name
                        ));
                    }
                }
            }
        }
    }

    if module.capabilities.contains(&CAP_DOT_PRODUCT_KHR)
        && !profile.features.contains(FeatureFlags::INT8_DOT_PRODUCT)
    {
        blockers.push("requires shaderIntegerDotProduct (DP4a)".to_string());
    }

    if module.capabilities.contains(&CAP_RAY_TRACING_KHR)
        && !profile.features.contains(FeatureFlags::RT_PIPELINE)
    {
        blockers.push("requires VK_KHR_ray_tracing_pipeline".to_string());
    }

    if module.capabilities.contains(&CAP_RAY_QUERY_KHR)
        && !profile.features.contains(FeatureFlags::RAY_QUERY)
    {
        blockers.push("requires VK_KHR_ray_query".to_string());
    }

    if let Some([x, y, z]) = module.local_size {
        let invocations = x.saturating_mul(y).saturating_mul(z);
        if invocations > profile.max_compute_workgroup_invocations {
            blockers.push(format!(
                "LocalSize {x}x{y}x{z} ({invocations}) exceeds maxComputeWorkGroupInvocations ({})",
                profile.max_compute_workgroup_invocations
            ));
        }
    }

    if let Some(sg) = module.subgroup_size
        && (sg < profile.min_subgroup_size || sg > profile.max_subgroup_size)
    {
        blockers.push(format!(
            "SubgroupSize {sg} outside supported range [{}..={}]",
            profile.min_subgroup_size, profile.max_subgroup_size
        ));
    }

    if let Some(max_set) = module.bindings.iter().map(|b| b.set).max()
        && max_set + 1 > profile.max_bound_descriptor_sets
    {
        blockers.push(format!(
            "descriptor set {max_set} exceeds maxBoundDescriptorSets ({})",
            profile.max_bound_descriptor_sets
        ));
    }

    SpirvProfileCompat {
        gpu_id: profile.id.slug().to_string(),
        driver_model: profile.driver.as_str().to_string(),
        compatible: blockers.is_empty(),
        blockers,
    }
}

/// Scans a standalone `.spv` file or carves embedded SPIR-V & compressed container signatures
/// from a PE32+ (`.dll`) or ELF64 (`.so`) binary without executing it.
pub fn scan_binary_file(path: &Path) -> Result<BinaryScanResult, HwsimError> {
    let bytes = fs::read(path).map_err(|source| HwsimError::Io {
        path: path.display().to_string(),
        source,
    })?;
    Ok(scan_binary_bytes(&path.display().to_string(), &bytes))
}

/// Scans an in-memory `.spv`, PE32+, or ELF64 buffer.
pub fn scan_binary_bytes(label: &str, bytes: &[u8]) -> BinaryScanResult {
    let file_blake3 = blake3::hash(bytes).to_hex().to_string();

    // Direct standalone .spv module
    if bytes.len() >= 20 && bytes.len() % 4 == 0 && u32_le(bytes, 0) == Some(SPIRV_MAGIC) {
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if let Ok(audit) = parse_spirv_words(label, &words) {
            return BinaryScanResult {
                path: label.to_string(),
                container_format: "SPIR-V".to_string(),
                file_blake3,
                carved_modules: vec![audit],
                compressed_hits: Vec::new(),
                inconclusive_compressed: false,
            };
        }
    }

    let (container_format, sections) = if bytes.len() >= 64 && &bytes[0..2] == b"MZ" {
        ("PE32+".to_string(), extract_pe_data_sections(bytes))
    } else if bytes.len() >= 64 && &bytes[0..4] == b"\x7fELF" {
        ("ELF64".to_string(), extract_elf_data_sections(bytes))
    } else {
        ("RAW".to_string(), vec![("raw".to_string(), 0usize, bytes)])
    };

    let mut carved_modules = Vec::new();
    let mut seen_hashes = BTreeSet::new();
    let mut compressed_hits = Vec::new();

    for (sec_name, base_off, slice) in sections {
        carve_spirv_from_slice(
            label,
            &sec_name,
            base_off,
            slice,
            &mut seen_hashes,
            &mut carved_modules,
        );
        detect_compressed_signatures(&sec_name, base_off, slice, &mut compressed_hits);
    }

    let inconclusive_compressed = carved_modules.is_empty() && !compressed_hits.is_empty();

    BinaryScanResult {
        path: label.to_string(),
        container_format,
        file_blake3,
        carved_modules,
        compressed_hits,
        inconclusive_compressed,
    }
}

fn u16_le(b: &[u8], off: usize) -> Option<u16> {
    let s = b.get(off..off.checked_add(2)?)?;
    Some(u16::from_le_bytes([s[0], s[1]]))
}

fn u32_le(b: &[u8], off: usize) -> Option<u32> {
    let s = b.get(off..off.checked_add(4)?)?;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn u64_le(b: &[u8], off: usize) -> Option<u64> {
    let s = b.get(off..off.checked_add(8)?)?;
    Some(u64::from_le_bytes([
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
    ]))
}

fn extract_pe_data_sections<'a>(bytes: &'a [u8]) -> Vec<(String, usize, &'a [u8])> {
    let Some(pe_off) = u32_le(bytes, 0x3C).map(|v| v as usize) else {
        return vec![("pe_full".to_string(), 0, bytes)];
    };
    let Some(pe_sig_end) = pe_off.checked_add(4) else {
        return vec![("pe_full".to_string(), 0, bytes)];
    };
    if bytes.get(pe_off..pe_sig_end) != Some(b"PE\0\0") {
        return vec![("pe_full".to_string(), 0, bytes)];
    }
    let coff = pe_off + 4;
    let num_sec = u16_le(bytes, coff + 2).unwrap_or(0) as usize;
    let opt_size = u16_le(bytes, coff + 16).unwrap_or(0) as usize;
    let sec_table = coff + 20 + opt_size;
    let mut out = Vec::new();

    for idx in 0..num_sec {
        let off = sec_table.saturating_add(idx * 40);
        let Some(hdr) = bytes.get(off..off + 40) else {
            break;
        };
        let name = String::from_utf8_lossy(&hdr[0..8])
            .trim_end_matches('\0')
            .to_string();
        let raw_size = u32_le(hdr, 16).unwrap_or(0) as usize;
        let raw_ptr = u32_le(hdr, 20).unwrap_or(0) as usize;
        if let Some(end) = raw_ptr.checked_add(raw_size)
            && end <= bytes.len()
            && matches!(name.as_str(), ".rdata" | ".data" | ".rsrc" | ".rodata")
        {
            out.push((name, raw_ptr, &bytes[raw_ptr..end]));
        }
    }
    if out.is_empty() {
        out.push(("pe_full".to_string(), 0, bytes));
    }
    out
}

fn extract_elf_data_sections<'a>(bytes: &'a [u8]) -> Vec<(String, usize, &'a [u8])> {
    if bytes.get(4) != Some(&2) || bytes.get(5) != Some(&1) {
        return vec![("elf_full".to_string(), 0, bytes)];
    }
    let shoff = u64_le(bytes, 40).unwrap_or(0) as usize;
    let shentsize = u16_le(bytes, 58).unwrap_or(0) as usize;
    let shnum = u16_le(bytes, 60).unwrap_or(0) as usize;
    let mut out = Vec::new();

    if shoff != 0 && shentsize >= 64 {
        for idx in 0..shnum {
            let Some(off) = idx.checked_mul(shentsize).and_then(|d| shoff.checked_add(d)) else {
                break;
            };
            let Some(hdr_end) = off.checked_add(64) else {
                break;
            };
            let Some(hdr) = bytes.get(off..hdr_end) else {
                break;
            };
            let sh_type = u32_le(hdr, 4).unwrap_or(0);
            let sh_offset = u64_le(hdr, 24).unwrap_or(0) as usize;
            let sh_size = u64_le(hdr, 32).unwrap_or(0) as usize;
            // SHT_PROGBITS == 1
            if sh_type == 1
                && let Some(end) = sh_offset.checked_add(sh_size)
                && end <= bytes.len()
            {
                out.push((format!("elf_sec_{idx}"), sh_offset, &bytes[sh_offset..end]));
            }
        }
    }
    if out.is_empty() {
        out.push(("elf_full".to_string(), 0, bytes));
    }
    out
}

fn carve_spirv_from_slice(
    label: &str,
    sec_name: &str,
    base_off: usize,
    slice: &[u8],
    seen_hashes: &mut BTreeSet<String>,
    out: &mut Vec<SpirvModuleAudit>,
) {
    let mut pos = 0usize;
    while pos + 20 <= slice.len() {
        if u32_le(slice, pos) == Some(SPIRV_MAGIC) {
            let ver = u32_le(slice, pos + 4).unwrap_or(0);
            let bound = u32_le(slice, pos + 12).unwrap_or(0);
            let schema = u32_le(slice, pos + 16).unwrap_or(1);
            let major = (ver >> 16) & 0xFF;
            let minor = (ver >> 8) & 0xFF;

            if major == 1
                && minor <= 6
                && (ver & 0xFF00_00FF) == 0
                && bound > 0
                && bound < (1 << 24)
                && schema == 0
            {
                // Walk instructions to find valid end of module
                let mut words = vec![
                    SPIRV_MAGIC,
                    ver,
                    u32_le(slice, pos + 8).unwrap_or(0),
                    bound,
                    schema,
                ];
                let mut cur = pos + 20;
                let mut saw_capability = false;
                let mut fn_depth = 0i32;

                while cur + 4 <= slice.len() {
                    let instr = u32_le(slice, cur).unwrap_or(0);
                    if instr == SPIRV_MAGIC && words.len() > 5 {
                        break;
                    }
                    let wc = (instr >> 16) as usize;
                    let op = (instr & 0xFFFF) as u16;
                    if wc == 0 || op == 0 || cur + wc * 4 > slice.len() {
                        break;
                    }
                    if op == OP_CAPABILITY {
                        saw_capability = true;
                    } else if op == OP_FUNCTION {
                        fn_depth += 1;
                    } else if op == OP_FUNCTION_END {
                        fn_depth -= 1;
                    }
                    for w_idx in 0..wc {
                        words.push(u32_le(slice, cur + w_idx * 4).unwrap_or(0));
                    }
                    cur += wc * 4;
                    if words.len() > 262_144 {
                        break;
                    }
                }

                if saw_capability && fn_depth == 0 {
                    let mod_label = format!("{label}:{sec_name}@0x{:x}", base_off + pos);
                    if let Ok(audit) = parse_spirv_words(&mod_label, &words)
                        && seen_hashes.insert(audit.blake3_hex.clone())
                    {
                        out.push(audit);
                        pos = cur;
                        continue;
                    }
                }
            }
        }
        pos += 4;
    }
}

fn detect_compressed_signatures(
    sec_name: &str,
    base_off: usize,
    slice: &[u8],
    out: &mut Vec<CompressedPayloadHit>,
) {
    let mut i = 0usize;
    while i + 4 <= slice.len() && out.len() < 32 {
        let sig = &slice[i..i + 4];
        let kind = if sig == [0x28, 0xB5, 0x2F, 0xFD] {
            Some(CompressedContainerKind::Zstd)
        } else if sig == [0x04, 0x22, 0x4D, 0x18] {
            Some(CompressedContainerKind::Lz4Frame)
        } else if sig[0] == 0x78 && matches!(sig[1], 0x01 | 0x5E | 0x9C | 0xDA) {
            Some(CompressedContainerKind::Zlib)
        } else {
            None
        };
        if let Some(k) = kind {
            out.push(CompressedPayloadHit {
                kind: k,
                section: sec_name.to_string(),
                file_offset: base_off + i,
            });
            i += 64;
        } else {
            i += 4;
        }
    }
}

/// Builds a minimal synthetic FP8 + 16x16x16 Cooperative Matrix compute SPIR-V module
/// for offline self-tests and benchmarks (zero external files or vendor binaries).
pub fn build_synthetic_fp8_coop_spirv() -> Vec<u32> {
    let mut w = Vec::new();
    // Header: magic, SPIR-V 1.6, generator=0, bound=32, schema=0
    w.extend_from_slice(&[SPIRV_MAGIC, 0x0001_0600, 0, 32, 0]);
    // OpCapability Shader (1)
    w.extend_from_slice(&[(2 << 16) | OP_CAPABILITY as u32, 1]);
    // OpCapability Float8EXT (4212)
    w.extend_from_slice(&[(2 << 16) | OP_CAPABILITY as u32, CAP_FLOAT8_EXT]);
    // OpCapability Float8CooperativeMatrixEXT (4213)
    w.extend_from_slice(&[(2 << 16) | OP_CAPABILITY as u32, CAP_FLOAT8_COOP_MATRIX_EXT]);
    // OpCapability CooperativeMatrixKHR (6022)
    w.extend_from_slice(&[(2 << 16) | OP_CAPABILITY as u32, CAP_COOP_MATRIX_KHR]);
    // OpExtension "SPV_EXT_float8"
    push_op_extension(&mut w, "SPV_EXT_float8");
    // OpExtension "SPV_KHR_cooperative_matrix"
    push_op_extension(&mut w, "SPV_KHR_cooperative_matrix");
    // OpExecutionMode %1 LocalSize 8 8 1
    w.extend_from_slice(&[(6 << 16) | OP_EXECUTION_MODE as u32, 1, EXEC_MODE_LOCAL_SIZE, 8, 8, 1]);
    // OpExecutionMode %1 SubgroupSize 32
    w.extend_from_slice(&[(4 << 16) | OP_EXECUTION_MODE as u32, 1, EXEC_MODE_SUBGROUP_SIZE, 32]);
    // OpTypeInt %2 32 0 (u32)
    w.extend_from_slice(&[(4 << 16) | OP_TYPE_INT as u32, 2, 32, 0]);
    // OpConstant %2 %3 16 (rows/cols = 16)
    w.extend_from_slice(&[(4 << 16) | OP_CONSTANT as u32, 2, 3, 16]);
    // OpConstant %2 %4 3 (Subgroup scope = 3)
    w.extend_from_slice(&[(4 << 16) | OP_CONSTANT as u32, 2, 4, 3]);
    // OpTypeFloat %5 8 Float8E4M3EXT (4214)
    w.extend_from_slice(&[
        (4 << 16) | OP_TYPE_FLOAT as u32,
        5,
        8,
        FP_ENCODING_FLOAT8_E4M3_EXT,
    ]);
    // OpTypeCooperativeMatrixKHR %6 %5 %4 %3 %3 %4
    w.extend_from_slice(&[
        (7 << 16) | OP_TYPE_COOPERATIVE_MATRIX_KHR as u32,
        6,
        5,
        4,
        3,
        3,
        4,
    ]);
    // OpDecorate %10 DescriptorSet 0 / Binding 0
    w.extend_from_slice(&[(4 << 16) | OP_DECORATE as u32, 10, DEC_DESCRIPTOR_SET, 0]);
    w.extend_from_slice(&[(4 << 16) | OP_DECORATE as u32, 10, DEC_BINDING, 0]);
    // OpVariable %2 %10 StorageBuffer (12)
    w.extend_from_slice(&[(4 << 16) | OP_VARIABLE as u32, 2, 10, 12]);
    w
}

fn push_op_extension(w: &mut Vec<u32>, name: &str) {
    let mut bytes = name.as_bytes().to_vec();
    bytes.push(0);
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    let wc = 1 + (bytes.len() / 4);
    w.push(((wc as u32) << 16) | OP_EXTENSION as u32);
    for chunk in bytes.chunks_exact(4) {
        w.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
}

/// Evaluates a SPIR-V module against all 16 `(GPU x OS)` profiles.
pub fn audit_module_matrix(module: &SpirvModuleAudit) -> BTreeMap<String, SpirvProfileCompat> {
    let mut map = BTreeMap::new();
    for profile in GpuProfile::all_matrix() {
        let key = format!("{}_{}", profile.id.slug(), profile.driver.as_str());
        map.insert(key, audit_module_against_profile(module, &profile));
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiles::{DriverModel, GpuId};

    #[test]
    fn synthetic_fp8_coop_shader_passes_rdna4_and_fails_rdna3_and_ampere() {
        let words = build_synthetic_fp8_coop_spirv();
        let audit = parse_spirv_words("synthetic_fsr4_fp8.spv", &words).unwrap();
        assert!(audit.uses_fp8_e4m3);
        assert_eq!(audit.coop_matrices.len(), 1);
        assert_eq!(audit.local_size, Some([8, 8, 1]));
        assert_eq!(audit.subgroup_size, Some(32));

        let rdna4 = audit_module_against_profile(
            &audit,
            &GpuProfile::build(GpuId::Rx9070Xt, DriverModel::Linux),
        );
        assert!(rdna4.compatible, "RDNA 4 should pass: {:?}", rdna4.blockers);

        let rdna3 = audit_module_against_profile(
            &audit,
            &GpuProfile::build(GpuId::Rx7900Xtx, DriverModel::Linux),
        );
        assert!(!rdna3.compatible);

        let rtx3050 = audit_module_against_profile(
            &audit,
            &GpuProfile::build(GpuId::Rtx3050Mobile, DriverModel::Linux),
        );
        assert!(!rtx3050.compatible);
    }

    #[test]
    fn compressed_zstd_payload_is_flagged_inconclusive() {
        let mut fake_blob = vec![0u8; 128];
        fake_blob[16..20].copy_from_slice(&[0x28, 0xB5, 0x2F, 0xFD]);
        let scan = scan_binary_bytes("synthetic_compressed.dll", &fake_blob);
        assert!(scan.carved_modules.is_empty());
        assert!(!scan.compressed_hits.is_empty());
        assert!(scan.inconclusive_compressed);
    }
}
