use clap::{Parser, Subcommand, ValueEnum};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
    time::Instant,
};
use vsp_compiler::{
    HardwareShaderCaps, HookStage, PackManifest, RenderTier, ShaderStageKind, SpirvHookLinker,
    VslPreprocessor, VspCompiler, validate_pack_directory,
};

type PackStageJob = (String, ShaderStageKind, String);

#[derive(Debug, Parser)]
#[command(
    name = "vsp",
    version,
    about = "Hyaloclast (IGNIS) Vulkan Shader Package (.vsp) compiler, validator, profiler, and scaffolder"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum MigrationSource {
    Iris,
    #[value(name = "bedrock-rtx")]
    BedrockRtx,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Validate `pack.toml`, `#include` graphs, and all `.vsl` / `.vsh` / `.fsh` / `.comp` stages in a `.vsp` directory.
    Check {
        /// Path to the `.vsp` shader package directory.
        pack_dir: PathBuf,
        /// Target RenderTier (`0..=3`) to validate against.
        #[arg(short, long, default_value_t = 0)]
        tier: u32,
    },
    /// Compile all shader stages in a `.vsp` directory to `.spv` binaries.
    Compile {
        /// Path to the `.vsp` shader package directory.
        pack_dir: PathBuf,
        /// Output directory for compiled `.spv` modules.
        #[arg(short, long)]
        out_dir: PathBuf,
        /// Target RenderTier (`0..=3`).
        #[arg(short, long, default_value_t = 0)]
        tier: u32,
    },
    /// Thoroughly validate `pack.toml`, zip path-traversal safety (if archive), hook signatures,
    /// and compile all shader stages with exact `.vsl`/GLSL file:line diagnostics.
    Validate {
        /// Path to a `.vsp` directory or `.zip`/`.vsp` archive.
        pack_path: PathBuf,
        /// Target RenderTier (`0..=3`) to validate against.
        #[arg(short, long, default_value_t = 0)]
        tier: u32,
        /// Validate across every tier listed in `capabilities.tiers`.
        #[arg(long)]
        all_tiers: bool,
        /// Enable `VS_HAS_SER` / `VS_HAS_SER_EXT` during validation.
        #[arg(long)]
        ser: bool,
        /// Enable `VS_HAS_OPACITY_MICROMAP` during validation.
        #[arg(long)]
        omm: bool,
    },
    /// Compile all stages uncached and report per-stage compile time, SPIR-V size,
    /// descriptor bindings, push constant size, and workgroup / ray-tracing metadata.
    Profile {
        /// Path to the `.vsp` shader package directory.
        pack_path: PathBuf,
        /// Target RenderTier (`0..=3`).
        #[arg(short, long, default_value_t = 0)]
        tier: u32,
        /// Enable `VS_HAS_SER` / `VS_HAS_SER_EXT` during profiling.
        #[arg(long)]
        ser: bool,
        /// Enable `VS_HAS_OPACITY_MICROMAP` during profiling.
        #[arg(long)]
        omm: bool,
    },
    /// Generate a clean starter `.vsp` shader pack template ready to load in Hyaloclast.
    Init {
        /// Destination directory for the new `.vsp` shader pack.
        output_dir: PathBuf,
        /// Human-readable display name for the pack.
        #[arg(long)]
        name: Option<String>,
        /// Optional explicit pack ID (defaults to a slug derived from `--name` or directory name).
        #[arg(long)]
        id: Option<String>,
        /// Pack author string written to `pack.toml`.
        #[arg(long, default_value = "Hyaloclast Author")]
        author: String,
        /// Overwrite existing files in `output_dir`.
        #[arg(long)]
        force: bool,
    },
    /// Print (or write) a concrete migration mapping & checklist for porting Iris/OptiFine or
    /// Bedrock RTX shader/PBR packs to Hyaloclast `.vsp`.
    MigrateHelp {
        /// Source ecosystem to generate migration guidance for (`iris` or `bedrock-rtx`).
        #[arg(long, value_enum)]
        from: Option<MigrationSource>,
        /// Optional output file path to write the migration guide Markdown to.
        #[arg(short, long)]
        out: Option<PathBuf>,
    },
}

fn build_hw_caps(ser: bool, omm: bool) -> HardwareShaderCaps {
    HardwareShaderCaps {
        ser_ext: ser,
        ser_nv: false,
        ada_or_newer: ser || omm,
        high_end_gpu: ser || omm,
        opacity_micromap: omm,
    }
}

fn collect_pack_stages(
    chain_dirs: &[PathBuf],
) -> Result<Vec<PackStageJob>, Box<dyn std::error::Error>> {
    let mut stage_jobs = Vec::new();
    let mut seen_stages: HashSet<(String, String)> = HashSet::new();

    for sub in ["core", "compute", "rt"] {
        let mut per_sub: BTreeMap<String, (ShaderStageKind, PathBuf)> = BTreeMap::new();
        for dir in chain_dirs {
            let sub_dir = dir.join("shaders").join(sub);
            if !sub_dir.is_dir() {
                continue;
            }
            for entry in fs::read_dir(&sub_dir)?.flatten() {
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
            let raw_source = fs::read_to_string(&path)?;
            stage_jobs.push((file_name, stage_kind, raw_source));
        }
    }
    Ok(stage_jobs)
}

fn make_chain_loader(chain_dirs: &[PathBuf]) -> impl Fn(&str) -> Option<String> + '_ {
    move |inc: &str| -> Option<String> {
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
    }
}

fn run_check_or_compile(
    pack_dir: &Path,
    out_dir: Option<&Path>,
    tier_idx: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let tier = RenderTier::from_index(tier_idx)
        .ok_or_else(|| format!("Invalid tier index {tier_idx}; expected 0..=3"))?;

    let (manifest, _chain_dirs) = VspCompiler::resolve_pack_inheritance(pack_dir, &[])?;
    let param_layout = manifest.build_parameter_layout();

    println!(
        "[vsp] Validated manifest '{}' ({} v{} by {}) — runtime params: {}, spec constants: {}",
        manifest.pack.id,
        manifest.pack.name,
        manifest.pack.version,
        manifest.pack.author,
        param_layout.runtime_ubo_indices.len(),
        param_layout.specialization_constant_ids.len(),
    );

    let cache_dir = out_dir.map(|o| o.join(".vsp-cache"));
    let compiler = VspCompiler::new(cache_dir.as_deref());
    let artifacts = compiler.compile_pack_dir(pack_dir, tier)?;

    for artifact in &artifacts {
        let entry_points = SpirvHookLinker::inspect_entry_points(&artifact.spirv_words)?;
        println!(
            "[vsp] Compiled '{}' ({:?}) -> {} SPIR-V words (entry points: {:?}, hooks: {})",
            artifact.stage_name,
            artifact.stage_kind,
            artifact.spirv_words.len(),
            entry_points,
            artifact.hooks.len()
        );

        if let Some(out) = out_dir {
            fs::create_dir_all(out)?;
            let spv_path = out.join(format!("{}.spv", artifact.stage_name));
            let bytes: &[u8] = bytemuck::cast_slice(&artifact.spirv_words);
            fs::write(&spv_path, bytes)?;
        }
    }

    Ok(())
}

/// Validates a ZIP/`.vsp` archive's Central Directory for path-traversal safety and `pack.toml` presence
/// without requiring external archive crates.
fn validate_zip_archive(archive_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    const EOCD_SIG: [u8; 4] = [0x50, 0x4b, 0x05, 0x06];
    const CD_SIG: [u8; 4] = [0x50, 0x4b, 0x01, 0x02];
    const LFH_SIG: [u8; 4] = [0x50, 0x4b, 0x03, 0x04];

    let data = fs::read(archive_path)?;
    if data.len() < 22 {
        return Err("Archive is too small to be a valid ZIP/.vsp file".into());
    }

    // Locate End of Central Directory (EOCD) record within the last 65,557 bytes.
    let search_start = data.len().saturating_sub(65_557);
    let mut eocd_pos = None;
    for pos in (search_start..=data.len() - 22).rev() {
        if data[pos..pos + 4] == EOCD_SIG {
            eocd_pos = Some(pos);
            break;
        }
    }
    let Some(eocd) = eocd_pos else {
        return Err("Invalid ZIP/.vsp archive: missing End of Central Directory signature".into());
    };

    let entry_count = u16::from_le_bytes([data[eocd + 10], data[eocd + 11]]) as usize;
    let cd_size = u32::from_le_bytes([
        data[eocd + 12],
        data[eocd + 13],
        data[eocd + 14],
        data[eocd + 15],
    ]) as usize;
    let cd_offset = u32::from_le_bytes([
        data[eocd + 16],
        data[eocd + 17],
        data[eocd + 18],
        data[eocd + 19],
    ]) as usize;

    if cd_offset.saturating_add(cd_size) > data.len() {
        return Err("Invalid ZIP/.vsp archive: central directory extends past end of file".into());
    }

    let mut cursor = cd_offset;
    let cd_end = cd_offset + cd_size;
    let mut unsafe_entries = Vec::new();
    let mut found_pack_toml: Option<(u16, usize, usize)> = None;
    let mut parsed_entries = 0usize;

    while cursor + 46 <= cd_end && parsed_entries < entry_count {
        if data[cursor..cursor + 4] != CD_SIG {
            return Err(format!(
                "Invalid ZIP/.vsp archive: corrupt central directory header at offset {cursor}"
            )
            .into());
        }
        let method = u16::from_le_bytes([data[cursor + 10], data[cursor + 11]]);
        let comp_size = u32::from_le_bytes([
            data[cursor + 20],
            data[cursor + 21],
            data[cursor + 22],
            data[cursor + 23],
        ]) as usize;
        let name_len = u16::from_le_bytes([data[cursor + 28], data[cursor + 29]]) as usize;
        let extra_len = u16::from_le_bytes([data[cursor + 30], data[cursor + 31]]) as usize;
        let comment_len = u16::from_le_bytes([data[cursor + 32], data[cursor + 33]]) as usize;
        let local_header_off = u32::from_le_bytes([
            data[cursor + 42],
            data[cursor + 43],
            data[cursor + 44],
            data[cursor + 45],
        ]) as usize;

        let name_start = cursor + 46;
        let name_end = name_start + name_len;
        if name_end > cd_end {
            return Err("Invalid ZIP/.vsp archive: truncated entry filename".into());
        }
        let entry_name = String::from_utf8_lossy(&data[name_start..name_end]).into_owned();

        // Check path-traversal safety
        let is_unsafe = entry_name.contains('\0')
            || entry_name.starts_with('/')
            || entry_name.starts_with('\\')
            || entry_name.as_bytes().get(1) == Some(&b':')
            || entry_name.split(['/', '\\']).any(|seg| seg == "..");
        if is_unsafe {
            unsafe_entries.push(entry_name.clone());
        }

        if entry_name == "pack.toml" || entry_name.ends_with("/pack.toml") {
            found_pack_toml = Some((method, local_header_off, comp_size));
        }

        cursor = name_end + extra_len + comment_len;
        parsed_entries += 1;
    }

    if !unsafe_entries.is_empty() {
        for bad in &unsafe_entries {
            eprintln!("[vsp] [error] Unsafe archive entry (path traversal): '{bad}'");
        }
        return Err(format!(
            "Archive failed path-traversal safety check ({} unsafe entries)",
            unsafe_entries.len()
        )
        .into());
    }

    println!("[vsp] Archive path-traversal safety check passed ({parsed_entries} entries scanned)");

    let Some((method, lfh_off, comp_size)) = found_pack_toml else {
        return Err("Archive does not contain a 'pack.toml' manifest".into());
    };

    if method == 0 && lfh_off + 30 <= data.len() && data[lfh_off..lfh_off + 4] == LFH_SIG {
        let lfh_name_len = u16::from_le_bytes([data[lfh_off + 26], data[lfh_off + 27]]) as usize;
        let lfh_extra_len = u16::from_le_bytes([data[lfh_off + 28], data[lfh_off + 29]]) as usize;
        let payload_start = lfh_off + 30 + lfh_name_len + lfh_extra_len;
        let payload_end = payload_start.saturating_add(comp_size);
        if payload_end <= data.len() {
            let toml_str = std::str::from_utf8(&data[payload_start..payload_end])?;
            let manifest = PackManifest::parse_and_validate(toml_str)?;
            println!(
                "[vsp] Validated stored pack.toml for '{}' ({} v{})",
                manifest.pack.id, manifest.pack.name, manifest.pack.version
            );
        }
    } else {
        println!(
            "[vsp] Found compressed 'pack.toml' (method {method}); extract the archive to run full shader stage compilation."
        );
    }

    Ok(())
}

fn run_validate(
    pack_path: &Path,
    tier_idx: u32,
    all_tiers: bool,
    ser: bool,
    omm: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if pack_path.is_file() {
        return validate_zip_archive(pack_path);
    }

    let report = validate_pack_directory(pack_path);
    for w in &report.warnings {
        println!("[vsp] {w}");
    }
    for e in &report.errors {
        eprintln!("[vsp] {e}");
    }
    if !report.is_valid() {
        return Err(format!(
            "Pack directory validation failed with {} error(s)",
            report.errors.len()
        )
        .into());
    }

    let (manifest, chain_dirs) = VspCompiler::resolve_pack_inheritance(pack_path, &[])?;
    let loader = make_chain_loader(&chain_dirs);

    // Validate hook signatures and @hook annotations
    let mut hook_errors = 0usize;
    for (stage_key, rel_path) in &manifest.hooks {
        let Some(expected_stage) = HookStage::from_name(stage_key) else {
            eprintln!("[vsp] [error] Unknown hook stage '{stage_key}'");
            hook_errors += 1;
            continue;
        };
        let Some(raw_hook) = loader(rel_path) else {
            eprintln!("[vsp] [error] Hook file '{rel_path}' for stage '{stage_key}' not found");
            hook_errors += 1;
            continue;
        };
        let annotations = VslPreprocessor::scan_hook_annotations(&raw_hook);
        let has_matching_stage = annotations.iter().any(|a| a.stage == expected_stage);
        if !has_matching_stage {
            eprintln!(
                "[vsp] [error] {rel_path}:1: missing matching '// @hook(stage = \"{stage_key}\", priority = ...)' function signature"
            );
            hook_errors += 1;
        } else {
            println!(
                "[vsp] Hook '{stage_key}' -> '{rel_path}' ({} annotation(s) verified)",
                annotations.len()
            );
        }
    }
    if hook_errors > 0 {
        return Err(format!("Hook signature validation failed with {hook_errors} error(s)").into());
    }

    let tiers_to_check: Vec<RenderTier> = if all_tiers {
        manifest
            .capabilities
            .tiers
            .iter()
            .filter_map(|&t| RenderTier::from_index(t))
            .collect()
    } else {
        vec![
            RenderTier::from_index(tier_idx)
                .ok_or_else(|| format!("Invalid tier index {tier_idx}; expected 0..=3"))?,
        ]
    };

    let hw_caps = build_hw_caps(ser, omm);
    let compiler = VspCompiler::new(None);
    let stage_jobs = collect_pack_stages(&chain_dirs)?;

    let mut total_errors = 0usize;
    for tier in tiers_to_check {
        println!(
            "[vsp] Validating '{}' ({} stages) at {:?} (SER={}, OMM={})...",
            manifest.pack.id,
            stage_jobs.len(),
            tier,
            hw_caps.has_ser(),
            hw_caps.opacity_micromap
        );
        for (stage_name, stage_kind, raw_source) in &stage_jobs {
            match compiler.compile_stage_with_diagnostics(
                stage_name,
                *stage_kind,
                raw_source,
                &manifest,
                tier,
                hw_caps,
                &loader,
            ) {
                Ok(artifact) => {
                    println!(
                        "[vsp]   OK  {stage_name} ({stage_kind:?}) — {} words, {} bindings",
                        artifact.spirv_words.len(),
                        artifact.reflection.bindings.len()
                    );
                }
                Err(diags) => {
                    total_errors += diags.len().max(1);
                    for diag in diags {
                        eprintln!("[vsp]   ERR {diag}");
                    }
                }
            }
        }
    }

    if total_errors > 0 {
        return Err(
            format!("Validation failed with {total_errors} shader diagnostic error(s)").into(),
        );
    }

    println!(
        "[vsp] Validation succeeded for '{}' ({} preset(s), {} setting(s))",
        manifest.pack.id,
        manifest.presets.len(),
        manifest.settings.len()
    );
    Ok(())
}

fn run_profile(
    pack_path: &Path,
    tier_idx: u32,
    ser: bool,
    omm: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let tier = RenderTier::from_index(tier_idx)
        .ok_or_else(|| format!("Invalid tier index {tier_idx}; expected 0..=3"))?;
    let hw_caps = build_hw_caps(ser, omm);

    let (manifest, chain_dirs) = VspCompiler::resolve_pack_inheritance(pack_path, &[])?;
    let loader = make_chain_loader(&chain_dirs);
    let stage_jobs = collect_pack_stages(&chain_dirs)?;
    let compiler = VspCompiler::new(None);

    println!(
        "=== Hyaloclast Shader Pack Profile: {} ({} v{}) ===",
        manifest.pack.id, manifest.pack.name, manifest.pack.version
    );
    println!(
        "Tier: {:?} | Stages: {} | SER: {} | OMM: {}",
        tier,
        stage_jobs.len(),
        hw_caps.has_ser(),
        hw_caps.opacity_micromap
    );
    println!(
        "{:<30} {:<10} {:>9} {:>10} {:>10} {:>8} {:>6}  WORKGROUP / RT",
        "STAGE", "KIND", "TIME(ms)", "WORDS", "BYTES", "PUSH(B)", "BINDS"
    );

    let mut total_ms = 0.0f64;
    let mut total_words = 0usize;

    for (stage_name, stage_kind, raw_source) in &stage_jobs {
        let start = Instant::now();
        let artifact = compiler.compile_stage_with_caps(
            stage_name,
            *stage_kind,
            raw_source,
            &manifest,
            tier,
            hw_caps,
            &loader,
        )?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let words = artifact.spirv_words.len();
        let bytes = words * 4;
        total_ms += elapsed_ms;
        total_words += words;

        let refl = &artifact.reflection;
        let mut meta_parts = Vec::new();
        if *stage_kind == ShaderStageKind::Compute {
            meta_parts.push(format!(
                "wg=[{},{},{}]",
                refl.workgroup_size[0], refl.workgroup_size[1], refl.workgroup_size[2]
            ));
        }
        if refl.uses_ray_tracing_pipeline {
            meta_parts.push("RT-pipeline".to_owned());
        }
        if refl.uses_ray_query {
            meta_parts.push("RayQuery".to_owned());
        }
        if refl.uses_buffer_device_address {
            meta_parts.push("BDA".to_owned());
        }
        if !refl.spec_constant_ids.is_empty() {
            meta_parts.push(format!("spec={:?}", refl.spec_constant_ids));
        }
        if meta_parts.is_empty() {
            meta_parts.push("-".to_owned());
        }

        println!(
            "{:<30} {:<10} {:>9.2} {:>10} {:>10} {:>8} {:>6}  {}",
            stage_name,
            format!("{stage_kind:?}"),
            elapsed_ms,
            words,
            bytes,
            refl.push_constant_size,
            refl.bindings.len(),
            meta_parts.join(", ")
        );

        for b in &refl.bindings {
            let rw = if b.writable { "rw" } else { "ro" };
            println!(
                "    set={} binding={} {:?}[{}] ({}) '{}'",
                b.set, b.binding, b.descriptor_type, b.count, rw, b.name
            );
        }
    }

    println!(
        "TOTAL: {} stages | {:.2} ms | {} SPIR-V words ({} bytes)",
        stage_jobs.len(),
        total_ms,
        total_words,
        total_words * 4
    );
    Ok(())
}

fn slugify_pack_id(raw: &str) -> String {
    let mut out = String::new();
    let mut prev_dash = false;
    for c in raw.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            prev_dash = false;
        } else if (c == '-' || c == '_' || c.is_whitespace()) && !prev_dash && !out.is_empty() {
            out.push('-');
            prev_dash = true;
        }
    }
    let trimmed = out.trim_matches('-').to_owned();
    if trimmed.is_empty() {
        "my-hyaloclast-pack".to_owned()
    } else {
        trimmed
    }
}

fn run_init(
    output_dir: &Path,
    name: Option<String>,
    id: Option<String>,
    author: String,
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir_stem = output_dir
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("my-hyaloclast-pack");
    let pack_name = name.unwrap_or_else(|| {
        dir_stem
            .split(['-', '_'])
            .filter(|s| !s.is_empty())
            .map(|w| {
                let mut chars = w.chars();
                match chars.next() {
                    Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                    None => String::new(),
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    });
    let pack_id = id
        .map(|s| slugify_pack_id(&s))
        .unwrap_or_else(|| slugify_pack_id(&pack_name));

    let pack_toml_path = output_dir.join("pack.toml");
    if pack_toml_path.exists() && !force {
        return Err(format!(
            "Target '{}' already contains pack.toml (pass --force to overwrite)",
            output_dir.display()
        )
        .into());
    }

    fs::create_dir_all(output_dir.join("hooks"))?;
    fs::create_dir_all(output_dir.join("shaders").join("compute"))?;

    let pack_toml = format!(
        r#"[pack]
id = "{pack_id}"
name = "{pack_name}"
version = "0.1.0"
author = "{author}"
description = "Starter Hyaloclast (.vsp) shader pack template."

[capabilities]
tiers = [0, 1, 2, 3]
pbr_labpbr = true
hdr_output = true
restir_di = true
volumetric_clouds = true
volumetric_fog = true
distant_horizons_lod = true

[hooks]
surface = "hooks/surface.vsl"
sky = "hooks/sky.vsl"
post_grade = "hooks/post_grade.vsl"

[settings.exposure]
label = "Exposure"
type = "float"
default = 1.0
min = 0.25
max = 4.0
step = 0.05
runtime = true

[settings.saturation]
label = "Color Saturation"
type = "float"
default = 1.08
min = 0.5
max = 1.8
step = 0.02
runtime = true

[settings.wet_roughness_scale]
label = "Rain Wetness Scale"
type = "float"
default = 0.35
min = 0.0
max = 1.0
step = 0.05
runtime = true

[settings.enable_aces_grade]
label = "ACES Filmic Grade"
type = "bool"
default = true
runtime = false

[presets.low]
label = "Low"
description = "Lightweight defaults for integrated and entry GPUs"
order = 0
min_tier = 0
max_tier = 3
values = {{ exposure = 1.0, saturation = 1.0, wet_roughness_scale = 0.25, enable_aces_grade = false }}

[presets.medium]
label = "Medium"
description = "Balanced visuals and performance"
order = 1
min_tier = 0
max_tier = 3
values = {{ exposure = 1.0, saturation = 1.05, wet_roughness_scale = 0.35, enable_aces_grade = true }}

[presets.high]
label = "High"
description = "Enhanced grading and rain response"
order = 2
min_tier = 1
max_tier = 3
values = {{ exposure = 1.05, saturation = 1.12, wet_roughness_scale = 0.45, enable_aces_grade = true }}

[presets.ultra]
label = "Ultra"
description = "Full dynamic range and cinematic contrast for Tier 2/3 RT"
order = 3
min_tier = 2
max_tier = 3
values = {{ exposure = 1.1, saturation = 1.18, wet_roughness_scale = 0.55, enable_aces_grade = true }}
"#
    );

    let surface_vsl = r#"#include <vulkanshaders/v1/common.vsl>

// @hook(stage = "surface", priority = 100)
void starter_surface(inout VsSurface s, in uint flags, in uint bounce) {
    // Darken and polish exposed surfaces gently when raining.
    float wetness = clamp(vsWorld.rain * VS_PARAM_WET_ROUGHNESS_SCALE, 0.0, 0.85);
    s.roughness = mix(s.roughness, max(0.06, s.roughness * 0.35), wetness);
    s.albedo *= mix(1.0, 0.82, wetness);
}
"#;

    let sky_vsl = r#"#include <vulkanshaders/v1/common.vsl>

// @hook(stage = "sky", priority = 100)
void starter_sky(inout vec3 radiance, in vec3 dir, in VsSkyState sky) {
    float horizon = pow(1.0 - clamp(abs(dir.y), 0.0, 1.0), 4.0);
    float sunElevation = clamp(sky.sunDir.y, -0.2, 1.0);
    vec3 warmHorizon = vec3(0.95, 0.55, 0.28) * smoothstep(-0.15, 0.25, sunElevation) * (1.0 - smoothstep(0.25, 0.7, sunElevation));
    radiance += warmHorizon * horizon * 0.35 * (1.0 - sky.rain * 0.7);
}
"#;

    let post_grade_vsl = r#"#include <vulkanshaders/v1/common.vsl>

vec3 aces_approx(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// @hook(stage = "post_grade", priority = 100)
void starter_post_grade(inout vec3 hdrColor, in vec2 uv, in VsPostContext post) {
    vec3 graded = hdrColor * post.exposure * VS_PARAM_EXPOSURE;
    if (VS_SPEC_ENABLE_ACES_GRADE) {
        graded = aces_approx(graded);
    }
    float luma = dot(graded, vec3(0.2126, 0.7152, 0.0722));
    hdrColor = mix(vec3(luma), graded, VS_PARAM_SATURATION);
}
"#;

    let grade_comp = r#"#version 460
#include <vulkanshaders/v1/common.vsl>

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 1, rgba16f) uniform image2D outHdrColor;

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(outHdrColor);
    if (px.x >= size.x || px.y >= size.y) {
        return;
    }
    vec2 uv = (vec2(px) + 0.5) / vec2(size);
    vec3 hdr = imageLoad(outHdrColor, px).rgb;
    VsPostContext post = VsPostContext(1.0, 0.15, 0u);
    vs_dispatch_hook_post_grade(hdr, uv, post);
    imageStore(outHdrColor, px, vec4(hdr, 1.0));
}
"#;

    let readme = format!(
        r#"# {pack_name}

Starter Hyaloclast (`.vsp`) shader pack (`{pack_id}`).

## Structure
- `pack.toml` — Pack metadata, tier capabilities, `[hooks]`, `[settings.*]`, and `[presets.*]` (`Low`, `Medium`, `High`, `Ultra`).
- `hooks/surface.vsl` — `@hook(stage = "surface")` PBR rain/roughness adjustment.
- `hooks/sky.vsl` — `@hook(stage = "sky")` horizon sunset tint.
- `hooks/post_grade.vsl` — `@hook(stage = "post_grade")` exposure, ACES curve, and saturation grading.
- `shaders/compute/grade_preview.comp` — Minimal GLSL 460 compute stage demonstrating `#include <vulkanshaders/v1/common.vsl>` and `vs_dispatch_hook_post_grade`.

## CLI Commands
```bash
vsp validate {dir}
vsp profile {dir} --tier 0
vsp compile {dir} --out-dir ./build-spv --tier 0
```
"#,
        dir = output_dir.display()
    );

    fs::write(&pack_toml_path, pack_toml)?;
    fs::write(output_dir.join("hooks/surface.vsl"), surface_vsl)?;
    fs::write(output_dir.join("hooks/sky.vsl"), sky_vsl)?;
    fs::write(output_dir.join("hooks/post_grade.vsl"), post_grade_vsl)?;
    fs::write(
        output_dir.join("shaders/compute/grade_preview.comp"),
        grade_comp,
    )?;
    fs::write(output_dir.join("README.md"), readme)?;

    println!(
        "[vsp] Initialized starter shader pack '{}' ({}) in '{}'",
        pack_id,
        pack_name,
        output_dir.display()
    );
    Ok(())
}

fn build_migration_guide(source: Option<MigrationSource>) -> String {
    let iris_section = r#"# Migrating Iris / OptiFine Shader Packs to Hyaloclast (.vsp)

## 1. File & Pass Mapping
| Iris / OptiFine | Hyaloclast (.vsp) | Notes |
| :--- | :--- | :--- |
| `shaders.properties` | `pack.toml` (`[settings.*]`, `[presets.*]`, `[pipeline]`) | Runtime sliders (`runtime = true`) map to `VS_PARAM_*` in `WorldUBO.packParams`; compile-time `#define` toggles (`runtime = false`) map to `VS_SPEC_*` Vulkan specialization constants. |
| `gbuffers_terrain.{vsh,fsh}` | `hooks/vertex_displace.vsl` + `hooks/surface.vsl` + `hooks/gbuffer_pack.vsl` | Use `@hook(stage = "vertex_displace")`, `@hook(stage = "surface")`, and `@hook(stage = "gbuffer_pack")`. |
| `gbuffers_water.{vsh,fsh}` | `hooks/water_waves.vsl` + `hooks/water_surface.vsl` | Hook wave parameters via `water_waves` and surface shading via `water_surface`. |
| `shadow.{vsh,fsh}` | Tier 0 CSM / Tier 2 `rayQueryEXT` / Tier 3 `pathtrace.rgen` | Use `vsTraceShadow()` from `<vulkanshaders/v1/lighting.vsl>` for tier-agnostic visibility. |
| `deferred*.fsh` / `composite*.fsh` | `[[passes]]` (`kind = "compute"`) + `hooks/direct_light.vsl`, `hooks/indirect_gi.vsl`, `hooks/fog.vsl` | Declare custom targets in `[targets.*]` and splice compute passes via `before`/`after`/`replaces`. |
| `final.fsh` | `hooks/post_grade.vsl` | Tone mapping, exposure, color grading, and LUT lookup (`vsSamplePackTexture_<name>`). |
| `colortex0..15` | `[targets.<name>]` | Named typed targets (`RGBA16F`, `R11G11B10F`, `temporal = true` for ping-pong history). |

## 2. Uniform & Coordinate Checklist
- [ ] Replace `uniform mat4 gbufferModelViewInverse` / `gbufferProjectionInverse` with `vsWorld.invView` and `vsWorld.invProj` (`#include <vulkanshaders/v1/common.vsl>`).
- [ ] Replace `frameTimeCounter` -> `vsWorld.gameTimeSec`, `rainStrength` -> `vsWorld.rain`, `thunderStrength` -> `vsWorld.thunder`, `isEyeInWater` -> `vsWorld.eyeInFluid`.
- [ ] Convert `#define` option sliders to `[settings.<key>]` in `pack.toml` and reference `VS_PARAM_<KEY>` (hot-reload, 0 ms) or `VS_SPEC_<KEY>` (specialization constant).
- [ ] Port `profile.LOW/MEDIUM/HIGH/ULTRA` from `shaders.properties` to `[presets.low]`, `[presets.medium]`, `[presets.high]`, `[presets.ultra]`.
- [ ] Convert LabPBR `_s` / `_n` channel decoding into `hooks/surface.vsl` (Hyaloclast decodes LabPBR 1.3 into `VsSurface.{roughness, metallic, f0, emission, subsurface, ao}` automatically when `capabilities.pbr_labpbr = true`).
- [ ] Run `vsp validate <pack_dir> --all-tiers` and `vsp profile <pack_dir> --tier 0` to verify SPIR-V 1.6 compliance.
"#;

    let bedrock_section = r#"# Migrating Bedrock RTX Packs to Hyaloclast (.vsp)

## 1. Material & Texture Set Mapping (`*.texture_set.json` -> LabPBR / `VsSurface`)
| Bedrock RTX (`texture_set.json`) | Hyaloclast LabPBR / `VsSurface` | Conversion Notes |
| :--- | :--- | :--- |
| `color` (albedo RGBA) | Base block atlas texture (`s.albedo`) | Linearized automatically in G-Buffer / RT hit shaders. |
| `metalness_emissive_roughness` (MER) `.r` (Metalness `0..255`) | LabPBR `_s.g` (`230..255` = metals) -> `s.metallic` | Remap Bedrock linear metalness `0..1` into LabPBR F0/metalness range or override in `hooks/surface.vsl`. |
| MER `.g` (Emissive `0..255`) | LabPBR `_s.a` (`0..254`) -> `s.emission` | Note: LabPBR `_s.a == 255` means zero emission; map Bedrock `0` -> `255` and `1..255` -> `1..254`. |
| MER `.b` (Roughness `0..255`) | LabPBR `_s.r` (Smoothness = `1.0 - perceptualRoughness`) -> `s.roughness` | Invert channel: `smoothness = 1.0 - roughness`. |
| `normal` / `heightmap` | LabPBR `_n.rg` (XY normal) + `_n.b` (AO) + `_n.a` (height) | Ensure OpenGL +Y normal orientation (`green` channel up). |

## 2. Atmosphere, Fog & Lighting Mapping
| Bedrock RTX JSON | Hyaloclast (.vsp) | Notes |
| :--- | :--- | :--- |
| `fogs/*.json` (`distance`, `volumetric`) | `hooks/fog.vsl` + `hooks/volumetric_media.vsl` | Implement `@hook(stage = "fog")` and `@hook(stage = "volumetric_media")`. |
| `lighting/global.json` (sun/moon/orbital colors) | `hooks/sky.vsl` + `hooks/direct_light.vsl` | Use `sky.sunDir`, `sky.rain`, and `vsWorld.gameTimeSec`. |
| Emissive block definitions | `hooks/emissive_sample.vsl` + `hooks/block_light.vsl` | Hyaloclast feeds emissive triangles into ReSTIR DI automatically in Tiers 2 & 3. |

## 3. Porting Checklist
- [ ] Run `vsp init ./my-rtx-pack` to scaffold `pack.toml`, presets, and starter `.vsl` hooks.
- [ ] Convert MER textures to LabPBR `_s.png` (`R = 1 - roughness`, `G = F0/metal`, `B = porosity/SSS`, `A = emissive`).
- [ ] Enable `pbr_labpbr = true` and `restir_di = true` in `[capabilities]` of `pack.toml`.
- [ ] Validate RT compilation with `vsp validate <pack_dir> --tier 3 --ser --omm`.
"#;

    match source {
        Some(MigrationSource::Iris) => iris_section.to_owned(),
        Some(MigrationSource::BedrockRtx) => bedrock_section.to_owned(),
        None => format!("{iris_section}\n{bedrock_section}"),
    }
}

fn run_migrate_help(
    from: Option<MigrationSource>,
    out: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let guide = build_migration_guide(from);
    if let Some(out_path) = out {
        if let Some(parent) = out_path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(out_path, &guide)?;
        println!("[vsp] Wrote migration guide to '{}'", out_path.display());
    } else {
        print!("{guide}");
    }
    Ok(())
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let res = match cli.command {
        Commands::Check { pack_dir, tier } => run_check_or_compile(&pack_dir, None, tier),
        Commands::Compile {
            pack_dir,
            out_dir,
            tier,
        } => run_check_or_compile(&pack_dir, Some(&out_dir), tier),
        Commands::Validate {
            pack_path,
            tier,
            all_tiers,
            ser,
            omm,
        } => run_validate(&pack_path, tier, all_tiers, ser, omm),
        Commands::Profile {
            pack_path,
            tier,
            ser,
            omm,
        } => run_profile(&pack_path, tier, ser, omm),
        Commands::Init {
            output_dir,
            name,
            id,
            author,
            force,
        } => run_init(&output_dir, name, id, author, force),
        Commands::MigrateHelp { from, out } => run_migrate_help(from, out.as_deref()),
    };

    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("[vsp] error: {err}");
            ExitCode::FAILURE
        }
    }
}
