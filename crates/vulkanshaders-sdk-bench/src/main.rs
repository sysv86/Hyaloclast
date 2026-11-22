use clap::{Parser, Subcommand, ValueEnum};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use tracing_subscriber::EnvFilter;
use vulkanshaders_sdk_bench::binary_parser::{ParsedBinary, parse_binary_file};
use vulkanshaders_sdk_bench::downloader::{DownloadConfig, SdkKind, fetch_sdk_artifacts};
use vulkanshaders_sdk_bench::header_parser::{ParsedHeader, parse_c_header_file};
use vulkanshaders_sdk_bench::source_verifier::{BenchReport, VerifyOptions, run_all_suites};

#[derive(Parser, Debug)]
#[command(
    name = "vulkanshaders-sdk-bench",
    about = "Cross-verifier, downloader, and binary inspector for VulkanShaders vendor SDKs & FFI contracts"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SdkSelection {
    All,
    Ngx,
    Nrd,
    Fsr3,
    Xess,
}

impl SdkSelection {
    fn into_kinds(self) -> Vec<SdkKind> {
        match self {
            Self::All => vec![SdkKind::Ngx, SdkKind::Nrd, SdkKind::Fsr3, SdkKind::Xess],
            Self::Ngx => vec![SdkKind::Ngx],
            Self::Nrd => vec![SdkKind::Nrd],
            Self::Fsr3 => vec![SdkKind::Fsr3],
            Self::Xess => vec![SdkKind::Xess],
        }
    }
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run all 7 verification suites (SDK binaries, C headers, repr(C) layouts, FFI<->FFM, CAP_*, shaders, mixins)
    Verify {
        /// Path to the VulkanShaders repository root (defaults to current directory or workspace root)
        #[arg(long)]
        repo_root: Option<PathBuf>,

        /// Download official vendor SDK binaries and headers before verifying
        #[arg(long, default_value_t = false)]
        download: bool,

        /// Which SDKs to verify/download
        #[arg(long, value_enum, default_value_t = SdkSelection::All)]
        sdk: SdkSelection,

        /// Directory for downloaded SDK artifacts
        #[arg(long)]
        cache_dir: Option<PathBuf>,

        /// Optional directory of local staged binaries (.so/.dll/.a) to inspect
        #[arg(long)]
        natives_dir: Option<PathBuf>,

        /// Optional directory of local vendor C headers (.h) to inspect
        #[arg(long)]
        headers_dir: Option<PathBuf>,

        /// Write full structured JSON report to this file path
        #[arg(long)]
        json: Option<PathBuf>,
    },

    /// Download official vendor SDK binaries & headers and print BLAKE3 checksums
    Download {
        /// Which SDKs to download
        #[arg(long, value_enum, default_value_t = SdkSelection::All)]
        sdk: SdkSelection,

        /// Target cache directory
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },

    /// Parse any .so, .dll, or .a file and print format, architecture, BLAKE3, dependencies, and exports
    InspectBinary {
        /// Path to binary file (.so, .dll, .a)
        path: PathBuf,

        /// Optional substring filter for exported symbols
        #[arg(long)]
        filter: Option<String>,
    },
}

fn default_repo_root() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if cwd.join("crates/vulkanshaders-core").is_dir() {
        return cwd;
    }
    let manifest_parent = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    if manifest_parent.join("crates/vulkanshaders-core").is_dir() {
        return manifest_parent;
    }
    cwd
}

fn default_cache_dir(repo_root: &Path) -> PathBuf {
    repo_root.join("target/sdk-bench-cache")
}

fn classify_binary_sdk(path: &Path) -> Option<SdkKind> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let is_binary_file = name.ends_with(".dll")
        || name.ends_with(".so")
        || name.contains(".so.")
        || name.ends_with(".a");
    if !is_binary_file {
        return None;
    }
    if name.contains("ngx") || name.contains("dlss") {
        Some(SdkKind::Ngx)
    } else if name.contains("nrd") {
        Some(SdkKind::Nrd)
    } else if name.contains("ffx") || name.contains("fsr") || name.contains("fidelityfx") {
        Some(SdkKind::Fsr3)
    } else if name.contains("xess") {
        Some(SdkKind::Xess)
    } else {
        None
    }
}

#[cfg(test)]
mod verify_path_tests {
    use super::{SdkKind, classify_binary_sdk};
    use std::path::Path;

    #[test]
    fn binary_classifier_ignores_headers_and_partial_downloads() {
        assert_eq!(classify_binary_sdk(Path::new("NRDSettings.h")), None);
        assert_eq!(
            classify_binary_sdk(Path::new("amd_fidelityfx_vk.dll.part")),
            None
        );
    }

    #[test]
    fn binary_classifier_keeps_sdk_shared_and_static_libraries() {
        assert_eq!(
            classify_binary_sdk(Path::new("libnvidia-ngx-dlss.so.310.9.16")),
            Some(SdkKind::Ngx)
        );
        assert_eq!(
            classify_binary_sdk(Path::new("amd_fidelityfx_vk.dll")),
            Some(SdkKind::Fsr3)
        );
        assert_eq!(
            classify_binary_sdk(Path::new("xess.dll")),
            Some(SdkKind::Xess)
        );
    }
}

fn classify_header_sdk(path: &Path) -> Option<SdkKind> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if name.contains("ngx") {
        Some(SdkKind::Ngx)
    } else if name.contains("nrd") {
        Some(SdkKind::Nrd)
    } else if name.contains("ffx") || name.contains("fsr") {
        Some(SdkKind::Fsr3)
    } else if name.contains("xess") {
        Some(SdkKind::Xess)
    } else {
        None
    }
}

fn collect_local_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let p = entry.path();
        if p.is_dir() {
            collect_local_files(&p, out);
        } else if p.is_file() {
            out.push(p);
        }
    }
}

fn print_report(report: &BenchReport) {
    println!("VulkanShaders SDK Bench Verification Report");
    println!("Repository : {}", report.repo_root);
    println!(
        "Summary    : {} (PASS: {}, WARN: {}, FAIL: {})",
        report.overall_status, report.total_pass, report.total_warn, report.total_fail
    );
    println!();
    println!("{:<6} | {:<44} | {}", "STATUS", "CHECK", "DETAIL");
    println!("{:-<6}-+-{:-<44}-+-{:-<60}", "", "", "");

    for suite in &report.suites {
        println!("[{}] {}", suite.status, suite.suite);
        for check in &suite.checks {
            println!(
                "{:<6} | {:<44} | {}",
                check.status, check.name, check.detail
            );
        }
    }
}

fn run_verify(
    repo_root: Option<PathBuf>,
    download: bool,
    sdk: SdkSelection,
    cache_dir: Option<PathBuf>,
    natives_dir: Option<PathBuf>,
    headers_dir: Option<PathBuf>,
    json_out: Option<PathBuf>,
) -> ExitCode {
    let root = repo_root.unwrap_or_else(default_repo_root);
    let cache = cache_dir.unwrap_or_else(|| default_cache_dir(&root));

    let mut owned_binaries: BTreeMap<SdkKind, Vec<ParsedBinary>> = BTreeMap::new();
    let mut owned_headers: BTreeMap<SdkKind, Vec<(String, ParsedHeader)>> = BTreeMap::new();
    let mut download_errors: BTreeMap<SdkKind, Vec<String>> = BTreeMap::new();

    if download {
        let cfg = DownloadConfig {
            cache_dir: cache.clone(),
            force_download: false,
            offline: false,
            include_large_zips: false,
            local_natives_dirs: natives_dir.iter().cloned().collect(),
        };
        for kind in sdk.into_kinds() {
            let res = fetch_sdk_artifacts(kind, &cfg);
            if !res.errors.is_empty() {
                download_errors
                    .entry(res.sdk)
                    .or_default()
                    .extend(res.errors.clone());
            }
            for bin_art in &res.binaries {
                match parse_binary_file(&bin_art.local_path, true) {
                    Ok(parsed) => owned_binaries.entry(res.sdk).or_default().push(parsed),
                    Err(e) => download_errors.entry(res.sdk).or_default().push(format!(
                        "parse binary {}: {e}",
                        bin_art.local_path.display()
                    )),
                }
            }
            for hdr_art in &res.headers {
                match parse_c_header_file(&hdr_art.local_path) {
                    Ok(parsed) => {
                        let fname = hdr_art
                            .local_path
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("header.h")
                            .to_string();
                        owned_headers
                            .entry(res.sdk)
                            .or_default()
                            .push((fname, parsed));
                    }
                    Err(e) => download_errors.entry(res.sdk).or_default().push(format!(
                        "parse header {}: {e}",
                        hdr_art.local_path.display()
                    )),
                }
            }
        }
    }

    if let Some(dir) = natives_dir {
        let mut files = Vec::new();
        collect_local_files(&dir, &mut files);
        for p in files {
            if let Some(kind) = classify_binary_sdk(&p) {
                if let Ok(parsed) = parse_binary_file(&p, true) {
                    owned_binaries.entry(kind).or_default().push(parsed);
                }
            }
        }
    }

    if let Some(dir) = headers_dir {
        let mut files = Vec::new();
        collect_local_files(&dir, &mut files);
        for p in files {
            if p.extension().and_then(|e| e.to_str()) == Some("h") {
                if let Some(kind) = classify_header_sdk(&p) {
                    if let Ok(parsed) = parse_c_header_file(&p) {
                        let fname = p
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("header.h")
                            .to_string();
                        owned_headers.entry(kind).or_default().push((fname, parsed));
                    }
                }
            }
        }
    }

    let binaries_by_sdk: BTreeMap<SdkKind, Vec<&ParsedBinary>> = owned_binaries
        .iter()
        .map(|(k, v)| (*k, v.iter().collect()))
        .collect();
    let headers_by_sdk: BTreeMap<SdkKind, Vec<(String, &ParsedHeader)>> = owned_headers
        .iter()
        .map(|(k, v)| (*k, v.iter().map(|(n, h)| (n.clone(), h)).collect()))
        .collect();

    let opts = VerifyOptions {
        repo_root: root,
        binaries_by_sdk,
        headers_by_sdk,
        download_requested: download,
        download_errors,
    };

    let report = run_all_suites(&opts);
    print_report(&report);

    if let Some(json_path) = json_out {
        match serde_json::to_string_pretty(&report) {
            Ok(serialized) => {
                if let Err(e) = fs::write(&json_path, serialized) {
                    eprintln!(
                        "error: failed to write JSON report to {}: {e}",
                        json_path.display()
                    );
                    return ExitCode::from(1);
                }
            }
            Err(e) => {
                eprintln!("error: failed to serialize JSON report: {e}");
                return ExitCode::from(1);
            }
        }
    }

    if report.has_failures() {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

fn run_download(sdk: SdkSelection, cache_dir: Option<PathBuf>) -> ExitCode {
    let root = default_repo_root();
    let cache = cache_dir.unwrap_or_else(|| default_cache_dir(&root));
    let cfg = DownloadConfig {
        cache_dir: cache,
        force_download: false,
        offline: false,
        include_large_zips: false,
        local_natives_dirs: Vec::new(),
    };
    let results: Vec<_> = sdk
        .into_kinds()
        .into_iter()
        .map(|kind| fetch_sdk_artifacts(kind, &cfg))
        .collect();

    println!(
        "{:<8} | {:<8} | {:<64} | {}",
        "SDK", "KIND", "BLAKE3", "PATH"
    );
    println!("{:-<8}-+-{:-<8}-+-{:-<64}-+-{:-<40}", "", "", "", "");

    let mut had_error = false;
    for res in results {
        for art in &res.binaries {
            println!(
                "{:<8} | {:<8} | {:<64} | {}",
                format!("{:?}", res.sdk),
                "binary",
                art.blake3_hex,
                art.local_path.display()
            );
        }
        for art in &res.headers {
            println!(
                "{:<8} | {:<8} | {:<64} | {}",
                format!("{:?}", res.sdk),
                "header",
                art.blake3_hex,
                art.local_path.display()
            );
        }
        for err in &res.errors {
            had_error = true;
            eprintln!("error [{:?}]: {err}", res.sdk);
        }
    }

    if had_error {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

fn run_inspect_binary(path: &Path, filter: Option<&str>) -> ExitCode {
    let parsed = match parse_binary_file(path, true) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: failed to parse {}: {e}", path.display());
            return ExitCode::from(1);
        }
    };

    let bytes = fs::read(path).unwrap_or_default();
    let blake3_hex = blake3::hash(&bytes).to_hex().to_string();

    println!("File         : {}", path.display());
    println!("Format       : {:?}", parsed.format);
    println!("Architecture : {}", parsed.architecture);
    println!("BLAKE3       : {}", blake3_hex);
    println!("Dependencies : {}", parsed.dependencies.len());
    for dep in &parsed.dependencies {
        println!("  - {dep}");
    }

    let filtered: Vec<&String> = parsed
        .exported_symbols
        .iter()
        .filter(|sym| filter.is_none_or(|f| sym.contains(f)))
        .collect();

    println!(
        "Exports      : {} shown (total {})",
        filtered.len(),
        parsed.exported_symbols.len()
    );
    for sym in filtered {
        println!("  - {sym}");
    }

    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_ansi(false)
        .try_init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Verify {
            repo_root,
            download,
            sdk,
            cache_dir,
            natives_dir,
            headers_dir,
            json,
        } => run_verify(
            repo_root,
            download,
            sdk,
            cache_dir,
            natives_dir,
            headers_dir,
            json,
        ),
        Commands::Download { sdk, cache_dir } => run_download(sdk, cache_dir),
        Commands::InspectBinary { path, filter } => run_inspect_binary(&path, filter.as_deref()),
    }
}
