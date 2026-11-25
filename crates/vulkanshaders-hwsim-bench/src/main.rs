// SPDX-License-Identifier: GPL-3.0-only
// Additional permissions under GPLv3 Section 7 apply to optional proprietary vendor SDK
// runtime loading; see LICENSE-EXCEPTION.md in the repository root.

use clap::{Parser, Subcommand, ValueEnum};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::str::FromStr;
use tracing_subscriber::EnvFilter;
use vulkanshaders_hwsim_bench::abi_layout::run_abi_audit;
use vulkanshaders_hwsim_bench::bench_fuzz::{run_fuzz_suites, run_micro_benchmarks};
use vulkanshaders_hwsim_bench::fsr4_sim::run_profile_simulation;
use vulkanshaders_hwsim_bench::profiles::{DriverModel, GpuId, GpuProfile};
use vulkanshaders_hwsim_bench::report::{EpistemicStatus, HwsimReport};
use vulkanshaders_hwsim_bench::spirv_audit::{
    audit_module_against_profile, build_synthetic_fp8_coop_spirv, parse_spirv_words,
    scan_binary_file,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputFormat {
    Markdown,
    Json,
}

#[derive(Parser, Debug)]
#[command(
    name = "vulkanshaders-hwsim-bench",
    about = "Hardware-free Vulkan ICD simulator, AMD FSR 4.1.2 contract verifier, cross-ABI auditor, and SPIR-V scanner"
)]
struct Cli {
    /// Output format (markdown or json)
    #[arg(long, global = true, value_enum, default_value_t = OutputFormat::Markdown)]
    format: OutputFormat,

    /// Optional output file path (defaults to stdout)
    #[arg(long, global = true)]
    out: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run Mock Vulkan ICD + FSR 4.1.2 / 3.1.6 contract simulation
    Simulate {
        /// Target GPU slug (`rx9070xt`, `rx7900xtx`, `rx6800xt`, `strix_point`, `rtx5090`, `rtx4090`, `rtx3050m`, `arc_b580`, or `all`)
        #[arg(long, default_value = "all")]
        gpu: String,

        /// Target OS driver model (`linux`, `windows`, or `all`)
        #[arg(long, default_value = "all")]
        os: String,

        /// Optional FSR 4 network override (`fp8`, `i8`, `navi2`, `lite`, `316`)
        #[arg(long)]
        fsr_override: Option<String>,
    },

    /// Print the 8-GPU x 2-OS capability, FSR 4.1.2 network, and upscaler fallback matrix
    Matrix {
        /// Filter OS driver model (`linux`, `windows`, or `all`)
        #[arg(long, default_value = "all")]
        os: String,
    },

    /// Audit standalone .spv files or carve embedded SPIR-V from .dll/.so binaries
    AuditSpirv {
        /// Paths to .spv, .dll, or .so files (if omitted, runs synthetic FP8 cooperative-matrix shader self-audit)
        paths: Vec<PathBuf>,

        /// Target GPU slug or `all`
        #[arg(long, default_value = "all")]
        gpu: String,

        /// Target OS driver model or `all`
        #[arg(long, default_value = "all")]
        os: String,
    },

    /// Run compile-time offset_of! + SysV64/Win64 cross-ABI layout & SDK 2.4.0 constant audit
    AbiCheck {
        /// Fail if SysV64 LP64 and Win64 LLP64 struct layouts diverge
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        strict_cross_abi: bool,
    },

    /// Run deterministic CPU-side micro-benchmarks (p50/p95/p99/MAD)
    Bench {
        #[arg(long, default_value_t = 200)]
        warmup: usize,

        #[arg(long, default_value_t = 4000)]
        iters: usize,
    },

    /// Run deterministic BLAKE3-XOF property fuzz suites
    Fuzz {
        #[arg(long, default_value_t = 0x4859_414C_4F43_4C34)]
        seed: u64,

        #[arg(long, default_value_t = 256)]
        cases: usize,

        /// Suite filter (`all`, `pnext`, `layout`, `fsr`, `spirv`)
        #[arg(long, default_value = "all")]
        suite: String,
    },

    /// Run the complete hardware-free verification pipeline (matrix + simulate + abi-check + synthetic spirv + fuzz + bench)
    Full {
        #[arg(long, default_value_t = 0x4859_414C_4F43_4C34)]
        seed: u64,

        #[arg(long, default_value_t = 128)]
        fuzz_cases: usize,

        #[arg(long, default_value_t = 2000)]
        bench_iters: usize,
    },
}

fn resolve_profiles(gpu: &str, os: &str) -> Result<Vec<GpuProfile>, String> {
    let gpus: Vec<GpuId> = if gpu.eq_ignore_ascii_case("all") {
        GpuId::ALL.to_vec()
    } else {
        vec![GpuId::from_str(gpu).map_err(|e| e.to_string())?]
    };

    let drivers: Vec<DriverModel> = if os.eq_ignore_ascii_case("all") {
        DriverModel::ALL.to_vec()
    } else {
        vec![DriverModel::from_str(os).map_err(|e| e.to_string())?]
    };

    let mut out = Vec::with_capacity(gpus.len() * drivers.len());
    for g in gpus {
        for &d in &drivers {
            out.push(GpuProfile::build(g, d));
        }
    }
    Ok(out)
}

fn emit_report(report: &mut HwsimReport, format: OutputFormat, out: Option<PathBuf>) -> ExitCode {
    report.finalize_digest();
    let rendered = match format {
        OutputFormat::Markdown => report.to_markdown(),
        OutputFormat::Json => match serde_json::to_string_pretty(report) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: failed to serialize JSON report: {e}");
                return ExitCode::from(1);
            }
        },
    };

    if let Some(path) = out {
        if let Err(e) = fs::write(&path, rendered) {
            eprintln!("error: failed to write {}: {e}", path.display());
            return ExitCode::from(1);
        }
    } else {
        println!("{rendered}");
    }

    if report.has_failures() {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

fn main() -> ExitCode {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_ansi(false)
        .try_init();

    let cli = Cli::parse();
    let mut report = HwsimReport::new();

    match cli.command {
        Commands::Simulate {
            gpu,
            os,
            fsr_override,
        } => {
            let profiles = match resolve_profiles(&gpu, &os) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            for p in &profiles {
                report
                    .simulations
                    .push(run_profile_simulation(p, fsr_override.as_deref()));
            }
            report.epistemic_summary.insert(
                "mock_icd_fsr4_simulation".to_string(),
                EpistemicStatus::ProvenOffline,
            );
        }

        Commands::Matrix { os } => {
            let profiles = match resolve_profiles("all", &os) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            report.populate_matrix(&profiles);
        }

        Commands::AuditSpirv { paths, gpu, os } => {
            let profiles = match resolve_profiles(&gpu, &os) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };

            if paths.is_empty() {
                let spv = build_synthetic_fp8_coop_spirv();
                if let Ok(module) = parse_spirv_words("synthetic_fsr4_fp8.spv", &spv) {
                    let mut compat_map = std::collections::BTreeMap::new();
                    for p in &profiles {
                        let key = format!("{}_{}", p.id.slug(), p.driver.as_str());
                        compat_map.insert(key, audit_module_against_profile(&module, p));
                    }
                    report
                        .spirv_compat
                        .insert(module.source_label.clone(), compat_map);
                    report.epistemic_summary.insert(
                        "spirv_synthetic_fp8_audit".to_string(),
                        EpistemicStatus::StaticHeuristic,
                    );
                }
            } else {
                let mut any_inconclusive = false;
                for path in paths {
                    match scan_binary_file(&path) {
                        Ok(scan) => {
                            if scan.inconclusive_compressed {
                                any_inconclusive = true;
                            }
                            for module in &scan.carved_modules {
                                let mut compat_map = std::collections::BTreeMap::new();
                                for p in &profiles {
                                    let key = format!("{}_{}", p.id.slug(), p.driver.as_str());
                                    compat_map
                                        .insert(key, audit_module_against_profile(module, p));
                                }
                                report
                                    .spirv_compat
                                    .insert(module.source_label.clone(), compat_map);
                            }
                            report.spirv_scans.push(scan);
                        }
                        Err(e) => {
                            eprintln!("error: {e}");
                            return ExitCode::from(1);
                        }
                    }
                }
                report.epistemic_summary.insert(
                    "spirv_binary_audit".to_string(),
                    if any_inconclusive {
                        EpistemicStatus::InconclusiveUninspected
                    } else {
                        EpistemicStatus::StaticHeuristic
                    },
                );
            }
        }

        Commands::AbiCheck { strict_cross_abi } => {
            report.abi_audit = Some(run_abi_audit(strict_cross_abi));
            report.epistemic_summary.insert(
                "cross_abi_layout_and_constants".to_string(),
                EpistemicStatus::ProvenOffline,
            );
        }

        Commands::Bench { warmup, iters } => {
            report.benchmarks = run_micro_benchmarks(warmup, iters);
            report.epistemic_summary.insert(
                "cpu_side_micro_benchmarks".to_string(),
                EpistemicStatus::ProvenOffline,
            );
        }

        Commands::Fuzz { seed, cases, suite } => {
            report.fuzz = Some(run_fuzz_suites(seed, cases, &suite));
            report.epistemic_summary.insert(
                "deterministic_property_fuzz".to_string(),
                EpistemicStatus::ProvenOffline,
            );
        }

        Commands::Full {
            seed,
            fuzz_cases,
            bench_iters,
        } => {
            let profiles = GpuProfile::all_matrix();
            report.populate_matrix(&profiles);
            for p in &profiles {
                report.simulations.push(run_profile_simulation(p, None));
            }
            report.abi_audit = Some(run_abi_audit(true));
            let spv = build_synthetic_fp8_coop_spirv();
            if let Ok(module) = parse_spirv_words("synthetic_fsr4_fp8.spv", &spv) {
                let mut compat_map = std::collections::BTreeMap::new();
                for p in &profiles {
                    let key = format!("{}_{}", p.id.slug(), p.driver.as_str());
                    compat_map.insert(key, audit_module_against_profile(&module, p));
                }
                report
                    .spirv_compat
                    .insert(module.source_label.clone(), compat_map);
            }
            report.fuzz = Some(run_fuzz_suites(seed, fuzz_cases, "all"));
            report.benchmarks = run_micro_benchmarks(100, bench_iters);

            report.epistemic_summary.insert(
                "mock_icd_fsr4_simulation".to_string(),
                EpistemicStatus::ProvenOffline,
            );
            report.epistemic_summary.insert(
                "cross_abi_layout_and_constants".to_string(),
                EpistemicStatus::ProvenOffline,
            );
            report.epistemic_summary.insert(
                "spirv_synthetic_fp8_audit".to_string(),
                EpistemicStatus::StaticHeuristic,
            );
            report.epistemic_summary.insert(
                "deterministic_property_fuzz".to_string(),
                EpistemicStatus::ProvenOffline,
            );
            report.epistemic_summary.insert(
                "physical_silicon_validation".to_string(),
                EpistemicStatus::RequiresPhysicalHardware,
            );
        }
    }

    emit_report(&mut report, cli.format, cli.out)
}
