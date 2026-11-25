// SPDX-License-Identifier: GPL-3.0-only
// Additional permissions under GPLv3 Section 7 apply to optional proprietary vendor SDK
// runtime loading; see LICENSE-EXCEPTION.md in the repository root.

//! Deterministic JSON and Markdown report generator with 4-state epistemic status tagging.

use crate::abi_layout::AbiAuditReport;
use crate::bench_fuzz::{FuzzReport, MicroBenchResult};
use crate::fsr4_sim::SimulationResult;
use crate::profiles::GpuProfile;
use crate::spirv_audit::{BinaryScanResult, SpirvProfileCompat};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// Four-state epistemic status preventing offline mock proofs from being confused with physical GPU validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EpistemicStatus {
    /// Mathematically or deterministically verified without GPU hardware.
    ProvenOffline,
    /// Verified against modeled hardware profiles or uncompressed static bytecode.
    StaticHeuristic,
    /// Target payload could not be inspected (e.g., compressed shader blobs inside a vendor DLL).
    InconclusiveUninspected,
    /// Cannot be proven without physical silicon and vendor drivers.
    RequiresPhysicalHardware,
}

impl std::fmt::Display for EpistemicStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ProvenOffline => f.write_str("PROVEN_OFFLINE"),
            Self::StaticHeuristic => f.write_str("STATIC_HEURISTIC"),
            Self::InconclusiveUninspected => f.write_str("INCONCLUSIVE_UNINSPECTED"),
            Self::RequiresPhysicalHardware => f.write_str("REQUIRES_PHYSICAL_HARDWARE"),
        }
    }
}

/// One row in the 8-GPU x 2-OS capability & FSR 4.1.2 matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixRow {
    pub gpu_id: String,
    pub display_name: String,
    pub driver_model: String,
    pub pci_id: String,
    pub driver_name: String,
    pub fp8_coop_16x16x16: bool,
    pub fsr4_network: String,
    pub recommended_upscaler: String,
}

/// Top-level deterministic verification & benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HwsimReport {
    pub tool: String,
    pub epistemic_summary: BTreeMap<String, EpistemicStatus>,
    pub matrix: Vec<MatrixRow>,
    pub simulations: Vec<SimulationResult>,
    pub abi_audit: Option<AbiAuditReport>,
    pub spirv_scans: Vec<BinaryScanResult>,
    pub spirv_compat: BTreeMap<String, BTreeMap<String, SpirvProfileCompat>>,
    pub fuzz: Option<FuzzReport>,
    pub benchmarks: Vec<MicroBenchResult>,
    pub requires_physical_hardware_checklist: Vec<String>,
    pub content_blake3: String,
}

impl HwsimReport {
    pub fn new() -> Self {
        Self {
            tool: "vulkanshaders-hwsim-bench".to_string(),
            epistemic_summary: BTreeMap::new(),
            matrix: Vec::new(),
            simulations: Vec::new(),
            abi_audit: None,
            spirv_scans: Vec::new(),
            spirv_compat: BTreeMap::new(),
            fuzz: None,
            benchmarks: Vec::new(),
            requires_physical_hardware_checklist: vec![
                "Driver SPIR-V-to-ISA codegen, VGPR register pressure, and wave32/wave64 subgroup divergence on RDNA 2/3/4 and Ada/Blackwell".to_string(),
                "Real GPU L2/TCC cache coherency and cross-queue semaphore timing".to_string(),
                "Actual FP8/INT8 neural network visual output quality (ghosting, disocclusion flicker, NaN propagation)".to_string(),
                "Real GPU wall-clock frame time, VRAM bandwidth saturation, and thermal throttling".to_string(),
            ],
            content_blake3: String::new(),
        }
    }

    /// Populates the deterministic 8-GPU x 2-OS hardware capability matrix.
    pub fn populate_matrix(&mut self, profiles: &[GpuProfile]) {
        self.matrix = profiles
            .iter()
            .map(|p| MatrixRow {
                gpu_id: p.id.slug().to_string(),
                display_name: p.display_name.to_string(),
                driver_model: p.driver.as_str().to_string(),
                pci_id: format!("{:04X}:{:04X}", p.asic.vendor_id, p.asic.device_id),
                driver_name: p.driver_name.to_string(),
                fp8_coop_16x16x16: p.supports_fp8_coop_16x16x16(),
                fsr4_network: p
                    .select_fsr4_network(None)
                    .map(|n| n.amd_name().to_string())
                    .unwrap_or_else(|_| "error".to_string()),
                recommended_upscaler: format!("{:?}", p.recommended_upscaler()),
            })
            .collect();
        self.epistemic_summary
            .insert("gpu_capability_matrix".to_string(), EpistemicStatus::StaticHeuristic);
    }

    /// Computes a stable BLAKE3 digest over the canonical JSON payload.
    pub fn finalize_digest(&mut self) {
        self.content_blake3.clear();
        if let Ok(raw) = serde_json::to_string(self) {
            self.content_blake3 = blake3::hash(raw.as_bytes()).to_hex().to_string();
        }
    }

    /// Returns true if any offline check failed.
    pub fn has_failures(&self) -> bool {
        let sim_fail = self.simulations.iter().any(|s| !s.passed);
        let abi_fail = self
            .abi_audit
            .as_ref()
            .is_some_and(|a| !a.all_structs_passed || !a.all_constants_passed);
        let fuzz_fail = self.fuzz.as_ref().is_some_and(|f| !f.all_passed);
        sim_fail || abi_fail || fuzz_fail
    }

    /// Renders a clean, deterministic Markdown report.
    pub fn to_markdown(&self) -> String {
        let mut md = String::with_capacity(4096);
        let _ = writeln!(md, "# VulkanShaders Hardware-Free Simulation & Verification Report");
        let _ = writeln!(md);
        let _ = writeln!(md, "- **Tool**: `{}`", self.tool);
        let _ = writeln!(md, "- **Content BLAKE3**: `{}`", self.content_blake3);
        let _ = writeln!(
            md,
            "- **Offline Verdict**: {}",
            if self.has_failures() { "FAIL" } else { "PASS" }
        );
        let _ = writeln!(md);

        if !self.epistemic_summary.is_empty() {
            let _ = writeln!(md, "## Epistemic Status Partition");
            let _ = writeln!(md, "| Subsystem | Epistemic Status |");
            let _ = writeln!(md, "|---|---|");
            for (k, v) in &self.epistemic_summary {
                let _ = writeln!(md, "| `{k}` | `{v}` |");
            }
            let _ = writeln!(md);
        }

        if !self.matrix.is_empty() {
            let _ = writeln!(md, "## Hardware Profile & FSR 4.1.2 Matrix");
            let _ = writeln!(
                md,
                "| GPU | OS | PCI ID | FP8 16x16x16 | FSR 4.1.2 Network | Primary Upscaler |"
            );
            let _ = writeln!(md, "|---|---|---|---|---|---|");
            for row in &self.matrix {
                let _ = writeln!(
                    md,
                    "| {} | {} | `{}` | {} | `{}` | `{}` |",
                    row.display_name,
                    row.driver_model,
                    row.pci_id,
                    if row.fp8_coop_16x16x16 { "yes" } else { "no" },
                    row.fsr4_network,
                    row.recommended_upscaler
                );
            }
            let _ = writeln!(md);
        }

        if !self.simulations.is_empty() {
            let _ = writeln!(md, "## Mock Vulkan ICD + FSR 4.1.2 Contract Simulation");
            let _ = writeln!(
                md,
                "| GPU | OS | Network | Dispatches | 15-Frame Cap | Xbox-ID Rejected | Layout Check | Leaks | Result |"
            );
            let _ = writeln!(md, "|---|---|---|---|---|---|---|---|---|");
            for sim in &self.simulations {
                let _ = writeln!(
                    md,
                    "| `{}` | `{}` | `{}` | {} | {} | {} | {} | {} | {} |",
                    sim.gpu_id,
                    sim.driver_model,
                    sim.selected_network,
                    sim.dispatches_succeeded,
                    sim.in_flight_cap_enforced,
                    sim.xbox_backend_bug_rejected,
                    sim.layout_violation_caught,
                    if sim.leak_free { "0" } else { "LEAK" },
                    if sim.passed { "PASS" } else { "FAIL" }
                );
            }
            let _ = writeln!(md);
        }

        if let Some(abi) = &self.abi_audit {
            let _ = writeln!(md, "## Cross-ABI Struct & SDK 2.4.0 Constant Audit");
            let _ = writeln!(md, "| Struct | SDK | Host Size | SysV64 | Win64 | Status |");
            let _ = writeln!(md, "|---|---|---|---|---|---|");
            for s in &abi.structs {
                let _ = writeln!(
                    md,
                    "| `{}` | {} | {} | {} | {} | {} |",
                    s.struct_name,
                    s.sdk,
                    s.host_size,
                    s.sysv_size,
                    s.win64_size,
                    if s.passed { "PASS" } else { "FAIL" }
                );
            }
            let _ = writeln!(md);
            let _ = writeln!(md, "| Constant | Actual | Expected | Status | Detail |");
            let _ = writeln!(md, "|---|---|---|---|---|");
            for c in &abi.constants {
                let _ = writeln!(
                    md,
                    "| `{}` | `{}` | `{}` | {} | {} |",
                    c.name,
                    c.actual_hex,
                    c.expected_hex,
                    if c.passed { "PASS" } else { "FAIL" },
                    c.detail
                );
            }
            let _ = writeln!(md);
        }

        if !self.spirv_scans.is_empty() {
            let _ = writeln!(md, "## SPIR-V & Vendor Binary Scan");
            let _ = writeln!(
                md,
                "| File | Format | Carved SPIR-V | Compressed Hits | Status |"
            );
            let _ = writeln!(md, "|---|---|---|---|---|");
            for scan in &self.spirv_scans {
                let status = if scan.inconclusive_compressed {
                    EpistemicStatus::InconclusiveUninspected.to_string()
                } else {
                    EpistemicStatus::StaticHeuristic.to_string()
                };
                let _ = writeln!(
                    md,
                    "| `{}` | {} | {} | {} | `{}` |",
                    scan.path,
                    scan.container_format,
                    scan.carved_modules.len(),
                    scan.compressed_hits.len(),
                    status
                );
            }
            let _ = writeln!(md);
        }

        if !self.spirv_compat.is_empty() {
            let _ = writeln!(md, "## SPIR-V Hardware Profile Compatibility");
            let _ = writeln!(md, "| Module | Profile | Compatible | Blockers |");
            let _ = writeln!(md, "|---|---|---|---|");
            for (module_name, compat_map) in &self.spirv_compat {
                for (profile_key, compat) in compat_map {
                    let blockers = if compat.blockers.is_empty() {
                        "-".to_string()
                    } else {
                        compat.blockers.join("; ")
                    };
                    let _ = writeln!(
                        md,
                        "| `{module_name}` | `{profile_key}` | {} | {blockers} |",
                        if compat.compatible { "yes" } else { "no" }
                    );
                }
            }
            let _ = writeln!(md);
        }

        if let Some(fuzz) = &self.fuzz {
            let _ = writeln!(md, "## Deterministic Property Fuzzing (seed `{:#x}`)", fuzz.seed);
            let _ = writeln!(md, "| Suite | Cases | Status |");
            let _ = writeln!(md, "|---|---|---|");
            for s in &fuzz.suites {
                let _ = writeln!(
                    md,
                    "| `{}` | {} | {} |",
                    s.suite,
                    s.cases_run,
                    if s.passed { "PASS" } else { "FAIL" }
                );
            }
            let _ = writeln!(md);
        }

        if !self.benchmarks.is_empty() {
            let _ = writeln!(md, "## CPU-Side Verification Micro-Benchmarks (ns/op)");
            let _ = writeln!(md, "| Benchmark | Iters | p50 (ns) | p95 (ns) | p99 (ns) | MAD (ns) |");
            let _ = writeln!(md, "|---|---|---|---|---|---|");
            for b in &self.benchmarks {
                let _ = writeln!(
                    md,
                    "| `{}` | {} | {:.1} | {:.1} | {:.1} | {:.1} |",
                    b.name, b.iterations, b.p50_ns, b.p95_ns, b.p99_ns, b.mad_ns
                );
            }
            let _ = writeln!(md);
        }

        let _ = writeln!(md, "## Still Requires Physical Hardware (`REQUIRES_PHYSICAL_HARDWARE`)");
        for item in &self.requires_physical_hardware_checklist {
            let _ = writeln!(md, "- {item}");
        }

        md
    }
}

impl Default for HwsimReport {
    fn default() -> Self {
        Self::new()
    }
}
