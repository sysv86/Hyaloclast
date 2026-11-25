// SPDX-License-Identifier: GPL-3.0-only
// Additional permissions under GPLv3 Section 7 apply to optional proprietary vendor SDK
// runtime loading; see LICENSE-EXCEPTION.md in the repository root.

//! Hardware-free Vulkan ICD simulator, AMD FSR 4.1.2 / FidelityFX SDK 2.4.0 contract verifier,
//! cross-ABI layout auditor, SPIR-V capability scanner, and deterministic benchmark harness.

pub mod abi_layout;
pub mod bench_fuzz;
pub mod error;
pub mod fsr4_sim;
pub mod mock_icd;
pub mod profiles;
pub mod report;
pub mod spirv_audit;

pub use abi_layout::{AbiAuditReport, AbiModel, run_abi_audit};
pub use bench_fuzz::{FuzzReport, MicroBenchResult, run_fuzz_suites, run_micro_benchmarks};
pub use error::HwsimError;
pub use fsr4_sim::{MockFfxProvider, SimulationResult, run_profile_simulation};
pub use mock_icd::{LeakReport, MockIcdGuard, MockIcdState};
pub use profiles::{DriverModel, Fsr4Network, GpuId, GpuProfile};
pub use report::{EpistemicStatus, HwsimReport};
pub use spirv_audit::{
    BinaryScanResult, SpirvModuleAudit, SpirvProfileCompat, audit_module_against_profile,
    audit_module_matrix, build_synthetic_fp8_coop_spirv, parse_spirv_words, scan_binary_bytes,
    scan_binary_file,
};
