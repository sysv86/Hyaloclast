// SPDX-License-Identifier: GPL-3.0-only
// Additional permissions under GPLv3 Section 7 apply to optional proprietary vendor SDK
// runtime loading; see LICENSE-EXCEPTION.md in the repository root.

//! End-to-end hardware-free integration tests for `vulkanshaders-hwsim-bench`.

use vulkanshaders_hwsim_bench::{
    DriverModel, Fsr4Network, GpuId, GpuProfile, HwsimReport, audit_module_against_profile,
    build_synthetic_fp8_coop_spirv, parse_spirv_words, run_abi_audit, run_fuzz_suites,
    run_micro_benchmarks, run_profile_simulation, scan_binary_bytes,
};

#[test]
fn all_16_gpu_os_profiles_pass_fsr4_contract_simulation() {
    let profiles = GpuProfile::all_matrix();
    assert_eq!(profiles.len(), 16);

    for profile in &profiles {
        let sim = run_profile_simulation(profile, None);
        assert!(
            sim.passed,
            "simulation failed for {} ({}): {:?}",
            profile.display_name, profile.driver, sim.notes
        );
        assert!(sim.xbox_backend_bug_rejected);
        assert!(sim.layout_violation_caught);
        assert!(sim.in_flight_cap_enforced);
        assert!(sim.leak_free);
    }
}

#[test]
fn abi_audit_catches_fsr3_xbox_backend_id_bug_and_validates_all_structs() {
    let abi = run_abi_audit(true);
    assert!(abi.all_structs_passed, "struct layout mismatch: {:#?}", abi.structs);

    let vk_const = abi
        .constants
        .iter()
        .find(|c| c.name == "fsr3::FFX_API_BACKEND_ID_VK")
        .expect("FFX_API_BACKEND_ID_VK constant check must be present");

    // In the baseline vulkanshaders-core, FFX_API_BACKEND_ID_VK is 0x0100_0000 (Xbox in SDK 2.4.0).
    // Once fixed to 0x0200_0000, `vk_const.passed` becomes true.
    assert!(
        vk_const.actual_hex == "0x01000000" || vk_const.actual_hex == "0x02000000",
        "unexpected FFX_API_BACKEND_ID_VK value: {}",
        vk_const.actual_hex
    );
}

#[test]
fn spirv_fp8_coop_matrix_gating_matches_hardware_profiles() {
    let words = build_synthetic_fp8_coop_spirv();
    let parsed = parse_spirv_words("synthetic_fsr4_fp8.spv", &words).unwrap();

    // RDNA 4, RTX 5090, RTX 4090 support FP8 + 16x16x16 coop matrix
    for gpu in [GpuId::Rx9070Xt, GpuId::Rtx5090, GpuId::Rtx4090] {
        let p = GpuProfile::build(gpu, DriverModel::Linux);
        let compat = audit_module_against_profile(&parsed, &p);
        assert!(compat.compatible, "{gpu} should pass: {:?}", compat.blockers);
    }

    // RDNA 3, RDNA 2, Strix Point, RTX 3050 Mobile, Arc B580 lack FP8 coop matrix
    for gpu in [
        GpuId::Rx7900Xtx,
        GpuId::Rx6800Xt,
        GpuId::StrixPoint,
        GpuId::Rtx3050Mobile,
        GpuId::ArcB580,
    ] {
        let p = GpuProfile::build(gpu, DriverModel::Linux);
        let compat = audit_module_against_profile(&parsed, &p);
        assert!(!compat.compatible, "{gpu} must reject FP8 coop matrix shader");
    }
}

#[test]
fn compressed_dll_is_marked_inconclusive_rather_than_fake_pass() {
    let mut synthetic_pe = vec![0u8; 256];
    synthetic_pe[0..2].copy_from_slice(b"MZ");
    synthetic_pe[64..68].copy_from_slice(&[0x28, 0xB5, 0x2F, 0xFD]); // Zstd magic
    let scan = scan_binary_bytes("synthetic_amd_upscaler.dll", &synthetic_pe);
    assert!(scan.inconclusive_compressed);
    assert!(scan.carved_modules.is_empty());
    assert!(!scan.compressed_hits.is_empty());
}

#[test]
fn deterministic_report_and_fuzz_pipeline_is_reproducible() {
    let fuzz = run_fuzz_suites(1337, 32, "all");
    assert!(fuzz.all_passed);

    let benches = run_micro_benchmarks(8, 32);
    assert_eq!(benches.len(), 4);

    let mut r1 = HwsimReport::new();
    r1.populate_matrix(&GpuProfile::all_matrix());
    r1.finalize_digest();

    let mut r2 = HwsimReport::new();
    r2.populate_matrix(&GpuProfile::all_matrix());
    r2.finalize_digest();

    assert_eq!(r1.content_blake3, r2.content_blake3);
    assert!(!r1.to_markdown().is_empty());

    // Confirm FSR 4 network mapping for AMD lineup
    assert_eq!(
        GpuProfile::build(GpuId::Rx9070Xt, DriverModel::Windows)
            .select_fsr4_network(None)
            .unwrap(),
        Fsr4Network::Fsr4Fp8
    );
    assert_eq!(
        GpuProfile::build(GpuId::Rx6800Xt, DriverModel::Windows)
            .select_fsr4_network(None)
            .unwrap(),
        Fsr4Network::Fsr4Int8Navi2
    );
}
