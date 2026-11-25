// SPDX-License-Identifier: GPL-3.0-only
// Additional permissions under GPLv3 Section 7 apply to optional proprietary vendor SDK
// runtime loading; see LICENSE-EXCEPTION.md in the repository root.

//! Deterministic BLAKE3-XOF property fuzzer and CPU-side micro-benchmark harness
//! (zero external benchmark crate dependencies).

use crate::fsr4_sim::{
    FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE, FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION,
    FFX_API_DISPATCH_DESC_TYPE_UPSCALE, FFX_SDK24_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK,
    FfxCreateContextDescUpscaleVersion, FfxSdk24DispatchDescUpscale, MockFfxProvider,
    fsr_jitter_phase_count, halton_jitter,
};
use crate::mock_icd::{
    MockIcdGuard, mock_vk_get_device_proc_addr, reclaim_mock_physical_device, validate_pnext_chain,
};
use crate::profiles::{DriverModel, GpuId, GpuProfile};
use crate::spirv_audit::{
    audit_module_against_profile, build_synthetic_fp8_coop_spirv, parse_spirv_words,
    scan_binary_bytes,
};
use ash::vk;
use serde::{Deserialize, Serialize};
use std::ffi::c_void;
use std::hint::black_box;
use std::ptr;
use std::time::Instant;
use vulkanshaders_core::sdk::fsr3::{
    FfxApiDimensions2D, FfxApiFloatCoords2D, FfxApiHeader, FfxApiResource, FfxCreateBackendVKDesc,
    FfxVkDeviceContext, resource_state, resource_usage, upscale_flags,
};

/// Deterministic cross-platform PRNG backed by `blake3::OutputReader`.
pub struct Blake3Rng {
    reader: blake3::OutputReader,
}

impl Blake3Rng {
    pub fn from_seed(seed: u64, domain: &str) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(domain.as_bytes());
        hasher.update(&seed.to_le_bytes());
        Self {
            reader: hasher.finalize_xof(),
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut buf = [0u8; 8];
        self.reader.fill(&mut buf);
        u64::from_le_bytes(buf)
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() & 0xFFFF_FFFF) as u32
    }

    pub fn next_f32_unit(&mut self) -> f32 {
        ((self.next_u32() >> 8) as f32) / ((1u32 << 24) as f32)
    }

    pub fn fill_bytes(&mut self, dst: &mut [u8]) {
        self.reader.fill(dst);
    }
}

/// Summary of one deterministic property fuzz suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzSuiteResult {
    pub suite: String,
    pub seed: u64,
    pub cases_run: usize,
    pub passed: bool,
    pub failures: Vec<String>,
}

/// Aggregated fuzz report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzReport {
    pub seed: u64,
    pub suites: Vec<FuzzSuiteResult>,
    pub all_passed: bool,
}

/// Runs deterministic property fuzz suites (`pnext`, `layout`, `fsr`, `spirv`, or `all`).
pub fn run_fuzz_suites(seed: u64, cases: usize, suite_filter: &str) -> FuzzReport {
    let mut suites = Vec::new();
    let run_all = suite_filter == "all";

    if run_all || suite_filter == "pnext" {
        suites.push(fuzz_pnext_chains(seed, cases));
    }
    if run_all || suite_filter == "layout" {
        suites.push(fuzz_layout_fsm(seed, cases));
    }
    if run_all || suite_filter == "fsr" {
        suites.push(fuzz_fsr_dispatch_contracts(seed, cases));
    }
    if run_all || suite_filter == "spirv" {
        suites.push(fuzz_spirv_carver(seed, cases));
    }

    if suites.is_empty() {
        suites.push(FuzzSuiteResult {
            suite: format!("unknown_suite:{suite_filter}"),
            seed,
            cases_run: 0,
            passed: false,
            failures: vec![format!(
                "unknown fuzz suite `{suite_filter}` (expected: all, pnext, layout, fsr, spirv)"
            )],
        });
    }
    let all_passed = !suites.is_empty() && suites.iter().all(|s| s.passed);
    FuzzReport {
        seed,
        suites,
        all_passed,
    }
}

fn fuzz_pnext_chains(seed: u64, cases: usize) -> FuzzSuiteResult {
    let mut rng = Blake3Rng::from_seed(seed, "hyaloclast-hwsim-pnext-v1");
    let mut failures = Vec::new();
    let profile = GpuProfile::build(GpuId::Rx6800Xt, DriverModel::Linux);

    for case in 0..cases {
        let len = (rng.next_u32() % 8 + 1) as usize;
        let inject_cycle = (rng.next_u32() % 4) == 0;
        let mut nodes: Vec<vk::BaseInStructure<'static>> = (0..len)
            .map(|_| vk::BaseInStructure {
                s_type: vk::StructureType::APPLICATION_INFO,
                p_next: ptr::null(),
                _marker: std::marker::PhantomData,
            })
            .collect();

        for i in 0..len.saturating_sub(1) {
            let next_ptr = &nodes[i + 1] as *const vk::BaseInStructure<'static>;
            nodes[i].p_next = next_ptr;
        }
        if inject_cycle {
            let target = (rng.next_u32() as usize) % len;
            let back_ptr = &nodes[target] as *const vk::BaseInStructure<'static>;
            nodes[len - 1].p_next = back_ptr;
        }

        let head = &nodes[0] as *const vk::BaseInStructure<'static> as *const c_void;
        let res = unsafe { validate_pnext_chain(head, Some(&profile)) };
        if inject_cycle && res.is_ok() {
            failures.push(format!("case {case}: failed to detect pNext cycle"));
        } else if !inject_cycle && res != Ok(len) {
            failures.push(format!("case {case}: expected depth {len}, got {res:?}"));
        }
    }

    FuzzSuiteResult {
        suite: "pnext_chain_fuzz".to_string(),
        seed,
        cases_run: cases,
        passed: failures.is_empty(),
        failures,
    }
}

fn fuzz_layout_fsm(seed: u64, cases: usize) -> FuzzSuiteResult {
    let mut rng = Blake3Rng::from_seed(seed, "hyaloclast-hwsim-layout-v1");
    let mut failures = Vec::new();
    let layouts = [
        vk::ImageLayout::GENERAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    ];

    let profile = GpuProfile::build(GpuId::Rx9070Xt, DriverModel::Linux);
    let (guard, entry) = MockIcdGuard::install(profile);

    unsafe {
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let inst = entry
            .create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
            .unwrap();
        let phys = inst.enumerate_physical_devices().unwrap()[0];
        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(0)
            .queue_priorities(&prio)];
        let dev = inst
            .create_device(phys, &vk::DeviceCreateInfo::default().queue_create_infos(&qci), None)
            .unwrap();
        let img = dev
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R16G16B16A16_SFLOAT)
                    .extent(vk::Extent3D {
                        width: 64,
                        height: 64,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .unwrap();
        let pool = dev
            .create_command_pool(&vk::CommandPoolCreateInfo::default(), None)
            .unwrap();
        let cbs = dev
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .unwrap();
        let cb = cbs[0];

        let mut expected = vk::ImageLayout::UNDEFINED;
        for case in 0..cases {
            let next = layouts[(rng.next_u32() as usize) % layouts.len()];
            let use_sync2 = (rng.next_u32() & 1) == 0;
            let old = if (rng.next_u32() % 5) == 0 {
                vk::ImageLayout::UNDEFINED
            } else {
                expected
            };
            if use_sync2 {
                let b = [vk::ImageMemoryBarrier2::default()
                    .image(img)
                    .old_layout(old)
                    .new_layout(next)];
                dev.cmd_pipeline_barrier2(
                    cb,
                    &vk::DependencyInfo::default().image_memory_barriers(&b),
                );
            } else {
                let b = [vk::ImageMemoryBarrier::default()
                    .image(img)
                    .old_layout(old)
                    .new_layout(next)];
                dev.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &b,
                );
            }
            expected = next;
            if guard.state().query_image_layout(img) != Some(expected) {
                failures.push(format!("case {case}: layout diverged from {expected:?}"));
                break;
            }
        }

        dev.free_command_buffers(pool, &cbs);
        dev.destroy_command_pool(pool, None);
        dev.destroy_image(img, None);
        dev.destroy_device(None);
        reclaim_mock_physical_device(phys);
        inst.destroy_instance(None);
    }

    if !guard.state().leak_report().is_clean() {
        failures.push(format!(
            "MockIcd reported leaks/violations: {:?}",
            guard.state().leak_report()
        ));
    }

    FuzzSuiteResult {
        suite: "layout_fsm_fuzz".to_string(),
        seed,
        cases_run: cases,
        passed: failures.is_empty(),
        failures,
    }
}

fn fuzz_fsr_dispatch_contracts(seed: u64, cases: usize) -> FuzzSuiteResult {
    let mut rng = Blake3Rng::from_seed(seed, "hyaloclast-hwsim-fsr-v1");
    let mut failures = Vec::new();
    let profile = GpuProfile::build(GpuId::Rx9070Xt, DriverModel::Linux);
    let (guard, entry) = MockIcdGuard::install(profile);
    let mut provider = MockFfxProvider::new(guard.state().clone(), None);

    unsafe {
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let inst = entry
            .create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
            .unwrap();
        let phys = inst.enumerate_physical_devices().unwrap()[0];
        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(0)
            .queue_priorities(&prio)];
        let dev = inst
            .create_device(phys, &vk::DeviceCreateInfo::default().queue_create_infos(&qci), None)
            .unwrap();

        let mut ver = FfxCreateContextDescUpscaleVersion {
            header: FfxApiHeader {
                desc_type: FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION,
                p_next: ptr::null_mut(),
            },
            version: 0x0004_0001_0002,
        };
        let mut backend = FfxCreateBackendVKDesc {
            header: FfxApiHeader {
                desc_type: FFX_SDK24_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK,
                p_next: &mut ver.header,
            },
            device_context: FfxVkDeviceContext {
                vk_device: dev.handle(),
                vk_physical_device: phys,
                vk_device_proc_addr: mock_vk_get_device_proc_addr,
            },
        };
        let hdr = FfxApiHeader {
            desc_type: FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE,
            p_next: &mut backend.header,
        };
        provider
            .create_context(
                &hdr,
                upscale_flags::DEPTH_INVERTED | upscale_flags::DYNAMIC_RESOLUTION,
                FfxApiDimensions2D {
                    width: 1280,
                    height: 720,
                },
                FfxApiDimensions2D {
                    width: 1920,
                    height: 1080,
                },
            )
            .unwrap();

        let mk_img = |usage: vk::ImageUsageFlags| {
            dev.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R16G16B16A16_SFLOAT)
                    .extent(vk::Extent3D {
                        width: 1280,
                        height: 720,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .usage(usage)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .unwrap()
        };
        let color = mk_img(vk::ImageUsageFlags::SAMPLED);
        let depth = mk_img(vk::ImageUsageFlags::SAMPLED);
        let mv = mk_img(vk::ImageUsageFlags::SAMPLED);
        let output = mk_img(vk::ImageUsageFlags::STORAGE);
        let pool = dev
            .create_command_pool(&vk::CommandPoolCreateInfo::default(), None)
            .unwrap();
        let cbs = dev
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .unwrap();
        let cb = cbs[0];

        let barriers = [
            vk::ImageMemoryBarrier2::default()
                .image(color)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            vk::ImageMemoryBarrier2::default()
                .image(depth)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            vk::ImageMemoryBarrier2::default()
                .image(mv)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            vk::ImageMemoryBarrier2::default()
                .image(output)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL),
        ];
        dev.cmd_pipeline_barrier2(
            cb,
            &vk::DependencyInfo::default().image_memory_barriers(&barriers),
        );

        for case in 0..cases {
            let fault_kind = rng.next_u32() % 5;
            let mut desc = FfxSdk24DispatchDescUpscale {
                header: FfxApiHeader {
                    desc_type: FFX_API_DISPATCH_DESC_TYPE_UPSCALE,
                    p_next: ptr::null_mut(),
                },
                command_list: cb,
                color: FfxApiResource::from_vk_image(
                    color,
                    vk::Format::R16G16B16A16_SFLOAT,
                    1280,
                    720,
                    1,
                    resource_usage::READ_ONLY,
                    resource_state::COMPUTE_READ,
                ),
                depth: FfxApiResource::from_vk_image(
                    depth,
                    vk::Format::D32_SFLOAT,
                    1280,
                    720,
                    1,
                    resource_usage::READ_ONLY,
                    resource_state::COMPUTE_READ,
                ),
                motion_vectors: FfxApiResource::from_vk_image(
                    mv,
                    vk::Format::R16G16_SFLOAT,
                    1280,
                    720,
                    1,
                    resource_usage::READ_ONLY,
                    resource_state::COMPUTE_READ,
                ),
                exposure: FfxApiResource::default(),
                reactive: FfxApiResource::default(),
                transparency_and_composition: FfxApiResource::default(),
                output: FfxApiResource::from_vk_image(
                    output,
                    vk::Format::R16G16B16A16_SFLOAT,
                    1920,
                    1080,
                    1,
                    resource_usage::UAV,
                    resource_state::UNORDERED_ACCESS,
                ),
                jitter_offset: FfxApiFloatCoords2D {
                    x: rng.next_f32_unit() - 0.5,
                    y: rng.next_f32_unit() - 0.5,
                },
                motion_vector_scale: FfxApiFloatCoords2D {
                    x: -1280.0,
                    y: -720.0,
                },
                render_size: FfxApiDimensions2D {
                    width: 1280,
                    height: 720,
                },
                upscale_size: FfxApiDimensions2D {
                    width: 1920,
                    height: 1080,
                },
                enable_sharpening: 1,
                sharpness: 0.3,
                frame_time_delta: 16.6,
                pre_exposure: 1.0,
                reset: 0,
                camera_near: 512.0,
                camera_far: 0.05,
                camera_fov_angle_vertical: 1.2,
                view_space_to_meters_factor: 1.0,
                flags: 0,
            };

            let expect_err = match fault_kind {
                0 => {
                    desc.jitter_offset.x = f32::NAN;
                    true
                }
                1 => {
                    desc.jitter_offset.y = 0.75;
                    true
                }
                2 => {
                    desc.frame_time_delta = 0.0;
                    true
                }
                3 => {
                    desc.camera_near = 0.05;
                    desc.camera_far = 512.0; // invalid for DEPTH_INVERTED
                    true
                }
                _ => false,
            };

            let res = provider.dispatch(&desc);
            if expect_err && res.is_ok() {
                failures.push(format!(
                    "case {case}: adversarial fault {fault_kind} was not rejected"
                ));
            } else if !expect_err {
                if let Err((code, msg)) = res {
                    failures.push(format!("case {case}: valid dispatch failed ({code}): {msg}"));
                } else {
                    provider.retire_oldest_frame();
                }
            }
        }

        provider.destroy_context();
        dev.free_command_buffers(pool, &cbs);
        dev.destroy_command_pool(pool, None);
        dev.destroy_image(color, None);
        dev.destroy_image(depth, None);
        dev.destroy_image(mv, None);
        dev.destroy_image(output, None);
        dev.destroy_device(None);
        reclaim_mock_physical_device(phys);
        inst.destroy_instance(None);
    }

    FuzzSuiteResult {
        suite: "fsr_dispatch_fuzz".to_string(),
        seed,
        cases_run: cases,
        passed: failures.is_empty(),
        failures,
    }
}

fn fuzz_spirv_carver(seed: u64, cases: usize) -> FuzzSuiteResult {
    let mut rng = Blake3Rng::from_seed(seed, "hyaloclast-hwsim-spirv-v1");
    let valid_spv = build_synthetic_fp8_coop_spirv();
    let valid_bytes: &[u8] = bytemuck::cast_slice(&valid_spv);
    let mut failures = Vec::new();

    for case in 0..cases {
        let len = (rng.next_u32() % 512 + 32) as usize;
        let mut buf = vec![0u8; len];
        rng.fill_bytes(&mut buf);
        // Embed a real SPIR-V module in half the cases to verify zero-panic carving
        if (case % 2) == 0 && len > valid_bytes.len() + 16 {
            let off = ((rng.next_u32() as usize) % (len - valid_bytes.len())) & !3;
            buf[off..off + valid_bytes.len()].copy_from_slice(valid_bytes);
        }
        let scan = scan_binary_bytes("fuzz_blob.bin", &buf);
        if scan.file_blake3.len() != 64 {
            failures.push(format!("case {case}: invalid blake3 digest length"));
        }
    }

    FuzzSuiteResult {
        suite: "spirv_carver_fuzz".to_string(),
        seed,
        cases_run: cases,
        passed: failures.is_empty(),
        failures,
    }
}

/// Statistics for one CPU-side micro-benchmark (in nanoseconds per operation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroBenchResult {
    pub name: String,
    pub iterations: usize,
    pub min_ns: f64,
    pub p50_ns: f64,
    pub p95_ns: f64,
    pub p99_ns: f64,
    pub mean_ns: f64,
    pub mad_ns: f64,
}

fn run_timed<F>(name: &str, warmup: usize, iterations: usize, mut op: F) -> MicroBenchResult
where
    F: FnMut(usize),
{
    for i in 0..warmup {
        op(i);
    }
    let samples_count = iterations.clamp(16, 4096);
    let batch = (iterations / samples_count).max(1);
    let mut samples = Vec::with_capacity(samples_count);

    for s in 0..samples_count {
        let t0 = Instant::now();
        for b in 0..batch {
            op(s * batch + b);
        }
        let elapsed = Instant::now().saturating_duration_since(t0).as_nanos() as f64;
        samples.push(elapsed / (batch as f64));
    }

    samples.sort_by(|a, b| a.total_cmp(b));
    let n = samples.len();
    let min_ns = samples[0];
    let p50_ns = samples[n / 2];
    let p95_ns = samples[((n * 95) / 100).min(n - 1)];
    let p99_ns = samples[((n * 99) / 100).min(n - 1)];
    let mean_ns = samples.iter().sum::<f64>() / (n as f64);

    let mut abs_devs: Vec<f64> = samples.iter().map(|&x| (x - p50_ns).abs()).collect();
    abs_devs.sort_by(|a, b| a.total_cmp(b));
    let mad_ns = abs_devs[n / 2];

    MicroBenchResult {
        name: name.to_string(),
        iterations: samples_count * batch,
        min_ns,
        p50_ns,
        p95_ns,
        p99_ns,
        mean_ns,
        mad_ns,
    }
}

/// Runs the deterministic CPU-side micro-benchmark suite.
pub fn run_micro_benchmarks(warmup: usize, iterations: usize) -> Vec<MicroBenchResult> {
    let mut out = Vec::new();

    // 1. Halton(2, 3) jitter + jitter phase count math
    out.push(run_timed("fsr_halton_jitter_math", warmup, iterations, |i| {
        let phase = fsr_jitter_phase_count(black_box(1280), black_box(1920));
        let j = halton_jitter(black_box(i as u32), phase);
        black_box(j);
    }));

    // 2. 16-profile GPU matrix FSR 4.1.2 network selection & fallback chain
    let profiles = GpuProfile::all_matrix();
    out.push(run_timed(
        "gpu_matrix_fsr4_network_select_16x",
        warmup,
        iterations,
        |_| {
            let mut ml_count = 0u32;
            for p in &profiles {
                if let Ok(net) = black_box(p).select_fsr4_network(None)
                    && net.is_ml_fsr4()
                {
                    ml_count += 1;
                }
            }
            black_box(ml_count);
        },
    ));

    // 3. pNext chain validation (depth 8)
    let profile_rdna4 = GpuProfile::build(GpuId::Rx9070Xt, DriverModel::Linux);
    let mut nodes: Vec<vk::BaseInStructure<'static>> = (0..8)
        .map(|_| vk::BaseInStructure {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_next: ptr::null(),
            _marker: std::marker::PhantomData,
        })
        .collect();
    for i in 0..7 {
        let ptr_next = &nodes[i + 1] as *const vk::BaseInStructure<'static>;
        nodes[i].p_next = ptr_next;
    }
    let head = &nodes[0] as *const vk::BaseInStructure<'static> as *const c_void;
    out.push(run_timed("mock_icd_pnext_validate_d8", warmup, iterations, |_| {
        let depth = unsafe { validate_pnext_chain(black_box(head), Some(&profile_rdna4)) };
        black_box(depth).ok();
    }));

    // 4. SPIR-V FP8 + Cooperative Matrix parse & profile compatibility audit
    let spv = build_synthetic_fp8_coop_spirv();
    out.push(run_timed(
        "spirv_fp8_coopmat_parse_and_audit",
        warmup.min(100),
        iterations.min(2000),
        |_| {
            let parsed = parse_spirv_words("bench.spv", black_box(&spv)).unwrap();
            let compat = audit_module_against_profile(&parsed, &profile_rdna4);
            black_box(compat.compatible);
        },
    ));

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_fuzz_suites_pass_cleanly() {
        let report = run_fuzz_suites(0xDEAD_BEEF_CAFE_BABEu64, 64, "all");
        assert!(report.all_passed, "fuzz failures: {:#?}", report.suites);
    }

    #[test]
    fn micro_benchmarks_produce_valid_percentiles() {
        let results = run_micro_benchmarks(10, 64);
        assert_eq!(results.len(), 4);
        for r in results {
            assert!(r.min_ns <= r.p50_ns);
            assert!(r.p50_ns <= r.p95_ns);
            assert!(r.p95_ns <= r.p99_ns);
        }
    }
}
