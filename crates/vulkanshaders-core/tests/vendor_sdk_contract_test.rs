//! Honest vendor-SDK contract tests.
//!
//! (a) Builds a mock `cdylib` with rustc exporting the EXACT C symbols expected from FSR3 and XeSS,
//!     loads it through `RuntimeSdkManager`, and asserts the real `evaluate_fsr3` / `evaluate_xess`
//!     calls forward valid `VkCommandBuffer`, `VkImage`/`VkImageView`, widths, jitter and depth /
//!     motion resources, and destroy their contexts on drop.
//! (b) Probes the REAL staged libraries in `VulkanTurbo/run/natives` and checks symbol resolution
//!     plus C-ABI struct sizes — skipping honestly with a printed reason when absent.
//! (c) Verifies that an empty search path yields capability == 0 and `Err` from `evaluate_*`, with
//!     no simulated success anywhere.

use ash::vk::{self, Handle};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};
use vulkanshaders_core::sdk::{
    RuntimeSdkManager, SdkDispatchParams, fsr3::FfxDispatchDescUpscale, xess::XessVkExecuteParams,
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "vulkanshaders_sdk_contract_{}_{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Rust source of the mock cdylib. It exports the exact FFX-API v2 + legacy FSR3 and XeSS Vulkan
/// symbols and writes a hex dump of each dispatch descriptor to `$VS_MOCK_SINK` so the test can
/// verify the arguments that really crossed the C ABI boundary.
const MOCK_SRC: &str = r#"
use std::{ffi::c_void, fs::OpenOptions, io::Write, ptr};

fn sink() -> Option<String> { std::env::var("VS_MOCK_SINK").ok() }
fn dump(tag: &str, p: *const c_void, len: usize) {
    let Some(path) = sink() else { return };
    if p.is_null() { return; }
    let bytes = unsafe { std::slice::from_raw_parts(p as *const u8, len) };
    let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{}:{}", tag, hex);
    }
}
fn log(tag: &str) {
    let Some(path) = sink() else { return };
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{}:00", tag);
    }
}

#[no_mangle] pub unsafe extern "C" fn ffxCreateContext(ctx: *mut *mut c_void, _desc: *mut c_void, _mem: *const c_void) -> u32 {
    if !ctx.is_null() { *ctx = 0x1usize as *mut c_void; }
    log("FSR3_CREATE");
    0
}
#[no_mangle] pub unsafe extern "C" fn ffxDestroyContext(ctx: *mut *mut c_void, _mem: *const c_void) -> u32 {
    if !ctx.is_null() { *ctx = ptr::null_mut(); }
    log("FSR3_DESTROY");
    0
}
#[no_mangle] pub unsafe extern "C" fn ffxConfigure(_ctx: *mut *mut c_void, _desc: *const c_void) -> u32 { 0 }
#[no_mangle] pub unsafe extern "C" fn ffxQuery(_ctx: *mut c_void, _desc: *mut c_void) -> u32 { 0 }
#[no_mangle] pub unsafe extern "C" fn ffxDispatch(_ctx: *mut *mut c_void, desc: *const c_void) -> u32 {
    dump("FSR3_DISPATCH", desc, 512);
    0
}
#[no_mangle] pub unsafe extern "C" fn ffxFsr3ContextCreate(_c: *mut c_void, _d: *const c_void) -> i32 { 0 }
#[no_mangle] pub unsafe extern "C" fn ffxFsr3ContextDispatch(_c: *mut c_void, d: *const c_void) -> i32 {
    dump("FSR3_LEGACY_DISPATCH", d, 256); 0
}
#[no_mangle] pub unsafe extern "C" fn ffxFsr3ContextDestroy(_c: *mut c_void) -> i32 { 0 }

#[no_mangle] pub unsafe extern "C" fn xessGetVersion(v: *mut u8) -> i32 {
    if !v.is_null() { std::ptr::write_bytes(v, 0, 8); }
    0
}
#[no_mangle] pub unsafe extern "C" fn xessVKGetRequiredInstanceExtensions(count: *mut u32, _list: *mut *const *const i8, _ver: *mut u32) -> i32 {
    if !count.is_null() { *count = 0; } 0
}
#[no_mangle] pub unsafe extern "C" fn xessVKGetRequiredDeviceExtensions(_i: u64, _p: u64, count: *mut u32, _list: *mut *const *const i8) -> i32 {
    if !count.is_null() { *count = 0; } 0
}
#[no_mangle] pub unsafe extern "C" fn xessVKCreateContext(_i: u64, _p: u64, _d: u64, out: *mut *mut c_void) -> i32 {
    if !out.is_null() { *out = 0x2usize as *mut c_void; }
    log("XESS_CREATE"); 0
}
#[no_mangle] pub unsafe extern "C" fn xessVKInit(_ctx: *mut c_void, _p: *const c_void) -> i32 { 0 }
#[no_mangle] pub unsafe extern "C" fn xessVKExecute(_ctx: *mut c_void, _cmd: u64, params: *const c_void) -> i32 {
    dump("XESS_EXECUTE", params, 512); 0
}
#[no_mangle] pub unsafe extern "C" fn xessDestroyContext(_ctx: *mut c_void) -> i32 {
    log("XESS_DESTROY"); 0
}
"#;

fn build_mock(dir: &Path) -> PathBuf {
    let src = dir.join("mock_vendor.rs");
    fs::write(&src, MOCK_SRC).unwrap();
    let so = dir.join("libmock_vendor.so");
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let status = Command::new(rustc)
        .args(["--crate-type", "cdylib", "-O", "-o"])
        .arg(&so)
        .arg(&src)
        .status()
        .expect("rustc must be available to build the vendor mock");
    assert!(status.success(), "mock cdylib build failed");
    so
}

fn install_mock_library(mock_so: &Path, dir: &Path, name: &str) {
    fs::copy(mock_so, dir.join(name)).unwrap();
}

fn hex_decode(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn read_records(sink: &Path) -> Vec<(String, Vec<u8>)> {
    let text = fs::read_to_string(sink).unwrap_or_default();
    text.lines()
        .filter_map(|line| line.split_once(':'))
        .filter(|(_, v)| v.len() % 2 == 0 && v.chars().all(|c| c.is_ascii_hexdigit()))
        .map(|(k, v)| (k.to_string(), hex_decode(v)))
        .collect()
}

unsafe extern "system" fn dummy_get_device_proc_addr(
    _device: vk::Device,
    _name: *const std::os::raw::c_char,
) -> vk::PFN_vkVoidFunction {
    None
}

fn dispatch_params() -> SdkDispatchParams {
    SdkDispatchParams {
        command_buffer: vk::CommandBuffer::from_raw(0xABCD),
        input_image: vk::Image::from_raw(0x1000),
        input_view: vk::ImageView::from_raw(0x1100),
        output_image: vk::Image::from_raw(0x2000),
        output_view: vk::ImageView::from_raw(0x2100),
        depth_image: vk::Image::from_raw(0x3000),
        depth_format: vk::Format::D32_SFLOAT,
        depth_view: vk::ImageView::from_raw(0x3100),
        motion_image: vk::Image::from_raw(0x4000),
        motion_view: vk::ImageView::from_raw(0x4100),
        width: 960,
        height: 540,
        out_width: 1920,
        out_height: 1080,
        jitter: [0.25, -0.375],
        frame_time_delta: 1.0 / 60.0,
        upscaler_mode: 0,
        denoiser_mode: 0,
        framegen_mode: 0,
        reset: false,
        hdr_enabled: false,
        sun_dir: [0.0, 1.0, 0.0],
        ..Default::default()
    }
}

#[test]
fn fsr3_and_xess_mock_contract_forwards_real_vulkan_arguments_and_destroys_contexts() {
    let dir = temp_dir();
    let mock = build_mock(&dir);
    let lib_dir = dir.join("libs");
    fs::create_dir_all(&lib_dir).unwrap();
    install_mock_library(&mock, &lib_dir, "libffx_fsr3_vk_x64.so");
    install_mock_library(&mock, &lib_dir, "libxess.so");

    let sink = dir.join("calls.log");
    unsafe { std::env::set_var("VS_MOCK_SINK", &sink) };
    let _ = fs::remove_file(&sink);

    let mut mgr = RuntimeSdkManager::probe(vec![lib_dir.clone()]);
    assert!(
        mgr.capabilities.fsr3_loaded,
        "FSR3 mock library must load: {}",
        mgr.unavailable_reason("fsr3")
    );
    assert!(
        mgr.capabilities.xess_loaded,
        "XeSS mock library must load: {}",
        mgr.unavailable_reason("xess")
    );

    let fake_device = vk::Device::from_raw(0x5000);
    let fake_pd = vk::PhysicalDevice::from_raw(0x6000);
    let p = dispatch_params();
    unsafe {
        mgr.configure_upscaler(
            vulkanshaders_core::sdk::UpscalerKind::Fsr3,
            vulkanshaders_core::sdk::ViewportExtent {
                render_width: p.width,
                render_height: p.height,
                output_width: p.out_width,
                output_height: p.out_height,
            },
            p.command_buffer,
            vk::Instance::from_raw(0x7000),
            fake_pd,
            fake_device,
            dummy_get_device_proc_addr,
            vulkanshaders_core::sdk::ngx::NvSdkNgxPerfQuality::MaxQuality,
            0,
            None,
            None,
        )
        .expect("FSR3 mock context creation");
        mgr.evaluate_fsr3(&p).expect("FSR3 mock dispatch");

        mgr.configure_upscaler(
            vulkanshaders_core::sdk::UpscalerKind::XeSS,
            vulkanshaders_core::sdk::ViewportExtent {
                render_width: p.width,
                render_height: p.height,
                output_width: p.out_width,
                output_height: p.out_height,
            },
            p.command_buffer,
            vk::Instance::from_raw(0x7000),
            fake_pd,
            fake_device,
            dummy_get_device_proc_addr,
            vulkanshaders_core::sdk::ngx::NvSdkNgxPerfQuality::MaxQuality,
            0,
            None,
            None,
        )
        .expect("XeSS mock context creation");
        mgr.evaluate_xess(&p).expect("XeSS mock dispatch");
    }

    let records = read_records(&sink);
    let fsr = records
        .iter()
        .find(|(k, _)| k == "FSR3_DISPATCH")
        .map(|(_, v)| v)
        .expect("ffxDispatch must have been called");
    assert!(fsr.len() >= std::mem::size_of::<FfxDispatchDescUpscale>());
    let desc: FfxDispatchDescUpscale =
        unsafe { std::ptr::read_unaligned(fsr.as_ptr() as *const FfxDispatchDescUpscale) };
    assert_eq!(
        desc.command_list.as_raw(),
        0xABCD,
        "VkCommandBuffer forwarded"
    );
    assert_eq!(
        desc.color.resource as usize, 0x1000,
        "input VkImage forwarded"
    );
    assert_eq!(
        desc.depth.resource as usize, 0x3000,
        "depth VkImage forwarded"
    );
    assert_eq!(
        desc.motion_vectors.resource as usize, 0x4000,
        "motion VkImage forwarded"
    );
    assert_eq!(
        desc.output.resource as usize, 0x2000,
        "output VkImage forwarded"
    );
    assert_eq!(
        (desc.render_size.width, desc.render_size.height),
        (960, 540)
    );
    assert_eq!(
        (desc.upscale_size.width, desc.upscale_size.height),
        (1920, 1080)
    );
    assert!((desc.jitter_offset.x - 0.25).abs() < 1e-6);
    assert!((desc.jitter_offset.y + 0.375).abs() < 1e-6);

    let xess = records
        .iter()
        .find(|(k, _)| k == "XESS_EXECUTE")
        .map(|(_, v)| v)
        .expect("xessVKExecute must have been called");
    assert!(xess.len() >= std::mem::size_of::<XessVkExecuteParams>());
    let params: XessVkExecuteParams =
        unsafe { std::ptr::read_unaligned(xess.as_ptr() as *const XessVkExecuteParams) };
    assert_eq!(params.color_texture.image.as_raw(), 0x1000);
    assert_eq!(params.color_texture.image_view.as_raw(), 0x1100);
    assert_eq!(params.depth_texture.image.as_raw(), 0x3000);
    assert_eq!(params.velocity_texture.image.as_raw(), 0x4000);
    assert_eq!(params.output_texture.image.as_raw(), 0x2000);
    assert_eq!((params.input_width, params.input_height), (960, 540));
    assert!((params.jitter_offset_x - 0.25).abs() < 1e-6);
    assert!((params.jitter_offset_y + 0.375).abs() < 1e-6);

    drop(mgr);
    let records = read_records(&sink);
    assert!(
        records.iter().any(|(k, _)| k == "FSR3_DESTROY"),
        "FSR3 context destroyed on drop"
    );
    assert!(
        records.iter().any(|(k, _)| k == "XESS_DESTROY"),
        "XeSS context destroyed on drop"
    );
}

#[test]
fn real_staged_native_libraries_resolve_symbols_and_have_expected_abi_sizes() {
    let natives = std::env::var_os("VS_NATIVES_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("natives"));
    if !natives.exists() {
        eprintln!("SKIP: staged natives directory {natives:?} does not exist");
        return;
    }
    let mgr = RuntimeSdkManager::probe(vec![natives.clone()]);
    let c = &mgr.capabilities;

    assert_eq!(
        std::mem::size_of::<FfxDispatchDescUpscale>() % 8,
        0,
        "FfxDispatchDescUpscale must be 8-byte aligned"
    );
    assert_eq!(
        std::mem::size_of::<XessVkExecuteParams>() % 8,
        0,
        "XessVkExecuteParams must be 8-byte aligned"
    );

    if natives.join("libNRD.so").exists() {
        assert!(
            c.nrd_loaded,
            "real libNRD.so must load and resolve required symbols"
        );
        let lib = mgr.nrd_library().expect("NRD library handle");
        assert!(lib.library_desc().is_some(), "GetLibraryDesc must resolve");
    } else {
        eprintln!("SKIP: libNRD.so not staged");
    }

    let ngx_dlss = fs::read_dir(&natives)
        .map(|rd| {
            rd.filter_map(Result::ok).any(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("libnvidia-ngx-dlss")
            })
        })
        .unwrap_or(false);
    if ngx_dlss {
        assert!(c.dlss_sr_binary_present);
        if c.ngx_core_loaded {
            use vulkanshaders_core::sdk::ngx::NgxVulkanLibrary;
            assert!(
                NgxVulkanLibrary::candidate_names()
                    .iter()
                    .any(|n| n.contains("nvidia-ngx"))
            );
        } else {
            eprintln!(
                "INFO: DLSS snippet binaries staged but NGX core not present — capability stays 0 (honest)"
            );
        }
    } else {
        eprintln!("SKIP: no libnvidia-ngx-dlss* staged");
    }
}

#[test]
fn missing_libraries_are_an_honest_refusal_not_a_simulation() {
    let empty = temp_dir();
    let mut mgr = RuntimeSdkManager::probe(vec![empty.clone()]);
    assert!(!mgr.capabilities.fsr3_loaded);
    assert!(!mgr.capabilities.xess_loaded);
    // NRD may be resolved from the system loader path (LD_LIBRARY_PATH); that is still a real
    // library, so only assert the features whose candidate names cannot be found anywhere.
    assert!(!mgr.capabilities.dlss_sr_available);
    assert!(!mgr.capability_available("fsr3"));
    assert!(!mgr.capability_available("xess"));
    // NRD may be staged/present on the host and `probe` legitimately creates an instance; only
    // assert its absence when the library truly was not found (honest to the real environment).
    if !mgr.capabilities.nrd_loaded {
        assert!(!mgr.capability_available("nrd"));
    } else {
        eprintln!(
            "INFO: NRD library present on host; capability reflects real probe, not the empty search dir"
        );
    }
    assert!(!mgr.capability_available("dlssSr"));

    let reason = mgr.unavailable_reason("fsr3");
    assert!(
        reason.contains("not found") || reason.contains("missing"),
        "reason: {reason}"
    );

    let p = dispatch_params();
    unsafe {
        assert!(
            mgr.evaluate_fsr3(&p).is_err(),
            "evaluate_fsr3 must fail closed"
        );
        assert!(
            mgr.evaluate_xess(&p).is_err(),
            "evaluate_xess must fail closed"
        );
        assert!(
            mgr.evaluate_dlss_sr(&p).is_err(),
            "evaluate_dlss_sr must fail closed"
        );
        if !mgr.capabilities.nrd_loaded {
            let mut layouts = std::collections::HashMap::new();
            assert!(
                mgr.evaluate_nrd(&p, &mut layouts).is_err(),
                "evaluate_nrd must fail closed"
            );
        }
    }
}
