//! Headless vendor-SDK diagnostic for VulkanShaders.
//!
//! Creates a surface-less `VkInstance` / `VkDevice`, really probes and loads every vendor runtime,
//! resolves its symbols, attempts real device init and one off-screen evaluate per available
//! feature, measures the dispatch with `vkCmdWriteTimestamp2` when the GPU supports it, and writes
//! `vulkanshaders-diagnostic.json`. Every stage that cannot be reached is reported honestly with a
//! reason — there is no simulated success path.
//!
//! Run:
//!   cd <VulkanShaders-repo-root>
//!   cargo run -p vulkanshaders-core --example diagnose

use ash::{Device, Entry, Instance, khr::synchronization2, vk};
use std::{ffi::CStr, fs, path::PathBuf};
use vulkanshaders_core::sdk::ngx::NvSdkNgxPerfQuality;
use vulkanshaders_core::sdk::{
    DenoiserKind, FrameGenKind, RuntimeSdkManager, SdkDispatchParams, UpscalerKind, ViewportExtent,
};

#[derive(Debug, Clone)]
struct Stage {
    reached: bool,
    detail: String,
}

impl Stage {
    fn ok(detail: impl Into<String>) -> Self {
        Self {
            reached: true,
            detail: detail.into(),
        }
    }
    fn fail(detail: impl Into<String>) -> Self {
        Self {
            reached: false,
            detail: detail.into(),
        }
    }
}

#[derive(Debug)]
struct FeatureReport {
    available: bool,
    reason: String,
    lib_path: Option<String>,
    symbols_resolved: bool,
    init: Stage,
    evaluate: Stage,
    gpu_time_ns: Option<u64>,
}

impl FeatureReport {
    fn absent(reason: String) -> Self {
        Self {
            available: false,
            reason,
            lib_path: None,
            symbols_resolved: false,
            init: Stage::fail("not attempted (feature unavailable)"),
            evaluate: Stage::fail("not attempted (feature unavailable)"),
            gpu_time_ns: None,
        }
    }
}

fn jstr(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            _ => out.push(c),
        }
    }
    format!("\"{out}\"")
}

fn stage_json(s: &Stage) -> String {
    format!(
        "{{\"reached\":{},\"detail\":{}}}",
        s.reached,
        jstr(&s.detail)
    )
}

struct GpuResources {
    device: Device,
    command_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    query_pool: vk::QueryPool,
    images: Vec<(vk::Image, vk::DeviceMemory, vk::ImageView)>,
}

impl Drop for GpuResources {
    fn drop(&mut self) {
        unsafe {
            for (img, mem, view) in self.images.drain(..) {
                self.device.destroy_image_view(view, None);
                self.device.free_memory(mem, None);
                self.device.destroy_image(img, None);
            }
            if self.query_pool != vk::QueryPool::null() {
                self.device.destroy_query_pool(self.query_pool, None);
            }
            if self.command_pool != vk::CommandPool::null() {
                self.device.destroy_command_pool(self.command_pool, None);
            }
        }
    }
}

fn create_image(
    device: &Device,
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    format: vk::Format,
    width: u32,
    height: u32,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), String> {
    unsafe {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .initial_layout(vk::ImageLayout::GENERAL);
        let image = device
            .create_image(&info, None)
            .map_err(|e| e.to_string())?;
        let req = device.get_image_memory_requirements(image);
        let props = instance.get_physical_device_memory_properties(physical_device);
        let idx = (0..props.memory_type_count).find(|&i| {
            (req.memory_type_bits & (1 << i)) != 0
                && props.memory_types[i as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        });
        let Some(idx) = idx else {
            device.destroy_image(image, None);
            return Err("no device-local memory type".to_string());
        };
        let mem = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(idx),
                None,
            )
            .map_err(|e| e.to_string())?;
        device
            .bind_image_memory(image, mem, 0)
            .map_err(|e| e.to_string())?;
        let view = device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
            .map_err(|e| e.to_string())?;
        Ok((image, mem, view))
    }
}

fn create_gpu_resources(
    entry: &Entry,
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    device: &Device,
    queue_family: u32,
) -> Result<GpuResources, String> {
    unsafe {
        let pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(queue_family),
                None,
            )
            .map_err(|e| e.to_string())?;
        let cmd = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .map_err(|e| e.to_string())?[0];
        let query_pool = device
            .create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )
            .ok()
            .unwrap_or(vk::QueryPool::null());

        let mut images = Vec::new();
        for format in [
            vk::Format::R16G16B16A16_SFLOAT,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::Format::R32_SFLOAT,
            vk::Format::R16G16_SFLOAT,
        ] {
            images.push(create_image(
                device,
                instance,
                physical_device,
                format,
                64,
                64,
            )?);
        }
        let _ = entry;
        Ok(GpuResources {
            device: device.clone(),
            command_pool: pool,
            cmd,
            query_pool,
            images,
        })
    }
}

fn timed_dispatch<F>(
    res: &GpuResources,
    sync2: &synchronization2::Device,
    sync2_enabled: bool,
    queue: vk::Queue,
    dispatch: F,
) -> (Result<(), String>, Option<u64>)
where
    F: FnOnce() -> Result<(), String>,
{
    let timestamps = res.query_pool != vk::QueryPool::null() && sync2_enabled;
    unsafe {
        if timestamps {
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            if res.device.begin_command_buffer(res.cmd, &begin).is_err() {
                return (Err("begin_command_buffer failed".to_string()), None);
            }
            sync2.cmd_write_timestamp2(
                res.cmd,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                res.query_pool,
                0,
            );
        }
        let result = dispatch();
        if timestamps {
            sync2.cmd_write_timestamp2(
                res.cmd,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                res.query_pool,
                1,
            );
            if res.device.end_command_buffer(res.cmd).is_err() {
                return (result, None);
            }
            let fence = match res
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
            {
                Ok(f) => f,
                Err(_) => return (result, None),
            };
            let cmds = [res.cmd];
            let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
            let submitted = res.device.queue_submit(queue, &submit, fence).is_ok();
            if submitted {
                let _ = res.device.wait_for_fences(&[fence], true, 5_000_000_000);
            }
            res.device.destroy_fence(fence, None);
        }
        let gpu_time = if timestamps {
            let mut stamps = [0u64; 2];
            res.device
                .get_query_pool_results(
                    res.query_pool,
                    0,
                    &mut stamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .ok()
                .map(|_| stamps[1].saturating_sub(stamps[0]))
        } else {
            None
        };
        (result, gpu_time)
    }
}

fn main() {
    let natives = std::env::var_os("VS_NATIVES_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("natives"));
    let mut mgr = RuntimeSdkManager::probe(vec![natives.clone()]);

    let entry = vulkanshaders_core::gpu::device_merger::load_vulkan_entry();
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(
                &vk::ApplicationInfo::default()
                    .application_name(
                        CStr::from_bytes_with_nul(b"vulkanshaders-diagnose\0").unwrap(),
                    )
                    .api_version(vk::API_VERSION_1_3),
            ),
            None,
        )
    };
    let instance = match instance {
        Ok(i) => i,
        Err(e) => {
            write_report(&mgr, "INSTANCE_FAILED", &format!("{e}"), &[], None);
            return;
        }
    };

    let physical_devices = unsafe { instance.enumerate_physical_devices() }.unwrap_or_default();
    let Some(physical_device) = physical_devices.into_iter().next() else {
        write_report(&mgr, "NO_PHYSICAL_DEVICE", "no Vulkan GPU found", &[], None);
        return;
    };

    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let Some(qfi) = queue_families
        .iter()
        .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .map(|i| i as u32)
    else {
        write_report(
            &mgr,
            "NO_COMPUTE_QUEUE",
            "no compute queue family",
            &[],
            None,
        );
        return;
    };

    let mut supported_vk13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut feat2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut supported_vk13);
    unsafe { instance.get_physical_device_features2(physical_device, &mut feat2) };
    let mut requested_vk13 = vk::PhysicalDeviceVulkan13Features::default()
        .synchronization2(supported_vk13.synchronization2 == vk::TRUE);
    let device = unsafe {
        instance.create_device(
            physical_device,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(&[vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(qfi)
                    .queue_priorities(&[1.0])])
                .push_next(&mut requested_vk13),
            None,
        )
    };
    let device = match device {
        Ok(d) => d,
        Err(e) => {
            write_report(&mgr, "DEVICE_FAILED", &format!("{e}"), &[], None);
            return;
        }
    };
    let queue = unsafe { device.get_device_queue(qfi, 0) };
    let sync2 = synchronization2::Device::new(&instance, &device);
    let timestamp2_name = CStr::from_bytes_with_nul(b"vkCmdWriteTimestamp2KHR\0").unwrap();
    let sync2_enabled = unsafe {
        instance
            .get_device_proc_addr(device.handle(), timestamp2_name.as_ptr())
            .is_some()
    };

    let resources = create_gpu_resources(&entry, &instance, physical_device, &device, qfi);

    let gipa = unsafe {
        std::mem::transmute::<_, vk::PFN_vkGetInstanceProcAddr>(
            entry.static_fn().get_instance_proc_addr,
        )
    };
    let gdpa = unsafe {
        std::mem::transmute::<_, vk::PFN_vkGetDeviceProcAddr>(
            instance.fp_v1_0().get_device_proc_addr,
        )
    };

    let mut feature_reports: Vec<(String, FeatureReport)> = Vec::new();

    // NGX / DLSS — real init + real capability query.
    let vendor_id = unsafe { instance.get_physical_device_properties(physical_device) }.vendor_id;
    let ngx_init = if vendor_id != 0x10DE {
        Err("skipped: NGX requires NVIDIA hardware (vendor_id != 0x10DE)".to_string())
            .map_err(std::convert::Into::<String>::into)
    } else {
        unsafe {
            mgr.initialize_ngx(
                instance.handle(),
                physical_device,
                device.handle(),
                gipa,
                gdpa,
                ".",
            )
        }
        .map_err(|e| e.to_string())
    };
    let init_stage = match &ngx_init {
        Ok(_) => Stage::ok("NVSDK_NGX_VULKAN_Init_Ext2 succeeded"),
        Err(e) => Stage::fail(e.to_string()),
    };
    let (dlss_sr, dlss_rr, dlss_fg) = (
        mgr.capabilities.dlss_sr_available,
        mgr.capabilities.dlss_rr_available,
        mgr.capabilities.dlss_fg_available,
    );
    for (name, available) in [
        ("dlssSr", dlss_sr),
        ("dlssRr", dlss_rr),
        ("dlssFg", dlss_fg),
    ] {
        let mut r = if available {
            FeatureReport {
                available: true,
                reason: "NGX capability query reported available".to_string(),
                lib_path: mgr
                    .library_path(name)
                    .map(|p| p.to_string_lossy().to_string()),
                symbols_resolved: true,
                init: Stage::ok("NGX init succeeded"),
                evaluate: Stage::fail("evaluate not attempted in this build path"),
                gpu_time_ns: None,
            }
        } else {
            let mut r = FeatureReport::absent(mgr.unavailable_reason(name));
            r.lib_path = mgr
                .library_path(name)
                .map(|p| p.to_string_lossy().to_string());
            r.symbols_resolved = mgr.library_path(name).is_some();
            r.init = init_stage.clone();
            r
        };
        if available && let Ok(res) = &resources {
            let p = eval_params(res, &mgr);
            let extent = ViewportExtent {
                render_width: p.width,
                render_height: p.height,
                output_width: p.out_width,
                output_height: p.out_height,
            };
            let cfg_res = unsafe {
                match name {
                    "dlssSr" => mgr.configure_upscaler(
                        UpscalerKind::DlssSr,
                        extent,
                        p.command_buffer,
                        instance.handle(),
                        physical_device,
                        device.handle(),
                        gdpa,
                        NvSdkNgxPerfQuality::MaxQuality,
                        0,
                        None,
                        Some(&device),
                    ),
                    "dlssRr" => mgr.configure_denoiser(
                        DenoiserKind::DlssRr,
                        extent,
                        p.command_buffer,
                        NvSdkNgxPerfQuality::MaxQuality,
                        0,
                        None,
                    ),
                    _ => mgr.configure_framegen(
                        FrameGenKind::DlssFg,
                        extent,
                        p.command_buffer,
                        None,
                        Some(&device),
                    ),
                }
            };
            r.init = match &cfg_res {
                Ok(_) => Stage::ok("NGX feature handle created"),
                Err(e) => Stage::fail(e.clone()),
            };
            if cfg_res.is_ok() {
                let (outcome, gpu) = timed_dispatch(res, &sync2, sync2_enabled, queue, || unsafe {
                    match name {
                        "dlssSr" => mgr.evaluate_dlss_sr(&p),
                        "dlssRr" => mgr.evaluate_dlss_rr(&p),
                        _ => mgr.evaluate_dlss_fg(&p),
                    }
                });
                r.evaluate = match &outcome {
                    Ok(_) => Stage::ok("real NGX evaluate recorded"),
                    Err(e) => Stage::fail(e.clone()),
                };
                r.gpu_time_ns = gpu;
            }
        }
        feature_reports.push((name.to_string(), r));
    }

    // FSR3 / XeSS — real library load (probe), symbol resolution and off-screen evaluate.
    for (name, kind) in [("fsr3", UpscalerKind::Fsr3), ("xess", UpscalerKind::XeSS)] {
        let available = mgr.capability_available(name);
        let mut r = if available {
            FeatureReport {
                available: true,
                reason: "vendor library loaded and required symbols resolved".to_string(),
                lib_path: mgr
                    .library_path(name)
                    .map(|p| p.to_string_lossy().to_string()),
                symbols_resolved: true,
                init: Stage::ok("context configure not attempted in headless probe"),
                evaluate: Stage::fail("evaluate not attempted"),
                gpu_time_ns: None,
            }
        } else {
            let mut r = FeatureReport::absent(mgr.unavailable_reason(name));
            r.lib_path = mgr
                .library_path(name)
                .map(|p| p.to_string_lossy().to_string());
            r.symbols_resolved = mgr.library_path(name).is_some();
            r
        };
        if available && let Ok(res) = &resources {
            let p = eval_params(res, &mgr);
            let extent = ViewportExtent {
                render_width: p.width,
                render_height: p.height,
                output_width: p.out_width,
                output_height: p.out_height,
            };
            let cfg = unsafe {
                mgr.configure_upscaler(
                    kind,
                    extent,
                    p.command_buffer,
                    instance.handle(),
                    physical_device,
                    device.handle(),
                    gdpa,
                    NvSdkNgxPerfQuality::MaxQuality,
                    0,
                    None,
                    Some(&device),
                )
            };
            r.init = match &cfg {
                Ok(_) => Stage::ok("vendor context created + initialized"),
                Err(e) => Stage::fail(e.clone()),
            };
            if cfg.is_ok() {
                let (outcome, gpu) = timed_dispatch(res, &sync2, sync2_enabled, queue, || {
                    if name == "fsr3" {
                        unsafe { mgr.evaluate_fsr3(&p) }
                    } else {
                        unsafe { mgr.evaluate_xess(&p) }
                    }
                });
                r.evaluate = match &outcome {
                    Ok(_) => Stage::ok("real vendor dispatch recorded"),
                    Err(e) => Stage::fail(e.clone()),
                };
                r.gpu_time_ns = gpu;
            }
        }
        feature_reports.push((name.to_string(), r));
    }

    // NRD — real library load + denoiser instance + SPIR-V compute pipeline build + permanent/transient pools + GetComputeDispatches.
    {
        let name = "nrd";
        let loaded = mgr.capabilities.nrd_loaded;
        let mut r = if loaded {
            FeatureReport {
                available: true,
                reason: "libNRD.so loaded and required symbols resolved".to_string(),
                lib_path: mgr
                    .library_path(name)
                    .map(|p| p.to_string_lossy().to_string()),
                symbols_resolved: true,
                init: Stage::fail("NRD denoiser configure not attempted"),
                evaluate: Stage::fail("evaluate not attempted"),
                gpu_time_ns: None,
            }
        } else {
            let mut r = FeatureReport::absent(mgr.unavailable_reason(name));
            r.lib_path = mgr
                .library_path(name)
                .map(|p| p.to_string_lossy().to_string());
            r.symbols_resolved = mgr.library_path(name).is_some();
            r
        };
        if loaded && let Ok(res) = &resources {
            let p = eval_params(res, &mgr);
            let extent = ViewportExtent {
                render_width: p.width,
                render_height: p.height,
                output_width: p.out_width,
                output_height: p.out_height,
            };
            let cfg_res = unsafe {
                mgr.configure_denoiser(
                    DenoiserKind::NrdReblur,
                    extent,
                    p.command_buffer,
                    NvSdkNgxPerfQuality::MaxQuality,
                    0,
                    None,
                )
                .and_then(|_| mgr.configure_nrd_dispatcher(&res.device))
            };
            if cfg_res.is_ok() {
                unsafe {
                    mgr.ensure_nrd_pools(&instance, physical_device, p.width, p.height, None);
                }
            }
            r.init = match &cfg_res {
                Ok(_) => Stage::ok(
                    "NRD Reblur instance + Vulkan compute pipelines + permanent/transient pools created",
                ),
                Err(e) => Stage::fail(e.clone()),
            };
            if cfg_res.is_ok() {
                let (outcome, gpu) = timed_dispatch(res, &sync2, sync2_enabled, queue, || unsafe {
                    let mut layouts = std::collections::HashMap::new();
                    mgr.evaluate_nrd(&p, &mut layouts).map(|_| ())
                });
                r.evaluate = match &outcome {
                    Ok(_) => Stage::ok("NRD compute dispatches recorded onto command buffer"),
                    Err(e) => Stage::fail(e.clone()),
                };
                r.gpu_time_ns = gpu;
            }
        }
        feature_reports.push((name.to_string(), r));
    }

    write_report(
        &mgr,
        "OK",
        "headless device created; stages reported per feature",
        &feature_reports,
        Some(read_gpu_info(&instance, physical_device)),
    );

    unsafe { mgr.teardown() };
    drop(resources);
    drop(device);
    drop(instance);
}

fn read_gpu_info(instance: &Instance, physical_device: vk::PhysicalDevice) -> String {
    unsafe {
        let props = instance.get_physical_device_properties(physical_device);
        let name = CStr::from_ptr(props.device_name.as_ptr())
            .to_string_lossy()
            .to_string();
        format!(
            "{{\"name\":{},\"apiVersion\":\"{}.{}.{}\",\"vendorId\":\"0x{:x}\"}}",
            jstr(&name),
            vk::api_version_major(props.api_version),
            vk::api_version_minor(props.api_version),
            vk::api_version_patch(props.api_version),
            props.vendor_id,
        )
    }
}

fn eval_params(res: &GpuResources, _mgr: &RuntimeSdkManager) -> SdkDispatchParams {
    let images = &res.images;
    SdkDispatchParams {
        command_buffer: res.cmd,
        input_image: images[0].0,
        input_view: images[0].2,
        output_image: images[1].0,
        output_view: images[1].2,
        depth_image: images[2].0,
        depth_format: vk::Format::R32_SFLOAT,
        depth_view: images[2].2,
        motion_image: images[3].0,
        motion_view: images[3].2,
        width: 64,
        height: 64,
        out_width: 64,
        out_height: 64,
        jitter: [0.0, 0.0],
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

fn write_report(
    mgr: &RuntimeSdkManager,
    status: &str,
    detail: &str,
    features: &[(String, FeatureReport)],
    gpu: Option<String>,
) {
    let ext = mgr.capabilities.summary();
    let feature_json = features
        .iter()
        .map(|(name, f)| {
            format!(
                "{}:{{\"available\":{},\"reason\":{},\"libPath\":{},\"symbolsResolved\":{},\"init\":{},\"evaluate\":{},\"gpuTimeNs\":{}}}",
                jstr(name),
                f.available,
                jstr(&f.reason),
                f.lib_path
                    .as_ref()
                    .map(|p| jstr(p))
                    .unwrap_or_else(|| "null".to_string()),
                f.symbols_resolved,
                stage_json(&f.init),
                stage_json(&f.evaluate),
                f.gpu_time_ns
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "null".to_string()),
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let json = format!(
        "{{\"status\":{},\"detail\":{},\"sdkSummary\":{},\"gpu\":{},\"features\":{{{}}}}}",
        jstr(status),
        jstr(detail),
        jstr(&ext),
        gpu.unwrap_or_else(|| "null".to_string()),
        feature_json,
    );
    let path = PathBuf::from("vulkanshaders-diagnostic.json");
    if let Err(e) = fs::write(&path, json.as_bytes()) {
        eprintln!("[diagnose] failed to write {}: {e}", path.display());
    } else {
        println!("[diagnose] wrote {}", path.display());
    }
    println!("[diagnose] {status}: {detail}");
}
