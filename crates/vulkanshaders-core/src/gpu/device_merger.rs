use crate::gpu::{DeviceCapabilities, GpuArchitecture, GpuTierClass};
use ash::{
    Entry, Instance, StaticFn,
    vk::{self, Handle},
};
use parking_lot::Mutex;
use std::{
    collections::{HashMap, HashSet},
    ffi::{CStr, CString, c_char, c_void},
    path::Path,
    ptr,
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

/// Stores the `vkGetInstanceProcAddr` function pointer supplied by LWJGL (`VK.getFunctionProvider()`),
/// avoiding loader mismatches with `Entry::linked()`.
static LWJGL_GET_INSTANCE_PROC_ADDR: AtomicU64 = AtomicU64::new(0);

/// Real hardware/driver facts gathered from `vkGetPhysicalDeviceProperties2` and
/// `vkEnumerate*ExtensionProperties`. No field is ever guessed or defaulted to `true`.
#[derive(Debug, Clone, Default)]
pub struct DeviceProbeInfo {
    pub gpu_name: String,
    pub api_version: String,
    pub driver_name: String,
    pub driver_version: String,
    pub instance_extensions: Vec<String>,
    pub device_extensions: Vec<String>,
    pub hdr_surface_supported: bool,
    pub has_memory_budget: bool,
    pub is_moltenvk: bool,
    pub tier_policy_reason: String,
}

#[derive(Default)]
struct MergerRegistry {
    instances: Vec<u64>,
    physical_device_to_instance: HashMap<u64, u64>,
    device_capabilities: HashMap<u64, DeviceCapabilities>,
    last_merged_caps: Option<DeviceCapabilities>,
    last_physical_device: Option<u64>,
    last_tier_policy: Option<PlatformTierPolicy>,
    probe_info: DeviceProbeInfo,
}

static MERGER_REGISTRY: OnceLock<Mutex<MergerRegistry>> = OnceLock::new();

fn registry() -> &'static Mutex<MergerRegistry> {
    MERGER_REGISTRY.get_or_init(|| Mutex::new(MergerRegistry::default()))
}

/// Registers `vkGetInstanceProcAddr` obtained from LWJGL's `FunctionProvider`.
pub fn set_instance_proc_addr(pfn_raw: u64) {
    if pfn_raw != 0 {
        LWJGL_GET_INSTANCE_PROC_ADDR.store(pfn_raw, Ordering::Release);
    }
}

/// Loads an `ash::Entry` using LWJGL's `vkGetInstanceProcAddr` when available, or falls back to
/// `Entry::linked()`.
pub fn load_vulkan_entry() -> Entry {
    let raw = LWJGL_GET_INSTANCE_PROC_ADDR.load(Ordering::Acquire);
    if raw != 0 {
        let pfn: vk::PFN_vkGetInstanceProcAddr = unsafe { std::mem::transmute(raw as usize) };
        let static_fn = StaticFn {
            get_instance_proc_addr: pfn,
        };
        unsafe { Entry::from_static_fn(static_fn) }
    } else {
        Entry::linked()
    }
}

pub fn register_instance_handle(instance_raw: u64) {
    if instance_raw == 0 {
        return;
    }
    let entry = load_vulkan_entry();
    let vk_instance = vk::Instance::from_raw(instance_raw);
    let instance = unsafe { Instance::load(entry.static_fn(), vk_instance) };
    let phys_devs = unsafe { instance.enumerate_physical_devices().unwrap_or_default() };

    let mut reg = registry().lock();
    if !reg.instances.contains(&instance_raw) {
        reg.instances.push(instance_raw);
    }
    for pd in phys_devs {
        reg.physical_device_to_instance
            .insert(pd.as_raw(), instance_raw);
    }
}

pub fn resolve_instance_for_physical_device(
    entry: &Entry,
    physical_device_raw: u64,
) -> Option<u64> {
    {
        let reg = registry().lock();
        if let Some(&inst) = reg.physical_device_to_instance.get(&physical_device_raw) {
            return Some(inst);
        }
    }

    // Probe all known instances (in reverse creation order) to find which one owns `physical_device_raw`
    let candidate_instances: Vec<u64> = {
        let reg = registry().lock();
        reg.instances.iter().rev().copied().collect()
    };

    for inst_raw in &candidate_instances {
        let vk_inst = vk::Instance::from_raw(*inst_raw);
        let instance = unsafe { Instance::load(entry.static_fn(), vk_inst) };
        if let Ok(phys_list) = unsafe { instance.enumerate_physical_devices() } {
            let mut reg = registry().lock();
            for pd in &phys_list {
                reg.physical_device_to_instance
                    .insert(pd.as_raw(), *inst_raw);
            }
            if phys_list
                .iter()
                .any(|pd| pd.as_raw() == physical_device_raw)
            {
                return Some(*inst_raw);
            }
        }
    }

    candidate_instances.first().copied()
}

pub fn capabilities_for_device(vk_device_raw: u64) -> Option<DeviceCapabilities> {
    let reg = registry().lock();
    reg.device_capabilities
        .get(&vk_device_raw)
        .copied()
        .or(reg.last_merged_caps)
}

pub fn last_merged_capabilities() -> Option<DeviceCapabilities> {
    registry().lock().last_merged_caps
}

const DESIRED_INSTANCE_EXTENSIONS: &[&CStr] = &[
    vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME,
    vk::EXT_SWAPCHAIN_COLORSPACE_NAME,
    vk::KHR_PORTABILITY_ENUMERATION_NAME,
];

/// `VK_EXT_ray_tracing_invocation_reorder` extension name and official Vulkan spec constants.
pub const VK_EXT_RAY_TRACING_INVOCATION_REORDER_NAME: &CStr =
    c"VK_EXT_ray_tracing_invocation_reorder";
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT: i32 =
    1_000_581_000;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_EXT: i32 =
    1_000_581_001;
pub const VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER: i32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
struct VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    ray_tracing_invocation_reorder: vk::Bool32,
}

impl Default for VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT,
            ),
            p_next: ptr::null_mut(),
            ray_tracing_invocation_reorder: vk::FALSE,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkPhysicalDeviceRayTracingInvocationReorderPropertiesEXT {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    ray_tracing_invocation_reorder_reordering_hint: i32,
    max_shader_binding_table_record_index: u32,
}

impl Default for VkPhysicalDeviceRayTracingInvocationReorderPropertiesEXT {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_EXT,
            ),
            p_next: ptr::null_mut(),
            ray_tracing_invocation_reorder_reordering_hint: 0,
            max_shader_binding_table_record_index: 0,
        }
    }
}

/// Returns `true` only when a Shader Execution Reordering extension (`EXT` or `NV`) is supported
/// AND the driver reports hardware reordering (`REORDER_MODE_REORDER = 1`).
pub const fn evaluate_ser_activation(
    _arch: GpuArchitecture,
    ser_nv_supported: bool,
    ser_nv_hint: i32,
    ser_ext_supported: bool,
    ser_ext_hint: i32,
) -> bool {
    let ext_hw =
        ser_ext_supported && ser_ext_hint == VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER;
    let nv_hw = ser_nv_supported && ser_nv_hint == VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER;
    ext_hw || nv_hw
}

const DESIRED_DEVICE_EXTENSIONS: &[&CStr] = &[
    vk::KHR_ACCELERATION_STRUCTURE_NAME,
    vk::KHR_RAY_TRACING_PIPELINE_NAME,
    vk::KHR_RAY_QUERY_NAME,
    vk::KHR_DEFERRED_HOST_OPERATIONS_NAME,
    vk::KHR_SPIRV_1_4_NAME,
    vk::KHR_SYNCHRONIZATION2_NAME,
    vk::KHR_SHADER_ATOMIC_INT64_NAME,
    // Shader Execution Reordering (Ada Lovelace / Blackwell / cross-vendor EXT) and Opacity Micromap.
    vk::NV_RAY_TRACING_INVOCATION_REORDER_NAME,
    VK_EXT_RAY_TRACING_INVOCATION_REORDER_NAME,
    vk::EXT_OPACITY_MICROMAP_NAME,
    // Static, spec-derived NGX device extension names (documented by NVIDIA's NGX Vulkan
    // headers). They are still enabled ONLY when the driver really offers them, see the
    // `offered_exts.contains(...)` gate below. No NGX runtime code runs to produce these names.
    vk::KHR_PUSH_DESCRIPTOR_NAME,
    // NGX Ray Reconstruction imports CUDA kernels through this extension.
    vk::NVX_BINARY_IMPORT_NAME,
    vk::NVX_IMAGE_VIEW_HANDLE_NAME,
    vk::EXT_DESCRIPTOR_INDEXING_NAME,
    vk::KHR_BUFFER_DEVICE_ADDRESS_NAME,
    // Live VRAM usage/budget telemetry for OOM protection.
    vk::EXT_MEMORY_BUDGET_NAME,
    // Required by Vulkan spec whenever offered by a physical device (e.g. macOS MoltenVK).
    c"VK_KHR_portability_subset",
];

/// Optional opt-in for a LIVE `NVSDK_NGX_VULKAN_GetFeature*ExtensionRequirements` query during
/// VkInstance/VkDevice creation. Default OFF: calling into libnvidia-ngx.so.1 at this point crashes
/// on some driver/loader combinations (observed SIGSEGV inside libc during vkCreateDevice). Set
/// `VS_CORE_NGX_QUERY_EXTENSIONS=1` only on machines where this is known to work. Even then every
/// returned name is gated against the real `vkEnumerate*ExtensionProperties` result.
fn live_ngx_extension_query_enabled() -> bool {
    matches!(
        std::env::var("VS_CORE_NGX_QUERY_EXTENSIONS").as_deref(),
        Ok("1")
    )
}

/// Intercepts `VK12.vkCreateInstance` from `RenderPearl` (`VulkanInstanceMixin`) to upgrade
/// `apiVersion` to `>= 1.3`, deduplicate instance extensions/layers, enable
/// `VK_EXT_swapchain_colorspace` when available, and index the created `VkInstance` + its
/// `VkPhysicalDevice` handles in `MERGER_REGISTRY`.
///
/// # Safety
/// `create_info_ptr` and `out_instance_ptr` must be valid pointers passed to `vkCreateInstance`.
pub unsafe fn create_merged_instance(
    create_info_ptr: *const vk::InstanceCreateInfo<'_>,
    allocator_ptr: *const vk::AllocationCallbacks<'_>,
    out_instance_ptr: *mut vk::Instance,
) -> i32 {
    if create_info_ptr.is_null() || out_instance_ptr.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED.as_raw();
    }

    let entry = load_vulkan_entry();
    let available: HashSet<CString> = unsafe {
        entry
            .enumerate_instance_extension_properties(None)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|prop| {
                CStr::from_ptr(prop.extension_name.as_ptr())
                    .to_owned()
                    .into()
            })
            .collect()
    };

    let base_info = unsafe { *create_info_ptr };
    let mut app_info = if !base_info.p_application_info.is_null() {
        unsafe { *base_info.p_application_info }
    } else {
        vk::ApplicationInfo::default()
    };
    app_info.s_type = vk::StructureType::APPLICATION_INFO;
    app_info.api_version = app_info.api_version.max(vk::API_VERSION_1_3);

    let mut merged_exts: Vec<CString> = Vec::new();
    let mut seen: HashSet<CString> = HashSet::new();

    if base_info.enabled_extension_count > 0 && !base_info.pp_enabled_extension_names.is_null() {
        let slice = unsafe {
            std::slice::from_raw_parts(
                base_info.pp_enabled_extension_names,
                base_info.enabled_extension_count as usize,
            )
        };
        for &ptr_name in slice {
            if !ptr_name.is_null() {
                let cstr = unsafe { CStr::from_ptr(ptr_name).to_owned() };
                if seen.insert(cstr.clone()) {
                    merged_exts.push(cstr);
                }
            }
        }
    }

    for &desired in DESIRED_INSTANCE_EXTENSIONS {
        let owned = desired.to_owned();
        if available.contains(&owned) && seen.insert(owned.clone()) {
            merged_exts.push(owned);
        }
    }

    // NGX instance requirements are already covered by the static DESIRED_INSTANCE_EXTENSIONS
    // list above (KHR_get_physical_device_properties2). A LIVE NGX query is strictly opt-in via
    // VS_CORE_NGX_QUERY_EXTENSIONS=1 and is never executed by default — it can segfault the
    // process during VkCreateInstance/VkCreateDevice on NVIDIA drivers.
    if live_ngx_extension_query_enabled() {
        for name in ngx_required_instance_extensions() {
            let Ok(owned) = CString::new(name) else {
                continue;
            };
            if available.contains(&owned) && seen.insert(owned.clone()) {
                merged_exts.push(owned);
            }
        }
    }

    let ext_ptrs: Vec<*const c_char> = merged_exts.iter().map(|s| s.as_ptr()).collect();
    let mut patched_info = base_info;
    if seen.contains(vk::KHR_PORTABILITY_ENUMERATION_NAME) {
        patched_info.flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
    }
    patched_info.p_application_info = &app_info;
    patched_info.enabled_extension_count = ext_ptrs.len() as u32;
    patched_info.pp_enabled_extension_names = ext_ptrs.as_ptr();

    let alloc_ref = unsafe { allocator_ptr.as_ref() };
    match unsafe { entry.create_instance(&patched_info, alloc_ref) } {
        Ok(instance) => {
            let handle_raw = instance.handle().as_raw();
            unsafe {
                *out_instance_ptr = instance.handle();
            }
            register_instance_handle(handle_raw);
            {
                let mut reg = registry().lock();
                reg.probe_info.instance_extensions = available
                    .iter()
                    .map(|c| c.to_string_lossy().to_string())
                    .collect();
                reg.probe_info.hdr_surface_supported = false;
            }
            vk::Result::SUCCESS.as_raw()
        }
        Err(err) => err.as_raw(),
    }
}

/// Intercepts `VK12.vkCreateDevice` from `RenderPearl` (`VulkanBackendMixin`) to enable optional
/// Ray Tracing (`VK_KHR_acceleration_structure`, `VK_KHR_ray_query`, `VK_KHR_ray_tracing_pipeline`),
/// `bufferDeviceAddress`, `descriptorIndexing`, `synchronization2`, `dynamicRendering`,
/// `shaderInt64`, `shaderBufferInt64Atomics`, and `VK_NV_low_latency2` features when supported by
/// the physical GPU.
///
/// Strictly obeys Vulkan `pNext` uniqueness rules (`VUID-VkDeviceCreateInfo-pNext-pNext`,
/// `VUID-VkDeviceCreateInfo-pNext-02829`, `VUID-VkDeviceCreateInfo-pNext-06532`) by merging bits
/// into existing `VkPhysicalDeviceVulkan12Features` / `VkPhysicalDeviceVulkan13Features` or
/// standalone feature structs in `pNext` without introducing duplicate `sType` nodes.
///
/// # Safety
/// `physical_device_raw`, `create_info_ptr`, and `out_device_ptr` must be valid Vulkan arguments.
pub unsafe fn create_merged_device(
    physical_device_raw: u64,
    create_info_ptr: *const vk::DeviceCreateInfo<'_>,
    allocator_ptr: *const vk::AllocationCallbacks<'_>,
    out_device_ptr: *mut vk::Device,
) -> i32 {
    if physical_device_raw == 0 || create_info_ptr.is_null() || out_device_ptr.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED.as_raw();
    }
    let entry = load_vulkan_entry();
    let Some(instance_raw) = resolve_instance_for_physical_device(&entry, physical_device_raw)
    else {
        return vk::Result::ERROR_INITIALIZATION_FAILED.as_raw();
    };
    let vk_instance = vk::Instance::from_raw(instance_raw);
    let instance = unsafe { Instance::load(entry.static_fn(), vk_instance) };
    let physical_device = vk::PhysicalDevice::from_raw(physical_device_raw);

    let offered_exts: HashSet<CString> = unsafe {
        instance
            .enumerate_device_extension_properties(physical_device)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|prop| {
                CStr::from_ptr(prop.extension_name.as_ptr())
                    .to_owned()
                    .into()
            })
            .collect()
    };

    // Real GPU / driver identity via vkGetPhysicalDeviceProperties2 (+ driver/RT/SER pNext).
    let offers_rt_pipe = offered_exts.contains(vk::KHR_RAY_TRACING_PIPELINE_NAME);
    let offers_ser_nv = offered_exts.contains(vk::NV_RAY_TRACING_INVOCATION_REORDER_NAME);
    let offers_ser_ext = offered_exts.contains(VK_EXT_RAY_TRACING_INVOCATION_REORDER_NAME);

    let mut driver_props = vk::PhysicalDeviceDriverProperties::default();
    let mut rt_pipe_props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
    let mut ser_nv_props = vk::PhysicalDeviceRayTracingInvocationReorderPropertiesNV::default();
    let mut ser_ext_props = VkPhysicalDeviceRayTracingInvocationReorderPropertiesEXT::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut driver_props);
    if offers_rt_pipe {
        props2 = props2.push_next(&mut rt_pipe_props);
    }
    if offers_ser_nv {
        props2 = props2.push_next(&mut ser_nv_props);
    }
    if offers_ser_ext {
        ser_ext_props.p_next = props2.p_next;
        props2.p_next = (&mut ser_ext_props as *mut _) as *mut c_void;
    }
    unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };
    let raw_props = props2.properties;
    let gpu_name = unsafe { CStr::from_ptr(raw_props.device_name.as_ptr()) }
        .to_string_lossy()
        .to_string();

    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let vram_bytes: u64 = mem_props.memory_heaps[..mem_props.memory_heap_count as usize]
        .iter()
        .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|heap| heap.size)
        .sum();
    let gpu_arch = GpuArchitecture::classify(raw_props.vendor_id, raw_props.device_id, &gpu_name);
    let gpu_tier_class =
        GpuTierClass::classify(gpu_arch, raw_props.device_type, vram_bytes, &gpu_name);
    let api_version = format!(
        "{}.{}.{}",
        vk::api_version_major(raw_props.api_version),
        vk::api_version_minor(raw_props.api_version),
        vk::api_version_patch(raw_props.api_version)
    );
    let driver_name = unsafe { CStr::from_ptr(driver_props.driver_name.as_ptr()) }
        .to_string_lossy()
        .to_string();
    let driver_version = unsafe { CStr::from_ptr(driver_props.driver_info.as_ptr()) }
        .to_string_lossy()
        .to_string();

    let base_info = unsafe { *create_info_ptr };
    let mut merged_exts: Vec<CString> = Vec::new();
    let mut seen: HashSet<CString> = HashSet::new();

    if base_info.enabled_extension_count > 0 && !base_info.pp_enabled_extension_names.is_null() {
        let slice = unsafe {
            std::slice::from_raw_parts(
                base_info.pp_enabled_extension_names,
                base_info.enabled_extension_count as usize,
            )
        };
        for &ptr_name in slice {
            if !ptr_name.is_null() {
                let cstr = unsafe { CStr::from_ptr(ptr_name).to_owned() };
                if seen.insert(cstr.clone()) {
                    merged_exts.push(cstr);
                }
            }
        }
    }

    for &desired in DESIRED_DEVICE_EXTENSIONS {
        let owned = desired.to_owned();
        if offered_exts.contains(&owned) && seen.insert(owned.clone()) {
            merged_exts.push(owned);
        }
    }

    // Low Latency 2 depends on a present-ID extension as well as timeline semaphores.
    if offered_exts.contains(vk::NV_LOW_LATENCY2_NAME)
        && seen.contains(vk::KHR_SWAPCHAIN_NAME)
        && (seen.contains(c"VK_KHR_present_id2") || offered_exts.contains(vk::KHR_PRESENT_ID_NAME))
    {
        if !seen.contains(c"VK_KHR_present_id2") && seen.insert(vk::KHR_PRESENT_ID_NAME.to_owned())
        {
            merged_exts.push(vk::KHR_PRESENT_ID_NAME.to_owned());
        }
        if seen.insert(vk::NV_LOW_LATENCY2_NAME.to_owned()) {
            merged_exts.push(vk::NV_LOW_LATENCY2_NAME.to_owned());
        }
    }

    // The NGX device extensions (VK_KHR_push_descriptor, VK_NVX_image_view_handle) are already in
    // the static DESIRED_DEVICE_EXTENSIONS list and are enabled only when the driver really offers
    // them. Calling into libnvidia-ngx.so.1 here is strictly opt-in (VS_CORE_NGX_QUERY_EXTENSIONS=1).
    if live_ngx_extension_query_enabled() {
        for name in ngx_required_device_extensions(vk_instance, physical_device) {
            let Ok(owned) = CString::new(name) else {
                continue;
            };
            if offered_exts.contains(&owned) && seen.insert(owned.clone()) {
                merged_exts.push(owned);
            }
        }
    }

    let has_accel = seen.contains(&vk::KHR_ACCELERATION_STRUCTURE_NAME.to_owned());
    let has_rq = seen.contains(&vk::KHR_RAY_QUERY_NAME.to_owned());
    let has_rt_pipe = seen.contains(&vk::KHR_RAY_TRACING_PIPELINE_NAME.to_owned());
    let has_low_latency2 = seen.contains(&vk::NV_LOW_LATENCY2_NAME.to_owned());
    let has_ser_nv = seen.contains(&vk::NV_RAY_TRACING_INVOCATION_REORDER_NAME.to_owned());
    let has_ser_ext = seen.contains(&VK_EXT_RAY_TRACING_INVOCATION_REORDER_NAME.to_owned());
    let has_omm = seen.contains(&vk::EXT_OPACITY_MICROMAP_NAME.to_owned());

    // Query supported feature structs on the physical device
    let mut supp_accel = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
    let mut supp_rq = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
    let mut supp_rt_pipe = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
    let mut supp_present_id = vk::PhysicalDevicePresentIdFeaturesKHR::default();
    let mut supp_ser_nv = vk::PhysicalDeviceRayTracingInvocationReorderFeaturesNV::default();
    let mut supp_ser_ext = VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT::default();
    let mut supp_omm = vk::PhysicalDeviceOpacityMicromapFeaturesEXT::default();
    let mut supp_vk12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut supp_vk13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut supp_feat2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut supp_vk12)
        .push_next(&mut supp_vk13);

    if has_accel {
        supp_feat2 = supp_feat2.push_next(&mut supp_accel);
    }
    if has_rq {
        supp_feat2 = supp_feat2.push_next(&mut supp_rq);
    }
    if has_rt_pipe {
        supp_feat2 = supp_feat2.push_next(&mut supp_rt_pipe);
    }
    if has_ser_nv {
        supp_feat2 = supp_feat2.push_next(&mut supp_ser_nv);
    }
    if has_omm {
        supp_feat2 = supp_feat2.push_next(&mut supp_omm);
    }
    let has_present_id_ext = seen.contains(vk::KHR_PRESENT_ID_NAME) || has_low_latency2;
    if has_present_id_ext {
        supp_feat2 = supp_feat2.push_next(&mut supp_present_id);
    }
    if has_ser_ext {
        supp_ser_ext.p_next = supp_feat2.p_next;
        supp_feat2.p_next = (&mut supp_ser_ext as *mut _) as *mut c_void;
    }

    unsafe {
        instance.get_physical_device_features2(physical_device, &mut supp_feat2);
    }
    let supp_core = supp_feat2.features;

    let supports_ser_nv = has_ser_nv && supp_ser_nv.ray_tracing_invocation_reorder == vk::TRUE;
    let supports_ser_ext = has_ser_ext && supp_ser_ext.ray_tracing_invocation_reorder == vk::TRUE;
    let ser_nv_hint = ser_nv_props
        .ray_tracing_invocation_reorder_reordering_hint
        .as_raw();
    let ser_ext_hint = ser_ext_props.ray_tracing_invocation_reorder_reordering_hint;
    let ser_reordering_active = evaluate_ser_activation(
        gpu_arch,
        supports_ser_nv,
        ser_nv_hint,
        supports_ser_ext,
        ser_ext_hint,
    );
    let supports_opacity_micromap = has_omm && supp_omm.micromap == vk::TRUE;

    let has_portability_subset = seen.contains(&c"VK_KHR_portability_subset".to_owned());
    let has_memory_budget = seen.contains(&vk::EXT_MEMORY_BUDGET_NAME.to_owned());
    let is_moltenvk = driver_props.driver_id == vk::DriverId::MOLTENVK
        || driver_name.to_ascii_lowercase().contains("moltenvk")
        || (has_portability_subset && cfg!(target_os = "macos"));
    let mut caps = DeviceCapabilities {
        vendor_id: raw_props.vendor_id,
        device_id: raw_props.device_id,
        device_type: raw_props.device_type,
        vram_bytes,
        gpu_arch,
        gpu_tier_class,
        buffer_device_address: supp_vk12.buffer_device_address == vk::TRUE,
        descriptor_indexing: supp_vk12.descriptor_indexing == vk::TRUE,
        acceleration_structure: has_accel && supp_accel.acceleration_structure == vk::TRUE,
        ray_query: has_rq && supp_rq.ray_query == vk::TRUE,
        ray_tracing_pipeline: has_rt_pipe && supp_rt_pipe.ray_tracing_pipeline == vk::TRUE,
        synchronization2: supp_vk13.synchronization2 == vk::TRUE,
        dynamic_rendering: supp_vk13.dynamic_rendering == vk::TRUE,
        low_latency2_nv: has_low_latency2,
        // Reflect instance-level VK_EXT_swapchain_colorspace availability or confirmed surface probe.
        hdr_swapchain_colorspace: {
            let reg = registry().lock();
            reg.probe_info.hdr_surface_supported
                || reg
                    .probe_info
                    .instance_extensions
                    .iter()
                    .any(|name| name == "VK_EXT_swapchain_colorspace")
        },
        shader_int64_atomics: supp_vk12.shader_buffer_int64_atomics == vk::TRUE,
        portability_subset: has_portability_subset,
        supports_ser_nv,
        supports_ser_ext,
        ser_reordering_active,
        supports_opacity_micromap,
        has_memory_budget,
        is_moltenvk,
        max_ray_recursion_depth: if has_rt_pipe {
            rt_pipe_props.max_ray_recursion_depth.max(1)
        } else {
            1
        },
    };

    // Clone any known foreign pNext feature structs into local stack copies so we never mutate
    // RenderPearl's `const` pNext chain in place.
    let mut cloned_feat2: Option<vk::PhysicalDeviceFeatures2<'_>> = None;
    let mut cloned_vk12: Option<vk::PhysicalDeviceVulkan12Features<'_>> = None;
    let mut cloned_vk13: Option<vk::PhysicalDeviceVulkan13Features<'_>> = None;
    let mut cloned_bda: Option<vk::PhysicalDeviceBufferDeviceAddressFeatures<'_>> = None;
    let mut cloned_atomic64: Option<vk::PhysicalDeviceShaderAtomicInt64Features<'_>> = None;
    let mut cloned_desc_idx: Option<vk::PhysicalDeviceDescriptorIndexingFeatures<'_>> = None;
    let mut cloned_scalar_block: Option<vk::PhysicalDeviceScalarBlockLayoutFeatures<'_>> = None;
    let mut cloned_timeline_sem: Option<vk::PhysicalDeviceTimelineSemaphoreFeatures<'_>> = None;
    let mut cloned_demote: Option<vk::PhysicalDeviceShaderDemoteToHelperInvocationFeatures<'_>> =
        None;
    let mut cloned_sync2: Option<vk::PhysicalDeviceSynchronization2Features<'_>> = None;
    let mut cloned_dyn_render: Option<vk::PhysicalDeviceDynamicRenderingFeatures<'_>> = None;
    let mut cloned_accel: Option<vk::PhysicalDeviceAccelerationStructureFeaturesKHR<'_>> = None;
    let mut cloned_rq: Option<vk::PhysicalDeviceRayQueryFeaturesKHR<'_>> = None;
    let mut cloned_rtp: Option<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR<'_>> = None;
    let mut cloned_pid: Option<vk::PhysicalDevicePresentIdFeaturesKHR<'_>> = None;
    let mut cloned_vk11: Option<vk::PhysicalDeviceVulkan11Features<'_>> = None;
    let mut cloned_multi_draw: Option<vk::PhysicalDeviceMultiDrawFeaturesEXT<'_>> = None;
    let mut cloned_vertex_divisor: Option<vk::PhysicalDeviceVertexAttributeDivisorFeaturesKHR<'_>> =
        None;
    let mut all_foreign_known = true;
    {
        let mut cur = base_info.p_next as *const vk::BaseInStructure<'_>;
        let mut guard_steps = 0usize;
        while !cur.is_null() && guard_steps < 64 {
            let stype = unsafe { (*cur).s_type };
            match stype {
                vk::StructureType::PHYSICAL_DEVICE_FEATURES_2 => {
                    cloned_feat2 =
                        Some(unsafe { *(cur as *const vk::PhysicalDeviceFeatures2<'_>) });
                }
                vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES => {
                    cloned_vk12 =
                        Some(unsafe { *(cur as *const vk::PhysicalDeviceVulkan12Features<'_>) });
                }
                vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES => {
                    cloned_vk13 =
                        Some(unsafe { *(cur as *const vk::PhysicalDeviceVulkan13Features<'_>) });
                }
                vk::StructureType::PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES => {
                    cloned_bda = Some(unsafe {
                        *(cur as *const vk::PhysicalDeviceBufferDeviceAddressFeatures<'_>)
                    });
                }
                vk::StructureType::PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES => {
                    cloned_atomic64 = Some(unsafe {
                        *(cur as *const vk::PhysicalDeviceShaderAtomicInt64Features<'_>)
                    });
                }
                vk::StructureType::PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES => {
                    cloned_desc_idx = Some(unsafe {
                        *(cur as *const vk::PhysicalDeviceDescriptorIndexingFeatures<'_>)
                    });
                }
                vk::StructureType::PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES => {
                    cloned_scalar_block = Some(unsafe {
                        *(cur as *const vk::PhysicalDeviceScalarBlockLayoutFeatures<'_>)
                    });
                }
                vk::StructureType::PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES => {
                    cloned_timeline_sem = Some(unsafe {
                        *(cur as *const vk::PhysicalDeviceTimelineSemaphoreFeatures<'_>)
                    });
                }
                vk::StructureType::PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES => {
                    cloned_demote = Some(unsafe {
                        *(cur as *const vk::PhysicalDeviceShaderDemoteToHelperInvocationFeatures<
                            '_,
                        >)
                    });
                }
                vk::StructureType::PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES => {
                    cloned_sync2 = Some(unsafe {
                        *(cur as *const vk::PhysicalDeviceSynchronization2Features<'_>)
                    });
                }
                vk::StructureType::PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES => {
                    cloned_dyn_render = Some(unsafe {
                        *(cur as *const vk::PhysicalDeviceDynamicRenderingFeatures<'_>)
                    });
                }
                vk::StructureType::PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR => {
                    cloned_accel = Some(unsafe {
                        *(cur as *const vk::PhysicalDeviceAccelerationStructureFeaturesKHR<'_>)
                    });
                }
                vk::StructureType::PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR => {
                    cloned_rq =
                        Some(unsafe { *(cur as *const vk::PhysicalDeviceRayQueryFeaturesKHR<'_>) });
                }
                vk::StructureType::PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR => {
                    cloned_rtp = Some(unsafe {
                        *(cur as *const vk::PhysicalDeviceRayTracingPipelineFeaturesKHR<'_>)
                    });
                }
                vk::StructureType::PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR => {
                    cloned_pid = Some(unsafe {
                        *(cur as *const vk::PhysicalDevicePresentIdFeaturesKHR<'_>)
                    });
                }
                vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_1_FEATURES => {
                    cloned_vk11 =
                        Some(unsafe { *(cur as *const vk::PhysicalDeviceVulkan11Features<'_>) });
                }
                vk::StructureType::PHYSICAL_DEVICE_MULTI_DRAW_FEATURES_EXT => {
                    cloned_multi_draw = Some(unsafe {
                        *(cur as *const vk::PhysicalDeviceMultiDrawFeaturesEXT<'_>)
                    });
                }
                vk::StructureType::PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_KHR => {
                    cloned_vertex_divisor = Some(unsafe {
                        *(cur as *const vk::PhysicalDeviceVertexAttributeDivisorFeaturesKHR<'_>)
                    });
                }
                _ => {
                    all_foreign_known = false;
                }
            }
            cur = unsafe { (*cur).p_next };
            guard_steps += 1;
        }
    }

    let mut head_p_next: *mut vk::BaseOutStructure<'_> = if all_foreign_known {
        let mut h: *mut vk::BaseOutStructure<'_> = std::ptr::null_mut();
        macro_rules! push_cloned {
            ($opt:expr) => {
                if let Some(ref mut node) = $opt {
                    node.p_next = h as *mut _;
                    h = (node as *mut _) as *mut vk::BaseOutStructure<'_>;
                }
            };
        }
        push_cloned!(cloned_pid);
        push_cloned!(cloned_vertex_divisor);
        push_cloned!(cloned_multi_draw);
        push_cloned!(cloned_rtp);
        push_cloned!(cloned_rq);
        push_cloned!(cloned_accel);
        push_cloned!(cloned_dyn_render);
        push_cloned!(cloned_sync2);
        push_cloned!(cloned_demote);
        push_cloned!(cloned_timeline_sem);
        push_cloned!(cloned_scalar_block);
        push_cloned!(cloned_desc_idx);
        push_cloned!(cloned_atomic64);
        push_cloned!(cloned_bda);
        push_cloned!(cloned_vk13);
        push_cloned!(cloned_vk12);
        push_cloned!(cloned_vk11);
        push_cloned!(cloned_feat2);
        h
    } else {
        base_info.p_next as *mut vk::BaseOutStructure<'_>
    };

    // The shader pack uses int64 buffer addresses and independent blending for the G-buffer MRTs.
    let mut local_core_features = if !base_info.p_enabled_features.is_null() {
        unsafe { *base_info.p_enabled_features }
    } else {
        vk::PhysicalDeviceFeatures::default()
    };
    let enabled_core = if all_foreign_known
        && let Some(feat2_node) = unsafe {
            find_stype_mut::<vk::PhysicalDeviceFeatures2<'_>>(
                head_p_next,
                vk::StructureType::PHYSICAL_DEVICE_FEATURES_2,
            )
        } {
        &mut feat2_node.features
    } else {
        &mut local_core_features
    };
    enabled_core.shader_int64 |= supp_core.shader_int64;
    enabled_core.independent_blend |= supp_core.independent_blend;

    // Check if caller chained individual Vulkan 1.2 feature structs (which forbid chaining Vulkan12Features)
    let has_individual_vk12_structs = unsafe {
        chain_contains_stype(
            head_p_next,
            vk::StructureType::PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
        ) || chain_contains_stype(
            head_p_next,
            vk::StructureType::PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
        ) || chain_contains_stype(
            head_p_next,
            vk::StructureType::PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
        ) || chain_contains_stype(
            head_p_next,
            vk::StructureType::PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES,
        ) || chain_contains_stype(
            head_p_next,
            vk::StructureType::PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES,
        )
    };

    let mut req_vk12 = vk::PhysicalDeviceVulkan12Features::default()
        .buffer_device_address(caps.buffer_device_address)
        .descriptor_indexing(caps.descriptor_indexing)
        .descriptor_binding_partially_bound(
            supp_vk12.descriptor_binding_partially_bound == vk::TRUE,
        )
        .descriptor_binding_sampled_image_update_after_bind(
            supp_vk12.descriptor_binding_sampled_image_update_after_bind == vk::TRUE,
        )
        .descriptor_binding_storage_image_update_after_bind(
            supp_vk12.descriptor_binding_storage_image_update_after_bind == vk::TRUE,
        )
        .descriptor_binding_update_unused_while_pending(
            supp_vk12.descriptor_binding_update_unused_while_pending == vk::TRUE,
        )
        .shader_sampled_image_array_non_uniform_indexing(
            supp_vk12.shader_sampled_image_array_non_uniform_indexing == vk::TRUE,
        )
        .runtime_descriptor_array(supp_vk12.runtime_descriptor_array == vk::TRUE)
        .scalar_block_layout(supp_vk12.scalar_block_layout == vk::TRUE)
        .timeline_semaphore(supp_vk12.timeline_semaphore == vk::TRUE)
        .shader_buffer_int64_atomics(supp_vk12.shader_buffer_int64_atomics == vk::TRUE)
        .shader_shared_int64_atomics(supp_vk12.shader_shared_int64_atomics == vk::TRUE);

    let mut req_bda = vk::PhysicalDeviceBufferDeviceAddressFeatures::default()
        .buffer_device_address(caps.buffer_device_address);
    let mut req_atomic64 = vk::PhysicalDeviceShaderAtomicInt64Features::default()
        .shader_buffer_int64_atomics(supp_vk12.shader_buffer_int64_atomics == vk::TRUE)
        .shader_shared_int64_atomics(supp_vk12.shader_shared_int64_atomics == vk::TRUE);
    let mut req_desc_idx = vk::PhysicalDeviceDescriptorIndexingFeatures::default()
        .descriptor_binding_partially_bound(
            supp_vk12.descriptor_binding_partially_bound == vk::TRUE,
        )
        .descriptor_binding_sampled_image_update_after_bind(
            supp_vk12.descriptor_binding_sampled_image_update_after_bind == vk::TRUE,
        )
        .descriptor_binding_storage_image_update_after_bind(
            supp_vk12.descriptor_binding_storage_image_update_after_bind == vk::TRUE,
        )
        .descriptor_binding_update_unused_while_pending(
            supp_vk12.descriptor_binding_update_unused_while_pending == vk::TRUE,
        )
        .shader_sampled_image_array_non_uniform_indexing(
            supp_vk12.shader_sampled_image_array_non_uniform_indexing == vk::TRUE,
        )
        .runtime_descriptor_array(supp_vk12.runtime_descriptor_array == vk::TRUE);
    let mut req_scalar_block = vk::PhysicalDeviceScalarBlockLayoutFeatures::default()
        .scalar_block_layout(supp_vk12.scalar_block_layout == vk::TRUE);
    let mut req_timeline_sem = vk::PhysicalDeviceTimelineSemaphoreFeatures::default()
        .timeline_semaphore(supp_vk12.timeline_semaphore == vk::TRUE);

    let has_vk12_in_chain = unsafe {
        chain_contains_stype(
            head_p_next,
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        )
    };
    if has_vk12_in_chain {
        if all_foreign_known
            && let Some(vk12) = unsafe {
                find_stype_mut::<vk::PhysicalDeviceVulkan12Features<'_>>(
                    head_p_next,
                    vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                )
            }
        {
            if caps.buffer_device_address {
                vk12.buffer_device_address = vk::TRUE;
            }
            if caps.descriptor_indexing {
                vk12.descriptor_indexing = vk::TRUE;
                vk12.descriptor_binding_partially_bound =
                    supp_vk12.descriptor_binding_partially_bound;
                vk12.descriptor_binding_sampled_image_update_after_bind =
                    supp_vk12.descriptor_binding_sampled_image_update_after_bind;
                vk12.descriptor_binding_storage_image_update_after_bind =
                    supp_vk12.descriptor_binding_storage_image_update_after_bind;
                vk12.descriptor_binding_update_unused_while_pending =
                    supp_vk12.descriptor_binding_update_unused_while_pending;
                vk12.shader_sampled_image_array_non_uniform_indexing =
                    supp_vk12.shader_sampled_image_array_non_uniform_indexing;
                vk12.runtime_descriptor_array = supp_vk12.runtime_descriptor_array;
            }
            if supp_vk12.scalar_block_layout == vk::TRUE {
                vk12.scalar_block_layout = vk::TRUE;
            }
            if supp_vk12.timeline_semaphore == vk::TRUE {
                vk12.timeline_semaphore = vk::TRUE;
            }
            if supp_vk12.shader_buffer_int64_atomics == vk::TRUE {
                vk12.shader_buffer_int64_atomics = vk::TRUE;
            }
            if supp_vk12.shader_shared_int64_atomics == vk::TRUE {
                vk12.shader_shared_int64_atomics = vk::TRUE;
            }
        }
    } else if has_individual_vk12_structs {
        if unsafe {
            chain_contains_stype(
                head_p_next,
                vk::StructureType::PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
            )
        } {
            if all_foreign_known
                && let Some(bda) = unsafe {
                    find_stype_mut::<vk::PhysicalDeviceBufferDeviceAddressFeatures<'_>>(
                        head_p_next,
                        vk::StructureType::PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
                    )
                }
                && caps.buffer_device_address
            {
                bda.buffer_device_address = vk::TRUE;
            }
        } else if caps.buffer_device_address {
            req_bda.p_next = head_p_next as *mut _;
            head_p_next = (&mut req_bda as *mut _) as *mut vk::BaseOutStructure<'_>;
        }

        if unsafe {
            chain_contains_stype(
                head_p_next,
                vk::StructureType::PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES,
            )
        } {
            if all_foreign_known
                && let Some(a64) = unsafe {
                    find_stype_mut::<vk::PhysicalDeviceShaderAtomicInt64Features<'_>>(
                        head_p_next,
                        vk::StructureType::PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES,
                    )
                }
            {
                if supp_vk12.shader_buffer_int64_atomics == vk::TRUE {
                    a64.shader_buffer_int64_atomics = vk::TRUE;
                }
                if supp_vk12.shader_shared_int64_atomics == vk::TRUE {
                    a64.shader_shared_int64_atomics = vk::TRUE;
                }
            }
        } else if supp_vk12.shader_buffer_int64_atomics == vk::TRUE {
            req_atomic64.p_next = head_p_next as *mut _;
            head_p_next = (&mut req_atomic64 as *mut _) as *mut vk::BaseOutStructure<'_>;
        }

        if unsafe {
            chain_contains_stype(
                head_p_next,
                vk::StructureType::PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
            )
        } {
            if all_foreign_known
                && let Some(di) = unsafe {
                    find_stype_mut::<vk::PhysicalDeviceDescriptorIndexingFeatures<'_>>(
                        head_p_next,
                        vk::StructureType::PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
                    )
                }
            {
                di.descriptor_binding_partially_bound =
                    supp_vk12.descriptor_binding_partially_bound;
                di.descriptor_binding_sampled_image_update_after_bind =
                    supp_vk12.descriptor_binding_sampled_image_update_after_bind;
                di.descriptor_binding_storage_image_update_after_bind =
                    supp_vk12.descriptor_binding_storage_image_update_after_bind;
                di.descriptor_binding_update_unused_while_pending =
                    supp_vk12.descriptor_binding_update_unused_while_pending;
                di.shader_sampled_image_array_non_uniform_indexing =
                    supp_vk12.shader_sampled_image_array_non_uniform_indexing;
                di.runtime_descriptor_array = supp_vk12.runtime_descriptor_array;
            }
        } else if caps.descriptor_indexing {
            req_desc_idx.p_next = head_p_next as *mut _;
            head_p_next = (&mut req_desc_idx as *mut _) as *mut vk::BaseOutStructure<'_>;
        }

        if unsafe {
            chain_contains_stype(
                head_p_next,
                vk::StructureType::PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES,
            )
        } {
            if all_foreign_known
                && let Some(sb) = unsafe {
                    find_stype_mut::<vk::PhysicalDeviceScalarBlockLayoutFeatures<'_>>(
                        head_p_next,
                        vk::StructureType::PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES,
                    )
                }
                && supp_vk12.scalar_block_layout == vk::TRUE
            {
                sb.scalar_block_layout = vk::TRUE;
            }
        } else if supp_vk12.scalar_block_layout == vk::TRUE {
            req_scalar_block.p_next = head_p_next as *mut _;
            head_p_next = (&mut req_scalar_block as *mut _) as *mut vk::BaseOutStructure<'_>;
        }

        if unsafe {
            chain_contains_stype(
                head_p_next,
                vk::StructureType::PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
            )
        } {
            if all_foreign_known
                && let Some(tl) = unsafe {
                    find_stype_mut::<vk::PhysicalDeviceTimelineSemaphoreFeatures<'_>>(
                        head_p_next,
                        vk::StructureType::PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
                    )
                }
                && supp_vk12.timeline_semaphore == vk::TRUE
            {
                tl.timeline_semaphore = vk::TRUE;
            }
        } else if supp_vk12.timeline_semaphore == vk::TRUE {
            req_timeline_sem.p_next = head_p_next as *mut _;
            head_p_next = (&mut req_timeline_sem as *mut _) as *mut vk::BaseOutStructure<'_>;
        }
    } else {
        req_vk12.p_next = head_p_next as *mut _;
        head_p_next = (&mut req_vk12 as *mut _) as *mut vk::BaseOutStructure<'_>;
    }

    // 2. Vulkan 1.3 features (synchronization2, dynamicRendering)
    let has_individual_vk13_structs = unsafe {
        chain_contains_stype(
            head_p_next,
            vk::StructureType::PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
        ) || chain_contains_stype(
            head_p_next,
            vk::StructureType::PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
        ) || chain_contains_stype(
            head_p_next,
            vk::StructureType::PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES,
        )
    };
    let mut req_vk13 = vk::PhysicalDeviceVulkan13Features::default()
        .synchronization2(caps.synchronization2)
        .dynamic_rendering(caps.dynamic_rendering)
        .shader_demote_to_helper_invocation(
            supp_vk13.shader_demote_to_helper_invocation == vk::TRUE,
        );
    let mut req_demote = vk::PhysicalDeviceShaderDemoteToHelperInvocationFeatures::default()
        .shader_demote_to_helper_invocation(
            supp_vk13.shader_demote_to_helper_invocation == vk::TRUE,
        );
    let mut req_sync2 = vk::PhysicalDeviceSynchronization2Features::default()
        .synchronization2(caps.synchronization2);
    let mut req_dyn_render = vk::PhysicalDeviceDynamicRenderingFeatures::default()
        .dynamic_rendering(caps.dynamic_rendering);

    if unsafe {
        chain_contains_stype(
            head_p_next,
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        )
    } {
        if all_foreign_known
            && let Some(vk13) = unsafe {
                find_stype_mut::<vk::PhysicalDeviceVulkan13Features<'_>>(
                    head_p_next,
                    vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                )
            }
        {
            if caps.synchronization2 {
                vk13.synchronization2 = vk::TRUE;
            }
            if caps.dynamic_rendering {
                vk13.dynamic_rendering = vk::TRUE;
            }
            vk13.shader_demote_to_helper_invocation |= supp_vk13.shader_demote_to_helper_invocation;
        }
    } else if has_individual_vk13_structs {
        if unsafe { chain_contains_stype(head_p_next, req_demote.s_type) } {
            if all_foreign_known
                && let Some(demote) = unsafe {
                    find_stype_mut::<vk::PhysicalDeviceShaderDemoteToHelperInvocationFeatures<'_>>(
                        head_p_next,
                        req_demote.s_type,
                    )
                }
            {
                demote.shader_demote_to_helper_invocation |=
                    supp_vk13.shader_demote_to_helper_invocation;
            }
        } else if supp_vk13.shader_demote_to_helper_invocation == vk::TRUE {
            req_demote.p_next = head_p_next as *mut _;
            head_p_next = (&mut req_demote as *mut _) as *mut vk::BaseOutStructure<'_>;
        }
        if unsafe {
            chain_contains_stype(
                head_p_next,
                vk::StructureType::PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
            )
        } {
            if all_foreign_known
                && let Some(s2) = unsafe {
                    find_stype_mut::<vk::PhysicalDeviceSynchronization2Features<'_>>(
                        head_p_next,
                        vk::StructureType::PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
                    )
                }
                && caps.synchronization2
            {
                s2.synchronization2 = vk::TRUE;
            }
        } else if caps.synchronization2 {
            req_sync2.p_next = head_p_next as *mut _;
            head_p_next = (&mut req_sync2 as *mut _) as *mut vk::BaseOutStructure<'_>;
        }

        if unsafe {
            chain_contains_stype(
                head_p_next,
                vk::StructureType::PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
            )
        } {
            if all_foreign_known
                && let Some(dr) = unsafe {
                    find_stype_mut::<vk::PhysicalDeviceDynamicRenderingFeatures<'_>>(
                        head_p_next,
                        vk::StructureType::PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
                    )
                }
                && caps.dynamic_rendering
            {
                dr.dynamic_rendering = vk::TRUE;
            }
        } else if caps.dynamic_rendering {
            req_dyn_render.p_next = head_p_next as *mut _;
            head_p_next = (&mut req_dyn_render as *mut _) as *mut vk::BaseOutStructure<'_>;
        }
    } else {
        req_vk13.p_next = head_p_next as *mut _;
        head_p_next = (&mut req_vk13 as *mut _) as *mut vk::BaseOutStructure<'_>;
    }

    // 3. Ray Tracing feature structs
    let mut req_accel = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
        .acceleration_structure(caps.acceleration_structure)
        .acceleration_structure_capture_replay(
            supp_accel.acceleration_structure_capture_replay == vk::TRUE,
        )
        .descriptor_binding_acceleration_structure_update_after_bind(
            supp_accel.descriptor_binding_acceleration_structure_update_after_bind == vk::TRUE,
        );
    let mut req_rq = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(caps.ray_query);
    let mut req_rt_pipe = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default()
        .ray_tracing_pipeline(caps.ray_tracing_pipeline);

    if caps.acceleration_structure {
        if unsafe { chain_contains_stype(head_p_next, req_accel.s_type) } {
            if all_foreign_known
                && let Some(existing_accel) = unsafe {
                    find_stype_mut::<vk::PhysicalDeviceAccelerationStructureFeaturesKHR<'_>>(
                        head_p_next,
                        req_accel.s_type,
                    )
                }
            {
                existing_accel.acceleration_structure = vk::TRUE;
                existing_accel.descriptor_binding_acceleration_structure_update_after_bind =
                    supp_accel.descriptor_binding_acceleration_structure_update_after_bind;
            }
        } else {
            req_accel.p_next = head_p_next as *mut _;
            head_p_next = (&mut req_accel as *mut _) as *mut vk::BaseOutStructure<'_>;
        }
    }
    if caps.ray_query {
        if unsafe { chain_contains_stype(head_p_next, req_rq.s_type) } {
            if all_foreign_known
                && let Some(existing_rq) = unsafe {
                    find_stype_mut::<vk::PhysicalDeviceRayQueryFeaturesKHR<'_>>(
                        head_p_next,
                        req_rq.s_type,
                    )
                }
            {
                existing_rq.ray_query = vk::TRUE;
            }
        } else {
            req_rq.p_next = head_p_next as *mut _;
            head_p_next = (&mut req_rq as *mut _) as *mut vk::BaseOutStructure<'_>;
        }
    }
    if caps.ray_tracing_pipeline {
        if unsafe { chain_contains_stype(head_p_next, req_rt_pipe.s_type) } {
            if all_foreign_known
                && let Some(existing_rtp) = unsafe {
                    find_stype_mut::<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR<'_>>(
                        head_p_next,
                        req_rt_pipe.s_type,
                    )
                }
            {
                existing_rtp.ray_tracing_pipeline = vk::TRUE;
            }
        } else {
            req_rt_pipe.p_next = head_p_next as *mut _;
            head_p_next = (&mut req_rt_pipe as *mut _) as *mut vk::BaseOutStructure<'_>;
        }
    }

    // 3b. Shader Execution Reordering (EXT / NV) and Opacity Micromap feature structs
    let enable_ser_ext = caps.ser_reordering_active && caps.supports_ser_ext;
    let enable_ser_nv = caps.ser_reordering_active && caps.supports_ser_nv;
    let mut req_ser_ext = VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT {
        ray_tracing_invocation_reorder: vk::Bool32::from(enable_ser_ext),
        ..VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT::default()
    };
    let mut req_ser_nv = vk::PhysicalDeviceRayTracingInvocationReorderFeaturesNV::default()
        .ray_tracing_invocation_reorder(enable_ser_nv);
    let mut req_omm = vk::PhysicalDeviceOpacityMicromapFeaturesEXT::default()
        .micromap(caps.supports_opacity_micromap);

    if enable_ser_ext && !unsafe { chain_contains_stype(head_p_next, req_ser_ext.s_type) } {
        req_ser_ext.p_next = head_p_next as *mut c_void;
        head_p_next = (&mut req_ser_ext as *mut _) as *mut vk::BaseOutStructure<'_>;
    }
    if enable_ser_nv && !unsafe { chain_contains_stype(head_p_next, req_ser_nv.s_type) } {
        req_ser_nv.p_next = head_p_next as *mut _;
        head_p_next = (&mut req_ser_nv as *mut _) as *mut vk::BaseOutStructure<'_>;
    }
    if caps.supports_opacity_micromap
        && !unsafe { chain_contains_stype(head_p_next, req_omm.s_type) }
    {
        req_omm.p_next = head_p_next as *mut _;
        head_p_next = (&mut req_omm as *mut _) as *mut vk::BaseOutStructure<'_>;
    }

    // 4. Present ID feature (required by VK_KHR_present_id and VK_NV_low_latency2)
    let mut req_present_id = vk::PhysicalDevicePresentIdFeaturesKHR::default()
        .present_id(supp_present_id.present_id == vk::TRUE);
    if has_present_id_ext && supp_present_id.present_id == vk::TRUE {
        if unsafe {
            chain_contains_stype(
                head_p_next,
                vk::StructureType::PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR,
            )
        } {
            if all_foreign_known
                && let Some(existing_pid) = unsafe {
                    find_stype_mut::<vk::PhysicalDevicePresentIdFeaturesKHR<'_>>(
                        head_p_next,
                        vk::StructureType::PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR,
                    )
                }
            {
                existing_pid.present_id = vk::TRUE;
            }
        } else {
            req_present_id.p_next = head_p_next as *mut _;
            head_p_next = (&mut req_present_id as *mut _) as *mut vk::BaseOutStructure<'_>;
        }
    }

    // Reconcile reported DeviceCapabilities against actually-enabled feature bits in the final
    // pNext chain when unknown foreign nodes prevented in-place merging of existing structs.
    if !all_foreign_known {
        unsafe {
            if let Some(vk12) = find_stype_ref::<vk::PhysicalDeviceVulkan12Features<'_>>(
                head_p_next,
                vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            ) {
                caps.buffer_device_address &= vk12.buffer_device_address == vk::TRUE;
                caps.descriptor_indexing &= vk12.descriptor_indexing == vk::TRUE
                    && vk12.descriptor_binding_partially_bound == vk::TRUE
                    && vk12.descriptor_binding_sampled_image_update_after_bind == vk::TRUE;
                caps.shader_int64_atomics &= vk12.shader_buffer_int64_atomics == vk::TRUE;
            } else {
                caps.buffer_device_address &=
                    find_stype_ref::<vk::PhysicalDeviceBufferDeviceAddressFeatures<'_>>(
                        head_p_next,
                        vk::StructureType::PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
                    )
                    .is_some_and(|f| f.buffer_device_address == vk::TRUE);
                caps.descriptor_indexing &=
                    find_stype_ref::<vk::PhysicalDeviceDescriptorIndexingFeatures<'_>>(
                        head_p_next,
                        vk::StructureType::PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
                    )
                    .is_some_and(|f| {
                        f.descriptor_binding_partially_bound == vk::TRUE
                            && f.descriptor_binding_sampled_image_update_after_bind == vk::TRUE
                    });
                caps.shader_int64_atomics &=
                    find_stype_ref::<vk::PhysicalDeviceShaderAtomicInt64Features<'_>>(
                        head_p_next,
                        vk::StructureType::PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES,
                    )
                    .is_some_and(|f| f.shader_buffer_int64_atomics == vk::TRUE);
            }

            if let Some(vk13) = find_stype_ref::<vk::PhysicalDeviceVulkan13Features<'_>>(
                head_p_next,
                vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            ) {
                caps.synchronization2 &= vk13.synchronization2 == vk::TRUE;
                caps.dynamic_rendering &= vk13.dynamic_rendering == vk::TRUE;
            } else {
                caps.synchronization2 &=
                    find_stype_ref::<vk::PhysicalDeviceSynchronization2Features<'_>>(
                        head_p_next,
                        vk::StructureType::PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
                    )
                    .is_some_and(|f| f.synchronization2 == vk::TRUE);
                caps.dynamic_rendering &=
                    find_stype_ref::<vk::PhysicalDeviceDynamicRenderingFeatures<'_>>(
                        head_p_next,
                        vk::StructureType::PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
                    )
                    .is_some_and(|f| f.dynamic_rendering == vk::TRUE);
            }

            caps.acceleration_structure &=
                find_stype_ref::<vk::PhysicalDeviceAccelerationStructureFeaturesKHR<'_>>(
                    head_p_next,
                    vk::StructureType::PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
                )
                .is_some_and(|f| f.acceleration_structure == vk::TRUE);
            caps.ray_query &= find_stype_ref::<vk::PhysicalDeviceRayQueryFeaturesKHR<'_>>(
                head_p_next,
                vk::StructureType::PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
            )
            .is_some_and(|f| f.ray_query == vk::TRUE);
            caps.ray_tracing_pipeline &=
                find_stype_ref::<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR<'_>>(
                    head_p_next,
                    vk::StructureType::PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
                )
                .is_some_and(|f| f.ray_tracing_pipeline == vk::TRUE);
            caps.supports_ser_nv &=
                find_stype_ref::<vk::PhysicalDeviceRayTracingInvocationReorderFeaturesNV<'_>>(
                    head_p_next,
                    vk::StructureType::PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV,
                )
                .is_some_and(|f| f.ray_tracing_invocation_reorder == vk::TRUE);
            caps.supports_ser_ext &= find_stype_ref::<
                VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT,
            >(
                head_p_next,
                vk::StructureType::from_raw(
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT,
                ),
            )
            .is_some_and(|f| f.ray_tracing_invocation_reorder == vk::TRUE);
            caps.ser_reordering_active &= caps.supports_ser_nv || caps.supports_ser_ext;
            caps.supports_opacity_micromap &=
                find_stype_ref::<vk::PhysicalDeviceOpacityMicromapFeaturesEXT<'_>>(
                    head_p_next,
                    vk::StructureType::PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT,
                )
                .is_some_and(|f| f.micromap == vk::TRUE);
            if caps.low_latency2_nv && !seen.contains(c"VK_KHR_present_id2") {
                caps.low_latency2_nv &=
                    find_stype_ref::<vk::PhysicalDevicePresentIdFeaturesKHR<'_>>(
                        head_p_next,
                        vk::StructureType::PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR,
                    )
                    .is_some_and(|f| f.present_id == vk::TRUE);
            }
        }
    }

    let ext_ptrs: Vec<*const c_char> = merged_exts.iter().map(|s| s.as_ptr()).collect();
    let mut patched_info = base_info;
    patched_info.p_next = head_p_next as *const _;
    let has_feat2_in_chain =
        unsafe { chain_contains_stype(head_p_next, vk::StructureType::PHYSICAL_DEVICE_FEATURES_2) };
    if !has_feat2_in_chain
        && (!base_info.p_enabled_features.is_null()
            || supp_core.shader_int64 == vk::TRUE
            || supp_core.independent_blend == vk::TRUE)
    {
        patched_info.p_enabled_features = &local_core_features;
    }
    patched_info.enabled_extension_count = ext_ptrs.len() as u32;
    patched_info.pp_enabled_extension_names = ext_ptrs.as_ptr();

    let alloc_ref = unsafe { allocator_ptr.as_ref() };
    match unsafe { instance.create_device(physical_device, &patched_info, alloc_ref) } {
        Ok(device) => {
            let dev_raw = device.handle().as_raw();
            unsafe {
                *out_device_ptr = device.handle();
            }
            let tier_policy =
                evaluate_platform_tier_policy(&caps, caps.is_moltenvk, cfg!(target_os = "macos"));
            let mut reg = registry().lock();
            reg.device_capabilities.insert(dev_raw, caps);
            reg.last_merged_caps = Some(caps);
            reg.last_physical_device = Some(physical_device_raw);
            reg.probe_info.gpu_name = gpu_name;
            reg.probe_info.api_version = api_version;
            reg.probe_info.driver_name = driver_name;
            reg.probe_info.driver_version = driver_version;
            reg.probe_info.has_memory_budget = caps.has_memory_budget;
            reg.probe_info.is_moltenvk = caps.is_moltenvk;
            reg.probe_info.tier_policy_reason = tier_policy.reason.clone();
            reg.last_tier_policy = Some(tier_policy);
            reg.probe_info.device_extensions = offered_exts
                .iter()
                .map(|c| c.to_string_lossy().to_string())
                .collect();
            let _ = ptr::null::<u8>();
            vk::Result::SUCCESS.as_raw()
        }
        Err(err) => err.as_raw(),
    }
}

/// Probes the real surface format list with `vkGetPhysicalDeviceSurfaceFormatsKHR` and returns
/// `true` only when an HDR format is paired with an HDR/linear colour space. Never reports HDR
/// optimistically on an SDR-only X11/Wayland surface.
pub fn is_hdr_swapchain_supported(surface_raw: u64) -> bool {
    if surface_raw == 0 {
        return false;
    }
    let entry = load_vulkan_entry();
    let (instance_raw, physical_device_raw) = {
        let reg = registry().lock();
        (reg.instances.last().copied(), reg.last_physical_device)
    };
    let (Some(instance_raw), Some(physical_device_raw)) = (instance_raw, physical_device_raw)
    else {
        return false;
    };
    let instance =
        unsafe { Instance::load(entry.static_fn(), vk::Instance::from_raw(instance_raw)) };
    let surface_ext = ash::khr::surface::Instance::new(&entry, &instance);
    let surface = vk::SurfaceKHR::from_raw(surface_raw);
    let physical_device = vk::PhysicalDevice::from_raw(physical_device_raw);
    let formats = match unsafe {
        surface_ext.get_physical_device_surface_formats(physical_device, surface)
    } {
        Ok(f) => f,
        Err(_) => return false,
    };
    let supported = formats.iter().any(|f| {
        let hdr_format = matches!(
            f.format,
            vk::Format::R16G16B16A16_SFLOAT | vk::Format::A2B10G10R10_UNORM_PACK32
        );
        let hdr_space = matches!(
            f.color_space,
            vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT | vk::ColorSpaceKHR::HDR10_ST2084_EXT
        );
        hdr_format && hdr_space
    });
    registry().lock().probe_info.hdr_surface_supported = supported;
    supported
}

/// Returns a snapshot of the real GPU/driver facts gathered during instance/device creation.
pub fn probe_info() -> DeviceProbeInfo {
    registry().lock().probe_info.clone()
}

/// LIVE NGX query — only reachable when `VS_CORE_NGX_QUERY_EXTENSIONS=1`. Never called by default.
/// Returns an empty list when the vendor runtime is absent or the query fails.
fn ngx_required_instance_extensions() -> Vec<String> {
    use crate::sdk::ngx::{DEFAULT_NGX_APP_ID, NgxVulkanLibrary, NvSdkNgxFeature};
    let dirs: Vec<&Path> = vec![Path::new("natives"), Path::new("run/natives")];
    let Ok(lib) = NgxVulkanLibrary::load(&dirs) else {
        return Vec::new();
    };
    lib.query_instance_extensions(
        NvSdkNgxFeature::SuperSampling,
        DEFAULT_NGX_APP_ID,
        ".",
        &dirs,
    )
    .unwrap_or_default()
}

/// LIVE NGX device requirement query via `NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements`.
/// Only reachable when `VS_CORE_NGX_QUERY_EXTENSIONS=1`; never called by default because doing so
/// before NGX Init_Ext2 crashes the game during vkCreateDevice.
fn ngx_required_device_extensions(
    instance: vk::Instance,
    physical_device: vk::PhysicalDevice,
) -> Vec<String> {
    use crate::sdk::ngx::{DEFAULT_NGX_APP_ID, NgxVulkanLibrary, NvSdkNgxFeature};
    let dirs: Vec<&Path> = vec![Path::new("natives"), Path::new("run/natives")];
    let Ok(lib) = NgxVulkanLibrary::load(&dirs) else {
        return Vec::new();
    };
    lib.query_device_extensions(
        instance,
        physical_device,
        NvSdkNgxFeature::SuperSampling,
        DEFAULT_NGX_APP_ID,
        ".",
        &dirs,
    )
    .unwrap_or_default()
}

unsafe fn chain_contains_stype(
    mut cur: *const vk::BaseOutStructure<'_>,
    target: vk::StructureType,
) -> bool {
    while !cur.is_null() {
        let node = unsafe { &*cur };
        if node.s_type == target {
            return true;
        }
        cur = node.p_next;
    }
    false
}

unsafe fn find_stype_ref<'a, T>(
    mut cur: *const vk::BaseOutStructure<'_>,
    target: vk::StructureType,
) -> Option<&'a T> {
    while !cur.is_null() {
        let node = unsafe { &*cur };
        if node.s_type == target {
            return Some(unsafe { &*(cur as *const T) });
        }
        cur = node.p_next;
    }
    None
}

unsafe fn find_stype_mut<'a, T>(
    mut cur: *mut vk::BaseOutStructure<'_>,
    target: vk::StructureType,
) -> Option<&'a mut T> {
    while !cur.is_null() {
        let node = unsafe { &mut *cur };
        if node.s_type == target {
            return Some(unsafe { &mut *(cur as *mut T) });
        }
        cur = node.p_next;
    }
    None
}

// ---------------------------------------------------------------------------
// VRAM Budget Telemetry (VK_EXT_memory_budget) & OOM Pressure Policy
// ---------------------------------------------------------------------------

/// Coarse VRAM pressure level derived from `usage_bytes / budget_bytes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum VramPressureLevel {
    #[default]
    Normal,
    Elevated,
    Critical,
}

impl VramPressureLevel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Elevated => "elevated",
            Self::Critical => "critical",
        }
    }
}

/// Snapshot of device-local VRAM usage and driver budget.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct VramBudgetSnapshot {
    pub usage_bytes: u64,
    pub budget_bytes: u64,
    pub usage_ratio: f32,
    pub pressure: VramPressureLevel,
    pub from_memory_budget_ext: bool,
}

impl VramBudgetSnapshot {
    pub const ELEVATED_THRESHOLD: f32 = 0.75;
    pub const CRITICAL_THRESHOLD: f32 = 0.90;

    /// Constructs a snapshot from `(usage_bytes, budget_bytes)` and classifies pressure level.
    pub fn from_usage_and_budget(
        usage_bytes: u64,
        budget_bytes: u64,
        from_memory_budget_ext: bool,
    ) -> Self {
        let usage_ratio = if budget_bytes > 0 {
            ((usage_bytes as f64) / (budget_bytes as f64)).clamp(0.0, 2.0) as f32
        } else {
            0.0
        };
        let pressure = if usage_ratio >= Self::CRITICAL_THRESHOLD {
            VramPressureLevel::Critical
        } else if usage_ratio >= Self::ELEVATED_THRESHOLD {
            VramPressureLevel::Elevated
        } else {
            VramPressureLevel::Normal
        };
        Self {
            usage_bytes,
            budget_bytes,
            usage_ratio,
            pressure,
            from_memory_budget_ext,
        }
    }
}

/// Queries live device-local VRAM usage and budget via `VK_EXT_memory_budget` chained onto
/// `vk::PhysicalDeviceMemoryProperties2`, falling back safely to total `DEVICE_LOCAL` heap size
/// when `VK_EXT_memory_budget` is absent.
pub fn query_vram_budget(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    has_memory_budget: bool,
) -> VramBudgetSnapshot {
    if has_memory_budget {
        let mut budget_props = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut mem_props2 =
            vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget_props);
        unsafe {
            instance.get_physical_device_memory_properties2(physical_device, &mut mem_props2);
        }
        let heap_count =
            (mem_props2.memory_properties.memory_heap_count as usize).min(vk::MAX_MEMORY_HEAPS);
        let memory_heaps = mem_props2.memory_properties.memory_heaps;
        let heap_usage = budget_props.heap_usage;
        let heap_budget = budget_props.heap_budget;
        let mut usage_bytes = 0u64;
        let mut budget_bytes = 0u64;
        for i in 0..heap_count {
            let heap = memory_heaps[i];
            if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                usage_bytes = usage_bytes.saturating_add(heap_usage[i]);
                budget_bytes = budget_bytes.saturating_add(heap_budget[i]);
            }
        }
        if budget_bytes > 0 {
            return VramBudgetSnapshot::from_usage_and_budget(usage_bytes, budget_bytes, true);
        }
    }

    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let heap_count = (mem_props.memory_heap_count as usize).min(vk::MAX_MEMORY_HEAPS);
    let total_device_local: u64 = mem_props.memory_heaps[..heap_count]
        .iter()
        .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|h| h.size)
        .fold(0u64, u64::saturating_add);
    VramBudgetSnapshot::from_usage_and_budget(0, total_device_local, false)
}

/// Recommended OOM-protection actions derived from [`VramBudgetSnapshot`] / [`VramPressureLevel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VramPressurePolicy {
    pub pressure: VramPressureLevel,
    pub throttle_blas_compaction: bool,
    pub lod_cutoff_scale: f32,
    pub trim_reservoirs: bool,
}

impl Default for VramPressurePolicy {
    fn default() -> Self {
        Self::for_pressure(VramPressureLevel::Normal)
    }
}

impl VramPressurePolicy {
    pub const fn for_pressure(pressure: VramPressureLevel) -> Self {
        match pressure {
            VramPressureLevel::Normal => Self {
                pressure,
                throttle_blas_compaction: false,
                lod_cutoff_scale: 1.0,
                trim_reservoirs: false,
            },
            VramPressureLevel::Elevated => Self {
                pressure,
                throttle_blas_compaction: true,
                lod_cutoff_scale: 0.75,
                trim_reservoirs: false,
            },
            VramPressureLevel::Critical => Self {
                pressure,
                throttle_blas_compaction: true,
                lod_cutoff_scale: 0.50,
                trim_reservoirs: true,
            },
        }
    }

    pub const fn for_snapshot(snapshot: VramBudgetSnapshot) -> Self {
        Self::for_pressure(snapshot.pressure)
    }
}

// ---------------------------------------------------------------------------
// macOS / MoltenVK & Hardware Ray Tracing Tier Policy
// ---------------------------------------------------------------------------

/// Platform render tier ceiling and human-readable diagnostic reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlatformTierPolicy {
    pub max_render_tier: vsp_compiler::RenderTier,
    pub is_moltenvk: bool,
    pub hardware_rt_blocked: bool,
    pub reason: String,
}

impl PlatformTierPolicy {
    /// Clamps `requested` so it never exceeds `self.max_render_tier`.
    pub const fn clamp_render_tier(
        &self,
        requested: vsp_compiler::RenderTier,
    ) -> vsp_compiler::RenderTier {
        if (requested as u8) <= (self.max_render_tier as u8) {
            requested
        } else {
            self.max_render_tier
        }
    }
}

/// Evaluates the maximum safe [`vsp_compiler::RenderTier`] for `caps`, explicitly capping
/// macOS / MoltenVK (or any driver lacking `VK_KHR_ray_tracing_pipeline` /
/// `VK_KHR_acceleration_structure`) to Tier 0/1 raster rendering with a clear reason string.
pub fn evaluate_platform_tier_policy(
    caps: &DeviceCapabilities,
    is_moltenvk: bool,
    is_macos: bool,
) -> PlatformTierPolicy {
    use vsp_compiler::RenderTier;

    let raster_cap = if caps.descriptor_indexing && caps.synchronization2 && caps.dynamic_rendering
    {
        RenderTier::Tier1ComputeGi
    } else {
        RenderTier::Tier0Raster
    };

    let has_full_rt =
        caps.acceleration_structure && caps.ray_tracing_pipeline && caps.buffer_device_address;
    let has_ray_query = caps.acceleration_structure && caps.ray_query && caps.buffer_device_address;

    if (is_moltenvk || caps.portability_subset) && !has_full_rt {
        return PlatformTierPolicy {
            max_render_tier: raster_cap,
            is_moltenvk: true,
            hardware_rt_blocked: true,
            reason: format!(
                "MoltenVK / portability subset lacks VK_KHR_ray_tracing_pipeline or VK_KHR_acceleration_structure; capped to {:?} raster mode",
                raster_cap
            ),
        };
    }

    if is_macos && !has_full_rt {
        return PlatformTierPolicy {
            max_render_tier: raster_cap,
            is_moltenvk,
            hardware_rt_blocked: true,
            reason: format!(
                "macOS Vulkan driver lacks hardware ray tracing pipeline/acceleration structure support; capped to {:?} raster mode",
                raster_cap
            ),
        };
    }

    if !caps.acceleration_structure || (!caps.ray_tracing_pipeline && !caps.ray_query) {
        return PlatformTierPolicy {
            max_render_tier: raster_cap,
            is_moltenvk,
            hardware_rt_blocked: true,
            reason: format!(
                "Hardware ray tracing extensions unavailable (AS={}, RQ={}, RTPipeline={}); capped to {:?} raster mode",
                caps.acceleration_structure, caps.ray_query, caps.ray_tracing_pipeline, raster_cap
            ),
        };
    }

    if has_full_rt {
        PlatformTierPolicy {
            max_render_tier: RenderTier::Tier3PathTracing,
            is_moltenvk,
            hardware_rt_blocked: false,
            reason: "Full hardware ray tracing pipeline and acceleration structures supported"
                .to_string(),
        }
    } else if has_ray_query {
        PlatformTierPolicy {
            max_render_tier: RenderTier::Tier2RayQuery,
            is_moltenvk,
            hardware_rt_blocked: false,
            reason: "Hardware ray query supported without ray tracing pipeline; capped to Tier 2"
                .to_string(),
        }
    } else {
        PlatformTierPolicy {
            max_render_tier: raster_cap,
            is_moltenvk,
            hardware_rt_blocked: true,
            reason: format!(
                "Buffer device address required for hardware RT is unavailable; capped to {:?} raster mode",
                raster_cap
            ),
        }
    }
}

/// Returns the most recently evaluated [`PlatformTierPolicy`] from device creation, if any.
pub fn current_platform_tier_policy() -> Option<PlatformTierPolicy> {
    registry().lock().last_tier_policy.clone()
}
