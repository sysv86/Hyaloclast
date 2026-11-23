pub mod bindless;
pub mod buffer;
pub mod device_merger;
pub mod profiler;
pub mod retire;

pub use buffer::{GpuMemoryAllocator, PooledImageAllocation};
pub use device_merger::{
    PlatformTierPolicy, VramBudgetSnapshot, VramPressureLevel, VramPressurePolicy,
    evaluate_platform_tier_policy, query_vram_budget,
};
pub use profiler::{GpuPassTiming, GpuTimestampProfiler};

use ash::{
    Device, Entry, Instance,
    khr::{acceleration_structure, dynamic_rendering, ray_tracing_pipeline, synchronization2},
    vk::{self, Handle},
};
use parking_lot::{ReentrantMutex, ReentrantMutexGuard};
use std::cell::RefCell;
use std::sync::{
    OnceLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

/// Reentrant mutex shared between Java (`VulkanQueue`, `VulkanGpuSurface`, `VulkanCommandEncoder`)
/// and Rust (`PresentThread`, async BLAS builders) to serialize all `VkQueue` operations.
static QUEUE_MUTEX: OnceLock<ReentrantMutex<()>> = OnceLock::new();
static QUEUE_LOCK_COUNT: AtomicU64 = AtomicU64::new(0);
static SER_REORDERING_ENABLED: AtomicBool = AtomicBool::new(true);

thread_local! {
    static QUEUE_GUARDS: RefCell<Vec<ReentrantMutexGuard<'static, ()>>> = const { RefCell::new(Vec::new()) };
}

fn queue_mutex() -> &'static ReentrantMutex<()> {
    QUEUE_MUTEX.get_or_init(|| ReentrantMutex::new(()))
}

pub fn set_ser_reordering_enabled(enabled: bool) {
    SER_REORDERING_ENABLED.store(enabled, Ordering::Release);
}

pub fn ser_reordering_enabled() -> bool {
    SER_REORDERING_ENABLED.load(Ordering::Acquire)
        && ser_reordering_enabled_by_environment()
        && !crate::crash::global_device_lost_guard().is_rt_vsp_disabled()
}

pub fn ser_reordering_enabled_by_environment() -> bool {
    !matches!(
        std::env::var("VS_CORE_ENABLE_SER")
            .map(|value| value.trim().to_ascii_lowercase())
            .as_deref(),
        Ok("0" | "false" | "off")
    )
}

/// Acquires the global reentrant Vulkan queue lock (callable from FFM and internal Rust threads).
pub fn lock_queue() {
    QUEUE_GUARDS.with(|guards| guards.borrow_mut().push(queue_mutex().lock()));
    QUEUE_LOCK_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Releases the global reentrant Vulkan queue lock.
///
/// # Safety
/// The calling thread must currently hold a lock acquired via [`lock_queue`].
pub fn unlock_queue() {
    let guard = QUEUE_GUARDS.with(|guards| guards.borrow_mut().pop());
    if let Some(guard) = guard {
        drop(guard);
    } else {
        eprintln!("[vulkanshaders-core] unlock_queue called without a matching lock_queue");
    }
}

/// Executes a closure while holding the global reentrant Vulkan queue lock.
pub fn with_queue_lock<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = queue_mutex().lock();
    QUEUE_LOCK_COUNT.fetch_add(1, Ordering::Relaxed);
    f()
}

pub fn total_queue_locks() -> u64 {
    QUEUE_LOCK_COUNT.load(Ordering::Relaxed)
}

#[cfg(test)]
mod queue_tests {
    use super::*;

    #[test]
    fn ffi_queue_lock_can_be_reentered_by_native_present_hook() {
        lock_queue();
        let nested = queue_mutex().try_lock();
        let acquired = nested.is_some();
        drop(nested);
        unlock_queue();
        assert!(acquired);
    }

    #[test]
    fn unmatched_ffi_unlock_does_not_release_another_lock() {
        with_queue_lock(|| {
            unlock_queue();
            assert!(queue_mutex().try_lock().is_some());
        });
    }
}

/// Coarse GPU hardware architecture classification derived from PCI vendor/device IDs
/// and the Vulkan physical device name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GpuArchitecture {
    #[default]
    Unknown,
    NvidiaTuring,
    NvidiaAmpere,
    NvidiaAda,
    NvidiaBlackwell,
    AmdRdna2,
    AmdRdna3,
    AmdRdna4,
    IntelAlchemist,
    IntelBattlemage,
}

impl GpuArchitecture {
    pub const fn is_ada_or_newer(self) -> bool {
        matches!(self, Self::NvidiaAda | Self::NvidiaBlackwell)
    }

    pub const fn is_nvidia(self) -> bool {
        matches!(
            self,
            Self::NvidiaTuring | Self::NvidiaAmpere | Self::NvidiaAda | Self::NvidiaBlackwell
        )
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::NvidiaTuring => "nvidia_turing",
            Self::NvidiaAmpere => "nvidia_ampere",
            Self::NvidiaAda => "nvidia_ada",
            Self::NvidiaBlackwell => "nvidia_blackwell",
            Self::AmdRdna2 => "amd_rdna2",
            Self::AmdRdna3 => "amd_rdna3",
            Self::AmdRdna4 => "amd_rdna4",
            Self::IntelAlchemist => "intel_alchemist",
            Self::IntelBattlemage => "intel_battlemage",
        }
    }

    /// Classifies GPU architecture from PCI `vendor_id`, PCI `device_id`, and `gpu_name` fallback.
    pub fn classify(vendor_id: u32, device_id: u32, gpu_name: &str) -> Self {
        let lower = gpu_name.to_ascii_lowercase();

        match vendor_id {
            0x10DE => {
                // PCI device ID ranges for NVIDIA RTX generations
                if (0x2B00..=0x2FFF).contains(&device_id) {
                    return Self::NvidiaBlackwell;
                }
                if (0x2680..=0x28FF).contains(&device_id) {
                    return Self::NvidiaAda;
                }
                if (0x2200..=0x25FF).contains(&device_id) {
                    return Self::NvidiaAmpere;
                }
                if (0x1E00..=0x21FF).contains(&device_id) {
                    return Self::NvidiaTuring;
                }
                Self::classify_nvidia_by_name(&lower)
            }
            0x1002 => {
                if (0x7550..=0x759F).contains(&device_id) {
                    return Self::AmdRdna4;
                }
                if (0x7440..=0x74FF).contains(&device_id) {
                    return Self::AmdRdna3;
                }
                if (0x73A0..=0x73FF).contains(&device_id) {
                    return Self::AmdRdna2;
                }
                Self::classify_amd_by_name(&lower)
            }
            0x8086 => {
                if (0xE200..=0xE2FF).contains(&device_id) {
                    return Self::IntelBattlemage;
                }
                if (0x5690..=0x56CF).contains(&device_id) {
                    return Self::IntelAlchemist;
                }
                Self::classify_intel_by_name(&lower)
            }
            _ => {
                let nv = Self::classify_nvidia_by_name(&lower);
                if nv != Self::Unknown {
                    return nv;
                }
                let amd = Self::classify_amd_by_name(&lower);
                if amd != Self::Unknown {
                    return amd;
                }
                Self::classify_intel_by_name(&lower)
            }
        }
    }

    fn classify_nvidia_by_name(lower: &str) -> Self {
        if lower.contains("rtx 50") || lower.contains("blackwell") || lower.contains("gb20") {
            Self::NvidiaBlackwell
        } else if lower.contains("rtx 40")
            || lower.contains("ada")
            || lower.contains("ad10")
            || lower.contains("l40")
        {
            Self::NvidiaAda
        } else if lower.contains("rtx 30") || lower.contains("rtx a") || lower.contains("ampere") {
            Self::NvidiaAmpere
        } else if lower.contains("rtx 20")
            || lower.contains("gtx 16")
            || lower.contains("titan rtx")
            || lower.contains("quadro rtx")
            || lower.contains("turing")
        {
            Self::NvidiaTuring
        } else {
            Self::Unknown
        }
    }

    fn classify_amd_by_name(lower: &str) -> Self {
        if lower.contains("rx 9") || lower.contains("rdna 4") || lower.contains("rdna4") {
            Self::AmdRdna4
        } else if lower.contains("rx 7") || lower.contains("rdna 3") || lower.contains("rdna3") {
            Self::AmdRdna3
        } else if lower.contains("rx 6") || lower.contains("rdna 2") || lower.contains("rdna2") {
            Self::AmdRdna2
        } else {
            Self::Unknown
        }
    }

    fn classify_intel_by_name(lower: &str) -> Self {
        if lower.contains("arc b") || lower.contains("battlemage") {
            Self::IntelBattlemage
        } else if lower.contains("arc a") || lower.contains("alchemist") {
            Self::IntelAlchemist
        } else {
            Self::Unknown
        }
    }
}

/// Hardware performance tier used to scale RT budgets, shadow resolutions, and shader quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum GpuTierClass {
    #[default]
    Low,
    Mid,
    High,
    Enthusiast,
}

impl GpuTierClass {
    pub const fn is_high_end(self) -> bool {
        matches!(self, Self::High | Self::Enthusiast)
    }

    pub const fn is_enthusiast(self) -> bool {
        matches!(self, Self::Enthusiast)
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Mid => "mid",
            Self::High => "high",
            Self::Enthusiast => "enthusiast",
        }
    }

    /// Classifies GPU performance tier using architecture, device type, VRAM size, and GPU name.
    pub fn classify(
        arch: GpuArchitecture,
        device_type: vk::PhysicalDeviceType,
        vram_bytes: u64,
        gpu_name: &str,
    ) -> Self {
        if matches!(
            device_type,
            vk::PhysicalDeviceType::INTEGRATED_GPU
                | vk::PhysicalDeviceType::CPU
                | vk::PhysicalDeviceType::VIRTUAL_GPU
        ) {
            return Self::Low;
        }

        let lower = gpu_name.to_ascii_lowercase();
        let vram_mib = vram_bytes / (1024 * 1024);
        let is_discrete = device_type == vk::PhysicalDeviceType::DISCRETE_GPU;

        // Explicit Enthusiast flagship models
        if lower.contains("rtx 5090")
            || lower.contains("rtx 5080")
            || lower.contains("rtx 4090")
            || lower.contains("rtx 4080")
            || lower.contains("rx 7900")
            || lower.contains("rx 9070 xt")
            || lower.contains("rtx 6000 ada")
            || lower.contains("rtx pro 6000")
        {
            return Self::Enthusiast;
        }

        // Discrete >= 16 GB (allowing minor driver heap reservation) on Ada / Blackwell / RDNA3+
        if is_discrete
            && vram_mib >= 15_360
            && matches!(
                arch,
                GpuArchitecture::NvidiaAda
                    | GpuArchitecture::NvidiaBlackwell
                    | GpuArchitecture::AmdRdna3
                    | GpuArchitecture::AmdRdna4
            )
        {
            return Self::Enthusiast;
        }

        // Explicit High-tier models
        if lower.contains("rtx 5070")
            || lower.contains("rtx 4070")
            || lower.contains("rtx 3090")
            || lower.contains("rtx 3080")
            || lower.contains("rx 7800")
            || lower.contains("rx 9070")
            || lower.contains("rx 6950")
            || lower.contains("rx 6900")
            || lower.contains("rx 6800")
        {
            return Self::High;
        }

        // Discrete >= 12 GB VRAM
        if is_discrete && vram_mib >= 11_264 {
            return Self::High;
        }

        // Discrete >= 8 GB VRAM
        if is_discrete && vram_mib >= 7_168 {
            return Self::Mid;
        }

        Self::Low
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DeviceCapabilities {
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: vk::PhysicalDeviceType,
    pub vram_bytes: u64,
    pub gpu_arch: GpuArchitecture,
    pub gpu_tier_class: GpuTierClass,
    pub buffer_device_address: bool,
    pub descriptor_indexing: bool,
    pub acceleration_structure: bool,
    pub ray_query: bool,
    pub ray_tracing_pipeline: bool,
    pub synchronization2: bool,
    pub dynamic_rendering: bool,
    pub low_latency2_nv: bool,
    pub hdr_swapchain_colorspace: bool,
    pub shader_int64_atomics: bool,
    pub portability_subset: bool,
    pub supports_ser_nv: bool,
    pub supports_ser_ext: bool,
    pub ser_reordering_active: bool,
    pub supports_opacity_micromap: bool,
    pub has_memory_budget: bool,
    pub is_moltenvk: bool,
    pub max_ray_recursion_depth: u32,
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            vendor_id: 0,
            device_id: 0,
            device_type: vk::PhysicalDeviceType::OTHER,
            vram_bytes: 0,
            gpu_arch: GpuArchitecture::Unknown,
            gpu_tier_class: GpuTierClass::Low,
            buffer_device_address: false,
            descriptor_indexing: false,
            acceleration_structure: false,
            ray_query: false,
            ray_tracing_pipeline: false,
            synchronization2: false,
            dynamic_rendering: false,
            low_latency2_nv: false,
            hdr_swapchain_colorspace: false,
            shader_int64_atomics: false,
            portability_subset: false,
            supports_ser_nv: false,
            supports_ser_ext: false,
            ser_reordering_active: false,
            supports_opacity_micromap: false,
            has_memory_budget: false,
            is_moltenvk: false,
            max_ray_recursion_depth: 1,
        }
    }
}

impl DeviceCapabilities {
    pub const fn is_high_end(&self) -> bool {
        self.gpu_tier_class.is_high_end()
    }

    pub const fn is_ada_or_newer(&self) -> bool {
        self.gpu_arch.is_ada_or_newer()
    }

    pub const fn max_shadow_resolution(&self) -> u32 {
        if self.gpu_tier_class.is_high_end() {
            8192
        } else {
            4096
        }
    }

    pub const fn device_type_str(&self) -> &'static str {
        match self.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => "discrete",
            vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated",
            vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual",
            vk::PhysicalDeviceType::CPU => "cpu",
            _ => "other",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RayTracingDeviceProperties {
    pub shader_group_handle_size: u32,
    pub shader_group_handle_alignment: u32,
    pub shader_group_base_alignment: u32,
    pub max_ray_recursion_depth: u32,
}

impl Default for RayTracingDeviceProperties {
    fn default() -> Self {
        Self {
            shader_group_handle_size: 32,
            shader_group_handle_alignment: 32,
            shader_group_base_alignment: 64,
            max_ray_recursion_depth: 1,
        }
    }
}

/// Wraps the borrowed `RenderPearl` Vulkan handles (`VkInstance`, `VkPhysicalDevice`, `VkDevice`,
/// `VkQueue`) in `ash` dispatch tables and caches hardware RT/AS alignment and queue family indices.
pub struct VulkanContext {
    pub entry: Entry,
    pub instance: Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: Device,
    pub graphics_queue: vk::Queue,
    pub compute_queue: vk::Queue,
    pub graphics_queue_family_index: u32,
    pub compute_queue_family_index: u32,
    pub capabilities: DeviceCapabilities,
    pub timestamp_period_ns: f32,
    pub memory_allocator: Option<buffer::GpuMemoryAllocator>,
    pub min_scratch_alignment: vk::DeviceSize,
    pub rt_properties: RayTracingDeviceProperties,
    pub accel_struct_ext: Option<acceleration_structure::Device>,
    pub rt_pipeline_ext: Option<ray_tracing_pipeline::Device>,
    pub sync2_ext: synchronization2::Device,
    pub dyn_rendering_ext: dynamic_rendering::Device,
}

impl VulkanContext {
    /// Wraps raw handles created by `RenderPearl`'s `VulkanDevice`.
    ///
    /// # Safety
    /// All passed `u64` handles must be valid non-null Vulkan handles from the active process.
    pub unsafe fn from_raw_handles(
        vk_instance_raw: u64,
        vk_physical_device_raw: u64,
        vk_device_raw: u64,
        graphics_queue_raw: u64,
        compute_queue_raw: u64,
    ) -> Result<Self, &'static str> {
        unsafe {
            Self::from_raw_handles_with_families(
                vk_instance_raw,
                vk_physical_device_raw,
                vk_device_raw,
                graphics_queue_raw,
                compute_queue_raw,
                0,
                0,
            )
        }
    }

    /// Wraps raw handles created by `RenderPearl`'s `VulkanDevice` along with explicit queue family indices.
    ///
    /// # Safety
    /// All passed `u64` handles must be valid non-null Vulkan handles from the active process.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn from_raw_handles_with_families(
        vk_instance_raw: u64,
        vk_physical_device_raw: u64,
        vk_device_raw: u64,
        graphics_queue_raw: u64,
        compute_queue_raw: u64,
        graphics_queue_family_index: u32,
        compute_queue_family_index: u32,
    ) -> Result<Self, &'static str> {
        if vk_instance_raw == 0 || vk_physical_device_raw == 0 || vk_device_raw == 0 {
            return Err("Null Vulkan handle passed to VulkanContext::from_raw_handles");
        }

        let entry = device_merger::load_vulkan_entry();
        let vk_instance = vk::Instance::from_raw(vk_instance_raw);
        let physical_device = vk::PhysicalDevice::from_raw(vk_physical_device_raw);
        let vk_device = vk::Device::from_raw(vk_device_raw);
        let graphics_queue = vk::Queue::from_raw(graphics_queue_raw);
        let compute_queue = if compute_queue_raw != 0 {
            vk::Queue::from_raw(compute_queue_raw)
        } else {
            graphics_queue
        };

        let instance = unsafe { Instance::load(entry.static_fn(), vk_instance) };
        let device = unsafe { Device::load(instance.fp_v1_0(), vk_device) };

        let mut caps =
            device_merger::capabilities_for_device(vk_device_raw).unwrap_or(DeviceCapabilities {
                buffer_device_address: true,
                descriptor_indexing: true,
                synchronization2: true,
                dynamic_rendering: true,
                ..DeviceCapabilities::default()
            });

        // Query physical device AS scratch alignment and RT pipeline handle properties
        let mut as_props = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut rt_props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default();
        if caps.acceleration_structure {
            props2 = props2.push_next(&mut as_props);
        }
        if caps.ray_tracing_pipeline {
            props2 = props2.push_next(&mut rt_props);
        }
        unsafe {
            instance.get_physical_device_properties2(physical_device, &mut props2);
        }

        let timestamp_period_ns = props2.properties.limits.timestamp_period;
        let memory_allocator = buffer::GpuMemoryAllocator::new(
            instance.clone(),
            physical_device,
            device.clone(),
            caps.buffer_device_address,
            caps.vram_bytes,
        )
        .ok();

        let min_scratch_alignment = u64::from(
            as_props
                .min_acceleration_structure_scratch_offset_alignment
                .max(256),
        );
        let rt_properties = if caps.ray_tracing_pipeline && rt_props.shader_group_handle_size > 0 {
            RayTracingDeviceProperties {
                shader_group_handle_size: rt_props.shader_group_handle_size,
                shader_group_handle_alignment: rt_props.shader_group_handle_alignment.max(1),
                shader_group_base_alignment: rt_props.shader_group_base_alignment.max(1),
                max_ray_recursion_depth: rt_props.max_ray_recursion_depth.max(1),
            }
        } else {
            RayTracingDeviceProperties::default()
        };
        caps.max_ray_recursion_depth = rt_properties.max_ray_recursion_depth;

        let accel_struct_ext = caps
            .acceleration_structure
            .then(|| acceleration_structure::Device::new(&instance, &device));
        let rt_pipeline_ext = caps
            .ray_tracing_pipeline
            .then(|| ray_tracing_pipeline::Device::new(&instance, &device));
        let sync2_ext = synchronization2::Device::new(&instance, &device);
        let dyn_rendering_ext = dynamic_rendering::Device::new(&instance, &device);

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            graphics_queue,
            compute_queue,
            graphics_queue_family_index,
            compute_queue_family_index,
            capabilities: caps,
            timestamp_period_ns,
            memory_allocator,
            min_scratch_alignment,
            rt_properties,
            accel_struct_ext,
            rt_pipeline_ext,
            sync2_ext,
            dyn_rendering_ext,
        })
    }

    pub fn set_queue_family_indices(&mut self, graphics_family: u32, compute_family: u32) {
        self.graphics_queue_family_index = graphics_family;
        self.compute_queue_family_index = compute_family;
    }

    /// Queries the 64-bit GPU `VkDeviceAddress` of a `VkBuffer` created with
    /// `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT`.
    ///
    /// # Safety
    /// `vk_buffer` must be a valid `VkBuffer` created with `SHADER_DEVICE_ADDRESS_BIT`.
    pub unsafe fn buffer_device_address(&self, vk_buffer: vk::Buffer) -> vk::DeviceAddress {
        if vk_buffer == vk::Buffer::null() || !self.capabilities.buffer_device_address {
            return 0;
        }
        let info = vk::BufferDeviceAddressInfo::default().buffer(vk_buffer);
        unsafe { self.device.get_buffer_device_address(&info) }
    }

    /// Emits a `VK_KHR_synchronization2` global memory barrier on `cmd`.
    ///
    /// # Safety
    /// `cmd` must be in the recording state.
    pub unsafe fn cmd_memory_barrier2(
        &self,
        cmd: vk::CommandBuffer,
        src_stage: vk::PipelineStageFlags2,
        src_access: vk::AccessFlags2,
        dst_stage: vk::PipelineStageFlags2,
        dst_access: vk::AccessFlags2,
    ) {
        let barrier = [vk::MemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)];
        let dep = vk::DependencyInfo::default().memory_barriers(&barrier);
        unsafe {
            self.sync2_ext.cmd_pipeline_barrier2(cmd, &dep);
        }
    }

    /// Queries live VRAM usage and driver budget via `VK_EXT_memory_budget` (or falls back to
    /// total device-local heap capacity when the extension is absent).
    pub fn query_vram_budget(&self) -> VramBudgetSnapshot {
        device_merger::query_vram_budget(
            &self.instance,
            self.physical_device,
            self.capabilities.has_memory_budget,
        )
    }

    /// Computes recommended OOM/pressure mitigation actions for the current VRAM budget snapshot.
    pub fn vram_pressure_policy(&self) -> VramPressurePolicy {
        VramPressurePolicy::for_snapshot(self.query_vram_budget())
    }

    /// Evaluates the active platform/MoltenVK render tier cap for this device.
    pub fn platform_tier_policy(&self) -> PlatformTierPolicy {
        device_merger::evaluate_platform_tier_policy(
            &self.capabilities,
            self.capabilities.is_moltenvk,
            cfg!(target_os = "macos"),
        )
    }

    /// Creates a ring-buffered GPU timestamp profiler using this device's `timestamp_period_ns`.
    pub fn create_timestamp_profiler(
        &self,
        frames_in_flight: usize,
        max_passes_per_frame: usize,
    ) -> Result<GpuTimestampProfiler, vk::Result> {
        GpuTimestampProfiler::new(
            &self.device,
            self.timestamp_period_ns,
            frames_in_flight,
            max_passes_per_frame,
        )
    }
}
