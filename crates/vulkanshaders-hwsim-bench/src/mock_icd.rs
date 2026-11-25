// SPDX-License-Identifier: GPL-3.0-only
// Additional permissions under GPLv3 Section 7 apply to optional proprietary vendor SDK
// runtime loading; see LICENSE-EXCEPTION.md in the repository root.

//! Stateful, thread-isolated in-process Mock Vulkan ICD built on `ash 0.38`.
//!
//! Provides real heap-allocated dispatchable handle structs (`ICD_LOADER_MAGIC = 0x01CDC0DE`),
//! panic-safe `extern "system"` trampolines, raw-pointer `pNext` cycle/depth/extension validation,
//! `VkImageLayout` state machine tracking across both `vkCmdPipelineBarrier` and
//! `vkCmdPipelineBarrier2`, and deterministic leak/double-free detection with zero GPU hardware.

use crate::profiles::GpuProfile;
use ash::vk::{self, Handle};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::{CStr, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Standard Khronos Vulkan ICD dispatch table magic word.
pub const ICD_LOADER_MAGIC: usize = 0x01CD_C0DE;

/// Maximum `pNext` chain depth before rejecting as cyclic or malformed.
pub const MAX_PNEXT_DEPTH: usize = 64;

/// Custom Vulkan structure type for `VkPhysicalDeviceShaderFloat8FeaturesEXT` (`VK_EXT_shader_float8`).
pub const STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT: i32 = 1_000_573_000;

/// Recorded Vulkan API call entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VkCallRecord {
    pub seq: u64,
    pub function: String,
    pub result_code: i32,
    pub detail: String,
}

/// Summary of leaked or invalid Vulkan object handles at teardown.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LeakReport {
    pub leaked_images: usize,
    pub leaked_image_views: usize,
    pub leaked_command_pools: usize,
    pub leaked_command_buffers: usize,
    pub violations: Vec<String>,
    pub panicked_in_ffi: Option<String>,
}

impl LeakReport {
    pub fn is_clean(&self) -> bool {
        self.leaked_images == 0
            && self.leaked_image_views == 0
            && self.leaked_command_pools == 0
            && self.leaked_command_buffers == 0
            && self.violations.is_empty()
            && self.panicked_in_ffi.is_none()
    }
}

#[derive(Debug, Clone)]
struct TrackedImage {
    format: vk::Format,
    extent: vk::Extent3D,
    usage: vk::ImageUsageFlags,
    layout: vk::ImageLayout,
}

#[derive(Debug, Clone)]
struct TrackedImageView {
    image_raw: u64,
    format: vk::Format,
}

#[derive(Debug, Default)]
struct IcdInner {
    next_seq: u64,
    next_non_dispatchable_id: u64,
    enabled_instance_exts: BTreeSet<String>,
    enabled_device_exts: BTreeSet<String>,
    images: HashMap<u64, TrackedImage>,
    image_views: HashMap<u64, TrackedImageView>,
    command_pools: BTreeSet<u64>,
    command_buffers: BTreeSet<u64>,
    destroyed_handles: BTreeSet<u64>,
    call_log: Vec<VkCallRecord>,
    violations: Vec<String>,
    panic_message: Option<String>,
}

impl IcdInner {
    fn alloc_non_dispatchable(&mut self, tag: u8) -> u64 {
        self.next_non_dispatchable_id = self.next_non_dispatchable_id.saturating_add(1);
        ((tag as u64) << 56) | self.next_non_dispatchable_id
    }

    fn record(&mut self, function: &str, result_code: i32, detail: impl Into<String>) {
        self.next_seq = self.next_seq.saturating_add(1);
        self.call_log.push(VkCallRecord {
            seq: self.next_seq,
            function: function.to_string(),
            result_code,
            detail: detail.into(),
        });
    }

    fn push_violation(&mut self, msg: impl Into<String>) {
        self.violations.push(msg.into());
    }
}

/// Shared state for a single isolated Mock Vulkan ICD session.
pub struct MockIcdState {
    pub profile: GpuProfile,
    inner: Mutex<IcdInner>,
    panicked: AtomicBool,
    next_generation: AtomicU64,
}

impl MockIcdState {
    pub fn new(profile: GpuProfile) -> Arc<Self> {
        Arc::new(Self {
            profile,
            inner: Mutex::new(IcdInner::default()),
            panicked: AtomicBool::new(false),
            next_generation: AtomicU64::new(1),
        })
    }

    fn next_gen(&self) -> u64 {
        self.next_generation.fetch_add(1, Ordering::Relaxed)
    }

    fn latch_panic(&self, msg: String) {
        self.panicked.store(true, Ordering::Release);
        let mut inner = self.inner.lock();
        if inner.panic_message.is_none() {
            inner.panic_message = Some(msg);
        }
    }

    /// Queries the currently tracked `VkImageLayout` for a mock `VkImage`.
    pub fn query_image_layout(&self, image: vk::Image) -> Option<vk::ImageLayout> {
        let inner = self.inner.lock();
        inner.images.get(&image.as_raw()).map(|img| img.layout)
    }

    /// Queries the tracked `VkFormat` and extent for a mock `VkImage`.
    pub fn query_image_info(&self, image: vk::Image) -> Option<(vk::Format, vk::Extent3D, vk::ImageUsageFlags)> {
        let inner = self.inner.lock();
        inner
            .images
            .get(&image.as_raw())
            .map(|img| (img.format, img.extent, img.usage))
    }

    /// Returns a snapshot of all recorded Vulkan calls.
    pub fn call_log(&self) -> Vec<VkCallRecord> {
        self.inner.lock().call_log.clone()
    }

    /// Returns all enabled device extensions recorded at `vkCreateDevice`.
    pub fn enabled_device_extensions(&self) -> BTreeSet<String> {
        self.inner.lock().enabled_device_exts.clone()
    }

    /// Evaluates handle leaks, contract violations, and latched FFI panics.
    pub fn leak_report(&self) -> LeakReport {
        let inner = self.inner.lock();
        LeakReport {
            leaked_images: inner.images.len(),
            leaked_image_views: inner.image_views.len(),
            leaked_command_pools: inner.command_pools.len(),
            leaked_command_buffers: inner.command_buffers.len(),
            violations: inner.violations.clone(),
            panicked_in_ffi: inner.panic_message.clone(),
        }
    }

    /// Returns a deterministic summary map of call counts per Vulkan entrypoint.
    pub fn call_counts(&self) -> BTreeMap<String, usize> {
        let inner = self.inner.lock();
        let mut counts = BTreeMap::new();
        for rec in &inner.call_log {
            *counts.entry(rec.function.clone()).or_insert(0) += 1;
        }
        counts
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HandleKind {
    Instance = 1,
    PhysicalDevice = 2,
    Device = 3,
    Queue = 4,
    CommandBuffer = 5,
}

/// Real `#[repr(C)]` dispatchable handle backing struct with Khronos ICD loader magic.
#[repr(C)]
struct DispatchableHandle {
    loader_magic: usize,
    kind: HandleKind,
    generation: u64,
    state: Arc<MockIcdState>,
}

impl DispatchableHandle {
    fn alloc(kind: HandleKind, state: Arc<MockIcdState>) -> u64 {
        let generation = state.next_gen();
        let boxed = Box::new(Self {
            loader_magic: ICD_LOADER_MAGIC,
            kind,
            generation,
            state,
        });
        Box::into_raw(boxed) as usize as u64
    }

    /// Validates pointer alignment, non-nullness, loader magic, and expected kind before borrowing.
    ///
    /// # Safety
    /// `raw` must either be `0` or a pointer previously produced by [`Self::alloc`] that has not
    /// been freed.
    unsafe fn borrow<'a>(raw: u64, expected: HandleKind) -> Option<&'a Self> {
        if raw == 0 {
            return None;
        }
        let ptr = raw as usize as *const Self;
        if ptr.is_null() || !ptr.is_aligned() {
            return None;
        }
        // SAFETY: Caller guarantees `raw` is a live `DispatchableHandle` allocation.
        let h = unsafe { &*ptr };
        if h.loader_magic != ICD_LOADER_MAGIC || h.kind != expected || h.generation == 0 {
            return None;
        }
        Some(h)
    }

    /// Reclaims a dispatchable handle allocation.
    ///
    /// # Safety
    /// `raw` must be a live allocation from [`Self::alloc`] and never used again afterward.
    unsafe fn reclaim(raw: u64, expected: HandleKind) -> Option<Arc<MockIcdState>> {
        if raw == 0 {
            return None;
        }
        let ptr = raw as usize as *mut Self;
        if ptr.is_null() || !ptr.is_aligned() {
            return None;
        }
        // SAFETY: Checked non-null and aligned; verifies magic/kind before dropping Box.
        let magic_ok = unsafe { (*ptr).loader_magic == ICD_LOADER_MAGIC && (*ptr).kind == expected };
        if !magic_ok {
            return None;
        }
        // SAFETY: Reclaiming Box created via `Box::into_raw`.
        let boxed = unsafe { Box::from_raw(ptr) };
        Some(boxed.state)
    }
}

thread_local! {
    static ACTIVE_ICD: RefCell<Option<Arc<MockIcdState>>> = const { RefCell::new(None) };
}

fn current_tls_state() -> Option<Arc<MockIcdState>> {
    ACTIVE_ICD.with(|slot| slot.borrow().clone())
}

/// RAII guard installing a thread-isolated Mock Vulkan ICD for the current thread.
pub struct MockIcdGuard {
    prev: Option<Arc<MockIcdState>>,
    state: Arc<MockIcdState>,
}

impl MockIcdGuard {
    /// Installs a mock ICD configured with `profile` and returns `(guard, ash::Entry)`.
    pub fn install(profile: GpuProfile) -> (Self, ash::Entry) {
        let state = MockIcdState::new(profile);
        let prev = ACTIVE_ICD.with(|slot| slot.borrow_mut().replace(state.clone()));
        let static_fn = ash::StaticFn {
            get_instance_proc_addr: mock_vk_get_instance_proc_addr,
        };
        // SAFETY: `mock_vk_get_instance_proc_addr` implements valid Vulkan 1.0+ proc-addr semantics.
        let entry = unsafe { ash::Entry::from_static_fn(static_fn) };
        (Self { prev, state }, entry)
    }

    pub fn state(&self) -> &Arc<MockIcdState> {
        &self.state
    }
}

impl Drop for MockIcdGuard {
    fn drop(&mut self) {
        let prev = self.prev.take();
        ACTIVE_ICD.with(|slot| {
            *slot.borrow_mut() = prev;
        });
    }
}

/// Validates a raw Vulkan `pNext` chain for pointer alignment, cycles, depth bounds, and
/// extension gating. Never creates `&mut` references to caller memory.
///
/// # Safety
/// Every reachable non-null `p_next` pointer must point to a readable `vk::BaseInStructure`.
pub unsafe fn validate_pnext_chain(
    head: *const c_void,
    profile: Option<&GpuProfile>,
) -> Result<usize, String> {
    let mut visited = [0usize; MAX_PNEXT_DEPTH];
    let mut depth = 0usize;
    let mut cursor = head as *const vk::BaseInStructure<'_>;

    while !cursor.is_null() {
        if depth >= MAX_PNEXT_DEPTH {
            return Err(format!(
                "pNext chain exceeded max depth {MAX_PNEXT_DEPTH} (possible cycle)"
            ));
        }
        let addr = cursor as usize;
        if visited[..depth].contains(&addr) {
            return Err(format!("pNext cycle detected at address {addr:#x}"));
        }
        if !cursor.is_aligned() {
            return Err(format!("unaligned pNext pointer at {addr:#x}"));
        }
        visited[depth] = addr;
        depth += 1;

        // SAFETY: Verified non-null and aligned above.
        let node = unsafe { &*cursor };
        let stype_raw = node.s_type.as_raw();
        if stype_raw == STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT
            && let Some(prof) = profile
            && !prof.has_extension("VK_EXT_shader_float8")
        {
            return Err(format!(
                "VK_EXT_shader_float8 struct in pNext chain on GPU `{}` without VK_EXT_shader_float8",
                prof.display_name
            ));
        }
        cursor = node.p_next;
    }

    Ok(depth)
}

fn write_vk_cstr(dst: &mut [c_char], src: &str) {
    dst.fill(0);
    if dst.is_empty() {
        return;
    }
    let max_copy = dst.len() - 1;
    for (slot, &byte) in dst.iter_mut().take(max_copy).zip(src.as_bytes().iter()) {
        *slot = byte as c_char;
    }
    dst[dst.len() - 1] = 0;
}

fn read_cstr_lossy(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: Caller passes a valid NUL-terminated C string per Vulkan ABI.
    let s = unsafe { CStr::from_ptr(ptr) };
    Some(s.to_string_lossy().into_owned())
}

fn ffi_guard_result<F>(state: Option<&Arc<MockIcdState>>, f: F) -> vk::Result
where
    F: FnOnce() -> vk::Result,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(res) => res,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic in mock Vulkan ICD".to_string());
            if let Some(st) = state {
                st.latch_panic(msg);
            }
            vk::Result::ERROR_UNKNOWN
        }
    }
}

fn ffi_guard_void<F>(state: Option<&Arc<MockIcdState>>, f: F)
where
    F: FnOnce(),
{
    if let Err(payload) = catch_unwind(AssertUnwindSafe(f)) {
        let msg = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown panic in mock Vulkan ICD".to_string());
        if let Some(st) = state {
            st.latch_panic(msg);
        }
    }
}

// ============================================================================
// Mock Vulkan Entry Points
// ============================================================================

/// Exported mock `vkGetInstanceProcAddr` compatible with `ash::StaticFn`.
pub unsafe extern "system" fn mock_vk_get_instance_proc_addr(
    instance: vk::Instance,
    p_name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let name = read_cstr_lossy(p_name)?;
        let is_null_instance = instance == vk::Instance::null();

        if is_null_instance {
            return match name.as_str() {
                "vkGetInstanceProcAddr" => {
                    Some(unsafe { std::mem::transmute::<vk::PFN_vkGetInstanceProcAddr, unsafe extern "system" fn()>(mock_vk_get_instance_proc_addr) })
                }
                "vkEnumerateInstanceVersion" => {
                    Some(unsafe { std::mem::transmute::<vk::PFN_vkEnumerateInstanceVersion, unsafe extern "system" fn()>(mock_vk_enumerate_instance_version) })
                }
                "vkEnumerateInstanceExtensionProperties" => {
                    Some(unsafe { std::mem::transmute::<vk::PFN_vkEnumerateInstanceExtensionProperties, unsafe extern "system" fn()>(mock_vk_enumerate_instance_extension_properties) })
                }
                "vkEnumerateInstanceLayerProperties" => {
                    Some(unsafe { std::mem::transmute::<vk::PFN_vkEnumerateInstanceLayerProperties, unsafe extern "system" fn()>(mock_vk_enumerate_instance_layer_properties) })
                }
                "vkCreateInstance" => {
                    Some(unsafe { std::mem::transmute::<vk::PFN_vkCreateInstance, unsafe extern "system" fn()>(mock_vk_create_instance) })
                }
                _ => None,
            };
        }

        match name.as_str() {
            "vkGetInstanceProcAddr" => Some(unsafe {
                std::mem::transmute::<vk::PFN_vkGetInstanceProcAddr, unsafe extern "system" fn()>(
                    mock_vk_get_instance_proc_addr,
                )
            }),
            "vkDestroyInstance" => Some(unsafe {
                std::mem::transmute::<vk::PFN_vkDestroyInstance, unsafe extern "system" fn()>(
                    mock_vk_destroy_instance,
                )
            }),
            "vkEnumeratePhysicalDevices" => Some(unsafe {
                std::mem::transmute::<vk::PFN_vkEnumeratePhysicalDevices, unsafe extern "system" fn()>(
                    mock_vk_enumerate_physical_devices,
                )
            }),
            "vkGetPhysicalDeviceProperties" => Some(unsafe {
                std::mem::transmute::<vk::PFN_vkGetPhysicalDeviceProperties, unsafe extern "system" fn()>(
                    mock_vk_get_physical_device_properties,
                )
            }),
            "vkGetPhysicalDeviceProperties2" | "vkGetPhysicalDeviceProperties2KHR" => Some(unsafe {
                std::mem::transmute::<vk::PFN_vkGetPhysicalDeviceProperties2, unsafe extern "system" fn()>(
                    mock_vk_get_physical_device_properties2,
                )
            }),
            "vkGetPhysicalDeviceFeatures2" | "vkGetPhysicalDeviceFeatures2KHR" => Some(unsafe {
                std::mem::transmute::<vk::PFN_vkGetPhysicalDeviceFeatures2, unsafe extern "system" fn()>(
                    mock_vk_get_physical_device_features2,
                )
            }),
            "vkEnumerateDeviceExtensionProperties" => Some(unsafe {
                std::mem::transmute::<vk::PFN_vkEnumerateDeviceExtensionProperties, unsafe extern "system" fn()>(
                    mock_vk_enumerate_device_extension_properties,
                )
            }),
            "vkGetPhysicalDeviceQueueFamilyProperties" => Some(unsafe {
                std::mem::transmute::<vk::PFN_vkGetPhysicalDeviceQueueFamilyProperties, unsafe extern "system" fn()>(
                    mock_vk_get_physical_device_queue_family_properties,
                )
            }),
            "vkCreateDevice" => Some(unsafe {
                std::mem::transmute::<vk::PFN_vkCreateDevice, unsafe extern "system" fn()>(
                    mock_vk_create_device,
                )
            }),
            "vkGetDeviceProcAddr" => Some(unsafe {
                std::mem::transmute::<vk::PFN_vkGetDeviceProcAddr, unsafe extern "system" fn()>(
                    mock_vk_get_device_proc_addr,
                )
            }),
            _ => resolve_device_symbol(&name, None),
        }
    }));

    result.unwrap_or(None)
}

/// Exported mock `vkGetDeviceProcAddr` compatible with `vk::PFN_vkGetDeviceProcAddr`.
pub unsafe extern "system" fn mock_vk_get_device_proc_addr(
    device: vk::Device,
    p_name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let name = read_cstr_lossy(p_name)?;
        // Per Vulkan spec, vkGetDeviceProcAddr only resolves device-level commands.
        if matches!(
            name.as_str(),
            "vkCreateInstance"
                | "vkDestroyInstance"
                | "vkEnumeratePhysicalDevices"
                | "vkGetPhysicalDeviceProperties"
                | "vkGetPhysicalDeviceProperties2"
        ) {
            return None;
        }
        let handle = unsafe { DispatchableHandle::borrow(device.as_raw(), HandleKind::Device) };
        let state = handle.map(|h| h.state.clone());
        resolve_device_symbol(&name, state.as_ref())
    }));
    result.unwrap_or(None)
}

fn resolve_device_symbol(
    name: &str,
    state: Option<&Arc<MockIcdState>>,
) -> vk::PFN_vkVoidFunction {
    // Gate VK_KHR_synchronization2 KHR alias when neither core 1.3 nor extension is enabled.
    if name == "vkCmdPipelineBarrier2KHR"
        && let Some(st) = state
    {
        let enabled = st.inner.lock().enabled_device_exts.contains("VK_KHR_synchronization2");
        if !enabled {
            st.inner.lock().push_violation(
                "vkGetDeviceProcAddr requested vkCmdPipelineBarrier2KHR without enabling VK_KHR_synchronization2",
            );
            return None;
        }
    }

    match name {
        "vkGetDeviceProcAddr" => Some(unsafe {
            std::mem::transmute::<vk::PFN_vkGetDeviceProcAddr, unsafe extern "system" fn()>(
                mock_vk_get_device_proc_addr,
            )
        }),
        "vkDestroyDevice" => Some(unsafe {
            std::mem::transmute::<vk::PFN_vkDestroyDevice, unsafe extern "system" fn()>(
                mock_vk_destroy_device,
            )
        }),
        "vkGetDeviceQueue" => Some(unsafe {
            std::mem::transmute::<vk::PFN_vkGetDeviceQueue, unsafe extern "system" fn()>(
                mock_vk_get_device_queue,
            )
        }),
        "vkCreateImage" => Some(unsafe {
            std::mem::transmute::<vk::PFN_vkCreateImage, unsafe extern "system" fn()>(
                mock_vk_create_image,
            )
        }),
        "vkDestroyImage" => Some(unsafe {
            std::mem::transmute::<vk::PFN_vkDestroyImage, unsafe extern "system" fn()>(
                mock_vk_destroy_image,
            )
        }),
        "vkCreateImageView" => Some(unsafe {
            std::mem::transmute::<vk::PFN_vkCreateImageView, unsafe extern "system" fn()>(
                mock_vk_create_image_view,
            )
        }),
        "vkDestroyImageView" => Some(unsafe {
            std::mem::transmute::<vk::PFN_vkDestroyImageView, unsafe extern "system" fn()>(
                mock_vk_destroy_image_view,
            )
        }),
        "vkCreateCommandPool" => Some(unsafe {
            std::mem::transmute::<vk::PFN_vkCreateCommandPool, unsafe extern "system" fn()>(
                mock_vk_create_command_pool,
            )
        }),
        "vkDestroyCommandPool" => Some(unsafe {
            std::mem::transmute::<vk::PFN_vkDestroyCommandPool, unsafe extern "system" fn()>(
                mock_vk_destroy_command_pool,
            )
        }),
        "vkAllocateCommandBuffers" => Some(unsafe {
            std::mem::transmute::<vk::PFN_vkAllocateCommandBuffers, unsafe extern "system" fn()>(
                mock_vk_allocate_command_buffers,
            )
        }),
        "vkFreeCommandBuffers" => Some(unsafe {
            std::mem::transmute::<vk::PFN_vkFreeCommandBuffers, unsafe extern "system" fn()>(
                mock_vk_free_command_buffers,
            )
        }),
        "vkBeginCommandBuffer" => Some(unsafe {
            std::mem::transmute::<vk::PFN_vkBeginCommandBuffer, unsafe extern "system" fn()>(
                mock_vk_begin_command_buffer,
            )
        }),
        "vkEndCommandBuffer" => Some(unsafe {
            std::mem::transmute::<vk::PFN_vkEndCommandBuffer, unsafe extern "system" fn()>(
                mock_vk_end_command_buffer,
            )
        }),
        "vkCmdPipelineBarrier" => Some(unsafe {
            std::mem::transmute::<vk::PFN_vkCmdPipelineBarrier, unsafe extern "system" fn()>(
                mock_vk_cmd_pipeline_barrier,
            )
        }),
        "vkCmdPipelineBarrier2" | "vkCmdPipelineBarrier2KHR" => Some(unsafe {
            std::mem::transmute::<vk::PFN_vkCmdPipelineBarrier2, unsafe extern "system" fn()>(
                mock_vk_cmd_pipeline_barrier2,
            )
        }),
        _ => None,
    }
}

unsafe extern "system" fn mock_vk_enumerate_instance_version(p_api_version: *mut u32) -> vk::Result {
    let st = current_tls_state();
    ffi_guard_result(st.as_ref(), || {
        if p_api_version.is_null() {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }
        let ver = st
            .as_ref()
            .map(|s| s.profile.api_version)
            .unwrap_or_else(|| vk::make_api_version(0, 1, 3, 281));
        unsafe { *p_api_version = ver };
        vk::Result::SUCCESS
    })
}

unsafe extern "system" fn mock_vk_enumerate_instance_extension_properties(
    _p_layer_name: *const c_char,
    p_property_count: *mut u32,
    p_properties: *mut vk::ExtensionProperties,
) -> vk::Result {
    let st = current_tls_state();
    ffi_guard_result(st.as_ref(), || {
        if p_property_count.is_null() {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }
        let exts = [
            "VK_KHR_surface",
            "VK_KHR_get_physical_device_properties2",
            "VK_EXT_debug_utils",
        ];
        if p_properties.is_null() {
            unsafe { *p_property_count = exts.len() as u32 };
            return vk::Result::SUCCESS;
        }
        let cap = unsafe { *p_property_count } as usize;
        let to_write = cap.min(exts.len());
        for (idx, ext_name) in exts.iter().take(to_write).enumerate() {
            let mut prop = vk::ExtensionProperties::default();
            write_vk_cstr(&mut prop.extension_name, ext_name);
            prop.spec_version = 1;
            unsafe { *p_properties.add(idx) = prop };
        }
        unsafe { *p_property_count = to_write as u32 };
        if to_write < exts.len() {
            vk::Result::INCOMPLETE
        } else {
            vk::Result::SUCCESS
        }
    })
}

unsafe extern "system" fn mock_vk_enumerate_instance_layer_properties(
    p_property_count: *mut u32,
    _p_properties: *mut vk::LayerProperties,
) -> vk::Result {
    let st = current_tls_state();
    ffi_guard_result(st.as_ref(), || {
        if !p_property_count.is_null() {
            unsafe { *p_property_count = 0 };
        }
        vk::Result::SUCCESS
    })
}

unsafe extern "system" fn mock_vk_create_instance(
    p_create_info: *const vk::InstanceCreateInfo<'_>,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
    p_instance: *mut vk::Instance,
) -> vk::Result {
    let st = current_tls_state();
    ffi_guard_result(st.as_ref(), || {
        let Some(state) = st.clone() else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        if p_create_info.is_null() || p_instance.is_null() {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }
        let info = unsafe { &*p_create_info };
        if let Err(err) = unsafe { validate_pnext_chain(info.p_next, Some(&state.profile)) } {
            state.inner.lock().push_violation(err);
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }
        {
            let mut inner = state.inner.lock();
            if !info.pp_enabled_extension_names.is_null() {
                for idx in 0..info.enabled_extension_count as usize {
                    let ext_ptr = unsafe { *info.pp_enabled_extension_names.add(idx) };
                    if let Some(ext_name) = read_cstr_lossy(ext_ptr) {
                        inner.enabled_instance_exts.insert(ext_name);
                    }
                }
            }
            inner.record("vkCreateInstance", 0, state.profile.display_name);
        }
        let raw = DispatchableHandle::alloc(HandleKind::Instance, state);
        unsafe { *p_instance = vk::Instance::from_raw(raw) };
        vk::Result::SUCCESS
    })
}

unsafe extern "system" fn mock_vk_destroy_instance(
    instance: vk::Instance,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    let state = unsafe { DispatchableHandle::reclaim(instance.as_raw(), HandleKind::Instance) };
    ffi_guard_void(state.as_ref(), || {
        if let Some(st) = &state {
            st.inner.lock().record("vkDestroyInstance", 0, "destroyed");
        }
    });
}

unsafe extern "system" fn mock_vk_enumerate_physical_devices(
    instance: vk::Instance,
    p_physical_device_count: *mut u32,
    p_physical_devices: *mut vk::PhysicalDevice,
) -> vk::Result {
    let handle = unsafe { DispatchableHandle::borrow(instance.as_raw(), HandleKind::Instance) };
    let st = handle.map(|h| h.state.clone());
    ffi_guard_result(st.as_ref(), || {
        let Some(state) = st.clone() else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        if p_physical_device_count.is_null() {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }
        if p_physical_devices.is_null() {
            unsafe { *p_physical_device_count = 1 };
            return vk::Result::SUCCESS;
        }
        let cap = unsafe { *p_physical_device_count };
        if cap == 0 {
            return vk::Result::INCOMPLETE;
        }
        let raw = DispatchableHandle::alloc(HandleKind::PhysicalDevice, state.clone());
        unsafe {
            *p_physical_devices = vk::PhysicalDevice::from_raw(raw);
            *p_physical_device_count = 1;
        }
        state
            .inner
            .lock()
            .record("vkEnumeratePhysicalDevices", 0, "1 simulated GPU");
        vk::Result::SUCCESS
    })
}

unsafe extern "system" fn mock_vk_get_physical_device_properties(
    physical_device: vk::PhysicalDevice,
    p_properties: *mut vk::PhysicalDeviceProperties,
) {
    let handle =
        unsafe { DispatchableHandle::borrow(physical_device.as_raw(), HandleKind::PhysicalDevice) };
    let st = handle.map(|h| h.state.clone());
    ffi_guard_void(st.as_ref(), || {
        let (Some(state), false) = (st.as_ref(), p_properties.is_null()) else {
            return;
        };
        let mut props = vk::PhysicalDeviceProperties::default();
        props.api_version = state.profile.api_version;
        props.driver_version = state.profile.driver_version;
        props.vendor_id = state.profile.asic.vendor_id;
        props.device_id = state.profile.asic.device_id;
        props.device_type = if state.profile.asic.is_integrated {
            vk::PhysicalDeviceType::INTEGRATED_GPU
        } else {
            vk::PhysicalDeviceType::DISCRETE_GPU
        };
        write_vk_cstr(&mut props.device_name, state.profile.display_name);
        props.limits.max_compute_work_group_invocations =
            state.profile.max_compute_workgroup_invocations;
        props.limits.max_compute_shared_memory_size = state.profile.max_compute_shared_memory_size;
        props.limits.max_bound_descriptor_sets = state.profile.max_bound_descriptor_sets;
        unsafe { *p_properties = props };
        state.inner.lock().record(
            "vkGetPhysicalDeviceProperties",
            0,
            state.profile.display_name,
        );
    });
}

unsafe extern "system" fn mock_vk_get_physical_device_properties2(
    physical_device: vk::PhysicalDevice,
    p_properties: *mut vk::PhysicalDeviceProperties2<'_>,
) {
    let handle =
        unsafe { DispatchableHandle::borrow(physical_device.as_raw(), HandleKind::PhysicalDevice) };
    let st = handle.map(|h| h.state.clone());
    ffi_guard_void(st.as_ref(), || {
        let (Some(state), false) = (st.as_ref(), p_properties.is_null()) else {
            return;
        };
        let p_next = unsafe { (*p_properties).p_next as *const c_void };
        if let Err(err) = unsafe { validate_pnext_chain(p_next, Some(&state.profile)) } {
            state.inner.lock().push_violation(err);
            return;
        }
        unsafe {
            mock_vk_get_physical_device_properties(
                physical_device,
                &mut (*p_properties).properties as *mut vk::PhysicalDeviceProperties,
            );
        }
    });
}

unsafe extern "system" fn mock_vk_get_physical_device_features2(
    physical_device: vk::PhysicalDevice,
    p_features: *mut vk::PhysicalDeviceFeatures2<'_>,
) {
    let handle =
        unsafe { DispatchableHandle::borrow(physical_device.as_raw(), HandleKind::PhysicalDevice) };
    let st = handle.map(|h| h.state.clone());
    ffi_guard_void(st.as_ref(), || {
        let (Some(state), false) = (st.as_ref(), p_features.is_null()) else {
            return;
        };
        let p_next = unsafe { (*p_features).p_next as *const c_void };
        if let Err(err) = unsafe { validate_pnext_chain(p_next, Some(&state.profile)) } {
            state.inner.lock().push_violation(err);
            return;
        }
        let mut base = vk::PhysicalDeviceFeatures::default();
        base.shader_int16 = vk::TRUE;
        base.shader_storage_image_write_without_format = vk::TRUE;
        base.shader_storage_image_extended_formats = vk::TRUE;
        unsafe { (*p_features).features = base };
        state
            .inner
            .lock()
            .record("vkGetPhysicalDeviceFeatures2", 0, "populated");
    });
}

unsafe extern "system" fn mock_vk_enumerate_device_extension_properties(
    physical_device: vk::PhysicalDevice,
    _p_layer_name: *const c_char,
    p_property_count: *mut u32,
    p_properties: *mut vk::ExtensionProperties,
) -> vk::Result {
    let handle =
        unsafe { DispatchableHandle::borrow(physical_device.as_raw(), HandleKind::PhysicalDevice) };
    let st = handle.map(|h| h.state.clone());
    ffi_guard_result(st.as_ref(), || {
        let Some(state) = st.clone() else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        if p_property_count.is_null() {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }
        let exts = state.profile.extensions;
        if p_properties.is_null() {
            unsafe { *p_property_count = exts.len() as u32 };
            return vk::Result::SUCCESS;
        }
        let cap = unsafe { *p_property_count } as usize;
        let to_write = cap.min(exts.len());
        for (idx, ext_name) in exts.iter().take(to_write).enumerate() {
            let mut prop = vk::ExtensionProperties::default();
            write_vk_cstr(&mut prop.extension_name, ext_name);
            prop.spec_version = 1;
            unsafe { *p_properties.add(idx) = prop };
        }
        unsafe { *p_property_count = to_write as u32 };
        if to_write < exts.len() {
            vk::Result::INCOMPLETE
        } else {
            vk::Result::SUCCESS
        }
    })
}

unsafe extern "system" fn mock_vk_get_physical_device_queue_family_properties(
    physical_device: vk::PhysicalDevice,
    p_queue_family_property_count: *mut u32,
    p_queue_family_properties: *mut vk::QueueFamilyProperties,
) {
    let handle =
        unsafe { DispatchableHandle::borrow(physical_device.as_raw(), HandleKind::PhysicalDevice) };
    let st = handle.map(|h| h.state.clone());
    ffi_guard_void(st.as_ref(), || {
        if p_queue_family_property_count.is_null() {
            return;
        }
        if p_queue_family_properties.is_null() {
            unsafe { *p_queue_family_property_count = 1 };
            return;
        }
        let cap = unsafe { *p_queue_family_property_count };
        if cap >= 1 {
            let mut q = vk::QueueFamilyProperties::default();
            q.queue_flags = vk::QueueFlags::GRAPHICS
                | vk::QueueFlags::COMPUTE
                | vk::QueueFlags::TRANSFER;
            q.queue_count = 4;
            q.timestamp_valid_bits = 64;
            unsafe {
                *p_queue_family_properties = q;
                *p_queue_family_property_count = 1;
            }
        }
    });
}

unsafe extern "system" fn mock_vk_create_device(
    physical_device: vk::PhysicalDevice,
    p_create_info: *const vk::DeviceCreateInfo<'_>,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
    p_device: *mut vk::Device,
) -> vk::Result {
    let handle =
        unsafe { DispatchableHandle::borrow(physical_device.as_raw(), HandleKind::PhysicalDevice) };
    let st = handle.map(|h| h.state.clone());
    ffi_guard_result(st.as_ref(), || {
        let Some(state) = st.clone() else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        if p_create_info.is_null() || p_device.is_null() {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }
        let info = unsafe { &*p_create_info };
        if let Err(err) = unsafe { validate_pnext_chain(info.p_next, Some(&state.profile)) } {
            state.inner.lock().push_violation(err);
            return vk::Result::ERROR_EXTENSION_NOT_PRESENT;
        }

        let mut requested = Vec::new();
        if !info.pp_enabled_extension_names.is_null() {
            for idx in 0..info.enabled_extension_count as usize {
                let ext_ptr = unsafe { *info.pp_enabled_extension_names.add(idx) };
                if let Some(ext_name) = read_cstr_lossy(ext_ptr) {
                    if !state.profile.has_extension(&ext_name) {
                        state.inner.lock().push_violation(format!(
                            "vkCreateDevice requested unsupported extension `{ext_name}` on {}",
                            state.profile.display_name
                        ));
                        return vk::Result::ERROR_EXTENSION_NOT_PRESENT;
                    }
                    requested.push(ext_name);
                }
            }
        }

        {
            let mut inner = state.inner.lock();
            for ext in &requested {
                inner.enabled_device_exts.insert(ext.clone());
            }
            inner.record(
                "vkCreateDevice",
                0,
                format!("enabled {} extension(s)", requested.len()),
            );
        }

        let raw = DispatchableHandle::alloc(HandleKind::Device, state);
        unsafe { *p_device = vk::Device::from_raw(raw) };
        vk::Result::SUCCESS
    })
}

unsafe extern "system" fn mock_vk_destroy_device(
    device: vk::Device,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    let state = unsafe { DispatchableHandle::reclaim(device.as_raw(), HandleKind::Device) };
    ffi_guard_void(state.as_ref(), || {
        if let Some(st) = &state {
            st.inner.lock().record("vkDestroyDevice", 0, "destroyed");
        }
    });
}

unsafe extern "system" fn mock_vk_get_device_queue(
    device: vk::Device,
    _queue_family_index: u32,
    _queue_index: u32,
    p_queue: *mut vk::Queue,
) {
    let handle = unsafe { DispatchableHandle::borrow(device.as_raw(), HandleKind::Device) };
    let st = handle.map(|h| h.state.clone());
    ffi_guard_void(st.as_ref(), || {
        if let (Some(state), false) = (st.clone(), p_queue.is_null()) {
            let raw = DispatchableHandle::alloc(HandleKind::Queue, state);
            unsafe { *p_queue = vk::Queue::from_raw(raw) };
        }
    });
}

unsafe extern "system" fn mock_vk_create_image(
    device: vk::Device,
    p_create_info: *const vk::ImageCreateInfo<'_>,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
    p_image: *mut vk::Image,
) -> vk::Result {
    let handle = unsafe { DispatchableHandle::borrow(device.as_raw(), HandleKind::Device) };
    let st = handle.map(|h| h.state.clone());
    ffi_guard_result(st.as_ref(), || {
        let Some(state) = st.clone() else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        if p_create_info.is_null() || p_image.is_null() {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }
        let info = unsafe { &*p_create_info };
        let mut inner = state.inner.lock();
        let id = inner.alloc_non_dispatchable(0x04);
        inner.images.insert(
            id,
            TrackedImage {
                format: info.format,
                extent: info.extent,
                usage: info.usage,
                layout: info.initial_layout,
            },
        );
        inner.record(
            "vkCreateImage",
            0,
            format!(
                "id={id:#x} {}x{} fmt={:?} init={:?}",
                info.extent.width, info.extent.height, info.format, info.initial_layout
            ),
        );
        unsafe { *p_image = vk::Image::from_raw(id) };
        vk::Result::SUCCESS
    })
}

unsafe extern "system" fn mock_vk_destroy_image(
    device: vk::Device,
    image: vk::Image,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    let handle = unsafe { DispatchableHandle::borrow(device.as_raw(), HandleKind::Device) };
    let st = handle.map(|h| h.state.clone());
    ffi_guard_void(st.as_ref(), || {
        let Some(state) = st.clone() else {
            return;
        };
        let raw = image.as_raw();
        if raw == 0 {
            return;
        }
        let mut inner = state.inner.lock();
        if inner.images.remove(&raw).is_none() {
            inner.push_violation(format!(
                "vkDestroyImage called on unknown or already-destroyed handle {raw:#x}"
            ));
        } else {
            let dangling: Vec<String> = inner
                .image_views
                .iter()
                .filter(|(_, v)| v.image_raw == raw)
                .map(|(vid, v)| format!("{vid:#x}({:?})", v.format))
                .collect();
            if !dangling.is_empty() {
                inner.push_violation(format!(
                    "vkDestroyImage({raw:#x}) called while image view(s) still alive: {}",
                    dangling.join(", ")
                ));
            }
            inner.destroyed_handles.insert(raw);
            inner.record("vkDestroyImage", 0, format!("id={raw:#x}"));
        }
    });
}

unsafe extern "system" fn mock_vk_create_image_view(
    device: vk::Device,
    p_create_info: *const vk::ImageViewCreateInfo<'_>,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
    p_view: *mut vk::ImageView,
) -> vk::Result {
    let handle = unsafe { DispatchableHandle::borrow(device.as_raw(), HandleKind::Device) };
    let st = handle.map(|h| h.state.clone());
    ffi_guard_result(st.as_ref(), || {
        let Some(state) = st.clone() else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        if p_create_info.is_null() || p_view.is_null() {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }
        let info = unsafe { &*p_create_info };
        let mut inner = state.inner.lock();
        if !inner.images.contains_key(&info.image.as_raw()) {
            inner.push_violation(format!(
                "vkCreateImageView references unknown VkImage {:#x}",
                info.image.as_raw()
            ));
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }
        let id = inner.alloc_non_dispatchable(0x05);
        inner.image_views.insert(
            id,
            TrackedImageView {
                image_raw: info.image.as_raw(),
                format: info.format,
            },
        );
        inner.record("vkCreateImageView", 0, format!("id={id:#x}"));
        unsafe { *p_view = vk::ImageView::from_raw(id) };
        vk::Result::SUCCESS
    })
}

unsafe extern "system" fn mock_vk_destroy_image_view(
    device: vk::Device,
    image_view: vk::ImageView,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    let handle = unsafe { DispatchableHandle::borrow(device.as_raw(), HandleKind::Device) };
    let st = handle.map(|h| h.state.clone());
    ffi_guard_void(st.as_ref(), || {
        let Some(state) = st.clone() else {
            return;
        };
        let raw = image_view.as_raw();
        if raw == 0 {
            return;
        }
        let mut inner = state.inner.lock();
        if inner.image_views.remove(&raw).is_none() {
            inner.push_violation(format!(
                "vkDestroyImageView called on unknown or already-destroyed handle {raw:#x}"
            ));
        } else {
            inner.destroyed_handles.insert(raw);
            inner.record("vkDestroyImageView", 0, format!("id={raw:#x}"));
        }
    });
}

unsafe extern "system" fn mock_vk_create_command_pool(
    device: vk::Device,
    _p_create_info: *const vk::CommandPoolCreateInfo<'_>,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
    p_command_pool: *mut vk::CommandPool,
) -> vk::Result {
    let handle = unsafe { DispatchableHandle::borrow(device.as_raw(), HandleKind::Device) };
    let st = handle.map(|h| h.state.clone());
    ffi_guard_result(st.as_ref(), || {
        let Some(state) = st.clone() else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        if p_command_pool.is_null() {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }
        let mut inner = state.inner.lock();
        let id = inner.alloc_non_dispatchable(0x06);
        inner.command_pools.insert(id);
        inner.record("vkCreateCommandPool", 0, format!("id={id:#x}"));
        unsafe { *p_command_pool = vk::CommandPool::from_raw(id) };
        vk::Result::SUCCESS
    })
}

unsafe extern "system" fn mock_vk_destroy_command_pool(
    device: vk::Device,
    command_pool: vk::CommandPool,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    let handle = unsafe { DispatchableHandle::borrow(device.as_raw(), HandleKind::Device) };
    let st = handle.map(|h| h.state.clone());
    ffi_guard_void(st.as_ref(), || {
        let Some(state) = st.clone() else {
            return;
        };
        let raw = command_pool.as_raw();
        if raw == 0 {
            return;
        }
        let mut inner = state.inner.lock();
        if !inner.command_pools.remove(&raw) {
            inner.push_violation(format!(
                "vkDestroyCommandPool called on unknown handle {raw:#x}"
            ));
        } else {
            inner.record("vkDestroyCommandPool", 0, format!("id={raw:#x}"));
        }
    });
}

unsafe extern "system" fn mock_vk_allocate_command_buffers(
    device: vk::Device,
    p_allocate_info: *const vk::CommandBufferAllocateInfo<'_>,
    p_command_buffers: *mut vk::CommandBuffer,
) -> vk::Result {
    let handle = unsafe { DispatchableHandle::borrow(device.as_raw(), HandleKind::Device) };
    let st = handle.map(|h| h.state.clone());
    ffi_guard_result(st.as_ref(), || {
        let Some(state) = st.clone() else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        if p_allocate_info.is_null() || p_command_buffers.is_null() {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }
        let info = unsafe { &*p_allocate_info };
        let count = info.command_buffer_count as usize;
        for idx in 0..count {
            let raw = DispatchableHandle::alloc(HandleKind::CommandBuffer, state.clone());
            state.inner.lock().command_buffers.insert(raw);
            unsafe { *p_command_buffers.add(idx) = vk::CommandBuffer::from_raw(raw) };
        }
        state.inner.lock().record(
            "vkAllocateCommandBuffers",
            0,
            format!("allocated {count} cmdbuf(s)"),
        );
        vk::Result::SUCCESS
    })
}

unsafe extern "system" fn mock_vk_free_command_buffers(
    device: vk::Device,
    _command_pool: vk::CommandPool,
    command_buffer_count: u32,
    p_command_buffers: *const vk::CommandBuffer,
) {
    let handle = unsafe { DispatchableHandle::borrow(device.as_raw(), HandleKind::Device) };
    let st = handle.map(|h| h.state.clone());
    ffi_guard_void(st.as_ref(), || {
        let Some(state) = st.clone() else {
            return;
        };
        if p_command_buffers.is_null() {
            return;
        }
        for idx in 0..command_buffer_count as usize {
            let cb = unsafe { *p_command_buffers.add(idx) };
            let raw = cb.as_raw();
            if raw != 0 {
                let removed = state.inner.lock().command_buffers.remove(&raw);
                if removed {
                    let _ = unsafe { DispatchableHandle::reclaim(raw, HandleKind::CommandBuffer) };
                } else {
                    state.inner.lock().push_violation(format!(
                        "vkFreeCommandBuffers called on unknown handle {raw:#x}"
                    ));
                }
            }
        }
        state.inner.lock().record(
            "vkFreeCommandBuffers",
            0,
            format!("freed {command_buffer_count} cmdbuf(s)"),
        );
    });
}

unsafe extern "system" fn mock_vk_begin_command_buffer(
    command_buffer: vk::CommandBuffer,
    _p_begin_info: *const vk::CommandBufferBeginInfo<'_>,
) -> vk::Result {
    let handle =
        unsafe { DispatchableHandle::borrow(command_buffer.as_raw(), HandleKind::CommandBuffer) };
    let st = handle.map(|h| h.state.clone());
    ffi_guard_result(st.as_ref(), || {
        if let Some(state) = st.as_ref() {
            state.inner.lock().record("vkBeginCommandBuffer", 0, "ok");
        }
        vk::Result::SUCCESS
    })
}

unsafe extern "system" fn mock_vk_end_command_buffer(
    command_buffer: vk::CommandBuffer,
) -> vk::Result {
    let handle =
        unsafe { DispatchableHandle::borrow(command_buffer.as_raw(), HandleKind::CommandBuffer) };
    let st = handle.map(|h| h.state.clone());
    ffi_guard_result(st.as_ref(), || {
        if let Some(state) = st.as_ref() {
            state.inner.lock().record("vkEndCommandBuffer", 0, "ok");
        }
        vk::Result::SUCCESS
    })
}

fn apply_image_transition(
    inner: &mut IcdInner,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    fn_name: &str,
) {
    let raw = image.as_raw();
    let Some(tracked) = inner.images.get_mut(&raw) else {
        inner.push_violation(format!(
            "{fn_name}: layout transition on unknown VkImage {raw:#x}"
        ));
        return;
    };
    if old_layout != vk::ImageLayout::UNDEFINED && tracked.layout != old_layout {
        let expected = tracked.layout;
        inner.push_violation(format!(
            "{fn_name}: VkImage {raw:#x} expected oldLayout={expected:?}, got {old_layout:?}"
        ));
    }
    tracked.layout = new_layout;
}

unsafe extern "system" fn mock_vk_cmd_pipeline_barrier(
    command_buffer: vk::CommandBuffer,
    _src_stage_mask: vk::PipelineStageFlags,
    _dst_stage_mask: vk::PipelineStageFlags,
    _dependency_flags: vk::DependencyFlags,
    _memory_barrier_count: u32,
    _p_memory_barriers: *const vk::MemoryBarrier<'_>,
    _buffer_memory_barrier_count: u32,
    _p_buffer_memory_barriers: *const vk::BufferMemoryBarrier<'_>,
    image_memory_barrier_count: u32,
    p_image_memory_barriers: *const vk::ImageMemoryBarrier<'_>,
) {
    let handle =
        unsafe { DispatchableHandle::borrow(command_buffer.as_raw(), HandleKind::CommandBuffer) };
    let st = handle.map(|h| h.state.clone());
    ffi_guard_void(st.as_ref(), || {
        let Some(state) = st.as_ref() else {
            return;
        };
        let mut inner = state.inner.lock();
        if !p_image_memory_barriers.is_null() {
            for idx in 0..image_memory_barrier_count as usize {
                let b = unsafe { &*p_image_memory_barriers.add(idx) };
                apply_image_transition(
                    &mut inner,
                    b.image,
                    b.old_layout,
                    b.new_layout,
                    "vkCmdPipelineBarrier",
                );
            }
        }
        inner.record(
            "vkCmdPipelineBarrier",
            0,
            format!("{image_memory_barrier_count} image barrier(s)"),
        );
    });
}

unsafe extern "system" fn mock_vk_cmd_pipeline_barrier2(
    command_buffer: vk::CommandBuffer,
    p_dependency_info: *const vk::DependencyInfo<'_>,
) {
    let handle =
        unsafe { DispatchableHandle::borrow(command_buffer.as_raw(), HandleKind::CommandBuffer) };
    let st = handle.map(|h| h.state.clone());
    ffi_guard_void(st.as_ref(), || {
        let (Some(state), false) = (st.as_ref(), p_dependency_info.is_null()) else {
            return;
        };
        let info = unsafe { &*p_dependency_info };
        let mut inner = state.inner.lock();
        if !info.p_image_memory_barriers.is_null() {
            for idx in 0..info.image_memory_barrier_count as usize {
                let b = unsafe { &*info.p_image_memory_barriers.add(idx) };
                apply_image_transition(
                    &mut inner,
                    b.image,
                    b.old_layout,
                    b.new_layout,
                    "vkCmdPipelineBarrier2",
                );
            }
        }
        inner.record(
            "vkCmdPipelineBarrier2",
            0,
            format!("{} image barrier(s)", info.image_memory_barrier_count),
        );
    });
}

/// Helper to reclaim a physical device handle allocated during `vkEnumeratePhysicalDevices`.
///
/// # Safety
/// `physical_device` must be a live handle returned by this mock ICD and not used afterward.
pub unsafe fn reclaim_mock_physical_device(physical_device: vk::PhysicalDevice) {
    let _ = unsafe {
        DispatchableHandle::reclaim(physical_device.as_raw(), HandleKind::PhysicalDevice)
    };
}

/// Helper to reclaim a queue handle allocated during `vkGetDeviceQueue`.
///
/// # Safety
/// `queue` must be a live handle returned by this mock ICD and not used afterward.
pub unsafe fn reclaim_mock_queue(queue: vk::Queue) {
    let _ = unsafe { DispatchableHandle::reclaim(queue.as_raw(), HandleKind::Queue) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiles::{DriverModel, GpuId};

    #[test]
    fn mock_icd_tracks_layouts_and_detects_bad_old_layout() {
        let profile = GpuProfile::build(GpuId::Rx9070Xt, DriverModel::Linux);
        let (guard, entry) = MockIcdGuard::install(profile);

        unsafe {
            let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
            let inst_ci = vk::InstanceCreateInfo::default().application_info(&app);
            let instance = entry.create_instance(&inst_ci, None).unwrap();
            let phys = entry_enumerate_first(&instance);
            let prio = [1.0f32];
            let qci = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(0)
                .queue_priorities(&prio)];
            let dev_ci = vk::DeviceCreateInfo::default().queue_create_infos(&qci);
            let device = instance.create_device(phys, &dev_ci, None).unwrap();

            let img_ci = vk::ImageCreateInfo::default()
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
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let img = device.create_image(&img_ci, None).unwrap();

            let pool = device
                .create_command_pool(&vk::CommandPoolCreateInfo::default(), None)
                .unwrap();
            let cbs = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .unwrap();
            let cb = cbs[0];

            let b1 = [vk::ImageMemoryBarrier2::default()
                .image(img)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&b1));
            assert_eq!(
                guard.state().query_image_layout(img),
                Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            );

            // Intentional mismatched old_layout (GENERAL instead of SHADER_READ_ONLY_OPTIMAL)
            let b_bad = [vk::ImageMemoryBarrier2::default()
                .image(img)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)];
            device.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&b_bad),
            );

            device.free_command_buffers(pool, &cbs);
            device.destroy_command_pool(pool, None);
            device.destroy_image(img, None);
            device.destroy_device(None);
            reclaim_mock_physical_device(phys);
            instance.destroy_instance(None);
        }

        let report = guard.state().leak_report();
        assert_eq!(report.leaked_images, 0);
        assert_eq!(report.violations.len(), 1);
        assert!(report.violations[0].contains("expected oldLayout=SHADER_READ_ONLY_OPTIMAL"));
    }

    unsafe fn entry_enumerate_first(instance: &ash::Instance) -> vk::PhysicalDevice {
        let devs = unsafe { instance.enumerate_physical_devices().unwrap() };
        devs[0]
    }
}
