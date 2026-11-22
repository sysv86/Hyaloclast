use crate::gpu::{
    VulkanContext,
    buffer::{GpuBufferAllocation, SlabArenaAllocator, SlabSubAllocation},
    retire::RetireQueue,
};
use ash::vk;
use std::ffi::CStr;

/// Maximum supported Opacity Micromap subdivision level (`4^5 = 1024` microtriangles per triangle).
pub const MAX_OMM_SUBDIV_LEVEL: u32 = 5;

/// Default Opacity Micromap storage slab size (16 MiB device-local slabs).
pub const OMM_STORAGE_SLAB_BYTES: vk::DeviceSize = 16 * 1024 * 1024;

/// Default Opacity Micromap scratch slab size (8 MiB device-local slabs).
pub const OMM_SCRATCH_SLAB_BYTES: vk::DeviceSize = 8 * 1024 * 1024;

/// Required alignment for `VkMicromapEXT` storage sub-allocations (`256` bytes per Vulkan spec).
pub const OMM_STORAGE_ALIGNMENT: vk::DeviceSize = 256;

/// Bit flag set in `InstanceMetaGpu::material_flags` when a cutout terrain slice has an active
/// hardware `VkMicromapEXT` bound to its BLAS geometry.
pub const INSTANCE_FLAG_HAS_OMM: u32 = 1 << 9;

/// Per-microtriangle opacity state for `VK_EXT_opacity_micromap`.
///
/// Matches Vulkan's 2-state (1 bit/microtriangle) and 4-state (2 bits/microtriangle) encodings:
/// - `0` (`0b00`): `Transparent` — ray misses without invoking any-hit.
/// - `1` (`0b01`): `Opaque` — ray hits without invoking any-hit.
/// - `2` (`0b10`): `UnknownTransparent` (4-state only) — invokes any-hit (or treated as transparent if forced).
/// - `3` (`0b11`): `UnknownOpaque` (4-state only) — invokes any-hit (or treated as opaque if forced).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OmmState {
    Transparent = 0,
    Opaque = 1,
    UnknownTransparent = 2,
    UnknownOpaque = 3,
}

impl OmmState {
    /// Converts a state to its 2-state 1-bit representation (`0` = transparent, `1` = opaque).
    /// Unknown states conservatively map to `Opaque` or `Transparent` based on their bias.
    pub const fn to_2state_bit(self) -> u8 {
        match self {
            Self::Transparent | Self::UnknownTransparent => 0,
            Self::Opaque | Self::UnknownOpaque => 1,
        }
    }

    /// Converts a state to its 4-state 2-bit representation (`0b00..=0b11`).
    pub const fn to_4state_bits(self) -> u8 {
        self as u8
    }

    /// Returns the corresponding Vulkan special index when an entire triangle is uniformly this state.
    pub const fn to_special_index(self) -> vk::OpacityMicromapSpecialIndexEXT {
        match self {
            Self::Transparent => vk::OpacityMicromapSpecialIndexEXT::FULLY_TRANSPARENT,
            Self::Opaque => vk::OpacityMicromapSpecialIndexEXT::FULLY_OPAQUE,
            Self::UnknownTransparent => {
                vk::OpacityMicromapSpecialIndexEXT::FULLY_UNKNOWN_TRANSPARENT
            }
            Self::UnknownOpaque => vk::OpacityMicromapSpecialIndexEXT::FULLY_UNKNOWN_OPAQUE,
        }
    }
}

/// Opacity Micromap encoding format (`2-state` or `4-state`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OmmFormat {
    /// 1 bit per microtriangle (`Transparent` / `Opaque`).
    TwoState,
    /// 2 bits per microtriangle (`Transparent` / `Opaque` / `UnknownTransparent` / `UnknownOpaque`).
    FourState,
}

impl OmmFormat {
    pub const fn to_vk_format(self) -> vk::OpacityMicromapFormatEXT {
        match self {
            Self::TwoState => vk::OpacityMicromapFormatEXT::TYPE_2_STATE,
            Self::FourState => vk::OpacityMicromapFormatEXT::TYPE_4_STATE,
        }
    }

    pub const fn to_vk_raw_u16(self) -> u16 {
        self.to_vk_format().as_raw() as u16
    }

    pub const fn to_vk_raw_u32(self) -> u32 {
        self.to_vk_format().as_raw() as u32
    }

    pub const fn bits_per_microtriangle(self) -> usize {
        match self {
            Self::TwoState => 1,
            Self::FourState => 2,
        }
    }
}

/// Returns the number of microtriangles (`4^subdiv_level`) for subdivision level `0..=5`.
pub const fn microtriangle_count(subdiv_level: u32) -> usize {
    let clamped = if subdiv_level > MAX_OMM_SUBDIV_LEVEL {
        MAX_OMM_SUBDIV_LEVEL
    } else {
        subdiv_level
    };
    1usize << (2 * clamped)
}

/// Returns the packed byte length for one triangle's opacity micromap at `subdiv_level` and `format`.
pub const fn encoded_byte_size(subdiv_level: u32, format: OmmFormat) -> usize {
    let total_bits = microtriangle_count(subdiv_level) * format.bits_per_microtriangle();
    total_bits.div_ceil(8)
}

/// CPU-side view of an RGBA8 block texture atlas used to analyze cutout alpha masks.
#[derive(Debug, Clone, Copy)]
pub struct AtlasAlphaMask<'a> {
    /// Tightly packed RGBA8 texels (`width * height * 4` bytes).
    pub rgba: &'a [u8],
    pub width: u32,
    pub height: u32,
    /// Alpha cutoff threshold (Minecraft cutout default is `128`, i.e. `0.5`).
    pub alpha_cutoff: u8,
}

impl<'a> AtlasAlphaMask<'a> {
    pub const fn new(rgba: &'a [u8], width: u32, height: u32, alpha_cutoff: u8) -> Self {
        Self {
            rgba,
            width,
            height,
            alpha_cutoff,
        }
    }

    /// Samples the alpha channel at normalized UV coordinates with wrap-clamped texel indexing.
    #[inline]
    pub fn sample_alpha(&self, u: f32, v: f32) -> u8 {
        if self.width == 0 || self.height == 0 {
            return 255;
        }
        let u_clamped = u.clamp(0.0, 0.999_999);
        let v_clamped = v.clamp(0.0, 0.999_999);
        let x = (u_clamped * self.width as f32) as usize;
        let y = (v_clamped * self.height as f32) as usize;
        let idx = (y * self.width as usize + x) * 4 + 3;
        self.rgba.get(idx).copied().unwrap_or(255)
    }
}

/// Encoded opacity result for a single triangle after alpha-mask analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OmmTriangleEncoding {
    /// Entire triangle resolved to a uniform hardware special index (no micromap array entry needed).
    Uniform(vk::OpacityMicromapSpecialIndexEXT),
    /// Mixed triangle requiring a micromap entry with packed bit data.
    Micromap {
        subdiv_level: u16,
        format: OmmFormat,
        data: Vec<u8>,
    },
}

/// Encodes `states` into Vulkan 2-state (1 bit/microtriangle, LSB-first) packed bytes for `subdiv_level` (`0..=5`).
pub fn encode_triangle_omm_2state(states: &[OmmState], subdiv_level: u32) -> Vec<u8> {
    let clamped_level = subdiv_level.min(MAX_OMM_SUBDIV_LEVEL);
    let count = microtriangle_count(clamped_level);
    let byte_len = encoded_byte_size(clamped_level, OmmFormat::TwoState);
    let mut out = vec![0u8; byte_len];

    for (i, state) in states.iter().copied().take(count).enumerate() {
        let bit = state.to_2state_bit();
        out[i >> 3] |= (bit & 1) << (i & 7);
    }
    out
}

/// Encodes `states` into Vulkan 4-state (2 bits/microtriangle, LSB-first) packed bytes for `subdiv_level` (`0..=5`).
pub fn encode_triangle_omm_4state(states: &[OmmState], subdiv_level: u32) -> Vec<u8> {
    let clamped_level = subdiv_level.min(MAX_OMM_SUBDIV_LEVEL);
    let count = microtriangle_count(clamped_level);
    let byte_len = encoded_byte_size(clamped_level, OmmFormat::FourState);
    let mut out = vec![0u8; byte_len];

    for (i, state) in states.iter().copied().take(count).enumerate() {
        let bits = state.to_4state_bits() & 0b11;
        let bit_pos = i * 2;
        out[bit_pos >> 3] |= bits << (bit_pos & 7);
    }
    out
}

/// Checks whether all microtriangle states in a triangle collapse to a single Vulkan special index.
pub fn classify_uniform_states(
    states: &[OmmState],
    format: OmmFormat,
) -> Option<vk::OpacityMicromapSpecialIndexEXT> {
    let first = *states.first()?;
    match format {
        OmmFormat::TwoState => {
            let first_bit = first.to_2state_bit();
            if states.iter().all(|s| s.to_2state_bit() == first_bit) {
                Some(if first_bit == 1 {
                    vk::OpacityMicromapSpecialIndexEXT::FULLY_OPAQUE
                } else {
                    vk::OpacityMicromapSpecialIndexEXT::FULLY_TRANSPARENT
                })
            } else {
                None
            }
        }
        OmmFormat::FourState => {
            if states.iter().all(|&s| s == first) {
                Some(first.to_special_index())
            } else {
                None
            }
        }
    }
}

/// Computes the 3 barycentric `(u, v)` coordinates of microtriangle `micro_idx` at `subdiv_level`
/// following the Vulkan `VK_EXT_opacity_micromap` space-filling subdivision curve (bird curve).
pub fn microtriangle_barycentrics(subdiv_level: u32, micro_idx: usize) -> [[f32; 2]; 3] {
    let level = subdiv_level.min(MAX_OMM_SUBDIV_LEVEL);
    // Start with the root triangle vertices in (u, v) barycentric space:
    // v0 = (0, 0), v1 = (1, 0), v2 = (0, 1).
    let mut a = [0.0f32, 0.0f32];
    let mut b = [1.0f32, 0.0f32];
    let mut c = [0.0f32, 1.0f32];

    for l in (0..level).rev() {
        let sub = (micro_idx >> (2 * l)) & 3;
        let ab = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
        let bc = [(b[0] + c[0]) * 0.5, (b[1] + c[1]) * 0.5];
        let ca = [(c[0] + a[0]) * 0.5, (c[1] + a[1]) * 0.5];

        match sub {
            0 => {
                // Corner 0 subtriangle
                b = ab;
                c = ca;
            }
            1 => {
                // Center inverted subtriangle (orientation preserved along bird curve)
                a = bc;
                let prev_ca = ca;
                b = prev_ca;
                c = ab;
            }
            2 => {
                // Corner 1 subtriangle
                a = ab;
                c = bc;
            }
            _ => {
                // Corner 2 subtriangle
                a = ca;
                b = bc;
            }
        }
    }

    [a, b, c]
}

/// Analyzes a triangle's UV footprint against `mask` and encodes its per-microtriangle opacity
/// states at `subdiv_level` (`0..=5`) in either `TwoState` or `FourState` format.
///
/// Uniformly opaque or transparent triangles collapse to [`OmmTriangleEncoding::Uniform`]
/// so they consume zero micromap array bytes.
pub fn analyze_atlas_alpha_mask(
    mask: &AtlasAlphaMask<'_>,
    uv0: [f32; 2],
    uv1: [f32; 2],
    uv2: [f32; 2],
    subdiv_level: u32,
    format: OmmFormat,
) -> OmmTriangleEncoding {
    let clamped_level = subdiv_level.min(MAX_OMM_SUBDIV_LEVEL);
    let count = microtriangle_count(clamped_level);
    let mut states = Vec::with_capacity(count);

    let eval_uv = |bary: [f32; 2]| -> [f32; 2] {
        let w0 = 1.0 - bary[0] - bary[1];
        [
            uv0[0] * w0 + uv1[0] * bary[0] + uv2[0] * bary[1],
            uv0[1] * w0 + uv1[1] * bary[0] + uv2[1] * bary[1],
        ]
    };

    for m in 0..count {
        let [b0, b1, b2] = microtriangle_barycentrics(clamped_level, m);
        let bc = [(b0[0] + b1[0] + b2[0]) / 3.0, (b0[1] + b1[1] + b2[1]) / 3.0];
        // Sample centroid + 3 inset corner probes to detect alpha boundaries accurately.
        let probes = [
            bc,
            [b0[0] * 0.75 + bc[0] * 0.25, b0[1] * 0.75 + bc[1] * 0.25],
            [b1[0] * 0.75 + bc[0] * 0.25, b1[1] * 0.75 + bc[1] * 0.25],
            [b2[0] * 0.75 + bc[0] * 0.25, b2[1] * 0.75 + bc[1] * 0.25],
        ];

        let mut opaque_hits = 0u32;
        for probe in probes {
            let uv = eval_uv(probe);
            let alpha = mask.sample_alpha(uv[0], uv[1]);
            if alpha >= mask.alpha_cutoff {
                opaque_hits += 1;
            }
        }

        let state = match (format, opaque_hits) {
            (_, 4) => OmmState::Opaque,
            (_, 0) => OmmState::Transparent,
            (OmmFormat::FourState, hits) if hits >= 2 => OmmState::UnknownOpaque,
            (OmmFormat::FourState, _) => OmmState::UnknownTransparent,
            // In 2-state mode, conservatively preserve silhouette coverage on mixed microtris.
            (OmmFormat::TwoState, _) => OmmState::Opaque,
        };
        states.push(state);
    }

    if let Some(special) = classify_uniform_states(&states, format) {
        return OmmTriangleEncoding::Uniform(special);
    }

    let data = match format {
        OmmFormat::TwoState => encode_triangle_omm_2state(&states, clamped_level),
        OmmFormat::FourState => encode_triangle_omm_4state(&states, clamped_level),
    };
    OmmTriangleEncoding::Micromap {
        subdiv_level: clamped_level as u16,
        format,
        data,
    }
}

/// Owned per-slice cutout OMM build input prepared on the CPU and consumed during BLAS build.
#[derive(Debug, Clone)]
pub struct CutoutOmmBuildInput {
    pub triangles: Vec<OmmTriangleEncoding>,
}

impl CutoutOmmBuildInput {
    /// Analyzes a list of triangle UV triplets against an atlas alpha mask.
    pub fn from_triangle_uvs(
        mask: &AtlasAlphaMask<'_>,
        triangle_uvs: &[([f32; 2], [f32; 2], [f32; 2])],
        subdiv_level: u32,
        format: OmmFormat,
    ) -> Self {
        let triangles = triangle_uvs
            .iter()
            .map(|&(uv0, uv1, uv2)| {
                analyze_atlas_alpha_mask(mask, uv0, uv1, uv2, subdiv_level, format)
            })
            .collect();
        Self { triangles }
    }
}

/// Dynamically loaded `VK_EXT_opacity_micromap` device dispatch table.
///
/// Uses `vkGetDeviceProcAddr` directly so it compiles cleanly across all `ash 0.38` builds
/// without depending on optional handwritten extension wrappers.
#[derive(Clone)]
pub struct OpacityMicromapDevice {
    device: vk::Device,
    fp_create_micromap: vk::PFN_vkCreateMicromapEXT,
    fp_destroy_micromap: vk::PFN_vkDestroyMicromapEXT,
    fp_cmd_build_micromaps: vk::PFN_vkCmdBuildMicromapsEXT,
    fp_get_micromap_build_sizes: vk::PFN_vkGetMicromapBuildSizesEXT,
}

unsafe impl Send for OpacityMicromapDevice {}
unsafe impl Sync for OpacityMicromapDevice {}

impl OpacityMicromapDevice {
    /// Loads `VK_EXT_opacity_micromap` device entrypoints from `instance` and `device`.
    pub fn load(instance: &ash::Instance, device: &ash::Device) -> Option<Self> {
        let load_fn = |name: &'static CStr| -> vk::PFN_vkVoidFunction {
            unsafe { instance.get_device_proc_addr(device.handle(), name.as_ptr()) }
        };
        let p_create = load_fn(c"vkCreateMicromapEXT")?;
        let p_destroy = load_fn(c"vkDestroyMicromapEXT")?;
        let p_build = load_fn(c"vkCmdBuildMicromapsEXT")?;
        let p_sizes = load_fn(c"vkGetMicromapBuildSizesEXT")?;
        unsafe {
            Some(Self {
                device: device.handle(),
                fp_create_micromap: std::mem::transmute::<
                    unsafe extern "system" fn(),
                    vk::PFN_vkCreateMicromapEXT,
                >(p_create),
                fp_destroy_micromap: std::mem::transmute::<
                    unsafe extern "system" fn(),
                    vk::PFN_vkDestroyMicromapEXT,
                >(p_destroy),
                fp_cmd_build_micromaps: std::mem::transmute::<
                    unsafe extern "system" fn(),
                    vk::PFN_vkCmdBuildMicromapsEXT,
                >(p_build),
                fp_get_micromap_build_sizes: std::mem::transmute::<
                    unsafe extern "system" fn(),
                    vk::PFN_vkGetMicromapBuildSizesEXT,
                >(p_sizes),
            })
        }
    }

    /// Queries driver storage and scratch byte sizes via `vkGetMicromapBuildSizesEXT`.
    ///
    /// # Safety
    /// `build_info` pointers must remain valid for the duration of this call.
    pub unsafe fn get_micromap_build_sizes(
        &self,
        build_type: vk::AccelerationStructureBuildTypeKHR,
        build_info: &vk::MicromapBuildInfoEXT<'_>,
        size_info: &mut vk::MicromapBuildSizesInfoEXT<'_>,
    ) {
        unsafe {
            (self.fp_get_micromap_build_sizes)(self.device, build_type, build_info, size_info);
        }
    }

    /// Creates a `VkMicromapEXT` handle backed by a sub-allocated storage buffer region.
    ///
    /// # Safety
    /// `create_info.buffer` must be created with `VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT`.
    pub unsafe fn create_micromap(
        &self,
        create_info: &vk::MicromapCreateInfoEXT<'_>,
    ) -> Result<vk::MicromapEXT, vk::Result> {
        let mut handle = vk::MicromapEXT::null();
        let res = unsafe {
            (self.fp_create_micromap)(self.device, create_info, std::ptr::null(), &mut handle)
        };
        if res == vk::Result::SUCCESS {
            Ok(handle)
        } else {
            Err(res)
        }
    }

    /// Destroys a `VkMicromapEXT` handle.
    ///
    /// # Safety
    /// No in-flight GPU work may reference `micromap`.
    pub unsafe fn destroy_micromap(&self, micromap: vk::MicromapEXT) {
        if micromap != vk::MicromapEXT::null() {
            unsafe {
                (self.fp_destroy_micromap)(self.device, micromap, std::ptr::null());
            }
        }
    }

    /// Records `vkCmdBuildMicromapsEXT` onto `cmd`.
    ///
    /// # Safety
    /// `cmd` must be in the recording state and all device addresses in `infos` must be valid.
    pub unsafe fn cmd_build_micromaps(
        &self,
        cmd: vk::CommandBuffer,
        infos: &[vk::MicromapBuildInfoEXT<'_>],
    ) {
        if infos.is_empty() {
            return;
        }
        unsafe {
            (self.fp_cmd_build_micromaps)(cmd, infos.len() as u32, infos.as_ptr());
        }
    }
}

/// Owned GPU Opacity Micromap (`VkMicromapEXT`) plus its slab storage slice and per-triangle
/// index buffer, ready to attach via `p_next` to a cutout BLAS triangle geometry.
#[derive(Debug)]
pub struct OpacityMicromapAllocation {
    pub handle: vk::MicromapEXT,
    pub storage: Option<SlabSubAllocation>,
    pub index_buffer: GpuBufferAllocation,
    pub index_type: vk::IndexType,
    pub index_stride: vk::DeviceSize,
    pub usage_counts: Vec<vk::MicromapUsageEXT>,
    pub triangle_count: u32,
}

impl OpacityMicromapAllocation {
    /// Builds a `VkAccelerationStructureTrianglesOpacityMicromapEXT` pointing to this allocation's
    /// micromap handle, per-triangle index buffer, and usage counts.
    pub fn blas_attachment(&self) -> vk::AccelerationStructureTrianglesOpacityMicromapEXT<'_> {
        vk::AccelerationStructureTrianglesOpacityMicromapEXT::default()
            .index_type(self.index_type)
            .index_buffer(vk::DeviceOrHostAddressConstKHR {
                device_address: self.index_buffer.device_address,
            })
            .index_stride(self.index_stride)
            .base_triangle(0)
            .usage_counts(&self.usage_counts)
            .micromap(self.handle)
    }

    /// Retires the `VkMicromapEXT` handle, its `SlabSubAllocation` storage slice, and its
    /// per-triangle index buffer through `RetireQueue`.
    pub fn retire(
        mut self,
        ctx: &VulkanContext,
        omm_storage_slab: &SlabArenaAllocator,
        omm_device: Option<&OpacityMicromapDevice>,
        retire_queue: &RetireQueue,
    ) {
        let dev = ctx.device.clone();
        let pool = omm_storage_slab.clone();
        let omm_dev = omm_device.cloned();
        let handle = self.handle;
        self.handle = vk::MicromapEXT::null();
        let storage = self.storage.take();
        let mut index_buf = self.index_buffer;

        retire_queue.retire_later(move || unsafe {
            if let Some(ref ext) = omm_dev
                && handle != vk::MicromapEXT::null()
            {
                ext.destroy_micromap(handle);
            }
            if let Some(sub) = storage {
                pool.free_slice(&dev, sub);
            }
            index_buf.destroy(&dev);
        });
    }
}

/// Manages `VK_EXT_opacity_micromap` device dispatch, 16 MiB micromap storage slabs, and
/// scratch slabs for cutout (`layer_ordinal == 1`) terrain BLAS builds.
pub struct OpacityMicromapBuilder {
    pub device: Option<OpacityMicromapDevice>,
    pub storage_slab: SlabArenaAllocator,
    pub scratch_slab: SlabArenaAllocator,
    pub default_format: OmmFormat,
    pub default_subdiv_level: u32,
    pub enabled: bool,
    device_probed: bool,
}

impl Default for OpacityMicromapBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl OpacityMicromapBuilder {
    pub fn new() -> Self {
        Self {
            device: None,
            storage_slab: SlabArenaAllocator::new(
                OMM_STORAGE_SLAB_BYTES,
                vk::BufferUsageFlags::MICROMAP_STORAGE_EXT
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            ),
            scratch_slab: SlabArenaAllocator::new(
                OMM_SCRATCH_SLAB_BYTES,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            ),
            default_format: OmmFormat::FourState,
            default_subdiv_level: 3,
            enabled: true,
            device_probed: false,
        }
    }

    /// Lazily loads the `VK_EXT_opacity_micromap` device dispatch table when hardware support is present.
    pub fn ensure_device(&mut self, ctx: &VulkanContext) -> bool {
        if !self.enabled || !ctx.capabilities.supports_opacity_micromap {
            return false;
        }
        if !self.device_probed {
            self.device_probed = true;
            self.device = OpacityMicromapDevice::load(&ctx.instance, &ctx.device);
        }
        self.device.is_some()
    }

    /// Builds a `VkMicromapEXT` for a cutout terrain section from CPU-analyzed triangle encodings,
    /// allocating micromap storage and scratch via [`SlabArenaAllocator`], recording
    /// `vkCmdBuildMicromapsEXT`, and emitting a `MICROMAP_BUILD_EXT -> ACCELERATION_STRUCTURE_BUILD_KHR`
    /// memory barrier.
    pub fn build_cutout_micromap(
        &mut self,
        ctx: &VulkanContext,
        retire_queue: &RetireQueue,
        cmd: vk::CommandBuffer,
        input: &CutoutOmmBuildInput,
    ) -> Result<OpacityMicromapAllocation, vk::Result> {
        if input.triangles.is_empty() {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        if !self.ensure_device(ctx) {
            return Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT);
        }
        let omm_dev = self.device.as_ref().unwrap().clone();

        // Pack per-triangle indices (i32) and micromap descriptors/data for non-uniform triangles.
        let tri_count = input.triangles.len();
        let mut tri_indices: Vec<i32> = Vec::with_capacity(tri_count);
        let mut micromap_descs: Vec<vk::MicromapTriangleEXT> = Vec::new();
        let mut opacity_data: Vec<u8> = Vec::new();

        // Aggregate usage counts keyed by (subdiv_level, format_raw_u32).
        let mut usage_map: std::collections::BTreeMap<(u32, u32), u32> =
            std::collections::BTreeMap::new();
        let mut blas_usage_map: std::collections::BTreeMap<(u32, u32), u32> =
            std::collections::BTreeMap::new();

        for enc in &input.triangles {
            match enc {
                OmmTriangleEncoding::Uniform(special) => {
                    tri_indices.push(special.as_raw());
                }
                OmmTriangleEncoding::Micromap {
                    subdiv_level,
                    format,
                    data,
                } => {
                    let entry_idx = micromap_descs.len() as i32;
                    let data_offset = opacity_data.len() as u32;
                    opacity_data.extend_from_slice(data);
                    micromap_descs.push(vk::MicromapTriangleEXT {
                        data_offset,
                        subdivision_level: *subdiv_level,
                        format: format.to_vk_raw_u16(),
                    });
                    tri_indices.push(entry_idx);
                    let key = (u32::from(*subdiv_level), format.to_vk_raw_u32());
                    *usage_map.entry(key).or_insert(0) += 1;
                    *blas_usage_map.entry(key).or_insert(0) += 1;
                }
            }
        }

        let usage_counts: Vec<vk::MicromapUsageEXT> = usage_map
            .into_iter()
            .map(|((subdiv, fmt), count)| {
                vk::MicromapUsageEXT::default()
                    .count(count)
                    .subdivision_level(subdiv)
                    .format(fmt)
            })
            .collect();
        let blas_usage_counts: Vec<vk::MicromapUsageEXT> = blas_usage_map
            .into_iter()
            .map(|((subdiv, fmt), count)| {
                vk::MicromapUsageEXT::default()
                    .count(count)
                    .subdivision_level(subdiv)
                    .format(fmt)
            })
            .collect();

        // Upload persistent per-triangle index buffer (i32) referenced during BLAS build.
        let index_bytes: &[u8] = bytemuck::cast_slice(&tri_indices);
        let mut index_buffer = GpuBufferAllocation::allocate_host_visible_bda(
            ctx,
            index_bytes.len() as vk::DeviceSize,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        index_buffer.write_bytes(0, index_bytes);

        // If every triangle resolved to a uniform special index, no VkMicromapEXT object is needed;
        // Vulkan allows `micromap = VK_NULL_HANDLE` when all indices are special indices.
        if micromap_descs.is_empty() {
            return Ok(OpacityMicromapAllocation {
                handle: vk::MicromapEXT::null(),
                storage: None,
                index_buffer,
                index_type: vk::IndexType::UINT32,
                index_stride: std::mem::size_of::<i32>() as vk::DeviceSize,
                usage_counts: blas_usage_counts,
                triangle_count: tri_count as u32,
            });
        }

        // Upload staging opacity data and VkMicromapTriangleEXT[] descriptors (retired after GPU build).
        let mut data_buf = match GpuBufferAllocation::allocate_host_visible_bda(
            ctx,
            (opacity_data.len().max(64)) as vk::DeviceSize,
            vk::BufferUsageFlags::MICROMAP_BUILD_INPUT_READ_ONLY_EXT
                | vk::BufferUsageFlags::STORAGE_BUFFER,
        ) {
            Ok(b) => b,
            Err(err) => {
                unsafe {
                    index_buffer.destroy(&ctx.device);
                }
                return Err(err);
            }
        };
        data_buf.write_bytes(0, &opacity_data);

        let desc_bytes = unsafe {
            std::slice::from_raw_parts(
                micromap_descs.as_ptr() as *const u8,
                std::mem::size_of_val(micromap_descs.as_slice()),
            )
        };
        let mut tri_desc_buf = match GpuBufferAllocation::allocate_host_visible_bda(
            ctx,
            (desc_bytes.len().max(64)) as vk::DeviceSize,
            vk::BufferUsageFlags::MICROMAP_BUILD_INPUT_READ_ONLY_EXT
                | vk::BufferUsageFlags::STORAGE_BUFFER,
        ) {
            Ok(b) => b,
            Err(err) => {
                unsafe {
                    data_buf.destroy(&ctx.device);
                    index_buffer.destroy(&ctx.device);
                }
                return Err(err);
            }
        };
        tri_desc_buf.write_bytes(0, desc_bytes);

        // Query build sizes via vkGetMicromapBuildSizesEXT.
        let mut build_info =
            vk::MicromapBuildInfoEXT::default()
                .ty(vk::MicromapTypeEXT::OPACITY_MICROMAP)
                .flags(vk::BuildMicromapFlagsEXT::PREFER_FAST_TRACE)
                .mode(vk::BuildMicromapModeEXT::BUILD)
                .usage_counts(&usage_counts)
                .data(vk::DeviceOrHostAddressConstKHR {
                    device_address: data_buf.device_address,
                })
                .triangle_array(vk::DeviceOrHostAddressConstKHR {
                    device_address: tri_desc_buf.device_address,
                })
                .triangle_array_stride(
                    std::mem::size_of::<vk::MicromapTriangleEXT>() as vk::DeviceSize
                );

        let mut size_info = vk::MicromapBuildSizesInfoEXT::default();
        unsafe {
            omm_dev.get_micromap_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &mut size_info,
            );
        }

        let storage_size = size_info.micromap_size.max(OMM_STORAGE_ALIGNMENT);
        let scratch_size = size_info
            .build_scratch_size
            .max(ctx.min_scratch_alignment.max(256));

        let storage =
            match self
                .storage_slab
                .allocate_slice(ctx, storage_size, OMM_STORAGE_ALIGNMENT)
            {
                Ok(s) => s,
                Err(err) => {
                    unsafe {
                        tri_desc_buf.destroy(&ctx.device);
                        data_buf.destroy(&ctx.device);
                        index_buffer.destroy(&ctx.device);
                    }
                    return Err(err);
                }
            };

        let scratch = match self.scratch_slab.allocate_slice(
            ctx,
            scratch_size,
            ctx.min_scratch_alignment.max(256),
        ) {
            Ok(s) => s,
            Err(err) => {
                unsafe {
                    self.storage_slab.free_slice(&ctx.device, storage);
                    tri_desc_buf.destroy(&ctx.device);
                    data_buf.destroy(&ctx.device);
                    index_buffer.destroy(&ctx.device);
                }
                return Err(err);
            }
        };

        let create_info = vk::MicromapCreateInfoEXT::default()
            .create_flags(vk::MicromapCreateFlagsEXT::empty())
            .buffer(storage.buffer)
            .offset(storage.offset)
            .size(storage_size)
            .ty(vk::MicromapTypeEXT::OPACITY_MICROMAP);

        let handle = match unsafe { omm_dev.create_micromap(&create_info) } {
            Ok(h) => h,
            Err(err) => {
                unsafe {
                    self.scratch_slab.free_slice(&ctx.device, scratch);
                    self.storage_slab.free_slice(&ctx.device, storage);
                    tri_desc_buf.destroy(&ctx.device);
                    data_buf.destroy(&ctx.device);
                    index_buffer.destroy(&ctx.device);
                }
                return Err(err);
            }
        };

        build_info = build_info
            .dst_micromap(handle)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch.device_address,
            });

        unsafe {
            omm_dev.cmd_build_micromaps(cmd, &[build_info]);
            ctx.cmd_memory_barrier2(
                cmd,
                vk::PipelineStageFlags2::MICROMAP_BUILD_EXT,
                vk::AccessFlags2::MICROMAP_WRITE_EXT,
                vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::AccessFlags2::MICROMAP_READ_EXT
                    | vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
            );
        }

        // Retire transient build staging buffers and scratch slab slice once GPU work finishes.
        let dev = ctx.device.clone();
        let scratch_pool = self.scratch_slab.clone();
        retire_queue.retire_later(move || unsafe {
            scratch_pool.free_slice(&dev, scratch);
            tri_desc_buf.destroy(&dev);
            data_buf.destroy(&dev);
        });

        Ok(OpacityMicromapAllocation {
            handle,
            storage: Some(storage),
            index_buffer,
            index_type: vk::IndexType::UINT32,
            index_stride: std::mem::size_of::<i32>() as vk::DeviceSize,
            usage_counts: blas_usage_counts,
            triangle_count: tri_count as u32,
        })
    }

    /// Destroys all underlying OMM storage and scratch slabs on shutdown.
    pub fn destroy(&mut self, ctx: &VulkanContext, retire_queue: &RetireQueue) {
        let dev = ctx.device.clone();
        let storage_pool = self.storage_slab.clone();
        let scratch_pool = self.scratch_slab.clone();
        self.device = None;
        self.device_probed = false;
        retire_queue.retire_later(move || unsafe {
            storage_pool.destroy_all(&dev);
            scratch_pool.destroy_all(&dev);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subdiv_counts_and_byte_sizes_match_vulkan_spec() {
        let expected_counts = [1, 4, 16, 64, 256, 1024];
        let expected_2state_bytes = [1, 1, 2, 8, 32, 128];
        let expected_4state_bytes = [1, 1, 4, 16, 64, 256];

        for level in 0..=MAX_OMM_SUBDIV_LEVEL {
            let idx = level as usize;
            assert_eq!(microtriangle_count(level), expected_counts[idx]);
            assert_eq!(
                encoded_byte_size(level, OmmFormat::TwoState),
                expected_2state_bytes[idx]
            );
            assert_eq!(
                encoded_byte_size(level, OmmFormat::FourState),
                expected_4state_bytes[idx]
            );
        }
        // Clamped above level 5
        assert_eq!(microtriangle_count(8), 1024);
    }

    #[test]
    fn encodes_2state_and_4state_bits_lsb_first() {
        let states_lvl1 = [
            OmmState::Transparent,        // 0 / 00
            OmmState::Opaque,             // 1 / 01
            OmmState::UnknownTransparent, // 0 / 10
            OmmState::UnknownOpaque,      // 1 / 11
        ];

        let packed_2 = encode_triangle_omm_2state(&states_lvl1, 1);
        assert_eq!(packed_2.len(), 1);
        // Bits: [0]=0, [1]=1, [2]=0, [3]=1 => 0b0000_1010 = 0x0A
        assert_eq!(packed_2[0], 0b0000_1010);

        let packed_4 = encode_triangle_omm_4state(&states_lvl1, 1);
        assert_eq!(packed_4.len(), 1);
        // 2-bit pairs: [0..1]=00, [2..3]=01, [4..5]=10, [6..7]=11 => 0b11_10_01_00 = 0xE4
        assert_eq!(packed_4[0], 0b11_10_01_00);
    }

    #[test]
    fn analyzes_opaque_transparent_and_mixed_alpha_masks() {
        // 4x4 RGBA8 atlas: left half (x < 2) opaque, right half (x >= 2) transparent.
        let mut rgba = vec![0u8; 4 * 4 * 4];
        for y in 0..4 {
            for x in 0..4 {
                let off = (y * 4 + x) * 4;
                rgba[off] = 255;
                rgba[off + 1] = 255;
                rgba[off + 2] = 255;
                rgba[off + 3] = if x < 2 { 255 } else { 0 };
            }
        }
        let mask = AtlasAlphaMask::new(&rgba, 4, 4, 128);

        // Fully opaque triangle inside left half (u in [0.0, 0.45])
        let opaque_enc = analyze_atlas_alpha_mask(
            &mask,
            [0.05, 0.05],
            [0.45, 0.05],
            [0.05, 0.95],
            2,
            OmmFormat::FourState,
        );
        assert_eq!(
            opaque_enc,
            OmmTriangleEncoding::Uniform(vk::OpacityMicromapSpecialIndexEXT::FULLY_OPAQUE)
        );

        // Fully transparent triangle inside right half (u in [0.55, 0.95])
        let trans_enc = analyze_atlas_alpha_mask(
            &mask,
            [0.55, 0.05],
            [0.95, 0.05],
            [0.55, 0.95],
            2,
            OmmFormat::TwoState,
        );
        assert_eq!(
            trans_enc,
            OmmTriangleEncoding::Uniform(vk::OpacityMicromapSpecialIndexEXT::FULLY_TRANSPARENT)
        );

        // Mixed triangle spanning u in [0.0, 1.0] -> produces Micromap data
        let mixed_4 = analyze_atlas_alpha_mask(
            &mask,
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            2,
            OmmFormat::FourState,
        );
        match mixed_4 {
            OmmTriangleEncoding::Micromap {
                subdiv_level,
                format,
                data,
            } => {
                assert_eq!(subdiv_level, 2);
                assert_eq!(format, OmmFormat::FourState);
                assert_eq!(data.len(), encoded_byte_size(2, OmmFormat::FourState));
                assert!(data.iter().any(|&b| b != 0));
            }
            OmmTriangleEncoding::Uniform(_) => panic!("expected mixed micromap encoding"),
        }
    }
}
