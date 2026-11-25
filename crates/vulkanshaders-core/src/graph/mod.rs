use crate::{
    gpu::{
        VulkanContext,
        bindless::BindlessTextureTable,
        buffer::{GpuBufferAllocation, PooledImageAllocation},
        retire::RetireQueue,
    },
    modules::GpuPipelineManager,
    rt::ZeroCopyBlasManager,
    sdk::{RuntimeSdkManager, SdkDispatchParams},
};
use ash::vk::{self, Handle};
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;
use std::collections::{HashMap, HashSet, VecDeque};
use vsp_compiler::{
    PackManifest, PackPassDecl, PipelineGraphMode, ReflectedDescriptorType, RenderTier,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceAccess {
    ColorAttachmentWrite,
    DepthAttachmentWrite,
    SampledRead,
    StorageImageRead,
    StorageImageWrite,
    StorageImageReadWrite,
    StorageBufferReadWrite,
    TransferSrc,
    TransferDst,
    AccelerationStructureBuildWrite,
    AccelerationStructureTraceRead,
}

impl ResourceAccess {
    pub const fn to_sync2_masks(
        self,
    ) -> (vk::PipelineStageFlags2, vk::AccessFlags2, vk::ImageLayout) {
        match self {
            Self::ColorAttachmentWrite => (
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ),
            Self::DepthAttachmentWrite => (
                vk::PipelineStageFlags2::from_raw(
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw()
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw(),
                ),
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ),
            Self::SampledRead => (
                vk::PipelineStageFlags2::from_raw(
                    vk::PipelineStageFlags2::FRAGMENT_SHADER.as_raw()
                        | vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw()
                        | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR.as_raw(),
                ),
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ),
            Self::StorageImageRead => (
                vk::PipelineStageFlags2::from_raw(
                    vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw()
                        | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR.as_raw(),
                ),
                vk::AccessFlags2::SHADER_STORAGE_READ,
                vk::ImageLayout::GENERAL,
            ),
            Self::StorageImageWrite => (
                vk::PipelineStageFlags2::from_raw(
                    vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw()
                        | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR.as_raw(),
                ),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::ImageLayout::GENERAL,
            ),
            Self::StorageImageReadWrite => (
                vk::PipelineStageFlags2::from_raw(
                    vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw()
                        | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR.as_raw(),
                ),
                vk::AccessFlags2::from_raw(
                    vk::AccessFlags2::SHADER_STORAGE_READ.as_raw()
                        | vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw(),
                ),
                vk::ImageLayout::GENERAL,
            ),
            Self::StorageBufferReadWrite => (
                vk::PipelineStageFlags2::from_raw(
                    vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw()
                        | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR.as_raw(),
                ),
                vk::AccessFlags2::from_raw(
                    vk::AccessFlags2::SHADER_STORAGE_READ.as_raw()
                        | vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw(),
                ),
                vk::ImageLayout::UNDEFINED,
            ),
            Self::TransferSrc => (
                vk::PipelineStageFlags2::ALL_TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            ),
            Self::TransferDst => (
                vk::PipelineStageFlags2::ALL_TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            ),
            Self::AccelerationStructureBuildWrite => (
                vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
                vk::ImageLayout::UNDEFINED,
            ),
            Self::AccelerationStructureTraceRead => (
                vk::PipelineStageFlags2::from_raw(
                    vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw()
                        | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR.as_raw()
                        | vk::PipelineStageFlags2::FRAGMENT_SHADER.as_raw(),
                ),
                vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
                vk::ImageLayout::UNDEFINED,
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PassKind {
    BlasTlasUpdate,
    ShadowCsm,
    GbufferMrt,
    ComputeVolumetricFog,
    ComputeSsgi,
    ComputeFroxelGi,
    RayQueryLighting,
    FullPathTracing,
    DenoiserSvgfOrNrd,
    DepthReprojectOverlay,
    UpscalerTsrOrVendor,
    PostGradeAndToneMap,
    HdrCompose,
    FrameGenInterpolate,
    Custom(u32),
}

#[derive(Debug, Clone)]
pub struct GraphPassNode {
    pub kind: PassKind,
    pub name: String,
    pub reads: Vec<(ResourceId, ResourceAccess)>,
    pub writes: Vec<(ResourceId, ResourceAccess)>,
    pub enabled: bool,
    /// True if this node represents an external rasterization pass executed by Minecraft's
    /// `LevelRenderer` before `RenderGraph::execute()` runs at `onAfterWorldRenderBeforeGui`.
    pub is_external_raster: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct SynthesizedBarrier {
    pub resource: ResourceId,
    pub src_stage: vk::PipelineStageFlags2,
    pub src_access: vk::AccessFlags2,
    pub dst_stage: vk::PipelineStageFlags2,
    pub dst_access: vk::AccessFlags2,
    pub old_layout: vk::ImageLayout,
    pub new_layout: vk::ImageLayout,
}

#[derive(Debug, Clone)]
pub struct CompiledPassStep {
    pub kind: PassKind,
    pub name: String,
    pub pre_barriers: Vec<SynthesizedBarrier>,
    pub reads: Vec<(ResourceId, ResourceAccess)>,
    pub writes: Vec<(ResourceId, ResourceAccess)>,
    pub is_external_raster: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphConfig {
    pub tier: RenderTier,
    pub denoiser_mode: u32,
    pub upscaler_mode: u32,
    pub framegen_mode: u32,
    pub hdr_enabled: bool,
    pub overlay_pass_enabled: bool,
}

impl GraphConfig {
    /// Returns true when the full-resolution denoiser (SVGF/NRD/DLSS-RR) should execute.
    /// Enabled for Compute, Hybrid, and Path modes whenever a denoiser is selected; Classic
    /// raster mode bypasses denoising since its lighting is deterministic.
    #[inline]
    pub fn effective_denoiser_active(&self) -> bool {
        self.denoiser_mode != 0 && self.tier != RenderTier::Tier0Raster
    }

    /// Returns true when `PostGradeAndToneMap` can write directly into `RES_HUDLESS_COLOR`
    /// on Tier 0 SDR without an intermediate `RES_POST_OUTPUT -> RES_HUDLESS_COLOR` copy.
    #[inline]
    pub fn post_writes_directly_to_hudless(&self) -> bool {
        self.tier == RenderTier::Tier0Raster && !self.hdr_enabled && self.framegen_mode == 0
    }
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            tier: RenderTier::Tier0Raster,
            denoiser_mode: 0,
            upscaler_mode: 0,
            framegen_mode: 0,
            hdr_enabled: false,
            overlay_pass_enabled: true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BoundImageResource {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub format: vk::Format,
    pub aspect: vk::ImageAspectFlags,
    pub array_layers: u32,
    pub owned_view: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct BoundBufferResource {
    pub buffer: vk::Buffer,
    pub offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
}

/// Per-frame Distant Horizons uniform block (`80` bytes, `std140`) bound at:
/// - `compute_lighting.comp` Set 0 binding 14 (`uDh`)
/// - `rayquery_lighting.comp` Set 0 binding 13 (`uDh`)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DhFrameUniforms {
    pub dh_inv_view_proj: [f32; 16],
    pub fade_start: f32,
    pub fade_end: f32,
    pub dh_far_clip: f32,
    pub dh_active: u32,
}

const _: () = assert!(std::mem::size_of::<DhFrameUniforms>() == 80);

impl Default for DhFrameUniforms {
    fn default() -> Self {
        Self::zeroed()
    }
}

/// Declarative Render Graph (`DAG`) with automatic backward-reachability **Dead Pass Culling**,
/// cross-frame temporal layout persistence (`persistent_layouts`), `VK_KHR_synchronization2`
/// barrier synthesis (`VkImageMemoryBarrier2`, `VkBufferMemoryBarrier2`, `VkMemoryBarrier2`),
/// `VSP Graph v2` custom pass/target/buffer extensibility (`Extend` and `Custom` modes),
/// and live command-buffer execution (`execute`).
#[derive(Debug)]
pub struct InternalImageTarget {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub pooled: Option<PooledImageAllocation>,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub view_type: vk::ImageViewType,
    pub array_layers: u32,
    pub width: u32,
    pub height: u32,
}

impl InternalImageTarget {
    /// Destroys the image view, image, and backing memory (returning pooled allocations to
    /// `gpu-allocator` without calling `vkFreeMemory`, or freeing standalone `VkDeviceMemory`
    /// for legacy fallback allocations).
    ///
    /// # Safety
    /// Must only be called when no GPU command buffer referencing this target is in flight.
    pub unsafe fn destroy(mut self, device: &ash::Device) {
        if self.view != vk::ImageView::null() {
            unsafe { device.destroy_image_view(self.view, None) };
            self.view = vk::ImageView::null();
        }
        if let Some(mut pooled) = self.pooled.take() {
            unsafe { pooled.destroy(device) };
            self.image = vk::Image::null();
            self.memory = vk::DeviceMemory::null();
        } else {
            if self.image != vk::Image::null() {
                unsafe { device.destroy_image(self.image, None) };
                self.image = vk::Image::null();
            }
            if self.memory != vk::DeviceMemory::null() {
                unsafe { device.free_memory(self.memory, None) };
                self.memory = vk::DeviceMemory::null();
            }
        }
    }
}

fn allocate_internal_image_target(
    ctx: &VulkanContext,
    img_info: &vk::ImageCreateInfo<'_>,
    view_type: vk::ImageViewType,
    aspect: vk::ImageAspectFlags,
    name: &str,
) -> Option<InternalImageTarget> {
    let (image, memory, mut pooled) = if let Some(ref allocator) = ctx.memory_allocator {
        let pooled = allocator
            .allocate_image(&ctx.device, img_info, MemoryLocation::GpuOnly, name)
            .ok()?;
        let img = pooled.image;
        (img, vk::DeviceMemory::null(), Some(pooled))
    } else {
        let mem_props = unsafe {
            ctx.instance
                .get_physical_device_memory_properties(ctx.physical_device)
        };
        let image = unsafe { ctx.device.create_image(img_info, None) }.ok()?;
        let req = unsafe { ctx.device.get_image_memory_requirements(image) };
        let Some(mem_idx) = (0..mem_props.memory_type_count).find(|&i| {
            (req.memory_type_bits & (1u32 << i)) != 0
                && mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        }) else {
            unsafe { ctx.device.destroy_image(image, None) };
            return None;
        };
        let memory = match unsafe {
            ctx.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mem_idx),
                None,
            )
        } {
            Ok(m) => m,
            Err(_) => {
                unsafe { ctx.device.destroy_image(image, None) };
                return None;
            }
        };
        if unsafe { ctx.device.bind_image_memory(image, memory, 0) }.is_err() {
            unsafe {
                ctx.device.free_memory(memory, None);
                ctx.device.destroy_image(image, None);
            }
            return None;
        }
        (image, memory, None)
    };

    let layers = img_info.array_layers.max(1);
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(view_type)
        .format(img_info.format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: layers,
        });
    let view = match unsafe { ctx.device.create_image_view(&view_info, None) } {
        Ok(v) => v,
        Err(_) => {
            if let Some(mut p) = pooled.take() {
                unsafe { p.destroy(&ctx.device) };
            } else {
                unsafe {
                    ctx.device.free_memory(memory, None);
                    ctx.device.destroy_image(image, None);
                }
            }
            return None;
        }
    };

    Some(InternalImageTarget {
        image,
        memory,
        pooled,
        view,
        format: img_info.format,
        view_type,
        array_layers: layers,
        width: img_info.extent.width,
        height: img_info.extent.height,
    })
}

/// Pack-declared render target from `[targets.<name>]` in `pack.toml`.
#[derive(Debug, Clone)]
pub struct CustomTargetSpec {
    pub id: ResourceId,
    pub name: String,
    pub format: vk::Format,
    pub scale: f32,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub temporal: bool,
    pub history_id: Option<ResourceId>,
    pub clear_color: Option<[f32; 4]>,
    pub gbuffer_attachment: Option<u32>,
}

/// Pack-declared storage buffer from `[buffers.<name>]` in `pack.toml`.
#[derive(Debug, Clone)]
pub struct CustomBufferSpec {
    pub id: ResourceId,
    pub name: String,
    pub size_bytes: u64,
    pub clear_each_frame: bool,
}

/// Pack-declared programmable pass from `[[passes]]` in `pack.toml`.
#[derive(Debug, Clone)]
pub struct CustomPassSpec {
    pub custom_index: u32,
    pub kind: PassKind,
    pub id: String,
    pub shader: String,
    pub min_tier: RenderTier,
    pub max_tier: RenderTier,
    pub reads: Vec<(ResourceId, ResourceAccess)>,
    pub writes: Vec<(ResourceId, ResourceAccess)>,
    pub before: Option<String>,
    pub after: Option<String>,
    pub replaces: Option<String>,
    pub bindings: HashMap<u32, ResourceId>,
    pub scale: f32,
    pub enabled: bool,
}

pub struct RenderGraph {
    pub config: GraphConfig,
    pub graph_mode: PipelineGraphMode,
    pub compiled_steps: Vec<CompiledPassStep>,
    pub resource_images: HashMap<ResourceId, vk::Image>,
    pub bound_images: HashMap<ResourceId, BoundImageResource>,
    pub bound_buffers: HashMap<ResourceId, BoundBufferResource>,
    pub persistent_layouts:
        HashMap<ResourceId, (vk::PipelineStageFlags2, vk::AccessFlags2, vk::ImageLayout)>,
    pub tracked_image_layouts: HashMap<vk::Image, vk::ImageLayout>,
    frame_live_image_states: HashMap<
        vk::Image,
        (
            vk::PipelineStageFlags2,
            vk::AccessFlags2,
            vk::ImageLayout,
            vk::ImageAspectFlags,
        ),
    >,
    frame_mojang_owned_images: Vec<(vk::Image, vk::ImageAspectFlags)>,
    frame_init_clear_ids: Vec<(ResourceId, [f32; 4], bool)>,
    frame_exit_barriers: Vec<vk::ImageMemoryBarrier2<'static>>,
    frame_seen_exit_images: HashSet<vk::Image>,
    pub internal_targets: HashMap<ResourceId, InternalImageTarget>,
    pub internal_render_size: (u32, u32),
    pub internal_target_size: (u32, u32),
    pub shadow_resolution: u32,
    pub shadow_distance_chunks: u32,
    pub custom_targets: HashMap<ResourceId, CustomTargetSpec>,
    pub custom_buffers: HashMap<ResourceId, CustomBufferSpec>,
    pub custom_buffer_allocs: HashMap<ResourceId, GpuBufferAllocation>,
    pub custom_passes: HashMap<u32, CustomPassSpec>,
    pub named_resources: HashMap<String, ResourceId>,
    pub temporal_custom_resources: HashSet<ResourceId>,
    pub retired_views: Vec<vk::ImageView>,
    pub dh_uniforms: DhFrameUniforms,
    pub dh_ubo_slots: [Option<GpuBufferAllocation>; 3],
    pub dh_target_size: (u32, u32),
    pub dh_dummy_color: Option<InternalImageTarget>,
    pub dh_dummy_depth: Option<InternalImageTarget>,
    next_custom_resource_id: u32,
}

impl Default for RenderGraph {
    fn default() -> Self {
        let mut graph = Self {
            config: GraphConfig::default(),
            graph_mode: PipelineGraphMode::Extend,
            compiled_steps: Vec::new(),
            resource_images: HashMap::new(),
            bound_images: HashMap::new(),
            bound_buffers: HashMap::new(),
            persistent_layouts: HashMap::new(),
            tracked_image_layouts: HashMap::new(),
            frame_live_image_states: HashMap::with_capacity(256),
            frame_mojang_owned_images: Vec::with_capacity(32),
            frame_init_clear_ids: Vec::with_capacity(32),
            frame_exit_barriers: Vec::with_capacity(32),
            frame_seen_exit_images: HashSet::with_capacity(32),
            internal_targets: HashMap::new(),
            internal_render_size: (0, 0),
            internal_target_size: (0, 0),
            shadow_resolution: 2048,
            shadow_distance_chunks: 12,
            custom_targets: HashMap::new(),
            custom_buffers: HashMap::new(),
            custom_buffer_allocs: HashMap::new(),
            custom_passes: HashMap::new(),
            named_resources: HashMap::new(),
            temporal_custom_resources: HashSet::new(),
            retired_views: Vec::new(),
            dh_uniforms: DhFrameUniforms::default(),
            dh_ubo_slots: [None, None, None],
            dh_target_size: (0, 0),
            dh_dummy_color: None,
            dh_dummy_depth: None,
            next_custom_resource_id: Self::RES_CUSTOM_BASE,
        };
        graph.seed_canonical_named_resources();
        graph.rebuild(GraphConfig::default());
        graph
    }
}

impl RenderGraph {
    pub const RES_TLAS: ResourceId = ResourceId(1);
    pub const RES_SHADOW_CSM: ResourceId = ResourceId(2);
    pub const RES_GBUFFER_ALBEDO: ResourceId = ResourceId(3);
    pub const RES_GBUFFER_NORMAL: ResourceId = ResourceId(4);
    pub const RES_GBUFFER_MATERIAL: ResourceId = ResourceId(5);
    pub const RES_GBUFFER_VELOCITY: ResourceId = ResourceId(6);
    pub const RES_LINEAR_DEPTH: ResourceId = ResourceId(7);
    pub const RES_RAW_RADIANCE: ResourceId = ResourceId(8);
    pub const RES_DENOISED_RADIANCE: ResourceId = ResourceId(9);
    pub const RES_UPSCALED_HDR: ResourceId = ResourceId(10);
    pub const RES_HUDLESS_COLOR: ResourceId = ResourceId(11);
    pub const RES_MC_MAIN_DEPTH: ResourceId = ResourceId(12);
    pub const RES_SWAPCHAIN_OUT: ResourceId = ResourceId(13);
    pub const RES_PREV_HUDLESS_COLOR: ResourceId = ResourceId(14);
    pub const RES_UI_COLOR: ResourceId = ResourceId(15);
    pub const RES_SHARC_HASH_TABLE: ResourceId = ResourceId(16);
    pub const RES_INTERPOLATED_FRAME: ResourceId = ResourceId(17);
    pub const RES_SVGF_PING: ResourceId = ResourceId(18);
    pub const RES_POST_OUTPUT: ResourceId = ResourceId(19);
    pub const RES_SSGI_HISTORY: ResourceId = ResourceId(20);
    pub const RES_SSGI_OUT: ResourceId = ResourceId(21);
    pub const RES_SVGF_HISTORY_RADIANCE: ResourceId = ResourceId(22);
    pub const RES_SVGF_HISTORY_MOMENTS: ResourceId = ResourceId(23);
    pub const RES_SVGF_MOMENTS: ResourceId = ResourceId(24);
    pub const RES_TSR_HISTORY: ResourceId = ResourceId(25);
    pub const RES_VOLUMETRIC_FOG: ResourceId = ResourceId(26);
    pub const RES_VOLUMETRIC_FOG_HISTORY: ResourceId = ResourceId(30);
    pub const RES_EMISSIVE_LIGHTS: ResourceId = ResourceId(27);
    pub const RES_RESTIR_RESERVOIRS: ResourceId = ResourceId(28);
    // Tier-3 path tracing writes the primary albedo/normal from rgen/rchit into this
    // rgba16f target so SVGF/SHaRC can demodulate albedo without a raster G-Buffer.
    pub const RES_PT_ALBEDO: ResourceId = ResourceId(29);
    pub const RES_SHARP_RADIANCE: ResourceId = ResourceId(31);
    pub const RES_NRD_IN_DIFF: ResourceId = ResourceId(32);
    pub const RES_NRD_IN_SPEC: ResourceId = ResourceId(33);
    pub const RES_NRD_OUT_SPEC: ResourceId = ResourceId(34);
    pub const RES_NRD_NORMAL_ROUGH: ResourceId = ResourceId(35);
    pub const RES_NRD_VIEWZ: ResourceId = ResourceId(36);
    pub const RES_RT_HIT_DIST: ResourceId = ResourceId(37);
    pub const RES_DH_ALBEDO: ResourceId = ResourceId(38);
    pub const RES_DH_NORMAL: ResourceId = ResourceId(39);
    pub const RES_DH_MATERIAL: ResourceId = ResourceId(40);
    pub const RES_DH_DEPTH: ResourceId = ResourceId(41);
    pub const RES_CUSTOM_BASE: u32 = 1000;

    fn seed_canonical_named_resources(&mut self) {
        let pairs = [
            ("tlas", Self::RES_TLAS),
            ("uTlas", Self::RES_TLAS),
            ("shadow_csm", Self::RES_SHADOW_CSM),
            ("uShadowCsm", Self::RES_SHADOW_CSM),
            ("shadowtex0", Self::RES_SHADOW_CSM),
            ("gbuffer_albedo", Self::RES_GBUFFER_ALBEDO),
            ("uGbufferAlbedo", Self::RES_GBUFFER_ALBEDO),
            ("colortex0", Self::RES_GBUFFER_ALBEDO),
            ("gbuffer_normal", Self::RES_GBUFFER_NORMAL),
            ("uGbufferNormal", Self::RES_GBUFFER_NORMAL),
            ("colortex1", Self::RES_GBUFFER_NORMAL),
            ("gbuffer_material", Self::RES_GBUFFER_MATERIAL),
            ("uGbufferMaterial", Self::RES_GBUFFER_MATERIAL),
            ("colortex2", Self::RES_GBUFFER_MATERIAL),
            ("gbuffer_velocity", Self::RES_GBUFFER_VELOCITY),
            ("uVelocity", Self::RES_GBUFFER_VELOCITY),
            ("colortex3", Self::RES_GBUFFER_VELOCITY),
            ("linear_depth", Self::RES_LINEAR_DEPTH),
            ("uLinearDepth", Self::RES_LINEAR_DEPTH),
            ("raw_radiance", Self::RES_RAW_RADIANCE),
            ("uRawRadiance", Self::RES_RAW_RADIANCE),
            ("uOutRadiance", Self::RES_RAW_RADIANCE),
            ("denoised_radiance", Self::RES_DENOISED_RADIANCE),
            ("uDenoisedRadiance", Self::RES_DENOISED_RADIANCE),
            ("upscaled_hdr", Self::RES_UPSCALED_HDR),
            ("uUpscaledHdr", Self::RES_UPSCALED_HDR),
            ("hudless_color", Self::RES_HUDLESS_COLOR),
            ("uHudlessColor", Self::RES_HUDLESS_COLOR),
            ("uCurrColor", Self::RES_HUDLESS_COLOR),
            ("mc_main_depth", Self::RES_MC_MAIN_DEPTH),
            ("uMainDepth", Self::RES_MC_MAIN_DEPTH),
            ("depthtex0", Self::RES_MC_MAIN_DEPTH),
            ("swapchain_out", Self::RES_SWAPCHAIN_OUT),
            ("uSwapchainOut", Self::RES_SWAPCHAIN_OUT),
            ("prev_hudless_color", Self::RES_PREV_HUDLESS_COLOR),
            ("uPrevHudlessColor", Self::RES_PREV_HUDLESS_COLOR),
            ("uPrevColor", Self::RES_PREV_HUDLESS_COLOR),
            ("ui_color", Self::RES_UI_COLOR),
            ("uUiColor", Self::RES_UI_COLOR),
            ("uGuiColor", Self::RES_UI_COLOR),
            ("sharc_hash_table", Self::RES_SHARC_HASH_TABLE),
            ("interpolated_frame", Self::RES_INTERPOLATED_FRAME),
            ("uInterpolatedFrame", Self::RES_INTERPOLATED_FRAME),
            ("svgf_ping", Self::RES_SVGF_PING),
            ("post_output", Self::RES_POST_OUTPUT),
            ("ssgi_history", Self::RES_SSGI_HISTORY),
            ("uSsgiHistory", Self::RES_SSGI_HISTORY),
            ("ssgi_out", Self::RES_SSGI_OUT),
            ("uSsgi", Self::RES_SSGI_OUT),
            ("uSsgiOut", Self::RES_SSGI_OUT),
            ("svgf_history_radiance", Self::RES_SVGF_HISTORY_RADIANCE),
            ("uHistoryRadiance", Self::RES_SVGF_HISTORY_RADIANCE),
            ("svgf_history_moments", Self::RES_SVGF_HISTORY_MOMENTS),
            ("uHistoryMoments", Self::RES_SVGF_HISTORY_MOMENTS),
            ("svgf_moments", Self::RES_SVGF_MOMENTS),
            ("uMoments", Self::RES_SVGF_MOMENTS),
            ("uSvgfMoments", Self::RES_SVGF_MOMENTS),
            ("tsr_history", Self::RES_TSR_HISTORY),
            ("uTsrHistory", Self::RES_TSR_HISTORY),
            ("volumetric_fog", Self::RES_VOLUMETRIC_FOG),
            ("uVolumetricFog", Self::RES_VOLUMETRIC_FOG),
            ("uOutVolumetricFog", Self::RES_VOLUMETRIC_FOG),
            ("volumetric_fog_history", Self::RES_VOLUMETRIC_FOG_HISTORY),
            ("uFogHistory", Self::RES_VOLUMETRIC_FOG_HISTORY),
            ("emissive_lights", Self::RES_EMISSIVE_LIGHTS),
            ("restir_reservoirs", Self::RES_RESTIR_RESERVOIRS),
            ("pt_albedo", Self::RES_PT_ALBEDO),
            ("uPtAlbedo", Self::RES_PT_ALBEDO),
            ("sharp_radiance", Self::RES_SHARP_RADIANCE),
            ("uSharpRadiance", Self::RES_SHARP_RADIANCE),
            ("nrd_in_diff", Self::RES_NRD_IN_DIFF),
            ("uNrdInDiff", Self::RES_NRD_IN_DIFF),
            ("uOutNrdInDiff", Self::RES_NRD_IN_DIFF),
            ("nrd_in_spec", Self::RES_NRD_IN_SPEC),
            ("uNrdInSpec", Self::RES_NRD_IN_SPEC),
            ("uOutNrdInSpec", Self::RES_NRD_IN_SPEC),
            ("nrd_out_spec", Self::RES_NRD_OUT_SPEC),
            ("uNrdOutSpec", Self::RES_NRD_OUT_SPEC),
            ("uInNrdOutSpec", Self::RES_NRD_OUT_SPEC),
            ("nrd_normal_rough", Self::RES_NRD_NORMAL_ROUGH),
            ("uNrdNormalRough", Self::RES_NRD_NORMAL_ROUGH),
            ("uOutNrdNormalRough", Self::RES_NRD_NORMAL_ROUGH),
            ("nrd_viewz", Self::RES_NRD_VIEWZ),
            ("uNrdViewZ", Self::RES_NRD_VIEWZ),
            ("uNrdViewz", Self::RES_NRD_VIEWZ),
            ("uOutNrdViewZ", Self::RES_NRD_VIEWZ),
            ("uOutNrdViewz", Self::RES_NRD_VIEWZ),
            ("rt_hit_dist", Self::RES_RT_HIT_DIST),
            ("uRtHitDist", Self::RES_RT_HIT_DIST),
            ("uOutRtHitDist", Self::RES_RT_HIT_DIST),
            ("dh_albedo", Self::RES_DH_ALBEDO),
            ("uDhAlbedo", Self::RES_DH_ALBEDO),
            ("dh_normal", Self::RES_DH_NORMAL),
            ("uDhNormal", Self::RES_DH_NORMAL),
            ("dh_material", Self::RES_DH_MATERIAL),
            ("uDhMaterial", Self::RES_DH_MATERIAL),
            ("dh_depth", Self::RES_DH_DEPTH),
            ("uDhDepth", Self::RES_DH_DEPTH),
        ];
        for (name, id) in pairs {
            self.named_resources.insert(name.to_owned(), id);
            self.named_resources.insert(name.to_ascii_lowercase(), id);
        }
    }

    pub fn parse_vk_format(fmt: &str) -> vk::Format {
        match fmt.trim().to_ascii_lowercase().as_str() {
            "rgba16f" | "r16g16b16a16_sfloat" => vk::Format::R16G16B16A16_SFLOAT,
            "rgba8" | "r8g8b8a8_unorm" => vk::Format::R8G8B8A8_UNORM,
            "srgb8_a8" | "r8g8b8a8_srgb" => vk::Format::R8G8B8A8_SRGB,
            "rgba8_snorm" | "r8g8b8a8_snorm" => vk::Format::R8G8B8A8_SNORM,
            "rg16f" | "r16g16_sfloat" => vk::Format::R16G16_SFLOAT,
            "rg32f" | "r32g32_sfloat" => vk::Format::R32G32_SFLOAT,
            "r32f" | "r32_sfloat" => vk::Format::R32_SFLOAT,
            "r16f" | "r16_sfloat" => vk::Format::R16_SFLOAT,
            "r8" | "r8_unorm" => vk::Format::R8_UNORM,
            "r11g11b10f" | "b10g11r11_ufloat_pack32" => vk::Format::B10G11R11_UFLOAT_PACK32,
            "rgb10_a2" | "a2b10g10r10_unorm_pack32" => vk::Format::A2B10G10R10_UNORM_PACK32,
            "rgba32f" | "r32g32b32a32_sfloat" => vk::Format::R32G32B32A32_SFLOAT,
            "r32ui" | "r32_uint" => vk::Format::R32_UINT,
            "rg32ui" | "r32g32_uint" => vk::Format::R32G32_UINT,
            "rgba32ui" | "r32g32b32a32_uint" => vk::Format::R32G32B32A32_UINT,
            "d32f" | "d32_sfloat" => vk::Format::D32_SFLOAT,
            _ => vk::Format::R16G16B16A16_SFLOAT,
        }
    }

    /// Resolves a resource name (either canonical or pack-declared in `[targets.*]` / `[buffers.*]`)
    /// or a GLSL uniform variable name (`uVolumetricFog`, `u_custom_cloud`, `img_bloom`) to its `ResourceId`.
    pub fn resolve_resource_by_name(&self, raw_name: &str) -> Option<ResourceId> {
        let trimmed = raw_name.trim();
        if trimmed.is_empty() {
            return None;
        }
        if let Some(&id) = self.named_resources.get(trimmed) {
            return Some(id);
        }
        let lower = trimmed.to_ascii_lowercase();
        if let Some(&id) = self.named_resources.get(&lower) {
            return Some(id);
        }
        // Strip common uniform prefixes (`u_`, `u`, `img_`, `tex_`, `in_`, `out_`) and convert CamelCase to snake_case
        for prefix in ["u_out_", "u_in_", "u_", "img_", "tex_", "in_", "out_"] {
            if let Some(rest) = lower.strip_prefix(prefix)
                && let Some(&id) = self.named_resources.get(rest)
            {
                return Some(id);
            }
        }
        if trimmed.starts_with('u')
            && trimmed.len() > 1
            && trimmed.as_bytes()[1].is_ascii_uppercase()
        {
            let mut snake = String::new();
            for (i, ch) in trimmed[1..].chars().enumerate() {
                if ch.is_ascii_uppercase() {
                    if i > 0 {
                        snake.push('_');
                    }
                    snake.push(ch.to_ascii_lowercase());
                } else {
                    snake.push(ch);
                }
            }
            if let Some(&id) = self.named_resources.get(&snake) {
                return Some(id);
            }
            if let Some(stripped) = snake.strip_prefix("out_")
                && let Some(&id) = self.named_resources.get(stripped)
            {
                return Some(id);
            }
        }
        None
    }

    fn get_or_alloc_resource_id(&mut self, name: &str) -> ResourceId {
        if let Some(existing) = self.resolve_resource_by_name(name) {
            return existing;
        }
        let id = ResourceId(self.next_custom_resource_id);
        self.next_custom_resource_id += 1;
        self.named_resources.insert(name.to_owned(), id);
        self.named_resources.insert(name.to_ascii_lowercase(), id);
        id
    }

    /// Registers all `[pipeline]`, `[targets.<name>]`, `[buffers.<name>]`, and `[[passes]]`
    /// declarations from a `VSP Graph v2` `PackManifest` and rebuilds the active `RenderGraph`.
    pub fn register_pack_manifest_graph(&mut self, manifest: &PackManifest) {
        self.custom_targets.clear();
        self.custom_buffers.clear();
        self.custom_passes.clear();
        self.temporal_custom_resources.clear();
        self.named_resources.clear();
        self.next_custom_resource_id = Self::RES_CUSTOM_BASE;
        self.seed_canonical_named_resources();
        self.graph_mode = manifest.pipeline.mode;

        for (name, decl) in &manifest.targets {
            let id = self.get_or_alloc_resource_id(name);
            let format = Self::parse_vk_format(&decl.format);
            let history_id = if decl.temporal {
                let hist_name = format!("{name}_history");
                let hid = self.get_or_alloc_resource_id(&hist_name);
                self.temporal_custom_resources.insert(id);
                self.temporal_custom_resources.insert(hid);
                Some(hid)
            } else {
                None
            };
            self.custom_targets.insert(
                id,
                CustomTargetSpec {
                    id,
                    name: name.clone(),
                    format,
                    scale: decl.scale.clamp(0.125, 4.0),
                    mip_levels: decl.mip_levels.max(1),
                    array_layers: decl.array_layers.max(1),
                    temporal: decl.temporal,
                    history_id,
                    clear_color: decl.clear_color,
                    gbuffer_attachment: decl.gbuffer_attachment,
                },
            );
        }

        for (name, buf_decl) in &manifest.buffers {
            let id = self.get_or_alloc_resource_id(name);
            if buf_decl.persistent {
                self.temporal_custom_resources.insert(id);
            }
            let effective_bytes = if buf_decl.element_count > 0 && buf_decl.stride > 0 {
                u64::from(buf_decl.element_count) * u64::from(buf_decl.stride)
            } else {
                buf_decl.size_bytes
            };
            self.custom_buffers.insert(
                id,
                CustomBufferSpec {
                    id,
                    name: name.clone(),
                    size_bytes: effective_bytes.max(64),
                    clear_each_frame: buf_decl.clear_each_frame,
                },
            );
        }

        for (idx, pass_decl) in manifest.passes.iter().enumerate() {
            let custom_idx = (idx as u32) + 1;
            let kind = Self::resolve_pass_kind_from_decl(pass_decl, custom_idx);
            let min_tier =
                RenderTier::from_index(pass_decl.min_tier).unwrap_or(RenderTier::Tier0Raster);
            let max_tier =
                RenderTier::from_index(pass_decl.max_tier).unwrap_or(RenderTier::Tier3PathTracing);

            let mut reads = Vec::with_capacity(pass_decl.reads.len());
            for r_name in &pass_decl.reads {
                let res_id = self.get_or_alloc_resource_id(r_name);
                let access = if res_id == Self::RES_TLAS {
                    ResourceAccess::AccelerationStructureTraceRead
                } else if self.custom_buffers.contains_key(&res_id)
                    || res_id == Self::RES_SHARC_HASH_TABLE
                    || res_id == Self::RES_EMISSIVE_LIGHTS
                    || res_id == Self::RES_RESTIR_RESERVOIRS
                {
                    ResourceAccess::StorageBufferReadWrite
                } else {
                    ResourceAccess::SampledRead
                };
                reads.push((res_id, access));
            }

            let mut writes = Vec::with_capacity(pass_decl.writes.len());
            for w_name in &pass_decl.writes {
                let res_id = self.get_or_alloc_resource_id(w_name);
                let access = if self.custom_buffers.contains_key(&res_id)
                    || res_id == Self::RES_SHARC_HASH_TABLE
                    || res_id == Self::RES_RESTIR_RESERVOIRS
                {
                    ResourceAccess::StorageBufferReadWrite
                } else if res_id == Self::RES_MC_MAIN_DEPTH || res_id == Self::RES_SHADOW_CSM {
                    ResourceAccess::DepthAttachmentWrite
                } else {
                    ResourceAccess::StorageImageWrite
                };
                writes.push((res_id, access));
            }

            let mut bindings = HashMap::new();
            for (b_key, target_name) in &pass_decl.bindings {
                let trimmed = b_key.trim();
                let b_num_str = trimmed
                    .rsplit(':')
                    .next()
                    .unwrap_or(trimmed)
                    .trim()
                    .trim_start_matches("binding")
                    .trim_start_matches('b');
                if let Ok(b_num) = b_num_str.parse::<u32>() {
                    let res_id = self.get_or_alloc_resource_id(target_name);
                    bindings.insert(b_num, res_id);
                }
            }

            let shader_key = pass_decl
                .shader
                .rsplit('/')
                .next()
                .unwrap_or(pass_decl.shader.as_str())
                .to_owned();

            self.custom_passes.insert(
                custom_idx,
                CustomPassSpec {
                    custom_index: custom_idx,
                    kind,
                    id: pass_decl.id.clone(),
                    shader: shader_key,
                    min_tier,
                    max_tier,
                    reads,
                    writes,
                    before: pass_decl.before.clone(),
                    after: pass_decl.after.clone(),
                    replaces: pass_decl.replaces.clone(),
                    bindings,
                    scale: pass_decl.scale.clamp(0.125, 4.0),
                    enabled: pass_decl.enabled,
                },
            );
        }

        // Force internal target reallocation on the next frame if new custom targets were added.
        self.internal_render_size = (0, 0);
        self.internal_target_size = (0, 0);
        self.rebuild(self.config);
    }

    fn resolve_pass_kind_from_decl(decl: &PackPassDecl, custom_idx: u32) -> PassKind {
        let id_lower = decl.id.trim().to_ascii_lowercase();
        let shader_file = decl
            .shader
            .rsplit('/')
            .next()
            .unwrap_or(decl.shader.as_str())
            .to_ascii_lowercase();
        if id_lower == "volumetric_fog"
            || id_lower == "volumetric_fog_pass"
            || shader_file == "volumetric_fog.comp"
        {
            PassKind::ComputeVolumetricFog
        } else {
            PassKind::Custom(custom_idx)
        }
    }

    /// Returns true if `id` carries temporal history across frames and MUST NOT be transitioned
    /// from `VK_IMAGE_LAYOUT_UNDEFINED` after its first initialization frame.
    pub const fn is_temporal_resource(id: ResourceId) -> bool {
        matches!(
            id,
            Self::RES_PREV_HUDLESS_COLOR
                | Self::RES_GBUFFER_VELOCITY
                | Self::RES_DENOISED_RADIANCE
                | Self::RES_UPSCALED_HDR
                | Self::RES_SHARC_HASH_TABLE
                | Self::RES_INTERPOLATED_FRAME
                | Self::RES_SSGI_HISTORY
                | Self::RES_VOLUMETRIC_FOG_HISTORY
                | Self::RES_SVGF_HISTORY_RADIANCE
                | Self::RES_SVGF_HISTORY_MOMENTS
                | Self::RES_TSR_HISTORY
                | Self::RES_VOLUMETRIC_FOG
                | Self::RES_RESTIR_RESERVOIRS
        )
    }

    pub fn is_resource_temporal(&self, id: ResourceId) -> bool {
        Self::is_temporal_resource(id) || self.temporal_custom_resources.contains(&id)
    }

    pub const fn default_format_and_aspect(id: ResourceId) -> (vk::Format, vk::ImageAspectFlags) {
        match id {
            Self::RES_SHADOW_CSM | Self::RES_MC_MAIN_DEPTH | Self::RES_DH_DEPTH => {
                (vk::Format::D32_SFLOAT, vk::ImageAspectFlags::DEPTH)
            }
            Self::RES_LINEAR_DEPTH | Self::RES_NRD_VIEWZ => {
                (vk::Format::R32_SFLOAT, vk::ImageAspectFlags::COLOR)
            }
            Self::RES_RT_HIT_DIST => (vk::Format::R16_SFLOAT, vk::ImageAspectFlags::COLOR),
            Self::RES_PT_ALBEDO
            | Self::RES_NRD_IN_DIFF
            | Self::RES_NRD_IN_SPEC
            | Self::RES_NRD_OUT_SPEC
            | Self::RES_NRD_NORMAL_ROUGH => {
                (vk::Format::R16G16B16A16_SFLOAT, vk::ImageAspectFlags::COLOR)
            }
            Self::RES_GBUFFER_ALBEDO
            | Self::RES_GBUFFER_MATERIAL
            | Self::RES_DH_ALBEDO
            | Self::RES_DH_MATERIAL
            | Self::RES_HUDLESS_COLOR
            | Self::RES_PREV_HUDLESS_COLOR
            | Self::RES_UI_COLOR
            | Self::RES_POST_OUTPUT => (vk::Format::R8G8B8A8_UNORM, vk::ImageAspectFlags::COLOR),
            Self::RES_GBUFFER_VELOCITY => (vk::Format::R16G16_SFLOAT, vk::ImageAspectFlags::COLOR),
            _ => (vk::Format::R16G16B16A16_SFLOAT, vk::ImageAspectFlags::COLOR),
        }
    }

    pub fn bind_resource_image(&mut self, id: ResourceId, image: vk::Image) {
        let (fmt, aspect) = Self::default_format_and_aspect(id);
        self.bind_resource_image_detailed(
            None,
            id,
            image,
            vk::ImageView::null(),
            vk::Sampler::null(),
            fmt,
            aspect,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_resource_image_detailed(
        &mut self,
        ctx: Option<&VulkanContext>,
        id: ResourceId,
        image: vk::Image,
        mut view: vk::ImageView,
        sampler: vk::Sampler,
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
    ) {
        if image == vk::Image::null() {
            if let Some(old) = self.bound_images.remove(&id)
                && old.owned_view
                && old.view != vk::ImageView::null()
            {
                self.retired_views.push(old.view);
            }
            self.resource_images.remove(&id);
            self.persistent_layouts.remove(&id);
            return;
        }

        let array_layers = if id == Self::RES_SHADOW_CSM { 4 } else { 1 };
        let mut owned_view = false;
        if let Some(prev) = self.bound_images.get(&id) {
            if prev.image == image
                && view == vk::ImageView::null()
                && prev.view != vk::ImageView::null()
            {
                view = prev.view;
                owned_view = prev.owned_view;
            } else if prev.image != image {
                if prev.owned_view && prev.view != vk::ImageView::null() {
                    self.retired_views.push(prev.view);
                }
                self.persistent_layouts.remove(&id);
            }
        }

        if view == vk::ImageView::null()
            && let Some(c) = ctx
            && format != vk::Format::UNDEFINED
        {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(if array_layers > 1 {
                    vk::ImageViewType::TYPE_2D_ARRAY
                } else {
                    vk::ImageViewType::TYPE_2D
                })
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: aspect,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: array_layers,
                });
            if let Ok(created) = unsafe { c.device.create_image_view(&view_info, None) } {
                view = created;
                owned_view = true;
            }
        }

        self.resource_images.insert(id, image);
        self.bound_images.insert(
            id,
            BoundImageResource {
                image,
                view,
                sampler,
                format,
                aspect,
                array_layers,
                owned_view,
            },
        );
    }

    pub fn bind_resource_buffer(
        &mut self,
        id: ResourceId,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) {
        if buffer == vk::Buffer::null() {
            self.bound_buffers.remove(&id);
        } else {
            self.bound_buffers.insert(
                id,
                BoundBufferResource {
                    buffer,
                    offset,
                    size,
                },
            );
        }
    }

    pub fn resolve_image_view(&self, id: ResourceId) -> Option<vk::ImageView> {
        if matches!(
            id,
            Self::RES_DH_ALBEDO | Self::RES_DH_NORMAL | Self::RES_DH_MATERIAL | Self::RES_DH_DEPTH
        ) {
            if self.dh_uniforms.dh_active != 0
                && let Some(view) = self
                    .bound_images
                    .get(&id)
                    .map(|b| b.view)
                    .filter(|&v| v != vk::ImageView::null())
            {
                return Some(view);
            }
            return if id == Self::RES_DH_DEPTH {
                self.dh_dummy_depth.as_ref().map(|t| t.view)
            } else {
                self.dh_dummy_color.as_ref().map(|t| t.view)
            };
        }
        self.bound_images
            .get(&id)
            .map(|b| b.view)
            .filter(|&v| v != vk::ImageView::null())
    }

    /// Returns the host-visible `DhFrameUniforms` UBO handle for `frame_slot` (`0..3`).
    pub fn dh_ubo_buffer_for_slot(&self, frame_slot: usize) -> Option<vk::Buffer> {
        self.dh_ubo_slots[frame_slot % 3]
            .as_ref()
            .map(|b| b.buffer)
            .filter(|&b| b != vk::Buffer::null())
    }

    /// Stores external Distant Horizons G-Buffer target handles and per-frame uniforms.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_dh_targets(
        &mut self,
        ctx: Option<&VulkanContext>,
        dh_albedo: (u64, u64),
        dh_normal: (u64, u64),
        dh_material: (u64, u64),
        dh_depth: (u64, u64),
        width: u32,
        height: u32,
        uniforms: DhFrameUniforms,
    ) {
        self.dh_uniforms = uniforms;
        self.dh_target_size = (width, height);
        let targets = [
            (
                Self::RES_DH_ALBEDO,
                dh_albedo,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageAspectFlags::COLOR,
            ),
            (
                Self::RES_DH_NORMAL,
                dh_normal,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageAspectFlags::COLOR,
            ),
            (
                Self::RES_DH_MATERIAL,
                dh_material,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageAspectFlags::COLOR,
            ),
            (
                Self::RES_DH_DEPTH,
                dh_depth,
                vk::Format::D32_SFLOAT,
                vk::ImageAspectFlags::DEPTH,
            ),
        ];
        for (res_id, (img_raw, view_raw), fmt, aspect) in targets {
            if uniforms.dh_active != 0 && img_raw != 0 {
                self.bind_resource_image_detailed(
                    ctx,
                    res_id,
                    vk::Image::from_raw(img_raw),
                    vk::ImageView::from_raw(view_raw),
                    vk::Sampler::null(),
                    fmt,
                    aspect,
                );
            } else {
                if let Some(old) = self.bound_images.remove(&res_id)
                    && old.owned_view
                    && old.view != vk::ImageView::null()
                {
                    self.retired_views.push(old.view);
                }
                self.resource_images.remove(&res_id);
            }
        }
    }

    /// Ensures the 1x1 zeroed DH dummy fallback images and per-frame `DhFrameUniforms` UBO exist
    /// and uploads the current frame's `dh_uniforms` into `dh_ubo_slots[frame_slot % 3]`.
    fn ensure_dh_resources(
        &mut self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
    ) {
        let slot = frame_slot % 3;
        if self.dh_ubo_slots[slot].is_none()
            && let Ok(alloc) = GpuBufferAllocation::allocate_host_visible_bda(
                ctx,
                std::mem::size_of::<DhFrameUniforms>() as vk::DeviceSize,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )
        {
            self.dh_ubo_slots[slot] = Some(alloc);
        }
        if let Some(ref mut buf) = self.dh_ubo_slots[slot] {
            buf.write_bytes(0, bytemuck::bytes_of(&self.dh_uniforms));
        }

        let create_dummy_1x1 = |format: vk::Format| -> Option<InternalImageTarget> {
            let img_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            allocate_internal_image_target(
                ctx,
                &img_info,
                vk::ImageViewType::TYPE_2D,
                vk::ImageAspectFlags::COLOR,
                "graph_dh_dummy_1x1",
            )
        };

        let mut init_images = Vec::new();
        if self.dh_dummy_color.is_none()
            && let Some(target) = create_dummy_1x1(vk::Format::R8G8B8A8_UNORM)
        {
            init_images.push(target.image);
            self.dh_dummy_color = Some(target);
        }
        if self.dh_dummy_depth.is_none()
            && let Some(target) = create_dummy_1x1(vk::Format::R32_SFLOAT)
        {
            init_images.push(target.image);
            self.dh_dummy_depth = Some(target);
        }

        if !init_images.is_empty() {
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            for img in init_images {
                let to_clear = [vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::CLEAR)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(img)
                    .subresource_range(range)];
                let to_read = [vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(img)
                    .subresource_range(range)];
                unsafe {
                    ctx.sync2_ext.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&to_clear),
                    );
                    ctx.device.cmd_clear_color_image(
                        cmd,
                        img,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &vk::ClearColorValue { float32: [0.0; 4] },
                        &[range],
                    );
                    ctx.sync2_ext.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&to_read),
                    );
                }
            }
        }
    }

    pub fn resolve_sampler(&self, id: ResourceId) -> Option<vk::Sampler> {
        self.bound_images
            .get(&id)
            .map(|b| b.sampler)
            .filter(|&s| s != vk::Sampler::null())
    }

    /// Resets cross-frame temporal layout state (called on world change or swapchain resize).
    pub fn reset_temporal_history(&mut self) {
        self.persistent_layouts.clear();
        self.tracked_image_layouts.clear();
    }

    pub fn rebuild(&mut self, cfg: GraphConfig) {
        self.config = cfg;
        self.internal_render_size = (0, 0);
        self.internal_target_size = (0, 0);
        let nodes = self.build_nodes_with_custom(cfg);
        let sinks = if cfg.hdr_enabled || cfg.framegen_mode != 0 {
            [Self::RES_SWAPCHAIN_OUT, Self::RES_MC_MAIN_DEPTH]
        } else {
            [Self::RES_HUDLESS_COLOR, Self::RES_MC_MAIN_DEPTH]
        };
        self.compiled_steps = self.compile_and_cull_dynamic(&nodes, &sinks);
    }

    /// Retires all internal render targets and custom buffers through `RetireQueue` instead of
    /// calling `vkDeviceWaitIdle`.
    fn flush_retired_views(&mut self, ctx: &VulkanContext, retire: &RetireQueue) {
        if self.retired_views.is_empty() {
            return;
        }
        let views = std::mem::take(&mut self.retired_views);
        let dev = ctx.device.clone();
        retire.retire_later(move || unsafe {
            for v in views {
                if v != vk::ImageView::null() {
                    dev.destroy_image_view(v, None);
                }
            }
        });
    }

    fn destroy_internal_targets(&mut self, ctx: &VulkanContext, retire: &RetireQueue) {
        self.flush_retired_views(ctx, retire);
        for (res_id, tgt) in self.internal_targets.drain() {
            self.bound_images.remove(&res_id);
            self.resource_images.remove(&res_id);
            self.persistent_layouts.remove(&res_id);
            self.tracked_image_layouts.remove(&tgt.image);
            let dev = ctx.device.clone();
            retire.retire_later(move || unsafe {
                tgt.destroy(&dev);
            });
        }
        for (res_id, mut buf_alloc) in self.custom_buffer_allocs.drain() {
            self.bound_buffers.remove(&res_id);
            let dev = ctx.device.clone();
            retire.retire_later(move || unsafe {
                buf_alloc.destroy(&dev);
            });
        }
        self.internal_render_size = (0, 0);
        self.internal_target_size = (0, 0);
    }

    /// Invalidates resize-sensitive internal render targets and temporal history on display resize.
    pub fn invalidate_for_resize(&mut self, ctx: &VulkanContext, retire: &RetireQueue) {
        self.destroy_internal_targets(ctx, retire);
        self.reset_temporal_history();
    }

    /// Safely retires optional non-essential transient, reservoir, and temporal history targets
    /// under [`crate::gpu::VramPressureLevel::Critical`] and resets temporal layout state.
    ///
    /// Returns the number of internal targets retired.
    pub fn trim_under_memory_pressure(
        &mut self,
        ctx: &VulkanContext,
        retire_queue: &RetireQueue,
    ) -> usize {
        self.flush_retired_views(ctx, retire_queue);
        let mut trim_ids: Vec<ResourceId> = vec![
            Self::RES_SSGI_HISTORY,
            Self::RES_SSGI_OUT,
            Self::RES_SVGF_PING,
            Self::RES_SVGF_HISTORY_RADIANCE,
            Self::RES_SVGF_HISTORY_MOMENTS,
            Self::RES_SVGF_MOMENTS,
            Self::RES_TSR_HISTORY,
            Self::RES_PREV_HUDLESS_COLOR,
            Self::RES_INTERPOLATED_FRAME,
            Self::RES_VOLUMETRIC_FOG,
            Self::RES_VOLUMETRIC_FOG_HISTORY,
            Self::RES_NRD_IN_DIFF,
            Self::RES_NRD_IN_SPEC,
            Self::RES_NRD_OUT_SPEC,
            Self::RES_NRD_NORMAL_ROUGH,
            Self::RES_NRD_VIEWZ,
            Self::RES_RT_HIT_DIST,
            Self::RES_RESTIR_RESERVOIRS,
        ];
        trim_ids.extend(self.temporal_custom_resources.iter().copied());

        let mut trimmed = 0usize;
        for res_id in trim_ids {
            if let Some(tgt) = self.internal_targets.remove(&res_id) {
                self.bound_images.remove(&res_id);
                self.resource_images.remove(&res_id);
                self.persistent_layouts.remove(&res_id);
                self.tracked_image_layouts.remove(&tgt.image);
                let dev = ctx.device.clone();
                retire_queue.retire_later(move || unsafe {
                    tgt.destroy(&dev);
                });
                trimmed += 1;
            }
        }
        if trimmed > 0 {
            self.internal_render_size = (0, 0);
            self.internal_target_size = (0, 0);
        }
        self.reset_temporal_history();
        trimmed
    }

    pub fn ensure_internal_targets(
        &mut self,
        ctx: &VulkanContext,
        render_w: u32,
        render_h: u32,
        out_w: u32,
        out_h: u32,
        retire: &RetireQueue,
    ) {
        self.flush_retired_views(ctx, retire);
        let rw = render_w.max(1);
        let rh = render_h.max(1);
        let ow = out_w.max(1);
        let oh = out_h.max(1);
        let desired_shadow_res_check = self
            .shadow_resolution
            .clamp(512, ctx.capabilities.max_shadow_resolution());
        let shadow_changed_early = self
            .internal_targets
            .get(&Self::RES_SHADOW_CSM)
            .is_some_and(|t| {
                t.width != desired_shadow_res_check || t.height != desired_shadow_res_check
            });
        if shadow_changed_early {
            self.internal_render_size = (0, 0);
            self.internal_target_size = (0, 0);
        }
        let cfg = self.config;
        let restir_gi_target = self
            .custom_targets
            .values()
            .find(|target| target.name == "restir_gi_reservoir");
        let restir_gi_allocation_mismatch = restir_gi_target.is_some_and(|target| {
            let target_allocated = self.internal_targets.contains_key(&target.id);
            let history_allocated = target
                .history_id
                .is_some_and(|history| self.internal_targets.contains_key(&history));
            if cfg.tier == RenderTier::Tier3PathTracing {
                !target_allocated || !history_allocated
            } else {
                target_allocated || history_allocated
            }
        });
        if self.internal_render_size == (rw, rh)
            && self.internal_target_size == (ow, oh)
            && !self.internal_targets.is_empty()
            && !restir_gi_allocation_mismatch
        {
            return;
        }
        if ctx.device.handle() == vk::Device::null()
            || ctx.physical_device == vk::PhysicalDevice::null()
        {
            return;
        }
        self.destroy_internal_targets(ctx, retire);

        let rgba16 = vk::Format::R16G16B16A16_SFLOAT;
        let mut specs: Vec<(ResourceId, vk::Format, u32, u32, u32)> = vec![
            // Pre-upscale targets at (render_w, render_h)
            (Self::RES_RAW_RADIANCE, rgba16, rw, rh, 1),
            (Self::RES_SHARP_RADIANCE, rgba16, rw, rh, 1),
            (Self::RES_DENOISED_RADIANCE, rgba16, rw, rh, 1),
            (Self::RES_SVGF_MOMENTS, rgba16, rw, rh, 1),
            (
                Self::RES_VOLUMETRIC_FOG,
                rgba16,
                (rw / 8).max(1),
                (rh / 8).max(1),
                16,
            ),
            // Post-upscale targets at (out_w, out_h)
            (Self::RES_UPSCALED_HDR, rgba16, ow, oh, 1),
            (Self::RES_POST_OUTPUT, vk::Format::R8G8B8A8_UNORM, ow, oh, 1),
        ];
        // Allocate tier- and feature-specific targets only when enabled in GraphConfig.
        if cfg.tier == RenderTier::Tier3PathTracing {
            specs.push((Self::RES_LINEAR_DEPTH, vk::Format::R32_SFLOAT, rw, rh, 1));
            specs.push((Self::RES_PT_ALBEDO, rgba16, rw, rh, 1));
        }
        let half_w = (rw / 2).max(1);
        let half_h = (rh / 2).max(1);
        if matches!(
            cfg.tier,
            RenderTier::Tier0Raster | RenderTier::Tier1ComputeGi
        ) {
            specs.push((Self::RES_SSGI_OUT, rgba16, half_w, half_h, 1));
        }
        if cfg.tier == RenderTier::Tier1ComputeGi {
            specs.push((Self::RES_SSGI_HISTORY, rgba16, half_w, half_h, 1));
        }
        if cfg.tier >= RenderTier::Tier2RayQuery || cfg.effective_denoiser_active() {
            specs.push((Self::RES_SVGF_PING, rgba16, rw, rh, 1));
            specs.push((Self::RES_SVGF_HISTORY_RADIANCE, rgba16, rw, rh, 1));
            specs.push((Self::RES_SVGF_HISTORY_MOMENTS, rgba16, rw, rh, 1));
        }
        if cfg.effective_denoiser_active() && (cfg.denoiser_mode == 2 || cfg.denoiser_mode == 3) {
            specs.push((Self::RES_NRD_IN_DIFF, rgba16, rw, rh, 1));
            specs.push((Self::RES_NRD_IN_SPEC, rgba16, rw, rh, 1));
            specs.push((Self::RES_NRD_OUT_SPEC, rgba16, rw, rh, 1));
            specs.push((Self::RES_NRD_NORMAL_ROUGH, rgba16, rw, rh, 1));
            specs.push((Self::RES_NRD_VIEWZ, vk::Format::R32_SFLOAT, rw, rh, 1));
            specs.push((Self::RES_RT_HIT_DIST, vk::Format::R16_SFLOAT, rw, rh, 1));
        }
        if cfg.upscaler_mode != 0 {
            specs.push((Self::RES_TSR_HISTORY, rgba16, ow, oh, 1));
        }
        if cfg.tier == RenderTier::Tier1ComputeGi || cfg.framegen_mode != 0 {
            specs.push((
                Self::RES_PREV_HUDLESS_COLOR,
                vk::Format::R8G8B8A8_UNORM,
                ow,
                oh,
                1,
            ));
        }
        if cfg.framegen_mode != 0 {
            specs.push((Self::RES_INTERPOLATED_FRAME, rgba16, ow, oh, 1));
        }
        if cfg.hdr_enabled || cfg.framegen_mode != 0 {
            specs.push((Self::RES_SWAPCHAIN_OUT, rgba16, ow, oh, 1));
        }

        for tgt in self.custom_targets.values() {
            if tgt.name == "restir_gi_reservoir" && cfg.tier != RenderTier::Tier3PathTracing {
                continue;
            }
            let (base_w, base_h) =
                if tgt.id == Self::RES_VOLUMETRIC_FOG || tgt.name == "restir_gi_reservoir" {
                    (rw, rh)
                } else {
                    (ow, oh)
                };
            let (scaled_w, scaled_h) = if tgt.name == "restir_gi_reservoir" {
                (base_w.div_ceil(4), base_h.div_ceil(4))
            } else {
                (
                    ((base_w as f32) * tgt.scale).round().max(1.0) as u32,
                    ((base_h as f32) * tgt.scale).round().max(1.0) as u32,
                )
            };
            if let Some(pos) = specs.iter().position(|(id, ..)| *id == tgt.id) {
                specs[pos] = (tgt.id, tgt.format, scaled_w, scaled_h, tgt.array_layers);
            } else {
                specs.push((tgt.id, tgt.format, scaled_w, scaled_h, tgt.array_layers));
            }
            if let Some(hid) = tgt.history_id
                && !specs.iter().any(|(id, ..)| *id == hid)
            {
                specs.push((hid, tgt.format, scaled_w, scaled_h, tgt.array_layers));
            }
        }

        for (res_id, format, tw, th, layers) in specs {
            let mip_levels = self
                .custom_targets
                .get(&res_id)
                .map(|t| t.mip_levels.max(1))
                .unwrap_or(1);
            if self.bound_images.contains_key(&res_id) {
                continue;
            }
            let aspect = if format == vk::Format::D32_SFLOAT {
                vk::ImageAspectFlags::DEPTH
            } else {
                vk::ImageAspectFlags::COLOR
            };
            let usage = if aspect == vk::ImageAspectFlags::DEPTH {
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
            } else {
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
            };
            let img_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: tw,
                    height: th,
                    depth: 1,
                })
                .mip_levels(mip_levels)
                .array_layers(layers.max(1))
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let view_type = if layers > 1 {
                vk::ImageViewType::TYPE_2D_ARRAY
            } else {
                vk::ImageViewType::TYPE_2D
            };
            let Some(target) = allocate_internal_image_target(
                ctx,
                &img_info,
                view_type,
                aspect,
                "graph_internal_target",
            ) else {
                continue;
            };
            let image = target.image;
            let view = target.view;

            self.internal_targets.insert(res_id, target);
            self.resource_images.insert(res_id, image);
            self.tracked_image_layouts
                .insert(image, vk::ImageLayout::UNDEFINED);
            self.bound_images.insert(
                res_id,
                BoundImageResource {
                    image,
                    view,
                    sampler: vk::Sampler::null(),
                    format,
                    aspect,
                    array_layers: layers.max(1),
                    owned_view: false,
                },
            );
        }

        // Allocate custom pack storage buffers (`[buffers.<name>]`)
        let custom_buf_specs: Vec<CustomBufferSpec> =
            self.custom_buffers.values().cloned().collect();
        for buf_spec in custom_buf_specs {
            if self.bound_buffers.contains_key(&buf_spec.id) {
                continue;
            }
            if let Ok(alloc) = GpuBufferAllocation::allocate_device_local_bda(
                ctx,
                buf_spec.size_bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC,
            ) {
                self.bound_buffers.insert(
                    buf_spec.id,
                    BoundBufferResource {
                        buffer: alloc.buffer,
                        offset: 0,
                        size: alloc.size,
                    },
                );
                self.custom_buffer_allocs.insert(buf_spec.id, alloc);
            }
        }

        // Cascaded shadow map: shadow_resolution x shadow_resolution x 4 D32_SFLOAT 2D array.
        let desired_shadow_res = self
            .shadow_resolution
            .clamp(512, ctx.capabilities.max_shadow_resolution());
        let shadow_res_changed = self
            .internal_targets
            .get(&Self::RES_SHADOW_CSM)
            .is_some_and(|t| t.width != desired_shadow_res || t.height != desired_shadow_res);
        if shadow_res_changed && let Some(old) = self.internal_targets.remove(&Self::RES_SHADOW_CSM)
        {
            self.bound_images.remove(&Self::RES_SHADOW_CSM);
            self.resource_images.remove(&Self::RES_SHADOW_CSM);
            self.tracked_image_layouts.remove(&old.image);
            let dev = ctx.device.clone();
            retire.retire_later(move || unsafe {
                old.destroy(&dev);
            });
        }
        if !self.bound_images.contains_key(&Self::RES_SHADOW_CSM) {
            self.create_shadow_csm_target(ctx);
        }
        self.internal_render_size = (rw, rh);
        self.internal_target_size = (ow, oh);
    }

    /// Creates the four-layer `D32_SFLOAT` cascaded shadow-map array and exposes it as
    /// `RES_SHADOW_CSM` so `compute_lighting.comp` can sample all four cascades via `sampler2DArray`.
    fn create_shadow_csm_target(&mut self, ctx: &VulkanContext) {
        let shadow_res = self
            .shadow_resolution
            .clamp(512, ctx.capabilities.max_shadow_resolution());
        let format = vk::Format::D32_SFLOAT;
        let extent = vk::Extent3D {
            width: shadow_res,
            height: shadow_res,
            depth: 1,
        };
        let img_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(4)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let Some(target) = allocate_internal_image_target(
            ctx,
            &img_info,
            vk::ImageViewType::TYPE_2D_ARRAY,
            vk::ImageAspectFlags::DEPTH,
            "graph_shadow_csm",
        ) else {
            return;
        };
        let image = target.image;
        let view = target.view;
        self.internal_targets.insert(Self::RES_SHADOW_CSM, target);
        self.resource_images.insert(Self::RES_SHADOW_CSM, image);
        self.bound_images.insert(
            Self::RES_SHADOW_CSM,
            BoundImageResource {
                image,
                view,
                sampler: vk::Sampler::null(),
                format,
                aspect: vk::ImageAspectFlags::DEPTH,
                array_layers: 4,
                owned_view: false,
            },
        );
    }

    /// Dispatches a named compute shader (`"denoiser_prepare"`, `"denoiser_recompose"`, etc.)
    /// directly from `GpuPipelineManager::compute_pipelines` when compiled by the active pack,
    /// inserting exact `VK_KHR_synchronization2` image layout transitions and updating
    /// `live_image_states` + `self.tracked_image_layouts`.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_named_compute_helper(
        &mut self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        pipelines: &mut GpuPipelineManager,
        shader_keys: &[&str],
        explicit_bindings: &[(u32, ResourceId)],
        push_constants: Option<&[u8]>,
        bindless: Option<&BindlessTextureTable>,
        pack_set: vk::DescriptorSet,
        frame_slot: usize,
        width: u32,
        height: u32,
        live_image_states: &mut HashMap<
            vk::Image,
            (
                vk::PipelineStageFlags2,
                vk::AccessFlags2,
                vk::ImageLayout,
                vk::ImageAspectFlags,
            ),
        >,
    ) -> bool {
        let slot = frame_slot % 3;
        let ubo_buf = pipelines.world_ubo_buffer_for_slot(slot);
        let nearest_sampler = pipelines.nearest_sampler;
        let default_sampler = pipelines.default_sampler;
        let Some((matched_key, bundle)) = shader_keys
            .iter()
            .find_map(|&k| pipelines.compute_pipelines.get(k).map(|b| (k, b)))
        else {
            return false;
        };
        if bundle.pipeline == vk::Pipeline::null() {
            return false;
        }
        let set0 = bundle.descriptor_sets_set0[slot];
        let set1 = bundle.descriptor_sets_set1[slot];
        if set0 == vk::DescriptorSet::null() {
            return false;
        }

        let mut barriers: Vec<vk::ImageMemoryBarrier2> = Vec::new();
        let mut sig = crate::modules::fnv1a_init();
        sig = crate::modules::fnv1a_mix_u64(sig, set0.as_raw());

        // Pass 1: emit image layout barriers and compute the 64-bit FNV-1a descriptor signature.
        for b in &bundle.reflection.bindings {
            if b.set != 0 {
                continue;
            }
            match b.descriptor_type {
                ReflectedDescriptorType::UniformBuffer => {
                    if ubo_buf == vk::Buffer::null() {
                        continue;
                    }
                    sig = crate::modules::fnv1a_mix_u64(sig, u64::from(b.binding));
                    sig = crate::modules::fnv1a_mix_u64(sig, ubo_buf.as_raw());
                    sig = crate::modules::fnv1a_mix_u64(sig, 0);
                    sig = crate::modules::fnv1a_mix_u64(sig, vk::WHOLE_SIZE);
                }
                ReflectedDescriptorType::StorageBuffer => {
                    let res_id = GpuPipelineManager::resolve_resource_for_binding_named(
                        PassKind::DenoiserSvgfOrNrd,
                        b.binding,
                        &b.name,
                        b.writable,
                        self,
                    );
                    let buf = self
                        .bound_buffers
                        .get(&res_id)
                        .map(|bb| bb.buffer)
                        .unwrap_or(ubo_buf);
                    if buf == vk::Buffer::null() {
                        continue;
                    }
                    sig = crate::modules::fnv1a_mix_u64(sig, u64::from(b.binding));
                    sig = crate::modules::fnv1a_mix_u64(sig, buf.as_raw());
                    sig = crate::modules::fnv1a_mix_u64(sig, 0);
                    sig = crate::modules::fnv1a_mix_u64(sig, vk::WHOLE_SIZE);
                }
                ReflectedDescriptorType::CombinedImageSampler
                | ReflectedDescriptorType::SampledImage
                | ReflectedDescriptorType::StorageImage => {
                    let is_storage = b.descriptor_type == ReflectedDescriptorType::StorageImage;
                    let res_id = explicit_bindings
                        .iter()
                        .find(|&&(bnd, _)| bnd == b.binding)
                        .map(|&(_, id)| id)
                        .unwrap_or_else(|| {
                            GpuPipelineManager::resolve_resource_for_binding_named(
                                PassKind::DenoiserSvgfOrNrd,
                                b.binding,
                                &b.name,
                                is_storage || b.writable,
                                self,
                            )
                        });
                    let Some(&bound) = self.bound_images.get(&res_id) else {
                        continue;
                    };
                    if bound.image == vk::Image::null() || bound.view == vk::ImageView::null() {
                        continue;
                    }
                    let (dst_access, new_layout) = if is_storage {
                        (
                            vk::AccessFlags2::SHADER_STORAGE_READ
                                | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                            vk::ImageLayout::GENERAL,
                        )
                    } else {
                        (
                            vk::AccessFlags2::SHADER_SAMPLED_READ,
                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        )
                    };
                    let (src_stage, src_access, old_layout, aspect) =
                        live_image_states.get(&bound.image).copied().unwrap_or((
                            vk::PipelineStageFlags2::TOP_OF_PIPE,
                            vk::AccessFlags2::NONE,
                            self.tracked_image_layouts
                                .get(&bound.image)
                                .copied()
                                .unwrap_or(vk::ImageLayout::UNDEFINED),
                            bound.aspect,
                        ));
                    let write_mask = vk::AccessFlags2::SHADER_WRITE.as_raw()
                        | vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw()
                        | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw()
                        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw()
                        | vk::AccessFlags2::TRANSFER_WRITE.as_raw()
                        | vk::AccessFlags2::MEMORY_WRITE.as_raw();
                    if old_layout != new_layout
                        || (src_access.as_raw() & write_mask) != 0
                        || is_storage
                    {
                        barriers.push(
                            vk::ImageMemoryBarrier2::default()
                                .src_stage_mask(if src_stage.is_empty() {
                                    vk::PipelineStageFlags2::ALL_COMMANDS
                                } else {
                                    src_stage
                                })
                                .src_access_mask(src_access)
                                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                                .dst_access_mask(dst_access)
                                .old_layout(old_layout)
                                .new_layout(new_layout)
                                .image(bound.image)
                                .subresource_range(vk::ImageSubresourceRange {
                                    aspect_mask: aspect,
                                    base_mip_level: 0,
                                    level_count: 1,
                                    base_array_layer: 0,
                                    layer_count: bound.array_layers.max(1),
                                }),
                        );
                    }
                    live_image_states.insert(
                        bound.image,
                        (
                            vk::PipelineStageFlags2::COMPUTE_SHADER,
                            dst_access,
                            new_layout,
                            aspect,
                        ),
                    );
                    self.tracked_image_layouts.insert(bound.image, new_layout);

                    let sampler = if is_storage {
                        vk::Sampler::null()
                    } else if matches!(
                        res_id,
                        Self::RES_MC_MAIN_DEPTH
                            | Self::RES_LINEAR_DEPTH
                            | Self::RES_NRD_VIEWZ
                            | Self::RES_RT_HIT_DIST
                            | Self::RES_GBUFFER_NORMAL
                            | Self::RES_NRD_NORMAL_ROUGH
                            | Self::RES_GBUFFER_MATERIAL
                            | Self::RES_GBUFFER_VELOCITY
                    ) {
                        nearest_sampler
                    } else {
                        default_sampler
                    };
                    sig = crate::modules::fnv1a_mix_u64(sig, u64::from(b.binding));
                    sig = crate::modules::fnv1a_mix_u64(sig, bound.view.as_raw());
                    sig = crate::modules::fnv1a_mix_u64(sig, sampler.as_raw());
                    sig = crate::modules::fnv1a_mix_u64(sig, new_layout.as_raw() as u64);
                }
                _ => {}
            }
        }

        // Pass 2: only allocate and submit descriptor writes when the handle signature changed.
        let cache_key = (matched_key.to_owned(), slot);
        if pipelines.descriptor_sig_cache.get(&cache_key).copied() != Some(sig) {
            let cap = bundle.reflection.bindings.len();
            let mut buf_infos: Vec<[vk::DescriptorBufferInfo; 1]> = Vec::with_capacity(cap);
            let mut img_infos: Vec<[vk::DescriptorImageInfo; 1]> = Vec::with_capacity(cap);
            let mut pending_writes: Vec<(vk::WriteDescriptorSet, Option<usize>, Option<usize>)> =
                Vec::with_capacity(cap);

            for b in &bundle.reflection.bindings {
                if b.set != 0 {
                    continue;
                }
                match b.descriptor_type {
                    ReflectedDescriptorType::UniformBuffer => {
                        if ubo_buf == vk::Buffer::null() {
                            continue;
                        }
                        let idx = buf_infos.len();
                        buf_infos.push([vk::DescriptorBufferInfo {
                            buffer: ubo_buf,
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        }]);
                        let write = vk::WriteDescriptorSet::default()
                            .dst_set(set0)
                            .dst_binding(b.binding)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER);
                        pending_writes.push((write, Some(idx), None));
                    }
                    ReflectedDescriptorType::StorageBuffer => {
                        let res_id = GpuPipelineManager::resolve_resource_for_binding_named(
                            PassKind::DenoiserSvgfOrNrd,
                            b.binding,
                            &b.name,
                            b.writable,
                            self,
                        );
                        let buf = self
                            .bound_buffers
                            .get(&res_id)
                            .map(|bb| bb.buffer)
                            .unwrap_or(ubo_buf);
                        if buf == vk::Buffer::null() {
                            continue;
                        }
                        let idx = buf_infos.len();
                        buf_infos.push([vk::DescriptorBufferInfo {
                            buffer: buf,
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        }]);
                        let write = vk::WriteDescriptorSet::default()
                            .dst_set(set0)
                            .dst_binding(b.binding)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER);
                        pending_writes.push((write, Some(idx), None));
                    }
                    ReflectedDescriptorType::CombinedImageSampler
                    | ReflectedDescriptorType::SampledImage
                    | ReflectedDescriptorType::StorageImage => {
                        let is_storage = b.descriptor_type == ReflectedDescriptorType::StorageImage;
                        let res_id = explicit_bindings
                            .iter()
                            .find(|&&(bnd, _)| bnd == b.binding)
                            .map(|&(_, id)| id)
                            .unwrap_or_else(|| {
                                GpuPipelineManager::resolve_resource_for_binding_named(
                                    PassKind::DenoiserSvgfOrNrd,
                                    b.binding,
                                    &b.name,
                                    is_storage || b.writable,
                                    self,
                                )
                            });
                        let Some(&bound) = self.bound_images.get(&res_id) else {
                            continue;
                        };
                        if bound.image == vk::Image::null() || bound.view == vk::ImageView::null() {
                            continue;
                        }
                        let new_layout = if is_storage {
                            vk::ImageLayout::GENERAL
                        } else {
                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                        };
                        let sampler = if is_storage {
                            vk::Sampler::null()
                        } else if matches!(
                            res_id,
                            Self::RES_MC_MAIN_DEPTH
                                | Self::RES_LINEAR_DEPTH
                                | Self::RES_NRD_VIEWZ
                                | Self::RES_RT_HIT_DIST
                                | Self::RES_GBUFFER_NORMAL
                                | Self::RES_NRD_NORMAL_ROUGH
                                | Self::RES_GBUFFER_MATERIAL
                                | Self::RES_GBUFFER_VELOCITY
                        ) {
                            nearest_sampler
                        } else {
                            default_sampler
                        };
                        let idx = img_infos.len();
                        img_infos.push([vk::DescriptorImageInfo {
                            sampler,
                            image_view: bound.view,
                            image_layout: new_layout,
                        }]);
                        let desc_ty = match b.descriptor_type {
                            ReflectedDescriptorType::StorageImage => {
                                vk::DescriptorType::STORAGE_IMAGE
                            }
                            ReflectedDescriptorType::SampledImage => {
                                vk::DescriptorType::SAMPLED_IMAGE
                            }
                            _ => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        };
                        let write = vk::WriteDescriptorSet::default()
                            .dst_set(set0)
                            .dst_binding(b.binding)
                            .descriptor_type(desc_ty);
                        pending_writes.push((write, None, Some(idx)));
                    }
                    _ => {}
                }
            }

            if !pending_writes.is_empty() {
                let writes: Vec<vk::WriteDescriptorSet> = pending_writes
                    .into_iter()
                    .map(|(mut w, bi, ii)| {
                        if let Some(i) = bi {
                            w = w.buffer_info(&buf_infos[i]);
                        }
                        if let Some(i) = ii {
                            w = w.image_info(&img_infos[i]);
                        }
                        w
                    })
                    .collect();
                unsafe {
                    ctx.device.update_descriptor_sets(&writes, &[]);
                }
            }
            pipelines.descriptor_sig_cache.insert(cache_key, sig);
        }

        let wg_x = bundle.workgroup_size[0].max(1);
        let wg_y = bundle.workgroup_size[1].max(1);
        let groups_x = width.max(1).div_ceil(wg_x);
        let groups_y = height.max(1).div_ceil(wg_y);

        unsafe {
            if !barriers.is_empty() {
                let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                ctx.sync2_ext.cmd_pipeline_barrier2(cmd, &dep);
            }
            ctx.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, bundle.pipeline);
            let sets = [
                set0,
                set1,
                bindless
                    .map(|b| b.descriptor_set)
                    .unwrap_or(vk::DescriptorSet::null()),
                pack_set,
            ];
            let mut idx = 0usize;
            while idx < sets.len() {
                if sets[idx] == vk::DescriptorSet::null() {
                    idx += 1;
                    continue;
                }
                let start = idx;
                while idx < sets.len() && sets[idx] != vk::DescriptorSet::null() {
                    idx += 1;
                }
                ctx.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    bundle.pipeline_layout,
                    start as u32,
                    &sets[start..idx],
                    &[],
                );
            }
            if let Some(pc_bytes) = push_constants
                && !pc_bytes.is_empty()
                && bundle.push_constant_size > 0
            {
                let len = pc_bytes.len().min(bundle.push_constant_size as usize);
                ctx.device.cmd_push_constants(
                    cmd,
                    bundle.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &pc_bytes[..len],
                );
            }
            ctx.device.cmd_dispatch(cmd, groups_x, groups_y, 1);
        }
        true
    }

    /// Executes the compiled pre-GUI DAG steps onto `cmd` (invoked immediately after
    /// `LevelRenderer.renderLevel()` and BEFORE `GuiRenderer.render()` so `RES_HUDLESS_COLOR` and
    /// `RES_MC_MAIN_DEPTH` contain pure world geometry without HUD).
    /// Post-GUI composition (`HdrCompose` and `FrameGenInterpolate`) is deferred to
    /// `execute_after_gui_with_context` after `GuiRenderer.render()` draws the HUD into `RES_UI_COLOR`.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_with_context(
        &mut self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        pipelines: &mut GpuPipelineManager,
        bindless: Option<&BindlessTextureTable>,
        pack_set: vk::DescriptorSet,
        tlas_handle: vk::AccelerationStructureKHR,
        frame_slot: usize,
        width: u32,
        height: u32,
        blas: &ZeroCopyBlasManager,
        mut sdk: Option<&mut RuntimeSdkManager>,
        dispatch: &SdkDispatchParams,
        retire: &RetireQueue,
    ) -> usize {
        let out_w = if dispatch.out_width > 0 {
            dispatch.out_width
        } else {
            width
        }
        .max(1);
        let out_h = if dispatch.out_height > 0 {
            dispatch.out_height
        } else {
            height
        }
        .max(1);
        let rw = if dispatch.width > 0 {
            dispatch.width
        } else {
            out_w
        }
        .max(1);
        let rh = if dispatch.height > 0 {
            dispatch.height
        } else {
            out_h
        }
        .max(1);
        let prev_target_sizes = (self.internal_render_size, self.internal_target_size);
        self.ensure_internal_targets(ctx, rw, rh, out_w, out_h, retire);
        if prev_target_sizes != (self.internal_render_size, self.internal_target_size) {
            pipelines.descriptor_sig_cache.clear();
        }
        self.ensure_dh_resources(ctx, cmd, frame_slot);
        let mut live_image_states = std::mem::take(&mut self.frame_live_image_states);
        let mut mojang_owned_images = std::mem::take(&mut self.frame_mojang_owned_images);
        let mut init_clear_ids = std::mem::take(&mut self.frame_init_clear_ids);
        let mut exit_barriers = std::mem::take(&mut self.frame_exit_barriers);
        let mut seen_exit = std::mem::take(&mut self.frame_seen_exit_images);
        live_image_states.clear();
        mojang_owned_images.clear();
        init_clear_ids.clear();
        exit_barriers.clear();
        seen_exit.clear();

        for (&res_id, &bound) in &self.bound_images {
            if bound.image == vk::Image::null() {
                continue;
            }
            let is_private_compute_img = self.internal_targets.contains_key(&res_id)
                || res_id == Self::RES_SHADOW_CSM
                || res_id == Self::RES_RAW_RADIANCE
                || res_id == Self::RES_INTERPOLATED_FRAME;
            let init_layout = if is_private_compute_img {
                self.tracked_image_layouts
                    .get(&bound.image)
                    .copied()
                    .unwrap_or(vk::ImageLayout::UNDEFINED)
            } else {
                mojang_owned_images.push((bound.image, bound.aspect));
                vk::ImageLayout::GENERAL
            };
            let (init_stage, init_access) = if init_layout == vk::ImageLayout::UNDEFINED {
                (vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::NONE)
            } else {
                (
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
                        | vk::PipelineStageFlags2::COMPUTE_SHADER
                        | vk::PipelineStageFlags2::ALL_TRANSFER,
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                        | vk::AccessFlags2::SHADER_WRITE
                        | vk::AccessFlags2::TRANSFER_WRITE,
                )
            };
            live_image_states.entry(bound.image).or_insert((
                init_stage,
                init_access,
                init_layout,
                bound.aspect,
            ));
        }

        // Clear uninitialized temporal history images, Tier 0 RES_SSGI_OUT (no producer on Tier 0),
        // and custom targets with clear_color so no shader ever samples undefined VRAM.
        init_clear_ids.extend([
            (
                Self::RES_SVGF_MOMENTS,
                [0.0; 4],
                !self.config.effective_denoiser_active(),
            ),
            (Self::RES_SVGF_HISTORY_MOMENTS, [0.0; 4], false),
            (Self::RES_SVGF_HISTORY_RADIANCE, [0.0; 4], false),
            (Self::RES_SSGI_HISTORY, [0.0; 4], false),
            (Self::RES_TSR_HISTORY, [0.0; 4], false),
            (Self::RES_PREV_HUDLESS_COLOR, [0.0; 4], false),
        ]);
        if self.config.tier == RenderTier::Tier0Raster {
            init_clear_ids.push((Self::RES_SSGI_OUT, [0.0; 4], false));
        }
        for tgt in self.custom_targets.values() {
            if let Some(cc) = tgt.clear_color {
                init_clear_ids.push((tgt.id, cc, true));
            }
            if let Some(hid) = tgt.history_id {
                init_clear_ids.push((hid, [0.0; 4], false));
            }
        }
        for (res_id, clear_rgba, every_frame) in init_clear_ids.drain(..) {
            let Some(&image) = self.resource_images.get(&res_id) else {
                continue;
            };
            if image == vk::Image::null() {
                continue;
            }
            let cur_state = live_image_states.get(&image).copied();
            let old_layout = cur_state.map(|s| s.2).unwrap_or(vk::ImageLayout::UNDEFINED);
            if !every_frame && old_layout != vk::ImageLayout::UNDEFINED {
                continue;
            }
            let src_stage = cur_state
                .map(|s| s.0)
                .unwrap_or(vk::PipelineStageFlags2::TOP_OF_PIPE);
            let src_access = cur_state.map(|s| s.1).unwrap_or(vk::AccessFlags2::NONE);
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(
                    self.bound_images
                        .get(&res_id)
                        .map(|bound| bound.array_layers.max(1))
                        .unwrap_or(1),
                );
            let barrier = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(src_stage)
                .src_access_mask(src_access)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY | vk::PipelineStageFlags2::CLEAR)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(old_layout)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(image)
                .subresource_range(range)];
            unsafe {
                ctx.sync2_ext.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&barrier),
                );
                ctx.device.cmd_clear_color_image(
                    cmd,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue {
                        float32: clear_rgba,
                    },
                    &[range],
                );
            }
            live_image_states.insert(
                image,
                (
                    vk::PipelineStageFlags2::CLEAR,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageAspectFlags::COLOR,
                ),
            );
            self.tracked_image_layouts
                .insert(image, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        }
        // Clear custom storage buffers that requested clear_each_frame = true
        for spec in self.custom_buffers.values() {
            if spec.clear_each_frame
                && let Some(resource) = self.bound_buffers.get(&spec.id)
                && resource.buffer != vk::Buffer::null()
            {
                unsafe {
                    ctx.device.cmd_fill_buffer(
                        cmd,
                        resource.buffer,
                        resource.offset,
                        resource.size,
                        0,
                    );
                }
            }
        }

        let mut executed_steps = 0usize;
        let mut dlss_rr_upscaled_this_frame = false;

        let direct_hudless_post = self.config.post_writes_directly_to_hudless();
        for step_idx in 0..self.compiled_steps.len() {
            let step_kind = self.compiled_steps[step_idx].kind;
            let is_external_raster = self.compiled_steps[step_idx].is_external_raster;
            // HdrCompose and FrameGenInterpolate require the rendered GUI (RES_UI_COLOR) and
            // therefore execute in `execute_after_gui_with_context` after `GuiRenderer.render()`.
            if matches!(
                step_kind,
                PassKind::HdrCompose | PassKind::FrameGenInterpolate
            ) {
                executed_steps += 1;
                continue;
            }
            if !is_external_raster && step_kind != PassKind::BlasTlasUpdate {
                let mut img_barriers = Vec::new();
                let mut buf_barriers = Vec::new();
                let mut mem_barriers = Vec::new();

                for &(raw_res_id, access) in self.compiled_steps[step_idx]
                    .reads
                    .iter()
                    .chain(self.compiled_steps[step_idx].writes.iter())
                {
                    if raw_res_id == Self::RES_MC_MAIN_DEPTH
                        && access == ResourceAccess::DepthAttachmentWrite
                        && step_kind == PassKind::PostGradeAndToneMap
                    {
                        continue;
                    }
                    let res_id = if step_kind == PassKind::PostGradeAndToneMap
                        && raw_res_id == Self::RES_HUDLESS_COLOR
                        && access == ResourceAccess::StorageImageWrite
                        && !direct_hudless_post
                    {
                        Self::RES_POST_OUTPUT
                    } else {
                        raw_res_id
                    };
                    if res_id == Self::RES_TLAS {
                        let (dst_stage, dst_access, _) = access.to_sync2_masks();
                        mem_barriers.push(
                            vk::MemoryBarrier2::default()
                                .src_stage_mask(
                                    vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                                )
                                .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
                                .dst_stage_mask(dst_stage)
                                .dst_access_mask(dst_access),
                        );
                    } else if matches!(
                        res_id,
                        Self::RES_SHARC_HASH_TABLE | Self::RES_RESTIR_RESERVOIRS
                    ) {
                        let (mut dst_stage, dst_access, _) = access.to_sync2_masks();
                        if !ctx.capabilities.ray_tracing_pipeline {
                            dst_stage &= !vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR;
                        }
                        if dst_stage.is_empty() {
                            dst_stage = vk::PipelineStageFlags2::ALL_COMMANDS;
                        }
                        mem_barriers.push(
                            vk::MemoryBarrier2::default()
                                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                                .dst_stage_mask(dst_stage)
                                .dst_access_mask(dst_access),
                        );
                    } else if let Some(&buf_res) = self.bound_buffers.get(&res_id) {
                        let (dst_stage, dst_access, _) = access.to_sync2_masks();
                        buf_barriers.push(
                            vk::BufferMemoryBarrier2::default()
                                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                                .dst_stage_mask(dst_stage)
                                .dst_access_mask(dst_access)
                                .buffer(buf_res.buffer)
                                .offset(buf_res.offset)
                                .size(buf_res.size),
                        );
                    } else if let Some(&img) = self.resource_images.get(&res_id) {
                        let (dst_stage, dst_access, new_layout) = access.to_sync2_masks();
                        if new_layout == vk::ImageLayout::UNDEFINED {
                            continue;
                        }
                        let default_aspect = Self::default_format_and_aspect(res_id).1;
                        let layer_count = self
                            .bound_images
                            .get(&res_id)
                            .map(|b| b.array_layers.max(1))
                            .unwrap_or(if res_id == Self::RES_SHADOW_CSM { 4 } else { 1 });
                        let (src_stage, src_access, old_layout, aspect) =
                            live_image_states.get(&img).copied().unwrap_or((
                                vk::PipelineStageFlags2::TOP_OF_PIPE,
                                vk::AccessFlags2::NONE,
                                vk::ImageLayout::UNDEFINED,
                                default_aspect,
                            ));
                        let sanitize_stage =
                            |mut s: vk::PipelineStageFlags2| -> vk::PipelineStageFlags2 {
                                if !ctx.capabilities.ray_tracing_pipeline {
                                    s &= !vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR;
                                }
                                if !ctx.capabilities.acceleration_structure {
                                    s &= !vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR;
                                }
                                if s.is_empty() {
                                    vk::PipelineStageFlags2::ALL_COMMANDS
                                } else {
                                    s
                                }
                            };
                        let src_stage = sanitize_stage(src_stage);
                        let dst_stage = sanitize_stage(dst_stage);
                        let write_mask = vk::AccessFlags2::SHADER_WRITE.as_raw()
                            | vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw()
                            | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw()
                            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw()
                            | vk::AccessFlags2::TRANSFER_WRITE.as_raw()
                            | vk::AccessFlags2::MEMORY_WRITE.as_raw();
                        let need_barrier = old_layout != new_layout
                            || (src_access.as_raw() & write_mask) != 0
                            || (dst_access.as_raw() & write_mask) != 0;
                        if need_barrier {
                            img_barriers.push(
                                vk::ImageMemoryBarrier2::default()
                                    .src_stage_mask(src_stage)
                                    .src_access_mask(src_access)
                                    .dst_stage_mask(dst_stage)
                                    .dst_access_mask(dst_access)
                                    .old_layout(old_layout)
                                    .new_layout(new_layout)
                                    .image(img)
                                    .subresource_range(vk::ImageSubresourceRange {
                                        aspect_mask: aspect,
                                        base_mip_level: 0,
                                        level_count: 1,
                                        base_array_layer: 0,
                                        layer_count,
                                    }),
                            );
                        }
                        live_image_states.insert(img, (dst_stage, dst_access, new_layout, aspect));
                        self.tracked_image_layouts.insert(img, new_layout);
                        self.persistent_layouts
                            .insert(res_id, (dst_stage, dst_access, new_layout));
                    }
                }

                if !img_barriers.is_empty() || !buf_barriers.is_empty() || !mem_barriers.is_empty()
                {
                    let dep = vk::DependencyInfo::default()
                        .memory_barriers(&mem_barriers)
                        .buffer_memory_barriers(&buf_barriers)
                        .image_memory_barriers(&img_barriers);
                    unsafe {
                        ctx.sync2_ext.cmd_pipeline_barrier2(cmd, &dep);
                    }
                }
                match step_kind {
                    PassKind::ShadowCsm => {
                        pipelines.dispatch_csm_shadow(
                            ctx,
                            cmd,
                            self,
                            blas,
                            bindless,
                            pack_set,
                            tlas_handle,
                            frame_slot,
                            rw,
                            rh,
                            dispatch.sun_dir,
                        );
                    }
                    PassKind::ComputeVolumetricFog => {
                        let (vf_w, vf_h) = self
                            .internal_targets
                            .get(&Self::RES_VOLUMETRIC_FOG)
                            .map(|t| (t.width, t.height))
                            .unwrap_or(((rw / 2).max(1), (rh / 2).max(1)));
                        pipelines.dispatch_pass_with_graph(
                            ctx,
                            cmd,
                            step_kind,
                            self,
                            bindless,
                            pack_set,
                            tlas_handle,
                            frame_slot,
                            vf_w,
                            vf_h,
                            Some(blas),
                            Some(retire),
                        );
                        if let Some(target) = self.custom_targets.get(&Self::RES_VOLUMETRIC_FOG)
                            && target.temporal
                            && let Some(history_id) = target.history_id
                        {
                            pipelines.copy_custom_temporal_history(
                                ctx,
                                cmd,
                                self,
                                Self::RES_VOLUMETRIC_FOG,
                                history_id,
                                vf_w,
                                vf_h,
                            );
                            if let Some(&history_image) = self.resource_images.get(&history_id) {
                                self.tracked_image_layouts.insert(
                                    history_image,
                                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                );
                            }
                        }
                    }
                    PassKind::ComputeSsgi => {
                        let (ssgi_w, ssgi_h) = self
                            .internal_targets
                            .get(&Self::RES_SSGI_OUT)
                            .map(|t| (t.width, t.height))
                            .unwrap_or(((rw / 2).max(1), (rh / 2).max(1)));
                        pipelines.dispatch_pass_with_graph(
                            ctx,
                            cmd,
                            step_kind,
                            self,
                            bindless,
                            pack_set,
                            tlas_handle,
                            frame_slot,
                            ssgi_w,
                            ssgi_h,
                            Some(blas),
                            Some(retire),
                        );
                        pipelines.copy_ssgi_history(ctx, cmd, self, ssgi_w, ssgi_h);
                        if let Some(&hist_img) = self.resource_images.get(&Self::RES_SSGI_HISTORY) {
                            self.tracked_image_layouts
                                .insert(hist_img, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                        }
                    }
                    PassKind::ComputeFroxelGi | PassKind::RayQueryLighting => {
                        pipelines.dispatch_pass_with_graph(
                            ctx,
                            cmd,
                            step_kind,
                            self,
                            bindless,
                            pack_set,
                            tlas_handle,
                            frame_slot,
                            rw,
                            rh,
                            Some(blas),
                            Some(retire),
                        );
                    }
                    PassKind::FullPathTracing => {
                        pipelines.dispatch_pass_with_graph(
                            ctx,
                            cmd,
                            step_kind,
                            self,
                            bindless,
                            pack_set,
                            tlas_handle,
                            frame_slot,
                            rw,
                            rh,
                            Some(blas),
                            Some(retire),
                        );
                        if tlas_handle != vk::AccelerationStructureKHR::null()
                            && let Some(target) = self
                                .custom_targets
                                .values()
                                .find(|target| target.name == "restir_gi_reservoir")
                            && target.temporal
                            && let Some(history_id) = target.history_id
                            && let Some(internal) = self.internal_targets.get(&target.id)
                        {
                            pipelines.copy_custom_temporal_history(
                                ctx,
                                cmd,
                                self,
                                target.id,
                                history_id,
                                internal.width,
                                internal.height,
                            );
                            if let Some(&history_image) = self.resource_images.get(&history_id) {
                                self.tracked_image_layouts.insert(
                                    history_image,
                                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                );
                            }
                        }
                    }
                    PassKind::DepthReprojectOverlay => {
                        pipelines.dispatch_depth_blit(
                            ctx, cmd, self, bindless, pack_set, frame_slot, out_w, out_h,
                        );
                    }
                    PassKind::DenoiserSvgfOrNrd => {
                        if self.config.tier == RenderTier::Tier1ComputeGi {
                            executed_steps += 1;
                            continue;
                        }
                        let mut used_vendor = false;
                        let mut denoise_dispatch = *dispatch;
                        denoise_dispatch.width = rw;
                        denoise_dispatch.height = rh;
                        denoise_dispatch.out_width = out_w;
                        denoise_dispatch.out_height = out_h;
                        denoise_dispatch.is_tier3 =
                            self.config.tier == RenderTier::Tier3PathTracing;
                        if let Some(&den_img) =
                            self.resource_images.get(&Self::RES_DENOISED_RADIANCE)
                        {
                            denoise_dispatch.output_image = den_img;
                        }
                        if let Some(den_view) = self.resolve_image_view(Self::RES_DENOISED_RADIANCE)
                        {
                            denoise_dispatch.output_view = den_view;
                        }
                        let resolve_pair = |g: &Self, id: ResourceId| {
                            (
                                g.resource_images
                                    .get(&id)
                                    .copied()
                                    .unwrap_or(vk::Image::null()),
                                g.resolve_image_view(id).unwrap_or(vk::ImageView::null()),
                            )
                        };
                        denoise_dispatch.nrd_in_diff = resolve_pair(self, Self::RES_NRD_IN_DIFF);
                        denoise_dispatch.nrd_in_spec = resolve_pair(self, Self::RES_NRD_IN_SPEC);
                        denoise_dispatch.nrd_out_spec = resolve_pair(self, Self::RES_NRD_OUT_SPEC);
                        denoise_dispatch.nrd_normal_rough =
                            resolve_pair(self, Self::RES_NRD_NORMAL_ROUGH);
                        denoise_dispatch.nrd_viewz = resolve_pair(self, Self::RES_NRD_VIEWZ);
                        denoise_dispatch.rt_hit_dist = resolve_pair(self, Self::RES_RT_HIT_DIST);

                        let dlss_rr_target_id = if self.config.upscaler_mode != 0 {
                            Self::RES_UPSCALED_HDR
                        } else {
                            Self::RES_DENOISED_RADIANCE
                        };
                        let (rr_out_img, rr_out_view) = resolve_pair(self, dlss_rr_target_id);
                        denoise_dispatch.dlss_rr_output_image = rr_out_img;
                        denoise_dispatch.dlss_rr_output_view = rr_out_view;

                        let is_tier3 = self.config.tier == RenderTier::Tier3PathTracing;
                        let active_depth_id = if is_tier3 {
                            Self::RES_LINEAR_DEPTH
                        } else {
                            Self::RES_MC_MAIN_DEPTH
                        };
                        let active_albedo_id = if is_tier3 {
                            Self::RES_PT_ALBEDO
                        } else {
                            Self::RES_GBUFFER_ALBEDO
                        };
                        let prepare_bindings: [(u32, ResourceId); 11] = [
                            (1, Self::RES_RAW_RADIANCE),
                            (2, Self::RES_SHARP_RADIANCE),
                            (3, Self::RES_GBUFFER_NORMAL),
                            (4, active_depth_id),
                            (5, active_albedo_id),
                            (6, Self::RES_NRD_IN_DIFF),
                            (7, Self::RES_NRD_IN_SPEC),
                            (8, Self::RES_NRD_NORMAL_ROUGH),
                            (9, Self::RES_NRD_VIEWZ),
                            (10, Self::RES_RT_HIT_DIST),
                            (11, Self::RES_GBUFFER_MATERIAL),
                        ];
                        let nrd_settings = sdk.as_ref().and_then(|mgr| {
                            let lib = mgr.nrd_library()?.library_desc()?;
                            Some(
                                1 | ((lib.normal_encoding as u32) << 8)
                                    | ((lib.roughness_encoding as u32) << 16),
                            )
                        });
                        let prepare_flags = if dispatch.denoiser_mode == 2 {
                            nrd_settings.unwrap_or(0)
                        } else {
                            0
                        };
                        let prepare_pc: [u32; 4] = [u32::from(is_tier3), rw, rh, prepare_flags];
                        let prepare_pc_bytes: [u8; 16] = unsafe { std::mem::transmute(prepare_pc) };

                        if dispatch.denoiser_mode == 2 {
                            let nrd_ready = nrd_settings.is_some()
                                && self.config.tier >= RenderTier::Tier2RayQuery
                                && sdk.as_ref().is_some_and(|mgr| mgr.is_nrd_available())
                                && (pipelines
                                    .compute_pipelines
                                    .contains_key("denoiser_recompose.comp")
                                    || pipelines
                                        .compute_pipelines
                                        .contains_key("denoiser_recompose"));
                            let nrd_prepared = nrd_ready
                                && self.dispatch_named_compute_helper(
                                    ctx,
                                    cmd,
                                    pipelines,
                                    &["denoiser_prepare.comp", "denoiser_prepare"],
                                    &prepare_bindings,
                                    Some(&prepare_pc_bytes),
                                    bindless,
                                    pack_set,
                                    frame_slot,
                                    rw,
                                    rh,
                                    &mut live_image_states,
                                );
                            if nrd_prepared {
                                // When `denoiser_recompose` is compiled, route NRD's diffuse output
                                // into `RES_SVGF_PING` so `denoiser_recompose` reads `RES_SVGF_PING`
                                // (inDiff) and writes `RES_DENOISED_RADIANCE` (outDenoised) with
                                // zero image layout aliasing.
                                let nrd_diff_out_id = Self::RES_SVGF_PING;
                                let (diff_out_img, diff_out_view) =
                                    resolve_pair(self, nrd_diff_out_id);
                                if diff_out_img != vk::Image::null()
                                    && diff_out_view != vk::ImageView::null()
                                {
                                    denoise_dispatch.output_image = diff_out_img;
                                    denoise_dispatch.output_view = diff_out_view;
                                }

                                // Seed NRD with real live layouts for all user textures.
                                let mut nrd_layouts: HashMap<vk::Image, vk::ImageLayout> =
                                    HashMap::new();
                                for img in [
                                    denoise_dispatch.motion_image,
                                    denoise_dispatch.input_image,
                                    denoise_dispatch.output_image,
                                    denoise_dispatch.depth_image,
                                    denoise_dispatch.nrd_in_diff.0,
                                    denoise_dispatch.nrd_in_spec.0,
                                    denoise_dispatch.nrd_out_spec.0,
                                    denoise_dispatch.nrd_normal_rough.0,
                                    denoise_dispatch.nrd_viewz.0,
                                ] {
                                    if img != vk::Image::null() {
                                        let cur = live_image_states
                                            .get(&img)
                                            .map(|s| s.2)
                                            .or_else(|| {
                                                self.tracked_image_layouts.get(&img).copied()
                                            })
                                            .unwrap_or(vk::ImageLayout::UNDEFINED);
                                        nrd_layouts.insert(img, cur);
                                    }
                                }

                                if let Some(ref mut mgr) = sdk {
                                    unsafe {
                                        mgr.ensure_nrd_pools(
                                            &ctx.instance,
                                            ctx.physical_device,
                                            rw,
                                            rh,
                                            Some(retire),
                                        );
                                    }
                                    let res = unsafe {
                                        mgr.evaluate_nrd(&denoise_dispatch, &mut nrd_layouts)
                                    };
                                    used_vendor = res.is_ok();
                                    mgr.last_evaluate_result = Some(match &res {
                                        Ok(n) => format!("NRD: ok ({n} dispatches)"),
                                        Err(e) => format!("NRD: {e}"),
                                    });
                                }

                                // Sync final NRD layouts back into RenderGraph state maps.
                                for (img, final_layout) in nrd_layouts {
                                    let aspect = live_image_states
                                        .get(&img)
                                        .map(|s| s.3)
                                        .unwrap_or(vk::ImageAspectFlags::COLOR);
                                    let access = if final_layout == vk::ImageLayout::GENERAL {
                                        vk::AccessFlags2::SHADER_STORAGE_WRITE
                                            | vk::AccessFlags2::SHADER_STORAGE_READ
                                    } else {
                                        vk::AccessFlags2::SHADER_SAMPLED_READ
                                    };
                                    live_image_states.insert(
                                        img,
                                        (
                                            vk::PipelineStageFlags2::COMPUTE_SHADER,
                                            access,
                                            final_layout,
                                            aspect,
                                        ),
                                    );
                                    self.tracked_image_layouts.insert(img, final_layout);
                                }

                                if used_vendor {
                                    let recompose_bindings: [(u32, ResourceId); 7] = [
                                        (0, nrd_diff_out_id),
                                        (1, Self::RES_NRD_OUT_SPEC),
                                        (2, active_albedo_id),
                                        (3, Self::RES_DENOISED_RADIANCE),
                                        (4, Self::RES_SHARP_RADIANCE),
                                        (5, Self::RES_RAW_RADIANCE),
                                        (6, Self::RES_NRD_VIEWZ),
                                    ];
                                    let recompose_pc: [u32; 4] = [rw, rh, 0, 0];
                                    let recompose_pc_bytes: [u8; 16] =
                                        unsafe { std::mem::transmute(recompose_pc) };
                                    used_vendor = self.dispatch_named_compute_helper(
                                        ctx,
                                        cmd,
                                        pipelines,
                                        &["denoiser_recompose.comp", "denoiser_recompose"],
                                        &recompose_bindings,
                                        Some(&recompose_pc_bytes),
                                        bindless,
                                        pack_set,
                                        frame_slot,
                                        rw,
                                        rh,
                                        &mut live_image_states,
                                    );
                                }
                            }
                            if let Some(ref mut mgr) = sdk {
                                mgr.report_denoiser_status(if used_vendor {
                                    "NRD REBLUR active"
                                } else {
                                    "NRD guides or dispatch unavailable; using SVGF"
                                });
                            }
                        } else if dispatch.denoiser_mode == 3 {
                            let rr_ready = self.config.tier >= RenderTier::Tier2RayQuery
                                && sdk.as_ref().is_some_and(|mgr| mgr.is_dlss_rr_available());
                            let rr_prepared = rr_ready
                                && self.dispatch_named_compute_helper(
                                    ctx,
                                    cmd,
                                    pipelines,
                                    &["denoiser_prepare.comp", "denoiser_prepare"],
                                    &prepare_bindings,
                                    Some(&prepare_pc_bytes),
                                    bindless,
                                    pack_set,
                                    frame_slot,
                                    rw,
                                    rh,
                                    &mut live_image_states,
                                );
                            if rr_ready
                                && !rr_prepared
                                && let Some(ref mut mgr) = sdk
                            {
                                mgr.report_denoiser_status(
                                    "DLSS Ray Reconstruction guides unavailable; using SVGF",
                                );
                            }
                            if rr_prepared {
                                // Transition DLSS-RR prep inputs to SHADER_READ_ONLY_OPTIMAL and
                                // DLSS-RR output (`RES_UPSCALED_HDR`) to GENERAL before evaluation.
                                let mut rr_barriers = Vec::new();
                                for prep_id in [
                                    Self::RES_NRD_NORMAL_ROUGH,
                                    Self::RES_RT_HIT_DIST,
                                    Self::RES_NRD_IN_DIFF,
                                    Self::RES_NRD_IN_SPEC,
                                ] {
                                    if let Some(&img) = self.resource_images.get(&prep_id)
                                        && img != vk::Image::null()
                                    {
                                        let (src_stage, src_access, old_layout, aspect) =
                                            live_image_states.get(&img).copied().unwrap_or((
                                                vk::PipelineStageFlags2::COMPUTE_SHADER,
                                                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                                                vk::ImageLayout::GENERAL,
                                                vk::ImageAspectFlags::COLOR,
                                            ));
                                        if old_layout != vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
                                            rr_barriers.push(
                                                vk::ImageMemoryBarrier2::default()
                                                    .src_stage_mask(src_stage)
                                                    .src_access_mask(src_access)
                                                    .dst_stage_mask(
                                                        vk::PipelineStageFlags2::COMPUTE_SHADER,
                                                    )
                                                    .dst_access_mask(
                                                        vk::AccessFlags2::SHADER_SAMPLED_READ,
                                                    )
                                                    .old_layout(old_layout)
                                                    .new_layout(
                                                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                                    )
                                                    .image(img)
                                                    .subresource_range(vk::ImageSubresourceRange {
                                                        aspect_mask: aspect,
                                                        base_mip_level: 0,
                                                        level_count: 1,
                                                        base_array_layer: 0,
                                                        layer_count: 1,
                                                    }),
                                            );
                                            live_image_states.insert(
                                                img,
                                                (
                                                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                                                    vk::AccessFlags2::SHADER_SAMPLED_READ,
                                                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                                    aspect,
                                                ),
                                            );
                                            self.tracked_image_layouts.insert(
                                                img,
                                                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                            );
                                        }
                                    }
                                }
                                if rr_out_img != vk::Image::null() {
                                    let (src_stage, src_access, old_layout, aspect) =
                                        live_image_states.get(&rr_out_img).copied().unwrap_or((
                                            vk::PipelineStageFlags2::TOP_OF_PIPE,
                                            vk::AccessFlags2::NONE,
                                            self.tracked_image_layouts
                                                .get(&rr_out_img)
                                                .copied()
                                                .unwrap_or(vk::ImageLayout::UNDEFINED),
                                            vk::ImageAspectFlags::COLOR,
                                        ));
                                    rr_barriers.push(
                                        vk::ImageMemoryBarrier2::default()
                                            .src_stage_mask(src_stage)
                                            .src_access_mask(src_access)
                                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                                            .dst_access_mask(
                                                vk::AccessFlags2::SHADER_STORAGE_WRITE
                                                    | vk::AccessFlags2::SHADER_STORAGE_READ,
                                            )
                                            .old_layout(old_layout)
                                            .new_layout(vk::ImageLayout::GENERAL)
                                            .image(rr_out_img)
                                            .subresource_range(vk::ImageSubresourceRange {
                                                aspect_mask: aspect,
                                                base_mip_level: 0,
                                                level_count: 1,
                                                base_array_layer: 0,
                                                layer_count: 1,
                                            }),
                                    );
                                    live_image_states.insert(
                                        rr_out_img,
                                        (
                                            vk::PipelineStageFlags2::COMPUTE_SHADER,
                                            vk::AccessFlags2::SHADER_STORAGE_WRITE,
                                            vk::ImageLayout::GENERAL,
                                            aspect,
                                        ),
                                    );
                                    self.tracked_image_layouts
                                        .insert(rr_out_img, vk::ImageLayout::GENERAL);
                                }
                                if !rr_barriers.is_empty() {
                                    let dep = vk::DependencyInfo::default()
                                        .image_memory_barriers(&rr_barriers);
                                    unsafe {
                                        ctx.sync2_ext.cmd_pipeline_barrier2(cmd, &dep);
                                    }
                                }

                                if let Some(ref mut mgr) = sdk {
                                    let res = unsafe {
                                        mgr.evaluate_dlss_rr_with_retire(
                                            &denoise_dispatch,
                                            Some(&ctx.device),
                                            Some(retire),
                                        )
                                    };
                                    used_vendor = res.is_ok();
                                    match &res {
                                        Ok(()) => mgr.report_denoiser_status("DLSS Ray Reconstruction active"),
                                        Err(error) => mgr.report_denoiser_status(&format!(
                                            "DLSS Ray Reconstruction evaluation failed ({error}); using SVGF"
                                        )),
                                    }
                                    if used_vendor
                                        && dlss_rr_target_id == Self::RES_UPSCALED_HDR
                                        && rr_out_img != vk::Image::null()
                                    {
                                        dlss_rr_upscaled_this_frame = true;
                                        live_image_states.insert(
                                            rr_out_img,
                                            (
                                                vk::PipelineStageFlags2::COMPUTE_SHADER,
                                                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                                                vk::ImageLayout::GENERAL,
                                                vk::ImageAspectFlags::COLOR,
                                            ),
                                        );
                                        self.tracked_image_layouts
                                            .insert(rr_out_img, vk::ImageLayout::GENERAL);
                                    }
                                }
                            }
                        }
                        if let Some(ref mut mgr) = sdk
                            && dispatch.denoiser_mode == 1
                        {
                            mgr.report_denoiser_status("SVGF active");
                        }
                        if !used_vendor {
                            pipelines.dispatch_svgf_denoise(
                                ctx,
                                cmd,
                                self,
                                bindless,
                                pack_set,
                                tlas_handle,
                                frame_slot,
                                rw,
                                rh,
                                Some(blas),
                            );
                            if let Some(&den_img) =
                                self.resource_images.get(&Self::RES_DENOISED_RADIANCE)
                            {
                                live_image_states.insert(
                                    den_img,
                                    (
                                        vk::PipelineStageFlags2::COMPUTE_SHADER,
                                        vk::AccessFlags2::SHADER_STORAGE_WRITE,
                                        vk::ImageLayout::GENERAL,
                                        vk::ImageAspectFlags::COLOR,
                                    ),
                                );
                                self.tracked_image_layouts
                                    .insert(den_img, vk::ImageLayout::GENERAL);
                            }
                            for hist_id in [
                                Self::RES_SVGF_HISTORY_RADIANCE,
                                Self::RES_SVGF_HISTORY_MOMENTS,
                            ] {
                                if let Some(&h_img) = self.resource_images.get(&hist_id) {
                                    self.tracked_image_layouts
                                        .insert(h_img, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                                }
                            }
                        }
                    }
                    PassKind::UpscalerTsrOrVendor => {
                        if dlss_rr_upscaled_this_frame {
                            // DLSS-RR already denoised and upscaled directly into RES_UPSCALED_HDR.
                        } else {
                            let mut used_vendor = false;
                            let requested = match dispatch.upscaler_mode {
                                2 => Some("FSR3"),
                                3 => Some("DLSS-SR"),
                                4 => Some("XeSS"),
                                _ => None,
                            };
                            // Route denoised radiance into the upscaler when denoiser ran.
                            let mut upscale_dispatch = *dispatch;
                            upscale_dispatch.width = rw;
                            upscale_dispatch.height = rh;
                            upscale_dispatch.out_width = out_w;
                            upscale_dispatch.out_height = out_h;
                            if self.config.effective_denoiser_active()
                                && let Some(&den_img) =
                                    self.resource_images.get(&Self::RES_DENOISED_RADIANCE)
                                && let Some(den_view) =
                                    self.resolve_image_view(Self::RES_DENOISED_RADIANCE)
                            {
                                upscale_dispatch.input_image = den_img;
                                upscale_dispatch.input_view = den_view;
                            }
                            if let Some(ref mut mgr) = sdk {
                                let res = match dispatch.upscaler_mode {
                                    3 if mgr.is_dlss_sr_available() => {
                                        Some(unsafe { mgr.evaluate_dlss_sr(&upscale_dispatch) })
                                    }
                                    2 if mgr.is_fsr3_available() => {
                                        Some(unsafe { mgr.evaluate_fsr3(&upscale_dispatch) })
                                    }
                                    4 if mgr.is_xess_available() => {
                                        Some(unsafe { mgr.evaluate_xess(&upscale_dispatch) })
                                    }
                                    _ => None,
                                };
                                if let Some(r) = res {
                                    used_vendor = r.is_ok();
                                    mgr.last_evaluate_result = Some(match &r {
                                        Ok(_) => format!("{}: ok", requested.unwrap_or("vendor")),
                                        Err(e) => format!("{}: {e}", requested.unwrap_or("vendor")),
                                    });
                                }
                            }
                            if !used_vendor {
                                pipelines.dispatch_pass_with_graph(
                                    ctx,
                                    cmd,
                                    step_kind,
                                    self,
                                    bindless,
                                    pack_set,
                                    tlas_handle,
                                    frame_slot,
                                    out_w,
                                    out_h,
                                    Some(blas),
                                    Some(retire),
                                );
                                pipelines.copy_upscaled_history(ctx, cmd, self, out_w, out_h);
                                if let Some(&tsr_hist) =
                                    self.resource_images.get(&Self::RES_TSR_HISTORY)
                                {
                                    self.tracked_image_layouts.insert(
                                        tsr_hist,
                                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                    );
                                }
                            }
                        }
                    }
                    PassKind::Custom(custom_idx) => {
                        let scale = self
                            .custom_passes
                            .get(&custom_idx)
                            .map(|p| p.scale)
                            .unwrap_or(1.0);
                        let pass_w = ((out_w as f32) * scale).round().max(1.0) as u32;
                        let pass_h = ((out_h as f32) * scale).round().max(1.0) as u32;
                        pipelines.dispatch_pass_with_graph(
                            ctx,
                            cmd,
                            step_kind,
                            self,
                            bindless,
                            pack_set,
                            tlas_handle,
                            frame_slot,
                            pass_w,
                            pass_h,
                            Some(blas),
                            Some(retire),
                        );
                        // If any written custom target is marked `temporal = true`, copy it into its history target.
                        let write_count = self.compiled_steps[step_idx].writes.len();
                        for write_idx in 0..write_count {
                            let written_id = self.compiled_steps[step_idx].writes[write_idx].0;
                            if let Some(tgt) = self.custom_targets.get(&written_id)
                                && tgt.temporal
                                && let Some(hist_id) = tgt.history_id
                            {
                                let (tw, th) = self
                                    .internal_targets
                                    .get(&written_id)
                                    .map(|t| (t.width, t.height))
                                    .unwrap_or((pass_w, pass_h));
                                pipelines.copy_custom_temporal_history(
                                    ctx, cmd, self, written_id, hist_id, tw, th,
                                );
                                if let Some(&h_img) = self.resource_images.get(&hist_id) {
                                    self.tracked_image_layouts
                                        .insert(h_img, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                                }
                            }
                        }
                    }
                    _ => {
                        pipelines.dispatch_pass_with_graph(
                            ctx,
                            cmd,
                            step_kind,
                            self,
                            bindless,
                            pack_set,
                            tlas_handle,
                            frame_slot,
                            out_w,
                            out_h,
                            Some(blas),
                            Some(retire),
                        );
                    }
                }

                if step_kind == PassKind::PostGradeAndToneMap
                    && !direct_hudless_post
                    && let (Some(&post_img), Some(&hudless_img)) = (
                        self.resource_images.get(&Self::RES_POST_OUTPUT),
                        self.resource_images.get(&Self::RES_HUDLESS_COLOR),
                    )
                {
                    let (hud_src_stage, hud_src_access, hud_old_layout, hud_aspect) =
                        live_image_states.get(&hudless_img).copied().unwrap_or((
                            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                                | vk::PipelineStageFlags2::COMPUTE_SHADER
                                | vk::PipelineStageFlags2::ALL_TRANSFER,
                            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                                | vk::AccessFlags2::SHADER_WRITE
                                | vk::AccessFlags2::TRANSFER_WRITE,
                            vk::ImageLayout::GENERAL,
                            vk::ImageAspectFlags::COLOR,
                        ));
                    let copy_barriers = [
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                            .old_layout(vk::ImageLayout::GENERAL)
                            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                            .image(post_img)
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_mip_level: 0,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: 1,
                            }),
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(hud_src_stage)
                            .src_access_mask(hud_src_access)
                            .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .old_layout(hud_old_layout)
                            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                            .image(hudless_img)
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: hud_aspect,
                                base_mip_level: 0,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: 1,
                            }),
                    ];
                    let dep = vk::DependencyInfo::default().image_memory_barriers(&copy_barriers);
                    let region = [vk::ImageCopy {
                        src_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        src_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        dst_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        dst_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        extent: vk::Extent3D {
                            width: out_w,
                            height: out_h,
                            depth: 1,
                        },
                    }];
                    unsafe {
                        ctx.sync2_ext.cmd_pipeline_barrier2(cmd, &dep);
                        ctx.device.cmd_copy_image(
                            cmd,
                            post_img,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            hudless_img,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &region,
                        );
                    }
                    live_image_states.insert(
                        post_img,
                        (
                            vk::PipelineStageFlags2::ALL_TRANSFER,
                            vk::AccessFlags2::TRANSFER_READ,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            vk::ImageAspectFlags::COLOR,
                        ),
                    );
                    self.tracked_image_layouts
                        .insert(post_img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
                    live_image_states.insert(
                        hudless_img,
                        (
                            vk::PipelineStageFlags2::ALL_TRANSFER,
                            vk::AccessFlags2::TRANSFER_WRITE,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            hud_aspect,
                        ),
                    );
                    self.tracked_image_layouts
                        .insert(hudless_img, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
                }
            }
            executed_steps += 1;
        }

        // In HDR mode, `PostGradeAndToneMap` is skipped so linear HDR radiance stays in
        // `radiance_after_upscale`. Clear `RES_HUDLESS_COLOR` (`mainTarget.colorTexture`) to
        // transparent `(0, 0, 0, 0)` before `GuiRenderer.render()` runs so `hdr_compose.comp`
        // receives pure HUD alpha on `RES_UI_COLOR`.
        if self.config.hdr_enabled
            && let Some(&hudless_img) = self.resource_images.get(&Self::RES_HUDLESS_COLOR)
            && hudless_img != vk::Image::null()
        {
            let (src_stage, src_access, old_layout, aspect) =
                live_image_states.get(&hudless_img).copied().unwrap_or((
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                    vk::ImageLayout::GENERAL,
                    vk::ImageAspectFlags::COLOR,
                ));
            let clear_range = [vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            }];
            let to_clear = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(src_stage)
                .src_access_mask(src_access)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(old_layout)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(hudless_img)
                .subresource_range(clear_range[0])];
            unsafe {
                let dep = vk::DependencyInfo::default().image_memory_barriers(&to_clear);
                ctx.sync2_ext.cmd_pipeline_barrier2(cmd, &dep);
                ctx.device.cmd_clear_color_image(
                    cmd,
                    hudless_img,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    },
                    &clear_range,
                );
            }
            live_image_states.insert(
                hudless_img,
                (
                    vk::PipelineStageFlags2::ALL_TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    aspect,
                ),
            );
            self.tracked_image_layouts
                .insert(hudless_img, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        }

        // Return all Mojang-owned images cleanly to VK_IMAGE_LAYOUT_GENERAL before GuiRenderer / next frame
        let mut exit_src_default =
            vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::ALL_TRANSFER;
        if ctx.capabilities.ray_tracing_pipeline {
            exit_src_default |= vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR;
        }
        for (img, default_aspect) in mojang_owned_images.drain(..) {
            if !seen_exit.insert(img) {
                continue;
            }
            let Some(&(src_stage, src_access, old_layout, aspect)) = live_image_states.get(&img)
            else {
                continue;
            };
            let write_mask = vk::AccessFlags2::SHADER_WRITE.as_raw()
                | vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw()
                | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw()
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw()
                | vk::AccessFlags2::TRANSFER_WRITE.as_raw()
                | vk::AccessFlags2::MEMORY_WRITE.as_raw();
            let needs_restore =
                old_layout != vk::ImageLayout::GENERAL || (src_access.as_raw() & write_mask) != 0;
            if needs_restore {
                let final_aspect = if aspect.is_empty() {
                    default_aspect
                } else {
                    aspect
                };
                let eff_src_stage =
                    if src_stage.is_empty() || src_stage == vk::PipelineStageFlags2::ALL_COMMANDS {
                        exit_src_default
                    } else {
                        src_stage
                    };
                let mut dst_stage = vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags2::FRAGMENT_SHADER
                    | vk::PipelineStageFlags2::ALL_TRANSFER;
                let mut dst_access = vk::AccessFlags2::COLOR_ATTACHMENT_READ
                    | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags2::SHADER_READ
                    | vk::AccessFlags2::TRANSFER_READ;
                if final_aspect.contains(vk::ImageAspectFlags::DEPTH) {
                    dst_stage |= vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
                    dst_access |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
                }
                exit_barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(eff_src_stage)
                        .src_access_mask(src_access)
                        .dst_stage_mask(dst_stage)
                        .dst_access_mask(dst_access)
                        .old_layout(old_layout)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .image(img)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: final_aspect,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                );
                self.tracked_image_layouts
                    .insert(img, vk::ImageLayout::GENERAL);
            }
        }
        if !exit_barriers.is_empty() {
            let dep = vk::DependencyInfo::default().image_memory_barriers(&exit_barriers);
            unsafe {
                ctx.sync2_ext.cmd_pipeline_barrier2(cmd, &dep);
            }
        }

        self.frame_live_image_states = live_image_states;
        self.frame_mojang_owned_images = mojang_owned_images;
        self.frame_init_clear_ids = init_clear_ids;
        self.frame_exit_barriers = exit_barriers;
        self.frame_seen_exit_images = seen_exit;

        executed_steps
    }

    /// Executes post-GUI passes (`HdrCompose` and `FrameGenInterpolate`) immediately after
    /// `GuiRenderer.render()` completes, when `ui_img` (`mainTarget.colorTexture`) contains the
    /// rendered HUD/GUI and `RES_POST_OUTPUT` contains the pure HUD-less world color, then copies
    /// `RES_POST_OUTPUT` into `RES_PREV_HUDLESS_COLOR` for the next frame's temporal history.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_after_gui_with_context(
        &mut self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        pipelines: &mut GpuPipelineManager,
        bindless: Option<&BindlessTextureTable>,
        pack_set: vk::DescriptorSet,
        tlas_handle: vk::AccelerationStructureKHR,
        frame_slot: usize,
        width: u32,
        height: u32,
        mut sdk: Option<&mut RuntimeSdkManager>,
        dispatch: &SdkDispatchParams,
        retire: &RetireQueue,
        ui_img: vk::Image,
        ui_view: vk::ImageView,
        ui_only: bool,
    ) -> usize {
        let out_w = if dispatch.out_width > 0 {
            dispatch.out_width
        } else {
            width
        }
        .max(1);
        let out_h = if dispatch.out_height > 0 {
            dispatch.out_height
        } else {
            height
        }
        .max(1);
        let rw = if dispatch.width > 0 {
            dispatch.width
        } else {
            out_w
        }
        .max(1);
        let rh = if dispatch.height > 0 {
            dispatch.height
        } else {
            out_h
        }
        .max(1);
        let prev_target_sizes = (self.internal_render_size, self.internal_target_size);
        self.ensure_internal_targets(ctx, rw, rh, out_w, out_h, retire);
        if prev_target_sizes != (self.internal_render_size, self.internal_target_size) {
            pipelines.descriptor_sig_cache.clear();
        }
        if ui_img != vk::Image::null() {
            let (fmt, aspect) = Self::default_format_and_aspect(Self::RES_UI_COLOR);
            self.bind_resource_image_detailed(
                Some(ctx),
                Self::RES_UI_COLOR,
                ui_img,
                ui_view,
                vk::Sampler::null(),
                fmt,
                aspect,
            );
        }

        // Save the original RES_HUDLESS_COLOR binding (`mainTarget.colorTexture`) and temporarily
        // alias RES_HUDLESS_COLOR to RES_POST_OUTPUT (pure HUD-less world color) so
        // `FrameGenInterpolate` reads HUD-less current world color at binding 1 (`uCurrColor`)
        // and HUD color at binding 4 (`uGuiColor`).
        let saved_hudless = self.bound_images.get(&Self::RES_HUDLESS_COLOR).copied();
        let hudless_src_id = if self.config.hdr_enabled {
            Self::RES_SWAPCHAIN_OUT
        } else {
            Self::RES_POST_OUTPUT
        };
        if !self.config.hdr_enabled
            && let Some(&post_bound) = self.bound_images.get(&Self::RES_POST_OUTPUT)
        {
            self.resource_images
                .insert(Self::RES_HUDLESS_COLOR, post_bound.image);
            self.bound_images
                .insert(Self::RES_HUDLESS_COLOR, post_bound);
        }

        let subrange = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let mut executed = 0usize;

        if self.config.hdr_enabled {
            let rad_id = if self.config.upscaler_mode != 0 {
                Self::RES_UPSCALED_HDR
            } else if self.config.effective_denoiser_active() {
                Self::RES_DENOISED_RADIANCE
            } else {
                Self::RES_RAW_RADIANCE
            };
            let mut barriers = Vec::new();
            if let Some(&rad_img) = self.resource_images.get(&rad_id)
                && rad_img != vk::Image::null()
            {
                let old = self
                    .tracked_image_layouts
                    .get(&rad_img)
                    .copied()
                    .unwrap_or(vk::ImageLayout::UNDEFINED);
                barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                        .old_layout(old)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image(rad_img)
                        .subresource_range(subrange),
                );
                self.tracked_image_layouts
                    .insert(rad_img, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            }
            if ui_img != vk::Image::null() {
                barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image(ui_img)
                        .subresource_range(subrange),
                );
            }
            if let Some(&swap_out) = self.resource_images.get(&Self::RES_SWAPCHAIN_OUT)
                && swap_out != vk::Image::null()
            {
                let old = self
                    .tracked_image_layouts
                    .get(&swap_out)
                    .copied()
                    .unwrap_or(vk::ImageLayout::UNDEFINED);
                barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .src_access_mask(vk::AccessFlags2::MEMORY_READ)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                        .old_layout(old)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .image(swap_out)
                        .subresource_range(subrange),
                );
                self.tracked_image_layouts
                    .insert(swap_out, vk::ImageLayout::GENERAL);
            }
            if !barriers.is_empty() {
                let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                unsafe {
                    ctx.sync2_ext.cmd_pipeline_barrier2(cmd, &dep);
                }
            }
            let prev_ui_only = pipelines.hdr_push_constants.pad0;
            pipelines.hdr_push_constants.pad0 = u32::from(ui_only);
            pipelines.dispatch_pass_with_graph(
                ctx,
                cmd,
                PassKind::HdrCompose,
                self,
                bindless,
                pack_set,
                tlas_handle,
                frame_slot,
                out_w,
                out_h,
                None,
                Some(retire),
            );
            pipelines.hdr_push_constants.pad0 = prev_ui_only;
            executed += 1;
        }

        if !ui_only
            && self.config.hdr_enabled
            && let Some(&swap_bound) = self.bound_images.get(&Self::RES_SWAPCHAIN_OUT)
        {
            self.resource_images
                .insert(Self::RES_HUDLESS_COLOR, swap_bound.image);
            self.bound_images
                .insert(Self::RES_HUDLESS_COLOR, swap_bound);
        }

        if !ui_only && self.config.framegen_mode != 0 {
            let active_depth = if self.config.tier == RenderTier::Tier3PathTracing {
                Self::RES_LINEAR_DEPTH
            } else {
                Self::RES_MC_MAIN_DEPTH
            };
            let mut barriers = Vec::new();
            for read_id in [
                Self::RES_PREV_HUDLESS_COLOR,
                Self::RES_HUDLESS_COLOR,
                Self::RES_GBUFFER_VELOCITY,
                active_depth,
                Self::RES_UI_COLOR,
            ] {
                if let Some(&r_img) = self.resource_images.get(&read_id)
                    && r_img != vk::Image::null()
                {
                    let aspect = Self::default_format_and_aspect(read_id).1;
                    let old = if r_img == ui_img && self.config.hdr_enabled {
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    } else if r_img == ui_img {
                        vk::ImageLayout::GENERAL
                    } else {
                        self.tracked_image_layouts
                            .get(&r_img)
                            .copied()
                            .unwrap_or(vk::ImageLayout::GENERAL)
                    };
                    if old != vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
                        barriers.push(
                            vk::ImageMemoryBarrier2::default()
                                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                                .old_layout(old)
                                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .image(r_img)
                                .subresource_range(vk::ImageSubresourceRange {
                                    aspect_mask: aspect,
                                    base_mip_level: 0,
                                    level_count: 1,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                }),
                        );
                        self.tracked_image_layouts
                            .insert(r_img, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                    }
                }
            }
            if let Some(&interp_img) = self.resource_images.get(&Self::RES_INTERPOLATED_FRAME)
                && interp_img != vk::Image::null()
            {
                let old = self
                    .tracked_image_layouts
                    .get(&interp_img)
                    .copied()
                    .unwrap_or(vk::ImageLayout::UNDEFINED);
                barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .src_access_mask(vk::AccessFlags2::MEMORY_READ)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                        .old_layout(old)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .image(interp_img)
                        .subresource_range(subrange),
                );
                self.tracked_image_layouts
                    .insert(interp_img, vk::ImageLayout::GENERAL);
            }
            if !barriers.is_empty() {
                let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                unsafe {
                    ctx.sync2_ext.cmd_pipeline_barrier2(cmd, &dep);
                }
            }

            let mut fg_dispatch = *dispatch;
            if let Some(&src_img) = self.resource_images.get(&hudless_src_id)
                && src_img != vk::Image::null()
            {
                fg_dispatch.input_image = src_img;
            }
            if let Some(src_view) = self.resolve_image_view(hudless_src_id) {
                fg_dispatch.input_view = src_view;
            }
            if self.config.framegen_mode == 1 {
                pipelines.dispatch_pass_with_graph(
                    ctx,
                    cmd,
                    PassKind::FrameGenInterpolate,
                    self,
                    bindless,
                    pack_set,
                    tlas_handle,
                    frame_slot,
                    out_w,
                    out_h,
                    None,
                    Some(retire),
                );
            } else if let Some(ref mut mgr) = sdk
                && mgr.framegen_failed_mode == 0
            {
                let mode = self.config.framegen_mode;
                let result = match mode {
                    2 => unsafe { mgr.evaluate_fsr3_fg(&fg_dispatch) },
                    3 => unsafe { mgr.evaluate_dlss_fg(&fg_dispatch) },
                    _ => Err("Unknown vendor frame generation mode".to_owned()),
                };
                if let Err(error) = result {
                    tracing::warn!(mode, %error, "Vendor frame generation failed");
                    mgr.framegen_failed_mode = mode;
                }
            }
            executed += 1;
        }

        // Copy RES_POST_OUTPUT (current frame's pure HUD-less world color) into
        // RES_PREV_HUDLESS_COLOR for the next frame's SSGI and FrameGen temporal history.
        let needs_world_history = self.compiled_steps.iter().any(|step| {
            step.reads
                .iter()
                .any(|&(resource, _)| resource == Self::RES_PREV_HUDLESS_COLOR)
        });
        if !ui_only
            && needs_world_history
            && let (Some(&post_img), Some(&prev_img)) = (
                self.resource_images.get(&hudless_src_id),
                self.resource_images.get(&Self::RES_PREV_HUDLESS_COLOR),
            )
            && post_img != vk::Image::null()
            && prev_img != vk::Image::null()
            && post_img != prev_img
        {
            let post_old = self
                .tracked_image_layouts
                .get(&post_img)
                .copied()
                .unwrap_or(vk::ImageLayout::GENERAL);
            let prev_old = self
                .tracked_image_layouts
                .get(&prev_img)
                .copied()
                .unwrap_or(vk::ImageLayout::UNDEFINED);
            let prev_next = if self
                .internal_targets
                .contains_key(&Self::RES_PREV_HUDLESS_COLOR)
            {
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            } else {
                vk::ImageLayout::GENERAL
            };
            let pre_copy = [
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .old_layout(post_old)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(post_img)
                    .subresource_range(subrange),
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags2::MEMORY_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(prev_old)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(prev_img)
                    .subresource_range(subrange),
            ];
            let region = [vk::ImageCopy {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                extent: vk::Extent3D {
                    width: out_w,
                    height: out_h,
                    depth: 1,
                },
            }];
            let post_copy = [
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(post_img)
                    .subresource_range(subrange),
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(prev_next)
                    .image(prev_img)
                    .subresource_range(subrange),
            ];
            let src_fmt = self
                .bound_images
                .get(&hudless_src_id)
                .map(|b| b.format)
                .unwrap_or(vk::Format::R8G8B8A8_UNORM);
            let dst_fmt = self
                .bound_images
                .get(&Self::RES_PREV_HUDLESS_COLOR)
                .map(|b| b.format)
                .unwrap_or(vk::Format::R8G8B8A8_UNORM);
            unsafe {
                let dep_pre = vk::DependencyInfo::default().image_memory_barriers(&pre_copy);
                ctx.sync2_ext.cmd_pipeline_barrier2(cmd, &dep_pre);
                if src_fmt == dst_fmt {
                    ctx.device.cmd_copy_image(
                        cmd,
                        post_img,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        prev_img,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &region,
                    );
                } else {
                    let blit = [vk::ImageBlit {
                        src_subresource: region[0].src_subresource,
                        src_offsets: [
                            vk::Offset3D { x: 0, y: 0, z: 0 },
                            vk::Offset3D {
                                x: out_w as i32,
                                y: out_h as i32,
                                z: 1,
                            },
                        ],
                        dst_subresource: region[0].dst_subresource,
                        dst_offsets: [
                            vk::Offset3D { x: 0, y: 0, z: 0 },
                            vk::Offset3D {
                                x: out_w as i32,
                                y: out_h as i32,
                                z: 1,
                            },
                        ],
                    }];
                    ctx.device.cmd_blit_image(
                        cmd,
                        post_img,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        prev_img,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &blit,
                        vk::Filter::LINEAR,
                    );
                }
                let dep_post = vk::DependencyInfo::default().image_memory_barriers(&post_copy);
                ctx.sync2_ext.cmd_pipeline_barrier2(cmd, &dep_post);
            }
            executed += 1;
            self.tracked_image_layouts
                .insert(post_img, vk::ImageLayout::GENERAL);
            self.tracked_image_layouts.insert(prev_img, prev_next);
            self.persistent_layouts.insert(
                Self::RES_PREV_HUDLESS_COLOR,
                (
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_SAMPLED_READ,
                    prev_next,
                ),
            );
        }

        // Restore `ui_img` (`mainTarget.colorTexture`) and any Mojang depth image to GENERAL.
        let mut exit_src =
            vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::ALL_TRANSFER;
        if ctx.capabilities.ray_tracing_pipeline {
            exit_src |= vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR;
        }
        let exit_dst_color_stage = vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
            | vk::PipelineStageFlags2::FRAGMENT_SHADER
            | vk::PipelineStageFlags2::ALL_TRANSFER;
        let exit_dst_color_access = vk::AccessFlags2::COLOR_ATTACHMENT_READ
            | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags2::SHADER_READ
            | vk::AccessFlags2::TRANSFER_READ;
        let mut restore_barriers = Vec::new();
        if ui_img != vk::Image::null()
            && (self.config.hdr_enabled || self.config.framegen_mode != 0)
        {
            restore_barriers.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .dst_stage_mask(exit_dst_color_stage)
                    .dst_access_mask(exit_dst_color_access)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(ui_img)
                    .subresource_range(subrange),
            );
            self.tracked_image_layouts
                .insert(ui_img, vk::ImageLayout::GENERAL);
        }
        for restore_id in [Self::RES_GBUFFER_VELOCITY, Self::RES_MC_MAIN_DEPTH] {
            if let Some(&r_img) = self.resource_images.get(&restore_id)
                && r_img != vk::Image::null()
                && r_img != ui_img
                && let Some(&cur_layout) = self.tracked_image_layouts.get(&r_img)
                && cur_layout != vk::ImageLayout::GENERAL
            {
                let aspect = Self::default_format_and_aspect(restore_id).1;
                let mut dst_stage = exit_dst_color_stage;
                let mut dst_access = exit_dst_color_access;
                if aspect.contains(vk::ImageAspectFlags::DEPTH) {
                    dst_stage |= vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
                    dst_access |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
                }
                restore_barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                        .dst_stage_mask(dst_stage)
                        .dst_access_mask(dst_access)
                        .old_layout(cur_layout)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .image(r_img)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: aspect,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                );
                self.tracked_image_layouts
                    .insert(r_img, vk::ImageLayout::GENERAL);
            }
        }
        // Return RES_SWAPCHAIN_OUT to GENERAL after FrameGenInterpolate (which may have
        // transitioned it to SHADER_READ_ONLY_OPTIMAL when aliased to RES_HUDLESS_COLOR in HDR).
        if let Some(&swap_out) = self.resource_images.get(&Self::RES_SWAPCHAIN_OUT)
            && swap_out != vk::Image::null()
            && let Some(&cur_layout) = self.tracked_image_layouts.get(&swap_out)
            && cur_layout != vk::ImageLayout::GENERAL
        {
            restore_barriers.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(exit_src)
                    .src_access_mask(
                        vk::AccessFlags2::SHADER_READ
                            | vk::AccessFlags2::SHADER_WRITE
                            | vk::AccessFlags2::TRANSFER_READ
                            | vk::AccessFlags2::TRANSFER_WRITE,
                    )
                    .dst_stage_mask(exit_dst_color_stage)
                    .dst_access_mask(exit_dst_color_access)
                    .old_layout(cur_layout)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(swap_out)
                    .subresource_range(subrange),
            );
            self.tracked_image_layouts
                .insert(swap_out, vk::ImageLayout::GENERAL);
        }
        if !restore_barriers.is_empty() {
            let dep = vk::DependencyInfo::default().image_memory_barriers(&restore_barriers);
            unsafe {
                ctx.sync2_ext.cmd_pipeline_barrier2(cmd, &dep);
            }
        }

        // Restore the original RES_HUDLESS_COLOR binding.
        if let Some(orig) = saved_hudless {
            self.resource_images
                .insert(Self::RES_HUDLESS_COLOR, orig.image);
            self.bound_images.insert(Self::RES_HUDLESS_COLOR, orig);
        }

        executed
    }

    /// Backwards-compatible execution entrypoint.
    pub fn execute(
        &self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        pipelines: &mut GpuPipelineManager,
        width: u32,
        height: u32,
    ) -> usize {
        let mut executed_steps = 0usize;
        for step in &self.compiled_steps {
            if !step.is_external_raster && step.kind != PassKind::BlasTlasUpdate {
                pipelines.dispatch_pass(ctx, cmd, step.kind, width, height);
            }
            executed_steps += 1;
        }
        executed_steps
    }

    fn build_nodes_with_custom(&self, cfg: GraphConfig) -> Vec<GraphPassNode> {
        let mut nodes =
            if self.graph_mode == PipelineGraphMode::Custom && !self.custom_passes.is_empty() {
                self.build_custom_mode_nodes(cfg)
            } else {
                Self::build_canonical_nodes(cfg)
            };

        if cfg.tier == RenderTier::Tier3PathTracing
            && let Some(target) = self
                .custom_targets
                .values()
                .find(|target| target.name == "restir_gi_reservoir")
            && let Some(history_id) = target.history_id
            && let Some(pathtrace) = nodes
                .iter_mut()
                .find(|node| node.kind == PassKind::FullPathTracing)
        {
            pathtrace
                .reads
                .push((history_id, ResourceAccess::SampledRead));
            pathtrace
                .writes
                .push((target.id, ResourceAccess::StorageImageWrite));
        }

        if self.graph_mode == PipelineGraphMode::Extend && !self.custom_passes.is_empty() {
            let mut ordered: Vec<&CustomPassSpec> = self.custom_passes.values().collect();
            ordered.sort_by_key(|p| p.custom_index);
            for spec in ordered {
                let tier_ok = cfg.tier.index() >= spec.min_tier.index()
                    && cfg.tier.index() <= spec.max_tier.index();
                let node = GraphPassNode {
                    kind: spec.kind,
                    name: spec.id.clone(),
                    reads: spec.reads.clone(),
                    writes: spec.writes.clone(),
                    enabled: spec.enabled && tier_ok,
                    is_external_raster: false,
                };
                if let Some(ref rep) = spec.replaces
                    && let Some(pos) = Self::find_node_index(&nodes, rep)
                {
                    nodes[pos] = node;
                    continue;
                }
                if let Some(ref bef) = spec.before
                    && let Some(pos) = Self::find_node_index(&nodes, bef)
                {
                    nodes.insert(pos, node);
                    continue;
                }
                if let Some(ref aft) = spec.after
                    && let Some(pos) = Self::find_node_index(&nodes, aft)
                {
                    nodes.insert(pos + 1, node);
                    continue;
                }
                // Default splice position: before denoiser or post-tonemap
                if let Some(pos) = Self::find_node_index(&nodes, "denoiser_svgf_nrd")
                    .or_else(|| Self::find_node_index(&nodes, "post_grade_and_tonemap"))
                {
                    nodes.insert(pos, node);
                } else {
                    nodes.push(node);
                }
            }
        }
        nodes
    }

    fn find_node_index(nodes: &[GraphPassNode], target: &str) -> Option<usize> {
        let t = target.trim().to_ascii_lowercase();
        nodes.iter().position(|n| {
            let name_l = n.name.to_ascii_lowercase();
            name_l == t
                || matches!(
                    (n.kind, t.as_str()),
                    (
                        PassKind::BlasTlasUpdate,
                        "blas_tlas_update" | "rt_blas_tlas_update"
                    ) | (
                        PassKind::ShadowCsm,
                        "shadow_csm" | "shadow_csm_pass" | "csm"
                    ) | (
                        PassKind::GbufferMrt,
                        "gbuffer" | "gbuffer_mrt" | "gbuffer_mrt_pass"
                    ) | (
                        PassKind::ComputeVolumetricFog,
                        "volumetric_fog" | "volumetric_fog_pass" | "fog"
                    ) | (PassKind::ComputeSsgi, "ssgi" | "ssgi_screen_space_gi")
                        | (
                            PassKind::ComputeFroxelGi
                                | PassKind::RayQueryLighting
                                | PassKind::FullPathTracing,
                            "lighting"
                                | "compute_lighting"
                                | "lighting_raster_compute_gi"
                                | "lighting_hybrid_rayquery"
                                | "lighting_full_path_tracing"
                        )
                        | (
                            PassKind::DenoiserSvgfOrNrd,
                            "denoiser" | "denoiser_svgf_nrd" | "svgf"
                        )
                        | (
                            PassKind::UpscalerTsrOrVendor,
                            "upscaler" | "upscaler_tsr_vendor" | "tsr"
                        )
                        | (
                            PassKind::PostGradeAndToneMap,
                            "post_tonemap" | "post_grade_and_tonemap" | "tonemap"
                        )
                        | (PassKind::HdrCompose, "hdr_compose" | "hdr_scrgb_compose")
                        | (
                            PassKind::FrameGenInterpolate,
                            "framegen" | "framegen_optical_flow_present"
                        )
                )
        })
    }

    fn build_custom_mode_nodes(&self, cfg: GraphConfig) -> Vec<GraphPassNode> {
        let is_rt = matches!(
            cfg.tier,
            RenderTier::Tier2RayQuery | RenderTier::Tier3PathTracing
        );
        let is_tier3 = cfg.tier == RenderTier::Tier3PathTracing;
        let mut nodes = vec![
            GraphPassNode {
                kind: PassKind::BlasTlasUpdate,
                name: "rt_blas_tlas_update".to_owned(),
                reads: vec![],
                writes: vec![
                    (
                        Self::RES_TLAS,
                        ResourceAccess::AccelerationStructureBuildWrite,
                    ),
                    (
                        Self::RES_SHARC_HASH_TABLE,
                        ResourceAccess::StorageBufferReadWrite,
                    ),
                ],
                enabled: is_rt,
                is_external_raster: false,
            },
            GraphPassNode {
                kind: PassKind::ShadowCsm,
                name: "shadow_csm_pass".to_owned(),
                reads: vec![],
                writes: vec![(Self::RES_SHADOW_CSM, ResourceAccess::DepthAttachmentWrite)],
                enabled: !is_tier3,
                is_external_raster: false,
            },
            GraphPassNode {
                kind: PassKind::GbufferMrt,
                name: "gbuffer_mrt_pass".to_owned(),
                reads: vec![],
                writes: vec![
                    (
                        Self::RES_GBUFFER_ALBEDO,
                        ResourceAccess::ColorAttachmentWrite,
                    ),
                    (
                        Self::RES_GBUFFER_NORMAL,
                        ResourceAccess::ColorAttachmentWrite,
                    ),
                    (
                        Self::RES_GBUFFER_MATERIAL,
                        ResourceAccess::ColorAttachmentWrite,
                    ),
                    (
                        Self::RES_GBUFFER_VELOCITY,
                        ResourceAccess::ColorAttachmentWrite,
                    ),
                    (
                        Self::RES_MC_MAIN_DEPTH,
                        ResourceAccess::DepthAttachmentWrite,
                    ),
                    (Self::RES_LINEAR_DEPTH, ResourceAccess::DepthAttachmentWrite),
                ],
                enabled: !is_tier3,
                is_external_raster: true,
            },
        ];

        let mut ordered: Vec<&CustomPassSpec> = self.custom_passes.values().collect();
        ordered.sort_by_key(|p| p.custom_index);
        for spec in ordered {
            let tier_ok = cfg.tier.index() >= spec.min_tier.index()
                && cfg.tier.index() <= spec.max_tier.index();
            nodes.push(GraphPassNode {
                kind: spec.kind,
                name: spec.id.clone(),
                reads: spec.reads.clone(),
                writes: spec.writes.clone(),
                enabled: spec.enabled && tier_ok,
                is_external_raster: false,
            });
        }
        nodes
    }

    fn build_canonical_nodes(cfg: GraphConfig) -> Vec<GraphPassNode> {
        let mut nodes = Vec::with_capacity(14);
        let is_rt = matches!(
            cfg.tier,
            RenderTier::Tier2RayQuery | RenderTier::Tier3PathTracing
        );
        let is_tier3 = cfg.tier == RenderTier::Tier3PathTracing;
        let active_depth = if is_tier3 {
            Self::RES_LINEAR_DEPTH
        } else {
            Self::RES_MC_MAIN_DEPTH
        };
        let active_albedo = if is_tier3 {
            Self::RES_PT_ALBEDO
        } else {
            Self::RES_GBUFFER_ALBEDO
        };

        nodes.push(GraphPassNode {
            kind: PassKind::BlasTlasUpdate,
            name: "rt_blas_tlas_update".to_owned(),
            reads: vec![],
            writes: vec![
                (
                    Self::RES_TLAS,
                    ResourceAccess::AccelerationStructureBuildWrite,
                ),
                (
                    Self::RES_SHARC_HASH_TABLE,
                    ResourceAccess::StorageBufferReadWrite,
                ),
            ],
            enabled: is_rt,
            is_external_raster: false,
        });

        nodes.push(GraphPassNode {
            kind: PassKind::ShadowCsm,
            name: "shadow_csm_pass".to_owned(),
            reads: vec![],
            writes: vec![(Self::RES_SHADOW_CSM, ResourceAccess::DepthAttachmentWrite)],
            enabled: !is_tier3,
            is_external_raster: false,
        });

        nodes.push(GraphPassNode {
            kind: PassKind::GbufferMrt,
            name: "gbuffer_mrt_pass".to_owned(),
            reads: vec![],
            writes: vec![
                (
                    Self::RES_GBUFFER_ALBEDO,
                    ResourceAccess::ColorAttachmentWrite,
                ),
                (
                    Self::RES_GBUFFER_NORMAL,
                    ResourceAccess::ColorAttachmentWrite,
                ),
                (
                    Self::RES_GBUFFER_MATERIAL,
                    ResourceAccess::ColorAttachmentWrite,
                ),
                (
                    Self::RES_GBUFFER_VELOCITY,
                    ResourceAccess::ColorAttachmentWrite,
                ),
                (
                    Self::RES_MC_MAIN_DEPTH,
                    ResourceAccess::DepthAttachmentWrite,
                ),
                (Self::RES_LINEAR_DEPTH, ResourceAccess::DepthAttachmentWrite),
            ],
            enabled: !is_tier3,
            is_external_raster: true,
        });

        nodes.push(GraphPassNode {
            kind: PassKind::ComputeVolumetricFog,
            name: "volumetric_fog_pass".to_owned(),
            reads: vec![
                (Self::RES_SHADOW_CSM, ResourceAccess::SampledRead),
                (
                    Self::RES_VOLUMETRIC_FOG_HISTORY,
                    ResourceAccess::SampledRead,
                ),
            ],
            writes: vec![(Self::RES_VOLUMETRIC_FOG, ResourceAccess::StorageImageWrite)],
            enabled: true,
            is_external_raster: false,
        });

        match cfg.tier {
            RenderTier::Tier0Raster | RenderTier::Tier1ComputeGi => {
                if cfg.tier == RenderTier::Tier1ComputeGi {
                    // Screen-space global illumination runs BEFORE the final lighting pass on Tier 1.
                    nodes.push(GraphPassNode {
                        kind: PassKind::ComputeSsgi,
                        name: "ssgi_screen_space_gi".to_owned(),
                        reads: vec![
                            (Self::RES_GBUFFER_ALBEDO, ResourceAccess::SampledRead),
                            (Self::RES_GBUFFER_NORMAL, ResourceAccess::SampledRead),
                            (Self::RES_MC_MAIN_DEPTH, ResourceAccess::SampledRead),
                            (Self::RES_GBUFFER_VELOCITY, ResourceAccess::SampledRead),
                            (Self::RES_PREV_HUDLESS_COLOR, ResourceAccess::SampledRead),
                            (Self::RES_SSGI_HISTORY, ResourceAccess::SampledRead),
                        ],
                        writes: vec![(Self::RES_SSGI_OUT, ResourceAccess::StorageImageWrite)],
                        enabled: true,
                        is_external_raster: false,
                    });
                }
                // `compute_lighting.comp` unconditionally samples `uSsgi` at Set 0 binding 8 and
                // `uVolumetricFog` at Set 0 binding 9 on Tier 0 and Tier 1.
                let gi_reads = vec![
                    (Self::RES_SHADOW_CSM, ResourceAccess::SampledRead),
                    (Self::RES_GBUFFER_ALBEDO, ResourceAccess::SampledRead),
                    (Self::RES_GBUFFER_NORMAL, ResourceAccess::SampledRead),
                    (Self::RES_GBUFFER_MATERIAL, ResourceAccess::SampledRead),
                    (Self::RES_MC_MAIN_DEPTH, ResourceAccess::SampledRead),
                    (Self::RES_SSGI_OUT, ResourceAccess::SampledRead),
                    (Self::RES_VOLUMETRIC_FOG, ResourceAccess::SampledRead),
                    (Self::RES_DH_ALBEDO, ResourceAccess::SampledRead),
                    (Self::RES_DH_NORMAL, ResourceAccess::SampledRead),
                    (Self::RES_DH_MATERIAL, ResourceAccess::SampledRead),
                    (Self::RES_DH_DEPTH, ResourceAccess::SampledRead),
                ];
                nodes.push(GraphPassNode {
                    kind: PassKind::ComputeFroxelGi,
                    name: "lighting_raster_compute_gi".to_owned(),
                    reads: gi_reads,
                    writes: vec![(Self::RES_RAW_RADIANCE, ResourceAccess::StorageImageWrite)],
                    enabled: true,
                    is_external_raster: false,
                });
            }
            RenderTier::Tier2RayQuery => {
                nodes.push(GraphPassNode {
                    kind: PassKind::RayQueryLighting,
                    name: "lighting_hybrid_rayquery".to_owned(),
                    reads: vec![
                        (
                            Self::RES_TLAS,
                            ResourceAccess::AccelerationStructureTraceRead,
                        ),
                        (
                            Self::RES_SHARC_HASH_TABLE,
                            ResourceAccess::StorageBufferReadWrite,
                        ),
                        (
                            Self::RES_RESTIR_RESERVOIRS,
                            ResourceAccess::StorageBufferReadWrite,
                        ),
                        (Self::RES_SHADOW_CSM, ResourceAccess::SampledRead),
                        (Self::RES_GBUFFER_ALBEDO, ResourceAccess::SampledRead),
                        (Self::RES_GBUFFER_NORMAL, ResourceAccess::SampledRead),
                        (Self::RES_GBUFFER_MATERIAL, ResourceAccess::SampledRead),
                        (Self::RES_MC_MAIN_DEPTH, ResourceAccess::SampledRead),
                        (Self::RES_GBUFFER_VELOCITY, ResourceAccess::SampledRead),
                        (Self::RES_VOLUMETRIC_FOG, ResourceAccess::SampledRead),
                        (Self::RES_DH_ALBEDO, ResourceAccess::SampledRead),
                        (Self::RES_DH_NORMAL, ResourceAccess::SampledRead),
                        (Self::RES_DH_MATERIAL, ResourceAccess::SampledRead),
                        (Self::RES_DH_DEPTH, ResourceAccess::SampledRead),
                    ],
                    writes: vec![
                        (Self::RES_RAW_RADIANCE, ResourceAccess::StorageImageWrite),
                        (Self::RES_SHARP_RADIANCE, ResourceAccess::StorageImageWrite),
                    ],
                    enabled: true,
                    is_external_raster: false,
                });
            }
            RenderTier::Tier3PathTracing => {
                nodes.push(GraphPassNode {
                    kind: PassKind::FullPathTracing,
                    name: "lighting_full_path_tracing".to_owned(),
                    reads: vec![
                        (
                            Self::RES_TLAS,
                            ResourceAccess::AccelerationStructureTraceRead,
                        ),
                        (
                            Self::RES_SHARC_HASH_TABLE,
                            ResourceAccess::StorageBufferReadWrite,
                        ),
                        (
                            Self::RES_RESTIR_RESERVOIRS,
                            ResourceAccess::StorageBufferReadWrite,
                        ),
                        (Self::RES_VOLUMETRIC_FOG, ResourceAccess::SampledRead),
                    ],
                    writes: vec![
                        (Self::RES_RAW_RADIANCE, ResourceAccess::StorageImageWrite),
                        (Self::RES_GBUFFER_NORMAL, ResourceAccess::StorageImageWrite),
                        (
                            Self::RES_GBUFFER_MATERIAL,
                            ResourceAccess::StorageImageWrite,
                        ),
                        (
                            Self::RES_GBUFFER_VELOCITY,
                            ResourceAccess::StorageImageWrite,
                        ),
                        (Self::RES_LINEAR_DEPTH, ResourceAccess::StorageImageWrite),
                        (Self::RES_PT_ALBEDO, ResourceAccess::StorageImageWrite),
                        (Self::RES_SHARP_RADIANCE, ResourceAccess::StorageImageWrite),
                    ],
                    enabled: true,
                    is_external_raster: false,
                });

                nodes.push(GraphPassNode {
                    kind: PassKind::DepthReprojectOverlay,
                    name: "tier3_mc_depth_reproject".to_owned(),
                    reads: vec![(Self::RES_LINEAR_DEPTH, ResourceAccess::SampledRead)],
                    writes: vec![(
                        Self::RES_MC_MAIN_DEPTH,
                        ResourceAccess::DepthAttachmentWrite,
                    )],
                    enabled: cfg.overlay_pass_enabled,
                    is_external_raster: false,
                });
            }
        }

        let radiance_after_denoise = if cfg.effective_denoiser_active() {
            nodes.push(GraphPassNode {
                kind: PassKind::DenoiserSvgfOrNrd,
                name: "denoiser_svgf_nrd".to_owned(),
                reads: vec![
                    (Self::RES_RAW_RADIANCE, ResourceAccess::SampledRead),
                    (Self::RES_GBUFFER_NORMAL, ResourceAccess::SampledRead),
                    (Self::RES_GBUFFER_VELOCITY, ResourceAccess::SampledRead),
                    (active_depth, ResourceAccess::SampledRead),
                    (
                        if is_tier3 {
                            Self::RES_PT_ALBEDO
                        } else {
                            Self::RES_GBUFFER_ALBEDO
                        },
                        ResourceAccess::SampledRead,
                    ),
                    (Self::RES_SVGF_HISTORY_RADIANCE, ResourceAccess::SampledRead),
                    (Self::RES_SVGF_HISTORY_MOMENTS, ResourceAccess::SampledRead),
                ],
                writes: vec![
                    (
                        Self::RES_DENOISED_RADIANCE,
                        ResourceAccess::StorageImageWrite,
                    ),
                    (Self::RES_SVGF_MOMENTS, ResourceAccess::StorageImageWrite),
                    (Self::RES_SVGF_PING, ResourceAccess::StorageImageWrite),
                ],
                enabled: true,
                is_external_raster: false,
            });
            if is_rt {
                nodes
                    .last_mut()
                    .unwrap()
                    .reads
                    .push((Self::RES_SHARP_RADIANCE, ResourceAccess::SampledRead));
            }
            if cfg.denoiser_mode == 3 {
                nodes
                    .last_mut()
                    .unwrap()
                    .reads
                    .push((Self::RES_GBUFFER_MATERIAL, ResourceAccess::SampledRead));
            }
            Self::RES_DENOISED_RADIANCE
        } else {
            Self::RES_RAW_RADIANCE
        };

        let radiance_after_upscale = if cfg.upscaler_mode != 0 {
            nodes.push(GraphPassNode {
                kind: PassKind::UpscalerTsrOrVendor,
                name: "upscaler_tsr_vendor".to_owned(),
                reads: vec![
                    (radiance_after_denoise, ResourceAccess::SampledRead),
                    (Self::RES_TSR_HISTORY, ResourceAccess::SampledRead),
                    (Self::RES_GBUFFER_VELOCITY, ResourceAccess::SampledRead),
                    (active_depth, ResourceAccess::SampledRead),
                    (Self::RES_GBUFFER_NORMAL, ResourceAccess::SampledRead),
                ],
                writes: vec![(Self::RES_UPSCALED_HDR, ResourceAccess::StorageImageWrite)],
                enabled: true,
                is_external_raster: false,
            });
            Self::RES_UPSCALED_HDR
        } else {
            radiance_after_denoise
        };

        let mut post_writes = vec![(Self::RES_HUDLESS_COLOR, ResourceAccess::StorageImageWrite)];
        if !is_tier3 {
            post_writes.push((
                Self::RES_MC_MAIN_DEPTH,
                ResourceAccess::DepthAttachmentWrite,
            ));
        }

        // In SDR mode (`!cfg.hdr_enabled`), `PostGradeAndToneMap` ACES-tonemaps linear radiance
        // into `RES_POST_OUTPUT` / `RES_HUDLESS_COLOR`. In HDR mode (`cfg.hdr_enabled`), this pass
        // is disabled so `HdrCompose` receives unclamped linear RGBA16F radiance.
        let mut post_reads = vec![
            (radiance_after_upscale, ResourceAccess::SampledRead),
            (Self::RES_RAW_RADIANCE, ResourceAccess::SampledRead),
            (Self::RES_SHARP_RADIANCE, ResourceAccess::SampledRead),
            (active_albedo, ResourceAccess::SampledRead),
            (Self::RES_GBUFFER_NORMAL, ResourceAccess::SampledRead),
            (Self::RES_GBUFFER_MATERIAL, ResourceAccess::SampledRead),
            (Self::RES_GBUFFER_VELOCITY, ResourceAccess::SampledRead),
            (active_depth, ResourceAccess::SampledRead),
            (Self::RES_SVGF_MOMENTS, ResourceAccess::SampledRead),
        ];
        if is_rt {
            post_reads.extend([
                (
                    Self::RES_SHARC_HASH_TABLE,
                    ResourceAccess::StorageBufferReadWrite,
                ),
                (
                    Self::RES_RESTIR_RESERVOIRS,
                    ResourceAccess::StorageBufferReadWrite,
                ),
            ]);
        }
        nodes.push(GraphPassNode {
            kind: PassKind::PostGradeAndToneMap,
            name: "post_grade_and_tonemap".to_owned(),
            reads: post_reads,
            writes: post_writes,
            enabled: !cfg.hdr_enabled,
            is_external_raster: false,
        });

        if cfg.hdr_enabled {
            let mut hdr_writes = vec![(Self::RES_SWAPCHAIN_OUT, ResourceAccess::StorageImageWrite)];
            if !is_tier3 {
                hdr_writes.push((
                    Self::RES_MC_MAIN_DEPTH,
                    ResourceAccess::DepthAttachmentWrite,
                ));
            }
            nodes.push(GraphPassNode {
                kind: PassKind::HdrCompose,
                name: "hdr_scrgb_compose".to_owned(),
                reads: vec![
                    (radiance_after_upscale, ResourceAccess::SampledRead),
                    (Self::RES_UI_COLOR, ResourceAccess::SampledRead),
                ],
                writes: hdr_writes,
                enabled: true,
                is_external_raster: false,
            });
        }

        if cfg.framegen_mode != 0 {
            let fg_color_input = if cfg.hdr_enabled {
                Self::RES_SWAPCHAIN_OUT
            } else {
                Self::RES_HUDLESS_COLOR
            };
            nodes.push(GraphPassNode {
                kind: PassKind::FrameGenInterpolate,
                name: "framegen_optical_flow_present".to_owned(),
                reads: vec![
                    (Self::RES_PREV_HUDLESS_COLOR, ResourceAccess::SampledRead),
                    (fg_color_input, ResourceAccess::SampledRead),
                    (Self::RES_GBUFFER_VELOCITY, ResourceAccess::SampledRead),
                    (active_depth, ResourceAccess::SampledRead),
                    (Self::RES_UI_COLOR, ResourceAccess::SampledRead),
                ],
                writes: vec![
                    (
                        Self::RES_INTERPOLATED_FRAME,
                        ResourceAccess::StorageImageWrite,
                    ),
                    (Self::RES_SWAPCHAIN_OUT, ResourceAccess::TransferDst),
                ],
                enabled: true,
                is_external_raster: false,
            });
        }

        nodes
    }

    fn compile_and_cull_dynamic(
        &self,
        nodes: &[GraphPassNode],
        sink_outputs: &[ResourceId],
    ) -> Vec<CompiledPassStep> {
        let mut producers_of: HashMap<ResourceId, Vec<usize>> = HashMap::new();
        for (idx, node) in nodes.iter().enumerate() {
            if !node.enabled {
                continue;
            }
            for &(res, _) in &node.writes {
                producers_of.entry(res).or_default().push(idx);
            }
        }

        // Backward BFS from sink outputs to mark live passes (Dead Pass Culling)
        let mut live_nodes: HashSet<usize> = HashSet::new();
        let mut queue: VecDeque<ResourceId> = sink_outputs.iter().copied().collect();

        while let Some(res) = queue.pop_front() {
            if let Some(prod_indices) = producers_of.get(&res) {
                for &prod_idx in prod_indices {
                    if live_nodes.insert(prod_idx) {
                        for &(input_res, _) in &nodes[prod_idx].reads {
                            queue.push_back(input_res);
                        }
                    }
                }
            }
        }

        // Seed resource_state with cross-frame persistent layouts for temporal resources
        let mut resource_state: HashMap<
            ResourceId,
            (vk::PipelineStageFlags2, vk::AccessFlags2, vk::ImageLayout),
        > = HashMap::new();
        for (&res, &state) in &self.persistent_layouts {
            if self.is_resource_temporal(res) {
                resource_state.insert(res, state);
            }
        }

        let mut compiled = Vec::with_capacity(live_nodes.len());

        for (idx, node) in nodes.iter().enumerate() {
            if !live_nodes.contains(&idx) {
                continue;
            }
            let mut pre_barriers = Vec::new();

            for &(res, access) in &node.reads {
                let (dst_stage, dst_access, new_layout) = access.to_sync2_masks();
                if let Some(&(src_stage, src_access, old_layout)) = resource_state.get(&res)
                    && (old_layout != new_layout
                        || (src_access.as_raw() & vk::AccessFlags2::SHADER_WRITE.as_raw()) != 0
                        || src_access != vk::AccessFlags2::empty())
                {
                    pre_barriers.push(SynthesizedBarrier {
                        resource: res,
                        src_stage,
                        src_access,
                        dst_stage,
                        dst_access,
                        old_layout,
                        new_layout,
                    });
                }
                resource_state.insert(res, (dst_stage, dst_access, new_layout));
            }

            for &(res, access) in &node.writes {
                let (dst_stage, dst_access, new_layout) = access.to_sync2_masks();
                if let Some(&(src_stage, src_access, old_layout)) = resource_state.get(&res)
                    && (old_layout != new_layout || self.is_resource_temporal(res))
                {
                    pre_barriers.push(SynthesizedBarrier {
                        resource: res,
                        src_stage,
                        src_access,
                        dst_stage,
                        dst_access,
                        old_layout,
                        new_layout,
                    });
                }
                resource_state.insert(res, (dst_stage, dst_access, new_layout));
            }

            compiled.push(CompiledPassStep {
                kind: node.kind,
                name: node.name.clone(),
                pre_barriers,
                reads: node.reads.clone(),
                writes: node.writes.clone(),
                is_external_raster: node.is_external_raster,
            });
        }

        compiled
    }

    pub fn destroy(&mut self, ctx: &VulkanContext, retire: &RetireQueue) {
        self.destroy_internal_targets(ctx, retire);
        let dh_ubos = std::mem::take(&mut self.dh_ubo_slots);
        let dh_dummy_color = self.dh_dummy_color.take();
        let dh_dummy_depth = self.dh_dummy_depth.take();
        if dh_ubos.iter().any(Option::is_some)
            || dh_dummy_color.is_some()
            || dh_dummy_depth.is_some()
        {
            let dev = ctx.device.clone();
            retire.retire_later(move || unsafe {
                for mut b in dh_ubos.into_iter().flatten() {
                    b.destroy(&dev);
                }
                for t in [dh_dummy_color, dh_dummy_depth].into_iter().flatten() {
                    t.destroy(&dev);
                }
            });
        }
        for (_, bound) in self.bound_images.drain() {
            if bound.owned_view && bound.view != vk::ImageView::null() {
                unsafe {
                    ctx.device.destroy_image_view(bound.view, None);
                }
            }
        }
        self.resource_images.clear();
        self.bound_buffers.clear();
        self.persistent_layouts.clear();
        self.tracked_image_layouts.clear();
    }
}
