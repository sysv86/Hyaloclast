mod vsl_params;

use crate::{
    gpu::{
        VulkanContext, bindless::BindlessTextureTable, buffer::GpuBufferAllocation,
        retire::RetireQueue,
    },
    graph::{PassKind, RenderGraph, ResourceId},
    rt::{
        ZeroCopyBlasManager, sbt::ShaderBindingTable, sharc::SharcConfigUniform,
        wind::WindPushConstants,
    },
};
use ash::vk::{self, Handle};
use bytemuck::{Pod, Zeroable};
use std::{
    collections::{HashMap, HashSet},
    env,
    ffi::{CStr, CString},
    path::{Path, PathBuf},
};
use vsp_compiler::{
    CompiledShaderArtifact, ReflectedDescriptorBinding, ReflectedDescriptorType, RenderTier,
    ShaderStageKind, SpirvReflection, VspCompiler,
};

type PendingDescriptorWrite<'a> = (
    vk::WriteDescriptorSet<'a>,
    Option<usize>,
    Option<usize>,
    Option<usize>,
);
type ShadowLayerGroup = (bool, u32, u64, u64, [i32; 3], u32, u32);

/// Exact 544-byte `std140` uniform buffer matching `WorldStateCollector.kt` and `common.vsl`:
/// - `0..63`:   `cur_view_proj` (`mat4`)
/// - `64..127`: `prev_view_proj` (`mat4`)
/// - `128..191`: `inv_proj` (`mat4`)
/// - `192..255`: `inv_view` (`mat4`)
/// - `256..271`: `camera_pos_frac` (`[f32; 3]`) + `frame_index` (`i32`)
/// - `272..287`: `camera_delta` (`[f32; 3]`) + `game_time_sec` (`f32`)
/// - `288..303`: `sun_dir` (`[f32; 3]`) + `rain` (`f32`)
/// - `304..319`: `thunder` (`f32`), `eye_in_fluid` (`i32`), `depth_far` (`f32`), `held_light` (`f32`)
/// - `320..511`: `pack_params` (`[[f32; 4]; 12]` = 48 scalar floats)
/// - `512..527`: camera-relative held-light position (`vec4`)
/// - `528..535`: `pack_params_bda` (`u64` address of the VSL live pack-parameter buffer)
/// - `536..543`: alignment padding (the `std140` block is rounded to a 16-byte multiple)
///
/// The first 528 bytes are byte-for-byte the v1 WorldUBO ABI; v1 shaders may keep declaring the
/// original 528-byte block. Only the producer (`WorldStateCollector.kt`) must supply 544 bytes.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WorldUboData {
    pub cur_view_proj: [[f32; 4]; 4],
    pub prev_view_proj: [[f32; 4]; 4],
    pub inv_proj: [[f32; 4]; 4],
    pub inv_view: [[f32; 4]; 4],
    pub camera_pos_frac: [f32; 3],
    pub frame_index: i32,
    pub camera_delta: [f32; 3],
    pub game_time_sec: f32,
    pub sun_dir: [f32; 3],
    pub rain: f32,
    pub thunder: f32,
    pub eye_in_fluid: i32,
    pub depth_far: f32,
    pub held_light: f32,
    pub pack_params: [[f32; 4]; 12],
    pub held_light_position: [f32; 4],
    pub pack_params_bda: u64,
    pub pack_params_bda_pad: [u8; 8],
}

const _: () = assert!(std::mem::size_of::<WorldUboData>() == 544);
const _: () = assert!(std::mem::offset_of!(WorldUboData, pack_params) == 320);
const _: () = assert!(std::mem::offset_of!(WorldUboData, held_light_position) == 512);
const _: () = assert!(std::mem::offset_of!(WorldUboData, pack_params_bda) == 528);

/// Fixed capacity of the VSL pack-parameter BDA buffer. Reflected offsets are pack-schema
/// relative and comfortably fit this size; allocating once avoids mid-frame buffer churn.
pub const VSL_PACK_PARAM_BUFFER_BYTES: usize = 4096;
/// Byte size of the world UBO bound at set 0 / binding 0 for native and VSL stages alike.
/// VSL's `WORLD_UBO_LAYOUT` mirrors this block, including `pack_params_bda` at offset 528, so the
/// bound buffer must cover all 544 bytes. This replaces the retired 16-byte BDA carrier.
pub const WORLD_UBO_BUFFER_BYTES: usize = std::mem::size_of::<WorldUboData>();
const _: () = assert!(WORLD_UBO_BUFFER_BYTES >= 544);

impl Default for WorldUboData {
    fn default() -> Self {
        Self::zeroed()
    }
}

/// Push constants for Tier 3 `mc_depth.comp` / `depth_blit` depth reprojection into Minecraft's `mainTarget.depthTexture`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DepthReprojectPushConstants {
    pub z_scale: f32,
    pub z_offset: f32,
    pub w_scale: f32,
    pub w_offset: f32,
    pub source_size: [u32; 2],
    pub target_size: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<DepthReprojectPushConstants>() == 32);

impl DepthReprojectPushConstants {
    /// Computes non-linear clip-space depth parameters from `CameraRenderState.projectionMatrix`
    /// elements `(m22, m32, m23, m33)` and linear ray first-hit distance `range`:
    /// `view_z = -(range * 1.004 + 0.03)`, `depth = (z_scale * view_z + z_offset) / (w_scale * view_z + w_offset)`.
    pub fn evaluate_device_depth(&self, linear_hit_distance: f32) -> f32 {
        let sanitized = if !linear_hit_distance.is_finite() || linear_hit_distance > 60_000.0 {
            1.0e7
        } else {
            linear_hit_distance.abs() * 1.004 + 0.03
        };
        let view_z = -sanitized;
        let clip_z = self.z_scale * view_z + self.z_offset;
        let clip_w = self.w_scale * view_z + self.w_offset;
        if clip_w.abs() > 1.0e-12 {
            (clip_z / clip_w).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }
}

/// Push constants for `hdr_compose.comp` scRGB / HDR10 swapchain output composition (`16` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct HdrComposePushConstants {
    pub peak_brightness_nits: f32,
    pub paper_white_nits: f32,
    pub output_mode: u32,
    pub pad0: u32,
}

impl Default for HdrComposePushConstants {
    fn default() -> Self {
        Self {
            peak_brightness_nits: 1000.0,
            paper_white_nits: 200.0,
            output_mode: 1, // scRGB 16-bit float
            pad0: 0,
        }
    }
}

/// Push constants for `svgf_atrous.comp` edge-stopping wavelet denoiser (`16` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct AtrousPushConstants {
    pub step_size: i32,
    pub phi_color: f32,
    pub phi_normal: f32,
    pub phi_depth: f32,
}

impl Default for AtrousPushConstants {
    fn default() -> Self {
        Self {
            step_size: 1,
            phi_color: 4.0,
            phi_normal: 128.0,
            phi_depth: 1.0,
        }
    }
}

/// Push constants for `tsr_upscale.comp` Temporal Super Resolution (`16` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TsrPushConstants {
    pub jitter_offset: [f32; 2],
    pub history_weight: f32,
    pub sharpness: f32,
}

impl Default for TsrPushConstants {
    fn default() -> Self {
        Self {
            jitter_offset: [0.0, 0.0],
            history_weight: 0.90,
            sharpness: 0.75,
        }
    }
}

/// Per-frame cascaded-shadow uniform block (288 bytes `std140`): four column-major orthographic
/// `viewProj` matrices derived from the sun direction, split distances, and per-cascade texel sizes.
/// Matches `csm_shadow.vsh` Set 0 binding 1 and `compute_lighting.comp` Set 0 binding 7 (`uCsm`).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CsmShadowUBO {
    pub cascade_view_proj: [[[f32; 4]; 4]; 4],
    pub split_distances: [f32; 4],
    pub texel_sizes: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<CsmShadowUBO>() == 288);

impl CsmShadowUBO {
    /// Builds four column-major orthographic cascade matrices aligned to `sun_dir` with `[0, 1]`
    /// Vulkan depth mapping and positive per-cascade texel sizes.
    pub fn from_sun_dir(sun_dir: [f32; 3], shadow_map_size: f32) -> Self {
        Self::from_sun_dir_with_config(sun_dir, shadow_map_size, 384.0)
    }

    pub fn from_sun_dir_with_config(
        sun_dir: [f32; 3],
        shadow_map_size: f32,
        max_shadow_dist_blocks: f32,
    ) -> Self {
        let dist_scale = (max_shadow_dist_blocks.clamp(64.0, 1024.0) / 384.0).clamp(0.25, 2.5);
        let mut len =
            (sun_dir[0] * sun_dir[0] + sun_dir[1] * sun_dir[1] + sun_dir[2] * sun_dir[2]).sqrt();
        if !len.is_finite() || len < 1.0e-6 {
            len = 1.0;
        }
        let dir = [sun_dir[0] / len, sun_dir[1] / len, sun_dir[2] / len];
        // Right-handed look-at basis looking from the sun toward the camera origin (-dir).
        let zaxis = [-dir[0], -dir[1], -dir[2]];
        let up = if zaxis[1].abs() > 0.99 {
            [0.0, 0.0, 1.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let cross = |a: [f32; 3], b: [f32; 3]| -> [f32; 3] {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let xaxis = cross(up, zaxis);
        let xlen = (xaxis[0] * xaxis[0] + xaxis[1] * xaxis[1] + xaxis[2] * xaxis[2])
            .sqrt()
            .max(1e-6);
        let xaxis = [xaxis[0] / xlen, xaxis[1] / xlen, xaxis[2] / xlen];
        let yaxis = cross(zaxis, xaxis);
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

        // Cascade half-extents cover split_distances ([16, 48, 128, 384] * dist_scale) with
        // blend margin so distant shadows in Cascade 3 never clip before split_distances[3].
        const BASE_EXTENTS: [f32; 4] = [24.0, 64.0, 160.0, 416.0];
        let mut matrices = [[[0.0f32; 4]; 4]; 4];
        for (cascade, mat) in matrices.iter_mut().enumerate() {
            let extent = BASE_EXTENTS[cascade] * dist_scale;
            // Guarantee at least +-256 blocks along the sun ray so tall cliffs/trees never clip
            // the near plane in Cascade 0/1 even at 12-chunk shadow distance.
            let z_range = (extent * 4.0).max(256.0);
            let center = [0.0f32, 0.0f32, 0.0f32];
            // Column-major storage (`mat[col][row]`) matching GLSL `mat4`:
            // x_clip = dot(xaxis, pos - center) / extent
            // y_clip = dot(yaxis, pos - center) / extent
            // z_clip = 0.5 + dot(zaxis, pos - center) / (2.0 * z_range) in [0, 1]
            *mat = [
                [
                    xaxis[0] / extent,
                    yaxis[0] / extent,
                    zaxis[0] / (2.0 * z_range),
                    0.0,
                ],
                [
                    xaxis[1] / extent,
                    yaxis[1] / extent,
                    zaxis[1] / (2.0 * z_range),
                    0.0,
                ],
                [
                    xaxis[2] / extent,
                    yaxis[2] / extent,
                    zaxis[2] / (2.0 * z_range),
                    0.0,
                ],
                [
                    -dot(xaxis, center) / extent,
                    -dot(yaxis, center) / extent,
                    0.5 - dot(zaxis, center) / (2.0 * z_range),
                    1.0,
                ],
            ];
        }
        let inv_res = 1.0 / shadow_map_size.max(1.0);
        Self {
            cascade_view_proj: matrices,
            split_distances: [
                16.0 * dist_scale,
                48.0 * dist_scale,
                128.0 * dist_scale,
                384.0 * dist_scale,
            ],
            texel_sizes: [inv_res, inv_res * 1.25, inv_res * 1.6, inv_res * 2.1],
        }
    }

    /// Tests whether a camera-relative axis-aligned bounding box `(min_rel, max_rel)` intersects
    /// the orthographic light frustum of `cascade` (`0..4`), including an upstream shadow-caster
    /// extrusion along `-sun_dir` (`z_clip in [-0.25, 1.05]`).
    /// Builds four texel-snapped orthographic cascade matrices anchored to `camera_pos`
    /// in world-space light coordinates (eliminates shadow swimming during camera movement).
    pub fn from_sun_dir_snapped(
        sun_dir: [f32; 3],
        shadow_map_size: f32,
        max_shadow_dist_blocks: f32,
        camera_pos: [f64; 3],
    ) -> Self {
        let mut ubo =
            Self::from_sun_dir_with_config(sun_dir, shadow_map_size, max_shadow_dist_blocks);
        let dist_scale = (max_shadow_dist_blocks.clamp(64.0, 1024.0) / 384.0).clamp(0.25, 2.5);
        let res_f = f64::from(shadow_map_size.max(1.0));
        let mut len =
            (sun_dir[0] * sun_dir[0] + sun_dir[1] * sun_dir[1] + sun_dir[2] * sun_dir[2]).sqrt();
        if !len.is_finite() || len < 1.0e-6 {
            len = 1.0;
        }
        let dir = [
            f64::from(sun_dir[0] / len),
            f64::from(sun_dir[1] / len),
            f64::from(sun_dir[2] / len),
        ];
        let zaxis = [-dir[0], -dir[1], -dir[2]];
        let up = if zaxis[1].abs() > 0.99 {
            [0.0, 0.0, 1.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let cross = |a: [f64; 3], b: [f64; 3]| -> [f64; 3] {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let xraw = cross(up, zaxis);
        let xlen = (xraw[0] * xraw[0] + xraw[1] * xraw[1] + xraw[2] * xraw[2])
            .sqrt()
            .max(1e-6);
        let xaxis = [xraw[0] / xlen, xraw[1] / xlen, xraw[2] / xlen];
        let yaxis = cross(zaxis, xaxis);
        let dot64 = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let cam_x_ls = dot64(xaxis, camera_pos);
        let cam_y_ls = dot64(yaxis, camera_pos);

        const BASE_EXTENTS: [f64; 4] = [24.0, 64.0, 160.0, 416.0];
        for (cascade, mat) in ubo.cascade_view_proj.iter_mut().enumerate() {
            let extent = BASE_EXTENTS[cascade] * f64::from(dist_scale);
            let texel = (2.0 * extent) / res_f;
            let frac_x = cam_x_ls - (cam_x_ls / texel).floor() * texel;
            let frac_y = cam_y_ls - (cam_y_ls / texel).floor() * texel;
            mat[3][0] = (frac_x / extent) as f32;
            mat[3][1] = (frac_y / extent) as f32;
        }
        ubo
    }

    pub fn is_aabb_visible_in_cascade(
        &self,
        cascade: usize,
        min_rel: [f32; 3],
        max_rel: [f32; 3],
    ) -> bool {
        let Some(mat) = self.cascade_view_proj.get(cascade) else {
            return false;
        };
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut min_z = f32::INFINITY;
        let mut max_z = f32::NEG_INFINITY;

        for ix in 0..2 {
            let x = if ix == 0 { min_rel[0] } else { max_rel[0] };
            for iy in 0..2 {
                let y = if iy == 0 { min_rel[1] } else { max_rel[1] };
                for iz in 0..2 {
                    let z = if iz == 0 { min_rel[2] } else { max_rel[2] };
                    let cx = mat[0][0] * x + mat[1][0] * y + mat[2][0] * z + mat[3][0];
                    let cy = mat[0][1] * x + mat[1][1] * y + mat[2][1] * z + mat[3][1];
                    let cz = mat[0][2] * x + mat[1][2] * y + mat[2][2] * z + mat[3][2];
                    min_x = min_x.min(cx);
                    max_x = max_x.max(cx);
                    min_y = min_y.min(cy);
                    max_y = max_y.max(cy);
                    min_z = min_z.min(cz);
                    max_z = max_z.max(cz);
                }
            }
        }

        max_x >= -1.05
            && min_x <= 1.05
            && max_y >= -1.05
            && min_y <= 1.05
            && max_z >= -0.25
            && min_z <= 1.05
    }
}

/// Push constants for `optical_flow_fg.comp` motion-compensated frame interpolation (`16` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct OpticalFlowPushConstants {
    pub interpolation_factor: f32,
    pub depth_rejection_scale: f32,
    pub width: u32,
    pub height: u32,
}

/// Compiled compute `VkPipeline` along with reflected descriptor set layouts and per-frame
/// allocated `VkDescriptorSet`s (`3` frames-in-flight).
pub struct ComputePipelineBundle {
    pub stage_name: String,
    pub shader_module: vk::ShaderModule,
    pub set0_layout: vk::DescriptorSetLayout,
    pub set1_layout: vk::DescriptorSetLayout,
    pub uses_external_bindless_set1: bool,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_sets_set0: [vk::DescriptorSet; 3],
    pub descriptor_sets_set1: [vk::DescriptorSet; 3],
    pub atrous_reverse_sets: Option<[vk::DescriptorSet; 3]>,
    pub workgroup_size: [u32; 3],
    pub push_constant_size: u32,
    pub reflection: SpirvReflection,
}

/// Compiled graphics MRT `VkPipeline` (`gbuffer_terrain.vsh` + `gbuffer_terrain.fsh`).
pub struct GraphicsPipelineBundle {
    pub name: String,
    pub vertex_module: vk::ShaderModule,
    pub fragment_module: vk::ShaderModule,
    pub set0_layout: vk::DescriptorSetLayout,
    pub set1_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub color_attachment_count: u32,
    /// Per-frame descriptor sets for set 0 (WorldUBO + CsmShadowUBO) and set 1 (block atlas).
    /// Empty when the pipeline does not declare those descriptors.
    pub descriptor_sets_set0: [vk::DescriptorSet; 3],
    pub descriptor_sets_set1: [vk::DescriptorSet; 3],
}

/// Vertex input layout for dynamic-rendering depth pipelines (`depth_blit`, `csm_shadow`, `csm_shadow_28b`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DepthVertexInput {
    None,
    Terrain40B,
    Terrain28B,
}

/// Push constant consumed by `csm_shadow.vsh` (16 bytes: `ivec3 RegionOrigin`, `uint cascadeIndex`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CsmShadowPush {
    pub region_origin: [i32; 3],
    pub cascade_index: u32,
}

/// Compiled hardware Ray Tracing `VkPipeline` (`VK_KHR_ray_tracing_pipeline`) + `ShaderBindingTable`.
pub struct RayTracingPipelineBundle {
    pub set0_layout: vk::DescriptorSetLayout,
    pub set1_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_sets_set0: [vk::DescriptorSet; 3],
    pub descriptor_sets_set1: [vk::DescriptorSet; 3],
    pub sbt: ShaderBindingTable,
    pub reflection: SpirvReflection,
}

/// Manages triple-buffered GPU `WorldUBO` uniform buffers, compiled `VkShaderModule`s,
/// `VkDescriptorPool`, `VkDescriptorSetLayout`s, `VkPipelineLayout`s, and real `VkPipeline`s
/// (Compute, Graphics MRT, and Hardware Ray Tracing) with `VkSpecializationInfo` support.
#[derive(Default)]
pub struct GpuPipelineManager {
    pub world_ubo_ring: [Option<GpuBufferAllocation>; 3],
    /// Valid zeroed SHaRC UBOs for raster passes that still declare set 1 / binding 2.
    pub sharc_fallback_ubo_ring: [Option<GpuBufferAllocation>; 3],
    pub descriptor_pool: vk::DescriptorPool,
    pub default_sampler: vk::Sampler,
    pub nearest_sampler: vk::Sampler,
    pub compiled_artifacts: HashMap<String, CompiledShaderArtifact>,
    /// VSL 2.0 units merged into the next pack compile, before the single pipeline rebuild.
    pub pending_vsl_artifacts: Vec<CompiledShaderArtifact>,
    /// `used_params` merged from VSL reflection sidecars for the active pack.
    pub vsl_param_bindings: Vec<vsl_params::VslUsedParam>,
    /// Host-visible, device-addressable buffer holding live VSL pack-parameter values.
    pub vsl_param_buffer: Option<GpuBufferAllocation>,
    /// Stage names of VSL units in the active pack. VSL units bind the full world UBO at
    /// set 0 / binding 0; the pack-parameter BDA lives at offset 528 of that same buffer.
    pub vsl_param_stages: HashSet<String>,
    pub shader_modules: HashMap<String, vk::ShaderModule>,
    pub compute_pipelines: HashMap<String, ComputePipelineBundle>,
    pub graphics_pipelines: HashMap<String, GraphicsPipelineBundle>,
    pub rt_pipeline: Option<RayTracingPipelineBundle>,
    pub rt_uses_reservoirs: bool,
    pub rt_sharc_enabled: bool,
    pub active_spec_constants: Vec<(u32, u32)>,
    pub last_overlay_depth_params: Option<DepthReprojectPushConstants>,
    pub hdr_push_constants: HdrComposePushConstants,
    pub tsr_push_constants: TsrPushConstants,
    pub last_compile_error: Option<String>,
    /// Set 3 layout for `[resources.*]` pack textures, supplied by the engine at compile time.
    pub pack_descriptor_set_layout: vk::DescriptorSetLayout,
    /// Per-layer single-layer views of the 2048²×4 CSM depth array, cached by backing image.
    pub csm_layer_views: [vk::ImageView; 4],
    pub csm_layer_view_image: vk::Image,
    /// Triple-buffered host-visible CSM uniform block bound at compute_lighting.comp b7.
    pub csm_ubo_slots: [Option<GpuBufferAllocation>; 3],
    pub csm_indirect_slots: [Option<GpuBufferAllocation>; 3],
    pub retired_csm_views: Vec<vk::ImageView>,
    pub retired_csm_ibos: Vec<GpuBufferAllocation>,
    pub pipeline_cache: vk::PipelineCache,
    pub pipeline_cache_key: Option<[u8; 32]>,
    pub descriptor_sig_cache: HashMap<(String, usize), u64>,
    pub camera_pos_world: [f64; 3],
    pub csm_ubo_sun_dir: [f32; 3],
    /// Integer block coordinates (`floor(camera_pos)`) of the active camera for CSM `RegionOrigin`.
    pub camera_block_pos: [i32; 3],
    /// Fallback `uint32` quad index buffer (`0, 1, 2, 2, 3, 0` per quad) for CSM indexed draws.
    pub csm_quad_ibo: Option<GpuBufferAllocation>,
    pub csm_quad_capacity: u32,
    pub shadow_resolution: u32,
    pub shadow_distance_chunks: u32,
    pub csm_ubo_shadow_res: u32,
    pub csm_ubo_shadow_dist: u32,
    pub csm_bound_world_bufs: [vk::Buffer; 3],
    pub csm_bound_ubo_bufs: [vk::Buffer; 3],
    pub csm_bound_atlas_views: [vk::ImageView; 3],
    pub csm_bound_atlas_samplers: [vk::Sampler; 3],
    pub csm_frame_counter: u64,
    pub csm_cascade_last_sun_dir: [[f32; 3]; 4],
    pub csm_cascade_last_cam_pos: [[f64; 3]; 4],
    pub csm_cascade_valid: [bool; 4],
    pub csm_inactive_cleared: [bool; 4],
    pub csm_last_slice_count: usize,
    pub csm_last_geometry_sig: u64,
    pub csm_cached_cascade_vp: [[[f32; 4]; 4]; 4],
}

#[inline]
pub(crate) fn fnv1a_init() -> u64 {
    0xcbf2_9ce4_8422_2325
}

#[inline]
pub(crate) fn fnv1a_mix_u64(mut hash: u64, val: u64) -> u64 {
    for byte in val.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

impl GpuPipelineManager {
    fn pipeline_cache_disk_path() -> std::path::PathBuf {
        if let Some(dir) = std::env::var_os("VS_CACHE_DIR") {
            std::path::PathBuf::from(dir).join("vk_pipeline_cache.bin")
        } else {
            std::path::PathBuf::from("config/vulkanshaders/cache/vk_pipeline_cache.bin")
        }
    }

    pub fn ensure_pipeline_cache(&mut self, ctx: &VulkanContext) -> vk::PipelineCache {
        if self.pipeline_cache != vk::PipelineCache::null() {
            return self.pipeline_cache;
        }
        if let Some(key) = self.pipeline_cache_key {
            let disk = PersistentPipelineDiskCache::default();
            if let Ok(cache) = disk.create_vk_pipeline_cache(ctx, &key) {
                self.pipeline_cache = cache;
                return self.pipeline_cache;
            }
        }
        let initial_data = std::fs::read(Self::pipeline_cache_disk_path()).unwrap_or_default();
        let create_info = if initial_data.is_empty() {
            vk::PipelineCacheCreateInfo::default()
        } else {
            vk::PipelineCacheCreateInfo::default().initial_data(&initial_data)
        };
        if let Ok(cache) = unsafe { ctx.device.create_pipeline_cache(&create_info, None) } {
            self.pipeline_cache = cache;
        }
        self.pipeline_cache
    }

    /// Initializes or loads a keyed `VkPipelineCache` from disk using `blake3(pack_hash || driver_uuid || pipeline_cache_uuid || spec_constants)`.
    pub fn ensure_pipeline_cache_for_key(
        &mut self,
        ctx: &VulkanContext,
        key: [u8; 32],
    ) -> vk::PipelineCache {
        self.pipeline_cache_key = Some(key);
        self.ensure_pipeline_cache(ctx)
    }

    /// Persists the active `VkPipelineCache` blob to disk (keyed if `pipeline_cache_key` is set).
    pub fn save_pipeline_cache_to_disk(&self, ctx: &VulkanContext) {
        if self.pipeline_cache == vk::PipelineCache::null() {
            return;
        }
        if let Some(key) = self.pipeline_cache_key {
            let disk = PersistentPipelineDiskCache::default();
            let _ = disk.flush_vk_pipeline_cache(ctx, self.pipeline_cache, &key);
        } else if let Ok(data) = unsafe { ctx.device.get_pipeline_cache_data(self.pipeline_cache) }
            && !data.is_empty()
        {
            let cache_path = Self::pipeline_cache_disk_path();
            if let Some(parent) = cache_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(&cache_path, &data);
        }
    }

    /// Flushes deferred CSM image views / IBOs into `retire_queue` so they are never destroyed
    /// immediately while GPU command buffers are in flight.
    pub fn flush_deferred_to_retire(&mut self, ctx: &VulkanContext, retire_queue: &RetireQueue) {
        let views = std::mem::take(&mut self.retired_csm_views);
        let ibos = std::mem::take(&mut self.retired_csm_ibos);
        if !views.is_empty() || !ibos.is_empty() {
            let dev = ctx.device.clone();
            retire_queue.retire_later(move || unsafe {
                for v in views {
                    dev.destroy_image_view(v, None);
                }
                for mut b in ibos {
                    b.destroy(&dev);
                }
            });
        }
    }
    pub fn new() -> Self {
        Self::default()
    }

    /// Merges SPIR-V descriptor reflections across multiple pipeline stages (e.g. VS+FS or
    /// RGEN+RMISS+RCHIT+RAHIT) so every binding declared by any stage is present in the layout.
    pub fn merge_reflections(reflections: &[&SpirvReflection]) -> SpirvReflection {
        let mut merged = SpirvReflection::default();
        for refl in reflections {
            if merged.workgroup_size == [0, 0, 0] && refl.workgroup_size != [0, 0, 0] {
                merged.workgroup_size = refl.workgroup_size;
            }
            merged.push_constant_size = merged.push_constant_size.max(refl.push_constant_size);
            merged.uses_ray_query |= refl.uses_ray_query;
            merged.uses_buffer_device_address |= refl.uses_buffer_device_address;
            for b in &refl.bindings {
                if let Some(existing) = merged
                    .bindings
                    .iter_mut()
                    .find(|m| m.set == b.set && m.binding == b.binding)
                {
                    existing.count = existing.count.max(b.count);
                    existing.writable |= b.writable;
                } else {
                    merged.bindings.push(b.clone());
                }
            }
        }
        merged.bindings.sort_by_key(|b| (b.set, b.binding));
        merged
    }

    fn to_vk_descriptor_type(ty: ReflectedDescriptorType) -> vk::DescriptorType {
        match ty {
            ReflectedDescriptorType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
            ReflectedDescriptorType::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
            ReflectedDescriptorType::CombinedImageSampler => {
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER
            }
            ReflectedDescriptorType::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
            ReflectedDescriptorType::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
            ReflectedDescriptorType::Sampler => vk::DescriptorType::SAMPLER,
            ReflectedDescriptorType::AccelerationStructure => {
                vk::DescriptorType::ACCELERATION_STRUCTURE_KHR
            }
        }
    }

    fn ensure_descriptor_pool_and_sampler(
        &mut self,
        ctx: &VulkanContext,
    ) -> Result<(), vk::Result> {
        unsafe {
            if self.default_sampler == vk::Sampler::null() {
                let sampler_info = vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(16.0);
                self.default_sampler = ctx.device.create_sampler(&sampler_info, None)?;
            }
            if self.nearest_sampler == vk::Sampler::null() {
                let nearest_info = vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(16.0);
                self.nearest_sampler = ctx.device.create_sampler(&nearest_info, None)?;
            }

            if self.descriptor_pool == vk::DescriptorPool::null() {
                let mut pool_sizes = vec![
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: 128,
                    },
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::STORAGE_BUFFER,
                        descriptor_count: 128,
                    },
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: 512,
                    },
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::SAMPLED_IMAGE,
                        descriptor_count: 128,
                    },
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::SAMPLER,
                        descriptor_count: 64,
                    },
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::STORAGE_IMAGE,
                        descriptor_count: 256,
                    },
                ];
                if ctx.capabilities.acceleration_structure {
                    pool_sizes.push(vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                        descriptor_count: 64,
                    });
                }

                let pool_info = vk::DescriptorPoolCreateInfo::default()
                    .max_sets(256)
                    .pool_sizes(&pool_sizes);
                self.descriptor_pool = ctx.device.create_descriptor_pool(&pool_info, None)?;
            }
        }
        Ok(())
    }

    /// Uploads `WorldUboData` into the triple-buffered GPU uniform buffer for `frame_slot`.
    pub fn upload_world_ubo(
        &mut self,
        ctx: &VulkanContext,
        frame_slot: usize,
        ubo: &WorldUboData,
    ) -> Result<vk::DeviceAddress, vk::Result> {
        let slot = frame_slot % 3;
        if self.world_ubo_ring[slot].is_none() {
            let alloc = GpuBufferAllocation::allocate_host_visible_bda(
                ctx,
                WORLD_UBO_BUFFER_BYTES as vk::DeviceSize,
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
            )?;
            self.world_ubo_ring[slot] = Some(alloc);
        }
        if self.sharc_fallback_ubo_ring[slot].is_none() {
            let alloc = GpuBufferAllocation::allocate_host_visible_bda(
                ctx,
                std::mem::size_of::<SharcConfigUniform>() as vk::DeviceSize,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?;
            self.sharc_fallback_ubo_ring[slot] = Some(alloc);
        }
        let mut fallback = SharcConfigUniform::zeroed();
        fallback.camera_position[..3]
            .copy_from_slice(&self.camera_pos_world.map(|value| value as f32));
        if let Some(buf) = &mut self.sharc_fallback_ubo_ring[slot] {
            buf.write_bytes(0, bytemuck::bytes_of(&fallback));
        }
        // VSL pack parameters live behind a device address, not inline in the fixed pack_params
        // array. Publish the address in the appended WorldUBO field at offset 528; VSL stages bind
        // this same full world UBO at set 0 / binding 0, so no separate BDA carrier is needed.
        let mut ubo = *ubo;
        let mut params_addr = 0u64;
        if !self.vsl_param_bindings.is_empty() && self.vsl_param_buffer.is_none() {
            self.vsl_param_buffer = Some(GpuBufferAllocation::allocate_host_visible_bda(
                ctx,
                VSL_PACK_PARAM_BUFFER_BYTES as vk::DeviceSize,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?);
        }
        if let Some(buf) = self.vsl_param_buffer.as_mut() {
            let bytes = vsl_params::pack_param_bytes(&self.vsl_param_bindings, &ubo.pack_params);
            buf.write_bytes(0, &bytes);
            params_addr = buf.device_address;
        }
        ubo.pack_params_bda = params_addr;

        let buf = self.world_ubo_ring[slot].as_mut().unwrap();
        buf.write_bytes(0, bytemuck::bytes_of(&ubo));
        Ok(buf.device_address)
    }

    pub fn world_ubo_buffer_for_slot(&self, frame_slot: usize) -> vk::Buffer {
        self.world_ubo_ring[frame_slot % 3]
            .as_ref()
            .map(|b| b.buffer)
            .unwrap_or(vk::Buffer::null())
    }

    /// Updates compile-time `VkSpecializationInfo` overrides (`(constant_id, raw_u32_bits)`) and
    /// recreates all active `VkPipeline`s cleanly when specialization settings change.
    pub fn update_specialization_constants(
        &mut self,
        ctx: &VulkanContext,
        bindless_layout: vk::DescriptorSetLayout,
        retire_queue: &RetireQueue,
        spec_overrides: Vec<(u32, u32)>,
    ) {
        self.update_specialization_constants_with_layouts(
            ctx,
            bindless_layout,
            self.pack_descriptor_set_layout,
            retire_queue,
            spec_overrides,
        );
    }

    /// Variant that also installs the Set 3 pack-resource layout before rebuilding pipelines.
    pub fn update_specialization_constants_with_layouts(
        &mut self,
        ctx: &VulkanContext,
        bindless_layout: vk::DescriptorSetLayout,
        pack_layout: vk::DescriptorSetLayout,
        retire_queue: &RetireQueue,
        spec_overrides: Vec<(u32, u32)>,
    ) {
        self.pack_descriptor_set_layout = pack_layout;
        if self.active_spec_constants == spec_overrides {
            return;
        }
        self.active_spec_constants = spec_overrides;
        if !self.compiled_artifacts.is_empty() {
            let _ = self.rebuild_all_pipelines_from_artifacts(ctx, bindless_layout, retire_queue);
        }
    }

    /// Compiles all `.vsh`, `.fsh`, `.comp`, `.rgen`, `.rchit`, `.rmiss`, and `.rahit` stages inside
    /// `pack_dir` via `VspCompiler::compile_pack_dir` and creates real `VkShaderModule`,
    /// `VkDescriptorSetLayout`, `VkPipelineLayout`, and `VkPipeline` objects on `ctx.device`.
    pub fn compile_pack_directory(
        &mut self,
        ctx: &VulkanContext,
        compiler: &mut VspCompiler,
        pack_dir: &Path,
        tier: RenderTier,
    ) -> usize {
        self.compile_pack_directory_with_bindless(
            ctx,
            compiler,
            pack_dir,
            tier,
            vk::DescriptorSetLayout::null(),
            None,
        )
    }

    /// Compiles the pack and installs both the Set 2 bindless layout and the Set 3 pack-resource
    /// layout before constructing all pipelines.
    // Carries both descriptor layouts and optional retirement state into compilation.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_pack_directory_with_layouts(
        &mut self,
        ctx: &VulkanContext,
        compiler: &mut VspCompiler,
        pack_dir: &Path,
        tier: RenderTier,
        bindless_layout: vk::DescriptorSetLayout,
        pack_layout: vk::DescriptorSetLayout,
        retire_queue: Option<&RetireQueue>,
    ) -> usize {
        self.pack_descriptor_set_layout = pack_layout;
        self.compile_pack_directory_with_bindless(
            ctx,
            compiler,
            pack_dir,
            tier,
            bindless_layout,
            retire_queue,
        )
    }

    pub fn compile_pack_directory_with_bindless(
        &mut self,
        ctx: &VulkanContext,
        compiler: &mut VspCompiler,
        pack_dir: &Path,
        tier: RenderTier,
        bindless_layout: vk::DescriptorSetLayout,
        retire_queue: Option<&RetireQueue>,
    ) -> usize {
        let hw_caps = vsp_compiler::preprocessor::HardwareShaderCaps {
            ser_ext: crate::gpu::ser_reordering_enabled()
                && ctx.capabilities.supports_ser_ext
                && ctx.capabilities.ser_reordering_active,
            ser_nv: crate::gpu::ser_reordering_enabled()
                && ctx.capabilities.supports_ser_nv
                && ctx.capabilities.ser_reordering_active,
            ada_or_newer: ctx.capabilities.gpu_arch.is_ada_or_newer(),
            high_end_gpu: ctx.capabilities.gpu_tier_class.is_high_end(),
            opacity_micromap: ctx.capabilities.supports_opacity_micromap,
        };
        let artifacts = match compiler.compile_pack_dir_with_caps(pack_dir, tier, hw_caps) {
            Ok(list) => {
                self.last_compile_error = None;
                let guard = crate::crash::global_device_lost_guard();
                if guard.snapshot().reason == crate::crash::RecoveryReason::ShaderCompileFailure {
                    guard.reset();
                }
                list
            }
            Err(err) => {
                let msg = err.to_string();
                self.last_compile_error = Some(msg.clone());
                let diag = crate::crash::ShaderCompileDiagnostic::parse_from_compiler_error(
                    &pack_dir.to_string_lossy(),
                    &msg,
                );
                crate::crash::global_device_lost_guard().record_shader_compile_failure(diag);
                return 0;
            }
        };

        self.compiled_artifacts.clear();
        for artifact in artifacts {
            self.compiled_artifacts
                .insert(artifact.stage_name.clone(), artifact);
        }
        let vsl_dirs = [
            pack_dir.join("vsl-units"),
            env::var_os("VSL_UNIT_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("shaderpacks/vsl-units")),
        ];
        let mut vsl_bindings = Vec::new();
        let mut vsl_stages = HashSet::new();
        for artifact in &self.pending_vsl_artifacts {
            vsl_stages.insert(artifact.stage_name.clone());
            vsl_bindings.extend(vsl_params::load_used_params_from_dirs(
                vsl_dirs.iter().map(PathBuf::as_path),
                &artifact.stage_name,
            ));
        }
        self.vsl_param_bindings = vsl_params::merge_used_params(vsl_bindings);
        self.vsl_param_stages = vsl_stages;
        if !self.vsl_param_bindings.is_empty() {
            let dirs = vsl_dirs
                .iter()
                .map(|dir| dir.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!(
                "[vulkanshaders-core] VSL live pack params: {} from {}",
                self.vsl_param_bindings.len(),
                dirs
            );
        }
        for artifact in self.pending_vsl_artifacts.drain(..) {
            self.compiled_artifacts
                .insert(artifact.stage_name.clone(), artifact);
        }

        // Pipeline rebuild replaces shader modules and descriptor sets that in-flight frames may
        // still reference. Drain the GPU before doing so; otherwise the first frame submitted
        // after a tier switch can stall the queue until RenderPearl's 5 s semaphore timeout.
        if let Err(err) = unsafe { ctx.device.device_wait_idle() } {
            tracing::warn!(?err, "device_wait_idle before pipeline rebuild failed");
        }

        if let Some(rq) = retire_queue {
            self.rebuild_all_pipelines_from_artifacts(ctx, bindless_layout, rq)
        } else {
            let temp_rq = RetireQueue::new(1);
            let count = self.rebuild_all_pipelines_from_artifacts(ctx, bindless_layout, &temp_rq);
            temp_rq.drain_all();
            count
        }
    }

    fn build_specialization_entries(
        spec_values: &[(u32, u32)],
    ) -> (Vec<vk::SpecializationMapEntry>, Vec<u8>) {
        let mut entries = Vec::with_capacity(spec_values.len());
        let mut data = Vec::with_capacity(spec_values.len() * 4);
        for (idx, &(constant_id, raw_bits)) in spec_values.iter().enumerate() {
            entries.push(vk::SpecializationMapEntry {
                constant_id,
                offset: (idx * 4) as u32,
                size: 4,
            });
            data.extend_from_slice(&raw_bits.to_le_bytes());
        }
        (entries, data)
    }

    pub fn rebuild_all_pipelines_from_artifacts(
        &mut self,
        ctx: &VulkanContext,
        bindless_layout: vk::DescriptorSetLayout,
        retire_queue: &RetireQueue,
    ) -> usize {
        // Keep in-flight sets in their old pool while the replacement generation is built.
        self.retire_active_pipelines(ctx, retire_queue);
        self.rt_uses_reservoirs = false;
        self.rt_sharc_enabled = false;
        if self.ensure_descriptor_pool_and_sampler(ctx).is_err() {
            return 0;
        }

        let entry_name = c"main";
        let (spec_entries, spec_bytes) =
            Self::build_specialization_entries(&self.active_spec_constants);
        let spec_info = vk::SpecializationInfo::default()
            .map_entries(&spec_entries)
            .data(&spec_bytes);

        let mut compiled_count = 0usize;
        let artifacts: Vec<CompiledShaderArtifact> =
            self.compiled_artifacts.values().cloned().collect();

        for artifact in &artifacts {
            if artifact.reflection.uses_ray_query && !ctx.capabilities.ray_query {
                continue;
            }
            if artifact.reflection.uses_buffer_device_address
                && !ctx.capabilities.buffer_device_address
            {
                continue;
            }
            if artifact.stage_kind.is_ray_tracing_stage() && !ctx.capabilities.ray_tracing_pipeline
            {
                continue;
            }

            let create_info = vk::ShaderModuleCreateInfo::default().code(&artifact.spirv_words);
            let module = match unsafe { ctx.device.create_shader_module(&create_info, None) } {
                Ok(m) => m,
                Err(err) => {
                    let msg = format!(
                        "Failed to create VkShaderModule '{}': {:?}",
                        artifact.stage_name, err
                    );
                    self.last_compile_error = Some(msg);
                    continue;
                }
            };
            self.shader_modules
                .insert(artifact.stage_name.clone(), module);
            compiled_count += 1;

            if artifact.stage_kind == ShaderStageKind::Compute {
                let entry = artifact
                    .reflection
                    .entry_points
                    .first()
                    .and_then(|name| CString::new(name.as_str()).ok());
                let compute_entry_name: &CStr = entry.as_deref().unwrap_or(entry_name);
                match self.build_compute_pipeline_bundle(
                    ctx,
                    artifact,
                    module,
                    compute_entry_name,
                    &spec_info,
                    bindless_layout,
                ) {
                    Ok(bundle) => {
                        self.compute_pipelines
                            .insert(artifact.stage_name.clone(), bundle);
                    }
                    Err(err) => {
                        let msg = format!(
                            "Failed to build compute pipeline '{}': {:?}",
                            artifact.stage_name, err
                        );
                        self.last_compile_error = Some(msg);
                    }
                }
            }
        }
        let mut built_names: Vec<&String> = self.compute_pipelines.keys().collect();
        built_names.sort();
        // CSM shadow-depth pipelines (40B IGNIS_TERRAIN_40B and 28B vanilla terrain vertex input):
        // dedicated depth-only BACK-culled variants for opaque solid terrain (layer 0) and
        // alpha-tested NONE-culled variants for cutout terrain (layer 1), plus Tier 3 depth_blit.
        if let (Some(&csv), Some(vs_art)) = (
            self.shader_modules.get("csm_shadow.vsh"),
            self.compiled_artifacts.get("csm_shadow.vsh"),
        ) {
            if let Ok(p) = self.build_dynamic_depth_pipeline(
                ctx,
                csv,
                None,
                &vs_art.reflection,
                None,
                bindless_layout,
                "csm_shadow_solid",
                vk::CompareOp::LESS,
                vk::CullModeFlags::BACK,
                std::mem::size_of::<CsmShadowPush>() as u32,
                DepthVertexInput::Terrain40B,
                &spec_info,
            ) {
                self.graphics_pipelines
                    .insert("csm_shadow_solid".to_owned(), p);
            }
            if let (Some(&csf), Some(fs_art)) = (
                self.shader_modules.get("csm_shadow.fsh"),
                self.compiled_artifacts.get("csm_shadow.fsh"),
            ) && let Ok(p) = self.build_dynamic_depth_pipeline(
                ctx,
                csv,
                Some(csf),
                &vs_art.reflection,
                Some(&fs_art.reflection),
                bindless_layout,
                "csm_shadow",
                vk::CompareOp::LESS,
                vk::CullModeFlags::NONE,
                std::mem::size_of::<CsmShadowPush>() as u32,
                DepthVertexInput::Terrain40B,
                &spec_info,
            ) {
                self.graphics_pipelines.insert("csm_shadow".to_owned(), p);
            }
        }
        if let (Some(&csv28), Some(vs_art)) = (
            self.shader_modules.get("csm_shadow_28b.vsh"),
            self.compiled_artifacts.get("csm_shadow_28b.vsh"),
        ) {
            if let Ok(p) = self.build_dynamic_depth_pipeline(
                ctx,
                csv28,
                None,
                &vs_art.reflection,
                None,
                bindless_layout,
                "csm_shadow_28b_solid",
                vk::CompareOp::LESS,
                vk::CullModeFlags::BACK,
                std::mem::size_of::<CsmShadowPush>() as u32,
                DepthVertexInput::Terrain28B,
                &spec_info,
            ) {
                self.graphics_pipelines
                    .insert("csm_shadow_28b_solid".to_owned(), p);
            }
            if let (Some(&csf28), Some(fs_art)) = (
                self.shader_modules
                    .get("csm_shadow_28b.fsh")
                    .or_else(|| self.shader_modules.get("csm_shadow.fsh")),
                self.compiled_artifacts
                    .get("csm_shadow_28b.fsh")
                    .or_else(|| self.compiled_artifacts.get("csm_shadow.fsh")),
            ) && let Ok(p) = self.build_dynamic_depth_pipeline(
                ctx,
                csv28,
                Some(csf28),
                &vs_art.reflection,
                Some(&fs_art.reflection),
                bindless_layout,
                "csm_shadow_28b",
                vk::CompareOp::LESS,
                vk::CullModeFlags::NONE,
                std::mem::size_of::<CsmShadowPush>() as u32,
                DepthVertexInput::Terrain28B,
                &spec_info,
            ) {
                self.graphics_pipelines
                    .insert("csm_shadow_28b".to_owned(), p);
            }
        }
        if let (Some(&dbv), Some(&dbf), Some(vs_art), Some(fs_art)) = (
            self.shader_modules.get("depth_blit.vsh"),
            self.shader_modules.get("depth_blit.fsh"),
            self.compiled_artifacts.get("depth_blit.vsh"),
            self.compiled_artifacts.get("depth_blit.fsh"),
        ) && let Ok(p) = self.build_dynamic_depth_pipeline(
            ctx,
            dbv,
            Some(dbf),
            &vs_art.reflection,
            Some(&fs_art.reflection),
            bindless_layout,
            "depth_blit",
            vk::CompareOp::ALWAYS,
            vk::CullModeFlags::NONE,
            std::mem::size_of::<DepthReprojectPushConstants>() as u32,
            DepthVertexInput::None,
            &spec_info,
        ) {
            self.graphics_pipelines.insert("depth_blit".to_owned(), p);
        }

        // Build Hardware Ray Tracing pipeline + SBT if the full .rgen/.rmiss/.rchit/.rahit set is present.
        if ctx.capabilities.ray_tracing_pipeline
            && let (
                Some(&rgen_mod),
                Some(&rmiss_mod),
                Some(&rchit_mod),
                Some(&rahit_mod),
                Some(rgen_art),
                Some(rmiss_art),
                Some(rchit_art),
                Some(rahit_art),
            ) = (
                self.shader_modules.get("pathtrace.rgen"),
                self.shader_modules.get("pathtrace.rmiss"),
                self.shader_modules.get("pathtrace.rchit"),
                self.shader_modules.get("pathtrace.rahit"),
                self.compiled_artifacts.get("pathtrace.rgen"),
                self.compiled_artifacts.get("pathtrace.rmiss"),
                self.compiled_artifacts.get("pathtrace.rchit"),
                self.compiled_artifacts.get("pathtrace.rahit"),
            )
        {
            let merged_rt_refl = Self::merge_reflections(&[
                &rgen_art.reflection,
                &rmiss_art.reflection,
                &rchit_art.reflection,
                &rahit_art.reflection,
            ]);
            match self.build_ray_tracing_pipeline_bundle(
                ctx,
                rgen_mod,
                rmiss_mod,
                rchit_mod,
                rahit_mod,
                &merged_rt_refl,
                entry_name,
                &spec_info,
                bindless_layout,
            ) {
                Ok(rt_bundle) => self.rt_pipeline = Some(rt_bundle),
                Err(error) => {
                    self.last_compile_error =
                        Some(format!("Failed to build Path Tracing pipeline: {error:?}"))
                }
            }
        }

        let declares_reservoirs = |reflection: &vsp_compiler::SpirvReflection| {
            reflection
                .bindings
                .iter()
                .any(|binding| binding.set == 1 && binding.binding == 4)
        };
        self.rt_uses_reservoirs = self
            .rt_pipeline
            .as_ref()
            .is_some_and(|pipeline| declares_reservoirs(&pipeline.reflection))
            || self
                .compute_pipelines
                .get("rayquery_lighting.comp")
                .is_some_and(|pipeline| declares_reservoirs(&pipeline.reflection));
        self.rt_sharc_enabled = self.compute_pipelines.contains_key("sharc_resolve.comp");

        compiled_count
    }

    fn create_set_layout_for_bindings(
        ctx: &VulkanContext,
        bindings: &[ReflectedDescriptorBinding],
        target_set: u32,
        stage_flags: vk::ShaderStageFlags,
    ) -> Result<vk::DescriptorSetLayout, vk::Result> {
        let mut vk_bindings = Vec::new();
        for b in bindings.iter().filter(|b| b.set == target_set) {
            vk_bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b.binding)
                    .descriptor_type(Self::to_vk_descriptor_type(b.descriptor_type))
                    .descriptor_count(b.count.max(1))
                    .stage_flags(stage_flags),
            );
        }
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&vk_bindings);
        unsafe { ctx.device.create_descriptor_set_layout(&info, None) }
    }

    fn allocate_triple_sets(
        &self,
        ctx: &VulkanContext,
        layout: vk::DescriptorSetLayout,
    ) -> Result<[vk::DescriptorSet; 3], vk::Result> {
        if layout == vk::DescriptorSetLayout::null() {
            return Ok([vk::DescriptorSet::null(); 3]);
        }
        let layouts = [layout, layout, layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        let sets = unsafe { ctx.device.allocate_descriptor_sets(&alloc_info)? };
        Ok([sets[0], sets[1], sets[2]])
    }

    fn empty_set_layout(ctx: &VulkanContext) -> Result<vk::DescriptorSetLayout, vk::Result> {
        let info = vk::DescriptorSetLayoutCreateInfo::default();
        unsafe { ctx.device.create_descriptor_set_layout(&info, None) }
    }

    /// Builds a dynamic-rendering depth graphics pipeline for `csm_shadow[_solid]` (`Terrain40B`),
    /// `csm_shadow_28b[_solid]` (`Terrain28B`), or `depth_blit` (`None`, fullscreen triangle).
    #[allow(clippy::too_many_arguments)]
    fn build_dynamic_depth_pipeline(
        &self,
        ctx: &VulkanContext,
        vs: vk::ShaderModule,
        fs: Option<vk::ShaderModule>,
        vs_refl: &SpirvReflection,
        fs_refl: Option<&SpirvReflection>,
        bindless_layout: vk::DescriptorSetLayout,
        name: &str,
        depth_compare: vk::CompareOp,
        cull_mode: vk::CullModeFlags,
        push_size: u32,
        vertex_input_kind: DepthVertexInput,
        spec_info: &vk::SpecializationInfo<'_>,
    ) -> Result<GraphicsPipelineBundle, vk::Result> {
        let merged = if let Some(fr) = fs_refl {
            Self::merge_reflections(&[vs_refl, fr])
        } else {
            vs_refl.clone()
        };
        let stage_flags = if fs.is_some() {
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT
        } else {
            vk::ShaderStageFlags::VERTEX
        };
        let set0 = Self::create_set_layout_for_bindings(ctx, &merged.bindings, 0, stage_flags)?;
        let set1 = Self::create_set_layout_for_bindings(ctx, &merged.bindings, 1, stage_flags)?;
        let set2 = if bindless_layout != vk::DescriptorSetLayout::null() {
            bindless_layout
        } else {
            Self::empty_set_layout(ctx)?
        };
        let set3 = if self.pack_descriptor_set_layout != vk::DescriptorSetLayout::null() {
            self.pack_descriptor_set_layout
        } else {
            Self::empty_set_layout(ctx)?
        };
        let layouts = [set0, set1, set2, set3];
        let pc_range = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: push_size,
        }];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&pc_range);
        let pipeline_layout = unsafe { ctx.device.create_pipeline_layout(&layout_info, None)? };

        let entry = c"main";
        let mut vs_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vs)
            .name(entry);
        if !self.active_spec_constants.is_empty() {
            vs_stage = vs_stage.specialization_info(spec_info);
        }
        let mut stages = Vec::with_capacity(2);
        stages.push(vs_stage);
        if let Some(fs_mod) = fs {
            let mut fs_stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fs_mod)
                .name(entry);
            if !self.active_spec_constants.is_empty() {
                fs_stage = fs_stage.specialization_info(spec_info);
            }
            stages.push(fs_stage);
        }
        let terrain_binding_40 = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: 40,
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        let terrain_binding_28 = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: 28,
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        let terrain_attrs_40 = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R8G8B8A8_UNORM,
                offset: 12,
            },
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 16,
            },
            vk::VertexInputAttributeDescription {
                location: 3,
                binding: 0,
                format: vk::Format::R16G16_SINT,
                offset: 24,
            },
            vk::VertexInputAttributeDescription {
                location: 4,
                binding: 0,
                format: vk::Format::R8G8B8A8_SNORM,
                offset: 28,
            },
            vk::VertexInputAttributeDescription {
                location: 5,
                binding: 0,
                format: vk::Format::R8G8B8A8_SNORM,
                offset: 32,
            },
            vk::VertexInputAttributeDescription {
                location: 6,
                binding: 0,
                format: vk::Format::R16G16_SFLOAT,
                offset: 36,
            },
        ];
        let terrain_attrs_28 = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R8G8B8A8_UNORM,
                offset: 12,
            },
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 16,
            },
            vk::VertexInputAttributeDescription {
                location: 3,
                binding: 0,
                format: vk::Format::R16G16_SINT,
                offset: 24,
            },
        ];
        let vertex_input = match vertex_input_kind {
            DepthVertexInput::Terrain40B => vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&terrain_binding_40)
                .vertex_attribute_descriptions(&terrain_attrs_40),
            DepthVertexInput::Terrain28B => vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&terrain_binding_28)
                .vertex_attribute_descriptions(&terrain_attrs_28),
            DepthVertexInput::None => vk::PipelineVertexInputStateCreateInfo::default(),
        };
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let enable_shadow_bias = vertex_input_kind != DepthVertexInput::None;
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(cull_mode)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(enable_shadow_bias)
            .depth_bias_constant_factor(if enable_shadow_bias { 1.25 } else { 0.0 })
            .depth_bias_slope_factor(if enable_shadow_bias { 1.75 } else { 0.0 })
            .depth_bias_clamp(0.0)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(depth_compare)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default();
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .depth_attachment_format(vk::Format::D32_SFLOAT);
        let info = [vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic)
            .layout(pipeline_layout)
            .push_next(&mut rendering)];

        let pipeline = match unsafe {
            ctx.device
                .create_graphics_pipelines(self.pipeline_cache, &info, None)
        } {
            Ok(mut p) => p.remove(0),
            Err((_, err)) => {
                unsafe {
                    ctx.device.destroy_pipeline_layout(pipeline_layout, None);
                    ctx.device.destroy_descriptor_set_layout(set1, None);
                    ctx.device.destroy_descriptor_set_layout(set0, None);
                    if set2 != bindless_layout {
                        ctx.device.destroy_descriptor_set_layout(set2, None);
                    }
                    if set3 != self.pack_descriptor_set_layout {
                        ctx.device.destroy_descriptor_set_layout(set3, None);
                    }
                }
                return Err(err);
            }
        };
        unsafe {
            if set2 != bindless_layout {
                ctx.device.destroy_descriptor_set_layout(set2, None);
            }
            if set3 != self.pack_descriptor_set_layout {
                ctx.device.destroy_descriptor_set_layout(set3, None);
            }
        }
        let descriptor_sets_set0 = self.allocate_triple_sets(ctx, set0)?;
        let descriptor_sets_set1 = self.allocate_triple_sets(ctx, set1)?;
        Ok(GraphicsPipelineBundle {
            name: name.to_owned(),
            vertex_module: vs,
            fragment_module: fs.unwrap_or(vk::ShaderModule::null()),
            set0_layout: set0,
            set1_layout: set1,
            pipeline_layout,
            pipeline,
            color_attachment_count: 0,
            descriptor_sets_set0,
            descriptor_sets_set1,
        })
    }

    /// Builds a compute pipeline whose layout exposes all four descriptor sets:
    /// Set 0 = pass resources + WorldUBO(b0), Set 1 = TLAS/InstanceMeta/SHaRC (or empty),
    /// Set 2 = the global bindless texture table (always present), Set 3 = pack resources.
    fn build_compute_pipeline_bundle(
        &self,
        ctx: &VulkanContext,
        artifact: &CompiledShaderArtifact,
        module: vk::ShaderModule,
        entry_name: &CStr,
        spec_info: &vk::SpecializationInfo<'_>,
        bindless_layout: vk::DescriptorSetLayout,
    ) -> Result<ComputePipelineBundle, vk::Result> {
        let refl = &artifact.reflection;
        let set0_layout = Self::create_set_layout_for_bindings(
            ctx,
            &refl.bindings,
            0,
            vk::ShaderStageFlags::COMPUTE,
        )?;

        let set1_layout = if refl.bindings.iter().any(|b| b.set == 1) {
            Self::create_set_layout_for_bindings(
                ctx,
                &refl.bindings,
                1,
                vk::ShaderStageFlags::COMPUTE,
            )?
        } else {
            Self::empty_set_layout(ctx)?
        };

        // Set 2 is always the global bindless table and Set 3 the global pack resource table.
        // If either engine layout is unavailable, an empty layout keeps the indices contiguous.
        let set2_layout = if bindless_layout != vk::DescriptorSetLayout::null() {
            bindless_layout
        } else {
            Self::empty_set_layout(ctx)?
        };
        let set3_layout = if self.pack_descriptor_set_layout != vk::DescriptorSetLayout::null() {
            self.pack_descriptor_set_layout
        } else {
            Self::empty_set_layout(ctx)?
        };
        let set_layouts = [set0_layout, set1_layout, set2_layout, set3_layout];

        let pc_size = refl.push_constant_size.max(32);
        let pc_ranges = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: pc_size,
        }];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&pc_ranges);
        let pipeline_layout = unsafe { ctx.device.create_pipeline_layout(&layout_info, None)? };

        let mut stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(entry_name);
        if !self.active_spec_constants.is_empty() {
            stage_info = stage_info.specialization_info(spec_info);
        }

        let comp_info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout)];

        let pipeline = match unsafe {
            ctx.device
                .create_compute_pipelines(self.pipeline_cache, &comp_info, None)
        } {
            Ok(mut pipes) => pipes.remove(0),
            Err((_, err)) => {
                unsafe {
                    ctx.device.destroy_pipeline_layout(pipeline_layout, None);
                    ctx.device.destroy_descriptor_set_layout(set1_layout, None);
                    ctx.device.destroy_descriptor_set_layout(set0_layout, None);
                    if set2_layout != bindless_layout {
                        ctx.device.destroy_descriptor_set_layout(set2_layout, None);
                    }
                    if set3_layout != self.pack_descriptor_set_layout {
                        ctx.device.destroy_descriptor_set_layout(set3_layout, None);
                    }
                }
                return Err(err);
            }
        };
        unsafe {
            if set2_layout != bindless_layout {
                ctx.device.destroy_descriptor_set_layout(set2_layout, None);
            }
            if set3_layout != self.pack_descriptor_set_layout {
                ctx.device.destroy_descriptor_set_layout(set3_layout, None);
            }
        }

        let descriptor_sets_set0 = self.allocate_triple_sets(ctx, set0_layout)?;
        let descriptor_sets_set1 = self.allocate_triple_sets(ctx, set1_layout)?;

        let atrous_reverse_sets = if artifact.stage_name == "svgf_atrous.comp" {
            Some(self.allocate_triple_sets(ctx, set0_layout)?)
        } else {
            None
        };

        Ok(ComputePipelineBundle {
            stage_name: artifact.stage_name.clone(),
            shader_module: module,
            set0_layout,
            set1_layout,
            uses_external_bindless_set1: false,
            pipeline_layout,
            pipeline,
            descriptor_sets_set0,
            descriptor_sets_set1,
            atrous_reverse_sets,
            workgroup_size: refl.workgroup_size,
            push_constant_size: pc_size,
            reflection: refl.clone(),
        })
    }

    /// Builds the hardware RT pipeline with four descriptor sets and TWO hit groups:
    /// Group 0 = opaque (`pathtrace.rchit`, no any-hit) and Group 1 = cutout
    /// (`pathtrace.rchit` + `pathtrace.rahit`). The recursion depth is 2 so the closest-hit shader
    /// can trace a shadow/NEE ray.
    #[allow(clippy::too_many_arguments)]
    fn build_ray_tracing_pipeline_bundle(
        &self,
        ctx: &VulkanContext,
        rgen_mod: vk::ShaderModule,
        rmiss_mod: vk::ShaderModule,
        rchit_mod: vk::ShaderModule,
        rahit_mod: vk::ShaderModule,
        merged_refl: &SpirvReflection,
        entry_name: &CStr,
        spec_info: &vk::SpecializationInfo<'_>,
        bindless_layout: vk::DescriptorSetLayout,
    ) -> Result<RayTracingPipelineBundle, vk::Result> {
        let Some(ref rt_ext) = ctx.rt_pipeline_ext else {
            return Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT);
        };

        // Now that `pathtrace.rchit` uses a single rayPayloadEXT layout matching `pathtrace.rmiss`
        // and `VK_KHR_shader_atomic_int64` is enabled on the device, the full RT pipeline is enabled
        // by default. Users can explicitly opt out via VS_CORE_ENABLE_FULL_RT_PIPELINE=0 or
        // VS_DISABLE_RT_HIT_SHADERS=1, which returns ERROR_FEATURE_NOT_PRESENT so Tier 3 cleanly
        // fails during pack activation with a diagnostic.
        let full_rt = !matches!(
            std::env::var("VS_CORE_ENABLE_FULL_RT_PIPELINE").as_deref(),
            Ok("0") | Ok("false")
        ) && !matches!(
            std::env::var("VS_DISABLE_RT_HIT_SHADERS").as_deref(),
            Ok("1") | Ok("true")
        );
        if !full_rt {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }

        let rt_stages = vk::ShaderStageFlags::RAYGEN_KHR
            | vk::ShaderStageFlags::MISS_KHR
            | vk::ShaderStageFlags::CLOSEST_HIT_KHR
            | vk::ShaderStageFlags::ANY_HIT_KHR;

        let set0_layout =
            Self::create_set_layout_for_bindings(ctx, &merged_refl.bindings, 0, rt_stages)?;
        let set1_layout = if merged_refl.bindings.iter().any(|b| b.set == 1) {
            Self::create_set_layout_for_bindings(ctx, &merged_refl.bindings, 1, rt_stages)?
        } else {
            Self::empty_set_layout(ctx)?
        };
        let set2_layout = if bindless_layout != vk::DescriptorSetLayout::null() {
            bindless_layout
        } else {
            Self::empty_set_layout(ctx)?
        };
        let set3_layout = if self.pack_descriptor_set_layout != vk::DescriptorSetLayout::null() {
            self.pack_descriptor_set_layout
        } else {
            Self::empty_set_layout(ctx)?
        };
        let set_layouts = [set0_layout, set1_layout, set2_layout, set3_layout];
        let pc_range = [vk::PushConstantRange {
            stage_flags: rt_stages,
            offset: 0,
            size: merged_refl.push_constant_size,
        }];
        let mut layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        if merged_refl.push_constant_size > 0 {
            layout_info = layout_info.push_constant_ranges(&pc_range);
        }
        let pipeline_layout = unsafe { ctx.device.create_pipeline_layout(&layout_info, None)? };

        let mut s_rgen = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::RAYGEN_KHR)
            .module(rgen_mod)
            .name(entry_name);
        let mut s_rmiss = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::MISS_KHR)
            .module(rmiss_mod)
            .name(entry_name);
        let mut s_rchit = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .module(rchit_mod)
            .name(entry_name);
        let mut s_rahit = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::ANY_HIT_KHR)
            .module(rahit_mod)
            .name(entry_name);
        if !self.active_spec_constants.is_empty() {
            s_rgen = s_rgen.specialization_info(spec_info);
            s_rmiss = s_rmiss.specialization_info(spec_info);
            s_rchit = s_rchit.specialization_info(spec_info);
            s_rahit = s_rahit.specialization_info(spec_info);
        }

        let general_group = |shader: u32| {
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(shader)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR)
        };
        let hit_group = |any_hit: u32| {
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(2)
                .any_hit_shader(any_hit)
                .intersection_shader(vk::SHADER_UNUSED_KHR)
        };

        let stages = vec![s_rgen, s_rmiss, s_rchit, s_rahit];
        let groups = vec![
            general_group(0),
            general_group(1),
            hit_group(vk::SHADER_UNUSED_KHR), // Group 2: opaque
            hit_group(3),                     // Group 3: cutout (.rahit)
        ];
        // Iterative path tracing in pathtrace.rgen never recurses from hit shaders;
        // depth 1 minimizes per-lane RT continuation stack in local memory.
        let recursion = 1u32;
        let hit_groups = 2u32;

        let rt_info = [vk::RayTracingPipelineCreateInfoKHR::default()
            .stages(&stages)
            .groups(&groups)
            .max_pipeline_ray_recursion_depth(recursion)
            .layout(pipeline_layout)];

        let pipeline = match unsafe {
            rt_ext.create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                &rt_info,
                None,
            )
        } {
            Ok(mut pipes) => pipes.remove(0),
            Err((_, err)) => {
                unsafe {
                    ctx.device.destroy_pipeline_layout(pipeline_layout, None);
                    ctx.device.destroy_descriptor_set_layout(set1_layout, None);
                    ctx.device.destroy_descriptor_set_layout(set0_layout, None);
                    if set2_layout != bindless_layout {
                        ctx.device.destroy_descriptor_set_layout(set2_layout, None);
                    }
                    if set3_layout != self.pack_descriptor_set_layout {
                        ctx.device.destroy_descriptor_set_layout(set3_layout, None);
                    }
                }
                return Err(err);
            }
        };
        unsafe {
            if set2_layout != bindless_layout {
                ctx.device.destroy_descriptor_set_layout(set2_layout, None);
            }
            if set3_layout != self.pack_descriptor_set_layout {
                ctx.device.destroy_descriptor_set_layout(set3_layout, None);
            }
        }

        let sbt = ShaderBindingTable::build_for_pipeline(ctx, pipeline, 1, hit_groups)?;
        let descriptor_sets_set0 = self.allocate_triple_sets(ctx, set0_layout)?;
        let descriptor_sets_set1 = self.allocate_triple_sets(ctx, set1_layout)?;

        Ok(RayTracingPipelineBundle {
            set0_layout,
            set1_layout,
            pipeline_layout,
            pipeline,
            descriptor_sets_set0,
            descriptor_sets_set1,
            sbt,
            reflection: merged_refl.clone(),
        })
    }

    /// Dispatches `wind_displace.comp` on `cmd` with a `COMPUTE_SHADER -> ACCELERATION_STRUCTURE_BUILD_KHR`
    /// memory barrier so subsequent BLAS `UPDATE` refits read the freshly displaced foliage vertices.
    pub fn dispatch_wind_displace(
        &self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        push_constants: &WindPushConstants,
        max_vertex_count: u32,
    ) {
        if push_constants.job_count == 0 || max_vertex_count == 0 {
            return;
        }
        let Some(bundle) = self.compute_pipelines.get("wind_displace.comp") else {
            return;
        };
        if bundle.pipeline == vk::Pipeline::null() {
            return;
        }
        let groups_x = max_vertex_count.div_ceil(bundle.workgroup_size[0].max(1));
        let groups_y = push_constants.job_count;
        unsafe {
            ctx.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, bundle.pipeline);
            ctx.device.cmd_push_constants(
                cmd,
                bundle.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(push_constants),
            );
            ctx.device.cmd_dispatch(cmd, groups_x, groups_y, 1);
            ctx.cmd_memory_barrier2(
                cmd,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::AccessFlags2::from_raw(
                    vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR.as_raw()
                        | vk::AccessFlags2::SHADER_READ.as_raw(),
                ),
            );
        }
    }

    /// Binds `sharc_config_buffer` (`SharcConfigUniform`) and dispatches `sharc_resolve.comp`.
    pub fn dispatch_sharc_resolve(
        &self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        sharc_config_buffer: vk::Buffer,
        frame_slot: usize,
        capacity: u32,
    ) {
        if sharc_config_buffer == vk::Buffer::null() || capacity == 0 {
            return;
        }
        let Some(bundle) = self.compute_pipelines.get("sharc_resolve.comp") else {
            return;
        };
        let set0 = bundle.descriptor_sets_set0[frame_slot % 3];
        if bundle.pipeline == vk::Pipeline::null() || set0 == vk::DescriptorSet::null() {
            return;
        }
        let buf_info = [vk::DescriptorBufferInfo {
            buffer: sharc_config_buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
        }];
        let write = [vk::WriteDescriptorSet::default()
            .dst_set(set0)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&buf_info)];
        let groups_x = capacity.div_ceil(bundle.workgroup_size[0].max(64));
        unsafe {
            ctx.device.update_descriptor_sets(&write, &[]);
            ctx.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, bundle.pipeline);
            ctx.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                bundle.pipeline_layout,
                0,
                &[set0],
                &[],
            );
            ctx.device.cmd_dispatch(cmd, groups_x, 1, 1);
            ctx.cmd_memory_barrier2(
                cmd,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER
                    | if ctx.capabilities.ray_tracing_pipeline {
                        vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
                    } else {
                        vk::PipelineStageFlags2::empty()
                    },
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            );
        }
    }

    /// Dispatches a graph pass onto `cmd`. Every four-set pipeline binds:
    /// Set 0 = pass resources, Set 1 = ray-tracing resources, Set 2 = bindless table,
    /// Set 3 = pack resources. The bindless set is NEVER dropped when a TLAS is present.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_pass_with_graph(
        &mut self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        pass_kind: PassKind,
        graph: &RenderGraph,
        bindless: Option<&BindlessTextureTable>,
        pack_set: vk::DescriptorSet,
        tlas_handle: vk::AccelerationStructureKHR,
        frame_slot: usize,
        width: u32,
        height: u32,
        blas: Option<&ZeroCopyBlasManager>,
        _retire: Option<&RetireQueue>,
    ) {
        let slot = frame_slot % 3;

        // Tier 3 Full Path Tracing via hardware RT pipeline + SBT when available with both hit groups.
        if pass_kind == PassKind::FullPathTracing {
            let Some(rt_bundle) = self
                .rt_pipeline
                .as_ref()
                .filter(|_| tlas_handle != vk::AccelerationStructureKHR::null())
            else {
                return;
            };
            let ubo_buf = self.world_ubo_ring[slot]
                .as_ref()
                .map(|b| b.buffer)
                .unwrap_or(vk::Buffer::null());
            let sharc_fb_buf = self.sharc_fallback_ubo_ring[slot]
                .as_ref()
                .map(|b| b.buffer)
                .unwrap_or(vk::Buffer::null());
            let csm_ubo_buf = self.csm_ubo_slots[slot]
                .as_ref()
                .map(|b| b.buffer)
                .unwrap_or(vk::Buffer::null());
            Self::write_pass_descriptors(
                ctx,
                "pathtrace.rt",
                pass_kind,
                &rt_bundle.reflection,
                rt_bundle.descriptor_sets_set0[slot],
                rt_bundle.descriptor_sets_set1[slot],
                graph,
                tlas_handle,
                slot,
                blas,
                ubo_buf,
                sharc_fb_buf,
                csm_ubo_buf,
                self.default_sampler,
                &mut self.descriptor_sig_cache,
            );
            unsafe {
                ctx.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    rt_bundle.pipeline,
                );
                let sets = [
                    rt_bundle.descriptor_sets_set0[slot],
                    rt_bundle.descriptor_sets_set1[slot],
                    bindless
                        .map(|b| b.descriptor_set)
                        .unwrap_or(vk::DescriptorSet::null()),
                    pack_set,
                ];
                Self::bind_valid_descriptor_sets(
                    ctx,
                    cmd,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    rt_bundle.pipeline_layout,
                    &sets,
                );
                rt_bundle.sbt.cmd_trace_rays(ctx, cmd, width, height);
            }
            return;
        }

        let custom_shader_owned: Option<String> = match pass_kind {
            PassKind::Custom(custom_idx) => graph
                .custom_passes
                .get(&custom_idx)
                .map(|p| p.shader.clone()),
            _ => None,
        };
        let key: &str = match pass_kind {
            PassKind::ComputeVolumetricFog => "volumetric_fog.comp",
            PassKind::ComputeSsgi => "ssgi.comp",
            PassKind::ComputeFroxelGi => "compute_lighting.comp",
            PassKind::RayQueryLighting => "rayquery_lighting.comp",
            PassKind::UpscalerTsrOrVendor => "tsr_upscale.comp",
            PassKind::PostGradeAndToneMap => "post_tonemap.comp",
            PassKind::HdrCompose => {
                if graph
                    .resolve_image_view(RenderGraph::RES_SWAPCHAIN_OUT)
                    .is_some()
                {
                    "hdr_compose.comp"
                } else {
                    return;
                }
            }
            PassKind::FrameGenInterpolate => "optical_flow_fg.comp",
            PassKind::Custom(_) => match custom_shader_owned.as_deref() {
                Some(s) if !s.is_empty() => s,
                _ => return,
            },
            _ => return,
        };

        let Some(bundle) = self.compute_pipelines.get(key) else {
            return;
        };
        if bundle.pipeline == vk::Pipeline::null() {
            return;
        }

        let set0 = bundle.descriptor_sets_set0[slot];
        let set1 = bundle.descriptor_sets_set1[slot];
        let ubo_buf = self.world_ubo_ring[slot]
            .as_ref()
            .map(|b| b.buffer)
            .unwrap_or(vk::Buffer::null());
        let sharc_fb_buf = self.sharc_fallback_ubo_ring[slot]
            .as_ref()
            .map(|b| b.buffer)
            .unwrap_or(vk::Buffer::null());
        let csm_ubo_buf = self.csm_ubo_slots[slot]
            .as_ref()
            .map(|b| b.buffer)
            .unwrap_or(vk::Buffer::null());
        // VSL stages bind the same full world UBO as native stages (set 0 / binding 0); the
        // pack-parameter BDA they read is at offset 528 of this buffer.
        Self::write_pass_descriptors(
            ctx,
            key,
            pass_kind,
            &bundle.reflection,
            set0,
            set1,
            graph,
            tlas_handle,
            slot,
            blas,
            ubo_buf,
            sharc_fb_buf,
            csm_ubo_buf,
            self.default_sampler,
            &mut self.descriptor_sig_cache,
        );

        let wg_x = bundle.workgroup_size[0].max(1);
        let wg_y = bundle.workgroup_size[1].max(1);
        let groups_x = width.div_ceil(wg_x);
        let groups_y = height.div_ceil(wg_y);
        let groups_z = if pass_kind == PassKind::ComputeVolumetricFog {
            graph
                .internal_targets
                .get(&RenderGraph::RES_VOLUMETRIC_FOG)
                .map(|target| target.array_layers.max(1))
                .unwrap_or(1)
                .div_ceil(bundle.workgroup_size[2].max(1))
        } else {
            1
        };

        unsafe {
            ctx.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, bundle.pipeline);

            if set0 != vk::DescriptorSet::null() {
                let sets = [
                    set0,
                    set1,
                    bindless
                        .map(|b| b.descriptor_set)
                        .unwrap_or(vk::DescriptorSet::null()),
                    pack_set,
                ];
                Self::bind_valid_descriptor_sets(
                    ctx,
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    bundle.pipeline_layout,
                    &sets,
                );
            }

            match pass_kind {
                PassKind::UpscalerTsrOrVendor => {
                    ctx.device.cmd_push_constants(
                        cmd,
                        bundle.pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        bytemuck::bytes_of(&self.tsr_push_constants),
                    );
                    ctx.device.cmd_dispatch(cmd, groups_x, groups_y, 1);
                }
                PassKind::PostGradeAndToneMap | PassKind::HdrCompose => {
                    ctx.device.cmd_push_constants(
                        cmd,
                        bundle.pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        bytemuck::bytes_of(&self.hdr_push_constants),
                    );
                    ctx.device.cmd_dispatch(cmd, groups_x, groups_y, 1);
                }
                PassKind::FrameGenInterpolate => {
                    let pc = OpticalFlowPushConstants {
                        interpolation_factor: 0.5,
                        depth_rejection_scale: 2.0,
                        width,
                        height,
                    };
                    ctx.device.cmd_push_constants(
                        cmd,
                        bundle.pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        bytemuck::bytes_of(&pc),
                    );
                    ctx.device.cmd_dispatch(cmd, groups_x, groups_y, 1);
                }
                _ => {
                    ctx.device.cmd_dispatch(cmd, groups_x, groups_y, groups_z);
                }
            }
        }
    }

    /// Sets the `ui_only` flag (`pad0` in `HdrComposePushConstants`) for `hdr_compose.comp`.
    pub fn set_hdr_ui_only(&mut self, ui_only: bool) {
        self.hdr_push_constants.pad0 = u32::from(ui_only);
    }

    /// Dispatches `hdr_compose.comp`, encoding `ui_only` into the 4th push-constant uint (`pad0`).
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_hdr_compose(
        &mut self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        graph: &RenderGraph,
        bindless: Option<&BindlessTextureTable>,
        pack_set: vk::DescriptorSet,
        tlas_handle: vk::AccelerationStructureKHR,
        frame_slot: usize,
        width: u32,
        height: u32,
        ui_only: bool,
        blas: Option<&ZeroCopyBlasManager>,
    ) -> bool {
        if graph
            .resolve_image_view(RenderGraph::RES_SWAPCHAIN_OUT)
            .is_none()
        {
            return false;
        }
        let Some(bundle) = self.compute_pipelines.get("hdr_compose.comp") else {
            return false;
        };
        if bundle.pipeline == vk::Pipeline::null() {
            return false;
        }
        let slot = frame_slot % 3;
        let set0 = bundle.descriptor_sets_set0[slot];
        let set1 = bundle.descriptor_sets_set1[slot];
        let ubo_buf = self.world_ubo_ring[slot]
            .as_ref()
            .map(|b| b.buffer)
            .unwrap_or(vk::Buffer::null());
        let sharc_fb_buf = self.sharc_fallback_ubo_ring[slot]
            .as_ref()
            .map(|b| b.buffer)
            .unwrap_or(vk::Buffer::null());
        let csm_ubo_buf = self.csm_ubo_slots[slot]
            .as_ref()
            .map(|b| b.buffer)
            .unwrap_or(vk::Buffer::null());
        Self::write_pass_descriptors(
            ctx,
            "hdr_compose.comp",
            PassKind::HdrCompose,
            &bundle.reflection,
            set0,
            set1,
            graph,
            tlas_handle,
            slot,
            blas,
            ubo_buf,
            sharc_fb_buf,
            csm_ubo_buf,
            self.default_sampler,
            &mut self.descriptor_sig_cache,
        );
        let mut pc = self.hdr_push_constants;
        pc.pad0 = u32::from(ui_only);
        let wg_x = bundle.workgroup_size[0].max(1);
        let wg_y = bundle.workgroup_size[1].max(1);
        let groups_x = width.div_ceil(wg_x);
        let groups_y = height.div_ceil(wg_y);
        unsafe {
            ctx.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, bundle.pipeline);
            if set0 != vk::DescriptorSet::null() {
                let sets = [
                    set0,
                    set1,
                    bindless
                        .map(|b| b.descriptor_set)
                        .unwrap_or(vk::DescriptorSet::null()),
                    pack_set,
                ];
                Self::bind_valid_descriptor_sets(
                    ctx,
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    bundle.pipeline_layout,
                    &sets,
                );
            }
            ctx.device.cmd_push_constants(
                cmd,
                bundle.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&pc),
            );
            ctx.device.cmd_dispatch(cmd, groups_x, groups_y, 1);
        }
        true
    }

    /// Backwards-compatible wrapper for dispatching a pass without explicit graph/bindless refs.
    pub fn dispatch_pass(
        &mut self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        pass_kind: PassKind,
        width: u32,
        height: u32,
    ) {
        let dummy_graph = RenderGraph::default();
        self.dispatch_pass_with_graph(
            ctx,
            cmd,
            pass_kind,
            &dummy_graph,
            None,
            vk::DescriptorSet::null(),
            vk::AccelerationStructureKHR::null(),
            0,
            width,
            height,
            None,
            None,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn write_pass_descriptors(
        ctx: &VulkanContext,
        pass_key: &str,
        pass_kind: PassKind,
        refl: &SpirvReflection,
        set0: vk::DescriptorSet,
        set1: vk::DescriptorSet,
        graph: &RenderGraph,
        tlas_handle: vk::AccelerationStructureKHR,
        frame_slot: usize,
        blas: Option<&ZeroCopyBlasManager>,
        ubo_buf: vk::Buffer,
        sharc_fallback_buf: vk::Buffer,
        csm_ubo_buf: vk::Buffer,
        default_sampler: vk::Sampler,
        sig_cache: &mut HashMap<(String, usize), u64>,
    ) {
        let resolve_buf = |b: &ReflectedDescriptorBinding| -> vk::Buffer {
            if b.set == 1 {
                match b.binding {
                    1 => blas
                        .map(|manager| manager.instance_meta_buffer_for_slot(frame_slot).0)
                        .unwrap_or(vk::Buffer::null()),
                    2 => blas
                        .map(|manager| manager.sharc.config_buffer(frame_slot).0)
                        .filter(|&buffer| buffer != vk::Buffer::null())
                        .unwrap_or(sharc_fallback_buf),
                    3 => blas
                        .map(|manager| manager.emissive_lights.light_buffer_for_slot(frame_slot).0)
                        .unwrap_or(vk::Buffer::null()),
                    4 => blas
                        .map(|manager| {
                            manager
                                .emissive_lights
                                .reservoir_buffer_for_slot(frame_slot)
                                .0
                        })
                        .unwrap_or(vk::Buffer::null()),
                    _ => ubo_buf,
                }
            } else if b.descriptor_type == ReflectedDescriptorType::StorageBuffer {
                let named_res = Self::resolve_resource_for_binding_named(
                    pass_kind, b.binding, &b.name, b.writable, graph,
                );
                if let Some(bound_buf) = graph.bound_buffers.get(&named_res) {
                    bound_buf.buffer
                } else if named_res == RenderGraph::RES_EMISSIVE_LIGHTS {
                    blas.map(|m| m.emissive_lights.light_buffer_for_slot(frame_slot).0)
                        .unwrap_or(ubo_buf)
                } else if named_res == RenderGraph::RES_RESTIR_RESERVOIRS {
                    blas.map(|m| m.emissive_lights.reservoir_buffer_for_slot(frame_slot).0)
                        .unwrap_or(ubo_buf)
                } else {
                    ubo_buf
                }
            } else if (pass_kind == PassKind::ComputeFroxelGi && b.binding == 7)
                || (pass_kind == PassKind::ComputeVolumetricFog && b.binding == 2)
                || b.name.eq_ignore_ascii_case("CsmShadowUBO")
                || b.name.eq_ignore_ascii_case("uCsm")
            {
                if csm_ubo_buf != vk::Buffer::null() {
                    csm_ubo_buf
                } else {
                    ubo_buf
                }
            } else if (pass_kind == PassKind::ComputeFroxelGi && b.binding == 14)
                || (pass_kind == PassKind::RayQueryLighting && b.binding == 13)
                || (pass_kind == PassKind::ComputeVolumetricFog && b.binding == 6)
                || b.name.eq_ignore_ascii_case("DhFrameUniforms")
                || b.name.eq_ignore_ascii_case("uDh")
            {
                graph.dh_ubo_buffer_for_slot(frame_slot).unwrap_or(ubo_buf)
            } else {
                ubo_buf
            }
        };

        // Compute 64-bit FNV-1a signature over all bound handles before allocating any Vecs.
        let mut sig = fnv1a_init();
        sig = fnv1a_mix_u64(sig, set0.as_raw());
        sig = fnv1a_mix_u64(sig, set1.as_raw());
        for b in &refl.bindings {
            let dst_set = match b.set {
                0 => set0,
                1 => set1,
                _ => continue,
            };
            if dst_set == vk::DescriptorSet::null() {
                continue;
            }
            sig = fnv1a_mix_u64(sig, (u64::from(b.set) << 32) | u64::from(b.binding));
            match b.descriptor_type {
                ReflectedDescriptorType::UniformBuffer | ReflectedDescriptorType::StorageBuffer => {
                    let buf = resolve_buf(b);
                    if buf == vk::Buffer::null() {
                        continue;
                    }
                    sig = fnv1a_mix_u64(sig, buf.as_raw());
                    sig = fnv1a_mix_u64(sig, 0);
                    sig = fnv1a_mix_u64(sig, vk::WHOLE_SIZE);
                }
                ReflectedDescriptorType::AccelerationStructure => {
                    if tlas_handle == vk::AccelerationStructureKHR::null() {
                        continue;
                    }
                    sig = fnv1a_mix_u64(sig, tlas_handle.as_raw());
                }
                ReflectedDescriptorType::CombinedImageSampler
                | ReflectedDescriptorType::SampledImage
                | ReflectedDescriptorType::StorageImage => {
                    let res_id = Self::resolve_resource_for_binding_named(
                        pass_kind, b.binding, &b.name, b.writable, graph,
                    );
                    let Some(view) = graph.resolve_image_view(res_id) else {
                        continue;
                    };
                    if view == vk::ImageView::null() {
                        continue;
                    }
                    let sampler = graph
                        .resolve_sampler(res_id)
                        .filter(|&s| s != vk::Sampler::null())
                        .unwrap_or(default_sampler);
                    let layout = if b.descriptor_type == ReflectedDescriptorType::StorageImage {
                        vk::ImageLayout::GENERAL
                    } else {
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    };
                    sig = fnv1a_mix_u64(sig, view.as_raw());
                    sig = fnv1a_mix_u64(sig, sampler.as_raw());
                    sig = fnv1a_mix_u64(sig, layout.as_raw() as u64);
                }
                ReflectedDescriptorType::Sampler => {
                    sig = fnv1a_mix_u64(sig, default_sampler.as_raw());
                }
            }
        }

        let cache_key = (pass_key.to_owned(), frame_slot);
        if sig_cache.get(&cache_key).copied() == Some(sig) {
            return;
        }

        let tlas_slice = [tlas_handle];
        let cap = refl.bindings.len();
        let mut buf_infos: Vec<[vk::DescriptorBufferInfo; 1]> = Vec::with_capacity(cap);
        let mut img_infos: Vec<[vk::DescriptorImageInfo; 1]> = Vec::with_capacity(cap);
        let mut as_infos: Vec<vk::WriteDescriptorSetAccelerationStructureKHR> =
            Vec::with_capacity(2);
        let mut pending_writes: Vec<PendingDescriptorWrite<'_>> = Vec::with_capacity(cap);

        for b in &refl.bindings {
            // Set 2 (bindless) and Set 3 (pack resources) are global sets bound by the engine and
            // are never written here.
            let dst_set = match b.set {
                0 => set0,
                1 => set1,
                _ => continue,
            };
            if dst_set == vk::DescriptorSet::null() {
                continue;
            }

            match b.descriptor_type {
                ReflectedDescriptorType::UniformBuffer | ReflectedDescriptorType::StorageBuffer => {
                    let buf = resolve_buf(b);
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
                        .dst_set(dst_set)
                        .dst_binding(b.binding)
                        .descriptor_type(Self::to_vk_descriptor_type(b.descriptor_type));
                    pending_writes.push((write, Some(idx), None, None));
                }
                ReflectedDescriptorType::AccelerationStructure => {
                    if tlas_handle == vk::AccelerationStructureKHR::null() {
                        continue;
                    }
                    let idx = as_infos.len();
                    as_infos.push(
                        vk::WriteDescriptorSetAccelerationStructureKHR::default()
                            .acceleration_structures(&tlas_slice),
                    );
                    let mut write = vk::WriteDescriptorSet::default()
                        .dst_set(dst_set)
                        .dst_binding(b.binding)
                        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR);
                    write.descriptor_count = 1;
                    pending_writes.push((write, None, None, Some(idx)));
                }
                ReflectedDescriptorType::CombinedImageSampler
                | ReflectedDescriptorType::SampledImage
                | ReflectedDescriptorType::StorageImage => {
                    let res_id = Self::resolve_resource_for_binding_named(
                        pass_kind, b.binding, &b.name, b.writable, graph,
                    );
                    let Some(view) = graph.resolve_image_view(res_id) else {
                        continue;
                    };
                    if view == vk::ImageView::null() {
                        continue;
                    }
                    let sampler = graph
                        .resolve_sampler(res_id)
                        .filter(|&s| s != vk::Sampler::null())
                        .unwrap_or(default_sampler);
                    let layout = if b.descriptor_type == ReflectedDescriptorType::StorageImage {
                        vk::ImageLayout::GENERAL
                    } else {
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    };
                    let idx = img_infos.len();
                    img_infos.push([vk::DescriptorImageInfo {
                        sampler,
                        image_view: view,
                        image_layout: layout,
                    }]);
                    let write = vk::WriteDescriptorSet::default()
                        .dst_set(dst_set)
                        .dst_binding(b.binding)
                        .descriptor_type(Self::to_vk_descriptor_type(b.descriptor_type));
                    pending_writes.push((write, None, Some(idx), None));
                }
                ReflectedDescriptorType::Sampler => {
                    let idx = img_infos.len();
                    img_infos.push([vk::DescriptorImageInfo {
                        sampler: default_sampler,
                        image_view: vk::ImageView::null(),
                        image_layout: vk::ImageLayout::UNDEFINED,
                    }]);
                    let write = vk::WriteDescriptorSet::default()
                        .dst_set(dst_set)
                        .dst_binding(b.binding)
                        .descriptor_type(vk::DescriptorType::SAMPLER);
                    pending_writes.push((write, None, Some(idx), None));
                }
            }
        }

        if !pending_writes.is_empty() {
            let mut final_writes: Vec<vk::WriteDescriptorSet> =
                Vec::with_capacity(pending_writes.len());
            for (mut w, buf_idx, img_idx, as_idx) in pending_writes {
                if let Some(bi) = buf_idx {
                    w = w.buffer_info(&buf_infos[bi]);
                } else if let Some(ii) = img_idx {
                    w = w.image_info(&img_infos[ii]);
                } else if let Some(ai) = as_idx {
                    w.p_next = (&as_infos[ai]
                        as *const vk::WriteDescriptorSetAccelerationStructureKHR)
                        .cast();
                }
                final_writes.push(w);
            }
            unsafe {
                ctx.device.update_descriptor_sets(&final_writes, &[]);
            }
        }
        sig_cache.insert(cache_key, sig);
    }

    /// Maps a Set 0 image binding to its graph resource, matching `hyaloclast-sunlit-vanilla.vsp` 1:1.
    pub fn resolve_resource_for_binding(
        pass_kind: PassKind,
        binding: u32,
        writable: bool,
        cfg: &crate::graph::GraphConfig,
    ) -> ResourceId {
        use RenderGraph as R;
        let active_depth = if cfg.tier == RenderTier::Tier3PathTracing {
            R::RES_LINEAR_DEPTH
        } else {
            R::RES_MC_MAIN_DEPTH
        };
        let radiance_after_denoise = if cfg.effective_denoiser_active() {
            R::RES_DENOISED_RADIANCE
        } else {
            R::RES_RAW_RADIANCE
        };
        let radiance_after_upscale = if cfg.upscaler_mode != 0 {
            R::RES_UPSCALED_HDR
        } else {
            radiance_after_denoise
        };

        match pass_kind {
            // volumetric_fog.comp: b0 WorldUBO, b1 FOG_HISTORY, b2 CsmShadowUBO,
            // b3 SHADOW_CSM, b4 VOLUMETRIC_FOG (write).
            PassKind::ComputeVolumetricFog => match binding {
                1 => R::RES_VOLUMETRIC_FOG_HISTORY,
                3 => R::RES_SHADOW_CSM,
                _ => R::RES_VOLUMETRIC_FOG,
            },
            // ssgi.comp: b0 WorldUBO, b1 ALBEDO, b2 NORMAL, b3 DEPTH, b4 VELOCITY,
            // b5 PREV_HUDLESS_COLOR, b6 SSGI_HISTORY, b7 SSGI_OUT (write).
            PassKind::ComputeSsgi => match binding {
                1 => R::RES_GBUFFER_ALBEDO,
                2 => R::RES_GBUFFER_NORMAL,
                3 => R::RES_MC_MAIN_DEPTH,
                4 => R::RES_GBUFFER_VELOCITY,
                5 => R::RES_PREV_HUDLESS_COLOR,
                6 => R::RES_SSGI_HISTORY,
                _ => R::RES_SSGI_OUT,
            },
            // compute_lighting.comp: b0 WorldUBO, b1 ALBEDO, b2 NORMAL, b3 MATERIAL,
            // b4 DEPTH, b5 RAW_RADIANCE, b6 SHADOW_CSM, b7 CsmShadowUBO, b8 SSGI_OUT,
            // b9 VOLUMETRIC_FOG, b10 DH_ALBEDO, b11 DH_NORMAL, b12 DH_MATERIAL, b13 DH_DEPTH, b14 uDh.
            PassKind::ComputeFroxelGi => match binding {
                1 => R::RES_GBUFFER_ALBEDO,
                2 => R::RES_GBUFFER_NORMAL,
                3 => R::RES_GBUFFER_MATERIAL,
                4 => R::RES_MC_MAIN_DEPTH,
                5 => R::RES_RAW_RADIANCE,
                6 => R::RES_SHADOW_CSM,
                9 => R::RES_VOLUMETRIC_FOG,
                10 => R::RES_DH_ALBEDO,
                11 => R::RES_DH_NORMAL,
                12 => R::RES_DH_MATERIAL,
                13 => R::RES_DH_DEPTH,
                _ => R::RES_SSGI_OUT,
            },
            // rayquery_lighting.comp Set0: b0 WorldUBO, b1 ALBEDO, b2 NORMAL, b3 MATERIAL,
            // b4 DEPTH, b5 RAW_RADIANCE, b6 VELOCITY, b7 VOLUMETRIC_FOG, b8 SHARP_RADIANCE,
            // b9 DH_ALBEDO, b10 DH_NORMAL, b11 DH_MATERIAL, b12 DH_DEPTH, b13 uDh.
            PassKind::RayQueryLighting => match binding {
                1 => R::RES_GBUFFER_ALBEDO,
                2 => R::RES_GBUFFER_NORMAL,
                3 => R::RES_GBUFFER_MATERIAL,
                4 => R::RES_MC_MAIN_DEPTH,
                5 => R::RES_RAW_RADIANCE,
                6 => R::RES_GBUFFER_VELOCITY,
                7 => R::RES_VOLUMETRIC_FOG,
                8 => R::RES_SHARP_RADIANCE,
                9 => R::RES_DH_ALBEDO,
                10 => R::RES_DH_NORMAL,
                11 => R::RES_DH_MATERIAL,
                12 => R::RES_DH_DEPTH,
                _ => R::RES_RAW_RADIANCE,
            },
            // pathtrace.rgen/.rchit/.rmiss Set0: b0 WorldUBO, b1 RAW_RADIANCE, b2 NORMAL,
            // b3 VELOCITY, b4 LINEAR_DEPTH, b5 PT_ALBEDO, b6 SHARP_RADIANCE, b7 GBUFFER_MATERIAL,
            // b8 VOLUMETRIC_FOG.
            PassKind::FullPathTracing => match binding {
                1 => R::RES_RAW_RADIANCE,
                2 => R::RES_GBUFFER_NORMAL,
                3 => R::RES_GBUFFER_VELOCITY,
                4 => R::RES_LINEAR_DEPTH,
                5 => R::RES_PT_ALBEDO,
                6 => R::RES_SHARP_RADIANCE,
                7 => R::RES_GBUFFER_MATERIAL,
                8 => R::RES_VOLUMETRIC_FOG,
                _ => R::RES_RAW_RADIANCE,
            },
            // svgf_temporal.comp: b0 WorldUBO, b1 RAW_RADIANCE, b2 NORMAL, b3 VELOCITY,
            // b4 active_depth, b5/6 history, b7 DENOISED (write), b8 MOMENTS (write).
            PassKind::DenoiserSvgfOrNrd => match binding {
                1 => R::RES_RAW_RADIANCE,
                2 => R::RES_GBUFFER_NORMAL,
                3 => R::RES_GBUFFER_VELOCITY,
                4 => active_depth,
                5 => R::RES_SVGF_HISTORY_RADIANCE,
                6 => R::RES_SVGF_HISTORY_MOMENTS,
                8 => R::RES_SVGF_MOMENTS,
                _ => R::RES_DENOISED_RADIANCE,
            },
            // tsr_upscale.comp: b0 radiance, b1 TSR_HISTORY, b2 VELOCITY, b3 UPSCALED (write),
            // b4 active_depth, b5 NORMAL.
            PassKind::UpscalerTsrOrVendor => match binding {
                0 => radiance_after_denoise,
                1 => R::RES_TSR_HISTORY,
                2 => R::RES_GBUFFER_VELOCITY,
                4 => active_depth,
                5 => R::RES_GBUFFER_NORMAL,
                _ => R::RES_UPSCALED_HDR,
            },
            PassKind::PostGradeAndToneMap => match binding {
                0 | 1 => radiance_after_upscale,
                10 => R::RES_SVGF_MOMENTS,
                _ => {
                    if cfg.post_writes_directly_to_hudless() {
                        R::RES_HUDLESS_COLOR
                    } else {
                        R::RES_POST_OUTPUT
                    }
                }
            },
            // hdr_compose.comp: b0 linear HDR radiance (radiance_after_upscale), b1 UI_COLOR,
            // b2 SWAPCHAIN_OUT (write).
            PassKind::HdrCompose => match binding {
                0 => radiance_after_upscale,
                1 => R::RES_UI_COLOR,
                _ => R::RES_SWAPCHAIN_OUT,
            },
            PassKind::FrameGenInterpolate => match binding {
                0 => R::RES_PREV_HUDLESS_COLOR,
                1 => R::RES_HUDLESS_COLOR,
                2 => R::RES_GBUFFER_VELOCITY,
                3 => active_depth,
                4 => R::RES_UI_COLOR,
                _ => R::RES_INTERPOLATED_FRAME,
            },
            _ => {
                let _ = writable;
                R::RES_HUDLESS_COLOR
            }
        }
    }

    /// Resolves a Set 0 descriptor binding by consulting:
    /// 1. Explicit `[[passes]]` `bindings` table in `pack.toml`,
    /// 2. Reflected SPIR-V `OpName` variable identifier (`name`) via `RenderGraph::resolve_resource_by_name`,
    /// 3. Canonical `(PassKind, binding)` fallback (`resolve_resource_for_binding`).
    pub fn resolve_resource_for_binding_named(
        pass_kind: PassKind,
        binding: u32,
        name: &str,
        writable: bool,
        graph: &RenderGraph,
    ) -> ResourceId {
        if let PassKind::Custom(custom_idx) = pass_kind
            && let Some(spec) = graph.custom_passes.get(&custom_idx)
            && let Some(&explicit_id) = spec.bindings.get(&binding)
        {
            return explicit_id;
        }
        for spec in graph.custom_passes.values() {
            if spec.kind == pass_kind
                && let Some(&explicit_id) = spec.bindings.get(&binding)
            {
                return explicit_id;
            }
        }

        if !name.is_empty() {
            let cfg = &graph.config;
            let active_depth = if cfg.tier == RenderTier::Tier3PathTracing {
                RenderGraph::RES_LINEAR_DEPTH
            } else {
                RenderGraph::RES_MC_MAIN_DEPTH
            };
            let radiance_after_denoise = if cfg.effective_denoiser_active() {
                RenderGraph::RES_DENOISED_RADIANCE
            } else {
                RenderGraph::RES_RAW_RADIANCE
            };
            let radiance_after_upscale = if cfg.upscaler_mode != 0 {
                RenderGraph::RES_UPSCALED_HDR
            } else {
                radiance_after_denoise
            };
            let lower = name.trim().to_ascii_lowercase();
            match lower.as_str() {
                "udepth" | "uactive_depth" | "uactivedepth" => return active_depth,
                "uactivealbedo" => {
                    return if cfg.tier == RenderTier::Tier3PathTracing {
                        RenderGraph::RES_PT_ALBEDO
                    } else {
                        RenderGraph::RES_GBUFFER_ALBEDO
                    };
                }
                "uinradiance" | "uradiance" => {
                    return match pass_kind {
                        PassKind::UpscalerTsrOrVendor => radiance_after_denoise,
                        PassKind::PostGradeAndToneMap | PassKind::HdrCompose => {
                            radiance_after_upscale
                        }
                        _ => RenderGraph::RES_RAW_RADIANCE,
                    };
                }
                _ => {}
            }
            if let Some(resolved) = graph.resolve_resource_by_name(name) {
                if resolved == RenderGraph::RES_HUDLESS_COLOR
                    && pass_kind == PassKind::PostGradeAndToneMap
                    && writable
                    && !cfg.post_writes_directly_to_hudless()
                {
                    return RenderGraph::RES_POST_OUTPUT;
                }
                return resolved;
            }
        }

        if pass_kind == PassKind::FullPathTracing {
            if binding == 9
                && let Some(id) = graph.resolve_resource_by_name("restir_gi_reservoir_history")
            {
                return id;
            }
            if binding == 10
                && let Some(id) = graph.resolve_resource_by_name("restir_gi_reservoir")
            {
                return id;
            }
        }
        if pass_kind == PassKind::PostGradeAndToneMap {
            let active_depth = if graph.config.tier == RenderTier::Tier3PathTracing {
                RenderGraph::RES_LINEAR_DEPTH
            } else {
                RenderGraph::RES_MC_MAIN_DEPTH
            };
            let active_albedo = if graph.config.tier == RenderTier::Tier3PathTracing {
                RenderGraph::RES_PT_ALBEDO
            } else {
                RenderGraph::RES_GBUFFER_ALBEDO
            };
            match binding {
                3 => return active_albedo,
                4 => return RenderGraph::RES_GBUFFER_NORMAL,
                5 => return RenderGraph::RES_GBUFFER_MATERIAL,
                6 => return RenderGraph::RES_GBUFFER_VELOCITY,
                7 => return active_depth,
                8 => return RenderGraph::RES_RAW_RADIANCE,
                9 => return RenderGraph::RES_SHARP_RADIANCE,
                10 => return RenderGraph::RES_SVGF_MOMENTS,
                _ => {}
            }
        }

        Self::resolve_resource_for_binding(pass_kind, binding, writable, &graph.config)
    }

    /// Writes a single combined-image-sampler (`storage = false`) or storage-image (`storage = true`)
    /// descriptor into `set` at `binding`.
    unsafe fn write_image_binding(
        &self,
        ctx: &VulkanContext,
        set: vk::DescriptorSet,
        binding: u32,
        view: vk::ImageView,
        sampler: vk::Sampler,
        storage: bool,
    ) {
        if set == vk::DescriptorSet::null() || view == vk::ImageView::null() {
            return;
        }
        let info = [vk::DescriptorImageInfo {
            sampler: if storage {
                vk::Sampler::null()
            } else {
                sampler
            },
            image_view: view,
            image_layout: if storage {
                vk::ImageLayout::GENERAL
            } else {
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            },
        }];
        let write = [vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(binding)
            .descriptor_type(if storage {
                vk::DescriptorType::STORAGE_IMAGE
            } else {
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER
            })
            .image_info(&info)];
        unsafe {
            ctx.device.update_descriptor_sets(&write, &[]);
        }
    }

    fn image_view_or_null(graph: &RenderGraph, id: ResourceId) -> vk::ImageView {
        graph
            .resolve_image_view(id)
            .unwrap_or(vk::ImageView::null())
    }

    /// Four-pass SVGF denoise: `svgf_temporal.comp`, then `svgf_atrous.comp` at step sizes 1, 2 and 4,
    /// ping-ponging between `RES_DENOISED_RADIANCE` and `RES_SVGF_PING` with explicit layout
    /// transitions. The temporal result is copied into `RES_SVGF_HISTORY_RADIANCE`, the final à-trous
    /// iteration is remodulated with G-buffer albedo into `RES_DENOISED_RADIANCE`, and
    /// `RES_SVGF_MOMENTS` is copied into `RES_SVGF_HISTORY_MOMENTS`.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_svgf_denoise(
        &mut self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        graph: &RenderGraph,
        bindless: Option<&BindlessTextureTable>,
        pack_set: vk::DescriptorSet,
        tlas_handle: vk::AccelerationStructureKHR,
        frame_slot: usize,
        width: u32,
        height: u32,
        blas: Option<&ZeroCopyBlasManager>,
    ) {
        let slot = frame_slot % 3;
        let (active_depth, active_albedo) = if graph.config.tier == RenderTier::Tier3PathTracing {
            (RenderGraph::RES_LINEAR_DEPTH, RenderGraph::RES_PT_ALBEDO)
        } else {
            (
                RenderGraph::RES_MC_MAIN_DEPTH,
                RenderGraph::RES_GBUFFER_ALBEDO,
            )
        };
        let albedo_view = Self::image_view_or_null(graph, active_albedo);

        let color_subrange = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        // Pass A: temporal accumulation.
        let ubo_buf = self.world_ubo_buffer_for_slot(slot);
        let default_sampler = self.default_sampler;
        if let Some(bundle) = self.compute_pipelines.get("svgf_temporal.comp") {
            let s0 = bundle.descriptor_sets_set0[slot];
            let s1 = bundle.descriptor_sets_set1[slot];
            let has_ubo = bundle
                .reflection
                .bindings
                .iter()
                .any(|b| b.set == 0 && b.binding == 0)
                && ubo_buf != vk::Buffer::null();
            let views = [
                (
                    1,
                    Self::image_view_or_null(graph, RenderGraph::RES_RAW_RADIANCE),
                ),
                (
                    2,
                    Self::image_view_or_null(graph, RenderGraph::RES_GBUFFER_NORMAL),
                ),
                (
                    3,
                    Self::image_view_or_null(graph, RenderGraph::RES_GBUFFER_VELOCITY),
                ),
                (4, Self::image_view_or_null(graph, active_depth)),
                (
                    5,
                    Self::image_view_or_null(graph, RenderGraph::RES_SVGF_HISTORY_RADIANCE),
                ),
                (
                    6,
                    Self::image_view_or_null(graph, RenderGraph::RES_SVGF_HISTORY_MOMENTS),
                ),
                (
                    7,
                    Self::image_view_or_null(graph, RenderGraph::RES_DENOISED_RADIANCE),
                ),
                (
                    8,
                    Self::image_view_or_null(graph, RenderGraph::RES_SVGF_MOMENTS),
                ),
                (9, albedo_view),
                (
                    10,
                    Self::image_view_or_null(graph, RenderGraph::RES_SHARP_RADIANCE),
                ),
            ];
            if s0 != vk::DescriptorSet::null() {
                let mut sig = fnv1a_init();
                sig = fnv1a_mix_u64(sig, s0.as_raw());
                if has_ubo {
                    sig = fnv1a_mix_u64(sig, ubo_buf.as_raw());
                    sig = fnv1a_mix_u64(sig, 0);
                    sig = fnv1a_mix_u64(sig, vk::WHOLE_SIZE);
                }
                for (b, v) in views {
                    if v == vk::ImageView::null()
                        || !bundle
                            .reflection
                            .bindings
                            .iter()
                            .any(|binding| binding.set == 0 && binding.binding == b)
                    {
                        continue;
                    }
                    let storage = b == 7 || b == 8;
                    let sampler = if storage {
                        vk::Sampler::null()
                    } else {
                        default_sampler
                    };
                    sig = fnv1a_mix_u64(sig, u64::from(b));
                    sig = fnv1a_mix_u64(sig, v.as_raw());
                    sig = fnv1a_mix_u64(sig, sampler.as_raw());
                }
                let cache_key = ("svgf_temporal.comp".to_owned(), slot);
                if self.descriptor_sig_cache.get(&cache_key).copied() != Some(sig) {
                    let ubo_info = [vk::DescriptorBufferInfo {
                        buffer: ubo_buf,
                        offset: 0,
                        range: vk::WHOLE_SIZE,
                    }];
                    let mut img_infos: Vec<[vk::DescriptorImageInfo; 1]> =
                        Vec::with_capacity(views.len());
                    let mut valid_bindings: Vec<(u32, bool, usize)> =
                        Vec::with_capacity(views.len());
                    for (b, v) in views {
                        if v == vk::ImageView::null()
                            || !bundle
                                .reflection
                                .bindings
                                .iter()
                                .any(|binding| binding.set == 0 && binding.binding == b)
                        {
                            continue;
                        }
                        let storage = b == 7 || b == 8;
                        let idx = img_infos.len();
                        img_infos.push([vk::DescriptorImageInfo {
                            sampler: if storage {
                                vk::Sampler::null()
                            } else {
                                default_sampler
                            },
                            image_view: v,
                            image_layout: if storage {
                                vk::ImageLayout::GENERAL
                            } else {
                                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                            },
                        }]);
                        valid_bindings.push((b, storage, idx));
                    }
                    let mut writes: Vec<vk::WriteDescriptorSet> =
                        Vec::with_capacity(valid_bindings.len() + usize::from(has_ubo));
                    if has_ubo {
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(s0)
                                .dst_binding(0)
                                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                .buffer_info(&ubo_info),
                        );
                    }
                    for &(b, storage, idx) in &valid_bindings {
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(s0)
                                .dst_binding(b)
                                .descriptor_type(if storage {
                                    vk::DescriptorType::STORAGE_IMAGE
                                } else {
                                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                                })
                                .image_info(&img_infos[idx]),
                        );
                    }
                    if !writes.is_empty() {
                        unsafe {
                            ctx.device.update_descriptor_sets(&writes, &[]);
                        }
                    }
                    self.descriptor_sig_cache.insert(cache_key, sig);
                }
            }
            self.dispatch_compute_bundle(
                ctx,
                cmd,
                bundle,
                s0,
                s1,
                bindless,
                pack_set,
                tlas_handle,
                slot,
                width,
                height,
                None,
                blas,
            );
        }

        // Feed the *temporal* accumulation result into the history buffer. The à-trous output must
        // never be written back as history: doing that re-filtered the accumulated image every
        // frame and turned the denoiser into an unbounded low-pass filter (the "soapy" look).
        self.copy_temporal_history_pair(
            ctx,
            cmd,
            graph,
            RenderGraph::RES_DENOISED_RADIANCE,
            RenderGraph::RES_SVGF_HISTORY_RADIANCE,
            width,
            height,
        );

        self.copy_temporal_history_pair(
            ctx,
            cmd,
            graph,
            RenderGraph::RES_SVGF_MOMENTS,
            RenderGraph::RES_SVGF_HISTORY_MOMENTS,
            width,
            height,
        );

        let Some(bundle) = self.compute_pipelines.get("svgf_atrous.comp") else {
            return;
        };
        let Some(reverse_sets) = bundle.atrous_reverse_sets else {
            return;
        };
        let s1 = bundle.descriptor_sets_set1[slot];
        let normal_view = Self::image_view_or_null(graph, RenderGraph::RES_GBUFFER_NORMAL);
        let depth_view = Self::image_view_or_null(graph, active_depth);
        let denoised = RenderGraph::RES_DENOISED_RADIANCE;
        let ping = RenderGraph::RES_SVGF_PING;
        let (Some(&denoised_img), Some(&ping_img)) = (
            graph.resource_images.get(&denoised),
            graph.resource_images.get(&ping),
        ) else {
            return;
        };
        let sharp_view = Self::image_view_or_null(graph, RenderGraph::RES_SHARP_RADIANCE);
        let forward = bundle.descriptor_sets_set0[slot];
        let reverse = reverse_sets[slot];

        // Descriptor contents remain fixed through command execution. Both directions need
        // their own set per frame slot even though they use the same compute pipeline.
        {
            let denoised_view = Self::image_view_or_null(graph, denoised);
            let ping_view = Self::image_view_or_null(graph, ping);
            let mut sig = fnv1a_init();
            sig = fnv1a_mix_u64(sig, forward.as_raw());
            sig = fnv1a_mix_u64(sig, reverse.as_raw());
            sig = fnv1a_mix_u64(sig, default_sampler.as_raw());
            for v in [
                denoised_view,
                ping_view,
                normal_view,
                depth_view,
                albedo_view,
                sharp_view,
            ] {
                sig = fnv1a_mix_u64(sig, v.as_raw());
            }
            let cache_key = ("svgf_atrous.comp".to_owned(), slot);
            if self.descriptor_sig_cache.get(&cache_key).copied() != Some(sig) {
                let mut img_infos: Vec<[vk::DescriptorImageInfo; 1]> = Vec::with_capacity(12);
                let mut valid_bindings: Vec<(vk::DescriptorSet, u32, bool, usize)> =
                    Vec::with_capacity(12);
                for (set, input_view, output_view) in [
                    (forward, denoised_view, ping_view),
                    (reverse, ping_view, denoised_view),
                ] {
                    if set == vk::DescriptorSet::null() {
                        continue;
                    }
                    for (binding, view) in [
                        (0, input_view),
                        (1, normal_view),
                        (2, depth_view),
                        (3, output_view),
                        (4, albedo_view),
                        (5, sharp_view),
                    ] {
                        if view == vk::ImageView::null()
                            || !bundle
                                .reflection
                                .bindings
                                .iter()
                                .any(|b| b.set == 0 && b.binding == binding)
                        {
                            continue;
                        }
                        let storage = binding == 3;
                        let idx = img_infos.len();
                        img_infos.push([vk::DescriptorImageInfo {
                            sampler: if storage {
                                vk::Sampler::null()
                            } else {
                                default_sampler
                            },
                            image_view: view,
                            image_layout: if storage {
                                vk::ImageLayout::GENERAL
                            } else {
                                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                            },
                        }]);
                        valid_bindings.push((set, binding, storage, idx));
                    }
                }
                if !valid_bindings.is_empty() {
                    let writes: Vec<vk::WriteDescriptorSet> = valid_bindings
                        .iter()
                        .map(|&(set, binding, storage, idx)| {
                            vk::WriteDescriptorSet::default()
                                .dst_set(set)
                                .dst_binding(binding)
                                .descriptor_type(if storage {
                                    vk::DescriptorType::STORAGE_IMAGE
                                } else {
                                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                                })
                                .image_info(&img_infos[idx])
                        })
                        .collect();
                    unsafe {
                        ctx.device.update_descriptor_sets(&writes, &[]);
                    }
                }
                self.descriptor_sig_cache.insert(cache_key, sig);
            }
        }

        for (iter_idx, (set, input, output, step)) in [
            (forward, denoised_img, ping_img, 1u32),
            (reverse, ping_img, denoised_img, 2u32),
            (forward, denoised_img, ping_img, 4u32),
            (reverse, ping_img, denoised_img, 0u32),
        ]
        .into_iter()
        .enumerate()
        {
            let barriers = [
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(input)
                    .subresource_range(color_subrange),
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .old_layout(if iter_idx == 0 {
                        vk::ImageLayout::GENERAL
                    } else {
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    })
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(output)
                    .subresource_range(color_subrange),
            ];
            unsafe {
                ctx.sync2_ext.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&barriers),
                );
            }
            let push = AtrousPushConstants {
                step_size: step as i32,
                ..AtrousPushConstants::default()
            };
            self.dispatch_compute_bundle(
                ctx,
                cmd,
                bundle,
                set,
                s1,
                bindless,
                pack_set,
                tlas_handle,
                slot,
                width,
                height,
                Some(bytemuck::bytes_of(&push)),
                blas,
            );
        }

        let restore = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(ping_img)
            .subresource_range(color_subrange)];
        unsafe {
            ctx.sync2_ext.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&restore),
            );
        }
    }

    /// Binds only contiguous non-null descriptor sets so `VK_NULL_HANDLE` is never passed to
    /// `vkCmdBindDescriptorSets`.
    unsafe fn bind_valid_descriptor_sets(
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        bind_point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        sets: &[vk::DescriptorSet],
    ) {
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
            unsafe {
                ctx.device.cmd_bind_descriptor_sets(
                    cmd,
                    bind_point,
                    layout,
                    start as u32,
                    &sets[start..idx],
                    &[],
                );
            }
        }
    }

    /// Records TSR upscaling via the generic compute path, used by the fallback branch in the graph.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_compute_bundle(
        &self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        bundle: &ComputePipelineBundle,
        s0: vk::DescriptorSet,
        s1: vk::DescriptorSet,
        bindless: Option<&BindlessTextureTable>,
        pack_set: vk::DescriptorSet,
        _tlas_handle: vk::AccelerationStructureKHR,
        _slot: usize,
        width: u32,
        height: u32,
        push: Option<&[u8]>,
        _blas: Option<&ZeroCopyBlasManager>,
    ) {
        let wg_x = bundle.workgroup_size[0].max(1);
        let wg_y = bundle.workgroup_size[1].max(1);
        unsafe {
            ctx.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, bundle.pipeline);
            let sets = [
                s0,
                s1,
                bindless
                    .map(|b| b.descriptor_set)
                    .unwrap_or(vk::DescriptorSet::null()),
                pack_set,
            ];
            Self::bind_valid_descriptor_sets(
                ctx,
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                bundle.pipeline_layout,
                &sets,
            );
            if let Some(bytes) = push {
                ctx.device.cmd_push_constants(
                    cmd,
                    bundle.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytes,
                );
            }
            ctx.device
                .cmd_dispatch(cmd, width.div_ceil(wg_x), height.div_ceil(wg_y), 1);
        }
    }

    /// Copies `src_id` (in `GENERAL`) into `dst_id` (in `SHADER_READ_ONLY_OPTIMAL`) using explicit
    /// `TRANSFER_SRC_OPTIMAL` / `TRANSFER_DST_OPTIMAL` barriers and restores both layouts.
    // Source and destination resources need independent image dimensions and IDs.
    #[allow(clippy::too_many_arguments)]
    fn copy_temporal_history_pair(
        &self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        graph: &RenderGraph,
        src_id: ResourceId,
        dst_id: ResourceId,
        width: u32,
        height: u32,
    ) {
        if let (Some(&src), Some(&dst)) = (
            graph.resource_images.get(&src_id),
            graph.resource_images.get(&dst_id),
        ) && src != vk::Image::null()
            && dst != vk::Image::null()
        {
            let layer_count = graph
                .bound_images
                .get(&src_id)
                .map(|bound| bound.array_layers.max(1))
                .unwrap_or(1)
                .min(
                    graph
                        .bound_images
                        .get(&dst_id)
                        .map(|bound| bound.array_layers.max(1))
                        .unwrap_or(1),
                );
            let subrange = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count,
            };
            let pre = [
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(src)
                    .subresource_range(subrange),
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(dst)
                    .subresource_range(subrange),
            ];
            let post = [
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(
                        vk::AccessFlags2::SHADER_STORAGE_WRITE
                            | vk::AccessFlags2::SHADER_SAMPLED_READ,
                    )
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(src)
                    .subresource_range(subrange),
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(dst)
                    .subresource_range(subrange),
            ];
            let region = [vk::ImageCopy {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count,
                },
                src_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count,
                },
                dst_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                extent: vk::Extent3D {
                    width: width.max(1),
                    height: height.max(1),
                    depth: 1,
                },
            }];
            unsafe {
                let dep_pre = vk::DependencyInfo::default().image_memory_barriers(&pre);
                ctx.sync2_ext.cmd_pipeline_barrier2(cmd, &dep_pre);
                ctx.device.cmd_copy_image(
                    cmd,
                    src,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &region,
                );
                let dep_post = vk::DependencyInfo::default().image_memory_barriers(&post);
                ctx.sync2_ext.cmd_pipeline_barrier2(cmd, &dep_post);
            }
        }
    }

    /// After TSR upscaling, copy `RES_UPSCALED_HDR` into the linear RGBA16F TSR history so the next
    /// frame's `tsr_upscale.comp` reads a valid reprojection source.
    pub fn copy_upscaled_history(
        &self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        graph: &RenderGraph,
        width: u32,
        height: u32,
    ) {
        self.copy_temporal_history_pair(
            ctx,
            cmd,
            graph,
            RenderGraph::RES_UPSCALED_HDR,
            RenderGraph::RES_TSR_HISTORY,
            width,
            height,
        );
    }

    /// After `ssgi.comp` executes, copy `RES_SSGI_OUT` into `RES_SSGI_HISTORY` so the next frame's
    /// temporal accumulation in `ssgi.comp` reads valid history radiance.
    pub fn copy_ssgi_history(
        &self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        graph: &RenderGraph,
        width: u32,
        height: u32,
    ) {
        self.copy_temporal_history_pair(
            ctx,
            cmd,
            graph,
            RenderGraph::RES_SSGI_OUT,
            RenderGraph::RES_SSGI_HISTORY,
            width,
            height,
        );
    }

    /// Copies a custom temporal target (`src_id`) into its paired `<name>_history` target (`dst_id`).
    #[allow(clippy::too_many_arguments)]
    pub fn copy_custom_temporal_history(
        &self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        graph: &RenderGraph,
        src_id: ResourceId,
        dst_id: ResourceId,
        width: u32,
        height: u32,
    ) {
        self.copy_temporal_history_pair(ctx, cmd, graph, src_id, dst_id, width, height);
    }

    fn ensure_csm_quad_ibo(
        &mut self,
        ctx: &VulkanContext,
        required_quads: u32,
    ) -> Option<vk::Buffer> {
        let needed = required_quads.max(16_384).next_power_of_two();
        if self.csm_quad_ibo.is_none() || self.csm_quad_capacity < needed {
            if let Some(old) = self.csm_quad_ibo.take() {
                self.retired_csm_ibos.push(old);
            }
            let byte_size = (needed as vk::DeviceSize) * 6 * 4;
            let mut alloc = GpuBufferAllocation::allocate_host_visible_bda(
                ctx,
                byte_size,
                vk::BufferUsageFlags::INDEX_BUFFER,
            )
            .ok()?;
            let mut indices = Vec::with_capacity((needed as usize) * 6);
            for q in 0..needed {
                let base = q * 4;
                indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
            }
            alloc.write_bytes(0, bytemuck::cast_slice(&indices));
            self.csm_quad_capacity = needed;
            self.csm_quad_ibo = Some(alloc);
        }
        self.csm_quad_ibo.as_ref().map(|b| b.buffer)
    }

    /// Dispatches `denoiser_prepare.comp` to unpack G-Buffer + raw/sharp radiance into external
    /// denoiser inputs (`outDiff`, `outSpec`, `outNormalRough`, `outViewZ`, `outHitDist`).
    ///
    /// Set 0 bindings:
    /// - b0: `WorldUBO`
    /// - b1: `rawRadiance` (`RES_RAW_RADIANCE`)
    /// - b2: `sharpRadiance` (`RES_SHARP_RADIANCE`)
    /// - b3: `normalRough` (`RES_GBUFFER_NORMAL`)
    /// - b4: `depth` (`RES_LINEAR_DEPTH` on Tier 3, `RES_MC_MAIN_DEPTH` otherwise)
    /// - b5: `albedo` (`RES_PT_ALBEDO` on Tier 3, `RES_GBUFFER_ALBEDO` otherwise)
    /// - b6..b10: `outDiff`, `outSpec`, `outNormalRough`, `outViewZ`, `outHitDist` (storage images)
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_denoiser_prepare(
        &mut self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        graph: &RenderGraph,
        bindless: Option<&BindlessTextureTable>,
        pack_set: vk::DescriptorSet,
        tlas_handle: vk::AccelerationStructureKHR,
        frame_slot: usize,
        width: u32,
        height: u32,
        out_diff: vk::ImageView,
        out_spec: vk::ImageView,
        out_normal_rough: vk::ImageView,
        out_view_z: vk::ImageView,
        out_hit_dist: vk::ImageView,
        blas: Option<&ZeroCopyBlasManager>,
    ) -> bool {
        let slot = frame_slot % 3;
        let ubo_buf = self.world_ubo_buffer_for_slot(slot);
        let default_sampler = self.default_sampler;
        let Some(bundle) = self
            .compute_pipelines
            .get("denoiser_prepare.comp")
            .or_else(|| self.compute_pipelines.get("denoiser_prepare"))
        else {
            return false;
        };
        if bundle.pipeline == vk::Pipeline::null() {
            return false;
        }
        let s0 = bundle.descriptor_sets_set0[slot];
        let s1 = bundle.descriptor_sets_set1[slot];
        let (active_depth, active_albedo) = if graph.config.tier == RenderTier::Tier3PathTracing {
            (RenderGraph::RES_LINEAR_DEPTH, RenderGraph::RES_PT_ALBEDO)
        } else {
            (
                RenderGraph::RES_MC_MAIN_DEPTH,
                RenderGraph::RES_GBUFFER_ALBEDO,
            )
        };

        if s0 != vk::DescriptorSet::null() {
            let has_ubo = ubo_buf != vk::Buffer::null()
                && bundle
                    .reflection
                    .bindings
                    .iter()
                    .any(|b| b.set == 0 && b.binding == 0);
            let views = [
                (
                    1u32,
                    Self::image_view_or_null(graph, RenderGraph::RES_RAW_RADIANCE),
                ),
                (
                    2u32,
                    Self::image_view_or_null(graph, RenderGraph::RES_SHARP_RADIANCE),
                ),
                (
                    3u32,
                    Self::image_view_or_null(graph, RenderGraph::RES_GBUFFER_NORMAL),
                ),
                (4u32, Self::image_view_or_null(graph, active_depth)),
                (5u32, Self::image_view_or_null(graph, active_albedo)),
                (6u32, out_diff),
                (7u32, out_spec),
                (8u32, out_normal_rough),
                (9u32, out_view_z),
                (10u32, out_hit_dist),
            ];
            let mut sig = fnv1a_init();
            sig = fnv1a_mix_u64(sig, s0.as_raw());
            if has_ubo {
                sig = fnv1a_mix_u64(sig, ubo_buf.as_raw());
                sig = fnv1a_mix_u64(sig, 0);
                sig = fnv1a_mix_u64(sig, vk::WHOLE_SIZE);
            }
            for (b, v) in views {
                if v == vk::ImageView::null() {
                    continue;
                }
                let Some(refl) = bundle
                    .reflection
                    .bindings
                    .iter()
                    .find(|binding| binding.set == 0 && binding.binding == b)
                else {
                    continue;
                };
                let storage =
                    b >= 6 || refl.descriptor_type == ReflectedDescriptorType::StorageImage;
                let sampler = if storage {
                    vk::Sampler::null()
                } else {
                    default_sampler
                };
                sig = fnv1a_mix_u64(sig, u64::from(b));
                sig = fnv1a_mix_u64(sig, v.as_raw());
                sig = fnv1a_mix_u64(sig, sampler.as_raw());
            }
            let cache_key = ("denoiser_prepare.comp".to_owned(), slot);
            if self.descriptor_sig_cache.get(&cache_key).copied() != Some(sig) {
                let ubo_info = [vk::DescriptorBufferInfo {
                    buffer: ubo_buf,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                }];
                let mut img_infos: Vec<[vk::DescriptorImageInfo; 1]> =
                    Vec::with_capacity(views.len());
                let mut valid_bindings: Vec<(u32, bool, usize)> = Vec::with_capacity(views.len());
                for (b, v) in views {
                    if v == vk::ImageView::null() {
                        continue;
                    }
                    let Some(refl) = bundle
                        .reflection
                        .bindings
                        .iter()
                        .find(|binding| binding.set == 0 && binding.binding == b)
                    else {
                        continue;
                    };
                    let storage =
                        b >= 6 || refl.descriptor_type == ReflectedDescriptorType::StorageImage;
                    let idx = img_infos.len();
                    img_infos.push([vk::DescriptorImageInfo {
                        sampler: if storage {
                            vk::Sampler::null()
                        } else {
                            default_sampler
                        },
                        image_view: v,
                        image_layout: if storage {
                            vk::ImageLayout::GENERAL
                        } else {
                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                        },
                    }]);
                    valid_bindings.push((b, storage, idx));
                }
                let mut writes: Vec<vk::WriteDescriptorSet> =
                    Vec::with_capacity(valid_bindings.len() + usize::from(has_ubo));
                if has_ubo {
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(s0)
                            .dst_binding(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .buffer_info(&ubo_info),
                    );
                }
                for &(b, storage, idx) in &valid_bindings {
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(s0)
                            .dst_binding(b)
                            .descriptor_type(if storage {
                                vk::DescriptorType::STORAGE_IMAGE
                            } else {
                                vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                            })
                            .image_info(&img_infos[idx]),
                    );
                }
                if !writes.is_empty() {
                    unsafe {
                        ctx.device.update_descriptor_sets(&writes, &[]);
                    }
                }
                self.descriptor_sig_cache.insert(cache_key, sig);
            }
        }

        let is_tier3 = u32::from(graph.config.tier == RenderTier::Tier3PathTracing);
        let encode_unorm = u32::from(graph.config.denoiser_mode == 2);
        let prepare_pc: [u32; 4] = [is_tier3, width, height, encode_unorm];
        let prepare_pc_bytes: [u8; 16] = unsafe { std::mem::transmute(prepare_pc) };
        self.dispatch_compute_bundle(
            ctx,
            cmd,
            bundle,
            s0,
            s1,
            bindless,
            pack_set,
            tlas_handle,
            slot,
            width,
            height,
            Some(&prepare_pc_bytes),
            blas,
        );
        true
    }

    /// Dispatches `denoiser_recompose.comp` to demodulate/recompose denoised diffuse & specular
    /// radiance back into `RES_DENOISED_RADIANCE` (and optional `RES_SHARP_RADIANCE`).
    ///
    /// Set 0 bindings:
    /// - b0: `inDiff`
    /// - b1: `inSpec`
    /// - b2: `inAlbedo` (`RES_PT_ALBEDO` on Tier 3, `RES_GBUFFER_ALBEDO` otherwise)
    /// - b3: `outDenoised` (`RES_DENOISED_RADIANCE`, storage image)
    /// - b4: `outSharp` (`RES_SHARP_RADIANCE`)
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_denoiser_recompose(
        &mut self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        graph: &RenderGraph,
        bindless: Option<&BindlessTextureTable>,
        pack_set: vk::DescriptorSet,
        tlas_handle: vk::AccelerationStructureKHR,
        frame_slot: usize,
        width: u32,
        height: u32,
        in_diff: vk::ImageView,
        in_spec: vk::ImageView,
        blas: Option<&ZeroCopyBlasManager>,
    ) -> bool {
        let default_sampler = self.default_sampler;
        let Some(bundle) = self
            .compute_pipelines
            .get("denoiser_recompose.comp")
            .or_else(|| self.compute_pipelines.get("denoiser_recompose"))
        else {
            return false;
        };
        if bundle.pipeline == vk::Pipeline::null() {
            return false;
        }
        let slot = frame_slot % 3;
        let s0 = bundle.descriptor_sets_set0[slot];
        let s1 = bundle.descriptor_sets_set1[slot];
        let active_albedo = if graph.config.tier == RenderTier::Tier3PathTracing {
            RenderGraph::RES_PT_ALBEDO
        } else {
            RenderGraph::RES_GBUFFER_ALBEDO
        };

        if s0 != vk::DescriptorSet::null() {
            let views = [
                (0u32, in_diff),
                (1u32, in_spec),
                (2u32, Self::image_view_or_null(graph, active_albedo)),
                (
                    3u32,
                    Self::image_view_or_null(graph, RenderGraph::RES_DENOISED_RADIANCE),
                ),
                (
                    4u32,
                    Self::image_view_or_null(graph, RenderGraph::RES_SHARP_RADIANCE),
                ),
            ];
            let mut sig = fnv1a_init();
            sig = fnv1a_mix_u64(sig, s0.as_raw());
            for (b, v) in views {
                if v == vk::ImageView::null() {
                    continue;
                }
                let Some(refl) = bundle
                    .reflection
                    .bindings
                    .iter()
                    .find(|binding| binding.set == 0 && binding.binding == b)
                else {
                    continue;
                };
                let storage =
                    b == 3 || refl.descriptor_type == ReflectedDescriptorType::StorageImage;
                let sampler = if storage {
                    vk::Sampler::null()
                } else {
                    default_sampler
                };
                sig = fnv1a_mix_u64(sig, u64::from(b));
                sig = fnv1a_mix_u64(sig, v.as_raw());
                sig = fnv1a_mix_u64(sig, sampler.as_raw());
            }
            let cache_key = ("denoiser_recompose.comp".to_owned(), slot);
            if self.descriptor_sig_cache.get(&cache_key).copied() != Some(sig) {
                let mut img_infos: Vec<[vk::DescriptorImageInfo; 1]> =
                    Vec::with_capacity(views.len());
                let mut valid_bindings: Vec<(u32, bool, usize)> = Vec::with_capacity(views.len());
                for (b, v) in views {
                    if v == vk::ImageView::null() {
                        continue;
                    }
                    let Some(refl) = bundle
                        .reflection
                        .bindings
                        .iter()
                        .find(|binding| binding.set == 0 && binding.binding == b)
                    else {
                        continue;
                    };
                    let storage =
                        b == 3 || refl.descriptor_type == ReflectedDescriptorType::StorageImage;
                    let idx = img_infos.len();
                    img_infos.push([vk::DescriptorImageInfo {
                        sampler: if storage {
                            vk::Sampler::null()
                        } else {
                            default_sampler
                        },
                        image_view: v,
                        image_layout: if storage {
                            vk::ImageLayout::GENERAL
                        } else {
                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                        },
                    }]);
                    valid_bindings.push((b, storage, idx));
                }
                if !valid_bindings.is_empty() {
                    let writes: Vec<vk::WriteDescriptorSet> = valid_bindings
                        .iter()
                        .map(|&(b, storage, idx)| {
                            vk::WriteDescriptorSet::default()
                                .dst_set(s0)
                                .dst_binding(b)
                                .descriptor_type(if storage {
                                    vk::DescriptorType::STORAGE_IMAGE
                                } else {
                                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                                })
                                .image_info(&img_infos[idx])
                        })
                        .collect();
                    unsafe {
                        ctx.device.update_descriptor_sets(&writes, &[]);
                    }
                }
                self.descriptor_sig_cache.insert(cache_key, sig);
            }
        }

        let recompose_pc: [u32; 4] = [width, height, 0, 0];
        let recompose_pc_bytes: [u8; 16] = unsafe { std::mem::transmute(recompose_pc) };
        self.dispatch_compute_bundle(
            ctx,
            cmd,
            bundle,
            s0,
            s1,
            bindless,
            pack_set,
            tlas_handle,
            slot,
            width,
            height,
            Some(&recompose_pc_bytes),
            blas,
        );
        true
    }

    /// Renders real cascaded shadow depth for every registered non-translucent terrain slice using
    /// the `csm_shadow.vsh/.fsh` (40B) and `csm_shadow_28b.vsh/.fsh` (28B) graphics pipelines,
    /// binding each slice's vertex buffer and the shared `uint32` quad index buffer per cascade layer.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_csm_shadow(
        &mut self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        graph: &RenderGraph,
        blas: &ZeroCopyBlasManager,
        bindless: Option<&BindlessTextureTable>,
        pack_set: vk::DescriptorSet,
        _tlas_handle: vk::AccelerationStructureKHR,
        frame_slot: usize,
        _width: u32,
        _height: u32,
        sun_dir: [f32; 3],
    ) {
        let slot = frame_slot % 3;
        let max_shadow_res = ctx.capabilities.max_shadow_resolution();
        let shadow_res = if graph.shadow_resolution >= 512 {
            graph.shadow_resolution.clamp(512, max_shadow_res)
        } else if self.shadow_resolution >= 512 {
            self.shadow_resolution.clamp(512, max_shadow_res)
        } else {
            2048
        };
        let shadow_dist_chunks = if graph.shadow_distance_chunks >= 4 {
            graph.shadow_distance_chunks.clamp(4, 64)
        } else if self.shadow_distance_chunks >= 4 {
            self.shadow_distance_chunks.clamp(4, 64)
        } else {
            12
        };
        let max_shadow_dist_blocks = (shadow_dist_chunks * 16) as f32;

        // Match compute_lighting.comp / volumetric_fog.comp night-time moon light direction
        // so shadows at night are cast from the moon rather than from underground.
        let effective_light_dir = if sun_dir[1] <= -0.04 {
            let mx = -sun_dir[0];
            let my = (-sun_dir[1]).max(0.14);
            let mz = -sun_dir[2];
            let inv_len = 1.0 / (mx * mx + my * my + mz * mz).sqrt().max(1e-6);
            [mx * inv_len, my * inv_len, mz * inv_len]
        } else {
            sun_dir
        };
        // Triple-buffered CsmShadowUBO per frame slot with world-locked texel snapping.
        if self.csm_ubo_slots[slot].is_none()
            && let Ok(mut alloc) = GpuBufferAllocation::allocate_host_visible_bda(
                ctx,
                std::mem::size_of::<CsmShadowUBO>() as vk::DeviceSize,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )
        {
            let zero_ubo: CsmShadowUBO = bytemuck::Zeroable::zeroed();
            alloc.write_bytes(0, bytemuck::bytes_of(&zero_ubo));
            self.csm_ubo_slots[slot] = Some(alloc);
        }
        let mut csm_ubo = CsmShadowUBO::from_sun_dir_snapped(
            effective_light_dir,
            shadow_res as f32,
            max_shadow_dist_blocks,
            self.camera_pos_world,
        );

        let cam_block = self.camera_block_pos;
        let max_dist_cull = max_shadow_dist_blocks + 32.0;
        let max_dist_cull_sq = max_dist_cull * max_dist_cull;

        struct ShadowDrawSlice {
            is_cutout: bool,
            vk_buffer: u64,
            byte_offset: u64,
            vertex_stride: u32,
            index_count: u32,
            origin: [i32; 3],
            min_rel: [f32; 3],
            max_rel: [f32; 3],
        }

        let has_28b_pipeline = self.graphics_pipelines.contains_key("csm_shadow_28b");
        let mut max_quads = 0u32;
        let mut cached_atlas: Option<(vk::ImageView, vk::Sampler)> = None;
        let mut geometry_sig = 0u64;
        let mut draw_slices: Vec<ShadowDrawSlice> = Vec::with_capacity(blas.slices.len().min(2048));
        for slice in blas.slices.values() {
            if cached_atlas.is_none()
                && (slice.vertex_stride == 40 || slice.vertex_stride == 28)
                && let Some(table) = bindless
            {
                cached_atlas = table.view_sampler_for_slot(slice.bindless_albedo_slot);
            }
            if slice.key.layer_ordinal == 2
                || slice.vk_buffer == 0
                || slice.vertex_count < 4
                || !(slice.vertex_stride == 40 || (slice.vertex_stride == 28 && has_28b_pipeline))
            {
                continue;
            }
            let (min_rel, max_rel) = slice.section_aabb_rel(cam_block);
            let cx = (min_rel[0] + max_rel[0]) * 0.5;
            let cz = (min_rel[2] + max_rel[2]) * 0.5;
            if cx * cx + cz * cz > max_dist_cull_sq {
                continue;
            }
            let quads = slice.vertex_count / 4;
            let index_count = quads * 6;
            if index_count == 0 {
                continue;
            }
            max_quads = max_quads.max(quads);
            let slice_sig = fnv1a_mix_u64(
                fnv1a_mix_u64(
                    fnv1a_mix_u64(slice.key.allocation_key_id, slice.vk_buffer),
                    slice.byte_offset,
                ),
                u64::from(slice.vertex_count),
            );
            geometry_sig = geometry_sig.wrapping_add(slice_sig);
            let (ax, ay, az) = slice.world_anchor_blocks();
            let origin = [
                ax.wrapping_sub(cam_block[0]),
                ay.wrapping_sub(cam_block[1]),
                az.wrapping_sub(cam_block[2]),
            ];
            draw_slices.push(ShadowDrawSlice {
                is_cutout: slice.key.layer_ordinal != 0,
                vk_buffer: slice.vk_buffer,
                byte_offset: slice.byte_offset,
                vertex_stride: slice.vertex_stride,
                index_count,
                origin,
                min_rel,
                max_rel,
            });
        }
        if max_quads == 0 || draw_slices.is_empty() {
            let zero_ubo: CsmShadowUBO = bytemuck::Zeroable::zeroed();
            if let Some(ref mut buf) = self.csm_ubo_slots[slot] {
                buf.write_bytes(0, bytemuck::bytes_of(&zero_ubo));
            }
            return;
        }
        // Sort solid (depth-only BACK-culled) before cutout (alpha-tested NONE-culled), then by
        // (vertex_stride, vk_buffer, origin, byte_offset) to minimize pipeline and VBO switches.
        draw_slices.sort_unstable_by_key(|s| {
            (
                s.is_cutout,
                s.vertex_stride,
                s.vk_buffer,
                s.origin,
                s.byte_offset,
            )
        });

        let quad_ibo = if let Some(ref fb) = blas.fallback_quad_ibo
            && blas.fallback_quad_capacity >= max_quads
        {
            fb.buffer
        } else {
            match self.ensure_csm_quad_ibo(ctx, max_quads) {
                Some(b) => b,
                None => return,
            }
        };

        let Some(bundle_cutout_40) = self.graphics_pipelines.get("csm_shadow") else {
            return;
        };
        let bundle_cutout_28 = self.graphics_pipelines.get("csm_shadow_28b");
        let bundle_solid_40 = self
            .graphics_pipelines
            .get("csm_shadow_solid")
            .unwrap_or(bundle_cutout_40);
        let bundle_solid_28 = self
            .graphics_pipelines
            .get("csm_shadow_28b_solid")
            .or(bundle_cutout_28);
        let Some(&csm_ubo_buf) = self.csm_ubo_slots[slot].as_ref().map(|b| &b.buffer) else {
            return;
        };
        let world_buf = self.world_ubo_buffer_for_slot(slot);
        if world_buf == vk::Buffer::null() {
            return;
        }
        let Some((atlas_view, atlas_sampler)) = cached_atlas.or_else(|| {
            bindless.and_then(|table| {
                table
                    .view_sampler_for_slot(table.primary_pbr_slots().0)
                    .or_else(|| table.view_sampler_for_slot(0))
            })
        }) else {
            return;
        };

        let eff_sampler = if atlas_sampler != vk::Sampler::null() {
            atlas_sampler
        } else {
            self.default_sampler
        };
        unsafe {
            if self.csm_bound_world_bufs[slot] != world_buf
                || self.csm_bound_ubo_bufs[slot] != csm_ubo_buf
            {
                let world_info = [vk::DescriptorBufferInfo {
                    buffer: world_buf,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                }];
                let csm_info = [vk::DescriptorBufferInfo {
                    buffer: csm_ubo_buf,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                }];
                let mut writes = Vec::with_capacity(8);
                for b in [
                    Some(bundle_cutout_40),
                    bundle_cutout_28,
                    self.graphics_pipelines.get("csm_shadow_solid"),
                    self.graphics_pipelines.get("csm_shadow_28b_solid"),
                ]
                .into_iter()
                .flatten()
                {
                    let s0 = b.descriptor_sets_set0[slot];
                    if s0 != vk::DescriptorSet::null() {
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(s0)
                                .dst_binding(0)
                                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                .buffer_info(&world_info),
                        );
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(s0)
                                .dst_binding(1)
                                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                .buffer_info(&csm_info),
                        );
                    }
                }
                if !writes.is_empty() {
                    ctx.device.update_descriptor_sets(&writes, &[]);
                }
                self.csm_bound_world_bufs[slot] = world_buf;
                self.csm_bound_ubo_bufs[slot] = csm_ubo_buf;
            }
            if self.csm_bound_atlas_views[slot] != atlas_view
                || self.csm_bound_atlas_samplers[slot] != eff_sampler
            {
                let atlas_info = [vk::DescriptorImageInfo {
                    sampler: eff_sampler,
                    image_view: atlas_view,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }];
                let mut writes = Vec::with_capacity(2);
                for b in [Some(bundle_cutout_40), bundle_cutout_28]
                    .into_iter()
                    .flatten()
                {
                    let s1 = b.descriptor_sets_set1[slot];
                    if s1 != vk::DescriptorSet::null() {
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(s1)
                                .dst_binding(0)
                                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .image_info(&atlas_info),
                        );
                    }
                }
                if !writes.is_empty() {
                    ctx.device.update_descriptor_sets(&writes, &[]);
                }
                self.csm_bound_atlas_views[slot] = atlas_view;
                self.csm_bound_atlas_samplers[slot] = eff_sampler;
            }
        }
        let Some(&shadow_image) = graph.resource_images.get(&RenderGraph::RES_SHADOW_CSM) else {
            return;
        };
        if self.csm_layer_view_image != shadow_image
            || self.csm_ubo_shadow_res != shadow_res
            || self.csm_ubo_shadow_dist != shadow_dist_chunks
        {
            self.csm_cascade_valid = [false; 4];
            self.csm_inactive_cleared = [false; 4];
        }
        if self.csm_layer_view_image != shadow_image {
            for (i, view) in self.csm_layer_views.iter_mut().enumerate() {
                if *view != vk::ImageView::null() {
                    self.retired_csm_views.push(*view);
                    *view = vk::ImageView::null();
                }
                let info = vk::ImageViewCreateInfo::default()
                    .image(shadow_image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::D32_SFLOAT)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: i as u32,
                        layer_count: 1,
                    });
                if let Ok(v) = unsafe { ctx.device.create_image_view(&info, None) } {
                    *view = v;
                }
            }
            self.csm_layer_view_image = shadow_image;
        }
        let active_cascades = if graph.config.tier == RenderTier::Tier2RayQuery {
            2usize
        } else {
            4usize
        };
        // Amortize far cascades (Cascade 2 every 2nd frame, Cascade 3 every 4th frame) while
        // forcing an immediate refresh on sun rotation (> 0.005 rad), camera jump (> 16 blocks),
        // or chunk geometry changes. Reused cascades translate their cached light_vp to the
        // current camera-relative frame so shadows remain world-locked with zero swimming.
        let any_dirty_chunk = blas.slices.len() != self.csm_last_slice_count
            || geometry_sig != self.csm_last_geometry_sig;
        self.csm_frame_counter = self.csm_frame_counter.wrapping_add(1);
        let mut cascade_needs_update = [false; 4];
        let active_layer_count = active_cascades.min(4);
        for (layer, needs_update) in cascade_needs_update
            .iter_mut()
            .enumerate()
            .take(active_layer_count)
        {
            let needs = if layer < 2 || !self.csm_cascade_valid[layer] || any_dirty_chunk {
                true
            } else {
                let last_dir = self.csm_cascade_last_sun_dir[layer];
                let sun_dot = (effective_light_dir[0] * last_dir[0]
                    + effective_light_dir[1] * last_dir[1]
                    + effective_light_dir[2] * last_dir[2])
                    .clamp(-1.0, 1.0);
                let sun_moved = sun_dot < 0.999_987_5; // cos(0.005 rad)
                let last_cam = self.csm_cascade_last_cam_pos[layer];
                let dx = self.camera_pos_world[0] - last_cam[0];
                let dy = self.camera_pos_world[1] - last_cam[1];
                let dz = self.camera_pos_world[2] - last_cam[2];
                let cam_moved = (dx * dx + dy * dy + dz * dz) > 256.0; // > 16 blocks
                let cadence_hit = if layer == 2 {
                    self.csm_frame_counter.is_multiple_of(2)
                } else {
                    self.csm_frame_counter.is_multiple_of(4)
                };
                sun_moved || cam_moved || cadence_hit
            };
            *needs_update = needs;
            self.csm_inactive_cleared[layer] = false;
            if needs {
                self.csm_cached_cascade_vp[layer] = csm_ubo.cascade_view_proj[layer];
                self.csm_cascade_last_sun_dir[layer] = effective_light_dir;
                self.csm_cascade_last_cam_pos[layer] = self.camera_pos_world;
                self.csm_cascade_valid[layer] = true;
            } else {
                let last_cam = self.csm_cascade_last_cam_pos[layer];
                let dx = (self.camera_pos_world[0] - last_cam[0]) as f32;
                let dy = (self.camera_pos_world[1] - last_cam[1]) as f32;
                let dz = (self.camera_pos_world[2] - last_cam[2]) as f32;
                let mut m = self.csm_cached_cascade_vp[layer];
                let x_column = m[0];
                let y_column = m[1];
                let z_column = m[2];
                for (row, value) in m[3][..3].iter_mut().enumerate() {
                    *value += x_column[row] * dx + y_column[row] * dy + z_column[row] * dz;
                }
                csm_ubo.cascade_view_proj[layer] = m;
            }
        }
        for (layer, needs_update) in cascade_needs_update
            .iter_mut()
            .enumerate()
            .skip(active_layer_count)
        {
            if !self.csm_inactive_cleared[layer] {
                *needs_update = true;
                self.csm_inactive_cleared[layer] = true;
            }
        }
        self.csm_last_slice_count = blas.slices.len();
        self.csm_last_geometry_sig = geometry_sig;

        // All prerequisites verified; publish the valid CsmShadowUBO once for this frame slot.
        self.csm_ubo_sun_dir = effective_light_dir;
        self.csm_ubo_shadow_res = shadow_res;
        self.csm_ubo_shadow_dist = shadow_dist_chunks;
        if let Some(ref mut buf) = self.csm_ubo_slots[slot] {
            buf.write_bytes(0, bytemuck::bytes_of(&csm_ubo));
        }

        unsafe {
            ctx.device
                .cmd_bind_index_buffer(cmd, quad_ibo, 0, vk::IndexType::UINT32);
            let viewport = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: shadow_res as f32,
                height: shadow_res as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];
            let scissor = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: shadow_res,
                    height: shadow_res,
                },
            }];
            ctx.device.cmd_set_viewport(cmd, 0, &viewport);
            ctx.device.cmd_set_scissor(cmd, 0, &scissor);

            // Build indirect draw lists only for cascades updating this frame.
            let cmd_stride = std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u64;
            let mut all_indirect_cmds: Vec<vk::DrawIndexedIndirectCommand> = Vec::new();
            let mut layer_groups: [Vec<ShadowLayerGroup>; 4] =
                [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            let mut layer_base_cmd: [u64; 4] = [0; 4];

            for layer in 0..active_cascades.min(4) {
                if !cascade_needs_update[layer] {
                    continue;
                }
                layer_base_cmd[layer] = all_indirect_cmds.len() as u64;
                for slice in &draw_slices {
                    if !csm_ubo.is_aabb_visible_in_cascade(layer, slice.min_rel, slice.max_rel) {
                        continue;
                    }
                    let stride = u64::from(slice.vertex_stride);
                    let (bind_offset, vertex_offset) = if slice.byte_offset % stride == 0
                        && (slice.byte_offset / stride) <= (i32::MAX as u64)
                    {
                        (0u64, (slice.byte_offset / stride) as i32)
                    } else {
                        (slice.byte_offset, 0i32)
                    };
                    let local_cmd_idx =
                        (all_indirect_cmds.len() as u64 - layer_base_cmd[layer]) as u32;
                    all_indirect_cmds.push(vk::DrawIndexedIndirectCommand {
                        index_count: slice.index_count,
                        instance_count: 1,
                        first_index: 0,
                        vertex_offset,
                        first_instance: 0,
                    });
                    let groups = &mut layer_groups[layer];
                    if let Some(last) = groups.last_mut()
                        && last.0 == slice.is_cutout
                        && last.1 == slice.vertex_stride
                        && last.2 == slice.vk_buffer
                        && last.3 == bind_offset
                        && last.4 == slice.origin
                    {
                        last.6 += 1;
                    } else {
                        groups.push((
                            slice.is_cutout,
                            slice.vertex_stride,
                            slice.vk_buffer,
                            bind_offset,
                            slice.origin,
                            local_cmd_idx,
                            1,
                        ));
                    }
                }
            }

            let total_needed_bytes = (all_indirect_cmds.len() as u64) * cmd_stride;
            if total_needed_bytes > 0
                && self.csm_indirect_slots[slot]
                    .as_ref()
                    .is_none_or(|b| b.size < total_needed_bytes)
            {
                let alloc_bytes =
                    (4 * 4096 * cmd_stride).max(total_needed_bytes + total_needed_bytes / 2);
                if let Ok(new_buf) = GpuBufferAllocation::allocate_host_visible_bda(
                    ctx,
                    alloc_bytes,
                    vk::BufferUsageFlags::INDIRECT_BUFFER,
                ) && let Some(old_ind) = self.csm_indirect_slots[slot].replace(new_buf)
                {
                    self.retired_csm_ibos.push(old_ind);
                }
            }

            let mut active_ind_vk = vk::Buffer::null();
            if total_needed_bytes > 0
                && let Some(ref mut ind_buf) = self.csm_indirect_slots[slot]
                && ind_buf.size >= total_needed_bytes
            {
                let raw_bytes = std::slice::from_raw_parts(
                    all_indirect_cmds.as_ptr() as *const u8,
                    all_indirect_cmds.len() * (cmd_stride as usize),
                );
                ind_buf.write_bytes(0, raw_bytes);
                active_ind_vk = ind_buf.buffer;
                ctx.cmd_memory_barrier2(
                    cmd,
                    vk::PipelineStageFlags2::HOST,
                    vk::AccessFlags2::HOST_WRITE,
                    vk::PipelineStageFlags2::DRAW_INDIRECT,
                    vk::AccessFlags2::INDIRECT_COMMAND_READ,
                );
            }

            let bindless_set = bindless
                .map(|b| b.descriptor_set)
                .unwrap_or(vk::DescriptorSet::null());
            let mut active_pipe_key: Option<(bool, u32)> = None;
            let mut active_layout = bundle_solid_40.pipeline_layout;

            for (layer, view) in self.csm_layer_views.iter().enumerate() {
                if *view == vk::ImageView::null() || !cascade_needs_update[layer] {
                    continue;
                }
                let attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(*view)
                    .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    });
                let rendering = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: shadow_res,
                            height: shadow_res,
                        },
                    })
                    .layer_count(1)
                    .depth_attachment(&attachment);
                ctx.dyn_rendering_ext.cmd_begin_rendering(cmd, &rendering);
                if layer < active_cascades && active_ind_vk != vk::Buffer::null() {
                    let layer_base_bytes = layer_base_cmd[layer] * cmd_stride;
                    let mut last_origin: Option<[i32; 3]> = None;
                    let mut last_vbo: (u64, u64) = (0, u64::MAX);
                    for &(is_cutout, stride, vk_buf, bind_off, origin, first_cmd, draw_count) in
                        &layer_groups[layer]
                    {
                        if active_pipe_key != Some((is_cutout, stride)) {
                            let target = match (is_cutout, stride) {
                                (false, 28) => match bundle_solid_28 {
                                    Some(b) => b,
                                    None => continue,
                                },
                                (false, _) => bundle_solid_40,
                                (true, 28) => match bundle_cutout_28 {
                                    Some(b) => b,
                                    None => continue,
                                },
                                (true, _) => bundle_cutout_40,
                            };
                            ctx.device.cmd_bind_pipeline(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                target.pipeline,
                            );
                            let target_sets = [
                                target.descriptor_sets_set0[slot],
                                target.descriptor_sets_set1[slot],
                                bindless_set,
                                pack_set,
                            ];
                            Self::bind_valid_descriptor_sets(
                                ctx,
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                target.pipeline_layout,
                                &target_sets,
                            );
                            active_layout = target.pipeline_layout;
                            active_pipe_key = Some((is_cutout, stride));
                            last_origin = None;
                        }
                        if last_origin != Some(origin) {
                            let push = CsmShadowPush {
                                region_origin: origin,
                                cascade_index: layer as u32,
                            };
                            ctx.device.cmd_push_constants(
                                cmd,
                                active_layout,
                                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                                0,
                                bytemuck::bytes_of(&push),
                            );
                            last_origin = Some(origin);
                        }
                        if last_vbo != (vk_buf, bind_off) {
                            let vb = [vk::Buffer::from_raw(vk_buf)];
                            let offsets = [bind_off];
                            ctx.device.cmd_bind_vertex_buffers(cmd, 0, &vb, &offsets);
                            last_vbo = (vk_buf, bind_off);
                        }
                        ctx.device.cmd_draw_indexed_indirect(
                            cmd,
                            active_ind_vk,
                            layer_base_bytes + (first_cmd as u64) * cmd_stride,
                            draw_count,
                            cmd_stride as u32,
                        );
                    }
                }
                ctx.dyn_rendering_ext.cmd_end_rendering(cmd);
            }
        }
    }

    /// Tier 3 depth reprojection as a real dynamic GRAPHICS pass: a fullscreen triangle rendered
    /// with `depth_blit.vsh/.fsh` into Minecraft's `RES_MC_MAIN_DEPTH` attachment with
    /// `depthCompareOp = ALWAYS` and `depthWriteEnable = true`.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_depth_blit(
        &mut self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        graph: &RenderGraph,
        bindless: Option<&BindlessTextureTable>,
        pack_set: vk::DescriptorSet,
        frame_slot: usize,
        width: u32,
        height: u32,
    ) {
        let slot = frame_slot % 3;
        let Some(bundle) = self.graphics_pipelines.get("depth_blit") else {
            return;
        };
        let Some(&_depth_image) = graph.resource_images.get(&RenderGraph::RES_MC_MAIN_DEPTH) else {
            return;
        };
        let Some(depth_view) = graph.resolve_image_view(RenderGraph::RES_MC_MAIN_DEPTH) else {
            return;
        };
        let linear_view = Self::image_view_or_null(graph, RenderGraph::RES_LINEAR_DEPTH);
        if linear_view == vk::ImageView::null() {
            return;
        }
        let set0 = bundle.descriptor_sets_set0[slot];
        let set1 = bundle.descriptor_sets_set1[slot];
        let params = self
            .last_overlay_depth_params
            .unwrap_or(DepthReprojectPushConstants {
                z_scale: -1.0,
                z_offset: -0.1,
                w_scale: -1.0,
                w_offset: 0.0,
                source_size: [width.max(1), height.max(1)],
                target_size: [width.max(1), height.max(1)],
            });
        let target_w = params.target_size[0].max(1);
        let target_h = params.target_size[1].max(1);
        let depth_sampler = if self.nearest_sampler != vk::Sampler::null() {
            self.nearest_sampler
        } else {
            self.default_sampler
        };
        unsafe {
            self.write_image_binding(ctx, set0, 0, linear_view, depth_sampler, false);
            ctx.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, bundle.pipeline);
            let sets = [
                set0,
                set1,
                bindless
                    .map(|b| b.descriptor_set)
                    .unwrap_or(vk::DescriptorSet::null()),
                pack_set,
            ];
            Self::bind_valid_descriptor_sets(
                ctx,
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                bundle.pipeline_layout,
                &sets,
            );
            let viewport = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: target_w as f32,
                height: target_h as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];
            let scissor = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: target_w,
                    height: target_h,
                },
            }];
            ctx.device.cmd_set_viewport(cmd, 0, &viewport);
            ctx.device.cmd_set_scissor(cmd, 0, &scissor);
            ctx.device.cmd_push_constants(
                cmd,
                bundle.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&params),
            );
            let attachment = vk::RenderingAttachmentInfo::default()
                .image_view(depth_view)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE);
            let rendering = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: target_w,
                        height: target_h,
                    },
                })
                .layer_count(1)
                .depth_attachment(&attachment);
            ctx.dyn_rendering_ext.cmd_begin_rendering(cmd, &rendering);
            ctx.device.cmd_draw(cmd, 3, 1, 0, 0);
            ctx.dyn_rendering_ext.cmd_end_rendering(cmd);
        }
    }

    fn retire_active_pipelines(&mut self, ctx: &VulkanContext, retire_queue: &RetireQueue) {
        self.descriptor_sig_cache.clear();
        self.csm_bound_world_bufs = [vk::Buffer::null(); 3];
        self.csm_bound_ubo_bufs = [vk::Buffer::null(); 3];
        self.csm_bound_atlas_views = [vk::ImageView::null(); 3];
        self.csm_bound_atlas_samplers = [vk::Sampler::null(); 3];
        self.csm_cascade_valid = [false; 4];
        self.csm_inactive_cleared = [false; 4];
        self.csm_last_slice_count = 0;
        self.csm_last_geometry_sig = 0;
        let compute: Vec<ComputePipelineBundle> =
            self.compute_pipelines.drain().map(|(_, v)| v).collect();
        let graphics: Vec<GraphicsPipelineBundle> =
            self.graphics_pipelines.drain().map(|(_, v)| v).collect();
        let mut rt = self.rt_pipeline.take();
        let modules: Vec<vk::ShaderModule> = self.shader_modules.drain().map(|(_, m)| m).collect();
        let dev = ctx.device.clone();

        if let Some(ref mut rt_bundle) = rt {
            rt_bundle.sbt.destroy(ctx, retire_queue);
        }

        let pool = std::mem::replace(&mut self.descriptor_pool, vk::DescriptorPool::null());
        retire_queue.retire_later(move || unsafe {
            for b in compute {
                if b.pipeline != vk::Pipeline::null() {
                    dev.destroy_pipeline(b.pipeline, None);
                }
                if b.pipeline_layout != vk::PipelineLayout::null() {
                    dev.destroy_pipeline_layout(b.pipeline_layout, None);
                }
                if !b.uses_external_bindless_set1
                    && b.set1_layout != vk::DescriptorSetLayout::null()
                {
                    dev.destroy_descriptor_set_layout(b.set1_layout, None);
                }
                if b.set0_layout != vk::DescriptorSetLayout::null() {
                    dev.destroy_descriptor_set_layout(b.set0_layout, None);
                }
            }
            for g in graphics {
                if g.pipeline != vk::Pipeline::null() {
                    dev.destroy_pipeline(g.pipeline, None);
                }
                if g.pipeline_layout != vk::PipelineLayout::null() {
                    dev.destroy_pipeline_layout(g.pipeline_layout, None);
                }
                if g.set1_layout != vk::DescriptorSetLayout::null() {
                    dev.destroy_descriptor_set_layout(g.set1_layout, None);
                }
                if g.set0_layout != vk::DescriptorSetLayout::null() {
                    dev.destroy_descriptor_set_layout(g.set0_layout, None);
                }
            }
            if let Some(r) = rt {
                if r.pipeline != vk::Pipeline::null() {
                    dev.destroy_pipeline(r.pipeline, None);
                }
                if r.pipeline_layout != vk::PipelineLayout::null() {
                    dev.destroy_pipeline_layout(r.pipeline_layout, None);
                }
                if r.set1_layout != vk::DescriptorSetLayout::null() {
                    dev.destroy_descriptor_set_layout(r.set1_layout, None);
                }
                if r.set0_layout != vk::DescriptorSetLayout::null() {
                    dev.destroy_descriptor_set_layout(r.set0_layout, None);
                }
            }
            for m in modules {
                if m != vk::ShaderModule::null() {
                    dev.destroy_shader_module(m, None);
                }
            }
            if pool != vk::DescriptorPool::null() {
                dev.destroy_descriptor_pool(pool, None);
            }
        });
    }

    pub fn destroy(&mut self, ctx: &VulkanContext, retire_queue: &RetireQueue) {
        self.retire_active_pipelines(ctx, retire_queue);
        let dev = ctx.device.clone();
        let ring = std::mem::take(&mut self.world_ubo_ring);
        let sharc_fb_ring = std::mem::take(&mut self.sharc_fallback_ubo_ring);
        let csm_ubo_slots = std::mem::take(&mut self.csm_ubo_slots);
        let csm_ind_slots = std::mem::take(&mut self.csm_indirect_slots);
        let retired_views = std::mem::take(&mut self.retired_csm_views);
        let retired_ibos = std::mem::take(&mut self.retired_csm_ibos);
        self.descriptor_sig_cache.clear();
        self.save_pipeline_cache_to_disk(ctx);
        let pcache = self.pipeline_cache;
        self.pipeline_cache = vk::PipelineCache::null();
        self.pipeline_cache_key = None;
        if pcache != vk::PipelineCache::null() {
            let dev = ctx.device.clone();
            retire_queue.retire_later(move || unsafe {
                dev.destroy_pipeline_cache(pcache, None);
            });
        }
        if csm_ubo_slots.iter().any(Option::is_some)
            || csm_ind_slots.iter().any(Option::is_some)
            || !retired_views.is_empty()
            || !retired_ibos.is_empty()
            || !sharc_fb_ring.is_empty()
        {
            let dev = ctx.device.clone();
            retire_queue.retire_later(move || unsafe {
                for mut b in sharc_fb_ring.into_iter().flatten() {
                    b.destroy(&dev);
                }
                for mut b in csm_ubo_slots.into_iter().flatten() {
                    b.destroy(&dev);
                }
                for mut b in csm_ind_slots.into_iter().flatten() {
                    b.destroy(&dev);
                }
                for v in retired_views {
                    dev.destroy_image_view(v, None);
                }
                for mut b in retired_ibos {
                    b.destroy(&dev);
                }
            });
        }
        let csm_quad_ibo = self.csm_quad_ibo.take();
        self.csm_quad_capacity = 0;
        let csm_layer_views = self.csm_layer_views;
        self.csm_layer_views = [vk::ImageView::null(); 4];
        self.csm_layer_view_image = vk::Image::null();
        let sampler = self.default_sampler;
        let nearest_sampler = self.nearest_sampler;
        self.default_sampler = vk::Sampler::null();
        self.nearest_sampler = vk::Sampler::null();

        retire_queue.retire_later(move || unsafe {
            for mut b in ring.into_iter().flatten() {
                b.destroy(&dev);
            }
            if let Some(mut b) = csm_quad_ibo {
                b.destroy(&dev);
            }
            for view in csm_layer_views {
                if view != vk::ImageView::null() {
                    dev.destroy_image_view(view, None);
                }
            }
            if sampler != vk::Sampler::null() {
                dev.destroy_sampler(sampler, None);
            }
            if nearest_sampler != vk::Sampler::null() {
                dev.destroy_sampler(nearest_sampler, None);
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Persistent SPIR-V & VkPipelineCache Disk Cache + Async Background Prewarmer
// ---------------------------------------------------------------------------

/// Inputs hashed by `blake3` to derive a deterministic disk cache key for SPIR-V and
/// `VkPipelineCache` blobs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineCacheKeyInput {
    pub pack_hash: [u8; 32],
    pub driver_uuid: [u8; 16],
    pub pipeline_cache_uuid: [u8; 16],
    pub spec_constants: Vec<(u32, u32)>,
}

impl PipelineCacheKeyInput {
    /// Computes `blake3(pack_hash || driver_uuid || pipeline_cache_uuid || sorted_spec_constants)`.
    pub fn compute_key(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"hyaloclast-pipeline-cache-v1");
        hasher.update(&self.pack_hash);
        hasher.update(&self.driver_uuid);
        hasher.update(&self.pipeline_cache_uuid);

        let mut sorted_specs = self.spec_constants.clone();
        sorted_specs.sort_unstable();
        hasher.update(&(sorted_specs.len() as u64).to_le_bytes());
        for (constant_id, raw_bits) in sorted_specs {
            hasher.update(&constant_id.to_le_bytes());
            hasher.update(&raw_bits.to_le_bytes());
        }
        *hasher.finalize().as_bytes()
    }

    /// Returns the lowercase 64-character hex string of [`Self::compute_key`].
    pub fn compute_hex(&self) -> String {
        encode_hex_32(&self.compute_key())
    }
}

fn encode_hex_32(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0F) as usize] as char);
    }
    out
}

/// Disk-backed cache for compiled SPIR-V binaries and Vulkan `VkPipelineCache` blobs keyed by
/// `blake3(pack_hash || driver_uuid || pipeline_cache_uuid || spec_constants)`.
#[derive(Debug, Clone)]
pub struct PersistentPipelineDiskCache {
    root_dir: std::path::PathBuf,
}

impl Default for PersistentPipelineDiskCache {
    fn default() -> Self {
        Self::new(Self::default_cache_dir())
    }
}

impl PersistentPipelineDiskCache {
    const SPIRV_MAGIC: u32 = 0x0723_0203;

    pub fn new(root_dir: std::path::PathBuf) -> Self {
        Self { root_dir }
    }

    pub fn default_cache_dir() -> std::path::PathBuf {
        if let Some(dir) = std::env::var_os("VS_CACHE_DIR") {
            std::path::PathBuf::from(dir)
        } else {
            std::path::PathBuf::from("config/vulkanshaders/cache")
        }
    }

    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    /// Deterministically hashes all shader/manifest files under `pack_dir` with `blake3`.
    pub fn compute_pack_hash(pack_dir: &Path) -> [u8; 32] {
        let mut files: Vec<std::path::PathBuf> = Vec::new();
        Self::collect_pack_files(pack_dir, &mut files);
        files.sort();

        let mut hasher = blake3::Hasher::new();
        hasher.update(b"hyaloclast-pack-v1");
        for path in files {
            if let Ok(rel) = path.strip_prefix(pack_dir) {
                let rel_str = rel.to_string_lossy().replace('\\', "/");
                hasher.update(rel_str.as_bytes());
                hasher.update(&[0u8]);
            }
            if let Ok(bytes) = std::fs::read(&path) {
                hasher.update(&(bytes.len() as u64).to_le_bytes());
                hasher.update(&bytes);
            }
        }
        *hasher.finalize().as_bytes()
    }

    fn collect_pack_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                Self::collect_pack_files(&path, out);
            } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                let lower = ext.to_ascii_lowercase();
                if matches!(
                    lower.as_str(),
                    "toml"
                        | "vsl"
                        | "vsh"
                        | "fsh"
                        | "vert"
                        | "frag"
                        | "comp"
                        | "rgen"
                        | "rchit"
                        | "rmiss"
                        | "rahit"
                        | "glsl"
                ) {
                    out.push(path);
                }
            }
        }
    }

    /// Queries `(driver_uuid, pipeline_cache_uuid)` from the active Vulkan physical device.
    pub fn query_driver_uuids(ctx: &VulkanContext) -> ([u8; 16], [u8; 16]) {
        let mut id_props = vk::PhysicalDeviceIDProperties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut id_props);
        unsafe {
            ctx.instance
                .get_physical_device_properties2(ctx.physical_device, &mut props2);
        }
        let pipeline_cache_uuid = props2.properties.pipeline_cache_uuid;
        (id_props.driver_uuid, pipeline_cache_uuid)
    }

    pub fn pipeline_cache_path(&self, key: &[u8; 32]) -> std::path::PathBuf {
        self.root_dir
            .join("pipelines")
            .join(format!("{}.vkcache", encode_hex_32(key)))
    }

    pub fn spirv_stage_path(&self, key: &[u8; 32], stage_name: &str) -> std::path::PathBuf {
        let sanitized: String = stage_name
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        self.root_dir
            .join("spirv")
            .join(encode_hex_32(key))
            .join(format!("{sanitized}.spv"))
    }

    /// Loads a previously persisted `VkPipelineCache` binary blob from disk.
    pub fn load_pipeline_cache_blob(&self, key: &[u8; 32]) -> Option<Vec<u8>> {
        let bytes = std::fs::read(self.pipeline_cache_path(key)).ok()?;
        if bytes.is_empty() { None } else { Some(bytes) }
    }

    /// Atomically writes a `VkPipelineCache` binary blob to disk.
    pub fn save_pipeline_cache_blob(&self, key: &[u8; 32], blob: &[u8]) -> std::io::Result<()> {
        if blob.is_empty() {
            return Ok(());
        }
        let path = self.pipeline_cache_path(key);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("vkcache.tmp");
        std::fs::write(&tmp, blob)?;
        std::fs::rename(&tmp, &path)
    }

    /// Loads cached SPIR-V words for `stage_name` if present and valid (aligned + SPIR-V magic).
    pub fn load_spirv_words(&self, key: &[u8; 32], stage_name: &str) -> Option<Vec<u32>> {
        let bytes = std::fs::read(self.spirv_stage_path(key, stage_name)).ok()?;
        if bytes.len() < 20 || !bytes.len().is_multiple_of(4) {
            return None;
        }
        let words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect();
        if words.first().copied() != Some(Self::SPIRV_MAGIC) {
            return None;
        }
        Some(words)
    }

    /// Atomically persists compiled SPIR-V words for `stage_name` on disk.
    pub fn save_spirv_words(
        &self,
        key: &[u8; 32],
        stage_name: &str,
        words: &[u32],
    ) -> std::io::Result<()> {
        if words.is_empty() || words[0] != Self::SPIRV_MAGIC {
            return Ok(());
        }
        let path = self.spirv_stage_path(key, stage_name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for &w in words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        let tmp = path.with_extension("spv.tmp");
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, &path)
    }

    /// Creates a `VkPipelineCache` seeded with the disk blob for `key` (or empty if missing).
    pub fn create_vk_pipeline_cache(
        &self,
        ctx: &VulkanContext,
        key: &[u8; 32],
    ) -> Result<vk::PipelineCache, vk::Result> {
        let initial_data = self.load_pipeline_cache_blob(key).unwrap_or_default();
        let create_info = if initial_data.is_empty() {
            vk::PipelineCacheCreateInfo::default()
        } else {
            vk::PipelineCacheCreateInfo::default().initial_data(&initial_data)
        };
        unsafe { ctx.device.create_pipeline_cache(&create_info, None) }
    }

    /// Extracts bytes from `cache` via `vkGetPipelineCacheData` and persists them under `key`.
    pub fn flush_vk_pipeline_cache(
        &self,
        ctx: &VulkanContext,
        cache: vk::PipelineCache,
        key: &[u8; 32],
    ) -> std::io::Result<usize> {
        if cache == vk::PipelineCache::null() {
            return Ok(0);
        }
        let Ok(data) = (unsafe { ctx.device.get_pipeline_cache_data(cache) }) else {
            return Ok(0);
        };
        if data.is_empty() {
            return Ok(0);
        }
        self.save_pipeline_cache_blob(key, &data)?;
        Ok(data.len())
    }
}

/// Snapshot of asynchronous background shader/pipeline prewarming progress.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PrewarmProgress {
    pub completed: usize,
    pub total: usize,
    pub ready: bool,
    pub cancelled: bool,
    pub last_error: Option<String>,
}

/// Background worker (`hyaloclast-prewarm` thread) that compiles shader packs and prewarms
/// SPIR-V/pipeline caches asynchronously without stalling the render thread.
pub struct AsyncPipelinePrewarmer {
    cancel_token: std::sync::Arc<std::sync::atomic::AtomicBool>,
    progress: std::sync::Arc<parking_lot::Mutex<PrewarmProgress>>,
    artifacts: std::sync::Arc<parking_lot::Mutex<Option<Vec<CompiledShaderArtifact>>>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl AsyncPipelinePrewarmer {
    /// Launches asynchronous shader pack compilation on a dedicated `"hyaloclast-prewarm"` thread.
    pub fn start_pack_prewarm(
        pack_dir: std::path::PathBuf,
        tier: RenderTier,
        hw_caps: vsp_compiler::preprocessor::HardwareShaderCaps,
        cache_dir: Option<std::path::PathBuf>,
    ) -> Self {
        let cancel_token = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let progress = std::sync::Arc::new(parking_lot::Mutex::new(PrewarmProgress {
            completed: 0,
            total: 1,
            ready: false,
            cancelled: false,
            last_error: None,
        }));
        let artifacts = std::sync::Arc::new(parking_lot::Mutex::new(None));

        let cancel_bg = std::sync::Arc::clone(&cancel_token);
        let progress_bg = std::sync::Arc::clone(&progress);
        let artifacts_bg = std::sync::Arc::clone(&artifacts);

        let worker = std::thread::Builder::new()
            .name("hyaloclast-prewarm".to_owned())
            .spawn(move || {
                if cancel_bg.load(std::sync::atomic::Ordering::Acquire) {
                    let mut p = progress_bg.lock();
                    p.cancelled = true;
                    return;
                }
                let compiler = VspCompiler::new(cache_dir.as_deref());
                match compiler.compile_pack_dir_with_caps(&pack_dir, tier, hw_caps) {
                    Ok(compiled) => {
                        if cancel_bg.load(std::sync::atomic::Ordering::Acquire) {
                            let mut p = progress_bg.lock();
                            p.cancelled = true;
                            return;
                        }
                        let count = compiled.len().max(1);
                        *artifacts_bg.lock() = Some(compiled);
                        let mut p = progress_bg.lock();
                        p.total = count;
                        p.completed = count;
                        p.ready = true;
                        p.last_error = None;
                    }
                    Err(err) => {
                        let mut p = progress_bg.lock();
                        p.ready = false;
                        p.cancelled = cancel_bg.load(std::sync::atomic::Ordering::Acquire);
                        p.last_error = Some(err.to_string());
                    }
                }
            })
            .ok();

        Self {
            cancel_token,
            progress,
            artifacts,
            worker,
        }
    }

    /// Launches a custom prewarm task on the `"hyaloclast-prewarm"` background thread with
    /// explicit step-progress reporting and cooperative cancellation.
    pub fn start_custom<F>(total_steps: usize, work_fn: F) -> Self
    where
        F: FnOnce(
                &std::sync::Arc<std::sync::atomic::AtomicBool>,
                &dyn Fn(usize, usize),
            ) -> Result<(), String>
            + Send
            + 'static,
    {
        let cancel_token = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let progress = std::sync::Arc::new(parking_lot::Mutex::new(PrewarmProgress {
            completed: 0,
            total: total_steps,
            ready: false,
            cancelled: false,
            last_error: None,
        }));
        let artifacts = std::sync::Arc::new(parking_lot::Mutex::new(None));

        let cancel_bg = std::sync::Arc::clone(&cancel_token);
        let progress_bg = std::sync::Arc::clone(&progress);

        let worker = std::thread::Builder::new()
            .name("hyaloclast-prewarm".to_owned())
            .spawn(move || {
                let report = |completed: usize, total: usize| {
                    let mut p = progress_bg.lock();
                    p.completed = completed;
                    p.total = total;
                };
                match work_fn(&cancel_bg, &report) {
                    Ok(()) => {
                        let cancelled = cancel_bg.load(std::sync::atomic::Ordering::Acquire);
                        let mut p = progress_bg.lock();
                        p.cancelled = cancelled;
                        if !cancelled {
                            p.completed = p.total;
                            p.ready = true;
                        }
                    }
                    Err(err) => {
                        let mut p = progress_bg.lock();
                        p.cancelled = cancel_bg.load(std::sync::atomic::Ordering::Acquire);
                        p.ready = false;
                        p.last_error = Some(err);
                    }
                }
            })
            .ok();

        Self {
            cancel_token,
            progress,
            artifacts,
            worker,
        }
    }

    /// Signals cooperative cancellation to the background prewarm thread.
    pub fn cancel(&self) {
        self.cancel_token
            .store(true, std::sync::atomic::Ordering::Release);
        self.progress.lock().cancelled = true;
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.load(std::sync::atomic::Ordering::Acquire)
    }

    pub fn progress(&self) -> PrewarmProgress {
        self.progress.lock().clone()
    }

    pub fn is_ready(&self) -> bool {
        self.progress.lock().ready
    }

    /// Takes the compiled SPIR-V artifacts once prewarming has completed (`ready == true`).
    pub fn take_compiled_artifacts(&self) -> Option<Vec<CompiledShaderArtifact>> {
        self.artifacts.lock().take()
    }

    /// Waits for the background worker thread to finish.
    pub fn wait(&mut self) {
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for AsyncPipelinePrewarmer {
    fn drop(&mut self) {
        self.cancel();
        self.wait();
    }
}

#[cfg(test)]
mod world_ubo_layout_tests {
    use super::*;

    #[test]
    fn world_ubo_keeps_v1_offsets_and_appends_pack_params_bda() {
        assert_eq!(std::mem::size_of::<WorldUboData>(), 544);
        assert_eq!(std::mem::offset_of!(WorldUboData, cur_view_proj), 0);
        assert_eq!(std::mem::offset_of!(WorldUboData, pack_params), 320);
        assert_eq!(std::mem::offset_of!(WorldUboData, held_light_position), 512);
        assert_eq!(std::mem::offset_of!(WorldUboData, pack_params_bda), 528);
        let ubo = WorldUboData::default();
        assert_eq!(ubo.pack_params_bda, 0);
    }

    #[test]
    fn vsl_buffer_capacities_cover_embedded_pack_params() {
        assert_eq!(VSL_PACK_PARAM_BUFFER_BYTES, 4096);
    }

    #[test]
    fn vsl_world_binding_uses_full_544_byte_world_ubo() {
        // VSL stages bind the world UBO at set 0 / binding 0, the same buffer native stages use.
        // The separate 16-byte BDA carrier was retired: the pack-parameter address now lives at
        // offset 528 of this buffer, so the bound allocation must cover the full WORLD_UBO_LAYOUT.
        assert!(WORLD_UBO_BUFFER_BYTES >= 544);
        assert_eq!(WORLD_UBO_BUFFER_BYTES, std::mem::size_of::<WorldUboData>());
        assert!(WORLD_UBO_BUFFER_BYTES > std::mem::offset_of!(WorldUboData, pack_params_bda) + 8);
    }
}
