// SPDX-License-Identifier: GPL-3.0-only
// Additional permissions under GPLv3 Section 7 apply to optional proprietary vendor SDK
// runtime loading; see LICENSE-EXCEPTION.md in the repository root.

//! Clean-room NVIDIA Real-Time Denoisers (NRD v4.18+) runtime dynamic loader (`libNRD.so` / `NRD.dll`).
//!
//! Dynamically loads `libNRD.so` (which embeds precompiled SPIR-V compute shaders), exposes the
//! exact unmangled C-ABI entry points exported by NRD (`CreateInstance`, `DestroyInstance`,
//! `GetLibraryDesc`, `GetInstanceDesc`, `SetCommonSettings`, `SetDenoiserSettings`,
//! `GetComputeDispatches`, `GetResourceTypeString`, `GetDenoiserString`), and implements a full
//! Vulkan compute dispatcher (`NrdVulkanDispatcher`) that allocates `PermanentPool` and
//! `TransientPool` GPU textures, manages descriptor sets and constant buffers, transitions layouts,
//! and binds all resources per dispatch.

use crate::gpu::{
    buffer::{GpuMemoryAllocator, PooledImageAllocation},
    retire::RetireQueue,
};
use ash::vk;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};
use libloading::Library;
use std::{
    collections::HashMap,
    ffi::{CStr, c_char, c_void},
    fmt,
    path::{Path, PathBuf},
    ptr,
};
use thiserror::Error;

pub type NrdIdentifier = u32;

/// Opaque `nrd::Instance`.
#[repr(C)]
pub struct NrdInstanceHandle {
    _private: [u8; 0],
}

#[derive(Debug, Error)]
pub enum NrdError {
    #[error("failed to load NVIDIA NRD library ({0})")]
    LibraryLoad(String),
    #[error("missing required NRD symbol `{0}`: {1}")]
    MissingSymbol(&'static str, String),
    #[error("NRD call `{call}` failed with {result:?}")]
    ApiFailure {
        call: &'static str,
        result: NrdResult,
    },
    #[error("null NRD instance or descriptor pointer")]
    NullPointer,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NrdResult {
    Success = 0,
    Failure = 1,
    InvalidArgument = 2,
    Unsupported = 3,
    NonUniqueIdentifier = 4,
    MaxNum = 5,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NrdDenoiser {
    ReblurDiffuse = 0,
    ReblurDiffuseOcclusion = 1,
    ReblurDiffuseSh = 2,
    ReblurSpecular = 3,
    ReblurSpecularOcclusion = 4,
    ReblurSpecularSh = 5,
    ReblurDiffuseSpecular = 6,
    ReblurDiffuseSpecularOcclusion = 7,
    ReblurDiffuseSpecularSh = 8,
    ReblurDiffuseDirectionalOcclusion = 9,
    RelaxDiffuse = 10,
    RelaxDiffuseSh = 11,
    RelaxSpecular = 12,
    RelaxSpecularSh = 13,
    RelaxDiffuseSpecular = 14,
    RelaxDiffuseSpecularSh = 15,
    SigmaShadow = 16,
    SigmaShadowTranslucency = 17,
    Reference = 18,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NrdResourceType {
    InMv = 0,
    InNormalRoughness = 1,
    InViewZ = 2,
    InDiffConfidence = 3,
    InSpecConfidence = 4,
    InDisocclusionThresholdMix = 5,
    InDiffRadianceHitdist = 6,
    InSpecRadianceHitdist = 7,
    InDiffHitdist = 8,
    InSpecHitdist = 9,
    InDiffDirectionHitdist = 10,
    InDiffSh0 = 11,
    InDiffSh1 = 12,
    InSpecSh0 = 13,
    InSpecSh1 = 14,
    InPenumbra = 15,
    InTranslucency = 16,
    InSignal = 17,
    OutDiffRadianceHitdist = 18,
    OutSpecRadianceHitdist = 19,
    OutDiffSh0 = 20,
    OutDiffSh1 = 21,
    OutSpecSh0 = 22,
    OutSpecSh1 = 23,
    OutDiffHitdist = 24,
    OutSpecHitdist = 25,
    OutDiffDirectionHitdist = 26,
    OutShadowTranslucency = 27,
    OutSignal = 28,
    OutValidation = 29,
    TransientPool = 30,
    PermanentPool = 31,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NrdDescriptorType {
    Texture = 0,
    StorageTexture = 1,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NrdSampler {
    NearestClamp = 0,
    LinearClamp = 1,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NrdNormalEncoding {
    Rgba8Unorm = 0,
    Rgba8Snorm = 1,
    R10G10B10A2Unorm = 2,
    Rgba16Unorm = 3,
    Rgba16Snorm = 4,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NrdRoughnessEncoding {
    SqLinear = 0,
    Linear = 1,
    SqrtLinear = 2,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NrdCheckerboardMode {
    Off = 0,
    Black = 1,
    White = 2,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NrdAccumulationMode {
    Continue = 0,
    Restart = 1,
    ClearAndRestart = 2,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NrdHitDistanceReconstructionMode {
    Off = 0,
    Area3x3 = 1,
    Area5x5 = 2,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NrdAllocationCallbacks {
    pub allocate: Option<unsafe extern "C" fn(*mut c_void, usize, usize) -> *mut c_void>,
    pub reallocate:
        Option<unsafe extern "C" fn(*mut c_void, *mut c_void, usize, usize) -> *mut c_void>,
    pub free: Option<unsafe extern "C" fn(*mut c_void, *mut c_void)>,
    pub user_arg: *mut c_void,
}

impl Default for NrdAllocationCallbacks {
    fn default() -> Self {
        Self {
            allocate: None,
            reallocate: None,
            free: None,
            user_arg: ptr::null_mut(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NrdSpirvBindingOffsets {
    pub sampler_offset: u32,
    pub texture_offset: u32,
    pub constant_buffer_offset: u32,
    pub storage_texture_and_buffer_offset: u32,
}

/// `nrd::LibraryDesc` (40 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NrdLibraryDesc {
    pub spirv_binding_offsets: NrdSpirvBindingOffsets,
    pub supported_denoisers: *const NrdDenoiser,
    pub supported_denoisers_num: u32,
    pub version_major: u8,
    pub version_minor: u8,
    pub version_build: u8,
    pub normal_encoding: NrdNormalEncoding,
    pub roughness_encoding: NrdRoughnessEncoding,
}

impl fmt::Debug for NrdLibraryDesc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NrdLibraryDesc")
            .field(
                "version",
                &format!(
                    "{}.{}.{}",
                    self.version_major, self.version_minor, self.version_build
                ),
            )
            .field("supported_denoisers_num", &self.supported_denoisers_num)
            .field("normal_encoding", &self.normal_encoding)
            .field("roughness_encoding", &self.roughness_encoding)
            .finish()
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NrdDenoiserDesc {
    pub identifier: NrdIdentifier,
    pub denoiser: NrdDenoiser,
}

/// `nrd::InstanceCreationDesc` (48 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NrdInstanceCreationDesc {
    pub allocation_callbacks: NrdAllocationCallbacks,
    pub denoisers: *const NrdDenoiserDesc,
    pub denoisers_num: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NrdTextureDesc {
    pub format: u32,
    pub downsample_factor: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NrdResourceDesc {
    pub descriptor_type: NrdDescriptorType,
    pub resource_type: NrdResourceType,
    pub index_in_pool: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NrdResourceRangeDesc {
    pub descriptor_type: NrdDescriptorType,
    pub descriptors_num: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NrdComputeShaderDesc {
    pub bytecode: *const c_void,
    pub size: u64,
}

/// `nrd::PipelineDesc` (320 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NrdPipelineDesc {
    pub compute_shader_dxbc: NrdComputeShaderDesc,
    pub compute_shader_dxil: NrdComputeShaderDesc,
    pub compute_shader_spirv: NrdComputeShaderDesc,
    pub resource_ranges: *const NrdResourceRangeDesc,
    pub resource_ranges_num: u32,
    pub has_constant_data: bool,
    pub shader_identifier: [c_char; 256],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NrdDescriptorPoolDesc {
    pub per_set_textures_max_num: u32,
    pub per_set_storage_textures_max_num: u32,
    pub total_textures_num: u32,
    pub total_storage_textures_num: u32,
    pub sets_max_num: u32,
}

/// `nrd::InstanceDesc` (112 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NrdInstanceDesc {
    pub constant_buffer_and_samplers_space_index: u32,
    pub resources_space_index: u32,
    pub constant_buffer_register_index: u32,
    pub samplers_base_register_index: u32,
    pub resources_base_register_index: u32,
    pub constant_buffer_max_data_size: u32,
    pub samplers: *const NrdSampler,
    pub samplers_num: u32,
    pub shader_entry_point: *const c_char,
    pub pipelines: *const NrdPipelineDesc,
    pub pipelines_num: u32,
    pub permanent_pool: *const NrdTextureDesc,
    pub permanent_pool_size: u32,
    pub transient_pool: *const NrdTextureDesc,
    pub transient_pool_size: u32,
    pub descriptor_pool_desc: NrdDescriptorPoolDesc,
}

/// `nrd::DispatchDesc` (56 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NrdDispatchDesc {
    pub name: *const c_char,
    pub identifier: NrdIdentifier,
    pub resources: *const NrdResourceDesc,
    pub resources_num: u32,
    pub constant_buffer_data: *const u8,
    pub constant_buffer_data_size: u32,
    pub constant_buffer_data_matches_previous_dispatch: bool,
    pub pipeline_index: u16,
    pub grid_width: u16,
    pub grid_height: u16,
}

/// `nrd::CommonSettings` (452 bytes, align 4).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NrdCommonSettings {
    pub view_to_clip_matrix: [f32; 16],
    pub view_to_clip_matrix_prev: [f32; 16],
    pub world_to_view_matrix: [f32; 16],
    pub world_to_view_matrix_prev: [f32; 16],
    pub world_prev_to_world_matrix: [f32; 16],
    pub motion_vector_scale: [f32; 3],
    pub motion_vector_bias: [f32; 3],
    pub camera_jitter: [f32; 2],
    pub camera_jitter_prev: [f32; 2],
    pub resource_size: [u16; 2],
    pub resource_size_prev: [u16; 2],
    pub rect_size: [u16; 2],
    pub rect_size_prev: [u16; 2],
    pub view_z_scale: f32,
    pub time_delta_between_frames: f32,
    pub denoising_range: f32,
    pub disocclusion_threshold: f32,
    pub disocclusion_threshold_alternate: f32,
    pub camera_attached_reflection_material_id: f32,
    pub strand_material_id: f32,
    pub history_fix_alternate_pixel_stride_material_id: f32,
    pub strand_thickness: f32,
    pub split_screen: f32,
    pub printf_at: [u16; 2],
    pub debug: f32,
    pub input_rect_origin: [u32; 2],
    pub output_rect_origin: [u32; 2],
    pub frame_index: u32,
    pub accumulation_mode: NrdAccumulationMode,
    pub is_motion_vector_in_world_space: bool,
    pub is_history_confidence_available: bool,
    pub is_disocclusion_threshold_mix_available: bool,
    pub enable_validation: bool,
}

impl Default for NrdCommonSettings {
    fn default() -> Self {
        // Non-degenerate perspective & view defaults so NRD matrix inversion never divides by zero.
        let default_proj: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0002, -1.0, 0.0, 0.0, -0.1, 0.0,
        ];
        let identity: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        Self {
            view_to_clip_matrix: default_proj,
            view_to_clip_matrix_prev: default_proj,
            world_to_view_matrix: identity,
            world_to_view_matrix_prev: identity,
            world_prev_to_world_matrix: identity,
            motion_vector_scale: [1.0, 1.0, 0.0],
            motion_vector_bias: [0.0; 3],
            camera_jitter: [0.0; 2],
            camera_jitter_prev: [0.0; 2],
            resource_size: [0; 2],
            resource_size_prev: [0; 2],
            rect_size: [0; 2],
            rect_size_prev: [0; 2],
            view_z_scale: 1.0,
            time_delta_between_frames: 16.667,
            denoising_range: 500_000.0,
            disocclusion_threshold: 0.01,
            disocclusion_threshold_alternate: 0.05,
            camera_attached_reflection_material_id: 999.0,
            strand_material_id: 999.0,
            history_fix_alternate_pixel_stride_material_id: 999.0,
            strand_thickness: 80e-6,
            split_screen: 0.0,
            printf_at: [9999, 9999],
            debug: 0.0,
            input_rect_origin: [0; 2],
            output_rect_origin: [0; 2],
            frame_index: 0,
            accumulation_mode: NrdAccumulationMode::Continue,
            is_motion_vector_in_world_space: false,
            is_history_confidence_available: false,
            is_disocclusion_threshold_mix_available: false,
            enable_validation: false,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NrdReblurHitDistanceParameters {
    pub a: f32,
    pub b: f32,
    pub c: f32,
}

impl Default for NrdReblurHitDistanceParameters {
    fn default() -> Self {
        Self {
            a: 3.0,
            b: 0.1,
            c: 20.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NrdReblurAntilagSettings {
    pub luminance_sigma_scale: f32,
    pub luminance_sensitivity: f32,
}

impl Default for NrdReblurAntilagSettings {
    fn default() -> Self {
        Self {
            luminance_sigma_scale: 2.0,
            luminance_sensitivity: 3.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NrdReblurResponsiveAccumulationSettings {
    pub roughness_threshold: f32,
    pub min_accumulated_frame_num: u32,
}

impl Default for NrdReblurResponsiveAccumulationSettings {
    fn default() -> Self {
        Self {
            roughness_threshold: 0.0,
            min_accumulated_frame_num: 3,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NrdReblurConvergenceSettings {
    pub s: f32,
    pub b: f32,
    pub p: f32,
}

impl Default for NrdReblurConvergenceSettings {
    fn default() -> Self {
        Self {
            s: 1.0,
            b: 0.2,
            p: 0.8,
        }
    }
}

/// `nrd::ReblurSettings` (120 bytes, align 4).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NrdReblurSettings {
    pub hit_distance_parameters: NrdReblurHitDistanceParameters,
    pub antilag_settings: NrdReblurAntilagSettings,
    pub responsive_accumulation_settings: NrdReblurResponsiveAccumulationSettings,
    pub convergence_settings: NrdReblurConvergenceSettings,
    pub max_accumulated_frame_num: u32,
    pub max_fast_accumulated_frame_num: u32,
    pub max_stabilized_frame_num: u32,
    pub history_fix_frame_num: u32,
    pub history_fix_base_pixel_stride: u32,
    pub history_fix_alternate_pixel_stride: u32,
    pub fast_history_clamping_sigma_scale: f32,
    pub diffuse_prepass_blur_radius: f32,
    pub specular_prepass_blur_radius: f32,
    pub min_hit_distance_weight: f32,
    pub min_blur_radius: f32,
    pub max_blur_radius: f32,
    pub lobe_angle_fraction: f32,
    pub roughness_fraction: f32,
    pub plane_distance_sensitivity: f32,
    pub firefly_suppressor_min_relative_scale: f32,
    pub min_material_for_diffuse: f32,
    pub min_material_for_specular: f32,
    pub checkerboard_mode: NrdCheckerboardMode,
    pub hit_distance_reconstruction_mode: NrdHitDistanceReconstructionMode,
    pub enable_anti_firefly: bool,
    pub use_prepass_only_for_specular_motion_estimation: bool,
    pub return_history_length_instead_of_occlusion: bool,
}

impl Default for NrdReblurSettings {
    fn default() -> Self {
        Self {
            hit_distance_parameters: NrdReblurHitDistanceParameters::default(),
            antilag_settings: NrdReblurAntilagSettings::default(),
            responsive_accumulation_settings: NrdReblurResponsiveAccumulationSettings::default(),
            convergence_settings: NrdReblurConvergenceSettings::default(),
            max_accumulated_frame_num: 30,
            max_fast_accumulated_frame_num: 6,
            max_stabilized_frame_num: 63,
            history_fix_frame_num: 3,
            history_fix_base_pixel_stride: 14,
            history_fix_alternate_pixel_stride: 14,
            fast_history_clamping_sigma_scale: 2.0,
            diffuse_prepass_blur_radius: 30.0,
            specular_prepass_blur_radius: 50.0,
            min_hit_distance_weight: 0.1,
            min_blur_radius: 1.0,
            max_blur_radius: 30.0,
            lobe_angle_fraction: 0.15,
            roughness_fraction: 0.15,
            plane_distance_sensitivity: 0.02,
            firefly_suppressor_min_relative_scale: 2.0,
            min_material_for_diffuse: 4.0,
            min_material_for_specular: 4.0,
            checkerboard_mode: NrdCheckerboardMode::Off,
            hit_distance_reconstruction_mode: NrdHitDistanceReconstructionMode::Off,
            enable_anti_firefly: true,
            use_prepass_only_for_specular_motion_estimation: false,
            return_history_length_instead_of_occlusion: false,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NrdRelaxAntilagSettings {
    pub acceleration_amount: f32,
    pub spatial_sigma_scale: f32,
    pub temporal_sigma_scale: f32,
    pub reset_amount: f32,
}

impl Default for NrdRelaxAntilagSettings {
    fn default() -> Self {
        Self {
            acceleration_amount: 0.3,
            spatial_sigma_scale: 4.5,
            temporal_sigma_scale: 0.5,
            reset_amount: 0.5,
        }
    }
}

/// `nrd::RelaxSettings` (148 bytes, align 4).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NrdRelaxSettings {
    pub antilag_settings: NrdRelaxAntilagSettings,
    pub diffuse_max_accumulated_frame_num: u32,
    pub specular_max_accumulated_frame_num: u32,
    pub diffuse_max_fast_accumulated_frame_num: u32,
    pub specular_max_fast_accumulated_frame_num: u32,
    pub history_fix_frame_num: u32,
    pub history_fix_base_pixel_stride: u32,
    pub history_fix_alternate_pixel_stride: u32,
    pub history_fix_edge_stopping_normal_power: f32,
    pub fast_history_clamping_sigma_scale: f32,
    pub diffuse_prepass_blur_radius: f32,
    pub specular_prepass_blur_radius: f32,
    pub min_hit_distance_weight: f32,
    pub spatial_variance_estimation_history_threshold: u32,
    pub diffuse_phi_luminance: f32,
    pub specular_phi_luminance: f32,
    pub lobe_angle_fraction: f32,
    pub roughness_fraction: f32,
    pub specular_variance_boost: f32,
    pub specular_lobe_angle_slack: f32,
    pub atrous_iteration_num: u32,
    pub diffuse_min_luminance_weight: f32,
    pub specular_min_luminance_weight: f32,
    pub depth_threshold: f32,
    pub confidence_driven_relaxation_multiplier: f32,
    pub confidence_driven_luminance_edge_stopping_relaxation: f32,
    pub confidence_driven_normal_edge_stopping_relaxation: f32,
    pub luminance_edge_stopping_relaxation: f32,
    pub normal_edge_stopping_relaxation: f32,
    pub roughness_edge_stopping_relaxation: f32,
    pub checkerboard_mode: NrdCheckerboardMode,
    pub hit_distance_reconstruction_mode: NrdHitDistanceReconstructionMode,
    pub min_material_for_diffuse: f32,
    pub min_material_for_specular: f32,
    pub enable_anti_firefly: bool,
    pub enable_roughness_edge_stopping: bool,
}

impl Default for NrdRelaxSettings {
    fn default() -> Self {
        Self {
            antilag_settings: NrdRelaxAntilagSettings::default(),
            diffuse_max_accumulated_frame_num: 30,
            specular_max_accumulated_frame_num: 30,
            diffuse_max_fast_accumulated_frame_num: 6,
            specular_max_fast_accumulated_frame_num: 6,
            history_fix_frame_num: 3,
            history_fix_base_pixel_stride: 14,
            history_fix_alternate_pixel_stride: 14,
            history_fix_edge_stopping_normal_power: 8.0,
            fast_history_clamping_sigma_scale: 2.0,
            diffuse_prepass_blur_radius: 30.0,
            specular_prepass_blur_radius: 50.0,
            min_hit_distance_weight: 0.1,
            spatial_variance_estimation_history_threshold: 3,
            diffuse_phi_luminance: 2.0,
            specular_phi_luminance: 1.0,
            lobe_angle_fraction: 0.5,
            roughness_fraction: 0.15,
            specular_variance_boost: 0.0,
            specular_lobe_angle_slack: 0.15,
            atrous_iteration_num: 5,
            diffuse_min_luminance_weight: 0.0,
            specular_min_luminance_weight: 0.0,
            depth_threshold: 0.003,
            confidence_driven_relaxation_multiplier: 0.0,
            confidence_driven_luminance_edge_stopping_relaxation: 0.0,
            confidence_driven_normal_edge_stopping_relaxation: 0.0,
            luminance_edge_stopping_relaxation: 0.5,
            normal_edge_stopping_relaxation: 0.3,
            roughness_edge_stopping_relaxation: 1.0,
            checkerboard_mode: NrdCheckerboardMode::Off,
            hit_distance_reconstruction_mode: NrdHitDistanceReconstructionMode::Off,
            min_material_for_diffuse: 4.0,
            min_material_for_specular: 4.0,
            enable_anti_firefly: false,
            enable_roughness_edge_stopping: true,
        }
    }
}

/// `nrd::SigmaSettings` (24 bytes, align 4).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NrdSigmaSettings {
    pub light_direction: [f32; 3],
    pub plane_distance_sensitivity: f32,
    pub max_stabilized_frame_num: u32,
    pub checkerboard_mode: NrdCheckerboardMode,
}

impl Default for NrdSigmaSettings {
    fn default() -> Self {
        Self {
            light_direction: [0.0; 3],
            plane_distance_sensitivity: 0.02,
            max_stabilized_frame_num: 5,
            checkerboard_mode: NrdCheckerboardMode::Off,
        }
    }
}

/// `nrd::ReferenceSettings` (4 bytes, align 4).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NrdReferenceSettings {
    pub max_accumulated_frame_num: u32,
}

impl Default for NrdReferenceSettings {
    fn default() -> Self {
        Self {
            max_accumulated_frame_num: 120,
        }
    }
}

// Compile-time ABI size assertions matching NRD v4.18 headers
const _: () = assert!(std::mem::size_of::<NrdLibraryDesc>() == 40);
const _: () = assert!(std::mem::size_of::<NrdInstanceCreationDesc>() == 48);
const _: () = assert!(std::mem::size_of::<NrdInstanceDesc>() == 112);
const _: () = assert!(std::mem::size_of::<NrdPipelineDesc>() == 320);
const _: () = assert!(std::mem::size_of::<NrdDispatchDesc>() == 56);
const _: () = assert!(std::mem::size_of::<NrdCommonSettings>() == 452);
const _: () = assert!(std::mem::size_of::<NrdReblurSettings>() == 120);
const _: () = assert!(std::mem::size_of::<NrdRelaxSettings>() == 148);
const _: () = assert!(std::mem::size_of::<NrdSigmaSettings>() == 24);
const _: () = assert!(std::mem::size_of::<NrdReferenceSettings>() == 4);

// Unmangled `extern "C"` entry points exported by `libNRD.so` / `NRD.dll`
pub type PfnNrdCreateInstance = unsafe extern "C" fn(
    instance_creation_desc: *const NrdInstanceCreationDesc,
    instance: *mut *mut NrdInstanceHandle,
) -> NrdResult;

pub type PfnNrdDestroyInstance = unsafe extern "C" fn(instance: *mut NrdInstanceHandle);

pub type PfnNrdGetLibraryDesc = unsafe extern "C" fn() -> *const NrdLibraryDesc;

pub type PfnNrdGetInstanceDesc =
    unsafe extern "C" fn(instance: *const NrdInstanceHandle) -> *const NrdInstanceDesc;

pub type PfnNrdSetCommonSettings = unsafe extern "C" fn(
    instance: *mut NrdInstanceHandle,
    common_settings: *const NrdCommonSettings,
) -> NrdResult;

pub type PfnNrdSetDenoiserSettings = unsafe extern "C" fn(
    instance: *mut NrdInstanceHandle,
    identifier: NrdIdentifier,
    denoiser_settings: *const c_void,
) -> NrdResult;

pub type PfnNrdGetComputeDispatches = unsafe extern "C" fn(
    instance: *mut NrdInstanceHandle,
    identifiers: *const NrdIdentifier,
    identifiers_num: u32,
    dispatch_descs: *mut *const NrdDispatchDesc,
    dispatch_descs_num: *mut u32,
) -> NrdResult;

pub type PfnNrdGetResourceTypeString =
    unsafe extern "C" fn(resource_type: NrdResourceType) -> *const c_char;

pub type PfnNrdGetDenoiserString = unsafe extern "C" fn(denoiser: NrdDenoiser) -> *const c_char;

/// Dynamically loaded NVIDIA NRD runtime library.
pub struct NrdLibrary {
    _library: Library,
    pub loaded_path: PathBuf,
    pub create_instance: PfnNrdCreateInstance,
    pub destroy_instance: PfnNrdDestroyInstance,
    pub get_library_desc: PfnNrdGetLibraryDesc,
    pub get_instance_desc: PfnNrdGetInstanceDesc,
    pub set_common_settings: PfnNrdSetCommonSettings,
    pub set_denoiser_settings: PfnNrdSetDenoiserSettings,
    pub get_compute_dispatches: PfnNrdGetComputeDispatches,
    pub get_resource_type_string: Option<PfnNrdGetResourceTypeString>,
    pub get_denoiser_string: Option<PfnNrdGetDenoiserString>,
}

impl NrdLibrary {
    pub fn candidate_names() -> &'static [&'static str] {
        #[cfg(target_os = "windows")]
        {
            &["NRD.dll", "libNRD.dll"]
        }
        #[cfg(target_os = "macos")]
        {
            &[]
        }
        #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
        {
            &["libNRD.so", "libNRD.so.4"]
        }
    }

    pub fn load(search_dirs: &[&Path]) -> Result<Self, NrdError> {
        #[cfg(target_os = "macos")]
        {
            let _ = search_dirs;
            return Err(NrdError::LibraryLoad(
                "NVIDIA NRD is not supported on macOS".to_string(),
            ));
        }
        let mut candidates: Vec<PathBuf> = Vec::new();
        for dir in search_dirs {
            for name in Self::candidate_names() {
                let p = dir.join(name);
                if p.exists() {
                    candidates.push(p);
                }
            }
        }
        for name in Self::candidate_names() {
            candidates.push(PathBuf::from(name));
        }

        let mut last_err = String::from("no candidate paths attempted");
        for path in candidates {
            match unsafe { Library::new(&path) } {
                Ok(lib) => return Self::from_library(lib, path),
                Err(e) => last_err = format!("{}: {e}", path.display()),
            }
        }

        Err(NrdError::LibraryLoad(last_err))
    }

    fn from_library(lib: Library, loaded_path: PathBuf) -> Result<Self, NrdError> {
        unsafe {
            let create_instance = *lib
                .get::<PfnNrdCreateInstance>(b"CreateInstance\0")
                .map_err(|e| NrdError::MissingSymbol("CreateInstance", e.to_string()))?;
            let destroy_instance = *lib
                .get::<PfnNrdDestroyInstance>(b"DestroyInstance\0")
                .map_err(|e| NrdError::MissingSymbol("DestroyInstance", e.to_string()))?;
            let get_library_desc = *lib
                .get::<PfnNrdGetLibraryDesc>(b"GetLibraryDesc\0")
                .map_err(|e| NrdError::MissingSymbol("GetLibraryDesc", e.to_string()))?;
            let get_instance_desc = *lib
                .get::<PfnNrdGetInstanceDesc>(b"GetInstanceDesc\0")
                .map_err(|e| NrdError::MissingSymbol("GetInstanceDesc", e.to_string()))?;
            let set_common_settings =
                *lib.get::<PfnNrdSetCommonSettings>(b"SetCommonSettings\0")
                    .map_err(|e| NrdError::MissingSymbol("SetCommonSettings", e.to_string()))?;
            let set_denoiser_settings = *lib
                .get::<PfnNrdSetDenoiserSettings>(b"SetDenoiserSettings\0")
                .map_err(|e| NrdError::MissingSymbol("SetDenoiserSettings", e.to_string()))?;
            let get_compute_dispatches = *lib
                .get::<PfnNrdGetComputeDispatches>(b"GetComputeDispatches\0")
                .map_err(|e| NrdError::MissingSymbol("GetComputeDispatches", e.to_string()))?;
            let get_resource_type_string = lib
                .get::<PfnNrdGetResourceTypeString>(b"GetResourceTypeString\0")
                .ok()
                .map(|s| *s);
            let get_denoiser_string = lib
                .get::<PfnNrdGetDenoiserString>(b"GetDenoiserString\0")
                .ok()
                .map(|s| *s);

            Ok(Self {
                _library: lib,
                loaded_path,
                create_instance,
                destroy_instance,
                get_library_desc,
                get_instance_desc,
                set_common_settings,
                set_denoiser_settings,
                get_compute_dispatches,
                get_resource_type_string,
                get_denoiser_string,
            })
        }
    }

    pub fn library_desc(&self) -> Option<&NrdLibraryDesc> {
        let ptr = unsafe { (self.get_library_desc)() };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { &*ptr })
        }
    }

    pub fn denoiser_name(&self, denoiser: NrdDenoiser) -> Option<&str> {
        let func = self.get_denoiser_string?;
        let ptr = unsafe { func(denoiser) };
        if ptr.is_null() {
            None
        } else {
            unsafe { CStr::from_ptr(ptr) }.to_str().ok()
        }
    }
}

/// Owned NRD denoiser instance (`nrd::CreateInstance` / `nrd::DestroyInstance`).
pub struct NrdInstance {
    lib: std::sync::Arc<NrdLibrary>,
    handle: *mut NrdInstanceHandle,
    pending_history_reset: bool,
}

unsafe impl Send for NrdInstance {}

impl NrdInstance {
    pub fn create(
        lib: std::sync::Arc<NrdLibrary>,
        denoisers: &[NrdDenoiserDesc],
    ) -> Result<Self, NrdError> {
        let desc = NrdInstanceCreationDesc {
            allocation_callbacks: NrdAllocationCallbacks::default(),
            denoisers: denoisers.as_ptr(),
            denoisers_num: denoisers.len() as u32,
        };
        let mut handle: *mut NrdInstanceHandle = ptr::null_mut();
        let res = unsafe { (lib.create_instance)(&desc, &mut handle) };
        if res != NrdResult::Success {
            return Err(NrdError::ApiFailure {
                call: "nrd::CreateInstance",
                result: res,
            });
        }
        if handle.is_null() {
            return Err(NrdError::NullPointer);
        }
        Ok(Self {
            lib,
            handle,
            pending_history_reset: true,
        })
    }

    /// Marks NRD temporal accumulation for `ClearAndRestart` on the next dispatch
    /// (for example on teleport, dimension change, camera cut, or denoiser switch).
    pub fn request_history_reset(&mut self) {
        self.pending_history_reset = true;
    }

    /// Consumes and returns whether a temporal history reset was requested.
    pub fn take_pending_history_reset(&mut self) -> bool {
        std::mem::take(&mut self.pending_history_reset)
    }

    pub fn instance_desc(&self) -> Result<&NrdInstanceDesc, NrdError> {
        let ptr = unsafe { (self.lib.get_instance_desc)(self.handle) };
        if ptr.is_null() {
            Err(NrdError::NullPointer)
        } else {
            Ok(unsafe { &*ptr })
        }
    }

    pub fn pipelines(&self) -> Result<&[NrdPipelineDesc], NrdError> {
        let desc = self.instance_desc()?;
        if desc.pipelines.is_null() || desc.pipelines_num == 0 {
            return Ok(&[]);
        }
        Ok(unsafe { std::slice::from_raw_parts(desc.pipelines, desc.pipelines_num as usize) })
    }

    pub fn set_common_settings(&mut self, settings: &NrdCommonSettings) -> Result<(), NrdError> {
        let res = unsafe { (self.lib.set_common_settings)(self.handle, settings) };
        if res != NrdResult::Success {
            return Err(NrdError::ApiFailure {
                call: "nrd::SetCommonSettings",
                result: res,
            });
        }
        Ok(())
    }

    pub fn set_reblur_settings(
        &mut self,
        identifier: NrdIdentifier,
        settings: &NrdReblurSettings,
    ) -> Result<(), NrdError> {
        let res = unsafe {
            (self.lib.set_denoiser_settings)(
                self.handle,
                identifier,
                (settings as *const NrdReblurSettings).cast(),
            )
        };
        if res != NrdResult::Success {
            return Err(NrdError::ApiFailure {
                call: "nrd::SetDenoiserSettings(Reblur)",
                result: res,
            });
        }
        Ok(())
    }

    pub fn set_relax_settings(
        &mut self,
        identifier: NrdIdentifier,
        settings: &NrdRelaxSettings,
    ) -> Result<(), NrdError> {
        let res = unsafe {
            (self.lib.set_denoiser_settings)(
                self.handle,
                identifier,
                (settings as *const NrdRelaxSettings).cast(),
            )
        };
        if res != NrdResult::Success {
            return Err(NrdError::ApiFailure {
                call: "nrd::SetDenoiserSettings(Relax)",
                result: res,
            });
        }
        Ok(())
    }

    pub fn set_sigma_settings(
        &mut self,
        identifier: NrdIdentifier,
        settings: &NrdSigmaSettings,
    ) -> Result<(), NrdError> {
        let res = unsafe {
            (self.lib.set_denoiser_settings)(
                self.handle,
                identifier,
                (settings as *const NrdSigmaSettings).cast(),
            )
        };
        if res != NrdResult::Success {
            return Err(NrdError::ApiFailure {
                call: "nrd::SetDenoiserSettings(Sigma)",
                result: res,
            });
        }
        Ok(())
    }

    pub fn get_compute_dispatches(
        &mut self,
        identifiers: &[NrdIdentifier],
    ) -> Result<&[NrdDispatchDesc], NrdError> {
        let mut descs_ptr: *const NrdDispatchDesc = ptr::null();
        let mut descs_num = 0u32;
        let res = unsafe {
            (self.lib.get_compute_dispatches)(
                self.handle,
                identifiers.as_ptr(),
                identifiers.len() as u32,
                &mut descs_ptr,
                &mut descs_num,
            )
        };
        if res != NrdResult::Success {
            return Err(NrdError::ApiFailure {
                call: "nrd::GetComputeDispatches",
                result: res,
            });
        }
        if descs_ptr.is_null() || descs_num == 0 {
            return Ok(&[]);
        }
        Ok(unsafe { std::slice::from_raw_parts(descs_ptr, descs_num as usize) })
    }
}

impl Drop for NrdInstance {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                (self.lib.destroy_instance)(self.handle);
            }
            self.handle = ptr::null_mut();
        }
    }
}

/// Maps `nrd::Format` ordinal to `vk::Format`.
pub const fn nrd_format_to_vk(nrd_format: u32) -> vk::Format {
    match nrd_format {
        0 => vk::Format::R8_UNORM,
        1 => vk::Format::R8_SNORM,
        2 => vk::Format::R8_UINT,
        3 => vk::Format::R8_SINT,
        4 => vk::Format::R8G8_UNORM,
        5 => vk::Format::R8G8_SNORM,
        6 => vk::Format::R8G8_UINT,
        7 => vk::Format::R8G8_SINT,
        8 => vk::Format::R8G8B8A8_UNORM,
        9 => vk::Format::R8G8B8A8_SNORM,
        10 => vk::Format::R8G8B8A8_UINT,
        11 => vk::Format::R8G8B8A8_SINT,
        12 => vk::Format::R8G8B8A8_SRGB,
        13 => vk::Format::R16_UNORM,
        14 => vk::Format::R16_SNORM,
        15 => vk::Format::R16_UINT,
        16 => vk::Format::R16_SINT,
        17 => vk::Format::R16_SFLOAT,
        18 => vk::Format::R16G16_UNORM,
        19 => vk::Format::R16G16_SNORM,
        20 => vk::Format::R16G16_UINT,
        21 => vk::Format::R16G16_SINT,
        22 => vk::Format::R16G16_SFLOAT,
        23 => vk::Format::R16G16B16A16_UNORM,
        24 => vk::Format::R16G16B16A16_SNORM,
        25 => vk::Format::R16G16B16A16_UINT,
        26 => vk::Format::R16G16B16A16_SINT,
        27 => vk::Format::R16G16B16A16_SFLOAT,
        28 => vk::Format::R32_UINT,
        29 => vk::Format::R32_SINT,
        30 => vk::Format::R32_SFLOAT,
        31 => vk::Format::R32G32_UINT,
        32 => vk::Format::R32G32_SINT,
        33 => vk::Format::R32G32_SFLOAT,
        34 => vk::Format::R32G32B32_UINT,
        35 => vk::Format::R32G32B32_SINT,
        36 => vk::Format::R32G32B32_SFLOAT,
        37 => vk::Format::R32G32B32A32_UINT,
        38 => vk::Format::R32G32B32A32_SINT,
        39 => vk::Format::R32G32B32A32_SFLOAT,
        40 => vk::Format::A2B10G10R10_UNORM_PACK32,
        41 => vk::Format::A2B10G10R10_UINT_PACK32,
        42 => vk::Format::B10G11R11_UFLOAT_PACK32,
        _ => vk::Format::R16G16B16A16_SFLOAT,
    }
}

/// Owned GPU texture allocated by `NrdVulkanDispatcher` for `PermanentPool` or `TransientPool`.
pub struct NrdPoolTexture {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub pooled: Option<PooledImageAllocation>,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub layout: vk::ImageLayout,
}

/// Caller-supplied user input/output textures bound into NRD dispatches.
#[derive(Debug, Clone, Copy, Default)]
pub struct NrdUserTextureSet {
    pub in_mv: (vk::Image, vk::ImageView),
    pub in_normal_roughness: (vk::Image, vk::ImageView),
    pub in_viewz: (vk::Image, vk::ImageView),
    pub in_diff_radiance_hitdist: (vk::Image, vk::ImageView),
    pub in_spec_radiance_hitdist: (vk::Image, vk::ImageView),
    pub out_diff_radiance_hitdist: (vk::Image, vk::ImageView),
    pub out_spec_radiance_hitdist: (vk::Image, vk::ImageView),
    pub in_viewz_format: vk::Format,
}

pub fn aspect_mask_for_format(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM
        | vk::Format::X8_D24_UNORM_PACK32
        | vk::Format::D32_SFLOAT
        | vk::Format::D16_UNORM_S8_UINT
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D32_SFLOAT_S8_UINT => vk::ImageAspectFlags::DEPTH,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

/// Compiles NRD's embedded SPIR-V compute shaders (`NrdInstanceDesc::pipelines`) into real Vulkan
/// `VkShaderModule`, `VkDescriptorSetLayout`, `VkPipelineLayout`, and `VkPipeline` objects,
/// allocates `PermanentPool` and `TransientPool` GPU images, manages constant buffers and
/// descriptor sets, and records compute dispatches returned by `nrd::GetComputeDispatches`.
pub struct NrdVulkanDispatcher {
    device: ash::Device,
    allocator: Option<GpuMemoryAllocator>,
    pub offsets: NrdSpirvBindingOffsets,
    pub cb_space_idx: u32,
    pub res_space_idx: u32,
    pub cb_reg_idx: u32,
    pub samplers_base_reg_idx: u32,
    pub res_base_reg_idx: u32,
    pub cb_max_size: u32,
    pub permanent_descs: Vec<NrdTextureDesc>,
    pub transient_descs: Vec<NrdTextureDesc>,
    pub shader_modules: Vec<vk::ShaderModule>,
    pub set_layouts: Vec<vk::DescriptorSetLayout>,
    pub pipeline_res_set_layouts: Vec<vk::DescriptorSetLayout>,
    pub pipeline_layouts: Vec<vk::PipelineLayout>,
    pub pipelines: Vec<vk::Pipeline>,
    pub samplers: [vk::Sampler; 2],
    pub shared_set_layout: vk::DescriptorSetLayout,
    pub frame_descriptor_pools: [vk::DescriptorPool; 3],
    pub cbuffer: vk::Buffer,
    pub cbuffer_memory: vk::DeviceMemory,
    cbuffer_allocation: Option<Allocation>,
    pub cbuffer_mapped: *mut u8,
    pub cbuffer_slot_stride: usize,
    pub cbuffer_total_slots: usize,
    pub permanent_pool: Vec<NrdPoolTexture>,
    pub permanent_layouts: Vec<vk::ImageLayout>,
    pub transient_pool: Vec<NrdPoolTexture>,
    pub transient_layouts: Vec<vk::ImageLayout>,
    pub frame_counter: u64,
    pub fallback_tex: Option<NrdPoolTexture>,
    pub fallback_layout: vk::ImageLayout,
    pub pool_extent: (u32, u32),
}

unsafe impl Send for NrdVulkanDispatcher {}

impl NrdVulkanDispatcher {
    /// Builds Vulkan compute pipelines, samplers, and descriptor pools from `inst_desc.pipelines`
    /// using the SPIR-V binding offsets advertised in `lib_desc`.
    ///
    /// # Safety
    /// `device` must be a valid Vulkan device handle and `inst_desc` must come from a live
    /// `NrdInstance`.
    pub unsafe fn create(
        device: &ash::Device,
        inst_desc: &NrdInstanceDesc,
        lib_desc: Option<&NrdLibraryDesc>,
    ) -> Result<Self, String> {
        let offsets = lib_desc
            .map(|d| d.spirv_binding_offsets)
            .unwrap_or(NrdSpirvBindingOffsets {
                sampler_offset: 0,
                texture_offset: 64,
                constant_buffer_offset: 128,
                storage_texture_and_buffer_offset: 192,
            });
        let entry_cstr = if !inst_desc.shader_entry_point.is_null() {
            unsafe { CStr::from_ptr(inst_desc.shader_entry_point) }
        } else {
            c"main"
        };
        let pipe_descs = if inst_desc.pipelines.is_null() || inst_desc.pipelines_num == 0 {
            &[][..]
        } else {
            unsafe {
                std::slice::from_raw_parts(inst_desc.pipelines, inst_desc.pipelines_num as usize)
            }
        };
        let permanent_descs =
            if inst_desc.permanent_pool.is_null() || inst_desc.permanent_pool_size == 0 {
                Vec::new()
            } else {
                unsafe {
                    std::slice::from_raw_parts(
                        inst_desc.permanent_pool,
                        inst_desc.permanent_pool_size as usize,
                    )
                    .to_vec()
                }
            };
        let transient_descs =
            if inst_desc.transient_pool.is_null() || inst_desc.transient_pool_size == 0 {
                Vec::new()
            } else {
                unsafe {
                    std::slice::from_raw_parts(
                        inst_desc.transient_pool,
                        inst_desc.transient_pool_size as usize,
                    )
                    .to_vec()
                }
            };

        let nearest_sampler = unsafe {
            device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::NEAREST)
                        .min_filter(vk::Filter::NEAREST)
                        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                    None,
                )
                .map_err(|e| format!("failed to create NRD nearest sampler: {e}"))?
        };
        let linear_sampler = unsafe {
            device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                    None,
                )
                .map_err(|e| format!("failed to create NRD linear sampler: {e}"))?
        };

        let mut shader_modules = Vec::with_capacity(pipe_descs.len());
        let mut set_layouts = Vec::new();
        let mut pipeline_res_set_layouts = Vec::with_capacity(pipe_descs.len());
        let mut pipeline_layouts = Vec::with_capacity(pipe_descs.len());
        let mut pipelines = Vec::with_capacity(pipe_descs.len());

        // Set for constant buffer + samplers (space 0 / constant_buffer_and_samplers_space_index).
        let mut shared_bindings = Vec::new();
        shared_bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(offsets.constant_buffer_offset + inst_desc.constant_buffer_register_index)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        );
        for s_idx in 0..inst_desc.samplers_num {
            shared_bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(
                        offsets.sampler_offset + inst_desc.samplers_base_register_index + s_idx,
                    )
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            );
        }
        let shared_set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&shared_bindings),
                    None,
                )
                .map_err(|e| format!("failed to create NRD shared set layout: {e}"))?
        };
        set_layouts.push(shared_set_layout);

        for p_desc in pipe_descs {
            let spirv = &p_desc.compute_shader_spirv;
            if spirv.bytecode.is_null() || spirv.size < 4 || (spirv.size % 4) != 0 {
                shader_modules.push(vk::ShaderModule::null());
                pipeline_res_set_layouts.push(vk::DescriptorSetLayout::null());
                pipeline_layouts.push(vk::PipelineLayout::null());
                pipelines.push(vk::Pipeline::null());
                continue;
            }
            let word_count = (spirv.size / 4) as usize;
            let mut words = Vec::with_capacity(word_count);
            unsafe {
                ptr::copy_nonoverlapping(
                    spirv.bytecode as *const u32,
                    words.as_mut_ptr(),
                    word_count,
                );
                words.set_len(word_count);
            }
            let mod_info = vk::ShaderModuleCreateInfo::default().code(&words);
            let module = match unsafe { device.create_shader_module(&mod_info, None) } {
                Ok(m) => m,
                Err(_) => {
                    shader_modules.push(vk::ShaderModule::null());
                    pipeline_res_set_layouts.push(vk::DescriptorSetLayout::null());
                    pipeline_layouts.push(vk::PipelineLayout::null());
                    pipelines.push(vk::Pipeline::null());
                    continue;
                }
            };
            shader_modules.push(module);

            let ranges = if p_desc.resource_ranges.is_null() || p_desc.resource_ranges_num == 0 {
                &[][..]
            } else {
                unsafe {
                    std::slice::from_raw_parts(
                        p_desc.resource_ranges,
                        p_desc.resource_ranges_num as usize,
                    )
                }
            };

            let mut res_bindings = if inst_desc.resources_space_index
                == inst_desc.constant_buffer_and_samplers_space_index
            {
                shared_bindings.clone()
            } else {
                Vec::new()
            };
            let mut tex_reg = inst_desc.resources_base_register_index;
            let mut uav_reg = inst_desc.resources_base_register_index;
            for range in ranges {
                for _ in 0..range.descriptors_num {
                    match range.descriptor_type {
                        NrdDescriptorType::Texture => {
                            res_bindings.push(
                                vk::DescriptorSetLayoutBinding::default()
                                    .binding(offsets.texture_offset + tex_reg)
                                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                    .descriptor_count(1)
                                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                            );
                            tex_reg += 1;
                        }
                        NrdDescriptorType::StorageTexture => {
                            res_bindings.push(
                                vk::DescriptorSetLayoutBinding::default()
                                    .binding(offsets.storage_texture_and_buffer_offset + uav_reg)
                                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                                    .descriptor_count(1)
                                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                            );
                            uav_reg += 1;
                        }
                    }
                }
            }
            let res_set_layout = match unsafe {
                device.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&res_bindings),
                    None,
                )
            } {
                Ok(l) => {
                    set_layouts.push(l);
                    pipeline_res_set_layouts.push(l);
                    l
                }
                Err(_) => {
                    pipeline_res_set_layouts.push(vk::DescriptorSetLayout::null());
                    pipeline_layouts.push(vk::PipelineLayout::null());
                    pipelines.push(vk::Pipeline::null());
                    continue;
                }
            };

            let max_space = inst_desc
                .constant_buffer_and_samplers_space_index
                .max(inst_desc.resources_space_index) as usize;
            let mut pipe_set_layouts = vec![shared_set_layout; max_space + 1];
            pipe_set_layouts[inst_desc.constant_buffer_and_samplers_space_index as usize] =
                shared_set_layout;
            pipe_set_layouts[inst_desc.resources_space_index as usize] = res_set_layout;

            let layout = match unsafe {
                device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&pipe_set_layouts),
                    None,
                )
            } {
                Ok(pl) => pl,
                Err(_) => {
                    pipeline_layouts.push(vk::PipelineLayout::null());
                    pipelines.push(vk::Pipeline::null());
                    continue;
                }
            };
            pipeline_layouts.push(layout);

            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(entry_cstr);
            let comp_info = [vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)];
            let pipeline = unsafe {
                device
                    .create_compute_pipelines(vk::PipelineCache::null(), &comp_info, None)
                    .ok()
                    .and_then(|mut v| v.pop())
                    .unwrap_or(vk::Pipeline::null())
            };
            pipelines.push(pipeline);
        }

        // Create per-frame-in-flight resettable descriptor pools so every dispatch in a frame
        // receives its own unique descriptor sets without stomping earlier or in-flight dispatches.
        let max_sets = (pipe_descs.len() as u32 * 4 + 32).max(128);
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: max_sets * 2,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: max_sets * 4,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: max_sets * 16,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: max_sets * 16,
            },
        ];
        let mut frame_descriptor_pools = [vk::DescriptorPool::null(); 3];
        for slot in &mut frame_descriptor_pools {
            *slot = unsafe {
                device
                    .create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .max_sets(max_sets)
                            .pool_sizes(&pool_sizes),
                        None,
                    )
                    .unwrap_or(vk::DescriptorPool::null())
            };
        }

        let cb_slot_stride =
            ((inst_desc.constant_buffer_max_data_size.max(512) as usize) + 255) & !255;
        let cbuffer_total_slots = 64 * 3;

        Ok(Self {
            device: device.clone(),
            allocator: None,
            offsets,
            cb_space_idx: inst_desc.constant_buffer_and_samplers_space_index,
            res_space_idx: inst_desc.resources_space_index,
            cb_reg_idx: inst_desc.constant_buffer_register_index,
            samplers_base_reg_idx: inst_desc.samplers_base_register_index,
            res_base_reg_idx: inst_desc.resources_base_register_index,
            cb_max_size: inst_desc.constant_buffer_max_data_size.max(512),
            permanent_descs,
            transient_descs,
            shader_modules,
            set_layouts,
            pipeline_res_set_layouts,
            pipeline_layouts,
            pipelines,
            samplers: [nearest_sampler, linear_sampler],
            shared_set_layout,
            frame_descriptor_pools,
            cbuffer: vk::Buffer::null(),
            cbuffer_memory: vk::DeviceMemory::null(),
            cbuffer_allocation: None,
            cbuffer_mapped: ptr::null_mut(),
            cbuffer_slot_stride: cb_slot_stride,
            cbuffer_total_slots,
            permanent_pool: Vec::new(),
            permanent_layouts: Vec::new(),
            transient_pool: Vec::new(),
            transient_layouts: Vec::new(),
            frame_counter: 0,
            fallback_tex: None,
            fallback_layout: vk::ImageLayout::UNDEFINED,
            pool_extent: (0, 0),
        })
    }

    /// Attaches a shared [`GpuMemoryAllocator`] so NRD `PermanentPool`, `TransientPool`,
    /// fallback, and constant-buffer allocations are pooled through `gpu-allocator` instead of
    /// issuing standalone `vkAllocateMemory` calls.
    pub fn with_allocator(mut self, allocator: Option<GpuMemoryAllocator>) -> Self {
        self.allocator = allocator;
        self
    }

    /// Updates the optional [`GpuMemoryAllocator`] used for subsequent pool allocations.
    pub fn set_allocator(&mut self, allocator: Option<GpuMemoryAllocator>) {
        self.allocator = allocator;
    }

    /// Same as [`Self::create`], attaching an optional [`GpuMemoryAllocator`] up front.
    ///
    /// # Safety
    /// Same safety requirements as [`Self::create`].
    pub unsafe fn create_with_allocator(
        device: &ash::Device,
        inst_desc: &NrdInstanceDesc,
        lib_desc: Option<&NrdLibraryDesc>,
        allocator: Option<GpuMemoryAllocator>,
    ) -> Result<Self, String> {
        let disp = unsafe { Self::create(device, inst_desc, lib_desc)? };
        Ok(disp.with_allocator(allocator))
    }

    unsafe fn allocate_pool_texture(
        device: &ash::Device,
        allocator: Option<&GpuMemoryAllocator>,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        format: vk::Format,
        width: u32,
        height: u32,
        name: &str,
    ) -> Option<NrdPoolTexture> {
        let w = width.max(1);
        let h = height.max(1);
        let img_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: w,
                height: h,
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
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let (image, memory, mut pooled) = if let Some(alloc) = allocator {
            let pooled = alloc
                .allocate_image(device, &img_info, MemoryLocation::GpuOnly, name)
                .ok()?;
            let img = pooled.image;
            (img, vk::DeviceMemory::null(), Some(pooled))
        } else {
            let image = unsafe { device.create_image(&img_info, None).ok()? };
            let req = unsafe { device.get_image_memory_requirements(image) };
            let mem_idx = (0..mem_props.memory_type_count).find(|&i| {
                (req.memory_type_bits & (1u32 << i)) != 0
                    && mem_props.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            });
            let Some(mem_idx) = mem_idx else {
                unsafe { device.destroy_image(image, None) };
                return None;
            };
            let memory = match unsafe {
                device.allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(mem_idx),
                    None,
                )
            } {
                Ok(m) => m,
                Err(_) => {
                    unsafe { device.destroy_image(image, None) };
                    return None;
                }
            };
            if unsafe { device.bind_image_memory(image, memory, 0) }.is_err() {
                unsafe {
                    device.free_memory(memory, None);
                    device.destroy_image(image, None);
                }
                return None;
            }
            (image, memory, None)
        };

        let view = match unsafe {
            device.create_image_view(
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
        } {
            Ok(v) => v,
            Err(_) => {
                if let Some(mut p) = pooled.take() {
                    unsafe { p.destroy(device) };
                } else {
                    unsafe {
                        device.free_memory(memory, None);
                        device.destroy_image(image, None);
                    }
                }
                return None;
            }
        };
        Some(NrdPoolTexture {
            image,
            memory,
            pooled,
            view,
            format,
            width: w,
            height: h,
            layout: vk::ImageLayout::UNDEFINED,
        })
    }

    /// # Safety
    /// The textures must not be referenced by in-flight GPU commands unless `retire` defers destruction.
    pub unsafe fn destroy_pool_textures(&mut self, retire: Option<&RetireQueue>) {
        let textures: Vec<NrdPoolTexture> = self
            .permanent_pool
            .drain(..)
            .chain(self.transient_pool.drain(..))
            .chain(self.fallback_tex.take())
            .collect();
        if !textures.is_empty() {
            let destroy_all = {
                let dev = self.device.clone();
                move || unsafe {
                    for mut t in textures {
                        if t.view != vk::ImageView::null() {
                            dev.destroy_image_view(t.view, None);
                        }
                        if let Some(mut pooled) = t.pooled.take() {
                            pooled.destroy(&dev);
                        } else {
                            if t.image != vk::Image::null() {
                                dev.destroy_image(t.image, None);
                            }
                            if t.memory != vk::DeviceMemory::null() {
                                dev.free_memory(t.memory, None);
                            }
                        }
                    }
                }
            };
            if let Some(rq) = retire {
                rq.retire_later(destroy_all);
            } else {
                destroy_all();
            }
        }
        self.permanent_layouts.clear();
        self.transient_layouts.clear();
        self.fallback_layout = vk::ImageLayout::UNDEFINED;
        self.pool_extent = (0, 0);
    }

    /// Destroys all Vulkan resources owned by this dispatcher, deferring destruction through
    /// `retire` when provided so in-flight command buffers never suffer use-after-free.
    ///
    /// # Safety
    /// If `retire` is `None`, all GPU work referencing this dispatcher must have completed.
    pub unsafe fn destroy_with_retire(&mut self, retire: Option<&RetireQueue>) {
        unsafe { self.destroy_pool_textures(retire) };

        let was_mapped = !self.cbuffer_mapped.is_null();
        self.cbuffer_mapped = ptr::null_mut();
        let cbuffer = std::mem::replace(&mut self.cbuffer, vk::Buffer::null());
        let cbuffer_memory = std::mem::replace(&mut self.cbuffer_memory, vk::DeviceMemory::null());
        let cbuffer_allocation = self.cbuffer_allocation.take();
        let cbuffer_allocator = self.allocator.clone();
        let pipelines = std::mem::take(&mut self.pipelines);
        let pipeline_layouts = std::mem::take(&mut self.pipeline_layouts);
        let set_layouts = std::mem::take(&mut self.set_layouts);
        self.pipeline_res_set_layouts.clear();
        let shader_modules = std::mem::take(&mut self.shader_modules);
        let frame_descriptor_pools = std::mem::replace(
            &mut self.frame_descriptor_pools,
            [vk::DescriptorPool::null(); 3],
        );
        let samplers = std::mem::replace(&mut self.samplers, [vk::Sampler::null(); 2]);
        self.shared_set_layout = vk::DescriptorSetLayout::null();

        let destroy_fn = {
            let dev = self.device.clone();
            move || unsafe {
                if let (Some(alloc), Some(allocator)) = (cbuffer_allocation, cbuffer_allocator) {
                    if cbuffer != vk::Buffer::null() {
                        dev.destroy_buffer(cbuffer, None);
                    }
                    let _ = allocator.free(alloc);
                } else {
                    if was_mapped && cbuffer_memory != vk::DeviceMemory::null() {
                        dev.unmap_memory(cbuffer_memory);
                    }
                    if cbuffer != vk::Buffer::null() {
                        dev.destroy_buffer(cbuffer, None);
                    }
                    if cbuffer_memory != vk::DeviceMemory::null() {
                        dev.free_memory(cbuffer_memory, None);
                    }
                }
                for p in pipelines {
                    if p != vk::Pipeline::null() {
                        dev.destroy_pipeline(p, None);
                    }
                }
                for pl in pipeline_layouts {
                    if pl != vk::PipelineLayout::null() {
                        dev.destroy_pipeline_layout(pl, None);
                    }
                }
                for sl in set_layouts {
                    if sl != vk::DescriptorSetLayout::null() {
                        dev.destroy_descriptor_set_layout(sl, None);
                    }
                }
                for sm in shader_modules {
                    if sm != vk::ShaderModule::null() {
                        dev.destroy_shader_module(sm, None);
                    }
                }
                for pool in frame_descriptor_pools {
                    if pool != vk::DescriptorPool::null() {
                        dev.destroy_descriptor_pool(pool, None);
                    }
                }
                for s in samplers {
                    if s != vk::Sampler::null() {
                        dev.destroy_sampler(s, None);
                    }
                }
            }
        };

        if let Some(rq) = retire {
            rq.retire_later(destroy_fn);
        } else {
            destroy_fn();
        }
    }

    /// Allocates `permanent_pool`, `transient_pool`, and the host-visible constant buffer on the
    /// GPU for the given viewport `width x height`.
    ///
    /// # Safety
    /// `instance` and `physical_device` must be valid Vulkan handles matching `self.device`.
    pub unsafe fn ensure_pools(
        &mut self,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        width: u32,
        height: u32,
        retire: Option<&RetireQueue>,
    ) {
        let w = width.max(1);
        let h = height.max(1);
        if self.pool_extent == (w, h)
            && (!self.permanent_pool.is_empty() || self.permanent_descs.is_empty())
            && self.cbuffer != vk::Buffer::null()
        {
            return;
        }
        unsafe { self.destroy_pool_textures(retire) };

        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        // Ensure constant buffer is allocated
        if self.cbuffer == vk::Buffer::null() {
            let total_cb_bytes =
                (self.cbuffer_slot_stride * self.cbuffer_total_slots.max(192)) as vk::DeviceSize;
            let buf_info = vk::BufferCreateInfo::default()
                .size(total_cb_bytes.max(4096))
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            if let Ok(buf) = unsafe { self.device.create_buffer(&buf_info, None) } {
                let req = unsafe { self.device.get_buffer_memory_requirements(buf) };
                if let Some(ref allocator) = self.allocator {
                    if let Ok(alloc) = allocator.allocate(&AllocationCreateDesc {
                        name: "nrd_constant_buffer",
                        requirements: req,
                        location: MemoryLocation::CpuToGpu,
                        linear: true,
                        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                    }) {
                        let mem = unsafe { alloc.memory() };
                        let offset = alloc.offset();
                        if unsafe { self.device.bind_buffer_memory(buf, mem, offset) }.is_ok() {
                            self.cbuffer = buf;
                            self.cbuffer_memory = vk::DeviceMemory::null();
                            self.cbuffer_mapped = alloc
                                .mapped_ptr()
                                .map(|p| p.as_ptr().cast::<u8>())
                                .unwrap_or(ptr::null_mut());
                            self.cbuffer_allocation = Some(alloc);
                        } else {
                            let _ = allocator.free(alloc);
                            unsafe { self.device.destroy_buffer(buf, None) };
                        }
                    } else {
                        unsafe { self.device.destroy_buffer(buf, None) };
                    }
                } else if let Some(mem_idx) = (0..mem_props.memory_type_count).find(|&i| {
                    (req.memory_type_bits & (1u32 << i)) != 0
                        && mem_props.memory_types[i as usize].property_flags.contains(
                            vk::MemoryPropertyFlags::HOST_VISIBLE
                                | vk::MemoryPropertyFlags::HOST_COHERENT,
                        )
                }) && let Ok(mem) = unsafe {
                    self.device.allocate_memory(
                        &vk::MemoryAllocateInfo::default()
                            .allocation_size(req.size)
                            .memory_type_index(mem_idx),
                        None,
                    )
                } {
                    if unsafe { self.device.bind_buffer_memory(buf, mem, 0) }.is_ok() {
                        self.cbuffer = buf;
                        self.cbuffer_memory = mem;
                        self.cbuffer_mapped = unsafe {
                            self.device
                                .map_memory(mem, 0, req.size, vk::MemoryMapFlags::empty())
                                .map(|p| p as *mut u8)
                                .unwrap_or(ptr::null_mut())
                        };
                    } else {
                        unsafe {
                            self.device.free_memory(mem, None);
                            self.device.destroy_buffer(buf, None);
                        }
                    }
                } else {
                    unsafe { self.device.destroy_buffer(buf, None) };
                }
            }
        }

        let alloc_ref = self.allocator.as_ref();
        for desc in &self.permanent_descs {
            let factor = u32::from(desc.downsample_factor.max(1));
            let fmt = nrd_format_to_vk(desc.format);
            if let Some(tex) = unsafe {
                Self::allocate_pool_texture(
                    &self.device,
                    alloc_ref,
                    &mem_props,
                    fmt,
                    w.div_ceil(factor),
                    h.div_ceil(factor),
                    "nrd_permanent_pool",
                )
            } {
                self.permanent_pool.push(tex);
                self.permanent_layouts.push(vk::ImageLayout::UNDEFINED);
            }
        }
        for desc in &self.transient_descs {
            let factor = u32::from(desc.downsample_factor.max(1));
            let fmt = nrd_format_to_vk(desc.format);
            if let Some(tex) = unsafe {
                Self::allocate_pool_texture(
                    &self.device,
                    alloc_ref,
                    &mem_props,
                    fmt,
                    w.div_ceil(factor),
                    h.div_ceil(factor),
                    "nrd_transient_pool",
                )
            } {
                self.transient_pool.push(tex);
                self.transient_layouts.push(vk::ImageLayout::UNDEFINED);
            }
        }
        self.fallback_tex = unsafe {
            Self::allocate_pool_texture(
                &self.device,
                alloc_ref,
                &mem_props,
                vk::Format::R16G16B16A16_SFLOAT,
                w,
                h,
                "nrd_fallback_tex",
            )
        };
        self.pool_extent = (w, h);
    }

    fn resolve_resource_image_view(
        &self,
        res: &NrdResourceDesc,
        user: &NrdUserTextureSet,
    ) -> (vk::Image, vk::ImageView) {
        let pair = match res.resource_type {
            NrdResourceType::InMv => user.in_mv,
            NrdResourceType::InNormalRoughness => user.in_normal_roughness,
            NrdResourceType::InViewZ => user.in_viewz,
            NrdResourceType::InDiffRadianceHitdist => user.in_diff_radiance_hitdist,
            NrdResourceType::InSpecRadianceHitdist => {
                if user.in_spec_radiance_hitdist.1 != vk::ImageView::null() {
                    user.in_spec_radiance_hitdist
                } else {
                    user.in_diff_radiance_hitdist
                }
            }
            NrdResourceType::OutDiffRadianceHitdist => user.out_diff_radiance_hitdist,
            NrdResourceType::OutSpecRadianceHitdist => user.out_spec_radiance_hitdist,
            NrdResourceType::PermanentPool => self
                .permanent_pool
                .get(res.index_in_pool as usize)
                .map(|t| (t.image, t.view))
                .unwrap_or((vk::Image::null(), vk::ImageView::null())),
            NrdResourceType::TransientPool => self
                .transient_pool
                .get(res.index_in_pool as usize)
                .map(|t| (t.image, t.view))
                .unwrap_or((vk::Image::null(), vk::ImageView::null())),
            _ => (vk::Image::null(), vk::ImageView::null()),
        };
        if pair.1 != vk::ImageView::null() {
            pair
        } else if let Some(ref fb) = self.fallback_tex {
            (fb.image, fb.view)
        } else {
            (vk::Image::null(), vk::ImageView::null())
        }
    }

    /// Updates constant buffers and descriptor sets, emits image layout barriers, binds pipelines
    /// and descriptor sets, and records `vkCmdDispatch` onto `cmd`.
    ///
    /// # Safety
    /// `cmd` must be in the recording state.
    pub unsafe fn record_dispatches_with_resources(
        &mut self,
        cmd: vk::CommandBuffer,
        dispatches: &[NrdDispatchDesc],
        user: &NrdUserTextureSet,
        user_layouts: &mut HashMap<vk::Image, vk::ImageLayout>,
    ) -> usize {
        if cmd == vk::CommandBuffer::null() {
            return 0;
        }
        let mut recorded = 0usize;

        let frame_slot = (self.frame_counter as usize) % self.frame_descriptor_pools.len();
        self.frame_counter = self.frame_counter.wrapping_add(1);
        let frame_pool = self.frame_descriptor_pools[frame_slot];
        if frame_pool != vk::DescriptorPool::null() {
            unsafe {
                let _ = self
                    .device
                    .reset_descriptor_pool(frame_pool, vk::DescriptorPoolResetFlags::empty());
            }
        }
        let fallback_img = self
            .fallback_tex
            .as_ref()
            .map(|t| t.image)
            .unwrap_or(vk::Image::null());

        for (dispatch_idx, d) in dispatches.iter().enumerate() {
            let idx = d.pipeline_index as usize;
            let Some(&pipe) = self.pipelines.get(idx) else {
                continue;
            };
            let Some(&pipe_layout) = self.pipeline_layouts.get(idx) else {
                continue;
            };
            if pipe == vk::Pipeline::null() || pipe_layout == vk::PipelineLayout::null() {
                continue;
            }

            let res_layout = self
                .pipeline_res_set_layouts
                .get(idx)
                .copied()
                .unwrap_or(vk::DescriptorSetLayout::null());

            let shared_set = if frame_pool != vk::DescriptorPool::null()
                && self.shared_set_layout != vk::DescriptorSetLayout::null()
                && self.res_space_idx != self.cb_space_idx
            {
                let s_layout = [self.shared_set_layout];
                unsafe {
                    self.device
                        .allocate_descriptor_sets(
                            &vk::DescriptorSetAllocateInfo::default()
                                .descriptor_pool(frame_pool)
                                .set_layouts(&s_layout),
                        )
                        .ok()
                        .and_then(|v| v.first().copied())
                        .unwrap_or(vk::DescriptorSet::null())
                }
            } else {
                vk::DescriptorSet::null()
            };

            let res_set = if frame_pool != vk::DescriptorPool::null()
                && res_layout != vk::DescriptorSetLayout::null()
            {
                let r_layout = [res_layout];
                unsafe {
                    self.device
                        .allocate_descriptor_sets(
                            &vk::DescriptorSetAllocateInfo::default()
                                .descriptor_pool(frame_pool)
                                .set_layouts(&r_layout),
                        )
                        .ok()
                        .and_then(|v| v.first().copied())
                        .unwrap_or(vk::DescriptorSet::null())
                }
            } else {
                vk::DescriptorSet::null()
            };

            let target_res_set = res_set;
            let target_cb_set = if self.res_space_idx == self.cb_space_idx {
                res_set
            } else {
                shared_set
            };

            // 1. Upload constant buffer data into a unique per-frame/per-dispatch slot
            let cb_slot = (frame_slot * 64 + dispatch_idx) % self.cbuffer_total_slots.max(1);
            let cb_offset = cb_slot * self.cbuffer_slot_stride;
            if !self.cbuffer_mapped.is_null()
                && !d.constant_buffer_data.is_null()
                && d.constant_buffer_data_size > 0
            {
                let copy_len = (d.constant_buffer_data_size as usize).min(self.cbuffer_slot_stride);
                unsafe {
                    ptr::copy_nonoverlapping(
                        d.constant_buffer_data,
                        self.cbuffer_mapped.add(cb_offset),
                        copy_len,
                    );
                }
            }

            if target_cb_set != vk::DescriptorSet::null() && self.cbuffer != vk::Buffer::null() {
                let cb_info = [vk::DescriptorBufferInfo {
                    buffer: self.cbuffer,
                    offset: cb_offset as vk::DeviceSize,
                    range: u64::from(self.cb_max_size),
                }];
                let s0_info = [vk::DescriptorImageInfo {
                    sampler: self.samplers[0],
                    image_view: vk::ImageView::null(),
                    image_layout: vk::ImageLayout::UNDEFINED,
                }];
                let s1_info = [vk::DescriptorImageInfo {
                    sampler: self.samplers[1],
                    image_view: vk::ImageView::null(),
                    image_layout: vk::ImageLayout::UNDEFINED,
                }];
                let writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_set(target_cb_set)
                        .dst_binding(self.offsets.constant_buffer_offset + self.cb_reg_idx)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&cb_info),
                    vk::WriteDescriptorSet::default()
                        .dst_set(target_cb_set)
                        .dst_binding(self.offsets.sampler_offset + self.samplers_base_reg_idx)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .image_info(&s0_info),
                    vk::WriteDescriptorSet::default()
                        .dst_set(target_cb_set)
                        .dst_binding(self.offsets.sampler_offset + self.samplers_base_reg_idx + 1)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .image_info(&s1_info),
                ];
                unsafe {
                    self.device.update_descriptor_sets(&writes, &[]);
                }
            }

            // 2. Update resource descriptors and record accurate image memory barriers
            let resources = if d.resources.is_null() || d.resources_num == 0 {
                &[][..]
            } else {
                unsafe { std::slice::from_raw_parts(d.resources, d.resources_num as usize) }
            };

            let mut barriers = Vec::new();
            let mut tex_reg = self.res_base_reg_idx;
            let mut uav_reg = self.res_base_reg_idx;

            for r_desc in resources {
                let (img, view) = self.resolve_resource_image_view(r_desc, user);
                let is_uav = r_desc.descriptor_type == NrdDescriptorType::StorageTexture;
                let dst_layout = if is_uav {
                    vk::ImageLayout::GENERAL
                } else {
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                };
                if img != vk::Image::null() {
                    let aspect_mask = if img == fallback_img {
                        vk::ImageAspectFlags::COLOR
                    } else {
                        match r_desc.resource_type {
                            NrdResourceType::PermanentPool => self
                                .permanent_pool
                                .get(r_desc.index_in_pool as usize)
                                .map(|t| aspect_mask_for_format(t.format))
                                .unwrap_or(vk::ImageAspectFlags::COLOR),
                            NrdResourceType::TransientPool => self
                                .transient_pool
                                .get(r_desc.index_in_pool as usize)
                                .map(|t| aspect_mask_for_format(t.format))
                                .unwrap_or(vk::ImageAspectFlags::COLOR),
                            NrdResourceType::InViewZ => {
                                aspect_mask_for_format(user.in_viewz_format)
                            }
                            _ => vk::ImageAspectFlags::COLOR,
                        }
                    };
                    let subrange = vk::ImageSubresourceRange {
                        aspect_mask,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    };
                    let old_layout = if img == fallback_img {
                        let prev = self.fallback_layout;
                        self.fallback_layout = dst_layout;
                        prev
                    } else {
                        match r_desc.resource_type {
                            NrdResourceType::PermanentPool => {
                                let pool_idx = r_desc.index_in_pool as usize;
                                let prev = self
                                    .permanent_layouts
                                    .get(pool_idx)
                                    .copied()
                                    .unwrap_or(vk::ImageLayout::UNDEFINED);
                                if let Some(slot) = self.permanent_layouts.get_mut(pool_idx) {
                                    *slot = dst_layout;
                                }
                                prev
                            }
                            NrdResourceType::TransientPool => {
                                let pool_idx = r_desc.index_in_pool as usize;
                                let prev = self
                                    .transient_layouts
                                    .get(pool_idx)
                                    .copied()
                                    .unwrap_or(vk::ImageLayout::UNDEFINED);
                                if let Some(slot) = self.transient_layouts.get_mut(pool_idx) {
                                    *slot = dst_layout;
                                }
                                prev
                            }
                            _ => {
                                let prev = user_layouts
                                    .get(&img)
                                    .copied()
                                    .unwrap_or(vk::ImageLayout::GENERAL);
                                user_layouts.insert(img, dst_layout);
                                prev
                            }
                        }
                    };
                    barriers.push(
                        vk::ImageMemoryBarrier::default()
                            .src_access_mask(if old_layout == vk::ImageLayout::UNDEFINED {
                                vk::AccessFlags::empty()
                            } else {
                                vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ
                            })
                            .dst_access_mask(if is_uav {
                                vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ
                            } else {
                                vk::AccessFlags::SHADER_READ
                            })
                            .old_layout(old_layout)
                            .new_layout(dst_layout)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(img)
                            .subresource_range(subrange),
                    );
                }
                if target_res_set != vk::DescriptorSet::null() && view != vk::ImageView::null() {
                    let binding = if is_uav {
                        let b = self.offsets.storage_texture_and_buffer_offset + uav_reg;
                        uav_reg += 1;
                        b
                    } else {
                        let b = self.offsets.texture_offset + tex_reg;
                        tex_reg += 1;
                        b
                    };
                    let img_info = [vk::DescriptorImageInfo {
                        sampler: vk::Sampler::null(),
                        image_view: view,
                        image_layout: dst_layout,
                    }];
                    let write = [vk::WriteDescriptorSet::default()
                        .dst_set(target_res_set)
                        .dst_binding(binding)
                        .descriptor_type(if is_uav {
                            vk::DescriptorType::STORAGE_IMAGE
                        } else {
                            vk::DescriptorType::SAMPLED_IMAGE
                        })
                        .image_info(&img_info)];
                    unsafe {
                        self.device.update_descriptor_sets(&write, &[]);
                    }
                } else if is_uav {
                    uav_reg += 1;
                } else {
                    tex_reg += 1;
                }
            }

            unsafe {
                if !barriers.is_empty() {
                    self.device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER
                            | vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &barriers,
                    );
                }

                self.device
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);

                if self.res_space_idx == self.cb_space_idx {
                    if target_res_set != vk::DescriptorSet::null() {
                        self.device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            pipe_layout,
                            self.res_space_idx,
                            &[target_res_set],
                            &[],
                        );
                    }
                } else {
                    let max_space = self.cb_space_idx.max(self.res_space_idx) as usize;
                    let mut sets = vec![shared_set; max_space + 1];
                    sets[self.cb_space_idx as usize] = shared_set;
                    sets[self.res_space_idx as usize] = res_set;
                    if sets.iter().all(|&s| s != vk::DescriptorSet::null()) {
                        self.device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            pipe_layout,
                            0,
                            &sets,
                            &[],
                        );
                    }
                }

                self.device.cmd_dispatch(
                    cmd,
                    u32::from(d.grid_width.max(1)),
                    u32::from(d.grid_height.max(1)),
                    1,
                );
            }
            recorded += 1;
        }
        recorded
    }

    /// Backwards-compatible entry point that records dispatches using fallback/internal pool views.
    ///
    /// # Safety
    /// `cmd` must be in the recording state.
    pub unsafe fn record_dispatches(
        &mut self,
        cmd: vk::CommandBuffer,
        dispatches: &[NrdDispatchDesc],
    ) -> usize {
        let mut layouts = HashMap::new();
        unsafe {
            self.record_dispatches_with_resources(
                cmd,
                dispatches,
                &NrdUserTextureSet::default(),
                &mut layouts,
            )
        }
    }
}

impl Drop for NrdVulkanDispatcher {
    fn drop(&mut self) {
        unsafe {
            self.destroy_with_retire(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn verify_nrd_c_abi_layouts() {
        assert_eq!(size_of::<NrdLibraryDesc>(), 40);
        assert_eq!(align_of::<NrdLibraryDesc>(), 8);
        assert_eq!(size_of::<NrdInstanceCreationDesc>(), 48);
        assert_eq!(align_of::<NrdInstanceCreationDesc>(), 8);
        assert_eq!(size_of::<NrdInstanceDesc>(), 112);
        assert_eq!(align_of::<NrdInstanceDesc>(), 8);
        assert_eq!(size_of::<NrdPipelineDesc>(), 320);
        assert_eq!(align_of::<NrdPipelineDesc>(), 8);
        assert_eq!(size_of::<NrdDispatchDesc>(), 56);
        assert_eq!(align_of::<NrdDispatchDesc>(), 8);
        assert_eq!(size_of::<NrdCommonSettings>(), 452);
        assert_eq!(align_of::<NrdCommonSettings>(), 4);
        assert_eq!(size_of::<NrdReblurSettings>(), 120);
        assert_eq!(align_of::<NrdReblurSettings>(), 4);
        assert_eq!(size_of::<NrdRelaxSettings>(), 148);
        assert_eq!(align_of::<NrdRelaxSettings>(), 4);
        assert_eq!(size_of::<NrdSigmaSettings>(), 24);
        assert_eq!(align_of::<NrdSigmaSettings>(), 4);
        assert_eq!(size_of::<NrdReferenceSettings>(), 4);
        assert_eq!(align_of::<NrdReferenceSettings>(), 4);
    }

    #[test]
    fn load_staged_nrd_if_present() {
        let natives_dir = std::env::var_os("VS_NATIVES_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("natives"));
        if natives_dir.join("libNRD.so").exists() {
            let lib = std::sync::Arc::new(
                NrdLibrary::load(&[natives_dir.as_path()]).expect("libNRD.so loads"),
            );
            let desc = lib.library_desc().expect("GetLibraryDesc non-null");
            assert_eq!(desc.version_major, 4);
            assert!(desc.supported_denoisers_num > 0);
            let inst = NrdInstance::create(
                lib,
                &[NrdDenoiserDesc {
                    identifier: 1,
                    denoiser: NrdDenoiser::ReblurDiffuseSpecular,
                }],
            )
            .expect("NrdInstance::create succeeds");
            let pipelines = inst.pipelines().expect("pipelines available");
            assert!(!pipelines.is_empty());
            assert!(pipelines[0].compute_shader_spirv.size > 0);
        }
    }
}
