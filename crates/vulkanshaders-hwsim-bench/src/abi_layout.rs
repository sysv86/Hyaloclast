// SPDX-License-Identifier: GPL-3.0-only
// Additional permissions under GPLv3 Section 7 apply to optional proprietary vendor SDK
// runtime loading; see LICENSE-EXCEPTION.md in the repository root.

//! Compile-time `core::mem::offset_of!` and cross-ABI (Linux SysV64 LP64 vs Windows Win64 LLP64)
//! layout verification for vendor SDK FFI structs, plus FidelityFX SDK 2.4.0 constant auditing.

use serde::{Deserialize, Serialize};
use std::mem::{align_of, offset_of, size_of};
use vulkanshaders_core::sdk::{fsr3, ngx, xess};

/// Target 64-bit C ABI model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbiModel {
    SysV64Lp64,
    Win64Llp64,
}

/// Primitive or nested C ABI type used by the cross-ABI layout simulator.
#[derive(Debug, Clone, Copy)]
pub enum CType {
    U8,
    U32,
    I32,
    U64,
    F32,
    CLong,
    WcharT,
    Ptr,
    VkNonDispatchableHandle,
    VkDispatchableHandle,
    Struct {
        size_sysv: usize,
        align_sysv: usize,
        size_win64: usize,
        align_win64: usize,
    },
}

impl CType {
    pub const fn size_align(self, abi: AbiModel) -> (usize, usize) {
        match self {
            Self::U8 => (1, 1),
            Self::U32 | Self::I32 | Self::F32 => (4, 4),
            Self::U64 | Self::Ptr | Self::VkNonDispatchableHandle | Self::VkDispatchableHandle => {
                (8, 8)
            }
            Self::CLong => match abi {
                AbiModel::SysV64Lp64 => (8, 8),
                AbiModel::Win64Llp64 => (4, 4),
            },
            Self::WcharT => match abi {
                AbiModel::SysV64Lp64 => (4, 4),
                AbiModel::Win64Llp64 => (2, 2),
            },
            Self::Struct {
                size_sysv,
                align_sysv,
                size_win64,
                align_win64,
            } => match abi {
                AbiModel::SysV64Lp64 => (size_sysv, align_sysv),
                AbiModel::Win64Llp64 => (size_win64, align_win64),
            },
        }
    }
}

/// Declarative field specification for a C struct.
#[derive(Debug, Clone, Copy)]
pub struct CFieldSpec {
    pub name: &'static str,
    pub ty: CType,
    pub count: usize,
}

/// Declarative struct specification paired with live Rust `#[repr(C)]` measurements.
#[derive(Debug, Clone, Copy)]
pub struct CStructSpec {
    pub name: &'static str,
    pub sdk: &'static str,
    pub fields: &'static [CFieldSpec],
    pub host_size: usize,
    pub host_align: usize,
    pub host_field_offsets: &'static [(&'static str, usize)],
}

/// Computed layout for a struct under a specific C ABI model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputedLayout {
    pub size: usize,
    pub align: usize,
    pub fields: Vec<ComputedField>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputedField {
    pub name: String,
    pub offset: usize,
    pub size: usize,
    pub align: usize,
}

/// Result of verifying one FFI struct across host Rust layout, SysV64, and Win64.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructAbiCheck {
    pub struct_name: String,
    pub sdk: String,
    pub passed: bool,
    pub cross_abi_identical: bool,
    pub host_size: usize,
    pub host_align: usize,
    pub sysv_size: usize,
    pub win64_size: usize,
    pub diagnostics: Vec<String>,
}

/// Result of auditing a scalar FFX SDK 2.4.0 ABI constant against `vulkanshaders-core`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantAbiCheck {
    pub name: String,
    pub passed: bool,
    pub actual_hex: String,
    pub expected_hex: String,
    pub detail: String,
}

/// Aggregated ABI verification report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbiAuditReport {
    pub structs: Vec<StructAbiCheck>,
    pub constants: Vec<ConstantAbiCheck>,
    pub all_structs_passed: bool,
    pub all_constants_passed: bool,
}

/// Computes exact C struct layout under `abi` (alignment padding + tail padding).
pub fn compute_c_layout(fields: &[CFieldSpec], abi: AbiModel) -> ComputedLayout {
    let mut offset = 0usize;
    let mut max_align = 1usize;
    let mut out = Vec::with_capacity(fields.len());

    for f in fields {
        let (elem_size, elem_align) = f.ty.size_align(abi);
        let align = elem_align.max(1);
        max_align = max_align.max(align);
        let aligned_off = (offset + (align - 1)) & !(align - 1);
        let total_size = elem_size * f.count;
        out.push(ComputedField {
            name: f.name.to_string(),
            offset: aligned_off,
            size: total_size,
            align,
        });
        offset = aligned_off + total_size;
    }

    let final_size = (offset + (max_align - 1)) & !(max_align - 1);
    ComputedLayout {
        size: final_size,
        align: max_align,
        fields: out,
    }
}

// Compile-time offset_of! locks for critical vulkanshaders-core FFI structs on 64-bit hosts.
#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(size_of::<fsr3::FfxApiHeader>() == 16);
    assert!(align_of::<fsr3::FfxApiHeader>() == 8);
    assert!(offset_of!(fsr3::FfxApiHeader, desc_type) == 0);
    assert!(offset_of!(fsr3::FfxApiHeader, p_next) == 8);

    assert!(size_of::<fsr3::FfxApiResourceDescription>() == 32);
    assert!(align_of::<fsr3::FfxApiResourceDescription>() == 4);

    assert!(size_of::<fsr3::FfxApiResource>() == 48);
    assert!(align_of::<fsr3::FfxApiResource>() == 8);
    assert!(offset_of!(fsr3::FfxApiResource, resource) == 0);
    assert!(offset_of!(fsr3::FfxApiResource, description) == 8);
    assert!(offset_of!(fsr3::FfxApiResource, state) == 40);

    assert!(size_of::<fsr3::FfxVkDeviceContext>() == 24);
    assert!(align_of::<fsr3::FfxVkDeviceContext>() == 8);
    assert!(offset_of!(fsr3::FfxVkDeviceContext, vk_device) == 0);
    assert!(offset_of!(fsr3::FfxVkDeviceContext, vk_physical_device) == 8);
    assert!(offset_of!(fsr3::FfxVkDeviceContext, vk_device_proc_addr) == 16);

    assert!(size_of::<fsr3::FfxCreateBackendVKDesc>() == 40);
    assert!(align_of::<fsr3::FfxCreateBackendVKDesc>() == 8);
    assert!(offset_of!(fsr3::FfxCreateBackendVKDesc, header) == 0);
    assert!(offset_of!(fsr3::FfxCreateBackendVKDesc, device_context) == 16);
    assert!(offset_of!(fsr3::FfxCreateBackendVKDesc, device_context.vk_device_proc_addr) == 32);

    assert!(size_of::<fsr3::FfxCreateContextDescUpscale>() == 48);
    assert!(align_of::<fsr3::FfxCreateContextDescUpscale>() == 8);
    assert!(offset_of!(fsr3::FfxCreateContextDescUpscale, header) == 0);
    assert!(offset_of!(fsr3::FfxCreateContextDescUpscale, flags) == 16);
    assert!(offset_of!(fsr3::FfxCreateContextDescUpscale, max_render_size) == 20);
    assert!(offset_of!(fsr3::FfxCreateContextDescUpscale, max_upscale_size) == 28);
    assert!(offset_of!(fsr3::FfxCreateContextDescUpscale, fp_message) == 40);

    assert!(size_of::<fsr3::FfxDispatchDescUpscale>() == 432);
    assert!(align_of::<fsr3::FfxDispatchDescUpscale>() == 8);
    assert!(offset_of!(fsr3::FfxDispatchDescUpscale, header) == 0);
    assert!(offset_of!(fsr3::FfxDispatchDescUpscale, command_list) == 16);
    assert!(offset_of!(fsr3::FfxDispatchDescUpscale, color) == 24);
    assert!(offset_of!(fsr3::FfxDispatchDescUpscale, output) == 312);
    assert!(offset_of!(fsr3::FfxDispatchDescUpscale, flags) == 428);

    assert!(size_of::<ngx::NvSdkNgxImageViewInfoVk>() == 48);
    assert!(align_of::<ngx::NvSdkNgxImageViewInfoVk>() == 8);
    assert!(offset_of!(ngx::NvSdkNgxImageViewInfoVk, image_view) == 0);
    assert!(offset_of!(ngx::NvSdkNgxImageViewInfoVk, image) == 8);
    assert!(offset_of!(ngx::NvSdkNgxImageViewInfoVk, subresource_range) == 16);
    assert!(offset_of!(ngx::NvSdkNgxImageViewInfoVk, format) == 36);
    assert!(offset_of!(ngx::NvSdkNgxImageViewInfoVk, width) == 40);
    assert!(offset_of!(ngx::NvSdkNgxImageViewInfoVk, height) == 44);

    assert!(size_of::<ngx::NvSdkNgxResourceVk>() == 56);
    assert!(align_of::<ngx::NvSdkNgxResourceVk>() == 8);
    assert!(offset_of!(ngx::NvSdkNgxResourceVk, resource) == 0);
    assert!(offset_of!(ngx::NvSdkNgxResourceVk, resource_type) == 48);
    assert!(offset_of!(ngx::NvSdkNgxResourceVk, read_write) == 52);

    assert!(size_of::<xess::XessVkInitParams>() == 64);
    assert!(align_of::<xess::XessVkInitParams>() == 8);
    assert!(offset_of!(xess::XessVkInitParams, output_resolution) == 0);
    assert!(offset_of!(xess::XessVkInitParams, temp_buffer_heap) == 24);
    assert!(offset_of!(xess::XessVkInitParams, pipeline_cache) == 56);

    assert!(size_of::<xess::XessVkExecuteParams>() == 360);
    assert!(align_of::<xess::XessVkExecuteParams>() == 8);
    assert!(offset_of!(xess::XessVkExecuteParams, color_texture) == 0);
    assert!(offset_of!(xess::XessVkExecuteParams, output_texture) == 240);
    assert!(offset_of!(xess::XessVkExecuteParams, jitter_offset_x) == 288);
    assert!(offset_of!(xess::XessVkExecuteParams, output_color_base) == 352);
};

const FFX_HEADER_TY: CType = CType::Struct {
    size_sysv: 16,
    align_sysv: 8,
    size_win64: 16,
    align_win64: 8,
};
const FFX_DIM_TY: CType = CType::Struct {
    size_sysv: 8,
    align_sysv: 4,
    size_win64: 8,
    align_win64: 4,
};
const FFX_FLOAT2_TY: CType = CType::Struct {
    size_sysv: 8,
    align_sysv: 4,
    size_win64: 8,
    align_win64: 4,
};
const FFX_RES_DESC_TY: CType = CType::Struct {
    size_sysv: 32,
    align_sysv: 4,
    size_win64: 32,
    align_win64: 4,
};
const FFX_RESOURCE_TY: CType = CType::Struct {
    size_sysv: 48,
    align_sysv: 8,
    size_win64: 48,
    align_win64: 8,
};
const FFX_VK_DEV_CTX_TY: CType = CType::Struct {
    size_sysv: 24,
    align_sysv: 8,
    size_win64: 24,
    align_win64: 8,
};
const XESS_VIEW_INFO_TY: CType = CType::Struct {
    size_sysv: 48,
    align_sysv: 8,
    size_win64: 48,
    align_win64: 8,
};
const XESS_2D_TY: CType = CType::Struct {
    size_sysv: 8,
    align_sysv: 4,
    size_win64: 8,
    align_win64: 4,
};
const VK_SUBRESOURCE_RANGE_TY: CType = CType::Struct {
    size_sysv: 20,
    align_sysv: 4,
    size_win64: 20,
    align_win64: 4,
};

fn canonical_specs() -> Vec<CStructSpec> {
    vec![
        CStructSpec {
            name: "FfxApiHeader",
            sdk: "FSR3/FSR4",
            fields: &[
                CFieldSpec { name: "desc_type", ty: CType::U64, count: 1 },
                CFieldSpec { name: "p_next", ty: CType::Ptr, count: 1 },
            ],
            host_size: size_of::<fsr3::FfxApiHeader>(),
            host_align: align_of::<fsr3::FfxApiHeader>(),
            host_field_offsets: &[
                ("desc_type", offset_of!(fsr3::FfxApiHeader, desc_type)),
                ("p_next", offset_of!(fsr3::FfxApiHeader, p_next)),
            ],
        },
        CStructSpec {
            name: "FfxApiResource",
            sdk: "FSR3/FSR4",
            fields: &[
                CFieldSpec { name: "resource", ty: CType::Ptr, count: 1 },
                CFieldSpec { name: "description", ty: FFX_RES_DESC_TY, count: 1 },
                CFieldSpec { name: "state", ty: CType::U32, count: 1 },
            ],
            host_size: size_of::<fsr3::FfxApiResource>(),
            host_align: align_of::<fsr3::FfxApiResource>(),
            host_field_offsets: &[
                ("resource", offset_of!(fsr3::FfxApiResource, resource)),
                ("description", offset_of!(fsr3::FfxApiResource, description)),
                ("state", offset_of!(fsr3::FfxApiResource, state)),
            ],
        },
        CStructSpec {
            name: "FfxCreateBackendVKDesc",
            sdk: "FSR3/FSR4",
            fields: &[
                CFieldSpec { name: "header", ty: FFX_HEADER_TY, count: 1 },
                CFieldSpec { name: "device_context", ty: FFX_VK_DEV_CTX_TY, count: 1 },
            ],
            host_size: size_of::<fsr3::FfxCreateBackendVKDesc>(),
            host_align: align_of::<fsr3::FfxCreateBackendVKDesc>(),
            host_field_offsets: &[
                ("header", offset_of!(fsr3::FfxCreateBackendVKDesc, header)),
                ("device_context", offset_of!(fsr3::FfxCreateBackendVKDesc, device_context)),
            ],
        },
        CStructSpec {
            name: "FfxCreateContextDescUpscale",
            sdk: "FSR3/FSR4",
            fields: &[
                CFieldSpec { name: "header", ty: FFX_HEADER_TY, count: 1 },
                CFieldSpec { name: "flags", ty: CType::U32, count: 1 },
                CFieldSpec { name: "max_render_size", ty: FFX_DIM_TY, count: 1 },
                CFieldSpec { name: "max_upscale_size", ty: FFX_DIM_TY, count: 1 },
                CFieldSpec { name: "fp_message", ty: CType::Ptr, count: 1 },
            ],
            host_size: size_of::<fsr3::FfxCreateContextDescUpscale>(),
            host_align: align_of::<fsr3::FfxCreateContextDescUpscale>(),
            host_field_offsets: &[
                ("header", offset_of!(fsr3::FfxCreateContextDescUpscale, header)),
                ("flags", offset_of!(fsr3::FfxCreateContextDescUpscale, flags)),
                ("max_render_size", offset_of!(fsr3::FfxCreateContextDescUpscale, max_render_size)),
                ("max_upscale_size", offset_of!(fsr3::FfxCreateContextDescUpscale, max_upscale_size)),
                ("fp_message", offset_of!(fsr3::FfxCreateContextDescUpscale, fp_message)),
            ],
        },
        CStructSpec {
            name: "FfxDispatchDescUpscale",
            sdk: "FSR3/FSR4",
            fields: &[
                CFieldSpec { name: "header", ty: FFX_HEADER_TY, count: 1 },
                CFieldSpec { name: "command_list", ty: CType::VkDispatchableHandle, count: 1 },
                CFieldSpec { name: "color", ty: FFX_RESOURCE_TY, count: 1 },
                CFieldSpec { name: "depth", ty: FFX_RESOURCE_TY, count: 1 },
                CFieldSpec { name: "motion_vectors", ty: FFX_RESOURCE_TY, count: 1 },
                CFieldSpec { name: "exposure", ty: FFX_RESOURCE_TY, count: 1 },
                CFieldSpec { name: "reactive", ty: FFX_RESOURCE_TY, count: 1 },
                CFieldSpec { name: "transparency_and_composition", ty: FFX_RESOURCE_TY, count: 1 },
                CFieldSpec { name: "output", ty: FFX_RESOURCE_TY, count: 1 },
                CFieldSpec { name: "jitter_offset", ty: FFX_FLOAT2_TY, count: 1 },
                CFieldSpec { name: "motion_vector_scale", ty: FFX_FLOAT2_TY, count: 1 },
                CFieldSpec { name: "render_size", ty: FFX_DIM_TY, count: 1 },
                CFieldSpec { name: "upscale_size", ty: FFX_DIM_TY, count: 1 },
                CFieldSpec { name: "enable_sharpening", ty: CType::U8, count: 1 },
                CFieldSpec { name: "sharpness", ty: CType::F32, count: 1 },
                CFieldSpec { name: "frame_time_delta", ty: CType::F32, count: 1 },
                CFieldSpec { name: "pre_exposure", ty: CType::F32, count: 1 },
                CFieldSpec { name: "reset", ty: CType::U8, count: 1 },
                CFieldSpec { name: "camera_near", ty: CType::F32, count: 1 },
                CFieldSpec { name: "camera_far", ty: CType::F32, count: 1 },
                CFieldSpec { name: "camera_fov_angle_vertical", ty: CType::F32, count: 1 },
                CFieldSpec { name: "view_space_to_meters_factor", ty: CType::F32, count: 1 },
                CFieldSpec { name: "flags", ty: CType::U32, count: 1 },
            ],
            host_size: size_of::<fsr3::FfxDispatchDescUpscale>(),
            host_align: align_of::<fsr3::FfxDispatchDescUpscale>(),
            host_field_offsets: &[
                ("header", offset_of!(fsr3::FfxDispatchDescUpscale, header)),
                ("command_list", offset_of!(fsr3::FfxDispatchDescUpscale, command_list)),
                ("color", offset_of!(fsr3::FfxDispatchDescUpscale, color)),
                ("output", offset_of!(fsr3::FfxDispatchDescUpscale, output)),
                ("enable_sharpening", offset_of!(fsr3::FfxDispatchDescUpscale, enable_sharpening)),
                ("reset", offset_of!(fsr3::FfxDispatchDescUpscale, reset)),
                ("flags", offset_of!(fsr3::FfxDispatchDescUpscale, flags)),
            ],
        },
        CStructSpec {
            name: "NvSdkNgxImageViewInfoVk",
            sdk: "NGX/DLSS4",
            fields: &[
                CFieldSpec { name: "image_view", ty: CType::VkNonDispatchableHandle, count: 1 },
                CFieldSpec { name: "image", ty: CType::VkNonDispatchableHandle, count: 1 },
                CFieldSpec { name: "subresource_range", ty: VK_SUBRESOURCE_RANGE_TY, count: 1 },
                CFieldSpec { name: "format", ty: CType::I32, count: 1 },
                CFieldSpec { name: "width", ty: CType::U32, count: 1 },
                CFieldSpec { name: "height", ty: CType::U32, count: 1 },
            ],
            host_size: size_of::<ngx::NvSdkNgxImageViewInfoVk>(),
            host_align: align_of::<ngx::NvSdkNgxImageViewInfoVk>(),
            host_field_offsets: &[
                ("image_view", offset_of!(ngx::NvSdkNgxImageViewInfoVk, image_view)),
                ("image", offset_of!(ngx::NvSdkNgxImageViewInfoVk, image)),
                ("subresource_range", offset_of!(ngx::NvSdkNgxImageViewInfoVk, subresource_range)),
                ("format", offset_of!(ngx::NvSdkNgxImageViewInfoVk, format)),
                ("width", offset_of!(ngx::NvSdkNgxImageViewInfoVk, width)),
                ("height", offset_of!(ngx::NvSdkNgxImageViewInfoVk, height)),
            ],
        },
        CStructSpec {
            name: "NvSdkNgxResourceVk",
            sdk: "NGX/DLSS4",
            fields: &[
                CFieldSpec {
                    name: "resource",
                    ty: CType::Struct {
                        size_sysv: 48,
                        align_sysv: 8,
                        size_win64: 48,
                        align_win64: 8,
                    },
                    count: 1,
                },
                CFieldSpec { name: "resource_type", ty: CType::U32, count: 1 },
                CFieldSpec { name: "read_write", ty: CType::U8, count: 1 },
            ],
            host_size: size_of::<ngx::NvSdkNgxResourceVk>(),
            host_align: align_of::<ngx::NvSdkNgxResourceVk>(),
            host_field_offsets: &[
                ("resource", offset_of!(ngx::NvSdkNgxResourceVk, resource)),
                ("resource_type", offset_of!(ngx::NvSdkNgxResourceVk, resource_type)),
                ("read_write", offset_of!(ngx::NvSdkNgxResourceVk, read_write)),
            ],
        },
        CStructSpec {
            name: "XessVkInitParams",
            sdk: "XeSS",
            fields: &[
                CFieldSpec { name: "output_resolution", ty: XESS_2D_TY, count: 1 },
                CFieldSpec { name: "quality_setting", ty: CType::U32, count: 1 },
                CFieldSpec { name: "init_flags", ty: CType::U32, count: 1 },
                CFieldSpec { name: "creation_node_mask", ty: CType::U32, count: 1 },
                CFieldSpec { name: "visible_node_mask", ty: CType::U32, count: 1 },
                CFieldSpec { name: "temp_buffer_heap", ty: CType::VkNonDispatchableHandle, count: 1 },
                CFieldSpec { name: "buffer_heap_offset", ty: CType::U64, count: 1 },
                CFieldSpec { name: "temp_texture_heap", ty: CType::VkNonDispatchableHandle, count: 1 },
                CFieldSpec { name: "texture_heap_offset", ty: CType::U64, count: 1 },
                CFieldSpec { name: "pipeline_cache", ty: CType::VkNonDispatchableHandle, count: 1 },
            ],
            host_size: size_of::<xess::XessVkInitParams>(),
            host_align: align_of::<xess::XessVkInitParams>(),
            host_field_offsets: &[
                ("output_resolution", offset_of!(xess::XessVkInitParams, output_resolution)),
                ("temp_buffer_heap", offset_of!(xess::XessVkInitParams, temp_buffer_heap)),
                ("pipeline_cache", offset_of!(xess::XessVkInitParams, pipeline_cache)),
            ],
        },
        CStructSpec {
            name: "XessVkExecuteParams",
            sdk: "XeSS",
            fields: &[
                CFieldSpec { name: "color_texture", ty: XESS_VIEW_INFO_TY, count: 1 },
                CFieldSpec { name: "velocity_texture", ty: XESS_VIEW_INFO_TY, count: 1 },
                CFieldSpec { name: "depth_texture", ty: XESS_VIEW_INFO_TY, count: 1 },
                CFieldSpec { name: "exposure_scale_texture", ty: XESS_VIEW_INFO_TY, count: 1 },
                CFieldSpec { name: "responsive_pixel_mask_texture", ty: XESS_VIEW_INFO_TY, count: 1 },
                CFieldSpec { name: "output_texture", ty: XESS_VIEW_INFO_TY, count: 1 },
                CFieldSpec { name: "jitter_offset_x", ty: CType::F32, count: 1 },
                CFieldSpec { name: "jitter_offset_y", ty: CType::F32, count: 1 },
                CFieldSpec { name: "exposure_scale", ty: CType::F32, count: 1 },
                CFieldSpec { name: "reset_history", ty: CType::U32, count: 1 },
                CFieldSpec { name: "input_width", ty: CType::U32, count: 1 },
                CFieldSpec { name: "input_height", ty: CType::U32, count: 1 },
                CFieldSpec { name: "input_color_base", ty: XESS_2D_TY, count: 1 },
                CFieldSpec { name: "input_motion_vector_base", ty: XESS_2D_TY, count: 1 },
                CFieldSpec { name: "input_depth_base", ty: XESS_2D_TY, count: 1 },
                CFieldSpec { name: "input_responsive_mask_base", ty: XESS_2D_TY, count: 1 },
                CFieldSpec { name: "reserved0", ty: XESS_2D_TY, count: 1 },
                CFieldSpec { name: "output_color_base", ty: XESS_2D_TY, count: 1 },
            ],
            host_size: size_of::<xess::XessVkExecuteParams>(),
            host_align: align_of::<xess::XessVkExecuteParams>(),
            host_field_offsets: &[
                ("color_texture", offset_of!(xess::XessVkExecuteParams, color_texture)),
                ("output_texture", offset_of!(xess::XessVkExecuteParams, output_texture)),
                ("jitter_offset_x", offset_of!(xess::XessVkExecuteParams, jitter_offset_x)),
                ("output_color_base", offset_of!(xess::XessVkExecuteParams, output_color_base)),
            ],
        },
    ]
}

/// Runs the full struct layout and FidelityFX SDK 2.4.0 constant audit.
pub fn run_abi_audit(strict_cross_abi: bool) -> AbiAuditReport {
    let host_abi = if cfg!(windows) {
        AbiModel::Win64Llp64
    } else {
        AbiModel::SysV64Lp64
    };

    let mut structs = Vec::new();
    for spec in canonical_specs() {
        let sysv = compute_c_layout(spec.fields, AbiModel::SysV64Lp64);
        let win64 = compute_c_layout(spec.fields, AbiModel::Win64Llp64);
        let host_expected = match host_abi {
            AbiModel::SysV64Lp64 => &sysv,
            AbiModel::Win64Llp64 => &win64,
        };

        let mut diagnostics = Vec::new();
        if spec.host_size != host_expected.size {
            diagnostics.push(format!(
                "host size {} != expected {} for {:?}",
                spec.host_size, host_expected.size, host_abi
            ));
        }
        if spec.host_align != host_expected.align {
            diagnostics.push(format!(
                "host align {} != expected {} for {:?}",
                spec.host_align, host_expected.align, host_abi
            ));
        }
        for &(field_name, host_off) in spec.host_field_offsets {
            match host_expected.fields.iter().find(|f| f.name == field_name) {
                Some(expected_field) if expected_field.offset != host_off => {
                    diagnostics.push(format!(
                        "field `{field_name}` offset {host_off} != expected {}",
                        expected_field.offset
                    ));
                }
                None => diagnostics.push(format!("field `{field_name}` missing from spec")),
                _ => {}
            }
        }

        let cross_abi_identical = sysv == win64;
        if strict_cross_abi && !cross_abi_identical {
            diagnostics.push(format!(
                "SysV64 ({} B) and Win64 ({} B) layouts diverge",
                sysv.size, win64.size
            ));
        }

        structs.push(StructAbiCheck {
            struct_name: spec.name.to_string(),
            sdk: spec.sdk.to_string(),
            passed: diagnostics.is_empty(),
            cross_abi_identical,
            host_size: spec.host_size,
            host_align: spec.host_align,
            sysv_size: sysv.size,
            win64_size: win64.size,
            diagnostics,
        });
    }

    // Audit FidelityFX SDK 2.4.0 backend/effect constants in `vulkanshaders-core::sdk::fsr3`.
    let mut constants = Vec::new();

    let vk_backend_actual = fsr3::FFX_API_BACKEND_ID_VK;
    let vk_backend_sdk24: u64 = 0x0200_0000;
    let vk_backend_passed = vk_backend_actual == vk_backend_sdk24;
    constants.push(ConstantAbiCheck {
        name: "fsr3::FFX_API_BACKEND_ID_VK".to_string(),
        passed: vk_backend_passed,
        actual_hex: format!("{vk_backend_actual:#010x}"),
        expected_hex: format!("{vk_backend_sdk24:#010x}"),
        detail: if vk_backend_passed {
            "Matches FidelityFX SDK 2.4.0 FFX_API_BACKEND_ID_VK".to_string()
        } else if vk_backend_actual == 0x0100_0000 {
            "BUG DETECTED in vulkanshaders-core::sdk::fsr3: 0x0100_0000 is FFX_API_BACKEND_ID_XBOX in FidelityFX SDK 2.4.0; Vulkan is 0x0200_0000 (legacy SDK 1.1.x used 0x0000_0003)".to_string()
        } else {
            "Unexpected FFX_API_BACKEND_ID_VK value for FidelityFX SDK 2.4.0".to_string()
        },
    });

    let upscale_create_actual = fsr3::FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    constants.push(ConstantAbiCheck {
        name: "fsr3::FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE".to_string(),
        passed: upscale_create_actual == 0x0001_0000,
        actual_hex: format!("{upscale_create_actual:#010x}"),
        expected_hex: "0x00010000".to_string(),
        detail: "Matches FidelityFX SDK 2.4.0 ffx_upscale.h".to_string(),
    });

    let upscale_dispatch_actual = fsr3::FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    constants.push(ConstantAbiCheck {
        name: "fsr3::FFX_API_DISPATCH_DESC_TYPE_UPSCALE".to_string(),
        passed: upscale_dispatch_actual == 0x0001_0001,
        actual_hex: format!("{upscale_dispatch_actual:#010x}"),
        expected_hex: "0x00010001".to_string(),
        detail: "Matches FidelityFX SDK 2.4.0 ffx_upscale.h".to_string(),
    });

    let fg_create_actual = fsr3::FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    constants.push(ConstantAbiCheck {
        name: "fsr3::FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION".to_string(),
        passed: fg_create_actual == 0x0002_0001,
        actual_hex: format!("{fg_create_actual:#010x}"),
        expected_hex: "0x00020001".to_string(),
        detail: "Matches FidelityFX SDK 2.4.0 ffx_framegeneration.h".to_string(),
    });

    let all_structs_passed = structs.iter().all(|s| s.passed);
    let all_constants_passed = constants.iter().all(|c| c.passed);

    AbiAuditReport {
        structs,
        constants,
        all_structs_passed,
        all_constants_passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_core_ffi_structs_pass_cross_abi_layout_check() {
        let report = run_abi_audit(true);
        assert!(
            report.all_structs_passed,
            "ABI struct layout check failed: {:#?}",
            report.structs
        );
    }

    #[test]
    fn c_long_and_wchar_divergence_is_detected_by_simulator() {
        let fields = [
            CFieldSpec { name: "a", ty: CType::CLong, count: 1 },
            CFieldSpec { name: "b", ty: CType::WcharT, count: 1 },
        ];
        let sysv = compute_c_layout(&fields, AbiModel::SysV64Lp64);
        let win64 = compute_c_layout(&fields, AbiModel::Win64Llp64);
        assert_eq!(sysv.size, 16);
        assert_eq!(win64.size, 8);
    }
}
