// SPDX-License-Identifier: GPL-3.0-only
// Additional permissions under GPLv3 Section 7 apply to optional proprietary vendor SDK
// runtime loading; see LICENSE-EXCEPTION.md in the repository root.

//! Modeled GPU hardware profiles across 8 real GPUs and 2 OS driver stacks (Linux / Windows),
//! together with AMD FSR 4.1.2 (FidelityFX SDK 2.4.0) network selection and upscaler fallback rules.

use crate::error::HwsimError;
use ash::vk;
use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// PCI vendor IDs.
pub const VENDOR_AMD: u32 = 0x1002;
pub const VENDOR_NVIDIA: u32 = 0x10DE;
pub const VENDOR_INTEL: u32 = 0x8086;

/// Target GPU hardware profiles supported by the hardware-free simulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuId {
    Rx9070Xt,
    Rx7900Xtx,
    Rx6800Xt,
    StrixPoint,
    Rtx5090,
    Rtx4090,
    Rtx3050Mobile,
    ArcB580,
}

impl GpuId {
    pub const ALL: &'static [GpuId] = &[
        GpuId::Rx9070Xt,
        GpuId::Rx7900Xtx,
        GpuId::Rx6800Xt,
        GpuId::StrixPoint,
        GpuId::Rtx5090,
        GpuId::Rtx4090,
        GpuId::Rtx3050Mobile,
        GpuId::ArcB580,
    ];

    pub const fn slug(self) -> &'static str {
        match self {
            Self::Rx9070Xt => "rx9070xt",
            Self::Rx7900Xtx => "rx7900xtx",
            Self::Rx6800Xt => "rx6800xt",
            Self::StrixPoint => "strix_point",
            Self::Rtx5090 => "rtx5090",
            Self::Rtx4090 => "rtx4090",
            Self::Rtx3050Mobile => "rtx3050m",
            Self::ArcB580 => "arc_b580",
        }
    }
}

impl fmt::Display for GpuId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.slug())
    }
}

impl FromStr for GpuId {
    type Err = HwsimError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "rx9070xt" | "rx-9070-xt" | "rdna4" | "navi48" => Ok(Self::Rx9070Xt),
            "rx7900xtx" | "rx-7900-xtx" | "rdna3" | "navi31" => Ok(Self::Rx7900Xtx),
            "rx6800xt" | "rx-6800-xt" | "rdna2" | "navi21" => Ok(Self::Rx6800Xt),
            "strix_point" | "strix-point" | "890m" | "rdna35" => Ok(Self::StrixPoint),
            "rtx5090" | "rtx-5090" | "blackwell" => Ok(Self::Rtx5090),
            "rtx4090" | "rtx-4090" | "ada" => Ok(Self::Rtx4090),
            "rtx3050m" | "rtx3050mobile" | "rtx-3050-mobile" | "ampere" => Ok(Self::Rtx3050Mobile),
            "arc_b580" | "arc-b580" | "b580" | "battlemage" => Ok(Self::ArcB580),
            other => Err(HwsimError::UnknownGpuProfile(other.to_string())),
        }
    }
}

/// Operating system driver model being simulated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriverModel {
    Linux,
    Windows,
}

impl DriverModel {
    pub const ALL: &'static [DriverModel] = &[DriverModel::Linux, DriverModel::Windows];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Linux => "linux",
            Self::Windows => "windows",
        }
    }
}

impl fmt::Display for DriverModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for DriverModel {
    type Err = HwsimError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "linux" | "radv" | "anv" => Ok(Self::Linux),
            "windows" | "win" | "win64" | "adrenalin" => Ok(Self::Windows),
            other => Err(HwsimError::UnknownDriverModel(other.to_string())),
        }
    }
}

/// PCI and AMD kernel-driver ASIC identity fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AsicIdentity {
    pub vendor_id: u32,
    pub device_id: u32,
    pub is_integrated: bool,
    /// AMD kernel driver family ID (`D3DKMTQueryAdapterInfo` on Windows / `AMDGPU_INFO_DEV_INFO` on Linux).
    pub amd_family: Option<u32>,
    /// AMD kernel driver revision ID.
    pub amd_revision: Option<u32>,
    pub gfx_ip: &'static str,
}

/// Cooperative matrix element component type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoopComponentType {
    Float8E4M3,
    Float8E5M2,
    Float16,
    Float32,
    Int8,
    Int32,
}

/// Cooperative matrix execution scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoopScope {
    Subgroup,
}

/// Hardware-supported `VK_KHR_cooperative_matrix` tile shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CoopMatrixShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a_type: CoopComponentType,
    pub b_type: CoopComponentType,
    pub c_type: CoopComponentType,
    pub result_type: CoopComponentType,
    pub scope: CoopScope,
}

bitflags! {
    /// Modeled Vulkan and vendor upscaler hardware feature flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct FeatureFlags: u64 {
        const SHADER_FLOAT8                 = 1 << 0;  // VK_EXT_shader_float8
        const SHADER_FLOAT8_COOP_MATRIX     = 1 << 1;  // shaderFloat8CooperativeMatrix
        const COOP_MATRIX_KHR               = 1 << 2;  // VK_KHR_cooperative_matrix
        const INT8_DOT_PRODUCT              = 1 << 3;  // shaderIntegerDotProduct (DP4a)
        const SHADER_FLOAT16                = 1 << 4;  // shaderFloat16
        const SHADER_INT8                   = 1 << 5;  // shaderInt8
        const COMPUTE_DERIVATIVES_LINEAR    = 1 << 6;  // VK_KHR_compute_shader_derivatives
        const SUBGROUP_SIZE_CONTROL         = 1 << 7;  // subgroupSizeControl (core 1.3)
        const STORAGE_IMAGE_WRITE_NO_FMT    = 1 << 8;  // shaderStorageImageWriteWithoutFormat
        const RT_PIPELINE                   = 1 << 9;  // VK_KHR_ray_tracing_pipeline
        const RAY_QUERY                     = 1 << 10; // VK_KHR_ray_query
        const SER_EXT                       = 1 << 11; // VK_EXT_ray_tracing_invocation_reorder
        const SER_NV                        = 1 << 12; // VK_NV_ray_tracing_invocation_reorder
        const OMM_EXT                       = 1 << 13; // VK_EXT_opacity_micromap
        const SYNC2                         = 1 << 14; // VK_KHR_synchronization2 (core 1.3)
        const DYNAMIC_RENDERING             = 1 << 15; // VK_KHR_dynamic_rendering (core 1.3)
        const NGX_DLSS4                     = 1 << 16; // NVIDIA DLSS 4 / 4.5 Transformer
        const XESS_XMX                      = 1 << 17; // Intel XeSS XMX hardware path
    }
}

impl Serialize for FeatureFlags {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(self.bits())
    }
}

impl<'de> Deserialize<'de> for FeatureFlags {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bits = u64::deserialize(deserializer)?;
        Ok(Self::from_bits_retain(bits))
    }
}

/// Bundled AMD FidelityFX SDK 2.4.0 upscaler network selected for a physical device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fsr4Network {
    /// RDNA 4 discrete FP8 WMMA network (`FSR4`).
    Fsr4Fp8,
    /// RDNA 3 / Strix Halo full INT8 network (`FSR4-i8`).
    Fsr4Int8,
    /// RDNA 2 discrete INT8 network (`FSR4-i8-2`).
    Fsr4Int8Navi2,
    /// Van Gogh / Strix Point compact INT8 network (`FSR4L-i8`).
    Fsr4LiteInt8,
    /// Analytical fallback (`FSR 3.1.6`) when no FSR 4 network accepts the device.
    Fsr316Fallback,
}

impl Fsr4Network {
    pub const fn amd_name(self) -> &'static str {
        match self {
            Self::Fsr4Fp8 => "FSR4",
            Self::Fsr4Int8 => "FSR4-i8",
            Self::Fsr4Int8Navi2 => "FSR4-i8-2",
            Self::Fsr4LiteInt8 => "FSR4L-i8",
            Self::Fsr316Fallback => "FSR 3.1.6",
        }
    }

    pub const fn is_ml_fsr4(self) -> bool {
        !matches!(self, Self::Fsr316Fallback)
    }

    pub const fn version_id(self) -> u64 {
        if self.is_ml_fsr4() {
            0xF5A5_CA1E_0100_1002
        } else {
            0xF5A5_CA1E_0003_0106
        }
    }
}

impl fmt::Display for Fsr4Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.amd_name())
    }
}

/// Primary upscaler recommendation for a given GPU profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecommendedUpscaler {
    Dlss4Transformer,
    Fsr4Ml,
    XessXmx,
    Fsr316Analytical,
}

/// Complete modeled hardware + driver profile for hardware-free verification.
#[derive(Debug, Clone, Serialize)]
pub struct GpuProfile {
    pub id: GpuId,
    pub driver: DriverModel,
    pub display_name: &'static str,
    pub driver_name: &'static str,
    pub api_version: u32,
    pub driver_version: u32,
    pub asic: AsicIdentity,
    pub vram_mib: u32,
    pub subgroup_size: u32,
    pub min_subgroup_size: u32,
    pub max_subgroup_size: u32,
    pub max_compute_workgroup_invocations: u32,
    pub max_compute_shared_memory_size: u32,
    pub max_bound_descriptor_sets: u32,
    pub features: FeatureFlags,
    pub extensions: &'static [&'static str],
    pub coop_matrix_shapes: &'static [CoopMatrixShape],
}

const COOP_RDNA4: &[CoopMatrixShape] = &[
    CoopMatrixShape {
        m: 16,
        n: 16,
        k: 16,
        a_type: CoopComponentType::Float8E4M3,
        b_type: CoopComponentType::Float8E4M3,
        c_type: CoopComponentType::Float32,
        result_type: CoopComponentType::Float32,
        scope: CoopScope::Subgroup,
    },
    CoopMatrixShape {
        m: 16,
        n: 16,
        k: 16,
        a_type: CoopComponentType::Float8E5M2,
        b_type: CoopComponentType::Float8E5M2,
        c_type: CoopComponentType::Float32,
        result_type: CoopComponentType::Float32,
        scope: CoopScope::Subgroup,
    },
    CoopMatrixShape {
        m: 16,
        n: 16,
        k: 16,
        a_type: CoopComponentType::Float16,
        b_type: CoopComponentType::Float16,
        c_type: CoopComponentType::Float32,
        result_type: CoopComponentType::Float32,
        scope: CoopScope::Subgroup,
    },
    CoopMatrixShape {
        m: 16,
        n: 16,
        k: 16,
        a_type: CoopComponentType::Int8,
        b_type: CoopComponentType::Int8,
        c_type: CoopComponentType::Int32,
        result_type: CoopComponentType::Int32,
        scope: CoopScope::Subgroup,
    },
];

const COOP_RDNA3: &[CoopMatrixShape] = &[
    CoopMatrixShape {
        m: 16,
        n: 16,
        k: 16,
        a_type: CoopComponentType::Float16,
        b_type: CoopComponentType::Float16,
        c_type: CoopComponentType::Float32,
        result_type: CoopComponentType::Float32,
        scope: CoopScope::Subgroup,
    },
    CoopMatrixShape {
        m: 16,
        n: 16,
        k: 16,
        a_type: CoopComponentType::Int8,
        b_type: CoopComponentType::Int8,
        c_type: CoopComponentType::Int32,
        result_type: CoopComponentType::Int32,
        scope: CoopScope::Subgroup,
    },
];

const COOP_NV_FP8: &[CoopMatrixShape] = &[
    CoopMatrixShape {
        m: 16,
        n: 16,
        k: 16,
        a_type: CoopComponentType::Float8E4M3,
        b_type: CoopComponentType::Float8E4M3,
        c_type: CoopComponentType::Float32,
        result_type: CoopComponentType::Float32,
        scope: CoopScope::Subgroup,
    },
    CoopMatrixShape {
        m: 16,
        n: 16,
        k: 16,
        a_type: CoopComponentType::Float8E5M2,
        b_type: CoopComponentType::Float8E5M2,
        c_type: CoopComponentType::Float32,
        result_type: CoopComponentType::Float32,
        scope: CoopScope::Subgroup,
    },
    CoopMatrixShape {
        m: 16,
        n: 16,
        k: 16,
        a_type: CoopComponentType::Float16,
        b_type: CoopComponentType::Float16,
        c_type: CoopComponentType::Float32,
        result_type: CoopComponentType::Float32,
        scope: CoopScope::Subgroup,
    },
    CoopMatrixShape {
        m: 16,
        n: 16,
        k: 16,
        a_type: CoopComponentType::Int8,
        b_type: CoopComponentType::Int8,
        c_type: CoopComponentType::Int32,
        result_type: CoopComponentType::Int32,
        scope: CoopScope::Subgroup,
    },
];

const COOP_AMPERE: &[CoopMatrixShape] = &[
    CoopMatrixShape {
        m: 16,
        n: 16,
        k: 16,
        a_type: CoopComponentType::Float16,
        b_type: CoopComponentType::Float16,
        c_type: CoopComponentType::Float32,
        result_type: CoopComponentType::Float32,
        scope: CoopScope::Subgroup,
    },
    CoopMatrixShape {
        m: 16,
        n: 16,
        k: 16,
        a_type: CoopComponentType::Int8,
        b_type: CoopComponentType::Int8,
        c_type: CoopComponentType::Int32,
        result_type: CoopComponentType::Int32,
        scope: CoopScope::Subgroup,
    },
];

const COOP_INTEL_XE2: &[CoopMatrixShape] = &[
    CoopMatrixShape {
        m: 8,
        n: 16,
        k: 16,
        a_type: CoopComponentType::Float16,
        b_type: CoopComponentType::Float16,
        c_type: CoopComponentType::Float32,
        result_type: CoopComponentType::Float32,
        scope: CoopScope::Subgroup,
    },
    CoopMatrixShape {
        m: 8,
        n: 16,
        k: 32,
        a_type: CoopComponentType::Int8,
        b_type: CoopComponentType::Int8,
        c_type: CoopComponentType::Int32,
        result_type: CoopComponentType::Int32,
        scope: CoopScope::Subgroup,
    },
];

const EXT_RDNA4: &[&str] = &[
    "VK_KHR_swapchain",
    "VK_KHR_cooperative_matrix",
    "VK_EXT_shader_float8",
    "VK_KHR_compute_shader_derivatives",
    "VK_KHR_shader_float16_int8",
    "VK_KHR_16bit_storage",
    "VK_KHR_8bit_storage",
    "VK_KHR_shader_integer_dot_product",
    "VK_KHR_synchronization2",
    "VK_KHR_dynamic_rendering",
    "VK_KHR_buffer_device_address",
    "VK_KHR_acceleration_structure",
    "VK_KHR_ray_tracing_pipeline",
    "VK_KHR_ray_query",
    "VK_EXT_subgroup_size_control",
    "VK_EXT_descriptor_indexing",
];

const EXT_RDNA3: &[&str] = &[
    "VK_KHR_swapchain",
    "VK_KHR_cooperative_matrix",
    "VK_KHR_compute_shader_derivatives",
    "VK_KHR_shader_float16_int8",
    "VK_KHR_16bit_storage",
    "VK_KHR_8bit_storage",
    "VK_KHR_shader_integer_dot_product",
    "VK_KHR_synchronization2",
    "VK_KHR_dynamic_rendering",
    "VK_KHR_buffer_device_address",
    "VK_KHR_acceleration_structure",
    "VK_KHR_ray_tracing_pipeline",
    "VK_KHR_ray_query",
    "VK_EXT_subgroup_size_control",
    "VK_EXT_descriptor_indexing",
];

const EXT_RDNA2: &[&str] = &[
    "VK_KHR_swapchain",
    "VK_KHR_compute_shader_derivatives",
    "VK_KHR_shader_float16_int8",
    "VK_KHR_16bit_storage",
    "VK_KHR_8bit_storage",
    "VK_KHR_shader_integer_dot_product",
    "VK_KHR_synchronization2",
    "VK_KHR_dynamic_rendering",
    "VK_KHR_buffer_device_address",
    "VK_KHR_acceleration_structure",
    "VK_KHR_ray_tracing_pipeline",
    "VK_KHR_ray_query",
    "VK_EXT_subgroup_size_control",
    "VK_EXT_descriptor_indexing",
];

const EXT_NV_BLACKWELL: &[&str] = &[
    "VK_KHR_swapchain",
    "VK_KHR_cooperative_matrix",
    "VK_EXT_shader_float8",
    "VK_KHR_compute_shader_derivatives",
    "VK_KHR_shader_float16_int8",
    "VK_KHR_16bit_storage",
    "VK_KHR_8bit_storage",
    "VK_KHR_shader_integer_dot_product",
    "VK_KHR_synchronization2",
    "VK_KHR_dynamic_rendering",
    "VK_KHR_buffer_device_address",
    "VK_KHR_acceleration_structure",
    "VK_KHR_ray_tracing_pipeline",
    "VK_KHR_ray_query",
    "VK_NV_ray_tracing_invocation_reorder",
    "VK_EXT_ray_tracing_invocation_reorder",
    "VK_EXT_opacity_micromap",
    "VK_EXT_subgroup_size_control",
    "VK_EXT_descriptor_indexing",
];

const EXT_NV_ADA: &[&str] = &[
    "VK_KHR_swapchain",
    "VK_KHR_cooperative_matrix",
    "VK_EXT_shader_float8",
    "VK_KHR_compute_shader_derivatives",
    "VK_KHR_shader_float16_int8",
    "VK_KHR_16bit_storage",
    "VK_KHR_8bit_storage",
    "VK_KHR_shader_integer_dot_product",
    "VK_KHR_synchronization2",
    "VK_KHR_dynamic_rendering",
    "VK_KHR_buffer_device_address",
    "VK_KHR_acceleration_structure",
    "VK_KHR_ray_tracing_pipeline",
    "VK_KHR_ray_query",
    "VK_NV_ray_tracing_invocation_reorder",
    "VK_EXT_opacity_micromap",
    "VK_EXT_subgroup_size_control",
    "VK_EXT_descriptor_indexing",
];

const EXT_NV_AMPERE: &[&str] = &[
    "VK_KHR_swapchain",
    "VK_KHR_cooperative_matrix",
    "VK_KHR_compute_shader_derivatives",
    "VK_KHR_shader_float16_int8",
    "VK_KHR_16bit_storage",
    "VK_KHR_8bit_storage",
    "VK_KHR_shader_integer_dot_product",
    "VK_KHR_synchronization2",
    "VK_KHR_dynamic_rendering",
    "VK_KHR_buffer_device_address",
    "VK_KHR_acceleration_structure",
    "VK_KHR_ray_tracing_pipeline",
    "VK_KHR_ray_query",
    "VK_EXT_subgroup_size_control",
    "VK_EXT_descriptor_indexing",
];

const EXT_INTEL_XE2: &[&str] = &[
    "VK_KHR_swapchain",
    "VK_KHR_cooperative_matrix",
    "VK_KHR_compute_shader_derivatives",
    "VK_KHR_shader_float16_int8",
    "VK_KHR_16bit_storage",
    "VK_KHR_8bit_storage",
    "VK_KHR_shader_integer_dot_product",
    "VK_KHR_synchronization2",
    "VK_KHR_dynamic_rendering",
    "VK_KHR_buffer_device_address",
    "VK_KHR_acceleration_structure",
    "VK_KHR_ray_tracing_pipeline",
    "VK_KHR_ray_query",
    "VK_EXT_subgroup_size_control",
    "VK_EXT_descriptor_indexing",
];

impl GpuProfile {
    /// Builds the canonical hardware + driver profile for `(gpu, driver)`.
    pub fn build(id: GpuId, driver: DriverModel) -> Self {
        let common_compute = FeatureFlags::SHADER_FLOAT16
            | FeatureFlags::SHADER_INT8
            | FeatureFlags::INT8_DOT_PRODUCT
            | FeatureFlags::COMPUTE_DERIVATIVES_LINEAR
            | FeatureFlags::SUBGROUP_SIZE_CONTROL
            | FeatureFlags::STORAGE_IMAGE_WRITE_NO_FMT
            | FeatureFlags::SYNC2
            | FeatureFlags::DYNAMIC_RENDERING
            | FeatureFlags::RT_PIPELINE
            | FeatureFlags::RAY_QUERY;

        match id {
            GpuId::Rx9070Xt => Self {
                id,
                driver,
                display_name: "AMD Radeon RX 9070 XT (RDNA 4 / GFX1201)",
                driver_name: match driver {
                    DriverModel::Linux => "Mesa RADV 26.2.6 (ACO)",
                    DriverModel::Windows => "AMD Adrenalin 26.10.1",
                },
                api_version: vk::make_api_version(0, 1, 4, 341),
                driver_version: match driver {
                    DriverModel::Linux => vk::make_api_version(0, 26, 2, 6),
                    DriverModel::Windows => vk::make_api_version(0, 26, 10, 1),
                },
                asic: AsicIdentity {
                    vendor_id: VENDOR_AMD,
                    device_id: 0x7550,
                    is_integrated: false,
                    amd_family: Some(0x98),
                    amd_revision: Some(0xC0),
                    gfx_ip: "gfx1201",
                },
                vram_mib: 16384,
                subgroup_size: 32,
                min_subgroup_size: 32,
                max_subgroup_size: 64,
                max_compute_workgroup_invocations: 1024,
                max_compute_shared_memory_size: 65536,
                max_bound_descriptor_sets: 32,
                features: common_compute
                    | FeatureFlags::COOP_MATRIX_KHR
                    | FeatureFlags::SHADER_FLOAT8
                    | FeatureFlags::SHADER_FLOAT8_COOP_MATRIX,
                extensions: EXT_RDNA4,
                coop_matrix_shapes: COOP_RDNA4,
            },
            GpuId::Rx7900Xtx => Self {
                id,
                driver,
                display_name: "AMD Radeon RX 7900 XTX (RDNA 3 / GFX1100)",
                driver_name: match driver {
                    DriverModel::Linux => "Mesa RADV 26.2.6 (ACO)",
                    DriverModel::Windows => "AMD Adrenalin 26.10.1",
                },
                api_version: vk::make_api_version(0, 1, 4, 341),
                driver_version: match driver {
                    DriverModel::Linux => vk::make_api_version(0, 26, 2, 6),
                    DriverModel::Windows => vk::make_api_version(0, 26, 10, 1),
                },
                asic: AsicIdentity {
                    vendor_id: VENDOR_AMD,
                    device_id: 0x744C,
                    is_integrated: false,
                    amd_family: Some(0x91),
                    amd_revision: Some(0x01),
                    gfx_ip: "gfx1100",
                },
                vram_mib: 24576,
                subgroup_size: 32,
                min_subgroup_size: 32,
                max_subgroup_size: 64,
                max_compute_workgroup_invocations: 1024,
                max_compute_shared_memory_size: 65536,
                max_bound_descriptor_sets: 32,
                features: common_compute | FeatureFlags::COOP_MATRIX_KHR,
                extensions: EXT_RDNA3,
                coop_matrix_shapes: COOP_RDNA3,
            },
            GpuId::Rx6800Xt => Self {
                id,
                driver,
                display_name: "AMD Radeon RX 6800 XT (RDNA 2 / GFX1030)",
                driver_name: match driver {
                    DriverModel::Linux => "Mesa RADV 26.2.6 (ACO)",
                    DriverModel::Windows => "AMD Adrenalin 26.10.1",
                },
                api_version: vk::make_api_version(0, 1, 3, 295),
                driver_version: match driver {
                    DriverModel::Linux => vk::make_api_version(0, 26, 2, 6),
                    DriverModel::Windows => vk::make_api_version(0, 26, 10, 1),
                },
                asic: AsicIdentity {
                    vendor_id: VENDOR_AMD,
                    device_id: 0x73BF,
                    is_integrated: false,
                    amd_family: Some(0x8F),
                    amd_revision: Some(0x28),
                    gfx_ip: "gfx1030",
                },
                vram_mib: 16384,
                subgroup_size: 32,
                min_subgroup_size: 32,
                max_subgroup_size: 64,
                max_compute_workgroup_invocations: 1024,
                max_compute_shared_memory_size: 65536,
                max_bound_descriptor_sets: 32,
                features: common_compute,
                extensions: EXT_RDNA2,
                coop_matrix_shapes: &[],
            },
            GpuId::StrixPoint => Self {
                id,
                driver,
                display_name: "AMD Radeon 890M Strix Point iGPU (RDNA 3.5 / GFX1150)",
                driver_name: match driver {
                    DriverModel::Linux => "Mesa RADV 26.2.6 (ACO)",
                    DriverModel::Windows => "AMD Adrenalin 26.10.1",
                },
                api_version: vk::make_api_version(0, 1, 4, 341),
                driver_version: match driver {
                    DriverModel::Linux => vk::make_api_version(0, 26, 2, 6),
                    DriverModel::Windows => vk::make_api_version(0, 26, 10, 1),
                },
                asic: AsicIdentity {
                    vendor_id: VENDOR_AMD,
                    device_id: 0x150E,
                    is_integrated: true,
                    amd_family: Some(0x96),
                    amd_revision: Some(0x01),
                    gfx_ip: "gfx1150",
                },
                vram_mib: 4096,
                subgroup_size: 32,
                min_subgroup_size: 32,
                max_subgroup_size: 64,
                max_compute_workgroup_invocations: 1024,
                max_compute_shared_memory_size: 65536,
                max_bound_descriptor_sets: 32,
                features: common_compute | FeatureFlags::COOP_MATRIX_KHR,
                extensions: EXT_RDNA3,
                coop_matrix_shapes: COOP_RDNA3,
            },
            GpuId::Rtx5090 => Self {
                id,
                driver,
                display_name: "NVIDIA GeForce RTX 5090 (Blackwell / GB202)",
                driver_name: match driver {
                    DriverModel::Linux => "NVIDIA Linux 580.95",
                    DriverModel::Windows => "NVIDIA Game Ready 580.95",
                },
                api_version: vk::make_api_version(0, 1, 4, 341),
                driver_version: vk::make_api_version(0, 580, 95, 0),
                asic: AsicIdentity {
                    vendor_id: VENDOR_NVIDIA,
                    device_id: 0x2B85,
                    is_integrated: false,
                    amd_family: None,
                    amd_revision: None,
                    gfx_ip: "gb202",
                },
                vram_mib: 32768,
                subgroup_size: 32,
                min_subgroup_size: 32,
                max_subgroup_size: 32,
                max_compute_workgroup_invocations: 1024,
                max_compute_shared_memory_size: 49152,
                max_bound_descriptor_sets: 32,
                features: common_compute
                    | FeatureFlags::COOP_MATRIX_KHR
                    | FeatureFlags::SHADER_FLOAT8
                    | FeatureFlags::SHADER_FLOAT8_COOP_MATRIX
                    | FeatureFlags::SER_NV
                    | FeatureFlags::SER_EXT
                    | FeatureFlags::OMM_EXT
                    | FeatureFlags::NGX_DLSS4,
                extensions: EXT_NV_BLACKWELL,
                coop_matrix_shapes: COOP_NV_FP8,
            },
            GpuId::Rtx4090 => Self {
                id,
                driver,
                display_name: "NVIDIA GeForce RTX 4090 (Ada Lovelace / AD102)",
                driver_name: match driver {
                    DriverModel::Linux => "NVIDIA Linux 580.95",
                    DriverModel::Windows => "NVIDIA Game Ready 580.95",
                },
                api_version: vk::make_api_version(0, 1, 4, 341),
                driver_version: vk::make_api_version(0, 580, 95, 0),
                asic: AsicIdentity {
                    vendor_id: VENDOR_NVIDIA,
                    device_id: 0x2684,
                    is_integrated: false,
                    amd_family: None,
                    amd_revision: None,
                    gfx_ip: "ad102",
                },
                vram_mib: 24576,
                subgroup_size: 32,
                min_subgroup_size: 32,
                max_subgroup_size: 32,
                max_compute_workgroup_invocations: 1024,
                max_compute_shared_memory_size: 49152,
                max_bound_descriptor_sets: 32,
                features: common_compute
                    | FeatureFlags::COOP_MATRIX_KHR
                    | FeatureFlags::SHADER_FLOAT8
                    | FeatureFlags::SHADER_FLOAT8_COOP_MATRIX
                    | FeatureFlags::SER_NV
                    | FeatureFlags::OMM_EXT
                    | FeatureFlags::NGX_DLSS4,
                extensions: EXT_NV_ADA,
                coop_matrix_shapes: COOP_NV_FP8,
            },
            GpuId::Rtx3050Mobile => Self {
                id,
                driver,
                display_name: "NVIDIA GeForce RTX 3050 Mobile (Ampere / GA107M)",
                driver_name: match driver {
                    DriverModel::Linux => "NVIDIA Linux 580.95",
                    DriverModel::Windows => "NVIDIA Game Ready 580.95",
                },
                api_version: vk::make_api_version(0, 1, 3, 295),
                driver_version: vk::make_api_version(0, 580, 95, 0),
                asic: AsicIdentity {
                    vendor_id: VENDOR_NVIDIA,
                    device_id: 0x25A2,
                    is_integrated: false,
                    amd_family: None,
                    amd_revision: None,
                    gfx_ip: "ga107m",
                },
                vram_mib: 4096,
                subgroup_size: 32,
                min_subgroup_size: 32,
                max_subgroup_size: 32,
                max_compute_workgroup_invocations: 1024,
                max_compute_shared_memory_size: 49152,
                max_bound_descriptor_sets: 32,
                features: common_compute
                    | FeatureFlags::COOP_MATRIX_KHR
                    | FeatureFlags::NGX_DLSS4,
                extensions: EXT_NV_AMPERE,
                coop_matrix_shapes: COOP_AMPERE,
            },
            GpuId::ArcB580 => Self {
                id,
                driver,
                display_name: "Intel Arc B580 (Battlemage / Xe2-G21)",
                driver_name: match driver {
                    DriverModel::Linux => "Mesa ANV 26.2.6",
                    DriverModel::Windows => "Intel Arc 32.0.101",
                },
                api_version: vk::make_api_version(0, 1, 3, 295),
                driver_version: match driver {
                    DriverModel::Linux => vk::make_api_version(0, 26, 2, 6),
                    DriverModel::Windows => vk::make_api_version(0, 32, 0, 101),
                },
                asic: AsicIdentity {
                    vendor_id: VENDOR_INTEL,
                    device_id: 0xE20B,
                    is_integrated: false,
                    amd_family: None,
                    amd_revision: None,
                    gfx_ip: "xe2-g21",
                },
                vram_mib: 12288,
                subgroup_size: 16,
                min_subgroup_size: 16,
                max_subgroup_size: 32,
                max_compute_workgroup_invocations: 1024,
                max_compute_shared_memory_size: 65536,
                max_bound_descriptor_sets: 32,
                features: common_compute
                    | FeatureFlags::COOP_MATRIX_KHR
                    | FeatureFlags::XESS_XMX,
                extensions: EXT_INTEL_XE2,
                coop_matrix_shapes: COOP_INTEL_XE2,
            },
        }
    }

    /// Returns all 16 `(GPU x OS)` profiles in deterministic order.
    pub fn all_matrix() -> Vec<Self> {
        let mut out = Vec::with_capacity(GpuId::ALL.len() * DriverModel::ALL.len());
        for &gpu in GpuId::ALL {
            for &driver in DriverModel::ALL {
                out.push(Self::build(gpu, driver));
            }
        }
        out
    }

    /// Checks whether the profile advertises a given Vulkan device extension.
    pub fn has_extension(&self, name: &str) -> bool {
        self.extensions.iter().any(|&ext| ext == name)
    }

    /// Checks whether the profile supports 16x16x16 FP8 cooperative matrix operations.
    pub fn supports_fp8_coop_16x16x16(&self) -> bool {
        self.features.contains(
            FeatureFlags::COOP_MATRIX_KHR
                | FeatureFlags::SHADER_FLOAT8
                | FeatureFlags::SHADER_FLOAT8_COOP_MATRIX,
        ) && self.coop_matrix_shapes.iter().any(|s| {
            s.m == 16
                && s.n == 16
                && s.k == 16
                && matches!(
                    s.a_type,
                    CoopComponentType::Float8E4M3 | CoopComponentType::Float8E5M2
                )
        })
    }

    /// Selects the active AMD FidelityFX SDK 2.4.0 FSR 4.1.2 network using the exact
    /// device and ASIC family/revision checks documented for `amd_fidelityfx_upscaler_vk`.
    pub fn select_fsr4_network(&self, override_req: Option<&str>) -> Result<Fsr4Network, HwsimError> {
        if let Some(req) = override_req {
            let norm = req.trim().to_ascii_lowercase();
            return match norm.as_str() {
                "fp8" | "fsr4" => {
                    if self.asic.vendor_id == VENDOR_AMD
                        && !self.asic.is_integrated
                        && self.supports_fp8_coop_16x16x16()
                    {
                        Ok(Fsr4Network::Fsr4Fp8)
                    } else {
                        Err(HwsimError::UnsupportedNetworkOverride {
                            requested: req.to_string(),
                            gpu: self.display_name.to_string(),
                            reason: "requires discrete AMD GPU with VK_KHR_cooperative_matrix + VK_EXT_shader_float8 (16x16x16 FP8)".to_string(),
                        })
                    }
                }
                "i8" | "fsr4-i8" => {
                    if self.asic.vendor_id == VENDOR_AMD
                        && self.features.contains(
                            FeatureFlags::INT8_DOT_PRODUCT
                                | FeatureFlags::SHADER_FLOAT16
                                | FeatureFlags::SHADER_INT8,
                        )
                    {
                        Ok(Fsr4Network::Fsr4Int8)
                    } else {
                        Err(HwsimError::UnsupportedNetworkOverride {
                            requested: req.to_string(),
                            gpu: self.display_name.to_string(),
                            reason: "requires AMD GPU with shaderIntegerDotProduct + shaderFloat16 + shaderInt8".to_string(),
                        })
                    }
                }
                "navi2" | "fsr4-i8-2" => {
                    if self.asic.vendor_id == VENDOR_AMD
                        && self
                            .features
                            .contains(FeatureFlags::INT8_DOT_PRODUCT | FeatureFlags::SHADER_FLOAT16)
                    {
                        Ok(Fsr4Network::Fsr4Int8Navi2)
                    } else {
                        Err(HwsimError::UnsupportedNetworkOverride {
                            requested: req.to_string(),
                            gpu: self.display_name.to_string(),
                            reason: "requires AMD GPU with shaderIntegerDotProduct + shaderFloat16"
                                .to_string(),
                        })
                    }
                }
                "lite" | "fsr4l-i8" => {
                    if self.asic.vendor_id == VENDOR_AMD
                        && self
                            .features
                            .contains(FeatureFlags::INT8_DOT_PRODUCT | FeatureFlags::SHADER_FLOAT16)
                    {
                        Ok(Fsr4Network::Fsr4LiteInt8)
                    } else {
                        Err(HwsimError::UnsupportedNetworkOverride {
                            requested: req.to_string(),
                            gpu: self.display_name.to_string(),
                            reason: "requires AMD GPU/APU with DP4a + FP16 support".to_string(),
                        })
                    }
                }
                "316" | "fsr3" | "fallback" => Ok(Fsr4Network::Fsr316Fallback),
                other => Err(HwsimError::UnsupportedNetworkOverride {
                    requested: other.to_string(),
                    gpu: self.display_name.to_string(),
                    reason: "expected one of: fp8, i8, navi2, lite, 316".to_string(),
                }),
            };
        }

        if self.asic.vendor_id != VENDOR_AMD {
            return Ok(Fsr4Network::Fsr316Fallback);
        }

        if !self.asic.is_integrated && self.supports_fp8_coop_16x16x16() {
            return Ok(Fsr4Network::Fsr4Fp8);
        }

        let has_i8_base = self.features.contains(
            FeatureFlags::INT8_DOT_PRODUCT
                | FeatureFlags::SHADER_FLOAT16
                | FeatureFlags::SUBGROUP_SIZE_CONTROL,
        );
        if !has_i8_base {
            return Ok(Fsr4Network::Fsr316Fallback);
        }

        match (self.asic.amd_family, self.asic.amd_revision) {
            // RDNA 3 (Navi 3x) or Strix Halo
            (Some(0x91), Some(0x01..=0xFE)) | (Some(0x96), Some(0xC0..=0xCF)) => {
                Ok(Fsr4Network::Fsr4Int8)
            }
            // RDNA 2 discrete (Navi 21..24)
            (Some(0x8F), Some(0x28..=0x4F)) => Ok(Fsr4Network::Fsr4Int8Navi2),
            // Van Gogh (Steam Deck) or Strix Point (RDNA 3.5 APU)
            (Some(0x90), Some(0x01..=0x7F)) | (Some(0x96), Some(0x01..=0x3F)) => {
                Ok(Fsr4Network::Fsr4LiteInt8)
            }
            _ => {
                if self.asic.device_id == 0x7481 {
                    Ok(Fsr4Network::Fsr4Int8)
                } else {
                    Ok(Fsr4Network::Fsr316Fallback)
                }
            }
        }
    }

    /// Resolves the highest-tier primary upscaler for this GPU profile.
    pub fn recommended_upscaler(&self) -> RecommendedUpscaler {
        if self.features.contains(FeatureFlags::NGX_DLSS4) {
            RecommendedUpscaler::Dlss4Transformer
        } else if self
            .select_fsr4_network(None)
            .is_ok_and(Fsr4Network::is_ml_fsr4)
        {
            RecommendedUpscaler::Fsr4Ml
        } else if self.features.contains(FeatureFlags::XESS_XMX) {
            RecommendedUpscaler::XessXmx
        } else {
            RecommendedUpscaler::Fsr316Analytical
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsr4_network_selection_matches_sdk_2_4_matrix() {
        assert_eq!(
            GpuProfile::build(GpuId::Rx9070Xt, DriverModel::Linux)
                .select_fsr4_network(None)
                .unwrap(),
            Fsr4Network::Fsr4Fp8
        );
        assert_eq!(
            GpuProfile::build(GpuId::Rx7900Xtx, DriverModel::Windows)
                .select_fsr4_network(None)
                .unwrap(),
            Fsr4Network::Fsr4Int8
        );
        assert_eq!(
            GpuProfile::build(GpuId::Rx6800Xt, DriverModel::Linux)
                .select_fsr4_network(None)
                .unwrap(),
            Fsr4Network::Fsr4Int8Navi2
        );
        assert_eq!(
            GpuProfile::build(GpuId::StrixPoint, DriverModel::Linux)
                .select_fsr4_network(None)
                .unwrap(),
            Fsr4Network::Fsr4LiteInt8
        );
        assert_eq!(
            GpuProfile::build(GpuId::Rtx5090, DriverModel::Linux)
                .select_fsr4_network(None)
                .unwrap(),
            Fsr4Network::Fsr316Fallback
        );
        assert_eq!(
            GpuProfile::build(GpuId::ArcB580, DriverModel::Windows)
                .select_fsr4_network(None)
                .unwrap(),
            Fsr4Network::Fsr316Fallback
        );
    }

    #[test]
    fn fp8_override_rejected_on_non_rdna4_hardware() {
        let rdna3 = GpuProfile::build(GpuId::Rx7900Xtx, DriverModel::Linux);
        assert!(rdna3.select_fsr4_network(Some("fp8")).is_err());

        let rtx3050m = GpuProfile::build(GpuId::Rtx3050Mobile, DriverModel::Linux);
        assert!(rtx3050m.select_fsr4_network(Some("fp8")).is_err());
    }
}
