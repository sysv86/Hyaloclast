use crate::VspError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

/// Hardware & algorithmic rendering tier supported by Hyaloclast (`IGNIS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum RenderTier {
    /// Tier 0: Pure raster G-Buffer + Cascaded Shadow Maps (CSM) + Hi-Z SSR + GTAO + Volumetric Fog.
    Tier0Raster = 0,
    /// Tier 1: Raster G-Buffer + Compute Froxel GI / Screen-Space Radiance Cache + Volumetric Fog.
    Tier1ComputeGi = 1,
    /// Tier 2: Hybrid Raster G-Buffer + inline `VK_KHR_ray_query` shadows, reflections, GI, and ReSTIR DI over `UberGpuBuffer` BLAS/TLAS.
    Tier2RayQuery = 2,
    /// Tier 3: Full Path Tracing (`VK_KHR_ray_tracing_pipeline` / wavefront compute PT) with ReSTIR DI + SHaRC + SVGF/NRD + `depth_blit` overlay reprojection.
    Tier3PathTracing = 3,
}

impl RenderTier {
    pub fn from_index(idx: u32) -> Option<Self> {
        match idx {
            0 => Some(Self::Tier0Raster),
            1 => Some(Self::Tier1ComputeGi),
            2 => Some(Self::Tier2RayQuery),
            3 => Some(Self::Tier3PathTracing),
            _ => None,
        }
    }

    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    /// Numeric tier index (0..=3). Convenience counterpart to `from_index` used by the render
    /// graph's tier-range filtering of pack-declared passes.
    pub const fn index(self) -> u32 {
        self as u32
    }
}

/// All 16 composable `.vsl` hook stages in the VulkanShaders v2 (`VSP Graph v2`) specification.
/// Fully backwards-compatible with all 10 v1 hook stages while adding fine-grained hooks for
/// custom G-Buffer packing, emissive triangle/point-light sampling, ReSTIR DI reservoir weighting,
/// multi-bounce indirect GI, 3D volumetric participating media, and AO/denoiser guidance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookStage {
    VertexDisplace,
    GbufferPack,
    Surface,
    EmissiveSample,
    RestirReservoir,
    DirectLight,
    BlockLight,
    IndirectGi,
    WaterWaves,
    WaterSurface,
    Sky,
    Clouds,
    VolumetricMedia,
    Fog,
    AoFilter,
    PostGrade,
}

impl HookStage {
    pub const ALL: &'static [Self] = &[
        Self::VertexDisplace,
        Self::GbufferPack,
        Self::Surface,
        Self::EmissiveSample,
        Self::RestirReservoir,
        Self::DirectLight,
        Self::BlockLight,
        Self::IndirectGi,
        Self::WaterWaves,
        Self::WaterSurface,
        Self::Sky,
        Self::Clouds,
        Self::VolumetricMedia,
        Self::Fog,
        Self::AoFilter,
        Self::PostGrade,
    ];

    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim() {
            "vertex_displace" => Some(Self::VertexDisplace),
            "gbuffer_pack" => Some(Self::GbufferPack),
            "surface" => Some(Self::Surface),
            "emissive_sample" => Some(Self::EmissiveSample),
            "restir_reservoir" => Some(Self::RestirReservoir),
            "direct_light" => Some(Self::DirectLight),
            "block_light" => Some(Self::BlockLight),
            "indirect_gi" => Some(Self::IndirectGi),
            "water_waves" => Some(Self::WaterWaves),
            "water_surface" => Some(Self::WaterSurface),
            "sky" => Some(Self::Sky),
            "clouds" => Some(Self::Clouds),
            "volumetric_media" => Some(Self::VolumetricMedia),
            "fog" => Some(Self::Fog),
            "ao_filter" => Some(Self::AoFilter),
            "post_grade" => Some(Self::PostGrade),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::VertexDisplace => "vertex_displace",
            Self::GbufferPack => "gbuffer_pack",
            Self::Surface => "surface",
            Self::EmissiveSample => "emissive_sample",
            Self::RestirReservoir => "restir_reservoir",
            Self::DirectLight => "direct_light",
            Self::BlockLight => "block_light",
            Self::IndirectGi => "indirect_gi",
            Self::WaterWaves => "water_waves",
            Self::WaterSurface => "water_surface",
            Self::Sky => "sky",
            Self::Clouds => "clouds",
            Self::VolumetricMedia => "volumetric_media",
            Self::Fog => "fog",
            Self::AoFilter => "ao_filter",
            Self::PostGrade => "post_grade",
        }
    }

    pub const fn dispatch_function_name(self) -> &'static str {
        match self {
            Self::VertexDisplace => "vs_dispatch_hook_vertex_displace",
            Self::GbufferPack => "vs_dispatch_hook_gbuffer_pack",
            Self::Surface => "vs_dispatch_hook_surface",
            Self::EmissiveSample => "vs_dispatch_hook_emissive_sample",
            Self::RestirReservoir => "vs_dispatch_hook_restir_reservoir",
            Self::DirectLight => "vs_dispatch_hook_direct_light",
            Self::BlockLight => "vs_dispatch_hook_block_light",
            Self::IndirectGi => "vs_dispatch_hook_indirect_gi",
            Self::WaterWaves => "vs_dispatch_hook_water_waves",
            Self::WaterSurface => "vs_dispatch_hook_water_surface",
            Self::Sky => "vs_dispatch_hook_sky",
            Self::Clouds => "vs_dispatch_hook_clouds",
            Self::VolumetricMedia => "vs_dispatch_hook_volumetric_media",
            Self::Fog => "vs_dispatch_hook_fog",
            Self::AoFilter => "vs_dispatch_hook_ao_filter",
            Self::PostGrade => "vs_dispatch_hook_post_grade",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackHeader {
    pub id: String,
    pub name: String,
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default = "default_author")]
    pub author: String,
    #[serde(default)]
    pub description: String,
    /// Optional base pack ID that this `.vsp` extends (inherits unoverridden hooks, targets, passes, and settings).
    #[serde(default)]
    pub extends: Option<String>,
}

fn default_version() -> String {
    "0.1.0".to_owned()
}

fn default_author() -> String {
    "Unknown".to_owned()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackCapabilities {
    #[serde(default = "default_tiers")]
    pub tiers: Vec<u32>,
    #[serde(default)]
    pub pbr_labpbr: bool,
    #[serde(default)]
    pub hdr_output: bool,
    #[serde(default = "default_true")]
    pub restir_di: bool,
    #[serde(default = "default_true")]
    pub volumetric_clouds: bool,
    #[serde(default = "default_true")]
    pub volumetric_fog: bool,
    #[serde(default = "default_true")]
    pub distant_horizons_lod: bool,
}

impl Default for PackCapabilities {
    fn default() -> Self {
        Self {
            tiers: default_tiers(),
            pbr_labpbr: true,
            hdr_output: true,
            restir_di: true,
            volumetric_clouds: true,
            volumetric_fog: true,
            distant_horizons_lod: true,
        }
    }
}

fn default_tiers() -> Vec<u32> {
    vec![0, 1, 2, 3]
}

/// Execution mode for the programmable `VSP Graph v2` RenderGraph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PipelineGraphMode {
    /// Extends the canonical IGNIS DAG by splicing custom `[[passes]]`, `[targets.*]`, and
    /// `[buffers.*]` before, after, or in place of canonical nodes (`replaces = "..."`).
    #[default]
    Extend,
    /// Replaces the canonical DAG with a 100% pack-authored custom RenderGraph while retaining
    /// automatic `VK_KHR_synchronization2` barrier synthesis and backward-BFS Dead Pass Culling.
    Custom,
}

/// Programmable pipeline configuration section (`[pipeline]`) in `pack.toml`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PackPipelineConfig {
    #[serde(default)]
    pub mode: PipelineGraphMode,
    /// Number of G-Buffer MRT colour attachments (legacy `gbuffer_MRT_count` key in pack.toml).
    #[serde(default = "default_gbuffer_mrt_count", alias = "gbuffer_MRT_count")]
    pub gbuffer_mrt_count: u32,
    /// Shared `.vsl` module files automatically linked with namespace isolation across passes.
    #[serde(default)]
    pub modules: Vec<String>,
}

const fn default_gbuffer_mrt_count() -> u32 {
    4
}

/// Pack-declared transient, persistent, or temporal ping-pong image target (`[targets.<name>]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackTargetDecl {
    #[serde(default = "default_target_format")]
    pub format: String,
    #[serde(default = "default_scale_one")]
    pub scale: f32,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default = "default_u32_one")]
    pub layers: u32,
    #[serde(default = "default_u32_one")]
    pub mips: u32,
    /// Mip level count consumed by the native render graph's `CustomTargetSpec`.
    /// Kept in sync with `mips` for packs that only declare the legacy key.
    #[serde(default = "default_u32_one")]
    pub mip_levels: u32,
    /// Array layer count consumed by the native render graph's `CustomTargetSpec`.
    /// Kept in sync with `layers` for packs that only declare the legacy key.
    #[serde(default = "default_u32_one")]
    pub array_layers: u32,
    /// Optional `[r, g, b, a]` clear colour; `None` means the target is not explicitly cleared.
    #[serde(default)]
    pub clear_color: Option<[f32; 4]>,
    /// When `true`, preserves image layout across frames and exposes `<name>_history` for ping-pong.
    #[serde(default)]
    pub temporal: bool,
    /// Optional extra G-Buffer MRT color attachment index (`4..7`).
    #[serde(default)]
    pub gbuffer_attachment: Option<u32>,
}

impl Default for PackTargetDecl {
    fn default() -> Self {
        Self {
            format: default_target_format(),
            scale: 1.0,
            width: None,
            height: None,
            layers: 1,
            mips: 1,
            mip_levels: 1,
            array_layers: 1,
            clear_color: None,
            temporal: false,
            gbuffer_attachment: None,
        }
    }
}

fn default_target_format() -> String {
    "RGBA16F".to_owned()
}

const fn default_scale_one() -> f32 {
    1.0
}

const fn default_u32_one() -> u32 {
    1
}

/// Pack-declared GPU storage buffer (`[buffers.<name>]`) with optional Buffer Device Address support.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackBufferDecl {
    #[serde(default = "default_buffer_size")]
    pub size_bytes: u64,
    #[serde(default = "default_buffer_stride")]
    pub stride: u32,
    #[serde(default)]
    pub element_count: u32,
    #[serde(default = "default_true")]
    pub persistent: bool,
    #[serde(default)]
    pub clear_each_frame: bool,
    #[serde(default = "default_true")]
    pub bda: bool,
}

impl Default for PackBufferDecl {
    fn default() -> Self {
        Self {
            size_bytes: default_buffer_size(),
            stride: default_buffer_stride(),
            element_count: 0,
            persistent: true,
            clear_each_frame: false,
            bda: true,
        }
    }
}

const fn default_buffer_size() -> u64 {
    65_536
}

const fn default_buffer_stride() -> u32 {
    16
}

/// Kind of a pack-declared programmable RenderGraph pass (`[[passes]]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PackPassKind {
    #[default]
    Compute,
    Raster,
    RayTrace,
    Transfer,
}

fn default_dispatch_scale() -> [f32; 2] {
    [1.0, 1.0]
}

const fn default_max_tier() -> u32 {
    3
}

/// Pack-declared programmable RenderGraph pass (`[[passes]]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackPassDecl {
    #[serde(default)]
    pub name: String,
    /// Stable pass identifier used by the render graph for graph-node matching. Defaults to
    /// `name` when omitted in `pack.toml` (normalised in `parse_and_validate` / `load_from_dir`).
    #[serde(default)]
    pub id: String,
    /// Resolution scale of the pass. Defaults to 1.0 when omitted.
    #[serde(default = "default_scale_one")]
    pub scale: f32,
    #[serde(default)]
    pub kind: PackPassKind,
    #[serde(default)]
    pub shader: String,
    #[serde(default)]
    pub vertex_shader: Option<String>,
    #[serde(default)]
    pub fragment_shader: Option<String>,
    #[serde(default = "default_dispatch_scale")]
    pub dispatch_scale: [f32; 2],
    #[serde(default)]
    pub min_tier: u32,
    #[serde(default = "default_max_tier")]
    pub max_tier: u32,
    #[serde(default)]
    pub before: Option<String>,
    #[serde(default)]
    pub after: Option<String>,
    #[serde(default)]
    pub replaces: Option<String>,
    #[serde(default)]
    pub reads: Vec<String>,
    #[serde(default)]
    pub writes: Vec<String>,
    /// Explicit mapping of descriptor binding keys (`"0:1"`, `"1"`, or uniform variable name)
    /// to canonical or custom resource names.
    #[serde(default)]
    pub bindings: BTreeMap<String, String>,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SettingValueType {
    Bool,
    Int,
    Float,
    Enum,
    ColorRgb,
    Vec3,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SettingValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Vec3([f32; 3]),
}

impl SettingValue {
    pub const MAX_RGB24: u32 = 0x00FF_FFFF;

    /// Parses `#RRGGBB`, `#RGB`, `0xRRGGBB`, `RRGGBB`, integer `r,g,b` (`0..255`), or
    /// normalized float `r,g,b` (`0.0..1.0`) into a 24-bit `0x00RRGGBB` value.
    pub fn parse_color_rgb(raw: &str) -> Option<u32> {
        let s = raw.trim();
        if s.is_empty() {
            return None;
        }
        let hex_body = if let Some(stripped) = s.strip_prefix('#') {
            Some((stripped, true))
        } else if let Some(stripped) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
            Some((stripped, false))
        } else if !s.contains(',') {
            Some((s, false))
        } else {
            None
        };

        if let Some((hex, had_hash)) = hex_body {
            if hex.len() == 6 {
                return u32::from_str_radix(hex, 16)
                    .ok()
                    .map(|v| v & Self::MAX_RGB24);
            }
            if hex.len() == 3 && had_hash {
                let parsed = u32::from_str_radix(hex, 16).ok()?;
                let r = ((parsed >> 8) & 0xF) * 17;
                let g = ((parsed >> 4) & 0xF) * 17;
                let b = (parsed & 0xF) * 17;
                return Some((r << 16) | (g << 8) | b);
            }
            return None;
        }

        let parts: Vec<&str> = s.split(',').map(str::trim).collect();
        if parts.len() != 3 || parts.iter().any(|p| p.is_empty()) {
            return None;
        }
        let has_decimal = parts.iter().any(|p| p.contains('.'));
        let mut floats = [0.0f32; 3];
        for (i, part) in parts.iter().enumerate() {
            let v = part.parse::<f32>().ok()?;
            if !v.is_finite() {
                return None;
            }
            floats[i] = v;
        }
        let normalized = has_decimal && floats.iter().all(|&v| (0.0..=1.0).contains(&v));
        let (r, g, b) = if normalized {
            (
                (floats[0] * 255.0).round().clamp(0.0, 255.0) as u32,
                (floats[1] * 255.0).round().clamp(0.0, 255.0) as u32,
                (floats[2] * 255.0).round().clamp(0.0, 255.0) as u32,
            )
        } else {
            (
                floats[0].round().clamp(0.0, 255.0) as u32,
                floats[1].round().clamp(0.0, 255.0) as u32,
                floats[2].round().clamp(0.0, 255.0) as u32,
            )
        };
        Some((r << 16) | (g << 8) | b)
    }

    /// Packs a 24-bit `0x00RRGGBB` integer into a normalized `[0.0, 1.0]` `f32`
    /// matching `WorldStateCollector.packRgbAsFloat` in Kotlin.
    pub fn pack_rgb_as_f32(rgb24: u32) -> f32 {
        (rgb24 & Self::MAX_RGB24) as f32 / (Self::MAX_RGB24 as f32)
    }

    pub fn to_color_f32(&self) -> f32 {
        match self {
            Self::Str(s) => Self::parse_color_rgb(s)
                .map(Self::pack_rgb_as_f32)
                .or_else(|| s.parse::<f32>().ok())
                .unwrap_or(0.0),
            Self::Int(i) => Self::pack_rgb_as_f32((*i).max(0) as u32),
            Self::Float(f) => *f as f32,
            Self::Vec3(v) => {
                let normalized = v.iter().all(|&c| (0.0..=1.0).contains(&c));
                let (r, g, b) = if normalized {
                    (
                        (v[0] * 255.0).round().clamp(0.0, 255.0) as u32,
                        (v[1] * 255.0).round().clamp(0.0, 255.0) as u32,
                        (v[2] * 255.0).round().clamp(0.0, 255.0) as u32,
                    )
                } else {
                    (
                        v[0].round().clamp(0.0, 255.0) as u32,
                        v[1].round().clamp(0.0, 255.0) as u32,
                        v[2].round().clamp(0.0, 255.0) as u32,
                    )
                };
                Self::pack_rgb_as_f32((r << 16) | (g << 8) | b)
            }
            Self::Bool(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    pub fn to_f32(&self, options: &[String]) -> f32 {
        match self {
            Self::Bool(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Int(i) => *i as f32,
            Self::Float(f) => *f as f32,
            Self::Str(s) => {
                if !options.is_empty() {
                    options
                        .iter()
                        .position(|opt| opt.eq_ignore_ascii_case(s))
                        .unwrap_or(0) as f32
                } else if let Some(rgb) = Self::parse_color_rgb(s) {
                    Self::pack_rgb_as_f32(rgb)
                } else {
                    s.parse::<f32>().unwrap_or(0.0)
                }
            }
            Self::Vec3(v) => v[0],
        }
    }

    pub fn to_u32_bits(&self, options: &[String]) -> u32 {
        match self {
            Self::Bool(b) => u32::from(*b),
            Self::Int(i) => *i as u32,
            Self::Float(f) => (*f as f32).to_bits(),
            Self::Str(s) => {
                if !options.is_empty() {
                    options
                        .iter()
                        .position(|opt| opt.eq_ignore_ascii_case(s))
                        .unwrap_or(0) as u32
                } else if let Some(rgb) = Self::parse_color_rgb(s) {
                    Self::pack_rgb_as_f32(rgb).to_bits()
                } else {
                    s.parse::<f32>().map(f32::to_bits).unwrap_or(0)
                }
            }
            Self::Vec3(v) => v[0].to_bits(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackSettingDecl {
    pub label: String,
    #[serde(rename = "type")]
    pub value_type: SettingValueType,
    pub default: SettingValue,
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
    #[serde(default)]
    pub step: Option<f64>,
    #[serde(default)]
    pub options: Vec<String>,
    #[serde(default = "default_true")]
    pub runtime: bool,
    #[serde(default)]
    pub min_tier: u32,
    #[serde(default = "default_max_tier")]
    pub max_tier: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackPresetDecl {
    pub label: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub order: i32,
    #[serde(default)]
    pub min_tier: u32,
    #[serde(default = "default_max_tier")]
    pub max_tier: u32,
    #[serde(default)]
    pub values: BTreeMap<String, SettingValue>,
}

const fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackResourceDecl {
    pub path: String,
    #[serde(default = "default_resource_format")]
    pub format: String,
    #[serde(default)]
    pub filter: Option<String>,
    #[serde(default)]
    pub wrap: Option<String>,
}

fn default_resource_format() -> String {
    "RGBA8_UNORM".to_owned()
}

/// Parsed and validated `pack.toml` manifest supporting both v1 hooks and v2 programmable DAG
/// (`[pipeline]`, `[targets.*]`, `[buffers.*]`, `[[passes]]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackManifest {
    pub pack: PackHeader,
    #[serde(default)]
    pub capabilities: PackCapabilities,
    #[serde(default)]
    pub pipeline: PackPipelineConfig,
    #[serde(default)]
    pub hooks: BTreeMap<String, String>,
    #[serde(default)]
    pub settings: BTreeMap<String, PackSettingDecl>,
    #[serde(default)]
    pub presets: BTreeMap<String, PackPresetDecl>,
    #[serde(default)]
    pub resources: BTreeMap<String, PackResourceDecl>,
    #[serde(default)]
    pub targets: BTreeMap<String, PackTargetDecl>,
    #[serde(default)]
    pub buffers: BTreeMap<String, PackBufferDecl>,
    #[serde(default)]
    pub passes: Vec<PackPassDecl>,
}

/// Assignment of pack settings to either `WorldUBO.packParams` indices or SPIR-V specialization constants.
#[derive(Debug, Clone)]
pub struct PackParameterLayout {
    /// Maps runtime setting key -> scalar index inside `WorldUBO.packParams` (`0..40`).
    pub runtime_ubo_indices: BTreeMap<String, usize>,
    /// Maps compile-time specialization setting key -> `constant_id` (`100..`).
    pub specialization_constant_ids: BTreeMap<String, u32>,
}

impl PackManifest {
    pub const MAX_RUNTIME_PACK_PARAMS: usize = 40;
    pub const FIRST_SPEC_CONSTANT_ID: u32 = 100;

    pub fn parse_and_validate(toml_str: &str) -> Result<Self, VspError> {
        let raw_value: toml::Value = toml::from_str(toml_str)?;
        Self::validate_required_fields(&raw_value)?;
        let mut manifest: Self = toml::from_str(toml_str)?;
        manifest.normalize();
        manifest.validate()?;
        Ok(manifest)
    }

    /// Inspects the parsed TOML AST before typed deserialization so missing required fields
    /// produce explicit, field-level `ManifestValidation` errors instead of opaque serde errors.
    fn validate_required_fields(root: &toml::Value) -> Result<(), VspError> {
        let Some(root_table) = root.as_table() else {
            return Err(VspError::ManifestValidation {
                pack_id: "<missing>".to_owned(),
                reason: "pack.toml root must be a TOML table".to_owned(),
            });
        };

        let Some(pack_table) = root_table.get("pack").and_then(toml::Value::as_table) else {
            return Err(VspError::ManifestValidation {
                pack_id: "<missing>".to_owned(),
                reason: "Missing required table '[pack]' in pack.toml".to_owned(),
            });
        };

        let pack_id = match pack_table.get("id").and_then(toml::Value::as_str) {
            Some(id) if !id.trim().is_empty() => id.trim().to_owned(),
            Some(_) => {
                return Err(VspError::ManifestValidation {
                    pack_id: "<empty>".to_owned(),
                    reason: "Required field 'pack.id' must not be empty".to_owned(),
                });
            }
            None => {
                return Err(VspError::ManifestValidation {
                    pack_id: "<missing>".to_owned(),
                    reason: "Missing required field 'pack.id' in [pack]".to_owned(),
                });
            }
        };

        match pack_table.get("name").and_then(toml::Value::as_str) {
            Some(name) if !name.trim().is_empty() => {}
            Some(_) => {
                return Err(VspError::ManifestValidation {
                    pack_id,
                    reason: "Required field 'pack.name' must not be empty".to_owned(),
                });
            }
            None => {
                return Err(VspError::ManifestValidation {
                    pack_id,
                    reason: "Missing required field 'pack.name' in [pack]".to_owned(),
                });
            }
        }

        if let Some(settings_table) = root_table.get("settings").and_then(toml::Value::as_table) {
            for (key, val) in settings_table {
                let Some(st) = val.as_table() else {
                    return Err(VspError::ManifestValidation {
                        pack_id: pack_id.clone(),
                        reason: format!("Setting 'settings.{key}' must be a TOML table"),
                    });
                };
                for req in ["label", "type", "default"] {
                    if !st.contains_key(req) {
                        return Err(VspError::ManifestValidation {
                            pack_id: pack_id.clone(),
                            reason: format!(
                                "Missing required field 'settings.{key}.{req}' in [settings.{key}]"
                            ),
                        });
                    }
                }
            }
        }

        if let Some(presets_table) = root_table.get("presets").and_then(toml::Value::as_table) {
            for (preset_id, val) in presets_table {
                let Some(pt) = val.as_table() else {
                    return Err(VspError::ManifestValidation {
                        pack_id: pack_id.clone(),
                        reason: format!("Preset 'presets.{preset_id}' must be a TOML table"),
                    });
                };
                if !pt.contains_key("label") {
                    return Err(VspError::ManifestValidation {
                        pack_id: pack_id.clone(),
                        reason: format!(
                            "Missing required field 'presets.{preset_id}.label' in [presets.{preset_id}]"
                        ),
                    });
                }
            }
        }

        if let Some(res_table) = root_table.get("resources").and_then(toml::Value::as_table) {
            for (res_key, val) in res_table {
                let Some(rt) = val.as_table() else {
                    return Err(VspError::ManifestValidation {
                        pack_id: pack_id.clone(),
                        reason: format!("Resource 'resources.{res_key}' must be a TOML table"),
                    });
                };
                if !rt.contains_key("path") {
                    return Err(VspError::ManifestValidation {
                        pack_id: pack_id.clone(),
                        reason: format!(
                            "Missing required field 'resources.{res_key}.path' in [resources.{res_key}]"
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    /// Loads and validates a shader pack manifest from `<pack_dir>/pack.toml`. Fallback-free: a
    /// missing or invalid manifest is a real `VspError`, never a guessed/defaulted manifest.
    pub fn load_from_dir(pack_dir: &std::path::Path) -> Result<Self, VspError> {
        let manifest_path = pack_dir.join("pack.toml");
        let raw = std::fs::read_to_string(&manifest_path).map_err(|source| VspError::Io {
            path: manifest_path.display().to_string(),
            source,
        })?;
        let manifest = Self::parse_and_validate(&raw)?;
        if manifest
            .pack
            .extends
            .as_deref()
            .is_some_and(|s| !s.trim().is_empty())
        {
            let (merged, _) =
                crate::compiler::VspCompiler::resolve_pack_inheritance(pack_dir, &[])?;
            return Ok(merged);
        }
        Ok(manifest)
    }

    /// Fills fields defaulted from sibling keys that serde cannot express statically:
    /// pass `id` from `name`, `mip_levels`/`array_layers` from the legacy `mips`/`layers` keys,
    /// and clamps pass `scale` to a positive value.
    fn normalize(&mut self) {
        for target in self.targets.values_mut() {
            target.mip_levels = target.mip_levels.max(target.mips);
            target.array_layers = target.array_layers.max(target.layers);
        }
        for pass in &mut self.passes {
            // The newer VSP Graph v2 schema uses `id`; older packs used `name`. Keep both fields
            // populated so either consumer (native graph vs. Kotlin parser) sees a valid value.
            if pass.id.trim().is_empty() {
                pass.id = pass.name.clone();
            }
            if pass.name.trim().is_empty() {
                pass.name = pass.id.clone();
            }
            if !pass.scale.is_finite() || pass.scale <= 0.0 {
                pass.scale = 1.0;
            }
        }
    }

    pub fn validate(&self) -> Result<(), VspError> {
        let pack_id = self.pack.id.trim();
        if pack_id.is_empty() {
            return Err(VspError::ManifestValidation {
                pack_id: "<empty>".to_owned(),
                reason: "pack.id must not be empty".to_owned(),
            });
        }
        if !pack_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err(VspError::ManifestValidation {
                pack_id: pack_id.to_owned(),
                reason: "pack.id may only contain ASCII alphanumeric, '-', or '_' characters"
                    .to_owned(),
            });
        }

        if self.pack.name.trim().is_empty() {
            return Err(VspError::ManifestValidation {
                pack_id: pack_id.to_owned(),
                reason: "pack.name must not be empty".to_owned(),
            });
        }

        if self.capabilities.tiers.is_empty() {
            return Err(VspError::ManifestValidation {
                pack_id: pack_id.to_owned(),
                reason: "capabilities.tiers must list at least one supported tier (0..=3)"
                    .to_owned(),
            });
        }

        for &tier in &self.capabilities.tiers {
            if tier > 3 {
                return Err(VspError::ManifestValidation {
                    pack_id: pack_id.to_owned(),
                    reason: format!("Unsupported tier index {tier}; valid tiers are 0, 1, 2, 3"),
                });
            }
        }

        for (stage_name, hook_path) in &self.hooks {
            if HookStage::from_name(stage_name).is_none() {
                return Err(VspError::ManifestValidation {
                    pack_id: pack_id.to_owned(),
                    reason: format!("Unknown hook stage '{stage_name}' in [hooks]"),
                });
            }
            if hook_path.trim().is_empty() || hook_path.contains("..") {
                return Err(VspError::ManifestValidation {
                    pack_id: pack_id.to_owned(),
                    reason: format!("Invalid hook path '{hook_path}' for stage '{stage_name}'"),
                });
            }
        }

        for mod_path in &self.pipeline.modules {
            if mod_path.trim().is_empty() || mod_path.contains("..") {
                return Err(VspError::ManifestValidation {
                    pack_id: pack_id.to_owned(),
                    reason: format!("Invalid module path '{mod_path}' in [pipeline.modules]"),
                });
            }
        }

        for (target_name, target_decl) in &self.targets {
            if target_name.trim().is_empty()
                || !target_name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                return Err(VspError::ManifestValidation {
                    pack_id: pack_id.to_owned(),
                    reason: format!("Invalid custom target name '{target_name}' in [targets]"),
                });
            }
            if !target_decl.scale.is_finite() || target_decl.scale <= 0.0 || target_decl.scale > 4.0
            {
                return Err(VspError::ManifestValidation {
                    pack_id: pack_id.to_owned(),
                    reason: format!(
                        "Custom target '{target_name}' scale ({}) must be in (0.0, 4.0]",
                        target_decl.scale
                    ),
                });
            }
            if let Some(att) = target_decl.gbuffer_attachment
                && !(4..=7).contains(&att)
            {
                return Err(VspError::ManifestValidation {
                    pack_id: pack_id.to_owned(),
                    reason: format!(
                        "Custom target '{target_name}' gbuffer_attachment ({att}) must be in 4..=7"
                    ),
                });
            }
        }

        for (buf_name, buf_decl) in &self.buffers {
            if buf_name.trim().is_empty()
                || !buf_name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                return Err(VspError::ManifestValidation {
                    pack_id: pack_id.to_owned(),
                    reason: format!("Invalid custom buffer name '{buf_name}' in [buffers]"),
                });
            }
            let effective_bytes = if buf_decl.element_count > 0 && buf_decl.stride > 0 {
                u64::from(buf_decl.element_count) * u64::from(buf_decl.stride)
            } else {
                buf_decl.size_bytes
            };
            if effective_bytes == 0 || effective_bytes > 512 * 1024 * 1024 {
                return Err(VspError::ManifestValidation {
                    pack_id: pack_id.to_owned(),
                    reason: format!(
                        "Custom buffer '{buf_name}' effective byte size ({effective_bytes}) is invalid"
                    ),
                });
            }
        }

        if self.resources.len() > 16 {
            return Err(VspError::ManifestValidation {
                pack_id: pack_id.to_owned(),
                reason: format!(
                    "Pack declares {} [resources] entries, exceeding the Set 3 limit of 16",
                    self.resources.len()
                ),
            });
        }

        let mut seen_passes = HashSet::new();
        let mut seen_pass_ids = HashSet::new();
        for pass in &self.passes {
            let pname = pass.name.trim();
            let pid = pass.id.trim();
            let effective_id = if !pid.is_empty() { pid } else { pname };
            if pname.is_empty()
                || !pname
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            {
                return Err(VspError::ManifestValidation {
                    pack_id: pack_id.to_owned(),
                    reason: format!("Invalid pass name '{pname}' in [[passes]]"),
                });
            }
            if !seen_pass_ids.insert(effective_id.to_owned()) {
                return Err(VspError::ManifestValidation {
                    pack_id: pack_id.to_owned(),
                    reason: format!("Duplicate pass id '{effective_id}' in [[passes]]"),
                });
            }
            if !seen_passes.insert(pname.to_owned()) {
                return Err(VspError::ManifestValidation {
                    pack_id: pack_id.to_owned(),
                    reason: format!("Duplicate pass name '{pname}' in [[passes]]"),
                });
            }
            if pass.min_tier > pass.max_tier || pass.max_tier > 3 {
                return Err(VspError::ManifestValidation {
                    pack_id: pack_id.to_owned(),
                    reason: format!(
                        "Pass '{pname}' has invalid tier range {}..={}",
                        pass.min_tier, pass.max_tier
                    ),
                });
            }
            if pass.kind != PackPassKind::Transfer
                && pass.shader.trim().is_empty()
                && pass.vertex_shader.is_none()
                && pass.fragment_shader.is_none()
            {
                return Err(VspError::ManifestValidation {
                    pack_id: pack_id.to_owned(),
                    reason: format!("Pass '{pname}' must specify a shader path"),
                });
            }
        }

        let mut runtime_count = 0usize;
        for (key, setting) in &self.settings {
            if key.trim().is_empty() || !key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                return Err(VspError::ManifestValidation {
                    pack_id: pack_id.to_owned(),
                    reason: format!("Invalid setting identifier '{key}'"),
                });
            }
            if setting.value_type == SettingValueType::Enum {
                if setting.options.is_empty() {
                    return Err(VspError::ManifestValidation {
                        pack_id: pack_id.to_owned(),
                        reason: format!(
                            "Enum setting '{key}' must define a non-empty options array"
                        ),
                    });
                }
                if let SettingValue::Str(ref def_s) = setting.default
                    && !setting
                        .options
                        .iter()
                        .any(|o| o.eq_ignore_ascii_case(def_s))
                {
                    return Err(VspError::ManifestValidation {
                        pack_id: pack_id.to_owned(),
                        reason: format!(
                            "Enum setting '{key}' default '{def_s}' is not in options {:?}",
                            setting.options
                        ),
                    });
                }
            }
            if let (Some(min), Some(max)) = (setting.min, setting.max) {
                if min > max {
                    return Err(VspError::ManifestValidation {
                        pack_id: pack_id.to_owned(),
                        reason: format!("Setting '{key}' has min ({min}) > max ({max})"),
                    });
                }
                let def_val = setting.default.to_f32(&setting.options) as f64;
                if matches!(
                    setting.value_type,
                    SettingValueType::Int | SettingValueType::Float
                ) && (def_val < min - 1e-6 || def_val > max + 1e-6)
                {
                    return Err(VspError::ManifestValidation {
                        pack_id: pack_id.to_owned(),
                        reason: format!(
                            "Setting '{key}' default ({def_val}) is outside [{min}, {max}]"
                        ),
                    });
                }
            }
            if setting.min_tier > setting.max_tier || setting.max_tier > 3 {
                return Err(VspError::ManifestValidation {
                    pack_id: pack_id.to_owned(),
                    reason: format!(
                        "Setting '{key}' has invalid tier range {}..={}",
                        setting.min_tier, setting.max_tier
                    ),
                });
            }
            if setting.runtime {
                runtime_count += if setting.value_type == SettingValueType::Vec3 {
                    3
                } else {
                    1
                };
            }
        }

        if runtime_count > Self::MAX_RUNTIME_PACK_PARAMS {
            return Err(VspError::ManifestValidation {
                pack_id: pack_id.to_owned(),
                reason: format!(
                    "Pack declares {runtime_count} runtime settings, exceeding WorldUBO capacity ({})",
                    Self::MAX_RUNTIME_PACK_PARAMS
                ),
            });
        }

        for (preset_id, preset) in &self.presets {
            let pid = preset_id.trim();
            if pid.is_empty()
                || !pid
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            {
                return Err(VspError::ManifestValidation {
                    pack_id: pack_id.to_owned(),
                    reason: format!("Invalid preset identifier '{preset_id}'"),
                });
            }
            if pid.eq_ignore_ascii_case("custom") {
                return Err(VspError::ManifestValidation {
                    pack_id: pack_id.to_owned(),
                    reason: "Preset id 'custom' is reserved".to_owned(),
                });
            }
            if preset.label.trim().is_empty() {
                return Err(VspError::ManifestValidation {
                    pack_id: pack_id.to_owned(),
                    reason: format!("Preset '{preset_id}' label must not be empty"),
                });
            }
            if preset.min_tier > preset.max_tier || preset.max_tier > 3 {
                return Err(VspError::ManifestValidation {
                    pack_id: pack_id.to_owned(),
                    reason: format!(
                        "Preset '{preset_id}' has invalid tier range {}..={}",
                        preset.min_tier, preset.max_tier
                    ),
                });
            }
            for (setting_key, val) in &preset.values {
                let Some(setting) = self.settings.get(setting_key) else {
                    return Err(VspError::ManifestValidation {
                        pack_id: pack_id.to_owned(),
                        reason: format!(
                            "Preset '{preset_id}' references unknown setting '{setting_key}'"
                        ),
                    });
                };
                Self::validate_preset_value(pack_id, preset_id, setting_key, setting, val)?;
            }
        }

        Ok(())
    }

    fn validate_preset_value(
        pack_id: &str,
        preset_id: &str,
        setting_key: &str,
        setting: &PackSettingDecl,
        val: &SettingValue,
    ) -> Result<(), VspError> {
        let invalid_type = || VspError::ManifestValidation {
            pack_id: pack_id.to_owned(),
            reason: format!(
                "Preset '{preset_id}' value for '{setting_key}' does not match setting type {:?}",
                setting.value_type
            ),
        };
        match setting.value_type {
            SettingValueType::Bool => {
                if !matches!(val, SettingValue::Bool(_)) {
                    return Err(invalid_type());
                }
            }
            SettingValueType::Int => {
                let SettingValue::Int(i) = val else {
                    return Err(invalid_type());
                };
                let v = *i as f64;
                if let Some(min) = setting.min
                    && v < min - 1e-6
                {
                    return Err(VspError::ManifestValidation {
                        pack_id: pack_id.to_owned(),
                        reason: format!(
                            "Preset '{preset_id}' value {i} for '{setting_key}' is below min ({min})"
                        ),
                    });
                }
                if let Some(max) = setting.max
                    && v > max + 1e-6
                {
                    return Err(VspError::ManifestValidation {
                        pack_id: pack_id.to_owned(),
                        reason: format!(
                            "Preset '{preset_id}' value {i} for '{setting_key}' exceeds max ({max})"
                        ),
                    });
                }
            }
            SettingValueType::Float => {
                let v = match val {
                    SettingValue::Float(f) => *f,
                    SettingValue::Int(i) => *i as f64,
                    _ => return Err(invalid_type()),
                };
                if !v.is_finite() {
                    return Err(invalid_type());
                }
                if let Some(min) = setting.min
                    && v < min - 1e-6
                {
                    return Err(VspError::ManifestValidation {
                        pack_id: pack_id.to_owned(),
                        reason: format!(
                            "Preset '{preset_id}' value {v} for '{setting_key}' is below min ({min})"
                        ),
                    });
                }
                if let Some(max) = setting.max
                    && v > max + 1e-6
                {
                    return Err(VspError::ManifestValidation {
                        pack_id: pack_id.to_owned(),
                        reason: format!(
                            "Preset '{preset_id}' value {v} for '{setting_key}' exceeds max ({max})"
                        ),
                    });
                }
            }
            SettingValueType::Enum => {
                let SettingValue::Str(s) = val else {
                    return Err(invalid_type());
                };
                if !setting.options.iter().any(|o| o.eq_ignore_ascii_case(s)) {
                    return Err(VspError::ManifestValidation {
                        pack_id: pack_id.to_owned(),
                        reason: format!(
                            "Preset '{preset_id}' value '{s}' for enum '{setting_key}' is not in options {:?}",
                            setting.options
                        ),
                    });
                }
            }
            SettingValueType::ColorRgb => match val {
                SettingValue::Str(s) => {
                    if SettingValue::parse_color_rgb(s).is_none() {
                        return Err(invalid_type());
                    }
                }
                SettingValue::Int(i) => {
                    if *i < 0 || *i > i64::from(SettingValue::MAX_RGB24) {
                        return Err(invalid_type());
                    }
                }
                SettingValue::Vec3(v) => {
                    if !v.iter().all(|c| c.is_finite()) {
                        return Err(invalid_type());
                    }
                }
                _ => return Err(invalid_type()),
            },
            SettingValueType::Vec3 => {
                let SettingValue::Vec3(v) = val else {
                    return Err(invalid_type());
                };
                if !v.iter().all(|c| c.is_finite()) {
                    return Err(invalid_type());
                }
            }
        }
        Ok(())
    }

    /// Merges a child pack manifest on top of its parent (`extends`) manifest.
    pub fn inherit_from(&self, parent: &Self) -> Self {
        let mut hooks = parent.hooks.clone();
        for (k, v) in &self.hooks {
            hooks.insert(k.clone(), v.clone());
        }

        let mut settings = parent.settings.clone();
        for (k, v) in &self.settings {
            settings.insert(k.clone(), v.clone());
        }

        let mut presets = parent.presets.clone();
        for (k, v) in &self.presets {
            presets.insert(k.clone(), v.clone());
        }

        let mut resources = parent.resources.clone();
        for (k, v) in &self.resources {
            resources.insert(k.clone(), v.clone());
        }

        let mut targets = parent.targets.clone();
        for (k, v) in &self.targets {
            targets.insert(k.clone(), v.clone());
        }

        let mut buffers = parent.buffers.clone();
        for (k, v) in &self.buffers {
            buffers.insert(k.clone(), v.clone());
        }

        let mut modules = parent.pipeline.modules.clone();
        for m in &self.pipeline.modules {
            if !modules.contains(m) {
                modules.push(m.clone());
            }
        }

        let mut passes = parent.passes.clone();
        for child_pass in &self.passes {
            let child_key = if !child_pass.id.is_empty() {
                child_pass.id.as_str()
            } else {
                child_pass.name.as_str()
            };
            if let Some(existing) = passes.iter_mut().find(|p| {
                let p_key = if !p.id.is_empty() {
                    p.id.as_str()
                } else {
                    p.name.as_str()
                };
                p_key == child_key
            }) {
                *existing = child_pass.clone();
            } else {
                passes.push(child_pass.clone());
            }
        }

        Self {
            pack: self.pack.clone(),
            capabilities: self.capabilities.clone(),
            pipeline: PackPipelineConfig {
                mode: if self.pipeline.mode != PipelineGraphMode::Extend {
                    self.pipeline.mode
                } else {
                    parent.pipeline.mode
                },
                gbuffer_mrt_count: self
                    .pipeline
                    .gbuffer_mrt_count
                    .max(parent.pipeline.gbuffer_mrt_count),
                modules,
            },
            hooks,
            settings,
            presets,
            resources,
            targets,
            buffers,
            passes,
        }
    }

    /// Computes deterministic slot assignments for runtime (`WorldUBO.packParams`) and specialization settings.
    pub fn build_parameter_layout(&self) -> PackParameterLayout {
        let mut runtime_ubo_indices = BTreeMap::new();
        let mut specialization_constant_ids = BTreeMap::new();
        let mut rt_idx = 0usize;
        let mut spec_id = Self::FIRST_SPEC_CONSTANT_ID;

        for (key, decl) in &self.settings {
            if decl.runtime {
                let slots = if decl.value_type == SettingValueType::Vec3 {
                    3
                } else {
                    1
                };
                if rt_idx + slots <= Self::MAX_RUNTIME_PACK_PARAMS {
                    runtime_ubo_indices.insert(key.clone(), rt_idx);
                    rt_idx += slots;
                }
            } else {
                specialization_constant_ids.insert(key.clone(), spec_id);
                spec_id += 1;
            }
        }

        PackParameterLayout {
            runtime_ubo_indices,
            specialization_constant_ids,
        }
    }

    /// Resolves specialization constant `(constant_id, raw_u32_bits)` pairs given user overrides.
    pub fn resolve_specialization_overrides(
        &self,
        user_overrides: &BTreeMap<String, String>,
    ) -> Vec<(u32, u32)> {
        let layout = self.build_parameter_layout();
        let mut out = Vec::with_capacity(layout.specialization_constant_ids.len());
        for (key, &const_id) in &layout.specialization_constant_ids {
            let Some(decl) = self.settings.get(key) else {
                continue;
            };
            let bits = if let Some(raw) = user_overrides.get(key) {
                match decl.value_type {
                    SettingValueType::Bool => {
                        u32::from(raw.eq_ignore_ascii_case("true") || raw == "1")
                    }
                    SettingValueType::Int => raw
                        .parse::<i32>()
                        .map(|v| v as u32)
                        .unwrap_or_else(|_| decl.default.to_u32_bits(&decl.options)),
                    SettingValueType::Enum => decl
                        .options
                        .iter()
                        .position(|opt| opt.eq_ignore_ascii_case(raw))
                        .map(|i| i as u32)
                        .unwrap_or_else(|| decl.default.to_u32_bits(&decl.options)),
                    SettingValueType::ColorRgb => SettingValue::parse_color_rgb(raw)
                        .map(|rgb| SettingValue::pack_rgb_as_f32(rgb).to_bits())
                        .or_else(|| raw.parse::<f32>().ok().map(f32::to_bits))
                        .unwrap_or_else(|| decl.default.to_color_f32().to_bits()),
                    _ => raw
                        .parse::<f32>()
                        .map(f32::to_bits)
                        .unwrap_or_else(|_| decl.default.to_u32_bits(&decl.options)),
                }
            } else if decl.value_type == SettingValueType::ColorRgb {
                decl.default.to_color_f32().to_bits()
            } else {
                decl.default.to_u32_bits(&decl.options)
            };
            out.push((const_id, bits));
        }
        out
    }

    /// Standard quality preset identifiers recommended by Hyaloclast (`low`, `medium`, `high`, `ultra`).
    pub const STANDARD_PRESET_IDS: &'static [&'static str] = &["low", "medium", "high", "ultra"];

    /// Returns declared presets sorted by `(preset.order, preset_id)`.
    pub fn ordered_presets(&self) -> Vec<(&str, &PackPresetDecl)> {
        let mut items: Vec<(&str, &PackPresetDecl)> =
            self.presets.iter().map(|(k, v)| (k.as_str(), v)).collect();
        items.sort_by(|(ka, va), (kb, vb)| va.order.cmp(&vb.order).then_with(|| ka.cmp(kb)));
        items
    }

    /// Resolves effective setting values for a named preset (case-insensitive match on preset key
    /// or display label), starting from each setting's default and overlaying the preset's overrides.
    pub fn resolve_preset_values(
        &self,
        preset_name: &str,
    ) -> Option<BTreeMap<String, SettingValue>> {
        let preset = self.presets.get(preset_name).or_else(|| {
            self.presets
                .iter()
                .find(|(k, v)| {
                    k.eq_ignore_ascii_case(preset_name) || v.label.eq_ignore_ascii_case(preset_name)
                })
                .map(|(_, v)| v)
        })?;
        let mut resolved = BTreeMap::new();
        for (key, decl) in &self.settings {
            resolved.insert(key.clone(), decl.default.clone());
        }
        for (key, val) in &preset.values {
            resolved.insert(key.clone(), val.clone());
        }
        Some(resolved)
    }
}

/// Severity of an issue reported by [`validate_pack_directory`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationSeverity {
    Warning,
    Error,
}

/// A single structured warning or error produced when validating a `.vsp` pack directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackValidationIssue {
    pub severity: ValidationSeverity,
    pub field: Option<String>,
    pub path: Option<String>,
    pub message: String,
}

impl std::fmt::Display for PackValidationIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let level = match self.severity {
            ValidationSeverity::Warning => "warning",
            ValidationSeverity::Error => "error",
        };
        match (&self.field, &self.path) {
            (Some(field), Some(path)) => {
                write!(f, "[{level}] ({field} @ {path}): {}", self.message)
            }
            (Some(field), None) => write!(f, "[{level}] ({field}): {}", self.message),
            (None, Some(path)) => write!(f, "[{level}] ({path}): {}", self.message),
            (None, None) => write!(f, "[{level}]: {}", self.message),
        }
    }
}

/// Structured validation result returned by [`validate_pack_directory`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackValidationReport {
    pub pack_id: Option<String>,
    pub errors: Vec<PackValidationIssue>,
    pub warnings: Vec<PackValidationIssue>,
}

impl PackValidationReport {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn add_error(
        &mut self,
        field: Option<String>,
        path: Option<String>,
        message: impl Into<String>,
    ) {
        self.errors.push(PackValidationIssue {
            severity: ValidationSeverity::Error,
            field,
            path,
            message: message.into(),
        });
    }

    pub fn add_warning(
        &mut self,
        field: Option<String>,
        path: Option<String>,
        message: impl Into<String>,
    ) {
        self.warnings.push(PackValidationIssue {
            severity: ValidationSeverity::Warning,
            field,
            path,
            message: message.into(),
        });
    }
}

fn is_safe_relative_pack_path(rel: &str) -> bool {
    let trimmed = rel.trim();
    if trimmed.is_empty() || trimmed.starts_with('/') || trimmed.starts_with('\\') {
        return false;
    }
    if trimmed.as_bytes().get(1) == Some(&b':') {
        return false;
    }
    for comp in std::path::Path::new(trimmed).components() {
        match comp {
            std::path::Component::ParentDir
            | std::path::Component::RootDir
            | std::path::Component::Prefix(_) => return false,
            _ => {}
        }
    }
    true
}

/// Validates a `.vsp` shader pack directory on disk, checking `pack.toml` schema & required
/// fields, preset definitions, path-traversal safety, and existence of referenced hooks,
/// shared modules, resources, and custom pass shaders.
pub fn validate_pack_directory(pack_dir: &std::path::Path) -> PackValidationReport {
    let mut report = PackValidationReport::default();

    if !pack_dir.exists() {
        report.add_error(
            None,
            Some(pack_dir.display().to_string()),
            "Shader pack directory does not exist",
        );
        return report;
    }
    if !pack_dir.is_dir() {
        report.add_error(
            None,
            Some(pack_dir.display().to_string()),
            "Shader pack path is not a directory",
        );
        return report;
    }

    let manifest_path = pack_dir.join("pack.toml");
    if !manifest_path.is_file() {
        report.add_error(
            Some("pack.toml".to_owned()),
            Some(manifest_path.display().to_string()),
            "Missing required 'pack.toml' manifest in pack root",
        );
        return report;
    }

    let raw = match std::fs::read_to_string(&manifest_path) {
        Ok(s) => s,
        Err(err) => {
            report.add_error(
                Some("pack.toml".to_owned()),
                Some(manifest_path.display().to_string()),
                format!("Failed to read pack.toml: {err}"),
            );
            return report;
        }
    };

    let manifest = match PackManifest::parse_and_validate(&raw) {
        Ok(m) => m,
        Err(err) => {
            report.add_error(
                Some("pack.toml".to_owned()),
                Some(manifest_path.display().to_string()),
                err.to_string(),
            );
            return report;
        }
    };

    report.pack_id = Some(manifest.pack.id.clone());

    if manifest.pack.description.trim().is_empty() {
        report.add_warning(
            Some("pack.description".to_owned()),
            None,
            "Pack description is empty; consider adding a short summary for the shader pack list",
        );
    }
    if manifest.pack.author.trim().is_empty() || manifest.pack.author == "Unknown" {
        report.add_warning(
            Some("pack.author".to_owned()),
            None,
            "Pack author is unspecified or 'Unknown'",
        );
    }

    let has_extends = manifest
        .pack
        .extends
        .as_deref()
        .is_some_and(|s| !s.trim().is_empty());

    // Validate hook files exist and are safe relative paths
    for (stage_name, hook_rel) in &manifest.hooks {
        if !is_safe_relative_pack_path(hook_rel) {
            report.add_error(
                Some(format!("hooks.{stage_name}")),
                Some(hook_rel.clone()),
                "Hook path escapes pack root or is absolute",
            );
            continue;
        }
        let full = pack_dir.join(hook_rel);
        if !full.is_file() && !has_extends {
            report.add_error(
                Some(format!("hooks.{stage_name}")),
                Some(hook_rel.clone()),
                format!("Referenced hook file '{hook_rel}' not found"),
            );
        }
    }

    // Validate shared pipeline modules
    for mod_rel in &manifest.pipeline.modules {
        if !is_safe_relative_pack_path(mod_rel) {
            report.add_error(
                Some("pipeline.modules".to_owned()),
                Some(mod_rel.clone()),
                "Module path escapes pack root or is absolute",
            );
            continue;
        }
        let full = pack_dir.join(mod_rel);
        if !full.is_file() && !has_extends {
            report.add_error(
                Some("pipeline.modules".to_owned()),
                Some(mod_rel.clone()),
                format!("Referenced pipeline module '{mod_rel}' not found"),
            );
        }
    }

    // Validate resources
    for (res_name, res_decl) in &manifest.resources {
        if !is_safe_relative_pack_path(&res_decl.path) {
            report.add_error(
                Some(format!("resources.{res_name}.path")),
                Some(res_decl.path.clone()),
                "Resource path escapes pack root or is absolute",
            );
            continue;
        }
        let full = pack_dir.join(&res_decl.path);
        if !full.is_file() && !has_extends {
            report.add_error(
                Some(format!("resources.{res_name}.path")),
                Some(res_decl.path.clone()),
                format!("Referenced resource file '{}' not found", res_decl.path),
            );
        }
    }

    // Validate custom pass shader paths
    for pass in &manifest.passes {
        let pass_label = if !pass.id.is_empty() {
            pass.id.as_str()
        } else {
            pass.name.as_str()
        };
        let mut check_shader = |field: &str, shader_rel: &str| {
            if shader_rel.trim().is_empty() {
                return;
            }
            if !is_safe_relative_pack_path(shader_rel) {
                report.add_error(
                    Some(format!("passes.{pass_label}.{field}")),
                    Some(shader_rel.to_owned()),
                    "Pass shader path escapes pack root or is absolute",
                );
            } else if !pack_dir.join(shader_rel).is_file()
                && !pack_dir.join(format!("{shader_rel}.spv")).is_file()
                && !has_extends
            {
                report.add_error(
                    Some(format!("passes.{pass_label}.{field}")),
                    Some(shader_rel.to_owned()),
                    format!("Referenced pass shader '{shader_rel}' not found"),
                );
            }
        };
        check_shader("shader", &pass.shader);
        if let Some(ref vs) = pass.vertex_shader {
            check_shader("vertex_shader", vs);
        }
        if let Some(ref fs) = pass.fragment_shader {
            check_shader("fragment_shader", fs);
        }
    }

    // Warn on missing presets when settings are declared
    if manifest.presets.is_empty() && !manifest.settings.is_empty() {
        report.add_warning(
            Some("presets".to_owned()),
            None,
            "Pack defines settings but no [presets.*] profiles (e.g., Low, Medium, High, Ultra)",
        );
    }

    // Warn if pack defines neither hooks nor shaders nor extends
    let has_shaders_dir = ["core", "compute", "rt"]
        .iter()
        .any(|sub| pack_dir.join("shaders").join(sub).is_dir());
    if manifest.hooks.is_empty() && !has_shaders_dir && !has_extends {
        report.add_warning(
            None,
            None,
            "Pack defines no [hooks], no shaders/{core,compute,rt} directory, and no 'pack.extends'",
        );
    }

    report
}
