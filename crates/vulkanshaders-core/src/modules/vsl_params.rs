//! VSL 2.0 reflection sidecar ingestion for pack-parameter BDA wiring.
//!
//! `vslc reflect <unit>.vsl --tier N` writes a `CompiledShaderUnit` JSON array whose
//! `used_params` entries name pack settings and their byte offsets inside the device-addressable
//! parameter buffer. The native pipeline path only ingests the `.spv` binary, so this module reads
//! the matching `<unit>.json` sidecar and converts those offsets into a deterministic, packed
//! host buffer layout. Missing or malformed sidecars are tolerated: a unit without `used_params`
//! simply has no live pack parameters to wire.

use serde::Deserialize;
use std::{collections::HashSet, fs, path::Path};

/// One reflected pack parameter, copied from VSL's `used_params` reflection.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct VslUsedParam {
    /// Pack setting name from `pack.vsl` (or the legacy `pack.toml` key).
    pub name: String,
    /// Byte offset of the value inside the BDA parameter buffer.
    #[serde(default)]
    pub byte_offset: u32,
    /// Value size in bytes (4 for scalars, 12 for `color_rgb`).
    #[serde(default)]
    pub byte_size: u32,
}

/// Minimal view of a `CompiledShaderUnit` reflection JSON object.
#[derive(Debug, Clone, Deserialize)]
pub struct VslReflectedUnit {
    #[serde(default)]
    pub used_params: Vec<VslUsedParam>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ReflectionFile {
    Units(Vec<VslReflectedUnit>),
    Unit(VslReflectedUnit),
}

/// Parses `vslc reflect` JSON, accepting either a single unit or an array of units.
pub fn parse_reflection_json(text: &str) -> Vec<VslReflectedUnit> {
    match serde_json::from_str::<ReflectionFile>(text) {
        Ok(ReflectionFile::Units(units)) => units,
        Ok(ReflectionFile::Unit(unit)) => vec![unit],
        Err(_) => Vec::new(),
    }
}

/// Loads `used_params` from `<dir>/<stage_name>.json`. Returns no bindings when the sidecar is
/// absent, unreadable, or does not match the expected reflection shape.
pub fn load_used_params(dir: &Path, stage_name: &str) -> Vec<VslUsedParam> {
    let path = dir.join(format!("{stage_name}.json"));
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    parse_reflection_json(&text)
        .into_iter()
        .flat_map(|unit| unit.used_params)
        .collect()
}

/// Deduplicates and sorts reflected bindings so buffer layout and writes stay deterministic.
pub fn merge_used_params(bindings: impl IntoIterator<Item = VslUsedParam>) -> Vec<VslUsedParam> {
    let mut seen = HashSet::new();
    let mut merged: Vec<VslUsedParam> = bindings
        .into_iter()
        .filter(|binding| {
            seen.insert((binding.name.clone(), binding.byte_offset, binding.byte_size))
        })
        .collect();
    merged.sort_by(|left, right| {
        left.byte_offset
            .cmp(&right.byte_offset)
            .then_with(|| left.name.cmp(&right.name))
    });
    merged
}

/// Size of the BDA parameter buffer needed for `bindings`, rounded up to a 16-byte multiple.
pub fn pack_param_buffer_size(bindings: &[VslUsedParam]) -> usize {
    let highest_end = bindings
        .iter()
        .map(|binding| binding.byte_offset as usize + binding.byte_size as usize)
        .max()
        .unwrap_or(0);
    highest_end.div_ceil(16) * 16
}

/// Packs live parameter values into the BDA buffer image.
///
/// The engine caches runtime pack settings in `WorldUboData::pack_params` as 48 little-endian
/// `f32`s (320..511). VSL's reflected offsets are relative to the same pack-setting sequence, so
/// each parameter reads its value from `pack_params[byte_offset / 4]`. Values whose reflected range
/// falls outside the cached 48 floats are written as zero rather than reading out of bounds.
pub fn pack_param_bytes(bindings: &[VslUsedParam], pack_params: &[[f32; 4]; 12]) -> Vec<u8> {
    let floats: Vec<f32> = pack_params.iter().flatten().copied().collect();
    let mut bytes = vec![0u8; pack_param_buffer_size(bindings)];
    for binding in bindings {
        let start = binding.byte_offset as usize;
        let len = binding.byte_size as usize;
        if !start.is_multiple_of(4) || start >= bytes.len() {
            continue;
        }
        let copy_len = len.min(bytes.len() - start);
        for index in 0..copy_len.div_ceil(4) {
            let float_index = start / 4 + index;
            let value = floats.get(float_index).copied().unwrap_or(0.0);
            let chunk = (start + index * 4).min(bytes.len());
            let remaining = bytes.len() - chunk;
            bytes[chunk..chunk + remaining.min(4)]
                .copy_from_slice(&value.to_le_bytes()[..remaining.min(4)]);
        }
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    const REFLECTION: &str = r#"[
      {
        "entry_point": "probe",
        "stage": "compute",
        "spirv": [],
        "descriptors": [],
        "push_constants": [],
        "used_params": [
          {"name": "sun_intensity", "byte_offset": 0, "byte_size": 4},
          {"name": "sun_tint", "byte_offset": 16, "byte_size": 12}
        ]
      },
      {
        "entry_point": "probe_aux",
        "used_params": [{"name": "tone_mapping", "byte_offset": 32, "byte_size": 4}]
      }
    ]"#;

    #[test]
    fn parses_vslc_reflection_and_ignores_unknown_fields() {
        let units = parse_reflection_json(REFLECTION);
        assert_eq!(units.len(), 2);
        assert_eq!(units[0].used_params.len(), 2);
        assert_eq!(units[0].used_params[1].byte_offset, 16);
        assert_eq!(units[1].used_params[0].name, "tone_mapping");
    }

    #[test]
    fn accepts_single_unit_object_and_is_tolerant_of_garbage() {
        let single =
            r#"{"entry_point":"solo","used_params":[{"name":"x","byte_offset":8,"byte_size":4}]}"#;
        assert_eq!(parse_reflection_json(single)[0].used_params.len(), 1);
        assert!(parse_reflection_json("not json").is_empty());
        assert!(parse_reflection_json("{}")[0].used_params.is_empty());
    }

    #[test]
    fn missing_sidecar_yields_no_bindings() {
        let bindings = load_used_params(Path::new("/definitely/not/a/real/path"), "unit");
        assert!(bindings.is_empty());
    }

    #[test]
    fn merge_deduplicates_and_sorts_by_offset() {
        let bindings = merge_used_params([
            VslUsedParam {
                name: "b".to_owned(),
                byte_offset: 8,
                byte_size: 4,
            },
            VslUsedParam {
                name: "a".to_owned(),
                byte_offset: 0,
                byte_size: 4,
            },
            VslUsedParam {
                name: "b".to_owned(),
                byte_offset: 8,
                byte_size: 4,
            },
        ]);
        assert_eq!(bindings.len(), 2);
        assert_eq!(bindings[0].name, "a");
        assert_eq!(bindings[1].name, "b");
    }

    #[test]
    fn buffer_size_is_rounded_up_to_16_bytes() {
        assert_eq!(pack_param_buffer_size(&[]), 0);
        assert_eq!(
            pack_param_buffer_size(&[VslUsedParam {
                name: "x".to_owned(),
                byte_offset: 0,
                byte_size: 4,
            }]),
            16
        );
        assert_eq!(
            pack_param_buffer_size(&[VslUsedParam {
                name: "x".to_owned(),
                byte_offset: 17,
                byte_size: 12,
            }]),
            32
        );
    }

    #[test]
    fn packs_values_at_reflected_offsets() {
        let mut pack_params = [[0.0f32; 4]; 12];
        pack_params[0][0] = 2.75;
        pack_params[1][0] = 0.25;
        pack_params[1][1] = 0.5;
        pack_params[1][2] = 0.75;
        let bindings = [
            VslUsedParam {
                name: "sun_intensity".to_owned(),
                byte_offset: 0,
                byte_size: 4,
            },
            VslUsedParam {
                name: "sun_tint".to_owned(),
                byte_offset: 16,
                byte_size: 12,
            },
        ];
        let bytes = pack_param_bytes(&bindings, &pack_params);
        assert_eq!(bytes.len(), 32);
        assert_eq!(f32::from_le_bytes(bytes[0..4].try_into().unwrap()), 2.75);
        assert_eq!(f32::from_le_bytes(bytes[16..20].try_into().unwrap()), 0.25);
        assert_eq!(f32::from_le_bytes(bytes[20..24].try_into().unwrap()), 0.5);
        assert_eq!(f32::from_le_bytes(bytes[24..28].try_into().unwrap()), 0.75);
        assert_eq!(&bytes[28..32], &[0u8; 4]);
    }

    #[test]
    fn out_of_range_offsets_do_not_panic() {
        let bindings = [VslUsedParam {
            name: "far".to_owned(),
            byte_offset: 4096,
            byte_size: 4,
        }];
        let bytes = pack_param_bytes(&bindings, &[[0.0f32; 4]; 12]);
        assert_eq!(bytes.len(), 4112);
        assert_eq!(bytes[4096..].iter().sum::<u8>(), 0);
    }
}
