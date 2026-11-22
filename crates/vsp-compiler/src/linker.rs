use crate::{VspError, manifest::HookStage, preprocessor::HookAnnotation};
use rspirv::{
    binary::{Assemble, Parser},
    dr::{Loader, Module, Operand},
};
use serde::{Deserialize, Serialize};
use spirv::Op;
use std::collections::HashMap;

/// Descriptor type reflected from a compiled SPIR-V binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReflectedDescriptorType {
    UniformBuffer,
    StorageBuffer,
    CombinedImageSampler,
    SampledImage,
    StorageImage,
    Sampler,
    AccelerationStructure,
}

/// Single descriptor binding reflected from `OpDecorate DescriptorSet` / `Binding` and `OpVariable`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReflectedDescriptorBinding {
    pub set: u32,
    pub binding: u32,
    pub descriptor_type: ReflectedDescriptorType,
    pub count: u32,
    pub writable: bool,
    pub name: String,
}

/// Complete SPIR-V reflection metadata for a compiled shader stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpirvReflection {
    pub entry_points: Vec<String>,
    pub bindings: Vec<ReflectedDescriptorBinding>,
    pub push_constant_size: u32,
    pub workgroup_size: [u32; 3],
    pub spec_constant_ids: Vec<u32>,
    pub color_attachment_outputs: u32,
    pub uses_ray_query: bool,
    pub uses_buffer_device_address: bool,
    pub uses_ray_tracing_pipeline: bool,
}

impl Default for SpirvReflection {
    fn default() -> Self {
        Self {
            entry_points: vec!["main".to_owned()],
            bindings: Vec::new(),
            push_constant_size: 0,
            workgroup_size: [8, 8, 1],
            spec_constant_ids: Vec::new(),
            color_attachment_outputs: 0,
            uses_ray_query: false,
            uses_buffer_device_address: false,
            uses_ray_tracing_pipeline: false,
        }
    }
}

/// Ordered composite hook entry across multiple layered `.vsp` shader packages.
#[derive(Debug, Clone)]
pub struct CompositeHookCall {
    pub pack_id: String,
    pub stage: HookStage,
    pub priority: i32,
    pub function_name: String,
}

/// Inspects, reflects, validates, and transforms SPIR-V binary modules via `rspirv`:
/// - Orders composite `@hook(stage, priority)` chains across active shader packs for all 16 stages.
/// - Generates GLSL/VSL dispatch wrappers (`vs_dispatch_hook_<stage>`) that invoke every active
///   pack's hook function in ascending `priority` order within a single SPIR-V module.
/// - Reflects descriptor set bindings, push constant block sizes, specialization constant IDs,
///   workgroup dimensions, and MRT output locations from SPIR-V binaries.
/// - Rewrites `OpSpecConstant` / `OpSpecConstantTrue` / `OpSpecConstantFalse` default literals
///   in precompiled `.spv` modules in `< 1 ms` without re-invoking a shader compiler.
pub struct SpirvHookLinker;

impl SpirvHookLinker {
    /// Sorts hook annotations across one or more layered packs by `(priority ASC, pack_order ASC)`
    /// and generates the composite GLSL dispatch block for the target `HookStage`.
    pub fn build_composite_dispatch_glsl(stage: HookStage, hooks: &[CompositeHookCall]) -> String {
        let mut sorted: Vec<&CompositeHookCall> =
            hooks.iter().filter(|h| h.stage == stage).collect();
        sorted.sort_by_key(|h| h.priority);

        let fn_name = stage.dispatch_function_name();
        let (signature, call_args) = match stage {
            HookStage::VertexDisplace => (
                "void vs_dispatch_hook_vertex_displace(inout vec3 localPos, in VsVertexInput v)",
                "(localPos, v)",
            ),
            HookStage::GbufferPack => (
                "void vs_dispatch_hook_gbuffer_pack(in VsSurface s, inout vec4 outAlbedo, inout vec4 outNormalRoughness, inout vec4 outMaterialPbr)",
                "(s, outAlbedo, outNormalRoughness, outMaterialPbr)",
            ),
            HookStage::Surface => (
                "void vs_dispatch_hook_surface(inout VsSurface s, in uint flags, in uint bounce)",
                "(s, flags, bounce)",
            ),
            HookStage::EmissiveSample => (
                "void vs_dispatch_hook_emissive_sample(in vec3 worldPos, in vec3 normal, in uint lightIndex, inout vec3 lightRadiance, inout float lightWeight)",
                "(worldPos, normal, lightIndex, lightRadiance, lightWeight)",
            ),
            HookStage::RestirReservoir => (
                "void vs_dispatch_hook_restir_reservoir(in VsSurface s, in vec3 hitPos, in vec3 sampleLightPos, in vec3 sampleLightColor, inout float targetPdf)",
                "(s, hitPos, sampleLightPos, sampleLightColor, targetPdf)",
            ),
            HookStage::DirectLight => (
                "void vs_dispatch_hook_direct_light(in VsSurface s, in vec3 viewDir, in vec3 lightDir, inout vec3 brdf)",
                "(s, viewDir, lightDir, brdf)",
            ),
            HookStage::BlockLight => (
                "void vs_dispatch_hook_block_light(inout vec3 radiance, in uint lightKind, in float dist)",
                "(radiance, lightKind, dist)",
            ),
            HookStage::IndirectGi => (
                "void vs_dispatch_hook_indirect_gi(in VsSurface s, in vec3 worldPos, inout vec3 indirectRadiance, inout float confidence)",
                "(s, worldPos, indirectRadiance, confidence)",
            ),
            HookStage::WaterWaves => (
                "void vs_dispatch_hook_water_waves(inout float strength, inout float speed, inout float scale)",
                "(strength, speed, scale)",
            ),
            HookStage::WaterSurface => (
                "void vs_dispatch_hook_water_surface(inout VsSurface s, in vec3 worldPos, in float waveHeight)",
                "(s, worldPos, waveHeight)",
            ),
            HookStage::Sky => (
                "void vs_dispatch_hook_sky(inout vec3 radiance, in vec3 dir, in VsSkyState sky)",
                "(radiance, dir, sky)",
            ),
            HookStage::Clouds => (
                "vec4 vs_dispatch_hook_clouds(in vec3 dir, in vec3 cameraPos, in VsSkyState sky)",
                "(dir, cameraPos, sky)",
            ),
            HookStage::VolumetricMedia => (
                "void vs_dispatch_hook_volumetric_media(in vec3 samplePos, in vec3 viewDir, in float shadowVis, inout vec3 inScattering, inout float extinction)",
                "(samplePos, viewDir, shadowVis, inScattering, extinction)",
            ),
            HookStage::Fog => (
                "void vs_dispatch_hook_fog(in float dist, in vec3 dir, inout vec3 scatter, inout vec3 transmittance)",
                "(dist, dir, scatter, transmittance)",
            ),
            HookStage::AoFilter => (
                "void vs_dispatch_hook_ao_filter(in vec2 uv, in vec3 normal, in float linearDepth, inout float aoValue)",
                "(uv, normal, linearDepth, aoValue)",
            ),
            HookStage::PostGrade => (
                "void vs_dispatch_hook_post_grade(inout vec3 hdrColor, in vec2 uv, in VsPostContext post)",
                "(hdrColor, uv, post)",
            ),
        };

        let mut out = String::with_capacity(256);
        out.push_str(signature);
        out.push_str(" {\n");
        if stage == HookStage::Clouds {
            out.push_str("    vec4 accum = vec4(0.0, 0.0, 0.0, 1.0);\n");
            for hook in sorted {
                out.push_str(&format!(
                    "    {{ vec4 layer = {}{}; accum.rgb += layer.rgb * accum.a; accum.a *= layer.a; }}\n",
                    hook.function_name, call_args
                ));
            }
            out.push_str("    return accum;\n");
        } else {
            let _ = fn_name;
            for hook in sorted {
                out.push_str(&format!("    {}{};\n", hook.function_name, call_args));
            }
        }
        out.push_str("}\n");
        out
    }

    /// Reflects descriptor set bindings, push constant sizes, specialization constant IDs,
    /// workgroup dimensions, and capabilities from a SPIR-V word stream.
    pub fn reflect_spirv(spirv_words: &[u32]) -> Result<SpirvReflection, VspError> {
        Self::reflect_module(spirv_words)
    }

    /// Reflects descriptor set bindings, push constant sizes, specialization constant IDs,
    /// workgroup dimensions, and capabilities from a SPIR-V word stream.
    pub fn reflect_module(spirv_words: &[u32]) -> Result<SpirvReflection, VspError> {
        let module = Self::load_module(spirv_words)?;
        let mut reflection = SpirvReflection {
            entry_points: Vec::new(),
            bindings: Vec::new(),
            push_constant_size: 0,
            workgroup_size: [1, 1, 1],
            spec_constant_ids: Vec::new(),
            color_attachment_outputs: 0,
            uses_ray_query: false,
            uses_buffer_device_address: false,
            uses_ray_tracing_pipeline: false,
        };

        for cap in &module.capabilities {
            if let Some(Operand::Capability(c)) = cap.operands.first() {
                match c {
                    spirv::Capability::RayQueryKHR => reflection.uses_ray_query = true,
                    spirv::Capability::PhysicalStorageBufferAddresses => {
                        reflection.uses_buffer_device_address = true;
                    }
                    spirv::Capability::RayTracingKHR => {
                        reflection.uses_ray_tracing_pipeline = true;
                    }
                    _ => {}
                }
            }
        }

        for ep in &module.entry_points {
            for op in &ep.operands {
                if let Operand::LiteralString(s) = op {
                    reflection.entry_points.push(s.clone());
                }
            }
        }

        for em in &module.execution_modes {
            if em.operands.len() >= 5
                && let Operand::ExecutionMode(spirv::ExecutionMode::LocalSize) = em.operands[1]
                && let (
                    Operand::LiteralBit32(lx),
                    Operand::LiteralBit32(ly),
                    Operand::LiteralBit32(lz),
                ) = (&em.operands[2], &em.operands[3], &em.operands[4])
            {
                reflection.workgroup_size = [(*lx).max(1), (*ly).max(1), (*lz).max(1)];
            }
        }

        let mut debug_names: HashMap<u32, String> = HashMap::new();
        for inst in &module.debug_names {
            if inst.class.opcode == Op::Name
                && inst.operands.len() >= 2
                && let (Operand::IdRef(id), Operand::LiteralString(s)) =
                    (&inst.operands[0], &inst.operands[1])
            {
                debug_names.insert(*id, s.clone());
            }
        }

        let mut set_map: HashMap<u32, u32> = HashMap::new();
        let mut binding_map: HashMap<u32, u32> = HashMap::new();
        let mut location_map: HashMap<u32, u32> = HashMap::new();
        let mut non_writable: HashMap<u32, bool> = HashMap::new();
        let mut buffer_block_types: HashMap<u32, bool> = HashMap::new();
        let mut array_strides: HashMap<u32, u32> = HashMap::new();
        let mut member_offsets: HashMap<(u32, u32), u32> = HashMap::new();
        let mut member_matrix_strides: HashMap<(u32, u32), u32> = HashMap::new();
        let mut member_row_major: HashMap<(u32, u32), bool> = HashMap::new();
        let mut member_non_writable: HashMap<(u32, u32), bool> = HashMap::new();

        for inst in &module.annotations {
            match inst.class.opcode {
                Op::Decorate if inst.operands.len() >= 2 => {
                    let Operand::IdRef(target_id) = inst.operands[0] else {
                        continue;
                    };
                    let Operand::Decoration(dec) = inst.operands[1] else {
                        continue;
                    };
                    match dec {
                        spirv::Decoration::DescriptorSet => {
                            if let Some(Operand::LiteralBit32(v)) = inst.operands.get(2) {
                                set_map.insert(target_id, *v);
                            }
                        }
                        spirv::Decoration::Binding => {
                            if let Some(Operand::LiteralBit32(v)) = inst.operands.get(2) {
                                binding_map.insert(target_id, *v);
                            }
                        }
                        spirv::Decoration::Location => {
                            if let Some(Operand::LiteralBit32(v)) = inst.operands.get(2) {
                                location_map.insert(target_id, *v);
                            }
                        }
                        spirv::Decoration::SpecId => {
                            if let Some(Operand::LiteralBit32(v)) = inst.operands.get(2)
                                && !reflection.spec_constant_ids.contains(v)
                            {
                                reflection.spec_constant_ids.push(*v);
                            }
                        }
                        spirv::Decoration::NonWritable => {
                            non_writable.insert(target_id, true);
                        }
                        spirv::Decoration::BufferBlock => {
                            buffer_block_types.insert(target_id, true);
                        }
                        spirv::Decoration::ArrayStride => {
                            if let Some(Operand::LiteralBit32(stride)) = inst.operands.get(2) {
                                array_strides.insert(target_id, *stride);
                            }
                        }
                        _ => {}
                    }
                }
                Op::MemberDecorate if inst.operands.len() >= 3 => {
                    if let (
                        Operand::IdRef(struct_id),
                        Operand::LiteralBit32(member_idx),
                        Operand::Decoration(dec),
                    ) = (&inst.operands[0], &inst.operands[1], &inst.operands[2])
                    {
                        match dec {
                            spirv::Decoration::Offset => {
                                if let Some(Operand::LiteralBit32(offset)) = inst.operands.get(3) {
                                    member_offsets.insert((*struct_id, *member_idx), *offset);
                                }
                            }
                            spirv::Decoration::MatrixStride => {
                                if let Some(Operand::LiteralBit32(stride)) = inst.operands.get(3) {
                                    member_matrix_strides
                                        .insert((*struct_id, *member_idx), *stride);
                                }
                            }
                            spirv::Decoration::RowMajor => {
                                member_row_major.insert((*struct_id, *member_idx), true);
                            }
                            spirv::Decoration::ColMajor => {
                                member_row_major.insert((*struct_id, *member_idx), false);
                            }
                            spirv::Decoration::NonWritable => {
                                member_non_writable.insert((*struct_id, *member_idx), true);
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        reflection.spec_constant_ids.sort_unstable();

        // Index type instructions by result_id
        let mut type_insts: HashMap<u32, &rspirv::dr::Instruction> = HashMap::new();
        let mut const_u32: HashMap<u32, u32> = HashMap::new();
        for inst in &module.types_global_values {
            if let Some(res_id) = inst.result_id {
                type_insts.insert(res_id, inst);
                if inst.class.opcode == Op::Constant
                    && let Some(Operand::LiteralBit32(v)) = inst.operands.first()
                {
                    const_u32.insert(res_id, *v);
                }
            }
        }

        for inst in &module.types_global_values {
            if inst.class.opcode != Op::Variable {
                continue;
            }
            let Some(var_id) = inst.result_id else {
                continue;
            };
            let Some(ptr_type_id) = inst.result_type else {
                continue;
            };
            let Some(Operand::StorageClass(storage_class)) = inst.operands.first() else {
                continue;
            };

            let pointee_type_id = type_insts
                .get(&ptr_type_id)
                .and_then(|ptr_inst| {
                    if ptr_inst.class.opcode == Op::TypePointer {
                        ptr_inst.operands.get(1).and_then(|op| match op {
                            Operand::IdRef(id) => Some(*id),
                            _ => None,
                        })
                    } else {
                        None
                    }
                })
                .unwrap_or(0);

            if *storage_class == spirv::StorageClass::PushConstant {
                let exact_size = Self::compute_spirv_type_size(
                    pointee_type_id,
                    &type_insts,
                    &const_u32,
                    &array_strides,
                    &member_offsets,
                    &member_matrix_strides,
                    &member_row_major,
                    0,
                );
                let aligned_size = exact_size.max(4).div_ceil(4) * 4;
                reflection.push_constant_size = reflection.push_constant_size.max(aligned_size);
                continue;
            }

            if *storage_class == spirv::StorageClass::Output
                && let Some(&loc) = location_map.get(&var_id)
            {
                reflection.color_attachment_outputs =
                    reflection.color_attachment_outputs.max(loc + 1);
            }

            let Some(&binding) = binding_map.get(&var_id) else {
                continue;
            };
            let set = set_map.get(&var_id).copied().unwrap_or(0);

            let (unwrapped_type_id, count) = match type_insts.get(&pointee_type_id) {
                Some(t_inst) if t_inst.class.opcode == Op::TypeArray => {
                    let elem = match t_inst.operands.first() {
                        Some(Operand::IdRef(id)) => *id,
                        _ => pointee_type_id,
                    };
                    let len = match t_inst.operands.get(1) {
                        Some(Operand::IdRef(cid)) => const_u32.get(cid).copied().unwrap_or(1),
                        _ => 1,
                    };
                    (elem, len)
                }
                Some(t_inst) if t_inst.class.opcode == Op::TypeRuntimeArray => {
                    let elem = match t_inst.operands.first() {
                        Some(Operand::IdRef(id)) => *id,
                        _ => pointee_type_id,
                    };
                    (elem, 4096)
                }
                _ => (pointee_type_id, 1),
            };

            let descriptor_type = match storage_class {
                spirv::StorageClass::StorageBuffer => ReflectedDescriptorType::StorageBuffer,
                spirv::StorageClass::Uniform => {
                    if buffer_block_types
                        .get(&unwrapped_type_id)
                        .copied()
                        .unwrap_or(false)
                    {
                        ReflectedDescriptorType::StorageBuffer
                    } else {
                        ReflectedDescriptorType::UniformBuffer
                    }
                }
                spirv::StorageClass::UniformConstant => match type_insts.get(&unwrapped_type_id) {
                    Some(t_inst) => match t_inst.class.opcode {
                        Op::TypeSampledImage => ReflectedDescriptorType::CombinedImageSampler,
                        Op::TypeSampler => ReflectedDescriptorType::Sampler,
                        Op::TypeAccelerationStructureKHR => {
                            ReflectedDescriptorType::AccelerationStructure
                        }
                        Op::TypeImage => {
                            let sampled = match t_inst.operands.get(5) {
                                Some(Operand::LiteralBit32(v)) => *v,
                                _ => 1,
                            };
                            if sampled == 2 {
                                ReflectedDescriptorType::StorageImage
                            } else {
                                ReflectedDescriptorType::SampledImage
                            }
                        }
                        _ => ReflectedDescriptorType::CombinedImageSampler,
                    },
                    None => ReflectedDescriptorType::CombinedImageSampler,
                },
                _ => continue,
            };

            let all_members_non_writable = type_insts
                .get(&unwrapped_type_id)
                .filter(|t_inst| {
                    t_inst.class.opcode == Op::TypeStruct && !t_inst.operands.is_empty()
                })
                .is_some_and(|t_inst| {
                    (0..t_inst.operands.len() as u32).all(|idx| {
                        member_non_writable
                            .get(&(unwrapped_type_id, idx))
                            .copied()
                            .unwrap_or(false)
                    })
                });
            let is_non_writable = non_writable.get(&var_id).copied().unwrap_or(false)
                || non_writable
                    .get(&unwrapped_type_id)
                    .copied()
                    .unwrap_or(false)
                || all_members_non_writable;
            let writable = matches!(
                descriptor_type,
                ReflectedDescriptorType::StorageImage | ReflectedDescriptorType::StorageBuffer
            ) && !is_non_writable;

            let name = debug_names
                .get(&var_id)
                .filter(|s| !s.is_empty())
                .cloned()
                .or_else(|| {
                    debug_names
                        .get(&unwrapped_type_id)
                        .filter(|s| !s.is_empty())
                        .cloned()
                })
                .unwrap_or_else(|| format!("set{set}_binding{binding}"));

            reflection.bindings.push(ReflectedDescriptorBinding {
                set,
                binding,
                descriptor_type,
                count,
                writable,
                name,
            });
        }

        reflection.bindings.sort_by_key(|b| (b.set, b.binding));

        Ok(reflection)
    }

    /// Parses a SPIR-V word stream into an `rspirv::dr::Module`, verifies entry points, and
    /// patches any specialization constant defaults matching `(constant_id, raw_u32_bits)`.
    pub fn patch_specialization_constants(
        spirv_words: &[u32],
        spec_overrides: &[(u32, u32)],
    ) -> Result<Vec<u32>, VspError> {
        let mut module = Self::load_module(spirv_words)?;

        if !spec_overrides.is_empty() {
            let mut id_to_spec_id = HashMap::new();
            for inst in &module.annotations {
                if inst.class.opcode == Op::Decorate
                    && inst.operands.len() >= 3
                    && let (
                        Operand::IdRef(target_id),
                        Operand::Decoration(spirv::Decoration::SpecId),
                        Operand::LiteralBit32(spec_id),
                    ) = (&inst.operands[0], &inst.operands[1], &inst.operands[2])
                {
                    id_to_spec_id.insert(*target_id, *spec_id);
                }
            }

            let override_map: HashMap<u32, u32> = spec_overrides.iter().copied().collect();

            for inst in &mut module.types_global_values {
                let Some(result_id) = inst.result_id else {
                    continue;
                };
                let Some(&spec_id) = id_to_spec_id.get(&result_id) else {
                    continue;
                };
                let Some(&new_bits) = override_map.get(&spec_id) else {
                    continue;
                };

                match inst.class.opcode {
                    Op::SpecConstant => {
                        if let Some(Operand::LiteralBit32(val)) = inst.operands.first_mut() {
                            *val = new_bits;
                        }
                    }
                    Op::SpecConstantTrue | Op::SpecConstantFalse => {
                        inst.class = rspirv::grammar::INSTRUCTION_TABLE.get(if new_bits != 0 {
                            Op::SpecConstantTrue
                        } else {
                            Op::SpecConstantFalse
                        });
                    }
                    _ => {}
                }
            }
        }

        Ok(module.assemble())
    }

    /// Extracts the list of `OpEntryPoint` names from a SPIR-V module using `rspirv`.
    pub fn inspect_entry_points(spirv_words: &[u32]) -> Result<Vec<String>, VspError> {
        let module = Self::load_module(spirv_words)?;
        let mut names = Vec::with_capacity(module.entry_points.len());
        for ep in &module.entry_points {
            for op in &ep.operands {
                if let Operand::LiteralString(s) = op {
                    names.push(s.clone());
                }
            }
        }
        Ok(names)
    }

    pub fn from_hook_annotations(
        pack_id: &str,
        annotations: &[HookAnnotation],
    ) -> Vec<CompositeHookCall> {
        annotations
            .iter()
            .map(|a| CompositeHookCall {
                pack_id: pack_id.to_owned(),
                stage: a.stage,
                priority: a.priority,
                function_name: a.function_name.clone(),
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_spirv_type_size(
        type_id: u32,
        type_insts: &HashMap<u32, &rspirv::dr::Instruction>,
        const_u32: &HashMap<u32, u32>,
        array_strides: &HashMap<u32, u32>,
        member_offsets: &HashMap<(u32, u32), u32>,
        member_matrix_strides: &HashMap<(u32, u32), u32>,
        member_row_major: &HashMap<(u32, u32), bool>,
        depth: u32,
    ) -> u32 {
        if depth > 16 {
            return 0;
        }
        let Some(inst) = type_insts.get(&type_id) else {
            return 0;
        };
        match inst.class.opcode {
            Op::TypeBool => 4,
            Op::TypeInt | Op::TypeFloat => match inst.operands.first() {
                Some(Operand::LiteralBit32(bits)) => (*bits / 8).max(1),
                _ => 4,
            },
            Op::TypeVector => {
                let elem_id = match inst.operands.first() {
                    Some(Operand::IdRef(id)) => *id,
                    _ => return 0,
                };
                let count = match inst.operands.get(1) {
                    Some(Operand::LiteralBit32(n)) => *n,
                    _ => 1,
                };
                let elem_size = Self::compute_spirv_type_size(
                    elem_id,
                    type_insts,
                    const_u32,
                    array_strides,
                    member_offsets,
                    member_matrix_strides,
                    member_row_major,
                    depth + 1,
                );
                elem_size * count
            }
            Op::TypeMatrix => {
                let col_type_id = match inst.operands.first() {
                    Some(Operand::IdRef(id)) => *id,
                    _ => return 0,
                };
                let col_count = match inst.operands.get(1) {
                    Some(Operand::LiteralBit32(n)) => *n,
                    _ => 4,
                };
                let col_size = Self::compute_spirv_type_size(
                    col_type_id,
                    type_insts,
                    const_u32,
                    array_strides,
                    member_offsets,
                    member_matrix_strides,
                    member_row_major,
                    depth + 1,
                );
                col_size * col_count
            }
            Op::TypeArray => {
                let elem_id = match inst.operands.first() {
                    Some(Operand::IdRef(id)) => *id,
                    _ => return 0,
                };
                let len = match inst.operands.get(1) {
                    Some(Operand::IdRef(cid)) => const_u32.get(cid).copied().unwrap_or(1),
                    _ => 1,
                };
                let stride = array_strides.get(&type_id).copied().unwrap_or_else(|| {
                    Self::compute_spirv_type_size(
                        elem_id,
                        type_insts,
                        const_u32,
                        array_strides,
                        member_offsets,
                        member_matrix_strides,
                        member_row_major,
                        depth + 1,
                    )
                });
                stride * len
            }
            Op::TypeStruct => {
                let mut max_end = 0u32;
                for (member_idx, op) in inst.operands.iter().enumerate() {
                    let Operand::IdRef(member_ty_id) = op else {
                        continue;
                    };
                    let idx_u32 = member_idx as u32;
                    let offset = member_offsets
                        .get(&(type_id, idx_u32))
                        .copied()
                        .unwrap_or(max_end);
                    let member_size = if let Some(m_inst) = type_insts.get(member_ty_id)
                        && m_inst.class.opcode == Op::TypeMatrix
                        && let Some(&mat_stride) = member_matrix_strides.get(&(type_id, idx_u32))
                    {
                        let col_count = match m_inst.operands.get(1) {
                            Some(Operand::LiteralBit32(n)) => *n,
                            _ => 4,
                        };
                        let row_major = member_row_major
                            .get(&(type_id, idx_u32))
                            .copied()
                            .unwrap_or(false);
                        let rows = if row_major {
                            m_inst
                                .operands
                                .first()
                                .and_then(|o| match o {
                                    Operand::IdRef(cid) => type_insts.get(cid),
                                    _ => None,
                                })
                                .and_then(|v| match v.operands.get(1) {
                                    Some(Operand::LiteralBit32(n)) => Some(*n),
                                    _ => None,
                                })
                                .unwrap_or(col_count)
                        } else {
                            col_count
                        };
                        mat_stride * rows
                    } else {
                        Self::compute_spirv_type_size(
                            *member_ty_id,
                            type_insts,
                            const_u32,
                            array_strides,
                            member_offsets,
                            member_matrix_strides,
                            member_row_major,
                            depth + 1,
                        )
                    };
                    max_end = max_end.max(offset + member_size);
                }
                max_end
            }
            _ => 0,
        }
    }

    fn load_module(spirv_words: &[u32]) -> Result<Module, VspError> {
        let bytes: &[u8] = bytemuck::cast_slice(spirv_words);
        let mut loader = Loader::new();
        let parser = Parser::new(bytes, &mut loader);
        match parser.parse() {
            Ok(()) => Ok(loader.module()),
            Err(err) => Err(VspError::SpirvLink(format!(
                "Failed to parse SPIR-V binary: {err:?}"
            ))),
        }
    }
}
