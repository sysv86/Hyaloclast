# VSP Format & Compiler (`VSP Graph v2`)

Hyaloclast loads shader packs in the **VSP Graph v2** format (`.vsp` directory or ZIP archive) compiled by `crates/vsp-compiler` and inspectable offline via `crates/vsp-cli` (`vsp`).

---

## 1. Pack Layout

A `.vsp` pack can be either a directory (`<gameDir>/shaderpacks/<id>.vsp/`) or a `.vsp` ZIP archive containing a root `pack.toml`:

```text
my-pack.vsp/
├── pack.toml
├── include/               # Optional local .vsl headers
├── hooks/                 # .vsl files bound in [hooks]
├── resources/             # Custom LUT / noise textures declared in [resources]
└── shaders/               # Optional custom compute / raster / RT shaders for [[passes]]
```

ZIP imports enforce path-traversal rejection, entry count limits, total uncompressed size limits, and compression-ratio checks before extraction.

---

## 2. `pack.toml` Schema

Defined in `crates/vsp-compiler/src/manifest.rs` and mirrored in `src/main/kotlin/dev/vulkanshaders/pack/ShaderPackMetadata.kt`.

### `[pack]`

| Field | Type | Default / Constraint | Description |
| :--- | :--- | :--- | :--- |
| `id` | `string` | Required; ASCII `[A-Za-z0-9_-]` | Unique pack identifier. |
| `name` | `string` | Required | Display name in the Shader Packs screen. |
| `version` | `string` | `"0.1.0"` (required by importer) | Pack semantic version. |
| `author` | `string` | `"Unknown"` | Author name. |
| `description` | `string` | `""` | Tooltip description. |
| `extends` | `string` | Optional | Parent pack ID. Supports inheritance up to depth `8` with cycle detection. Child packs inherit shaders, includes, resources, targets, and passes and can override hooks, settings, and presets. |

### `[capabilities]`

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `tiers` | `int[]` | `[0, 1, 2, 3]` | Supported Render Tiers (`0..=3`). |
| `pbr_labpbr` | `bool` | `false` (`true` if `[capabilities]` omitted) | Declares LabPBR material channel support. |
| `hdr_output` | `bool` | `false` (`true` if `[capabilities]` omitted) | Declares HDR output support. |
| `restir_di` | `bool` | `true` | Enables ReSTIR DI direct-light reservoir sampling on RT tiers. |
| `volumetric_clouds` | `bool` | `true` | Enables volumetric cloud evaluation. |
| `volumetric_fog` | `bool` | `true` | Enables volumetric fog pass. |
| `distant_horizons_lod` | `bool` | `true` | Declares Distant Horizons LOD integration support. |

### `[pipeline]`

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `mode` | `"extend" \| "custom"` | `"extend"` | `extend` injects hooks/passes into the built-in render graph; `custom` defines a pack-driven pass graph. |
| `gbuffer_mrt_count` | `int` (`4..=8`) | `4` | Active G-Buffer color attachment count (alias: `gbuffer_MRT_count`). |
| `modules` | `string[]` | `[]` | Shared `.vsl` module paths included across compilation units. |

### `[hooks]`

Maps any of the 16 hook stage names to a pack-relative `.vsl` path:

```toml
[hooks]
sky = "hooks/sky.vsl"
post_grade = "hooks/post_grade.vsl"
```

### `[settings.<key>]`

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `label` | `string` | Required (`vsp-compiler`); key name fallback in Kotlin UI | UI label. |
| `type` | `string` | Required | `bool`, `int`, `float`, `enum`, `color_rgb` (`vec3` also supported in Rust compiler). |
| `default` | value | Required | Default value matching `type`. |
| `min` / `max` / `step` | number | Optional | Numeric slider bounds and increment. |
| `options` | `string[]` | `[]` | Allowed values for `enum`. |
| `runtime` | `bool` | `true` | `true`: hot-updated via `WorldUBO.packParams` without recompilation. `false`: compiled as a SPIR-V specialization constant (`constant_id >= 100`). |
| `min_tier` / `max_tier` | `int` | `0` / `3` | Tier visibility/applicability range. |
| `category` / `description` | `string` | Optional | GUI grouping and tooltip text (Kotlin UI). |

**Runtime parameter budget**: `MAX_RUNTIME_PACK_PARAMS = 40` scalar floats (`vec3` consumes 3 slots), packed into `WorldUBO.packParams[0..9]`. Slots `40..47` (`packParams[10..11]`) are reserved by `WorldStateCollector` for biome atmosphere telemetry.

### `[presets.<id>]`

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `label` | `string` | Required | Preset name (`id = "custom"` is reserved). |
| `description` | `string` | `""` | Optional tooltip. |
| `order` | `int` | `0` | Display ordering. |
| `min_tier` / `max_tier` | `int` | `0` / `3` | Supported tier range. |
| `values` | table | `{}` | Map of `<setting_key> = <value>` overrides. |

### `[resources.<name>]`

Declares custom pack textures bound to Descriptor Set `3` (maximum `16` resources, `binding = 0..15`):

- `path` (required): relative image path.
- `format`: default `"RGBA8_UNORM"`.
- `filter` / `wrap`: sampler filtering and addressing modes.

Exposed to shaders as `uVsPackRes_<name>` and helper `vsSamplePackTexture_<name>(uv)`.

### `[targets.<name>]`

Declares custom render targets managed by `RenderGraph`:

- `format` (default `"RGBA16F"`), `scale` (`(0.0, 4.0]` in Rust; `0.125..4.0` in Kotlin), optional explicit `width`/`height`, `layers`/`array_layers`, `mips`/`mip_levels`, `clear_color` (`[f32; 4]`).
- `temporal = true`: exposes previous-frame history as `<name>_history`.
- `gbuffer_attachment` (`4..=7`): binds the target as an extra G-Buffer MRT attachment.

### `[buffers.<name>]`

Declares custom GPU storage buffers:

- `size_bytes` (default `65536`; max `512 MiB` Rust / `256 MiB` Kotlin), `stride` (default `16`), optional `element_count`.
- `persistent` (default `true`), `clear_each_frame` (default `false`), `bda` (default `true` for Buffer Device Address access).

### `[[passes]]`

Declares custom or replacement render graph passes:

- `id` (or `name`), `kind` (`compute` | `raster` | `ray_trace` | `transfer`), `enabled` (default `true`).
- Shader paths: `shader` (compute/RT), or `vertex_shader` + `fragment_shader` (raster).
- Execution & ordering: `scale` (default `1.0`), `dispatch_scale` (`[1.0, 1.0]`), `min_tier` / `max_tier`, `before`, `after`, `replaces`.
- Dependencies: `reads`, `writes`, `bindings` (map of `"set:binding"`, `"binding"`, or variable name to resource name).

---

## 3. `.vsl` Hook Stages (`HookStage`)

Hyaloclast defines **16** typed hook stages across geometry, material, lighting, atmosphere, and post-processing:

| Stage | Domain |
| :--- | :--- |
| `vertex_displace` | Terrain / foliage vertex displacement |
| `gbuffer_pack` | G-Buffer MRT packing |
| `surface` | Albedo, roughness, metallic, normal, wetness, POM surface evaluation |
| `emissive_sample` | Emissive light sampling |
| `restir_reservoir` | ReSTIR DI reservoir weighting / resampling |
| `direct_light` | Sun / moon direct illumination |
| `block_light` | Block / torchlight illumination |
| `indirect_gi` | Indirect global illumination contribution |
| `water_waves` | Water surface wave displacement / normals |
| `water_surface` | Water shading, refraction, reflection |
| `sky` | Sky dome and sun/moon disc |
| `clouds` | Cloud layer shading |
| `volumetric_media` | Participating media scattering |
| `fog` | Distance / height / volumetric fog composition |
| `ao_filter` | Ambient occlusion filtering |
| `post_grade` | Tone mapping and final color grading |

Hooks declare their stage and optional ascending execution priority (default `100`):

```glsl
// @hook(stage = "post_grade", priority = 100)
```

The preprocessor synthesizes `vs_dispatch_hook_<stage>(...)` dispatchers (and supports `// @hook <stage>` injection markers in raster G-Buffer shaders).

---

## 4. Preprocessor, Macros & Uniforms

Implemented in `crates/vsp-compiler/src/preprocessor.rs`:

- **Includes**: `#include <...>` and `#include "..."` with unit-wide deduplication (max depth `12` in Rust, `8` in Kotlin). Built-in headers:
  - `<vulkanshaders/v1/common.vsl>`
  - `<vulkanshaders/v1/lighting.vsl>`
  - `<vulkanshaders/v1/restir.vsl>`
- **Source mapping**: `SourceLineMap` maps compiler diagnostics back to the originating `.vsl` file and line number.
- **Namespace isolation**: Lexer-driven, comment/string-aware renaming (`vs_ns_<pack>_<stage>_<idx>_<ident>`) isolates helper functions, structs, and constants across chained hooks.
- **Generated macros**:
  - `VS_ACTIVE_RENDER_TIER`
  - `VS_PARAM_<KEY>` — maps to `vsWorld.packParams[idx / 4].<xyzw>` in compute/RT stages (or `vsMotion.packParams[...]` in raster).
  - `VS_SPEC_<KEY>` — emits `layout(constant_id = 100 + N)` in compute/RT stages or an inline `#define` in raster stages.
  - Hardware capability flags: `VS_HAS_SER`, `VS_HAS_SER_EXT`, `VS_HAS_SER_NV`, `VS_GPU_ADA_PLUS`, `VS_HIGH_END_GPU`, `VS_HAS_OPACITY_MICROMAP`.
- **Core bindings**:
  - `set = 0, binding = 0`: `WorldUBO` (`528`-byte `std140` layout).
  - `set = 3, binding = 0..15`: custom `[resources]` textures.

---

## 5. Compiler Pipeline & Disk Cache

- **Primary backend (`shaderc` 0.10, default `shaderc-backend` feature)**: Uses a thread-local `shaderc::Compiler` targeting **Vulkan 1.3 / SPIR-V 1.6** with `OptimizationLevel::Performance` (debug info optionally enabled via `VS_SHADER_DEBUG_INFO`).
- **Fallback backend (`naga` 30.0)**: Available when `shaderc-backend` is disabled, targeting SPIR-V 1.5 (rejects RT, BDA, SER, and OMM stages).
- **Reflection & specialization patching (`rspirv` 0.13)**: `SpirvHookLinker::reflect_module` extracts entry points, descriptor set/binding layouts, push constant sizes, workgroup dimensions, specialization constant IDs, MRT output counts, and RT/BDA/RayQuery capabilities. `patch_specialization_constants` rewrites `OpSpecConstant*` defaults in-place without recompiling GLSL/VSL.
- **Parallel compilation & `blake3` disk cache (`SpirvDiskCache`)**: Compiles across up to `16` worker threads when a pack has more than 2 stage jobs. Cache keys hash `(pack_hash, stage, entry_point, driver_uuid, spec_constants, defines, hw_caps_bits, debug_info)` with `blake3` and write atomically (temp file + rename) using a `VSPCACHE` v1 header.

---

## 6. `vsp-cli` Reference

`crates/vsp-cli` builds the `vsp` binary for offline pack authoring, validation, and compilation:

```sh
# Check manifest and compile all stages for a target tier (0..3)
cargo run --locked -p vsp-cli -- check <pack_dir> --tier 3

# Compile a pack to SPIR-V modules in an output directory
cargo run --locked -p vsp-cli -- compile <pack_dir> --out-dir <dir> --tier 2

# Validate a directory or .vsp ZIP archive across one or all tiers (with optional SER/OMM flags)
cargo run --locked -p vsp-cli -- validate <pack_path> --all-tiers [--ser] [--omm]

# Print stage compilation timings and reflection stats
cargo run --locked -p vsp-cli -- profile <pack_path> --tier 3 [--ser] [--omm]

# Scaffold a new VSP Graph v2 pack
cargo run --locked -p vsp-cli -- init <output_dir> --name "My Pack" --id my-pack --author "Author" [--force]

# Print migration guidance for porting from Iris or Bedrock RTX
cargo run --locked -p vsp-cli -- migrate-help --from iris [--out guide.md]
```
