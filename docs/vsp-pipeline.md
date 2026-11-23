# VSP Pipeline & Shader Pack Format (`0.2.0-beta.1`)

Developer reference for the **VSP Graph v2** (`.vsp`) shader package format, `.vsl` `@hook` system, `vsp-compiler`, `vsp-cli`, and porting guidance from Iris/OptiFine and Bedrock-RTX packs.

## 1. Package Directory Layout

A `.vsp` shader pack is a directory (or `.vsp`/`.zip` archive validated for path-traversal and zip-bomb safety before extraction) rooted at `pack.toml`:

```text
my-pack.vsp/
├── pack.toml                 # Manifest: metadata, capabilities, targets, passes, hooks, settings, presets
├── hooks/                    # Typed .vsl hook implementations (up to 16 stages)
│   ├── vertex_displace.vsl
│   ├── surface.vsl
│   ├── direct_light.vsl
│   ├── block_light.vsl
│   ├── water_waves.vsl
│   ├── water_surface.vsl
│   ├── sky.vsl
│   ├── clouds.vsl
│   ├── fog.vsl
│   └── post_grade.vsl
├── include/                  # Optional pack-local #include headers
├── shaders/
│   ├── core/                 # Raster .vsh / .fsh overrides (e.g. csm_shadow, depth_blit)
│   ├── compute/              # Compute .comp stages (lighting, volumetric_fog, svgf, sharc, post)
│   └── rt/                   # Ray-tracing .rgen / .rchit / .rahit / .rmiss stages
└── resources/                # Static pack textures (e.g. blue_noise_64x64.png)
```

### Pack Inheritance (`extends`)

A pack may declare `extends = "<parent-pack-id>"` in `[pack]`. `VspCompiler::resolve_pack_inheritance` resolves the inheritance chain so child packs only need to override changed hooks, passes, settings, or presets while inheriting base shaders and resources.

### Built-In Shader Packs

Hyaloclast ships two built-in packs under `src/main/resources/builtin-packs/`:

| Pack ID | Display Name | Role |
| :--- | :--- | :--- |
| `hyaloclast-sunlit-vanilla.vsp` | **Sunlit Vanilla** | Default gameplay-tuned pack covering Tiers 0–3 with vanilla-faithful lighting, Gerstner water, voxel clouds, foliage wind, and `pbr_neutral` tone mapping |
| `hyaloclast-obsidian-showcase.vsp` | **Obsidian Showcase** | Engine technology showcase (`extends = "hyaloclast-sunlit-vanilla"`) demonstrating multi-bounce SHaRC GI, ReSTIR DI, POM, volumetric cumulus/aurora, and AgX grading |

## 2. `pack.toml` Manifest Schema

Parsed by `crates/vsp-compiler/src/manifest.rs` (`PackManifest`) and `src/main/kotlin/dev/vulkanshaders/pack/ShaderPackMetadata.kt`:

- **`[pack]`**: `id`, `name`, `version`, `author`, `description`, optional `extends`.
- **`[capabilities]`**: `tiers = [0, 1, 2, 3]`, `pbr_labpbr = true`, `hdr_output = true`.
- **`[pipeline]`**: `mode = "extend"` (or custom graph mode), `gbuffer_mrt_count = 4`.
- **`[targets.<name>]`**: Custom render targets with `format` (e.g. `"rgba16f"`), `scale` (e.g. `0.5`), and `temporal` flag.
- **`[[passes]]`**: Declarative graph passes with `id`, `kind` (`compute`, `graphics`, `raytrace`), `shader`, `min_tier`, `max_tier`, optional `replaces` (overrides a built-in pass), `reads`, `writes`, and `scale`.
- **`[hooks]`**: Maps any of the 16 `HookStage` names to a `.vsl` file path.
- **`[settings.<key>]`**: Typed user settings (`float`, `int`, `bool`, `enum`, `color_rgb`) with `label`, `category`, `description`, `default`, optional `min`/`max`/`step`/`options`, optional `min_tier`/`max_tier`, and `runtime`:
  - `runtime = true`: Mapped by `PackParameterLayout` into `WorldUboData.pack_params` (up to 48 scalar floats, updated live without shader recompilation).
  - `runtime = false`: Mapped to SPIR-V specialization constants (`vs_core_update_spec_constants`), triggering pipeline specialization when changed.
- **`[resources.<key>]`**: Static textures uploaded via `vs_core_upload_pack_texture` (`path`, `format`, `filter`, `wrap`).
- **`[presets.<id>]`**: Named profiles (`label`, `description`, `order`, optional `min_tier`/`max_tier`, `[presets.<id>.values]`) exposed in `PackSettingsScreen.kt`. Per-pack overrides are persisted per Render Tier in `vulkanshaders.json`.

## 3. VSP Graph v2 Hook Stages

`HookStage` (`crates/vsp-compiler/src/manifest.rs`) defines 16 composable `.vsl` hook stages (10 v1 stages + 6 v2 stages). Each stage compiles to a deterministic dispatch function (`vs_dispatch_hook_<stage>`):

| Stage | Dispatch Function | Purpose |
| :--- | :--- | :--- |
| `vertex_displace` | `vs_dispatch_hook_vertex_displace` | Terrain/foliage vertex animation |
| `gbuffer_pack` | `vs_dispatch_hook_gbuffer_pack` | Custom G-Buffer MRT attribute packing |
| `surface` | `vs_dispatch_hook_surface` | Surface albedo, roughness, metallic, POM, and LabPBR material adjustment |
| `emissive_sample` | `vs_dispatch_hook_emissive_sample` | Emissive block/triangle light sampling |
| `restir_reservoir` | `vs_dispatch_hook_restir_reservoir` | ReSTIR DI reservoir target function and weighting |
| `direct_light` | `vs_dispatch_hook_direct_light` | Direct sun/moon BRDF and shadow evaluation |
| `block_light` | `vs_dispatch_hook_block_light` | Artificial block/torch light color and falloff |
| `indirect_gi` | `vs_dispatch_hook_indirect_gi` | Multi-bounce indirect GI / SHaRC radiance modulation |
| `water_waves` | `vs_dispatch_hook_water_waves` | Animated water surface height and normal perturbation |
| `water_surface` | `vs_dispatch_hook_water_surface` | Water Fresnel, refraction, absorption, and caustics |
| `sky` | `vs_dispatch_hook_sky` | Atmospheric scattering, sun disk, stars, and aurora |
| `clouds` | `vs_dispatch_hook_clouds` | Voxel or volumetric cloud density and lighting |
| `volumetric_media` | `vs_dispatch_hook_volumetric_media` | 3D participating media phase and scattering |
| `fog` | `vs_dispatch_hook_fog` | Distance, height, and underwater fog |
| `ao_filter` | `vs_dispatch_hook_ao_filter` | Ambient occlusion and denoiser guidance |
| `post_grade` | `vs_dispatch_hook_post_grade` | Exposure, color grading, and HDR tone mapping |

## 4. Compilation Pipeline (`vsp-compiler`)

1. **Preprocessing (`preprocessor.rs`)**: `VslPreprocessor` resolves `#include` directives (including built-in virtual headers `vulkanshaders/v1/common.vsl`, `lighting.vsl`, and `restir.vsl`), extracts `@hook` blocks, injects tier/hardware capability defines (`HardwareShaderCaps`, `VS_HAS_SER`, `VS_HAS_OPACITY_MICROMAP`), and binds pack parameter indices.
2. **SPIR-V Compilation (`compiler.rs`)**: Compiles GLSL 460 / VSL stages (`vertex`, `fragment`, `compute`, `rgen`, `rchit`, `rahit`, `rmiss`) to SPIR-V 1.6 via `naga` 30.0 / `shaderc` 0.10.
3. **Reflection & Linking (`linker.rs`)**: `SpirvHookLinker` parses compiled SPIR-V with `rspirv` 0.13 to extract `SpirvReflection` (entry points, descriptor set bindings, push constants) and verify hook signatures.
4. **Caching (`SpirvDiskCache`)**: Compiled SPIR-V modules and pipeline caches are keyed by a `blake3` content hash of preprocessed source, tier, driver UUID, and spec constants so unchanged packs reload without recompiling.

## 5. Porting Guidance: Iris / OptiFine & Bedrock-RTX

### Porting from Iris / OptiFine Shaders
- **Start with `extends = "hyaloclast-sunlit-vanilla"`**: Inherit the default 4-tier G-Buffer, shadow, compute GI, and path-tracing passes instead of rewriting every OpenGL `gbuffers_*` program from scratch.
- **Map shader programs to hooks**:
  - `gbuffers_terrain.vsh` foliage/water waving → `hooks/vertex_displace.vsl` and `hooks/water_waves.vsl`.
  - `gbuffers_terrain.fsh` LabPBR decoding / POM → `hooks/surface.vsl` and `hooks/gbuffer_pack.vsl`.
  - `deferred.fsh` / `composite*.fsh` lighting, sky, clouds, fog → `hooks/direct_light.vsl`, `hooks/block_light.vsl`, `hooks/sky.vsl`, `hooks/clouds.vsl`, `hooks/fog.vsl`.
  - `final.fsh` tone mapping & grading → `hooks/post_grade.vsl`.
- **Map `shaders.properties` options**: Convert `#define` toggles into `[settings.<key>]` with `runtime = false` (SPIR-V specialization constants) and slider floats into `runtime = true` (`WorldUboData.pack_params`).
- **Custom render passes**: Declare additional `colortex` buffers under `[targets.<name>]` and full-screen compute/graphics passes under `[[passes]]`.

### Porting from Bedrock-RTX PBR Resource & Shader Packs
- **Material channels**: Bedrock-RTX `*.mer` textures (Metallic, Emissive, Roughness) map directly to Hyaloclast's LabPBR `gMaterialPbr` / `gNormalRoughness` channels via `hooks/surface.vsl` (`roughness = 1.0 - smoothness`, `metallic`, and `emissive`).
- **Emissive & direct lighting**: Tune emissive block radiance in `hooks/emissive_sample.vsl` and ReSTIR DI weighting in `hooks/restir_reservoir.vsl` so path-traced Tier 3 lighting matches Bedrock-RTX intensities without fireflies.
- **Atmospherics & water**: Translate Bedrock JSON fog/atmosphere parameters into `hooks/sky.vsl`, `hooks/fog.vsl`, and `hooks/water_surface.vsl`.

## 6. CLI Validation, Offline Compilation & Smoke Testing

Use `vsp-cli` and `dev/smoke/smoke.py` to validate packs in CI or during shader development without launching Minecraft:

```sh
# Validate manifest, #include graph, hooks, and compilation for Tier 0..3
cargo run --locked -p vsp-cli -- check src/main/resources/builtin-packs/hyaloclast-sunlit-vanilla.vsp --tier 0
cargo run --locked -p vsp-cli -- check src/main/resources/builtin-packs/hyaloclast-sunlit-vanilla.vsp --tier 3

# Compile all shader stages to .spv files
cargo run --locked -p vsp-cli -- compile src/main/resources/builtin-packs/hyaloclast-sunlit-vanilla.vsp --out-dir /tmp/vsp-out --tier 2

# Run pack, zip-safety, localization parity, Fabric-only, and headless Vulkan validation smoke checks
python3 dev/smoke/smoke.py --ci
```

Non-zero exit codes indicate manifest errors, missing includes, invalid `@hook` signatures, zip-safety violations, or SPIR-V compilation failures.
