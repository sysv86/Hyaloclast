# Built-In Shader Packs

Hyaloclast ships two built-in VSP Graph v2 shader packs under `src/main/resources/builtin-packs/`. On startup, `ShaderPackManager` extracts them into `<gameDir>/shaderpacks/<id>.vsp/` alongside a SHA-256 `.builtin-stamp` file.

---

## 1. Sunlit Vanilla (`hyaloclast-sunlit-vanilla.vsp`)

Default gameplay-tuned pack preserving Minecraft's visual identity while scaling across all four Render Tiers.

- **ID / Version**: `hyaloclast-sunlit-vanilla` (`1.0.0`)
- **Supported Tiers**: `0, 1, 2, 3`
- **Capabilities**: `pbr_labpbr = true`, `hdr_output = true`, `restir_di = true`, `volumetric_clouds = true`, `volumetric_fog = true`, `distant_horizons_lod = true`
- **Pipeline**: `mode = "extend"`, `gbuffer_mrt_count = 4`
- **Custom Target & Pass**:
  - `[targets.volumetric_fog]` — `rgba16f` at `scale = 0.5`
  - `[[passes]] id = "volumetric_fog_pass"` — `compute` (`shaders/compute/volumetric_fog.comp`), active on Tiers `0..=2`, `replaces = "volumetric_fog_pass"`, reads `mc_main_depth` and `shadow_csm`, writes `volumetric_fog`

### Contents (40 files)

| Directory | Files |
| :--- | :--- |
| Root | `pack.toml`, `CREDITS.md`, `THIRD_PARTY_NOTICES.txt` |
| `resources/` | `blue_noise_64x64.png` |
| `include/vulkanshaders/v1/` | `common.vsl`, `lighting.vsl`, `restir.vsl` |
| `hooks/` (10 hooks) | `vertex_displace.vsl`, `surface.vsl`, `direct_light.vsl`, `block_light.vsl`, `water_waves.vsl`, `water_surface.vsl`, `sky.vsl`, `clouds.vsl`, `fog.vsl`, `post_grade.vsl` |
| `shaders/core/` (5) | `csm_shadow.vsh`, `csm_shadow_28b.vsh`, `csm_shadow.fsh`, `depth_blit.vsh`, `depth_blit.fsh` |
| `shaders/compute/` (14) | `compute_lighting.comp`, `denoiser_prepare.comp`, `denoiser_recompose.comp`, `hdr_compose.comp`, `optical_flow_fg.comp`, `post_tonemap.comp`, `rayquery_lighting.comp`, `sharc_resolve.comp`, `ssgi.comp`, `svgf_atrous.comp`, `svgf_temporal.comp`, `tsr_upscale.comp`, `volumetric_fog.comp`, `wind_displace.comp` |
| `shaders/rt/` (4) | `pathtrace.rgen`, `pathtrace.rchit`, `pathtrace.rahit`, `pathtrace.rmiss` |

---

## 2. Obsidian Showcase (`hyaloclast-obsidian-showcase.vsp`)

Engine capability showcase demonstrating VSP pack inheritance, stronger atmospheric scattering, parallax occlusion mapping (POM), and multi-bounce path tracing.

- **ID / Version**: `hyaloclast-obsidian-showcase` (`1.0.0`)
- **Inheritance**: `extends = "hyaloclast-sunlit-vanilla"`
- **Supported Tiers**: `0, 1, 2, 3`
- **Contents (11 files)**: `pack.toml` and 10 overridden `.vsl` hooks (`vertex_displace`, `surface`, `direct_light`, `block_light`, `water_waves`, `water_surface`, `sky`, `clouds`, `fog`, `post_grade`). All core, compute, RT shaders, includes, resources, targets, and passes are inherited from `hyaloclast-sunlit-vanilla`.

---

## 3. Settings & Default Comparison

Both built-in packs expose the same 11 configurable settings and 4 presets (`low`, `medium`, `high`, `ultra`):

| Setting Key | Runtime | Tiers | Sunlit Vanilla Default | Obsidian Showcase Default |
| :--- | :--- | :--- | :--- | :--- |
| `sun_intensity` | Yes | 0–3 | `2.75` | `3.3` |
| `fog_density` | Yes | 0–3 | `0.0016` | `0.0030` |
| `water_wave_strength` | Yes | 0–3 | `0.28` | `0.45` |
| `tone_mapping` (`agx` / `aces` / `pbr_neutral`) | Yes | 0–3 | `pbr_neutral` | `agx` |
| `torch_intensity` | Yes | 0–3 | `1.15` | `1.45` |
| `godrays_intensity` | Yes | 0–3 | `0.55` | `1.35` |
| `gi_bounce_strength` | Yes | 1–3 | `0.95` | `1.25` |
| `wetness_strength` | Yes | 0–3 | `0.60` | `0.90` |
| `pom_depth` | Yes | 0–3 | `0.0` | `0.03` |
| `pt_max_bounces` | No (spec const) | 3 | `2` | `3` |
| `enable_wind` | No (spec const) | 0–3 | `true` | `true` |

---

## 4. Lifecycle, Legacy Cleanup & Crash Guard

Implemented in `src/main/kotlin/dev/vulkanshaders/pack/ShaderPackManager.kt`:

- **Stamping & pruning**: Extracted built-in directories carry `.builtin-stamp` (SHA-256). Updating the mod refreshes stamped files and prunes obsolete files from `hyaloclast-sunlit-vanilla` (`shaders/compute/mc_depth.comp`, `shaders/core/gbuffer_terrain.vsh`, `shaders/core/gbuffer_terrain.fsh`).
- **Legacy built-in pack hiding**: Older built-in IDs (`hyaloclast-default`, `hyaloclast-vanilla-reverie`, `hyaloclast-golden-solstice`, `hyaloclast-nordic-borealis`, `hyaloclast-cinder-mist`) remain resolvable as `extends` parents for third-party packs, but stamped unmodified directories are hidden from the pack list while preserving user-modified copies.
- **Crash guard**: Before activating a pack, Hyaloclast writes `<configDir>/vulkanshaders.trying`. If that marker still exists on the next startup (indicating a crash during shader activation), shaders are automatically disabled (`selectedPackId = "off"`).

## ReLight Vanilla (`relight-vanilla.vsp`)

ReLight Vanilla is the flagship VSL 2.0 reference pack in `vsl-integration`. It pairs physically grounded atmospheric optics, vanilla-faithful LabPBR materials, multi-layer volumetric aurora, interactive water dynamics, and a native VSL 2.0 SPIR-V tiled lighting compute unit (`vsl-units/vsl_tiled_lighting`).
