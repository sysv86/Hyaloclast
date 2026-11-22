# Engine Capabilities & Optimizations

Technical reference for Hyaloclast's native Vulkan 1.3+ rendering core (`crates/vulkanshaders-core`) and Fabric/FFM integration layer (`src/main`).

---

## 1. Render Tiers (0–3)

Hyaloclast scales across four explicit tiers (`RenderTier`, default `TIER_1_COMPUTE_GI` in `VulkanShadersConfig`):

| Tier | Name | Pipeline Summary |
| :--- | :--- | :--- |
| **Tier 0** | `TIER_0_RASTER` | Raster 4-MRT G-Buffer + 4-cascade CSM + Hi-Z SSR + GTAO + Volumetric Fog. |
| **Tier 1** | `TIER_1_COMPUTE_GI` | Raster 4-MRT G-Buffer + Compute Froxel GI / SSGI + Volumetric Fog. |
| **Tier 2** | `TIER_2_RAY_QUERY` | Hybrid Raster G-Buffer + inline `VK_KHR_ray_query` shadows, reflections, and GI + ReSTIR DI + SHaRC over BLAS/TLAS. |
| **Tier 3** | `TIER_3_PATH_TRACING` | Full path tracing (`VK_KHR_ray_tracing_pipeline` / wavefront PT) + ReSTIR DI + SHaRC + SVGF / NRD / DLSS-RR + `depth_blit` overlay reprojection. |

On macOS / MoltenVK, Tiers 2–3 are automatically gated when `VK_KHR_ray_tracing_pipeline` and `VK_KHR_acceleration_structure` are unavailable.

---

## 2. RenderPearl Integration & Device Merger

- **Mixin layering**: `vulkanshaders.mixins.json` sets `mixinPriority = 1100` (`FluidRendererMixin` uses `1200`), layering cleanly above Pyroclast (`900`).
- **Extension injection (`gpu/device_merger.rs`)**: Intercepts RenderPearl's Vulkan device creation to enable required Vulkan 1.2/1.3 and hardware RT extensions when supported:
  - `VK_KHR_acceleration_structure`, `VK_KHR_ray_query`, `VK_KHR_ray_tracing_pipeline`
  - `VK_KHR_synchronization2`, `VK_KHR_buffer_device_address`
  - `VK_EXT_descriptor_indexing`, `VK_EXT_memory_budget`
  - `VK_EXT_opacity_micromap`, `VK_EXT_ray_tracing_invocation_reorder` / `VK_NV_ray_tracing_invocation_reorder`
- **Dual integration mode**:
  - **With Pyroclast (`VulkanShadersTurboExtension`)**: Registers via Fabric's `"turboshaders"` entrypoint. Uses the 40-byte `IGNIS_TERRAIN_40B` vertex layout (first 12 bytes = `RGB32_FLOAT` local position) so `UberGpuBuffer` terrain slices feed both rasterization and Tier 2/3 BLAS builds with zero vertex copy, alongside 4-MRT G-Buffer view overrides and post-world / post-GUI callbacks.
  - **Standalone (`StandaloneRenderPearlBridge`)**: Runs on vanilla Fabric RenderPearl without Pyroclast, using `TextureRegistry`, `SectionSliceRegistry`, `GbufferTargetManager`, and the 28-byte terrain shadow path (`csm_shadow_28b.vsh`).

---

## 3. Declarative DAG Render Graph

`crates/vulkanshaders-core/src/graph/mod.rs` compiles active passes (`BlasTlasUpdate`, `ShadowCsm`, `GbufferMrt`, `ComputeVolumetricFog`, `ComputeSsgi`, `ComputeFroxelGi`, `RayQueryLighting`, `FullPathTracing`, `DenoiserSvgfOrNrd`, `DepthReprojectOverlay`, upscaling, tonemapping, and post-GUI `HdrCompose` / `FrameGenInterpolate`) into a dependency DAG:

- Supports `extend` (`before` / `after` / `replaces`) and `custom` pack modes.
- Automatically synthesizes `VK_KHR_synchronization2` pipeline barriers and image layout transitions between producers and consumers.
- Runs **backward-BFS Dead Pass Culling** from frame outputs so unused passes are never dispatched.

---

## 4. Ray Tracing, Lighting & Denoising

- **Zero-Copy BLAS/TLAS (`rt/as_builder.rs`)**: Sub-allocates 256-byte-aligned `VkAccelerationStructureKHR` storage from 64 MiB `DEVICE_LOCAL` arenas via `SlabArenaAllocator` with compacted BLAS copies.
- **Opacity Micromaps (`rt/omm.rs`)**: Supports 2-state (1-bit) and 4-state (2-bit) `VK_EXT_opacity_micromap` up to `MAX_OMM_SUBDIV_LEVEL = 5` (1,024 microtriangles/triangle) using 16 MiB storage and 8 MiB scratch slabs (`INSTANCE_FLAG_HAS_OMM = 1 << 9`).
- **Shader Execution Reordering (SER)**: Probes both `VK_EXT_ray_tracing_invocation_reorder` and `VK_NV_ray_tracing_invocation_reorder` and exposes `VS_HAS_SER`, `VS_HAS_SER_EXT`, and `VS_HAS_SER_NV` to shaders.
- **SHaRC Spatial Hash Radiance Cache (`rt/sharc.rs`)**: `SHARC_DEFAULT_CAPACITY = 1 << 20` (1,048,576 cells), 112-byte BDA `SharcConfigUniform` triple-buffered per frame-in-flight, with GPU-tier-scaled update downsampling (`2` Enthusiast, `3` High, `4` Low/Mid).
- **ReSTIR DI (`rt/lights.rs`)**: Supports up to `MAX_EMISSIVE_LIGHTS = 512` (48-byte `GpuEmissiveLight`, 16-byte header, 32-byte `VsRestirReservoir`) with dimension-aware emissive/ambient scaling (Overworld `1.0 / 1.0`, Nether `1.20 / 1.35`, End `1.30 / 0.85`).
- **Denoisers**:
  - Built-in **SVGF** (`svgf_temporal.comp` + `svgf_atrous.comp` with 16-byte `AtrousPushConstants`).
  - Optional **NVIDIA NRD** (`REBLUR` / `RELAX`) via `sdk/nrd.rs`.
  - Optional **NVIDIA DLSS Ray Reconstruction** (`DLSS_RR`).

---

## 5. Runtime Optimizations

### GPU & Native Core (`crates/vulkanshaders-core`)

- **Dynamic CSM culling & texel snapping (`modules/mod.rs`, `ffi/mod.rs`)**: 4 orthographic cascades in a 288-byte `CsmShadowUBO`, live-configurable via `vs_core_set_shadow_config(shadow_resolution, shadow_distance_chunks)` (default `2048` / `12` chunks; resolution `512..4096` or `8192` on high-end GPUs, distance `4..64` chunks). Uses `from_sun_dir_snapped` texel snapping and per-cascade light-frustum AABB culling (`is_aabb_visible_in_cascade`).
- **Batched descriptor updates**: `descriptor_sig_cache` deduplicates unchanged writes and flushes dirty `VkWriteDescriptorSet` entries in batched `vkUpdateDescriptorSets` calls.
- **O(1) bindless texture table (`gpu/bindless.rs`)**: `4096`-slot `VK_EXT_descriptor_indexing` table (`PARTIALLY_BOUND_BIT | UPDATE_AFTER_BIND_BIT`) backed by a flat `slot_entries` vector and free-slot stack over RenderPearl's native image views/samplers.
- **Sub-allocated GPU memory & VRAM pressure trimming (`gpu/buffer.rs`, `lib.rs`)**: Routes `VkBuffer`/`VkImage` allocations through `gpu-allocator 0.28` to avoid driver `maxMemoryAllocationCount` limits. Periodically polls `VK_EXT_memory_budget` (`VRAM_POLL_INTERVAL_FRAMES`) via `VramPressurePolicy` to throttle BLAS compaction, tighten LOD cutoffs, clear SHaRC, and invoke `render_graph.trim_under_memory_pressure(...)` on pressure cooldown.
- **Non-blocking GPU pass profiler (`gpu/profiler.rs`)**: 3 frames-in-flight × 64 passes × 2 timestamps (`VkQueryPool`, `WITH_AVAILABILITY_BIT`), reading only completed frames without CPU stalls.

### JVM & Companion Bridges (`src/main/kotlin`)

- **Zero-allocation frame collection (`WorldStateCollector.kt`)**: Pre-allocated direct off-heap `MemoryUtil.memCalloc` buffers (`528`-byte `uboBuffer` and `emissiveLightBuffer`), reusable scratch `Matrix4f`/`Vector3f`/`MutableBlockPos`, cached `FloatArray(40)` pack parameters invalidated only on pack/config/tier/override hash changes, block-coordinate-cached biome atmosphere profiles, and a bounded `SectionLightCache` (`MAX_CACHED_SECTIONS = 4096`).
- **Distant Horizons 3.x bridge (`DistantHorizonsBridge.kt`)**: Zero compile-time dependency; uses cached reflection (`ContainerReflectionCache`, `ListReflectionCache`, `Vec3iReflectionCache`) and `ThreadLocal` buffers. On Tiers 0–2 redirects `BlazeDhTerrainRenderer` into a 3-MRT DH G-Buffer (`dhColor`, `dhNormalRoughness`, `dhMaterialPbr`) while suppressing DH's vanilla clouds/SSAO/fog/far-fade/TAA; on Tiers 2–3 unpacks 24-byte (or 16-byte) DH vertices into 28-byte `LodVertexGpu` structs for `LodBlasManager`.

---

## 6. Proprietary SDK Loading & Distribution Policy

`RuntimeSdkManager` probes `<gameDir>/natives/` and OS library directories (Windows `%SystemRoot%\System32` + NVIDIA DriverStore; Linux `/usr/lib`, `/usr/lib64`, `/usr/lib/{x86_64,aarch64}-linux-gnu`, `/usr/local/lib`; macOS `/usr/local/lib`, `/opt/homebrew/lib`) via `libloading`:

- **NVIDIA NGX / DLSS / NRD**: NGX initializes on a dedicated 8 MiB stack thread (`config/vulkanshaders/ngx`; disableable via `VS_CORE_ENABLE_NGX=0`). DLSS SR/RR/FG and NRD require manual library placement.
- **AMD FSR 3 & Intel XeSS**: Redistributable Windows x64 binaries can be downloaded on opt-in from allowlisted GitHub hosts (`github.com`, `raw.githubusercontent.com`, `objects.githubusercontent.com`, `release-assets.githubusercontent.com`, `codeload.github.com`) with SHA-256 verification on `modrinth` and `github` builds. Linux/macOS builds require manual placement.
- **Channel enforcement (`DistributionPolicy.kt`)**: Controlled at build time via `-Pvs.distribution=<modrinth|curseforge|github>` (or runtime override `-Dvulkanshaders.distribution=...`). `curseforge` disables all external downloads and downgrades every library to manual install.
