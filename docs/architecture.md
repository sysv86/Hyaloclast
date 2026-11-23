# Hyaloclast Architecture (`0.2.0-beta.1`, Fabric-Only)

Technical reference for **Hyaloclast** (`VulkanShaders` / `IGNIS`), targeting Minecraft: Java Edition **26.3+** (Fabric Loader `0.20.1+`) on Mojang's native **RenderPearl** Vulkan backend (`com.mojang.renderpearl.backend.vulkan.*`).

## 1. Layering & Workspace Layout

Hyaloclast splits responsibilities across a Rust 2024 native core (`crates/`) and a Fabric-only Java 25 / Kotlin 2.4.21 mod layer (`src/main/`):

| Component | Language / Stack | Responsibility |
| :--- | :--- | :--- |
| `crates/vulkanshaders-core` | Rust 2024 (`cdylib`), `ash` 0.38, `gpu-allocator` 0.28 | Vulkan device/instance merging, pooled VRAM allocation & `VK_EXT_memory_budget` tracking, DAG `RenderGraph`, zero-copy BLAS/TLAS, OMM/SER, SHaRC, ReSTIR DI, SVGF/NRD, upscaler/FG loaders, GPU timestamp profiler, C FFI (`ffi/mod.rs`) |
| `crates/vsp-compiler` | Rust 2024, `naga` 30.0, `shaderc` 0.10, `rspirv` 0.13, `blake3` | `pack.toml` parsing, `.vsl` `@hook` preprocessing, SPIR-V 1.6 compilation, reflection, persistent disk cache & async pipeline prewarming |
| `crates/vsp-cli` | Rust 2024, `clap` | Standalone `.vsp` validation (`vsp check`) and offline compilation (`vsp compile`) |
| `crates/vulkanshaders-sdk-bench` | Rust 2024 | Hardware-free 7-suite ABI, C header, binary symbol, FFI/FFM, shader, and mixin contract verifier |
| `src/main/java/dev/vulkanshaders/mixin` | Java 25, Spongepowered Mixin (`mixinPriority = 1100`) | Thin RenderPearl Vulkan, world, texture, Distant Horizons, F3 debug HUD, and Reflex hooks |
| `src/main/kotlin/dev/vulkanshaders` | Kotlin 2.4.21, Java 25 FFM (`java.lang.foreign`) | `NativeCoreBridge`, `GbufferTargetManager`, `WorldStateCollector`, `VulkanShadersTurboExtension`, `StandaloneRenderPearlBridge`, `DistantHorizonsBridge`, `ProprietaryLibraryManager`, keybinds, GUI |

### Standalone vs. Pyroclast Mode

`vulkanshaders.mixins.json` sets `mixinPriority = 1100` so Hyaloclast layers deterministically above **Pyroclast** (`vulkanturbo.mixins.json`, `mixinPriority = 900`) when Pyroclast is installed.

- **With Pyroclast (`pyroclast`)**: Registers `dev.vulkanshaders.bridge.VulkanShadersTurboExtension` via the `"turboshaders"` entrypoint in `fabric.mod.json` and uses Pyroclast's `TurboShadersBridge` SPI for terrain vertex layout extension, 4-MRT attachment overrides, descriptor set binding, and zero-copy section slice notifications.
- **Standalone Mode (without Pyroclast)**: `StandaloneRenderPearlBridge.kt` attaches directly to RenderPearl's terrain, texture atlas (`TextureAtlasPbrMixin`), `SectionSliceRegistry`, and `GbufferTargetManager` with no hard dependency on Pyroclast.

### Native Packaging & Extraction Integrity

`build.gradle.kts` packages stripped release binaries for four platforms under `natives/<platformDir>/`:
- `linux-x86-64/libvulkanshaders_core.so` (`x86_64-unknown-linux-gnu`)
- `linux-aarch64/libvulkanshaders_core.so` (`aarch64-unknown-linux-gnu`)
- `win32-x86-64/vulkanshaders_core.dll` (`x86_64-pc-windows-msvc` / `-gnu`)
- `macos-aarch64/libvulkanshaders_core.dylib` (`aarch64-apple-darwin`)

Each packaged binary ships alongside a `.sha256` sidecar and a consolidated `natives/natives.sha256` manifest. At startup, `NativeCoreBridge` extracts the matching platform binary into `<config>/vulkanshaders/natives/<sha256-prefix>_<libname>` and binds all FFM symbols via Java 25 `SymbolLookup`.

## 2. RenderPearl Vulkan Hooking & GPU Memory

Hyaloclast shares Mojang's `VkInstance`, `VkPhysicalDevice`, `VkDevice`, and `VkQueue` instead of creating a second Vulkan context:

1. **Instance & Device Merging (`gpu/device_merger.rs`, `VulkanInstanceMixin`, `VulkanDeviceMixin`)**:
   - Before instance creation, `NativeCoreBridge.syncLwjglInstanceProcAddr()` passes LWJGL's `vkGetInstanceProcAddr` to `vs_core_set_instance_proc_addr`.
   - `vs_core_create_merged_instance` and `vs_core_create_merged_device` intercept RenderPearl's create-info structs, merge required Vulkan 1.3+ extensions and `pNext` feature chains (buffer device address, descriptor indexing/bindless, acceleration structure, ray query, ray tracing pipeline, SER `VK_EXT_ray_tracing_invocation_reorder` / `NV`, opacity micromap `VK_EXT_opacity_micromap`, `VK_EXT_memory_budget`, `VK_NV_low_latency2`), and record the resolved capability bitmask.
   - **macOS / MoltenVK Tier Gating**: When `VK_KHR_ray_tracing_pipeline` or `VK_KHR_acceleration_structure` is unavailable (as on MoltenVK), device capability resolution explicitly caps selectable tiers to Tier 0–1 so raster and compute GI run cleanly without device-creation errors.
2. **VRAM Pooling & OOM Protection (`gpu/buffer.rs`, `gpu-allocator`)**:
   - Standalone buffers and images allocate through pooled Vulkan memory blocks (`gpu-allocator` and 64 MB `SlabArenaAllocator` slabs) to stay well below `maxMemoryAllocationCount`.
   - `VK_EXT_memory_budget` telemetry feeds VRAM pressure tracking (and F3 HUD reporting), throttling BLAS compaction and LOD cutoff distances before hitting `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
3. **Queue Synchronization & Device-Lost Recovery (`VulkanQueueLockMixins`, `ffi/mod.rs`, `crash.rs`)**:
   - RenderPearl and native Rust submissions share the graphics queue under a reentrant lock (`vs_core_lock_queue` / `vs_core_unlock_queue`).
   - Timeline semaphore values are mirrored into the native `RetireQueue` (`vs_core_bind_timeline_semaphore`, `vs_core_record_submission`, `vs_core_notify_timeline_completed`) so transient GPU buffers, displaced foliage buffers, and old BLAS structures are freed only after the GPU passes their retirement value.
   - On `VK_ERROR_DEVICE_LOST` or shader compilation failure, Hyaloclast records diagnostics in `HyaloclastErrorLog`, notifies the player, and falls back to safe vanilla rendering without crashing the JVM.
4. **Swapchain & Reflex Pacing (`ReflexSurfaceMixin`, `framegen/`)**:
   - `vs_core_reflex_set_swapchain`, `vs_core_reflex_marker` (markers `0..6`), and `vs_core_reflex_sleep_and_sample_input` drive `VK_NV_low_latency2` pacing.
   - Frame generation intercepts `vkAcquireNextImageKHR` and `vkQueuePresentKHR` (`vs_core_intercept_acquire_next_image`, `vs_core_intercept_queue_present`) and runs post-GUI composition in `vs_core_execute_after_gui`.

## 3. Rendering Tiers & DAG Render Graph

`crates/vsp-compiler/src/manifest.rs` defines four hardware/algorithmic tiers (`RenderTier`):

| Tier | Enum | Pipeline Summary |
| :--- | :--- | :--- |
| **0** | `Tier0Raster` | 4-MRT raster G-Buffer + Cascaded Shadow Maps (CSM) + Hi-Z SSR + GTAO + Volumetric Fog |
| **1** | `Tier1ComputeGi` | Tier 0 G-Buffer + Compute Froxel GI / Screen-Space Radiance Cache + Volumetric Fog |
| **2** | `Tier2RayQuery` | Hybrid 4-MRT G-Buffer + inline `VK_KHR_ray_query` shadows, reflections, GI, ReSTIR DI, and SHaRC over `UberGpuBuffer` BLAS/TLAS |
| **3** | `Tier3PathTracing` | Full path tracing (`VK_KHR_ray_tracing_pipeline` / wavefront compute PT with optional SER & OMM) + ReSTIR DI + SHaRC + SVGF/NRD/DLSS-RR; skips vanilla world rasterization and reprojects depth via `depth_blit` (`DepthReprojectPushConstants`, 32 bytes) for vanilla overlays and block outlines |

`RenderGraph` (`crates/vulkanshaders-core/src/graph/mod.rs`) compiles active built-in and pack-declared passes into a dependency-ordered sequence (`CompiledPassStep`) filtered by `[min_tier, max_tier]` and synthesizes `VK_KHR_synchronization2` barriers (`SynthesizedBarrier`) from declared `ResourceAccess` edges. Optional Vulkan timestamp queries bracket each pass for the in-game GPU profiler and F3 HUD.

## 4. G-Buffer & Uniform Buffer Contracts

### 4-MRT G-Buffer (`GbufferTargetManager.kt`)

When a shader pack is active in Tiers 0–2, Hyaloclast attaches three auxiliary render targets alongside Minecraft's main color buffer:

| Target | Format | Purpose |
| :--- | :--- | :--- |
| `mainColor` | RenderPearl main color | Albedo / base color |
| `gNormalRoughness` | `RGBA16_FLOAT` | Encoded world/view normal + perceptual roughness |
| `gMaterialPbr` | `RGBA8_UNORM` | LabPBR metallic/F0, porosity/SSS, emissive, material flags |
| `gVelocity` | `RG16_FLOAT` | Per-pixel motion vectors for TAA, SVGF, TSR, DLSS/FSR3/XeSS, and FrameGen |

Motion uniforms use a `352`-byte UBO (`GbufferTargetManager.VS_MOTION_UBO_BYTES`). Dynamic CSM shadow settings (`shadowResolution` `512..8192`, `shadowDistanceChunks` `4..64`) are pushed via `vs_core_set_shadow_config`.

### `WorldUboData` (`528` bytes, `std140`)

Both `WorldStateCollector.kt`, `crates/vulkanshaders-core/src/modules/mod.rs`, and `include/vulkanshaders/v1/common.vsl` enforce an exact `528`-byte layout (checked by a compile-time `assert!` in Rust and by `vulkanshaders-sdk-bench`):

| Byte Offset | Field | Type |
| :--- | :--- | :--- |
| `0..63` | `cur_view_proj` | `mat4` |
| `64..127` | `prev_view_proj` | `mat4` |
| `128..191` | `inv_proj` | `mat4` |
| `192..255` | `inv_view` | `mat4` |
| `256..271` | `camera_pos_frac` + `frame_index` | `vec3` + `int` |
| `272..287` | `camera_delta` + `game_time_sec` | `vec3` + `float` |
| `288..303` | `sun_dir` + `rain` | `vec3` + `float` |
| `304..319` | `thunder`, `eye_in_fluid`, `depth_far`, `held_light` | `float`, `int`, `float`, `float` |
| `320..511` | `pack_params[12]` | `vec4[12]` (48 runtime shader settings) |
| `512..527` | `held_light_position` | `vec4` (camera-relative) |

## 5. Zero-Copy Ray Tracing, SHaRC, ReSTIR DI & Diagnostics

- **Zero-Copy Terrain BLAS (`rt/mod.rs`, `rt/as_builder.rs`)**: Terrain sections uploaded into `UberGpuBuffer` heaps are registered via `vs_core_register_section_slice_with_pbr`. Native code builds BLAS directly from the existing `VkBuffer` device address and shared sequential quad index buffer without duplicating vertex memory. When vertices are region-shifted (`isRegionShifted = true`), TLAS instances anchor at `RegionOrigin = (sectionOrigin & ~63)`.
- **Dynamic Entities (`rt/entities.rs`)**: In Tiers 2–3, dynamic entities, block entities, and particles stream into a per-frame `EntityBlasManager` via `vs_core_submit_entity_vertices`.
- **SHaRC (`rt/sharc.rs`, `sharc_resolve.comp`)**: Spatial Hash Radiance Cache maintains `2^20` (`1,048,576`) world-space hash cells addressed via 64-bit buffer device addresses (`SharcConfigUniform`, `112` bytes, `std430`) with temporal EMA, Karis anti-firefly clamping, and stale cell eviction.
- **ReSTIR DI (`include/vulkanshaders/v1/restir.vsl`)**: Ping-pong `RestirReservoirSSBO` (`set = 1, binding = 4`) performs motion-vector temporal reprojection and spatial neighbor reuse over dynamic emissive lights (`vs_core_submit_emissive_lights`).
- **Bindless Textures (`gpu/bindless.rs`)**: `BindlessTextureTable` tracks registered `VkImage`/`VkImageView`/sampler slots in an O(1) array and links LabPBR normal/specular companions via `vs_core_link_pbr_companions`.
- **F3 HUD & GPU Profiler**: Surfaces active pack, Render Tier, active upscaler/FG mode, BLAS/TLAS instance counts, VRAM usage, and optional per-pass GPU timings directly in the Minecraft `F3` overlay.

## 6. Distant Horizons Compatibility

`DistantHorizonsBridge.kt` detects **Distant Horizons 3.x** (verified with 3.3.4 on Fabric 26.3) via reflection (`DhApi`, no compile-time dependency):

- **Tiers 0–2**: Redirects `BlazeDhTerrainRenderer` into a 3-MRT DH G-Buffer (`[dhColor, dhNormalRoughness, dhMaterialPbr]`), suppresses conflicting DH post-processing (`DhPostProcessSuppressMixins`), and expands `vsWorld.depthFar` to the DH LOD radius.
- **Tiers 2–3**: Unpacks DH's 24-byte `POS_META_COLOR_MATERIAL_NORMAL_TEXTURE_OFFSET` vertices (16-byte fallback) into 28-byte `LodVertexGpu` structs and uploads them into `LodBlasManager` via `vs_core_upload_lod_mesh_cpu`.

Voxy support is not included because Voxy requires Sodium and an OpenGL renderer, which conflicts with RenderPearl Vulkan.
