<p align="center">
  <img src="docs/icons/hyaloclast.svg" alt="Hyaloclast (VulkanShaders)" width="440">
</p>

- <a href="docs/architecture.md"><img src="docs/icons/icon-rt.svg" alt="" width="20" height="20" align="middle"></a>&nbsp;&nbsp;[Architecture & Ray Tracing](docs/architecture.md) — RenderPearl Vulkan hooking, 4-tier DAG render graph, BLAS/TLAS, SHaRC, ReSTIR DI, and GPU profiler
- <a href="docs/vsp-pipeline.md"><img src="docs/icons/icon-vsp.svg" alt="" width="20" height="20" align="middle"></a>&nbsp;&nbsp;[VSP Compiler & Shader Packs](docs/vsp-pipeline.md) — `.vsp` Graph v2 manifest, 16 `.vsl` `@hook` stages, `vsp-compiler`, `vsp-cli`, and Iris/Bedrock-RTX porting guide
- <a href="docs/sdk-verification.md"><img src="docs/icons/icon-sdk.svg" alt="" width="20" height="20" align="middle"></a>&nbsp;&nbsp;[Vendor SDKs & Verification](docs/sdk-verification.md) — runtime DLSS/NRD/FSR3/XeSS loading, distribution policy, and 7-suite ABI bench
- <a href="CREDITS.md"><img src="docs/icons/icon-credits.svg" alt="" width="20" height="20" align="middle"></a>&nbsp;&nbsp;[Credits & Notices](CREDITS.md) — Radiante architectural lineage and shader pack attributions
- <a href="AGENTS.md"><img src="docs/icons/hyaloclast-q.svg" alt="" width="20" height="20" align="middle"></a>&nbsp;&nbsp;[Contributor & Agent Rules](AGENTS.md) — clean-room boundaries, `TASTE.md`, ABI invariants, and verification workflow

> **Version:** `0.2.0-beta.1` (Fabric-only). Targets **Minecraft: Java Edition 26.3+** on Mojang's **RenderPearl** Vulkan backend, standalone or paired with [**Pyroclast**](../Pyroclast/README.md) (`VulkanTurbo`).

**Hyaloclast** (`VulkanShaders` / `IGNIS`) is a Vulkan 1.3+ shader pack, hardware ray-tracing, path-tracing, and frame-generation engine implemented as a Rust 2024 native core (`cdylib`) bound to Minecraft via Java 25 Foreign Function & Memory (FFM).

## Supported GPUs, Drivers & Platform Policy

| Vendor / Platform | Supported Hardware | Minimum Driver | Supported Tiers & Features |
| :--- | :--- | :--- | :--- |
| **NVIDIA** | Turing (RTX 20), Ampere (RTX 30), Ada Lovelace (RTX 40), Blackwell (RTX 50); GTX 16 for Tier 0–1 | Vulkan 1.3+ (Game Ready 555+ / Linux 550+) | Tiers 0–3, `VK_KHR_ray_query`, `VK_KHR_ray_tracing_pipeline`, SER & Opacity Micromaps (Ada+), DLSS SR/RR/FG + Reflex |
| **AMD** | RDNA 2 (RX 6000), RDNA 3 (RX 7000), RDNA 4 (RX 9000); Vega/RDNA 1 for Tier 0–1 | Vulkan 1.3+ (Adrenalin 24.x+ / Mesa RADV 24.1+) | Tiers 0–3 (`VK_KHR_ray_query` / RT pipeline on RDNA 2+), AMD FSR 3 SR & FG |
| **Intel** | Arc Alchemist (A-series), Arc Battlemage (B-series); Xe iGPU for Tier 0–1 | Vulkan 1.3+ (Intel Graphics 31.0.101+ / Mesa ANV 24.1+) | Tiers 0–3 on Arc discrete GPUs, Intel XeSS |
| **macOS (Apple Silicon / MoltenVK)** | M1 / M2 / M3 / M4 via `macos-aarch64` native binary | macOS 14+ with Vulkan 1.3 MoltenVK | **Tier 0–1 only** (raster G-Buffer & compute GI). Tier 2–3 hardware RT is explicitly gated when MoltenVK lacks `VK_KHR_ray_tracing_pipeline` / `VK_KHR_acceleration_structure`. |

Pre-packaged native binaries are shipped for `linux-x86-64`, `linux-aarch64`, `win32-x86-64`, and `macos-aarch64` with SHA-256 integrity manifests (`natives/natives.sha256`) verified on extraction into `<config>/vulkanshaders/natives`.

## Installation

1. Install **Fabric Loader `0.20.1+`** for **Minecraft `26.3+`** running on **Java 25**.
2. Place **Fabric Language Kotlin `1.14.2+kotlin.2.4.21`** (bundled inside the Hyaloclast JAR or installed separately) and `vulkanshaders-0.2.0-beta.1.jar` (fat JAR or platform-specific JAR) into `<gameDir>/mods/`.
3. **Recommended companion mods**:
   - **Pyroclast** (`pyroclast`) — optional; enables zero-copy `UberGpuBuffer` terrain BLAS sharing and higher chunk throughput. Hyaloclast also runs standalone without Pyroclast.
   - **Mod Menu** (`modmenu`) — quick access to Hyaloclast settings from the Mods list.
   - **Distant Horizons** (`distanthorizons` 3.3.4+) — optional; detected automatically via reflection for 3-MRT DH G-Buffer (Tiers 0–2) and LOD BLAS ray tracing (Tiers 2–3).
4. **Incompatible mods** (declared in `fabric.mod.json` `breaks`/`conflicts`): `iris`, `oculus`, `vulkanmod`, `optifabric`, and `canvas`. Remove them before launching.

## Controls, F3 HUD & GPU Profiler

- **Keybinds** (configurable under *Options → Controls → Key Binds → Hyaloclast*):
  - **Toggle Shaders** — quickly enable/disable the active shader pack without restarting.
  - **Reload Shader Pack** — hot-reload `.vsp` hooks, `pack.toml`, and pipelines in-world.
  - **Open Shader Packs / Settings** — open the Hyaloclast shader pack and preset screen.
- **F3 Debug Overlay**: Displays a compact Hyaloclast status line with the active pack, Render Tier, active upscaler/FrameGen mode, BLAS/TLAS instance counts, and Vulkan VRAM usage.
- **GPU Pass Profiler**: Optional per-pass GPU timestamp breakdown (Vulkan timestamp queries) available in the debug overlay to inspect RT, NRD/SVGF, and post-processing frame times.

## Proprietary SDK Library Setup

Hyaloclast loads optional proprietary upscaler, denoiser, and frame-generation libraries dynamically at runtime from `<gameDir>/natives/`:

- **NVIDIA**: `nvngx_dlss.dll` / `libnvidia-ngx-dlss.so`, `nvngx_dlssd.dll` (Ray Reconstruction), `nvngx_dlssg.dll` (Frame Generation), `NRD.dll` / `libNRD.so`.
- **AMD**: `amd_fidelityfx_vk.dll` / `libamd_fidelityfx_vk.so` (FSR 3).
- **Intel**: `libxess.dll` / `libxess.so` (XeSS).

Open **Shader Libraries** (`ShaderLibrariesScreen`) in-game to inspect per-library status (*found*, *missing*, *version mismatch*, or *unsupported OS*). On the default `modrinth` distribution build, allowlisted opt-in downloads are available; on the `curseforge` build, place vendor DLL/SO files into `<gameDir>/natives/` manually.

## Building & Testing

Requires **Rust 1.88+** (Edition 2024), **JDK 25**, and Python 3.10+ for smoke checks.

```sh
cargo test --workspace --locked
cargo run --locked -p vulkanshaders-sdk-bench -- verify --repo-root .
python3 dev/smoke/smoke.py --ci
./gradlew test build
```

Distribution channel and per-platform packaging options:

```sh
./gradlew build -Pvs.distribution=modrinth                         # default fat JAR (all staged platforms)
./gradlew build -Pvs.distribution=curseforge                       # manual vendor library install only
./gradlew jar -Pvs.cargo=false -Pvs.nativePlatform=linux-x86-64    # per-platform JAR (linux-x86-64, linux-aarch64, win32-x86-64, macos-aarch64)
```

## Components

| Crate / Layer | Role | Core Dependencies |
| :--- | :--- | :--- |
| `crates/vulkanshaders-core` | Native `cdylib`: RenderPearl device merger, DAG `RenderGraph`, zero-copy BLAS/TLAS, OMM/SER, SHaRC, ReSTIR DI, SVGF/NRD, Reflex/FG pacing, C FFI | `ash` 0.38, `gpu-allocator` 0.28, `parking_lot`, `libloading` |
| `crates/vsp-compiler` | `pack.toml` parser, `.vsl` `@hook` preprocessor, SPIR-V 1.6 compiler, reflection, `blake3` disk cache | `naga` 30.0, `shaderc` 0.10, `rspirv` 0.13, `blake3` |
| `crates/vsp-cli` | Standalone `.vsp` validator and offline SPIR-V compiler (`vsp check`, `vsp compile`) | `clap`, `vsp-compiler` |
| `crates/vulkanshaders-sdk-bench` | Hardware-free 7-suite ABI, C header, binary export, FFI↔FFM, shader, and mixin verifier | `clap`, `blake3` |
| `src/main` (Java 25 / Kotlin) | Fabric-only RenderPearl mixins (`mixinPriority = 1100`), FFM `NativeCoreBridge`, 4-MRT G-Buffer, Distant Horizons bridge, GUI, F3 HUD | Java 25 FFM, Kotlin 2.4.21, Fabric Loader 0.20.1+ |

## Built-In Shader Packs

- **Sunlit Vanilla** (`hyaloclast-sunlit-vanilla.vsp`) — default gameplay-tuned pack covering Tiers 0–3 (raster G-Buffer, compute GI, inline `VK_KHR_ray_query`, and full path tracing).
- **Obsidian Showcase** (`hyaloclast-obsidian-showcase.vsp`) — engine capability showcase extending `hyaloclast-sunlit-vanilla` with multi-bounce SHaRC GI, ReSTIR DI, POM, volumetric clouds/aurora, and AgX grading.

## Known Limitations

- **Fabric-only**: NeoForge is not supported in `0.2.0-beta.1` (all NeoForge entrypoints and reflection fallbacks have been removed).
- **macOS / MoltenVK**: Restricted to Tiers 0–1 until MoltenVK exposes full `VK_KHR_ray_tracing_pipeline` and `VK_KHR_acceleration_structure` support.
- **Voxy**: Not supported. Voxy requires Sodium and an OpenGL renderer, which conflicts with RenderPearl Vulkan.
- **Proprietary SDKs**: DLSS, NRD, FSR 3, and XeSS binaries are not bundled in the mod JAR for licensing reasons; see [Vendor SDKs & Verification](docs/sdk-verification.md) for setup details.

## License & Attributions

Licensed under **GPL-3.0-only** ([`LICENSE`](LICENSE)) with **Section 7 Additional Permissions** ([`LICENSE-EXCEPTION.md`](LICENSE-EXCEPTION.md)) permitting dynamic linking with optional proprietary upscaler/frame-generation SDKs (NVIDIA NGX/DLSS/NRD/Reflex, AMD FSR3, Intel XeSS) and independent licensing of `.vsp` shader packs.

See [`CREDITS.md`](CREDITS.md) and [`THIRD_PARTY_NOTICES.txt`](THIRD_PARTY_NOTICES.txt) for architectural lineage ([Radiante](https://github.com/Gabrieli2806/Radiante)) and built-in pack attributions.

<p align="center"><sub>Verified on NVIDIA GeForce RTX 3050 Mobile + Intel Core i5 · Developed with AI assistance</sub></p>
