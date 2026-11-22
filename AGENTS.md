# AGENTS.md

Operating rules for AI agents and human contributors working in **VulkanShaders** (`Hyaloclast` / `IGNIS`). Read this file in full before changing anything.

## 1. Mission & License Framework

**VulkanShaders** (`Hyaloclast`) is a Vulkan 1.3+ shader, ray-tracing, and frame-generation engine for **Minecraft: Java Edition 26.3+** (Fabric-only, RenderPearl Vulkan backend), running standalone or paired with **Pyroclast** (`VulkanTurbo`).

- **Project license**: **GPL-3.0-only** ([`LICENSE`](LICENSE)) with **Section 7 Additional Permissions** ([`LICENSE-EXCEPTION.md`](LICENSE-EXCEPTION.md)) permitting linking with proprietary upscaler/frame-gen SDKs (NVIDIA NGX/DLSS/NRD/Reflex, AMD FSR3, Intel XeSS) and independent licensing of `.vsp` shader packs.
- **Approved lineage**: **Radiante** (`Gabrieli2806`, GPL-3.0 + Section 7) served as an architectural reference for RenderPearl Vulkan backend hooks. Built-in `hyaloclast-sunlit-vanilla.vsp` adapts specific optical/material concepts from **RTX Remix for Minecraft** (Apache-2.0) and **Minecraft Miniature Shader** (MIT), all recorded in [`CREDITS.md`](CREDITS.md) and [`THIRD_PARTY_NOTICES.txt`](THIRD_PARTY_NOTICES.txt).

## 2. Clean-Room Policy & Competitor Restrictions

Do not copy, transcribe, or decompile other Minecraft mods or shader loaders unless the source has a verified compatible permissive/GPL-3.0 license and explicit maintainer approval with full credits.

| Project | License | Policy |
| :--- | :--- | :--- |
| **Sodium** (`CaffeineMC/sodium`) | **PolyForm Shield 1.0.0** (`LicenseRef-Polyform-Shield-1.0.0`) | **Strictly forbidden.** PolyForm Shield is a non-FOSS, non-compete source-available license prohibiting any reuse in competing Minecraft rendering products. |
| **OptiFine** / closed shaders | Proprietary | **Strictly forbidden.** Never decompile, reverse-engineer, or copy. |
| **Iris** (`IrisShaders/Iris`) | **LGPL-3.0** (+ `glsl-transformer` **AGPL-3.0**) | **Do not copy.** Implement RenderPearl/VSP integration cleanly from our architecture and approved references. |
| **VulkanMod** / **Embeddium** | **LGPL-3.0** | **Do not copy.** |

Rules for any permitted third-party reuse:
- Only compatible licenses (MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, or explicit GPL-3.0 with compatible terms) may be adapted—never PolyForm Shield, proprietary, or unknown-license code.
- Every reused concept or snippet must be documented in [`CREDITS.md`](CREDITS.md), [`THIRD_PARTY_NOTICES.txt`](THIRD_PARTY_NOTICES.txt), and file-level notices. Never strip existing copyright or license headers.

## 3. Mojang Code: Inspection Allowed, Shipping Forbidden

- **Allowed**: `javap -p` / `unzip -l` signature checks on locally installed Minecraft 26.3 JARs, reading public `com.mojang.renderpearl.*` / `com.mojang.blaze3d.*` contracts.
- **Forbidden**: committing decompiled Mojang code or shipping `com.mojang.*` / `net.minecraft.*` classes.

## 4. Language & Code Comment Policy

- **English only**: All Rust, Kotlin, Java, GLSL/VSL code, identifiers, comments, commit messages, PRs, and issues must be strictly in English. (Only `src/main/resources/assets/vulkanshaders/lang/*.json` may contain localized strings.)
- **High-signal comments only**: Keep comments **short, current, and professional**. Document *why*, FFI/FFM C ABI contracts, `#[repr(C)]` struct sizes/offsets (e.g., `528`-byte `std140` `WorldUboData`), Vulkan `VK_KHR_synchronization2` barriers, and mixin target signatures.
- **Forbidden comments**: Russian or non-English text, zero-information narration, stale `Fix:`/`TODO` comments, or debug suppression notes.

## 5. Architecture, Documentation, UI & Path Rules

- **Documentation (`docs/`)**: Keep [`docs/vsp-format.md`](docs/vsp-format.md), [`docs/capabilities-and-optimization.md`](docs/capabilities-and-optimization.md), and [`docs/builtin-shaders.md`](docs/builtin-shaders.md) synchronized with code changes.
- **Rust workspace (`crates/`)**: Edition 2024 (`rust-version = "1.88"`). `vsp-compiler` (`.vsp`/`.vsl` -> SPIR-V 1.6 via `shaderc`/`naga`/`rspirv`), `vulkanshaders-core` (`cdylib` FFM engine, `ash`, `gpu-allocator`, RT/BLAS/TLAS, SDK loaders), `vsp-cli`, and `vulkanshaders-sdk-bench`.
- **JVM layer (`src/main/`)**: Java 25 FFM (`NativeCoreBridge.kt`) + thin Java mixins (`dev.vulkanshaders.mixin.*`) + Kotlin state/pipeline/GUI (`dev.vulkanshaders.*`).
- **UI design**: Follow [`TASTE.md`](TASTE.md) for all shader menu screens (vanilla 20px buttons, 25px rows, 310px column, no decorative artwork).
- **OS-agnostic paths**: Never hardcode `/home/<user>`, `/Users/<user>`, `C:\Users\...`, or personal directories. Use `std::env::temp_dir()`, `Path.of`, `System.getProperty("user.home")`, or environment variables.
- **No stubs**: Every FFI entrypoint, shader hook, and SDK path must have complete real logic or fail explicitly with structured diagnostics.

## 6. Build & Verification

- **Rust workspace & built-in packs**:
  ```sh
  cargo fmt --all -- --check
  cargo clippy --workspace --all-targets -- -D warnings
  cargo test --workspace --locked
  cargo test --locked -p vulkanshaders-sdk-bench
  cargo test --locked -p vulkanshaders-core --test vendor_sdk_contract_test
  cargo run --locked -p vulkanshaders-sdk-bench -- verify --repo-root .
  python3 dev/smoke/smoke.py --ci
  cargo run --locked -p vsp-cli -- check src/main/resources/builtin-packs/hyaloclast-sunlit-vanilla.vsp --tier 3
  cargo run --locked -p vsp-cli -- check src/main/resources/builtin-packs/hyaloclast-obsidian-showcase.vsp --tier 3
  ```
- **Gradle / Fabric mod**:
  ```sh
  ./gradlew test
  ./gradlew build
  ```
- Never claim a check passed unless you ran it in the current session.
