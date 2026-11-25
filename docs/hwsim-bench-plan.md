# Hardware-Free Vulkan & AMD FSR 4.1.2 Verification Benchmark (`vulkanshaders-hwsim-bench`)

## 1. Purpose and Epistemic Scope

`vulkanshaders-hwsim-bench` is a standalone Rust 2024 crate added alongside the existing `crates/vulkanshaders-sdk-bench` (which remains untouched). It addresses a concrete development constraint: validating Vulkan rendering integrations—specifically **AMD FSR 4.1.2 (FidelityFX "Redstone" SDK 2.4.0)**, **FSR 3.1.6**, **NVIDIA DLSS 4 / 4.5 (NGX)**, **NVIDIA NRD**, and **Intel XeSS**—on a Linux workstation that lacks AMD RDNA 2/3/4 GPUs, NVIDIA Ada/Blackwell GPUs, and Windows drivers.

Unlike `vulkanshaders-sdk-bench` (which performs static symbol export, C-header parsing, and `size_of`/`align_of` checks), `vulkanshaders-hwsim-bench` executes **behavioral, contract-level, and hardware-profile-driven verification** completely offline and without physical GPU hardware.

### Epistemic Boundary (What Is Proven vs. What Still Requires Physical Hardware)

Every report emitted by `vulkanshaders-hwsim-bench` explicitly partitions findings into four statuses so offline checks never masquerade as physical GPU validation:

| Status | Meaning | Examples |
|---|---|---|
| `PROVEN_OFFLINE` | Mathematically or deterministically verified without GPU hardware | Field-by-field `offset_of!` + SysV64/Win64 ABI parity, SDK 2.4.0 `desc_type` constants, `pNext` chain cycle/depth/extension gating, `VkImageLayout` transition state machine, handle leak/double-free tracking, FSR 4.1.2 network selection & 15-frame in-flight limit |
| `STATIC_HEURISTIC` | Verified against modeled hardware profiles or uncompressed static bytecode | Uncompressed SPIR-V `OpCapability` / `SPV_EXT_float8` / `OpTypeCooperativeMatrixKHR` compatibility against modeled GPU profiles |
| `INCONCLUSIVE_UNINSPECTED` | Target payload could not be inspected (never rolled up into `PASS`) | Compressed shader blobs (zstd/zlib/lz4) embedded inside vendor DLLs such as `amd_fidelityfx_upscaler_vk.dll` |
| `REQUIRES_PHYSICAL_HARDWARE` | Explicitly enumerated checklist of items that require real silicon + drivers | Driver SPIR-V-to-ISA codegen bugs, VGPR spills, L2/TCC cache coherency, real FP8/INT8 neural network visual quality (ghosting/NaNs), real GPU frame time & VRAM bandwidth |

---

## 2. Critical Real-World Findings & Target Verification Surface

1. **AMD FSR 4.1.2 / FidelityFX SDK 2.4.0 (October 2026) Vulkan ABI**:
   - Official FidelityFX SDK 2.4.0 restores Vulkan parity for FSR Upscaling 4.1.2 (`0xf5a5ca1e01001002`), FSR 3.1.6 / 2.3.4 analytical fallback, and FSR Frame Generation 4.0.2 / 3.1.6.
   - `ffx_api.h` defines `FFX_API_BACKEND_ID_DX12 = 0x0000_0000`, `FFX_API_BACKEND_ID_XBOX = 0x0100_0000`, and `FFX_API_BACKEND_ID_VK = 0x0200_0000` (legacy SDK 1.1.x used `0x0000_0003` for Vulkan backend desc).
   - **Detected bug in current `vulkanshaders-core/src/sdk/fsr3.rs`**: `FFX_API_BACKEND_ID_VK` is currently set to `0x0100_0000` (Xbox in SDK 2.4.0), producing `0x0100_0003` instead of SDK 2.4.0 Vulkan (`0x0200_0003`) or legacy 1.1.x (`0x0000_0003`). `vulkanshaders-hwsim-bench` explicitly audits and flags this constant mismatch.
   - Mandatory SDK 2.4.0 lifecycle contracts:
     - Version discovery (`FFX_API_QUERY_DESC_TYPE_GET_VERSIONS = 4`), explicit pinning (`FFX_API_DESC_TYPE_OVERRIDE_VERSION = 5`), active provider query (`FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION = 6`), global debug (`FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG = 7`), and mandatory `ffxCreateContextDescUpscaleVersion` chaining.
     - 4 bundled FSR 4.1.2 networks under `0xf5a5ca1e01001002`:
       - `FSR4` (FP8): AMD discrete RDNA 4 (`0x1002`), `VK_KHR_cooperative_matrix` (`16x16x16` FP8 E4M3/E5M2) + `VK_EXT_shader_float8` (`shaderFloat8`, `shaderFloat8CooperativeMatrix`) + wave32 `subgroupSizeControl`.
       - `FSR4-i8` (INT8): RDNA 3 (family `0x91`, rev `0x01..=0xFE`), Strix Halo (`0x96`, rev `0xC0..=0xCF`), or PCI `0x7481`, requiring `shaderIntegerDotProduct` (DP4a), `shaderFloat16`, `shaderInt8`, wave32.
       - `FSR4-i8-2` (INT8): RDNA 2 discrete (family `0x8F`, rev `0x28..=0x4F`).
       - `FSR4L-i8` (INT8 Lite): Van Gogh (`0x90`, rev `0x01..=0x7F`) and Strix Point (`0x96`, rev `0x01..=0x3F`).
       - Fallback: `FSR 3.1.6` / `2.3.4` on non-AMD GPUs or devices lacking required features.
     - Runtime dispatch invariants: single-threaded context, maximum 15 frames in flight, all inputs in `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`, output in `VK_IMAGE_LAYOUT_GENERAL`, valid finite jitter/motion-vector scales, consistent depth inversion (`near`/`far`).

2. **8-GPU × 2-OS Hardware Simulation Matrix**:
   - AMD Radeon RX 9070 XT (RDNA 4 / Navi 48 / `1002:7550`) — Windows Adrenalin 26.10.1 & Linux Mesa RADV
   - AMD Radeon RX 7900 XTX (RDNA 3 / Navi 31 / `1002:744C`) — Windows & Linux RADV
   - AMD Radeon RX 6800 XT (RDNA 2 / Navi 21 / `1002:73BF`) — Windows & Linux RADV
   - AMD Radeon 890M Strix Point iGPU (RDNA 3.5 / GFX1150 / `1002:150E`) — Windows & Linux RADV
   - NVIDIA GeForce RTX 5090 (Blackwell / GB202 / `10DE:2B85`) — Windows & Linux
   - NVIDIA GeForce RTX 4090 (Ada Lovelace / AD102 / `10DE:2684`) — Windows & Linux
   - NVIDIA GeForce RTX 3050 Mobile (Ampere / GA107M / `10DE:25A2`) — Windows & Linux
   - Intel Arc B580 (Battlemage / Xe2 / `8086:E20B`) — Windows & Linux ANV

---

## 3. Crate Architecture (`crates/vulkanshaders-hwsim-bench`)

Zero new external crates are added; the crate uses only existing workspace dependencies (`vulkanshaders-core`, `ash`, `rspirv`, `spirv`, `bytemuck`, `serde`, `serde_json`, `clap`, `blake3`, `bitflags`, `thiserror`, `tracing`, `tracing-subscriber`, `parking_lot`).

```text
crates/vulkanshaders-hwsim-bench/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs           # Public API re-exports
│   ├── main.rs          # clap 4.5 CLI (simulate, matrix, audit-spirv, abi-check, bench, fuzz, full)
│   ├── error.rs         # thiserror 2.0 domain error types
│   ├── profiles.rs      # 8 GPU × 2 OS profiles, Vulkan feature/coop-matrix tables, FSR 4.1.2 selector
│   ├── mock_icd.rs      # Thread-isolated in-process Mock Vulkan ICD with real dispatchable structs
│   ├── fsr4_sim.rs      # Clean-room FFX SDK 2.4 / FSR 4.1.2 provider + live contract simulator
│   ├── abi_layout.rs    # Compile-time offset_of! + SysV64 LP64 vs Win64 LLP64 layout verifier
│   ├── spirv_audit.rs   # Pure-Rust SPIR-V decoder, PE32+/ELF64 carver, compressed blob detector
│   ├── bench_fuzz.rs    # Deterministic BLAKE3-XOF property fuzzer + p50/p95/p99/MAD micro-bench
│   └── report.rs        # 4-status epistemic JSON + Markdown report generator with BLAKE3 digest
└── tests/
    └── hwsim_integration.rs  # End-to-end tests across all 8 GPUs × 2 OSes
```

---

## 4. Mandatory Engineering & Safety Safeguards

1. **Zero UB at FFI boundaries**: Every `extern "system"` / `extern "C"` mock entrypoint wraps its body in `std::panic::catch_unwind(AssertUnwindSafe(...))` and latches any panic onto per-instance state.
2. **Valid dispatchable handles**: Dispatchable Vulkan handles (`VkInstance`, `VkPhysicalDevice`, `VkDevice`, `VkQueue`, `VkCommandBuffer`) are real `#[repr(C)]` heap allocations starting with `loader_magic: usize = 0x01CDC0DE`, a `HandleKind` discriminant, and a `generation: u64` tag (never raw small integers).
3. **Raw-pointer `pNext` traversal**: `pNext` chains are traversed strictly with aligned raw pointers, visited-address cycle detection, and a hard depth limit of 64.
4. **Parallel `cargo test` isolation**: Per-instance state is owned by the dispatchable handle structs and bound via `thread_local!` guards; `static mut` is forbidden.
5. **Cross-platform FFI safety**: `[std::ffi::c_char; 256]` strings are zero-initialized and NUL-terminated; `FfxWchar` is `u16` on Windows and `u32` on non-Windows; new `#[repr(C)]` FFI structs use `u8`/`u32` instead of raw Rust `bool`.
6. **Honest SPIR-V carving**: Every `0x07230203` match is validated against the 5-word SPIR-V header and opcode stream; zstd/zlib/lz4 headers in vendor binaries are reported as `INCONCLUSIVE_UNINSPECTED` instead of a silent pass.
7. **Rust 2024 + `ash 0.38.0+1.3.281` compliance**: Explicit `unsafe { ... }` inside `unsafe fn`, no `::builder()`/`.build()`, no `ash::prelude`, explicit `ash::vk::Handle` imports, `ash` without `loaded`/`linked` requirement in `vulkanshaders-hwsim-bench`, and custom constants for Vulkan 1.4 / `VK_EXT_shader_float8` types absent in `ash 0.38.0+1.3.281`.
8. **Clean-room & licensing compliance**: `GPL-3.0-only` with workspace `LICENSE-EXCEPTION.md`; zero committed vendor DLLs, SOs, or extracted shader blobs; synthetic-only unit test fixtures.
