# `vulkanshaders-hwsim-bench`

Hardware-free Vulkan ICD simulator, AMD FSR 4.1.2 / FidelityFX SDK 2.4.0 contract verifier, cross-ABI (`SysV64 LP64` vs `Win64 LLP64`) layout auditor, pure-Rust SPIR-V capability scanner, and deterministic property fuzzer / micro-benchmark harness.

This crate is standalone and leaves `crates/vulkanshaders-sdk-bench` untouched. It adds zero new external crate dependencies beyond the workspace `Cargo.toml`.

## Epistemic Scope

Every report partitions checks into four explicit statuses:
- `PROVEN_OFFLINE`: Struct sizes/alignments/field offsets (`offset_of!`), SDK 2.4.0 `desc_type` constants, `pNext` cycle/depth/extension gating, `VkImageLayout` state transitions, handle leak/double-free counts, and FSR 4.1.2 15-frame in-flight cap.
- `STATIC_HEURISTIC`: Modeled 8-GPU x 2-OS hardware capability profiles and uncompressed SPIR-V capability/cooperative-matrix compatibility checks.
- `INCONCLUSIVE_UNINSPECTED`: Compressed shader payloads (zstd/zlib/lz4) inside vendor binaries that cannot be parsed without decompression.
- `REQUIRES_PHYSICAL_HARDWARE`: Driver SPIR-V-to-ISA codegen, cache coherency, real FP8/INT8 neural network image quality, and real GPU frame time.

## Commands

```bash
# Run unit & integration tests (no GPU or libvulkan.so required)
cargo test -p vulkanshaders-hwsim-bench

# Print the 8-GPU x 2-OS hardware profile & FSR 4.1.2 network selection matrix
cargo run -p vulkanshaders-hwsim-bench -- matrix

# Run Mock Vulkan ICD + FSR 4.1.2 contract simulation on all 16 GPU x OS profiles
cargo run -p vulkanshaders-hwsim-bench -- simulate --gpu all --os all

# Run compile-time offset_of! + SysV64/Win64 layout & SDK 2.4.0 constant audit
cargo run -p vulkanshaders-hwsim-bench -- abi-check

# Audit a standalone .spv or vendor .dll/.so binary without executing it
cargo run -p vulkanshaders-hwsim-bench -- audit-spirv /path/to/amd_fidelityfx_upscaler_vk.dll

# Run deterministic property fuzzing and CPU-side micro-benchmarks
cargo run -p vulkanshaders-hwsim-bench -- fuzz --seed 42 --cases 512
cargo run -p vulkanshaders-hwsim-bench -- bench --iters 5000

# Run the complete pipeline and write JSON report
cargo run -p vulkanshaders-hwsim-bench -- --format json --out hwsim-report.json full
```
