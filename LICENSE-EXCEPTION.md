# Additional Permissions under GPLv3 Section 7

In addition to the terms of the GNU General Public License, version 3 (GPL-3.0), the copyright holders of **VulkanShaders (Hyaloclast / IGNIS)** grant the following additional permissions pursuant to Section 7 of GPL-3.0:

1. **Proprietary Upscaler and Frame Generation Runtimes**
   You are granted permission to link, load at runtime, and distribute VulkanShaders alongside proprietary vendor libraries required for hardware upscaling, ray reconstruction, low-latency, or frame generation (including NVIDIA NGX / DLSS / Streamline / Reflex / NRD, Intel XeSS, and AMD FidelityFX SDK binaries), without requiring the source code of those proprietary libraries to be disclosed under GPL-3.0.

2. **Independent Shader Pack Licensing**
   Authors of `.vsp` shader packages are granted permission to distribute their shader packs under any license of their choice, provided the shader pack interacts with VulkanShaders exclusively through the public `<vulkanshaders/v1/...>` include headers, `pack.toml` manifest schema, and standard `hooks/*.vsl` hook stages.

---

*Note: This GPLv3 Section 7 additional-permissions structure is carried over and adapted from [Radiante's `LICENSE-EXCEPTION.md`](https://github.com/Gabrieli2806/Radiante/blob/main/LICENSE-EXCEPTION.md) (by Gabrieli2806, with contributors Maximiliano-Santana and SinanAkkoyun), which served as an important reference for VulkanShaders.*
