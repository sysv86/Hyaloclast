# Credits & Acknowledgements

## Radiante (Important Architectural Reference & License Lineage)

- **Project:** [Radiante](https://github.com/Gabrieli2806/Radiante) ([Modrinth](https://modrinth.com/project/radiante))
- **Author:** Gabrieli2806
- **Contributors:** Maximiliano-Santana, SinanAkkoyun
- **License:** GNU General Public License v3.0 (`GPL-3.0`) with Section 7 Additional Permissions (`LICENSE-EXCEPTION.md`)
- **Attribution:** Radiante was an important architectural reference for **VulkanShaders (Hyaloclast)**, particularly for hooking Minecraft 26.3's native RenderPearl Vulkan backend (`VulkanInstance`, `VulkanBackend`, `VulkanDevice`, reentrant `VkQueue` locking, `VulkanCommandEncoder`, and `VK_NV_low_latency2` Reflex swapchain pacing). VulkanShaders adopts and carries over Radiante's `GPL-3.0` license (`LICENSE`) and Section 7 additional-permissions structure (`LICENSE-EXCEPTION.md`).
- **Upstream Lineage:** Radiante itself originated as a fork of **Radiance** and its **MCVR** renderer (GPL-3.0).

## Built-in Shader Pack (`hyaloclast-sunlit-vanilla.vsp`)

See `src/main/resources/builtin-packs/hyaloclast-sunlit-vanilla.vsp/CREDITS.md` and `THIRD_PARTY_NOTICES.txt` for shader-level attributions (including RTX Remix for Minecraft — Apache-2.0, NVIDIA; and Minecraft Miniature Shader — MIT, Mateus A. Kreuch).
