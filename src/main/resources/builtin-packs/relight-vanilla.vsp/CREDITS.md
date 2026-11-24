# Credits & Third-Party Notices — Sunlit Vanilla (Hyaloclast)

Sunlit Vanilla is a built-in VulkanShaders (Hyaloclast) path-tracing and ray-query shader pack.
Selected optical formulas, material calibration concepts, and lighting policies are adapted from
the following permissive open-source projects:

## 1. RTX Remix for Minecraft — Apache License 2.0
- **Copyright**: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
- **License**: Apache License, Version 2.0 (`http://www.apache.org/licenses/LICENSE-2.0`)
- **Adapted concepts**:
  - Kasten-Young optical air-mass atmospheric transmittance curve and Rayleigh/Mie extinction weights
    (`SkyLightRenderer.java`) used in `hooks/sky.vsl`, `hooks/fog.vsl`, and `shaders/compute/volumetric_fog.comp`.
  - Underwater open-flame held-light extinguishing policy (`HeldLightPolicy.java`) used in
    `WorldStateCollector.kt`.
  - Dusk/night exposure floor and selective bloom calibration (`dusk.conf`, `night.conf`, `rtx.conf`)
    used in `shaders/compute/post_tonemap.comp`.
  - LabPBR roughness/metallic/SSS decoding conventions (`LabPBRConverter.java`) and non-blooming
    restrained emission discipline (`BuiltinEmissionPolicy.java`).

## 2. Minecraft Miniature Shader — MIT License
- **Copyright**: Copyright (c) 2022 Mateus A. Kreuch
- **License**: MIT License
- **Adapted concepts**:
  - Three-stop warm blackbody torchlight color ramp (`TORCH_INNER_COLOR` / `TORCH_MIDDLE_COLOR` /
    `TORCH_OUTER_COLOR`) and ambient-aware torch contrast (`common/getTorchColor.fsh`) adapted for
    linear HDR shading in `hooks/block_light.vsl`.
  - Grounded non-plastic block reflectivity and roughness calibration philosophy
    (`common/blockReflectivity.glsl`) adapted for natural stone, wood, metal, and ice materials in
    `hooks/surface.vsl`.

Full license texts are included in `THIRD_PARTY_NOTICES.txt`.
