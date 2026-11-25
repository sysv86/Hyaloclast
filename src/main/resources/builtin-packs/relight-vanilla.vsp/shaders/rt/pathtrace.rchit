#version 460
#extension GL_EXT_ray_tracing : require

#include <vulkanshaders/v1/common.vsl>
#include <vulkanshaders/v1/lighting.vsl>

#if VS_ACTIVE_RENDER_TIER >= 2

layout(set = 1, binding = 1, std430) readonly buffer InstanceMetaSSBO {
    VsInstanceMeta instances[];
} uInstanceMeta;

layout(set = 1, binding = 2, std140) uniform SharcConfigUBO {
    SharcConfig cfg;
} uSharc;

layout(set = 2, binding = 0) uniform sampler2D uBindlessTextures[];

layout(location = 0) rayPayloadInEXT VsPathPayload payload;
hitAttributeEXT vec2 barycentrics;

void main() {
    uint bounce = (payload.flags >> 8u) & 0xFFu;
    float hitDist = gl_HitTEXT;
    vec3 rayDir = normalize(gl_WorldRayDirectionEXT);
    vec3 hitPosRel = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * hitDist;
    vec3 V = -rayDir;

    VsSurface s = VsSurface(vec3(0.65), -rayDir, 0.80, 0.0, 0.0, 1.0, 0.0, 0.04);
    uint flags = 0u;
    float skyLight = 1.0;
    float blockLight = 0.0;
    float hitAlpha = 1.0;
    bool isGlass = false;
    bool isDhLod = false;
    vec3 geometryNormal = V;
    float labPbrPorosity = 0.5;
    bool hasLabPbrPorosity = false;

    uint instIdx = gl_InstanceCustomIndexEXT;
    if (instIdx < uInstanceMeta.instances.length()) {
        VsInstanceMeta meta = uInstanceMeta.instances[instIdx];
        if (vsPackU64(meta.vertexBda) != 0ul) {
            uvec3 idx = vsLoadTriangleIndices(meta, gl_PrimitiveID);
            VsDecodedVertex v0 = vsLoadVertex(meta, idx.x);
            VsDecodedVertex v1 = vsLoadVertex(meta, idx.y);
            VsDecodedVertex v2 = vsLoadVertex(meta, idx.z);

            float w0 = 1.0 - barycentrics.x - barycentrics.y;
            vec2 hitUv = v0.uv0 * w0 + v1.uv0 * barycentrics.x + v2.uv0 * barycentrics.y;
            vec4 hitColor = v0.color * w0 + v1.color * barycentrics.x + v2.color * barycentrics.y;

            if (meta.layerOrdinal == 5u) {
                vec4 decal = textureLod(uBindlessTextures[nonuniformEXT(meta.bindlessAlbedoSlot)], hitUv, 0.0);
                vsPayloadSetRadianceRough(payload, pow(clamp(decal.rgb * 2.0, vec3(0.0), vec3(1.0)), vec3(2.2)), 0.8);
                vsPayloadSetPbr0(payload, 0.0, 1.0, 0.0, decal.a);
                vsPayloadSetPbr1(payload, 0.0, 0.04, decal.a, 0.0);
                payload.flags = 1u;
                return;
            }

            isDhLod = (meta.layerOrdinal == 4u);
            vec3 rawGeomN = mat3(gl_ObjectToWorldEXT) * cross(v1.pos - v0.pos, v2.pos - v0.pos);
            vec3 geomN = dot(rawGeomN, rawGeomN) > 1e-10 ? normalize(rawGeomN) : vec3(0.0, 1.0, 0.0);
            if (dot(geomN, V) < 0.0) {
                geomN = -geomN;
            }
            geometryNormal = geomN;
            vec3 rawInterpN = mat3(gl_ObjectToWorldEXT) * (
                v0.normal * w0 + v1.normal * barycentrics.x + v2.normal * barycentrics.y
            );
            vec3 interpN = dot(rawInterpN, rawInterpN) > 1e-8 ? normalize(rawInterpN) : geomN;
            if ((!isDhLod && meta.vertexStride < 36u) || dot(interpN, geomN) <= 0.0 || dot(interpN, V) <= 0.0) {
                interpN = geomN;
            }
            vec4 interpT = v0.tangent * w0 + v1.tangent * barycentrics.x + v2.tangent * barycentrics.y;
            vec3 rawInterpT = mat3(gl_ObjectToWorldEXT) * interpT.xyz;
            interpT.xyz = dot(rawInterpT, rawInterpT) > 1e-8 ? normalize(rawInterpT) : vec3(0.0);

            flags = v0.materialFlags;
            bool water = vsIsWater(flags);

            // Fluid sprites encode still/flowing animation, so use the biome tint for optical shading.
            vec4 texel;
            if (isDhLod) {
                vec4 lodTint = unpackUnorm4x8(meta.bindlessNormalSlot != 0u ? meta.bindlessNormalSlot : 0xFFFFFFFFu);
                texel = hitColor * lodTint;
            } else if (water) {
                texel = hitColor;
            } else {
                uint albedoSlot = meta.bindlessAlbedoSlot;
                texel = textureLod(uBindlessTextures[nonuniformEXT(albedoSlot)], hitUv, 0.0) * hitColor;
            }
            hitAlpha = ((flags >> 3u) & 15u) == 9u ? 1.0 : texel.a;
            uint materialId = (flags >> 3u) & 15u;
            isGlass = !water && !isDhLod && ((materialId == 7u) || (meta.layerOrdinal == 2u && texel.a > 0.02 && texel.a < 0.98));

            blockLight = clamp((float(v0.uv2.x) * w0 + float(v1.uv2.x) * barycentrics.x + float(v2.uv2.x) * barycentrics.y) / 240.0, 0.0, 1.0);
            skyLight = clamp((float(v0.uv2.y) * w0 + float(v1.uv2.y) * barycentrics.x + float(v2.uv2.y) * barycentrics.y) / 240.0, 0.0, 1.0);

            s.albedo = pow(max(texel.rgb, vec3(0.0)), vec3(2.2));
            s.normal = interpN;
            s.roughness = water ? 0.06 : (isGlass ? 0.05 : (((flags & 1u) != 0u) ? 0.72 : 0.78));
            s.emission = !water && (flags & 4u) != 0u ? 1.5 : 0.0;
            s.ao = 1.0;
            s.subsurface = ((flags & 1u) != 0u) ? 0.45 : 0.0;
            s.f0 = water ? 0.02 : (isGlass ? 0.042 : 0.04);

            // Run pack surface hook BEFORE LabPBR so authored LabPBR _s textures are never overwritten.
            vs_dispatch_hook_surface(s, hitPosRel + uSharc.cfg.cameraPosition.xyz, flags, bounce);

            bool hasNorm = !water && !isDhLod && meta.bindlessNormalSlot != 0u;
            bool hasSpec = !water && !isDhLod && meta.bindlessSpecularSlot != 0u;
            if (hasNorm || hasSpec) {
                float mipLod = clamp(log2(max(hitDist * 0.08, 1.0)), 0.0, 3.0);
                vec4 nSample = hasNorm ? textureLod(uBindlessTextures[nonuniformEXT(meta.bindlessNormalSlot)], hitUv, mipLod) : vec4(0.5, 0.5, 1.0, 1.0);
                vec4 sSample = hasSpec ? textureLod(uBindlessTextures[nonuniformEXT(meta.bindlessSpecularSlot)], hitUv, mipLod) : vec4(0.0, 0.04, 0.0, 1.0);
                vsApplyLabPbr(nSample, sSample, interpN, interpT, hasNorm, hasSpec, s, labPbrPorosity);
                hasLabPbrPorosity = hasSpec;
            }
        }
    }

    bool isWater = vsIsWater(flags);
    if (isWater) {
        s.normal = geometryNormal;
        float waveStrength = 0.35;
        float waveSpeed = 1.8;
        float waveScale = 2.2;
        vs_dispatch_hook_water_waves(waveStrength, waveSpeed, waveScale);
        vec3 wavePos = hitPosRel + uSharc.cfg.cameraPosition.xyz;
        float waveHeight = sin(wavePos.x * waveScale + vsWorld.gameTimeSec * waveSpeed) * waveStrength;
        vsWaterSkyLight = skyLight;
        vs_dispatch_hook_water_surface(s, wavePos, waveHeight);
        // Horizontal surfaces only: vertical waterfall faces keep their flat per-face normal.
        if (abs(s.normal.y) > 0.5) {
            s.normal = vsFaceWaterNormal(s.normal, V);
        }
    } else if (bounce == 0u && !isGlass && !isDhLod && vsDimensionId() == 0 && vsWorld.rain > 0.01) {
        vec3 worldAbs = hitPosRel + uSharc.cfg.cameraPosition.xyz;
        float wetStrength = clamp(VS_PARAM_WETNESS_STRENGTH, 0.0, 1.0);
        if (hasLabPbrPorosity) {
            vsApplyRainWetness(s, worldAbs, skyLight, vsWorld.rain, wetStrength, labPbrPorosity);
        } else {
            vsApplyRainWetness(s, worldAbs, skyLight, vsWorld.rain, wetStrength);
        }
    }

    vsPayloadSetRadianceRough(payload, s.albedo, s.roughness);
    payload.hitT = hitDist;
    vsPayloadSetNormal(payload, s.normal);
    vsPayloadSetWaterNormal(payload, geometryNormal);
    vsPayloadSetPbr0(payload, s.metallic, skyLight, blockLight, s.ao);
    vsPayloadSetPbr1(payload, s.subsurface, s.f0, hitAlpha, s.emission);
    payload.flags = 1u | (isWater ? 4u : 0u) | (isGlass ? 64u : 0u) | (bounce << 8u);
}

#else

void main() {}

#endif
