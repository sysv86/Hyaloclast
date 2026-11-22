#version 330
#extension GL_ARB_separate_shader_objects : require

#include <minecraft:fog.glsl>
#include <minecraft:globals.glsl>

#ifndef VS_PARAM_POM_DEPTH
#define VS_PARAM_POM_DEPTH 0.0
#endif

uniform sampler2D Sampler0;
uniform sampler2D VsNormalAtlas;
uniform sampler2D VsSpecularAtlas;

layout(std140) uniform VsMotion {
    mat4 prevViewProj;
    vec4 cameraDeltaTime;
    vec4 pbrParams;
    vec4 sunDirRain;
    vec4 heldLightData;
    mat4 invView;
    vec4 packParams[10];
} vsMotion;

layout(location = 0) in float sphericalVertexDistance;
layout(location = 1) in float cylindricalVertexDistance;
layout(location = 2) in vec4 vertexColor;
layout(location = 3) in vec2 texCoord0;
layout(location = 4) in float chunkVisibility;
layout(location = 5) in vec3 worldNormal;
layout(location = 6) in vec4 curClipPos;
layout(location = 7) in vec4 prevClipPos;
layout(location = 8) flat in uint materialFlags;
layout(location = 9) in vec2 blockSkyLight;
layout(location = 10) in vec4 worldTangent;
layout(location = 11) in vec3 camRelPos;
layout(location = 12) in vec2 midUv;

// 4-target MRT G-Buffer outputs:
// Attachment 0: Clean Unfogged Surface Albedo (mainRenderTarget colorTexture)
// Attachment 1: World Normal (encoded 0..1) + Roughness (> 0 marks terrain MRT pixels; > 1 marks water)
// Attachment 2: Metallic / clamp(emission * 0.25, 0..1) / pack4x4(skyLight, blockLight) / pack4x4(sqrt(f0 / 0.9), subsurface)
// Attachment 3: Screen-space Velocity (gVelocity RG16F)
layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 outNormalRoughness;
layout(location = 2) out vec4 outMaterialPbr;
layout(location = 3) out vec2 outVelocity;

float vsPackNibble2(float hi, float lo) {
    float h = floor(clamp(hi, 0.0, 1.0) * 15.0 + 0.5);
    float l = floor(clamp(lo, 0.0, 1.0) * 15.0 + 0.5);
    return (h * 16.0 + l) / 255.0;
}

// LabPBR 1.3 hardcoded linear metal F0 RGB table for specular green codes 230..237:
// 230 Iron, 231 Gold, 232 Aluminum, 233 Chrome, 234 Copper, 235 Lead, 236 Platinum, 237 Silver.
vec3 labPbrHardcodedMetalF0(int code) {
    const vec3 METALS[8] = vec3[8](
        vec3(0.53123, 0.51236, 0.49583),
        vec3(0.94423, 0.77610, 0.37340),
        vec3(0.91230, 0.91385, 0.91968),
        vec3(0.55560, 0.55454, 0.55478),
        vec3(0.92595, 0.72090, 0.50415),
        vec3(0.63248, 0.62594, 0.64148),
        vec3(0.67885, 0.64240, 0.58841),
        vec3(0.96200, 0.94947, 0.92212)
    );
    return METALS[clamp(code - 230, 0, 7)];
}

// 4-tap rotated grid supersampling with hardware trilinear mip interpolation.
vec4 sampleCrispTexel(sampler2D tex, vec2 uv, vec2 du, vec2 dv) {
    vec2 texDims = max(vec2(textureSize(tex, 0)), vec2(1.0));
    vec2 pixelSize = 1.0 / texDims;
    vec2 footprint = max(sqrt(du * du + dv * dv), vec2(1e-6));

    float minPixel = min(pixelSize.x, pixelSize.y);
    float rgssBlend = 0.0;
    if (UseRgss != 0) {
        float maxFootprint = max(footprint.x, footprint.y);
        rgssBlend = smoothstep(minPixel, minPixel * 2.0, maxFootprint);
    }

    vec4 nearestSample = vec4(0.0);
    if (rgssBlend < 0.999) {
        vec2 uvTexels = uv * texDims;
        vec2 texelEdge = round(uvTexels);
        vec2 texelDelta = uvTexels - texelEdge;
        vec2 sharpOffset = clamp(texelDelta * (pixelSize / footprint) + 0.5, 0.0, 1.0);
        vec2 sharpUv = (texelEdge - 0.5 + sharpOffset) * pixelSize;
        nearestSample = textureGrad(tex, sharpUv, du, dv);
        if (rgssBlend <= 0.001) {
            return nearestSample;
        }
    }

    float duLen = length(du);
    float dvLen = length(dv);
    float effDeriv = sqrt(max(duLen * dvLen, 1e-12));
    float mipExact = max(0.0, log2(effDeriv / minPixel));

    const vec2 rgssOffsets[4] = vec2[4](
        vec2( 0.125,  0.375),
        vec2(-0.125, -0.375),
        vec2( 0.375, -0.125),
        vec2(-0.375,  0.125)
    );

    vec4 rgssSample = vec4(0.0);
    for (int i = 0; i < 4; ++i) {
        vec2 sampleUv = uv + rgssOffsets[i] * pixelSize;
        rgssSample += textureLod(tex, sampleUv, mipExact);
    }
    rgssSample *= 0.25;
    return rgssBlend >= 0.999 ? rgssSample : mix(nearestSample, rgssSample, rgssBlend);
}

void main() {
    bool isFoliage = (materialFlags & 1u) != 0u;
    bool isFluid = (materialFlags & 2u) != 0u;
    bool isEmissive = (materialFlags & 4u) != 0u;
    uint matId = (materialFlags >> 3u) & 15u;
    bool isWater = isFluid && !isEmissive && (matId != 9u);

    vec3 N = dot(worldNormal, worldNormal) > 1e-8 ? normalize(worldNormal) : vec3(0.0, 1.0, 0.0);
    vec3 fallbackT = abs(N.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 0.0, 1.0);
    vec3 T = worldTangent.xyz - N * dot(N, worldTangent.xyz);
    T = dot(T, T) > 1e-8 ? normalize(T) : normalize(fallbackT - N * dot(N, fallbackT));
    vec3 B = cross(N, T) * (worldTangent.w >= 0.0 ? 1.0 : -1.0);
    mat3 tbn = mat3(T, B, N);

    vec2 du = dFdx(texCoord0);
    vec2 dv = dFdy(texCoord0);
    vec2 uv = texCoord0;

    // Optional 8-step Steep Parallax Occlusion Mapping using LabPBR _n.a height channel
    float pomDepth = clamp(float(VS_PARAM_POM_DEPTH), 0.0, 0.08);
    float distToCam = length(camRelPos);
    if (pomDepth > 0.001 && vsMotion.pbrParams.x > 0.5 && !isWater && distToCam > 0.05 && distToCam < 24.0) {
        vec3 viewDirWorld = -camRelPos / distToCam;
        vec3 viewDirTan = transpose(tbn) * viewDirWorld;
        if (viewDirTan.z > 0.08) {
            vec2 texDims = max(vec2(textureSize(Sampler0, 0)), vec2(1.0));
            vec2 halfTexel = 0.5 / texDims;
            float posDeriv = max(max(length(dFdx(camRelPos)), length(dFdy(camRelPos))), 1e-5);
            float uvDeriv = max(length(du), length(dv));
            float uvPerBlock = clamp(uvDeriv / posDeriv, 1.0 / max(texDims.x, texDims.y), 0.25);

            vec2 halfTile = max(abs(texCoord0 - midUv), vec2(uvPerBlock * 0.5));
            bool hasMidUv = (midUv.x > 1e-5 || midUv.y > 1e-5);
            vec2 uvMin = hasMidUv ? (midUv - halfTile + halfTexel) : vec2(0.0);
            vec2 uvMax = hasMidUv ? (midUv + halfTile - halfTexel) : vec2(1.0);

            float distFade = 1.0 - smoothstep(16.0, 24.0, distToCam);
            const int POM_STEPS = 8;
            float layerDepth = 1.0 / float(POM_STEPS);
            vec2 totalShift = (viewDirTan.xy / max(viewDirTan.z, 0.25)) * (pomDepth * distFade * uvPerBlock);
            vec2 stepUv = totalShift * layerDepth;

            float curLayer = 0.0;
            vec2 curUv = texCoord0;
            vec2 prevUv = curUv;
            float rawH = textureGrad(VsNormalAtlas, curUv, du, dv).a;
            float curDepthMap = 1.0 - (rawH > 0.001 ? rawH : 1.0);
            float prevDepthMap = curDepthMap;

            for (int i = 0; i < POM_STEPS; ++i) {
                if (curLayer >= curDepthMap) break;
                prevUv = curUv;
                prevDepthMap = curDepthMap;
                curUv = clamp(curUv - stepUv, uvMin, uvMax);
                curLayer += layerDepth;
                float h = textureGrad(VsNormalAtlas, curUv, du, dv).a;
                curDepthMap = 1.0 - (h > 0.001 ? h : 1.0);
            }

            float afterDepth = curDepthMap - curLayer;
            float beforeDepth = prevDepthMap - (curLayer - layerDepth);
            float denom = afterDepth - beforeDepth;
            float weight = abs(denom) > 1e-5 ? clamp(afterDepth / denom, 0.0, 1.0) : 0.0;
            uv = clamp(mix(curUv, prevUv, weight), uvMin, uvMax);
        }
    }

    vec4 color = sampleCrispTexel(Sampler0, uv, du, dv) * vertexColor;
#ifdef ALPHA_CUTOUT
    if (color.a < ALPHA_CUTOUT) {
        discard;
    }
#endif
    // Preserve pure surface albedo in G-Buffer Attachment 0; atmospheric and volumetric fog
    // are evaluated once in the deferred / ray-tracing lighting passes.
    fragColor = vec4(color.rgb, color.a);

    float roughness = isWater ? 0.06 : (isFoliage ? 0.72 : (isEmissive ? 0.45 : 0.78));
    float metallic = 0.0;
    float f0 = 0.04;
    float emission = isEmissive ? 1.0 : 0.0;
    float subsurface = (isFoliage || matId == 4u) ? 0.56 : 0.0;

    // Run pack surface hook BEFORE LabPBR so authored LabPBR _s textures are never overwritten
    // by heuristic materialId clamps.
    // @hook surface

    // LabPBR 1.3 companion atlases: _n = (tangentNormal.xy, AO, height), _s = (smoothness, F0/metal, porosity/SSS, emission).
    // Allowed on lava (!isWater instead of !isFluid).
    if (vsMotion.pbrParams.x > 0.5 && !isWater) {
        vec4 nSample = textureGrad(VsNormalAtlas, uv, du, dv);
        vec4 sSample = textureGrad(VsSpecularAtlas, uv, du, dv);

        vec2 normalXY = nSample.rg * 2.0 - 1.0;
        if (dot(normalXY, normalXY) < 1.2) {
            float normalZ = sqrt(max(0.0, 1.0 - dot(normalXY, normalXY)));
            vec3 tangentNormal = vec3(normalXY, normalZ);
            if (dot(tangentNormal, tangentNormal) > 1e-5) {
                N = normalize(tbn * normalize(tangentNormal));
            }
        }

        // Detect missing _s sprite sentinel (0, 0, 0, 255) so unauthored textures keep pack defaults.
        bool hasAuthoredSpec = !(sSample.r < 0.002 && sSample.g < 0.002 && sSample.b < 0.002 && sSample.a >= (254.5 / 255.0));
        bool hardcodedMetal = false;
        if (hasAuthoredSpec) {
            float perceptualSmoothness = clamp(sSample.r, 0.0, 1.0);
            roughness = max(0.04, (1.0 - perceptualSmoothness) * (1.0 - perceptualSmoothness));

            int gCode = int(floor(sSample.g * 255.0 + 0.5));
            if (gCode >= 230 && gCode <= 237) {
                vec3 metalF0Lin = labPbrHardcodedMetalF0(gCode);
                metallic = 1.0;
                f0 = max(metalF0Lin.r, max(metalF0Lin.g, metalF0Lin.b));
                color.rgb = pow(clamp(metalF0Lin, vec3(0.0), vec3(1.0)), vec3(1.0 / 2.2));
                fragColor.rgb = color.rgb;
                hardcodedMetal = true;
            } else if (gCode >= 238) {
                metallic = 1.0;
                f0 = 0.90;
            } else {
                metallic = 0.0;
                f0 = clamp(sSample.g, 0.02, 0.90);
            }

            if (sSample.b > (64.5 / 255.0)) {
                subsurface = max(subsurface, clamp((sSample.b - 64.5 / 255.0) / (190.5 / 255.0), 0.0, 1.0));
            }
            if (sSample.a < (254.5 / 255.0)) {
                emission = max(emission, sSample.a * (255.0 / 254.0) * 2.2);
            }
        }

        // Apply LabPBR _n.b AO in sRGB-compensated space to avoid ao^2.2 double-darkening.
        float ao = clamp(nSample.b, 0.0, 1.0);
        if (!hardcodedMetal && ao < 0.998) {
            color.rgb *= mix(1.0, pow(max(ao, 0.05), 1.0 / 2.2), 0.72);
            fragColor.rgb = color.rgb;
        }
    }

    roughness = clamp(roughness, 0.02, 1.0);

    vec2 curNdc = curClipPos.xy / max(abs(curClipPos.w), 1e-6);
    vec2 prevNdc = prevClipPos.xy / max(abs(prevClipPos.w), 1e-6);
    vec2 velocity = (curNdc - prevNdc) * 0.5;

    // Keep a dedicated >1 alpha marker for water. Lighting shaders decode alpha - 1 back to
    // roughness; without this, RT treated every visible water pixel as an independent hidden
    // water ray and produced block-shaped reflection patches.
    if (isWater) { roughness = min(roughness, 0.04); }
    float encodedRoughness = clamp(roughness, 0.02, 1.0) + (isWater ? 1.0 : 0.0);
    outNormalRoughness = vec4(N * 0.5 + 0.5, encodedRoughness);
    outMaterialPbr = vec4(
        clamp(metallic, 0.0, 1.0),
        clamp(emission * 0.25, 0.0, 1.0),
        vsPackNibble2(blockSkyLight.y, blockSkyLight.x),
        vsPackNibble2(sqrt(clamp(f0 / 0.9, 0.0, 1.0)), clamp(subsurface, 0.0, 1.0))
    );
    outVelocity = velocity;
    // @hook gbuffer_pack
}
