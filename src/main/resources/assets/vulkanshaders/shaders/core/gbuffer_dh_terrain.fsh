#version 330
#extension GL_ARB_separate_shader_objects : require

// Distant Horizons LOD terrain G-Buffer fragment shader.
// Outputs 3 MRTs matching Hyaloclast's unified G-Buffer PBR encoding contract.

layout(std140) uniform fragUniformBlock {
    float uClipDistance;
    float uNoiseIntensity;
    int uNoiseSteps;
    int uNoiseDropoff;
    int uDitherDhRendering;
    int uNoiseEnabled;
};

uniform sampler2D uLightMap;
uniform sampler2D uBlockAtlas;

layout(location = 0) in vec4 vVertColor;
layout(location = 1) in vec3 vWorldNormal;
layout(location = 2) in vec3 vCamRelPos;
layout(location = 3) flat in uint vDhMaterialId;
layout(location = 4) in vec2 vBlockSkyLight;

// 3-target MRT G-Buffer outputs:
// Attachment 0: Pure unfogged surface albedo
// Attachment 1: Encoded world normal (0..1) + roughness (+1.0 watermark for water)
// Attachment 2: Metallic / clamp(emission * 0.25, 0..1) / pack4x4(skyLight, blockLight) / pack4x4(sqrt(f0 / 0.9), subsurface)
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outNormalRoughness;
layout(location = 2) out vec4 outMaterialPbr;

float vsPackNibble2(float hi, float lo) {
    float h = floor(clamp(hi, 0.0, 1.0) * 15.0 + 0.5);
    float l = floor(clamp(lo, 0.0, 1.0) * 15.0 + 0.5);
    return (h * 16.0 + l) / 255.0;
}

float dhInterleavedGradientNoise(vec2 px) {
    return fract(52.9829189 * fract(dot(px, vec2(0.06711056, 0.00583715))));
}

float dhHash31(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

void main() {
    float dist = length(vCamRelPos);
    if (uClipDistance > 0.0) {
        if (uDitherDhRendering != 0) {
            float ditherZone = max(uClipDistance * 0.12, 8.0);
            float t = clamp((dist - (uClipDistance - ditherZone)) / ditherZone, 0.0, 1.0);
            if (t <= dhInterleavedGradientNoise(gl_FragCoord.xy)) {
                discard;
            }
        } else if (dist < uClipDistance) {
            discard;
        }
    }

    vec4 color = vVertColor;

    // Optional subtle procedural micro-variation to soften flat distant LOD faces.
    if (uNoiseEnabled != 0 && uNoiseIntensity > 0.001 && vDhMaterialId != 12u) {
        float dropoff = max(float(uNoiseDropoff), 32.0);
        float noiseFade = 1.0 - smoothstep(dropoff * 0.5, dropoff, dist);
        if (noiseFade > 0.01) {
            int steps = max(uNoiseSteps, 1);
            vec3 cell = floor(vCamRelPos * float(steps));
            float n = (dhHash31(cell) - 0.5) * 2.0;
            color.rgb = clamp(color.rgb + vec3(n * uNoiseIntensity * 0.08 * noiseFade), 0.0, 1.0);
        }
    }

    bool isWater = (vDhMaterialId == 12u);
    float roughness = 0.82;
    float metallic = 0.0;
    float f0 = 0.04;
    float emission = 0.0;
    float subsurface = 0.0;

    if (vDhMaterialId == 1u) {
        // Leaves / foliage
        roughness = 0.72;
        f0 = 0.030;
        subsurface = 0.56;
    } else if (vDhMaterialId == 2u) {
        // Stone / deepslate
        roughness = 0.80;
        f0 = 0.038;
    } else if (vDhMaterialId == 3u) {
        // Wood / logs
        roughness = 0.76;
        f0 = 0.034;
    } else if (vDhMaterialId == 4u) {
        // Metal
        metallic = 0.84;
        roughness = 0.30;
        f0 = 0.85;
    } else if (vDhMaterialId == 5u) {
        // Dirt / grass / clay
        roughness = 0.90;
        f0 = 0.028;
    } else if (vDhMaterialId == 6u) {
        // Lava
        roughness = 0.40;
        emission = 2.40;
        f0 = 0.04;
    } else if (vDhMaterialId == 7u) {
        // Snow
        roughness = 0.62;
        f0 = 0.038;
        subsurface = 0.42;
    } else if (vDhMaterialId == 8u) {
        // Sand / gravel
        roughness = 0.86;
        f0 = 0.036;
    } else if (vDhMaterialId == 9u) {
        // Terracotta / ceramic
        roughness = 0.68;
        f0 = 0.042;
    } else if (vDhMaterialId == 10u) {
        // Grass / plants
        roughness = 0.78;
        f0 = 0.030;
        subsurface = 0.48;
    } else if (vDhMaterialId == 11u) {
        // Illuminated / emissive blocks
        roughness = 0.45;
        emission = 1.60;
        f0 = 0.04;
    } else if (isWater) {
        // Water
        roughness = 0.04;
        f0 = 0.02;
    } else if (vDhMaterialId == 13u) {
        // Ice / glass
        roughness = 0.08;
        f0 = 0.042;
        subsurface = 0.25;
    }

    vec3 N = dot(vWorldNormal, vWorldNormal) > 1e-6 ? normalize(vWorldNormal) : vec3(0.0, 1.0, 0.0);
    float encodedRoughness = clamp(roughness, 0.02, 1.0) + (isWater ? 1.0 : 0.0);

    outColor = vec4(color.rgb, color.a);
    outNormalRoughness = vec4(N * 0.5 + 0.5, encodedRoughness);
    outMaterialPbr = vec4(
        clamp(metallic, 0.0, 1.0),
        clamp(emission * 0.25, 0.0, 1.0),
        vsPackNibble2(vBlockSkyLight.y, vBlockSkyLight.x),
        vsPackNibble2(sqrt(clamp(f0 / 0.9, 0.0, 1.0)), clamp(subsurface, 0.0, 1.0))
    );
}
