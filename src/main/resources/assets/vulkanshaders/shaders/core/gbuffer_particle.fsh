#version 330
#extension GL_ARB_separate_shader_objects : require

#include <minecraft:globals.glsl>
#include <minecraft:fog.glsl>

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
layout(location = 2) in vec2 texCoord0;
layout(location = 3) in vec4 vertexColor;
layout(location = 4) in vec4 curClipPos;
layout(location = 5) in vec4 prevClipPos;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 outNormalRoughness;
layout(location = 2) out vec4 outMaterialPbr;
layout(location = 3) out vec2 outVelocity;

float vsPackNibble2(float hi, float lo) {
    float h = floor(clamp(hi, 0.0, 1.0) * 15.0 + 0.5);
    float l = floor(clamp(lo, 0.0, 1.0) * 15.0 + 0.5);
    return (h * 16.0 + l) / 255.0;
}

void main() {
    vec4 color = texture(Sampler0, texCoord0) * vertexColor;
    if (color.a < 0.01) discard;
    fragColor = color;

    // Camera-facing billboards: normal points toward the viewer, encoded into the G-Buffer.
    vec3 N = normalize(vec3(0.0, 1.0, 0.0)); // Upward hemisphere billboard normal for consistent sky/sun lighting
    float roughness = 0.82;

    // Derive soft self-emission for bright, high-chroma particles (flames, soul fire, lava sparks,
    // end rod, glow, enchant) while keeping low-chroma smoke/dust and water-splash particles non-emissive.
    float maxCh = max(max(color.r, color.g), color.b);
    float minCh = min(min(color.r, color.g), color.b);
    float chroma = (maxCh > 1e-4) ? ((maxCh - minCh) / maxCh) : 0.0;
    float luma = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
    bool isWaterSplash = (color.b > color.r * 1.35) && (color.b > color.g * 1.08) && (color.r < 0.45) && (luma < 0.62);
    float chromaEmission = smoothstep(0.32, 0.68, chroma) * smoothstep(0.45, 0.82, maxCh) * 1.15;
    float hotCoreEmission = smoothstep(0.90, 0.99, luma) * smoothstep(0.80, 0.98, color.a) * 0.95;
    float emission = isWaterSplash ? 0.0 : clamp(max(chromaEmission, hotCoreEmission), 0.0, 1.4);

    vec2 curNdc = curClipPos.xy / max(abs(curClipPos.w), 1e-6);
    vec2 prevNdc = prevClipPos.xy / max(abs(prevClipPos.w), 1e-6);
    vec2 velocity = (curNdc - prevNdc) * 0.5;

    outNormalRoughness = vec4(N * 0.5 + 0.5, roughness);
    outMaterialPbr = vec4(
        0.0,
        clamp(emission * 0.25, 0.0, 1.0),
        vsPackNibble2(0.85, 0.0),
        vsPackNibble2(sqrt(clamp(0.04 / 0.9, 0.0, 1.0)), 0.0)
    );
    outVelocity = velocity;
}
