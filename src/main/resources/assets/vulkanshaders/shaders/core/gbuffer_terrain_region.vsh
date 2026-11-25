#version 330
#extension GL_ARB_separate_shader_objects : require

#include <minecraft:fog.glsl>
#include <minecraft:globals.glsl>
#include <minecraft:projection.glsl>
#include <minecraft:sample_lightmap.glsl>
#include <minecraft:terrainglobals.glsl>

layout(push_constant) uniform RegionPush {
    ivec3 RegionOrigin;
    float RegionVisibility;
};

// Populated by GbufferTargetManager from WorldStateCollector's WorldUBO (352 bytes std140).
// cameraDeltaTime = (cameraDelta.xyz, gameTimeSec); pbrParams = (pbr, windHook, wavesHook, _);
// sunDirRain = (sunDir.xyz, rain); heldLightData = (heldLight, thunder, eyeInFluid, depthFar).
layout(std140) uniform VsMotion {
    mat4 prevViewProj;
    vec4 cameraDeltaTime;
    vec4 pbrParams;
    vec4 sunDirRain;
    vec4 heldLightData;
    mat4 invView;
    vec4 packParams[10];
} vsMotion;

uniform sampler2D VsNormalAtlas;
uniform sampler2D VsSpecularAtlas;

// Matches IGNIS_TERRAIN_40B (40 bytes per vertex)
layout(location = 0) in vec3 Position;
layout(location = 1) in vec4 Color;
layout(location = 2) in vec2 UV0;
layout(location = 3) in ivec2 UV2;
layout(location = 4) in vec4 Normal;
layout(location = 5) in vec4 Tangent;
layout(location = 6) in vec2 MidUV;

uniform sampler2D Sampler2;

layout(location = 0) out float sphericalVertexDistance;
layout(location = 1) out float cylindricalVertexDistance;
layout(location = 2) out vec4 vertexColor;
layout(location = 3) out vec2 texCoord0;
layout(location = 4) out float chunkVisibility;
layout(location = 5) out vec3 worldNormal;
layout(location = 6) out vec4 curClipPos;
layout(location = 7) out vec4 prevClipPos;
layout(location = 8) flat out uint materialFlags;
layout(location = 9) out vec2 blockSkyLight;
layout(location = 10) out vec4 worldTangent;
layout(location = 11) out vec3 camRelPos;
layout(location = 12) out vec2 midUv;

uint materialFlagsOf(float packedNormalW);

void main() {
    vec3 localPos = Position;
    uint vertexHookFlags = materialFlagsOf(Normal.w);
    uint matId = (vertexHookFlags >> 3u) & 0x0Fu;

    // The active .vsp pack's vertex_displace / water_waves hook bodies are injected here by
    // GbufferPipelineRegistry.applyPackHooks, mutating localPos and wave parameters.
    // @hook vertex_displace
    // @hook water_waves

    vec3 pos = localPos + (RegionOrigin - CameraBlockPos) + CameraOffset;

    // Apply the water-wave parameters produced by the injected pack hook (defaults if absent).
#ifndef VS_HOOK_WATER_WAVES_BOUND
    float vsWaveStrength = 0.0;
    float vsWaveSpeed = 1.8;
    float vsWaveScale = 2.2;
#endif
    // Only horizontal water surfaces bob with the wave field. Vertical waterfall quads must keep
    // their exact block-aligned silhouette or the top edge detaches from the source surface.
    if (matId == 8u && vsWaveStrength > 0.0 && abs(Normal.y) > 0.5) {
        float t = vsMotion.cameraDeltaTime.w * vsWaveSpeed;
        vec2 worldXZ = vec2(RegionOrigin.xz) + localPos.xz;
        float wave = sin(worldXZ.x * vsWaveScale + t) * cos(worldXZ.y * vsWaveScale * 0.85 - t * 0.7);
        pos.y += wave * vsWaveStrength * 0.06;
    }

    vec4 clipPos = ProjMat * ModelViewMat * vec4(pos, 1.0);
    gl_Position = clipPos;

    // True motion vectors: re-project the same world-space vertex through last frame's view-projection
    // after undoing this frame's camera translation.
    vec3 prevCamRelPos = pos + vsMotion.cameraDeltaTime.xyz;
    vec4 prevClip = vsMotion.prevViewProj * vec4(prevCamRelPos, 1.0);
    if (abs(vsMotion.prevViewProj[0][0]) + abs(vsMotion.prevViewProj[1][1]) < 1e-8) {
        prevClip = clipPos;
    }

    sphericalVertexDistance = fog_spherical_distance(pos);
    cylindricalVertexDistance = fog_cylindrical_distance(pos);
    vertexColor = Color;
    texCoord0 = UV0;
    worldNormal = dot(Normal.xyz, Normal.xyz) > 1e-8 ? normalize(Normal.xyz) : vec3(0.0, 1.0, 0.0);
    worldTangent = Tangent;
    curClipPos = clipPos;
    prevClipPos = prevClip;
    materialFlags = vertexHookFlags;
    blockSkyLight = vec2(
        clamp(float(UV2.x) / 240.0, 0.0, 1.0),
        clamp(float(UV2.y) / 240.0, 0.0, 1.0)
    );
    camRelPos = pos;
    midUv = MidUV;

    if (RegionVisibility >= 1.0) {
        chunkVisibility = 1.0;
    } else {
        const float chunkFullyVisibleRange = 16.0;
        float dist = length(pos);
        chunkVisibility = mix(1.0, RegionVisibility, clamp((dist - chunkFullyVisibleRange) / chunkFullyVisibleRange, 0.0, 1.0));
    }
}

uint materialFlagsOf(float packedNormalW) {
    return uint(clamp(packedNormalW * 127.0 + 0.5, 0.0, 127.0));
}
