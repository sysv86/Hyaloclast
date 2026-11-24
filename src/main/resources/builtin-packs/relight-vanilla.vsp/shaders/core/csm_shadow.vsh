#version 460

#include <vulkanshaders/v1/common.vsl>

layout(std140, set = 0, binding = 1) uniform CsmShadowUBO {
    mat4 cascadeViewProj[4];
    vec4 cascadeSplits;
    vec4 texelSizes;
} uCsm;

layout(push_constant) uniform CsmShadowPush {
    ivec3 RegionOrigin;
    uint cascadeIndex;
};

// Matches IGNIS_TERRAIN_40B (40 bytes per vertex)
layout(location = 0) in vec3 Position;
layout(location = 1) in vec4 Color;
layout(location = 2) in vec2 UV0;
layout(location = 3) in ivec2 UV2;
layout(location = 4) in vec4 Normal;
layout(location = 5) in vec4 Tangent;
layout(location = 6) in vec2 MidUV;

layout(location = 0) out vec2 outUv0;
layout(location = 1) out vec4 outColor;

void main() {
    uint materialFlags = uint(clamp(Normal.w * 127.0 + 0.5, 0.0, 127.0));
    vec3 localPos = Position;
    vec3 unitNormal = dot(Normal.xyz, Normal.xyz) > 1e-6 ? normalize(Normal.xyz) : vec3(0.0, 1.0, 0.0);

    VsVertexInput vInput = VsVertexInput(
        localPos,
        Color,
        UV0,
        UV2,
        unitNormal,
        Tangent,
        MidUV,
        materialFlags
    );
    vs_dispatch_hook_vertex_displace(localPos, vInput);

    vec3 worldPos = localPos + vec3(RegionOrigin);
    vec3 camRelPos = worldPos - vsWorld.cameraPosFrac;
    uint idx = min(cascadeIndex, 3u);

    outUv0 = UV0;
    outColor = Color;
    gl_Position = uCsm.cascadeViewProj[idx] * vec4(camRelPos, 1.0);
}
