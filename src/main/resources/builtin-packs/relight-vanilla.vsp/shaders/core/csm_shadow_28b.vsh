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

// Matches 28-byte compact terrain vertex layout (Position, Color, UV0, UV2) without Normal/Tangent/MidUV
layout(location = 0) in vec3 Position;
layout(location = 1) in vec4 Color;
layout(location = 2) in vec2 UV0;
layout(location = 3) in ivec2 UV2;

layout(location = 0) out vec2 vUv;
layout(location = 1) out vec4 outColor;

void main() {
    vec3 worldPos = Position + vec3(RegionOrigin);
    vec3 camRelPos = worldPos - vsWorld.cameraPosFrac;
    uint idx = min(cascadeIndex, 3u);

    vUv = UV0;
    outColor = Color;
    gl_Position = uCsm.cascadeViewProj[idx] * vec4(camRelPos, 1.0);
}
