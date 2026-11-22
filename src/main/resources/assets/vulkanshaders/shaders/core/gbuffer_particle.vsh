#version 330
#extension GL_ARB_separate_shader_objects : require

#include <minecraft:globals.glsl>
#include <minecraft:fog.glsl>
#include <minecraft:dynamictransforms.glsl>
#include <minecraft:projection.glsl>
#include <minecraft:sample_lightmap.glsl>

layout(std140) uniform VsMotion {
    mat4 prevViewProj;
    vec4 cameraDeltaTime;
    vec4 pbrParams;
    vec4 sunDirRain;
    vec4 heldLightData;
    mat4 invView;
    vec4 packParams[10];
} vsMotion;

layout(location = 0) in vec3 Position;
layout(location = 1) in vec2 UV0;
layout(location = 2) in vec4 Color;
layout(location = 3) in ivec2 UV2;

uniform sampler2D Sampler2;

layout(location = 0) out float sphericalVertexDistance;
layout(location = 1) out float cylindricalVertexDistance;
layout(location = 2) out vec2 texCoord0;
layout(location = 3) out vec4 vertexColor;
layout(location = 4) out vec4 curClipPos;
layout(location = 5) out vec4 prevClipPos;

void main() {
    vec4 viewPos = ModelViewMat * vec4(Position, 1.0);
    vec4 clipPos = ProjMat * viewPos;
    gl_Position = clipPos;

    bool hasInvView = (abs(vsMotion.invView[0][0]) + abs(vsMotion.invView[1][1]) + abs(vsMotion.invView[2][2])) >= 1e-6;
    bool hasPrevVp = (abs(vsMotion.prevViewProj[0][0]) + abs(vsMotion.prevViewProj[1][1])) >= 1e-8;
    vec3 worldRelPos = hasInvView ? (vsMotion.invView * viewPos).xyz : viewPos.xyz;
    vec4 prevClip = (hasInvView && hasPrevVp)
        ? (vsMotion.prevViewProj * vec4(worldRelPos + vsMotion.cameraDeltaTime.xyz, 1.0))
        : clipPos;

    sphericalVertexDistance = fog_spherical_distance(worldRelPos);
    cylindricalVertexDistance = fog_cylindrical_distance(worldRelPos);
    texCoord0 = UV0;
    vertexColor = Color * sample_lightmap(Sampler2, UV2);
    curClipPos = clipPos;
    prevClipPos = prevClip;
}
