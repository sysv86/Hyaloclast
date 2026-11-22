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
layout(location = 1) in vec4 Color;
layout(location = 2) in vec2 UV0;
#ifdef NO_OVERLAY_UV1
layout(location = 3) in ivec2 UV2;
#ifndef NO_NORMAL
layout(location = 4) in vec3 Normal;
#endif
#else
layout(location = 3) in ivec2 UV1;
layout(location = 4) in ivec2 UV2;
#ifndef NO_NORMAL
layout(location = 5) in vec3 Normal;
#endif
#endif

#if !defined(EMISSIVE) && !defined(OIT_ALPHA_ONLY)
uniform sampler2D Sampler2;
#endif

layout(location = 0) out float sphericalVertexDistance;
layout(location = 1) out float cylindricalVertexDistance;
layout(location = 2) out vec4 vertexColor;
layout(location = 3) out vec2 texCoord0;
layout(location = 4) out vec3 worldNormal;
layout(location = 5) out vec4 curClipPos;
layout(location = 6) out vec4 prevClipPos;
layout(location = 7) out vec2 blockSkyLight;
layout(location = 8) out vec4 overlayColor;
layout(location = 9) out vec3 viewRelPos;

void main() {
    vec4 viewPos = ModelViewMat * vec4(Position, 1.0);
    vec4 clipPos = ProjMat * viewPos;
    gl_Position = clipPos;

    // Transform model-local Position into camera-relative world space via invView * ModelViewMat
    // before undoing this frame's camera translation and re-projecting with prevViewProj.
    bool hasInvView = (abs(vsMotion.invView[0][0]) + abs(vsMotion.invView[1][1]) + abs(vsMotion.invView[2][2])) >= 1e-6;
    bool hasPrevVp = (abs(vsMotion.prevViewProj[0][0]) + abs(vsMotion.prevViewProj[1][1])) >= 1e-8;
    vec3 worldRelPos = hasInvView ? (vsMotion.invView * viewPos).xyz : viewPos.xyz;
    vec4 prevClip = (hasInvView && hasPrevVp)
        ? (vsMotion.prevViewProj * vec4(worldRelPos + vsMotion.cameraDeltaTime.xyz, 1.0))
        : clipPos;

    sphericalVertexDistance = fog_spherical_distance(worldRelPos);
    cylindricalVertexDistance = fog_cylindrical_distance(worldRelPos);
    vertexColor = Color;
    texCoord0 = UV0;
#ifdef NO_NORMAL
    vec3 transformedNormal = vec3(0.0, 1.0, 0.0);
#else
    vec3 transformedNormal = hasInvView
        ? (mat3(vsMotion.invView) * (mat3(ModelViewMat) * Normal))
        : Normal;
#endif
    worldNormal = length(transformedNormal) > 1e-5 ? normalize(transformedNormal) : vec3(0.0, 1.0, 0.0);
    curClipPos = clipPos;
    prevClipPos = prevClip;
    blockSkyLight = vec2(
        clamp(float(UV2.x) / 240.0, 0.0, 1.0),
        clamp(float(UV2.y) / 240.0, 0.0, 1.0)
    );
#ifdef NO_OVERLAY_UV1
    overlayColor = vec4(1.0);
#else
    if (UV1.y < 8) {
        overlayColor = vec4(1.0, 0.0, 0.0, 178.0 / 255.0);
    } else {
        float whiteAlpha = clamp(1.0 - (float(UV1.x) / 15.0) * 0.75, 0.0, 1.0);
        overlayColor = vec4(1.0, 1.0, 1.0, whiteAlpha);
    }
#endif
    viewRelPos = worldRelPos;
}
