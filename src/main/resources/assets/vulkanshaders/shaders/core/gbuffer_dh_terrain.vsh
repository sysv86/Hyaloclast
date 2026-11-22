#version 330
#extension GL_ARB_separate_shader_objects : require

// Distant Horizons LOD terrain G-Buffer vertex shader.
// Vertex format matches BlazeDhTerrainRenderer (SHORT_XYZ_POS_AND_W_META, RGBA_UBYTE_COLOR,
// MATERIAL_NORMAL_TILE, OFFSET_DATA, UV_NORMAL) and DH's std140 uniform blocks.

layout(location = 0) in uvec4 vPositionAndMeta;
layout(location = 1) in vec4 vColor;
layout(location = 2) in uvec4 materialNormalTile;
#ifndef VS_DH_PACKED_16
layout(location = 3) in uvec4 offsetData;
layout(location = 4) in uvec4 uvNormal;
#endif

layout(std140) uniform vertUniqueUniformBlock {
#ifdef VS_DH_PACKED_16
    vec3 uModelOffset;
#else
    ivec3 uModelOffset;
#endif
};

layout(std140) uniform vertSharedUniformBlock {
    int uIsWhiteWorld;
    float uWorldYOffset;
    float uMircoOffset;
    float uEarthRadius;
    float uFrameMod8;
    float uViewWidth;
    float uViewHeight;
#ifdef VS_DH_PACKED_16
    vec3 uCameraPos;
#else
    ivec3 uCoarseCameraPos;
    vec3 uFineCameraPos;
#endif
    mat4 uCombinedMatrix;
};

layout(location = 0) out vec4 vVertColor;
layout(location = 1) out vec3 vWorldNormal;
layout(location = 2) out vec3 vCamRelPos;
layout(location = 3) flat out uint vDhMaterialId;
layout(location = 4) out vec2 vBlockSkyLight;

vec3 dhDecodeCardinalNormal(uint idx) {
    if (idx == 0u) return vec3( 0.0,  1.0,  0.0);
    if (idx == 1u) return vec3( 0.0, -1.0,  0.0);
    if (idx == 2u) return vec3( 0.0,  0.0, -1.0);
    if (idx == 3u) return vec3( 0.0,  0.0,  1.0);
    if (idx == 4u) return vec3(-1.0,  0.0,  0.0);
    return vec3(1.0, 0.0, 0.0);
}

void main() {
    uint wMeta = vPositionAndMeta.w;
    uint normalIdx = materialNormalTile.y & 7u;
    vec3 N = dhDecodeCardinalNormal(min(normalIdx, 5u));
    vec3 localPos = vec3(vPositionAndMeta.xyz);
#ifdef VS_DH_PACKED_16
    // DH 3.3.4 packs light in the low byte and signed two-bit X/Z offsets in the high byte.
    uvec2 offsetCode = (uvec2(wMeta) >> uvec2(8u, 12u)) & 3u;
    vec2 offset = vec2(offsetCode & 1u) * (1.0 - vec2(offsetCode >> 1u) * 2.0);
    vec3 camRelPos = localPos + (uModelOffset - uCameraPos);
    camRelPos.xz += offset * uMircoOffset;
    uint lights = wMeta & 0xFFu;
#else
    localPos += vec3(offsetData.xyz) * (1.0 / 256.0);
    vec3 camRelPos = vec3(uModelOffset - uCoarseCameraPos) - uFineCameraPos + localPos;
    camRelPos.y -= uWorldYOffset;
    if (((wMeta >> 4u) & 1u) != 0u) {
        camRelPos += N * uMircoOffset;
    }
    uint lights = wMeta >> 8u;
#endif

    if (abs(uEarthRadius) > 1.0) {
        float horizDistSq = dot(camRelPos.xz, camRelPos.xz);
        camRelPos.y -= horizDistSq / (2.0 * uEarthRadius);
    }

    gl_Position = uCombinedMatrix * vec4(camRelPos, 1.0);

    uint blockLightNibble = lights & 0xFu;
    uint skyLightNibble = (lights >> 4u) & 0xFu;

    vVertColor = (uIsWhiteWorld != 0) ? vec4(1.0, 1.0, 1.0, vColor.a) : vColor;
    vWorldNormal = N;
    vCamRelPos = camRelPos;
    vDhMaterialId = materialNormalTile.x;
    vBlockSkyLight = vec2(float(blockLightNibble) / 15.0, float(skyLightNibble) / 15.0);
}
