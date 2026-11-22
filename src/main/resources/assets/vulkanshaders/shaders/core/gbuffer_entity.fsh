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
layout(location = 2) in vec4 vertexColor;
layout(location = 3) in vec2 texCoord0;
layout(location = 4) in vec3 worldNormal;
layout(location = 5) in vec4 curClipPos;
layout(location = 6) in vec4 prevClipPos;
layout(location = 7) in vec2 blockSkyLight;
layout(location = 8) in vec4 overlayColor;
layout(location = 9) in vec3 viewRelPos;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 outNormalRoughness;
layout(location = 2) out vec4 outMaterialPbr;
layout(location = 3) out vec2 outVelocity;

float vsPackNibble2(float hi, float lo) {
    float h = floor(clamp(hi, 0.0, 1.0) * 15.0 + 0.5);
    float l = floor(clamp(lo, 0.0, 1.0) * 15.0 + 0.5);
    return (h * 16.0 + l) / 255.0;
}

// LabPBR 1.3 hardcoded linear metal F0 RGB table for specular green codes 230..237.
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

void main() {
    vec4 color = texture(Sampler0, texCoord0) * vertexColor;
#ifdef ALPHA_CUTOUT
    if (color.a < ALPHA_CUTOUT) discard;
#endif
    color.rgb = mix(overlayColor.rgb, color.rgb, overlayColor.a);
    fragColor = color;

    vec3 N = normalize(length(worldNormal) > 1e-4 ? worldNormal : vec3(0.0, 1.0, 0.0));
    float roughness = 0.62;
    float metallic = 0.0;
    float f0 = 0.04;
    float emission = 0.0;
    float subsurface = 0.0;

    uint materialFlags = 0u;
#ifdef EMISSIVE
    emission = 0.85;
    roughness = 0.45;
    materialFlags = 4u;
#endif

    // Run pack surface hook BEFORE LabPBR so authored LabPBR _s textures are never overwritten.
    // @hook surface

    // LabPBR 1.3 for entities via Cotangent Frame with independent T and B normalization
    // so non-square entity textures do not skew tangent-space normals.
    if (vsMotion.pbrParams.x > 0.5) {
        vec3 dp1 = dFdx(viewRelPos);
        vec3 dp2 = dFdy(viewRelPos);
        vec2 duv1 = dFdx(texCoord0);
        vec2 duv2 = dFdy(texCoord0);
        vec3 dp2perp = cross(dp2, N);
        vec3 dp1perp = cross(N, dp1);
        vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
        vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
        float tLenSq = dot(T, T);
        float bLenSq = dot(B, B);

        vec4 nSample = texture(VsNormalAtlas, texCoord0);
        vec4 sSample = texture(VsSpecularAtlas, texCoord0);
        if (tLenSq > 1e-10 && bLenSq > 1e-10) {
            mat3 tbn = mat3(T * inversesqrt(tLenSq), B * inversesqrt(bLenSq), N);
            vec2 normalXY = nSample.rg * 2.0 - 1.0;
            if (dot(normalXY, normalXY) < 1.2) {
                float normalZ = sqrt(max(0.0, 1.0 - dot(normalXY, normalXY)));
                vec3 tangentNormal = vec3(normalXY, normalZ);
                if (dot(tangentNormal, tangentNormal) > 1e-5) {
                    N = normalize(tbn * normalize(tangentNormal));
                }
            }
        }

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

        float ao = clamp(nSample.b, 0.0, 1.0);
        if (!hardcodedMetal && ao < 0.998) {
            color.rgb *= mix(1.0, pow(max(ao, 0.05), 1.0 / 2.2), 0.72);
            fragColor.rgb = color.rgb;
        }
    }

    // Clamp roughness to >= 0.02 so outNormalRoughness.a > 0 always marks valid G-Buffer geometry
    // even if a custom pack's surface.vsl sets s.roughness = 0.0.
    roughness = clamp(roughness, 0.02, 1.0);

    outNormalRoughness = vec4(N * 0.5 + 0.5, roughness);
    outMaterialPbr = vec4(
        clamp(metallic, 0.0, 1.0),
        clamp(emission * 0.25, 0.0, 1.0),
        vsPackNibble2(blockSkyLight.y, blockSkyLight.x),
        vsPackNibble2(sqrt(clamp(f0 / 0.9, 0.0, 1.0)), clamp(subsurface, 0.0, 1.0))
    );
    vec2 curNdc = curClipPos.xy / max(abs(curClipPos.w), 1e-6);
    vec2 prevNdc = prevClipPos.xy / max(abs(prevClipPos.w), 1e-6);
    vec2 velocity = (curNdc - prevNdc) * 0.5;
    outVelocity = clamp(velocity, vec2(-1.0), vec2(1.0));
    // @hook gbuffer_pack
}
