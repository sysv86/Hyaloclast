#version 460
#extension GL_EXT_ray_tracing : require

#include <vulkanshaders/v1/common.vsl>

layout(set = 1, binding = 2, std140) uniform SharcConfigUBO {
    SharcConfig cfg;
} uSharc;

layout(location = 0) rayPayloadInEXT VsPathPayload payload;

void main() {
    if ((payload.flags & 2u) != 0u) {
        payload.flags |= 8u;
        return;
    }
    if ((payload.flags & 32u) != 0u) {
        return;
    }

    vec3 rayDir = vsSafeNormalize(gl_WorldRayDirectionEXT, vec3(0.0, 1.0, 0.0));
    vec3 sunDir = vsSafeNormalize(vsWorld.sunDir + vec3(0.0, 1e-5, 0.0), normalize(vec3(0.4, 0.85, 0.2)));
    VsSkyState sky = VsSkyState(sunDir, vsWorld.rain, vsWorld.thunder, vsWorld.gameTimeSec);

    vec3 radiance = vec3(0.0);
    vs_dispatch_hook_sky(radiance, rayDir, sky);

    // The clouds hook now handles all dimensions internally (Overworld voxel clouds, End void
    // atmosphere and its black hole). Keep it on primary, specular and optical rays (bit 17);
    // do not gate by dimension here or the End sky never receives the hook at all.
    if ((payload.flags & (1u << 17u)) != 0u) {
        vec4 clouds = vs_dispatch_hook_clouds(rayDir, uSharc.cfg.cameraPosition.xyz, sky);
        radiance = radiance * clouds.a + clouds.rgb;
        if (((payload.flags >> 8u) & 255u) > 0u) {
            radiance = min(radiance, vec3(2.4));
        }
    }

    // Optical reflection/refraction misses set bit 8 (shadeOpticalRay) and can hit the
    // analytic sun disc in the sky hook while the surface already adds direct specular.
    // Cap them like secondary misses; primary rays (bit 8 clear) and shadow rays (early
    // return above) are untouched.
    if ((payload.flags & 256u) != 0u) {
        radiance = min(radiance, vec3(2.4));
    }

    vsPayloadSetRadianceRough(payload, max(radiance, vec3(0.0)), 1.0);
    payload.hitT = 65536.0;
    payload.flags &= ~1u;
}
