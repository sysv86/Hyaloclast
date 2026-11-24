#version 460

layout(set = 0, binding = 0) uniform sampler2D uLinearDepth;

layout(push_constant) uniform DepthReprojectPush {
    float zScale;
    float zOffset;
    float wScale;
    float wOffset;
    uint srcWidth;
    uint srcHeight;
    uint dstWidth;
    uint dstHeight;
} pc;

layout(location = 0) in vec2 inUv;

void main() {
    float viewZ = textureLod(uLinearDepth, inUv, 0.0).r;
    float sanitized = (viewZ <= 0.0 || viewZ >= 60000.0)
        ? 1.0e7
        : (abs(viewZ) * 1.004 + 0.03);

    float zCam = -sanitized;
    float clipZ = pc.zScale * zCam + pc.zOffset;
    float clipW = pc.wScale * zCam + pc.wOffset;
    float ndcDepth = (abs(clipW) > 1e-12) ? clamp(clipZ / clipW, 0.0, 1.0) : 0.0;

    gl_FragDepth = ndcDepth;
}
