#version 460

layout(set = 1, binding = 0) uniform sampler2D uBlockAtlas;

layout(location = 0) in vec2 inUv0;
layout(location = 1) in vec4 inColor;

void main() {
    float alpha = texture(uBlockAtlas, inUv0).a * inColor.a;
    if (alpha < 0.5) {
        discard;
    }
}
