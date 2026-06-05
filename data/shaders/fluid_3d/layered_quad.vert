#version 450 core

layout(constant_id = 0) const uint GRID_DEPTH = 1;

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 inUv;

layout(location = 0) out vec3 outUv;
layout(location = 1) flat out int outLayer;

void main() {
    uint layer = gl_InstanceIndex % max(GRID_DEPTH, 1u);
    outLayer = int(layer);
    outUv = vec3(inUv, (float(layer) + 0.5) / float(max(GRID_DEPTH, 1u)));
    gl_Position = vec4(pos, 0, 1);
}
