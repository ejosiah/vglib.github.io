#version 460

#extension GL_EXT_scalar_block_layout : enable

layout(set = 0, binding = 1, scalar) buffer Constants {
    vec3 color;
    float step_size;
    int nextVertex;
};

layout(location = 0) in vec2 position;

layout(location = 0) out vec3 vColor;

void main() {
    vColor = color;
    gl_Position = vec4(position, 0, 1);
}