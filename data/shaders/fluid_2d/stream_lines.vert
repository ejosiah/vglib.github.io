#version 460

#extension GL_EXT_scalar_block_layout : enable

layout(set = 0, binding = 1, scalar) buffer Constants {
    vec3 color;
    ivec2 grid_size;
    float step_size;
    uint next_vertex;
    uint offset;
    vec2 domain_min;
    vec2 domain_size;
};

layout(push_constant) uniform Transform {
    mat4 transform;
};

layout(location = 0) in vec2 position;

layout(location = 0) out vec3 vColor;

void main() {
    vColor = color;
    gl_Position = transform * vec4(position, 0, 1);
}
