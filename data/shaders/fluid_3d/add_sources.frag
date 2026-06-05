#version 450 core

layout(set = 0, binding = 0) uniform sampler3D sourceField;
layout(set = 1, binding = 0) uniform sampler3D destinationField;

layout(location = 0) in vec3 uv;
layout(location = 0) out vec4 value;

layout(push_constant) uniform Constants {
    float dt;
};

void main() {
    value = texture(sourceField, uv) * dt + texture(destinationField, uv);
}
