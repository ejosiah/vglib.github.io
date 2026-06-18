#version 460 core

layout(binding = 0) uniform sampler2D vectorField;

layout(push_constant) uniform Transform {
    mat4 transform;
};

layout(location = 0) in vec2 vertex;
layout(location = 1) in vec2 position;
layout(location = 2) in vec2 fieldUv;

mat2 rot(float angle) {
    float c = cos(angle);
    float s = sin(angle);

    return mat2(
    vec2(c, -s),
    vec2(s, c)
    );
}

void main(){
    vec2 v = texture(vectorField, fieldUv).xy;
    float scale = 0.05 * length(v);
    float angle = atan(v.y, v.x);
    mat2 rotation = rot(-angle);
    gl_Position = transform * vec4((rotation * (scale * vertex.xy)) + position, 0.0, 1.0);
}
