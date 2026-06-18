#version 460 core

layout(push_constant) uniform Constants {
    mat4 transform;
};

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec2 vUv;

void main(){
    vUv = uv;
    gl_Position = transform * vec4(pos, 0, 1);
}