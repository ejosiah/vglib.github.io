#version 460 core

layout(push_constant) uniform MVP {
    mat4 model;
    mat4 view;
    mat4 projection;
};

layout(location = 0) in vec4 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 tanget;
layout(location = 3) in vec3 bitangent;
layout(location = 4) in vec3 color;
layout(location = 5) in vec2 uv;

layout(location = 0) out vec2 vUv;

void main(){
    vUv = uv;
    gl_Position = projection * view * model * position;
}