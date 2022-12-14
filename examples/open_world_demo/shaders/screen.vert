#version 460 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 uv;


layout(location = 0) out struct {
    vec2 uv;
} vs_out;

void main(){
    vs_out.uv = uv;
    gl_Position = vec4(position, 0 ,1.0);
}