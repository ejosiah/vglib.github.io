#version 460 core

layout(location = 0) in vec3 vColor;
layout(location = 0) out vec4 fragColor;

void main(){
    fragColor = vec4(vColor, 1);
}