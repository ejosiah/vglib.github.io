#version 460

layout(location = 0) in struct {
    vec2 uv;
} fs_in;

layout(location = 0) out vec4 fragColor;

void main(){
    fragColor = vec4(1);
}