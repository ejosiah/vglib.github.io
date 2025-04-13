#version 460

#define PI 3.14159265358979323846

layout(set = 0, binding = 0) uniform sampler2D pressure_field;

layout(set = 1, binding = 0) buffer MinMax {
    float data;
} min_max[2];

layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 fragColor;

void main(){
    fragColor = vec4(1);

    float x = texture(pressure_field, vUv).x;
    float a = min_max[1].data;
    float b = min_max[0].data;
    float t = (x - a)/(b - a);

    fragColor.r = sin(.5 * PI * t);
    fragColor.b = sin(PI * t);
    fragColor.b = cos(.5 * PI * t);
}