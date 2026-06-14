#version 450 core


layout(set = 0, binding = 0) uniform Globals{
    vec2 dx;
    vec2 dy;
    float dt;
    int ensureBoundaryCondition;
};

layout(set = 1, binding = 0) uniform sampler2D vectorField;
layout(set = 2, binding = 0) uniform sampler2D pressure;

#define BOUNDARY_SET 3
#include "common.glsl"

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 velocity_out;

float p(vec2 centerUv, vec2 coord) {
    return texture(pressure, scalarBoundarySampleUv(centerUv, coord)).x;
}

vec2 u(vec2 coord) {
    return texture(vectorField, st(coord)).xy;
}

vec2 pg(vec2 coord){
    float dudx = (p(coord, coord + dx) - p(coord, coord - dx))/(2*dx.x);
    float dudy = (p(coord, coord + dy) - p(coord, coord - dy))/(2*dy.y);

    return vec2(dudx, dudy);
}

void main() {
    if(checkBoundary(uv)){
        velocity_out = vec4(0);
        return;
    }

    velocity_out.xy = u(uv) - pg(uv);
    velocity_out.zw = vec2(0);
}
