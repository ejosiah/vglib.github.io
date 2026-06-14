#version 450 core


layout(set = 0, binding = 0) uniform Globals{
    vec2 dx;
    vec2 dy;
    float dt;
    int ensureBoundaryCondition;
};

layout(set = 1, binding = 0) uniform sampler2D vectorField;

#define BOUNDARY_SET 2
#include "common.glsl"

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 divOut;

vec2 u(vec2 centerUv, vec2 coord) {
    vec2 velocity = texture(vectorField, scalarBoundarySampleUv(centerUv, coord)).xy;
    return reflectVelocityAtBoundary(velocity, centerUv, coord);
}

void main() {
    if(checkBoundary(uv)){
        divOut = vec4(0);
        return;
    }

    float dudx = (u(uv, uv + dx).x - u(uv, uv - dx).x)/(2*dx.x);
    float dudy = (u(uv, uv + dy).y - u(uv, uv - dy).y)/(2*dy.y);

    divOut.x = dudx + dudy;
}
