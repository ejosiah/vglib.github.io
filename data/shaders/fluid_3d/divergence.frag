#version 450 core


layout(set = 0, binding = 0) uniform Globals{
    vec4 dx;
    vec4 dy;
    vec4 dz;
    float dt;
    int ensureBoundaryCondition;
};

#include "common.glsl"

layout(set = 1, binding = 0) uniform sampler3D vectorField;

layout(location = 0) in vec3 uv;
layout(location = 0) out vec4 divOut;

vec3 u(vec3 coord) {
    return applyBoundaryCondition(coord, texture(vectorField, st(coord)).xyz);
}

void main() {
    float dudx = (u(uv + dx.xyz).x - u(uv - dx.xyz).x)/(2*dx.x);
    float dudy = (u(uv + dy.xyz).y - u(uv - dy.xyz).y)/(2*dy.y);
    float dudz = (u(uv + dz.xyz).z - u(uv - dz.xyz).z)/(2*dz.z);

    divOut.x = dudx + dudy + dudz;
}
