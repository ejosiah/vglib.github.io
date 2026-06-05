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
layout(location = 0) out vec4 vort;

vec3 u(vec3 coord) {
    return applyBoundaryCondition(coord, texture(vectorField, st(coord)).xyz);
}

void main() {
    float dwdy = (u(uv + dy.xyz).z - u(uv - dy.xyz).z) / (2 * dy.y);
    float dvdz = (u(uv + dz.xyz).y - u(uv - dz.xyz).y) / (2 * dz.z);
    float dudz = (u(uv + dz.xyz).x - u(uv - dz.xyz).x) / (2 * dz.z);
    float dwdx = (u(uv + dx.xyz).z - u(uv - dx.xyz).z) / (2 * dx.x);
    float dvdx = (u(uv + dx.xyz).y - u(uv - dx.xyz).y) / (2 * dx.x);
    float dudy = (u(uv + dy.xyz).x - u(uv - dy.xyz).x) / (2 * dy.y);

    vort.xyz = vec3(dwdy - dvdz, dudz - dwdx, dvdx - dudy);
    vort.w = length(vort.xyz);
}
