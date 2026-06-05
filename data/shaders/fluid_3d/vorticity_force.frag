#version 450 core

layout(set = 0, binding = 0) uniform Globals{
    vec4 dx;
    vec4 dy;
    vec4 dz;
    float dt;
    int ensureBoundaryCondition;
};

#include "common.glsl"

layout(set = 1, binding = 0) uniform sampler3D vorticityField;
layout(set = 2, binding = 0) uniform sampler3D forceField;

layout(push_constant) uniform Constants{
    float cScale;
};

layout(location = 0) in vec3 uv;
layout(location = 0) out vec4 force;

vec3 vort(vec3 coord) {
    return texture(vorticityField, st(coord)).xyz;
}

float vortMag(vec3 coord) {
    return length(vort(coord));
}

vec3 accumForce(vec3 coord) {
    return texture(forceField, st(coord)).xyz;
}

void main() {
    float dwdx = (vortMag(uv + dx.xyz) - vortMag(uv - dx.xyz)) / (2 * dx.x);
    float dwdy = (vortMag(uv + dy.xyz) - vortMag(uv - dy.xyz)) / (2 * dy.y);
    float dwdz = (vortMag(uv + dz.xyz) - vortMag(uv - dz.xyz)) / (2 * dz.z);

    vec3 n = vec3(dwdx, dwdy, dwdz);
    float epsilon = 2.4414e-4;
    float magSqr = max(epsilon, dot(n, n));
    n = n * inversesqrt(magSqr);

    float cellSize = max(dx.x, max(dy.y, dz.z));
    force.xyz = accumForce(uv) + cellSize * cScale * cross(n, vort(uv));
    force.w = 0;
}
