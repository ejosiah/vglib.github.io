#version 450 core


layout(set = 0, binding = 0) uniform Globals{
    vec2 dx;
    vec2 dy;
    float dt;
    int ensureBoundaryCondition;
};

layout(set = 1, binding = 0) uniform sampler2D solution;
layout(set = 2, binding = 0) uniform sampler2D unknown;

#define BOUNDARY_SET 3
#include "common.glsl"

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 x;

layout(push_constant) uniform Constants {
    float alpha;
    float rBeta;
    int isVectorField;
};

vec4 b(vec2 coord){
    return texture(solution, st(coord));
}

vec4 x0(vec2 centerUv, vec2 coord){
    vec4 _x0 = texture(unknown, scalarBoundarySampleUv(centerUv, coord));
    if(isVectorField == 1){
        _x0.xy = reflectVelocityAtBoundary(_x0.xy, centerUv, coord);
    }
    return _x0;
}

void main(){
    if(checkBoundary(uv)){
        x = vec4(0);
        return;
    }

    float dxdx = dx.x * dx.x;
    float dydy = dy.y * dy.y;
    x = ((x0(uv, uv + dx) + x0(uv, uv - dx)) * dydy + (x0(uv, uv + dy) + x0(uv, uv - dy)) * dxdx + alpha * b(uv)) * rBeta;
}
