#version 450 core

layout(set = 0, binding = 0) uniform Globals{
    vec2 dx;
    vec2 dy;
    float dt;
    int ensureBoundaryCondition;
};

layout(set = 1, binding = 0) uniform sampler2D vectorField;
layout(set = 2, binding = 0) uniform texture2D quantity;
layout(set = 3, binding = 0) uniform sampler linerSampler;

#define BOUNDARY_SET 4
#include "common.glsl"

layout(push_constant) uniform Constants {
    int isVectorField;
};

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 quantityOut;

vec4 sampleQuantity(vec2 centerUv, vec2 sampleUv){
    vec2 resolvedUv = st(sampleUv);
    vec4 q = texture(sampler2D(quantity, linerSampler), resolvedUv);
    return q;
}

void main(){
    vec2 u = texture(vectorField, uv).xy;

    vec2 p = uv - dt * u;
    quantityOut = sampleQuantity(uv, p);
}
