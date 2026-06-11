#version 450 core

layout(set = 0, binding = 0) uniform sampler2D fieldIn;

layout(push_constant) uniform Constants{
    int ensureBoundaryCondition;
};

#define BOUNDARY_SET 1
#include "common.glsl"

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fieldOut;

void main(){
    fieldOut = texture(fieldIn, uv);
}
