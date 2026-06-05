#version 450 core

layout(set = 0, binding = 0) uniform sampler2D sourceField;
layout(set = 1, binding = 0) uniform sampler2D destinationField;

layout(push_constant) uniform Constants{
    float sourceDt;
    int ensureBoundaryCondition;
};

#define BOUNDARY_SET 2
#include "common.glsl"

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 value;

void main(){
    if(checkBoundary(uv)){
        value = vec4(0);
        return;
    }

    value = texture(sourceField, uv) * sourceDt + texture(destinationField, uv);
}
