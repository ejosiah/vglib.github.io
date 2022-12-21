#version 450

#include "terrain_ubo.glsl"

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

layout(location = 0) in vec3 worldPosition_in[3];
layout(location = 1) in vec3 normal_in[3];
layout(location = 2) in vec2 uv_in[3];
layout(location = 3) in vec2 patch_uv_in[3];
layout(location = 4) in vec3 color_in[3];

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 normal;
layout(location = 2) out vec2 uv_out;
layout(location = 3) out vec2 patch_uv_out;
layout(location = 4) noperspective out vec3 edgeDist;
layout(location = 5) out vec3 color_out;

vec3 edgeDistance(vec3 p0, vec3 p1, vec3 p2){
    float a = distance(p1, p2);
    float b = distance(p0, p1);
    float c = distance(p0, p2);

    float alpha = acos((b * b + c * c - a * a)/(2 * b * c));
    float beta =  acos((a * a + c * c - b * b)/(2 * a * c));

    float ha = abs(c * sin(beta));
    float hb = abs(c * sin(alpha));
    float hc = abs(b * sin(alpha));

    return vec3(ha, hb, hc);
}

void main(){
    vec3 p0 = gl_in[0].gl_Position.xyz;
    vec3 p1 = gl_in[1].gl_Position.xyz;
    vec3 p2 = gl_in[2].gl_Position.xyz;

    vec3 edgeDisComb = edgeDistance(p0, p1, p2);

    for(int i = 0; i < gl_in.length(); i++){
        worldPosition = worldPosition_in[i];
        normal = normal_in[i];
        uv_out = uv_in[i];
        patch_uv_out = patch_uv_in[i];
        color_out = color_in[i];

        edgeDist = vec3(0);
        edgeDist[i] = edgeDisComb[i];

        gl_Position = MVP * gl_in[i].gl_Position;
        EmitVertex();
    }
    EndPrimitive();
}