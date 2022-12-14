#version 460 core

#extension GL_EXT_ray_query : require
#extension GL_GOOGLE_include_directive : enable

#include "ray_query_lang.glsl"

layout(set = 0, binding = 0) uniform accelerationStructure tlas;
#include "ray_traced_shadows.glsl"

const float shine = 50;
const vec3 globalAmbience = vec3(0.2);

layout(location = 0) in struct {
    vec3 pos;
    vec3  normal;
    vec3 color;
    vec3 lightDir;
    vec3 eyes;
} v_in;

layout(location = 0) out vec4 fragColor;

void main(){
    vec3 L = normalize(v_in.lightDir);
    vec3 N = normalize(v_in.normal);
    vec3 E = normalize(v_in.eyes - v_in.pos);
    vec3 H = normalize(E + L);

    vec3 albedo = v_in.color;

    vec3 ambience = globalAmbience * albedo;
    vec3 diffuse = max(0, dot(L, N)) * albedo;
    vec3 specular = pow(max(0, dot(H, N)), shine) * albedo;

    float occlusion = shadow(v_in.pos, L * 1000, 0xff, 1);

    vec3 color = ambience + (1 - occlusion) * (diffuse + specular);

    fragColor = vec4(color, 1);
}