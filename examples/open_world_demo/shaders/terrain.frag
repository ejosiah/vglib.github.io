#version 450

#define PI 3.1415926535897932384626433832795
#include "quaternion.glsl"
#include "terrain_ubo.glsl"

layout(set = 1, binding = 0) uniform sampler2D albedoMap;
layout(set = 1, binding = 1) uniform sampler2D normalMap;

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv_in;
layout(location = 3) in vec2 patch_uv;
layout(location = 4) noperspective in vec3 edgeDist;
layout(location = 5) in vec3 color_in;

layout(location = 0) out vec4 fragColor;

vec3 checkerboard(){
    vec2 id = floor(numPatches * uv_in);
    float c = step(1, mod(id.x + id.y, 2));
    return mix(vec3(0.5), vec3(1), c);
}

void main(){
    vec2 uv = patch_uv;

    vec3 N = vec3(0);
    vec3 albedo = vec3(0);

    if(shading == 1){
        vec3 n1 = normal;
        vec3 n2 = 2 * texture(normalMap, uv).rgb - 1;
        N = normalize(vec3(n1.xy*n2.z + n2.xy*n1.z, n1.z*n2.z));
        albedo = texture(albedoMap, patch_uv).rgb;
    }else{
        N = normalize(maxHeight <= 0 ? vec3(0, 1, 0) : normal);
        albedo = checkerboard();
    }


    vec4 q = axisAngle(vec3(1, 0, 0), -PI/2);
    N = rotatePoint(q, N);
    N = normalize(N);

    vec3 L = normalize(sunPosition);
    vec3 diffuse = albedo * max(0, dot(N, L));

    if(bool(wireframe)){
        float d = min(edgeDist.x, min(edgeDist.y, edgeDist.z));
        float t = smoothstep(-wireframeWidth, wireframeWidth, d);

        vec3 color = lod == 1 && tessLevelColor == 1 ? color_in : wireframeColor;
        diffuse = mix(color, diffuse, t);
    }

    fragColor.rgb =  diffuse;
}