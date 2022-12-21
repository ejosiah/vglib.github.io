#version 450

#define PI 3.1415926535897932384626433832795
#include "quaternion.glsl"
#include "terrain_ubo.glsl"
#include "pbr/common.glsl"

layout(set = 1, binding = 0) uniform sampler2D albedoMap;
layout(set = 1, binding = 1) uniform sampler2D metalicMap;
layout(set = 1, binding = 2) uniform sampler2D roughnessMap;
layout(set = 1, binding = 3) uniform sampler2D normalMap;
layout(set = 1, binding = 4) uniform sampler2D aoMap;
layout(set = 1, binding = 5) uniform sampler2D displacementMap;

layout(location = 0) in struct {
    vec3 worldPosition;
    vec3 normal;
    vec2 uv;
    vec2 patch_uv;
    vec3 color;
    vec3 viewPosition;
} fs_in;

layout(location = 6) noperspective in vec3 edgeDist;


layout(location = 0) out vec4 fragColor;

const float preventDivideByZero = 0.0001;

vec3 checkerboard(){
    vec2 id = floor(numPatches * fs_in.uv);
    float c = step(1, mod(id.x + id.y, 2));
    return mix(vec3(0.5), vec3(1), c);
}


vec3 shadeFragment(){
    vec2 uv = fs_in.patch_uv;

    vec3 albedo = texture(albedoMap, uv).rgb;
    float metalness = texture(metalicMap, uv).r;
    float roughness = texture(roughnessMap, uv).r;
    roughness = (invertRoughness == 1) ? 1 - roughness : roughness;
    float ao = texture(aoMap, uv).r;

    vec3 n1 = fs_in.normal;
    vec3 n2 = 2 * texture(normalMap, uv).rgb - 1;
    vec3 N = normalize(vec3(n1.xy*n2.z + n2.xy*n1.z, n1.z*n2.z));

    // rotate normal into world space
    vec4 q = axisAngle(vec3(1, 0, 0), -PI/2);
    N = rotatePoint(q, N);
    N = normalize(N);

    vec3 viewDir = fs_in.viewPosition - fs_in.worldPosition;
    vec3 E = normalize(viewDir);
    vec3 R = reflect(-E, N);

    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metalness);

    vec3 lightDir = sunPosition;
    vec3 L = normalize(lightDir);

    vec3 H = normalize(E + L);
    float attenuation = 1;  // no attenuation for sun light
    vec3 radiance = vec3(10) * attenuation;

    // Cook-Torrance BRDF
    float NDF = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, E, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H,E), 0), F0);

    vec3 numerator    = NDF * G * F;
    float denominator = 4.0 * max(dot(N, E), 0.0) * max(dot(N, L), 0.0) + preventDivideByZero;
    vec3 specular = numerator / denominator;

    vec3 kS = F;

    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metalness;

    float NdotL = max(dot(N, L), 0.0);

    vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL;

    return Lo;
}

void main(){
    vec2 uv = fs_in.patch_uv;
    
    vec3 color = vec3(0);
    if(shading == 1){
        color = shadeFragment();
    }else{
        vec3 N = normalize(maxHeight <= 0 ? vec3(0, 1, 0) : fs_in.normal);
        vec3 albedo = checkerboard();
        vec4 q = axisAngle(vec3(1, 0, 0), -PI/2);
        N = rotatePoint(q, N);
        N = normalize(N);

        vec3 L = normalize(sunPosition);
        color = albedo * max(0, dot(N, L));
    }
    

    if(bool(wireframe)){
        float d = min(edgeDist.x, min(edgeDist.y, edgeDist.z));
        float t = smoothstep(-wireframeWidth, wireframeWidth, d);

        vec3 wcolor = lod == 1 && tessLevelColor == 1 ? fs_in.color : wireframeColor;
        color = mix(wcolor, color, t);
    }

    fragColor.rgb = color/(1 + color);
}