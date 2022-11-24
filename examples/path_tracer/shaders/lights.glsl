#ifndef LIGHTS_GLSL
#define LIGHTS_GLSL

#include "scene_push_constants.glsl"
#include "eval_brdf.glsl"

layout(location = 1) rayPayload OcclusionData occData;

bool sampleLightRIS(inout RngStateType rngState, Surface surface, out LightInfo selectedSampl, out float lightSampleWeight);

LightInfo sampleLight(inout RngStateType rngState, Surface surface, out float lightWeight);

bool isVisisble(LightInfo light, Surface surface);

vec3 visibility(LightInfo light, Surface surface);

vec3 evalLightContribution(inout HitData hitData, vec3 wo);

vec3 evalLightContributionWithTranmission(inout HitData hitData, vec3 wo);

float fallOff(Light light, vec3 pointToLightDir);

vec3 evalBrdf(Surface surface, vec3 wo, vec3 wi);

bool sampleLightRIS(inout RngStateType rngState, Surface surface, out LightInfo selectedSample, out float lightSampleWeight){
    if(numLights <= 0) return false;

    float totalWeights = 0;
    float samplePdfG = 0;

    for(int i = 0; i < RIS_CANDIDATES_LIGHTS; i++){
        float lightWeight;
        LightInfo light = sampleLight(rngState, surface, lightWeight);
        if(isBlack(light.radiance)){
            continue;
        }


        if(bool(shadow_ray_in_ris) && !isVisisble(light, surface)){
            continue;
        }

        float candidatePdfG = luminance(light.radiance);
        float candidateRISWeight = candidatePdfG * lightWeight;

        totalWeights += candidateRISWeight;
        if(rand(rngState) < (candidateRISWeight / totalWeights)){
            selectedSample = light;
            samplePdfG = candidatePdfG;
        }
    }

    if(totalWeights == 0.0f){
        return false;
    }else {
        lightSampleWeight = (totalWeights / float(RIS_CANDIDATES_LIGHTS)) / samplePdfG;
        return true;
    }

}

LightInfo sampleLight(inout RngStateType rngState, Surface surface, out float lightWeight){
    LightInfo lightInfo;
    lightInfo.radiance = vec3(0);
    lightInfo.sx = surface.x;
    lightInfo.sn = surface.sN;
    lightInfo.pdf = 0;
    lightWeight = 0;
    if(numLights <= 0){
        return lightInfo;
    }
    int id = int(rand(rngState) * numLights);
    Light light = lights[id];
    lightInfo.x = light.position;
    lightInfo.n = light.normal;
    lightInfo.area = light.area;
    lightInfo.flags = light.flags;
    lightInfo.value = light.value;
    lightInfo.pdf = 0.0;

    if(isPositional(light)){
        vec3 wi = light.position - lightInfo.sx;
        lightInfo.wi = normalize(wi);
        lightInfo.dist = length(wi);
        lightInfo.NdotL = clamp(dot(lightInfo.wi, lightInfo.sn), 0.00001, 1);
        lightInfo.pdf = 1;
        lightInfo.radiance = light.value * fallOff(light, -lightInfo.wi) / (lightInfo.dist * lightInfo.dist);
    }

    if(isDistant(light)){
        lightInfo.wi = -light.normal;
        lightInfo.x = lightInfo.sx + lightInfo.wi  * 2 * worldRadius;
        lightInfo.dist = distance(lightInfo.x, lightInfo.sx);
        lightInfo.NdotL = clamp(dot(lightInfo.wi, lightInfo.sn), 0.00001, 1);
        lightInfo.pdf = 1;
        lightInfo.radiance = light.value;
    }

    if(isArea(light)){
        SceneObject sceneObj = sceneObjs[light.instanceId];
        int objId = sceneObj.objId;

        uint primitiveId = uint(floor(rand(rngState) * light.numTriangles));

        Vertex v0, v1, v2;
        getTriangle(objId, mat4x3(1), light.triangleOffset, primitiveId, v0, v1, v2);

        vec2 u = vec2(rand(rngState), rand(rngState));
        vec2 uv = uniformSampleTriangle(u);
        lightInfo.x = u.x * v0.position + u.y * v1.position + (1 - u.x - u.y) * v2.position;
        lightInfo.n = u.x * v0.normal + u.y * v1.normal + (1 - u.x - u.y) * v2.normal;

        vec3 wi = lightInfo.x - lightInfo.sx;
        lightInfo.wi = normalize(wi);
        lightInfo.dist = length(wi);
        lightInfo.NdotL = clamp(dot(lightInfo.wi, lightInfo.sn), 0.00001, 1);
        lightInfo.radiance = light.value;
        lightInfo.pdf = (lightInfo.dist * lightInfo.dist)/(lightInfo.NdotL * light.area);
    }

    if(isInfinite(light)){
        vec2 u = vec2(rand(rngState), rand(rngState));
        float pdf;
        vec2 uv = sampleContinuous2D(pConditionalVFunc, pConditionalVCdf, pMarginal, pMarginalCdf, pMarginalIntegral, u, pdf);
        if(pdf == 0){
            lightInfo.value = vec3(0);
            lightInfo.pdf = 0;
            return lightInfo;
        }
        float sinTheta;
        light.normal = -direction_from_spherical(uv, sinTheta);
        lightInfo.wi = direction_from_spherical(uv, sinTheta);
        lightInfo.x = lightInfo.sx + lightInfo.wi  * 2 * worldRadius;
        lightInfo.dist = distance(lightInfo.x, lightInfo.sx);
        lightInfo.NdotL = clamp(dot(lightInfo.wi, lightInfo.sn), 0.00001, 1);
        lightInfo.pdf = pdf / ( 2 * PI * PI * sinTheta);
        if(sinTheta == 0){
            lightInfo.pdf = 0;
        }
        lightInfo.value = texture(environmentMap, uv).rgb * envMapIntensity;
        lightInfo.radiance = texture(environmentMap, uv).rgb * envMapIntensity;
    }

    lightInfo.radiance = dot(lightInfo.wi, lightInfo.sn) < 0.00001 ? vec3(0) :  lightInfo.radiance;
    lightWeight = 1/lightInfo.pdf;
    return lightInfo;
}

bool isVisisble(LightInfo light, Surface surface){
    occData.isShadowed = true;
    occData.Continue = true;

    occData.ray.origin = offsetRay(surface.x, surface.gN);
    occData.ray.direction = normalize(light.x - surface.x);

    float dist = distance(light.x, surface.x);
    uint flags = gl_RayFlagsTerminateOnFirstHit | gl_RayFlagsSkipClosestHitShader;

    traceRay(topLevelAs, flags, (mask & ~OBJECT_TYPE_LIGHT), 0, 0, 1, occData.ray.origin, 0, occData.ray.direction, dist, 1);


    return !occData.isShadowed;
}

vec3 visibility(LightInfo light, Surface surface){
    occData.isShadowed = true;
    occData.Continue = true;
    occData.transmission = vec3(1);

    occData.ray.origin = offsetRay(surface.x, surface.gN);
    occData.ray.direction = normalize(light.x - surface.x);

    float dist = distance(light.x, surface.x);
    uint flags = gl_RayFlagsNoOpaque;

    while(occData.Continue){
        traceRay(topLevelAs, flags, (mask & ~OBJECT_TYPE_LIGHT), 3, 0, 1, occData.ray.origin, 0, occData.ray.direction, dist, 1);
    }

    return occData.transmission;
}


vec3 evalLightContribution(inout HitData hitData, vec3 wo){
    vec3 lightContribution = vec3(0);
    Surface surface = hitData.surface;

    float lightWeight;
    LightInfo light;
    if(sampleLightRIS(hitData.rngState, surface, light, lightWeight)){
        if(lightWeight != 0){
            if(bool(shadow_ray_in_ris) || isVisisble(light, surface)){
                vec3 wi = normalize(light.x - surface.x);
                lightContribution = light.radiance * evalBrdf(surface, wo, wi) * lightWeight;
            }
        }
    }

    return lightContribution;
}

vec3 evalLightContributionWithTranmission(inout HitData hitData, vec3 wo){
    vec3 lightContribution = vec3(0);
    Surface surface = hitData.surface;

    float lightWeight;
    LightInfo light;
    if(sampleLightRIS(hitData.rngState, surface, light, lightWeight)){
        if(lightWeight != 0){
            vec3 wi = normalize(light.x - surface.x);
            lightContribution = light.radiance * evalBrdf(surface, wo, wi) * lightWeight * visibility(light, surface);
        }
    }

    return lightContribution;
}

float fallOff(Light light, vec3 pointToLightDir){
    vec3 n = light.normal;
    float cutOff = light.cosWidth;
    float fallOffStart = light.fallOffStart;

    float cos0 = max(0, dot(pointToLightDir, n));
    if(cos0 < cutOff) return 0;
    if(cos0 > fallOffStart) return 1;
    return smoothstep(cutOff, fallOffStart, cos0);
}
#endif