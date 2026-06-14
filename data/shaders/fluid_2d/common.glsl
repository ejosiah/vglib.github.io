#ifndef BOUNDARY_SET
#error BOUNDARY_SET must be defined before including common.glsl
#endif

layout(set = BOUNDARY_SET, binding = 0) uniform sampler2D boundaryField;

#define st(p) (bool(ensureBoundaryCondition) ? clamp((p), vec2(0), vec2(1)) : fract(p))

bool outsideDomain(vec2 uv){
    return any(lessThan(uv, vec2(0))) || any(greaterThan(uv, vec2(1)));
}

bool isObstacle(vec2 uv){
    if(outsideDomain(uv)){
        return true;
    }

    ivec2 size = textureSize(boundaryField, 0);
    ivec2 coord = clamp(ivec2(floor(uv * vec2(size))), ivec2(0), size - ivec2(1));
    return texelFetch(boundaryField, coord, 0).r > 0.5;
}

bool checkBoundary(vec2 uv){
    return bool(ensureBoundaryCondition) && isObstacle(uv);
}

vec2 boundaryNormal(vec2 centerUv, vec2 sampleUv){
    vec2 offset = sampleUv - centerUv;
    if(abs(offset.x) > abs(offset.y)){
        return vec2(sign(offset.x), 0);
    }
    if(abs(offset.y) > 0){
        return vec2(0, sign(offset.y));
    }
    return vec2(0);
}

vec2 reflectVelocityAtBoundary(vec2 velocity, vec2 centerUv, vec2 sampleUv){
    if(!checkBoundary(sampleUv)){
        return velocity;
    }

    vec2 normal = boundaryNormal(centerUv, sampleUv);
    return velocity - 2.0 * dot(velocity, normal) * normal;
}

vec2 scalarBoundarySampleUv(vec2 centerUv, vec2 sampleUv){
    return checkBoundary(sampleUv) ? centerUv : st(sampleUv);
}

vec2 applyBoundaryCondition(vec2 uv, vec2 u){
    if(checkBoundary(uv)){
        u *= -1;
    }
    return u;
}

vec4 applyBoundaryCondition(vec2 uv, vec4 u){
    if(checkBoundary(uv)){
        u *= -1;
    }
    return u;
}
