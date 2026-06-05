#extension GL_EXT_scalar_block_layout : enable

layout(set = 0, binding = 0, scalar) uniform Globals{
    ivec2 grid_size;
    vec2 dx;
    vec2 dy;
    float dt;
    float density;
    int ensure_boundary_condition;
    int use_hermite;
};

ivec2 gid = ivec2(gl_GlobalInvocationID);

#ifdef BOUNDARY_SET
#define USE_BOUNDARY_TEXTURE 1
#else
#define USE_BOUNDARY_TEXTURE 0
#endif

#if USE_BOUNDARY_TEXTURE
layout(set = BOUNDARY_SET, binding = 0) uniform sampler2D boundaryField;
#endif

#define st(p) (bool(ensure_boundary_condition) ? clamp((p), vec2(0), vec2(1)) : fract(p))

bool outsideDomain(vec2 uv){
    return any(lessThan(uv, vec2(0))) || any(greaterThan(uv, vec2(1)));
}

bool isObstacle(vec2 uv){
    if(outsideDomain(uv)){
        return true;
    }

#if USE_BOUNDARY_TEXTURE
    ivec2 size = textureSize(boundaryField, 0);
    ivec2 coord = clamp(ivec2(floor(uv * vec2(size))), ivec2(0), size - ivec2(1));
    return texelFetch(boundaryField, coord, 0).r > 0.5;
#else
    return uv.x <= 0 || uv.x >= 1 || uv.y <= 0 || uv.y >= 1;
#endif
}

bool checkBoundary(vec2 uv){
    return bool(ensure_boundary_condition) && isObstacle(uv);
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

float reflectVelocityComponentAtBoundary(float velocity, uint component, vec2 centerUv, vec2 sampleUv){
    if(!checkBoundary(sampleUv)){
        return velocity;
    }

    vec2 normal = boundaryNormal(centerUv, sampleUv);
    if((component == 1 && abs(normal.x) > 0) || (component == 2 && abs(normal.y) > 0)){
        return -velocity;
    }
    return velocity;
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

bool outOfBounds() {
    return gid.x >= grid_size.x || gid.y >= grid_size.y;
}

vec2 get_uv() {
    return (vec2(gid) + 0.5)/grid_size.xy;
}
