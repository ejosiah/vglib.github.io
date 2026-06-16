#ifndef FLUID_COMMON_USE_EXTERNAL_GLOBALS
#extension GL_EXT_scalar_block_layout : enable

#define EPSILON 1e-6

layout(set = 0, binding = 0, scalar) uniform Globals{
    ivec2 grid_size;
    vec2 dx;
    vec2 dy;
    float dt;
    float density;
    int ensure_boundary_condition;
    int use_hermite;
    int use_collider;
};

ivec2 gid = ivec2(gl_GlobalInvocationID);

bool outOfBounds() {
    return gid.x >= grid_size.x || gid.y >= grid_size.y;
}


bool outOfBounds(ivec2 coord) {
    return coord.x >= grid_size.x || coord.y >= grid_size.y;
}

ivec2 clamp(ivec2 coord) {
    return clamp(coord, ivec2(0), grid_size - 1);
}


vec2 clamp(vec2 coord) {
    return clamp(coord, vec2(0), vec2(1));
}


#ifdef COLLISION_SET
#define USE_COLLISION_TEXTURE 1
#else
#define USE_COLLISION_TEXTURE 0
#endif // COLLISION_SET

#if USE_COLLISION_TEXTURE

#define MARKER_TYPE_SOLID 0
#define MARKER_TYPE_FLUID 1

#define BOUNDARY_TYPE_INVALID (~0u)
#define BOUNDARY_TYPE_WALL 0u
#define BOUNDARY_TYPE_SDF 1u

layout(set = COLLISION_SET, binding = 0) uniform sampler2D collisionField;
layout(set = COLLISION_SET, binding = 1) uniform sampler2D collisionVelocityField;

vec2 collisionSampleUv(vec2 uv) {
    return bool(ensure_boundary_condition) ? clamp(uv) : fract(uv);
}

ivec2 collisionSampleCoord(ivec2 coord) {
    if(bool(ensure_boundary_condition)) {
        return clamp(coord);
    }

    return (coord % grid_size + grid_size) % grid_size;
}

vec2 sampleCollider(vec2 uv) {
    return texture(collisionField, collisionSampleUv(uv)).xy;
}

vec2 sampleCollider(ivec2 coord) {
    return texelFetch(collisionField, collisionSampleCoord(coord), 0).xy;
}

float sampleBoundary(vec2 uv) {
    return sampleCollider(uv).x;
}

float sampleBoundary(ivec2 coord) {
    return sampleCollider(coord).x;
}

float sampleBoundary(vec2 uv, out uint type) {
    vec2 s = sampleCollider(uv);
    type = uint(s.y);
    return s.x;
}

float sampleBoundary(ivec2 coord, out uint type) {
    vec2 s = sampleCollider(coord);
    type = uint(s.y);
    return s.x;
}

uint sampleColliderType(vec2 uv) {
    return uint(round(sampleCollider(uv).y));
}

uint sampleColliderType(ivec2 coord) {
    return uint(round(sampleCollider(coord).y));
}

bool isInsideWall(ivec2 coord) {
    uint type = BOUNDARY_TYPE_INVALID;
    return sampleBoundary(coord, type) < 0 && type == BOUNDARY_TYPE_WALL;
}

vec2 sampleColliderVelocity(vec2 uv) {
    return texture(collisionVelocityField, collisionSampleUv(uv)).xy;
}

vec2 sampleColliderVelocity(ivec2 coord) {
    return texelFetch(collisionVelocityField, collisionSampleCoord(coord), 0).xy;
}

bool isInsideSolid(ivec2 coord) {
    return sampleBoundary(coord) < 0;
}

uint marker(ivec2 coord) {
    return isInsideSolid(coord) ? 0u : 1u;
}
#endif


vec2 get_uv() {
    return (vec2(gid) + 0.5)/grid_size.xy;
}

#endif // FLUID_COMMON_USE_EXTERNAL_GLOBALS
