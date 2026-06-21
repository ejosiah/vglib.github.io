#ifndef FLUID_COMMON_USE_EXTERNAL_GLOBALS
#extension GL_EXT_scalar_block_layout : enable

#define EPSILON 1e-6

layout(set = 0, binding = 0, scalar) uniform Globals{
    ivec3 grid_size;
    vec3 dx;
    vec3 dy;
    vec3 dz;
    float dt;
    float density;
    uint wrapping_enabled;
    uint use_hermite;
    uint dimension;
    uint open_boundary_edges;
};

bool wrappingEnabled = wrapping_enabled == 1;
bool is3D = dimension == 3u;

const uint FLUID_BOUNDARY_EDGE_LEFT = 1u << 0u;
const uint FLUID_BOUNDARY_EDGE_RIGHT = 1u << 1u;
const uint FLUID_BOUNDARY_EDGE_BOTTOM = 1u << 2u;
const uint FLUID_BOUNDARY_EDGE_TOP = 1u << 3u;
const uint FLUID_BOUNDARY_EDGE_BACK = 1u << 4u;
const uint FLUID_BOUNDARY_EDGE_FRONT = 1u << 5u;

bool isOpenBoundaryEdge(uint edge) {
    return (open_boundary_edges & edge) != 0u;
}

ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
ivec3 gid3 = ivec3(gl_GlobalInvocationID.xyz);

bool outOfBounds() {
    return gid3.x >= grid_size.x || gid3.y >= grid_size.y || (is3D && gid3.z >= grid_size.z);
}

bool outOfBounds(ivec2 coord) {
    return coord.x < 0 || coord.y < 0 || coord.x >= grid_size.x || coord.y >= grid_size.y;
}

bool outOfBounds(ivec3 coord) {
    return coord.x < 0 || coord.y < 0 || coord.z < 0 ||
           coord.x >= grid_size.x || coord.y >= grid_size.y || (is3D && coord.z >= grid_size.z);
}

bool outOfBounds(vec2 uv) {
    if(wrappingEnabled) return false;
    return any(lessThan(uv, vec2(0))) || any(greaterThan(uv, vec2(1)));
}

bool outOfBounds(vec3 uv) {
    if(wrappingEnabled) return false;
    const vec3 maxUv = is3D ? vec3(1) : vec3(1, 1, 1);
    const bvec3 below = lessThan(uv, vec3(0));
    const bvec3 above = greaterThan(uv, maxUv);
    return below.x || below.y || above.x || above.y || (is3D && (below.z || above.z));
}

ivec2 clamp(ivec2 coord) {
    return clamp(coord, ivec2(0), grid_size.xy - 1);
}

ivec3 clamp(ivec3 coord) {
    const ivec3 maxCoord = is3D ? grid_size - 1 : ivec3(grid_size.xy - 1, 0);
    return clamp(coord, ivec3(0), maxCoord);
}


vec2 clamp(vec2 coord) {
    return clamp(coord, vec2(0), vec2(1));
}

vec3 clamp(vec3 coord) {
    return is3D ? clamp(coord, vec3(0), vec3(1)) : vec3(clamp(coord.xy, vec2(0), vec2(1)), 0.5);
}

bool isOutsideOpenDomain(ivec2 coord) {
    return !wrappingEnabled && outOfBounds(coord);
}

bool isOutsideOpenDomain(ivec3 coord) {
    return !wrappingEnabled && outOfBounds(coord);
}

bool outOfBoundsThroughOpenEdge(vec2 uv) {
    if(wrappingEnabled) return false;
    return (uv.x < 0.0 && isOpenBoundaryEdge(FLUID_BOUNDARY_EDGE_LEFT)) ||
           (uv.x > 1.0 && isOpenBoundaryEdge(FLUID_BOUNDARY_EDGE_RIGHT)) ||
           (uv.y < 0.0 && isOpenBoundaryEdge(FLUID_BOUNDARY_EDGE_BOTTOM)) ||
           (uv.y > 1.0 && isOpenBoundaryEdge(FLUID_BOUNDARY_EDGE_TOP));
}

bool outOfBoundsThroughOpenEdge(vec3 uv) {
    if(wrappingEnabled) return false;
    return (uv.x < 0.0 && isOpenBoundaryEdge(FLUID_BOUNDARY_EDGE_LEFT)) ||
           (uv.x > 1.0 && isOpenBoundaryEdge(FLUID_BOUNDARY_EDGE_RIGHT)) ||
           (uv.y < 0.0 && isOpenBoundaryEdge(FLUID_BOUNDARY_EDGE_BOTTOM)) ||
           (uv.y > 1.0 && isOpenBoundaryEdge(FLUID_BOUNDARY_EDGE_TOP)) ||
           (is3D && uv.z < 0.0 && isOpenBoundaryEdge(FLUID_BOUNDARY_EDGE_BACK)) ||
           (is3D && uv.z > 1.0 && isOpenBoundaryEdge(FLUID_BOUNDARY_EDGE_FRONT));
}

bool outOfBoundsThroughOpenEdge(ivec2 coord) {
    if(wrappingEnabled) return false;
    return (coord.x < 0 && isOpenBoundaryEdge(FLUID_BOUNDARY_EDGE_LEFT)) ||
           (coord.x >= grid_size.x && isOpenBoundaryEdge(FLUID_BOUNDARY_EDGE_RIGHT)) ||
           (coord.y < 0 && isOpenBoundaryEdge(FLUID_BOUNDARY_EDGE_BOTTOM)) ||
           (coord.y >= grid_size.y && isOpenBoundaryEdge(FLUID_BOUNDARY_EDGE_TOP));
}

bool outOfBoundsThroughOpenEdge(ivec3 coord) {
    if(wrappingEnabled) return false;
    return (coord.x < 0 && isOpenBoundaryEdge(FLUID_BOUNDARY_EDGE_LEFT)) ||
           (coord.x >= grid_size.x && isOpenBoundaryEdge(FLUID_BOUNDARY_EDGE_RIGHT)) ||
           (coord.y < 0 && isOpenBoundaryEdge(FLUID_BOUNDARY_EDGE_BOTTOM)) ||
           (coord.y >= grid_size.y && isOpenBoundaryEdge(FLUID_BOUNDARY_EDGE_TOP)) ||
           (is3D && coord.z < 0 && isOpenBoundaryEdge(FLUID_BOUNDARY_EDGE_BACK)) ||
           (is3D && coord.z >= grid_size.z && isOpenBoundaryEdge(FLUID_BOUNDARY_EDGE_FRONT));
}

ivec3 toCell3(ivec2 coord) {
    return ivec3(coord, gid3.z);
}

vec3 toUv3(vec2 uv) {
    return vec3(uv, 0.5);
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

layout(set = COLLISION_SET, binding = 0) uniform sampler3D collisionField;
layout(set = COLLISION_SET, binding = 1) uniform sampler3D collisionVelocityField;

vec3 collisionSampleUv(vec3 uv);
ivec3 collisionSampleCoord(ivec3 coord);
vec2 sampleCollider(vec3 uv);
vec2 sampleCollider(ivec3 coord);
bool isInsideWall(ivec3 coord);
vec2 sampleColliderVelocity(vec3 uv);
vec2 sampleColliderVelocity(ivec3 coord);
bool isInsideSolid(ivec3 coord);

ivec3 wrapCoord(ivec3 coord) {
    const ivec3 size = is3D ? grid_size : ivec3(grid_size.xy, 1);
    return (coord % size + size) % size;
}

vec2 collisionSampleUv(vec2 uv) {
    return collisionSampleUv(toUv3(uv)).xy;
}

vec3 collisionSampleUv(vec3 uv) {
    if(!is3D) {
        uv.z = 0.5;
    }

    return wrappingEnabled ? fract(uv) : clamp(uv);
}

ivec2 collisionSampleCoord(ivec2 coord) {
    return collisionSampleCoord(toCell3(coord)).xy;
}

ivec3 collisionSampleCoord(ivec3 coord) {
    if(!is3D) {
        coord.z = 0;
    }

    return wrappingEnabled ? wrapCoord(coord) : clamp(coord);
}

vec2 sampleCollider(vec2 uv) {
    return sampleCollider(toUv3(uv));
}

vec2 sampleCollider(vec3 uv) {
    return texture(collisionField, collisionSampleUv(uv)).xy;
}

vec2 sampleCollider(ivec2 coord) {
    return sampleCollider(toCell3(coord));
}

vec2 sampleCollider(ivec3 coord) {
    return texelFetch(collisionField, collisionSampleCoord(coord), 0).xy;
}

float sampleBoundary(vec2 uv) {
    return sampleCollider(uv).x;
}

float sampleBoundary(vec3 uv) {
    return sampleCollider(uv).x;
}

float sampleBoundary(ivec2 coord) {
    return sampleCollider(coord).x;
}

float sampleBoundary(ivec3 coord) {
    return sampleCollider(coord).x;
}

float sampleBoundary(vec2 uv, out uint type) {
    vec2 s = sampleCollider(uv);
    type = uint(s.y);
    return s.x;
}

float sampleBoundary(vec3 uv, out uint type) {
    vec2 s = sampleCollider(uv);
    type = uint(s.y);
    return s.x;
}

float sampleBoundary(ivec2 coord, out uint type) {
    vec2 s = sampleCollider(coord);
    type = uint(s.y);
    return s.x;
}

float sampleBoundary(ivec3 coord, out uint type) {
    vec2 s = sampleCollider(coord);
    type = uint(s.y);
    return s.x;
}

uint sampleColliderType(vec2 uv) {
    return uint(round(sampleCollider(uv).y));
}

uint sampleColliderType(vec3 uv) {
    return uint(round(sampleCollider(uv).y));
}

uint sampleColliderType(ivec2 coord) {
    return uint(round(sampleCollider(coord).y));
}

uint sampleColliderType(ivec3 coord) {
    return uint(round(sampleCollider(coord).y));
}

bool isInsideWall(ivec2 coord) {
    return isInsideWall(toCell3(coord));
}

bool isInsideWall(ivec3 coord) {
    uint type = BOUNDARY_TYPE_INVALID;
    return sampleBoundary(coord, type) < 0 && type == BOUNDARY_TYPE_WALL;
}

vec2 sampleColliderVelocity(vec2 uv) {
    return sampleColliderVelocity(toUv3(uv));
}

vec2 sampleColliderVelocity(vec3 uv) {
    return texture(collisionVelocityField, collisionSampleUv(uv)).xy;
}

vec2 sampleColliderVelocity(ivec2 coord) {
    return sampleColliderVelocity(toCell3(coord));
}

vec2 sampleColliderVelocity(ivec3 coord) {
    return texelFetch(collisionVelocityField, collisionSampleCoord(coord), 0).xy;
}

bool isInsideSolid(ivec2 coord) {
    return isInsideSolid(toCell3(coord));
}

bool isInsideSolid(ivec3 coord) {
    return sampleBoundary(coord) < 0;
}

uint marker(ivec2 coord) {
    return isInsideSolid(coord) ? 0u : 1u;
}

uint marker(ivec3 coord) {
    return isInsideSolid(coord) ? 0u : 1u;
}
#endif


vec2 get_uv() {
    return (vec2(gid) + 0.5)/grid_size.xy;
}

vec3 get_uvw() {
    vec3 uvw = (vec3(gid3) + 0.5) / vec3(grid_size);
    if(!is3D) {
        uvw.z = 0.5;
    }
    return uvw;
}

#endif // FLUID_COMMON_USE_EXTERNAL_GLOBALS
