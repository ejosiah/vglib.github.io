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
layout(set = BOUNDARY_SET, binding = 0) uniform sampler2D boundaryField;

ivec2 boundaryCoord(ivec2 coord) {
    return clamp(coord, ivec2(0), grid_size - ivec2(1));
}

float boundaryMask(ivec2 coord) {
    if(ensure_boundary_condition == 0) {
        return 0.0;
    }

    return texelFetch(boundaryField, boundaryCoord(coord), 0).r;
}

bool boundaryIsSolid(ivec2 coord) {
    return boundaryMask(coord) > 0.5;
}

vec2 boundaryGradient(ivec2 coord) {
    float L = boundaryMask(coord - ivec2(1, 0));
    float R = boundaryMask(coord + ivec2(1, 0));
    float B = boundaryMask(coord - ivec2(0, 1));
    float T = boundaryMask(coord + ivec2(0, 1));

    return vec2(R - L, T - B);
}

bool boundaryHasZeroGradient(ivec2 coord) {
    vec2 grad = boundaryGradient(coord);
    return boundaryIsSolid(coord) && dot(grad, grad) < 1e-12;
}
#endif

#define st(p) ensure_boundary_condition == 1 ? clamp(p, vec2(0), vec2(1)) : fract(p)

bool outOfBounds() {
    return gid.x >= grid_size.x || gid.y >= grid_size.y;
}

bool outOfBounds(ivec2 coord) {
    return coord.x < 0 || coord.y < 0 || coord.x >= grid_size.x || coord.y >= grid_size.y;
}

vec2 get_uv() {
    return (vec2(gid) + 0.5)/grid_size.xy;
}
