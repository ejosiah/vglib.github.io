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
#endif

#define st(p) p

bool outOfBounds() {
    return gid.x >= grid_size.x || gid.y >= grid_size.y;
}

vec2 get_uv() {
    return (vec2(gid) + 0.5)/grid_size.xy;
}
