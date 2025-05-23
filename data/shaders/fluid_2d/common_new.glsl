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

bool checkBoundary(vec2 uv){
    return bool(ensure_boundary_condition) && (uv.x <= 0 || uv.x >= 1 || uv.y <= 0 || uv.y >= 1);
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