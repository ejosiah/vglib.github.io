layout(set = 0, binding = 0, scalar) uniform Globals{
    ivec2 grid_size;
    vec2 dx;
    vec2 dy;
    float dt;
    uint ensure_boundary_condition;
};

layout(set = 1, binding = 10) uniform sampler2D gTextures[];
layout(set = 1, binding = 11) uniform writeonly image2D gImages[];
layout(set = 1, binding = 12) uniform texture2D gTexture2D[];
layout(set = 1, binding = 13) uniform sampler gSamplers[];

#define st(p) ((ensure_boundary_condition == 0) ? p : fract(p))

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

vec2 force(vec2 xy, uint id) {
    return texture(gTextures[id], xy).xy;
}

vec2 vectorField(vec2 xy, uvec2 id) {
    vec2 u;
    u.x = texture(gTextures[id.x], xy).x;
    u.y = texture(gTextures[id.y], xy).x;

    return u;
}

void updateVectorField(vec2 u, uvec2 id) {
    imageStore(gImages[id.x], gid, vec4(u.x, 0,  0, 0));
    imageStore(gImages[id.y], gid, vec4(u.y, 0,  0, 0));
}

vec2 get_uv() {
    return (vec2(gl_GlobalInvocationID) + 0.5)/grid_size.xy;
}