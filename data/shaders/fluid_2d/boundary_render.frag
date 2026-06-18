#version 460

#extension GL_EXT_scalar_block_layout : enable

layout(set = 0, binding = 0) uniform sampler2D colliderField;

layout(push_constant, scalar) uniform Constants {
    mat4 transform;
    vec4 color;
    uint closedDomain;
    uint openBoundaryEdges;
    uint showColliders;
};

layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 fragColor;

const uint BOUNDARY_EDGE_LEFT = 1u << 0u;
const uint BOUNDARY_EDGE_RIGHT = 1u << 1u;
const uint BOUNDARY_EDGE_BOTTOM = 1u << 2u;
const uint BOUNDARY_EDGE_TOP = 1u << 3u;
const int COLLIDER_RADIUS = 2;

bool isEdgeClosed(uint edge) {
    return (openBoundaryEdges & edge) == 0u;
}

bool onClosedDomainBoundary(ivec2 coord, ivec2 size) {
    if(closedDomain == 0u) return false;

    return (coord.x == 0 && isEdgeClosed(BOUNDARY_EDGE_LEFT)) ||
           (coord.x == size.x - 1 && isEdgeClosed(BOUNDARY_EDGE_RIGHT)) ||
           (coord.y == 0 && isEdgeClosed(BOUNDARY_EDGE_BOTTOM)) ||
           (coord.y == size.y - 1 && isEdgeClosed(BOUNDARY_EDGE_TOP));
}

bool nearCollider(ivec2 base, ivec2 size) {
    if(showColliders == 0u) return false;

    float boundary = 1.0;
    for(int y = -COLLIDER_RADIUS; y <= COLLIDER_RADIUS; ++y) {
        for(int x = -COLLIDER_RADIUS; x <= COLLIDER_RADIUS; ++x) {
            ivec2 coord = clamp(base + ivec2(x, y), ivec2(0), size - ivec2(1));
            boundary = min(boundary, texelFetch(colliderField, coord, 0).r);
        }
    }

    return boundary <= 0.5;
}

void main() {
    ivec2 size = textureSize(colliderField, 0);
    ivec2 base = clamp(ivec2(floor(vUv * vec2(size))), ivec2(0), size - ivec2(1));

    if(!onClosedDomainBoundary(base, size) && !nearCollider(base, size)) {
        discard;
    }

    fragColor = color;
}
