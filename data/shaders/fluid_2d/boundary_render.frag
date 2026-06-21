#version 460

#extension GL_EXT_scalar_block_layout : enable

layout(set = 0, binding = 0) uniform sampler3D colliderField;

layout(push_constant, scalar) uniform Constants {
    mat4 transform;
    vec4 color;
    uint closedDomain;
    uint openBoundaryEdges;
    uint showColliders;
    float boundaryWidth;
};

layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 fragColor;

const uint BOUNDARY_EDGE_LEFT = 1u << 0u;
const uint BOUNDARY_EDGE_RIGHT = 1u << 1u;
const uint BOUNDARY_EDGE_BOTTOM = 1u << 2u;
const uint BOUNDARY_EDGE_TOP = 1u << 3u;
bool isEdgeClosed(uint edge) {
    return (openBoundaryEdges & edge) == 0u;
}

bool onClosedDomainBoundary(vec2 coord, ivec2 size) {
    if(closedDomain == 0u) return false;

    float halfWidth = boundaryWidth * 0.5;
    vec2 gridSize = vec2(size);

    return (coord.x < halfWidth && isEdgeClosed(BOUNDARY_EDGE_LEFT)) ||
           (coord.x > gridSize.x - halfWidth && isEdgeClosed(BOUNDARY_EDGE_RIGHT)) ||
           (coord.y < halfWidth && isEdgeClosed(BOUNDARY_EDGE_BOTTOM)) ||
           (coord.y > gridSize.y - halfWidth && isEdgeClosed(BOUNDARY_EDGE_TOP));
}

bool nearCollider(ivec2 base, ivec2 size) {
    if(showColliders == 0u) return false;

    int radius = int(ceil(boundaryWidth));
    float boundary = 1.0;
    for(int y = -radius; y <= radius; ++y) {
        for(int x = -radius; x <= radius; ++x) {
            ivec2 coord = clamp(base + ivec2(x, y), ivec2(0), size - ivec2(1));
            boundary = min(boundary, texelFetch(colliderField, ivec3(coord, 0), 0).r);
        }
    }

    return boundary <= 0.5;
}

void main() {
    ivec2 size = textureSize(colliderField, 0).xy;
    vec2 coord = vUv * vec2(size);
    ivec2 base = clamp(ivec2(floor(coord)), ivec2(0), size - ivec2(1));

    if(!onClosedDomainBoundary(coord, size) && !nearCollider(base, size)) {
        discard;
    }

    fragColor = color;
}
