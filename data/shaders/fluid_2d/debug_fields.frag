#version 460

layout(set = 0, binding = 0) uniform sampler2D field0;
layout(set = 1, binding = 0) uniform sampler2D field1;
layout(set = 2, binding = 0) uniform sampler2D field2;
layout(set = 3, binding = 0) uniform sampler2D field3;
layout(set = 4, binding = 0) uniform sampler2D field4;
layout(set = 5, binding = 0) uniform sampler2D field5;
layout(set = 6, binding = 0) uniform sampler2D field6;
layout(set = 7, binding = 0) uniform sampler2D field7;
layout(set = 8, binding = 0) uniform sampler2D field8;
layout(set = 9, binding = 0) uniform sampler2D field9;
layout(set = 10, binding = 0) uniform sampler2D field10;
layout(set = 11, binding = 0) uniform sampler2D field11;

layout(set = 12, binding = 0) buffer MinMax {
    float data;
} min_max[2];

layout(push_constant) uniform Constants {
    uint fieldCount;
    uint columns;
    uint rows;
    uint closedDomain;
    uint openBoundaryEdges;
};

layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 fragColor;

const float PI = 3.14159265358979323846;
const uint BOUNDARY_EDGE_LEFT = 1u << 0u;
const uint BOUNDARY_EDGE_RIGHT = 1u << 1u;
const uint BOUNDARY_EDGE_BOTTOM = 1u << 2u;
const uint BOUNDARY_EDGE_TOP = 1u << 3u;

vec4 sampleField(uint index, vec2 uv) {
    switch(index) {
        case 0: return texture(field0, uv);
        case 1: return texture(field1, uv);
        case 2: return texture(field2, uv);
        case 3: return texture(field3, uv);
        case 4: return texture(field4, uv);
        case 5: return texture(field5, uv);
        case 6: return texture(field6, uv);
        case 7: return texture(field7, uv);
        case 8: return texture(field8, uv);
        case 9: return texture(field9, uv);
        case 10: return texture(field10, uv);
        case 11: return texture(field11, uv);
    }

    return vec4(0);
}

bool bad(vec4 value) {
    return any(isnan(value)) || any(isinf(value));
}

vec3 signedColor(float value) {
    float t = 0.5 + atan(value) / PI;
    vec3 cold = vec3(0.08, 0.20, 0.95);
    vec3 neutral = vec3(0.08);
    vec3 hot = vec3(1.0, 0.18, 0.04);
    return t < 0.5
        ? mix(cold, neutral, t * 2.0)
        : mix(neutral, hot, (t - 0.5) * 2.0);
}

vec3 tenMinutePhysicsColor(float t) {
    t = clamp(t, 0.0, 0.999999);

    float band = floor(4.0 * t);
    float localT = fract(4.0 * t);

    if(band < 1.0) {
        return vec3(0.0, localT, 1.0);
    }
    if(band < 2.0) {
        return vec3(0.0, 1.0, 1.0 - localT);
    }
    if(band < 3.0) {
        return vec3(localT, 1.0, 0.0);
    }
    return vec3(1.0, 1.0 - localT, 0.0);
}

vec3 pressureColor(float pressure) {
    float minPressure = min_max[0].data;
    float maxPressure = min_max[1].data;
    float pressureScale = max(abs(minPressure), abs(maxPressure));
    float t = pressureScale > 1e-6 ? 0.5 + 0.5 * pressure / pressureScale : 0.5;
    return tenMinutePhysicsColor(t);
}

vec3 vectorColor(vec2 value) {
    float m = length(value);
    if(m < 1e-7) {
        return vec3(0.04);
    }

    vec2 dir = normalize(value) * 0.5 + 0.5;
    float intensity = m / (1.0 + m);
    return vec3(dir, intensity);
}

vec3 quantityColor(vec4 value) {
    vec3 signedChannels = 0.5 + atan(value.xyz) / PI;
    float intensity = length(value.xyz) / (1.0 + length(value.xyz));
    return mix(vec3(0.04), signedChannels, max(intensity, 0.25));
}

bool isEdgeClosed(uint edge) {
    return (openBoundaryEdges & edge) == 0u;
}

bool onClosedDomainBoundary(vec2 uv) {
    if(closedDomain == 0u) return false;

    ivec2 size = textureSize(field6, 0);
    ivec2 coord = clamp(ivec2(floor(uv * vec2(size))), ivec2(0), size - ivec2(1));

    return (coord.x == 0 && isEdgeClosed(BOUNDARY_EDGE_LEFT)) ||
           (coord.x == size.x - 1 && isEdgeClosed(BOUNDARY_EDGE_RIGHT)) ||
           (coord.y == 0 && isEdgeClosed(BOUNDARY_EDGE_BOTTOM)) ||
           (coord.y == size.y - 1 && isEdgeClosed(BOUNDARY_EDGE_TOP));
}

vec3 boundaryColor(vec4 value, vec2 uv) {
    float boundary = max(1.0 - smoothstep(0.45, 0.55, value.x), onClosedDomainBoundary(uv) ? 1.0 : 0.0);
    return mix(vec3(0.035), vec3(1.0, 0.03, 0.02), boundary);
}

vec3 fieldColor(uint index, vec4 value, vec2 uv) {
    if(bad(value)) {
        return vec3(1.0, 0.0, 1.0);
    }

    if(index == 2) {
        return pressureColor(value.x);
    }

    if(index == 4 || index + 1 == fieldCount) {
        return vectorColor(value.xy);
    }

    if(index == 6) {
        return boundaryColor(value, uv);
    }

    if(index >= 8) {
        return quantityColor(value);
    }

    return signedColor(value.x);
}

void main() {
    vec2 grid = vec2(max(columns, 1), max(rows, 1));
    vec2 layoutUv = vec2(vUv.x, 1.0 - vUv.y);
    ivec2 tile = ivec2(clamp(floor(layoutUv * grid), vec2(0), grid - 1.0));
    uint index = uint(tile.y) * columns + uint(tile.x);

    vec2 tileUv = fract(layoutUv * grid);
    if(index >= fieldCount) {
        fragColor = vec4(0.01, 0.01, 0.01, 1.0);
        return;
    }

    vec4 value = sampleField(index, vec2(tileUv.x, 1.0 - tileUv.y));
    vec3 color = fieldColor(index, value, vec2(tileUv.x, 1.0 - tileUv.y));

    vec2 edge = min(tileUv, 1.0 - tileUv);
    if(edge.x < 0.01 || edge.y < 0.01) {
        color = vec3(0.0);
    }

    fragColor = vec4(color, 1.0);
}
