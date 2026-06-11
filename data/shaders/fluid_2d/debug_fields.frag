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

layout(push_constant) uniform Constants {
    uint fieldCount;
    uint columns;
    uint rows;
};

layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 fragColor;

const float PI = 3.14159265358979323846;

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

vec3 fieldColor(uint index, vec4 value) {
    if(bad(value)) {
        return vec3(1.0, 0.0, 1.0);
    }

    if(index == 4 || index == 6 || index + 1 == fieldCount) {
        return vectorColor(value.xy);
    }

    if(index >= 7) {
        return quantityColor(value);
    }

    return signedColor(value.x);
}

void main() {
    vec2 grid = vec2(max(columns, 1), max(rows, 1));
    ivec2 tile = ivec2(clamp(floor(vUv * grid), vec2(0), grid - 1.0));
    uint index = uint(tile.y) * columns + uint(tile.x);

    vec2 tileUv = fract(vUv * grid);
    if(index >= fieldCount) {
        fragColor = vec4(0.01, 0.01, 0.01, 1.0);
        return;
    }

    vec4 value = sampleField(index, tileUv);
    vec3 color = fieldColor(index, value);

    vec2 edge = min(tileUv, 1.0 - tileUv);
    if(edge.x < 0.01 || edge.y < 0.01) {
        color = vec3(0.0);
    }

    fragColor = vec4(color, 1.0);
}
