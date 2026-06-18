#version 460

layout(set = 0, binding = 0) uniform sampler2D pressure_field;

layout(set = 1, binding = 0) buffer MinMax {
    float data;
} min_max[2];

layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 fragColor;

vec3 heatMap(float t) {
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

void main() {
    vec2 uv = vUv;
    float pressure = texture(pressure_field, uv).x;
    float minPressure = min_max[0].data;
    float maxPressure = min_max[1].data;
    float pressureScale = max(abs(minPressure), abs(maxPressure));
    float t = pressureScale > 1e-6 ? 0.5 + 0.5 * pressure / pressureScale : 0.5;

    fragColor = vec4(heatMap(t), 1.0);
}
