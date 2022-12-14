#version 460

layout(set = 0, binding = 0) uniform sampler2D colorMap;
layout(set = 0, binding = 1) uniform sampler2D intensityMap;

layout(push_constant) uniform SETTINGS{
    int gammaOn;
    int hdrOn;
    int bloomOn;
    float exposure;
};

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 fragColor;

void main(){
    vec3 color = texture(colorMap, uv).rgb;
    vec3 blur = texture(intensityMap, uv).rgb;

    color += bool(bloomOn) ? blur : vec3(0);

    if(bool(hdrOn)){
        color = 1 - exp(-color * exposure);
    }

    float gamma = bool(gammaOn) ? 2.2 : 1.0;
    color = pow(color, vec3(1/gamma));

    fragColor = vec4(color, 1);
}

