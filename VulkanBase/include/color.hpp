#pragma once

#include <glm/glm.hpp>
#include <variant>
#include <string>

namespace color{

    const glm::vec4 black{0};
    const glm::vec4 white{1, 0, 0, 1};
    const glm::vec4 red{1, 0, 0, 1};
    const glm::vec4 green{0, 1, 0, 1};
    const glm::vec4 blue{0, 0, 1, 1};

    inline constexpr glm::vec4 rgb(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255){
        return glm::vec4(r, g, b, a) * 0.0039215686274509803921568627451f;
    }

    inline constexpr glm::vec4 rgb(uint32_t hexCode) {
        glm::vec4 c{1};

        c.r = static_cast<float>((hexCode >> 16) & 0xFF)/255.f;
        c.g = static_cast<float>((hexCode >> 8) & 0xFF)/255.f;
        c.b = static_cast<float>(hexCode & 0xFF)/255.f;

        return c;
    }

//    inline constexpr glm::vec4 rgb(float r, float g, float b, float a = 1){
//        auto c = [](auto x){ return glm::clamp(x, 0.f, 1.f); };4//        return {c(r, 1), c(g), c(b), c(a)};
//    }


    inline float luminance(glm::vec3 rgb){
        return glm::dot(rgb, {0.2126f, 0.7152f, 0.0722f});
    }

    inline glm::vec4 hsv_to_rgb(float h, float s, float v) {
        auto h_i = static_cast<int>(h*6);
        auto f = glm::fract(h*6.);
        auto p = v * (1. - s);
        auto q = v * (1. - f*s);
        auto t = v * (1. - (1. - f) * s);

        switch(h_i) {
            case 0 : return {v, t, p, 1};
            case 1 : return {q, v, p, 1};
            case 2 : return {p, v, t, 1};
            case 3 : return {p, q, v, 1};
            case 4 : return {t, p, v, 1};
            default: return {v, p, q, 1};
        }

    }


}