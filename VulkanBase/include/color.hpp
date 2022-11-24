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

//    inline constexpr glm::vec4 rgb(float r, float g, float b, float a = 1){
//        auto c = [](auto x){ return glm::clamp(x, 0.f, 1.f); };
//        return {c(r), c(g), c(b), c(a)};
//    }

    inline constexpr glm::vec4 rgb(const std::string& hexRep);

    inline constexpr glm::vec4 rgb(uint32_t hexRep){
        uint8_t r = (hexRep >> 16) & 0xFF;
        uint8_t g = (hexRep >> 8) & 0xFF;
        uint8_t b = hexRep & 0xFF;
        return rgb(r, g, b);
    }

    inline float luminance(glm::vec3 rgb){
        return glm::dot(rgb, {0.2126f, 0.7152f, 0.0722f});
    }
}