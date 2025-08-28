#pragma once

#include "common.h"
#include "AtmosphereContants.hpp"

#include <glm/glm.hpp>
#include <array>
#include <vulkan/vulkan.h>

#include <filesystem>

constexpr uint32_t NUM_CHANNELS = 4;

using Dim2 = glm::ivec2;
using Dim3 = glm::ivec3;

using vec2 = glm::vec2;
using vec3 = glm::vec3;

constexpr uint32_t COMP_SIZE = sizeof(float) * NUM_CHANNELS;


constexpr uint32_t TRANSMISSION_DATA_SIZE = TRANSMITTANCE_TEXTURE_WIDTH * TRANSMITTANCE_TEXTURE_HEIGHT * COMP_SIZE;
constexpr uint32_t IRRADIANCE_DATA_SIZE = IRRADIANCE_TEXTURE_WIDTH * IRRADIANCE_TEXTURE_HEIGHT * COMP_SIZE;
constexpr uint32_t SCATTERING_DATA_SIZE = SCATTERING_TEXTURE_WIDTH * SCATTERING_TEXTURE_HEIGHT * SCATTERING_TEXTURE_DEPTH * COMP_SIZE;
constexpr uint32_t DATA_SIZE = TRANSMISSION_DATA_SIZE + IRRADIANCE_DATA_SIZE + SCATTERING_DATA_SIZE;

namespace Atmosphere {

    struct alignas(16) DensityProfileLayer {
        float width;
        float exp_term;
        float exp_scale;
        float linear_term;
        float constant_term;
    };

    struct Header {
        Dim3 scatteringDimensions{SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_DEPTH};
        Dim2 transmittanceDimensions{TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT};
        Dim2 irradianceDimensions{IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT};

        vec3 solarIrradiance;
        vec3 rayleighScattering;
        vec3 mieScattering;
        vec3 mieExtinction;
        vec3 absorptionExtinction;
        vec3 groundAlbedo;
        float sunAngularRadius;
        float bottomRadius;
        float topRadius;
        float mu_s_min;
        float mieAnisotropicFactor;
        float lengthUnitInMeters;
    };


    struct Format {
        Header header;
        std::vector<char> data{};
    };

    inline std::istream& operator>>(std::istream& in, Dim2& d) {
        return in >> d.x >> d.y;
    }

    inline std::istream& operator>>(std::istream& in, Dim3& d) {
        return in >> d.x >> d.y >> d.z;
    }

    inline std::istream& operator>>(std::istream& in, vec3& f) {
        return in >> f.x >> f.y >> f.z;
    }

    inline std::ostream& operator<<(std::ostream& out, const Dim2& d) {
        return out << d.x << d.y;
    }

    inline std::ostream& operator<<(std::ostream& out, const Dim3& d) {
        return out << d.x << d.y << d.z;
    }

    inline std::ostream& operator<<(std::ostream& out, const vec3& f) {
        return out << f.x << f.y << f.z;
    }

    std::istream& operator>>(std::istream& in, Format& format);

    std::ostream& operator<<(std::ostream& out, const Format& format);

    Format load(const std::filesystem::path &path);

    void save(const std::filesystem::path& path, const Format& format);

    struct Params {
        glm::vec3 solarIrradiance{1.474000, 1.850400, 1.911980};
        float sunAngularRadius{0.004675};

        struct {
            float bottom{6360 * km};
            float top{6420 * km};
        } radius;

        struct {
            glm::vec3 scattering{0.005802/km, 0.013558/km, 0.033100/km};
            float height{8 * km};
        } rayleigh;

        struct {
            glm::vec3 scattering{0.003996f/km};
            glm::vec3 extinction{0.004440/km};
            float height{1.2 * km};
            float anisotropicFactor{0.8};
        } mie;

        struct {
            struct {
                float width{25 * km};
                float linearHeight{15 * km};
                float constant{-2.0/3.0};
            } bottom;
            struct {
                float linearHeight{15 * km};
                float constant{8.0/3.0};
            } top;
            glm::vec3  absorptionExtinction{0.000650/km,0.001881/km,0.000085/km};
        } ozone;

        glm::vec3 groundAlbedo{0.1};
        float mu_s_min{glm::cos(MAX_SUN_ZENITH_ANGLE)};
        int numScatteringOrder{4};
        float lengthUnitInMeters{1 * km};
    };

    static Params Defaults{};
}

