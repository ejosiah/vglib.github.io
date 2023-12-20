#pragma once

#include "AtmosphereContants.hpp"

#include <glm/glm.hpp>
#include <array>
#include <vulkan/vulkan.h>

#include <filesystem>

constexpr uint32_t NUM_CHANNELS = 4;

using Dim2 = glm::ivec2;
using Dim3 = glm::ivec3;

using float2 = glm::vec2;
using float3 = glm::vec3;

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

        float3 solarIrradiance;
        float3 rayleighScattering;
        float3 mieScattering;
        float3 mieExtinction;
        float3 absorptionExtinction;
        float3 groundAlbedo;
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

    inline std::istream& operator>>(std::istream& in, float3& f) {
        return in >> f.x >> f.y >> f.z;
    }

    inline std::ostream& operator<<(std::ostream& out, const Dim2& d) {
        return out << d.x << d.y;
    }

    inline std::ostream& operator<<(std::ostream& out, const Dim3& d) {
        return out << d.x << d.y << d.z;
    }

    inline std::ostream& operator<<(std::ostream& out, const float3& f) {
        return out << f.x << f.y << f.z;
    }

    std::istream& operator>>(std::istream& in, Format& format);

    std::ostream& operator<<(std::ostream& out, const Format& format);

    Format load(const std::filesystem::path &path);

    void save(const std::filesystem::path& path, const Format& format);
}