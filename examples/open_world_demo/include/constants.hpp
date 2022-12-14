#pragma once
#include "common.h"

constexpr float EARTH_RADIUS = 6371 * km;
constexpr float CLOUD_MIN = 1.5 * km;
constexpr float CLOUD_MAX = 5.0 * km;
constexpr float SUN_DISTANCE = 100000 * km;
constexpr float MAX_HEIGHT = 8.849 * km;

constexpr uint32_t TRANSMITTANCE_TEXTURE_WIDTH = 256;
constexpr uint32_t TRANSMITTANCE_TEXTURE_HEIGHT = 64;
constexpr uint32_t SCATTERING_TEXTURE_WIDTH = 256;
constexpr uint32_t SCATTERING_TEXTURE_HEIGHT = 128;
constexpr uint32_t SCATTERING_TEXTURE_DEPTH = 32;
constexpr uint32_t IRRADIANCE_TEXTURE_WIDTH = 64;
constexpr uint32_t IRRADIANCE_TEXTURE_HEIGHT = 16;
