#ifndef ATMOSPHERE_LUT_GLSL
#define ATMOSPHERE_LUT_GLSL

#ifndef ATMOSPHERE_META_DATA
#define ATMOSPHERE_META_DATA
#define DENSITY_PROFILE_RAYLEIGH 0
#define DENSITY_PROFILE_MIE 1
#define DENSITY_PROFILE_OZONE_BOTTOM 2
#define DENSITY_PROFILE_OZONE_TOP 3
#define NUM_DENSITY_PROFILES 4

#define TEMPLATE(x)
#define TEMPLATE_ARGUMENT(x)
#define assert(x)
const int TRANSMITTANCE_TEXTURE_WIDTH = 256;
const int TRANSMITTANCE_TEXTURE_HEIGHT = 64;
const int SCATTERING_TEXTURE_R_SIZE = 32;
const int SCATTERING_TEXTURE_MU_SIZE = 128;
const int SCATTERING_TEXTURE_MU_S_SIZE = 32;
const int SCATTERING_TEXTURE_NU_SIZE = 8;
const int IRRADIANCE_TEXTURE_WIDTH = 64;
const int IRRADIANCE_TEXTURE_HEIGHT = 16;
#define COMBINED_SCATTERING_TEXTURES

#define Length float
#define Wavelength float
#define Angle float
#define SolidAngle float
#define Power float
#define LuminousPower float
#define Number float
#define InverseLength float
#define Area float
#define Volume float
#define NumberDensity float
#define Irradiance float
#define Radiance float
#define SpectralPower float
#define SpectralIrradiance float
#define SpectralRadiance float
#define SpectralRadianceDensity float
#define ScatteringCoefficient float
#define InverseSolidAngle float
#define LuminousIntensity float
#define Luminance float
#define Illuminance float
#define AbstractSpectrum vec3
#define DimensionlessSpectrum vec3
#define PowerSpectrum vec3
#define IrradianceSpectrum vec3
#define RadianceSpectrum vec3
#define RadianceDensitySpectrum vec3
#define ScatteringSpectrum vec3
#define Position vec3
#define Direction vec3
#define Luminance3 vec3
#define Illuminance3 vec3
#define TransmittanceTexture sampler2D
#define AbstractScatteringTexture sampler3D
#define ReducedScatteringTexture sampler3D
#define ScatteringTexture sampler3D
#define ScatteringDensityTexture sampler3D
#define IrradianceTexture sampler2D
const Length m = 1.0;
const Wavelength nm = 1.0;
const Angle rad = 1.0;
const SolidAngle sr = 1.0;
const Power watt = 1.0;
const LuminousPower lm = 1.0;
const float PI = 3.14159265358979323846;
const Length km = 1000.0 * m;
const Area m2 = m * m;
const Volume m3 = m * m * m;
const Angle pi = PI * rad;
const Angle deg = pi / 180.0;
const Irradiance watt_per_square_meter = watt / m2;
const Radiance watt_per_square_meter_per_sr = watt / (m2 * sr);
const SpectralIrradiance watt_per_square_meter_per_nm = watt / (m2 * nm);
const SpectralRadiance watt_per_square_meter_per_sr_per_nm =
watt / (m2 * sr * nm);
const SpectralRadianceDensity watt_per_cubic_meter_per_sr_per_nm =
watt / (m3 * sr * nm);
const LuminousIntensity cd = lm / sr;
const LuminousIntensity kcd = 1000.0 * cd;
const Luminance cd_per_square_meter = cd / m2;
const Luminance kcd_per_square_meter = kcd / m2;
struct DensityProfileLayer {
Length width;
    Number exp_term;
    InverseLength exp_scale;
    InverseLength linear_term;
    Number constant_term;
};
struct DensityProfile {
    DensityProfileLayer layers[2];
};
struct AtmosphereParameters {
IrradianceSpectrum solar_irradiance;
    Angle sun_angular_radius;
    Length bottom_radius;
    Length top_radius;
    DensityProfile rayleigh_density;
    ScatteringSpectrum rayleigh_scattering;
    DensityProfile mie_density;
    ScatteringSpectrum mie_scattering;
    ScatteringSpectrum mie_extinction;
    Number mie_phase_function_g;
    DensityProfile absorption_density;
    ScatteringSpectrum absorption_extinction;
    DimensionlessSpectrum ground_albedo;
    Number mu_s_min;
};

#endif // ATMOSPHERE_META_DATA

#include "atmosphere_model.glsl"

layout(set = 1, binding = 0) uniform sampler2D irradiance_texture;
layout(set = 1, binding = 1) uniform sampler2D transmittance_texture;
layout(set = 1, binding = 2) uniform sampler3D scattering_texture;
layout(set = 1, binding = 3) uniform sampler3D single_mie_scattering_texture;

#ifdef RADIANCE_API_ENABLED
RadianceSpectrum GetSolarRadiance() {
    return ATMOSPHERE.solar_irradiance /
    (PI * ATMOSPHERE.sun_angular_radius * ATMOSPHERE.sun_angular_radius);
}
RadianceSpectrum GetSkyRadiance(Position camera, Direction view_ray, Length shadow_length, Direction sun_direction, out DimensionlessSpectrum transmittance) {
    camera /= ubo.lengthUnitInMeters;
    shadow_length /= ubo.lengthUnitInMeters;
    return GetSkyRadiance(ATMOSPHERE, transmittance_texture,scattering_texture, single_mie_scattering_texture, camera, view_ray, shadow_length, sun_direction, transmittance);
}

RadianceSpectrum GetSkyRadianceToPoint(Position camera, Position point, Length shadow_length, Direction sun_direction, out DimensionlessSpectrum transmittance) {
    camera /= ubo.lengthUnitInMeters;
    point /= ubo.lengthUnitInMeters;
    shadow_length /= ubo.lengthUnitInMeters;
    return GetSkyRadianceToPoint(ATMOSPHERE, transmittance_texture,scattering_texture, single_mie_scattering_texture, camera, point, shadow_length, sun_direction, transmittance);
}

IrradianceSpectrum GetSunAndSkyIrradiance(Position p, Direction normal, Direction sun_direction, out IrradianceSpectrum sky_irradiance) {
    p /= ubo.lengthUnitInMeters;
    return GetSunAndSkyIrradiance(ATMOSPHERE, transmittance_texture, irradiance_texture, p, normal, sun_direction, sky_irradiance);
}
#endif// RADIANCE_API_ENABLED

Luminance3 GetSkyLuminance(Position camera, Direction view_ray, Length shadow_length,
                           Direction sun_direction, out DimensionlessSpectrum transmittance){

    camera /= ubo.lengthUnitInMeters;
    shadow_length /= ubo.lengthUnitInMeters;

    return
        GetSkyRadiance(ATMOSPHERE, transmittance_texture, scattering_texture, single_mie_scattering_texture,
                camera, view_ray, shadow_length, sun_direction, transmittance) * SKY_SPECTRAL_RADIANCE_TO_LUMINANCE;
}

Luminance3 GetSkyLuminanceToPoint(
    Position camera,
    Position point,
    Length shadow_length,
    Direction sun_direction,
    out DimensionlessSpectrum transmittance
) {
    camera /= ubo.lengthUnitInMeters;
    point /= ubo.lengthUnitInMeters;
    shadow_length /= ubo.lengthUnitInMeters;

    return GetSkyRadianceToPoint(ATMOSPHERE, transmittance_texture, scattering_texture, single_mie_scattering_texture,
                    camera, point, shadow_length, sun_direction, transmittance) * SKY_SPECTRAL_RADIANCE_TO_LUMINANCE;
}

Illuminance3 GetSunAndSkyIlluminance(Position p, Direction normal, Direction sun_direction, out IrradianceSpectrum sky_irradiance) {
    p /= ubo.lengthUnitInMeters;

    IrradianceSpectrum sun_irradiance =
        GetSunAndSkyIrradiance(ATMOSPHERE, transmittance_texture, irradiance_texture, p, normal, sun_direction, sky_irradiance);
    sky_irradiance *= SKY_SPECTRAL_RADIANCE_TO_LUMINANCE;
    return sun_irradiance * SUN_SPECTRAL_RADIANCE_TO_LUMINANCE;
}

#ifdef USE_LUMINANCE
#define GetSolarRadiance GetSolarLuminance
#define GetSkyRadiance GetSkyLuminance
#define GetSkyRadianceToPoint GetSkyLuminanceToPoint
#define GetSunAndSkyIrradiance GetSunAndSkyIlluminance
#endif

vec3 GetSolarRadiance();

vec3 GetSkyRadiance(vec3 camera, vec3 view_ray, float shadow_length, vec3 sun_direction, out vec3 transmittance);

vec3 GetSkyRadianceToPoint(vec3 camera, vec3 point, float shadow_length, vec3 sun_direction, out vec3 transmittance);

vec3 GetSunAndSkyIrradiance(vec3 p, vec3 normal, vec3 sun_direction, out vec3 sky_irradiance);

#endif // ATMOSPHERE_LUT_GLSL