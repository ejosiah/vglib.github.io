#include "atmosphere/Atmosphere.hpp"

#include <fstream>

std::istream &Atmosphere::operator>>(std::istream &in, Atmosphere::Format &format) {
    auto& header = format.header;
    in
        >> header.scatteringDimensions
        >> header.transmittanceDimensions
        >> header.irradianceDimensions
        >> header.solarIrradiance
        >> header.rayleighScattering
        >> header.mieScattering
        >> header.mieExtinction
        >> header.absorptionExtinction
        >> header.groundAlbedo
        >> header.sunAngularRadius
        >> header.bottomRadius
        >> header.topRadius
        >> header.mu_s_min
        >> header.mieAnisotropicFactor
        >> header.lengthUnitInMeters;

    in.read(format.data.data(), TRANSMISSION_DATA_SIZE);

    return in;
}

std::ostream &Atmosphere::operator<<(std::ostream &out, const Atmosphere::Format &format) {
    auto& header = format.header;
    out
        << header.scatteringDimensions
        << header.transmittanceDimensions
        << header.irradianceDimensions
        << header.solarIrradiance
        << header.rayleighScattering
        << header.mieScattering
        << header.mieExtinction
        << header.absorptionExtinction
        << header.groundAlbedo
        << header.sunAngularRadius
        << header.bottomRadius
        << header.topRadius
        << header.mu_s_min
        << header.mieAnisotropicFactor
        << header.lengthUnitInMeters;

    out.write(format.data.data(), DATA_SIZE);

    out.flush();

    return out;
}


Atmosphere::Format Atmosphere::load(const std::filesystem::path &path) {
    std::ifstream fin{path.string(), std::ios::binary};
    if(!fin.good()) throw std::runtime_error{std::format("unable to open {}", path.string())};

    Format format{};
    fin.read(reinterpret_cast<char*>(&format.header), sizeof(format.header));

    format.data.resize(DATA_SIZE);

    fin.read(format.data.data(), DATA_SIZE);

    return format;
}

void Atmosphere::save(const std::filesystem::path &path, const Atmosphere::Format &format) {
    std::ofstream fout{path.string(), std::ios::binary};
    if(!fout.good()) throw std::runtime_error{std::format("unable to open {}", path.string())};
    fout.write(reinterpret_cast<const char*>(&format.header), sizeof(format.header));
    fout.write(format.data.data(), DATA_SIZE);
    fout.flush();
}
