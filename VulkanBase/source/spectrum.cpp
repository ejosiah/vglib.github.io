#include "spectrum/spectrum.hpp"
#include <algorithm>
#include <glm/glm.hpp>

namespace spectrum {

    void blackbody(const float *lambda, int n, float T, float *Le) {
        if (T <= 0) {
            for (int i = 0; i < n; ++i) Le[i] = 0.f;
            return;
        }
        const float c = 299792458.f;
        const float h = 6.62606957e-34f;
        const float kb = 1.3806488e-23f;
        for (int i = 0; i < n; ++i) {
            // Compute emitted radiance for blackbody at wavelength _lambda[i]_
            float l = lambda[i] * 1e-9f;
            float lambda5 = (l * l) * (l * l) * l;
            Le[i] = (2 * h * c * c) /
                    (lambda5 * (std::exp((h * c) / (l * kb * T)) - 1));
            ASSERT(!std::isnan(Le[i]));
        }
    }

    void blackbodyNormalized(const float *lambda, int n, float T, float *Le) {
        blackbody(lambda, n, T, Le);
        // Normalize _Le_ values based on maximum blackbody radiance
        float lambdaMax = 2.8977721e-3f / T * 1e9f;
        float maxL;
        blackbody(&lambdaMax, 1, T, &maxL);
        for (int i = 0; i < n; ++i) Le[i] /= maxL;
    }


    std::vector<glm::vec3> blackbodySpectrum(std::vector<float> values) {
        auto nValues = values.size();
        ASSERT(nValues % 2 == 0); // temperature (K), scale, ...
        nValues /= 2;
        std::vector<glm::vec3> s(nValues);
        std::vector<float> v(CIE_SAMPLES);
        std::vector<float> lambdas{CIE_Lambda.begin(), CIE_Lambda.end()};
        for (int i = 0; i < nValues; i++) {
            blackbodyNormalized(lambdas.data(), CIE_SAMPLES, values[2 * i], v.data());
            s[i] = values[2 * i + 1] * rgbFromSampled(lambdas, v);
        }
        return s;
    }
}
