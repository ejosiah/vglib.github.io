#pragma once
#include "common.h"

namespace sampling {

    inline glm::vec2 uniformSampleDisk(const glm::vec2& u){
        float r = std::sqrt(u[0]);
        float theta = glm::two_pi<float>() * u[1];
        return {r * std::cos(theta), r * std::sin(theta)};
    }

    struct Distribution1D {
        uint32_t size;
        float funcIntegral{0};
        std::vector<float> func, cdf;

        static Distribution1D create(const float* aFunc, uint32_t size){
            std::vector<float> cdf(size + 1);
            cdf[0] = 0;
            for(auto i = 1; i < size + 1; i++){
                cdf[i] = cdf[i - 1] + aFunc[i - 1] / float(size);
            }
            auto funcIntegral = cdf[size];
            if(funcIntegral == 0){
                for(int i = 1; i < size + 1; i++){
                    cdf[i] = float(i)/float(size);
                }
            }else{
                for(int i = 1; i < size + 1; i++){
                    cdf[i] /= funcIntegral;
                }
            }
            std::vector<float> func{aFunc, aFunc + size};
            return {size, funcIntegral, func, cdf};
        }
    };

    struct Distribution2D{
        std::vector<Distribution1D> pConditionalV;
        Distribution1D pMarginal;

        static Distribution2D create(const float* func, uint32_t nu, uint32_t nv){
            std::vector<Distribution1D> pConditionalV;
            pConditionalV.reserve(nv);
            for(auto v = 0; v < nv; v++){
                // Compute conditional sampling distribution for v
                pConditionalV.emplace_back(Distribution1D::create(&func[v * nu], nu));
            }
            // compute marginal sampling distribution for v;
            std::vector<float> marginalFunc;
            marginalFunc.reserve(nv);
            for(auto v = 0; v < nv; v++){
                marginalFunc.push_back(pConditionalV[v].funcIntegral);
            }
            auto pMarginal = Distribution1D::create(marginalFunc.data(), nv);
            return { pConditionalV, pMarginal};
        }
    };
}