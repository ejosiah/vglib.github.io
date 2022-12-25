#pragma once

#include <cmath>
#include <vector>
#include <complex>
#include <tuple>
#include <algorithm>

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif // PI

template<typename T>
inline std::vector<T> evenIndices(const std::vector<T>& v){
    auto n = v.size();
    std::vector<T> result;
    result.reserve(n/2);

    for(int i = 0; i < n; i +=2){
        result.push_back(v[i]);
    }
    return result;
}

template<typename T>
inline std::vector<T> oddIndices(const std::vector<T>& v){
    auto n = v.size();
    std::vector<T> result;
    result.reserve(n/2);

    for(int i = 1; i < n; i += 2){
        result.push_back(v[i]);
    }

    return result;
}

inline glm::vec2 rootsOfUnity(float n){
    auto f = glm::two_pi<float>() * n;
    return {glm::cos(f), glm::sin(f)};
}

inline std::vector<std::complex<double>> fft(const std::vector<std::complex<double>>& p){
    auto n = p.size();
    if(n <= 1) return p;

    auto even = evenIndices(p);
    auto odd = oddIndices(p);

    auto yEven = fft(even);
    auto yOdd = fft(odd);
    static std::complex<double> J{0, 1};
    auto dw = std::exp(J * 2.0 * PI/static_cast<double>(n));
    std::complex<double> w{1, 0};

    std::vector<std::complex<double>> y(n);
    for(int k = 0; k < n/2; k++){
        y[k] = yEven[k] + w * yOdd[k];
        y[k + n/2] = yEven[k] - w * yOdd[k];
        w *= dw;
    }

    return y;
}

inline std::tuple<std::vector<float>, std::vector<float>> fft(const std::vector<float>& p){
    auto n = p.size();
    if(n <= 1) return std::make_tuple(p, std::vector<float>(n));

    std::vector<std::complex<double>> cp;
    cp.reserve(n);

    for(auto real : p){
        cp.push_back(std::complex<float>{real, 0});
    }

    auto result = fft(cp);
    auto realPart = std::vector<float>(n);
    auto imaginaryPart = std::vector<float>(n);

    for(const auto& c : result){
//        spdlog::info("{:2f} + {:2f}i", std::real(c), std::imag(c));
    }

    std::transform(begin(result), end(result), begin(realPart), [](const auto cp){ return std::real(cp); });
    std::transform(begin(result), end(result), begin(imaginaryPart), [](const auto cp){ return std::imag(cp); });

    return std::make_tuple(realPart, imaginaryPart);
}