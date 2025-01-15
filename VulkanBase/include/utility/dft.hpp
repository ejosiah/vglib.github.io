#pragma once

#include <cmath>
#include <vector>
#include <complex>
#include <tuple>
#include <algorithm>
#include <fmt/format.h>
#include <ranges>

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif // PI

template<typename T>
struct fmt::formatter<std::complex<T>> {
    char presentation = 'f';    // TODO extract and send to fmt;

    // Parses format specifications of the form ['f' | 'e'].
    constexpr auto parse(format_parse_context& ctx) {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && (*it == 'f' || *it == 'e')) presentation = *it++;

        if (it != end && *it != '}')
            throw format_error("invalid format");

        return it;
    }

    template <typename FormatContext>
    auto format(const std::complex<T>& c, FormatContext& ctx) {
        return format_to(
                ctx.out(),
//                presentation == 'f' ? "{:.3f} + {:.3f}i" : "{:.3e} + {:.3e}i",
                presentation == 'f' ? "({:.3f}, {:.3f})" : "({:.3e}, {:.3e})",
                std::real(c), std::imag(c));

    }
};

int bitReverse(int x, int N) {
    int log2n = static_cast<int>(std::log2(N));
    int n = 0;
    for (int i=0; i < log2n; i++) {
        n <<= 1;
        n |= (x & 1);
        x >>= 1;
    }
    return n;
}

std::complex<double> ComputeWeight(int N, int k, bool inverse = false){
    std::complex<double> J{0, 1};
    auto kN = static_cast<double>(k)/static_cast<double>(N);
    double sign = inverse ? 1 : -1;
    return std::exp(sign * J * 2.0 * PI * kN);
}


void createButterflyLookups(std::vector<int>& lookupIndex, std::vector<std::complex<double>>& bufferflyLookup, int NButterflies, bool inverse = false){
    const int N = 1 << NButterflies;
    auto log2N = NButterflies;
    auto *ptr0 = lookupIndex.data();
    auto *ptr1 = bufferflyLookup.data();

    int i, j, k, i1, i2, j1, j2;
    int nBlocks, nHInputs;
    std::complex<double> w;
    int *qtr0;
    std::complex<double> *qtr1;

    for (i = 0; i < NButterflies; i++) {
        nBlocks  = powf(2.0, (float)(NButterflies - 1 - i));
        nHInputs = powf(2.0, (float)(i));
        qtr0 = ptr0;
        qtr1 = ptr1;
        for (j = 0; j < nBlocks; j++) {

            for (k = 0; k < nHInputs; k++) {

                if (i == 0) {
                    i1 = j*nHInputs*2 + k;
                    i2 = j*nHInputs*2 + nHInputs + k;
                    j1 = bitReverse(i1, N);
                    j2 = bitReverse(i2, N);
                }
                else {
                    i1 = j*nHInputs*2 + k;
                    i2 = j*nHInputs*2 + nHInputs + k;
                    j1 = i1;
                    j2 = i2;
                }

                w = ComputeWeight(N, k*nBlocks, inverse);

                *(qtr0 + 2*i1)   = j1;
                *(qtr0 + 2*i1+1) = j2;
                *(qtr1 + i1) = w;

                *(qtr0 + 2*i2)   = j1;
                *(qtr0 + 2*i2+1) = j2;
                *(qtr1 + i2) = -w;

            }
        }
        ptr0 += 2*N;
        ptr1 += N;
    }
}


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

inline std::vector<std::complex<double>> fft(const std::vector<std::complex<double>>& p){
    auto n = p.size();
    if(n <= 1) return p;

    auto even = evenIndices(p);
    auto odd = oddIndices(p);

    auto yEven = fft(even);
    auto yOdd = fft(odd);
    static std::complex<double> J{0, 1};
    auto dw = std::exp(-J * 2.0 * PI/static_cast<double>(n));
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
    std::vector<float> realPart;
    std::vector<float> imaginaryPart;

    for(const auto& c : result){
//        spdlog::info("{:2f} + {:2f}i", std::real(c), std::imag(c));
    }

    auto realView = result | std::views::transform([](const auto cp){ return std::real(cp); });
    auto imgView = result | std::views::transform([](const auto cp){ return std::imag(cp); });

    std::ranges::copy(realView, std::back_inserter(realPart));
    std::ranges::copy(imgView, std::back_inserter(imaginaryPart));


    return std::make_tuple(realPart, imaginaryPart);
}