#pragma once

#include "VulkanDevice.h"
#include "VulkanBuffer.h"

#include <vector>
#include <algorithm>

namespace TestUtils {


    template<typename T = uint32_t>
    std::vector<T> randomEntries(int numElements, T start = 0, T limit = std::numeric_limits<T>::max())  {
        auto rng = rngFunc<T>(start, limit - 1, 1 << 20);
        std::vector<T> hostBuffer(numElements);
        std::generate(begin(hostBuffer), end(hostBuffer), rng);

        return hostBuffer;
    }

    template<typename T = uint32_t>
    bool isSorted(VulkanBuffer& buffer) {
        auto data = std::span{ reinterpret_cast<T*>(buffer.map()), buffer.sizeAs<T>() };
        auto sorted = std::is_sorted(data.begin(), data.end());
        if(!sorted) {
            for(int i = data.size() - 1; i > 0; --i){
                if(data[i] < data[i - 1]){
                    auto a = data[i];
                    auto b = data[i - 1];
                    auto c = 0;
                }
            }
        }
        buffer.unmap();
        return sorted;
    }

    template<typename T = uint32_t>
    bool matches(VulkanBuffer buffer, std::span<T> expected) {
        auto actual = std::span{ reinterpret_cast<T*>(buffer.map()), buffer.sizeAs<T>() };
        auto [a, b] = std::mismatch(expected.begin(), expected.end(), actual.begin());
        auto aId = std::distance(expected.begin(), a);
        auto bId = std::distance(actual.begin(), b);
        auto  result = a == expected.end() && b == actual.end();
        buffer.unmap();
        return result;
    }

    template<typename T = uint32_t>
    bool sortedMatch(VulkanBuffer buffer, std::vector<T> expected) {
        std::sort(expected.begin(), expected.end());
        return matches<T>(buffer, std::span{ expected.data(), expected.size() });
    }


    template<typename T = uint32_t>
    bool matches(VulkanBuffer buffer, std::vector<T> expected) {
        return matches<T>(buffer, std::span{ expected.data(), expected.size() });
    }


    inline bool isStable(VulkanBuffer& buffer, VulkanBuffer& indexBuffer)  {
        bool result = true;
        int numElements = buffer.size/sizeof(uint32_t);

        buffer.map<uint32_t>([&](auto valuePtr){
            indexBuffer.map<uint32_t>([&](auto indexPtr){
                for(int i = 0; i < numElements - 1; i++){
                    auto a = *(valuePtr + i);
                    auto b = *(valuePtr + i + 1);
                    if(a == b){
                        auto aIndex = *(indexPtr + i);
                        auto bIndex = *(indexPtr + i + 1);
                        result = aIndex < bIndex;
                        if(!result) break;
                    }
                }
            });
        });
        return result;
    }
};