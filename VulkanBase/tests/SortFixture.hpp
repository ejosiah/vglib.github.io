#pragma

#include "Sort.hpp"
#include "VulkanFixture.hpp"
#include <initializer_list>
#include <span>
#include <algorithm>

class SortFixture : public VulkanFixture{
protected:

    template<typename T = uint32_t>
    std::vector<T> randomEntries(int numElements, T start = 0, T limit = std::numeric_limits<T>::max()) const {
        auto rng = rngFunc<T>(start, limit - 1, 1 << 20);
        std::vector<T> hostBuffer(numElements);
        std::generate(begin(hostBuffer), end(hostBuffer), rng);

        return hostBuffer;
    }

    template<typename T = uint32_t>
    VulkanBuffer entries(std::vector<T> data) {
        return entries(std::span{ data.data(), data.size()});
    }

    template<typename T = uint32_t>
    VulkanBuffer entries(std::span<T> span) {
        return device.createCpuVisibleBuffer(span.data(), BYTE_SIZE(span), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }

    template<typename T = uint32_t>
    bool isSorted(VulkanBuffer& buffer) const {
        auto data = std::span{ reinterpret_cast<T*>(buffer.map()), buffer.sizeAs<T>() };
        auto sorted = std::is_sorted(data.begin(), data.end());
        buffer.unmap();
        return sorted;
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

    template<typename T = uint32_t>
    bool matches(VulkanBuffer buffer, std::span<T> expected) {
        auto actual = std::span{ reinterpret_cast<T*>(buffer.map()), buffer.sizeAs<T>() };
        auto [a, b] = std::mismatch(expected.begin(), expected.end(), actual.begin());
        auto  result = a == expected.end() && b == actual.end();
        buffer.unmap();
        return result;
    }

    bool isStable(VulkanBuffer& buffer, VulkanBuffer& indexBuffer) const {
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