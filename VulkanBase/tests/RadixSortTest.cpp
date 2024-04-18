#include "SortFixture.hpp"

class RadixSortFixture : public SortFixture{
protected:
   RadixSort _sort;

    void postVulkanInit() override {
        _sort = RadixSort(&device, true);
        _sort.init();
    }

    template<typename T = uint32_t>
    void sort(VulkanBuffer& buffer){
        execute([&](auto commandBuffer){
            if constexpr (std::is_same_v<T, uint32_t>) {
                _sort(commandBuffer, buffer);
            }else {
                _sort.sortTyped<T>(commandBuffer, buffer);
            }
        });
    }

    VulkanBuffer sortWithIndex(VulkanBuffer& buffer){
        VulkanBuffer indexBuffer;
        execute([&](auto commandBuffer){
            indexBuffer = _sort.sortWithIndices(commandBuffer, buffer);
        });
        return indexBuffer;
    }
};

TEST_F(RadixSortFixture, sortGivenData){
    auto items = randomEntries(1 << 14);
    VulkanBuffer buffer = device.createCpuVisibleBuffer(items.data(), BYTE_SIZE(items), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ASSERT_FALSE(isSorted(buffer)) << "buffer initial state should not be sorted";

    sort(buffer);

    ASSERT_TRUE(sortedMatch(buffer, items)) << "buffer should be sorted";
}

TEST_F(RadixSortFixture, sortFloatingPointNumbers){
    auto items = randomEntries<float>(1 << 14, -glm::pi<float>(), glm::pi<float>());
    VulkanBuffer buffer = device.createCpuVisibleBuffer(items.data(), BYTE_SIZE(items), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ASSERT_FALSE(isSorted<float>(buffer)) << "buffer initial state should not be sorted";

    sort<float>(buffer);

    ASSERT_TRUE(sortedMatch<float>(buffer, items)) << "buffer should be sorted";
}

TEST_F(RadixSortFixture, sortIntsWithNegativeValues) {
    auto items = randomEntries<int>(1 << 14, -(1 << 20), (1 << 20));
    VulkanBuffer buffer = device.createCpuVisibleBuffer(items.data(), BYTE_SIZE(items), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ASSERT_FALSE(isSorted<int>(buffer)) << "buffer initial state should not be sorted";

    sort<int>(buffer);

    ASSERT_TRUE(sortedMatch<int>(buffer, items)) << "buffer should be sorted";
}

TEST_F(RadixSortFixture, sortWithIndices) {
    auto buffer = entries({5, 1, 8, 11, 15, 20, 10, 6, 9, 7, 3, 4, 2, 13, 16, 14, 17, 19, 18, 12});
    std::vector<uint32_t> expectedIndices{ 1, 12, 10, 11, 0, 7, 9, 2, 8, 6, 3, 19, 13, 15, 4, 14, 16, 18, 17, 5 };

    VulkanBuffer indexBuffer = sortWithIndex(buffer);

    ASSERT_TRUE(matches(indexBuffer, expectedIndices)) << "indices are not the same";
}

TEST_F(RadixSortFixture, clearIndicesBeforeSorting) {
    auto buffer = entries({5, 1, 8, 11, 15, 20, 10, 6, 9, 7, 3, 4, 2, 13, 16, 14, 17, 19, 18, 12});
    auto buffer1 = entries({5, 1, 8, 11, 15, 20, 10, 6, 9, 7, 3, 4, 2, 13, 16, 14, 17, 19, 18, 12});
    std::vector<uint32_t> expectedIndices{ 1, 12, 10, 11, 0, 7, 9, 2, 8, 6, 3, 19, 13, 15, 4, 14, 16, 18, 17, 5 };

    sortWithIndex(buffer);
    VulkanBuffer indexBuffer = sortWithIndex(buffer1);

    ASSERT_TRUE(matches(indexBuffer, expectedIndices)) << "indices are not the same";
}

TEST_F(RadixSortFixture, sortHostData){
    auto items = randomEntries(1 << 14);
    ASSERT_FALSE(std::is_sorted(begin(items), end(items)));

    _sort.sort(begin(items), end(items));

    ASSERT_TRUE(std::is_sorted(begin(items), end(items))) << "items should be sorted";
}

TEST_F(RadixSortFixture, sortIsStable){
    auto items = randomEntries(1 << 20);
    VulkanBuffer buffer = device.createCpuVisibleBuffer(items.data(), BYTE_SIZE(items), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ASSERT_TRUE(!isSorted(buffer)) << "buffer initial state should not be sorted";

    VulkanBuffer indexBuffer = sortWithIndex(buffer);

    ASSERT_TRUE(isSorted(buffer)) << "buffer should be sorted";
    ASSERT_TRUE(isStable(buffer, indexBuffer)) << "sort should be stable";
}