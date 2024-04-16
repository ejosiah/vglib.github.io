#include "SortFixture.hpp"

class RadixSortFixture : public SortFixture{
protected:
   RadixSort _sort;

    void postVulkanInit() override {
        _sort = RadixSort(&device, true);
        _sort.init();
    }

    void sort(VulkanBuffer& buffer){
        execute([&](auto commandBuffer){
           _sort(commandBuffer, buffer);
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
    auto items = randomInts(1 << 14);
    VulkanBuffer buffer = device.createCpuVisibleBuffer(items.data(), BYTE_SIZE(items), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ASSERT_FALSE(isSorted(buffer)) << "buffer initial state should not be sorted";

    sort(buffer);

    ASSERT_TRUE(isSorted(buffer)) << "buffer should be sorted";
}

TEST_F(RadixSortFixture, sortWithIndices) {
    auto buffer = entries({5, 1, 8, 11, 15, 20, 10, 6, 9, 7, 3, 4, 2, 13, 16, 14, 17, 19, 18, 12});
    std::vector<int> expectedIndices{ 1, 12, 10, 11, 0, 7, 9, 2, 8, 6, 3, 19, 13, 15, 4, 14, 16, 18, 17, 5 };

    VulkanBuffer indexBuffer = sortWithIndex(buffer);
    std::vector<int> actualIndices(20);
    auto source = indexBuffer.map();
    std::memcpy(actualIndices.data(), source, indexBuffer.size);
    indexBuffer.unmap();

    for(auto i = 0; i < expectedIndices.size(); ++i){
        EXPECT_EQ(expectedIndices[i], actualIndices[i]) << "indices are not the same";
    }
}

TEST_F(RadixSortFixture, sortHostData){
    auto items = randomInts(1 << 14);
    ASSERT_FALSE(std::is_sorted(begin(items), end(items)));

    _sort.sort(begin(items), end(items));

    ASSERT_TRUE(std::is_sorted(begin(items), end(items))) << "items should be sorted";
}

TEST_F(RadixSortFixture, sortIsStable){
    auto items = randomInts(1 << 20);
    VulkanBuffer buffer = device.createCpuVisibleBuffer(items.data(), BYTE_SIZE(items), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ASSERT_TRUE(!isSorted(buffer)) << "buffer initial state should not be sorted";

    VulkanBuffer indexBuffer = sortWithIndex(buffer);

    ASSERT_TRUE(isSorted(buffer)) << "buffer should be sorted";
    ASSERT_TRUE(isStable(buffer, indexBuffer)) << "sort should be stable";
}