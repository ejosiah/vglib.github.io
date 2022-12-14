#include "SortFixture.hpp"

class RadixSortFixture : public SortFixture{
protected:
   RadixSort _sort;

    void postVulkanInit() override {
        _sort = RadixSort(&device);
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
        return std::move(indexBuffer);
    }
};

TEST_F(RadixSortFixture, sortGivenData){
    auto items = randomInts(1 << 14);
    VulkanBuffer buffer = device.createCpuVisibleBuffer(items.data(), BYTE_SIZE(items), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ASSERT_FALSE(isSorted(buffer)) << "buffer initial state should not be sorted";

    sort(buffer);

    ASSERT_TRUE(isSorted(buffer)) << "buffer should be sorted";
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