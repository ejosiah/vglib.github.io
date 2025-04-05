#pragma once

#include "OrderChecker.hpp"
#include "VulkanFixture.hpp"

struct Data {
    uint8_t a;
    uint8_t b;
    uint8_t c;
    uint8_t d;
};

class OrderCheckerTest : public VulkanFixture {

protected:
    void postVulkanInit() override {
        _isSorted = OrderChecker{&device };
        _isSorted.init();
    }

    bool isSorted(std::span<uint32_t> items, uint32_t numBlocks, uint32_t block) {
        auto buffer = entries(items);
        auto resultBuffer = createBuffer(1);
        execute([&](auto commandBuffer){
            _isSorted(commandBuffer, { &buffer, 0, buffer.size}, { &resultBuffer, 0, resultBuffer.size}, numBlocks, block);
        });
        auto span = resultBuffer.span<uint32_t>();
        bool result = span.front() == 0;
        resultBuffer.unmap();
        return result;
    }

protected:
    OrderChecker _isSorted;
};

TEST_F(OrderCheckerTest, all32BitsAreSorted) {
    auto items = randomEntries(1 << 20);
    std::sort(items.begin(), items.end());

    EXPECT_TRUE(isSorted(items, 1, 0)) << "items should be reported as sorted";

}

TEST_F(OrderCheckerTest, all32BitsAreNotSorted) {
    auto items = randomEntries(1 << 20, 1u);

    EXPECT_FALSE(isSorted(items, 1, 0)) << "items should be reported as not sorted";
}

TEST_F(OrderCheckerTest, checkSmallDataSetIsSorted) {
    auto items = randomEntries(8192);
    std::sort(items.begin(), items.end());

    EXPECT_TRUE(isSorted(items, 1, 0)) << "items should be reported as sorted";
}


TEST_F(OrderCheckerTest, checkLargeDataSetIsSorted) {
    auto items = randomEntries((1 << 20) * 15, 1u);
    std::sort(items.begin(), items.end());

    EXPECT_TRUE(isSorted(items, 1, 0)) << "items should be reported as sorted";
}