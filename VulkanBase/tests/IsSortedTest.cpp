#pragma once

#include "IsSorted.hpp"
#include "VulkanFixture.hpp"

struct Data {
    uint8_t a;
    uint8_t b;
    uint8_t c;
    uint8_t d;
};

class IsSortedTest : public VulkanFixture {

protected:
    void postVulkanInit() override {
        _isSorted = IsSorted{ &device };
        _isSorted.init();
    }

    bool isSorted(std::span<uint32_t> items, uint32_t numBlocks, uint32_t block) {
        auto buffer = entries(items);
        auto resultBuffer = createBuffer(1);
        execute([&](auto commandBuffer){
            _isSorted(commandBuffer, { &buffer, 0, buffer.size}, { &resultBuffer, 0, resultBuffer.size}, numBlocks, block);
        });

        auto span = resultBuffer.span<uint32_t>();
//        auto bitset = _isSorted._internal.bitSet.span<uint32_t>(items.size());
//        for(int i = 0; i < items.size(); i++){
//            if(bitset[i] != 0) {
//                if(i > 0) {
//                    fmt::print("bit set is {} at {}, {} < {}\n", bitset[i], i, items[i - 1], items[i]);
//                }else{
//                    fmt::print("bit set is {} at {}, {}\n", bitset[i], i, items[i]);
//                }
//            }
//        }
        bool result = span.front() == 0;
        resultBuffer.unmap();
        return result;
    }

protected:
    IsSorted _isSorted;
};

TEST_F(IsSortedTest, all32BitsAreSorted) {
    auto items = randomEntries(1 << 20);
    std::sort(items.begin(), items.end());

    EXPECT_TRUE(isSorted(items, 1, 0)) << "items should be reported as sorted";

}

TEST_F(IsSortedTest, all32BitsAreNotSorted) {
    auto items = randomEntries(1 << 20, 1u);

    EXPECT_FALSE(isSorted(items, 1, 0)) << "items should be reported as not sorted";
}

TEST_F(IsSortedTest, checkIndividualBlocksAreSortedOf4Blocks) {
    for(int block = 0; block < 4; ++block){
        auto items = randomEntries(1 << 20, 1u);
        std::span<Data> raw = { reinterpret_cast<Data*>(items.data()), items.size() };
        std::sort(items.begin(), items.end(), [block, raw](const auto& a, const auto& b){
            auto aBlock = ((a >> (block * 8)) & 0xFF);
            auto bBlock = ((b >> (block * 8)) & 0xFF);
           return  aBlock < bBlock;
        });


        EXPECT_TRUE(isSorted(items, 4, block)) << std::format("block {} should be reported as sorted", block).c_str();
    }
}

TEST_F(IsSortedTest, checkIndividualBlocksAreNotSortedOf4Blocks) {
    auto items = randomEntries(1 << 20, 1u);
    for(int block = 0; block < 4; ++block){
        EXPECT_FALSE(isSorted(items, 4, block)) << std::format("block {} should not be reported as sorted", block).c_str();
    }
}

TEST_F(IsSortedTest, checkIndividualBlocksAreSortedOf16Blocks) {
    for(int block = 0; block < 16; ++block ) {
        auto items = randomEntries(1 << 20, 1u);
        std::span<Data> raw = { reinterpret_cast<Data*>(items.data()), items.size() };
        std::sort(items.begin(), items.end(), [block, raw](const auto& a, const auto& b){
            auto aBlock = ((a >> (block * 2)) & 0x03);
            auto bBlock = ((b >> (block * 2)) & 0x03);
            return  aBlock < bBlock;
        });


        EXPECT_TRUE(isSorted(items, 16, block)) << std::format("block {} should be reported as sorted", block).c_str();
    }
}

TEST_F(IsSortedTest, checkIndividualBlocksAreNotSortedOf16Blocks) {
    auto items = randomEntries(1 << 20, 1u);
    for(int block = 0; block < 16; ++block){
        EXPECT_FALSE(isSorted(items, 16, block)) << std::format("block {} should not be reported as sorted", block).c_str();
    }
}

TEST_F(IsSortedTest, checkLargeDataSetIsSorted) {
    auto items = randomEntries((1 << 20) * 15);
    std::sort(items.begin(), items.end());

    EXPECT_TRUE(isSorted(items, 1, 0)) << "items should be reported as sorted";
}