#include "SortFixture.hpp"
#include "OrderChecker.hpp"

struct StudentRecord {
    uint id;
    char sCode[4];
    float score;

    bool operator==(const StudentRecord& b) const {
        const auto& a = *this;
        return a.id == b.id && a.score == b.score &&
               a.sCode[0] == b.sCode[0] && a.sCode[1] == b.sCode[1]
               && a.sCode[2] == b.sCode[2] && a.sCode[3] == b.sCode[3];
    }
};

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

    void sort(VulkanBuffer keys, Records records) {
        execute([&](auto commandBuffer){
           _sort(commandBuffer, keys, records);
        });
    }

    void sortWithIndex(VulkanBuffer& buffer, VulkanBuffer indexBuffer){
        execute([&](auto commandBuffer){
             _sort.sortWithIndices(commandBuffer, buffer, indexBuffer);
        });
    }

    std::tuple<VulkanBuffer, Records, std::vector<StudentRecord>> randomRecords(int numRecords, int keyOffset = 0) {
        static auto randomCode = [] () -> std::vector<char> {
            static auto rngAlpha = rngFunc(65, 90, 1 << 20);

            return { static_cast<char>(rngAlpha()), static_cast<char>(rngAlpha()),
                     static_cast<char>(rngAlpha()), static_cast<char>(rngAlpha())};
        };
        auto randomScore = rngFunc(4.9f, 99.f, 1 << 20);

        std::vector<StudentRecord> records(numRecords);
        int nextID = 0;
        std::generate(records.begin(), records.end(), [&]{
            StudentRecord record{};
            record.id = nextID++;
            record.score = randomScore();

            auto sCode = randomCode();
            record.sCode[0] = sCode[0];
            record.sCode[1] = sCode[1];
            record.sCode[2] = sCode[2];
            record.sCode[3] = sCode[3];
            return record;
        });
        std::shuffle(records.begin(), records.end(), std::default_random_engine{1 << 20});

        std::vector<uint> keys;
        for(auto record : records) {
            keys.push_back(*(reinterpret_cast<uint*>(&record) +  keyOffset));
        }

//        for(auto i = 0; i < keys.size(); i++) {
//            auto check = *reinterpret_cast<float *>(&keys[i]);
//            assert(check == records[i].score);
//        }

        VulkanBuffer keyBuffer = device.createCpuVisibleBuffer(keys.data(), BYTE_SIZE(keys), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        VulkanBuffer buffer = device.createCpuVisibleBuffer(records.data(), BYTE_SIZE(records), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        return std::make_tuple(keyBuffer, Records{buffer, sizeof(StudentRecord) }, records);
    }
};

TEST_F(RadixSortFixture, sortGivenData){
    auto items = randomEntries(1 << 20);
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

TEST_F(RadixSortFixture, sortFloatingPointNumbersCountsNotPowerOf2){
    auto items = randomEntries<float>((1 << 14) - 1, -glm::pi<float>(), glm::pi<float>());
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
    std::vector<uint32_t> items = randomEntries(20000);
    VulkanBuffer buffer = device.createCpuVisibleBuffer(items.data(), BYTE_SIZE(items), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    std::vector<std::pair<uint32_t, uint32_t>> itemsWithIndexes(items.size());
    for(auto i = 0; i < items.size(); ++i){
        itemsWithIndexes[i] = std::make_pair(items[i], i);
    }
    std::stable_sort(itemsWithIndexes.begin(), itemsWithIndexes.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    std::vector<uint32_t> expectedIndices{};
    for(auto [_, index] : itemsWithIndexes) {
        expectedIndices.push_back(index);
    }

    VulkanBuffer indexBuffer = device.createStagingBuffer(BYTE_SIZE(items));

    sortWithIndex(buffer, indexBuffer);

    ASSERT_TRUE(matches(indexBuffer, expectedIndices)) << "indices are not the same";
}

TEST_F(RadixSortFixture, clearIndicesBeforeSorting) {
    auto buffer = entries({5, 1, 8, 11, 15, 20, 10, 6, 9, 7, 3, 4, 2, 13, 16, 14, 17, 19, 18, 12});
    auto buffer1 = entries({5, 1, 8, 11, 15, 20, 10, 6, 9, 7, 3, 4, 2, 13, 16, 14, 17, 19, 18, 12});
    std::vector<uint32_t> expectedIndices{ 1, 12, 10, 11, 0, 7, 9, 2, 8, 6, 3, 19, 13, 15, 4, 14, 16, 18, 17, 5 };

    VulkanBuffer indexBuffer = device.createStagingBuffer(BYTE_SIZE(expectedIndices));

    sortWithIndex(buffer, indexBuffer);
    sortWithIndex(buffer1, indexBuffer);

    ASSERT_TRUE(matches(indexBuffer, expectedIndices)) << "indices are not the same";
}

TEST_F(RadixSortFixture, sortHostData){
    auto items = randomEntries(1 << 14);
    ASSERT_FALSE(std::is_sorted(begin(items), end(items)));

    _sort.sort(begin(items), end(items));

    ASSERT_TRUE(std::is_sorted(begin(items), end(items))) << "items should be sorted";
}

TEST_F(RadixSortFixture, sortRecordsById) {
    auto [keys, records, expected] = randomRecords(1000);
    sort(keys, records);

    std::stable_sort(expected.begin(), expected.end(), [](const auto& a, const auto& b){ return a.id < b.id; });
    ASSERT_TRUE(isSorted(keys));
    ASSERT_TRUE(matches<StudentRecord>(records.buffer, expected));
}

TEST_F(RadixSortFixture, sortRecordByCode) {
    auto [keys, records, expected] = randomRecords(1000, 1);
    sort(keys, records);

    std::stable_sort(expected.begin(), expected.end(), [](const auto& a, const auto& b){
        return *reinterpret_cast<const uint*>(&a.sCode[0]) < *reinterpret_cast<const uint*>(&b.sCode[0]);
    });
    ASSERT_TRUE(isSorted(keys));
    ASSERT_TRUE(matches<StudentRecord>(records.buffer, expected));
}

TEST_F(RadixSortFixture, sortRecordByScore) {
    auto [keys, records, expected] = randomRecords(1 << 14, 2);
    records.keyType = KeyType::Float;
    sort(keys, records);

    std::stable_sort(expected.begin(), expected.end(), [](const auto& a, const auto& b){
        return a.score < b.score;
    });
    ASSERT_TRUE(isSorted<float>(keys));
    ASSERT_TRUE(matches<StudentRecord>(records.buffer, expected));
}

TEST_F(RadixSortFixture, sortIsStable){
    auto items = randomEntries(1 << 20);
    VulkanBuffer buffer = device.createCpuVisibleBuffer(items.data(), BYTE_SIZE(items), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ASSERT_TRUE(!isSorted(buffer)) << "buffer initial state should not be sorted";

    VulkanBuffer indexBuffer = device.createStagingBuffer(buffer.size);

    sortWithIndex(buffer, indexBuffer);

    ASSERT_TRUE(isSorted(buffer)) << "buffer should be sorted";
    ASSERT_TRUE(isStable(buffer, indexBuffer)) << "sort should be stable";
}

TEST_F(RadixSortFixture, dataNotSorted50000ItemsBug){
    auto items = randomEntries(500000, 1u);
    VulkanBuffer buffer = device.createCpuVisibleBuffer(items.data(), BYTE_SIZE(items), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ASSERT_FALSE(isSorted(buffer)) << "buffer initial state should not be sorted";

    OrderChecker _isSorted{&device };
    _isSorted.init();

    sort(buffer);

//    auto block = 0u;
//    std::stable_sort(items.begin(), items.end(), [block](const auto& a, const auto& b){
//        auto aBlock = ((a >> (block * 8)) & 0xFF);
//        auto bBlock = ((b >> (block * 8)) & 0xFF);
//        return  aBlock < bBlock;
//    });
//
//    auto counts = _sort.countsBuffer.span<uint>();
//
//    ASSERT_TRUE(matches(buffer, items));

//    static constexpr VkBufferUsageFlags usage =
//            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
//            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
//            | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
//
//    VulkanBuffer result = device.createBuffer(usage, VMA_MEMORY_USAGE_GPU_TO_CPU, sizeof(uint32_t));
//    auto block = 0u;
//    execute([&](auto commandBuffer){
//        _isSorted(commandBuffer, {&buffer, 0, buffer.size} , {&result, 0, result.size}, 4, block);
//    });
//    auto sortStatus = *reinterpret_cast<uint32_t*>(result.map());
//    result.unmap();
//    EXPECT_EQ(sortStatus, 0) << std::format("block {} was not sorted", block).c_str();

    ASSERT_TRUE(sortedMatch(buffer, items)) << "buffer should be sorted";
}

TEST_F(RadixSortFixture, only255ItemsAfterSortingBug){
    auto items = randomEntries(80000);
    VulkanBuffer buffer = device.createCpuVisibleBuffer(items.data(), BYTE_SIZE(items), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ASSERT_FALSE(isSorted(buffer)) << "buffer initial state should not be sorted";

    sort(buffer);

    ASSERT_TRUE(sortedMatch(buffer, items)) << "buffer should be sorted";
}

TEST_F(RadixSortFixture, sortWithOrderCheckingEnabled){
    _sort.enableOrderChecking();
    auto items = randomEntries(1 << 20);
    VulkanBuffer buffer = device.createCpuVisibleBuffer(items.data(), BYTE_SIZE(items), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ASSERT_FALSE(isSorted(buffer)) << "buffer initial state should not be sorted";

    sort(buffer);

    ASSERT_TRUE(sortedMatch(buffer, items)) << "buffer should be sorted";
}