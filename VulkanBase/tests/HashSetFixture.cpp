#include "VulkanFixture.hpp"
#include "Cuckoo.hpp"
#include "gpu/HashSet.hpp"

#include <stdexcept>

class HashSetFixture : public VulkanFixture {
protected:
    static constexpr VkBufferUsageFlags usage{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    static constexpr uint32_t maxIterations{5};

    void SetUp() override {
        VulkanFixture::SetUp();

        set = gpu::HashSet{ device, descriptorPool, TABLE_SIZE };
        set.init();
        setupTestData();
    }

    void setupTestData(bool duplicates = false, uint32_t numDuplicates = 8) {
        keys.resize(NUM_ITEMS);

        if(duplicates){
            const auto repeatCount = NUM_ITEMS/numDuplicates;
            auto start = keys.begin();
            auto end = keys.begin();
            std::advance(end, repeatCount);

            const auto n = numDuplicates - 1;
            for(auto i = 0; i < n; ++i) {
                std::generate(start, end, rngFunc(100U, TABLE_SIZE - 1, 1 << 20));
                std::advance(start, repeatCount);
                std::advance(end, repeatCount);
            }
            std::sort(keys.begin(), keys.end());

        }else {
            std::generate(keys.begin(), keys.end(), rngFunc(100U, TABLE_SIZE - 1, 1 << 20));
        }

        auto device_values = keys_buffer.span<uint32_t>(NUM_ITEMS);

        std::memcpy(device_values.data(), keys.data(), BYTE_SIZE(keys));
    }

    void postVulkanInit() override {
        keys_buffer = device.createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, NUM_ITEMS * sizeof(uint32_t), "hash_values");
        insert_status = device.createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, NUM_ITEMS * sizeof(uint32_t), "insert_status");
        query_results = device.createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, NUM_ITEMS * sizeof(uint32_t), "query_results");
        table = device.createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, TABLE_SIZE * sizeof(uint32_t), "hash_table");
    }

    void hash_set_insert() {
        execute([&](auto commandBuffer){
            set.insert(commandBuffer, keys_buffer.region(0));
            set.getValue(commandBuffer, table);
            set.getInsertStatus(commandBuffer, insert_status);
        });
    }

    void hash_set_query(std::optional<VulkanBuffer> opt_keys = {}) {
        execute([&](auto commandBuffer){
            auto keys_to_query = opt_keys.has_value() ? opt_keys->region(0) : keys_buffer.region(0);
            set.find(commandBuffer, keys_to_query, query_results.region(0));
        });
    }

    void remove_items(VulkanBuffer buffer) {
        execute([&](auto commandBuffer){
            set.remove(commandBuffer, buffer.region(0));
        });
    }

    std::vector<uint32_t> keys;
    VulkanBuffer keys_buffer;
    VulkanBuffer table;
    VulkanBuffer insert_status;
    VulkanBuffer query_results;
    gpu::HashSet set;

};

TEST_F(HashSetFixture, hashSetInsert) {
    hash_set_insert();

    auto is = insert_status.span<uint32_t>(NUM_ITEMS);

    auto failedInserts = std::count_if(is.begin(), is.end(), [](auto status){ return status != 1; });
    ASSERT_TRUE(failedInserts == 0) << "eviction chain was too long";

    auto tableKeys = table.span<uint32_t>(TABLE_SIZE);

    for(auto i = 0; i < NUM_ITEMS; ++i) {
        auto key = keys[i];

        auto loc1 = hash1(key);
        auto loc2 = hash2(key);
        auto loc3 = hash3(key);
        auto loc4 = hash4(key);

        bool found = key == tableKeys[loc1];
        found |= key == tableKeys[loc2];
        found |= key == tableKeys[loc3];
        found |= key == tableKeys[loc4];

        ASSERT_TRUE(found) << fmt::format("{} status: {}, not found in hash set", key, is[i]);
    }
}

auto count_unique(std::span<uint> sp) {
    std::vector<uint> v{sp.begin(), sp.end()};
    std::sort(v.begin(), v.end());
    auto last = std::unique(v.begin(), v.end());
    v.erase(last, v.end());

    return v.size();
}

TEST_F(HashSetFixture, duplicateInserts) {
    setupTestData(true);

    hash_set_insert();

    auto tableKeys = table.span<uint32_t>(TABLE_SIZE);
    auto unique_entries = count_unique(tableKeys) - 1; // EMPTY_KEY is also counted, so we - 1 to acount for it

    auto expected_unique_entries = count_unique(keys);

    ASSERT_EQ(expected_unique_entries, unique_entries) << "duplicate entries found in set";
}

TEST_F(HashSetFixture, find_value) {
    hash_set_insert();
    hash_set_query();

    auto q_values = query_results.span<uint32_t>(NUM_ITEMS);

    for(auto i = 0; i < NUM_ITEMS; ++i) {
        ASSERT_EQ(q_values[i], keys[i]) << fmt::format("result was not equal to expected: {} != {}", q_values[i], keys[i]);
    }
}

TEST_F(HashSetFixture, remove_value_from_set) {
    hash_set_insert();

    std::vector<int> indexes(keys.size());
    std::iota(indexes.begin(), indexes.end(), 0);
    std::shuffle(indexes.begin(), indexes.end(), std::default_random_engine{1 << 20});

    const auto N = NUM_ITEMS/4;
    std::vector<uint32_t> items{};
    items.reserve(N);
    for(auto i = 0; i < N; ++i) items.push_back(keys[indexes[i]]);

    VulkanBuffer buffer = device.createCpuVisibleBuffer(items.data(), BYTE_SIZE(items), usage);
    remove_items(buffer);

    hash_set_query(buffer);
    auto q_values = query_results.span<uint32_t>(N);
    for(auto i = 0; i < N; ++i) {
        ASSERT_EQ(q_values[i], NOT_FOUND) << fmt::format("item: {} was not removed from set as expected", q_values[i]);
    }
}