#include "VulkanFixture.hpp"
#include <stdexcept>
#include "Cuckoo.hpp"
#include "gpu/HashMap.hpp"

class HashTableFixture : public VulkanFixture {
protected:

    static constexpr VkBufferUsageFlags usage{  VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT};

    void SetUp() override {
        VulkanFixture::SetUp();

        hashMap = gpu::HashMap{device, descriptorPool, TABLE_SIZE};
        hashMap.init();
        setupTestData();
    }

    void postVulkanInit() override {
        keys_buffer = device.createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, NUM_ITEMS * sizeof(uint32_t), "hash_keys");
        values_buffer = device.createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, NUM_ITEMS * sizeof(uint32_t), "hash_values");
        insert_status = device.createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, NUM_ITEMS * sizeof(uint32_t), "insert_status");
        query_results = device.createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, NUM_ITEMS * sizeof(uint32_t), "query_results");
        table_keys = device.createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, TABLE_SIZE * sizeof(uint32_t), "table_keys");
        table_values = device.createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, TABLE_SIZE * sizeof(uint32_t), "table_values");
    }

    void hash_table_insert(uint32_t numItems = std::numeric_limits<uint32_t>::max()) {
        execute([&](auto commandBuffer){
            VkDeviceSize end = numItems == std::numeric_limits<uint32_t>::max() ? VK_WHOLE_SIZE : numItems * sizeof(uint);
            hashMap.insert(commandBuffer, keys_buffer.region(0, end), values_buffer.region(0, end));
            hashMap.getEntries(commandBuffer, table_keys, table_values);
            hashMap.getInsertStatus(commandBuffer, insert_status);
        });
    }

    void query_hash_table(BufferRegion key_region = {}) {
        execute([&](auto commandBuffer){
            key_region = key_region.buffer ? key_region : keys_buffer.region(0);
            hashMap.find(commandBuffer, key_region, query_results.region(0));
        });
    }

    void remove_items(VulkanBuffer buffer) {
        execute([&](auto commandBuffer){
            hashMap.remove(commandBuffer, buffer.region(0));
        });
    }

    void setupTestData() {
        values.resize(NUM_ITEMS);

        auto keyGenerator = rngFunc(100U, NUM_ITEMS * 2, 1 << 20);
        auto valueGenerator = rngFunc(100u, std::numeric_limits<uint32_t>::max()/2, 1 << 19);

        std::generate(values.begin(), values.end(), valueGenerator);

        keys.reserve(NUM_ITEMS);
        std::set<uint32_t> used_keys;
        for(auto i = 0; i < NUM_ITEMS; ++i){
            uint32_t key;
            do { key = keyGenerator(); } while(used_keys.contains(key));
            keys.push_back(key);
            used_keys.insert(key);
        }
        auto device_keys = keys_buffer.span<uint32_t>(NUM_ITEMS);
        auto device_values = values_buffer.span<uint32_t>(NUM_ITEMS);

        std::memcpy(device_keys.data(), keys.data(), BYTE_SIZE(keys));
        std::memcpy(device_values.data(), values.data(), BYTE_SIZE(values));
    }

    void checkItemsInsertedIntoHashMap(uint32_t numItems = NUM_ITEMS) {
        auto tableKeys = table_keys.span<uint32_t>(TABLE_SIZE);
        auto tableValues =  table_values.span<uint32_t>(TABLE_SIZE);
        auto is = insert_status.span<uint32_t>(numItems);

        auto failedInserts = std::count_if(is.begin(), is.end(), [](auto status){ return status != 1; });
        ASSERT_TRUE(failedInserts == 0) << "eviction chain was too long";

        for(auto i = 0; i < numItems; ++i) {
            auto key = keys[i];
            auto value = values[i];

            auto loc1 = hash1(key);
            auto loc2 = hash2(key);
            auto loc3 = hash3(key);
            auto loc4 = hash4(key);

            bool found = key == tableKeys[loc1] && value == tableValues[loc1];
            found |= key == tableKeys[loc2] && value == tableValues[loc2];
            found |= key == tableKeys[loc3] && value == tableValues[loc3];
            found |= key == tableKeys[loc4] && value == tableValues[loc4];

            ASSERT_TRUE(found) << fmt::format("entry({} -> {}) status: {}, not found in hash table", key, value, is[i]);
        }
    }

    std::vector<uint32_t> keys;
    std::vector<uint32_t> values;
    VulkanBuffer keys_buffer;
    VulkanBuffer values_buffer;
    VulkanBuffer table_keys;
    VulkanBuffer table_values;
    VulkanBuffer insert_status;
    VulkanBuffer query_results;
    gpu::HashMap hashMap;
};


TEST_F(HashTableFixture, insert_items_into_hash_map) {
    hash_table_insert();
    checkItemsInsertedIntoHashMap();
}

TEST_F(HashTableFixture, non_powers_of_2_insert) {
    const auto N = NUM_ITEMS - 100;
    hash_table_insert(N);
    checkItemsInsertedIntoHashMap(N);
}

TEST_F(HashTableFixture, find_items_in_hashmap) {
    hash_table_insert();
    query_hash_table();

    auto q_values = query_results.span<uint32_t>(NUM_ITEMS);

    for(auto i = 0; i < NUM_ITEMS; ++i) {
        ASSERT_EQ(values[i], q_values[i]) << fmt::format("result was not equal to expected: {} != {}", q_values[i], values[i]);
    }
}

TEST_F(HashTableFixture, find_items_on_powers_of_2) {
    const auto N = NUM_ITEMS - 100;
    const VkDeviceSize end = N * sizeof(uint32_t);

    hash_table_insert(N);
    query_hash_table(keys_buffer.region(0, end));

    auto q_values = query_results.span<uint32_t>(N);

    for(auto i = 0; i < N; ++i) {
        ASSERT_EQ(values[i], q_values[i]) << fmt::format("result was not equal to expected: {} != {}", q_values[i], values[i]);
    }
}

TEST_F(HashTableFixture, remove_keys_from_hashmap) {
    hash_table_insert();

    std::vector<int> indexes(keys.size());
    std::iota(indexes.begin(), indexes.end(), 0);
    std::shuffle(indexes.begin(), indexes.end(), std::default_random_engine{1 << 20});

    const auto N = NUM_ITEMS/4;
    std::vector<uint32_t> items{};
    items.reserve(N);
    for(auto i = 0; i < N; ++i) items.push_back(keys[indexes[i]]);

    VulkanBuffer buffer = device.createCpuVisibleBuffer(items.data(), BYTE_SIZE(items), usage);
    remove_items(buffer);

    query_hash_table(buffer.region(0));
    auto q_values = query_results.span<uint32_t>(N);
    for(auto i = 0; i < N; ++i) {
        ASSERT_EQ(q_values[i], NOT_FOUND) << fmt::format("item: {} was not removed from set as expected", q_values[i]);
    }
}