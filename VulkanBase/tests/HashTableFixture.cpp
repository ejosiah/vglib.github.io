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

        std::vector<uint> nullEntries(TABLE_SIZE);
        std::generate(nullEntries.begin(), nullEntries.end(), []{ return 0xFFFFFFFFu; });

        table_keys = device.createCpuVisibleBuffer(nullEntries.data(), BYTE_SIZE(nullEntries), usage);
        table_values = device.createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, TABLE_SIZE * sizeof(uint32_t), "table_values");
    }

    void hash_table_insert() {
        execute([&](auto commandBuffer){
            hashMap.insert(commandBuffer, keys_buffer.region(0), values_buffer.region(0));
            hashMap.getEntries(commandBuffer, table_keys, table_values);
            hashMap.getInsertStatus(commandBuffer, insert_status);
        });
    }

    void query_hash_table() {
        execute([&](auto commandBuffer){
            hashMap.find(commandBuffer, keys_buffer.region(0), query_results.region(0));
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


TEST_F(HashTableFixture, HashTableInsert) {
    hash_table_insert();


    auto tableKeys = table_keys.span<uint32_t>(TABLE_SIZE);
    auto tableValues =  table_values.span<uint32_t>(TABLE_SIZE);
    auto is = insert_status.span<uint32_t>(NUM_ITEMS);

    auto failedInserts = std::count_if(is.begin(), is.end(), [](auto status){ return status != 1; });
    ASSERT_TRUE(failedInserts == 0) << "eviction chain was too long";

    for(auto i = 0; i < NUM_ITEMS; ++i) {
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

TEST_F(HashTableFixture, hashTableQuery) {
    hash_table_insert();
    query_hash_table();

    auto q_values = query_results.span<uint32_t>(NUM_ITEMS);

    for(auto i = 0; i < NUM_ITEMS; ++i) {
        ASSERT_EQ(q_values[i], values[i]) << fmt::format("result was not equal to expected: {} != {}", q_values[i], values[i]);
    }
}