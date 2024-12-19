#include "VulkanFixture.hpp"
#include <stdexcept>
#include <ranges>
#define NULL_LOCATION 0xFFFFFFFFu
#define KEY_EMPTY 0xFFFFFFFFu
#define NOT_FOUND 0xFFFFFFFFu

#define a1 100000u
#define b1 200u
#define a2 300000u
#define b2 489902u
#define a3 800000u
#define b3 10248089u
#define a4 9458373u
#define b4 1234838u

class HashTableFixture : public VulkanFixture {
protected:
    static constexpr uint32_t NUM_ITEMS{1 << 16};
    static constexpr uint32_t TABLE_SIZE{4 * NUM_ITEMS};
    static constexpr VkBufferUsageFlags usage{ VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    static constexpr uint32_t p = 334214459;    // hash prime number
    static constexpr uint32_t maxIterations{5};

    std::vector<PipelineMetaData> pipelineMetaData() override {
        return {
                {
                    .name = "hash_table_insert",
                    .shadePath{ "VulkanBase/tests/test_hash_table_insert.comp.spv" },
                    .layouts{  &setLayout },
                    .ranges{ { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants)  }}
                },
                {
                    .name = "hash_table_query",
                    .shadePath{ "VulkanBase/tests/test_hash_table_query.comp.spv" },
                    .layouts{  &setLayout },
                    .ranges{ { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants)  }}
                },
        };
    }

    void SetUp() override {
        VulkanFixture::SetUp();
        setupTestData();
    }

    void postVulkanInit() override {
        keys_buffer = device.createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, NUM_ITEMS * sizeof(uint32_t), "hash_keys");
        values_buffer = device.createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, NUM_ITEMS * sizeof(uint32_t), "hash_values");
        insert_status = device.createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, NUM_ITEMS * sizeof(uint32_t), "insert_status");
        insert_locations = device.createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, NUM_ITEMS * sizeof(uint32_t), "insert_locations");
        query_results = device.createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, NUM_ITEMS * sizeof(uint32_t), "query_results");

        std::vector<uint> nullEntries(TABLE_SIZE);
        std::generate(nullEntries.begin(), nullEntries.end(), []{ return 0xFFFFFFFFu; });

        table_keys = device.createCpuVisibleBuffer(nullEntries.data(), BYTE_SIZE(nullEntries), usage);
        table_values = device.createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, TABLE_SIZE * sizeof(uint32_t), "table_values");

        setLayout =
            device.descriptorSetLayoutBuilder()
                .binding(0)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(2)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(2)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(2)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(2)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(3)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(4)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(5)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .createLayout();

        descriptorSet = descriptorPool.allocate({ setLayout }).front();
        auto writes = initializers::writeDescriptorSets<6>();
        writes[0].dstSet = descriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].descriptorCount = 2;
        std::vector<VkDescriptorBufferInfo> tableInfo{
                { table_keys, 0, VK_WHOLE_SIZE },
                {table_values, 0, VK_WHOLE_SIZE}
        };
        writes[0].pBufferInfo = tableInfo.data();

        writes[1].dstSet = descriptorSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].descriptorCount = 1;
        VkDescriptorBufferInfo isInfo{ insert_status, 0, VK_WHOLE_SIZE};
        writes[1].pBufferInfo = &isInfo;

        writes[2].dstSet = descriptorSet;
        writes[2].dstBinding = 2;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].descriptorCount = 1;
        VkDescriptorBufferInfo ilInfo{ insert_locations, 0, VK_WHOLE_SIZE};
        writes[2].pBufferInfo = &ilInfo;

        writes[3].dstSet = descriptorSet;
        writes[3].dstBinding = 3;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[3].descriptorCount = 1;
        VkDescriptorBufferInfo keyInfo{ keys_buffer, 0, VK_WHOLE_SIZE};
        writes[3].pBufferInfo = &keyInfo;

        writes[4].dstSet = descriptorSet;
        writes[4].dstBinding = 4;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[4].descriptorCount = 1;
        VkDescriptorBufferInfo valueInfo{ values_buffer, 0, VK_WHOLE_SIZE};
        writes[4].pBufferInfo = &valueInfo;

        writes[5].dstSet = descriptorSet;
        writes[5].dstBinding = 5;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[5].descriptorCount = 1;
        VkDescriptorBufferInfo queryInfo{ query_results, 0, VK_WHOLE_SIZE};
        writes[5].pBufferInfo = &queryInfo;

        device.updateDescriptorSets(writes);

    }

    void transferData(VkMemoryBarrier2& barrier, VkDependencyInfo& info, VkCommandBuffer commandBuffer) {
        barrier =  {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
        };

        info = {
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .memoryBarrierCount = 1,
                .pMemoryBarriers = &barrier
        };

        vkCmdFillBuffer(commandBuffer, insert_status, 0, NUM_ITEMS * sizeof(uint32_t), 0);
        vkCmdFillBuffer(commandBuffer, insert_locations, 0, NUM_ITEMS * sizeof(uint32_t), 0xFFFFFFFFu);

        vkCmdPipelineBarrier2(commandBuffer, &info);

        barrier.srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    }

    void hash_table_insert() {
        execute([&](auto commandBuffer){
            uint32_t groupCount = NUM_ITEMS / 1024;

            VkMemoryBarrier2 barrier;
            VkDependencyInfo depInfo;
            transferData(barrier, depInfo, commandBuffer);

            for(auto i = 0; i < maxIterations; ++i) {
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("hash_table_insert"));
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("hash_table_insert"), 0, 1,&descriptorSet, 0, 0);
                vkCmdPushConstants(commandBuffer, layout("hash_table_insert"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
                vkCmdDispatch(commandBuffer, groupCount, 1, 1);

                if(i < maxIterations - 1) {
                    vkCmdPipelineBarrier2(commandBuffer, &depInfo);
                }
            }
        });
    }

    void query_hash_table() {
        execute([&](auto commandBuffer){
            uint32_t groupCount = NUM_ITEMS / 1024;

            VkMemoryBarrier2 barrier;
            VkDependencyInfo depInfo;
            transferData(barrier, depInfo, commandBuffer);

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("hash_table_query"));
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("hash_table_query"), 0, 1,&descriptorSet, 0, 0);
            vkCmdPushConstants(commandBuffer, layout("hash_table_query"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
            vkCmdDispatch(commandBuffer, groupCount, 1, 1);
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

    static constexpr uint32_t hash1(uint32_t key) {
        return (a1 ^ key + b1) % p % TABLE_SIZE;
    }

    static constexpr uint32_t hash2(uint32_t key) {
        return (a2 ^ key + b2) % p % TABLE_SIZE;
    }

    static constexpr uint32_t hash3(uint32_t key) {
        return (a3 ^ key + b3) % p % TABLE_SIZE;
    }

    static constexpr uint32_t hash4(uint32_t key) {
        return (a4 ^ key + b4) % p % TABLE_SIZE;
    }

    std::vector<uint32_t> keys;
    std::vector<uint32_t> values;
    VulkanBuffer keys_buffer;
    VulkanBuffer values_buffer;
    VulkanBuffer table_keys;
    VulkanBuffer table_values;
    VulkanBuffer insert_status;
    VulkanBuffer insert_locations;
    VulkanBuffer query_results;
    VulkanDescriptorSetLayout setLayout;
    VkDescriptorSet descriptorSet{};
    struct {
        uint32_t tableSize{TABLE_SIZE};
        uint32_t numItems{NUM_ITEMS};
    } constants;
};


TEST_F(HashTableFixture, HashTableInsert) {
    hash_table_insert();

    auto tableKeys = table_keys.span<uint32_t>(TABLE_SIZE);
    auto tableValues =  table_values.span<uint32_t>(TABLE_SIZE);
    auto is = insert_status.span<uint32_t>(NUM_ITEMS);
    auto locs = insert_locations.span<uint32_t>(NUM_ITEMS);

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

        ASSERT_TRUE(found) << fmt::format("entry({} -> {}) status: {}, loc: {} not found in hash table", key, value, is[i], locs[i]);
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