#include "VulkanFixture.hpp"
#include "Cuckoo.hpp"

#include <stdexcept>

class HashSetFixture : public VulkanFixture {
protected:
    static constexpr VkBufferUsageFlags usage{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    static constexpr uint32_t maxIterations{5};

    void SetUp() override {
        VulkanFixture::SetUp();
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
        insert_locations = device.createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, NUM_ITEMS * sizeof(uint32_t), "insert_locations");
        query_results = device.createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, NUM_ITEMS * sizeof(uint32_t), "query_results");

        std::vector<uint> nullEntries(TABLE_SIZE);
        std::generate(nullEntries.begin(), nullEntries.end(), []{ return 0xFFFFFFFFu; });

        table = device.createCpuVisibleBuffer(nullEntries.data(), BYTE_SIZE(nullEntries), usage);

        setLayout =
            device.descriptorSetLayoutBuilder()
                .binding(0)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(2)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(3)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(4)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();

        descriptorSet = descriptorPool.allocate({ setLayout }).front();
        auto writes = initializers::writeDescriptorSets<5>();
        writes[0].dstSet = descriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].descriptorCount = 1;
        std::vector<VkDescriptorBufferInfo> tableInfo{
                { table, 0, VK_WHOLE_SIZE },
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
        VkDescriptorBufferInfo valueInfo{keys_buffer, 0, VK_WHOLE_SIZE};
        writes[3].pBufferInfo = &valueInfo;

        writes[4].dstSet = descriptorSet;
        writes[4].dstBinding = 4;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[4].descriptorCount = 1;
        VkDescriptorBufferInfo queryInfo{ query_results, 0, VK_WHOLE_SIZE};
        writes[4].pBufferInfo = &queryInfo;

        device.updateDescriptorSets(writes);

    }

    std::vector<PipelineMetaData> pipelineMetaData() override {
        return {
                {
                        .name = "hash_set_insert",
                        .shadePath{"VulkanBase/tests/test_hash_set_insert.comp.spv"},
                        .layouts{&setLayout},
                        .ranges{{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants)}}
                },
                {
                        .name = "hash_set_remove_duplicates",
                        .shadePath{"VulkanBase/tests/test_hash_set_remove_duplicates.comp.spv"},
                        .layouts{&setLayout},
                        .ranges{{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants)}}
                },
            {
                    .name = "hash_set_query",
                    .shadePath{ "VulkanBase/tests/test_hash_set_query.comp.spv" },
                    .layouts{  &setLayout },
                    .ranges{ { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants)  }}
                },
        };
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

    void hash_set_insert() {
        execute([&](auto commandBuffer){
            uint32_t groupCount = NUM_ITEMS / 1024;

            VkMemoryBarrier2 barrier;
            VkDependencyInfo depInfo;
            transferData(barrier, depInfo, commandBuffer);

            for(auto i = 0; i < maxIterations; ++i) {
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("hash_set_insert"));
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("hash_set_insert"), 0, 1,&descriptorSet, 0, 0);
                vkCmdPushConstants(commandBuffer, layout("hash_set_insert"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
                vkCmdDispatch(commandBuffer, groupCount, 1, 1);

                vkCmdPipelineBarrier2(commandBuffer, &depInfo);
            }

            // hash_set_remove_duplicates
//            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("hash_set_remove_duplicates"));
//            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("hash_set_remove_duplicates"), 0, 1,&descriptorSet, 0, 0);
//            vkCmdPushConstants(commandBuffer, layout("hash_set_remove_duplicates"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
//            vkCmdDispatch(commandBuffer, groupCount, 1, 1);
        });
    }

    void hash_set_query() {
        execute([&](auto commandBuffer){
            uint32_t groupCount = NUM_ITEMS / 1024;

            VkMemoryBarrier2 barrier;
            VkDependencyInfo depInfo;
            transferData(barrier, depInfo, commandBuffer);

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("hash_set_query"));
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("hash_set_query"), 0, 1,&descriptorSet, 0, 0);
            vkCmdPushConstants(commandBuffer, layout("hash_set_query"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
            vkCmdDispatch(commandBuffer, groupCount, 1, 1);
        });
    }

    std::vector<uint32_t> keys;
    VulkanBuffer keys_buffer;
    VulkanBuffer table;
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
    }}