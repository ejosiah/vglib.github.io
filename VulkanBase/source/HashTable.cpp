#include "gpu/HashTable.hpp"

gpu::HashTable::HashTable(VulkanDevice &device, VulkanDescriptorPool& descriptorPool_, uint32_t capacity, bool keysOnly_)
: ComputePipelines(&device)
, descriptorPool(&descriptorPool_)
, constants{ .tableSize = capacity }
, keysOnly(keysOnly_)
{}

void gpu::HashTable::insert(VkCommandBuffer commandBuffer, BufferRegion keys, std::optional<BufferRegion> values) {
    auto numItems = keys.sizeAs<uint32_t>();

    constants.numItems = numItems;
    uint32_t groupCount = numItems / 1024;

    prepareBuffers(commandBuffer, numItems);
    copyFrom(commandBuffer, keys, values);

    vkCmdPipelineBarrier2(commandBuffer, &depInfo);

    barrier.srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    for(auto i = 0; i < maxIterations; ++i) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("hash_table_insert"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("hash_table_insert"), 0, 1,&descriptorSet, 0, 0);
        vkCmdPushConstants(commandBuffer, layout("hash_table_insert"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
        vkCmdDispatch(commandBuffer, groupCount, 1, 1);

        if(i < maxIterations - 1) {
            vkCmdPipelineBarrier2(commandBuffer, &depInfo);
        }
    }
}

void gpu::HashTable::find(VkCommandBuffer commandBuffer, BufferRegion keys, BufferRegion result,
                          VkPipelineStageFlags2 srcStageMask, VkAccessFlags2 srcAccessMask) {
    auto numItems = keys.sizeAs<uint32_t>();

    constants.numItems = numItems;
    uint32_t groupCount = numItems / 1024;

    barrier =  { // TODO do we need this barrier
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask =  srcStageMask,
            .srcAccessMask =  srcAccessMask,
            .dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT
    };

    depInfo = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &barrier
    };

    vkCmdPipelineBarrier2(commandBuffer, &depInfo);

    copyFrom(commandBuffer, keys);

    barrier.srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier2(commandBuffer, &depInfo);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("hash_table_query"));
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("hash_table_query"), 0, 1,&descriptorSet, 0, 0);
    vkCmdPushConstants(commandBuffer, layout("hash_table_query"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
    vkCmdDispatch(commandBuffer, groupCount, 1, 1);

    barrier.srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier2(commandBuffer, &depInfo);

    copy(commandBuffer, query_results.region(0), result);
}

void gpu::HashTable::getKeys(VkCommandBuffer commandBuffer, VulkanBuffer dst) {
    copy(commandBuffer, table_keys.region(0), dst.region(0));
}

void gpu::HashTable::getValue(VkCommandBuffer commandBuffer, VulkanBuffer dst) {
    auto src = keysOnly ? table_keys.region(0) : table_values.region(0);
    copy(commandBuffer, src, dst.region(0));
}

void gpu::HashTable::getEntries(VkCommandBuffer commandBuffer, VulkanBuffer dstKeys, VulkanBuffer dstValues) {
    assert(!keysOnly);
    copy(commandBuffer, table_keys.region(0), dstKeys.region(0));
    copy(commandBuffer, table_values.region(0), dstValues.region(0));
}


std::vector<PipelineMetaData> gpu::HashTable::pipelineMetaData() {
    return {
            {
                    .name = "hash_table_insert",
                    .shadePath{ insertShaderPath() },
                    .layouts{  &setLayout },
                    .ranges{ { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants)  }}
            },
            {
                    .name = "hash_table_query",
                    .shadePath{ findShaderPath() },
                    .layouts{  &setLayout },
                    .ranges{ { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants)  }}
            },
    };}

void gpu::HashTable::init() {
    createBuffers(constants.tableSize/2);
    creatDescriptorSetLayout();
    createDescriptorSet();
    createPipelines();
}

void gpu::HashTable::createBuffers(uint32_t numItems) {
    constexpr VkBufferUsageFlags usage{ VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
    keys_buffer = device->createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, numItems * sizeof(uint32_t), "hash_keys");
    values_buffer = device->createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, numItems * sizeof(uint32_t), "hash_values");
    insert_status = device->createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, numItems * sizeof(uint32_t), "insert_status");
    insert_locations = device->createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, numItems * sizeof(uint32_t), "insert_locations");
    query_results = device->createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, numItems * sizeof(uint32_t), "query_results");

    const auto tableSize = constants.tableSize;
    std::vector<uint> nullEntries(tableSize);
    std::generate(nullEntries.begin(), nullEntries.end(), []{ return 0xFFFFFFFFu; });

    table_keys = device->createCpuVisibleBuffer(nullEntries.data(), BYTE_SIZE(nullEntries), usage);
    table_values = device->createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, tableSize * sizeof(uint32_t), "table_values");
}

void gpu::HashTable::creatDescriptorSetLayout() {
    setLayout =
            device->descriptorSetLayoutBuilder()
                .binding(0)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(2)
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
                .binding(5)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();
}

void gpu::HashTable::createDescriptorSet() {

    descriptorSet = descriptorPool->allocate({ setLayout }).front();
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

    device->updateDescriptorSets(writes);
}

void gpu::HashTable::copyFrom(VkCommandBuffer commandBuffer, BufferRegion keys, std::optional<BufferRegion> values) {
    copy(commandBuffer, keys, keys_buffer.region(0));
    if(values.has_value()) {
        copy(commandBuffer, *values, values_buffer.region(0));
    }
}

void gpu::HashTable::copyTo(VkCommandBuffer commandBuffer, BufferRegion values) {
    if(keysOnly) {
        copy(commandBuffer, keys_buffer.region(0), values);
    }else {
        copy(commandBuffer, values_buffer.region(0), values);
    }
}

void gpu::HashTable::copy(VkCommandBuffer commandBuffer, BufferRegion src, BufferRegion dst) {
    VkBufferCopy copyRegion{0, 0, dst.size()};
    vkCmdCopyBuffer(commandBuffer, *src.buffer, *dst.buffer, 1, &copyRegion);
}

void gpu::HashTable::getInsertStatus(VkCommandBuffer commandBuffer, VulkanBuffer dst) {
    copy(commandBuffer, insert_status.region(0), dst.region(0));
}

void gpu::HashTable::prepareBuffers(VkCommandBuffer commandBuffer, uint32_t numItems) {
    barrier =  {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_NONE,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_NONE,
            .dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
    };

    depInfo = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &barrier
    };

    vkCmdFillBuffer(commandBuffer, insert_status, 0, numItems * sizeof(uint32_t), 0);
    vkCmdFillBuffer(commandBuffer, insert_locations, 0, numItems * sizeof(uint32_t), 0xFFFFFFFFu);
}
