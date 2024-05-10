#include "OrderChecker.hpp"
#include "VulkanInitializers.h"
#include "Barrier.hpp"

OrderChecker::OrderChecker(VulkanDevice *device, VkDeviceSize capacity)
: ComputePipelines(device)
, _prefixSum{ device }
, _capacity{ capacity }
{}

void OrderChecker::init() {
    createDescriptorPool();
    createDescriptorSetLayout();
    createPipelines();
    resizeInternalBuffer();
    _prefixSum.init();
}

std::vector<PipelineMetaData> OrderChecker::pipelineMetaData() {
    return {
            {
                "is_sorted_less_than",
                R"(C:\Users\Josiah Ebhomenye\CLionProjects\vglib\data\shaders\less_than_equal.comp.spv)",
                { &_lessThanDescriptorSetLayout },
                { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(_constants) }}
            }
    };
}

void OrderChecker::operator()(VkCommandBuffer commandBuffer, const BufferRegion &data
                         , const BufferRegion &result, uint32_t numBlocks, uint32_t block) {

    if(_capacity < data.size()){
        _capacity = (data.size() * 3)/2;
        resizeInternalBuffer();
    }

    _constants.wordSize = 32/numBlocks;
    _constants.mask = (1ull << _constants.wordSize) - 1;
    _constants.block = block;
    _constants.numEntries = data.sizeAs<uint32_t>();

    vkCmdFillBuffer(commandBuffer, *result.buffer, 0, DataUnitSize, 0xFFFFFFFF);
    copyToInternalDataBuffer(commandBuffer, data);
    isLessThan(commandBuffer);
    _prefixSum.accumulate(commandBuffer, { &_internal.bitSet, 0, data.size() }, *result.buffer);
}

void OrderChecker::isLessThan(VkCommandBuffer commandBuffer) {
    uint32_t gx = std::max( 1u, static_cast<uint32_t>(_constants.numEntries /WorkGroupSize)) + 1;

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("is_sorted_less_than"));
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("is_sorted_less_than"), 0, 1, &_lessThanDescriptorSet, 0, VK_NULL_HANDLE);
    vkCmdPushConstants(commandBuffer, layout("is_sorted_less_than"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(_constants), &_constants);
    vkCmdDispatch(commandBuffer, gx, 1, 1);
    Barrier::computeWriteToRead(commandBuffer, { _internal.bitSet  });
}

void OrderChecker::resizeInternalBuffer() {
    _internal.data = device->createBuffer(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY, _capacity, "is_sorted_internal_data_buffer");
    _internal.bitSet = device->createBuffer(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_TO_CPU, _capacity, "is_sorted_internal_bitset_buffer");

    updateDescriptorSetLayout();
}

void OrderChecker::createDescriptorPool() {
    constexpr uint maxSets = 1;
    std::vector<VkDescriptorPoolSize> poolSizes{
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, maxSets * 2}
    };

    _descriptorPool = device->createDescriptorPool(maxSets, poolSizes);
}

void OrderChecker::createDescriptorSetLayout() {
    _lessThanDescriptorSetLayout =
        device->descriptorSetLayoutBuilder()
            .name("is_sorted_less_than")
            .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .binding(1)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();
    _lessThanDescriptorSet = _descriptorPool.allocate( { _lessThanDescriptorSetLayout }).front();

}

void OrderChecker::updateDescriptorSetLayout() {
    auto writes = initializers::writeDescriptorSets<2>();
    
    writes[0].dstSet = _lessThanDescriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].descriptorCount = 1;
    VkDescriptorBufferInfo dataInfo{ _internal.data, 0, _internal.data.size };
    writes[0].pBufferInfo = &dataInfo;

    writes[1].dstSet = _lessThanDescriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1;
    VkDescriptorBufferInfo bitSetInfo{ _internal.bitSet, 0, _internal.bitSet.size };
    writes[1].pBufferInfo = &bitSetInfo;

    device->updateDescriptorSets(writes);
}

void OrderChecker::copyToInternalDataBuffer(VkCommandBuffer commandBuffer, const BufferRegion &src) {
    VkBufferCopy region{src.offset, 0, src.size()};
    vkCmdCopyBuffer(commandBuffer, *src.buffer, _internal.data, 1, &region);
    Barrier::transferWriteToComputeRead(commandBuffer,  _internal.data );
}

