#include "OrderChecker.hpp"
#include "VulkanInitializers.h"
#include "Barrier.hpp"
#include "glsl_shaders.h"

#include <stdexcept>

OrderChecker::OrderChecker(VulkanDevice *device, VkDeviceSize _capacity)
: ComputePipelines(device)
, _prefixSum{ device }
, _capacity{ _capacity }
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
                    "order_checker_scan",
                    data_shaders_order_checking_comp,
                    { &_descriptorSetLayout },
                    { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(_constants)} }
            },
            {
                    "order_checker_sum_scan",
                    data_shaders_prefix_scan_scan_comp,
                    { &_descriptorSetLayout },
                    { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(_constants)} }
            },
            {
                    "add",
                    data_shaders_order_checking_add_comp,
                    { &_descriptorSetLayout },
                    { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(_constants)} }
            }
    };
}

void OrderChecker::operator()(VkCommandBuffer commandBuffer, const BufferRegion &data
                         , const BufferRegion &result, uint32_t numBlocks, uint32_t block) {

    static constexpr uint32_t SumSlot = 1;

    if(data.sizeAs<uint32_t>() > MAX_NUM_ITEMS){
        throw std::runtime_error("too many items to process");
    }


    if(_capacity < data.size() + SumSlot){
        _capacity = (data.size() * 3)/2;
        resizeInternalBuffer();
    }

    const auto numEntries = data.sizeAs<uint32_t>() + SumSlot;
    _constants.numEntries =  numEntries;

    uint32_t numWorkGroups = glm::ceil(static_cast<float>(_constants.numEntries)/static_cast<float>(ITEMS_PER_WORKGROUP));

    vkCmdFillBuffer(commandBuffer, *result.buffer, 0, DataUnitSize, 0xFFFFFFFF);
    Barrier::transferWriteToComputeWrite(commandBuffer, *result.buffer);


//    vkCmdFillBuffer(commandBuffer, _internal.data, 0, _constants.numEntries * DataUnitSize + padding, 0xFFFFFFFF);
    vkCmdFillBuffer(commandBuffer, _internal.data, 0, _internal.data.size, 0xFFFFFFFF);
    copyToInternalDataBuffer(commandBuffer, data);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("order_checker_scan"));
    vkCmdPushConstants(commandBuffer, layout("order_checker_scan"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(_constants), &_constants);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("order_checker_scan"), 0, 1, &_descriptorSet, 0, nullptr);
    vkCmdDispatch(commandBuffer, numWorkGroups, 1, 1);

    if(numWorkGroups > 1){
        _constants.numEntries = numWorkGroups;
        Barrier::computeWriteToRead(commandBuffer, {_internal.data, _internal.sumsBuffer});
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("order_checker_sum_scan"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("order_checker_sum_scan"), 0, 1, &_sumScanDescriptorSet, 0, nullptr);
        vkCmdPushConstants(commandBuffer, layout("order_checker_sum_scan"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(_constants), &_constants);
        vkCmdDispatch(commandBuffer, 1, 1, 1);

        _constants.numEntries = numEntries;
        Barrier::computeWriteToRead(commandBuffer, {_internal.data, _internal.sumOfSumsBuffer});
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("add"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("add"), 0, 1, &_descriptorSet, 0, nullptr);
        vkCmdPushConstants(commandBuffer, layout("add"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(_constants), &_constants);
        vkCmdDispatch(commandBuffer, numWorkGroups, 1, 1);
    }
    copySum(commandBuffer, result);
}

void OrderChecker::isLessThan(VkCommandBuffer commandBuffer) {
    uint32_t gx = std::max( 1u, static_cast<uint32_t>(_constants.numEntries /WorkGroupSize)) + 1;

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("is_sorted_less_than"));
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("is_sorted_less_than"), 0, 1, &_descriptorSet, 0, VK_NULL_HANDLE);
    vkCmdPushConstants(commandBuffer, layout("is_sorted_less_than"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(_constants), &_constants);
    vkCmdDispatch(commandBuffer, gx, 1, 1);
    Barrier::computeWriteToRead(commandBuffer, { _internal.bitSet  });
}

void OrderChecker::resizeInternalBuffer() {
    _internal.data =
            device->createBuffer(
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_GPU_ONLY, _capacity, "order_checker_data");

    _internal.sumOfSumsBuffer =
            device->createBuffer(
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_GPU_ONLY, sizeof(uint), "order_checker_sum_of_sums");

    updateDescriptorSetLayout();
}

void OrderChecker::createDescriptorPool() {
    constexpr uint maxSets = 2;
    std::vector<VkDescriptorPoolSize> poolSizes{
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, maxSets * 2}
    };

    _descriptorPool = device->createDescriptorPool(maxSets, poolSizes);
}

void OrderChecker::createDescriptorSetLayout() {
    _descriptorSetLayout =
        device->descriptorSetLayoutBuilder()
            .name("order_checker_set_layout")
            .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .binding(1)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();
    auto sets = _descriptorPool.allocate( { _descriptorSetLayout, _descriptorSetLayout });
    _descriptorSet = sets[0];
    _sumScanDescriptorSet = sets[1];

}

void OrderChecker::updateDescriptorSetLayout() {
    size_t numItems = _internal.data.sizeAs<int>();
    uint32_t sumsSize = glm::ceil(static_cast<float>(numItems)/static_cast<float>(ITEMS_PER_WORKGROUP)) * sizeof(uint32_t);
    _internal.sumsBuffer = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY, sumsSize);

    VkDescriptorBufferInfo info{ _internal.data, 0, VK_WHOLE_SIZE};
    auto writes = initializers::writeDescriptorSets<4>(_descriptorSet);
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &info;

    VkDescriptorBufferInfo sumsInfo{ _internal.sumsBuffer, 0, VK_WHOLE_SIZE};
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &sumsInfo;

    // for sum scan
    writes[2].dstSet = _sumScanDescriptorSet;
    writes[2].dstBinding = 0;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &sumsInfo;

    VkDescriptorBufferInfo sumsSumInfo{ _internal.sumOfSumsBuffer, 0, VK_WHOLE_SIZE};
    writes[3].dstSet = _sumScanDescriptorSet;
    writes[3].dstBinding = 1;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[3].pBufferInfo = &sumsSumInfo;

    device->updateDescriptorSets(writes);
}

void OrderChecker::copyToInternalDataBuffer(VkCommandBuffer commandBuffer, const BufferRegion &src) {
    VkBufferCopy region{src.offset, 0, src.size()};
    vkCmdCopyBuffer(commandBuffer, *src.buffer, _internal.data, 1, &region);
    Barrier::transferWriteToComputeRead(commandBuffer,  _internal.data );
}

void OrderChecker::copySum(VkCommandBuffer commandBuffer, const BufferRegion &dst) {
    VkDeviceSize  last = (_constants.numEntries - 1) * DataUnitSize;
    VkBufferCopy cRegion{last, 0, DataUnitSize };

    Barrier::computeWriteToTransferRead(commandBuffer, { _internal.data });
    vkCmdCopyBuffer(commandBuffer, _internal.data, *dst.buffer, 1, &cRegion);
    Barrier::transferWriteToComputeRead(commandBuffer, { dst });
}

