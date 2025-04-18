#include "PrefixSum.hpp"

#include <utility>
#include "prefix_sum_glsl_shaders.h"
#include "Barrier.hpp"
#include "glsl_shaders.hpp"

PrefixSum::PrefixSum(VulkanDevice *device, VulkanCommandPool* commandPool)
: ComputePipelines(device)
, _commandPool(commandPool){

}

void PrefixSum::init() {
    bufferOffsetAlignment = device->getLimits().minStorageBufferOffsetAlignment;
    createDescriptorPool();
    createDescriptorSet();
    createPipelines();
    resizeInternalBuffer();
}

void PrefixSum::createDescriptorSet() {
    setLayout =
        device->descriptorSetLayoutBuilder()
            .binding(0)
                .descriptorCount(1)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .binding(1)
                .descriptorCount(1)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
        .createLayout();

    auto sets = descriptorPool.allocate({ setLayout, setLayout });
    descriptorSet = sets.front();
    sumScanDescriptorSet = sets.back();
}

std::vector<PipelineMetaData> PrefixSum::pipelineMetaData() {
    return {
            {
                    "prefix_scan",
                    R"(C:\Users\Josiah Ebhomenye\CLionProjects\vglib\data\shaders\prefix_scan\scan.comp.spv)",
                    { &setLayout },
                    { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants)} }

            },
            {
                    "add",
                    R"(C:\Users\Josiah Ebhomenye\CLionProjects\vglib\data\shaders\prefix_scan\add.comp.spv)",
                    { &setLayout },
                    { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants)} }
            },
            {
                    "prefix_scan_float",
                    R"(C:\Users\Josiah Ebhomenye\CLionProjects\vglib\data\shaders\prefix_scan\scan_float.comp.spv)",
                    { &setLayout },
                    { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants)} }

            },
            {
                    "add_float",
                    R"(C:\Users\Josiah Ebhomenye\CLionProjects\vglib\data\shaders\prefix_scan\add_float.comp.spv)",
                    { &setLayout },
                    { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants)} }
            }
    };
}

void PrefixSum::operator()(VkCommandBuffer commandBuffer, VulkanBuffer &buffer, Operation operation, DataType dataType) {
    (*this)(commandBuffer, { &buffer, 0, buffer.size }, operation, dataType);
}

void PrefixSum::operator()(VkCommandBuffer commandBuffer, const BufferRegion& region, Operation operation, DataType dataType) {
    scanInternal(commandBuffer, region, operation, dataType);
    copyFromInternalBuffer(commandBuffer, region);
}

void PrefixSum::inclusive(VkCommandBuffer commandBuffer, VulkanBuffer &buffer, VkAccessFlags dstAccessMask, VkPipelineStageFlags dstStage) {
    inclusive(commandBuffer, { &buffer, 0, buffer.size});

    auto barrier = initializers::bufferMemoryBarrier();
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = dstAccessMask;
    barrier.buffer = buffer;
    barrier.offset = 0;
    barrier.size = buffer.size;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, dstStage, 0, 0, nullptr, 1, &barrier, 0, nullptr);
}

void PrefixSum::inclusive(VkCommandBuffer commandBuffer, const BufferRegion& region) {
    scanInternal(commandBuffer, region, Operation::Add, DataType::Int);
    copyFromInternalBuffer(commandBuffer, region, DataUnitSize);
}

void PrefixSum::min(VkCommandBuffer commandBuffer, const BufferRegion& data,  VulkanBuffer& result, DataType dataType) {
    accumulate(commandBuffer, data, result, Operation::Min, dataType);
}

void PrefixSum::max(VkCommandBuffer commandBuffer, const BufferRegion& data,  VulkanBuffer& result, DataType dataType) {
    accumulate(commandBuffer, data, result, Operation::Max, dataType);
}



void PrefixSum::accumulate(VkCommandBuffer commandBuffer, VulkanBuffer& data, VulkanBuffer& result, Operation operation, DataType dataType) {
    accumulate(commandBuffer, { &data, 0, data.size}, result, operation, dataType);
}

void PrefixSum::accumulate(VkCommandBuffer commandBuffer, const BufferRegion& data, VulkanBuffer &result, Operation operation, DataType dataType) {
    scanInternal(commandBuffer, data, operation, dataType);
    copySum(commandBuffer, result);
}

void PrefixSum::resizeInternalBuffer() {
    internalDataBuffer =
            device->createBuffer(
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_GPU_ONLY, capacity, "prefix_sum_data");

    sumOfSumsBuffer =
            device->createBuffer(
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_GPU_ONLY, sizeof(uint));
    updateDataDescriptorSets(internalDataBuffer);
}

void PrefixSum::updateDataDescriptorSets(VulkanBuffer &buffer) {
    stagingBuffer = device->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY, buffer.size);
    size_t numItems = buffer.sizeAs<int>();
    uint32_t sumsSize = glm::ceil(static_cast<float>(numItems)/static_cast<float>(ITEMS_PER_WORKGROUP)) * sizeof(uint32_t);
    sumsBuffer = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_GPU_ONLY, sumsSize);
    constants.N = numItems; // TODO move this to when scan is requested as we are now copying data into our internal buffer

    VkDescriptorBufferInfo info{ buffer, 0, VK_WHOLE_SIZE};
    auto writes = initializers::writeDescriptorSets<4>(descriptorSet);
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &info;

    VkDescriptorBufferInfo sumsInfo{ sumsBuffer, 0, VK_WHOLE_SIZE};
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &sumsInfo;

    // for sum scan
    writes[2].dstSet = sumScanDescriptorSet;
    writes[2].dstBinding = 0;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &sumsInfo;

    VkDescriptorBufferInfo sumsSumInfo{ sumOfSumsBuffer, 0, VK_WHOLE_SIZE};
    writes[3].dstSet = sumScanDescriptorSet;
    writes[3].dstBinding = 1;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[3].pBufferInfo = &sumsSumInfo;

    device->updateDescriptorSets(writes);
}

void PrefixSum::createDescriptorPool() {
    constexpr uint maxSets = 2;
    std::vector<VkDescriptorPoolSize> poolSizes{
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, maxSets * 2}
    };

    descriptorPool = device->createDescriptorPool(maxSets, poolSizes);
}

void PrefixSum::copyToInternalBuffer(VkCommandBuffer commandBuffer, const BufferRegion &region) {
    VkBufferCopy cRegion{region.offset, 0, region.size()};
    vkCmdCopyBuffer(commandBuffer, *region.buffer, internalDataBuffer.buffer, 1, &cRegion);
    Barrier::transferWriteToComputeRead(commandBuffer, {internalDataBuffer});
}

void PrefixSum::copyFromInternalBuffer(VkCommandBuffer commandBuffer, const BufferRegion &region, VkDeviceSize srcOffset) {
    VkBufferCopy cRegion{srcOffset, region.offset, region.size()};

    Barrier::computeWriteToTransferRead(commandBuffer, {internalDataBuffer});
    vkCmdCopyBuffer(commandBuffer, internalDataBuffer.buffer, *region.buffer, 1, &cRegion);
    Barrier::transferWriteToComputeRead(commandBuffer, {*region.buffer});
}

void PrefixSum::scanInternal(VkCommandBuffer commandBuffer, BufferRegion data, Operation operation, DataType dataType) {
    if(data.sizeAs<uint32_t>() > MAX_NUM_ITEMS){
        throw DataSizeExceedsMaxSupported{};
    }
    if(capacity < data.size()){
        capacity = data.size() * 2;
        resizeInternalBuffer();
    }
    static constexpr uint32_t SumSlot = 1;
    const  auto numEntries = data.sizeAs<uint32_t>() + SumSlot;
    uint32_t numWorkGroups = glm::ceil(static_cast<float>(numEntries)/static_cast<float>(ITEMS_PER_WORKGROUP));

    auto prefix_scan = dataType == DataType::Int ? "prefix_scan" : "prefix_scan_float";
    auto add = dataType == DataType::Int ? "add" : "add_float";

    constants.N = numEntries;
    constants.operation = to<uint32_t>(operation);
    vkCmdFillBuffer(commandBuffer, internalDataBuffer, data.size(), sizeof(uint32_t), 0);
    copyToInternalBuffer(commandBuffer, data);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline(prefix_scan));
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout(prefix_scan), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer, layout(prefix_scan), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
    vkCmdDispatch(commandBuffer, numWorkGroups, 1, 1);

    if(numWorkGroups > 1){
        constants.N = numWorkGroups;
        Barrier::computeWriteToRead(commandBuffer, {internalDataBuffer, sumsBuffer});
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout(prefix_scan), 0, 1, &sumScanDescriptorSet, 0, nullptr);
        vkCmdPushConstants(commandBuffer, layout(prefix_scan), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
        vkCmdDispatch(commandBuffer, 1, 1, 1);

        constants.N = numEntries;
        Barrier::computeWriteToRead(commandBuffer, {internalDataBuffer, sumOfSumsBuffer});
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline(add));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout(add), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(commandBuffer, layout(add), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
        vkCmdDispatch(commandBuffer, numWorkGroups, 1, 1);
    }
}

void PrefixSum::copySum(VkCommandBuffer commandBuffer, VulkanBuffer& dst) {
    VkBufferCopy cRegion{0, 0, DataUnitSize };

    Barrier::computeWriteToTransferRead(commandBuffer, { sumOfSumsBuffer });
    vkCmdCopyBuffer(commandBuffer, sumOfSumsBuffer, dst, 1, &cRegion);
    Barrier::transferWriteToComputeRead(commandBuffer, { dst });
}
