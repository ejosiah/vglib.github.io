#include "PrefixSum.hpp"
#include "prefix_sum_glsl_shaders.h"

PrefixSum::PrefixSum(VulkanDevice *device, VulkanCommandPool* commandPool)
: ComputePipelines(device)
, _commandPool(commandPool){

}

void PrefixSum::init() {
    bufferOffsetAlignment = device->getLimits().minStorageBufferOffsetAlignment;
    createDescriptorPool();
    createDescriptorSet();
    createPipelines();
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
                    __ps_glsl_scan_comp_spv,
                    { &setLayout }
            },
            {
                    "add",
                    __ps_glsl_add_comp_spv,
                    { &setLayout },
                    { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants)} }
            }
    };
}

void PrefixSum::operator()(VkCommandBuffer commandBuffer, VulkanBuffer &buffer) {
    int size = buffer.sizeAs<int>();
    int numWorkGroups = std::abs(size - 1)/ITEMS_PER_WORKGROUP + 1;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("prefix_scan"));
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("prefix_scan"), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdDispatch(commandBuffer, numWorkGroups, 1, 1);

    if(numWorkGroups > 1){
        addBufferMemoryBarriers(commandBuffer, {&buffer, &sumsBuffer});
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("prefix_scan"), 0, 1, &sumScanDescriptorSet, 0, nullptr);
        vkCmdDispatch(commandBuffer, 1, 1, 1);

        addBufferMemoryBarriers(commandBuffer, { &buffer, &sumsBuffer });
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("add"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("add"), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(commandBuffer, layout("add"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
        vkCmdDispatch(commandBuffer, numWorkGroups, 1, 1);
    }
}

void PrefixSum::inclusive(VkCommandBuffer commandBuffer, VulkanBuffer &buffer, VkAccessFlags dstAccessMask, VkPipelineStageFlags dstStage) {
    (*this)(commandBuffer, buffer);
    VkDeviceSize offset = sizeof(int);

    VkBufferCopy region{offset, 0, buffer.size - offset};
    vkCmdCopyBuffer(commandBuffer, buffer, stagingBuffer, 1, &region);
    addBufferTransferBarriers(commandBuffer, { &stagingBuffer });

    region = VkBufferCopy{0, buffer.size - offset, offset};
    vkCmdCopyBuffer(commandBuffer, sumsBuffer, stagingBuffer, 1, &region);

    addBufferTransferBarriers(commandBuffer, { &stagingBuffer });

    region = VkBufferCopy{0, 0, stagingBuffer.size};
    vkCmdCopyBuffer(commandBuffer, stagingBuffer, buffer, 1, &region);

    auto barrier = initializers::bufferMemoryBarrier();
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = dstAccessMask;
    barrier.buffer = buffer;
    barrier.offset = 0;
    barrier.size = buffer.size;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, dstStage, 0, 0, nullptr, 1, &barrier, 0, nullptr);
}

void PrefixSum::updateDataDescriptorSets(VulkanBuffer &buffer) {
    stagingBuffer = device->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY, buffer.size);
    size_t numItems = buffer.sizeAs<int>();
    VkDeviceSize sumsSize = (std::abs(int(numItems - 1))/ITEMS_PER_WORKGROUP + 1) * sizeof(int);  // FIXME
    sumsSize = alignedSize(sumsSize, bufferOffsetAlignment) + sizeof(int);
    sumsBuffer = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, sumsSize);
    constants.N = numItems;

    VkDescriptorBufferInfo info{ buffer, 0, VK_WHOLE_SIZE};
    auto writes = initializers::writeDescriptorSets<4>(descriptorSet);
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &info;

    VkDescriptorBufferInfo sumsInfo{ sumsBuffer, 0, sumsBuffer.size - sizeof(int)}; // TODO FIXME
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

    VkDescriptorBufferInfo sumsSumInfo{ sumsBuffer, sumsBuffer.size - sizeof(int), VK_WHOLE_SIZE}; // TODO FIXME
    writes[3].dstSet = sumScanDescriptorSet;
    writes[3].dstBinding = 1;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[3].pBufferInfo = &sumsSumInfo;

    device->updateDescriptorSets(writes);
}

void PrefixSum::addBufferMemoryBarriers(VkCommandBuffer commandBuffer, const std::vector<VulkanBuffer *> &buffers) {
    std::vector<VkBufferMemoryBarrier> barriers(buffers.size());

    for(int i = 0; i < buffers.size(); i++) {
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barriers[i].offset = 0;
        barriers[i].buffer = *buffers[i];
        barriers[i].size = buffers[i]->size;
    }

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                         nullptr, COUNT(barriers), barriers.data(), 0, nullptr);
}

void PrefixSum::addBufferTransferBarriers(VkCommandBuffer commandBuffer, const std::vector<VulkanBuffer *> &buffers) {
    std::vector<VkBufferMemoryBarrier> barriers(buffers.size());

    for(int i = 0; i < buffers.size(); i++) {
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barriers[i].offset = 0;
        barriers[i].buffer = *buffers[i];
        barriers[i].size = buffers[i]->size;
    }

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         nullptr, COUNT(barriers), barriers.data(), 0, nullptr);
}

void PrefixSum::createDescriptorPool() {
    constexpr uint maxSets = 2;
    std::vector<VkDescriptorPoolSize> poolSizes{
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, maxSets * 2}
    };

    descriptorPool = device->createDescriptorPool(maxSets, poolSizes);
}