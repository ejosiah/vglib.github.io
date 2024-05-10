#pragma once

#include "ComputePipelins.hpp"
#include "PrefixSum.hpp"
#include "VulkanBuffer.h"

class OrderChecker : public ComputePipelines {
public:
    OrderChecker() = default;

    explicit OrderChecker(VulkanDevice* device, VkDeviceSize capacity = INITIAL_CAPACITY);

    void init();

    std::vector<PipelineMetaData> pipelineMetaData() override;

    void operator()(VkCommandBuffer commandBuffer, const BufferRegion& data, const BufferRegion& result) {
        operator()(commandBuffer, data, result, 1u, 0u);
    }

    void operator()(VkCommandBuffer commandBuffer, const BufferRegion& data, const BufferRegion& result, uint32_t numBlocks, uint32_t block);

    struct {
        VulkanBuffer data;
        VulkanBuffer bitSet;
    } _internal{};

protected:
    void isLessThan(VkCommandBuffer commandBuffer);

    void resizeInternalBuffer();

    void createDescriptorPool();

    void createDescriptorSetLayout();

    void updateDescriptorSetLayout();

    void copyToInternalDataBuffer(VkCommandBuffer commandBuffer, const BufferRegion& src);


protected:
    static constexpr uint32_t WorkGroupSize{256};
    static constexpr VkDeviceSize DataUnitSize = sizeof(uint32_t);
    static constexpr const VkDeviceSize INITIAL_CAPACITY = (1 << 20) * DataUnitSize;
    VkDeviceSize _capacity{INITIAL_CAPACITY};

    PrefixSum _prefixSum;


    struct {
        uint32_t wordSize;
        uint32_t block;
        uint32_t mask;
        uint32_t numEntries;
    } _constants;

    VulkanDescriptorPool _descriptorPool;
    VulkanDescriptorSetLayout _lessThanDescriptorSetLayout;
    VkDescriptorSet _lessThanDescriptorSet{};
};