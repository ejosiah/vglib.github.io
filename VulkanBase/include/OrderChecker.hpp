#pragma once

#include "ComputePipelins.hpp"
#include "PrefixSum.hpp"
#include "VulkanBuffer.h"

class OrderChecker : public ComputePipelines {
public:
    static constexpr VkDeviceSize DataUnitSize = sizeof(uint32_t);
    static constexpr uint32_t ITEMS_PER_WORKGROUP = 8192;
    static constexpr uint32_t MAX_NUM_ITEMS = ITEMS_PER_WORKGROUP * ITEMS_PER_WORKGROUP;

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
        VulkanBuffer sumsBuffer;
        VulkanBuffer sumOfSumsBuffer;
    } _internal{};

protected:
    void isLessThan(VkCommandBuffer commandBuffer);

    void resizeInternalBuffer();

    void createDescriptorPool();

    void createDescriptorSetLayout();

    void updateDescriptorSetLayout();

    void copyToInternalDataBuffer(VkCommandBuffer commandBuffer, const BufferRegion& src);

    void copySum(VkCommandBuffer commandBuffer, const BufferRegion& dst);


protected:
    static constexpr uint32_t WorkGroupSize{256};
    static constexpr const VkDeviceSize INITIAL_CAPACITY = (2 << 20) * DataUnitSize;
    VkDeviceSize _capacity{INITIAL_CAPACITY};

    PrefixSum _prefixSum;


    struct {
        uint32_t itemsPerWorkGroup = ITEMS_PER_WORKGROUP;
        uint32_t numEntries;
    } _constants{};

    VulkanDescriptorPool _descriptorPool;
    VulkanDescriptorSetLayout _descriptorSetLayout;
    VkDescriptorSet _descriptorSet{};
    VkDescriptorSet _sumScanDescriptorSet{};
};