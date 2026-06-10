#pragma once

#include "PrefixSum.hpp"
#include "VulkanBuffer.h"

namespace gpu::linalg {
    struct SourceEntry {
        float value;
        uint32_t row;
        uint32_t col;
    };

    struct CSRMatrix {
        VulkanBuffer values;
        VulkanBuffer colIndices;
        VulkanBuffer rowOffsets;
        VulkanBuffer counts;
        uint32_t numRows{};
        uint32_t numCols{};
    };

    class CSRMatrixBuilder {
    public:
        CSRMatrixBuilder() = default;

        CSRMatrixBuilder(VulkanDevice& device);

        CSRMatrixBuilder& init(CSRMatrix& matrix, VulkanBuffer source, VulkanBuffer flags);

        void build(VkCommandBuffer cmd, CSRMatrix& matrix, const VulkanBuffer &source, const VulkanBuffer &flags);


    private:
        void clearBuffers(VkCommandBuffer cmd, CSRMatrix& matrix);

        void build(VkCommandBuffer cmd, CSRMatrix& matrix, uint32_t numEntries);

        void assign(VkCommandBuffer commandBuffer, VulkanBuffer from, VulkanBuffer to);

        void createDescriptorSetLayout();

        void updateDescriptorSet(CSRMatrix& matrix, VulkanBuffer source, VulkanBuffer flags);

        void scan(VkCommandBuffer commandBuffer, const VulkanBuffer& input);

        std::vector<PipelineMetaData> pipelines();

        VulkanDevice* device_{};
        ComputePipelines compute_;
        PrefixSum prefixSum_;
        VulkanBuffer offsets_;

        VulkanDescriptorPool descriptorPool_;
        VulkanDescriptorSetLayout descriptorSetLayout_;
        VkDescriptorSet descriptorSet_{};
    };
}
