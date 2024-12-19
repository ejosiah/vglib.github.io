#pragma once

#include "VulkanDevice.h"
#include "VulkanBuffer.h"
#include "ComputePipelins.hpp"

#include <optional>

namespace gpu {
    class HashTable : public ComputePipelines {
    public:
        HashTable() = default;

        HashTable(VulkanDevice &device, VulkanDescriptorPool& descriptorPool, uint32_t capacity, bool keysOnly = false);

        void init();

        void insert(VkCommandBuffer commandBuffer, BufferRegion keys, std::optional<BufferRegion> values = {});

    void find(VkCommandBuffer commandBuffer, BufferRegion keys, BufferRegion result,
              VkPipelineStageFlags2 srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
              VkAccessFlags2 srcAccessMask = VK_ACCESS_2_NONE);

        void getKeys(VkCommandBuffer commandBuffer, VulkanBuffer dst);

        void getValue(VkCommandBuffer commandBuffer, VulkanBuffer dst);

        void getEntries(VkCommandBuffer commandBuffer, VulkanBuffer dstKeys, VulkanBuffer dstValues);

        void getInsertStatus(VkCommandBuffer commandBuffer, VulkanBuffer dst);

    protected:
        std::vector<PipelineMetaData> pipelineMetaData() final;

    protected:
        virtual std::string insertShaderPath()  = 0;
        virtual std::string findShaderPath()  = 0;

    private:
        void createBuffers(uint32_t numItems);

        void creatDescriptorSetLayout();

        void createDescriptorSet();

        static void copy(VkCommandBuffer commandBuffer, BufferRegion src, BufferRegion dst);

        void copyFrom(VkCommandBuffer commandBuffer, BufferRegion keys, std::optional<BufferRegion> values = {});

        void copyTo(VkCommandBuffer commandBuffer, BufferRegion values);

        void prepareBuffers(VkCommandBuffer commandBuffer, uint32_t numItems);

    public:
        VulkanBuffer keys_buffer;
        VulkanBuffer values_buffer;
        VulkanBuffer table_keys;
        VulkanBuffer table_values;
        VulkanBuffer insert_status;
        VulkanBuffer insert_locations;
        VulkanBuffer query_results;
        VulkanDescriptorPool* descriptorPool{};
        uint32_t maxIterations{5};
        VulkanDescriptorSetLayout setLayout;
        VkDescriptorSet descriptorSet{};
        bool keysOnly{};
        VkMemoryBarrier2 barrier{};
        VkDependencyInfo depInfo{};
        struct {
            uint32_t tableSize{};
            uint32_t numItems{};
        } constants;

    };
}