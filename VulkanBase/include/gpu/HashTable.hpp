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

        virtual ~HashTable() = default;

        void init();

        void insert(VkCommandBuffer commandBuffer, BufferRegion keys, std::optional<BufferRegion> values = {});

        void remove(VkCommandBuffer commandBuffer, BufferRegion region,
                    VkPipelineStageFlags2 srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                    VkAccessFlags2 srcAccessMask = VK_ACCESS_2_NONE);

    void find(VkCommandBuffer commandBuffer, BufferRegion keys, BufferRegion result,
              VkPipelineStageFlags2 srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
              VkAccessFlags2 srcAccessMask = VK_ACCESS_2_NONE);

        void getKeys(VkCommandBuffer commandBuffer, VulkanBuffer dst);

        void getValue(VkCommandBuffer commandBuffer, VulkanBuffer dst);

        void getEntries(VkCommandBuffer commandBuffer, VulkanBuffer dstKeys, VulkanBuffer dstValues);

        void getInsertStatus(VkCommandBuffer commandBuffer, VulkanBuffer dst);

        VulkanDescriptorSetLayout& descriptorSetLayout();

        VkDescriptorSet descriptorSet();

        uint32_t capacity() const;

    protected:
        std::vector<PipelineMetaData> pipelineMetaData() final;

    protected:
        virtual std::vector<uint32_t> insert_shader_source() = 0;
        virtual std::vector<uint32_t> find_shader_source() = 0;
        virtual std::vector<uint32_t> remove_shader_source() = 0;

    private:
        void query(VkCommandBuffer commandBuffer, BufferRegion keys, const std::string& shader,
                   VkPipelineStageFlags2 srcStageMask , VkAccessFlags2 srcAccessMask);

        void createBuffers(uint32_t numItems);

        void creatDescriptorSetLayout();

        void createDescriptorSet();

        static void copy(VkCommandBuffer commandBuffer, BufferRegion src, BufferRegion dst);

        void copyFrom(VkCommandBuffer commandBuffer, BufferRegion keys, std::optional<BufferRegion> values = {});

        void copyTo(VkCommandBuffer commandBuffer, BufferRegion values);

        void prepareBuffers(VkCommandBuffer commandBuffer, uint32_t numItems, bool isQuery = false);

        uint32_t computeWorkGroupSize(int numItems);

    private:
        static constexpr int wgSize = 1024;
        VulkanBuffer keys_buffer;
        VulkanBuffer values_buffer;
        VulkanBuffer table_keys;
        VulkanBuffer table_values;
        VulkanBuffer insert_status;
        VulkanBuffer insert_locations;
        VulkanBuffer query_results;
        VulkanBuffer hash_table_info;
        VulkanDescriptorPool* descriptorPool{};
        uint32_t maxIterations{5};
        VulkanDescriptorSetLayout setLayout;
        VkDescriptorSet descriptorSet_{};
        bool keysOnly{};
        VkMemoryBarrier2 barrier{};
        VkDependencyInfo depInfo{};
        struct {
            uint32_t tableSize{};
            uint32_t numItems{};
        } constants;

    };
}