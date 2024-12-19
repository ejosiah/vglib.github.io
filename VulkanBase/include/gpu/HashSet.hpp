#pragma once

#include "VulkanDevice.h"
#include "VulkanBuffer.h"
#include "ComputePipelins.hpp"

namespace gpu {
    class HashSet : public ComputePipelines {
    public:
        HashSet(VulkanDevice& device, VulkanDescriptorPool& descriptorPool, uint32_t capacity_);

        void init();

        void insert(VkCommandBuffer commandBuffer, BufferRegion keys, BufferRegion values);

        void find(VkCommandBuffer commandBuffer, BufferRegion values, BufferRegion result);

    private:
        void createInternalBuffers();

        void createDescriptorSet();


    private:
        uint32_t capacity;
        uint32_t maxIterations{5};
        VulkanBuffer keys_buffer;
        VulkanBuffer table;
        VulkanBuffer insert_status;
        VulkanBuffer insert_locations;
        VulkanBuffer query_results;
        VulkanDescriptorSetLayout setLayout;
        VkDescriptorSet descriptorSet{};
        VulkanDescriptorPool* descriptorPool_{};

        struct {
            uint32_t tableSize{};
            uint32_t numItems{};
        } constants;
    };
}