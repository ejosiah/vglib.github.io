#pragma once

#include "VulkanBuffer.h"
#include <initializer_list>

namespace Barrier {

    void fragmentReadToComputeWrite(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);

    void computeWriteToFragmentRead(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);
    
    void computeWriteToRead(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);

    void computeWriteToTransferRead(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);

    void transferWriteToRead(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);

    void transferWriteToComputeRead(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);

    void transferWriteToWrite(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);

    void transferReadToWrite(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);

    void fragmentReadToComputeWrite(VkCommandBuffer commandBuffer, std::initializer_list<BufferRegion> regions);

    void computeWriteToFragmentRead(VkCommandBuffer commandBuffer, std::initializer_list<BufferRegion> regions);

    void computeWriteToRead(VkCommandBuffer commandBuffer, std::initializer_list<BufferRegion> regions);

    void computeWriteToTransferRead(VkCommandBuffer commandBuffer, std::initializer_list<BufferRegion> regions);

    void transferWriteToRead(VkCommandBuffer commandBuffer, std::initializer_list<BufferRegion> regions);

    void transferWriteToComputeRead(VkCommandBuffer commandBuffer, std::initializer_list<BufferRegion> regions);

    void transferWriteToWrite(VkCommandBuffer commandBuffer, std::initializer_list<BufferRegion> regions);

    void transferReadToWrite(VkCommandBuffer commandBuffer, std::initializer_list<BufferRegion> regions);

}