#pragma once

#include "VulkanBuffer.h"
#include <initializer_list>

class Barrier {
private:
    Barrier() = default;
public:

    static void gpuToCpu(VkCommandBuffer commandBuffer);

    [[deprecated("user version without buffer specification")]]
    static void fragmentReadToComputeWrite(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);

    static void fragmentReadToComputeWrite(VkCommandBuffer commandBuffer);

    [[deprecated("user version without buffer specification")]]
    static void computeWriteToFragmentRead(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);

    static void computeWriteToFragmentRead(VkCommandBuffer commandBuffer);

    static void computeWriteToRead(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);

    [[deprecated("user version without buffer specification")]]
    static void computeWriteToRead(VkCommandBuffer commandBuffer, VulkanBuffer& buffer);

    static void computeWriteToRead(VkCommandBuffer commandBuffer);

    [[deprecated("user version without buffer specification")]]
    static void computeWriteToTransferRead(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);

    static void computeWriteToTransferRead(VkCommandBuffer commandBuffer);

    static void computeWriteToVertexDraw(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);

    static void transferWriteToRead(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);

    [[deprecated("user version without buffer specification")]]
    static void transferWriteToComputeRead(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);

    [[deprecated("user version without buffer specification")]]
    static void transferWriteToComputeRead(VkCommandBuffer commandBuffer, VulkanBuffer& buffers);

    static void transferWriteToComputeRead(VkCommandBuffer commandBuffer);

    static void transferWriteToHostRead(VkCommandBuffer commandBuffer, VulkanBuffer& buffers);

    static void transferWriteToComputeWrite(VkCommandBuffer commandBuffer, VulkanBuffer& buffers);

    [[deprecated("user version without buffer specification")]]
    static void transferWriteToFragmentRead(VkCommandBuffer commandBuffer, VulkanBuffer& buffers);

    static void transferWriteToFragmentRead(VkCommandBuffer commandBuffer);

    static void transferWriteToWrite(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);

    static void transferReadToWrite(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);

    [[deprecated("user version without buffer specification")]]
    static void fragmentReadToComputeWrite(VkCommandBuffer commandBuffer, std::initializer_list<BufferRegion> regions);

    static void computeWriteToFragmentRead(VkCommandBuffer commandBuffer, std::initializer_list<BufferRegion> regions);

    static void computeWriteToRead(VkCommandBuffer commandBuffer, std::initializer_list<BufferRegion> regions);

    static void computeWriteToTransferRead(VkCommandBuffer commandBuffer, std::initializer_list<BufferRegion> regions);

    static void transferWriteToRead(VkCommandBuffer commandBuffer, std::initializer_list<BufferRegion> regions);

    [[deprecated("user version without buffer specification")]]
    static void transferWriteToComputeRead(VkCommandBuffer commandBuffer, std::initializer_list<BufferRegion> regions);

    static void transferWriteToWrite(VkCommandBuffer commandBuffer, std::initializer_list<BufferRegion> regions);

    static void transferReadToWrite(VkCommandBuffer commandBuffer, std::initializer_list<BufferRegion> regions);

};