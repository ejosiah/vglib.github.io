#pragma once

#include "VulkanBuffer.h"
#include "VulkanImage.h"
#include <initializer_list>

class Barrier {
private:
    Barrier() = default;
public:

    static void gpuToCpu(VkCommandBuffer commandBuffer);

    [[deprecated("user version without buffer specification")]]
    static void fragmentReadToComputeWrite(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);

    static void fragmentReadToComputeWrite(VkCommandBuffer commandBuffer);

    static void fragmentWriteToComputeRead(VkCommandBuffer commandBuffer);

    static void fragmentWriteToFragmentRead(VkCommandBuffer commandBuffer);

    static void fragmentReadToFragmentWrite(VkCommandBuffer commandBuffer);

    [[deprecated("user version without buffer specification")]]
    static void computeWriteToFragmentRead(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);

    static void computeWriteToFragmentRead(VkCommandBuffer commandBuffer);

    [[deprecated("user version without buffer specification")]]
    static void computeWriteToRead(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);


    [[deprecated("user version without buffer specification")]]
    static void computeWriteToRead(VkCommandBuffer commandBuffer, VulkanBuffer& buffer);

    static void computeWriteToRead(VkCommandBuffer commandBuffer);

    static void computeWriteToHostRead(VkCommandBuffer commandBuffer);

    [[deprecated("user version without buffer specification")]]
    static void computeWriteToTransferRead(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);

    static void computeWriteToTransferRead(VkCommandBuffer commandBuffer);

    static void computeWriteToVertexDraw(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);

    static void computeWriteToDrawIndirect(VkCommandBuffer commandBuffer);

    static void transferWriteToRead(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);

    [[deprecated("user version without buffer specification")]]
    static void transferWriteToComputeRead(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers);

    [[deprecated("user version without buffer specification")]]
    static void transferWriteToComputeRead(VkCommandBuffer commandBuffer, VulkanBuffer& buffers);

    static void transferWriteToComputeRead(VkCommandBuffer commandBuffer);

    static void computeReadToTransferWrite(VkCommandBuffer commandBuffer);

    static void transferWriteToHostRead(VkCommandBuffer commandBuffer, VulkanBuffer& buffers);

    [[deprecated("user version without buffer specification")]]
    static void transferWriteToComputeWrite(VkCommandBuffer commandBuffer, VulkanBuffer& buffers);

    static void transferWriteToComputeWrite(VkCommandBuffer commandBuffer);

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

    static void accelerationStructureUpdateToRayTraceRead(VkCommandBuffer commandBuffer);

    static void accelerationStructureUpdateToRayQueryRead(VkCommandBuffer commandBuffer);

    static void rayTraceReadToAccelerationStructureUpdate(VkCommandBuffer commandBuffer);

    static void rayQueryReadToAccelerationStructureUpdate(VkCommandBuffer commandBuffer);

    static void rayTraceWriteToComputeRead(VkCommandBuffer commandBuffer);

    static void rayTraceWriteToFragmentRead(VkCommandBuffer commandBuffer);

};

class Barriers {
public:
    static void push(const VulkanImage& image, VkImageSubresourceRange subresourceRange,
                        VkPipelineStageFlags2 srcStageMask,VkPipelineStageFlags2 dstStageMask,
                        VkAccessFlags2 srcAccessMask, VkAccessFlags2 dstAccessMask,
                        VkImageLayout oldLayout, VkImageLayout newLayout);

    static void pushAndFlush(VkCommandBuffer commandBuffer, const VulkanImage& image, VkImageSubresourceRange subresourceRange,
                             VkPipelineStageFlags2 srcStageMask,VkPipelineStageFlags2 dstStageMask,
                             VkAccessFlags2 srcAccessMask, VkAccessFlags2 dstAccessMask,
                             VkImageLayout oldLayout, VkImageLayout newLayout);

    static void push(VkPipelineStageFlags2 srcStageMask,VkPipelineStageFlags2 dstStageMask,
                        VkAccessFlags2 srcAccessMask, VkAccessFlags2 dstAccessMask);

    static void pushAndFlush(VkCommandBuffer commandBuffer, VkPipelineStageFlags2 srcStageMask,VkPipelineStageFlags2 dstStageMask,
                        VkAccessFlags2 srcAccessMask, VkAccessFlags2 dstAccessMask);

    static void release(const VulkanImage& image, VkImageSubresourceRange subresourceRange,
                        VkPipelineStageFlags2 srcStageMask, VkAccessFlags2 srcAccessMask,
                        VkImageLayout oldLayout, VkImageLayout newLayout,
                        uint32_t srcQueueFamilyIndex, uint32_t dstQueueFamilyIndex);

    static void acquire(const VulkanImage& image, VkImageSubresourceRange subresourceRange,
                        VkImageLayout oldLayout, VkImageLayout newLayout,
                        uint32_t srcQueueFamilyIndex, uint32_t dstQueueFamilyIndex);

    static void flush(VkCommandBuffer commandBuffer, VkDependencyFlags dependencyFlag = 0);

    static bool flushed();

private:
    Barriers() = default;

    static std::vector<VkImageMemoryBarrier2> imageMemoryBarriers;
    static std::vector<VkBufferMemoryBarrier2> bufferMemoryBarriers;
    static std::vector<VkMemoryBarrier2> memoryBarriers;
    static VkDependencyInfo dependencyInfo;
};