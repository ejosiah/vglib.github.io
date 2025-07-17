#include "Barrier.hpp"

void Barrier::computeWriteToRead(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers) {
    std::vector<VkBufferMemoryBarrier> barriers(buffers.size());

    for(int i = 0; i < buffers.size(); i++) {
        auto buffer = std::next(buffers.begin(), i);
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barriers[i].offset = 0;
        barriers[i].buffer = *buffer;
        barriers[i].size = buffer->size;
    }

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                         VK_NULL_HANDLE, COUNT(barriers), barriers.data(), 0, VK_NULL_HANDLE);
}

void Barrier::computeWriteToRead(VkCommandBuffer commandBuffer, VulkanBuffer& buffer) {
    VkBufferMemoryBarrier barrier{};

    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.offset = 0;
    barrier.buffer = buffer;
    barrier.size = buffer.size;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                         VK_NULL_HANDLE, 1, &barrier, 0, VK_NULL_HANDLE);
}

void Barrier::computeWriteToRead(VkCommandBuffer commandBuffer) {
    VkMemoryBarrier barrier{};

    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &barrier, 0, VK_NULL_HANDLE, 0, VK_NULL_HANDLE);
}

void
Barrier::computeWriteToTransferRead(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers) {
    std::vector<VkBufferMemoryBarrier> barriers(buffers.size());

    for(int i = 0; i < buffers.size(); i++) {
        auto buffer = std::next(buffers.begin(), i);
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barriers[i].offset = 0;
        barriers[i].buffer = *buffer;
        barriers[i].size = buffer->size;
    }

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         VK_NULL_HANDLE, COUNT(barriers), barriers.data(), 0, VK_NULL_HANDLE);
}
void
Barrier::computeWriteToTransferRead(VkCommandBuffer commandBuffer) {
    VkMemoryBarrier barrier{};

    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                         &barrier, 0, VK_NULL_HANDLE, 0, VK_NULL_HANDLE);
}

void Barrier::computeWriteToVertexDraw(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers) {
    std::vector<VkBufferMemoryBarrier> barriers(buffers.size());

    for(int i = 0; i < buffers.size(); i++) {
        auto buffer = std::next(buffers.begin(), i);
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        barriers[i].offset = 0;
        barriers[i].buffer = *buffer;
        barriers[i].size = buffer->size;
    }

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 0,
                         VK_NULL_HANDLE, COUNT(barriers), barriers.data(), 0, VK_NULL_HANDLE);
}

void Barrier::transferWriteToRead(VkCommandBuffer commandBuffer,
                                  std::initializer_list<VulkanBuffer> buffers) {
    std::vector<VkBufferMemoryBarrier> barriers(buffers.size());

    for(int i = 0; i < buffers.size(); i++) {
        auto buffer = std::next(buffers.begin(), i);
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barriers[i].offset = 0;
        barriers[i].buffer = *buffer;
        barriers[i].size = buffer->size;
    }

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         VK_NULL_HANDLE, COUNT(barriers), barriers.data(), 0, VK_NULL_HANDLE);

}

void Barrier::transferWriteToComputeRead(VkCommandBuffer commandBuffer,
                                         std::initializer_list<VulkanBuffer> buffers) {
    std::vector<VkBufferMemoryBarrier> barriers(buffers.size());

    for(int i = 0; i < buffers.size(); i++) {
        auto buffer = std::next(buffers.begin(), i);
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barriers[i].offset = 0;
        barriers[i].buffer = *buffer;
        barriers[i].size = buffer->size;
    }

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                         VK_NULL_HANDLE, COUNT(barriers), barriers.data(), 0, VK_NULL_HANDLE);
}

void Barrier::transferWriteToComputeRead(VkCommandBuffer commandBuffer, VulkanBuffer& buffer) {
    VkBufferMemoryBarrier barrier{};

    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.offset = 0;
    barrier.buffer = buffer;
    barrier.size = buffer.size;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                         VK_NULL_HANDLE, 1, &barrier, 0, VK_NULL_HANDLE);
}


void Barrier::transferWriteToComputeRead(VkCommandBuffer commandBuffer) {
    VkMemoryBarrier barrier{};

    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &barrier, 0, VK_NULL_HANDLE, 0, VK_NULL_HANDLE);
}

void Barrier::transferWriteToHostRead(VkCommandBuffer commandBuffer, VulkanBuffer& buffer) {
    VkBufferMemoryBarrier barrier{};

    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.offset = 0;
    barrier.buffer = buffer;
    barrier.size = buffer.size;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0,
                         VK_NULL_HANDLE, 1, &barrier, 0, VK_NULL_HANDLE);
}

void Barrier::transferWriteToComputeWrite(VkCommandBuffer commandBuffer, VulkanBuffer& buffer) {
    VkBufferMemoryBarrier barrier{};

    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.offset = 0;
    barrier.buffer = buffer;
    barrier.size = buffer.size;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                         VK_NULL_HANDLE, 1, &barrier, 0, VK_NULL_HANDLE);
}

void Barrier::transferWriteToFragmentRead(VkCommandBuffer commandBuffer, VulkanBuffer& buffer) {
    VkBufferMemoryBarrier barrier{};

    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.offset = 0;
    barrier.buffer = buffer;
    barrier.size = buffer.size;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                         VK_NULL_HANDLE, 1, &barrier, 0, VK_NULL_HANDLE);
}

void Barrier::transferWriteToWrite(VkCommandBuffer commandBuffer,
                                   std::initializer_list<VulkanBuffer> buffers) {
    std::vector<VkBufferMemoryBarrier> barriers(buffers.size());

    for(int i = 0; i < buffers.size(); i++) {
        auto buffer = std::next(buffers.begin(), i);
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barriers[i].offset = 0;
        barriers[i].buffer = *buffer;
        barriers[i].size = buffer->size;
    }

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         VK_NULL_HANDLE, COUNT(barriers), barriers.data(), 0, VK_NULL_HANDLE);
}

void Barrier::transferReadToWrite(VkCommandBuffer commandBuffer,
                                  std::initializer_list<VulkanBuffer> buffers) {
    std::vector<VkBufferMemoryBarrier> barriers(buffers.size());

    for(int i = 0; i < buffers.size(); i++) {
        auto buffer = std::next(buffers.begin(), i);
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barriers[i].offset = 0;
        barriers[i].buffer = *buffer;
        barriers[i].size = buffer->size;
    }

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         VK_NULL_HANDLE, COUNT(barriers), barriers.data(), 0, VK_NULL_HANDLE);
}

void Barrier::fragmentReadToComputeWrite(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers) {
    std::vector<VkBufferMemoryBarrier> barriers(buffers.size());

    for(int i = 0; i < buffers.size(); i++) {
        auto buffer = std::next(buffers.begin(), i);
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[i].offset = 0;
        barriers[i].buffer = *buffer;
        barriers[i].size = buffer->size;
    }

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                         VK_NULL_HANDLE, COUNT(barriers), barriers.data(), 0, VK_NULL_HANDLE);
}

void Barrier::fragmentReadToComputeWrite(VkCommandBuffer commandBuffer) {
    VkMemoryBarrier barrier{};

    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &barrier, 0, VK_NULL_HANDLE, 0, VK_NULL_HANDLE);
}

void Barrier::computeWriteToFragmentRead(VkCommandBuffer commandBuffer, std::initializer_list<VulkanBuffer> buffers) {
    std::vector<VkBufferMemoryBarrier> barriers(buffers.size());

    for(int i = 0; i < buffers.size(); i++) {
        auto buffer = std::next(buffers.begin(), i);
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[i].offset = 0;
        barriers[i].buffer = *buffer;
        barriers[i].size = buffer->size;
    }

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                         VK_NULL_HANDLE, COUNT(barriers), barriers.data(), 0, VK_NULL_HANDLE);
}


void Barrier::computeWriteToFragmentRead(VkCommandBuffer commandBuffer) {
    VkMemoryBarrier barrier{};

    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1,
                         &barrier, 0, VK_NULL_HANDLE, 0, VK_NULL_HANDLE);
}


void Barrier::computeWriteToRead(VkCommandBuffer commandBuffer, std::initializer_list<BufferRegion> regions) {
    std::vector<VkBufferMemoryBarrier> barriers(regions.size());

    for(int i = 0; i < regions.size(); i++) {
        auto region = std::next(regions.begin(), i);
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barriers[i].offset = region->offset;
        barriers[i].buffer = *region->buffer;
        barriers[i].size = region->size();
    }

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                         VK_NULL_HANDLE, COUNT(barriers), barriers.data(), 0, VK_NULL_HANDLE);
}

void
Barrier::computeWriteToTransferRead(VkCommandBuffer commandBuffer, std::initializer_list<BufferRegion> regions) {
    std::vector<VkBufferMemoryBarrier> barriers(regions.size());

    for(int i = 0; i < regions.size(); i++) {
        auto region = std::next(regions.begin(), i);
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barriers[i].offset = region->offset;
        barriers[i].buffer = *region->buffer;
        barriers[i].size = region->size();
    }

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         VK_NULL_HANDLE, COUNT(barriers), barriers.data(), 0, VK_NULL_HANDLE);
}

void Barrier::transferWriteToRead(VkCommandBuffer commandBuffer,
                                  std::initializer_list<BufferRegion> regions) {
    std::vector<VkBufferMemoryBarrier> barriers(regions.size());

    for(int i = 0; i < regions.size(); i++) {
        auto region = std::next(regions.begin(), i);
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barriers[i].offset = region->offset;
        barriers[i].buffer = *region->buffer;
        barriers[i].size = region->size();
    }

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         VK_NULL_HANDLE, COUNT(barriers), barriers.data(), 0, VK_NULL_HANDLE);

}

void Barrier::transferWriteToComputeRead(VkCommandBuffer commandBuffer,
                                         std::initializer_list<BufferRegion> regions) {
    std::vector<VkBufferMemoryBarrier> barriers(regions.size());

    for(int i = 0; i < regions.size(); i++) {
        auto region = std::next(regions.begin(), i);
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barriers[i].offset = region->offset;
        barriers[i].buffer = *region->buffer;
        barriers[i].size = region->size();
    }

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                         VK_NULL_HANDLE, COUNT(barriers), barriers.data(), 0, VK_NULL_HANDLE);
}

void Barrier::transferWriteToWrite(VkCommandBuffer commandBuffer,
                                   std::initializer_list<BufferRegion> regions) {
    std::vector<VkBufferMemoryBarrier> barriers(regions.size());

    for(int i = 0; i < regions.size(); i++) {
        auto region = std::next(regions.begin(), i);
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barriers[i].offset = region->offset;
        barriers[i].buffer = *region->buffer;
        barriers[i].size = region->size();
    }

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         VK_NULL_HANDLE, COUNT(barriers), barriers.data(), 0, VK_NULL_HANDLE);
}

void Barrier::transferReadToWrite(VkCommandBuffer commandBuffer,
                                  std::initializer_list<BufferRegion> regions) {
    std::vector<VkBufferMemoryBarrier> barriers(regions.size());

    for(int i = 0; i < regions.size(); i++) {
        auto region = std::next(regions.begin(), i);
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barriers[i].offset = region->offset;
        barriers[i].buffer = *region->buffer;
        barriers[i].size = region->size();
    }

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         VK_NULL_HANDLE, COUNT(barriers), barriers.data(), 0, VK_NULL_HANDLE);
}

void Barrier::fragmentReadToComputeWrite(VkCommandBuffer commandBuffer, std::initializer_list<BufferRegion> regions) {
    std::vector<VkBufferMemoryBarrier> barriers(regions.size());

    for(int i = 0; i < regions.size(); i++) {
        auto region = std::next(regions.begin(), i);
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[i].offset = region->offset;
        barriers[i].buffer = *region->buffer;
        barriers[i].size = region->size();
    }

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                         VK_NULL_HANDLE, COUNT(barriers), barriers.data(), 0, VK_NULL_HANDLE);
}

void Barrier::computeWriteToFragmentRead(VkCommandBuffer commandBuffer, std::initializer_list<BufferRegion> regions) {
    std::vector<VkBufferMemoryBarrier> barriers(regions.size());

    for(int i = 0; i < regions.size(); i++) {
        auto region = std::next(regions.begin(), i);
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[i].offset = region->offset;
        barriers[i].buffer = *region->buffer;
        barriers[i].size = region->size();
    }

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                         VK_NULL_HANDLE, COUNT(barriers), barriers.data(), 0, VK_NULL_HANDLE);
}

void Barrier::gpuToCpu(VkCommandBuffer commandBuffer) {
    VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, VK_NULL_HANDLE, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT };
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, VK_NULL_HANDLE, 0, VK_NULL_HANDLE);
}

void Barrier::computeWriteToDrawIndirect(VkCommandBuffer commandBuffer) {
    VkMemoryBarrier barrier{};

    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                         &barrier, 0, VK_NULL_HANDLE, 0, VK_NULL_HANDLE);
}

void Barrier::accelerationStructureUpdateToRayTraceRead(VkCommandBuffer commandBuffer) {
    static VkMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
    };

    static VkDependencyInfo info { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    info.memoryBarrierCount = 1;
    info.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &info);
}

void Barrier::rayTraceReadToAccelerationStructureUpdate(VkCommandBuffer commandBuffer) {
    static VkMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR
    };

    static VkDependencyInfo info { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    info.memoryBarrierCount = 1;
    info.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &info);
}

void Barrier::rayTraceWriteToComputeRead(VkCommandBuffer commandBuffer) {
    static VkMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
    };

    static VkDependencyInfo info { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    info.memoryBarrierCount = 1;
    info.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &info);
}

void Barrier::accelerationStructureUpdateToRayQueryRead(VkCommandBuffer commandBuffer) {
    static VkMemoryBarrier2 barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            .dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
    };

    static VkDependencyInfo info { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    info.memoryBarrierCount = 1;
    info.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &info);
}

void Barrier::rayQueryReadToAccelerationStructureUpdate(VkCommandBuffer commandBuffer) {
    static VkMemoryBarrier2 barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
            .dstStageMask = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR
    };

    static VkDependencyInfo info { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    info.memoryBarrierCount = 1;
    info.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &info);
}

std::vector<VkImageMemoryBarrier2> Barriers::imageMemoryBarriers = { };
std::vector<VkBufferMemoryBarrier2> Barriers::bufferMemoryBarriers = { };
std::vector<VkMemoryBarrier2> Barriers::memoryBarriers = { };
VkDependencyInfo Barriers::dependencyInfo = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };

void Barriers::push(const VulkanImage& image, VkImageSubresourceRange subresourceRange,
                 VkPipelineStageFlags2 srcStageMask,VkPipelineStageFlags2 dstStageMask,
                 VkAccessFlags2 srcAccessMask, VkAccessFlags2 dstAccessMask,
                 VkImageLayout oldLayout, VkImageLayout newLayout) {
    imageMemoryBarriers.push_back({
          .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
          .srcStageMask = srcStageMask,
          .srcAccessMask = srcAccessMask,
          .dstStageMask = dstStageMask,
          .dstAccessMask = dstAccessMask,
          .oldLayout = oldLayout,
          .newLayout = newLayout,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .image = image.image,
          .subresourceRange = subresourceRange,
    });
}

void
Barriers::push(VkPipelineStageFlags2 srcStageMask, VkPipelineStageFlags2 dstStageMask, VkAccessFlags2 srcAccessMask,
               VkAccessFlags2 dstAccessMask) {

    memoryBarriers.push_back({
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
         .srcStageMask = srcStageMask,
         .srcAccessMask = srcAccessMask,
         .dstStageMask = dstStageMask,
         .dstAccessMask = dstAccessMask,
    });
}


void Barriers::release(const VulkanImage &image, VkImageSubresourceRange subresourceRange,
                       VkPipelineStageFlags2 srcStageMask, VkAccessFlags2 srcAccessMask, VkImageLayout oldLayout,
                       VkImageLayout newLayout, uint32_t srcQueueFamilyIndex, uint32_t dstQueueFamilyIndex) {

    imageMemoryBarriers.push_back({
          .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
          .srcStageMask = srcStageMask,
          .srcAccessMask = srcAccessMask,
          .dstStageMask = VK_PIPELINE_STAGE_NONE,
          .dstAccessMask = VK_ACCESS_NONE,
          .oldLayout = oldLayout,
          .newLayout = newLayout,
          .srcQueueFamilyIndex = srcQueueFamilyIndex,
          .dstQueueFamilyIndex = dstQueueFamilyIndex,
          .image = image.image,
          .subresourceRange = subresourceRange,
    });
}

void Barriers::acquire(const VulkanImage &image, VkImageSubresourceRange subresourceRange, VkImageLayout oldLayout,
                       VkImageLayout newLayout, uint32_t srcQueueFamilyIndex, uint32_t dstQueueFamilyIndex) {

    imageMemoryBarriers.push_back({
          .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
          .srcStageMask = VK_PIPELINE_STAGE_NONE,
          .srcAccessMask = VK_PIPELINE_STAGE_NONE,
          .dstStageMask = VK_PIPELINE_STAGE_NONE,
          .dstAccessMask = VK_ACCESS_NONE,
          .oldLayout = oldLayout,
          .newLayout = newLayout,
          .srcQueueFamilyIndex = srcQueueFamilyIndex,
          .dstQueueFamilyIndex = dstQueueFamilyIndex,
          .image = image.image,
          .subresourceRange = subresourceRange,
    });
}

void Barriers::flush(VkCommandBuffer commandBuffer) {
    dependencyInfo.imageMemoryBarrierCount = COUNT(imageMemoryBarriers);
    dependencyInfo.pImageMemoryBarriers = imageMemoryBarriers.data();
    dependencyInfo.bufferMemoryBarrierCount = COUNT(bufferMemoryBarriers);
    dependencyInfo.pBufferMemoryBarriers = bufferMemoryBarriers.data();
    dependencyInfo.memoryBarrierCount = COUNT(memoryBarriers);
    dependencyInfo.pMemoryBarriers = memoryBarriers.data();

    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);

    imageMemoryBarriers.clear();
    bufferMemoryBarriers.clear();
    memoryBarriers.clear();
}

bool Barriers::flushed() {
    return imageMemoryBarriers.empty() && bufferMemoryBarriers.empty() && memoryBarriers.empty();
}