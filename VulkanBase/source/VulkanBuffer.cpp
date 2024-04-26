#include "VulkanBuffer.h"

std::map<VkBuffer, std::atomic_uint32_t> VulkanBuffer::refCounts;

BufferRegion VulkanBuffer::region(VkDeviceSize start, VkDeviceSize end)  {
    assert(start < end && end <= size);
    return BufferRegion{this, start, end};
}
