#pragma once

#include "VulkanRAII.h"
#include "VulkanFence.h"
#include <vulkan/vulkan.h>
#include <vector>

struct WaitSemaphores{
    std::vector<VkPipelineStageFlags> stages;
    std::vector<VkSemaphore> semaphores;

    [[nodiscard]]
    inline auto size() const {
        return semaphores.size();
    }
};

struct Synchronization {
    WaitSemaphores waitSemaphores{};
    std::vector<VkSemaphore> signalSemaphores{};
    VulkanFence _fence{};

    VkFence fence() const {
        return _fence.fence ? _fence.fence : VK_NULL_HANDLE;
    }
};