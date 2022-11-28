#pragma once

#include "VulkanRAII.h"
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