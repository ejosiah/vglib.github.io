#pragma once

#include "VulkanRAII.h"

namespace eular {

    struct Collider {
        enum class Type : uint32_t { Wall, Sdf, Invalid };

        VkDescriptorSet field{VK_NULL_HANDLE};
        VkDescriptorSet velocity{VK_NULL_HANDLE};

        static VulkanDescriptorSetLayout inputDescriptorSetLayout;
        static VulkanDescriptorSetLayout outputDescriptorSetLayout;
        static bool initialized;
    };
}