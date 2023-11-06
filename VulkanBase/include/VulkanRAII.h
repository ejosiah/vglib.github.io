#pragma once

#include <vulkan/vulkan.h>
#include "RefCounted.hpp"

#include <string>
#include <memory>

extern VkDevice vkDevice;

inline std::string toString(VkObjectType objectType) {
    switch(objectType) {
        case VK_OBJECT_TYPE_PIPELINE:
            return "VkPipeline";
        case VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT:
            return "VkDescriptorSetLayout";
        case VK_OBJECT_TYPE_IMAGE_VIEW:
            return "VkImageView";
        case VK_OBJECT_TYPE_SAMPLER:
            return "VkSampler";
        case VK_OBJECT_TYPE_PIPELINE_CACHE:
            return "VkPipelineCache";
        case VK_OBJECT_TYPE_PIPELINE_LAYOUT:
            return "VkPipelineLayout";
        case VK_OBJECT_TYPE_SHADER_MODULE:
            return "VkShaderModule";
        default:
            return "Object type unknown";
    }
}



template<typename Handle, typename Deleter>
struct VulkanHandle {

    VulkanHandle() = default;

    VulkanHandle(VkDevice device, Handle handle)
    : device(device)
    , handle(handle)
    , shared_ref(handle, Deleter())
    {}


    operator uint64_t() const {
        return (uint64_t)handle;
    }

    operator bool() const {
        return handle != VK_NULL_HANDLE;
    }

    VkDevice device = VK_NULL_HANDLE;
    Handle handle = VK_NULL_HANDLE;

private:
    std::shared_ptr<void> shared_ref;
};

//struct SemaphoreDeleter{
//
//    inline void operator()(VkDevice device, VkSemaphore semaphore){
//        vkDestroySemaphore(device, semaphore, nullptr);
//    }
//};

#define VULKAN_RAII(Resource, ObjectType) \
struct Resource##Deleter{ \
    inline void operator()(void* resource){ \
        spdlog::debug("No more references to {}[{}], will be destroying it now", toString(ObjectType), (uint64_t)resource);  \
        vkDestroy##Resource(vkDevice, reinterpret_cast<Vk##Resource>(resource), nullptr);          \
    }      \
};                       \
using Vulkan##Resource = VulkanHandle<Vk##Resource, Resource##Deleter>;

VULKAN_RAII(Pipeline, VK_OBJECT_TYPE_PIPELINE)
VULKAN_RAII(DescriptorSetLayout, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT)
VULKAN_RAII(ImageView, VK_OBJECT_TYPE_IMAGE_VIEW)
VULKAN_RAII(Sampler, VK_OBJECT_TYPE_SAMPLER)
VULKAN_RAII(PipelineCache, VK_OBJECT_TYPE_PIPELINE_CACHE)
VULKAN_RAII(PipelineLayout, VK_OBJECT_TYPE_PIPELINE_LAYOUT)
VULKAN_RAII(ShaderModule, VK_OBJECT_TYPE_SHADER_MODULE)
//VULKAN_RAII(QueryPool)