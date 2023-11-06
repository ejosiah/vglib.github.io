#pragma once

#include <vulkan/vulkan.h>
#include "RefCounted.hpp"

#include <string>

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



template<typename Handle, typename Deleter, VkObjectType objectType>
struct VulkanHandle : RefCounted {

    VulkanHandle() = default;

    VulkanHandle(VkDevice device, Handle handle)
    : RefCounted((ResourceHandle)handle, [device, handle](ResourceHandle){ Deleter()(device, handle); }, toString(objectType))
    , device(device)
    , handle(handle)
    {}

    VulkanHandle(const VulkanHandle& source)
    : RefCounted(source)
    , device(source.device)
    , handle(source.handle)
    {}

    VulkanHandle(VulkanHandle&& source) noexcept {
        operator=(static_cast<VulkanHandle&&>(source));
    }

    ~VulkanHandle() override = default;

    VulkanHandle& operator=(const VulkanHandle& source) {
        if(this == &source) return *this;
        copyRef(source);

        this->device = source.device;
        this->handle = source.handle;

        return *this;
    };

    VulkanHandle& operator=(VulkanHandle&& source) noexcept {
        if(this == &source) return *this;

        if(handle){
            this->~VulkanHandle();
        }

        moveRef(static_cast<RefCounted&&>(source));
        this->device = std::exchange(source.device, VK_NULL_HANDLE);
        this->handle = std::exchange(source.handle, VK_NULL_HANDLE);

        return *this;
    }

    operator uint64_t() const {
        return (uint64_t)handle;
    }

    operator bool() const {
        return handle != VK_NULL_HANDLE;
    }

    VkDevice device = VK_NULL_HANDLE;
    Handle handle = VK_NULL_HANDLE;
};

//struct SemaphoreDeleter{
//
//    inline void operator()(VkDevice device, VkSemaphore semaphore){
//        vkDestroySemaphore(device, semaphore, nullptr);
//    }
//};

#define VULKAN_RAII(Resource, ObjectType) \
struct Resource##Deleter{ \
    inline void operator()(VkDevice device, Vk##Resource resource){ \
        vkDestroy##Resource(device, resource, nullptr);          \
    }      \
};                       \
using Vulkan##Resource = VulkanHandle<Vk##Resource, Resource##Deleter, ObjectType>;

VULKAN_RAII(Pipeline, VK_OBJECT_TYPE_PIPELINE)
VULKAN_RAII(DescriptorSetLayout, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT)
VULKAN_RAII(ImageView, VK_OBJECT_TYPE_IMAGE_VIEW)
VULKAN_RAII(Sampler, VK_OBJECT_TYPE_SAMPLER)
VULKAN_RAII(PipelineCache, VK_OBJECT_TYPE_PIPELINE_CACHE)
VULKAN_RAII(PipelineLayout, VK_OBJECT_TYPE_PIPELINE_LAYOUT)
VULKAN_RAII(ShaderModule, VK_OBJECT_TYPE_SHADER_MODULE)
//VULKAN_RAII(QueryPool)