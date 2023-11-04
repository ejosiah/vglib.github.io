#pragma once

#include <vulkan/vulkan.h>
#include "RefCounted.hpp"



template<typename Handle, typename Deleter>
struct VulkanHandle : RefCounted {

    VulkanHandle() = default;

    VulkanHandle(VkDevice device, Handle handle)
    : RefCounted((ResourceHandle)handle, [&](ResourceHandle){ Deleter()(device, handle); })
    , device(device)
    , handle(handle)
    {

    }

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

    operator Handle() const {
        return handle;
    }

    operator Handle*()  {
        return &handle;
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

#define VULKAN_RAII(Resource) \
struct Resource##Deleter{ \
    inline void operator()(VkDevice device, Vk##Resource resource){ \
        vkDestroy##Resource(device, resource, nullptr);          \
    }      \
};                       \
using Vulkan##Resource = VulkanHandle<Vk##Resource, Resource##Deleter>;

VULKAN_RAII(Pipeline)
VULKAN_RAII(DescriptorSetLayout)
VULKAN_RAII(ImageView)
VULKAN_RAII(Sampler)
VULKAN_RAII(PipelineCache)
//VULKAN_RAII(QueryPool)