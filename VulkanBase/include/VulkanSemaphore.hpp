#pragma once

#include "common.h"
#include <vulkan/vulkan.h>

struct VulkanSemaphore{

    VulkanSemaphore() = default;

    VulkanSemaphore(VkDevice device, VkSemaphore semaphore)
            :device(device)
            , semaphore(semaphore)
    {

    }

    VulkanSemaphore(const VulkanSemaphore&) = delete;

    VulkanSemaphore(VulkanSemaphore&& source) noexcept {
        operator=(static_cast<VulkanSemaphore&&>(source));
    }

    ~VulkanSemaphore(){
        if(semaphore){
            vkDestroySemaphore(device, semaphore, VK_NULL_HANDLE);
            semaphore = VK_NULL_HANDLE;
        }
    }

    VulkanSemaphore& operator=(const VulkanSemaphore&) = delete;

    VulkanSemaphore& operator=(VulkanSemaphore&& source) noexcept {
        if(this == &source) return *this;

        if(semaphore){
            vkDestroySemaphore(device, semaphore, VK_NULL_HANDLE);
        }

        this->device = std::exchange(source.device, VK_NULL_HANDLE);
        this->semaphore = std::exchange(source.semaphore, VK_NULL_HANDLE);

        return *this;
    }

#ifdef WIN32
    [[nodiscard]]
     HANDLE getHandle() const {
        VkSemaphoreGetWin32HandleInfoKHR info{VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
        info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        info.semaphore = semaphore;

        HANDLE handle;
        vkGetSemaphoreWin32HandleKHR(device, &info, &handle);


        return handle;
    }
#else
    int getHandle() const {

        VkSemaphoreGetFdInfoKHR info{ VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
        fInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
        fInfo.semaphore = semaphore;

        int handle;
        vkGetSemaphoreFdKHR(device, &info, handle);

        return handle;
    }
#endif

    operator VkSemaphore() const {
        return semaphore;
    }

    operator VkSemaphore*()  {
        return &semaphore;
    }

    operator uint64_t() const {
        return (uint64_t)semaphore;
    }

    operator bool() const {
        return semaphore != VK_NULL_HANDLE;
    }

    VkDevice device = VK_NULL_HANDLE;
    VkSemaphore semaphore = VK_NULL_HANDLE;
};
