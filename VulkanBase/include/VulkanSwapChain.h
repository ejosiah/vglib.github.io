#pragma once

#include "common.h"
#include "VulkanDevice.h"
#include "VulkanSurface.h"
#include "Settings.hpp"

struct VulkanSwapChain{

    VulkanSwapChain() = default;

    inline VulkanSwapChain(const VulkanDevice& device, const VulkanSurface& surface, const Settings& settings,  VkSwapchainKHR oldSwapChain = VK_NULL_HANDLE)
    : preferredSurfaceFormat(settings.surfaceFormat)
    {
        auto capabilities = surface.getCapabilities(device);
        auto formats = surface.getFormats(device);
        auto presentModes = surface.getPresentModes(device);

      //  extent = capabilities
        VkSurfaceFormatKHR surfaceFormat = choose(formats);
        auto presentMode = choose(presentModes, settings.vSync);
        auto extent = chooseExtent(capabilities, {settings.width, settings.height});

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = std::min(capabilities.minImageCount + 1, capabilities.maxImageCount);
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        if(device.queueFamilyIndex.graphics == device.queueFamilyIndex.present) {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }else{
            std::vector<uint32_t> indices{ *device.queueFamilyIndex.graphics, *device.queueFamilyIndex.present };
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = indices.data();
        }
        createInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = true;
        createInfo.oldSwapchain = oldSwapChain;
        auto res = vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain);
//        ERR_GUARD_VULKAN(vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain));
        ERR_GUARD_VULKAN(res);

        this->extent = extent;
        this->format = surfaceFormat.format;
        this->device = const_cast<VulkanDevice*>(&device);
        this->images = get<VkImage>([&](uint32_t* count, VkImage* ptr){ vkGetSwapchainImagesKHR(device, swapChain, count, ptr); });
        createImageViews();

    }

    VulkanSwapChain(const VulkanSwapChain&) = delete;

    VulkanSwapChain(VulkanSwapChain&& source) noexcept {
        operator=(static_cast<VulkanSwapChain&&>(source));
    }

    VulkanSwapChain& operator=(const VulkanSwapChain&) = delete;

    VulkanSwapChain& operator=(VulkanSwapChain&& source) noexcept {
        this->swapChain = source.swapChain;
        this->format = source.format;
        this->extent = source.extent;
        this->device = source.device;
        this->images = std::move(source.images);
        this->imageViews = std::move(source.imageViews);

        source.swapChain = VK_NULL_HANDLE;

        return *this;
    }

    ~VulkanSwapChain(){
        if(swapChain){
            vkDestroySwapchainKHR(*device, swapChain, nullptr);
            for(auto& imageView : imageViews){
                vkDestroyImageView(*device, imageView, nullptr);
            }
        }
    }

    inline VkSurfaceFormatKHR choose(const std::vector<VkSurfaceFormatKHR>& formats) {
        auto itr = std::find_if(begin(formats), end(formats), [&](const auto& fmt){
           return fmt.format == preferredSurfaceFormat.format && fmt.colorSpace == preferredSurfaceFormat.colorSpace;
        });
        return itr != end(formats) ?  *itr : formats.front();
    }

    inline VkPresentModeKHR choose(const std::vector<VkPresentModeKHR>& presentModes, bool vSync) {
        if(vSync) return VK_PRESENT_MODE_FIFO_KHR;
        auto itr = std::find_if(begin(presentModes), end(presentModes), [](const auto& presentMode){
           return presentMode == VK_PRESENT_MODE_MAILBOX_KHR || presentMode == VK_PRESENT_MODE_IMMEDIATE_KHR;
        });
        return itr != end(presentModes) ? *itr : presentModes.front();
    }

    inline VkExtent2D chooseExtent(const VkSurfaceCapabilitiesKHR& capabilities, VkExtent2D actualExtent){
        if(capabilities.currentExtent.width != UINT32_MAX){
            return capabilities.currentExtent;
        }else{
            return {
                    std::clamp(actualExtent.width,
                               capabilities.minImageExtent.width,
                               capabilities.maxImageExtent.width),
                    std::clamp(actualExtent.height,
                               capabilities.minImageExtent.height,
                               capabilities.maxImageExtent.height)
            };
        }
    }

    inline void createImageViews(){
        imageViews.resize(images.size());
        for(int i = 0; i < images.size(); i++){
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = images[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = format;
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.layerCount = 1;
            createInfo.subresourceRange.levelCount = 1;

            ERR_GUARD_VULKAN(vkCreateImageView(*device, &createInfo, nullptr, &imageViews[i]));
        }
    }

    [[nodiscard]]
    uint32_t imageCount() const{
        return static_cast<uint32_t>(images.size());
    }

    template<typename T = uint32_t>
    [[nodiscard]]
    T width() const {
        return static_cast<T>(extent.width);
    }

    [[nodiscard]]
    float aspectRatio() const {
        return width<float>()/height<float>();
    };

    template<typename T = uint32_t>
    [[nodiscard]]
    T height() const {
        return static_cast<T>(extent.height);
    }

    operator VkSwapchainKHR() const {
        return swapChain;
    }

    operator VkSwapchainKHR*() {
        return &swapChain;
    }

    uint32_t acquireNextImage(VkSemaphore semaphore, VkFence fence = VK_NULL_HANDLE, uint64_t timeout = UINT64_MAX) const  {
        uint32_t imageIndex;
        state = vkAcquireNextImageKHR(*device, swapChain, timeout, semaphore, fence, &imageIndex);

        return imageIndex;
    }

    void present(uint32_t imageIndex, const std::vector<VkSemaphore>& waitSemaphores) const {
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = COUNT(waitSemaphores);
        presentInfo.pWaitSemaphores = waitSemaphores.data();
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapChain;
        presentInfo.pImageIndices = &imageIndex;

        state = vkQueuePresentKHR(device->queues.present, &presentInfo);
    }

    bool isOutOfDate() const {
        return state == VK_ERROR_OUT_OF_DATE_KHR;
    }

    bool isSubOptimal() const {
        return state == VK_SUBOPTIMAL_KHR;
    }

    VkImage getImage(uint32_t index) const {
        return images[index];
    }

    VkSwapchainKHR swapChain = VK_NULL_HANDLE;
    VulkanDevice* device = nullptr;
    VkSurfaceFormatKHR preferredSurfaceFormat{};
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{0, 0};
    std::vector<VkImage> images;
    std::vector<VkImageView> imageViews;

private:
    mutable VkResult state = VK_SUCCESS;
};