#pragma once

#include "common.h"
#include "VulkanCopyable.h"
#include "VulkanCommandBuffer.h"
#include "VulkanRAII.h"
#include "VulkanBuffer.h"

static const VkImageSubresourceRange DEFAULT_SUB_RANGE{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

struct VulkanImage : public Copyable{
    
    DISABLE_COPY(VulkanImage)

    VulkanImage() = default;

    inline VulkanImage(VkDevice device, VmaAllocator allocator, VkImage image, VkFormat format, VmaAllocation allocation
                       , VkImageLayout layout, VkExtent3D dimension, uint32_t levels = 1u, uint32_t layers = 1u)
    : device(device)
    , allocator(allocator)
    , image(image)
    , format(format)
    , allocation(allocation)
    , currentLayout(layout)
    , dimension(dimension)
    , mipLevels(levels)
    , arrayLayers(layers)
    {}

    VulkanImage(VulkanImage&& source) noexcept {
        operator=(static_cast<VulkanImage&&>(source));
    }

    VulkanImage& operator=(VulkanImage&& source) noexcept {
        if(&source == this) return *this;

        this->~VulkanImage();

        device = source.device;
        allocator = source.allocator;
        image = source.image;
        allocation = source.allocation;
        currentLayout = source.currentLayout;
        dimension = source.dimension;
        format = source.format;

        source.allocator = VK_NULL_HANDLE;
        source.image = VK_NULL_HANDLE;
        source.allocation = VK_NULL_HANDLE;
        source.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        return *this;
    }

    ~VulkanImage(){
        if(image){
            vmaDestroyImage(allocator, image, allocation);
        }
    }

    operator VkImage() const {
        return image;
    }

    [[nodiscard]]
    VmaAllocator Allocator() const override {
        return allocator;
    }

    [[nodiscard]]
    VmaAllocation Allocation() const override {
        return allocation;
    }

    [[nodiscard]]
    glm::uvec3 getDimensions(){
        return {dimension.width, dimension.height, dimension.depth};
    }

    // TODO change from pool to commandBufffer
    void transitionLayout(const VulkanCommandPool& pool, VkImageLayout newLayout,
                          const VkImageSubresourceRange& subresourceRange = DEFAULT_SUB_RANGE) const {
        pool.oneTimeCommand([&](VkCommandBuffer commandBuffer) {

            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = currentLayout;
            barrier.newLayout = newLayout;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image;
            barrier.subresourceRange = subresourceRange;

            if(newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL){
                barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
                if(hasStencil(format)){
                    barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
                }
            }

            VkPipelineStageFlags sourceStage;
            VkPipelineStageFlags destinationStage;

            barrier.srcAccessMask = VK_ACCESS_NONE;
            if (currentLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

                sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            } else if (currentLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                       newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

                sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            }else if(currentLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL){
                barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

                sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            }else if(currentLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL){
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

                sourceStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
                destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            }else if((currentLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL || currentLayout == VK_IMAGE_LAYOUT_GENERAL)
                     && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL){
                barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

                sourceStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
                destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            }else if(currentLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
                    (newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL || newLayout == VK_IMAGE_LAYOUT_GENERAL)) {
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            }
            else {
                throw std::runtime_error{"unsupported layout transition!"};
            }

            vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
            currentLayout = newLayout;
        });
    }

    void transitionLayout(const VulkanCommandPool& pool, VkImageLayout newLayout, const VkImageSubresourceRange& subresourceRange,
                          VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask, VkPipelineStageFlags srcStageMask,
                          VkPipelineStageFlags dstStageMask) const  {

        pool.oneTimeCommand( [&](auto commandBuffer) {
            transitionLayout(commandBuffer, newLayout, subresourceRange, srcAccessMask, dstAccessMask, srcStageMask, dstStageMask);
        });
    }

    void transitionLayout(VkCommandBuffer commandBuffer, VkImageLayout newLayout, VkImageSubresourceRange subresourceRange,
                          VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask, VkPipelineStageFlags srcStageMask,
                          VkPipelineStageFlags dstStageMask) const {

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = srcAccessMask;
        barrier.dstAccessMask = dstAccessMask;
        barrier.oldLayout = currentLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = subresourceRange;

        vkCmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, 0,
                             0, nullptr, 0, nullptr, 1, &barrier);
        currentLayout = newLayout;
    }

    [[nodiscard]] VulkanImageView createView(VkFormat format, VkImageViewType viewType,
                                             const VkImageSubresourceRange& subresourceRange,
                                             VkImageViewCreateFlags flags = 0) const {
        VkImageViewCreateInfo  createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.viewType = viewType;
        createInfo.flags = flags;
        createInfo.image = image;
        createInfo.format = format;
        createInfo.subresourceRange = subresourceRange;

        VkImageView view;
        ERR_GUARD_VULKAN(vkCreateImageView(device, &createInfo, nullptr, &view));

        return VulkanImageView{ device, view };
    }
    
    void copyToBuffer(VkCommandBuffer commandBuffer, const VulkanBuffer& buffer, VkImageLayout oldLayout = VK_IMAGE_LAYOUT_UNDEFINED) const {
        oldLayout = oldLayout == VK_IMAGE_LAYOUT_UNDEFINED ? currentLayout : oldLayout;
        transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                , DEFAULT_SUB_RANGE, VK_ACCESS_SHADER_READ_BIT
                , VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                , VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkBufferImageCopy region{0, 0, 0
                                 , {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}
                                 , {0, 0, 0}, dimension};
        vkCmdCopyImageToBuffer(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);

        transitionLayout(commandBuffer, oldLayout
                , DEFAULT_SUB_RANGE, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT
                , VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    }

    void copyFromBuffer(VkCommandBuffer commandBuffer, const VulkanBuffer& buffer, VkImageLayout oldLayout = VK_IMAGE_LAYOUT_UNDEFINED) const {
        oldLayout = oldLayout == VK_IMAGE_LAYOUT_UNDEFINED ? currentLayout : oldLayout;
        transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                , DEFAULT_SUB_RANGE, VK_ACCESS_NONE
                , VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_NONE
                , VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkBufferImageCopy region{0, 0, 0
                , {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}
                , {0, 0, 0}, dimension};

        vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        transitionLayout(commandBuffer, oldLayout
                , DEFAULT_SUB_RANGE, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT
                , VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    }

    [[nodiscard]]
    inline VmaAllocationInfo allocationInfo() const {
        VmaAllocationInfo info;
        vmaGetAllocationInfo(allocator, allocation, &info);
        return info;
    }

    explicit operator bool() const {
        return image != VK_NULL_HANDLE;
    }

    VkDevice device = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VmaAllocation allocation = VK_NULL_HANDLE;
    mutable VkImageLayout currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkExtent3D dimension = { 0u, 0u, 0u };
    uint32_t  mipLevels{1};
    uint32_t arrayLayers{1};

    VkDeviceSize  size = 0;
};