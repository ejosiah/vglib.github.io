#pragma once

#include "VulkanDevice.h"
#include "VulkanBuffer.h"
#include "Texture.h"
#include <memory>

class VulkanImageOps {
    friend class ImageInfo;
public:
    VulkanImageOps(VulkanDevice *device = nullptr);

    ImageInfo &srcImage(VulkanImage &image);

    ImageInfo &dstImage(VulkanImage &texture);

    VulkanImageOps& srcBuffer(VulkanBuffer& buffer);

    VulkanImageOps& width(uint32_t w);

    VulkanImageOps& height(uint32_t h);

    VulkanImageOps& depth(uint32_t w);

    VulkanImageOps& extent(uint32_t width, uint32_t height, uint32_t depth = 1);

    VulkanImageOps& imageSubresourceRange(VkImageAspectFlags aspectMask,
                               uint32_t baseMipLevel,
                               uint32_t levelCount,
                               uint32_t baseArrayLayer,
                               uint32_t layerCount);

    virtual void copy(VkCommandBuffer commandBuffer);

    virtual void transfer(VkCommandBuffer commandBuffer);

protected:
    class Impl;
    Impl* pimpl{ };
    VulkanImageOps* _parent{};
};

class ImageInfo : public VulkanImageOps{
public:
    friend class Impl;
    ImageInfo(VulkanImageOps* parent, VulkanImage* image);

    ImageInfo& initialPipelineStage(VkPipelineStageFlags flags);

    ImageInfo& pipelineStage(VkPipelineStageFlags flags);

    ImageInfo& finalPipelineStage(VkPipelineStageFlags flags);

    ImageInfo& aspectMask(VkImageAspectFlags flags);

    ImageInfo& initialAspectMask(VkImageAspectFlags flags);

    ImageInfo& finalAspectMask(VkImageAspectFlags flags);


private:
    VulkanImage* _image{};
    VkPipelineStageFlags _initialPipelineStage{};
    VkPipelineStageFlags _finalPipelineStage{};
    VkImageAspectFlags _initialAspectMask{};
    VkImageAspectFlags _finalAspectMask{};
};