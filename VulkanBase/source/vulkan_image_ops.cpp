#include "vulkan_image_ops.h"

#include <utility>

class VulkanImageOps::Impl{
public:
    Impl(VulkanImageOps* ops = nullptr, VulkanDevice* device = nullptr)
    : _ops{ ops }
    , _device{ device }
    {}

    ImageInfo &srcImage(VulkanImage &image){
        _srcImageInfo = std::make_unique<ImageInfo>(_ops, &image);
//        _srcImage = &image;
        return *_srcImageInfo;
    }

    ImageInfo &dstImage(VulkanImage &image){
        _dstImageInfo = std::make_unique<ImageInfo>(_ops, &image);
//        _dstImage = &image;
        return *_dstImageInfo;
    }

    VulkanImageOps& srcBuffer(VulkanBuffer& buffer){
        _srcBuffer = &buffer;
        return *_ops;
    }

    VulkanImageOps& imageSubresourceRange(VkImageAspectFlags aspectMask,
                                          uint32_t baseMipLevel,
                                          uint32_t levelCount,
                                          uint32_t baseArrayLayer,
                                          uint32_t layerCount){
        _subresourceRange.aspectMask = aspectMask;
        _subresourceRange.baseMipLevel = baseMipLevel;
        _subresourceRange.levelCount = levelCount;
        _subresourceRange.baseArrayLayer = baseArrayLayer;
        _subresourceRange.layerCount = layerCount;

        return *_ops;
    }

    VulkanImageOps& currentPipelineStage(VkPipelineStageFlags flags){
        _srcImageInfo->initialPipelineStage(flags);
//        _currentPipelineStage = flags;
        return *_ops;
    }

    VulkanImageOps& srcPipelineStage(VkPipelineStageFlags flags){
        _dstImageInfo->initialPipelineStage(flags);
//        _srcPipelineStage = flags;
        return *_ops;
    }

    VulkanImageOps& dstPipelineStage(VkPipelineStageFlags flags){
//        _dstPipelineStage = flags;
        return *_ops;
    }

    VulkanImageOps& currentAccessMask(VkAccessFlags flags){
        _srcImageInfo->initialAspectMask(flags);
//        _currentAccessMask = flags;
        return *_ops;
    }
    VulkanImageOps& srcAccessMask(VkAccessFlags flags){
        _dstImageInfo->initialAspectMask(flags);
//        _srcAccessMask = flags;
        return *_ops;
    }

    VulkanImageOps& dstAccessMask(VkAccessFlags flags){
//        _dstAccessMask = flags;
        return *_ops;
    }

    VulkanImageOps& width(uint32_t w){
        _extent.width = w;
        return *_ops;
    }

    VulkanImageOps& height(uint32_t h){
        _extent.height = h;
        return *_ops;
    }

    VulkanImageOps& depth(uint32_t d){
        _extent.depth = d;
        return *_ops;
    }

    VulkanImageOps& extent(uint32_t width, uint32_t height, uint32_t depth = 0){
        _extent.width = width;
        _extent.height = height;
        _extent.depth = depth;
        return *_ops;
    }

    void copy(VkCommandBuffer commandBuffer){
        assert(_srcImageInfo);
        assert(_dstImageInfo);
        assert(_srcImageInfo->_initialPipelineStage != VK_PIPELINE_STAGE_NONE);
        assert(_dstImageInfo->_initialPipelineStage != VK_PIPELINE_STAGE_NONE);
        assert(_srcImageInfo->_initialAspectMask != VK_ACCESS_NONE);
        assert(_dstImageInfo->_initialAspectMask != VK_ACCESS_NONE);
        assert(_extent.width != 0 && _extent.height != 0);

        _dstImageInfo->_finalPipelineStage = (_dstImageInfo->_finalPipelineStage == VK_PIPELINE_STAGE_NONE) ? _dstImageInfo->_initialPipelineStage : _dstImageInfo->_finalPipelineStage;
        _dstImageInfo->_finalAspectMask = (_dstImageInfo->_finalAspectMask == VK_ACCESS_NONE) ? _dstImageInfo->_initialAspectMask : _dstImageInfo->_finalAspectMask;

        auto _dstImage = _dstImageInfo->_image;
        auto _srcAccessMask = _dstImageInfo->_initialAspectMask;
        auto _srcPipelineStage = _dstImageInfo->_initialPipelineStage;
        auto dstOldLayout = _dstImage->currentLayout;
        if(dstOldLayout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL){
            _dstImage->transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, _subresourceRange
                    , _srcAccessMask, VK_ACCESS_TRANSFER_WRITE_BIT
                    , _srcPipelineStage, VK_PIPELINE_STAGE_TRANSFER_BIT);
        }

        auto _srcImage = _srcImageInfo->_image;
        auto _currentAccessMask = _srcImageInfo->_initialAspectMask;
        auto _currentPipelineStage = _srcImageInfo->_initialPipelineStage;
        auto srcOldLayout = _srcImage->currentLayout;
        if(srcOldLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL){
            _srcImage->transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, _subresourceRange
                    , _currentAccessMask, VK_ACCESS_TRANSFER_READ_BIT
                    , _currentPipelineStage, VK_PIPELINE_STAGE_TRANSFER_BIT);
        }

        VkImageSubresourceLayers imageSubresource{_subresourceRange.aspectMask, _subresourceRange.baseMipLevel,
                                                  _subresourceRange.baseArrayLayer, _subresourceRange.layerCount};

        VkImageCopy region{};
        region.srcSubresource = imageSubresource;
        region.srcOffset = {0, 0, 0};
        region.dstSubresource = imageSubresource;
        region.dstOffset = {0, 0, 0};
        region.extent = _extent;

        vkCmdCopyImage(commandBuffer, _srcImage->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, _dstImage->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);


        auto _dstAccessMask = _dstImageInfo->_finalAspectMask;
        auto _dstPipelineStage = _dstImageInfo->_finalPipelineStage;
        if(dstOldLayout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL){
            _dstImage->transitionLayout(commandBuffer, dstOldLayout, _subresourceRange
                    , VK_ACCESS_TRANSFER_WRITE_BIT, _dstAccessMask
                    , VK_PIPELINE_STAGE_TRANSFER_BIT, _dstPipelineStage);
        }

        if(srcOldLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL){
            _srcImage->transitionLayout(commandBuffer, srcOldLayout, _subresourceRange
                    , VK_ACCESS_TRANSFER_WRITE_BIT, _currentAccessMask
                    , VK_PIPELINE_STAGE_TRANSFER_BIT, _currentPipelineStage);
        }
    }

    void transfer(VkCommandBuffer commandBuffer){

    }

private:
    VulkanDevice *_device = nullptr;
    VulkanImageOps* _ops = nullptr;
//    VulkanImage *_srcImage = nullptr;
//    VulkanImage *_dstImage = nullptr;
    VulkanBuffer* _srcBuffer = nullptr;
    VkExtent3D _extent;
//    VkPipelineStageFlags _srcPipelineStage{VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
//    VkPipelineStageFlags _dstPipelineStage{VK_PIPELINE_STAGE_NONE};
//    VkAccessFlags _currentAccessMask{VK_ACCESS_NONE};
//    VkAccessFlags _srcAccessMask{VK_ACCESS_NONE};
//    VkAccessFlags _dstAccessMask{VK_ACCESS_NONE};
    VkImageSubresourceRange _subresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
//    VkPipelineStageFlags _currentPipelineStage{VK_PIPELINE_STAGE_NONE};
    std::unique_ptr<ImageInfo> _srcImageInfo;
    std::unique_ptr<ImageInfo> _dstImageInfo;

};

VulkanImageOps::VulkanImageOps(VulkanDevice *device)
: pimpl{new Impl{ this, device }}
{

}

ImageInfo &VulkanImageOps::srcImage(VulkanImage &image) {
    if(_parent){
        return _parent->srcImage(image);
    }
    return pimpl->srcImage(image);
}

ImageInfo &VulkanImageOps::dstImage(VulkanImage &image) {
    if(_parent){
        return _parent->dstImage(image);
    }
    return pimpl->dstImage(image);
}

VulkanImageOps &VulkanImageOps::srcBuffer(VulkanBuffer& buffer) {
    if(_parent){
        return _parent->srcBuffer(buffer);
    }
    return pimpl->srcBuffer(buffer);
}

VulkanImageOps &
VulkanImageOps::imageSubresourceRange(VkImageAspectFlags aspectMask, uint32_t baseMipLevel, uint32_t levelCount,
                                      uint32_t baseArrayLayer, uint32_t layerCount) {
    return pimpl->imageSubresourceRange(aspectMask, baseMipLevel, levelCount, baseMipLevel, layerCount);
}

void VulkanImageOps::copy(VkCommandBuffer commandBuffer) {
    if(_parent){
        _parent->copy(commandBuffer);
    }else {
        pimpl->copy(commandBuffer);
    }
}

void VulkanImageOps::transfer(VkCommandBuffer commandBuffer) {
    if(_parent){
        _parent->transfer(commandBuffer);
    }else {
        pimpl->transfer(commandBuffer);
    }
}

VulkanImageOps &VulkanImageOps::width(uint32_t w) {
    if(_parent){
        return _parent->width(w);
    }
    return pimpl->width(w);
}

VulkanImageOps &VulkanImageOps::height(uint32_t h) {
    if(_parent){
        return _parent->height(h);
    }
    return pimpl->height(h);
}

VulkanImageOps &VulkanImageOps::depth(uint32_t d) {
    if(_parent){
        return _parent->depth(d);
    }
    return pimpl->depth(d);
}

VulkanImageOps &VulkanImageOps::extent(uint32_t width, uint32_t height, uint32_t depth) {
    if(_parent){
        return _parent->extent(width, height, depth);
    }
    return pimpl->extent(width, height, depth);
}


VulkanImageOps VulkanDevice::imageOps() {
    return VulkanImageOps{this};
}

ImageInfo::ImageInfo(VulkanImageOps *parent, VulkanImage *image) : _image{ image }{
    _parent = parent;
}

ImageInfo &ImageInfo::initialPipelineStage(VkPipelineStageFlags flags) {
    _initialPipelineStage = flags;
    return *this;
}

ImageInfo &ImageInfo::finalPipelineStage(VkPipelineStageFlags flags) {
    _finalPipelineStage = flags;
    return *this;
}

ImageInfo &ImageInfo::initialAspectMask(VkImageAspectFlags flags) {
    _initialAspectMask = flags;
    return *this;
}

ImageInfo &ImageInfo::finalAspectMask(VkImageAspectFlags flags) {
    _finalAspectMask = flags;
    return *this;
}

ImageInfo &ImageInfo::pipelineStage(VkPipelineStageFlags flags) {
    _initialPipelineStage = _finalPipelineStage = flags;
    return *this;
}

ImageInfo &ImageInfo::aspectMask(VkImageAspectFlags flags) {
    _initialAspectMask = _finalAspectMask = flags;
    return *this;
}
