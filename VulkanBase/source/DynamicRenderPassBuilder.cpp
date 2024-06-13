#include "DynamicRenderPassBuilder.hpp"

DynamicRenderPassBuilder::DynamicRenderPassBuilder(VulkanDevice *device, GraphicsPipelineBuilder *parent)
: GraphicsPipelineBuilder(device, parent) {}

DynamicRenderPassBuilder &DynamicRenderPassBuilder::viewMask(uint32_t value) {
    m_renderingCreateInfo.viewMask = value;
    return *this;
}

DynamicRenderPassBuilder &DynamicRenderPassBuilder::addColorAttachment(VkFormat format) {
    m_colorAttachments.push_back(format);
    return *this;
}

DynamicRenderPassBuilder &DynamicRenderPassBuilder::depthAttachment(VkFormat format) {
    m_renderingCreateInfo.depthAttachmentFormat = format;
    return *this;
}

DynamicRenderPassBuilder &DynamicRenderPassBuilder::stencilAttachment(VkFormat format) {
    m_renderingCreateInfo.stencilAttachmentFormat = format;
    return *this;
}

DynamicRenderPassBuilder &DynamicRenderPassBuilder::enable() {
    m_enabled = true;
    m_colorAttachments.clear();
    m_renderingCreateInfo = VkPipelineRenderingCreateInfo{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    return *this;
}

bool DynamicRenderPassBuilder::enabled() const {
    return m_enabled;
}

const VkPipelineRenderingCreateInfo& DynamicRenderPassBuilder::buildDynamicRenderInfo() {
    assert(!m_colorAttachments.empty() || m_renderingCreateInfo.depthAttachmentFormat != VK_FORMAT_UNDEFINED);
    m_renderingCreateInfo.colorAttachmentCount = m_colorAttachments.size();
    m_renderingCreateInfo.pColorAttachmentFormats = m_colorAttachments.data();
    return m_renderingCreateInfo;
}
