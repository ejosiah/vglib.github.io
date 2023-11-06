#include "PipelineLayoutBuilder.hpp"

PipelineLayoutBuilder::PipelineLayoutBuilder(VulkanDevice *device, GraphicsPipelineBuilder *builder)
        : GraphicsPipelineBuilder(device, builder) {

}

PipelineLayoutBuilder &PipelineLayoutBuilder::addDescriptorSetLayout(VulkanDescriptorSetLayout layout) {
    _descriptorSetLayouts.push_back(layout);
    return *this;
}

PipelineLayoutBuilder &
PipelineLayoutBuilder::addPushConstantRange(VkShaderStageFlags stage, uint32_t offset, uint32_t size) {
    _ranges.push_back({stage, offset, size});
    return *this;
}

VulkanPipelineLayout PipelineLayoutBuilder::buildPipelineLayout() const {
    return device().createPipelineLayout(_descriptorSetLayouts, _ranges);
}

PipelineLayoutBuilder &PipelineLayoutBuilder::addPushConstantRange(VkPushConstantRange range) {
    _ranges.push_back(range);
    return *this;
}

PipelineLayoutBuilder &PipelineLayoutBuilder::clearRanges() {
    _ranges.clear();
    return *this;
}

PipelineLayoutBuilder &PipelineLayoutBuilder::clearLayouts() {
    _descriptorSetLayouts.clear();
    return *this;
}

PipelineLayoutBuilder &PipelineLayoutBuilder::clear() {
    _ranges.clear();
    _descriptorSetLayouts.clear();
    return *this;
}


void PipelineLayoutBuilder::copy(const PipelineLayoutBuilder& source) {
    _ranges = decltype(_ranges)(source._ranges.begin(), source._ranges.end());
    _descriptorSetLayouts = decltype(_descriptorSetLayouts)(source._descriptorSetLayouts.begin(), source._descriptorSetLayouts.end());

}
