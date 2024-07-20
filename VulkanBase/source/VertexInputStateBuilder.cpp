#include "VertexInputStateBuilder.hpp"
#include <spdlog/spdlog.h>

VertexInputStateBuilder::VertexInputStateBuilder(VulkanDevice *device, GraphicsPipelineBuilder *parent)
        : GraphicsPipelineBuilder(device, parent) {

}

VertexInputStateBuilder &
VertexInputStateBuilder::addVertexBindingDescription(uint32_t binding, uint32_t stride, VkVertexInputRate inputRate) {
    _bindings.push_back({binding, stride, inputRate});
    return *this;
}

VertexInputStateBuilder &
VertexInputStateBuilder::addVertexBindingDescription(const VkVertexInputBindingDescription &description) {
    _bindings.push_back(description);
    return *this;
}

VertexInputStateBuilder &
VertexInputStateBuilder::addVertexAttributeDescription(uint32_t location, uint32_t binding, VkFormat format,
                                                       uint32_t offset) {
    _attributes.push_back({location, binding, format, offset});
    return *this;
}

VertexInputStateBuilder &
VertexInputStateBuilder::addVertexAttributeDescription(const VkVertexInputAttributeDescription &description) {
    _attributes.push_back(description);
    return *this;
}

void VertexInputStateBuilder::validate() const {
    if(_bindings.empty()){
        spdlog::warn("No vertex binding descriptions defined for vertexInputState");
    }

    for(const auto& binding : _bindings){
        auto itr = std::find_if(begin(_attributes), end(_attributes), [&binding](const auto& attribute){
            return binding.binding == attribute.binding;
        });
        if(itr == end(_attributes)){
            throw std::runtime_error{fmt::format("No vertex attribute description defined for binding {}", binding.binding)};
        }
    }
}

VkPipelineVertexInputStateCreateInfo& VertexInputStateBuilder::buildVertexInputState() {
    validate();
    _info = initializers::vertexInputState(_bindings, _attributes);
    return _info;
}

VertexInputStateBuilder& VertexInputStateBuilder::clear() {
    _bindings.clear();
    _attributes.clear();
    return *this;
}

VertexInputStateBuilder &VertexInputStateBuilder::clearBindingDesc() {
    _bindings.clear();
    return *this;
}

VertexInputStateBuilder &VertexInputStateBuilder::clearAttributeDesc() {
    _attributes.clear();
    return *this;
}


void VertexInputStateBuilder::copy(const VertexInputStateBuilder& source) {
    _bindings = std::vector<VkVertexInputBindingDescription>(source._bindings.begin(), source._bindings.end());
    _attributes = std::vector<VkVertexInputAttributeDescription>(source._attributes.begin(), source._attributes.end());
}