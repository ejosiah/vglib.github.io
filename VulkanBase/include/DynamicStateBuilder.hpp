#pragma once

#include "GraphicsPipelineBuilder.hpp"

class DynamicStateBuilder : public GraphicsPipelineBuilder{
public:
    DynamicStateBuilder(VulkanDevice* device, GraphicsPipelineBuilder* parent);

    DynamicStateBuilder& viewport();

    DynamicStateBuilder& scissor();

    DynamicStateBuilder& lineWidth();

    DynamicStateBuilder& depthBias();

    DynamicStateBuilder& blendConstants();

    DynamicStateBuilder& depthBounds();

    DynamicStateBuilder& stencilCompareMask();

    DynamicStateBuilder& stencilWriteMask();

    DynamicStateBuilder& stencilReferenceMask();

    DynamicStateBuilder& cullMode();

    DynamicStateBuilder& frontFace();

    DynamicStateBuilder& primitiveTopology();

    DynamicStateBuilder& viewportWithCount();

    DynamicStateBuilder& scissorWithCount();

    DynamicStateBuilder& vertexInputBindingStride();

    DynamicStateBuilder& depthTestEnable();

    DynamicStateBuilder& depthWriteEnable();

    DynamicStateBuilder& depthCompareOp();

    DynamicStateBuilder& DepthBoundsTestEnable();

    DynamicStateBuilder& stencilTestEnable();

    DynamicStateBuilder& stencilOp();

    DynamicStateBuilder& rasterDiscardEnable();

    DynamicStateBuilder& depthBiasEnable();

    DynamicStateBuilder& primitiveRestartEnable();

    DynamicStateBuilder& colorWriteEnable();

    DynamicStateBuilder& clear();

    VkPipelineDynamicStateCreateInfo& buildPipelineDynamicState();

private:
    std::vector<VkDynamicState> _dynamicStates;
    VkPipelineDynamicStateCreateInfo _info;
};