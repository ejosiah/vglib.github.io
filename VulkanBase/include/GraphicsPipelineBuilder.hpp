#pragma once

#include "common.h"
#include "Builder.hpp"
#include "VulkanDevice.h"
#include "VulkanRAII.h"
#include "VulkanInitializers.h"

class GraphicsPipelineBuilder : public Builder {
public:
    friend  class TessellationStateBuilder;
    explicit GraphicsPipelineBuilder(VulkanDevice* device);

    GraphicsPipelineBuilder(VulkanDevice* device, GraphicsPipelineBuilder* parent);

    GraphicsPipelineBuilder() = default;

    GraphicsPipelineBuilder(GraphicsPipelineBuilder&& source);

    virtual ~GraphicsPipelineBuilder();

    virtual ShaderStageBuilder& shaderStage();

    virtual VertexInputStateBuilder& vertexInputState();

    virtual InputAssemblyStateBuilder& inputAssemblyState();

    virtual TessellationStateBuilder& tessellationState();

    virtual ViewportStateBuilder& viewportState();

    virtual RasterizationStateBuilder& rasterizationState();

    virtual DepthStencilStateBuilder& depthStencilState();

    virtual ColorBlendStateBuilder& colorBlendState(void* next = nullptr);

    virtual MultisampleStateBuilder& multisampleState();

    virtual PipelineLayoutBuilder& layout();

    virtual DynamicStateBuilder& dynamicState();

    GraphicsPipelineBuilder& allowDerivatives();

    GraphicsPipelineBuilder& setDerivatives();

    GraphicsPipelineBuilder& subpass(uint32_t value);

    GraphicsPipelineBuilder& layout(VulkanPipelineLayout&  aLayout);

    GraphicsPipelineBuilder& renderPass(VkRenderPass  aRenderPass);

    DynamicRenderPassBuilder& dynamicRenderPass();

    GraphicsPipelineBuilder& name(const std::string& value);

    GraphicsPipelineBuilder& reuse();

    GraphicsPipelineBuilder& basePipeline(VulkanPipeline& pipeline);

    GraphicsPipelineBuilder& pipelineCache(VulkanPipelineCache pCache);

    [[nodiscard]]
    GraphicsPipelineBuilder *parent() override;

    VulkanPipeline build();

    VulkanPipeline build(VulkanPipelineLayout& pipelineLayout);

    VkGraphicsPipelineCreateInfo createInfo();

    [[nodiscard]]
    GraphicsPipelineBuilder clone() const;

    void copy(const GraphicsPipelineBuilder& source);

    [[nodiscard]]
    VulkanPipelineLayout pipelineLayout() const {
        return _pipelineLayoutOwned;
    }

protected:
    VkPipelineCreateFlags _flags = 0;
    VkRenderPass _renderPass = VK_NULL_HANDLE;
    VulkanPipelineLayout _pipelineLayout;
    VulkanPipelineLayout _pipelineLayoutOwned;
    uint32_t _subpass = 0;
    std::string _name;

    std::unique_ptr<ShaderStageBuilder> _shaderStageBuilder = nullptr;
    std::unique_ptr<VertexInputStateBuilder> _vertexInputStateBuilder = nullptr;
    std::unique_ptr<InputAssemblyStateBuilder> _inputAssemblyStateBuilder = nullptr;
    std::unique_ptr<PipelineLayoutBuilder> _pipelineLayoutBuilder = nullptr;
    std::unique_ptr<ViewportStateBuilder> _viewportStateBuilder = nullptr;
    std::unique_ptr<RasterizationStateBuilder> _rasterizationStateBuilder = nullptr;
    std::unique_ptr<MultisampleStateBuilder> _multisampleStateBuilder = nullptr;
    std::unique_ptr<DepthStencilStateBuilder> _depthStencilStateBuilder = nullptr;
    std::unique_ptr<ColorBlendStateBuilder> _colorBlendStateBuilder = nullptr ;
    std::unique_ptr<DynamicStateBuilder> _dynamicStateBuilder = nullptr;
    std::unique_ptr<TessellationStateBuilder> _tessellationStateBuilder = nullptr;
    std::unique_ptr<DynamicRenderPassBuilder> _dynamicRenderStateBuilder = nullptr;

    VulkanPipeline _basePipeline{};
    VulkanPipelineCache _pipelineCache{};
    void* nextChain = nullptr;

};

#include "ShaderStageBuilder.hpp"
#include "VertexInputStateBuilder.hpp"
#include "InputAssemblyStateBuilder.hpp"
#include "PipelineLayoutBuilder.hpp"
#include "ViewportStateBuilder.hpp"
#include "RasterizationStateBuilder.hpp"
#include "MultisampleStateBuilder.hpp"
#include "DepthStencilStateBuilder.hpp"
#include "ColorBlendStateBuilder.hpp"
#include "DynamicStateBuilder.hpp"
#include "TessellationStateBuilder.hpp"
#include "DynamicRenderPassBuilder.hpp"