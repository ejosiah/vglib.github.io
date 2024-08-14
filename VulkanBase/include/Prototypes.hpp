#pragma once

#include "VulkanDevice.h"
#include "GraphicsPipelineBuilder.hpp"
#include "Vertex.h"
#include "Camera.h"
#include "VulkanSwapChain.h"

class Prototypes {
public:
    Prototypes(VulkanDevice& device, VulkanSwapChain& _swapchain, VkRenderPass renderPass = VK_NULL_HANDLE)
    : _device(&device)
    , _swapchain(&_swapchain)
    , _renderPass(renderPass)
    , _pipelineBuilder(device.graphicsPipelineBuilder())
    , _screenQuadPipelineBuilder(device.graphicsPipelineBuilder()) {
        initPipelineBuilderPrototype();
        initScreenQuadPipelineBuilderPrototype();
    }

    GraphicsPipelineBuilder cloneGraphicsPipeline() const {
        return _pipelineBuilder.clone();
    }

    GraphicsPipelineBuilder cloneScreenSpaceGraphicsPipeline() const {
        return _screenQuadPipelineBuilder.clone();
    }

protected:
    void initPipelineBuilderPrototype() {
		_pipelineBuilder
            .allowDerivatives()
            .vertexInputState()
                .addVertexBindingDescriptions(Vertex::bindingDisc())
                .addVertexAttributeDescriptions(Vertex::attributeDisc())
            .inputAssemblyState()
                .triangles()
            .viewportState()
                .viewport()
                    .origin(0, 0)
                    .dimension(_swapchain->extent)
                    .minDepth(0)
                    .maxDepth(1)
                .scissor()
                    .offset(0, 0)
                    .extent(_swapchain->extent)
                .add()
                .rasterizationState()
                    .cullBackFace()
                    .frontFaceCounterClockwise()
                    .polygonModeFill()
                .multisampleState()
                    .rasterizationSamples(_device->settings.msaaSamples)
                .depthStencilState()
                    .enableDepthWrite()
                    .enableDepthTest()
                    .compareOpLess()
                    .minDepthBounds(0)
                    .maxDepthBounds(1)
                .colorBlendState()
                    .attachment()
                    .add()
                .layout()
                    .addPushConstantRange(Camera::pushConstant())
                .renderPass(_renderPass)
                .subpass(0)
                .name("gp_prototype");
    }

    void initScreenQuadPipelineBuilderPrototype() {
        _screenQuadPipelineBuilder
            .allowDerivatives()
            .vertexInputState()
                .addVertexBindingDescriptions(ClipSpace::bindingDescription())
                .addVertexAttributeDescriptions(ClipSpace::attributeDescriptions())
            .inputAssemblyState()
                .triangleStrip()
            .viewportState()
                .viewport()
                    .origin(0, 0)
                    .dimension(_swapchain->extent)
                    .minDepth(0)
                    .maxDepth(1)
                .scissor()
                    .offset(0, 0)
                    .extent(_swapchain->extent)
                .add()
                .rasterizationState()
                    .cullNone()
                    .frontFaceCounterClockwise()
                    .polygonModeFill()
                .multisampleState()
                    .rasterizationSamples(_device->settings.msaaSamples)
                .depthStencilState()
                    .enableDepthWrite()
                    .enableDepthTest()
                    .compareOpLess()
                    .minDepthBounds(0)
                    .maxDepthBounds(1)
                .colorBlendState()
                    .attachment()
                    .add()
                .renderPass(_renderPass)
                .subpass(0)
                .name("clip_gp_prototype");
    }

private:
    VulkanDevice* _device;
    VulkanSwapChain* _swapchain;
    VkRenderPass _renderPass;
    GraphicsPipelineBuilder _pipelineBuilder;
    GraphicsPipelineBuilder _screenQuadPipelineBuilder;
};