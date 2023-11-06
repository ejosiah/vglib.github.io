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
    , _pipelineBuilder(device.graphicsPipelineBuilder()) {
        initPipelineBuilderPrototype();
    }

    GraphicsPipelineBuilder cloneGraphicsPipeline() const {
        return _pipelineBuilder.clone();
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

private:
    VulkanDevice* _device;
    VulkanSwapChain* _swapchain;
    VkRenderPass _renderPass;
    GraphicsPipelineBuilder _pipelineBuilder;
};