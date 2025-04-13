#include <utility>

#include "fluid/FieldVisualizer.hpp"
#include "GraphicsPipelineBuilder.hpp"
#include "Vertex.h"
#include "glsl_shaders.hpp"

FieldVisualizer::FieldVisualizer(VulkanDevice *device, VulkanDescriptorPool* descriptorPool, 
                                 VulkanRenderPass* renderPass, VulkanDescriptorSetLayout fieldSetLayout,
                                 glm::uvec2 screenResolution, glm::ivec2 gridSize)
: ComputePipelines(device)
, _descriptorPool(descriptorPool)
, _renderPass(renderPass)
, _fieldSetLayout(std::move(fieldSetLayout))
, _screenResolution(screenResolution)
, _gridSize(gridSize)
{}

void FieldVisualizer::init() {
    createBuffers();
    initPrefixSum();
    createDescriptorSets();
    updateDescriptorSets();
    createPipelines();
    createRenderPipeline();
}

void FieldVisualizer::createBuffers() {
    const auto N = _gridSize.x * _gridSize.y;

    std::vector<glm::vec2> allocation(N * 4);
    auto usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    _streamLines.buffer = device->createDeviceLocalBuffer(allocation.data(), BYTE_SIZE(allocation), usage);

    Uniforms uniforms{ .gridSize = _gridSize };
    usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    _streamLines.uniformBuffer = device->createCpuVisibleBuffer(&uniforms, sizeof(uniforms), usage);
    _streamLines.uniforms = reinterpret_cast<Uniforms*>(_streamLines.uniformBuffer.map());

    usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    _pressure.field = device->createBuffer(usage, VMA_MEMORY_USAGE_GPU_ONLY, N * sizeof(float), "visualize_pressure_field");
    _pressure.minValue = device->createBuffer(usage, VMA_MEMORY_USAGE_GPU_ONLY, sizeof(float), "min_pressure_value");
    _pressure.maxValue = device->createBuffer(usage, VMA_MEMORY_USAGE_GPU_ONLY, sizeof(float), "max_pressure_value");

    auto quad = ClipSpace::Quad::positions;
    _screenQuad.vertices = device->createDeviceLocalBuffer(quad.data(), BYTE_SIZE(quad), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
}

void FieldVisualizer::set(eular::FluidSolver* solver) {
    _solver = solver;
}

void FieldVisualizer::setStreamLineColor(const glm::vec3 &streamColor) {
    _streamLines.color = streamColor;
}

void FieldVisualizer::update(VkCommandBuffer commandBuffer) {
    computeStreamLines(commandBuffer);
    computeMinMaxPressure(commandBuffer);
}

void FieldVisualizer::renderStreamLines(VkCommandBuffer commandBuffer) {
    static std::array<VkDescriptorSet, 1> sets;
    sets[0] = _streamLines.descriptorSet;

    VkDeviceSize offset = 0;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _streamLines.pipeline.handle);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _streamLines.layout.handle, 0, COUNT(sets), sets.data(), 0, 0);
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &_streamLines.buffer.buffer, &offset);
    vkCmdDraw(commandBuffer, _streamLines.uniforms->next_vertex, 1, 0, 0);
    _streamLines.uniforms->next_vertex = 0;
}

void FieldVisualizer::renderPressure(VkCommandBuffer commandBuffer) {
    VkDeviceSize offset = 0;

    static std::array<VkDescriptorSet, 2> sets;
    sets[0] = _solver->pressureField().descriptorSet[0];
    sets[1] = _pressure.descriptorSet;

    vkCmdBindVertexBuffers(commandBuffer, 0, 1, _screenQuad.vertices, &offset);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _pressure.pipeline.handle);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _pressure.layout.handle
            , 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
    vkCmdDraw(commandBuffer, 4, 1, 0, 0);
}


void FieldVisualizer::createDescriptorSets() {
    _streamLines.setDescriptorSet =
        device->descriptorSetLayoutBuilder()
            .name("stream_lines")
            .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT)
            .binding(1)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT)
        .createLayout();

    _pressure.setDescriptorSet =
        device->descriptorSetLayoutBuilder()
            .name("pressure_field")
            .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(2)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
        .createLayout();
}

void FieldVisualizer::updateDescriptorSets() {
    auto sets = _descriptorPool->allocate({ _streamLines.setDescriptorSet, _pressure.setDescriptorSet });
    _streamLines.descriptorSet = sets[0];
    _pressure.descriptorSet = sets[1];
    
    auto writes = initializers::writeDescriptorSets<3>();
    
    writes[0].dstSet = _streamLines.descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].descriptorCount = 1;
    VkDescriptorBufferInfo linesInfo{ _streamLines.buffer, 0, VK_WHOLE_SIZE };
    writes[0].pBufferInfo = &linesInfo;

    writes[1].dstSet = _streamLines.descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1;
    VkDescriptorBufferInfo constInfo{ _streamLines.uniformBuffer, 0, VK_WHOLE_SIZE };
    writes[1].pBufferInfo = &constInfo;


    std::vector<VkDescriptorBufferInfo> minMaxInfo(2, {VK_NULL_HANDLE, 0, VK_WHOLE_SIZE});
    minMaxInfo[0].buffer = _pressure.minValue;
    minMaxInfo[1].buffer = _pressure.maxValue;

    writes[2].dstSet = _pressure.descriptorSet;
    writes[2].dstBinding = 0;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].descriptorCount = COUNT(minMaxInfo);
    writes[2].pBufferInfo = minMaxInfo.data();

    device->updateDescriptorSets(writes);

}

void FieldVisualizer::createRenderPipeline() {
    _streamLines.pipeline =
        device->graphicsPipelineBuilder()
            .shaderStage()
                .vertexShader(R"(C:\Users\Josiah Ebhomenye\CLionProjects\vglib_examples\dependencies\vglib.github.io\data\shaders\fluid_2d\stream_lines.vert.spv)")
                .fragmentShader(R"(C:\Users\Josiah Ebhomenye\CLionProjects\vglib_examples\dependencies\vglib.github.io\data\shaders\fluid_2d\stream_lines.frag.spv)")
            .vertexInputState()
                .addVertexBindingDescription(0, sizeof(glm::vec2), VK_VERTEX_INPUT_RATE_VERTEX)
                .addVertexAttributeDescription(0, 0, VK_FORMAT_R32G32_SFLOAT, 0)
            .inputAssemblyState()
                .lines()
            .viewportState()
                .viewport()
                    .origin(0, 0)
                    .dimension(_screenResolution.x, _screenResolution.y)
                    .minDepth(0)
                    .maxDepth(1)
                .scissor()
                    .offset(0, 0)
                    .extent(_screenResolution.x, _screenResolution.y)
                .add()
            .rasterizationState()
                .cullBackFace()
                .frontFaceCounterClockwise()
                .polygonModeFill()
            .multisampleState()
                .rasterizationSamples(VK_SAMPLE_COUNT_1_BIT)
            .depthStencilState()
                .enableDepthWrite()
                .enableDepthTest()
                .compareOpAlways()
                .minDepthBounds(0)
                .maxDepthBounds(1)
            .colorBlendState()
                .attachment()
                .add()
            .layout()
                .addDescriptorSetLayout(_streamLines.setDescriptorSet)
            .renderPass(*_renderPass)
            .subpass(0)
            .name("stream_lines")
        .build(_streamLines.layout);

    _pressure.pipeline =
        device->graphicsPipelineBuilder()
            .shaderStage()
                .vertexShader(data_shaders_quad_vert)
                .fragmentShader(R"(C:\Users\Josiah Ebhomenye\CLionProjects\vglib_examples\dependencies\vglib.github.io\data\shaders\fluid_2d\pressure_render.frag.spv)")
            .vertexInputState()
                .addVertexBindingDescriptions(ClipSpace::bindingDescription())
                .addVertexAttributeDescriptions(ClipSpace::attributeDescriptions())
            .inputAssemblyState()
                .triangleStrip()
            .viewportState()
                .viewport()
                    .origin(0, 0)
                    .dimension(_screenResolution.x, _screenResolution.y)
                    .minDepth(0)
                    .maxDepth(1)
                .scissor()
                    .offset(0, 0)
                    .extent(_screenResolution.x, _screenResolution.y)
                .add()
            .rasterizationState()
                .cullBackFace()
                .frontFaceCounterClockwise()
                .polygonModeFill()
            .multisampleState()
                .rasterizationSamples(VK_SAMPLE_COUNT_1_BIT)
            .depthStencilState()
                .enableDepthWrite()
                .enableDepthTest()
                .compareOpAlways()
                .minDepthBounds(0)
                .maxDepthBounds(1)
            .colorBlendState()
                .attachment()
                .add()
            .layout()
                .addDescriptorSetLayout(_fieldSetLayout)
                .addDescriptorSetLayout(_pressure.setDescriptorSet)
            .renderPass(*_renderPass)
            .subpass(0)
            .name("pressure_field")
        .build(_pressure.layout);
}

std::vector<PipelineMetaData> FieldVisualizer::pipelineMetaData() {
    return {
            {
                    .name = "compute_stream_lines",
                    .shadePath = R"(C:\Users\Josiah Ebhomenye\CLionProjects\vglib_examples\dependencies\vglib.github.io\data\shaders\fluid_2d\compute_stream_lines.comp.spv)",
                    .layouts =  { &_streamLines.setDescriptorSet, &_fieldSetLayout, &_fieldSetLayout }
            },
    };
}

void FieldVisualizer::computeMinMaxPressure(VkCommandBuffer commandBuffer) {
    copyPressure(commandBuffer);
    _prefixSum.min(commandBuffer, _pressure.field.region(0), _pressure.minValue, DataType::Float);
    _prefixSum.max(commandBuffer, _pressure.field.region(0), _pressure.maxValue, DataType::Float);
    Barrier::computeWriteToFragmentRead(commandBuffer);
}

void FieldVisualizer::copyPressure(VkCommandBuffer commandBuffer) {
    _solver->pressureField()[0].image.copyToBuffer(commandBuffer, _pressure.field, VK_IMAGE_LAYOUT_GENERAL);
     Barrier::transferWriteToComputeRead(commandBuffer);
}

void FieldVisualizer::initPrefixSum() {
    _prefixSum = PrefixSum{device};
    _prefixSum.init();
}

void FieldVisualizer::computeStreamLines(VkCommandBuffer commandBuffer) {
    static std::array<VkDescriptorSet, 3> sets;
    sets[0] = _streamLines.descriptorSet;
    sets[1] = _solver->vectorField().u.descriptorSet[0];
    sets[2] = _solver->vectorField().v.descriptorSet[0];

    const auto offset = _streamLines.uniforms->offset;
    auto gc = glm::uvec2(_gridSize)/glm::max(1u, offset);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("compute_stream_lines"));
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("compute_stream_lines"), 0, COUNT(sets), sets.data(), 0, 0);
    vkCmdDispatch(commandBuffer, gc.x, gc.y, 1);
}

