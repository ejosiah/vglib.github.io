#include <utility>

#include "fluid/FieldVisualizer.hpp"
#include "Barrier.hpp"
#include "GraphicsPipelineBuilder.hpp"
#include "Vertex.h"
#include "glsl_shaders.hpp"
#include "imgui.h"
#include "xforms.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace {
    constexpr auto scalarCellWidth = 16;

    std::string scalarCell(float value) {
        std::ostringstream valueStream;
        valueStream << std::fixed << std::setprecision(6) << value;

        std::ostringstream cellStream;
        cellStream << std::setw(scalarCellWidth) << valueStream.str();
        return cellStream.str();
    }
}

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
    updateProjection();
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

    Uniforms uniforms{};
    uniforms.gridSize = _gridSize;
    uniforms.domainMin = _domainMin;
    uniforms.domainSize = domainSize();
    usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    _streamLines.uniformBuffer = device->createCpuVisibleBuffer(&uniforms, sizeof(uniforms), usage);
    _streamLines.uniforms = reinterpret_cast<Uniforms*>(_streamLines.uniformBuffer.map());

    usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    _pressure.field = device->createBuffer(usage, VMA_MEMORY_USAGE_GPU_ONLY, N * sizeof(float), "visualize_pressure_field");
    _pressure.minValue = device->createBuffer(usage, VMA_MEMORY_USAGE_GPU_ONLY, sizeof(float), "min_pressure_value");
    _pressure.maxValue = device->createBuffer(usage, VMA_MEMORY_USAGE_GPU_ONLY, sizeof(float), "max_pressure_value");

    Globals globals{
        .gridSize = _gridSize,
        .dx = {1.0f / float(_gridSize.x), 0.0f},
        .dy = {0.0f, 1.0f / float(_gridSize.y)},
    };
    usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    _globals.buffer = device->createCpuVisibleBuffer(&globals, sizeof(globals), usage);
    _globals.data = reinterpret_cast<Globals*>(_globals.buffer.map());

    auto quad = domainQuadVertices();
    _screenQuad.vertices = device->createDeviceLocalBuffer(quad.data(), BYTE_SIZE(quad), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

    createVectorFieldResources();
}

void FieldVisualizer::createVectorFieldResources() {
    constexpr int interval = 5;
    const auto span = domainSize();
    const auto arrowScale = glm::min(span.x, span.y);
    const std::array<glm::vec2, 3> triangle{
        arrowScale * glm::vec2{0.0f, 0.2f},
        arrowScale * glm::vec2{1.0f, 0.0f},
        arrowScale * glm::vec2{0.0f, -0.2f},
    };

    std::vector<VectorArrow> arrows;
    for(auto y = interval / 2; y < _gridSize.y; y += interval) {
        for(auto x = interval / 2; x < _gridSize.x; x += interval) {
            const glm::vec2 uv{
                float(x) / float(_gridSize.x),
                float(y) / float(_gridSize.y)
            };
            const auto position = _domainMin + uv * span;

            for(const auto& vertex : triangle) {
                arrows.push_back(VectorArrow{vertex, position, uv});
            }
        }
    }

    _vectorField.numArrows = static_cast<uint32_t>(arrows.size());
    if(!arrows.empty()) {
        _vectorField.vertices = device->createDeviceLocalBuffer(arrows.data(), BYTE_SIZE(arrows), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    }

    textures::createNoTransition(*device, _vectorField.field, VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT,
                                 {uint32_t(_gridSize.x), uint32_t(_gridSize.y), 1u},
                                 VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    device->setName<VK_OBJECT_TYPE_IMAGE>("visualizer_combined_vector_field", _vectorField.field.image.image);

    device->firstActiveCommandPool().oneTimeCommand([&](auto commandBuffer) {
        Barriers::pushAndFlush(commandBuffer, _vectorField.field.image, DEFAULT_SUB_RANGE,
                               VK_PIPELINE_STAGE_2_NONE, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                               VK_ACCESS_2_NONE, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

        constexpr VkClearColorValue clearColor{};
        vkCmdClearColorImage(commandBuffer, _vectorField.field.image, VK_IMAGE_LAYOUT_GENERAL,
                             &clearColor, 1, &DEFAULT_SUB_RANGE);

        Barriers::pushAndFlush(commandBuffer, _vectorField.field.image, DEFAULT_SUB_RANGE,
                               VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                               VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);

        _vectorField.field.image.currentLayout = VK_IMAGE_LAYOUT_GENERAL;
    });
}

void FieldVisualizer::set(eular::FluidSolver* solver) {
    _solver = solver;
}

void FieldVisualizer::setDomain(const glm::vec2& max) {
    setDomain(glm::vec2{0.0f}, max);
}

void FieldVisualizer::setDomain(const glm::vec2& min, const glm::vec2& max) {
    _domainMin = min;
    _domainMax = max;
    updateProjection();

    if(_streamLines.uniforms) {
        _streamLines.uniforms->domainMin = _domainMin;
        _streamLines.uniforms->domainSize = domainSize();
    }
}

void FieldVisualizer::updateProjection() {
    _projection = vkn::ortho(_domainMin.x, _domainMax.x, _domainMin.y, _domainMax.y);
}

glm::vec2 FieldVisualizer::domainSize() const {
    return _domainMax - _domainMin;
}

std::array<glm::vec2, 8> FieldVisualizer::domainQuadVertices() const {
    return {
        glm::vec2{_domainMin.x, _domainMin.y}, glm::vec2{0.0f, 0.0f},
        glm::vec2{_domainMin.x, _domainMax.y}, glm::vec2{0.0f, 1.0f},
        glm::vec2{_domainMax.x, _domainMin.y}, glm::vec2{1.0f, 0.0f},
        glm::vec2{_domainMax.x, _domainMax.y}, glm::vec2{1.0f, 1.0f},
    };
}

void FieldVisualizer::setStreamLineColor(const glm::vec3 &streamColor) {
    _streamLines.color = streamColor;
}

void FieldVisualizer::update(VkCommandBuffer commandBuffer) {
    combineVectorFields(commandBuffer);
    computeStreamLines(commandBuffer);
    computeMinMaxPressure(commandBuffer);
}

void FieldVisualizer::initFieldDumpReadback(fs::path dumpDirectory) {
    const auto sampleCount = static_cast<VkDeviceSize>(_gridSize.x) * _gridSize.y;

    _fieldDump.vector = device->createStagingBuffer(sampleCount * sizeof(glm::vec4));
    _fieldDump.divergence = device->createStagingBuffer(sampleCount * sizeof(float));
    _fieldDump.pressure = device->createStagingBuffer(sampleCount * sizeof(float));

    _fieldDump.directory = dumpDirectory.empty()
        ? fs::current_path() / "field_visualizer_dumps"
        : std::move(dumpDirectory);
    fs::create_directories(_fieldDump.directory);
}

void FieldVisualizer::copyFieldDumpReadback(VkCommandBuffer commandBuffer) {
    if (!_fieldDump.vector) {
        initFieldDumpReadback();
    }

    copyTextureToDumpBuffer(commandBuffer, _vectorField.field, _fieldDump.vector);
    copyTextureToDumpBuffer(commandBuffer, _solver->_vectorGrid->divergenceField()[eular::in], _fieldDump.divergence);
    copyPressureToDumpBuffer(commandBuffer);

    _fieldDump.pendingStep = ++_fieldDump.step;
    _fieldDump.pending = true;
}

void FieldVisualizer::writePendingFieldDump() {
    if (!_fieldDump.pending) return;

    writeFieldDumpCsvFiles(_fieldDump.pendingStep);
    _fieldDump.pending = false;
    spdlog::info("Wrote fluid field dump {} to {}", _fieldDump.pendingStep, _fieldDump.directory.string());
}

void FieldVisualizer::renderStreamLines(VkCommandBuffer commandBuffer) {
    static std::array<VkDescriptorSet, 1> sets;
    sets[0] = _streamLines.descriptorSet;

    VkDeviceSize offset = 0;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _streamLines.pipeline.handle);
    vkCmdPushConstants(commandBuffer, _streamLines.layout.handle, VK_SHADER_STAGE_VERTEX_BIT,
                       0, sizeof(_projection), &_projection);
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
    vkCmdPushConstants(commandBuffer, _pressure.layout.handle, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(_projection), &_projection);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _pressure.layout.handle, 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
    vkCmdDraw(commandBuffer, 4, 1, 0, 0);
}

void FieldVisualizer::renderVectorField(VkCommandBuffer commandBuffer) {
    if(_vectorField.numArrows == 0) return;

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &_vectorField.vertices.buffer, &offset);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _vectorField.pipeline.handle);
    vkCmdPushConstants(commandBuffer, _vectorField.layout.handle, VK_SHADER_STAGE_VERTEX_BIT,
                       0, sizeof(_projection), &_projection);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _vectorField.layout.handle,
                            0, 1, &_vectorField.descriptorSet, 0, VK_NULL_HANDLE);

    vkCmdDraw(commandBuffer, _vectorField.numArrows, 1, 0, 0);
}

void FieldVisualizer::renderBoundary(VkCommandBuffer commandBuffer, glm::vec4 color, bool showColliders) {
    if(!_solver) return;

    _boundary.constants.transform = _projection;
    _boundary.constants.color = color;
    _boundary.constants.closedDomain = static_cast<uint32_t>(_solver->options.closedDomain);
    _boundary.constants.openBoundaryEdges = _solver->options.openBoundaryEdges;
    _boundary.constants.showColliders = static_cast<uint32_t>(showColliders);

    const auto set = _solver->colliderField().descriptorSet[eular::in];
    VkDeviceSize offset = 0;

    vkCmdBindVertexBuffers(commandBuffer, 0, 1, _screenQuad.vertices, &offset);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _boundary.pipeline.handle);
    vkCmdPushConstants(commandBuffer, _boundary.layout.handle, VK_SHADER_STAGE_ALL,
                       0, sizeof(_boundary.constants), &_boundary.constants);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _boundary.layout.handle,
                            0, 1, &set, 0, VK_NULL_HANDLE);
    vkCmdDraw(commandBuffer, 4, 1, 0, 0);
}

void FieldVisualizer::renderDebugFields(VkCommandBuffer commandBuffer) {
    if(!_solver) return;

    auto debugSets = _solver->debugFieldDescriptorSets();

    std::array<VkDescriptorSet, MaxDebugFields> sets{};
    auto fieldCount = 0u;
    for(auto set : debugSets) {
        if(fieldCount + 1 >= MaxDebugFields) break;
        sets[fieldCount++] = set;
    }

    sets[fieldCount++] = _debugFields.combinedVectorDescriptorSet;
    _debugFields.constants.fieldCount = fieldCount;
    _debugFields.constants.closedDomain = static_cast<uint32_t>(_solver->options.closedDomain);
    _debugFields.constants.openBoundaryEdges = _solver->options.openBoundaryEdges;

    for(auto i = fieldCount; i < MaxDebugFields; ++i) {
        sets[i] = _debugFields.combinedVectorDescriptorSet;
    }

    std::array<VkDescriptorSet, MaxDebugFields + 1> boundSets{};
    std::copy(sets.begin(), sets.end(), boundSets.begin());
    boundSets[MaxDebugFields] = _pressure.descriptorSet;

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, _screenQuad.vertices, &offset);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _debugFields.pipeline.handle);
    vkCmdPushConstants(commandBuffer, _debugFields.layout.handle, VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(_debugFields.constants), &_debugFields.constants);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _debugFields.layout.handle,
                            0, COUNT(boundSets), boundSets.data(), 0, VK_NULL_HANDLE);
    vkCmdDraw(commandBuffer, 4, 1, 0, 0);

    drawDebugFieldLabels(fieldCount);
}


void FieldVisualizer::createDescriptorSets() {
    _globals.setDescriptorSet =
        device->descriptorSetLayoutBuilder()
            .name("field_visualizer_globals")
            .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();

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

    _vectorField.setDescriptorSet =
        device->descriptorSetLayoutBuilder()
            .name("visualizer_vector_field")
            .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_VERTEX_BIT)
            .binding(2)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
        .createLayout();
}

void FieldVisualizer::updateDescriptorSets() {
    auto sets = _descriptorPool->allocate({
        _globals.setDescriptorSet,
        _streamLines.setDescriptorSet,
        _pressure.setDescriptorSet,
        _vectorField.setDescriptorSet,
        _fieldSetLayout
    });
    _globals.descriptorSet = sets[0];
    _streamLines.descriptorSet = sets[1];
    _pressure.descriptorSet = sets[2];
    _vectorField.descriptorSet = sets[3];
    _debugFields.combinedVectorDescriptorSet = sets[4];
    
    auto writes = initializers::writeDescriptorSets<9>();
    
    writes[0].dstSet = _globals.descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].descriptorCount = 1;
    VkDescriptorBufferInfo globalsInfo{ _globals.buffer, 0, VK_WHOLE_SIZE };
    writes[0].pBufferInfo = &globalsInfo;

    writes[1].dstSet = _streamLines.descriptorSet;
    writes[1].dstBinding = 0;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1;
    VkDescriptorBufferInfo linesInfo{ _streamLines.buffer, 0, VK_WHOLE_SIZE };
    writes[1].pBufferInfo = &linesInfo;

    writes[2].dstSet = _streamLines.descriptorSet;
    writes[2].dstBinding = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].descriptorCount = 1;
    VkDescriptorBufferInfo constInfo{ _streamLines.uniformBuffer, 0, VK_WHOLE_SIZE };
    writes[2].pBufferInfo = &constInfo;


    std::vector<VkDescriptorBufferInfo> minMaxInfo(2, {VK_NULL_HANDLE, 0, VK_WHOLE_SIZE});
    minMaxInfo[0].buffer = _pressure.minValue;
    minMaxInfo[1].buffer = _pressure.maxValue;

    writes[3].dstSet = _pressure.descriptorSet;
    writes[3].dstBinding = 0;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[3].descriptorCount = COUNT(minMaxInfo);
    writes[3].pBufferInfo = minMaxInfo.data();

    VkDescriptorImageInfo vectorTextureInfo{
        _vectorField.field.sampler.handle,
        _vectorField.field.imageView.handle,
        VK_IMAGE_LAYOUT_GENERAL
    };
    writes[4].dstSet = _vectorField.descriptorSet;
    writes[4].dstBinding = 0;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[4].descriptorCount = 1;
    writes[4].pImageInfo = &vectorTextureInfo;

    VkDescriptorImageInfo vectorStorageInfo{
        VK_NULL_HANDLE,
        _vectorField.field.imageView.handle,
        VK_IMAGE_LAYOUT_GENERAL
    };
    writes[5].dstSet = _vectorField.descriptorSet;
    writes[5].dstBinding = 2;
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[5].descriptorCount = 1;
    writes[5].pImageInfo = &vectorStorageInfo;

    VkDescriptorImageInfo debugCombinedTextureInfo{
        VK_NULL_HANDLE,
        _vectorField.field.imageView.handle,
        VK_IMAGE_LAYOUT_GENERAL
    };
    writes[6].dstSet = _debugFields.combinedVectorDescriptorSet;
    writes[6].dstBinding = 0;
    writes[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[6].descriptorCount = 1;
    writes[6].pImageInfo = &debugCombinedTextureInfo;

    writes[7].dstSet = _debugFields.combinedVectorDescriptorSet;
    writes[7].dstBinding = 1;
    writes[7].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    writes[7].descriptorCount = 1;
    writes[7].pImageInfo = &debugCombinedTextureInfo;

    writes[8].dstSet = _debugFields.combinedVectorDescriptorSet;
    writes[8].dstBinding = 2;
    writes[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[8].descriptorCount = 1;
    writes[8].pImageInfo = &debugCombinedTextureInfo;

    device->updateDescriptorSets(writes);

}

void FieldVisualizer::createRenderPipeline() {
    _streamLines.pipeline =
        device->graphicsPipelineBuilder()
            .shaderStage()
                .vertexShader(data_shaders_fluid_2d_stream_lines_vert)
                .fragmentShader(data_shaders_fluid_2d_stream_lines_frag)
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
                .cullNone()
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
                .addPushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(_projection))
            .renderPass(*_renderPass)
            .subpass(0)
            .name("stream_lines")
        .build(_streamLines.layout);

    _pressure.pipeline =
        device->graphicsPipelineBuilder()
            .shaderStage()
                .vertexShader(data_shaders_fluid_2d_visualizer_vert)
                .fragmentShader(data_shaders_fluid_2d_pressure_render_frag)
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
                .cullNone()
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
            .layout().clear()
                .addDescriptorSetLayout(_fieldSetLayout)
                .addDescriptorSetLayout(_pressure.setDescriptorSet)
                .addPushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(_projection))
            .renderPass(*_renderPass)
            .subpass(0)
            .name("pressure_field")
        .build(_pressure.layout);

    _vectorField.pipeline =
        device->graphicsPipelineBuilder()
            .shaderStage()
                .vertexShader(data_shaders_fluid_2d_vectorField_vert)
                .fragmentShader(data_shaders_fluid_2d_vectorField_frag)
            .vertexInputState()
                .addVertexBindingDescription(0, sizeof(VectorArrow), VK_VERTEX_INPUT_RATE_VERTEX)
                .addVertexAttributeDescription(0, 0, VK_FORMAT_R32G32_SFLOAT, offsetOf(VectorArrow, vertex))
                .addVertexAttributeDescription(1, 0, VK_FORMAT_R32G32_SFLOAT, offsetOf(VectorArrow, position))
                .addVertexAttributeDescription(2, 0, VK_FORMAT_R32G32_SFLOAT, offsetOf(VectorArrow, uv))
            .inputAssemblyState()
                .triangles()
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
                .cullNone()
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
                .addDescriptorSetLayout(_vectorField.setDescriptorSet)
                .addPushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(_projection))
            .renderPass(*_renderPass)
            .subpass(0)
            .name("visualizer_vector_field")
        .build(_vectorField.layout);

    std::vector<VulkanDescriptorSetLayout> debugLayouts(MaxDebugFields, _fieldSetLayout);
    _debugFields.pipeline =
        device->graphicsPipelineBuilder()
            .shaderStage()
                .vertexShader(data_shaders_quad_vert)
                .fragmentShader(data_shaders_fluid_2d_debug_fields_frag)
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
                .cullNone()
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
                .addDescriptorSetLayouts(debugLayouts)
                .addDescriptorSetLayout(_pressure.setDescriptorSet)
                .addPushConstantRange(VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(_debugFields.constants))
            .renderPass(*_renderPass)
            .subpass(0)
            .name("fluid_debug_fields")
        .build(_debugFields.layout);

    _boundary.pipeline =
        device->graphicsPipelineBuilder()
            .shaderStage()
                .vertexShader(data_shaders_fluid_2d_visualizer_vert)
                .fragmentShader(data_shaders_fluid_2d_boundary_render_frag)
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
                .cullFrontFace()
                .frontFaceCounterClockwise()
                .polygonModeFill()
            .multisampleState()
                .rasterizationSamples(VK_SAMPLE_COUNT_1_BIT)
            .depthStencilState()
                .disableDepthWrite()
                .disableDepthTest()
            .colorBlendState()
                .attachment()
                    .clear()
                    .enableBlend()
                    .srcColorBlendFactor().srcAlpha()
                    .dstColorBlendFactor().oneMinusSrcAlpha()
                    .srcAlphaBlendFactor().one()
                    .dstAlphaBlendFactor().oneMinusSrcAlpha()
                    .add()
            .layout()
                .addDescriptorSetLayout(_fieldSetLayout)
                .addPushConstantRange(VK_SHADER_STAGE_ALL, 0, sizeof(_boundary.constants))
            .renderPass(*_renderPass)
            .subpass(0)
            .name("fluid_boundary_overlay")
        .build(_boundary.layout);
}

std::vector<PipelineMetaData> FieldVisualizer::pipelineMetaData() {
    return {
            {
                    .name = "compute_stream_lines",
                    .shadePath = data_shaders_fluid_2d_compute_stream_lines_comp,
                    .layouts =  { &_streamLines.setDescriptorSet, &_fieldSetLayout, &_fieldSetLayout }
            },
            {
                    .name = "combine_vector_field",
                    .shadePath = data_shaders_fluid_2d_combine_vector_field_comp_comp,
                    .layouts =  { &_globals.setDescriptorSet, &_fieldSetLayout, &_fieldSetLayout, &_vectorField.setDescriptorSet },
                    .ranges = { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) } }
            },
    };
}

void FieldVisualizer::combineVectorFields(VkCommandBuffer commandBuffer) {
    static std::array<VkDescriptorSet, 4> sets;
    sets[0] = _globals.descriptorSet;
    sets[1] = _solver->vectorField().u.descriptorSet[0];
    sets[2] = _solver->vectorField().v.descriptorSet[0];
    sets[3] = _vectorField.descriptorSet;

    constexpr uint32_t combineAction = 0;
    const auto gc = glm::uvec2(glm::ceil(glm::vec2(_gridSize) / 32.0f));

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("combine_vector_field"));
    vkCmdPushConstants(commandBuffer, layout("combine_vector_field"), VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(combineAction), &combineAction);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("combine_vector_field"),
                            0, COUNT(sets), sets.data(), 0, 0);
    vkCmdDispatch(commandBuffer, gc.x, gc.y, 1);

    Barriers::pushAndFlush(commandBuffer, _vectorField.field.image, DEFAULT_SUB_RANGE,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                           VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
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

void FieldVisualizer::copyTextureToDumpBuffer(VkCommandBuffer commandBuffer, Texture& texture, VulkanBuffer& buffer) {
    auto oldLayout = texture.image.currentLayout;
    auto subresourceRange = DEFAULT_SUB_RANGE;
    subresourceRange.aspectMask = texture.aspectMask;
    constexpr auto shaderStages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                  VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;

    Barriers::pushAndFlush(commandBuffer, texture.image, subresourceRange,
                           shaderStages,
                           VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
                           VK_ACCESS_2_TRANSFER_READ_BIT,
                           oldLayout,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    texture.image.currentLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = texture.aspectMask;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {texture.width, texture.height, 1};

    vkCmdCopyImageToBuffer(commandBuffer, texture.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);

    Barriers::pushAndFlush(commandBuffer, texture.image, subresourceRange,
                           VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           shaderStages,
                           VK_ACCESS_2_TRANSFER_READ_BIT,
                           VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           oldLayout);
    texture.image.currentLayout = oldLayout;

    Barrier::transferWriteToHostRead(commandBuffer, buffer);
}

void FieldVisualizer::copyPressureToDumpBuffer(VkCommandBuffer commandBuffer) {
    Barrier::transferWriteToRead(commandBuffer, {_pressure.field});
    device->copy(commandBuffer, _pressure.field, _fieldDump.pressure, _fieldDump.pressure.size);
    Barrier::transferWriteToHostRead(commandBuffer, _fieldDump.pressure);
}

void FieldVisualizer::writeFieldDumpCsvFiles(uint32_t step) {
    const auto* vectorValues = reinterpret_cast<const glm::vec4*>(_fieldDump.vector.map());
    const auto* divergence = reinterpret_cast<const float*>(_fieldDump.divergence.map());
    const auto* pressure = reinterpret_cast<const float*>(_fieldDump.pressure.map());

    {
        std::ofstream out(_fieldDump.directory / fmt::format("u_{:04}.csv", step));
        writeVectorComponentGrid(out, vectorValues, 0);
    }

    {
        std::ofstream out(_fieldDump.directory / fmt::format("v_{:04}.csv", step));
        writeVectorComponentGrid(out, vectorValues, 1);
    }

    {
        std::ofstream out(_fieldDump.directory / fmt::format("divergence_{:04}.csv", step));
        writeScalarGrid(out, divergence);
    }

    {
        std::ofstream out(_fieldDump.directory / fmt::format("pressure_{:04}.csv", step));
        writeScalarGrid(out, pressure);
    }

    _fieldDump.vector.unmap();
    _fieldDump.divergence.unmap();
    _fieldDump.pressure.unmap();
}

void FieldVisualizer::writeVectorComponentGrid(std::ostream& out, const glm::vec4* vectorValues, uint32_t component) const {
    for (auto y = 0; y < _gridSize.y; ++y) {
        for (auto x = 0; x < _gridSize.x; ++x) {
            if (x > 0) out << ",";
            const auto index = static_cast<size_t>(y) * _gridSize.x + x;
            out << scalarCell(component == 0 ? vectorValues[index].x : vectorValues[index].y);
        }
        out << "\n";
    }
}

void FieldVisualizer::writeScalarGrid(std::ostream& out, const float* values) const {
    for (auto y = 0; y < _gridSize.y; ++y) {
        for (auto x = 0; x < _gridSize.x; ++x) {
            if (x > 0) out << ",";
            const auto index = static_cast<size_t>(y) * _gridSize.x + x;
            out << scalarCell(values[index]);
        }
        out << "\n";
    }
}

std::vector<std::string> FieldVisualizer::debugFieldLabels(uint32_t fieldCount) const {
    std::vector<std::string> labels{
        "Velocity U",
        "Velocity V",
        "Pressure",
        "Divergence",
        "Force",
        "Vorticity",
        "Boundary",
    };

    auto quantityIndex = 1u;
    for(const auto& quantityRef : _solver->_quantities) {
        if(labels.size() + 1 >= MaxDebugFields) break;

        const auto& quantity = quantityRef.get();
        const auto quantityName = quantity.name.empty()
            ? std::string{"Quantity "} + std::to_string(quantityIndex)
            : quantity.name;

        labels.push_back(quantityName);
        ++quantityIndex;

        if(labels.size() + 1 >= MaxDebugFields) break;
        labels.push_back(quantityName + " Source");
    }

    if(labels.size() < fieldCount) {
        labels.push_back("Velocity");
    }

    while(labels.size() < fieldCount) {
        labels.push_back("Field " + std::to_string(labels.size()));
    }

    if(labels.size() > fieldCount) {
        labels.resize(fieldCount);
    }

    return labels;
}

glm::vec4 FieldVisualizer::debugLabelColor(uint32_t index) const {
    static const std::array<glm::vec4, MaxDebugFields> colors{{
        {0.05f, 0.95f, 1.00f, 1.0f},
        {1.00f, 0.82f, 0.12f, 1.0f},
        {1.00f, 1.00f, 1.00f, 1.0f},
        {1.00f, 0.35f, 0.95f, 1.0f},
        {0.55f, 1.00f, 0.10f, 1.0f},
        {1.00f, 0.45f, 0.08f, 1.0f},
        {1.00f, 0.12f, 0.10f, 1.0f},
        {0.35f, 0.65f, 1.00f, 1.0f},
        {1.00f, 0.25f, 0.35f, 1.0f},
        {0.62f, 1.00f, 0.75f, 1.0f},
        {0.95f, 0.70f, 1.00f, 1.0f},
        {1.00f, 0.95f, 0.45f, 1.0f},
    }};

    return colors[index % colors.size()];
}

void FieldVisualizer::drawDebugFieldLabels(uint32_t fieldCount) const {
    if(fieldCount == 0 || ImGui::GetCurrentContext() == nullptr) return;

    const auto labels = debugFieldLabels(fieldCount);
    const auto columns = std::max(_debugFields.constants.columns, 1u);
    const auto rows = std::max(_debugFields.constants.rows, 1u);
    const auto tileWidth = static_cast<float>(_screenResolution.x) / static_cast<float>(columns);
    const auto tileHeight = static_cast<float>(_screenResolution.y) / static_cast<float>(rows);

    constexpr ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMouseInputs |
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoBackground |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoSavedSettings;

    ImGui::SetNextWindowPos({0.0f, 0.0f}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({static_cast<float>(_screenResolution.x), static_cast<float>(_screenResolution.y)}, ImGuiCond_Always);

    if(ImGui::Begin("Fluid Debug Field Labels", nullptr, flags)) {
        auto* drawList = ImGui::GetWindowDrawList();
        const auto origin = ImGui::GetWindowPos();
        const auto shadowColor = ImGui::ColorConvertFloat4ToU32(ImVec4{0.0f, 0.0f, 0.0f, 0.85f});
        const auto backingColor = ImGui::ColorConvertFloat4ToU32(ImVec4{0.0f, 0.0f, 0.0f, 0.58f});

        for(auto i = 0u; i < fieldCount && i < labels.size(); ++i) {
            const auto column = i % columns;
            const auto row = i / columns;
            const ImVec2 position{
                origin.x + static_cast<float>(column) * tileWidth + 10.0f,
                origin.y + static_cast<float>(row) * tileHeight + 8.0f
            };

            const auto textSize = ImGui::CalcTextSize(labels[i].c_str());
            const ImVec2 min{position.x - 5.0f, position.y - 3.0f};
            const ImVec2 max{position.x + textSize.x + 5.0f, position.y + textSize.y + 3.0f};
            drawList->AddRectFilled(min, max, backingColor, 3.0f);
            drawList->AddText({position.x + 1.0f, position.y + 1.0f}, shadowColor, labels[i].c_str());

            const auto color = debugLabelColor(i);
            drawList->AddText(position, ImGui::ColorConvertFloat4ToU32(ImVec4{color.r, color.g, color.b, color.a}), labels[i].c_str());
        }
    }
    ImGui::End();
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

    Barriers::pushAndFlush(commandBuffer,
                           VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("compute_stream_lines"));
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("compute_stream_lines"), 0, COUNT(sets), sets.data(), 0, 0);
    vkCmdDispatch(commandBuffer, gc.x, gc.y, 1);

    Barrier::computeWriteToVertexDraw(commandBuffer, {_streamLines.buffer});
}
