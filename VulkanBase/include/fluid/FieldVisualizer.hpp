#pragma once

#include "FluidSolver2.hpp"
#include "ComputePipelins.hpp"
#include "filemanager.hpp"
#include "PrefixSum.hpp"

#include <iosfwd>

class FieldVisualizer : ComputePipelines {
public:
    FieldVisualizer() = default;

    FieldVisualizer(VulkanDevice* device, VulkanDescriptorPool* descriptorPool,
                    VulkanRenderPass* renderPass, VulkanDescriptorSetLayout fieldSetLayout,
                    glm::uvec2 screenResolution, glm::ivec2 gridSize);

    void init();

    void set(eular::FluidSolver* solver);

    void setStreamLineColor(const glm::vec3& streamColor);

    void update(VkCommandBuffer commandBuffer);

    void initFieldDumpReadback(fs::path dumpDirectory = {});

    void copyFieldDumpReadback(VkCommandBuffer commandBuffer);

    void writePendingFieldDump();

    void renderStreamLines(VkCommandBuffer commandBuffer);

    void renderPressure(VkCommandBuffer commandBuffer);

    void renderVectorField(VkCommandBuffer commandBuffer);

    void renderBoundary(VkCommandBuffer commandBuffer,
                        glm::vec4 color = glm::vec4{1.0f, 0.0f, 0.0f, 0.85f},
                        bool showColliders = false);

    void renderDebugFields(VkCommandBuffer commandBuffer);

protected:
    std::vector<PipelineMetaData> pipelineMetaData() override;

private:
    void initPrefixSum();
    void createBuffers();
    void createVectorFieldResources();
    void createDescriptorSets();
    void updateDescriptorSets();

    void createRenderPipeline();

    void combineVectorFields(VkCommandBuffer commandBuffer);

    void computeMinMaxPressure(VkCommandBuffer commandBuffer);

    void computeStreamLines(VkCommandBuffer commandBuffer);

    void copyPressure(VkCommandBuffer commandBuffer);

    void copyTextureToDumpBuffer(VkCommandBuffer commandBuffer, Texture& texture, VulkanBuffer& buffer);

    void copyPressureToDumpBuffer(VkCommandBuffer commandBuffer);

    void writeFieldDumpCsvFiles(uint32_t step);

    void writeVectorComponentGrid(std::ostream& out, const glm::vec4* vectorValues, uint32_t component) const;

    void writeScalarGrid(std::ostream& out, const float* values) const;

private:
    VulkanDescriptorPool* _descriptorPool{};
    VulkanRenderPass* _renderPass{};
    VulkanDescriptorSetLayout _fieldSetLayout;
    eular::FluidSolver* _solver{};
    glm::ivec2 _gridSize{};
    glm::uvec2 _screenResolution{};
    glm::vec3 _streamColor{1};
    static constexpr uint32_t MaxDebugFields = 12;

    struct Globals {
        glm::ivec2 gridSize{1};
        glm::vec2 dx{1, 0};
        glm::vec2 dy{0, 1};
        float dt{0};
        float density{1};
        uint32_t ensureBoundaryCondition{1};
        uint32_t useHermite{0};
    };

    struct {
        VulkanDescriptorSetLayout setDescriptorSet;
        VkDescriptorSet descriptorSet{};
        VulkanBuffer buffer;
        Globals* data{};
    } _globals;

    struct Uniforms {
        glm::vec3 color{0};
        glm::ivec2 gridSize{1};
        float step_size{0.25};
        uint next_vertex{0};
        uint offset{5};
    };

    struct {
        VulkanPipeline pipeline;
        VulkanPipelineLayout layout;
        VulkanDescriptorSetLayout setDescriptorSet;
        VkDescriptorSet descriptorSet{};
        VulkanBuffer buffer;
        VulkanBuffer uniformBuffer;
        Uniforms* uniforms{};
        glm::vec3 color{1};
    } _streamLines;

    struct {
        VulkanPipeline pipeline;
        VulkanPipelineLayout layout;
        VulkanDescriptorSetLayout setDescriptorSet;
        VkDescriptorSet descriptorSet{};
        VulkanBuffer minValue;
        VulkanBuffer maxValue;
        VulkanBuffer field;
    } _pressure;

    struct VectorArrow {
        glm::vec2 vertex;
        glm::vec2 position;
    };

    struct {
        VulkanPipeline pipeline;
        VulkanPipelineLayout layout;
        VulkanDescriptorSetLayout setDescriptorSet;
        VkDescriptorSet descriptorSet{};
        VulkanBuffer vertices;
        Texture field;
        uint32_t numArrows{};
    } _vectorField;

    struct {
        VulkanPipeline pipeline;
        VulkanPipelineLayout layout;
        VkDescriptorSet combinedVectorDescriptorSet{};
        struct {
            uint32_t fieldCount{};
            uint32_t columns{4};
            uint32_t rows{3};
        } constants;
    } _debugFields;

    struct {
        VulkanPipeline pipeline;
        VulkanPipelineLayout layout;
        struct {
            glm::vec4 color{1.0f, 0.0f, 0.0f, 0.85f};
            uint32_t closedDomain{};
            uint32_t openBoundaryEdges{};
            uint32_t showColliders{};
        } constants;
    } _boundary;

    struct {
        VulkanBuffer vector;
        VulkanBuffer divergence;
        VulkanBuffer pressure;
        fs::path directory;
        uint32_t step{};
        uint32_t pendingStep{};
        bool pending{};
    } _fieldDump;

    PrefixSum _prefixSum;

    struct {
        VulkanBuffer vertices;
    } _screenQuad;
};
