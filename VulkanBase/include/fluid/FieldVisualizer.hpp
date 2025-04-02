#pragma once

#include "Field.hpp"
#include "ComputePipelins.hpp"
#include "filemanager.hpp"

class FieldVisualizer : ComputePipelines {
public:
    FieldVisualizer() = default;

    FieldVisualizer(VulkanDevice* device, VulkanDescriptorPool* descriptorPool,
                    VulkanRenderPass* renderPass, VulkanDescriptorSetLayout fieldSetLayout,
                    glm::uvec2 screenResolution, glm::ivec2 gridSize);

    void init();

    void add(eular::VectorField* vectorField);

    void setStreamLineColor(const glm::vec3& streamColor);

    void update(VkCommandBuffer commandBuffer);

    void renderStreamLines(VkCommandBuffer commandBuffer);

protected:
    std::vector<PipelineMetaData> pipelineMetaData() override;

private:
    void createBuffers();
    void createDescriptorSets();
    void updateDescriptorSets();

    void createRenderPipeline();


private:
    VulkanDescriptorPool* _descriptorPool{};
    VulkanRenderPass* _renderPass{};
    VulkanDescriptorSetLayout _fieldSetLayout;
    eular::VectorField* _vectorField{};
    glm::ivec2 _gridSize{};
    glm::uvec2 _screenResolution{};
    glm::vec3 _streamColor{1};

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

};