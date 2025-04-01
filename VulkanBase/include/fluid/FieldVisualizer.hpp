#pragma once

#include "Field.hpp"
#include "ComputePipelins.hpp"

class FieldVisualizer : ComputePipelines {
public:
    FieldVisualizer() = default;

    FieldVisualizer(VulkanDevice* device, glm::ivec2 gridSize);

    void init();

    void add(eular::VectorField* vectorField);

    void update(VkCommandBuffer commandBuffer);

    void streamLines(VkCommandBuffer commandBuffer);

private:
    eular::VectorField* _vectorField{};
    glm::ivec2 _gridSize;
    VulkanBuffer _streamLines;

};