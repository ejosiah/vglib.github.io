#pragma once

#include "common.h"
#include "constants.hpp"
#include "Camera.h"
#include "Texture.h"
#include "VulkanDevice.h"
#include "VulkanBuffer.h"
#include "filemanager.hpp"
#include <glm/glm.hpp>
#include "scene.hpp"

class Clouds {
public:
    Clouds(const VulkanDevice& device, const VulkanDescriptorPool& descriptorPool, const FileManager& fileManager,
           VulkanRenderPass& renderPass, uint32_t width, uint32_t height, std::shared_ptr<GBuffer> terrainGBuffer);

private:

    struct Ubo {
        glm::mat4 inverseProjection;
        glm::mat4 inverseView;
        alignas(16) glm::vec3 camera{0};
        alignas(16) glm::vec3 earth_center;
        alignas(16) glm::vec3 sun_direction;
        float innerRadius;
        float outerRadius;
        int viewPortWidth;
        int viewPortHeight;
    };

    Ubo* ubo{};
    VulkanBuffer uboBuffer;
};