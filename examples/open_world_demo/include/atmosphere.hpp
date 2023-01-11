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

class Atmosphere{
public:
    Atmosphere(const VulkanDevice& device, const VulkanDescriptorPool& descriptorPool, const FileManager& fileManager,
               VulkanRenderPass& renderPass, uint32_t width, uint32_t height, std::shared_ptr<GBuffer> terrainGBuffer,
               std::shared_ptr<ShadowVolume> terrainShadowVolume);

    void update(const SceneData& sceneData);

    void render(VkCommandBuffer commandBuffer);

    void renderUI();

    void resize(VulkanRenderPass& renderPass, std::shared_ptr<GBuffer> terrainGBuffer,
                std::shared_ptr<ShadowVolume> terrainShadowVolume, uint32_t width, uint32_t height);

private:
    void loadAtmosphereLUT();

    void initBuffers();

    void initUbo();

    void createDescriptorSetLayouts();

    void updateDescriptorSets();

    void createPipelines();

    inline const VulkanDevice& device() const {
        return *m_device;
    }

    inline const VulkanDescriptorPool& descriptorPool() const {
        return *m_descriptorPool;
    }

    std::string resource(const std::string& name);

private:
    const VulkanDevice* m_device;
    const VulkanDescriptorPool* m_descriptorPool;
    const FileManager* m_filemanager;
    VulkanRenderPass* m_renderPass;

    struct {
        VulkanPipelineLayout layout;
        VulkanPipeline pipeline;
    } atmosphere;


    struct {
        Texture irradiance;
        Texture transmittance;
        Texture scattering;
    } atmosphereLUT;

    struct Ubo {
        glm::mat4 model_from_view{1};
        glm::mat4 view_from_clip{1};
        alignas(16) glm::vec3 camera{0};
        alignas(16) glm::vec3 white_point{1};
        alignas(16) glm::vec3 earth_center;
        alignas(16) glm::vec3 sun_direction;
        alignas(16) glm::vec3 sun_size{0};
        float exposure;
        int lightShaft;
    };
    Ubo* ubo{};
    VulkanBuffer uboBuffer;


    uint32_t m_width{0};
    uint32_t m_height{0};

    bool debugMode = false;

    VulkanBuffer screenBuffer;
    VulkanSampler valueSampler;
    VulkanDescriptorSetLayout uboSetLayout;
    VkDescriptorSet uboSet;

    VulkanDescriptorSetLayout atmosphereLutSetLayout;
    VkDescriptorSet atmosphereLutSet;

    std::shared_ptr<GBuffer> m_terrainGBuffer;
    std::shared_ptr<ShadowVolume> m_terrainShadowVolume;
};