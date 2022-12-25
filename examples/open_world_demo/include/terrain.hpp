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

struct PatchVertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

class Terrain{
public:
    Terrain(const VulkanDevice& device, const VulkanDescriptorPool& descriptorPool, const FileManager& fileManager,
            uint32_t width, uint32_t height, VulkanRenderPass& renderPass);

    void update(const SceneData& sceneData);

    void render(VkCommandBuffer commandBuffer);

    void renderUI();

    void resize(VulkanRenderPass& renderPass, uint32_t width, uint32_t height);

private:
    void loadHeightMap();

    void loadShadingTextures();

    void createPatches();

    void initUBO();

    void createDescriptorSetLayout();

    void updateDescriptorSet();

    void createPipelines();

    std::vector<glm::vec3> generateNormals();

    inline const VulkanDevice& device() const {
        return *m_device;
    }

    inline const VulkanDescriptorPool& descriptorPool() const {
        return *m_descriptorPool;
    }

    inline const VulkanRenderPass& renderPass() const {
        return *m_renderPass;
    }

    std::string resource(const std::string& name);

private:
    const VulkanDevice* m_device;
    const VulkanDescriptorPool* m_descriptorPool;
    const FileManager* m_filemanager;
    VulkanRenderPass* m_renderPass;


    enum class LodStrategy{
        DistanceFromCamera = 0, SphereProjection
    };

    struct  UniformBufferObject{
        glm::mat4 model;
        glm::mat4 view;
        glm::mat4 projection;
        glm::mat4 mvp;

        glm::vec3 sunPosition;
        float heightScale;

        glm::vec3 wireframeColor;
        int wireframe;

        glm::vec2 numPatches;
        float wireframeWidth;
        int lod;

        float lodMinDepth;
        float lodMaxDepth;
        int minTessLevel;
        int maxTessLevel;

        glm::vec2 viewportSize;
        int lighting;
        int tessLevelColor;

        glm::vec3 cameraPosition;
        float lodTargetTriangleWidth;

        int lodStrategy;
        int invertRoughness;
        int materialId;
        int greenGrass;
        int dirt;
        int dirtRock;
        int snowFresh;
        float minHeight;
        float maxHeight;
        float snowStart;
    };

    UniformBufferObject* ubo{};

    VulkanDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSet descriptorSet;

    VulkanDescriptorSetLayout shadingSetLayout;
    VkDescriptorSet shadingSet;
    struct{
        Texture displacement;
        Texture normal;
    } heightMap;

    struct {
        Texture albedo;
        Texture metalness;
        Texture roughness;
        Texture normal;
        Texture ambientOcclusion;
        Texture displacement;
        Texture groundMask;
    } shadingMap;

    Texture randomTexture;

    VulkanBuffer patchesBuffer;
    VulkanBuffer indexBuffer;
    VulkanBuffer uboBuffer;

    struct {
        VulkanPipelineLayout layout;
        VulkanPipeline pipeline;
    } terrain;

    uint32_t m_width{0};
    uint32_t m_height{0};

    static constexpr int SQRT_NUM_PATCHES = 64;
    static constexpr float PATCH_SIZE = 60 * km;
    static constexpr VkShaderStageFlags ALL_SHADER_STAGES =
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT
            | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
};