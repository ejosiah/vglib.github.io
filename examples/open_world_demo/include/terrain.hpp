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

    void renderTerrain();

    void renderUI();

    void resize(VulkanRenderPass& renderPass, uint32_t width, uint32_t height);

    std::unique_ptr<GBuffer> gBuffer;
    std::unique_ptr<ShadowVolume> shadowVolume;

private:

    void loadHeightMap();

    void loadShadingTextures();

    void createPatches();

    void initUBO();

    void initSamplers();

    void createDescriptorSetLayout();

    void updateDescriptorSet();

    void createFrameBufferAttachments();

    void createGBufferFrameBuffer();

    void createShadowVolumeFrameBuffer();

    void createPipelines();

    std::vector<glm::vec3> generateNormals();

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

    VulkanRenderPass m_gBufferRenderPass;
    VulkanFramebuffer m_gBufferFramebuffer;

    VulkanRenderPass m_shadowVolumeRenderPass;
    VulkanFramebuffer m_shadowVolumeFramebuffer;

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
        float minZ;
        float maxZ;
        float snowStart;
    };

    UniformBufferObject* ubo{};

    VulkanDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSet descriptorSet;

    VulkanDescriptorSetLayout  gBufferSetLayout;
    VkDescriptorSet gBufferSet;

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

    int* triangleCount;
    VulkanBuffer triangleCountBuffer;
    VulkanBuffer vertexBuffer;
    VulkanDescriptorSetLayout trianglesSetLayout;
    VkDescriptorSet trianglesSet;
    uint32_t triangleCapacity = 35000000;

    struct {
        VulkanPipelineLayout layout;
        VulkanPipeline pipeline;
    } terrain;

    struct {
        VulkanPipelineLayout layout;
        VulkanPipeline pipeline;
    } screen;


    uint32_t m_width{0};
    uint32_t m_height{0};

    VulkanBuffer screenBuffer;
    VulkanSampler valueSampler;

    static constexpr int SQRT_NUM_PATCHES = 64;
    static constexpr float PATCH_SIZE = 60 * km;
    static constexpr VkShaderStageFlags ALL_SHADER_STAGES =
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT
            | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
};