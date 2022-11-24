#include "VulkanRayTraceModel.hpp"
#include "VulkanRayTraceBaseApp.hpp"
#include "shader_binding_table.hpp"

#include "Model.hpp"
#include "Gui.hpp"
#include <taskflow/taskflow.hpp>

enum ShaderIndex{
    eRayGen = 0,
    eMiss,
    eLightMiss,

    eClosestHit,
    eOcclusionPrimary,

    eVolumeHit,
    eVolumeAnyHit,
    eOcclusionVolumeHit,
    eOcclusionVolumeAnyHit,

    eGlassHit,
    eGlassOcclusion,
    eShaderCount
};

struct Rectangle{
    glm::vec2 min;
    glm::vec4 max;
};

struct Sphere{
    glm::vec3 center;
    float radius;
};


class PathTracer : public VulkanRayTraceBaseApp {
public:
    explicit PathTracer(const Settings& settings = {});

protected:
    void initApp() final;

    void loadEnvironmentMap();

    void loadModel();

    void loadDragon();

    void initLights();

    void createCornellBox(phong::VulkanDrawableInfo info);

    void initCamera();

    void createDescriptorPool();

    void createDescriptorSetLayouts();

    void updateDescriptorSets();

    void createCommandPool();

    void createPipelineCache();

    void initCanvas();

    void createInverseCam();

    void createRayTracingPipeline();

    void createPostProcessPipeline();

    void rayTrace(VkCommandBuffer commandBuffer);

    void transferImage(VkCommandBuffer commandBuffer);

    void rayTraceToTransferBarrier(VkCommandBuffer commandBuffer) const;

    void transferToRenderBarrier(VkCommandBuffer commandBuffer) const;

    void onSwapChainDispose() final;

    void onSwapChainRecreation() final;

    VkCommandBuffer *buildCommandBuffers(uint32_t imageIndex, uint32_t &numCommandBuffers) final;

    void renderUI(VkCommandBuffer commandBuffer);

    void update(float time) final;

    void checkAppInputs() final;

    void cleanup() final;

    void onPause() final;

    void newFrame() final;

    void endFrame() final;

protected:
    struct {
        VulkanPipeline pipeline;
        VulkanPipelineLayout layout;
        VulkanDescriptorSetLayout descriptorSetLayout;
        VulkanDescriptorSetLayout instanceDescriptorSetLayout;
        VulkanDescriptorSetLayout vertexDescriptorSetLayout;
        VkDescriptorSet descriptorSet;
        VkDescriptorSet instanceDescriptorSet;
        VkDescriptorSet vertexDescriptorSet;
    } raytrace;

    struct {
        VulkanPipeline pipeline;
        VulkanPipelineLayout layout;
    } postProcess;

    ShaderTablesDescription shaderTablesDesc;
    ShaderBindingTables bindingTables;

    VulkanBuffer inverseCamProj;
    Canvas canvas{};

    VulkanDescriptorPool descriptorPool;
    VulkanCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    VulkanPipelineCache pipelineCache;
    std::unique_ptr<FirstPersonCameraController> camera;

    Model m{};
    Gui gui;

    std::vector<VulkanDrawable*> objects;

    VulkanBuffer lightsBuffer;
    Texture environmentMap;
    Distribution2DTexture envMapDistribution;
    VulkanDescriptorSetLayout envMapDescriptorSetLayout;
    VkDescriptorSet envMapDescriptorSet;

    VulkanDescriptorSetLayout sceneDescriptorSetLayout;
    VkDescriptorSet sceneDescriptorSet;
    static constexpr VkPipelineStageFlags ALL_RAY_TRACE_STAGES =
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
            VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
            VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_CALLABLE_BIT_KHR;

    std::vector<rt::MeshObjectInstance> instances;
    tf::Executor executor;
    std::future<void> dragonLoad{};
    Light cornellLight;
    VkPhysicalDeviceSynchronization2Features syncFeatures;
    VkPhysicalDeviceRayQueryFeaturesKHR  rayQueryFeatures{};
    VkPhysicalDeviceFeatures2 features2;
    Texture rayTracedTexture;
    Texture gBuffer;

    struct {
        VulkanBuffer albedo;
        VulkanBuffer normal;
    } guide;
};