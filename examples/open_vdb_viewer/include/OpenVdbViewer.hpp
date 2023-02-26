#include "VulkanBaseApp.h"
#include "VulkanDescriptorSet.h"
#include <glm/glm.hpp>
#include <openvdb/openvdb.h>

struct VolumeData{
    glm::vec3 boxMin;
    glm::vec3 boxMax;
    std::vector<float> data;
};

struct CameraUbo{
    glm::mat4 projection;
    glm::mat4 view;
    glm::mat4 inverseProjection;
    glm::mat4 inverseView;
};

struct VolumeUbo{
    alignas(16) glm::vec3 boxMin;
    alignas(16) glm::vec3 boxMax;
    alignas(16) glm::vec3 lightPosition;
    float invMaxDensity;
    int numSamples;
    float coneSpread;
    float g;
    float lightIntensity;
    float time;
    int frame;
    int width;
    int height;
};

enum class Renderer :  int {
    RAY_MARCHING = 0, DELTA_TRACKING, PATH_TRACING
};

class OpenVdbViewer : public VulkanBaseApp{
public:
    explicit OpenVdbViewer(const Settings& settings = {});

protected:
    void initApp() override;

    void createPlaceHolderTexture();

    void initCamera();

    void updateCamera();

    void createBuffers();

    void createDescriptorPool();

    void createDescriptorSetLayouts();

    void updateDescriptorSets();

    void updateVolumeDescriptorSets();

    void loadVolume(openvdb::io::File& file);

    void createCommandPool();

    void createPipelineCache();

    void createRenderPipeline();

    void onSwapChainDispose() override;

    void onSwapChainRecreation() override;

    VkCommandBuffer *buildCommandBuffers(uint32_t imageIndex, uint32_t &numCommandBuffers) override;

    void renderUI(VkCommandBuffer commandBuffer);

    void renderVolume(VkCommandBuffer commandBuffer);

    void renderBackground(VkCommandBuffer commandBuffer);

    void renderLight(VkCommandBuffer commandBuffer);

    void renderWithRayMarching(VkCommandBuffer commandBuffer);

    void renderWithDeltaTracking(VkCommandBuffer commandBuffer);

    void renderVolumeSlices(VkCommandBuffer commandBuffer);

    bool openFileDialog();

    void update(float time) override;

    void checkAppInputs() override;

    void cleanup() override;

    void onPause() override;

    void fileInfo();

protected:
    struct {
        VulkanPipelineLayout layout;
        VulkanPipeline pipeline;
    } rayMarching;

    struct {
        VulkanPipelineLayout layout;
        VulkanPipeline pipeline;
    } deltaTracking;

    struct {
        VulkanPipelineLayout layout;
        VulkanPipeline pipeline;
    } background;

    struct {
        VulkanPipelineLayout layout;
        VulkanPipeline pipeline;
        glm::vec3 position{0};
        VulkanBuffer vertexBuffer;
        VulkanBuffer indexBuffer;
        float scale{1};
        float intensity;
    } light;

    VulkanDescriptorPool descriptorPool;
    VulkanCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    VulkanPipelineCache pipelineCache;
    std::unique_ptr<OrbitingCameraController> camera;
    std::string vdbPath;
    bool fileValid = true;
    std::string startPath = R"(C:\Users\Josiah Ebhomenye\OneDrive\media\volumes\VDB-Clouds-Pack-Pixel-Lab\VDB Cloud Files)";


    VulkanDescriptorSetLayout descriptorSetLayout;
    VulkanDescriptorSetLayout volumeDescriptorSetLayout;
    VkDescriptorSet descriptorSet;
    VkDescriptorSet volumeDescriptor;
    Texture volumeTexture;

    struct {
        VulkanPipelineLayout layout;
        VulkanPipeline pipeline;
        struct {
            glm::mat4 mvp{1};
            alignas(16) glm::vec3 viewDir{1};
            alignas(16) glm::vec3 scale{1};
            int numSlices{512};
        } constants;

    } sliceRenderer;

    std::future<VolumeData> volumeData;

    VulkanBuffer cameraUboBuffer;
    VulkanBuffer volumeUboBuffer;
    VulkanBuffer vertexBuffer;
    VulkanBuffer placeHolderVertexBuffer;
    CameraUbo* cameraUbo{};
    VolumeUbo* volumeUbo{};
    VulkanBuffer uboBuffer;
    Renderer renderer{Renderer::RAY_MARCHING};
};