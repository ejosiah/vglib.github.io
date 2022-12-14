#pragma once
#include "VulkanBaseApp.h"
#include <array>

struct VertexInput{
    glm::vec2 position;
    glm::vec2 uv;
    glm::vec4 color;
};

class ComputeDemo final : public VulkanBaseApp{
public:
    explicit ComputeDemo(const Settings& settings);

protected:
    void initApp() final;

    VkCommandBuffer& dispatchCompute();

    VkCommandBuffer *buildCommandBuffers(uint32_t imageIndex, uint32_t &numCommandBuffers) final;

    void update(float time) final;

    void checkAppInputs() final;

    void loadTexture();

    void blurImage();

    void updateBlurFunc();

    void createSamplers();

    void createVertexBuffer();

    void createDescriptorSetLayout();

    void createDescriptorPool();

    void createDescriptorSet();

    void createGraphicsPipeline();

    void createComputeDescriptorSetLayout();

    void createComputePipeline();

    void createComputeImage();

    void onSwapChainDispose() override;

    void onSwapChainRecreation() override;

    void renderUI(VkCommandBuffer commandBuffer);

    void initRenderBlur();

    void blurImageRender();

private:
    VulkanCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkCommandBuffer> cpuCmdBuffers;
    VulkanPipelineLayout pipelineLayout;
    VulkanPipeline pipeline;
    VulkanBuffer vertexBuffer;
    VulkanBuffer vertexColorBuffer;
    VulkanDescriptorPool descriptorPool;
    VulkanDescriptorSetLayout textureSetLayout;
    VkDescriptorSet descriptorSet;
    Texture texture;
    struct {
        VulkanDescriptorSetLayout imageSetLayout;
        VulkanPipelineLayout pipelineLayout;
        VkDescriptorSet descriptorSet;
        VulkanPipeline pipeline;
        Texture texture;
    } compute;

    VulkanSampler sampler;

    struct {
        VulkanPipeline pipeline;
        VulkanPipelineLayout layout;
        VkDescriptorSet inSet;
        VkDescriptorSet outSet;

        struct {
            FramebufferAttachment colorAttachment;
            VulkanFramebuffer framebuffer;
            VulkanRenderPass renderPass;
            VulkanPipeline pipeline;
            VulkanPipelineLayout layout;
            VkDescriptorSet descriptorSet;
        } renderBlur[2];

        struct {
//            float weights[5][5];
//            float weights[5]{0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216};
            std::array<float, 32> weights;
            int horizontal{true};
        } constants;
        float sd{1.0};
        glm::vec2 avg{0};
        bool on{false};
        int iterations{10};
        bool useRender;
    } blur;
};