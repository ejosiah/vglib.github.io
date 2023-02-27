#pragma once

#include "VulkanDevice.h"
#include "VulkanDescriptorSet.h"
#include "Texture.h"
#include "filemanager.hpp"

class Blur{
public:
    Blur(VulkanDevice* device, VulkanDescriptorPool* descriptorPool, FileManager* fileManager, uint32_t width, uint32_t height);

    void operator()(VkCommandBuffer commandBuffer, VulkanImage& inputImage, VulkanImage& outputImage, int iterations = 9);

    void execute(VkCommandBuffer commandBuffer, VulkanImage& inputImage, VulkanImage& outputImage, int iterations = 9);

    void refresh(uint32_t width, uint32_t height);

protected:
    void createTexture();

    void createDescriptorSetLayout();

    void updateDescriptorSets();

    void createPipeline();

private:
    VulkanDevice* m_device{};
    VulkanDescriptorPool* m_descriptorPool;
    FileManager* m_fileManager{};
    mutable VulkanPipelineLayout m_layout;
    mutable VulkanPipeline  m_pipeline;
    VulkanDescriptorSetLayout m_descriptorSetLayout;
    VkDescriptorSet m_descriptorSet;
    Texture m_texture;
    uint32_t m_width{};
    uint32_t m_height{};
    VulkanBuffer m_transferBuffer;
};