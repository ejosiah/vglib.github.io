#include "blur_image.hpp"

Blur::Blur(VulkanDevice *device, VulkanDescriptorPool *descriptorPool, FileManager *fileManager, uint32_t width,
           uint32_t height):
 m_device{ device },
 m_descriptorPool{ descriptorPool },
 m_fileManager{ fileManager },
 m_width{ width },
 m_height{ height }
  
{
    createTexture();
    createDescriptorSetLayout();
    updateDescriptorSets();
    createPipeline();
}

void Blur::operator()(VkCommandBuffer commandBuffer, VulkanImage& inputImage, VulkanImage& outputImage, int iterations) {
    VkImageMemoryBarrier barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.image = m_texture.image;

    inputImage.copyToBuffer(commandBuffer, m_transferBuffer, VK_IMAGE_LAYOUT_GENERAL);
    m_texture.image.copyFromBuffer(commandBuffer, m_transferBuffer, VK_IMAGE_LAYOUT_GENERAL);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline.handle);


    const int limit = iterations * 2;
    for (int i = 0; i < limit; i++) {
        int horizontal = 1 - (i % 2);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_layout.handle, 0, 1,
                                &m_descriptorSet, 0, VK_NULL_HANDLE);
        vkCmdPushConstants(commandBuffer, m_layout.handle, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int),
                           &horizontal);
        vkCmdDispatch(commandBuffer, m_width, m_height, 1);

        if ((i + 1) < iterations) {
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, VK_NULL_HANDLE, 0, VK_NULL_HANDLE,
                                 1, &barrier);
        }
    }

    m_texture.image.copyToBuffer(commandBuffer, m_transferBuffer, VK_IMAGE_LAYOUT_GENERAL);
    outputImage.copyFromBuffer(commandBuffer, m_transferBuffer, VK_IMAGE_LAYOUT_GENERAL);
}

void Blur::execute(VkCommandBuffer commandBuffer, VulkanImage &inputImage, VulkanImage &outputImage, int iterations) {
    (*this)(commandBuffer, inputImage, outputImage, iterations);
}

void Blur::createTexture() {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    m_texture.sampler = m_device->createSampler(samplerInfo);

    textures::create(*m_device, m_texture, VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, {m_width, m_height, 1}, VK_SAMPLER_ADDRESS_MODE_REPEAT, sizeof(float));
    m_texture.image.transitionLayout(m_device->graphicsCommandPool(), VK_IMAGE_LAYOUT_GENERAL);

    m_transferBuffer = m_device->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY, m_texture.image.size);
}

void Blur::createDescriptorSetLayout() {
    m_descriptorSetLayout =
        m_device->descriptorSetLayoutBuilder()
            .name("linear_blur")
                .binding(0)
                    .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
        .createLayout();
}

void Blur::updateDescriptorSets() {
    m_descriptorSet = m_descriptorPool->allocate({ m_descriptorSetLayout }).front();

    auto writes = initializers::writeDescriptorSets<2>();

    writes[0].dstSet = m_descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].descriptorCount = 1;
    VkDescriptorImageInfo iInfo{m_texture.sampler.handle, m_texture.imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
    writes[0].pImageInfo = &iInfo;

    writes[1].dstSet = m_descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].descriptorCount = 1;
    VkDescriptorImageInfo oInfo{VK_NULL_HANDLE, m_texture.imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
    writes[1].pImageInfo = &oInfo;

    m_device->updateDescriptorSets(writes);
}

void Blur::createPipeline() {
    auto linearBlurShaderModule = m_device->createShaderModule(m_fileManager->getFullPath("linear_blur.comp.spv")->string()) ;
    auto stage = initializers::shaderStage({ linearBlurShaderModule, VK_SHADER_STAGE_COMPUTE_BIT});

    m_layout = m_device->createPipelineLayout({ m_descriptorSetLayout }, {{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int)}});
    auto info = initializers::computePipelineCreateInfo();


    info.stage = stage;
    info.layout = m_layout.handle;
    m_pipeline = m_device->createComputePipeline(info);
}

void Blur::refresh(uint32_t width, uint32_t height) {
    m_width = width;
    m_height = height;
    createTexture();
    updateDescriptorSets();
}
