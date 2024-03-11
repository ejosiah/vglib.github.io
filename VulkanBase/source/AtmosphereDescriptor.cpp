#include "atmosphere/AtmosphereDescriptor.hpp"

AtmosphereDescriptor::AtmosphereDescriptor(VulkanDevice *device, VulkanDescriptorPool *m_descriptorPool, BindlessDescriptor* bindlessDescriptor)
: m_device{ device }
, m_descriptorPool{ m_descriptorPool }
, m_bindlessDescriptor{ bindlessDescriptor }
{}

void AtmosphereDescriptor::init() {
    createBuffers();
    createSampler();
    createLutTextures();
    createDescriptorSetLayout();
    updateDescriptorSet();
}

void AtmosphereDescriptor::createBuffers() {
    m_uboBuffer = m_device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(UBO), "atmosphere_params");
    m_densityProfileBuffer = m_device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(Atmosphere::DensityProfileLayer) * 4, "density_profile_layers");

    ubo = reinterpret_cast<UBO*>(m_uboBuffer.map());
    layers = reinterpret_cast<Atmosphere::DensityProfileLayer*>(m_densityProfileBuffer.map());
}

void AtmosphereDescriptor::createSampler() {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    sampler = m_device->createSampler(samplerInfo);
}

void AtmosphereDescriptor::createLutTextures() {
    textures::create(*m_device, transmittanceLUT, VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT
            , {TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT, 1}
            , VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, sizeof(float ));

    textures::create(*m_device, irradianceLut, VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT
            , {IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT, 1}
            , VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, sizeof(float ));

    textures::create(*m_device, scatteringLUT, VK_IMAGE_TYPE_3D, VK_FORMAT_R32G32B32A32_SFLOAT
            , {SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_DEPTH}
            , VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, sizeof(float ));

    transmittanceLUT.image.transitionLayout(m_device->graphicsCommandPool(), VK_IMAGE_LAYOUT_GENERAL);
    irradianceLut.image.transitionLayout(m_device->graphicsCommandPool(), VK_IMAGE_LAYOUT_GENERAL);
    scatteringLUT.image.transitionLayout(m_device->graphicsCommandPool(), VK_IMAGE_LAYOUT_GENERAL);
}

void AtmosphereDescriptor::createDescriptorSetLayout() {
    uboDescriptorSetLayout =
        m_device->descriptorSetLayoutBuilder()
            .name("atmosphere_ubo")
            .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_ALL)
            .binding(1)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(NUM_DENSITY_PROFILES)
                .shaderStages(VK_SHADER_STAGE_ALL)
            .createLayout();

    lutDescriptorSetLayout =
        m_device->descriptorSetLayoutBuilder()
            .name("atmosphere_lut")
            .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_ALL)
                .immutableSamplers(sampler)
            .binding(1)
                .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_ALL)
                .immutableSamplers(sampler)
            .binding(2)
                .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_ALL)
                .immutableSamplers(sampler)
            .binding(3)
                .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_ALL)
                .immutableSamplers(sampler)
            .createLayout();

    auto sets = m_descriptorPool->allocate({uboDescriptorSetLayout, lutDescriptorSetLayout});
    uboDescriptorSet = sets[0];
    lutDescriptorSet = sets[1];

}

void AtmosphereDescriptor::updateDescriptorSet() {
    auto writes = initializers::writeDescriptorSets<2>();

    writes[0].dstSet = uboDescriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].descriptorCount = 1;
    VkDescriptorBufferInfo uboInfo{m_uboBuffer, 0, VK_WHOLE_SIZE};
    writes[0].pBufferInfo = &uboInfo;

    VkDeviceSize DensityProfileLayerSize = sizeof(Atmosphere::DensityProfileLayer);
    std::array<VkDescriptorBufferInfo, NUM_DENSITY_PROFILES> layerInfos{
            VkDescriptorBufferInfo{m_densityProfileBuffer, 0, DensityProfileLayerSize},
            VkDescriptorBufferInfo{m_densityProfileBuffer, DensityProfileLayerSize * 1, DensityProfileLayerSize},
            VkDescriptorBufferInfo{m_densityProfileBuffer, DensityProfileLayerSize * 2, DensityProfileLayerSize},
            VkDescriptorBufferInfo{m_densityProfileBuffer, DensityProfileLayerSize * 3, DensityProfileLayerSize},
    };

    writes[1].dstSet = uboDescriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = layerInfos.size();
    writes[1].pBufferInfo = layerInfos.data();
    m_device->updateDescriptorSets(writes);

    writes = initializers::writeDescriptorSets<4>();

    writes[0].dstSet = lutDescriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].descriptorCount = 1;
    VkDescriptorImageInfo irradianceInfo{ VK_NULL_HANDLE, irradianceLut.imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
    writes[0].pImageInfo = &irradianceInfo;

    writes[1].dstSet = lutDescriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].descriptorCount = 1;
    VkDescriptorImageInfo transmittanceInfo{ VK_NULL_HANDLE, transmittanceLUT.imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
    writes[1].pImageInfo = &transmittanceInfo;

    writes[2].dstSet = lutDescriptorSet ;
    writes[2].dstBinding = 2;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].descriptorCount = 1;
    VkDescriptorImageInfo scatteringInfo{ VK_NULL_HANDLE, scatteringLUT.imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
    writes[2].pImageInfo = &scatteringInfo;

    // single_mie_scattering
    writes[3] = writes[2];
    writes[3].dstBinding = 3;
    m_device->updateDescriptorSets(writes);

    if(m_bindlessDescriptor) {
        m_bindlessDescriptor->update({&irradianceLut, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2U});
        m_bindlessDescriptor->update({&transmittanceLUT, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3U});
        m_bindlessDescriptor->update({&scatteringLUT, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4U});
    }
}

void AtmosphereDescriptor::load(const std::filesystem::path &path) {
    auto atmosphere = Atmosphere::load(path);
    assert(atmosphere.header.scatteringDimensions.x == SCATTERING_TEXTURE_WIDTH);
    assert(atmosphere.header.scatteringDimensions.y == SCATTERING_TEXTURE_HEIGHT);
    assert(atmosphere.header.scatteringDimensions.z == SCATTERING_TEXTURE_DEPTH);
    ubo->solarIrradiance = atmosphere.header.solarIrradiance;
    ubo->sunAngularRadius = atmosphere.header.sunAngularRadius;
    ubo->bottomRadius = atmosphere.header.bottomRadius;
    ubo->topRadius = atmosphere.header.topRadius;
    ubo->rayleighScattering = atmosphere.header.rayleighScattering;
    ubo->mieScattering = atmosphere.header.mieScattering;
    ubo->mieExtinction = atmosphere.header.mieExtinction;
    ubo->mieAnisotropicFactor = atmosphere.header.mieAnisotropicFactor;
    ubo->absorptionExtinction = atmosphere.header.absorptionExtinction;
    ubo->groundAlbedo = atmosphere.header.groundAlbedo;
    ubo->mu_s_min = atmosphere.header.mu_s_min;
    ubo->lengthUnitInMeters = atmosphere.header.lengthUnitInMeters;

    auto staging = m_device->createStagingBuffer(DATA_SIZE);
    staging.copy(atmosphere.data);


    m_device->graphicsCommandPool().oneTimeCommand([&](auto cmdBuffer){
        std::vector<VkImageMemoryBarrier> barriers(3, { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER });

        for(auto& barrier : barriers) {
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;
        }

        barriers[0].image = transmittanceLUT.image.image;
        barriers[1].image = irradianceLut.image.image;
        barriers[2].image = scatteringLUT.image.image;

        vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                , VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, barriers.size(), barriers.data());

        VkBufferImageCopy region{0, 0, 0};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.bufferOffset = 0;
        region.imageExtent = {IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT, 1};
        vkCmdCopyBufferToImage(cmdBuffer, staging.buffer, irradianceLut.image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        region.bufferOffset += IRRADIANCE_DATA_SIZE;
        region.imageExtent = { TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT, 1 };
        vkCmdCopyBufferToImage(cmdBuffer, staging.buffer, transmittanceLUT.image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        region.bufferOffset += TRANSMISSION_DATA_SIZE;
        region.imageExtent = { SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_DEPTH};
        vkCmdCopyBufferToImage(cmdBuffer, staging.buffer, scatteringLUT.image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        region.bufferOffset += SCATTERING_DATA_SIZE;
        assert(region.bufferOffset == DATA_SIZE);

        for(auto& barrier : barriers) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        }


        vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT
                , VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                , 0, 0, nullptr, 0, nullptr, barriers.size(), barriers.data());
    });
}

const VulkanDescriptorSetLayout &AtmosphereDescriptor::bindnessSetLayout() const {
    return *m_bindlessDescriptor->descriptorSetLayout;
}

 VkDescriptorSet AtmosphereDescriptor::bindessDescriptorSet()  {
    return m_bindlessDescriptor->descriptorSet;
}