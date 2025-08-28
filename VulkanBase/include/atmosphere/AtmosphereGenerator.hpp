#pragma once
#include "Texture.h"
#include "VulkanBuffer.h"
#include "VulkanDevice.h"
#include "VulkanDescriptorSet.h"
#include "filemanager.hpp"
#include "atmosphere/AtmosphereContants.hpp"
#include "atmosphere/AtmosphereDescriptor.hpp"

#include <glm/glm.hpp>

#include <array>
#include <functional>
#include <filesystem>

class AtmosphereGenerator{
public:
    using Params = Atmosphere::Params;

    AtmosphereGenerator(VulkanDevice* device, VulkanDescriptorPool* descriptorPool, FileManager* fileManager, BindlessDescriptor* bindlessDescriptor = nullptr);

    void generateLUT();

    void load();

    std::function<void()> ui();

    const AtmosphereDescriptor& atmosphereDescriptor() const;

protected:

    void initAtmosphereDescriptor();

    void createBarriers();

    void createSampler();

    void createTextures();

    void createDescriptorSetLayout();

    void updateDescriptorSet();

    void createPipelines();

    void computeTransmittanceLUT(VkCommandBuffer commandBuffer);

    void computeDirectIrradiance(VkCommandBuffer commandBuffer);

    void computeSingleScattering(VkCommandBuffer commandBuffer);

    void computeScatteringDensity(VkCommandBuffer commandBuffer, int scatteringOrder);

    void computeIndirectIrradiance(VkCommandBuffer commandBuffer, int scatteringOrder);

    void computeMultipleScattering(VkCommandBuffer commandBuffer);

    void barrier(VkCommandBuffer commandBuffer, const std::vector<int>& images);

    void refresh();

    void save(const std::filesystem::path& path);

    Texture& transmittanceLUT();

    Texture& irradianceLut();

    Texture& scatteringLUT();

    AtmosphereDescriptor::UBO* ubo();

    Atmosphere::DensityProfileLayer* layers();

    VulkanDescriptorSetLayout& uboDescriptorSetLayout();

    VkDescriptorSet uboDescriptorSet();

    VulkanDescriptorSetLayout& lutDescriptorSetLayout();

    VkDescriptorSet lutDescriptorSet();

public:
    Params params;

    VulkanDescriptorSetLayout  imageDescriptorSetLayout;
    VkDescriptorSet  imageDescriptorSet;

private:
    VulkanDevice* m_device;
    VulkanDescriptorPool* m_descriptorPool;
    FileManager* m_fileMgr;
    VulkanSampler sampler;

    struct Pipeline{
        VulkanPipeline pipeline;
        VulkanPipelineLayout layout;
    };

    Texture deltaIrradianceTexture;
    Texture deltaRayleighScatteringTexture;
    Texture deltaMieScatteringTexture;
    Texture deltaScatteringDensityTexture;
    Texture deltaMultipleScatteringTexture;

    VulkanDescriptorSetLayout tempDescriptorSetLayout;
    VkDescriptorSet tempDescriptorSet;

    struct {
        Pipeline compute_transmittance;
        Pipeline compute_direct_irradiance;
        Pipeline compute_single_scattering;
        Pipeline compute_scattering_density;
        Pipeline compute_indirect_irradiance;
        Pipeline compute_multiple_scattering;
    } pipelines;

    BindlessDescriptor* m_bindlessDescriptor;
    AtmosphereDescriptor m_atmosphereDescriptor;

    std::vector<VkImageMemoryBarrier> m_barriers{};
    static constexpr int TRANSMITTANCE_BARRIER = 0;
    static constexpr int IRRADIANCE_BARRIER = 1;
    static constexpr int DELTA_RAYLEIGH_BARRIER = 2;
    static constexpr int DELTA_MIE_BARRIER = 3;
    static constexpr int DELTA_IRRADIANCE_BARRIER = 4;
    static constexpr int DELTA_SCATTERING_DENSITY_BARRIER = 5;
    static constexpr int DELTA_MULTIPLE_DENSITY_BARRIER = 6;
    static constexpr int SCATTERING_BARRIER = 7;
    static constexpr int NUM_BARRIERS = 8;

    static constexpr int NUM_SCATTERING_ORDER = 4;
};