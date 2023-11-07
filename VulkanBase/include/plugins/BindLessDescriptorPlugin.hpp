#pragma once

#include "Plugin.hpp"

#include "VulkanRAII.h"
#include "Texture.h"
#include "VulkanBuffer.h"
#include "SequenceGenerator.hpp"

#include <spdlog/spdlog.h>

#include <atomic>

static constexpr const char* PLUGIN_NAME_BINDLESS_DESCRIPTORS = "Bindless descriptors";

struct BindlessTexture {
    const Texture* texture{};
    VkDescriptorType type{};
    int index{-1};
};

struct BindlessBuffer {
    VulkanBuffer buffer;
    VkDescriptorType type{};
    int index{-1};
};

class BindLessDescriptorPlugin : public Plugin {
public:
    BindLessDescriptorPlugin()
    {
        m_indexingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
        m_indexingFeatures.descriptorBindingPartiallyBound = VK_TRUE;
        m_indexingFeatures.runtimeDescriptorArray = VK_TRUE;
        m_indexingFeatures.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
        m_indexingFeatures.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
        m_indexingFeatures.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
        m_indexingFeatures.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE;
        m_indexingFeatures.descriptorBindingUniformTexelBufferUpdateAfterBind = VK_TRUE;
        m_indexingFeatures.descriptorBindingStorageTexelBufferUpdateAfterBind = VK_TRUE;
    }

    ~BindLessDescriptorPlugin() override = default;

    [[nodiscard]]
    std::vector<const char *> deviceExtensions() const override {
        return { VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME };
    }

    void *nextChain() const override {
        auto ptr = &m_indexingFeatures;
        return ptr;
    }

    void preInit() override {
        Plugin::preInit();
    }

    void init() override {
        setBindingPoints();
        createDefaultSampler();
        createDescriptorPool();
        createDescriptorSetLayout();
    }

    std::string name() const override {
        return PLUGIN_NAME_BINDLESS_DESCRIPTORS;
    }


    void update(float time) override {
        Plugin::update(time);
    }

    void newFrame() override {
        Plugin::newFrame();
    }

    void endFrame() override {
        Plugin::endFrame();
    }

    void cleanup() override {
        Plugin::cleanup();
    }

    void onSwapChainDispose() override {

    }

    void onSwapChainRecreation() override {

    }

    bool supported(VkPhysicalDevice physicalDevice) override {
        VkPhysicalDeviceDescriptorIndexingFeatures iFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES };
        VkPhysicalDeviceFeatures2 deviceFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        deviceFeatures.pNext = &iFeatures;

        vkGetPhysicalDeviceFeatures2(physicalDevice, &deviceFeatures);

        return iFeatures.descriptorBindingPartiallyBound && iFeatures.runtimeDescriptorArray;
    }

    VkDescriptorSet descriptorSet() const {
        return m_descriptorSet;
    }

    BindlessTexture next(const Texture& texture, VkDescriptorType type) {
        return BindlessTexture{ &texture, type, nextIndex(type) };
    }

    BindlessBuffer next(const VulkanBuffer& buffer, VkDescriptorType type) {
        return BindlessBuffer{ buffer, type, nextIndex(type) };
    }

protected:
    void setBindingPoints() {
        auto nextBinding = sequence(int(TextureResourceBindingPoint));
        bindings[VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER] = nextBinding();
        bindings[VK_DESCRIPTOR_TYPE_STORAGE_IMAGE] = nextBinding();
        bindings[VK_DESCRIPTOR_TYPE_STORAGE_BUFFER] = nextBinding();
        bindings[VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER] = nextBinding();
        bindings[VK_DESCRIPTOR_TYPE_SAMPLER] = nextBinding();
        bindings[VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE] = nextBinding();
        bindings[VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER] = nextBinding();
        bindings[VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER] = nextBinding();


        bindingIds[VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER] = 0;
        bindingIds[VK_DESCRIPTOR_TYPE_STORAGE_IMAGE] = 0;
        bindingIds[VK_DESCRIPTOR_TYPE_STORAGE_BUFFER] = 0;
        bindingIds[VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER] = 0;
        bindingIds[VK_DESCRIPTOR_TYPE_SAMPLER] = 0;
        bindingIds[VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE] = 0;
        bindingIds[VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER] = 0;
        bindingIds[VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER] = 0;
    }

    void createDefaultSampler() {
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter =  VK_FILTER_LINEAR;;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        m_defaultSampler = device().createSampler(samplerInfo);
    }
    
    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, DescriptorPoolSize> poolSizes{
                {
                        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MaxDescriptorResources},
                        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, MaxDescriptorResources},
                        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MaxDescriptorResources},
                        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MaxDescriptorResources },
                        { VK_DESCRIPTOR_TYPE_SAMPLER, MaxDescriptorResources },
                        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, MaxDescriptorResources },
                        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, MaxDescriptorResources },
                        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, MaxDescriptorResources },
                }
        };

        constexpr uint32_t maxSets = MaxDescriptorResources * DescriptorPoolSize;
        m_descriptorPool = device().createDescriptorPool(maxSets, poolSizes, VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT);
    }

    void createDescriptorSetLayout() {
        auto nextBinding = sequence(TextureResourceBindingPoint);
        m_descriptorSetLayout =
            device().descriptorSetLayoutBuilder()
                .name("bind_less_descriptor_set_layout")
                .bindless()
                .binding(nextBinding())
                    .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(MaxDescriptorResources)
                    .shaderStages(VK_SHADER_STAGE_ALL)
                .binding(nextBinding())
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(MaxDescriptorResources)
                    .shaderStages(VK_SHADER_STAGE_ALL)
                .binding(nextBinding())
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(MaxDescriptorResources)
                    .shaderStages(VK_SHADER_STAGE_ALL)
                .binding(nextBinding())
                    .descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                    .descriptorCount(MaxDescriptorResources)
                    .shaderStages(VK_SHADER_STAGE_ALL)
                .binding(nextBinding())
                    .descriptorType(VK_DESCRIPTOR_TYPE_SAMPLER)
                    .descriptorCount(MaxDescriptorResources)
                    .shaderStages(VK_SHADER_STAGE_ALL)
                .binding(nextBinding())
                    .descriptorType(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
                    .descriptorCount(MaxDescriptorResources)
                    .shaderStages(VK_SHADER_STAGE_ALL)
                .binding(nextBinding())
                    .descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER)
                    .descriptorCount(MaxDescriptorResources)
                    .shaderStages(VK_SHADER_STAGE_ALL)
                .binding(nextBinding())
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER)
                    .descriptorCount(MaxDescriptorResources)
                    .shaderStages(VK_SHADER_STAGE_ALL)
                .createLayout();

        m_descriptorSet = createDescriptorSet();
    }

    VkDescriptorSet createDescriptorSet() {
        return m_descriptorPool.allocate( { m_descriptorSetLayout }).front();
    }

    int nextIndex(VkDescriptorType type) {
        return bindingIds[type]++;
    }

private:
    static constexpr uint32_t MaxDescriptorResources = 1024u;
    static constexpr uint32_t TextureResourceBindingPoint = 10u;
    static constexpr uint32_t DescriptorPoolSize = 8;

    mutable VkPhysicalDeviceDescriptorIndexingFeatures m_indexingFeatures{};
    std::map<VkDescriptorType, int> bindings;
    std::map<VkDescriptorType, std::atomic_int> bindingIds;
    VulkanDescriptorPool m_descriptorPool{};
    VulkanDescriptorSetLayout m_descriptorSetLayout{};
    VkDescriptorSet m_descriptorSet{};
    VulkanSampler m_defaultSampler;
};