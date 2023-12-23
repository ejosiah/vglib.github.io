#pragma once

#include "Plugin.hpp"

#include "VulkanRAII.h"
#include "Texture.h"
#include "VulkanBuffer.h"
#include "SequenceGenerator.hpp"
#include "VulkanInitializers.h"
#include <spdlog/spdlog.h>

#include <atomic>
#include <utility>
#include <span>

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

struct BindlessDescriptor {
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    std::map<VkDescriptorType, std::atomic_int> bindingIds;
    std::map<VkDescriptorType, int> bindings;
    const VulkanDevice* device;

    BindlessDescriptor(const VulkanDevice& device, VkDescriptorSet descriptorSet ,std::map<VkDescriptorType, int> bindings)
    : device(&device)
    , descriptorSet(descriptorSet) 
    , bindings(std::move(bindings))
    {
        bindingIds[VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER] = 0;
        bindingIds[VK_DESCRIPTOR_TYPE_STORAGE_IMAGE] = 0;
        bindingIds[VK_DESCRIPTOR_TYPE_STORAGE_BUFFER] = 0;
        bindingIds[VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER] = 0;
        bindingIds[VK_DESCRIPTOR_TYPE_SAMPLER] = 0;
        bindingIds[VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE] = 0;
        bindingIds[VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER] = 0;
        bindingIds[VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER] = 0;
    }

    int nextIndex(VkDescriptorType type) {
        return bindingIds[type]++;
    }

    BindlessTexture next(const Texture& texture, VkDescriptorType type) {
        return BindlessTexture{ &texture, type, nextIndex(type) };
    }

    BindlessBuffer next(const VulkanBuffer& buffer, VkDescriptorType type) {
        return BindlessBuffer{ buffer, type, nextIndex(type) };
    }

    void update(const BindlessTexture& bTexture) {
        assert(device != VK_NULL_HANDLE && descriptorSet != VK_NULL_HANDLE);
        
        auto texture = bTexture.texture;
        auto write = initializers::writeDescriptorSet();
        write.dstSet = descriptorSet;
        write.dstBinding = bindings[bTexture.type];
        write.descriptorCount = 1;
        write.dstArrayElement = bTexture.index;
        VkDescriptorImageInfo imageInfo{texture->sampler.handle, texture->imageView.handle, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        write.pImageInfo = &imageInfo;
        
        device->updateDescriptorSets(std::span{ &write, 1});
        
    }
    
    void update(const BindlessBuffer& bBuffer) {
        assert(device != VK_NULL_HANDLE && descriptorSet != VK_NULL_HANDLE);

        auto write = initializers::writeDescriptorSet();
        write.dstSet = descriptorSet;
        write.dstBinding = bindings[bBuffer.type];
        write.descriptorCount = 1;
        write.dstArrayElement = bBuffer.index;
        VkDescriptorBufferInfo bufferInfo{ bBuffer.buffer, 0, VK_WHOLE_SIZE };
        write.pBufferInfo = &bufferInfo;

        device->updateDescriptorSets(std::span{ &write, 1});
    }

    void update(std::span<BindlessTexture> textures) {
        assert(device != VK_NULL_HANDLE && descriptorSet != VK_NULL_HANDLE);

        std::sort(textures.begin(), textures.end(), [](const auto& t0, const auto& t1){ return t0.type < t1.type; });

        auto writes = initializers::writeDescriptorSets();
        std::vector<VkDescriptorImageInfo> imageInfos;

        for(auto& bTexture : textures) {
            auto texture = bTexture.texture;
            auto write = initializers::writeDescriptorSet();
            write.dstSet = descriptorSet;
            write.dstBinding = bindings[bTexture.type];
            write.descriptorCount = 1;
            write.dstArrayElement = bTexture.index;
            VkDescriptorImageInfo imageInfo{texture->sampler.handle, texture->imageView.handle, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            imageInfos.push_back(imageInfo);
            write.pImageInfo = &imageInfos[imageInfos.size() - 1];
        }

       device->updateDescriptorSets(writes);
    }

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
        return &m_indexingFeatures;
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

    BindlessDescriptor descriptorSet() const {
        return BindlessDescriptor{ device(), createDescriptorSet(), bindings};
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
    }

    VkDescriptorSet createDescriptorSet() const {
        return m_descriptorPool.allocate( { m_descriptorSetLayout }).front();
    }

private:
    static constexpr uint32_t MaxDescriptorResources = 1024u;
    static constexpr uint32_t TextureResourceBindingPoint = 10u;
    static constexpr uint32_t DescriptorPoolSize = 8;

    mutable VkPhysicalDeviceDescriptorIndexingFeatures m_indexingFeatures{};
    std::map<VkDescriptorType, int> bindings;
    VulkanDescriptorPool m_descriptorPool{};
    VulkanDescriptorSetLayout m_descriptorSetLayout{};
    VulkanSampler m_defaultSampler;
};