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
    uint32_t index{~0u};
    VkImageLayout imageLayout{VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
};

struct BindlessSampler {
    const VulkanSampler* sampler{};
    uint32_t index{~0u};
};

struct BindlessBuffer {
    VulkanBuffer buffer;
    VkDescriptorType type{};
    uint32_t index{~0u};
};

struct BindlessDescriptor {
    const VulkanDevice* device{};
    const VulkanDescriptorSetLayout* descriptorSetLayout{};
    VkDescriptorSet descriptorSet{};
    std::map<VkDescriptorType, std::atomic_int> bindingIds;
    std::map<VkDescriptorType, int> bindings;
    VulkanSampler* defaultSampler;

    BindlessDescriptor() = default;

    BindlessDescriptor(const VulkanDevice& device, const VulkanDescriptorSetLayout& descriptorSetLayout, VkDescriptorSet descriptorSet , VulkanSampler* sampler, std::map<VkDescriptorType, int> bindings, int reserveSlots = 0)
    : device(&device)
    , descriptorSetLayout(&descriptorSetLayout)
    , descriptorSet(descriptorSet) 
    , bindings(std::move(bindings))
    , defaultSampler(sampler)
    {
        bindingIds[VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER] = reserveSlots;
        bindingIds[VK_DESCRIPTOR_TYPE_STORAGE_IMAGE] = 0;
        bindingIds[VK_DESCRIPTOR_TYPE_SAMPLER] = 0;
        bindingIds[VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE] = 0;
        bindingIds[VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER] = 0;
        bindingIds[VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER] = 0;
    }

    uint32_t nextIndex(VkDescriptorType type) {
        return bindingIds[type]++;
    }

    BindlessTexture next(const Texture& texture, VkDescriptorType type) {
        return BindlessTexture{ &texture, type, nextIndex(type) };
    }

    int reserveSlots(VkDescriptorType type, int numSlots) {
        auto offset = bindingIds[type].load();
        bindingIds[type] += numSlots;
        return offset;
    }

    BindlessBuffer next(const VulkanBuffer& buffer, VkDescriptorType type) {
        return BindlessBuffer{ buffer, type, nextIndex(type) };
    }

    uint32_t update(const Texture& texture, VkDescriptorType type) {
        auto id = texture.bindingId == ~0u ? nextIndex(type) : texture.bindingId;
        update({ &texture, type, id});
        return id;
    }

    void update(const BindlessTexture& bTexture) {
        assert(bTexture.index <= bindingIds[bTexture.type]);
        assert(device != VK_NULL_HANDLE && descriptorSet != VK_NULL_HANDLE);


        auto texture = bTexture.texture;
        auto sampler = texture->sampler.handle ? texture->sampler.handle : defaultSampler->handle;
        auto write = initializers::writeDescriptorSet();
        write.dstSet = descriptorSet;
        write.dstBinding = bindings[bTexture.type];
        write.descriptorType = bTexture.type;
        write.descriptorCount = 1;
        write.dstArrayElement = bTexture.index;
        VkDescriptorImageInfo imageInfo{sampler, texture->imageView.handle, bTexture.imageLayout};
        write.pImageInfo = &imageInfo;
        
        device->updateDescriptorSets(std::span{ &write, 1});
    }

    void update(std::span<BindlessTexture> textures) {
        assert(device != VK_NULL_HANDLE && descriptorSet != VK_NULL_HANDLE);

        std::sort(textures.begin(), textures.end(), [](const auto& t0, const auto& t1){ return t0.type < t1.type; });

        auto writes = initializers::writeDescriptorSets();
        std::vector<VkDescriptorImageInfo> imageInfos;

        for(auto& bTexture : textures) {
            assert(bTexture.index <= bindingIds[bTexture.type]);
            auto texture = bTexture.texture;
            auto write = initializers::writeDescriptorSet();
            write.dstSet = descriptorSet;
            write.dstBinding = bindings[bTexture.type];
            write.descriptorCount = 1;
            write.descriptorType = bTexture.type;
            write.dstArrayElement = bTexture.index;
            VkDescriptorImageInfo imageInfo{texture->sampler.handle, texture->imageView.handle, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            imageInfos.push_back(imageInfo);
            write.pImageInfo = &imageInfos[imageInfos.size() - 1];
        }

       device->updateDescriptorSets(writes);
    }

    void update(BindlessSampler sampler) {
        assert(sampler.index != ~0u);

        auto writes = initializers::writeDescriptorSets<1>();

        writes[0].dstSet = descriptorSet;
        writes[0].dstBinding = bindings[VK_DESCRIPTOR_TYPE_SAMPLER];
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        writes[0].descriptorCount = 1;
        writes[0].dstArrayElement = sampler.index;
        VkDescriptorImageInfo samplerInfo{sampler.sampler->handle, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
        writes[0].pImageInfo = &samplerInfo;

        device->updateDescriptorSets(writes);

    }

};

class BindLessDescriptorPlugin : public Plugin {
public:
    static constexpr uint32_t MaxDescriptorResources = 1024;
    static constexpr uint32_t TextureResourceBindingPoint = 10u;
    static constexpr uint32_t DescriptorPoolSize = 6;

    BindLessDescriptorPlugin()
    {
        m_Vulkan11Features.shaderDrawParameters = VK_TRUE;
        m_Vulkan11Features.pNext = &m_Vulkan12Features;

        m_Vulkan12Features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
        m_Vulkan12Features.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
        m_Vulkan12Features.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
        m_Vulkan12Features.descriptorIndexing = VK_TRUE;
        m_Vulkan12Features.descriptorBindingPartiallyBound = VK_TRUE;
        m_Vulkan12Features.descriptorBindingPartiallyBound = VK_TRUE;
        m_Vulkan12Features.runtimeDescriptorArray = VK_TRUE;
        m_Vulkan12Features.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
        m_Vulkan12Features.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
        m_Vulkan12Features.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
        m_Vulkan12Features.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE;
        m_Vulkan12Features.descriptorBindingUniformTexelBufferUpdateAfterBind = VK_TRUE;
        m_Vulkan12Features.descriptorBindingStorageTexelBufferUpdateAfterBind = VK_TRUE;

    }

    ~BindLessDescriptorPlugin() override = default;

    [[nodiscard]]
    std::vector<const char *> deviceExtensions() const override {
        return { VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME };
    }

    void *appendTo(void *nextChain) const override {
        m_Vulkan12Features.pNext = nextChain;
        return &m_Vulkan11Features;
    }

    void preInit() override {
        Plugin::preInit();
    }

    void init() override {
        setBindingPoints();
        createDescriptorPool();
        createDescriptorSetLayout();
        createDefaultSampler();
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

    BindlessDescriptor descriptorSet(int reserveSlots = 0) const {
        return BindlessDescriptor{ device(), m_descriptorSetLayout, createDescriptorSet(), &m_defaultSampler, bindings, reserveSlots};
    }

    VulkanDescriptorSetLayout& descriptorSetLayout() {
        return m_descriptorSetLayout;
    }

    BindLessDescriptorPlugin& addBindings(const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
        for(const auto& binding : bindings) {
            addBinding(binding);
        }
        return *this;
    }

    BindLessDescriptorPlugin& addBinding(const VkDescriptorSetLayoutBinding& binding) {
        assert(binding.binding < TextureResourceBindingPoint || binding.binding > (TextureResourceBindingPoint + 6));
        m_additionalBindings.push_back(binding);
        return *this;
    }

    void createDescriptorSetLayout() {
        auto nextBinding = sequence(TextureResourceBindingPoint);
        auto bindings =
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
                        .descriptorType(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
                        .descriptorCount(MaxDescriptorResources)
                        .shaderStages(VK_SHADER_STAGE_ALL)
                    .binding(nextBinding())
                        .descriptorType(VK_DESCRIPTOR_TYPE_SAMPLER)
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
                    .build();

        for(const auto& binding : m_additionalBindings) {
            bindings.push_back(binding);
        }

        std::vector<VkDescriptorBindingFlags> bindlessFlags(bindings.size());
        VkDescriptorSetLayoutBindingFlagsCreateInfo extendedInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
        auto flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        std::generate(bindlessFlags.begin(), bindlessFlags.end(), []{
            return VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
        });
        extendedInfo.bindingCount = COUNT(bindlessFlags);
        extendedInfo.pBindingFlags = bindlessFlags.data();

        m_descriptorSetLayout = device().createDescriptorSetLayout(bindings, flags, &extendedInfo);
        device().setName<VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT>("bind_less_descriptor_set_layout", m_descriptorSetLayout.handle);

    }


protected:
    void setBindingPoints() {
        auto nextBinding = sequence(int(TextureResourceBindingPoint));
        bindings[VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER] = nextBinding();
        bindings[VK_DESCRIPTOR_TYPE_STORAGE_IMAGE] = nextBinding();
        bindings[VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE] = nextBinding();
        bindings[VK_DESCRIPTOR_TYPE_SAMPLER] = nextBinding();
        bindings[VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER] = nextBinding();
        bindings[VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER] = nextBinding();

    }
    
    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, DescriptorPoolSize> poolSizes{
                {
                        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MaxDescriptorResources},
                        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, MaxDescriptorResources},
                        { VK_DESCRIPTOR_TYPE_SAMPLER, MaxDescriptorResources },
                        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, MaxDescriptorResources },
                        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, MaxDescriptorResources },
                        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, MaxDescriptorResources },
                }
        };

        constexpr uint32_t maxSets = MaxDescriptorResources * DescriptorPoolSize;
        m_descriptorPool = device().createDescriptorPool(maxSets, poolSizes, VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT);
    }

    VkDescriptorSet createDescriptorSet() const {
        return m_descriptorPool.allocate( { m_descriptorSetLayout }).front();
    }

    void createDefaultSampler() {
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.maxLod = 1;

        m_defaultSampler = device().createSampler(samplerInfo);
    }

private:
    mutable VkPhysicalDeviceVulkan11Features  m_Vulkan11Features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
    mutable VkPhysicalDeviceVulkan12Features  m_Vulkan12Features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    std::map<VkDescriptorType, int> bindings;
    mutable VulkanSampler m_defaultSampler;
    VulkanDescriptorPool m_descriptorPool{};
    VulkanDescriptorSetLayout m_descriptorSetLayout{};
    std::vector<VkDescriptorSetLayoutBinding> m_additionalBindings;
};
