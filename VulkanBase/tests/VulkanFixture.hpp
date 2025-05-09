#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "VulkanInstance.h"
#include "VulkanDevice.h"
#include "VulkanShaderModule.h"
#include "VulkanBaseApp.h"
#include "ComputePipelins.hpp"
#include "VulkanExtensions.h"
#include "TestUtils.hpp"
#include "glsl_shaders.hpp"

#define NYI FAIL() << "Not yet implemented";

static std::vector<const char*> instanceExtensions{VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
static std::vector<const char*> validationLayers{};
static std::vector<const char*> deviceExtensions{ };

using namespace TestUtils;

class VulkanFixture : public ::testing::Test {
protected:
    VulkanInstance instance;
    VulkanDevice device;
    VulkanDebug debug;
    FileManager _fileManager;
    VulkanDescriptorPool descriptorPool;
    Settings settings;
    std::map<std::string, Pipeline> pipelines;
    uint32_t maxSets = 100;
    bool _autoCreatePipeline = true;

    void SetUp() override {
        spdlog::set_level(spdlog::level::warn);
        initVulkan();
        ext::init(instance);
        initFileManager();
        postVulkanInit();
        createPipelines();
    }

    void initVulkan(){
        settings.queueFlags = VK_QUEUE_COMPUTE_BIT;
        createInstance();
        debug = VulkanDebug{ instance };
        createDevice();
        createDescriptorPool();
    }

    void initFileManager(){
        _fileManager.addSearchPath(".");
        // TODO install resource files to avoid relative/absolute paths specifications
        _fileManager.addSearchPathFront("../../examples/fluid_dynamics");
        _fileManager.addSearchPathFront("../../examples/fluid_dynamics/spv");
        _fileManager.addSearchPathFront("../../data/shaders");
        _fileManager.addSearchPathFront("../../data/shaders/algorithm");
        _fileManager.addSearchPathFront("../../data");
        _fileManager.addSearchPathFront("../../data/shaders/test");
    }

    void createDescriptorPool(){
        std::array<VkDescriptorPoolSize, 10> poolSizes{
                {
                        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100 * maxSets},
                        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100 * maxSets},
                        { VK_DESCRIPTOR_TYPE_SAMPLER, 100 * maxSets },
                        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100 * maxSets },
                        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 100 * maxSets },
                        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 100 * maxSets },
                        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 100 * maxSets },
                        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 100 * maxSets },
                        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 100 * maxSets },
                        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100 * maxSets },

                }
        };
        descriptorPool = device.createDescriptorPool(maxSets, poolSizes, VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

    }

    void createInstance(){
        VkApplicationInfo appInfo{};
        appInfo.sType  = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.applicationVersion = VK_MAKE_VERSION(0, 0, 0);
        appInfo.pApplicationName = "Vulkan Test";
        appInfo.apiVersion = VK_API_VERSION_1_3;
        appInfo.pEngineName = "";

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
        createInfo.ppEnabledExtensionNames = instanceExtensions.data();

        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
        auto debugInfo = VulkanDebug::debugCreateInfo();
        createInfo.pNext = &debugInfo;

        instance = VulkanInstance{appInfo, {instanceExtensions, validationLayers}};
    }

    void createDevice(){
        VkPhysicalDeviceVulkan12Features features12{};
        features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        features12.hostQueryReset = VK_TRUE;

        VkPhysicalDeviceVulkan13Features features13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
        features13.maintenance4 = true;
        features13.synchronization2 = true;
        features12.pNext = &features13;

        auto pDevice = enumerate<VkPhysicalDevice>([&](uint32_t* size, VkPhysicalDevice* pDevice){
            return vkEnumeratePhysicalDevices(instance, size, pDevice);
        }).front();
        device = VulkanDevice{ instance, pDevice, settings};

        VkPhysicalDeviceFeatures enabledFeatures{};
        enabledFeatures.robustBufferAccess = VK_TRUE;
        device.createLogicalDevice(enabledFeatures, deviceExtensions, validationLayers, VK_NULL_HANDLE, VK_QUEUE_COMPUTE_BIT, &features12);
        vkDevice = device.logicalDevice;
    }

    template<typename Func>
    void execute(Func&& func){
        device.computeCommandPool().oneTimeCommand(func);
    }

    void createPipelines(){
        if(_autoCreatePipeline) {
            for (auto &metaData : pipelineMetaData()) {
                auto shaderModule = get(metaData.shadePath, &device);
                auto stage = initializers::shaderStage({ shaderModule, VK_SHADER_STAGE_COMPUTE_BIT});
                auto& sc = metaData.specializationConstants;
                VkSpecializationInfo specialization{COUNT(sc.entries), sc.entries.data(), sc.dataSize, sc.data };
                stage.pSpecializationInfo = &specialization;
                Pipeline pipeline;
                std::vector<VulkanDescriptorSetLayout> setLayouts;
                for(auto& layout : metaData.layouts){
                    setLayouts.push_back(*layout);
                }
                pipeline.layout = device.createPipelineLayout(setLayouts, metaData.ranges);

                auto createInfo = initializers::computePipelineCreateInfo();
                createInfo.stage = stage;
                createInfo.layout = pipeline.layout.handle;

                pipeline.pipeline = device.createComputePipeline(createInfo);
                device.setName<VK_OBJECT_TYPE_PIPELINE>(metaData.name, pipeline.pipeline.handle);
                pipelines.insert(std::make_pair(metaData.name, pipeline));
            }
        }
    }

    VulkanShaderModule get(std::variant<std::string, std::vector<uint32_t>>& shaderPath, VulkanDevice* device) {
        return std::visit(overloaded{
                [&](std::string path){ return device->createShaderModule( path ); },
                [&](std::vector<uint32_t> data){ return device->createShaderModule( data ); }
        }, shaderPath);
    }

    VkPipeline pipeline(const std::string& name){
        assert(pipelines.find(name) != end(pipelines));
        return pipelines[name].pipeline.handle;
    }

    VkPipelineLayout layout(const std::string& name){
        assert(pipelines.find(name) != end(pipelines));
        return pipelines[name].layout.handle;
    }

    virtual std::vector<PipelineMetaData> pipelineMetaData() {
        return {};
    }

    void autoCreatePipeline(bool val){
        _autoCreatePipeline = val;
    }

    virtual void postVulkanInit() {}

    template<typename Iter0, typename Iter1>
    void assertEqual(Iter0 _first0, Iter0 _last0, Iter1 _first1, Iter1 _last1){
        auto dist0 = std::distance(_first0, _last0);
        auto dist1 = std::distance(_first1, _last1);
        ASSERT_EQ(dist0, dist1);

        auto next0 = _first0;
        auto next1 = _first1;
        while(next0 != _last0){
            ASSERT_EQ(*next0, *next1);
            std::advance(*next0, 1);
            std::advance(*next1, 1);
        }
    }

    std::string resource(const std::string &name) {
        auto res = _fileManager.getFullPath(name);
        assert(res.has_value());
        return res->string();
    }

    template<typename T = uint32_t>
    VulkanBuffer entries(std::vector<T> data) {
        return entries(std::span{ data.data(), data.size()});
    }

    template<typename T = uint32_t>
    VulkanBuffer entries(std::span<T> span) const {
        return device.createCpuVisibleBuffer(span.data(), BYTE_SIZE(span), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }

    template<typename T = uint32_t>
    VulkanBuffer createBuffer(uint32_t size) const {
        return device.createBuffer(
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU, sizeof(T) * size);
    }
};