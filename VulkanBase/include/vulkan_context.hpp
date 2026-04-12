#pragma once

#include "VulkanDevice.h"
#include "VulkanInstance.h"
#include "VulkanExtensions.h"
#include "VulkanDevice.h"
#include "filemanager.hpp"
#include <memory>
#include <vector>

struct ContextCreateInfo{
    VkApplicationInfo applicationInfo{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "No Name",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "",
        .apiVersion = VK_API_VERSION_1_3,
    };
    ExtensionsAndValidationLayers instanceExtAndLayers{
            {VK_EXT_DEBUG_UTILS_EXTENSION_NAME}
    };
    ExtensionsAndValidationLayers deviceExtAndLayers;
    Settings settings;
    VkSurfaceKHR surface{ VK_NULL_HANDLE };
    VkPhysicalDeviceFeatures enabledFeature{};
    void* deviceCreateNextChain{ nullptr };
    std::vector<fs::path>  searchPaths{};
};


class VulkanContext{
public:
    VulkanContext() = default;

    explicit VulkanContext(ContextCreateInfo createInfo);

    ~VulkanContext();

    VulkanInstance instance{};
    VulkanDevice device{};
    VulkanExtensions extensions{};
    VulkanDebug vulkanDebug{};
    FileManager fileManager;

    void init();

    std::string resource(const std::string& path);

private:
    class Impl;
    Impl* pimpl{ nullptr };

};