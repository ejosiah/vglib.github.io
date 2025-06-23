#include "VulkanRayQuerySupport.hpp"
#include "ExtensionChain.hpp"

void VulkanRayQuerySupport::enableRayQuery() {
    auto self = dynamic_cast<VulkanBaseApp*>(this);
    assert(self);
    auto& app = *self;

    if(!app.device.extensionSupported(VK_KHR_RAY_QUERY_EXTENSION_NAME)){
        throw std::runtime_error{ "Ray query not supported by your device" };
    }

    app.deviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    app.deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    app.deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    app.deviceExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);

    auto bufferDeviceAddressFeatures = findExtension<VkPhysicalDeviceBufferDeviceAddressFeatures>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES, app.deviceCreateNextChain);
    bufferDeviceAddressFeatures->bufferDeviceAddress = VK_TRUE;
    bufferDeviceAddressFeatures->pNext = app.deviceCreateNextChain;

    auto accelerationStructureFeatures = findExtension<VkPhysicalDeviceAccelerationStructureFeaturesKHR>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, app.deviceCreateNextChain);
    accelerationStructureFeatures->accelerationStructure = VK_TRUE;
    accelerationStructureFeatures->pNext = &bufferDeviceAddressFeatures;

    auto rayQueryFeatures = findExtension<VkPhysicalDeviceRayQueryFeaturesKHR>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR, app.deviceCreateNextChain);
    rayQueryFeatures->rayQuery = VK_TRUE;
    rayQueryFeatures->pNext = &accelerationStructureFeatures;

}
