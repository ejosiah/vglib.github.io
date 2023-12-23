#include "VulkanRayTraceBaseApp.hpp"

VulkanRayTraceBaseApp::VulkanRayTraceBaseApp(std::string_view name, const Settings &settings, std::vector<std::unique_ptr<Plugin>> plugins)
    : VulkanBaseApp(name, settings, std::move(plugins))
{
    // Add raytracing device extensions
    deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    deviceExtensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);

    // Enable features required for ray tracing using feature chaining via pNext
    enabledBufferDeviceAddressFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    enabledBufferDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;

    enabledRayTracingPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    enabledRayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;
    enabledRayTracingPipelineFeatures.pNext = &enabledBufferDeviceAddressFeatures;

    enabledAccelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    enabledAccelerationStructureFeatures.accelerationStructure = VK_TRUE;
    enabledAccelerationStructureFeatures.pNext = &enabledRayTracingPipelineFeatures;

    enabledDescriptorIndexingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
    enabledDescriptorIndexingFeatures.pNext = &enabledAccelerationStructureFeatures;
    enabledDescriptorIndexingFeatures.runtimeDescriptorArray = VK_TRUE;
    enabledDescriptorIndexingFeatures.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;

    deviceCreateNextChain = &enabledDescriptorIndexingFeatures;
}

void VulkanRayTraceBaseApp::postVulkanInit() {
    loadRayTracingProperties();
    rtBuilder = rt::AccelerationStructureBuilder{&device};
}

void VulkanRayTraceBaseApp::clearAccelerationStructure() {
    rtBuilder.dispose();
    rtBuilder = rt::AccelerationStructureBuilder{&device};
}

void VulkanRayTraceBaseApp::loadRayTracingProperties() {
    rayTracingPipelineProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &rayTracingPipelineProperties;

    vkGetPhysicalDeviceProperties2(device, &properties);
}

void VulkanRayTraceBaseApp::createAccelerationStructure(const std::vector<rt::MeshObjectInstance>& drawableInstances) {
    if(drawableInstances.empty()) return;

    sceneObjects = rtBuilder.add(drawableInstances);
    asInstances = rtBuilder.buildTlas();
    VkDeviceSize size = sizeof(rt::ObjectInstance);
    auto stagingBuffer = device.createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY, size * sceneObjects.size());

    std::vector<rt::ObjectInstance> sceneDesc;
    sceneDesc.reserve(sceneObjects.size());
    for(auto& instanceGroup : sceneObjects){
        for(auto& instance : instanceGroup.objectInstances){
            sceneDesc.push_back(instance);
        }
    }
    // FIXME use ping pong or barrier as sceneObjectBuffer probably in use
    sceneObjectBuffer = device.createDeviceLocalBuffer(sceneDesc.data(), size * sceneDesc.size(), VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
}

void
VulkanRayTraceBaseApp::createShaderBindingTable(ShaderBindingTable &shaderBindingTable, void *shaderHandleStoragePtr,
                                                VkBufferUsageFlags usageFlags, VmaMemoryUsage memoryUsage,
                                                uint32_t handleCount) {
    const auto [handleSize, _] = getShaderGroupHandleSizingInfo();

    VkDeviceSize size = handleSize * handleCount;
    auto stagingBuffer = device.createStagingBuffer(size);
    stagingBuffer.copy(shaderHandleStoragePtr, size);

    shaderBindingTable.buffer = device.createBuffer(usageFlags | VK_BUFFER_USAGE_TRANSFER_DST_BIT, memoryUsage, size);
    device.copy(stagingBuffer, shaderBindingTable.buffer, size, 0, 0);

    shaderBindingTable.stridedDeviceAddressRegion = getSbtEntryStridedDeviceAddressRegion(shaderBindingTable.buffer, handleCount);

}

VkStridedDeviceAddressRegionKHR
VulkanRayTraceBaseApp::getSbtEntryStridedDeviceAddressRegion(const VulkanBuffer &buffer, uint32_t handleCount) const {
    const auto [_, handleSizeAligned] = getShaderGroupHandleSizingInfo();
    VkStridedDeviceAddressRegionKHR stridedDeviceAddressRegion{};
    stridedDeviceAddressRegion.deviceAddress = device.getAddress(buffer);
    stridedDeviceAddressRegion.stride = handleSizeAligned;
    stridedDeviceAddressRegion.size = handleSizeAligned * handleCount;

    return stridedDeviceAddressRegion;
}

std::tuple<uint32_t, uint32_t> VulkanRayTraceBaseApp::getShaderGroupHandleSizingInfo() const {
    const uint32_t handleSize = rayTracingPipelineProperties.shaderGroupHandleSize;
    const uint32_t handleSizeAligned = alignedSize(handleSize, rayTracingPipelineProperties.shaderGroupHandleAlignment);

    return std::make_tuple(handleSize, handleSizeAligned);
}

VulkanRayTraceBaseApp::~VulkanRayTraceBaseApp() {

}

void VulkanRayTraceBaseApp::framebufferReady() {
    canvas = std::move(Canvas{this, VK_IMAGE_USAGE_STORAGE_BIT});
    canvas.init();
}

void VulkanRayTraceBaseApp::accelerationStructureBuildBarrier(VkCommandBuffer commandBuffer) {
//    VkBufferMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
//    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
//    barrier.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
//    barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
//    barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
//    barrier.buffer = rtBuilder.topLevelAs().buffer;
//    barrier.offset = 0;
//    barrier.size = rtBuilder.topLevelAs().buffer.size;
//
//    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
//    dependency.bufferMemoryBarrierCount = 1;
//    dependency.pBufferMemoryBarriers = &barrier;
//    vkCmdPipelineBarrier2(commandBuffer, &dependency);

    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    barrier.buffer = rtBuilder.topLevelAs().buffer;
    barrier.offset = 0;
    barrier.size = rtBuilder.topLevelAs().buffer.size;
    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                         0, 0, VK_NULL_HANDLE,
                         1, &barrier, 0, VK_NULL_HANDLE);
}