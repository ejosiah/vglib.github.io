#include <cassert>
#include "VulkanExtensions.h"


static PFN_vkCmdTraceRaysKHR pfn_vkCmdTraceRaysKHR = nullptr;
static PFN_vkCreateDebugUtilsMessengerEXT  pfn_createDebugUtilsMessenger = nullptr;
static PFN_vkDestroyDebugUtilsMessengerEXT pfn_destroyDebugUtilsMessenger = nullptr;
static PFN_vkGetAccelerationStructureBuildSizesKHR pfn_vkGetAccelerationStructureBuildSizesKHR = nullptr;
static PFN_vkCreateAccelerationStructureKHR pfn_vkCreateAccelerationStructureKHR = nullptr;
static PFN_vkDestroyAccelerationStructureKHR pfn_vkDestroyAccelerationStructureKHR = nullptr;
static PFN_vkCmdBuildAccelerationStructuresKHR pfn_vkCmdBuildAccelerationStructuresKHR = nullptr;
static PFN_vkGetAccelerationStructureDeviceAddressKHR pfn_vkGetAccelerationStructureDeviceAddressKHR = nullptr;
static PFN_vkGetRayTracingShaderGroupHandlesKHR pfn_vkGetRayTracingShaderGroupHandlesKHR = nullptr;
static PFN_vkCreateRayTracingPipelinesKHR pfn_vkCreateRayTracingPipelinesKHR = nullptr;
static PFN_vkSetDebugUtilsObjectNameEXT pfn_vkSetDebugUtilsObjectNameEXT = nullptr;
static PFN_vkGetSemaphoreWin32HandleKHR pfn_vkGetSemaphoreWin32HandleKHR = nullptr;
static PFN_vkCmdDrawMeshTasksEXT pfn_vkCmdDrawMeshTasksEXT = nullptr;
static PFN_vkCmdSetPolygonModeEXT pfn_vkCmdSetPolygonModeEXT = nullptr;
static PFN_vkCmdSetColorBlendEnableEXT pfn_vkCmdSetColorBlendEnableEXT = nullptr;

#ifdef WIN32
static PFN_vkGetMemoryWin32HandleKHR pfn_vkGetMemoryWin32HandleKHR = nullptr;
#endif



namespace ext {

    void init(VkInstance instance){
#ifdef DEBUG_MODE
        pfn_createDebugUtilsMessenger = procAddress<PFN_vkCreateDebugUtilsMessengerEXT>(instance, "vkCreateDebugUtilsMessengerEXT");
        pfn_destroyDebugUtilsMessenger = procAddress<PFN_vkDestroyDebugUtilsMessengerEXT>(instance, "vkDestroyDebugUtilsMessengerEXT");
        pfn_vkSetDebugUtilsObjectNameEXT = procAddress<PFN_vkSetDebugUtilsObjectNameEXT>(instance, "vkSetDebugUtilsObjectNameEXT");
#endif
        pfn_vkCmdTraceRaysKHR = procAddress<PFN_vkCmdTraceRaysKHR>(instance, "vkCmdTraceRaysKHR");
        pfn_vkGetAccelerationStructureBuildSizesKHR = procAddress<PFN_vkGetAccelerationStructureBuildSizesKHR>(instance, "vkGetAccelerationStructureBuildSizesKHR");
        pfn_vkCreateAccelerationStructureKHR = procAddress<PFN_vkCreateAccelerationStructureKHR>(instance, "vkCreateAccelerationStructureKHR");
        pfn_vkCmdBuildAccelerationStructuresKHR = procAddress<PFN_vkCmdBuildAccelerationStructuresKHR>(instance, "vkCmdBuildAccelerationStructuresKHR");
        pfn_vkGetAccelerationStructureDeviceAddressKHR = procAddress<PFN_vkGetAccelerationStructureDeviceAddressKHR>(instance, "vkGetAccelerationStructureDeviceAddressKHR");
        pfn_vkDestroyAccelerationStructureKHR = procAddress<PFN_vkDestroyAccelerationStructureKHR>(instance, "vkDestroyAccelerationStructureKHR");
        pfn_vkCreateRayTracingPipelinesKHR = procAddress<PFN_vkCreateRayTracingPipelinesKHR>(instance, "vkCreateRayTracingPipelinesKHR");
        pfn_vkGetRayTracingShaderGroupHandlesKHR = procAddress<PFN_vkGetRayTracingShaderGroupHandlesKHR>(instance, "vkGetRayTracingShaderGroupHandlesKHR");
        pfn_vkCmdDrawMeshTasksEXT = procAddress<PFN_vkCmdDrawMeshTasksEXT>(instance, "vkCmdDrawMeshTasksEXT");
        pfn_vkCmdSetPolygonModeEXT = procAddress<PFN_vkCmdSetPolygonModeEXT>(instance, "vkCmdSetPolygonModeEXT");
        pfn_vkCmdSetColorBlendEnableEXT = procAddress<PFN_vkCmdSetColorBlendEnableEXT>(instance, "vkCmdSetColorBlendEnableEXT");

#ifdef WIN32
        pfn_vkGetMemoryWin32HandleKHR = procAddress<PFN_vkGetMemoryWin32HandleKHR>(instance, "vkGetMemoryWin32HandleKHR");
        pfn_vkGetSemaphoreWin32HandleKHR = procAddress<PFN_vkGetSemaphoreWin32HandleKHR>(instance, "vkGetSemaphoreWin32HandleKHR");
#endif

    }

}

VKAPI_ATTR void VKAPI_CALL  vkCmdTraceRaysKHR(VkCommandBuffer commandBuffer, const VkStridedDeviceAddressRegionKHR* pRaygenShaderBindingTable, const VkStridedDeviceAddressRegionKHR* pMissShaderBindingTable, const VkStridedDeviceAddressRegionKHR* pHitShaderBindingTable, const VkStridedDeviceAddressRegionKHR* pCallableShaderBindingTable, uint32_t width, uint32_t height, uint32_t depth){
    assert(pfn_vkCmdTraceRaysKHR != nullptr);
    pfn_vkCmdTraceRaysKHR(commandBuffer, pRaygenShaderBindingTable, pMissShaderBindingTable, pHitShaderBindingTable, pCallableShaderBindingTable, width, height, depth);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateDebugUtilsMessengerEXT(
        VkInstance                                  instance,
        const VkDebugUtilsMessengerCreateInfoEXT*   pCreateInfo,
        const VkAllocationCallbacks*                pAllocator,
        VkDebugUtilsMessengerEXT*                   pMessenger){
    assert(pfn_createDebugUtilsMessenger);
    return pfn_createDebugUtilsMessenger(instance, pCreateInfo, pAllocator, pMessenger);
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDebugUtilsMessengerEXT(
        VkInstance                                  instance,
        VkDebugUtilsMessengerEXT                    messenger,
        const VkAllocationCallbacks*                pAllocator){
    assert(pfn_destroyDebugUtilsMessenger);
    return pfn_destroyDebugUtilsMessenger(instance, messenger, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL vkGetAccelerationStructureBuildSizesKHR(
        VkDevice                                    device,
        VkAccelerationStructureBuildTypeKHR         buildType,
        const VkAccelerationStructureBuildGeometryInfoKHR* pBuildInfo,
        const uint32_t*                             pMaxPrimitiveCounts,
        VkAccelerationStructureBuildSizesInfoKHR*   pSizeInfo){
    assert(pfn_vkGetAccelerationStructureBuildSizesKHR);
    pfn_vkGetAccelerationStructureBuildSizesKHR(device, buildType, pBuildInfo, pMaxPrimitiveCounts, pSizeInfo);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateAccelerationStructureKHR(
        VkDevice                                    device,
        const VkAccelerationStructureCreateInfoKHR* pCreateInfo,
        const VkAllocationCallbacks*                pAllocator,
        VkAccelerationStructureKHR*                 pAccelerationStructure){
    assert(pfn_vkCreateAccelerationStructureKHR);
    return pfn_vkCreateAccelerationStructureKHR(device, pCreateInfo, pAllocator, pAccelerationStructure);
}

VKAPI_ATTR void VKAPI_CALL vkCmdBuildAccelerationStructuresKHR(
        VkCommandBuffer                             commandBuffer,
        uint32_t                                    infoCount,
        const VkAccelerationStructureBuildGeometryInfoKHR* pInfos,
        const VkAccelerationStructureBuildRangeInfoKHR* const* ppBuildRangeInfos) {
    assert(pfn_vkCmdBuildAccelerationStructuresKHR);
    pfn_vkCmdBuildAccelerationStructuresKHR(commandBuffer, infoCount, pInfos, ppBuildRangeInfos);
}

VKAPI_ATTR VkDeviceAddress VKAPI_CALL vkGetAccelerationStructureDeviceAddressKHR(
        VkDevice                                    device,
        const VkAccelerationStructureDeviceAddressInfoKHR* pInfo){
        return pfn_vkGetAccelerationStructureDeviceAddressKHR(device, pInfo);
}

VKAPI_ATTR void VKAPI_CALL vkDestroyAccelerationStructureKHR(
        VkDevice                                    device,
        VkAccelerationStructureKHR                  accelerationStructure,
        const VkAllocationCallbacks*                pAllocator){
    assert(pfn_vkDestroyAccelerationStructureKHR);
    pfn_vkDestroyAccelerationStructureKHR(device, accelerationStructure, pAllocator);
}

VKAPI_ATTR VkResult VKAPI_CALL vkSetDebugUtilsObjectNameEXT(
        VkDevice                                    device,
        const VkDebugUtilsObjectNameInfoEXT*        pNameInfo){
    assert(pfn_vkSetDebugUtilsObjectNameEXT);
    return pfn_vkSetDebugUtilsObjectNameEXT(device, pNameInfo);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateRayTracingPipelinesKHR(
        VkDevice                                    device,
        VkDeferredOperationKHR                      deferredOperation,
        VkPipelineCache                             pipelineCache,
        uint32_t                                    createInfoCount,
        const VkRayTracingPipelineCreateInfoKHR*    pCreateInfos,
        const VkAllocationCallbacks*                pAllocator,
        VkPipeline*                                 pPipelines){
    assert(pfn_vkCreateRayTracingPipelinesKHR);
    return pfn_vkCreateRayTracingPipelinesKHR(device, deferredOperation, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetRayTracingShaderGroupHandlesKHR(
        VkDevice                                    device,
        VkPipeline                                  pipeline,
        uint32_t                                    firstGroup,
        uint32_t                                    groupCount,
        size_t                                      dataSize,
        void*                                       pData){

    assert(pfn_vkGetRayTracingShaderGroupHandlesKHR);
    return pfn_vkGetRayTracingShaderGroupHandlesKHR(device, pipeline, firstGroup, groupCount, dataSize, pData);

}

#ifdef WIN32
VKAPI_ATTR VkResult  VKAPI_CALL vkGetMemoryWin32HandleKHR(
        VkDevice                                    device,
        const VkMemoryGetWin32HandleInfoKHR*        pGetWin32HandleInfo,
        HANDLE*                                     pHandle){

        assert(pfn_vkGetMemoryWin32HandleKHR);
        return pfn_vkGetMemoryWin32HandleKHR(device, pGetWin32HandleInfo, pHandle);
}
#endif

VKAPI_ATTR VkResult VKAPI_CALL vkGetSemaphoreWin32HandleKHR(
        VkDevice                                    device,
        const VkSemaphoreGetWin32HandleInfoKHR*     pGetWin32HandleInfo,
        HANDLE*                                     pHandle){
    assert(pfn_vkGetSemaphoreWin32HandleKHR);
    return pfn_vkGetSemaphoreWin32HandleKHR(device, pGetWin32HandleInfo, pHandle);
}

VKAPI_ATTR void VKAPI_CALL vkCmdDrawMeshTasksEXT(
        VkCommandBuffer                             commandBuffer,
        uint32_t                                    groupCountX,
        uint32_t                                    groupCountY,
        uint32_t                                    groupCountZ){
    assert(pfn_vkCmdDrawMeshTasksEXT);
    return pfn_vkCmdDrawMeshTasksEXT(commandBuffer, groupCountX, groupCountY, groupCountZ);
}

VKAPI_ATTR void VKAPI_CALL vkCmdSetPolygonModeEXT(
        VkCommandBuffer                             commandBuffer,
        VkPolygonMode                               polygonMode) {

    assert(pfn_vkCmdSetPolygonModeEXT);
    return pfn_vkCmdSetPolygonModeEXT(commandBuffer, polygonMode);
}

VKAPI_ATTR void VKAPI_CALL vkCmdSetColorBlendEnableEXT(
        VkCommandBuffer                             commandBuffer,
        uint32_t                                    firstAttachment,
        uint32_t                                    attachmentCount,
        const VkBool32*                             pColorBlendEnables) {
    assert(pfn_vkCmdSetColorBlendEnableEXT);
    return pfn_vkCmdSetColorBlendEnableEXT(commandBuffer, firstAttachment, attachmentCount, pColorBlendEnables);
}