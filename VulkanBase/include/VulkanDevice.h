#pragma once

#include "common.h"
#include "Settings.hpp"
#include "VulkanBuffer.h"
#include "VulkanRAII.h"
#include "VulkanSemaphore.hpp"
#include "VulkanDescriptorSet.h"
#include "VulkanCommandBuffer.h"
#include "VulkanImage.h"
#include "VulkanFence.h"
#include "VulkanRenderPass.h"
#include "VulkanFramebuffer.h"
#include "VulkanDebug.h"
#include "VulkanExtensions.h"
#include "builder_forwards.hpp"
#include <bitset>
#include "VulkanShaderModule.h"

#include <span>

class VulkanImageOps;

// TODO make shared
struct VulkanDevice{

    using DeviceDisposeListener = std::function<void(const VulkanDevice&)>;

    std::vector<DeviceDisposeListener> disposeListeners;

    struct {
        std::optional<uint32_t> graphics;
        std::optional<uint32_t> compute;
        std::optional<uint32_t> transfer;
        std::optional<uint32_t> present;
    } queueFamilyIndex;

    struct {
        VkQueue graphics;
        VkQueue compute;
        VkQueue transfer;
        VkQueue present;
    } queues{};

    std::set<uint32_t> uniqueQueueIndices;

    DISABLE_COPY(VulkanDevice)

    VulkanDevice() = default;

    explicit VulkanDevice(VkInstance instance, VkPhysicalDevice pDevice, const Settings& settings)
    : instance(instance)
    , physicalDevice(pDevice)
    , settings(settings)
    {
    }

    VulkanDevice(VulkanDevice&& source) noexcept{
        operator=(static_cast<VulkanDevice&&>(source));
    }

    VulkanDevice& operator=(VulkanDevice&& source) noexcept{
        physicalDevice = std::exchange(source.physicalDevice, nullptr);
        logicalDevice = std::exchange(source.logicalDevice, nullptr);
        queueFamilyIndex = std::exchange(source.queueFamilyIndex, {});
        queues = std::exchange(source.queues, {});
        allocator = std::exchange(source.allocator, nullptr);
        instance = std::exchange(source.instance, nullptr);
        settings = std::exchange(source.settings, {});

        return *this;
    }

    ~VulkanDevice(){
        if(logicalDevice){
            for(const auto& listener : disposeListeners){
                listener(*this);
            }
            for(auto& [_, commandPool] : commandPools){
                dispose(commandPool);
            }
            vmaDestroyAllocator(allocator);
            vkDestroyDevice(logicalDevice, nullptr);
        }
    }

    inline void registerDisposeListener(DeviceDisposeListener&& listener){
        disposeListeners.push_back(listener);
    }

    inline void initQueueFamilies(VkQueueFlags queueFlags, VkSurfaceKHR surface = VK_NULL_HANDLE){
        auto queueFamilies = getQueueFamilyProperties();
        for(uint32_t i = 0; i < queueFamilies.size(); i++){
            if(!queueFamilyIndex.graphics && (queueFamilies[i].queueFlags & queueFlags) && (queueFlags & VK_QUEUE_GRAPHICS_BIT) == VK_QUEUE_GRAPHICS_BIT){
                queueFamilyIndex.graphics = i;
                uniqueQueueIndices.insert(i);
            }
            if(!queueFamilyIndex.compute && (queueFamilies[i].queueFlags & queueFlags) && (queueFlags & VK_QUEUE_COMPUTE_BIT) == VK_QUEUE_COMPUTE_BIT){
               queueFamilyIndex.compute = i;
                uniqueQueueIndices.insert(i);
            }
            if(!queueFamilyIndex.transfer && (queueFamilies[i].queueFlags & queueFlags) && (queueFlags & VK_QUEUE_TRANSFER_BIT) == VK_QUEUE_TRANSFER_BIT){
                queueFamilyIndex.transfer = i;
                uniqueQueueIndices.insert(i);
            }

            if(surface && !queueFamilyIndex.present) {
                VkBool32 present;
                vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &present);
                if (present) {
                    queueFamilyIndex.present = i;
                }
            }
        }
        if(settings.uniqueQueueFlags){
            if(settings.uniqueQueueFlags & VK_QUEUE_COMPUTE_BIT){
                if(auto queueIndex = findQueueFamily(VK_QUEUE_COMPUTE_BIT)){
                    queueFamilyIndex.compute = *queueIndex;
                    uniqueQueueIndices.insert(*queueIndex);
                }
            }
            if(settings.uniqueQueueFlags & VK_QUEUE_TRANSFER_BIT){
                if(auto queueIndex = findQueueFamily(VK_QUEUE_TRANSFER_BIT)){
                    queueFamilyIndex.transfer = *queueIndex;
                    uniqueQueueIndices.insert(*queueIndex);
                }

            }
        }
    }

    inline std::optional<uint32_t> findQueueFamily(VkQueueFlagBits queueFlagBits) const {

        auto queueFamilies = getQueueFamilyProperties();

        std::vector<std::pair<uint32_t, VkQueueFamilyProperties>> matches;
        for(auto i = 0; i < queueFamilies.size(); i++){
            if(queueFamilies[i].queueFlags & queueFlagBits){
                matches.emplace_back(i, queueFamilies[i]);
            }
        }

        std::sort(matches.begin(), matches.end(), [](const auto& lhs, const auto& rhs){
            return
                    std::bitset<32>{lhs.second.queueFlags}.count() < std::bitset<32>{rhs.second.queueFlags}.count();
        });

        return !matches.empty() ?  std::optional<uint32_t>{matches.front().first} : std::nullopt;
    }

    inline void createLogicalDevice(const VkPhysicalDeviceFeatures& enabledFeatures,
                                    const std::vector<const char*>& enabledExtensions,
                                    const std::vector<const char*>& enabledLayers = {},
                                    VkSurfaceKHR surface = VK_NULL_HANDLE,
                                    VkQueueFlags queueFlags = VK_QUEUE_GRAPHICS_BIT,
                                    void* pNext = VK_NULL_HANDLE){
                initQueueFamilies(queueFlags, surface);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        for(auto queueIndex : uniqueQueueIndices){
            std::array<float, 2> priorities{ 1.0f, 1.0f};
            VkDeviceQueueCreateInfo qCreateInfo{};
            qCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            qCreateInfo.queueFamilyIndex = queueIndex;
            qCreateInfo.queueCount = 2;
            qCreateInfo.pQueuePriorities = priorities.data();
            queueCreateInfos.push_back((qCreateInfo));
        }

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pNext = pNext;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.enabledLayerCount = static_cast<uint32_t>(enabledLayers.size());
        createInfo.ppEnabledLayerNames = enabledLayers.data();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
        createInfo.ppEnabledExtensionNames = enabledExtensions.data();
        createInfo.pEnabledFeatures = &enabledFeatures;

        ERR_GUARD_VULKAN(vkCreateDevice(physicalDevice, &createInfo, nullptr, &logicalDevice));
        initQueues();

        auto deviceAddressExtensionEnabled = std::any_of(begin(enabledExtensions), end(enabledExtensions), [](auto& ext){
            return std::strcmp(ext, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) == 0;
        });

        VmaAllocatorCreateInfo allocatorInfo{};
        if(deviceAddressExtensionEnabled){
            allocatorInfo.flags |=  VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        }

        allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
        allocatorInfo.instance = instance;
        allocatorInfo.physicalDevice = physicalDevice;
        allocatorInfo.device = logicalDevice;

        ERR_GUARD_VULKAN(vmaCreateAllocator(&allocatorInfo, &allocator));


        auto memoryTypeCount = getMemoryProperties().memoryTypeCount;
        std::vector<VkExternalMemoryHandleTypeFlags> externalMemoryHandleTypeFlags(memoryTypeCount);
#ifdef WIN32
        std::generate(externalMemoryHandleTypeFlags.begin(), externalMemoryHandleTypeFlags.end(), []{
            return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        });
#else
        std::generate(externalMemoryHandleTypeFlags.begin(), externalMemoryHandleTypeFlags.end(), []{
            return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        });
#endif
        allocatorInfo.pTypeExternalMemoryHandleTypes = externalMemoryHandleTypeFlags.data();
        ERR_GUARD_VULKAN(vmaCreateAllocator(&allocatorInfo, &exportableMemoryAllocator));


        VulkanCommandPool  commandPool;
        if(queueFamilyIndex.graphics) {
            commandPool = createCommandPool(*queueFamilyIndex.graphics, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, 1); // TODO get num queues from settings
            commandPools.emplace(std::make_pair(*queueFamilyIndex.graphics, std::move(commandPool)));
        }

        if(queueFamilyIndex.compute) {
            commandPool = createCommandPool(*queueFamilyIndex.compute, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
            commandPools.emplace(std::make_pair(*queueFamilyIndex.compute, std::move(commandPool)));
        }

        if(queueFamilyIndex.transfer) {
            commandPool = createCommandPool(*queueFamilyIndex.transfer, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
            commandPools.emplace(std::make_pair(*queueFamilyIndex.transfer, std::move(commandPool)));
        }
        assert(!commandPools.empty());
    }

    inline void initQueues(){
        if(queueFamilyIndex.graphics.has_value()){
            vkGetDeviceQueue(logicalDevice, *queueFamilyIndex.graphics, 0, &queues.graphics);
        }
        if(queueFamilyIndex.compute.has_value()) {
            vkGetDeviceQueue(logicalDevice, *queueFamilyIndex.compute, 0, &queues.compute);
        }
        if(queueFamilyIndex.transfer.has_value()) {
            vkGetDeviceQueue(logicalDevice, *queueFamilyIndex.transfer, 0, &queues.transfer);
        }
        if(queueFamilyIndex.present.has_value()) {
            vkGetDeviceQueue(logicalDevice, *queueFamilyIndex.present, 0, &queues.present);
        }
    }

    [[nodiscard]]
    std::string name() const {
        return getProperties().deviceName;
    }

    [[nodiscard]]
    inline uint32_t score() const {
        auto deviceProps = getProperties();
        uint32_t score = deviceProps.limits.maxImageDimension2D;
        if(deviceProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU){
            score += 1000;
        }else if(deviceProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU){
            score += 500;
        }else if(deviceProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU){
            score += 100;
        }
        return score;
    }

    [[nodiscard]]
    inline VkPhysicalDeviceProperties getProperties() const{
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        return props;
    }

    [[nodiscard]]
    inline std::vector<VkQueueFamilyProperties> getQueueFamilyProperties() const {
       return get<VkQueueFamilyProperties>([&](uint32_t* size, VkQueueFamilyProperties* propsPtr){
          vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, size, propsPtr);
       });
    }

    std::vector<VkExtensionProperties> getExtensions(const char* layer = nullptr) const {
        return enumerate<VkExtensionProperties>([&](uint32_t* count, VkExtensionProperties* ptr){
            return vkEnumerateDeviceExtensionProperties(physicalDevice, layer, count, ptr);
        });
    }

    [[nodiscard]] VkPhysicalDeviceMemoryProperties getMemoryProperties() const {
        VkPhysicalDeviceMemoryProperties memoryProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
        return memoryProperties;
    }

    inline VkPhysicalDeviceLimits getLimits() const {
        return getProperties().limits;
    }

    inline VkSampleCountFlagBits getMaxUsableSampleCount() const {
        auto counts = getLimits().framebufferColorSampleCounts
                        | (settings.depthTest ? getLimits().framebufferDepthSampleCounts : 0);

        if(counts & VK_SAMPLE_COUNT_64_BIT){
            return VK_SAMPLE_COUNT_64_BIT;
        }
        if(counts & VK_SAMPLE_COUNT_16_BIT){
            return VK_SAMPLE_COUNT_16_BIT;
        }
        if(counts & VK_SAMPLE_COUNT_8_BIT){
            return VK_SAMPLE_COUNT_8_BIT;
        }
        if(counts & VK_SAMPLE_COUNT_4_BIT){
            return VK_SAMPLE_COUNT_4_BIT;
        }
        if(counts & VK_SAMPLE_COUNT_2_BIT){
            return VK_SAMPLE_COUNT_2_BIT;
        }

        return VK_SAMPLE_COUNT_1_BIT;
    }

    [[nodiscard]]
    bool supportsMemoryType(VkMemoryPropertyFlags flags) const {
        auto memoryProps = getMemoryProperties();
        for(auto i = 0; i < memoryProps.memoryTypeCount; i++){
            if((memoryProps.memoryTypes[i].propertyFlags & flags)){
                return true;
            }
        }
        return false;
    }


    inline bool extensionSupported(const char* extension) const noexcept {
        auto extensions = getExtensions();
        return std::any_of(begin(extensions), end(extensions), [&](auto& ext){
            return strcmp(extension, ext.extensionName) == 0;
        });
    }


    inline VulkanBuffer createDeviceLocalBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage, std::set<uint32_t> queueIndices = {}) const {
        // TODO use transfer queue and then transfer ownership
        VulkanBuffer stagingBuffer = createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY, size, "", queueIndices);
        stagingBuffer.copy(data, size);

        usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        VulkanBuffer buffer = createBuffer(usage, VMA_MEMORY_USAGE_GPU_ONLY, size, "", queueIndices);

        auto qfIndex = queueFamilyIndex.transfer.has_value() ? queueFamilyIndex.transfer : queueFamilyIndex.graphics;
        qfIndex = qfIndex.has_value() ? qfIndex : queueFamilyIndex.compute;
        assert(qfIndex.has_value());

        commandPoolFor(*qfIndex).oneTimeCommand([&](auto cmdBuffer){
            VkBufferCopy copy{};
            copy.size = size;
            copy.dstOffset = 0;
            copy.srcOffset = 0;
            vkCmdCopyBuffer(cmdBuffer, stagingBuffer, buffer, 1u, &copy);

            VkAccessFlags dstAccessMask = (usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)
                                                ? VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT : VK_ACCESS_INDEX_READ_BIT;

            VkBufferMemoryBarrier barrier = initializers::bufferMemoryBarrier();
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = 0;
            barrier.srcQueueFamilyIndex = *qfIndex;
            barrier.dstQueueFamilyIndex = queueFamilyIndex.graphics.has_value() ? *queueFamilyIndex.graphics : *qfIndex;
            barrier.buffer = buffer;
            barrier.offset = 0;
            barrier.size = buffer.size;

            vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                    , 0, 0, VK_NULL_HANDLE, 1, &barrier, 0, VK_NULL_HANDLE);

        });

        return buffer;

    }

    inline VulkanBuffer createDeviceLocalBuffer(const VulkanBuffer& source, VkBufferUsageFlags usage, std::set<uint32_t> queueIndices = {}) const {
        usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        auto size = source.size;
        VulkanBuffer buffer = createBuffer(usage, VMA_MEMORY_USAGE_GPU_ONLY, size, "", queueIndices);

        commandPoolFor(*this->queueFamilyIndex.graphics).oneTimeCommand( [&](auto cmdBuffer){
            VkBufferCopy copy{};
            copy.size = size;
            copy.dstOffset = 0;
            copy.srcOffset = 0;
            vkCmdCopyBuffer(cmdBuffer, source, buffer, 1u, &copy);
        });

        return buffer;
    }

    inline void copy(const VulkanBuffer& source, const VulkanBuffer& destination, VkDeviceSize size, VkDeviceSize srcOffset = 0u, VkDeviceSize dstOffset = 0u) const {
        commandPoolFor(*findFirstActiveQueue()).oneTimeCommand([&](auto cmdBuffer){
            VkBufferCopy copy{};
            copy.size = size;
            copy.srcOffset = srcOffset;
            copy.dstOffset = dstOffset;
            vkCmdCopyBuffer(cmdBuffer, source, destination, 1u, &copy);
        });
    }

    inline VulkanBuffer createCpuVisibleBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage, std::set<uint32_t> queueIndices = {}) const {

        VulkanBuffer buffer = createBuffer(usage, VMA_MEMORY_USAGE_CPU_TO_GPU, size, "", queueIndices);
        buffer.copy(data, size);
        return buffer;
    }

    inline VulkanBuffer createStagingBuffer(VkDeviceSize size, std::set<uint32_t> queueIndices = {}) const {
        return createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY, size, "StagingBuffer", queueIndices);
    }

    [[nodiscard]]
    inline VulkanBuffer createBuffer(VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage, VkDeviceSize size, const std::string name, std::set<uint32_t> queueIndices, VmaAllocator allocator, void* next = VK_NULL_HANDLE) const {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.pNext = next;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        std::vector<uint32_t> pIndices{queueIndices.begin(), queueIndices.end()};
        if(pIndices.size() > 1){
            bufferInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
            bufferInfo.queueFamilyIndexCount = pIndices.size();
            bufferInfo.pQueueFamilyIndices = pIndices.data();
        }else{
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }
        VkBuffer buffer;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = memoryUsage;
        VmaAllocation allocation;

        ERR_GUARD_VULKAN(vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr));

#ifdef DEBUG_MODE
        if(!name.empty()){
            //   VulkanDebug::setObjectName(logicalDevice, buffer, name);
            VkDebugUtilsObjectNameInfoEXT s{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, nullptr, VK_OBJECT_TYPE_BUFFER, (uint64_t)buffer, name.c_str()};
            auto SetDebugUtilsObjectName = procAddress<PFN_vkSetDebugUtilsObjectNameEXT>(instance, "vkSetDebugUtilsObjectNameEXT");
            SetDebugUtilsObjectName(logicalDevice, &s);
        }
#endif
        bool mappable = memoryUsage == VMA_MEMORY_USAGE_CPU_ONLY || memoryUsage == VMA_MEMORY_USAGE_CPU_TO_GPU  || memoryUsage == VMA_MEMORY_USAGE_GPU_TO_CPU;
        return VulkanBuffer{ allocator, buffer, allocation, size, name, mappable };
    }

    [[nodiscard]]
    inline VulkanBuffer createBuffer(VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage, VkDeviceSize size, const std::string name = "", std::set<uint32_t> queueIndices = {}) const {
        return createBuffer(usage, memoryUsage, size, name, queueIndices, allocator);
    }

    [[nodiscard]]
    inline VulkanBuffer createExportableBuffer(VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage, VkDeviceSize size, const std::string name = "", std::set<uint32_t> queueIndices = {}) const {
        VkExternalMemoryBufferCreateInfo info{ VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
#ifdef WIN32
        info.handleTypes = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
        info.handleTypes = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
        return createBuffer(usage, memoryUsage, size, name, queueIndices, exportableMemoryAllocator, &info);
    }

    operator VkDevice() const {
        return logicalDevice;
    }

    operator VkPhysicalDevice() const {
        return physicalDevice;
    }

    inline VulkanPipeline createGraphicsPipeline(const VkGraphicsPipelineCreateInfo& createInfo, VkPipelineCache pipelineCache = VK_NULL_HANDLE) const {
        assert(logicalDevice);
        VkPipeline pipeline;
        ERR_GUARD_VULKAN(vkCreateGraphicsPipelines(logicalDevice, pipelineCache, 1, &createInfo, nullptr, &pipeline));
        return VulkanPipeline { logicalDevice, pipeline};
    }

    inline VulkanPipeline createComputePipeline(const VkComputePipelineCreateInfo& createInfo, VkPipelineCache pipelineCache = VK_NULL_HANDLE) const {
        assert(logicalDevice);
        VkPipeline pipeline;
        ERR_GUARD_VULKAN(vkCreateComputePipelines(logicalDevice, pipelineCache, 1, &createInfo, nullptr, &pipeline));
        return VulkanPipeline{ logicalDevice, pipeline};
    }

    inline std::vector<VulkanPipeline> createGraphicsPipelines(const std::vector<VkGraphicsPipelineCreateInfo>& createInfos, VkPipelineCache pipelineCache = VK_NULL_HANDLE) const {
        assert(logicalDevice);
        std::vector<VkPipeline> pipelines(createInfos.size());
        ERR_GUARD_VULKAN(vkCreateGraphicsPipelines(logicalDevice, pipelineCache, COUNT(createInfos), createInfos.data(), nullptr, pipelines.data()));

        std::vector<VulkanPipeline> vkPipelines(createInfos.size());
        std::transform(begin(pipelines), end(pipelines), begin(vkPipelines), [&](auto pipeline){
            return VulkanPipeline{ logicalDevice, pipeline };
        });
        return vkPipelines;
    }

    inline VulkanPipeline createRayTracingPipeline(const VkRayTracingPipelineCreateInfoKHR& createInfo, VkPipelineCache pipelineCache = VK_NULL_HANDLE){
        assert(logicalDevice);
        VkPipeline pipeline = VK_NULL_HANDLE;
        vkCreateRayTracingPipelinesKHR(logicalDevice, VK_NULL_HANDLE, pipelineCache, 1, &createInfo, nullptr, &pipeline);
        return VulkanPipeline{logicalDevice, pipeline};
    }

    [[nodiscard]] inline VulkanPipelineLayout createPipelineLayout(const std::vector<VulkanDescriptorSetLayout>& layouts = {}
            , const std::vector<VkPushConstantRange>& ranges = {}) const {
        assert(logicalDevice);

        std::vector<VkDescriptorSetLayout> handles{};
        for(const auto& layout : layouts) handles.push_back(layout.handle);

        VkPipelineLayoutCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        createInfo.setLayoutCount = COUNT(layouts);
        createInfo.pSetLayouts = handles.data();
        createInfo.pushConstantRangeCount = COUNT(ranges);
        createInfo.pPushConstantRanges = ranges.data();

        VkPipelineLayout handle;
        ERR_GUARD_VULKAN(vkCreatePipelineLayout(logicalDevice, &createInfo, nullptr, &handle));


        return VulkanPipelineLayout{ logicalDevice, handle };
    }

    template<typename PoolSizes>
    [[nodiscard]] inline VulkanDescriptorPool createDescriptorPool(uint32_t maxSet, const PoolSizes& poolSizes, VkDescriptorPoolCreateFlags flags = 0) const {
        assert(logicalDevice);
        VkDescriptorPoolCreateInfo createInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        createInfo.flags = flags;
        createInfo.maxSets = maxSet;
        createInfo.poolSizeCount = COUNT(poolSizes);
        createInfo.pPoolSizes = poolSizes.data();

        VkDescriptorPool handle;
        ERR_GUARD_VULKAN(vkCreateDescriptorPool(logicalDevice, &createInfo, nullptr, &handle));

        return VulkanDescriptorPool{ logicalDevice, handle};
    }

    template<typename Bindings>
    [[nodiscard]] inline VulkanDescriptorSetLayout createDescriptorSetLayout(const Bindings& bindings, VkDescriptorSetLayoutCreateFlags flags = 0u, void* next = nullptr) const {
        assert(logicalDevice);
        VkDescriptorSetLayoutCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        createInfo.pNext = next;
        createInfo.flags = flags;
        createInfo.bindingCount = COUNT(bindings);
        createInfo.pBindings = bindings.data();

        VkDescriptorSetLayout setLayout;
        ERR_GUARD_VULKAN(vkCreateDescriptorSetLayout(logicalDevice, &createInfo, nullptr, &setLayout));
        return VulkanDescriptorSetLayout{ logicalDevice, setLayout };
    }

    [[nodiscard]] inline VulkanCommandPool createCommandPool(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags flags = 0, uint32_t queue = 0) const {
        assert(logicalDevice);
        return VulkanCommandPool{ logicalDevice, queueFamilyIndex, flags, queue};
    }

    [[nodiscard]] inline VulkanImage createImage(const VkImageCreateInfo& createInfo, VmaMemoryUsage usage = VMA_MEMORY_USAGE_GPU_ONLY) const{
        assert(logicalDevice);
        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        VkImage image;
        VmaAllocation allocation;
        vmaCreateImage(allocator, &createInfo, &allocInfo, &image, &allocation, nullptr);

        return VulkanImage{ logicalDevice, allocator, image, createInfo.format, allocation, createInfo.initialLayout, createInfo.extent };

    }

    [[nodiscard]] inline VulkanImage createExportableImage(const VkImageCreateInfo& createInfo, VmaMemoryUsage usage = VMA_MEMORY_USAGE_GPU_ONLY) const {
        assert(logicalDevice);
        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        VkImage image;
        VmaAllocation allocation;
        vmaCreateImage(exportableMemoryAllocator, &createInfo, &allocInfo, &image, &allocation, nullptr);

        return VulkanImage{ logicalDevice, exportableMemoryAllocator, image, createInfo.format, allocation, createInfo.initialLayout, createInfo.extent };

    }

    inline VulkanSampler createSampler(const VkSamplerCreateInfo& createInfo) const {
        assert(logicalDevice);
        VkSampler sampler;
        ERR_GUARD_VULKAN(vkCreateSampler(logicalDevice, &createInfo, nullptr, &sampler));
        return VulkanSampler { logicalDevice, sampler};
    }

    [[nodiscard]] inline VulkanFence createFence(VkFenceCreateFlags flags = VK_FENCE_CREATE_SIGNALED_BIT) const {
        assert(logicalDevice);
        return VulkanFence{ logicalDevice, flags};
    }

    [[nodiscard]] inline VulkanSemaphore createSemaphore(VkSemaphoreCreateFlags flags = 0) const {
        assert(logicalDevice);
        VkSemaphoreCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        createInfo.flags = flags;

        VkSemaphore semaphore;
        ERR_GUARD_VULKAN(vkCreateSemaphore(logicalDevice, &createInfo, nullptr, &semaphore));
        return VulkanSemaphore { logicalDevice, semaphore };
    }

    [[nodiscard]] inline VulkanSemaphore createTimelineSemaphore(uint64_t initialValue = 0u, VkSemaphoreCreateFlags flags = 0) const {
        assert(logicalDevice);
        VkSemaphoreTypeCreateInfo typeInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        typeInfo.initialValue = initialValue;

        VkExportSemaphoreCreateInfo exportInfo{
            VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
            &typeInfo,
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT};

        VkSemaphoreCreateInfo createInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &exportInfo, flags};

        VkSemaphore semaphore;
        ERR_GUARD_VULKAN(vkCreateSemaphore(logicalDevice, &createInfo, nullptr, &semaphore));
        return VulkanSemaphore { logicalDevice, semaphore };

    }

    [[nodiscard]] inline VulkanRenderPass createRenderPass(
            const std::vector<VkAttachmentDescription>& attachmentDescriptions
            , const std::vector<SubpassDescription>& subpassDescriptions
            , const std::vector<VkSubpassDependency>& dependencies = {}) const {
        assert(logicalDevice);
        return VulkanRenderPass{ logicalDevice,  attachmentDescriptions, subpassDescriptions, dependencies };
    }

    inline VulkanFramebuffer createFramebuffer(VkRenderPass renderPass
            , const std::vector<VkImageView>& attachments
            , uint32_t width, uint32_t height, uint32_t layers = 1) const {
        assert(logicalDevice);

        return VulkanFramebuffer{ logicalDevice, renderPass, attachments, width, height, layers};
    }

    [[nodiscard]] inline std::optional<VkFormat> findSupportedFormat(const std::vector<VkFormat>& choices, VkImageTiling tiling, VkFormatFeatureFlags features) const {
        for(auto format : choices){
            auto props = getFormatProperties(format);
            if((tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features)) ||
                    (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features))){
                return format;
            }
        }
        return {};
    }

    [[nodiscard]]
    inline VkFormatProperties getFormatProperties(VkFormat format) const {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
        return props;
    }

    [[nodiscard]]
    inline VulkanPipelineCache createPipelineCache(void* data = nullptr, uint32_t size = 0){
        VkPipelineCacheCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        createInfo.initialDataSize = size;
        createInfo.pInitialData = data;

        VkPipelineCache cache;
        vkCreatePipelineCache(logicalDevice, &createInfo, nullptr, &cache);

        return VulkanPipelineCache{ logicalDevice, cache };
    }

    [[nodiscard]]
    inline const VulkanCommandPool& commandPoolFor(uint32_t queueFamilyIndex) const {
        return commandPools[queueFamilyIndex];
    }

    inline const VulkanCommandPool& graphicsCommandPool() const {
        return  commandPools[*queueFamilyIndex.graphics];
    };

    inline const VulkanCommandPool& computeCommandPool() const {
        return commandPools[*queueFamilyIndex.compute];
    }

    inline const VulkanCommandPool& transferCommandPool() const {
        return commandPools[*queueFamilyIndex.transfer];
    }

    [[nodiscard]]
    inline uint32_t getMemoryTypeIndex(uint32_t memoryTypeBitsReq, VkMemoryPropertyFlags requiredProperties) const{
        VkPhysicalDeviceMemoryProperties memoryProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
        for(uint32_t memoryIndex = 0; memoryIndex < memoryProperties.memoryTypeCount; memoryIndex++){
            const uint32_t memoryTypeBits = (1u << memoryIndex);
            const bool isRequiredMemoryType = memoryTypeBits & memoryTypeBitsReq;

            const bool hasRequiredMemoryProperties = memoryProperties.memoryTypes[memoryIndex].propertyFlags & requiredProperties;

            if(isRequiredMemoryType && hasRequiredMemoryProperties){
                return memoryIndex;
            }
        }
        throw std::runtime_error{"Failed to find Required memory type"};
    }

    inline VkDeviceAddress getAddress(const VulkanBuffer& buffer) const{
        VkBufferDeviceAddressInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        info.buffer = buffer;
        return vkGetBufferDeviceAddress(logicalDevice, &info);
    }

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice logicalDevice = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;
    VmaAllocator exportableMemoryAllocator = VK_NULL_HANDLE;
    Settings settings{};
    mutable std::map<uint32_t ,VulkanCommandPool> commandPools;

    template<typename Writes = std::vector<VkWriteDescriptorSet>, typename Copies = std::vector<VkCopyDescriptorSet>>
    inline void updateDescriptorSets(const Writes& writes, const Copies& copies = {}) const {
        assert(logicalDevice != VK_NULL_HANDLE);
        vkUpdateDescriptorSets(logicalDevice, COUNT(writes), writes.data(), COUNT(copies), copies.data());
    }

    inline std::optional<uint32_t> findFirstActiveQueue() const {
        if(queueFamilyIndex.graphics.has_value()){
            return queueFamilyIndex.graphics;
        }
        if(queueFamilyIndex.compute.has_value()){
            return queueFamilyIndex.compute;
        }
        return {};
    }

    template<VkObjectType objectType>
    inline void setName(const std::string& objectName, void* ptr) const {
#ifdef DEBUG_MODE
        VkDebugUtilsObjectNameInfoEXT nameInfo{};
        nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        nameInfo.pObjectName = objectName.c_str();
        nameInfo.objectType = objectType;
        nameInfo.objectHandle = (uint64_t)ptr;
        vkSetDebugUtilsObjectNameEXT(logicalDevice, &nameInfo);
#endif
    }

    inline float timestampPeriod(){
        return getLimits().timestampPeriod;
    }

    [[nodiscard]]
    DescriptorSetLayoutBuilder descriptorSetLayoutBuilder() const ;

    [[nodiscard]]
    GraphicsPipelineBuilder graphicsPipelineBuilder() const;

    VulkanImageOps imageOps();

    [[nodiscard]]
    VmaTotalStatistics memoryUsage() const {
        VmaTotalStatistics stats;
        vmaCalculateStatistics(allocator, &stats);
        return stats;
    }

    inline VulkanShaderModule createShaderModule(const std::string& path) const {
        auto data = loadFile(path);
        return createShaderModule(data);
    }

    inline VulkanShaderModule createShaderModule(const byte_string& data) const {
        auto ptr = reinterpret_cast<uint32_t*>(const_cast<char*>(data.data()));
        return createShaderModule(std::span<uint32_t>{
                ptr,
                data.size()/sizeof(uint32_t),
        });
    }

    inline VulkanShaderModule createShaderModule(const std::vector<uint32_t> &data) const {
        auto ptr = reinterpret_cast<uint32_t*>(const_cast<uint32_t*>(data.data()));
        return createShaderModule( { ptr, data.size() } );
    }

    inline VulkanShaderModule createShaderModule(std::span<uint32_t> data) const {

        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = data.size() * sizeof(uint32_t);
        createInfo.pCode = data.data();

        VkShaderModule handle;
        auto status = vkCreateShaderModule(logicalDevice, &createInfo, nullptr, &handle);
        REPORT_ERROR(status, "Failed to create shader module");

        return VulkanShaderModule{ logicalDevice, handle };
    }

    inline void wait() const {
        vkDeviceWaitIdle(logicalDevice);
    }

};

#include "DescriptorSetBuilder.hpp"