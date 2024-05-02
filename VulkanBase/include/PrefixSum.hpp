#pragma once

#include "Barrier.hpp"

#include "ComputePipelins.hpp"
#include "DescriptorSetBuilder.hpp"
#include <string>
#include <string_view>

class PrefixSum : public ComputePipelines {
public:
    static constexpr int ITEMS_PER_WORKGROUP = 8192;
    static constexpr int MAX_NUM_ITEMS = ITEMS_PER_WORKGROUP * ITEMS_PER_WORKGROUP;

    struct DataSizeExceedsMaxSupported : public std::runtime_error {
        DataSizeExceedsMaxSupported() : std::runtime_error("dataset exceeds max supported items of " + std::to_string(MAX_NUM_ITEMS)){}

    };

    PrefixSum() = default;

    PrefixSum(VulkanDevice* device, VulkanCommandPool* commandPool = nullptr);

    void init();

    void operator()(VkCommandBuffer commandBuffer, VulkanBuffer& buffer);

    void operator()(VkCommandBuffer commandBuffer, const BufferRegion& region);

    void inclusive(VkCommandBuffer commandBuffer, VulkanBuffer& buffer, VkAccessFlags dstAccessMask, VkPipelineStageFlags dstStage);

    void inclusive(VkCommandBuffer commandBuffer, const BufferRegion& region);

    template<typename Itr>
    void scan(const Itr _first, const Itr _last){
        VkDeviceSize size = sizeof(decltype(*_first)) * std::distance(_first, _last);
        void* source = reinterpret_cast<void*>(&*_first);
        VulkanBuffer buffer = device->createCpuVisibleBuffer(source, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        _commandPool = _commandPool ? _commandPool : const_cast<VulkanCommandPool*>(&device->graphicsCommandPool());
        _commandPool->oneTimeCommand([&buffer, this](auto cmdBuffer) {
            operator()(cmdBuffer, { &buffer, 0, buffer.size});
        });
        void* result = buffer.map();
        std::memcpy(source, result, size);
        buffer.unmap();
    }

    template<typename Itr>
    void inclusive(const Itr _first, const Itr _last) {
        VkDeviceSize size = sizeof(decltype(*_first)) * std::distance(_first, _last);
        void* source = reinterpret_cast<void*>(&*_first);
        VulkanBuffer buffer = device->createCpuVisibleBuffer(source, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        _commandPool = _commandPool ? _commandPool : const_cast<VulkanCommandPool*>(&device->graphicsCommandPool());
        _commandPool->oneTimeCommand([&buffer, this](auto cmdBuffer) {
            inclusive(cmdBuffer, buffer, VK_ACCESS_HOST_READ_BIT, VK_PIPELINE_STAGE_HOST_BIT);
        });
        void* result = buffer.map();
        std::memcpy(source, result, size);
        buffer.unmap();
    }

protected:
    void resizeInternalBuffer();

    std::vector<PipelineMetaData> pipelineMetaData() override;

    void createDescriptorPool();

    void updateDataDescriptorSets(VulkanBuffer& buffer);

    void copyToInternalBuffer(VkCommandBuffer commandBuffer, const BufferRegion& region);

    void copyFromInternalBuffer(VkCommandBuffer commandBuffer, const BufferRegion& region, VkDeviceSize srcOffset = 0);

    void scanInternal(VkCommandBuffer commandBuffer, BufferRegion section);

    void createDescriptorSet();

    static constexpr VkDeviceSize DataUnitSize = sizeof(uint32_t);

private:
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkDescriptorSet sumScanDescriptorSet = VK_NULL_HANDLE;
    VulkanDescriptorSetLayout setLayout;
    uint32_t bufferOffsetAlignment;
    VulkanDescriptorPool descriptorPool;
    VulkanCommandPool* _commandPool{};
    VulkanBuffer stagingBuffer;
    VulkanBuffer internalDataBuffer;
    VkDeviceSize capacity{ITEMS_PER_WORKGROUP * 128 * DataUnitSize};

    struct {
        int itemsPerWorkGroup = ITEMS_PER_WORKGROUP;
        int N = 0;
    } constants;

    VulkanBuffer sumsBuffer;
    VulkanBuffer sumOfSumsBuffer;
};
