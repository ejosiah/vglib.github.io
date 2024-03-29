#pragma once

#include "ComputePipelins.hpp"
#include "DescriptorSetBuilder.hpp"
#include <string>
#include <string_view>

class PrefixSum : public ComputePipelines{
public:
    PrefixSum() = default;

    PrefixSum(VulkanDevice* device, VulkanCommandPool* commandPool = nullptr);

    void init();

    std::vector<PipelineMetaData> pipelineMetaData() override;

    void operator()(VkCommandBuffer commandBuffer, VulkanBuffer& buffer);

    void inclusive(VkCommandBuffer commandBuffer, VulkanBuffer& buffer, VkAccessFlags dstAccessMask, VkPipelineStageFlags dstStage);

    template<typename Itr>
    void scan(const Itr _first, const Itr _last){
        VkDeviceSize size = sizeof(decltype(*_first)) * std::distance(_first, _last);
        void* source = reinterpret_cast<void*>(&*_first);
        VulkanBuffer buffer = device->createCpuVisibleBuffer(source, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        updateDataDescriptorSets(buffer);
        _commandPool = _commandPool ? _commandPool : const_cast<VulkanCommandPool*>(&device->graphicsCommandPool());
        _commandPool->oneTimeCommand([&buffer, this](auto cmdBuffer) {
            operator()(cmdBuffer, buffer);
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
        updateDataDescriptorSets(buffer);
        _commandPool = _commandPool ? _commandPool : const_cast<VulkanCommandPool*>(&device->graphicsCommandPool());
        _commandPool->oneTimeCommand([&buffer, this](auto cmdBuffer) {
            inclusive(cmdBuffer, buffer);
        });
        void* result = buffer.map();
        std::memcpy(source, result, size);
        buffer.unmap();
    }

    void createDescriptorPool();

    void updateDataDescriptorSets(VulkanBuffer& buffer);

    void addBufferMemoryBarriers(VkCommandBuffer commandBuffer, const std::vector<VulkanBuffer*>& buffers);

    void addBufferTransferBarriers(VkCommandBuffer commandBuffer, const std::vector<VulkanBuffer*>& buffers);

protected:
    static constexpr int ITEMS_PER_WORKGROUP = 8 << 10;

    void createDescriptorSet();

private:
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkDescriptorSet sumScanDescriptorSet = VK_NULL_HANDLE;
    VulkanDescriptorSetLayout setLayout;
    VulkanBuffer sumsBuffer;
    uint32_t bufferOffsetAlignment;
    VulkanDescriptorPool descriptorPool;
    VulkanCommandPool* _commandPool{};
    VulkanBuffer stagingBuffer;

    struct {
        int itemsPerWorkGroup = ITEMS_PER_WORKGROUP;
        int N = 0;
    } constants;

};
