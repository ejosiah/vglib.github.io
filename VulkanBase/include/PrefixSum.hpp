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

    void resizeInternalBuffer();

    std::vector<PipelineMetaData> pipelineMetaData() override;

    [[deprecated("use PrefixSum::operator()(VkCommandBuffer commandBuffer, BufferSection section) instead")]]
    void operator()(VkCommandBuffer commandBuffer, VulkanBuffer& buffer);

    void operator()(VkCommandBuffer commandBuffer, BufferSection section);

    [[deprecated("use PrefixSum::inclusive(VkCommandBuffer commandBuffer, BufferSection section) instead")]]
    void inclusive(VkCommandBuffer commandBuffer, VulkanBuffer& buffer, VkAccessFlags dstAccessMask, VkPipelineStageFlags dstStage);

    void inclusive(VkCommandBuffer commandBuffer, BufferSection section);

    template<typename Itr>
    void scan(const Itr _first, const Itr _last){
        VkDeviceSize size = sizeof(decltype(*_first)) * std::distance(_first, _last);
        void* source = reinterpret_cast<void*>(&*_first);
        VulkanBuffer buffer = device->createCpuVisibleBuffer(source, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        _commandPool = _commandPool ? _commandPool : const_cast<VulkanCommandPool*>(&device->graphicsCommandPool());
        _commandPool->oneTimeCommand([&buffer, this](auto cmdBuffer) {
            operator()(cmdBuffer, { buffer, 0, buffer.size});
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
            inclusive(cmdBuffer, buffer, VK_ACCESS_HOST_READ_BIT, VK_PIPELINE_STAGE_HOST_BIT);
        });
        void* result = buffer.map();
        std::memcpy(source, result, size);
        buffer.unmap();
    }

    void createDescriptorPool();

    void updateDataDescriptorSets(VulkanBuffer& buffer);

    void addBufferMemoryBarriers(VkCommandBuffer commandBuffer, const std::vector<VulkanBuffer*>& buffers);

    void addComputeWriteToTransferReadBarrier(VkCommandBuffer commandBuffer, const std::vector<VulkanBuffer*>& buffers);

    void addBufferTransferWriteToReadBarriers(VkCommandBuffer commandBuffer, const std::vector<VulkanBuffer*>& buffers);

    void addBufferTransferWriteToComputeReadBarriers(VkCommandBuffer commandBuffer, const std::vector<VulkanBuffer*>& buffers);

    void addBufferTransferWriteToWriteBarriers(VkCommandBuffer commandBuffer, const std::vector<VulkanBuffer*>& buffers);

    void addBufferTransferReadToWriteBarriers(VkCommandBuffer commandBuffer, const std::vector<VulkanBuffer*>& buffers);

    void copyToInternalBuffer(VkCommandBuffer commandBuffer, BufferSection& section);

    void copyFromInternalBuffer(VkCommandBuffer commandBuffer, BufferSection& section);

protected:
    static constexpr int ITEMS_PER_WORKGROUP = 8192;

    void scanInternal(VkCommandBuffer commandBuffer, BufferSection section);

    void createDescriptorSet();

private:
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkDescriptorSet sumScanDescriptorSet = VK_NULL_HANDLE;
    VulkanDescriptorSetLayout setLayout;
    uint32_t bufferOffsetAlignment;
    VulkanDescriptorPool descriptorPool;
    VulkanCommandPool* _commandPool{};
    VulkanBuffer stagingBuffer;
    VulkanBuffer internalDataBuffer;
    VkDeviceSize capacity{ITEMS_PER_WORKGROUP * 128 * sizeof(uint32_t)};

    struct {
        int itemsPerWorkGroup = ITEMS_PER_WORKGROUP;
        int N = 0;
    } constants;

public:
    VulkanBuffer sumsBuffer;
    VulkanBuffer sumOfSumsBuffer;
};
