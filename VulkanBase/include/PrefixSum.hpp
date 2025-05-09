#pragma once

#include "Barrier.hpp"

#include "ComputePipelins.hpp"
#include "DescriptorSetBuilder.hpp"
#include <string>
#include <string_view>


enum class Operation : uint32_t { Add, Min, Max };
enum class DataType : uint32_t { Int, Float };

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

    void operator()(VkCommandBuffer commandBuffer, VulkanBuffer& buffer,
            Operation operation = Operation::Add, DataType dataType = DataType::Int);

    void operator()(VkCommandBuffer commandBuffer, const BufferRegion& region,
            Operation operation = Operation::Add, DataType dataType = DataType::Int);

    void min(VkCommandBuffer commandBuffer, const BufferRegion& data,  VulkanBuffer& result, DataType dataType = DataType::Int);
    void max(VkCommandBuffer commandBuffer, const BufferRegion& data,  VulkanBuffer& result, DataType dataType = DataType::Int);

    void accumulate(VkCommandBuffer commandBuffer, VulkanBuffer& data,  VulkanBuffer& result,
                    Operation operation = Operation::Add, DataType dataType = DataType::Int);

    void accumulate(VkCommandBuffer commandBuffer, const BufferRegion& data, VulkanBuffer& result,
                    Operation operation = Operation::Add, DataType dataType = DataType::Int);

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

    void copySum(VkCommandBuffer commandBuffer, VulkanBuffer& dst);

    void scanInternal(VkCommandBuffer commandBuffer, BufferRegion section, Operation operation, DataType dataType);

    void createDescriptorSet();

    static constexpr VkDeviceSize DataUnitSize = sizeof(uint32_t);

private:
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkDescriptorSet sumScanDescriptorSet = VK_NULL_HANDLE;
    VulkanDescriptorSetLayout setLayout;
    uint32_t bufferOffsetAlignment{};
    VulkanDescriptorPool descriptorPool;
    VulkanCommandPool* _commandPool{};
    VulkanBuffer stagingBuffer;
    VulkanBuffer internalDataBuffer;
    VkDeviceSize capacity{ITEMS_PER_WORKGROUP * 128 * DataUnitSize};

    struct {
        uint32_t itemsPerWorkGroup = ITEMS_PER_WORKGROUP;
        uint32_t N = 0;
        uint32_t operation = to<uint32_t>(Operation::Add);
    } constants;

    VulkanBuffer sumsBuffer;
    VulkanBuffer sumOfSumsBuffer;
};
