#pragma once

#include "ComputePipelins.hpp"
#include "VulkanQuery.hpp"
#include "Profiler.hpp"
#include "OrderChecker.hpp"

#include <type_traits>
#include <map>
#include <optional>
#include <string_view>

enum class KeyType : uint {
    Int = 0, Float, Uint
};

struct Records {
    VulkanBuffer buffer;
    uint size;
    KeyType keyType{KeyType::Uint};
};

class GpuSort : public ComputePipelines {
public:
    explicit GpuSort(VulkanDevice* device = nullptr) : ComputePipelines(device){};

    virtual void init() {};

    virtual void operator()(VkCommandBuffer commandBuffer, VulkanBuffer& buffer) = 0;

    virtual void operator()(VkCommandBuffer commandBuffer, const BufferRegion& region) = 0;

    template<typename Itr>
    void sort(const Itr _first, const Itr _last){
        VkDeviceSize size = sizeof(decltype(*_first)) * std::distance(_first, _last);
        void* source = reinterpret_cast<void*>(&*_first);
        VkBufferUsageFlags flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VulkanBuffer buffer = device->createCpuVisibleBuffer(source, size, flags);
        device->commandPoolFor(device->findFirstActiveQueue().value()).oneTimeCommand([&buffer, this](auto cmdBuffer){
            operator()(cmdBuffer, buffer);
        });
        void* sorted = buffer.map();
        std::memcpy(source, sorted, size);
        buffer.unmap();
    }
};



class RadixSort : public GpuSort{
    // FIXME these constants are based on shared memory size of 8,192
    // FIXME set values based on device memory limits
    static constexpr uint WORD_SIZE = 32;
    static constexpr uint BLOCK_SIZE = 8;
    static constexpr uint RADIX = 256;
    static constexpr uint PASSES = WORD_SIZE / BLOCK_SIZE;
    static constexpr uint DATA_IN = 0;
    static constexpr uint DATA_OUT = 1;
    static constexpr uint INDEX_IN = 2;
    static constexpr uint INDEX_OUT = 3;
    static constexpr uint RECORDS_IN = 4;
    static constexpr uint RECORDS_OUT = 5;
    static constexpr uint DATA = 0;
    static constexpr uint INDICES = 1;
    static constexpr uint RECORDS = 2;
    static constexpr uint ADD_IN = 0;
    static constexpr uint ADD_OUT = 1;
    static constexpr uint KEY = 0;
    static constexpr uint COUNTS = 0;
    static constexpr uint SUMS = 1;
    static constexpr uint ORDER_CHECKING = 2;
    static constexpr uint NUM_DATA_ELEMENTS = 1;
    static constexpr uint ELEMENTS_PER_WG = 1 << 14;
    static constexpr uint MAX_WORKGROUPS = 64;
    static constexpr uint NUM_THREADS_PER_BLOCK = 1024;
    static constexpr uint NUM_GROUPS_PER_WORKGROUP = NUM_THREADS_PER_BLOCK / WORD_SIZE;

    enum Query  { COUNT, PREFIX_SUM, REORDER, NUM_QUERIES };

public:
    explicit RadixSort(VulkanDevice* device = nullptr, bool debug = false);

    void init() override;

    void enableOrderChecking();

    void createDescriptorPool();

    void createDescriptorSetLayouts();

    void createDescriptorSets();

    void createProfiler();

    std::vector<PipelineMetaData> pipelineMetaData() override;

    void operator()(VkCommandBuffer commandBuffer, VulkanBuffer& keys, Records& records);

    void operator()(VkCommandBuffer commandBuffer, VulkanBuffer &buffer) override;

    void operator()(VkCommandBuffer commandBuffer, const BufferRegion &region) override;

    void operator()(VkCommandBuffer commandBuffer, const BufferRegion &region, std::string_view reorderPipeline);

    template<typename T>
    void sortTyped(VkCommandBuffer commandBuffer, VulkanBuffer& buffer) {
        bitFlipConstants.numEntries = buffer.sizeAs<T>();
        bitFlipConstants.reverse = 0;

        if constexpr (std::is_same_v<T, int>) {
            bitFlipConstants.dataType = 0;
            flipBits(commandBuffer, buffer);
        }else if constexpr (std::is_same_v<T, float>) {
            bitFlipConstants.dataType = 1;
            flipBits(commandBuffer, buffer);
        }

        operator()(commandBuffer, buffer);

        if constexpr (!std::is_same_v<T, uint>) {
            bitFlipConstants.reverse = 1;
            flipBits(commandBuffer, buffer);
        }
    }

    void flipBits(VkCommandBuffer commandBuffer, VulkanBuffer& buffer);

    void updateBitFlipDescriptorSet(VulkanBuffer& buffer);

    void sortWithIndices(VkCommandBuffer commandBuffer, VulkanBuffer &keys, VulkanBuffer& indexes);

    void generateSequence(VkCommandBuffer commandBuffer, uint32_t numEntries);

    void updateConstants(const BufferRegion& region);

    static uint numWorkGroups(const BufferRegion& region);

    void checkOrder(VkCommandBuffer commandBuffer, const BufferRegion& region);

    void count(VkCommandBuffer commandBuffer, VkDescriptorSet dataDescriptorSet);

    void prefixSum(VkCommandBuffer commandBuffer);

    void reorder(VkCommandBuffer commandBuffer, std::array<VkDescriptorSet, 2>& dataDescriptorSets, std::string_view reorderPipeline);

    void reorderIndices(VkCommandBuffer commandBuffer, std::array<VkDescriptorSet, 2>& dataDescriptorSets);

    void copyToInternalKeyBuffer(VkCommandBuffer commandBuffer, const BufferRegion& src);

    void copyFromInternalKeyBuffer(VkCommandBuffer commandBuffer, const BufferRegion& dst);

    void copyFromInternalIndexBuffer(VkCommandBuffer commandBuffer, const BufferRegion& dst);

    void copyToInternalRecordBuffer(VkCommandBuffer commandBuffer, const BufferRegion& src);

    void copyFromInternalRecordBuffer(VkCommandBuffer commandBuffer, const BufferRegion& dst);

    void copyBuffer(VkCommandBuffer commandBuffer, const BufferRegion& src, const BufferRegion& dst);

    void updateDataDescriptorSets();

    void resizeInternalBuffer();

    void commitProfiler();

    Profiler profiler;

    bool debug = false;

public:
    static constexpr VkDeviceSize DataUnitSize = sizeof(uint32_t);
    static constexpr const VkDeviceSize INITIAL_CAPACITY = (1 << 20) * DataUnitSize;
    static constexpr const VkDeviceSize NUM_ENTRIES_PER_RECORD = 20;
    static constexpr const char* REORDER_TYPE_KEYS = "radix_sort_reorder";
    static constexpr const char* REORDER_TYPE_INDEXES = "radix_sort_reorder_indices";
    static constexpr const char* REORDER_TYPE_RECORDS = "radix_sort_reorder_records";

    VulkanDescriptorPool descriptorPool;
    VulkanDescriptorSetLayout dataSetLayout;
    VulkanDescriptorSetLayout countsSetLayout;
    std::array<VkDescriptorSet, 2> dataDescriptorSets;
    VkDescriptorSet countsDescriptorSet;
    VulkanBuffer countsBuffer;
    VulkanBuffer sumBuffer;
    VulkanBuffer orderBuffer;
    VulkanBuffer dataScratchBuffer;
    uint workGroupCount = 0;
    VulkanDescriptorSetLayout bitFlipSetLayout;
    VkDescriptorSet bitFlipDescriptorSet;

    VulkanDescriptorSetLayout sequenceSetLayout;
    VkDescriptorSet sequenceDescriptorSet;

    struct {
        uint dataType;
        uint reverse;
        uint numEntries;
    } bitFlipConstants{};

    struct {
        std::array<VulkanBuffer, 2> keys;
        std::array<VulkanBuffer, 2> indexes;
        std::array<VulkanBuffer, 2> records;
    } internal;

    struct {
        uint start{0};
        uint numEntries{};
    } seqConstants;

    struct {
        uint block;
        uint R = WORD_SIZE;
        uint Radix = RADIX;
        uint Num_Groups_per_WorkGroup;
        uint Num_Elements_per_WorkGroup;
        uint Num_Elements_Per_Group;
        uint Num_Elements;
        uint Num_Radices_Per_WorkGroup;
        uint Num_Groups;
        uint recordSize{};
    } constants{};
    VkBuffer previousBuffer{};
    std::optional<OrderChecker> _orderChecker;

    VkDeviceSize capacity{INITIAL_CAPACITY};

};