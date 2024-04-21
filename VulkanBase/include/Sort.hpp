#pragma once

#include "ComputePipelins.hpp"
#include "VulkanQuery.hpp"
#include "Profiler.hpp"

#include <type_traits>
#include <map>
#include <optional>

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
    static constexpr uint NUM_DATA_ELEMENTS = 1;
    static constexpr uint ELEMENTS_PER_WG = 1 << 14;
    static constexpr uint MAX_WORKGROUPS = 64;
    static constexpr uint NUM_THREADS_PER_BLOCK = 1024;
    static constexpr uint NUM_GROUPS_PER_WORKGROUP = NUM_THREADS_PER_BLOCK / WORD_SIZE;

    enum Query  { COUNT, PREFIX_SUM, REORDER, NUM_QUERIES };

public:

public:
    explicit RadixSort(VulkanDevice* device = nullptr, bool debug = false);

    void init() override;

    void createDescriptorPool();

    void createDescriptorSetLayouts();

    void createDescriptorSets();

    void createProfiler();

    std::vector<PipelineMetaData> pipelineMetaData() override;

    void operator()(VkCommandBuffer commandBuffer, VulkanBuffer& keys, Records& records);

    void operator()(VkCommandBuffer commandBuffer, VulkanBuffer &buffer) override;

    void operator()(VkCommandBuffer commandBuffer, VulkanBuffer &buffer, bool reorderIndices);

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

    void updateSequenceDescriptorSet(VulkanBuffer& buffer);

    VulkanBuffer& sortWithIndices(VkCommandBuffer commandBuffer, VulkanBuffer &buffer);

    void generateSequence(VkCommandBuffer commandBuffer, VulkanBuffer& buffer);

    void updateConstants(VulkanBuffer& buffer);

    static uint numWorkGroups(VulkanBuffer& buffer);

    void count(VkCommandBuffer commandBuffer, VkDescriptorSet dataDescriptorSet);

    void prefixSum(VkCommandBuffer commandBuffer);

    void reorder(VkCommandBuffer commandBuffer, std::array<VkDescriptorSet, 2>& dataDescriptorSets);

    void updateDataDescriptorSets(VulkanBuffer& inBuffer);

    void updateRecordsDescriptorSets(Records& records);

    void commitProfiler();

    Profiler profiler;

    bool debug = false;

protected:
    VulkanDescriptorPool descriptorPool;
    VulkanDescriptorSetLayout dataSetLayout;
    VulkanDescriptorSetLayout countsSetLayout;
    std::array<VkDescriptorSet, 2> dataDescriptorSets;
    VkDescriptorSet countsDescriptorSet;
    VulkanBuffer countsBuffer;
    VulkanBuffer sumBuffer;
    std::array<VulkanBuffer*, 2> dataBuffers;
    std::array<VulkanBuffer, 2> indexBuffers;
    std::array<VulkanBuffer, 2> recordBuffers;
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
        uint reorderIndices{};
        uint reorderRecords{};
        uint recordSize{};
    } constants{};
    VkBuffer previousBuffer{};

};