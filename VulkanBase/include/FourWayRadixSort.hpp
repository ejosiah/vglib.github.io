#include "Sort.hpp"
#include "PrefixSum.hpp"
#include "Profiler.hpp"

class FourWayRadixSort : public GpuSort{
public:
    explicit FourWayRadixSort(VulkanDevice* device = nullptr, uint maxElementsPerWorkGroup = 128, bool debug = false);

    void init() override;

    void operator()(VkCommandBuffer commandBuffer, VulkanBuffer &buffer) override;

    void operator()(VkCommandBuffer commandBuffer, const BufferRegion &region) override;

protected:
    std::vector<PipelineMetaData> pipelineMetaData() final;

    void createDescriptorPool();

    void createDescriptorSetLayouts();

    void createDescriptorSets();

    void initProfiler();

    void updateDataDescriptorSets(VulkanBuffer& buffer);

    void localSort(VkCommandBuffer commandBuffer);

    void scan(VkCommandBuffer commandBuffer);

    void globalShuffle(VkCommandBuffer commandBuffer);

    uint32_t calculateNumWorkGroups(VulkanBuffer& buffer);

public:
    VulkanDescriptorPool descriptorPool;
    VulkanBuffer prefixSumBuffer;
    VulkanBuffer blockSumBuffer;
    VulkanBuffer scratchBuffer;
    VulkanDescriptorSetLayout scanLayoutSet;
    VulkanDescriptorSetLayout dataLayoutSet;
    VkDescriptorSet scanDescriptorSet = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 2> dataDescriptorSets{};
    std::array<VulkanBuffer*, 2> dataBuffers;
    PrefixSum prefixSum;
    uint numWorkGroups;
    Profiler profiler;
    bool debug = false;

    struct{
        uint shift_width{0};
        uint data_length{0};
    } constants{};

    std::array<uint32_t, 2> constData{};

    static constexpr uint DATA_IN = 0;
    static constexpr uint DATA_OUT = 1;
};