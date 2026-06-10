#include "gpu/algorithm.h"
#include "VulkanRAII.h"
#include "ComputePipelins.hpp"
#include "memory"
#include "fmt/format.h"
#include <array>
#include <spdlog/spdlog.h>
#include "glsl_shaders.hpp"

namespace gpu {
    static VulkanDescriptorPool* g_descriptorPool{};

    static VulkanDescriptorPool createDescriptorPool(VulkanDevice& device) {
        constexpr uint32_t maxSets = 4;
        std::vector<VkDescriptorPoolSize> poolSizes{
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 9 }
        };

        return device.createDescriptorPool(maxSets, poolSizes);
    }

    std::string resource(const std::string &name) {
        auto res = g_fileMgr->getFullPath(name);
        assert(res.has_value());
        return res->string();
    }

    static void addBufferMemoryBarriers(VkCommandBuffer commandBuffer, const std::vector<const VulkanBuffer *> &buffers){
        std::vector<VkBufferMemoryBarrier> barriers(buffers.size());

        for(int i = 0; i < buffers.size(); i++) {
            barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barriers[i].offset = 0;
            barriers[i].buffer = *buffers[i];
            barriers[i].size = buffers[i]->size;
        }

        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                             nullptr, COUNT(barriers), barriers.data(), 0, nullptr);
    }

    class AlgorithmPipelines : public ComputePipelines{
    public:
        AlgorithmPipelines()
            :ComputePipelines(g_device){}

        void init(){
            createDescriptorSet();
            createPipelines();
        }

        void createDescriptorSet(){
            inoutSetLayout =
                device->descriptorSetLayoutBuilder()
                    .binding(0)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1)
                        .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
                    .binding(1)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1)
                        .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
                .createLayout();
        }

        std::vector<PipelineMetaData> pipelineMetaData() override {
            return {
                    {
                            "average",
                            data_shaders_algorithm_average_comp,
                            {&inoutSetLayout, &inoutSetLayout},
                            { {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float) * 2}}
                    }
            };
        }

    private:
        VulkanDescriptorSetLayout inoutSetLayout;
    };

    class PrefixScan : public ComputePipelines {
    public:
        PrefixScan()
            :ComputePipelines(g_device){}
            
        void init() {
            bufferOffsetAlignment = g_device->getLimits().minStorageBufferOffsetAlignment;
            createDescriptorSet();
            createPipelines();
        }

        void scan(VkCommandBuffer commandBuffer, VulkanBuffer &buffer, VkDescriptorSet inoutDescriptorSet = VK_NULL_HANDLE){
            if(!inoutDescriptorSet) {
                updateDataDescriptorSets(buffer);
            }
            updateSumsDescriptorSet(buffer);

            auto lDataDescriptorSet = inoutDescriptorSet ? inoutDescriptorSet : dataDescriptorSet;

            int size = buffer.size/sizeof(float);
            static std::array<VkDescriptorSet, 2> sets{};
            sets[0] = lDataDescriptorSet;
            sets[1] = sumScanDescriptorSet;
            numWorkGroups = std::abs(size - 1)/ITEMS_PER_WORKGROUP + 1;
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("prefix_scan"));
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("prefix_scan"), 0, COUNT(sets), sets.data(), 0, nullptr);
            vkCmdDispatch(commandBuffer, numWorkGroups, 1, 1);

            if(numWorkGroups > 1){
                sets[0] = sumScanDescriptorSet;
                sets[1] = finalSumDescriptorSet;
                addBufferMemoryBarriers(commandBuffer, {&buffer, &sumsBuffer});
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("prefix_scan"), 0, COUNT(sets), sets.data(), 0, nullptr);
                vkCmdDispatch(commandBuffer, 1, 1, 1);

                sets[0] = lDataDescriptorSet;
                sets[1] = sumScanDescriptorSet;
                addBufferMemoryBarriers(commandBuffer, { &buffer, &sumsBuffer });
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("add"));
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("add"), 0, COUNT(sets), sets.data(), 0, nullptr);
                vkCmdPushConstants(commandBuffer, layout("add"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
                vkCmdDispatch(commandBuffer, numWorkGroups, 1, 1);
            }
        }

        void reduce(VkCommandBuffer commandBuffer, VulkanBuffer &input, VkDescriptorSet inoutDescriptorSet){
            scan(commandBuffer, input, inoutDescriptorSet);
            addBufferMemoryBarriers(commandBuffer, {&input, &sumsBuffer});
            static std::array<VkDescriptorSet, 2> sets{};
            sets[0] = inoutDescriptorSet;
            sets[1] = sumScanDescriptorSet;
            reduceConstants.count = input.sizeAs<float>();
            reduceConstants.tail = lastSumOffset()/sizeof(float);
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("reduce"));
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("reduce"), 0, COUNT(sets), sets.data(), 0, nullptr);
            vkCmdPushConstants(commandBuffer, layout("reduce"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(reduceConstants), &reduceConstants);
            vkCmdDispatch(commandBuffer, 1, 1, 1);
        }

        void updateDataDescriptorSets(VulkanBuffer &buffer){
            size_t numItems = buffer.size/sizeof(float);
            VkDeviceSize sumsSize = (std::abs(int(numItems - 1))/ITEMS_PER_WORKGROUP + 1) * sizeof(float);  // FIXME
            sumsSize = alignedSize(sumsSize, bufferOffsetAlignment) + sizeof(int);
            sumsBuffer = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, sumsSize);
            constants.N = numItems;

            VkDescriptorBufferInfo info{ buffer, 0, VK_WHOLE_SIZE};
            auto writes = initializers::writeDescriptorSets<1>();
            writes[0].dstSet = dataDescriptorSet;
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[0].pBufferInfo = &info;

            device->updateDescriptorSets(writes);
        }

        void updateSumsDescriptorSet(VulkanBuffer &buffer){
            size_t numItems = buffer.size/sizeof(float);
            VkDeviceSize sumsSize = (std::abs(int(numItems - 1))/ITEMS_PER_WORKGROUP + 1) * sizeof(float);  // FIXME
            sumsSize = alignedSize(sumsSize, bufferOffsetAlignment) + sizeof(int);
            sumsBuffer = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, sumsSize);
            constants.N = numItems;

            auto writes = initializers::writeDescriptorSets<2>();

            VkDescriptorBufferInfo sumsInfo{ sumsBuffer, 0, VK_WHOLE_SIZE}; // TODO FIXME use separate buffers
            writes[0].dstSet = sumScanDescriptorSet;
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[0].pBufferInfo = &sumsInfo;

            // for sum scan
            VkDescriptorBufferInfo sumsSumInfo{ sumsBuffer, sumsBuffer.size - sizeof(float), VK_WHOLE_SIZE}; // TODO FIXME use separate buffers
            writes[1].dstSet = finalSumDescriptorSet;
            writes[1].dstBinding = 0;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[1].pBufferInfo = &sumsSumInfo;


            device->updateDescriptorSets(writes);
        }

        static constexpr int ITEMS_PER_WORKGROUP = 8 << 10;

        void createDescriptorSet(){
            setLayout = device->descriptorSetLayoutBuilder()
                .binding(0)
                    .descriptorCount(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
                .createLayout();

            inoutSetLayout =
                device->descriptorSetLayoutBuilder()
                    .binding(0)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1)
                        .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
                    .binding(1)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1)
                        .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
                .createLayout();

            assert(g_descriptorPool != nullptr);
            auto sets = g_descriptorPool->allocate({ inoutSetLayout, inoutSetLayout, inoutSetLayout });
            dataDescriptorSet = sets[0];
            sumScanDescriptorSet = sets[1];
            finalSumDescriptorSet = sets[2];
        }

        std::vector<PipelineMetaData> pipelineMetaData() override {
            return {
                    {
                            "prefix_scan",
                            data_shaders_algorithm_scan_comp,
                            { &inoutSetLayout, &inoutSetLayout }
                    },
                    {
                            "add",
                            data_shaders_algorithm_add_comp,
                            { &inoutSetLayout, &inoutSetLayout },
                            { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants)} }
                    },
                    {
                            "reduce",
                            data_shaders_algorithm_reduce_output_comp,
                            {&inoutSetLayout, &inoutSetLayout},
                            { {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(reduceConstants)}}
                    }
            };
        }

        float lastSum() const {
            auto sums = reinterpret_cast<float*>(sumsBuffer.map());
            auto tail = glm::step(1, numWorkGroups - 1) * numWorkGroups;
            auto result = sums[tail];
            sumsBuffer.unmap();
            return result;
        }

        inline VkDeviceSize lastSumOffset() {
            return lastSumIndex() * sizeof(float);
        }

        inline int lastSumIndex(){
            return glm::step(1, numWorkGroups - 1) * numWorkGroups;
        }

        VulkanBuffer sumsBuffer;
        
        VkDescriptorSet dataDescriptorSet{};
        VkDescriptorSet sumScanDescriptorSet{};
        VkDescriptorSet finalSumDescriptorSet{};
        VulkanDescriptorSetLayout setLayout;
        VulkanDescriptorSetLayout inoutSetLayout;
        uint32_t bufferOffsetAlignment{};
        int numWorkGroups{};

        struct {
            int itemsPerWorkGroup = ITEMS_PER_WORKGROUP;
            int N = 0;
        } constants;

        struct {
            int count;
            int tail;
        } reduceConstants;
    };

    class MathOperations : public ComputePipelines {
    public:
        MathOperations()
            : ComputePipelines(g_device) {}

        void init() {
            createDescriptorSetLayout();
            createDescriptorSet();
            createPipelines();
        }

        void execute(VkCommandBuffer commandBuffer, const BufferRegion& as, const BufferRegion& bs, const BufferRegion& cs, Operation operation) {
            const auto count = sizeAsFloat(as);
            if(count == 0) {
                return;
            }

            assert(sizeAsFloat(bs) >= count);
            assert(sizeAsFloat(cs) >= count);

            updateDescriptorSet(as, bs, cs);
            PushConstants constants{
                static_cast<uint32_t>(operation),
                count
            };
            const auto gx = (count + WorkGroupSize - 1u)/WorkGroupSize;
            const std::array<VkDescriptorSet, 1> sets{ descriptorSet_ };

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("math_operations"));
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("math_operations"), 0, COUNT(sets), sets.data(), 0, nullptr);
            vkCmdPushConstants(commandBuffer, layout("math_operations"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
            vkCmdDispatch(commandBuffer, gx, 1, 1);

            addBufferMemoryBarriers(commandBuffer, { cs.buffer });
        }

    private:
        static constexpr uint32_t WorkGroupSize = 32;

        struct PushConstants {
            uint32_t operation;
            uint32_t count;
        };

        void createDescriptorSetLayout() {
            descriptorSetLayout_ =
                device->descriptorSetLayoutBuilder()
                    .name("gpu_math_algorithms")
                    .binding(0)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1)
                        .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                    .binding(1)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1)
                        .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                    .binding(2)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1)
                        .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .createLayout();
        }

        void createDescriptorSet() {
            assert(g_descriptorPool != nullptr);
            descriptorSet_ = g_descriptorPool->allocate({ descriptorSetLayout_ }).front();
        }

        void updateDescriptorSet(const BufferRegion& as, const BufferRegion& bs, const BufferRegion& cs) {
            std::array<VkDescriptorBufferInfo, 3> infos{{
                { as.buffer->buffer, as.offset, size(as) },
                { bs.buffer->buffer, bs.offset, size(bs) },
                { cs.buffer->buffer, cs.offset, size(cs) }
            }};

            auto writes = initializers::writeDescriptorSets<3>(descriptorSet_);
            for(uint32_t i = 0; i < COUNT(writes); ++i) {
                writes[i].dstBinding = i;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].descriptorCount = 1;
                writes[i].pBufferInfo = &infos[i];
            }

            device->updateDescriptorSets(writes);
        }

        std::vector<PipelineMetaData> pipelineMetaData() override {
            return {
                {
                    "math_operations",
                    data_shaders_algorithm_math_operations_comp,
                    { &descriptorSetLayout_ },
                    { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants) } }
                }
            };
        }

        static VkDeviceSize size(const BufferRegion& region) {
            return region.end == VK_WHOLE_SIZE ? region.buffer->size - region.offset : region.size();
        }

        static uint32_t sizeAsFloat(const BufferRegion& region) {
            return static_cast<uint32_t>(size(region)/sizeof(float));
        }

        VulkanDescriptorSetLayout descriptorSetLayout_;
        VkDescriptorSet descriptorSet_{};
    };

    static PrefixScan* g_scan{};
    static AlgorithmPipelines* g_algoPipelines{};
    static MathOperations* g_mathOperations{};

    void init(VulkanDevice &device, FileManager &fileManager) {
        g_device = &device;
        g_fileMgr = &fileManager;
        g_descriptorPool = new VulkanDescriptorPool{ createDescriptorPool(device) };
        g_scan = new PrefixScan{};
        g_algoPipelines = new AlgorithmPipelines{};
        g_mathOperations = new MathOperations{};
        g_scan->init();
        g_algoPipelines->init();
        g_mathOperations->init();

        if(device.queueFamilyIndex.compute.has_value()){
            g_commandPool = const_cast<VulkanCommandPool*>(&device.computeCommandPool());
        } else {
            g_commandPool = const_cast<VulkanCommandPool*>(&device.graphicsCommandPool());
        }
    }

    VkPipeline pipeline(const std::string& name){
        assert(g_algoPipelines);
        return g_algoPipelines->pipeline(name);
    }

    VkPipelineLayout layout(const std::string& name){
        assert(g_algoPipelines);
        return g_algoPipelines->layout(name);
    }

    void shutdown(){
        delete g_scan;
        delete g_algoPipelines;
        delete g_mathOperations;
        delete g_descriptorPool;
        g_scan = nullptr;
        g_algoPipelines = nullptr;
        g_mathOperations = nullptr;
        g_descriptorPool = nullptr;
    }

    float average(VulkanBuffer &buffer) {
        assert(g_scan != nullptr);
        g_commandPool->oneTimeCommand([&](auto commandBuffer){
            g_scan->scan(commandBuffer, buffer);
        });
        auto count = buffer.sizeAs<float>();
        auto sum = g_scan->lastSum();
        return sum/static_cast<float>(count);
    }

    void average(VkCommandBuffer commandBuffer, VkDescriptorSet inOutDescriptorSet, VulkanBuffer& input) {
        assert(g_scan != nullptr);
        g_scan->scan(commandBuffer, input, inOutDescriptorSet);
        static std::array<int, 2> constants;
        constants[0] = input.sizeAs<float>();
        constants[1] = g_scan->lastSumIndex();

        static std::array<VkDescriptorSet, 2> sets;
        sets[0] = inOutDescriptorSet;
        sets[1] = g_scan->sumScanDescriptorSet;

        addBufferMemoryBarriers(commandBuffer, {&input, &g_scan->sumsBuffer});
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("average"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("average"), 0, COUNT(sets), sets.data(), 0, nullptr);
        vkCmdPushConstants(commandBuffer, layout("average"), VK_SHADER_STAGE_COMPUTE_BIT, 0, BYTE_SIZE(constants), constants.data());
        vkCmdDispatch(commandBuffer, 1, 1, 1);

    }

    void average(VulkanBuffer &input, VulkanBuffer &output) {
        assert(g_scan != nullptr);
        g_commandPool->oneTimeCommand([&](auto commandBuffer){
            g_scan->scan(commandBuffer, input);
            auto offset = g_scan->lastSumOffset();
            VkBufferCopy region{offset, 0u, sizeof(float)};
            vkCmdCopyBuffer(commandBuffer, g_scan->sumsBuffer, output, 1, &region);
        });
    }

    void reduce(VkCommandBuffer commandBuffer, VkDescriptorSet inOutDescriptorSet, VulkanBuffer& input, Operation operation) {
        assert(g_scan != nullptr);
        g_scan->reduce(commandBuffer, input, inOutDescriptorSet);
    }

    void add(VkCommandBuffer commandBuffer, const BufferRegion& a, const BufferRegion& b, const BufferRegion& c) {
        assert(g_mathOperations != nullptr);
        g_mathOperations->execute(commandBuffer, a, b, c, Operation::ADD);
    }

    void subtract(VkCommandBuffer commandBuffer, const BufferRegion& a, const BufferRegion& b, const BufferRegion& c) {
        assert(g_mathOperations != nullptr);
        g_mathOperations->execute(commandBuffer, a, b, c, Operation::SUBTRACT);
    }

    void multiply(VkCommandBuffer commandBuffer, const BufferRegion& a, const BufferRegion& b, const BufferRegion& c) {
        assert(g_mathOperations != nullptr);
        // BUG: This updates a shared descriptor set; repeated calls while recording the same command buffer can invalidate it.
        g_mathOperations->execute(commandBuffer, a, b, c, Operation::MULTIPLY);
    }

    void divide(VkCommandBuffer commandBuffer, const BufferRegion& a, const BufferRegion& b, const BufferRegion& c) {
        assert(g_mathOperations != nullptr);
        g_mathOperations->execute(commandBuffer, a, b, c, Operation::DIVIDE);
    }

    float lastSum() {
        return g_scan->lastSum();
    }
}
