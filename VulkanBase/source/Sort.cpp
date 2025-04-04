#include "Sort.hpp"
#include "vulkan_util.h"
#include "VulkanInitializers.h"
#include "glsl_shaders.hpp"
#include "Barrier.hpp"

RadixSort::RadixSort(VulkanDevice *device, bool debug)
: GpuSort(device)
, debug(debug)
{

}

void RadixSort::init() {
    createDescriptorPool();
    createDescriptorSetLayouts();
    createDescriptorSets();
    resizeInternalBuffer();
    createPipelines();
    createProfiler();
}

void RadixSort::enableOrderChecking() {
    _orderChecker = OrderChecker{device, INITIAL_CAPACITY};
    _orderChecker->init();
}

void RadixSort::disableOrderChecking() {
    _orderChecker = {};
}

void RadixSort::createDescriptorPool() {
    constexpr uint maxSets = 6;
    std::vector<VkDescriptorPoolSize> poolSizes{
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, maxSets * 10}
    };

    descriptorPool = device->createDescriptorPool(maxSets, poolSizes);
}

void RadixSort::createDescriptorSetLayouts() {
    std::vector<VkDescriptorSetLayoutBinding> bindings(3);

    bindings[DATA].binding = DATA;
    bindings[DATA].descriptorCount = NUM_DATA_ELEMENTS;
    bindings[DATA].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[DATA].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[INDICES].binding = INDICES;
    bindings[INDICES].descriptorCount = NUM_DATA_ELEMENTS;
    bindings[INDICES].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[INDICES].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[RECORDS].binding = RECORDS;
    bindings[RECORDS].descriptorCount = NUM_DATA_ELEMENTS;
    bindings[RECORDS].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[RECORDS].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    dataSetLayout = device->createDescriptorSetLayout(bindings);

    bindings.resize(3);
    bindings[COUNTS].binding = COUNTS;
    bindings[COUNTS].descriptorCount = NUM_DATA_ELEMENTS;
    bindings[COUNTS].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[COUNTS].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[SUMS].binding = SUMS;
    bindings[SUMS].descriptorCount = NUM_DATA_ELEMENTS;
    bindings[SUMS].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[SUMS].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[ORDER_CHECKING].binding = ORDER_CHECKING;
    bindings[ORDER_CHECKING].descriptorCount = NUM_DATA_ELEMENTS;
    bindings[ORDER_CHECKING].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[ORDER_CHECKING].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    countsSetLayout = device->createDescriptorSetLayout(bindings);

    bindings.resize(1);
    bindings[0].binding = 0;
    bindings[0].descriptorCount = 1;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bitFlipSetLayout = device->createDescriptorSetLayout(bindings);

    bindings.resize(1);
    bindings[0].binding = 0;
    bindings[0].descriptorCount = 1;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    sequenceSetLayout = device->createDescriptorSetLayout(bindings);
}

void RadixSort::createDescriptorSets() {
    auto sets = descriptorPool.allocate({dataSetLayout, dataSetLayout, countsSetLayout, bitFlipSetLayout, sequenceSetLayout});
    dataDescriptorSets[DATA_IN] = sets[0];
    dataDescriptorSets[DATA_OUT] = sets[1];
    countsDescriptorSet = sets[2];
    bitFlipDescriptorSet = sets[3];
    sequenceDescriptorSet = sets[4];

    device->setName<VK_OBJECT_TYPE_DESCRIPTOR_SET>("data_in", dataDescriptorSets[DATA_IN]);
    device->setName<VK_OBJECT_TYPE_DESCRIPTOR_SET>("data_out", dataDescriptorSets[DATA_OUT]);
    device->setName<VK_OBJECT_TYPE_DESCRIPTOR_SET>("counts", countsDescriptorSet);
    device->setName<VK_OBJECT_TYPE_DESCRIPTOR_SET>("add_value", countsDescriptorSet);
    device->setName<VK_OBJECT_TYPE_DESCRIPTOR_SET>("radix_sort_bit_flip", bitFlipDescriptorSet);
    device->setName<VK_OBJECT_TYPE_DESCRIPTOR_SET>("sequence_generator", sequenceDescriptorSet);
}


std::vector<PipelineMetaData> RadixSort::pipelineMetaData() {
    return {
            {
                "radix_sort_count_radices",
                data_shaders_radix_sort_li_grand_count_radices_comp,
                {&dataSetLayout, &countsSetLayout},
                { {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants)}}
            },
            {
                "radix_sort_prefix_sum",
                data_shaders_radix_sort_li_grand_prefix_sum_comp,
                { &countsSetLayout },
                { {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants)}}
            },
            {
                "radix_sort_reorder",
                data_shaders_radix_sort_li_grand_reorder_comp,
                {&dataSetLayout, &dataSetLayout, &countsSetLayout},
                { {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants)}}
            },
            {
                "radix_sort_reorder_indices",
                data_shaders_radix_sort_li_grand_reorder_indices_comp,
                {&dataSetLayout, &dataSetLayout, &countsSetLayout},
                { {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants)}}
            },
            {
                "radix_sort_reorder_records",
                data_shaders_radix_sort_li_grand_reorder_records_comp,
                {&dataSetLayout, &dataSetLayout, &countsSetLayout},
                { {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants)}}
            },
            {
                "radix_sort_bit_flip",
                data_shaders_bit_flip_comp,
                { &bitFlipSetLayout},
                { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(bitFlipConstants)}}
            },
            {
                "radix_sort_sequence",
                data_shaders_sequence_comp,
                { &sequenceSetLayout },
                { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(seqConstants)}}
            }
    };
}

void RadixSort::sortWithIndices(VkCommandBuffer commandBuffer, VulkanBuffer &buffer, VulkanBuffer& indexes) {
    operator()(commandBuffer, { &buffer, 0, buffer.size}, REORDER_TYPE_INDEXES);
    copyFromInternalIndexBuffer(commandBuffer, { &indexes, 0, indexes.size });
}

void RadixSort::operator()(VkCommandBuffer commandBuffer, VulkanBuffer &keys, Records &records) {
    if(records.size % sizeof(uint) != 0) {
        throw std::runtime_error{ "record size should be multiple of uint size" };
    }
    if(records.keyType != KeyType::Uint) {
        bitFlipConstants.numEntries = keys.sizeAs<uint>();
        bitFlipConstants.reverse = 0;
        bitFlipConstants.dataType = static_cast<uint>(records.keyType);
        flipBits(commandBuffer, keys);
    }

    constants.recordSize = records.size/sizeof(uint);
    BufferRegion recordRegion{ &records.buffer, 0, records.buffer.size };
    copyToInternalRecordBuffer(commandBuffer, recordRegion);
    operator()(commandBuffer, { &keys, 0, keys.size }, REORDER_TYPE_RECORDS);

    if(records.keyType != KeyType::Uint) {
        bitFlipConstants.reverse = 1;
        flipBits(commandBuffer, keys);
    }
    copyFromInternalRecordBuffer(commandBuffer, recordRegion);
}

void RadixSort::operator()(VkCommandBuffer commandBuffer, VulkanBuffer &buffer) {
    operator()(commandBuffer, { &buffer, 0, buffer.size }, REORDER_TYPE_KEYS);
}

void RadixSort::operator()(VkCommandBuffer commandBuffer, const BufferRegion &region) {
    operator()(commandBuffer, region, REORDER_TYPE_KEYS);
}

void RadixSort::operator()(VkCommandBuffer commandBuffer, const BufferRegion &region, std::string_view reorderPipeline) {
    if(capacity < region.size()){
        capacity = region.size() * 2;
        resizeInternalBuffer();
    }
    updateConstants(region);
    copyToInternalKeyBuffer(commandBuffer, region);

    static std::array<VkDescriptorSet, 2> localDataSets;
    std::copy(begin(dataDescriptorSets), end(dataDescriptorSets), begin(localDataSets));

    if(reorderPipeline == REORDER_TYPE_INDEXES) {
        generateSequence(commandBuffer, region.sizeAs<uint32_t>());
    }


    for(auto block = 0; block < PASSES; block++){
        constants.block = block;
        checkOrder(commandBuffer, { &internal.keys[DATA_IN], 0, region.size()  });
        count(commandBuffer, localDataSets[DATA_IN]);
        prefixSum(commandBuffer);
        reorder(commandBuffer, localDataSets, reorderPipeline);
        std::swap(localDataSets[DATA_IN], localDataSets[DATA_OUT]);
    }
    copyFromInternalKeyBuffer(commandBuffer, region);
    previousBuffer = *region.buffer;
}


void RadixSort::generateSequence(VkCommandBuffer commandBuffer, uint32_t numEntries) {
    seqConstants.start = 0;
    seqConstants.numEntries = numEntries;
    const auto gx = glm::max(1u, seqConstants.numEntries);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("radix_sort_sequence"));
    vkCmdPushConstants(commandBuffer, layout("radix_sort_sequence"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(seqConstants), &seqConstants);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("radix_sort_sequence"), 0, 1, &sequenceDescriptorSet, 0, VK_NULL_HANDLE);
    vkCmdDispatch(commandBuffer, gx, 1, 1);
    Barrier::computeWriteToRead(commandBuffer, { internal.indexes[DATA_IN] });
}

void RadixSort::flipBits(VkCommandBuffer commandBuffer, VulkanBuffer &buffer) {
    static constexpr uint32_t wgSize = 256;
    const auto gx =  (buffer.sizeAs<uint32_t>() + wgSize)/wgSize;

    updateBitFlipDescriptorSet(buffer);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("radix_sort_bit_flip"));
    vkCmdPushConstants(commandBuffer, layout("radix_sort_bit_flip"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(bitFlipConstants), &bitFlipConstants);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("radix_sort_bit_flip"), 0, 1, &bitFlipDescriptorSet, 0, VK_NULL_HANDLE);
    vkCmdDispatch(commandBuffer, gx, 1, 1);
    Barrier::computeWriteToRead(commandBuffer, { buffer });
}

void RadixSort::updateBitFlipDescriptorSet(VulkanBuffer &buffer) {
    if(previousBuffer == buffer.buffer) return;
    auto writes = initializers::writeDescriptorSets();

    writes[0].dstSet = bitFlipDescriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].descriptorCount = 1;
    VkDescriptorBufferInfo info{ buffer, 0, VK_WHOLE_SIZE };
    writes[0].pBufferInfo = &info;

    device->updateDescriptorSets(writes);
}

void RadixSort::updateConstants(const BufferRegion& region) {
    workGroupCount = numWorkGroups(region);
    constants.Num_Elements = region.size()/sizeof(uint);
    constants.Num_Groups_per_WorkGroup = NUM_THREADS_PER_BLOCK / WORD_SIZE;
    constants.Num_Elements_per_WorkGroup = nearestMultiple(constants.Num_Elements / workGroupCount , NUM_THREADS_PER_BLOCK);
    constants.Num_Elements_Per_Group = constants.Num_Elements_per_WorkGroup / constants.Num_Groups_per_WorkGroup;
    constants.Num_Radices_Per_WorkGroup = RADIX  / workGroupCount;
    constants.Num_Groups = workGroupCount * constants.Num_Groups_per_WorkGroup;

}
void RadixSort::updateDataDescriptorSets() {
    auto workGroupCount = numWorkGroups({ &internal.keys[0], 0, internal.keys[0].size});

    auto dataWrites = initializers::writeDescriptorSets<6>();

    std::array<VkDescriptorBufferInfo, 1> dataInfos{};
    dataInfos[KEY] = { internal.keys[0], 0, internal.keys[0].size };

    dataWrites[DATA_IN].dstSet = dataDescriptorSets[DATA_IN];
    dataWrites[DATA_IN].dstBinding = DATA;
    dataWrites[DATA_IN].descriptorCount = NUM_DATA_ELEMENTS;
    dataWrites[DATA_IN].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dataWrites[DATA_IN].pBufferInfo = dataInfos.data();

    std::array<VkDescriptorBufferInfo, 1> dataOutInfos{};
    dataOutInfos[KEY] = {internal.keys[1], 0, internal.keys[1].size};

    dataWrites[DATA_OUT].dstSet = dataDescriptorSets[DATA_OUT];
    dataWrites[DATA_OUT].dstBinding = DATA;
    dataWrites[DATA_OUT].descriptorCount = NUM_DATA_ELEMENTS;
    dataWrites[DATA_OUT].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dataWrites[DATA_OUT].pBufferInfo = dataOutInfos.data();

    dataWrites[INDEX_IN].dstSet = dataDescriptorSets[DATA_IN];
    dataWrites[INDEX_IN].dstBinding = INDICES;
    dataWrites[INDEX_IN].descriptorCount = NUM_DATA_ELEMENTS;
    dataWrites[INDEX_IN].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    VkDescriptorBufferInfo indexInInfo{ internal.indexes[DATA_IN], 0, internal.indexes[DATA_IN].size};
    dataWrites[INDEX_IN].pBufferInfo = &indexInInfo;

    dataWrites[INDEX_OUT].dstSet = dataDescriptorSets[DATA_OUT];
    dataWrites[INDEX_OUT].dstBinding = INDICES;
    dataWrites[INDEX_OUT].descriptorCount = NUM_DATA_ELEMENTS;
    dataWrites[INDEX_OUT].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    VkDescriptorBufferInfo indexOutInfo{ internal.indexes[DATA_OUT], 0, internal.indexes[DATA_IN].size};
    dataWrites[INDEX_OUT].pBufferInfo = &indexOutInfo;

    dataWrites[RECORDS_IN].dstSet = dataDescriptorSets[DATA_IN];
    dataWrites[RECORDS_IN].dstBinding = RECORDS;
    dataWrites[RECORDS_IN].descriptorCount = NUM_DATA_ELEMENTS;
    dataWrites[RECORDS_IN].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    VkDescriptorBufferInfo recordsInInfo{ internal.records[DATA_IN], 0, internal.records[DATA_IN].size};
    dataWrites[RECORDS_IN].pBufferInfo = &recordsInInfo;

    dataWrites[RECORDS_OUT].dstSet = dataDescriptorSets[DATA_OUT];
    dataWrites[RECORDS_OUT].dstBinding = RECORDS;
    dataWrites[RECORDS_OUT].descriptorCount = NUM_DATA_ELEMENTS;
    dataWrites[RECORDS_OUT].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    VkDescriptorBufferInfo recordsOutInfo{ internal.records[DATA_OUT], 0, internal.records[DATA_IN].size};
    dataWrites[RECORDS_OUT].pBufferInfo = &recordsOutInfo;

    device->updateDescriptorSets(dataWrites);

    VkDeviceSize countsSize = RADIX * workGroupCount * NUM_GROUPS_PER_WORKGROUP * sizeof(uint);
    countsBuffer = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, countsSize);
    auto countWrites = initializers::writeDescriptorSets<3>();

    VkDescriptorBufferInfo countsInfo{countsBuffer, 0, countsBuffer.size };
    countWrites[COUNTS].dstSet = countsDescriptorSet;
    countWrites[COUNTS].dstBinding = COUNTS;
    countWrites[COUNTS].descriptorCount = 1;
    countWrites[COUNTS].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    countWrites[COUNTS].pBufferInfo = &countsInfo;

    VkDeviceSize sumSize = (RADIX + 1) * sizeof(uint);
    sumBuffer = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, sumSize);
    VkDescriptorBufferInfo sumInfo{sumBuffer, 0, sumBuffer.size};
    countWrites[SUMS].dstSet = countsDescriptorSet;
    countWrites[SUMS].dstBinding = SUMS;
    countWrites[SUMS].descriptorCount = 1;
    countWrites[SUMS].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    countWrites[SUMS].pBufferInfo = &sumInfo;

    uint unOrdered = 0xFFFFFFFF;
    orderBuffer = device->createDeviceLocalBuffer(&unOrdered, DataUnitSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    VkDescriptorBufferInfo orderCheckingInfo{ orderBuffer, 0, orderBuffer.size };
    countWrites[ORDER_CHECKING].dstSet = countsDescriptorSet;
    countWrites[ORDER_CHECKING].dstBinding = ORDER_CHECKING;
    countWrites[ORDER_CHECKING].descriptorCount = 1;
    countWrites[ORDER_CHECKING].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    countWrites[ORDER_CHECKING].pBufferInfo = &orderCheckingInfo;

    device->updateDescriptorSets(countWrites);

    auto seqWrites = initializers::writeDescriptorSets();
    seqWrites[0].dstSet = sequenceDescriptorSet;
    seqWrites[0].dstBinding = 0;
    seqWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    seqWrites[0].descriptorCount = 1;
    VkDescriptorBufferInfo info{ internal.indexes[DATA_IN], 0, VK_WHOLE_SIZE };
    seqWrites[0].pBufferInfo = &info;
    device->updateDescriptorSets(seqWrites);
}

uint RadixSort::numWorkGroups(const BufferRegion& region) {
    const uint Num_Elements = alignedSize(region.size()/sizeof(uint), ELEMENTS_PER_WG);
    auto count = std::min(std::max(1u, Num_Elements/ ELEMENTS_PER_WG), MAX_WORKGROUPS);

    return nearestPowerOfTwo(count);
}

void RadixSort::checkOrder(VkCommandBuffer commandBuffer, const BufferRegion& data) {
    if(_orderChecker.has_value()) {
        (*_orderChecker)(commandBuffer, data, {&orderBuffer, 0, DataUnitSize });
    }
}

void RadixSort::count(VkCommandBuffer commandBuffer, VkDescriptorSet dataDescriptorSet) {
    static std::array<VkDescriptorSet, 2> sets{};
    sets[0] = dataDescriptorSet;
    sets[1] = countsDescriptorSet;
    auto query = fmt::format("{}_{}", "count", constants.block);
    profiler.profile(query, commandBuffer, [&]{
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("radix_sort_count_radices"));
        vkCmdPushConstants(commandBuffer, layout("radix_sort_count_radices"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("radix_sort_count_radices"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdDispatch(commandBuffer, workGroupCount, 1, 1);
        Barrier::computeWriteToRead(commandBuffer, { internal.keys[DATA_IN], countsBuffer });
    });

}

void RadixSort::prefixSum(VkCommandBuffer commandBuffer) {
    auto query = fmt::format("{}_{}", "radix_sort_prefix_sum", constants.block);
    profiler.profile(query, commandBuffer,  [&] {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("radix_sort_prefix_sum"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("radix_sort_prefix_sum"), 0, 1,  &countsDescriptorSet, 0, VK_NULL_HANDLE);
        vkCmdPushConstants(commandBuffer, layout("radix_sort_prefix_sum"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
        vkCmdDispatch(commandBuffer, workGroupCount, 1, 1);
        Barrier::computeWriteToRead(commandBuffer,{countsBuffer, sumBuffer, internal.keys[DATA_IN], internal.keys[DATA_OUT]});
    });
}

void RadixSort::reorder(VkCommandBuffer commandBuffer, std::array<VkDescriptorSet, 2> &dataDescriptorSets, std::string_view reorderPipeline) {
    static std::array<VkDescriptorSet, 3> sets{};
    sets[0] = dataDescriptorSets[DATA_IN];
    sets[1] = dataDescriptorSets[DATA_OUT];
    sets[2] = countsDescriptorSet;
    auto query = fmt::format("{}_{}", "reorder", constants.block);
    profiler.profile(query, commandBuffer, [&] {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline(reorderPipeline.data()));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout(reorderPipeline.data()), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdPushConstants(commandBuffer, layout(reorderPipeline.data()), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
        vkCmdDispatch(commandBuffer, workGroupCount, 1, 1);
        if (constants.block < PASSES - 1) {
            Barrier::computeWriteToRead(commandBuffer,
               { internal.keys[DATA_IN], internal.keys[DATA_OUT],
                 internal.indexes[DATA_IN], internal.indexes[DATA_OUT],
                 internal.records[DATA_IN], internal.records[DATA_OUT]
               });
        }
    });
}

void RadixSort::createProfiler() {
    if(debug) {
        profiler = Profiler{device};
        profiler.addGroup("copy_to_key", 1);
        profiler.addGroup("copy_from_key", 1);
        profiler.addGroup("count", PASSES);
        profiler.addGroup("radix_sort_prefix_sum", PASSES);
        profiler.addGroup("reorder", PASSES);
    }
}

void RadixSort::resizeInternalBuffer() {
    internal.keys[0] =
        device->createBuffer(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY, capacity, "radix_sort_keys_0");
    internal.keys[1] =
        device->createBuffer(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY, capacity, "radix_sort_keys_1");

    internal.indexes[0] =
        device->createBuffer(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY, capacity, "radix_sort_indexes_0");

    internal.indexes[1] =
        device->createBuffer(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY, capacity, "radix_sort_indexes_1");

    internal.records[0] =
            device->createBuffer(
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_GPU_ONLY, capacity * NUM_ENTRIES_PER_RECORD, "radix_sort_records_0");

    internal.records[1] =
            device->createBuffer(
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_GPU_ONLY, capacity * NUM_ENTRIES_PER_RECORD, "radix_sort_records_1");

    updateDataDescriptorSets();
}

void RadixSort::commitProfiler() {
    profiler.commit();
}

void RadixSort::copyToInternalKeyBuffer(VkCommandBuffer commandBuffer, const BufferRegion &src) {
    profiler.profile("copy_to_key_0", commandBuffer, [&]{
        BufferRegion dst{ &internal.keys[0], 0, src.size() };
        copyBuffer(commandBuffer, src, dst);
        Barrier::transferWriteToComputeRead(commandBuffer, { dst });
    });
}

void RadixSort::copyFromInternalKeyBuffer(VkCommandBuffer commandBuffer, const BufferRegion &dst) {
    profiler.profile("copy_from_key_0", commandBuffer, [&]{
        BufferRegion src{ &internal.keys[0], 0, dst.size() };
        Barrier::computeWriteToTransferRead(commandBuffer, { src });
        copyBuffer(commandBuffer, src, dst);
        Barrier::transferWriteToComputeRead(commandBuffer, { dst });
    });
}

void RadixSort::copyFromInternalIndexBuffer(VkCommandBuffer commandBuffer, const BufferRegion &dst) {
    BufferRegion src{ &internal.indexes[0], 0, dst.size() };

    Barrier::computeWriteToTransferRead(commandBuffer, { src });
    copyBuffer(commandBuffer, src, dst);
    Barrier::transferWriteToComputeRead(commandBuffer, { dst });
}

void RadixSort::copyBuffer(VkCommandBuffer commandBuffer, const BufferRegion &src, const BufferRegion &dst) {
    assert(dst.size() >= src.size());
    VkBufferCopy region{ src.offset, dst.offset, src.size() };
    vkCmdCopyBuffer(commandBuffer, *src.buffer, *dst.buffer, 1, &region);
}

void RadixSort::copyToInternalRecordBuffer(VkCommandBuffer commandBuffer, const BufferRegion &src) {
    BufferRegion dst{ &internal.records[0], 0, src.size() };
    copyBuffer(commandBuffer, src, dst);
    Barrier::transferWriteToComputeRead(commandBuffer, { dst });
}

void RadixSort::copyFromInternalRecordBuffer(VkCommandBuffer commandBuffer, const BufferRegion &dst) {
    BufferRegion src{ &internal.records[0], 0, dst.size() };

    Barrier::computeWriteToTransferRead(commandBuffer, { src });
    copyBuffer(commandBuffer, src, dst);
    Barrier::transferWriteToComputeRead(commandBuffer, { dst });
}


