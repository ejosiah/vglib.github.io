#include "PointHashGrid.hpp"
#include "glsl_shaders.hpp"

PointHashGrid::PointHashGrid(VulkanDevice* device, VulkanDescriptorPool* descriptorPool, VulkanDescriptorSetLayout* particleDescriptorSetLayout, glm::vec3 resolution, float gridSpacing)
        :
        ComputePipelines(device),
        device(device),
        descriptorPool(descriptorPool),
        particleDescriptorSetLayout(particleDescriptorSetLayout)
{
    constants.resolution = resolution;
    constants.gridSpacing = gridSpacing;
    bufferOffsetAlignment = device->getLimits().minStorageBufferOffsetAlignment;

}

void PointHashGrid::init() {
    updateGridBuffer();
    createDescriptorSetLayouts();
    createPrefixScanDescriptorSetLayouts();
    createDescriptorSets();
    updateDescriptorSet();
    updateScanDescriptorSet();
    createNeighbourListSetLayout();
    createPipelines();
}

void PointHashGrid::initNeighbourList(){
    initNeighbourListBuffers();
    createNeighbourListDescriptorSets();
    updateNeighbourListScanDescriptorSet();
}

void PointHashGrid::initNeighbourListBuffers(){
    neighbourList.neighbourListBuffer = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, LIST_HEAP_SIZE);
    neighbourList.atomicCounterBuffer = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, constants.numParticles * sizeof(int));

    VkDeviceSize size = constants.numParticles * sizeof(int);
    neighbourList.neighbourSizeBuffer = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, size);
    neighbourList.neighbourOffsetsBuffer = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, size);

    VkDeviceSize sumsSize = (std::abs(int(constants.numParticles - 1))/ITEMS_PER_WORKGROUP + 1) * sizeof(int);
    sumsSize = alignedSize(sumsSize, bufferOffsetAlignment) + sizeof(int);
    neighbourList.prefixScan.sumsBuffer = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, sumsSize);
    neighbourList.prefixScan.constants.N = constants.numParticles/sizeof(int);
}

void PointHashGrid::createDescriptorSetLayouts() {
    std::vector<VkDescriptorSetLayoutBinding> bindings(1);
    bindings.resize(2);
    bindings[0].binding = 0;
    bindings[0].descriptorCount = 1;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorCount = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    gridDescriptorSetLayout = device->createDescriptorSetLayout(bindings);

    bindings.resize(1);
    bindings[0].binding = 0;
    bindings[0].descriptorCount = 1;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bucketSizeSetLayout = device->createDescriptorSetLayout(bindings);

    bindings.resize(2);
    bindings[0].binding = 0;
    bindings[0].descriptorCount = 1;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorCount = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bucket.setLayout = device->createDescriptorSetLayout(bindings);
}

void PointHashGrid::createNeighbourListSetLayout() {
    std::vector<VkDescriptorSetLayoutBinding> bindings(2);
    bindings[0].binding = 0;
    bindings[0].descriptorCount = 1;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorCount = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    neighbourList.setLayout = device->createDescriptorSetLayout(bindings);

    bindings.resize(1);
    bindings[0].binding = 0;
    bindings[0].descriptorCount = 1;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    neighbourList.neighbourSizeSetLayout = device->createDescriptorSetLayout(bindings);
}

void PointHashGrid::createPrefixScanDescriptorSetLayouts() {
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};

    bindings[0].binding = 0;
    bindings[0].descriptorCount = 1;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorCount = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    prefixScan.setLayout = device->createDescriptorSetLayout(bindings);

    neighbourList.prefixScan.setLayout = device->createDescriptorSetLayout(bindings);
}

void PointHashGrid::createDescriptorSets() {
    auto sets = descriptorPool->allocate({
                                          gridDescriptorSetLayout,
                                          prefixScan.setLayout, prefixScan.setLayout, bucketSizeSetLayout, bucketSizeSetLayout, bucket.setLayout } );
    gridDescriptorSet = sets[0];
    prefixScan.descriptorSet = sets[1];
    prefixScan.sumScanDescriptorSet = sets[2];
    bucketSizeDescriptorSet = sets[3];
    bucketSizeOffsetDescriptorSet = sets[4];
    bucket.descriptorSet = sets[5];
}

void PointHashGrid::createNeighbourListDescriptorSets() {
    auto sets = descriptorPool->allocate({ neighbourList.setLayout, neighbourList.neighbourSizeSetLayout,
                                           neighbourList.neighbourSizeSetLayout, neighbourList.prefixScan.setLayout,
                                           neighbourList.prefixScan.setLayout});

    neighbourList.descriptorSet = sets[0];
    neighbourList.neighbourSizeDescriptorSet = sets[1];
    neighbourList.neighbourSizeOffsetDescriptorSet = sets[2];
    neighbourList.prefixScan.descriptorSet = sets[3];
    neighbourList.prefixScan.sumScanDescriptorSet = sets[4];

    auto writes = initializers::writeDescriptorSets<4>();

    VkDescriptorBufferInfo neighbourListInfo{ neighbourList.neighbourListBuffer, 0, VK_WHOLE_SIZE };
    writes[0].dstSet = neighbourList.descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &neighbourListInfo;

    VkDescriptorBufferInfo counterInfo{ neighbourList.atomicCounterBuffer, 0, VK_WHOLE_SIZE };
    writes[1].dstSet = neighbourList.descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &counterInfo;

    VkDescriptorBufferInfo neighbourListSizeInfo{ neighbourList.neighbourSizeBuffer, 0, VK_WHOLE_SIZE };
    writes[2].dstSet = neighbourList.neighbourSizeDescriptorSet;
    writes[2].dstBinding = 0;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &neighbourListSizeInfo;

    VkDescriptorBufferInfo neighbourListSizeOffsetInfo{ neighbourList.neighbourOffsetsBuffer, 0, VK_WHOLE_SIZE };
    writes[3].dstSet = neighbourList.neighbourSizeOffsetDescriptorSet;
    writes[3].dstBinding = 0;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[3].pBufferInfo = &neighbourListSizeOffsetInfo;



    device->updateDescriptorSets(writes);
}

void PointHashGrid::updateDescriptorSet() {
    auto writes = initializers::writeDescriptorSets<3>();

    // grid descriptor set
    VkDescriptorBufferInfo  nextBucketIndexInfo{nextBufferIndexBuffer, 0, VK_WHOLE_SIZE };
    writes[0].dstSet = gridDescriptorSet;
    writes[0].dstBinding = 1;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &nextBucketIndexInfo;

    // bucket size _ offset
    VkDescriptorBufferInfo  bucketSizeInfo{bucketSizeBuffer, 0, VK_WHOLE_SIZE };
    writes[1].dstSet = bucketSizeDescriptorSet;
    writes[1].dstBinding = 0;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &bucketSizeInfo;

    VkDescriptorBufferInfo  bucketOffsetInfo{bucketSizeOffsetBuffer, 0, VK_WHOLE_SIZE };
    writes[2].dstSet = bucketSizeOffsetDescriptorSet;
    writes[2].dstBinding = 0;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &bucketOffsetInfo;

    device->updateDescriptorSets(writes);
    updateBucketDescriptor();
}

void PointHashGrid::updateBucketDescriptor() {
    auto writes = initializers::writeDescriptorSets<2>();

    VkDescriptorBufferInfo  bucketSizeInfo{bucketSizeBuffer, 0, VK_WHOLE_SIZE };
    writes[0].dstSet = bucket.descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &bucketSizeInfo;

    VkDescriptorBufferInfo  bucketOffsetInfo{bucketSizeOffsetBuffer, 0, VK_WHOLE_SIZE };
    writes[1].dstSet = bucket.descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &bucketOffsetInfo;

    device->updateDescriptorSets(writes);
}

void PointHashGrid::updateGridBuffer() {
    auto res = constants.resolution;
    VkDeviceSize gridSize = res.x * res.y * res.z * sizeof(uint32_t);

    nextBufferIndexBuffer = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, gridSize);
    bucketSizeBuffer = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, gridSize);
    bucketSizeOffsetBuffer = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, gridSize);
    updateScanBuffer();
}

void PointHashGrid::updateScanBuffer() {
    auto res = constants.resolution;
    VkDeviceSize numCells = res.x * res.y * res.z;

    VkDeviceSize sumsSize = (std::abs(int(numCells - 1))/ITEMS_PER_WORKGROUP + 1) * sizeof(int);
    sumsSize = alignedSize(sumsSize, bufferOffsetAlignment) + sizeof(int);
    prefixScan.sumsBuffer = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, sumsSize);
    prefixScan.constants.N = constants.numParticles/sizeof(int);
}

void PointHashGrid::updateScanDescriptorSet() {
    VkDescriptorBufferInfo info{bucketSizeOffsetBuffer, 0, VK_WHOLE_SIZE};
    auto writes = initializers::writeDescriptorSets<4>(prefixScan.descriptorSet);
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &info;

    VkDescriptorBufferInfo sumsInfo{ prefixScan.sumsBuffer, 0, prefixScan.sumsBuffer.size - sizeof(int)};
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &sumsInfo;

    // for sum scan
    writes[2].dstSet = prefixScan.sumScanDescriptorSet;
    writes[2].dstBinding = 0;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &sumsInfo;

    VkDescriptorBufferInfo sumsSumInfo{ prefixScan.sumsBuffer, prefixScan.sumsBuffer.size - sizeof(int), VK_WHOLE_SIZE}; // TODO FIXME
    writes[3].dstSet = prefixScan.sumScanDescriptorSet;
    writes[3].dstBinding = 1;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[3].pBufferInfo = &sumsSumInfo;

    device->updateDescriptorSets(writes);
}

void PointHashGrid::updateNeighbourListScanDescriptorSet() {
    VkDescriptorBufferInfo info{neighbourList.neighbourOffsetsBuffer, 0, VK_WHOLE_SIZE};
    auto writes = initializers::writeDescriptorSets<4>(neighbourList.prefixScan.descriptorSet);
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &info;

    VkDescriptorBufferInfo sumsInfo{ neighbourList.prefixScan.sumsBuffer, 0, neighbourList.prefixScan.sumsBuffer.size - sizeof(int)};
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &sumsInfo;

    // for sum scan
    writes[2].dstSet = neighbourList.prefixScan.sumScanDescriptorSet;
    writes[2].dstBinding = 0;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &sumsInfo;

    VkDescriptorBufferInfo sumsSumInfo{ neighbourList.prefixScan.sumsBuffer, neighbourList.prefixScan.sumsBuffer.size - sizeof(int), VK_WHOLE_SIZE}; // TODO FIXME
    writes[3].dstSet = neighbourList.prefixScan.sumScanDescriptorSet;
    writes[3].dstBinding = 1;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[3].pBufferInfo = &sumsSumInfo;

    device->updateDescriptorSets(writes);
}

void PointHashGrid::scan(VkCommandBuffer commandBuffer) {

    VkBufferCopy region{0, 0, bucketSizeBuffer.size};
    vkCmdCopyBuffer(commandBuffer, bucketSizeBuffer, bucketSizeOffsetBuffer, 1, &region);
    prefixScan.scan(commandBuffer, bucketSizeOffsetBuffer, pipeline("prefix_scan"), layout("prefix_scan"));
}

void PointHashGrid::scanNeighbourList(VkCommandBuffer commandBuffer) {
    VkBufferCopy region{0, 0, neighbourList.neighbourSizeBuffer.size};
    vkCmdCopyBuffer(commandBuffer, neighbourList.neighbourSizeBuffer, neighbourList.neighbourOffsetsBuffer, 1, &region);
    neighbourList.prefixScan.scan(commandBuffer, neighbourList.neighbourOffsetsBuffer, pipeline("prefix_scan"), layout("prefix_scan"));
}

void PointHashGrid::buildHashGrid(VkCommandBuffer commandBuffer,  VkDescriptorSet particleDescriptorSet) {
    nextBufferIndexBuffer.clear(commandBuffer);
    bucketSizeBuffer.clear(commandBuffer);
    bucketSizeOffsetBuffer.clear(commandBuffer);
    generateHashGrid(commandBuffer, particleDescriptorSet, 0);
    scan(commandBuffer);
    generateHashGrid(commandBuffer, particleDescriptorSet, 1);
}

void PointHashGrid::generateHashGrid(VkCommandBuffer commandBuffer, VkDescriptorSet particleDescriptorSet, int pass) {
    // TODO ADD buffer barriers
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("point_hash_grid_builder"));

    constants.pass = pass;
    vkCmdPushConstants(commandBuffer, layout("point_hash_grid_builder"), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(constants), &constants);

    static std::array<VkDescriptorSet, 3> sets;
    sets[0] = particleDescriptorSet;
    sets[1] = gridDescriptorSet;
    sets[2] = (pass%2) ? bucketSizeOffsetDescriptorSet : bucketSizeDescriptorSet;

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("point_hash_grid_builder"), 0, COUNT(sets), sets.data(), 0, nullptr);
    vkCmdDispatch(commandBuffer, (constants.numParticles - 1)/1024 + 1, 1, 1);
}

void PointHashGrid::generateNeighbourList(VkCommandBuffer commandBuffer, VkDescriptorSet particleDescriptorSet) {
    neighbourList.atomicCounterBuffer.clear(commandBuffer);
    vkCmdFillBuffer(commandBuffer, neighbourList.atomicCounterBuffer, 0, neighbourList.atomicCounterBuffer.size, 0);
    vkCmdFillBuffer(commandBuffer, neighbourList.neighbourSizeBuffer, 0, neighbourList.neighbourSizeBuffer.size, 0);
    vkCmdFillBuffer(commandBuffer, neighbourList.neighbourOffsetsBuffer, 0, neighbourList.neighbourOffsetsBuffer.size, 0);
    generateNeighbourList(commandBuffer, particleDescriptorSet, 0);
    scanNeighbourList(commandBuffer);
    generateNeighbourList(commandBuffer, particleDescriptorSet, 1);
}

void PointHashGrid::generateNeighbourList(VkCommandBuffer commandBuffer, VkDescriptorSet particleDescriptorSet, int pass) {
    static std::array<VkDescriptorSet, 5> sets;
    sets[0] = particleDescriptorSet;
    sets[1] = gridDescriptorSet;
    sets[2] = bucket.descriptorSet;
    sets[3] = neighbourList.descriptorSet;
    sets[4] = (pass % 2) == 0 ? neighbourList.neighbourSizeDescriptorSet : neighbourList.neighbourSizeOffsetDescriptorSet;

    addBufferMemoryBarriers(commandBuffer, { &bucketBuffer, &bucketSizeBuffer, &bucketSizeOffsetBuffer});
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("neighbour_list"));
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("neighbour_list"), 0, COUNT(sets), sets.data(), 0, nullptr);
    constants.pass = pass;
    vkCmdPushConstants(commandBuffer, layout("neighbour_list"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
    vkCmdDispatch(commandBuffer, (constants.numParticles - 1)/1024 + 1, 1, 1);
}

void PointHashGrid::addBufferMemoryBarriers(VkCommandBuffer commandBuffer, const std::vector<VulkanBuffer *> &buffers) {
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

std::vector<PipelineMetaData> PointHashGrid::pipelineMetaData() {
    return {
            {
                    "point_hash_grid_builder",
                    data_shaders_sph_point_hash_grid_builder_comp,
                    { particleDescriptorSetLayout, &gridDescriptorSetLayout, &bucketSizeSetLayout},
                    {{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants)}}
            },
            {
                    "prefix_scan",
                    data_shaders_sph_scan_comp,
                    { &prefixScan.setLayout }
            },
            {
                    "add",
                    data_shaders_sph_add_comp,
                    { &prefixScan.setLayout },
                    { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(prefixScan.constants)} }
            },
            {
                "neighbour_list",
                data_shaders_sph_neighbour_list_comp,
                    { particleDescriptorSetLayout, &gridDescriptorSetLayout, &bucket.setLayout, &neighbourList.setLayout, &neighbourList.neighbourSizeSetLayout},
                    {{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants)}}
            }
    };
}

void PointHashGrid::setNumParticles(int numParticles) {
    if(bucketBuffer.buffer == VK_NULL_HANDLE || bucketBuffer.size/sizeof(int) != numParticles) {
        constants.numParticles = numParticles;
        bucketBuffer = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU,
                                            constants.numParticles * sizeof(int));

        auto writes = initializers::writeDescriptorSets<1>();
        VkDescriptorBufferInfo bucketInfo{bucketBuffer, 0, VK_WHOLE_SIZE };
        writes[0].dstSet = gridDescriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &bucketInfo;
        device->updateDescriptorSets(writes);
        initNeighbourList();
    }
}

std::vector<VulkanBuffer *> PointHashGrid::bucketBuffers() {
    return { &bucketSizeBuffer, &bucketSizeOffsetBuffer, &bucketBuffer};
}

std::vector<VulkanBuffer *> PointHashGrid::neighbourBuffers() {
    return { &neighbourList.neighbourListBuffer, &neighbourList.neighbourSizeBuffer, &neighbourList.neighbourOffsetsBuffer};
}

uint32_t PointHashGrid::numPointsInGrid() {
    return prefixScan.sumsBuffer.get<uint32_t>(0);
}

void PrefixScan::scan(VkCommandBuffer commandBuffer, VulkanBuffer &buffer, VkPipeline pipeline, VkPipelineLayout layout) {
    int size = buffer.size/sizeof(int);
    int numWorkGroups = std::abs(size - 1)/ITEMS_PER_WORKGROUP + 1;

    PointHashGrid::addBufferMemoryBarriers(commandBuffer, {&buffer});  // make sure grid build for pass 0 finished
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdDispatch(commandBuffer, numWorkGroups, 1, 1);

    PointHashGrid::addBufferMemoryBarriers(commandBuffer, {&buffer});
}
