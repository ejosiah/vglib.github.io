#include "fluid/VectorGrid.hpp"
#include "Barrier.hpp"

#include <array>
#include <cassert>
#include <format>

namespace eular {

    VectorGrid::VectorGrid(const Params& params)
        : ComputePipelines(params.device)
        , _device(params.device)
        , _descriptorPool(params.descriptorPool)
        , _globalConstantsDescriptorSet(params.globalConstantsDescriptorSet)
        , _colliderDescriptorSet(params.colliderDescriptorSet)
        , _globalConstantsSetLayout(params.globalConstantsSetLayout)
        , _colliderDescriptorSetLayout(params.colliderDescriptorSetLayout)
        , _imageType(params.imageType)
        , _gridSize(params.gridSize)
        , _dimension(glm::clamp(params.dimension, 2u, 3u))
        , _macCormackAdvection(params.macCormackAdvection)
        , _wrappingEnabled(params.wrappingEnabled) {
        _groupCount = glm::uvec3(glm::ceil(params.gridSize / 8.0f));
        _groupCount.z = _dimension == 3u ? static_cast<uint32_t>(params.gridSize.z) : 1u;
    }

    VectorGrid::~VectorGrid() {
        releaseDescriptorSets();
    }

    void VectorGrid::init() {
        createSamplers();
        initFields();
        createDescriptorSetLayouts();
        updateDescriptorSets();
        createPipelines();
    }

    VectorField& VectorGrid::vectorField() {
        return _vectorField;
    }

    const VectorField& VectorGrid::vectorField() const {
        return _vectorField;
    }

    DivergenceField& VectorGrid::divergenceField() {
        return _divergenceField;
    }

    const DivergenceField& VectorGrid::divergenceField() const {
        return _divergenceField;
    }

    ForceField& VectorGrid::forceField() {
        return _forceField;
    }

    const ForceField& VectorGrid::forceField() const {
        return _forceField;
    }

    VulkanDescriptorSetLayout VectorGrid::fieldDescriptorSetLayout() const {
        return Field::descriptorSetLayout;
    }

    void VectorGrid::fill(std::span<glm::vec2> vectorField) {
        assert(vectorField.size() == static_cast<size_t>(_gridSize.x * _gridSize.y));
        const auto byteSize = vectorField.size() * sizeof(float);
        auto stagingBufferU = device->createStagingBuffer(byteSize);
        auto stagingBufferV = device->createStagingBuffer(byteSize);

        auto uBuffer = map_range(vectorField, [](const auto v){ return v.x; });
        auto vBuffer = map_range(vectorField, [](const auto v){ return v.y; });
        stagingBufferU.copy(uBuffer);
        stagingBufferV.copy(vBuffer);

        device->firstActiveCommandPool().oneTimeCommand([&](auto commandBuffer) {
            for(auto texture : {&_vectorField.u[0], &_vectorField.v[0]}) {
                Barriers::push(texture->image, DEFAULT_SUB_RANGE,
                               VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                               VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                               VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               texture->image.currentLayout,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                texture->image.currentLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            }
            Barriers::flush(commandBuffer);

            const auto gs = glm::uvec3(_gridSize);
            VkBufferImageCopy region{};
            region.bufferOffset = 0;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;
            region.imageOffset = {0, 0, 0};
            region.imageExtent = {gs.x, gs.y, 1};

            vkCmdCopyBufferToImage(commandBuffer, stagingBufferU, _vectorField.u[0].image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            vkCmdCopyBufferToImage(commandBuffer, stagingBufferV, _vectorField.v[0].image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            for(auto texture : {&_vectorField.u[0], &_vectorField.v[0]}) {
                Barriers::push(texture->image, DEFAULT_SUB_RANGE,
                               VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               VK_ACCESS_2_SHADER_READ_BIT,
                               texture->image.currentLayout,
                               VK_IMAGE_LAYOUT_GENERAL);
                texture->image.currentLayout = VK_IMAGE_LAYOUT_GENERAL;
            }
            Barriers::flush(commandBuffer);
        });
    }

    void VectorGrid::fill(std::span<glm::vec3> vectorField) {
        assert(vectorField.size() == static_cast<size_t>(_gridSize.x * _gridSize.y * _gridSize.z));
        const auto byteSize = vectorField.size() * sizeof(float);
        auto stagingBufferU = device->createStagingBuffer(byteSize);
        auto stagingBufferV = device->createStagingBuffer(byteSize);
        auto stagingBufferW = device->createStagingBuffer(byteSize);

        auto uBuffer = map_range(vectorField, [](const auto v){ return v.x; });
        auto vBuffer = map_range(vectorField, [](const auto v){ return v.y; });
        auto wBuffer = map_range(vectorField, [](const auto v){ return v.z; });
        stagingBufferU.copy(uBuffer);
        stagingBufferV.copy(vBuffer);
        stagingBufferW.copy(wBuffer);

        device->firstActiveCommandPool().oneTimeCommand([&](auto commandBuffer) {
            for(auto texture : {&_vectorField.u[0], &_vectorField.v[0], &_vectorField.w[0]}) {
                Barriers::push(texture->image, DEFAULT_SUB_RANGE,
                               VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                               VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                               VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               texture->image.currentLayout,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                texture->image.currentLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            }
            Barriers::flush(commandBuffer);

            const auto gs = glm::uvec3(_gridSize);
            VkBufferImageCopy region{};
            region.bufferOffset = 0;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;
            region.imageOffset = {0, 0, 0};
            region.imageExtent = {gs.x, gs.y, gs.z};

            vkCmdCopyBufferToImage(commandBuffer, stagingBufferU, _vectorField.u[0].image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            vkCmdCopyBufferToImage(commandBuffer, stagingBufferV, _vectorField.v[0].image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            vkCmdCopyBufferToImage(commandBuffer, stagingBufferW, _vectorField.w[0].image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            for(auto texture : {&_vectorField.u[0], &_vectorField.v[0], &_vectorField.w[0]}) {
                Barriers::push(texture->image, DEFAULT_SUB_RANGE,
                               VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               VK_ACCESS_2_SHADER_READ_BIT,
                               texture->image.currentLayout,
                               VK_IMAGE_LAYOUT_GENERAL);
                texture->image.currentLayout = VK_IMAGE_LAYOUT_GENERAL;
            }
            Barriers::flush(commandBuffer);
        });
    }

    void VectorGrid::fill(glm::vec2 value) {
        std::vector<glm::vec2> data(to<size_t>(_gridSize.x * _gridSize.y), value);
        fill(data);
    }

    void VectorGrid::fill(glm::vec3 value) {
        std::vector<glm::vec3> data(to<size_t>(_gridSize.x * _gridSize.y * _gridSize.z), value);
        fill(data);
    }

    void VectorGrid::initFields() {
        const auto size = glm::ivec3(_gridSize);

        _vectorField.u.name = "vector_grid_u";
        _vectorField.v.name = "vector_grid_v";
        _vectorField.w.name = "vector_grid_w";
        _divergenceField.name = "vector_grid_divergence";
        _forceField.name = "vector_grid_force";
        _macCormackData.name = "vector_grid_maccormack_intermediate";

        const auto addressMode = _wrappingEnabled ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

        textures::createNoTransition(*device, _vectorField.u[0], _imageType, VK_FORMAT_R32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _vectorField.u[1], _imageType, VK_FORMAT_R32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _vectorField.v[0], _imageType, VK_FORMAT_R32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _vectorField.v[1], _imageType, VK_FORMAT_R32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _vectorField.w[0], _imageType, VK_FORMAT_R32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _vectorField.w[1], _imageType, VK_FORMAT_R32_SFLOAT, size, addressMode);

        textures::createNoTransition(*device, _divergenceField[0], _imageType, VK_FORMAT_R32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _divergenceField[1], _imageType, VK_FORMAT_R32_SFLOAT, size, addressMode);

        textures::createNoTransition(*device, _forceField[0], _imageType, VK_FORMAT_R32G32B32A32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _forceField[1], _imageType, VK_FORMAT_R32G32B32A32_SFLOAT, size, addressMode);

        textures::createNoTransition(*device, _macCormackData[0], _imageType, VK_FORMAT_R32G32B32A32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _macCormackData[1], _imageType, VK_FORMAT_R32G32B32A32_SFLOAT, size, addressMode);

        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vectorField.u.name, 0), _vectorField.u[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vectorField.u.name, 1), _vectorField.u[1].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vectorField.v.name, 0), _vectorField.v[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vectorField.v.name, 1), _vectorField.v[1].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vectorField.w.name, 0), _vectorField.w[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vectorField.w.name, 1), _vectorField.w[1].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _divergenceField.name, 0), _divergenceField[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _divergenceField.name, 1), _divergenceField[1].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _forceField.name, 0), _forceField[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _forceField.name, 1), _forceField[1].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _macCormackData.name, 0), _macCormackData[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _macCormackData.name, 1), _macCormackData[1].image.image);

        prepTextures();
    }

    void VectorGrid::createSamplers() {
        const auto addressMode = _wrappingEnabled ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;


        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = addressMode;
        samplerInfo.addressModeV = addressMode;
        samplerInfo.addressModeW = addressMode;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

        _linearSampler = device->createSampler(samplerInfo);
    }

    void VectorGrid::createDescriptorSetLayouts() {
        Field::descriptorSetLayout =
            device->descriptorSetLayoutBuilder()
                .name("vector_grid_field_set_layout")
                .binding(0)
                    .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
                    .immutableSamplers(_linearSampler)
                .binding(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(2)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .createLayout();

        _samplerDescriptorSetLayout =
            device->descriptorSetLayoutBuilder()
                .name("vector_grid_sampler_set_layout")
                .binding(0)
                    .descriptorType(VK_DESCRIPTOR_TYPE_SAMPLER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .createLayout();
    }

    void VectorGrid::updateDescriptorSets() {
        auto sets = _descriptorPool->allocate({_samplerDescriptorSetLayout});
        _linearSamplerDescriptorSet = sets[0];

        auto writes = initializers::writeDescriptorSets<40>();
        auto writeOffset = 0u;

        writes[writeOffset].dstSet = _linearSamplerDescriptorSet;
        writes[writeOffset].dstBinding = 0;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo{_linearSampler.handle, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
        ++writeOffset;

        writeOffset = createDescriptorSet(writes, writeOffset, _vectorField.u);
        writeOffset = createDescriptorSet(writes, writeOffset, _vectorField.v);
        writeOffset = createDescriptorSet(writes, writeOffset, _vectorField.w);
        writeOffset = createDescriptorSet(writes, writeOffset, _divergenceField);
        writeOffset = createDescriptorSet(writes, writeOffset, _forceField);
        writeOffset = createDescriptorSet(writes, writeOffset, _macCormackData);

        writes.resize(writeOffset);
        device->updateDescriptorSets(writes);

        for(auto& write : writes) {
            if(write.pImageInfo) delete write.pImageInfo;
            if(write.pBufferInfo) delete write.pBufferInfo;
        }
    }

    uint32_t VectorGrid::createDescriptorSet(std::vector<VkWriteDescriptorSet>& writes, uint32_t writeOffset, Field& field) {
        auto sets = _descriptorPool->allocate({Field::descriptorSetLayout, Field::descriptorSetLayout});

        field.descriptorSet[0] = sets[0];
        field.descriptorSet[1] = sets[1];

        writes[writeOffset].dstSet = field.descriptorSet[0];
        writes[writeOffset].dstBinding = 0;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo{VK_NULL_HANDLE, field[0].imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
        ++writeOffset;

        writes[writeOffset].dstSet = field.descriptorSet[0];
        writes[writeOffset].dstBinding = 1;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo{VK_NULL_HANDLE, field[0].imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
        ++writeOffset;

        writes[writeOffset].dstSet = field.descriptorSet[0];
        writes[writeOffset].dstBinding = 2;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo{VK_NULL_HANDLE, field[0].imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
        ++writeOffset;

        writes[writeOffset].dstSet = field.descriptorSet[1];
        writes[writeOffset].dstBinding = 0;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo{VK_NULL_HANDLE, field[1].imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
        ++writeOffset;

        writes[writeOffset].dstSet = field.descriptorSet[1];
        writes[writeOffset].dstBinding = 1;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo{VK_NULL_HANDLE, field[1].imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
        ++writeOffset;

        writes[writeOffset].dstSet = field.descriptorSet[1];
        writes[writeOffset].dstBinding = 2;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo{VK_NULL_HANDLE, field[1].imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
        ++writeOffset;

        return writeOffset;
    }

    void VectorGrid::releaseDescriptorSets() {
        releaseDescriptorSet(_linearSamplerDescriptorSet);
        releaseFieldDescriptorSets(_vectorField.u);
        releaseFieldDescriptorSets(_vectorField.v);
        releaseFieldDescriptorSets(_vectorField.w);
        releaseFieldDescriptorSets(_divergenceField);
        releaseFieldDescriptorSets(_forceField);
        releaseFieldDescriptorSets(_macCormackData);
    }

    void VectorGrid::releaseDescriptorSet(VkDescriptorSet& descriptorSet) {
        if(!_descriptorPool || descriptorSet == VK_NULL_HANDLE) {
            return;
        }

        _descriptorPool->free(descriptorSet);
        descriptorSet = VK_NULL_HANDLE;
    }

    void VectorGrid::releaseFieldDescriptorSets(Field& field) {
        releaseDescriptorSet(field.descriptorSet[0]);
        releaseDescriptorSet(field.descriptorSet[1]);
    }

    void VectorGrid::prepTextures() {
        device->firstActiveCommandPool().oneTimeCommand([&](auto commandBuffer) {
            std::vector<Texture*> textures{
                &_vectorField.u[0],
                &_vectorField.u[1],
                &_vectorField.v[0],
                &_vectorField.v[1],
                &_vectorField.w[0],
                &_vectorField.w[1],
                &_divergenceField[0],
                &_divergenceField[1],
                &_forceField[0],
                &_forceField[1],
                &_macCormackData[0],
                &_macCormackData[1],
            };

            for(auto texture : textures) {
                Barriers::push(texture->image, DEFAULT_SUB_RANGE,
                               VK_PIPELINE_STAGE_2_NONE,
                               VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                               VK_ACCESS_2_NONE,
                               VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               VK_IMAGE_LAYOUT_UNDEFINED,
                               VK_IMAGE_LAYOUT_GENERAL);
                texture->image.currentLayout = VK_IMAGE_LAYOUT_GENERAL;
            }
            Barriers::flush(commandBuffer);

            VkClearColorValue zero{{0.0f, 0.0f, 0.0f, 0.0f}};
            for(auto texture : textures) {
                vkCmdClearColorImage(commandBuffer, texture->image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &DEFAULT_SUB_RANGE);
            }

            Barriers::pushAndFlush(commandBuffer,
                                   VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                   VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
        });
    }

    void VectorGrid::macCormackAdvect(VkCommandBuffer commandBuffer, Field& field, uint32_t boundaryMode) {
        advect(commandBuffer, field.descriptorSet[in], _macCormackData.descriptorSet[in],
               TimeDirection::Forward, boundaryMode, &_macCormackData[in]);
        advect(commandBuffer, _macCormackData.descriptorSet[in], _macCormackData.descriptorSet[out],
               TimeDirection::Backword, boundaryMode, &_macCormackData[out]);

        auto& vf = _vectorField;
        static std::array<VkDescriptorSet, 9> sets;

        sets[0] = _globalConstantsDescriptorSet;
        sets[1] = vf.u.descriptorSet[in];
        sets[2] = vf.v.descriptorSet[in];
        sets[3] = vf.w.descriptorSet[in];
        sets[4] = _macCormackData.descriptorSet[in];
        sets[5] = _macCormackData.descriptorSet[out];
        sets[6] = field.descriptorSet[in];
        sets[7] = field.descriptorSet[out];
        sets[8] = _colliderDescriptorSet;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("maccormack"));
        advectConstants.time_sign = 1.0f;
        advectConstants.boundary_mode = boundaryMode;
        vkCmdPushConstants(commandBuffer, layout("maccormack"), VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(advectConstants), &advectConstants);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("maccormack"),
                                0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
        Barriers::pushAndFlush(commandBuffer, field[out].image, DEFAULT_SUB_RANGE,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_WRITE_BIT,
                               VK_ACCESS_2_SHADER_READ_BIT,
                               field[out].image.currentLayout,
                               field[out].image.currentLayout);
    }
}
