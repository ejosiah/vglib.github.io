#include "fluid/VectorGrid.hpp"
#include "Barrier.hpp"

#include <array>
#include <format>

namespace eular {

    VectorGrid::VectorGrid(const Params& params)
        : ComputePipelines(params.device)
        , _device(params.device)
        , _descriptorPool(params.descriptorPool)
        , _globalConstantsDescriptorSet(params.globalConstantsDescriptorSet)
        , _boundaryDescriptorSet(params.boundaryDescriptorSet)
        , _globalConstantsSetLayout(params.globalConstantsSetLayout)
        , _boundaryDescriptorSetLayout(params.boundaryDescriptorSetLayout)
        , _imageType(VK_IMAGE_TYPE_2D)
        , _gridSize(params.gridSize, 1.0f)
        , _macCormackAdvection(params.macCormackAdvection)
        , _ensureBoundaryCondition(params.ensureBoundaryCondition) {
        _groupCount.xy = glm::uvec2(glm::ceil(params.gridSize / 32.0f));
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

    void VectorGrid::initFields() {
        const auto size = glm::ivec3(_gridSize);

        _vectorField.u.name = "vector_grid_u";
        _vectorField.v.name = "vector_grid_v";
        _divergenceField.name = "vector_grid_divergence";
        _forceField.name = "vector_grid_force";
        _macCormackData.name = "vector_grid_maccormack_intermediate";

        const auto addressMode = _ensureBoundaryCondition
            ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
            : VK_SAMPLER_ADDRESS_MODE_REPEAT;

        textures::createNoTransition(*device, _vectorField.u[0], VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _vectorField.u[1], VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _vectorField.v[0], VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _vectorField.v[1], VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, size, addressMode);

        textures::createNoTransition(*device, _divergenceField[0], VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _divergenceField[1], VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, size, addressMode);

        textures::createNoTransition(*device, _forceField[0], VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _forceField[1], VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, size, addressMode);

        textures::createNoTransition(*device, _macCormackData[0], VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _macCormackData[1], VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, size, addressMode);

        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vectorField.u.name, 0), _vectorField.u[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vectorField.u.name, 1), _vectorField.u[1].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vectorField.v.name, 0), _vectorField.v[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vectorField.v.name, 1), _vectorField.v[1].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _divergenceField.name, 0), _divergenceField[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _divergenceField.name, 1), _divergenceField[1].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _forceField.name, 0), _forceField[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _forceField.name, 1), _forceField[1].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _macCormackData.name, 0), _macCormackData[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _macCormackData.name, 1), _macCormackData[1].image.image);

        prepTextures();
    }

    void VectorGrid::createSamplers() {
        const auto addressMode = _ensureBoundaryCondition
            ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
            : VK_SAMPLER_ADDRESS_MODE_REPEAT;

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

    void VectorGrid::prepTextures() {
        device->firstActiveCommandPool().oneTimeCommand([&](auto commandBuffer) {
            const std::array<Texture*, 10> textures{
                &_vectorField.u[0],
                &_vectorField.u[1],
                &_vectorField.v[0],
                &_vectorField.v[1],
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
        static std::array<VkDescriptorSet, 8> sets;

        sets[0] = _globalConstantsDescriptorSet;
        sets[1] = vf.u.descriptorSet[in];
        sets[2] = vf.v.descriptorSet[in];
        sets[3] = _macCormackData.descriptorSet[in];
        sets[4] = _macCormackData.descriptorSet[out];
        sets[5] = field.descriptorSet[in];
        sets[6] = field.descriptorSet[out];
        sets[7] = _boundaryDescriptorSet;

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
