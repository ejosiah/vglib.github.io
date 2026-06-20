#include "fluid/CollocatedVectorGrid.hpp"
#include "Barrier.hpp"

#include <array>

namespace eular {

    std::vector<PipelineMetaData> CollocatedVectorGrid::pipelineMetaData() {
        return {
            {
                .name = "advect",
                .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\advect.comp.spv)",
                .layouts = {
                    _globalConstantsSetLayout, &Field::descriptorSetLayout, &Field::descriptorSetLayout,
                    &Field::descriptorSetLayout, &Field::descriptorSetLayout, &_samplerDescriptorSetLayout,
                    _colliderDescriptorSetLayout
                },
                .ranges = {{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(advectConstants)}}
            },
            {
                .name = "apply_force",
                .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\apply_force.comp.spv)",
                .layouts = {
                    _globalConstantsSetLayout, &Field::descriptorSetLayout, &Field::descriptorSetLayout,
                    &Field::descriptorSetLayout, &Field::descriptorSetLayout, &Field::descriptorSetLayout,
                    _colliderDescriptorSetLayout
                }
            },
            {
                .name = "divergence",
                .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\divergence.comp.spv)",
                .layouts = {
                    _globalConstantsSetLayout, &Field::descriptorSetLayout, &Field::descriptorSetLayout,
                    &Field::descriptorSetLayout, _colliderDescriptorSetLayout
                }
            },
            {
                .name = "divergence_free_field",
                .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\divergence_free_field.comp.spv)",
                .layouts = {
                    _globalConstantsSetLayout, &Field::descriptorSetLayout, &Field::descriptorSetLayout,
                    &Field::descriptorSetLayout, &Field::descriptorSetLayout, &Field::descriptorSetLayout,
                    _colliderDescriptorSetLayout
                }
            },
            {
                .name = "maccormack",
                .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\maccormack_advection.comp.spv)",
                .layouts = {
                    _globalConstantsSetLayout, &Field::descriptorSetLayout, &Field::descriptorSetLayout,
                    &Field::descriptorSetLayout, &Field::descriptorSetLayout, &Field::descriptorSetLayout,
                    &Field::descriptorSetLayout, _colliderDescriptorSetLayout
                },
                .ranges = {{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(advectConstants)}}
            },
        };
    }

    void CollocatedVectorGrid::advectVectorField(VkCommandBuffer commandBuffer) {
        auto& vf = vectorField();
        if(_macCormackAdvection) {
            advect(commandBuffer, vf.u, 1);
            advect(commandBuffer, vf.v, 2);
        } else {
            advect(commandBuffer, vf.u, 1, false);
            advect(commandBuffer, vf.v, 2, false);
            for(auto texture : {&vf.u[out], &vf.v[out]}) {
                Barriers::push(texture->image, DEFAULT_SUB_RANGE,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_WRITE_BIT,
                               VK_ACCESS_2_SHADER_READ_BIT,
                               texture->image.currentLayout,
                               texture->image.currentLayout);
            }
            Barriers::flush(commandBuffer);
        }
        vf.swap();
    }

    void CollocatedVectorGrid::advect(VkCommandBuffer commandBuffer, Field& field, uint32_t boundaryMode,
                                      bool addBarrier) {
        if(_macCormackAdvection) {
            macCormackAdvect(commandBuffer, field, boundaryMode);
        } else {
            advect(commandBuffer, field.descriptorSet[in], field.descriptorSet[out],
                   TimeDirection::Forward, boundaryMode, addBarrier ? &field[out] : nullptr);
        }
    }

    void CollocatedVectorGrid::advect(VkCommandBuffer commandBuffer, VkDescriptorSet inDescriptor,
                                      VkDescriptorSet outDescriptor, TimeDirection timeDirection,
                                      uint32_t boundaryMode, Texture* writeTexture) {
        auto& vf = vectorField();
        static std::array<VkDescriptorSet, 7> sets;

        sets[0] = _globalConstantsDescriptorSet;
        sets[1] = vf.u.descriptorSet[in];
        sets[2] = vf.v.descriptorSet[in];
        sets[3] = inDescriptor;
        sets[4] = outDescriptor;
        sets[5] = _linearSamplerDescriptorSet;
        sets[6] = _colliderDescriptorSet;

        advectConstants.time_sign = timeDirection == TimeDirection::Forward ? 1.0f : -1.0f;
        advectConstants.boundary_mode = boundaryMode;
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("advect"));
        vkCmdPushConstants(commandBuffer, layout("advect"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(advectConstants), &advectConstants);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("advect"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);

        if(writeTexture) {
            Barriers::pushAndFlush(commandBuffer, writeTexture->image, DEFAULT_SUB_RANGE,
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_WRITE_BIT,
                                   VK_ACCESS_2_SHADER_READ_BIT,
                                   writeTexture->image.currentLayout,
                                   writeTexture->image.currentLayout);
        }
    }

    void CollocatedVectorGrid::computeDivergence(VkCommandBuffer commandBuffer) {
        auto& vf = vectorField();
        static std::array<VkDescriptorSet, 5> sets;

        sets[0] = _globalConstantsDescriptorSet;
        sets[1] = vf.u.descriptorSet[in];
        sets[2] = vf.v.descriptorSet[in];
        sets[3] = _divergenceField.descriptorSet[in];
        sets[4] = _colliderDescriptorSet;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("divergence"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("divergence"),
                                0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
        Barriers::pushAndFlush(commandBuffer, _divergenceField[in].image, DEFAULT_SUB_RANGE,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_WRITE_BIT,
                               VK_ACCESS_2_SHADER_READ_BIT,
                               _divergenceField[in].image.currentLayout,
                               _divergenceField[in].image.currentLayout);
    }

    void CollocatedVectorGrid::computeDivergenceFreeField(VkCommandBuffer commandBuffer, PressureField& pressureField) {
        auto& vf = vectorField();
        static std::array<VkDescriptorSet, 7> sets;

        sets[0] = _globalConstantsDescriptorSet;
        sets[1] = vf.u.descriptorSet[in];
        sets[2] = vf.v.descriptorSet[in];
        sets[3] = pressureField.descriptorSet[in];
        sets[4] = vf.u.descriptorSet[out];
        sets[5] = vf.v.descriptorSet[out];
        sets[6] = _colliderDescriptorSet;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("divergence_free_field"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("divergence_free_field"),
                                0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);

        for(auto texture : {&vf.u[out], &vf.v[out]}) {
            Barriers::push(texture->image, DEFAULT_SUB_RANGE,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT,
                           VK_ACCESS_2_SHADER_READ_BIT,
                           texture->image.currentLayout,
                           texture->image.currentLayout);
        }
        Barriers::flush(commandBuffer);
        vf.swap();
    }

    void CollocatedVectorGrid::addForcesToVectorField(VkCommandBuffer commandBuffer) {
        auto& vf = vectorField();
        static std::array<VkDescriptorSet, 7> sets;

        sets[0] = _globalConstantsDescriptorSet;
        sets[1] = vf.u.descriptorSet[in];
        sets[2] = vf.v.descriptorSet[in];
        sets[3] = _forceField.descriptorSet[in];
        sets[4] = vf.u.descriptorSet[out];
        sets[5] = vf.v.descriptorSet[out];
        sets[6] = _colliderDescriptorSet;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("apply_force"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("apply_force"),
                                0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
        for(auto texture : {&vf.u[out], &vf.v[out]}) {
            Barriers::push(texture->image, DEFAULT_SUB_RANGE,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT,
                           VK_ACCESS_2_SHADER_READ_BIT,
                           texture->image.currentLayout,
                           texture->image.currentLayout);
        }
        Barriers::flush(commandBuffer);

        vf.swap();
    }

    void CollocatedVectorGrid::generate(VectorFieldFunc2D generator) {
        if(!generator) return;

        const auto rows = static_cast<size_t>(_gridSize.y);
        const auto columns = static_cast<size_t>(_gridSize.x);
        const auto size = static_cast<size_t>(_gridSize.x * _gridSize.y);

        std::vector<float> uBuffer;
        std::vector<float> vBuffer;

        uBuffer.reserve(size);
        vBuffer.reserve(size);

        for(auto row = 0u; row < rows; ++row) {
            for(auto column = 0u; column < columns; ++column) {
                const auto x = 2.0f * static_cast<float>(column) / _gridSize.x - 1.0f;
                const auto y = 2.0f * static_cast<float>(row) / _gridSize.y - 1.0f;
                const auto value = generator(x, y);
                uBuffer.push_back(value.x);
                vBuffer.push_back(value.y);
            }
        }

        const auto byteSize = size * sizeof(float);
        auto stagingBufferU = device->createStagingBuffer(byteSize);
        auto stagingBufferV = device->createStagingBuffer(byteSize);

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

            const auto gs = glm::uvec2(_gridSize);
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
}
