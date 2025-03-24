#include "fluid/FluidSolver2.hpp"

namespace eular {
    
    FluidSolver::FluidSolver(VulkanDevice *device, VulkanDescriptorPool* descriptorPool, glm::vec2 gridSize)
        : ComputePipelines(device)
        , _descriptorPool(descriptorPool)
        , _gridSize(gridSize, 1)
        , _delta(1.f/gridSize, 0)
        , _imageType(VK_IMAGE_TYPE_2D){
        _groupCount.xy = glm::uvec2(glm::ceil(gridSize/32.f));
    }

    
    void FluidSolver::init() {
        createSamplers();
        initFields();
        initGlobalConstants();
        createDescriptorSetLayouts();
        updateDescriptorSets();
        createPipelines();
    }

    void FluidSolver::createSamplers() {
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_NEAREST;
        samplerInfo.minFilter = VK_FILTER_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST ;

        _valueSampler = device->createSampler(samplerInfo);

        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        _linearSampler = device->createSampler(samplerInfo);

        _vectorField.u[0].sampler = _valueSampler;
        _vectorField.u[1].sampler = _valueSampler;

        _vectorField.v[0].sampler = _valueSampler;
        _vectorField.v[1].sampler = _valueSampler;

        _forceField[0].sampler = _valueSampler;
        _forceField[1].sampler = _valueSampler;

        _vorticityField[0].sampler = _valueSampler;
        _vorticityField[1].sampler = _valueSampler;

        _divergenceField[0].sampler = _valueSampler;

        _pressureField[0].sampler = _valueSampler;
        _pressureField[1].sampler = _valueSampler;
    }

    void FluidSolver::initGlobalConstants() {
        GlobalData data{};
        data.grid_size = glm::ivec3(_gridSize);
        data.dx = {_delta.x, 0};
        data.dy = {0, _delta.y};
        data.dt = _timeStep;
        globalConstants.gpu = device->createCpuVisibleBuffer(&data, sizeof(GlobalData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        globalConstants.cpu =  reinterpret_cast<GlobalData*>(globalConstants.gpu.map());
    }
    
    void FluidSolver::initFields() {
        auto size = glm::ivec3(_gridSize);

        _vectorField.u.name = "vector_field_u";
        _vectorField.v.name = "vector_field_v";
        _forceField.name = "force_field";
        _vorticityField.name = "vorticity_field";
        _divergenceField.name = "divergence_field";
        _pressureField.name = "pressure_field";

        textures::createNoTransition(*device, _vectorField.u[0], VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, size, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        textures::createNoTransition(*device, _vectorField.u[1], VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, size, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        textures::createNoTransition(*device, _vectorField.v[0], VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, size, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        textures::createNoTransition(*device, _vectorField.v[1], VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, size, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

        textures::createNoTransition(*device, _forceField[0], VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, size, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        textures::createNoTransition(*device, _forceField[1], VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, size, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);


        textures::createNoTransition(*device, _vorticityField[0], VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, size, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        textures::createNoTransition(*device, _vorticityField[1], VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, size, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

        textures::createNoTransition(*device, _divergenceField[0], VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, size, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        textures::createNoTransition(*device, _divergenceField[1], VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, size, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

        textures::createNoTransition(*device, _pressureField[0], VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, size, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        textures::createNoTransition(*device, _pressureField[1], VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, size, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vectorField.u.name, 0), _vectorField.u[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vectorField.u.name, 1), _vectorField.u[1].image.image);

        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vectorField.v.name, 0), _vectorField.v[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vectorField.v.name, 1), _vectorField.v[1].image.image);

        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _forceField.name, 0), _forceField[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _forceField.name, 1), _forceField[1].image.image);

        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vorticityField.name, 0), _vorticityField[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vorticityField.name, 1), _vorticityField[1].image.image);

        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _divergenceField.name, 0), _divergenceField[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _divergenceField.name, 1), _divergenceField[1].image.image);

        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _pressureField.name, 0), _pressureField[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _pressureField.name, 1), _pressureField[1].image.image);

        prepTextures();
    }

    void FluidSolver::createDescriptorSetLayouts() {
        uniformsSetLayout =
            device->descriptorSetLayoutBuilder()
                .name("fluid_solver_global_uniforms")
                .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();

        _fieldDescriptorSetLayout =
            device->descriptorSetLayoutBuilder()
                .name("fluid_solver_field_set_layout")
                .binding(0)
                    .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
                    .immutableSamplers(_valueSampler)
                .binding(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(2)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .createLayout();

        _imageDescriptorSetLayout =
            device->descriptorSetLayoutBuilder()
                .name("fluid_solver_image_set_layout")
                .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();

        _textureDescriptorSetLayout =
            device->descriptorSetLayoutBuilder()
                .name("fluid_solver_texture_set_layout")
                .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();

        _samplerDescriptorSetLayout =
            device->descriptorSetLayoutBuilder()
                .name("fluid_solver_sampler_set_layout")
                .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_SAMPLER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();

        _debugDescriptorSetLayout =
            device->descriptorSetLayoutBuilder()
                .name("fluid_solver_debug_set_layout")
                .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                .descriptorCount(20)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();
    }

    void FluidSolver::updateDescriptorSets() {
        auto sets = _descriptorPool->allocate({ uniformsSetLayout, _samplerDescriptorSetLayout, _samplerDescriptorSetLayout });
        uniformDescriptorSet = sets[0];
        _valueSamplerDescriptorSet = sets[1];
        _linearSamplerDescriptorSet = sets[2];
        auto writes = initializers::writeDescriptorSets<33>();

        auto writeOffset = 0u;

        writes[writeOffset].dstSet = uniformDescriptorSet;
        writes[writeOffset].dstBinding = 0;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[writeOffset].descriptorCount = 1;
        auto info = VkDescriptorBufferInfo{ globalConstants.gpu, 0, VK_WHOLE_SIZE };
        writes[writeOffset].pBufferInfo = &info;
        ++writeOffset;

        writes[writeOffset].dstSet = _valueSamplerDescriptorSet;
        writes[writeOffset].dstBinding = 0;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo { _valueSampler.handle, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
        ++writeOffset;

        writes[writeOffset].dstSet = _linearSamplerDescriptorSet;
        writes[writeOffset].dstBinding = 0;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo { _linearSampler.handle, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
        ++writeOffset;

        writeOffset = createDescriptorSet(writes, writeOffset, _vectorField.u);
        writeOffset = createDescriptorSet(writes, writeOffset, _vectorField.v);
        writeOffset = createDescriptorSet(writes, writeOffset, _divergenceField);
        writeOffset = createDescriptorSet(writes, writeOffset, _pressureField);
        writeOffset = createDescriptorSet(writes, writeOffset, _forceField);

        device->updateDescriptorSets(writes);

        for(auto& write : writes) {
            delete write.pImageInfo;

        }
    }

    uint32_t FluidSolver::createDescriptorSet(std::vector<VkWriteDescriptorSet>& writes, uint32_t writeOffset, Field& field) {
        auto sets = _descriptorPool->allocate( { _fieldDescriptorSetLayout, _fieldDescriptorSetLayout});

        field.descriptorSet[0] = sets[0];
        field.descriptorSet[1] = sets[1];

        writes[writeOffset].dstSet = field.descriptorSet[0];
        writes[writeOffset].dstBinding = 0;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo {VK_NULL_HANDLE, field[0].imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
        ++writeOffset;

        writes[writeOffset].dstSet = field.descriptorSet[0];
        writes[writeOffset].dstBinding = 1;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo {VK_NULL_HANDLE, field[0].imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
        ++writeOffset;

        writes[writeOffset].dstSet = field.descriptorSet[0];
        writes[writeOffset].dstBinding = 2;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo {VK_NULL_HANDLE, field[0].imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
        ++writeOffset;

        writes[writeOffset].dstSet = field.descriptorSet[1];
        writes[writeOffset].dstBinding = 0;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo {VK_NULL_HANDLE, field[1].imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
        ++writeOffset;

        writes[writeOffset].dstSet = field.descriptorSet[1];
        writes[writeOffset].dstBinding = 1;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo {VK_NULL_HANDLE, field[1].imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
        ++writeOffset;

        writes[writeOffset].dstSet = field.descriptorSet[1];
        writes[writeOffset].dstBinding = 2;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo {VK_NULL_HANDLE, field[1].imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
        ++writeOffset;

        return writeOffset;
    }

    
    void FluidSolver::prepTextures() {
        device->graphicsCommandPool().oneTimeCommand([&](auto commandBuffer) {
            std::vector<VkImageMemoryBarrier2> barriers;

            VkImageMemoryBarrier2 barrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                    .srcStageMask = VK_PIPELINE_STAGE_NONE,
                    .srcAccessMask = VK_ACCESS_NONE,
                    .dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .subresourceRange = {
                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .baseMipLevel =  0,
                            .levelCount = 1,
                            .baseArrayLayer = 0,
                            .layerCount = 1,
                    }
            };

            barrier.image = _forceField[0].image;
            barriers.push_back(barrier);

            barrier.image = _vorticityField[0].image;
            barriers.push_back(barrier);

            barrier.image = _divergenceField[0].image;
            barriers.push_back(barrier);

            barrier.image = _pressureField[0].image;
            barriers.push_back(barrier);


            barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

            barrier.image = _vectorField.u[1].image;
            barriers.push_back(barrier);

            barrier.image = _vectorField.v[1].image;
            barriers.push_back(barrier);

            barrier.image = _forceField[1].image;

            barrier.image = _vorticityField[1].image;
            barriers.push_back(barrier);

            barrier.image = _divergenceField[1].image;
            barriers.push_back(barrier);

            barrier.image = _pressureField[1].image;
            barriers.push_back(barrier);

            VkDependencyInfo dInfo {
                    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                    .imageMemoryBarrierCount = COUNT(barriers),
                    .pImageMemoryBarriers = barriers.data()
            };

            vkCmdPipelineBarrier2(commandBuffer, &dInfo);
        });
    }

    void FluidSolver::set(VectorFieldSource2D vectorField) {
        auto size = size_t(_gridSize.x * _gridSize.y);

        std::vector<float> uBuffer;
        std::vector<float> vBuffer;

        uBuffer.reserve(size);
        vBuffer.reserve(size);

        for(const auto& u : vectorField) {
            uBuffer.push_back(u.x);
            vBuffer.push_back(u.y);
        }

        auto byteSize = size * sizeof(float);
        auto stagingBuffer_u = device->createStagingBuffer(byteSize);
        auto stagingBuffer_v = device->createStagingBuffer(byteSize);

        stagingBuffer_u.copy(uBuffer);
        stagingBuffer_v.copy(vBuffer);


        VkImageMemoryBarrier2 barrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_NONE,
                .srcAccessMask = VK_ACCESS_NONE,
                .dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel =  0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                }
        };
        std::vector<VkImageMemoryBarrier2> barriers;
        barrier.image = _vectorField.u[0].image;
        barriers.push_back(barrier);

        barrier.image = _vectorField.v[0].image;
        barriers.push_back(barrier);

        VkDependencyInfo dInfo {
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .imageMemoryBarrierCount = COUNT(barriers),
                .pImageMemoryBarriers = barriers.data()
        };


        device->graphicsCommandPool().oneTimeCommand([&](auto commandBuffer){
            vkCmdPipelineBarrier2(commandBuffer, &dInfo);

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

            vkCmdCopyBufferToImage(commandBuffer, stagingBuffer_u, _vectorField.u[0].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            vkCmdCopyBufferToImage(commandBuffer, stagingBuffer_v, _vectorField.v[0].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            barriers[0].srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
            barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barriers[0].dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;

            barriers[1].srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
            barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barriers[1].dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            barriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barriers[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;

            vkCmdPipelineBarrier2(commandBuffer, &dInfo);

        });
    }

    
    void FluidSolver::set(VectorFieldSource3D vectorField) {
        auto size = size_t(_gridSize.x * _gridSize.y * _gridSize.z);

        std::vector<float> uBuffer;
        std::vector<float> vBuffer;
        std::vector<float> wBuffer;

        uBuffer.reserve(size);
        vBuffer.reserve(size);
        wBuffer.reserve(size);

        for(const auto& u : vectorField) {
            uBuffer.push_back(u.x);
            vBuffer.push_back(u.y);
            wBuffer.push_back(u.z);
        }

        auto byteSize = size * sizeof(float);
        auto stagingBuffer_u = device->createStagingBuffer(byteSize);
        auto stagingBuffer_v = device->createStagingBuffer(byteSize);
        auto stagingBuffer_w = device->createStagingBuffer(byteSize);

        stagingBuffer_u.copy(uBuffer);
        stagingBuffer_v.copy(vBuffer);
        stagingBuffer_w.copy(vBuffer);

        VkImageMemoryBarrier2 barrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_NONE,
                .srcAccessMask = VK_ACCESS_NONE,
                .dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel =  0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                }
        };
        std::vector<VkImageMemoryBarrier2> barriers;
        barrier.image = _vectorField.u[0].image;
        barriers.push_back(barrier);

        barrier.image = _vectorField.v[0].image;
        barriers.push_back(barrier);

        barrier.image = _vectorField.w[0].image;
        barriers.push_back(barrier);

        VkDependencyInfo dInfo {
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .imageMemoryBarrierCount = COUNT(barriers),
                .pImageMemoryBarriers = barriers.data()
        };


        device->graphicsCommandPool().oneTimeCommand([&](auto commandBuffer){
            vkCmdPipelineBarrier2(commandBuffer, &dInfo);

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

            vkCmdCopyBufferToImage(commandBuffer, stagingBuffer_u, _vectorField.u[0].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            vkCmdCopyBufferToImage(commandBuffer, stagingBuffer_v, _vectorField.v[0].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            vkCmdCopyBufferToImage(commandBuffer, stagingBuffer_w, _vectorField.w[0].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            barriers[0].srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
            barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barriers[0].dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;

            barriers[1].srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
            barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barriers[1].dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            barriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barriers[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;

            barriers[2].srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
            barriers[2].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barriers[2].dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            barriers[2].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barriers[2].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barriers[2].newLayout = VK_IMAGE_LAYOUT_GENERAL;

            vkCmdPipelineBarrier2(commandBuffer, &dInfo);

        });
    }

    
    std::vector<PipelineMetaData> FluidSolver::pipelineMetaData() {
        return {
                {
                    .name = "advect",
                    .shadePath = R"(C:\Users\Josiah Ebhomenye\CLionProjects\vglib_examples\dependencies\vglib.github.io\data\shaders\fluid_2d\advect.comp.spv)",
                    .layouts =  {
                            &uniformsSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                            &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout, &_samplerDescriptorSetLayout
                    }

                },
                {
                    .name = "apply_force",
                    .shadePath = R"(C:\Users\Josiah Ebhomenye\CLionProjects\vglib_examples\dependencies\vglib.github.io\data\shaders\fluid_2d\apply_force.comp.spv)",
                    .layouts =  {
                            &uniformsSetLayout,  &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                            &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout
                    },
                    .ranges = { {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(forceConstants) } }

                },
                {
                        .name = "add_sources",
                        .shadePath = R"(C:\Users\Josiah Ebhomenye\CLionProjects\vglib_examples\dependencies\vglib.github.io\data\shaders\fluid_2d\add_sources.comp.spv)",
                        .layouts =  {
                                &uniformsSetLayout,  &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                                &_fieldDescriptorSetLayout
                        },
                        .ranges = { {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(projectConstants) } }
                },
                {
                    .name = "jacobi",
                    .shadePath = R"(C:\Users\Josiah Ebhomenye\CLionProjects\vglib_examples\dependencies\vglib.github.io\data\shaders\fluid_2d\jacobi.comp.spv)",
                    .layouts =  {
                            &uniformsSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout
                    },
                    .ranges = { {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(linearSolverConstants) } }
                },
                {
                    .name = "rbgs",
                    .shadePath = R"(C:\Users\Josiah Ebhomenye\CLionProjects\vglib_examples\dependencies\vglib.github.io\data\shaders\fluid_2d\rbgs.comp.spv)",
                    .layouts =  {
                            &uniformsSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                            &_fieldDescriptorSetLayout,
                    },
                    .ranges = { {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(linearSolverConstants) } }
                },
                {
                    .name = "divergence",
                    .shadePath = R"(C:\Users\Josiah Ebhomenye\CLionProjects\vglib_examples\dependencies\vglib.github.io\data\shaders\fluid_2d\divergence.comp.spv)",
                    .layouts =  {
                            &uniformsSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                            &_fieldDescriptorSetLayout
                     },
                    .ranges = { {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(projectConstants) } }
                },
                {
                    .name = "divergence_free_field",
                    .shadePath = R"(C:\Users\Josiah Ebhomenye\CLionProjects\vglib_examples\dependencies\vglib.github.io\data\shaders\fluid_2d\divergence_free_field.comp.spv)",
                    .layouts =  {
                            &uniformsSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                            &_fieldDescriptorSetLayout,  &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout
                      },
                    .ranges = { {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(projectConstants) } }
                }
        };
    }

    
    void FluidSolver::velocityStep(VkCommandBuffer commandBuffer) {
        if(!options.advectVField) return;

        clearForces(commandBuffer);
        applyForces(commandBuffer);
        diffuseVelocityField(commandBuffer);

        advectVectorField(commandBuffer);
        project(commandBuffer);
    }

    
    void FluidSolver::clearForces(VkCommandBuffer commandBuffer) {
        clear(commandBuffer, _forceField[0]);
        clear(commandBuffer, _forceField[1]);
    }

    
    void FluidSolver::applyForces(VkCommandBuffer commandBuffer) {
        applyExternalForces(commandBuffer);
        computeVorticityConfinement(commandBuffer);
        addForcesToVectorField(commandBuffer, _forceField);
    }

    
    void FluidSolver::clear(VkCommandBuffer commandBuffer, Texture &texture) {
        texture.image.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, DEFAULT_SUB_RANGE
                , VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT
                , VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkClearColorValue color{ {0.f, 0.f, 0.f, 0.f}};
        VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(commandBuffer, texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);

        texture.image.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_GENERAL, DEFAULT_SUB_RANGE
                , VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT
                , VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    }

    void FluidSolver::diffuseVelocityField(VkCommandBuffer commandBuffer) {
        if(options.viscosity == MIN_FLOAT) return;
        linearSolverConstants.is_vector_field = true;
        diffuse(commandBuffer, _vectorField.u, options.viscosity);
        diffuse(commandBuffer, _vectorField.v, options.viscosity);
        project(commandBuffer);
    }

    void FluidSolver::diffuse(VkCommandBuffer commandBuffer, Field& field, float rate) {
        if(rate == MIN_FLOAT) return;
        linearSolverConstants.alpha = (_delta.x * _delta.x * _delta.x * _delta.y)/(_timeStep * rate);
        linearSolverConstants.rBeta = 1.0f/((2.0f * glm::dot(_delta, _delta)) + linearSolverConstants.alpha);
        if(linearSolverStrategy == LinearSolverStrategy::Jacobi) {
            jacobiSolver(commandBuffer, field, field);
        }else {
            rbgsSolver(commandBuffer, field, field);
        }
    }

    void FluidSolver::project(VkCommandBuffer commandBuffer) {
        if(!options.project) return;

        computeDivergence(commandBuffer);
        solvePressure(commandBuffer);
        computeDivergenceFreeField(commandBuffer);
        _vectorField.swap();
    }

    void FluidSolver::advectVectorField(VkCommandBuffer commandBuffer) {
        advect(commandBuffer, _vectorField.u);
        advect(commandBuffer, _vectorField.v);

        _vectorField.swap();
//        bridgeOut(commandBuffer);

    }

    void FluidSolver::quantityStep(VkCommandBuffer commandBuffer) {
        for(auto& quantity : _quantities) {
            quantityStep(commandBuffer, quantity);
        }
    }

    void FluidSolver::quantityStep(VkCommandBuffer commandBuffer, Quantity& quantity) {
        clearSources(commandBuffer, quantity);
        updateSources(commandBuffer, quantity);
        addSource(commandBuffer, quantity);
        diffuseQuantity(commandBuffer, quantity);
        advectQuantity(commandBuffer, quantity);
        postAdvection(commandBuffer, quantity);
    }

    void FluidSolver::advect(VkCommandBuffer commandBuffer, Field& field) {
        auto& vf = _vectorField;
        static std::array<VkDescriptorSet, 6> sets;

        sets[0] = uniformDescriptorSet;
        sets[1] = vf.u.descriptorSet[in];
        sets[2] = vf.v.descriptorSet[in];
        sets[3] = field.descriptorSet[in];
        sets[4] = field.descriptorSet[out];
        sets[5] = _linearSamplerDescriptorSet;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("advect"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("advect"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
        addComputeBarrier(commandBuffer);
    }


    void FluidSolver::applyExternalForces(VkCommandBuffer commandBuffer) {
        static std::array<VkDescriptorSet, 2> sets;
//        sets[2] = _valueSamplerDescriptorSet;
        for(const auto& externalForce : _externalForces){
            sets[0] = _forceField.descriptorSet[in];
            sets[1] = _forceField.descriptorSet[out];
            externalForce(commandBuffer, sets, _groupCount);
            addComputeBarrier(commandBuffer);
            _forceField.swap();
        }
    }

    void FluidSolver::addForcesToVectorField(VkCommandBuffer commandBuffer, ForceField &sourceField) {
        static std::array<VkDescriptorSet, 6> sets;
        sets[0] = uniformDescriptorSet;
        sets[1] = _vectorField.u.descriptorSet[in];
        sets[2] = _vectorField.v.descriptorSet[in];
        sets[3] = sourceField.descriptorSet[in];
        sets[4] = _vectorField.u.descriptorSet[out];
        sets[5] = _vectorField.v.descriptorSet[out];

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("apply_force"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("apply_force"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdPushConstants(commandBuffer, layout("apply_force"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(forceConstants), &forceConstants);
        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
        addComputeBarrier(commandBuffer);

        _vectorField.swap();
    }

    void FluidSolver::computeVorticityConfinement(VkCommandBuffer commandBuffer) {

    }

    void FluidSolver::add(ExternalForce &&force) {
        _externalForces.push_back(force);
    }

    void FluidSolver::jacobiSolver(VkCommandBuffer commandBuffer, Field& solution, Field& unknown) {
        static std::array<VkDescriptorSet, 4> sets;
        sets[0] = uniformDescriptorSet;
        sets[1] = solution.descriptorSet[in];

        const auto N = options.poissonIterations;
        for(auto i = 0; i < N; ++i) {
            sets[2] = unknown.descriptorSet[in];
            sets[3] = unknown.descriptorSet[out];

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("jacobi"));
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("jacobi"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
            vkCmdPushConstants(commandBuffer, layout("jacobi"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(linearSolverConstants), &linearSolverConstants);
            vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);

            if(i < N - 1) {
                addComputeBarrier(commandBuffer);
            }

            unknown.swap();
        }
    }

    void FluidSolver::rbgsSolver(VkCommandBuffer commandBuffer, Field& solution, Field& unknown) {
        static std::array<VkDescriptorSet, 4> sets;
        sets[0] = uniformDescriptorSet;
        sets[1] = solution.descriptorSet[in];

        const auto N = options.poissonIterations;

        for(auto i = 0; i < N; ++i) {
            sets[2] = unknown.descriptorSet[in];
            sets[3] = unknown.descriptorSet[out];

            linearSolverConstants.pass = 0;
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("rbgs"));
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("rbgs"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
            vkCmdPushConstants(commandBuffer, layout("rbgs"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(linearSolverConstants), &linearSolverConstants);
            vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
            addComputeBarrier(commandBuffer);
            unknown.swap();

            sets[2] = unknown.descriptorSet[in];
            sets[3] = unknown.descriptorSet[out];

            linearSolverConstants.pass = 1;
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("rbgs"));
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("rbgs"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
            vkCmdPushConstants(commandBuffer, layout("rbgs"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(linearSolverConstants), &linearSolverConstants);
            vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
            unknown.swap();

            if(i < N - 1) {
                addComputeBarrier(commandBuffer);
            }
        }
    }

    void FluidSolver::addComputeBarrier(VkCommandBuffer commandBuffer) {
        static VkMemoryBarrier2 barrier {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
        };

        static VkDependencyInfo dInfo {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &barrier
        };

        vkCmdPipelineBarrier2(commandBuffer, &dInfo);
    }

    void FluidSolver::computeDivergence(VkCommandBuffer commandBuffer) {
        auto& vf = _vectorField;
        static std::array<VkDescriptorSet, 4> sets;

        sets[0] = uniformDescriptorSet;
        sets[1] = vf.u.descriptorSet[in];
        sets[2] = vf.v.descriptorSet[in];
        sets[3] = _divergenceField.descriptorSet[in];

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("divergence"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("divergence"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdPushConstants(commandBuffer, layout("divergence"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(projectConstants), &projectConstants);
        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
        addComputeBarrier(commandBuffer);
    }

    void FluidSolver::solvePressure(VkCommandBuffer commandBuffer) {
        linearSolverConstants.alpha = -_delta.x * _delta.x * _delta.y * _delta.y;
        linearSolverConstants.rBeta = (1.0f/(2.0f * glm::dot(_delta, _delta)));
        linearSolverConstants.is_vector_field = true;

        if(linearSolverStrategy == LinearSolverStrategy::Jacobi) {
            jacobiSolver(commandBuffer, _divergenceField, _pressureField);
        }else {
            rbgsSolver(commandBuffer, _divergenceField, _pressureField);
        }
        addComputeBarrier(commandBuffer);
    }

    void FluidSolver::computeDivergenceFreeField(VkCommandBuffer commandBuffer) {
        auto& vf = _vectorField;
        static std::array<VkDescriptorSet, 6> sets;

        sets[0] = uniformDescriptorSet;
        sets[1] = vf.u.descriptorSet[in];
        sets[2] = vf.v.descriptorSet[in];
        sets[3] = _pressureField.descriptorSet[in];
        sets[4] = vf.u.descriptorSet[out];
        sets[5] = vf.v.descriptorSet[out];

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("divergence_free_field"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("divergence_free_field"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdPushConstants(commandBuffer, layout("divergence_free_field"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(projectConstants), &projectConstants);
        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
        addComputeBarrier(commandBuffer);
    }

    void FluidSolver::dt(float value) {
        _timeStep = value;
        globalConstants.cpu->dt = value;
    }

    float FluidSolver::dt() const {
        return _timeStep;
    }

    void FluidSolver::runSimulation(VkCommandBuffer commandBuffer) {
        velocityStep(commandBuffer);
        quantityStep(commandBuffer);
    }

    std::vector<VulkanDescriptorSetLayout> FluidSolver::forceFieldSetLayouts() {
        return  { _fieldDescriptorSetLayout, _fieldDescriptorSetLayout };
    }

    std::vector<VulkanDescriptorSetLayout> FluidSolver::sourceFieldSetLayouts() {
        return  { _fieldDescriptorSetLayout, _fieldDescriptorSetLayout };
    }

    void FluidSolver::add(Quantity &quantity) {
        auto writes = initializers::writeDescriptorSets<12>();
        auto offset = createDescriptorSet(writes, 0, quantity.field);
        createDescriptorSet(writes, offset, quantity.source);

        device->updateDescriptorSets(writes);

        _quantities.emplace_back(quantity);
    }

    void FluidSolver::clearSources(VkCommandBuffer commandBuffer, Quantity &quantity) {
        clear(commandBuffer, quantity.source[in]);
        clear(commandBuffer, quantity.source[out]);
    }

    void FluidSolver::updateSources(VkCommandBuffer commandBuffer, Quantity &quantity) {
        quantity.update(commandBuffer, quantity.source, _groupCount);
        addComputeBarrier(commandBuffer);
    }

    void FluidSolver::addSource(VkCommandBuffer commandBuffer, Quantity &quantity) {
        static std::array<VkDescriptorSet, 4> sets;
        sets[0] = uniformDescriptorSet;
        sets[1] = quantity.source.descriptorSet[in];
        sets[2] = quantity.field.descriptorSet[in];
        sets[3] = quantity.field.descriptorSet[out];

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("add_sources"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("add_sources"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
        addComputeBarrier(commandBuffer);
        quantity.field.swap();
    }

    void FluidSolver::diffuseQuantity(VkCommandBuffer commandBuffer, Quantity &quantity) {
        linearSolverConstants.is_vector_field = false;
        diffuse(commandBuffer, quantity.field, quantity.diffuseRate);
    }

    void FluidSolver::advectQuantity(VkCommandBuffer commandBuffer, Quantity &quantity) {
        advect(commandBuffer, quantity.field);
        quantity.field.swap();
    }

    void FluidSolver::postAdvection(VkCommandBuffer commandBuffer, Quantity &quantity) {
        quantity.postAdvect(commandBuffer, quantity.field, _groupCount);
        addComputeBarrier(commandBuffer);
    }

}