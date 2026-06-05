#include "fluid_solver_3d.h"
#include "GraphicsPipelineBuilder.hpp"
#include "Vertex.h"
#include "glsl_shaders.hpp"

#include <spdlog/spdlog.h>

namespace {
    constexpr VkFormat FieldFormat = VK_FORMAT_R32G32B32A32_SFLOAT;

    VkImageSubresourceRange colorRange(uint32_t layerCount = 1) {
        return {
                VK_IMAGE_ASPECT_COLOR_BIT,
                0,
                1,
                0,
                layerCount
        };
    }

    void createVolumeTexture(const VulkanDevice& device, Texture& texture, const glm::uvec3& dimensions) {
        texture.format = FieldFormat;
        texture.width = dimensions.x;
        texture.height = dimensions.y;
        texture.depth = dimensions.z;
        texture.layers = 1;
        texture.levels = 1;

        const VkDeviceSize imageSize = sizeof(glm::vec4) * dimensions.x * dimensions.y * dimensions.z;
        std::vector<glm::vec4> data(dimensions.x * dimensions.y * dimensions.z);

        auto stagingBuffer = device.createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY, imageSize);
        stagingBuffer.copy(data.data(), imageSize);

        VkImageCreateInfo imageCreateInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageCreateInfo.flags = VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
        imageCreateInfo.imageType = VK_IMAGE_TYPE_3D;
        imageCreateInfo.format = FieldFormat;
        imageCreateInfo.extent = {dimensions.x, dimensions.y, dimensions.z};
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT
                                | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                                | VK_IMAGE_USAGE_STORAGE_BIT
                                | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                                | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        auto& commandPool = device.commandPoolFor(*device.findFirstActiveQueue());

        texture.image = device.createImage(imageCreateInfo, VMA_MEMORY_USAGE_GPU_ONLY);
        texture.spec = imageCreateInfo;
        texture.image.size = imageSize;

        auto subresource = colorRange();
        texture.image.transitionLayout(commandPool, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresource);

        commandPool.oneTimeCommand([&](auto commandBuffer) {
            VkBufferImageCopy region{};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;
            region.imageOffset = {0, 0, 0};
            region.imageExtent = {dimensions.x, dimensions.y, dimensions.z};

            vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, texture.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        });

        texture.image.transitionLayout(commandPool, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource);

        texture.imageView = texture.image.createView(FieldFormat, VK_IMAGE_VIEW_TYPE_3D, subresource);
        texture.viewType = VK_IMAGE_VIEW_TYPE_3D;
    }
}

FluidSolver3D::FluidSolver3D(VulkanDevice* device, VulkanDescriptorPool* descriptorPool,
                             VulkanRenderPass* displayRenderPass, FileManager* fileManager, glm::vec3 gridSize)
        : FluidSolver(device, descriptorPool, displayRenderPass, fileManager,
                      {static_cast<uint32_t>(gridSize.x), static_cast<uint32_t>(gridSize.y), static_cast<uint32_t>(gridSize.z)})
        , gridSize(gridSize)
        , delta(1.0f / gridSize) {
    globalConstants.dx = {delta.x, 0, 0, gridSize.x};
    globalConstants.dy = {0, delta.y, 0, gridSize.y};
    globalConstants.dz = {0, 0, delta.z, gridSize.z};
    fileManager->addSearchPath("data/shaders/fluid_3d");
}

FluidSolver3D::FluidSolver3D(VulkanDevice* device, VulkanDescriptorPool* descriptorPool,
                             VulkanRenderPass* displayRenderPass, FileManager* fileManager, glm::vec2 gridSize)
        : FluidSolver3D(device, descriptorPool, displayRenderPass, fileManager, {gridSize.x, gridSize.y, 1}) {
}

void FluidSolver3D::init() {
    initBuffers();
    createSamplers();
    initViewVectors();
    createRenderPass();
    initSimData();
    initFullScreenQuad();
    createDescriptorSetLayouts();
    updateDescriptorSets();
    createPipelines();
}

void FluidSolver3D::initSimData() {
    attachmentViews.reserve(18);
    const glm::uvec3 dimensions{width, height, depth};

    createVolumeTexture(*device, vectorField.texture[0], dimensions);
    createVolumeTexture(*device, vectorField.texture[1], dimensions);
    createVolumeTexture(*device, divergenceField.texture[0], dimensions);
    createVolumeTexture(*device, pressureField.texture[0], dimensions);
    createVolumeTexture(*device, pressureField.texture[1], dimensions);
    createVolumeTexture(*device, forceField.texture[0], dimensions);
    createVolumeTexture(*device, forceField.texture[1], dimensions);
    createVolumeTexture(*device, vorticityField.texture[0], dimensions);
    createVolumeTexture(*device, diffuseHelper.texture, dimensions);

    divergenceField.framebuffer[0] = device->createFramebuffer(renderPass, {createAttachmentView(divergenceField.texture[0])}, width, height, depth);
    device->setName<VK_OBJECT_TYPE_FRAMEBUFFER>("divergence_field_3d", divergenceField.framebuffer[0].frameBuffer);

    forceField.framebuffer[0] = device->createFramebuffer(renderPass, {createAttachmentView(forceField.texture[0])}, width, height, depth);
    forceField.framebuffer[1] = device->createFramebuffer(renderPass, {createAttachmentView(forceField.texture[1])}, width, height, depth);
    device->setName<VK_OBJECT_TYPE_FRAMEBUFFER>("force_field_3d_0", forceField.framebuffer[0].frameBuffer);
    device->setName<VK_OBJECT_TYPE_FRAMEBUFFER>("force_field_3d_1", forceField.framebuffer[1].frameBuffer);

    vorticityField.framebuffer[0] = device->createFramebuffer(renderPass, {createAttachmentView(vorticityField.texture[0])}, width, height, depth);
    device->setName<VK_OBJECT_TYPE_FRAMEBUFFER>("vorticity_field_3d", vorticityField.framebuffer[0].frameBuffer);

    for(auto i = 0; i < 2; i++) {
        vectorField.framebuffer[i] = device->createFramebuffer(renderPass, {createAttachmentView(vectorField.texture[i])}, width, height, depth);
        pressureField.framebuffer[i] = device->createFramebuffer(renderPass, {createAttachmentView(pressureField.texture[i])}, width, height, depth);
        device->setName<VK_OBJECT_TYPE_FRAMEBUFFER>(fmt::format("{}_{}", "vector_field_3d", i), vectorField.framebuffer[i].frameBuffer);
        device->setName<VK_OBJECT_TYPE_FRAMEBUFFER>(fmt::format("{}_{}", "pressure_field_3d", i), pressureField.framebuffer[i].frameBuffer);
    }
}

VkImageView FluidSolver3D::createAttachmentView(Texture& texture) {
    auto range = colorRange(depth);
    auto view = texture.image.createView(FieldFormat, VK_IMAGE_VIEW_TYPE_2D_ARRAY, range);
    auto handle = view.handle;
    attachmentViews.push_back(std::move(view));
    return handle;
}

void FluidSolver3D::createPipelines() {
//    @formatter:off
    auto builder = device->graphicsPipelineBuilder();
    screenQuad.pipeline =
            builder
                    .allowDerivatives()
                    .shaderStage()
                    .vertexShader(data_shaders_quad_vert)
                    .fragmentShader(data_shaders_quad_frag)
                    .vertexInputState()
                    .addVertexBindingDescriptions(ClipSpace::bindingDescription())
                    .addVertexAttributeDescriptions(ClipSpace::attributeDescriptions())
                    .inputAssemblyState()
                    .triangleStrip()
                    .viewportState()
                    .viewport()
                    .origin(0, 0)
                    .dimension(width, height)
                    .minDepth(0)
                    .maxDepth(1)
                    .scissor()
                    .offset(0, 0)
                    .extent(width, height)
                    .add()
                    .rasterizationState()
                    .cullBackFace()
                    .frontFaceCounterClockwise()
                    .polygonModeFill()
                    .multisampleState()
                    .rasterizationSamples(VK_SAMPLE_COUNT_1_BIT)
                    .depthStencilState()
                    .enableDepthWrite()
                    .enableDepthTest()
                    .compareOpLess()
                    .minDepthBounds(0)
                    .maxDepthBounds(1)
                    .colorBlendState()
                    .attachment()
                    .add()
                    .layout()
                    .addDescriptorSetLayout(textureSetLayout)
                    .renderPass(renderPass)
                    .subpass(0)
                    .name("fullscreen_quad_3d")
                    .build(screenQuad.layout);

    const auto gridDepth = std::max(depth, 1u);

    advectPipeline.pipeline =
            builder
                    .basePipeline(screenQuad.pipeline)
                    .shaderStage().clear()
                    .vertexShader(data_shaders_fluid_3d_layered_quad_vert)
                    .addSpecialization(gridDepth, 0)
                    .geometryShader(data_shaders_fluid_3d_layered_quad_geom)
                    .fragmentShader(data_shaders_fluid_3d_advect_frag)
                    .layout().clear()
                    .addDescriptorSetLayouts({globalConstantsSet, textureSetLayout, advectTextureSet, samplerSet})
                    .renderPass(renderPass)
                    .name("advect_3d")
                    .build(advectPipeline.layout);

    divergence.pipeline =
            builder
                    .basePipeline(screenQuad.pipeline)
                    .shaderStage().clear()
                    .vertexShader(data_shaders_fluid_3d_layered_quad_vert)
                    .addSpecialization(gridDepth, 0)
                    .geometryShader(data_shaders_fluid_3d_layered_quad_geom)
                    .fragmentShader(data_shaders_fluid_3d_divergence_frag)
                    .layout().clear()
                    .addDescriptorSetLayouts({globalConstantsSet, textureSetLayout})
                    .renderPass(renderPass)
                    .name("divergence_3d")
                    .build(divergence.layout);

    divergenceFree.pipeline =
            builder
                    .basePipeline(screenQuad.pipeline)
                    .shaderStage().clear()
                    .vertexShader(data_shaders_fluid_3d_layered_quad_vert)
                    .addSpecialization(gridDepth, 0)
                    .geometryShader(data_shaders_fluid_3d_layered_quad_geom)
                    .fragmentShader(data_shaders_fluid_3d_divergence_free_field_frag)
                    .layout().clear()
                    .addDescriptorSetLayouts({globalConstantsSet, textureSetLayout, textureSetLayout})
                    .renderPass(renderPass)
                    .name("divergence_free_field_3d")
                    .build(divergenceFree.layout);

    jacobi.pipeline =
            builder
                    .basePipeline(screenQuad.pipeline)
                    .shaderStage().clear()
                    .vertexShader(data_shaders_fluid_3d_layered_quad_vert)
                    .addSpecialization(gridDepth, 0)
                    .geometryShader(data_shaders_fluid_3d_layered_quad_geom)
                    .fragmentShader(data_shaders_fluid_3d_jacobi_frag)
                    .layout().clear()
                    .addDescriptorSetLayouts({globalConstantsSet, textureSetLayout, textureSetLayout})
                    .addPushConstantRange(VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(jacobi.constants))
                    .renderPass(renderPass)
                    .name("jacobi_3d")
                    .build(jacobi.layout);

    addSourcePipeline.pipeline =
            builder
                    .basePipeline(screenQuad.pipeline)
                    .shaderStage().clear()
                    .vertexShader(data_shaders_fluid_3d_layered_quad_vert)
                    .addSpecialization(gridDepth, 0)
                    .geometryShader(data_shaders_fluid_3d_layered_quad_geom)
                    .fragmentShader(data_shaders_fluid_3d_add_sources_frag)
                    .layout().clear()
                    .addDescriptorSetLayouts({textureSetLayout, textureSetLayout})
                    .addPushConstantRange(VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(addSourcePipeline.constants))
                    .renderPass(renderPass)
                    .name("add_sources_3d")
                    .build(addSourcePipeline.layout);

    vorticity.pipeline =
            builder
                    .basePipeline(screenQuad.pipeline)
                    .shaderStage().clear()
                    .vertexShader(data_shaders_fluid_3d_layered_quad_vert)
                    .addSpecialization(gridDepth, 0)
                    .geometryShader(data_shaders_fluid_3d_layered_quad_geom)
                    .fragmentShader(data_shaders_fluid_3d_vorticity_frag)
                    .layout().clear()
                    .addDescriptorSetLayouts({globalConstantsSet, textureSetLayout})
                    .renderPass(renderPass)
                    .name("vorticity_3d")
                    .build(vorticity.layout);

    vorticityForce.pipeline =
            builder
                    .basePipeline(screenQuad.pipeline)
                    .shaderStage().clear()
                    .vertexShader(data_shaders_fluid_3d_layered_quad_vert)
                    .addSpecialization(gridDepth, 0)
                    .geometryShader(data_shaders_fluid_3d_layered_quad_geom)
                    .fragmentShader(data_shaders_fluid_3d_vorticity_force_frag)
                    .layout().clear()
                    .addDescriptorSetLayouts({globalConstantsSet, textureSetLayout, textureSetLayout})
                    .addPushConstantRange(VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(vorticityForce.constants))
                    .renderPass(renderPass)
                    .name("vorticity_force_3d")
                    .build(vorticityForce.layout);
//    @formatter:on
}

void FluidSolver3D::set(VulkanBuffer vectorFieldBuffer) {
    device->graphicsCommandPool().oneTimeCommand([&](auto commandBuffer) {
        vectorField.texture[in].image.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, colorRange(),
                                                       VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                                                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {width, height, depth};

        vkCmdCopyBufferToImage(commandBuffer, vectorFieldBuffer, vectorField.texture[in].image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        vectorField.texture[in].image.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, colorRange(),
                                                       VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                                       VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    });
}

void FluidSolver3D::add(ExternalForce&& force) {
    externalForces.push_back(force);
}

void FluidSolver3D::runSimulation(VkCommandBuffer commandBuffer) {
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, screenQuad.vertices, &offset);
    velocityStep(commandBuffer);
    quantityStep(commandBuffer);
    _elapsedTime += timeStep;
}

void FluidSolver3D::velocityStep(VkCommandBuffer commandBuffer) {
    if(!options.advectVField) return;

    advectVectorField(commandBuffer);
    if(options.viscosity > 0) {
        jacobi.constants.isVectorField = 1;
        diffuse(commandBuffer, vectorField, options.viscosity);
    }
    clearForces(commandBuffer);
    applyForces(commandBuffer);
    project(commandBuffer);
}

void FluidSolver3D::quantityStep(VkCommandBuffer commandBuffer) {
    for(auto& quantity : quantities) {
        quantityStep(commandBuffer, quantity);
    }
}

void FluidSolver3D::quantityStep(VkCommandBuffer commandBuffer, Quantity& quantity) {
    clearSources(commandBuffer, quantity);
    updateSources(commandBuffer, quantity);
    addSource(commandBuffer, quantity);
    diffuseQuantity(commandBuffer, quantity);
    advectQuantity(commandBuffer, quantity);
    postAdvection(commandBuffer, quantity);
}

void FluidSolver3D::clearSources(VkCommandBuffer commandBuffer, Quantity& quantity) {
    clear(commandBuffer, quantity.source.texture[in]);
    clear(commandBuffer, quantity.source.texture[out]);
}

void FluidSolver3D::updateSources(VkCommandBuffer commandBuffer, Quantity& quantity) {
    quantity.update(commandBuffer, quantity.source);
}

void FluidSolver3D::addSource(VkCommandBuffer commandBuffer, Quantity& quantity) {
    addSources(commandBuffer, quantity.source, quantity.field);
}

void FluidSolver3D::diffuseQuantity(VkCommandBuffer commandBuffer, Quantity& quantity) {
    jacobi.constants.isVectorField = 0;
    diffuse(commandBuffer, quantity.field, quantity.diffuseRate);
}

void FluidSolver3D::advectQuantity(VkCommandBuffer commandBuffer, Quantity& quantity) {
    static std::array<VkDescriptorSet, 2> sets;
    sets[0] = vectorField.descriptorSet[in];
    sets[1] = quantity.field.advectDescriptorSet[in];

    advect(commandBuffer, sets, quantity.field.framebuffer[out]);
    quantity.field.swap();
}

void FluidSolver3D::postAdvection(VkCommandBuffer commandBuffer, Quantity& quantity) {
    withRenderPass(commandBuffer, quantity.field.framebuffer[out], [&](VkCommandBuffer commandBuffer) {
        auto postAction = quantity.postAdvect(commandBuffer, quantity.field);
        if(postAction) {
            quantity.field.swap();
        }
    });
}

void FluidSolver3D::applyForces(VkCommandBuffer commandBuffer) {
    applyExternalForces(commandBuffer);
    computeVorticityConfinement(commandBuffer);
    addSources(commandBuffer, forceField, vectorField);
}

void FluidSolver3D::applyExternalForces(VkCommandBuffer commandBuffer) {
    for(const auto& externalForce : externalForces) {
        withRenderPass(commandBuffer, forceField.framebuffer[out], [&](auto commandBuffer) {
            externalForce(commandBuffer, forceField.descriptorSet[in]);
        });
        forceField.swap();
    }
}

void FluidSolver3D::computeVorticityConfinement(VkCommandBuffer commandBuffer) {
    if(!options.vorticity) return;
    withRenderPass(commandBuffer, vorticityField.framebuffer[0], [&](auto commandBuffer) {
        static std::array<VkDescriptorSet, 2> sets;
        sets[0] = globalConstantsDescriptorSet;
        sets[1] = vectorField.descriptorSet[in];
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vorticity.pipeline.handle);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vorticity.layout.handle,
                                0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        drawVolume(commandBuffer);
    });
    withRenderPass(commandBuffer, forceField.framebuffer[out], [&](auto commandBuffer) {
        static std::array<VkDescriptorSet, 3> sets;
        sets[0] = globalConstantsDescriptorSet;
        sets[1] = vorticityField.descriptorSet[in];
        sets[2] = forceField.descriptorSet[in];
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vorticityForce.pipeline.handle);
        vkCmdPushConstants(commandBuffer, vorticityForce.layout.handle, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(vorticityForce.constants), &vorticityForce.constants);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vorticityForce.layout.handle,
                                0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        drawVolume(commandBuffer);
    });
    forceField.swap();
}

void FluidSolver3D::clearForces(VkCommandBuffer commandBuffer) {
    clear(commandBuffer, forceField.texture[in]);
    clear(commandBuffer, forceField.texture[out]);
}

void FluidSolver3D::clear(VkCommandBuffer commandBuffer, Texture& texture) {
    texture.image.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, colorRange(),
                                   VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkClearColorValue color{{0.f, 0.f, 0.f, 0.f}};
    auto range = colorRange();
    vkCmdClearColorImage(commandBuffer, texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);

    texture.image.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, colorRange(),
                                   VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
}

void FluidSolver3D::addSources(VkCommandBuffer commandBuffer, Field& sourceField, Field& destinationField) {
    addSourcePipeline.constants.dt = globalConstants.dt;
    withRenderPass(commandBuffer, destinationField.framebuffer[out], [&](auto commandBuffer) {
        static std::array<VkDescriptorSet, 2> sets;
        sets[0] = sourceField.descriptorSet[in];
        sets[1] = destinationField.descriptorSet[in];
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, addSourcePipeline.pipeline.handle);
        vkCmdPushConstants(commandBuffer, addSourcePipeline.layout.handle, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(addSourcePipeline.constants), &addSourcePipeline.constants);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, addSourcePipeline.layout.handle,
                                0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        drawVolume(commandBuffer);
    });
    destinationField.swap();
}

void FluidSolver3D::advectVectorField(VkCommandBuffer commandBuffer) {
    static std::array<VkDescriptorSet, 2> sets;
    sets[0] = vectorField.descriptorSet[in];
    sets[1] = vectorField.advectDescriptorSet[in];

    advect(commandBuffer, sets, vectorField.framebuffer[out]);
    vectorField.swap();
}

void FluidSolver3D::advect(VkCommandBuffer commandBuffer, const std::array<VkDescriptorSet, 2>& inSets,
                           VulkanFramebuffer& framebuffer) {
    static std::array<VkDescriptorSet, 4> sets;
    sets[0] = globalConstantsDescriptorSet;
    sets[1] = inSets[0];
    sets[2] = inSets[1];
    sets[3] = samplerDescriptorSet;
    withRenderPass(commandBuffer, framebuffer, [&](auto commandBuffer) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, advectPipeline.pipeline.handle);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, advectPipeline.layout.handle,
                                0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        drawVolume(commandBuffer);
    });
}

void FluidSolver3D::project(VkCommandBuffer commandBuffer) {
    if(!options.project) return;
    computeDivergence(commandBuffer);
    solvePressure(commandBuffer);
    computeDivergenceFreeField(commandBuffer);
    vectorField.swap();
}

void FluidSolver3D::computeDivergence(VkCommandBuffer commandBuffer) {
    static std::array<VkDescriptorSet, 2> sets;
    sets[0] = globalConstantsDescriptorSet;
    sets[1] = vectorField.descriptorSet[in];
    withRenderPass(commandBuffer, divergenceField.framebuffer[0], [&](auto commandBuffer) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, divergence.pipeline.handle);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, divergence.layout.handle,
                                0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        drawVolume(commandBuffer);
    });
}

void FluidSolver3D::solvePressure(VkCommandBuffer commandBuffer) {
    jacobi.constants.isVectorField = 0;
    const auto dx2 = delta.x * delta.x;
    const auto dy2 = delta.y * delta.y;
    const auto dz2 = delta.z * delta.z;
    const auto alpha = -dx2 * dy2 * dz2;
    const auto rBeta = 1.0f / (2.0f * (dy2 * dz2 + dx2 * dz2 + dx2 * dy2));
    for(int i = 0; i < options.poissonIterations; i++) {
        withRenderPass(commandBuffer, pressureField.framebuffer[out], [&](auto commandBuffer) {
            jacobiIteration(commandBuffer, pressureField.descriptorSet[in], divergenceField.descriptorSet[in], alpha, rBeta);
            pressureField.swap();
        });
    }
}

void FluidSolver3D::jacobiIteration(VkCommandBuffer commandBuffer, VkDescriptorSet unknownDescriptor,
                                    VkDescriptorSet solutionDescriptor, float alpha, float rBeta) {
    jacobi.constants.alpha = alpha;
    jacobi.constants.rBeta = rBeta;
    static std::array<VkDescriptorSet, 3> sets;
    sets[0] = globalConstantsDescriptorSet;
    sets[1] = solutionDescriptor;
    sets[2] = unknownDescriptor;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, jacobi.pipeline.handle);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, jacobi.layout.handle,
                            0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);

    vkCmdPushConstants(commandBuffer, jacobi.layout.handle, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(jacobi.constants), &jacobi.constants);
    drawVolume(commandBuffer);
}

void FluidSolver3D::computeDivergenceFreeField(VkCommandBuffer commandBuffer) {
    withRenderPass(commandBuffer, vectorField.framebuffer[out], [&](auto commandBuffer) {
        static std::array<VkDescriptorSet, 3> sets;
        sets[0] = globalConstantsDescriptorSet;
        sets[1] = vectorField.descriptorSet[in];
        sets[2] = pressureField.descriptorSet[in];
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, divergenceFree.pipeline.handle);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, divergenceFree.layout.handle,
                                0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        drawVolume(commandBuffer);
    });
}

void FluidSolver3D::diffuse(VkCommandBuffer commandBuffer, Field& field, float rate) {
    if(rate <= 0) return;
    diffuseHelper.texture.image.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, colorRange(),
                                                 VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    field.texture[in].image.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, colorRange(),
                                             VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkImageSubresourceLayers subresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    VkExtent3D extent3D{width, height, depth};
    VkImageCopy region{subresourceLayers, {0, 0, 0}, subresourceLayers, {0, 0, 0}, extent3D};

    vkCmdCopyImage(commandBuffer, field.texture[in].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   diffuseHelper.texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    field.texture[in].image.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, colorRange(),
                                             VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                                             VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    diffuseHelper.texture.image.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, colorRange(),
                                                 VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    const auto dx2 = delta.x * delta.x;
    const auto dy2 = delta.y * delta.y;
    const auto dz2 = delta.z * delta.z;
    const auto alpha = (dx2 * dy2 * dz2) / (timeStep * rate);
    const auto rBeta = 1.0f / (2.0f * (dy2 * dz2 + dx2 * dz2 + dx2 * dy2) + alpha);
    for(int i = 0; i < options.poissonIterations; i++) {
        withRenderPass(commandBuffer, field.framebuffer[out], [&](auto commandBuffer) {
            jacobiIteration(commandBuffer, field.descriptorSet[in], diffuseHelper.solutionDescriptorSet, alpha, rBeta);
            field.swap();
        });
    }
}

void FluidSolver3D::renderVectorField(VkCommandBuffer) {
    if(options.showArrows) {
        spdlog::debug("FluidSolver3D::renderVectorField is not implemented for volume fields");
    }
}

void FluidSolver3D::add(Quantity& quantity) {
    createFrameBuffer(quantity);
    createDescriptorSets(quantity);
    quantities.emplace_back(quantity);
}

void FluidSolver3D::createFrameBuffer(Quantity& quantity) {
    for(auto i = 0; i < 2; i++) {
        quantity.field.framebuffer[i] = device->createFramebuffer(renderPass, {createAttachmentView(quantity.field.texture[i])}, width, height, depth);
        quantity.source.framebuffer[i] = device->createFramebuffer(renderPass, {createAttachmentView(quantity.source.texture[i])}, width, height, depth);
        device->setName<VK_OBJECT_TYPE_FRAMEBUFFER>(fmt::format("{}_3d_{}", quantity.name, i), quantity.field.framebuffer[i].frameBuffer);
        device->setName<VK_OBJECT_TYPE_FRAMEBUFFER>(fmt::format("{}_3d_source_{}", quantity.name, i), quantity.source.framebuffer[i].frameBuffer);
    }
}

void FluidSolver3D::dt(float value) {
    timeStep = value;
    globalConstants.dt = timeStep;
    updateUBOs();
}

float FluidSolver3D::dt() const {
    return timeStep;
}

void FluidSolver3D::advectVelocity(bool flag) {
    options.advectVField = flag;
}

void FluidSolver3D::project(bool flag) {
    options.project = flag;
}

void FluidSolver3D::showVectors(bool flag) {
    options.showArrows = flag;
}

void FluidSolver3D::applyVorticity(bool flag) {
    options.vorticity = flag;
}

void FluidSolver3D::poissonIterations(int value) {
    options.poissonIterations = value;
}

void FluidSolver3D::viscosity(float value) {
    options.viscosity = value;
}

void FluidSolver3D::updateUBOs() {
    globalConstantsBuffer.copy(&globalConstants, sizeof(globalConstants));
}

void FluidSolver3D::ensureBoundaryCondition(bool flag) {
    globalConstants.ensureBoundaryCondition = static_cast<int>(flag);
    updateUBOs();
}

float FluidSolver3D::elapsedTime() {
    return _elapsedTime;
}

std::tuple<VkDeviceSize, void*> FluidSolver3D::getGlobalConstants() {
    return std::make_tuple(sizeof(globalConstants), &globalConstants);
}

void FluidSolver3D::drawVolume(VkCommandBuffer commandBuffer) const {
    vkCmdDraw(commandBuffer, 4, std::max(depth, 1u), 0, 0);
}
