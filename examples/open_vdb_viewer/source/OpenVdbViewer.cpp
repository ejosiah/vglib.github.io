#include "OpenVdbViewer.hpp"
#include "GraphicsPipelineBuilder.hpp"
#include "DescriptorSetBuilder.hpp"
#include "ImGuiPlugin.hpp"
#include "L2DFileDialog.h"
#include <openvdb/openvdb.h>
#include <sstream>
#include "glm_format.h"
#include "primitives.h"

OpenVdbViewer::OpenVdbViewer(const Settings& settings) : VulkanBaseApp("Open Vdb viewer", settings) {
    fileManager.addSearchPathFront(".");
    fileManager.addSearchPathFront("../../examples/open_vdb_viewer");
    fileManager.addSearchPathFront("../../examples/open_vdb_viewer/data");
    fileManager.addSearchPathFront("../../examples/open_vdb_viewer/spv");
    fileManager.addSearchPathFront("../../examples/open_vdb_viewer/models");
    fileManager.addSearchPathFront("../../examples/open_vdb_viewer/textures");
}

void OpenVdbViewer::initApp() {
    openvdb::initialize();
    initCamera();
    createBuffers();
    createPlaceHolderTexture();
    createDescriptorPool();
    createDescriptorSetLayouts();
    updateDescriptorSets();
    updateVolumeDescriptorSets();
    createCommandPool();
    createPipelineCache();
    createRenderPipeline();
}

void OpenVdbViewer::initCamera() {
    OrbitingCameraSettings cameraSettings;
//    FirstPersonSpectatorCameraSettings cameraSettings;
    cameraSettings.orbitMinZoom = 0.1;
    cameraSettings.orbitMaxZoom = 1000.0f;
    cameraSettings.offsetDistance = 2.0f;
    cameraSettings.modelHeight = 0.5;
    cameraSettings.fieldOfView = 60.0f;
    cameraSettings.aspectRatio = float(swapChain.extent.width)/float(swapChain.extent.height);

    camera = std::make_unique<OrbitingCameraController>(dynamic_cast<InputManager&>(*this), cameraSettings);
}


void OpenVdbViewer::updateCamera() {
    auto diagonal = volumeUbo->boxMax - volumeUbo->boxMin;
    auto halfMaxDist = glm::max(diagonal.x, glm::max(diagonal.y, diagonal.z)) * 0.5f;
    OrbitingCameraSettings cameraSettings;
//    FirstPersonSpectatorCameraSettings cameraSettings;
    cameraSettings.orbitMinZoom = 0.1;
    cameraSettings.orbitMaxZoom = 1000.0f;
    cameraSettings.offsetDistance = halfMaxDist + halfMaxDist * .25f;
    cameraSettings.modelHeight = diagonal.y * 0.5f;
    cameraSettings.fieldOfView = 90.0f;
    cameraSettings.horizontalFov = true;
    cameraSettings.aspectRatio = float(swapChain.extent.width)/float(swapChain.extent.height);

    camera = std::make_unique<OrbitingCameraController>(dynamic_cast<InputManager&>(*this), cameraSettings);
    camera->zoomDelta = 1;
}

void OpenVdbViewer::createBuffers() {
    cameraUboBuffer = device.createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(CameraUbo));
    volumeUboBuffer = device.createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(VolumeUbo));

    cameraUbo = reinterpret_cast<CameraUbo*>(cameraUboBuffer.map());
    volumeUbo = reinterpret_cast<VolumeUbo*>(volumeUboBuffer.map());
    volumeUbo->time = 0;
    volumeUbo->boxMin = glm::vec3(-.5);
    volumeUbo->boxMax = glm::vec3(.5);
    volumeUbo->numSamples = 100;
    volumeUbo->coneSpread = 10;
    volumeUbo->lightIntensity = 200;
    volumeUbo->g = 0.2;
    volumeUbo->frame = 0;
    volumeUbo->width = width;
    volumeUbo->height = height;
    volumeUbo->invMaxDensity = 0;
    volumeUbo->lightPosition = glm::vec3(0);
    updateCamera();

    auto vertices = ClipSpace::Quad::positions;
    vertexBuffer = device.createDeviceLocalBuffer(vertices.data(), BYTE_SIZE(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    placeHolderVertexBuffer = device.createBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, sizeof(glm::vec4));

    auto sphere = primitives::sphere(50, 50, 1.0f, glm::mat4{1}, glm::vec4(1, 0, 0, 0));
    light.vertexBuffer = device.createDeviceLocalBuffer(sphere.vertices.data(), BYTE_SIZE(sphere.vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    light.indexBuffer = device.createDeviceLocalBuffer(sphere.indices.data(), BYTE_SIZE(sphere.indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    light.intensity = 200;
}

void OpenVdbViewer::createPlaceHolderTexture() {
    std::vector<float> placeHolder(8);

    textures::create(device, volumeTexture, VK_IMAGE_TYPE_3D, VK_FORMAT_R32_SFLOAT, placeHolder.data(),
                     {2, 2, 2}, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, sizeof(float));
}

void OpenVdbViewer::createDescriptorPool() {
    constexpr uint32_t maxSets = 100;
    std::array<VkDescriptorPoolSize, 16> poolSizes{
            {
                    {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100 * maxSets},
                    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 100 * maxSets},
                    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100 * maxSets},
                    { VK_DESCRIPTOR_TYPE_SAMPLER, 100 * maxSets },
                    { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100 * maxSets },
                    { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 100 * maxSets },
                    { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 100 * maxSets },
                    { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 100 * maxSets },
                    { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 100 * maxSets },
                    { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 100 * maxSets },
                    { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100 * maxSets },
                    { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 100 * maxSets },
                    { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 100 * maxSets },
                    { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 100 * maxSets },
                    { VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT, 100 * maxSets },
                    { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 100 * maxSets }
            }
    };
    descriptorPool = device.createDescriptorPool(maxSets, poolSizes, VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

}

void OpenVdbViewer::createDescriptorSetLayouts() {
    descriptorSetLayout =
        device.descriptorSetLayoutBuilder()
            .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
        .createLayout();
    
    volumeDescriptorSetLayout =
        device.descriptorSetLayoutBuilder()
            .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_FRAGMENT_BIT)
            .binding(1)
                .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_FRAGMENT_BIT)
        .createLayout();

    auto sets = descriptorPool.allocate({ descriptorSetLayout, volumeDescriptorSetLayout});
    descriptorSet = sets[0];
    volumeDescriptor = sets[1];

}

void OpenVdbViewer::updateDescriptorSets(){
    
    auto writes = initializers::writeDescriptorSets();
    
    writes[0].dstSet = descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].descriptorCount = 1;
    VkDescriptorBufferInfo cameraInfo{cameraUboBuffer, 0, VK_WHOLE_SIZE};
    writes[0].pBufferInfo = &cameraInfo;
    
    device.updateDescriptorSets(writes);
}

void OpenVdbViewer::updateVolumeDescriptorSets() {
    auto writes = initializers::writeDescriptorSets<2>();
    
    writes[0].dstSet = volumeDescriptor;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].descriptorCount = 1;
    VkDescriptorBufferInfo bufferInfo{ volumeUboBuffer, 0, VK_WHOLE_SIZE };
    writes[0].pBufferInfo = &bufferInfo;
    
    writes[1].dstSet = volumeDescriptor;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].descriptorCount = 1;
    VkDescriptorImageInfo imageInfo{ volumeTexture.sampler, volumeTexture.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    writes[1].pImageInfo = &imageInfo;
    
    device.updateDescriptorSets(writes);
}

void OpenVdbViewer::createCommandPool() {
    commandPool = device.createCommandPool(*device.queueFamilyIndex.graphics, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    commandBuffers = commandPool.allocateCommandBuffers(swapChainImageCount);
}

void OpenVdbViewer::createPipelineCache() {
    pipelineCache = device.createPipelineCache();
}


void OpenVdbViewer::createRenderPipeline() {
    //    @formatter:off
    auto builder = device.graphicsPipelineBuilder();
    rayMarching.pipeline =
        builder
            .allowDerivatives()
            .shaderStage()
                .vertexShader(resource("volume.vert.spv"))
                .fragmentShader(resource("ray_marcher.frag.spv"))
            .vertexInputState()
                .addVertexBindingDescriptions(ClipSpace::bindingDescription())
                .addVertexAttributeDescriptions(ClipSpace::attributeDescriptions())
            .inputAssemblyState()
                .triangleStrip()
            .viewportState()
                .viewport()
                    .origin(0, 0)
                    .dimension(swapChain.extent)
                    .minDepth(0)
                    .maxDepth(1)
                .scissor()
                    .offset(0, 0)
                    .extent(swapChain.extent)
                .add()
                .rasterizationState()
                    .cullBackFace()
                    .frontFaceCounterClockwise()
                    .polygonModeFill()
                .multisampleState()
                    .rasterizationSamples(settings.msaaSamples)
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
                    .addDescriptorSetLayout(descriptorSetLayout)
                    .addDescriptorSetLayout(volumeDescriptorSetLayout)
                .renderPass(renderPass)
                .subpass(0)
                .name("ray_marching")
                .pipelineCache(pipelineCache)
            .build(rayMarching.layout);

    deltaTracking.pipeline =
        builder
            .basePipeline(rayMarching.pipeline)
            .shaderStage()
                .fragmentShader(resource("delta_tracking.frag.spv"))
            .name("delta_tracking")
        .build(deltaTracking.layout);

    background.pipeline =
        builder
            .shaderStage()
                .fragmentShader(resource("background.frag.spv"))
            .depthStencilState()
                .disableDepthTest()
                .disableDepthWrite()
            .name("background")
        .build(background.layout);
			
//    sliceRenderer.pipeline =
//        builder
//            .allowDerivatives()
//            .shaderStage()
//                .vertexShader(resource("slice.vert.spv"))
//                .geometryShader(resource("slice.geom.spv"))
//                .fragmentShader(resource("slice.frag.spv"))
//            .vertexInputState().clear()
//                .addVertexBindingDescription(0, sizeof(glm::vec4), VK_VERTEX_INPUT_RATE_VERTEX)
//                .addVertexAttributeDescription(0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0)
//            .inputAssemblyState()
//                .points()
//                .colorBlendState()
//                    .attachment().clear()
//                        .enableBlend()
//                        .colorBlendOp().add()
//                        .alphaBlendOp().add()
//                        .srcColorBlendFactor().srcAlpha()
//                        .dstColorBlendFactor().oneMinusSrcAlpha()
//                        .srcAlphaBlendFactor().one()
//                        .dstAlphaBlendFactor().zero()
//                    .add()
//                .layout().clear()
//                    .addDescriptorSetLayout(volumeDescriptorSetLayout)
//                    .addPushConstantRange(VK_SHADER_STAGE_GEOMETRY_BIT, 0, sizeof(sliceRenderer.constants))
//                .renderPass(renderPass)
//                .subpass(0)
//                .name("volume_slice_renderer")
//                .pipelineCache(pipelineCache)
//            .build(sliceRenderer.layout);
    //    @formatter:on

    light.pipeline =
        builder
            .shaderStage().clear()
                .vertexShader(resource("flat.vert.spv"))
                .fragmentShader(resource("flat.frag.spv"))
            .vertexInputState().clear()
                .addVertexBindingDescriptions(Vertex::bindingDisc())
                .addVertexAttributeDescriptions(Vertex::attributeDisc())
            .inputAssemblyState()
                .triangleStrip()
                .enablePrimitiveRestart()
            .depthStencilState()
                .enableDepthWrite()
                .enableDepthTest()
                .compareOpLess()
                .minDepthBounds(0)
                .maxDepthBounds(1)
            .colorBlendState()
                .attachments(1)
            .layout().clear()
                .addPushConstantRange(Camera::pushConstant())
            .name("light_renderer")
        .build(light.layout);
}


void OpenVdbViewer::onSwapChainDispose() {
    dispose(rayMarching.pipeline);
}

void OpenVdbViewer::onSwapChainRecreation() {
    updateDescriptorSets();
    createRenderPipeline();
}

VkCommandBuffer *OpenVdbViewer::buildCommandBuffers(uint32_t imageIndex, uint32_t &numCommandBuffers) {
    numCommandBuffers = 1;
    auto& commandBuffer = commandBuffers[imageIndex];

    VkCommandBufferBeginInfo beginInfo = initializers::commandBufferBeginInfo();
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    static std::array<VkClearValue, 2> clearValues;
    clearValues[0].color = {0, 0, 0, 1};
    clearValues[1].depthStencil = {1.0, 0u};

    VkRenderPassBeginInfo rPassInfo = initializers::renderPassBeginInfo();
    rPassInfo.clearValueCount = COUNT(clearValues);
    rPassInfo.pClearValues = clearValues.data();
    rPassInfo.framebuffer = framebuffers[imageIndex];
    rPassInfo.renderArea.offset = {0u, 0u};
    rPassInfo.renderArea.extent = swapChain.extent;
    rPassInfo.renderPass = renderPass;

    vkCmdBeginRenderPass(commandBuffer, &rPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    renderBackground(commandBuffer);
    renderLight(commandBuffer);
    renderVolume(commandBuffer);
    renderUI(commandBuffer);

    vkCmdEndRenderPass(commandBuffer);

    vkEndCommandBuffer(commandBuffer);

    return &commandBuffer;
}

void OpenVdbViewer::renderLight(VkCommandBuffer commandBuffer) {
    static glm::mat4 xform;
    xform = glm::translate(glm::mat4{1}, light.position);
    xform = glm::scale(xform , glm::vec3(light.scale));
    VkDeviceSize offset = 0;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, light.pipeline);
    camera->push(commandBuffer, light.layout, xform);
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, light.vertexBuffer, &offset);
    vkCmdBindIndexBuffer(commandBuffer, light.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(commandBuffer, light.indexBuffer.sizeAs<uint32_t>(), 1, 0, 0, 0);
}

void OpenVdbViewer::renderVolume(VkCommandBuffer commandBuffer) {
    switch(renderer){
        case Renderer::DELTA_TRACKING:
            renderWithDeltaTracking(commandBuffer);
            break;
        case Renderer::RAY_MARCHING:
            renderWithRayMarching(commandBuffer);
            break;
        default:
            throw std::runtime_error{"invalid renderer"};
    }
}

void OpenVdbViewer::renderWithRayMarching(VkCommandBuffer commandBuffer) {
    static std::array<VkDescriptorSet, 2> sets;
    sets[0] = descriptorSet;
    sets[1] = volumeDescriptor;
    VkDeviceSize  offset = 0;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, rayMarching.pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, rayMarching.layout, 0, COUNT(sets), sets.data(), 0, 0);
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffer, &offset);
    vkCmdDraw(commandBuffer, 4, 1, 0, 0);
}

void OpenVdbViewer::renderWithDeltaTracking(VkCommandBuffer commandBuffer) {
    static std::array<VkDescriptorSet, 2> sets;
    sets[0] = descriptorSet;
    sets[1] = volumeDescriptor;
    VkDeviceSize  offset = 0;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, deltaTracking.pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, deltaTracking.layout, 0, COUNT(sets), sets.data(), 0, 0);
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffer, &offset);
    vkCmdDraw(commandBuffer, 4, 1, 0, 0);
}

void OpenVdbViewer::renderBackground(VkCommandBuffer commandBuffer) {
    static std::array<VkDescriptorSet, 2> sets;
    sets[0] = descriptorSet;
    sets[1] = volumeDescriptor;
    VkDeviceSize  offset = 0;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, background.pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, background.layout, 0, COUNT(sets), sets.data(), 0, 0);
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffer, &offset);
    vkCmdDraw(commandBuffer, 4, 1, 0, 0);
}

void OpenVdbViewer::renderVolumeSlices(VkCommandBuffer commandBuffer) {
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, sliceRenderer.pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, sliceRenderer.layout, 0, 1, &volumeDescriptor,
                            0, VK_NULL_HANDLE);
    vkCmdPushConstants(commandBuffer, sliceRenderer.layout, VK_SHADER_STAGE_GEOMETRY_BIT, 0,
                       sizeof(sliceRenderer.constants), &sliceRenderer.constants);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffer, &offset);
    vkCmdDraw(commandBuffer, 1, sliceRenderer.constants.numSlices, 0, 0);
}

void OpenVdbViewer::renderUI(VkCommandBuffer commandBuffer) {

    if(ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {

            FileDialog::file_dialog_open = ImGui::MenuItem("Open...");

            if(ImGui::MenuItem("Exit")){
                this->exit->press();
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    openFileDialog();

    if(!fileValid){
        ImGui::Begin("Alert");
        ImGui::SetWindowSize({350, 150});
        ImGui::Text("failed opening file %s", fs::path{vdbPath}.filename().c_str());
        if(ImGui::Button("Close")){
            fileValid = true;
            vdbPath.clear();
        }
        ImGui::End();
    }

    ImGui::Begin("Settings");
    ImGui::SetWindowSize({0, 0});

    static int rendererOption = static_cast<int>(renderer);
    ImGui::Text("Renderer:");
    ImGui::Indent(16);
    ImGui::RadioButton("ray marching", &rendererOption, 0); ImGui::SameLine();
    ImGui::RadioButton("delta tracking", &rendererOption, 1); ImGui::SameLine();
    ImGui::RadioButton("path tracer", &rendererOption, 2);
    renderer = static_cast<Renderer>(rendererOption);
    ImGui::Indent(-16);
    ImGui::Separator();

    ImGui::SliderInt("num samples", &volumeUbo->numSamples, 1, 1000);
    ImGui::SliderFloat("Cone spread", &volumeUbo->coneSpread, 0.1, 50);
    ImGui::SliderFloat("g", &volumeUbo->g, -0.999, 0.999);
    ImGui::SliderFloat("intensity", &volumeUbo->lightIntensity, 1, 1000);
    ImGui::End();

    plugin(IM_GUI_PLUGIN).draw(commandBuffer);
}

bool OpenVdbViewer::openFileDialog() {
    static char* file_dialog_buffer = nullptr;
    static char path[500] = "";

    file_dialog_buffer = path;
    FileDialog::file_dialog_open_type = FileDialog::FileDialogType::OpenFile;

    static bool closed = false;
    if (FileDialog::file_dialog_open) {
        FileDialog::ShowFileDialog(&FileDialog::file_dialog_open, file_dialog_buffer, &closed, FileDialog::file_dialog_open_type);
    }

    if(closed) {
        vdbPath = std::string{file_dialog_buffer};
        closed = false;
    }

    return FileDialog::file_dialog_open;
}

void OpenVdbViewer::update(float time) {
    if(!ImGui::IsAnyItemActive()){
        camera->update(time);
    }
    volumeUbo->time += time;
    volumeUbo->frame++;

    auto cam = camera->cam();
    cameraUbo->projection = cam.proj;
    cameraUbo->view = cam.view;
    cameraUbo->inverseProjection = glm::inverse(cam.proj);
    cameraUbo->inverseView = glm::inverse(cam.view);

    sliceRenderer.constants.mvp = cam.proj * cam.view * cam.model;
    sliceRenderer.constants.viewDir = camera->viewDir;

    fileInfo();

    if(int(volumeUbo->time)% 5 == 0){
//        spdlog::info("ligth pos {}", light.position);
    }
}

void OpenVdbViewer::checkAppInputs() {
    camera->processInput();

    static float pDelta = .1;
    if(ImGui::IsKeyDown(static_cast<int>(Key::W))){
        light.position.z -= pDelta;
    }
    if(ImGui::IsKeyDown(static_cast<int>(Key::S))){
        light.position.z += pDelta;
    }
    if(ImGui::IsKeyDown(static_cast<int>(Key::A))){
        light.position.x -= pDelta;
    }
    if(ImGui::IsKeyDown(static_cast<int>(Key::D))){
        light.position.x += pDelta;
    }
    if(ImGui::IsKeyDown(static_cast<int>(Key::E))) {
        light.position.y += pDelta;
    }

    if(ImGui::IsKeyDown(static_cast<int>(Key::Q))){
        light.position.y -= pDelta;
    }

    volumeUbo->lightPosition = light.position;

}

void OpenVdbViewer::cleanup() {
    VulkanBaseApp::cleanup();
}

void OpenVdbViewer::onPause() {
    VulkanBaseApp::onPause();
}

void OpenVdbViewer::fileInfo() {
    if(!vdbPath.empty() && fileValid){
        openvdb::io::File file(vdbPath);

        try{
            file.open();
            loadVolume(file);
//            std::stringstream ss;
//
//            openvdb::GridBase::Ptr grid;
//            ss << "grids:\n";
//            for (auto nameIter = file.beginName(); nameIter != file.endName(); ++nameIter) {
//                ss << "\tgrid: " << nameIter.gridName() << "\n";
//            }
//
//            grid = file.readGrid(file.beginName().gridName());
//
//            ss << "\n\nMetadata:\n";
//            for (auto metaItr = grid->beginMeta(); metaItr != grid->endMeta(); metaItr++) {
//                ss << "\tmetadata: [" << metaItr->first << ", " << metaItr->second->str() << "]" << ", type: "
//                   << metaItr->second->typeName() << "\n";
//            }
//
//
//            auto fGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid);
//            ss << "\nbackground: " << fGrid->background() << "\n";
//
//            auto accessor = fGrid->getAccessor();
//            openvdb::Coord xyz(8, 38, 20);
//            ss << "value at center: " << accessor.getValue(xyz) << "\n";
//
//            auto boxMin = fGrid->getMetadata<openvdb::Vec3IMetadata>("file_bbox_min")->value();
//            auto boxMax = fGrid->getMetadata<openvdb::Vec3IMetadata>("file_bbox_max")->value();
//            decltype(boxMin) center{};
//            center = center.add(boxMin, boxMax);
//            center = center.div(2, center);
//
//            ss << "min bounds: " << boxMin << "\n";
//            ss << "center:" << center << "\n";
//            ss << "max bounds:" << boxMax << "\n";
//
////    ss << "\n\nvalues in grid";
////    for(auto iter = fGrid->cbeginValueOn(); iter; ++iter){
////        ss << "Grid" << iter.getCoord() << " = " << *iter << "\n";
////    }
//            spdlog::info("{}", ss.str());
            file.close();
            vdbPath.clear();
        }catch(...){
            fileValid = false;
        }
    }
}

void OpenVdbViewer::loadVolume(openvdb::io::File& file) {
    static auto remap = [](auto x, auto a, auto b, auto c, auto d){
        return glm::mix(c, d, (x - a)/(b - a));
    };

    static auto to_glm_vec3 = [](openvdb::Vec3i v){
        return glm::vec3(v.x(), v.y(), v.z());
    };
    auto grid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid(file.beginName().gridName()));
    auto boxMin = grid->getMetadata<openvdb::Vec3IMetadata>("file_bbox_min")->value();
    auto boxMax = grid->getMetadata<openvdb::Vec3IMetadata>("file_bbox_max")->value();

    decltype(boxMin) size;
    size = size.sub(boxMax, boxMin);

    std::vector<float> buffer(size.x() * size.y() * size.z());

    openvdb::Coord xyz;
    auto accessor = grid->getAccessor();

    auto& z = xyz.z();
    auto& y = xyz.y();
    auto& x = xyz.x();

    auto to_uvw = [=](openvdb::Coord& xyz){
        glm::vec3 x{xyz.x(), xyz.y(), xyz.z()};
        glm::vec3 a = to_glm_vec3(boxMin);
        glm::vec3 b = to_glm_vec3(boxMax);
        glm::vec3 c{0};
        glm::vec3 d{size.x(), size.y(), size.z()};
        d -= 1.0f;

        auto uvw = remap(x, a, b, c, d);
        return glm::ivec3(uvw);
    };

    static int count = 0;
    float maxDensity = MIN_FLOAT;
    for(z = boxMin.z(); z <= boxMax.z(); z++){
        for(y = boxMin.y(); y <= boxMax.y(); y++){
            for(x = boxMin.x(); x <= boxMax.x(); x++){
                auto voxel = accessor.getValue(xyz);
                if(voxel != 0 && count < 10){
                    count++;
                    spdlog::info("value: {}", voxel);
                }
                auto uvw = to_uvw(xyz);
                auto i = (uvw.z * size.y() + uvw.y) * size.x() + uvw.x;
                buffer[i] = voxel;
                maxDensity = glm::max(voxel, maxDensity);
            }
        }
    }
    textures::create(device, volumeTexture, VK_IMAGE_TYPE_3D, VK_FORMAT_R32_SFLOAT, buffer.data(),
                     {size.x(), size.y(), size.z()}, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, sizeof(float));

    volumeUbo->boxMin = to_glm_vec3(boxMin);
    volumeUbo->boxMax = to_glm_vec3(boxMax);
    volumeUbo->invMaxDensity = 1/maxDensity;
    updateCamera();

    auto scale = to_glm_vec3(size);
    auto max = glm::max(scale.x, glm::max(scale.y, scale.z));
    sliceRenderer.constants.scale = scale/max;

    light.position = (volumeUbo->boxMax + volumeUbo->boxMin) * .5f;
    light.position.y += size.y() * 0.01f;
    light.scale = max * 0.01f;

    updateVolumeDescriptorSets();
    spdlog::info("loaded volume {} with dimensions [[{}], [{}]]", grid->getName(), volumeUbo->boxMin, volumeUbo->boxMax);
}

int main(){
    try{

        Settings settings;
        settings.depthTest = true;
        settings.enabledFeatures.geometryShader = VK_TRUE;
        settings.enabledFeatures.wideLines = VK_TRUE;

        std::unique_ptr<Plugin> plugin = std::make_unique<ImGuiPlugin>();

        auto app = OpenVdbViewer{ settings };
        app.addPlugin(plugin);
        app.run();
    }catch(std::runtime_error& err){
        spdlog::error(err.what());
    }
}