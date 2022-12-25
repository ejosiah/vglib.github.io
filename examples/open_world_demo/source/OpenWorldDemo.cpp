#include "OpenWorldDemo.hpp"
#include "GraphicsPipelineBuilder.hpp"
#include "DescriptorSetBuilder.hpp"
#include "ImGuiPlugin.hpp"

OpenWorldDemo::OpenWorldDemo(const Settings& settings) : VulkanBaseApp("Open World Demo", settings) {
    fileManager.addSearchPathFront(".");
    fileManager.addSearchPathFront("../../examples/open_world_demo");
    fileManager.addSearchPathFront("../../examples/open_world_demo/data");
    fileManager.addSearchPathFront("../../examples/open_world_demo/spv");
    fileManager.addSearchPathFront("../../examples/open_world_demo/models");
    fileManager.addSearchPathFront("../../examples/open_world_demo/textures");
}

void OpenWorldDemo::initApp() {
    createDescriptorPool();
    initCamera();
    createDescriptorSetLayouts();
    updateDescriptorSets();
    createCommandPool();
    createPipelineCache();
    createRenderPipeline();
    createComputePipeline();
    terrain = std::make_unique<Terrain>(device, descriptorPool, fileManager, swapChain.width(), swapChain.height(), renderPass);
    skyDome = std::make_unique<SkyDome>(device, descriptorPool, fileManager, renderPass, swapChain.width(), swapChain.height());
}

void OpenWorldDemo::initCamera() {
    FirstPersonSpectatorCameraSettings cameraSettings;
    cameraSettings.fieldOfView = sceneData.fieldOfView;
    cameraSettings.aspectRatio = float(width)/float(height);
    cameraSettings.zNear = sceneData.zNear;
    cameraSettings.zFar = sceneData.zFar;
    cameraSettings.acceleration = glm::vec3(60 * km);
    cameraSettings.velocity = glm::vec3(200 * km);
    camera = std::make_unique<FirstPersonCameraController>(dynamic_cast<InputManager&>(*this), cameraSettings);
    camera->lookAt(glm::vec3(-28 * km, 1.849 * km, 14 * km), glm::vec3(0, 0, 0), {0, 1, 0});
//    camera->lookAt(glm::vec3(-3.4 * km, 1.2 * km, 13 * km), glm::vec3(0, 0, 0), {0, 1, 0});

//    FirstPersonSpectatorCameraSettings cameraSettings;
//    cameraSettings.fieldOfView = 90.0f;
//    cameraSettings.aspectRatio = float(width)/float(height);
//    cameraSettings.zNear = 0.01;
//    cameraSettings.zFar = 100000;
//    camera = std::make_unique<FirstPersonCameraController>(dynamic_cast<InputManager&>(*this), cameraSettings);
//    camera->lookAt(glm::vec3(0, 0, 3), glm::vec3(0, 0, 0), {0, 1, 0});
}


void OpenWorldDemo::createDescriptorPool() {
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

void OpenWorldDemo::createDescriptorSetLayouts() {
}

void OpenWorldDemo::updateDescriptorSets(){
}

void OpenWorldDemo::createCommandPool() {
    commandPool = device.createCommandPool(*device.queueFamilyIndex.graphics, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    commandBuffers = commandPool.allocateCommandBuffers(swapChainImageCount);
}

void OpenWorldDemo::createPipelineCache() {
    pipelineCache = device.createPipelineCache();
}


void OpenWorldDemo::createRenderPipeline() {
    //    @formatter:off
    auto builder = device.graphicsPipelineBuilder();
    render.pipeline =
        builder
            .shaderStage()
                .vertexShader("../../data/shaders/pass_through.vert.spv")
                .fragmentShader("../../data/shaders/pass_through.frag.spv")
            .vertexInputState()
                .addVertexBindingDescriptions(Vertex::bindingDisc())
                .addVertexAttributeDescriptions(Vertex::attributeDisc())
            .inputAssemblyState()
                .triangles()
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
                .renderPass(renderPass)
                .subpass(0)
                .name("render")
                .pipelineCache(pipelineCache)
            .build(render.layout);
    //    @formatter:on
}

void OpenWorldDemo::createComputePipeline() {
    auto module = VulkanShaderModule{ "../../data/shaders/pass_through.comp.spv", device};
    auto stage = initializers::shaderStage({ module, VK_SHADER_STAGE_COMPUTE_BIT});

    compute.layout = device.createPipelineLayout();

    auto computeCreateInfo = initializers::computePipelineCreateInfo();
    computeCreateInfo.stage = stage;
    computeCreateInfo.layout = compute.layout;

    compute.pipeline = device.createComputePipeline(computeCreateInfo, pipelineCache);
}


void OpenWorldDemo::onSwapChainDispose() {
    dispose(render.pipeline);
    dispose(compute.pipeline);
}

void OpenWorldDemo::onSwapChainRecreation() {
    updateDescriptorSets();
    createRenderPipeline();
    createComputePipeline();
    terrain->resize(renderPass, width, height);
    skyDome->resize(renderPass, width, height);
}

VkCommandBuffer *OpenWorldDemo::buildCommandBuffers(uint32_t imageIndex, uint32_t &numCommandBuffers) {
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

    skyDome->render(commandBuffer);
    terrain->render(commandBuffer);
    renderUI(commandBuffer);
    vkCmdEndRenderPass(commandBuffer);

    vkEndCommandBuffer(commandBuffer);

    return &commandBuffer;
}

void OpenWorldDemo::renderUI(VkCommandBuffer commandBuffer) {
    ImGui::Begin("Open world");
    ImGui::SetWindowSize({0, 0});

    ImGui::Text("Camera:");
    ImGui::Indent(16);
    ImGui::SliderFloat("FOV", &sceneData.fieldOfView, 5, 120);
    ImGui::Indent(-16);

    ImGui::Text("Sun:");
    ImGui::Indent(16);
    ImGui::SliderFloat("Sun Azimuth", &sceneData.sun.azimuth, 0, 360);
    ImGui::SliderFloat("Sun Elevation", &sceneData.sun.elevation, 0, 360);
    ImGui::Indent(-16);

    static bool showTerrain = false;
    ImGui::Text("Systems:");
    ImGui::Indent(16);
    ImGui::Checkbox("Terrain", &showTerrain);
    ImGui::Indent(-16);

    ImGui::End();

    if(showTerrain){
        terrain->renderUI();
    }

    plugin(IM_GUI_PLUGIN).draw(commandBuffer);
}

void OpenWorldDemo::update(float time) {
    if(!ImGui::IsAnyItemActive()) {
        camera->update(time);
    }
    camera->fieldOfView(sceneData.fieldOfView);
    auto cam = camera->cam();

    updateScene(time);
    terrain->update(sceneData);
    skyDome->update(sceneData);
    auto msg = fmt::format("{} - FPS {}, position: {}", title, framePerSecond, sceneData.eyes);
    glfwSetWindowTitle(window, msg.c_str());
}

void OpenWorldDemo::updateScene(float time) {
    sceneData.camera = camera->cam();
    sceneData.eyes = camera->position();
    sceneData.time = time;

    auto theta = glm::radians(sceneData.sun.azimuth);
    auto phi = glm::radians(sceneData.sun.elevation);

    auto axis = glm::angleAxis(phi, glm::vec3{0, 0, 1}) * glm::angleAxis(theta, glm::vec3(0, 1, 0));
    sceneData.sun.position = glm::normalize(axis)  * glm::vec3(SUN_DISTANCE, 0, 0);

}

void OpenWorldDemo::checkAppInputs() {
    camera->processInput();
}

void OpenWorldDemo::cleanup() {
    VulkanBaseApp::cleanup();
}

void OpenWorldDemo::onPause() {
    VulkanBaseApp::onPause();
}


int main(){
    try{

        Settings settings;
        settings.fullscreen = true;
        settings.screen = 1;
        settings.depthTest = true;
        settings.enabledFeatures.tessellationShader = VK_TRUE;
        settings.enabledFeatures.fillModeNonSolid = VK_TRUE;
        settings.enabledFeatures.geometryShader = VK_TRUE;

        auto app = OpenWorldDemo{ settings };
        std::unique_ptr<Plugin> imGui = std::make_unique<ImGuiPlugin>();
        app.addPlugin(imGui);
        app.run();
    }catch(std::runtime_error& err){
        spdlog::error(err.what());
    }
}