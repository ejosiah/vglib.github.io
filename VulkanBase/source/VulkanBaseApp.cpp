//
// Created by Josiah on 1/17/2021.
//
#ifndef VMA_IMPLEMENTATION
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
#endif
#include <set>
#include <chrono>
#include <fstream>
#include <VulkanShaderModule.h>

#include "VulkanBaseApp.h"
#include "keys.h"
#include "events.h"
#include "VulkanInitializers.h"
#include "Plugin.hpp"
#include "VulkanRayQuerySupport.hpp"
#include "gpu/algorithm.h"

namespace chrono = std::chrono;

const std::string VulkanBaseApp::kAttachment_BACK =  "BACK_BUFFER_INDEX";
const std::string VulkanBaseApp::kAttachment_MSAA =  "MSAA_BUFFER_INDEX";
const std::string VulkanBaseApp::kAttachment_DEPTH = "DEPTH_BUFFER_INDEX";

VulkanBaseApp::VulkanBaseApp(std::string_view name, const Settings& settings, std::vector<std::unique_ptr<Plugin>> plugins)
        : Window(name, settings.width, settings.height, settings.fullscreen, settings.screen)
        , InputManager(settings.relativeMouseMode)
        , enabledFeatures(settings.enabledFeatures)
        , settings(settings)
        , plugins(std::move(plugins))
{
    appInstance = this;
    this->settings.deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    this->settings.deviceExtensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    fileManager.addSearchPath("../../data/shaders");
    fileManager.addSearchPath("../../data/models");
    fileManager.addSearchPath("../../data/textures");
    fileManager.addSearchPath("../../data");
}

VulkanBaseApp::~VulkanBaseApp(){
    ready = false;
    appInstance = nullptr;
}

void VulkanBaseApp::init() {
    checkInstanceExtensionSupport();
    initWindow();
    initInputMgr(*this);
    exit = &mapToKey(Key::ESCAPE, "Exit", Action::detectInitialPressOnly());
    pause = &mapToKey(Key::P, "Pause", Action::detectInitialPressOnly());
    addPluginExtensions();
    initVulkan();
    postVulkanInit();

    gpu::init(device, fileManager);

    createSwapChain();
    createSyncObjects();
    swapChainReady();

    createColorBuffer();
    createDepthBuffer();
    createRenderPass();
    createFramebuffer();
    framebufferReady();

    initPlugins();
    prototypes = std::make_unique<Prototypes>( device, swapChain, renderPass);
    initApp();
    ready = true;
}

void VulkanBaseApp::initMixins() {
    if(auto rayQuery = dynamic_cast<VulkanRayQuerySupport*>(this)){
        rayQuery->enableRayQuery();
    }
}

void VulkanBaseApp::initWindow() {
    Window::initWindow();
    uint32_t size;
    auto extensions = glfwGetRequiredInstanceExtensions(&size);
    instanceExtensions = std::vector<const char*>(extensions, extensions + size);

    for(auto& extension : settings.instanceExtensions){
        instanceExtensions.push_back(extension);
    }

    for(auto& extension : settings.deviceExtensions){
        deviceExtensions.push_back(extension);
    }

    for(auto& layer : settings.validationLayers){
        validationLayers.push_back(layer);
    }

    if(enableValidation){
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        instanceExtensions.push_back(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
        validationLayers.push_back("VK_LAYER_KHRONOS_validation");
    }
}

void VulkanBaseApp::initVulkan() {
    createInstance();
    this->ext = VulkanExtensions{instance};
    ext::init(instance);
    createDebugMessenger();
    pickPhysicalDevice();
    initMixins();
    createLogicalDevice();
}

void VulkanBaseApp::postVulkanInit() {}

void VulkanBaseApp::framebufferReady() {}

void VulkanBaseApp::swapChainReady() {}

void VulkanBaseApp::addPluginExtensions() {
    for(auto& plugin : plugins){
        for(auto extension : plugin->instanceExtensions()){
            instanceExtensions.push_back(extension);
        }
        for(auto layer : plugin->validationLayers()){
            validationLayers.push_back(layer);
        }
        for(auto extension : plugin->deviceExtensions()){
            deviceExtensions.push_back(extension);
        }
    }
}

void VulkanBaseApp::initPlugins() {

    for(auto& plugin : plugins){
        spdlog::info("initializing plugin: {}", plugin->name());
        plugin->set({ &instance, &device, &renderPass, &swapChain, window, &currentImageIndex, settings.msaaSamples});
        plugin->init();
        registerPluginEventListeners(plugin.get());
    }
}

void VulkanBaseApp::createInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType  = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 0, 0);
    appInfo.pApplicationName = title.data();
    appInfo.apiVersion = VK_API_VERSION_1_3;
    appInfo.pEngineName = "";

    instance = VulkanInstance{appInfo, {instanceExtensions, validationLayers}};
}

void VulkanBaseApp::createSwapChain() {
    glfwGetFramebufferSize(window, &width, &height);    // TODO use settings and remove width/height
    settings.width = width;
    settings.height = height;
    swapChain = VulkanSwapChain{ device, surface, settings};
    swapChainImageCount = swapChain.imageCount();
}

void VulkanBaseApp::createDepthBuffer() {
    if(!settings.depthTest) return;

    auto format = findDepthFormat();
    VkImageCreateInfo createInfo = initializers::imageCreateInfo(
            VK_IMAGE_TYPE_2D,
            format,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            swapChain.extent.width,
            swapChain.extent.height);
    createInfo.samples = settings.msaaSamples;

    depthBuffer.image = device.createImage(createInfo, VMA_MEMORY_USAGE_GPU_ONLY);

    VkImageSubresourceRange subresourceRange = initializers::imageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT);
    depthBuffer.imageView = depthBuffer.image.createView(format, VK_IMAGE_VIEW_TYPE_2D, subresourceRange);
    depthBuffer.width = width;
    depthBuffer.height = height;
    depthBuffer.format = format;
    auto byteSize = 0.f;
    if(format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D24_UNORM_S8_UINT){
        byteSize = 4;
    }else if(format == VK_FORMAT_D32_SFLOAT_S8_UINT) {
        byteSize = 5;
    }
    depthBuffer.image.size = width * height * byteSize;
}

void VulkanBaseApp::createColorBuffer(){
    if(settings.msaaSamples == VK_SAMPLE_COUNT_1_BIT) return;

    VkImageCreateInfo createInfo = initializers::imageCreateInfo(
            VK_IMAGE_TYPE_2D,
            swapChain.format,
            VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            swapChain.extent.width,
            swapChain.extent.height);
    createInfo.samples = settings.msaaSamples;

    colorBuffer.image = device.createImage(createInfo, VMA_MEMORY_USAGE_GPU_ONLY);
    VkImageSubresourceRange subresourceRange = initializers::imageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT);
    colorBuffer.imageView = colorBuffer.image.createView(swapChain.format, VK_IMAGE_VIEW_TYPE_2D, subresourceRange);
    colorBuffer.width = swapChain.extent.width;
    colorBuffer.height = swapChain.extent.height;
    colorBuffer.format = swapChain.format;
}

VkFormat VulkanBaseApp::findDepthFormat() {
    auto formats = depthFormats.formats;

    if(settings.stencilTest){
        std::reverse(formats.begin(), formats.end());
    }

    auto possibleFormat = device.findSupportedFormat(formats, depthFormats.tiling, depthFormats.features);
    if(!possibleFormat.has_value()){
        throw std::runtime_error{"Failed to find a suitable depth format"};
    }
    spdlog::info("App will be using depth buffer with: format: {}", *possibleFormat);
    return *possibleFormat;
}

void VulkanBaseApp::createLogicalDevice() {
    device.createLogicalDevice(enabledFeatures, deviceExtensions, validationLayers, surface, settings.queueFlags, deviceCreateNextChain);
}

void VulkanBaseApp::pickPhysicalDevice() {
    surface = VulkanSurface{instance, window};
    auto pDevices = enumerate<VkPhysicalDevice>([&](uint32_t* size, VkPhysicalDevice* pDevice){
        return vkEnumeratePhysicalDevices(instance, size, pDevice);
    });

    std::vector<VulkanDevice> devices(pDevices.size());
    std::transform(begin(pDevices), end(pDevices), begin(devices),[&](auto pDevice){
        return VulkanDevice{instance, pDevice, settings};
    });

    std::sort(begin(devices), end(devices), [](auto& a, auto& b){
        return a.score() > b.score();
    });

    device = std::move(devices.front());
    settings.msaaSamples = std::min(settings.msaaSamples, device.getMaxUsableSampleCount());
    checkDeviceExtensionSupport();
    spdlog::info("selected device: {}", device.name());
}

void VulkanBaseApp::addPlugin(std::unique_ptr<Plugin>& plugin) {
    plugins.push_back(std::move(plugin));
}

void VulkanBaseApp::run() {
    init();
    mainLoop();
    cleanupPlugins();
    cleanup0();
}

void VulkanBaseApp::mainLoop() {
    while(isRunning()){
        recenter();
        glfwPollEvents();
        fullscreenCheck();

        if(swapChainInvalidated || swapChain.isOutOfDate()){
            swapChainInvalidated = false;
            recreateSwapChain();
        }

        checkSystemInputs();
        if(!isRunning()) break;

        if(!paused) {
            checkAppInputs();
            notifyPluginsOfNewFrameStart();
            waitForNextFrame();
            newFrame();
            drawFrame();
            presentFrame();
            notifyPluginsOfEndFrame();
            processIdleProcs();
            endFrame();
            nextFrame();
        }else{
            glfwSetTime(elapsedTime);
            onPause();
        }
    }

    vkDeviceWaitIdle(device);
}

void VulkanBaseApp::checkSystemInputs() {
    if(exit->isPressed()){
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    if(pause->isPressed()){
        setPaused(!paused);
    }
}

void VulkanBaseApp::createDebugMessenger() {
#ifdef DEBUG_MODE
    vulkanDebug = VulkanDebug{ instance };
#endif
}

void VulkanBaseApp::createFramebuffer() {
    assert(renderPass.renderPass != VK_NULL_HANDLE);

    framebuffers.resize(swapChain.imageCount());
    auto numAttachments = numFrameBufferAttachments;
    for(int i = 0; i < framebuffers.size(); i++){
        std::vector<VkImageView> attachments(numAttachments);
        attachments[attachmentIndices[kAttachment_BACK]] = swapChain.imageViews[i];
        if(settings.depthTest){
            assert(depthBuffer.imageView.handle != VK_NULL_HANDLE);
            attachments[attachmentIndices[kAttachment_DEPTH]] = depthBuffer.imageView.handle;
        }
        if(settings.msaaSamples != VK_SAMPLE_COUNT_1_BIT){
            attachments[attachmentIndices[kAttachment_MSAA]] = colorBuffer.imageView.handle;
        }
        framebuffers[i] = device.createFramebuffer(renderPass, attachments
                                               , static_cast<uint32_t>(width), static_cast<uint32_t>(height) );
    }
}


void VulkanBaseApp::createRenderPass() {
    auto [attachments, subpassDesc, dependency] = buildRenderPass();    // TODO use default if empty
    renderPass = device.createRenderPass(attachments, subpassDesc, dependency);
}

// TODO make this private
RenderPassInfo VulkanBaseApp::buildRenderPass() {
    bool msaaEnabled = settings.msaaSamples != VK_SAMPLE_COUNT_1_BIT;
    VkAttachmentDescription attachmentDesc{};
    attachmentDesc.format = swapChain.format;
    attachmentDesc.samples = settings.msaaSamples;
    attachmentDesc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachmentDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachmentDesc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachmentDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachmentDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachmentDesc.finalLayout = msaaEnabled ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;


    std::vector<VkAttachmentDescription> attachments;
    VkAttachmentReference ref{};
    ref.attachment = attachments.size();
    ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments.push_back(attachmentDesc);
    if(msaaEnabled){
        attachmentIndices[kAttachment_MSAA] = ref.attachment;
    }else{
        attachmentIndices[kAttachment_BACK] = ref.attachment;
    }


    SubpassDescription subpassDesc{};
    subpassDesc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDesc.colorAttachments.push_back(ref);

    if(settings.depthTest){
        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = depthBuffer.image.format;
        depthAttachment.samples = settings.msaaSamples;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp =  VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.stencilLoadOp = settings.stencilTest ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.stencilStoreOp = settings.stencilTest ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachments.push_back(depthAttachment);

        VkAttachmentReference depthRef{};
        depthRef.attachment = attachments.size() - 1;
        depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachmentIndices[kAttachment_DEPTH] = depthRef.attachment;
        subpassDesc.depthStencilAttachments = depthRef;
    }

    if(msaaEnabled){
        VkAttachmentDescription colorAttachmentResolve{};
        colorAttachmentResolve.format = swapChain.format;
        colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        attachments.push_back(colorAttachmentResolve);

        VkAttachmentReference resolveRef{};
        resolveRef.attachment = attachments.size() - 1;
        resolveRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        subpassDesc.resolveAttachments.push_back(resolveRef);
        attachmentIndices[kAttachment_BACK] = attachments.size() - 1;
    }


    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    std::vector<SubpassDescription> subpassDescs{ subpassDesc };
    std::vector<VkSubpassDependency> dependencies{ dependency };

    numFrameBufferAttachments = attachments.size();

    return std::make_tuple(attachments, subpassDescs, dependencies);
}

void VulkanBaseApp::createSyncObjects() {
    imageAcquired.resize(MAX_IN_FLIGHT_FRAMES);
    renderingFinished.resize(MAX_IN_FLIGHT_FRAMES);
    inFlightFences.resize(MAX_IN_FLIGHT_FRAMES);
    inFlightImages.resize(swapChain.imageCount());

    VkSemaphoreCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for(auto i = 0; i < MAX_IN_FLIGHT_FRAMES; i++){
        imageAcquired[i] = device.createSemaphore();
        renderingFinished[i] = device.createSemaphore();
        inFlightFences[i] = device.createFence();
    }
}

void VulkanBaseApp::waitForNextFrame(){
    if(swapChainInvalidated) return;
    inFlightFences[currentFrame].wait();
    currentImageIndex = swapChain.acquireNextImage(imageAcquired[currentFrame]);
}

void VulkanBaseApp::drawFrame() {
    frameCount++;

    auto imageIndex = currentImageIndex;

    if(inFlightImages[imageIndex]){
        inFlightImages[imageIndex]->wait();
    }
    inFlightImages[imageIndex] = &inFlightFences[currentFrame];

    auto time = getTime();
    updatePlugins(time);
    update(time);
    calculateFPS(time);

    static std::vector<VkPipelineStageFlags> waitStages_;
    static std::vector<VkSemaphore> waitSemaphores_;
    static std::vector<VkSemaphore> signalSemaphores_;

    waitStages_.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    waitSemaphores_.push_back(imageAcquired[currentFrame].semaphore);
    signalSemaphores_.push_back(renderingFinished[currentFrame].semaphore);

    if(!waitSemaphores.empty()) {
        for(int i = 0; i < waitSemaphores.size(); i++){
            const auto& stages = waitStages[i];
            const auto& semaphores = waitSemaphores[i];
            ASSERT(semaphores.size() == swapChainImageCount);
            waitStages_.push_back(stages[imageIndex]);
            waitSemaphores_.push_back(semaphores[imageIndex]);
        }
    }

    if(!signalSemaphores.empty()) {
        for(int i = 0; i < waitSemaphores.size(); i++){
            const auto& semaphores = signalSemaphores[i];
            ASSERT(semaphores.size() == swapChainImageCount);
            signalSemaphores_.push_back(semaphores[imageIndex]);
        }
    }

    uint32_t commandBufferCount;
    auto commandBuffers = buildCommandBuffers(imageIndex, commandBufferCount);


    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.pNext = queueSubmitNextChain;
    submitInfo.waitSemaphoreCount = COUNT(waitSemaphores_);
    submitInfo.pWaitSemaphores = waitSemaphores_.data();
    submitInfo.pWaitDstStageMask = waitStages_.data();
    submitInfo.commandBufferCount = commandBufferCount;
    submitInfo.pCommandBuffers = commandBuffers;
    submitInfo.signalSemaphoreCount = COUNT(signalSemaphores_);
    submitInfo.pSignalSemaphores = signalSemaphores_.data();

    inFlightFences[currentFrame].reset();

    ERR_GUARD_VULKAN(vkQueueSubmit(device.queues.graphics, 1, &submitInfo, inFlightFences[currentFrame]));

    waitStages_.clear();
    waitSemaphores_.clear();
    signalSemaphores_.clear();
}

void VulkanBaseApp::presentFrame() {
    if(swapChainInvalidated) return;

    swapChain.present(currentImageIndex, { renderingFinished[currentFrame] });
    if(swapChain.isSubOptimal() || swapChain.isOutOfDate() || resized) {
        resized = false;
        swapChainInvalidated = true;
        return;
    }
}

void VulkanBaseApp::nextFrame() {
    mouse.left.released = mouse.middle.released = mouse.right.released = false;
    currentFrame = (currentFrame + 1)%MAX_IN_FLIGHT_FRAMES;
}

void VulkanBaseApp::calculateFPS(float dt) {
    static float oneSecond = 0.f;

    oneSecond += dt;
    if(oneSecond > 1.0f){
        framePerSecond = frameCount;
        totalFrames += frameCount;
        frameCount = 0;
        oneSecond = 0.f;
    }
}

void VulkanBaseApp::recreateSwapChain() {
    do{
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }while(width == 0 && height == 0);

    vkDeviceWaitIdle(device);
    cleanupSwapChain();

    createSwapChain();

    if(settings.depthTest){
        createDepthBuffer();
    }
    if(settings.msaaSamples != VK_SAMPLE_COUNT_1_BIT){
        createColorBuffer();
    }
    createRenderPass();
    createFramebuffer();

    notifyPluginsOfSwapChainRecreation();
    onSwapChainRecreation();
}



void VulkanBaseApp::update(float time) {

}

void VulkanBaseApp::cleanupSwapChain() {
    notifyPluginsOfSwapChainDisposal();
    onSwapChainDispose();

    for(auto& framebuffer : framebuffers){
        dispose(framebuffer);
    }
    dispose(renderPass);

    if(settings.depthTest){
        dispose(depthBuffer.image);
        dispose(depthBuffer.imageView);
    }
    if(settings.msaaSamples != VK_SAMPLE_COUNT_1_BIT){
        dispose(colorBuffer.image);
        dispose(colorBuffer.imageView);
    }

    dispose(swapChain);
}

void VulkanBaseApp::setPaused(bool flag) {
    if(paused != flag){
        paused = flag;
        resetAllActions();
    }
}

float VulkanBaseApp::getTime() {
    auto now = static_cast<float>(glfwGetTime());
    auto dt = now - elapsedTime;
    elapsedTime += dt;
    return dt;
}

void VulkanBaseApp::onSwapChainRecreation() {

}

void VulkanBaseApp::onSwapChainDispose() {

}

inline void VulkanBaseApp::checkAppInputs() {
}

inline bool VulkanBaseApp::isRunning() const {
    return !glfwWindowShouldClose(window);
}

void VulkanBaseApp::cleanup() {

}

void VulkanBaseApp::onPause() {

}

void VulkanBaseApp::newFrame() {

}

void VulkanBaseApp::endFrame() {

}

void VulkanBaseApp::copyToSwapChain(VkCommandBuffer commandBuffer, VkImage srcImage, int swapChainImageIndex) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.image = swapChain.getImage(swapChainImageIndex);
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;



    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,

                         0,
                         VK_NULL_HANDLE,
                         0,
                         VK_NULL_HANDLE,
                         1,
                         &barrier);

    VkImageCopy copy{};
    copy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.srcSubresource.mipLevel = 0;
    copy.srcSubresource.baseArrayLayer = 0;
    copy.srcSubresource.layerCount = 1;
    copy.srcOffset = {0, 0, 0};

    copy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.dstSubresource.mipLevel = 0;
    copy.dstSubresource.baseArrayLayer = 0;
    copy.dstSubresource.layerCount = 1;
    copy.dstOffset = {0, 0, 0};
    copy.extent = {swapChain.width(), swapChain.height(), 1};

    vkCmdCopyImage(commandBuffer, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
            , swapChain.getImage(swapChainImageIndex), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         0,

                         0,
                         VK_NULL_HANDLE,
                         0,
                         VK_NULL_HANDLE,
                         1,
                         &barrier);
}

void VulkanBaseApp::notifyPluginsOfNewFrameStart() {
    for(auto& plugin : plugins){
        plugin->newFrame();
    }
}

void VulkanBaseApp::notifyPluginsOfEndFrame() {
    for(auto& plugin : plugins){
        plugin->endFrame();
    }
}

void VulkanBaseApp::notifyPluginsOfSwapChainDisposal() {
    for(auto& plugin : plugins){
        plugin->onSwapChainDispose();
    }
}

void VulkanBaseApp::notifyPluginsOfSwapChainRecreation() {
    for(auto& plugin : plugins){
        plugin->onSwapChainRecreation();
    }
}

void VulkanBaseApp::cleanupPlugins() {
    for(auto& plugin : plugins){
        plugin->cleanup();
    }
}

void VulkanBaseApp::registerPluginEventListeners(Plugin* plugin) {
    addWindowResizeListeners(plugin->windowResizeListener());
    addMousePressListener(plugin->mousePressListener());
    addMouseReleaseListener(plugin->mouseReleaseListener());
    addMouseClickListener(plugin->mouseClickListener());
    addMouseMoveListener(plugin->mouseMoveListener());
    addMouseWheelMoveListener(plugin->mouseWheelMoveListener());
    addKeyPressListener(plugin->keyPressListener());
    addKeyReleaseListener(plugin->keyReleaseListener());
}

void VulkanBaseApp::updatePlugins(float dt){
    for(auto& plugin : plugins){
        plugin->update(dt);
    }
}

VulkanBaseApp* VulkanBaseApp::appInstance = nullptr;

void VulkanBaseApp::fullscreenCheck() {
    if(toggleFullscreen && !fullscreen){
        toggleFullscreen = false;
        swapChainInvalidated = setFullScreen();

    }else if(toggleFullscreen && fullscreen){
        toggleFullscreen = false;
        swapChainInvalidated = unsetFullScreen();
    }
}

void VulkanBaseApp::checkDeviceExtensionSupport() {
    std::vector<const char*> unsupported;
    for(auto extension : deviceExtensions){
        if(!device.extensionSupported(extension)){
            unsupported.push_back(extension);
        }
    }

    if(!unsupported.empty()){
        throw std::runtime_error{
            fmt::format("Vulkan device [{}] does not support the following extensions {}", device.name(), unsupported) };
    }
}

void VulkanBaseApp::checkInstanceExtensionSupport() {
    auto supportedExtensions = enumerate<VkExtensionProperties>([](auto size, auto properties){
        return vkEnumerateInstanceExtensionProperties("", size, properties);
    });

    std::vector<const char*> unsupported;
    for(auto& extension : instanceExtensions){
        auto supported = std::any_of(begin(supportedExtensions), end(supportedExtensions), [&](auto supported){
            return std::strcmp(extension, supported.extensionName) == 0;
        });
        if(supported){
            unsupported.push_back(extension);
        }
    }

    if(!unsupported.empty()){
        throw std::runtime_error{fmt::format("this Vulkan instance does not support the following extensions {}", unsupported) };
    }
}

byte_string VulkanBaseApp::load(const std::string &resource) {
    return fileManager.load(resource);
}

std::string VulkanBaseApp::resource(const std::string& name) {
    auto res = fileManager.getFullPath(name);
    assert(res.has_value());
    return res->string();
}

Entity VulkanBaseApp::createEntity(const std::string &name) {
    Entity entity{m_registry };
    entity.add<component::Position>();
    entity.add<component::Rotation>();
    entity.add<component::Scale>();
    entity.add<component::Transform>();
    auto& nameTag = entity.add<component::Name>();
    nameTag.value = name.empty() ? fmt::format("{}_{}", "Entity", m_registry.size()) : name;
    return entity;
}

void VulkanBaseApp::updateEntityTransforms(entt::registry& registry) {
    auto view = registry.view<component::Position, component::Rotation, component::Scale, component::Transform>();

    for(auto entity : view){
        auto& position = view.get<component::Position>(entity);
        auto& scale = view.get<component::Scale>(entity);
        auto& rotation = view.get<component::Rotation>(entity);
        auto& transform = view.get<component::Transform>(entity);
        auto localTransform = glm::translate(glm::mat4(1), position.value) * glm::mat4(rotation.value) * glm::scale(glm::mat4(1), scale.value);

        transform.value = transform.parent ? transform.parent->value * localTransform : localTransform;
    }
}


void VulkanBaseApp::destroyEntity(Entity entity) {
    m_registry.destroy(entity);
}

glm::vec3 VulkanBaseApp::mousePositionToWorldSpace(const Camera &camera) {
    auto mousePos = glm::vec3(mouse.position, 1);
    glm::vec4 viewport{0, 0, swapChain.width(), swapChain.height()};
    return glm::unProject(mousePos, camera.view, camera.proj, viewport);
}

void VulkanBaseApp::renderEntities(VkCommandBuffer commandBuffer, entt::registry& registry) {
    auto camView = registry.view<const component::Camera>();

    Camera* camera{nullptr};
    for(auto entity : camView){
        auto cam = camView.get<const component::Camera>(entity);
        if(cam.main){
            camera = cam.camera;
            break;
        }
    }
    if(!camera){
        spdlog::error("no camera entity set");
    }
    assert(camera);

    auto view = registry.view<const component::Render, const component::Transform,  const component::Pipelines>();
    static std::vector<VkBuffer> buffers;
    view.each([&](const component::Render& renderComp, const auto& transform,  const auto& pipelines){
        if(renderComp.instanceCount > 0) {
            auto model = transform.value;
            camera->model = model;
            std::vector<VkDeviceSize> offsets(renderComp.vertexBuffers.size(), 0);
            buffers.clear();
            for(auto& buffer : renderComp.vertexBuffers){
                buffers.push_back(buffer.buffer);
            }
            vkCmdBindVertexBuffers(commandBuffer, 0, COUNT(buffers), buffers.data(), offsets.data());
            if (renderComp.indexCount > 0) {
                vkCmdBindIndexBuffer(commandBuffer, renderComp.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            }
            for (const auto &pipeline : pipelines) {
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,  (VkPipeline)pipeline.pipeline);
                vkCmdPushConstants(commandBuffer, (VkPipelineLayout)pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Camera), camera);
                if (!pipeline.descriptorSets.empty()) {
                    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, (VkPipelineLayout)pipeline.layout, 0,
                                            COUNT(pipeline.descriptorSets), (VkDescriptorSet*)pipeline.descriptorSets.data(), 0,
                                            VK_NULL_HANDLE);
                }
                for (auto primitive : renderComp.primitives) {
                    if (renderComp.indexCount > 0) {
                        primitive.drawIndexed(commandBuffer, 0, renderComp.instanceCount);
                    } else {
                        primitive.draw(commandBuffer, 0, renderComp.instanceCount);
                    }
                }
            }
        }
    });
}

inline InputManager &VulkanBaseApp::inputManager() {
    return dynamic_cast<InputManager&>(*this);
}


void VulkanBaseApp::onIdle(Proc &&proc) {
    idleProcs.push_back(proc);
}

void VulkanBaseApp::processIdleProcs() {
    while(!idleProcs.empty()){
        auto proc = idleProcs.front();
        proc();
        idleProcs.pop_front();
    }
}

void VulkanBaseApp::runInBackground(Proc &&proc) {
    threadPool.async(proc);
}

void VulkanBaseApp::cleanup0() {
    cleanup();
    gpu::shutdown();
}

void VulkanBaseApp::addBufferMemoryBarriers(VkCommandBuffer commandBuffer, const std::vector<VulkanBuffer> &buffers
                                            ,VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask) {
    std::vector<VkBufferMemoryBarrier> barriers(buffers.size());

    for(int i = 0; i < buffers.size(); i++) {
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barriers[i].offset = 0;
        barriers[i].buffer = buffers[i];
        barriers[i].size = buffers[i].size;
    }

    vkCmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, 0, 0,nullptr
                         , COUNT(barriers), barriers.data(), 0, nullptr);
}

void VulkanBaseApp::addImageMemoryBarriers(VkCommandBuffer commandBuffer, const std::vector<std::reference_wrapper<VulkanImage>> &images,
                                           VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask) {
    std::vector<VkImageMemoryBarrier> barriers(images.size());

    for(int i = 0; i < images.size(); i++) {
        barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; // TODO add as param
        barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;  // TODO add as param
        barriers[i].oldLayout = images[i].get().currentLayout;
        barriers[i].newLayout = images[i].get().currentLayout;
        barriers[i].image = images[i].get();
        barriers[i].subresourceRange = DEFAULT_SUB_RANGE;
    }

    vkCmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, 0, 0,nullptr
            , 0, nullptr, COUNT(barriers), barriers.data());
}

void VulkanBaseApp::addMemoryBarrier(VkCommandBuffer commandBuffer, VkPipelineStageFlags srcStageMask
                                     , VkPipelineStageFlags dstStageMask, VkAccessFlags srcAccessMask
                                     , VkAccessFlags dstAccessMask) {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, VK_NULL_HANDLE, srcAccessMask, dstAccessMask};
    vkCmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, 0, 1, &barrier, 0, VK_NULL_HANDLE, 0, VK_NULL_HANDLE);
}

void VulkanBaseApp::invalidateSwapChain() {
    swapChainInvalidated = true;
}

void VulkanBaseApp::save(const FramebufferAttachment &attachment) {

}