#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <array>
#include <span>

struct ShaderInfo{
    VulkanShaderModule module;
    VkShaderStageFlagBits stage;
    const char*  entry = "main";
};

namespace initializers{
    inline std::vector<VkPipelineShaderStageCreateInfo> vertexShaderStages(const std::vector<ShaderInfo>& shaderInfos){
        std::vector<VkPipelineShaderStageCreateInfo> createInfos;

        for(auto& shaderInfo : shaderInfos){
            VkPipelineShaderStageCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            createInfo.stage = shaderInfo.stage;
            createInfo.module = shaderInfo.module.handle;
            createInfo.pName = shaderInfo.entry;

            createInfos.push_back(createInfo);
        }

        return createInfos;
    }

    inline std::vector<VkPipelineShaderStageCreateInfo> rayTraceShaderStages(const std::vector<ShaderInfo>& shaderInfos){
        return vertexShaderStages(shaderInfos);
    }

    inline VkPipelineShaderStageCreateInfo shaderStage(const ShaderInfo& shaderInfo){
        VkPipelineShaderStageCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        createInfo.stage = shaderInfo.stage;
        createInfo.module = shaderInfo.module.handle;
        createInfo.pName = shaderInfo.entry;

        return createInfo;
    }

    inline VkPipelineShaderStageCreateInfo computeShaderStage(const ShaderInfo& shaderInfo){
        VkPipelineShaderStageCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        createInfo.stage = shaderInfo.stage;
        createInfo.module = shaderInfo.module.handle;
        createInfo.pName = shaderInfo.entry;

        return createInfo;
    }

    template<typename BindingDescriptions = std::vector<VkVertexInputBindingDescription>, typename AttributeDescriptions = std::vector<VkVertexInputAttributeDescription>>
    inline VkPipelineVertexInputStateCreateInfo vertexInputState(const BindingDescriptions& bindings = {}, const AttributeDescriptions& attributes = {}){
        VkPipelineVertexInputStateCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        createInfo.vertexBindingDescriptionCount = COUNT(bindings);
        createInfo.pVertexBindingDescriptions = bindings.data();
        createInfo.vertexAttributeDescriptionCount = COUNT(attributes);
        createInfo.pVertexAttributeDescriptions = attributes.data();

        return createInfo;
    }

    static inline VkPipelineInputAssemblyStateCreateInfo inputAssemblyState(VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VkBool32 primitiveRestart = VK_FALSE){
        VkPipelineInputAssemblyStateCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        createInfo.topology = topology;
        createInfo.primitiveRestartEnable = primitiveRestart;

        return createInfo;
    }

    inline VkViewport viewport(float width, float height, float x = 0.0f, float y = 0.0f, float minDepth = 0.0f, float maxDepth = 1.0f){
        VkViewport viewport{};
        viewport.x = x;
        viewport.y = y;
        viewport.width = width;
        viewport.height = height;
        viewport.minDepth = minDepth;
        viewport.maxDepth = maxDepth;

        return viewport;
    }

    inline VkViewport viewport(const VkExtent2D& extent, float minDepth = 0.0f, float maxDepth = 1.0f){
        VkViewport viewport{};
        viewport.width = static_cast<float>(extent.width);
        viewport.height = static_cast<float>(extent.height);
        viewport.minDepth = minDepth;
        viewport.maxDepth = maxDepth;

        return viewport;
    }

    inline VkRect2D scissor(uint32_t width, uint32_t height, int32_t offsetX = 0, int32_t offsetY = 0){
        VkRect2D rect{};
        rect.extent.width = width;
        rect.extent.height = height;
        rect.offset.x = offsetX;
        rect.offset.y = offsetY;

        return rect;
    }

    inline VkRect2D scissor(VkExtent2D extent, VkOffset2D offset = {0u, 0u}){
        VkRect2D rect;
        rect.offset = offset;
        rect.extent = extent;

        return rect;
    }

    inline VkPipelineViewportStateCreateInfo viewportState(const std::vector<VkViewport>& viewports = {}, const std::vector<VkRect2D>& scissors = {}){
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = COUNT(viewports);
        viewportState.pViewports = viewports.data();
        viewportState.scissorCount = COUNT(scissors);
        viewportState.pScissors = scissors.data();

        return viewportState;
    }

    inline VkPipelineViewportStateCreateInfo viewportState(const VkViewport& viewport, const VkRect2D& scissor){
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        return viewportState;
    }

    inline VkPipelineRasterizationStateCreateInfo rasterizationState(){
        VkPipelineRasterizationStateCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        createInfo.lineWidth = 1.0f;

        return createInfo;
    }

    inline VkPipelineMultisampleStateCreateInfo multisampleState(){
        VkPipelineMultisampleStateCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        createInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        return createInfo;
    }

    inline VkPipelineDepthStencilStateCreateInfo depthStencilState(){
        VkPipelineDepthStencilStateCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        createInfo.depthTestEnable = VK_FALSE;
        createInfo.depthWriteEnable = VK_FALSE;

        return createInfo;
    }

    template<typename ColorBlendStates>
    inline VkPipelineColorBlendStateCreateInfo colorBlendState(const ColorBlendStates& blendAttachmentState = {}){
        VkPipelineColorBlendStateCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        createInfo.attachmentCount = COUNT(blendAttachmentState);
        createInfo.pAttachments = blendAttachmentState.data();

        return createInfo;
    }


    inline VkPipelineDynamicStateCreateInfo dynamicState(const std::vector<VkDynamicState>& dynamicStates  = {}) {
        VkPipelineDynamicStateCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        createInfo.dynamicStateCount = COUNT(dynamicStates);
        createInfo.pDynamicStates = dynamicStates.data();

        return createInfo;
    }

    inline VkRenderPassCreateInfo renderPassCreateInfo(
              const std::vector<VkAttachmentDescription>& attachmentDescriptions
            , const std::vector<VkSubpassDescription>& subpassDescriptions
            , const std::vector<VkSubpassDependency>& dependencies = {}){

        VkRenderPassCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        createInfo.attachmentCount = COUNT(attachmentDescriptions);
        createInfo.pAttachments = attachmentDescriptions.data();
        createInfo.subpassCount = COUNT(subpassDescriptions);
        createInfo.pSubpasses = subpassDescriptions.data();
        createInfo.dependencyCount = COUNT(dependencies);
        createInfo.pDependencies = dependencies.data();

        return createInfo;
    }

    inline VkFramebufferCreateInfo framebufferCreateInfo(
            VkRenderPass renderPass
            , const std::vector<VkImageView>& attachments
            , uint32_t width
            , uint32_t height
            , uint32_t layers = 1){

        VkFramebufferCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        createInfo.renderPass = renderPass;
        createInfo.attachmentCount = COUNT(attachments);
        createInfo.pAttachments = attachments.data();
        createInfo.width = width;
        createInfo.height = height;
        createInfo.layers = layers;

        return createInfo;
    }

    inline VkImageCreateInfo imageCreateInfo(VkImageType imageType, VkFormat format, VkImageUsageFlags usage, uint32_t width = 0u, uint32_t height = 0u, uint32_t depth = 1){
        VkImageCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        createInfo.imageType = imageType;
        createInfo.format = format;
        createInfo.extent.width = width;
        createInfo.extent.height = height;
        createInfo.extent.depth = depth;
        createInfo.mipLevels = 1;
        createInfo.arrayLayers = 1;
        createInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        createInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        createInfo.usage = usage;
        createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0;
        createInfo.pQueueFamilyIndices = nullptr;
        createInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        return createInfo;
    }

    inline VkImageViewCreateInfo imageViewCreateInfo(){
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;

        return createInfo;
    }

    inline VkSamplerCreateInfo samplerCreateInfo(){
        VkSamplerCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;

        return createInfo;
    }

    inline VkImageSubresourceRange imageSubresourceRange(VkImageAspectFlags  aspect = VK_IMAGE_ASPECT_COLOR_BIT){
        VkImageSubresourceRange subresourceRange;
        subresourceRange.aspectMask = aspect;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = 1;
        subresourceRange.baseArrayLayer = 0;
        subresourceRange.layerCount = 1;

        return subresourceRange;
    }

    inline VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo(const std::vector<VkDescriptorSetLayoutBinding>& bindings, VkDescriptorSetLayoutCreateFlags flags = 0) {
        VkDescriptorSetLayoutCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        createInfo.flags = flags;
        createInfo.bindingCount = COUNT(bindings);
        createInfo.pBindings = bindings.data();

        return createInfo;
    }

    inline VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo() {
        VkPipelineLayoutCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

        return createInfo;
    }

    inline VkGraphicsPipelineCreateInfo pipelineCreateInfo() {
        VkGraphicsPipelineCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

        return createInfo;
    }


    inline VkCommandBufferBeginInfo commandBufferBeginInfo() {
        VkCommandBufferBeginInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        return info;
    }

    inline VkRenderPassBeginInfo renderPassBeginInfo() {
        VkRenderPassBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        return beginInfo;
    }

    inline VkWriteDescriptorSet writeDescriptorSet(VkDescriptorSet descriptorSet = VK_NULL_HANDLE) {
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptorSet;
        return write;
    }

    template<size_t size = 1u>
    inline std::vector<VkWriteDescriptorSet> writeDescriptorSets(VkDescriptorSet descriptorSet = VK_NULL_HANDLE) {
        std::vector<VkWriteDescriptorSet> writes(size);
        for(auto& write : writes){
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = descriptorSet;
        }
        return writes;
    }


    constexpr VkGraphicsPipelineCreateInfo graphicsPipelineCreateInfo() {
        VkGraphicsPipelineCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

        return createInfo;
    }

    inline VkDescriptorPoolCreateInfo descriptorPoolCreateInfo(const std::vector<VkDescriptorPoolSize>& poolSizes, uint32_t maxSets, VkDescriptorPoolCreateFlags flags = 0){
        VkDescriptorPoolCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        createInfo.flags = flags;
        createInfo.poolSizeCount = COUNT(poolSizes);
        createInfo.pPoolSizes = poolSizes.data();
        createInfo.maxSets = maxSets;

        return createInfo;
    }


    inline VkComputePipelineCreateInfo computePipelineCreateInfo() {
        VkComputePipelineCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        return createInfo;
    }


    inline VkImageMemoryBarrier ImageMemoryBarrier() {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        return barrier;
    }

    template<size_t Size = 1>
    inline std::array<VkPipelineColorBlendAttachmentState, Size> colorBlendStateAttachmentStates() {
        std::array<VkPipelineColorBlendAttachmentState, Size> states{};
        for(VkPipelineColorBlendAttachmentState& state : states){
            state.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        }
        return states;
    }


    inline VkBufferMemoryBarrier bufferMemoryBarrier() {
        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        return barrier;
    }

    template<size_t Size = 1 >
    inline std::array<VkBufferMemoryBarrier, Size> bufferMemoryBarriers(){
        std::array<VkBufferMemoryBarrier, Size> barriers{};
        barriers.fill(bufferMemoryBarrier());
        return barriers;
    }


    inline VkApplicationInfo AppInfo() {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.applicationVersion = VK_MAKE_VERSION(0, 0, 0);
        appInfo.pApplicationName = "";
        appInfo.apiVersion = VK_API_VERSION_1_2;
        appInfo.pEngineName = "";
        return appInfo;
    }

    inline VkBufferCreateInfo bufferCreateInfo() {
        VkBufferCreateInfo  bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        return bufferInfo;
    }


    inline VkRayTracingShaderGroupCreateInfoKHR rayTracingShaderGroupCreateInfo() {
        VkRayTracingShaderGroupCreateInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        return info;
    }


    inline VkRayTracingPipelineCreateInfoKHR rayTracingPipelineCreateInfo() {
        VkRayTracingPipelineCreateInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
        return info;
    }


    inline VkQueryPoolCreateInfo queryPoolCreateInfo() {
        VkQueryPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        return info;
    }

    inline VkImageCopy imageCopy(uint32_t width, uint32_t height, uint32_t depth = 1) {
        VkImageCopy region{};
        region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.srcSubresource.mipLevel = 0;
        region.srcSubresource.baseArrayLayer = 0;
        region.srcSubresource.layerCount = 1;
        region.srcOffset = {0, 0};
        region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.dstSubresource.mipLevel = 0;
        region.dstSubresource.baseArrayLayer = 0;
        region.dstSubresource.layerCount = 1;
        region.dstOffset = {0, 0};
        region.extent = { width, height, 1 };

        return region;
    }

    inline VkImageCopy imageCopy(glm::uvec3 size) {
        return imageCopy(size.x, size.y, size.z);
    }
}