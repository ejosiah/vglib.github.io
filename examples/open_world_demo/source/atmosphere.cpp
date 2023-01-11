#include "atmosphere.hpp"
#include "GraphicsPipelineBuilder.hpp"

#include <utility>

Atmosphere::Atmosphere(const VulkanDevice& device, const VulkanDescriptorPool& descriptorPool, const FileManager& fileManager,
                       VulkanRenderPass& renderPass, uint32_t width, uint32_t height, std::shared_ptr<GBuffer> terrainGBuffer,
                       std::shared_ptr<ShadowVolume> terrainShadowVolume)
        :m_device{&device}
        ,m_descriptorPool{&descriptorPool}
        , m_filemanager(&fileManager)
        , m_width{width}
        , m_height{ height }
        , m_renderPass{ &renderPass }
        , m_terrainGBuffer{ std::move(terrainGBuffer) }
        , m_terrainShadowVolume{ std::move(terrainShadowVolume) }
{
    loadAtmosphereLUT();
    initUbo();
    initBuffers();
    createDescriptorSetLayouts();
    updateDescriptorSets();
    createPipelines();
}

void Atmosphere::loadAtmosphereLUT() {
    auto data = loadFile(resource("atmosphere/irradiance.dat"));

    textures::create(device() ,atmosphereLUT.irradiance, VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, data.data(),
                     {IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT, 1}, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, sizeof(float) );


    data = loadFile(resource("atmosphere/transmittance.dat"));
    textures::create(device() ,atmosphereLUT.transmittance, VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, data.data(),
                     {TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT, 1}, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, sizeof(float) );

    data = loadFile(resource("atmosphere/scattering.dat"));
    textures::create(device() ,atmosphereLUT.scattering, VK_IMAGE_TYPE_3D, VK_FORMAT_R32G32B32A32_SFLOAT, data.data(),
                     {SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_DEPTH}, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, sizeof(float) );
}

void Atmosphere::initBuffers() {
    auto positions = ClipSpace::Quad::positions;
    screenBuffer = device().createDeviceLocalBuffer(positions.data(), BYTE_SIZE(positions), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
}

void Atmosphere::initUbo() {
    uboBuffer = device().createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(Ubo), "atmosphere_uniforms");
    ubo = reinterpret_cast<Ubo*>(uboBuffer.map());

    ubo->white_point = glm::vec3(1);
    ubo->earth_center = (EARTH_CENTER + glm::vec3(0, MAX_HEIGHT, 0))/kLengthUnitInMeters;
    ubo->sun_size = glm::vec3(glm::tan(kSunAngularRadius), glm::cos(kSunAngularRadius), 0);
    ubo->exposure = 10.f;
    ubo->lightShaft = 0;
}

void Atmosphere::createDescriptorSetLayouts() {
    atmosphereLutSetLayout =
        device().descriptorSetLayoutBuilder()
            .name("atmosphere")
            .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_FRAGMENT_BIT)
            .binding(1)
                .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_FRAGMENT_BIT)
            .binding(2)
                .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_FRAGMENT_BIT)
            .binding(3)
                .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_FRAGMENT_BIT)
        .createLayout();
    
    uboSetLayout =
        device().descriptorSetLayoutBuilder()
            .name("ubo")
            .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
        .createLayout();
}

void Atmosphere::updateDescriptorSets() {
    auto sets = m_descriptorPool->allocate({ atmosphereLutSetLayout, uboSetLayout });
    atmosphereLutSet = sets[0];
    uboSet = sets[1];

    auto writes = initializers::writeDescriptorSets<5>();

    writes[0].dstSet = atmosphereLutSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].descriptorCount = 1;
    VkDescriptorImageInfo irradianceInfo{atmosphereLUT.irradiance.sampler, atmosphereLUT.irradiance.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    writes[0].pImageInfo = &irradianceInfo;

    writes[1].dstSet = atmosphereLutSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].descriptorCount = 1;
    VkDescriptorImageInfo transmittanceInfo{atmosphereLUT.transmittance.sampler, atmosphereLUT.transmittance.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    writes[1].pImageInfo = &transmittanceInfo;

    writes[2].dstSet = atmosphereLutSet;
    writes[2].dstBinding = 2;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].descriptorCount = 1;
    VkDescriptorImageInfo scatteringInfo{atmosphereLUT.scattering.sampler, atmosphereLUT.scattering.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    writes[2].pImageInfo = &scatteringInfo;

    // single_mie_scattering
    writes[3] = writes[2];
    writes[3].dstBinding = 3;

    writes[4].dstSet = uboSet;
    writes[4].dstBinding = 0;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[4].descriptorCount = 1;
    VkDescriptorBufferInfo uboInfo{ uboBuffer, 0, VK_WHOLE_SIZE };
    writes[4].pBufferInfo = &uboInfo;


    device().updateDescriptorSets(writes);
}

void Atmosphere::createPipelines() {
    //    @formatter:off
    auto builder = device().graphicsPipelineBuilder();
    atmosphere.pipeline =
        builder
            .shaderStage()
                .vertexShader(resource("atmosphere.vert.spv"))
                .fragmentShader(resource("atmosphere.frag.spv"))
            .vertexInputState()
                .addVertexBindingDescriptions(ClipSpace::bindingDescription())
                .addVertexAttributeDescriptions(ClipSpace::attributeDescriptions())
            .inputAssemblyState()
                .triangleStrip()
            .viewportState()
                .viewport()
                    .origin(0, 0)
                    .dimension(m_width, m_height)
                    .minDepth(0)
                    .maxDepth(1)
                .scissor()
                    .offset(0, 0)
                    .extent(m_width, m_height)
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
                    .addDescriptorSetLayout(atmosphereLutSetLayout)
                    .addDescriptorSetLayout(uboSetLayout)
                    .addDescriptorSetLayout(m_terrainGBuffer->setLayout)
                    .addDescriptorSetLayout(m_terrainShadowVolume->setLayout)
                .renderPass(*m_renderPass)
                .subpass(0)
                .name("render")
            .build(atmosphere.layout);
    //    @formatter:on
}

std::string Atmosphere::resource(const std::string &name) {
    auto res = m_filemanager->getFullPath(name);
    assert(res.has_value());
    return res->string();
}

void Atmosphere::update(const SceneData &sceneData) {
    const auto& camera = sceneData.camera;
    ubo->model_from_view = glm::inverse(camera.view * camera.model);
    ubo->view_from_clip = glm::inverse(camera.proj);
    ubo->sun_direction = glm::normalize(sceneData.sun.position);
    ubo->camera = sceneData.eyes/kLengthUnitInMeters;
    ubo->exposure = sceneData.exposure;
    ubo->lightShaft = static_cast<int>(sceneData.enableLightShaft);
}

void Atmosphere::render(VkCommandBuffer commandBuffer) {
    VkDeviceSize offset = 0;
    static std::array<VkDescriptorSet,4> sets;
    sets[0] = atmosphereLutSet;
    sets[1] = uboSet;
    sets[2] = m_terrainGBuffer->descriptorSet;
    sets[3] = m_terrainShadowVolume->descriptorSet;

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, atmosphere.pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, atmosphere.layout, 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, screenBuffer, &offset);
    vkCmdDraw(commandBuffer, 4, 1, 0, 0);
}

void Atmosphere::resize(VulkanRenderPass& renderPass, std::shared_ptr<GBuffer> terrainGBuffer,
                        std::shared_ptr<ShadowVolume> terrainShadowVolume, uint32_t width, uint32_t height) {
    m_renderPass = &renderPass;
    m_width = width;
    m_height = height;
    m_terrainGBuffer = std::move(terrainGBuffer);
    m_terrainShadowVolume = std::move(terrainShadowVolume);
    updateDescriptorSets();
    createPipelines();
}
