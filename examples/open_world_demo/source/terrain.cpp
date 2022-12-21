#include "terrain.hpp"
#include <spdlog/spdlog.h>
#include "GraphicsPipelineBuilder.hpp"
#include "DescriptorSetBuilder.hpp"
#include "VulkanInitializers.h"
#include <imgui.h>

Terrain::Terrain(const VulkanDevice &device, const VulkanDescriptorPool &descriptorPool, const FileManager& fileManager,
                 uint32_t width, uint32_t height, VulkanRenderPass& renderPass)
    :m_device{&device}
    ,m_descriptorPool{&descriptorPool}
    , m_filemanager(&fileManager)
    , m_width{width}
    , m_height{ height }
    , m_renderPass{ &renderPass }
{
    loadHeightMap();
    loadShadingTextures();
    initUBO();
    createPatches();
    createDescriptorSetLayout();
    updateDescriptorSet();
    createPipelines();
}

void Terrain::loadHeightMap() {
    std::string terrainPath = "height_map";
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    heightMap.displacement.sampler = device().createSampler(samplerInfo);
    heightMap.normal.sampler = device().createSampler(samplerInfo);
    textures::fromFile(device(), heightMap.displacement, resource(fmt::format("terrain/{}.png", terrainPath)));
    textures::fromFile(device(), heightMap.normal, resource(fmt::format("terrain/{}_normal.png", terrainPath)));
}

void Terrain::loadShadingTextures() {
    textures::fromFile(device(), shadingMap.albedo, resource("ground_dirt_rocky/GroundDirtRocky012_COL_4K.jpg"));
    textures::fromFile(device(), shadingMap.normal, resource("ground_dirt_rocky/GroundDirtRocky012_NRM_4K.jpg"));
}

std::vector<glm::vec3> Terrain::generateNormals() {
    VulkanBuffer minMaxBuffer = device().createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU, sizeof(int) * 2);
    glm::ivec2 resolution(heightMap.displacement.width, heightMap.displacement.height);

    auto descriptorSetLayout =
            device().descriptorSetLayoutBuilder()
                .binding(0)
                    .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();
    
    auto descriptorSet = descriptorPool().allocate( { descriptorSetLayout }).front();

    auto writes = initializers::writeDescriptorSets<2>();
    writes[0].dstSet = descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].descriptorCount = 1;
    VkDescriptorImageInfo imageInfo{ heightMap.displacement.sampler, heightMap.displacement.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    writes[0].pImageInfo = &imageInfo;
    
    writes[1].dstSet = descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1;
    VkDescriptorBufferInfo bufferInfo{minMaxBuffer, 0, VK_WHOLE_SIZE};
    writes[1].pBufferInfo = &bufferInfo;
    device().updateDescriptorSets(writes);

    auto module = VulkanShaderModule{resource("stats.comp.spv"), device()};
    auto stage = initializers::shaderStage({ module, VK_SHADER_STAGE_COMPUTE_BIT});

    auto layout = device().createPipelineLayout( {descriptorSetLayout}, {{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(resolution)}} );

    auto computeCreateInfo = initializers::computePipelineCreateInfo();
    computeCreateInfo.stage = stage;
    computeCreateInfo.layout = layout;

    auto pipeline = device().createComputePipeline(computeCreateInfo);

    device().graphicsCommandPool().oneTimeCommand([&](auto commandBuffer){
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descriptorSet, 0, VK_NULL_HANDLE);
        vkCmdPushConstants(commandBuffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(resolution), &resolution);
        auto groupX = static_cast<uint32_t>(glm::ceil(float(resolution.x)/32.f));
        auto groupY = static_cast<uint32_t>(glm::ceil(float(resolution.y)/32.f));

        vkCmdDispatch(commandBuffer, groupX, groupY, 1);
    });

    int* minMax = reinterpret_cast<int*>(minMaxBuffer.map());
    spdlog::info("height map range [{}, {}]", glm::intBitsToFloat(minMax[0]), glm::intBitsToFloat(minMax[1]));

    VkDeviceSize size = (SQRT_NUM_PATCHES + 1) * (SQRT_NUM_PATCHES + 1) * sizeof(glm::vec4);

    auto n = size / sizeof(glm::vec4);
    std::vector<glm::vec3> normals(n);
    normals.reserve(n);


    return normals;
}

void Terrain::createPatches() {
    float aspectRatio = glm::round(float(heightMap.displacement.width)/float(heightMap.displacement.height));
    int w = SQRT_NUM_PATCHES;
    int h = SQRT_NUM_PATCHES/aspectRatio;

    glm::vec2 whole = {w, h};
    glm::vec2 halfPatchSize{ PATCH_SIZE * 0.5f, PATCH_SIZE * 0.5f/aspectRatio };

    auto normals = generateNormals();

    std::vector<PatchVertex> vertices;
    for(int i = 0; i <= h; i++){
        for(int j = 0; j <= w; j++){
            glm::vec2 uv{ static_cast<float>(j)/whole.x, static_cast<float>(i)/whole.y };
            glm::vec2 uvOffset = 2.f * uv - 1.f; // [0, 1] => [-1, 1]
            glm::vec3 position{uvOffset.x * halfPatchSize.x, 0, uvOffset.y * halfPatchSize.y};
            auto normal = normals[i * (SQRT_NUM_PATCHES + 1) + j];

            if(glm::any(glm::isnan(normal))){
                spdlog::info("No normal: {} at : [{}, {}]", normal, j, i);
            }

            vertices.push_back({ position, normal, uv});
//            spdlog::info("index: [{}, {}] <=> uv: {}", j, i, uv);
        }
    }
    
    patchesBuffer = device().createDeviceLocalBuffer(vertices.data(), BYTE_SIZE(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

    std::vector<uint32_t> indices;
    for(int j = 0; j < h; j++){
        for(int i = 0; i < w; i++){
            indices.push_back((j + 1) * (w + 1) + i);
            indices.push_back((j + 1) * (w + 1) + i + 1);
            indices.push_back(j * (w + 1) + i + 1);
            indices.push_back(j * (w + 1) + i);
        }
    }
    spdlog::info("indices {}", indices.size()/4);
    indexBuffer = device().createDeviceLocalBuffer(indices.data(), BYTE_SIZE(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    ubo->numPatches = {w, h};
}

void Terrain::initUBO() {
    uboBuffer = device().createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(UniformBufferObject), "terrain");
    ubo = reinterpret_cast<UniformBufferObject*>(uboBuffer.map());
    ubo->maxHeight = MAX_HEIGHT;
    ubo->wireframeColor = {1, 0, 0};
    ubo->wireframe = 0;
    ubo->wireframeWidth = 5;
    ubo->lod = 0;
    ubo->lodMinDepth = 0.5 * km;
    ubo->lodMaxDepth = 5 * km;
    ubo->minTessLevel = 2;
    ubo->maxTessLevel = 64;
    ubo->lighting = 1;
    ubo->tessLevelColor = 0;
    ubo->viewportSize = { m_width, m_height };
    ubo->lodTargetTriangleWidth = 20.f;
    ubo->lodStrategy = static_cast<int>(LodStrategy::SphereProjection);
}

void Terrain::createDescriptorSetLayout() {
    descriptorSetLayout =
        device().descriptorSetLayoutBuilder()
            .name("terrain")
            .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                .descriptorCount(1)
                .shaderStages(ALL_SHADER_STAGES)
            .binding(1)
                .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT)
            .binding(2)
                .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT)
        .createLayout();

    shadingSetLayout =
        device().descriptorSetLayoutBuilder()
            .name("terrain_shading")
            .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_FRAGMENT_BIT)
            .binding(1)
                .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_FRAGMENT_BIT)
        .createLayout();
}

void Terrain::updateDescriptorSet() {
    auto sets = descriptorPool().allocate( { descriptorSetLayout, shadingSetLayout });
    
    descriptorSet = sets[0];
    shadingSet = sets[1];
    
    auto writes = initializers::writeDescriptorSets<5>();
    
    writes[0].dstSet = descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].descriptorCount = 1;
    VkDescriptorBufferInfo uboInfo{ uboBuffer, 0, VK_WHOLE_SIZE};
    writes[0].pBufferInfo = &uboInfo;

    writes[1].dstSet = descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].descriptorCount = 1;
    VkDescriptorImageInfo dispInfo{heightMap.displacement.sampler, heightMap.displacement.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    writes[1].pImageInfo = &dispInfo;

    writes[2].dstSet = descriptorSet;
    writes[2].dstBinding = 2;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].descriptorCount = 1;
    VkDescriptorImageInfo normalInfo{heightMap.normal.sampler, heightMap.normal.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    writes[2].pImageInfo = &normalInfo;

    writes[3].dstSet = shadingSet;
    writes[3].dstBinding = 0;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[3].descriptorCount = 1;
    VkDescriptorImageInfo albedoInfo{shadingMap.albedo.sampler, shadingMap.albedo.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    writes[3].pImageInfo = &albedoInfo;

    writes[4].dstSet = shadingSet;
    writes[4].dstBinding = 1;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[4].descriptorCount = 1;
    VkDescriptorImageInfo shadingNormalInfo{shadingMap.normal.sampler, shadingMap.normal.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    writes[4].pImageInfo = &shadingNormalInfo;
    
    device().updateDescriptorSets(writes);
}

void Terrain::createPipelines() {
    //    @formatter:off
    auto builder = device().graphicsPipelineBuilder();
    terrain.pipeline =
        builder
            .shaderStage()
                .vertexShader(resource("terrain.vert.spv"))
                .tessellationControlShader(resource("terrain.tesc.spv"))
                .tessellationEvaluationShader(resource("terrain.tese.spv"))
                .geometryShader(resource("terrain.geom.spv"))
                .fragmentShader(resource("terrain.frag.spv"))
            .vertexInputState()
                .addVertexBindingDescription(0, sizeof(PatchVertex), VK_VERTEX_INPUT_RATE_VERTEX)
                .addVertexAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
                .addVertexAttributeDescription(1, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(glm::vec3))
                .addVertexAttributeDescription(2, 0, VK_FORMAT_R32G32_SFLOAT, sizeof(glm::vec3) * 2)
            .inputAssemblyState()
                .patches()
            .tessellationState()
                .patchControlPoints(4)
                .domainOrigin(VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT)
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
                    .addDescriptorSetLayout(descriptorSetLayout)
                    .addDescriptorSetLayout(shadingSetLayout)
                .renderPass(renderPass())
                .subpass(0)
                .name("terrain")
            .build(terrain.layout);
    //    @formatter:on
}


void Terrain::resize(VulkanRenderPass &renderPass, uint32_t width, uint32_t height) {
    m_renderPass = &renderPass;
    m_width = width;
    m_height = height;
    createPipelines();
}

void Terrain::render(VkCommandBuffer commandBuffer) {
    static std::array<VkDescriptorSet,2> sets;
    sets[0] = descriptorSet;
    sets[1] = shadingSet;
    VkDeviceSize offset = 0;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, terrain.pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, terrain.layout, 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, patchesBuffer, &offset);
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(commandBuffer, indexBuffer.sizeAs<uint32_t>(), 1, 0, 0, 0);
}

void Terrain::update(const SceneData& sceneData) {
    const auto& camera = sceneData.camera;
    ubo->model = camera.model;
    ubo->view = camera.view;
    ubo->projection = camera.proj;
    ubo->mvp = camera.proj * camera.view * camera.model;
    ubo->sunPosition = sceneData.sun.position;
}

std::string Terrain::resource(const std::string &name) {
    auto res = m_filemanager->getFullPath(name);
    assert(res.has_value());
    return res->string();
}

void Terrain::renderUI() {
    ImGui::Begin("Terrain");
    ImGui::SetWindowSize({0, 0});

    static bool lighting = static_cast<bool>(ubo->lighting);
    ImGui::Checkbox("shade", &lighting);
    ubo->lighting = static_cast<int>(lighting);

    ImGui::SameLine();
    static bool wireframe = static_cast<bool>(ubo->wireframe);
    ImGui::Checkbox("wireframe", &wireframe);
    ubo->wireframe = static_cast<int>(wireframe);

    if(wireframe && ubo->tessLevelColor == 0){
        ImGui::SameLine();
        ImGui::ColorEdit3("color", &ubo->wireframeColor.x);
    }

    static bool disableHeightMap = false;
    ImGui::Checkbox("disable height map", &disableHeightMap);
    ubo->maxHeight = disableHeightMap ? 0 : MAX_HEIGHT;

    static bool lod = static_cast<bool>(ubo->lod);
    ImGui::Checkbox("Dynamic LOD", &lod);
    ubo->lod = static_cast<int>(lod);

    if(lod){
        ImGui::SameLine();
        static bool tessLevelColor = static_cast<bool>(ubo->tessLevelColor);
        ImGui::Checkbox("tess level color", &tessLevelColor);
        ubo->tessLevelColor = static_cast<int>(tessLevelColor);

        if(ImGui::CollapsingHeader("Level of Detail", &lod,  ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::RadioButton("Distance from camera", &ubo->lodStrategy, static_cast<int>(LodStrategy::DistanceFromCamera));
            ImGui::SameLine();
            ImGui::RadioButton("Sphere projection", &ubo->lodStrategy, static_cast<int>(LodStrategy::SphereProjection));

            if(ubo->lodStrategy == static_cast<int>(LodStrategy::SphereProjection)) {
                ImGui::SliderFloat("target triangle width", &ubo->lodTargetTriangleWidth, 20, 1000);
            }

        }
    }



    ImGui::End();
}

