#include "VulkanFixture.hpp"
#include <array>

class StaggeredGridTest : public VulkanFixture  {
protected:


    void postVulkanInit() override {
        createVectorField();
        createSampler();
        createTexture();
        createDescriptorSetLayout();
    }

        void createVectorField() {
        static const auto two_pi = 6.2831853071795864f;

        std::array<float, udim * dim> vu{};
        std::array<float, dim * udim> vv{};
        for(auto j = 0; j < dim; ++j) {
            for(auto i = 0; i < udim; ++i) {
                auto y = 2 * float(j)/float(udim) - 1;

                vu[j * udim + i] = glm::sin(two_pi * y);
            }
        }

        for(auto j = 0; j < udim; ++j) {
            for(auto i = 0; i < dim; ++i) {
                auto x = 2 * float(i)/float(udim) - 1;
                vv[j * dim + i] = glm::sin(two_pi * x);
            }
        }
        vector_field.cpu.u = device.createCpuVisibleBuffer(vu.data(), BYTE_SIZE(vu), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        vector_field.cpu.v = device.createCpuVisibleBuffer(vv.data(), BYTE_SIZE(vv), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    }
    
    void createSampler() {
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST ;

        vSampler = device.createSampler(samplerInfo);

        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        lSampler = device.createSampler(samplerInfo);
    }

    void createTexture() {
        textures::createNoTransition(device, vector_field.gpu.u, VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, {udim, dim, 1});
        textures::createNoTransition(device, vector_field.gpu.v, VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, {dim, udim, 1});

        execute([&](auto commandBuffer){
            std::vector<VkImageMemoryBarrier2> barriers(2, {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                    .srcStageMask = VK_PIPELINE_STAGE_NONE,
                    .srcAccessMask = VK_ACCESS_NONE,
                    .dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL
            });

            barriers[0].image = vector_field.gpu.u.image;
            barriers[1].image = vector_field.gpu.v.image;

            VkDependencyInfo dependencyInfo{
                    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                    .imageMemoryBarrierCount = COUNT(barriers),
                    .pImageMemoryBarriers = barriers.data()
            };

            vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);

            vector_field.gpu.u.image.copyFromBuffer(commandBuffer, vector_field.cpu.u, VK_IMAGE_LAYOUT_GENERAL);
            vector_field.gpu.v.image.copyFromBuffer(commandBuffer, vector_field.cpu.v, VK_IMAGE_LAYOUT_GENERAL);

        });
    }

    void createBuffer() {
        std::vector<glm::vec4> allocation(dim * dim);
        auto usage =  VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        result.buffer = device.createCpuVisibleBuffer(allocation.data(), BYTE_SIZE(allocation), usage);
    }
    
    void createDescriptorSetLayout() {
        descriptorSetLayout =
            device.descriptorSetLayoutBuilder()
                .binding(0)
                    .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                    .immutableSamplers(lSampler)
                .binding(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                    .immutableSamplers(lSampler)
                .binding(2)
                    .descriptorType(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .createLayout();
        
        descriptorSet = descriptorPool.allocate( { descriptorSetLayout }).front();
        
        auto writes = initializers::writeDescriptorSets<2>();
        
        writes[0].dstSet = descriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].descriptorCount = 1;
        VkDescriptorImageInfo uInfo{ VK_NULL_HANDLE, vector_field.gpu.u.imageView.handle, VK_IMAGE_LAYOUT_GENERAL };
        writes[0].pImageInfo = &uInfo;

        writes[1].dstSet = descriptorSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].descriptorCount = 1;
        VkDescriptorImageInfo vInfo{ VK_NULL_HANDLE, vector_field.gpu.v.imageView.handle, VK_IMAGE_LAYOUT_GENERAL };
        writes[1].pImageInfo = &vInfo;

        device.updateDescriptorSets(writes);
    }

    std::vector<PipelineMetaData> pipelineMetaData() override {
        return {
                {
                    .name = "vector_field",
                    .shadePath = R"(C:\Users\Josiah Ebhomenye\CLionProjects\vglib\data\shaders\test\vector_field_at_center.comp.spv)",
                    .layouts = { &descriptorSetLayout },
                    .ranges = { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(glm::vec4) } }
                }
        };
    }

    void prepOutput(size_t dimx, size_t dimy) {
        std::vector<glm::vec4> allocation(dimx * dimy);
        auto usage =  VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        result.buffer = device.createCpuVisibleBuffer(allocation.data(), BYTE_SIZE(allocation), usage);

        textures::createNoTransition(device, result.texture, VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, {dimx, dimy, 1});
        execute([&](auto commandBuffer) {
            VkImageMemoryBarrier2 barrier {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                    .srcStageMask = VK_PIPELINE_STAGE_NONE,
                    .srcAccessMask = VK_ACCESS_NONE,
                    .dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .image = result.texture.image
            };

            VkDependencyInfo dependencyInfo{
                    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                    .imageMemoryBarrierCount = 1,
                    .pImageMemoryBarriers = &barrier
            };
            vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
        });

        auto writes = initializers::writeDescriptorSets<1>();
        writes[0].dstSet = descriptorSet;
        writes[0].dstBinding = 2;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes[0].descriptorCount = 1;
        VkDescriptorImageInfo rInfo{ VK_NULL_HANDLE, result.texture.imageView.handle, VK_IMAGE_LAYOUT_GENERAL };
        writes[0].pImageInfo = &rInfo;

        device.updateDescriptorSets(writes);
    }

    void statement(const glm::vec4& center) {
        execute([&](auto commandBuffer) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("vector_field"));
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("vector_field"), 0, 1, &descriptorSet, 0, VK_NULL_HANDLE);
            vkCmdPushConstants(commandBuffer,layout("vector_field"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(glm::vec4), &center);
            vkCmdDispatch(commandBuffer, 1, 1, 1);
            result.texture.image.copyToBuffer(commandBuffer, result.buffer, VK_IMAGE_LAYOUT_GENERAL);
        });
    }

protected:
    static constexpr std::size_t dim = 31;
    static constexpr std::size_t udim = dim+1;

    struct {
        struct {
            VulkanBuffer u;
            VulkanBuffer v;
        } cpu{};
        struct {
            Texture u;
            Texture v;
        } gpu;
    } vector_field;

    struct {
        Texture texture;
        VulkanBuffer buffer;
        std::array<glm::vec2, dim * dim> cpu{};
    } result;

    VulkanSampler vSampler;
    VulkanSampler lSampler;
    VulkanDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSet descriptorSet{};
};

TEST_F(StaggeredGridTest, vectorFieldAtCellCenter) {

    prepOutput(dim, dim);
    statement({1, .5, .5, 1});

    auto vu = vector_field.cpu.u.span<float>();
    auto vv = vector_field.cpu.v.span<float>();

    auto actual = result.buffer.span<glm::vec4>();
    auto expected = std::array<glm::vec2, dim * dim>{};
    for(auto j = 0; j < dim; j++) {
        for(auto i = 0; i < dim; i++) {
            auto u0 = vu[j * udim + i];
            auto u1 = vu[j * udim + i + 1];
            auto v0 = vv[j * dim + i];
            auto v1 = vv[(j+1) * dim + i];
            expected[j * dim + i] = glm::vec2{u0 + u1, v0 + v1} * 0.5f;

//            auto u = vu[j * udim + i];
//            auto v = vv[j * dim + i];
//            expected[j * dim + i] = glm::vec2{u, v};

            ASSERT_NEAR(expected[j * dim + i].x, actual[j * dim + i].x, 1e-3);
            ASSERT_NEAR(expected[j * dim + i].y, actual[j * dim + i].y, 1e-3);
        }
    }
}

TEST_F(StaggeredGridTest, uComponentVectorField) {
    prepOutput(udim, dim);
    statement({.5, .5, 0, 0});

    auto uu = vector_field.cpu.u.span<float>();
    auto vv = vector_field.cpu.v.span<float>();

    auto actual = result.buffer.span<glm::vec4>();
//    auto expected = std::array<glm::vec2, udim * dim>{};
    for(auto j = 0; j < dim; ++j) {
        for(auto i = 0; i < udim; ++i) {
            auto u = uu[j * udim + i];
            auto j0 = j * dim + std::max((i - 1), 0);
            auto j1 = j * dim + std::min(i, int(dim - 1));
            auto j2 = (j+1) * dim + std::max((i - 1), 0);
            auto j3 = (j+1) * dim + std::min(i, int(dim - 1));

            auto v0 = vv[j0];
            auto v1 = vv[j1];
            auto v2 = vv[j2];
            auto v3 = vv[j3];

            auto expected = glm::vec2{u, (v0 + v1 + v2 + v3) * 0.25f};

            ASSERT_NEAR(expected.x, actual[j * udim + i].x, 1e-3);
            ASSERT_NEAR(expected.y, actual[j * udim + i].y, 1e-3);
        }
    }

}

TEST_F(StaggeredGridTest, vComponentVectorField) {
    prepOutput(dim, udim);
    statement({0, 0, .5, .5});

    auto uu = vector_field.cpu.u.span<float>();
    auto vv = vector_field.cpu.v.span<float>();

    auto actual = result.buffer.span<glm::vec4>();

    for(auto j = 0; j < udim; ++j) {
        for(auto i = 0; i < dim; ++i) {
            auto v = vv[j * dim + i];

            auto u0 = uu[std::max(0, (j-1)) * udim + i];
            auto u1 =  uu[std::min(j, int(dim - 1)) * udim + i];
            auto u2 = uu[std::max(0, (j-1)) * udim + i + 1];
            auto u3 =  uu[std::min(j, int(dim - 1)) * udim + i + 1];

            auto expected = glm::vec2{(u0 + u1 + u2 + u3) * 0.25f, v};

            ASSERT_NEAR(expected.x, actual[j * dim + i].x, 1e-3);
            ASSERT_NEAR(expected.y, actual[j * dim + i].y, 1e-3);
        }
    }
}