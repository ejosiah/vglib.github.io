#include "VulkanFixture.hpp"

class FindIntervalFixture : public VulkanFixture {
protected:
    void postVulkanInit() override {
        initTestData();
        createDescriptorSet();
    }

    std::vector<PipelineMetaData> pipelineMetaData() override {
        return {
                {
                    "find_interval_test",
                    resource("find_interval_test.comp.spv"),
                    { &descriptorSetLayout}
                }
        };
    }

    void initTestData(){
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_NEAREST;
        samplerInfo.minFilter = VK_FILTER_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        data.sampler = device.createSampler(samplerInfo);

        std::vector<float> array{10, 20, 30, 40, 60, 110, 120, 130, 170};
        textures::create(device, data, VK_IMAGE_TYPE_1D, VK_FORMAT_R32_SFLOAT, array.data(), {9, 1, 1}, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, sizeof(int));

        std::vector<int> storage(12);
        results = device.createCpuVisibleBuffer(storage.data(), BYTE_SIZE(storage), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }

    void createDescriptorSet(){
        descriptorSetLayout =
            device.descriptorSetLayoutBuilder()
                .name("find_interval")
                .binding(0)
                    .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();

        descriptorSet = descriptorPool.allocate( { descriptorSetLayout }).front();

        auto writes = initializers::writeDescriptorSets<2>();
        writes[0].dstSet = descriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].descriptorCount = 1;

        VkDescriptorImageInfo dataImageInfo{data.sampler, data.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        writes[0].pImageInfo = &dataImageInfo;

        writes[1].dstSet = descriptorSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].descriptorCount = 1;

        VkDescriptorBufferInfo resultInfo{results, 0, VK_WHOLE_SIZE};
        writes[1].pBufferInfo = &resultInfo;

        device.updateDescriptorSets(writes);

    }

protected:
    Texture data;
    VulkanBuffer results;
    VulkanDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSet descriptorSet;
};

TEST_F(FindIntervalFixture, findInterval){
    execute([&](auto commandBuffer){
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE
                                , layout("find_interval_test"), 0, 1
                                , &descriptorSet, 0, VK_NULL_HANDLE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("find_interval_test"));
        vkCmdDispatch(commandBuffer, 1, 1, 1);
    });

    auto resultPtr = reinterpret_cast<int*>(results.map());
    EXPECT_EQ(0, resultPtr[0]);
    EXPECT_EQ(1, resultPtr[1]);
    EXPECT_EQ(2, resultPtr[2]);
    EXPECT_EQ(3, resultPtr[3]);
    EXPECT_EQ(4, resultPtr[4]);
    EXPECT_EQ(5, resultPtr[5]);
    EXPECT_EQ(6, resultPtr[6]);
    EXPECT_EQ(7, resultPtr[7]);
    EXPECT_EQ(7, resultPtr[8]);

    EXPECT_EQ(2, resultPtr[9]);
    EXPECT_EQ(5, resultPtr[10]);
    EXPECT_EQ(0, resultPtr[11]);

    results.unmap();
}