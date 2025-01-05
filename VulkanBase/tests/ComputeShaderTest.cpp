#include "VulkanFixture.hpp"

class ComputeShaderFixture : public VulkanFixture {
protected:

    void postVulkanInit() override {
        createTestBuffers();
        createDescriptorSet();
    }

    void createTestBuffers() {
        VkDeviceSize size = (1 << 20) * sizeof(uint32_t);
        inputBuffer = device.createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, size, "input_buffer");
        outputBuffer = device.createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU, size, "output_buffer");

        input = inputBuffer.span<uint32_t>();
        output = outputBuffer.span<uint32_t>();
    }

    void createDescriptorSet() {
        descriptorSetLayout =
            device.descriptorSetLayoutBuilder()
                .name("test_descriptor_set_layout")
                .binding(0)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(2)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .createLayout();

        descriptorSet = descriptorPool.allocate({descriptorSetLayout}).front();

        std::vector<VkDescriptorBufferInfo> infos{
                { inputBuffer, 0, VK_WHOLE_SIZE },
                { outputBuffer, 0, VK_WHOLE_SIZE },
        };

        auto writes = initializers::writeDescriptorSets<1>();
        writes[0].dstSet = descriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].descriptorCount = COUNT(infos);
        writes[0].pBufferInfo = infos.data();

        device.updateDescriptorSets(writes);


    }

    std::vector<PipelineMetaData> pipelineMetaData() override {
        return {
                {
                    .name = "serial_execution",
                    .shadePath = data_shaders_test_return_gid_single_warp_comp,
                    .layouts = { &descriptorSetLayout }
                },
                {
                    .name = "multi_warp_execution",
                    .shadePath = data_shaders_test_return_gid_multiple_warp_comp,
                    .layouts = { &descriptorSetLayout }
                },
                {
                    .name = "single_thread_set_shared_state",
                    .shadePath = data_shaders_test_single_thread_set_shared_state_comp,
                    .layouts = { &descriptorSetLayout }
                },
                {
                    .name = "partial_barrier_meet_up",
                    .shadePath = data_shaders_test_partial_barrier_meet_up_comp,
                    .layouts = { &descriptorSetLayout }
                },
                {
                    .name = "barrier_meet_up_on_continue",
                    .shadePath = data_shaders_test_barrier_meet_up_on_continue_comp,
                    .layouts = { &descriptorSetLayout }
                },
        };
    }

    void serial_execution() {
        execute([&](auto commandBuffer){
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("serial_execution"));
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("serial_execution"), 0, 1, &descriptorSet, 0, 0);
            vkCmdDispatch(commandBuffer, 1, 1, 1);
        });
    }

    void multiple_warp_execution() {
        execute([&](auto commandBuffer){
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("multi_warp_execution"));
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("multi_warp_execution"), 0, 1, &descriptorSet, 0, 0);
            vkCmdDispatch(commandBuffer, 1024, 1, 1);
        });
    }

    void single_thread_set_shared_state() {
        execute([&](auto commandBuffer){
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("single_thread_set_shared_state"));
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("single_thread_set_shared_state"), 0, 1, &descriptorSet, 0, 0);
            vkCmdDispatch(commandBuffer, 1, 1, 1);
        });
    }

    void partial_barrier_meet_up() {
        execute([&](auto commandBuffer){
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("partial_barrier_meet_up"));
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("partial_barrier_meet_up"), 0, 1, &descriptorSet, 0, 0);
            vkCmdDispatch(commandBuffer, 1, 1, 1);
        });
    }

    void barrier_meet_up_on_continue() {
        execute([&](auto commandBuffer){
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("barrier_meet_up_on_continue"));
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("barrier_meet_up_on_continue"), 0, 1, &descriptorSet, 0, 0);
            vkCmdDispatch(commandBuffer, 1, 1, 1);
        });
    }

    VulkanBuffer outputBuffer;
    VulkanBuffer inputBuffer;
    std::span<uint32_t> input;
    std::span<uint32_t> output;
    VulkanDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSet descriptorSet{};
    static constexpr uint32_t warpSize{ 32u };
};

TEST_F(ComputeShaderFixture, thread_in_warp_execute_serially) {
    serial_execution();

    for(auto i = 0; i < warpSize; ++i) {
        ASSERT_EQ(i, output[i]);
    }
}

TEST_F(ComputeShaderFixture, DISABLED_execution_order_not_guaranteed_between_warps) {
    multiple_warp_execution();

    std::vector<uint32_t> result;
    for(auto i = 0; i < 1024; i += warpSize) {
        result.push_back(output[i]);
    }

    ASSERT_NE(0, std::accumulate(result.begin(), result.end(), 0u));

    bool prior_is_less_than = true;
    for(auto i = 1; i < warpSize; ++i) {
        prior_is_less_than &= output[i - 1] < output[i];
    }
    ASSERT_FALSE(prior_is_less_than);
}

TEST_F(ComputeShaderFixture, only_on_thread_sets_shared_state) {
    single_thread_set_shared_state();

    ASSERT_EQ(2048, std::accumulate(output.begin(), output.begin() + warpSize * warpSize, 0u));
}

TEST_F(ComputeShaderFixture, remaing_thread_meet_up_at_barrier_when_others_return_early) {
    partial_barrier_meet_up();

    ASSERT_EQ(512, std::accumulate(output.begin(), output.begin() + warpSize * warpSize, 0u));
}

TEST_F(ComputeShaderFixture, thread_meet_at_barrier_on_loop_continue) {
    for(auto i = 0; i < 1024; i += warpSize) {
        input[i] = 1;
    }

    barrier_meet_up_on_continue();

    ASSERT_EQ(1024, std::accumulate(output.begin(), output.begin() + warpSize * warpSize, 0u));
}