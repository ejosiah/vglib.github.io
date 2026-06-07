#include "VulkanFixture.hpp"
#include "linalg/linear_system.hpp"
#include "linalg/solver.hpp"

#include <array>
#include <filesystem>
#include <vector>

namespace {
    constexpr uint32_t systemSize = 256;
    constexpr uint32_t jacobiIterations = 30;
    constexpr uint32_t jacobiLocalSize = 32;

    using CpuMatrix = linalg::sparse_csr_matrix<float, systemSize, systemSize>;
    using CpuVector = linalg::vector<float, systemSize>;

    struct JacobiParams {
        float tolerance{};
        uint32_t numRows{};
        uint32_t residualCheckEnabled{};
        uint32_t skip{};
    };

    std::filesystem::path dataPath(const std::filesystem::path& relativePath) {
        return std::filesystem::path{__FILE__}.parent_path().parent_path().parent_path() / "data" / relativePath;
    }
}

class LinearSolversTest : public VulkanFixture {
protected:
    void postVulkanInit() override {
        createDescriptorSetLayout();
        compute = ComputePipelines{
            &device,
            {
                {
                    .name = "jacobi",
                    .shadePath = std::string{LINEAR_SOLVER_JACOBI_SPV},
                    .layouts = {&descriptorSetLayout},
                },
            },
        };
        compute.createPipelines();
    }

    void createDescriptorSetLayout() {
        descriptorSetLayout =
            device.descriptorSetLayoutBuilder()
                .name("linear_solver_jacobi_descriptor_set_layout")
                .binding(0)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(2)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(3)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(4)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(5)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(6)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(7)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .createLayout();
    }

    VkDescriptorSet createJacobiDescriptorSet(const VulkanBuffer& values,
                                              const VulkanBuffer& colIndices,
                                              const VulkanBuffer& rowOffsets,
                                              const VulkanBuffer& b,
                                              const VulkanBuffer& xIn,
                                              const VulkanBuffer& xOut,
                                              const VulkanBuffer& residual,
                                              const VulkanBuffer& params) {
        auto descriptorSet = descriptorPool.allocate({descriptorSetLayout}).front();

        std::array<VkDescriptorBufferInfo, 8> infos{
            VkDescriptorBufferInfo{values, 0, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{colIndices, 0, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{rowOffsets, 0, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{b, 0, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{xIn, 0, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{xOut, 0, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{residual, 0, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{params, 0, VK_WHOLE_SIZE},
        };

        auto writes = initializers::writeDescriptorSets<8>();
        for(uint32_t binding = 0; binding < writes.size(); ++binding) {
            writes[binding].dstSet = descriptorSet;
            writes[binding].dstBinding = binding;
            writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[binding].descriptorCount = 1;
            writes[binding].pBufferInfo = &infos[binding];
        }

        device.updateDescriptorSets(writes);
        return descriptorSet;
    }

    void jacobiBarrier(VkCommandBuffer commandBuffer) {
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                             1,
                             &barrier,
                             0,
                             nullptr,
                             0,
                             nullptr);
    }

    ComputePipelines compute;
    VulkanDescriptorSetLayout descriptorSetLayout;
};

TEST_F(LinearSolversTest, jacobiMatchesCpu) {
    const auto system = linear_system::load(dataPath("linear_system/euler_spd_16.txt"));
    ASSERT_EQ(systemSize, system.rows);
    ASSERT_EQ(systemSize, system.cols);
    ASSERT_EQ(systemSize, system.b.size());

    const auto A = linear_system::create_matrix<CpuMatrix>(system);
    const auto b = linear_system::create_vector<CpuVector>(system);
    ASSERT_EQ(systemSize + 1, A.row_offsets.size());

    const auto expected = linalg::jacobi(A, b, linalg::solver_options{
        .max_iterations = jacobiIterations,
        .tolerance = 0.0,
    });

    auto values = device.createCpuVisibleBuffer(A.data.data(), BYTE_SIZE(A.data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto colIndices = device.createCpuVisibleBuffer(A.col_indices.data(), BYTE_SIZE(A.col_indices), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto rowOffsets = device.createCpuVisibleBuffer(A.row_offsets.data(), BYTE_SIZE(A.row_offsets), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto rhs = device.createCpuVisibleBuffer(b.data.data(), BYTE_SIZE(b.data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    std::vector<float> xZero(systemSize, 0.0f);
    std::vector<float> residual(systemSize, 0.0f);
    auto x0 = device.createCpuVisibleBuffer(xZero.data(), BYTE_SIZE(xZero), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto x1 = device.createCpuVisibleBuffer(xZero.data(), BYTE_SIZE(xZero), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto residualBuffer = device.createCpuVisibleBuffer(residual.data(), BYTE_SIZE(residual), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    JacobiParams params{
        .tolerance = 0.0f,
        .numRows = systemSize,
        .residualCheckEnabled = 0,
        .skip = 0,
    };
    auto paramsBuffer = device.createCpuVisibleBuffer(&params, sizeof(params), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    std::array<VkDescriptorSet, 2> descriptorSets{
        createJacobiDescriptorSet(values, colIndices, rowOffsets, rhs, x0, x1, residualBuffer, paramsBuffer),
        createJacobiDescriptorSet(values, colIndices, rowOffsets, rhs, x1, x0, residualBuffer, paramsBuffer),
    };

    execute([&](auto commandBuffer) {
        for(uint32_t iteration = 0; iteration < jacobiIterations; ++iteration) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipeline("jacobi"));
            vkCmdBindDescriptorSets(commandBuffer,
                                    VK_PIPELINE_BIND_POINT_COMPUTE,
                                    compute.layout("jacobi"),
                                    0,
                                    1,
                                    &descriptorSets[iteration % descriptorSets.size()],
                                    0,
                                    nullptr);
            vkCmdDispatch(commandBuffer, (systemSize + jacobiLocalSize - 1) / jacobiLocalSize, 1, 1);

            if(iteration + 1 < jacobiIterations) {
                jacobiBarrier(commandBuffer);
            }
        }
    });

    const auto& resultBuffer = (jacobiIterations % 2 == 0) ? x0 : x1;
    auto actual = resultBuffer.span<float>(systemSize);
    for(uint32_t i = 0; i < systemSize; ++i) {
        ASSERT_NEAR(expected.x(i), actual[i], 1e-4f);
    }
    resultBuffer.unmap();
}
