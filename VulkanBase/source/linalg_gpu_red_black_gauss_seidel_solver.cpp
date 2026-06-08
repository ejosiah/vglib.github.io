#include "linalg/gpu/red_black_gauss_seidel_solver.hpp"

#include "Barrier.hpp"
#include "glsl_shaders.hpp"

namespace gpu::linalg {
    std::vector<PipelineMetaData> RedBlackGaussSeidelSolver::pipelineMetaData() {
        return {{
            {
                .name = "red_black_gauss_seidel",
                .shadePath = data_shaders_linalg_red_black_gauss_seidel_comp,
                .layouts = {&descriptorSetLayout},
                .ranges = {{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PassConstants)}},
            },
        }};
    }

    void RedBlackGaussSeidelSolver::solveImpl(VkCommandBuffer commandBuffer, const Params& params) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.pipeline("red_black_gauss_seidel"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.layout("red_black_gauss_seidel"), 0, 1, &descriptorSet_, 0, nullptr);

        const auto vectorSize = params.Coefficients.numRows * sizeof(float);
        for(uint32_t iteration = 0; iteration < params.numIterations; ++iteration) {
            PassConstants pass{0};
            vkCmdPushConstants(commandBuffer, compute_.layout("red_black_gauss_seidel"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pass), &pass);
            vkCmdDispatch(commandBuffer, groupCount(params.Coefficients.numRows), 1, 1);
            assign(commandBuffer, unknown_, params.unknown, vectorSize);
            Barrier::transferReadWriteToComputeReadWrite(commandBuffer);

            pass.pass = 1;
            vkCmdPushConstants(commandBuffer, compute_.layout("red_black_gauss_seidel"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pass), &pass);
            vkCmdDispatch(commandBuffer, groupCount(params.Coefficients.numRows), 1, 1);
            assign(commandBuffer, unknown_, params.unknown, vectorSize);

            if(iteration + 1 < params.numIterations) {
                Barrier::transferReadWriteToComputeReadWrite(commandBuffer);
            }
        }
    }
}
