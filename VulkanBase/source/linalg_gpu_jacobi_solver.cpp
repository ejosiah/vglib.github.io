#include "linalg/gpu/jacobi_solver.hpp"

#include "Barrier.hpp"
#include "glsl_shaders.hpp"

namespace gpu::linalg {
    std::vector<PipelineMetaData> JacobiSolver::pipelineMetaData() {
        return {{
            {
                .name = "jacobi",
                .shadePath = data_shaders_linalg_jacobi_comp,
                .layouts = {&descriptorSetLayout},
            },
        }};
    }

    void JacobiSolver::solveImpl(VkCommandBuffer commandBuffer, const Params& params) {
        if(params.numIterations == 0) {
            return;
        }

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.pipeline("jacobi"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.layout("jacobi"), 0, 1, &descriptorSet_, 0, nullptr);

        const auto vectorSize = params.Coefficients.numRows * sizeof(float);
        for(uint32_t iteration = 0; iteration < params.numIterations; ++iteration) {
            vkCmdDispatch(commandBuffer, groupCount(params.Coefficients.numRows), 1, 1);
            assign(commandBuffer, unknown_, params.unknown, vectorSize);

            if(iteration + 1 < params.numIterations) {
                Barrier::transferReadWriteToComputeReadWrite(commandBuffer);
            }
        }
    }
}
