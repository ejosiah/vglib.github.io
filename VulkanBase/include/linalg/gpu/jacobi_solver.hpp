#pragma once

#include "linalg/gpu/abstract_solver.hpp"

namespace gpu::linalg {
    class JacobiSolver : public AbstractSolver {
    public:
        using AbstractSolver::AbstractSolver;

    private:
        std::vector<PipelineMetaData> pipelineMetaData() override;

        void solveImpl(VkCommandBuffer commandBuffer, const Params& params) override;
    };
}
