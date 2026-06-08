#pragma once

#include "linalg/gpu/abstract_solver.hpp"

namespace gpu::linalg {
    class RedBlackGaussSeidelSolver : public AbstractSolver {
    public:
        using AbstractSolver::AbstractSolver;

    private:
        struct PassConstants {
            uint32_t pass{};
        };

        std::vector<PipelineMetaData> pipelineMetaData() override;

        void solveImpl(VkCommandBuffer commandBuffer, const Params& params) override;
    };

    class GaussSeidelSolver : public RedBlackGaussSeidelSolver {
    public:
        using RedBlackGaussSeidelSolver::RedBlackGaussSeidelSolver;
    };
}
