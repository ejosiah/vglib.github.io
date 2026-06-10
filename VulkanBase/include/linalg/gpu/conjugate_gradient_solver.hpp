#pragma once

#include "linalg/gpu/abstract_solver.hpp"
#include "PrefixSum.hpp"

#include <array>

namespace gpu::linalg {
    class ConjugateGradientSolver : public AbstractSolver {
    public:
        ConjugateGradientSolver() = default;

        explicit ConjugateGradientSolver(VulkanDevice& device);

        ConjugateGradientSolver & init(VkDeviceSize reserveSize) override;

    private:
        static constexpr uint32_t cgBindingCount = 6;
        static constexpr uint32_t axpyBindingCount = 4;

        struct CGScalars {
            float rsOld{};
            float rsNew{};
            float alpha{};
            int converged{};
        };

        enum class DotProductInput : uint32_t {
            ResidualResidual,
            PAp
        };

        struct DotProductConstants {
            uint32_t count{};
            uint32_t input{};
        };

        void createSolverDescriptorSetLayouts() override;

        void createSolverBuffers(VkDeviceSize reserveSize) override;

        uint32_t solverDescriptorSetCount() const override;

        uint32_t solverStorageDescriptorCount() const override;

        std::vector<VulkanDescriptorSetLayout> solverDescriptorSetLayouts() override;

        void bindSolverDescriptorSets(const std::vector<VkDescriptorSet>& descriptorSets) override;

        void writeSolverDescriptorSets() override;

        void updateSolverDescriptorSets(const Params& params) override;

        std::vector<PipelineMetaData> pipelineMetaData() override;

        void afterCreatePipelines() override;

        void solveImpl(VkCommandBuffer commandBuffer, const Params& params) override;

        void computeResidual(VkCommandBuffer commandBuffer, const Params& params);

        void checkConvergence(VkCommandBuffer commandBuffer);

        void computeAp(VkCommandBuffer commandBuffer);

        void computeRsOld(VkCommandBuffer commandBuffer);

        void computeRsNew(VkCommandBuffer commandBuffer);

        void computeAlpha(VkCommandBuffer commandBuffer);

        void computeBeta(VkCommandBuffer commandBuffer);

        void computeScalars(VkCommandBuffer commandBuffer, int which);

        void r_minus_alpha_Ap(VkCommandBuffer commandBuffer);

        void x_plus_alpha_p(VkCommandBuffer commandBuffer);

        void r_plus_beta_p(VkCommandBuffer commandBuffer);

        void dot(VkCommandBuffer commandBuffer, DotProductInput input);

        void axpy(VkCommandBuffer commandBuffer, VkDescriptorSet descriptorSet, float sign);

        PrefixSum prefixSum_;

        VulkanDescriptorSetLayout axpyDescriptorSetLayout;
        VkDescriptorSet axpyDescriptorSet0_{};
        VkDescriptorSet axpyDescriptorSet1_{};
        VkDescriptorSet axpyDescriptorSet2_{};

        VulkanDescriptorSetLayout cgDescriptorSetLayout;
        VkDescriptorSet cgDescriptorSet_{};
        std::array<VkDescriptorSet, 2> computeResidualDescriptorSets_{};

        struct {
            VulkanBuffer residual;
            VulkanBuffer Ap;
            VulkanBuffer p;
            VulkanBuffer scalars;
            VulkanBuffer dotProductIntermediate;
            VulkanBuffer dotProductResult;
        } cg;
    };
}
