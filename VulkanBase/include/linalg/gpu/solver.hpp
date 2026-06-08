#pragma once

#include "ComputePipelins.hpp"
#include "VulkanDevice.h"
#include "PrefixSum.hpp"

#include <array>

namespace linalg::gpu {
    struct CSRMatrix {
        VulkanBuffer values;
        VulkanBuffer colIndices;
        VulkanBuffer rowOffsets;
        uint32_t numRows{};
        uint32_t numCols{};
    };

    class Solver {
    public:
        enum class Method : uint32_t {
            Jacobi,
            GaussSeidel,
            ConjugateGradient,
            RedBlackGaussSeidel,
        };

        struct Params {
            CSRMatrix Coefficients;
            VulkanBuffer solution;
            VulkanBuffer unknown;
            uint32_t id{~0u};
            Method method;
            uint32_t numIterations;
            float tolerance{1e-6f};
        };

        Solver() = default;

        Solver(VulkanDevice& device);

        void init(VkDeviceSize reserveSize);

        void solve(VkCommandBuffer commandBuffer, const Params& params);

    private:
        bool preCheck(const Params& params);

        void createDescriptorSetLayout();

        void createDescriptorPool();

        void updateConstants(const Params& params);

        void updateDescriptorSets(const Params& params);

        void jacobi(VkCommandBuffer commandBuffer, const Params& params);

        void gaussSeidel(VkCommandBuffer commandBuffer, const Params& params);

        void redBlackGaussSeidel(VkCommandBuffer commandBuffer, const Params& params);

        void conjugateGradient(VkCommandBuffer commandBuffer, const Params& params);

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

        void dot(VkCommandBuffer commandBuffer, const VulkanBuffer& a, const VulkanBuffer& b);

        void axpy(VkCommandBuffer commandBuffer, VkDescriptorSet descriptorSet, float sign);

        VulkanDevice* device_{};
        ComputePipelines compute_;
        PrefixSum prefixSum_;
        VulkanDescriptorPool descriptorPool_;

        VulkanDescriptorSetLayout descriptorSetLayout;
        VkDescriptorSet descriptorSet_{};

        VulkanDescriptorSetLayout axpyDescriptorSetLayout;
        VkDescriptorSet axpyDescriptorSet0_{};
        VkDescriptorSet axpyDescriptorSet1_{};
        VkDescriptorSet axpyDescriptorSet2_{};

        VulkanDescriptorSetLayout cgDescriptorSetLayout;
        VkDescriptorSet cgDescriptorSet_{};
        std::array<VkDescriptorSet, 2> computeResidualDescriptorSets_{};

        struct CGScalars {
            float rsOld{};
            float rsNew{};
            float alpha{};
            int converged{};
        } cgScalars;

        struct {
            VulkanBuffer residual;
            VulkanBuffer Ap;
            VulkanBuffer p;
            VulkanBuffer scalars;
            VulkanBuffer dotProductIntermediate;
            VulkanBuffer dotProductResult;
        } cg;

        struct BoundBuffers {
            VkBuffer values{};
            VkBuffer colIndices{};
            VkBuffer rowOffsets{};
            VkBuffer solution{};
            VkBuffer unknown{};
        } boundBuffers_;
        bool hasBoundBuffers_{};
        VulkanBuffer unknown_;
        VulkanBuffer gpuConstants_;

        struct Constants {
            float tolerance{};
            uint32_t numRows{};
            uint32_t residualCheckEnabled{};
            uint32_t skip{};
        } constants_;
    };
}
