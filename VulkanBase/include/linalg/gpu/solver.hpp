#pragma once

#include "ComputePipelins.hpp"
#include "VulkanDevice.h"

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
        enum class Method : uint32_t { Jacobi, GuassSeidel, ConjugateGradient };

        struct Params {
            CSRMatrix Coefficients;
            VulkanBuffer solution;
            VulkanBuffer unknown;
            uint32_t id{~0u};
            Method method;
            uint32_t numIterations;
        };

        Solver() = default;

        Solver(VulkanDevice& device);

        void init(const Params& params);


        void solve(VkCommandBuffer commandBuffer, const Params& params);

        static VulkanDescriptorSetLayout descriptorSetLayout;

    private:
        void preCheck(const Params& params);

        void createDescriptorSetLayout();

        void createDescriptorSetLayout(const VulkanDevice& device);

        void updateDescriptorSetLayout(const Params& params);

        void jacobi(VkCommandBuffer commandBuffer, const Params& params);

        VulkanDevice* device_{};
        ComputePipelines compute_;
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