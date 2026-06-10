#pragma once

#include "ComputePipelins.hpp"
#include "VulkanDevice.h"
#include "csr_matrix.hpp"

#include <array>
#include <vector>

namespace gpu::linalg {

    class AbstractSolver {
    public:
        struct Params {
            CSRMatrix Coefficients;
            VulkanBuffer solution;
            VulkanBuffer unknown;
            uint32_t id{~0u};
            uint32_t numIterations{};
            float tolerance{1e-6f};
        };

        AbstractSolver() = default;

        explicit AbstractSolver(VulkanDevice& device);

        virtual ~AbstractSolver() = default;

        virtual AbstractSolver& init(VkDeviceSize reserveSize);

        void solve(VkCommandBuffer commandBuffer, const Params& params);

    protected:
        struct Constants {
            float tolerance{};
            uint32_t numRows{};
            uint32_t residualCheckEnabled{};
            uint32_t skip{};
        };

        static constexpr uint32_t localSize = 32;
        static constexpr uint32_t bindingCount = 8;

        static uint32_t groupCount(uint32_t numRows);

        static void assign(VkCommandBuffer commandBuffer, const VulkanBuffer& from, const VulkanBuffer& to, VkDeviceSize size);

        virtual void createSolverDescriptorSetLayouts();

        virtual void createSolverBuffers(VkDeviceSize reserveSize);

        virtual uint32_t solverDescriptorSetCount() const;

        virtual uint32_t solverStorageDescriptorCount() const;

        virtual std::vector<VulkanDescriptorSetLayout> solverDescriptorSetLayouts();

        virtual void bindSolverDescriptorSets(const std::vector<VkDescriptorSet>& descriptorSets);

        virtual void writeSolverDescriptorSets();

        virtual void updateSolverDescriptorSets(const Params& params);

        virtual std::vector<PipelineMetaData> pipelineMetaData() = 0;

        virtual void afterCreatePipelines();

        virtual void solveImpl(VkCommandBuffer commandBuffer, const Params& params) = 0;

        VulkanDevice* device_{};
        ComputePipelines compute_;
        VulkanDescriptorPool descriptorPool_;

        VulkanDescriptorSetLayout descriptorSetLayout;
        VkDescriptorSet descriptorSet_{};

        VulkanBuffer unknown_;
        VulkanBuffer gpuConstants_;
        Constants constants_;

    private:
        struct BoundBuffers {
            VkBuffer values{};
            VkBuffer colIndices{};
            VkBuffer rowOffsets{};
            VkBuffer solution{};
            VkBuffer unknown{};
        };

        bool preCheck(const Params& params);

        void createDescriptorSetLayout();

        void createDescriptorPool();

        void updateConstants(const Params& params);

        void updateDescriptorSets(const Params& params);

        BoundBuffers boundBuffers_;
        bool hasBoundBuffers_{};
    };
}
