#include "linalg/gpu/abstract_solver.hpp"

#include "Barrier.hpp"

#include <stdexcept>

namespace gpu::linalg {
    uint32_t AbstractSolver::groupCount(uint32_t numRows) {
        return (numRows + localSize - 1) / localSize;
    }

    void AbstractSolver::assign(VkCommandBuffer commandBuffer, const VulkanBuffer& from, const VulkanBuffer& to, VkDeviceSize size) {
        Barrier::computeWriteToTransferRead(commandBuffer);

        VkBufferCopy copy{};
        copy.size = size;
        vkCmdCopyBuffer(commandBuffer, from, to, 1, &copy);
        Barrier::transferWriteToComputeRead(commandBuffer);
    }

    AbstractSolver::AbstractSolver(VulkanDevice& device): device_{&device} {}

    void AbstractSolver::init(VkDeviceSize reserveSize) {
        if(!device_) {
            throw std::runtime_error{"gpu::linalg::AbstractSolver requires a VulkanDevice before init"};
        }

        createDescriptorSetLayout();
        createSolverDescriptorSetLayouts();
        createDescriptorPool();

        unknown_ = device_->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         VMA_MEMORY_USAGE_GPU_ONLY,
                                         reserveSize);
        gpuConstants_ = device_->createCpuVisibleBuffer(&constants_, sizeof(constants_), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        createSolverBuffers(reserveSize);

        auto layouts = std::vector<VulkanDescriptorSetLayout>{descriptorSetLayout};
        auto solverLayouts = solverDescriptorSetLayouts();
        layouts.insert(layouts.end(), solverLayouts.begin(), solverLayouts.end());

        auto sets = descriptorPool_.allocate(layouts);
        descriptorSet_ = sets[0];
        std::vector<VkDescriptorSet> solverSets(sets.begin() + 1, sets.end());
        bindSolverDescriptorSets(solverSets);
        writeSolverDescriptorSets();

        compute_ = ComputePipelines{device_, pipelineMetaData()};
        compute_.createPipelines();
        afterCreatePipelines();
    }

    void AbstractSolver::solve(VkCommandBuffer commandBuffer, const Params& params) {
        const auto shouldUpdateDescriptorSet = preCheck(params);
        updateConstants(params);
        if(shouldUpdateDescriptorSet) {
            updateDescriptorSets(params);
        }

        solveImpl(commandBuffer, params);
    }

    bool AbstractSolver::preCheck(const Params& params) {
        if(!device_) {
            throw std::runtime_error{"gpu::linalg::AbstractSolver has no VulkanDevice"};
        }
        if(!descriptorSetLayout || !descriptorPool_ || !unknown_ || !gpuConstants_) {
            throw std::runtime_error{"gpu::linalg::AbstractSolver::init must be called before solve"};
        }
        if(params.Coefficients.numRows == 0 || params.Coefficients.numCols == 0) {
            throw std::runtime_error{"gpu::linalg::AbstractSolver requires non-empty matrix dimensions"};
        }
        if(!params.Coefficients.values || !params.Coefficients.colIndices || !params.Coefficients.rowOffsets || !params.solution || !params.unknown) {
            throw std::runtime_error{"gpu::linalg::AbstractSolver received an invalid buffer"};
        }

        const auto vectorSize = params.Coefficients.numRows * sizeof(float);
        const auto rowOffsetSize = (params.Coefficients.numRows + 1) * sizeof(uint32_t);
        if(params.solution.size < vectorSize || params.unknown.size < vectorSize || unknown_.size < vectorSize || params.Coefficients.rowOffsets.size < rowOffsetSize) {
            throw std::runtime_error{"gpu::linalg::AbstractSolver buffer is smaller than the declared system size"};
        }

        const BoundBuffers currentBindings{
            .values = params.Coefficients.values.buffer,
            .colIndices = params.Coefficients.colIndices.buffer,
            .rowOffsets = params.Coefficients.rowOffsets.buffer,
            .solution = params.solution.buffer,
            .unknown = params.unknown.buffer,
        };

        const auto shouldUpdateDescriptorSet =
            !hasBoundBuffers_ ||
            boundBuffers_.values != currentBindings.values ||
            boundBuffers_.colIndices != currentBindings.colIndices ||
            boundBuffers_.rowOffsets != currentBindings.rowOffsets ||
            boundBuffers_.solution != currentBindings.solution ||
            boundBuffers_.unknown != currentBindings.unknown;

        if(shouldUpdateDescriptorSet) {
            boundBuffers_ = currentBindings;
            hasBoundBuffers_ = true;
        }

        return shouldUpdateDescriptorSet;
    }

    void AbstractSolver::createDescriptorSetLayout() {
        descriptorSetLayout =
            device_->descriptorSetLayoutBuilder()
                .name("linear_solver_descriptor_set_layout")
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

    void AbstractSolver::createDescriptorPool() {
        const auto maxSets = 1u + solverDescriptorSetCount();
        const std::array<VkDescriptorPoolSize, 1> poolSizes{{
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bindingCount + solverStorageDescriptorCount()},
        }};

        descriptorPool_ = device_->createDescriptorPool(maxSets, poolSizes);
    }

    void AbstractSolver::updateConstants(const Params& params) {
        constants_ = {
            .tolerance = params.tolerance,
            .numRows = params.Coefficients.numRows,
            .residualCheckEnabled = 0,
            .skip = 0,
        };
        gpuConstants_.copy(&constants_, sizeof(constants_));
    }

    void AbstractSolver::updateDescriptorSets(const Params& params) {
        const std::array<VkDescriptorBufferInfo, bindingCount> infos{{
            {params.Coefficients.values, 0, VK_WHOLE_SIZE},
            {params.Coefficients.colIndices, 0, VK_WHOLE_SIZE},
            {params.Coefficients.rowOffsets, 0, VK_WHOLE_SIZE},
            {params.solution, 0, VK_WHOLE_SIZE},
            {params.unknown, 0, VK_WHOLE_SIZE},
            {unknown_, 0, VK_WHOLE_SIZE},
            {unknown_, 0, VK_WHOLE_SIZE},
            {gpuConstants_, 0, VK_WHOLE_SIZE},
        }};

        auto writes = initializers::writeDescriptorSets<bindingCount>();
        for(uint32_t binding = 0; binding < bindingCount; ++binding) {
            writes[binding].dstSet = descriptorSet_;
            writes[binding].dstBinding = binding;
            writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[binding].descriptorCount = 1;
            writes[binding].pBufferInfo = &infos[binding];
        }

        device_->updateDescriptorSets(writes);
        updateSolverDescriptorSets(params);
    }

    void AbstractSolver::createSolverDescriptorSetLayouts() {}

    void AbstractSolver::createSolverBuffers(VkDeviceSize) {}

    uint32_t AbstractSolver::solverDescriptorSetCount() const {
        return 0;
    }

    uint32_t AbstractSolver::solverStorageDescriptorCount() const {
        return 0;
    }

    std::vector<VulkanDescriptorSetLayout> AbstractSolver::solverDescriptorSetLayouts() {
        return {};
    }

    void AbstractSolver::bindSolverDescriptorSets(const std::vector<VkDescriptorSet>&) {}

    void AbstractSolver::writeSolverDescriptorSets() {}

    void AbstractSolver::updateSolverDescriptorSets(const Params&) {}

    void AbstractSolver::afterCreatePipelines() {}
}
