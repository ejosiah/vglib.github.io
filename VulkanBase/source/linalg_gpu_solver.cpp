#include "linalg/gpu/solver.hpp"

#include "Barrier.hpp"
#include "glsl_shaders.hpp"
#include "gpu/algorithm.h"

#include <array>
#include <stdexcept>


namespace linalg::gpu {
    namespace {
        constexpr uint32_t localSize = 32;
        constexpr uint32_t bindingCount = 8;
        constexpr uint32_t cgBindingCount = 5;
        constexpr uint32_t axpyBindingCount = 4;

        struct PassConstants {
            uint32_t pass{};
        };

        uint32_t groupCount(uint32_t numRows) {
            return (numRows + localSize - 1) / localSize;
        }

        void assign(VkCommandBuffer commandBuffer,
                                  const VulkanBuffer& from,
                                  const VulkanBuffer& to,
                                  VkDeviceSize size) {
            Barrier::computeWriteToTransferRead(commandBuffer);

            VkBufferCopy copy{};
            copy.size = size;
            vkCmdCopyBuffer(commandBuffer, from, to, 1, &copy);
            Barrier::transferWriteToComputeRead(commandBuffer);
        }
    }

    Solver::Solver(VulkanDevice& device): device_{&device}, prefixSum_{&device} {}

    void Solver::init(VkDeviceSize reserveSize) {
        if(!device_) {
            throw std::runtime_error{"linalg::gpu::Solver requires a VulkanDevice before init"};
        }

        createDescriptorSetLayout();
        createDescriptorPool();

        unknown_ = device_->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         VMA_MEMORY_USAGE_GPU_ONLY,
                                         reserveSize);
        gpuConstants_ = device_->createCpuVisibleBuffer(&constants_,
                                                        sizeof(constants_),
                                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        CGScalars scalars{};
        cg.scalars = device_->createDeviceLocalBuffer(&scalars, sizeof(scalars), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        device_->setName<VK_OBJECT_TYPE_BUFFER>("cg_scalar_buffer", cg.scalars.buffer);

        cg.residual = device_->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_GPU_ONLY, reserveSize, "cg_residual_buffer");
        cg.Ap = device_->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, reserveSize, "cg_Ap_buffer");
        cg.p = device_->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY, reserveSize, "cg_p_buffer");
        cg.dotProductIntermediate = device_->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_GPU_ONLY, reserveSize, "cg_dot_product_intermediate_buffer");
        cg.dotProductResult = device_->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY, sizeof(float), "cg_dot_product_result_buffer");

        auto sets = descriptorPool_.allocate({descriptorSetLayout, cgDescriptorSetLayout, axpyDescriptorSetLayout, axpyDescriptorSetLayout, axpyDescriptorSetLayout});
        descriptorSet_ = sets[0];
        cgDescriptorSet_ = sets[1];
        axpyDescriptorSet0_ = sets[2];
        axpyDescriptorSet1_ = sets[3];
        axpyDescriptorSet2_ = sets[4];
        computeResidualDescriptorSets_ = {descriptorSet_, cgDescriptorSet_};

        std::vector<VkWriteDescriptorSet> writes = initializers::writeDescriptorSets<8>(cgDescriptorSet_);

        writes[0].dstSet = cgDescriptorSet_;
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].descriptorCount = 1;
        VkDescriptorBufferInfo residualInfo{ cg.residual, 0, VK_WHOLE_SIZE };
        writes[0].pBufferInfo = &residualInfo;

        writes[1].dstSet = cgDescriptorSet_;
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].descriptorCount = 1;
        VkDescriptorBufferInfo pInfo{ cg.p, 0, VK_WHOLE_SIZE };
        writes[1].pBufferInfo = &pInfo;

        writes[2].dstSet = cgDescriptorSet_;
        writes[2].dstBinding = 2;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].descriptorCount = 1;
        VkDescriptorBufferInfo aPInfo{ cg.Ap, 0, VK_WHOLE_SIZE };
        writes[2].pBufferInfo = &aPInfo;

        VkDescriptorBufferInfo cgScalarInfo{cg.scalars, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo dotProductInfo{cg.dotProductResult, 0, VK_WHOLE_SIZE};
        writes[3].dstSet = cgDescriptorSet_;
        writes[3].dstBinding = 3;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[3].descriptorCount = 1;
        writes[3].pBufferInfo = &cgScalarInfo;

        writes[4].dstSet = cgDescriptorSet_;
        writes[4].dstBinding = 4;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[4].descriptorCount = 1;
        writes[4].pBufferInfo = &dotProductInfo;

        writes[5].dstSet = axpyDescriptorSet0_;
        writes[5].dstBinding = 0;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[5].descriptorCount = 1;
        writes[5].pBufferInfo = &cgScalarInfo;

        writes[6].dstSet = axpyDescriptorSet1_;
        writes[6].dstBinding = 0;
        writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[6].descriptorCount = 1;
        writes[6].pBufferInfo = &cgScalarInfo;

        writes[7].dstSet = axpyDescriptorSet2_;
        writes[7].dstBinding = 0;
        writes[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[7].descriptorCount = 1;
        writes[7].pBufferInfo = &cgScalarInfo;

        device_->updateDescriptorSets(writes);

        compute_ = ComputePipelines{
            device_,
            {
                {
                    .name = "jacobi",
                    .shadePath = data_shaders_linalg_jacobi_comp,
                    .layouts = {&descriptorSetLayout},
                },
                {
                    .name = "red_black_gauss_seidel",
                    .shadePath = data_shaders_linalg_red_black_gauss_seidel_comp,
                    .layouts = {&descriptorSetLayout},
                    .ranges = {{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PassConstants)}},
                },
                {
                    .name = "axpy",
                    .shadePath = data_shaders_linalg_axpy_comp,
                    .layouts = {&axpyDescriptorSetLayout},
                    .ranges = {{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float)}},
                },
                {
                    .name = "compute_residual",
                    .shadePath = data_shaders_linalg_compute_residual_comp,
                    .layouts = {&descriptorSetLayout, &cgDescriptorSetLayout},
                },
                {
                    .name = "compute_ap",
                    .shadePath = data_shaders_linalg_compute_ap_comp,
                    .layouts = {&descriptorSetLayout, &cgDescriptorSetLayout},
                },
                {
                    .name = "cg_check_convergence",
                    .shadePath = data_shaders_linalg_cg_check_convergence_comp,
                    .layouts = {&descriptorSetLayout, &cgDescriptorSetLayout},
                },
                {
                    .name = "cg_compute_scalars",
                    .shadePath = data_shaders_linalg_cg_compute_scalars_comp,
                    .layouts = {&cgDescriptorSetLayout},
                    .ranges = {{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int)}},
                },
            },
        };
        compute_.createPipelines();
        prefixSum_.init();
    }

    void Solver::solve(VkCommandBuffer commandBuffer, const Params& params) {
        const auto shouldUpdateDescriptorSet = preCheck(params);
        updateConstants(params);
        if(shouldUpdateDescriptorSet) {
            updateDescriptorSets(params);
        }

        switch(params.method) {
            case Method::Jacobi:
                jacobi(commandBuffer, params);
                break;
            case Method::GaussSeidel:
            case Method::RedBlackGaussSeidel:
                redBlackGaussSeidel(commandBuffer, params);
                break;
            case Method::ConjugateGradient:
                conjugateGradient(commandBuffer, params);
                break;
        }
    }

    bool Solver::preCheck(const Params& params) {
        if(!device_) {
            throw std::runtime_error{"linalg::gpu::Solver has no VulkanDevice"};
        }
        if(!descriptorSetLayout || !descriptorPool_ || !unknown_ || !gpuConstants_) {
            throw std::runtime_error{"linalg::gpu::Solver::init must be called before solve"};
        }
        if(params.Coefficients.numRows == 0 || params.Coefficients.numCols == 0) {
            throw std::runtime_error{"linalg::gpu::Solver requires non-empty matrix dimensions"};
        }
        if(!params.Coefficients.values || !params.Coefficients.colIndices || !params.Coefficients.rowOffsets ||
           !params.solution || !params.unknown) {
            throw std::runtime_error{"linalg::gpu::Solver received an invalid buffer"};
        }

        const auto vectorSize = params.Coefficients.numRows * sizeof(float);
        const auto rowOffsetSize = (params.Coefficients.numRows + 1) * sizeof(uint32_t);
        if(params.solution.size < vectorSize || params.unknown.size < vectorSize ||
           unknown_.size < vectorSize || params.Coefficients.rowOffsets.size < rowOffsetSize) {
            throw std::runtime_error{"linalg::gpu::Solver buffer is smaller than the declared system size"};
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

    void Solver::createDescriptorSetLayout() {
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

        cgDescriptorSetLayout =
                device_->descriptorSetLayoutBuilder()
                    .name("conjugate_gradient_scalar_descriptor_set_layout")
                    .binding(0) // residual
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1)
                        .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                    .binding(1) // p
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1)
                        .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                    .binding(2) // Ap
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1)
                        .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                    .binding(3) // scalars
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1)
                        .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                    .binding(4) // dot product result
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1)
                        .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();

        axpyDescriptorSetLayout =
                device_->descriptorSetLayoutBuilder()
                    .name("conjugate_gradient_axpy_descriptor_set_layout")
                    .binding(0) // a
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1)
                        .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                    .binding(1) // x
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1)
                        .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                    .binding(2) // y in
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1)
                        .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                    .binding(3) // y out
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1)
                        .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();
    }

    void Solver::createDescriptorPool() {
        constexpr uint32_t maxSets = 5;
        constexpr std::array<VkDescriptorPoolSize, 1> poolSizes{{
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bindingCount + cgBindingCount + 3 * axpyBindingCount},
        }};

        descriptorPool_ = device_->createDescriptorPool(maxSets, poolSizes);
    }

    void Solver::updateConstants(const Params& params) {
        constants_ = {
            .tolerance = params.tolerance,
            .numRows = params.Coefficients.numRows,
            .residualCheckEnabled = 0,
            .skip = 0,
        };
        gpuConstants_.copy(&constants_, sizeof(constants_));
    }

    void Solver::updateDescriptorSets(const Params& params) {
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

        VkDescriptorBufferInfo pInfo{cg.p, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo unknownInfo{params.unknown, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo apInfo{cg.Ap, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo residualInfo{cg.residual, 0, VK_WHOLE_SIZE};

        auto writes = initializers::writeDescriptorSets<bindingCount + 9>();
        for(uint32_t binding = 0; binding < bindingCount; ++binding) {
            writes[binding].dstSet = descriptorSet_;
            writes[binding].dstBinding = binding;
            writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[binding].descriptorCount = 1;
            writes[binding].pBufferInfo = &infos[binding];
        }

        uint32_t writeIndex = bindingCount;

        // x = x + alpha * p
        writes[writeIndex].dstSet = axpyDescriptorSet0_;
        writes[writeIndex].dstBinding = 1;
        writes[writeIndex].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[writeIndex].descriptorCount = 1;
        writes[writeIndex].pBufferInfo = &pInfo;
        ++writeIndex;

        writes[writeIndex].dstSet = axpyDescriptorSet0_;
        writes[writeIndex].dstBinding = 2;
        writes[writeIndex].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[writeIndex].descriptorCount = 1;
        writes[writeIndex].pBufferInfo = &unknownInfo;
        ++writeIndex;

        writes[writeIndex].dstSet = axpyDescriptorSet0_;
        writes[writeIndex].dstBinding = 3;
        writes[writeIndex].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[writeIndex].descriptorCount = 1;
        writes[writeIndex].pBufferInfo = &unknownInfo;
        ++writeIndex;

        // r = r - alpha * Ap
        writes[writeIndex].dstSet = axpyDescriptorSet1_;
        writes[writeIndex].dstBinding = 1;
        writes[writeIndex].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[writeIndex].descriptorCount = 1;
        writes[writeIndex].pBufferInfo = &apInfo;
        ++writeIndex;

        writes[writeIndex].dstSet = axpyDescriptorSet1_;
        writes[writeIndex].dstBinding = 2;
        writes[writeIndex].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[writeIndex].descriptorCount = 1;
        writes[writeIndex].pBufferInfo = &residualInfo;
        ++writeIndex;

        writes[writeIndex].dstSet = axpyDescriptorSet1_;
        writes[writeIndex].dstBinding = 3;
        writes[writeIndex].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[writeIndex].descriptorCount = 1;
        writes[writeIndex].pBufferInfo = &residualInfo;
        ++writeIndex;

        // p = r + beta * p
        writes[writeIndex].dstSet = axpyDescriptorSet2_;
        writes[writeIndex].dstBinding = 1;
        writes[writeIndex].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[writeIndex].descriptorCount = 1;
        writes[writeIndex].pBufferInfo = &pInfo;
        ++writeIndex;

        writes[writeIndex].dstSet = axpyDescriptorSet2_;
        writes[writeIndex].dstBinding = 2;
        writes[writeIndex].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[writeIndex].descriptorCount = 1;
        writes[writeIndex].pBufferInfo = &residualInfo;
        ++writeIndex;

        writes[writeIndex].dstSet = axpyDescriptorSet2_;
        writes[writeIndex].dstBinding = 3;
        writes[writeIndex].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[writeIndex].descriptorCount = 1;
        writes[writeIndex].pBufferInfo = &pInfo;

        device_->updateDescriptorSets(writes);
    }

    void Solver::jacobi(VkCommandBuffer commandBuffer, const Params& params) {
        const auto n = params.numIterations;
        if(n == 0) {
            return;
        }

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.pipeline("jacobi"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.layout("jacobi"), 0, 1, &descriptorSet_, 0, nullptr);

        const auto vectorSize = params.Coefficients.numRows * sizeof(float);
        for(uint32_t iteration = 0; iteration < n; ++iteration) {
            vkCmdDispatch(commandBuffer, groupCount(params.Coefficients.numRows), 1, 1);
            assign(commandBuffer, unknown_, params.unknown, vectorSize);

            if(iteration + 1 < n) {
                Barrier::transferReadWriteToComputeReadWrite(commandBuffer);
            }
        }
    }

    void Solver::gaussSeidel(VkCommandBuffer commandBuffer, const Params& params) {
        redBlackGaussSeidel(commandBuffer, params);
    }

    void Solver::redBlackGaussSeidel(VkCommandBuffer commandBuffer, const Params& params) {
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

    void Solver::conjugateGradient(VkCommandBuffer cmd, const Params &params) {
        const auto vectorSize = params.Coefficients.numRows * sizeof(float);

        computeResidual(cmd, params);
        assign(cmd, cg.residual, cg.p, vectorSize);
        dot(cmd, cg.residual, cg.residual);
        computeRsOld(cmd);

        for (auto itr = 0; itr < params.numIterations; ++itr) {
            computeAp(cmd);
            dot(cmd, cg.p, cg.Ap);
            computeAlpha(cmd);

            x_plus_alpha_p(cmd);
            r_minus_alpha_Ap(cmd);

            dot(cmd, cg.residual, cg.residual);
            computeRsNew(cmd);
            checkConvergence(cmd);
            computeBeta(cmd);

            r_plus_beta_p(cmd);
        }
    }

    void Solver::axpy(VkCommandBuffer commandBuffer, VkDescriptorSet descriptorSet, float sign) {
        const auto gx = (cg.residual.sizeAs<float>() + 31u)/32u;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.pipeline("axpy"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.layout("axpy"), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(commandBuffer, compute_.layout("axpy"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float), &sign);
        vkCmdDispatch(commandBuffer, gx, 1, 1);
        Barrier::computeWriteToRead(commandBuffer);
    }

    void Solver::computeResidual(VkCommandBuffer commandBuffer, const Params& params) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.pipeline("compute_residual"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.layout("compute_residual"), 0, static_cast<uint32_t>(computeResidualDescriptorSets_.size()), computeResidualDescriptorSets_.data(), 0, nullptr);
        vkCmdDispatch(commandBuffer, groupCount(params.Coefficients.numRows), 1, 1);
        Barrier::computeWriteToRead(commandBuffer);
    }

    void Solver::computeAp(VkCommandBuffer commandBuffer) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.pipeline("compute_ap"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.layout("compute_ap"), 0, static_cast<uint32_t>(computeResidualDescriptorSets_.size()), computeResidualDescriptorSets_.data(), 0, nullptr);
        vkCmdDispatch(commandBuffer, groupCount(constants_.numRows), 1, 1);
        Barrier::computeWriteToRead(commandBuffer);
    }

    void Solver::checkConvergence(VkCommandBuffer commandBuffer) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.pipeline("cg_check_convergence"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.layout("cg_check_convergence"), 0, static_cast<uint32_t>(computeResidualDescriptorSets_.size()), computeResidualDescriptorSets_.data(), 0, nullptr);
        vkCmdDispatch(commandBuffer, 1, 1, 1);
        Barrier::computeWriteToRead(commandBuffer);
    }

    void Solver::computeRsOld(VkCommandBuffer commandBuffer) {
        computeScalars(commandBuffer, 0);
    }

    void Solver::computeRsNew(VkCommandBuffer commandBuffer) {
        computeScalars(commandBuffer, 2);
    }

    void Solver::computeAlpha(VkCommandBuffer commandBuffer) {
        computeScalars(commandBuffer, 1);
    }

    void Solver::computeBeta(VkCommandBuffer commandBuffer) {
        computeScalars(commandBuffer, 3);
    }

    void Solver::computeScalars(VkCommandBuffer commandBuffer, int which) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.pipeline("cg_compute_scalars"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.layout("cg_compute_scalars"), 0, 1, &cgDescriptorSet_, 0, nullptr);
        vkCmdPushConstants(commandBuffer, compute_.layout("cg_compute_scalars"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(which), &which);
        vkCmdDispatch(commandBuffer, 1, 1, 1);
        Barrier::computeWriteToRead(commandBuffer);
    }

    void Solver::x_plus_alpha_p(VkCommandBuffer commandBuffer) {
        axpy(commandBuffer, axpyDescriptorSet0_, 1.0f);
    }

    void Solver::r_minus_alpha_Ap(VkCommandBuffer commandBuffer) {
        axpy(commandBuffer, axpyDescriptorSet1_, -1.0f);
    }

    void Solver::r_plus_beta_p(VkCommandBuffer commandBuffer) {
        axpy(commandBuffer, axpyDescriptorSet2_, 1.0f);
    }

    void Solver::dot(VkCommandBuffer commandBuffer, const VulkanBuffer& a, const VulkanBuffer& b) {
        ::gpu::multiply(commandBuffer, a, b, cg.dotProductIntermediate);
        Barrier::computeWriteToTransferRead(commandBuffer, {cg.dotProductIntermediate});
        prefixSum_.accumulate(commandBuffer, cg.dotProductIntermediate, cg.dotProductResult, Operation::Add, DataType::Float);
        Barrier::computeWriteToRead(commandBuffer);
    }
}
