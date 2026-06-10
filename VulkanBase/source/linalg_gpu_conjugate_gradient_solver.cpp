#include "linalg/gpu/conjugate_gradient_solver.hpp"

#include "Barrier.hpp"
#include "glsl_shaders.hpp"

namespace gpu::linalg {
    ConjugateGradientSolver::ConjugateGradientSolver(VulkanDevice& device): AbstractSolver(device), prefixSum_{&device} {}

    ConjugateGradientSolver & ConjugateGradientSolver::init(VkDeviceSize reserveSize) {
        AbstractSolver::init(reserveSize);

        return *this;
    }

    void ConjugateGradientSolver::createSolverDescriptorSetLayouts() {
        cgDescriptorSetLayout =
            device_->descriptorSetLayoutBuilder()
                .name("conjugate_gradient_scalar_descriptor_set_layout")
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
                .createLayout();

        axpyDescriptorSetLayout =
            device_->descriptorSetLayoutBuilder()
                .name("conjugate_gradient_axpy_descriptor_set_layout")
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
                .createLayout();
    }

    void ConjugateGradientSolver::createSolverBuffers(VkDeviceSize reserveSize) {
        CGScalars scalars{};
        cg.scalars = device_->createDeviceLocalBuffer(&scalars, sizeof(scalars),
                                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        device_->setName<VK_OBJECT_TYPE_BUFFER>("cg_scalar_buffer", cg.scalars.buffer);

        cg.residual = device_->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_GPU_ONLY, reserveSize, "cg_residual_buffer");
        cg.Ap = device_->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, reserveSize, "cg_Ap_buffer");
        cg.p = device_->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY, reserveSize, "cg_p_buffer");
        cg.dotProductIntermediate = device_->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_GPU_ONLY, reserveSize, "cg_dot_product_intermediate_buffer");
        cg.dotProductResult = device_->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY, sizeof(float), "cg_dot_product_result_buffer");
    }

    uint32_t ConjugateGradientSolver::solverDescriptorSetCount() const {
        return 4;
    }

    uint32_t ConjugateGradientSolver::solverStorageDescriptorCount() const {
        return cgBindingCount + 3 * axpyBindingCount;
    }

    std::vector<VulkanDescriptorSetLayout> ConjugateGradientSolver::solverDescriptorSetLayouts() {
        return {cgDescriptorSetLayout, axpyDescriptorSetLayout, axpyDescriptorSetLayout, axpyDescriptorSetLayout};
    }

    void ConjugateGradientSolver::bindSolverDescriptorSets(const std::vector<VkDescriptorSet>& descriptorSets) {
        cgDescriptorSet_ = descriptorSets[0];
        axpyDescriptorSet0_ = descriptorSets[1];
        axpyDescriptorSet1_ = descriptorSets[2];
        axpyDescriptorSet2_ = descriptorSets[3];
        computeResidualDescriptorSets_ = {descriptorSet_, cgDescriptorSet_};
    }

    void ConjugateGradientSolver::writeSolverDescriptorSets() {
        VkDescriptorBufferInfo residualInfo{cg.residual, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo pInfo{cg.p, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo aPInfo{cg.Ap, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo cgScalarInfo{cg.scalars, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo dotProductInfo{cg.dotProductResult, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo dotProductIntermediateInfo{cg.dotProductIntermediate, 0, VK_WHOLE_SIZE};

        auto writes = initializers::writeDescriptorSets<9>();

        writes[0].dstSet = cgDescriptorSet_;
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &residualInfo;

        writes[1].dstSet = cgDescriptorSet_;
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].descriptorCount = 1;
        writes[1].pBufferInfo = &pInfo;

        writes[2].dstSet = cgDescriptorSet_;
        writes[2].dstBinding = 2;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].descriptorCount = 1;
        writes[2].pBufferInfo = &aPInfo;

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

        writes[5].dstSet = cgDescriptorSet_;
        writes[5].dstBinding = 5;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[5].descriptorCount = 1;
        writes[5].pBufferInfo = &dotProductIntermediateInfo;

        writes[6].dstSet = axpyDescriptorSet0_;
        writes[6].dstBinding = 0;
        writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[6].descriptorCount = 1;
        writes[6].pBufferInfo = &cgScalarInfo;

        writes[7].dstSet = axpyDescriptorSet1_;
        writes[7].dstBinding = 0;
        writes[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[7].descriptorCount = 1;
        writes[7].pBufferInfo = &cgScalarInfo;

        writes[8].dstSet = axpyDescriptorSet2_;
        writes[8].dstBinding = 0;
        writes[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[8].descriptorCount = 1;
        writes[8].pBufferInfo = &cgScalarInfo;

        device_->updateDescriptorSets(writes);
    }

    void ConjugateGradientSolver::updateSolverDescriptorSets(const Params& params) {
        VkDescriptorBufferInfo pInfo{cg.p, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo unknownInfo{params.unknown, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo apInfo{cg.Ap, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo residualInfo{cg.residual, 0, VK_WHOLE_SIZE};

        auto writes = initializers::writeDescriptorSets<9>();

        writes[0].dstSet = axpyDescriptorSet0_;
        writes[0].dstBinding = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &pInfo;

        writes[1].dstSet = axpyDescriptorSet0_;
        writes[1].dstBinding = 2;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].descriptorCount = 1;
        writes[1].pBufferInfo = &unknownInfo;

        writes[2].dstSet = axpyDescriptorSet0_;
        writes[2].dstBinding = 3;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].descriptorCount = 1;
        writes[2].pBufferInfo = &unknownInfo;

        writes[3].dstSet = axpyDescriptorSet1_;
        writes[3].dstBinding = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[3].descriptorCount = 1;
        writes[3].pBufferInfo = &apInfo;

        writes[4].dstSet = axpyDescriptorSet1_;
        writes[4].dstBinding = 2;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[4].descriptorCount = 1;
        writes[4].pBufferInfo = &residualInfo;

        writes[5].dstSet = axpyDescriptorSet1_;
        writes[5].dstBinding = 3;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[5].descriptorCount = 1;
        writes[5].pBufferInfo = &residualInfo;

        writes[6].dstSet = axpyDescriptorSet2_;
        writes[6].dstBinding = 1;
        writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[6].descriptorCount = 1;
        writes[6].pBufferInfo = &pInfo;

        writes[7].dstSet = axpyDescriptorSet2_;
        writes[7].dstBinding = 2;
        writes[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[7].descriptorCount = 1;
        writes[7].pBufferInfo = &residualInfo;

        writes[8].dstSet = axpyDescriptorSet2_;
        writes[8].dstBinding = 3;
        writes[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[8].descriptorCount = 1;
        writes[8].pBufferInfo = &pInfo;

        device_->updateDescriptorSets(writes);
    }

    std::vector<PipelineMetaData> ConjugateGradientSolver::pipelineMetaData() {
        return {{
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
                .name = "cg_dot_product_terms",
                .shadePath = data_shaders_linalg_cg_dot_product_comp,
                .layouts = {&cgDescriptorSetLayout},
                .ranges = {{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DotProductConstants)}},
            },
            {
                .name = "cg_compute_scalars",
                .shadePath = data_shaders_linalg_cg_compute_scalars_comp,
                .layouts = {&cgDescriptorSetLayout},
                .ranges = {{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int)}},
            },
        }};
    }

    void ConjugateGradientSolver::afterCreatePipelines() {
        prefixSum_.init();
    }

    void ConjugateGradientSolver::solveImpl(VkCommandBuffer commandBuffer, const Params& params) {
        const auto vectorSize = params.Coefficients.numRows * sizeof(float);

        vkCmdFillBuffer(commandBuffer, cg.scalars, 0, sizeof(CGScalars), 0);
        Barrier::transferWriteToComputeRead(commandBuffer, cg.scalars);

        computeResidual(commandBuffer, params);
        assign(commandBuffer, cg.residual, cg.p, vectorSize);
        dot(commandBuffer, DotProductInput::ResidualResidual);
        computeRsOld(commandBuffer);

        for(auto itr = 0u; itr < params.numIterations; ++itr) {
            computeAp(commandBuffer);
            dot(commandBuffer, DotProductInput::PAp);
            computeAlpha(commandBuffer);

            x_plus_alpha_p(commandBuffer);
            r_minus_alpha_Ap(commandBuffer);

            dot(commandBuffer, DotProductInput::ResidualResidual);
            computeRsNew(commandBuffer);
            checkConvergence(commandBuffer);
            computeBeta(commandBuffer);

            r_plus_beta_p(commandBuffer);
        }
    }

    void ConjugateGradientSolver::axpy(VkCommandBuffer commandBuffer, VkDescriptorSet descriptorSet, float sign) {
        const auto gx = (cg.residual.sizeAs<float>() + 31u) / 32u;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.pipeline("axpy"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.layout("axpy"), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(commandBuffer, compute_.layout("axpy"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float), &sign);
        vkCmdDispatch(commandBuffer, gx, 1, 1);
        Barrier::computeWriteToRead(commandBuffer);
    }

    void ConjugateGradientSolver::computeResidual(VkCommandBuffer commandBuffer, const Params& params) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.pipeline("compute_residual"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.layout("compute_residual"), 0, static_cast<uint32_t>(computeResidualDescriptorSets_.size()), computeResidualDescriptorSets_.data(), 0, nullptr);
        vkCmdDispatch(commandBuffer, groupCount(params.Coefficients.numRows), 1, 1);
        Barrier::computeWriteToRead(commandBuffer);
    }

    void ConjugateGradientSolver::computeAp(VkCommandBuffer commandBuffer) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.pipeline("compute_ap"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.layout("compute_ap"), 0, static_cast<uint32_t>(computeResidualDescriptorSets_.size()), computeResidualDescriptorSets_.data(), 0, nullptr);
        vkCmdDispatch(commandBuffer, groupCount(constants_.numRows), 1, 1);
        Barrier::computeWriteToRead(commandBuffer);
    }

    void ConjugateGradientSolver::checkConvergence(VkCommandBuffer commandBuffer) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.pipeline("cg_check_convergence"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.layout("cg_check_convergence"), 0, static_cast<uint32_t>(computeResidualDescriptorSets_.size()), computeResidualDescriptorSets_.data(), 0, nullptr);
        vkCmdDispatch(commandBuffer, 1, 1, 1);
        Barrier::computeWriteToRead(commandBuffer);
    }

    void ConjugateGradientSolver::computeRsOld(VkCommandBuffer commandBuffer) {
        computeScalars(commandBuffer, 0);
    }

    void ConjugateGradientSolver::computeRsNew(VkCommandBuffer commandBuffer) {
        computeScalars(commandBuffer, 2);
    }

    void ConjugateGradientSolver::computeAlpha(VkCommandBuffer commandBuffer) {
        computeScalars(commandBuffer, 1);
    }

    void ConjugateGradientSolver::computeBeta(VkCommandBuffer commandBuffer) {
        computeScalars(commandBuffer, 3);
    }

    void ConjugateGradientSolver::computeScalars(VkCommandBuffer commandBuffer, int which) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.pipeline("cg_compute_scalars"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.layout("cg_compute_scalars"), 0, 1, &cgDescriptorSet_, 0, nullptr);
        vkCmdPushConstants(commandBuffer, compute_.layout("cg_compute_scalars"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(which), &which);
        vkCmdDispatch(commandBuffer, 1, 1, 1);
        Barrier::computeWriteToRead(commandBuffer);
    }

    void ConjugateGradientSolver::x_plus_alpha_p(VkCommandBuffer commandBuffer) {
        axpy(commandBuffer, axpyDescriptorSet0_, 1.0f);
    }

    void ConjugateGradientSolver::r_minus_alpha_Ap(VkCommandBuffer commandBuffer) {
        axpy(commandBuffer, axpyDescriptorSet1_, -1.0f);
    }

    void ConjugateGradientSolver::r_plus_beta_p(VkCommandBuffer commandBuffer) {
        axpy(commandBuffer, axpyDescriptorSet2_, 1.0f);
    }

    void ConjugateGradientSolver::dot(VkCommandBuffer commandBuffer, DotProductInput input) {
        const DotProductConstants constants{
            .count = constants_.numRows,
            .input = static_cast<uint32_t>(input)
        };

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.pipeline("cg_dot_product_terms"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.layout("cg_dot_product_terms"), 0, 1, &cgDescriptorSet_, 0, nullptr);
        vkCmdPushConstants(commandBuffer, compute_.layout("cg_dot_product_terms"), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(constants), &constants);
        vkCmdDispatch(commandBuffer, groupCount(constants.count), 1, 1);

        Barrier::computeWriteToTransferRead(commandBuffer, {cg.dotProductIntermediate});
        prefixSum_.accumulate(commandBuffer, cg.dotProductIntermediate, cg.dotProductResult, ::Operation::Add, ::DataType::Float);
        Barrier::computeWriteToRead(commandBuffer);
    }
}
