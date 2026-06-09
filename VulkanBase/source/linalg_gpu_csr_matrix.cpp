#include "linalg/gpu/csr_matrix.hpp"

#include "glsl_shaders.hpp"

#include <array>

namespace gpu::linalg {
    CSRMatrixBuilder::CSRMatrixBuilder(VulkanDevice &device)
    : device_{&device} {}

    void CSRMatrixBuilder::init(CSRMatrix &matrix, VulkanBuffer source, VulkanBuffer flags) {
        createDescriptorSetLayout();
        const std::array<VkDescriptorPoolSize, 1> poolSizes{{
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7},
        }};
        descriptorPool_ = device_->createDescriptorPool(1, poolSizes);
        descriptorSet_ = descriptorPool_.allocate({descriptorSetLayout_}).front();
        offsets_ = device_->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY, flags.size, "csr_matrix_builder_offsets");
        updateDescriptorSet(matrix, source, flags);
        compute_ = ComputePipelines{device_, pipelines()};
        compute_.createPipelines();
        prefixSum_ = PrefixSum{device_};
        prefixSum_.init();
    }

    void CSRMatrixBuilder::build(VkCommandBuffer cmd, CSRMatrix &matrix, const VulkanBuffer &source, const VulkanBuffer &flags) {
        clearBuffers(cmd, matrix);
        assign(cmd, flags, offsets_);
        scan(cmd, offsets_);
        build(cmd, matrix, (source.sizeAs<float>() + 31u)/32u);
        scan(cmd, matrix.rowOffsets);
    }

    void CSRMatrixBuilder::clearBuffers(VkCommandBuffer cmd, CSRMatrix &matrix) {
        vkCmdFillBuffer(cmd, matrix.values, 0, matrix.values.size, 0u);
        vkCmdFillBuffer(cmd, matrix.colIndices, 0, matrix.colIndices.size, 0u);
        vkCmdFillBuffer(cmd, matrix.rowOffsets, 0, matrix.rowOffsets.size, 0u);
        vkCmdFillBuffer(cmd, matrix.counts, sizeof(uint32_t) * 2, sizeof(uint32_t), 0u);
        Barrier::transferWriteToComputeRead(cmd);
    }

    void CSRMatrixBuilder::build(VkCommandBuffer cmd, CSRMatrix &matrix, uint32_t numEntries) {
        const auto gx = numEntries;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.pipeline("build_csr_matrix"));
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_.layout("build_csr_matrix"), 0, 1, &descriptorSet_, 0, nullptr);
        vkCmdDispatch(cmd, gx, 1, 1);
        Barrier::computeWriteToRead(cmd);
    }

    void CSRMatrixBuilder::assign(VkCommandBuffer commandBuffer, VulkanBuffer from, VulkanBuffer to) {
        device_->copy(commandBuffer, from, to, to.size);
        Barrier::transferWriteToComputeRead(commandBuffer);
    }

    void CSRMatrixBuilder::createDescriptorSetLayout() {
        descriptorSetLayout_ =
            device_->descriptorSetLayoutBuilder()
                .name("csr_matrix_builder_descriptor_set_layout")
                .binding(0) // CSR matrix values
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(1) // CSR column indices
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(2) // CSR row offsets
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(3) // CSR counts
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(4) // source
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(5) // flags
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(6) // offsets
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .createLayout();
    }

    void CSRMatrixBuilder::updateDescriptorSet(CSRMatrix& matrix, VulkanBuffer source, VulkanBuffer flags) {
        const std::array<VkDescriptorBufferInfo, 7> infos{{
            {matrix.values, 0, VK_WHOLE_SIZE},
            {matrix.colIndices, 0, VK_WHOLE_SIZE},
            {matrix.rowOffsets, 0, VK_WHOLE_SIZE},
            {matrix.counts, 0, VK_WHOLE_SIZE},
            {source, 0, VK_WHOLE_SIZE},
            {flags, 0, VK_WHOLE_SIZE},
            {offsets_, 0, VK_WHOLE_SIZE},
        }};

        auto writes = initializers::writeDescriptorSets<7>();
        for(uint32_t binding = 0; binding < infos.size(); ++binding) {
            writes[binding].dstSet = descriptorSet_;
            writes[binding].dstBinding = binding;
            writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[binding].descriptorCount = 1;
            writes[binding].pBufferInfo = &infos[binding];
        }

        device_->updateDescriptorSets(writes);
    }

    void CSRMatrixBuilder::scan(VkCommandBuffer commandBuffer, const VulkanBuffer& input) {
        prefixSum_(commandBuffer, input);
    }

    std::vector<PipelineMetaData> CSRMatrixBuilder::pipelines() {
        return {{
            {
                .name = "build_csr_matrix",
                .shadePath = data_shaders_linalg_build_csr_matrix_comp,
                .layouts = {&descriptorSetLayout_},
            },
        }};
    }
}
