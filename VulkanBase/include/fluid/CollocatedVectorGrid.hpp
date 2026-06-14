#pragma once

#include "VectorGrid.hpp"

namespace eular {
    class CollocatedVectorGrid : public VectorGrid {
    public:
        CollocatedVectorGrid() = default;

        explicit CollocatedVectorGrid(const Params& params)
            : VectorGrid(params) {
        }

        ~CollocatedVectorGrid() override = default;

        void advectVectorField(VkCommandBuffer commandBuffer) override;

        void advect(VkCommandBuffer commandBuffer, Field& field, uint32_t boundaryMode = 0,
            bool addBarrier = true) override;

        void advect(VkCommandBuffer commandBuffer, VkDescriptorSet inDescriptor, VkDescriptorSet outDescriptor,
            TimeDirection timeDirection = TimeDirection::Forward, uint32_t boundaryMode = 0,
            Texture* writeTexture = nullptr) override;

        void computeDivergence(VkCommandBuffer commandBuffer) override;

        void computeDivergenceFreeField(VkCommandBuffer commandBuffer, PressureField& pressureField) override;

        void addForcesToVectorField(VkCommandBuffer commandBuffer) override;

        void fill(VectorFieldFunc2D generator) override;

    protected:
        std::vector<PipelineMetaData> pipelineMetaData() override;
    };
}
