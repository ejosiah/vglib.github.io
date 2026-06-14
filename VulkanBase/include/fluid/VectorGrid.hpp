#pragma once

#include "common.hpp"
#include "ComputePipelins.hpp"
#include "VulkanDevice.h"

#include <initializer_list>

namespace eular {

    class VectorGrid : public ComputePipelines {
    public:
        struct Params {
            VulkanDevice* device{};
            VulkanDescriptorPool* descriptorPool{};
            glm::vec2 gridSize{0.0f};
            VkDescriptorSet globalConstantsDescriptorSet{};
            VulkanDescriptorSetLayout* globalConstantsSetLayout{};
            VkDescriptorSet boundaryDescriptorSet{};
            VulkanDescriptorSetLayout* boundaryDescriptorSetLayout{};
            bool macCormackAdvection{};
            bool ensureBoundaryCondition{true};
        };

        VectorGrid() = default;

        explicit VectorGrid(const Params& params);

        virtual ~VectorGrid() = default;

        virtual void init();

        VectorField& vectorField();

        const VectorField& vectorField() const;

        DivergenceField& divergenceField();

        const DivergenceField& divergenceField() const;

        ForceField& forceField();

        const ForceField& forceField() const;

        VulkanDescriptorSetLayout fieldDescriptorSetLayout() const;

        virtual void advectVectorField(VkCommandBuffer commandBuffer) = 0;

        virtual void advect(VkCommandBuffer commandBuffer, Field& field, uint32_t boundaryMode = 0,
            bool addBarrier = true) = 0;

        virtual void advect(VkCommandBuffer commandBuffer, VkDescriptorSet inDescriptor,
            VkDescriptorSet outDescriptor, TimeDirection timeDirection = TimeDirection::Forward,
            uint32_t boundaryMode = 0, Texture* writeTexture = nullptr) = 0;

        virtual void computeDivergence(VkCommandBuffer commandBuffer) = 0;

        virtual void computeDivergenceFreeField(VkCommandBuffer commandBuffer, PressureField& pressureField) = 0;

        virtual void addForcesToVectorField(VkCommandBuffer commandBuffer) = 0;

        virtual void fill(VectorFieldFunc2D generator) = 0;

    protected:
        void initFields();

        void createSamplers();

        void createDescriptorSetLayouts();

        void updateDescriptorSets();

        uint32_t createDescriptorSet(std::vector<VkWriteDescriptorSet>& writes, uint32_t writeOffset, Field& field);

        void prepTextures();

        void macCormackAdvect(VkCommandBuffer commandBuffer, Field& field, uint32_t boundaryMode = 0);

        VulkanDevice* _device{};
        VulkanDescriptorPool* _descriptorPool{};
        VectorField _vectorField;
        DivergenceField _divergenceField;
        ForceField _forceField;
        Field _macCormackData;

        VulkanDescriptorSetLayout _samplerDescriptorSetLayout;

        VkDescriptorSet _globalConstantsDescriptorSet{};
        VkDescriptorSet _linearSamplerDescriptorSet{};
        VkDescriptorSet _boundaryDescriptorSet{};

        VulkanDescriptorSetLayout* _globalConstantsSetLayout{};
        VulkanDescriptorSetLayout* _boundaryDescriptorSetLayout{};

        VulkanSampler _linearSampler;

        VkImageType _imageType{VK_IMAGE_TYPE_2D};
        glm::vec3 _gridSize{};
        glm::uvec3 _groupCount{1};
        bool _macCormackAdvection{};
        bool _ensureBoundaryCondition{true};

        struct {
            float time_sign{1.0f};
            uint32_t boundary_mode{};
        } advectConstants;
    };
}
