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
            VkDescriptorSet colliderDescriptorSet{};
            VulkanDescriptorSetLayout* colliderDescriptorSetLayout{};
            bool macCormackAdvection{};
            bool wrappingEnabled{};
        };

        VectorGrid() = default;

        explicit VectorGrid(const Params& params);

        ~VectorGrid() override;

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

        virtual void generate(VectorFieldFunc2D generator) = 0;

        void fill(std::span<glm::vec2> vectorField);

        void fill(glm::vec2 value);

    protected:
        void initFields();

        void createSamplers();

        void createDescriptorSetLayouts();

        void updateDescriptorSets();

        uint32_t createDescriptorSet(std::vector<VkWriteDescriptorSet>& writes, uint32_t writeOffset, Field& field);

        void prepTextures();

        void macCormackAdvect(VkCommandBuffer commandBuffer, Field& field, uint32_t boundaryMode = 0);

        void releaseDescriptorSets();

        void releaseDescriptorSet(VkDescriptorSet& descriptorSet);

        void releaseFieldDescriptorSets(Field& field);

        VulkanDevice* _device{};
        VulkanDescriptorPool* _descriptorPool{};
        VectorField _vectorField;
        DivergenceField _divergenceField;
        ForceField _forceField;
        Field _macCormackData;

        VulkanDescriptorSetLayout _samplerDescriptorSetLayout;

        VkDescriptorSet _globalConstantsDescriptorSet{};
        VkDescriptorSet _linearSamplerDescriptorSet{};
        VkDescriptorSet _colliderDescriptorSet{};

        VulkanDescriptorSetLayout* _globalConstantsSetLayout{};
        VulkanDescriptorSetLayout* _colliderDescriptorSetLayout{};

        VulkanSampler _linearSampler;

        VkImageType _imageType{VK_IMAGE_TYPE_2D};
        glm::vec3 _gridSize{};
        glm::uvec3 _groupCount{1};
        bool _macCormackAdvection{};
        bool _wrappingEnabled{true};

        struct {
            float time_sign{1.0f};
            uint32_t boundary_mode{};
        } advectConstants;
    };
}
