#pragma once

#include "common.h"
#include "Texture.h"
#include "ComputePipelins.hpp"
#include "VulkanDevice.h"
#include <variant>
#include "Field.hpp"

namespace eular {

    enum class TimeDirection { Forward, Backword };

    enum class LinearSolverStrategy  {
        Jacobi, RBGS
    };


    using VectorFieldSource3D = std::vector<glm::vec3>;
    using VectorFieldSource2D = std::vector<glm::vec2>;

    using DivergenceField = Field;
    using PressureField = Field;
    using ForceField = Field;
    using VorticityField = Field;

    using ExternalForce = std::function<void(VkCommandBuffer, std::span<VkDescriptorSet>, glm::uvec3)>;

    class FluidSolver : public ComputePipelines {
    public:

        FluidSolver() = default;

        FluidSolver(VulkanDevice *device, VulkanDescriptorPool* descriptorPool, glm::vec2 gridSize);

        void init();

        void initFields();

        FluidSolver& set(VectorFieldSource2D vectorField);

        FluidSolver& set(VectorFieldSource3D vectorField);

        FluidSolver& add(ExternalForce&& force);

        void runSimulation(VkCommandBuffer commandBuffer);

        void add(Quantity &quantity);

        FluidSolver& dt(float value);

        float dt() const;

        float elapsedTime() const;

        FluidSolver& poissonIterations(int value);

        FluidSolver& viscosity(float value);

        FluidSolver& ensureBoundaryCondition(bool flag);

        FluidSolver& enableVorticity(bool flag);

        FluidSolver& poissonEquationSolver(LinearSolverStrategy strategy);

        VulkanDescriptorSetLayout fieldDescriptorSetLayout() const;

        std::vector<VulkanDescriptorSetLayout> forceFieldSetLayouts();
        std::vector<VulkanDescriptorSetLayout> sourceFieldSetLayouts();

    public:
        void createSamplers();

        void createDescriptorSetLayouts();

        void updateDescriptorSets();

        uint32_t createDescriptorSet(std::vector<VkWriteDescriptorSet>& writes, uint32_t writeOffset, Field& field);

        void initGlobalConstants();

        void velocityStep(VkCommandBuffer commandBuffer);

        void quantityStep(VkCommandBuffer commandBuffer);

        void quantityStep(VkCommandBuffer commandBuffer, Quantity& quantity);

        void clearSources(VkCommandBuffer commandBuffer, Quantity& quantity);

        void updateSources(VkCommandBuffer commandBuffer, Quantity& quantity);

        void addSource(VkCommandBuffer commandBuffer, Quantity& quantity);

        void diffuseQuantity(VkCommandBuffer commandBuffer, Quantity& quantity);

        void advectQuantity(VkCommandBuffer commandBuffer, Quantity& quantity);

        void postAdvection(VkCommandBuffer commandBuffer, Quantity& quantity);

        void advectVectorField(VkCommandBuffer commandBuffer);

        void macCormackAdvect(VkCommandBuffer commandBuffer, Field& field);

        void advect(VkCommandBuffer commandBuffer, Field& field);

        void advect(VkCommandBuffer commandBuffer, VkDescriptorSet inDescriptor,
                    VkDescriptorSet outDescriptor, TimeDirection timeDirection = TimeDirection::Forward);

        void clearForces(VkCommandBuffer commandBuffer);

        void applyForces(VkCommandBuffer commandBuffer);

        void applyExternalForces(VkCommandBuffer commandBuffer);

        void addForcesToVectorField(VkCommandBuffer commandBuffer, ForceField& sourceField);

        void computeVorticityConfinement(VkCommandBuffer commandBuffer);

        void computeVorticity(VkCommandBuffer commandBuffer);

        void applyVorticity(VkCommandBuffer commandBuffer);

        void diffuseVelocityField(VkCommandBuffer commandBuffer);

        void diffuse(VkCommandBuffer commandBuffer, Field& field, float rate);

        void project(VkCommandBuffer commandBuffer);

        void computeDivergence(VkCommandBuffer commandBuffer);

        void solvePressure(VkCommandBuffer commandBuffer);

        void computeDivergenceFreeField(VkCommandBuffer commandBuffer);

        void addComputeBarrier(VkCommandBuffer commandBuffer);

        void jacobiSolver(VkCommandBuffer commandBuffer, Field& solution, Field& unknown);

        void rbgsSolver(VkCommandBuffer commandBuffer, Field& solution, Field& unknown);

        std::vector<PipelineMetaData> pipelineMetaData() final;

        void prepTextures();

        static void clear(VkCommandBuffer commandBuffer, Texture& texture);

    public:
        VulkanDescriptorPool* _descriptorPool{};

        VectorField _vectorField;
        DivergenceField _divergenceField;
        PressureField _pressureField;
        ForceField _forceField;
        VorticityField _vorticityField;
        Field _macCormackData;

        VulkanDescriptorSetLayout _fieldDescriptorSetLayout;
        VulkanDescriptorSetLayout _imageDescriptorSetLayout;
        VulkanDescriptorSetLayout _textureDescriptorSetLayout;
        VulkanDescriptorSetLayout _samplerDescriptorSetLayout;
        VulkanDescriptorSetLayout _debugDescriptorSetLayout;

        std::vector<std::reference_wrapper<Quantity>> _quantities;
        VkImageType _imageType{};

        glm::vec3 _gridSize{};
        glm::vec3 _delta{};
        float _timeStep{1.0f / 120.f};

        struct GlobalData {
            glm::ivec2 grid_size{0};
            glm::vec2 dx{1};
            glm::vec2 dy{1};
            float dt{1.0f / 120.f};
            uint32_t ensure_boundary_condition{1};
            uint32_t use_hermite{0};
        };

        struct {
            VulkanBuffer gpu;
            GlobalData* cpu{};
        } globalConstants;

        struct {
            bool advectVField = true;
            bool macCormackAdvection = false;
            bool project = true;
            bool vorticity = false;
            int poissonIterations = 30;
            float viscosity = 0;
            float vorticityConfinementScale{1};
        } options;

        struct {
            float alpha{};
            float rBeta{};
            uint is_vector_field{};
            uint pass{0};
        } linearSolverConstants;


        glm::uvec3 _groupCount{1};
        VulkanDescriptorSetLayout uniformsSetLayout;
        VkDescriptorSet uniformDescriptorSet{};

        VulkanSampler _valueSampler;
        VulkanSampler _linearSampler;

        VkDescriptorSet _valueSamplerDescriptorSet{};
        VkDescriptorSet _linearSamplerDescriptorSet{};

        static constexpr uint32_t in = 0;
        static constexpr uint32_t out = 1;

        std::vector<ExternalForce> _externalForces;
        float _elapsedTime{};
        LinearSolverStrategy linearSolverStrategy{LinearSolverStrategy::Jacobi};
    };
}