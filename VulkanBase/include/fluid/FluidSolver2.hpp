#pragma once

#include "common.h"
#include "Texture.h"
#include "ComputePipelins.hpp"
#include "VulkanDevice.h"
#include "Field.hpp"

#include <memory>
#include <optional>

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

        class Builder;

        FluidSolver() = default;

        FluidSolver(VulkanDevice *device, VulkanDescriptorPool* descriptorPool, glm::vec2 gridSize);

        void runSimulation(VkCommandBuffer commandBuffer);

        FluidSolver& density(float rho);

        float dt() const;

        float elapsedTime() const;

        VulkanDescriptorSetLayout fieldDescriptorSetLayout() const;

        std::vector<VulkanDescriptorSetLayout> forceFieldSetLayouts();

        std::vector<VulkanDescriptorSetLayout> sourceFieldSetLayouts();

        VectorField& vectorField();

        PressureField& pressureField();

    protected:
        void init();

        void initFields();

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

    private:
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

        struct GlobalData {
            glm::ivec2 grid_size{0};
            glm::vec2 dx{1};
            glm::vec2 dy{1};
            float dt{1.0f / 120.f};
            float density{1};
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
            bool ensureBoundaryCondition = true;
            int poissonIterations = 30;
            float viscosity = 0;
            float vorticityConfinementScale{0};
            float density{1};
            float timeStep{1.0f / 120.f};
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

    class FluidSolver::Builder {
    public:
        Builder(VulkanDevice *device, VulkanDescriptorPool* descriptorPool);

        Builder& dt(float value);

        Builder& density(float rho);

        Builder& generate(const VectorFieldFunc2D& func);

        Builder& add(ExternalForce&& force);

        Builder& poissonIterations(int value);

        Builder& viscosity(float value);

        Builder& ensureBoundaryCondition(bool flag);

        Builder& poissonEquationSolver(LinearSolverStrategy strategy);

        Builder& vorticityConfinementScale(float scale);

        Builder& add(Quantity &quantity);

        Builder& gridSize(glm::vec2 size);

        std::unique_ptr<FluidSolver> build();

    private:
        void generateVectorField(FluidSolver& solver);

        void addQuantities(FluidSolver& solver);

        VulkanDevice *_device{};
        VulkanDescriptorPool* _descriptorPool{};
        bool _advectVField = true;
        bool _macCormackAdvection = false;
        bool _project = true;
        bool _ensureBoundaryCondition = true;
        int _poissonIterations = 30;
        float _viscosity = 0;
        float _vorticityConfinementScale{0};
        float _density{1};
        float _dt{1.0f / 120.f};
        glm::vec2 _gridSize{0};
        std::vector<std::reference_wrapper<Quantity>> _quantities;
        LinearSolverStrategy _linearSolverStrategy{LinearSolverStrategy::Jacobi};

        std::vector<ExternalForce> _externalForces;
        std::optional<VectorFieldFunc2D> _generator;
    };
}