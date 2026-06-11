#pragma once

#include "common.h"
#include "Texture.h"
#include "ComputePipelins.hpp"
#include "VulkanDevice.h"
#include "Field.hpp"
#include "linalg/gpu/conjugate_gradient_solver.hpp"

#include <array>
#include <initializer_list>
#include <memory>
#include <optional>

namespace eular {

    enum class TimeDirection { Forward, Backword };

    enum class LinearSolverStrategy  {
        Jacobi, RBGS, ConjugateGradient
    };

    enum class BoundaryMode : uint32_t { VectorField, ScalarField };

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

        FluidSolver(VulkanDevice *device, VulkanDescriptorPool* descriptorPool, glm::vec2 gridSize,
                    std::optional<VkDescriptorSet> boundaryDescriptorSet = std::nullopt);

        void runSimulation(VkCommandBuffer commandBuffer);

        FluidSolver& density(float rho);

        float dt() const;

        float elapsedTime() const;

        VulkanDescriptorSetLayout fieldDescriptorSetLayout() const;

        std::vector<VulkanDescriptorSetLayout> forceFieldSetLayouts();

        std::vector<VulkanDescriptorSetLayout> sourceFieldSetLayouts();

        std::vector<VkDescriptorSet> debugFieldDescriptorSets() const;

        VectorField& vectorField();

        PressureField& pressureField();

    protected:
        void init();

        void initFields();

        void initConjugateGradientSupport();

        bool isJacobiSolver() const;

        bool isRbgsSolver() const;

        bool isConjugateGradientSolver() const;

        void createSamplers();

        void createDescriptorSetLayouts();

        void updateDescriptorSets();

        uint32_t createDescriptorSet(std::vector<VkWriteDescriptorSet>& writes, uint32_t writeOffset, Field& field);

        void createDefaultBoundaryTexture();

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

        void macCormackAdvect(VkCommandBuffer commandBuffer, Field& field, uint32_t boundaryMode = 0);

        void advect(VkCommandBuffer commandBuffer, Field& field, uint32_t boundaryMode = 0, bool addBarrier = true);

        void advect(VkCommandBuffer commandBuffer, VkDescriptorSet inDescriptor,
                    VkDescriptorSet outDescriptor, TimeDirection timeDirection = TimeDirection::Forward,
                    uint32_t boundaryMode = 0, Texture* writeTexture = nullptr);

        void clearForces(VkCommandBuffer commandBuffer);

        void applyForces(VkCommandBuffer commandBuffer);

        void applyExternalForces(VkCommandBuffer commandBuffer);

        void addForcesToVectorField(VkCommandBuffer commandBuffer, ForceField& sourceField);

        void computeVorticityConfinement(VkCommandBuffer commandBuffer);

        void computeVorticity(VkCommandBuffer commandBuffer);

        void applyVorticity(VkCommandBuffer commandBuffer);

        void diffuseVelocityField(VkCommandBuffer commandBuffer);

        void diffuse(VkCommandBuffer commandBuffer, Field& field, float rate, uint32_t vectorFieldComponent = 0);

        void project(VkCommandBuffer commandBuffer);

        void computeDivergence(VkCommandBuffer commandBuffer);

        void solvePressure(VkCommandBuffer commandBuffer);

        void computeDivergenceFreeField(VkCommandBuffer commandBuffer);

        void boundaryCheck(VkCommandBuffer commandBuffer, VectorField& field);

        void boundaryCheck(VkCommandBuffer commandBuffer, Field& field);

        void addComputeBarrier(VkCommandBuffer commandBuffer, Texture& texture);

        void addComputeBarrier(VkCommandBuffer commandBuffer, std::initializer_list<Texture*> textures);

        void jacobiSolver(VkCommandBuffer commandBuffer, Field& solution, Field& unknown);

        void rbgsSolver(VkCommandBuffer commandBuffer, Field& solution, Field& unknown);

        void conjugateGradientSolve(VkCommandBuffer commandBuffer, uint32_t index);

        void buildCoefficientMatrix(VkCommandBuffer commandBuffer, uint32_t index);

        void setDiffuseConstants(uint32_t index, float rate, uint32_t vectorFieldComponent);

        void setPressureConstants(uint32_t index);

        void assign(VkCommandBuffer commandBuffer, Texture& from, VulkanBuffer& to);

        void assignScaled(VkCommandBuffer commandBuffer, Field& from, VulkanBuffer& to, VkDescriptorSet toDescriptorSet, float scale);

        void assign(VkCommandBuffer commandBuffer, VulkanBuffer& from, Texture& to);

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
        VulkanDescriptorSetLayout _boundaryDescriptorSetLayout;
        VulkanDescriptorSetLayout _debugDescriptorSetLayout;
        VkDescriptorSet _boundaryDescriptorSet{};
        Texture _defaultBoundaryTexture;
        bool _useDefaultBoundaryTexture{true};

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
            uint vector_field_component{};
            uint pass{0};
        } linearSolverConstants;

        struct {
            float time_sign{1};
            uint32_t boundary_mode{0};
        } advectConstants;


        glm::uvec3 _groupCount{1};
        VulkanDescriptorSetLayout uniformsSetLayout;
        VkDescriptorSet uniformDescriptorSet{};

        VulkanSampler _valueSampler;
        VulkanSampler _linearSampler;

        struct ScaledFieldCopyConstants {
            float scale{};
            uint32_t count{};
        };



        struct {
            gpu::linalg::AbstractSolver::Params params{};
            gpu::linalg::ConjugateGradientSolver solver;
            VkDescriptorSet descriptorSet{};
            VkDescriptorSet rhsDescriptorSet{};

            struct {
                glm::uvec2 gridSize{};
                glm::vec2 alpha{};
                float identity{};
                uint32_t batchOffset{};
                uint32_t batchSize{};
                uint32_t ensureBoundaryCondition{};
                uint32_t vectorFieldComponent{};
            } constants;
        } _cg[2];

        VulkanDescriptorSetLayout cgDescriptorSetLayout;
        VulkanDescriptorSetLayout cgVectorDescriptorSetLayout;

        static constexpr uint32_t cgStencilEntriesPerRow = 5;
        static constexpr uint32_t cgRowsPerBatch = 4096;

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

        Builder& diffuseIterations(int value);

        Builder& viscosity(float value);

        Builder& ensureBoundaryCondition(bool flag);

        Builder& vorticityConfinementScale(float scale);

        Builder& add(Quantity &quantity);

        Builder& gridSize(glm::vec2 size);

        Builder& boundary(VkDescriptorSet descriptorSet);

        Builder& enableProjection();

        Builder& disableProjection();

        Builder& useMacCormackAdvection();

        Builder& useStandingAdvection();

        Builder& enableAdvection();

        Builder& disableAdvection();

        Builder& useJacobiSolver();

        Builder& useConjugateGradientSolver();

        Builder& useGaussSeidelSolver();

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
        int _diffuseIterations = 30;
        float _viscosity = 0;
        float _vorticityConfinementScale{0};
        float _density{1};
        float _dt{1.0f / 120.f};
        glm::vec2 _gridSize{0};
        std::vector<std::reference_wrapper<Quantity>> _quantities;
        LinearSolverStrategy _linearSolverStrategy{LinearSolverStrategy::RBGS};

        std::vector<ExternalForce> _externalForces;
        std::optional<VectorFieldFunc2D> _generator{[](float, float) { return glm::vec2{0.0f}; }};
        std::optional<VkDescriptorSet> _boundaryDescriptorSet;
    };
}
