#pragma once

#include "common.h"
#include "Texture.h"
#include "ComputePipelins.hpp"
#include "VulkanDevice.h"
#include "Field.hpp"
#include "VectorGrid.hpp"
#include "common.hpp"
#include "linalg/gpu/conjugate_gradient_solver.hpp"
#include "linalg/gpu/jacobi_solver.hpp"
#include "linalg/gpu/red_black_gauss_seidel_solver.hpp"

#include <array>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>

#include "Collider.hpp"

class FieldVisualizer;

namespace eular {

    class FluidSolver : public ComputePipelines {
        friend class ::FieldVisualizer;

    public:

        class Builder;
        static constexpr uint32_t maxColliderFields = 10;
        enum BoundaryEdge : uint32_t {
            BoundaryEdgeLeft = 1u << 0,
            BoundaryEdgeRight = 1u << 1,
            BoundaryEdgeBottom = 1u << 2,
            BoundaryEdgeTop = 1u << 3,
            BoundaryEdgeAll = BoundaryEdgeLeft | BoundaryEdgeRight | BoundaryEdgeBottom | BoundaryEdgeTop
        };

        FluidSolver() = default;

        FluidSolver(VulkanDevice *device, VulkanDescriptorPool* descriptorPool, glm::vec2 gridSize);

        ~FluidSolver() override;

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

        Field& colliderField();

        const Field& colliderField() const;

        Field& colliderVelocityField();

        const Field& colliderVelocityField() const;

        Texture& colliderTexture();

        const Texture& colliderTexture() const;

        Texture& colliderVelocityTexture();

        const Texture& colliderVelocityTexture() const;

        void setColliders(std::span<const Collider> colliders);

        uint32_t activeColliderCount() const;

        FluidSolver& closedDomain(bool flag);

        FluidSolver& openBoundaryEdges(uint32_t flags);

    protected:
        void init();

        void initVectorGrid();

        void initFields();

        void initLinearSolverSupport();

        void createSamplers();

        void createDescriptorSetLayouts();

        void updateDescriptorSets();

        void updateFieldDescriptorSets();

        void updateSourceColliderDescriptorSet();

        void ensureSourceColliderDescriptorSet();

        void ensureZeroColliderVelocityTexture();

        bool hasActiveColliders() const;

        uint32_t createDescriptorSet(std::vector<VkWriteDescriptorSet>& writes, uint32_t writeOffset, Field& field);

        void createDefaultColliderFields();

        void updateColliderFields(VkCommandBuffer commandBuffer);

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

        void applyBoundaryConditions(VkCommandBuffer commandBuffer);

        void constrainVelocity(VkCommandBuffer commandBuffer);

        void advect(VkCommandBuffer commandBuffer, Field& field, uint32_t boundaryMode = 0, bool addBarrier = true);

        void advect(VkCommandBuffer commandBuffer, VkDescriptorSet inDescriptor,
                    VkDescriptorSet outDescriptor, TimeDirection timeDirection = TimeDirection::Forward,
                    uint32_t boundaryMode = 0, Texture* writeTexture = nullptr);

        void clearForces(VkCommandBuffer commandBuffer);

        void applyForces(VkCommandBuffer commandBuffer);

        void applyExternalForces(VkCommandBuffer commandBuffer);

        void computeVorticityConfinement(VkCommandBuffer commandBuffer);

        void computeVorticity(VkCommandBuffer commandBuffer);

        void applyVorticity(VkCommandBuffer commandBuffer);

        void diffuseVelocityField(VkCommandBuffer commandBuffer);

        void diffuse(VkCommandBuffer commandBuffer, Field& field, float rate, uint32_t vectorFieldComponent = 0);

        void project(VkCommandBuffer commandBuffer);

        void clearPressureField(VkCommandBuffer commandBuffer);

        void computeDivergence(VkCommandBuffer commandBuffer);

        void solvePressure(VkCommandBuffer commandBuffer);

        void subtractMeanDrift(VkCommandBuffer commandBuffer, Field& field);

        void computeDivergenceFreeField(VkCommandBuffer commandBuffer);

        void addComputeBarrier(VkCommandBuffer commandBuffer, Texture& texture);

        void addComputeBarrier(VkCommandBuffer commandBuffer, std::initializer_list<Texture*> textures);

        void solveLinearSystem(VkCommandBuffer commandBuffer, uint32_t index);

        void buildCoefficientMatrix(VkCommandBuffer commandBuffer, uint32_t index);

        void setDiffuseConstants(uint32_t index, float rate, uint32_t vectorFieldComponent);

        void setPressureConstants(uint32_t index);

        void assign(VkCommandBuffer commandBuffer, Texture& from, VulkanBuffer& to);

        void assignScaled(VkCommandBuffer commandBuffer, Field& from, VulkanBuffer& to, VkDescriptorSet toDescriptorSet, float scale);

        void assign(VkCommandBuffer commandBuffer, VulkanBuffer& from, Texture& to);

        std::vector<PipelineMetaData> pipelineMetaData() final;

        void prepTextures();

        static void clear(VkCommandBuffer commandBuffer, Texture& texture);

        void releaseDescriptorSets();

        void releaseDescriptorSet(VkDescriptorSet& descriptorSet);

        void releaseFieldDescriptorSets(Field& field);

    private:
        VulkanDescriptorPool* _descriptorPool{};

        PressureField _pressureField;
        VorticityField _vorticityField;
        std::unique_ptr<VectorGrid> _vectorGrid;

        VulkanDescriptorSetLayout _fieldDescriptorSetLayout;
        VulkanDescriptorSetLayout _colliderDescriptorSetLayout;
        VulkanDescriptorSetLayout _debugDescriptorSetLayout;
        VkDescriptorSet _colliderDescriptorSet{};
        VkDescriptorSet _sourceColliderDescriptorSet{};
        Field _colliderField;
        Field _colliderVelocityField;
        Texture _zeroColliderVelocityTexture;
        std::array<VkSampler, maxColliderFields> _colliderSamplers{};

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
            uint32_t wrapping_enabled{0};
            uint32_t use_hermite{0};
        };

        struct {
            VulkanBuffer gpu;
            GlobalData* cpu{};
        } globalConstants;

        struct {
            uint32_t openBoundaryEdges = 0;
            int poissonIterations = 30;
            float viscosity = 0;
            float vorticityConfinementScale{0};
            float density{1};
            float timeStep{1.0f / 120.f};
            bool advectVField = true;
            bool macCormackAdvection = false;
            bool project = true;
            bool wrappingEnabled = true;
            bool closedDomain = false;
        } options;

        struct {
            float time_sign{1};
            uint32_t boundary_mode{0};
        } advectConstants;

        struct {
            uint32_t colliderCount{};
            uint32_t closedDomain{};
            uint32_t openBoundaryEdges{};
        } updateColliderConstants;


        glm::uvec3 _groupCount{1};
        VulkanDescriptorSetLayout uniformsSetLayout;
        VkDescriptorSet uniformDescriptorSet{};

        VulkanSampler _valueSampler;

        struct ScaledFieldCopyConstants {
            float scale{};
            uint32_t count{};
        };

        struct MeanDriftStats {
            float mean{};
            uint32_t count{};
        };

        struct MeanDriftConstants {
            uint32_t count{};
        };

        struct LinearSystem {
            gpu::linalg::AbstractSolver::Params params{};
            std::unique_ptr<gpu::linalg::AbstractSolver> solver;
            VkDescriptorSet descriptorSet{};
            VkDescriptorSet rhsDescriptorSet{};

            struct {
                glm::uvec2 gridSize{};
                glm::vec2 alpha{};
                float identity{};
                uint32_t batchOffset{};
                uint32_t batchSize{};
                uint32_t vectorFieldComponent{};
            } constants;
        } _linearSystems[2];

        VulkanDescriptorSetLayout linearSystemDescriptorSetLayout;
        VulkanDescriptorSetLayout linearSystemVectorDescriptorSetLayout;
        VulkanDescriptorSetLayout meanDriftDescriptorSetLayout;
        VkDescriptorSet meanDriftDescriptorSet{};
        VulkanBuffer meanDriftBuffer;

        static constexpr uint32_t linearSystemStencilEntriesPerRow = 5;
        static constexpr uint32_t linearSystemRowsPerBatch = 4096;

        std::vector<ExternalForce> _externalForces;
        std::vector<Collider> _colliders;
        uint32_t _activeColliderCount{};
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

        Builder& enableWrapping();

        Builder& disableWrapping();

        Builder& vorticityConfinementScale(float scale);

        Builder& add(Quantity &quantity);

        Builder& gridSize(glm::vec2 size);

        Builder& closedDomain();

        Builder& openDomain();

        Builder& openBoundaryEdges(uint32_t flags);

        Builder& addCollider(VkDescriptorSet fieldDescriptorSet, VkDescriptorSet velocityDescriptorSet = VK_NULL_HANDLE);

        Builder& addCollider(const Field& field, VkDescriptorSet velocityDescriptorSet = VK_NULL_HANDLE);

        Builder& enableProjection();

        Builder& disableProjection();

        Builder& useMacCormackAdvection();

        Builder& useStandingAdvection();

        Builder& enableAdvection();

        Builder& disableAdvection();

        Builder& useJacobiSolver();

        Builder& useConjugateGradientSolver();

        Builder& useGaussSeidelSolver();

        Builder& vectorField(std::span<glm::vec2> field);

        std::unique_ptr<FluidSolver> build();

    private:
        void generateVectorField(FluidSolver& solver);

        void addQuantities(FluidSolver& solver);

        VulkanDevice *_device{};
        VulkanDescriptorPool* _descriptorPool{};
        bool _advectVField = true;
        bool _macCormackAdvection = false;
        bool _project = true;
        bool _wrappingEnabled = false;
        bool _closedDomain = false;
        uint32_t _openBoundaryEdges = 0;
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
        std::optional<VectorFieldFunc2D> _generator{};
        std::vector<Collider> _colliders;
        std::vector<glm::vec2> _data;
    };
}
