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
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "Collider.hpp"

class FieldVisualizer;

namespace eular {

    template<typename T>
    struct QuantityTexelFormat;

    template<>
    struct QuantityTexelFormat<float> {
        static constexpr VkFormat value = VK_FORMAT_R32_SFLOAT;
    };

    template<>
    struct QuantityTexelFormat<glm::vec2> {
        static constexpr VkFormat value = VK_FORMAT_R32G32_SFLOAT;
    };

    template<>
    struct QuantityTexelFormat<glm::vec4> {
        static constexpr VkFormat value = VK_FORMAT_R32G32B32A32_SFLOAT;
    };

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
            BoundaryEdgeBack = 1u << 4,
            BoundaryEdgeFront = 1u << 5,
            BoundaryEdgeAll2D = BoundaryEdgeLeft | BoundaryEdgeRight | BoundaryEdgeBottom | BoundaryEdgeTop,
            BoundaryEdgeAll = BoundaryEdgeAll2D | BoundaryEdgeBack | BoundaryEdgeFront
        };

        FluidSolver() = default;

        FluidSolver(VulkanDevice *device, VulkanDescriptorPool* descriptorPool, glm::vec2 gridSize);

        FluidSolver(VulkanDevice *device, VulkanDescriptorPool* descriptorPool, glm::vec3 gridSize,
                    uint32_t dimension = 3);

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
        uint32_t _dimension{3};

        struct GlobalData {
            glm::ivec3 grid_size{0};
            glm::vec3 dx{1};
            glm::vec3 dy{1};
            glm::vec3 dz{1};
            float dt{1.0f / 120.f};
            float density{1};
            uint32_t wrapping_enabled{0};
            uint32_t use_hermite{0};
            uint32_t dimension{3};
            uint32_t open_boundary_edges{0};
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
                glm::uvec3 gridSize{};
                glm::vec3 alpha{};
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

        Builder& generate(const VectorFieldFunc3D& func);

        Builder& generate2D(const VectorFieldFunc2D& func);

        Builder& generate3D(const VectorFieldFunc3D& func);

        Builder& addExternalForce(ExternalForce&& force);

        Builder& poissonIterations(int value);

        Builder& diffuseIterations(int value);

        Builder& viscosity(float value);

        Builder& enableWrapping();

        Builder& disableWrapping();

        Builder& vorticityConfinementScale(float scale);

        Builder& addQuantity(Quantity &quantity);

        template<typename T>
        Builder& addQuantity(Quantity& quantity, const std::string& name, std::span<const T> data) {
            static_assert(std::is_trivially_copyable_v<T>, "Quantity data must be trivially copyable");
            return addQuantityData(quantity, name, QuantityTexelFormat<T>::value, std::as_bytes(data));
        }

        template<typename T>
        Builder& addQuantity(Quantity& quantity, const std::string& name, const std::vector<T>& data) {
            return addQuantity(quantity, name, std::span<const T>{data.data(), data.size()});
        }

        Builder& gridSize(glm::vec2 size);

        Builder& gridSize(glm::vec3 size);

        Builder& gridSize2D(glm::vec2 size);

        Builder& gridSize3D(glm::vec3 size);

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

        Builder& vectorField(std::span<glm::vec3> field);

        Builder& vectorField2D(std::span<glm::vec2> field);

        Builder& vectorField3D(std::span<glm::vec3> field);

        std::unique_ptr<FluidSolver> build();

    private:
        void generateVectorField(FluidSolver& solver);

        void addQuantities(FluidSolver& solver);

        Builder& addQuantityData(Quantity& quantity, std::string name, VkFormat format, std::span<const std::byte> data);

        void initQuantityTextures(FluidSolver& solver, Quantity& quantity, const std::string& name,
                                  VkFormat format, std::span<const std::byte> data);

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
        glm::vec3 _gridSize{0};
        VkImageType _imageType{VK_IMAGE_TYPE_3D};
        uint32_t _dimension{3};
        std::vector<std::reference_wrapper<Quantity>> _quantities;
        struct QuantityData {
            std::reference_wrapper<Quantity> quantity;
            std::string name;
            VkFormat format{};
            std::vector<std::byte> data;
        };
        std::vector<QuantityData> _quantityData;
        LinearSolverStrategy _linearSolverStrategy{LinearSolverStrategy::RBGS};

        std::vector<ExternalForce> _externalForces;
        std::optional<VectorFieldFunc2D> _generator2D{};
        std::optional<VectorFieldFunc3D> _generator3D{};
        std::vector<Collider> _colliders;
        std::vector<glm::vec2> _data2D;
        std::vector<glm::vec3> _data3D;
    };
}
