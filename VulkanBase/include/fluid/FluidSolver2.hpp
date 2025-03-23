#pragma once

#include "common.h"
#include "Texture.h"
#include "ComputePipelins.hpp"
#include "VulkanDevice.h"
#include "plugins/BindLessDescriptorPlugin.hpp"
#include <variant>

namespace eular {

    enum class LinearSolverStrategy  {
        Jacobi, RGGS
    };

    struct Field : std::array<Texture, 2> {
        std::string name;
        uint32_t in{~0u};
        uint32_t out{~0u};
        std::array<VkDescriptorSet, 2> imageDescriptorSets{};
        std::array<VkDescriptorSet, 2> textureDescriptorSets{};

        void swap() {
            std::swap(in, out);
            std::swap(imageDescriptorSets[0], imageDescriptorSets[1]);
            std::swap(textureDescriptorSets[0], textureDescriptorSets[1]);
        }
    };

    struct BridgeField : std::array<Texture*, 2>{
        uint32_t in{~0u};
        uint32_t out{~0u};

        void swap() {
            std::swap(in, out);
        }
    };

    struct VectorField {
        Field u;
        Field v;
        Field w;

        void swap() {
            u.swap();
            v.swap();
            w.swap();
        }
    };


    struct Quantity {
        std::string name;
        Field field;
        Field source;
        float diffuseRate{MIN_FLOAT};

        void update(VkCommandBuffer, Field &) {};

        bool postAdvect(VkCommandBuffer, Field &) { return false; };

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

        FluidSolver(VulkanDevice *device, VulkanDescriptorPool* descriptorPool, BindlessDescriptor *bindlessDescriptor, glm::vec2 gridSize);

        void init();

        void initFields();

        void set(VectorFieldSource2D vectorField);

        void set(VectorFieldSource3D vectorField);

        void add(ExternalForce&& force);

        void runSimulation(VkCommandBuffer commandBuffer);

        void add(Quantity &quantity);

        void dt(float value);

        float dt() const;

        void poissonIterations(int value);

        void viscosity(float value);

        void ensureBoundaryCondition(bool flag);

        BridgeField _oldVectorField;

    public:
        void createSamplers();

        void createDescriptorSetLayouts();

        void updateDescriptorSets();

        uint32_t createDescriptorSet(std::vector<VkWriteDescriptorSet>& writeOffset, uint32_t index, Field& field);

        void initGlobalConstants();

        void velocityStep(VkCommandBuffer commandBuffer);

        void advectVectorField(VkCommandBuffer commandBuffer);

        void advect(VkCommandBuffer commandBuffer, Field& field);

        void clearForces(VkCommandBuffer commandBuffer);

        void applyForces(VkCommandBuffer commandBuffer);

        void applyExternalForces(VkCommandBuffer commandBuffer);

        void addForcesToVectorField(VkCommandBuffer commandBuffer, ForceField& sourceField);

        void computeVorticityConfinement(VkCommandBuffer commandBuffer);

        void diffuseVelocityField(VkCommandBuffer commandBuffer);

        void diffuse(VkCommandBuffer commandBuffer, Field& field);

        void project(VkCommandBuffer commandBuffer);

        void computeDivergence(VkCommandBuffer commandBuffer);

        void solvePressure(VkCommandBuffer commandBuffer);

        void computeDivergenceFreeField(VkCommandBuffer commandBuffer);

        void addComputeBarrier(VkCommandBuffer commandBuffer);

        void updateProjectConstants();

        void quantityStep(VkCommandBuffer commandBuffer);

        void jacobiSolver(VkCommandBuffer commandBuffer, Field& solution, Field& unknown);

        void rbgsSolver(VkCommandBuffer commandBuffer, Field& solution, Field& unknown);

        void bridgeOut(VkCommandBuffer commandBuffer);

        void bridgeIn(VkCommandBuffer commandBuffer);

        std::vector<PipelineMetaData> pipelineMetaData() final;

        BindlessDescriptor &bindlessDescriptor();

        void bindTextures(VkDescriptorType descriptorType);

        void prepTextures();

        [[maybe_unused]] void bindDescriptorSet(VkCommandBuffer commandBuffer, VkPipelineLayout layout);

        static void clear(VkCommandBuffer commandBuffer, Texture& texture);

    public:
        VulkanDescriptorPool* _descriptorPool{};
        BindlessDescriptor *_bindlessDescriptor{};

        VectorField _vectorField;
        DivergenceField _divergenceField;
        PressureField _pressureField;
        Field _forceField;
        Field _vorticityField;

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
            float dt{};
            uint32_t ensure_boundary_condition{1};
        };

        struct {
            VulkanBuffer gpu;
            GlobalData* cpu{};
        } globalConstants;

        struct {
            bool advectVField = true;
            bool project = true;
            bool vorticity = false;
            int poissonIterations = 30;
            float viscosity = MIN_FLOAT;
        } options;

        struct {
            float alpha{};
            float rBeta{};
            uint solution_in{~0u};
            uint unknown_in{~0u};
            uint unknown_out{~0u};
            uint is_vector_field{};
            uint pass{0};
        } linearSolverConstants;

        struct {
            glm::uvec2 vector_field_id{~0u};
            uint quantity_in{~0u};
            uint quantity_out{~0u};
            uint sampler_id{0};
        } advectConstants;

        struct {
            glm::uvec4 vector_field_id{~0u};
            uint divergence_field_id{~0u};
            uint pressure_field_id{~0u};
        } projectConstants;

        struct {
            glm::uvec4 vector_field_id;
            uint force_field_id;
        } forceConstants{};

        glm::uvec3 _groupCount{1};
        VulkanDescriptorSetLayout uniformsSetLayout;
        VkDescriptorSet uniformDescriptorSet{};

        VulkanSampler _valueSampler;
        VulkanSampler _linearSampler;

        VkDescriptorSet _valueSamplerDescriptorSet{};
        VkDescriptorSet _linearSamplerDescriptorSet{};

        struct {
            glm::uvec2 vector_field_id{~0u};
            uint32_t dst_vector_field{~0u};
        } brideConstants;

        static constexpr uint32_t in = 0;
        static constexpr uint32_t out = 1;

        std::vector<ExternalForce> _externalForces;
        LinearSolverStrategy linearSolverStrategy{LinearSolverStrategy::RGGS};

        std::vector<VulkanDescriptorSetLayout> forceFieldSetLayouts();
    };
}