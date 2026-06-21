#include "fluid/FluidSolver2.hpp"
#include "fluid/CollocatedVectorGrid.hpp"
#include "Barrier.hpp"

#include <algorithm>
#include <cmath>
#include <format>

namespace eular {


    VulkanDescriptorSetLayout Collider::inputDescriptorSetLayout;
    VulkanDescriptorSetLayout Collider::outputDescriptorSetLayout;
    bool Collider::initialized = false;

    FluidSolver::FluidSolver(VulkanDevice *device, VulkanDescriptorPool* descriptorPool, glm::vec2 gridSize)
        : FluidSolver(device, descriptorPool, glm::vec3(gridSize, 1.0f), 2) {
    }

    FluidSolver::FluidSolver(VulkanDevice *device, VulkanDescriptorPool* descriptorPool, glm::vec3 gridSize,
                             uint32_t dimension)
        : ComputePipelines(device)
        , _descriptorPool(descriptorPool)
        , _imageType(VK_IMAGE_TYPE_3D)
        , _gridSize(gridSize)
        , _delta(1.f / gridSize)
        , _dimension(glm::clamp(dimension, 2u, 3u)) {
        _groupCount = glm::uvec3(glm::ceil(gridSize / 32.f));
        _groupCount.z = _dimension == 3u ? static_cast<uint32_t>(gridSize.z) : 1u;
    }

    FluidSolver::~FluidSolver() {
        releaseDescriptorSets();
    }

    
    void FluidSolver::init() {
        initGlobalConstants();
        initLinearSolverSupport();
        createSamplers();
        createDescriptorSetLayouts();
        updateDescriptorSets();
        initVectorGrid();
        initFields();
        updateFieldDescriptorSets();
        createPipelines();
    }

    void FluidSolver::initVectorGrid() {
        _vectorGrid = std::make_unique<CollocatedVectorGrid>(VectorGrid::Params{
            .device = device,
            .descriptorPool = _descriptorPool,
            .gridSize = _gridSize,
            .imageType = _imageType,
            .dimension = _dimension,
            .globalConstantsDescriptorSet = uniformDescriptorSet,
            .globalConstantsSetLayout = &uniformsSetLayout,
            .colliderDescriptorSet = _colliderDescriptorSet,
            .colliderDescriptorSetLayout = &_colliderDescriptorSetLayout,
            .macCormackAdvection = options.macCormackAdvection,
            .wrappingEnabled = options.wrappingEnabled
        });
        _vectorGrid->init();
        _fieldDescriptorSetLayout = _vectorGrid->fieldDescriptorSetLayout();
    }

    void FluidSolver::createSamplers() {
        VkSamplerAddressMode addressMode = options.wrappingEnabled ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_NEAREST;
        samplerInfo.minFilter = VK_FILTER_NEAREST;
        samplerInfo.addressModeU = addressMode;
        samplerInfo.addressModeV = addressMode;
        samplerInfo.addressModeW = addressMode;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST ;

        _valueSampler = device->createSampler(samplerInfo);
    }

    void FluidSolver::initGlobalConstants() {
        GlobalData data{};
        data.grid_size = glm::ivec3(_gridSize);
        data.dx = {_delta.x, 0, 0};
        data.dy = {0, _delta.y, 0};
        data.dz = {0, 0, _dimension == 3u ? _delta.z : 1.0f};
        data.dt = options.timeStep;
        data.density = options.density;
        data.wrapping_enabled = static_cast<int>(options.wrappingEnabled);
        data.dimension = _dimension;
        data.open_boundary_edges = options.openBoundaryEdges;
        globalConstants.gpu = device->createCpuVisibleBuffer(&data, sizeof(GlobalData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        globalConstants.cpu =  reinterpret_cast<GlobalData*>(globalConstants.gpu.map());
    }
    
    void FluidSolver::initFields() {
        auto size = glm::ivec3(_gridSize);

        _vorticityField.name = "vorticity_field";
        _pressureField.name = "pressure_field";

        auto addressMode = options.wrappingEnabled ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

        textures::createNoTransition(*device, _vorticityField[0], _imageType, VK_FORMAT_R32G32B32A32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _vorticityField[1], _imageType, VK_FORMAT_R32G32B32A32_SFLOAT, size, addressMode);

        textures::createNoTransition(*device, _pressureField[0], _imageType, VK_FORMAT_R32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _pressureField[1], _imageType, VK_FORMAT_R32_SFLOAT, size, addressMode);

        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vorticityField.name, 0), _vorticityField[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vorticityField.name, 1), _vorticityField[1].image.image);

        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _pressureField.name, 0), _pressureField[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _pressureField.name, 1), _pressureField[1].image.image);

        prepTextures();
    }

    void FluidSolver::initLinearSolverSupport() {
        auto unknownCount = static_cast<uint32_t>(_gridSize.x * _gridSize.y * _gridSize.z);
        auto vectorSize = static_cast<VkDeviceSize>(unknownCount) * sizeof(float);
        const auto stencilEntriesPerRow = _dimension == 3u ? 7u : linearSystemStencilEntriesPerRow;
        auto maxNonZeroCount = static_cast<VkDeviceSize>(unknownCount) * stencilEntriesPerRow;

        for (auto i = 0; i < 2; ++i) {
            auto& linearSystem = _linearSystems[i];
            auto& A = linearSystem.params.Coefficients;

            A.values = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY,
                                            maxNonZeroCount * sizeof(float), fmt::format("fluid_linear_solver_values_{}", i));
            A.colIndices = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY,
                                                maxNonZeroCount * sizeof(uint32_t), fmt::format("fluid_linear_solver_column_indices_{}", i));
            A.rowOffsets = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY,
                                                (static_cast<VkDeviceSize>(unknownCount) + 1) * sizeof(uint32_t), fmt::format("fluid_linear_solver_row_offsets_{}", i));
            A.counts = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY,
                                            3 * sizeof(uint32_t), fmt::format("fluid_linear_solver_counts_{}", i));
            A.numRows = unknownCount;
            A.numCols = unknownCount;

            linearSystem.params.unknown = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                               VMA_MEMORY_USAGE_GPU_ONLY, vectorSize, fmt::format("fluid_linear_solver_unknown_{}", i));
            linearSystem.params.solution = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                                VMA_MEMORY_USAGE_GPU_ONLY, vectorSize, fmt::format("fluid_linear_solver_rhs_{}", i));

            switch(linearSolverStrategy) {
                case LinearSolverStrategy::Jacobi:
                    linearSystem.solver = std::make_unique<gpu::linalg::JacobiSolver>(*device);
                    break;
                case LinearSolverStrategy::RBGS:
                    linearSystem.solver = std::make_unique<gpu::linalg::GaussSeidelSolver>(*device);
                    break;
                case LinearSolverStrategy::ConjugateGradient:
                    linearSystem.solver = std::make_unique<gpu::linalg::ConjugateGradientSolver>(*device);
                    break;
            }

            linearSystem.solver->init(vectorSize);
            linearSystem.params.numIterations = options.poissonIterations;
            linearSystem.params.id = i;
            linearSystem.constants.gridSize = glm::uvec3(_gridSize);
        }

        meanDriftBuffer = device->createBuffer(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY,
            sizeof(MeanDriftStats),
            "fluid_mean_drift_buffer");
    }

    void FluidSolver::createDescriptorSetLayouts() {
        uniformsSetLayout =
            device->descriptorSetLayoutBuilder()
                .name("fluid_solver_global_uniforms")
                .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();

        _colliderDescriptorSetLayout =
            device->descriptorSetLayoutBuilder()
                .name("fluid_solver_collider_textures")
                .binding(0)
                    .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();

        _colliderSamplers.fill(_valueSampler.handle);

        if (!Collider::initialized) {
            Collider::outputDescriptorSetLayout =
                device->descriptorSetLayoutBuilder()
                    .name("fluid_solver_source_collider_output_textures")
                    .binding(0)
                        .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .descriptorCount(maxColliderFields)
                        .shaderStages(VK_SHADER_STAGE_ALL)
                        .immutableSamplers(_colliderSamplers.data())
                    .binding(1)
                        .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .descriptorCount(maxColliderFields)
                        .shaderStages(VK_SHADER_STAGE_ALL)
                        .immutableSamplers(_colliderSamplers.data())
                .createLayout();
            
            Collider::inputDescriptorSetLayout =
                device->descriptorSetLayoutBuilder()
                    .name("fluid_solver_source_collider_input_textures")
                    .binding(0)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .descriptorCount(1)
                        .shaderStages(VK_SHADER_STAGE_ALL)
                    .binding(1)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .descriptorCount(1)
                        .shaderStages(VK_SHADER_STAGE_ALL)
                .createLayout();
        }

        createDefaultColliderFields();

        _debugDescriptorSetLayout =
            device->descriptorSetLayoutBuilder()
                .name("fluid_solver_debug_set_layout")
                .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                .descriptorCount(20)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();

        linearSystemDescriptorSetLayout =
            device->descriptorSetLayoutBuilder()
                .name("fluid_linear_solver_stencil_matrix_descriptor_set_layout")
                .binding(0)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(2)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(3)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();

        linearSystemVectorDescriptorSetLayout =
            device->descriptorSetLayoutBuilder()
                .name("fluid_linear_solver_vector_descriptor_set_layout")
                .binding(0)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();

        meanDriftDescriptorSetLayout =
            device->descriptorSetLayoutBuilder()
                .name("fluid_mean_drift_descriptor_set_layout")
                .binding(0)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();
    }

    void FluidSolver::updateDescriptorSets() {
        std::vector<VulkanDescriptorSetLayout> layouts{
            uniformsSetLayout,
            _colliderDescriptorSetLayout
        };

        const auto sourceColliderSetOffset = layouts.size();
        if(hasActiveColliders()) {
            layouts.push_back(Collider::outputDescriptorSetLayout);
        }

        const auto meanDriftSetOffset = layouts.size();
        layouts.push_back(meanDriftDescriptorSetLayout);

        const auto linearSystemSetOffset = layouts.size();
        layouts.push_back(linearSystemDescriptorSetLayout);
        layouts.push_back(linearSystemDescriptorSetLayout);
        layouts.push_back(linearSystemVectorDescriptorSetLayout);
        layouts.push_back(linearSystemVectorDescriptorSetLayout);

        auto sets = _descriptorPool->allocate(layouts);
        uniformDescriptorSet = sets[0];
        _colliderDescriptorSet = sets[1];
        _sourceColliderDescriptorSet = hasActiveColliders() ? sets[sourceColliderSetOffset] : VK_NULL_HANDLE;
        meanDriftDescriptorSet = sets[meanDriftSetOffset];
        _linearSystems[0].descriptorSet = sets[linearSystemSetOffset];
        _linearSystems[1].descriptorSet = sets[linearSystemSetOffset + 1];
        _linearSystems[0].rhsDescriptorSet = sets[linearSystemSetOffset + 2];
        _linearSystems[1].rhsDescriptorSet = sets[linearSystemSetOffset + 3];

        auto writes = initializers::writeDescriptorSets<32>();
        auto writeOffset = 0u;

        writes[writeOffset].dstSet = uniformDescriptorSet;
        writes[writeOffset].dstBinding = 0;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pBufferInfo = new VkDescriptorBufferInfo{globalConstants.gpu, 0, VK_WHOLE_SIZE};
        ++writeOffset;

        writes[writeOffset].dstSet = _colliderDescriptorSet;
        writes[writeOffset].dstBinding = 0;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo{
            _valueSampler.handle,
            _colliderField[in].imageView.handle,
            VK_IMAGE_LAYOUT_GENERAL
        };
        ++writeOffset;

        writes[writeOffset].dstSet = _colliderDescriptorSet;
        writes[writeOffset].dstBinding = 1;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo{
            _valueSampler.handle,
            _colliderVelocityField[in].imageView.handle,
            VK_IMAGE_LAYOUT_GENERAL
        };
        ++writeOffset;

        writes[writeOffset].dstSet = meanDriftDescriptorSet;
        writes[writeOffset].dstBinding = 0;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pBufferInfo = new VkDescriptorBufferInfo{meanDriftBuffer, 0, VK_WHOLE_SIZE};
        ++writeOffset;

        for(auto i = 0; i < 2; ++i) {
            const std::array<VulkanBuffer*, 4> buffers{
                &_linearSystems[i].params.Coefficients.values,
                &_linearSystems[i].params.Coefficients.colIndices,
                &_linearSystems[i].params.Coefficients.rowOffsets,
                &_linearSystems[i].params.Coefficients.counts
            };

            for(auto binding = 0u; binding < buffers.size(); ++binding) {
                writes[writeOffset].dstSet = _linearSystems[i].descriptorSet;
                writes[writeOffset].dstBinding = binding;
                writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[writeOffset].descriptorCount = 1;
                writes[writeOffset].pBufferInfo = new VkDescriptorBufferInfo{*buffers[binding], 0, VK_WHOLE_SIZE};
                ++writeOffset;
            }

            writes[writeOffset].dstSet = _linearSystems[i].rhsDescriptorSet;
            writes[writeOffset].dstBinding = 0;
            writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[writeOffset].descriptorCount = 1;
            writes[writeOffset].pBufferInfo = new VkDescriptorBufferInfo{_linearSystems[i].params.solution, 0, VK_WHOLE_SIZE};
            ++writeOffset;
        }

        writes.resize(writeOffset);
        device->updateDescriptorSets(writes);
        if(hasActiveColliders() && Collider::outputDescriptorSetLayout.handle != VK_NULL_HANDLE) {
            updateSourceColliderDescriptorSet();
        }

        for(auto& write : writes) {
            if(write.pImageInfo) delete write.pImageInfo;
            if(write.pBufferInfo) delete write.pBufferInfo;
        }
    }

    void FluidSolver::updateFieldDescriptorSets() {
        auto writes = initializers::writeDescriptorSets<24>();
        auto writeOffset = 0u;

        writeOffset = createDescriptorSet(writes, writeOffset, _pressureField);
        writeOffset = createDescriptorSet(writes, writeOffset, _vorticityField);
        writeOffset = createDescriptorSet(writes, writeOffset, _colliderField);
        writeOffset = createDescriptorSet(writes, writeOffset, _colliderVelocityField);

        writes.resize(writeOffset);
        device->updateDescriptorSets(writes);

        for(auto& write : writes) {
            if(write.pImageInfo) delete write.pImageInfo;
            if(write.pBufferInfo) delete write.pBufferInfo;
        }
    }

    void FluidSolver::updateSourceColliderDescriptorSet() {
        if(!hasActiveColliders()) {
            return;
        }

        ensureSourceColliderDescriptorSet();

        const auto fallbackField = _colliders.front().field;
        ensureZeroColliderVelocityTexture();
        std::array<VkDescriptorImageInfo, maxColliderFields> fallbackVelocityInfos{};
        fallbackVelocityInfos.fill({
            _valueSampler.handle,
            _zeroColliderVelocityTexture.imageView.handle,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        });

        VkWriteDescriptorSet fallbackVelocityWrite{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = _sourceColliderDescriptorSet,
            .dstBinding = 1,
            .descriptorCount = maxColliderFields,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = fallbackVelocityInfos.data()
        };

        device->updateDescriptorSets(std::array{fallbackVelocityWrite});

        std::vector<VkCopyDescriptorSet> copies;
        copies.reserve(maxColliderFields * 2);
        for(auto i = 0u; i < maxColliderFields; ++i) {
            const auto sourceIndex = std::min(i, _activeColliderCount - 1u);
            copies.push_back({
                .sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET,
                .srcSet = i < _activeColliderCount ? _colliders[sourceIndex].field : fallbackField,
                .srcBinding = 0,
                .srcArrayElement = 0,
                .dstSet = _sourceColliderDescriptorSet,
                .dstBinding = 0,
                .dstArrayElement = i,
                .descriptorCount = 1
            });

            if(i < _activeColliderCount && _colliders[i].velocity != VK_NULL_HANDLE) {
                copies.push_back({
                    .sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET,
                    .srcSet = _colliders[i].velocity,
                    .srcBinding = 0,
                    .srcArrayElement = 0,
                    .dstSet = _sourceColliderDescriptorSet,
                    .dstBinding = 1,
                    .dstArrayElement = i,
                    .descriptorCount = 1
                });
            }
        }

        if(!copies.empty()) {
            device->updateDescriptorSets(std::vector<VkWriteDescriptorSet>{}, copies);
        }
    }

    void FluidSolver::ensureSourceColliderDescriptorSet() {
        if(_sourceColliderDescriptorSet != VK_NULL_HANDLE) {
            return;
        }

        _sourceColliderDescriptorSet = _descriptorPool->allocate({Collider::outputDescriptorSetLayout}).front();
    }

    void FluidSolver::ensureZeroColliderVelocityTexture() {
        if(_zeroColliderVelocityTexture.isValid()) {
            return;
        }

        glm::vec2 emptyVelocity{0.0f};
        textures::create(*device, _zeroColliderVelocityTexture, _imageType, VK_FORMAT_R32G32_SFLOAT,
                         &emptyVelocity, {1u, 1u, 1u}, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, sizeof(glm::vec2));
        device->setName<VK_OBJECT_TYPE_IMAGE>("fluid_solver_zero_collider_velocity_texture",
                                              _zeroColliderVelocityTexture.image.image);
    }

    bool FluidSolver::hasActiveColliders() const {
        return _activeColliderCount > 0;
    }

    uint32_t FluidSolver::createDescriptorSet(std::vector<VkWriteDescriptorSet>& writes, uint32_t writeOffset, Field& field) {
        auto sets = _descriptorPool->allocate({_fieldDescriptorSetLayout, _fieldDescriptorSetLayout});

        field.descriptorSet[0] = sets[0];
        field.descriptorSet[1] = sets[1];

        writes[writeOffset].dstSet = field.descriptorSet[0];
        writes[writeOffset].dstBinding = 0;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo {VK_NULL_HANDLE, field[0].imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
        ++writeOffset;

        writes[writeOffset].dstSet = field.descriptorSet[0];
        writes[writeOffset].dstBinding = 1;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo {VK_NULL_HANDLE, field[0].imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
        ++writeOffset;

        writes[writeOffset].dstSet = field.descriptorSet[0];
        writes[writeOffset].dstBinding = 2;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo {VK_NULL_HANDLE, field[0].imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
        ++writeOffset;

        writes[writeOffset].dstSet = field.descriptorSet[1];
        writes[writeOffset].dstBinding = 0;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo {VK_NULL_HANDLE, field[1].imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
        ++writeOffset;

        writes[writeOffset].dstSet = field.descriptorSet[1];
        writes[writeOffset].dstBinding = 1;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo {VK_NULL_HANDLE, field[1].imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
        ++writeOffset;

        writes[writeOffset].dstSet = field.descriptorSet[1];
        writes[writeOffset].dstBinding = 2;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo {VK_NULL_HANDLE, field[1].imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
        ++writeOffset;

        return writeOffset;
    }

    void FluidSolver::releaseDescriptorSets() {
        _vectorGrid.reset();

        releaseDescriptorSet(uniformDescriptorSet);
        releaseDescriptorSet(_colliderDescriptorSet);
        releaseDescriptorSet(_sourceColliderDescriptorSet);
        releaseDescriptorSet(meanDriftDescriptorSet);

        for(auto& linearSystem : _linearSystems) {
            releaseDescriptorSet(linearSystem.descriptorSet);
            releaseDescriptorSet(linearSystem.rhsDescriptorSet);
        }

        releaseFieldDescriptorSets(_pressureField);
        releaseFieldDescriptorSets(_vorticityField);
        releaseFieldDescriptorSets(_colliderField);
        releaseFieldDescriptorSets(_colliderVelocityField);

        for(auto& quantity : _quantities) {
            releaseFieldDescriptorSets(quantity.get().field);
            releaseFieldDescriptorSets(quantity.get().source);
        }
    }

    void FluidSolver::releaseDescriptorSet(VkDescriptorSet& descriptorSet) {
        if(!_descriptorPool || descriptorSet == VK_NULL_HANDLE) {
            return;
        }

        _descriptorPool->free(descriptorSet);
        descriptorSet = VK_NULL_HANDLE;
    }

    void FluidSolver::releaseFieldDescriptorSets(Field& field) {
        releaseDescriptorSet(field.descriptorSet[0]);
        releaseDescriptorSet(field.descriptorSet[1]);
    }

    void FluidSolver::createDefaultColliderFields() {
        const auto width = static_cast<uint32_t>(_gridSize.x);
        const auto height = static_cast<uint32_t>(_gridSize.y);
        const auto depth = static_cast<uint32_t>(_gridSize.z);
        const auto cellCount = width * height * depth;

        _colliderField.name = "fluid_solver_collider";
        _colliderVelocityField.name = "fluid_solver_collider_velocity";

        std::vector<glm::vec2> colliderData(
            cellCount,
            glm::vec2{1.0f, colliderTypeValue(ColliderType::Wall)});
        std::vector<glm::vec2> colliderVelocity(cellCount, glm::vec2{0.0f});
        for(auto i = 0u; i < 2; ++i) {
            textures::create(*device, _colliderField[i], _imageType, VK_FORMAT_R32G32_SFLOAT,
                             colliderData.data(), {width, height, depth}, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, sizeof(glm::vec2));
            textures::create(*device, _colliderVelocityField[i], _imageType, VK_FORMAT_R32G32_SFLOAT,
                             colliderVelocity.data(), {width, height, depth}, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, sizeof(glm::vec2));
            _colliderField[i].image.transitionLayout(device->graphicsCommandPool(), VK_IMAGE_LAYOUT_GENERAL);
            _colliderVelocityField[i].image.transitionLayout(device->graphicsCommandPool(), VK_IMAGE_LAYOUT_GENERAL);

            device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _colliderField.name, i), _colliderField[i].image.image);
            device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _colliderVelocityField.name, i), _colliderVelocityField[i].image.image);
        }
    }


    void FluidSolver::prepTextures() {
        device->firstActiveCommandPool().oneTimeCommand([&](auto commandBuffer) {
            std::vector<VkImageMemoryBarrier2> barriers;

            VkImageMemoryBarrier2 barrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                    .srcStageMask = VK_PIPELINE_STAGE_NONE,
                    .srcAccessMask = VK_ACCESS_NONE,
                    .dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .subresourceRange = {
                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .baseMipLevel =  0,
                            .levelCount = 1,
                            .baseArrayLayer = 0,
                            .layerCount = 1,
                    }
            };

            barrier.image = _vorticityField[0].image;
            barriers.push_back(barrier);

            barrier.image = _pressureField[0].image;
            barriers.push_back(barrier);

            barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

            barrier.image = _vorticityField[1].image;
            barriers.push_back(barrier);

            barrier.image = _pressureField[1].image;
            barriers.push_back(barrier);

            VkDependencyInfo dInfo {
                    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                    .imageMemoryBarrierCount = COUNT(barriers),
                    .pImageMemoryBarriers = barriers.data()
            };

            vkCmdPipelineBarrier2(commandBuffer, &dInfo);

            auto clearTexture = [&](Texture& texture) {
                texture.image.currentLayout = VK_IMAGE_LAYOUT_GENERAL;

                VkClearColorValue zero{{0.0f, 0.0f, 0.0f, 0.0f}};
                vkCmdClearColorImage(commandBuffer, texture.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &DEFAULT_SUB_RANGE);
            };

            clearTexture(_vorticityField[0]);
            clearTexture(_vorticityField[1]);
            clearTexture(_pressureField[0]);
            clearTexture(_pressureField[1]);

            Barriers::pushAndFlush(commandBuffer,
                                   VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                   VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
        });
    }

    std::vector<PipelineMetaData> FluidSolver::pipelineMetaData() {
        return {
                {
                    .name = "update_collider",
                    .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\update_collider.comp.spv)",
                    .layouts = {
                        &uniformsSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout
                    },
                    .ranges = { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(updateColliderConstants) } }
                },
                {
                    .name = "update_collider_with_sources",
                    .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\update_collider_with_sources.comp.spv)",
                    .layouts = {
                        &uniformsSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                        &Collider::outputDescriptorSetLayout
                    },
                    .ranges = { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(updateColliderConstants) } }
                },
                {
                        .name = "add_sources",
                        .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\add_sources.comp.spv)",
                        .layouts =  {
                                &uniformsSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                                &_fieldDescriptorSetLayout, &_colliderDescriptorSetLayout
                        }
                },
                {
                    .name = "vorticity",
                    .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\vorticity.comp.spv)",
                    .layouts =  {
                            &uniformsSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                            &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout, &_colliderDescriptorSetLayout
                      }
                },
                {
                    .name = "vorticity_force",
                    .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\vorticity_force.comp.spv)",
                    .layouts =  {
                            &uniformsSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                            &_fieldDescriptorSetLayout, &_colliderDescriptorSetLayout
                      },
                      .ranges = { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float) } }
                },
                {
                    .name = "generate_coefficients",
                    .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\generate_coefficients.comp.spv)",
                    .layouts = { &uniformsSetLayout, &_colliderDescriptorSetLayout, &linearSystemDescriptorSetLayout },
                    .ranges = { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(_linearSystems[0].constants) } }
                },
                {
                    .name = "copy_scaled_to_buffer",
                    .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\copy_scaled_to_buffer.comp.spv)",
                    .layouts = { &_fieldDescriptorSetLayout, &linearSystemVectorDescriptorSetLayout },
                    .ranges = { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ScaledFieldCopyConstants) } }
                },
                {
                    .name = "mean_drift_reduce",
                    .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\mean_drift_reduce.comp.spv)",
                    .layouts = { &uniformsSetLayout, &_fieldDescriptorSetLayout, &_colliderDescriptorSetLayout, &meanDriftDescriptorSetLayout },
                    .ranges = { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MeanDriftConstants) } }
                },
                {
                    .name = "mean_drift_subtract",
                    .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\mean_drift_subtract.comp.spv)",
                    .layouts = { &uniformsSetLayout, &_fieldDescriptorSetLayout, &meanDriftDescriptorSetLayout, &_fieldDescriptorSetLayout, &_colliderDescriptorSetLayout },
                    .ranges = { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MeanDriftConstants) } }
                },
                {
                    .name = "constrain_velocity",
                    .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\constrain_velocity.comp.spv)",
                    .layouts = { &uniformsSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                                &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                                &_fieldDescriptorSetLayout, &_colliderDescriptorSetLayout },
                },
        };
    }

    void FluidSolver::updateColliderFields(VkCommandBuffer commandBuffer) {
        VULKAN_COMMAND_BUFFER_SECTION(device, commandBuffer, update_collider);
        updateColliderConstants.colliderCount = _activeColliderCount;
        updateColliderConstants.closedDomain = static_cast<uint32_t>(options.closedDomain);
        updateColliderConstants.openBoundaryEdges = options.openBoundaryEdges;

        if(hasActiveColliders()) {
            ensureSourceColliderDescriptorSet();
            const std::array<VkDescriptorSet, 4> sets{
                uniformDescriptorSet,
                _colliderField.descriptorSet[in],
                _colliderVelocityField.descriptorSet[in],
                _sourceColliderDescriptorSet
            };

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("update_collider_with_sources"));
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("update_collider_with_sources"),
                                    0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
            vkCmdPushConstants(commandBuffer, layout("update_collider_with_sources"), VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(updateColliderConstants), &updateColliderConstants);
        }else {
            const std::array<VkDescriptorSet, 3> sets{
                uniformDescriptorSet,
                _colliderField.descriptorSet[in],
                _colliderVelocityField.descriptorSet[in]
            };

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("update_collider"));
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("update_collider"),
                                    0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
            vkCmdPushConstants(commandBuffer, layout("update_collider"), VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(updateColliderConstants), &updateColliderConstants);
        }

        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
        addComputeBarrier(commandBuffer, {&_colliderField[in], &_colliderVelocityField[in]});
    }


    void FluidSolver::velocityStep(VkCommandBuffer commandBuffer) {
        if(!options.advectVField) return;

        VULKAN_COMMAND_BUFFER_SECTION(device, commandBuffer, velocity_step);
        clearForces(commandBuffer);
        applyForces(commandBuffer);
        diffuseVelocityField(commandBuffer);
        project(commandBuffer);
        advectVectorField(commandBuffer);
    }


    void FluidSolver::clearForces(VkCommandBuffer commandBuffer) {
        auto& forceField = _vectorGrid->forceField();
        clear(commandBuffer, forceField[0]);
        clear(commandBuffer, forceField[1]);
    }


    void FluidSolver::applyForces(VkCommandBuffer commandBuffer) {
        VULKAN_COMMAND_BUFFER_SECTION(device, commandBuffer, apply_forces);
        applyExternalForces(commandBuffer);
        computeVorticityConfinement(commandBuffer);
        _vectorGrid->addForcesToVectorField(commandBuffer);
        applyBoundaryConditions(commandBuffer);
    }


    void FluidSolver::clear(VkCommandBuffer commandBuffer, Texture &texture) {
        texture.image.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, DEFAULT_SUB_RANGE
                , VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT
                , VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkClearColorValue color{ {0.f, 0.f, 0.f, 0.f}};
        VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(commandBuffer, texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);

        texture.image.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_GENERAL, DEFAULT_SUB_RANGE
                , VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT
                , VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    void FluidSolver::diffuseVelocityField(VkCommandBuffer commandBuffer) {
        if(options.viscosity <= 0) return;
        const auto rho = options.density;
        auto& vf = _vectorGrid->vectorField();
        VULKAN_COMMAND_BUFFER_SECTION(device, commandBuffer, diffuse);
        diffuse(commandBuffer, vf.u, options.viscosity/rho, 1);
        diffuse(commandBuffer, vf.v, options.viscosity/rho, 2);
        if(_dimension == 3u) {
            diffuse(commandBuffer, vf.w, options.viscosity/rho, 3);
        }
        project(commandBuffer);
    }

    void FluidSolver::diffuse(VkCommandBuffer commandBuffer, Field& field, float rate, uint32_t vectorFieldComponent) {
        if(rate <= 0) return;
        const auto index = vectorFieldComponent == 2 ? 1u : 0u;
        auto& linearSystem = _linearSystems[index];

        assign(commandBuffer, field[in], linearSystem.params.solution);
        assign(commandBuffer, field[in], linearSystem.params.unknown);
        setDiffuseConstants(index, rate, vectorFieldComponent);
        solveLinearSystem(commandBuffer, index);
        assign(commandBuffer, linearSystem.params.unknown, field[out]);
        field.swap();
    }

    void FluidSolver::project(VkCommandBuffer commandBuffer) {
        if(!options.project) return;

        VULKAN_COMMAND_BUFFER_SECTION(device, commandBuffer, projection);
        applyBoundaryConditions(commandBuffer);

        _vectorGrid->computeDivergence(commandBuffer);
        subtractMeanDrift(commandBuffer, _vectorGrid->divergenceField());

        solvePressure(commandBuffer);
        subtractMeanDrift(commandBuffer, _pressureField);

        _vectorGrid->computeDivergenceFreeField(commandBuffer, _pressureField);
        applyBoundaryConditions(commandBuffer);
    }

    void FluidSolver::clearPressureField(VkCommandBuffer commandBuffer) {
        clear(commandBuffer, _pressureField[0]);
        clear(commandBuffer, _pressureField[1]);
    }

    void FluidSolver::advectVectorField(VkCommandBuffer commandBuffer) {
        VULKAN_COMMAND_BUFFER_SECTION(device, commandBuffer, advect);
        _vectorGrid->advectVectorField(commandBuffer);
        applyBoundaryConditions(commandBuffer);
    }

    void FluidSolver::constrainVelocity(VkCommandBuffer commandBuffer) {
        auto& vf = vectorField();
        static std::array<VkDescriptorSet, 8> sets;

        sets[0] = uniformDescriptorSet;
        sets[1] = vf.u.descriptorSet[in];
        sets[2] = vf.v.descriptorSet[in];
        sets[3] = vf.w.descriptorSet[in];
        sets[4] = vf.u.descriptorSet[out];
        sets[5] = vf.v.descriptorSet[out];
        sets[6] = vf.w.descriptorSet[out];
        sets[7] = _colliderDescriptorSet;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("constrain_velocity"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("constrain_velocity"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);

        addComputeBarrier(commandBuffer, {&vf.u[out], &vf.v[out], &vf.w[out]});
        vf.swap();
    }


    void FluidSolver::quantityStep(VkCommandBuffer commandBuffer) {
        VULKAN_COMMAND_BUFFER_SECTION(device, commandBuffer, quantity_step);
        for(auto& quantity : _quantities) {
            quantityStep(commandBuffer, quantity);
        }
    }

    void FluidSolver::quantityStep(VkCommandBuffer commandBuffer, Quantity& quantity) {
        clearSources(commandBuffer, quantity);
        updateSources(commandBuffer, quantity);
        addSource(commandBuffer, quantity);
        diffuseQuantity(commandBuffer, quantity);
        advectQuantity(commandBuffer, quantity);
        postAdvection(commandBuffer, quantity);
    }

    void FluidSolver::macCormackAdvect(VkCommandBuffer commandBuffer, Field& field, uint32_t boundaryMode) {
        _vectorGrid->advect(commandBuffer, field, boundaryMode);
    }

    void FluidSolver::applyBoundaryConditions(VkCommandBuffer commandBuffer) {
        constrainVelocity(commandBuffer);
    }

    void FluidSolver::advect(VkCommandBuffer commandBuffer, Field& field, uint32_t boundaryMode, bool addBarrier) {
        _vectorGrid->advect(commandBuffer, field, boundaryMode, addBarrier);
    }

    void FluidSolver::advect(VkCommandBuffer commandBuffer, VkDescriptorSet inDescriptor, VkDescriptorSet outDescriptor,
                             TimeDirection timeDirection, uint32_t boundaryMode, Texture* writeTexture) {

        _vectorGrid->advect(commandBuffer, inDescriptor, outDescriptor, timeDirection, boundaryMode, writeTexture);
    }

    void FluidSolver::applyExternalForces(VkCommandBuffer commandBuffer) {
        static std::array<VkDescriptorSet, 2> sets;
        auto& forceField = _vectorGrid->forceField();
        for(const auto& externalForce : _externalForces){
            sets[0] = forceField.descriptorSet[in];
            sets[1] = forceField.descriptorSet[out];
            externalForce(commandBuffer, sets, _groupCount);
            addComputeBarrier(commandBuffer, forceField[out]);
            forceField.swap();
        }
    }

    void FluidSolver::computeVorticityConfinement(VkCommandBuffer commandBuffer) {
        if(options.vorticityConfinementScale < 1) return;
        computeVorticity(commandBuffer);
        applyVorticity(commandBuffer);
    }

    void FluidSolver::solveLinearSystem(VkCommandBuffer commandBuffer, uint32_t index) {
        buildCoefficientMatrix(commandBuffer, index);
        _linearSystems[index].solver->solve(commandBuffer, _linearSystems[index].params);
    }

    void FluidSolver::buildCoefficientMatrix(VkCommandBuffer commandBuffer, uint32_t index) {
        auto unknownCount = _linearSystems[index].params.Coefficients.numRows;
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("generate_coefficients"));
        const std::array<VkDescriptorSet, 3> sets{
            uniformDescriptorSet,
            _colliderDescriptorSet,
            _linearSystems[index].descriptorSet};
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("generate_coefficients"), 0,
                                COUNT(sets), sets.data(), 0, nullptr);

        for(auto offset = 0u; offset < unknownCount; offset += linearSystemRowsPerBatch) {
            auto batchSize = std::min(linearSystemRowsPerBatch, unknownCount - offset);
            _linearSystems[index].constants.batchOffset = offset;
            _linearSystems[index].constants.batchSize = batchSize;
            vkCmdPushConstants(commandBuffer, layout("generate_coefficients"), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(_linearSystems[index].constants), &_linearSystems[index].constants);
            vkCmdDispatch(commandBuffer, (batchSize + 31u) / 32u, 1, 1);
        }

        auto& A = _linearSystems[index].params.Coefficients;
        Barrier::computeWriteToRead(commandBuffer, {A.values, A.colIndices, A.rowOffsets, A.counts});
    }

    void FluidSolver::setDiffuseConstants(uint32_t index, float rate, uint32_t vectorFieldComponent) {
        auto& constants = _linearSystems[index].constants;
        constants.identity = 1.0f;
        constants.vectorFieldComponent = vectorFieldComponent;

        const auto rateTime = options.timeStep * rate;
        constants.alpha = {
            rateTime / (_delta.x * _delta.x),
            rateTime / (_delta.y * _delta.y),
            _dimension == 3u ? rateTime / (_delta.z * _delta.z) : 0.0f
        };
    }

    void FluidSolver::setPressureConstants(uint32_t index) {
        auto& constants = _linearSystems[index].constants;
        constants.identity = 0.0f;
        constants.vectorFieldComponent = 0;
        constants.alpha = {
            _delta.y * _delta.y,
            _delta.x * _delta.x,
            _dimension == 3u ? _delta.z * _delta.z : 0.0f
        };
    }

    void FluidSolver::assign(VkCommandBuffer commandBuffer, Texture &from, VulkanBuffer &to) {
        VkImageSubresourceRange subresourceRange{};
        subresourceRange.aspectMask = from.aspectMask;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = 1;
        subresourceRange.baseArrayLayer = 0;
        subresourceRange.layerCount = 1;

        const auto oldLayout = from.image.currentLayout;
        if(oldLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
            const auto srcAccess = oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
                ? VK_ACCESS_NONE
                : VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            const auto srcStage = oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
                ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

            from.image.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, subresourceRange,
                                        srcAccess,
                                        VK_ACCESS_TRANSFER_READ_BIT,
                                        srcStage,
                                        VK_PIPELINE_STAGE_TRANSFER_BIT);
        }

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = from.aspectMask;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {from.width, from.height, from.depth};

        vkCmdCopyImageToBuffer(commandBuffer, from.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, to, 1, &region);

        const auto finalLayout = oldLayout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_GENERAL : oldLayout;
        if(finalLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
            from.image.transitionLayout(commandBuffer, finalLayout, subresourceRange,
                                        VK_ACCESS_TRANSFER_READ_BIT,
                                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }

        Barrier::transferWriteToComputeRead(commandBuffer, to);
    }

    void FluidSolver::assignScaled(VkCommandBuffer commandBuffer, Field& from, VulkanBuffer& to, VkDescriptorSet toDescriptorSet, float scale) {
        const ScaledFieldCopyConstants constants{
            .scale = scale,
            .count = static_cast<uint32_t>(_gridSize.x * _gridSize.y * _gridSize.z)
        };
        const std::array<VkDescriptorSet, 2> sets{from.descriptorSet[in], toDescriptorSet};

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("copy_scaled_to_buffer"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("copy_scaled_to_buffer"), 0,
                                COUNT(sets), sets.data(), 0, nullptr);
        vkCmdPushConstants(commandBuffer, layout("copy_scaled_to_buffer"), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(constants), &constants);
        vkCmdDispatch(commandBuffer, (constants.count + 31u) / 32u, 1, 1);
        Barrier::computeWriteToRead(commandBuffer, {to});
    }

    void FluidSolver::assign(VkCommandBuffer commandBuffer, VulkanBuffer &from, Texture &to) {
        Barrier::computeWriteToTransferRead(commandBuffer, {from});

        VkImageSubresourceRange subresourceRange{};
        subresourceRange.aspectMask = to.aspectMask;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = 1;
        subresourceRange.baseArrayLayer = 0;
        subresourceRange.layerCount = 1;

        const auto oldLayout = to.image.currentLayout;
        const auto srcAccess = oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
            ? VK_ACCESS_NONE
            : VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        const auto srcStage = oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
            ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        if(oldLayout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            to.image.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresourceRange,
                                      srcAccess,
                                      VK_ACCESS_TRANSFER_WRITE_BIT,
                                      srcStage,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT);
        }

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = to.aspectMask;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {to.width, to.height, to.depth};

        vkCmdCopyBufferToImage(commandBuffer, from, to.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        const auto finalLayout = oldLayout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_GENERAL : oldLayout;
        if(finalLayout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            to.image.transitionLayout(commandBuffer, finalLayout, subresourceRange,
                                      VK_ACCESS_TRANSFER_WRITE_BIT,
                                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
    }

    void FluidSolver::addComputeBarrier(VkCommandBuffer commandBuffer, Texture& texture) {
        addComputeBarrier(commandBuffer, {&texture});
    }

    void FluidSolver::addComputeBarrier(VkCommandBuffer commandBuffer, std::initializer_list<Texture*> textures) {
        for(auto texture : textures) {
            Barriers::push(texture->image, DEFAULT_SUB_RANGE, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, texture->image.currentLayout, texture->image.currentLayout);
        }
        Barriers::flush(commandBuffer);
    }

    void FluidSolver::computeDivergence(VkCommandBuffer commandBuffer) {
        _vectorGrid->computeDivergence(commandBuffer);
    }

    void FluidSolver::solvePressure(VkCommandBuffer commandBuffer) {
        VULKAN_COMMAND_BUFFER_SECTION(device, commandBuffer, solve_pressure);
        const auto rho = options.density;
        const auto dt = options.timeStep;
        constexpr auto index = 0u;
        auto& linearSystem = _linearSystems[index];
        const auto pressureScale = -(rho * _delta.x * _delta.x * _delta.y * _delta.y *
                                     (_dimension == 3u ? _delta.z * _delta.z : 1.0f)) / dt;

        assignScaled(commandBuffer, _vectorGrid->divergenceField(), linearSystem.params.solution, linearSystem.rhsDescriptorSet, pressureScale);
        assign(commandBuffer, _pressureField[in], linearSystem.params.unknown);
        setPressureConstants(index);
        solveLinearSystem(commandBuffer, index);
        assign(commandBuffer, linearSystem.params.unknown, _pressureField[out]);
        _pressureField.swap();
        addComputeBarrier(commandBuffer, _pressureField[in]);
    }

    void FluidSolver::subtractMeanDrift(VkCommandBuffer commandBuffer, Field& field) {
        VULKAN_COMMAND_BUFFER_SECTION(device, commandBuffer, subtract_mean_drift);

        const MeanDriftConstants constants{
            .count = static_cast<uint32_t>(_gridSize.x * _gridSize.y * _gridSize.z)
        };

        vkCmdFillBuffer(commandBuffer, meanDriftBuffer, 0, VK_WHOLE_SIZE, 0);
        Barrier::transferWriteToComputeWrite(commandBuffer, meanDriftBuffer);

        const std::array<VkDescriptorSet, 4> reduceSets{
            uniformDescriptorSet,
            field.descriptorSet[in],
            _colliderDescriptorSet,
            meanDriftDescriptorSet
        };

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("mean_drift_reduce"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("mean_drift_reduce"),
                                0, COUNT(reduceSets), reduceSets.data(), 0, VK_NULL_HANDLE);
        vkCmdPushConstants(commandBuffer, layout("mean_drift_reduce"), VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(constants), &constants);
        vkCmdDispatch(commandBuffer, 1, 1, 1);
        Barrier::computeWriteToRead(commandBuffer, meanDriftBuffer);

        const std::array<VkDescriptorSet, 5> subtractSets{
            uniformDescriptorSet,
            field.descriptorSet[in],
            meanDriftDescriptorSet,
            field.descriptorSet[out],
            _colliderDescriptorSet
        };

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("mean_drift_subtract"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("mean_drift_subtract"),
                                0, COUNT(subtractSets), subtractSets.data(), 0, VK_NULL_HANDLE);
        vkCmdPushConstants(commandBuffer, layout("mean_drift_subtract"), VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(constants), &constants);
        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
        addComputeBarrier(commandBuffer, field[out]);
        field.swap();
    }

    void FluidSolver::computeDivergenceFreeField(VkCommandBuffer commandBuffer) {
        _vectorGrid->computeDivergenceFreeField(commandBuffer, _pressureField);
    }


    float FluidSolver::dt() const {
        return options.timeStep;
    }

    FluidSolver& FluidSolver::density(float rho) {
        options.density = glm::max(1.f, rho);
        if(globalConstants.cpu) {
            globalConstants.cpu->density = options.density;
        }
        return *this;
    }

    void FluidSolver::runSimulation(VkCommandBuffer commandBuffer) {
        updateColliderFields(commandBuffer);
        velocityStep(commandBuffer);
        quantityStep(commandBuffer);
        _elapsedTime += options.timeStep;
    }

    std::vector<VulkanDescriptorSetLayout> FluidSolver::forceFieldSetLayouts() {
        return {_fieldDescriptorSetLayout, _fieldDescriptorSetLayout};
    }

    std::vector<VulkanDescriptorSetLayout> FluidSolver::sourceFieldSetLayouts() {
        return {_fieldDescriptorSetLayout, _fieldDescriptorSetLayout};
    }

    void FluidSolver::clearSources(VkCommandBuffer commandBuffer, Quantity &quantity) {
        clear(commandBuffer, quantity.source[in]);
        clear(commandBuffer, quantity.source[out]);
    }

    void FluidSolver::updateSources(VkCommandBuffer commandBuffer, Quantity &quantity) {
        quantity.update(commandBuffer, quantity.source, _groupCount);
        addComputeBarrier(commandBuffer, quantity.source[in]);
    }

    void FluidSolver::addSource(VkCommandBuffer commandBuffer, Quantity &quantity) {
        static std::array<VkDescriptorSet, 5> sets;
        sets[0] = uniformDescriptorSet;
        sets[1] = quantity.source.descriptorSet[in];
        sets[2] = quantity.field.descriptorSet[in];
        sets[3] = quantity.field.descriptorSet[out];
        sets[4] = _colliderDescriptorSet;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("add_sources"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("add_sources"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
        addComputeBarrier(commandBuffer, quantity.field[out]);
        quantity.field.swap();
    }

    void FluidSolver::diffuseQuantity(VkCommandBuffer commandBuffer, Quantity &quantity) {
        diffuse(commandBuffer, quantity.field, quantity.diffuseRate);
    }

    void FluidSolver::advectQuantity(VkCommandBuffer commandBuffer, Quantity &quantity) {
        advect(commandBuffer, quantity.field);
        quantity.field.swap();
    }

    void FluidSolver::postAdvection(VkCommandBuffer commandBuffer, Quantity &quantity) {
        for(auto& postAdvect : quantity.postAdvectActions) {
            if(postAdvect(commandBuffer, quantity.field, _groupCount)) {
                addComputeBarrier(commandBuffer, quantity.field[out]);
                quantity.field.swap();
            }
        }
    }

    void FluidSolver::computeVorticity(VkCommandBuffer commandBuffer) {
        static std::array<VkDescriptorSet, 6> sets;
        auto& vf = _vectorGrid->vectorField();
        sets[0] = uniformDescriptorSet;
        sets[1] = vf.u.descriptorSet[in];
        sets[2] = vf.v.descriptorSet[in];
        sets[3] = vf.w.descriptorSet[in];
        sets[4] = _vorticityField.descriptorSet[in];
        sets[5] = _colliderDescriptorSet;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("vorticity"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("vorticity"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
        addComputeBarrier(commandBuffer, _vorticityField[in]);
    }

    void FluidSolver::applyVorticity(VkCommandBuffer commandBuffer) {
        static std::array<VkDescriptorSet, 5> sets;
        auto& forceField = _vectorGrid->forceField();
        sets[0] = uniformDescriptorSet;
        sets[1] = _vorticityField.descriptorSet[in];
        sets[2] = forceField.descriptorSet[in];
        sets[3] = forceField.descriptorSet[out];
        sets[4] = _colliderDescriptorSet;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("vorticity_force"));
        vkCmdPushConstants(commandBuffer, layout("vorticity_force"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float), &options.vorticityConfinementScale);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("vorticity_force"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
        addComputeBarrier(commandBuffer, forceField[out]);
        forceField.swap();
    }

    VulkanDescriptorSetLayout FluidSolver::fieldDescriptorSetLayout() const {
        return _fieldDescriptorSetLayout;
    }

    float FluidSolver::elapsedTime() const {
        return _elapsedTime;
    }

    VectorField &FluidSolver::vectorField() {
        return _vectorGrid->vectorField();
    }

    PressureField &FluidSolver::pressureField() {
        return _pressureField;
    }

    Field& FluidSolver::colliderField() {
        return _colliderField;
    }

    const Field& FluidSolver::colliderField() const {
        return _colliderField;
    }

    Field& FluidSolver::colliderVelocityField() {
        return _colliderVelocityField;
    }

    const Field& FluidSolver::colliderVelocityField() const {
        return _colliderVelocityField;
    }

    Texture &FluidSolver::colliderTexture() {
        return _colliderField[in];
    }

    const Texture &FluidSolver::colliderTexture() const {
        return _colliderField[in];
    }

    Texture& FluidSolver::colliderVelocityTexture() {
        return _colliderVelocityField[in];
    }

    const Texture& FluidSolver::colliderVelocityTexture() const {
        return _colliderVelocityField[in];
    }

    void FluidSolver::setColliders(std::span<const Collider> colliders) {
        _colliders.clear();
        _colliders.reserve(std::min<std::size_t>(colliders.size(), maxColliderFields));
        for(const auto& collider : colliders) {
            if(collider.field == VK_NULL_HANDLE) {
                continue;
            }

            if(_colliders.size() >= maxColliderFields) {
                throw std::runtime_error{std::format("FluidSolver supports at most {} active collider fields", maxColliderFields)};
            }

            _colliders.push_back(collider);
        }

        _activeColliderCount = static_cast<uint32_t>(_colliders.size());
        if(hasActiveColliders() && Collider::outputDescriptorSetLayout.handle != VK_NULL_HANDLE) {
            updateSourceColliderDescriptorSet();
        }
    }

    uint32_t FluidSolver::activeColliderCount() const {
        return _activeColliderCount;
    }

    FluidSolver& FluidSolver::closedDomain(bool flag) {
        options.closedDomain = flag;
        if(flag) {
            options.wrappingEnabled = false;
            if(globalConstants.cpu) {
                globalConstants.cpu->wrapping_enabled = 0;
            }
        }
        return *this;
    }

    FluidSolver& FluidSolver::openBoundaryEdges(uint32_t flags) {
        options.openBoundaryEdges = flags;
        if(globalConstants.cpu) {
            globalConstants.cpu->open_boundary_edges = flags;
        }
        return *this;
    }

    std::vector<VkDescriptorSet> FluidSolver::debugFieldDescriptorSets() const {
        std::vector<VkDescriptorSet> sets{
            _vectorGrid->vectorField().u.descriptorSet[in],
            _vectorGrid->vectorField().v.descriptorSet[in],
            _pressureField.descriptorSet[in],
            _vectorGrid->divergenceField().descriptorSet[in],
            _vectorGrid->forceField().descriptorSet[in],
            _vorticityField.descriptorSet[in],
            _colliderField.descriptorSet[in],
        };

        for(const auto& quantity : _quantities) {
            sets.push_back(quantity.get().field.descriptorSet[in]);
            sets.push_back(quantity.get().source.descriptorSet[in]);
        }

        return sets;
    }

    FluidSolver::Builder::Builder(VulkanDevice *device, VulkanDescriptorPool *descriptorPool)
    : _device(device)
    , _descriptorPool(descriptorPool){}

    FluidSolver::Builder& FluidSolver::Builder::dt(float value) {
        _dt = value;
        return *this;
    }

    FluidSolver::Builder& FluidSolver::Builder::density(float rho) {
        _density = rho;
        return *this;
    }

    FluidSolver::Builder& FluidSolver::Builder::generate(const VectorFieldFunc2D& func) {
        return generate2D(func);
    }

    FluidSolver::Builder& FluidSolver::Builder::generate(const VectorFieldFunc3D& func) {
        return generate3D(func);
    }

    FluidSolver::Builder& FluidSolver::Builder::generate2D(const VectorFieldFunc2D& func) {
        _generator2D = func;
        _generator3D.reset();
        _dimension = 2;
        _imageType = VK_IMAGE_TYPE_3D;
        gridSize2D(glm::vec2(_gridSize));
        return *this;
    }

    FluidSolver::Builder& FluidSolver::Builder::generate3D(const VectorFieldFunc3D& func) {
        _generator3D = func;
        _generator2D.reset();
        _dimension = 3;
        _imageType = VK_IMAGE_TYPE_3D;
        return *this;
    }

    FluidSolver::Builder& FluidSolver::Builder::addExternalForce(ExternalForce&& force) {
        _externalForces.push_back(force);
        return *this;
    }

    FluidSolver::Builder& FluidSolver::Builder::poissonIterations(int iterations) {
        _poissonIterations = iterations;
        return *this;
    }

    FluidSolver::Builder & FluidSolver::Builder::diffuseIterations(int value) {
        _diffuseIterations = value;
        return *this;
    }

    FluidSolver::Builder& FluidSolver::Builder::viscosity(float value) {
        _viscosity = value;
        return *this;
    }

    FluidSolver::Builder& FluidSolver::Builder::enableWrapping() {
        _wrappingEnabled = true;
        _closedDomain = false;
        return *this;
    }

    FluidSolver::Builder & FluidSolver::Builder::disableWrapping() {
        _wrappingEnabled = false;
        return *this;
    }

    FluidSolver::Builder& FluidSolver::Builder::gridSize(glm::vec2 size) {
        return gridSize2D(size);
    }

    FluidSolver::Builder& FluidSolver::Builder::gridSize(glm::vec3 size) {
        return gridSize3D(size);
    }

    FluidSolver::Builder& FluidSolver::Builder::gridSize2D(glm::vec2 size) {
        _gridSize = glm::vec3(size, 1.0f);
        _dimension = 2;
        _imageType = VK_IMAGE_TYPE_3D;
        return *this;
    }

    FluidSolver::Builder& FluidSolver::Builder::gridSize3D(glm::vec3 size) {
        _gridSize = size;
        _dimension = 3;
        _imageType = VK_IMAGE_TYPE_3D;
        return *this;
    }

    FluidSolver::Builder& FluidSolver::Builder::closedDomain() {
        _closedDomain = true;
        _wrappingEnabled = false;
        return *this;
    }

    FluidSolver::Builder & FluidSolver::Builder::openDomain() {
        _closedDomain = false;
        return *this;
    }

    FluidSolver::Builder& FluidSolver::Builder::openBoundaryEdges(uint32_t flags) {
        _openBoundaryEdges = flags;
        return *this;
    }

    FluidSolver::Builder& FluidSolver::Builder::addCollider(VkDescriptorSet fieldDescriptorSet, VkDescriptorSet velocityDescriptorSet) {
        if(fieldDescriptorSet == VK_NULL_HANDLE) return *this;
        if(_colliders.size() >= maxColliderFields) {
            throw std::runtime_error{std::format("FluidSolver supports at most {} collider fields", maxColliderFields)};
        }

        _colliders.push_back({fieldDescriptorSet, velocityDescriptorSet});
        return *this;
    }

    FluidSolver::Builder& FluidSolver::Builder::addCollider(const Field& field, VkDescriptorSet velocityDescriptorSet) {
        return addCollider(field.descriptorSet[in], velocityDescriptorSet);
    }

    FluidSolver::Builder & FluidSolver::Builder::enableProjection() {
        _project = true;
        return *this;
    }

    FluidSolver::Builder & FluidSolver::Builder::disableProjection() {
        _project = false;
        return *this;
    }

    FluidSolver::Builder & FluidSolver::Builder::useMacCormackAdvection() {
        _macCormackAdvection = true;
        return *this;
    }

    FluidSolver::Builder & FluidSolver::Builder::useStandingAdvection() {
        _macCormackAdvection = false;
        return *this;
    }

    FluidSolver::Builder & FluidSolver::Builder::enableAdvection() {
        _advectVField = true;
        return *this;
    }

    FluidSolver::Builder & FluidSolver::Builder::disableAdvection() {
        _advectVField = false;
        return *this;
    }

    FluidSolver::Builder & FluidSolver::Builder::useJacobiSolver() {
        _linearSolverStrategy = LinearSolverStrategy::Jacobi;
        return *this;
    }

    FluidSolver::Builder & FluidSolver::Builder::useConjugateGradientSolver() {
        _linearSolverStrategy = LinearSolverStrategy::ConjugateGradient;
        return *this;
    }

    FluidSolver::Builder & FluidSolver::Builder::useGaussSeidelSolver() {
        _linearSolverStrategy = LinearSolverStrategy::RBGS;
        return *this;
    }

    FluidSolver::Builder & FluidSolver::Builder::vectorField(std::span<glm::vec2> field) {
        return vectorField2D(field);
    }

    FluidSolver::Builder& FluidSolver::Builder::vectorField(std::span<glm::vec3> field) {
        return vectorField3D(field);
    }

    FluidSolver::Builder& FluidSolver::Builder::vectorField2D(std::span<glm::vec2> field) {
        _data2D = { field.begin(), field.end() };
        _data3D.clear();
        _dimension = 2;
        _imageType = VK_IMAGE_TYPE_3D;
        gridSize2D(glm::vec2(_gridSize));
        return *this;
    }

    FluidSolver::Builder& FluidSolver::Builder::vectorField3D(std::span<glm::vec3> field) {
        _data3D = { field.begin(), field.end() };
        _data2D.clear();
        _dimension = 3;
        _imageType = VK_IMAGE_TYPE_3D;
        return *this;
    }

    FluidSolver::Builder& FluidSolver::Builder::vorticityConfinementScale(float scale) {
        _vorticityConfinementScale = scale;
        return *this;
    }

    FluidSolver::Builder& FluidSolver::Builder::addQuantity(Quantity &quantity) {
        _quantities.emplace_back(quantity);
        return *this;
    }

    FluidSolver::Builder& FluidSolver::Builder::addQuantityData(Quantity& quantity, std::string name, VkFormat format,
                                                                std::span<const std::byte> data) {
        assert(!data.empty());
        _quantityData.push_back({
            .quantity = quantity,
            .name = std::move(name),
            .format = format,
            .data = {data.begin(), data.end()}
        });
        return *this;
    }

    std::unique_ptr<FluidSolver> FluidSolver::Builder::build() {
        assert(_gridSize.x > 0 && _gridSize.y > 0 && _gridSize.z > 0);
        assert(_poissonIterations > 0);
        assert(_density >= 1);
        assert(_dt > 0);
        assert(_viscosity >= 0);

        auto solver = std::make_unique<FluidSolver>(_device, _descriptorPool, _gridSize, _dimension);
        solver->options.advectVField = _advectVField;
        solver->options.project = _project;
        solver->options.wrappingEnabled = _wrappingEnabled;
        solver->options.closedDomain = _closedDomain;
        solver->options.openBoundaryEdges = _openBoundaryEdges;
        solver->options.poissonIterations = _poissonIterations;
        solver->options.viscosity = _viscosity;
        solver->options.vorticityConfinementScale = _vorticityConfinementScale;
        solver->options.density = _density;
        solver->options.timeStep = _dt;
        solver->linearSolverStrategy = _linearSolverStrategy;
        solver->_gridSize = _gridSize;
        solver->_dimension = _dimension;
        solver->_colliders = _colliders;
        solver->_activeColliderCount = static_cast<uint32_t>(solver->_colliders.size());

        solver->init();
        solver->_externalForces = _externalForces;
        generateVectorField(*solver);
        addQuantities(*solver);

        return solver;
    }

    void FluidSolver::Builder::generateVectorField(FluidSolver& solver) {
        if(_generator3D.has_value()) {
            solver._vectorGrid->generate(*_generator3D);
        } else if(_generator2D.has_value()) {
            solver._vectorGrid->generate(*_generator2D);
        } else if(!_data3D.empty()) {
            solver._vectorGrid->fill(_data3D);
        } else if(!_data2D.empty()) {
            solver._vectorGrid->fill(_data2D);
        } else if(solver._dimension == 3u) {
            solver._vectorGrid->fill(glm::vec3{0});
        }else {
            solver._vectorGrid->fill(glm::vec2{0});
        }
    }

    void FluidSolver::Builder::addQuantities(FluidSolver &solver) {
        for(auto& quantityData : _quantityData) {
            initQuantityTextures(solver, quantityData.quantity, quantityData.name, quantityData.format, quantityData.data);
            _quantities.emplace_back(quantityData.quantity);
        }

        for(auto& quantity : _quantities) {
            auto writes = initializers::writeDescriptorSets<12>();
            auto offset = solver.createDescriptorSet(writes, 0, quantity.get().field);
            solver.createDescriptorSet(writes, offset, quantity.get().source);

            _device->updateDescriptorSets(writes);

            solver._quantities.emplace_back(quantity);
        }
    }

    void FluidSolver::Builder::initQuantityTextures(FluidSolver& solver, Quantity& quantity, const std::string& name,
                                                    VkFormat format, std::span<const std::byte> data) {
        const auto dimensions = glm::uvec3{
            static_cast<uint32_t>(solver._gridSize.x),
            static_cast<uint32_t>(solver._gridSize.y),
            static_cast<uint32_t>(solver._gridSize.z)
        };
        const auto texelCount = static_cast<std::size_t>(dimensions.x) * dimensions.y * dimensions.z;
        assert(texelCount > 0);
        assert(data.size() % texelCount == 0);

        quantity.name = name;
        quantity.field.name = name;
        quantity.source.name = name + "_source";

        std::vector<std::byte> sourceData(data.size(), std::byte{0});
        auto addressMode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        for(auto i = 0u; i < 2; ++i) {
            textures::create(*solver.device, quantity.field[i], solver._imageType, format,
                             const_cast<std::byte*>(data.data()), dimensions, addressMode, sizeof(float));
            textures::create(*solver.device, quantity.source[i], solver._imageType, format,
                             sourceData.data(), dimensions, addressMode, sizeof(float));

            quantity.field[i].image.transitionLayout(solver.device->graphicsCommandPool(), VK_IMAGE_LAYOUT_GENERAL);
            quantity.source[i].image.transitionLayout(solver.device->graphicsCommandPool(), VK_IMAGE_LAYOUT_GENERAL);

            solver.device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", quantity.field.name, i),
                                                         quantity.field[i].image.image);
            solver.device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", quantity.source.name, i),
                                                         quantity.source[i].image.image);
        }
    }
}
