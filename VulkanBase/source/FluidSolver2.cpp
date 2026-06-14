#include "fluid/FluidSolver2.hpp"
#include "Barrier.hpp"

#include <algorithm>

namespace eular {
    
    FluidSolver::FluidSolver(VulkanDevice *device, VulkanDescriptorPool* descriptorPool, glm::vec2 gridSize,
                             std::optional<VkDescriptorSet> optionalBoundaryDescriptorSet)
        : ComputePipelines(device)
        , _descriptorPool(descriptorPool)
        , _boundaryDescriptorSet(optionalBoundaryDescriptorSet.value_or(VK_NULL_HANDLE))
        , _useDefaultBoundaryTexture(!optionalBoundaryDescriptorSet.has_value() || *optionalBoundaryDescriptorSet == VK_NULL_HANDLE)
        , _gridSize(gridSize, 1)
        , _delta(1.f/gridSize, 0)
        , _imageType(VK_IMAGE_TYPE_2D){
        _groupCount.xy = glm::uvec2(glm::ceil(gridSize/32.f));
    }

    
    void FluidSolver::init() {
        initGlobalConstants();
        initConjugateGradientSupport();
        createSamplers();
        initFields();
        createDescriptorSetLayouts();
        updateDescriptorSets();
        createPipelines();
    }

    void FluidSolver::createSamplers() {
        VkSamplerAddressMode addressMode = globalConstants.cpu->ensure_boundary_condition == 1 ?
                                           VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE :VK_SAMPLER_ADDRESS_MODE_REPEAT;
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

        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        _linearSampler = device->createSampler(samplerInfo);
    }

    void FluidSolver::initGlobalConstants() {
        GlobalData data{};
        data.grid_size = glm::ivec3(_gridSize);
        data.dx = {_delta.x, 0};
        data.dy = {0, _delta.y};
        data.dt = options.timeStep;
        data.ensure_boundary_condition = static_cast<int>(options.ensureBoundaryCondition);
        globalConstants.gpu = device->createCpuVisibleBuffer(&data, sizeof(GlobalData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        globalConstants.cpu =  reinterpret_cast<GlobalData*>(globalConstants.gpu.map());
    }
    
    void FluidSolver::initFields() {
        auto size = glm::ivec3(_gridSize);

        _vectorField.u.name = "vector_field_u";
        _vectorField.v.name = "vector_field_v";
        _forceField.name = "force_field";
        _vorticityField.name = "vorticity_field";
        _divergenceField.name = "divergence_field";
        _pressureField.name = "pressure_field";
        _macCormackData.name = "macCormack_intermediate_data";

        auto addressMode = options.ensureBoundaryCondition ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_REPEAT;


        textures::createNoTransition(*device, _vectorField.u[0], VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _vectorField.u[1], VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _vectorField.v[0], VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _vectorField.v[1], VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, size, addressMode);

        textures::createNoTransition(*device, _forceField[0], VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _forceField[1], VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, size, addressMode);


        textures::createNoTransition(*device, _vorticityField[0], VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _vorticityField[1], VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, size, addressMode);

        textures::createNoTransition(*device, _divergenceField[0], VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _divergenceField[1], VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, size, addressMode);

        textures::createNoTransition(*device, _pressureField[0], VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _pressureField[1], VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, size, addressMode);

        textures::createNoTransition(*device, _macCormackData[0], VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, size, addressMode);
        textures::createNoTransition(*device, _macCormackData[1], VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, size, addressMode);

        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vectorField.u.name, 0), _vectorField.u[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vectorField.u.name, 1), _vectorField.u[1].image.image);

        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vectorField.v.name, 0), _vectorField.v[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vectorField.v.name, 1), _vectorField.v[1].image.image);

        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _forceField.name, 0), _forceField[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _forceField.name, 1), _forceField[1].image.image);

        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vorticityField.name, 0), _vorticityField[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _vorticityField.name, 1), _vorticityField[1].image.image);

        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _divergenceField.name, 0), _divergenceField[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _divergenceField.name, 1), _divergenceField[1].image.image);

        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _pressureField.name, 0), _pressureField[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _pressureField.name, 1), _pressureField[1].image.image);

        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _macCormackData.name, 0), _macCormackData[0].image.image);
        device->setName<VK_OBJECT_TYPE_IMAGE>(std::format("{}_{}", _macCormackData.name, 1), _macCormackData[1].image.image);

        prepTextures();
    }

    bool FluidSolver::isJacobiSolver() const {
        return linearSolverStrategy == LinearSolverStrategy::Jacobi;
    }

    bool FluidSolver::isRbgsSolver() const {
        return linearSolverStrategy == LinearSolverStrategy::RBGS;
    }

    bool FluidSolver::isConjugateGradientSolver() const {
        return linearSolverStrategy == LinearSolverStrategy::ConjugateGradient;
    }

    void FluidSolver::initConjugateGradientSupport() {
        if (!isConjugateGradientSolver()) return;

        auto unknownCount = static_cast<uint32_t>(_gridSize.x * _gridSize.y);
        auto vectorSize = static_cast<VkDeviceSize>(unknownCount) * sizeof(float);
        auto maxNonZeroCount = static_cast<VkDeviceSize>(unknownCount) * cgStencilEntriesPerRow;

        for (auto i = 0; i < 2; ++i) {
            auto& cg = _cg[i];
            auto& A = cg.params.Coefficients;

            A.values = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY,
                                            maxNonZeroCount * sizeof(float), fmt::format("fluid_cg_values_{}", i));
            A.colIndices = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY,
                                                maxNonZeroCount * sizeof(uint32_t), fmt::format("fluid_cg_column_indices_{}", i));
            A.rowOffsets = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY,
                                                (static_cast<VkDeviceSize>(unknownCount) + 1) * sizeof(uint32_t), fmt::format("fluid_cg_row_offsets_{}", i));
            A.counts = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY,
                                            3 * sizeof(uint32_t), fmt::format("fluid_cg_counts_{}", i));
            A.numRows = unknownCount;
            A.numCols = unknownCount;

            cg.params.unknown = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                     VMA_MEMORY_USAGE_GPU_ONLY, vectorSize, fmt::format("fluid_cg_unknown_{}", i));
            cg.params.solution = device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                      VMA_MEMORY_USAGE_GPU_ONLY, vectorSize, fmt::format("fluid_cg_rhs_{}", i));

            cg.solver = gpu::linalg::ConjugateGradientSolver{*device};
            cg.solver.init(vectorSize);
            cg.params.numIterations = options.poissonIterations;
            cg.params.id = i;
            cg.constants.gridSize = glm::uvec2(_gridSize);
            cg.constants.ensureBoundaryCondition = static_cast<uint32_t>(options.ensureBoundaryCondition);
        }

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

        _fieldDescriptorSetLayout =
            device->descriptorSetLayoutBuilder()
                .name("fluid_solver_field_set_layout")
                .binding(0)
                    .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
                    .immutableSamplers(_linearSampler)
                .binding(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .binding(2)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                .createLayout();

        _imageDescriptorSetLayout =
            device->descriptorSetLayoutBuilder()
                .name("fluid_solver_image_set_layout")
                .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();

        _textureDescriptorSetLayout =
            device->descriptorSetLayoutBuilder()
                .name("fluid_solver_texture_set_layout")
                .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();

        _samplerDescriptorSetLayout =
            device->descriptorSetLayoutBuilder()
                .name("fluid_solver_sampler_set_layout")
                .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_SAMPLER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();

        _boundaryDescriptorSetLayout =
            device->descriptorSetLayoutBuilder()
                .name("fluid_solver_boundary_texture")
                .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                .descriptorCount(1)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();

        if(_useDefaultBoundaryTexture) {
            createDefaultBoundaryTexture();
        }

        _debugDescriptorSetLayout =
            device->descriptorSetLayoutBuilder()
                .name("fluid_solver_debug_set_layout")
                .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                .descriptorCount(20)
                .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();

        cgDescriptorSetLayout =
            device->descriptorSetLayoutBuilder()
                .name("fluid_cg_stencil_matrix_descriptor_set_layout")
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

        cgVectorDescriptorSetLayout =
            device->descriptorSetLayoutBuilder()
                .name("fluid_cg_vector_descriptor_set_layout")
                .binding(0)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();
    }

    void FluidSolver::updateDescriptorSets() {
        std::vector<VulkanDescriptorSetLayout> layouts{ uniformsSetLayout, _samplerDescriptorSetLayout, _samplerDescriptorSetLayout };
        if(_useDefaultBoundaryTexture) {
            layouts.push_back(_boundaryDescriptorSetLayout);
        }

        const auto cgMatrixSetOffset = layouts.size();
        if (isConjugateGradientSolver()) {
            layouts.push_back(cgDescriptorSetLayout);
            layouts.push_back(cgDescriptorSetLayout);
            layouts.push_back(cgVectorDescriptorSetLayout);
            layouts.push_back(cgVectorDescriptorSetLayout);
        }

        auto sets = _descriptorPool->allocate(layouts);
        uniformDescriptorSet = sets[0];
        _valueSamplerDescriptorSet = sets[1];
        _linearSamplerDescriptorSet = sets[2];

        if(_useDefaultBoundaryTexture) {
            _boundaryDescriptorSet = sets[3];
        }
        if (isConjugateGradientSolver()) {
            _cg[0].descriptorSet = sets[cgMatrixSetOffset];
            _cg[1].descriptorSet = sets[cgMatrixSetOffset + 1];
            _cg[0].rhsDescriptorSet = sets[cgMatrixSetOffset + 2];
            _cg[1].rhsDescriptorSet = sets[cgMatrixSetOffset + 3];
        }

        auto writes = initializers::writeDescriptorSets<64>();

        auto writeOffset = 0u;

        writes[writeOffset].dstSet = uniformDescriptorSet;
        writes[writeOffset].dstBinding = 0;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pBufferInfo = new VkDescriptorBufferInfo{ globalConstants.gpu, 0, VK_WHOLE_SIZE };
        ++writeOffset;

        writes[writeOffset].dstSet = _valueSamplerDescriptorSet;
        writes[writeOffset].dstBinding = 0;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo { _valueSampler.handle, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
        ++writeOffset;

        writes[writeOffset].dstSet = _linearSamplerDescriptorSet;
        writes[writeOffset].dstBinding = 0;
        writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        writes[writeOffset].descriptorCount = 1;
        writes[writeOffset].pImageInfo = new VkDescriptorImageInfo { _linearSampler.handle, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
        ++writeOffset;

        if(_useDefaultBoundaryTexture) {
            writes[writeOffset].dstSet = _boundaryDescriptorSet;
            writes[writeOffset].dstBinding = 0;
            writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[writeOffset].descriptorCount = 1;
            writes[writeOffset].pImageInfo = new VkDescriptorImageInfo {
                    _valueSampler.handle,
                    _defaultBoundaryTexture.imageView.handle,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            };
            ++writeOffset;
        }

        if (isConjugateGradientSolver()) {
            for (auto i = 0; i < 2; ++i) {
                const std::array<VulkanBuffer*, 4> buffers{
                    &_cg[i].params.Coefficients.values,
                    &_cg[i].params.Coefficients.colIndices,
                    &_cg[i].params.Coefficients.rowOffsets,
                    &_cg[i].params.Coefficients.counts
                };

                for(auto binding = 0u; binding < buffers.size(); ++binding) {
                    writes[writeOffset].dstSet = _cg[i].descriptorSet;
                    writes[writeOffset].dstBinding = binding;
                    writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    writes[writeOffset].descriptorCount = 1;
                    writes[writeOffset].pBufferInfo = new VkDescriptorBufferInfo { *buffers[binding], 0, VK_WHOLE_SIZE };
                    ++writeOffset;
                }

                writes[writeOffset].dstSet = _cg[i].rhsDescriptorSet;
                writes[writeOffset].dstBinding = 0;
                writes[writeOffset].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[writeOffset].descriptorCount = 1;
                writes[writeOffset].pBufferInfo = new VkDescriptorBufferInfo { _cg[i].params.solution, 0, VK_WHOLE_SIZE };
                ++writeOffset;
            }
        }

        writeOffset = createDescriptorSet(writes, writeOffset, _vectorField.u);
        writeOffset = createDescriptorSet(writes, writeOffset, _vectorField.v);
        writeOffset = createDescriptorSet(writes, writeOffset, _divergenceField);
        writeOffset = createDescriptorSet(writes, writeOffset, _pressureField);
        writeOffset = createDescriptorSet(writes, writeOffset, _forceField);
        writeOffset = createDescriptorSet(writes, writeOffset, _vorticityField);
        writeOffset = createDescriptorSet(writes, writeOffset, _macCormackData);

        writes.resize(writeOffset);
        device->updateDescriptorSets(writes);

        for(auto& write : writes) {
            if (write.pImageInfo) delete write.pImageInfo;
            if (write.pBufferInfo) delete write.pBufferInfo;

        }
    }

    uint32_t FluidSolver::createDescriptorSet(std::vector<VkWriteDescriptorSet>& writes, uint32_t writeOffset, Field& field) {
        auto sets = _descriptorPool->allocate( { _fieldDescriptorSetLayout, _fieldDescriptorSetLayout});

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

    void FluidSolver::createDefaultBoundaryTexture() {
        const auto width = static_cast<uint32_t>(_gridSize.x);
        const auto height = static_cast<uint32_t>(_gridSize.y);

        std::vector<float> boundary(width * height, 0.0f);
        for(auto y = 0u; y < height; ++y) {
            for(auto x = 0u; x < width; ++x) {
                if(x == 0 || y == 0 || x == width - 1 || y == height - 1) {
                    boundary[y * width + x] = 1.0f;
                }
            }
        }

        textures::create(*device, _defaultBoundaryTexture, VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT,
                         boundary.data(), {width, height, 1u}, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, sizeof(float));
        device->setName<VK_OBJECT_TYPE_IMAGE>("fluid_solver_default_boundary_texture", _defaultBoundaryTexture.image.image);
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

            barrier.image = _forceField[0].image;
            barriers.push_back(barrier);

            barrier.image = _vorticityField[0].image;
            barriers.push_back(barrier);

            barrier.image = _divergenceField[0].image;
            barriers.push_back(barrier);

            barrier.image = _pressureField[0].image;
            barriers.push_back(barrier);

            barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

            barrier.image = _vectorField.u[0].image;
            barriers.push_back(barrier);

            barrier.image = _vectorField.v[0].image;
            barriers.push_back(barrier);

            barrier.image = _vectorField.u[1].image;
            barriers.push_back(barrier);

            barrier.image = _vectorField.v[1].image;
            barriers.push_back(barrier);

            barrier.image = _forceField[1].image;
            barriers.push_back(barrier);

            barrier.image = _vorticityField[1].image;
            barriers.push_back(barrier);

            barrier.image = _divergenceField[1].image;
            barriers.push_back(barrier);

            barrier.image = _pressureField[1].image;
            barriers.push_back(barrier);

            barrier.image = _macCormackData[0].image;
            barriers.push_back(barrier);

            barrier.image = _macCormackData[1].image;
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

            clearTexture(_vectorField.u[0]);
            clearTexture(_vectorField.u[1]);
            clearTexture(_vectorField.v[0]);
            clearTexture(_vectorField.v[1]);
            clearTexture(_forceField[0]);
            clearTexture(_forceField[1]);
            clearTexture(_vorticityField[0]);
            clearTexture(_vorticityField[1]);
            clearTexture(_divergenceField[0]);
            clearTexture(_divergenceField[1]);
            clearTexture(_pressureField[0]);
            clearTexture(_pressureField[1]);
            clearTexture(_macCormackData[0]);
            clearTexture(_macCormackData[1]);

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
                    .name = "advect",
                    .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\advect.comp.spv)",
                    .layouts =  {
                            &uniformsSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                            &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout, &_samplerDescriptorSetLayout,
                            &_boundaryDescriptorSetLayout
                    },
                    .ranges = { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(advectConstants) } }
                },
                {
                    .name = "apply_force",
                    .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\apply_force.comp.spv)",
                    .layouts =  {
                            &uniformsSetLayout,  &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                            &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                            &_boundaryDescriptorSetLayout
                    }
                },
                {
                        .name = "add_sources",
                        .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\add_sources.comp.spv)",
                        .layouts =  {
                                &uniformsSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                                &_fieldDescriptorSetLayout, &_boundaryDescriptorSetLayout
                        }
                },
                {
                    .name = "jacobi",
                    .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\jacobi.comp.spv)",
                    .layouts =  {
                            &uniformsSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                            &_fieldDescriptorSetLayout, &_boundaryDescriptorSetLayout
                    },
                    .ranges = { {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(linearSolverConstants) } }
                },
                {
                    .name = "rbgs",
                    .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\rbgs.comp.spv)",
                    .layouts =  {
                            &uniformsSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                            &_fieldDescriptorSetLayout, &_boundaryDescriptorSetLayout,
                    },
                    .ranges = { {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(linearSolverConstants) } }
                },
                {
                    .name = "divergence",
                    .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\divergence.comp.spv)",
                    .layouts =  {
                            &uniformsSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                            &_fieldDescriptorSetLayout, &_boundaryDescriptorSetLayout
                     }
                },
                {
                    .name = "divergence_free_field",
                    .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\divergence_free_field.comp.spv)",
                    .layouts =  {
                            &uniformsSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                            &_fieldDescriptorSetLayout,  &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                            &_boundaryDescriptorSetLayout
                      }
                },
                {
                    .name = "vorticity",
                    .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\vorticity.comp.spv)",
                    .layouts =  {
                            &uniformsSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                            &_fieldDescriptorSetLayout, &_boundaryDescriptorSetLayout
                      }
                },
                {
                    .name = "vorticity_force",
                    .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\vorticity_force.comp.spv)",
                    .layouts =  {
                            &uniformsSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                            &_fieldDescriptorSetLayout, &_boundaryDescriptorSetLayout
                      },
                      .ranges = { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float) } }
                },
                {
                    .name = "maccormack",
                    .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\maccormack_advection.comp.spv)",
                    .layouts =  {
                            &uniformsSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                            &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout, &_fieldDescriptorSetLayout,
                            &_fieldDescriptorSetLayout, &_boundaryDescriptorSetLayout
                      },
                      .ranges = { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(advectConstants) } }
                },
                {
                    .name = "generate_coefficients",
                    .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\generate_coefficients.comp.spv)",
                    .layouts = { &_boundaryDescriptorSetLayout, &cgDescriptorSetLayout },
                    .ranges = { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(_cg[0].constants) } }
                },
                {
                    .name = "copy_scaled_to_buffer",
                    .shadePath = R"(C:\Users\joebh\CLionProjects\vglib\dependencies\vglib.github.io\data\shaders\fluid_2d\copy_scaled_to_buffer.comp.spv)",
                    .layouts = { &_fieldDescriptorSetLayout, &cgVectorDescriptorSetLayout },
                    .ranges = { { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ScaledFieldCopyConstants) } }
                },
        };
    }

    
    void FluidSolver::velocityStep(VkCommandBuffer commandBuffer) {
        if(!options.advectVField) return;

        VULKAN_COMMAND_BUFFER_SECTION(device, commandBuffer, velocity_step);
        clearForces(commandBuffer);
        applyForces(commandBuffer);
        advectVectorField(commandBuffer);
        diffuseVelocityField(commandBuffer);
        project(commandBuffer);
    }

    
    void FluidSolver::clearForces(VkCommandBuffer commandBuffer) {
        clear(commandBuffer, _forceField[0]);
        clear(commandBuffer, _forceField[1]);
    }

    
    void FluidSolver::applyForces(VkCommandBuffer commandBuffer) {
        VULKAN_COMMAND_BUFFER_SECTION(device, commandBuffer, apply_forces);
        applyExternalForces(commandBuffer);
        computeVorticityConfinement(commandBuffer);
        addForcesToVectorField(commandBuffer, _forceField);
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
        VULKAN_COMMAND_BUFFER_SECTION(device, commandBuffer, diffuse);
        diffuse(commandBuffer, _vectorField.u, options.viscosity/rho, 1);
        diffuse(commandBuffer, _vectorField.v, options.viscosity/rho, 2);
        linearSolverConstants.vector_field_component = 0;
        if(!isConjugateGradientSolver()) {
            addComputeBarrier(commandBuffer, {&_vectorField.u[in], &_vectorField.v[in]});
        }
        project(commandBuffer);
    }

    void FluidSolver::diffuse(VkCommandBuffer commandBuffer, Field& field, float rate, uint32_t vectorFieldComponent) {
        if(rate <= 0) return;
        const auto dt = options.timeStep;
        linearSolverConstants.alpha = (_delta.x * _delta.x * _delta.x * _delta.y)/(dt * rate);
        linearSolverConstants.rBeta = 1.0f/((2.0f * glm::dot(_delta, _delta)) + linearSolverConstants.alpha);
        linearSolverConstants.vector_field_component = vectorFieldComponent;
        if(isJacobiSolver()) {
            jacobiSolver(commandBuffer, field, field);
        }else if(isRbgsSolver()) {
            rbgsSolver(commandBuffer, field, field);
        }else {
            const auto index = vectorFieldComponent == 2 ? 1u : 0u;
            assign(commandBuffer, field[in], _cg[index].params.solution);
            assign(commandBuffer, field[in], _cg[index].params.unknown);
            setDiffuseConstants(index, rate, vectorFieldComponent);
            conjugateGradientSolve(commandBuffer, index);
            assign(commandBuffer, _cg[index].params.unknown, field[out]);
            field.swap();
        }
    }

    void FluidSolver::project(VkCommandBuffer commandBuffer) {
        if(!options.project) return;
        VULKAN_COMMAND_BUFFER_SECTION(device, commandBuffer, projection);
        computeDivergence(commandBuffer);
        solvePressure(commandBuffer);
        computeDivergenceFreeField(commandBuffer);
    }

    void FluidSolver::advectVectorField(VkCommandBuffer commandBuffer) {
        VULKAN_COMMAND_BUFFER_SECTION(device, commandBuffer, advect);
        if(options.macCormackAdvection) {
            advect(commandBuffer, _vectorField.u, 1);
            advect(commandBuffer, _vectorField.v, 2);
        }else {
            advect(commandBuffer, _vectorField.u, 1, false);
            advect(commandBuffer, _vectorField.v, 2, false);
            addComputeBarrier(commandBuffer, {&_vectorField.u[out], &_vectorField.v[out]});
        }
        _vectorField.swap();

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
        advect(commandBuffer, field.descriptorSet[in], _macCormackData.descriptorSet[in], TimeDirection::Forward, boundaryMode, &_macCormackData[in]);
        advect(commandBuffer, _macCormackData.descriptorSet[in], _macCormackData.descriptorSet[out], TimeDirection::Backword, boundaryMode, &_macCormackData[out]);

        auto& vf = _vectorField;
        static std::array<VkDescriptorSet, 8> sets;

        sets[0] = uniformDescriptorSet;
        sets[1] = vf.u.descriptorSet[in];
        sets[2] = vf.v.descriptorSet[in];
        sets[3] = _macCormackData.descriptorSet[in];
        sets[4] = _macCormackData.descriptorSet[out];
        sets[5] = field.descriptorSet[in];
        sets[6] = field.descriptorSet[out];
        sets[7] = _boundaryDescriptorSet;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("maccormack"));
        advectConstants.time_sign = 1.0f;
        advectConstants.boundary_mode = boundaryMode;
        vkCmdPushConstants(commandBuffer, layout("maccormack"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(advectConstants), &advectConstants);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("maccormack"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
        addComputeBarrier(commandBuffer, field[out]);
    }

    void FluidSolver::advect(VkCommandBuffer commandBuffer, Field& field, uint32_t boundaryMode, bool addBarrier) {
        if(options.macCormackAdvection){
            macCormackAdvect(commandBuffer, field, boundaryMode);
        }else {
            advect(commandBuffer, field.descriptorSet[in], field.descriptorSet[out], TimeDirection::Forward, boundaryMode, addBarrier ? &field[out] : nullptr);
        }
    }

    void FluidSolver::advect(VkCommandBuffer commandBuffer, VkDescriptorSet inDescriptor, VkDescriptorSet outDescriptor,
                             TimeDirection timeDirection, uint32_t boundaryMode, Texture* writeTexture) {

        auto& vf = _vectorField;
        static std::array<VkDescriptorSet, 7> sets;

        sets[0] = uniformDescriptorSet;
        sets[1] = vf.u.descriptorSet[in];
        sets[2] = vf.v.descriptorSet[in];
        sets[3] = inDescriptor;
        sets[4] = outDescriptor;
        sets[5] = _linearSamplerDescriptorSet;
        sets[6] = _boundaryDescriptorSet;

        advectConstants.time_sign = timeDirection == TimeDirection::Forward ? 1.f : -1.f;
        advectConstants.boundary_mode = boundaryMode;
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("advect"));
        vkCmdPushConstants(commandBuffer, layout("advect"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(advectConstants), &advectConstants);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("advect"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
        if(writeTexture) {
            addComputeBarrier(commandBuffer, *writeTexture);
        }
    }


    void FluidSolver::applyExternalForces(VkCommandBuffer commandBuffer) {
        static std::array<VkDescriptorSet, 2> sets;
        for(const auto& externalForce : _externalForces){
            sets[0] = _forceField.descriptorSet[in];
            sets[1] = _forceField.descriptorSet[out];
            externalForce(commandBuffer, sets, _groupCount);
            addComputeBarrier(commandBuffer, _forceField[out]);
            _forceField.swap();
        }
    }

    void FluidSolver::addForcesToVectorField(VkCommandBuffer commandBuffer, ForceField &sourceField) {
        static std::array<VkDescriptorSet, 7> sets;
        sets[0] = uniformDescriptorSet;
        sets[1] = _vectorField.u.descriptorSet[in];
        sets[2] = _vectorField.v.descriptorSet[in];
        sets[3] = sourceField.descriptorSet[in];
        sets[4] = _vectorField.u.descriptorSet[out];
        sets[5] = _vectorField.v.descriptorSet[out];
        sets[6] = _boundaryDescriptorSet;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("apply_force"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("apply_force"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
        addComputeBarrier(commandBuffer, {&_vectorField.u[out], &_vectorField.v[out]});

        _vectorField.swap();
    }

    void FluidSolver::computeVorticityConfinement(VkCommandBuffer commandBuffer) {
        if(options.vorticityConfinementScale < 1) return;
        computeVorticity(commandBuffer);
        applyVorticity(commandBuffer);
    }

    void FluidSolver::jacobiSolver(VkCommandBuffer commandBuffer, Field& solution, Field& unknown) {
        static std::array<VkDescriptorSet, 5> sets;
        sets[0] = uniformDescriptorSet;
        sets[1] = solution.descriptorSet[in];
        sets[4] = _boundaryDescriptorSet;

        const auto N = options.poissonIterations;
        for(auto i = 0; i < N; ++i) {
            sets[2] = unknown.descriptorSet[in];
            sets[3] = unknown.descriptorSet[out];

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("jacobi"));
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("jacobi"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
            vkCmdPushConstants(commandBuffer, layout("jacobi"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(linearSolverConstants), &linearSolverConstants);
            vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);

            if(i < N - 1) {
                addComputeBarrier(commandBuffer, unknown[out]);
            }

            unknown.swap();
        }
    }

    void FluidSolver::rbgsSolver(VkCommandBuffer commandBuffer, Field& solution, Field& unknown) {
        static std::array<VkDescriptorSet, 5> sets;
        sets[0] = uniformDescriptorSet;
        sets[1] = solution.descriptorSet[in];
        sets[4] = _boundaryDescriptorSet;

        const auto N = options.poissonIterations;

        for(auto i = 0; i < N; ++i) {
            sets[2] = unknown.descriptorSet[in];
            sets[3] = unknown.descriptorSet[out];

            linearSolverConstants.pass = 0;
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("rbgs"));
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("rbgs"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
            vkCmdPushConstants(commandBuffer, layout("rbgs"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(linearSolverConstants), &linearSolverConstants);
            vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
            addComputeBarrier(commandBuffer, unknown[out]);
            unknown.swap();

            sets[2] = unknown.descriptorSet[in];
            sets[3] = unknown.descriptorSet[out];

            linearSolverConstants.pass = 1;
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("rbgs"));
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("rbgs"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
            vkCmdPushConstants(commandBuffer, layout("rbgs"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(linearSolverConstants), &linearSolverConstants);
            vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);

            if(i < N - 1) {
                addComputeBarrier(commandBuffer, unknown[out]);
            }
            unknown.swap();
        }
    }

    void FluidSolver::conjugateGradientSolve(VkCommandBuffer commandBuffer, uint32_t index) {
        buildCoefficientMatrix(commandBuffer, index);
        _cg[index].solver.solve(commandBuffer, _cg[index].params);
    }

    void FluidSolver::buildCoefficientMatrix(VkCommandBuffer commandBuffer, uint32_t index) {
        auto unknownCount = _cg[index].params.Coefficients.numRows;
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("generate_coefficients"));
        const std::array<VkDescriptorSet, 2> sets{_boundaryDescriptorSet, _cg[index].descriptorSet};
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("generate_coefficients"), 0,
                                COUNT(sets), sets.data(), 0, nullptr);

        for(auto offset = 0u; offset < unknownCount; offset += cgRowsPerBatch) {
            auto batchSize = std::min(cgRowsPerBatch, unknownCount - offset);
            _cg[index].constants.batchOffset = offset;
            _cg[index].constants.batchSize = batchSize;
            _cg[index].constants.ensureBoundaryCondition = static_cast<uint32_t>(options.ensureBoundaryCondition);
            vkCmdPushConstants(commandBuffer, layout("generate_coefficients"), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(_cg[index].constants), &_cg[index].constants);
            vkCmdDispatch(commandBuffer, (batchSize + 31u) / 32u, 1, 1);
        }

        auto& A = _cg[index].params.Coefficients;
        Barrier::computeWriteToRead(commandBuffer, {A.values, A.colIndices, A.rowOffsets, A.counts});
    }

    void FluidSolver::setDiffuseConstants(uint32_t index, float rate, uint32_t vectorFieldComponent) {
        auto& constants = _cg[index].constants;
        constants.identity = 1.0f;
        constants.vectorFieldComponent = vectorFieldComponent;

        const auto rateTime = options.timeStep * rate;
        constants.alpha = {
            rateTime / (_delta.x * _delta.x),
            rateTime / (_delta.y * _delta.y)
        };
    }

    void FluidSolver::setPressureConstants(uint32_t index) {
        auto& constants = _cg[index].constants;
        constants.identity = 0.0f;
        constants.vectorFieldComponent = 0;
        constants.alpha = {
            _delta.y * _delta.y,
            _delta.x * _delta.x
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
        region.imageExtent = {from.width, from.height, 1};

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
            .count = static_cast<uint32_t>(_gridSize.x * _gridSize.y)
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
        region.imageExtent = {to.width, to.height, 1};

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
        VULKAN_COMMAND_BUFFER_SECTION(device, commandBuffer, compute_divergence);
        auto& vf = _vectorField;
        static std::array<VkDescriptorSet, 5> sets;

        sets[0] = uniformDescriptorSet;
        sets[1] = vf.u.descriptorSet[in];
        sets[2] = vf.v.descriptorSet[in];
        sets[3] = _divergenceField.descriptorSet[in];
        sets[4] = _boundaryDescriptorSet;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("divergence"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("divergence"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
        addComputeBarrier(commandBuffer, _divergenceField[in]);
    }

    void FluidSolver::solvePressure(VkCommandBuffer commandBuffer) {
        VULKAN_COMMAND_BUFFER_SECTION(device, commandBuffer, solve_pressure);
        const auto rho = options.density;
        const auto dt = options.timeStep;
        linearSolverConstants.alpha = -(rho * _delta.x * _delta.x * _delta.y * _delta.y)/dt;
        linearSolverConstants.rBeta = (1.0f/(2.0f * glm::dot(_delta, _delta)));
        linearSolverConstants.vector_field_component = false;

        if(isJacobiSolver()) {
            jacobiSolver(commandBuffer, _divergenceField, _pressureField);
        }else if (isRbgsSolver()){
            rbgsSolver(commandBuffer, _divergenceField, _pressureField);
        }else if (isConjugateGradientSolver()){
            constexpr auto index = 0u;
            const auto pressureScale = -(rho * _delta.x * _delta.x * _delta.y * _delta.y) / dt;

            assignScaled(commandBuffer, _divergenceField, _cg[index].params.solution, _cg[index].rhsDescriptorSet, pressureScale);
            assign(commandBuffer, _pressureField[in], _cg[index].params.unknown);
            setPressureConstants(index);
            conjugateGradientSolve(commandBuffer, index);
            assign(commandBuffer, _cg[index].params.unknown, _pressureField[out]);
            _pressureField.swap();
        }
        addComputeBarrier(commandBuffer, _pressureField[in]);
        if (!isConjugateGradientSolver()) {
        }
    }

    void FluidSolver::computeDivergenceFreeField(VkCommandBuffer commandBuffer) {
        VULKAN_COMMAND_BUFFER_SECTION(device, commandBuffer, compute_divergence_free_field);
        auto& vf = _vectorField;
        static std::array<VkDescriptorSet, 7> sets;

        sets[0] = uniformDescriptorSet;
        sets[1] = vf.u.descriptorSet[in];
        sets[2] = vf.v.descriptorSet[in];
        sets[3] = _pressureField.descriptorSet[in];
        sets[4] = vf.u.descriptorSet[out];
        sets[5] = vf.v.descriptorSet[out];
        sets[6] = _boundaryDescriptorSet;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("divergence_free_field"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("divergence_free_field"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
        addComputeBarrier(commandBuffer, {&vf.u[out], &vf.v[out]});
        vf.swap();
    }


    float FluidSolver::dt() const {
        return options.timeStep;
    }

    FluidSolver& FluidSolver::density(float rho) {
        options.density = glm::max(1.f, rho);
        return *this;
    }

    void FluidSolver::runSimulation(VkCommandBuffer commandBuffer) {
        velocityStep(commandBuffer);
        quantityStep(commandBuffer);
        _elapsedTime += options.timeStep;
    }

    std::vector<VulkanDescriptorSetLayout> FluidSolver::forceFieldSetLayouts() {
        return  { _fieldDescriptorSetLayout, _fieldDescriptorSetLayout };
    }

    std::vector<VulkanDescriptorSetLayout> FluidSolver::sourceFieldSetLayouts() {
        return  { _fieldDescriptorSetLayout, _fieldDescriptorSetLayout };
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
        sets[4] = _boundaryDescriptorSet;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("add_sources"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("add_sources"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
        addComputeBarrier(commandBuffer, quantity.field[out]);
        quantity.field.swap();
    }

    void FluidSolver::diffuseQuantity(VkCommandBuffer commandBuffer, Quantity &quantity) {
        linearSolverConstants.vector_field_component = false;
        diffuse(commandBuffer, quantity.field, quantity.diffuseRate);
        if(quantity.diffuseRate > 0 && !isConjugateGradientSolver()) {
            addComputeBarrier(commandBuffer, quantity.field[in]);
        }
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
        static std::array<VkDescriptorSet, 5> sets;
        sets[0] = uniformDescriptorSet;
        sets[1] = _vectorField.u.descriptorSet[in];
        sets[2] = _vectorField.v.descriptorSet[in];
        sets[3] = _vorticityField.descriptorSet[in];
        sets[4] = _boundaryDescriptorSet;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("vorticity"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("vorticity"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
        addComputeBarrier(commandBuffer, _vorticityField[in]);
    }

    void FluidSolver::applyVorticity(VkCommandBuffer commandBuffer) {
        static std::array<VkDescriptorSet, 5> sets;
        sets[0] = uniformDescriptorSet;
        sets[1] = _vorticityField.descriptorSet[in];
        sets[2] = _forceField.descriptorSet[in];
        sets[3] = _forceField.descriptorSet[out];
        sets[4] = _boundaryDescriptorSet;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("vorticity_force"));
        vkCmdPushConstants(commandBuffer, layout("vorticity_force"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float), &options.vorticityConfinementScale);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("vorticity_force"), 0, COUNT(sets), sets.data(), 0, VK_NULL_HANDLE);
        vkCmdDispatch(commandBuffer, _groupCount.x, _groupCount.y, _groupCount.z);
        addComputeBarrier(commandBuffer, _forceField[out]);
        _forceField.swap();
    }

    VulkanDescriptorSetLayout FluidSolver::fieldDescriptorSetLayout() const {
        return _fieldDescriptorSetLayout;
    }

    float FluidSolver::elapsedTime() const {
        return _elapsedTime;
    }

    VectorField &FluidSolver::vectorField() {
        return _vectorField;
    }

    PressureField &FluidSolver::pressureField() {
        return _pressureField;
    }

    Texture &FluidSolver::boundaryTexture() {
        return _defaultBoundaryTexture;
    }

    const Texture &FluidSolver::boundaryTexture() const {
        return _defaultBoundaryTexture;
    }

    std::vector<VkDescriptorSet> FluidSolver::debugFieldDescriptorSets() const {
        std::vector<VkDescriptorSet> sets{
            _vectorField.u.descriptorSet[in],
            _vectorField.v.descriptorSet[in],
            _pressureField.descriptorSet[in],
            _divergenceField.descriptorSet[in],
            _forceField.descriptorSet[in],
            _vorticityField.descriptorSet[in],
            _macCormackData.descriptorSet[in],
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
        _generator = func;
        return *this;
    }

    FluidSolver::Builder& FluidSolver::Builder::add(ExternalForce&& force) {
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

    FluidSolver::Builder& FluidSolver::Builder::ensureBoundaryCondition(bool flag) {
        _ensureBoundaryCondition = flag;
        return *this;
    }

    FluidSolver::Builder& FluidSolver::Builder::gridSize(glm::vec2 size) {
        _gridSize = size;
        return *this;
    }

    FluidSolver::Builder& FluidSolver::Builder::boundary(VkDescriptorSet descriptorSet) {
        _boundaryDescriptorSet = descriptorSet;
        return *this;
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

    FluidSolver::Builder& FluidSolver::Builder::vorticityConfinementScale(float scale) {
        _vorticityConfinementScale = scale;
        return *this;
    }

    FluidSolver::Builder& FluidSolver::Builder::add(Quantity &quantity) {
        _quantities.emplace_back(quantity);
        return *this;
    }

    std::unique_ptr<FluidSolver> FluidSolver::Builder::build() {
        assert(_gridSize.x > 0 && _gridSize.y > 0);
        assert(_poissonIterations > 0);
        assert(_density >= 1);
        assert(_dt > 0);
        assert(_viscosity >= 0);

        auto solver = std::make_unique<FluidSolver>(_device, _descriptorPool, _gridSize, _boundaryDescriptorSet);
        solver->options.advectVField = _advectVField;
        solver->options.project = _project;
        solver->options.ensureBoundaryCondition = _ensureBoundaryCondition;
        solver->options.poissonIterations = _poissonIterations;
        solver->options.viscosity = _viscosity;
        solver->options.vorticityConfinementScale = _vorticityConfinementScale;
        solver->options.density = _density;
        solver->options.timeStep = _dt;
        solver->linearSolverStrategy = _linearSolverStrategy;
        solver->_gridSize = glm::vec3(_gridSize, 1);

        solver->init();
        solver->_externalForces = _externalForces;
        generateVectorField(*solver);
        addQuantities(*solver);

        return solver;
    }

    void FluidSolver::Builder::generateVectorField(FluidSolver& solver) {
        if(!_generator.has_value()) return;

        auto func = *_generator;

        const auto M  = static_cast<size_t>(_gridSize.y);
        const auto N = static_cast<size_t>(_gridSize.x);
        auto size = size_t(_gridSize.x * _gridSize.y);

        std::vector<float> uBuffer;
        std::vector<float> vBuffer;

        uBuffer.reserve(size);
        vBuffer.reserve(size);

        for(auto i = 0; i < M; ++i) {
            for(auto j = 0; j < N; ++j) {
                auto x = 2 * float(j)/_gridSize.x - 1;
                auto y = 2 * float(i)/_gridSize.y - 1;
                auto u = func(x, y);
                uBuffer.push_back(u.x);
                vBuffer.push_back(u.y);
            }
        }


        auto byteSize = size * sizeof(float);
        auto stagingBuffer_u = _device->createStagingBuffer(byteSize);
        auto stagingBuffer_v = _device->createStagingBuffer(byteSize);

        stagingBuffer_u.copy(uBuffer);
        stagingBuffer_v.copy(vBuffer);


        VkImageMemoryBarrier2 barrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel =  0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                }
        };
        std::vector<VkImageMemoryBarrier2> barriers;
        barrier.image = solver._vectorField.u[0].image;
        barrier.oldLayout = solver._vectorField.u[0].image.currentLayout;
        barriers.push_back(barrier);

        barrier.image = solver._vectorField.v[0].image;
        barrier.oldLayout = solver._vectorField.v[0].image.currentLayout;
        barriers.push_back(barrier);

        VkDependencyInfo dInfo {
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .imageMemoryBarrierCount = COUNT(barriers),
                .pImageMemoryBarriers = barriers.data()
        };


        _device->firstActiveCommandPool().oneTimeCommand([&](auto commandBuffer){
            vkCmdPipelineBarrier2(commandBuffer, &dInfo);

            const auto gs = glm::uvec2(_gridSize);
            VkBufferImageCopy region{};
            region.bufferOffset = 0;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;

            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;
            region.imageOffset = {0, 0, 0};
            region.imageExtent = {gs.x, gs.y, 1};

            vkCmdCopyBufferToImage(commandBuffer, stagingBuffer_u, solver._vectorField.u[0].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            vkCmdCopyBufferToImage(commandBuffer, stagingBuffer_v, solver._vectorField.v[0].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            barriers[0].srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
            barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barriers[0].dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;

            barriers[1].srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
            barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barriers[1].dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            barriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barriers[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;

            vkCmdPipelineBarrier2(commandBuffer, &dInfo);

            solver._vectorField.u[0].image.currentLayout = VK_IMAGE_LAYOUT_GENERAL;
            solver._vectorField.v[0].image.currentLayout = VK_IMAGE_LAYOUT_GENERAL;

        });
    }

    void FluidSolver::Builder::addQuantities(FluidSolver &solver) {
        for(auto& quantity : _quantities) {
            auto writes = initializers::writeDescriptorSets<12>();
            auto offset = solver.createDescriptorSet(writes, 0, quantity.get().field);
            solver.createDescriptorSet(writes, offset, quantity.get().source);

            _device->updateDescriptorSets(writes);

            solver._quantities.emplace_back(quantity);
        }
    }
}
