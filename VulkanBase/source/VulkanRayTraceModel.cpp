#include "VulkanRayTraceModel.hpp"

rt::AccelerationStructureBuilder::AccelerationStructureBuilder(const VulkanDevice *device)
    :m_device{device}
{

}

rt::AccelerationStructureBuilder::AccelerationStructureBuilder(AccelerationStructureBuilder &&source) noexcept {
    operator=(static_cast<AccelerationStructureBuilder&&>(source));
}

rt::AccelerationStructureBuilder& rt::AccelerationStructureBuilder::operator=(AccelerationStructureBuilder &&source) noexcept {
    if(this == &source) return *this;
    this->m_blas = std::move(source.m_blas);
    this->m_tlas = std::move(source.m_tlas);
    this->m_instanceBuffer = std::move(source.m_instanceBuffer);
    this->m_device = source.m_device;

    source.m_tlas.as.handle = VK_NULL_HANDLE;
    for(auto& input : source.m_blas){
        input.as.handle = VK_NULL_HANDLE;
    }
    source.m_device = nullptr;

    return *this;
}

rt::AccelerationStructureBuilder::~AccelerationStructureBuilder() {
    dispose();
}

void rt::AccelerationStructureBuilder::dispose() {
    if(m_device && m_tlas.as.handle) {
        vkDestroyAccelerationStructureKHR(*m_device, m_tlas.as.handle, nullptr);
        for (auto &input : m_blas) {
            vkDestroyAccelerationStructureKHR(*m_device, input.as.handle, nullptr);
        }
    }
}

std::vector<rt::InstanceGroup> rt::AccelerationStructureBuilder::add(const std::vector<MeshObjectInstance> &drawableInstances,
                                               VkBuildAccelerationStructureFlagsKHR flags) {
    std::vector<VulkanDrawable*> drawables;
    for(auto& dInstance : drawableInstances){
        auto itr = std::find_if(begin(drawables), end(drawables), [&](auto drawable){ return dInstance.object.drawable == drawable;});
        if(itr != end(drawables)) continue;
        drawables.push_back(dInstance.object.drawable);
    }
    auto [offsets, blasIds] = buildBlas(drawables, flags);

    std::vector<InstanceGroup> instanceGroups;


    auto findObjId = [&](VulkanDrawable* drawable) -> std::optional<uint32_t> {
        for(int i = 0; i < drawables.size(); i++){
            if(drawable == drawables[i]) return i;
        }
        return {};
    };


    for(const auto & dInstance : drawableInstances){
        auto objId = findObjId(dInstance.object.drawable);
        assert(objId.has_value());
        InstanceGroup instanceGroup{ dInstance, *objId };


        auto& meshes = dInstance.object.drawable->meshes;
        for(int j = 0; j < meshes.size(); j++){
            Instance instance;
            instance.blasId = blasIds[offsets[*objId] + j];
            instance.instanceCustomId = j;
            if(dInstance.hitGroupId != ~0u){
                instance.hitGroupId = dInstance.hitGroupId;
            }else{
                instance.hitGroupId = dInstance.object.metaData[j].hitGroupId;
            }
            instance.mask = dInstance.object.metaData[j].mask;
            instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            instance.xform = dInstance.xform;
            instanceGroup.instanceIds[meshes[j].name] = m_instances.size();
            m_instances.push_back(instance);
            instanceGroup.add(&m_instances.back());

        }
        instanceGroups.push_back(std::move(instanceGroup));
    }

    return instanceGroups;
}

rt::ImplicitObject rt::AccelerationStructureBuilder::add(const std::vector<imp::Sphere>& spheres, uint32_t customInstanceId, uint32_t hitGroup,
                                                         uint32_t cullMask, VkBuildAccelerationStructureFlagsKHR flags) {

    uint32_t numSpheres = spheres.size();
    std::vector<imp::Box> aabbs;
    aabbs.reserve(numSpheres);
    for(auto& sphere : spheres){
        imp::Box aabb{};
        aabb.max = sphere.center + glm::vec3(sphere.radius);
        aabb.min = sphere.center - glm::vec3(sphere.radius);
        aabbs.push_back(aabb);
    }

    return add(aabbs, customInstanceId, hitGroup, cullMask, flags);
}

rt::ImplicitObject rt::AccelerationStructureBuilder::add(const std::vector<imp::Cylinder>& cylinders, uint32_t customInstanceId, uint32_t hitGroup,
                                                         uint32_t cullMask, VkBuildAccelerationStructureFlagsKHR flags) {

    std::vector<imp::Box> aabbs;
    aabbs.reserve(cylinders.size());
    for(auto& cylinder : cylinders){
        imp::Box aabb{};
        auto center = (cylinder.top + cylinder.bottom) * 0.5f;
        auto height = glm::distance(cylinder.top, cylinder.bottom);
        aabb.max = center + glm::vec3(cylinder.radius, 0.5 * height, cylinder.radius);
        aabb.min = center - glm::vec3(cylinder.radius, 0.5 * height, cylinder.radius);
        aabbs.push_back(aabb);
    }

    return add(aabbs, customInstanceId, hitGroup, cullMask, flags);
}

rt::ImplicitObject rt::AccelerationStructureBuilder::add(const std::vector<imp::Plane>& planes, uint32_t customInstanceId, float length, uint32_t hitGroup,
                                                         uint32_t cullMask, VkBuildAccelerationStructureFlagsKHR flags) {

    auto project = [](glm::vec3 q, const imp::Plane& p){
        float t = glm::dot(p.normal, q) - p.d;
        return q - t* p.normal;
    };

    std::vector<imp::Box> aabbs;
    auto numPlanes = planes.size();
    aabbs.reserve(numPlanes);
    for(auto& plane : planes){
        imp::Box aabb{};
        aabb.max = project(glm::vec3{ length }, plane);
        aabb.min = project(glm::vec3{ -length }, plane);
        aabbs.push_back(aabb);
    }

    return add(aabbs, customInstanceId, hitGroup, cullMask, flags);
}

rt::ImplicitObject rt::AccelerationStructureBuilder::add(const std::vector<imp::Box> &aabbs, uint32_t customInstanceId, uint32_t hitGroup,
                                                         uint32_t  cullMask, VkBuildAccelerationStructureFlagsKHR flags) {

    rt::ImplicitObject object;
    object.numObjects = aabbs.size();
    object.hitGroupId = hitGroup;
    object.aabbBuffer = m_device->createCpuVisibleBuffer(aabbs.data(), aabbs.size() * sizeof(imp::Box)
            , VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
              | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT );

    auto blasId = buildBlas({ &object }, flags).front();
    rt::Instance instance{};
    instance.blasId = blasId;
    instance.hitGroupId = hitGroup;
    instance.mask = cullMask;
    instance.instanceCustomId = customInstanceId;
    m_instances.push_back(instance);

    return object;
}

std::tuple<std::vector<uint32_t>, std::vector<rt::BlasId>> rt::AccelerationStructureBuilder::buildBlas(const std::vector<VulkanDrawable*>& drawables, VkBuildAccelerationStructureFlagsKHR flags){
    std::vector<BlasInput> inputs;
    inputs.reserve(drawables.size());
    std::vector<uint32_t> offsets;
    offsets.reserve(drawables.size());
    uint32_t offset = 0;
    for(auto drawable : drawables){
        for(int meshId = 0; meshId < drawable->meshes.size(); meshId++){
            inputs.emplace_back(*m_device, *drawable, meshId);
        }
        offsets.push_back(offset);
        offset += drawable->meshes.size();
    }
    auto blasIds = buildBlas(inputs, flags);
    return std::make_tuple(offsets, blasIds);
}

std::vector<rt::BlasId> rt::AccelerationStructureBuilder::buildBlas(const std::vector<ImplicitObject*>& implicits,
                                                 VkBuildAccelerationStructureFlagsKHR flags) {
    std::vector<BlasInput> inputs;
    inputs.reserve(implicits.size());
    for(auto implicit : implicits){
        inputs.emplace_back(*m_device, *implicit);
    }

    return buildBlas(inputs, flags);
}

std::vector<rt::BlasId> rt::AccelerationStructureBuilder::buildBlas(const std::vector<BlasInput> &inputs,
                                                 VkBuildAccelerationStructureFlagsKHR flags) {

    std::vector<uint32_t> offsets;
    std::vector<rt::BlasId> blasIds;
    blasIds.reserve(inputs.size());

    std::vector<rt::ScratchBuffer> scratchBuffers;
    scratchBuffers.reserve(inputs.size());

    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> asBuildGeomInfos;
    asBuildGeomInfos.reserve(inputs.size());

    std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> buildRangeInfos;

    uint32_t blasIdOffset = m_blas.size();
    for(auto& input : inputs){
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
        buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildInfo.flags = flags;
        buildInfo.geometryCount = COUNT(input.asGeomentry);
        buildInfo.pGeometries = input.asGeomentry.data();

        std::vector<uint32_t> numTriangles = input.maxPrimitiveCounts();

        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        vkGetAccelerationStructureBuildSizesKHR(*m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, numTriangles.data(), &sizeInfo);

        BlasEntry entry{input};
        entry.flags = flags;
        entry.as.buffer = m_device->createBuffer(VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                , VMA_MEMORY_USAGE_GPU_ONLY, sizeInfo.accelerationStructureSize);

        VkAccelerationStructureCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        createInfo.buffer = entry.as.buffer;
        createInfo.size = sizeInfo.accelerationStructureSize;
        createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        ERR_GUARD_VULKAN(vkCreateAccelerationStructureKHR(*m_device, &createInfo, nullptr, &entry.as.handle));

        ensureAlignmentScratchBufferSize(sizeInfo);
        auto scratchBuffer = createScratchBuffer(sizeInfo.buildScratchSize);
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        buildInfo.dstAccelerationStructure = entry.as.handle;
        buildInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

        scratchBuffers.push_back(std::move(scratchBuffer));
        blasIds.push_back(m_blas.size());
        m_blas.push_back(std::move(entry));
        buildRangeInfos.push_back(input.asBuildOffsetInfo.data());
        asBuildGeomInfos.push_back(buildInfo);
    }

    m_device->commandPoolFor(*m_device->queueFamilyIndex.graphics).oneTimeCommand([&](auto commandBuffer){
       vkCmdBuildAccelerationStructuresKHR(commandBuffer, COUNT(asBuildGeomInfos), asBuildGeomInfos.data(), buildRangeInfos.data());
    });

    for(auto i = blasIdOffset; i < m_blas.size(); i++){
        auto& entry = m_blas[i];
        VkAccelerationStructureDeviceAddressInfoKHR asDeviceAddressInfo{};
        asDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        asDeviceAddressInfo.accelerationStructure = entry.as.handle;
        entry.as.deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(*m_device, &asDeviceAddressInfo);
    }
    return blasIds;
}

std::vector<rt::Instance> rt::AccelerationStructureBuilder::updateTlas(const std::vector<Instance> &instances) {
    assert(m_tlas.as.handle && !instances.empty());
    return buildTlas(m_tlas.flags | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR, instances);
}

std::vector<rt::Instance> rt::AccelerationStructureBuilder::buildTlas(VkBuildAccelerationStructureFlagsKHR flags, const std::vector<Instance>& instances) {
    bool update = !instances.empty();
    // Cannot call buildTlas twice except to update.
    assert(m_tlas.as.handle == VK_NULL_HANDLE || update);
    m_tlas.flags = flags;
    if(update){
        m_instances = instances;
    }
    uint32_t numInstances = m_instances.size();
    std::vector<VkAccelerationStructureInstanceKHR> asInstances(numInstances);
    std::transform(begin(m_instances), end(m_instances), begin(asInstances), [&](auto& instance){
        return toVkAccStruct(instance);
    });


    m_instanceBuffer = m_device->createDeviceLocalBuffer(asInstances.data(), sizeof(VkAccelerationStructureInstanceKHR) * numInstances
                                                         , VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

    VkDeviceOrHostAddressConstKHR instanceDataDeviceAddress{};
    instanceDataDeviceAddress.deviceAddress = m_device->getAddress(m_instanceBuffer);

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType =  VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.arrayOfPointers = VK_FALSE;
    geometry.geometry.instances.data = instanceDataDeviceAddress;

    VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo{};
    accelerationStructureBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    accelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    accelerationStructureBuildGeometryInfo.flags = flags;
    accelerationStructureBuildGeometryInfo.geometryCount = 1;
    accelerationStructureBuildGeometryInfo.pGeometries = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo{};
    accelerationStructureBuildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(
            *m_device,
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &accelerationStructureBuildGeometryInfo,
            &numInstances,
            &accelerationStructureBuildSizesInfo);

    if(!update) {
        m_tlas.as.buffer = m_device->createBuffer(
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                VMA_MEMORY_USAGE_GPU_ONLY,
                accelerationStructureBuildSizesInfo.accelerationStructureSize
        );
        VkAccelerationStructureCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        createInfo.buffer = m_tlas.as.buffer;
        createInfo.size = accelerationStructureBuildSizesInfo.accelerationStructureSize;
        createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        vkCreateAccelerationStructureKHR(*m_device, &createInfo, nullptr, &m_tlas.as.handle);
    }

    ensureAlignmentScratchBufferSize(accelerationStructureBuildSizesInfo);
    auto scratchBuffer = createScratchBuffer(accelerationStructureBuildSizesInfo.buildScratchSize);

    accelerationStructureBuildGeometryInfo.mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    accelerationStructureBuildGeometryInfo.srcAccelerationStructure = update ? m_tlas.as.handle : VK_NULL_HANDLE;
    accelerationStructureBuildGeometryInfo.dstAccelerationStructure = m_tlas.as.handle;
    accelerationStructureBuildGeometryInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR  accelerationStructureBuildRangeInfo{};
    accelerationStructureBuildRangeInfo.primitiveCount = numInstances;
    accelerationStructureBuildRangeInfo.primitiveOffset = 0;
    accelerationStructureBuildRangeInfo.firstVertex = 0;
    accelerationStructureBuildRangeInfo.transformOffset = 0;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR*> accelerationBuildStructureRangeInfos = { &accelerationStructureBuildRangeInfo };

    m_device->commandPoolFor(*m_device->queueFamilyIndex.graphics).oneTimeCommand( [&](auto commandBuffer){
        vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &accelerationStructureBuildGeometryInfo, accelerationBuildStructureRangeInfos.data());
    });

    VkAccelerationStructureDeviceAddressInfoKHR accelerationStructureDeviceAddressInfo{};
    accelerationStructureDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    accelerationStructureDeviceAddressInfo.accelerationStructure = m_tlas.as.handle;
    m_tlas.as.deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(*m_device, &accelerationStructureDeviceAddressInfo);

    return std::move(m_instances);
}

VkAccelerationStructureInstanceKHR rt::AccelerationStructureBuilder::toVkAccStruct(const Instance& instance){
    assert(!m_blas.empty() && instance.blasId < m_blas.size());
    VkAccelerationStructureInstanceKHR asInstance{};
    asInstance.instanceCustomIndex = instance.instanceCustomId;
    asInstance.mask = instance.mask;
    asInstance.instanceShaderBindingTableRecordOffset = instance.hitGroupId;
    asInstance.flags = instance.flags;
    asInstance.accelerationStructureReference = m_blas[instance.blasId].as.deviceAddress;

    auto xform = glm::transpose(instance.xform);
    std::memcpy(&asInstance.transform, glm::value_ptr(xform), sizeof(VkTransformMatrixKHR));

    return asInstance;
}

rt::ScratchBuffer rt::AccelerationStructureBuilder::createScratchBuffer(VkDeviceSize size) const {
    spdlog::info("scratch size after: {}", size);
    rt::ScratchBuffer scratchBuffer;
    scratchBuffer.buffer = m_device->createBuffer(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY,
            size, "acceleration_struct_scratch_buffer");

    VkBufferDeviceAddressInfo bufferDeviceAddressInfo{};
    bufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    bufferDeviceAddressInfo.buffer = scratchBuffer.buffer;
    scratchBuffer.deviceAddress = vkGetBufferDeviceAddress(*m_device, &bufferDeviceAddressInfo);

    return scratchBuffer;
}

[[nodiscard]]
VkPhysicalDeviceAccelerationStructurePropertiesKHR rt::AccelerationStructureBuilder::getAccelerationStructureProperties() const {
    VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 props{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &asProps };
    vkGetPhysicalDeviceProperties2(*m_device, &props);
    return asProps;
}

void rt::AccelerationStructureBuilder::addOrUpdateInstances(rt::InstanceBuilder &&builder) {
    builder(m_instances);
}

void rt::AccelerationStructureBuilder::add(rt::Instance instance) {
    m_instances.push_back(instance);
}

void rt::AccelerationStructureBuilder::ensureAlignmentScratchBufferSize(
        VkAccelerationStructureBuildSizesInfoKHR &info) const {
    auto props = getAccelerationStructureProperties();
    info.buildScratchSize = alignedSize(info.buildScratchSize, props.minAccelerationStructureScratchOffsetAlignment);
}

const rt::AccelerationStructure &rt::AccelerationStructureBuilder::topLevelAs() const {
    return m_tlas.as;
}