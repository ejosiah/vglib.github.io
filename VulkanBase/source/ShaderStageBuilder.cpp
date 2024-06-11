#include "ShaderStageBuilder.hpp"
#include "VulkanInitializers.h"

#include <stdexcept>
#include <algorithm>

ShaderStageBuilder::ShaderStageBuilder(VulkanDevice *device, GraphicsPipelineBuilder *parent)
: GraphicsPipelineBuilder(device, parent)
{
    _features.pNext = &_meshFeatures;
    vkGetPhysicalDeviceFeatures2(device->physicalDevice, &_features);
}

ShaderStageBuilder::ShaderStageBuilder(ShaderStageBuilder *parent)
: GraphicsPipelineBuilder(parent->_device, parent)
{
    _meshFeatures = parent->_meshFeatures;
}

ShaderBuilder &ShaderStageBuilder::vertexShader(const ShaderSource &source) {
    return addShader(source, VK_SHADER_STAGE_VERTEX_BIT);
}

ShaderBuilder &ShaderStageBuilder::taskSShader(const ShaderSource &source) {
    if(!taskShaderSupported()) throw std::runtime_error{ "Task Shader not supported" };
    return addShader(source, VK_SHADER_STAGE_TASK_BIT_EXT);
}

ShaderBuilder &ShaderStageBuilder::meshShader(const ShaderSource &source) {
    if(!meshShaderSupported()) throw std::runtime_error{ "Mesh Shader not supported" };
    return addShader(source, VK_SHADER_STAGE_MESH_BIT_EXT);
}

ShaderBuilder &ShaderStageBuilder::fragmentShader(const ShaderSource &source) {
    return addShader(source, VK_SHADER_STAGE_FRAGMENT_BIT);
}


ShaderBuilder &ShaderStageBuilder::geometryShader(const ShaderSource &source) {
   return addShader(source, VK_SHADER_STAGE_GEOMETRY_BIT);
}

ShaderBuilder &ShaderStageBuilder::tessellationEvaluationShader(const ShaderSource &source) {
    return addShader(source, VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT);
}

ShaderBuilder &ShaderStageBuilder::tessellationControlShader(const ShaderSource& source) {
    return addShader(source, VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT);
}

void ShaderStageBuilder::validate() const {

    if(!hasVertexShader()){
        throw std::runtime_error{"at least vertex/mesh shader should be provided"};
    }

    if(hasTessControlShader() && !hasTessEvalShader()){
        throw std::runtime_error{"tessellation eval shader required if tessellation control shader provided"};
    }
}

std::vector<VkPipelineShaderStageCreateInfo>& ShaderStageBuilder::buildShaderStage()  {
    validate();

    for(auto& builder : _shaderBuilders) {
        auto& stage = builder->buildShader();
        _vkStages.push_back(stage);
    }

    return _vkStages;
}

ShaderStageBuilder& ShaderStageBuilder::clear() {
    _shaderBuilders.clear();
    return *this;
}

void ShaderStageBuilder::copy(const ShaderStageBuilder &source) {
    for(auto& sBuilder : source._shaderBuilders) {
        auto builder = std::make_unique<ShaderBuilder>(this);
        builder->copy(*sBuilder);
        _shaderBuilders.push_back(std::move(builder));
    }
}

ShaderBuilder &
ShaderStageBuilder::addShader(const ShaderStageBuilder::ShaderSource &source, VkShaderStageFlagBits stage) {
    auto itr = std::find_if(_shaderBuilders.begin(), _shaderBuilders.end(),[&](const auto& builder){
        return builder->isStage(stage);
    });
    if(itr != _shaderBuilders.end()) {
        _shaderBuilders.erase(itr);
    }
    _shaderBuilders.push_back(std::make_unique<ShaderBuilder>(source, stage, this));
    return *_shaderBuilders.back();
}

bool ShaderStageBuilder::hasVertexShader() const {
    auto vertItr = std::find_if(_shaderBuilders.begin(), _shaderBuilders.end(), [](const auto& builder){ return builder->isVertexShader(); });
    auto meshItr = std::find_if(_shaderBuilders.begin(), _shaderBuilders.end(), [](const auto& builder){ return builder->isMeshShader(); });
    return vertItr != _shaderBuilders.end() || meshItr != _shaderBuilders.end();
}

bool ShaderStageBuilder::hasTessControlShader() const {
    auto itr = std::find_if(_shaderBuilders.begin(), _shaderBuilders.end(), [](const auto& builder){ return builder->isTessControlShader(); });
    return itr != _shaderBuilders.end();
}

bool ShaderStageBuilder::hasTessEvalShader() const {
    auto itr = std::find_if(_shaderBuilders.begin(), _shaderBuilders.end(), [](const auto& builder){ return builder->isTessEvalShader(); });
    return itr != _shaderBuilders.end();
}

void ShaderStageBuilder::clearStages() {
    _vkStages.clear();
}

bool ShaderStageBuilder::meshShaderSupported() const {
    return static_cast<bool>(_meshFeatures.meshShader);
}

bool ShaderStageBuilder::taskShaderSupported() const {
    return static_cast<bool>(_meshFeatures.taskShader);
}


ShaderBuilder::ShaderBuilder(ShaderStageBuilder *parent)
: ShaderStageBuilder(parent) {}

ShaderBuilder::ShaderBuilder(const ShaderSource &source, VkShaderStageFlagBits stage, ShaderStageBuilder *parent)
: ShaderStageBuilder(parent)
{
    _stage.stage = stage;
    std::visit(overloaded{
        [&](const byte_string source) { _stage.module = device().createShaderModule(source); },
        [&](const std::vector<uint32_t> source) { _stage.module = device().createShaderModule(source); },
        [&](const std::string &source) { _stage.module = device().createShaderModule(source); },
    }, source);
}

VkPipelineShaderStageCreateInfo& ShaderBuilder::buildShader() {
    _createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    _createInfo.stage = _stage.stage;
    _createInfo.module = _stage.module.handle;
    _createInfo.pName = _stage.entry;

    _specialization.mapEntryCount = _entries.size();
    _specialization.pMapEntries = _entries.data();
    _specialization.dataSize = _data.size();
    _specialization.pData = _data.data();
    _createInfo.pSpecializationInfo = &_specialization;

    return _createInfo;
}

ShaderBuilder &ShaderBuilder::vertexShader(const ShaderStageBuilder::ShaderSource &source) {
    return parent()->vertexShader(source);
}

ShaderBuilder &ShaderBuilder::taskSShader(const ShaderStageBuilder::ShaderSource &source) {
    return parent()->taskSShader(source);
}

ShaderBuilder &ShaderBuilder::meshShader(const ShaderStageBuilder::ShaderSource &source) {
    return parent()->meshShader(source);
}

ShaderBuilder &ShaderBuilder::fragmentShader(const ShaderStageBuilder::ShaderSource &source) {
    return parent()->fragmentShader(source);
}

ShaderStageBuilder *ShaderBuilder::parent() {
    return reinterpret_cast<ShaderStageBuilder*>(_parent);
}

ShaderBuilder &ShaderBuilder::geometryShader(const ShaderStageBuilder::ShaderSource &source) {
    return parent()->geometryShader(source);
}

ShaderBuilder &ShaderBuilder::tessellationEvaluationShader(const ShaderStageBuilder::ShaderSource &source) {
    return parent()->tessellationEvaluationShader(source);
}

ShaderBuilder &ShaderBuilder::tessellationControlShader(const ShaderStageBuilder::ShaderSource &source) {
    return parent()->tessellationControlShader(source);
}

bool ShaderBuilder::isVertexShader() const {
    return _stage.stage == VK_SHADER_STAGE_VERTEX_BIT;
}

bool ShaderBuilder::isMeshShader() const {
    return _stage.stage == VK_SHADER_STAGE_MESH_BIT_EXT;
}

bool ShaderBuilder::isTessEvalShader() const {
    return _stage.stage == VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
}

bool ShaderBuilder::isTessControlShader() const {
    return _stage.stage == VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
}

void ShaderBuilder::copy(const ShaderBuilder &source) {
    _stage = source._stage;
    _entries = source._entries;
    _data.resize(source._data.size());
    std::memcpy(_data.data(), source._data.data(), source._data.size());
}

bool ShaderBuilder::isStage(VkShaderStageFlagBits stage) const {
    return _stage.stage == stage;
}
