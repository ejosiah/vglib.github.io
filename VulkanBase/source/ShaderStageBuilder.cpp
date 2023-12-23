#include "ShaderStageBuilder.hpp"
#include "VulkanInitializers.h"

ShaderStageBuilder::ShaderStageBuilder(VulkanDevice *device, GraphicsPipelineBuilder *parent)
: GraphicsPipelineBuilder(device, parent)
{

}

ShaderStageBuilder &ShaderStageBuilder::vertexShader(const std::string &path) {
    auto shaderModule = device().createShaderModule(path);
    _stages[VK_SHADER_STAGE_VERTEX_BIT] = { shaderModule, VK_SHADER_STAGE_VERTEX_BIT };
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::vertexShader(const byte_string &data) {
    auto shaderModule = device().createShaderModule(data);
    _stages[VK_SHADER_STAGE_VERTEX_BIT] = { shaderModule, VK_SHADER_STAGE_VERTEX_BIT };
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::vertexShader(const std::vector<uint32_t> &data) {
    auto shaderModule = device().createShaderModule(data);
    _stages[VK_SHADER_STAGE_VERTEX_BIT] = { shaderModule, VK_SHADER_STAGE_VERTEX_BIT };
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::fragmentShader(const std::string &path) {
    auto shaderModule = device().createShaderModule(path);
    _stages[VK_SHADER_STAGE_FRAGMENT_BIT] = { shaderModule, VK_SHADER_STAGE_FRAGMENT_BIT };
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::fragmentShader(const byte_string &data) {
    auto shaderModule = device().createShaderModule(data);
    _stages[VK_SHADER_STAGE_FRAGMENT_BIT] = { shaderModule, VK_SHADER_STAGE_FRAGMENT_BIT };
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::fragmentShader(const std::vector<uint32_t>& data) {
    auto shaderModule = device().createShaderModule(data);
    _stages[VK_SHADER_STAGE_FRAGMENT_BIT] = { shaderModule, VK_SHADER_STAGE_FRAGMENT_BIT };
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::geometryShader(const std::string &path) {
    auto shaderModule = device().createShaderModule(path);
    _stages[VK_SHADER_STAGE_GEOMETRY_BIT] = { shaderModule, VK_SHADER_STAGE_GEOMETRY_BIT};
    return *this;
}


ShaderStageBuilder &ShaderStageBuilder::geometryShader(const byte_string& data) {
    auto shaderModule = device().createShaderModule(data);
    _stages[VK_SHADER_STAGE_GEOMETRY_BIT] = { shaderModule, VK_SHADER_STAGE_GEOMETRY_BIT};
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::geometryShader(const std::vector<uint32_t>& data) {
    auto shaderModule = device().createShaderModule(data);
    _stages[VK_SHADER_STAGE_GEOMETRY_BIT] = { shaderModule, VK_SHADER_STAGE_GEOMETRY_BIT};
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::tessellationEvaluationShader(const std::string &path) {
    auto shaderModule = device().createShaderModule(path);
    _stages[VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT] = { shaderModule, VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT};
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::tessellationEvaluationShader(const byte_string &data) {
    auto shaderModule = device().createShaderModule(data);
    _stages[VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT] = { shaderModule, VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT};
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::tessellationEvaluationShader(const std::vector<uint32_t> &data) {
    auto shaderModule = device().createShaderModule(data);
    _stages[VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT] = { shaderModule, VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT};
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::tessellationControlShader(const std::string &path) {
    auto shaderModule = device().createShaderModule(path);
    _stages[VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT] = { shaderModule, VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT};
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::tessellationControlShader(const byte_string &data) {
    auto shaderModule = device().createShaderModule(data);
    _stages[VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT] = { shaderModule, VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT};
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::tessellationControlShader(const std::vector<uint32_t> &data) {
    auto shaderModule = device().createShaderModule(data);
    _stages[VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT] = { shaderModule, VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT};
    return *this;
}

void ShaderStageBuilder::validate() const {

    if(!_stages.contains(VK_SHADER_STAGE_VERTEX_BIT)){
        throw std::runtime_error{"at least vertex shader should be provided"};
    }

    if(_stages.contains(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT) && !_stages.contains(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT)){
        throw std::runtime_error{"tessellation eval shader required if tessellation control shader provided"};
    }
}

std::vector<VkPipelineShaderStageCreateInfo>& ShaderStageBuilder::buildShaderStage()  {
    validate();

    std::vector<ShaderInfo> stages;
    for(auto [_, stage] : _stages) {
        stages.push_back(stage);
    }
    _vkStages = initializers::vertexShaderStages(stages);

    return _vkStages;
}

ShaderStageBuilder& ShaderStageBuilder::clear() {
    _stages.clear();
    return *this;
}

void ShaderStageBuilder::copy(const ShaderStageBuilder &source) {
    _stages = source._stages;
}