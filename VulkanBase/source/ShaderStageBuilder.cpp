#include "ShaderStageBuilder.hpp"
#include "VulkanInitializers.h"

ShaderStageBuilder::ShaderStageBuilder(VulkanDevice *device, GraphicsPipelineBuilder *parent)
: GraphicsPipelineBuilder(device, parent)
{

}

ShaderStageBuilder &ShaderStageBuilder::vertexShader(const std::string &path) {
    _vertexModule = device().createShaderModule(path);
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::vertexShader(const byte_string &data) {
    _vertexModule = device().createShaderModule(data);
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::vertexShader(const std::vector<uint32_t> &data) {
    _vertexModule = device().createShaderModule(data);
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::fragmentShader(const std::string &path) {
    _fragmentModule = device().createShaderModule(path);
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::fragmentShader(const byte_string &data) {
    _fragmentModule = device().createShaderModule(data);
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::fragmentShader(const std::vector<uint32_t>& data) {
    _fragmentModule = device().createShaderModule(data);
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::geometryShader(const std::string &path) {
    _geometryModule = device().createShaderModule(path);
    return *this;
}


ShaderStageBuilder &ShaderStageBuilder::geometryShader(const byte_string& data) {
    _geometryModule = device().createShaderModule(data);
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::geometryShader(const std::vector<uint32_t>& data) {
    _geometryModule = device().createShaderModule(data);
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::tessellationEvaluationShader(const std::string &path) {
    _tessEvalModule = device().createShaderModule(path);
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::tessellationEvaluationShader(const byte_string &data) {
    _tessEvalModule = device().createShaderModule(data);
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::tessellationEvaluationShader(const std::vector<uint32_t> &data) {
    _tessEvalModule = device().createShaderModule(data);
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::tessellationControlShader(const std::string &path) {
    _tessControlModule = device().createShaderModule(path);
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::tessellationControlShader(const byte_string &data) {
    _tessControlModule = device().createShaderModule(data);
    return *this;
}

ShaderStageBuilder &ShaderStageBuilder::tessellationControlShader(const std::vector<uint32_t> &data) {
    _tessControlModule = device().createShaderModule(data);
    return *this;
}

void ShaderStageBuilder::validate() const {
    if(!_vertexModule){
        throw std::runtime_error{"at least vertex shader should be provided"};
    }
    if(_tessControlModule && !_tessEvalModule){
        throw std::runtime_error{"tessellation eval shader required if tessellation control shader provided"};
    }
}

std::vector<VkPipelineShaderStageCreateInfo>& ShaderStageBuilder::buildShaderStage()  {
    validate();
    _stages.clear();
    _stages.push_back({_vertexModule, VK_SHADER_STAGE_VERTEX_BIT});

    if(_fragmentModule){
        _stages.push_back({ _fragmentModule, VK_SHADER_STAGE_FRAGMENT_BIT});
    }
    if(_geometryModule){
        _stages.push_back({ _geometryModule, VK_SHADER_STAGE_GEOMETRY_BIT });
    }

    if(_tessEvalModule){
        _stages.push_back({ _tessEvalModule, VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT});
    }

    if(_tessControlModule){
        _stages.push_back({ _tessControlModule, VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT});
    }

    _vkStages = initializers::vertexShaderStages(_stages);

    return _vkStages;
}

ShaderStageBuilder& ShaderStageBuilder::clear() {
    dispose(_vertexModule);
    dispose(_fragmentModule);
    dispose(_geometryModule);
    dispose(_tessControlModule);
    dispose(_tessEvalModule);
    _stages.clear();
    return *this;
}

void ShaderStageBuilder::copy(const ShaderStageBuilder &source) {
    _vertexModule = source._vertexModule;
    _fragmentModule = source._fragmentModule;
    _geometryModule = source._geometryModule;
    _tessEvalModule = source._tessEvalModule;
    _tessControlModule = source._tessControlModule;
}