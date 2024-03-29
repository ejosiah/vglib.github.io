#pragma once

#include "common.h"
#include "GraphicsPipelineBuilder.hpp"
#include "VulkanShaderModule.h"

#include <map>

class ShaderStageBuilder : public GraphicsPipelineBuilder{
public:
   explicit ShaderStageBuilder(VulkanDevice* device, GraphicsPipelineBuilder* parent);

   [[nodiscard]]
   ShaderStageBuilder& vertexShader(const std::string& path);

   [[nodiscard]]
   ShaderStageBuilder& vertexShader(const byte_string& data);

   [[nodiscard]]
   ShaderStageBuilder& vertexShader(const std::vector<uint32_t>& data);

   [[nodiscard]]
   ShaderStageBuilder& fragmentShader(const std::string& path);

   [[nodiscard]]
   ShaderStageBuilder& fragmentShader(const byte_string& data);

   [[nodiscard]]
   ShaderStageBuilder& fragmentShader(const std::vector<uint32_t>& data);

   [[nodiscard]]
   ShaderStageBuilder& geometryShader(const std::string& path);

   [[nodiscard]]
   ShaderStageBuilder& geometryShader(const byte_string& data);

   [[nodiscard]]
   ShaderStageBuilder& geometryShader(const std::vector<uint32_t>& data);

   [[nodiscard]]
   ShaderStageBuilder& tessellationEvaluationShader(const std::string& path);

   [[nodiscard]]
   ShaderStageBuilder& tessellationEvaluationShader(const byte_string& data);

   [[nodiscard]]
   ShaderStageBuilder& tessellationEvaluationShader(const std::vector<uint32_t>& data);

   [[nodiscard]]
   ShaderStageBuilder& tessellationControlShader(const std::string& path);

   [[nodiscard]]
   ShaderStageBuilder& tessellationControlShader(const byte_string& data);

   [[nodiscard]]
   ShaderStageBuilder& tessellationControlShader(const std::vector<uint32_t>& data);

   ShaderStageBuilder& clear();

   void validate() const;

   void copy(const ShaderStageBuilder& source);

   [[nodiscard]]
   std::vector<VkPipelineShaderStageCreateInfo>& buildShaderStage();

private:
    std::map<VkShaderStageFlagBits, ShaderInfo> _stages;
    std::vector<VkPipelineShaderStageCreateInfo> _vkStages;
};