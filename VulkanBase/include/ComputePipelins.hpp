#pragma once

#include "common.h"
#include "VulkanDevice.h"
#include "VulkanShaderModule.h"
#include "VulkanExtensions.h"
#include <variant>
#include <vector>

struct SpecializationConstants{
    std::vector<VkSpecializationMapEntry> entries{};
    void* data = nullptr;
    size_t dataSize = 0;
};

struct PipelineMetaData{
    std::string name;
    std::variant<std::string, std::vector<uint32_t>> shadePath{std::string{}};
    std::vector<VulkanDescriptorSetLayout*> layouts;
    std::vector<VkPushConstantRange> ranges;
    SpecializationConstants specializationConstants{};
};

struct Pipeline{
    VulkanPipeline pipeline;
    VulkanPipelineLayout layout;
};

class ComputePipelines {
public:
    explicit ComputePipelines(VulkanDevice* device = nullptr);

    VkPipeline pipeline(const std::string& name) const;

    VkPipelineLayout layout(const std::string& name) const;

protected:
    void createPipelines();

    virtual std::vector<PipelineMetaData> pipelineMetaData();

    static VulkanShaderModule get(std::variant<std::string, std::vector<uint32_t>>& shaderPath, VulkanDevice* device);

protected:
    VulkanDevice* device{};
    mutable std::map<std::string, Pipeline> pipelines{};
};