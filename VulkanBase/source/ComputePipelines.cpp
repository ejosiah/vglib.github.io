#include "ComputePipelins.hpp"

ComputePipelines::ComputePipelines(VulkanDevice *device): device(device) {

}

void ComputePipelines::createPipelines() {
    for(auto& metaData : pipelineMetaData()){
        auto shaderModule = std::move(get(metaData.shadePath, device));
        auto stage = initializers::shaderStage({ shaderModule, VK_SHADER_STAGE_COMPUTE_BIT});
        auto& sc = metaData.specializationConstants;
        VkSpecializationInfo specialization{COUNT(sc.entries), sc.entries.data(), sc.dataSize, sc.data };
        stage.pSpecializationInfo = &specialization;
        Pipeline pipeline;
        std::vector<VulkanDescriptorSetLayout> setLayouts;
        for(auto layout : metaData.layouts){
            setLayouts.push_back(*layout);
        }
        pipeline.layout = device->createPipelineLayout(setLayouts, metaData.ranges);

        auto createInfo = initializers::computePipelineCreateInfo();
        createInfo.stage = stage;
        createInfo.layout = pipeline.layout.handle;

        pipeline.pipeline = device->createComputePipeline(createInfo);
        device->setName<VK_OBJECT_TYPE_PIPELINE>(metaData.name, pipeline.pipeline.handle);
        pipelines.insert(std::make_pair(metaData.name, std::move(pipeline)));
    }
}

std::vector<PipelineMetaData> ComputePipelines::pipelineMetaData() {
    return {};
}


VkPipeline ComputePipelines::pipeline(const std::string& name) const {
    assert(pipelines.find(name) != end(pipelines));
    return pipelines[name].pipeline.handle;
}

VkPipelineLayout ComputePipelines::layout(const std::string& name) const {
    assert(pipelines.find(name) != end(pipelines));
    return pipelines[name].layout.handle;
}

VulkanShaderModule ComputePipelines::get(std::variant<std::string, std::vector<uint32_t>>& shaderPath, VulkanDevice* device) {
    return std::visit(overloaded{
       [&](std::string path){ return device->createShaderModule( path ); },
       [&](std::vector<uint32_t> data){ return device->createShaderModule( data ); }
    }, shaderPath);
}