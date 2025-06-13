#pragma once

#include "Video.hpp"
#include "VulkanDevice.h"
#include "ComputePipelins.hpp"

/*
 * TODO implement decode capability for VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_DISTINCT_BIT_KHR
 * TODO make YUV sampler private and resolve picture to RGB
 * FIXME video stutter
 */

class VideoDecoder : public ComputePipelines {
public:
    VideoDecoder() = default;

    explicit VideoDecoder(VulkanDevice& device);

    ~VideoDecoder();

    void init();

    void decode(std::shared_ptr<VideoInstance>& instance);

private:
    VulkanDevice& device();

    void createDescriptorPool();

    void createDescriptorSetLayout();

    void updateDescriptors(OutputTexture& output);

    void updateSrcDescriptor(OutputTexture& output);

    void createDpbOutputTexture(OutputTexture& output, const std::string& name);

    void translate(const h264::SPS& sps, StdVideoH264SequenceParameterSet& vk_sps, StdVideoH264SequenceParameterSetVui& vk_vui, StdVideoH264HrdParameters& vk_hrd);

    void translate(const h264::PPS& pps, StdVideoH264PictureParameterSet& vk_pps, StdVideoH264ScalingLists& vk_scalinglist);

    void decode(const std::shared_ptr<VideoInstance>& instance, VkCommandBuffer commandBuffer);

    void decode(const VideoDecodeOperation& decodeOperation, VkCommandBuffer commandBuffer);

    void resolveToRGB(const std::shared_ptr<VideoInstance>& instance, VkCommandBuffer commandBuffer);

    void getVideoCapabilities();

    void createSemaphores();

    void createYUVSampler();

    void initialize(std::shared_ptr<VideoInstance>& instance);

    void createVideoSession(std::shared_ptr<VideoInstance>& instance);

    void createDpbResources(std::shared_ptr<VideoInstance>& instance);

protected:
    std::vector<PipelineMetaData> pipelineMetaData() override;

private:
    uint64_t VIDEO_DECODE_BITSTREAM_ALIGNMENT = 1u;
    VulkanDevice* _device{};
    VideoCapabilities cb;
    VulkanSampler yuvSampler;
    VkSamplerYcbcrConversion ycbcrConversion{};
    struct {
        VulkanSemaphore renderingFinished;
        VulkanSemaphore frameDecoded;
    } semaphores;
    std::vector<VideoSession*> activeSessions;
    VulkanDescriptorPool descriptorPool;
    VulkanDescriptorSetLayout rgbResolveDescriptorSetLayout;
    VkDescriptorSet rgbResolveDescriptorSet{};
    static constexpr uint32_t MaxDescriptorResources = 128;
};