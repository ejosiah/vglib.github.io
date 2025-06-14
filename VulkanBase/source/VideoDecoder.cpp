#include "video/VideoDecoder.hpp"
#include "Barrier.hpp"
#include "filemanager.hpp"
#include <algorithm>


VideoDecoder::VideoDecoder(VulkanDevice &device)
: ComputePipelines(&device)
, _device(&device){}

VideoDecoder::~VideoDecoder() {
    for(auto session : activeSessions) {
        vkDestroyVideoSessionKHR(device(), session->handle, nullptr);
        vkDestroyVideoSessionParametersKHR(device(), session->parameters, nullptr);
        vkDestroySamplerYcbcrConversion(device().logicalDevice, ycbcrConversion, nullptr);
        for (auto memory: session->allocations) {
            vkFreeMemory(device().logicalDevice, memory, nullptr);
        }
    }    
}

void VideoDecoder::init() {
    createSemaphores();
    getVideoCapabilities();
    createYUVSampler();
    createDescriptorPool();
    createDescriptorSetLayout();
    createPipelines();
}

void VideoDecoder::decode(std::shared_ptr<VideoInstance> &instance) {
    if (!(cb.decodeCapabilities.flags & VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR)) {
        spdlog::error("VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_DISTINCT_BIT_KHR not yet implemented");
        std::exit(500);
    }
    initialize(instance);

    static Synchronization sync{};

    sync.signalSemaphores.push_back(semaphores.frameDecoded);
    std::vector<std::string> acquired;
    device().videoDecodeCommandPool().oneTimeCommand([&](auto commandBuffer) {
        for (auto &out: instance->output_textures_free) {
            if (out.src.state.layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) continue;
            Barriers::acquire(out.src.texture->image, out.src.subresource_luminance,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR,
                              *device().queueFamilyIndex.graphics, *device().queueFamilyIndex.video_decode);

            Barriers::acquire(out.src.texture->image, out.src.subresource_chrominance,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR,
                              *device().queueFamilyIndex.graphics, *device().queueFamilyIndex.video_decode);

            out.src.state = { VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR, VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR, VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR };
            acquired.push_back(out.name);
        }
        for (auto &out: instance->output_textures_used) {
            if (out.src.state.layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) continue;
            Barriers::acquire(out.src.texture->image, out.src.subresource_luminance,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR,
                              *device().queueFamilyIndex.graphics, *device().queueFamilyIndex.video_decode);

            Barriers::acquire(out.src.texture->image, out.src.subresource_chrominance,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR,
                              *device().queueFamilyIndex.graphics, *device().queueFamilyIndex.video_decode);

            out.src.state = { VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR, VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR, VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR };
            acquired.push_back(out.name);
        }
        if(!acquired.empty()) {
            spdlog::warn("acquired {}, from graphics to video_decode", acquired);
            acquired.clear();
        }
        Barriers::flush(commandBuffer);
        decode(instance, commandBuffer);
    }, sync);


    sync.clear();
    sync.waitSemaphores.semaphores.push_back(semaphores.frameDecoded);
    sync.waitSemaphores.stages.push_back(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    sync.signalSemaphores.push_back(semaphores.renderingFinished);

    device().graphicsCommandPool().oneTimeCommand([&](auto commandBuffer) {
        resolveToRGB(instance, commandBuffer);
    }, sync);

    sync.clear();
    sync.waitSemaphores.semaphores.push_back(semaphores.renderingFinished);
    sync.waitSemaphores.stages.push_back(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
}


VulkanDevice &VideoDecoder::device() {
    return *_device;
}

void VideoDecoder::createDpbOutputTexture(OutputTexture &output, const std::string &name) {
    output.name = name;
    auto& texture = output.display.texture;

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;  // TODO confirm if converter converts so linear or srgb
    imageCreateInfo.extent = {texture.width, texture.height, 1};
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    texture.image = device().createImage(imageCreateInfo);
    device().setName<VK_OBJECT_TYPE_IMAGE>(name, texture.image.image);

    VkImageViewCreateInfo  createInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = texture.image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .subresourceRange = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
            },
    };
    texture.imageView = device().createImageView(createInfo);
    device().setName<VK_OBJECT_TYPE_IMAGE_VIEW>(fmt::format("{}_image_view", name), texture.imageView.handle);

    output.display.subresource = VkImageSubresourceRange{
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
    };

    updateDescriptors(output);
}

void VideoDecoder::createSemaphores() {
    semaphores.frameDecoded = device().createSemaphore();
    semaphores.renderingFinished = device().createSemaphore();

    device().setName<VK_OBJECT_TYPE_SEMAPHORE>("frameDecoded", semaphores.frameDecoded.semaphore);
    device().setName<VK_OBJECT_TYPE_SEMAPHORE>("vdc_renderingFinished", semaphores.renderingFinished.semaphore);
}

void VideoDecoder::createYUVSampler() {
    VkSamplerYcbcrConversionCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
            .format = cb.formats.front().format,
            .ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709,
            .ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_FULL,
            .components = {
                    .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .a = VK_COMPONENT_SWIZZLE_IDENTITY,
            },
            .xChromaOffset = VK_CHROMA_LOCATION_MIDPOINT,
            .yChromaOffset = VK_CHROMA_LOCATION_MIDPOINT,
            .chromaFilter = VK_FILTER_LINEAR,
            .forceExplicitReconstruction = VK_FALSE
    };

    ERR_GUARD_VULKAN(vkCreateSamplerYcbcrConversion(device().logicalDevice, &createInfo, nullptr, &ycbcrConversion));

    VkSamplerYcbcrConversionInfo  conversionInfo {
            .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO,
            .conversion = ycbcrConversion
    };

    VkSamplerCreateInfo samplerCreateInfo{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .pNext = &conversionInfo,
            .magFilter = VK_FILTER_NEAREST,
            .minFilter = VK_FILTER_NEAREST,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .minLod = 0,
            .maxLod = 1,
            .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
    };

    yuvSampler = device().createSampler(samplerCreateInfo);
}

void VideoDecoder::getVideoCapabilities() {
    Video::getCapabilities(device(), cb);
    VIDEO_DECODE_BITSTREAM_ALIGNMENT = cb.capabilities.minBitstreamBufferOffsetAlignment;
}

void VideoDecoder::createVideoSession(std::shared_ptr<VideoInstance> &instance) {
    if(instance->session.handle) return;
    const auto& video = instance->video;
    auto videoFormat = cb.formats.front();
    
    auto num_reference_frames = 0u;
    for (const auto &sps: video->sps_datas) {
        num_reference_frames = std::max(num_reference_frames, to<uint32_t>(sps.num_ref_frames));
    }
    num_reference_frames = std::min(num_reference_frames, cb.capabilities.maxActiveReferencePictures);

    VkVideoSessionCreateInfoKHR createInfo{
            .sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR,
            .queueFamilyIndex = device().queueFamilyIndex.video_decode.value(),
            .pVideoProfile = &cb.profile,
            .pictureFormat = videoFormat.format,
            .maxCodedExtent = {video->width, video->height},
            .referencePictureFormat = videoFormat.format,
            .maxDpbSlots = cb.capabilities.maxDpbSlots,
            .maxActiveReferencePictures = num_reference_frames * 2,
            .pStdHeaderVersion = &cb.capabilities.stdHeaderVersion
    };

    uint32_t count;
    ERR_GUARD_VULKAN(
            vkCreateVideoSessionKHR(device().logicalDevice, &createInfo, nullptr, &instance->session.handle));

    count = 0;
    ERR_GUARD_VULKAN(
            vkGetVideoSessionMemoryRequirementsKHR(device().logicalDevice, instance->session.handle, &count,
                                                   nullptr));
    std::vector<VkVideoSessionMemoryRequirementsKHR> videoMemoryRequirements(count,
                                                                             {.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR});
    ERR_GUARD_VULKAN(
            vkGetVideoSessionMemoryRequirementsKHR(device().logicalDevice, instance->session.handle, &count,
                                                   videoMemoryRequirements.data()));

    std::vector<VkBindVideoSessionMemoryInfoKHR> sessionMemoryInfo(count,
                                                                   {VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR});

    for (auto i = 0; i < count; ++i) {
        auto &memoryInfo = sessionMemoryInfo[i];
        memoryInfo.memoryBindIndex = videoMemoryRequirements[i].memoryBindIndex;
        memoryInfo.memoryOffset = 0;
        memoryInfo.memorySize = videoMemoryRequirements[i].memoryRequirements.size;

        auto memoryRequirement = videoMemoryRequirements[i].memoryRequirements;
        auto memoryProperty = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        auto memoryTypeIndex = device().getMemoryTypeIndex(memoryRequirement.memoryTypeBits, memoryProperty);

        VkMemoryAllocateInfo info{
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize = memoryInfo.memorySize,
                .memoryTypeIndex = memoryTypeIndex
        };
        VkDeviceMemory memory;
        ERR_GUARD_VULKAN(vkAllocateMemory(device().logicalDevice, &info, nullptr, &memory)); // TODO use VMA

        memoryInfo.memory = memory;
        instance->session.allocations.push_back(memory);

    }
    ERR_GUARD_VULKAN(
            vkBindVideoSessionMemoryKHR(device().logicalDevice, instance->session.handle, sessionMemoryInfo.size(),
                                        sessionMemoryInfo.data()));

    // create Session Params
    const auto spsCount = to<uint32_t>(video->sps_datas.size());
    std::vector<StdVideoH264SequenceParameterSet> vk_sps_list(spsCount);
    std::vector<StdVideoH264SequenceParameterSetVui> vk_vui_list(spsCount);
    std::vector<StdVideoH264HrdParameters> vk_hrd_list(spsCount);

    for (auto i = 0; i < spsCount; ++i) {
        translate(video->sps_datas[i], vk_sps_list[i], vk_vui_list[i], vk_hrd_list[i]);
    }

    const auto ppsCount = to<uint32_t>(video->pps_datas.size());
    std::vector<StdVideoH264PictureParameterSet> vk_pps_list(ppsCount);
    std::vector<StdVideoH264ScalingLists> vk_scalinglist(ppsCount);
    for (auto i = 0; i < ppsCount; ++i) {
        translate(video->pps_datas[i], vk_pps_list[i], vk_scalinglist[i]);
    }

    VkVideoDecodeH264SessionParametersAddInfoKHR addInfo{
            VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR};
    addInfo.stdSPSCount = spsCount;
    addInfo.pStdSPSs = vk_sps_list.data();
    addInfo.stdPPSCount = ppsCount;
    addInfo.pStdPPSs = vk_pps_list.data();

    VkVideoDecodeH264SessionParametersCreateInfoKHR H264SessionParamInfo{
            VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR};
    H264SessionParamInfo.maxStdSPSCount = spsCount;
    H264SessionParamInfo.maxStdPPSCount = ppsCount;
    H264SessionParamInfo.pParametersAddInfo = &addInfo;

    VkVideoSessionParametersCreateInfoKHR sessionParamInfo{VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR};
    sessionParamInfo.pNext = &H264SessionParamInfo;
    sessionParamInfo.videoSession = instance->session.handle;
    ERR_GUARD_VULKAN(vkCreateVideoSessionParametersKHR(device().logicalDevice, &sessionParamInfo, nullptr,
                                                       &instance->session.parameters));
    
    activeSessions.push_back(&instance->session);

}

void VideoDecoder::translate(const h264::SPS &sps, StdVideoH264SequenceParameterSet &vk_sps,
                             StdVideoH264SequenceParameterSetVui &vk_vui, StdVideoH264HrdParameters &vk_hrd) {
    vk_sps.flags.constraint_set0_flag = sps.constraint_set0_flag;
    vk_sps.flags.constraint_set1_flag = sps.constraint_set1_flag;
    vk_sps.flags.constraint_set2_flag = sps.constraint_set2_flag;
    vk_sps.flags.constraint_set3_flag = sps.constraint_set3_flag;
    vk_sps.flags.constraint_set4_flag = sps.constraint_set4_flag;
    vk_sps.flags.constraint_set5_flag = sps.constraint_set5_flag;
    vk_sps.flags.direct_8x8_inference_flag = sps.direct_8x8_inference_flag;
    vk_sps.flags.mb_adaptive_frame_field_flag = sps.mb_adaptive_frame_field_flag;
    vk_sps.flags.frame_mbs_only_flag = sps.frame_mbs_only_flag;
    vk_sps.flags.delta_pic_order_always_zero_flag = sps.delta_pic_order_always_zero_flag;
    vk_sps.flags.separate_colour_plane_flag = sps.separate_colour_plane_flag;
    vk_sps.flags.gaps_in_frame_num_value_allowed_flag = sps.gaps_in_frame_num_value_allowed_flag;
    vk_sps.flags.qpprime_y_zero_transform_bypass_flag = sps.qpprime_y_zero_transform_bypass_flag;
    vk_sps.flags.frame_cropping_flag = sps.frame_cropping_flag;
    vk_sps.flags.seq_scaling_matrix_present_flag = sps.seq_scaling_matrix_present_flag;
    vk_sps.flags.vui_parameters_present_flag = sps.vui_parameters_present_flag;

    if (vk_sps.flags.vui_parameters_present_flag) {
        vk_sps.pSequenceParameterSetVui = &vk_vui;
        vk_vui.flags.aspect_ratio_info_present_flag = sps.vui.aspect_ratio_info_present_flag;
        vk_vui.flags.overscan_info_present_flag = sps.vui.overscan_info_present_flag;
        vk_vui.flags.overscan_appropriate_flag = sps.vui.overscan_appropriate_flag;
        vk_vui.flags.video_signal_type_present_flag = sps.vui.video_signal_type_present_flag;
        vk_vui.flags.video_full_range_flag = sps.vui.video_full_range_flag;
        vk_vui.flags.color_description_present_flag = sps.vui.colour_description_present_flag;
        vk_vui.flags.chroma_loc_info_present_flag = sps.vui.chroma_loc_info_present_flag;
        vk_vui.flags.timing_info_present_flag = sps.vui.timing_info_present_flag;
        vk_vui.flags.fixed_frame_rate_flag = sps.vui.fixed_frame_rate_flag;
        vk_vui.flags.bitstream_restriction_flag = sps.vui.bitstream_restriction_flag;
        vk_vui.flags.nal_hrd_parameters_present_flag = sps.vui.nal_hrd_parameters_present_flag;
        vk_vui.flags.vcl_hrd_parameters_present_flag = sps.vui.vcl_hrd_parameters_present_flag;

        vk_vui.aspect_ratio_idc = (StdVideoH264AspectRatioIdc) sps.vui.aspect_ratio_idc;
        vk_vui.sar_width = sps.vui.sar_width;
        vk_vui.sar_height = sps.vui.sar_height;
        vk_vui.video_format = sps.vui.video_format;
        vk_vui.colour_primaries = sps.vui.colour_primaries;
        vk_vui.transfer_characteristics = sps.vui.transfer_characteristics;
        vk_vui.matrix_coefficients = sps.vui.matrix_coefficients;
        vk_vui.num_units_in_tick = sps.vui.num_units_in_tick;
        vk_vui.time_scale = sps.vui.time_scale;
        vk_vui.max_num_reorder_frames = sps.vui.num_reorder_frames;
        vk_vui.max_dec_frame_buffering = sps.vui.max_dec_frame_buffering;
        vk_vui.chroma_sample_loc_type_top_field = sps.vui.chroma_sample_loc_type_top_field;
        vk_vui.chroma_sample_loc_type_bottom_field = sps.vui.chroma_sample_loc_type_bottom_field;

        vk_vui.pHrdParameters = &vk_hrd;
        vk_hrd.cpb_cnt_minus1 = sps.hrd.cpb_cnt_minus1;
        vk_hrd.bit_rate_scale = sps.hrd.bit_rate_scale;
        vk_hrd.cpb_size_scale = sps.hrd.cpb_size_scale;
        for (int j = 0; j < std::size(sps.hrd.bit_rate_value_minus1); ++j) {
            vk_hrd.bit_rate_value_minus1[j] = sps.hrd.bit_rate_value_minus1[j];
            vk_hrd.cpb_size_value_minus1[j] = sps.hrd.cpb_size_value_minus1[j];
            vk_hrd.cbr_flag[j] = sps.hrd.cbr_flag[j];
        }
        vk_hrd.initial_cpb_removal_delay_length_minus1 = sps.hrd.initial_cpb_removal_delay_length_minus1;
        vk_hrd.cpb_removal_delay_length_minus1 = sps.hrd.cpb_removal_delay_length_minus1;
        vk_hrd.dpb_output_delay_length_minus1 = sps.hrd.dpb_output_delay_length_minus1;
        vk_hrd.time_offset_length = sps.hrd.time_offset_length;
    }

    vk_sps.profile_idc = (StdVideoH264ProfileIdc) sps.profile_idc;
    switch (sps.level_idc) {
        case 0:
            vk_sps.level_idc = STD_VIDEO_H264_LEVEL_IDC_1_0;
            break;
        case 11:
            vk_sps.level_idc = STD_VIDEO_H264_LEVEL_IDC_1_1;
            break;
        case 12:
            vk_sps.level_idc = STD_VIDEO_H264_LEVEL_IDC_1_2;
            break;
        case 13:
            vk_sps.level_idc = STD_VIDEO_H264_LEVEL_IDC_1_3;
            break;
        case 20:
            vk_sps.level_idc = STD_VIDEO_H264_LEVEL_IDC_2_0;
            break;
        case 21:
            vk_sps.level_idc = STD_VIDEO_H264_LEVEL_IDC_2_1;
            break;
        case 22:
            vk_sps.level_idc = STD_VIDEO_H264_LEVEL_IDC_2_2;
            break;
        case 30:
            vk_sps.level_idc = STD_VIDEO_H264_LEVEL_IDC_3_0;
            break;
        case 31:
            vk_sps.level_idc = STD_VIDEO_H264_LEVEL_IDC_3_1;
            break;
        case 32:
            vk_sps.level_idc = STD_VIDEO_H264_LEVEL_IDC_3_2;
            break;
        case 40:
            vk_sps.level_idc = STD_VIDEO_H264_LEVEL_IDC_4_0;
            break;
        case 41:
            vk_sps.level_idc = STD_VIDEO_H264_LEVEL_IDC_4_1;
            break;
        case 42:
            vk_sps.level_idc = STD_VIDEO_H264_LEVEL_IDC_4_2;
            break;
        case 50:
            vk_sps.level_idc = STD_VIDEO_H264_LEVEL_IDC_5_0;
            break;
        case 51:
            vk_sps.level_idc = STD_VIDEO_H264_LEVEL_IDC_5_1;
            break;
        case 52:
            vk_sps.level_idc = STD_VIDEO_H264_LEVEL_IDC_5_2;
            break;
        case 60:
            vk_sps.level_idc = STD_VIDEO_H264_LEVEL_IDC_6_0;
            break;
        case 61:
            vk_sps.level_idc = STD_VIDEO_H264_LEVEL_IDC_6_1;
            break;
        case 62:
            vk_sps.level_idc = STD_VIDEO_H264_LEVEL_IDC_6_2;
            break;
        default:
            assert(0);
            break;
    }
    assert(vk_sps.level_idc <= cb.h264DecodeCapabilities.maxLevelIdc);
    vk_sps.chroma_format_idc = (StdVideoH264ChromaFormatIdc)sps.chroma_format_idc;
    vk_sps.seq_parameter_set_id = sps.seq_parameter_set_id;
    vk_sps.bit_depth_luma_minus8 = sps.bit_depth_luma_minus8;
    vk_sps.bit_depth_chroma_minus8 = sps.bit_depth_chroma_minus8;
    vk_sps.log2_max_frame_num_minus4 = sps.log2_max_frame_num_minus4;
    vk_sps.pic_order_cnt_type = (StdVideoH264PocType) sps.pic_order_cnt_type;
    vk_sps.offset_for_non_ref_pic = sps.offset_for_non_ref_pic;
    vk_sps.offset_for_top_to_bottom_field = sps.offset_for_top_to_bottom_field;
    vk_sps.log2_max_pic_order_cnt_lsb_minus4 = sps.log2_max_pic_order_cnt_lsb_minus4;
    vk_sps.num_ref_frames_in_pic_order_cnt_cycle = sps.num_ref_frames_in_pic_order_cnt_cycle;
    vk_sps.max_num_ref_frames = sps.num_ref_frames;
    vk_sps.pic_width_in_mbs_minus1 = sps.pic_width_in_mbs_minus1;
    vk_sps.pic_height_in_map_units_minus1 = sps.pic_height_in_map_units_minus1;
    vk_sps.frame_crop_left_offset = sps.frame_crop_left_offset;
    vk_sps.frame_crop_right_offset = sps.frame_crop_right_offset;
    vk_sps.frame_crop_top_offset = sps.frame_crop_top_offset;
    vk_sps.frame_crop_bottom_offset = sps.frame_crop_bottom_offset;
    vk_sps.pOffsetForRefFrame = sps.offset_for_ref_frame;
}

void VideoDecoder::translate(const h264::PPS &pps, StdVideoH264PictureParameterSet &vk_pps,
                             StdVideoH264ScalingLists &vk_scalinglist) {
    vk_pps.flags.transform_8x8_mode_flag = pps.transform_8x8_mode_flag;
    vk_pps.flags.redundant_pic_cnt_present_flag = pps.redundant_pic_cnt_present_flag;
    vk_pps.flags.constrained_intra_pred_flag = pps.constrained_intra_pred_flag;
    vk_pps.flags.deblocking_filter_control_present_flag = pps.deblocking_filter_control_present_flag;
    vk_pps.flags.weighted_pred_flag = pps.weighted_pred_flag;
    vk_pps.flags.bottom_field_pic_order_in_frame_present_flag = pps.pic_order_present_flag;
    vk_pps.flags.entropy_coding_mode_flag = pps.entropy_coding_mode_flag;
    vk_pps.flags.pic_scaling_matrix_present_flag = pps.pic_scaling_matrix_present_flag;

    vk_pps.seq_parameter_set_id = pps.seq_parameter_set_id;
    vk_pps.pic_parameter_set_id = pps.pic_parameter_set_id;
    vk_pps.num_ref_idx_l0_default_active_minus1 = pps.num_ref_idx_l0_active_minus1;
    vk_pps.num_ref_idx_l1_default_active_minus1 = pps.num_ref_idx_l1_active_minus1;
    vk_pps.weighted_bipred_idc = to<StdVideoH264WeightedBipredIdc>(pps.weighted_bipred_idc);
    vk_pps.pic_init_qp_minus26 = to<int8_t>(pps.pic_init_qp_minus26);
    vk_pps.pic_init_qs_minus26 = to<int8_t>(pps.pic_init_qs_minus26);
    vk_pps.chroma_qp_index_offset = to<int8_t>(pps.chroma_qp_index_offset);
    vk_pps.second_chroma_qp_index_offset = to<int8_t>(pps.second_chroma_qp_index_offset);

    vk_pps.pScalingLists = &vk_scalinglist;
    for (int j = 0; j < std::size(pps.pic_scaling_list_present_flag); ++j) {
        vk_scalinglist.scaling_list_present_mask |= pps.pic_scaling_list_present_flag[j] << j;
    }
    for (int j = 0; j < std::size(pps.UseDefaultScalingMatrix4x4Flag); ++j) {
        vk_scalinglist.use_default_scaling_matrix_mask |= pps.UseDefaultScalingMatrix4x4Flag[j] << j;
    }
    for (int j = 0; j < std::size(pps.ScalingList4x4); ++j) {
        for (int k = 0; k < std::size(pps.ScalingList4x4[j]); ++k) {
            vk_scalinglist.ScalingList4x4[j][k] = (uint8_t) pps.ScalingList4x4[j][k];
        }
    }
    for (int j = 0; j < std::size(pps.ScalingList8x8); ++j) {
        for (int k = 0; k < std::size(pps.ScalingList8x8[j]); ++k) {
            vk_scalinglist.ScalingList8x8[j][k] = (uint8_t) pps.ScalingList8x8[j][k];
        }
    }
}

void VideoDecoder::decode(const std::shared_ptr<VideoInstance> &instance, VkCommandBuffer commandBuffer) {
    auto& dpb = instance->dpb;
    const auto& video = instance->video;

    static int sequence = 0;

    while(instance->isDecodingRequired(device())) {
        if(instance->output_textures_free.empty()) {
            auto& output = instance->output_textures_free.emplace_back();
            output.display.texture.width = instance->width();
            output.display.texture.height = instance->height();
            output.textureId = sequence++;
            createDpbOutputTexture(output, fmt::format("dpb_output_{}", output.textureId));
        }

        auto oDecode = std::move(instance->output_textures_free.back());
        instance->output_textures_free.pop_back();

        oDecode.display_order = instance->video->frame_infos[instance->current_decode_frame].display_order;
        oDecode.src.texture = &instance->dpb.texture;
        oDecode.src.subresource_luminance = instance->dpb.subresources_luminance[instance->dpb.next_slot];
        oDecode.src.subresource_chrominance = instance->dpb.subresources_chrominance[instance->dpb.next_slot];
        oDecode.src.imageview = instance->dpb.image_views[instance->dpb.next_slot].handle;

        if(oDecode.display_order < instance->target_display_order) {
            // next decoded is lower display order than we will need, it can be immediately freed after decode
            instance->output_textures_free.push_back(std::move(oDecode));
        }else {
            instance->output_textures_resolve_request.push_back(instance->output_textures_used.size());
            instance->output_textures_used.push_back(std::move(oDecode));
        }

        instance->current_decode_frame = std::clamp(instance->current_decode_frame, 0, (int)video->frame_infos.size() - 1);
        const Video::FrameInfo& frame = video->frame_infos[instance->current_decode_frame];

        const auto slice_header = (const h264::SliceHeader*)video->slice_header_datas.data() + instance->current_decode_frame;
        const auto pps = (const h264::PPS*)video->pps_datas.data() + slice_header->pic_parameter_set_id;
        const auto sps = (const h264::SPS*)video->sps_datas.data() + pps->seq_parameter_set_id;

        VideoDecodeOperation decodeOp{};
        if (instance->current_decode_frame == 0 || has_flag(instance->flags, VideoInstance::Flags::DecoderReset)){
            decodeOp.flags = VideoDecodeOperation::Flag::SESSION_RESET;
            instance->flags &= ~VideoInstance::Flags::DecoderReset;
        }
        if (frame.type == VideoFrameType::Intra){
            instance->dpb.reference_usage.clear();
            instance->dpb.next_ref = 0;
            instance->dpb.next_slot = 0;
        }

        instance->dpb.current_slot = instance->dpb.next_slot;
        instance->dpb.poc_status[instance->dpb.current_slot] = frame.poc;
        instance->dpb.framenum_status[instance->dpb.current_slot] = slice_header->frame_num;
        decodeOp.stream = video->data_stream;
        decodeOp.stream_offset = frame.offset;
        decodeOp.stream_size = frame.size;
        decodeOp.poc[0] = frame.poc;
        decodeOp.poc[1] = frame.poc;
        decodeOp.frame_type = frame.type;
        decodeOp.reference_priority = frame.reference_priority;
        decodeOp.decoded_frame_index = instance->current_decode_frame;
        decodeOp.slice_header = slice_header;
        decodeOp.pps = pps;
        decodeOp.sps = sps;
        decodeOp.current_dpb = instance->dpb.current_slot;
        decodeOp.dpb_reference_count = (uint32_t)instance->dpb.reference_usage.size();
        decodeOp.dpb_reference_slots = instance->dpb.reference_usage;
        decodeOp.dpb_poc = instance->dpb.poc_status;
        decodeOp.dpb_framenum = instance->dpb.framenum_status;
        decodeOp.DPB = &instance->dpb.texture;
        decodeOp.session = &instance->session;

        if(!instance->output_textures_used.empty() && instance->output_textures_used.back().display_order > instance->target_display_order || frame.reference_priority > 0) {
            if(dpb.resource_states[dpb.current_slot].layout != VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR) {
                Barriers::push(instance->dpb.texture.image, dpb.subresources_luminance[dpb.current_slot], VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR,
                               dpb.resource_states[dpb.current_slot].access, VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR, dpb.resource_states[dpb.current_slot].layout, VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);

                Barriers::push(instance->dpb.texture.image, dpb.subresources_chrominance[dpb.current_slot], VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR,
                               dpb.resource_states[dpb.current_slot].access, VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR, dpb.resource_states[dpb.current_slot].layout, VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);

                dpb.resource_states[dpb.current_slot] = { VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR, VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR, VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR};
            }

            for(auto i = 0; i < dpb.reference_usage.size(); ++i) {
                auto ref = dpb.reference_usage[i];
                if(dpb.resource_states[ref].layout != VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR) {
                    Barriers::push(instance->dpb.texture.image, dpb.subresources_luminance[ref], VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR,
                                   dpb.resource_states[ref].access, VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR, dpb.resource_states[ref].layout, VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);

                    Barriers::push(instance->dpb.texture.image, dpb.subresources_chrominance[ref], VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR,
                                   dpb.resource_states[ref].access, VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR, dpb.resource_states[ref].layout, VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);

                    dpb.resource_states[dpb.current_slot] = { VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR, VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR, VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR };
                }
            }

            Barriers::flush(commandBuffer);
            decode(decodeOp, commandBuffer);

        }

        // DPB slot management:
        //	When current frame was a reference, then the next frame can not overwrite its DPB slot, so increment next_slot as a ring buffer
        //	However, the ring buffer will wrap around so older reference frames can be overwritten by this
        if (frame.reference_priority > 0)
        {
            if (instance->dpb.next_ref >= instance->dpb.reference_usage.size())
            {
                instance->dpb.reference_usage.resize(instance->dpb.next_ref + 1);
            }
            instance->dpb.reference_usage[instance->dpb.next_ref] = instance->dpb.current_slot;
            instance->dpb.next_ref = (instance->dpb.next_ref + 1) % (instance->dpb.texture.layers - 1);
            instance->dpb.next_slot = (instance->dpb.next_slot + 1) % instance->dpb.texture.layers;
        }

        instance->flags |= VideoInstance::Flags::NeedsResolve;
        instance->flags |= VideoInstance::Flags::FirstFrameDecoded;
        instance->current_decode_frame++;
    }

    std::vector<std::string> released;
    for(auto& rid : instance->output_textures_resolve_request) {
        auto& out = instance->output_textures_used[rid];
        Barriers::release(out.src.texture->image, out.src.subresource_luminance, VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR,
                          VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR, VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                          *device().queueFamilyIndex.video_decode, *device().queueFamilyIndex.graphics);

        Barriers::release(out.src.texture->image, out.src.subresource_chrominance, VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR,
                          VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR, VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                          *device().queueFamilyIndex.video_decode, *device().queueFamilyIndex.graphics);

        released.push_back(out.name);
    }
    if(!released.empty()) {
        spdlog::info("released {}, from video_decode to graphics", released);
        released.clear();
    }
    Barriers::flush(commandBuffer);
}

void VideoDecoder::decode(const VideoDecodeOperation &decodeOperation, VkCommandBuffer commandBuffer) {
    const auto op = decodeOperation;
    const auto slice_header = op.slice_header;
    const auto pps = op.pps;
    const auto sps = op.sps;

    StdVideoDecodeH264PictureInfo std_picture_info_h264 = {};
    std_picture_info_h264.pic_parameter_set_id = slice_header->pic_parameter_set_id;
    std_picture_info_h264.seq_parameter_set_id = pps->seq_parameter_set_id;
    std_picture_info_h264.frame_num = slice_header->frame_num;
    std_picture_info_h264.PicOrderCnt[0] = op.poc[0];
    std_picture_info_h264.PicOrderCnt[1] = op.poc[1];
    std_picture_info_h264.idr_pic_id = slice_header->idr_pic_id;
    std_picture_info_h264.flags.is_intra = op.frame_type == VideoFrameType::Intra ? 1 : 0;
    std_picture_info_h264.flags.is_reference = op.reference_priority > 0 ? 1 : 0;
    std_picture_info_h264.flags.IdrPicFlag = (std_picture_info_h264.flags.is_intra &&
                                              std_picture_info_h264.flags.is_reference) ? 1 : 0;
    std_picture_info_h264.flags.field_pic_flag = slice_header->field_pic_flag;
    std_picture_info_h264.flags.bottom_field_flag = slice_header->bottom_field_flag;
    std_picture_info_h264.flags.complementary_field_pair = 0;

    std::array<VkVideoReferenceSlotInfoKHR, 17> reference_slot_infos{};
    std::array<VkVideoPictureResourceInfoKHR, 17> reference_slot_pictures{};
    std::array<VkVideoDecodeH264DpbSlotInfoKHR, 17> dpb_slots_h264{};
    std::array<StdVideoDecodeH264ReferenceInfo, 17> reference_infos_h264{};
    for (int32_t i = 0; i < op.DPB->layers; ++i) {

        VkVideoPictureResourceInfoKHR &pic = reference_slot_pictures[i];
        pic.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
        pic.codedOffset.x = 0;
        pic.codedOffset.y = 0;
        pic.codedExtent.width = op.DPB->width;
        pic.codedExtent.height = op.DPB->height;
        pic.baseArrayLayer = i;
        pic.imageViewBinding = op.DPB->imageView.handle;

        StdVideoDecodeH264ReferenceInfo &ref = reference_infos_h264[i];
        ref.flags.bottom_field_flag = 0;
        ref.flags.top_field_flag = 0;
        ref.flags.is_non_existing = 0;
        ref.flags.used_for_long_term_reference = 0;
        ref.FrameNum = op.dpb_framenum[i];
        ref.PicOrderCnt[0] = op.dpb_poc[i];
        ref.PicOrderCnt[1] = op.dpb_poc[i];

        VkVideoDecodeH264DpbSlotInfoKHR &dpb = dpb_slots_h264[i];
        dpb.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR;
        dpb.pStdReferenceInfo = &ref;

        VkVideoReferenceSlotInfoKHR &slot = reference_slot_infos[i];
        slot.sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
        slot.pPictureResource = &pic;
        slot.slotIndex = i;
        slot.pNext = &dpb;

    }

    std::array<VkVideoReferenceSlotInfoKHR, 17> reference_slots{};
    for (size_t i = 0; i < op.dpb_reference_count; ++i) {
        uint32_t ref_slot = op.dpb_reference_slots[i];
        assert(ref_slot != op.current_dpb);
        reference_slots[i] = reference_slot_infos[ref_slot];
    }
    reference_slots[op.dpb_reference_count] = reference_slot_infos[op.current_dpb];
    reference_slots[op.dpb_reference_count].slotIndex = -1;

    VkVideoBeginCodingInfoKHR begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR;
    begin_info.videoSession = op.session->handle;
    begin_info.videoSessionParameters = op.session->parameters;
    begin_info.referenceSlotCount = op.dpb_reference_count + 1; // add in the current reconstructed DPB image
    begin_info.pReferenceSlots = begin_info.referenceSlotCount == 0 ? nullptr : reference_slots.data();

    vkCmdBeginVideoCodingKHR(commandBuffer, &begin_info);

    if (op.flags & VideoDecodeOperation::Flag::SESSION_RESET) {
        VkVideoCodingControlInfoKHR control_info = {};
        control_info.sType = VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR;
        control_info.flags = VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR;
        vkCmdControlVideoCodingKHR(commandBuffer, &control_info);
    }

    VkVideoDecodeInfoKHR decode_info = {};
    decode_info.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR;
    decode_info.srcBuffer = op.stream.buffer;
    decode_info.srcBufferOffset = (VkDeviceSize) op.stream_offset;
    decode_info.srcBufferRange = (VkDeviceSize) alignedSize(op.stream_size, VIDEO_DECODE_BITSTREAM_ALIGNMENT);
    if (op.output == nullptr) {
        decode_info.dstPictureResource = *reference_slot_infos[op.current_dpb].pPictureResource;
    } else {
        auto output_internal = op.output;
        decode_info.dstPictureResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
        decode_info.dstPictureResource.codedOffset.x = 0;
        decode_info.dstPictureResource.codedOffset.y = 0;
        decode_info.dstPictureResource.codedExtent.width = op.DPB->width;
        decode_info.dstPictureResource.codedExtent.height = op.DPB->height;
        decode_info.dstPictureResource.baseArrayLayer = 0;
        decode_info.dstPictureResource.imageViewBinding = output_internal->imageView.handle;
    }
    decode_info.referenceSlotCount = op.dpb_reference_count;
    decode_info.pReferenceSlots = decode_info.referenceSlotCount == 0 ? nullptr : reference_slots.data();
    decode_info.pSetupReferenceSlot = &reference_slot_infos[op.current_dpb];

    uint32_t slice_offset = 0;

    // https://vulkan.lunarg.com/doc/view/1.3.239.0/windows/1.3-extensions/vkspec.html#_h_264_decoding_parameters
    VkVideoDecodeH264PictureInfoKHR picture_info_h264 = {};
    picture_info_h264.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PICTURE_INFO_KHR;
    picture_info_h264.pStdPictureInfo = &std_picture_info_h264;
    picture_info_h264.sliceCount = 1;
    picture_info_h264.pSliceOffsets = &slice_offset;
    decode_info.pNext = &picture_info_h264;

    vkCmdDecodeVideoKHR(commandBuffer, &decode_info);

    VkVideoEndCodingInfoKHR end_info = {};
    end_info.sType = VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR;
    vkCmdEndVideoCodingKHR(commandBuffer, &end_info);
}

void VideoDecoder::createDpbResources(std::shared_ptr<VideoInstance> &instance) {
    if(instance->dpb.texture.image.image) return;
    const auto& video = instance->video;
    auto videoFormat = cb.formats.front();

    auto &dpb = instance->dpb;
    const auto num_dpb_slots = video->num_dpb_slots;

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.pNext = &cb.profiles;
    imageCreateInfo.imageType = videoFormat.imageType;
    imageCreateInfo.format = videoFormat.format;
    imageCreateInfo.extent = {video->width, video->height, 1};
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = num_dpb_slots;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = videoFormat.imageTiling;
    imageCreateInfo.usage =
            VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR
            | VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR
            | VK_IMAGE_USAGE_SAMPLED_BIT
            | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
            | VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    dpb.texture.image = device().createImage(imageCreateInfo);
    device().setName<VK_OBJECT_TYPE_IMAGE>("video_dpb_image", dpb.texture.image.image);

    VkImageSubresourceRange subresourceRange;
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = 1;
    subresourceRange.baseArrayLayer = 0;
    subresourceRange.layerCount = num_dpb_slots;


    VkSamplerYcbcrConversionInfo  conversionInfo {
            .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO,
            .conversion = ycbcrConversion
    };
    VkImageViewCreateInfo  createImageviewInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = &conversionInfo,
            .image = dpb.texture.image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
            .format = videoFormat.format,
            .subresourceRange = subresourceRange,
    };
    dpb.texture.imageView = device().createImageView(createImageviewInfo);
    device().setName<VK_OBJECT_TYPE_IMAGE_VIEW>("video_dpb_image_view", dpb.texture.imageView.handle);

    for(auto i = 0u; i < num_dpb_slots; ++i) {
        dpb.subresources_luminance[i] = {
                .aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = i,
                .layerCount = 1
        };
    }

    for(auto i = 0u; i < num_dpb_slots; ++i) {
        dpb.subresources_chrominance[i] = {
                .aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = i,
                .layerCount = 1
        };
    }

    createImageviewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    for(auto i = 0; i < num_dpb_slots; ++i) {
        createImageviewInfo.subresourceRange = dpb.subresources_luminance[i];
        createImageviewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        dpb.image_views[i] = device().createImageView(createImageviewInfo);
        device().setName<VK_OBJECT_TYPE_IMAGE_VIEW>(fmt::format("video_dpb_image_view_{}", i), dpb.image_views[i].handle);
    }

    device().videoDecodeCommandPool().oneTimeCommand([&](auto commandBuffer) {
        const auto numBarriers = num_dpb_slots * 2;
        std::vector<VkImageMemoryBarrier2> barriers(
                numBarriers, {
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                        .srcStageMask = VK_PIPELINE_STAGE_NONE,
                        .srcAccessMask = VK_ACCESS_NONE,
                        .dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
                        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        .image = dpb.texture.image.image,
                        .subresourceRange = subresourceRange
                }
        );
        for(auto i = 0; i < numBarriers;  ++i) {
            if(i < num_dpb_slots) {
                barriers[i].subresourceRange = dpb.subresources_luminance[i];
            }else {
                barriers[i].subresourceRange = dpb.subresources_chrominance[i % num_dpb_slots];
            }
        }

        VkDependencyInfo info {
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .imageMemoryBarrierCount = COUNT(barriers),
                .pImageMemoryBarriers = barriers.data()
        };

        vkCmdPipelineBarrier2(commandBuffer, &info);
    });

    dpb.texture.format = videoFormat.format;
    dpb.texture.width = video->width;
    dpb.texture.height = video->height;
    dpb.texture.layers = num_dpb_slots;
    dpb.texture.depth = 1;
    dpb.texture.spec = imageCreateInfo;
}

void VideoDecoder::initialize(std::shared_ptr<VideoInstance> &instance) {
    createDpbResources(instance);
    createVideoSession(instance);
}

std::vector<PipelineMetaData> VideoDecoder::pipelineMetaData() {
    return {
            {
                .name = "resolve_rgb",
                .shadePath = FileManager::resource("rgb_resolve.comp.spv"),
                .layouts = { &rgbResolveDescriptorSetLayout },
                .ranges = { {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int) * 2} }
            }
    };
}

void VideoDecoder::resolveToRGB(const std::shared_ptr<VideoInstance> &instance, VkCommandBuffer commandBuffer) {
    if(!instance->video) return;
    if(!has_flag(instance->flags, VideoInstance::Flags::NeedsResolve)) return;
    instance->flags &= ~VideoInstance::Flags::NeedsResolve;
    
    auto decode_queue = device().queueFamilyIndex.video_decode.value();
    auto graphics_queue = device().queueFamilyIndex.graphics.value();
    auto gx = to<uint32_t>(std::ceil((instance->video->width + 32u)/32u));
    auto gy = to<uint32_t>(std::ceil((instance->video->height + 32u)/32u));

    std::vector<std::string> acquired;
    std::vector<std::string> released;
    for(auto resolveId : instance->output_textures_resolve_request) {
        auto& out = instance->output_textures_used[resolveId];
        Barriers::acquire(out.src.texture->image, out.src.subresource_luminance, VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, decode_queue, graphics_queue);
        Barriers::acquire(out.src.texture->image, out.src.subresource_chrominance, VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, decode_queue, graphics_queue);

        out.src.state = { VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT };

        if(out.display.state.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
            Barriers::push(out.display.texture.image, out.display.subresource, VK_PIPELINE_STAGE_NONE,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_NONE, VK_ACCESS_SHADER_WRITE_BIT,
                           VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }

        acquired.push_back(out.name);
        Barriers::flush(commandBuffer);

        updateSrcDescriptor(out);

        static std::array<int, 2> constants;
        constants[0] = to<int>(out.src.subresource_luminance.baseArrayLayer);
        constants[1] = out.textureId;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline("resolve_rgb"));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout("resolve_rgb"), 0, 1, &rgbResolveDescriptorSet, 0, nullptr);
        vkCmdPushConstants(commandBuffer, layout("resolve_rgb"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int) * 2, constants.data());
        vkCmdDispatch(commandBuffer, gx, gy, 1);
        
        Barrier::computeWriteToFragmentRead(commandBuffer);

        Barriers::release(out.src.texture->image, out.src.subresource_luminance, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR,
                          graphics_queue, decode_queue);

        Barriers::release(out.src.texture->image, out.src.subresource_chrominance, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR,
                          graphics_queue, decode_queue);
        released.push_back(out.name);
        Barriers::flush(commandBuffer);
    }
    if(!acquired.empty()) {
        spdlog::info("acquired {}, from video_decode to graphics", acquired);
        acquired.clear();
    }
    if(!released.empty()) {
        spdlog::info("released {}, from graphics to video_decode", released);
        released.clear();
    }
    instance->output_textures_resolve_request.clear();
}

void VideoDecoder::createDescriptorPool() {
    constexpr uint32_t maxSets = MaxDescriptorResources;
    std::array<VkDescriptorPoolSize, 2> poolSizes{
            {
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,  maxSets},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, maxSets},
            }
    };
    descriptorPool = device().createDescriptorPool(maxSets, poolSizes, VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT);
}

void VideoDecoder::createDescriptorSetLayout() {
    std::vector<VkSampler> samplers(MaxDescriptorResources);
    std::generate(samplers.begin(), samplers.end(), [&]{ return yuvSampler.handle; });

    rgbResolveDescriptorSetLayout =
            device().descriptorSetLayoutBuilder()
                .name("rgb_resolve_descriptor_set_layout")
                .bindless()
                .binding(0)
                    .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
                    .immutableSamplers(samplers.data())
                .binding(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(MaxDescriptorResources)
                    .shaderStages(VK_SHADER_STAGE_COMPUTE_BIT)
            .createLayout();
    auto sets = descriptorPool.allocate( { rgbResolveDescriptorSetLayout } );
    rgbResolveDescriptorSet = sets[0];
}

void VideoDecoder::updateSrcDescriptor(OutputTexture &output) {
    
    static auto writes = initializers::writeDescriptorSets<1>();
    writes[0].dstSet = rgbResolveDescriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].descriptorCount = 1;
    VkDescriptorImageInfo srcImageInfo{nullptr, output.src.texture->imageView.handle, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    writes[0].pImageInfo = &srcImageInfo;


    device().updateDescriptorSets(writes);
}


void VideoDecoder::updateDescriptors(OutputTexture &output) {
    static auto writes = initializers::writeDescriptorSets<1>();

    writes[0].dstSet = rgbResolveDescriptorSet;
    writes[0].dstBinding = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].descriptorCount = 1;
    writes[0].dstArrayElement = output.textureId;
    VkDescriptorImageInfo dstImageInfo{nullptr, output.display.texture.imageView.handle, VK_IMAGE_LAYOUT_GENERAL};
    writes[0].pImageInfo = &dstImageInfo;

    device().updateDescriptorSets(writes);
}