#include "Texture.h"
#include <fmt/format.h>
#include <glm/glm.hpp>
#include "sampling.hpp"
#include <ImfArray.h>
#include <ImfChannelList.h>
#include <ImfMatrixAttribute.h>
#include <ImfStringAttribute.h>
#include <ImfTiledInputFile.h>
#include <ImfTiledOutputFile.h>
#include <ImfNamespace.h>
#include <ImfRgbaFile.h>

#ifndef STBI_MSC_SECURE_CRT
#define STBI_MSC_SECURE_CRT
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif // STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#endif // STBI_MSC_SECURE_CRT

namespace IMF = OPENEXR_IMF_NAMESPACE;
using namespace IMF;
using namespace IMATH_NAMESPACE;

bool isIntegral(VkFormat format){
    switch(format){
        case VK_FORMAT_R32_SINT:
        case VK_FORMAT_R32G32_SINT:
        case VK_FORMAT_R32G32B32_SINT:
        case VK_FORMAT_R32G32B32A32_SINT:
        case VK_FORMAT_R32_UINT:
        case VK_FORMAT_R32G32_UINT:
        case VK_FORMAT_R32G32B32_UINT:
        case VK_FORMAT_R32G32B32A32_UINT:
            return true;
        default:
            return false;
    }
}

VkDeviceSize textures::byteSize(VkFormat format){
    switch(format){
        case VK_FORMAT_R8_UNORM:
        case VK_FORMAT_R8_SNORM:
        case VK_FORMAT_R8_USCALED:
        case VK_FORMAT_R8_SSCALED:
        case VK_FORMAT_R8_UINT:
        case VK_FORMAT_R8_SINT:
        case VK_FORMAT_R8_SRGB:
        case VK_FORMAT_R8G8_UNORM:
        case VK_FORMAT_R8G8_SNORM:
        case VK_FORMAT_R8G8_USCALED:
        case VK_FORMAT_R8G8_SSCALED:
        case VK_FORMAT_R8G8_UINT:
        case VK_FORMAT_R8G8_SINT:
        case VK_FORMAT_R8G8_SRGB:
        case VK_FORMAT_R8G8B8_UNORM:
        case VK_FORMAT_R8G8B8_SNORM:
        case VK_FORMAT_R8G8B8_USCALED:
        case VK_FORMAT_R8G8B8_SSCALED:
        case VK_FORMAT_R8G8B8_UINT:
        case VK_FORMAT_R8G8B8_SINT:
        case VK_FORMAT_R8G8B8_SRGB:
        case VK_FORMAT_B8G8R8_UNORM:
        case VK_FORMAT_B8G8R8_SNORM:
        case VK_FORMAT_B8G8R8_USCALED:
        case VK_FORMAT_B8G8R8_SSCALED:
        case VK_FORMAT_B8G8R8_UINT:
        case VK_FORMAT_B8G8R8_SINT:
        case VK_FORMAT_B8G8R8_SRGB:
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SNORM:
        case VK_FORMAT_R8G8B8A8_USCALED:
        case VK_FORMAT_R8G8B8A8_SSCALED:
        case VK_FORMAT_R8G8B8A8_UINT:
        case VK_FORMAT_R8G8B8A8_SINT:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SNORM:
        case VK_FORMAT_B8G8R8A8_USCALED:
        case VK_FORMAT_B8G8R8A8_SSCALED:
        case VK_FORMAT_B8G8R8A8_UINT:
        case VK_FORMAT_B8G8R8A8_SINT:
        case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
        case VK_FORMAT_A8B8G8R8_SNORM_PACK32:
        case VK_FORMAT_A8B8G8R8_USCALED_PACK32:
        case VK_FORMAT_A8B8G8R8_SSCALED_PACK32:
        case VK_FORMAT_A8B8G8R8_UINT_PACK32:
        case VK_FORMAT_A8B8G8R8_SINT_PACK32:
        case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
            return 1;
        case VK_FORMAT_R16_UNORM:
        case VK_FORMAT_R16_SNORM:
        case VK_FORMAT_R16_USCALED:
        case VK_FORMAT_R16_SSCALED:
        case VK_FORMAT_R16_UINT:
        case VK_FORMAT_R16_SINT:
        case VK_FORMAT_R16_SFLOAT:
        case VK_FORMAT_R16G16_UNORM:
        case VK_FORMAT_R16G16_SNORM:
        case VK_FORMAT_R16G16_USCALED:
        case VK_FORMAT_R16G16_SSCALED:
        case VK_FORMAT_R16G16_UINT:
        case VK_FORMAT_R16G16_SINT:
        case VK_FORMAT_R16G16_SFLOAT:
        case VK_FORMAT_R16G16B16_UNORM:
        case VK_FORMAT_R16G16B16_SNORM:
        case VK_FORMAT_R16G16B16_USCALED:
        case VK_FORMAT_R16G16B16_SSCALED:
        case VK_FORMAT_R16G16B16_UINT:
        case VK_FORMAT_R16G16B16_SINT:
        case VK_FORMAT_R16G16B16_SFLOAT:
        case VK_FORMAT_R16G16B16A16_UNORM:
        case VK_FORMAT_R16G16B16A16_SNORM:
        case VK_FORMAT_R16G16B16A16_USCALED:
        case VK_FORMAT_R16G16B16A16_SSCALED:
        case VK_FORMAT_R16G16B16A16_UINT:
        case VK_FORMAT_R16G16B16A16_SINT:
        case VK_FORMAT_R16G16B16A16_SFLOAT:
        case VK_FORMAT_D16_UNORM:
            return 2;
        case VK_FORMAT_R32_UINT:
        case VK_FORMAT_R32_SINT:
        case VK_FORMAT_R32_SFLOAT:
        case VK_FORMAT_R32G32_UINT:
        case VK_FORMAT_R32G32_SINT:
        case VK_FORMAT_R32G32_SFLOAT:
        case VK_FORMAT_R32G32B32_UINT:
        case VK_FORMAT_R32G32B32_SINT:
        case VK_FORMAT_R32G32B32_SFLOAT:
        case VK_FORMAT_R32G32B32A32_UINT:
        case VK_FORMAT_R32G32B32A32_SINT:
        case VK_FORMAT_R32G32B32A32_SFLOAT:
            return 4;
        default:
            throw std::runtime_error{fmt::format("format: {} not implemented", format)};
    }
}

VkImageViewType getImageViewType(VkImageType imageType){
    switch(imageType){
        case VK_IMAGE_TYPE_1D: return VK_IMAGE_VIEW_TYPE_1D;
        case VK_IMAGE_TYPE_2D: return VK_IMAGE_VIEW_TYPE_2D;
        case VK_IMAGE_TYPE_3D: return VK_IMAGE_VIEW_TYPE_3D;
        default:
            throw std::runtime_error{"invalid imageType"};
    }
}

uint32_t nunChannels(VkFormat format) {
    switch (format) {
        case VK_FORMAT_R8_SRGB:
        case VK_FORMAT_R32_SFLOAT:
        case VK_FORMAT_D32_SFLOAT:
        case VK_FORMAT_R8_UNORM:
        case VK_FORMAT_R32_UINT:
            return 1;
        case VK_FORMAT_R8G8_SRGB:
        case VK_FORMAT_R8G8_UNORM:
        case VK_FORMAT_R16G16_SFLOAT:
        case VK_FORMAT_R32G32_SINT:
        case VK_FORMAT_R32G32_SFLOAT:
        case VK_FORMAT_D16_UNORM:
            return 2;
        case VK_FORMAT_R8G8B8_SRGB:
        case VK_FORMAT_B8G8R8_SRGB:
        case VK_FORMAT_R8G8B8_UNORM:
        case VK_FORMAT_R16G16B16_SFLOAT:
        case VK_FORMAT_R32G32B32_SFLOAT:
            return 3;
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_R16G16B16A16_SFLOAT:
        case VK_FORMAT_R32G32B32A32_SFLOAT:
            return 4;
        default:
            throw std::runtime_error{fmt::format("format: {}, not implemented", format)};
    }
}

constexpr VkFormat getFormat(uint32_t numChannels){
    switch(numChannels){
        case 1:
            return VK_FORMAT_R8_SRGB;
        case 2:
            return VK_FORMAT_R8G8_SRGB;
        case 3:
            return VK_FORMAT_R8G8B8_SRGB;
        case 4:
            return VK_FORMAT_R8G8B8A8_SRGB;
        default:
            return VK_FORMAT_UNDEFINED;
    }
}

constexpr bool isDepthTexture(VkFormat format) {
    switch(format) {
        case VK_FORMAT_D16_UNORM:
        case VK_FORMAT_D16_UNORM_S8_UINT:
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
        case VK_FORMAT_D32_SFLOAT:
            return true;
        default:
            return false;
    }
}

RawImage textures::loadImage(std::string_view path, bool flipUv) {
    int texWidth, texHeight,  texChannels;
    stbi_set_flip_vertically_on_load(flipUv ? 1 : 0);
    stbi_uc* pixels = stbi_load(path.data(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    return RawImage{
        std::unique_ptr<stbi_uc, stbi_image_deleter>(pixels),
                static_cast<uint32_t>(texWidth),
                static_cast<uint32_t>(texHeight),
                texChannels
    };
}

void textures::fromFile(const VulkanDevice &device, Texture &texture, std::string_view path, bool flipUv, VkFormat format, uint32_t levelCount) {
    int texWidth, texHeight, texChannels;
    stbi_set_flip_vertically_on_load(flipUv ? 1 : 0);
    stbi_uc* pixels = stbi_load(path.data(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    texture.width = texWidth;
    texture.height = texHeight;
    if(!pixels){
        throw std::runtime_error{fmt::format("failed to load texture image {}!", path)};
    }
    create(device, texture, VK_IMAGE_TYPE_2D, format, pixels, {texWidth, texHeight, 1u}, VK_SAMPLER_ADDRESS_MODE_REPEAT, 1, VK_IMAGE_TILING_OPTIMAL, levelCount);
    stbi_image_free(pixels);
}

void textures::fromFile(const VulkanDevice &device, Texture &texture, const std::vector<std::string> &paths, bool flipUv,
                        VkFormat format, uint32_t levels) {

    auto load = [flipUv](std::string_view path, int& width, int& height, int& channel){
        stbi_set_flip_vertically_on_load(flipUv ? 1 : 0);
        stbi_uc* pixels = stbi_load(path.data(), &width, &height, &channel, STBI_rgb_alpha);
        if(!pixels){
            throw std::runtime_error{fmt::format("failed to load texture image {}!", path)};
        }
        return pixels;
    };

    std::vector<void*> data;
    int texWidth, texHeight, texChannels;
    auto itr = paths.begin();
    auto pixels = load(itr->data(), texWidth, texHeight, texChannels);
    data.push_back(pixels);

    std::advance(itr, 1);
    while(itr != end(paths)){
        int width, height;
        pixels = load(itr->data(), width, height, texChannels);
        if(width != texWidth || height != texHeight){
            throw std::runtime_error(fmt::format("{} dimensions: [{}, {}] does not match previously loaded dimensions: [{}, {}]",
                                                 *itr, width, height, texWidth, texHeight));
        }
        data.push_back(pixels);
        texWidth = width;
        texHeight = height;
        std::advance(itr, 1);
    }

    createTextureArray(device, texture, VK_IMAGE_TYPE_2D, format, data, {texWidth, texHeight, 1u}, VK_SAMPLER_ADDRESS_MODE_REPEAT, 1, VK_IMAGE_TILING_OPTIMAL, levels);
    for(auto memory : data){
        stbi_image_free(memory);
    }

}


Texture textures::equirectangularToOctahedralMap(const VulkanDevice& device, const std::string& path, uint32_t size, VkImageLayout finalLayout){
    Texture equiTexture;
    equiTexture.width = equiTexture.height = size;
    textures::hdr(device, equiTexture, path);
    auto octahedralMap = equirectangularToOctahedralMap(device, equiTexture, size, finalLayout);
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    octahedralMap.sampler = device.createSampler(samplerInfo);

    return octahedralMap;
}

void textures::hdr(const VulkanDevice &device, Texture &texture, std::string_view path) {
    // FIXME use openEXR for .exr format
    int texWidth, texHeight, texChannels;
    float* data = stbi_loadf(path.data(), &texWidth, &texHeight, &texChannels, 0);
    texture.width = texWidth;
    texture.height = texHeight;
    if(!data){
        throw std::runtime_error{"failed to load texture image!"};
    }
    int desiredChannels = 4;
    std::vector<float> pixels;
    int size = texWidth * texHeight * texChannels;
    for(auto i = 0; i < size; i+= texChannels){
        float r = data[i];
        float g = data[i + 1];
        float b = data[i + 2];
        pixels.push_back(r);
        pixels.push_back(g);
        pixels.push_back(b);
        pixels.push_back(1);
    }
    stbi_image_free(data);

    create(device, texture, VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, reinterpret_cast<char*>(pixels.data())
           , {texWidth, texHeight, 1u}, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, sizeof(float));
}


void textures::exr(const VulkanDevice &device, Texture &texture, std::string_view path) {
    RgbaInputFile file{path.data()};
    const RgbaChannels channels = file.channels();
    Box2i dw = file.dataWindow();
    int width = dw.max.x - dw.min.x + 1;
    int height = dw.max.y - dw.min.y + 1;
    spdlog::info("width: {}, height: {}\n\n", width, height);
    Array2D<Rgba> pixels;
    pixels.resizeErase (height, width);

    file.setFrameBuffer (&pixels[0][0] - dw.min.x - dw.min.y * width, 1, width);
    file.readPixels (dw.min.y, dw.max.y);

    std::vector<float> data;
    data.reserve(width * height * 4);
        for(int i = 0; i < height; i++){
        auto pixel = pixels[i];
        for(int j = 0; j < width; j++){
            data.push_back(pixel[i].r);
            data.push_back(pixel[i].g);
            data.push_back(pixel[i].b);
            data.push_back(pixel[i].a);
        }
    }
    create(device, texture, VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, reinterpret_cast<char*>(data.data())
            , {width, height, 1u}, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, sizeof(float));
}

void textures::create(const VulkanDevice &device, Texture &texture, VkImageType imageType, VkFormat format, void *data,
                      Dimension3D<uint32_t> dimensions, VkSamplerAddressMode addressMode, uint32_t sizeMultiplier,
                      VkImageTiling tiling, uint32_t levelCount) {

    texture.format = format;
    VkDeviceSize imageSize = dimensions.x * dimensions.y * dimensions.z * nunChannels(format) * byteSize(format);

    VulkanBuffer stagingBuffer = device.createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY, imageSize);
    stagingBuffer.copy(data, imageSize);

    VkImageUsageFlags usageFlags = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |  VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    if(format != VK_FORMAT_R8G8B8A8_SRGB) {
        usageFlags |=  VK_IMAGE_USAGE_STORAGE_BIT;
    }

     VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = imageType;
    imageCreateInfo.format = format;
    imageCreateInfo.extent = { static_cast<uint32_t>(dimensions.x), static_cast<uint32_t>(dimensions.y), dimensions.z};
    imageCreateInfo.mipLevels = levelCount;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = tiling;
    imageCreateInfo.usage = usageFlags;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    auto& commandPool = device.commandPoolFor(*device.findFirstActiveQueue());

    texture.image = device.createImage(imageCreateInfo, VMA_MEMORY_USAGE_GPU_ONLY);
    texture.spec = imageCreateInfo;
    texture.image.size = imageSize;
    texture.width = dimensions.x;
    texture.height = dimensions.y;
    texture.depth = dimensions.z;

    auto subResource = DEFAULT_SUB_RANGE;
    subResource.baseMipLevel = 0;
    subResource.levelCount = levelCount;
    texture.image.transitionLayout(commandPool, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subResource);

    commandPool.oneTimeCommand([&](auto cmdBuffer) {
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;

        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {static_cast<uint32_t>(dimensions.x), static_cast<uint32_t>(dimensions.y), dimensions.z};

        vkCmdCopyBufferToImage(cmdBuffer, stagingBuffer, texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,  &region);
    });

    texture.image.transitionLayout(commandPool, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subResource);

    VkImageSubresourceRange subresourceRange;
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = levelCount;
    subresourceRange.baseArrayLayer = 0;
    subresourceRange.layerCount = 1;

    auto imageViewType = getImageViewType(imageType);
    texture.imageView = texture.image.createView(format, imageViewType, subresourceRange);  // FIXME derive image view type

    if(!texture.sampler.handle) {
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = isIntegral(format) ? VK_FILTER_NEAREST :  VK_FILTER_LINEAR;
        samplerInfo.minFilter = isIntegral(format) ? VK_FILTER_NEAREST :  VK_FILTER_LINEAR;;
        samplerInfo.addressModeU = addressMode;
        samplerInfo.addressModeV = addressMode;
        samplerInfo.addressModeW = addressMode;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.mipmapMode = isIntegral(format) ? VK_SAMPLER_MIPMAP_MODE_NEAREST : VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.maxLod = levelCount;

        texture.sampler = device.createSampler(samplerInfo);
    }
}

void textures::createTextureArray(const VulkanDevice &device, Texture &texture, VkImageType imageType, VkFormat format,
                      const std::vector<void *> &data, Dimension3D<uint32_t> dimensions,
                      VkSamplerAddressMode addressMode, uint32_t sizeMultiplier, VkImageTiling tiling, uint32_t levels) {

    const auto layers = data.size();
    texture.format = format;
    VkDeviceSize imageSize = dimensions.x * dimensions.y * dimensions.z * nunChannels(format) * sizeMultiplier;
    VkDeviceSize totalImageSize = imageSize * layers;

    VulkanBuffer stagingBuffer = device.createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY, totalImageSize);

    for(auto layer = 0; layer < layers; layer++){
        stagingBuffer.copy(data[layer], imageSize, imageSize * layer);
    }

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = imageType;
    imageCreateInfo.format = format;
    imageCreateInfo.extent = { static_cast<uint32_t>(dimensions.x), static_cast<uint32_t>(dimensions.y), dimensions.z};
    imageCreateInfo.mipLevels = levels;
    imageCreateInfo.arrayLayers = layers;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = tiling;
    imageCreateInfo.usage =
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
            ( (format != VK_FORMAT_R8G8B8A8_SRGB) * VK_IMAGE_USAGE_STORAGE_BIT);
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    auto& commandPool = device.commandPoolFor(*device.findFirstActiveQueue());

    texture.image = device.createImage(imageCreateInfo, VMA_MEMORY_USAGE_GPU_ONLY);
    texture.image.size = totalImageSize;
    texture.width = dimensions.x;
    texture.height = dimensions.y;
    texture.depth = dimensions.z;
    texture.layers = layers;
    texture.levels = levels;

    commandPool.oneTimeCommand([&](auto commandBuffer) {
        std::vector<VkBufferImageCopy> regions;
        std::vector<VkImageMemoryBarrier> transferBarriers;
        std::vector<VkImageMemoryBarrier> readBarriers;

        for(auto layer = 0; layer < layers; layer++) {

            VkBufferImageCopy region{};
            region.bufferOffset = imageSize * layer;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;

            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = layer;
            region.imageSubresource.layerCount = 1;
            region.imageOffset = {0, 0, 0};
            region.imageExtent = {static_cast<uint32_t>(dimensions.x), static_cast<uint32_t>(dimensions.y),
                                  dimensions.z};

            regions.push_back(region);

            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = texture.image;
            barrier.subresourceRange = DEFAULT_SUB_RANGE;
            barrier.subresourceRange.baseArrayLayer = layer;
            barrier.srcAccessMask = VK_ACCESS_NONE;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.subresourceRange.levelCount = levels;

            transferBarriers.push_back(barrier);

            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            readBarriers.push_back(barrier);
        }

        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                             nullptr, 0, nullptr, COUNT(transferBarriers), transferBarriers.data());

        vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, COUNT(regions),
                               regions.data());



        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, COUNT(readBarriers), readBarriers.data());

    });

    VkImageSubresourceRange subresourceRange;
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = levels;
    subresourceRange.baseArrayLayer = 0;
    subresourceRange.layerCount = layers;

    auto imageViewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    texture.imageView = texture.image.createView(format, imageViewType, subresourceRange);

    if(!texture.sampler.handle) {
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = addressMode;
        samplerInfo.addressModeV = addressMode;
        samplerInfo.addressModeW = addressMode;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.maxLod = static_cast<float>(levels - 1);

        texture.sampler = device.createSampler(samplerInfo);
    }
}


void textures::create(const VulkanDevice &device, Texture &texture, VkImageType imageType, VkFormat format,
                      Dimension3D<uint32_t> dimensions, VkSamplerAddressMode addressMode, uint32_t sizeMultiplier,
                      VkImageTiling tiling) {

    texture.format = format;
    VkDeviceSize imageSize = dimensions.x * dimensions.y * dimensions.z * nunChannels(format) * byteSize(format);

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = imageType;
    imageCreateInfo.format = format;
    imageCreateInfo.extent = { static_cast<uint32_t>(dimensions.x), static_cast<uint32_t>(dimensions.y), dimensions.z};
    imageCreateInfo.mipLevels = texture.levels;
    imageCreateInfo.arrayLayers = texture.layers;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = tiling;
    imageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    if(isDepthTexture(format)){
        imageCreateInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    }

    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    auto& commandPool = device.commandPoolFor(*device.findFirstActiveQueue());

    VkImageSubresourceRange subresourceRange;
    subresourceRange.aspectMask = isDepthTexture(format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = texture.levels;
    subresourceRange.baseArrayLayer = 0;
    subresourceRange.layerCount = texture.layers;

    texture.image = device.createImage(imageCreateInfo, VMA_MEMORY_USAGE_GPU_ONLY);
    texture.image.size = imageSize;
    texture.width = dimensions.x;
    texture.height = dimensions.y;
    texture.depth = dimensions.z;
    texture.image.transitionLayout(commandPool, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresourceRange);
    texture.spec = imageCreateInfo;


    auto imageViewType = getImageViewType(imageType);
    texture.imageView = texture.image.createView(format, imageViewType, subresourceRange);

    if(!texture.sampler.handle) {
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = addressMode;
        samplerInfo.addressModeV = addressMode;
        samplerInfo.addressModeW = addressMode;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.maxLod = static_cast<float>(texture.levels - 1);

        texture.sampler = device.createSampler(samplerInfo);
    }

}

void textures::createNoTransition(const VulkanDevice &device, Texture &texture, VkImageType imageType, VkFormat format,
                      Dimension3D<uint32_t> dimensions, VkSamplerAddressMode addressMode, VkImageTiling tiling) {

    texture.format = format;
    VkDeviceSize imageSize = dimensions.x * dimensions.y * dimensions.z * nunChannels(format) * byteSize(format);

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = imageType;
    imageCreateInfo.format = format;
    imageCreateInfo.extent = { static_cast<uint32_t>(dimensions.x), static_cast<uint32_t>(dimensions.y), dimensions.z};
    imageCreateInfo.mipLevels = texture.levels;
    imageCreateInfo.arrayLayers = texture.layers;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = tiling;
    imageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    texture.image = device.createImage(imageCreateInfo, VMA_MEMORY_USAGE_GPU_ONLY);
    texture.image.size = imageSize;
    texture.width = dimensions.x;
    texture.height = dimensions.y;
    texture.depth = dimensions.z;
    texture.spec = imageCreateInfo;

    VkImageSubresourceRange subresourceRange;
    subresourceRange.aspectMask = isDepthTexture(format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = texture.levels;
    subresourceRange.baseArrayLayer = 0;
    subresourceRange.layerCount = texture.layers;

    auto imageViewType = getImageViewType(imageType);
    texture.imageView = texture.image.createView(format, imageViewType, subresourceRange);

    if(!texture.sampler.handle) {
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = addressMode;
        samplerInfo.addressModeV = addressMode;
        samplerInfo.addressModeW = addressMode;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.minLod = 0;
        samplerInfo.maxLod = static_cast<float>(texture.levels - 1);

        texture.sampler = device.createSampler(samplerInfo);
    }

}

void textures::createExportable(const VulkanDevice &device, Texture &texture, VkImageType imageType, VkFormat format,
                      Dimension3D<uint32_t> dimensions, VkSamplerAddressMode addressMode, uint32_t sizeMultiplier,
                      VkImageTiling tiling) {

    texture.format = format;
    VkDeviceSize imageSize = dimensions.x * dimensions.y * dimensions.z * nunChannels(format) * sizeMultiplier;

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = imageType;
    imageCreateInfo.format = format;
    imageCreateInfo.extent = { static_cast<uint32_t>(dimensions.x), static_cast<uint32_t>(dimensions.y), dimensions.z};
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = tiling;
    imageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    auto& commandPool = device.commandPoolFor(*device.findFirstActiveQueue());

    texture.image = device.createExportableImage(imageCreateInfo, VMA_MEMORY_USAGE_GPU_ONLY);
    texture.image.size = imageSize;
    texture.width = dimensions.x;
    texture.height = dimensions.y;
    texture.depth = dimensions.z;
    texture.image.transitionLayout(commandPool, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    texture.image.transitionLayout(commandPool, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    texture.image.transitionLayout(commandPool, VK_IMAGE_LAYOUT_GENERAL);

    VkImageSubresourceRange subresourceRange;
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = 1;
    subresourceRange.baseArrayLayer = 0;
    subresourceRange.layerCount = 1;

    auto imageViewType = getImageViewType(imageType);
    texture.imageView = texture.image.createView(format, imageViewType, subresourceRange);  // FIXME derive image view type

    if(!texture.sampler.handle) {
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = addressMode;
        samplerInfo.addressModeV = addressMode;
        samplerInfo.addressModeW = addressMode;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

        texture.sampler = device.createSampler(samplerInfo);
    }

}

void textures::allocate(const VulkanDevice& device, Texture &texture, VkImageType imageType, VkFormat format, VkDeviceSize size,
                        Dimension3D<uint32_t> dimensions, VkImageTiling tiling) {

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = imageType;
    imageCreateInfo.format = format;
    imageCreateInfo.extent = { static_cast<uint32_t>(dimensions.x), static_cast<uint32_t>(dimensions.y), dimensions.z};
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = tiling;
    imageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    auto& commandPool = device.commandPoolFor(*device.queueFamilyIndex.graphics);

    texture.image = device.createImage(imageCreateInfo, VMA_MEMORY_USAGE_GPU_ONLY);
    texture.image.size = size;
    texture.width = dimensions.x;
    texture.height = dimensions.y;
    texture.depth = dimensions.z;
    texture.image.transitionLayout(commandPool, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkImageSubresourceRange subresourceRange;
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = 1;
    subresourceRange.baseArrayLayer = 0;
    subresourceRange.layerCount = 1;

    auto imageViewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewType = imageType == VK_IMAGE_TYPE_3D ? VK_IMAGE_VIEW_TYPE_3D : imageViewType;
    texture.imageView = texture.image.createView(format, imageViewType, subresourceRange);

}


void textures::checkerboard(const VulkanDevice &device, Texture &texture, const glm::vec3 &colorA, const glm::vec3 &colorB) {
    texture.width = texture.height = 256;
    auto data = new unsigned char[256 * 256 * 4];
    checkerboard(data, {256, 256 });
    create(device, texture, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, data, {256, 256, 1}, VK_SAMPLER_ADDRESS_MODE_REPEAT);
    delete[] data;
}

void textures::normalMap(const VulkanDevice &device, Texture &texture, const Dimension2D<uint32_t>& dimensions) {
    color(device, texture, glm::vec3{0.5, 0.5, 1}, dimensions);
}



void textures::color(const VulkanDevice &device, Texture &texture, const glm::vec3 &color, const Dimension2D<uint32_t>& dimensions) {
    auto data = new unsigned char[dimensions.x * dimensions.y * 4];
    textures::color(data, dimensions, color);
    create(device, texture, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, data, Dimension3D<uint32_t>{dimensions, 1u}, VK_SAMPLER_ADDRESS_MODE_REPEAT);
    delete[] data;
}


void textures::checkerboard(unsigned char* data, const Dimension2D<uint32_t>& dimensions, const glm::vec3& colorA, const glm::vec3& colorB){
    for(int i = 0; i < 256; i++){
        for(int j = 0; j < 256; j++){
            auto color = (((i / 8) % 2) && ((j / 8) % 2)) || (!((i / 8) % 2) && !((j / 8) % 2)) ? colorB : colorA;
            auto idx = (i * 256 + j) * 4;
            data[idx + 0]  = static_cast<unsigned char>(color.r * 255);
            data[idx + 1]  = static_cast<unsigned char>(color.b * 255);
            data[idx + 2]  = static_cast<unsigned char>(color.g * 255);
            data[idx + 3] = 255;
        }
    }
}

void textures::checkerboard1(unsigned char* data, const Dimension2D<uint32_t>& dimensions, const glm::vec3& colorA, const glm::vec3& colorB, float repeat){
    for(int i = 0; i < dimensions.y; i++){
        for(int j = 0; j < dimensions.x; j++){
            float x = glm::floor(static_cast<float>(j)/static_cast<float>(dimensions.x) * repeat);
            float y = glm::floor(static_cast<float>(i)/static_cast<float>(dimensions.y) * repeat);

            float t = glm::step(glm::mod(x + y, 2.f), 0.f);
            auto color = glm::mix(colorA, colorB, t);
            auto idx = (i * dimensions.x + j) * 4;
            data[idx + 0]  = static_cast<unsigned char>(color.r * 255);
            data[idx + 1]  = static_cast<unsigned char>(color.b * 255);
            data[idx + 2]  = static_cast<unsigned char>(color.g * 255);
            data[idx + 3] = 255;
        }
    }
}

void textures::color(unsigned char* data, const Dimension2D<uint32_t>& dimensions, const glm::vec3& color){
    for(auto i = 0; i < dimensions.y; i++){
        for(auto j = 0; j < dimensions.x; j++){
            auto idx = (i * dimensions.x + j) * 4;
            data[idx + 0]  = static_cast<unsigned char>(color.r * 255);
            data[idx + 1]  = static_cast<unsigned char>(color.g * 255);
            data[idx + 2]  = static_cast<unsigned char>(color.b * 255);
            data[idx + 3] = 255;
        }
    }
}

void textures::normalMap(unsigned char* data, const Dimension2D<uint32_t>& dimensions){
    color(data, dimensions, glm::vec3{0.5, 0.5, 1});
}

void textures::uv(const VulkanDevice &device, Texture &texture, const Dimension2D<uint32_t> &dimensions) {
    auto data = uvData(dimensions);
    create(device, texture, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8_UNORM, data.data(), Dimension3D<uint32_t>{dimensions, 1u}, VK_SAMPLER_ADDRESS_MODE_REPEAT);
}

std::vector<unsigned char> textures::uvData(const Dimension2D<uint32_t> &dimensions) {
    std::vector<unsigned char> data(dimensions.x * dimensions.y * 2);
    glm::vec2 size{dimensions};
    for(int i = 0; i < dimensions.y; i++){
        for(int j = 0; j < dimensions.x; j++){
            glm::vec2 coord{i, j};
            auto uv = coord/size;
            auto id = (i * dimensions.x + j) * 2;
            data[id + 0] = static_cast<unsigned char>(uv.x * 255);
            data[id + 1] = static_cast<unsigned char>(uv.y * 255);
        }
    }
    return data;
}

void textures::transfer(VkCommandBuffer commandBuffer, const VulkanBuffer& srcBuffer, VulkanImage& dstImage,
                        Dimension2D<uint32_t> dimension2D, VkImageLayout sourceLayout) {

    VkImageSubresourceRange subresourceRange;
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = 1;
    subresourceRange.baseArrayLayer = 0;
    subresourceRange.layerCount = 1;

    dstImage.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresourceRange
                              , VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT
                              , VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {static_cast<uint32_t>(dimension2D.x), static_cast<uint32_t>(dimension2D.y), 1};

    vkCmdCopyBufferToImage(commandBuffer, srcBuffer, dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                           , 1,&region);

    dstImage.transitionLayout(commandBuffer, sourceLayout, subresourceRange, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
}

void textures::copy(VkCommandBuffer commandBuffer, Texture &srcTexture, Texture &dstTexture) {

    VkImageSubresourceRange subresourceRange;
    subresourceRange.aspectMask = srcTexture.aspectMask;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = 1;
    subresourceRange.baseArrayLayer = 0;
    subresourceRange.layerCount = 1;

    auto dstOldLayout = dstTexture.image.currentLayout;
    if(dstOldLayout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL){
        dstTexture.image.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresourceRange
            , 0, VK_ACCESS_TRANSFER_WRITE_BIT
            , VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_TRANSFER_BIT);
    }

    auto srcOldLayout = srcTexture.image.currentLayout;
    if(srcOldLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL){
        srcTexture.image.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, subresourceRange
                , 0, VK_ACCESS_TRANSFER_READ_BIT
                , VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_TRANSFER_BIT);
    }

    VkImageSubresourceLayers imageSubresource{srcTexture.aspectMask, 0, 0, 1};

    VkImageCopy region{};
    region.srcSubresource = imageSubresource;
    region.srcOffset = {0, 0, 0};
    region.dstSubresource = imageSubresource;
    region.dstOffset = {0, 0, 0};
    region.extent = {srcTexture.width, srcTexture.height, 1};

    vkCmdCopyImage(commandBuffer, srcTexture.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstTexture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    if(dstOldLayout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL){
        dstTexture.image.transitionLayout(commandBuffer, dstOldLayout, subresourceRange
                ,  VK_ACCESS_TRANSFER_WRITE_BIT, 0
                ,  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_NONE);
    }

    if(srcOldLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL){
        srcTexture.image.transitionLayout(commandBuffer, srcOldLayout, subresourceRange
                ,  VK_ACCESS_TRANSFER_WRITE_BIT, 0
                ,  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_NONE);
    }
}

void textures::copy(VkCommandBuffer commandBuffer, Texture& srcTexture, VulkanBuffer& dstBuffer, Dimension2D<uint32_t> dimension2D){
    VkImageSubresourceRange subresourceRange;
    subresourceRange.aspectMask = srcTexture.aspectMask;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = 1;
    subresourceRange.baseArrayLayer = 0;
    subresourceRange.layerCount = 1;

    auto srcOldLayout = srcTexture.image.currentLayout;
    if(srcOldLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL){
        srcTexture.image.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, subresourceRange
                , 0, VK_ACCESS_TRANSFER_READ_BIT
                , VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_TRANSFER_BIT);
    }

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {static_cast<uint32_t>(dimension2D.x), static_cast<uint32_t>(dimension2D.y), 1};

    vkCmdCopyImageToBuffer(commandBuffer, srcTexture.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstBuffer, 1, &region);

    if(srcOldLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL){
        srcTexture.image.transitionLayout(commandBuffer, srcOldLayout, subresourceRange
                ,  VK_ACCESS_TRANSFER_WRITE_BIT, 0
                ,  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_NONE);
    }
}

void textures::generate(const VulkanDevice &device, Texture &texture, uint32_t width, uint32_t height,
                       ColorGen&& generator, VkFormat format) {
    std::vector<unsigned char> canvas(width * height * 4);

    for(auto i = 0; i < height; i++){
        for(auto j = 0; j < width; j++){
            auto idx = (i * width + j) * 4;
            auto color = generator(j, i, width, height);
            canvas[idx + 0]  = static_cast<unsigned char>(color.r * 255);
            canvas[idx + 1]  = static_cast<unsigned char>(color.g * 255);
            canvas[idx + 2]  = static_cast<unsigned char>(color.b * 255);
            canvas[idx + 3] = 255;
        }
    }
    create(device, texture, VK_IMAGE_TYPE_2D, format, canvas.data(), {width, height, 1u});
}

void textures::createDistribution(const VulkanDevice &device, Texture &source, Distribution2DTexture& distribution, float scale) {
    Texture blitsTexture;
    auto width = source.width;
    auto height = source.height;
    textures::create(device, blitsTexture, VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, {width * scale, height * scale, 1}, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, sizeof(float));

    device.graphicsCommandPool().oneTimeCommand([&](auto commandBuffer){
        VkImageSubresourceLayers srcSubResource{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        VkImageSubresourceLayers dstSubResource{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};

        VkImageBlit blit{};
        blit.srcSubresource = srcSubResource;
        blit.dstSubresource = dstSubResource;
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {static_cast<int32_t>(width), static_cast<int32_t>(height), 1};
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {static_cast<int32_t>(width * scale), static_cast<int32_t>(height * scale), 1};

        blitsTexture.image.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, DEFAULT_SUB_RANGE,  VK_ACCESS_NONE, VK_ACCESS_NONE, VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_TRANSFER_BIT);
        source.image.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, DEFAULT_SUB_RANGE, VK_ACCESS_NONE, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_TRANSFER_BIT);

        vkCmdBlitImage(commandBuffer, source.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, blitsTexture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

        blitsTexture.image.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, DEFAULT_SUB_RANGE,  VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        source.image.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, DEFAULT_SUB_RANGE, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    });

    width *= scale;
    height *= scale;
    auto size = blitsTexture.image.size;
    VulkanBuffer stageBuffer = device.createStagingBuffer(size);
    VulkanBuffer luminanceBuffer = device.createStagingBuffer(width * height * sizeof(float));
    VulkanBuffer luminanceCdfBuffer = device.createStagingBuffer(width * height * sizeof(float));
    VulkanBuffer pMarginalBuffer = device.createStagingBuffer( height * sizeof(float));
    VulkanBuffer pMarginalCdfBuffer = device.createStagingBuffer( height * sizeof(float));

    device.graphicsCommandPool().oneTimeCommand([&](auto commandBuffer){
        textures::copy(commandBuffer, blitsTexture, stageBuffer, { width, height});
    });

    auto lenvMap = reinterpret_cast<glm::vec4*>(stageBuffer.map());
    auto lumPtr = reinterpret_cast<float*>(luminanceBuffer.map());
    auto lumCdfPtr = reinterpret_cast<float*>(luminanceCdfBuffer.map());
    auto pMarginalPtr = reinterpret_cast<float*>(pMarginalBuffer.map());
    auto pMarginalCdfPtr = reinterpret_cast<float*>(pMarginalCdfBuffer.map());

    for(int v = 0; v < height; v++){
        for(int u = 0; u < width; u++){
            auto index = v * width + u;
            auto sinTheta = glm::sin(glm::pi<float>() * float(v + .5)/float(height));
            auto rgb = lenvMap[index].rgb();
            lumPtr[index] = color::luminance(rgb) * sinTheta;
        }
    }

    auto distribution2D = sampling::Distribution2D::create(lumPtr, width, height);

    for(int v = 0; v < height; v++){
        for(int u = 0; u < width; u++){
            auto& distribution1D = distribution2D.pConditionalV[v];
            auto index = v * width + u;
            lumPtr[index] = distribution1D.func[u];
            lumCdfPtr[index] = distribution1D.cdf[u];
        }
        pMarginalPtr[v] = distribution2D.pMarginal.func[v];
        pMarginalCdfPtr[v] = distribution2D.pMarginal.cdf[v];
    }
    distribution.pMarginal.funcIntegral = distribution2D.pMarginal.funcIntegral;


    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

    distribution.sampler = device.createSampler(samplerInfo);

    textures::create(device, distribution.pConditionalVFunc, VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, lumPtr, {width, height, 1}, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, sizeof(float));
    textures::create(device, distribution.pConditionalVCdf, VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, lumCdfPtr, {width, height, 1}, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, sizeof(float));
    textures::create(device, distribution.pMarginal.func, VK_IMAGE_TYPE_1D, VK_FORMAT_R32_SFLOAT, pMarginalPtr, {height, 1, 1}, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, sizeof(float));
    textures::create(device, distribution.pMarginal.cdf, VK_IMAGE_TYPE_1D, VK_FORMAT_R32_SFLOAT, pMarginalCdfPtr, {height, 1, 1}, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, sizeof(float));

    luminanceBuffer.unmap();
    luminanceCdfBuffer.unmap();
    pMarginalBuffer.unmap();
    pMarginalCdfBuffer.unmap();
    stageBuffer.unmap();
}


void saveAsPing(VkFormat format, const std::string& path, int width, int height, const VulkanBuffer& data){
    throw std::runtime_error{"ping save not yet implemented!"};
}

void saveAsBmp(VkFormat format, const std::string& path, int width, int height, const VulkanBuffer& data){
    throw std::runtime_error{"bmp save not yet implemented!"};
}

void saveAsJpg(VkFormat format, const std::string& path, int width, int height, const VulkanBuffer& data){
    int comp = to<int>(nunChannels(format));
    stbi_write_jpg(path.c_str(), width, height, comp, data.map(), 90);
}

void saveAsHdr(VkFormat format, const std::string& path, int width, int height, const VulkanBuffer& data){
    int comp = data.sizeAs<float>()/(width * height);
    ASSERT(comp >= 1 && comp <= 4);
    if(comp == 1){
        ASSERT(format == VK_FORMAT_R32_SFLOAT || format == VK_FORMAT_D32_SFLOAT)
    }
    if(comp == 2){
        ASSERT(format == VK_FORMAT_R32G32_SFLOAT);
    }
    if(comp == 3){
        ASSERT(format == VK_FORMAT_R32G32B32_SFLOAT);
    }
    if(comp == 4){
        ASSERT(format == VK_FORMAT_R32G32B32A32_SFLOAT);
    }

    stbi_write_hdr(path.c_str(), width, height, comp, reinterpret_cast<const float*>(data.map()));
    data.unmap();
}

void saveAsExr(VkFormat format, const std::string& path, int width, int height, const VulkanBuffer& data){
    throw std::runtime_error{"exr save not yet implemented!"};
}

void textures::save(const VulkanDevice& device, const std::string& path, uint32_t width, uint32_t height, VkFormat format, const VulkanImage& image, FileFormat fileFormat){
    VkDeviceSize imageSize = width * height * nunChannels(format) * byteSize(format);
    VulkanBuffer buffer = device.createStagingBuffer(imageSize);

    device.graphicsCommandPool().oneTimeCommand([&](auto cb){
        VkBufferImageCopy region{
                0, 0, 0,
                {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1},
                {0, 0, 0},
                {width, height, 1}
        };
        vkCmdCopyImageToBuffer(cb, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);
    });
    if(format == VK_FORMAT_R32_SFLOAT || format == VK_FORMAT_D32_SFLOAT){
        auto temp = buffer;
        buffer = device.createBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY, temp.size * 4);
        auto src = reinterpret_cast<float*>(temp.map());
        auto dst = reinterpret_cast<float*>(buffer.map());
        for(auto i = 0; i < temp.sizeAs<float>(); i++){
            spdlog::info(src[i]);
            dst[i * 4 + 0] = src[i];
            dst[i * 4 + 1] = src[i];
            dst[i * 4 + 2] = src[i];
            dst[i * 4 + 3] = src[i];
        }
        format = VK_FORMAT_R32G32B32A32_SFLOAT;
    }
    save(device, buffer, format, fileFormat, path, width, height);
}

void textures::save(const VulkanDevice &device, Texture &texture,  FileFormat fileFormat, const std::string& path) {
    ASSERT(texture.format != VK_FORMAT_UNDEFINED)
    int width = texture.width;
    int height = texture.height;
    VulkanBuffer buffer = device.createStagingBuffer(texture.image.size);

    device.graphicsCommandPool().oneTimeCommand([&](auto cb){
        VkBufferImageCopy region{
            0, 0, 0,
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            {0, 0, 0},
            {texture.width, texture.height, 1}
        };
        vkCmdCopyImageToBuffer(cb, texture.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);
    });

    save(device, buffer, texture.format, fileFormat, path, width, height);
}

void textures::save(const VulkanDevice& device, const VulkanBuffer& buffer, VkFormat imageFormat, FileFormat format,
                    const std::string& path, int width, int height){
    switch(format) {
        case FileFormat::PNG : {
            saveAsPing(imageFormat, path, width, height, buffer);
            break;
            case FileFormat::BMP:
                saveAsBmp(imageFormat, path, width, height, buffer);
            break;
            case FileFormat::JPG:
                saveAsJpg(imageFormat, path, width, height, buffer);
            break;
            case FileFormat::HDR:
                saveAsHdr(imageFormat, path, width, height, buffer);
            break;
            case FileFormat::EXR:
                saveAsExr(imageFormat, path, width, height, buffer);
            break;
            default:
                throw std::runtime_error{"unsupported file format"};

        }
    }
}

void textures::generateLOD(const VulkanDevice &device, Texture &texture, uint32_t levels, uint32_t layers) {
    generateLOD(device, texture.image, texture.width, texture.height, levels, layers);
}

void textures::generateLOD(const VulkanDevice& device, VulkanImage& image, uint32_t width, uint32_t height, uint32_t levels, uint32_t layers) {
    if(levels <= 1) {
        spdlog::warn("texture LOD requested with  mipLevels set to {}", levels);
        return;
    }

    device.commandPoolFor(*device.findFirstActiveQueue()).oneTimeCommand([&](auto commandBuffer) {
        generateLOD(commandBuffer, image, width, height, levels, layers);
    });
}

void textures::generateLOD(VkCommandBuffer commandBuffer, VulkanImage &image, uint32_t width, uint32_t height, uint32_t levels, uint32_t layers) {
    auto w = static_cast<int32_t>(width);
    auto h = static_cast<int32_t>(height);
    VkImageBlit blit{};
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.baseArrayLayer = 0;
    blit.srcSubresource.layerCount = 1;
    blit.srcOffsets[0] = {0, 0, 0};

    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.dstSubresource.baseArrayLayer = 0;
    blit.dstSubresource.layerCount = 1;
    blit.dstOffsets[0] = {0, 0, 0};
    blit.dstOffsets[1] = {w/2, h/2, 1};

    VkImageMemoryBarrier barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;


    for(auto layer = 0; layer < layers; ++layer) {
        blit.srcOffsets[1] = {w, h, 1};
        blit.dstOffsets[1] = {w/2, h/2, 1};

        barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = layer;
        barrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 0, 0, 1, &barrier);

        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.subresourceRange.baseMipLevel = 1;
        barrier.subresourceRange.levelCount = levels - 1;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 0, 0, 1, &barrier);


        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

        for (auto level = 1u; level < levels; ++level) {
            blit.srcSubresource.mipLevel = level - 1;
            blit.dstSubresource.mipLevel = level;
            blit.srcSubresource.baseArrayLayer = layer;
            blit.dstSubresource.baseArrayLayer = layer;

            vkCmdBlitImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

            barrier.subresourceRange.baseMipLevel = level;
            barrier.subresourceRange.baseArrayLayer = layer;
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,0, 0, 0, 0, 1, &barrier);

            blit.srcOffsets[1].x >>= 1;
            blit.srcOffsets[1].y >>= 1;

            blit.dstOffsets[1].x = std::max(blit.dstOffsets[1].x >> 1, 1);
            blit.dstOffsets[1].y = std::max(blit.dstOffsets[1].y >> 1, 1);
        }
        blit.srcOffsets[1].x >>= 1;
        blit.srcOffsets[1].y >>= 1;
    }

    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = levels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = layers;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, 0, 0, 0, 1, &barrier);
}