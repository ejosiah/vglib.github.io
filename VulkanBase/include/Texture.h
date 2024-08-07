#pragma once

#include "common.h"
#include "VulkanImage.h"
#include "VulkanDevice.h"
#include "VulkanBuffer.h"
#include  <stb_image.h>
#include <atomic>

enum class FileFormat {
    PNG, BMP, TGA, JPG, HDR, EXR
};

struct Texture{
    VulkanImage image;
    VulkanImageView imageView;
    VulkanSampler sampler;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspectMask{VK_IMAGE_ASPECT_COLOR_BIT};
    VkImageCreateInfo spec{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    uint32_t width{0};
    uint32_t height{0};
    uint32_t depth{1};
    uint32_t layers = 1;
    uint32_t levels = 1;
    uint32_t bindingId = std::numeric_limits<uint32_t>::max();
    bool lod{};
    bool flipped{};
    std::string path;
};

struct Distribution1DTexture {
    Texture func;
    Texture cdf;
    float funcIntegral;
};

struct Distribution2DTexture {
    Texture pConditionalVFunc;
    Texture pConditionalVCdf;
    Distribution1DTexture pMarginal;
    VulkanSampler sampler;
};

struct stbi_image_deleter{
    void operator()(stbi_uc* pixels){
        stbi_image_free(pixels);
    }
};

struct RawImage{
    std::unique_ptr<stbi_uc, stbi_image_deleter> data{nullptr};
    uint32_t width{0};
    uint32_t height{0};
    int numChannels{0};
};

template<typename T>
using Dimension3D = glm::vec<3, T, glm::defaultp>;
using iDimension3D = Dimension3D<int32_t>;
using uDimension3D = Dimension3D<uint32_t>;

template<typename T>
using Dimension2D = glm::vec<2, T, glm::defaultp>;
using iDimension2D = Dimension2D<int32_t>;
using uDimension2D = Dimension2D<uint32_t>;

namespace textures{

    using ColorGen = std::function<glm::vec3(int, int, float , float)>;

    VkDeviceSize byteSize(VkFormat format);

    void create(const VulkanDevice& device, Texture& texture, VkImageType imageType, VkFormat format, void* data
                , Dimension3D<uint32_t> dimensions, VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT
                , uint32_t sizeMultiplier = 1, VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL, uint32_t levelCount = 1);

    void createTextureArray(const VulkanDevice& device, Texture& texture, VkImageType imageType, VkFormat format, const std::vector<void*>& data
                , Dimension3D<uint32_t> dimensions, VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT
                , uint32_t sizeMultiplier = 1, VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL, uint32_t levels = 1);

    void create(const VulkanDevice& device, Texture& texture, VkImageType imageType, VkFormat format
            , Dimension3D<uint32_t> dimensions, VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT
            , uint32_t sizeMultiplier = 1, VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL);

    void createNoTransition(const VulkanDevice& device, Texture& texture, VkImageType imageType, VkFormat format
            , Dimension3D<uint32_t> dimensions, VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT
            , VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL);

    void createExportable(const VulkanDevice& device, Texture& texture, VkImageType imageType, VkFormat format
            , Dimension3D<uint32_t> dimensions, VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT
            , uint32_t sizeMultiplier = 1, VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL);

    void allocate(const VulkanDevice& device, Texture& texture, VkImageType imageType, VkFormat format, VkDeviceSize size
                  , Dimension3D<uint32_t> dimensions, VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL);


    RawImage loadImage(std::string_view path, bool flipUv = false);

    void fromFile(const VulkanDevice& device, Texture& texture, std::string_view path, bool flipUv = false, VkFormat format = VK_FORMAT_R8G8B8A8_UNORM, uint32_t levelCount = 1);

    void fromFile(const VulkanDevice& device, Texture& texture, const std::vector<std::string>& paths, bool flipUv = false, VkFormat format = VK_FORMAT_R8G8B8A8_UNORM, uint32_t levelCount = 1);

    void generate(const VulkanDevice& device, Texture& texture, uint32_t width, uint32_t height, ColorGen&& generator, VkFormat format = VK_FORMAT_R8G8B8A8_UNORM);

    void hdr(const VulkanDevice& device, Texture& texture, std::string_view path);

    void exr(const VulkanDevice& device, Texture& texture, std::string_view path);

    void checkerboard(const VulkanDevice& device, Texture& texture, const glm::vec3& colorA = glm::vec3(1), const glm::vec3& colorB = glm::vec3(0));

    void normalMap(const VulkanDevice& device, Texture& texture, const Dimension2D<uint32_t>& dimensions);

    void color(const VulkanDevice& device, Texture& texture, const glm::vec3& color, const Dimension2D<uint32_t>& dimensions);

    void uv(const VulkanDevice& device, Texture& texture, const Dimension2D<uint32_t>& dimensions);

    std::vector<unsigned char> uvData(const Dimension2D<uint32_t>& dimensions);

    void checkerboard(unsigned char* data, const Dimension2D<uint32_t>& dimensions, const glm::vec3& colorA = glm::vec3(1), const glm::vec3& colorB = glm::vec3(0));

    void checkerboard1(unsigned char* data, const Dimension2D<uint32_t>& dimensions, const glm::vec3& colorA = glm::vec3(1), const glm::vec3& colorB = glm::vec3(0), float repeat = 8);

    void color(unsigned char* data, const Dimension2D<uint32_t>& dimensions, const glm::vec3& color);

    void normalMap(unsigned char* data, const Dimension2D<uint32_t>& dimensions);

    Texture equirectangularToOctahedralMap(const VulkanDevice& device, const std::string& path, uint32_t size, VkImageLayout finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    Texture equirectangularToOctahedralMap(const VulkanDevice& device, const Texture& equirectangularTexture, uint32_t size, VkImageLayout finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    Texture brdf_lut(const VulkanDevice& device);

    void transfer(VkCommandBuffer commandBuffer, const VulkanBuffer& srcBuffer, VulkanImage& dstImage, Dimension2D<uint32_t> dimension2D, VkImageLayout sourceLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    void copy(VkCommandBuffer commandBuffer, Texture& srcTexture, Texture& dstTexture);

    void copy(VkCommandBuffer commandBuffer, Texture& srcTexture, VulkanBuffer& dstBuffer, Dimension2D<uint32_t> dimension2D);

    void ibl(const VulkanDevice& device, const Texture& envMap, Texture& irradianceMap, Texture& specularMap, VkImageLayout finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    Texture distanceMap(const VulkanDevice& device, Texture& source, int depth, bool invert = true);

    void createDistribution(const VulkanDevice& device, Texture& source, Distribution2DTexture& distribution, float scale = 0.25f);

    void save(const VulkanDevice& device, Texture& texture, FileFormat format, const std::string& path);

    void save(const VulkanDevice& device, const VulkanBuffer& buffer, VkFormat imageFormat, FileFormat format, const std::string& path, int width, int height);

    void save(const VulkanDevice& device, const std::string& path, uint32_t width, uint32_t height, VkFormat format, const VulkanImage& image, FileFormat fileFormat = FileFormat::HDR);

    void generateLOD(const VulkanDevice& device, Texture& texture, uint32_t levels, uint32_t layers = 1);

    void generateLOD(const VulkanDevice& device, VulkanImage& image, uint32_t width, uint32_t height, uint32_t levels, uint32_t layers = 1);

    void generateLOD(VkCommandBuffer commandBuffer, VulkanImage& image, uint32_t width, uint32_t height, uint32_t levels, uint32_t layers = 1);
}