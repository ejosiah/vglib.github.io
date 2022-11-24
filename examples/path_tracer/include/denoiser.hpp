#pragma once

#include "VulkanDevice.h"
#include "VulkanBuffer.h"
#include "Texture.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table.h>
#include <optix_stubs.h>
#include <optix_types.h>
#include <driver_types.h>
#include <spdlog/spdlog.h>
#include <iostream>
#include <sstream>

#define OPTIX_CHECK(call)                                                                                              \
  do                                                                                                                   \
  {                                                                                                                    \
    OptixResult res = call;                                                                                            \
    if(res != OPTIX_SUCCESS)                                                                                           \
    {                                                                                                                  \
      std::stringstream ss;                                                                                            \
      ss << "Optix call (" << #call << " ) failed with code " << res << " (" __FILE__ << ":" << __LINE__ << ")\n";     \
      std::cerr << ss.str().c_str() << std::endl;                                                                      \
      throw std::runtime_error(ss.str().c_str());                                                                      \
    }                                                                                                                  \
  } while(false)

#define CUDA_CHECK(call)                                                                                               \
  do                                                                                                                   \
  {                                                                                                                    \
    cudaError_t error = call;                                                                                          \
    if(error != cudaSuccess)                                                                                           \
    {                                                                                                                  \
      std::stringstream ss;                                                                                            \
      ss << "CUDA call (" << #call << " ) failed with code " << error << " (" __FILE__ << ":" << __LINE__ << ")\n";    \
      throw std::runtime_error(ss.str().c_str());                                                                      \
    }                                                                                                                  \
  } while(false)

class DenoiserOptix{
public:
    DenoiserOptix() = default;
    ~DenoiserOptix();

    void setup(const VulkanDevice& device, uint32_t queueIndex);
//    bool initOptiX(OptixDenoiserInputKind inputKind, OptixPixelFormat pixelFormat, bool hdr);

    void denoiseImageBuffer(uint64_t& fenceValue);
    void destroy();
    bool uiSetup();

    void allocateBuffers(const VkExtent2D& imgSize);
    void bufferToImage(const VkCommandBuffer& cmdBuf, Texture imgOut);
    void imageToBuffer(const VkCommandBuffer& cmdBuf, const std::vector<Texture>& imgIn);
    void bufferToBuffer(const VkCommandBuffer& cmdBuf, const std::vector<VulkanBuffer>& bufIn);

private:
    OptixDenoiser        m_denoiser{nullptr};
    OptixDenoiserOptions m_dOptions{};
    OptixDenoiserSizes   m_dSizes{};
    CUdeviceptr          m_dState{0};
    CUdeviceptr          m_dScratch{0};
    CUdeviceptr          m_dIntensity{0};
    CUdeviceptr          m_dMinRGB{0};
    CUcontext            m_cudaContext{nullptr};
};