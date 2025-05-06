#pragma once

#include "ThreadPool.hpp"
#include "oclHelper.h"
#include "Vertex.h"
#include "VulkanDevice.h"
#include "VulkanCommandBuffer.h"
#include "VHACD.h"
#include <vector>
#include <future>

struct ConvexHullPoint{
    alignas(16) glm::vec3 position;
    alignas(16) glm::vec3 normal;
};

struct ConvexHulls{
    std::vector<VulkanBuffer> vertices;
    std::vector<VulkanBuffer> indices;
    std::vector<glm::vec4> colors;
    std::vector<float> points;
    std::vector<uint32_t> triangles;
};


class Callback final : public VHACD::IVHACD::IUserCallback{
public:
    ~Callback() final = default;

    void Update(const double overallProgress,
                                const double stageProgress,
                                const char* const stage,
                                const char* operation) final;
};

class LoggingAdaptor final : public VHACD::IVHACD::IUserLogger{
public:
    ~LoggingAdaptor() final = default;

    void Log(const char *const msg) final;
};

class ConvexHullBuilder{
public:
    ConvexHullBuilder() = default;

    ConvexHullBuilder(VulkanDevice* device, bool async = true);

    ConvexHullBuilder& setData(const VulkanBuffer& vertices, const VulkanBuffer& sourceIndexBuffer);

    ConvexHullBuilder& setData(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices);

    ConvexHullBuilder& setData(const std::vector<float>& points, const std::vector<uint32_t>& indices);

    ConvexHullBuilder& setParams(const VHACD::IVHACD::Parameters& params);

    ConvexHullBuilder& setCallBack(Callback& callback);

    ConvexHullBuilder& maxNumVerticesPerCH(int value);

    std::future<ConvexHulls> build();

protected:
    void createCommandPool();
private:
    ConvexHulls m_convexHulls;
    OCLHelper m_oclHelper;
    VHACD::IVHACD* m_interfaceVHACD{nullptr};
    LoggingAdaptor m_loggerVHACD{};
    VHACD::IVHACD::Parameters m_params{};
    VulkanDevice* m_device{nullptr};
    VulkanCommandPool m_commandPool;
    Callback n_defaultCallback{};
};