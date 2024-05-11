#pragma once

#include "vulkan_context.hpp"
#include "Statistics.hpp"

#include <string>

class BasePerformance {
public:

    using Report = std::map<std::string, stats::Statistics<float>>;

    explicit BasePerformance(VulkanContext& context)
    : _context{context}
    {}

    virtual ~BasePerformance() = default;

    virtual std::string report() = 0;

    void execute(auto&& func){
        _context.device.computeCommandPool().oneTimeCommand(func);
    }

protected:
    VulkanContext& _context;
    static constexpr VkBufferUsageFlags usage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    static constexpr int warmUpRuns = 1000;
    static constexpr int runs = 10000;
};