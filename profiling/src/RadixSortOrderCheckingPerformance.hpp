#pragma once

#include "RadixSortPerformanceBase.hpp"
#include <spdlog/spdlog.h>
#include <fmt/format.h>

#include <random>
#include <map>
#include <vector>


class RadixSortOrderCheckingPerformance : public RadixSortPerformanceBase {

public:
    RadixSortOrderCheckingPerformance(VulkanContext& context)
    : RadixSortPerformanceBase(context, false)
    {
        _profiler = Profiler{ &context.device };
        _profiler.addQuery("radix_sort_no_order_checking");
        _profiler.addQuery("radix_sort_order_checking");
    }

    ~RadixSortOrderCheckingPerformance() override = default;


    std::map<size_t, Report> SortWithNoOrderChecking() {
        std::array<size_t, 6> numItems{ 500000, 3000000, 6000000, 9000000, 12000000, 15000000 };
        auto maxNumItems = numItems.back();
        VulkanBuffer stagingBuffer = _context.device.createStagingBuffer(maxNumItems * sizeof(uint32_t));
        VulkanBuffer deviceBuffer = _context.device.createBuffer(usage, VMA_MEMORY_USAGE_GPU_ONLY, maxNumItems * sizeof(uint32_t));
        VulkanBuffer resultBuffer = _context.device.createBuffer(usage, VMA_MEMORY_USAGE_GPU_TO_CPU, sizeof(uint32_t));

        std::map<size_t, Report> reports;

        std::span<uint32_t> items = { reinterpret_cast<uint32_t*>(stagingBuffer.map()), maxNumItems };
        std::iota(items.begin(), items.end(), 0);

        for(int i = 0; i < numItems.size(); i++) {
            for(int j = 0; j < runs; j++) {
                execute([&](auto commandBuffer) {
                    VkDeviceSize size = numItems[i] * sizeof(uint32_t);

                    VkBufferCopy region{0, 0, size};
                    vkCmdCopyBuffer(commandBuffer, stagingBuffer, deviceBuffer, 1, &region);

                    _profiler.profile("radix_sort_no_order_checking", commandBuffer, [&]{
                        _sort(commandBuffer, { &deviceBuffer, 0, size });
                    });

                    Barrier::computeWriteToRead(commandBuffer, deviceBuffer);

                });
                _profiler.commit();
                if(j % (runs/10) == 0){
                    spdlog::info("order checking performance test run {} {:.2f} % complete", (i+1), ((j * 100/float(runs))));
                }
            }

            reports.insert(std::make_pair(numItems[i], _profiler.queryStats()));
            _profiler.clearRunTimes();
        }

        return reports;
    }

    std::string report() final {
        warmup();

        return "";
    }
    

private:
    Profiler _profiler;

};