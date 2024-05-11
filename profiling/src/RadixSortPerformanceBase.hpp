#pragma once

#include "BasePerformance.hpp"
#include "Sort.hpp"
#include "OrderChecker.hpp"
#include "Barrier.hpp"

class RadixSortPerformanceBase : public BasePerformance {
public:
    RadixSortPerformanceBase(VulkanContext &context, bool useInternalProfiler = true)
    : BasePerformance(context) {
        _sort = RadixSort{&context.device, useInternalProfiler};
        _isSorted = OrderChecker{&context.device, (1 << 10) * 90};
        _sort.init();
        _isSorted.init();
    }

    void warmup() {
        VulkanBuffer buffer = _context.device.createBuffer(usage, VMA_MEMORY_USAGE_GPU_ONLY, 1024);
        spdlog::info("warming up Radix sort performance, {} runs", warmUpRuns);
        for(int i = 0; i < warmUpRuns; i++) {
            execute([&](auto commandBuffer){
                _sort(commandBuffer, buffer);
            });
            _sort.commitProfiler();
        }
        _sort.profiler.clearRunTimes();
    }

protected:
    RadixSort _sort;
    OrderChecker _isSorted;
};