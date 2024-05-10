#pragma once

#include "BasePerformace.hpp"
#include "Sort.hpp"
#include "IsSorted.hpp"
#include "Barrier.hpp"

class RadixSortPerformanceBase : public BasePerformance {
public:
    RadixSortPerformanceBase(VulkanContext &context)
    : BasePerformance(context) {
        _sort = RadixSort{&context.device, true};
        _isSorted = IsSorted{&context.device, (1 << 10) * 90};
        _sort.init();
        _isSorted.init();
    }

    void warmup() {
        VulkanBuffer buffer = _context.device.createBuffer(usage, VMA_MEMORY_USAGE_GPU_ONLY, 1024);
        spdlog::info("warming up Radix sort performance, 1000 runs");
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
    IsSorted _isSorted;
    static constexpr int warmUpRuns = 1000;
    static constexpr int runs = 10000;
};