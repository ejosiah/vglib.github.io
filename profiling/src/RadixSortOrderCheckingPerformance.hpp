#pragma once

#include "RadixSortPerformanceBase.hpp"
#include <spdlog/spdlog.h>
#include <fmt/format.h>

#include <random>
#include <map>
#include <vector>


class RadixSortOrderCheckingPerformance : public RadixSortPerformanceBase {
    struct Outcome {
        std::string label;
        float value;
    };
public:
    RadixSortOrderCheckingPerformance(VulkanContext& context)
    : RadixSortPerformanceBase(context)
    {
    }

    ~RadixSortOrderCheckingPerformance() override = default;

    Outcome  runWithoutOrderChecking() {
        std::span<uint32_t> items = { reinterpret_cast<uint32_t*>(_stagingBuffer.map()), _numItems };
        std::iota(items.begin(), items.end(), 0);
        std::shuffle(items.begin(), items.end(), std::default_random_engine{ 1 << 20 });
        
        for(auto i = 0; i < runs; i++){
            execute([&](auto commandBuffer) {
                VkBufferCopy region{0, 0, _stagingBuffer.size};
                vkCmdCopyBuffer(commandBuffer, _stagingBuffer, _deviceBuffer, 1, &region);
                BufferRegion data{&_deviceBuffer, 0, _deviceBuffer.size};
                _sort(commandBuffer, data);
            });
            if(i % (runs/10) == 0){
                spdlog::debug("radix sorts (No order checking) {:.2f} % complete", ((float(i) * 100.f/float(runs))));
            }
            _sort.commitProfiler();
        }
        
        auto report = _sort.profiler.groupStats();
        float total = 0;
        for(auto [_, stats] : report){
            total += stats.meanValue;
        }
        return { "No Order Checking", total };
    }
    
    Outcome  runWithAllBlocksOrdered() {
        return {};
    }
    
    Outcome  runWithBlockZeroOrdered() {
        return {};
    }
    
    Outcome  runWithBlockOneOrdered() {
        return {};
    }
    
    Outcome  runWithBlockTwoOrdered() {
        return {};
    }
    
    Outcome  runWithBlockThreeOrdered() {
        return {};
    }
    
    std::string report() final {
        warmup();
        initializeBuffers();

        std::vector<Outcome> outcomes{};
        outcomes.push_back(runWithoutOrderChecking());

        std::string report{};
        report += fmt::format("Radix Sort (Order checking)\n");

        for(auto [label, result] : outcomes) {
            report += fmt::format("{:25}{:15.4f}\n", label, result);
        }

        return report;
    }
    
    void initializeBuffers() {
        VkDeviceSize size = _numItems * sizeof(uint32_t);
        _stagingBuffer = _context.device.createStagingBuffer(size);
        _deviceBuffer = _context.device.createBuffer(usage, VMA_MEMORY_USAGE_GPU_ONLY, size);
        _resultBuffer = _context.device.createBuffer(usage, VMA_MEMORY_USAGE_GPU_TO_CPU, sizeof(uint32_t));
    }

private:
    size_t _numItems{500000};
    VulkanBuffer _stagingBuffer;
    VulkanBuffer _deviceBuffer;
    VulkanBuffer _resultBuffer;

};