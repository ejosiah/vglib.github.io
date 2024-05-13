#pragma once

#include "RadixSortPerformanceBase.hpp"


#include <spdlog/spdlog.h>

#include <random>
#include <map>
#include <tuple>

class RadixSortPerformance: public RadixSortPerformanceBase {
public:
    RadixSortPerformance(VulkanContext& context)
    : RadixSortPerformanceBase(context)
    {

    }

    ~RadixSortPerformance() override = default;

    std::string generateReport(const std::string& title, auto&& setup, auto&& testFunc) {
        std::array<size_t, 6> numItems{ 500000, 3000000, 6000000, 9000000, 12000000, 15000000 };
        setup(numItems.back());
        VulkanBuffer resultBuffer = _context.device.createBuffer(usage, VMA_MEMORY_USAGE_GPU_TO_CPU, sizeof(uint32_t));
        auto isSorted =  reinterpret_cast<uint32_t*>(resultBuffer.map());
        std::map<size_t, Report> reports;

        for(int i = 0; i < numItems.size(); i++) {
            for(int j = 0; j < runs; j++) {
                execute([&](auto commandBuffer) {
                    VkDeviceSize size = numItems[i] * sizeof(uint32_t);
                    testFunc(commandBuffer, numItems[i]);
                    _isSorted(commandBuffer, {&deviceBuffer, 0, size}, { &resultBuffer, 0, resultBuffer.size } );
                    Barrier::transferWriteToHostRead(commandBuffer, resultBuffer);
                });
//                assert(*isSorted == 0u);
                _sort.commitProfiler();
                if(j % (runs/10) == 0){
                    spdlog::info("{} performance test run {} {:.2f} % complete",title, (i+1), ((j * 100/float(runs))));
                }
            }

            reports.insert(std::make_pair(numItems[i], _sort.profiler.groupStats()));
            _sort.profiler.clearRunTimes();
        }

        std::string report{};
        report += fmt::format("{:_>110}\n", "");
        report += fmt::format("{:^110}\n", title);
        report += fmt::format("{:_>110}\n", "");
        report += fmt::format("{:20}{:15}{:15}{:15}{:15}{:15}{:15}\n", "num items (1e6)", 0.5, 3, 6, 9, 12, 15);
        report += fmt::format("{:20}{:15.4f}{:15.4f}{:15.4f}{:15.4f}{:15.4f}{:15.4f}\n","count (ms)"
                , reports[500000]["count"].meanValue
                , reports[3000000]["count"].meanValue
                , reports[6000000]["count"].meanValue
                , reports[9000000]["count"].meanValue
                , reports[12000000]["count"].meanValue
                , reports[15000000]["count"].meanValue
        );
        report +=
                fmt::format("{:20}{:15.4f}{:15.4f}{:15.4f}{:15.4f}{:15.4f}{:15.4f}\n", "prefix sum (ms)"
                        , reports[500000]["radix_sort_prefix_sum"].meanValue
                        , reports[3000000]["radix_sort_prefix_sum"].meanValue
                        , reports[6000000]["radix_sort_prefix_sum"].meanValue
                        , reports[9000000]["radix_sort_prefix_sum"].meanValue
                        , reports[12000000]["radix_sort_prefix_sum"].meanValue
                        , reports[15000000]["radix_sort_prefix_sum"].meanValue
                );
        report += fmt::format("{:20}{:15.4f}{:15.4f}{:15.4f}{:15.4f}{:15.4f}{:15.4f}\n", "reorder (ms)"
                , reports[500000]["reorder"].meanValue
                , reports[3000000]["reorder"].meanValue
                , reports[6000000]["reorder"].meanValue
                , reports[9000000]["reorder"].meanValue
                , reports[12000000]["reorder"].meanValue
                , reports[15000000]["reorder"].meanValue
        );
        report += fmt::format("{:20}{:15.4f}{:15.4f}{:15.4f}{:15.4f}{:15.4f}{:15.4f}\n", "copy data in (ms)"
                , reports[500000]["copy_to_key"].meanValue
                , reports[3000000]["copy_to_key"].meanValue
                , reports[6000000]["copy_to_key"].meanValue
                , reports[9000000]["copy_to_key"].meanValue
                , reports[12000000]["copy_to_key"].meanValue
                , reports[15000000]["copy_to_key"].meanValue
        );
        report += fmt::format("{:20}{:15.4f}{:15.4f}{:15.4f}{:15.4f}{:15.4f}{:15.4f}\n", "copy data out (ms)"
                , reports[500000]["copy_from_key"].meanValue
                , reports[3000000]["copy_from_key"].meanValue
                , reports[6000000]["copy_from_key"].meanValue
                , reports[9000000]["copy_from_key"].meanValue
                , reports[12000000]["copy_from_key"].meanValue
                , reports[15000000]["copy_from_key"].meanValue
        );
        report += fmt::format("{:_>110}\n", "");

        std::map<size_t, float> totals{};
        for(auto key : numItems){
            totals[key] = 0.f;
            for(auto [_, stats] : reports[key]){
                totals[key] += stats.meanValue;
            }
        }

        report += fmt::format("{:20}{:15.4f}{:15.4f}{:15.4f}{:15.4f}{:15.4f}{:15.4f}\n", "totals (ms)"
                , totals[500000]
                , totals[3000000]
                , totals[6000000]
                , totals[9000000]
                , totals[12000000]
                , totals[15000000]
        );

        report += fmt::format("{:->110}\n", "");

        return report;
    }


    std::string profileParts() {
        return
            generateReport("Radix Sort parts", [&](auto maxNumItems){
                stagingBuffer = _context.device.createStagingBuffer(maxNumItems * sizeof(uint32_t));
                deviceBuffer = _context.device.createBuffer(usage, VMA_MEMORY_USAGE_GPU_ONLY, maxNumItems * sizeof(uint32_t));
                std::span<uint32_t> items = { reinterpret_cast<uint32_t*>(stagingBuffer.map()), maxNumItems };
                std::iota(items.begin(), items.end(), 0);
                std::shuffle(items.begin(), items.end(), std::default_random_engine{ 1 << 20 });
                assert(std::is_sorted(items.begin(), items.end()) != true);
            },
            [&](auto commandBuffer, auto numItems){
                VkDeviceSize size = numItems * sizeof(uint32_t);
                VkBufferCopy region{0, 0, size};
                vkCmdCopyBuffer(commandBuffer, stagingBuffer, deviceBuffer, 1, &region);
                Barrier::transferWriteToRead(commandBuffer, { deviceBuffer });
                _sort(commandBuffer, {&deviceBuffer, 0, size});
            });
    }

    std::string profileWithIndices() {
        return
            generateReport("Radix Sort With Index Reorder", [&](auto maxNumItems){
                   stagingBuffer = _context.device.createStagingBuffer(maxNumItems * sizeof(uint32_t));
                   deviceBuffer = _context.device.createBuffer(usage, VMA_MEMORY_USAGE_GPU_ONLY, maxNumItems * sizeof(uint32_t));
                   indexBuffer = _context.device.createBuffer(usage, VMA_MEMORY_USAGE_GPU_ONLY, maxNumItems * sizeof(uint32_t));
                   std::span<uint32_t> items = { reinterpret_cast<uint32_t*>(stagingBuffer.map()), maxNumItems };
                   std::iota(items.begin(), items.end(), 0);
                   std::shuffle(items.begin(), items.end(), std::default_random_engine{ 1 << 20 });
                   assert(std::is_sorted(items.begin(), items.end()) != true);
               },
               [&](auto commandBuffer, auto numItems){
                   VkDeviceSize size = numItems * sizeof(uint32_t);
                   VkBufferCopy region{0, 0, size};
                   vkCmdCopyBuffer(commandBuffer, stagingBuffer, deviceBuffer, 1, &region);
                   Barrier::transferWriteToRead(commandBuffer, { deviceBuffer });
                   _sort.sortWithIndices(commandBuffer, deviceBuffer, indexBuffer);
               });
}

    std::string report() override {
        warmup();
        auto partsReport = profileParts();
        auto indexReorderReport = profileWithIndices();
        return fmt::format("{}\n\n{}", partsReport, indexReorderReport) ;
    }

private:
    VulkanBuffer stagingBuffer;
    VulkanBuffer deviceBuffer;
    VulkanBuffer indexBuffer;
};
