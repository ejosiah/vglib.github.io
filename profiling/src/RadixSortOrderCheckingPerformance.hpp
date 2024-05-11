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
    {}

    ~RadixSortOrderCheckingPerformance() override = default;


    std::map<size_t, Report> SortWithNoOrderChecking() {
        _sort.disableOrderChecking();
        Profiler profiler{ &_context.device };
        profiler.addQuery("no_order_checking");

        auto maxNumItems = numItems.back();
        VulkanBuffer stagingBuffer = _context.device.createStagingBuffer(maxNumItems * sizeof(uint32_t));
        VulkanBuffer deviceBuffer = _context.device.createBuffer(usage, VMA_MEMORY_USAGE_GPU_ONLY, maxNumItems * sizeof(uint32_t));
        VulkanBuffer resultBuffer = _context.device.createBuffer(usage, VMA_MEMORY_USAGE_GPU_TO_CPU, sizeof(uint32_t));

        std::map<size_t, Report> reports;

        std::span<uint32_t> items = { reinterpret_cast<uint32_t*>(stagingBuffer.map()), maxNumItems };
        std::iota(items.begin(), items.end(), 0);


        for(int i = 0; i < numItems.size(); i++) {
            VkDeviceSize size = numItems[i] * sizeof(uint32_t);

            execute([&](auto commandBuffer){
                VkBufferCopy region{0, 0, size};
                vkCmdCopyBuffer(commandBuffer, stagingBuffer, deviceBuffer, 1, &region);
                Barrier::transferWriteToComputeRead(commandBuffer, { deviceBuffer });
            });
            for(int j = 0; j < runs; j++) {
                execute([&](auto commandBuffer) {
                    profiler.profile("no_order_checking", commandBuffer, [&]{
                        _sort(commandBuffer, { &deviceBuffer, 0, size });
                    });

                });
                profiler.commit();
                if(j % (runs/10) == 0){
                    spdlog::info("sort with order checking disabled performance test run {} {:.2f} % complete", (i+1), ((j * 100/float(runs))));
                }
            }

            reports.insert(std::make_pair(numItems[i], profiler.queryStats()));
            profiler.clearRunTimes();
        }

        return reports;
    }
    
    std::map<size_t, Report> SortWithOrderChecking() {
        _sort.enableOrderChecking();
        Profiler profiler{ &_context.device };
        profiler.addQuery("order_checking");


        auto maxNumItems = numItems.back();
        VulkanBuffer stagingBuffer = _context.device.createStagingBuffer(maxNumItems * sizeof(uint32_t));
        VulkanBuffer deviceBuffer = _context.device.createBuffer(usage, VMA_MEMORY_USAGE_GPU_ONLY, maxNumItems * sizeof(uint32_t));
        VulkanBuffer resultBuffer = _context.device.createBuffer(usage, VMA_MEMORY_USAGE_GPU_TO_CPU, sizeof(uint32_t));

        std::map<size_t, Report> reports;

        std::span<uint32_t> items = { reinterpret_cast<uint32_t*>(stagingBuffer.map()), maxNumItems };
        std::iota(items.begin(), items.end(), 0);

        for(int i = 0; i < numItems.size(); i++) {
            VkDeviceSize size = numItems[i] * sizeof(uint32_t);

            execute([&](auto commandBuffer){
                VkBufferCopy region{0, 0, size};
                vkCmdCopyBuffer(commandBuffer, stagingBuffer, deviceBuffer, 1, &region);
                Barrier::transferWriteToComputeRead(commandBuffer, { deviceBuffer });
            });
            for(int j = 0; j < runs; j++) {
                execute([&](auto commandBuffer) {

                    profiler.profile("order_checking", commandBuffer, [&]{
                        _sort(commandBuffer, { &deviceBuffer, 0, size });
                    });

                });
                profiler.commit();
                if(j % (runs/10) == 0){
                    spdlog::info("sort with order checking enabled performance test run {} {:.2f} % complete", (i+1), ((j * 100/float(runs))));
                }
            }

            reports.insert(std::make_pair(numItems[i], profiler.queryStats()));
            profiler.clearRunTimes();
        }

        return reports;
    }
    
    

    std::string report() final {
        warmup();
        auto sortNoOrderChecking = SortWithNoOrderChecking();
        auto sortOrderChecking = SortWithOrderChecking();

        std::map<size_t, Report> reports;
        
        for(auto count : numItems) {
            reports[count] = {};
        }

       for(auto [count, report] : sortNoOrderChecking) {
           for(auto [key, stats] : report) {
               reports[count][key] = stats;
           }
       }

       for(auto [count, report] : sortOrderChecking) {
           for(auto [key, stats] : report) {
               reports[count][key] = stats;
           }
       }

        std::string report{};
        report += fmt::format("{:_>112}\n", "");
        report += fmt::format("{:^112}\n", "Sort With Order checking ");
        report += fmt::format("{:_>112}\n", "");
        report += fmt::format("{:22}{:15}{:15}{:15}{:15}{:15}{:15}\n", "num items (1e6)", 0.5, 3, 6, 9, 12, 15);
        report += fmt::format("{:22}{:15.4f}{:15.4f}{:15.4f}{:15.4f}{:15.4f}{:15.4f}\n","No Order Checking (ms)"
                , reports[500000]["no_order_checking"].meanValue
                , reports[3000000]["no_order_checking"].meanValue
                , reports[6000000]["no_order_checking"].meanValue
                , reports[9000000]["no_order_checking"].meanValue
                , reports[12000000]["no_order_checking"].meanValue
                , reports[15000000]["no_order_checking"].meanValue
        );
        report +=
                fmt::format("{:22}{:15.4f}{:15.4f}{:15.4f}{:15.4f}{:15.4f}{:15.4f}\n", "Order Checking (ms)"
                        , reports[500000]["order_checking"].meanValue
                        , reports[3000000]["order_checking"].meanValue
                        , reports[6000000]["order_checking"].meanValue
                        , reports[9000000]["order_checking"].meanValue
                        , reports[12000000]["order_checking"].meanValue
                        , reports[15000000]["order_checking"].meanValue
                );

        report += fmt::format("{:_>112}\n", "");
        
        return report;
    }
    

private:
    std::array<size_t, 6> numItems{ 500000, 3000000, 6000000, 9000000, 12000000, 15000000 };


};