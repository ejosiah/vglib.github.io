#pragma once

#include "OrderChecker.hpp"
#include "Profiler.hpp"
#include "BasePerformance.hpp"

class OrderCheckingPerformanceTest : public BasePerformance {
public:
    OrderCheckingPerformanceTest(VulkanContext& ctx)
    : BasePerformance(ctx)
    {
        _checkOrder = OrderChecker{ &ctx.device, 15000000};
        _checkOrder.init();
        _profiler = Profiler{ &ctx.device };
        _profiler.addQuery("order_checking");
    }

    ~OrderCheckingPerformanceTest() override = default;

    std::string report() override {
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
                    
                    _profiler.profile("order_checking", commandBuffer, [&]{
                        _checkOrder(commandBuffer, { &deviceBuffer, 0, size }, { &resultBuffer, 0, resultBuffer.size});
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

        std::string report{};
        report += fmt::format("{:_>110}\n", "");
        report += fmt::format("{:^105}\n", "Order Checking");
        report += fmt::format("{:_>105}\n", "");
        report += fmt::format("{:20}{:15}{:15}{:15}{:15}{:15}{:15}\n", "num items (1e6)", 0.5, 3, 6, 9, 12, 15);
        report += fmt::format("{:20}{:15.4f}{:15.4f}{:15.4f}{:15.4f}{:15.4f}{:15.4f}\n","order_checking (ms)"
                , reports[500000]["order_checking"].meanValue
                , reports[3000000]["order_checking"].meanValue
                , reports[6000000]["order_checking"].meanValue
                , reports[9000000]["order_checking"].meanValue
                , reports[12000000]["order_checking"].meanValue
                , reports[15000000]["order_checking"].meanValue
        );
        report += fmt::format("{:_>110}\n", "");
        
        return report;
    }

private:
    OrderChecker _checkOrder;
    Profiler _profiler;
};