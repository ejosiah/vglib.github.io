#include "Console.hpp"
#include "RadixSortPerformance.hpp"
#include "RadixSortOrderCheckingPerformance.hpp"
#include "OrderCheckingPerformanceTest.hpp"

#include <fmt/format.h>
#include <vulkan/vulkan.h>
#include <fmt/color.h>

int main() {
    fs::current_path("../../../");
    ContextCreateInfo createInfo{};
    createInfo.applicationInfo.sType  = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    createInfo.applicationInfo.applicationVersion = VK_MAKE_VERSION(0, 0, 0);
    createInfo.applicationInfo.pApplicationName = "Vulkan Performance Test";
    createInfo.applicationInfo.apiVersion = VK_API_VERSION_1_3;
    createInfo.applicationInfo.pEngineName = "";
    createInfo.settings.uniqueQueueFlags = VK_QUEUE_COMPUTE_BIT;

    VulkanContext context{ createInfo };
    context.init();
    vkDevice = context.device.logicalDevice;

    Console::start();
    RadixSortPerformance radixSortPerformance{ context };
    RadixSortOrderCheckingPerformance radixSortOrderChecking{ context };
    OrderCheckingPerformanceTest orderCheckingPerformanceTest{ context };

    try {
        auto report = radixSortPerformance.report();
//    auto report = radixSortOrderChecking.report();
//        auto report = orderCheckingPerformanceTest.report();
        fmt::print("{}\n", report);
    }catch(...){
        fmt::print(bg(fmt::color::yellow), "error encountered while running performance tests");
    }

    Console::stop();
    return 0;
}