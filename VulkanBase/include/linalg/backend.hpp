#pragma once

#include "VulkanDevice.h"

#include <VulkanBuffer.h>

#include <stdexcept>
#include <stack>
#include <vector>

namespace linalg {
    struct cpu_backend {};

    struct vulkan_backend {};

    template<typename Backend, typename T>
    struct buffer_type;

    template<typename T>
    struct buffer_type<cpu_backend, T> {
        using type = std::vector<T>;
    };

    template<typename T>
    struct buffer_type<vulkan_backend, T> {
        using type = VulkanBuffer;
    };

    template<typename Backend, typename T>
    using buffer_type_t = typename buffer_type<Backend, T>::type;

    enum class backend_type { cpu, gpu };

    struct execution_context {
        VulkanDevice* device{};
        VkCommandBuffer commandBuffer{};

        static execution_context& get();
    };

    inline std::stack<execution_context> context_stack;

    class execution_context_janitor {
    public:
        explicit execution_context_janitor(execution_context context) {
            context_stack.push(context);
        }

        execution_context_janitor(VulkanDevice& device, VkCommandBuffer commandBuffer)
            : execution_context_janitor({&device, commandBuffer}) {}

        ~execution_context_janitor() {
            context_stack.pop();
        }
    };

    inline execution_context& execution_context::get() {
        if (context_stack.empty()) {
            throw std::runtime_error{"No execution context set"};
        }

        return context_stack.top();
    }
}
