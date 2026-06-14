#pragma once

#include "Field.hpp"

#include <functional>
#include <span>
#include <vector>

namespace eular {
    enum class TimeDirection { Forward, Backword };

    enum class LinearSolverStrategy  {
        Jacobi, RBGS, ConjugateGradient
    };

    using VectorFieldSource3D = std::vector<glm::vec3>;
    using VectorFieldSource2D = std::vector<glm::vec2>;

    using DivergenceField = Field;
    using PressureField = Field;
    using ForceField = Field;
    using VorticityField = Field;

    using ExternalForce = std::function<void(VkCommandBuffer, std::span<VkDescriptorSet>, glm::uvec3)>;

    constexpr uint32_t in = 0;
    constexpr uint32_t out = 1;
}
