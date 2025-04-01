#pragma once

#include "Texture.h"
#include "VulkanDescriptorSet.h"
#include <array>
#include <functional>

namespace eular {

    struct Field : std::array<Texture, 2> {
        std::string name;
        std::array<VkDescriptorSet, 2> descriptorSet{};

        void swap() {
            std::swap(descriptorSet[0], descriptorSet[1]);
        }
    };

    struct VectorField {
        Field u;
        Field v;
        Field w;

        void swap() {
            u.swap();
            v.swap();
            w.swap();
        }
    };

    using UpdateSource = std::function<void(VkCommandBuffer, Field&, glm::uvec3)>;
    using PostAdvect = std::function<bool(VkCommandBuffer, Field&, glm::uvec3)>;

    struct Quantity {
        std::string name;
        Field field;
        Field source;
        float diffuseRate{MIN_FLOAT};

        UpdateSource update = [](VkCommandBuffer, Field&, glm::uvec3){};
        std::vector<PostAdvect> postAdvectActions;
    };


    using VectorFieldSource3D = std::vector<glm::vec3>;
    using VectorFieldSource2D = std::vector<glm::vec2>;

    using DivergenceField = Field;
    using PressureField = Field;
    using ForceField = Field;
    using VorticityField = Field;
}