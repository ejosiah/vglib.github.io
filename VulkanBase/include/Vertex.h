#pragma once

#include <vector>
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <array>
#include <vector>

struct ClipSpace{

    struct Quad{
        static constexpr  std::array<glm::vec2, 8> positions{
                glm::vec2{-1, -1},glm::vec2{0, 0},
                glm::vec2{-1, 1}, glm::vec2{0, 1},
                glm::vec2{1, -1}, glm::vec2{1, 0},
                glm::vec2{1, 1}, glm::vec2{1, 1}
        };
        static constexpr VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;

        static constexpr VkFrontFace  frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    };

    struct Triangle{
        static constexpr  std::array<glm::vec2, 6> positions{
                glm::vec2{-1, 1},glm::vec2{0, 1},
                glm::vec2{1, 1}, glm::vec2{1, 1},
                glm::vec2{0, -1}, glm::vec2{0.5, 0.5},
        };
        static constexpr VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        static constexpr VkFrontFace  frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    };


    static std::vector<VkVertexInputBindingDescription> bindingDescription(){
        return {
                {0, 2 * sizeof(glm::vec2), VK_VERTEX_INPUT_RATE_VERTEX}
        };
    }

    static std::vector<VkVertexInputAttributeDescription> attributeDescriptions(){
        return {
                {0, 0, VK_FORMAT_R32G32_SFLOAT, 0},
                {1, 0, VK_FORMAT_R32G32_SFLOAT, sizeof(glm::vec2)}
        };
    }
};

struct Ndc{
    static constexpr std::array<glm::vec4, 8> points{
            glm::vec4{-1, 1, 0, 1}, glm::vec4{1, 1, 0, 1},
            glm::vec4{-1, -1, 0, 1}, glm::vec4{1, -1, 0, 1},
            glm::vec4{-1, 1, 1, 1}, glm::vec4{1, 1, 1, 1},
            glm::vec4{-1, -1, 1, 1}, glm::vec4{1, -1, 1, 1}
    };

    static constexpr std::array<glm::uint32_t, 24> indices{
            0, 1, 2, 3, 0, 2, 1, 3,
            4, 5, 6, 7, 4, 6, 5, 7,
            0, 4, 1, 5, 2, 6, 3, 7
    };
};

struct Vertex{
    glm::vec4 position;
    glm::vec4 color;
    alignas(16) glm::vec3 normal;
    alignas(16) glm::vec3 tangent;
    alignas(16) glm::vec3 bitangent;
    glm::vec2 uv;

    static std::vector<VkVertexInputBindingDescription> bindingDisc(){
        return {
                {0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX}
        };
    }

    static std::vector<VkVertexInputAttributeDescription> attributeDisc(){
        return {
                {0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, (uint32_t)offsetof(Vertex, position)},
                {1, 0, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(Vertex, normal)},
                {2, 0, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(Vertex, tangent)},
                {3, 0, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(Vertex, bitangent)},
                {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, (uint32_t)offsetof(Vertex, color)},
                {5, 0, VK_FORMAT_R32G32_SFLOAT, (uint32_t)offsetof(Vertex, uv)}
        };
    }
};

struct VertexMultiAttributes{
    glm::vec4 position;

    union {
        struct {
            glm::vec4 color0;
            glm::vec4 color1;
        };
        std::array<glm::vec4, 2> color;
    };

    glm::vec3 normal;
    glm::vec3 tangent;
    glm::vec3 bitangent;

    union {
        struct {
            glm::vec2 uv0;
            glm::vec2 uv1;
        };
        std::array<glm::vec2, 2> uv;

    };

    static VkVertexInputBindingDescription bindingDescription() {
        return { 0, sizeof(VertexMultiAttributes), VK_VERTEX_INPUT_RATE_VERTEX};
    }

    static std::vector<VkVertexInputAttributeDescription> attributeDescription() {
        return {
                {0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, (uint32_t)offsetof(VertexMultiAttributes, position)},
                {1, 0, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(VertexMultiAttributes, normal)},
                {2, 0, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(VertexMultiAttributes, tangent)},
                {3, 0, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(VertexMultiAttributes, bitangent)},
                {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, (uint32_t)offsetof(VertexMultiAttributes, color0)},
                {5, 0, VK_FORMAT_R32G32B32A32_SFLOAT, (uint32_t)offsetof(VertexMultiAttributes, color1)},
                {6, 0, VK_FORMAT_R32G32_SFLOAT, (uint32_t)offsetof(VertexMultiAttributes, uv0)},
                {7, 0, VK_FORMAT_R32G32_SFLOAT, (uint32_t)offsetof(VertexMultiAttributes, uv1)}
        };
    }
};


struct Vertices{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    VkPrimitiveTopology topology;

    [[nodiscard]]
    static float surfaceArea(const Vertices& vertices){
        assert(vertices.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

        auto area = 0.f;
        auto numTriangles = vertices.indices.size()/3;
        for(auto i = 0; i < numTriangles; i += 3){
            const auto& i0 = vertices.indices[i * 3 + 0];
            const auto& i1 = vertices.indices[i * 3 + 1];
            const auto& i2 = vertices.indices[i * 3 + 2];

            const auto v0 = vertices.vertices[i0].position.xyz();
            const auto v1 = vertices.vertices[i1].position.xyz();
            const auto v2 = vertices.vertices[i2].position.xyz();

            const auto a = v1 - v0;
            const auto b = v2 - v0;

            area += glm::length(glm::cross(a, b)) * 0.5f;
        }

        return area;
    }
};