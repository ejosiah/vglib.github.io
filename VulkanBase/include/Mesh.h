#pragma once

#include "common.h"
#include "Vertex.h"
#include "primitives.h"

#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

#include <filesystem>
#include <vector>

namespace mesh {

    constexpr uint32_t DEFAULT_PROCESS_FLAGS = aiProcess_GenSmoothNormals | aiProcess_Triangulate
                                                | aiProcess_CalcTangentSpace | aiProcess_JoinIdenticalVertices
                                                | aiProcess_ValidateDataStructure | aiProcess_FixInfacingNormals;

    struct Material{
        std::string name;
        alignas(16) glm::vec3 diffuse = glm::vec3(0.6f);
        alignas(16) glm::vec3 ambient = glm::vec3(0.6f);
        alignas(16) glm::vec3 specular = glm::vec3(1);
        alignas(16) glm::vec3 emission = glm::vec3(0);
        alignas(16) glm::vec3 transmittance = glm::vec3(0);
        float shininess = 0;
        float ior = 0;
        float opacity = 1;
        float illum = 1;
    };

    struct TextureMaterial{
        std::string diffuseMap;
        std::string ambientMap;
        std::string specularMap;
        std::string normalMap;
        std::string ambientOcclusionMap;
    };

    struct Mesh {
        std::string name{};
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        Material material{};
        TextureMaterial textureMaterial{};
        VkPrimitiveTopology primitiveTopology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        struct {
            glm::vec3 min = glm::vec3(MAX_FLOAT);
            glm::vec3 max = glm::vec3(MIN_FLOAT);
        } bounds;
    };

    int load(std::vector<Mesh>& meshes, const std::string& path, uint32_t flags = DEFAULT_PROCESS_FLAGS);

    void transform(std::vector<Mesh>& meshes, glm::mat4 xform);

    void bounds(const std::vector<Mesh>& meshes, glm::vec3& vMin, glm::vec3& vMax);

    inline void normalize(std::vector<Mesh>& meshes, float scale = 1.0f){
        primitives::normalize(meshes, scale);
    }

    [[nodiscard]]
    static float surfaceArea(const Mesh& mesh){
        assert(mesh.primitiveTopology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

        auto area = 0.f;
        auto numTriangles = mesh.indices.size()/3;
        for(auto i = 0; i < numTriangles; i++){
            const auto& i0 = mesh.indices[i * 3 + 0];
            const auto& i1 = mesh.indices[i * 3 + 1];
            const auto& i2 = mesh.indices[i * 3 + 2];

            const auto v0 = mesh.vertices[i0].position.xyz();
            const auto v1 = mesh.vertices[i1].position.xyz();
            const auto v2 = mesh.vertices[i2].position.xyz();

            const auto a = v1 - v0;
            const auto b = v2 - v0;

            area += glm::length(glm::cross(a, b)) * 0.5f;
        }

        return area;
    }

    static glm::vec3 center(const Mesh& mesh){
        glm::vec3 center{0};

        auto numTriangles = mesh.indices.size()/3;
        for(auto i = 0; i < numTriangles; i++){
            const auto& i0 = mesh.indices[i * 3 + 0];
            const auto& i1 = mesh.indices[i * 3 + 1];
            const auto& i2 = mesh.indices[i * 3 + 2];

            center += mesh.vertices[i0].position.xyz();
            center += mesh.vertices[i1].position.xyz();
            center += mesh.vertices[i2].position.xyz();
        }
        center /= numTriangles * 3;

        return center;
    }
}