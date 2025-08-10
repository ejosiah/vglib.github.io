#include <glm/gtc/type_ptr.hpp>
#include "Mesh.h"
#include <fstream>
#include <meshoptimizer.h>

VkPrimitiveTopology toVulkan(uint32_t MeshType){
    switch (MeshType) {
        case aiPrimitiveType_POINT: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        case aiPrimitiveType_LINE: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        case aiPrimitiveType_TRIANGLE: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        default:
//            throw std::runtime_error{"unknown primitive type"};
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; // TODO new version of assimp causing this to fail
    }
}

void optimize(mesh::Mesh& mesh){
    std::vector<uint32_t> remap(mesh.indices.size());
    auto vertexCount = meshopt_generateVertexRemap(remap.data(), mesh.indices.data(), mesh.indices.size(),
                                                   mesh.vertices.data(), mesh.vertices.size(), sizeof(Vertex));
    std::vector<uint32_t> remappedIndices(mesh.indices.size());
    meshopt_remapIndexBuffer(remappedIndices.data(), mesh.indices.data(), mesh.indices.size(), remap.data());

    std::vector<Vertex> remappedVertices(vertexCount);
    meshopt_remapVertexBuffer(remappedVertices.data(), mesh.vertices.data(), mesh.vertices.size(), sizeof(Vertex), remap.data());

    std::vector<uint32_t> reOrderedIndices(remappedIndices.size());
    meshopt_optimizeVertexCache(reOrderedIndices.data(), remappedIndices.data(), remappedIndices.size(), vertexCount);

    mesh.vertices = remappedVertices;
    mesh.indices = reOrderedIndices;
}

mesh::Mesh loadMesh(const std::string& parent, const aiNode* node, const aiScene* scene, uint32_t* numVertices, uint32_t meshId){
    auto aiMesh = scene->mMeshes[node->mMeshes[meshId]];
    mesh::Mesh mesh;
    mesh.name = aiMesh->mName.C_Str();
    mesh.primitiveTopology = toVulkan(aiMesh->mPrimitiveTypes);
    *numVertices = aiMesh->mNumVertices;
    for(auto j = 0u; j < aiMesh->mNumVertices; j++){
        Vertex vertex{};
        auto aiVec = aiMesh->mVertices[j];


        vertex.position = {aiVec.x, aiVec.y, aiVec.z, 1.0f};
        mesh.bounds.min = glm::min(mesh.bounds.min, glm::vec3(vertex.position));
        mesh.bounds.max = glm::max(mesh.bounds.max, glm::vec3(vertex.position));

        if(aiMesh->HasNormals()){
            auto aiNorm = aiMesh->mNormals[j];
            vertex.normal = {aiNorm.x, aiNorm.y, aiNorm.z};
        }

        if(aiMesh->HasTangentsAndBitangents()){
            auto aiTan = aiMesh->mTangents[j];
            auto aiBi = aiMesh->mBitangents[j];
            vertex.tangent = {aiTan.x, aiTan.y, aiTan.z};
            vertex.bitangent = {aiBi.x, aiBi.y, aiBi.z};
        }

        if(aiMesh->HasVertexColors(0)){
            auto aiCol = aiMesh->mColors[j][0];
            vertex.color = {aiCol.r, aiCol.g, aiCol.b, aiCol.a};
        }

        if(aiMesh->HasTextureCoords((0))){  // TODO retrieve up to AI_MAX_NUMBER_OF_TEXTURECOORDS textures coordinates
            auto aiTex = aiMesh->mTextureCoords[0][j];
            vertex.uv = {aiTex.x, aiTex.y};
        }
        mesh.vertices.push_back(vertex);
    }

    if(aiMesh->HasFaces()){
        for(auto j = 0; j < aiMesh->mNumFaces; j++){
            auto face = aiMesh->mFaces[j];
            for(auto k = 0; k < face.mNumIndices; k++){
                mesh.indices.push_back(face.mIndices[k]);
            }
        }
    }

    auto aiMaterial = scene->mMaterials[aiMesh->mMaterialIndex];
    mesh::Material material;
    aiString aiString;

    auto ret = aiMaterial->Get(AI_MATKEY_NAME, aiString);
    if(ret == aiReturn_SUCCESS){
        material.name = aiString.C_Str();
    }

    float scalar = 1.0f;
    ret = aiMaterial->Get(AI_MATKEY_OPACITY, scalar);
    if(ret == aiReturn_SUCCESS){
        material.opacity = scalar;
    }

    ret = aiMaterial->Get(AI_MATKEY_REFRACTI, scalar);
    if(ret == aiReturn_SUCCESS){
        material.ior = scalar;
    }


    ret = aiMaterial->Get(AI_MATKEY_SHININESS, scalar);
    if(ret == aiReturn_SUCCESS){
        material.shininess = scalar;
    }
    ret = aiMaterial->Get(AI_MATKEY_SHADING_MODEL, scalar);
    if(ret == aiReturn_SUCCESS){
        material.illum = scalar;
    }

    glm::vec3 color;
    uint32_t size = 3;
    ret = aiMaterial->Get(AI_MATKEY_COLOR_DIFFUSE, glm::value_ptr(color), &size);
    if(ret == aiReturn_SUCCESS){
        material.diffuse = color;
    }

    ret = aiMaterial->Get(AI_MATKEY_COLOR_AMBIENT, glm::value_ptr(color), &size);
    if(ret == aiReturn_SUCCESS){
        material.ambient = color;
    }

    ret = aiMaterial->Get(AI_MATKEY_COLOR_SPECULAR, glm::value_ptr(color), &size);
    if(ret == aiReturn_SUCCESS){
        material.specular = color;
    }

    ret = aiMaterial->Get(AI_MATKEY_COLOR_EMISSIVE, glm::value_ptr(color), &size);
    if(ret == aiReturn_SUCCESS){
        material.emission = color;
    }

    ret = aiMaterial->Get(AI_MATKEY_COLOR_TRANSPARENT, glm::value_ptr(color), &size);
    if(ret == aiReturn_SUCCESS){
        material.transmittance = color;
    }

    mesh::TextureMaterial texMaterial;
    ret = aiMaterial->GetTexture(aiTextureType_DIFFUSE, 0, &aiString);
    if(ret == aiReturn_SUCCESS){
        texMaterial.diffuseMap = fmt::format("{}/{}", parent, aiString.C_Str());
    }

    ret = aiMaterial->GetTexture(aiTextureType_AMBIENT, 0, &aiString);
    if(ret == aiReturn_SUCCESS){
        texMaterial.ambientMap =  fmt::format("{}/{}", parent, aiString.C_Str());
    }

    ret = aiMaterial->GetTexture(aiTextureType_SPECULAR, 0, &aiString);
    if(ret == aiReturn_SUCCESS){
        texMaterial.specularMap =  fmt::format("{}/{}", parent, aiString.C_Str());
    }

    if(aiMaterial->GetTexture(aiTextureType_NORMALS, 0, &aiString) == aiReturn_SUCCESS
      || aiMaterial->GetTexture(aiTextureType_HEIGHT, 0, &aiString) == aiReturn_SUCCESS){
        texMaterial.normalMap =  fmt::format("{}/{}", parent, aiString.C_Str());
    }

    ret = aiMaterial->GetTexture(aiTextureType_AMBIENT_OCCLUSION, 0, &aiString);
    if(ret == aiReturn_SUCCESS){
        texMaterial.ambientOcclusionMap =  fmt::format("{}/{}", parent, aiString.C_Str());
    }
    mesh.material = material;
    mesh.textureMaterial = texMaterial;

//    optimize(mesh);

    return mesh;
}

int load(std::vector<mesh::Mesh>& meshes, const std::string& parent, const aiNode* node, const aiScene* scene){
    uint32_t numVertices = 0;
 //   std::vector<std::future<mesh::Mesh>> futures;
    for(auto i = 0; i < node->mNumMeshes; i++){
        meshes.push_back(loadMesh(parent, node, scene, &numVertices, i));
    }

    for(auto i = 0; i < node->mNumChildren; i++){
        numVertices += load(meshes, parent, node->mChildren[i], scene);
    }
    return numVertices;
}

int mesh::load(std::vector<Mesh> &meshes, const std::string& path, uint32_t flags) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path.data(), flags);
    auto i = path.find_last_of("/") + 1;
    auto parentPath = fs::path{path}.parent_path().string();

    auto start = chrono::high_resolution_clock::now();
    auto res =  load(meshes, parentPath, scene->mRootNode, scene);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    return res;
}

void mesh::transform(std::vector<Mesh>& meshes, glm::mat4 xform){
    for(auto& mesh : meshes){
        for(auto& vertex : mesh.vertices){
            vertex.position = xform * vertex.position;
            vertex.normal = glm::mat3(xform) * vertex.normal;
        }
    }
}

void mesh::bounds(const std::vector<Mesh> &meshes, glm::vec3 &vMin, glm::vec3 &vMax) {
    vMin = glm::vec3(MAX_FLOAT);
    vMax = glm::vec3(MIN_FLOAT);
    for(auto& mesh : meshes){
        for(auto& vertex : mesh.vertices){
            auto& pos = vertex.position;
            vMin = glm::min(vMin, pos.xyz());
            vMax = glm::max(vMax, pos.xyz());
        }
    }
}

void writeMeshToObj(std::ofstream& objFile, const mesh::Mesh& mesh, size_t& vertexOffset) {
    objFile << "# Mesh: " << mesh.name << std::endl;

    // Write vertices to .obj file
    for (const auto& vertex : mesh.vertices) {
        objFile << "v "
                << vertex.position.x << " "
                << vertex.position.y << " "
                << vertex.position.z << std::endl;

        // Optional: Write colors (if needed)
        objFile << "vc "
                << vertex.color.r << " "
                << vertex.color.g << " "
                << vertex.color.b << std::endl;

        // Write normals
        objFile << "vn "
                << vertex.normal.x << " "
                << vertex.normal.y << " "
                << vertex.normal.z << std::endl;

        // Write texture coordinates (UVs)
        objFile << "vt "
                << vertex.uv.x << " "
                << vertex.uv.y << std::endl;
    }

    // Write faces using indices
    for (size_t i = 0; i < mesh.indices.size(); i += 3) {
        uint32_t idx1 = mesh.indices[i] + vertexOffset;
        uint32_t idx2 = mesh.indices[i + 1] + vertexOffset;
        uint32_t idx3 = mesh.indices[i + 2] + vertexOffset;

        objFile << "f "
                << idx1 << "/" << idx1 << "/" << idx1 << " "
                << idx2 << "/" << idx2 << "/" << idx2 << " "
                << idx3 << "/" << idx3 << "/" << idx3 << std::endl;
    }

    // Update vertex offset for the next mesh
    vertexOffset += mesh.vertices.size();
}

// Function to write material properties to an MTL file
void writeMaterialToMtl(std::ofstream& mtlFile, const mesh::Material& material) {
    mtlFile << "newmtl " << material.name << std::endl;
    mtlFile << "Ka " << material.ambient.r << " " << material.ambient.g << " " << material.ambient.b << std::endl;
    mtlFile << "Kd " << material.diffuse.r << " " << material.diffuse.g << " " << material.diffuse.b << std::endl;
    mtlFile << "Ks " << material.specular.r << " " << material.specular.g << " " << material.specular.b << std::endl;
    mtlFile << "Ke " << material.emission.r << " " << material.emission.g << " " << material.emission.b << std::endl;
    mtlFile << "Ni " << material.ior << std::endl;
    mtlFile << "d " << material.opacity << std::endl;
    mtlFile << "illum " << material.illum << std::endl;
    // Add texture maps if they exist (optional)
    if (!material.name.empty()) {
        mtlFile << "map_Kd " << material.name << "_diffuse.jpg" << std::endl;  // Assuming a diffuse map
    }
}


void mesh::writeToObject(const std::vector<mesh::Mesh>& meshes, const std::string& filename) {
    std::ofstream objFile(filename + ".obj");
    std::ofstream mtlFile(filename + ".mtl");

    if (!objFile.is_open() || !mtlFile.is_open()) {
        spdlog::error("Failed to open files for writing!");
        return;
    }

    // Write the material library reference at the top of the .obj file
    objFile << "mtllib " << filename << ".mtl" << std::endl;

    size_t vertexOffset = 1; // Wavefront .obj indices start at 1

    // Write all meshes
    for (const auto& mesh : meshes) {
        objFile << "usemtl " << mesh.material.name << std::endl;

        // Write mesh vertices and faces
        writeMeshToObj(objFile, mesh, vertexOffset);

        // Write the material for the mesh into the MTL file
        writeMaterialToMtl(mtlFile, mesh.material);
    }

    objFile.close();
    mtlFile.close();

    spdlog::info("Successfully wrote meshes to {}.obj and materials to {}.mtl", filename, filename);
}