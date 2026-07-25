#include "assets/ObjLoader.hpp"
#include <unordered_map>
#include "gpu/Geometry.hpp"
#include <stb_image.h>

#ifndef TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#endif


namespace vkrt {
namespace assets {
using namespace vkrt::gpu;

std::string normalizePath(const std::string& baseDir, const std::string& textureName)
{
    if (textureName.empty()) return "";

    // Remove "./"
    std::string cleanTextureName = textureName;
    if (cleanTextureName.substr(0, 2) == "./")
    {
        cleanTextureName = cleanTextureName.substr(2);
    }

    return baseDir + cleanTextureName;
}

bool isNormalMap(const std::string& path)
{
    if (path.empty()) return false;

    int width, height, channels;
    if (!stbi_info(path.c_str(), &width, &height, &channels)) return false;

    return channels >= 3;
}

std::string selectNormalMap(const std::string& normalPath, const std::string& bumpPath)
{
    if (isNormalMap(normalPath)) return normalPath;
    if (isNormalMap(bumpPath)) return bumpPath;
    return "";
}

void computeTangents(MeshInfo& mesh)
{
    // Iterate over each triangle
    for (int i = 0; i < mesh.indices.size(); i += 3)
    {
        uint32_t i1 = mesh.indices[i];
        uint32_t i2 = mesh.indices[i+1];
        uint32_t i3 = mesh.indices[i+2];

        Vertex& v1 = mesh.vertices[i1];
        Vertex& v2 = mesh.vertices[i2];
        Vertex& v3 = mesh.vertices[i3];

        glm::vec3 edge1 = v2.pos - v1.pos;
        glm::vec3 edge2 = v3.pos - v1.pos;
        glm::vec2 deltaUV1 = v2.texCoord - v1.texCoord;
        glm::vec2 deltaUV2 = v3.texCoord - v1.texCoord;

        float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
        glm::vec3 tangent;
        tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
        tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
        tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);

        v1.tangent += tangent;
        v2.tangent += tangent;
        v3.tangent += tangent;
        
        glm::vec3 bitangent;
        bitangent.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
        bitangent.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
        bitangent.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);

        v1.bitangent += bitangent;
        v2.bitangent += bitangent;
        v3.bitangent += bitangent;
    }
    for (Vertex& vertex : mesh.vertices)
    {
        vertex.tangent = glm::normalize(vertex.tangent);
        vertex.bitangent = glm::normalize(vertex.bitangent);
    }
}

ModelInfo ObjLoader::loadObj(const std::string& objPath)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    std::string baseDir = objPath.substr(0, objPath.find_last_of("/\\") + 1);

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, objPath.c_str(), baseDir.c_str(), true))
    {
        throw std::runtime_error("Failed to load OBJ: " + warn + err);
    }

    ModelInfo model;

    // Load materials
    for (const auto& mat : materials)
    {
        PBRMaterialInfo matInfo{};
        matInfo.name = mat.name;

        matInfo.albedoTexture = normalizePath(baseDir, mat.diffuse_texname);
        matInfo.normalTexture = selectNormalMap(
            normalizePath(baseDir, mat.normal_texname),
            normalizePath(baseDir, mat.bump_texname));
        matInfo.metallicTexture = normalizePath(baseDir, mat.metallic_texname);
        matInfo.roughnessTexture = normalizePath(baseDir, mat.roughness_texname);
        matInfo.aoTexture = normalizePath(baseDir, mat.ambient_texname);
        matInfo.displacementTexture = normalizePath(baseDir, mat.displacement_texname);

        matInfo.albedoFactor[0] = mat.diffuse[0];
        matInfo.albedoFactor[1] = mat.diffuse[1];
        matInfo.albedoFactor[2] = mat.diffuse[2];

        matInfo.metallicFactor = mat.metallic;
        matInfo.roughnessFactor = mat.roughness;
        matInfo.aoFactor = 0;

        model.materials.push_back(matInfo);
    }

    // Load meshes
    for (const auto& shape : shapes)
    {
        MeshInfo mesh;
        std::unordered_map<Vertex, uint32_t> uniqueVertices;

        for (const auto& index : shape.mesh.indices)
        {
            Vertex vertex{};
            vertex.pos =
            {
                attrib.vertices[3 * index.vertex_index + 0],
                attrib.vertices[3 * index.vertex_index + 1],
                attrib.vertices[3 * index.vertex_index + 2]
            };

            if (index.normal_index >= 0) 
            {
                vertex.normal = 
                {
                    attrib.normals[3 * index.normal_index + 0],
                    attrib.normals[3 * index.normal_index + 1],
                    attrib.normals[3 * index.normal_index + 2]
                };
            }

            if (index.texcoord_index >= 0) 
            {
                vertex.texCoord =
                {
                    attrib.texcoords[2 * index.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                };
            }

            if (uniqueVertices.count(vertex) == 0)
            {
                uniqueVertices[vertex] = static_cast<uint32_t>(mesh.vertices.size());
                mesh.vertices.push_back(vertex);
            }

            mesh.indices.push_back(uniqueVertices[vertex]);
        }
        computeTangents(mesh);
        model.meshes.push_back(mesh);

        // Assign material index
        int materialIndex = shape.mesh.material_ids.empty() ? -1 : shape.mesh.material_ids[0];
        model.meshMaterialIndices.push_back(materialIndex);
    }

    return model;
}

} // namespace assets
} // namespace vkrt
