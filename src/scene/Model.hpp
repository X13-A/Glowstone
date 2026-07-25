#pragma once

#include "core/Constants.hpp"
#include "gpu/Geometry.hpp"
#include "gpu/Texture.hpp"
#include "gpu/Context.hpp"
#include "gpu/CommandBuffers.hpp"
#include "core/Vk.hpp"
#include <vector>
#include "scene/Transform.hpp"
#include "scene/Mesh.hpp"
#include "assets/ObjLoader.hpp"


namespace vkrt {
namespace scene {
using namespace vkrt::core;
using namespace vkrt::gpu;
using namespace vkrt::assets;

struct ModelUBO
{
    alignas(16) glm::mat4 modelMat;
    alignas(16) glm::mat4 viewMat;
    alignas(16) glm::mat4 projMat;
    alignas(16) glm::mat4 normalMat;
    alignas(16) glm::mat4 prevModelViewProj;
};

struct ShadedMesh
{
    // 1 to 1 relationship
    Mesh mesh;
    Material material;
};

class Model
{
public:
    std::string name;
    Transform transform;

    std::vector<ShadedMesh> shadedMeshes;

    // Model uniforms
    std::vector<vk::Buffer> uniformBuffers;
    std::vector<vk::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    // These descriptor sets are used for model-unique data (no textures)
    std::vector<vk::DescriptorSet> modelDescriptorSets;

    // Ray tracing
    vk::AccelerationStructureKHR blasHandle;
    vk::Buffer blasBuffer;
    vk::DeviceMemory blasBufferMemory;
    vk::DeviceAddress blasBufferAddress;

public:

    void load(ModelInfo info, const Context& context, CommandBuffers& commandBufferManager, vk::DescriptorPool descriptorPool);
    void cleanup(vk::Device device);

    void createDescriptorSets(const Context& context, vk::DescriptorPool descriptorPool);
    void createUniformBuffers(const Context& context);

    void createBLAS(
        const Context& context,
        CommandBuffers& commandBufferManager);

};

// One model has multiple meshes
// One mesh has one material
// One material has a few textures (albedo, etc)

// Eeach model has descriptor sets for their geometry (Transform data)
// Each mesh has specific descriptor sets for it's textures etc

// Each Model builds one BLAS with geometry indexing :
//    - buildInfo.geometryCount = submeshes.size();
//    - buildInfo.pGeometries = geometries.data();

// This allows material indexing in the RT shader

} // namespace scene
} // namespace vkrt
