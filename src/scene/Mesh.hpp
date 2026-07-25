#pragma once
#include <vector>
#include "gpu/Geometry.hpp"
#include <string>
#include "gpu/Context.hpp"
#include "gpu/CommandBuffers.hpp"
#include "scene/Material.hpp"


namespace vkrt {
namespace scene {
using namespace vkrt::gpu;

class Mesh
{
public:
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    vk::Buffer vertexBuffer;
    vk::DeviceMemory vertexBufferMemory;
    vk::Buffer indexBuffer;
    vk::DeviceMemory indexBufferMemory;

public:
    void init(const Context& context, CommandBuffers& commandBufferManager);
    void cleanup(vk::Device device);
};

} // namespace scene
} // namespace vkrt
