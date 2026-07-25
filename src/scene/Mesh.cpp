#include "scene/Mesh.hpp"
#include <tiny_obj_loader.h>
#include <stdexcept>
#include <iostream>
#include "gpu/Utils.hpp"


namespace vkrt {
namespace scene {
using namespace vkrt::gpu;

void Mesh::init(const Context& context, CommandBuffers& commandBufferManager)
{
    vk::BufferUsageFlags vertexUsageFlags = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;
    vk::MemoryPropertyFlags vertexMemoryFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    gpu::Buffers::createAndFillBuffer<Vertex>(context, commandBufferManager, vertices, vertexBuffer, vertexBufferMemory, vertexUsageFlags, vertexMemoryFlags, true);

    vk::BufferUsageFlags indexUsageFlags = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;
    vk::MemoryPropertyFlags indexMemoryFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    gpu::Buffers::createAndFillBuffer<uint32_t>(context, commandBufferManager, indices, indexBuffer, indexBufferMemory, indexUsageFlags, indexMemoryFlags, true);
}

void Mesh::cleanup(vk::Device device)
{
    device.destroyBuffer(indexBuffer);
    device.freeMemory(indexBufferMemory);
    device.destroyBuffer(vertexBuffer);
    device.freeMemory(vertexBufferMemory);
    vertices.clear();
    indices.clear();
}

} // namespace scene
} // namespace vkrt
