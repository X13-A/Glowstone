#pragma once

#include <vector>
#include "core/Math.hpp"
#include "core/Vk.hpp"
#include "gpu/Context.hpp"
#include "gpu/CommandBuffers.hpp"
#include "scene/Model.hpp"


namespace vkrt {
namespace gpu {
using namespace vkrt::core;
using namespace vkrt::scene;

struct BLASInstance
{
    vk::AccelerationStructureKHR blas;
    glm::mat4 transform;

    uint32_t instanceId;
    uint32_t hitGroupIndex;
};

class Tlas
{
private:
    vk::AccelerationStructureKHR tlas;
    vk::Buffer tlasBuffer;
    vk::DeviceMemory tlasMemory;
    vk::Buffer instanceBuffer;
    vk::DeviceMemory instanceMemory;
    vk::Buffer scratchBuffer;
    vk::DeviceMemory scratchMemory;

public:
    void createTLAS(const Context& context, const std::vector<Model>& models, CommandBuffers& commandBufferManager);

    void cleanup(const Context& context);

    vk::AccelerationStructureKHR getTLAS() const;
private:
    void createInstanceBuffer(const Context& context, const std::vector<BLASInstance>& instances);
    vk::AccelerationStructureBuildSizesInfoKHR getBuildSizes(const Context& context, uint32_t instanceCount);

    void createAccelerationStructure(const Context& context, vk::DeviceSize size);

    void buildTLAS(const Context& context, uint32_t instanceCount, vk::CommandBuffer commandBuffer);
};

} // namespace gpu
} // namespace vkrt
