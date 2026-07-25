#include "gpu/Tlas.hpp"
#include <stdexcept>
#include "gpu/Utils.hpp"
#include <iostream>


namespace vkrt {
namespace gpu {
using namespace vkrt::core;
using namespace vkrt::scene;

void Tlas::createTLAS(const Context& context, const std::vector<Model>& models, CommandBuffers& commandBufferManager)
{
    std::vector<BLASInstance> BLASintances;
    int instanceIndex = 0;
    for (const Model& model : models)
    {
        BLASInstance instance;
        instance.blas = model.blasHandle;
        instance.transform = model.transform.getTransformMatrix();
        instance.instanceId = instanceIndex++;
        instance.hitGroupIndex = RT_CLOSEST_HIT_GENERAL_SHADER_INDEX;
        BLASintances.push_back(instance);
    }

    createInstanceBuffer(context, BLASintances);

    vk::AccelerationStructureBuildSizesInfoKHR buildSizes = getBuildSizes(context, BLASintances.size());

    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress;
    gpu::Buffers::createBuffer(context, buildSizes.accelerationStructureSize, usage, vk::MemoryPropertyFlagBits::eDeviceLocal, tlasBuffer, tlasMemory, true);

    gpu::Buffers::createScratchBuffer(context, buildSizes.buildScratchSize, scratchBuffer, scratchMemory);

    createAccelerationStructure(context, buildSizes.accelerationStructureSize);

    vk::CommandBuffer commandBuffer = commandBufferManager.beginSingleTimeCommands(context.device);
    buildTLAS(context, BLASintances.size(), commandBuffer);
    commandBufferManager.endSingleTimeCommands(context.device, context.graphicsQueue, commandBuffer);

    if (VULKAN_HPP_DEFAULT_DISPATCHER.vkSetDebugUtilsObjectNameEXT)
    {
        vk::DebugUtilsObjectNameInfoEXT nameInfo{};
        nameInfo.objectType = vk::ObjectType::eAccelerationStructureKHR;
        nameInfo.objectHandle = (uint64_t)(VkAccelerationStructureKHR)tlas;
        nameInfo.pObjectName = "Scene TLAS";
        context.device.setDebugUtilsObjectNameEXT(nameInfo);
    }
}

void Tlas::createInstanceBuffer(const Context& context, const std::vector<BLASInstance>& instances)
{
    vk::DeviceSize bufferSize = sizeof(vk::AccelerationStructureInstanceKHR) * instances.size();
    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress;
    gpu::Buffers::createBuffer(context, bufferSize, usage, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, instanceBuffer, instanceMemory, true);

    vk::AccelerationStructureInstanceKHR* instanceData = static_cast<vk::AccelerationStructureInstanceKHR*>(context.device.mapMemory(instanceMemory, 0, bufferSize));

    for (size_t i = 0; i < instances.size(); i++)
    {
        const auto& instance = instances[i];

        vk::AccelerationStructureDeviceAddressInfoKHR addressInfo{};
        addressInfo.accelerationStructure = instance.blas;
        vk::DeviceAddress blasAddress = context.device.getAccelerationStructureAddressKHR(addressInfo);

        vk::TransformMatrixKHR vulkanTransform;
        for (int row = 0; row < 3; row++)
        {
            for (int col = 0; col < 4; col++)
            {
                vulkanTransform.matrix[row][col] = instance.transform[col][row];
            }
        }

        instanceData[i].transform = vulkanTransform;
        instanceData[i].instanceCustomIndex = instance.instanceId;
        instanceData[i].mask = 0xFF; // Visible to all rays
        instanceData[i].instanceShaderBindingTableRecordOffset = 0;
        instanceData[i].flags = 0;
        instanceData[i].accelerationStructureReference = blasAddress;
    }

    context.device.unmapMemory(instanceMemory);
}

vk::AccelerationStructureBuildSizesInfoKHR Tlas::getBuildSizes(const Context& context, uint32_t instanceCount)
{
    vk::BufferDeviceAddressInfo addressInfo{};
    addressInfo.buffer = instanceBuffer;

    vk::AccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.arrayOfPointers = VK_FALSE;
    instancesData.data.deviceAddress = context.device.getBufferAddress(addressInfo);

    vk::AccelerationStructureGeometryKHR geometry{};
    geometry.geometryType = vk::GeometryTypeKHR::eInstances;
    geometry.geometry.instances = instancesData;

    vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.type = vk::AccelerationStructureTypeKHR::eTopLevel;
    buildInfo.flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
    buildInfo.mode = vk::BuildAccelerationStructureModeKHR::eBuild;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    return context.device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo, instanceCount);
}

void Tlas::createAccelerationStructure(const Context& context, vk::DeviceSize size)
{
    vk::AccelerationStructureCreateInfoKHR createInfo{};
    createInfo.buffer = tlasBuffer;
    createInfo.offset = 0;
    createInfo.size = size;
    createInfo.type = vk::AccelerationStructureTypeKHR::eTopLevel;

    tlas = context.device.createAccelerationStructureKHR(createInfo);
}

void Tlas::buildTLAS(const Context& context, uint32_t instanceCount, vk::CommandBuffer commandBuffer)
{
    vk::BufferDeviceAddressInfo addressInfo{};
    addressInfo.buffer = instanceBuffer;

    vk::AccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.arrayOfPointers = VK_FALSE;
    instancesData.data.deviceAddress = context.device.getBufferAddress(addressInfo);

    vk::AccelerationStructureGeometryKHR geometry{};
    geometry.geometryType = vk::GeometryTypeKHR::eInstances;
    geometry.geometry.instances = instancesData;

    vk::BufferDeviceAddressInfo scratchAddressInfo{};
    scratchAddressInfo.buffer = scratchBuffer;
    vk::DeviceAddress scratchAddress = context.device.getBufferAddress(scratchAddressInfo);

    vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.type = vk::AccelerationStructureTypeKHR::eTopLevel;
    buildInfo.flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
    buildInfo.mode = vk::BuildAccelerationStructureModeKHR::eBuild;
    buildInfo.dstAccelerationStructure = tlas;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;
    buildInfo.scratchData.deviceAddress = scratchAddress;

    vk::AccelerationStructureBuildRangeInfoKHR buildRange{};
    buildRange.primitiveCount = instanceCount;
    buildRange.primitiveOffset = 0;
    buildRange.firstVertex = 0;
    buildRange.transformOffset = 0;

    const vk::AccelerationStructureBuildRangeInfoKHR* pBuildRange = &buildRange;

    commandBuffer.buildAccelerationStructuresKHR(buildInfo, pBuildRange);
}

vk::AccelerationStructureKHR Tlas::getTLAS() const
{
    return tlas;
}

void Tlas::cleanup(const Context& context)
{
    if (tlas)
    {
        context.device.destroyAccelerationStructureKHR(tlas);
        tlas = nullptr;
    }

    if (tlasBuffer)
    {
        context.device.destroyBuffer(tlasBuffer);
        tlasBuffer = nullptr;
    }
    if (tlasMemory)
    {
        context.device.freeMemory(tlasMemory);
        tlasMemory = nullptr;
    }

    if (instanceBuffer)
    {
        context.device.destroyBuffer(instanceBuffer);
        instanceBuffer = nullptr;
    }
    if (instanceMemory)
    {
        context.device.freeMemory(instanceMemory);
        instanceMemory = nullptr;
    }

    if (scratchBuffer)
    {
        context.device.destroyBuffer(scratchBuffer);
        scratchBuffer = nullptr;
    }
    if (scratchMemory)
    {
        context.device.freeMemory(scratchMemory);
        scratchMemory = nullptr;
    }
}

} // namespace gpu
} // namespace vkrt
