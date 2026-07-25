#include "scene/Model.hpp"
#include "gpu/Utils.hpp"
#include <iostream>
#include "assets/ObjLoader.hpp"
#include "gpu/DescriptorLayouts.hpp"


namespace vkrt {
namespace scene {
using namespace vkrt::assets;
using namespace vkrt::core;
using namespace vkrt::gpu;

void Model::load(ModelInfo info, const Context& context, CommandBuffers& commandBufferManager, vk::DescriptorPool descriptorPool)
{
    for (int i = 0; i < info.meshes.size(); ++i)
    {
        ShadedMesh shadedMesh;
        shadedMesh.mesh.vertices = info.meshes[i].vertices;
        shadedMesh.mesh.indices = info.meshes[i].indices;
        shadedMesh.mesh.init(context, commandBufferManager);

        int matIndex = info.meshMaterialIndices[i];

        bool hasError = matIndex == -1;
        if (hasError)
        {
            shadedMesh.material.init({}, context, commandBufferManager, descriptorPool, true);
        }
        else
        {
            shadedMesh.material.init(info.materials[matIndex], context, commandBufferManager, descriptorPool, false);
        }
        shadedMeshes.push_back(shadedMesh);
    }

    createUniformBuffers(context);
    createDescriptorSets(context, descriptorPool);
    createBLAS(context, commandBufferManager);
}

void Model::createDescriptorSets(const Context& context, vk::DescriptorPool descriptorPool)
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, DescriptorLayouts::getModelLayout());
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    allocInfo.pSetLayouts = layouts.data();

    modelDescriptorSets = context.device.allocateDescriptorSets(allocInfo);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vk::DescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(ModelUBO);

        std::array<vk::WriteDescriptorSet, 1> descriptorWrites{};

        descriptorWrites[0].dstSet = modelDescriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        context.device.updateDescriptorSets(descriptorWrites, {});
    }
}

void Model::createUniformBuffers(const Context& context)
{
    vk::DeviceSize bufferSize = sizeof(ModelUBO);

    uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        gpu::Buffers::createBuffer(context, bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, uniformBuffers[i], uniformBuffersMemory[i]);
        uniformBuffersMapped[i] = context.device.mapMemory(uniformBuffersMemory[i], 0, bufferSize);
    }
}

void Model::cleanup(vk::Device device)
{
    for (ShadedMesh& shadedMesh : shadedMeshes)
    {
        shadedMesh.mesh.cleanup(device);
        shadedMesh.material.cleanup(device);
    }

    device.destroyAccelerationStructureKHR(blasHandle);
    blasHandle = nullptr;

    device.destroyBuffer(blasBuffer);
    device.freeMemory(blasBufferMemory);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        device.destroyBuffer(uniformBuffers[i]);
        device.freeMemory(uniformBuffersMemory[i]);
    }
}

void Model::createBLAS(
    const Context& context,
    CommandBuffers& commandBufferManager)
{
    std::vector<vk::AccelerationStructureGeometryKHR> geometries;
    std::vector<vk::AccelerationStructureBuildRangeInfoKHR> buildRanges;
    std::vector<uint32_t> primitiveCounts;

    // Create geometry for each mesh
    for (const auto& shadedMesh : shadedMeshes)
    {
        const Mesh& mesh = shadedMesh.mesh;

        vk::DeviceAddress vertexBufferAddress = gpu::Buffers::getBufferDeviceAdress(context, mesh.vertexBuffer);
        vk::DeviceAddress indexBufferAddress = gpu::Buffers::getBufferDeviceAdress(context, mesh.indexBuffer);

        // Triangle data
        vk::AccelerationStructureGeometryTrianglesDataKHR trianglesData{};
        trianglesData.vertexFormat = vk::Format::eR32G32B32Sfloat;
        trianglesData.vertexData.deviceAddress = vertexBufferAddress;
        trianglesData.vertexStride = sizeof(Vertex);
        trianglesData.maxVertex = static_cast<uint32_t>(mesh.vertices.size()) - 1;
        trianglesData.indexType = vk::IndexType::eUint32;
        trianglesData.indexData.deviceAddress = indexBufferAddress;
        trianglesData.transformData = {};

        vk::AccelerationStructureGeometryKHR accelGeometry{};
        accelGeometry.geometryType = vk::GeometryTypeKHR::eTriangles;
        accelGeometry.flags = vk::GeometryFlagBitsKHR::eOpaque;
        accelGeometry.geometry.triangles = trianglesData;

        geometries.push_back(accelGeometry);

        // Build range info for this mesh
        uint32_t primitiveCount = static_cast<uint32_t>(mesh.indices.size() / 3);
        primitiveCounts.push_back(primitiveCount);

        vk::AccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
        buildRangeInfo.primitiveCount = primitiveCount;
        buildRangeInfo.primitiveOffset = 0;
        buildRangeInfo.firstVertex = 0;
        buildRangeInfo.transformOffset = 0;

        buildRanges.push_back(buildRangeInfo);
    }

    // Build params
    vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.type = vk::AccelerationStructureTypeKHR::eBottomLevel;
    buildInfo.flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
    buildInfo.geometryCount = static_cast<uint32_t>(geometries.size());
    buildInfo.pGeometries = geometries.data();
    buildInfo.mode = vk::BuildAccelerationStructureModeKHR::eBuild;

    // Get required sizes
    vk::AccelerationStructureBuildSizesInfoKHR sizeInfo = context.device.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice,
        buildInfo,
        primitiveCounts);

    // Create scratch buffer
    vk::Buffer scratchBuffer;
    vk::DeviceMemory scratchBufferMemory;

    gpu::Buffers::createScratchBuffer(
        context,
        sizeInfo.buildScratchSize,
        scratchBuffer,
        scratchBufferMemory);

    vk::DeviceAddress scratchBufferAddress = gpu::Buffers::getBufferDeviceAdress(context, scratchBuffer);

    // Create BLAS buffer
    vk::BufferUsageFlags blasBufferUsage =
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
        vk::BufferUsageFlagBits::eShaderDeviceAddress;

    gpu::Buffers::createBuffer(
        context,
        sizeInfo.accelerationStructureSize,
        blasBufferUsage,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        blasBuffer,
        blasBufferMemory,
        true);

    // Create BLAS
    vk::AccelerationStructureCreateInfoKHR createInfo{};
    createInfo.buffer = blasBuffer;
    createInfo.offset = 0;
    createInfo.size = sizeInfo.accelerationStructureSize;
    createInfo.type = vk::AccelerationStructureTypeKHR::eBottomLevel;

    blasHandle = context.device.createAccelerationStructureKHR(createInfo);

    buildInfo.dstAccelerationStructure = blasHandle;
    buildInfo.scratchData.deviceAddress = scratchBufferAddress;

    const vk::AccelerationStructureBuildRangeInfoKHR* pBuildRange = buildRanges.data();

    vk::CommandBuffer commandBuffer = commandBufferManager.beginSingleTimeCommands(context.device);
    commandBuffer.buildAccelerationStructuresKHR(buildInfo, pBuildRange);
    commandBufferManager.endSingleTimeCommands(context.device, context.graphicsQueue, commandBuffer);

    // Cleanup scratch buffer
    context.device.destroyBuffer(scratchBuffer);
    context.device.freeMemory(scratchBufferMemory);

    if (VULKAN_HPP_DEFAULT_DISPATCHER.vkSetDebugUtilsObjectNameEXT)
    {
        vk::DebugUtilsObjectNameInfoEXT nameInfo{};
        nameInfo.objectType = vk::ObjectType::eAccelerationStructureKHR;
        nameInfo.objectHandle = (uint64_t)(VkAccelerationStructureKHR)blasHandle;
        nameInfo.pObjectName = name.c_str();
        context.device.setDebugUtilsObjectNameEXT(nameInfo);
    }
}

} // namespace scene
} // namespace vkrt
