#include "render/pipeline/RayTracingPipeline.hpp"
#include "core/Utils.hpp"
#include <stdexcept>
#include "gpu/Utils.hpp"
#include "core/Constants.hpp"
#include <iostream>
#include "assets/TextureManager.hpp"
#include "core/Settings.hpp"
#include <random>
#include "gpu/DescriptorLayouts.hpp"


namespace vkrt {
namespace render {
using namespace vkrt::assets;
using namespace vkrt::core;
using namespace vkrt::gpu;
using namespace vkrt::scene;

void RayTracingPipeline::init(const Context& context, uint32_t width, uint32_t height)
{
    sampleCount = 0;
    createRayTracingPipelineLayout(context);
    createRayTracingPipeline(context);
    createShaderBindingTable(context);
    createDescriptorPool(context);
    createDescriptorSet(context);
    createUniformBuffer(context);
    createStorageImage(context, width, height);
    createReservoirBuffers(context, width, height);
}

void RayTracingPipeline::writeDescriptors(const Context& context, CommandBuffers& commandBufferManager, const std::vector<Model>& models, vk::AccelerationStructureKHR tlas, vk::ImageView depthImageView, vk::ImageView normalsImageView, vk::ImageView albedoImageView, vk::ImageView roughnessImageView, vk::ImageView metalnessImageView, vk::ImageView velocityImageView)
{
    std::vector<vk::ImageView> allAlbedoTextureViews;
    std::vector<vk::ImageView> allNormalTextureViews;
    std::vector<vk::ImageView> allRoughnessTextureViews;
    std::vector<vk::ImageView> allMetalnessTextureViews;
    createRayTracingResources(context, commandBufferManager, tlas, models, allAlbedoTextureViews, allNormalTextureViews, allRoughnessTextureViews, allMetalnessTextureViews);
    writeDescriptorSet(context, depthImageView, normalsImageView, albedoImageView, roughnessImageView, metalnessImageView, velocityImageView, tlas, allAlbedoTextureViews, allNormalTextureViews, allRoughnessTextureViews, allMetalnessTextureViews);
}

void RayTracingPipeline::handleResize(const Context& context, uint32_t width, uint32_t height, vk::ImageView depthImageView, vk::ImageView normalsImageView, vk::ImageView albedoImageView, vk::ImageView roughnessImageView, vk::ImageView metalnessImageView, vk::ImageView velocityImageView)
{
    if (storageImageView)
    {
        context.device.destroyImageView(storageImageView);
    }
    if (storageImage)
    {
        context.device.destroyImage(storageImage);
    }
    if (storageImageMemory)
    {
        context.device.freeMemory(storageImageMemory);
    }
    if (last_storageImageView)
    {
        context.device.destroyImageView(last_storageImageView);
    }
    if (storageImage)
    {
        context.device.destroyImage(last_storageImage);
    }
    if (storageImageMemory)
    {
        context.device.freeMemory(last_storageImageMemory);
    }

    for (int i = 0; i < 2; i++)
    {
        if (reservoirBuffers[i])
        {
            context.device.destroyBuffer(reservoirBuffers[i]);
        }
        if (reservoirBufferMemories[i])
        {
            context.device.freeMemory(reservoirBufferMemories[i]);
        }
    }

    createStorageImage(context, width, height);
    createReservoirBuffers(context, width, height);

    std::vector<vk::WriteDescriptorSet> descriptorWrites;

    // Binding 1: Storage image
    vk::DescriptorImageInfo storageInfo{};
    storageInfo.imageLayout = vk::ImageLayout::eGeneral;
    storageInfo.imageView = storageImageView;
    storageInfo.sampler = nullptr;

    vk::WriteDescriptorSet storageWrite{};
    storageWrite.dstSet = descriptorSet;
    storageWrite.dstBinding = 1;
    storageWrite.dstArrayElement = 0;
    storageWrite.descriptorType = vk::DescriptorType::eStorageImage;
    storageWrite.descriptorCount = 1;
    storageWrite.pImageInfo = &storageInfo;
    descriptorWrites.push_back(storageWrite);

    // Binding 11: Depth
    vk::DescriptorImageInfo depthInfos;
    depthInfos.imageLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
    depthInfos.imageView = depthImageView;
    depthInfos.sampler = globalTextureSampler;

    vk::WriteDescriptorSet depthWrite{};
    depthWrite.dstSet = descriptorSet;
    depthWrite.dstBinding = 11;
    depthWrite.dstArrayElement = 0;
    depthWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    depthWrite.descriptorCount = 1;
    depthWrite.pImageInfo = &depthInfos;
    descriptorWrites.push_back(depthWrite);

    // Binding 12: Normals
    vk::DescriptorImageInfo normalsInfos;
    normalsInfos.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    normalsInfos.imageView = normalsImageView;
    normalsInfos.sampler = globalTextureSampler;

    vk::WriteDescriptorSet normalsWrite{};
    normalsWrite.dstSet = descriptorSet;
    normalsWrite.dstBinding = 12;
    normalsWrite.dstArrayElement = 0;
    normalsWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    normalsWrite.descriptorCount = 1;
    normalsWrite.pImageInfo = &normalsInfos;
    descriptorWrites.push_back(normalsWrite);

    // Binding 13: Albedo
    vk::DescriptorImageInfo albedoInfos;
    albedoInfos.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    albedoInfos.imageView = albedoImageView;
    albedoInfos.sampler = globalTextureSampler;

    vk::WriteDescriptorSet albedoWrite{};
    albedoWrite.dstSet = descriptorSet;
    albedoWrite.dstBinding = 13;
    albedoWrite.dstArrayElement = 0;
    albedoWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    albedoWrite.descriptorCount = 1;
    albedoWrite.pImageInfo = &albedoInfos;
    descriptorWrites.push_back(albedoWrite);

    // Binding 14: Roughness
    vk::DescriptorImageInfo roughnessInfos;
    roughnessInfos.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    roughnessInfos.imageView = roughnessImageView;
    roughnessInfos.sampler = globalTextureSampler;

    vk::WriteDescriptorSet roughnessWrite{};
    roughnessWrite.dstSet = descriptorSet;
    roughnessWrite.dstBinding = 14;
    roughnessWrite.dstArrayElement = 0;
    roughnessWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    roughnessWrite.descriptorCount = 1;
    roughnessWrite.pImageInfo = &roughnessInfos;
    descriptorWrites.push_back(roughnessWrite);

    // Binding 15: Metalness
    vk::DescriptorImageInfo metalnessInfos;
    metalnessInfos.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    metalnessInfos.imageView = metalnessImageView;
    metalnessInfos.sampler = globalTextureSampler;

    vk::WriteDescriptorSet metalnessWrite{};
    metalnessWrite.dstSet = descriptorSet;
    metalnessWrite.dstBinding = 15;
    metalnessWrite.dstArrayElement = 0;
    metalnessWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    metalnessWrite.descriptorCount = 1;
    metalnessWrite.pImageInfo = &metalnessInfos;
    descriptorWrites.push_back(metalnessWrite);

    // Binding 16: Frame accumulation
    vk::DescriptorImageInfo lastImageInfos;
    lastImageInfos.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    lastImageInfos.imageView = last_storageImageView;
    lastImageInfos.sampler = globalTextureSampler;

    vk::WriteDescriptorSet lastImageWrite{};
    lastImageWrite.dstSet = descriptorSet;
    lastImageWrite.dstBinding = 16;
    lastImageWrite.dstArrayElement = 0;
    lastImageWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    lastImageWrite.descriptorCount = 1;
    lastImageWrite.pImageInfo = &lastImageInfos;
    descriptorWrites.push_back(lastImageWrite);

    // Bindings 17 & 18: reservoir buffers
    vk::DescriptorBufferInfo reservoirBufferInfoA{};
    reservoirBufferInfoA.buffer = reservoirBuffers[0];
    reservoirBufferInfoA.offset = 0;
    reservoirBufferInfoA.range = VK_WHOLE_SIZE;

    vk::WriteDescriptorSet reservoirWriteA{};
    reservoirWriteA.dstSet = descriptorSet;
    reservoirWriteA.dstBinding = 17;
    reservoirWriteA.dstArrayElement = 0;
    reservoirWriteA.descriptorType = vk::DescriptorType::eStorageBuffer;
    reservoirWriteA.descriptorCount = 1;
    reservoirWriteA.pBufferInfo = &reservoirBufferInfoA;
    descriptorWrites.push_back(reservoirWriteA);

    vk::DescriptorBufferInfo reservoirBufferInfoB{};
    reservoirBufferInfoB.buffer = reservoirBuffers[1];
    reservoirBufferInfoB.offset = 0;
    reservoirBufferInfoB.range = VK_WHOLE_SIZE;

    vk::WriteDescriptorSet reservoirWriteB{};
    reservoirWriteB.dstSet = descriptorSet;
    reservoirWriteB.dstBinding = 18;
    reservoirWriteB.dstArrayElement = 0;
    reservoirWriteB.descriptorType = vk::DescriptorType::eStorageBuffer;
    reservoirWriteB.descriptorCount = 1;
    reservoirWriteB.pBufferInfo = &reservoirBufferInfoB;
    descriptorWrites.push_back(reservoirWriteB);

    // Motion vectors
    vk::DescriptorImageInfo velocityInfo{};
    velocityInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    velocityInfo.imageView = velocityImageView;
    velocityInfo.sampler = globalTextureSampler;

    vk::WriteDescriptorSet velocityWrite{};
    velocityWrite.dstSet = descriptorSet;
    velocityWrite.dstBinding = 19;
    velocityWrite.dstArrayElement = 0;
    velocityWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    velocityWrite.descriptorCount = 1;
    velocityWrite.pImageInfo = &velocityInfo;
    descriptorWrites.push_back(velocityWrite);

    context.device.updateDescriptorSets(descriptorWrites, {});
}

void RayTracingPipeline::createRayTracingPipelineLayout(const Context& context)
{
    // Push constants
    vk::PushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eMissKHR;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PushConstants);

    vk::DescriptorSetLayout layout = DescriptorLayouts::getRayTracingLayout();
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &layout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    pipelineLayout = context.device.createPipelineLayout(pipelineLayoutInfo);
}

void RayTracingPipeline::createRayTracingPipeline(const Context& context)
{
    // Load rt shaders
    std::vector<char> raygenShaderCode = readFile("assets/shaders/ray_gen.spv");
    std::vector<char> missShaderCode = readFile("assets/shaders/ray_miss.spv");
    std::vector<char> closestHitShaderCode = readFile("assets/shaders/ray_closesthit.spv");

    vk::ShaderModule raygenShaderModule = gpu::Shaders::createShaderModule(context, raygenShaderCode);
    vk::ShaderModule missShaderModule = gpu::Shaders::createShaderModule(context, missShaderCode);
    vk::ShaderModule closestHitShaderModule = gpu::Shaders::createShaderModule(context, closestHitShaderCode);

    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;

    // Ray gen
    vk::PipelineShaderStageCreateInfo raygenStage{};
    raygenStage.stage = vk::ShaderStageFlagBits::eRaygenKHR;
    raygenStage.module = raygenShaderModule;
    raygenStage.pName = "main";
    shaderStages.push_back(raygenStage);

    // Miss
    vk::PipelineShaderStageCreateInfo missStage{};
    missStage.stage = vk::ShaderStageFlagBits::eMissKHR;
    missStage.module = missShaderModule;
    missStage.pName = "main";
    shaderStages.push_back(missStage);

    // Closest hit
    vk::PipelineShaderStageCreateInfo closestHitStage{};
    closestHitStage.stage = vk::ShaderStageFlagBits::eClosestHitKHR;
    closestHitStage.module = closestHitShaderModule;
    closestHitStage.pName = "main";
    shaderStages.push_back(closestHitStage);

    std::vector<vk::RayTracingShaderGroupCreateInfoKHR> shaderGroups;

    // Raygen group
    vk::RayTracingShaderGroupCreateInfoKHR raygenGroup{};
    raygenGroup.type = vk::RayTracingShaderGroupTypeKHR::eGeneral;
    raygenGroup.generalShader = RT_RAYGEN_SHADER_INDEX;
    raygenGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
    raygenGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
    raygenGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
    shaderGroups.push_back(raygenGroup);

    // Miss group
    vk::RayTracingShaderGroupCreateInfoKHR missGroup{};
    missGroup.type = vk::RayTracingShaderGroupTypeKHR::eGeneral;
    missGroup.generalShader = RT_MISS_SHADER_INDEX;
    missGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
    missGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
    missGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
    shaderGroups.push_back(missGroup);

    // Hit group
    vk::RayTracingShaderGroupCreateInfoKHR hitGroup{};
    hitGroup.type = vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup;
    hitGroup.generalShader = VK_SHADER_UNUSED_KHR;
    hitGroup.closestHitShader = RT_CLOSEST_HIT_GENERAL_SHADER_INDEX;
    hitGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
    hitGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
    shaderGroups.push_back(hitGroup);

    // Create pipeline
    vk::RayTracingPipelineCreateInfoKHR pipelineInfo{};
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.groupCount = static_cast<uint32_t>(shaderGroups.size());
    pipelineInfo.pGroups = shaderGroups.data();
    pipelineInfo.maxPipelineRayRecursionDepth = RT_MAX_RECURSION_DEPTH;
    pipelineInfo.layout = pipelineLayout;

    pipeline = context.device.createRayTracingPipelineKHR(nullptr, nullptr, pipelineInfo).value;

    // Cleanup
    context.device.destroyShaderModule(raygenShaderModule);
    context.device.destroyShaderModule(missShaderModule);
    context.device.destroyShaderModule(closestHitShaderModule);
}

void RayTracingPipeline::createShaderBindingTable(const Context& context)
{
    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{};

    vk::PhysicalDeviceProperties2 props{};
    props.pNext = &rtProps;
    context.physicalDevice.getProperties2(&props);

    uint32_t handleSize = rtProps.shaderGroupHandleSize;
    uint32_t handleSizeAligned = ((handleSize + rtProps.shaderGroupBaseAlignment - 1) / rtProps.shaderGroupBaseAlignment) * rtProps.shaderGroupBaseAlignment;

    uint32_t groupCount = 3; // raygen + miss + hit
    uint32_t sbtSize = groupCount * handleSizeAligned;

    std::vector<uint8_t> shaderHandleStorage = context.device.getRayTracingShaderGroupHandlesKHR<uint8_t>(pipeline, 0, groupCount, sbtSize);

    vk::BufferUsageFlags sbtUsage = vk::BufferUsageFlagBits::eShaderBindingTableKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress;
    vk::MemoryPropertyFlags sbtMemProps = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

    gpu::Buffers::createBuffer(context, sbtSize, sbtUsage, sbtMemProps, sbtBuffer, sbtBufferMemory, true);

    void* data = context.device.mapMemory(sbtBufferMemory, 0, sbtSize);

    uint8_t* pData = reinterpret_cast<uint8_t*>(data);
    for (uint32_t g = 0; g < groupCount; g++)
    {
        memcpy(pData, shaderHandleStorage.data() + g * handleSize, handleSize);
        pData += handleSizeAligned;
    }

    context.device.unmapMemory(sbtBufferMemory);

    vk::DeviceAddress sbtAddress = gpu::Buffers::getBufferDeviceAdress(context, sbtBuffer);

    raygenSbtEntry.deviceAddress = sbtAddress;
    raygenSbtEntry.stride = handleSizeAligned;
    raygenSbtEntry.size = handleSizeAligned;

    missSbtEntry.deviceAddress = sbtAddress + handleSizeAligned;
    missSbtEntry.stride = handleSizeAligned;
    missSbtEntry.size = handleSizeAligned;

    hitSbtEntry.deviceAddress = sbtAddress + handleSizeAligned * 2;
    hitSbtEntry.stride = 0; // All geometries use same entry
    hitSbtEntry.size = handleSizeAligned;

    callableSbtEntry = vk::StridedDeviceAddressRegionKHR{};
}

void RayTracingPipeline::reloadShaders(const Context& context)
{
    context.device.destroyBuffer(sbtBuffer);
    context.device.freeMemory(sbtBufferMemory);
    context.device.destroyPipeline(pipeline);

    createRayTracingPipeline(context);
    createShaderBindingTable(context);
}

void RayTracingPipeline::createStorageImage(const Context& context, uint32_t width, uint32_t height)
{
    vk::Format imageFormat = RT_STORAGE_IMAGE_FORMAT;
    vk::ImageUsageFlags imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled;
    vk::MemoryPropertyFlags memProps = vk::MemoryPropertyFlagBits::eDeviceLocal;

    gpu::Image::createImage(context, width, height, imageFormat, vk::ImageTiling::eOptimal, imageUsage, memProps, storageImage, storageImageMemory);
    storageImageWidth = width;
    storageImageHeight = height;

    storageImageView = gpu::Image::createImageView(context, storageImage, imageFormat, vk::ImageAspectFlagBits::eColor);
    std::cout << "Storage image size: " << width << ", " << height << std::endl;

    // Create last storage image (for frame blending)
    imageFormat = RT_STORAGE_IMAGE_FORMAT;
    imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
    memProps = vk::MemoryPropertyFlagBits::eDeviceLocal;

    gpu::Image::createImage(context, width, height, imageFormat, vk::ImageTiling::eOptimal, imageUsage, memProps, last_storageImage, last_storageImageMemory);
    last_storageImageView = gpu::Image::createImageView(context, last_storageImage, imageFormat, vk::ImageAspectFlagBits::eColor);
}

void RayTracingPipeline::createReservoirBuffers(const Context& context, uint32_t width, uint32_t height)
{
    reservoirBufferSize = static_cast<vk::DeviceSize>(width) * height * sizeof(PackedReservoir);

    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
    vk::MemoryPropertyFlags memProps = vk::MemoryPropertyFlagBits::eDeviceLocal;

    for (int i = 0; i < 2; i++)
    {
        gpu::Buffers::createBuffer(context, reservoirBufferSize, usage, memProps, reservoirBuffers[i], reservoirBufferMemories[i], false);
    }

    // Reservoirs start empty (M = 0); cleared on the first frame that uses them
    reservoirsNeedClear = true;
}

void RayTracingPipeline::createUniformBuffer(const Context& context)
{
    // One SceneData slot per frame in flight
    vk::DeviceSize alignment = context.physicalDevice.getProperties().limits.minUniformBufferOffsetAlignment;
    uniformBufferStride = (sizeof(SceneData) + alignment - 1) & ~(alignment - 1);

    vk::DeviceSize bufferSize = uniformBufferStride * MAX_FRAMES_IN_FLIGHT;
    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eUniformBuffer;
    vk::MemoryPropertyFlags properties = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

    gpu::Buffers::createBuffer(context, bufferSize, usage, properties, uniformBuffer, uniformBufferMemory, false);

    uniformBufferMapped = context.device.mapMemory(uniformBufferMemory, 0, bufferSize);
}

void RayTracingPipeline::updateUniformBuffer(const SceneData& sceneData, uint32_t currentFrame)
{
    memcpy(static_cast<char*>(uniformBufferMapped) + currentFrame * uniformBufferStride, &sceneData, sizeof(sceneData));
}

void RayTracingPipeline::createDescriptorPool(const Context& context)
{
    std::array<vk::DescriptorPoolSize, 5> poolSizes{};
    poolSizes[0].type = vk::DescriptorType::eAccelerationStructureKHR;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = vk::DescriptorType::eStorageImage;
    poolSizes[1].descriptorCount = 1;
    poolSizes[2].type = vk::DescriptorType::eUniformBufferDynamic;
    poolSizes[2].descriptorCount = 1;
    poolSizes[3].type = vk::DescriptorType::eStorageBuffer;
    poolSizes[3].descriptorCount = 6; // vertex + index + mesh + instance + 2 reservoirs
    poolSizes[4].type = vk::DescriptorType::eCombinedImageSampler;
    // MAX_MESHES * 4 for albedo + normals + roughness + metalness
    // + 6 for GBuffer (incl. velocity), + 1 for last image
    poolSizes[4].descriptorCount = MAX_MESHES * 4 + 6 + 1;

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;

    descriptorPool = context.device.createDescriptorPool(poolInfo);
}

void RayTracingPipeline::createRayTracingResources(const Context& context, CommandBuffers& commandBufferManager, vk::AccelerationStructureKHR tlas, const std::vector<Model>& models, std::vector<vk::ImageView>& outAlbedoTextureViews, std::vector<vk::ImageView>& outNormalTextureViews, std::vector<vk::ImageView>& outRoughnessTextureViews, std::vector<vk::ImageView>& outMetalnessTextureViews)
{
    // Compute sizes across all submeshes
    size_t totalVertices = 0;
    size_t totalIndices = 0;
    size_t totalMeshes = 0;
    size_t totalModels = models.size();

    for (const auto& model : models)
    {
        for (const auto& shadedMesh : model.shadedMeshes)
        {
            totalVertices += shadedMesh.mesh.vertices.size();
            totalIndices += shadedMesh.mesh.indices.size();
            totalMeshes++;
        }
    }

    // Create combined buffers
    std::vector<Vertex> allVertices;
    std::vector<uint32_t> allIndices;

    allVertices.reserve(totalVertices);
    allIndices.reserve(totalIndices);

    std::vector<MeshData> allMeshData;
    std::vector<InstanceData> allInstanceData;

    allMeshData.reserve(totalMeshes);
    allInstanceData.reserve(totalModels);

    uint32_t vertexOffset = 0;
    uint32_t indexOffset = 0;
    uint32_t meshOffset = 0;

    // Process all models and their submeshes
    for (const auto& model : models)
    {
        // Safer to assign attributes explicitly since the struct might change
        InstanceData instanceData;
        instanceData.normalMatrix = glm::transpose(glm::inverse(model.transform.getTransformMatrix()));
        instanceData.meshOffset = meshOffset;
        allInstanceData.push_back(instanceData);

        for (const auto& shadedMesh : model.shadedMeshes)
        {
            const Mesh& mesh = shadedMesh.mesh;

            // Store mesh data
            MeshData meshData;
            meshData.indexOffset = indexOffset;
            meshData.vertexOffset = vertexOffset;
            allMeshData.push_back(meshData);

            // Collect vertex and index data
            allVertices.insert(allVertices.end(), mesh.vertices.begin(), mesh.vertices.end());
            allIndices.insert(allIndices.end(), mesh.indices.begin(), mesh.indices.end());

            // Collect textures from material (already resolved to the error textures on failure)
            const Material& material = shadedMesh.material;
            outAlbedoTextureViews.push_back(material.albedoMap->imageView);
            outNormalTextureViews.push_back(material.normalMap->imageView);
            outRoughnessTextureViews.push_back(material.roughnessMap->imageView);
            outMetalnessTextureViews.push_back(material.metalnessMap->imageView);

            // Update offsets
            vertexOffset += static_cast<uint32_t>(mesh.vertices.size());
            indexOffset += static_cast<uint32_t>(mesh.indices.size());
            meshOffset++;
        }
    }

    // Create vertex buffer
    vk::BufferUsageFlags vertexUsageFlags = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
    vk::MemoryPropertyFlags vertexMemoryFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    gpu::Buffers::createAndFillBuffer<Vertex>(context, commandBufferManager, allVertices, globalVertexBuffer, globalVertexBufferMemory, vertexUsageFlags, vertexMemoryFlags, false);

    // Create index buffer
    vk::BufferUsageFlags indexUsageFlags = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
    vk::MemoryPropertyFlags indexMemoryFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    gpu::Buffers::createAndFillBuffer<uint32_t>(context, commandBufferManager, allIndices, globalIndexBuffer, globalIndexBufferMemory, indexUsageFlags, indexMemoryFlags, false);

    // Create mesh data buffer
    vk::BufferUsageFlags meshDataUsageFlags = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
    vk::MemoryPropertyFlags meshDataMemoryFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    gpu::Buffers::createAndFillBuffer<MeshData>(context, commandBufferManager, allMeshData, meshDataBuffer, meshDataBufferMemory, meshDataUsageFlags, meshDataMemoryFlags, false);

    // Create offset buffer
    vk::BufferUsageFlags offsetUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
    vk::MemoryPropertyFlags offsetMemory = vk::MemoryPropertyFlagBits::eDeviceLocal;
    gpu::Buffers::createAndFillBuffer<InstanceData>(context, commandBufferManager, allInstanceData, instanceDataBuffer, instanceDataBufferMemory, offsetUsage, offsetMemory, false);

    // Create samplers
    globalTextureSampler = gpu::Textures::createSampler(context, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear);
    pointTextureSampler = gpu::Textures::createSampler(context, vk::Filter::eNearest, vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest);
}

void RayTracingPipeline::createDescriptorSet(const Context& context)
{
    vk::DescriptorSetLayout layout = DescriptorLayouts::getRayTracingLayout();
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &layout;

    descriptorSet = context.device.allocateDescriptorSets(allocInfo).front();
}

void RayTracingPipeline::writeDescriptorSet(const Context& context, vk::ImageView depthImageView, vk::ImageView normalsImageView, vk::ImageView albedoImageView, vk::ImageView roughnessImageView, vk::ImageView metalnessImageView, vk::ImageView velocityImageView, vk::AccelerationStructureKHR tlas, const std::vector<vk::ImageView>& albedoTextureViews, const std::vector<vk::ImageView>& normalTextureViews, const std::vector<vk::ImageView>& roughnessTextureViews, const std::vector<vk::ImageView>& metalnessTextureViews)
{
    std::vector<vk::WriteDescriptorSet> descriptorWrites;

    // TLAS
    vk::WriteDescriptorSetAccelerationStructureKHR accelerationStructureDescriptor{};
    accelerationStructureDescriptor.accelerationStructureCount = 1;
    accelerationStructureDescriptor.pAccelerationStructures = &tlas;

    vk::WriteDescriptorSet tlasWrite{};
    tlasWrite.pNext = &accelerationStructureDescriptor;
    tlasWrite.dstSet = descriptorSet;
    tlasWrite.dstBinding = 0;
    tlasWrite.dstArrayElement = 0;
    tlasWrite.descriptorType = vk::DescriptorType::eAccelerationStructureKHR;
    tlasWrite.descriptorCount = 1;
    descriptorWrites.push_back(tlasWrite);

    // Storage Image
    vk::DescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = vk::ImageLayout::eGeneral;
    imageInfo.imageView = storageImageView;
    imageInfo.sampler = nullptr;

    vk::WriteDescriptorSet imageWrite{};
    imageWrite.dstSet = descriptorSet;
    imageWrite.dstBinding = 1;
    imageWrite.dstArrayElement = 0;
    imageWrite.descriptorType = vk::DescriptorType::eStorageImage;
    imageWrite.descriptorCount = 1;
    imageWrite.pImageInfo = &imageInfo;
    descriptorWrites.push_back(imageWrite);

    // Uniform Buffer
    vk::DescriptorBufferInfo uniformBufferInfo{};
    uniformBufferInfo.buffer = uniformBuffer;
    uniformBufferInfo.offset = 0;
    uniformBufferInfo.range = sizeof(SceneData);

    vk::WriteDescriptorSet uniformWrite{};
    uniformWrite.dstSet = descriptorSet;
    uniformWrite.dstBinding = 2;
    uniformWrite.dstArrayElement = 0;
    uniformWrite.descriptorType = vk::DescriptorType::eUniformBufferDynamic;
    uniformWrite.descriptorCount = 1;
    uniformWrite.pBufferInfo = &uniformBufferInfo;
    descriptorWrites.push_back(uniformWrite);

    // Vertex Buffer
    vk::DescriptorBufferInfo vertexBufferInfo{};
    vertexBufferInfo.buffer = globalVertexBuffer;
    vertexBufferInfo.offset = 0;
    vertexBufferInfo.range = VK_WHOLE_SIZE;

    vk::WriteDescriptorSet vertexWrite{};
    vertexWrite.dstSet = descriptorSet;
    vertexWrite.dstBinding = 3;
    vertexWrite.dstArrayElement = 0;
    vertexWrite.descriptorType = vk::DescriptorType::eStorageBuffer;
    vertexWrite.descriptorCount = 1;
    vertexWrite.pBufferInfo = &vertexBufferInfo;
    descriptorWrites.push_back(vertexWrite);

    // Index Buffer
    vk::DescriptorBufferInfo indexBufferInfo{};
    indexBufferInfo.buffer = globalIndexBuffer;
    indexBufferInfo.offset = 0;
    indexBufferInfo.range = VK_WHOLE_SIZE;

    vk::WriteDescriptorSet indexWrite{};
    indexWrite.dstSet = descriptorSet;
    indexWrite.dstBinding = 4;
    indexWrite.dstArrayElement = 0;
    indexWrite.descriptorType = vk::DescriptorType::eStorageBuffer;
    indexWrite.descriptorCount = 1;
    indexWrite.pBufferInfo = &indexBufferInfo;
    descriptorWrites.push_back(indexWrite);

    // Mesh data Buffer
    vk::DescriptorBufferInfo meshDataBufferInfo{};
    meshDataBufferInfo.buffer = meshDataBuffer;
    meshDataBufferInfo.offset = 0;
    meshDataBufferInfo.range = VK_WHOLE_SIZE;

    vk::WriteDescriptorSet meshDataWrite{};
    meshDataWrite.dstSet = descriptorSet;
    meshDataWrite.dstBinding = 5;
    meshDataWrite.dstArrayElement = 0;
    meshDataWrite.descriptorType = vk::DescriptorType::eStorageBuffer;
    meshDataWrite.descriptorCount = 1;
    meshDataWrite.pBufferInfo = &meshDataBufferInfo;
    descriptorWrites.push_back(meshDataWrite);

    // Instance data Buffer
    vk::DescriptorBufferInfo instanceDataBufferInfo{};
    instanceDataBufferInfo.buffer = instanceDataBuffer;
    instanceDataBufferInfo.offset = 0;
    instanceDataBufferInfo.range = VK_WHOLE_SIZE;

    vk::WriteDescriptorSet instanceDataWrite{};
    instanceDataWrite.dstSet = descriptorSet;
    instanceDataWrite.dstBinding = 6;
    instanceDataWrite.dstArrayElement = 0;
    instanceDataWrite.descriptorType = vk::DescriptorType::eStorageBuffer;
    instanceDataWrite.descriptorCount = 1;
    instanceDataWrite.pBufferInfo = &instanceDataBufferInfo;
    descriptorWrites.push_back(instanceDataWrite);

    if (std::max(albedoTextureViews.size(), normalTextureViews.size()) > MAX_MESHES)
    {
        std::cerr << "WARNING: reached maximum mesh count !" << std::endl;
    }

    // Material textures
    std::vector<vk::DescriptorImageInfo> albedoTextureInfos(MAX_MESHES);
    std::vector<vk::DescriptorImageInfo> normalTextureInfos(MAX_MESHES);
    std::vector<vk::DescriptorImageInfo> roughnessTextureInfos(MAX_MESHES);
    std::vector<vk::DescriptorImageInfo> metalnessTextureInfos(MAX_MESHES);

    for (size_t i = 0; i < MAX_MESHES; i++)
    {
        // Albedo
        albedoTextureInfos[i].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        albedoTextureInfos[i].sampler = globalTextureSampler;
        if (i < albedoTextureViews.size())
        {
            albedoTextureInfos[i].imageView = albedoTextureViews[i];
        }
        else
        {
            // TODO: find better alternative
            albedoTextureInfos[i].imageView = TextureManager::errorAlbedoTexture.imageView;
        }

        // Normals
        normalTextureInfos[i].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        normalTextureInfos[i].sampler = globalTextureSampler;
        if (i < normalTextureViews.size())
        {
            normalTextureInfos[i].imageView = normalTextureViews[i];
        }
        else
        {
            normalTextureInfos[i].imageView = TextureManager::errorNormalTexture.imageView;
        }

        // Roughness
        roughnessTextureInfos[i].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        roughnessTextureInfos[i].sampler = globalTextureSampler;
        if (i < roughnessTextureViews.size())
        {
            roughnessTextureInfos[i].imageView = roughnessTextureViews[i];
        }
        else
        {
            roughnessTextureInfos[i].imageView = TextureManager::errorRoughnessTexture.imageView;
        }

        // Metalness
        metalnessTextureInfos[i].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        metalnessTextureInfos[i].sampler = globalTextureSampler;
        if (i < metalnessTextureViews.size())
        {
            metalnessTextureInfos[i].imageView = metalnessTextureViews[i];
        }
        else
        {
            metalnessTextureInfos[i].imageView = TextureManager::errorMetalnessTexture.imageView;
        }
    }

    vk::WriteDescriptorSet albedoWrite{};
    albedoWrite.dstSet = descriptorSet;
    albedoWrite.dstBinding = 7;
    albedoWrite.dstArrayElement = 0;
    albedoWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    albedoWrite.descriptorCount = MAX_MESHES;
    albedoWrite.pImageInfo = albedoTextureInfos.data();
    descriptorWrites.push_back(albedoWrite);

    vk::WriteDescriptorSet normalWrite{};
    normalWrite.dstSet = descriptorSet;
    normalWrite.dstBinding = 8;
    normalWrite.dstArrayElement = 0;
    normalWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    normalWrite.descriptorCount = MAX_MESHES;
    normalWrite.pImageInfo = normalTextureInfos.data();
    descriptorWrites.push_back(normalWrite);

    vk::WriteDescriptorSet roughnessWrite{};
    roughnessWrite.dstSet = descriptorSet;
    roughnessWrite.dstBinding = 9;
    roughnessWrite.dstArrayElement = 0;
    roughnessWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    roughnessWrite.descriptorCount = MAX_MESHES;
    roughnessWrite.pImageInfo = roughnessTextureInfos.data();
    descriptorWrites.push_back(roughnessWrite);

    vk::WriteDescriptorSet metalnessWrite{};
    metalnessWrite.dstSet = descriptorSet;
    metalnessWrite.dstBinding = 10;
    metalnessWrite.dstArrayElement = 0;
    metalnessWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    metalnessWrite.descriptorCount = MAX_MESHES;
    metalnessWrite.pImageInfo = metalnessTextureInfos.data();
    descriptorWrites.push_back(metalnessWrite);

    // Depth
    vk::DescriptorImageInfo depthInfos;
    depthInfos.imageLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
    depthInfos.imageView = depthImageView;
    depthInfos.sampler = globalTextureSampler;

    vk::WriteDescriptorSet depthWrite{};
    depthWrite.dstSet = descriptorSet;
    depthWrite.dstBinding = 11;
    depthWrite.dstArrayElement = 0;
    depthWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    depthWrite.descriptorCount = 1;
    depthWrite.pImageInfo = &depthInfos;

    descriptorWrites.push_back(depthWrite);

    // G-Normals
    vk::DescriptorImageInfo normalsInfos;
    normalsInfos.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    normalsInfos.imageView = normalsImageView;
    normalsInfos.sampler = globalTextureSampler;

    vk::WriteDescriptorSet gBufferNormalsWrite{};
    gBufferNormalsWrite.dstSet = descriptorSet;
    gBufferNormalsWrite.dstBinding = 12;
    gBufferNormalsWrite.dstArrayElement = 0;
    gBufferNormalsWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    gBufferNormalsWrite.descriptorCount = 1;
    gBufferNormalsWrite.pImageInfo = &normalsInfos;

    descriptorWrites.push_back(gBufferNormalsWrite);

    // G-Albedo
    vk::DescriptorImageInfo gBufferAlbedoInfos;
    gBufferAlbedoInfos.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    gBufferAlbedoInfos.imageView = albedoImageView;
    gBufferAlbedoInfos.sampler = globalTextureSampler;

    vk::WriteDescriptorSet gBufferAlbedoWrite{};
    gBufferAlbedoWrite.dstSet = descriptorSet;
    gBufferAlbedoWrite.dstBinding = 13;
    gBufferAlbedoWrite.dstArrayElement = 0;
    gBufferAlbedoWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    gBufferAlbedoWrite.descriptorCount = 1;
    gBufferAlbedoWrite.pImageInfo = &gBufferAlbedoInfos;

    descriptorWrites.push_back(gBufferAlbedoWrite);

    // G-Roughness
    vk::DescriptorImageInfo gBufferRoughnessInfos;
    gBufferRoughnessInfos.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    gBufferRoughnessInfos.imageView = roughnessImageView;
    gBufferRoughnessInfos.sampler = globalTextureSampler;

    vk::WriteDescriptorSet gBufferRoughnessWrite{};
    gBufferRoughnessWrite.dstSet = descriptorSet;
    gBufferRoughnessWrite.dstBinding = 14;
    gBufferRoughnessWrite.dstArrayElement = 0;
    gBufferRoughnessWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    gBufferRoughnessWrite.descriptorCount = 1;
    gBufferRoughnessWrite.pImageInfo = &gBufferRoughnessInfos;

    descriptorWrites.push_back(gBufferRoughnessWrite);

    // G-Metalness
    vk::DescriptorImageInfo gBufferMetalnessInfos;
    gBufferMetalnessInfos.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    gBufferMetalnessInfos.imageView = metalnessImageView;
    gBufferMetalnessInfos.sampler = globalTextureSampler;

    vk::WriteDescriptorSet gBufferMetalnessWrite{};
    gBufferMetalnessWrite.dstSet = descriptorSet;
    gBufferMetalnessWrite.dstBinding = 15;
    gBufferMetalnessWrite.dstArrayElement = 0;
    gBufferMetalnessWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    gBufferMetalnessWrite.descriptorCount = 1;
    gBufferMetalnessWrite.pImageInfo = &gBufferMetalnessInfos;

    descriptorWrites.push_back(gBufferMetalnessWrite);


    // Frame accumulation
    vk::DescriptorImageInfo lastImageInfos;
    lastImageInfos.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    lastImageInfos.imageView = last_storageImageView;
    lastImageInfos.sampler = pointTextureSampler;

    vk::WriteDescriptorSet lastImageWrite{};
    lastImageWrite.dstSet = descriptorSet;
    lastImageWrite.dstBinding = 16;
    lastImageWrite.dstArrayElement = 0;
    lastImageWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    lastImageWrite.descriptorCount = 1;
    lastImageWrite.pImageInfo = &lastImageInfos;

    descriptorWrites.push_back(lastImageWrite);

    // Reservoir buffers (ReSTIR temporal reuse)
    vk::DescriptorBufferInfo reservoirBufferInfoA{};
    reservoirBufferInfoA.buffer = reservoirBuffers[0];
    reservoirBufferInfoA.offset = 0;
    reservoirBufferInfoA.range = VK_WHOLE_SIZE;

    vk::WriteDescriptorSet reservoirWriteA{};
    reservoirWriteA.dstSet = descriptorSet;
    reservoirWriteA.dstBinding = 17;
    reservoirWriteA.dstArrayElement = 0;
    reservoirWriteA.descriptorType = vk::DescriptorType::eStorageBuffer;
    reservoirWriteA.descriptorCount = 1;
    reservoirWriteA.pBufferInfo = &reservoirBufferInfoA;
    descriptorWrites.push_back(reservoirWriteA);

    vk::DescriptorBufferInfo reservoirBufferInfoB{};
    reservoirBufferInfoB.buffer = reservoirBuffers[1];
    reservoirBufferInfoB.offset = 0;
    reservoirBufferInfoB.range = VK_WHOLE_SIZE;

    vk::WriteDescriptorSet reservoirWriteB{};
    reservoirWriteB.dstSet = descriptorSet;
    reservoirWriteB.dstBinding = 18;
    reservoirWriteB.dstArrayElement = 0;
    reservoirWriteB.descriptorType = vk::DescriptorType::eStorageBuffer;
    reservoirWriteB.descriptorCount = 1;
    reservoirWriteB.pBufferInfo = &reservoirBufferInfoB;
    descriptorWrites.push_back(reservoirWriteB);

    // Motion vectors
    vk::DescriptorImageInfo velocityInfo{};
    velocityInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    velocityInfo.imageView = velocityImageView;
    velocityInfo.sampler = globalTextureSampler;

    vk::WriteDescriptorSet velocityWrite{};
    velocityWrite.dstSet = descriptorSet;
    velocityWrite.dstBinding = 19;
    velocityWrite.dstArrayElement = 0;
    velocityWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    velocityWrite.descriptorCount = 1;
    velocityWrite.pImageInfo = &velocityInfo;
    descriptorWrites.push_back(velocityWrite);

    context.device.updateDescriptorSets(descriptorWrites, {});
}

void RayTracingPipeline::traceRays(vk::CommandBuffer commandBuffer, uint32_t frameCount, uint32_t currentFrame)
{
    sampleCount = Settings::spp * frameCount;

    // TODO: use or create gpu function
    vk::ImageMemoryBarrier barrier1{};
    barrier1.oldLayout = vk::ImageLayout::eUndefined;
    barrier1.newLayout = vk::ImageLayout::eGeneral;
    barrier1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier1.image = storageImage;
    barrier1.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
    barrier1.srcAccessMask = {};
    barrier1.dstAccessMask = vk::AccessFlagBits::eShaderWrite;
    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eRayTracingShaderKHR, {}, {}, {}, barrier1);

    vk::ImageMemoryBarrier barrier2{};
    barrier2.oldLayout = vk::ImageLayout::eUndefined;
    barrier2.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier2.image = last_storageImage;
    barrier2.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
    barrier2.srcAccessMask = {};
    barrier2.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eRayTracingShaderKHR, {}, {}, {}, barrier2);

    // Reservoir buffers: zero on first use, otherwise make the previous frame's
    // writes visible to this frame's reads (they alternate read/write via parity)
    if (reservoirsNeedClear)
    {
        for (int i = 0; i < 2; i++)
        {
            commandBuffer.fillBuffer(reservoirBuffers[i], 0, reservoirBufferSize, 0u);
        }

        vk::MemoryBarrier clearBarrier{};
        clearBarrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        clearBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eRayTracingShaderKHR, {}, clearBarrier, {}, {});

        reservoirsNeedClear = false;
    }
    else
    {
        vk::MemoryBarrier reservoirBarrier{};
        reservoirBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
        reservoirBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eRayTracingShaderKHR, vk::PipelineStageFlagBits::eRayTracingShaderKHR, {}, reservoirBarrier, {}, {});
    }

    // Bind pipeline
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, pipeline);
    uint32_t sceneDataOffset = static_cast<uint32_t>(currentFrame * uniformBufferStride);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, pipelineLayout, 0, descriptorSet, sceneDataOffset);

    PushConstants push;
    push.frameCount = frameCount;
    static std::random_device rd;
    push.rng = rd();

    // Push constants
    commandBuffer.pushConstants<PushConstants>(
        pipelineLayout,
        vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eMissKHR,
        0,
        push);


    if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdBeginDebugUtilsLabelEXT)
    {
        vk::DebugUtilsLabelEXT labelInfo{};
        labelInfo.pLabelName = "RTX ON";
        labelInfo.color[0] = 1.0f;
        labelInfo.color[1] = 0.0f;
        labelInfo.color[2] = 0.0f;
        labelInfo.color[3] = 1.0f;
        commandBuffer.beginDebugUtilsLabelEXT(labelInfo);
    }

    // RAY TRACE !
    commandBuffer.traceRaysKHR(raygenSbtEntry, missSbtEntry, hitSbtEntry, callableSbtEntry, storageImageWidth, storageImageHeight, 1);

    if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdEndDebugUtilsLabelEXT)
    {
        commandBuffer.endDebugUtilsLabelEXT();
    }
}

void RayTracingPipeline::cleanup(vk::Device device)
{
    if (uniformBufferMapped)
    {
        device.unmapMemory(uniformBufferMemory);
    }
    if (descriptorPool)
    {
        device.destroyDescriptorPool(descriptorPool);
    }
    if (uniformBuffer)
    {
        device.destroyBuffer(uniformBuffer);
    }
    if (uniformBufferMemory)
    {
        device.freeMemory(uniformBufferMemory);
    }
    if (storageImageView)
    {
        device.destroyImageView(storageImageView);
    }
    if (storageImage)
    {
        device.destroyImage(storageImage);
    }
    if (storageImageMemory)
    {
        device.freeMemory(storageImageMemory);
    }
    if (last_storageImageView)
    {
        device.destroyImageView(last_storageImageView);
    }
    if (last_storageImage)
    {
        device.destroyImage(last_storageImage);
    }
    if (last_storageImageMemory)
    {
        device.freeMemory(last_storageImageMemory);
    }

    for (int i = 0; i < 2; i++)
    {
        if (reservoirBuffers[i])
        {
            device.destroyBuffer(reservoirBuffers[i]);
        }
        if (reservoirBufferMemories[i])
        {
            device.freeMemory(reservoirBufferMemories[i]);
        }
    }

    if (globalVertexBuffer)
    {
        device.destroyBuffer(globalVertexBuffer);
    }
    if (globalVertexBufferMemory)
    {
        device.freeMemory(globalVertexBufferMemory);
    }
    if (globalIndexBuffer)
    {
        device.destroyBuffer(globalIndexBuffer);
    }
    if (globalIndexBufferMemory)
    {
        device.freeMemory(globalIndexBufferMemory);
    }
    if (meshDataBuffer)
    {
        device.destroyBuffer(meshDataBuffer);
    }
    if (meshDataBufferMemory)
    {
        device.freeMemory(meshDataBufferMemory);
    }
    if (instanceDataBuffer)
    {
        device.destroyBuffer(instanceDataBuffer);
    }
    if (instanceDataBufferMemory)
    {
        device.freeMemory(instanceDataBufferMemory);
    }
    if (globalTextureSampler)
    {
        device.destroySampler(globalTextureSampler);
        globalTextureSampler = nullptr;
    }
    if (pointTextureSampler)
    {
        device.destroySampler(pointTextureSampler);
        pointTextureSampler = nullptr;
    }

    if (sbtBuffer)
    {
        device.destroyBuffer(sbtBuffer);
    }
    if (sbtBufferMemory)
    {
        device.freeMemory(sbtBufferMemory);
    }
    if (pipeline)
    {
        device.destroyPipeline(pipeline);
    }
    if (pipelineLayout)
    {
        device.destroyPipelineLayout(pipelineLayout);
    }
}

// Getters
vk::ImageView RayTracingPipeline::getStorageImageView() const
{
    return storageImageView;
}

vk::ImageView RayTracingPipeline::getLastStorageImageView() const
{
    return last_storageImageView;
}

vk::Image RayTracingPipeline::getStorageImage() const
{
    return storageImage;
}

vk::Image RayTracingPipeline::getLastStorageImage() const
{
    return last_storageImage;
}

vk::DescriptorSet RayTracingPipeline::getDescriptorSet() const
{
    return descriptorSet;
}

vk::PipelineLayout RayTracingPipeline::getPipelineLayout() const
{
    return pipelineLayout;
}

uint32_t RayTracingPipeline::getStorageImageWidth() const
{
    return storageImageWidth;
}

uint32_t RayTracingPipeline::getStorageImageHeight() const
{
    return storageImageHeight;
}

} // namespace render
} // namespace vkrt
