#include "render/FullScreenQuad.hpp"
#include <stdexcept>
#include "gpu/Utils.hpp"
#include "gpu/DescriptorLayouts.hpp"
#include <iostream>



namespace vkrt {
namespace render {
using namespace vkrt::gpu;

void FullScreenQuad::init(const Context& context, CommandBuffers& commandBufferManager, vk::DescriptorPool pool, vk::ImageView depthImageView, vk::ImageView normalImageView, vk::ImageView albedoImageView, vk::ImageView roughnessImageView, vk::ImageView metalnessImageView, vk::ImageView velocityImageView)
{
    vertices = {
        // Position                      // Texture Coordinates // Normal
        { glm::vec3(-1.0f,  1.0f, 0.0f), glm::vec2(0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f) },
        { glm::vec3(1.0f,  1.0f, 0.0f),  glm::vec2(1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f) },
        { glm::vec3(-1.0f, -1.0f, 0.0f), glm::vec2(0.0f, 1.0f), glm::vec3(0.0f, 0.0f, 1.0f) },

        { glm::vec3(-1.0f, -1.0f, 0.0f), glm::vec2(0.0f, 1.0f), glm::vec3(0.0f, 0.0f, 1.0f) },
        { glm::vec3(1.0f,  1.0f, 0.0f),  glm::vec2(1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f) },
        { glm::vec3(1.0f, -1.0f, 0.0f),  glm::vec2(1.0f, 1.0f), glm::vec3(0.0f, 0.0f, 1.0f) }
    };

    gBufferSampler = gpu::Textures::createSampler(context, vk::Filter::eNearest, vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest);
    vk::BufferUsageFlags usageFlags = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;
    vk::MemoryPropertyFlags memoryFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    gpu::Buffers::createAndFillBuffer<Vertex>(context, commandBufferManager, vertices, vertexBuffer, vertexBufferMemory, usageFlags, memoryFlags);

    createUniformBuffers(context);
    createDescriptorSets(context, pool);
    writeDescriptorSets(context, depthImageView, normalImageView, albedoImageView, roughnessImageView, metalnessImageView, velocityImageView);
}

void FullScreenQuad::createDescriptorSets(const Context& context, vk::DescriptorPool pool)
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, DescriptorLayouts::getFullScreenQuadLayout());
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = pool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets = context.device.allocateDescriptorSets(allocInfo);
}

void FullScreenQuad::writeDescriptorSets(const Context& context, vk::ImageView depthImageView, vk::ImageView normalImageView, vk::ImageView albedoImageView, vk::ImageView roughnessImageView, vk::ImageView metalnessImageView, vk::ImageView velocityImageView)
{
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vk::DescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(FullScreenQuadUBO);

        // Depth image sampler
        vk::DescriptorImageInfo depthImageInfo{};
        depthImageInfo.imageLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
        depthImageInfo.imageView = depthImageView;
        depthImageInfo.sampler = gBufferSampler;

        // Normal image sampler
        vk::DescriptorImageInfo normalImageInfo{};
        normalImageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        normalImageInfo.imageView = normalImageView;
        normalImageInfo.sampler = gBufferSampler;

        // Albedo image sampler
        vk::DescriptorImageInfo albedoImageInfo{};
        albedoImageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        albedoImageInfo.imageView = albedoImageView;
        albedoImageInfo.sampler = gBufferSampler;

        // Roughness image sampler
        vk::DescriptorImageInfo roughnessImageInfo{};
        roughnessImageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        roughnessImageInfo.imageView = roughnessImageView;
        roughnessImageInfo.sampler = gBufferSampler;

        // Metalness image sampler
        vk::DescriptorImageInfo metalnessImageInfo{};
        metalnessImageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        metalnessImageInfo.imageView = metalnessImageView;
        metalnessImageInfo.sampler = gBufferSampler;

        // Velocity image sampler
        vk::DescriptorImageInfo velocityImageInfo{};
        velocityImageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        velocityImageInfo.imageView = velocityImageView;
        velocityImageInfo.sampler = gBufferSampler;

        std::array<vk::WriteDescriptorSet, 7> descriptorWrites{};

        descriptorWrites[0].dstSet = descriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        descriptorWrites[1].dstSet = descriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &depthImageInfo;

        descriptorWrites[2].dstSet = descriptorSets[i];
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].dstArrayElement = 0;
        descriptorWrites[2].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].pImageInfo = &normalImageInfo;

        descriptorWrites[3].dstSet = descriptorSets[i];
        descriptorWrites[3].dstBinding = 3;
        descriptorWrites[3].dstArrayElement = 0;
        descriptorWrites[3].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        descriptorWrites[3].descriptorCount = 1;
        descriptorWrites[3].pImageInfo = &albedoImageInfo;

        descriptorWrites[4].dstSet = descriptorSets[i];
        descriptorWrites[4].dstBinding = 4;
        descriptorWrites[4].dstArrayElement = 0;
        descriptorWrites[4].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        descriptorWrites[4].descriptorCount = 1;
        descriptorWrites[4].pImageInfo = &roughnessImageInfo;

        descriptorWrites[5].dstSet = descriptorSets[i];
        descriptorWrites[5].dstBinding = 5;
        descriptorWrites[5].dstArrayElement = 0;
        descriptorWrites[5].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        descriptorWrites[5].descriptorCount = 1;
        descriptorWrites[5].pImageInfo = &metalnessImageInfo;

        descriptorWrites[6].dstSet = descriptorSets[i];
        descriptorWrites[6].dstBinding = 6;
        descriptorWrites[6].dstArrayElement = 0;
        descriptorWrites[6].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        descriptorWrites[6].descriptorCount = 1;
        descriptorWrites[6].pImageInfo = &velocityImageInfo;

        context.device.updateDescriptorSets(descriptorWrites, {});
    }
}

void FullScreenQuad::createUniformBuffers(const Context& context)
{
    vk::DeviceSize bufferSize = sizeof(FullScreenQuadUBO);

    uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        gpu::Buffers::createBuffer(context, bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, uniformBuffers[i], uniformBuffersMemory[i]);
        uniformBuffersMapped[i] = context.device.mapMemory(uniformBuffersMemory[i], 0, bufferSize);
    }
}

void FullScreenQuad::cleanup(vk::Device device)
{
    device.destroySampler(gBufferSampler);
    device.destroyBuffer(vertexBuffer);
    device.freeMemory(vertexBufferMemory);
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        device.destroyBuffer(uniformBuffers[i]);
        device.freeMemory(uniformBuffersMemory[i]);
    }
}

} // namespace render
} // namespace vkrt
