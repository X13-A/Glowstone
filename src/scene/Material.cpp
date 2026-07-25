#include "scene/Material.hpp"
#include "gpu/DescriptorLayouts.hpp"
#include <array>
#include <stdexcept>
#include "assets/TextureManager.hpp"
#include <iostream>


namespace vkrt {
namespace scene {
using namespace vkrt::assets;
using namespace vkrt::gpu;

void Material::init(const PBRMaterialInfo& info, const Context& context, CommandBuffers& commandBufferManager, vk::DescriptorPool descriptorPool, bool hasError)
{
    this->hasError = hasError;

    if (hasError)
    {
        albedoMap = &TextureManager::errorAlbedoTexture;
        normalMap = &TextureManager::errorNormalTexture;
        roughnessMap = &TextureManager::errorRoughnessTexture;
        metalnessMap = &TextureManager::errorMetalnessTexture;
    }
    else
    {
        // Fall back to a solid texel built from the material factor when a map is absent
        albedoMap = info.albedoTexture.empty()
            ? &TextureManager::acquireSolidRGBA(
                static_cast<uint8_t>(info.albedoFactor[0] * 255),
                static_cast<uint8_t>(info.albedoFactor[1] * 255),
                static_cast<uint8_t>(info.albedoFactor[2] * 255),
                0, ALBEDO_MAP_FORMAT, context, commandBufferManager)
            : &TextureManager::acquire(info.albedoTexture, ALBEDO_MAP_FORMAT, context, commandBufferManager);

        normalMap = info.normalTexture.empty()
            ? &TextureManager::acquireSolidRGBA(128, 128, 255, 0, NORMAL_MAP_FORMAT, context, commandBufferManager)
            : &TextureManager::acquire(info.normalTexture, NORMAL_MAP_FORMAT, context, commandBufferManager);

        roughnessMap = info.roughnessTexture.empty()
            ? &TextureManager::acquireSolidR(static_cast<uint8_t>(info.roughnessFactor * 255), ROUGHNESS_MAP_FORMAT, context, commandBufferManager)
            : &TextureManager::acquire(info.roughnessTexture, ROUGHNESS_MAP_FORMAT, context, commandBufferManager);

        metalnessMap = info.metallicTexture.empty()
            ? &TextureManager::acquireSolidR(static_cast<uint8_t>(info.metallicFactor * 255), METALNESS_MAP_FORMAT, context, commandBufferManager)
            : &TextureManager::acquire(info.metallicTexture, METALNESS_MAP_FORMAT, context, commandBufferManager);
    }

    createDescriptorSet(context, DescriptorLayouts::getMaterialLayout(), descriptorPool);
}

void Material::createDescriptorSet(const Context& context, vk::DescriptorSetLayout materialDescriptorSetLayout, vk::DescriptorPool descriptorPool)
{
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &materialDescriptorSetLayout;

    descriptorSet = context.device.allocateDescriptorSets(allocInfo).front();

    const std::array<const Texture*, 4> maps = { albedoMap, normalMap, roughnessMap, metalnessMap };

    std::array<vk::DescriptorImageInfo, 4> imageInfos{};
    std::array<vk::WriteDescriptorSet, 4> descriptorWrites{};

    for (uint32_t binding = 0; binding < maps.size(); binding++)
    {
        imageInfos[binding].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        imageInfos[binding].imageView = maps[binding]->imageView;
        imageInfos[binding].sampler = maps[binding]->sampler;

        descriptorWrites[binding].dstSet = descriptorSet;
        descriptorWrites[binding].dstBinding = binding;
        descriptorWrites[binding].dstArrayElement = 0;
        descriptorWrites[binding].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        descriptorWrites[binding].descriptorCount = 1;
        descriptorWrites[binding].pImageInfo = &imageInfos[binding];
    }

    context.device.updateDescriptorSets(descriptorWrites, {});
}

void Material::cleanup(vk::Device device)
{
    // Textures are owned by TextureManager and the set is freed with its pool
    albedoMap = nullptr;
    normalMap = nullptr;
    roughnessMap = nullptr;
    metalnessMap = nullptr;
    descriptorSet = nullptr;
}

} // namespace scene
} // namespace vkrt
