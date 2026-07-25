#include "render/pipeline/GraphicsPipeline.hpp"
#include <array>
#include <iostream>
#include "core/Settings.hpp"


namespace vkrt {
namespace render {
using namespace vkrt::core;
using namespace vkrt::gpu;

void GraphicsPipeline::initPipelines(int nativeWidth, int nativeHeight, int scaledWidth, int scaledHeight, const Context& context, CommandBuffers& commandBufferManager, vk::Format swapChainImageFormat)
{
    gBufferManager.init(context, commandBufferManager, scaledWidth, scaledHeight);
    geometryPipeline.init(context, commandBufferManager, scaledWidth, scaledHeight, gBufferManager);
    lightingPipeline.init(context, commandBufferManager, swapChainImageFormat);
    rtPipeline.init(context, scaledWidth, scaledHeight);
}


// TODO: Either use separate pools for models, full screen quad, etc or use this one for ray tracing as well
void GraphicsPipeline::createDescriptorPool(const Context& context, size_t modelCount, size_t meshCount, size_t fullScreenQuadCount)
{
    std::array<vk::DescriptorPoolSize, 2> poolSizes{};

    // Uniform buffer descriptors
    // - 1 per model
    // - 1 per fullscreen quad
    poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
    poolSizes[0].descriptorCount = static_cast<uint32_t>((modelCount + fullScreenQuadCount) * MAX_FRAMES_IN_FLIGHT);

    // Combined image sampler descriptors
    // - 4 per material (albedo + normal maps + roughness + metalness) -> Use meshCount because a mesh has exactly one material
    // - 6 per fullscreen quad (G-Buffer textures: depth, normal, albedo, roughness, metalness, velocity)
    poolSizes[1].type = vk::DescriptorType::eCombinedImageSampler;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(meshCount * 4 + fullScreenQuadCount * 6 * MAX_FRAMES_IN_FLIGHT);

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    // Total descriptor sets needed:
    // - 1 per model (geometry/transform) per frame in flight since its UBO is written every frame
    // - 1 per fullscreen quad (lighting) per frame in flight since its UBO is written every frame
    // - 1 per material (textures), read-only
    poolInfo.maxSets = static_cast<uint32_t>(meshCount + (modelCount + fullScreenQuadCount) * MAX_FRAMES_IN_FLIGHT);
    
    std::cout << "Creating descriptor pool with " << poolInfo.maxSets << " max sets" << std::endl;
    descriptorPool = context.device.createDescriptorPool(poolInfo);
}

void GraphicsPipeline::handleResize(int nativeWidth, int nativeHeight, int scaledWidth, int scaledHeight, const Context& context, CommandBuffers& commandBufferManager)
{
    context.device.waitIdle();

    // GBuffer
    gBufferManager.cleanup(context.device);
    gBufferManager.init(context, commandBufferManager, scaledWidth, scaledHeight);

    geometryPipeline.handleResize(context, commandBufferManager, gBufferManager, scaledWidth, scaledHeight);
    lightingPipeline.handleResize(context, commandBufferManager);
    
    rtPipeline.handleResize(context, scaledWidth, scaledHeight, gBufferManager.depthImageView, gBufferManager.normalImageView, gBufferManager.albedoImageView, gBufferManager.roughnessImageView, gBufferManager.metalnessImageView, gBufferManager.velocityImageView);
}

void GraphicsPipeline::reloadShaders(const Context& context)
{
    geometryPipeline.reloadShaders(context);
    lightingPipeline.reloadShaders(context);
    rtPipeline.reloadShaders(context);
}

void GraphicsPipeline::cleanup(vk::Device device)
{
    rtPipeline.cleanup(device);
    geometryPipeline.cleanup(device);
    lightingPipeline.cleanup(device);
    gBufferManager.cleanup(device);

    if (descriptorPool)
    {
        device.destroyDescriptorPool(descriptorPool);
    }
}

} // namespace render
} // namespace vkrt
