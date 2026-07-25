#pragma once

#include "core/Vk.hpp"
#include "gpu/Context.hpp"
#include "gpu/Swapchain.hpp"
#include "render/GBuffer.hpp"
#include "render/pipeline/RayTracingPipeline.hpp"
#include "render/pipeline/GeometryPipeline.hpp"
#include "render/pipeline/LightingPipeline.hpp"


namespace vkrt {
namespace render {
using namespace vkrt::core;
using namespace vkrt::gpu;

class GraphicsPipeline
{
public:
    vk::DescriptorPool descriptorPool;

    GeometryPipeline geometryPipeline;
    LightingPipeline lightingPipeline;
    RayTracingPipeline rtPipeline;

    // TODO: Move out gbuffer from this class
    GBuffer gBufferManager;

public:
    void initPipelines(int nativeWidth, int nativeHeight, int scaledWidth, int scaledHeight, const Context& context, CommandBuffers& commandBufferManager, vk::Format swapChainImageFormat);
    void createDescriptorPool(const Context& context, size_t modelCount, size_t materialCount, size_t fullScreenQuadCount);
    void handleResize(int nativeWidth, int nativeHeight, int scaledWidth, int scaledHeight, const Context& context, CommandBuffers& commandBufferManager);
    void reloadShaders(const Context& context);
    void cleanup(vk::Device device);
};

} // namespace render
} // namespace vkrt
