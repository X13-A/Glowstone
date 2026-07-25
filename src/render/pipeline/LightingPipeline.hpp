#pragma once
#include "core/Vk.hpp"
#include "gpu/Context.hpp"
#include "gpu/CommandBuffers.hpp"
#include "gpu/Swapchain.hpp"
#include "scene/Model.hpp"
#include "render/FullScreenQuad.hpp"


namespace vkrt {
namespace render {
using namespace vkrt::core;
using namespace vkrt::gpu;
using namespace vkrt::scene;

class LightingPipeline
{
private:
    vk::RenderPass renderPass;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline pipeline;

public:
    void init(const Context& context, CommandBuffers& commandBufferManager, vk::Format swapChainImageFormat);
    void cleanup(vk::Device device);
    void handleResize(const Context& context, CommandBuffers& commandBufferManager);
    void reloadShaders(const Context& context);
    void recordDrawCommands(int width, int height, const Swapchain& swapChainManager, const FullScreenQuad& fullScreenQuad, vk::CommandBuffer commandBuffer, uint32_t currentFrame, uint32_t imageIndex);
    vk::RenderPass getRenderPass() const;
    vk::Pipeline getPipeline() const;
    vk::PipelineLayout getPipelineLayout() const;

private:
    void createRenderPasses(const Context& context, vk::Format swapChainImageFormat);
    void createPipelineLayouts(const Context& context);
    void createPipeline(const Context& context);
};

} // namespace render
} // namespace vkrt
