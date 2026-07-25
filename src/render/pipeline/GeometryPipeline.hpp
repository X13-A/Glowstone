#pragma once
#include "core/Vk.hpp"
#include "gpu/Context.hpp"
#include "gpu/CommandBuffers.hpp"
#include "gpu/Swapchain.hpp"
#include "render/GBuffer.hpp"
#include "scene/Model.hpp"


namespace vkrt {
namespace render {
using namespace vkrt::core;
using namespace vkrt::gpu;
using namespace vkrt::scene;

class GeometryPipeline
{
private:
    vk::RenderPass renderPass;
    vk::Framebuffer framebuffer;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline pipeline;

    int width, height;

public:
    void init(const Context& context, CommandBuffers& commandBufferManager, int width, int height, const GBuffer& gBufferManager);
    void cleanup(vk::Device device);
    void handleResize(const Context& context, CommandBuffers& commandBufferManager, const GBuffer& gBufferManager, int width, int height);
    void reloadShaders(const Context& context);
    void drawMesh(const ShadedMesh& shadedMesh, vk::CommandBuffer cmdBuffer, uint32_t currentFrame);
    void recordDrawCommands(int width, int height, const std::vector<Model>& models, vk::CommandBuffer commandBuffer, uint32_t currentFrame);

    vk::RenderPass getRenderPass() const;
    vk::Framebuffer getFrameBuffer() const;
    vk::Pipeline getPipeline() const;
    vk::PipelineLayout getPipelineLayout() const;
private:
    void createRenderPasses(const Context& context);
    void createFramebuffers(const Context& context, const GBuffer& gBufferManager, uint32_t width, uint32_t height);
    void createPipelineLayouts(const Context& context);
    void createPipeline(const Context& context);
    void drawTBNGizmo(const std::vector<Model>& models, uint32_t currentFrame, vk::CommandBuffer commandBuffer);
};

} // namespace render
} // namespace vkrt
