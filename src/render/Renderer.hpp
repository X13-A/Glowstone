#pragma once

#include <vector>
#include "core/Constants.hpp"
#include "core/Vk.hpp"
#include "gpu/Context.hpp"
#include "render/pipeline/GraphicsPipeline.hpp"
#include "gpu/Swapchain.hpp"
#include "gpu/CommandBuffers.hpp"
#include "gpu/GpuTimer.hpp"
#include "scene/Model.hpp"
#include "render/FullScreenQuad.hpp"
#include "render/pass/VariancePass.hpp"
#include "render/pass/DenoisingPass.hpp"
#include "render/OverlayPass.hpp"
#include "scene/Camera.hpp"


namespace vkrt {
namespace render {
using namespace vkrt::core;
using namespace vkrt::gpu;
using namespace vkrt::scene;

class Renderer
{
public:
    uint32_t currentFrame = 0;

    std::vector<vk::Semaphore> imageAvailableSemaphores;
    std::vector<vk::Semaphore> renderFinishedSemaphores;
    std::vector<vk::Fence> inFlightFences;

    VariancePass varianceCompute;
    DenoisingPass denoisingPass;

public:
    void init(const Context& context, const Swapchain& swapChainManager);
    void setOverlayPass(OverlayPass* pass) { overlayPass = pass; }
    void createSyncObjects(const Context& context, const Swapchain& swapChainManager);
    void reloadShaders(const Context& context);
    void recordCommandBuffer(int nativeWidth, int nativeHeight, int scaledWidth, int scaledHeight, const Context& context, CommandBuffers& commandBufferManager, const Swapchain swapChainManager, GraphicsPipeline& graphicsPipeline, uint32_t imageIndex, uint32_t currentFrame, const std::vector<Model>& models, const FullScreenQuad& fullScreenQuad);
    void triggerResize(GLFWwindow* window, const Context& context, Swapchain& swapChainManager, GraphicsPipeline& graphicsPipeline, CommandBuffers& commandBufferManager, FullScreenQuad& fullScreenQuad);
    void drawFrame(int nativeWidth, int nativeHeight, int scaledWidth, int scaledHeight, GLFWwindow* window, const Context& context, Swapchain& swapChainManager, GraphicsPipeline& graphicsPipeline, CommandBuffers& commandBufferManager, const Camera& camera, const std::vector<Model>& models, FullScreenQuad& fullScreenQuad);
    void updateUniformBuffers(int scaledWidth, int scaledHeight, const Camera& camera, const std::vector<Model>& models, const FullScreenQuad& fullScreenQuad, const Swapchain& swapChain, RayTracingPipeline& rtPipeline, uint32_t currentImage);
    void cleanup(const Context& context);

    double getGpuFrameTimeMs() const { return gpuTimer.getElapsedMs(); }

private:
    OverlayPass* overlayPass = nullptr;
    GpuTimer gpuTimer;
    glm::mat4 previousViewProj = glm::mat4(1.0f); // Used for motion vectors
    uint32_t reservoirFrameIndex = 0; // Used for "ping-ponging" the reservoir buffers
};

} // namespace render
} // namespace vkrt
