#pragma once

#include <array>
#include <vector>
#include "core/Vk.hpp"
#include "gpu/Context.hpp"
#include "gpu/Swapchain.hpp"
#include "render/OverlayPass.hpp"


namespace vkrt {
namespace render {
using namespace vkrt::gpu;

class EngineUI : public OverlayPass
{
public:
    bool visible = true;

public:
    void init(const Context& context, const Swapchain& swapchain, GLFWwindow* window);
    void handleResize(const Context& context, const Swapchain& swapchain);
    void setInputEnabled(bool enabled);
    void beginFrame(double gpuFrameTimeMs, double cpuFrameTimeMs);

    void record(vk::CommandBuffer commandBuffer, uint32_t imageIndex) override;
    void cleanup(const Context& context);

private:
    static constexpr size_t HISTORY_SIZE = 128;

    void createRenderPass(const Context& context, vk::Format colorFormat);
    void createFramebuffers(const Context& context, const Swapchain& swapchain);
    void destroyFramebuffers(const Context& context);

    void pushSample(double gpuFrameTimeMs, double cpuFrameTimeMs);
    void buildPanel();
    void buildPerformanceSection() const;
    void buildRenderingSection();
    void buildDenoisingSection() const;
    void buildDebugSection() const;

    vk::RenderPass renderPass;
    std::vector<vk::Framebuffer> framebuffers;
    vk::Extent2D extent;
    bool frameBuilt = false;

    std::array<float, HISTORY_SIZE> gpuHistory{};
    std::array<float, HISTORY_SIZE> cpuHistory{};
    size_t historyOffset = 0;

    float pendingRenderScale = 1.0f;
};

} // namespace render
} // namespace vkrt
