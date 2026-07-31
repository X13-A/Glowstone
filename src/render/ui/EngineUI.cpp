#include "render/ui/EngineUI.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include "core/Constants.hpp"
#include "core/Settings.hpp"
#include "core/Time.hpp"
#include "input/EventManager.hpp"
#include "input/Events.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"


namespace vkrt {
namespace render {
using namespace vkrt::core;
using namespace vkrt::gpu;
using namespace vkrt::input;

namespace {
constexpr float PANEL_MARGIN = 12.0f;
constexpr float PANEL_ALPHA = 0.65f;
constexpr float ITEM_WIDTH = 160.0f;
constexpr float PLOT_WIDTH = 260.0f;
constexpr float PLOT_HEIGHT = 60.0f;
constexpr float PLOT_HEADROOM = 1.25f;
constexpr float PLOT_MIN_RANGE_MS = 1.0f;

constexpr float RENDER_SCALE_MIN = 0.1f;
constexpr float RENDER_SCALE_MAX = 1.0f;
constexpr int SPP_MAX = 64;
constexpr int RIS_CANDIDATES_MAX = 64;

const char* const SAMPLING_MODE_NAMES[] = { "Cosine", "MIS", "MIS + RIS", "ReSTIR" };

PFN_vkVoidFunction loadVulkanFunction(const char* name, void* instance)
{
    return VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr(static_cast<VkInstance>(instance), name);
}

template <size_t N>
float average(const std::array<float, N>& values)
{
    return std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
}

float toFps(float milliseconds)
{
    return milliseconds > 0.0f ? 1000.0f / milliseconds : 0.0f;
}

void shortcutHint(const char* shortcut)
{
    ImGui::SameLine();
    ImGui::TextDisabled("%s", shortcut);
}
}

void EngineUI::init(const Context& context, const Swapchain& swapchain, GLFWwindow* window)
{
    createRenderPass(context, swapchain.swapChainImageFormat);
    createFramebuffers(context, swapchain);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_LoadFunctions(VULKAN_API_VERSION, loadVulkanFunction, static_cast<VkInstance>(context.instance));

    const uint32_t imageCount = static_cast<uint32_t>(swapchain.swapChainImages.size());

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VULKAN_API_VERSION;
    initInfo.Instance = context.instance;
    initInfo.PhysicalDevice = context.physicalDevice;
    initInfo.Device = context.device;
    initInfo.QueueFamily = context.findQueueFamilies(context.physicalDevice).graphicsFamily.value();
    initInfo.Queue = context.graphicsQueue;
    initInfo.DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE;
    initInfo.RenderPass = renderPass;
    initInfo.MinImageCount = imageCount;
    initInfo.ImageCount = imageCount;
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    if (!ImGui_ImplVulkan_Init(&initInfo))
    {
        throw std::runtime_error("failed to initialize the ImGui Vulkan backend!");
    }

    pendingRenderScale = Settings::renderScale;
    setInputEnabled(false);
}

void EngineUI::setInputEnabled(bool enabled)
{
    ImGuiIO& io = ImGui::GetIO();
    if (enabled)
    {
        io.ConfigFlags &= ~ImGuiConfigFlags_NoMouse;
    }
    else
    {
        io.ConfigFlags |= ImGuiConfigFlags_NoMouse;
    }
}

void EngineUI::createRenderPass(const Context& context, vk::Format colorFormat)
{
    // UI composited on top of the swapchain image
    vk::AttachmentDescription colorAttachment{};
    colorAttachment.format = colorFormat;
    colorAttachment.samples = vk::SampleCountFlagBits::e1;
    colorAttachment.loadOp = vk::AttachmentLoadOp::eLoad;
    colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    colorAttachment.initialLayout = vk::ImageLayout::ePresentSrcKHR;
    colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::AttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::SubpassDescription subpass{};
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    vk::SubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eTransfer;
    dependency.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eTransferWrite;
    dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eColorAttachmentRead;

    vk::RenderPassCreateInfo renderPassInfo{};
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    renderPass = context.device.createRenderPass(renderPassInfo);
}

void EngineUI::createFramebuffers(const Context& context, const Swapchain& swapchain)
{
    extent = swapchain.extent;
    framebuffers.resize(swapchain.swapChainImageViews.size());

    for (size_t i = 0; i < swapchain.swapChainImageViews.size(); i++)
    {
        vk::FramebufferCreateInfo framebufferInfo{};
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = &swapchain.swapChainImageViews[i];
        framebufferInfo.width = swapchain.extent.width;
        framebufferInfo.height = swapchain.extent.height;
        framebufferInfo.layers = 1;

        framebuffers[i] = context.device.createFramebuffer(framebufferInfo);
    }
}

void EngineUI::destroyFramebuffers(const Context& context)
{
    for (vk::Framebuffer framebuffer : framebuffers)
    {
        context.device.destroyFramebuffer(framebuffer);
    }
    framebuffers.clear();
}

void EngineUI::handleResize(const Context& context, const Swapchain& swapchain)
{
    destroyFramebuffers(context);
    createFramebuffers(context, swapchain);

    ImGui_ImplVulkan_SetMinImageCount(static_cast<uint32_t>(swapchain.swapChainImages.size()));
}

void EngineUI::pushSample(const GpuProfiler& gpuProfiler, double cpuFrameTimeMs)
{
    const std::vector<GpuProfiler::Section>& sections = gpuProfiler.getSections();

    frameHistory[historyOffset] = static_cast<float>(gpuProfiler.getFrameTimeMs());
    cpuHistory[historyOffset] = static_cast<float>(cpuFrameTimeMs);

    // Mirrors the sections recorded last frame
    sectionStats.resize(sections.size());

    for (size_t i = 0; i < sections.size(); i++)
    {
        SectionStats& stats = sectionStats[i];
        const float elapsedMs = static_cast<float>(sections[i].elapsedMs);

        if (stats.name != sections[i].name)
        {
            stats.name = sections[i].name;
            stats.history.fill(elapsedMs);
        }

        stats.history[historyOffset] = elapsedMs;
    }

    historyOffset = (historyOffset + 1) % HISTORY_SIZE;
}

void EngineUI::buildPerformanceSection() const
{
    const float gpuAverage = average(frameHistory);
    const float cpuAverage = average(cpuHistory);
    const float plotRange = std::max(*std::max_element(frameHistory.begin(), frameHistory.end()) * PLOT_HEADROOM, PLOT_MIN_RANGE_MS);

    ImGui::SeparatorText("Performance");
    ImGui::Text("GPU  %6.2f ms  (%5.0f FPS)", gpuAverage, toFps(gpuAverage));
    ImGui::Text("CPU  %6.2f ms  (%5.0f FPS)", cpuAverage, toFps(cpuAverage));
    ImGui::PlotLines("##gpuFrameTime", frameHistory.data(), static_cast<int>(HISTORY_SIZE), static_cast<int>(historyOffset),
        "GPU", 0.0f, plotRange, ImVec2(PLOT_WIDTH, PLOT_HEIGHT));

    for (const SectionStats& stats : sectionStats)
    {
        ImGui::TextDisabled("%-12s %6.2f ms", stats.name.c_str(), average(stats.history));
    }

    ImGui::Checkbox("Cap frame rate", &Settings::frameRateCap);
}

void EngineUI::buildRenderingSection()
{
    ImGui::SeparatorText("Rendering");

    bool changed = false;

    changed |= ImGui::Checkbox("Ray tracing", &Settings::displayRayTracing);
    shortcutHint("R");
    changed |= ImGui::Checkbox("Accumulate frames", &Settings::frameAccumulation);

    int samplingMode = Settings::samplingMode;
    if (ImGui::Combo("Sampling", &samplingMode, SAMPLING_MODE_NAMES, IM_ARRAYSIZE(SAMPLING_MODE_NAMES)))
    {
        EventManager::get().enqueue(RequestSamplingModeChangeEvent{ samplingMode });
    }

    changed |= ImGui::SliderInt("Samples per pixel", &Settings::spp, 1, SPP_MAX);
    changed |= ImGui::SliderInt("Bounces", &Settings::rt_recursion_depth, 0, RT_MAX_RECURSION_DEPTH);
    changed |= ImGui::SliderInt("RIS candidates", &Settings::risCandidates, 1, RIS_CANDIDATES_MAX);

    ImGui::SliderFloat("Render scale", &pendingRenderScale, RENDER_SCALE_MIN, RENDER_SCALE_MAX, "%.2f");
    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        EventManager::get().enqueue(RequestRenderScaleChangeEvent{ pendingRenderScale });
    }

    if (changed)
    {
        Time::resetFrameCount();
    }
}

void EngineUI::buildDenoisingSection() const
{
    ImGui::SeparatorText("Denoising");

    ImGui::Checkbox("Denoise", &Settings::denoisingEnabled);
    ImGui::Checkbox("Show variance", &Settings::displayVariance);

    if (ImGui::Button("Print variance sum"))
    {
        Settings::printVarianceSum = true;
    }
}

void EngineUI::buildDebugSection() const
{
    ImGui::SeparatorText("Debug");

    if (ImGui::Button("Reload shaders"))
    {
        EventManager::get().enqueue(RequestShaderReloadEvent{});
    }
    shortcutHint("Ctrl+R");

    if (ImGui::Checkbox("Debug toggle", &Settings::debugBool1))
    {
        Time::resetFrameCount();
    }
}

void EngineUI::buildPanel()
{
    ImGui::SetNextWindowPos(ImVec2(PANEL_MARGIN, PANEL_MARGIN), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(PANEL_ALPHA);

    if (ImGui::Begin("Glowstone", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::PushItemWidth(ITEM_WIDTH);

        buildPerformanceSection();
        buildRenderingSection();
        buildDenoisingSection();
        buildDebugSection();

        ImGui::PopItemWidth();
    }
    ImGui::End();
}

void EngineUI::beginFrame(const GpuProfiler& gpuProfiler, double cpuFrameTimeMs)
{
    frameBuilt = visible;
    if (!frameBuilt)
    {
        return;
    }

    pushSample(gpuProfiler, cpuFrameTimeMs);

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    buildPanel();

    ImGui::Render();
}

void EngineUI::record(vk::CommandBuffer commandBuffer, uint32_t imageIndex)
{
    if (!frameBuilt)
    {
        return;
    }
    frameBuilt = false;

    vk::RenderPassBeginInfo renderPassInfo{};
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = framebuffers[imageIndex];
    renderPassInfo.renderArea.extent = extent;

    commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
    commandBuffer.endRenderPass();
}

void EngineUI::cleanup(const Context& context)
{
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    destroyFramebuffers(context);
    context.device.destroyRenderPass(renderPass);
}

} // namespace render
} // namespace vkrt
