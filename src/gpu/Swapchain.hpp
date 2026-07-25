#pragma once

#include "core/Constants.hpp"
#include <vector>
#include "gpu/Utils.hpp"
#include "gpu/Context.hpp"
#include "gpu/CommandBuffers.hpp"
#include "core/Vk.hpp"


namespace vkrt {
namespace gpu {
using namespace vkrt::core;

struct SwapChainSupportDetails
{
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

class Swapchain
{
public:
    vk::SwapchainKHR swapChain;
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat;
    vk::Extent2D extent;
    std::vector<vk::ImageView> swapChainImageViews;
    std::vector<vk::Framebuffer> swapChainFramebuffers;

    bool framebufferResized = false;

public:
    static SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device, vk::SurfaceKHR surface);
    static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
    static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
    static vk::Extent2D chooseSwapExtent(int width, int height, const vk::SurfaceCapabilitiesKHR& capabilities);

    void init(int width, int height, const Context& context);
    void createSwapChain(int width, int height, const Context& context);
    void createImageViews(const Context& context);
    void cleanup(vk::Device device);
    void createFramebuffers(const Context& context, vk::RenderPass renderPass);
    void handleResize(int width, int height, const Context& context, CommandBuffers& commandBufferManager, vk::RenderPass renderPass);
};

} // namespace gpu
} // namespace vkrt
