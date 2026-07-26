#include "gpu/Swapchain.hpp"
#include "core/Vk.hpp"

#include <algorithm>
#include <iostream>
#include <array>


namespace vkrt {
namespace gpu {
using namespace vkrt::core;

SwapChainSupportDetails Swapchain::querySwapChainSupport(vk::PhysicalDevice device, vk::SurfaceKHR surface)
{
    SwapChainSupportDetails details;
    details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
    details.formats = device.getSurfaceFormatsKHR(surface);
    details.presentModes = device.getSurfacePresentModesKHR(surface);
    return details;
}

vk::SurfaceFormatKHR Swapchain::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
    for (const vk::SurfaceFormatKHR& availableFormat : availableFormats)
    {
        if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
        {
            return availableFormat;
        }
    }
    throw std::runtime_error("No suitable VK color format available!");
}

vk::PresentModeKHR Swapchain::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes)
{
    for (vk::PresentModeKHR preferredPresentMode : { vk::PresentModeKHR::eImmediate, vk::PresentModeKHR::eMailbox })
    {
        if (std::find(availablePresentModes.begin(), availablePresentModes.end(), preferredPresentMode) != availablePresentModes.end())
        {
            return preferredPresentMode;
        }
    }

    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D Swapchain::chooseSwapExtent(int width, int height, const vk::SurfaceCapabilitiesKHR& capabilities)
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
    {
        return capabilities.currentExtent;
    }
    else
    {
        vk::Extent2D actualExtent =
        {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };

        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actualExtent;
    }
}

void Swapchain::init(int width, int height, const Context& context)
{
    createSwapChain(width, height, context);
    createImageViews(context);
}

void Swapchain::createSwapChain(int width, int height, const Context& context)
{
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(context.physicalDevice, context.surface);

    vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    vk::Extent2D extent = chooseSwapExtent(width, height, swapChainSupport.capabilities);

    swapChainImageFormat = surfaceFormat.format;
    this->extent = extent;

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
    {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR createInfo{};
    createInfo.surface = context.surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst;

    QueueFamilyIndices indices = context.findQueueFamilies(context.physicalDevice);
    uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

    if (indices.graphicsFamily != indices.presentFamily)
    {
        createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else
    {
        createInfo.imageSharingMode = vk::SharingMode::eExclusive;
        createInfo.queueFamilyIndexCount = 0;
        createInfo.pQueueFamilyIndices = nullptr;
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = nullptr;

    swapChain = context.device.createSwapchainKHR(createInfo);
    swapChainImages = context.device.getSwapchainImagesKHR(swapChain);
}

void Swapchain::createImageViews(const Context& context)
{
    swapChainImageViews.resize(swapChainImages.size());

    for (uint32_t i = 0; i < swapChainImages.size(); i++)
    {
        swapChainImageViews[i] = gpu::Image::createImageView(context, swapChainImages[i], swapChainImageFormat, vk::ImageAspectFlagBits::eColor);
    }
}

void Swapchain::cleanup(vk::Device device)
{
    for (vk::Framebuffer framebuffer : swapChainFramebuffers)
    {
        device.destroyFramebuffer(framebuffer);
    }

    for (vk::ImageView imageView : swapChainImageViews)
    {
        device.destroyImageView(imageView);
    }

    device.destroySwapchainKHR(swapChain);
}

void Swapchain::createFramebuffers(const Context& context, vk::RenderPass renderPass)
{
    swapChainFramebuffers.resize(swapChainImageViews.size());

    for (size_t i = 0; i < swapChainImageViews.size(); i++)
    {
        std::array<vk::ImageView, 1> attachments =
        {
            swapChainImageViews[i]
        };

        vk::FramebufferCreateInfo framebufferInfo{};
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = extent.width;
        framebufferInfo.height = extent.height;
        framebufferInfo.layers = 1;

        swapChainFramebuffers[i] = context.device.createFramebuffer(framebufferInfo);
    }
}

void Swapchain::handleResize(int width, int height, const Context& context, CommandBuffers& commandBufferManager, vk::RenderPass renderPass)
{
    context.device.waitIdle();

    cleanup(context.device);
    createSwapChain(width, height, context);
    createImageViews(context);
    createFramebuffers(context, renderPass);
}

} // namespace gpu
} // namespace vkrt
