#include "render/GBuffer.hpp"
#include "gpu/Utils.hpp"


namespace vkrt {
namespace render {
using namespace vkrt::gpu;

void GBuffer::init(const Context& context, CommandBuffers& commandBufferManager, uint32_t width, uint32_t height)
{
    createDepthResources(context, commandBufferManager, width, height);
    createNormalResources(context, commandBufferManager, width, height);
    createAlbedoResources(context, commandBufferManager, width, height);
    createRoughnessResources(context, commandBufferManager, width, height);
    createMetalnessResources(context, commandBufferManager, width, height);
    createVelocityResources(context, commandBufferManager, width, height);
}

void GBuffer::createDepthResources(const Context& context, CommandBuffers& commandBufferManager, uint32_t width, uint32_t height)
{
    vk::Format depthFormat = gpu::DepthStencil::findDepthFormat(context.physicalDevice);

    gpu::Image::createImage(
        context,
        width,
        height,
        depthFormat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        depthImage,
        depthImageMemory
    );

    depthImageView = gpu::Image::createImageView(
        context,
        depthImage,
        depthFormat,
        vk::ImageAspectFlagBits::eDepth
    );

    gpu::Image::transitionImageLayout(
        context,
        commandBufferManager,
        depthImage,
        depthFormat,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthStencilAttachmentOptimal
    );
}

void GBuffer::createNormalResources(const Context& context, CommandBuffers& commandBufferManager, uint32_t width, uint32_t height)
{
    gpu::Image::createImage(
        context,
        width,
        height,
        NORMAL_MAP_FORMAT,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        normalImage,
        normalImageMemory
    );

    normalImageView = gpu::Image::createImageView(
        context,
        normalImage,
        NORMAL_MAP_FORMAT,
        vk::ImageAspectFlagBits::eColor
    );

    gpu::Image::transitionImageLayout(
        context,
        commandBufferManager,
        normalImage,
        NORMAL_MAP_FORMAT,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal
    );
}

void GBuffer::createAlbedoResources(const Context& context, CommandBuffers& commandBufferManager, uint32_t width, uint32_t height)
{
    gpu::Image::createImage(
        context,
        width,
        height,
        ALBEDO_MAP_FORMAT,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        albedoImage,
        albedoImageMemory
    );

    albedoImageView = gpu::Image::createImageView(
        context,
        albedoImage,
        ALBEDO_MAP_FORMAT,
        vk::ImageAspectFlagBits::eColor
    );

    gpu::Image::transitionImageLayout(
        context,
        commandBufferManager,
        albedoImage,
        ALBEDO_MAP_FORMAT,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal
    );
}

void GBuffer::createRoughnessResources(const Context& context, CommandBuffers& commandBufferManager, uint32_t width, uint32_t height)
{
    gpu::Image::createImage(
        context,
        width,
        height,
        ROUGHNESS_MAP_FORMAT,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        roughnessImage,
        roughnessImageMemory
    );

    roughnessImageView = gpu::Image::createImageView(
        context,
        roughnessImage,
        ROUGHNESS_MAP_FORMAT,
        vk::ImageAspectFlagBits::eColor
    );

    gpu::Image::transitionImageLayout(
        context,
        commandBufferManager,
        roughnessImage,
        ROUGHNESS_MAP_FORMAT,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal
    );
}

void GBuffer::createMetalnessResources(const Context& context, CommandBuffers& commandBufferManager, uint32_t width, uint32_t height)
{
    gpu::Image::createImage(
        context,
        width,
        height,
        METALNESS_MAP_FORMAT,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        metalnessImage,
        metalnessImageMemory
    );

    metalnessImageView = gpu::Image::createImageView(
        context,
        metalnessImage,
        METALNESS_MAP_FORMAT,
        vk::ImageAspectFlagBits::eColor
    );

    gpu::Image::transitionImageLayout(
        context,
        commandBufferManager,
        metalnessImage,
        METALNESS_MAP_FORMAT,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal
    );
}


void GBuffer::createVelocityResources(const Context& context, CommandBuffers& commandBufferManager, uint32_t width, uint32_t height)
{
    gpu::Image::createImage(
        context,
        width,
        height,
        MOTION_VECTOR_FORMAT,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        velocityImage,
        velocityImageMemory
    );

    velocityImageView = gpu::Image::createImageView(
        context,
        velocityImage,
        MOTION_VECTOR_FORMAT,
        vk::ImageAspectFlagBits::eColor
    );

    gpu::Image::transitionImageLayout(
        context,
        commandBufferManager,
        velocityImage,
        MOTION_VECTOR_FORMAT,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal
    );
}


void GBuffer::cleanup(vk::Device device)
{
    device.destroyImageView(depthImageView);
    device.freeMemory(depthImageMemory);
    device.destroyImage(depthImage);

    device.destroyImageView(normalImageView);
    device.freeMemory(normalImageMemory);
    device.destroyImage(normalImage);

    device.destroyImageView(albedoImageView);
    device.freeMemory(albedoImageMemory);
    device.destroyImage(albedoImage);

    device.destroyImageView(roughnessImageView);
    device.freeMemory(roughnessImageMemory);
    device.destroyImage(roughnessImage);

    device.destroyImageView(metalnessImageView);
    device.freeMemory(metalnessImageMemory);
    device.destroyImage(metalnessImage);

    device.destroyImageView(velocityImageView);
    device.freeMemory(velocityImageMemory);
    device.destroyImage(velocityImage);
}

// Getters
vk::ImageView GBuffer::getDepthImageView() const
{
    return depthImageView;
}
vk::ImageView GBuffer::getNormalImageView() const
{
    return normalImageView;
}
vk::ImageView GBuffer::getAlbedoImageView() const
{
    return albedoImageView;
}
vk::ImageView GBuffer::getRoughnessImageView() const
{
    return roughnessImageView;
}
vk::ImageView GBuffer::getMetalnessImageView() const
{
    return metalnessImageView;
}
vk::ImageView GBuffer::getVelocityImageView() const
{
    return velocityImageView;
}

} // namespace render
} // namespace vkrt
