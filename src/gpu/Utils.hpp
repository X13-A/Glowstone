#pragma once

#include "core/Constants.hpp"
#include "gpu/CommandBuffers.hpp"
#include "gpu/Geometry.hpp"


namespace vkrt {
namespace gpu
{
    namespace Hardware
    {
        vk::Format findSupportedFormat(vk::PhysicalDevice physicalDevice, const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features);
    };

    namespace Shaders
    {
        vk::ShaderModule createShaderModule(const Context& context, const std::vector<char>& code);
    };

    namespace DepthStencil
    {
        bool hasStencilComponent(vk::Format format);
        vk::Format findDepthFormat(vk::PhysicalDevice physicalDevice);
    };

    namespace Memory
    {
        uint32_t findMemoryType(vk::PhysicalDevice physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties);
    };

    namespace Image
    {
        void copyBufferToImage(const Context& context, CommandBuffers& commandBufferManager, vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height);
        void createImage(const Context& context, uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Image& image, vk::DeviceMemory& imageMemory);
        vk::ImageView createImageView(const Context& context, vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags);
        void blitImage(
            vk::CommandBuffer commandBuffer,
            vk::Image srcImage, vk::Image dstImage,
            vk::Format srcFormat, vk::Format dstFormat,
            vk::AccessFlags srcOriginalAccessMask, vk::AccessFlags dstOriginalAccessMask,
            vk::ImageLayout srcOriginalLayout, vk::ImageLayout dstOriginalLayout,
            uint32_t srcWidth, uint32_t srcHeight,
            uint32_t dstWidth, uint32_t dstHeight,
            vk::Filter filter
        );
        void transition_depthRW_to_depthR(const Context& context, vk::CommandBuffer commandBuffer, vk::Image image, vk::Format format);
        void transitionImageLayout(const Context& context, CommandBuffers& commandBufferManager, vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);
    };

    namespace Textures
    {
        vk::Sampler createSampler(const Context& context, vk::Filter minFilter, vk::Filter magFilter, vk::SamplerMipmapMode mipMapMode);
    };

    namespace Buffers
    {
        void createBuffer(const Context& context, vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer& buffer, vk::DeviceMemory& bufferMemory, bool deviceAdressing = false);

        template<typename T>
        void createAndFillBuffer(
            const Context& context,
            CommandBuffers& commandBuffers,
            const std::vector<T>& data,
            vk::Buffer& buffer,
            vk::DeviceMemory& bufferMemory,
            vk::BufferUsageFlags usageFlags,
            vk::MemoryPropertyFlags memoryFlags,
            bool deviceAdressing = false);

        void createScratchBuffer(const Context& context, vk::DeviceSize size, vk::Buffer& scratchBuffer, vk::DeviceMemory& scratchBufferMemory);
        void copyBuffer(const Context& context, CommandBuffers& commandBuffers, vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size);
        vk::DeviceAddress getBufferDeviceAdress(const Context& context, vk::Buffer buffer);
    };
};

template<typename T>
void gpu::Buffers::createAndFillBuffer(
    const Context& context,
    CommandBuffers& commandBuffers,
    const std::vector<T>& data,
    vk::Buffer& buffer,
    vk::DeviceMemory& bufferMemory,
    vk::BufferUsageFlags usageFlags,
    vk::MemoryPropertyFlags memoryFlags,
    bool deviceAdressing)
{
    vk::DeviceSize bufferSize = sizeof(T) * data.size();

    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingBufferMemory;

    gpu::Buffers::createBuffer(
        context,
        bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer,
        stagingBufferMemory,
        deviceAdressing
    );

    void* mappedData = context.device.mapMemory(stagingBufferMemory, 0, bufferSize);
    memcpy(mappedData, data.data(), static_cast<size_t>(bufferSize));
    context.device.unmapMemory(stagingBufferMemory);

    gpu::Buffers::createBuffer(
        context,
        bufferSize,
        usageFlags,
        memoryFlags,
        buffer,
        bufferMemory,
        true
    );

    gpu::Buffers::copyBuffer(context, commandBuffers, stagingBuffer, buffer, bufferSize);

    context.device.destroyBuffer(stagingBuffer);
    context.device.freeMemory(stagingBufferMemory);
}

} // namespace vkrt
