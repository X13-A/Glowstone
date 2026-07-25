#include "gpu/Utils.hpp"
#include <iostream>
#include <vector>
#include "core/Utils.hpp"

namespace vkrt {
using namespace gpu;

vk::Format gpu::Hardware::findSupportedFormat(vk::PhysicalDevice physicalDevice, const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features)
{
    for (vk::Format format : candidates)
    {
        vk::FormatProperties props = physicalDevice.getFormatProperties(format);

        if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features)
        {
            return format;
        }
        else if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features)
        {
            return format;
        }
    }
    throw std::runtime_error("failed to find supported format!");
}

vk::ShaderModule gpu::Shaders::createShaderModule(const Context& context, const std::vector<char>& code)
{
    vk::ShaderModuleCreateInfo createInfo{};
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    return context.device.createShaderModule(createInfo);
}

bool gpu::DepthStencil::hasStencilComponent(vk::Format format)
{
    return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
}

vk::Format gpu::DepthStencil::findDepthFormat(vk::PhysicalDevice physicalDevice)
{
    return gpu::Hardware::findSupportedFormat(
        physicalDevice,
        { vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint },
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment
    );
}

uint32_t gpu::Memory::findMemoryType(vk::PhysicalDevice physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties)
{
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void gpu::Image::copyBufferToImage(const Context& context, CommandBuffers& commandBufferManager, vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height)
{
    vk::CommandBuffer commandBuffer = commandBufferManager.beginSingleTimeCommands(context.device);

    vk::BufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = vk::Offset3D{ 0, 0, 0 };
    region.imageExtent = vk::Extent3D{ width, height, 1 };

    commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);

    commandBufferManager.endSingleTimeCommands(context.device, context.graphicsQueue, commandBuffer);
}

void gpu::Image::createImage(const Context& context, uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Image& image, vk::DeviceMemory& imageMemory)
{
    vk::ImageCreateInfo imageInfo{};
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.extent = vk::Extent3D{ width, height, 1 };
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;
    imageInfo.usage = usage;
    imageInfo.samples = vk::SampleCountFlagBits::e1;
    imageInfo.sharingMode = vk::SharingMode::eExclusive;

    image = context.device.createImage(imageInfo);

    vk::MemoryRequirements memRequirements = context.device.getImageMemoryRequirements(image);

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = gpu::Memory::findMemoryType(context.physicalDevice, memRequirements.memoryTypeBits, properties);

    imageMemory = context.device.allocateMemory(allocInfo);

    context.device.bindImageMemory(image, imageMemory, 0);
}

vk::ImageView gpu::Image::createImageView(const Context& context, vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags)
{
    vk::ImageViewCreateInfo viewInfo{};
    viewInfo.image = image;
    viewInfo.viewType = vk::ImageViewType::e2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    viewInfo.subresourceRange.aspectMask = aspectFlags;

    return context.device.createImageView(viewInfo);
}

void gpu::Image::blitImage(
    vk::CommandBuffer commandBuffer,
    vk::Image srcImage, vk::Image dstImage,
    vk::Format srcFormat, vk::Format dstFormat,
    vk::AccessFlags srcOriginalAccessMask, vk::AccessFlags dstOriginalAccessMask,
    vk::ImageLayout srcOriginalLayout, vk::ImageLayout dstOriginalLayout,
    uint32_t srcWidth, uint32_t srcHeight,
    uint32_t dstWidth, uint32_t dstHeight,
    vk::Filter filter
) {
    std::vector<vk::ImageMemoryBarrier> preBlitBarriers;

    if (srcOriginalLayout != vk::ImageLayout::eTransferSrcOptimal)
    {
        vk::ImageMemoryBarrier preSrcBarrier{};
        preSrcBarrier.oldLayout = srcOriginalLayout;
        preSrcBarrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
        preSrcBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preSrcBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preSrcBarrier.image = srcImage;
        preSrcBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        preSrcBarrier.subresourceRange.baseMipLevel = 0;
        preSrcBarrier.subresourceRange.levelCount = 1;
        preSrcBarrier.subresourceRange.baseArrayLayer = 0;
        preSrcBarrier.subresourceRange.layerCount = 1;

        preSrcBarrier.srcAccessMask = srcOriginalAccessMask;
        preSrcBarrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
        preBlitBarriers.push_back(preSrcBarrier);
    }

    if (dstOriginalLayout != vk::ImageLayout::eTransferDstOptimal)
    {
        vk::ImageMemoryBarrier preDstBarrier{};
        preDstBarrier.oldLayout = dstOriginalLayout;
        preDstBarrier.newLayout = vk::ImageLayout::eTransferDstOptimal;
        preDstBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preDstBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preDstBarrier.image = dstImage;
        preDstBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        preDstBarrier.subresourceRange.baseMipLevel = 0;
        preDstBarrier.subresourceRange.levelCount = 1;
        preDstBarrier.subresourceRange.baseArrayLayer = 0;
        preDstBarrier.subresourceRange.layerCount = 1;

        preDstBarrier.srcAccessMask = dstOriginalAccessMask;
        preDstBarrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
        preBlitBarriers.push_back(preDstBarrier);
    }

    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eTransfer,
        {},
        {}, {}, preBlitBarriers
    );

    vk::ImageBlit blitRegion{};
    blitRegion.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    blitRegion.srcSubresource.mipLevel = 0;
    blitRegion.srcSubresource.baseArrayLayer = 0;
    blitRegion.srcSubresource.layerCount = 1;
    blitRegion.srcOffsets[0] = vk::Offset3D{ 0, 0, 0 };
    blitRegion.srcOffsets[1] = vk::Offset3D{ static_cast<int32_t>(srcWidth), static_cast<int32_t>(srcHeight), 1 };

    blitRegion.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    blitRegion.dstSubresource.mipLevel = 0;
    blitRegion.dstSubresource.baseArrayLayer = 0;
    blitRegion.dstSubresource.layerCount = 1;
    blitRegion.dstOffsets[0] = vk::Offset3D{ 0, 0, 0 };
    blitRegion.dstOffsets[1] = vk::Offset3D{ static_cast<int32_t>(dstWidth), static_cast<int32_t>(dstHeight), 1 };

    commandBuffer.blitImage(
        srcImage, vk::ImageLayout::eTransferSrcOptimal,
        dstImage, vk::ImageLayout::eTransferDstOptimal,
        blitRegion,
        filter
    );

    std::vector<vk::ImageMemoryBarrier> postBlitBarriers;

    vk::ImageMemoryBarrier finalDstBarrier{};
    finalDstBarrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    finalDstBarrier.newLayout = dstOriginalLayout;
    finalDstBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    finalDstBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    finalDstBarrier.image = dstImage;
    finalDstBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    finalDstBarrier.subresourceRange.baseMipLevel = 0;
    finalDstBarrier.subresourceRange.levelCount = 1;
    finalDstBarrier.subresourceRange.baseArrayLayer = 0;
    finalDstBarrier.subresourceRange.layerCount = 1;
    finalDstBarrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    finalDstBarrier.dstAccessMask = dstOriginalAccessMask;
    postBlitBarriers.push_back(finalDstBarrier);

    vk::ImageMemoryBarrier finalSrcBarrier{};
    finalSrcBarrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
    finalSrcBarrier.newLayout = srcOriginalLayout;
    finalSrcBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    finalSrcBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    finalSrcBarrier.image = srcImage;
    finalSrcBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    finalSrcBarrier.subresourceRange.baseMipLevel = 0;
    finalSrcBarrier.subresourceRange.levelCount = 1;
    finalSrcBarrier.subresourceRange.baseArrayLayer = 0;
    finalSrcBarrier.subresourceRange.layerCount = 1;
    finalSrcBarrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
    finalSrcBarrier.dstAccessMask = srcOriginalAccessMask;
    postBlitBarriers.push_back(finalSrcBarrier);

    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eAllCommands,
        {},
        {}, {}, postBlitBarriers
    );
}

void gpu::Image::transition_depthRW_to_depthR(const Context& context, vk::CommandBuffer commandBuffer, vk::Image image, vk::Format format)
{
    vk::ImageMemoryBarrier barrier{};
    barrier.oldLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    barrier.newLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;

    if (gpu::DepthStencil::hasStencilComponent(format))
    {
        barrier.subresourceRange.aspectMask |= vk::ImageAspectFlagBits::eStencil;
    }

    barrier.srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentRead;
    barrier.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead;

    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eLateFragmentTests,
        vk::PipelineStageFlagBits::eFragmentShader,
        {},
        {}, {}, barrier
    );
}

// TODO: this is a mess...
void gpu::Image::transitionImageLayout(const Context& context, CommandBuffers& commandBufferManager, vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout)
{
    vk::CommandBuffer commandBuffer = commandBufferManager.beginSingleTimeCommands(context.device);

    vk::ImageMemoryBarrier barrier{};
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    if (newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal || newLayout == vk::ImageLayout::eDepthStencilReadOnlyOptimal)
    {
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;

        if (gpu::DepthStencil::hasStencilComponent(format))
        {
            barrier.subresourceRange.aspectMask |= vk::ImageAspectFlagBits::eStencil;
        }
    }
    else
    {
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    }

    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal)
    {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    }
    else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
    {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    }
    else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal)
    {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite;

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
    }
    else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eColorAttachmentOptimal)
    {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    }
    else if (oldLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal && newLayout == vk::ImageLayout::eDepthStencilReadOnlyOptimal)
    {
        barrier.srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentRead;
        barrier.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead;

        sourceStage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    }
    else if (oldLayout == vk::ImageLayout::eDepthStencilReadOnlyOptimal && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal)
    {
        barrier.srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead;
        barrier.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite;

        sourceStage = vk::PipelineStageFlagBits::eFragmentShader;
        destinationStage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
    }
    else
    {
        throw std::invalid_argument("unsupported layout transition!");
    }

    commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, barrier);

    commandBufferManager.endSingleTimeCommands(context.device, context.graphicsQueue, commandBuffer);
}

vk::Sampler gpu::Textures::createSampler(const Context& context, vk::Filter minFilter, vk::Filter magFilter, vk::SamplerMipmapMode mipMapMode)
{
    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = magFilter;
    samplerInfo.minFilter = minFilter;

    samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;

    vk::PhysicalDeviceProperties properties = context.physicalDevice.getProperties();
    samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;

    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = vk::CompareOp::eAlways;

    samplerInfo.mipmapMode = mipMapMode;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    return context.device.createSampler(samplerInfo);
}

void gpu::Buffers::createBuffer(const Context& context, vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer& buffer, vk::DeviceMemory& bufferMemory, bool deviceAdressing)
{
    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = vk::SharingMode::eExclusive;

    buffer = context.device.createBuffer(bufferInfo);

    vk::MemoryRequirements memRequirements = context.device.getBufferMemoryRequirements(buffer);

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = gpu::Memory::findMemoryType(context.physicalDevice, memRequirements.memoryTypeBits, properties);

    vk::MemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
    if (deviceAdressing)
    {
        memoryAllocateFlagsInfo.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
        allocInfo.pNext = &memoryAllocateFlagsInfo;
    }

    bufferMemory = context.device.allocateMemory(allocInfo);

    context.device.bindBufferMemory(buffer, bufferMemory, 0);
}

void gpu::Buffers::copyBuffer(const Context& context, CommandBuffers& commandBuffers, vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size)
{
    vk::CommandBuffer commandBuffer = commandBuffers.beginSingleTimeCommands(context.device);

    vk::BufferCopy copyRegion{};
    copyRegion.size = size;
    commandBuffer.copyBuffer(srcBuffer, dstBuffer, copyRegion);

    commandBuffers.endSingleTimeCommands(context.device, context.graphicsQueue, commandBuffer);
}

void gpu::Buffers::createScratchBuffer(const Context& context, vk::DeviceSize size, vk::Buffer& scratchBuffer, vk::DeviceMemory& scratchBufferMemory)
{
    vk::BufferUsageFlags scratchBufferUsage =
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eShaderDeviceAddress;

    gpu::Buffers::createBuffer(
        context,
        size,
        scratchBufferUsage,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        scratchBuffer,
        scratchBufferMemory,
        true);
}

vk::DeviceAddress gpu::Buffers::getBufferDeviceAdress(const Context& context, vk::Buffer buffer)
{
    vk::BufferDeviceAddressInfo addressInfo{};
    addressInfo.buffer = buffer;

    return context.device.getBufferAddress(addressInfo);
}

} // namespace vkrt
