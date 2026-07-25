#include "gpu/Texture.hpp"
#include "gpu/Utils.hpp"
#include <iostream>
#include <vector>

// One time stb_image implementation
#ifndef STB_IMAGE_IMPLEMENTATION
    #define STB_IMAGE_IMPLEMENTATION
    #include <stb_image.h>
#endif


namespace vkrt {
namespace gpu {
using namespace vkrt::core;

void Texture::init(std::string path, const Context& context, CommandBuffers& commandBufferManager, vk::Format format)
{
    createImage(path, context, commandBufferManager, format);
    createImageView(context, format);

    // TODO: don't create a sampler everytime, reuse a sampler instead
    sampler = gpu::Textures::createSampler(context, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear);
}

void Texture::createImageView(const Context& context, vk::Format format)
{
    imageView = gpu::Image::createImageView(context, image, format, vk::ImageAspectFlagBits::eColor);
}



void Texture::createImage(std::string path, const Context& context, CommandBuffers& commandBufferManager, vk::Format format)
{
    vk::DeviceSize pixelSize = 0;
    int stbi_format = STBI_default;

    if (format == vk::Format::eR32G32B32Sfloat)
    {
        pixelSize = 3 * 4;
        stbi_format = STBI_rgb;
    }
    else if (format == vk::Format::eR32G32B32A32Sfloat)
    {
        pixelSize = 4 * 4;
        stbi_format = STBI_rgb_alpha;
    }
    else if (format == vk::Format::eR8G8B8A8Srgb)
    {
        pixelSize = 4 * 1;
        stbi_format = STBI_rgb_alpha;
    }
    else if (format == vk::Format::eR8Unorm)
    {
        pixelSize = 1 * 1;
        stbi_format = STBI_grey;
    }
    else if (format == vk::Format::eR16Unorm)
    {
        pixelSize = 1 * 2;
        stbi_format = STBI_grey;
    }
    else if (format == vk::Format::eR8G8B8A8Unorm)
    {
        pixelSize = 4 * 1;
        stbi_format = STBI_rgb_alpha;
    }
    else if (format == vk::Format::eR16G16B16A16Unorm)
    {
        pixelSize = 4 * 2;
        stbi_format = STBI_rgb_alpha;
    }
    else
    {
        throw std::runtime_error("Image format not implemented (Texture.cpp) !");
    }

    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, stbi_format);
    vk::DeviceSize imageSize = texWidth * texHeight * pixelSize;

    if (!pixels)
    {
        throw std::runtime_error("failed to load texture image!");
    }

    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingBufferMemory;
    gpu::Buffers::createBuffer(context, imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);


    void* data = context.device.mapMemory(stagingBufferMemory, 0, imageSize);

    // Handle floats
    // TODO: find more elegant alternative
    if (format == vk::Format::eR32G32B32Sfloat || format == vk::Format::eR32G32B32A32Sfloat)
    {
        std::vector<float> floatPixels(texWidth * texHeight * (pixelSize / 4));
        for (size_t i = 0; i < floatPixels.size(); ++i)
        {
            floatPixels[i] = static_cast<float>(pixels[i]) / 255.0f;
        }
        memcpy(data, floatPixels.data(), imageSize);
    }
    else
    {
        memcpy(data, pixels, static_cast<size_t>(imageSize));
    }

    context.device.unmapMemory(stagingBufferMemory);

    stbi_image_free(pixels);

    gpu::Image::createImage(context, texWidth, texHeight, format, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, image, imageMemory);

    gpu::Image::transitionImageLayout(context, commandBufferManager, image, format, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
    gpu::Image::copyBufferToImage(context, commandBufferManager, stagingBuffer, image, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
    gpu::Image::transitionImageLayout(context, commandBufferManager, image, format, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    context.device.destroyBuffer(stagingBuffer);
    context.device.freeMemory(stagingBufferMemory);
}

void Texture::cleanup(vk::Device device)
{
    device.destroySampler(sampler);
    device.destroyImageView(imageView);
    device.destroyImage(image);
    device.freeMemory(imageMemory);
}

Texture Texture::create1x1Texture(const std::vector<uint8_t>& pixelData, const Context& context, CommandBuffers& commandBufferManager, vk::Format format)
{
    Texture texture;

    vk::DeviceSize dataSize = pixelData.size();

    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingBufferMemory;
    gpu::Buffers::createBuffer(
        context,
        dataSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer,
        stagingBufferMemory,
        false
    );

    void* data = context.device.mapMemory(stagingBufferMemory, 0, dataSize);
    memcpy(data, pixelData.data(), dataSize);
    context.device.unmapMemory(stagingBufferMemory);

    gpu::Image::createImage(
        context,
        1, 1,
        format,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        texture.image,
        texture.imageMemory
    );

    gpu::Image::transitionImageLayout(
        context,
        commandBufferManager,
        texture.image,
        format,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal
    );

    gpu::Image::copyBufferToImage(
        context,
        commandBufferManager,
        stagingBuffer,
        texture.image,
        1, 1
    );

    gpu::Image::transitionImageLayout(
        context,
        commandBufferManager,
        texture.image,
        format,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal
    );

    texture.imageView = gpu::Image::createImageView(
        context,
        texture.image,
        format,
        vk::ImageAspectFlagBits::eColor
    );

    context.device.destroyBuffer(stagingBuffer);
    context.device.freeMemory(stagingBufferMemory);

    texture.sampler = gpu::Textures::createSampler(context, vk::Filter::eNearest, vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest);
    return texture;
}

Texture Texture::create1x1TextureR(uint8_t r, const Context& context, CommandBuffers& commandBufferManager, vk::Format format)
{
    std::vector<uint8_t> pixelData = { r };
    return create1x1Texture(pixelData, context, commandBufferManager, format);
}

Texture Texture::create1x1TextureRGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a, const Context& context, CommandBuffers& commandBufferManager, vk::Format format)
{
    std::vector<uint8_t> pixelData = { r, g, b, a };
    return create1x1Texture(pixelData, context, commandBufferManager, format);
}

} // namespace gpu
} // namespace vkrt
