#pragma once

#include "core/Constants.hpp"
#include "gpu/Context.hpp"
#include "gpu/CommandBuffers.hpp"
#include <string>


namespace vkrt {
namespace gpu {
using namespace vkrt::core;

class Texture
{
public:
    vk::Image image;
    vk::DeviceMemory imageMemory;
    vk::ImageView imageView;
    vk::Sampler sampler;

public:
    void init(std::string path, const Context& context, CommandBuffers& commandBufferManager, vk::Format format);
    void createImageView(const Context& context, vk::Format format);
    void createImage(std::string path, const Context& context, CommandBuffers& commandBufferManager, vk::Format format);
    void cleanup(vk::Device device);

    static Texture create1x1Texture(const std::vector<uint8_t>& pixelData, const Context& context, CommandBuffers& commandBufferManager, vk::Format format);
    static Texture create1x1TextureRGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a, const Context& context, CommandBuffers& commandBufferManager, vk::Format format);
    static Texture create1x1TextureR(uint8_t r, const Context& context, CommandBuffers& commandBufferManager, vk::Format format);
};

} // namespace gpu
} // namespace vkrt
