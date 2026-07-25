#pragma once

#include "gpu/Utils.hpp"
#include "gpu/Context.hpp"
#include "gpu/CommandBuffers.hpp"
#include "core/Vk.hpp"


namespace vkrt {
namespace render {
using namespace vkrt::gpu;

class GBuffer
{
public:
    vk::Image depthImage;
    vk::DeviceMemory depthImageMemory;
    vk::ImageView depthImageView;

    vk::Image normalImage;
    vk::DeviceMemory normalImageMemory;
    vk::ImageView normalImageView;

    vk::Image albedoImage;
    vk::DeviceMemory albedoImageMemory;
    vk::ImageView albedoImageView;

    vk::Image roughnessImage;
    vk::DeviceMemory roughnessImageMemory;
    vk::ImageView roughnessImageView;

    vk::Image metalnessImage;
    vk::DeviceMemory metalnessImageMemory;
    vk::ImageView metalnessImageView;

    vk::Image velocityImage;
    vk::DeviceMemory velocityImageMemory;
    vk::ImageView velocityImageView;
public:
    void init(const Context& context, CommandBuffers& commandBufferManager, uint32_t width, uint32_t height);
    void createDepthResources(const Context& context, CommandBuffers& commandBufferManager, uint32_t width, uint32_t height);
    void createNormalResources(const Context& context, CommandBuffers& commandBufferManager, uint32_t width, uint32_t height);
    void createAlbedoResources(const Context& context, CommandBuffers& commandBufferManager, uint32_t width, uint32_t height);
    void createRoughnessResources(const Context& context, CommandBuffers& commandBufferManager, uint32_t width, uint32_t height);
    void createMetalnessResources(const Context& context, CommandBuffers& commandBufferManager, uint32_t width, uint32_t height);
    void createVelocityResources(const Context& context, CommandBuffers& commandBufferManager, uint32_t width, uint32_t height);
    void cleanup(vk::Device device);

    // Getters
    vk::ImageView getDepthImageView() const;
    vk::ImageView getNormalImageView() const;
    vk::ImageView getAlbedoImageView() const;
    vk::ImageView getRoughnessImageView() const;
	vk::ImageView getMetalnessImageView() const;
	vk::ImageView getVelocityImageView() const;
};

} // namespace render
} // namespace vkrt
