#pragma once
#include <vector>
#include "gpu/Context.hpp"
#include "core/Constants.hpp"

#include "core/Vk.hpp"


namespace vkrt {
namespace gpu {
using namespace vkrt::core;

class CommandBuffers
{
public:
    vk::CommandPool commandPool;
    std::vector<vk::CommandBuffer> commandBuffers;

public:
    vk::CommandBuffer beginSingleTimeCommands(vk::Device device) const;
    void endSingleTimeCommands(vk::Device device, vk::Queue graphicsQueue, vk::CommandBuffer commandBuffer);
    void createCommandBuffers(const Context& context);
    void createCommandPool(const Context& context);
    void cleanup(vk::Device device);
};

} // namespace gpu
} // namespace vkrt
