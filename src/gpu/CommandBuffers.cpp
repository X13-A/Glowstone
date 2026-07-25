#include "gpu/CommandBuffers.hpp"
#include <stdexcept>
#include <iostream>


namespace vkrt {
namespace gpu {
using namespace vkrt::core;

vk::CommandBuffer CommandBuffers::beginSingleTimeCommands(vk::Device device) const
{
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    vk::CommandBuffer commandBuffer = device.allocateCommandBuffers(allocInfo).front();

    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

    commandBuffer.begin(beginInfo);
    return commandBuffer;
}

void CommandBuffers::endSingleTimeCommands(vk::Device device, vk::Queue graphicsQueue, vk::CommandBuffer commandBuffer)
{
    commandBuffer.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    graphicsQueue.submit(submitInfo, nullptr);
    graphicsQueue.waitIdle();

    device.freeCommandBuffers(commandPool, commandBuffer);
}

void CommandBuffers::createCommandBuffers(const Context& context)
{
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.commandPool = commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

    commandBuffers = context.device.allocateCommandBuffers(allocInfo);
}

void CommandBuffers::createCommandPool(const Context& context)
{
    QueueFamilyIndices queueFamilyIndices = context.findQueueFamilies(context.physicalDevice);

    vk::CommandPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

    commandPool = context.device.createCommandPool(poolInfo);
}

void CommandBuffers::cleanup(vk::Device device)
{
    device.destroyCommandPool(commandPool);
}

} // namespace gpu
} // namespace vkrt
