#include "gpu/GpuTimer.hpp"

#include <array>
#include <iostream>
#include "core/Constants.hpp"


namespace vkrt {
namespace gpu {
using namespace vkrt::core;

void GpuTimer::init(const Context& context)
{
    const vk::PhysicalDeviceLimits& limits = context.physicalDevice.getProperties().limits;
    uint32_t graphicsFamily = context.findQueueFamilies(context.physicalDevice).graphicsFamily.value();
    uint32_t validBits = context.physicalDevice.getQueueFamilyProperties()[graphicsFamily].timestampValidBits;

    supported = limits.timestampPeriod > 0.0f && validBits > 0;
    if (!supported)
    {
        std::cout << "GPU timestamps unsupported, frame time unavailable" << std::endl;
        return;
    }

    timestampPeriod = limits.timestampPeriod;
    timestampMask = validBits >= 64 ? ~0ull : (1ull << validBits) - 1;
    pending.assign(MAX_FRAMES_IN_FLIGHT, false);

    vk::QueryPoolCreateInfo queryPoolInfo{};
    queryPoolInfo.queryType = vk::QueryType::eTimestamp;
    queryPoolInfo.queryCount = MAX_FRAMES_IN_FLIGHT * QUERIES_PER_FRAME;

    queryPool = context.device.createQueryPool(queryPoolInfo);
}

void GpuTimer::resolve(const Context& context, uint32_t frameIndex)
{
    if (!supported || !pending[frameIndex])
    {
        return;
    }

    std::array<uint64_t, QUERIES_PER_FRAME> timestamps{};
    vk::Result result = context.device.getQueryPoolResults(
        queryPool,
        frameIndex * QUERIES_PER_FRAME,
        QUERIES_PER_FRAME,
        sizeof(timestamps),
        timestamps.data(),
        sizeof(uint64_t),
        vk::QueryResultFlagBits::e64);

    if (result != vk::Result::eSuccess)
    {
        return;
    }

    pending[frameIndex] = false;
    uint64_t elapsedTicks = ((timestamps[1] & timestampMask) - (timestamps[0] & timestampMask)) & timestampMask;
    elapsedMs = elapsedTicks * static_cast<double>(timestampPeriod) * 1e-6;
}

void GpuTimer::begin(vk::CommandBuffer commandBuffer, uint32_t frameIndex)
{
    if (!supported)
    {
        return;
    }

    commandBuffer.resetQueryPool(queryPool, frameIndex * QUERIES_PER_FRAME, QUERIES_PER_FRAME);
    commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, queryPool, frameIndex * QUERIES_PER_FRAME);
}

void GpuTimer::end(vk::CommandBuffer commandBuffer, uint32_t frameIndex)
{
    if (!supported)
    {
        return;
    }

    commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, queryPool, frameIndex * QUERIES_PER_FRAME + 1);
    pending[frameIndex] = true;
}

void GpuTimer::cleanup(const Context& context)
{
    if (!supported)
    {
        return;
    }

    context.device.destroyQueryPool(queryPool);
}

} // namespace gpu
} // namespace vkrt
