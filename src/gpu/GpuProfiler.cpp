#include "gpu/GpuProfiler.hpp"

#include <iostream>
#include "core/Constants.hpp"


namespace vkrt {
namespace gpu {
using namespace vkrt::core;

GpuProfiler::TimedSection::TimedSection(GpuProfiler& profiler, std::string_view name)
    : profiler(profiler)
    , sectionIndex(profiler.openSection(name))
{
}

GpuProfiler::TimedSection::~TimedSection()
{
    profiler.closeSection(sectionIndex);
}

uint32_t GpuProfiler::frameQuery(uint32_t frameIndex)
{
    return frameIndex * QUERIES_PER_FRAME;
}

uint32_t GpuProfiler::sectionQuery(uint32_t frameIndex, uint32_t sectionIndex)
{
    return frameQuery(frameIndex) + (sectionIndex + 1) * QUERIES_PER_SECTION;
}

double GpuProfiler::toMilliseconds(uint64_t begin, uint64_t end) const
{
    uint64_t elapsedTicks = ((end & timestampMask) - (begin & timestampMask)) & timestampMask;
    return elapsedTicks * static_cast<double>(timestampPeriod) * 1e-6;
}

void GpuProfiler::init(const Context& context)
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
    frames.resize(MAX_FRAMES_IN_FLIGHT);
    timestamps.resize(QUERIES_PER_FRAME);

    vk::QueryPoolCreateInfo queryPoolInfo{};
    queryPoolInfo.queryType = vk::QueryType::eTimestamp;
    queryPoolInfo.queryCount = MAX_FRAMES_IN_FLIGHT * QUERIES_PER_FRAME;

    queryPool = context.device.createQueryPool(queryPoolInfo);
}

void GpuProfiler::resolve(const Context& context, uint32_t frameIndex)
{
    if (!supported)
    {
        return;
    }

    FrameRecord& frame = frames[frameIndex];
    if (!frame.pending)
    {
        return;
    }
    frame.pending = false;

    const uint32_t queryCount = (static_cast<uint32_t>(frame.sectionNames.size()) + 1) * QUERIES_PER_SECTION;
    vk::Result result = context.device.getQueryPoolResults(
        queryPool,
        frameQuery(frameIndex),
        queryCount,
        queryCount * sizeof(uint64_t),
        timestamps.data(),
        sizeof(uint64_t),
        vk::QueryResultFlagBits::e64);

    if (result != vk::Result::eSuccess)
    {
        return;
    }

    frameTimeMs = toMilliseconds(timestamps[0], timestamps[1]);

    sections.resize(frame.sectionNames.size());
    for (size_t i = 0; i < sections.size(); i++)
    {
        const size_t firstTimestamp = (i + 1) * QUERIES_PER_SECTION;
        sections[i].name = frame.sectionNames[i];
        sections[i].elapsedMs = toMilliseconds(timestamps[firstTimestamp], timestamps[firstTimestamp + 1]);
    }
}

void GpuProfiler::beginFrame(vk::CommandBuffer commandBuffer, uint32_t frameIndex)
{
    if (!supported)
    {
        return;
    }

    recordingCommandBuffer = commandBuffer;
    recordingFrame = frameIndex;

    FrameRecord& frame = frames[frameIndex];
    frame.sectionNames.clear();
    frame.pending = true;

    commandBuffer.resetQueryPool(queryPool, frameQuery(frameIndex), QUERIES_PER_FRAME);
    commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, queryPool, frameQuery(frameIndex));
}

void GpuProfiler::endFrame()
{
    if (!supported)
    {
        return;
    }

    recordingCommandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, queryPool, frameQuery(recordingFrame) + 1);
}

uint32_t GpuProfiler::openSection(std::string_view name)
{
    if (!supported)
    {
        return INVALID_SECTION;
    }

    FrameRecord& frame = frames[recordingFrame];
    if (frame.sectionNames.size() >= MAX_SECTIONS)
    {
        return INVALID_SECTION;
    }

    const uint32_t sectionIndex = static_cast<uint32_t>(frame.sectionNames.size());
    frame.sectionNames.emplace_back(name);

    recordingCommandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, queryPool, sectionQuery(recordingFrame, sectionIndex));
    return sectionIndex;
}

void GpuProfiler::closeSection(uint32_t sectionIndex)
{
    if (sectionIndex == INVALID_SECTION)
    {
        return;
    }

    recordingCommandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, queryPool, sectionQuery(recordingFrame, sectionIndex) + 1);
}

void GpuProfiler::cleanup(const Context& context)
{
    if (!supported)
    {
        return;
    }

    context.device.destroyQueryPool(queryPool);
}

} // namespace gpu
} // namespace vkrt
