#pragma once

#include <string>
#include <string_view>
#include <vector>
#include "core/Vk.hpp"
#include "gpu/Context.hpp"


namespace vkrt {
namespace gpu {
using namespace vkrt::core;

class GpuProfiler
{
public:
    struct Section
    {
        std::string name;
        double elapsedMs = 0.0;
    };

    // Times every command recorded during its lifetime
    class TimedSection
    {
    public:
        TimedSection(GpuProfiler& profiler, std::string_view name);
        ~TimedSection();

        TimedSection(const TimedSection&) = delete;
        TimedSection(TimedSection&&) = delete;
        TimedSection& operator=(const TimedSection&) = delete;
        TimedSection& operator=(TimedSection&&) = delete;

    private:
        GpuProfiler& profiler;
        const uint32_t sectionIndex;
    };

public:
    void init(const Context& context);
    void cleanup(const Context& context);

    void resolve(const Context& context, uint32_t frameIndex);

    void beginFrame(vk::CommandBuffer commandBuffer, uint32_t frameIndex);
    void endFrame();

    double getFrameTimeMs() const { return frameTimeMs; }
    const std::vector<Section>& getSections() const { return sections; }

private:
    static constexpr uint32_t MAX_SECTIONS = 32;
    static constexpr uint32_t QUERIES_PER_SECTION = 2;
    static constexpr uint32_t QUERIES_PER_FRAME = (MAX_SECTIONS + 1) * QUERIES_PER_SECTION;
    static constexpr uint32_t INVALID_SECTION = ~0u;

    uint32_t openSection(std::string_view name);
    void closeSection(uint32_t sectionIndex);

    static uint32_t frameQuery(uint32_t frameIndex);
    static uint32_t sectionQuery(uint32_t frameIndex, uint32_t sectionIndex);
    double toMilliseconds(uint64_t begin, uint64_t end) const;

    struct FrameRecord
    {
        std::vector<std::string> sectionNames;
        bool pending = false;
    };

    vk::QueryPool queryPool;
    float timestampPeriod = 0.0f;
    uint64_t timestampMask = 0;
    bool supported = false;

    vk::CommandBuffer recordingCommandBuffer;
    uint32_t recordingFrame = 0;

    std::vector<FrameRecord> frames;
    std::vector<uint64_t> timestamps;

    std::vector<Section> sections;
    double frameTimeMs = 0.0;
};

} // namespace gpu
} // namespace vkrt
