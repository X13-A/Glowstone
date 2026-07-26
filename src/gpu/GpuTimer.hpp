#pragma once

#include <vector>
#include "core/Vk.hpp"
#include "gpu/Context.hpp"


namespace vkrt {
namespace gpu {
using namespace vkrt::core;

class GpuTimer
{
public:
    void init(const Context& context);
    void cleanup(const Context& context);

    void resolve(const Context& context, uint32_t frameIndex);
    void begin(vk::CommandBuffer commandBuffer, uint32_t frameIndex);
    void end(vk::CommandBuffer commandBuffer, uint32_t frameIndex);

    double getElapsedMs() const { return elapsedMs; }

private:
    static constexpr uint32_t QUERIES_PER_FRAME = 2;

    vk::QueryPool queryPool;
    float timestampPeriod = 0.0f;
    uint64_t timestampMask = 0;
    bool supported = false;
    double elapsedMs = 0.0;
    std::vector<bool> pending;
};

} // namespace gpu
} // namespace vkrt
