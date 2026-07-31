#pragma once

#include "core/Vk.hpp"


namespace vkrt {
namespace render {

/// <summary>
/// Pass recorded over the presented swapchain image, after the scene and before submission.
/// Lets the renderer composite content it knows nothing about.
/// </summary>
class OverlayPass
{
public:
    virtual ~OverlayPass() = default;

    virtual void record(vk::CommandBuffer commandBuffer, uint32_t imageIndex) = 0;
};

} // namespace render
} // namespace vkrt
