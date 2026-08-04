#pragma once
#include "entt.hpp"


namespace vkrt {
namespace input {

struct MouseMoveEvent 
{
    double xPos, yPos;
};

struct MouseOffsetEvent
{
    double xOffset, yOffset;
};

struct MouseScrollEvent
{
    double xOffset, yOffset;
};

struct RequestResourceResizeEvent
{
    int nativeWidth, nativeHeight;
    int scaledWidth, scaledHeight;
};

struct RequestShaderReloadEvent
{
};

struct RequestSamplingModeChangeEvent
{
    int mode;
};

struct RequestRenderScaleChangeEvent
{
    float scale;
};

struct RequestFrameAccumulationChangeEvent
{
    bool enabled;
};

} // namespace input
} // namespace vkrt
