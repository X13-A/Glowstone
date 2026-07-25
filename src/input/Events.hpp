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

struct WindowResizeEvent
{
    int nativeWidth, nativeHeight;
    int scaledWidth, scaledHeight;
};

} // namespace input
} // namespace vkrt
