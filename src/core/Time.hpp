#pragma once
#include <stdint.h>


namespace vkrt {
namespace core {

class Time
{
private:
    static double lastTime;
    static double _deltaTime;
    static uint32_t frameCount;

public:
    /// <summary>
    /// Returns the time elapsed between the current and the last frame
    /// </summary>
    static double deltaTime();

    /// <summary>
    /// Updates the time variables. Call this at the beginning of every frame.
    /// </summary>
    static void update();

    /// <summary>
    /// GLFW time
    /// </summary>
    static double time();

    static double FPS();

    static void resetFrameCount();
    static unsigned int getFrameCount();
};

} // namespace core
} // namespace vkrt
