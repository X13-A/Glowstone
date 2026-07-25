#include "core/Time.hpp"
#include "core/Vk.hpp"


namespace vkrt {
namespace core {

double Time::lastTime = 0;
double Time::_deltaTime = 0;
uint32_t Time::frameCount = 0;

double Time::deltaTime()
{
    return Time::_deltaTime;
}

double Time::time()
{
    return glfwGetTime();
}

void Time::update()
{
    double currentTime = Time::time();
    _deltaTime = currentTime - lastTime;
    lastTime = currentTime;
    frameCount++;
}

double Time::FPS()
{
    return 1.0f / Time::deltaTime();
}

void Time::resetFrameCount()
{
    frameCount = 0;
}

uint32_t Time::getFrameCount()
{
    return frameCount;
}

} // namespace core
} // namespace vkrt
