#pragma once

#include "core/Vk.hpp"
#include "gpu/Context.hpp"
#include <functional>
#include "entt.hpp"


namespace vkrt {
namespace app {
using namespace vkrt::core;
using namespace vkrt::gpu;

class Window
{
private:
    GLFWwindow* window;
    std::function<void()> resizeCallback;

public:
    GLFWwindow* getWindow() const;

    void init();
    void setResizeCallback(std::function<void()> callback);
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallbaack(GLFWwindow* window, double xoffset, double yoffset);
    void createSurface(Context& context);
    void cleanup();
};

} // namespace app
} // namespace vkrt
