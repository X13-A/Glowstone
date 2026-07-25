#include "app/Window.hpp"
#include "core/Constants.hpp"
#include <stdexcept>
#include "input/EventManager.hpp"
#include "input/Events.hpp"


namespace vkrt {
namespace app {
using namespace vkrt::core;
using namespace vkrt::gpu;
using namespace vkrt::input;

void Window::init()
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    window = glfwCreateWindow(GLFW_WINDOW_WIDTH, GLFW_WINDOW_HEIGHT, GLFW_WINDOW_NAME, nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);

    // Set callbacks
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetFramebufferSizeCallback(window, Window::framebufferResizeCallback);
    glfwSetCursorPosCallback(window, Window::mouseCallback);
    glfwSetScrollCallback(window, Window::scrollCallbaack);
}

GLFWwindow* Window::getWindow() const
{
    return window;
}

void Window::mouseCallback(GLFWwindow* window, double xpos, double ypos)
{
    EventManager::get().trigger(MouseMoveEvent{ xpos, ypos });
}

void Window::scrollCallbaack(GLFWwindow* window, double xoffset, double yoffset)
{
    EventManager::get().trigger(MouseScrollEvent{ xoffset, yoffset });
}

void Window::setResizeCallback(std::function<void()> callback)
{
    resizeCallback = std::move(callback);
}

void Window::framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
    auto windowManager = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (windowManager->resizeCallback)
    {
        windowManager->resizeCallback();
    }
}

void Window::createSurface(Context& context)
{
    VkSurfaceKHR rawSurface;
    if (glfwCreateWindowSurface(context.instance, window, nullptr, &rawSurface) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create window surface!");
    }
    context.surface = rawSurface;
}

void Window::cleanup()
{
    glfwDestroyWindow(window);
}

} // namespace app
} // namespace vkrt
