#pragma once

#include <optional>
#include "core/Vk.hpp"
#include <vector>
#include <string>


namespace vkrt {
namespace gpu {
using namespace vkrt::core;

struct QueueFamilyIndices
{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete();
};

class Context
{
public:
    vk::Instance instance;
    vk::DebugUtilsMessengerEXT debugMessenger;

    vk::PhysicalDevice physicalDevice = nullptr;
    vk::Device device;

    vk::Queue graphicsQueue;
    vk::Queue presentQueue;
    vk::SurfaceKHR surface;

public:
    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageType, const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);

    void init();

    void initDevice();

    void setupDebugMessenger();

    bool checkValidationLayerSupport() const;

    std::vector<const char*> getRequiredExtensions() const;

    void createInstance();

    void createLogicalDevice();

    void pickPhysicalDevice();

    bool isDeviceSuitable(vk::PhysicalDevice physicalDevice) const;

    static int deviceTypeRank(vk::PhysicalDeviceType type);

    bool checkDeviceExtensionSupport(vk::PhysicalDevice physicalDevice) const;

    QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice physicalDevice) const;

    void cleanup();
};

} // namespace gpu
} // namespace vkrt
