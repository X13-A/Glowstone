#include "gpu/Context.hpp"
#include <set>
#include <iostream>
#include "core/Constants.hpp"
#include "gpu/Swapchain.hpp"
#include <regex>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace vkrt {
namespace gpu {
using namespace vkrt::core;

bool QueueFamilyIndices::isComplete()
{
    return graphicsFamily.has_value() && presentFamily.has_value();
}

void printColoredValidationMessage(const std::string& message, std::string titleColor, std::string bodyColor)
{
    std::regex pattern(R"(^([^\:]+:))", std::regex::icase);
    std::smatch match;
    std::string formattedMessage = std::regex_replace(message, std::regex(": "), ":\n", std::regex_constants::format_first_only);

    if (std::regex_search(formattedMessage, match, pattern)) {
        std::string highlighted = titleColor + match.str(1) + bodyColor;
        std::string formatted = std::regex_replace(formattedMessage, pattern, highlighted, std::regex_constants::format_first_only);
        std::cerr << std::endl << formatted << std::endl;
    }
    else
    {
        std::cerr << std::endl << formattedMessage << std::endl;
    }
}

VKAPI_ATTR vk::Bool32 VKAPI_CALL Context::debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageType, const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
{
    const char* color;
    switch (messageSeverity)
    {
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose: color = "\033[90m"; break; // gray
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo:    color = "\033[36m"; break; // cyan
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning: color = "\033[33m"; break; // yellow
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:   color = "\033[31m"; break; // red
        default: color = "\033[0m"; break; // White
    }
    printColoredValidationMessage("[Vulkan] " + std::string(pCallbackData->pMessage), color, "\033[0m");

    return vk::False;
}

void Context::init()
{
    createInstance();
    setupDebugMessenger();
}

void Context::initDevice()
{
    pickPhysicalDevice();
    createLogicalDevice();
}


void Context::setupDebugMessenger()
{
    if (!ENABLE_VALIDATION_LAYERS) return;

    vk::DebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
    createInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
    createInfo.pfnUserCallback = Context::debugCallback;

    debugMessenger = instance.createDebugUtilsMessengerEXT(createInfo);
    std::cout << "Debug messenger successfully setup" << std::endl;
}

bool Context::checkValidationLayerSupport() const
{
    std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();

    for (const char* layerName : VALIDATION_LAYERS)
    {
        bool layerFound = false;

        for (const vk::LayerProperties& layerProperties : availableLayers)
        {
            if (strcmp(layerName, layerProperties.layerName.data()) == 0)
            {
                layerFound = true;
                break;
            }
        }

        if (!layerFound)
        {
            return false;
        }
    }

    return true;
}

std::vector<const char*> Context::getRequiredExtensions() const
{
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (ENABLE_VALIDATION_LAYERS)
    {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

void Context::createInstance()
{
    VULKAN_HPP_DEFAULT_DISPATCHER.init(reinterpret_cast<PFN_vkGetInstanceProcAddr>(glfwGetInstanceProcAddress(nullptr, "vkGetInstanceProcAddr")));

    if (ENABLE_VALIDATION_LAYERS && !checkValidationLayerSupport())
    {
        throw std::runtime_error("VK validation layers requested, but not available!");
    }

    vk::ApplicationInfo appInfo{};
    appInfo.pApplicationName = GLFW_WINDOW_NAME;
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VULKAN_API_VERSION;

    vk::InstanceCreateInfo createInfo{};
    createInfo.pApplicationInfo = &appInfo;

    vk::ValidationFeatureEnableEXT enabledValidationFeatures[] =
    {
        vk::ValidationFeatureEnableEXT::eSynchronizationValidation,
    };

    vk::ValidationFeaturesEXT validationFeatures{};
    validationFeatures.enabledValidationFeatureCount = static_cast<uint32_t>(std::size(enabledValidationFeatures));
    validationFeatures.pEnabledValidationFeatures = enabledValidationFeatures;

    if (ENABLE_VALIDATION_LAYERS)
    {
        createInfo.pNext = &validationFeatures;
    }

    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    if (ENABLE_VALIDATION_LAYERS)
    {
        createInfo.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
        createInfo.ppEnabledLayerNames = VALIDATION_LAYERS.data();
    }

    instance = vk::createInstance(createInfo);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);
}

void Context::createLogicalDevice()
{
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

    float queuePriority = 1.0f;

    for (uint32_t queueFamily : uniqueQueueFamilies)
    {
        vk::DeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    vk::PhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.samplerAnisotropy = VK_TRUE;

    char* envVar = nullptr;
    size_t envVarSize = 0;
    errno_t err = _dupenv_s(&envVar, &envVarSize, "NV_ALLOW_RAYTRACING_VALIDATION");

    bool rtValidationEnabled = false;
    if (err == 0 && envVar != nullptr)
    {
        rtValidationEnabled = (std::string(envVar) == "1");
        free(envVar);
    }

    if (!rtValidationEnabled)
    {
        std::cerr << "Ray tracing validation disabled (set NV_ALLOW_RAYTRACING_VALIDATION=1 to enable)" << std::endl;
    }

    vk::PhysicalDeviceRayTracingValidationFeaturesNV validationFeatures{};

    vk::PhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{};
    bufferDeviceAddressFeatures.pNext = &validationFeatures;

    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{};
    rayTracingPipelineFeatures.pNext = &bufferDeviceAddressFeatures;

    vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{};
    accelerationStructureFeatures.pNext = &rayTracingPipelineFeatures;

    vk::PhysicalDeviceFeatures2 deviceFeatures2{};
    deviceFeatures2.features = deviceFeatures;
    deviceFeatures2.pNext = &accelerationStructureFeatures;

    physicalDevice.getFeatures2(&deviceFeatures2);

    if (!accelerationStructureFeatures.accelerationStructure)
    {
        throw std::runtime_error("Acceleration structure feature not supported!");
    }
    if (!rayTracingPipelineFeatures.rayTracingPipeline)
    {
        throw std::runtime_error("Ray tracing pipeline feature not supported!");
    }
    if (!bufferDeviceAddressFeatures.bufferDeviceAddress)
    {
        throw std::runtime_error("Buffer device address feature not supported!");
    }
    if (!validationFeatures.rayTracingValidation)
    {
        std::cerr << "RT validation features are not available." << std::endl;
    }

    accelerationStructureFeatures.accelerationStructure = VK_TRUE;
    rayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;
    bufferDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;

    if (validationFeatures.rayTracingValidation && rtValidationEnabled)
    {
        validationFeatures.rayTracingValidation = VK_TRUE;
        std::cout << "Ray tracing validation enabled!" << std::endl;
    }
    else
    {
        validationFeatures.rayTracingValidation = VK_FALSE;
    }

    vk::DeviceCreateInfo createInfo{};
    createInfo.pNext = &deviceFeatures2;
    createInfo.pEnabledFeatures = nullptr;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(REQUIRED_DEVICE_EXTENSIONS.size());
    createInfo.ppEnabledExtensionNames = REQUIRED_DEVICE_EXTENSIONS.data();

    if (ENABLE_VALIDATION_LAYERS)
    {
        createInfo.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
        createInfo.ppEnabledLayerNames = VALIDATION_LAYERS.data();
    }

    device = physicalDevice.createDevice(createInfo);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device);

    presentQueue = device.getQueue(indices.presentFamily.value(), 0);
    graphicsQueue = device.getQueue(indices.graphicsFamily.value(), 0);
}

void Context::pickPhysicalDevice()
{
    std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();

    if (devices.empty())
    {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::cout << "Available Physical Devices:\n";
    int bestRank = -1;
    for (const vk::PhysicalDevice& candidate : devices)
    {
        vk::PhysicalDeviceProperties deviceProperties = candidate.getProperties();

        std::cout << "Device: " << deviceProperties.deviceName.data() << "\n";
        std::cout << "  Type: " << vk::to_string(deviceProperties.deviceType) << "\n";
        std::cout << "  API Version: "
            << VK_VERSION_MAJOR(deviceProperties.apiVersion) << "."
            << VK_VERSION_MINOR(deviceProperties.apiVersion) << "."
            << VK_VERSION_PATCH(deviceProperties.apiVersion) << "\n";

        // Select device based on a ranking system
        const int rank = deviceTypeRank(deviceProperties.deviceType);
        if (rank > bestRank && isDeviceSuitable(candidate))
        {
            physicalDevice = candidate;
            bestRank = rank;
        }
    }

    if (!physicalDevice)
    {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
    std::cout << "Successfully found suitable GPU" << std::endl;

    vk::PhysicalDeviceProperties props = physicalDevice.getProperties();
    std::cout << "SELECTED GPU: " << props.deviceName.data() << " (Type: " << vk::to_string(props.deviceType) << ")" << std::endl;

    vk::PhysicalDeviceMemoryProperties memProps = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProps.memoryHeapCount; i++)
    {
        if (memProps.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
        {
            std::cout << "GPU VRAM: " << memProps.memoryHeaps[i].size / (1024.0f * 1024.0f * 1024.0f) << "GB" << std::endl;
        }
    }
}

int Context::deviceTypeRank(vk::PhysicalDeviceType type)
{
    // Use dGPU when suitable
    switch (type)
    {
        case vk::PhysicalDeviceType::eDiscreteGpu:   return 4;
        case vk::PhysicalDeviceType::eIntegratedGpu: return 3;
        case vk::PhysicalDeviceType::eVirtualGpu:    return 2;
        case vk::PhysicalDeviceType::eCpu:           return 1;
        default:                                     return 0;
    }
}

bool Context::isDeviceSuitable(vk::PhysicalDevice physicalDevice) const
{
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    bool extensionsSupported = checkDeviceExtensionSupport(physicalDevice);

    bool swapChainAdequate = false;
    if (extensionsSupported)
    {
        SwapChainSupportDetails swapChainSupport = Swapchain::querySwapChainSupport(physicalDevice, surface);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        std::cout << "Required VK extensions are all suported" << std::endl;
    }
    if (swapChainAdequate)
    {
        std::cout << "VK swapchain suitable" << std::endl;
    }

    vk::PhysicalDeviceFeatures supportedFeatures = physicalDevice.getFeatures();

    return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
}

bool Context::checkDeviceExtensionSupport(vk::PhysicalDevice physicalDevice) const
{
    std::vector<vk::ExtensionProperties> availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();

    std::set<std::string> requiredExtensions(REQUIRED_DEVICE_EXTENSIONS.begin(), REQUIRED_DEVICE_EXTENSIONS.end());

    for (const vk::ExtensionProperties& extension : availableExtensions)
    {
        requiredExtensions.erase(extension.extensionName.data());
    }

    if (!requiredExtensions.empty())
    {
        for (const auto& ext : requiredExtensions)
        {
            std::cerr << "[Missing Extension] " << ext << "\n";
        }
    }

    return requiredExtensions.empty();
}

QueueFamilyIndices Context::findQueueFamilies(vk::PhysicalDevice physicalDevice) const
{
    QueueFamilyIndices indices;
    std::vector<vk::QueueFamilyProperties> queueFamilies = physicalDevice.getQueueFamilyProperties();

    int i = 0;
    for (const vk::QueueFamilyProperties& queueFamily : queueFamilies)
    {
        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
        {
            indices.graphicsFamily = i;

            if (!(queueFamily.queueFlags & vk::QueueFlagBits::eCompute))
            {
                std::cout << " Error: Graphics queue does not support COMPUTE !" << std::endl;
            }
        }

        if (physicalDevice.getSurfaceSupportKHR(i, surface))
        {
            indices.presentFamily = i;
        }

        std::cout << std::endl;

        if (indices.isComplete())
        {
            break;
        }
        i++;
    }
    return indices;
}

void Context::cleanup()
{
    if (ENABLE_VALIDATION_LAYERS)
    {
        instance.destroyDebugUtilsMessengerEXT(debugMessenger);
    }
    device.destroy();
    instance.destroySurfaceKHR(surface);
    instance.destroy();
}

} // namespace gpu
} // namespace vkrt
