#include "core/Constants.hpp"
#include "core/Vk.hpp"

namespace vkrt {
namespace core {

const uint32_t VULKAN_API_VERSION = VK_API_VERSION_1_4;

const int MAX_FRAMES_IN_FLIGHT = 2;
const uint32_t GLFW_WINDOW_WIDTH = 1920;
const uint32_t GLFW_WINDOW_HEIGHT = 1080;
const char* GLFW_WINDOW_NAME = "Glowstone";

const char* SHADERS_DIR = "assets/shaders";
const char* SHADERS_MANIFEST = "assets/shaders/shaders.json";

// Ray tracing
const int RT_MAX_RECURSION_DEPTH = 6;
const int RT_RAYGEN_SHADER_INDEX = 0;
const int RT_MISS_SHADER_INDEX = 1;
const int RT_MAX_SAMPLES = 100000;
const int RT_CLOSEST_HIT_GENERAL_SHADER_INDEX = 2;

const int MAX_MESHES = 2048;
const int FULLSCREEN_QUAD_COUNT = 1;

#ifdef NDEBUG
const bool ENABLE_VALIDATION_LAYERS = false;
#else
const bool ENABLE_VALIDATION_LAYERS = true;
#endif

const std::vector<const char*> VALIDATION_LAYERS =
{
    "VK_LAYER_KHRONOS_validation",
};

const std::vector<const char*> REQUIRED_DEVICE_EXTENSIONS =
{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    // RTX ON
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_RAY_QUERY_EXTENSION_NAME,
    VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME
};

// Formats
const vk::Format RT_STORAGE_IMAGE_FORMAT = vk::Format::eR16G16B16A16Unorm;

const vk::Format ALBEDO_MAP_FORMAT = vk::Format::eR8G8B8A8Srgb;
const vk::Format NORMAL_MAP_FORMAT = vk::Format::eR8G8B8A8Unorm;
const vk::Format ROUGHNESS_MAP_FORMAT = vk::Format::eR8Unorm;
const vk::Format METALNESS_MAP_FORMAT = vk::Format::eR8Unorm;
const vk::Format MOTION_VECTOR_FORMAT = vk::Format::eR16G16Sfloat;

} // namespace core
} // namespace vkrt
