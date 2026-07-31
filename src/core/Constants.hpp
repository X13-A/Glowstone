#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "core/Vk.hpp"


namespace vkrt {
namespace core {

extern const uint32_t VULKAN_API_VERSION;

extern const int MAX_FRAMES_IN_FLIGHT;
extern const uint32_t GLFW_WINDOW_WIDTH;
extern const uint32_t GLFW_WINDOW_HEIGHT;
extern const char* GLFW_WINDOW_NAME;

extern const char* SHADERS_DIR;
extern const char* SHADERS_MANIFEST;

extern const bool ENABLE_VALIDATION_LAYERS;
extern const std::vector<const char*> VALIDATION_LAYERS;
extern const std::vector<const char*> REQUIRED_DEVICE_EXTENSIONS;

// Ray tracing
extern const int RT_MAX_RECURSION_DEPTH;
extern const int RT_RAYGEN_SHADER_INDEX;
extern const int RT_MISS_SHADER_INDEX;
extern const int RT_MAX_SAMPLES;
extern const int RT_CLOSEST_HIT_GENERAL_SHADER_INDEX;

extern const int MAX_MESHES;
extern const int FULLSCREEN_QUAD_COUNT;

// Formats
extern const vk::Format RT_STORAGE_IMAGE_FORMAT;

extern const vk::Format ALBEDO_MAP_FORMAT;
extern const vk::Format NORMAL_MAP_FORMAT;
extern const vk::Format ROUGHNESS_MAP_FORMAT;
extern const vk::Format METALNESS_MAP_FORMAT;
extern const vk::Format MOTION_VECTOR_FORMAT;

} // namespace core
} // namespace vkrt
