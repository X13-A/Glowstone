#pragma once

#include "core/Math.hpp"
#include "core/Vk.hpp"
#include <array>

namespace vkrt {
namespace gpu {

struct Vertex
{
    glm::vec3 pos = { 0, 0, 0 };
    glm::vec2 texCoord = { 0, 0 };
    glm::vec3 normal = { 0, 0, 0 };
    glm::vec3 tangent = { 0, 0, 0 };
    glm::vec3 bitangent = { 0, 0, 0 };

    bool operator==(const Vertex& other) const;

    static vk::VertexInputBindingDescription getBindingDescription();

    static std::array<vk::VertexInputAttributeDescription, 5> getAttributeDescriptions();
};

} // namespace gpu
} // namespace vkrt

namespace std
{
    template<> struct hash<vkrt::gpu::Vertex>
    {
        size_t operator()(vkrt::gpu::Vertex const& vertex) const;
    };
}
