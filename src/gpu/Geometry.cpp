#include "gpu/Geometry.hpp"
#include "core/Vk.hpp"
#include "core/Math.hpp"

namespace vkrt {
namespace gpu {

bool Vertex::operator==(const Vertex& other) const
{
    return pos == other.pos && texCoord == other.texCoord && normal == other.normal;
}

vk::VertexInputBindingDescription Vertex::getBindingDescription()
{
    vk::VertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = vk::VertexInputRate::eVertex;
    return bindingDescription;
}

std::array<vk::VertexInputAttributeDescription, 5> Vertex::getAttributeDescriptions()
{
    std::array<vk::VertexInputAttributeDescription, 5> attributeDescriptions{};

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = vk::Format::eR32G32B32Sfloat;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = vk::Format::eR32G32Sfloat;
    attributeDescriptions[1].offset = offsetof(Vertex, texCoord);

    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = vk::Format::eR32G32B32Sfloat;
    attributeDescriptions[2].offset = offsetof(Vertex, normal);

    attributeDescriptions[3].binding = 0;
    attributeDescriptions[3].location = 3;
    attributeDescriptions[3].format = vk::Format::eR32G32B32Sfloat;
    attributeDescriptions[3].offset = offsetof(Vertex, tangent);

    attributeDescriptions[4].binding = 0;
    attributeDescriptions[4].location = 4;
    attributeDescriptions[4].format = vk::Format::eR32G32B32Sfloat;
    attributeDescriptions[4].offset = offsetof(Vertex, bitangent);

    return attributeDescriptions;
}

} // namespace gpu
} // namespace vkrt

namespace std
{
    size_t hash<vkrt::gpu::Vertex>::operator()(vkrt::gpu::Vertex const& vertex) const
    {
        size_t h1 = hash<glm::vec3>()(vertex.pos);
        size_t h2 = hash<glm::vec2>()(vertex.texCoord);
        size_t h3 = hash<glm::vec3>()(vertex.normal);
        size_t h4 = hash<glm::vec3>()(vertex.tangent);

        size_t combined = h1 ^ (h2 << 1);
        combined = (combined >> 1) ^ (h3 << 1);
        combined = (combined >> 1) ^ (h4 << 1);

        return combined;
    }
}
