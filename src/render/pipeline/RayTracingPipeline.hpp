#pragma once

#include "gpu/Context.hpp"
#include "scene/Model.hpp"


namespace vkrt {
namespace render {
using namespace vkrt::gpu;
using namespace vkrt::scene;

struct SceneData
{
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewInverse;
    glm::mat4 projInverse;
    glm::vec3 cameraPos;
    uint32_t recursionDepth;
    glm::vec2 nearFar;
    int resolutionX;
    int resolutionY;
    int spp;
    int risCandidates;
    int reservoirParity;
};

// Screen-space reservoir persisted across frames
// Must match PackedReservoir in restir.slang
struct PackedReservoir
{
    glm::vec3 y_position;
    uint32_t y_flags;
    glm::vec3 y_normal;
    float W;
    glm::vec3 y_le;
    uint32_t M;
    glm::vec3 q_normal;
    float q_depth;
};

struct InstanceData
{
    glm::mat4 normalMatrix;
    uint32_t meshOffset;
    glm::vec3 padding;
};

struct MeshData
{
    uint32_t indexOffset;
    uint32_t vertexOffset;
};

struct PushConstants
{
    uint32_t frameCount;
    uint32_t rng;
};

class RayTracingPipeline
{
private:
    // Pipeline
    vk::Pipeline pipeline;
    vk::PipelineLayout pipelineLayout;

    // Shader binding table
    vk::Buffer sbtBuffer;
    vk::DeviceMemory sbtBufferMemory;
    vk::StridedDeviceAddressRegionKHR raygenSbtEntry{};
    vk::StridedDeviceAddressRegionKHR missSbtEntry{};
    vk::StridedDeviceAddressRegionKHR hitSbtEntry{};
    vk::StridedDeviceAddressRegionKHR callableSbtEntry{};

    // Storage Image
    // TODO: Ping-pong ?
    uint32_t storageImageWidth;
    uint32_t storageImageHeight;
    
    vk::Image storageImage;
    vk::DeviceMemory storageImageMemory;
    vk::ImageView storageImageView;

    vk::Image last_storageImage;
    vk::DeviceMemory last_storageImageMemory;
    vk::ImageView last_storageImageView;

    // Ping-ponged reservoir buffers
    vk::Buffer reservoirBuffers[2];
    vk::DeviceMemory reservoirBufferMemories[2];
    vk::DeviceSize reservoirBufferSize;
    bool reservoirsNeedClear;

    // Uniform Buffer
    vk::Buffer uniformBuffer;
    vk::DeviceMemory uniformBufferMemory;
    void* uniformBufferMapped;
    vk::DeviceSize uniformBufferStride;

    // Descriptor Set
    vk::DescriptorPool descriptorPool;
    vk::DescriptorSet descriptorSet;

    // buffers
    vk::Buffer instanceDataBuffer;
    vk::DeviceMemory instanceDataBufferMemory;

    vk::Buffer globalIndexBuffer;
    vk::DeviceMemory globalIndexBufferMemory;

    vk::Buffer meshDataBuffer;
    vk::DeviceMemory meshDataBufferMemory;

    vk::Buffer globalVertexBuffer;
    vk::DeviceMemory globalVertexBufferMemory;

    // Sampler
    vk::Sampler globalTextureSampler;
    vk::Sampler pointTextureSampler;

    int sampleCount;

public:
    void init(const Context& context, uint32_t width, uint32_t height);
    void writeDescriptors(const Context& context, CommandBuffers& commandBufferManager, const std::vector<Model>& models, vk::AccelerationStructureKHR tlas, vk::ImageView depthImageView, vk::ImageView normalsImageView, vk::ImageView albedoImageView, vk::ImageView roughnessImageView, vk::ImageView metalnessImageView, vk::ImageView velocityImageView);

    void createRayTracingPipelineLayout(const Context& context);
    void createRayTracingPipeline(const Context& context);
    void createShaderBindingTable(const Context& context);

    void createRayTracingResources(const Context& context, CommandBuffers& commandBufferManager, vk::AccelerationStructureKHR tlas, const std::vector<Model>& models, std::vector<vk::ImageView>& outAlbedoTextureViews, std::vector<vk::ImageView>& outNormalTextureViews, std::vector<vk::ImageView>& outRoughnessTextureViews, std::vector<vk::ImageView>& outMetalnessTextureViews);

    void createDescriptorPool(const Context& context);
    void createDescriptorSet(const Context& context);
    void writeDescriptorSet(const Context& context, vk::ImageView depthImageView, vk::ImageView normalsImageView, vk::ImageView albedoImageView, vk::ImageView roughnessImageView, vk::ImageView metalnessImageView, vk::ImageView velocityImageView, vk::AccelerationStructureKHR tlas, const std::vector<vk::ImageView>& albedoTextureViews, const std::vector<vk::ImageView>& normalTextureViews, const std::vector<vk::ImageView>& roughnessTextureViews, const std::vector<vk::ImageView>& metalnessTextureViews);

    void createStorageImage(const Context& context, uint32_t width, uint32_t height);
    void createReservoirBuffers(const Context& context, uint32_t width, uint32_t height);
    void createUniformBuffer(const Context& context);
    void updateUniformBuffer(const SceneData& sceneData, uint32_t currentFrame);

    void traceRays(vk::CommandBuffer commandBuffer, uint32_t frameCount, uint32_t currentFrame);
    void handleResize(const Context& context, uint32_t width, uint32_t height, vk::ImageView depthImageView, vk::ImageView normalsImageView, vk::ImageView albedoImageView, vk::ImageView roughnessImageView, vk::ImageView metalnessImageView, vk::ImageView velocityImageView);
    void reloadShaders(const Context& context);
    void cleanup(vk::Device device);

    // Getters
    vk::ImageView getStorageImageView() const;
    vk::ImageView getLastStorageImageView() const;
    vk::Image getStorageImage() const;
    vk::Image getLastStorageImage() const;
    vk::DescriptorSet getDescriptorSet() const;
    vk::PipelineLayout getPipelineLayout() const;
    uint32_t getStorageImageWidth() const;
    uint32_t getStorageImageHeight() const;
};

} // namespace render
} // namespace vkrt
