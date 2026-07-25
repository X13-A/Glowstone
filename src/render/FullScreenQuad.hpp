#pragma once
#include "gpu/Geometry.hpp"
#include "gpu/Context.hpp"
#include "gpu/CommandBuffers.hpp"


namespace vkrt {
namespace render {
using namespace vkrt::gpu;

struct FullScreenQuadUBO
{
	float time;
};

class FullScreenQuad
{
public:
	std::vector<Vertex> vertices;
	vk::Buffer vertexBuffer;
	vk::DeviceMemory vertexBufferMemory;

	std::vector<vk::Buffer> uniformBuffers;
	std::vector<vk::DeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;

	std::vector<vk::DescriptorSet> descriptorSets;
	vk::Sampler gBufferSampler;

public:
	void init(const Context& context, CommandBuffers& commandBufferManager, vk::DescriptorPool pool, vk::ImageView depthImageView, vk::ImageView normalImageView, vk::ImageView albedoImageView, vk::ImageView roughnessImageView, vk::ImageView metalnessImageView, vk::ImageView velocityImageView);
	void createDescriptorSets(const Context& context, vk::DescriptorPool descriptorPool);
	void writeDescriptorSets(const Context& context, vk::ImageView depthImageView, vk::ImageView normalImageView, vk::ImageView albedoImageView, vk::ImageView roughnessImageView, vk::ImageView metalnessImageView, vk::ImageView velocityImageView);
	void createUniformBuffers(const Context& context);
	void cleanup(vk::Device device);
};

} // namespace render
} // namespace vkrt
