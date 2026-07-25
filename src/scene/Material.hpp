#pragma once
#include "gpu/Texture.hpp"
#include "assets/ObjLoader.hpp"


namespace vkrt {
namespace scene {
using namespace vkrt::assets;
using namespace vkrt::gpu;

class Material
{
public:
	// Owned by TextureManager, shared between materials, never null after init
	const Texture* albedoMap = nullptr;
	const Texture* normalMap = nullptr;
	const Texture* roughnessMap = nullptr;
	const Texture* metalnessMap = nullptr;

	vk::DescriptorSet descriptorSet;
	bool hasError = false;

	void init(const PBRMaterialInfo& info, const Context& context, CommandBuffers& commandBufferManager, vk::DescriptorPool descriptorPool, bool hasError);
	void createDescriptorSet(const Context& context, vk::DescriptorSetLayout materialDescriptorSetLayout, vk::DescriptorPool descriptorPool);
	void cleanup(vk::Device device);
};

} // namespace scene
} // namespace vkrt
