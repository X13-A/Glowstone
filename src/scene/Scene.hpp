#pragma once
#include <vector>
#include "scene/Model.hpp"
#include "gpu/Context.hpp"
#include "gpu/CommandBuffers.hpp"
#include "core/Vk.hpp"


namespace vkrt {
namespace scene {
using namespace vkrt::core;
using namespace vkrt::gpu;
using namespace vkrt::assets;

struct ModelLoadInfo
{
	std::string name;
	std::string objPath;
	glm::vec3 position;
	glm::vec3 scale;
	glm::vec3 rotation;
};

class Scene
{
private:
	static std::vector<Model> models; // The loaded models
	static std::vector<ModelInfo> modelInfos; // Information on models, fetched at runtime
	static const ModelLoadInfo modelLoadInfos[]; // Configuration to load model files

public:	
	static uint32_t getModelCount();
	static uint32_t getMaterialCount();
	static uint32_t getMeshCount();
	static const std::vector<Model>& getModels();
	static void fetchModels();
	static void loadModels(const Context& context, CommandBuffers& commandBufferManager, vk::DescriptorPool descriptorPool);
	static void update();
	static void cleanup(vk::Device device);
};

} // namespace scene
} // namespace vkrt
