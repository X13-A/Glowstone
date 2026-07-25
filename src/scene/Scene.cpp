#include "scene/Scene.hpp"
#include <chrono>
#include "core/Time.hpp"


namespace vkrt {
namespace scene {
using namespace vkrt::core;
using namespace vkrt::gpu;
using namespace vkrt::assets;

const ModelLoadInfo Scene::modelLoadInfos[] =
{
	{
		"atrium",
		"assets/models/Atrium/atrium.obj",
		glm::vec3(0, 0, 0),
		glm::vec3(1, 1, 1),
		glm::vec3(0, 0, 0)
	},
	{
		"companion_cube",
		"assets/models/companion_cube/companion.obj",
		glm::vec3(0, 1.2, 0),
		glm::vec3(1, 1, 1),
		glm::vec3(0, 0, 0)
	},
	{
		"wall",
		"assets/models/wall/quad.obj",
		glm::vec3(0, -10, 0),
		glm::vec3(1, 1, 1),
		glm::vec3(0, 0, 0)
	},
};

std::vector<Model> Scene::models = {};
std::vector<ModelInfo> Scene::modelInfos = {};

uint32_t Scene::getModelCount()
{
	return sizeof(Scene::modelLoadInfos) / sizeof(ModelLoadInfo);
}

uint32_t Scene::getMaterialCount()
{
	uint32_t count = 0;
	for (const ModelInfo& model : modelInfos)
	{
		count += model.materials.size();
	}
	return count;
}

uint32_t Scene::getMeshCount()
{
	uint32_t count = 0;
	for (const ModelInfo& model : modelInfos)
	{
		count += model.meshes.size();
	}
	return count;
}


const std::vector<Model>& Scene::getModels()
{
	return models;
}

void Scene::fetchModels()
{
	for (const ModelLoadInfo& loadInfo : modelLoadInfos)
	{
		ModelInfo info = ObjLoader::loadObj(loadInfo.objPath);
		modelInfos.push_back(info);
	}
}

void Scene::loadModels(const Context& context, CommandBuffers& commandBufferManager, vk::DescriptorPool descriptorPool)
{
	for (int i = 0; i < modelInfos.size(); i++)
	{
		ModelInfo info = modelInfos[i];
		ModelLoadInfo loadInfo = modelLoadInfos[i];

		Model model;
		model.name = loadInfo.name;
		model.transform.setPosition(loadInfo.position);
		model.transform.setScale(loadInfo.scale);
		model.transform.setRotation(loadInfo.rotation);

		model.load(info, context, commandBufferManager, descriptorPool);
		models.push_back(model);
	}
}

void Scene::update()
{
	// Pass
}

void Scene::cleanup(vk::Device device)
{
	for (Model& model : models)
	{
		model.cleanup(device);
	}
	models.clear();
}

} // namespace scene
} // namespace vkrt
