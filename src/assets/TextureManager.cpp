#include "assets/TextureManager.hpp"


namespace vkrt {
namespace assets {
using namespace vkrt::gpu;

Texture TextureManager::errorAlbedoTexture = {};
Texture TextureManager::errorNormalTexture = {};
Texture TextureManager::errorRoughnessTexture = {};
Texture TextureManager::errorMetalnessTexture = {};

std::map<std::pair<std::string, vk::Format>, Texture> TextureManager::fileTextures;
std::map<std::pair<uint32_t, vk::Format>, Texture> TextureManager::solidTextures;

namespace {

uint32_t packRGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
	return (static_cast<uint32_t>(r) << 24) | (static_cast<uint32_t>(g) << 16) |
	       (static_cast<uint32_t>(b) << 8) | static_cast<uint32_t>(a);
}

} // namespace

void TextureManager::loadTextures(const Context& context, CommandBuffers& commandBufferManager)
{
	errorAlbedoTexture.init("assets/textures/error/albedo.png", context, commandBufferManager, ALBEDO_MAP_FORMAT);
	errorNormalTexture.init("assets/textures/error/normal.png", context, commandBufferManager, NORMAL_MAP_FORMAT);
	errorRoughnessTexture.init("assets/textures/error/roughness.jpg", context, commandBufferManager, ROUGHNESS_MAP_FORMAT);
	errorMetalnessTexture.init("assets/textures/error/metallic.jpg", context, commandBufferManager, METALNESS_MAP_FORMAT);
}

const Texture& TextureManager::acquire(const std::string& path, vk::Format format, const Context& context, CommandBuffers& commandBufferManager)
{
	auto [it, inserted] = fileTextures.try_emplace({ path, format });
	if (inserted)
	{
		it->second.init(path, context, commandBufferManager, format);
	}
	return it->second;
}

const Texture& TextureManager::acquireSolidRGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a, vk::Format format, const Context& context, CommandBuffers& commandBufferManager)
{
	auto [it, inserted] = solidTextures.try_emplace({ packRGBA(r, g, b, a), format });
	if (inserted)
	{
		it->second = Texture::create1x1TextureRGBA(r, g, b, a, context, commandBufferManager, format);
	}
	return it->second;
}

const Texture& TextureManager::acquireSolidR(uint8_t r, vk::Format format, const Context& context, CommandBuffers& commandBufferManager)
{
	auto [it, inserted] = solidTextures.try_emplace({ packRGBA(r, 0, 0, 0), format });
	if (inserted)
	{
		it->second = Texture::create1x1TextureR(r, context, commandBufferManager, format);
	}
	return it->second;
}

void TextureManager::cleanup(vk::Device device)
{
	for (auto& [key, texture] : fileTextures)
	{
		texture.cleanup(device);
	}
	fileTextures.clear();

	for (auto& [key, texture] : solidTextures)
	{
		texture.cleanup(device);
	}
	solidTextures.clear();

	errorAlbedoTexture.cleanup(device);
	errorNormalTexture.cleanup(device);
	errorRoughnessTexture.cleanup(device);
	errorMetalnessTexture.cleanup(device);
}

} // namespace assets
} // namespace vkrt
