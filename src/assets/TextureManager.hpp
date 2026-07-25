#pragma once
#include "gpu/Texture.hpp"
#include <map>
#include <string>
#include <utility>


namespace vkrt {
namespace assets {
using namespace vkrt::gpu;

// Owns every texture in the scene and hands out shared references
class TextureManager
{
public:
	static Texture errorAlbedoTexture;
	static Texture errorNormalTexture;
	static Texture errorRoughnessTexture;
	static Texture errorMetalnessTexture;

	static void loadTextures(const Context& context, CommandBuffers& commandBufferManager);

	static const Texture& acquire(const std::string& path, vk::Format format, const Context& context, CommandBuffers& commandBufferManager);
	static const Texture& acquireSolidRGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a, vk::Format format, const Context& context, CommandBuffers& commandBufferManager);
	static const Texture& acquireSolidR(uint8_t r, vk::Format format, const Context& context, CommandBuffers& commandBufferManager);

	static void cleanup(vk::Device device);

private:
	// A file may be sampled under multiple formats, so the format is part of the key
	static std::map<std::pair<std::string, vk::Format>, Texture> fileTextures;

	// Solid textures contain a single color, their size is 1x1
	// They are used when no control texture is present for an attribute (e.g albedo, roughness...)
	static std::map<std::pair<uint32_t, vk::Format>, Texture> solidTextures;
};

} // namespace assets
} // namespace vkrt
