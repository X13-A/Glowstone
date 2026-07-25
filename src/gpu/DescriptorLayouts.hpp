#pragma once
#include "core/Vk.hpp"
#include "gpu/Context.hpp"


namespace vkrt {
namespace gpu {
using namespace vkrt::core;

class DescriptorLayouts
{
private:
	static vk::DescriptorSetLayout modelLayout;
	static vk::DescriptorSetLayout materialLayout;
	static vk::DescriptorSetLayout fullScreenQuadLayout;
	static vk::DescriptorSetLayout rayTracingDescriptorSetLayout;

public:
	static void createLayouts(const Context& context);
	static void createModelLayout(const Context& context);
	static void createMaterialLayout(const Context& context);
	static void createFullScreenQuadLayout(const Context& context);
	static void createRayTracingDescriptorSetLayout(const Context& context);

	static vk::DescriptorSetLayout getModelLayout();
	static vk::DescriptorSetLayout getMaterialLayout();
	static vk::DescriptorSetLayout getFullScreenQuadLayout();
	static vk::DescriptorSetLayout getRayTracingLayout();

	static void cleanup(vk::Device device);
};

} // namespace gpu
} // namespace vkrt
