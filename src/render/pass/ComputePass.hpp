#pragma once

#include "core/Vk.hpp"
#include "gpu/Context.hpp"
#include <vector>
#include <string>

// Base class for compute passes

namespace vkrt {
namespace render {
using namespace vkrt::core;
using namespace vkrt::gpu;

class ComputePass
{
protected:
    vk::Pipeline computePipeline = nullptr;
    vk::PipelineLayout pipelineLayout = nullptr;
    vk::DescriptorPool descriptorPool = nullptr;
    vk::DescriptorSet descriptorSet = nullptr;
    vk::DescriptorSetLayout descriptorSetLayout = nullptr;

    uint32_t workgroupSizeX = 8;
    uint32_t workgroupSizeY = 8;
    uint32_t workgroupSizeZ = 1;

    virtual std::string getShaderPath() const = 0;
    virtual std::vector<vk::DescriptorSetLayoutBinding> getDescriptorBindings() const = 0;
    virtual std::vector<vk::DescriptorPoolSize> getDescriptorPoolSizes() const = 0;
    virtual vk::PushConstantRange* getPushConstantRange() const { return nullptr; }

    void createDescriptorSetLayout(const Context& context);
    void createPipelineLayout(const Context& context, vk::PushConstantRange* pushConstantRange = nullptr);
    void createDescriptorPool(const Context& context);
    void allocateDescriptorSet(const Context& context);
    void createComputePipeline(const Context& context, const std::string& shaderPath);

public:
    virtual ~ComputePass() = default;

    virtual void init(const Context& context);
    virtual void reloadShaders(const Context& context);
    virtual void dispatch(vk::CommandBuffer commandBuffer, uint32_t width, uint32_t height);
    virtual void cleanup(const Context& context);

    vk::DescriptorSet getDescriptorSet() const { return descriptorSet; }
    vk::PipelineLayout getPipelineLayout() const { return pipelineLayout; }
};

} // namespace render
} // namespace vkrt
