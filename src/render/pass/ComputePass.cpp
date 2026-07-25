#include "render/pass/ComputePass.hpp"
#include "gpu/Utils.hpp"
#include "core/Utils.hpp"
#include <stdexcept>


namespace vkrt {
namespace render {
using namespace vkrt::core;
using namespace vkrt::gpu;

void ComputePass::createDescriptorSetLayout(const Context& context)
{
    auto bindings = getDescriptorBindings();

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    descriptorSetLayout = context.device.createDescriptorSetLayout(layoutInfo);
}

void ComputePass::createPipelineLayout(const Context& context, vk::PushConstantRange* pushConstantRange)
{
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    if (pushConstantRange)
    {
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange;
    }

    pipelineLayout = context.device.createPipelineLayout(pipelineLayoutInfo);
}

void ComputePass::createDescriptorPool(const Context& context)
{
    auto poolSizes = getDescriptorPoolSizes();

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;

    descriptorPool = context.device.createDescriptorPool(poolInfo);
}

void ComputePass::allocateDescriptorSet(const Context& context)
{
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    descriptorSet = context.device.allocateDescriptorSets(allocInfo).front();
}

void ComputePass::createComputePipeline(const Context& context, const std::string& shaderPath)
{
    std::vector<char> computeCode = readFile(shaderPath);
    vk::ShaderModule computeModule = gpu::Shaders::createShaderModule(context, computeCode);

    vk::PipelineShaderStageCreateInfo shaderStage{};
    shaderStage.stage = vk::ShaderStageFlagBits::eCompute;
    shaderStage.module = computeModule;
    shaderStage.pName = "main";

    vk::ComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.stage = shaderStage;

    computePipeline = context.device.createComputePipeline(nullptr, pipelineInfo).value;

    context.device.destroyShaderModule(computeModule);
}

void ComputePass::init(const Context& context)
{
    createDescriptorSetLayout(context);
    createPipelineLayout(context, getPushConstantRange());
    createDescriptorPool(context);
    allocateDescriptorSet(context);
    createComputePipeline(context, getShaderPath());
}

void ComputePass::reloadShaders(const Context& context)
{
    context.device.destroyPipeline(computePipeline);
    createComputePipeline(context, getShaderPath());
}

void ComputePass::dispatch(vk::CommandBuffer commandBuffer, uint32_t width, uint32_t height)
{
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0, descriptorSet, {});

    uint32_t groupCountX = (width + workgroupSizeX - 1) / workgroupSizeX;
    uint32_t groupCountY = (height + workgroupSizeY - 1) / workgroupSizeY;
    uint32_t groupCountZ = workgroupSizeZ;

    commandBuffer.dispatch(groupCountX, groupCountY, groupCountZ);
}

void ComputePass::cleanup(const Context& context)
{
    if (computePipeline)
    {
        context.device.destroyPipeline(computePipeline);
        computePipeline = nullptr;
    }
    if (pipelineLayout)
    {
        context.device.destroyPipelineLayout(pipelineLayout);
        pipelineLayout = nullptr;
    }
    if (descriptorPool)
    {
        context.device.destroyDescriptorPool(descriptorPool);
        descriptorPool = nullptr;
    }
    if (descriptorSetLayout)
    {
        context.device.destroyDescriptorSetLayout(descriptorSetLayout);
        descriptorSetLayout = nullptr;
    }
}

} // namespace render
} // namespace vkrt
