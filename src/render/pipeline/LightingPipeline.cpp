#include "render/pipeline/LightingPipeline.hpp"
#include <stdexcept>
#include "core/Utils.hpp"
#include "gpu/DescriptorLayouts.hpp"
#include "core/Settings.hpp"


namespace vkrt {
namespace render {
using namespace vkrt::core;
using namespace vkrt::gpu;
using namespace vkrt::scene;

void LightingPipeline::init(const Context& context, CommandBuffers& commandBufferManager, vk::Format swapChainImageFormat)
{
    createRenderPasses(context, swapChainImageFormat);
    createPipelineLayouts(context);
    createPipeline(context);
}

void LightingPipeline::createRenderPasses(const Context& context, vk::Format swapChainImageFormat)
{
    vk::AttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = vk::SampleCountFlagBits::e1;
    colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
    colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::AttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::SubpassDescription lightingSubpass{};
    lightingSubpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    lightingSubpass.colorAttachmentCount = 1;
    lightingSubpass.pColorAttachments = &colorAttachmentRef;

    std::array<vk::AttachmentDescription, 1> lightingAttachments = { colorAttachment };
    vk::RenderPassCreateInfo lightingRenderPassInfo{};
    lightingRenderPassInfo.attachmentCount = static_cast<uint32_t>(lightingAttachments.size());
    lightingRenderPassInfo.pAttachments = lightingAttachments.data();
    lightingRenderPassInfo.subpassCount = 1;
    lightingRenderPassInfo.pSubpasses = &lightingSubpass;

    renderPass = context.device.createRenderPass(lightingRenderPassInfo);
}

void LightingPipeline::createPipelineLayouts(const Context& context)
{
    std::array<vk::DescriptorSetLayout, 1> lightingDescriptorSetLayouts =
    {
        DescriptorLayouts::getFullScreenQuadLayout()
    };

    vk::PipelineLayoutCreateInfo lightingPipelineLayoutInfo{};
    lightingPipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(lightingDescriptorSetLayouts.size());
    lightingPipelineLayoutInfo.pSetLayouts = lightingDescriptorSetLayouts.data();
    lightingPipelineLayoutInfo.pushConstantRangeCount = 0;

    pipelineLayout = context.device.createPipelineLayout(lightingPipelineLayoutInfo);
}

void LightingPipeline::createPipeline(const Context& context)
{
    std::vector<char> lightingVertShaderCode = readFile("assets/shaders/lighting_vert.spv");
    std::vector<char> lightingFragShaderCode = readFile("assets/shaders/lighting_frag.spv");

    vk::ShaderModule lightingVertShaderModule = gpu::Shaders::createShaderModule(context, lightingVertShaderCode);
    vk::ShaderModule lightingFragShaderModule = gpu::Shaders::createShaderModule(context, lightingFragShaderCode);

    vk::PipelineShaderStageCreateInfo lightingVertShaderStageInfo{};
    lightingVertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
    lightingVertShaderStageInfo.module = lightingVertShaderModule;
    lightingVertShaderStageInfo.pName = "main";

    vk::PipelineShaderStageCreateInfo lightingFragShaderStageInfo{};
    lightingFragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
    lightingFragShaderStageInfo.module = lightingFragShaderModule;
    lightingFragShaderStageInfo.pName = "main";

    vk::PipelineShaderStageCreateInfo lightingShaderStages[] = { lightingVertShaderStageInfo, lightingFragShaderStageInfo };

    vk::PipelineDepthStencilStateCreateInfo depthStencilLighting{};
    depthStencilLighting.depthTestEnable = VK_FALSE;
    depthStencilLighting.depthWriteEnable = VK_FALSE;
    depthStencilLighting.depthCompareOp = vk::CompareOp::eLess;
    depthStencilLighting.depthBoundsTestEnable = VK_FALSE;
    depthStencilLighting.stencilTestEnable = VK_FALSE;

    vk::PipelineColorBlendAttachmentState lightingBlendAttachment{};
    lightingBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    lightingBlendAttachment.blendEnable = VK_FALSE;

    vk::PipelineColorBlendAttachmentState lightingBlendAttachments[]{ lightingBlendAttachment };

    vk::PipelineColorBlendStateCreateInfo lightingPassColorBlending{};
    lightingPassColorBlending.logicOpEnable = VK_FALSE;
    lightingPassColorBlending.attachmentCount = 1;
    lightingPassColorBlending.pAttachments = lightingBlendAttachments;

    // Define vertex input state (vertex binding and attributes)
    vk::VertexInputBindingDescription bindingDescription = Vertex::getBindingDescription();
    std::array<vk::VertexInputAttributeDescription, 5> attributeDescriptions = Vertex::getAttributeDescriptions();

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    // Input assembly state
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // Multisampling state (no anti-aliasing for now)
    vk::PipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;

    vk::PipelineViewportStateCreateInfo viewportState{};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    // Dynamic state (viewport and scissor)
    std::vector<vk::DynamicState> dynamicStates =
    {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor
    };

    // Rasterizer state
    vk::PipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = vk::PolygonMode::eFill;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = vk::CullModeFlagBits::eBack;
    rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizer.depthBiasEnable = VK_FALSE;

    vk::PipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    // Set up the lighting pass pipeline
    vk::GraphicsPipelineCreateInfo lightingPipelineInfo{};
    lightingPipelineInfo.stageCount = 2;
    lightingPipelineInfo.pStages = lightingShaderStages;
    lightingPipelineInfo.pVertexInputState = &vertexInputInfo;
    lightingPipelineInfo.pInputAssemblyState = &inputAssembly;
    lightingPipelineInfo.pViewportState = &viewportState;
    lightingPipelineInfo.pRasterizationState = &rasterizer;
    lightingPipelineInfo.pMultisampleState = &multisampling;
    lightingPipelineInfo.pColorBlendState = &lightingPassColorBlending;
    lightingPipelineInfo.pDynamicState = &dynamicState;
    lightingPipelineInfo.layout = pipelineLayout;
    lightingPipelineInfo.renderPass = renderPass;
    lightingPipelineInfo.subpass = 0;
    lightingPipelineInfo.pDepthStencilState = &depthStencilLighting;

    pipeline = context.device.createGraphicsPipeline(nullptr, lightingPipelineInfo).value;

    context.device.destroyShaderModule(lightingVertShaderModule);
    context.device.destroyShaderModule(lightingFragShaderModule);
}

void LightingPipeline::handleResize(const Context& context, CommandBuffers& commandBufferManager)
{
    // TODO: this is not necessary because of dynamic state ?
    context.device.destroyPipeline(pipeline);
    createPipeline(context);
}

void LightingPipeline::reloadShaders(const Context& context)
{
    context.device.destroyPipeline(pipeline);
    createPipeline(context);
}

vk::RenderPass LightingPipeline::getRenderPass() const
{
    return renderPass;
}

vk::Pipeline LightingPipeline::getPipeline() const
{
    return pipeline;
}

vk::PipelineLayout LightingPipeline::getPipelineLayout() const
{
    return pipelineLayout;
}

void LightingPipeline::cleanup(vk::Device device)
{
    device.destroyPipeline(pipeline);
    device.destroyPipelineLayout(pipelineLayout);
    device.destroyRenderPass(renderPass);
}

void LightingPipeline::recordDrawCommands(int width, int height, const Swapchain& swapChainManager, const FullScreenQuad& fullScreenQuad, vk::CommandBuffer commandBuffer, uint32_t currentFrame, uint32_t imageIndex)
{
    vk::RenderPassBeginInfo lightingRenderPassInfo{};
    lightingRenderPassInfo.renderPass = renderPass;
    lightingRenderPassInfo.framebuffer = swapChainManager.swapChainFramebuffers[imageIndex];
    lightingRenderPassInfo.renderArea.offset = vk::Offset2D{ 0, 0 };
    lightingRenderPassInfo.renderArea.extent = swapChainManager.extent;

    std::array<vk::ClearValue, 1> lightingClearValues{};
    lightingClearValues[0].color = vk::ClearColorValue(std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f });

    lightingRenderPassInfo.clearValueCount = static_cast<uint32_t>(lightingClearValues.size());
    lightingRenderPassInfo.pClearValues = lightingClearValues.data();

    commandBuffer.beginRenderPass(lightingRenderPassInfo, vk::SubpassContents::eInline);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

    vk::Extent2D extent;
    extent.width = width;
    extent.height = height;

    vk::Viewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = width;
    viewport.height = height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    commandBuffer.setViewport(0, viewport);

    vk::Rect2D scissor{};
    scissor.offset = vk::Offset2D{ 0, 0 };
    scissor.extent = extent;
    commandBuffer.setScissor(0, scissor);

    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, fullScreenQuad.descriptorSets[currentFrame], {});

    vk::Buffer vertexBuffers[] = { fullScreenQuad.vertexBuffer };
    vk::DeviceSize offsets[] = { 0 };
    commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);

    commandBuffer.draw(6, 1, 0, 0);

    commandBuffer.endRenderPass();
}

} // namespace render
} // namespace vkrt
