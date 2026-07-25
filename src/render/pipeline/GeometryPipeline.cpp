#include "render/pipeline/GeometryPipeline.hpp"
#include <stdexcept>
#include "core/Utils.hpp"
#include "gpu/DescriptorLayouts.hpp"
#include "core/Settings.hpp"
#include <iostream>


namespace vkrt {
namespace render {
using namespace vkrt::core;
using namespace vkrt::gpu;
using namespace vkrt::scene;

void GeometryPipeline::init(const Context& context, CommandBuffers& commandBufferManager, int width, int height, const GBuffer& gBufferManager)
{
    createRenderPasses(context);
    createPipelineLayouts(context);
    createPipeline(context);
    createFramebuffers(context, gBufferManager, width, height);
}

void GeometryPipeline::createRenderPasses(const Context& context)
{
    vk::AttachmentDescription depthAttachment{};
    depthAttachment.format = gpu::DepthStencil::findDepthFormat(context.physicalDevice);
    depthAttachment.samples = vk::SampleCountFlagBits::e1;
    depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    depthAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    depthAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    depthAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    depthAttachment.initialLayout = vk::ImageLayout::eUndefined;
    depthAttachment.finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal; // TODO: Figure out why implicit transition to DepthStencilReadOnlyOptimal does not seem to work

    vk::AttachmentDescription normalAttachment{};
    normalAttachment.format = NORMAL_MAP_FORMAT;
    normalAttachment.samples = vk::SampleCountFlagBits::e1;
    normalAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    normalAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    normalAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    normalAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    normalAttachment.initialLayout = vk::ImageLayout::eUndefined;
    normalAttachment.finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    vk::AttachmentDescription albedoAttachment{};
    albedoAttachment.format = ALBEDO_MAP_FORMAT;
    albedoAttachment.samples = vk::SampleCountFlagBits::e1;
    albedoAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    albedoAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    albedoAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    albedoAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    albedoAttachment.initialLayout = vk::ImageLayout::eUndefined;
    albedoAttachment.finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    vk::AttachmentDescription roughnessAttachment{};
    roughnessAttachment.format = ROUGHNESS_MAP_FORMAT;
    roughnessAttachment.samples = vk::SampleCountFlagBits::e1;
    roughnessAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    roughnessAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    roughnessAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    roughnessAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    roughnessAttachment.initialLayout = vk::ImageLayout::eUndefined;
    roughnessAttachment.finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    vk::AttachmentDescription metalnessAttachment{};
    metalnessAttachment.format = METALNESS_MAP_FORMAT;
    metalnessAttachment.samples = vk::SampleCountFlagBits::e1;
    metalnessAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    metalnessAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    metalnessAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    metalnessAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    metalnessAttachment.initialLayout = vk::ImageLayout::eUndefined;
    metalnessAttachment.finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    vk::AttachmentDescription velocityAttachment{};
    velocityAttachment.format = MOTION_VECTOR_FORMAT;
    velocityAttachment.samples = vk::SampleCountFlagBits::e1;
    velocityAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    velocityAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    velocityAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    velocityAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    velocityAttachment.initialLayout = vk::ImageLayout::eUndefined;
    velocityAttachment.finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    vk::AttachmentReference normalAttachmentRef{};
    normalAttachmentRef.attachment = 0;
    normalAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::AttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

    vk::AttachmentReference albedoAttachmentRef{};
    albedoAttachmentRef.attachment = 2;
    albedoAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::AttachmentReference roughnessAttachmentRef{};
    roughnessAttachmentRef.attachment = 3;
    roughnessAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::AttachmentReference metalnessAttachmentRef{};
    metalnessAttachmentRef.attachment = 4;
    metalnessAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::AttachmentReference velocityAttachmentRef{};
    velocityAttachmentRef.attachment = 5;
    velocityAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

    std::array<vk::AttachmentReference, 5> colorAttachmentRefs
    {
        normalAttachmentRef,
        albedoAttachmentRef,
        roughnessAttachmentRef,
        metalnessAttachmentRef,
        velocityAttachmentRef
    };

    vk::SubpassDescription subpass{};
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount = static_cast<uint32_t>(colorAttachmentRefs.size());
    subpass.pColorAttachments = colorAttachmentRefs.data();
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    std::array<vk::AttachmentDescription, 6> attachments =
    {
        normalAttachment,
        depthAttachment,
        albedoAttachment,
        roughnessAttachment,
        metalnessAttachment,
        velocityAttachment
    };

    vk::RenderPassCreateInfo renderPassInfo{};
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;

    renderPass = context.device.createRenderPass(renderPassInfo);
}

void GeometryPipeline::createFramebuffers(const Context& context, const GBuffer& gBufferManager, uint32_t width, uint32_t height)
{
    std::array<vk::ImageView, 6> geometryAttachments =
    {
        gBufferManager.normalImageView,
        gBufferManager.depthImageView,
        gBufferManager.albedoImageView,
        gBufferManager.roughnessImageView,
        gBufferManager.metalnessImageView,
        gBufferManager.velocityImageView
    };

    vk::FramebufferCreateInfo geometryFramebufferInfo{};
    geometryFramebufferInfo.renderPass = renderPass;
    geometryFramebufferInfo.attachmentCount = static_cast<uint32_t>(geometryAttachments.size());
    geometryFramebufferInfo.pAttachments = geometryAttachments.data();
    geometryFramebufferInfo.width = width;
    geometryFramebufferInfo.height = height;
    geometryFramebufferInfo.layers = 1;

    framebuffer = context.device.createFramebuffer(geometryFramebufferInfo);
}

void GeometryPipeline::createPipelineLayouts(const Context& context)
{
    // Geometry Pipeline Layout
    std::array<vk::DescriptorSetLayout, 2> geometryDescriptorSetLayouts =
    {
        DescriptorLayouts::getModelLayout(),
        DescriptorLayouts::getMaterialLayout()
    };

    vk::PipelineLayoutCreateInfo geometryPipelineLayoutInfo{};
    geometryPipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(geometryDescriptorSetLayouts.size());
    geometryPipelineLayoutInfo.pSetLayouts = geometryDescriptorSetLayouts.data();
    geometryPipelineLayoutInfo.pushConstantRangeCount = 0;

    pipelineLayout = context.device.createPipelineLayout(geometryPipelineLayoutInfo);
}

void GeometryPipeline::createPipeline(const Context& context)
{
    std::vector<char> vertShaderCode = readFile("assets/shaders/geometry_vert.spv");
    std::vector<char> fragShaderCode = readFile("assets/shaders/geometry_frag.spv");

    vk::ShaderModule vertShaderModule = gpu::Shaders::createShaderModule(context, vertShaderCode);
    vk::ShaderModule fragShaderModule = gpu::Shaders::createShaderModule(context, fragShaderCode);

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    // Define vertex input state
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

    vk::PipelineViewportStateCreateInfo viewportState{};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    // Rasterizer state
    vk::PipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = vk::PolygonMode::eFill;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = vk::CullModeFlagBits::eBack;
    rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizer.depthBiasEnable = VK_FALSE;

    // Multisampling state (no anti-aliasing for now)
    vk::PipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;

    // Depth and Stencil state for geometry pass (writes depth and normal)
    vk::PipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = vk::CompareOp::eLess;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    // Color blend state for geometry pass
    vk::PipelineColorBlendAttachmentState normalBlendAttachment{};
    normalBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    normalBlendAttachment.blendEnable = VK_FALSE;

    vk::PipelineColorBlendAttachmentState albedoBlendAttachment{};
    albedoBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    albedoBlendAttachment.blendEnable = VK_FALSE;

    vk::PipelineColorBlendAttachmentState roughnessBlendAttachment{};
    roughnessBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR;
    roughnessBlendAttachment.blendEnable = VK_FALSE;

    vk::PipelineColorBlendAttachmentState metalnessBlendAttachment{};
    metalnessBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR;
    metalnessBlendAttachment.blendEnable = VK_FALSE;

    vk::PipelineColorBlendAttachmentState velocityBlendAttachment{};
    velocityBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG;
    velocityBlendAttachment.blendEnable = VK_FALSE;

    std::array<vk::PipelineColorBlendAttachmentState, 5> geometryBlendAttachments
    {
        normalBlendAttachment,
        albedoBlendAttachment,
        roughnessBlendAttachment,
        metalnessBlendAttachment,
        velocityBlendAttachment
    };

    vk::PipelineColorBlendStateCreateInfo geometryPassColorBlending{};
    geometryPassColorBlending.logicOpEnable = VK_FALSE;
    geometryPassColorBlending.attachmentCount = static_cast<uint32_t>(geometryBlendAttachments.size());
    geometryPassColorBlending.pAttachments = geometryBlendAttachments.data();

    // Dynamic state (viewport and scissor)
    std::vector<vk::DynamicState> dynamicStates =
    {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor
    };

    vk::PipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    // Create geometry pass pipeline
    vk::GraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &geometryPassColorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.pDepthStencilState = &depthStencil;

    pipeline = context.device.createGraphicsPipeline(nullptr, pipelineInfo).value;

    context.device.destroyShaderModule(vertShaderModule);
    context.device.destroyShaderModule(fragShaderModule);
}

void GeometryPipeline::handleResize(const Context& context, CommandBuffers& commandBufferManager, const GBuffer& gBufferManager, int width, int height)
{
    // Pipelines
    // TODO: this is not necessary because of dynamic state ?
    context.device.destroyPipeline(pipeline);
    createPipeline(context);

    // Framebuffers
    context.device.destroyFramebuffer(framebuffer);
    createFramebuffers(context, gBufferManager, width, height);
}

void GeometryPipeline::reloadShaders(const Context& context)
{
    context.device.destroyPipeline(pipeline);
    createPipeline(context);
}

void GeometryPipeline::drawMesh(const ShadedMesh& shadedMesh, vk::CommandBuffer cmdBuffer, uint32_t currentFrame)
{
    const Mesh& mesh = shadedMesh.mesh;
    const Material& material = shadedMesh.material;

    // Bind vertex and index buffers for this mesh
    vk::Buffer vertexBuffers[] = { mesh.vertexBuffer };
    vk::DeviceSize offsets[] = { 0 };
    cmdBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
    cmdBuffer.bindIndexBuffer(mesh.indexBuffer, 0, vk::IndexType::eUint32);

    // Bind material descriptor set for this mesh
    cmdBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        pipelineLayout,
        1,  // Set 1: Material layout
        material.descriptorSet,
        {});

    // Draw this mesh
    cmdBuffer.drawIndexed(
        static_cast<uint32_t>(mesh.indices.size()),
        1, 0, 0, 0);
}

void GeometryPipeline::recordDrawCommands(int width, int height, const std::vector<Model>& models, vk::CommandBuffer commandBuffer, uint32_t currentFrame)
{
    vk::Extent2D extent;
    extent.width = width;
    extent.height = height;

    vk::RenderPassBeginInfo geometryRenderPassInfo{};
    geometryRenderPassInfo.renderPass = renderPass;
    geometryRenderPassInfo.framebuffer = framebuffer;
    geometryRenderPassInfo.renderArea.offset = vk::Offset2D{ 0, 0 };
    geometryRenderPassInfo.renderArea.extent = extent;

    std::array<vk::ClearValue, 6> geometryClearValues{};
    geometryClearValues[0].color = vk::ClearColorValue(std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f });
    geometryClearValues[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };
    geometryClearValues[2].color = vk::ClearColorValue(std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f });
    geometryClearValues[3].color = vk::ClearColorValue(std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f });
    geometryClearValues[4].color = vk::ClearColorValue(std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f });
    geometryClearValues[5].color = vk::ClearColorValue(std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f });

    geometryRenderPassInfo.clearValueCount = static_cast<uint32_t>(geometryClearValues.size());
    geometryRenderPassInfo.pClearValues = geometryClearValues.data();

    commandBuffer.beginRenderPass(geometryRenderPassInfo, vk::SubpassContents::eInline);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

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

    int i = 0;
    for (const Model& model : models)
    {
        // Bind model descriptor set once per model (transform/geometry data)
        commandBuffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            pipelineLayout,
            0,  // Set 0: Model layout
            model.modelDescriptorSets[currentFrame],
            {});

        // Render each submesh
        for (const ShadedMesh& shadedMesh : model.shadedMeshes)
        {
            drawMesh(shadedMesh, commandBuffer, currentFrame);
        }
    }

    commandBuffer.endRenderPass();
}

// TOOD: Legacy, can delete
void GeometryPipeline::drawTBNGizmo(const std::vector<Model>& models, uint32_t currentFrame, vk::CommandBuffer commandBuffer)
{
    // const Model* arrowGizmo = &models.at(0);

    // const Model& model = models.at(1);
    // const ShadedMesh& shadedMesh = model.shadedMeshes.at(0);
    // int currentVertex = Settings::debugIndex1 % shadedMesh.mesh.vertices.size();
    // const Vertex& v = shadedMesh.mesh.vertices[currentVertex];

    // glm::mat4x4 modelMat = model.transform.getTransformMatrix();
    // glm::mat4x4 normalMat = glm::transpose(glm::inverse(modelMat));

    // glm::vec3 normal = glm::normalize(glm::vec3(normalMat * glm::vec4(v.normal, 0.0f)));
    // glm::vec3 tangent = glm::normalize(glm::vec3(normalMat * glm::vec4(v.tangent, 0.0f)));
    // glm::vec3 bitangent = glm::normalize(glm::vec3(normalMat * glm::vec4(v.bitangent, 0.0f)));
    // glm::vec3 pos = glm::vec3(modelMat * glm::vec4(v.pos, 1.0f));

    // glm::vec3 gizmoAxes[3] = { normal, tangent, bitangent };
    // Transform t;
    // t.setTransformMatrix(glm::inverse(glm::lookAt(pos, pos + gizmoAxes[Settings::debugIndex2 % 3], glm::vec3(0, 1, 0))));

    // float scale = 0.1;
    // t.setScale(glm::vec3(scale, scale, scale));

    // glm::mat4 gizmoModelMat = t.getTransformMatrix();
    // glm::mat4 gizmoNormalMat = glm::transpose(glm::inverse(gizmoModelMat));

    // // Move gizmo to the right place
    // uint8_t* base = static_cast<uint8_t*>(arrowGizmo->uniformBuffersMapped[currentFrame]);

    // memcpy(base + offsetof(ModelUBO, modelMat), &gizmoModelMat, sizeof(gizmoModelMat));
    // memcpy(base + offsetof(ModelUBO, normalMat), &gizmoNormalMat, sizeof(gizmoNormalMat));

    // commandBuffer.bindDescriptorSets(
    //     vk::PipelineBindPoint::eGraphics,
    //     pipelineLayout,
    //     0,  // Set 0: Model layout
    //     arrowGizmo->modelDescriptorSets[currentFrame],
    //     {});

    // for (const ShadedMesh& gizmoShadedMesh : arrowGizmo->shadedMeshes)
    // {
    //     drawMesh(gizmoShadedMesh, commandBuffer, currentFrame);
    // }
}

vk::RenderPass GeometryPipeline::getRenderPass() const
{
    return renderPass;
}

vk::Framebuffer GeometryPipeline::getFrameBuffer() const
{
    return framebuffer;
}

vk::Pipeline GeometryPipeline::getPipeline() const
{
    return pipeline;
}

vk::PipelineLayout GeometryPipeline::getPipelineLayout() const
{
    return pipelineLayout;
}

void GeometryPipeline::cleanup(vk::Device device)
{
    device.destroyPipeline(pipeline);
    device.destroyPipelineLayout(pipelineLayout);
    device.destroyRenderPass(renderPass);
    device.destroyFramebuffer(framebuffer);
}

} // namespace render
} // namespace vkrt
