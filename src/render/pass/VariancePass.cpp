#include "render/pass/VariancePass.hpp"
#include "gpu/Utils.hpp"
#include <stdexcept>
#include <iostream>


namespace vkrt {
namespace render {
using namespace vkrt::gpu;

std::string VariancePass::getShaderPath() const
{
    return "assets/shaders/variance_compute.spv";
}

std::vector<vk::DescriptorSetLayoutBinding> VariancePass::getDescriptorBindings() const
{
    // 0: color (input)
    // 1: variance (output)
    // 2: normals
    // 3: albedo
    // 4: depth
    std::vector<vk::DescriptorSetLayoutBinding> bindings(5);

    for (uint32_t i = 0; i < bindings.size(); i++)
    {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = vk::ShaderStageFlagBits::eCompute;
        bindings[i].pImmutableSamplers = nullptr;
        bindings[i].descriptorType = (i == 1)
            ? vk::DescriptorType::eStorageImage
            : vk::DescriptorType::eSampledImage;
    }

    return bindings;
}

std::vector<vk::DescriptorPoolSize> VariancePass::getDescriptorPoolSizes() const
{
    std::vector<vk::DescriptorPoolSize> poolSizes(2);
    poolSizes[0].type = vk::DescriptorType::eSampledImage;
    poolSizes[0].descriptorCount = 4; // color + normals + albedo + depth
    poolSizes[1].type = vk::DescriptorType::eStorageImage;
    poolSizes[1].descriptorCount = 1; // output
    return poolSizes;
}

void VariancePass::init(const Context& context, uint32_t width, uint32_t height)
{
    this->width = width;
    this->height = height;

    // Create variance output image
    gpu::Image::createImage(
        context,
        width,
        height,
        vk::Format::eR32Sfloat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        varianceImage,
        varianceImageMemory
    );

    varianceImageView = gpu::Image::createImageView(
        context,
        varianceImage,
        vk::Format::eR32Sfloat,
        vk::ImageAspectFlagBits::eColor
    );

    varianceSampler = gpu::Textures::createSampler(
        context,
        vk::Filter::eLinear,
        vk::Filter::eLinear,
        vk::SamplerMipmapMode::eLinear
    );

    // Create staging buffer
    size_t bufferSize = width * height * sizeof(float);
    gpu::Buffers::createBuffer(
        context,
        bufferSize,
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer,
        stagingBufferMemory,
        false
    );

    ComputePass::init(context);
}

void VariancePass::handleResize(const Context& context, uint32_t newWidth, uint32_t newHeight)
{
    context.device.waitIdle();

    if (varianceImage)
    {
        context.device.destroyImage(varianceImage);
        varianceImage = nullptr;
    }
    if (varianceImageMemory)
    {
        context.device.freeMemory(varianceImageMemory);
        varianceImageMemory = nullptr;
    }
    if (varianceImageView)
    {
        context.device.destroyImageView(varianceImageView);
        varianceImageView = nullptr;
    }
    if (stagingBuffer)
    {
        context.device.destroyBuffer(stagingBuffer);
        stagingBuffer = nullptr;
    }
    if (stagingBufferMemory)
    {
        context.device.freeMemory(stagingBufferMemory);
        stagingBufferMemory = nullptr;
    }

    width = newWidth;
    height = newHeight;
    firstFrame = true;

    gpu::Image::createImage(
        context,
        width,
        height,
        vk::Format::eR32Sfloat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        varianceImage,
        varianceImageMemory
    );

    varianceImageView = gpu::Image::createImageView(
        context,
        varianceImage,
        vk::Format::eR32Sfloat,
        vk::ImageAspectFlagBits::eColor
    );

    size_t bufferSize = width * height * sizeof(float);
    gpu::Buffers::createBuffer(
        context,
        bufferSize,
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer,
        stagingBufferMemory,
        false
    );
}

void VariancePass::compute(vk::CommandBuffer commandBuffer, vk::ImageView colorImageView, vk::Image inputImage)
{
    // Transition input image to SHADER_READ_ONLY_OPTIMAL for variance compute
    vk::ImageMemoryBarrier inputBarrier{};
    inputBarrier.oldLayout = vk::ImageLayout::eGeneral;
    inputBarrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    inputBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    inputBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    inputBarrier.image = inputImage;
    inputBarrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
    inputBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
    inputBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eRayTracingShaderKHR,
        vk::PipelineStageFlagBits::eComputeShader,
        {}, {}, {}, inputBarrier
    );

    // Transition variance image to GENERAL for writing
    vk::ImageLayout varCurrentLayout = firstFrame ? vk::ImageLayout::eUndefined : vk::ImageLayout::eShaderReadOnlyOptimal;
    vk::ImageMemoryBarrier varBarrier{};
    varBarrier.oldLayout = varCurrentLayout;
    varBarrier.newLayout = vk::ImageLayout::eGeneral;
    varBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    varBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    varBarrier.image = varianceImage;
    varBarrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
    varBarrier.srcAccessMask = firstFrame ? vk::AccessFlags{} : vk::AccessFlags{ vk::AccessFlagBits::eShaderRead };
    varBarrier.dstAccessMask = vk::AccessFlagBits::eShaderWrite;

    vk::PipelineStageFlags srcStage = firstFrame ? vk::PipelineStageFlagBits::eTopOfPipe : vk::PipelineStageFlagBits::eFragmentShader;

    commandBuffer.pipelineBarrier(srcStage, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, varBarrier);

    // Dispatch compute shader
    dispatch(commandBuffer, width, height);

    // Transition variance to SHADER_READ_ONLY_OPTIMAL for sampling
    varBarrier.oldLayout = vk::ImageLayout::eGeneral;
    varBarrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    varBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
    varBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, varBarrier);

    // Transition input image back to GENERAL
    inputBarrier.oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    inputBarrier.newLayout = vk::ImageLayout::eGeneral;
    inputBarrier.srcAccessMask = vk::AccessFlagBits::eShaderRead;
    inputBarrier.dstAccessMask = vk::AccessFlagBits::eShaderWrite;

    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eRayTracingShaderKHR,
        {}, {}, {}, inputBarrier
    );

    firstFrame = false;
}

void VariancePass::updateDescriptors(const Context& context, vk::ImageView colorImageView,
    vk::ImageView normalImageView, vk::ImageView albedoImageView, vk::ImageView depthImageView)
{
    vk::DescriptorImageInfo colorInfo{};
    colorInfo.imageView = colorImageView;
    colorInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    vk::DescriptorImageInfo outputInfo{};
    outputInfo.imageView = varianceImageView;
    outputInfo.imageLayout = vk::ImageLayout::eGeneral;

    vk::DescriptorImageInfo normalInfo{};
    normalInfo.imageView = normalImageView;
    normalInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    vk::DescriptorImageInfo albedoInfo{};
    albedoInfo.imageView = albedoImageView;
    albedoInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    vk::DescriptorImageInfo depthInfo{};
    depthInfo.imageView = depthImageView;
    depthInfo.imageLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;

    vk::WriteDescriptorSet writes[5]{};

    writes[0].dstSet = descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = vk::DescriptorType::eSampledImage;
    writes[0].pImageInfo = &colorInfo;

    writes[1].dstSet = descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = vk::DescriptorType::eStorageImage;
    writes[1].pImageInfo = &outputInfo;

    writes[2].dstSet = descriptorSet;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = vk::DescriptorType::eSampledImage;
    writes[2].pImageInfo = &normalInfo;

    writes[3].dstSet = descriptorSet;
    writes[3].dstBinding = 3;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = vk::DescriptorType::eSampledImage;
    writes[3].pImageInfo = &albedoInfo;

    writes[4].dstSet = descriptorSet;
    writes[4].dstBinding = 4;
    writes[4].descriptorCount = 1;
    writes[4].descriptorType = vk::DescriptorType::eSampledImage;
    writes[4].pImageInfo = &depthInfo;

    context.device.updateDescriptorSets(writes, {});
}

float VariancePass::sumVarianceTexture(const Context& context, CommandBuffers& commandBufferManager)
{
    vk::CommandBuffer cmdBuffer = commandBufferManager.beginSingleTimeCommands(context.device);

    // Transition to TRANSFER_SRC_OPTIMAL
    vk::ImageMemoryBarrier barrier{};
    barrier.oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = varianceImage;
    barrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
    barrier.srcAccessMask = vk::AccessFlagBits::eShaderRead;
    barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

    cmdBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, barrier);

    // Copy image to staging buffer
    vk::BufferImageCopy region{};
    region.bufferOffset = 0;
    region.imageSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 };
    region.imageExtent = vk::Extent3D{ width, height, 1 };

    cmdBuffer.copyImageToBuffer(varianceImage, vk::ImageLayout::eTransferSrcOptimal, stagingBuffer, region);

    // Transition back to SHADER_READ_ONLY_OPTIMAL
    barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    cmdBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

    commandBufferManager.endSingleTimeCommands(context.device, context.graphicsQueue, cmdBuffer);

    // Map and sum values
    float* data = static_cast<float*>(context.device.mapMemory(stagingBufferMemory, 0, width * height * sizeof(float)));

    float sum = 0.0f;
    for (uint32_t i = 0; i < width * height; ++i)
    {
		float value = data[i];
        sum += value * value;
    }

    context.device.unmapMemory(stagingBufferMemory);

    return sum;
}

void VariancePass::cleanup(const Context& context)
{
    if (varianceImage)
    {
        context.device.destroyImage(varianceImage);
        varianceImage = nullptr;
    }
    if (varianceImageMemory)
    {
        context.device.freeMemory(varianceImageMemory);
        varianceImageMemory = nullptr;
    }
    if (varianceImageView)
    {
        context.device.destroyImageView(varianceImageView);
        varianceImageView = nullptr;
    }
    if (varianceSampler)
    {
        context.device.destroySampler(varianceSampler);
        varianceSampler = nullptr;
    }
    if (stagingBuffer)
    {
        context.device.destroyBuffer(stagingBuffer);
        stagingBuffer = nullptr;
    }
    if (stagingBufferMemory)
    {
        context.device.freeMemory(stagingBufferMemory);
        stagingBufferMemory = nullptr;
    }

    ComputePass::cleanup(context);
}

} // namespace render
} // namespace vkrt
