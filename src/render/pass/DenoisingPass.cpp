#include "render/pass/DenoisingPass.hpp"
#include "gpu/Utils.hpp"
#include <stdexcept>
#include <iostream>


namespace vkrt {
namespace render {
using namespace vkrt::gpu;

std::string DenoisingPass::getShaderPath() const
{
    return "assets/shaders/denoise_compute.spv";
}

std::vector<vk::DescriptorSetLayoutBinding> DenoisingPass::getDescriptorBindings() const
{
    std::vector<vk::DescriptorSetLayoutBinding> bindings(6);

    // Binding 0: Input color
    bindings[0].binding = 0;
    bindings[0].descriptorType = vk::DescriptorType::eSampledImage;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = vk::ShaderStageFlagBits::eCompute;
    bindings[0].pImmutableSamplers = nullptr;

    // Binding 1: Previous color
    bindings[1].binding = 1;
    bindings[1].descriptorType = vk::DescriptorType::eSampledImage;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = vk::ShaderStageFlagBits::eCompute;
    bindings[1].pImmutableSamplers = nullptr;

    // Binding 2: Normals
    bindings[2].binding = 2;
    bindings[2].descriptorType = vk::DescriptorType::eSampledImage;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = vk::ShaderStageFlagBits::eCompute;
    bindings[2].pImmutableSamplers = nullptr;

    // Binding 3: Albedo
    bindings[3].binding = 3;
    bindings[3].descriptorType = vk::DescriptorType::eSampledImage;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = vk::ShaderStageFlagBits::eCompute;
    bindings[3].pImmutableSamplers = nullptr;

    // Binding 4: Depth
    bindings[4].binding = 4;
    bindings[4].descriptorType = vk::DescriptorType::eSampledImage;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = vk::ShaderStageFlagBits::eCompute;
    bindings[4].pImmutableSamplers = nullptr;

    // Binding 5: Output denoised image
    bindings[5].binding = 5;
    bindings[5].descriptorType = vk::DescriptorType::eStorageImage;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = vk::ShaderStageFlagBits::eCompute;
    bindings[5].pImmutableSamplers = nullptr;

    return bindings;
}

std::vector<vk::DescriptorPoolSize> DenoisingPass::getDescriptorPoolSizes() const
{
    std::vector<vk::DescriptorPoolSize> poolSizes(2);
    poolSizes[0].type = vk::DescriptorType::eSampledImage;
    poolSizes[0].descriptorCount = 5;
    poolSizes[1].type = vk::DescriptorType::eStorageImage;
    poolSizes[1].descriptorCount = 1;
    return poolSizes;
}

vk::PushConstantRange* DenoisingPass::getPushConstantRange() const
{
    static vk::PushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eCompute;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(float);  // 4 bytes for deltaTime
    return &pushConstantRange;
}

void DenoisingPass::init(const Context& context, uint32_t width, uint32_t height)
{
    this->width = width;
    this->height = height;
    this->isFirstFrame = true;
    gpu::Image::createImage(
        context,
        width,
        height,
        vk::Format::eR32G32B32A32Sfloat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        denoisedImage,
        denoisedImageMemory
    );

    denoisedImageView = gpu::Image::createImageView(
        context,
        denoisedImage,
        vk::Format::eR32G32B32A32Sfloat,
        vk::ImageAspectFlagBits::eColor
    );

    // Create previous denoised image for temporal accumulation
    gpu::Image::createImage(
        context,
        width,
        height,
        vk::Format::eR32G32B32A32Sfloat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        denoisedImagePrevious,
        denoisedImagePreviousMemory
    );

    denoisedImagePreviousView = gpu::Image::createImageView(
        context,
        denoisedImagePrevious,
        vk::Format::eR32G32B32A32Sfloat,
        vk::ImageAspectFlagBits::eColor
    );

    // Create sampler
    denoisedSampler = gpu::Textures::createSampler(
        context,
        vk::Filter::eLinear,
        vk::Filter::eLinear,
        vk::SamplerMipmapMode::eNearest
    );

    ComputePass::init(context);
}

void DenoisingPass::handleResize(const Context& context, uint32_t newWidth, uint32_t newHeight)
{
    width = newWidth;
    height = newHeight;
    isFirstFrame = true;

    if (denoisedImageView)
    {
        context.device.destroyImageView(denoisedImageView);
        denoisedImageView = nullptr;
    }
    if (denoisedImage)
    {
        context.device.destroyImage(denoisedImage);
        denoisedImage = nullptr;
    }
    if (denoisedImageMemory)
    {
        context.device.freeMemory(denoisedImageMemory);
        denoisedImageMemory = nullptr;
    }
    if (denoisedImagePreviousView)
    {
        context.device.destroyImageView(denoisedImagePreviousView);
        denoisedImagePreviousView = nullptr;
    }
    if (denoisedImagePrevious)
    {
        context.device.destroyImage(denoisedImagePrevious);
        denoisedImagePrevious = nullptr;
    }
    if (denoisedImagePreviousMemory)
    {
        context.device.freeMemory(denoisedImagePreviousMemory);
        denoisedImagePreviousMemory = nullptr;
    }

    gpu::Image::createImage(
        context,
        width,
        height,
        vk::Format::eR32G32B32A32Sfloat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        denoisedImage,
        denoisedImageMemory
    );

    denoisedImageView = gpu::Image::createImageView(
        context,
        denoisedImage,
        vk::Format::eR32G32B32A32Sfloat,
        vk::ImageAspectFlagBits::eColor
    );

    gpu::Image::createImage(
        context,
        width,
        height,
        vk::Format::eR32G32B32A32Sfloat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        denoisedImagePrevious,
        denoisedImagePreviousMemory
    );

    denoisedImagePreviousView = gpu::Image::createImageView(
        context,
        denoisedImagePrevious,
        vk::Format::eR32G32B32A32Sfloat,
        vk::ImageAspectFlagBits::eColor
    );
}

void DenoisingPass::updateDescriptors(const Context& context, vk::ImageView colorImageView, vk::ImageView previousColorImageView,
    vk::ImageView normalImageView, vk::ImageView albedoImageView, vk::ImageView depthImageView)
{
    vk::DescriptorImageInfo colorInfo{};
    colorInfo.imageView = colorImageView;
    colorInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    vk::DescriptorImageInfo previousColorInfo{};
    previousColorInfo.imageView = previousColorImageView;
    previousColorInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    vk::DescriptorImageInfo normalInfo{};
    normalInfo.imageView = normalImageView;
    normalInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    vk::DescriptorImageInfo albedoInfo{};
    albedoInfo.imageView = albedoImageView;
    albedoInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    vk::DescriptorImageInfo depthInfo{};
    depthInfo.imageView = depthImageView;
    depthInfo.imageLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;

    vk::DescriptorImageInfo outputInfo{};
    outputInfo.imageView = denoisedImageView;
    outputInfo.imageLayout = vk::ImageLayout::eGeneral;

    std::vector<vk::WriteDescriptorSet> writes(6);

    writes[0].dstSet = descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = vk::DescriptorType::eSampledImage;
    writes[0].pImageInfo = &colorInfo;

    writes[1].dstSet = descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = vk::DescriptorType::eSampledImage;
    writes[1].pImageInfo = &previousColorInfo;

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

    writes[5].dstSet = descriptorSet;
    writes[5].dstBinding = 5;
    writes[5].descriptorCount = 1;
    writes[5].descriptorType = vk::DescriptorType::eStorageImage;
    writes[5].pImageInfo = &outputInfo;

    context.device.updateDescriptorSets(writes, {});
}

void DenoisingPass::denoise(vk::CommandBuffer commandBuffer, float deltaTime)
{
    // On first frame, initialize both images to their starting layouts
    if (isFirstFrame)
    {
        // Initialize previous image to shader read layout
        vk::ImageMemoryBarrier initPrevBarrier{};
        initPrevBarrier.oldLayout = vk::ImageLayout::eUndefined;
        initPrevBarrier.newLayout = vk::ImageLayout::eTransferDstOptimal;
        initPrevBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        initPrevBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        initPrevBarrier.image = denoisedImagePrevious;
        initPrevBarrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
        initPrevBarrier.srcAccessMask = {};
        initPrevBarrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

        // Initialize output image to general layout for compute
        vk::ImageMemoryBarrier initOutBarrier{};
        initOutBarrier.oldLayout = vk::ImageLayout::eUndefined;
        initOutBarrier.newLayout = vk::ImageLayout::eGeneral;
        initOutBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        initOutBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        initOutBarrier.image = denoisedImage;
        initOutBarrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
        initOutBarrier.srcAccessMask = {};
        initOutBarrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

        std::vector<vk::ImageMemoryBarrier> initBarriers{ initPrevBarrier, initOutBarrier };
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
            {}, {}, {}, initBarriers);

        // Clear both images to black/zero
        vk::ClearColorValue clearColor(std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f });
        vk::ImageSubresourceRange range{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };

        commandBuffer.clearColorImage(denoisedImagePrevious, vk::ImageLayout::eTransferDstOptimal, clearColor, range);
        commandBuffer.clearColorImage(denoisedImage, vk::ImageLayout::eGeneral, clearColor, range);

        // Barrier to synchronize clear operations before next transitions
        vk::ImageMemoryBarrier clearBarrier{};
        clearBarrier.oldLayout = vk::ImageLayout::eGeneral;
        clearBarrier.newLayout = vk::ImageLayout::eGeneral;
        clearBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        clearBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        clearBarrier.image = denoisedImage;
        clearBarrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
        clearBarrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        clearBarrier.dstAccessMask = vk::AccessFlagBits::eShaderWrite;

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
            {}, {}, {}, clearBarrier);

        // Transition previous to shader read
        vk::ImageMemoryBarrier transitionBarrier{};
        transitionBarrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        transitionBarrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        transitionBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        transitionBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        transitionBarrier.image = denoisedImagePrevious;
        transitionBarrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
        transitionBarrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        transitionBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
            {}, {}, {}, transitionBarrier);

        isFirstFrame = false;
    }
    else
    {
        // On subsequent frames, transition denoised image from TRANSFER_SRC back to GENERAL for compute
        vk::ImageMemoryBarrier outputBarrier{};
        outputBarrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
        outputBarrier.newLayout = vk::ImageLayout::eGeneral;
        outputBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        outputBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        outputBarrier.image = denoisedImage;
        outputBarrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
        outputBarrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
        outputBarrier.dstAccessMask = vk::AccessFlagBits::eShaderWrite;

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
            {}, {}, {}, outputBarrier);
    }

    // Push constant with deltaTime
    commandBuffer.pushConstants<float>(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, deltaTime);

    // Dispatch compute shader
    // NOTE: denoisedImagePrevious remains in SHADER_READ_ONLY_OPTIMAL as expected by descriptor
    dispatch(commandBuffer, width, height);

    // After compute, transition denoised image to TRANSFER_SRC for copy
    vk::ImageMemoryBarrier outputBarrier{};
    outputBarrier.oldLayout = vk::ImageLayout::eGeneral;
    outputBarrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
    outputBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputBarrier.image = denoisedImage;
    outputBarrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
    outputBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
    outputBarrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
        {}, {}, {}, outputBarrier);

    // NOW transition previous image to transfer destination for copy
    vk::ImageMemoryBarrier prevBarrier{};
    prevBarrier.oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    prevBarrier.newLayout = vk::ImageLayout::eTransferDstOptimal;
    prevBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    prevBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    prevBarrier.image = denoisedImagePrevious;
    prevBarrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
    prevBarrier.srcAccessMask = vk::AccessFlagBits::eShaderRead;
    prevBarrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
        {}, {}, {}, prevBarrier);

    // Copy current denoised to previous
    vk::ImageCopy region{};
    region.srcSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 };
    region.srcOffset = vk::Offset3D{ 0, 0, 0 };
    region.dstSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 };
    region.dstOffset = vk::Offset3D{ 0, 0, 0 };
    region.extent = vk::Extent3D{ width, height, 1 };

    commandBuffer.copyImage(denoisedImage, vk::ImageLayout::eTransferSrcOptimal,
                   denoisedImagePrevious, vk::ImageLayout::eTransferDstOptimal, region);

    // Transition previous back to shader read for next frame
    prevBarrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    prevBarrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    prevBarrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    prevBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
        {}, {}, {}, prevBarrier);
}

void DenoisingPass::cleanup(const Context& context)
{
    if (denoisedSampler)
    {
        context.device.destroySampler(denoisedSampler);
        denoisedSampler = nullptr;
    }
    if (denoisedImageView)
    {
        context.device.destroyImageView(denoisedImageView);
        denoisedImageView = nullptr;
    }
    if (denoisedImage)
    {
        context.device.destroyImage(denoisedImage);
        denoisedImage = nullptr;
    }
    if (denoisedImageMemory)
    {
        context.device.freeMemory(denoisedImageMemory);
        denoisedImageMemory = nullptr;
    }
    if (denoisedImagePreviousView)
    {
        context.device.destroyImageView(denoisedImagePreviousView);
        denoisedImagePreviousView = nullptr;
    }
    if (denoisedImagePrevious)
    {
        context.device.destroyImage(denoisedImagePrevious);
        denoisedImagePrevious = nullptr;
    }
    if (denoisedImagePreviousMemory)
    {
        context.device.freeMemory(denoisedImagePreviousMemory);
        denoisedImagePreviousMemory = nullptr;
    }

    ComputePass::cleanup(context);
}

} // namespace render
} // namespace vkrt
