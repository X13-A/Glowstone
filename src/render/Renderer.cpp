#include "render/Renderer.hpp"
#include <iostream>
#include <chrono>
#include <array>
#include "core/Math.hpp"
#include "input/EventManager.hpp"
#include "input/Events.hpp"
#include "core/Time.hpp"
#include "core/Settings.hpp"


namespace vkrt {
namespace render {
using namespace vkrt::core;
using namespace vkrt::gpu;
using namespace vkrt::input;
using namespace vkrt::scene;

void Renderer::init(const Context& context, const Swapchain& swapChainManager)
{
    varianceCompute.init(context, swapChainManager.extent.width, swapChainManager.extent.height);
    denoisingPass.init(context, swapChainManager.extent.width, swapChainManager.extent.height);
    createSyncObjects(context, swapChainManager);
}

void Renderer::reloadShaders(const Context& context)
{
    varianceCompute.reloadShaders(context);
    denoisingPass.reloadShaders(context);
}

void Renderer::createSyncObjects(const Context& context, const Swapchain& swapChainManager)
{
    renderFinishedSemaphores.resize(swapChainManager.swapChainImages.size());
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    vk::SemaphoreCreateInfo semaphoreInfo{};

    for (size_t i = 0; i < swapChainManager.swapChainImages.size(); i++)
    {
        renderFinishedSemaphores[i] = context.device.createSemaphore(semaphoreInfo);
    }

    vk::FenceCreateInfo fenceInfo{};
    fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        inFlightFences[i] = context.device.createFence(fenceInfo);
        imageAvailableSemaphores[i] = context.device.createSemaphore(semaphoreInfo);
    }
}

void Renderer::recordCommandBuffer(int nativeWidth, int nativeHeight, int scaledWidth, int scaledHeight, const Context& context, CommandBuffers& commandBufferManager, const Swapchain swapChainManager, GraphicsPipeline& graphicsPipeline, uint32_t imageIndex, uint32_t currentFrame, const std::vector<Model>& models, const FullScreenQuad& fullScreenQuad)
{
    vk::CommandBufferBeginInfo beginInfo{};

    vk::CommandBuffer commandBuffer = commandBufferManager.commandBuffers[currentFrame];

    commandBuffer.begin(beginInfo);

    graphicsPipeline.geometryPipeline.recordDrawCommands(scaledWidth, scaledHeight, models, commandBuffer, currentFrame);

    gpu::Image::transition_depthRW_to_depthR(context, commandBuffer, graphicsPipeline.gBufferManager.depthImage, vk::Format::eD32Sfloat);

    // Lighting pipeline is only used when RT is disabled
    if (!Settings::displayRayTracing)
    {
        graphicsPipeline.lightingPipeline.recordDrawCommands(nativeWidth, nativeHeight, swapChainManager, fullScreenQuad, commandBuffer, currentFrame, imageIndex);
    }
}

void Renderer::triggerResize(GLFWwindow* window, const Context& context, Swapchain& swapChainManager, GraphicsPipeline& graphicsPipeline, CommandBuffers& commandBufferManager, FullScreenQuad& fullScreenQuad)
{
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    WindowResizeEvent e;
    e.nativeWidth = width;
    e.nativeHeight = height;
    e.scaledWidth = static_cast<int> ((float)width * Settings::renderScale);
    e.scaledHeight = static_cast<int> ((float)height * Settings::renderScale);
    EventManager::get().trigger(e);
}

void Renderer::drawFrame(int nativeWidth, int nativeHeight, int scaledWidth, int scaledHeight, GLFWwindow* window, const Context& context, Swapchain& swapChainManager, GraphicsPipeline& graphicsPipeline, CommandBuffers& commandBufferManager, const Camera& camera, const std::vector<Model>& models, FullScreenQuad& fullScreenQuad)
{
    (void)context.device.waitForFences(inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    // Get next image
    uint32_t imageIndex;
    vk::Result result;
    try
    {
        vk::ResultValue<uint32_t> acquired = context.device.acquireNextImageKHR(swapChainManager.swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], nullptr);
        result = acquired.result;
        imageIndex = acquired.value;
    }
    catch (const vk::OutOfDateKHRError&)
    {
        triggerResize(window, context, swapChainManager, graphicsPipeline, commandBufferManager, fullScreenQuad);
        return;
    }

    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
    {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    // Sync
    context.device.resetFences(inFlightFences[currentFrame]);
    commandBufferManager.commandBuffers[currentFrame].reset();

    // Update uniforms
    updateUniformBuffers(scaledWidth, scaledHeight, camera, models, fullScreenQuad, swapChainManager, graphicsPipeline.rtPipeline, currentFrame);

    recordCommandBuffer(nativeWidth, nativeHeight, scaledWidth, scaledHeight, context, commandBufferManager, swapChainManager, graphicsPipeline, imageIndex, currentFrame, models, fullScreenQuad);
    
    vk::CommandBuffer commandBuffer = commandBufferManager.commandBuffers[currentFrame];

    if (Settings::displayRayTracing)
    {
        // GBuffer -> RT barrier
        vk::MemoryBarrier gbufferToRtBarrier{};
        gbufferToRtBarrier.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
        gbufferToRtBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eLateFragmentTests,
            vk::PipelineStageFlagBits::eRayTracingShaderKHR,
            {},
            gbufferToRtBarrier, {}, {});

        // Trace rays into the per-frame command buffer
        graphicsPipeline.rtPipeline.traceRays(commandBuffer, Time::getFrameCount(), currentFrame);

        // Blit ray traced image to swapchain

        // Save last image for blending
        // CONSIDER: skip useless transition, keep TRANSFER_SRC_BIT
        // CONSIDER: even better, just do a raw copy instead of blit since they have same size, might be faster
        // TODO: even better, ping pong between two textures
        gpu::Image::blitImage(
            commandBuffer,
            graphicsPipeline.rtPipeline.getStorageImage(),
            graphicsPipeline.rtPipeline.getLastStorageImage(),
            RT_STORAGE_IMAGE_FORMAT,
            RT_STORAGE_IMAGE_FORMAT,
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead,
            vk::ImageLayout::eGeneral,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            graphicsPipeline.rtPipeline.getStorageImageWidth(),
            graphicsPipeline.rtPipeline.getStorageImageHeight(),
            graphicsPipeline.rtPipeline.getStorageImageWidth(),
            graphicsPipeline.rtPipeline.getStorageImageHeight(),
            vk::Filter::eNearest
        );

        // Prepare source image based on post-process settings
        vk::Image sourceImage;
        vk::Format sourceFormat;
        vk::AccessFlags srcAccessMask;
        vk::ImageLayout srcLayout;
        uint32_t srcWidth, srcHeight;

        // Denoised
        if (denoisingEnabled)
        {
            vk::ImageMemoryBarrier rtImageBarrier{};
            rtImageBarrier.oldLayout = vk::ImageLayout::eGeneral;
            rtImageBarrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            rtImageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            rtImageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            rtImageBarrier.image = graphicsPipeline.rtPipeline.getStorageImage();
            rtImageBarrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
            rtImageBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
            rtImageBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, rtImageBarrier);
            
            denoisingPass.denoise(commandBuffer, Time::deltaTime());

            sourceImage = denoisingPass.getDenoisedImage();
            sourceFormat = vk::Format::eR32G32B32A32Sfloat;
            srcAccessMask = vk::AccessFlagBits::eTransferRead;
            srcLayout = vk::ImageLayout::eTransferSrcOptimal;
            srcWidth = scaledWidth;
            srcHeight = scaledHeight;
        }
        // Variance
        else if (displayVariance)
        {
            varianceCompute.compute(commandBuffer, graphicsPipeline.rtPipeline.getStorageImageView(), graphicsPipeline.rtPipeline.getStorageImage());

            sourceImage = varianceCompute.getVarianceImage();
            sourceFormat = vk::Format::eR32Sfloat;
            srcAccessMask = vk::AccessFlagBits::eShaderRead;
            srcLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            srcWidth = scaledWidth;
            srcHeight = scaledHeight;

            if (displayVarianceSum)
            {
                float varianceSum = varianceCompute.sumVarianceTexture(context, commandBufferManager);
                std::cout << "Total Variance: " << varianceSum << std::endl;
            }
        }
        // Raw ray-traced image
        else
        {
            sourceImage = graphicsPipeline.rtPipeline.getStorageImage();
            sourceFormat = vk::Format::eR32G32B32A32Sfloat;
            srcAccessMask = vk::AccessFlagBits::eShaderWrite;
            srcLayout = vk::ImageLayout::eGeneral;
            srcWidth = scaledWidth;
            srcHeight = scaledHeight;
        }

        // Prepare swapchain image for blit
        vk::ImageMemoryBarrier swapchainAcquireBarrier{};
        swapchainAcquireBarrier.oldLayout = vk::ImageLayout::eUndefined;
        swapchainAcquireBarrier.newLayout = vk::ImageLayout::ePresentSrcKHR;
        swapchainAcquireBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        swapchainAcquireBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        swapchainAcquireBarrier.image = swapChainManager.swapChainImages[imageIndex];
        swapchainAcquireBarrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };

        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eTransfer,
            {},
            {}, {}, swapchainAcquireBarrier);

        // Blit to swapchain
        gpu::Image::blitImage(
            commandBuffer,
            sourceImage,
            swapChainManager.swapChainImages[imageIndex],
            sourceFormat,
            swapChainManager.swapChainImageFormat,
            srcAccessMask,
            {},
            srcLayout,
            vk::ImageLayout::ePresentSrcKHR,
            srcWidth,
            srcHeight,
            swapChainManager.extent.width,
            swapChainManager.extent.height,
            vk::Filter::eNearest
        );

    }

    // End and submit
    commandBuffer.end();

    vk::SubmitInfo submitInfo{};

    vk::Semaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
    vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eTopOfPipe };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vk::Semaphore signalSemaphores[] = { renderFinishedSemaphores[imageIndex] };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    context.graphicsQueue.submit(submitInfo, inFlightFences[currentFrame]);

    // Present image to swapchain
    vk::PresentInfoKHR presentInfo{};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    vk::SwapchainKHR swapChains[] = { swapChainManager.swapChain };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    try
    {
        result = context.presentQueue.presentKHR(presentInfo);
    }
    catch (const vk::OutOfDateKHRError&)
    {
        result = vk::Result::eErrorOutOfDateKHR;
    }

    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || swapChainManager.framebufferResized)
    {
        swapChainManager.framebufferResized = false;
        triggerResize(window, context, swapChainManager, graphicsPipeline, commandBufferManager, fullScreenQuad);
    }
    else if (result != vk::Result::eSuccess)
    {
        throw std::runtime_error("failed to present swap chain image!");
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void Renderer::updateUniformBuffers(int scaledWidth, int scaledHeight, const Camera& camera, const std::vector<Model>& models, const FullScreenQuad& fullScreenQuad, const Swapchain& swapChain, RayTracingPipeline& rtPipeline, uint32_t currentFrame)
{
    for (int i = 0; i < models.size(); i++)
    {
        ModelUBO ubo{};
        ubo.modelMat = models[i].transform.getTransformMatrix();
        ubo.normalMat = glm::transpose(glm::inverse(ubo.modelMat));

        ubo.viewMat = camera.getViewMatrix();
        ubo.projMat = camera.getProjectionMatrix();

        // TODO: feed previous model matrix so that motion vectors work with animated objects
        ubo.prevModelViewProj = previousViewProj * ubo.modelMat;

        memcpy(models[i].uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));
    }

    FullScreenQuadUBO fullScreenUBO{};
    fullScreenUBO.time = 0; // TODO ?
    memcpy(fullScreenQuad.uniformBuffersMapped[currentFrame], &fullScreenUBO, sizeof(fullScreenUBO));

    SceneData sceneData;
    sceneData.proj = camera.getProjectionMatrix();
    sceneData.view = camera.getViewMatrix();
    sceneData.projInverse = glm::inverse(camera.getProjectionMatrix());
    sceneData.viewInverse = glm::inverse(camera.getViewMatrix());
    sceneData.cameraPos = camera.transform.getPosition();
    sceneData.recursionDepth = std::clamp(Settings::rt_recursion_depth, 0, RT_MAX_RECURSION_DEPTH);
    sceneData.nearFar = glm::vec2(camera.getNearPlane(), camera.getFarPlane());
    sceneData.spp = Settings::spp;
    sceneData.resolutionX = scaledWidth;
    sceneData.resolutionY = scaledHeight;
	sceneData.risCandidates = Settings::risCandidates;
	sceneData.reservoirParity = reservoirFrameIndex % 2; // Ping-pong the reservoir buffers
	reservoirFrameIndex++;

	// Update for next frame's motion vectors
	previousViewProj = sceneData.proj * sceneData.view;

    rtPipeline.updateUniformBuffer(sceneData, currentFrame);
}

void Renderer::cleanup(const Context& context)
{
    varianceCompute.cleanup(context);
    denoisingPass.cleanup(context);
    
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        context.device.destroySemaphore(imageAvailableSemaphores[i]);
        context.device.destroyFence(inFlightFences[i]);
    }

    for (size_t i = 0; i < renderFinishedSemaphores.size(); i++)
    {
        context.device.destroySemaphore(renderFinishedSemaphores[i]);
    }
}

} // namespace render
} // namespace vkrt
