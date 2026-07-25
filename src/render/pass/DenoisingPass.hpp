#pragma once

#include "render/pass/ComputePass.hpp"
#include "gpu/Context.hpp"
#include "gpu/CommandBuffers.hpp"


namespace vkrt {
namespace render {
using namespace vkrt::gpu;

class DenoisingPass : public ComputePass
{
private:
    vk::Image denoisedImage = nullptr;
    vk::DeviceMemory denoisedImageMemory = nullptr;
    vk::ImageView denoisedImageView = nullptr;

    vk::Image denoisedImagePrevious = nullptr;
    vk::DeviceMemory denoisedImagePreviousMemory = nullptr;
    vk::ImageView denoisedImagePreviousView = nullptr;

    vk::Sampler denoisedSampler = nullptr;

    uint32_t width = 0;
    uint32_t height = 0;
    bool isFirstFrame = true;

protected:
    std::string getShaderPath() const override;
    std::vector<vk::DescriptorSetLayoutBinding> getDescriptorBindings() const override;
    std::vector<vk::DescriptorPoolSize> getDescriptorPoolSizes() const override;
    vk::PushConstantRange* getPushConstantRange() const override;

public:
    void init(const Context& context, uint32_t width, uint32_t height);
    void handleResize(const Context& context, uint32_t newWidth, uint32_t newHeight);
    void denoise(vk::CommandBuffer commandBuffer, float deltaTime);
    void updateDescriptors(const Context& context, vk::ImageView colorImageView, vk::ImageView previousColorImageView,
        vk::ImageView normalImageView, vk::ImageView albedoImageView, vk::ImageView depthImageView);
    void cleanup(const Context& context) override;

    vk::Image getDenoisedImage() const { return denoisedImage; }
    vk::ImageView getDenoisedImageView() const { return denoisedImageView; }
    vk::ImageView getDenoisedImagePreviousView() const { return denoisedImagePreviousView; }
    vk::Sampler getDenoisedSampler() const { return denoisedSampler; }
};

} // namespace render
} // namespace vkrt
