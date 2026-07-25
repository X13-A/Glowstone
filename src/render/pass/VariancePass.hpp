#pragma once

#include "render/pass/ComputePass.hpp"
#include "gpu/Context.hpp"
#include "gpu/CommandBuffers.hpp"


namespace vkrt {
namespace render {
using namespace vkrt::gpu;

class VariancePass : public ComputePass
{
private:
    vk::Image varianceImage = nullptr;
    vk::DeviceMemory varianceImageMemory = nullptr;
    vk::ImageView varianceImageView = nullptr;
    vk::Sampler varianceSampler = nullptr;

    vk::Buffer stagingBuffer = nullptr;
    vk::DeviceMemory stagingBufferMemory = nullptr;

    uint32_t width = 0;
    uint32_t height = 0;
    bool firstFrame = true;

protected:
    std::string getShaderPath() const override;
    std::vector<vk::DescriptorSetLayoutBinding> getDescriptorBindings() const override;
    std::vector<vk::DescriptorPoolSize> getDescriptorPoolSizes() const override;

public:
    void init(const Context& context, uint32_t width, uint32_t height);
    void handleResize(const Context& context, uint32_t newWidth, uint32_t newHeight);
    void compute(vk::CommandBuffer commandBuffer, vk::ImageView colorImageView, vk::Image inputImage);
    void updateDescriptors(const Context& context, vk::ImageView colorImageView,
        vk::ImageView normalImageView, vk::ImageView albedoImageView, vk::ImageView depthImageView);
    float sumVarianceTexture(const Context& context, CommandBuffers& commandBufferManager);
    void cleanup(const Context& context) override;

    vk::Image getVarianceImage() const { return varianceImage; }
    vk::ImageView getVarianceImageView() const { return varianceImageView; }
    vk::Sampler getVarianceSampler() const { return varianceSampler; }
};

} // namespace render
} // namespace vkrt
