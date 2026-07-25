#include "gpu/DescriptorLayouts.hpp"
#include <stdexcept>
#include <array>
#include "core/Constants.hpp"


namespace vkrt {
namespace gpu {
using namespace vkrt::core;

vk::DescriptorSetLayout DescriptorLayouts::modelLayout = nullptr;
vk::DescriptorSetLayout DescriptorLayouts::materialLayout = nullptr;
vk::DescriptorSetLayout DescriptorLayouts::fullScreenQuadLayout = nullptr;
vk::DescriptorSetLayout DescriptorLayouts::rayTracingDescriptorSetLayout = nullptr;

void DescriptorLayouts::createLayouts(const Context& context)
{
    DescriptorLayouts::createModelLayout(context);
    DescriptorLayouts::createMaterialLayout(context);
    DescriptorLayouts::createFullScreenQuadLayout(context);
    DescriptorLayouts::createRayTracingDescriptorSetLayout(context);
}

void DescriptorLayouts::createModelLayout(const Context& context)
{
    vk::DescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    uboLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;

    std::array<vk::DescriptorSetLayoutBinding, 1> bindings =
    {
        uboLayoutBinding,
    };

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    modelLayout = context.device.createDescriptorSetLayout(layoutInfo);
}

void DescriptorLayouts::createMaterialLayout(const Context& context)
{
    std::vector<vk::DescriptorSetLayoutBinding> bindings;

    vk::DescriptorSetLayoutBinding albedoBinding{};
    albedoBinding.binding = 0;
    albedoBinding.descriptorCount = 1;
    albedoBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    albedoBinding.pImmutableSamplers = nullptr;
    albedoBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
    bindings.push_back(albedoBinding);

    vk::DescriptorSetLayoutBinding normalBinding{};
    normalBinding.binding = 1;
    normalBinding.descriptorCount = 1;
    normalBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    normalBinding.pImmutableSamplers = nullptr;
    normalBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
    bindings.push_back(normalBinding);

    vk::DescriptorSetLayoutBinding roughnessBinding{};
    roughnessBinding.binding = 2;
    roughnessBinding.descriptorCount = 1;
    roughnessBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    roughnessBinding.pImmutableSamplers = nullptr;
    roughnessBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
    bindings.push_back(roughnessBinding);

    vk::DescriptorSetLayoutBinding metallicBinding{};
    metallicBinding.binding = 3;
    metallicBinding.descriptorCount = 1;
    metallicBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    metallicBinding.pImmutableSamplers = nullptr;
    metallicBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
    bindings.push_back(metallicBinding);

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    materialLayout = context.device.createDescriptorSetLayout(layoutInfo);
}

void DescriptorLayouts::createFullScreenQuadLayout(const Context& context)
{
    vk::DescriptorSetLayoutBinding lightingUboLayoutBinding{};
    lightingUboLayoutBinding.binding = 0;
    lightingUboLayoutBinding.descriptorCount = 1;
    lightingUboLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
    lightingUboLayoutBinding.pImmutableSamplers = nullptr;
    lightingUboLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

    vk::DescriptorSetLayoutBinding depthSamplerLayoutBinding{};
    depthSamplerLayoutBinding.binding = 1;
    depthSamplerLayoutBinding.descriptorCount = 1;
    depthSamplerLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    depthSamplerLayoutBinding.pImmutableSamplers = nullptr;
    depthSamplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

    vk::DescriptorSetLayoutBinding normalSamplerLayoutBinding{};
    normalSamplerLayoutBinding.binding = 2;
    normalSamplerLayoutBinding.descriptorCount = 1;
    normalSamplerLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    normalSamplerLayoutBinding.pImmutableSamplers = nullptr;
    normalSamplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

    vk::DescriptorSetLayoutBinding albedoSamplerLayoutBinding{};
    albedoSamplerLayoutBinding.binding = 3;
    albedoSamplerLayoutBinding.descriptorCount = 1;
    albedoSamplerLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    albedoSamplerLayoutBinding.pImmutableSamplers = nullptr;
    albedoSamplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

    vk::DescriptorSetLayoutBinding roughnessSamplerLayoutBinding{};
    roughnessSamplerLayoutBinding.binding = 4;
    roughnessSamplerLayoutBinding.descriptorCount = 1;
    roughnessSamplerLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    roughnessSamplerLayoutBinding.pImmutableSamplers = nullptr;
    roughnessSamplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

    vk::DescriptorSetLayoutBinding metalnessSamplerLayoutBinding{};
    metalnessSamplerLayoutBinding.binding = 5;
    metalnessSamplerLayoutBinding.descriptorCount = 1;
    metalnessSamplerLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    metalnessSamplerLayoutBinding.pImmutableSamplers = nullptr;
    metalnessSamplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

    vk::DescriptorSetLayoutBinding velocitySamplerLayoutBinding{};
    velocitySamplerLayoutBinding.binding = 6;
    velocitySamplerLayoutBinding.descriptorCount = 1;
    velocitySamplerLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    velocitySamplerLayoutBinding.pImmutableSamplers = nullptr;
    velocitySamplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

    std::array<vk::DescriptorSetLayoutBinding, 7> lightingBindings =
    {
        lightingUboLayoutBinding,
        depthSamplerLayoutBinding,
        normalSamplerLayoutBinding,
        albedoSamplerLayoutBinding,
        roughnessSamplerLayoutBinding,
        metalnessSamplerLayoutBinding,
        velocitySamplerLayoutBinding
    };

    vk::DescriptorSetLayoutCreateInfo lightingLayoutInfo{};
    lightingLayoutInfo.bindingCount = static_cast<uint32_t>(lightingBindings.size());
    lightingLayoutInfo.pBindings = lightingBindings.data();

    fullScreenQuadLayout = context.device.createDescriptorSetLayout(lightingLayoutInfo);
}

void DescriptorLayouts::createRayTracingDescriptorSetLayout(const Context& context)
{
    // TLAS
    vk::DescriptorSetLayoutBinding tlasBinding{};
    tlasBinding.binding = 0;
    tlasBinding.descriptorType = vk::DescriptorType::eAccelerationStructureKHR;
    tlasBinding.descriptorCount = 1;
    tlasBinding.stageFlags = vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR;
    tlasBinding.pImmutableSamplers = nullptr;

    // Storage Image
    vk::DescriptorSetLayoutBinding storageImageBinding{};
    storageImageBinding.binding = 1;
    storageImageBinding.descriptorType = vk::DescriptorType::eStorageImage;
    storageImageBinding.descriptorCount = 1;
    storageImageBinding.stageFlags = vk::ShaderStageFlagBits::eRaygenKHR;
    storageImageBinding.pImmutableSamplers = nullptr;

    // Uniform Buffer
    vk::DescriptorSetLayoutBinding uniformBinding{};
    uniformBinding.binding = 2;
    uniformBinding.descriptorType = vk::DescriptorType::eUniformBufferDynamic;
    uniformBinding.descriptorCount = 1;
    uniformBinding.stageFlags = vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eMissKHR;
    uniformBinding.pImmutableSamplers = nullptr;

    // Vertex Buffer
    vk::DescriptorSetLayoutBinding vertexBufferBinding{};
    vertexBufferBinding.binding = 3;
    vertexBufferBinding.descriptorType = vk::DescriptorType::eStorageBuffer;
    vertexBufferBinding.descriptorCount = 1;
    vertexBufferBinding.stageFlags = vk::ShaderStageFlagBits::eClosestHitKHR;
    vertexBufferBinding.pImmutableSamplers = nullptr;

    // Index Buffer
    vk::DescriptorSetLayoutBinding indexBufferBinding{};
    indexBufferBinding.binding = 4;
    indexBufferBinding.descriptorType = vk::DescriptorType::eStorageBuffer;
    indexBufferBinding.descriptorCount = 1;
    indexBufferBinding.stageFlags = vk::ShaderStageFlagBits::eClosestHitKHR;
    indexBufferBinding.pImmutableSamplers = nullptr;

    // Mesh Data Buffer
    vk::DescriptorSetLayoutBinding meshDataBufferBinding{};
    meshDataBufferBinding.binding = 5;
    meshDataBufferBinding.descriptorType = vk::DescriptorType::eStorageBuffer;
    meshDataBufferBinding.descriptorCount = 1;
    meshDataBufferBinding.stageFlags = vk::ShaderStageFlagBits::eClosestHitKHR;
    meshDataBufferBinding.pImmutableSamplers = nullptr;

    // Instance Data Buffer
    vk::DescriptorSetLayoutBinding instanceDataBufferBinding{};
    instanceDataBufferBinding.binding = 6;
    instanceDataBufferBinding.descriptorType = vk::DescriptorType::eStorageBuffer;
    instanceDataBufferBinding.descriptorCount = 1;
    instanceDataBufferBinding.stageFlags = vk::ShaderStageFlagBits::eClosestHitKHR;
    instanceDataBufferBinding.pImmutableSamplers = nullptr;

    // Textures array
    vk::DescriptorSetLayoutBinding instancesAlbedoBinding{};
    instancesAlbedoBinding.binding = 7;
    instancesAlbedoBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    instancesAlbedoBinding.descriptorCount = MAX_MESHES;
    instancesAlbedoBinding.stageFlags = vk::ShaderStageFlagBits::eClosestHitKHR;
    instancesAlbedoBinding.pImmutableSamplers = nullptr;

    // Textures array
    vk::DescriptorSetLayoutBinding instancesNormalBinding{};
    instancesNormalBinding.binding = 8;
    instancesNormalBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    instancesNormalBinding.descriptorCount = MAX_MESHES;
    instancesNormalBinding.stageFlags = vk::ShaderStageFlagBits::eClosestHitKHR;
    instancesNormalBinding.pImmutableSamplers = nullptr;

    // Textures array
    vk::DescriptorSetLayoutBinding instancesRoughnessBinding{};
    instancesRoughnessBinding.binding = 9;
    instancesRoughnessBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    instancesRoughnessBinding.descriptorCount = MAX_MESHES;
    instancesRoughnessBinding.stageFlags = vk::ShaderStageFlagBits::eClosestHitKHR;
    instancesRoughnessBinding.pImmutableSamplers = nullptr;

    // Textures array
    vk::DescriptorSetLayoutBinding instancesMetalnessBinding{};
    instancesMetalnessBinding.binding = 10;
    instancesMetalnessBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    instancesMetalnessBinding.descriptorCount = MAX_MESHES;
    instancesMetalnessBinding.stageFlags = vk::ShaderStageFlagBits::eClosestHitKHR;
    instancesMetalnessBinding.pImmutableSamplers = nullptr;

    // GBuffer
    vk::DescriptorSetLayoutBinding depthBinding{};
    depthBinding.binding = 11;
    depthBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    depthBinding.descriptorCount = 1;
    depthBinding.stageFlags = vk::ShaderStageFlagBits::eRaygenKHR;
    depthBinding.pImmutableSamplers = nullptr;

    vk::DescriptorSetLayoutBinding normalsBinding{};
    normalsBinding.binding = 12;
    normalsBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    normalsBinding.descriptorCount = 1;
    normalsBinding.stageFlags = vk::ShaderStageFlagBits::eRaygenKHR;
    normalsBinding.pImmutableSamplers = nullptr;

    vk::DescriptorSetLayoutBinding albedoBinding{};
    albedoBinding.binding = 13;
    albedoBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    albedoBinding.descriptorCount = 1;
    albedoBinding.stageFlags = vk::ShaderStageFlagBits::eRaygenKHR;
    albedoBinding.pImmutableSamplers = nullptr;

    vk::DescriptorSetLayoutBinding roughnessBinding{};
    roughnessBinding.binding = 14;
    roughnessBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    roughnessBinding.descriptorCount = 1;
    roughnessBinding.stageFlags = vk::ShaderStageFlagBits::eRaygenKHR;
    roughnessBinding.pImmutableSamplers = nullptr;

    vk::DescriptorSetLayoutBinding metalnessBinding{};
    metalnessBinding.binding = 15;
    metalnessBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    metalnessBinding.descriptorCount = 1;
    metalnessBinding.stageFlags = vk::ShaderStageFlagBits::eRaygenKHR;
    metalnessBinding.pImmutableSamplers = nullptr;

    // Frame accumulation
    vk::DescriptorSetLayoutBinding lastImageBinding{};
    lastImageBinding.binding = 16;
    lastImageBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    lastImageBinding.descriptorCount = 1;
    lastImageBinding.stageFlags = vk::ShaderStageFlagBits::eRaygenKHR;
    lastImageBinding.pImmutableSamplers = nullptr;

    // Reservoir buffers
    vk::DescriptorSetLayoutBinding reservoirBindingA{};
    reservoirBindingA.binding = 17;
    reservoirBindingA.descriptorType = vk::DescriptorType::eStorageBuffer;
    reservoirBindingA.descriptorCount = 1;
    reservoirBindingA.stageFlags = vk::ShaderStageFlagBits::eRaygenKHR;
    reservoirBindingA.pImmutableSamplers = nullptr;

    vk::DescriptorSetLayoutBinding reservoirBindingB{};
    reservoirBindingB.binding = 18;
    reservoirBindingB.descriptorType = vk::DescriptorType::eStorageBuffer;
    reservoirBindingB.descriptorCount = 1;
    reservoirBindingB.stageFlags = vk::ShaderStageFlagBits::eRaygenKHR;
    reservoirBindingB.pImmutableSamplers = nullptr;

    // Motion vectors
    vk::DescriptorSetLayoutBinding velocityBinding{};
    velocityBinding.binding = 19;
    velocityBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    velocityBinding.descriptorCount = 1;
    velocityBinding.stageFlags = vk::ShaderStageFlagBits::eRaygenKHR;
    velocityBinding.pImmutableSamplers = nullptr;

    std::array<vk::DescriptorSetLayoutBinding, 20> bindings =
    {
        tlasBinding,
        storageImageBinding,
        uniformBinding,
        vertexBufferBinding,
        indexBufferBinding,
        meshDataBufferBinding,
        instanceDataBufferBinding,
        instancesAlbedoBinding,
        instancesNormalBinding,
        instancesRoughnessBinding,
        instancesMetalnessBinding,
        depthBinding,
        normalsBinding,
        albedoBinding,
        roughnessBinding,
        metalnessBinding,
        lastImageBinding,
        reservoirBindingA,
        reservoirBindingB,
        velocityBinding
    };

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    rayTracingDescriptorSetLayout = context.device.createDescriptorSetLayout(layoutInfo);
}

vk::DescriptorSetLayout DescriptorLayouts::getModelLayout()
{
    if (!modelLayout)
    {
        throw std::runtime_error("Descriptor set layout not initialized !");
    }
    return modelLayout;
}

vk::DescriptorSetLayout DescriptorLayouts::getMaterialLayout()
{
    if (!materialLayout)
    {
        throw std::runtime_error("Descriptor set layout not initialized !");
    }
    return materialLayout;
}

vk::DescriptorSetLayout DescriptorLayouts::getFullScreenQuadLayout()
{
    if (!fullScreenQuadLayout)
    {
        throw std::runtime_error("Descriptor set layout not initialized !");
    }
    return fullScreenQuadLayout;
}

vk::DescriptorSetLayout DescriptorLayouts::getRayTracingLayout()
{
    if (!rayTracingDescriptorSetLayout)
    {
        throw std::runtime_error("Descriptor set layout not initialized !");
    }
    return rayTracingDescriptorSetLayout;
}

void DescriptorLayouts::cleanup(vk::Device device)
{
    if (modelLayout)
    {
        device.destroyDescriptorSetLayout(modelLayout);
    }
    if (materialLayout)
    {
        device.destroyDescriptorSetLayout(materialLayout);
    }
    if (fullScreenQuadLayout)
    {
        device.destroyDescriptorSetLayout(fullScreenQuadLayout);
    }
    if (rayTracingDescriptorSetLayout)
    {
        device.destroyDescriptorSetLayout(rayTracingDescriptorSetLayout);
    }
}

} // namespace gpu
} // namespace vkrt
