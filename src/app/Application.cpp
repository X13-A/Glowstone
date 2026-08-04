#include "app/Application.hpp"
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include "core/Math.hpp"
#include "input/EventManager.hpp"
#include "core/Time.hpp"
#include "gpu/Tlas.hpp"
#include "scene/Scene.hpp"
#include "core/Settings.hpp"
#include "gpu/DescriptorLayouts.hpp"
#include "assets/TextureManager.hpp"


namespace vkrt {
namespace app {
using namespace vkrt::assets;
using namespace vkrt::core;
using namespace vkrt::gpu;
using namespace vkrt::input;
using namespace vkrt::render;
using namespace vkrt::scene;

namespace {
constexpr int FRAME_RATE_CAP = 60;
constexpr double FRAME_RATE_CAP_MARGIN = 0.002;
}

void Application::handleResourceResizeRequest(const RequestResourceResizeEvent& e)
{
    if (e.scaledWidth <= 0 || e.scaledHeight <= 0 || e.nativeWidth <= 0 || e.nativeHeight <= 0)
    {
        return;
    }

    Time::resetFrameCount();
    scaledWidth = e.scaledWidth;
    scaledHeight = e.scaledHeight;
    nativeWidth = e.nativeWidth;
    nativeHeight = e.nativeHeight;

    std::cout << "Resizing resources..." << std::endl;
    camera.setPerspective(camera.getFOV(), e.scaledWidth / (float)e.scaledHeight, camera.getNearPlane(), camera.getFarPlane());
    swapChainManager.handleResize(e.nativeWidth, e.nativeHeight, context, commandBufferManager, graphicsPipelineManager.lightingPipeline.getRenderPass());
    graphicsPipelineManager.handleResize(e.nativeWidth, e.nativeHeight, e.scaledWidth, e.scaledHeight, context, commandBufferManager);
    fullScreenQuad.writeDescriptorSets(context,
        graphicsPipelineManager.gBufferManager.depthImageView,
        graphicsPipelineManager.gBufferManager.normalImageView,
        graphicsPipelineManager.gBufferManager.albedoImageView,
        graphicsPipelineManager.gBufferManager.roughnessImageView,
        graphicsPipelineManager.gBufferManager.metalnessImageView,
        graphicsPipelineManager.gBufferManager.velocityImageView);

    renderer.varianceCompute.handleResize(context, e.scaledWidth, e.scaledHeight);
    renderer.denoisingPass.handleResize(context, e.scaledWidth, e.scaledHeight);
    engineUI.handleResize(context, swapChainManager);
    renderer.denoisingPass.updateDescriptors(context,
        graphicsPipelineManager.rtPipeline.getStorageImageView(),
        renderer.denoisingPass.getDenoisedImagePreviousView(),
        graphicsPipelineManager.gBufferManager.getNormalImageView(),
        graphicsPipelineManager.gBufferManager.getAlbedoImageView(),
        graphicsPipelineManager.gBufferManager.getDepthImageView());
    renderer.varianceCompute.updateDescriptors(context,
        graphicsPipelineManager.rtPipeline.getStorageImageView(),
        graphicsPipelineManager.gBufferManager.getNormalImageView(),
        graphicsPipelineManager.gBufferManager.getAlbedoImageView(),
        graphicsPipelineManager.gBufferManager.getDepthImageView());
}

void Application::run()
{
    // Reset the raygen defines to their defaults on startup
    // FIXME: forces recompile of ray_gen on every start
    shaderCompiler.setDefine("SAMPLING_MODE", std::to_string(Settings::samplingMode));
    shaderCompiler.setDefine("ACCUMULATE_FRAMES", Settings::frameAccumulation ? "1" : "0");
    shaderCompiler.compileOutdated(SHADERS_DIR);
    shaderCompiler.compileShader("ray_gen", SHADERS_DIR);
    std::cout << "\nStarting engine...\n" << std::endl;

    inputManager.init();
    windowManager.init();

    camera.setPerspective(60, GLFW_WINDOW_WIDTH / (float) GLFW_WINDOW_HEIGHT, 0.1, 100.0f);
    controls = new CreativeControls(camera, 10.0, 100);
    //camera.transform.setRotation(glm::vec3(0, 0, 180));
    //camera.transform.setPosition(glm::vec3(0.312806, 6.96612, -0.113628));
    camera.transform.setTransformMatrix(glm::inverse(glm::lookAt(glm::vec3(0, 0, -5), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0))));
    camera.transform.setScale(glm::vec3(1));

    EventManager::get().sink<RequestResourceResizeEvent>().connect<&Application::handleResourceResizeRequest>(this);
    EventManager::get().sink<RequestShaderReloadEvent>().connect<&Application::handleShaderReloadRequest>(this);
    EventManager::get().sink<RequestSamplingModeChangeEvent>().connect<&Application::handleSamplingModeChangeRequest>(this);
    EventManager::get().sink<RequestRenderScaleChangeEvent>().connect<&Application::handleRenderScaleChangeRequest>(this);
    EventManager::get().sink<RequestFrameAccumulationChangeEvent>().connect<&Application::handleFrameAccumulationChangeRequest>(this);
    initVulkan();
    mainLoop();
    cleanup();
}

void Application::initVulkan()
{
    // Context
    context.init();
    windowManager.createSurface(context);
    context.initDevice();

    DescriptorLayouts::createLayouts(context);

    // CommandBuffers
    commandBufferManager.createCommandPool(context);
    commandBufferManager.createCommandBuffers(context);

    // Load textures
    TextureManager::loadTextures(context, commandBufferManager);

    // Swapchain, pipeline
    glfwGetFramebufferSize(windowManager.getWindow(), &nativeWidth, &nativeHeight);
    scaledWidth = static_cast<int> ((float)nativeWidth * Settings::renderScale);
    scaledHeight = static_cast<int> ((float)nativeHeight * Settings::renderScale);

    swapChainManager.init(nativeWidth, nativeHeight, context);
    graphicsPipelineManager.initPipelines(nativeWidth, nativeHeight, scaledWidth, scaledHeight, context, commandBufferManager, swapChainManager.swapChainImageFormat);

    // Swapchain ressources
    swapChainManager.createFramebuffers(context, graphicsPipelineManager.lightingPipeline.getRenderPass());
    
    Scene::fetchModels();
    
    graphicsPipelineManager.createDescriptorPool(context, Scene::getModelCount(), Scene::getMeshCount(), FULLSCREEN_QUAD_COUNT);

    Scene::loadModels(context, commandBufferManager, graphicsPipelineManager.descriptorPool);

    // Create TLAS
    sceneTLAS.createTLAS(context, Scene::getModels(), commandBufferManager);

    // Setup RT pipeline with scene info
    graphicsPipelineManager.rtPipeline.writeDescriptors(context, commandBufferManager, Scene::getModels(), sceneTLAS.getTLAS(), graphicsPipelineManager.gBufferManager.depthImageView, graphicsPipelineManager.gBufferManager.normalImageView, graphicsPipelineManager.gBufferManager.albedoImageView, graphicsPipelineManager.gBufferManager.roughnessImageView, graphicsPipelineManager.gBufferManager.metalnessImageView, graphicsPipelineManager.gBufferManager.velocityImageView);

    // Init fullscreen quad
    fullScreenQuad.init(context, commandBufferManager,
        graphicsPipelineManager.descriptorPool,
        graphicsPipelineManager.gBufferManager.depthImageView,
        graphicsPipelineManager.gBufferManager.normalImageView,
        graphicsPipelineManager.gBufferManager.albedoImageView,
        graphicsPipelineManager.gBufferManager.roughnessImageView,
        graphicsPipelineManager.gBufferManager.metalnessImageView,
        graphicsPipelineManager.gBufferManager.velocityImageView);

    // Renderer
    renderer.init(context, swapChainManager);

    engineUI.init(context, swapChainManager, windowManager.getWindow());
    renderer.setOverlayPass(&engineUI);

    // Update descriptors after RT & G-buffer are initialized
    renderer.denoisingPass.updateDescriptors(context,
        graphicsPipelineManager.rtPipeline.getStorageImageView(),
        renderer.denoisingPass.getDenoisedImagePreviousView(),
        graphicsPipelineManager.gBufferManager.getNormalImageView(),
        graphicsPipelineManager.gBufferManager.getAlbedoImageView(),
        graphicsPipelineManager.gBufferManager.getDepthImageView());
    renderer.varianceCompute.updateDescriptors(context,
        graphicsPipelineManager.rtPipeline.getStorageImageView(),
        graphicsPipelineManager.gBufferManager.getNormalImageView(),
        graphicsPipelineManager.gBufferManager.getAlbedoImageView(),
        graphicsPipelineManager.gBufferManager.getDepthImageView());

    std::cout << "VK initialization finished !" << std::endl;
}

void Application::setUiFocused(bool focused)
{
    uiFocused = focused;
    glfwSetInputMode(windowManager.getWindow(), GLFW_CURSOR, focused ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
    inputManager.setMouseLookEnabled(!focused);
    engineUI.setInputEnabled(focused);
}

void Application::handleInputs()
{
    if (inputManager.isKeyJustPressed(KeyboardKey::Tab))
    {
        setUiFocused(!uiFocused);
    }

    if (!uiFocused)
    {
        handleSceneInputs();
    }

    if (!Settings::frameAccumulation)
    {
        Time::resetFrameCount();
    }
}

void Application::handleSceneInputs()
{
    controls->update(inputManager);

    if (inputManager.isKeyJustPressed(KeyboardKey::R))
    {
        if (inputManager.isKeyPressed(KeyboardKey::LCTRL))
        {
            reloadShaders();
        }
        else
        {
            Time::resetFrameCount();
            Settings::displayRayTracing = !Settings::displayRayTracing;
        }
    }
}

void Application::handleShaderReloadRequest(const RequestShaderReloadEvent& e)
{
    reloadShaders();
}

void Application::handleSamplingModeChangeRequest(const RequestSamplingModeChangeEvent& e)
{
    setSamplingMode(e.mode);
}

void Application::handleFrameAccumulationChangeRequest(const RequestFrameAccumulationChangeEvent& e)
{
    setFrameAccumulation(e.enabled);
}

void Application::handleRenderScaleChangeRequest(const RequestRenderScaleChangeEvent& e)
{
    Settings::renderScale = e.scale;

    RequestResourceResizeEvent resize;
    glfwGetFramebufferSize(windowManager.getWindow(), &resize.nativeWidth, &resize.nativeHeight);
    resize.scaledWidth = static_cast<int> ((float)resize.nativeWidth * Settings::renderScale);
    resize.scaledHeight = static_cast<int> ((float)resize.nativeHeight * Settings::renderScale);
    EventManager::get().trigger(resize);
}

void Application::reloadShaders()
{
    if (!shaderCompiler.compileOutdated(SHADERS_DIR))
    {
        return;
    }

    context.device.waitIdle();
    graphicsPipelineManager.reloadShaders(context);
    renderer.reloadShaders(context);

    Time::resetFrameCount();
    std::cout << "Shaders reloaded" << std::endl;
}

bool Application::reloadRayTracingShader(const std::string& name)
{
    if (!shaderCompiler.compileShader(name, SHADERS_DIR))
    {
        return false;
    }

    context.device.waitIdle();
    graphicsPipelineManager.rtPipeline.reloadShaders(context);

    Time::resetFrameCount();
    return true;
}

void Application::setSamplingMode(int mode)
{
    Settings::samplingMode = std::clamp(mode, 0, 3);

    shaderCompiler.setDefine("SAMPLING_MODE", std::to_string(Settings::samplingMode));
    if (reloadRayTracingShader("ray_gen"))
    {
        std::cout << "Sampling mode: " << Settings::samplingMode << " (recompiled)" << std::endl;
    }
}

void Application::setFrameAccumulation(bool enabled)
{
    Settings::frameAccumulation = enabled;

    shaderCompiler.setDefine("ACCUMULATE_FRAMES", enabled ? "1" : "0");
    if (reloadRayTracingShader("ray_gen"))
    {
        std::cout << "Frame accumulation: " << (enabled ? "on" : "off") << " (recompiled)" << std::endl;
    }
}

void Application::mainLoop()
{
    while (!shouldTerminate())
    {
        try
        {
            double frameStart = Time::time();

            glfwPollEvents();
            inputManager.retrieveInputs(windowManager.getWindow());
            handleInputs();

            engineUI.beginFrame(renderer.getGpuProfiler(), Time::deltaTime() * 1000.0);

            EventManager::get().update();

            Scene::update();
            renderer.drawFrame(nativeWidth, nativeHeight, scaledWidth, scaledHeight, windowManager.getWindow(), context, swapChainManager, graphicsPipelineManager, commandBufferManager, camera, Scene::getModels(), fullScreenQuad);

            limitFrameRate(frameStart);

            Time::update();
        }
        catch (std::exception e)
        {
            std::cerr << e.what() << std::endl;
            break;
        }
    }

    context.device.waitIdle();
}

void Application::limitFrameRate(double frameStart)
{
    if (!Settings::frameRateCap)
    {
        return;
    }

    const double targetEnd = frameStart + 1.0 / FRAME_RATE_CAP;

    double remaining = targetEnd - Time::time();
    if (remaining > FRAME_RATE_CAP_MARGIN)
    {
        std::this_thread::sleep_for(std::chrono::duration<double>(remaining - FRAME_RATE_CAP_MARGIN));
    }

    while (Time::time() < targetEnd)
    {
        std::this_thread::yield();
    }
}

void Application::handleResize()
{
    swapChainManager.framebufferResized = true;
}

bool Application::shouldTerminate() const
{
    if (inputManager.isKeyPressed(KeyboardKey::Escape))
    {
        return true;
    }
    if (glfwWindowShouldClose(windowManager.getWindow()))
    {
        return true;
    }
    return false;
}

void Application::cleanup()
{
    sceneTLAS.cleanup(context);
    controls->cleanup();
    delete controls;
    EventManager::get().sink<RequestResourceResizeEvent>().disconnect<&Application::handleResourceResizeRequest>(this);
    EventManager::get().sink<RequestShaderReloadEvent>().disconnect<&Application::handleShaderReloadRequest>(this);
    EventManager::get().sink<RequestSamplingModeChangeEvent>().disconnect<&Application::handleSamplingModeChangeRequest>(this);
    EventManager::get().sink<RequestRenderScaleChangeEvent>().disconnect<&Application::handleRenderScaleChangeRequest>(this);
    EventManager::get().sink<RequestFrameAccumulationChangeEvent>().disconnect<&Application::handleFrameAccumulationChangeRequest>(this);
    inputManager.cleanup();
    engineUI.cleanup(context);
    renderer.cleanup(context);
    swapChainManager.cleanup(context.device);
    Scene::cleanup(context.device);
    fullScreenQuad.cleanup(context.device);
    graphicsPipelineManager.cleanup(context.device);
    commandBufferManager.cleanup(context.device);
    DescriptorLayouts::cleanup(context.device);
    TextureManager::cleanup(context.device);
    context.cleanup();
    windowManager.cleanup();
    glfwTerminate();
}

} // namespace app
} // namespace vkrt
