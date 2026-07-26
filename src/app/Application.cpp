#include "app/Application.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
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
constexpr double TITLE_UPDATE_INTERVAL = 0.25;
constexpr int FRAME_RATE_CAP = 60;
constexpr double FRAME_RATE_CAP_MARGIN = 0.002;
}

void Application::handleWindowResize(const WindowResizeEvent& e)
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
    // Reset sampling mode to default on startup
    // FIXME: forces recompile of ray_gen on every start
    shaderCompiler.setDefine("SAMPLING_MODE", std::to_string(Settings::samplingMode));
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

    EventManager::get().sink<WindowResizeEvent>().connect <&Application::handleWindowResize>(this);
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

void Application::handleInputs()
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
            std::cout << "Ray tracing enabled: " << Settings::displayRayTracing << std::endl;
        }
    }
    if (inputManager.isKeyJustPressed(KeyboardKey::T))
    {
        Settings::renderScale = std::max(std::fmod(Settings::renderScale, 1.0f) + 0.1, 0.1);
        std::cout << "New render scale: " << Settings::renderScale << std::endl;

        WindowResizeEvent e;
        glfwGetFramebufferSize(windowManager.getWindow(), &e.nativeWidth, &e.nativeHeight);
        e.scaledWidth = static_cast<int> ((float)nativeWidth * Settings::renderScale);
        e.scaledHeight = static_cast<int> ((float)nativeHeight * Settings::renderScale);
        EventManager::get().trigger(e);
    }
    int sppOffset = 1;
    if (inputManager.isKeyPressed(KeyboardKey::O))
    {
        Time::resetFrameCount();
        Settings::spp = std::max(1, (int) Settings::spp - sppOffset);
        std::cout << "Samples per pixel: " << Settings::spp << std::endl;
    }
    if (inputManager.isKeyPressed(KeyboardKey::P))
    {
        Time::resetFrameCount();
        Settings::spp = std::max(1, (int)Settings::spp + sppOffset);
        std::cout << "Samples per pixel: " << Settings::spp << std::endl;
    }
    if (inputManager.isKeyJustPressed(KeyboardKey::F))
    {
        frameAccumulationEnabled = !frameAccumulationEnabled;
    }
    if (inputManager.isKeyJustPressed(KeyboardKey::K))
    {
        Time::resetFrameCount();
        Settings::rt_recursion_depth -= 1;
        Settings::rt_recursion_depth = std::clamp(Settings::rt_recursion_depth, 0, RT_MAX_RECURSION_DEPTH);
        std::cout << "RT recursion depth: " << Settings::rt_recursion_depth << std::endl;
    }
    if (inputManager.isKeyJustPressed(KeyboardKey::L))
    {
        Time::resetFrameCount();
        Settings::rt_recursion_depth += 1;
        Settings::rt_recursion_depth = std::clamp(Settings::rt_recursion_depth, 0, RT_MAX_RECURSION_DEPTH);
        std::cout << "RT recursion depth: " << Settings::rt_recursion_depth << std::endl;
    }
    if (inputManager.isKeyJustPressed(KeyboardKey::N))
    {
        setSamplingMode(Settings::samplingMode + 1);
    }
    if (inputManager.isKeyJustPressed(KeyboardKey::B))
    {
        setSamplingMode(Settings::samplingMode - 1);
    }
    if (inputManager.isKeyJustPressed(KeyboardKey::V))
    {
        Time::resetFrameCount();
        Settings::risCandidates += 1;
        std::cout << "RIS candidates: " << Settings::risCandidates << std::endl;
    }
    if (inputManager.isKeyJustPressed(KeyboardKey::C))
    {
        if (inputManager.isKeyPressed(KeyboardKey::LCTRL))
        {
            frameRateCapEnabled = !frameRateCapEnabled;
            std::cout << "Frame rate cap: " << (frameRateCapEnabled ? std::to_string(FRAME_RATE_CAP) + " FPS" : "off") << std::endl;
        }
        else
        {
            Time::resetFrameCount();
            Settings::risCandidates -= 1;
            std::cout << "RIS candidates: " << Settings::risCandidates << std::endl;
        }
    }
    if (inputManager.isKeyJustPressed(KeyboardKey::B))
    {
        Time::resetFrameCount();
        Settings::debugBool1 = !Settings::debugBool1;
    }
    if (inputManager.isKeyJustPressed(KeyboardKey::Y))
    {
        renderer.displayVariance = !renderer.displayVariance;
        if (renderer.displayVariance)
        {
            std::cout << "Variance visualization enabled" << std::endl;
        }
        else
        {
            std::cout << "Variance visualization disabled" << std::endl;
        }
    }

    renderer.displayVarianceSum = false;
    if (inputManager.isKeyJustPressed(KeyboardKey::U))
    {
        renderer.displayVarianceSum = true;
    }

    if (inputManager.isKeyJustPressed(KeyboardKey::I))
    {
        renderer.denoisingEnabled = !renderer.denoisingEnabled;
        if (renderer.denoisingEnabled)
        {
            std::cout << "Denoising enabled" << std::endl;
        }
        else
        {
            std::cout << "Denoising disabled" << std::endl;
        }
    }

    if (!frameAccumulationEnabled)
    {
        Time::resetFrameCount();
    }
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

void Application::setSamplingMode(int mode)
{
    Settings::samplingMode = std::clamp(mode, 0, 3);

    // Push sampling define and recompile shader
    shaderCompiler.setDefine("SAMPLING_MODE", std::to_string(Settings::samplingMode));
    if (!shaderCompiler.compileShader("ray_gen", SHADERS_DIR))
    {
        return;
    }

    context.device.waitIdle();
    graphicsPipelineManager.rtPipeline.reloadShaders(context);

    Time::resetFrameCount();
    std::cout << "Sampling mode: " << Settings::samplingMode << " (recompiled)" << std::endl;
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

            Scene::update();
            renderer.drawFrame(nativeWidth, nativeHeight, scaledWidth, scaledHeight, windowManager.getWindow(), context, swapChainManager, graphicsPipelineManager, commandBufferManager, camera, Scene::getModels(), fullScreenQuad);

            updateWindowTitle();
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

void Application::updateWindowTitle()
{
    frameTimeAccumulator += renderer.getGpuFrameTimeMs();
    frameTimeSamples++;

    double currentTime = Time::time();
    if (currentTime - lastTitleUpdate < TITLE_UPDATE_INTERVAL)
    {
        return;
    }

    std::ostringstream title;
    title << GLFW_WINDOW_NAME << " - " << std::fixed << std::setprecision(2) << frameTimeAccumulator / frameTimeSamples << " ms";
    if (frameRateCapEnabled)
    {
        title << " (capped " << FRAME_RATE_CAP << " FPS)";
    }
    glfwSetWindowTitle(windowManager.getWindow(), title.str().c_str());

    frameTimeAccumulator = 0.0;
    frameTimeSamples = 0;
    lastTitleUpdate = currentTime;
}

void Application::limitFrameRate(double frameStart)
{
    if (!frameRateCapEnabled)
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
    EventManager::get().sink<WindowResizeEvent>().disconnect<&Application::handleWindowResize>(this);
    inputManager.cleanup();
    swapChainManager.cleanup(context.device);
    Scene::cleanup(context.device);
    fullScreenQuad.cleanup(context.device);
    graphicsPipelineManager.cleanup(context.device);
    renderer.cleanup(context);
    commandBufferManager.cleanup(context.device);
    DescriptorLayouts::cleanup(context.device);
    TextureManager::cleanup(context.device);
    context.cleanup();
    windowManager.cleanup();
    glfwTerminate();
}

} // namespace app
} // namespace vkrt
