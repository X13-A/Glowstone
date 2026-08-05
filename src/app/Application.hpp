#include "assets/ShaderCompiler.hpp"
#include "core/Constants.hpp"
#include "gpu/Context.hpp"
#include "gpu/Swapchain.hpp"
#include "render/pipeline/GraphicsPipeline.hpp"
#include "scene/Model.hpp"
#include "render/Renderer.hpp"
#include "render/ui/EngineUI.hpp"
#include "app/Window.hpp"
#include "input/Input.hpp"
#include "entt.hpp"
#include "scene/Camera.hpp"
#include "input/Events.hpp"
#include "input/CreativeControls.hpp"
#include "gpu/Tlas.hpp"



namespace vkrt {
namespace app {
using namespace vkrt::gpu;
using namespace vkrt::input;
using namespace vkrt::render;
using namespace vkrt::scene;

class Application
{
private:
    entt::dispatcher dispatcher;
    Window windowManager;
    Input inputManager;
    Camera camera;
    CreativeControls* controls;

    Context context;
    Swapchain swapChainManager;
    CommandBuffers commandBufferManager;
    GraphicsPipeline graphicsPipelineManager;
    
    FullScreenQuad fullScreenQuad;
    Tlas sceneTLAS;
    std::vector<BLASInstance> BLASintances;

    Renderer renderer;
    EngineUI engineUI;

    assets::ShaderCompiler shaderCompiler{ assets::ShaderManifest::loadFromFile(core::SHADERS_MANIFEST) };

    int nativeWidth, nativeHeight;
    int scaledWidth, scaledHeight;
    bool uiFocused = false;

public:
    void run();

    void handleResize();

    bool shouldTerminate() const;

    void handleInputs();

private:
    void initVulkan();

    void handleSceneInputs();

    void setUiFocused(bool focused);

    void reloadShaders();

    bool reloadRayTracingShader(const std::string& name);

    void setSamplingMode(int mode);

    void setFrameAccumulation(bool enabled);

    void setRestirSpatialReuse(bool enabled);

    void handleResourceResizeRequest(const RequestResourceResizeEvent& e);

    void handleShaderReloadRequest(const RequestShaderReloadEvent& e);

    void handleSamplingModeChangeRequest(const RequestSamplingModeChangeEvent& e);

    void handleRenderScaleChangeRequest(const RequestRenderScaleChangeEvent& e);

    void handleFrameAccumulationChangeRequest(const RequestFrameAccumulationChangeEvent& e);

    void handleRestirSpatialReuseChangeRequest(const RequestRestirSpatialReuseChangeEvent& e);

    void mainLoop();

    void limitFrameRate(double frameStart);

    void cleanup();
};

} // namespace app
} // namespace vkrt
