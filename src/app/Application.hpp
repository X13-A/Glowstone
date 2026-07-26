#include "assets/ShaderCompiler.hpp"
#include "core/Constants.hpp"
#include "gpu/Context.hpp"
#include "gpu/Swapchain.hpp"
#include "render/pipeline/GraphicsPipeline.hpp"
#include "scene/Model.hpp"
#include "render/Renderer.hpp"
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

    assets::ShaderCompiler shaderCompiler{ assets::ShaderManifest::loadFromFile(core::SHADERS_MANIFEST) };

    double frameTimeAccumulator = 0.0;
    uint32_t frameTimeSamples = 0;
    double lastTitleUpdate = 0.0;

    int nativeWidth, nativeHeight;
    int scaledWidth, scaledHeight;
    bool frameAccumulationEnabled = true;
    bool frameRateCapEnabled = false;

public:
    void run();

    void handleResize();

    bool shouldTerminate() const;

    void handleInputs();

private:
    void initVulkan();

    void reloadShaders();

    void setSamplingMode(int mode);

    void handleWindowResize(const WindowResizeEvent& e);

    void mainLoop();

    void updateWindowTitle();

    void limitFrameRate(double frameStart);

    void cleanup();
};

} // namespace app
} // namespace vkrt
