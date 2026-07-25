#ifndef CREATIVE_CONTROLS_H
#define CREATIVE_CONTROLS_H

#include "input/CameraControls.hpp"
#include "input/Input.hpp"
#include "input/Events.hpp"


namespace vkrt {
namespace input {
using namespace vkrt::scene;

class CreativeControls : public CameraControls
{
private:
    float moveSpeed;
    float rotateSpeed;
    float yaw;
    float pitch;

public:
    CreativeControls(Camera& camera, float moveSpeed, float rotateSpeed);

    void handleMouseOffset(const MouseOffsetEvent& e) override;

    void handleMouseScroll(const MouseScrollEvent& e) override;

    void update(Input& inputManager) override;

    void cleanup();
};

#endif

} // namespace input
} // namespace vkrt
