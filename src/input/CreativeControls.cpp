#include "input/CreativeControls.hpp"
#include "input/Input.hpp"
#include "core/Time.hpp"
#include "input/EventManager.hpp"
#include <iostream>


namespace vkrt {
namespace input {
using namespace vkrt::core;
using namespace vkrt::scene;

CreativeControls::CreativeControls(Camera& camera, float moveSpeed, float rotateSpeed) : CameraControls(camera), moveSpeed(moveSpeed), rotateSpeed(rotateSpeed), pitch(0), yaw(0)
{
    EventManager::get().sink<MouseOffsetEvent>().connect<&CreativeControls::handleMouseOffset>(this);
    EventManager::get().sink<MouseScrollEvent>().connect<&CreativeControls::handleMouseScroll>(this);
}

void CreativeControls::handleMouseOffset(const MouseOffsetEvent& e)
{
    yaw += (float) e.xOffset * rotateSpeed * (float) Time::deltaTime();
    pitch += (float) e.yOffset * rotateSpeed * (float) Time::deltaTime();

    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;

    camera.transform.setRotation(glm::vec3(pitch, yaw, 0.0));
    Time::resetFrameCount();
}

void CreativeControls::handleMouseScroll(const MouseScrollEvent& e)
{
    float fov = camera.getFOV() - (float) e.yOffset;
    if (fov < 1.0f) fov = 1.0f;
    if (fov > 90.0f) fov = 90.0f;
    camera.setFOV(fov);
    Time::resetFrameCount();
}

void CreativeControls::update(Input& inputManager)
{
    bool moved = false;
    if (inputManager.isKeyPressed(KeyboardKey::Space))
    {
        camera.transform.translate(glm::vec3(0, moveSpeed * Time::deltaTime(), 0));
        moved = true;
    }
    if (inputManager.isKeyPressed(KeyboardKey::LSHIFT))
    {
        camera.transform.translate(glm::vec3(0, -moveSpeed * Time::deltaTime(), 0));
        moved = true;
    }
    if (inputManager.isKeyPressed(KeyboardKey::A))
    {
        camera.transform.translate(glm::vec3(-moveSpeed * Time::deltaTime(), 0, 0));
        moved = true;
    }
    if (inputManager.isKeyPressed(KeyboardKey::D))
    {
        camera.transform.translate(glm::vec3(moveSpeed * Time::deltaTime(), 0, 0));
        moved = true;
    }
    if (inputManager.isKeyPressed(KeyboardKey::W))
    {
        camera.transform.translate(glm::vec3(0, 0, -moveSpeed * Time::deltaTime()));
        moved = true;
    }
    if (inputManager.isKeyPressed(KeyboardKey::S))
    {
        camera.transform.translate(glm::vec3(0, 0, moveSpeed * Time::deltaTime()));
        moved = true;
    }

    if (moved)
    {
        Time::resetFrameCount();
    }
}

void CreativeControls::cleanup()
{
    EventManager::get().sink<MouseOffsetEvent>().disconnect<&CreativeControls::handleMouseOffset>(this);
    EventManager::get().sink<MouseScrollEvent>().disconnect<&CreativeControls::handleMouseScroll>(this);
}

} // namespace input
} // namespace vkrt
