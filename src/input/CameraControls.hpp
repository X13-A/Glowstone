#pragma once
#include "scene/Camera.hpp"
#include "input/Input.hpp"
#include "input/Events.hpp"


namespace vkrt {
namespace input {
using namespace vkrt::scene;

class CameraControls
{
protected:
	Camera& camera;

public:
	CameraControls(Camera& camera) : camera(camera)
	{

	}

	virtual void handleMouseOffset(const MouseOffsetEvent& e)
	{

	}

	virtual void handleMouseScroll(const MouseScrollEvent& e)
	{

	}

	virtual void update(Input& inputManager)
	{

	}
};

} // namespace input
} // namespace vkrt
