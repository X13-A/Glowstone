#include "input/Input.hpp"
#include <iostream>
#include "core/Math.hpp"
#include "input/EventManager.hpp"


namespace vkrt {
namespace input {
using namespace vkrt::core;

Input::Input()
{
	lastMousePos = glm::vec2(0, 0);
}

void Input::init()
{
	keysPressed.resize(static_cast<size_t>(KeyboardKey::KeysCount), false);
	keysJustPressed.resize(static_cast<size_t>(KeyboardKey::KeysCount), false);
	EventManager::get().sink<MouseMoveEvent>().connect<&Input::onMouseMove>(this);
	EventManager::get().sink<MouseScrollEvent>().connect<&Input::onMouseScroll>(this);
}

void Input::retrieveInputs(GLFWwindow* window)
{
	for (const std::pair<int, KeyboardKey> pair : keyMap)
	{
		if (pair.second == KeyboardKey::KeysCount) break;
		bool keyPressed = glfwGetKey(window, pair.first) == GLFW_PRESS;

		bool wasKeyAlreadyPressed = keysPressed.at(static_cast<size_t>(pair.second));
		if (!wasKeyAlreadyPressed && keyPressed)
		{
			keysJustPressed.at(static_cast<size_t>(pair.second)) = true;
		}
		else
		{
			keysJustPressed.at(static_cast<size_t>(pair.second)) = false;
		}

		keysPressed.at(static_cast<size_t>(pair.second)) = keyPressed;
	}
}

void Input::setMouseLookEnabled(bool enabled)
{
	mouseLookEnabled = enabled;
	firstMouseInput = true;
}

void Input::onMouseMove(const MouseMoveEvent& e)
{
	if (!mouseLookEnabled)
	{
		return;
	}

	if (firstMouseInput)
	{
		firstMouseInput = false;
		lastMousePos.x = e.xPos;
		lastMousePos.y = e.yPos;
	}

	if ((glm::vec3(e.xPos, e.yPos, 0) - glm::vec3(lastMousePos.x, lastMousePos.y, 0)).length() > 100)
	{
		lastMousePos.x = e.xPos;
		lastMousePos.y = e.yPos;
		return;
	}
	float offsetX = lastMousePos.x - e.xPos;
	float offsetY = lastMousePos.y - e.yPos;

	EventManager::get().trigger(MouseOffsetEvent{ offsetX, offsetY });
	
	lastMousePos.x = e.xPos;
	lastMousePos.y = e.yPos;
}

void Input::onMouseScroll(const MouseScrollEvent& e)
{
	//std::cout << "Mouse scroll: " << e.xOffset << ", " << e.yOffset << std::endl;
}

bool Input::isKeyPressed(KeyboardKey key) const
{
	return keysPressed.at(static_cast<size_t>(key));
}

bool Input::isKeyJustPressed(KeyboardKey key) const
{
	return keysJustPressed.at(static_cast<size_t>(key));
}

void Input::update(GLFWwindow* window)
{
	retrieveInputs(window);
}

void Input::cleanup()
{
	EventManager::get().sink<MouseMoveEvent>().disconnect<&Input::onMouseMove>(this);
	EventManager::get().sink<MouseScrollEvent>().disconnect<&Input::onMouseScroll>(this);
}

} // namespace input
} // namespace vkrt
