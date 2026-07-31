#pragma once
#include <GLFW/glfw3.h>

#include <vector>
#include <map>
#include "core/Math.hpp"
#include "input/Events.hpp"


namespace vkrt {
namespace input {
using namespace vkrt::core;

enum class KeyboardKey { Escape, Enter, Space, LCTRL, LSHIFT, Tab, Backspace, Left, Right, Up, Down, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, KeysCount };

class Input
{
private:
	std::vector<bool> keysPressed;
	std::vector<bool> keysJustPressed;
	const std::map<int, KeyboardKey> keyMap =
	{
		{GLFW_KEY_ESCAPE, KeyboardKey::Escape},
		{GLFW_KEY_ENTER, KeyboardKey::Enter},
		{GLFW_KEY_SPACE, KeyboardKey::Space},
		{GLFW_KEY_LEFT_CONTROL, KeyboardKey::LCTRL},
		{GLFW_KEY_LEFT_SHIFT, KeyboardKey::LSHIFT},
		{GLFW_KEY_TAB, KeyboardKey::Tab},
		{GLFW_KEY_BACKSPACE, KeyboardKey::Backspace},
		{GLFW_KEY_LEFT, KeyboardKey::Left},
		{GLFW_KEY_RIGHT, KeyboardKey::Right},
		{GLFW_KEY_UP, KeyboardKey::Up},
		{GLFW_KEY_DOWN, KeyboardKey::Down},
		{GLFW_KEY_A, KeyboardKey::A},
		{GLFW_KEY_B, KeyboardKey::B},
		{GLFW_KEY_C, KeyboardKey::C},
		{GLFW_KEY_D, KeyboardKey::D},
		{GLFW_KEY_E, KeyboardKey::E},
		{GLFW_KEY_F, KeyboardKey::F},
		{GLFW_KEY_G, KeyboardKey::G},
		{GLFW_KEY_H, KeyboardKey::H},
		{GLFW_KEY_I, KeyboardKey::I},
		{GLFW_KEY_J, KeyboardKey::J},
		{GLFW_KEY_K, KeyboardKey::K},
		{GLFW_KEY_L, KeyboardKey::L},
		{GLFW_KEY_M, KeyboardKey::M},
		{GLFW_KEY_N, KeyboardKey::N},
		{GLFW_KEY_O, KeyboardKey::O},
		{GLFW_KEY_P, KeyboardKey::P},
		{GLFW_KEY_Q, KeyboardKey::Q},
		{GLFW_KEY_R, KeyboardKey::R},
		{GLFW_KEY_S, KeyboardKey::S},
		{GLFW_KEY_T, KeyboardKey::T},
		{GLFW_KEY_U, KeyboardKey::U},
		{GLFW_KEY_V, KeyboardKey::V},
		{GLFW_KEY_W, KeyboardKey::W},
		{GLFW_KEY_X, KeyboardKey::X},
		{GLFW_KEY_Y, KeyboardKey::Y},
		{GLFW_KEY_Z, KeyboardKey::Z}
	};
	glm::vec2 lastMousePos;
	bool firstMouseInput = true;
	bool mouseLookEnabled = true;

public:
	Input();

	void init();
	void retrieveInputs(GLFWwindow* window);
	void update(GLFWwindow* window);
	bool isKeyPressed(KeyboardKey key) const;
	bool isKeyJustPressed(KeyboardKey key) const;
	void setMouseLookEnabled(bool enabled);

	void onMouseMove(const MouseMoveEvent& e);
	void onMouseScroll(const MouseScrollEvent& e);
	void cleanup();
};

} // namespace input
} // namespace vkrt
