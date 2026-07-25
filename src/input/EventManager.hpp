#pragma once
#include "entt.hpp"


namespace vkrt {
namespace input {

class EventManager 
{
public:
    static entt::dispatcher& get();
};

} // namespace input
} // namespace vkrt
