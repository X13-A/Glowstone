#include "input/EventManager.hpp"


namespace vkrt {
namespace input {

entt::dispatcher& EventManager::get() 
{
    static entt::dispatcher instance;
    return instance;
}

} // namespace input
} // namespace vkrt
