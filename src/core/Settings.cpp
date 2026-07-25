#include "core/Settings.hpp"


namespace vkrt {
namespace core {

float Settings::renderScale = 1;
int Settings::spp = 1;
int Settings::rt_recursion_depth = 1;
bool Settings::displayRayTracing = false;

int Settings::debugIndex1 = 0;
int Settings::debugIndex2 = 0;
bool Settings::debugBool1 = false;
int Settings::risCandidates = 4;
int Settings::samplingMode = 3; // 0: Cosine, 1: MIS, 2: MIS + RIS, 3: RESTIR

} // namespace core
} // namespace vkrt
