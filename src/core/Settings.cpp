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
int Settings::risCandidates = 1;
int Settings::samplingMode = 3; // 0: Cosine, 1: MIS, 2: MIS + RIS, 3: RESTIR
bool Settings::restirSpatialReuse = true;

bool Settings::frameAccumulation = false;
bool Settings::frameRateCap = false;
bool Settings::denoisingEnabled = false;
bool Settings::displayVariance = false;
bool Settings::printVarianceSum = false;

} // namespace core
} // namespace vkrt
