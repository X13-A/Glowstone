#pragma once

namespace vkrt {
namespace core {

class Settings
{
public:
    static float renderScale;
    static int spp;
    static int rt_recursion_depth;
    static bool displayRayTracing;
    static int debugIndex1;
    static int debugIndex2;
    static bool debugBool1;
    static int risCandidates;
    static int samplingMode;

    static bool frameAccumulation;
    static bool frameRateCap;
    static bool denoisingEnabled;
    static bool displayVariance;
    static bool printVarianceSum;
};

} // namespace core
} // namespace vkrt
