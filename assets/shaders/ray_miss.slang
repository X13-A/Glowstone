#include "ray_common.slang"

// Binding 2: Uniform Buffer
ConstantBuffer<SceneData> sceneData;

[shader("miss")]
void main(inout RayPayload payload)
{
    float3 rayDir = WorldRayDirection();

    payload.color = getSkyLight(rayDir);
    payload.t = 1e10;
    payload.pos = 0;
    payload.hitGeometry = false;
}