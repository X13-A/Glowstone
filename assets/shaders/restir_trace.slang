#pragma once

#include "restir.slang"

// Ray helpers for the ReSTIR pass
// Includer must have declared the `scene` acceleration structure

static const float RESTIR_RAY_TMIN = 0.01;
static const float RESTIR_RAY_TMAX = 1000.0;

static const float RESTIR_SHADOW_PULLBACK = 1e-6;

struct CandidateHit
{
    float3 position;
    float3 normal;
    float3 albedo;
    bool hitGeometry;
};

CandidateHit traceCandidateRay(float3 origin, float3 direction)
{
    RayPayload payload = createRayPayload(origin);

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = RESTIR_RAY_TMIN;
    ray.TMax = RESTIR_RAY_TMAX;

    TraceRay(scene, RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, payload);

    CandidateHit hit;
    hit.position = payload.pos;
    hit.normal = payload.normal;
    hit.albedo = payload.albedo;
    hit.hitGeometry = payload.hitGeometry;
    return hit;
}

float traceShadowRay(float3 origin, float3 direction, float tMax)
{
    RayPayload payload = createRayPayload(origin);
    payload.hitGeometry = true; // Miss shader clears this

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = RESTIR_RAY_TMIN;
    ray.TMax = tMax;

    uint flags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
    TraceRay(scene, flags, 0xFF, 0, 1, 0, ray, payload);

    return payload.hitGeometry ? 0.0 : 1.0;
}

float traceVisibility(Surface q, LightSample y)
{
    float3 wi = directionToLight(q, y);

    if ((y.flags & kLightIsInfinite) != 0u)
    {
        return traceShadowRay(q.position, wi, RESTIR_RAY_TMAX);
    }

    float dist = distance(y.position, q.position);
    if (dist <= RESTIR_RAY_TMIN)
    {
        return 0.0;
    }

    return traceShadowRay(q.position, wi, dist - RESTIR_SHADOW_PULLBACK);
}
