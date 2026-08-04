#pragma once

#include "restir_trace.slang"

// Includer must have declared:
// - "scene" acceleration structure
// - "reservoirBufferA" and "reservoirBufferB"
// - "velocityBuffer", "sceneData" and "pushConstants"

// #============================================#
// # Reservoir storage (ping-pong)
// #============================================#

PackedReservoir loadReservoir(int parity, uint index)
{
    return parity == 0 ? reservoirBufferB[index] : reservoirBufferA[index];
}

void storeReservoir(int parity, uint index, PackedReservoir p)
{
    if (parity == 0)
    {
        reservoirBufferA[index] = p;
    }
    else
    {
        reservoirBufferB[index] = p;
    }
}

// #==========================================#
// # Initial candidate generation
// #==========================================#

// A single candidate (M = 1) resolved by one closest-hit ray
// That ray also proves the candidate unoccluded, so the visibility reuse needs no extra ray 
Reservoir generateCandidate(Surface q, inout uint rngState)
{
    Reservoir r = createReservoir();

    SamplingResult candidate = sample_mis(q.normal, q.wo, q.position, q.roughness, q.metalness, q.albedo, rngState);
    if (!candidate.isValid || candidate.pdf <= 0.0)
    {
        r.M = 1u; // Dead candidates still count as a sample
        return r;
    }

    CandidateHit hit = traceCandidateRay(q.position, candidate.direction);

    LightSample x = createLightSample();
    float p = 0.0; // PDF of x

    if (!hit.hitGeometry)
    {
        // The sky is infinitely distant, use solid-angle measure
        x.normal = -candidate.direction;
        x.le = getSkyLight(candidate.direction);
        x.flags = kLightIsInfinite;
        p = candidate.pdf;
    }
    else if (isEmissiveAlbedo(hit.albedo))
    {
        // Emitting surfaces use area measure
        x.position = hit.position;
        x.normal = hit.normal;
        x.le = EMISSIVE_RADIANCE;
        p = solidAnglePdfToAreaPdf(candidate.pdf, q, x);
    }
    // TODO: Add path tracing support here
    // else
    // {
    // }

    // w = p_hat(x) / p(x)
    float w = p > 0.0 ? pHat(q, x) / p : 0.0;

    r.update(x, w, rngFloat(rngState));
    r.finalize(q);
    return r;
}

// #============================================#
// # Reuse
// #============================================#

bool historyIsValid()
{
#if ACCUMULATE_FRAMES
    return pushConstants.frameCount > 0u;
#else
    return true;
#endif
}

Reservoir loadNeighbour(Surface q, float qDepth, int2 pixel, uint2 screenSize)
{
    if (!historyIsValid() || any(pixel < 0) || any(uint2(pixel) >= screenSize))
    {
        return createReservoir();
    }

    uint index = uint(pixel.x) + uint(pixel.y) * screenSize.x;
    PackedReservoir packed = loadReservoir(sceneData.reservoirParity, index);

    if (packed.M == 0u || !isSimilarSurface(q.normal, qDepth, packed.q_normal, packed.q_depth))
    {
        return createReservoir();
    }

    Reservoir r = unpackReservoir(packed);
    r.M = min(r.M, RESTIR_M_CAP);
    return r;
}

int2 reprojectPixel(uint2 pixelCoord, uint2 screenSize)
{
    float2 uv = (float2(pixelCoord) + 0.5) / float2(screenSize);

    // velocityBuffer stores the (prevUv - currUv) offset
    float2 motion = velocityBuffer.SampleLevel(uv, 0).rg * float2(screenSize);
    return int2(floor(float2(pixelCoord) + 0.5 + motion));
}

Reservoir temporalReuse(Surface q, float qDepth, Reservoir current, int2 prevPixel, uint2 screenSize, inout uint rngState)
{
    Reservoir s = createReservoir();
    s.merge(q, current, rngFloat(rngState));
    s.merge(q, loadNeighbour(q, qDepth, prevPixel, screenSize), rngFloat(rngState));
    s.finalize(q);
    return s;
}

Reservoir spatialReuse(Surface q, float qDepth, Reservoir temporal, int2 prevPixel, uint2 screenSize, inout uint rngState)
{
    Reservoir s = createReservoir();
    s.merge(q, temporal, rngFloat(rngState));

    for (uint i = 0u; i < RESTIR_SPATIAL_NEIGHBOURS; i++)
    {
        int2 neighbour = prevPixel + sampleDiskOffset(RESTIR_SPATIAL_RADIUS, rngState);
        s.merge(q, loadNeighbour(q, qDepth, neighbour, screenSize), rngFloat(rngState));
    }

    s.finalize(q);
    return s;
}

// #============================================#
// # Final shading
// #============================================#

// Target PDF leaves visibility out, so it is resolved by one shadow ray on the selected sample
float3 shade(Surface q, Reservoir r)
{
    if (r.W <= 0.0 || traceVisibility(q, r.y) <= 0.0)
    {
        return float3(0, 0, 0);
    }

    float3 radiance = unshadowedContribution(q, r.y) * r.W;
    return isValidFloat3(radiance) ? radiance : float3(0, 0, 0);
}
