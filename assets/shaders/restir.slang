#pragma once

#include "ray_common.slang"

// Biased ReSTIR DI
// [Bitterli et al. 2020, "Spatiotemporal reservoir resampling 
// for real-time ray tracing with dynamic direct lighting"]


#ifndef RESTIR_SPATIAL_REUSE
#define RESTIR_SPATIAL_REUSE 1
#endif

static const uint RESTIR_SPATIAL_NEIGHBOURS = 4u;
static const float RESTIR_SPATIAL_RADIUS = 50.0;
static const uint RESTIR_M_CAP = 20u;

// Geometric similarity of a neighbouring visible point
static const float RESTIR_MAX_DEPTH_RATIO = 0.1;
static const float RESTIR_MIN_NORMAL_COS = 0.9063; // cos(25°)

// #============================================#
// # Light samples
// #============================================#
static const uint kLightIsInfinite = 1u << 0u;

struct LightSample
{
    float3 position;
    float3 normal;
    float3 le;       // Emitted radiance
    uint flags;
};

LightSample createLightSample()
{
    LightSample x;
    x.position = float3(0, 0, 0);
    x.normal = float3(0, 0, 0);
    x.le = float3(0, 0, 0);
    x.flags = 0u;
    return x;
}

// The visible surface point of a pixel
struct Surface
{
    float3 position;
    float3 normal;
    float3 albedo;
    float roughness;
    float metalness;
    float3 wo; // Direction to the camera
};

float3 directionToLight(Surface q, LightSample x)
{
    return (x.flags & kLightIsInfinite) != 0u ? -x.normal : normalize(x.position - q.position);
}

// G(q, x) = |cos(theta_x)| / ||x - q||^2
// cos(theta_q) is already folded into brdf_ggx
float geometryTerm(Surface q, LightSample x)
{
    // An infinite light contributes G = 1
    if ((x.flags & kLightIsInfinite) != 0u)
    {
        return 1.0;
    }

    float3 d = x.position - q.position;
    float distSq = dot(d, d);
    if (distSq <= 0.0)
    {
        return 0.0;
    }

    return abs(dot(x.normal, -d * rsqrt(distSq))) / distSq;
}

// Move a solid-angle PDF at q into the area measure at x
float solidAnglePdfToAreaPdf(float pdf, Surface q, LightSample x)
{
    float3 d = x.position - q.position;
    float distSq = dot(d, d);
    if (distSq <= 0.0)
    {
        return 0.0;
    }

    return pdf * abs(dot(x.normal, -d * rsqrt(distSq))) / distSq;
}

// rho(q, x) * Le(x) * G(q, x)
float3 unshadowedContribution(Surface q, LightSample x)
{
    float3 wi = directionToLight(q, x);
    float3 rho = brdf_ggx(q.normal, q.wo, wi, q.roughness, q.metalness, q.albedo);
    return rho * x.le * geometryTerm(q, x);
}

// Target PDF p_hat_q(x), visibility is resolved later during final shading
float pHat(Surface q, LightSample x)
{
    return luminance(unshadowedContribution(q, x));
}

// #============================================#
// # Reservoir
// #============================================#

// Weighted reservoir sampling
struct Reservoir
{
    LightSample y; // Selected sample
    float wSum;    // Sum of the resampling weights seen so far
    uint M;        // Number of samples seen
    float W;       // Contribution weight of y

    [mutating]
    void update(LightSample xi, float wi, float u)
    {
        wSum += wi;
        M += 1u;

        if (u * wSum < wi)
        {
            y = xi;
        }
    }

    // Stream another reservoir in, resampled to the surface point q
    [mutating]
    void merge(Surface q, Reservoir r, float u)
    {
        uint sampleCount = M + r.M;
        update(r.y, pHat(q, r.y) * r.W * float(r.M), u);
        M = sampleCount;
    }

    // W = (1 / p_hat(y)) * (1 / M) * wSum
    [mutating]
    void finalize(Surface q)
    {
        float denom = pHat(q, y) * float(M);
        W = denom > 0.0 ? wSum / denom : 0.0;
    }
};

Reservoir createReservoir()
{
    Reservoir r;
    r.y = createLightSample();
    r.wSum = 0.0;
    r.M = 0u;
    r.W = 0.0;
    return r;
}

bool isSimilarSurface(float3 n0, float depth0, float3 n1, float depth1)
{
    return abs(depth0 - depth1) <= RESTIR_MAX_DEPTH_RATIO * depth0
        && dot(n0, n1) >= RESTIR_MIN_NORMAL_COS;
}

int2 sampleDiskOffset(float radius, inout uint rngState)
{
    float r = radius * sqrt(rngFloat(rngState));
    float phi = 2.0 * PI * rngFloat(rngState);
    return int2(round(float2(r * cos(phi), r * sin(phi))));
}

// #============================================#
// # Screen-space storage
// #============================================#

// Must match PackedReservoir in RayTracingPipeline.hpp
struct PackedReservoir
{
    float3 y_position;
    uint y_flags;
    float3 y_normal;
    float W;
    float3 y_le;
    uint M;
    float3 q_normal;
    float q_depth;
};

PackedReservoir emptyPackedReservoir()
{
    PackedReservoir p;
    p.y_position = float3(0, 0, 0);
    p.y_flags = 0u;
    p.y_normal = float3(0, 0, 0);
    p.W = 0.0;
    p.y_le = float3(0, 0, 0);
    p.M = 0u;
    p.q_normal = float3(0, 0, 0);
    p.q_depth = 0.0;
    return p;
}

PackedReservoir packReservoir(Reservoir r, Surface q, float qDepth)
{
    PackedReservoir p;
    p.y_position = r.y.position;
    p.y_flags = r.y.flags;
    p.y_normal = r.y.normal;
    p.W = r.W;
    p.y_le = r.y.le;
    p.M = r.M;
    p.q_normal = q.normal;
    p.q_depth = qDepth;
    return p;
}

Reservoir unpackReservoir(PackedReservoir p)
{
    Reservoir r = createReservoir();
    r.y.position = p.y_position;
    r.y.normal = p.y_normal;
    r.y.le = p.y_le;
    r.y.flags = p.y_flags;
    r.M = p.M;
    r.W = p.W;
    return r;
}
