#include "ray_common.slang"

#ifndef SAMPLING_MODE
#define SAMPLING_MODE SAMPLING_MODE_RESTIR
#endif

#if SAMPLING_MODE == SAMPLING_MODE_RESTIR
#include "restir.slang"
#endif

[[push_constant]]
PushConstants pushConstants;

[[vk::binding(0)]] RaytracingAccelerationStructure scene;
[[vk::binding(1)]] RWTexture2D<float4> renderTarget;
[[vk::binding(2)]] ConstantBuffer<SceneData> sceneData;

[[vk::binding(11)]] Sampler2D depthBuffer;
[[vk::binding(12)]] Sampler2D normalBuffer;
[[vk::binding(13)]] Sampler2D albedoBuffer;
[[vk::binding(14)]] Sampler2D roughnessBuffer; // TODO: combine these
[[vk::binding(15)]] Sampler2D metalnessBuffer;
[[vk::binding(16)]] Sampler2D previousFrame;

#if SAMPLING_MODE == SAMPLING_MODE_RESTIR
[[vk::binding(17)]] RWStructuredBuffer<PackedReservoir> reservoirBufferA;
[[vk::binding(18)]] RWStructuredBuffer<PackedReservoir> reservoirBufferB;
[[vk::binding(19)]] Sampler2D velocityBuffer;

#include "restir_stages.slang"
#endif

#if SAMPLING_MODE != SAMPLING_MODE_RESTIR
void blendResult(float4 previousColor, float4 currentColor, uint2 pixelCoord, uint frameCount)
{
    // Temporal accumulation
    float weight = 1.0 / float(frameCount + 1);
    float4 finalColor = lerp(previousColor, currentColor, weight);
    renderTarget[pixelCoord] = float4(finalColor.rgb, 1.0);
}
#endif

// Trace a ray and return the incoming radiance
float3 traceRadiance(float3 origin, float3 direction, uint rngState)
{
    RayPayload payload = createRayPayload(origin);
    payload.maxDepth = sceneData.maxDepth;
    payload.rngState = rngState;

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.01;
    ray.TMax = 1000.0;

    TraceRay(scene, RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, payload);
    return payload.color;
}

[shader("raygeneration")]
void main()
{
    uint2 pixelCoord = DispatchRaysIndex().xy;
    uint2 screenSize = DispatchRaysDimensions().xy;

    float2 uv = (float2(pixelCoord) + 0.5) / float2(screenSize);

    // Read buffers
    float3 encodedNormal = normalBuffer.SampleLevel(uv, 0).rgb;
    float3 normal = normalize((encodedNormal - 0.5) * 2.0);
    float4 albedo = albedoBuffer.SampleLevel(uv, 0).rgba;
    float depth = depthBuffer.SampleLevel(uv, 0).r;
    float metalness = metalnessBuffer.SampleLevel(uv, 0).r;
    float roughness = roughnessBuffer.SampleLevel(uv, 0).r;
    float4 previousColor = previousFrame.SampleLevel(uv, 0);

    // Linearize depth
    float near = sceneData.nearFar.x;
    float far = sceneData.nearFar.y;
    depth = linearize_depth(depth, near, far);

    // Compute ray dir and world position
    float2 ndc = uv * 2.0 - 1.0;
    float4 origin = mul(sceneData.viewInverse, float4(0, 0, 0, 1));
    float4 target = mul(sceneData.projInverse, float4(ndc, 1, 1));
    float3 dir = mul(sceneData.viewInverse, float4(target.xyz, 0)).xyz;
    float3 normalizedDir = normalize(dir);
    float3 worldPos = origin.xyz + dir * depth;

    uint pixelId = pixelCoord.x + pixelCoord.y * screenSize.x;

    // Render sky directly if no geometry
    if (depth >= far - 0.1)
    {
        float3 skyColor = getSkyLight(normalizedDir);
        renderTarget[pixelCoord] = float4(skyColor, 1.0);
#if SAMPLING_MODE == SAMPLING_MODE_RESTIR
        // Leave an empty reservoir
        storeReservoir(sceneData.reservoirParity, pixelId, emptyPackedReservoir());
#endif
        return;
    }

    // Handle emissive surfaces
    if (isEmissiveAlbedo(albedo.rgb))
    {
        renderTarget[pixelCoord] = float4(EMISSIVE_RADIANCE, 1.0);
#if SAMPLING_MODE == SAMPLING_MODE_RESTIR
        storeReservoir(sceneData.reservoirParity, pixelId, emptyPackedReservoir());
#endif
        return;
    }

    normal = clampNormalToView(normal, -normalizedDir);
    uint rngState = hash(pixelId) ^ hash(pushConstants.rng);

#if SAMPLING_MODE == SAMPLING_MODE_RESTIR

    Surface q;
    q.position = worldPos;
    q.normal = normal;
    q.albedo = albedo.rgb;
    q.roughness = roughness;
    q.metalness = metalness;
    q.wo = normalize(sceneData.cameraPos - worldPos);

    float qDepth = distance(sceneData.cameraPos, worldPos);
    
    int2 prevPixel = reprojectPixel(pixelCoord, screenSize);

    Reservoir current = generateCandidate(q, rngState);
    Reservoir temporal = temporalReuse(q, qDepth, current, prevPixel, screenSize, rngState);

    // Only the temporal result is fed forward
    storeReservoir(sceneData.reservoirParity, pixelId, packReservoir(temporal, q, qDepth));

#if RESTIR_SPATIAL_REUSE
    Reservoir final = spatialReuse(q, qDepth, temporal, prevPixel, screenSize, rngState);
#else
    Reservoir final = temporal;
#endif

    renderTarget[pixelCoord] = float4(shade(q, final), 1.0);
    return;
#else

    int spp = max(sceneData.spp, 1);
    float3 giColor = float3(0, 0, 0);

    for (int i = 0; i < spp; i++)
    {
        float3 brdfValue = float3(0, 0, 0);

#if SAMPLING_MODE == SAMPLING_MODE_REFERENCE
        SamplingResult result = sample_reference(
            normal, 
            -normalizedDir, 
            worldPos, 
            roughness, 
            metalness, 
            albedo.rgb,
            rngState
        );
#elif SAMPLING_MODE == SAMPLING_MODE_RIS
        SamplingResult result = sample_ris(
            normal,
            -normalizedDir,
            worldPos,
            roughness,
            metalness,
            albedo.rgb,
            sceneData.risCandidates,
            rngState,
            brdfValue
        );
#else // SAMPLING_MODE_MIS
        SamplingResult result = sample_mis(
            normal,
            -normalizedDir,
            worldPos,
            roughness,
            metalness,
            albedo.rgb,
            rngState
        );
#endif // SAMPLING_MODE == SAMPLING_MODE_REFERENCE

        if (!result.isValid || result.pdf <= 0.0)
        {
            continue;
        }

        float3 tracedColor = traceRadiance(worldPos, result.direction, rngState);

#if SAMPLING_MODE != SAMPLING_MODE_RIS
        // RIS already computed the BRDF
        brdfValue = brdf_ggx(normal, -normalizedDir, result.direction, roughness, metalness, albedo.rgb);
#endif // SAMPLING_MODE != SAMPLING_MODE_RIS

        float3 sampleContribution = result.weight * ((tracedColor * brdfValue) / result.pdf);

        giColor += sampleContribution;
    }

    // Divide by spp
    float3 currentColor = giColor / float(spp);

    blendResult(previousColor, float4(currentColor, 1), pixelCoord, pushConstants.frameCount);

#endif // SAMPLING_MODE == SAMPLING_MODE_RESTIR
}