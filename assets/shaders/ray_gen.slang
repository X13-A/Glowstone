#include "ray_common.slang"

#ifndef SAMPLING_MODE
#define SAMPLING_MODE SAMPLING_MODE_RESTIR
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

#if SAMPLING_MODE == SAMPLING_MODE_RESTIR
PackedReservoir loadReservoir(int parity, uint index)
{
    return parity == 0 ? reservoirBufferB[index] : reservoirBufferA[index];
}

void storeReservoir(int parity, uint index, PackedReservoir r)
{
    if (parity == 0)
    {
        reservoirBufferA[index] = r;
    }
    else
    {
        reservoirBufferB[index] = r;
    }
}
#endif

// Trace a ray and return the incoming radiance
float3 traceRadiance(float3 origin, float3 direction, uint rngState)
{
    RayPayload payload;
    payload.depth = 0;
    payload.maxDepth = sceneData.maxDepth;
    payload.color = float3(0, 0, 0);
    payload.hitGeometry = false;
    payload.t = 0;
    payload.pos = origin;
    payload.rngState = rngState;

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.01;
    ray.TMax = 1000.0;

    TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
    return payload.color;
}

#if SAMPLING_MODE == SAMPLING_MODE_RESTIR
float3 restirTemporal(
    uint2 pixelCoord,
    uint2 screenSize,
    float3 normal,
    float3 viewDir,
    float3 worldPos,
    float roughness,
    float metalness,
    float3 albedo,
    inout uint rngState)
{
    uint reservoirIndex = pixelCoord.x + pixelCoord.y * screenSize.x;

    Reservoir r = createReservoir();

    // 1. Initial candidate
    reservoirAddConfidence(r, 1.0);
    SamplingResult prop = sample_mis(normal, viewDir, worldPos, roughness, metalness, albedo, rngState);
    if (prop.isValid)
    {
        float3 Li = traceRadiance(worldPos, prop.direction, rngState);
        float3 f = brdf_ggx(normal, viewDir, prop.direction, roughness, metalness, albedo);
        float pHat = restirTarget(f, Li);
        float weight = pHat / prop.pdf;
        reservoirAddSample(r, prop.direction, f, Li, pHat, weight, rngState);
    }

    // 2. Temporal reuse
    float2 currentUv = (float2(pixelCoord) + 0.5) / float2(screenSize);
    float2 motion = velocityBuffer.SampleLevel(currentUv, 0).rg;
    float2 prevUv = currentUv + motion;
    if (all(prevUv >= 0.0) && all(prevUv < 1.0))
    {
        // Load previous reservoir
        uint2 prevPixel = uint2(prevUv * float2(screenSize));
        uint prevIndex = prevPixel.x + prevPixel.y * screenSize.x;
        PackedReservoir prev = loadReservoir(sceneData.reservoirParity, prevIndex);

        // Rejection heuristic
        float distToCam = length(worldPos - sceneData.cameraPos);
        bool sameSurface = prev.M > 0.0 && prev.W > 0.0
            && dot(prev.normal, normal) > RESTIR_REUSE_NORMAL_THRESHOLD
            && distance(prev.position, worldPos) < RESTIR_REUSE_POS_THRESHOLD * distToCam;

        // Combine
        if (sameSurface)
        {
            float confidence = min(prev.M, RESTIR_M_CAP);
            float3 fPrev = brdf_ggx(normal, viewDir, prev.wi, roughness, metalness, albedo);
            float pHatPrev = restirTarget(fPrev, prev.Li);
            float weightPrev = pHatPrev * prev.W * confidence;

            reservoirAddConfidence(r, confidence);
            reservoirAddSample(r, prev.wi, fPrev, prev.Li, pHatPrev, weightPrev, rngState);
        }
    }

    // 3. Store the combined reservoir for the next frame
    PackedReservoir packed = packReservoir(r, worldPos, normal);
    storeReservoir(sceneData.reservoirParity, reservoirIndex, packed);

    // 4. Shade with the selected sample: Lo = f(y) * Li(y) * W
    return r.f * r.Li * packed.W;
}
#endif

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

    // Render sky directly if no geometry
    if (depth >= far - 0.1)
    {
        float3 skyColor = getSkyLight(normalizedDir);
        renderTarget[pixelCoord] = float4(skyColor, 1.0);
        return;
    }

    normal = clampNormalToView(normal, -normalizedDir);

    uint pixelId = pixelCoord.x + pixelCoord.y * screenSize.x;
    uint rngState = hash(pixelId + pushConstants.rng);

#if SAMPLING_MODE == SAMPLING_MODE_RESTIR

    float3 currentColor = restirTemporal(
        pixelCoord,
        screenSize,
        normal,
        -normalizedDir,
        worldPos,
        roughness,
        metalness,
        albedo.rgb,
        rngState
    );

    renderTarget[pixelCoord] = float4(currentColor, 1.0);

#else

    int spp = max(sceneData.spp, 1);
    float3 giColor = float3(0, 0, 0);

    for (int i = 0; i < spp; i++)
    {
        float3 giRayDir = float3(0, 0, 0);
        float pdf = 0.0;
        bool isValidSample = false;
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

        // Use only valid samples
        if (result.isValid && result.pdf > 0.0)
        {
            giRayDir = result.direction;
            pdf = result.pdf;
        }
        else
        {
            continue;
        }

        float sampleWeight = 1.0 / pdf;
        float3 tracedColor = traceRadiance(worldPos, giRayDir, rngState);

#if SAMPLING_MODE != SAMPLING_MODE_RIS
        // RIS already computed the BRDF
        brdfValue = brdf_ggx(normal, -normalizedDir, giRayDir, roughness, metalness, albedo.rgb);
#endif // SAMPLING_MODE != SAMPLING_MODE_RIS

        float3 sampleContribution = tracedColor * brdfValue * sampleWeight;

        giColor += sampleContribution;
    }

    // Divide by spp
    float3 currentColor = giColor / float(spp);

    blendResult(previousColor, float4(currentColor, 1), pixelCoord, pushConstants.frameCount);

#endif // SAMPLING_MODE == SAMPLING_MODE_RESTIR
}