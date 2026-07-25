[[vk::binding(0)]] Texture2D<float4> colorBuffer;
[[vk::binding(1)]] Texture2D<float4> previousColorBuffer;
[[vk::binding(2)]] Texture2D<float4> normalBuffer;
[[vk::binding(3)]] Texture2D<float4> albedoBuffer;
[[vk::binding(4)]] Texture2D<float> depthBuffer;
[[vk::binding(5)]] RWTexture2D<float4> denoisedOutput;

[[vk::push_constant]]
cbuffer PushConstants
{
    float deltaTime;
};

#define neighborhoodRadius 3
#define fireflyThreshold 0.3f

float luminance(float3 color)
{
    return dot(color, float3(0.299, 0.587, 0.114));
}

// Does not work well
// Acts as a blur / firefly remover at best
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint2 imageDim;
    colorBuffer.GetDimensions(imageDim.x, imageDim.y);

    int2 pixelCoord = int2(dispatchThreadID.xy);
    if (any(pixelCoord >= int2(imageDim)))
        return;

    float3 centerColor = colorBuffer[pixelCoord].rgb;
    float centerLum = luminance(centerColor);

    // Sample all neighbors within radius
    float3 neighborSum = float3(0, 0, 0);
    int sampleCount = 0;

    for (int dx = -neighborhoodRadius; dx <= neighborhoodRadius; dx++) {
        for (int dy = -neighborhoodRadius; dy <= neighborhoodRadius; dy++) {
            if (dx == 0 && dy == 0) continue;
            
            float dist = sqrt(float(dx * dx + dy * dy));
            if (dist > neighborhoodRadius) continue;
            
            int2 neighborCoord = pixelCoord + int2(dx, dy);
            neighborCoord = clamp(neighborCoord, int2(0, 0), int2(imageDim) - 1);

            float3 neighborColor = colorBuffer[neighborCoord].rgb;
            neighborSum += neighborColor;
            sampleCount++;
        }
    }

    if (sampleCount == 0) {
        denoisedOutput[pixelCoord] = float4(centerColor, 1.0);
        return;
    }

    float3 neighborAvg = neighborSum / sampleCount;
    float neighborAvgLum = luminance(neighborAvg);

    float3 result = centerColor;
    if (abs(centerLum - neighborAvgLum) > fireflyThreshold)
    {
        result = neighborAvg;
    }

    denoisedOutput[pixelCoord] = float4(result, 1.0);
}