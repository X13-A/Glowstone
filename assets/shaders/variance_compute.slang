[[vk::binding(0)]] Texture2D<float4> colorBuffer;
[[vk::binding(1)]] RWTexture2D<float> varianceOutput;

// TODO: take these into account
[[vk::binding(2)]] Texture2D<float4> normalBuffer;
[[vk::binding(3)]] Texture2D<float4> albedoBuffer;
[[vk::binding(4)]] Texture2D<float> depthBuffer;

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int2 pixelCoord = int2(dispatchThreadID.xy);
    
    // Get image dimensions
    uint2 imageDim;
    colorBuffer.GetDimensions(imageDim.x, imageDim.y);
    
    // Bounds check
    if (any(pixelCoord >= int2(imageDim)))
        return;
    
    // Compute local statistics
    float3 sumColor = float3(0.0, 0.0, 0.0);
    float sumColorMagnitudeSq = 0.0;
    float weightSum = 0.0;
    
    const int radius = 1;
    
    // Iterate over neighborhood
    for (int dy = -radius; dy <= radius; dy++)
    {
        for (int dx = -radius; dx <= radius; dx++)
        {
            // Sample color
            int2 sampleCoord = pixelCoord + int2(dx, dy);
            sampleCoord = clamp(sampleCoord, int2(0, 0), int2(imageDim) - int2(1, 1));
            float3 sampleColor = colorBuffer[sampleCoord].rgb;
            
            // Gaussian weight (σ = 1.0)
            float distSq = float(dx * dx + dy * dy);
            float gaussianWeight = exp(-distSq / 2.0);
            
            // Accumulate weighted statistics
            sumColor += sampleColor * gaussianWeight;
            sumColorMagnitudeSq += dot(sampleColor, sampleColor) * gaussianWeight;
            weightSum += gaussianWeight;
        }
    }
    
    // Normalize by total weight
    float3 mean = sumColor / weightSum;
    float meanMagnitudeSq = dot(mean, mean);
    float expectedMagnitudeSq = sumColorMagnitudeSq / weightSum;
    
    // Compute variance
    float variance = max(0.0, expectedMagnitudeSq - meanMagnitudeSq);
    varianceOutput[pixelCoord] = variance;
}