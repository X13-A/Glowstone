#include "ray_common.slang"

struct Vertex
{
    float3 position;
    float2 texCoord;
    float3 normal;
};

struct Attributes
{
    float2 barycentrics;
};

struct InstanceData
{
    float4x4 normalMatrix;
    uint meshOffset;
    float padding1;
    float padding2;
    float padding3;
};

struct MeshData
{
    uint indexOffset;
    uint vertexOffset;
};

[[push_constant]]
PushConstants pushConstants;

// Ressources
[[vk::binding(0)]] RaytracingAccelerationStructure scene;
[[vk::binding(2)]] ConstantBuffer<SceneData> sceneData;
[[vk::binding(3)]] ByteAddressBuffer vertexBufferRaw;
[[vk::binding(4)]] StructuredBuffer<uint> indexBuffer;
[[vk::binding(5)]] StructuredBuffer<MeshData> meshDataBuffer;
[[vk::binding(6)]] StructuredBuffer<InstanceData> instanceDataBuffer;

// TODO: make 1024 match the C++ part (compile-time define)
[[vk::binding(7)]] Sampler2D materialsAlbedo[1024];
[[vk::binding(8)]] Sampler2D materialsNormal[1024];
[[vk::binding(9)]] Sampler2D materialsRoughness[1024];
[[vk::binding(10)]] Sampler2D materialsMetalness[1024];

Vertex readVertex(uint vertexIndex)
{
    uint baseOffset = vertexIndex * 32;
    
    Vertex v;

    v.position.x = asfloat(vertexBufferRaw.Load(baseOffset + 0));
    v.position.y = asfloat(vertexBufferRaw.Load(baseOffset + 4));
    v.position.z = asfloat(vertexBufferRaw.Load(baseOffset + 8));

    v.texCoord.x = asfloat(vertexBufferRaw.Load(baseOffset + 12));
    v.texCoord.y = asfloat(vertexBufferRaw.Load(baseOffset + 16));    

    v.normal.x = asfloat(vertexBufferRaw.Load(baseOffset + 20));
    v.normal.y = asfloat(vertexBufferRaw.Load(baseOffset + 24));
    v.normal.z = asfloat(vertexBufferRaw.Load(baseOffset + 28));
    
    return v;
}

[shader("closesthit")]
void main(inout RayPayload payload, in Attributes attribs)
{
    payload.depth = payload.depth + 1;

    InstanceData instanceData = instanceDataBuffer[InstanceIndex()];
    uint meshIndex = instanceData.meshOffset + GeometryIndex();
    MeshData meshData = meshDataBuffer[meshIndex];

    uint i0 = indexBuffer[meshData.indexOffset + PrimitiveIndex() * 3 + 0];
    uint i1 = indexBuffer[meshData.indexOffset + PrimitiveIndex() * 3 + 1];
    uint i2 = indexBuffer[meshData.indexOffset + PrimitiveIndex() * 3 + 2];

    Vertex v0 = readVertex(meshData.vertexOffset + i0);
    Vertex v1 = readVertex(meshData.vertexOffset + i1);
    Vertex v2 = readVertex(meshData.vertexOffset + i2);

    float3 barycentrics = float3(
        1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
        attribs.barycentrics.x,
        attribs.barycentrics.y
    );

    float2 uv = barycentrics.x * v0.texCoord + 
                barycentrics.y * v1.texCoord + 
                barycentrics.z * v2.texCoord;
    
    float3 normal = barycentrics.x * v0.normal + 
                    barycentrics.y * v1.normal + 
                    barycentrics.z * v2.normal;

    normal = mul(instanceData.normalMatrix, float4(normal, 0)).xyz;
    normal = normalize(normal);

    float3 hitPos = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
    payload.pos = hitPos;
    payload.t = RayTCurrent();
    payload.hitGeometry = true;

    float4 textureColor = materialsAlbedo[meshIndex].SampleLevel(uv, 0);

    // TODO: Use normal map value for bounces
    float4 normalMapValue = materialsNormal[meshIndex].SampleLevel(uv, 0);
    float roughness = materialsRoughness[meshIndex].SampleLevel(uv, 0).r;
    float metalness = materialsMetalness[meshIndex].SampleLevel(uv, 0).r;

    payload.color = float3(0, 0, 0);
    uint rngState = payload.rngState;

    if (payload.depth < payload.maxDepth)
    {
        float3 viewDir = -WorldRayDirection();

        normal = clampNormalToView(normal, viewDir);
        float3 bounceDir = float3(0, 0, 0);
        float pdf = 0.0;
        bool isValidSample = false;

#if SAMPLING_MODE == SAMPLING_MODE_REFERENCE
        SamplingResult result = sample_reference(
            normal,
            viewDir,
            hitPos,
            roughness,
            metalness,
            textureColor.rgb, 
            rngState
        );
#else
        SamplingResult result = sample_mis(
            normal,
            viewDir,
            hitPos,
            roughness,
            metalness,
            textureColor.rgb,
            rngState
        );
#endif // SAMPLING_MODE == SAMPLING_MODE_REFERENCE

        if (result.isValid && result.pdf > 0.0)
        {
            bounceDir = result.direction;
            pdf = result.pdf;
            isValidSample = true;
        }

        if (isValidSample)
        {
            float sampleWeight = 1.0 / pdf;

            RayPayload bouncePayload;
            bouncePayload.depth = payload.depth + 1;
            bouncePayload.maxDepth = payload.maxDepth;
            bouncePayload.color = float3(0, 0, 0);
            bouncePayload.hitGeometry = false;
            bouncePayload.rngState = rngState;

            RayDesc ray;
            ray.Origin = payload.pos;
            ray.Direction = bounceDir;
            ray.TMin = 0.01;
            ray.TMax = 1000.0;

            TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, bouncePayload);
            
            // Apply BRDF and weighted contribution
            float3 brdfValue = brdf_ggx(normal, viewDir, bounceDir, roughness, metalness, textureColor.rgb);
            payload.color = brdfValue * bouncePayload.color * sampleWeight;
        }
    }
}