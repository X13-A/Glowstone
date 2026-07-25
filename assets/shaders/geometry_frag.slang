struct FragmentInput
{
    float2 fragTexCoord : TEXCOORD0;
    float3 fragNormal : TEXCOORD1;
    float3x3 TBN : TEXCOORD2;
    float4 currClip : TEXCOORD5;
    float4 prevClip : TEXCOORD6;
};

struct UniformBufferObject
{
    float4x4 modelMat;
    float4x4 viewMat;
    float4x4 projMat;
    float4x4 normalMat;
    float4x4 prevModelViewProj;
    bool debug;
};

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBufferObject> ubo;

[[vk::binding(0, 1)]] Sampler2D albedoSampler;
[[vk::binding(1, 1)]] Sampler2D normalSampler;
[[vk::binding(2, 1)]] Sampler2D roughnessSampler;
[[vk::binding(3, 1)]] Sampler2D metalnessSampler;

void main(FragmentInput input, out float4 outNormal : SV_Target0, out float4 outAlbedo : SV_Target2, out float outRoughness : SV_Target3, out float outMetalness : SV_Target4, out float2 outVelocity : SV_Target5)
{
    outAlbedo = albedoSampler.Sample(input.fragTexCoord);

    // Screen-space motion vectors
    // TODO: add support for moving objects
    float2 currUv = (input.currClip.xy / input.currClip.w) * 0.5 + 0.5;
    float2 prevUv = (input.prevClip.xy / input.prevClip.w) * 0.5 + 0.5;
    outVelocity = prevUv - currUv;
    
    float3 defaultNormal = float3(128.0/255.0, 128.0/255.0, 255.0/255.0);

    // Compute normal using normal map
    float3 normalMapValue = normalSampler.Sample(input.fragTexCoord).rgb;    
    float3 worldNormal = normalize(normalMapValue * 2.0 - 1.0);
    worldNormal = normalize(mul(worldNormal, input.TBN));

    // Encode normal
    float3 encodedNormal = (worldNormal * 0.5) + 0.5;
    outNormal = float4(encodedNormal, 1);

    outRoughness = roughnessSampler.Sample(input.fragTexCoord).r;
    outMetalness = metalnessSampler.Sample(input.fragTexCoord).r;
}