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

struct VertexInput
{
    float3 position : POSITION;
    float2 texCoord : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float3 tangent : TEXCOORD2;
    float3 bitangent : TEXCOORD3;
}

struct VertexOutput
{
    float4 svPosition : SV_Position;
    float2 fragTexCoord : TEXCOORD0;
    float3 fragNormal : TEXCOORD1;
    float3x3 TBN : TEXCOORD2;
    float4 currClip : TEXCOORD5;
    float4 prevClip : TEXCOORD6;
};

VertexOutput main(VertexInput input)
{
    VertexOutput output;

    // TODO: add support for moving objects in motion vectors
    float4 clipPos = mul(ubo.projMat, mul(ubo.viewMat, mul(ubo.modelMat, float4(input.position, 1.0))));
    output.svPosition = clipPos;
    output.currClip = clipPos;
    output.prevClip = mul(ubo.prevModelViewProj, float4(input.position, 1.0));
    output.fragTexCoord = input.texCoord;

    // TODO: pass TBN as VertexInput ?
    float3 T = normalize(mul(ubo.normalMat, float4(input.tangent, 0.0)).xyz);
    float3 B = normalize(mul(ubo.normalMat, float4(input.bitangent, 0.0)).xyz);
    float3 N = normalize(mul(ubo.normalMat, float4(input.normal, 0.0)).xyz);

    output.TBN = float3x3(T, B, N);
    output.fragNormal = N;

    return output;
}