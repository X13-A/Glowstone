struct VertexInput
{
    float3 position : POSITION;
    float2 texCoord : TEXCOORD0;
    // TODO: remove these 
    float3 normal : TEXCOORD1;
    float3 tangent : TEXCOORD2;
    float3 bitangent : TEXCOORD3;
}

struct VertexOutput 
{
    float2 fragTexCoord : TEXCOORD0;
    float4 position : SV_POSITION;
};

VertexOutput main(VertexInput input) 
{
    VertexOutput output;

    output.position = float4(input.position, 1.0);
    output.fragTexCoord = input.texCoord;

    return output;
}
