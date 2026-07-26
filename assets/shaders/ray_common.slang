#pragma once

// #============================================#
// # Constants and defines
// #============================================#

static float PI = 3.14159265359;
static float3 SUN_DIR = normalize(-float3(0.5, 1, 0.5));
static float SUN_ANGULAR_RADIUS = 0.0873f; // Actual sun is ~0.5 degrees. 10x here for softer shadows
static float3 SUN_COLOR = float3(15.0, 12.0, 8.0) * 100.0;
static float3 SKY_COLOR = float3(62.0 / 255.0, 105.0 / 255.0, 196.0 / 255.0) * 2;

// FIXME: Emission hack
static float3 EMISSIVE_RADIANCE = float3(1.0, 1.0, 1.0) * 10.0;
static float EMISSIVE_ALBEDO_EPSILON = 0.01;

#define SAMPLING_MODE_REFERENCE 0
#define SAMPLING_MODE_MIS 1
#define SAMPLING_MODE_RIS 2
#define SAMPLING_MODE_RESTIR 3

// #============================================#
// # Etc
// #============================================#

float3 getSkyLight(float3 rayDir)
{
    float3 sunDir = -SUN_DIR;
    float cosSunRadius = cos(SUN_ANGULAR_RADIUS);
    float sunDot = max(0.0, dot(rayDir, sunDir) - cosSunRadius);
    float normalizer = 1.0 / (1.0 - cosSunRadius);
    return SKY_COLOR + SUN_COLOR * sunDot * normalizer;
}

bool isEmissiveAlbedo(float3 albedo)
{
    return all(albedo >= 1.0 - EMISSIVE_ALBEDO_EPSILON);
}

bool isValidFloat3(float3 c)
{
    return !(any(isnan(c)) || any(isinf(c)));
}

// #============================================#
// # Structures
// #============================================#

struct RayPayload
{
    float3 color;
    float t;
    bool hitGeometry;
    float3 pos;
    int depth;
    int maxDepth;
    uint rngState;
    float3 normal; // Shading normal at hitpoint (undefined on a miss)
    float3 albedo; // Albedo at hitpoint         (undefined on a miss)
};

RayPayload createRayPayload(float3 origin)
{
    RayPayload payload;
    payload.color = float3(0, 0, 0);
    payload.t = 0.0;
    payload.hitGeometry = false;
    payload.pos = origin;
    payload.depth = 0;
    payload.maxDepth = 0;
    payload.rngState = 0;
    payload.normal = float3(0, 0, 0);
    payload.albedo = float3(0, 0, 0);
    return payload;
}

struct PushConstants
{
    uint frameCount;
    uint rng;
};

struct SceneData
{
    float4x4 viewMatrix;
    float4x4 projMatrix;
    float4x4 viewInverse;
    float4x4 projInverse;
    float3 cameraPos;
    uint maxDepth;
    float2 nearFar;
    int resolutionX;
    int resolutionY;
    int spp;
    int risCandidates;
    int reservoirParity;
};

struct SamplingResult
{
    float3 direction;
    float pdf;
    float weight;
    bool isValid;
};

// #============================================#
// # Sampling functions
// #============================================#

SamplingResult sample_reference(
    float3 normal,
    float3 viewDir,
    float3 worldPos,
    float roughness,
    float metalness,
    float3 albedo,
    inout uint rngState)
{
    SamplingResult result;
    result.direction = sample_uniform(normal, rngState);
    result.pdf = sample_uniform_pdf();
    result.isValid = true;
    result.weight = 1.0;
    return result;
}

SamplingResult sample_mis(
    float3 normal,
    float3 viewDir,
    float3 worldPos,
    float roughness,
    float metalness,
    float3 albedo,
    inout uint rngState)
{
    SamplingResult result;
    result.isValid = false;
    result.pdf = 0.0;

    // Compute selection probabilities
    float specularWeight = (1.0 - roughness) * 0.5 + 0.5 * metalness;
    specularWeight = clamp(specularWeight, 0.1, 0.9);

    float probCone = 0.2;
    float probGGX = (1.0 - probCone) * specularWeight;
    float probCosine = (1.0 - probCone) * (1.0 - specularWeight);
    // probCone = 0.0;
    // probGGX = 0.0;
    // probCosine = 1.0;

    // Randomly select sampling technique
    float xi = rngFloat(rngState);
    if (xi < probCone)
    {
        result.direction = sample_cone(-SUN_DIR, SUN_ANGULAR_RADIUS, rngState);
    }
    else if (xi < probCone + probCosine)
    {
        result.direction = sample_cosine(normal, rngState);
    }
    else
    {
        result.direction = sample_ggx(normal, viewDir, roughness, rngState);
    }

    if (dot(result.direction, result.direction) < 0.5)
    {
        return result;
    }

    // Compute PDFs
    float3 halfVector = computeHalfVector(viewDir, result.direction);
    float conePdf = cone_sampling_pdf(result.direction, -SUN_DIR, SUN_ANGULAR_RADIUS);
    float ggxPdf = ggx_vndf_pdf(viewDir, result.direction, halfVector, normal, roughness);
    float cosinePdf = sample_cosine_pdf(normal, result.direction);

    float combinedPdf = probCone * conePdf + probCosine * cosinePdf + probGGX * ggxPdf;
    if (combinedPdf <= 0.0)
    {
        return result;
    }

    result.pdf = combinedPdf;
    result.weight = 1.0;
    result.isValid = true;
    return result;
}

// FIXME: Experimental, kind of useless in current setup:
// - No emissive surface sampling or anything worth running RIS
// - Structures and functions are local to SAMPLING_MODE_RIS. 
// - ReSTIR uses the dedicated structures in restir.slang
struct RisReservoir
{
    float3 wi;   // Selected direction
    float3 f;    // BRDF value
    float pHat;  // Target function value
    float wSum;  // Running sum of resampling weights
    bool hasValidSample;
};

RisReservoir createRisReservoir()
{
    RisReservoir r;
    r.wi = float3(0, 0, 0);
    r.f = float3(0, 0, 0);
    r.pHat = 0.0;
    r.wSum = 0.0;
    r.hasValidSample = false;
    return r;
}

// Assumes the direction is unoccluded, has no idea about emissive surfaces
float risTarget(float3 f, float3 wi)
{
    return luminance(f * getSkyLight(wi));
}

// Stream a candidate
// Sampling PDF is known, so W_Xi is 1/pdf
void risAddCandidate(
    inout RisReservoir r,
    float3 wi,
    float3 f,
    float pHat,
    float misWeight,
    float pdf,
    inout uint rngState)
{
    if (pdf <= 0.0 || pHat <= 0.0)
    {
        return;
    }

    float weight = misWeight * pHat * (1.0 / pdf); // <=> m_i(X_i) * pHat(X_i) * W_Xi
    r.wSum += weight;

    if (rngFloat(rngState) * r.wSum < weight)
    {
        r.wi = wi;
        r.f = f;
        r.pHat = pHat;
        r.hasValidSample = true;
    }
}

// Unbiased contribution weight of selected sample
float risWeight(RisReservoir r)
{
    if (!r.hasValidSample || r.pHat <= 0.0)
    {
        return 0.0;
    }

    return r.wSum / r.pHat;
}

SamplingResult sample_ris(
    float3 normal,
    float3 viewDir,
    float3 worldPos,
    float roughness,
    float metalness,
    float3 albedo,
    int numCandidates,
    inout uint rngState,
    out float3 brdfValueOut)
{
    SamplingResult result;
    result.isValid = false;
    result.pdf = 0.0;
    brdfValueOut = float3(0, 0, 0);

    RisReservoir reservoir = createRisReservoir();

    // All candidates are identically distributed, so m_i = 1/M
    float misWeight = 1.0 / float(max(numCandidates, 1));

    // Generate M candidates
    for (int i = 0; i < numCandidates; i++)
    {
        SamplingResult candidate = sample_mis(
            normal,
            viewDir,
            worldPos,
            roughness,
            metalness,
            albedo,
            rngState
        );

        if (!candidate.isValid || candidate.pdf <= 0.0)
        {
            continue;
        }

        // Evaluate target function
        // NOTE: Assumes the candidate is visible
        float3 f = brdf_ggx(normal, viewDir, candidate.direction, roughness, metalness, albedo);
        float pHat = risTarget(f, candidate.direction);
        risAddCandidate(reservoir, candidate.direction, f, pHat, misWeight, candidate.pdf, rngState);
    }

    float W = risWeight(reservoir);
    if (W <= 0.0)
    {
        return result;
    }

    // The unbiased contribution weight replaces 1/pdf for the selected sample
    result.direction = reservoir.wi;
    result.pdf = 1.0;
    result.isValid = true;
    result.weight = W;
    brdfValueOut = reservoir.f;

    return result;
}

float luminance(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

// GGX importance sampling function using VNDF
float3 sample_ggx(
    float3 normal,
    float3 viewDir,
    float roughness,
    inout uint rngState)
{
    float3 result = float3(0, 0, 0);

    // Clamp roughness to avoid singularities
    roughness = clamp(roughness, 0.04, 1.0);
    float alpha = roughness * roughness;

    // Build tangent space basis
    float3 tangent, bitangent;
    buildOrthonormalBasis(normal, tangent, bitangent);

    // Transform view direction to tangent space
    float3 Ve = worldToTangent(viewDir, normal, tangent, bitangent);

    // Generate random numbers
    float u1 = rngFloat(rngState);
    float u2 = rngFloat(rngState);

    // Sample half-vector in tangent space using VNDF
    float3 H_tangent = sample_ggx_vndf(Ve, alpha, u1, u2);

    // Transform half-vector to world space
    float3 H = tangentToWorld(H_tangent, normal, tangent, bitangent);
    H = normalize(H);

    // Compute reflected direction
    float VdotH = dot(viewDir, H);
    if (VdotH <= 0.0)
    {
        return result;
    }

    float3 L = reflect(-viewDir, H);

    // Reflections below the surface carry no energy and have no valid pdf
    if (dot(normal, L) <= 0.0)
    {
        return result;
    }

    return normalize(L);
}

float3 sample_cone(
    float3 coneAxis,
    float coneHalfAngle,
    inout uint rngState)
{
    // Sample uniformly within the cone's solid angle
    float u1 = rngFloat(rngState);
    float u2 = rngFloat(rngState);

    // (theta)^2 + cos(theta)^2 = 1
    // <=> sin(theta) = sqrt(1 - cos(theta)^2)
    float cosHalfAngle = cos(coneHalfAngle);
    float cosSampledAngle = lerp(1.0, cosHalfAngle, u1);
    float sinSampledAngle = sqrt(max(0.0, 1.0 - cosSampledAngle * cosSampledAngle));

    // Sample random azimuth
    float phi = 2.0 * PI * u2;

    // Create a local coordinate frame around the cone axis
    float3 up = abs(coneAxis.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, coneAxis));
    float3 bitangent = cross(coneAxis, tangent);

    // Construct the sampled direction in the cone
    float3 sampleDir =
        tangent * cos(phi) * sinSampledAngle +
        bitangent * sin(phi) * sinSampledAngle +
        coneAxis * cosSampledAngle;

    return normalize(sampleDir);
}

float sample_cosine_pdf(const float3 normal, const float3 direction)
{
    // Cosine sampling has no support below the hemisphere
    float cosTheta = max(dot(normalize(direction), normalize(normal)), 0.0);
    return cosTheta / PI;
}

float sample_uniform_pdf()
{
    return 1.0 / (2.0 * PI);
}

// TODO: double check formulas
float ggx_vndf_pdf(float3 viewDir, float3 lightDir, float3 halfVector, float3 normal, float roughness)
{
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;

    float NdotH = max(dot(normal, halfVector), 0.0);
    float NdotV = max(dot(normal, viewDir), 0.0);
    float VdotH = max(dot(viewDir, halfVector), 0.0);

    if (NdotV < 1e-7 || VdotH < 1e-7)
    {
        return 0.0;
    }

    float denom = (NdotH * NdotH * (alpha2 - 1.0) + 1.0);
    float D = alpha2 / (PI * denom * denom);

    float G1 = ggx_smith_g1(viewDir, normal, alpha2);

    float pdf_h = (D * G1 * VdotH) / NdotV;

    float pdf_l = pdf_h / (4.0 * VdotH);

    return pdf_l;
}

float cone_sampling_pdf(float3 direction, float3 coneAxis, float coneHalfAngle)
{
    float cosConeHalfAngle = cos(coneHalfAngle);
    float dotProduct = dot(normalize(direction), normalize(coneAxis));

    // Check if direction is within the cone
    if (dotProduct < cosConeHalfAngle) 
    {
        return 0.0;
    }

    // Uniform solid angle sampling PDF
    float solidAngle = 2.0 * PI * (1.0 - cosConeHalfAngle);
    return 1.0 / solidAngle;
}

float3 sample_cosine(const float3 normal, inout uint seed)
{
    // Generate two random numbers
    seed = seed * 1664525u + 1013904223u;
    float r1 = float(seed & 0x00FFFFFFu) / float(0x00FFFFFFu);

    seed = seed * 1664525u + 1013904223u;
    float r2 = float(seed & 0x00FFFFFFu) / float(0x00FFFFFFu);

    // Cosine-weighted sampling
    float cos_theta = sqrt(r1);
    float sin_theta = sqrt(1.0 - r1);     // Polar angle must be biased towards normal
    float phi = 2.0 * 3.14159265359 * r2; // Azimuth must be uniform

    float3 direction = float3(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta
    );

    // Transform to world space
    float3 up = abs(normal.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);

    return normalize(direction.x * tangent + direction.y * bitangent + direction.z * normal);
}

float3 sample_uniform(const float3 normal, inout uint seed)
{
    float3 on_unit_sphere = random_unit_vector(seed);
    if (dot(on_unit_sphere, normal) > 0.0)
    {
        return on_unit_sphere;
    }
    else
    {
        return -on_unit_sphere;
    }
}

// TODO: double check formulas
float ggx_ndf(float3 normal, float3 halfVector, float roughness)
{
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;

    float NdotH = max(dot(normal, halfVector), 0.0);
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (alpha2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return alpha2 / max(denom, 0.0001);
}

// TODO: double check formulas
float ggx_smith_g1(float3 v, float3 normal, float alpha2)
{
    float NdotV = max(dot(normal, v), 0.0);
    float NdotV2 = NdotV * NdotV;
    
    // Avoid division by zero
    if (NdotV2 < 1e-7)
    {
        return 0.0;
    }
    
    float tan2 = (1.0 - NdotV2) / NdotV2;
    return 2.0 / (1.0 + sqrt(1.0 + alpha2 * tan2));
}

// TODO: double check formulas
float ggx_geometry_smith_correlated(float3 normal, float3 viewDir, float3 lightDir, float roughness)
{
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    
    float NdotV = max(dot(normal, viewDir), 0.0);
    float NdotL = max(dot(normal, lightDir), 0.0);
    
    float NdotV2 = NdotV * NdotV;
    float NdotL2 = NdotL * NdotL;
    
    float lambdaV = (-1.0 + sqrt(1.0 + alpha2 * (1.0 - NdotV2) / NdotV2)) * 0.5;
    float lambdaL = (-1.0 + sqrt(1.0 + alpha2 * (1.0 - NdotL2) / NdotL2)) * 0.5;
    
    return 1.0 / (1.0 + lambdaV + lambdaL);
}

// TODO: double check formulas
float3 fresnel_schlick(float3 f0, float3 viewDir, float3 halfVector)
{
    float cosTheta = max(dot(viewDir, halfVector), 0.0);
    return f0 + (float3(1.0, 1.0, 1.0) - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float3 compute_f0(float3 albedo, float metalness)
{
    return lerp(float3(0.04, 0.04, 0.04), albedo, metalness);
}

// TODO: double check formulas
float3 brdf_ggx(
    float3 normal,
    float3 incomingDir,
    float3 lightDir,
    float roughness,
    float metalness,
    float3 albedo)
{
    // Clamp roughness
    // roughness = clamp(roughness, 0.01, 1.0);
    
    float3 h = normalize(incomingDir + lightDir);
    float NdotL = max(dot(normal, lightDir), 0.0);
    float NdotV = max(dot(normal, incomingDir), 0.0);
    
    if (NdotL <= 0.0 || NdotV <= 0.0)
    {
        return float3(0.0, 0.0, 0.0);
    }
    
    // Specular component
    float D = ggx_ndf(normal, h, roughness);
    float G = ggx_geometry_smith_correlated(normal, incomingDir, lightDir, roughness);
    float3 F = fresnel_schlick(compute_f0(albedo, metalness), incomingDir, h);
    
    float3 spec = (D * G * F) / (4.0 * NdotV * NdotL + 0.0001);
    
    // Diffuse component
    float3 kd = (1.0 - metalness) * (1.0 - F);
    float3 diffuse = kd * albedo / PI;
    
    return (diffuse + spec) * NdotL;
}

// TODO: double check formulas
// Based on Eric Heitz's 2018 paper: "Sampling the GGX Distribution of Visible Normals"
float3 sample_ggx_vndf(float3 Ve, float alpha, float u1, float u2)
{
    float3 Vh = normalize(float3(alpha * Ve.x, alpha * Ve.y, Ve.z));
    
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0 ? float3(-Vh.y, Vh.x, 0) / sqrt(lensq) : float3(1, 0, 0);
    float3 T2 = cross(Vh, T1);
    
    float r = sqrt(u1);
    float phi = 2.0 * PI * u2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;
    
    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    
    float3 Ne = normalize(float3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
    return Ne;
}

// #============================================#
// # Utility RNG functions
// #============================================#

uint hash(uint x) 
{
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return x;
}

uint stepRNG(uint rngState) 
{
    return rngState * 1664525u + 1013904223u;
}

uint rngInt(inout uint rngState) 
{
    rngState = stepRNG(rngState);
    return rngState;
}

float rngFloat(inout uint rngState) 
{
    rngState = stepRNG(rngState);
    return float(rngState) * (1.0 / 4294967296.0);
}

// Random float in [0, 1]
float rand(inout uint seed)
{
    // Step and normalize
    stepRNG(seed);
    return float(seed & 0x00FFFFFFu) / float(0x00FFFFFFu);
}

float3 random_unit_vector(inout uint seed)
{
    while (true)
    {
        seed = seed * 1664525u + 1013904223u;
        float x = (float(seed & 0x00FFFFFFu) / float(0x00FFFFFFu)) * 2.0 - 1.0;

        seed = seed * 1664525u + 1013904223u;
        float y = (float(seed & 0x00FFFFFFu) / float(0x00FFFFFFu)) * 2.0 - 1.0;

        seed = seed * 1664525u + 1013904223u;
        float z = (float(seed & 0x00FFFFFFu) / float(0x00FFFFFFu)) * 2.0 - 1.0;

        float3 p = float3(x, y, z);
        float lensq = dot(p, p);

        if (lensq > 1e-10 && lensq <= 1.0)
        {
            return p / sqrt(lensq);
        }
    }
}

// #============================================#
// # Other utility functions
// #============================================#

float linearize_depth(float d,float zNear,float zFar)
{
    return zNear * zFar / (zFar + d * (zNear - zFar));
}

float3 clampNormalToView(float3 n, float3 v)
{
    float NdotV = dot(n, v);
    if (NdotV < 0.0)
    {
        n = normalize(n - 1.01 * NdotV * v);
    }
    return n;
}

// Build orthonormal basis from normal
void buildOrthonormalBasis(float3 n, out float3 tangent, out float3 bitangent)
{
    // Choose a vector that is not parallel to n
    float3 up = abs(n.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
    tangent = normalize(cross(up, n));
    bitangent = cross(n, tangent);
}

float3 tangentToWorld(float3 v, float3 normal, float3 tangent, float3 bitangent)
{
    return v.x * tangent + v.y * bitangent + v.z * normal;
}

float3 worldToTangent(float3 v, float3 normal, float3 tangent, float3 bitangent)
{
    return float3(dot(v, tangent), dot(v, bitangent), dot(v, normal));
}

float3 computeHalfVector(float3 in, float3 out)
{
    return normalize(in + out);
}