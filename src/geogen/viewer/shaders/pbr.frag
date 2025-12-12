#version 330 core

in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vTexCoord;

// Material textures
uniform sampler2D uAlbedoMap;
uniform sampler2D uNormalMap;
uniform sampler2D uRoughnessMap;
uniform sampler2D uAOMap;

// Texture availability flags
uniform bool uHasAlbedoMap;
uniform bool uHasNormalMap;
uniform bool uHasRoughnessMap;
uniform bool uHasAOMap;

// Material properties (fallbacks when no texture)
uniform vec4 uBaseColor;
uniform float uRoughness;
uniform float uMetallic;
uniform float uNormalStrength;
uniform float uAOStrength;

// Camera
uniform vec3 uCameraPos;

// Ambient
uniform vec3 uAmbientColor;

// Lights (support up to 4)
#define MAX_LIGHTS 4
uniform int uLightCount;
uniform int uLightTypes[MAX_LIGHTS];        // 0 = directional, 1 = point
uniform vec3 uLightPositions[MAX_LIGHTS];   // Position for point, direction for directional
uniform vec3 uLightColors[MAX_LIGHTS];
uniform float uLightIntensities[MAX_LIGHTS];

out vec4 FragColor;

const float PI = 3.14159265359;

// PBR helper functions

// Normal Distribution Function (GGX/Trowbridge-Reitz)
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / max(denom, 0.0001);
}

// Geometry function (Schlick-GGX)
float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / max(denom, 0.0001);
}

// Geometry function (Smith's method)
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

// Fresnel equation (Schlick approximation)
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Get normal from normal map (tangent space to world space)
vec3 getNormalFromMap(vec3 worldNormal) {
    if (!uHasNormalMap) {
        return normalize(worldNormal);
    }

    vec3 tangentNormal = texture(uNormalMap, vTexCoord).xyz * 2.0 - 1.0;
    tangentNormal.xy *= uNormalStrength;
    tangentNormal = normalize(tangentNormal);

    // Compute TBN matrix from derivatives
    vec3 Q1 = dFdx(vWorldPos);
    vec3 Q2 = dFdy(vWorldPos);
    vec2 st1 = dFdx(vTexCoord);
    vec2 st2 = dFdy(vTexCoord);

    vec3 N = normalize(worldNormal);
    vec3 T = normalize(Q1 * st2.t - Q2 * st1.t);
    vec3 B = -normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * tangentNormal);
}

void main() {
    // Sample textures or use fallbacks
    vec3 albedo;
    if (uHasAlbedoMap) {
        albedo = texture(uAlbedoMap, vTexCoord).rgb;
    } else {
        albedo = uBaseColor.rgb;
    }
    // Convert from sRGB to linear for correct lighting math
    albedo = pow(albedo, vec3(2.2));

    float roughness;
    if (uHasRoughnessMap) {
        roughness = texture(uRoughnessMap, vTexCoord).r;
    } else {
        roughness = uRoughness;
    }
    roughness = clamp(roughness, 0.04, 1.0); // Prevent division issues

    float ao;
    if (uHasAOMap) {
        ao = mix(1.0, texture(uAOMap, vTexCoord).r, uAOStrength);
    } else {
        ao = 1.0;
    }

    float metallic = uMetallic;

    // Get normal
    vec3 N = getNormalFromMap(vNormal);
    vec3 V = normalize(uCameraPos - vWorldPos);

    // Calculate reflectance at normal incidence
    // For dielectrics use 0.04, for metals use albedo color
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    // Reflectance equation
    vec3 Lo = vec3(0.0);

    for (int i = 0; i < uLightCount && i < MAX_LIGHTS; i++) {
        vec3 L;
        float attenuation = 1.0;

        if (uLightTypes[i] == 0) {
            // Directional light
            L = normalize(-uLightPositions[i]);
        } else {
            // Point light
            vec3 toLight = uLightPositions[i] - vWorldPos;
            float distance = length(toLight);
            L = normalize(toLight);
            attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
        }

        vec3 H = normalize(V + L);
        vec3 radiance = uLightColors[i] * uLightIntensities[i] * attenuation;

        // Cook-Torrance BRDF
        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic; // Metals have no diffuse

        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        vec3 specular = numerator / denominator;

        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;
    }

    // Ambient lighting (simple approximation)
    vec3 ambient = uAmbientColor * albedo * ao;

    vec3 color = ambient + Lo;

    // HDR tonemapping (Reinhard)
    color = color / (color + vec3(1.0));

    // Gamma correction (linear to sRGB)
    color = pow(color, vec3(1.0 / 2.2));

    FragColor = vec4(color, 1.0);
}
