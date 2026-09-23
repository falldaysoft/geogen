#version 330 core

in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vTexCoord;
in vec4 vLightSpacePos;

// Material textures
uniform sampler2D uAlbedoMap;
uniform sampler2D uNormalMap;
uniform sampler2D uRoughnessMap;
uniform sampler2D uAOMap;
uniform sampler2D uShadowMap;

uniform bool uHasAlbedoMap;
uniform bool uHasNormalMap;
uniform bool uHasRoughnessMap;
uniform bool uHasAOMap;

uniform vec4 uBaseColor;
uniform float uRoughness;
uniform float uMetallic;
uniform float uNormalStrength;
uniform float uAOStrength;
uniform vec2 uUVScale;

// 0 = lit, 1 = clay (no textures), 2 = normals, 3 = UV checker
uniform int uDisplayMode;
uniform vec4 uHighlight;       // rgb colour, a = mix amount
uniform bool uShadowsEnabled;
uniform float uExposure;

uniform vec3 uCameraPos;
uniform vec3 uSkyColor;
uniform vec3 uGroundColor;

#define MAX_LIGHTS 4
uniform int uLightCount;
uniform int uLightTypes[MAX_LIGHTS];        // 0 = directional, 1 = point
uniform vec3 uLightPositions[MAX_LIGHTS];   // position (point) or direction (directional)
uniform vec3 uLightColors[MAX_LIGHTS];
uniform float uLightIntensities[MAX_LIGHTS];
uniform int uShadowLight;                   // index of the shadow-casting light, -1 for none

out vec4 FragColor;

const float PI = 3.14159265359;

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denom * denom, 0.0001);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / max(NdotV * (1.0 - k) + k, 0.0001);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return GeometrySchlickGGX(max(dot(N, V), 0.0), roughness)
         * GeometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 perturbNormal(vec3 N, vec2 texCoord) {
    if (!uHasNormalMap || uDisplayMode == 1) {
        return N;
    }
    vec3 tn = texture(uNormalMap, texCoord).xyz * 2.0 - 1.0;
    tn.xy *= uNormalStrength;
    tn = normalize(tn);

    // Cotangent frame from screen-space derivatives (no tangent attribute needed).
    vec3 dp1 = dFdx(vWorldPos);
    vec3 dp2 = dFdy(vWorldPos);
    vec2 duv1 = dFdx(texCoord);
    vec2 duv2 = dFdy(texCoord);
    vec3 dp2perp = cross(dp2, N);
    vec3 dp1perp = cross(N, dp1);
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    float invmax = inversesqrt(max(dot(T, T), dot(B, B)) + 1e-20);
    mat3 TBN = mat3(T * invmax, B * invmax, N);
    return normalize(TBN * tn);
}

float shadowFactor(vec3 N, vec3 L) {
    if (!uShadowsEnabled) {
        return 1.0;
    }
    vec3 proj = vLightSpacePos.xyz / vLightSpacePos.w * 0.5 + 0.5;
    if (proj.z > 1.0 || proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) {
        return 1.0;
    }
    float bias = max(0.0015 * (1.0 - dot(N, L)), 0.0004);
    vec2 texel = 1.0 / vec2(textureSize(uShadowMap, 0));
    float lit = 0.0;
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            float depth = texture(uShadowMap, proj.xy + vec2(x, y) * texel).r;
            lit += proj.z - bias > depth ? 0.0 : 1.0;
        }
    }
    return lit / 25.0;
}

vec3 checker(vec2 uv) {
    vec2 cell = floor(uv * 4.0);
    float c = mod(cell.x + cell.y, 2.0);
    vec2 f = fract(uv);
    vec3 base = mix(vec3(0.25), vec3(0.85), c);
    // Tint by tile position so stretching and seams are obvious.
    return base * mix(vec3(1.0), vec3(f.x, 0.6, f.y) + 0.4, 0.35);
}

vec3 acesTonemap(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    vec3 geomN = normalize(vNormal);
    if (!gl_FrontFacing) {
        geomN = -geomN;  // show back faces sensibly (open meshes)
    }

    if (uDisplayMode == 2) {
        FragColor = vec4(geomN * 0.5 + 0.5, 1.0);
        return;
    }

    vec2 texCoord = vTexCoord * uUVScale;

    vec3 albedo;
    if (uDisplayMode == 1) {
        albedo = vec3(0.72);
    } else if (uDisplayMode == 3) {
        albedo = checker(vTexCoord);  // metric UVs: one checker tile per metre
    } else if (uHasAlbedoMap) {
        albedo = texture(uAlbedoMap, texCoord).rgb;
    } else {
        albedo = uBaseColor.rgb;
    }
    albedo = pow(albedo, vec3(2.2));

    float roughness = (uHasRoughnessMap && uDisplayMode == 0) ? texture(uRoughnessMap, texCoord).r : uRoughness;
    if (uDisplayMode == 1 || uDisplayMode == 3) {
        roughness = 0.7;
    }
    roughness = clamp(roughness, 0.04, 1.0);
    float metallic = uDisplayMode == 0 ? uMetallic : 0.0;
    float ao = (uHasAOMap && uDisplayMode == 0) ? mix(1.0, texture(uAOMap, texCoord).r, uAOStrength) : 1.0;

    vec3 N = perturbNormal(geomN, texCoord);
    vec3 V = normalize(uCameraPos - vWorldPos);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    vec3 Lo = vec3(0.0);
    for (int i = 0; i < uLightCount && i < MAX_LIGHTS; i++) {
        vec3 L;
        float attenuation = 1.0;
        if (uLightTypes[i] == 0) {
            L = normalize(-uLightPositions[i]);
        } else {
            vec3 toLight = uLightPositions[i] - vWorldPos;
            float d = length(toLight);
            L = toLight / d;
            attenuation = 1.0 / (1.0 + 0.09 * d + 0.032 * d * d);
        }
        if (i == uShadowLight) {
            attenuation *= shadowFactor(geomN, L);
        }

        vec3 H = normalize(V + L);
        vec3 radiance = uLightColors[i] * uLightIntensities[i] * attenuation;
        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
        vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
        vec3 specular = NDF * G * F / (4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001);
        Lo += (kD * albedo / PI + specular) * radiance * max(dot(N, L), 0.0);
    }

    // Hemisphere ambient: sky from above, bounced ground light from below.
    vec3 hemi = mix(uGroundColor, uSkyColor, N.y * 0.5 + 0.5);
    vec3 ambient = hemi * albedo * ao * (1.0 - metallic * 0.5);
    vec3 color = (ambient + Lo) * uExposure;

    color = acesTonemap(color);
    color = pow(color, vec3(1.0 / 2.2));
    color = mix(color, uHighlight.rgb, uHighlight.a);
    FragColor = vec4(color, 1.0);
}
