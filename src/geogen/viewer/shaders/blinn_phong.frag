#version 330 core

in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vTexCoord;

// Material
uniform sampler2D uAlbedoMap;
uniform bool uHasTexture;
uniform vec4 uBaseColor;
uniform float uShininess;

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

vec3 calculateLight(int index, vec3 normal, vec3 viewDir, vec3 albedo) {
    vec3 lightDir;
    float attenuation = 1.0;

    if (uLightTypes[index] == 0) {
        // Directional light
        lightDir = normalize(-uLightPositions[index]);
    } else {
        // Point light
        vec3 toLight = uLightPositions[index] - vWorldPos;
        float distance = length(toLight);
        lightDir = normalize(toLight);
        // Quadratic attenuation
        attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
    }

    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);

    // Specular (Blinn-Phong)
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), uShininess * 128.0 + 1.0);

    vec3 lightColor = uLightColors[index] * uLightIntensities[index];

    vec3 diffuse = diff * albedo * lightColor;
    vec3 specular = spec * vec3(0.3) * lightColor;  // White-ish specular

    return (diffuse + specular) * attenuation;
}

void main() {
    // Get albedo color
    vec3 albedo;
    if (uHasTexture) {
        albedo = texture(uAlbedoMap, vTexCoord).rgb;
    } else {
        albedo = uBaseColor.rgb;
    }

    vec3 normal = normalize(vNormal);
    vec3 viewDir = normalize(uCameraPos - vWorldPos);

    // Ambient
    vec3 result = uAmbientColor * albedo;

    // Accumulate light contributions
    for (int i = 0; i < uLightCount && i < MAX_LIGHTS; i++) {
        result += calculateLight(i, normal, viewDir, albedo);
    }

    // Clamp to avoid over-bright
    result = clamp(result, 0.0, 1.0);

    FragColor = vec4(result, 1.0);
}
