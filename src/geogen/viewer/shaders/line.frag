#version 330 core

in vec4 vColor;
in vec3 vWorldPos;

uniform vec3 uCameraPos;
uniform float uFadeDistance;  // 0 disables distance fade

out vec4 FragColor;

void main() {
    float alpha = vColor.a;
    if (uFadeDistance > 0.0) {
        float d = length(vWorldPos.xz - uCameraPos.xz);
        alpha *= clamp(1.0 - d / uFadeDistance, 0.0, 1.0);
    }
    FragColor = vec4(vColor.rgb, alpha);
}
