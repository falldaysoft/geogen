#version 330 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec4 aColor;

uniform mat4 uView;
uniform mat4 uProjection;
uniform bool uUseVertexColor;
uniform vec4 uColor;

out vec4 vColor;
out vec3 vWorldPos;

void main() {
    vColor = uUseVertexColor ? aColor : uColor;
    vWorldPos = aPosition;
    gl_Position = uProjection * uView * vec4(aPosition, 1.0);
}
