#version 120

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;

attribute vec4 p3d_Vertex;
attribute vec3 p3d_Normal;

varying float tint_value;

void main() {
    vec4 view_pos = p3d_ModelViewMatrix * p3d_Vertex;
    float minVal = 2.2;
    float maxVal = 2.6;
    tint_value = clamp((-view_pos.z-minVal)/(maxVal-minVal), 0.0, 1.0);
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
