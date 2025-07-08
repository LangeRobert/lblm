#version 120

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;

attribute vec4 p3d_Vertex;
attribute vec3 p3d_Normal;

varying float depth;

void main() {
    vec4 view_pos = p3d_ModelViewMatrix * p3d_Vertex;
    depth = -view_pos.z;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
