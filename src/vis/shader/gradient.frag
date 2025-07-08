#version 120

varying float depth;

void main() {
    float normalized_depth = clamp(depth / 2.0, 0.0, 1.0);

    vec3 light_red = vec3(8.0, 0.3, 0.3);
    vec3 dark_red  = vec3(0.1, 0.0, 0.0);

    vec3 color = mix(light_red, dark_red, normalized_depth);
    gl_FragColor = vec4(color, 1.0);
}