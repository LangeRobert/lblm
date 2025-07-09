#version 120

varying float depth;

void main() {
    float normalized_depth = clamp(depth / 2.0, 0.0, 1.0);

    float light_val = 1;
    float dark_val = 0.7;

    vec3 color = mix(vec3(light_val, light_val, light_val), vec3(dark_val, dark_val, dark_val), normalized_depth);
    gl_FragColor = vec4(color, 1.0);
}