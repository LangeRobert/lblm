#version 120

varying float tint_value;

void main() {
    float light_val = 0.2;
    float dark_val = 0.0;
    vec3 color = mix(vec3(light_val, light_val, light_val), vec3(dark_val, dark_val, dark_val), tint_value);
    gl_FragColor = vec4(color, 1.0);
}