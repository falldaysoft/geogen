// Water for nodes with extras.geogen.water (geogen/generators/water.py):
// animated normals from a few travelling sine waves, a fresnel sheen, colour
// and opacity that deepen with the water's depth (from the depth buffer), and
// foam where it meets the shore or anything standing in it.
shader_type spatial;
render_mode blend_mix, depth_draw_always, cull_disabled, specular_schlick_ggx;

uniform vec3 shallow_color : source_color = vec3(0.24, 0.52, 0.56);
uniform vec3 deep_color : source_color = vec3(0.03, 0.14, 0.2);
uniform float depth_scale = 2.0;     // metres of water to reach deep_color
uniform float wave_scale = 1.2;      // metres per wave unit
uniform float wave_speed = 0.8;
uniform float foam = 0.18;           // foam band width (m)
uniform sampler2D depth_tex : hint_depth_texture, filter_linear_mipmap;
uniform sampler2D screen_tex : hint_screen_texture, filter_linear_mipmap;

varying vec3 world_pos;

void vertex() {
    world_pos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}

float height(vec2 p, float t) {
    return sin(p.x * 1.3 + t) * 0.5
        + sin(p.y * 1.7 - t * 1.2) * 0.4
        + sin((p.x + p.y) * 2.9 + t * 1.7) * 0.2
        + sin((p.x - p.y) * 4.1 - t * 2.3) * 0.1;
}

void fragment() {
    float t = TIME * wave_speed;
    vec2 p = world_pos.xz / wave_scale;
    float e = 0.05;
    float dx = height(p + vec2(e, 0.0), t) - height(p - vec2(e, 0.0), t);
    float dz = height(p + vec2(0.0, e), t) - height(p - vec2(0.0, e), t);
    vec3 n_world = normalize(vec3(-dx * 0.35, 1.0, -dz * 0.35));
    NORMAL = normalize((VIEW_MATRIX * vec4(n_world, 0.0)).xyz);

    // How much water lies between this surface and whatever is behind it.
    float raw = texture(depth_tex, SCREEN_UV).r;
    vec4 view = INV_PROJECTION_MATRIX * vec4(SCREEN_UV * 2.0 - 1.0, raw, 1.0);
    view.xyz /= view.w;
    float thickness = max(-view.z - (-VERTEX.z), 0.0);
    float deep = clamp(thickness / depth_scale, 0.0, 1.0);

    vec3 behind = texture(screen_tex, SCREEN_UV + n_world.xz * 0.012).rgb;
    vec3 water = mix(shallow_color, deep_color, deep);
    float fres = pow(1.0 - clamp(dot(NORMAL, VIEW), 0.0, 1.0), 4.0);
    vec3 col = mix(behind * 0.5 + shallow_color * 0.5, water, 0.55 + 0.45 * deep);
    col = mix(col, vec3(0.55, 0.7, 0.8), fres * 0.22);

    float edge = 1.0 - smoothstep(0.0, foam, thickness);
    float ripple = 0.5 + 0.5 * sin(height(p * 3.0, t * 1.5) * 3.0);
    col = mix(col, vec3(0.92, 0.95, 0.96), edge * (0.55 + 0.45 * ripple));

    ALBEDO = col;
    ALPHA = clamp(0.6 + 0.35 * deep + edge * 0.3, 0.0, 1.0);
    ROUGHNESS = 0.06;
    METALLIC = 0.0;
    SPECULAR = 0.6;
}
