// Ported from: https://github.com/thijskok/ghostty-shaders/blob/main/amber-crt.glsl
// Original author: thijskok, License: unspecified
// Ported to WGSL for wezterm custom_shaders pipeline.
//
// Features: Warm amber phosphor tint, inner + outer text glow,
// soft background halo, subtle scanlines and flicker.
// All struct/binding declarations are auto-prepended by wezterm.

// ---------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------
const AMBER: vec3<f32> = vec3<f32>(1.3, 0.75, 0.35);
const INNER_GLOW_RADIUS: f32 = 0.001;
const OUTER_GLOW_RADIUS: f32 = 0.0025;
const BG_GLOW_INTENSITY: f32 = 0.2;
const SCANLINE_SCALE: f32 = 1.5;
const FLICKER_SPEED: f32 = 100.0;

// ---------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------
@fragment
fn fs_postprocess(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;

    let text_alpha = textureSample(screen_texture, screen_sampler, uv).r;

    // Inner glow — tight neighbourhood
    var inner_glow = text_alpha * 1.2;
    inner_glow += textureSample(screen_texture, screen_sampler, uv + vec2<f32>( INNER_GLOW_RADIUS, 0.0)).r * 0.6;
    inner_glow += textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-INNER_GLOW_RADIUS, 0.0)).r * 0.6;
    inner_glow += textureSample(screen_texture, screen_sampler, uv + vec2<f32>(0.0,  INNER_GLOW_RADIUS)).r * 0.6;
    inner_glow += textureSample(screen_texture, screen_sampler, uv + vec2<f32>(0.0, -INNER_GLOW_RADIUS)).r * 0.6;

    // Outer glow — wider spread
    var outer_glow = text_alpha * 0.4;
    outer_glow += textureSample(screen_texture, screen_sampler, uv + vec2<f32>( OUTER_GLOW_RADIUS, 0.0)).r * 0.3;
    outer_glow += textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-OUTER_GLOW_RADIUS, 0.0)).r * 0.3;
    outer_glow += textureSample(screen_texture, screen_sampler, uv + vec2<f32>(0.0,  OUTER_GLOW_RADIUS)).r * 0.3;
    outer_glow += textureSample(screen_texture, screen_sampler, uv + vec2<f32>(0.0, -OUTER_GLOW_RADIUS)).r * 0.3;

    let glow = mix(inner_glow, outer_glow, 0.5);

    // Soft ambient background halo — centered, fades toward edges
    let bg_glow = BG_GLOW_INTENSITY * smoothstep(1.2, 0.2, length(uv - 0.5));
    let bg_color = AMBER * bg_glow * 0.5;

    // Scanlines
    let scanline = 0.98 + 0.02 * sin(uv.y * pp.resolution.y * 3.14159 * SCANLINE_SCALE);

    // Flicker
    let flicker = 0.99 + 0.01 * sin(pp.time * FLICKER_SPEED);

    let color = bg_color + AMBER * glow * scanline * flicker;

    return vec4<f32>(color, 1.0);
}
