// Ported from: https://github.com/hackr-sh/ghostty-shaders/blob/main/in-game-crt.glsl
// Original author: sarphiv, License: CC BY-NC-SA 4.0
// Ported to WGSL for wezterm custom_shaders pipeline.
//
// Features: CRT curvature, color fringing, ghosting, aperture grille,
// scanlines, flicker, noise, bloom, vignette. Stylized "in-game terminal" look.
// All struct/binding declarations are auto-prepended by wezterm.

// ---------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------
const CURVE: vec2<f32> = vec2<f32>(13.0, 11.0);
const COLOR_FRINGING_SPREAD: f32 = 0.1;
const GHOSTING_SPREAD: f32 = 0.75;
const GHOSTING_STRENGTH: f32 = 0.1;
const DARKEN_MIX: f32 = 0.4;
const VIGNETTE_SPREAD: f32 = 0.4;
const VIGNETTE_BRIGHTNESS: f32 = 20.0;
const TINT: vec3<f32> = vec3<f32>(0.93, 1.00, 0.96);
const SCAN_LINES_STRENGTH: f32 = 0.20;
const SCAN_LINES_VARIANCE: f32 = 0.35;
const SCAN_LINES_PERIOD: f32 = 4.0;
const APERTURE_GRILLE_STRENGTH: f32 = 0.3;
const APERTURE_GRILLE_PERIOD: f32 = 2.0;
const FLICKER_STRENGTH: f32 = 0.04;
const FLICKER_FREQUENCY: f32 = 15.0;
const NOISE_CONTENT_STRENGTH: f32 = 0.25;
const NOISE_UNIFORM_STRENGTH: f32 = 0.25;
const BLOOM_SPREAD: f32 = 8.0;
const BLOOM_STRENGTH: f32 = 0.004;
const PI: f32 = 3.14159265;
const PHI: f32 = 1.61803398;

// ---------------------------------------------------------------
// Noise
// ---------------------------------------------------------------
fn gold_noise(xy: vec2<f32>, seed: f32) -> f32 {
    return fract(sin(distance(xy * PHI, xy) * seed) * xy.x * xy.y);
}

// ---------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------
@fragment
fn fs_postprocess(in: VertexOutput) -> @location(0) vec4<f32> {
    var uv = in.uv;
    let frag = in.position.xy;

    // CRT curvature
    uv = (uv - 0.5) * 2.0;
    uv = uv * (1.0 + pow(abs(vec2<f32>(uv.y, uv.x)) / CURVE, vec2<f32>(2.0)));
    uv = (uv / 2.0) + 0.5;

    // Color fringing (chromatic aberration)
    let r_uv = vec2<f32>(uv.x + 0.0003 * COLOR_FRINGING_SPREAD, uv.y + 0.0003 * COLOR_FRINGING_SPREAD);
    let g_uv = vec2<f32>(uv.x, uv.y - 0.0006 * COLOR_FRINGING_SPREAD);
    let b_uv = vec2<f32>(uv.x - 0.0006 * COLOR_FRINGING_SPREAD, uv.y);

    var col_r = textureSample(screen_texture, screen_sampler, r_uv).r;
    var col_g = textureSample(screen_texture, screen_sampler, g_uv).g;
    var col_b = textureSample(screen_texture, screen_sampler, b_uv).b;
    let col_a = textureSample(screen_texture, screen_sampler, uv).a;

    // Ghost images
    col_r += 0.04 * GHOSTING_STRENGTH * textureSample(screen_texture, screen_sampler, GHOSTING_SPREAD * vec2<f32>(0.025, -0.027) + uv).r;
    col_g += 0.02 * GHOSTING_STRENGTH * textureSample(screen_texture, screen_sampler, GHOSTING_SPREAD * vec2<f32>(-0.022, -0.020) + uv).g;
    col_b += 0.04 * GHOSTING_STRENGTH * textureSample(screen_texture, screen_sampler, GHOSTING_SPREAD * vec2<f32>(-0.020, -0.018) + uv).b;

    // Bloom — 12 golden spiral taps
    let bstep = BLOOM_SPREAD * vec2<f32>(1.414) / pp.resolution;
    let b00 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 0.169,  0.986) * bstep);
    let b01 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-1.333,  0.472) * bstep);
    let b02 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-0.846, -1.511) * bstep);
    let b03 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 1.554, -1.259) * bstep);
    let b04 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 1.681,  1.474) * bstep);
    let b05 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-1.280,  2.089) * bstep);
    let b06 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-2.458, -0.980) * bstep);
    let b07 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 0.587, -2.767) * bstep);
    let b08 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 2.998,  0.117) * bstep);
    let b09 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 0.414,  3.135) * bstep);
    let b10 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-3.167,  0.984) * bstep);
    let b11 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-1.574, -3.086) * bstep);

    var color = vec3<f32>(col_r, col_g, col_b);

    // Quadratic darken
    color = mix(color, color * color, DARKEN_MIX);

    // Vignette
    color *= VIGNETTE_BRIGHTNESS * pow(clamp(uv.x * uv.y * (1.0 - uv.x) * (1.0 - uv.y), 0.0, 1.0), VIGNETTE_SPREAD);

    // Tint
    color *= TINT;

    // Scanlines
    let scan_mix = SCAN_LINES_VARIANCE / 2.0 * (1.0 + sin(2.0 * PI * uv.y * pp.resolution.y / SCAN_LINES_PERIOD));
    color *= mix(1.0, scan_mix, SCAN_LINES_STRENGTH);

    // Aperture grille
    let ag_phase = (8.0 * (frag.x % APERTURE_GRILLE_PERIOD)) / APERTURE_GRILLE_PERIOD;
    var ag_mask = 0.0;
    if (ag_phase >= 3.0 && ag_phase < 4.0) {
        ag_mask = (8.0 * frag.x % APERTURE_GRILLE_PERIOD) / APERTURE_GRILLE_PERIOD;
    } else if (ag_phase >= 4.0 && ag_phase < 7.0) {
        ag_mask = 1.0;
    } else if (ag_phase >= 7.0) {
        ag_mask = (APERTURE_GRILLE_PERIOD - (8.0 * frag.x % APERTURE_GRILLE_PERIOD)) / APERTURE_GRILLE_PERIOD;
    }
    color *= 1.0 - APERTURE_GRILLE_STRENGTH * ag_mask;

    // Flicker
    color *= 1.0 - FLICKER_STRENGTH / 2.0 * (1.0 + sin(2.0 * PI * FLICKER_FREQUENCY * pp.time));

    // Noise
    let noise = smoothstep(0.4, 0.6, gold_noise(frag, fract(pp.time * 0.001)));
    color *= clamp(noise + 1.0 - NOISE_CONTENT_STRENGTH, 0.0, 1.0);
    color = clamp(color + noise * NOISE_UNIFORM_STRENGTH / 100.0, vec3<f32>(0.0), vec3<f32>(1.0));

    // Bloom accumulation
    let bw = array<f32, 12>(1.0, 0.707, 0.577, 0.5, 0.447, 0.408, 0.378, 0.354, 0.333, 0.316, 0.302, 0.289);
    let blooms = array<vec4<f32>, 12>(b00, b01, b02, b03, b04, b05, b06, b07, b08, b09, b10, b11);
    for (var i = 0; i < 12; i++) {
        let n = blooms[i];
        let lum = 0.299 * n.r + 0.587 * n.g + 0.114 * n.b;
        color += lum * bw[i] * n.rgb * BLOOM_STRENGTH;
    }

    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));

    return vec4<f32>(color, 1.0);
}
