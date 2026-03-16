// Ported from: https://github.com/hackr-sh/ghostty-shaders/blob/main/crt.glsl
// Original: [CRTS] PUBLIC DOMAIN CRT-STYLED SCALAR by Timothy Lottes
// Adapted for Ghostty by qwerasd205. License: UNLICENSE (public domain)
// Ported to WGSL for wezterm custom_shaders pipeline.
//
// Features: Proper CRT phosphor emulation with shadow mask, gaussian
// horizontal blur, sine-wave scanlines, tube warp, sRGB-aware tonemapping.
// All struct/binding declarations are auto-prepended by wezterm.

// ---------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------
const SCALE: f32 = 0.33333333;
const WARP_X: f32 = 0.02;          // 1/(50*aspect) at 16:9 ≈ 0.011
const WARP_Y: f32 = 0.02;          // 1/50
const MIN_VIN: f32 = 0.5;
const INPUT_THIN: f32 = 0.75;
const INPUT_BLUR: f32 = -2.75;
const INPUT_MASK: f32 = 0.65;
const PI2: f32 = 6.28318530717958;

// ---------------------------------------------------------------
// sRGB helpers
// ---------------------------------------------------------------
fn from_srgb1(c: f32) -> f32 {
    if (c <= 0.04045) { return c / 12.92; }
    return pow(c / 1.055 + 0.055 / 1.055, 2.4);
}
fn from_srgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(from_srgb1(c.r), from_srgb1(c.g), from_srgb1(c.b));
}
fn to_srgb1(c: f32) -> f32 {
    if (c < 0.0031308) { return c * 12.92; }
    return 1.055 * pow(c, 0.41666) - 0.055;
}
fn to_srgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(to_srgb1(c.r), to_srgb1(c.g), to_srgb1(c.b));
}

fn fetch(uv: vec2<f32>) -> vec3<f32> {
    return from_srgb(textureSample(screen_texture, screen_sampler, uv).rgb);
}

// ---------------------------------------------------------------
// Shadow mask — diagonal offset pattern (CRTS_MASK_SHADOW)
// ---------------------------------------------------------------
fn crt_mask(pos: vec2<f32>, dark: f32) -> vec3<f32> {
    var p = pos;
    p.x += p.y * 3.0;
    var m = vec3<f32>(dark);
    let x = fract(p.x / 6.0);
    if (x < 0.333) { m.r = 1.0; }
    else if (x < 0.666) { m.g = 1.0; }
    else { m.b = 1.0; }
    return m;
}

// ---------------------------------------------------------------
// Tone mapping
// ---------------------------------------------------------------
fn crt_tone(thin: f32, mask: f32) -> vec2<f32> {
    let mid_out = 0.18 / ((1.5 - thin) * (0.5 * mask + 0.5));
    let p_mid_in = 0.18;
    let x = (-p_mid_in + mid_out) / ((1.0 - p_mid_in) * mid_out);
    let y = (-p_mid_in * mid_out + p_mid_in) / (mid_out * (-p_mid_in) + mid_out);
    return vec2<f32>(x, y);
}

// ---------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------
@fragment
fn fs_postprocess(in: VertexOutput) -> @location(0) vec4<f32> {
    let frag = in.position.xy;
    let res = pp.resolution;
    let half_input = res * SCALE * 0.5;
    let rcp_input = 1.0 / (res * SCALE);
    let two_div_out = 2.0 / res;
    let warp = vec2<f32>(WARP_X, WARP_Y);
    let tone = crt_tone(INPUT_THIN, INPUT_MASK);

    // Tube warp
    var pos = frag * two_div_out - vec2<f32>(1.0);
    pos *= vec2<f32>(1.0 + pos.y * pos.y * warp.x, 1.0 + pos.x * pos.x * warp.y);

    let vin_raw = (1.0 - clamp(pos.x * pos.x, 0.0, 1.0)) * (1.0 - clamp(pos.y * pos.y, 0.0, 1.0));
    let vin = clamp(-((1.0 - vin_raw)) * res.y + res.y, 0.0, 1.0);

    pos = pos * half_input + half_input;

    // Snap to scanline and pixel centres
    let y0 = floor(pos.y - 0.5) + 0.5;
    let x0 = floor(pos.x - 1.5) + 0.5;
    var p = vec2<f32>(x0 * rcp_input.x, y0 * rcp_input.y);

    // Fetch 4×2 neighbourhood (must be before any branches)
    let a0 = fetch(p); p.x += rcp_input.x;
    let a1 = fetch(p); p.x += rcp_input.x;
    let a2 = fetch(p); p.x += rcp_input.x;
    let a3 = fetch(p); p.y += rcp_input.y;
    let b3 = fetch(p); p.x -= rcp_input.x;
    let b2 = fetch(p); p.x -= rcp_input.x;
    let b1 = fetch(p); p.x -= rcp_input.x;
    let b0 = fetch(p);

    // Vertical scanline filter (sine wave)
    let off = pos.y - y0;
    let scan_a = cos(min(0.5, off * INPUT_THIN) * PI2) * 0.5 + 0.5;
    let scan_b = cos(min(0.5, (-off) * INPUT_THIN + INPUT_THIN) * PI2) * 0.5 + 0.5;

    // Horizontal gaussian blur
    let off0 = pos.x - x0;
    let pix0 = exp2(INPUT_BLUR * off0 * off0);
    let pix1 = exp2(INPUT_BLUR * (off0 - 1.0) * (off0 - 1.0));
    let pix2 = exp2(INPUT_BLUR * (off0 - 2.0) * (off0 - 2.0));
    let pix3 = exp2(INPUT_BLUR * (off0 - 3.0) * (off0 - 3.0));
    var pix_t = 1.0 / (pix0 + pix1 + pix2 + pix3);

    pix_t *= max(MIN_VIN, vin);

    let sa = scan_a * pix_t;
    let sb = scan_b * pix_t;

    var color = (a0 * pix0 + a1 * pix1 + a2 * pix2 + a3 * pix3) * sa +
                (b0 * pix0 + b1 * pix1 + b2 * pix2 + b3 * pix3) * sb;

    // Shadow mask
    color *= crt_mask(frag, INPUT_MASK);

    // Tonal curve
    let peak = max(1.0 / (256.0 * 65536.0), max(color.r, max(color.g, color.b)));
    let ratio = color / peak;
    let mapped = peak / (peak * tone.x + tone.y);
    color = ratio * mapped;

    return vec4<f32>(to_srgb(color), 1.0);
}
