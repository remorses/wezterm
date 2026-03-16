// Ported from: https://github.com/luiscarlospando/crt-shader-with-chromatic-aberration-glow-scanlines-dot-matrix/blob/main/crt-shader-with-chromatic-aberration-glow-scanlines-dot-matrix.glsl
// Original: GLSL for Ghostty, License: MIT
// Ported to WGSL for wezterm custom_shaders pipeline.
//
// Features: Oklab-space bloom, chromatic aberration, scanlines, dot matrix.
// All struct/binding declarations are auto-prepended by wezterm.

// ---------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------
const ABERRATION_FACTOR: f32 = 0.003;

const DIM_CUTOFF: f32 = 0.28;
const BRIGHT_CUTOFF: f32 = 0.65;
const BRIGHT_BOOST: f32 = 1.0;
const DIM_GLOW: f32 = 0.05;
const BRIGHT_GLOW: f32 = 0.10;
const COLOR_GLOW: f32 = 0.3;

const SCANLINE_INTENSITY: f32 = 1.0;
const SCANLINE_DENSITY: f32 = 0.25;

const MASK_INTENSITY: f32 = 0.15;
const MASK_SIZE: f32 = 1.0;

const PI: f32 = 3.14159265;

// ---------------------------------------------------------------
// sRGB <-> linear helpers
// ---------------------------------------------------------------
fn srgb_to_linear(x: f32) -> f32 {
    if (x >= 0.04045) { return pow((x + 0.055) / 1.055, 2.4); }
    return x / 12.92;
}
fn linear_to_srgb(x: f32) -> f32 {
    if (x >= 0.0031308) { return 1.055 * pow(x, 1.0 / 2.4) - 0.055; }
    return 12.92 * x;
}

// ---------------------------------------------------------------
// Oklab conversions
// ---------------------------------------------------------------
fn to_oklab(rgb: vec4<f32>) -> vec4<f32> {
    let c = vec3<f32>(srgb_to_linear(rgb.r), srgb_to_linear(rgb.g), srgb_to_linear(rgb.b));
    let l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    let m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    let s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
    let l_ = pow(max(l, 0.0), 1.0 / 3.0);
    let m_ = pow(max(m, 0.0), 1.0 / 3.0);
    let s_ = pow(max(s, 0.0), 1.0 / 3.0);
    return vec4<f32>(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        rgb.a
    );
}
fn to_rgb(oklab: vec4<f32>) -> vec4<f32> {
    let c = oklab.rgb;
    let l_ = c.r + 0.3963377774 * c.g + 0.2158037573 * c.b;
    let m_ = c.r - 0.1055613458 * c.g - 0.0638541728 * c.b;
    let s_ = c.r - 0.0894841775 * c.g - 1.2914855480 * c.b;
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    let lin = vec3<f32>(
         4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    );
    return vec4<f32>(
        clamp(linear_to_srgb(lin.r), 0.0, 1.0),
        clamp(linear_to_srgb(lin.g), 0.0, 1.0),
        clamp(linear_to_srgb(lin.b), 0.0, 1.0),
        oklab.a
    );
}

// ---------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------
@fragment
fn fs_postprocess(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let frag = in.position.xy;

    // Chromatic aberration — time-varying offset
    let t = pp.time;
    var amount = 1.0;
    amount *= 1.0 + 0.5 * sin(t * 6.0);
    amount *= 1.0 + 0.5 * sin(t * 16.0);
    amount *= 1.0 + 0.5 * sin(t * 19.0);
    amount *= 1.0 + 0.5 * sin(t * 27.0);
    amount *= 27.0;

    let r = textureSample(screen_texture, screen_sampler, vec2<f32>(uv.x - ABERRATION_FACTOR * amount / pp.resolution.x, uv.y)).r;
    let g = textureSample(screen_texture, screen_sampler, uv).g;
    let b = textureSample(screen_texture, screen_sampler, vec2<f32>(uv.x + ABERRATION_FACTOR * amount / pp.resolution.x, uv.y)).b;
    let base_a = textureSample(screen_texture, screen_sampler, uv).a;

    var col = vec4<f32>(r, g, b, base_a);

    // Oklab bloom — sample 12-tap neighbourhood, boost dim pixels near bright ones
    let source = to_oklab(col);
    var dest = source;
    let step = vec2<f32>(1.414) / pp.resolution;

    // Bloom sample offsets (golden spiral, 12 taps for performance)
    let s00 = vec3<f32>( 0.169,  0.986, 1.000);
    let s01 = vec3<f32>(-1.333,  0.472, 0.707);
    let s02 = vec3<f32>(-0.846, -1.511, 0.577);
    let s03 = vec3<f32>( 1.554, -1.259, 0.500);
    let s04 = vec3<f32>( 1.681,  1.474, 0.447);
    let s05 = vec3<f32>(-1.280,  2.089, 0.408);
    let s06 = vec3<f32>(-2.458, -0.980, 0.378);
    let s07 = vec3<f32>( 0.587, -2.767, 0.354);
    let s08 = vec3<f32>( 2.998,  0.117, 0.333);
    let s09 = vec3<f32>( 0.414,  3.135, 0.316);
    let s10 = vec3<f32>(-3.167,  0.984, 0.302);
    let s11 = vec3<f32>(-1.574, -3.086, 0.289);

    // Pre-sample all taps (WGSL requires uniform control flow for textureSample)
    let t00 = to_oklab(textureSample(screen_texture, screen_sampler, uv + s00.xy * step));
    let t01 = to_oklab(textureSample(screen_texture, screen_sampler, uv + s01.xy * step));
    let t02 = to_oklab(textureSample(screen_texture, screen_sampler, uv + s02.xy * step));
    let t03 = to_oklab(textureSample(screen_texture, screen_sampler, uv + s03.xy * step));
    let t04 = to_oklab(textureSample(screen_texture, screen_sampler, uv + s04.xy * step));
    let t05 = to_oklab(textureSample(screen_texture, screen_sampler, uv + s05.xy * step));
    let t06 = to_oklab(textureSample(screen_texture, screen_sampler, uv + s06.xy * step));
    let t07 = to_oklab(textureSample(screen_texture, screen_sampler, uv + s07.xy * step));
    let t08 = to_oklab(textureSample(screen_texture, screen_sampler, uv + s08.xy * step));
    let t09 = to_oklab(textureSample(screen_texture, screen_sampler, uv + s09.xy * step));
    let t10 = to_oklab(textureSample(screen_texture, screen_sampler, uv + s10.xy * step));
    let t11 = to_oklab(textureSample(screen_texture, screen_sampler, uv + s11.xy * step));

    if (source.x > DIM_CUTOFF) {
        dest.x = dest.x * BRIGHT_BOOST;
    } else {
        var glow = vec3<f32>(0.0);
        let taps = array<vec4<f32>, 12>(t00, t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11);
        let weights = array<f32, 12>(s00.z, s01.z, s02.z, s03.z, s04.z, s05.z, s06.z, s07.z, s08.z, s09.z, s10.z, s11.z);
        for (var i = 0; i < 12; i++) {
            let c = taps[i];
            let w = weights[i];
            if (c.x > DIM_CUTOFF) {
                glow.y += c.y * w * COLOR_GLOW;
                glow.z += c.z * w * COLOR_GLOW;
                if (c.x <= BRIGHT_CUTOFF) {
                    glow.x += c.x * w * DIM_GLOW;
                } else {
                    glow.x += c.x * w * BRIGHT_GLOW;
                }
            }
        }
        dest = vec4<f32>(dest.x + glow.x, dest.y + glow.y, dest.z + glow.z, dest.w);
    }

    var color = to_rgb(dest).rgb;

    // Scanlines
    let scanline = abs(sin(frag.y) * SCANLINE_DENSITY * SCANLINE_INTENSITY);
    color = mix(color, vec3<f32>(0.0), scanline);

    // Dot matrix mask
    let mask_pos = frag * MASK_SIZE;
    let mask = 1.0 - MASK_INTENSITY * (0.5 + 0.5 * sin(mask_pos.x * PI) * sin(mask_pos.y * PI));
    color *= mask;

    return vec4<f32>(color, 1.0);
}
