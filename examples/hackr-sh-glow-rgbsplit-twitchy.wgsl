// Ported from: https://github.com/hackr-sh/ghostty-shaders/blob/main/glow-rgbsplit-twitchy.glsl
// Credits: kalgynirae (glow), NickWest (chromatic aberration), qwerasd205 (bloom)
// License: unspecified
// Ported to WGSL for wezterm custom_shaders pipeline.
//
// Features: Time-varying chromatic aberration ("twitchy" RGB split)
// combined with Oklab-space perceptual glow/bloom.
// All struct/binding declarations are auto-prepended by wezterm.

// ---------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------
const DIM_CUTOFF: f32 = 0.35;
const BRIGHT_CUTOFF: f32 = 0.65;
const ABERRATION_FACTOR: f32 = 0.05;

// ---------------------------------------------------------------
// sRGB helpers
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
    let l = pow(max(0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b, 0.0), 1.0 / 3.0);
    let m = pow(max(0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b, 0.0), 1.0 / 3.0);
    let s = pow(max(0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b, 0.0), 1.0 / 3.0);
    return vec4<f32>(
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
        rgb.a
    );
}
fn to_rgb(oklab: vec4<f32>) -> vec4<f32> {
    let c = oklab.rgb;
    let l_ = c.r + 0.3963377774 * c.g + 0.2158037573 * c.b;
    let m_ = c.r - 0.1055613458 * c.g - 0.0638541728 * c.b;
    let s_ = c.r - 0.0894841775 * c.g - 1.2914855480 * c.b;
    let lin = vec3<f32>(
         4.0767416621 * l_*l_*l_ - 3.3077115913 * m_*m_*m_ + 0.2309699292 * s_*s_*s_,
        -1.2684380046 * l_*l_*l_ + 2.6097574011 * m_*m_*m_ - 0.3413193965 * s_*s_*s_,
        -0.0041960863 * l_*l_*l_ - 0.7034186147 * m_*m_*m_ + 1.7076147010 * s_*s_*s_
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
    let t = pp.time;

    // Time-varying chromatic aberration amount
    var amount = 1.0;
    amount *= 1.0 + 0.5 * sin(t * 6.0);
    amount *= 1.0 + 0.5 * sin(t * 16.0);
    amount *= 1.0 + 0.5 * sin(t * 19.0);
    amount *= 1.0 + 0.5 * sin(t * 27.0);
    amount *= 27.0;

    let r = textureSample(screen_texture, screen_sampler, vec2<f32>(uv.x - ABERRATION_FACTOR * amount / pp.resolution.x, uv.y)).r;
    let g = textureSample(screen_texture, screen_sampler, uv).g;
    let b = textureSample(screen_texture, screen_sampler, vec2<f32>(uv.x + ABERRATION_FACTOR * amount / pp.resolution.x, uv.y)).b;

    let col = vec4<f32>(r, g, b, 1.0);
    let source = to_oklab(col);
    var dest = source;

    // Bloom taps — 12-tap golden spiral (pre-sampled for uniform control flow)
    let step = vec2<f32>(1.414) / pp.resolution;
    let t00 = to_oklab(textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 0.169,  0.986) * step));
    let t01 = to_oklab(textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-1.333,  0.472) * step));
    let t02 = to_oklab(textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-0.846, -1.511) * step));
    let t03 = to_oklab(textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 1.554, -1.259) * step));
    let t04 = to_oklab(textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 1.681,  1.474) * step));
    let t05 = to_oklab(textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-1.280,  2.089) * step));
    let t06 = to_oklab(textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-2.458, -0.980) * step));
    let t07 = to_oklab(textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 0.587, -2.767) * step));
    let t08 = to_oklab(textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 2.998,  0.117) * step));
    let t09 = to_oklab(textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 0.414,  3.135) * step));
    let t10 = to_oklab(textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-3.167,  0.984) * step));
    let t11 = to_oklab(textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-1.574, -3.086) * step));

    if (source.x > DIM_CUTOFF) {
        dest.x *= 1.2;
    } else {
        var glow = vec3<f32>(0.0);
        let taps = array<vec4<f32>, 12>(t00,t01,t02,t03,t04,t05,t06,t07,t08,t09,t10,t11);
        let weights = array<f32, 12>(1.0, 0.707, 0.577, 0.5, 0.447, 0.408, 0.378, 0.354, 0.333, 0.316, 0.302, 0.289);
        for (var i = 0; i < 12; i++) {
            let c = taps[i];
            let w = weights[i];
            if (c.x > DIM_CUTOFF) {
                glow.y += c.y * w * 0.3;
                glow.z += c.z * w * 0.3;
                if (c.x <= BRIGHT_CUTOFF) {
                    glow.x += c.x * w * 0.05;
                } else {
                    glow.x += c.x * w * 0.10;
                }
            }
        }
        dest = vec4<f32>(dest.x + glow.x, dest.y + glow.y, dest.z + glow.z, dest.w);
    }

    return to_rgb(dest);
}
