// Ported from: https://github.com/hackr-sh/ghostty-shaders/blob/main/bloom.glsl
// Original: https://gist.github.com/qwerasd205/c3da6c610c8ffe17d6d2d3cc7068f17f
// Credits: qwerasd205, License: unspecified
// Ported to WGSL for wezterm custom_shaders pipeline.
//
// Features: Pure bloom/glow using 24-tap golden spiral sampling.
// Bright pixels bleed light into their neighbourhood.
// All struct/binding declarations are auto-prepended by wezterm.

// ---------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------
const BLOOM_THRESHOLD: f32 = 0.2;
const BLOOM_STRENGTH: f32 = 0.2;

// ---------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------
fn lum(c: vec4<f32>) -> f32 {
    return 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
}

@fragment
fn fs_postprocess(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let step = vec2<f32>(1.414) / pp.resolution;

    var color = textureSample(screen_texture, screen_sampler, uv);

    // 24 golden spiral taps — pre-sample all for uniform control flow
    let t00 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 0.169,  0.986) * step);
    let t01 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-1.333,  0.472) * step);
    let t02 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-0.846, -1.511) * step);
    let t03 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 1.554, -1.259) * step);
    let t04 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 1.681,  1.474) * step);
    let t05 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-1.280,  2.089) * step);
    let t06 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-2.458, -0.980) * step);
    let t07 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 0.587, -2.767) * step);
    let t08 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 2.998,  0.117) * step);
    let t09 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 0.414,  3.135) * step);
    let t10 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-3.167,  0.984) * step);
    let t11 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-1.574, -3.086) * step);
    let t12 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 2.888, -2.158) * step);
    let t13 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 2.715,  2.575) * step);
    let t14 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-2.150,  3.221) * step);
    let t15 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-3.655, -1.625) * step);
    let t16 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 1.013, -3.997) * step);
    let t17 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 4.230,  0.331) * step);
    let t18 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 0.401,  4.340) * step);
    let t19 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-4.319,  1.160) * step);
    let t20 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-1.921, -4.161) * step);
    let t21 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 3.864, -2.659) * step);
    let t22 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>( 3.349,  3.433) * step);
    let t23 = textureSample(screen_texture, screen_sampler, uv + vec2<f32>(-2.877,  3.965) * step);

    let taps = array<vec4<f32>, 24>(t00,t01,t02,t03,t04,t05,t06,t07,t08,t09,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23);
    let weights = array<f32, 24>(1.0, 0.707, 0.577, 0.5, 0.447, 0.408, 0.378, 0.354, 0.333, 0.316, 0.302, 0.289, 0.277, 0.267, 0.258, 0.25, 0.243, 0.236, 0.229, 0.224, 0.218, 0.213, 0.209, 0.204);

    for (var i = 0; i < 24; i++) {
        let c = taps[i];
        let l = lum(c);
        if (l > BLOOM_THRESHOLD) {
            color += l * weights[i] * c * BLOOM_STRENGTH;
        }
    }

    return color;
}
