// Ported from: https://github.com/hackr-sh/ghostty-shaders/blob/main/tft.glsl
// Credits: hackr-sh, License: unspecified
// Ported to WGSL for wezterm custom_shaders pipeline.
//
// Features: TFT/LCD pixel grid simulation. Draws a fine grid of dark
// lines between "subpixels" to mimic a flat panel display viewed close up.
// All struct/binding declarations are auto-prepended by wezterm.

// ---------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------
const RESOLUTION: f32 = 4.0;   // size of TFT "pixels" in screen pixels
const STRENGTH: f32 = 0.5;     // darkness of the grid lines (0 = none, 1 = full black)

// ---------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------
@fragment
fn fs_postprocess(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    var color = textureSample(screen_texture, screen_sampler, uv).rgb;

    // Grid mask: step() produces 0.0 inside the grid line, 1.0 outside
    let scanline = step(1.2, (uv.y * pp.resolution.y) % RESOLUTION);
    let grille   = step(1.2, (uv.x * pp.resolution.x) % RESOLUTION);
    color *= max(1.0 - STRENGTH, scanline * grille);

    return vec4<f32>(color, 1.0);
}
