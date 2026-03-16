// Ported from: https://github.com/hackr-sh/ghostty-shaders/blob/main/retro-terminal.glsl
// Original source: https://www.shadertoy.com/view/WsVSzV (CC BY-NC-SA 3.0)
// Ported to WGSL for wezterm custom_shaders pipeline.
//
// Features: CRT barrel curvature, scanlines, teal/green phosphor tint.
// Classic retro terminal look — simple and lightweight.
// All struct/binding declarations are auto-prepended by wezterm.

// ---------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------
const WARP: f32 = 0.25;
const SCAN: f32 = 0.50;
const TINT: vec3<f32> = vec3<f32>(0.0, 0.8, 0.6);

// ---------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------
@fragment
fn fs_postprocess(in: VertexOutput) -> @location(0) vec4<f32> {
    var uv = in.uv;
    let frag = in.position.xy;

    let dc = abs(0.5 - uv) * abs(0.5 - uv);

    uv.x = (uv.x - 0.5) * (1.0 + dc.y * 0.3 * WARP) + 0.5;
    uv.y = (uv.y - 0.5) * (1.0 + dc.x * 0.4 * WARP) + 0.5;

    let apply = abs(sin(frag.y) * 0.5 * SCAN);

    // Sample before branch (WGSL uniform control flow requirement)
    let color = textureSample(screen_texture, screen_sampler, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0))).rgb;

    if (uv.y > 1.0 || uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let result = mix(color * TINT, vec3<f32>(0.0), apply);
    return vec4<f32>(result, 1.0);
}
