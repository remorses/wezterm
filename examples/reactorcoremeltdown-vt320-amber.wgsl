// Ported from: https://gist.github.com/reactorcoremeltdown/2ad67cc107f803e5169f3a64f0340cd8
// Original source: https://www.shadertoy.com/view/WsVSzV (CC BY-NC-SA 3.0)
// Modified by reactorcoremeltdown for VT320-style amber phosphor look.
// Ported to WGSL for wezterm custom_shaders pipeline.
//
// Features: Warm amber tint, scanlines, optional CRT curvature.
// Minimal and clean — good for daily use.
// All struct/binding declarations are auto-prepended by wezterm.

// ---------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------
// Set WARP > 0 for barrel curvature (e.g. 0.25). 0 = flat.
const WARP: f32 = 0.0;
const SCAN: f32 = 0.50;
const TINT: vec3<f32> = vec3<f32>(1.0, 0.74, 0.2);

// ---------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------
@fragment
fn fs_postprocess(in: VertexOutput) -> @location(0) vec4<f32> {
    var uv = in.uv;
    let frag = in.position.xy;

    // Squared distance from center for warp
    let dc = abs(0.5 - uv) * abs(0.5 - uv);

    // Barrel distortion
    uv.x = (uv.x - 0.5) * (1.0 + dc.y * 0.3 * WARP) + 0.5;
    uv.y = (uv.y - 0.5) * (1.0 + dc.x * 0.4 * WARP) + 0.5;

    // Scanline mask
    let apply = abs(sin(frag.y) * 0.5 * SCAN);

    // Sample (must be before any early return for uniform control flow)
    let color = textureSample(screen_texture, screen_sampler, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0))).rgb;

    // Out of bounds → black
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let result = mix(color * TINT, vec3<f32>(0.0), apply);
    return vec4<f32>(result, 1.0);
}
