// Compositor shader for lamco-wgpu
// Renders textured quads with transform and alpha support

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coord: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
}

struct Uniforms {
    // Transform matrix (2D affine transform as 3x3, padded to 4x4)
    transform: mat4x4<f32>,
    // Source texture region (x, y, width, height) normalized
    src_rect: vec4<f32>,
    // Alpha multiplier
    alpha: f32,
    // Padding
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(1) @binding(0)
var t_texture: texture_2d<f32>;

@group(1) @binding(1)
var s_sampler: sampler;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Apply transform to position
    let pos = uniforms.transform * vec4<f32>(in.position, 0.0, 1.0);
    out.clip_position = pos;

    // Map tex coords to source rectangle
    out.tex_coord = uniforms.src_rect.xy + in.tex_coord * uniforms.src_rect.zw;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(t_texture, s_sampler, in.tex_coord);
    return vec4<f32>(color.rgb, color.a * uniforms.alpha);
}

// Solid color shader for clear and draw_solid operations
struct SolidUniforms {
    transform: mat4x4<f32>,
    color: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> solid_uniforms: SolidUniforms;

@vertex
fn vs_solid(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let pos = solid_uniforms.transform * vec4<f32>(in.position, 0.0, 1.0);
    out.clip_position = pos;
    out.tex_coord = in.tex_coord; // unused but needed for struct
    return out;
}

@fragment
fn fs_solid(in: VertexOutput) -> @location(0) vec4<f32> {
    return solid_uniforms.color;
}
