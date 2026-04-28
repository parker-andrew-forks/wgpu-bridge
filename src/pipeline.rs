//! Render pipeline management for the compositor.
//!
//! This module provides the GPU pipelines needed for compositing:
//! - Textured quad pipeline (for rendering window contents)
//! - Solid color pipeline (for backgrounds and solid rectangles)

use wgpu::util::DeviceExt;

/// Vertex data for a quad.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 2],
    pub tex_coord: [f32; 2],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x2,
        1 => Float32x2,
    ];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// Unit quad vertices (0,0 to 1,1).
pub const QUAD_VERTICES: &[Vertex] = &[
    Vertex {
        position: [0.0, 0.0],
        tex_coord: [0.0, 0.0],
    },
    Vertex {
        position: [1.0, 0.0],
        tex_coord: [1.0, 0.0],
    },
    Vertex {
        position: [1.0, 1.0],
        tex_coord: [1.0, 1.0],
    },
    Vertex {
        position: [0.0, 1.0],
        tex_coord: [0.0, 1.0],
    },
];

/// Quad indices for two triangles.
pub const QUAD_INDICES: &[u16] = &[0, 1, 2, 0, 2, 3];

/// Uniform data for textured rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TextureUniforms {
    /// 4x4 transform matrix
    pub transform: [[f32; 4]; 4],
    /// Source rectangle (x, y, w, h) in normalized coordinates
    pub src_rect: [f32; 4],
    /// Alpha multiplier
    pub alpha: f32,
    /// Padding to 16-byte alignment
    pub _pad: [f32; 3],
}

impl Default for TextureUniforms {
    fn default() -> Self {
        Self {
            transform: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            src_rect: [0.0, 0.0, 1.0, 1.0],
            alpha: 1.0,
            _pad: [0.0; 3],
        }
    }
}

/// Uniform data for solid color rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SolidUniforms {
    /// 4x4 transform matrix
    pub transform: [[f32; 4]; 4],
    /// RGBA color
    pub color: [f32; 4],
}

impl Default for SolidUniforms {
    fn default() -> Self {
        Self {
            transform: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            color: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

/// Compositor render pipelines.
pub struct WgpuPipelines {
    /// Pipeline for rendering textured quads
    pub texture_pipeline: wgpu::RenderPipeline,
    /// Pipeline for rendering solid colors
    pub solid_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for uniforms
    pub uniform_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout for textures
    pub texture_bind_group_layout: wgpu::BindGroupLayout,
    /// Solid uniform bind group layout
    pub solid_uniform_bind_group_layout: wgpu::BindGroupLayout,
    /// Vertex buffer for quad
    pub quad_vertex_buffer: wgpu::Buffer,
    /// Index buffer for quad
    pub quad_index_buffer: wgpu::Buffer,
    /// Default sampler (linear filtering)
    pub linear_sampler: wgpu::Sampler,
    /// Nearest-neighbor sampler
    pub nearest_sampler: wgpu::Sampler,
}

impl WgpuPipelines {
    /// Create new compositor pipelines.
    pub fn new(device: &wgpu::Device, output_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compositor-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/compositor.wgsl").into()),
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("texture-uniform-layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("texture-bind-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let solid_uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("solid-uniform-layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let texture_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("texture-pipeline-layout"),
                bind_group_layouts: &[&uniform_bind_group_layout, &texture_bind_group_layout],
                immediate_size: 0,
            });

        let solid_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("solid-pipeline-layout"),
                bind_group_layouts: &[&solid_uniform_bind_group_layout],
                immediate_size: 0,
            });

        let texture_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("texture-pipeline"),
            layout: Some(&texture_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let solid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("solid-pipeline"),
            layout: Some(&solid_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_solid"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_solid"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let quad_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad-vertices"),
            contents: bytemuck::cast_slice(QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let quad_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad-indices"),
            contents: bytemuck::cast_slice(QUAD_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("nearest-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Self {
            texture_pipeline,
            solid_pipeline,
            uniform_bind_group_layout,
            texture_bind_group_layout,
            solid_uniform_bind_group_layout,
            quad_vertex_buffer,
            quad_index_buffer,
            linear_sampler,
            nearest_sampler,
        }
    }

    /// Create a uniform buffer and bind group for textured rendering.
    pub fn create_texture_uniforms(
        &self,
        device: &wgpu::Device,
        uniforms: &TextureUniforms,
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("texture-uniforms"),
            contents: bytemuck::cast_slice(&[*uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("texture-uniform-bind-group"),
            layout: &self.uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        (buffer, bind_group)
    }

    /// Create a bind group for a texture.
    pub fn create_texture_bind_group(
        &self,
        device: &wgpu::Device,
        texture_view: &wgpu::TextureView,
        use_linear_filter: bool,
    ) -> wgpu::BindGroup {
        let sampler = if use_linear_filter {
            &self.linear_sampler
        } else {
            &self.nearest_sampler
        };

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("texture-bind-group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }

    /// Create a uniform buffer and bind group for solid color rendering.
    pub fn create_solid_uniforms(
        &self,
        device: &wgpu::Device,
        uniforms: &SolidUniforms,
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("solid-uniforms"),
            contents: bytemuck::cast_slice(&[*uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("solid-uniform-bind-group"),
            layout: &self.solid_uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        (buffer, bind_group)
    }
}

/// Compute an orthographic projection matrix for 2D rendering.
///
/// Maps (0,0)-(width,height) to clip space (-1,-1)-(1,1).
pub fn ortho_projection(width: f32, height: f32) -> [[f32; 4]; 4] {
    [
        [2.0 / width, 0.0, 0.0, 0.0],
        [0.0, -2.0 / height, 0.0, 0.0], // Flip Y for screen coordinates
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0, 1.0],
    ]
}

/// Compute a transform matrix for positioning a quad.
///
/// Takes the projection matrix and destination rectangle.
pub fn quad_transform(
    projection: &[[f32; 4]; 4],
    dst_x: f32,
    dst_y: f32,
    dst_w: f32,
    dst_h: f32,
) -> [[f32; 4]; 4] {
    // Scale and translate the unit quad to the destination rectangle
    // Then apply projection
    let scale_translate = [
        [dst_w, 0.0, 0.0, 0.0],
        [0.0, dst_h, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [dst_x, dst_y, 0.0, 1.0],
    ];

    mat4_multiply(projection, &scale_translate)
}

/// Multiply two 4x4 matrices.
fn mat4_multiply(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                result[i][j] += a[k][j] * b[i][k];
            }
        }
    }
    result
}
