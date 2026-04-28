//! Smithay Renderer implementation for wgpu.
//!
//! This module provides a `WgpuRenderer` that implements Smithay's `Renderer` trait,
//! enabling wgpu to be used as a rendering backend for Smithay-based compositors.

use crate::bridge::WgpuBridge;
use crate::error::RenderError;
use crate::pipeline::{
    ortho_projection, quad_transform, SolidUniforms, TextureUniforms, WgpuPipelines,
};
use crate::texture::WgpuTexture;
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::allocator::format::FormatSet;
use smithay::backend::allocator::Buffer as AllocatorBuffer;
use smithay::backend::allocator::Fourcc;
use smithay::backend::renderer::{
    sync::SyncPoint, Color32F, ContextId, DebugFlags, ExportMem, Frame, ImportDma, ImportDmaWl,
    ImportMem, ImportMemWl, Renderer, RendererSuper, Texture, TextureFilter, TextureMapping,
};
use smithay::utils::{Buffer, Physical, Rectangle, Size, Transform};
use smithay::wayland::compositor::SurfaceData;
use smithay::wayland::shm;
use wayland_server::protocol::wl_buffer;

/// Type alias for buffer coordinates (matches Smithay convention)
type BufferCoord = Buffer;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, trace};

/// A Smithay-compatible renderer using wgpu.
pub struct WgpuRenderer {
    /// The bridge to wgpu
    bridge: Arc<WgpuBridge>,
    /// Compositor pipelines
    pipelines: WgpuPipelines,
    /// Context ID for this renderer
    context_id: ContextId<WgpuTexture>,
    /// Texture cache (weak references to avoid keeping textures alive)
    texture_cache: HashMap<DmabufKey, WgpuTexture>,
    /// Current texture filtering modes
    downscale_filter: TextureFilter,
    upscale_filter: TextureFilter,
    /// Debug flags
    debug_flags: DebugFlags,
}

/// Key for texture cache lookup.
#[derive(Clone, PartialEq, Eq, Hash)]
struct DmabufKey {
    /// Unique identifier based on dmabuf properties
    id: u64,
}

impl From<&Dmabuf> for DmabufKey {
    fn from(dmabuf: &Dmabuf) -> Self {
        // TODO: use dmabuf's actual unique ID instead of dimensions
        let size = AllocatorBuffer::size(dmabuf);
        Self {
            id: (size.w as u64) << 32 | (size.h as u64),
        }
    }
}

impl WgpuRenderer {
    pub fn new(bridge: Arc<WgpuBridge>) -> Self {
        let pipelines = WgpuPipelines::new(bridge.device(), wgpu::TextureFormat::Bgra8Unorm);

        Self {
            bridge,
            pipelines,
            context_id: ContextId::new(),
            texture_cache: HashMap::new(),
            downscale_filter: TextureFilter::Linear,
            upscale_filter: TextureFilter::Linear,
            debug_flags: DebugFlags::empty(),
        }
    }

    pub fn bridge(&self) -> &WgpuBridge {
        &self.bridge
    }

    pub fn pipelines(&self) -> &WgpuPipelines {
        &self.pipelines
    }

    fn get_or_import(&mut self, dmabuf: &Dmabuf) -> Result<WgpuTexture, RenderError> {
        let key = DmabufKey::from(dmabuf);

        if let Some(texture) = self.texture_cache.get(&key) {
            trace!("Cache hit for dmabuf");
            return Ok(texture.clone());
        }

        debug!("Cache miss, importing dmabuf");
        // SAFETY: dmabuf is a valid Dmabuf from Smithay with valid fd and metadata.
        let texture = unsafe { self.bridge.import_dmabuf(dmabuf)? };
        self.texture_cache.insert(key, texture.clone());
        Ok(texture)
    }

    fn use_linear_filter(&self, scaling_up: bool) -> bool {
        let filter = if scaling_up {
            self.upscale_filter
        } else {
            self.downscale_filter
        };
        matches!(filter, TextureFilter::Linear)
    }

    fn fourcc_to_wgpu_format(&self, fourcc: Fourcc) -> Result<wgpu::TextureFormat, RenderError> {
        match fourcc {
            Fourcc::Argb8888 | Fourcc::Xrgb8888 => Ok(wgpu::TextureFormat::Bgra8Unorm),
            Fourcc::Abgr8888 | Fourcc::Xbgr8888 => Ok(wgpu::TextureFormat::Rgba8Unorm),
            Fourcc::Argb2101010 | Fourcc::Xrgb2101010 => Ok(wgpu::TextureFormat::Rgb10a2Unorm),
            Fourcc::R8 => Ok(wgpu::TextureFormat::R8Unorm),
            Fourcc::Rg88 | Fourcc::Gr88 => Ok(wgpu::TextureFormat::Rg8Unorm),
            _ => Err(RenderError::InvalidFramebuffer(format!(
                "Unsupported format: {:?}",
                fourcc
            ))),
        }
    }

    fn format_bytes_per_pixel(&self, fourcc: Fourcc) -> u32 {
        match fourcc {
            Fourcc::Argb8888
            | Fourcc::Xrgb8888
            | Fourcc::Abgr8888
            | Fourcc::Xbgr8888
            | Fourcc::Argb2101010
            | Fourcc::Xrgb2101010 => 4,
            Fourcc::Rgb888 | Fourcc::Bgr888 => 3,
            Fourcc::Rg88 | Fourcc::Gr88 => 2,
            Fourcc::R8 => 1,
            _ => 4, // Default to 4 bytes
        }
    }
}

impl std::fmt::Debug for WgpuRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuRenderer")
            .field("context_id", &self.context_id)
            .field("cache_size", &self.texture_cache.len())
            .finish()
    }
}

impl RendererSuper for WgpuRenderer {
    type Error = RenderError;
    type TextureId = WgpuTexture;
    type Framebuffer<'buffer> = WgpuFramebuffer;
    type Frame<'frame, 'buffer>
        = WgpuFrame<'frame>
    where
        'buffer: 'frame,
        Self: 'frame;
}

impl Renderer for WgpuRenderer {
    fn context_id(&self) -> ContextId<Self::TextureId> {
        self.context_id.clone()
    }

    fn downscale_filter(&mut self, filter: TextureFilter) -> Result<(), Self::Error> {
        self.downscale_filter = filter;
        Ok(())
    }

    fn upscale_filter(&mut self, filter: TextureFilter) -> Result<(), Self::Error> {
        self.upscale_filter = filter;
        Ok(())
    }

    fn set_debug_flags(&mut self, flags: DebugFlags) {
        self.debug_flags = flags;
    }

    fn debug_flags(&self) -> DebugFlags {
        self.debug_flags
    }

    fn render<'frame, 'buffer>(
        &'frame mut self,
        framebuffer: &'frame mut Self::Framebuffer<'buffer>,
        output_size: Size<i32, Physical>,
        dst_transform: Transform,
    ) -> Result<Self::Frame<'frame, 'buffer>, Self::Error>
    where
        'buffer: 'frame,
    {
        debug!(
            "Beginning render frame: {:?} transform: {:?}",
            output_size, dst_transform
        );

        let encoder =
            self.bridge
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("smithay-wgpu-frame"),
                });
        let projection = ortho_projection(output_size.w as f32, output_size.h as f32);

        Ok(WgpuFrame {
            renderer: self,
            framebuffer,
            encoder: Some(encoder),
            output_size,
            transform: dst_transform,
            projection,
        })
    }

    fn wait(&mut self, _sync: &SyncPoint) -> Result<(), Self::Error> {
        // Wait for all submitted GPU work to complete.
        //
        // Note: Smithay's SyncPoint is opaque and we can't extract the frame
        // number from it. True sync FD integration would require wgpu to
        // support external semaphores (tracked in wgpu#4067).
        //
        // For now, we use wgpu's poll mechanism which waits for all submitted
        // work to complete. This is correct but potentially less efficient than
        // waiting for a specific sync point.
        self.bridge.wait_idle()?;
        Ok(())
    }

    fn cleanup_texture_cache(&mut self) -> Result<(), Self::Error> {
        self.texture_cache.retain(|_, texture| !texture.is_unique());
        debug!(
            "Texture cache cleaned, {} entries remaining",
            self.texture_cache.len()
        );
        Ok(())
    }
}

impl ImportDma for WgpuRenderer {
    fn dmabuf_formats(&self) -> FormatSet {
        use drm_fourcc::DrmModifier;
        use smithay::backend::allocator::Format;

        self.bridge
            .supported_formats()
            .iter()
            .map(|supported| Format {
                code: supported.fourcc,
                modifier: DrmModifier::Linear,
            })
            .collect()
    }

    fn import_dmabuf(
        &mut self,
        dmabuf: &Dmabuf,
        _damage: Option<&[Rectangle<i32, Buffer>]>,
    ) -> Result<Self::TextureId, Self::Error> {
        self.get_or_import(dmabuf)
    }
}

impl ImportMem for WgpuRenderer {
    fn import_memory(
        &mut self,
        data: &[u8],
        format: Fourcc,
        size: Size<i32, BufferCoord>,
        flipped: bool,
    ) -> Result<Self::TextureId, Self::Error> {
        let width = size.w as u32;
        let height = size.h as u32;
        let wgpu_format = self.fourcc_to_wgpu_format(format)?;
        let bytes_per_pixel = self.format_bytes_per_pixel(format);
        let expected_len = (width * height * bytes_per_pixel) as usize;

        if data.len() < expected_len {
            return Err(RenderError::InvalidFramebuffer(format!(
                "Buffer too small: {} bytes, expected {}",
                data.len(),
                expected_len
            )));
        }

        let texture = self
            .bridge
            .device()
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("imported-memory"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

        self.bridge.queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * bytes_per_pixel),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        Ok(WgpuTexture::new(
            texture,
            wgpu_format,
            width,
            height,
            flipped,
        ))
    }

    fn update_memory(
        &mut self,
        texture: &Self::TextureId,
        data: &[u8],
        region: Rectangle<i32, BufferCoord>,
    ) -> Result<(), Self::Error> {
        let width = region.size.w as u32;
        let height = region.size.h as u32;
        let bytes_per_pixel = match texture.wgpu_format() {
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm => 4,
            wgpu::TextureFormat::Rgb10a2Unorm => 4,
            _ => 4,
        };

        self.bridge.queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: texture.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: region.loc.x as u32,
                    y: region.loc.y as u32,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * bytes_per_pixel),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        Ok(())
    }

    fn mem_formats(&self) -> Box<dyn Iterator<Item = Fourcc>> {
        Box::new(
            vec![
                Fourcc::Argb8888,
                Fourcc::Xrgb8888,
                Fourcc::Abgr8888,
                Fourcc::Xbgr8888,
                Fourcc::R8,
                Fourcc::Rg88,
                Fourcc::Gr88,
            ]
            .into_iter(),
        )
    }
}

impl ImportMemWl for WgpuRenderer {
    fn import_shm_buffer(
        &mut self,
        buffer: &wl_buffer::WlBuffer,
        surface: Option<&SurfaceData>,
        damage: &[Rectangle<i32, BufferCoord>],
    ) -> Result<Self::TextureId, Self::Error> {
        shm::with_buffer_contents(buffer, |ptr, len, data| {
            let format = data.format;
            let width = data.width as u32;
            let height = data.height as u32;
            let stride = data.stride as u32;

            let fourcc = shm_format_to_fourcc(format)?;
            let wgpu_format = self.fourcc_to_wgpu_format(fourcc)?;
            let bpp = self.format_bytes_per_pixel(fourcc);

            let expected_stride = width * bpp;
            // SAFETY: ptr and len come from SHM pool mapping which is valid for the buffer lifetime.
            let data_slice = unsafe { std::slice::from_raw_parts(ptr, len) };

            let texture_data: Vec<u8> = if stride == expected_stride {
                data_slice[..(width * height * bpp) as usize].to_vec()
            } else {
                // Row padding must be stripped
                let mut packed = Vec::with_capacity((width * height * bpp) as usize);
                for row in 0..height {
                    let start = (row * stride) as usize;
                    let end = start + expected_stride as usize;
                    packed.extend_from_slice(&data_slice[start..end]);
                }
                packed
            };

            if let Some(surface_data) = surface {
                if let Some(existing) = self.get_cached_texture(surface_data) {
                    if !damage.is_empty() {
                        for rect in damage {
                            self.update_texture_region(
                                &existing,
                                &texture_data,
                                *rect,
                                width,
                                bpp,
                            )?;
                        }
                        return Ok(existing);
                    }
                }
            }

            let texture = self
                .bridge
                .device()
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("shm-buffer"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu_format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });

            self.bridge.queue().write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &texture_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * bpp),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );

            let wgpu_texture = WgpuTexture::new(texture, wgpu_format, width, height, false);

            if let Some(surface_data) = surface {
                self.cache_texture(surface_data, wgpu_texture.clone());
            }

            Ok(wgpu_texture)
        })
        .map_err(|e| {
            RenderError::Import(crate::error::ImportError::ShmAccess(format!("{:?}", e)))
        })?
    }
}

// ImportDmaWl has a default implementation that uses import_dmabuf
impl ImportDmaWl for WgpuRenderer {}

// Helper functions for SHM buffer handling
fn shm_format_to_fourcc(
    format: wayland_server::protocol::wl_shm::Format,
) -> Result<Fourcc, RenderError> {
    use wayland_server::protocol::wl_shm::Format as ShmFormat;
    match format {
        ShmFormat::Argb8888 => Ok(Fourcc::Argb8888),
        ShmFormat::Xrgb8888 => Ok(Fourcc::Xrgb8888),
        ShmFormat::Abgr8888 => Ok(Fourcc::Abgr8888),
        ShmFormat::Xbgr8888 => Ok(Fourcc::Xbgr8888),
        _ => Err(RenderError::Import(
            crate::error::ImportError::UnsupportedShmFormat,
        )),
    }
}

/// Key for surface-based texture caching (reserved for future use)
#[allow(dead_code)]
#[derive(Clone, PartialEq, Eq, Hash)]
struct SurfaceCacheKey {
    id: usize,
}

impl WgpuRenderer {
    fn get_cached_texture(&self, _surface: &SurfaceData) -> Option<WgpuTexture> {
        // TODO: store texture refs in the surface's data_map
        None
    }

    fn cache_texture(&mut self, _surface: &SurfaceData, _texture: WgpuTexture) {
        // TODO: store texture refs in the surface's data_map
    }

    fn update_texture_region(
        &self,
        texture: &WgpuTexture,
        data: &[u8],
        region: Rectangle<i32, BufferCoord>,
        full_width: u32,
        bpp: u32,
    ) -> Result<(), RenderError> {
        let x = region.loc.x as u32;
        let y = region.loc.y as u32;
        let w = region.size.w as u32;
        let h = region.size.h as u32;

        // Extract the region data
        let mut region_data = Vec::with_capacity((w * h * bpp) as usize);
        for row in y..(y + h) {
            let start = ((row * full_width + x) * bpp) as usize;
            let end = start + (w * bpp) as usize;
            region_data.extend_from_slice(&data[start..end]);
        }

        self.bridge.queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: texture.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            &region_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * bpp),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        Ok(())
    }
}

/// A framebuffer for rendering with wgpu.
#[derive(Debug)]
pub struct WgpuFramebuffer {
    /// The render target texture
    texture: wgpu::Texture,
    /// The texture view
    view: wgpu::TextureView,
    /// Dimensions
    width: u32,
    height: u32,
    /// Texture format
    format: wgpu::TextureFormat,
}

impl WgpuFramebuffer {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        Self::with_format(device, width, height, wgpu::TextureFormat::Bgra8Unorm)
    }

    pub fn with_format(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("wgpu-framebuffer"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            texture,
            view,
            width,
            height,
            format,
        }
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }
}

impl smithay::backend::renderer::Texture for WgpuFramebuffer {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn format(&self) -> Option<smithay::backend::allocator::Fourcc> {
        Some(drm_fourcc::DrmFourcc::Argb8888)
    }
}

/// An in-progress rendering frame.
pub struct WgpuFrame<'a> {
    /// Reference to the renderer
    renderer: &'a mut WgpuRenderer,
    /// Reference to the framebuffer
    framebuffer: &'a WgpuFramebuffer,
    /// Command encoder (taken when frame finishes)
    encoder: Option<wgpu::CommandEncoder>,
    /// Output dimensions (reserved for viewport management)
    #[allow(dead_code)]
    output_size: Size<i32, Physical>,
    /// Output transformation
    transform: Transform,
    /// Projection matrix for this frame
    projection: [[f32; 4]; 4],
}

impl<'a> WgpuFrame<'a> {
    fn begin_render_pass<'pass>(
        encoder: &'pass mut wgpu::CommandEncoder,
        view: &'pass wgpu::TextureView,
        clear_color: Option<wgpu::Color>,
    ) -> wgpu::RenderPass<'pass> {
        let load_op = match clear_color {
            Some(color) => wgpu::LoadOp::Clear(color),
            None => wgpu::LoadOp::Load,
        };

        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("compositor-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: load_op,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }
}

impl<'a> Frame for WgpuFrame<'a> {
    type Error = RenderError;
    type TextureId = WgpuTexture;

    fn context_id(&self) -> ContextId<Self::TextureId> {
        self.renderer.context_id.clone()
    }

    fn clear(
        &mut self,
        color: Color32F,
        at: &[Rectangle<i32, Physical>],
    ) -> Result<(), Self::Error> {
        let encoder = self.encoder.as_mut().ok_or(RenderError::FrameFinished)?;

        let clear_color = wgpu::Color {
            r: color.r() as f64,
            g: color.g() as f64,
            b: color.b() as f64,
            a: color.a() as f64,
        };

        if at.is_empty() {
            let _pass = Self::begin_render_pass(encoder, &self.framebuffer.view, Some(clear_color));
        } else {
            // Scissor rects would be more efficient
            for rect in at {
                self.draw_solid(*rect, &[], color)?;
            }
        }

        debug!("Cleared frame with color: {:?}", color);
        Ok(())
    }

    fn draw_solid(
        &mut self,
        dst: Rectangle<i32, Physical>,
        _damage: &[Rectangle<i32, Physical>],
        color: Color32F,
    ) -> Result<(), Self::Error> {
        let encoder = self.encoder.as_mut().ok_or(RenderError::FrameFinished)?;

        let transform = quad_transform(
            &self.projection,
            dst.loc.x as f32,
            dst.loc.y as f32,
            dst.size.w as f32,
            dst.size.h as f32,
        );

        let uniforms = SolidUniforms {
            transform,
            color: [color.r(), color.g(), color.b(), color.a()],
        };

        let (_, bind_group) = self
            .renderer
            .pipelines
            .create_solid_uniforms(self.renderer.bridge.device(), &uniforms);

        let mut pass = Self::begin_render_pass(encoder, &self.framebuffer.view, None);

        pass.set_pipeline(&self.renderer.pipelines.solid_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, self.renderer.pipelines.quad_vertex_buffer.slice(..));
        pass.set_index_buffer(
            self.renderer.pipelines.quad_index_buffer.slice(..),
            wgpu::IndexFormat::Uint16,
        );
        pass.draw_indexed(0..6, 0, 0..1);

        trace!("Drew solid: {:?} color: {:?}", dst, color);
        Ok(())
    }

    fn render_texture_from_to(
        &mut self,
        texture: &Self::TextureId,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        _damage: &[Rectangle<i32, Physical>],
        _opaque_regions: &[Rectangle<i32, Physical>],
        _src_transform: Transform,
        alpha: f32,
    ) -> Result<(), Self::Error> {
        let encoder = self.encoder.as_mut().ok_or(RenderError::FrameFinished)?;

        let tex_width = texture.width() as f64;
        let tex_height = texture.height() as f64;
        let src_rect = [
            (src.loc.x / tex_width) as f32,
            (src.loc.y / tex_height) as f32,
            (src.size.w / tex_width) as f32,
            (src.size.h / tex_height) as f32,
        ];

        let transform = quad_transform(
            &self.projection,
            dst.loc.x as f32,
            dst.loc.y as f32,
            dst.size.w as f32,
            dst.size.h as f32,
        );

        let uniforms = TextureUniforms {
            transform,
            src_rect,
            alpha,
            _pad: [0.0; 3],
        };

        let (_, uniform_bind_group) = self
            .renderer
            .pipelines
            .create_texture_uniforms(self.renderer.bridge.device(), &uniforms);

        // Determine if we're scaling up or down
        let scaling_up = (dst.size.w as f64) > src.size.w || (dst.size.h as f64) > src.size.h;
        let use_linear = self.renderer.use_linear_filter(scaling_up);

        let texture_bind_group = self.renderer.pipelines.create_texture_bind_group(
            self.renderer.bridge.device(),
            texture.view(),
            use_linear,
        );

        let mut pass = Self::begin_render_pass(encoder, &self.framebuffer.view, None);

        pass.set_pipeline(&self.renderer.pipelines.texture_pipeline);
        pass.set_bind_group(0, &uniform_bind_group, &[]);
        pass.set_bind_group(1, &texture_bind_group, &[]);
        pass.set_vertex_buffer(0, self.renderer.pipelines.quad_vertex_buffer.slice(..));
        pass.set_index_buffer(
            self.renderer.pipelines.quad_index_buffer.slice(..),
            wgpu::IndexFormat::Uint16,
        );
        pass.draw_indexed(0..6, 0, 0..1);

        trace!(
            "Rendered texture: src={:?} dst={:?} alpha={}",
            src,
            dst,
            alpha
        );
        Ok(())
    }

    fn transformation(&self) -> Transform {
        self.transform
    }

    fn finish(mut self) -> Result<SyncPoint, Self::Error> {
        if let Some(encoder) = self.encoder.take() {
            self.renderer.bridge.queue().submit(Some(encoder.finish()));
        }

        let frame_num = self.renderer.bridge.submit_frame();
        debug!("Finished frame {}", frame_num);

        // True sync FD requires external semaphores (wgpu#4067); using poll() for now
        Ok(SyncPoint::signaled())
    }

    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        self.renderer.wait(sync)
    }
}

impl<'a> Drop for WgpuFrame<'a> {
    fn drop(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            self.renderer.bridge.queue().submit(Some(encoder.finish()));
        }
    }
}

/// A texture mapping that holds downloaded pixel data from the GPU.
///
/// This is returned by `ExportMem::copy_framebuffer` and `ExportMem::copy_texture`
/// and contains the actual pixel data that can be used for screenshots.
#[derive(Debug)]
pub struct WgpuTextureMapping {
    /// The pixel data
    data: Vec<u8>,
    /// Width in pixels
    width: u32,
    /// Height in pixels
    height: u32,
    /// Format of the data
    format: Fourcc,
    /// Whether the image is flipped vertically
    flipped: bool,
}

impl WgpuTextureMapping {
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn stride(&self) -> u32 {
        self.width * self.bytes_per_pixel()
    }

    fn bytes_per_pixel(&self) -> u32 {
        match self.format {
            Fourcc::Argb8888
            | Fourcc::Xrgb8888
            | Fourcc::Abgr8888
            | Fourcc::Xbgr8888
            | Fourcc::Argb2101010
            | Fourcc::Xrgb2101010 => 4,
            _ => 4, // Default
        }
    }
}

impl Texture for WgpuTextureMapping {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn format(&self) -> Option<Fourcc> {
        Some(self.format)
    }
}

impl TextureMapping for WgpuTextureMapping {
    fn flipped(&self) -> bool {
        self.flipped
    }
}

impl ExportMem for WgpuRenderer {
    type TextureMapping = WgpuTextureMapping;

    fn copy_framebuffer(
        &mut self,
        target: &Self::Framebuffer<'_>,
        region: Rectangle<i32, BufferCoord>,
        format: Fourcc,
    ) -> Result<Self::TextureMapping, Self::Error> {
        // Validate region
        if region.loc.x < 0 || region.loc.y < 0 {
            return Err(RenderError::InvalidFramebuffer(
                "Region has negative coordinates".into(),
            ));
        }

        let x = region.loc.x as u32;
        let y = region.loc.y as u32;
        let width = region.size.w as u32;
        let height = region.size.h as u32;

        if x + width > target.width || y + height > target.height {
            return Err(RenderError::InvalidFramebuffer(
                "Region exceeds framebuffer bounds".into(),
            ));
        }

        // For now, only support the framebuffer's native format
        // (format conversion would require a render pass)
        let fb_format = wgpu_format_to_fourcc(target.format());
        if format != fb_format {
            return Err(RenderError::InvalidFramebuffer(format!(
                "Format conversion not supported: requested {:?}, framebuffer is {:?}",
                format, fb_format
            )));
        }

        self.copy_texture_internal(target.texture(), x, y, width, height, format, false)
    }

    fn copy_texture(
        &mut self,
        texture: &Self::TextureId,
        region: Rectangle<i32, BufferCoord>,
        format: Fourcc,
    ) -> Result<Self::TextureMapping, Self::Error> {
        // Validate region
        if region.loc.x < 0 || region.loc.y < 0 {
            return Err(RenderError::InvalidFramebuffer(
                "Region has negative coordinates".into(),
            ));
        }

        let x = region.loc.x as u32;
        let y = region.loc.y as u32;
        let width = region.size.w as u32;
        let height = region.size.h as u32;

        if x + width > texture.width() || y + height > texture.height() {
            return Err(RenderError::InvalidFramebuffer(
                "Region exceeds texture bounds".into(),
            ));
        }

        // Check format compatibility
        let tex_format = texture
            .format()
            .ok_or_else(|| RenderError::InvalidFramebuffer("Texture has no format".into()))?;

        if format != tex_format {
            return Err(RenderError::InvalidFramebuffer(format!(
                "Format conversion not supported: requested {:?}, texture is {:?}",
                format, tex_format
            )));
        }

        self.copy_texture_internal(
            texture.texture(),
            x,
            y,
            width,
            height,
            format,
            texture.y_inverted(),
        )
    }

    fn can_read_texture(&mut self, texture: &Self::TextureId) -> Result<bool, Self::Error> {
        // We can read any texture that has COPY_SRC usage
        // Our imported textures are created with this usage
        Ok(texture
            .texture()
            .usage()
            .contains(wgpu::TextureUsages::COPY_SRC))
    }

    fn map_texture<'a>(
        &mut self,
        texture_mapping: &'a Self::TextureMapping,
    ) -> Result<&'a [u8], Self::Error> {
        // The texture mapping already contains the pixel data in memory,
        // so we just return a reference to it.
        Ok(texture_mapping.data())
    }
}

impl WgpuRenderer {
    #[allow(clippy::too_many_arguments)]
    fn copy_texture_internal(
        &self,
        texture: &wgpu::Texture,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        format: Fourcc,
        flipped: bool,
    ) -> Result<WgpuTextureMapping, RenderError> {
        let bytes_per_pixel = fourcc_bytes_per_pixel(format);

        // wgpu requires COPY_BYTES_PER_ROW_ALIGNMENT (256) for buffer rows
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let buffer_size = (padded_bytes_per_row * height) as u64;

        debug!(
            "Copying texture region: {}x{} at ({}, {}), format={:?}, buffer_size={}",
            width, height, x, y, format, buffer_size
        );

        let staging_buffer = self.bridge.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("texture-readback-staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.bridge
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("texture-readback"),
                });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        self.bridge.queue().submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        let _ = self
            .bridge
            .device()
            .poll(wgpu::PollType::wait_indefinitely());

        rx.recv()
            .map_err(|_| RenderError::Submit("Buffer mapping channel closed".into()))?
            .map_err(|e| RenderError::Submit(format!("buffer mapping failed: {:?}", e)))?;

        let mapped = buffer_slice.get_mapped_range();

        let data = if padded_bytes_per_row == unpadded_bytes_per_row {
            mapped.to_vec()
        } else {
            let mut unpacked = Vec::with_capacity((unpadded_bytes_per_row * height) as usize);
            for row in 0..height {
                let start = (row * padded_bytes_per_row) as usize;
                let end = start + unpadded_bytes_per_row as usize;
                unpacked.extend_from_slice(&mapped[start..end]);
            }
            unpacked
        };

        drop(mapped);
        staging_buffer.unmap();

        debug!("Successfully read {} bytes from texture", data.len());

        Ok(WgpuTextureMapping {
            data,
            width,
            height,
            format,
            flipped,
        })
    }
}

/// Convert wgpu TextureFormat to DRM Fourcc.
fn wgpu_format_to_fourcc(format: wgpu::TextureFormat) -> Fourcc {
    match format {
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => Fourcc::Argb8888,
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => Fourcc::Abgr8888,
        wgpu::TextureFormat::Rgb10a2Unorm => Fourcc::Argb2101010,
        wgpu::TextureFormat::R8Unorm => Fourcc::R8,
        wgpu::TextureFormat::Rg8Unorm => Fourcc::Rg88,
        _ => Fourcc::Argb8888, // Default fallback
    }
}

/// Get bytes per pixel for a DRM fourcc format.
fn fourcc_bytes_per_pixel(fourcc: Fourcc) -> u32 {
    match fourcc {
        Fourcc::Argb8888
        | Fourcc::Xrgb8888
        | Fourcc::Abgr8888
        | Fourcc::Xbgr8888
        | Fourcc::Argb2101010
        | Fourcc::Xrgb2101010 => 4,
        Fourcc::Rgb888 | Fourcc::Bgr888 => 3,
        Fourcc::Rg88 | Fourcc::Gr88 => 2,
        Fourcc::R8 => 1,
        _ => 4, // Default to 4 bytes
    }
}
