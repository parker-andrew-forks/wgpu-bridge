//! Texture types for the lamco-wgpu.

use smithay::backend::renderer::Texture as SmithayTexture;
use std::sync::Arc;

/// A wgpu texture that can be used with Smithay's renderer interface.
#[derive(Clone)]
pub struct WgpuTexture {
    /// The underlying wgpu texture (wrapped in Arc for cloning)
    inner: Arc<WgpuTextureInner>,
}

struct WgpuTextureInner {
    /// The wgpu texture handle
    #[allow(dead_code)]
    texture: wgpu::Texture,
    /// The texture view for sampling
    view: wgpu::TextureView,
    /// Texture format
    format: wgpu::TextureFormat,
    /// Texture dimensions
    width: u32,
    height: u32,
    /// Whether this texture was imported from external memory (dmabuf)
    external: bool,
    /// Whether the texture is y-inverted (flipped vertically)
    y_inverted: bool,
}

impl WgpuTexture {
    /// Create a new WgpuTexture from a wgpu texture.
    ///
    /// # Arguments
    ///
    /// * `texture` - The wgpu texture handle
    /// * `format` - The texture format
    /// * `width` - Texture width in pixels
    /// * `height` - Texture height in pixels
    /// * `external` - Whether this texture was imported from external memory
    pub fn new(
        texture: wgpu::Texture,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        external: bool,
    ) -> Self {
        // External textures (dmabufs) are typically y-inverted
        Self::new_with_flip(texture, format, width, height, external, external)
    }

    /// Create a new WgpuTexture with explicit y-inversion control.
    ///
    /// # Arguments
    ///
    /// * `texture` - The wgpu texture handle
    /// * `format` - The texture format
    /// * `width` - Texture width in pixels
    /// * `height` - Texture height in pixels
    /// * `external` - Whether this texture was imported from external memory
    /// * `y_inverted` - Whether the texture is vertically flipped
    pub fn new_with_flip(
        texture: wgpu::Texture,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        external: bool,
        y_inverted: bool,
    ) -> Self {
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("wgpu-texture-view"),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            usage: None,
        });

        Self {
            inner: Arc::new(WgpuTextureInner {
                texture,
                view,
                format,
                width,
                height,
                external,
                y_inverted,
            }),
        }
    }

    /// Create a WgpuTexture with default settings (non-external, not flipped).
    pub fn from_wgpu(
        texture: wgpu::Texture,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        Self::new_with_flip(texture, format, width, height, false, false)
    }

    /// Get the wgpu texture view for rendering.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.inner.view
    }

    /// Get the underlying wgpu texture.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.inner.texture
    }

    /// Get the wgpu texture format.
    pub fn wgpu_format(&self) -> wgpu::TextureFormat {
        self.inner.format
    }

    /// Check if this texture was imported from external memory (dmabuf).
    pub fn is_external(&self) -> bool {
        self.inner.external
    }

    /// Check if this texture is y-inverted (flipped vertically).
    ///
    /// This is typically true for textures imported from dmabufs due to
    /// different coordinate conventions between OpenGL and Vulkan/wgpu.
    pub fn y_inverted(&self) -> bool {
        self.inner.y_inverted
    }

    /// Check if this is the only reference to the texture.
    ///
    /// Useful for cache cleanup - if unique, the texture can be safely removed.
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }
}

impl SmithayTexture for WgpuTexture {
    fn width(&self) -> u32 {
        self.inner.width
    }

    fn height(&self) -> u32 {
        self.inner.height
    }

    fn format(&self) -> Option<smithay::backend::allocator::Fourcc> {
        // Map wgpu format back to DRM fourcc
        match self.inner.format {
            wgpu::TextureFormat::Bgra8Unorm => Some(drm_fourcc::DrmFourcc::Argb8888),
            wgpu::TextureFormat::Bgra8UnormSrgb => Some(drm_fourcc::DrmFourcc::Argb8888),
            wgpu::TextureFormat::Rgba8Unorm => Some(drm_fourcc::DrmFourcc::Abgr8888),
            wgpu::TextureFormat::Rgba8UnormSrgb => Some(drm_fourcc::DrmFourcc::Abgr8888),
            wgpu::TextureFormat::Rgb10a2Unorm => Some(drm_fourcc::DrmFourcc::Argb2101010),
            wgpu::TextureFormat::R8Unorm => Some(drm_fourcc::DrmFourcc::R8),
            wgpu::TextureFormat::Rg8Unorm => Some(drm_fourcc::DrmFourcc::Rg88),
            _ => None,
        }
    }
}

impl std::fmt::Debug for WgpuTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuTexture")
            .field("width", &self.inner.width)
            .field("height", &self.inner.height)
            .field("format", &self.inner.format)
            .field("external", &self.inner.external)
            .field("refs", &Arc::strong_count(&self.inner))
            .finish()
    }
}
