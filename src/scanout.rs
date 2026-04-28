//! Output scanout functionality for exporting rendered frames.
//!
//! This module provides the ability to export rendered framebuffers as
//! DMA-BUFs for display scanout. This completes the rendering pipeline:
//!
//! ```text
//! Client buffers (dmabuf) → [Import] → wgpu Render → [Export] → Display (dmabuf)
//! ```
//!
//! # Vulkan Extensions Used
//!
//! - `VK_KHR_external_memory` - Base external memory support
//! - `VK_KHR_external_memory_fd` - Export memory as FDs
//! - `VK_EXT_external_memory_dma_buf` - DMA-BUF specific handling
//! - `VK_EXT_image_drm_format_modifier` - DRM format modifiers

use ash::vk;
use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::Arc;
use tracing::{debug, trace};

use crate::error::BridgeError;

/// A render target that can be exported as a dmabuf.
///
/// This is used as the compositor's output buffer - wgpu renders into it,
/// then it's exported as a dmabuf for display.
///
/// # Resource Ownership
///
/// - `vk_image`: Owned by us, must be destroyed in Drop
/// - `vk_memory`: Owned by wgpu-hal (via `TextureMemory::Dedicated`), freed when texture drops
/// - `texture`: Owned by wgpu, drops first which triggers memory cleanup
pub struct RenderTarget {
    /// The wgpu texture for rendering
    texture: wgpu::Texture,
    /// The texture view for rendering
    view: wgpu::TextureView,
    /// Raw Vulkan image handle - we own this and must destroy it
    vk_image: vk::Image,
    /// Vulkan device memory (exportable) - wgpu-hal owns this via TextureMemory::Dedicated
    /// We store it only for export_dmabuf()
    vk_memory: vk::DeviceMemory,
    /// Dimensions
    width: u32,
    height: u32,
    /// DRM format
    drm_format: drm_fourcc::DrmFourcc,
    /// Format modifier
    modifier: u64,
    /// Stride (bytes per row)
    stride: u32,
    /// Reference to device for cleanup
    device_ref: Arc<VkDeviceRef>,
}

/// Holds Vulkan instance and device references for extension loading.
struct VkDeviceRef {
    instance: ash::Instance,
    device: ash::Device,
}

impl Drop for VkDeviceRef {
    fn drop(&mut self) {
        // Instance and device are owned by wgpu, don't destroy
    }
}

/// Capabilities for scanout/export.
#[derive(Debug, Clone, Default)]
pub struct ScanoutCapabilities {
    /// VK_KHR_external_memory_fd is available
    pub external_memory_fd: bool,
    /// VK_EXT_external_memory_dma_buf is available
    pub external_memory_dmabuf: bool,
    /// VK_EXT_image_drm_format_modifier is available
    pub drm_format_modifier: bool,
    /// Can export render targets as dmabufs
    pub can_export_dmabuf: bool,
}

impl ScanoutCapabilities {
    /// Query scanout capabilities from a Vulkan device.
    ///
    /// # Safety
    /// The physical device must be valid.
    pub unsafe fn query(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Self {
        let extensions = match instance.enumerate_device_extension_properties(physical_device) {
            Ok(exts) => exts,
            Err(_) => return Self::default(),
        };

        let extension_names: Vec<&std::ffi::CStr> = extensions
            .iter()
            .map(|ext| std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()))
            .collect();

        let external_memory_fd = extension_names.contains(&ash::khr::external_memory_fd::NAME);
        let external_memory_dmabuf =
            extension_names.contains(&ash::ext::external_memory_dma_buf::NAME);
        let drm_format_modifier =
            extension_names.contains(&ash::ext::image_drm_format_modifier::NAME);

        let can_export_dmabuf = external_memory_fd && external_memory_dmabuf;

        debug!(
            "Scanout capabilities: ext_mem_fd={}, dmabuf={}, drm_mod={}, can_export={}",
            external_memory_fd, external_memory_dmabuf, drm_format_modifier, can_export_dmabuf
        );

        Self {
            external_memory_fd,
            external_memory_dmabuf,
            drm_format_modifier,
            can_export_dmabuf,
        }
    }
}

/// Configuration for creating an exportable render target.
#[derive(Debug, Clone)]
pub struct RenderTargetConfig {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Desired DRM format
    pub format: drm_fourcc::DrmFourcc,
    /// Desired modifier (0 for linear)
    pub modifier: u64,
}

impl Default for RenderTargetConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            format: drm_fourcc::DrmFourcc::Argb8888,
            modifier: 0, // LINEAR
        }
    }
}

/// Exported dmabuf from a render target.
#[derive(Debug)]
pub struct ExportedDmabuf {
    /// The dmabuf file descriptor (owned)
    pub fd: OwnedFd,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// DRM fourcc format
    pub format: drm_fourcc::DrmFourcc,
    /// Format modifier
    pub modifier: u64,
    /// Stride (bytes per row)
    pub stride: u32,
    /// Offset to first pixel
    pub offset: u32,
}

impl RenderTarget {
    /// Create a new exportable render target.
    ///
    /// # Safety
    /// - The instance and device must support the required extensions
    /// - The wgpu device must be backed by the same Vulkan device
    pub unsafe fn new(
        wgpu_device: &wgpu::Device,
        vk_instance: ash::Instance,
        vk_device: ash::Device,
        _vk_physical_device: vk::PhysicalDevice,
        config: &RenderTargetConfig,
    ) -> Result<Self, BridgeError> {
        let width = config.width;
        let height = config.height;

        // Map DRM format to Vulkan format
        let vk_format = drm_to_vk_format(config.format).ok_or_else(|| {
            BridgeError::Render(format!("unsupported format: {:?}", config.format))
        })?;
        let wgpu_format = drm_to_wgpu_format(config.format).ok_or_else(|| {
            BridgeError::Render(format!("unsupported format: {:?}", config.format))
        })?;

        debug!(
            "Creating exportable render target: {}x{} format={:?} modifier={:#x}",
            width, height, config.format, config.modifier
        );

        // Create VkImage with external memory support for export
        let mut external_memory_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::LINEAR) // LINEAR for export compatibility
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external_memory_info);

        let vk_image = vk_device
            .create_image(&image_create_info, None)
            .map_err(|e| BridgeError::Render(format!("failed to create image: {:?}", e)))?;

        let mem_requirements = vk_device.get_image_memory_requirements(vk_image);

        // Query memory properties
        // Note: In a full implementation, we'd get this from the instance
        // For now, use type 0 (typically device-local)
        let memory_type_index = 0u32;

        // Allocate exportable memory
        let mut export_memory_info = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

        let mut dedicated_alloc_info = vk::MemoryDedicatedAllocateInfo::default().image(vk_image);

        let memory_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut export_memory_info)
            .push_next(&mut dedicated_alloc_info);

        let vk_memory = vk_device
            .allocate_memory(&memory_allocate_info, None)
            .map_err(|e| {
                vk_device.destroy_image(vk_image, None);
                BridgeError::Render(format!("failed to allocate memory: {:?}", e))
            })?;

        vk_device
            .bind_image_memory(vk_image, vk_memory, 0)
            .map_err(|e| {
                vk_device.free_memory(vk_memory, None);
                vk_device.destroy_image(vk_image, None);
                BridgeError::Render(format!("failed to bind memory: {:?}", e))
            })?;

        // Calculate stride (for LINEAR tiling)
        let bytes_per_pixel = format_bytes_per_pixel(config.format);
        let stride = width * bytes_per_pixel;

        // Create wgpu texture from the Vulkan image via wgpu-hal
        let hal_desc = wgpu_hal::TextureDescriptor {
            label: Some("exportable-render-target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu_types::TextureUses::COLOR_TARGET
                | wgpu_types::TextureUses::COPY_SRC
                | wgpu_types::TextureUses::COPY_DST,
            memory_flags: wgpu_hal::MemoryFlags::empty(),
            view_formats: vec![],
        };

        // Access wgpu-hal device
        let hal_device_guard = wgpu_device
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or_else(|| BridgeError::Render("Failed to access HAL device".into()))?;
        let hal_device: &wgpu_hal::vulkan::Device = std::ops::Deref::deref(&hal_device_guard);

        // Wrap in HAL texture
        let hal_texture = hal_device.texture_from_raw(
            vk_image,
            &hal_desc,
            None,
            wgpu_hal::vulkan::TextureMemory::Dedicated(vk_memory),
        );

        // Create wgpu texture
        let wgpu_desc = wgpu::TextureDescriptor {
            label: Some("exportable-render-target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };

        let texture =
            wgpu_device.create_texture_from_hal::<wgpu_hal::api::Vulkan>(hal_texture, &wgpu_desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        debug!("Created exportable render target successfully");

        Ok(Self {
            texture,
            view,
            vk_image,
            vk_memory,
            width,
            height,
            drm_format: config.format,
            modifier: config.modifier,
            stride,
            device_ref: Arc::new(VkDeviceRef {
                instance: vk_instance,
                device: vk_device,
            }),
        })
    }

    /// Get the wgpu texture for rendering into.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// Get the texture view for render pass attachment.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Get dimensions.
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get the DRM format.
    pub fn format(&self) -> drm_fourcc::DrmFourcc {
        self.drm_format
    }

    /// Export as a dmabuf file descriptor.
    ///
    /// # Safety
    /// - The GPU must have finished all work using this render target
    /// - Proper synchronization should be in place (use sync FDs)
    pub unsafe fn export_dmabuf(&self) -> Result<ExportedDmabuf, BridgeError> {
        // Load the external_memory_fd extension
        let external_memory_fd = ash::khr::external_memory_fd::Device::new(
            &self.device_ref.instance,
            &self.device_ref.device,
        );

        let get_fd_info = vk::MemoryGetFdInfoKHR::default()
            .memory(self.vk_memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

        let fd = external_memory_fd
            .get_memory_fd(&get_fd_info)
            .map_err(|e| BridgeError::Render(format!("failed to export dmabuf: {:?}", e)))?;

        trace!("Exported dmabuf fd {}", fd);

        // SAFETY: Vulkan transfers ownership of the FD to us
        let owned_fd = OwnedFd::from_raw_fd(fd);

        Ok(ExportedDmabuf {
            fd: owned_fd,
            width: self.width,
            height: self.height,
            format: self.drm_format,
            modifier: self.modifier,
            stride: self.stride,
            offset: 0,
        })
    }
}

impl Drop for RenderTarget {
    fn drop(&mut self) {
        // Resource cleanup order is important:
        //
        // 1. wgpu::Texture drops (happens automatically before this runs)
        //    - This drops the wgpu-hal texture
        //    - wgpu-hal frees vk_memory via TextureMemory::Dedicated
        //
        // 2. We destroy vk_image (wgpu-hal doesn't own images from texture_from_raw)
        //
        // Note: We do NOT free vk_memory - wgpu-hal already did that.

        // SAFETY: device_ref.device and vk_image are valid, and we own the image.
        unsafe {
            self.device_ref.device.destroy_image(self.vk_image, None);
        }

        debug!("Dropped exportable render target: destroyed VkImage");
    }
}

/// Map DRM fourcc format to Vulkan format.
fn drm_to_vk_format(fourcc: drm_fourcc::DrmFourcc) -> Option<vk::Format> {
    match fourcc {
        drm_fourcc::DrmFourcc::Argb8888 => Some(vk::Format::B8G8R8A8_UNORM),
        drm_fourcc::DrmFourcc::Xrgb8888 => Some(vk::Format::B8G8R8A8_UNORM),
        drm_fourcc::DrmFourcc::Abgr8888 => Some(vk::Format::R8G8B8A8_UNORM),
        drm_fourcc::DrmFourcc::Xbgr8888 => Some(vk::Format::R8G8B8A8_UNORM),
        drm_fourcc::DrmFourcc::Argb2101010 => Some(vk::Format::A2B10G10R10_UNORM_PACK32),
        drm_fourcc::DrmFourcc::Xrgb2101010 => Some(vk::Format::A2B10G10R10_UNORM_PACK32),
        _ => None,
    }
}

/// Map DRM fourcc format to wgpu format.
fn drm_to_wgpu_format(fourcc: drm_fourcc::DrmFourcc) -> Option<wgpu::TextureFormat> {
    match fourcc {
        drm_fourcc::DrmFourcc::Argb8888 => Some(wgpu::TextureFormat::Bgra8Unorm),
        drm_fourcc::DrmFourcc::Xrgb8888 => Some(wgpu::TextureFormat::Bgra8Unorm),
        drm_fourcc::DrmFourcc::Abgr8888 => Some(wgpu::TextureFormat::Rgba8Unorm),
        drm_fourcc::DrmFourcc::Xbgr8888 => Some(wgpu::TextureFormat::Rgba8Unorm),
        drm_fourcc::DrmFourcc::Argb2101010 => Some(wgpu::TextureFormat::Rgb10a2Unorm),
        drm_fourcc::DrmFourcc::Xrgb2101010 => Some(wgpu::TextureFormat::Rgb10a2Unorm),
        _ => None,
    }
}

/// Get bytes per pixel for a format.
fn format_bytes_per_pixel(fourcc: drm_fourcc::DrmFourcc) -> u32 {
    match fourcc {
        drm_fourcc::DrmFourcc::Argb8888
        | drm_fourcc::DrmFourcc::Xrgb8888
        | drm_fourcc::DrmFourcc::Abgr8888
        | drm_fourcc::DrmFourcc::Xbgr8888
        | drm_fourcc::DrmFourcc::Argb2101010
        | drm_fourcc::DrmFourcc::Xrgb2101010 => 4,
        drm_fourcc::DrmFourcc::Rgb888 | drm_fourcc::DrmFourcc::Bgr888 => 3,
        _ => 4, // Default to 4 bytes
    }
}

impl std::fmt::Debug for RenderTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderTarget")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &self.drm_format)
            .field("modifier", &format_args!("{:#x}", self.modifier))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_mapping() {
        assert_eq!(
            drm_to_vk_format(drm_fourcc::DrmFourcc::Argb8888),
            Some(vk::Format::B8G8R8A8_UNORM)
        );
        assert_eq!(
            drm_to_wgpu_format(drm_fourcc::DrmFourcc::Argb8888),
            Some(wgpu::TextureFormat::Bgra8Unorm)
        );
    }

    #[test]
    fn test_bytes_per_pixel() {
        assert_eq!(format_bytes_per_pixel(drm_fourcc::DrmFourcc::Argb8888), 4);
        assert_eq!(format_bytes_per_pixel(drm_fourcc::DrmFourcc::Rgb888), 3);
    }
}
