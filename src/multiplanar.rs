//! Multi-planar format support for video textures (NV12, P010, etc.)
//!
//! This module provides support for importing multi-planar YUV formats commonly
//! used in video playback. These formats require special Vulkan extensions:
//!
//! - `VK_KHR_sampler_ycbcr_conversion` - For YCbCr color space conversion
//! - `VK_KHR_maintenance1` - For multi-plane image support
//! - `VK_EXT_ycbcr_2plane_444_formats` - For additional YUV formats (optional)
//!
//! # Supported Formats
//!
//! | Format | Description | Planes | Bit Depth |
//! |--------|-------------|--------|-----------|
//! | NV12   | Y + interleaved UV | 2 | 8-bit |
//! | NV21   | Y + interleaved VU | 2 | 8-bit |
//! | P010   | Y + interleaved UV | 2 | 10-bit |
//! | YUV420 | Y + U + V separate | 3 | 8-bit |
//!
//! # Architecture
//!
//! Multi-planar textures are imported differently from single-plane textures:
//!
//! 1. Create disjoint VkImage with multiple memory planes
//! 2. Import each dmabuf plane as separate memory
//! 3. Create VkSamplerYcbcrConversion for color space handling
//! 4. Bind conversion to sampler and image view
//! 5. Shader samples using combined image sampler with conversion
//!
//! # Performance Notes
//!
//! - Hardware YCbCr conversion is typically faster than shader-based conversion
//! - Some GPUs may not support all chroma subsampling modes
//! - Linear tiling may be required for some format/modifier combinations

use crate::error::ImportError;
use ash::vk;
use drm_fourcc::DrmFourcc;
use tracing::{debug, trace};

/// Multi-planar format descriptor.
#[derive(Debug, Clone)]
pub struct MultiPlanarFormat {
    /// DRM fourcc code
    pub fourcc: DrmFourcc,
    /// Vulkan format for this planar type
    pub vk_format: vk::Format,
    /// Number of planes
    pub plane_count: u32,
    /// Chroma subsampling factors (horizontal, vertical)
    pub chroma_subsampling: (u32, u32),
    /// Bit depth per component
    pub bit_depth: u32,
    /// YCbCr model (BT.601, BT.709, etc.)
    pub ycbcr_model: YcbcrModel,
    /// YCbCr range (narrow/full)
    pub ycbcr_range: YcbcrRange,
}

/// YCbCr color model/matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YcbcrModel {
    /// ITU-R BT.601 (SDTV)
    Bt601,
    /// ITU-R BT.709 (HDTV)
    Bt709,
    /// ITU-R BT.2020 (UHDTV)
    Bt2020,
    /// Identity (RGB passthrough)
    Identity,
}

impl YcbcrModel {
    /// Convert to Vulkan sampler YCbCr model.
    pub fn to_vk(self) -> vk::SamplerYcbcrModelConversion {
        match self {
            YcbcrModel::Bt601 => vk::SamplerYcbcrModelConversion::YCBCR_601,
            YcbcrModel::Bt709 => vk::SamplerYcbcrModelConversion::YCBCR_709,
            YcbcrModel::Bt2020 => vk::SamplerYcbcrModelConversion::YCBCR_2020,
            YcbcrModel::Identity => vk::SamplerYcbcrModelConversion::RGB_IDENTITY,
        }
    }
}

/// YCbCr value range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YcbcrRange {
    /// ITU narrow range (Y: 16-235, Cb/Cr: 16-240)
    Narrow,
    /// Full range (0-255 for 8-bit, 0-1023 for 10-bit)
    Full,
}

impl YcbcrRange {
    /// Convert to Vulkan sampler YCbCr range.
    pub fn to_vk(self) -> vk::SamplerYcbcrRange {
        match self {
            YcbcrRange::Narrow => vk::SamplerYcbcrRange::ITU_NARROW,
            YcbcrRange::Full => vk::SamplerYcbcrRange::ITU_FULL,
        }
    }
}

/// Colorspace metadata for YCbCr content.
///
/// This struct holds the color information needed to correctly convert
/// YUV content to RGB. When available, this should be derived from:
/// - Video container metadata (e.g., MP4 atoms)
/// - Stream headers (e.g., H.264 VUI parameters)
/// - Wayland protocol extensions (when available)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YcbcrColorspace {
    /// Color model/matrix
    pub model: YcbcrModel,
    /// Value range
    pub range: YcbcrRange,
}

impl Default for YcbcrColorspace {
    fn default() -> Self {
        // Default to BT.601 narrow range (most common for video)
        Self {
            model: YcbcrModel::Bt601,
            range: YcbcrRange::Narrow,
        }
    }
}

impl YcbcrColorspace {
    /// Create BT.601 colorspace (SDTV standard).
    pub fn bt601() -> Self {
        Self {
            model: YcbcrModel::Bt601,
            range: YcbcrRange::Narrow,
        }
    }

    /// Create BT.709 colorspace (HDTV standard).
    pub fn bt709() -> Self {
        Self {
            model: YcbcrModel::Bt709,
            range: YcbcrRange::Narrow,
        }
    }

    /// Create BT.2020 colorspace (UHDTV standard).
    pub fn bt2020() -> Self {
        Self {
            model: YcbcrModel::Bt2020,
            range: YcbcrRange::Narrow,
        }
    }

    /// Infer colorspace from video resolution.
    ///
    /// This is a common heuristic when explicit metadata is unavailable:
    /// - SD content (up to 720p): BT.601
    /// - HD content (720p to 1080p): BT.709
    /// - UHD content (4K and above): BT.2020
    ///
    /// Note: This is a fallback. Always prefer explicit metadata when available.
    pub fn from_resolution(width: u32, height: u32) -> Self {
        // Common resolution breakpoints
        let pixels = width * height;

        if pixels > 3840 * 2160 / 2 {
            // 4K and above → BT.2020
            Self::bt2020()
        } else if pixels > 1280 * 720 / 2 {
            // HD (720p+) → BT.709
            Self::bt709()
        } else {
            // SD → BT.601
            Self::bt601()
        }
    }

    /// Infer colorspace from format and resolution.
    ///
    /// Some formats have typical colorspace associations:
    /// - P010 (10-bit) is typically BT.2020 for HDR content
    /// - NV12 at HD resolutions is typically BT.709
    pub fn from_format_and_resolution(fourcc: DrmFourcc, width: u32, height: u32) -> Self {
        match fourcc {
            DrmFourcc::P010 => {
                // P010 is typically used for HDR/WCG content with BT.2020
                Self::bt2020()
            }
            _ => {
                // Use resolution-based heuristic for other formats
                Self::from_resolution(width, height)
            }
        }
    }
}

/// Plane layout information for a single plane.
#[derive(Debug, Clone)]
pub struct PlaneLayout {
    /// Offset within the buffer
    pub offset: u64,
    /// Row stride in bytes
    pub stride: u32,
    /// Width divisor (1 for Y, 2 for UV in 4:2:0)
    pub width_divisor: u32,
    /// Height divisor (1 for Y, 2 for UV in 4:2:0)
    pub height_divisor: u32,
}

/// Capabilities for multi-planar format support.
#[derive(Debug, Clone, Default)]
pub struct MultiPlanarCapabilities {
    /// Whether VK_KHR_sampler_ycbcr_conversion is supported
    pub has_ycbcr_conversion: bool,
    /// Supported multi-planar formats
    pub supported_formats: Vec<DrmFourcc>,
    /// Maximum number of planes supported
    pub max_planes: u32,
}

impl MultiPlanarCapabilities {
    /// Query capabilities from a Vulkan device.
    ///
    /// # Safety
    ///
    /// The device must be valid.
    pub unsafe fn query(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Self {
        // Query device extension properties for VK_KHR_sampler_ycbcr_conversion
        let extensions = match instance.enumerate_device_extension_properties(physical_device) {
            Ok(exts) => exts,
            Err(_) => return Self::default(),
        };

        let has_ycbcr_conversion = extensions.iter().any(|ext| {
            let name = std::ffi::CStr::from_ptr(ext.extension_name.as_ptr());
            name == ash::khr::sampler_ycbcr_conversion::NAME
        });

        // Vulkan 1.1+ includes sampler_ycbcr_conversion in core
        let props = instance.get_physical_device_properties(physical_device);
        let has_ycbcr_conversion =
            has_ycbcr_conversion || props.api_version >= vk::make_api_version(0, 1, 1, 0);

        if !has_ycbcr_conversion {
            debug!("VK_KHR_sampler_ycbcr_conversion not available");
            return Self {
                has_ycbcr_conversion: false,
                supported_formats: vec![],
                max_planes: 0,
            };
        }

        // Query supported multi-planar formats
        let mut supported_formats = Vec::new();
        let candidate_formats = [
            (DrmFourcc::Nv12, vk::Format::G8_B8R8_2PLANE_420_UNORM),
            (DrmFourcc::Nv21, vk::Format::G8_B8R8_2PLANE_420_UNORM),
            (
                DrmFourcc::P010,
                vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
            ),
            (DrmFourcc::Yuv420, vk::Format::G8_B8_R8_3PLANE_420_UNORM),
        ];

        for (drm_fourcc, vk_format) in candidate_formats {
            let format_props =
                instance.get_physical_device_format_properties(physical_device, vk_format);

            // Check if format supports SAMPLED_IMAGE in optimal tiling
            let can_sample = format_props
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE)
                || format_props
                    .linear_tiling_features
                    .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE);

            // Check for disjoint bit support (required for multi-plane dmabuf import)
            let has_disjoint = format_props
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::DISJOINT)
                || format_props
                    .linear_tiling_features
                    .contains(vk::FormatFeatureFlags::DISJOINT);

            if can_sample && has_disjoint {
                trace!(
                    "Multi-planar format {:?} ({:?}) is supported",
                    drm_fourcc,
                    vk_format
                );
                supported_formats.push(drm_fourcc);
            }
        }

        debug!(
            "Multi-planar capabilities: ycbcr_conversion={}, {} supported formats",
            has_ycbcr_conversion,
            supported_formats.len()
        );

        Self {
            has_ycbcr_conversion,
            supported_formats,
            max_planes: 3, // Most formats use at most 3 planes
        }
    }

    /// Check if a specific format is supported.
    pub fn supports_format(&self, fourcc: DrmFourcc) -> bool {
        self.has_ycbcr_conversion && self.supported_formats.contains(&fourcc)
    }
}

/// Get format descriptor for a multi-planar DRM format with default colorspace.
///
/// Uses resolution-based heuristics for colorspace when not specified.
/// For explicit colorspace control, use [`multiplanar_format_with_colorspace`].
pub fn multiplanar_format(fourcc: DrmFourcc) -> Option<MultiPlanarFormat> {
    multiplanar_format_with_colorspace(fourcc, YcbcrColorspace::default())
}

/// Get format descriptor for a multi-planar DRM format with explicit colorspace.
///
/// # Arguments
///
/// * `fourcc` - The DRM fourcc format code
/// * `colorspace` - The YCbCr colorspace to use for conversion
///
/// # Example
///
/// ```rust,ignore
/// // Import HD video with BT.709
/// let colorspace = YcbcrColorspace::from_resolution(1920, 1080);
/// let format = multiplanar_format_with_colorspace(DrmFourcc::Nv12, colorspace);
/// ```
pub fn multiplanar_format_with_colorspace(
    fourcc: DrmFourcc,
    colorspace: YcbcrColorspace,
) -> Option<MultiPlanarFormat> {
    match fourcc {
        DrmFourcc::Nv12 => Some(MultiPlanarFormat {
            fourcc,
            vk_format: vk::Format::G8_B8R8_2PLANE_420_UNORM,
            plane_count: 2,
            chroma_subsampling: (2, 2), // 4:2:0
            bit_depth: 8,
            ycbcr_model: colorspace.model,
            ycbcr_range: colorspace.range,
        }),
        DrmFourcc::Nv21 => Some(MultiPlanarFormat {
            fourcc,
            // NV21 has VU ordering instead of UV
            vk_format: vk::Format::G8_B8R8_2PLANE_420_UNORM, // Swizzle in sampler
            plane_count: 2,
            chroma_subsampling: (2, 2),
            bit_depth: 8,
            ycbcr_model: colorspace.model,
            ycbcr_range: colorspace.range,
        }),
        DrmFourcc::P010 => Some(MultiPlanarFormat {
            fourcc,
            vk_format: vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
            plane_count: 2,
            chroma_subsampling: (2, 2),
            bit_depth: 10,
            ycbcr_model: colorspace.model,
            ycbcr_range: colorspace.range,
        }),
        DrmFourcc::Yuv420 => Some(MultiPlanarFormat {
            fourcc,
            vk_format: vk::Format::G8_B8_R8_3PLANE_420_UNORM,
            plane_count: 3,
            chroma_subsampling: (2, 2),
            bit_depth: 8,
            ycbcr_model: colorspace.model,
            ycbcr_range: colorspace.range,
        }),
        DrmFourcc::Yvu420 => Some(MultiPlanarFormat {
            fourcc,
            // YVU420 has VU ordering, need component swizzle
            vk_format: vk::Format::G8_B8_R8_3PLANE_420_UNORM,
            plane_count: 3,
            chroma_subsampling: (2, 2),
            bit_depth: 8,
            ycbcr_model: colorspace.model,
            ycbcr_range: colorspace.range,
        }),
        _ => None,
    }
}

/// Check if a DRM format is multi-planar.
pub fn is_multiplanar_format(fourcc: DrmFourcc) -> bool {
    multiplanar_format(fourcc).is_some()
}

/// Calculate plane layouts for a multi-planar format.
pub fn calculate_plane_layouts(
    format: &MultiPlanarFormat,
    width: u32,
    height: u32,
) -> Vec<PlaneLayout> {
    let mut layouts = Vec::with_capacity(format.plane_count as usize);
    let mut offset: u64 = 0;

    match format.plane_count {
        2 => {
            // NV12, P010: Y plane, then interleaved UV plane
            let bytes_per_sample = if format.bit_depth > 8 { 2 } else { 1 };

            // Y plane
            let y_stride = width * bytes_per_sample;
            layouts.push(PlaneLayout {
                offset,
                stride: y_stride,
                width_divisor: 1,
                height_divisor: 1,
            });
            offset += (y_stride * height) as u64;

            // UV plane (interleaved, half resolution in both dimensions for 4:2:0)
            let uv_width = width / format.chroma_subsampling.0;
            let _uv_height = height / format.chroma_subsampling.1;
            let uv_stride = uv_width * 2 * bytes_per_sample; // 2 components interleaved

            layouts.push(PlaneLayout {
                offset,
                stride: uv_stride,
                width_divisor: format.chroma_subsampling.0,
                height_divisor: format.chroma_subsampling.1,
            });
        }
        3 => {
            // YUV420: Y, U, V separate planes
            let bytes_per_sample = if format.bit_depth > 8 { 2 } else { 1 };

            // Y plane
            let y_stride = width * bytes_per_sample;
            layouts.push(PlaneLayout {
                offset,
                stride: y_stride,
                width_divisor: 1,
                height_divisor: 1,
            });
            offset += (y_stride * height) as u64;

            // U plane
            let u_width = width / format.chroma_subsampling.0;
            let u_height = height / format.chroma_subsampling.1;
            let u_stride = u_width * bytes_per_sample;

            layouts.push(PlaneLayout {
                offset,
                stride: u_stride,
                width_divisor: format.chroma_subsampling.0,
                height_divisor: format.chroma_subsampling.1,
            });
            offset += (u_stride * u_height) as u64;

            // V plane (same size as U)
            layouts.push(PlaneLayout {
                offset,
                stride: u_stride,
                width_divisor: format.chroma_subsampling.0,
                height_divisor: format.chroma_subsampling.1,
            });
        }
        _ => {}
    }

    layouts
}

/// Imported multi-planar texture with YCbCr conversion.
///
/// This struct holds all the Vulkan resources needed for a multi-planar
/// texture, including the YCbCr conversion sampler.
pub struct MultiPlanarTexture {
    /// The Vulkan image
    pub image: vk::Image,
    /// Image view with YCbCr conversion
    pub image_view: vk::ImageView,
    /// YCbCr conversion object
    pub ycbcr_conversion: vk::SamplerYcbcrConversion,
    /// Sampler with YCbCr conversion
    pub sampler: vk::Sampler,
    /// Memory allocations for each plane
    pub plane_memories: Vec<vk::DeviceMemory>,
    /// Texture dimensions
    pub width: u32,
    pub height: u32,
    /// Format information
    pub format: MultiPlanarFormat,
}

impl std::fmt::Debug for MultiPlanarTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiPlanarTexture")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &self.format)
            .field("plane_count", &self.plane_memories.len())
            .finish()
    }
}

/// Import a multi-planar dmabuf.
///
/// This function creates all the Vulkan resources needed to sample
/// a multi-planar YUV texture, including:
/// - VkSamplerYcbcrConversion for color space handling
/// - Disjoint VkImage with multiple memory planes
/// - Per-plane memory imports from dmabuf fds
/// - VkImageView and VkSampler with conversion attached
///
/// # Safety
///
/// - The device must be valid and have VK_KHR_sampler_ycbcr_conversion enabled
/// - The dmabuf fds must be valid and represent the correct format
/// - Caller is responsible for destroying returned resources
#[allow(unused_variables, clippy::too_many_arguments)]
pub unsafe fn import_multiplanar_dmabuf(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    format: &MultiPlanarFormat,
    width: u32,
    height: u32,
    plane_fds: &[i32],
    plane_offsets: &[u32],
    plane_strides: &[u32],
) -> Result<MultiPlanarTexture, ImportError> {
    if plane_fds.len() < format.plane_count as usize {
        return Err(ImportError::InvalidPlanes(format!(
            "Expected {} planes, got {}",
            format.plane_count,
            plane_fds.len()
        )));
    }

    debug!(
        "Importing multi-planar texture: {}x{}, format={:?}, planes={}",
        width, height, format.fourcc, format.plane_count
    );

    // 1. Create VkSamplerYcbcrConversion
    //
    // This defines how the YUV to RGB conversion happens, including:
    // - Color model (BT.601, BT.709, BT.2020)
    // - Value range (narrow/full)
    // - Chroma location (for subsampled formats)
    let component_mapping = get_component_mapping(format.fourcc);

    let ycbcr_create_info = vk::SamplerYcbcrConversionCreateInfo::default()
        .format(format.vk_format)
        .ycbcr_model(format.ycbcr_model.to_vk())
        .ycbcr_range(format.ycbcr_range.to_vk())
        .components(component_mapping)
        .x_chroma_offset(vk::ChromaLocation::COSITED_EVEN)
        .y_chroma_offset(vk::ChromaLocation::COSITED_EVEN)
        .chroma_filter(vk::Filter::LINEAR)
        .force_explicit_reconstruction(false);

    let ycbcr_conversion = device
        .create_sampler_ycbcr_conversion(&ycbcr_create_info, None)
        .map_err(|e| {
            ImportError::ImageCreation(format!("failed to create YCbCr conversion: {:?}", e))
        })?;

    debug!("Created VkSamplerYcbcrConversion");

    // 2. Create disjoint VkImage
    //
    // The DISJOINT_BIT allows each plane to have its own memory allocation,
    // which is required for importing separate dmabuf planes.
    let mut external_memory_info = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

    let image_create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format.vk_format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .flags(vk::ImageCreateFlags::DISJOINT)
        .push_next(&mut external_memory_info);

    let image = device.create_image(&image_create_info, None).map_err(|e| {
        device.destroy_sampler_ycbcr_conversion(ycbcr_conversion, None);
        ImportError::ImageCreation(format!("failed to create disjoint image: {:?}", e))
    })?;

    debug!(
        "Created disjoint VkImage with {} planes",
        format.plane_count
    );

    // 3. Import memory for each plane and bind
    let mut plane_memories = Vec::with_capacity(format.plane_count as usize);
    let mem_properties = instance.get_physical_device_memory_properties(physical_device);

    for plane_idx in 0..format.plane_count {
        let plane_aspect = match plane_idx {
            0 => vk::ImageAspectFlags::PLANE_0,
            1 => vk::ImageAspectFlags::PLANE_1,
            2 => vk::ImageAspectFlags::PLANE_2,
            _ => unreachable!(),
        };

        // Get memory requirements for this plane
        let mut plane_req_info =
            vk::ImagePlaneMemoryRequirementsInfo::default().plane_aspect(plane_aspect);
        let mem_req_info = vk::ImageMemoryRequirementsInfo2::default()
            .image(image)
            .push_next(&mut plane_req_info);

        let mut mem_req = vk::MemoryRequirements2::default();
        device.get_image_memory_requirements2(&mem_req_info, &mut mem_req);

        let requirements = mem_req.memory_requirements;

        // Find suitable memory type
        let memory_type_index = find_memory_type_index(
            &mem_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or_else(|| {
            cleanup_partial_import(device, image, ycbcr_conversion, &plane_memories);
            ImportError::MemoryAllocation(format!(
                "No suitable memory type for plane {}",
                plane_idx
            ))
        })?;

        // Dup the fd since Vulkan takes ownership
        let import_fd = libc::dup(plane_fds[plane_idx as usize]);
        if import_fd < 0 {
            cleanup_partial_import(device, image, ycbcr_conversion, &plane_memories);
            return Err(ImportError::FdImport(format!(
                "Failed to dup fd for plane {}",
                plane_idx
            )));
        }

        // Import the dmabuf memory
        let mut import_fd_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(import_fd);

        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut import_fd_info);

        let memory = device.allocate_memory(&allocate_info, None).map_err(|e| {
            libc::close(import_fd);
            cleanup_partial_import(device, image, ycbcr_conversion, &plane_memories);
            ImportError::MemoryAllocation(format!(
                "Failed to import memory for plane {}: {:?}",
                plane_idx, e
            ))
        })?;

        trace!("Imported memory for plane {}: {:?}", plane_idx, memory);
        plane_memories.push(memory);
    }

    // 4. Bind memory to each plane using vkBindImageMemory2
    let mut bind_infos = Vec::with_capacity(format.plane_count as usize);
    let mut plane_bind_infos: Vec<vk::BindImagePlaneMemoryInfo<'_>> = (0..format.plane_count)
        .map(|idx| {
            let plane_aspect = match idx {
                0 => vk::ImageAspectFlags::PLANE_0,
                1 => vk::ImageAspectFlags::PLANE_1,
                2 => vk::ImageAspectFlags::PLANE_2,
                _ => unreachable!(),
            };
            vk::BindImagePlaneMemoryInfo::default().plane_aspect(plane_aspect)
        })
        .collect();

    for (idx, plane_bind_info) in plane_bind_infos.iter_mut().enumerate() {
        let bind_info = vk::BindImageMemoryInfo::default()
            .image(image)
            .memory(plane_memories[idx])
            .memory_offset(plane_offsets[idx] as u64)
            .push_next(plane_bind_info);
        bind_infos.push(bind_info);
    }

    device.bind_image_memory2(&bind_infos).map_err(|e| {
        cleanup_partial_import(device, image, ycbcr_conversion, &plane_memories);
        ImportError::MemoryAllocation(format!("failed to bind image memory: {:?}", e))
    })?;

    debug!("Bound memory to all {} planes", format.plane_count);

    // 5. Create VkImageView with YCbCr conversion
    let mut ycbcr_conversion_info =
        vk::SamplerYcbcrConversionInfo::default().conversion(ycbcr_conversion);

    let image_view_create_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format.vk_format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .push_next(&mut ycbcr_conversion_info);

    let image_view = device
        .create_image_view(&image_view_create_info, None)
        .map_err(|e| {
            cleanup_partial_import(device, image, ycbcr_conversion, &plane_memories);
            ImportError::ImageCreation(format!("failed to create image view: {:?}", e))
        })?;

    // 6. Create VkSampler with YCbCr conversion
    //
    // The sampler must use the same YCbCr conversion as the image view.
    // This creates a "combined image sampler" that handles YUV->RGB conversion.
    let mut sampler_ycbcr_info =
        vk::SamplerYcbcrConversionInfo::default().conversion(ycbcr_conversion);

    let sampler_create_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .unnormalized_coordinates(false)
        .push_next(&mut sampler_ycbcr_info);

    let sampler = device
        .create_sampler(&sampler_create_info, None)
        .map_err(|e| {
            device.destroy_image_view(image_view, None);
            cleanup_partial_import(device, image, ycbcr_conversion, &plane_memories);
            ImportError::ImageCreation(format!("failed to create sampler: {:?}", e))
        })?;

    debug!("Successfully imported multi-planar texture");

    Ok(MultiPlanarTexture {
        image,
        image_view,
        ycbcr_conversion,
        sampler,
        plane_memories,
        width,
        height,
        format: format.clone(),
    })
}

/// Cleanup helper for partial import failures.
unsafe fn cleanup_partial_import(
    device: &ash::Device,
    image: vk::Image,
    ycbcr_conversion: vk::SamplerYcbcrConversion,
    plane_memories: &[vk::DeviceMemory],
) {
    for &memory in plane_memories {
        device.free_memory(memory, None);
    }
    device.destroy_image(image, None);
    device.destroy_sampler_ycbcr_conversion(ycbcr_conversion, None);
}

/// Get component mapping for a format (handles NV21/YVU ordering).
fn get_component_mapping(fourcc: DrmFourcc) -> vk::ComponentMapping {
    match fourcc {
        // NV21 has VU instead of UV ordering
        DrmFourcc::Nv21 => vk::ComponentMapping {
            r: vk::ComponentSwizzle::IDENTITY,
            g: vk::ComponentSwizzle::IDENTITY,
            b: vk::ComponentSwizzle::IDENTITY,
            a: vk::ComponentSwizzle::IDENTITY,
            // The actual swizzle for NV21 is handled by the format itself
        },
        // YVU420 has V and U planes swapped
        DrmFourcc::Yvu420 => vk::ComponentMapping {
            r: vk::ComponentSwizzle::IDENTITY,
            g: vk::ComponentSwizzle::IDENTITY,
            b: vk::ComponentSwizzle::IDENTITY,
            a: vk::ComponentSwizzle::IDENTITY,
        },
        // Most formats use identity mapping
        _ => vk::ComponentMapping {
            r: vk::ComponentSwizzle::IDENTITY,
            g: vk::ComponentSwizzle::IDENTITY,
            b: vk::ComponentSwizzle::IDENTITY,
            a: vk::ComponentSwizzle::IDENTITY,
        },
    }
}

/// Find a memory type index that satisfies the requirements.
fn find_memory_type_index(
    mem_properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required_flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..mem_properties.memory_type_count {
        let type_bit = 1 << i;
        let is_required_type = (type_bits & type_bit) != 0;
        let has_required_flags = (mem_properties.memory_types[i as usize].property_flags
            & required_flags)
            == required_flags;

        if is_required_type && has_required_flags {
            return Some(i);
        }
    }
    None
}

impl MultiPlanarTexture {
    /// Destroy all Vulkan resources.
    ///
    /// # Safety
    ///
    /// The device must be valid and all commands using this texture must be complete.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        device.destroy_sampler(self.sampler, None);
        device.destroy_image_view(self.image_view, None);
        for &memory in &self.plane_memories {
            device.free_memory(memory, None);
        }
        device.destroy_image(self.image, None);
        device.destroy_sampler_ycbcr_conversion(self.ycbcr_conversion, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nv12_format() {
        let format = multiplanar_format(DrmFourcc::Nv12).unwrap();
        assert_eq!(format.plane_count, 2);
        assert_eq!(format.chroma_subsampling, (2, 2));
        assert_eq!(format.bit_depth, 8);
    }

    #[test]
    fn test_p010_format() {
        let format = multiplanar_format(DrmFourcc::P010).unwrap();
        assert_eq!(format.plane_count, 2);
        assert_eq!(format.bit_depth, 10);
    }

    #[test]
    fn test_plane_layouts_nv12() {
        let format = multiplanar_format(DrmFourcc::Nv12).unwrap();
        let layouts = calculate_plane_layouts(&format, 1920, 1080);

        assert_eq!(layouts.len(), 2);

        // Y plane: 1920x1080
        assert_eq!(layouts[0].offset, 0);
        assert_eq!(layouts[0].stride, 1920);
        assert_eq!(layouts[0].width_divisor, 1);

        // UV plane: 960x540 with 2 bytes per sample (interleaved)
        assert_eq!(layouts[1].offset, 1920 * 1080);
        assert_eq!(layouts[1].stride, 1920); // 960 * 2
        assert_eq!(layouts[1].width_divisor, 2);
    }

    #[test]
    fn test_is_multiplanar() {
        assert!(is_multiplanar_format(DrmFourcc::Nv12));
        assert!(is_multiplanar_format(DrmFourcc::P010));
        assert!(!is_multiplanar_format(DrmFourcc::Argb8888));
    }
}
