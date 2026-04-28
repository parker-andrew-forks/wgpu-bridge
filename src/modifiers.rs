//! DRM format modifier support via VK_EXT_image_drm_format_modifier.
//!
//! This module provides functionality for:
//! - Querying supported DRM format modifiers from the Vulkan device
//! - Creating images with explicit tiling modifiers
//! - Importing dmabufs with tiled memory layouts
//!
//! ## Background
//!
//! DRM format modifiers are 64-bit vendor-prefixed integers that describe
//! GPU-specific memory tiling layouts. Common examples:
//! - `DRM_FORMAT_MOD_LINEAR` (0x0) - Linear/scanout-compatible layout
//! - `I915_FORMAT_MOD_X_TILED` - Intel X-tiling
//! - `I915_FORMAT_MOD_Y_TILED` - Intel Y-tiling
//! - `NVIDIA_FORMAT_MOD_16BX2_BLOCK(...)` - NVIDIA block-linear layouts
//!
//! Using tiled modifiers improves GPU memory bandwidth by 20-40% compared
//! to linear layouts, but requires proper negotiation between clients.

use ash::vk;
use std::ffi::CStr;
use tracing::{debug, trace, warn};

/// Well-known DRM format modifier constants.
pub mod drm_mod {
    /// Linear (non-tiled) memory layout.
    pub const LINEAR: u64 = 0;

    /// Invalid modifier - indicates "use implicit modifier" or "no modifier".
    pub const INVALID: u64 = 0x00ff_ffff_ffff_ffff;

    /// Vendor prefix mask (upper 8 bits).
    pub const VENDOR_MASK: u64 = 0xff00_0000_0000_0000;

    /// No vendor (used for LINEAR and INVALID).
    pub const VENDOR_NONE: u64 = 0;

    /// Intel vendor prefix.
    pub const VENDOR_INTEL: u64 = 0x0100_0000_0000_0000;

    /// AMD vendor prefix.
    pub const VENDOR_AMD: u64 = 0x0200_0000_0000_0000;

    /// NVIDIA vendor prefix.
    pub const VENDOR_NVIDIA: u64 = 0x0300_0000_0000_0000;

    /// Samsung vendor prefix.
    pub const VENDOR_SAMSUNG: u64 = 0x0400_0000_0000_0000;

    /// Qualcomm vendor prefix.
    pub const VENDOR_QCOM: u64 = 0x0500_0000_0000_0000;

    /// ARM vendor prefix.
    pub const VENDOR_ARM: u64 = 0x0800_0000_0000_0000;

    /// Broadcom vendor prefix.
    pub const VENDOR_BROADCOM: u64 = 0x0700_0000_0000_0000;
}

/// Capabilities for DRM format modifier support.
#[derive(Debug, Clone)]
pub struct ModifierCapabilities {
    /// Whether VK_EXT_image_drm_format_modifier is available.
    pub extension_available: bool,

    /// Whether the extension is enabled on the device.
    pub extension_enabled: bool,

    /// Maximum number of planes supported per image.
    pub max_planes: u32,
}

impl Default for ModifierCapabilities {
    fn default() -> Self {
        Self {
            extension_available: false,
            extension_enabled: false,
            max_planes: 1,
        }
    }
}

impl ModifierCapabilities {
    /// Query modifier capabilities from the Vulkan physical device.
    ///
    /// # Safety
    ///
    /// - `instance` must be a valid Vulkan instance
    /// - `physical_device` must be a valid physical device from that instance
    pub unsafe fn query(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Self {
        // Check if extension is available
        let extension_available = Self::check_extension_available(instance, physical_device);

        if !extension_available {
            debug!("VK_EXT_image_drm_format_modifier not available");
            return Self::default();
        }

        debug!("VK_EXT_image_drm_format_modifier is available");

        Self {
            extension_available,
            extension_enabled: false, // Will be set true when device is created with extension
            max_planes: 4,            // Typical maximum for multi-planar formats
        }
    }

    /// Check if VK_EXT_image_drm_format_modifier is available.
    unsafe fn check_extension_available(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> bool {
        let extensions = match instance.enumerate_device_extension_properties(physical_device) {
            Ok(ext) => ext,
            Err(e) => {
                warn!("Failed to enumerate device extensions: {:?}", e);
                return false;
            }
        };

        let target_name = ash::ext::image_drm_format_modifier::NAME;

        extensions.iter().any(|ext| {
            let name = CStr::from_ptr(ext.extension_name.as_ptr());
            name == target_name
        })
    }

    /// Check if modifiers are supported (extension available and enabled).
    pub fn supports_modifiers(&self) -> bool {
        self.extension_available && self.extension_enabled
    }
}

/// Properties of a single DRM format modifier.
#[derive(Debug, Clone)]
pub struct ModifierProperties {
    /// The DRM format modifier value.
    pub modifier: u64,

    /// Number of memory planes for this modifier.
    pub plane_count: u32,

    /// Vulkan format features supported with this modifier.
    pub format_features: vk::FormatFeatureFlags,
}

impl ModifierProperties {
    /// Check if this modifier supports being sampled as a texture.
    pub fn supports_sampling(&self) -> bool {
        self.format_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE)
    }

    /// Check if this modifier supports being used as a transfer source.
    pub fn supports_transfer_src(&self) -> bool {
        self.format_features
            .contains(vk::FormatFeatureFlags::TRANSFER_SRC)
    }

    /// Check if this modifier supports being rendered to.
    pub fn supports_color_attachment(&self) -> bool {
        self.format_features
            .contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT)
    }

    /// Get a human-readable description of this modifier.
    pub fn describe(&self) -> String {
        let vendor = self.modifier & drm_mod::VENDOR_MASK;
        let vendor_name = match vendor {
            drm_mod::VENDOR_NONE => "generic",
            drm_mod::VENDOR_INTEL => "Intel",
            drm_mod::VENDOR_AMD => "AMD",
            drm_mod::VENDOR_NVIDIA => "NVIDIA",
            drm_mod::VENDOR_SAMSUNG => "Samsung",
            drm_mod::VENDOR_QCOM => "Qualcomm",
            drm_mod::VENDOR_ARM => "ARM",
            drm_mod::VENDOR_BROADCOM => "Broadcom",
            _ => "unknown",
        };

        if self.modifier == drm_mod::LINEAR {
            "LINEAR".to_string()
        } else if self.modifier == drm_mod::INVALID {
            "INVALID".to_string()
        } else {
            format!(
                "{}:{:#x}",
                vendor_name,
                self.modifier & !drm_mod::VENDOR_MASK
            )
        }
    }
}

/// Query supported modifiers for a Vulkan format.
///
/// This uses `vkGetPhysicalDeviceFormatProperties2` with
/// `VkDrmFormatModifierPropertiesListEXT` to query the actual
/// supported modifiers for a given format.
///
/// # Safety
///
/// - `instance` must be a valid Vulkan instance with VK_KHR_get_physical_device_properties2
/// - `physical_device` must be a valid physical device from that instance
/// - The device should have VK_EXT_image_drm_format_modifier available
///
/// # Returns
///
/// A vector of `ModifierProperties` describing supported modifiers, or an empty
/// vector if the extension is not available or the format is not supported.
pub unsafe fn query_format_modifiers(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    vk_format: vk::Format,
) -> Vec<ModifierProperties> {
    // First, query to get the count of modifiers
    let mut modifier_list = vk::DrmFormatModifierPropertiesListEXT::default();
    let mut format_props2 = vk::FormatProperties2::default().push_next(&mut modifier_list);

    instance.get_physical_device_format_properties2(physical_device, vk_format, &mut format_props2);

    let modifier_count = modifier_list.drm_format_modifier_count;

    if modifier_count == 0 {
        trace!("No modifiers for format {:?}", vk_format);
        return vec![];
    }

    // Allocate space and query again
    let mut modifier_props: Vec<vk::DrmFormatModifierPropertiesEXT> =
        vec![vk::DrmFormatModifierPropertiesEXT::default(); modifier_count as usize];

    let mut modifier_list = vk::DrmFormatModifierPropertiesListEXT::default()
        .drm_format_modifier_properties(&mut modifier_props);
    let mut format_props2 = vk::FormatProperties2::default().push_next(&mut modifier_list);

    instance.get_physical_device_format_properties2(physical_device, vk_format, &mut format_props2);

    // Convert to our type
    let result: Vec<ModifierProperties> = modifier_props
        .iter()
        .map(|p| ModifierProperties {
            modifier: p.drm_format_modifier,
            plane_count: p.drm_format_modifier_plane_count,
            format_features: p.drm_format_modifier_tiling_features,
        })
        .collect();

    trace!(
        "Format {:?}: {} modifiers supported",
        vk_format,
        result.len()
    );

    for props in &result {
        trace!(
            "  - {} (planes={}, features={:?})",
            props.describe(),
            props.plane_count,
            props.format_features
        );
    }

    result
}

/// Check if a specific modifier is supported for a format.
///
/// # Safety
///
/// Same as `query_format_modifiers`.
pub unsafe fn is_modifier_supported(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    vk_format: vk::Format,
    modifier: u64,
) -> Option<ModifierProperties> {
    let modifiers = query_format_modifiers(instance, physical_device, vk_format);
    modifiers.into_iter().find(|m| m.modifier == modifier)
}

/// Plane layout information for modifier-based image creation.
#[derive(Debug, Clone, Default)]
pub struct ModifierPlaneLayout {
    /// Offset in bytes from the start of the memory allocation.
    pub offset: u64,
    /// Stride (bytes per row) for this plane.
    pub row_pitch: u64,
    /// Size in bytes of this plane.
    pub size: u64,
    /// Array pitch (for array images).
    pub array_pitch: u64,
    /// Depth pitch (for 3D images).
    pub depth_pitch: u64,
}

impl From<&ModifierPlaneLayout> for vk::SubresourceLayout {
    fn from(layout: &ModifierPlaneLayout) -> Self {
        vk::SubresourceLayout {
            offset: layout.offset,
            size: layout.size,
            row_pitch: layout.row_pitch,
            array_pitch: layout.array_pitch,
            depth_pitch: layout.depth_pitch,
        }
    }
}

/// Image create info extension for explicit modifier.
///
/// Helper to build `VkImageDrmFormatModifierExplicitCreateInfoEXT`.
#[derive(Debug)]
pub struct ExplicitModifierCreateInfo {
    /// The DRM format modifier to use.
    pub modifier: u64,
    /// Plane layouts (one per plane in the modifier).
    pub plane_layouts: Vec<ModifierPlaneLayout>,
}

impl ExplicitModifierCreateInfo {
    /// Create info for a single-plane modifier.
    pub fn single_plane(modifier: u64, offset: u64, row_pitch: u64) -> Self {
        Self {
            modifier,
            plane_layouts: vec![ModifierPlaneLayout {
                offset,
                row_pitch,
                size: 0, // Vulkan ignores this for image creation
                array_pitch: 0,
                depth_pitch: 0,
            }],
        }
    }

    /// Create info for a multi-plane modifier (e.g., NV12).
    pub fn multi_plane(modifier: u64, plane_offsets: &[u64], plane_strides: &[u64]) -> Self {
        assert_eq!(plane_offsets.len(), plane_strides.len());

        let plane_layouts = plane_offsets
            .iter()
            .zip(plane_strides.iter())
            .map(|(&offset, &row_pitch)| ModifierPlaneLayout {
                offset,
                row_pitch,
                size: 0,
                array_pitch: 0,
                depth_pitch: 0,
            })
            .collect();

        Self {
            modifier,
            plane_layouts,
        }
    }
}

/// Query the DRM format modifier of an image.
///
/// # Safety
///
/// - `instance` must be a valid Vulkan instance
/// - `device` must be a valid Vulkan device with VK_EXT_image_drm_format_modifier enabled
/// - `image` must be a valid image created with DRM_FORMAT_MODIFIER_EXT tiling
pub unsafe fn get_image_modifier(
    instance: &ash::Instance,
    device: &ash::Device,
    image: vk::Image,
) -> Result<u64, vk::Result> {
    let drm_modifier_fn = ash::ext::image_drm_format_modifier::Device::new(instance, device);

    let mut modifier_props = vk::ImageDrmFormatModifierPropertiesEXT::default();
    drm_modifier_fn.get_image_drm_format_modifier_properties(image, &mut modifier_props)?;

    Ok(modifier_props.drm_format_modifier)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modifier_describe_linear() {
        let props = ModifierProperties {
            modifier: drm_mod::LINEAR,
            plane_count: 1,
            format_features: vk::FormatFeatureFlags::SAMPLED_IMAGE,
        };
        assert_eq!(props.describe(), "LINEAR");
    }

    #[test]
    fn test_modifier_describe_invalid() {
        let props = ModifierProperties {
            modifier: drm_mod::INVALID,
            plane_count: 0,
            format_features: vk::FormatFeatureFlags::empty(),
        };
        assert_eq!(props.describe(), "INVALID");
    }

    #[test]
    fn test_modifier_describe_vendor() {
        let props = ModifierProperties {
            modifier: drm_mod::VENDOR_INTEL | 0x1,
            plane_count: 1,
            format_features: vk::FormatFeatureFlags::SAMPLED_IMAGE,
        };
        assert!(props.describe().contains("Intel"));
    }

    #[test]
    fn test_modifier_supports_sampling() {
        let props = ModifierProperties {
            modifier: drm_mod::LINEAR,
            plane_count: 1,
            format_features: vk::FormatFeatureFlags::SAMPLED_IMAGE
                | vk::FormatFeatureFlags::TRANSFER_SRC,
        };
        assert!(props.supports_sampling());
        assert!(props.supports_transfer_src());
        assert!(!props.supports_color_attachment());
    }

    #[test]
    fn test_explicit_modifier_single_plane() {
        let info = ExplicitModifierCreateInfo::single_plane(drm_mod::LINEAR, 0, 1920 * 4);
        assert_eq!(info.modifier, drm_mod::LINEAR);
        assert_eq!(info.plane_layouts.len(), 1);
        assert_eq!(info.plane_layouts[0].row_pitch, 1920 * 4);
    }

    #[test]
    fn test_explicit_modifier_multi_plane() {
        let offsets = [0, 1920 * 1080];
        let strides = [1920, 1920];
        let info = ExplicitModifierCreateInfo::multi_plane(drm_mod::LINEAR, &offsets, &strides);
        assert_eq!(info.plane_layouts.len(), 2);
        assert_eq!(info.plane_layouts[1].offset, 1920 * 1080);
    }

    #[test]
    fn test_capabilities_default() {
        let caps = ModifierCapabilities::default();
        assert!(!caps.extension_available);
        assert!(!caps.extension_enabled);
        assert!(!caps.supports_modifiers());
    }
}
