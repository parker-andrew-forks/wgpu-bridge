//! Core bridge functionality for connecting Smithay and wgpu.
//!
//! The `WgpuBridge` is the main entry point for creating wgpu resources
//! from Smithay's Vulkan context.

use crate::error::{BridgeError, ImportError};
use crate::modifiers::{
    self, drm_mod, ExplicitModifierCreateInfo, ModifierCapabilities, ModifierProperties,
};
use crate::multiplanar::MultiPlanarCapabilities;
#[cfg(feature = "dmabuf")]
use crate::multiplanar::{
    self, is_multiplanar_format, multiplanar_format, MultiPlanarTexture, YcbcrColorspace,
};
use crate::sync::SyncCapabilities;
use crate::texture::WgpuTexture;
use ash::vk;
use smithay::backend::allocator::dmabuf::Dmabuf;
#[cfg(feature = "dmabuf")]
use smithay::backend::allocator::Buffer as AllocatorBuffer;
#[cfg(feature = "dmabuf")]
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, info, trace};
use wgpu_hal::Instance as HalInstance; // For enumerate_adapters trait method

/// The bridge between Smithay's Vulkan infrastructure and wgpu.
///
/// This struct manages the wgpu device and queue created from Smithay's
/// Vulkan instance, enabling wgpu to act as a "guest renderer".
pub struct WgpuBridge {
    /// The wgpu instance (wraps Smithay's Vulkan instance)
    #[allow(dead_code)]
    pub instance: wgpu::Instance,
    /// The wgpu adapter
    adapter: wgpu::Adapter,
    /// The wgpu device
    device: wgpu::Device,
    /// The wgpu queue
    queue: wgpu::Queue,
    /// Supported dmabuf formats (queried from Vulkan)
    supported_formats: Vec<SupportedFormat>,
    /// Sync capabilities of this device
    sync_capabilities: SyncCapabilities,
    /// DRM format modifier capabilities
    modifier_capabilities: ModifierCapabilities,
    /// Multi-planar format capabilities (NV12, P010, etc.)
    multiplanar_capabilities: MultiPlanarCapabilities,
    /// Frame counter for tracking submissions
    frame_counter: AtomicU64,
    /// Last completed frame (approximation via polling)
    last_completed_frame: AtomicU64,
    /// Owned Vulkan objects (for new_with_explicit_sync mode)
    /// MUST BE LAST so they are destroyed after all wgpu objects
    owned_vulkan: Option<OwnedVulkanContext>,
}

/// Vulkan objects owned by WgpuBridge (when using new_with_explicit_sync).
///
/// # Lifecycle Warning
///
/// This struct exists only for the `new_with_explicit_sync()` **testing** path.
/// In production use with `from_smithay_vulkan()`, Smithay owns the Vulkan
/// objects and this struct is not used (owned_vulkan = None).
///
/// # Known Limitation: Vulkan Objects Are Leaked
///
/// We intentionally DO NOT destroy the Vulkan device/instance here because
/// wgpu-hal retains internal references that outlive our Drop. Attempting
/// to destroy them causes SIGSEGV.
///
/// This leak is acceptable because:
/// - `new_with_explicit_sync()` is for testing, not production
/// - Test processes are short-lived; the OS cleans up on exit
/// - The production path (`from_smithay_vulkan`) doesn't have this issue
///
/// # Proper Fix (Future)
///
/// To properly clean up, `new_with_explicit_sync()` should use wgpu-hal's
/// `DropCallback` mechanism to destroy Vulkan objects only after wgpu is
/// completely done. This requires restructuring to not use `from_smithay_vulkan_full()`.
struct OwnedVulkanContext {
    #[allow(dead_code)]
    device: ash::Device,
    #[allow(dead_code)]
    instance: ash::Instance,
}

impl Drop for OwnedVulkanContext {
    fn drop(&mut self) {
        // Intentionally leak Vulkan objects - see struct documentation.
        // wgpu-hal retains internal references, and destroying these causes SIGSEGV.
        debug!(
            "OwnedVulkanContext dropped: Vulkan objects intentionally leaked \
             (test-mode only, cleaned up by OS on process exit)"
        );
    }
}

impl Drop for WgpuBridge {
    fn drop(&mut self) {
        // When we own the Vulkan context (new_with_explicit_sync test mode),
        // ensure GPU work is complete before cleanup.
        if self.owned_vulkan.is_some() {
            debug!("WgpuBridge drop: waiting for GPU work to complete");
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        }
        // Fields drop in declaration order:
        // 1. wgpu objects (instance, adapter, device, queue)
        // 2. owned_vulkan (leaked - see OwnedVulkanContext docs)
    }
}

/// A supported dmabuf format with its modifiers.
#[derive(Debug, Clone)]
pub struct SupportedFormat {
    /// DRM fourcc format code
    pub fourcc: drm_fourcc::DrmFourcc,
    /// Supported modifiers for this format (with detailed properties)
    pub modifier_props: Vec<ModifierProperties>,
    /// Supported modifier values (for quick lookup)
    pub modifiers: Vec<u64>,
    /// Corresponding wgpu texture format
    pub wgpu_format: wgpu::TextureFormat,
    /// Corresponding Vulkan format
    pub vk_format: vk::Format,
}

/// DRM format modifier indicating "invalid" (use implicit modifier)
#[cfg(feature = "dmabuf")]
const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;

/// Linear modifier (no tiling)
#[cfg(feature = "dmabuf")]
const DRM_FORMAT_MOD_LINEAR: u64 = 0;

/// Result type for auto-routed dmabuf import.
///
/// This enum allows callers to handle both single-plane and multi-planar imports
/// uniformly while still accessing the appropriate texture type.
#[cfg(feature = "dmabuf")]
#[derive(Debug)]
pub enum ImportedDmabuf {
    /// Single-plane texture (ARGB, XRGB, etc.) that can be used directly with wgpu
    SinglePlane(WgpuTexture),
    /// Multi-planar YUV texture (NV12, P010, etc.) with Vulkan-level resources
    MultiPlanar(MultiPlanarTexture),
}

#[cfg(feature = "dmabuf")]
impl ImportedDmabuf {
    /// Returns `true` if this is a single-plane texture.
    pub fn is_single_plane(&self) -> bool {
        matches!(self, ImportedDmabuf::SinglePlane(_))
    }

    /// Returns `true` if this is a multi-planar YUV texture.
    pub fn is_multi_planar(&self) -> bool {
        matches!(self, ImportedDmabuf::MultiPlanar(_))
    }

    /// Get the single-plane texture, if this is one.
    pub fn as_single_plane(&self) -> Option<&WgpuTexture> {
        match self {
            ImportedDmabuf::SinglePlane(t) => Some(t),
            _ => None,
        }
    }

    /// Get the multi-planar texture, if this is one.
    pub fn as_multi_planar(&self) -> Option<&MultiPlanarTexture> {
        match self {
            ImportedDmabuf::MultiPlanar(t) => Some(t),
            _ => None,
        }
    }
}

impl WgpuBridge {
    /// Create a new WgpuBridge using wgpu's standard instance creation.
    ///
    /// This is the simple path that doesn't share Vulkan context with Smithay.
    /// For full integration, use `from_smithay_vulkan` instead.
    pub fn new() -> Result<Self, BridgeError> {
        info!("Creating WgpuBridge with new Vulkan instance");

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| BridgeError::AdapterCreation(format!("{:?}", e)))?;

        info!("Using adapter: {}", adapter.get_info().name);

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("lamco-wgpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            }))?;

        let supported_formats = Self::query_supported_formats(&adapter);

        // No sync manager in simple mode (would need Vulkan context sharing)
        let sync_capabilities = SyncCapabilities::default();
        let modifier_capabilities = ModifierCapabilities::default();
        let multiplanar_capabilities = MultiPlanarCapabilities::default();

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            supported_formats,
            sync_capabilities,
            modifier_capabilities,
            multiplanar_capabilities,
            frame_counter: AtomicU64::new(0),
            last_completed_frame: AtomicU64::new(0),
            owned_vulkan: None, // Simple mode doesn't own Vulkan objects
        })
    }

    /// Create a WgpuBridge with explicit sync extensions enabled.
    ///
    /// This creates a standalone Vulkan context (like `new()`) but enables
    /// the external semaphore extensions needed for explicit sync testing.
    ///
    /// Use this for testing explicit sync functionality without a full
    /// Smithay compositor setup.
    pub fn new_with_explicit_sync() -> Result<Self, BridgeError> {
        use std::ffi::CStr;

        info!("Creating WgpuBridge with explicit sync extensions");

        // SAFETY: Loading the Vulkan library is safe if a Vulkan driver is installed.
        let entry = unsafe { ash::Entry::load() }.map_err(|e| {
            BridgeError::InstanceCreation(format!("failed to load Vulkan: {:?}", e))
        })?;

        let instance_extensions: Vec<*const i8> = vec![
            ash::khr::get_physical_device_properties2::NAME.as_ptr(),
            ash::khr::external_memory_capabilities::NAME.as_ptr(),
            ash::khr::external_semaphore_capabilities::NAME.as_ptr(),
            ash::khr::wayland_surface::NAME.as_ptr(),
        ];

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"lamco-wgpu")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"wgpu")
            .engine_version(vk::make_api_version(0, 28, 0, 0))
            .api_version(vk::make_api_version(0, 1, 2, 0)); // Vulkan 1.2 for timeline semaphores

        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions);

        // SAFETY: instance_create_info is valid and outlives the call.
        let vk_instance =
            unsafe { entry.create_instance(&instance_create_info, None) }.map_err(|e| {
                BridgeError::InstanceCreation(format!("failed to create Vulkan instance: {:?}", e))
            })?;

        // SAFETY: vk_instance is valid.
        let physical_devices =
            unsafe { vk_instance.enumerate_physical_devices() }.map_err(|e| {
                BridgeError::AdapterCreation(format!("failed to enumerate devices: {:?}", e))
            })?;

        if physical_devices.is_empty() {
            return Err(BridgeError::AdapterCreation(
                "No Vulkan devices found".into(),
            ));
        }

        // Select first discrete GPU, or first device if none
        let vk_physical_device = physical_devices
            .iter()
            .find(|&&pd| {
                // SAFETY: vk_instance and pd are valid.
                let props = unsafe { vk_instance.get_physical_device_properties(pd) };
                props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .copied()
            .unwrap_or(physical_devices[0]);

        // SAFETY: vk_instance and vk_physical_device are valid.
        let device_props =
            unsafe { vk_instance.get_physical_device_properties(vk_physical_device) };
        // SAFETY: device_name is null-terminated by Vulkan spec.
        let device_name =
            unsafe { CStr::from_ptr(device_props.device_name.as_ptr()).to_string_lossy() };
        info!("Selected Vulkan device: {}", device_name);

        // SAFETY: vk_instance and vk_physical_device are valid.
        let queue_families =
            unsafe { vk_instance.get_physical_device_queue_family_properties(vk_physical_device) };

        let queue_family_index = queue_families
            .iter()
            .position(|qf| qf.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or_else(|| BridgeError::AdapterCreation("No graphics queue family".into()))?
            as u32;

        // Required device extensions for explicit sync
        let device_extensions: Vec<*const i8> = vec![
            ash::khr::swapchain::NAME.as_ptr(),
            ash::khr::maintenance1::NAME.as_ptr(),
            ash::khr::maintenance2::NAME.as_ptr(),
            ash::khr::multiview::NAME.as_ptr(),
            ash::khr::create_renderpass2::NAME.as_ptr(),
            ash::khr::imageless_framebuffer::NAME.as_ptr(),
            ash::khr::external_memory::NAME.as_ptr(),
            ash::khr::external_memory_fd::NAME.as_ptr(),
            ash::ext::external_memory_dma_buf::NAME.as_ptr(),
            ash::khr::external_semaphore::NAME.as_ptr(),
            ash::khr::external_semaphore_fd::NAME.as_ptr(),
            ash::khr::timeline_semaphore::NAME.as_ptr(),
        ];

        // Timeline semaphore features (required for Vulkan 1.2)
        let mut timeline_features =
            vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);

        // Vulkan 1.1 features
        let mut vk11_features = vk::PhysicalDeviceVulkan11Features::default().multiview(true);

        // Vulkan 1.2 features
        let mut vk12_features = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .imageless_framebuffer(true);

        let queue_priority = 1.0f32;
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(std::slice::from_ref(&queue_priority));

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info))
            .enabled_extension_names(&device_extensions)
            .push_next(&mut timeline_features)
            .push_next(&mut vk11_features)
            .push_next(&mut vk12_features);

        // SAFETY: vk_instance, vk_physical_device, and device_create_info are valid.
        let vk_device =
            unsafe { vk_instance.create_device(vk_physical_device, &device_create_info, None) }
                .map_err(|e| {
                    BridgeError::HalAccess(format!("failed to create Vulkan device: {:?}", e))
                })?;

        debug!("Created Vulkan device with explicit sync extensions");

        let instance_ext_names: Vec<&'static CStr> = vec![
            ash::khr::get_physical_device_properties2::NAME,
            ash::khr::external_memory_capabilities::NAME,
            ash::khr::external_semaphore_capabilities::NAME,
        ];

        let device_ext_names: Vec<&'static CStr> = vec![
            ash::khr::swapchain::NAME,
            ash::khr::maintenance1::NAME,
            ash::khr::maintenance2::NAME,
            ash::khr::multiview::NAME,
            ash::khr::create_renderpass2::NAME,
            ash::khr::imageless_framebuffer::NAME,
            ash::khr::external_memory::NAME,
            ash::khr::external_memory_fd::NAME,
            ash::ext::external_memory_dma_buf::NAME,
            ash::khr::external_semaphore::NAME,
            ash::khr::external_semaphore_fd::NAME,
            ash::khr::timeline_semaphore::NAME,
        ];

        let owned_device = vk_device.clone();
        let owned_instance = vk_instance.clone();

        // SAFETY: All Vulkan handles are valid and we pass the correct extensions.
        let mut bridge = unsafe {
            Self::from_smithay_vulkan_full(
                &vk_instance,
                vk_physical_device,
                &vk_device,
                queue_family_index,
                0, // queue_index
                &instance_ext_names,
                &device_ext_names,
            )
        }?;

        // Set owned_vulkan so they get destroyed after wgpu objects
        bridge.owned_vulkan = Some(OwnedVulkanContext {
            device: owned_device,
            instance: owned_instance,
        });

        Ok(bridge)
    }

    /// Create a WgpuBridge from Smithay's existing Vulkan instance.
    ///
    /// This is the "true guest renderer" mode where wgpu shares Smithay's
    /// Vulkan context, enabling efficient texture sharing without copies.
    ///
    /// # Safety
    ///
    /// - `vk_instance` must be a valid Vulkan instance
    /// - `vk_physical_device` must be a valid physical device from that instance
    /// - `vk_device` must be a valid logical device created from `vk_physical_device`
    /// - `vk_queue` must be a valid queue from the device at `queue_family_index`
    /// - The Vulkan objects must remain valid for the lifetime of this bridge
    /// - The instance must have been created with required wgpu extensions enabled
    /// - The device must have been created with required wgpu extensions enabled
    ///
    /// # Required Instance Extensions
    ///
    /// - VK_KHR_get_physical_device_properties2
    /// - VK_KHR_external_memory_capabilities
    /// - VK_KHR_external_semaphore_capabilities
    ///
    /// # Required Device Extensions
    ///
    /// - VK_KHR_external_memory
    /// - VK_KHR_external_memory_fd
    /// - VK_EXT_external_memory_dma_buf
    /// - VK_KHR_external_semaphore
    /// - VK_KHR_external_semaphore_fd
    /// - VK_KHR_timeline_semaphore (or Vulkan 1.2+)
    /// - VK_KHR_maintenance1
    /// - VK_KHR_maintenance2
    /// - VK_KHR_multiview
    /// - VK_KHR_create_renderpass2
    /// - VK_KHR_imageless_framebuffer
    ///
    /// # Enabled Instance Extensions List
    ///
    /// The `enabled_instance_extensions` parameter must list all instance extensions
    /// that were enabled when creating `vk_instance`.
    ///
    /// # Enabled Device Extensions List
    ///
    /// The `enabled_device_extensions` parameter must list all device extensions
    /// that were enabled when creating `vk_device`.
    pub unsafe fn from_smithay_vulkan(
        vk_instance: &ash::Instance,
        vk_physical_device: vk::PhysicalDevice,
        vk_device: &ash::Device,
        _vk_queue: vk::Queue,
        queue_family_index: u32,
    ) -> Result<Self, BridgeError> {
        Self::from_smithay_vulkan_full(
            vk_instance,
            vk_physical_device,
            vk_device,
            queue_family_index,
            0,   // queue_index
            &[], // Use default instance extensions
            &[], // Use default device extensions
        )
    }

    /// Create a WgpuBridge with full control over extension lists.
    ///
    /// This is the advanced version that allows specifying exactly which
    /// extensions were enabled on the Vulkan instance and device.
    ///
    /// # Safety
    ///
    /// Same as `from_smithay_vulkan`, plus:
    /// - `enabled_instance_extensions` must exactly match extensions enabled on instance
    /// - `enabled_device_extensions` must exactly match extensions enabled on device
    pub unsafe fn from_smithay_vulkan_full(
        vk_instance: &ash::Instance,
        vk_physical_device: vk::PhysicalDevice,
        vk_device: &ash::Device,
        queue_family_index: u32,
        queue_index: u32,
        enabled_instance_extensions: &[&'static std::ffi::CStr],
        enabled_device_extensions: &[&'static std::ffi::CStr],
    ) -> Result<Self, BridgeError> {
        use std::ffi::CStr;

        info!("Creating WgpuBridge with full Vulkan context sharing");

        let physical_device_properties =
            vk_instance.get_physical_device_properties(vk_physical_device);
        let api_version = physical_device_properties.api_version;

        let device_name =
            CStr::from_ptr(physical_device_properties.device_name.as_ptr()).to_string_lossy();

        debug!(
            "Sharing Vulkan device: {} (API version {}.{}.{})",
            device_name,
            vk::api_version_major(api_version),
            vk::api_version_minor(api_version),
            vk::api_version_patch(api_version),
        );

        // Determine instance extensions - use provided or query for minimum required
        let instance_extensions: Vec<&'static CStr> = if enabled_instance_extensions.is_empty() {
            // Minimum required for wgpu-hal
            vec![
                ash::khr::get_physical_device_properties2::NAME,
                ash::khr::external_memory_capabilities::NAME,
                ash::khr::external_semaphore_capabilities::NAME,
            ]
        } else {
            enabled_instance_extensions.to_vec()
        };

        // SAFETY: vk_instance and vk_physical_device are valid Vulkan handles.
        let has_modifier_ext = unsafe {
            ModifierCapabilities::query(vk_instance, vk_physical_device).extension_available
        };

        // Determine device extensions - use provided or query for minimum required
        let device_extensions: Vec<&'static CStr> = if enabled_device_extensions.is_empty() {
            // Minimum required for wgpu-hal + external memory
            let mut exts = vec![
                ash::khr::swapchain::NAME,
                ash::khr::maintenance1::NAME,
                ash::khr::maintenance2::NAME,
                ash::khr::multiview::NAME,
                ash::khr::create_renderpass2::NAME,
                ash::khr::imageless_framebuffer::NAME,
                ash::khr::external_memory::NAME,
                ash::khr::external_memory_fd::NAME,
                ash::ext::external_memory_dma_buf::NAME,
                ash::khr::external_semaphore::NAME,
                ash::khr::external_semaphore_fd::NAME,
                ash::khr::timeline_semaphore::NAME,
            ];

            // Add DRM format modifier extension if available
            if has_modifier_ext {
                exts.push(ash::ext::image_drm_format_modifier::NAME);
                debug!("Adding VK_EXT_image_drm_format_modifier to device extensions");
            }

            exts
        } else {
            enabled_device_extensions.to_vec()
        };

        debug!(
            "Using {} instance extensions, {} device extensions",
            instance_extensions.len(),
            device_extensions.len()
        );

        // SAFETY: loading from the same Vulkan library that Smithay uses
        let entry = ash::Entry::load().map_err(|e| {
            BridgeError::InstanceCreation(format!("failed to load Vulkan entry: {:?}", e))
        })?;

        let raw_instance = vk_instance.clone();

        let hal_instance = wgpu_hal::vulkan::Instance::from_raw(
            entry,
            raw_instance,
            api_version,
            0,    // android_sdk_version
            None, // debug_utils_create_info
            instance_extensions.clone(),
            wgpu_types::InstanceFlags::empty(),
            wgpu_types::MemoryBudgetThresholds::default(),
            false, // has_nv_optimus
            None,  // drop_callback - Smithay owns the instance
        )
        .map_err(|e| {
            BridgeError::InstanceCreation(format!("failed to create HAL instance: {:?}", e))
        })?;

        debug!("Created wgpu-hal Instance from Smithay's Vulkan instance");

        let hal_adapters = hal_instance.enumerate_adapters(None);

        let hal_exposed_adapter = hal_adapters
            .into_iter()
            .find(|adapter| adapter.adapter.raw_physical_device() == vk_physical_device)
            .ok_or_else(|| {
                BridgeError::AdapterCreation(format!(
                    "Could not find HAL adapter matching physical device {:?}",
                    vk_physical_device
                ))
            })?;

        debug!(
            "Found matching HAL adapter: {}",
            hal_exposed_adapter.info.name
        );

        let instance = wgpu::Instance::from_hal::<wgpu_hal::api::Vulkan>(hal_instance);
        let adapter =
            instance.create_adapter_from_hal::<wgpu_hal::api::Vulkan>(hal_exposed_adapter);

        debug!("Created wgpu Adapter from HAL adapter");

        let hal_adapter_guard = adapter
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or_else(|| BridgeError::HalAccess("Failed to access HAL adapter".into()))?;

        let raw_device = vk_device.clone();
        let hal_open_device = hal_adapter_guard
            .device_from_raw(
                raw_device,
                None, // drop_callback - Smithay owns the device
                &device_extensions,
                wgpu::Features::empty(),
                &wgpu::MemoryHints::Performance,
                queue_family_index,
                queue_index,
            )
            .map_err(|e| BridgeError::HalAccess(format!("failed to create HAL device: {:?}", e)))?;

        debug!("Created wgpu-hal Device from Smithay's Vulkan device");

        drop(hal_adapter_guard);

        let (device, queue) = adapter.create_device_from_hal::<wgpu_hal::api::Vulkan>(
            hal_open_device,
            &wgpu::DeviceDescriptor {
                label: Some("lamco-wgpu-shared"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            },
        )?;

        info!(
            "Successfully created WgpuBridge with full Vulkan context sharing on {}",
            adapter.get_info().name
        );

        let supported_formats = Self::query_supported_formats(&adapter);

        // SAFETY: vk_instance and vk_physical_device are valid handles from Smithay.
        let sync_capabilities = unsafe { SyncCapabilities::query(vk_instance, vk_physical_device) };
        // SAFETY: vk_instance and vk_physical_device are valid handles from Smithay.
        let mut modifier_capabilities =
            unsafe { ModifierCapabilities::query(vk_instance, vk_physical_device) };

        if modifier_capabilities.extension_available {
            modifier_capabilities.extension_enabled =
                device_extensions.contains(&ash::ext::image_drm_format_modifier::NAME);

            if modifier_capabilities.extension_enabled {
                info!("VK_EXT_image_drm_format_modifier enabled for tiled dmabuf import");
            } else {
                debug!(
                    "VK_EXT_image_drm_format_modifier available but not enabled in device extensions"
                );
            }
        }

        // SAFETY: vk_instance and vk_physical_device are valid handles from Smithay.
        let multiplanar_capabilities =
            unsafe { MultiPlanarCapabilities::query(vk_instance, vk_physical_device) };

        if multiplanar_capabilities.has_ycbcr_conversion {
            info!(
                "Multi-planar YUV support available: {} formats (NV12, P010, etc.)",
                multiplanar_capabilities.supported_formats.len()
            );
        }

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            supported_formats,
            sync_capabilities,
            modifier_capabilities,
            multiplanar_capabilities,
            frame_counter: AtomicU64::new(0),
            last_completed_frame: AtomicU64::new(0),
            owned_vulkan: None, // Smithay owns Vulkan objects
        })
    }

    /// Import a dmabuf as a wgpu texture.
    ///
    /// This uses raw Vulkan via ash to import the dmabuf, then wraps it
    /// using wgpu-hal's `texture_from_raw`.
    ///
    /// # Safety
    ///
    /// - The dmabuf file descriptor must remain valid until the returned texture is dropped.
    /// - The dmabuf must not be modified while the texture is in use by the GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The dmabuf format is not supported
    /// - The Vulkan device doesn't support VK_KHR_external_memory_fd
    /// - The import operation fails
    #[cfg(feature = "dmabuf")]
    pub unsafe fn import_dmabuf(&self, dmabuf: &Dmabuf) -> Result<WgpuTexture, ImportError> {
        use std::ops::Deref;

        // Get dmabuf properties
        let size = AllocatorBuffer::size(dmabuf);
        let width = size.w as u32;
        let height = size.h as u32;

        if width == 0 || height == 0 {
            return Err(ImportError::InvalidDimensions { width, height });
        }

        let wgpu_format = self.dmabuf_to_wgpu_format(dmabuf)?;
        let drm_format = AllocatorBuffer::format(dmabuf);

        // Get plane information
        let mut planes: Vec<_> = dmabuf
            .handles()
            .zip(dmabuf.offsets())
            .zip(dmabuf.strides())
            .map(|((handle, offset), stride)| (handle.as_raw_fd(), offset, stride))
            .collect();

        if planes.is_empty() {
            return Err(ImportError::InvalidPlanes("No planes in dmabuf".into()));
        }

        // For now, only support single-plane formats
        let (fd, offset, stride) = planes.remove(0);

        // Dup the fd because vkImportMemoryFdKHR takes ownership.
        // The original fd remains owned by the Dmabuf.
        // SAFETY: fd is a valid file descriptor from the dmabuf.
        let import_fd = unsafe {
            let duped = libc::dup(fd);
            if duped < 0 {
                return Err(ImportError::FdImport("failed to dup dmabuf fd".into()));
            }
            duped
        };

        // Get the modifier
        let modifier: u64 = drm_format.modifier.into();
        let modifier = if modifier == DRM_FORMAT_MOD_INVALID {
            DRM_FORMAT_MOD_LINEAR
        } else {
            modifier
        };

        debug!(
            "Importing dmabuf: {}x{}, format={:?}, modifier={:#x}, fd={}, stride={}, offset={}",
            width, height, drm_format.code, modifier, fd, stride, offset
        );

        // Access the wgpu-hal Vulkan device
        let hal_device_guard = self
            .device
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or(ImportError::BridgeUnavailable)?;
        let hal_device: &wgpu_hal::vulkan::Device = hal_device_guard.deref();

        // Get raw Vulkan handles
        let raw_device = hal_device.raw_device();
        let physical_device = hal_device.raw_physical_device();

        // Map wgpu format to Vulkan format
        let vk_format = self.wgpu_to_vk_format(wgpu_format)?;

        // Create the VkImage with external memory support and DRM format modifier
        let mut external_memory_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

        // Build plane layout for the modifier
        // For single-plane formats, we have one subresource layout
        let plane_layout = vk::SubresourceLayout {
            offset: offset as u64,
            size: 0, // Size is computed by the driver for DRM modifier images
            row_pitch: stride as u64,
            array_pitch: 0,
            depth_pitch: 0,
        };
        let plane_layouts = [plane_layout];

        // Use DRM format modifier tiling with explicit modifier info
        let mut modifier_explicit_info = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
            .drm_format_modifier(modifier)
            .plane_layouts(&plane_layouts);

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
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external_memory_info)
            .push_next(&mut modifier_explicit_info);

        // SAFETY: raw_device is valid and image_create_info is properly initialized.
        let vk_image = unsafe { raw_device.create_image(&image_create_info, None) }
            .map_err(|e| ImportError::ImageCreation(format!("vkCreateImage failed: {:?}", e)))?;

        // SAFETY: raw_device and vk_image are valid.
        let mem_requirements = unsafe { raw_device.get_image_memory_requirements(vk_image) };

        // SAFETY: physical_device is valid.
        let mem_properties =
            unsafe { self.get_physical_device_memory_properties(physical_device)? };

        let memory_type_index = self
            .find_memory_type_index(
                &mem_properties,
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .ok_or_else(|| {
                // SAFETY: raw_device and vk_image are valid.
                unsafe { raw_device.destroy_image(vk_image, None) };
                ImportError::MemoryAllocation("no suitable memory type found".into())
            })?;

        // Import the dmabuf fd (using the duped fd which Vulkan will take ownership of)
        let mut import_memory_fd_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(import_fd);

        // Dedicated allocation for external memory
        let mut dedicated_alloc_info = vk::MemoryDedicatedAllocateInfo::default().image(vk_image);

        let memory_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index as u32)
            .push_next(&mut import_memory_fd_info)
            .push_next(&mut dedicated_alloc_info);

        // SAFETY: raw_device is valid and memory_allocate_info is properly initialized.
        let vk_memory = unsafe { raw_device.allocate_memory(&memory_allocate_info, None) }
            .map_err(|e| {
                // SAFETY: raw_device and vk_image are valid.
                unsafe { raw_device.destroy_image(vk_image, None) };
                ImportError::MemoryAllocation(format!("vkAllocateMemory failed: {:?}", e))
            })?;

        // SAFETY: raw_device, vk_image, and vk_memory are valid.
        unsafe { raw_device.bind_image_memory(vk_image, vk_memory, 0) }.map_err(|e| {
            // SAFETY: raw_device and vk_memory are valid.
            unsafe { raw_device.free_memory(vk_memory, None) };
            // SAFETY: raw_device and vk_image are valid.
            unsafe { raw_device.destroy_image(vk_image, None) };
            ImportError::MemoryAllocation(format!("vkBindImageMemory failed: {:?}", e))
        })?;

        // Create the HAL texture descriptor
        let hal_desc = wgpu_hal::TextureDescriptor {
            label: Some("dmabuf-import"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu_types::TextureUses::RESOURCE | wgpu_types::TextureUses::COPY_SRC,
            memory_flags: wgpu_hal::MemoryFlags::empty(),
            view_formats: vec![],
        };

        // SAFETY: vk_image and vk_memory are valid Vulkan handles we just created,
        // and hal_desc accurately describes the image properties.
        let hal_texture = unsafe {
            hal_device.texture_from_raw(
                vk_image,
                &hal_desc,
                None,
                wgpu_hal::vulkan::TextureMemory::Dedicated(vk_memory),
            )
        };

        // Wrap the HAL texture in a wgpu Texture
        let wgpu_desc = wgpu::TextureDescriptor {
            label: Some("dmabuf-import"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };

        // SAFETY: hal_texture is a valid HAL texture and wgpu_desc matches its properties.
        let wgpu_texture = unsafe {
            self.device
                .create_texture_from_hal::<wgpu_hal::api::Vulkan>(hal_texture, &wgpu_desc)
        };

        debug!("Successfully imported dmabuf as wgpu texture");

        Ok(WgpuTexture::new(
            wgpu_texture,
            wgpu_format,
            width,
            height,
            true,
        ))
    }

    /// Import a dmabuf as a wgpu texture (stub for non-experimental builds).
    #[cfg(not(feature = "dmabuf"))]
    pub unsafe fn import_dmabuf(&self, _dmabuf: &Dmabuf) -> Result<WgpuTexture, ImportError> {
        Err(ImportError::FdImport(
            "dmabuf import requires 'dmabuf' feature (enabled by default)".into(),
        ))
    }

    /// Import a dmabuf with automatic format detection.
    ///
    /// This is the recommended entry point for dmabuf import. It automatically
    /// detects whether the buffer is single-plane (ARGB, XRGB, etc.) or
    /// multi-planar (NV12, P010, YUV420, etc.) and routes to the appropriate
    /// import function.
    ///
    /// # Returns
    ///
    /// - `ImportedDmabuf::SinglePlane` for standard formats - can be used with wgpu directly
    /// - `ImportedDmabuf::MultiPlanar` for YUV formats - requires Vulkan-level shader handling
    ///
    /// # Safety
    ///
    /// - The dmabuf file descriptors must remain valid until the texture is dropped
    /// - The dmabuf must not be modified while the texture is in use
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// match unsafe { bridge.import_dmabuf_auto(&dmabuf)? } {
    ///     ImportedDmabuf::SinglePlane(texture) => {
    ///         // Use with wgpu bind groups
    ///         let view = texture.inner().create_view(&Default::default());
    ///     }
    ///     ImportedDmabuf::MultiPlanar(yuv_texture) => {
    ///         // Use raw Vulkan sampler with YCbCr conversion
    ///         let sampler = yuv_texture.sampler;
    ///     }
    /// }
    /// ```
    #[cfg(feature = "dmabuf")]
    pub unsafe fn import_dmabuf_auto(
        &self,
        dmabuf: &Dmabuf,
    ) -> Result<ImportedDmabuf, ImportError> {
        let drm_format = AllocatorBuffer::format(dmabuf);
        let fourcc = drm_format.code;

        if is_multiplanar_format(fourcc) {
            debug!("Auto-routing {:?} to multi-planar import path", fourcc);
            let texture = self.import_multiplanar_dmabuf(dmabuf, None)?;
            Ok(ImportedDmabuf::MultiPlanar(texture))
        } else {
            let texture = self.import_dmabuf(dmabuf)?;
            Ok(ImportedDmabuf::SinglePlane(texture))
        }
    }

    /// Import a multi-planar dmabuf (NV12, P010, etc.) as a raw Vulkan texture.
    ///
    /// Multi-planar YUV formats require special handling with VkSamplerYcbcrConversion
    /// and cannot be directly wrapped as wgpu textures. This method returns a
    /// `MultiPlanarTexture` with all Vulkan resources needed for sampling.
    ///
    /// # Arguments
    ///
    /// * `dmabuf` - The multi-planar dmabuf to import
    /// * `colorspace` - Optional YCbCr colorspace (defaults to BT.601 narrow)
    ///
    /// # Safety
    ///
    /// - The dmabuf file descriptors must remain valid until the texture is dropped
    /// - The dmabuf must not be modified while the texture is in use
    /// - Caller must manage Vulkan resource destruction via `MultiPlanarTexture::destroy()`
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Import NV12 video frame with HD colorspace
    /// let colorspace = YcbcrColorspace::from_resolution(1920, 1080); // BT.709
    /// let texture = unsafe { bridge.import_multiplanar_dmabuf(&dmabuf, Some(colorspace))? };
    ///
    /// // Use with Vulkan combined image sampler in shader
    /// // ...
    ///
    /// // Clean up when done
    /// unsafe { texture.destroy(&device); }
    /// ```
    #[cfg(feature = "dmabuf")]
    pub unsafe fn import_multiplanar_dmabuf(
        &self,
        dmabuf: &Dmabuf,
        colorspace: Option<YcbcrColorspace>,
    ) -> Result<MultiPlanarTexture, ImportError> {
        use std::ops::Deref;
        use std::os::fd::AsRawFd;

        // Check multi-planar support
        if !self.multiplanar_capabilities.has_ycbcr_conversion {
            return Err(ImportError::ImageCreation(
                "Multi-planar import requires VK_KHR_sampler_ycbcr_conversion".into(),
            ));
        }

        // Get dmabuf properties
        let size = AllocatorBuffer::size(dmabuf);
        let width = size.w as u32;
        let height = size.h as u32;

        if width == 0 || height == 0 {
            return Err(ImportError::InvalidDimensions { width, height });
        }

        let drm_format = AllocatorBuffer::format(dmabuf);
        let fourcc = drm_format.code;

        // Get format descriptor
        let colorspace = colorspace
            .unwrap_or_else(|| YcbcrColorspace::from_format_and_resolution(fourcc, width, height));
        let format = multiplanar_format(fourcc).ok_or(ImportError::UnsupportedFormat(fourcc))?;

        // Check if this specific format is supported
        if !self.multiplanar_capabilities.supports_format(fourcc) {
            return Err(ImportError::UnsupportedFormat(fourcc));
        }

        // Collect plane information
        let plane_fds: Vec<i32> = dmabuf.handles().map(|h| h.as_raw_fd()).collect();
        let plane_offsets: Vec<u32> = dmabuf.offsets().collect();
        let plane_strides: Vec<u32> = dmabuf.strides().collect();

        if plane_fds.len() < format.plane_count as usize {
            return Err(ImportError::InvalidPlanes(format!(
                "Expected {} planes for {:?}, got {}",
                format.plane_count,
                fourcc,
                plane_fds.len()
            )));
        }

        debug!(
            "Importing multi-planar dmabuf: {}x{}, format={:?}, colorspace={:?}, planes={}",
            width,
            height,
            fourcc,
            colorspace,
            plane_fds.len()
        );

        // Access raw Vulkan handles
        let hal_device_guard = self
            .device
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or(ImportError::BridgeUnavailable)?;
        let hal_device: &wgpu_hal::vulkan::Device = hal_device_guard.deref();

        let raw_device = hal_device.raw_device();
        let physical_device = hal_device.raw_physical_device();

        let hal_adapter_guard = self
            .adapter
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or(ImportError::BridgeUnavailable)?;
        let hal_adapter: &wgpu_hal::vulkan::Adapter = hal_adapter_guard.deref();
        let instance = hal_adapter.shared_instance().raw_instance();

        // Use the multiplanar module's import function
        multiplanar::import_multiplanar_dmabuf(
            instance,
            raw_device,
            physical_device,
            &format,
            width,
            height,
            &plane_fds,
            &plane_offsets,
            &plane_strides,
        )
    }

    /// Import a dmabuf with explicit modifier support.
    ///
    /// This uses `VK_EXT_image_drm_format_modifier` to import dmabufs with
    /// tiled memory layouts, which provides better GPU performance than LINEAR.
    ///
    /// # Arguments
    ///
    /// * `dmabuf` - The dmabuf to import
    /// * `modifier_info` - Explicit modifier and plane layout information
    ///
    /// # Safety
    ///
    /// - The dmabuf file descriptors must remain valid until the texture is dropped
    /// - The dmabuf must not be modified while the texture is in use
    /// - The modifier must be supported by the device (check with `get_modifier_properties`)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `VK_EXT_image_drm_format_modifier` is not available
    /// - The modifier is not supported for the format
    /// - The Vulkan operations fail
    #[cfg(feature = "dmabuf")]
    pub unsafe fn import_dmabuf_with_modifier(
        &self,
        dmabuf: &Dmabuf,
        modifier_info: &ExplicitModifierCreateInfo,
    ) -> Result<WgpuTexture, ImportError> {
        use std::ops::Deref;

        // Check if modifier extension is enabled
        if !self.modifier_capabilities.supports_modifiers() {
            return Err(ImportError::UnsupportedModifier(modifier_info.modifier));
        }

        // Get dmabuf properties
        let size = AllocatorBuffer::size(dmabuf);
        let width = size.w as u32;
        let height = size.h as u32;

        if width == 0 || height == 0 {
            return Err(ImportError::InvalidDimensions { width, height });
        }

        let wgpu_format = self.dmabuf_to_wgpu_format(dmabuf)?;
        let drm_format = AllocatorBuffer::format(dmabuf);
        let vk_format = self.wgpu_to_vk_format(wgpu_format)?;

        // Verify the modifier is supported for this format
        let modifier_props = self
            .get_modifier_properties(drm_format.code, modifier_info.modifier)
            .ok_or(ImportError::UnsupportedModifier(modifier_info.modifier))?;

        // Verify plane count matches
        let expected_planes = modifier_props.plane_count as usize;
        if modifier_info.plane_layouts.len() != expected_planes {
            return Err(ImportError::InvalidPlanes(format!(
                "Modifier expects {} planes, got {}",
                expected_planes,
                modifier_info.plane_layouts.len()
            )));
        }

        // Get plane information from dmabuf
        let planes: Vec<_> = dmabuf
            .handles()
            .zip(dmabuf.offsets())
            .zip(dmabuf.strides())
            .map(|((handle, offset), stride)| (handle.as_raw_fd(), offset, stride))
            .collect();

        if planes.len() < expected_planes {
            return Err(ImportError::InvalidPlanes(format!(
                "Dmabuf has {} planes, need {}",
                planes.len(),
                expected_planes
            )));
        }

        debug!(
            "Importing dmabuf with modifier: {}x{}, format={:?}, modifier={:#x} ({}), planes={}",
            width,
            height,
            drm_format.code,
            modifier_info.modifier,
            modifier_props.describe(),
            expected_planes
        );

        // Access the wgpu-hal Vulkan device
        let hal_device_guard = self
            .device
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or(ImportError::BridgeUnavailable)?;
        let hal_device: &wgpu_hal::vulkan::Device = hal_device_guard.deref();

        let raw_device = hal_device.raw_device();
        let physical_device = hal_device.raw_physical_device();

        // Build plane layouts for Vulkan
        let plane_layouts: Vec<vk::SubresourceLayout> = modifier_info
            .plane_layouts
            .iter()
            .map(|p| p.into())
            .collect();

        // Create image with DRM format modifier tiling
        let mut external_memory_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

        let mut modifier_explicit_info = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
            .drm_format_modifier(modifier_info.modifier)
            .plane_layouts(&plane_layouts);

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
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external_memory_info)
            .push_next(&mut modifier_explicit_info);

        let vk_image = raw_device
            .create_image(&image_create_info, None)
            .map_err(|e| {
                ImportError::ImageCreation(format!("vkCreateImage with modifier failed: {:?}", e))
            })?;

        // Get memory requirements
        let mem_requirements = raw_device.get_image_memory_requirements(vk_image);

        // Find suitable memory type
        let mem_properties = self.get_physical_device_memory_properties(physical_device)?;

        let memory_type_index = self
            .find_memory_type_index(
                &mem_properties,
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .ok_or_else(|| {
                raw_device.destroy_image(vk_image, None);
                ImportError::MemoryAllocation("No suitable memory type found".into())
            })?;

        // Dup the fd for import (Vulkan takes ownership)
        let (fd, _offset, _stride) = planes[0];
        let import_fd = {
            let duped = libc::dup(fd);
            if duped < 0 {
                raw_device.destroy_image(vk_image, None);
                return Err(ImportError::FdImport("Failed to dup dmabuf fd".into()));
            }
            duped
        };

        // Import the dmabuf fd
        let mut import_memory_fd_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(import_fd);

        let mut dedicated_alloc_info = vk::MemoryDedicatedAllocateInfo::default().image(vk_image);

        let memory_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index as u32)
            .push_next(&mut import_memory_fd_info)
            .push_next(&mut dedicated_alloc_info);

        let vk_memory = raw_device
            .allocate_memory(&memory_allocate_info, None)
            .map_err(|e| {
                raw_device.destroy_image(vk_image, None);
                ImportError::MemoryAllocation(format!("vkAllocateMemory failed: {:?}", e))
            })?;

        // Bind memory to image
        raw_device
            .bind_image_memory(vk_image, vk_memory, 0)
            .map_err(|e| {
                raw_device.free_memory(vk_memory, None);
                raw_device.destroy_image(vk_image, None);
                ImportError::MemoryAllocation(format!("vkBindImageMemory failed: {:?}", e))
            })?;

        // Create the HAL texture descriptor
        let hal_desc = wgpu_hal::TextureDescriptor {
            label: Some("dmabuf-import-modifier"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu_types::TextureUses::RESOURCE | wgpu_types::TextureUses::COPY_SRC,
            memory_flags: wgpu_hal::MemoryFlags::empty(),
            view_formats: vec![],
        };

        // Wrap in wgpu-hal texture
        let hal_texture = hal_device.texture_from_raw(
            vk_image,
            &hal_desc,
            None,
            wgpu_hal::vulkan::TextureMemory::Dedicated(vk_memory),
        );

        // Wrap the HAL texture in a wgpu Texture
        let wgpu_desc = wgpu::TextureDescriptor {
            label: Some("dmabuf-import-modifier"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };

        let wgpu_texture = self
            .device
            .create_texture_from_hal::<wgpu_hal::api::Vulkan>(hal_texture, &wgpu_desc);

        debug!(
            "Successfully imported dmabuf with modifier {:#x} as wgpu texture",
            modifier_info.modifier
        );

        Ok(WgpuTexture::new(
            wgpu_texture,
            wgpu_format,
            width,
            height,
            true,
        ))
    }

    /// Import a dmabuf with explicit modifier support (stub for non-experimental builds).
    #[cfg(not(feature = "dmabuf"))]
    pub unsafe fn import_dmabuf_with_modifier(
        &self,
        _dmabuf: &Dmabuf,
        _modifier_info: &ExplicitModifierCreateInfo,
    ) -> Result<WgpuTexture, ImportError> {
        Err(ImportError::FdImport(
            "dmabuf import requires 'dmabuf' feature (enabled by default)".into(),
        ))
    }

    /// Get physical device memory properties (helper for dmabuf import).
    #[cfg(feature = "dmabuf")]
    unsafe fn get_physical_device_memory_properties(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<vk::PhysicalDeviceMemoryProperties, ImportError> {
        use std::ops::Deref;

        // Access adapter to get instance
        let hal_adapter_guard = self
            .adapter
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or(ImportError::BridgeUnavailable)?;
        let hal_adapter: &wgpu_hal::vulkan::Adapter = hal_adapter_guard.deref();

        // Get the ash::Instance from the adapter's shared instance
        let instance = hal_adapter.shared_instance().raw_instance();

        // Query actual memory properties from Vulkan
        let props = instance.get_physical_device_memory_properties(physical_device);

        debug!(
            "Queried memory properties: {} types, {} heaps",
            props.memory_type_count, props.memory_heap_count
        );

        // Log memory type details for debugging
        for i in 0..props.memory_type_count as usize {
            let flags = props.memory_types[i].property_flags;
            trace!(
                "Memory type {}: flags={:?}, heap={}",
                i,
                flags,
                props.memory_types[i].heap_index
            );
        }

        Ok(props)
    }

    /// Find a memory type index that satisfies the requirements.
    #[cfg(feature = "dmabuf")]
    fn find_memory_type_index(
        &self,
        mem_properties: &vk::PhysicalDeviceMemoryProperties,
        type_bits: u32,
        required_flags: vk::MemoryPropertyFlags,
    ) -> Option<usize> {
        for i in 0..mem_properties.memory_type_count as usize {
            let type_bit = 1 << i;
            let is_required_type = (type_bits & type_bit) != 0;
            let has_required_flags =
                (mem_properties.memory_types[i].property_flags & required_flags) == required_flags;

            if is_required_type && has_required_flags {
                return Some(i);
            }
        }
        None
    }

    /// Map wgpu TextureFormat to Vulkan VkFormat.
    #[cfg(feature = "dmabuf")]
    fn wgpu_to_vk_format(&self, format: wgpu::TextureFormat) -> Result<vk::Format, ImportError> {
        match format {
            wgpu::TextureFormat::Bgra8Unorm => Ok(vk::Format::B8G8R8A8_UNORM),
            wgpu::TextureFormat::Bgra8UnormSrgb => Ok(vk::Format::B8G8R8A8_SRGB),
            wgpu::TextureFormat::Rgba8Unorm => Ok(vk::Format::R8G8B8A8_UNORM),
            wgpu::TextureFormat::Rgba8UnormSrgb => Ok(vk::Format::R8G8B8A8_SRGB),
            wgpu::TextureFormat::Rgb10a2Unorm => Ok(vk::Format::A2B10G10R10_UNORM_PACK32),
            _ => Err(ImportError::UnsupportedFormat(
                drm_fourcc::DrmFourcc::Xrgb8888,
            )), // placeholder
        }
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    pub fn supported_formats(&self) -> &[SupportedFormat] {
        &self.supported_formats
    }

    pub fn sync_capabilities(&self) -> &SyncCapabilities {
        &self.sync_capabilities
    }

    pub fn modifier_capabilities(&self) -> &ModifierCapabilities {
        &self.modifier_capabilities
    }

    pub fn multiplanar_capabilities(&self) -> &MultiPlanarCapabilities {
        &self.multiplanar_capabilities
    }

    pub fn supports_multiplanar(&self) -> bool {
        self.multiplanar_capabilities.has_ycbcr_conversion
    }

    pub fn supports_multiplanar_format(&self, fourcc: drm_fourcc::DrmFourcc) -> bool {
        self.multiplanar_capabilities.supports_format(fourcc)
    }

    pub fn supports_modifiers(&self) -> bool {
        self.modifier_capabilities.supports_modifiers()
    }

    /// Returns `None` if the format is not supported.
    pub fn get_format_modifiers(
        &self,
        fourcc: drm_fourcc::DrmFourcc,
    ) -> Option<&[ModifierProperties]> {
        self.supported_formats
            .iter()
            .find(|f| f.fourcc == fourcc)
            .map(|f| f.modifier_props.as_slice())
    }

    pub fn get_modifier_properties(
        &self,
        fourcc: drm_fourcc::DrmFourcc,
        modifier: u64,
    ) -> Option<&ModifierProperties> {
        self.supported_formats
            .iter()
            .find(|f| f.fourcc == fourcc)?
            .modifier_props
            .iter()
            .find(|p| p.modifier == modifier)
    }

    pub fn can_export_sync_fd(&self) -> bool {
        self.sync_capabilities.can_export_sync_fd
    }

    /// Submit a frame and return the frame number.
    ///
    /// This increments the internal frame counter and can be used
    /// to track frame completion.
    pub fn submit_frame(&self) -> u64 {
        self.frame_counter.fetch_add(1, Ordering::AcqRel) + 1
    }

    pub fn current_frame(&self) -> u64 {
        self.frame_counter.load(Ordering::Acquire)
    }

    /// Wait for a specific frame to complete.
    ///
    /// This uses wgpu's poll mechanism since we can't directly wait
    /// on timeline semaphores through wgpu's API.
    ///
    /// # Note
    ///
    /// Due to wgpu's abstraction, this is an approximation. True sync FD
    /// waiting would require raw Vulkan queue submission or wgpu adding
    /// external semaphore support (tracked in wgpu#4067).
    pub fn wait_for_frame(&self, _frame: u64, timeout_ms: Option<u64>) -> Result<(), BridgeError> {
        // Use wgpu's poll mechanism for synchronization
        // This polls until all submitted work is complete
        let poll_type = wgpu::PollType::Wait {
            submission_index: None, // Wait for most recent submission
            timeout: timeout_ms.map(std::time::Duration::from_millis),
        };

        match self.device.poll(poll_type) {
            Ok(_status) => {
                // Update last completed frame
                let current = self.frame_counter.load(Ordering::Acquire);
                self.last_completed_frame.store(current, Ordering::Release);
                Ok(())
            }
            Err(wgpu::PollError::Timeout) => Err(BridgeError::Sync("Wait timed out".into())),
            Err(e) => Err(BridgeError::Sync(format!("poll error: {:?}", e))),
        }
    }

    /// Wait for all submitted GPU work to complete.
    pub fn wait_idle(&self) -> Result<(), BridgeError> {
        self.wait_for_frame(self.current_frame(), None)
    }

    /// Get the last known completed frame number.
    ///
    /// This is an approximation based on poll results.
    pub fn last_completed_frame(&self) -> u64 {
        self.last_completed_frame.load(Ordering::Acquire)
    }

    /// Check if a dmabuf format is supported.
    pub fn supports_format(&self, fourcc: drm_fourcc::DrmFourcc, modifier: u64) -> bool {
        self.supported_formats.iter().any(|f| {
            f.fourcc == fourcc && (f.modifiers.is_empty() || f.modifiers.contains(&modifier))
        })
    }

    /// Convert a dmabuf format to wgpu TextureFormat.
    #[cfg(feature = "dmabuf")]
    fn dmabuf_to_wgpu_format(&self, dmabuf: &Dmabuf) -> Result<wgpu::TextureFormat, ImportError> {
        let drm_format = AllocatorBuffer::format(dmabuf);
        let fourcc = drm_format.code;

        // Map common DRM formats to wgpu formats
        // Note: DRM uses different channel order conventions than wgpu
        match fourcc {
            // ARGB8888 = B8G8R8A8 in memory (little-endian)
            drm_fourcc::DrmFourcc::Argb8888 => Ok(wgpu::TextureFormat::Bgra8Unorm),
            drm_fourcc::DrmFourcc::Xrgb8888 => Ok(wgpu::TextureFormat::Bgra8Unorm),
            // ABGR8888 = R8G8B8A8 in memory
            drm_fourcc::DrmFourcc::Abgr8888 => Ok(wgpu::TextureFormat::Rgba8Unorm),
            drm_fourcc::DrmFourcc::Xbgr8888 => Ok(wgpu::TextureFormat::Rgba8Unorm),
            // RGB formats
            drm_fourcc::DrmFourcc::Rgb888 => Ok(wgpu::TextureFormat::Rgba8Unorm), // Need padding
            drm_fourcc::DrmFourcc::Bgr888 => Ok(wgpu::TextureFormat::Bgra8Unorm), // Need padding
            // Single and dual channel formats
            drm_fourcc::DrmFourcc::R8 => Ok(wgpu::TextureFormat::R8Unorm),
            drm_fourcc::DrmFourcc::Rg88 => Ok(wgpu::TextureFormat::Rg8Unorm),
            drm_fourcc::DrmFourcc::Gr88 => Ok(wgpu::TextureFormat::Rg8Unorm), // Swizzle in shader if needed
            // 10-bit formats
            drm_fourcc::DrmFourcc::Argb2101010 => Ok(wgpu::TextureFormat::Rgb10a2Unorm),
            drm_fourcc::DrmFourcc::Xrgb2101010 => Ok(wgpu::TextureFormat::Rgb10a2Unorm),
            // Multi-planar formats (NV12, P010, etc.) require special handling
            _ if is_multiplanar_format(fourcc) => Err(ImportError::MultiPlanarFormat {
                fourcc,
                hint: "Use import_multiplanar_dmabuf() for YUV formats".into(),
            }),
            _ => Err(ImportError::UnsupportedFormat(fourcc)),
        }
    }

    /// Query supported formats from the adapter.
    ///
    /// Uses wgpu-hal to access the Vulkan physical device and query
    /// actual format support via vkGetPhysicalDeviceFormatProperties.
    fn query_supported_formats(adapter: &wgpu::Adapter) -> Vec<SupportedFormat> {
        // Try to query actual Vulkan capabilities
        // SAFETY: We're only reading format properties, which is safe
        let formats = unsafe {
            adapter
                .as_hal::<wgpu_hal::api::Vulkan>()
                .map(|hal_adapter| {
                    use std::ops::Deref;
                    let hal_adapter: &wgpu_hal::vulkan::Adapter = hal_adapter.deref();
                    Self::query_vulkan_format_support(hal_adapter)
                })
                .unwrap_or_else(|| {
                    // Fallback to common formats if hal access fails
                    debug!("Could not access HAL adapter for format query, using defaults");
                    Self::default_supported_formats()
                })
        };

        debug!("Queried {} supported formats", formats.len());
        formats
    }

    /// Query format support from the Vulkan physical device.
    fn query_vulkan_format_support(
        hal_adapter: &wgpu_hal::vulkan::Adapter,
    ) -> Vec<SupportedFormat> {
        let mut formats = Vec::new();

        // Get the raw Vulkan handles
        let instance = hal_adapter.shared_instance().raw_instance();
        let physical_device = hal_adapter.raw_physical_device();

        // List of DRM formats to check with their corresponding Vulkan formats
        let format_mappings = [
            (
                drm_fourcc::DrmFourcc::Argb8888,
                vk::Format::B8G8R8A8_UNORM,
                wgpu::TextureFormat::Bgra8Unorm,
            ),
            (
                drm_fourcc::DrmFourcc::Xrgb8888,
                vk::Format::B8G8R8A8_UNORM,
                wgpu::TextureFormat::Bgra8Unorm,
            ),
            (
                drm_fourcc::DrmFourcc::Abgr8888,
                vk::Format::R8G8B8A8_UNORM,
                wgpu::TextureFormat::Rgba8Unorm,
            ),
            (
                drm_fourcc::DrmFourcc::Xbgr8888,
                vk::Format::R8G8B8A8_UNORM,
                wgpu::TextureFormat::Rgba8Unorm,
            ),
            (
                drm_fourcc::DrmFourcc::Argb2101010,
                vk::Format::A2B10G10R10_UNORM_PACK32,
                wgpu::TextureFormat::Rgb10a2Unorm,
            ),
            (
                drm_fourcc::DrmFourcc::Xrgb2101010,
                vk::Format::A2B10G10R10_UNORM_PACK32,
                wgpu::TextureFormat::Rgb10a2Unorm,
            ),
            // Single and dual channel formats
            (
                drm_fourcc::DrmFourcc::R8,
                vk::Format::R8_UNORM,
                wgpu::TextureFormat::R8Unorm,
            ),
            (
                drm_fourcc::DrmFourcc::Rg88,
                vk::Format::R8G8_UNORM,
                wgpu::TextureFormat::Rg8Unorm,
            ),
            (
                drm_fourcc::DrmFourcc::Gr88,
                vk::Format::R8G8_UNORM,
                wgpu::TextureFormat::Rg8Unorm,
            ),
            // sRGB variants
            (
                drm_fourcc::DrmFourcc::Argb8888,
                vk::Format::B8G8R8A8_SRGB,
                wgpu::TextureFormat::Bgra8UnormSrgb,
            ),
            (
                drm_fourcc::DrmFourcc::Abgr8888,
                vk::Format::R8G8B8A8_SRGB,
                wgpu::TextureFormat::Rgba8UnormSrgb,
            ),
        ];

        // Check each format for SAMPLED_IMAGE support (needed for textures)
        for (drm_fourcc, vk_format, wgpu_format) in format_mappings {
            // SAFETY: physical_device is valid from the adapter
            let props = unsafe {
                instance.get_physical_device_format_properties(physical_device, vk_format)
            };

            // Check if LINEAR tiling supports sampling (for dmabuf import)
            let linear_features = props.linear_tiling_features;
            let optimal_features = props.optimal_tiling_features;

            let supports_sampling = linear_features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE)
                || optimal_features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE);

            if supports_sampling {
                // Check if already added (some DRM formats map to same Vulkan format)
                if !formats.iter().any(|f: &SupportedFormat| {
                    f.fourcc == drm_fourcc && f.wgpu_format == wgpu_format
                }) {
                    // Query detailed modifier properties
                    let modifier_props =
                        Self::query_format_modifiers(instance, physical_device, vk_format);

                    // Extract just the modifier values for quick lookup
                    let modifiers: Vec<u64> = modifier_props.iter().map(|p| p.modifier).collect();

                    // If no modifiers returned, add LINEAR as fallback
                    let (modifier_props, modifiers) = if modifier_props.is_empty() {
                        (
                            vec![ModifierProperties {
                                modifier: drm_mod::LINEAR,
                                plane_count: 1,
                                format_features: linear_features,
                            }],
                            vec![drm_mod::LINEAR],
                        )
                    } else {
                        (modifier_props, modifiers)
                    };

                    let modifier_count = modifiers.len();
                    formats.push(SupportedFormat {
                        fourcc: drm_fourcc,
                        modifier_props,
                        modifiers,
                        wgpu_format,
                        vk_format,
                    });

                    trace!(
                        "Format {:?} ({:?}) supported: linear={:?}, optimal={:?}, {} modifiers",
                        drm_fourcc,
                        vk_format,
                        linear_features,
                        optimal_features,
                        modifier_count
                    );
                }
            }
        }

        if formats.is_empty() {
            // If no formats detected, use fallback
            debug!("No formats detected via Vulkan query, using defaults");
            return Self::default_supported_formats();
        }

        formats
    }

    /// Query supported modifiers for a format (requires VK_EXT_image_drm_format_modifier).
    fn query_format_modifiers(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        vk_format: vk::Format,
    ) -> Vec<ModifierProperties> {
        // Use the modifiers module to query actual supported modifiers
        // SAFETY: instance and physical_device are valid from the adapter
        unsafe { modifiers::query_format_modifiers(instance, physical_device, vk_format) }
    }

    /// Default supported formats (fallback when Vulkan query fails).
    fn default_supported_formats() -> Vec<SupportedFormat> {
        // Create default formats with only LINEAR modifier
        let default_props = vec![ModifierProperties {
            modifier: drm_mod::LINEAR,
            plane_count: 1,
            format_features: vk::FormatFeatureFlags::SAMPLED_IMAGE,
        }];
        let default_mods = vec![drm_mod::LINEAR];

        vec![
            SupportedFormat {
                fourcc: drm_fourcc::DrmFourcc::Argb8888,
                modifier_props: default_props.clone(),
                modifiers: default_mods.clone(),
                wgpu_format: wgpu::TextureFormat::Bgra8Unorm,
                vk_format: vk::Format::B8G8R8A8_UNORM,
            },
            SupportedFormat {
                fourcc: drm_fourcc::DrmFourcc::Xrgb8888,
                modifier_props: default_props.clone(),
                modifiers: default_mods.clone(),
                wgpu_format: wgpu::TextureFormat::Bgra8Unorm,
                vk_format: vk::Format::B8G8R8A8_UNORM,
            },
            SupportedFormat {
                fourcc: drm_fourcc::DrmFourcc::Abgr8888,
                modifier_props: default_props.clone(),
                modifiers: default_mods.clone(),
                wgpu_format: wgpu::TextureFormat::Rgba8Unorm,
                vk_format: vk::Format::R8G8B8A8_UNORM,
            },
            SupportedFormat {
                fourcc: drm_fourcc::DrmFourcc::Xbgr8888,
                modifier_props: default_props.clone(),
                modifiers: default_mods.clone(),
                wgpu_format: wgpu::TextureFormat::Rgba8Unorm,
                vk_format: vk::Format::R8G8B8A8_UNORM,
            },
            SupportedFormat {
                fourcc: drm_fourcc::DrmFourcc::Argb2101010,
                modifier_props: default_props,
                modifiers: default_mods,
                wgpu_format: wgpu::TextureFormat::Rgb10a2Unorm,
                vk_format: vk::Format::A2B10G10R10_UNORM_PACK32,
            },
        ]
    }
}

impl std::fmt::Debug for WgpuBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuBridge")
            .field("adapter", &self.adapter.get_info().name)
            .field("supported_formats", &self.supported_formats.len())
            .finish()
    }
}
