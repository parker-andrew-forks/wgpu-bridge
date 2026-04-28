//! Synchronization primitives for Wayland explicit sync.
//!
//! This module provides integration with the `linux-drm-syncobj-v1` Wayland protocol,
//! enabling proper explicit sync between wgpu rendering and Wayland clients.
//!
//! # Architecture
//!
//! ```text
//! DRM Syncobj (kernel)
//!       ↓ fd export
//! Import as VkSemaphore (this module)
//!       ↓
//! Queue::add_signal_semaphore() (wgpu PR #6813)
//!       ↓
//! GPU signals semaphore at submit
//!       ↓
//! Client receives release notification
//! ```
//!
//! # Key Discovery
//!
//! wgpu PR #6813 (merged January 2025) added:
//! - `Queue::add_signal_semaphore()` - register external semaphores for signaling
//! - `Queue::raw_device()` - access raw Vulkan device
//!
//! This enables compositors to signal release points without raw Vulkan queue submission.
//!
//! # Usage
//!
//! ```ignore
//! let sync_manager = SyncManager::new(&bridge)?;
//!
//! // For each frame:
//! // 1. Import acquire points and wait (CPU or GPU side)
//! // 2. Render frame
//! // 3. Register release points for signaling
//! // 4. Submit - release points are automatically signaled
//! ```

use ash::vk;
use std::ops::Deref;
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tracing::{debug, info, trace};

use crate::error::BridgeError;

/// Capabilities for explicit synchronization.
#[derive(Debug, Clone, Default)]
pub struct SyncCapabilities {
    /// VK_KHR_external_semaphore is available
    pub external_semaphore: bool,
    /// VK_KHR_external_semaphore_fd is available
    pub external_semaphore_fd: bool,
    /// VK_KHR_timeline_semaphore is available (Vulkan 1.2+)
    pub timeline_semaphore: bool,
    /// Can import SYNC_FD (linux sync_file)
    pub can_import_sync_fd: bool,
    /// Can export SYNC_FD
    pub can_export_sync_fd: bool,
    /// Can import OPAQUE_FD (for DRM syncobj)
    pub can_import_opaque_fd: bool,
    /// Can export OPAQUE_FD
    pub can_export_opaque_fd: bool,
}

impl SyncCapabilities {
    /// Query capabilities from a Vulkan physical device.
    ///
    /// # Safety
    /// The instance and physical device must be valid.
    pub unsafe fn query(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Self {
        let extensions = match instance.enumerate_device_extension_properties(physical_device) {
            Ok(exts) => exts,
            Err(_) => return Self::default(),
        };

        let extension_names: Vec<&std::ffi::CStr> = extensions
            .iter()
            .map(|ext| std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()))
            .collect();

        let external_semaphore = extension_names.contains(&ash::khr::external_semaphore::NAME);
        let external_semaphore_fd =
            extension_names.contains(&ash::khr::external_semaphore_fd::NAME);

        // Check for timeline semaphore (Vulkan 1.2 core or extension)
        let props = instance.get_physical_device_properties(physical_device);
        let timeline_semaphore = props.api_version >= vk::make_api_version(0, 1, 2, 0)
            || extension_names.contains(&ash::khr::timeline_semaphore::NAME);

        // Query external semaphore properties for SYNC_FD
        // For Vulkan 1.1+, this is part of core, so we can use the instance function
        let (can_import_sync_fd, can_export_sync_fd, can_import_opaque_fd, can_export_opaque_fd) =
            if external_semaphore_fd {
                // Query SYNC_FD capabilities using Vulkan 1.1+ core function
                let mut sync_fd_props = vk::ExternalSemaphoreProperties::default();
                let sync_fd_info = vk::PhysicalDeviceExternalSemaphoreInfo::default()
                    .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);

                instance.get_physical_device_external_semaphore_properties(
                    physical_device,
                    &sync_fd_info,
                    &mut sync_fd_props,
                );

                let can_import_sync = sync_fd_props
                    .external_semaphore_features
                    .contains(vk::ExternalSemaphoreFeatureFlags::IMPORTABLE);
                let can_export_sync = sync_fd_props
                    .external_semaphore_features
                    .contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE);

                // Query OPAQUE_FD capabilities
                let mut opaque_fd_props = vk::ExternalSemaphoreProperties::default();
                let opaque_fd_info = vk::PhysicalDeviceExternalSemaphoreInfo::default()
                    .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);

                instance.get_physical_device_external_semaphore_properties(
                    physical_device,
                    &opaque_fd_info,
                    &mut opaque_fd_props,
                );

                let can_import_opaque = opaque_fd_props
                    .external_semaphore_features
                    .contains(vk::ExternalSemaphoreFeatureFlags::IMPORTABLE);
                let can_export_opaque = opaque_fd_props
                    .external_semaphore_features
                    .contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE);

                (
                    can_import_sync,
                    can_export_sync,
                    can_import_opaque,
                    can_export_opaque,
                )
            } else {
                (false, false, false, false)
            };

        let caps = Self {
            external_semaphore,
            external_semaphore_fd,
            timeline_semaphore,
            can_import_sync_fd,
            can_export_sync_fd,
            can_import_opaque_fd,
            can_export_opaque_fd,
        };

        debug!(
            "Explicit sync capabilities: ext_sem={}, ext_sem_fd={}, timeline={}, \
             import_sync_fd={}, export_sync_fd={}, import_opaque={}, export_opaque={}",
            caps.external_semaphore,
            caps.external_semaphore_fd,
            caps.timeline_semaphore,
            caps.can_import_sync_fd,
            caps.can_export_sync_fd,
            caps.can_import_opaque_fd,
            caps.can_export_opaque_fd
        );

        caps
    }

    /// Check if explicit sync is fully supported.
    pub fn is_supported(&self) -> bool {
        self.external_semaphore && self.external_semaphore_fd && self.can_import_sync_fd
    }
}

/// A sync point representing a timeline position.
///
/// This corresponds to a DRM syncobj timeline point (syncobj + value).
#[derive(Debug, Clone)]
pub struct SyncPoint {
    /// The timeline value (0 for binary semaphores)
    pub timeline_value: u64,
    /// Whether this is a timeline (vs binary) semaphore
    pub is_timeline: bool,
}

impl SyncPoint {
    /// Create a binary sync point (value = 0).
    pub fn binary() -> Self {
        Self {
            timeline_value: 0,
            is_timeline: false,
        }
    }

    /// Create a timeline sync point with a specific value.
    pub fn timeline(value: u64) -> Self {
        Self {
            timeline_value: value,
            is_timeline: true,
        }
    }
}

/// An imported semaphore from an external source.
pub struct ImportedSemaphore {
    /// Raw Vulkan semaphore handle
    semaphore: vk::Semaphore,
    /// The sync point this represents
    sync_point: SyncPoint,
    /// Device reference for cleanup
    device: Arc<VulkanDeviceRef>,
    /// Source fd (for debugging)
    source_fd: Option<RawFd>,
}

impl ImportedSemaphore {
    pub fn raw(&self) -> vk::Semaphore {
        self.semaphore
    }

    pub fn sync_point(&self) -> &SyncPoint {
        &self.sync_point
    }
}

impl Drop for ImportedSemaphore {
    fn drop(&mut self) {
        // SAFETY: device and semaphore are valid, and we own the semaphore.
        unsafe {
            self.device.device.destroy_semaphore(self.semaphore, None);
        }
        trace!("Destroyed imported semaphore from fd {:?}", self.source_fd);
    }
}

/// An exportable semaphore that can be shared externally.
///
/// Note: This is different from `sync::ExportableSemaphore` - this version
/// integrates with the explicit sync workflow.
pub struct ExportSemaphore {
    /// Raw Vulkan semaphore handle
    semaphore: vk::Semaphore,
    /// Device reference
    device: Arc<VulkanDeviceRef>,
    /// Whether this is a timeline semaphore
    is_timeline: bool,
    /// Current timeline value (for timeline semaphores)
    timeline_value: AtomicU64,
}

impl ExportSemaphore {
    pub fn raw(&self) -> vk::Semaphore {
        self.semaphore
    }

    pub fn current_value(&self) -> u64 {
        self.timeline_value.load(Ordering::Acquire)
    }

    pub fn next_signal_value(&self) -> u64 {
        if self.is_timeline {
            self.timeline_value.fetch_add(1, Ordering::AcqRel) + 1
        } else {
            0
        }
    }

    /// Export as a sync fd.
    ///
    /// # Safety
    /// The semaphore must be in a valid state for export.
    ///
    /// # Note
    /// This requires VK_KHR_external_semaphore_fd to be enabled on the device.
    /// In wgpu standalone mode, this extension may not be enabled, and this
    /// function will return an error.
    pub unsafe fn export_sync_fd(&self) -> Result<OwnedFd, BridgeError> {
        // Check if the extension function is available before trying to use it
        // This prevents a panic when the extension isn't enabled
        let get_semaphore_fd_name = b"vkGetSemaphoreFdKHR\0";
        let func_ptr = self.device.instance.get_device_proc_addr(
            self.device.device.handle(),
            std::ffi::CStr::from_bytes_with_nul_unchecked(get_semaphore_fd_name).as_ptr(),
        );

        if func_ptr.is_none() {
            return Err(BridgeError::Sync(
                "VK_KHR_external_semaphore_fd extension not enabled on device".into(),
            ));
        }

        let external_semaphore_fd = ash::khr::external_semaphore_fd::Device::new(
            &self.device.instance,
            &self.device.device,
        );

        let get_fd_info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(self.semaphore)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);

        let fd = external_semaphore_fd
            .get_semaphore_fd(&get_fd_info)
            .map_err(|e| BridgeError::Sync(format!("failed to export sync fd: {:?}", e)))?;

        Ok(OwnedFd::from_raw_fd(fd))
    }
}

impl Drop for ExportSemaphore {
    fn drop(&mut self) {
        // SAFETY: device and semaphore are valid, and we own the semaphore.
        unsafe {
            self.device.device.destroy_semaphore(self.semaphore, None);
        }
        trace!("Destroyed exportable semaphore");
    }
}

/// Reference to Vulkan device and instance.
struct VulkanDeviceRef {
    instance: ash::Instance,
    device: ash::Device,
}

impl Drop for VulkanDeviceRef {
    fn drop(&mut self) {
        // Device and instance are owned by wgpu/Smithay, don't destroy
    }
}

/// Manager for explicit synchronization primitives.
///
/// This handles importing/exporting sync primitives for linux-drm-syncobj-v1 integration.
pub struct SyncManager {
    /// Vulkan device reference
    device_ref: Arc<VulkanDeviceRef>,
    /// Capabilities
    capabilities: SyncCapabilities,
    /// Pool of reusable binary semaphores
    binary_semaphore_pool: Mutex<Vec<vk::Semaphore>>,
    /// Pool of reusable timeline semaphores
    timeline_semaphore_pool: Mutex<Vec<vk::Semaphore>>,
    /// Semaphores pending signal (registered with queue)
    pending_signals: Mutex<Vec<vk::Semaphore>>,
    /// Statistics
    stats: SyncStats,
}

/// Statistics for sync operations.
#[derive(Debug, Default)]
struct SyncStats {
    semaphores_imported: AtomicU64,
    semaphores_exported: AtomicU64,
    semaphores_signaled: AtomicU64,
    pool_hits: AtomicU64,
    pool_misses: AtomicU64,
}

impl SyncManager {
    /// Create a new explicit sync manager.
    ///
    /// # Safety
    /// The instance and device must be valid Vulkan handles with external semaphore extensions.
    pub unsafe fn new(
        instance: ash::Instance,
        device: ash::Device,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self, BridgeError> {
        let capabilities = SyncCapabilities::query(&instance, physical_device);

        if !capabilities.is_supported() {
            return Err(BridgeError::Sync(
                "Explicit sync not supported: missing external semaphore extensions".into(),
            ));
        }

        info!(
            "Created SyncManager: timeline={}, sync_fd_import={}, sync_fd_export={}",
            capabilities.timeline_semaphore,
            capabilities.can_import_sync_fd,
            capabilities.can_export_sync_fd
        );

        Ok(Self {
            device_ref: Arc::new(VulkanDeviceRef { instance, device }),
            capabilities,
            binary_semaphore_pool: Mutex::new(Vec::new()),
            timeline_semaphore_pool: Mutex::new(Vec::new()),
            pending_signals: Mutex::new(Vec::new()),
            stats: SyncStats::default(),
        })
    }

    /// Get the capabilities.
    pub fn capabilities(&self) -> &SyncCapabilities {
        &self.capabilities
    }

    /// Import a sync fd as a semaphore.
    ///
    /// This is used for acquire points - the compositor waits on these before sampling buffers.
    ///
    /// # Arguments
    /// * `fd` - The sync fd to import (ownership is transferred)
    /// * `sync_point` - The sync point metadata
    ///
    /// # Safety
    /// The fd must be a valid sync_file or DRM syncobj fd.
    pub unsafe fn import_sync_fd(
        &self,
        fd: OwnedFd,
        sync_point: SyncPoint,
    ) -> Result<ImportedSemaphore, BridgeError> {
        let raw_fd = fd.as_raw_fd();

        // Check if the extension function is available before trying to use it
        let import_semaphore_fd_name = b"vkImportSemaphoreFdKHR\0";
        let func_ptr = self.device_ref.instance.get_device_proc_addr(
            self.device_ref.device.handle(),
            std::ffi::CStr::from_bytes_with_nul_unchecked(import_semaphore_fd_name).as_ptr(),
        );

        if func_ptr.is_none() {
            return Err(BridgeError::Sync(
                "VK_KHR_external_semaphore_fd extension not enabled on device".into(),
            ));
        }

        let semaphore = self.get_or_create_semaphore(sync_point.is_timeline)?;
        let external_semaphore_fd = ash::khr::external_semaphore_fd::Device::new(
            &self.device_ref.instance,
            &self.device_ref.device,
        );

        let import_info = vk::ImportSemaphoreFdInfoKHR::default()
            .semaphore(semaphore)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
            .fd(fd.into_raw_fd()) // Transfer ownership to Vulkan
            .flags(vk::SemaphoreImportFlags::TEMPORARY);

        external_semaphore_fd
            .import_semaphore_fd(&import_info)
            .map_err(|e| BridgeError::Sync(format!("failed to import sync fd: {:?}", e)))?;

        self.stats
            .semaphores_imported
            .fetch_add(1, Ordering::Relaxed);

        trace!("Imported sync fd {} as semaphore {:?}", raw_fd, semaphore);

        Ok(ImportedSemaphore {
            semaphore,
            sync_point,
            device: self.device_ref.clone(),
            source_fd: Some(raw_fd),
        })
    }

    /// Create an exportable semaphore for release points.
    ///
    /// # Arguments
    /// * `timeline` - Whether to create a timeline semaphore
    ///
    /// # Safety
    /// The device must be valid.
    pub unsafe fn create_exportable_semaphore(
        &self,
        timeline: bool,
    ) -> Result<ExportSemaphore, BridgeError> {
        let semaphore = self.create_semaphore_internal(timeline, true)?;

        Ok(ExportSemaphore {
            semaphore,
            device: self.device_ref.clone(),
            is_timeline: timeline,
            timeline_value: AtomicU64::new(0),
        })
    }

    /// Register a binary semaphore for signaling on the next queue submit.
    ///
    /// This uses wgpu's `Queue::add_signal_semaphore()` API (from PR #6813).
    ///
    /// # Safety
    /// The queue must be valid and the semaphore must be in a valid state.
    pub unsafe fn register_signal_semaphore_binary(
        &self,
        queue: &wgpu::Queue,
        semaphore: vk::Semaphore,
    ) -> Result<(), BridgeError> {
        // Use wgpu's as_hal to get the underlying Vulkan queue
        // This is the key integration point with PR #6813
        if let Some(hal_queue) = queue.as_hal::<wgpu_hal::api::Vulkan>() {
            // Binary semaphore: pass None for the value
            hal_queue.deref().add_signal_semaphore(semaphore, None);
            trace!("Registered binary semaphore {:?} for signaling", semaphore);
        } else {
            return Err(BridgeError::Sync("Failed to access Vulkan queue".into()));
        }

        self.pending_signals
            .lock()
            .expect("mutex poisoned")
            .push(semaphore);
        self.stats
            .semaphores_signaled
            .fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Register a timeline semaphore for signaling on the next queue submit.
    ///
    /// This uses wgpu's `Queue::add_signal_semaphore()` API (from PR #6813).
    ///
    /// # Arguments
    /// * `queue` - The wgpu queue
    /// * `semaphore` - The Vulkan semaphore
    /// * `timeline_value` - The timeline value to signal
    ///
    /// # Safety
    /// The queue must be valid and the semaphore must be in a valid state.
    pub unsafe fn register_signal_semaphore_timeline(
        &self,
        queue: &wgpu::Queue,
        semaphore: vk::Semaphore,
        timeline_value: u64,
    ) -> Result<(), BridgeError> {
        if let Some(hal_queue) = queue.as_hal::<wgpu_hal::api::Vulkan>() {
            // Timeline semaphore: pass the value
            hal_queue
                .deref()
                .add_signal_semaphore(semaphore, Some(timeline_value));
            trace!(
                "Registered timeline semaphore {:?} for signaling at value {}",
                semaphore,
                timeline_value
            );
        } else {
            return Err(BridgeError::Sync("Failed to access Vulkan queue".into()));
        }

        self.pending_signals
            .lock()
            .expect("mutex poisoned")
            .push(semaphore);
        self.stats
            .semaphores_signaled
            .fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Wait on a semaphore on the CPU.
    ///
    /// This blocks until the semaphore is signaled.
    ///
    /// # Safety
    /// The semaphore must be valid.
    pub unsafe fn wait_semaphore_cpu(
        &self,
        semaphore: &ImportedSemaphore,
        timeout_ns: u64,
    ) -> Result<(), BridgeError> {
        if semaphore.sync_point.is_timeline {
            // Timeline semaphore wait
            let wait_info = vk::SemaphoreWaitInfo::default()
                .semaphores(std::slice::from_ref(&semaphore.semaphore))
                .values(std::slice::from_ref(&semaphore.sync_point.timeline_value));

            self.device_ref
                .device
                .wait_semaphores(&wait_info, timeout_ns)
                .map_err(|e| {
                    BridgeError::Sync(format!("timeline semaphore wait failed: {:?}", e))
                })?;
        } else {
            // Binary semaphore - can't directly wait on CPU in Vulkan
            // For binary semaphores from sync_file, we need to use the fd
            return Err(BridgeError::Sync(
                "CPU wait on binary semaphore requires sync_file fd polling".into(),
            ));
        }

        Ok(())
    }

    /// Wait on a sync fd using poll().
    ///
    /// This is the fallback for binary semaphores.
    pub fn wait_sync_fd(fd: &OwnedFd, timeout_ms: i32) -> Result<bool, BridgeError> {
        let mut pollfd = libc::pollfd {
            fd: fd.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };

        // SAFETY: pollfd is valid and on the stack.
        let result = unsafe { libc::poll(&mut pollfd, 1, timeout_ms) };

        if result < 0 {
            let err = std::io::Error::last_os_error();
            return Err(BridgeError::Sync(format!("poll() failed: {}", err)));
        }

        Ok(result > 0 && (pollfd.revents & libc::POLLIN) != 0)
    }

    /// Clear pending signals after queue submission completes.
    /// Clear the pending signals tracking list.
    ///
    /// This just clears the tracking - the semaphores themselves are owned
    /// by their respective ExportSemaphore structs.
    pub fn clear_pending_signals(&self) {
        let mut pending = self.pending_signals.lock().expect("mutex poisoned");
        pending.clear();
        // Note: We do NOT move these to the pool because they're owned externally
    }

    /// Get statistics.
    pub fn stats(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.stats.semaphores_imported.load(Ordering::Relaxed),
            self.stats.semaphores_exported.load(Ordering::Relaxed),
            self.stats.semaphores_signaled.load(Ordering::Relaxed),
            self.stats.pool_hits.load(Ordering::Relaxed),
            self.stats.pool_misses.load(Ordering::Relaxed),
        )
    }

    /// Get a semaphore from pool or create new one.
    unsafe fn get_or_create_semaphore(&self, timeline: bool) -> Result<vk::Semaphore, BridgeError> {
        let pool = if timeline {
            &self.timeline_semaphore_pool
        } else {
            &self.binary_semaphore_pool
        };

        if let Some(semaphore) = pool.lock().expect("mutex poisoned").pop() {
            self.stats.pool_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(semaphore);
        }

        self.stats.pool_misses.fetch_add(1, Ordering::Relaxed);
        self.create_semaphore_internal(timeline, false)
    }

    /// Create a new Vulkan semaphore.
    unsafe fn create_semaphore_internal(
        &self,
        timeline: bool,
        exportable: bool,
    ) -> Result<vk::Semaphore, BridgeError> {
        let mut create_info = vk::SemaphoreCreateInfo::default();

        // For timeline semaphores
        let mut type_info;
        if timeline && self.capabilities.timeline_semaphore {
            type_info = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            create_info = create_info.push_next(&mut type_info);
        }

        // For exportable semaphores
        let mut export_info;
        if exportable && self.capabilities.can_export_sync_fd {
            export_info = vk::ExportSemaphoreCreateInfo::default()
                .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
            create_info = create_info.push_next(&mut export_info);
        }

        let semaphore = self
            .device_ref
            .device
            .create_semaphore(&create_info, None)
            .map_err(|e| BridgeError::Sync(format!("failed to create semaphore: {:?}", e)))?;

        Ok(semaphore)
    }
}

impl Drop for SyncManager {
    fn drop(&mut self) {
        // Clean up ONLY pooled semaphores - these are owned by the manager
        // Note: pending_signals contains raw handles to semaphores owned by
        // ExportSemaphore structs, which handle their own cleanup
        let binary_pool = self.binary_semaphore_pool.lock().expect("mutex poisoned");
        let timeline_pool = self.timeline_semaphore_pool.lock().expect("mutex poisoned");

        for sem in binary_pool.iter() {
            // SAFETY: device_ref.device is valid, and we own all pooled semaphores.
            unsafe { self.device_ref.device.destroy_semaphore(*sem, None) };
        }
        for sem in timeline_pool.iter() {
            // SAFETY: device_ref.device is valid, and we own all pooled semaphores.
            unsafe { self.device_ref.device.destroy_semaphore(*sem, None) };
        }
        // DO NOT destroy pending_signals - they're owned externally

        let (imported, exported, signaled, hits, misses) = self.stats();
        info!(
            "SyncManager shutdown: imported={}, exported={}, signaled={}, pool_hits={}, pool_misses={}",
            imported, exported, signaled, hits, misses
        );
    }
}

/// Test module for explicit sync functionality.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_point_binary() {
        let sp = SyncPoint::binary();
        assert_eq!(sp.timeline_value, 0);
        assert!(!sp.is_timeline);
    }

    #[test]
    fn test_sync_point_timeline() {
        let sp = SyncPoint::timeline(42);
        assert_eq!(sp.timeline_value, 42);
        assert!(sp.is_timeline);
    }

    #[test]
    fn test_capabilities_default() {
        let caps = SyncCapabilities::default();
        assert!(!caps.is_supported());
    }

    #[test]
    fn test_wait_sync_fd_timeout() {
        // Create a pipe to test poll behavior
        let mut fds = [0i32; 2];
        unsafe {
            if libc::pipe(fds.as_mut_ptr()) != 0 {
                panic!("Failed to create pipe");
            }
        }

        let read_fd = unsafe { OwnedFd::from_raw_fd(fds[0]) };
        let _write_fd = unsafe { OwnedFd::from_raw_fd(fds[1]) };

        // Poll with short timeout - should timeout since nothing written
        let result = SyncManager::wait_sync_fd(&read_fd, 10);
        assert!(result.is_ok());
        assert!(!result.unwrap()); // Should return false (timeout)
    }
}

/// Mock sync fd creation for integration tests.
#[cfg(test)]
pub mod mock_sync {
    use super::*;

    /// Create a mock sync fd using a signaled eventfd.
    pub fn create_signaled_fd() -> Result<OwnedFd, std::io::Error> {
        let fd = unsafe { libc::eventfd(1, libc::EFD_CLOEXEC) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }

        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    /// Create an unsignaled sync fd using eventfd.
    pub fn create_unsignaled_fd() -> Result<OwnedFd, std::io::Error> {
        let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_SEMAPHORE) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }

        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }
}
