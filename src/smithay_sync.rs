//! Integration between SyncManager and Smithay's drm_syncobj module.
//!
//! This module bridges our wgpu-based explicit sync with Smithay's
//! `linux-drm-syncobj-v1` protocol implementation.
//!
//! # Architecture
//!
//! ```text
//! Smithay DrmSyncobjState
//!       ↓
//! DrmSyncobjCachedState (per-surface acquire/release points)
//!       ↓
//! SyncBridge (this module)
//!       ↓
//! SyncManager → VkSemaphore → GPU signaling
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use lamco_wgpu::smithay_sync::SyncBridge;
//!
//! // In your frame rendering:
//! let sync_bridge = SyncBridge::new(&sync_manager);
//!
//! // Before sampling buffer - wait for acquire point
//! if let Some(acquire) = &cached_state.acquire_point {
//!     sync_bridge.wait_acquire_point(acquire)?;
//! }
//!
//! // After rendering - signal release point
//! if let Some(release) = &cached_state.release_point {
//!     sync_bridge.signal_release_point(&queue, release)?;
//! }
//! ```

use crate::error::BridgeError;
use crate::sync::SyncManager;
use smithay::backend::renderer::sync::Fence;
use smithay::wayland::drm_syncobj::DrmSyncPoint;
use std::sync::Arc;
use tracing::{debug, trace, warn};

/// Bridge between Smithay's DrmSyncPoint and our SyncManager.
///
/// This handles the conversion between DRM syncobj timeline points
/// and Vulkan semaphores for GPU-side synchronization.
pub struct SyncBridge<'a> {
    /// Reference to the explicit sync manager
    manager: &'a SyncManager,
}

impl<'a> SyncBridge<'a> {
    /// Create a new bridge with the given SyncManager.
    pub fn new(manager: &'a SyncManager) -> Self {
        Self { manager }
    }

    /// Wait for an acquire point before accessing a buffer.
    ///
    /// This should be called before the GPU reads from a client buffer.
    /// The acquire point indicates when the client's GPU work is complete.
    ///
    /// # Current Implementation
    ///
    /// Currently uses CPU-side waiting via poll() on the exported sync_file.
    /// A future optimization would be to import as a Vulkan semaphore and
    /// use GPU-side waiting via VkSubmitInfo::pWaitSemaphores.
    ///
    /// # Arguments
    ///
    /// * `acquire_point` - The DrmSyncPoint from the surface's cached state
    /// * `timeout_ms` - Timeout in milliseconds (0 = non-blocking check)
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - Acquire point is signaled, safe to access buffer
    /// * `Ok(false)` - Timeout reached, acquire point not yet signaled
    /// * `Err(_)` - Error waiting for acquire point
    pub fn wait_acquire_point(
        &self,
        acquire_point: &DrmSyncPoint,
        timeout_ms: u32,
    ) -> Result<bool, BridgeError> {
        if acquire_point.is_signaled() {
            trace!("Acquire point already signaled");
            return Ok(true);
        }

        if timeout_ms == 0 {
            return Ok(false);
        }

        let sync_fd = acquire_point.export_sync_file().map_err(|e| {
            BridgeError::Sync(format!("failed to export acquire sync_file: {:?}", e))
        })?;

        SyncManager::wait_sync_fd(&sync_fd, timeout_ms as i32)
    }

    /// Wait for an acquire point on the GPU side (blocking the queue).
    ///
    /// This imports the acquire point as a Vulkan semaphore and could be
    /// used for GPU-side waiting. Currently falls back to CPU waiting
    /// because wgpu doesn't expose wait semaphore APIs.
    ///
    /// # Future Enhancement
    ///
    /// When wgpu adds `Queue::add_wait_semaphore()`, this method can
    /// be updated to do true GPU-side waiting without blocking the CPU.
    pub fn wait_acquire_point_gpu(&self, acquire_point: &DrmSyncPoint) -> Result<(), BridgeError> {
        // For now, use CPU-side waiting
        // TODO: When wgpu supports wait semaphores, import and wait on GPU
        self.wait_acquire_point(acquire_point, 5000)?; // 5 second timeout
        Ok(())
    }

    /// Signal a release point after GPU work completes.
    ///
    /// This creates a Vulkan semaphore, registers it for GPU signaling,
    /// and after the work completes, signals the DrmSyncPoint to notify
    /// the client that the buffer can be reused.
    ///
    /// # Current Implementation
    ///
    /// Uses a hybrid approach:
    /// 1. Create and register a Vulkan semaphore for GPU signaling
    /// 2. Submit GPU work
    /// 3. CPU waits for semaphore to be signaled
    /// 4. CPU signals the DrmSyncPoint
    ///
    /// This is necessary because we can't directly import the DRM syncobj
    /// as a Vulkan semaphore for GPU-side signaling (Smithay doesn't expose
    /// the raw syncobj fd).
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue (for registering signal semaphore)
    /// * `release_point` - The DrmSyncPoint to signal when done
    ///
    /// # Note
    ///
    /// This method should be called BEFORE queue.submit(). The actual
    /// signaling happens after submit completes.
    pub fn prepare_release_signal(
        &self,
        queue: &wgpu::Queue,
    ) -> Result<PendingReleaseSignal, BridgeError> {
        let caps = self.manager.capabilities();

        if !caps.can_export_sync_fd {
            return Err(BridgeError::Sync(
                "SYNC_FD export not supported, cannot signal release points".into(),
            ));
        }

        // SAFETY: manager is valid and we check capabilities above.
        let semaphore = unsafe { self.manager.create_exportable_semaphore(false)? };

        // SAFETY: queue is valid and semaphore was just created.
        unsafe {
            self.manager
                .register_signal_semaphore_binary(queue, semaphore.raw())?;
        }

        debug!("Prepared release signal semaphore {:?}", semaphore.raw());

        Ok(PendingReleaseSignal {
            semaphore: Arc::new(semaphore),
        })
    }

    /// Complete the release signal after GPU work is submitted.
    ///
    /// Call this after queue.submit() to wait for the GPU and signal
    /// the DrmSyncPoint.
    ///
    /// # Arguments
    ///
    /// * `pending` - The pending release signal from prepare_release_signal()
    /// * `release_point` - The DrmSyncPoint to signal
    /// * `device` - The wgpu device (for polling)
    pub fn complete_release_signal(
        &self,
        pending: PendingReleaseSignal,
        release_point: &DrmSyncPoint,
        device: &wgpu::Device,
    ) -> Result<(), BridgeError> {
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        // SAFETY: semaphore is valid and was signaled by the GPU.
        let sync_fd = unsafe { pending.semaphore.export_sync_fd()? };

        let ready = SyncManager::wait_sync_fd(&sync_fd, 0)?;
        if !ready {
            warn!("Release semaphore not signaled after device poll");
            SyncManager::wait_sync_fd(&sync_fd, 1000)?;
        }

        release_point
            .signal()
            .map_err(|e| BridgeError::Sync(format!("failed to signal release point: {:?}", e)))?;

        debug!("Signaled release point");
        Ok(())
    }

    /// Convenience method to handle the full release signal flow.
    ///
    /// This combines prepare_release_signal and complete_release_signal
    /// into a callback-based flow.
    ///
    /// # Example
    ///
    /// ```ignore
    /// sync_bridge.with_release_signal(&queue, &device, &release_point, || {
    ///     // Your GPU work here
    ///     queue.submit([encoder.finish()]);
    /// })?;
    /// ```
    pub fn with_release_signal<F>(
        &self,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        release_point: &DrmSyncPoint,
        work: F,
    ) -> Result<(), BridgeError>
    where
        F: FnOnce(),
    {
        let pending = self.prepare_release_signal(queue)?;
        work();
        self.complete_release_signal(pending, release_point, device)
    }
}

/// A pending release signal that needs to be completed after GPU submit.
pub struct PendingReleaseSignal {
    semaphore: Arc<crate::sync::ExportSemaphore>,
}

impl PendingReleaseSignal {
    pub fn raw_semaphore(&self) -> ash::vk::Semaphore {
        self.semaphore.raw()
    }
}

/// Extension trait for convenient explicit sync handling on surfaces.
///
/// This trait can be implemented for compositor-specific surface types
/// to provide ergonomic sync point handling.
pub trait SyncSurface {
    /// Get the acquire point for this surface, if any.
    fn acquire_point(&self) -> Option<&DrmSyncPoint>;

    /// Get the release point for this surface, if any.
    fn release_point(&self) -> Option<&DrmSyncPoint>;

    /// Check if this surface uses explicit sync.
    fn uses_explicit_sync(&self) -> bool {
        self.acquire_point().is_some() && self.release_point().is_some()
    }
}

/// Helper to process multiple surfaces with explicit sync.
pub struct SyncProcessor<'a> {
    bridge: SyncBridge<'a>,
    pending_releases: Vec<(PendingReleaseSignal, DrmSyncPoint)>,
}

impl<'a> SyncProcessor<'a> {
    /// Create a new surface sync processor.
    pub fn new(manager: &'a SyncManager) -> Self {
        Self {
            bridge: SyncBridge::new(manager),
            pending_releases: Vec::new(),
        }
    }

    /// Wait for all acquire points to be ready.
    ///
    /// Returns the number of surfaces that were waited on.
    pub fn wait_all_acquires<S: SyncSurface>(
        &self,
        surfaces: &[S],
        timeout_ms: u32,
    ) -> Result<usize, BridgeError> {
        let mut waited = 0;
        for surface in surfaces {
            if let Some(acquire) = surface.acquire_point() {
                self.bridge.wait_acquire_point(acquire, timeout_ms)?;
                waited += 1;
            }
        }
        Ok(waited)
    }

    /// Prepare release signals for all surfaces.
    ///
    /// Call this before queue.submit().
    pub fn prepare_all_releases<S: SyncSurface>(
        &mut self,
        surfaces: &[S],
        queue: &wgpu::Queue,
    ) -> Result<(), BridgeError> {
        for surface in surfaces {
            if let Some(release) = surface.release_point() {
                let pending = self.bridge.prepare_release_signal(queue)?;
                self.pending_releases.push((pending, release.clone()));
            }
        }
        Ok(())
    }

    /// Complete all pending release signals.
    ///
    /// Call this after queue.submit() completes.
    pub fn complete_all_releases(&mut self, device: &wgpu::Device) -> Result<(), BridgeError> {
        for (pending, release) in self.pending_releases.drain(..) {
            self.bridge
                .complete_release_signal(pending, &release, device)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // Note: Full integration tests require a DRM device with syncobj support.
    // These tests verify the API structure compiles correctly.

    #[test]
    fn test_sync_bridge_creation() {
        // This test verifies the module compiles.
        // Full testing requires SyncManager (needs Vulkan) and DrmSyncPoint (needs DRM).
    }
}
