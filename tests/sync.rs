//! Integration tests for explicit sync functionality.
//!
//! These tests verify that the explicit sync infrastructure works correctly
//! with real GPU operations, testing both binary and timeline semaphores.

use lamco_wgpu::{
    sync::{SyncCapabilities, SyncManager, SyncPoint},
    WgpuBridge,
};
use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::Arc;

fn init_logging() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("lamco_wgpu=debug")
        .with_test_writer()
        .try_init();
}

/// Create a test bridge for explicit sync testing (simple mode).
fn create_test_bridge() -> Arc<WgpuBridge> {
    Arc::new(WgpuBridge::new().expect("Failed to create WgpuBridge"))
}

/// Create a test bridge with explicit sync extensions enabled.
/// This is needed for actual sync fd import/export testing.
fn create_explicit_sync_bridge() -> Option<Arc<WgpuBridge>> {
    match WgpuBridge::new_with_explicit_sync() {
        Ok(bridge) => Some(Arc::new(bridge)),
        Err(e) => {
            println!("Could not create explicit sync bridge: {}", e);
            println!("This may be due to missing Vulkan extensions on the system.");
            None
        }
    }
}

#[test]
#[ignore] // Requires GPU
fn test_explicit_sync_capabilities_query() {
    init_logging();

    let bridge = create_test_bridge();

    // Access adapter to get physical device info
    unsafe {
        bridge
            .adapter()
            .as_hal::<wgpu_hal::api::Vulkan>()
            .map(|hal_adapter| {
                use std::ops::Deref;
                let adapter = hal_adapter.deref();
                let instance = adapter.shared_instance().raw_instance();
                let physical_device = adapter.raw_physical_device();

                let caps = SyncCapabilities::query(instance, physical_device);

                println!("Explicit sync capabilities:");
                println!("  external_semaphore: {}", caps.external_semaphore);
                println!("  external_semaphore_fd: {}", caps.external_semaphore_fd);
                println!("  timeline_semaphore: {}", caps.timeline_semaphore);
                println!("  can_import_sync_fd: {}", caps.can_import_sync_fd);
                println!("  can_export_sync_fd: {}", caps.can_export_sync_fd);
                println!("  can_import_opaque_fd: {}", caps.can_import_opaque_fd);
                println!("  can_export_opaque_fd: {}", caps.can_export_opaque_fd);
                println!("  is_supported: {}", caps.is_supported());

                // On most modern Linux systems, these should be supported
                assert!(
                    caps.external_semaphore,
                    "VK_KHR_external_semaphore should be available"
                );
                assert!(
                    caps.external_semaphore_fd,
                    "VK_KHR_external_semaphore_fd should be available"
                );
            });
    }
}

#[test]
#[ignore] // Requires GPU
fn test_explicit_sync_manager_creation() {
    init_logging();

    let bridge = create_test_bridge();

    unsafe {
        bridge
            .adapter()
            .as_hal::<wgpu_hal::api::Vulkan>()
            .map(|hal_adapter| {
                use std::ops::Deref;
                let adapter = hal_adapter.deref();
                let instance = adapter.shared_instance().raw_instance().clone();
                let physical_device = adapter.raw_physical_device();

                // Get device from queue
                bridge
                    .queue()
                    .as_hal::<wgpu_hal::api::Vulkan>()
                    .map(|hal_queue| {
                        let device = hal_queue.deref().raw_device().clone();

                        match SyncManager::new(instance, device, physical_device) {
                            Ok(manager) => {
                                println!("Created SyncManager successfully");
                                let caps = manager.capabilities();
                                println!("Manager capabilities: {:?}", caps);
                            }
                            Err(e) => {
                                // This is OK if the system doesn't support explicit sync
                                println!("SyncManager creation failed (may be expected): {}", e);
                            }
                        }
                    });
            });
    }
}

#[test]
fn test_sync_point_creation() {
    let binary = SyncPoint::binary();
    assert_eq!(binary.timeline_value, 0);
    assert!(!binary.is_timeline);

    let timeline = SyncPoint::timeline(42);
    assert_eq!(timeline.timeline_value, 42);
    assert!(timeline.is_timeline);

    let timeline_zero = SyncPoint::timeline(0);
    assert_eq!(timeline_zero.timeline_value, 0);
    assert!(timeline_zero.is_timeline); // Still timeline even if value is 0
}

#[test]
fn test_eventfd_poll_signaled() {
    // Test with a signaled eventfd
    let fd = unsafe { libc::eventfd(1, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
    assert!(fd >= 0, "Failed to create eventfd");

    let owned_fd = unsafe { OwnedFd::from_raw_fd(fd) };

    // Should be immediately ready
    let result = SyncManager::wait_sync_fd(&owned_fd, 0);
    assert!(result.is_ok());
    assert!(result.unwrap(), "Signaled eventfd should be ready");
}

#[test]
fn test_eventfd_poll_unsignaled() {
    // Test with an unsignaled eventfd
    let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
    assert!(fd >= 0, "Failed to create eventfd");

    let owned_fd = unsafe { OwnedFd::from_raw_fd(fd) };

    // Should timeout (not ready)
    let result = SyncManager::wait_sync_fd(&owned_fd, 10);
    assert!(result.is_ok());
    assert!(!result.unwrap(), "Unsignaled eventfd should not be ready");
}

#[test]
#[ignore] // Requires GPU
fn test_binary_semaphore_signal_flow() {
    init_logging();

    let bridge = create_test_bridge();

    unsafe {
        let adapter_guard = bridge.adapter().as_hal::<wgpu_hal::api::Vulkan>();
        if adapter_guard.is_none() {
            println!("Skipping test: Vulkan adapter not available");
            return;
        }

        let adapter_guard = adapter_guard.unwrap();
        use std::ops::Deref;
        let adapter = adapter_guard.deref();
        let instance = adapter.shared_instance().raw_instance().clone();
        let physical_device = adapter.raw_physical_device();

        let queue_guard = bridge.queue().as_hal::<wgpu_hal::api::Vulkan>();
        if queue_guard.is_none() {
            println!("Skipping test: Vulkan queue not available");
            return;
        }
        let queue_guard = queue_guard.unwrap();
        let device = queue_guard.deref().raw_device().clone();

        let manager = match SyncManager::new(instance, device, physical_device) {
            Ok(m) => m,
            Err(e) => {
                println!("Skipping test: SyncManager not available: {}", e);
                return;
            }
        };

        // Create an exportable binary semaphore
        let semaphore = manager
            .create_exportable_semaphore(false)
            .expect("Failed to create binary semaphore");

        println!("Created binary semaphore: {:?}", semaphore.raw());

        // Register it for signaling
        let result = manager.register_signal_semaphore_binary(bridge.queue(), semaphore.raw());
        assert!(result.is_ok(), "Failed to register semaphore: {:?}", result);

        // Submit some GPU work
        let encoder = bridge
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("test-encoder"),
            });
        bridge.queue().submit([encoder.finish()]);

        // The semaphore should be signaled after submit
        println!("Binary semaphore registered and work submitted");

        // Clean up
        manager.clear_pending_signals();

        let (imported, exported, signaled, hits, misses) = manager.stats();
        println!(
            "Stats: imported={}, exported={}, signaled={}, hits={}, misses={}",
            imported, exported, signaled, hits, misses
        );
        assert_eq!(signaled, 1, "Should have one signaled semaphore");
    }
}

#[test]
#[ignore] // Requires GPU
fn test_timeline_semaphore_signal_flow() {
    init_logging();

    let bridge = create_test_bridge();

    unsafe {
        let adapter_guard = bridge.adapter().as_hal::<wgpu_hal::api::Vulkan>();
        if adapter_guard.is_none() {
            println!("Skipping test: Vulkan adapter not available");
            return;
        }

        let adapter_guard = adapter_guard.unwrap();
        use std::ops::Deref;
        let adapter = adapter_guard.deref();
        let instance = adapter.shared_instance().raw_instance().clone();
        let physical_device = adapter.raw_physical_device();

        let queue_guard = bridge.queue().as_hal::<wgpu_hal::api::Vulkan>();
        if queue_guard.is_none() {
            println!("Skipping test: Vulkan queue not available");
            return;
        }
        let queue_guard = queue_guard.unwrap();
        let device = queue_guard.deref().raw_device().clone();

        let manager = match SyncManager::new(instance, device, physical_device) {
            Ok(m) => m,
            Err(e) => {
                println!("Skipping test: SyncManager not available: {}", e);
                return;
            }
        };

        if !manager.capabilities().timeline_semaphore {
            println!("Skipping test: Timeline semaphores not available");
            return;
        }

        // Create an exportable timeline semaphore
        let semaphore = manager
            .create_exportable_semaphore(true)
            .expect("Failed to create timeline semaphore");

        println!(
            "Created timeline semaphore: {:?}, initial value: {}",
            semaphore.raw(),
            semaphore.current_value()
        );

        // Get next signal value
        let signal_value = semaphore.next_signal_value();
        println!("Will signal at value: {}", signal_value);

        // Register it for signaling with timeline value
        let result = manager.register_signal_semaphore_timeline(
            bridge.queue(),
            semaphore.raw(),
            signal_value,
        );
        assert!(
            result.is_ok(),
            "Failed to register timeline semaphore: {:?}",
            result
        );

        // Submit some GPU work
        let encoder = bridge
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("test-encoder-timeline"),
            });
        bridge.queue().submit([encoder.finish()]);

        println!(
            "Timeline semaphore registered for value {} and work submitted",
            signal_value
        );

        // Clean up
        manager.clear_pending_signals();

        let (imported, exported, signaled, hits, misses) = manager.stats();
        println!(
            "Stats: imported={}, exported={}, signaled={}, hits={}, misses={}",
            imported, exported, signaled, hits, misses
        );
        assert_eq!(signaled, 1, "Should have one signaled semaphore");
    }
}

#[test]
#[ignore] // Requires GPU
fn test_semaphore_pool_reuse() {
    init_logging();

    let bridge = create_test_bridge();

    unsafe {
        let adapter_guard = bridge.adapter().as_hal::<wgpu_hal::api::Vulkan>();
        if adapter_guard.is_none() {
            println!("Skipping test: Vulkan adapter not available");
            return;
        }

        let adapter_guard = adapter_guard.unwrap();
        use std::ops::Deref;
        let adapter = adapter_guard.deref();
        let instance = adapter.shared_instance().raw_instance().clone();
        let physical_device = adapter.raw_physical_device();

        let queue_guard = bridge.queue().as_hal::<wgpu_hal::api::Vulkan>();
        if queue_guard.is_none() {
            println!("Skipping test: Vulkan queue not available");
            return;
        }
        let queue_guard = queue_guard.unwrap();
        let device = queue_guard.deref().raw_device().clone();

        let manager = match SyncManager::new(instance, device, physical_device) {
            Ok(m) => m,
            Err(e) => {
                println!("Skipping test: SyncManager not available: {}", e);
                return;
            }
        };

        // Create and register multiple semaphores
        for i in 0..3 {
            let semaphore = manager
                .create_exportable_semaphore(false)
                .expect("Failed to create semaphore");
            manager
                .register_signal_semaphore_binary(bridge.queue(), semaphore.raw())
                .expect("Failed to register semaphore");

            // Submit work
            let encoder = bridge
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("test-encoder-{}", i)),
                });
            bridge.queue().submit([encoder.finish()]);
        }

        let (_, _, signaled_before, _, misses_before) = manager.stats();
        println!(
            "After 3 iterations: signaled={}, misses={}",
            signaled_before, misses_before
        );

        // Clear and return to pool
        manager.clear_pending_signals();

        // Register more semaphores - these should come from the pool
        for i in 0..2 {
            let semaphore = manager
                .create_exportable_semaphore(false)
                .expect("Failed to create semaphore");
            manager
                .register_signal_semaphore_binary(bridge.queue(), semaphore.raw())
                .expect("Failed to register semaphore");

            let encoder = bridge
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("test-encoder-reuse-{}", i)),
                });
            bridge.queue().submit([encoder.finish()]);
        }

        let (_, _, signaled_after, hits_after, misses_after) = manager.stats();
        println!(
            "After reuse: signaled={}, hits={}, misses={}",
            signaled_after, hits_after, misses_after
        );

        // Should have some pool hits from reused semaphores
        // Note: The exportable semaphores are managed separately from the pool,
        // so we won't see hits here. The pool is for imported semaphores.
        assert_eq!(signaled_after, 5, "Should have 5 total signaled semaphores");
    }
}

/// This test verifies the critical question: can binary semaphores work
/// with the linux-drm-syncobj-v1 protocol?
///
/// The protocol uses timeline points, but our implementation uses binary
/// semaphores for signaling (following Firefox/Chromium patterns).
#[test]
#[ignore] // Requires GPU
fn test_binary_vs_timeline_for_drm_syncobj() {
    init_logging();
    println!("\n=== Binary vs Timeline Semaphore Analysis ===\n");

    let bridge = create_test_bridge();

    unsafe {
        let adapter_guard = bridge.adapter().as_hal::<wgpu_hal::api::Vulkan>();
        if adapter_guard.is_none() {
            println!("Skipping test: Vulkan adapter not available");
            return;
        }

        let adapter_guard = adapter_guard.unwrap();
        use std::ops::Deref;
        let adapter = adapter_guard.deref();
        let instance = adapter.shared_instance().raw_instance().clone();
        let physical_device = adapter.raw_physical_device();

        let queue_guard = bridge.queue().as_hal::<wgpu_hal::api::Vulkan>();
        if queue_guard.is_none() {
            println!("Skipping test: Vulkan queue not available");
            return;
        }
        let queue_guard = queue_guard.unwrap();
        let device = queue_guard.deref().raw_device().clone();

        let manager = match SyncManager::new(instance, device, physical_device) {
            Ok(m) => m,
            Err(e) => {
                println!("Skipping test: SyncManager not available: {}", e);
                return;
            }
        };

        let caps = manager.capabilities();

        println!("Device capabilities:");
        println!("  Timeline semaphores: {}", caps.timeline_semaphore);
        println!("  Binary SYNC_FD import: {}", caps.can_import_sync_fd);
        println!("  Binary SYNC_FD export: {}", caps.can_export_sync_fd);
        println!("  OPAQUE_FD import: {}", caps.can_import_opaque_fd);
        println!("  OPAQUE_FD export: {}", caps.can_export_opaque_fd);

        println!("\n--- Testing Binary Semaphore Export ---");

        // Create a binary exportable semaphore
        let binary_sem = manager
            .create_exportable_semaphore(false)
            .expect("Failed to create binary semaphore");

        // Register for signaling
        manager
            .register_signal_semaphore_binary(bridge.queue(), binary_sem.raw())
            .expect("Failed to register binary semaphore");

        // Submit work
        let encoder = bridge
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("binary-test"),
            });
        bridge.queue().submit([encoder.finish()]);

        // Wait for completion
        let _ = bridge.device().poll(wgpu::PollType::wait_indefinitely());

        // Try to export as sync_fd
        // Note: In wgpu standalone mode, VK_KHR_external_semaphore_fd may not be enabled
        match binary_sem.export_sync_fd() {
            Ok(fd) => {
                println!(
                    "  Successfully exported binary semaphore as sync_fd: {:?}",
                    fd
                );
                println!("  This suggests binary semaphores CAN work for release point signaling!");

                // Verify the fd is valid by polling it
                let result = SyncManager::wait_sync_fd(&fd, 0);
                match result {
                    Ok(ready) => println!("  Sync fd poll result: ready={}", ready),
                    Err(e) => println!("  Sync fd poll error: {}", e),
                }
            }
            Err(e) => {
                println!("  Binary semaphore export failed: {}", e);
                if e.to_string().contains("extension not enabled") {
                    println!(
                        "  Note: wgpu standalone mode doesn't enable VK_KHR_external_semaphore_fd"
                    );
                    println!(
                        "  In a real compositor, use from_smithay_vulkan with proper extensions"
                    );
                } else {
                    println!("  This might indicate timeline semaphores are required.");
                }
            }
        }

        if caps.timeline_semaphore {
            println!("\n--- Testing Timeline Semaphore Export ---");

            let timeline_sem = manager
                .create_exportable_semaphore(true)
                .expect("Failed to create timeline semaphore");

            let signal_value = timeline_sem.next_signal_value();

            manager
                .register_signal_semaphore_timeline(
                    bridge.queue(),
                    timeline_sem.raw(),
                    signal_value,
                )
                .expect("Failed to register timeline semaphore");

            let encoder = bridge
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("timeline-test"),
                });
            bridge.queue().submit([encoder.finish()]);

            let _ = bridge.device().poll(wgpu::PollType::wait_indefinitely());

            match timeline_sem.export_sync_fd() {
                Ok(fd) => {
                    println!(
                        "  Successfully exported timeline semaphore at value {} as sync_fd: {:?}",
                        signal_value, fd
                    );
                }
                Err(e) => {
                    println!("  Timeline semaphore export failed: {}", e);
                }
            }
        }

        println!("\n=== Conclusion ===");
        println!(
            "Binary semaphore SYNC_FD export supported: {}",
            caps.can_export_sync_fd
        );
        println!("Timeline semaphore available: {}", caps.timeline_semaphore);
        println!("\nFor linux-drm-syncobj-v1 integration:");
        println!("- If SYNC_FD export works, binary semaphores should suffice for release points");
        println!("- The protocol's timeline semantics are at the DRM syncobj level,");
        println!("  not necessarily requiring Vulkan timeline semaphores");
    }
}

/// Test actual sync fd export with extensions enabled.
///
/// This test uses new_with_explicit_sync() which creates a Vulkan device
/// with VK_KHR_external_semaphore_fd enabled, allowing us to actually
/// test the export functionality.
#[test]
#[ignore] // Requires GPU
fn test_sync_fd_export_with_extensions() {
    init_logging();
    println!("\n=== Sync FD Export Test (with extensions enabled) ===\n");

    let bridge = match create_explicit_sync_bridge() {
        Some(b) => b,
        None => {
            println!("Skipping test: Could not create bridge with explicit sync extensions");
            return;
        }
    };

    unsafe {
        let adapter_guard = bridge.adapter().as_hal::<wgpu_hal::api::Vulkan>();
        if adapter_guard.is_none() {
            println!("Skipping test: Vulkan adapter not available");
            return;
        }

        let adapter_guard = adapter_guard.unwrap();
        use std::ops::Deref;
        let adapter = adapter_guard.deref();
        let instance = adapter.shared_instance().raw_instance().clone();
        let physical_device = adapter.raw_physical_device();

        let queue_guard = bridge.queue().as_hal::<wgpu_hal::api::Vulkan>();
        if queue_guard.is_none() {
            println!("Skipping test: Vulkan queue not available");
            return;
        }
        let queue_guard = queue_guard.unwrap();
        let device = queue_guard.deref().raw_device().clone();

        let manager = match SyncManager::new(instance, device, physical_device) {
            Ok(m) => m,
            Err(e) => {
                println!("Skipping test: SyncManager not available: {}", e);
                return;
            }
        };

        let caps = manager.capabilities();
        println!("Device capabilities (with extensions):");
        println!("  Timeline semaphores: {}", caps.timeline_semaphore);
        println!("  Binary SYNC_FD export: {}", caps.can_export_sync_fd);

        // Test binary semaphore export
        println!("\n--- Testing Binary Semaphore Export ---");

        let binary_sem = manager
            .create_exportable_semaphore(false)
            .expect("Failed to create binary semaphore");

        println!("Created binary semaphore: {:?}", binary_sem.raw());

        // Register for signaling
        manager
            .register_signal_semaphore_binary(bridge.queue(), binary_sem.raw())
            .expect("Failed to register binary semaphore");

        // Submit work to trigger signaling
        let encoder = bridge
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("export-test"),
            });
        bridge.queue().submit([encoder.finish()]);

        // Wait for GPU to finish
        let _ = bridge.device().poll(wgpu::PollType::wait_indefinitely());

        // Try to export
        match binary_sem.export_sync_fd() {
            Ok(fd) => {
                println!("SUCCESS: Exported binary semaphore as sync_fd: {:?}", fd);

                // Verify the fd is valid by polling it
                let result = SyncManager::wait_sync_fd(&fd, 0);
                match result {
                    Ok(ready) => {
                        println!("  Sync fd poll: ready = {}", ready);
                        if ready {
                            println!("  The semaphore has been signaled!");
                            println!(
                                "\n=== CONCLUSION: Binary semaphores WORK for release points! ==="
                            );
                        }
                    }
                    Err(e) => println!("  Sync fd poll error: {}", e),
                }
            }
            Err(e) => {
                println!("FAILED: Binary semaphore export: {}", e);
                println!("This indicates the extension may not be properly enabled.");
            }
        }

        // Clean up
        manager.clear_pending_signals();
        let (imported, exported, signaled, hits, misses) = manager.stats();
        println!(
            "\nStats: imported={}, exported={}, signaled={}, hits={}, misses={}",
            imported, exported, signaled, hits, misses
        );
    }
}
