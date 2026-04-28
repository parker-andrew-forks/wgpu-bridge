//! Tests for sync FD export and output scanout functionality.
//!
//! These tests require a GPU with Vulkan support and the necessary
//! external memory/semaphore extensions.

use lamco_wgpu::scanout::{RenderTargetConfig, ScanoutCapabilities};
use lamco_wgpu::sync::SyncCapabilities;
// Note: FrameSyncManager and RenderTarget require raw Vulkan access
// and are tested in explicit_sync.rs with new_with_explicit_sync().
use lamco_wgpu::WgpuBridge;
use std::sync::Arc;

fn create_test_bridge() -> Arc<WgpuBridge> {
    Arc::new(WgpuBridge::new().expect("Failed to create WgpuBridge"))
}

// === Sync Tests ===

#[test]
fn test_sync_capabilities_query() {
    // This test requires accessing the raw Vulkan instance which
    // wgpu doesn't easily expose. Skip for now.
    // The unit test in sync.rs tests the default capabilities.
    let caps = SyncCapabilities::default();
    assert!(!caps.can_export_sync_fd);
    println!("Sync capabilities (default): {:?}", caps);
}

#[test]
#[ignore] // Requires GPU
fn test_frame_sync_allocation() {
    // Test that we can create a bridge and understand sync requirements
    let bridge = create_test_bridge();

    // The bridge should be created successfully
    println!("Bridge created: {:?}", bridge);
    println!("Adapter: {}", bridge.adapter().get_info().name);

    // In a full implementation, we'd:
    // 1. Query sync capabilities from the Vulkan device
    // 2. Create a FrameSyncManager
    // 3. Allocate frame sync points
    // 4. Export sync FDs
    //
    // For now, this test just verifies the bridge works
}

// === Scanout Tests ===

#[test]
fn test_scanout_capabilities_default() {
    let caps = ScanoutCapabilities::default();
    assert!(!caps.can_export_dmabuf);
    println!("Scanout capabilities (default): {:?}", caps);
}

#[test]
fn test_render_target_config() {
    let config = RenderTargetConfig::default();
    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
    assert_eq!(config.format, drm_fourcc::DrmFourcc::Argb8888);
    assert_eq!(config.modifier, 0); // LINEAR
    println!("Default render target config: {:?}", config);
}

#[test]
fn test_render_target_config_custom() {
    let config = RenderTargetConfig {
        width: 2560,
        height: 1440,
        format: drm_fourcc::DrmFourcc::Xrgb8888,
        modifier: 0,
    };
    assert_eq!(config.width, 2560);
    assert_eq!(config.height, 1440);
    println!("Custom render target config: {:?}", config);
}

// === Integration Test: Full Pipeline ===

/// This test demonstrates the full compositor pipeline:
/// 1. Create bridge
/// 2. Create exportable render target
/// 3. Render into it
/// 4. (Would export as dmabuf for display)
///
/// Note: Actually exporting requires raw Vulkan handles that aren't
/// easily accessible through wgpu's safe API.
#[test]
#[ignore] // Requires GPU
fn test_full_render_pipeline_concept() {
    use lamco_wgpu::renderer::WgpuFramebuffer;
    use lamco_wgpu::WgpuRenderer;
    use smithay::backend::renderer::{Frame, Renderer};
    use smithay::utils::{Size, Transform};

    let bridge = create_test_bridge();
    let mut renderer = WgpuRenderer::new(bridge.clone());

    // Create a framebuffer (internal render target)
    let mut fb = WgpuFramebuffer::new(bridge.device(), 1920, 1080);

    // Render a frame
    let mut frame = renderer
        .render(&mut fb, Size::from((1920, 1080)), Transform::Normal)
        .expect("Failed to begin frame");

    // Clear to a color
    use smithay::backend::renderer::Color32F;
    frame
        .clear(Color32F::new(0.1, 0.2, 0.3, 1.0), &[])
        .expect("Failed to clear");

    // Finish the frame
    let sync = frame.finish().expect("Failed to finish frame");

    println!("Rendered frame successfully");
    println!("Sync point: {:?}", sync);

    // In a full implementation, we would:
    // 1. Create an RenderTarget instead of WgpuFramebuffer
    // 2. Export the rendered result as a dmabuf
    // 3. Pass the dmabuf to the display output
    // 4. Export a sync FD for frame timing
}

/// Test that demonstrates the intended API flow for sync FD export.
/// This doesn't actually test the Vulkan functionality (requires raw handles),
/// but documents the expected usage pattern.
#[test]
fn test_sync_fd_api_pattern() {
    println!("Sync FD Export API Pattern:");
    println!("1. FrameSyncManager::new(instance, device, capabilities)");
    println!("2. let (timeline_point, semaphore) = manager.allocate_frame_sync()");
    println!("3. Submit GPU work signaling the semaphore at timeline_point");
    println!("4. let sync_fd = manager.export_sync_fd(timeline_point)");
    println!("5. Pass sync_fd to Wayland linux-drm-syncobj-v1 protocol");
}

/// Test that demonstrates the intended API flow for dmabuf export.
#[test]
fn test_dmabuf_export_api_pattern() {
    println!("DMA-BUF Export API Pattern:");
    println!("1. RenderTarget::new(device, instance, phys_dev, config)");
    println!("2. Get texture/view for rendering into the target");
    println!("3. Render frame into the target");
    println!("4. Wait for GPU (or use sync FD)");
    println!("5. let dmabuf = target.export_dmabuf()");
    println!("6. Pass dmabuf to display output / scanout");
}

// === Format Tests ===

#[test]
#[ignore] // Requires GPU
fn test_drm_format_support() {
    let bridge = create_test_bridge();
    let formats = bridge.supported_formats();

    println!("Supported DRM formats:");
    for format in formats {
        println!("  {:?} -> {:?}", format.fourcc, format.wgpu_format);
    }

    // Should support at least ARGB8888
    assert!(bridge.supports_format(drm_fourcc::DrmFourcc::Argb8888, 0));
}
