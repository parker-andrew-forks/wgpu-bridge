//! Integration tests for smithay-wgpu-bridge.
//!
//! These tests require a Vulkan-capable GPU.
//!
//! **Note:** Tests must be run single-threaded due to GPU resource contention:
//! ```sh
//! cargo test -- --test-threads=1
//! ```

use lamco_wgpu::bridge::WgpuBridge;
use lamco_wgpu::drm_mod;
use std::sync::Arc;

/// Test that we can create a WgpuBridge successfully.
#[test]
fn test_bridge_creation() {
    // Skip test if no Vulkan device available
    let bridge = match WgpuBridge::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Skipping test: {:?}", e);
            return;
        }
    };

    // Verify we got a valid device
    let info = bridge.adapter().get_info();
    println!("Using adapter: {}", info.name);
    println!("Backend: {:?}", info.backend);
    println!("Device type: {:?}", info.device_type);

    // Check that we have some supported formats
    let formats = bridge.supported_formats();
    assert!(
        !formats.is_empty(),
        "Should have at least one supported format"
    );
    println!("Supported formats: {}", formats.len());

    // Explicit poll before drop
    let _ = bridge.device().poll(wgpu::PollType::wait_indefinitely());
}

/// Test format support checking.
#[test]
fn test_format_support() {
    let bridge = match WgpuBridge::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Skipping test: {:?}", e);
            return;
        }
    };

    // ARGB8888 should be universally supported
    assert!(
        bridge.supports_format(drm_fourcc::DrmFourcc::Argb8888, 0),
        "ARGB8888 should be supported"
    );

    // Check supported formats list
    for format in bridge.supported_formats() {
        println!(
            "Format: {:?} -> {:?} (modifiers: {:?})",
            format.fourcc, format.wgpu_format, format.modifiers
        );
    }

    // Explicit poll before drop
    let _ = bridge.device().poll(wgpu::PollType::wait_indefinitely());
}

/// Test basic texture creation (non-dmabuf).
#[test]
fn test_texture_creation() {
    let bridge = match WgpuBridge::new() {
        Ok(b) => Arc::new(b),
        Err(e) => {
            eprintln!("Skipping test: {:?}", e);
            return;
        }
    };

    // Create a simple texture
    let texture = bridge.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("test-texture"),
        size: wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Create a view
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    println!("Successfully created 256x256 BGRA8 texture");

    // Explicit drop in correct order
    drop(view);
    drop(texture);

    // Poll to ensure cleanup
    let _ = bridge.device().poll(wgpu::PollType::wait_indefinitely());
}

/// Test modifier capabilities query.
#[test]
fn test_modifier_capabilities() {
    let bridge = match WgpuBridge::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Skipping test: {:?}", e);
            return;
        }
    };

    let caps = bridge.modifier_capabilities();
    println!("Modifier capabilities:");
    println!("  Extension available: {}", caps.extension_available);
    println!("  Extension enabled: {}", caps.extension_enabled);
    println!("  Max planes: {}", caps.max_planes);
    println!("  Supports modifiers: {}", caps.supports_modifiers());

    // In standalone mode (WgpuBridge::new), extension is typically not enabled
    // because we don't control the device creation extensions
    // This is expected - the extension is only enabled in from_smithay_vulkan_full mode

    let _ = bridge.device().poll(wgpu::PollType::wait_indefinitely());
}

/// Test format modifier queries.
#[test]
fn test_format_modifiers() {
    let bridge = match WgpuBridge::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Skipping test: {:?}", e);
            return;
        }
    };

    // Check modifiers for ARGB8888
    if let Some(modifiers) = bridge.get_format_modifiers(drm_fourcc::DrmFourcc::Argb8888) {
        println!("ARGB8888 modifiers ({}):", modifiers.len());
        for props in modifiers {
            println!(
                "  - {} (planes={}, sampling={}, render={})",
                props.describe(),
                props.plane_count,
                props.supports_sampling(),
                props.supports_color_attachment()
            );
        }

        // Should have at least LINEAR modifier
        let has_linear = modifiers.iter().any(|m| m.modifier == drm_mod::LINEAR);
        assert!(has_linear, "Should support LINEAR modifier for ARGB8888");
    } else {
        panic!("ARGB8888 should be supported");
    }

    let _ = bridge.device().poll(wgpu::PollType::wait_indefinitely());
}

/// Test modifier properties lookup.
#[test]
fn test_modifier_properties_lookup() {
    let bridge = match WgpuBridge::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Skipping test: {:?}", e);
            return;
        }
    };

    // LINEAR modifier should be supported for ARGB8888
    let props = bridge.get_modifier_properties(drm_fourcc::DrmFourcc::Argb8888, drm_mod::LINEAR);

    if let Some(props) = props {
        println!("LINEAR modifier properties:");
        println!("  Modifier: {:#x}", props.modifier);
        println!("  Plane count: {}", props.plane_count);
        println!("  Description: {}", props.describe());

        assert_eq!(props.modifier, drm_mod::LINEAR);
        assert_eq!(props.plane_count, 1, "LINEAR should be single-plane");
    } else {
        println!("LINEAR modifier not found - may not have modifier extension support");
    }

    // INVALID modifier should not be supported
    let invalid_props =
        bridge.get_modifier_properties(drm_fourcc::DrmFourcc::Argb8888, drm_mod::INVALID);
    assert!(
        invalid_props.is_none(),
        "INVALID modifier should not be in supported list"
    );

    let _ = bridge.device().poll(wgpu::PollType::wait_indefinitely());
}

/// Test supports_modifiers helper.
#[test]
fn test_supports_modifiers() {
    let bridge = match WgpuBridge::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Skipping test: {:?}", e);
            return;
        }
    };

    // In standalone mode, modifiers are typically NOT enabled
    // (requires from_smithay_vulkan_full with explicit extension enabling)
    let supports = bridge.supports_modifiers();
    println!("Bridge supports modifiers: {}", supports);

    // This is informational - we don't assert on the value because
    // it depends on the device creation path

    let _ = bridge.device().poll(wgpu::PollType::wait_indefinitely());
}
