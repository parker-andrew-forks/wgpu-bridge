//! Test dmabuf import using GBM to create real dmabufs.
//!
//! This test requires:
//! - A GPU with DRM render node (/dev/dri/renderD128)
//! - The `dmabuf` feature enabled (enabled by default)
//!
//! Run with:
//! ```sh
//! cargo test dmabuf -- --test-threads=1 --nocapture
//! ```

#![cfg(feature = "dmabuf")]

use lamco_wgpu::bridge::WgpuBridge;
use smithay::backend::allocator::dmabuf::{AsDmabuf, Dmabuf};
use smithay::backend::allocator::gbm::{GbmAllocator, GbmBuffer, GbmBufferFlags, GbmDevice};
use smithay::backend::allocator::{Allocator, Fourcc, Modifier};
use smithay::backend::renderer::Texture;
use std::fs::File;
use std::sync::Arc;

/// DRM_FORMAT_MOD_LINEAR
const LINEAR: Modifier = Modifier::Linear;

/// Open a render node file, trying multiple paths.
fn open_render_node() -> Option<File> {
    // Try render nodes first, then primary cards
    for path in &[
        "/dev/dri/renderD128",
        "/dev/dri/renderD129",
        "/dev/dri/card0",
        "/dev/dri/card1",
    ] {
        if let Ok(file) = File::options().read(true).write(true).open(path) {
            println!("Opened DRI device: {}", path);
            return Some(file);
        }
    }
    None
}

/// Try to allocate a GBM buffer with various fallbacks.
fn try_allocate_buffer(
    allocator: &mut GbmAllocator<File>,
    width: u32,
    height: u32,
    format: Fourcc,
) -> Option<GbmBuffer> {
    // Try with LINEAR modifier first (most compatible)
    println!("  Trying LINEAR modifier...");
    if let Ok(buffer) = allocator.create_buffer(width, height, format, &[LINEAR]) {
        println!("  Success with LINEAR modifier");
        return Some(buffer);
    }

    // Try without any modifier constraints
    println!("  Trying without modifier constraints...");
    if let Ok(buffer) = allocator.create_buffer(width, height, format, &[]) {
        println!("  Success without modifiers");
        return Some(buffer);
    }

    // Try XRGB if ARGB fails
    if format == Fourcc::Argb8888 {
        println!("  Trying XRGB8888 instead...");
        if let Ok(buffer) = allocator.create_buffer(width, height, Fourcc::Xrgb8888, &[LINEAR]) {
            println!("  Success with XRGB8888");
            return Some(buffer);
        }
    }

    None
}

/// Test importing a real dmabuf created with GBM.
#[test]
fn test_dmabuf_import_argb8888() {
    // Open render node
    let render_node = match open_render_node() {
        Some(f) => f,
        None => {
            eprintln!("Skipping test: no DRI device available");
            return;
        }
    };

    // Create GBM device
    let gbm: GbmDevice<File> = match GbmDevice::new(render_node) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test: failed to open GBM device: {:?}", e);
            return;
        }
    };

    println!("Created GBM device");

    // Create GBM allocator with minimal flags for compatibility
    let mut allocator: GbmAllocator<File> = GbmAllocator::new(gbm, GbmBufferFlags::RENDERING);

    // Allocate a buffer
    let width = 256u32;
    let height = 256u32;

    println!("Allocating {}x{} buffer...", width, height);

    let buffer = match try_allocate_buffer(&mut allocator, width, height, Fourcc::Argb8888) {
        Some(b) => b,
        None => {
            eprintln!("Skipping test: failed to allocate buffer (driver may not support GBM)");
            return;
        }
    };

    println!("Created GBM buffer");

    // Export as dmabuf
    let dmabuf: Dmabuf = match buffer.export() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Skipping test: failed to export dmabuf: {:?}", e);
            return;
        }
    };

    println!("Exported dmabuf: planes={}", dmabuf.handles().count());

    // Create wgpu bridge
    let bridge = match WgpuBridge::new() {
        Ok(b) => Arc::new(b),
        Err(e) => {
            eprintln!("Skipping test: failed to create bridge: {:?}", e);
            return;
        }
    };

    println!(
        "Created WgpuBridge with adapter: {}",
        bridge.adapter().get_info().name
    );

    // Import the dmabuf
    println!("Importing dmabuf...");
    let texture = match unsafe { bridge.import_dmabuf(&dmabuf) } {
        Ok(t) => t,
        Err(e) => {
            eprintln!("FAILED to import dmabuf: {:?}", e);
            panic!("Dmabuf import failed: {:?}", e);
        }
    };

    println!("SUCCESS! Imported dmabuf as wgpu texture:");
    println!("  Size: {}x{}", texture.width(), texture.height());
    println!("  Format: {:?}", texture.wgpu_format());
    println!("  External: {}", texture.is_external());

    // Verify dimensions match
    assert_eq!(texture.width(), width);
    assert_eq!(texture.height(), height);

    // Clean up
    let _ = bridge.device().poll(wgpu::PollType::wait_indefinitely());

    println!("Test passed!");
}

/// Test that we can create a texture view from imported dmabuf.
#[test]
fn test_dmabuf_texture_view() {
    let render_node = match open_render_node() {
        Some(f) => f,
        None => {
            eprintln!("Skipping test: no DRI device available");
            return;
        }
    };

    let gbm: GbmDevice<File> = match GbmDevice::new(render_node) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test: {:?}", e);
            return;
        }
    };

    let mut allocator: GbmAllocator<File> = GbmAllocator::new(gbm, GbmBufferFlags::RENDERING);

    let buffer = match try_allocate_buffer(&mut allocator, 128, 128, Fourcc::Argb8888) {
        Some(b) => b,
        None => {
            eprintln!("Skipping test: failed to allocate buffer");
            return;
        }
    };

    let dmabuf: Dmabuf = match buffer.export() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Skipping test: {:?}", e);
            return;
        }
    };

    let bridge = match WgpuBridge::new() {
        Ok(b) => Arc::new(b),
        Err(e) => {
            eprintln!("Skipping test: {:?}", e);
            return;
        }
    };

    let texture = match unsafe { bridge.import_dmabuf(&dmabuf) } {
        Ok(t) => t,
        Err(e) => {
            panic!("Import failed: {:?}", e);
        }
    };

    // Create a texture view - this exercises the wgpu texture wrapper
    let view = texture.view();
    println!("Created texture view from imported dmabuf");

    // Try to use it in a bind group (validates the view is usable)
    let bind_group_layout =
        bridge
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("test-layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

    let _bind_group = bridge
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("test-bind-group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            }],
        });

    println!("Successfully created bind group with imported dmabuf texture!");

    let _ = bridge.device().poll(wgpu::PollType::wait_indefinitely());
}
