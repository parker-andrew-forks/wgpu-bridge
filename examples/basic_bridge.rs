//! Basic example demonstrating WgpuBridge creation and texture operations.
//!
//! Run with:
//! ```sh
//! cargo run --example basic_bridge
//! ```

use lamco_wgpu::bridge::WgpuBridge;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info,lamco_wgpu=debug")
        .init();

    println!("Creating WgpuBridge...");

    // Create the bridge
    let bridge = Arc::new(WgpuBridge::new()?);

    // Print adapter info
    let info = bridge.adapter().get_info();
    println!();
    println!("=== Adapter Info ===");
    println!("  Name: {}", info.name);
    println!("  Backend: {:?}", info.backend);
    println!("  Device Type: {:?}", info.device_type);
    println!("  Vendor ID: {:#x}", info.vendor);
    println!("  Device ID: {:#x}", info.device);

    // Print supported formats
    println!();
    println!("=== Supported DMA-BUF Formats ===");
    for format in bridge.supported_formats() {
        println!("  {:?} -> {:?}", format.fourcc, format.wgpu_format);
    }

    // Create a test texture
    println!();
    println!("=== Creating Test Texture ===");
    let texture = bridge.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("test-texture"),
        size: wgpu::Extent3d {
            width: 1920,
            height: 1080,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    println!("  Created 1920x1080 BGRA8 texture");

    // Create a command encoder and do a clear
    println!();
    println!("=== Render Test ===");
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = bridge
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test-encoder"),
        });

    {
        // Begin a render pass with a clear color
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("test-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        println!("  Cleared texture to blue-gray color");
    }

    // Submit the commands
    bridge.queue().submit(Some(encoder.finish()));
    let _ = bridge.device().poll(wgpu::PollType::wait_indefinitely());
    println!("  Submitted and waited for GPU");

    println!();
    println!("Success! The bridge is working correctly.");
    println!();
    println!("DMA-BUF import is enabled by default.");
    println!("Use bridge.import_dmabuf() to import client buffers.");

    Ok(())
}
