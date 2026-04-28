//! Tests for the rendering pipeline.

use lamco_wgpu::renderer::WgpuFramebuffer;
use lamco_wgpu::{WgpuBridge, WgpuRenderer};
use smithay::backend::allocator::Fourcc;
use smithay::backend::renderer::{Color32F, ExportMem, Frame, Renderer, Texture};
use smithay::utils::{Buffer, Physical, Rectangle, Size, Transform};
use std::sync::Arc;

fn create_test_bridge() -> Arc<WgpuBridge> {
    Arc::new(WgpuBridge::new().expect("Failed to create WgpuBridge"))
}

#[test]
#[ignore] // Requires GPU
fn test_renderer_creation() {
    let bridge = create_test_bridge();
    let renderer = WgpuRenderer::new(bridge);
    println!("Created renderer: {:?}", renderer);
}

#[test]
#[ignore] // Requires GPU
fn test_framebuffer_creation() {
    let bridge = create_test_bridge();
    let fb = WgpuFramebuffer::new(bridge.device(), 1920, 1080);

    assert_eq!(fb.width(), 1920);
    assert_eq!(fb.height(), 1080);
    println!(
        "Created framebuffer: {}x{} format={:?}",
        fb.width(),
        fb.height(),
        fb.format()
    );
}

#[test]
#[ignore] // Requires GPU
fn test_render_frame() {
    let bridge = create_test_bridge();
    let mut renderer = WgpuRenderer::new(bridge.clone());
    let mut fb = WgpuFramebuffer::new(bridge.device(), 800, 600);

    // Begin a frame
    let frame = renderer
        .render(&mut fb, Size::from((800, 600)), Transform::Normal)
        .expect("Failed to begin frame");

    // Finish the frame
    let sync = frame.finish().expect("Failed to finish frame");
    println!("Frame completed with sync point: {:?}", sync);
}

#[test]
#[ignore] // Requires GPU
fn test_clear_frame() {
    let bridge = create_test_bridge();
    let mut renderer = WgpuRenderer::new(bridge.clone());
    let mut fb = WgpuFramebuffer::new(bridge.device(), 800, 600);

    // Begin a frame
    let mut frame = renderer
        .render(&mut fb, Size::from((800, 600)), Transform::Normal)
        .expect("Failed to begin frame");

    // Clear with a blue color
    let clear_color = Color32F::new(0.0, 0.0, 1.0, 1.0);
    frame
        .clear(clear_color, &[])
        .expect("Failed to clear frame");

    // Finish the frame
    let sync = frame.finish().expect("Failed to finish frame");
    println!("Cleared frame with blue, sync: {:?}", sync);
}

#[test]
#[ignore] // Requires GPU
fn test_draw_solid_rect() {
    let bridge = create_test_bridge();
    let mut renderer = WgpuRenderer::new(bridge.clone());
    let mut fb = WgpuFramebuffer::new(bridge.device(), 800, 600);

    // Begin a frame
    let mut frame = renderer
        .render(&mut fb, Size::from((800, 600)), Transform::Normal)
        .expect("Failed to begin frame");

    // Clear background
    let clear_color = Color32F::new(0.1, 0.1, 0.1, 1.0);
    frame.clear(clear_color, &[]).expect("Failed to clear");

    // Draw a red rectangle
    let red = Color32F::new(1.0, 0.0, 0.0, 1.0);
    let rect = Rectangle::<i32, Physical>::new((100, 100).into(), (200, 150).into());
    frame
        .draw_solid(rect, &[], red)
        .expect("Failed to draw solid rect");

    // Draw a semi-transparent green rectangle
    let green = Color32F::new(0.0, 1.0, 0.0, 0.5);
    let rect2 = Rectangle::<i32, Physical>::new((150, 150).into(), (200, 150).into());
    frame
        .draw_solid(rect2, &[], green)
        .expect("Failed to draw solid rect");

    // Finish the frame
    let sync = frame.finish().expect("Failed to finish frame");
    println!("Drew solid rectangles, sync: {:?}", sync);
}

#[test]
#[ignore] // Requires GPU
fn test_multiple_frames() {
    let bridge = create_test_bridge();
    let mut renderer = WgpuRenderer::new(bridge.clone());
    let mut fb = WgpuFramebuffer::new(bridge.device(), 640, 480);

    // Render multiple frames
    for i in 0..5 {
        let mut frame = renderer
            .render(&mut fb, Size::from((640, 480)), Transform::Normal)
            .expect("Failed to begin frame");

        // Vary the clear color across frames
        let t = i as f32 / 4.0;
        let color = Color32F::new(t, 0.5, 1.0 - t, 1.0);
        frame.clear(color, &[]).expect("Failed to clear");

        let _ = frame.finish().expect("Failed to finish frame");
    }
    println!("Rendered 5 frames successfully");
}

#[test]
#[ignore] // Requires GPU
fn test_screenshot_export_mem() {
    let bridge = create_test_bridge();
    let mut renderer = WgpuRenderer::new(bridge.clone());
    let mut fb = WgpuFramebuffer::new(bridge.device(), 64, 64);

    // Begin a frame and render something
    let mut frame = renderer
        .render(&mut fb, Size::from((64, 64)), Transform::Normal)
        .expect("Failed to begin frame");

    // Clear with red color (BGRA: B=0, G=0, R=255, A=255)
    let red = Color32F::new(1.0, 0.0, 0.0, 1.0);
    frame.clear(red, &[]).expect("Failed to clear");

    // Finish the frame
    let _ = frame.finish().expect("Failed to finish frame");

    // Wait for GPU to complete
    renderer
        .wait(&smithay::backend::renderer::sync::SyncPoint::signaled())
        .expect("Failed to wait");

    // Take a screenshot of the entire framebuffer
    let region = Rectangle::<i32, Buffer>::new((0, 0).into(), (64, 64).into());
    let mapping = renderer
        .copy_framebuffer(&fb, region, Fourcc::Argb8888)
        .expect("Failed to copy framebuffer");

    // Verify the mapping dimensions
    assert_eq!(Texture::width(&mapping), 64);
    assert_eq!(Texture::height(&mapping), 64);
    assert_eq!(Texture::format(&mapping), Some(Fourcc::Argb8888));

    // Get the pixel data
    let data = renderer
        .map_texture(&mapping)
        .expect("Failed to map texture");

    // Check that we got the expected amount of data (64 * 64 * 4 bytes)
    assert_eq!(data.len(), 64 * 64 * 4);

    // Check that pixels are red (BGRA format: B=0, G=0, R=255, A=255)
    // Sample the first pixel
    let b = data[0];
    let g = data[1];
    let r = data[2];
    let a = data[3];

    println!(
        "First pixel: R={}, G={}, B={}, A={} (expected R=255, G=0, B=0, A=255)",
        r, g, b, a
    );

    // The pixel should be red (allowing some tolerance for GPU rounding)
    assert!(r > 250, "Red channel should be ~255, got {}", r);
    assert!(g < 5, "Green channel should be ~0, got {}", g);
    assert!(b < 5, "Blue channel should be ~0, got {}", b);
    assert!(a > 250, "Alpha channel should be ~255, got {}", a);

    println!(
        "Screenshot test passed: {}x{} pixels, {} bytes",
        mapping.width(),
        mapping.height(),
        data.len()
    );
}

#[test]
#[ignore] // Requires GPU
fn test_screenshot_partial_region() {
    let bridge = create_test_bridge();
    let mut renderer = WgpuRenderer::new(bridge.clone());
    let mut fb = WgpuFramebuffer::new(bridge.device(), 100, 100);

    // Render a frame with a solid color
    let mut frame = renderer
        .render(&mut fb, Size::from((100, 100)), Transform::Normal)
        .expect("Failed to begin frame");

    let green = Color32F::new(0.0, 1.0, 0.0, 1.0);
    frame.clear(green, &[]).expect("Failed to clear");
    let _ = frame.finish().expect("Failed to finish frame");

    // Wait for GPU
    renderer
        .wait(&smithay::backend::renderer::sync::SyncPoint::signaled())
        .expect("Failed to wait");

    // Copy only a 20x20 region from the center
    let region = Rectangle::<i32, Buffer>::new((40, 40).into(), (20, 20).into());
    let mapping = renderer
        .copy_framebuffer(&fb, region, Fourcc::Argb8888)
        .expect("Failed to copy partial region");

    assert_eq!(Texture::width(&mapping), 20);
    assert_eq!(Texture::height(&mapping), 20);

    let data = renderer
        .map_texture(&mapping)
        .expect("Failed to map texture");
    assert_eq!(data.len(), 20 * 20 * 4);

    println!("Partial screenshot test passed: {}x{} region", 20, 20);
}
