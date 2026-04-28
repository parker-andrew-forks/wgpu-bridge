# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-XX

### Added

- `WgpuBridge`: Core bridge wrapping wgpu Device/Queue with Vulkan interop
- `WgpuRenderer`: Full implementation of Smithay's `Renderer` trait
- DMA-BUF import via raw Vulkan (`VK_KHR_external_memory_fd`)
- Multi-planar YUV format support (NV12, P010) with hardware YCbCr conversion
- Explicit sync support via `SyncManager` and `SyncBridge`
  - Binary semaphore export using wgpu's `add_signal_semaphore()` ([wgpu#6813])
  - CPU-side acquire point waiting (GPU-side blocked on [wgpu#8996])
- Output scanout (DMA-BUF export for display)
- Shared Vulkan context mode (`from_smithay_vulkan`)

### Notes

This is a foundation release demonstrating wgpu viability as a "guest renderer"
for Smithay-based Wayland compositors. See README for current limitations and
the architectural approach.

[wgpu#6813]: https://github.com/gfx-rs/wgpu/pull/6813
[wgpu#8996]: https://github.com/gfx-rs/wgpu/issues/8996
