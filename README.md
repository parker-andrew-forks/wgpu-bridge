# lamco-wgpu

[![CI](https://github.com/lamco-admin/lamco-wgpu/actions/workflows/ci.yml/badge.svg)](https://github.com/lamco-admin/lamco-wgpu/actions)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

wgpu integration for Smithay-based Wayland compositors.

## Overview

This crate enables [wgpu](https://wgpu.rs/) as a "guest renderer" on top of [Smithay](https://github.com/Smithay/smithay)'s Wayland compositor infrastructure. It provides:

- **WgpuBridge** - Core bridge wrapping wgpu with Vulkan interop for DMA-BUF import
- **WgpuRenderer** - Full implementation of Smithay's `Renderer` trait
- **Explicit sync** - Support for `linux-drm-syncobj-v1` protocol (required for NVIDIA)

## Requirements

- **Rust 1.92+** (required by wgpu 28)
- Vulkan 1.2+ driver
- Linux with DMA-BUF support

## Installation

This crate requires Smithay features (`backend_vulkan`, `drm_syncobj`) only available in git master,
not in any published crates.io version. Install via git:

```toml
[dependencies]
lamco-wgpu = { git = "https://github.com/lamco-admin/lamco-wgpu" }
```

Once Smithay publishes a compatible version, this crate will be available on crates.io.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SMITHAY HOST LAYER                           │
│                    (Owns Vulkan instance & GPU resources)           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │      lamco-wgpu (this crate)  │
                    └───────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                        WGPU GUEST LAYER                             │
│                (High-level rendering via shared context)            │
└─────────────────────────────────────────────────────────────────────┘
```

Smithay owns the Vulkan instance and handles low-level GPU resource management.
wgpu operates as a "guest" using the shared Vulkan context for high-level rendering.

## Quick Start

```rust
use lamco_wgpu::{WgpuBridge, WgpuRenderer};
use std::sync::Arc;

// Create bridge (standalone mode)
let bridge = Arc::new(WgpuBridge::new()?);

// Or share Smithay's Vulkan context (production)
let bridge = Arc::new(unsafe {
    WgpuBridge::from_smithay_vulkan(
        &smithay_instance,
        smithay_physical_device,
        &smithay_device,
        smithay_queue,
        queue_family_index,
    )?
});

// Create renderer implementing Smithay's Renderer trait
let mut renderer = WgpuRenderer::new(bridge.clone());
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `dmabuf` | ✓ | DMA-BUF import via raw Vulkan |
| `explicit-sync` | ✓ | Smithay explicit sync integration |

## Explicit Sync

For compositors implementing `linux-drm-syncobj-v1` (required for NVIDIA):

```rust
use lamco_wgpu::{SyncManager, SyncBridge};

let sync_manager = SyncManager::new(&bridge)?;
let sync_bridge = SyncBridge::new(&sync_manager);

// Before sampling client buffer
if let Some(acquire) = surface.acquire_point() {
    sync_bridge.wait_acquire_point(acquire, 5000)?;
}

// Signal release after rendering
let pending = sync_bridge.prepare_release_signal(&queue)?;
queue.submit([encoder.finish()]);
sync_bridge.complete_release_signal(pending, &release_point, &device)?;
```

### Semaphore Limitations

Release point signaling uses wgpu's `add_signal_semaphore()` API ([wgpu#6813]).
GPU-side acquire waiting is not yet possible because wgpu lacks `add_wait_semaphore()` ([wgpu#8996]).
Current implementation uses CPU-side waiting via `poll()` on the exported sync_file.

## wgpu Ecosystem Context

This crate exists because wgpu does not natively support the low-level Vulkan
interop required for Wayland compositors. Relevant upstream issues:

- [wgpu#2320] - Texture memory import API (open since 2021)
- [wgpu#6813] - `add_signal_semaphore()` for external semaphore signaling (merged)
- [wgpu#8996] - `add_wait_semaphore()` for external semaphore waiting (open)

The Smithay maintainers concluded that wgpu is "too abstract for internal
compositor work" but suitable as a guest renderer ([Smithay#431], [Smithay#928]).
This crate implements that pattern.

## Status

This is a foundation release. Working:

- DMA-BUF import (single-plane and multi-planar YUV)
- Smithay Renderer trait implementation
- Explicit sync release signaling
- Shared Vulkan context with Smithay

Not yet production-tested:

- Real compositor integration at scale
- All multi-planar format variants
- DMA-BUF export (scanout)

## About

Developed by [Lamco Development](https://lamco.ai) as part of the Wayland compositor ecosystem.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

## Links

- [Documentation](https://docs.rs/lamco-wgpu)
- [Repository](https://github.com/lamco-admin/lamco-wgpu)
- [Smithay](https://github.com/Smithay/smithay)
- [wgpu](https://wgpu.rs/)

[wgpu#2320]: https://github.com/gfx-rs/wgpu/issues/2320
[wgpu#6813]: https://github.com/gfx-rs/wgpu/pull/6813
[wgpu#8996]: https://github.com/gfx-rs/wgpu/issues/8996
[Smithay#431]: https://github.com/Smithay/smithay/discussions/431
[Smithay#928]: https://github.com/Smithay/smithay/issues/928
