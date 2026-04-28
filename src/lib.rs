#![cfg_attr(docsrs, feature(doc_cfg))]

//! # lamco-wgpu
//!
//! Bridge layer enabling wgpu as a "guest renderer" on top of Smithay's infrastructure.
//!
//! ## Architecture
//!
//! This crate implements the "guest renderer" pattern where:
//! - **Smithay** owns the Vulkan instance, handles dmabuf import/export, and manages sync primitives
//! - **wgpu** provides high-level, safe rendering APIs using the shared Vulkan context
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                        SMITHAY HOST LAYER                           │
//! │                    (Owns all GPU resources)                         │
//! └─────────────────────────────────────────────────────────────────────┘
//!                                     │
//!                     ┌───────────────┼───────────────┐
//!                     │    BRIDGE LAYER (this crate)  │
//!                     └───────────────────────────────┘
//!                                     │
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                        WGPU GUEST LAYER                             │
//! │                (High-level rendering via shared context)            │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Status
//!
//! Key features:
//! - [x] Create wgpu Device (standalone, Vulkan backend)
//! - [x] Import dmabuf textures into wgpu
//! - [x] Implement Smithay's Renderer trait
//! - [x] Full render pipeline with WGSL shaders
//! - [x] Sync FD export via VK_KHR_external_semaphore_fd
//! - [x] Output scanout (dmabuf export for display)
//! - [x] Shared Vulkan context (from_smithay_vulkan) - requires Smithay to expose handles

pub mod bridge;
pub mod error;
pub mod modifiers;
pub mod multiplanar;
pub mod pipeline;
pub mod renderer;
pub mod scanout;
pub mod sync;
pub mod texture;

#[cfg(feature = "explicit-sync")]
#[cfg_attr(docsrs, doc(cfg(feature = "explicit-sync")))]
pub mod smithay_sync;

pub use bridge::{SupportedFormat, WgpuBridge};
pub use error::{BridgeError, ImportError};
pub use modifiers::{
    drm_mod, is_modifier_supported, query_format_modifiers, ExplicitModifierCreateInfo,
    ModifierCapabilities, ModifierPlaneLayout, ModifierProperties,
};
pub use multiplanar::{
    calculate_plane_layouts, is_multiplanar_format, multiplanar_format,
    multiplanar_format_with_colorspace, MultiPlanarCapabilities, MultiPlanarFormat,
    MultiPlanarTexture, PlaneLayout, YcbcrColorspace, YcbcrModel, YcbcrRange,
};
pub use renderer::{WgpuRenderer, WgpuTextureMapping};
pub use scanout::{ExportedDmabuf, RenderTarget, RenderTargetConfig, ScanoutCapabilities};
pub use sync::{ExportSemaphore, ImportedSemaphore, SyncCapabilities, SyncManager, SyncPoint};
pub use texture::WgpuTexture;

#[cfg(feature = "dmabuf")]
#[cfg_attr(docsrs, doc(cfg(feature = "dmabuf")))]
pub use bridge::ImportedDmabuf;

#[cfg(feature = "explicit-sync")]
#[cfg_attr(docsrs, doc(cfg(feature = "explicit-sync")))]
pub use smithay_sync::{PendingReleaseSignal, SyncBridge, SyncProcessor, SyncSurface};

/// Re-export key Smithay types for convenience
pub mod smithay_reexports {
    pub use smithay::backend::allocator::dmabuf::Dmabuf;
    pub use smithay::backend::allocator::{Format, Fourcc};
    pub use smithay::backend::renderer::{
        sync::SyncPoint, ContextId, ExportMem, Frame, ImportDma, ImportMem, Renderer, Texture,
        TextureFilter,
    };
    pub use smithay::utils::{Buffer, Physical, Rectangle, Size, Transform};
}

/// Re-export key wgpu types for convenience
pub mod wgpu_reexports {
    pub use wgpu::{
        Adapter, CommandEncoder, Device, Instance, Queue, RenderPass, RenderPipeline, Texture,
        TextureDescriptor, TextureFormat, TextureUsages,
    };
}
