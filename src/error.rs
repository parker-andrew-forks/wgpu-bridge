//! Error types for the lamco-wgpu crate.

use thiserror::Error;

/// Errors that can occur when creating or using the bridge.
#[derive(Error, Debug)]
pub enum BridgeError {
    /// Failed to create wgpu instance from Smithay's Vulkan instance
    #[error("failed to create wgpu instance: {0}")]
    InstanceCreation(String),

    /// Failed to create wgpu adapter
    #[error("failed to create wgpu adapter: {0}")]
    AdapterCreation(String),

    /// Failed to create wgpu device
    #[error("failed to create wgpu device: {0}")]
    DeviceCreation(#[from] wgpu::RequestDeviceError),

    /// The Vulkan instance doesn't support required extensions
    #[error("missing required Vulkan extension: {0}")]
    MissingExtension(String),

    /// Failed to access raw Vulkan handles via as_hal
    #[error("failed to access raw Vulkan handles: {0}")]
    HalAccess(String),

    /// Synchronization error
    #[error("synchronization error: {0}")]
    Sync(String),

    /// Rendering error
    #[error("rendering error: {0}")]
    Render(String),
}

/// Errors that can occur when importing textures.
#[derive(Error, Debug)]
pub enum ImportError {
    /// The dmabuf format is not supported
    #[error("unsupported dmabuf format: {0:?}")]
    UnsupportedFormat(drm_fourcc::DrmFourcc),

    /// Multi-planar format requires special import path
    #[error("multi-planar format {fourcc:?} requires special handling: {hint}")]
    MultiPlanarFormat {
        fourcc: drm_fourcc::DrmFourcc,
        hint: String,
    },

    /// The dmabuf modifier is not supported
    #[error("unsupported dmabuf modifier: {0:#x}")]
    UnsupportedModifier(u64),

    /// Failed to import the dmabuf file descriptor
    #[error("failed to import dmabuf fd: {0}")]
    FdImport(String),

    /// Failed to create Vulkan image from dmabuf
    #[error("failed to create Vulkan image: {0}")]
    ImageCreation(String),

    /// Failed to allocate memory for the imported texture
    #[error("memory allocation failed: {0}")]
    MemoryAllocation(String),

    /// The dmabuf has an invalid plane configuration
    #[error("invalid dmabuf plane configuration: {0}")]
    InvalidPlanes(String),

    /// Texture dimensions are invalid or unsupported
    #[error("invalid texture dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },

    /// Bridge is not initialized or has been dropped
    #[error("bridge not available")]
    BridgeUnavailable,

    /// Failed to access SHM buffer
    #[error("shm buffer access failed: {0}")]
    ShmAccess(String),

    /// Unsupported SHM format
    #[error("unsupported shm format")]
    UnsupportedShmFormat,
}

/// Errors that can occur during rendering.
#[derive(Error, Debug)]
pub enum RenderError {
    /// Frame has already been finished
    #[error("frame has already been finished")]
    FrameFinished,

    /// Failed to begin a render pass
    #[error("failed to begin render pass: {0}")]
    BeginRenderPass(String),

    /// Failed to submit commands
    #[error("failed to submit commands: {0}")]
    Submit(String),

    /// Texture is from a different renderer context
    #[error("texture context mismatch: expected {expected:?}, got {actual:?}")]
    ContextMismatch { expected: String, actual: String },

    /// Frame buffer is invalid or incompatible
    #[error("invalid framebuffer: {0}")]
    InvalidFramebuffer(String),

    /// The transformation is not supported
    #[error("unsupported transformation: {0:?}")]
    UnsupportedTransform(smithay::utils::Transform),

    /// Import error during rendering
    #[error("import error: {0}")]
    Import(#[from] ImportError),

    /// Bridge error during rendering
    #[error("bridge error: {0}")]
    Bridge(#[from] BridgeError),
}
