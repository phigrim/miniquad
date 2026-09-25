//! Native wgpu renderer.
//!
//! The backend records miniquad draw calls and emits one wgpu render pass per
//! miniquad pass. Uniforms are packed into a single dynamic ring buffer and
//! texture bind groups are cached, keeping the hot path allocation-free after
//! warm-up.

use super::*;
use ::wgpu::util::DeviceExt;
use smallvec::SmallVec;
use std::{collections::HashMap, num::NonZeroU64, sync::Arc};

const UNIFORM_ALIGNMENT: usize = 256;
const INITIAL_UNIFORM_CAPACITY: usize = 256 * 1024;

#[inline]
const fn aligned_buffer_size(size: usize) -> usize {
    (size + (::wgpu::COPY_BUFFER_ALIGNMENT as usize - 1))
        & !(::wgpu::COPY_BUFFER_ALIGNMENT as usize - 1)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PipelineKey {
    format: ::wgpu::TextureFormat,
    depth: Option<::wgpu::TextureFormat>,
    samples: u32,
}

struct Shader {
    module: ::wgpu::ShaderModule,
    meta: ShaderMeta,
    bind_group_layout: ::wgpu::BindGroupLayout,
    pipeline_layout: ::wgpu::PipelineLayout,
    uniform_size: u64,
    raw_uniform_size: usize,
    uniform_copies: Vec<(usize, usize, usize)>,
}

struct PipelineResource {
    shader: ShaderId,
    buffers: Vec<BufferLayout>,
    attributes: Vec<VertexAttribute>,
    params: PipelineParams,
    variants: HashMap<PipelineKey, ::wgpu::RenderPipeline>,
}

struct Buffer {
    raw: ::wgpu::Buffer,
    size: usize,
    element_size: usize,
}

struct Texture {
    raw: ::wgpu::Texture,
    view: ::wgpu::TextureView,
    params: TextureParams,
    format: ::wgpu::TextureFormat,
}

struct Pass {
    colors: Vec<TextureId>,
    resolves: Option<Vec<TextureId>>,
    depth: Option<TextureId>,
}

#[derive(Clone)]
struct DrawCall {
    pipeline: Pipeline,
    vertex_buffers: SmallVec<[BufferId; 4]>,
    index_buffer: BufferId,
    images: SmallVec<[TextureId; 4]>,
    uniform_offset: u32,
    viewport: Option<(f32, f32, f32, f32)>,
    scissor: Option<(u32, u32, u32, u32)>,
    base_element: u32,
    base_vertex: i32,
    elements: u32,
    instances: u32,
}

struct PassRecording {
    target: Option<RenderPass>,
    action: PassAction,
    draws: Vec<DrawCall>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct BindGroupKey {
    shader: ShaderId,
    images: SmallVec<[TextureId; 4]>,
    uniform_generation: u64,
    sampler_generation: u64,
}

#[derive(Clone)]
enum SurfaceTargetOwner {
    #[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
    Desktop(Arc<winit::window::Window>),
    #[cfg(target_os = "android")]
    Android(Arc<crate::native::android::AndroidNativeWindow>),
}

impl SurfaceTargetOwner {
    fn create_surface(
        &self,
        instance: &::wgpu::Instance,
    ) -> Result<::wgpu::Surface<'static>, ::wgpu::CreateSurfaceError> {
        match self {
            #[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
            Self::Desktop(window) => instance.create_surface(window.clone()),
            #[cfg(target_os = "android")]
            Self::Android(window) => instance.create_surface(
                ::wgpu::SurfaceTarget::from_window_without_display(window.clone()),
            ),
        }
    }
}

#[cfg(target_os = "android")]
#[derive(Debug)]
struct AndroidDisplayHandleOwner;

#[cfg(target_os = "android")]
impl ::wgpu::rwh::HasDisplayHandle for AndroidDisplayHandleOwner {
    fn display_handle(&self) -> Result<::wgpu::rwh::DisplayHandle<'_>, ::wgpu::rwh::HandleError> {
        Ok(::wgpu::rwh::DisplayHandle::android())
    }
}

#[derive(Clone)]
struct SurfaceSnapshot {
    generation: u64,
    target: Option<SurfaceTargetOwner>,
    width: u32,
    height: u32,
    ready: bool,
}

enum SurfaceSource {
    #[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
    Desktop(Arc<winit::window::Window>),
    #[cfg(target_os = "android")]
    Android(Arc<crate::native::android::AndroidSurfaceSource>),
}

impl SurfaceSource {
    fn snapshot(&self) -> SurfaceSnapshot {
        match self {
            #[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
            Self::Desktop(window) => {
                let size = window.inner_size();
                SurfaceSnapshot {
                    generation: 1,
                    target: Some(SurfaceTargetOwner::Desktop(window.clone())),
                    width: size.width.max(1),
                    height: size.height.max(1),
                    ready: true,
                }
            }
            #[cfg(target_os = "android")]
            Self::Android(source) => {
                let snapshot = source.snapshot();
                SurfaceSnapshot {
                    generation: snapshot.generation,
                    target: snapshot.target.map(SurfaceTargetOwner::Android),
                    width: snapshot.width,
                    height: snapshot.height,
                    ready: snapshot.ready,
                }
            }
        }
    }
}

#[cfg(target_os = "android")]
impl ::wgpu::rwh::HasWindowHandle for crate::native::android::AndroidNativeWindow {
    fn window_handle(&self) -> Result<::wgpu::rwh::WindowHandle<'_>, ::wgpu::rwh::HandleError> {
        let native_window = std::ptr::NonNull::new(self.as_ptr().cast())
            .expect("AndroidNativeWindow always owns a non-null native window");
        let raw = ::wgpu::rwh::AndroidNdkWindowHandle::new(native_window).into();
        // SAFETY: `self` owns an ANativeWindow reference and the returned
        // borrowed handle cannot outlive this owner.
        Ok(unsafe { ::wgpu::rwh::WindowHandle::borrow_raw(raw) })
    }
}

#[allow(dead_code)] // Every target constructs only its own platform variant.
#[derive(Clone, Copy)]
enum BackendPlatform {
    Windows,
    Linux,
    Macos,
    Android,
}

#[cfg(target_os = "windows")]
fn backend_platform() -> BackendPlatform {
    BackendPlatform::Windows
}
#[cfg(target_os = "linux")]
fn backend_platform() -> BackendPlatform {
    BackendPlatform::Linux
}
#[cfg(target_os = "macos")]
fn backend_platform() -> BackendPlatform {
    BackendPlatform::Macos
}
#[cfg(target_os = "android")]
fn backend_platform() -> BackendPlatform {
    BackendPlatform::Android
}

fn backend_candidates(
    requested: crate::conf::WgpuBackend,
    platform: BackendPlatform,
) -> Vec<(::wgpu::Backends, &'static str)> {
    use crate::conf::WgpuBackend;

    match requested {
        WgpuBackend::Auto => match platform {
            BackendPlatform::Windows => vec![
                (::wgpu::Backends::VULKAN, "Vulkan"),
                (::wgpu::Backends::DX12, "Direct3D 12"),
            ],
            BackendPlatform::Linux => vec![(::wgpu::Backends::VULKAN, "Vulkan")],
            BackendPlatform::Macos => vec![(::wgpu::Backends::METAL, "Metal")],
            BackendPlatform::Android => vec![
                (::wgpu::Backends::VULKAN, "Vulkan"),
                (::wgpu::Backends::GL, "GLES"),
            ],
        },
        WgpuBackend::Vulkan => vec![(::wgpu::Backends::VULKAN, "Vulkan")],
        WgpuBackend::Gles => vec![(::wgpu::Backends::GL, "GLES")],
        WgpuBackend::Dx12 => vec![(::wgpu::Backends::DX12, "Direct3D 12")],
        WgpuBackend::Metal => vec![(::wgpu::Backends::METAL, "Metal")],
    }
}

pub struct WgpuContext {
    _instance: ::wgpu::Instance,
    adapter: ::wgpu::Adapter,
    surface_source: SurfaceSource,
    surface_generation: u64,
    surface: Option<::wgpu::Surface<'static>>,
    device: ::wgpu::Device,
    queue: ::wgpu::Queue,
    surface_config: ::wgpu::SurfaceConfiguration,
    surface_frame: Option<::wgpu::SurfaceTexture>,
    surface_view: Option<::wgpu::TextureView>,
    default_msaa: Option<Texture>,
    default_depth: Option<Texture>,
    sample_count: u32,
    swap_interval: Option<i32>,

    shaders: Vec<Option<Shader>>,
    pipelines: Vec<Option<PipelineResource>>,
    buffers: Vec<Option<Buffer>>,
    textures: Vec<Option<Texture>>,
    passes: Vec<Option<Pass>>,

    current_pass: Option<PassRecording>,
    pending_pass: Option<PassRecording>,
    current_pipeline: Option<Pipeline>,
    current_vertices: SmallVec<[BufferId; 4]>,
    current_index: Option<BufferId>,
    current_images: SmallVec<[TextureId; 4]>,
    current_uniforms: Vec<u8>,
    viewport: Option<(f32, f32, f32, f32)>,
    scissor: Option<(u32, u32, u32, u32)>,
    frame_encoder: Option<::wgpu::CommandEncoder>,
    staging_belt: ::wgpu::util::StagingBelt,
    poll_frame: u8,
    // WGPU textures start at zero, which is the nearest possible depth value.
    // The swapchain depth attachment is owned by this backend (unlike an
    // explicitly supplied render-pass depth texture), so initialize it once
    // before the first depth-tested draw of every presented frame.
    default_depth_used_this_frame: bool,

    uniform_cpu: Vec<u8>,
    uniform_buffer: ::wgpu::Buffer,
    uniform_capacity: usize,
    uniform_generation: u64,
    sampler_generation: u64,
    bind_groups: HashMap<BindGroupKey, Arc<::wgpu::BindGroup>>,
    pending_texture_deletes: Vec<TextureId>,
    // `Queue::write_buffer` requires both offset and copy size to be aligned to
    // COPY_BUFFER_ALIGNMENT (4). Index buffers commonly contain an odd number
    // of u16 values, so retain one scratch allocation for the padded tail.
    buffer_upload_scratch: Vec<u8>,
}

struct WgpuInitialization {
    instance: ::wgpu::Instance,
    surface: ::wgpu::Surface<'static>,
    adapter: ::wgpu::Adapter,
    device: ::wgpu::Device,
    queue: ::wgpu::Queue,
    surface_config: ::wgpu::SurfaceConfiguration,
}

fn wgpu_adapter_for_target(
    target: &SurfaceTargetOwner,
    requested: crate::conf::WgpuBackend,
    size: (u32, u32),
    swap_interval: Option<i32>,
) -> Result<WgpuInitialization, String> {
    let candidates = backend_candidates(requested, backend_platform());
    let mut failures = Vec::with_capacity(candidates.len());
    for (backends, name) in candidates {
        #[cfg(target_os = "android")]
        let mut descriptor = ::wgpu::InstanceDescriptor::new_with_display_handle(Box::new(
            AndroidDisplayHandleOwner,
        ));
        #[cfg(not(target_os = "android"))]
        let mut descriptor = ::wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = backends;
        let instance = ::wgpu::Instance::new(descriptor);
        let surface = match target.create_surface(&instance) {
            Ok(surface) => surface,
            Err(error) => {
                failures.push(format!("{name}: surface creation failed: {error}"));
                continue;
            }
        };
        let adapter =
            match pollster::block_on(instance.request_adapter(&::wgpu::RequestAdapterOptions {
                power_preference: ::wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })) {
                Ok(adapter) => adapter,
                Err(error) => {
                    failures.push(format!("{name}: no compatible adapter: {error}"));
                    continue;
                }
            };
        let surface_config = match surface_configuration(&surface, &adapter, size, swap_interval) {
            Ok(config) => config,
            Err(error) => {
                failures.push(format!("{name}: surface is unusable: {error}"));
                continue;
            }
        };
        let limits = ::wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
        let (device, queue) =
            match pollster::block_on(adapter.request_device(&::wgpu::DeviceDescriptor {
                label: Some("miniquad wgpu device"),
                required_features: ::wgpu::Features::empty(),
                required_limits: limits,
                ..Default::default()
            })) {
                Ok(device_and_queue) => device_and_queue,
                Err(error) => {
                    failures.push(format!("{name}: device creation failed: {error}"));
                    continue;
                }
            };
        let error_scope = device.push_error_scope(::wgpu::ErrorFilter::Validation);
        surface.configure(&device, &surface_config);
        let _ = device.poll(::wgpu::PollType::wait_indefinitely());
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            failures.push(format!("{name}: surface configuration failed: {error}"));
            continue;
        }
        return Ok(WgpuInitialization {
            instance,
            surface,
            adapter,
            device,
            queue,
            surface_config,
        });
    }
    Err(format!(
        "failed to initialize WGPU ({:?}); attempted backends: {}",
        requested,
        failures.join("; ")
    ))
}

fn surface_configuration(
    surface: &::wgpu::Surface<'_>,
    adapter: &::wgpu::Adapter,
    size: (u32, u32),
    swap_interval: Option<i32>,
) -> Result<::wgpu::SurfaceConfiguration, String> {
    let caps = surface.get_capabilities(adapter);
    if !caps
        .usages
        .contains(::wgpu::TextureUsages::RENDER_ATTACHMENT)
    {
        return Err("surface does not support render-attachment usage".to_owned());
    }
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|format| !format.is_srgb())
        .or_else(|| caps.formats.first().copied())
        .ok_or_else(|| "surface reports no supported texture formats".to_owned())?;
    let present_mode = if swap_interval == Some(0) {
        if caps.present_modes.contains(&::wgpu::PresentMode::Immediate) {
            ::wgpu::PresentMode::Immediate
        } else if caps.present_modes.contains(&::wgpu::PresentMode::Mailbox) {
            ::wgpu::PresentMode::Mailbox
        } else {
            *caps
                .present_modes
                .first()
                .ok_or_else(|| "surface reports no supported present modes".to_owned())?
        }
    } else if caps.present_modes.contains(&::wgpu::PresentMode::Fifo) {
        ::wgpu::PresentMode::Fifo
    } else {
        *caps
            .present_modes
            .first()
            .ok_or_else(|| "surface reports no supported present modes".to_owned())?
    };
    let alpha_mode = *caps
        .alpha_modes
        .first()
        .ok_or_else(|| "surface reports no supported alpha modes".to_owned())?;

    Ok(::wgpu::SurfaceConfiguration {
        usage: ::wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.0.max(1),
        height: size.1.max(1),
        present_mode,
        alpha_mode,
        view_formats: vec![],
        desired_maximum_frame_latency: if swap_interval == Some(0) { 3 } else { 2 },
    })
}

impl WgpuContext {
    pub fn new() -> Self {
        let (surface_source, requested_backend, swap_interval) = {
            let display = crate::native_display().lock().unwrap();
            #[cfg(target_os = "android")]
            let source = SurfaceSource::Android(display.android_surface_source.clone());
            #[cfg(not(target_os = "android"))]
            let source = SurfaceSource::Desktop(crate::native::winit::window());
            (source, display.wgpu_backend, display.swap_interval)
        };
        let initial_snapshot = surface_source.snapshot();
        assert!(
            initial_snapshot.ready,
            "WGPU backend initialized before an Android Surface became ready"
        );
        let target = initial_snapshot
            .target
            .as_ref()
            .expect("a ready surface snapshot must have a window owner");
        let WgpuInitialization {
            instance,
            surface,
            adapter,
            device,
            queue,
            surface_config,
        } = wgpu_adapter_for_target(
            target,
            requested_backend,
            (initial_snapshot.width, initial_snapshot.height),
            swap_interval,
        )
        .unwrap_or_else(|error| panic!("{}", error));
        let uniform_buffer = device.create_buffer(&::wgpu::BufferDescriptor {
            label: Some("miniquad uniform ring"),
            size: INITIAL_UNIFORM_CAPACITY as u64,
            usage: ::wgpu::BufferUsages::UNIFORM | ::wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sample_count = crate::native_display().lock().unwrap().sample_count;
        let staging_belt = ::wgpu::util::StagingBelt::new(device.clone(), 1024 * 1024);
        let mut result = Self {
            _instance: instance,
            adapter,
            surface_source,
            surface_generation: initial_snapshot.generation,
            surface: Some(surface),
            device,
            queue,
            surface_config,
            surface_frame: None,
            surface_view: None,
            default_msaa: None,
            default_depth: None,
            sample_count,
            swap_interval,
            shaders: vec![],
            pipelines: vec![],
            buffers: vec![],
            textures: vec![],
            passes: vec![],
            current_pass: None,
            pending_pass: None,
            current_pipeline: None,
            current_vertices: SmallVec::new(),
            current_index: None,
            current_images: SmallVec::new(),
            current_uniforms: vec![],
            viewport: None,
            scissor: None,
            frame_encoder: None,
            staging_belt,
            poll_frame: 0,
            default_depth_used_this_frame: false,
            uniform_cpu: Vec::with_capacity(INITIAL_UNIFORM_CAPACITY),
            uniform_buffer,
            uniform_capacity: INITIAL_UNIFORM_CAPACITY,
            uniform_generation: 0,
            sampler_generation: 0,
            bind_groups: HashMap::new(),
            pending_texture_deletes: Vec::new(),
            buffer_upload_scratch: Vec::new(),
        };
        result.rebuild_default_attachments();
        result
    }

    fn insert<T>(slots: &mut Vec<Option<T>>, value: T) -> usize {
        if let Some(index) = slots.iter().position(Option::is_none) {
            slots[index] = Some(value);
            index
        } else {
            slots.push(Some(value));
            slots.len() - 1
        }
    }

    fn texture_format(format: TextureFormat) -> ::wgpu::TextureFormat {
        match format {
            TextureFormat::RGB8 | TextureFormat::RGBA8 => ::wgpu::TextureFormat::Rgba8Unorm,
            TextureFormat::RGBA16F => ::wgpu::TextureFormat::Rgba16Float,
            TextureFormat::Alpha => ::wgpu::TextureFormat::R8Unorm,
            TextureFormat::Depth => ::wgpu::TextureFormat::Depth24PlusStencil8,
            TextureFormat::Depth32 => ::wgpu::TextureFormat::Depth32Float,
        }
    }

    fn create_texture_resource(&self, params: TextureParams, label: &str) -> Texture {
        let format = Self::texture_format(params.format);
        let layers = if params.kind == TextureKind::CubeMap {
            6
        } else {
            1
        };
        let mip_levels = if params.allocate_mipmaps {
            32 - params.width.max(params.height).leading_zeros()
        } else {
            1
        };
        let mut usage = ::wgpu::TextureUsages::COPY_DST
            | ::wgpu::TextureUsages::COPY_SRC
            | ::wgpu::TextureUsages::RENDER_ATTACHMENT;
        if matches!(params.format, TextureFormat::Depth | TextureFormat::Depth32)
            || params.sample_count > 1
            || params.width > 0
        {
            usage |= ::wgpu::TextureUsages::TEXTURE_BINDING;
        }
        if params.sample_count > 1
            || matches!(params.format, TextureFormat::Depth | TextureFormat::Depth32)
        {
            usage |= ::wgpu::TextureUsages::RENDER_ATTACHMENT;
        }
        let raw = self.device.create_texture(&::wgpu::TextureDescriptor {
            label: Some(label),
            size: ::wgpu::Extent3d {
                width: params.width.max(1),
                height: params.height.max(1),
                depth_or_array_layers: layers,
            },
            mip_level_count: mip_levels,
            sample_count: params.sample_count.max(1) as u32,
            dimension: ::wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        let view = raw.create_view(&::wgpu::TextureViewDescriptor {
            dimension: Some(if layers == 6 {
                ::wgpu::TextureViewDimension::Cube
            } else {
                ::wgpu::TextureViewDimension::D2
            }),
            ..Default::default()
        });
        Texture {
            raw,
            view,
            params,
            format,
        }
    }

    fn rebuild_default_attachments(&mut self) {
        let color_params = TextureParams {
            width: self.surface_config.width,
            height: self.surface_config.height,
            format: TextureFormat::RGBA8,
            sample_count: self.sample_count as i32,
            ..Default::default()
        };
        self.default_msaa = (self.sample_count > 1).then(|| {
            let raw = self.device.create_texture(&::wgpu::TextureDescriptor {
                label: Some("miniquad default msaa"),
                size: ::wgpu::Extent3d {
                    width: self.surface_config.width,
                    height: self.surface_config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: self.sample_count,
                dimension: ::wgpu::TextureDimension::D2,
                format: self.surface_config.format,
                usage: ::wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = raw.create_view(&Default::default());
            Texture {
                raw,
                view,
                params: color_params,
                format: self.surface_config.format,
            }
        });
        let depth_params = TextureParams {
            width: self.surface_config.width,
            height: self.surface_config.height,
            format: TextureFormat::Depth,
            sample_count: self.sample_count as i32,
            ..Default::default()
        };
        self.default_depth =
            Some(self.create_texture_resource(depth_params, "miniquad default depth"));
    }

    fn drop_surface(&mut self, generation: u64) {
        // SurfaceTexture must go before Surface; Surface owns the native window
        // handle source and releases its ANativeWindow reference on drop.
        self.surface_view = None;
        self.surface_frame = None;
        self.surface = None;
        self.surface_generation = generation;
    }

    fn sync_surface_source(&mut self) -> bool {
        let snapshot = self.surface_source.snapshot();
        if !snapshot.ready || snapshot.target.is_none() {
            if self.surface.is_some() || self.surface_generation != snapshot.generation {
                self.drop_surface(snapshot.generation);
            }
            return false;
        }

        if self.surface.is_none() || self.surface_generation != snapshot.generation {
            self.drop_surface(snapshot.generation);
            let target = snapshot.target.as_ref().unwrap();
            let surface = target
                .create_surface(&self._instance)
                .unwrap_or_else(|error| {
                    panic!("failed to create replacement WGPU surface: {}", error)
                });
            let config = surface_configuration(
                &surface,
                &self.adapter,
                (snapshot.width, snapshot.height),
                self.swap_interval,
            )
            .unwrap_or_else(|error| panic!("replacement WGPU surface is incompatible: {}", error));
            surface.configure(&self.device, &config);
            self.surface = Some(surface);
            self.surface_config = config;
            self.surface_generation = snapshot.generation;
            self.rebuild_default_attachments();
        } else if (snapshot.width, snapshot.height)
            != (self.surface_config.width, self.surface_config.height)
        {
            self.surface_frame = None;
            self.surface_view = None;
            let surface = self.surface.as_ref().unwrap();
            let config = surface_configuration(
                surface,
                &self.adapter,
                (snapshot.width, snapshot.height),
                self.swap_interval,
            )
            .unwrap_or_else(|error| panic!("resized WGPU surface is incompatible: {}", error));
            surface.configure(&self.device, &config);
            self.surface_config = config;
            self.rebuild_default_attachments();
        }
        true
    }

    fn ensure_surface_frame(&mut self) -> bool {
        if !self.sync_surface_source() {
            return false;
        }
        if self.surface_frame.is_some() {
            return true;
        }
        let status = self.surface.as_ref().unwrap().get_current_texture();
        let frame = match status {
            ::wgpu::CurrentSurfaceTexture::Success(frame)
            | ::wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            ::wgpu::CurrentSurfaceTexture::Lost => {
                // A lost native surface must be recreated, not merely
                // reconfigured. The retained source still owns the current
                // window, so the next frame will attach it again.
                self.drop_surface(self.surface_generation);
                return false;
            }
            ::wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface
                    .as_ref()
                    .unwrap()
                    .configure(&self.device, &self.surface_config);
                match self.surface.as_ref().unwrap().get_current_texture() {
                    ::wgpu::CurrentSurfaceTexture::Success(frame)
                    | ::wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
                    ::wgpu::CurrentSurfaceTexture::Timeout
                    | ::wgpu::CurrentSurfaceTexture::Occluded => return false,
                    status => panic!("failed to reacquire WGPU surface: {:?}", status),
                }
            }
            ::wgpu::CurrentSurfaceTexture::Timeout | ::wgpu::CurrentSurfaceTexture::Occluded => {
                return false
            }
            status => panic!("failed to acquire WGPU surface: {:?}", status),
        };
        self.surface_view = Some(frame.texture.create_view(&Default::default()));
        self.surface_frame = Some(frame);
        true
    }

    fn ensure_uniform_capacity(&mut self) {
        if self.uniform_cpu.len() <= self.uniform_capacity {
            return;
        }
        // Commands already encoded this frame may still reference the old
        // ring. Populate it before replacing the handle.
        if !self.uniform_cpu.is_empty() {
            self.queue.write_buffer(
                &self.uniform_buffer,
                0,
                &self.uniform_cpu[..self.uniform_capacity],
            );
        }
        self.uniform_capacity = self.uniform_cpu.len().next_power_of_two();
        self.uniform_buffer = self.device.create_buffer(&::wgpu::BufferDescriptor {
            label: Some("miniquad uniform ring"),
            size: self.uniform_capacity as u64,
            usage: ::wgpu::BufferUsages::UNIFORM | ::wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.uniform_generation += 1;
        self.bind_groups.clear();
    }

    fn sampler(&self, texture: &Texture) -> ::wgpu::Sampler {
        let address = |wrap| match wrap {
            TextureWrap::Repeat => ::wgpu::AddressMode::Repeat,
            TextureWrap::Mirror => ::wgpu::AddressMode::MirrorRepeat,
            TextureWrap::Clamp => ::wgpu::AddressMode::ClampToEdge,
        };
        self.device.create_sampler(&::wgpu::SamplerDescriptor {
            label: Some("miniquad texture sampler"),
            address_mode_u: address(texture.params.wrap),
            address_mode_v: address(texture.params.wrap),
            address_mode_w: address(texture.params.wrap),
            mag_filter: if texture.params.mag_filter == FilterMode::Linear {
                ::wgpu::FilterMode::Linear
            } else {
                ::wgpu::FilterMode::Nearest
            },
            min_filter: if texture.params.min_filter == FilterMode::Linear {
                ::wgpu::FilterMode::Linear
            } else {
                ::wgpu::FilterMode::Nearest
            },
            mipmap_filter: if texture.params.mipmap_filter == MipmapFilterMode::Linear {
                ::wgpu::MipmapFilterMode::Linear
            } else {
                ::wgpu::MipmapFilterMode::Nearest
            },
            ..Default::default()
        })
    }

    fn bind_group(&mut self, shader_id: ShaderId, images: &[TextureId]) -> Arc<::wgpu::BindGroup> {
        let key = BindGroupKey {
            shader: shader_id,
            images: images.iter().copied().collect(),
            uniform_generation: self.uniform_generation,
            sampler_generation: self.sampler_generation,
        };
        if let Some(group) = self.bind_groups.get(&key) {
            return group.clone();
        }
        let shader = self.shaders[shader_id.0].as_ref().unwrap();
        assert_eq!(
            shader.meta.images.len(),
            images.len(),
            "shader image count does not match bindings"
        );
        let mut samplers = Vec::with_capacity(images.len());
        for image in images {
            samplers.push(
                self.sampler(
                    self.textures[match image.0 {
                        TextureIdInner::Managed(id) => id,
                        _ => panic!("raw textures unsupported by wgpu"),
                    }]
                    .as_ref()
                    .unwrap(),
                ),
            );
        }
        let mut entries = Vec::with_capacity(images.len() * 2 + 1);
        if shader.uniform_size != 0 {
            entries.push(::wgpu::BindGroupEntry {
                binding: 0,
                resource: ::wgpu::BindingResource::Buffer(::wgpu::BufferBinding {
                    buffer: &self.uniform_buffer,
                    offset: 0,
                    size: NonZeroU64::new(shader.uniform_size),
                }),
            });
        }
        for (index, image) in images.iter().enumerate() {
            let texture = self.textures[match image.0 {
                TextureIdInner::Managed(id) => id,
                _ => unreachable!(),
            }]
            .as_ref()
            .unwrap();
            entries.push(::wgpu::BindGroupEntry {
                binding: 1 + index as u32 * 2,
                resource: ::wgpu::BindingResource::TextureView(&texture.view),
            });
            entries.push(::wgpu::BindGroupEntry {
                binding: 2 + index as u32 * 2,
                resource: ::wgpu::BindingResource::Sampler(&samplers[index]),
            });
        }
        let group = Arc::new(self.device.create_bind_group(&::wgpu::BindGroupDescriptor {
            label: Some("miniquad bind group"),
            layout: &shader.bind_group_layout,
            entries: &entries,
        }));
        self.bind_groups.insert(key, group.clone());
        group
    }

    fn pipeline_key(&self, target: Option<RenderPass>, needs_depth: bool) -> PipelineKey {
        if let Some(pass) = target {
            let pass = self.passes[pass.0].as_ref().unwrap();
            let color = self.texture(pass.colors[0]);
            PipelineKey {
                format: color.format,
                depth: needs_depth
                    .then(|| pass.depth.map(|id| self.texture(id).format))
                    .flatten(),
                samples: color.params.sample_count.max(1) as u32,
            }
        } else {
            PipelineKey {
                format: self.surface_config.format,
                depth: needs_depth.then_some(::wgpu::TextureFormat::Depth24PlusStencil8),
                samples: self.sample_count,
            }
        }
    }

    fn texture(&self, id: TextureId) -> &Texture {
        match id.0 {
            TextureIdInner::Managed(id) => self.textures[id].as_ref().unwrap(),
            _ => panic!("raw texture is not a wgpu texture"),
        }
    }

    fn ensure_pipeline(&mut self, id: Pipeline, key: PipelineKey) {
        if self.pipelines[id.0]
            .as_ref()
            .unwrap()
            .variants
            .contains_key(&key)
        {
            return;
        }
        let resource = self.pipelines[id.0].as_ref().unwrap();
        let shader = self.shaders[resource.shader.0].as_ref().unwrap();
        let mut attribute_storage: Vec<Vec<::wgpu::VertexAttribute>> =
            vec![vec![]; resource.buffers.len()];
        let mut offsets = vec![0u64; resource.buffers.len()];
        for (location, attribute) in resource.attributes.iter().enumerate() {
            let buffer = attribute.buffer_index;
            let format = match attribute.format {
                VertexFormat::Float1 => ::wgpu::VertexFormat::Float32,
                VertexFormat::Float2 => ::wgpu::VertexFormat::Float32x2,
                VertexFormat::Float3 => ::wgpu::VertexFormat::Float32x3,
                VertexFormat::Float4 => ::wgpu::VertexFormat::Float32x4,
                VertexFormat::Byte1 | VertexFormat::Byte2 => {
                    panic!("wgpu has no 1/2 component byte vertex format")
                }
                VertexFormat::Byte3 | VertexFormat::Byte4 => ::wgpu::VertexFormat::Uint8x4,
                VertexFormat::Short1 | VertexFormat::Short3 => {
                    panic!("wgpu has no 1/3 component short vertex format")
                }
                VertexFormat::Short2 => ::wgpu::VertexFormat::Uint16x2,
                VertexFormat::Short4 => ::wgpu::VertexFormat::Uint16x4,
                VertexFormat::Int1 => ::wgpu::VertexFormat::Uint32,
                VertexFormat::Int2 => ::wgpu::VertexFormat::Uint32x2,
                VertexFormat::Int3 => ::wgpu::VertexFormat::Uint32x3,
                VertexFormat::Int4 => ::wgpu::VertexFormat::Uint32x4,
                VertexFormat::Mat4 => {
                    panic!("Mat4 attributes must be expanded into four Float4 attributes")
                }
            };
            attribute_storage[buffer].push(::wgpu::VertexAttribute {
                format,
                offset: offsets[buffer],
                shader_location: location as u32,
            });
            offsets[buffer] += attribute.format.size_bytes() as u64;
        }
        let layouts: Vec<_> = resource
            .buffers
            .iter()
            .enumerate()
            .map(|(index, layout)| ::wgpu::VertexBufferLayout {
                array_stride: if layout.stride == 0 {
                    offsets[index]
                } else {
                    layout.stride as u64
                },
                step_mode: if layout.step_func == VertexStep::PerInstance {
                    ::wgpu::VertexStepMode::Instance
                } else {
                    ::wgpu::VertexStepMode::Vertex
                },
                attributes: attribute_storage[index].as_slice(),
            })
            .collect();
        let blend = |state: BlendState| ::wgpu::BlendComponent {
            operation: match state.equation {
                Equation::Add => ::wgpu::BlendOperation::Add,
                Equation::Subtract => ::wgpu::BlendOperation::Subtract,
                Equation::ReverseSubtract => ::wgpu::BlendOperation::ReverseSubtract,
            },
            src_factor: blend_factor(state.sfactor),
            dst_factor: blend_factor(state.dfactor),
        };
        let blend_state = resource.params.color_blend.map(|color| ::wgpu::BlendState {
            color: blend(color),
            alpha: blend(resource.params.alpha_blend.unwrap_or(color)),
        });
        let write_mask = {
            let (r, g, b, a) = resource.params.color_write;
            let mut m = ::wgpu::ColorWrites::empty();
            if r {
                m |= ::wgpu::ColorWrites::RED
            }
            if g {
                m |= ::wgpu::ColorWrites::GREEN
            }
            if b {
                m |= ::wgpu::ColorWrites::BLUE
            }
            if a {
                m |= ::wgpu::ColorWrites::ALPHA
            }
            m
        };
        let depth_stencil = key.depth.map(|format| ::wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(resource.params.depth_write),
            depth_compare: Some(compare(resource.params.depth_test)),
            stencil: Default::default(),
            bias: resource
                .params
                .depth_write_offset
                .map(|(factor, units)| ::wgpu::DepthBiasState {
                    constant: units as i32,
                    slope_scale: factor,
                    clamp: 0.0,
                })
                .unwrap_or_default(),
        });
        let pipeline = self
            .device
            .create_render_pipeline(&::wgpu::RenderPipelineDescriptor {
                label: Some("miniquad pipeline"),
                layout: Some(&shader.pipeline_layout),
                vertex: ::wgpu::VertexState {
                    module: &shader.module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &layouts,
                },
                fragment: Some(::wgpu::FragmentState {
                    module: &shader.module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(::wgpu::ColorTargetState {
                        format: key.format,
                        blend: blend_state,
                        write_mask,
                    })],
                }),
                primitive: ::wgpu::PrimitiveState {
                    topology: match resource.params.primitive_type {
                        PrimitiveType::Triangles => ::wgpu::PrimitiveTopology::TriangleList,
                        PrimitiveType::Lines => ::wgpu::PrimitiveTopology::LineList,
                        PrimitiveType::Points => ::wgpu::PrimitiveTopology::PointList,
                    },
                    front_face: if resource.params.front_face_order == FrontFaceOrder::Clockwise {
                        ::wgpu::FrontFace::Cw
                    } else {
                        ::wgpu::FrontFace::Ccw
                    },
                    cull_mode: match resource.params.cull_face {
                        CullFace::Nothing => None,
                        CullFace::Front => Some(::wgpu::Face::Front),
                        CullFace::Back => Some(::wgpu::Face::Back),
                    },
                    ..Default::default()
                },
                depth_stencil,
                multisample: ::wgpu::MultisampleState {
                    count: key.samples,
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            });
        self.pipelines[id.0]
            .as_mut()
            .unwrap()
            .variants
            .insert(key, pipeline);
    }

    fn encode_pass(&mut self, recording: PassRecording, encoder: &mut ::wgpu::CommandEncoder) {
        if recording.target.is_none() && !self.ensure_surface_frame() {
            return;
        }
        let needs_depth = recording.draws.iter().any(|draw| {
            let params = &self.pipelines[draw.pipeline.0].as_ref().unwrap().params;
            params.depth_test != Comparison::Always
                || params.depth_write
                || params.stencil_test.is_some()
        });
        let initialize_default_depth =
            recording.target.is_none() && needs_depth && !self.default_depth_used_this_frame;
        if recording.target.is_none() && needs_depth {
            self.default_depth_used_this_frame = true;
        }
        let key = self.pipeline_key(recording.target, needs_depth);
        let (render_width, render_height) = if let Some(pass_id) = recording.target {
            let color = self.passes[pass_id.0].as_ref().unwrap().colors[0];
            let params = self.texture(color).params;
            (params.width, params.height)
        } else {
            (self.surface_config.width, self.surface_config.height)
        };
        for draw in &recording.draws {
            self.ensure_pipeline(draw.pipeline, key);
        }
        let groups: Vec<_> = recording
            .draws
            .iter()
            .map(|draw| {
                let shader = self.pipelines[draw.pipeline.0].as_ref().unwrap().shader;
                self.bind_group(shader, &draw.images)
            })
            .collect();
        let (color_views, resolve_views, depth_view): (
            Vec<&::wgpu::TextureView>,
            Vec<Option<&::wgpu::TextureView>>,
            Option<&::wgpu::TextureView>,
        ) = if let Some(pass_id) = recording.target {
            let pass = self.passes[pass_id.0].as_ref().unwrap();
            (
                pass.colors
                    .iter()
                    .map(|id| &self.texture(*id).view)
                    .collect(),
                pass.resolves
                    .as_ref()
                    .map(|ids| ids.iter().map(|id| Some(&self.texture(*id).view)).collect())
                    .unwrap_or_else(|| vec![None; pass.colors.len()]),
                needs_depth
                    .then(|| pass.depth.map(|id| &self.texture(id).view))
                    .flatten(),
            )
        } else {
            let surface = self.surface_view.as_ref().unwrap();
            if let Some(msaa) = &self.default_msaa {
                (
                    vec![&msaa.view],
                    vec![Some(surface)],
                    needs_depth
                        .then(|| self.default_depth.as_ref().map(|d| &d.view))
                        .flatten(),
                )
            } else {
                (
                    vec![surface],
                    vec![None],
                    needs_depth
                        .then(|| self.default_depth.as_ref().map(|d| &d.view))
                        .flatten(),
                )
            }
        };
        let (clear_color, clear_depth, clear_stencil) = match recording.action {
            PassAction::Nothing => (None, None, None),
            PassAction::Clear {
                color,
                depth,
                stencil,
            } => (color, depth, stencil),
        };
        let color_attachments: Vec<_> = color_views
            .iter()
            .zip(resolve_views.iter())
            .map(|(view, resolve)| {
                Some(::wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: *resolve,
                    ops: ::wgpu::Operations {
                        load: clear_color
                            .map(|(r, g, b, a)| {
                                ::wgpu::LoadOp::Clear(::wgpu::Color {
                                    r: r as f64,
                                    g: g as f64,
                                    b: b as f64,
                                    a: a as f64,
                                })
                            })
                            .unwrap_or(::wgpu::LoadOp::Load),
                        store: ::wgpu::StoreOp::Store,
                    },
                })
            })
            .collect();
        let depth_attachment = depth_view.map(|view| ::wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(::wgpu::Operations {
                load: clear_depth
                    .or(initialize_default_depth.then_some(1.0))
                    .map(::wgpu::LoadOp::Clear)
                    .unwrap_or(::wgpu::LoadOp::Load),
                store: ::wgpu::StoreOp::Store,
            }),
            stencil_ops: clear_stencil.map(|value| ::wgpu::Operations {
                load: ::wgpu::LoadOp::Clear(value as u32),
                store: ::wgpu::StoreOp::Store,
            }),
        });
        let mut pass = encoder.begin_render_pass(&::wgpu::RenderPassDescriptor {
            label: Some("miniquad render pass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: depth_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let mut active_pipeline = None;
        let mut active_vertices: SmallVec<[BufferId; 4]> = SmallVec::new();
        let mut active_index = None;
        let mut active_viewport = None;
        let mut active_scissor = None;
        for (draw, group) in recording.draws.into_iter().zip(groups.iter()) {
            let pipeline_resource = self.pipelines[draw.pipeline.0].as_ref().unwrap();
            if active_pipeline != Some(draw.pipeline) {
                pass.set_pipeline(&pipeline_resource.variants[&key]);
                active_pipeline = Some(draw.pipeline);
            }
            if self.shaders[pipeline_resource.shader.0]
                .as_ref()
                .unwrap()
                .uniform_size
                == 0
            {
                pass.set_bind_group(0, group.as_ref(), &[]);
            } else {
                pass.set_bind_group(0, group.as_ref(), &[draw.uniform_offset]);
            }
            if active_vertices != draw.vertex_buffers {
                for (slot, buffer) in draw.vertex_buffers.iter().enumerate() {
                    pass.set_vertex_buffer(
                        slot as u32,
                        self.buffers[buffer.0].as_ref().unwrap().raw.slice(..),
                    );
                }
                active_vertices = draw.vertex_buffers.clone();
            }
            if active_index != Some(draw.index_buffer) {
                let index = self.buffers[draw.index_buffer.0].as_ref().unwrap();
                pass.set_index_buffer(
                    index.raw.slice(..),
                    if index.element_size == 4 {
                        ::wgpu::IndexFormat::Uint32
                    } else {
                        ::wgpu::IndexFormat::Uint16
                    },
                );
                active_index = Some(draw.index_buffer);
            }
            if active_viewport != draw.viewport {
                let (x, y, w, h) =
                    draw.viewport
                        .unwrap_or((0.0, 0.0, render_width as f32, render_height as f32));
                pass.set_viewport(x, render_height as f32 - y - h, w, h, 0.0, 1.0);
                active_viewport = draw.viewport;
            }
            if active_scissor != draw.scissor {
                let (x, y, w, h) = draw.scissor.unwrap_or((0, 0, render_width, render_height));
                pass.set_scissor_rect(x, render_height.saturating_sub(y + h), w, h);
                active_scissor = draw.scissor;
            }
            pass.draw_indexed(
                draw.base_element..draw.base_element + draw.elements,
                draw.base_vertex,
                0..draw.instances,
            );
        }
        drop(pass);
    }

    fn submit_frame_encoder(&mut self) {
        let Some(encoder) = self.frame_encoder.take() else {
            return;
        };
        if !self.uniform_cpu.is_empty() {
            self.queue
                .write_buffer(&self.uniform_buffer, 0, &self.uniform_cpu);
        }
        self.staging_belt.finish();
        self.queue.submit(Some(encoder.finish()));
        self.staging_belt.recall();
    }

    fn take_frame_encoder(&mut self) -> ::wgpu::CommandEncoder {
        self.frame_encoder.take().unwrap_or_else(|| {
            self.device
                .create_command_encoder(&::wgpu::CommandEncoderDescriptor {
                    label: Some("miniquad frame encoder"),
                })
        })
    }

    fn flush_recorded_pass(&mut self) {
        let Some(pass) = self.pending_pass.take() else {
            return;
        };
        self.ensure_uniform_capacity();
        let mut encoder = self.take_frame_encoder();
        self.encode_pass(pass, &mut encoder);
        self.frame_encoder = Some(encoder);
    }
}

fn compare(value: Comparison) -> ::wgpu::CompareFunction {
    match value {
        Comparison::Never => ::wgpu::CompareFunction::Never,
        Comparison::Less => ::wgpu::CompareFunction::Less,
        Comparison::LessOrEqual => ::wgpu::CompareFunction::LessEqual,
        Comparison::Greater => ::wgpu::CompareFunction::Greater,
        Comparison::GreaterOrEqual => ::wgpu::CompareFunction::GreaterEqual,
        Comparison::Equal => ::wgpu::CompareFunction::Equal,
        Comparison::NotEqual => ::wgpu::CompareFunction::NotEqual,
        Comparison::Always => ::wgpu::CompareFunction::Always,
    }
}
fn blend_factor(value: BlendFactor) -> ::wgpu::BlendFactor {
    match value {
        BlendFactor::Zero => ::wgpu::BlendFactor::Zero,
        BlendFactor::One => ::wgpu::BlendFactor::One,
        BlendFactor::Value(BlendValue::SourceColor) => ::wgpu::BlendFactor::Src,
        BlendFactor::Value(BlendValue::SourceAlpha) => ::wgpu::BlendFactor::SrcAlpha,
        BlendFactor::Value(BlendValue::DestinationColor) => ::wgpu::BlendFactor::Dst,
        BlendFactor::Value(BlendValue::DestinationAlpha) => ::wgpu::BlendFactor::DstAlpha,
        BlendFactor::OneMinusValue(BlendValue::SourceColor) => ::wgpu::BlendFactor::OneMinusSrc,
        BlendFactor::OneMinusValue(BlendValue::SourceAlpha) => {
            ::wgpu::BlendFactor::OneMinusSrcAlpha
        }
        BlendFactor::OneMinusValue(BlendValue::DestinationColor) => {
            ::wgpu::BlendFactor::OneMinusDst
        }
        BlendFactor::OneMinusValue(BlendValue::DestinationAlpha) => {
            ::wgpu::BlendFactor::OneMinusDstAlpha
        }
        BlendFactor::SourceAlphaSaturate => ::wgpu::BlendFactor::SrcAlphaSaturated,
    }
}

impl RenderingBackend for WgpuContext {
    fn info(&self) -> ContextInfo {
        ContextInfo {
            backend: Backend::Wgpu,
            gl_version_string: String::new(),
            glsl_support: Default::default(),
            features: Default::default(),
        }
    }
    fn new_shader(
        &mut self,
        source: ShaderSource,
        meta: ShaderMeta,
    ) -> Result<ShaderId, ShaderError> {
        let program = match source {
            ShaderSource::Wgsl { program } => program,
            _ => return Err(ShaderError::LinkError(
                "wgpu requires ShaderSource::Wgsl; GLSL translation is intentionally unsupported"
                    .into(),
            )),
        };
        let error_scope = self
            .device
            .push_error_scope(::wgpu::ErrorFilter::Validation);
        let module = self
            .device
            .create_shader_module(::wgpu::ShaderModuleDescriptor {
                label: Some("miniquad WGSL shader"),
                source: ::wgpu::ShaderSource::Wgsl(program.into()),
            });
        let mut raw_uniform_size = 0usize;
        let mut packed_uniform_size = 0usize;
        let mut uniform_copies = Vec::new();
        for uniform in &meta.uniforms.uniforms {
            let element_size = uniform.uniform_type.size();
            let alignment = match uniform.uniform_type {
                UniformType::Float1 | UniformType::Int1 => 4,
                UniformType::Float2 | UniformType::Int2 => 8,
                _ => 16,
            };
            let stride = element_size.div_ceil(alignment) * alignment;
            packed_uniform_size = packed_uniform_size.div_ceil(alignment) * alignment;
            for element in 0..uniform.array_count {
                uniform_copies.push((
                    raw_uniform_size + element * element_size,
                    packed_uniform_size + element * stride,
                    element_size,
                ));
            }
            raw_uniform_size += element_size * uniform.array_count;
            packed_uniform_size += stride * uniform.array_count;
        }
        let uniform_size = if packed_uniform_size == 0 {
            0
        } else {
            packed_uniform_size.div_ceil(16) * 16
        } as u64;
        let mut entries = Vec::with_capacity(meta.images.len() * 2 + 1);
        if uniform_size != 0 {
            entries.push(::wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: ::wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: ::wgpu::BindingType::Buffer {
                    ty: ::wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: NonZeroU64::new(uniform_size),
                },
                count: None,
            });
        }
        for index in 0..meta.images.len() {
            entries.push(::wgpu::BindGroupLayoutEntry {
                binding: 1 + index as u32 * 2,
                visibility: ::wgpu::ShaderStages::FRAGMENT,
                ty: ::wgpu::BindingType::Texture {
                    sample_type: ::wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: ::wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
            entries.push(::wgpu::BindGroupLayoutEntry {
                binding: 2 + index as u32 * 2,
                visibility: ::wgpu::ShaderStages::FRAGMENT,
                ty: ::wgpu::BindingType::Sampler(::wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&::wgpu::BindGroupLayoutDescriptor {
                    label: Some("miniquad shader bindings"),
                    entries: &entries,
                });
        let pipeline_layout =
            self.device
                .create_pipeline_layout(&::wgpu::PipelineLayoutDescriptor {
                    label: Some("miniquad shader layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });
        let _ = self.device.poll(::wgpu::PollType::wait_indefinitely());
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            return Err(ShaderError::CompilationError {
                shader_type: ShaderType::Vertex,
                error_message: error.to_string(),
            });
        }
        Ok(ShaderId(Self::insert(
            &mut self.shaders,
            Shader {
                module,
                meta,
                bind_group_layout,
                pipeline_layout,
                uniform_size,
                raw_uniform_size,
                uniform_copies,
            },
        )))
    }
    fn new_texture(
        &mut self,
        _access: TextureAccess,
        data: TextureSource,
        mut params: TextureParams,
    ) -> TextureId {
        params.width = params.width.max(1);
        params.height = params.height.max(1);
        let texture = self.create_texture_resource(params, "miniquad texture");
        let id = TextureId(TextureIdInner::Managed(Self::insert(
            &mut self.textures,
            texture,
        )));
        match data {
            TextureSource::Empty => {}
            TextureSource::Bytes(bytes) => self.texture_update(id, bytes),
            TextureSource::Array(faces) => {
                for (layer, mips) in faces.iter().enumerate() {
                    for (mip, bytes) in mips.iter().enumerate() {
                        let width = (params.width >> mip).max(1);
                        let height = (params.height >> mip).max(1);
                        self.write_texture(id, mip as u32, layer as u32, width, height, bytes);
                    }
                }
            }
        }
        id
    }
    fn texture_params(&self, texture: TextureId) -> TextureParams {
        self.texture(texture).params
    }
    unsafe fn texture_raw_id(&self, _: TextureId) -> RawId {
        panic!("wgpu textures have no portable RawId")
    }
    fn texture_set_min_filter(
        &mut self,
        id: TextureId,
        filter: FilterMode,
        mipmap: MipmapFilterMode,
    ) {
        self.flush_recorded_pass();
        let t = self.texture_mut(id);
        t.params.min_filter = filter;
        t.params.mipmap_filter = mipmap;
        self.sampler_generation += 1;
        self.bind_groups.clear();
    }
    fn texture_set_mag_filter(&mut self, id: TextureId, filter: FilterMode) {
        self.flush_recorded_pass();
        self.texture_mut(id).params.mag_filter = filter;
        self.sampler_generation += 1;
        self.bind_groups.clear();
    }
    fn texture_set_wrap(&mut self, id: TextureId, x: TextureWrap, _y: TextureWrap) {
        self.flush_recorded_pass();
        self.texture_mut(id).params.wrap = x;
        self.sampler_generation += 1;
        self.bind_groups.clear();
    }
    fn texture_generate_mipmaps(&mut self, _: TextureId) {}
    fn texture_resize(&mut self, id: TextureId, width: u32, height: u32, bytes: Option<&[u8]>) {
        self.flush_recorded_pass();
        let mut p = self.texture(id).params;
        p.width = width;
        p.height = height;
        let replacement = self.create_texture_resource(p, "miniquad resized texture");
        self.textures[match id.0 {
            TextureIdInner::Managed(i) => i,
            _ => unreachable!(),
        }] = Some(replacement);
        self.bind_groups.clear();
        if let Some(bytes) = bytes {
            self.texture_update(id, bytes)
        }
    }
    fn texture_read_pixels(&mut self, id: TextureId, bytes: &mut [u8]) {
        self.read_texture(id, bytes);
    }
    fn texture_update_part(&mut self, id: TextureId, x: i32, y: i32, w: i32, h: i32, bytes: &[u8]) {
        self.flush_recorded_pass();
        self.write_texture_region(id, 0, 0, x as u32, y as u32, w as u32, h as u32, bytes);
    }
    fn new_render_pass_mrt(
        &mut self,
        colors: &[TextureId],
        resolves: Option<&[TextureId]>,
        depth: Option<TextureId>,
    ) -> RenderPass {
        RenderPass(Self::insert(
            &mut self.passes,
            Pass {
                colors: colors.to_vec(),
                resolves: resolves.map(<[TextureId]>::to_vec),
                depth,
            },
        ))
    }
    fn render_pass_color_attachments(&self, pass: RenderPass) -> &[TextureId] {
        &self.passes[pass.0].as_ref().unwrap().colors
    }
    fn delete_render_pass(&mut self, pass: RenderPass) {
        self.flush_recorded_pass();
        self.passes[pass.0] = None;
    }
    fn new_pipeline(
        &mut self,
        buffers: &[BufferLayout],
        attrs: &[VertexAttribute],
        shader: ShaderId,
        params: PipelineParams,
    ) -> Pipeline {
        Pipeline(Self::insert(
            &mut self.pipelines,
            PipelineResource {
                shader,
                buffers: buffers.to_vec(),
                attributes: attrs.to_vec(),
                params,
                variants: HashMap::new(),
            },
        ))
    }
    fn apply_pipeline(&mut self, pipeline: &Pipeline) {
        self.current_pipeline = Some(*pipeline)
    }
    fn delete_pipeline(&mut self, pipeline: Pipeline) {
        self.flush_recorded_pass();
        self.pipelines[pipeline.0] = None;
    }
    fn new_buffer(
        &mut self,
        kind: BufferType,
        _usage: BufferUsage,
        data: BufferSource,
    ) -> BufferId {
        let (size, element_size, contents) = match data {
            BufferSource::Empty { size, element_size } => (size, element_size, None),
            BufferSource::Slice(arg) => {
                let bytes = unsafe { std::slice::from_raw_parts(arg.ptr as *const u8, arg.size) };
                (arg.size, arg.element_size, Some(bytes))
            }
        };
        let flags = ::wgpu::BufferUsages::COPY_DST
            | match kind {
                BufferType::VertexBuffer => ::wgpu::BufferUsages::VERTEX,
                BufferType::IndexBuffer => ::wgpu::BufferUsages::INDEX,
            };
        let raw = if let Some(bytes) = contents {
            self.device
                .create_buffer_init(&::wgpu::util::BufferInitDescriptor {
                    label: Some("miniquad buffer"),
                    contents: bytes,
                    usage: flags,
                })
        } else {
            self.device.create_buffer(&::wgpu::BufferDescriptor {
                label: Some("miniquad buffer"),
                size: aligned_buffer_size(size).max(::wgpu::COPY_BUFFER_ALIGNMENT as usize) as u64,
                usage: flags,
                mapped_at_creation: false,
            })
        };
        BufferId(Self::insert(
            &mut self.buffers,
            Buffer {
                raw,
                size,
                element_size,
            },
        ))
    }
    fn buffer_update(&mut self, id: BufferId, data: BufferSource) {
        // The common macroquad path uploads its whole dynamic batch before
        // recording any pass. Queue writes are materially cheaper there than
        // encoding an extra GPU copy. Once a pass has been recorded we retain
        // the staging-copy path so API ordering remains exact.
        let can_write_direct = self.pending_pass.is_none() && self.frame_encoder.is_none();
        if !can_write_direct {
            self.flush_recorded_pass();
        }
        let bytes = match data {
            BufferSource::Slice(arg) => unsafe {
                std::slice::from_raw_parts(arg.ptr as *const u8, arg.size)
            },
            _ => panic!("buffer update needs data"),
        };
        let logical_size = self.buffers[id.0].as_ref().unwrap().size;
        assert!(bytes.len() <= logical_size);
        let write_size = aligned_buffer_size(bytes.len());
        if write_size == 0 {
            return;
        }
        if write_size != bytes.len() {
            self.buffer_upload_scratch.clear();
            self.buffer_upload_scratch.extend_from_slice(bytes);
            self.buffer_upload_scratch.resize(write_size, 0);
        }
        let mut encoder = self.take_frame_encoder();
        let upload = if write_size == bytes.len() {
            bytes
        } else {
            &self.buffer_upload_scratch
        };
        let buffer = self.buffers[id.0].as_ref().unwrap();
        if can_write_direct {
            self.queue.write_buffer(&buffer.raw, 0, upload);
            return;
        }
        let mut staging = self.staging_belt.write_buffer(
            &mut encoder,
            &buffer.raw,
            0,
            NonZeroU64::new(upload.len() as u64).unwrap(),
        );
        staging.copy_from_slice(upload);
        drop(staging);
        self.frame_encoder = Some(encoder);
    }
    fn buffer_size(&mut self, id: BufferId) -> usize {
        self.buffers[id.0].as_ref().unwrap().size
    }
    fn delete_buffer(&mut self, id: BufferId) {
        self.flush_recorded_pass();
        self.buffers[id.0] = None;
    }
    fn delete_texture(&mut self, id: TextureId) {
        // Draw calls are recorded until end_render_pass; keep resources alive
        // through submission even when the higher layer drops a scene mid-frame.
        self.pending_texture_deletes.push(id);
    }
    fn delete_shader(&mut self, id: ShaderId) {
        self.flush_recorded_pass();
        self.shaders[id.0] = None;
        self.bind_groups.retain(|key, _| key.shader != id);
    }
    fn apply_viewport(&mut self, x: i32, y: i32, w: i32, h: i32) {
        self.viewport = Some((x as f32, y as f32, w as f32, h as f32));
    }
    fn apply_scissor_rect(&mut self, x: i32, y: i32, w: i32, h: i32) {
        self.scissor = Some((
            x.max(0) as u32,
            y.max(0) as u32,
            w.max(0) as u32,
            h.max(0) as u32,
        ));
    }
    fn apply_bindings_from_slice(&mut self, v: &[BufferId], i: BufferId, t: &[TextureId]) {
        self.current_vertices.clear();
        self.current_vertices.extend_from_slice(v);
        self.current_index = Some(i);
        self.current_images.clear();
        self.current_images.extend_from_slice(t);
    }
    fn apply_uniforms_from_bytes(&mut self, ptr: *const u8, size: usize) {
        self.current_uniforms.clear();
        self.current_uniforms
            .extend_from_slice(unsafe { std::slice::from_raw_parts(ptr, size) });
    }
    fn clear(
        &mut self,
        color: Option<(f32, f32, f32, f32)>,
        depth: Option<f32>,
        stencil: Option<i32>,
    ) {
        if let Some(pass) = &mut self.current_pass {
            assert!(pass.draws.is_empty(), "clear after drawing is unsupported");
            pass.action = PassAction::Clear {
                color,
                depth,
                stencil,
            };
        }
    }
    fn begin_default_pass(&mut self, action: PassAction) {
        self.begin_pass(None, action)
    }
    fn begin_pass(&mut self, target: Option<RenderPass>, action: PassAction) {
        assert!(self.current_pass.is_none());
        let can_merge = matches!(action, PassAction::Nothing)
            && self
                .pending_pass
                .as_ref()
                .is_some_and(|pending| pending.target == target);
        if can_merge {
            self.current_pass = self.pending_pass.take();
        } else {
            self.flush_recorded_pass();
            self.current_pass = Some(PassRecording {
                target,
                action,
                draws: vec![],
            });
        }
        self.viewport = None;
        self.scissor = None;
    }
    fn end_render_pass(&mut self) {
        if let Some(pass) = self.current_pass.take() {
            debug_assert!(self.pending_pass.is_none());
            self.pending_pass = Some(pass);
        }
    }
    fn commit_frame(&mut self) {
        self.sync_surface_source();
        if self.current_pass.is_some() {
            self.end_render_pass()
        }
        self.flush_recorded_pass();
        self.submit_frame_encoder();
        if let Some(frame) = self.surface_frame.take() {
            self.surface_view = None;
            frame.present();
        }
        self.default_depth_used_this_frame = false;
        self.uniform_cpu.clear();
        // Polling wgpu-core every frame is measurable at uncapped frame rates.
        // The staging belt may keep a few chunks in flight; polling every eight
        // frames bounds that memory while amortizing the native synchronization.
        self.poll_frame = self.poll_frame.wrapping_add(1) & 7;
        if self.poll_frame == 0 {
            let _ = self.device.poll(::wgpu::PollType::Poll);
        }
        for id in self.pending_texture_deletes.drain(..) {
            if let TextureIdInner::Managed(i) = id.0 {
                self.textures[i] = None;
                self.bind_groups.retain(|key, _| !key.images.contains(&id));
            }
        }
    }
    fn draw(&mut self, base: i32, count: i32, instances: i32) {
        self.draw_with_base_vertex(base, count, instances, 0);
    }
    fn draw_with_base_vertex(&mut self, base: i32, count: i32, instances: i32, base_vertex: i32) {
        let pipeline = self.current_pipeline.expect("draw without pipeline");
        let index = self.current_index.expect("draw without index buffer");
        let shader = self.pipelines[pipeline.0].as_ref().unwrap().shader;
        let shader_resource = self.shaders[shader.0].as_ref().unwrap();
        assert_eq!(
            shader_resource.raw_uniform_size,
            self.current_uniforms.len(),
            "uniform data size does not match ShaderMeta"
        );
        let uniform_offset = if shader_resource.uniform_size == 0 {
            0
        } else {
            let offset = self.uniform_cpu.len().div_ceil(UNIFORM_ALIGNMENT) * UNIFORM_ALIGNMENT;
            self.uniform_cpu
                .resize(offset + shader_resource.uniform_size as usize, 0);
            for &(src, dst, size) in &shader_resource.uniform_copies {
                self.uniform_cpu[offset + dst..offset + dst + size]
                    .copy_from_slice(&self.current_uniforms[src..src + size]);
            }
            offset as u32
        };
        self.current_pass
            .as_mut()
            .expect("draw outside pass")
            .draws
            .push(DrawCall {
                pipeline,
                vertex_buffers: self.current_vertices.clone(),
                index_buffer: index,
                images: self.current_images.clone(),
                uniform_offset,
                viewport: self.viewport,
                scissor: self.scissor,
                base_element: base as u32,
                base_vertex,
                elements: count as u32,
                instances: instances.max(1) as u32,
            });
    }
}

impl WgpuContext {
    fn texture_mut(&mut self, id: TextureId) -> &mut Texture {
        match id.0 {
            TextureIdInner::Managed(i) => self.textures[i].as_mut().unwrap(),
            _ => panic!("raw texture"),
        }
    }
    fn write_texture(&self, id: TextureId, mip: u32, layer: u32, w: u32, h: u32, bytes: &[u8]) {
        self.write_texture_region(id, mip, layer, 0, 0, w, h, bytes)
    }
    fn write_texture_region(
        &self,
        id: TextureId,
        mip: u32,
        layer: u32,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        bytes: &[u8],
    ) {
        let t = self.texture(id);
        let owned;
        let bytes = if t.params.format == TextureFormat::RGB8 {
            owned = bytes
                .chunks_exact(3)
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect::<Vec<_>>();
            &owned
        } else {
            bytes
        };
        let bpp = match t.params.format {
            TextureFormat::Alpha => 1,
            TextureFormat::RGBA16F => 8,
            _ => 4,
        };
        self.queue.write_texture(
            ::wgpu::TexelCopyTextureInfo {
                texture: &t.raw,
                mip_level: mip,
                origin: ::wgpu::Origin3d { x, y, z: layer },
                aspect: ::wgpu::TextureAspect::All,
            },
            bytes,
            ::wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * bpp),
                rows_per_image: Some(h),
            },
            ::wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }
    fn read_texture(&mut self, id: TextureId, bytes: &mut [u8]) {
        self.flush_recorded_pass();
        self.submit_frame_encoder();
        let t = self.texture(id);
        let bpp = match t.params.format {
            TextureFormat::Alpha => 1,
            TextureFormat::RGBA16F => 8,
            _ => 4,
        };
        let row = t.params.width * bpp;
        let padded = row.div_ceil(::wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * ::wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let size = (padded * t.params.height) as u64;
        let staging = self.device.create_buffer(&::wgpu::BufferDescriptor {
            label: Some("miniquad readback"),
            size,
            usage: ::wgpu::BufferUsages::COPY_DST | ::wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            ::wgpu::TexelCopyTextureInfo {
                texture: &t.raw,
                mip_level: 0,
                origin: Default::default(),
                aspect: ::wgpu::TextureAspect::All,
            },
            ::wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: ::wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(t.params.height),
                },
            },
            ::wgpu::Extent3d {
                width: t.params.width,
                height: t.params.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        slice.map_async(::wgpu::MapMode::Read, move |r| {
            let _ = done_tx.send(r);
        });
        let _ = self.device.poll(::wgpu::PollType::wait_indefinitely());
        done_rx.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        for (y, dst) in bytes.chunks_mut(row as usize).enumerate() {
            let start = y * padded as usize;
            dst.copy_from_slice(&mapped[start..start + row as usize]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{aligned_buffer_size, backend_candidates, BackendPlatform};
    use crate::conf::WgpuBackend;

    #[test]
    fn buffer_upload_sizes_respect_wgpu_alignment() {
        assert_eq!(aligned_buffer_size(0), 0);
        assert_eq!(aligned_buffer_size(1), 4);
        assert_eq!(aligned_buffer_size(4), 4);
        assert_eq!(aligned_buffer_size(6), 8);
        assert_eq!(aligned_buffer_size(8), 8);
    }

    #[test]
    fn android_auto_tries_vulkan_before_gles() {
        let candidates = backend_candidates(WgpuBackend::Auto, BackendPlatform::Android);
        assert_eq!(
            candidates,
            vec![
                (::wgpu::Backends::VULKAN, "Vulkan"),
                (::wgpu::Backends::GL, "GLES"),
            ]
        );
    }

    #[test]
    fn explicit_android_backend_does_not_fall_back() {
        assert_eq!(
            backend_candidates(WgpuBackend::Vulkan, BackendPlatform::Android),
            vec![(::wgpu::Backends::VULKAN, "Vulkan")]
        );
        assert_eq!(
            backend_candidates(WgpuBackend::Gles, BackendPlatform::Android),
            vec![(::wgpu::Backends::GL, "GLES")]
        );
    }
}
