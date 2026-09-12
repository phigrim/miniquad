//! Portable renderer. Public handles stay backend independent; GPU resources are owned here.
use super::*;
use crate::ResourceManager;
use ::wgpu;
#[cfg(test)]
use std::cell::Cell;
use std::{borrow::Cow, cell::RefCell, collections::HashMap, num::NonZeroU64, rc::Rc, sync::Arc};
use wgpu::util::DeviceExt;
mod pipeline;
mod shader;
mod texture;
use pipeline::PipelineState;
use shader::Shader;
use texture::Texture;

struct Buffer {
    gpu: wgpu::Buffer,
    gpu_offset: u64,
    bytes: Vec<u8>,
    element_size: usize,
    kind: BufferType,
    usage: BufferUsage,
}

struct GeometryBufferArena {
    gpu: wgpu::Buffer,
    capacity: u64,
    offset: u64,
}

pub(super) struct ResolvedBuffer {
    gpu: wgpu::Buffer,
    offset: u64,
}

/// Fully resolved state for one miniquad draw. Keeping these until
/// `end_render_pass` lets the wgpu backend encode adjacent miniquad draws into
/// one native render pass instead of paying render-pass setup per draw.
struct DrawCommand {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_offset: u32,
    vertices: Vec<ResolvedBuffer>,
    index: wgpu::Buffer,
    index_offset: u64,
    index_format: wgpu::IndexFormat,
    base: u32,
    count: u32,
    instances: u32,
    viewport: Option<(f32, f32, f32, f32)>,
    scissor: Option<(u32, u32, u32, u32)>,
    stencil_reference: Option<u32>,
}

struct UniformBufferArena {
    gpu: wgpu::Buffer,
    capacity: u64,
    offset: u64,
}

#[derive(Hash, PartialEq, Eq)]
struct BindGroupKey {
    shader: usize,
    images: Vec<TextureId>,
}

/// The overwhelmingly common binding pattern is the same pipeline/material
/// across adjacent draws. Metal records that state directly; keep the same
/// fast path here so a wgpu draw does not allocate a HashMap key merely to
/// rediscover the bind group from the preceding draw.
struct CachedBindGroup {
    shader: usize,
    images: Vec<TextureId>,
    group: wgpu::BindGroup,
}

struct Pass {
    colors: Vec<TextureId>,
    resolves: Vec<TextureId>,
    depth: Option<TextureId>,
}
#[derive(Clone, Hash, PartialEq, Eq)]
struct TargetKey {
    colors: Vec<wgpu::TextureFormat>,
    depth: Option<wgpu::TextureFormat>,
    samples: u32,
}
#[derive(Clone)]
struct Target {
    colors: Vec<wgpu::TextureView>,
    resolves: Vec<wgpu::TextureView>,
    depth: Option<wgpu::TextureView>,
    key: TargetKey,
    width: u32,
    height: u32,
}

struct SurfaceState {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    surface: Option<wgpu::Surface<'static>>,
    config: Option<wgpu::SurfaceConfiguration>,
    window: Option<Arc<winit::window::Window>>,
    frame: Option<wgpu::SurfaceTexture>,
    default_color: Option<wgpu::Texture>,
    default_depth: Option<wgpu::Texture>,
    samples: u32,
    present_mode: wgpu::PresentMode,
    framebuffer_alpha: bool,
}

#[derive(Debug)]
pub(crate) enum SurfaceInitError {
    Surface(String),
    AdapterUnavailable,
    Device(String),
    Unsupported,
}

#[derive(Clone)]
pub(crate) struct SurfaceController(Rc<RefCell<SurfaceState>>);

impl SurfaceController {
    pub(crate) fn attach_window(
        &self,
        window: Arc<winit::window::Window>,
    ) -> Result<(), SurfaceInitError> {
        let mut state = self.0.borrow_mut();
        let surface = state
            .instance
            .create_surface(window.clone())
            .map_err(|error| SurfaceInitError::Surface(error.to_string()))?;
        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&state.adapter, size.width.max(1), size.height.max(1))
            .ok_or(SurfaceInitError::Unsupported)?;
        // Metal allocates `maximum_frame_latency + 1` drawables. The default
        // of two therefore keeps three full-size IOSurfaces alive; Entry does
        // not need that extra frame of latency and pays for it in footprint.
        // Do not force this on DX12: a single frame in flight serializes more
        // CPU/GPU work and measurably hurts uncapped high-frame-rate workloads.
        #[cfg(target_os = "macos")]
        {
            config.desired_maximum_frame_latency = 1;
        }
        let capabilities = surface.get_capabilities(&state.adapter);
        config.format = capabilities
            .formats
            .into_iter()
            .find(|format| !format.is_srgb())
            .unwrap_or(config.format);
        config.present_mode = state.present_mode;
        if !state.framebuffer_alpha
            && capabilities
                .alpha_modes
                .contains(&wgpu::CompositeAlphaMode::Opaque)
        {
            config.alpha_mode = wgpu::CompositeAlphaMode::Opaque;
        }
        surface.configure(&state.device, &config);
        state.surface = Some(surface);
        state.config = Some(config);
        state.window = Some(window);
        state.frame = None;
        state.default_color = None;
        state.default_depth = None;
        Ok(())
    }

    pub(crate) fn detach_window(&self) {
        let mut state = self.0.borrow_mut();
        state.frame = None;
        state.default_color = None;
        state.default_depth = None;
        state.surface = None;
        state.config = None;
        state.window = None;
    }
}

fn compressed_texture_support(features: wgpu::Features) -> CompressedTextureSupport {
    let mut support = CompressedTextureSupport::empty();
    if features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
        support.enable_all(&[
            CompressedTextureFormat::Bc1Rgb,
            CompressedTextureFormat::Bc1Rgba,
            CompressedTextureFormat::Bc2,
            CompressedTextureFormat::Bc3,
            CompressedTextureFormat::Bc4,
            CompressedTextureFormat::Bc5,
            CompressedTextureFormat::Bc6hUnsigned,
            CompressedTextureFormat::Bc6hSigned,
            CompressedTextureFormat::Bc7,
        ]);
    }
    if features.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2) {
        support.enable_all(&[
            CompressedTextureFormat::Etc2Rgb8,
            CompressedTextureFormat::Etc2Rgb8A1,
            CompressedTextureFormat::Etc2Rgba8,
            CompressedTextureFormat::EacR11,
            CompressedTextureFormat::EacRg11,
        ]);
    }
    if features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC) {
        for &(block_width, block_height) in &[
            (4, 4),
            (5, 4),
            (5, 5),
            (6, 5),
            (6, 6),
            (8, 5),
            (8, 6),
            (8, 8),
            (10, 5),
            (10, 6),
            (10, 8),
            (10, 10),
            (12, 10),
            (12, 12),
        ] {
            support.enable(CompressedTextureFormat::Astc {
                block_width,
                block_height,
            });
        }
    }
    support
}

/// The wgpu implementation of [`RenderingBackend`].
/// Created by `window::new_rendering_backend` when `GfxApi::Wgpu` is selected.
pub struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: Rc<RefCell<SurfaceState>>,
    compressed_texture_support: CompressedTextureSupport,
    shaders: ResourceManager<Shader>,
    textures: ResourceManager<Texture>,
    // Macroquad collects textures after commit_frame. A cached draw may still
    // reference one while switching scenes before the following commit. Keep
    // removed resources addressable for that final submission.
    retired_textures: HashMap<usize, Texture>,
    buffers: ResourceManager<Buffer>,
    pipelines: ResourceManager<PipelineState>,
    passes: ResourceManager<Pass>,
    encoder: RefCell<Option<wgpu::CommandEncoder>>,
    target: Option<Target>,
    target_pass: Option<Option<RenderPass>>,
    pass_active: bool,
    current_pipeline: Option<Pipeline>,
    bindings: Option<Bindings>,
    uniforms: Vec<u8>,
    packed_uniforms: RefCell<Vec<u8>>,
    uniform_buffer: RefCell<Option<UniformBufferArena>>,
    geometry_buffer: Option<GeometryBufferArena>,
    bind_groups: RefCell<HashMap<BindGroupKey, wgpu::BindGroup>>,
    last_bind_group: RefCell<Option<CachedBindGroup>>,
    staging_belt: RefCell<wgpu::util::StagingBelt>,
    pending_draws: RefCell<Vec<DrawCommand>>,
    vertex_buffer_pool: RefCell<Vec<Vec<ResolvedBuffer>>>,
    uniform_alignment: u64,
    viewport: Option<(f32, f32, f32, f32)>,
    scissor: Option<(u32, u32, u32, u32)>,
    #[cfg(test)]
    native_render_passes: Cell<usize>,
}
impl WgpuContext {
    pub(crate) async fn for_window(
        window: Arc<winit::window::Window>,
        conf: &crate::conf::Conf,
    ) -> Result<(Self, SurfaceController), SurfaceInitError> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance
            .create_surface(window.clone())
            .map_err(|error| SurfaceInitError::Surface(error.to_string()))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .map_err(|_| SurfaceInitError::AdapterUnavailable)?;
        drop(surface);
        let context = Self::from_adapter_with_instance(
            instance,
            adapter,
            conf.sample_count.max(1) as u32,
            if conf.platform.swap_interval == Some(0) {
                wgpu::PresentMode::AutoNoVsync
            } else {
                wgpu::PresentMode::AutoVsync
            },
            conf.platform.framebuffer_alpha,
        )
        .await?;
        let controller = SurfaceController(context.surface.clone());
        controller.attach_window(window)?;
        Ok((context, controller))
    }
    #[cfg(test)]
    async fn from_adapter(adapter: wgpu::Adapter) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        Self::from_adapter_with_instance(instance, adapter, 1, wgpu::PresentMode::AutoVsync, false)
            .await
            .expect("create wgpu device")
    }
    async fn from_adapter_with_instance(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        samples: u32,
        present_mode: wgpu::PresentMode,
        framebuffer_alpha: bool,
    ) -> Result<Self, SurfaceInitError> {
        let adapter_features = adapter.features();
        let compression_features = adapter_features
            & (wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("miniquad"),
                required_features: compression_features,
                ..Default::default()
            })
            .await
            .map_err(|error| SurfaceInitError::Device(error.to_string()))?;
        let surface = Rc::new(RefCell::new(SurfaceState {
            instance,
            adapter,
            device: device.clone(),
            surface: None,
            config: None,
            window: None,
            frame: None,
            default_color: None,
            default_depth: None,
            samples,
            present_mode,
            framebuffer_alpha,
        }));
        let uniform_alignment =
            u64::from(device.limits().min_uniform_buffer_offset_alignment.max(1));
        let staging_belt = wgpu::util::StagingBelt::new(device.clone(), 256 * 1024);
        Ok(Self {
            device,
            queue,
            surface,
            compressed_texture_support: compressed_texture_support(compression_features),
            shaders: Default::default(),
            textures: Default::default(),
            retired_textures: HashMap::new(),
            buffers: Default::default(),
            pipelines: Default::default(),
            passes: Default::default(),
            encoder: RefCell::new(None),
            target: None,
            target_pass: None,
            pass_active: false,
            current_pipeline: None,
            bindings: None,
            uniforms: vec![],
            packed_uniforms: RefCell::new(vec![]),
            uniform_buffer: RefCell::new(None),
            geometry_buffer: None,
            bind_groups: RefCell::new(HashMap::new()),
            last_bind_group: RefCell::new(None),
            staging_belt: RefCell::new(staging_belt),
            pending_draws: RefCell::new(vec![]),
            vertex_buffer_pool: RefCell::new(vec![]),
            uniform_alignment,
            viewport: None,
            scissor: None,
            #[cfg(test)]
            native_render_passes: Cell::new(0),
        })
    }
    fn submit(&mut self) {
        // Some resource operations submit outside the normal frame boundary.
        // Do not let a deferred logical draw disappear when one of them is
        // called between begin_pass and end_render_pass.
        self.flush_draws();
        self.staging_belt.get_mut().finish();
        if let Some(encoder) = self.encoder.borrow_mut().take() {
            self.queue.submit([encoder.finish()]);
        }
        self.staging_belt.get_mut().recall();
        let _ = self.device.poll(wgpu::PollType::Poll);
    }
    fn encode(&self, f: impl FnOnce(&mut wgpu::CommandEncoder)) {
        let mut encoder = self.encoder.borrow_mut();
        f(encoder.get_or_insert_with(|| {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("miniquad frame"),
                })
        }));
    }
    /// Finish the logical miniquad pass accumulated since `begin_pass`.
    ///
    /// wgpu render passes cannot outlive the closure that creates them, so
    /// retaining resolved draw state is the only way to preserve miniquad's
    /// immediate API while still submitting one native render pass.
    fn flush_draws(&self) {
        let mut commands = self.pending_draws.borrow_mut();
        if commands.is_empty() {
            return;
        }
        self.render(&PassAction::Nothing, |pass| {
            for command in commands.iter() {
                pass.set_pipeline(&command.pipeline);
                pass.set_bind_group(0, &command.bind_group, &[command.uniform_offset]);
                for (i, buffer) in command.vertices.iter().enumerate() {
                    pass.set_vertex_buffer(i as u32, buffer.gpu.slice(buffer.offset..));
                }
                pass.set_index_buffer(
                    command.index.slice(command.index_offset..),
                    command.index_format,
                );
                if let Some((x, y, w, h)) = command.viewport {
                    pass.set_viewport(x, y, w, h, 0., 1.);
                }
                if let Some((x, y, w, h)) = command.scissor {
                    pass.set_scissor_rect(x, y, w, h);
                }
                if let Some(stencil) = command.stencil_reference {
                    pass.set_stencil_reference(stencil);
                }
                pass.draw_indexed(
                    command.base..command.base + command.count,
                    0,
                    0..command.instances,
                );
            }
        });
        // Keep the allocation for the next logical pass. Macroquad opens and
        // closes several small passes per frame, so `mem::take` here turned
        // every pass into a fresh allocation on the CPU hot path.
        // Each DrawCommand owns a small Vec of retained vertex-buffer handles.
        // Recycle those Vec allocations too, while clearing their handles so
        // deleted GPU resources are not kept alive by the pool.
        let mut vertex_buffer_pool = self.vertex_buffer_pool.borrow_mut();
        for command in commands.drain(..) {
            let mut vertices = command.vertices;
            vertices.clear();
            vertex_buffer_pool.push(vertices);
        }
    }
    fn texture(&self, id: TextureId) -> &Texture {
        match id.0 {
            TextureIdInner::Managed(id) => self
                .textures
                .get(id)
                .or_else(|| self.retired_textures.get(&id))
                .unwrap_or_else(|| panic!("invalid or expired wgpu texture {}", id)),
            _ => panic!("raw GL/Metal textures cannot be used with wgpu"),
        }
    }
    fn texture_mut(&mut self, id: TextureId) -> &mut Texture {
        match id.0 {
            TextureIdInner::Managed(id) => &mut self.textures[id],
            _ => panic!("raw GL/Metal textures cannot be used with wgpu"),
        }
    }
    fn default_target(&mut self) -> Option<Target> {
        let window = self
            .surface
            .borrow()
            .window
            .clone()
            .expect("headless contexts require an offscreen pass");
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return None;
        }
        let resize = {
            let state = self.surface.borrow();
            let config = state.config.as_ref().unwrap();
            config.width != size.width || config.height != size.height
        };
        if resize {
            self.submit();
            let mut state = self.surface.borrow_mut();
            let mut config = state.config.take().unwrap();
            config.width = size.width;
            config.height = size.height;
            state
                .surface
                .as_ref()
                .unwrap()
                .configure(&self.device, &config);
            state.config = Some(config);
            state.default_color = None;
            state.default_depth = None;
        }
        let mut state = self.surface.borrow_mut();
        if state.frame.is_none() {
            let surface = state.surface.as_ref().unwrap();
            state.frame = match surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t)
                | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Some(t),
                wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                    surface.configure(&self.device, state.config.as_ref().unwrap());
                    None
                }
                wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                    None
                }
                wgpu::CurrentSurfaceTexture::Validation => panic!("wgpu surface validation failed"),
            };
        }
        let format = state.config.as_ref().unwrap().format;
        let frame = state.frame.as_ref()?;
        let mut color = frame.texture.create_view(&Default::default());
        let mut resolves = vec![];
        let device = &self.device;
        let samples = state.samples;
        if samples > 1 {
            let texture = state.default_color.get_or_insert_with(|| {
                texture::attachment(device, size.width, size.height, format, samples)
            });
            resolves.push(color);
            color = texture.create_view(&Default::default());
        }
        let depth = state.default_depth.get_or_insert_with(|| {
            texture::attachment(
                device,
                size.width,
                size.height,
                wgpu::TextureFormat::Depth24PlusStencil8,
                samples,
            )
        });
        Some(Target {
            colors: vec![color],
            resolves,
            depth: Some(depth.create_view(&Default::default())),
            key: TargetKey {
                colors: vec![format],
                depth: Some(wgpu::TextureFormat::Depth24PlusStencil8),
                samples,
            },
            width: size.width,
            height: size.height,
        })
    }
    fn render(&self, action: &PassAction, f: impl FnOnce(&mut wgpu::RenderPass<'_>)) {
        let Some(target) = &self.target else {
            return;
        };
        #[cfg(test)]
        self.native_render_passes
            .set(self.native_render_passes.get() + 1);
        let (color, depth, stencil) = match action {
            PassAction::Nothing => (None, None, None),
            PassAction::Clear {
                color,
                depth,
                stencil,
            } => (*color, *depth, *stencil),
        };
        let colors: Vec<_> = target
            .colors
            .iter()
            .enumerate()
            .map(|(i, view)| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: target.resolves.get(i),
                    ops: wgpu::Operations {
                        load: color.map_or(wgpu::LoadOp::Load, |(r, g, b, a)| {
                            wgpu::LoadOp::Clear(wgpu::Color {
                                r: r as _,
                                g: g as _,
                                b: b as _,
                                a: a as _,
                            })
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })
            })
            .collect();
        let depth_stencil =
            target
                .depth
                .as_ref()
                .map(|view| wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: Some(wgpu::Operations {
                        load: depth.map_or(wgpu::LoadOp::Load, wgpu::LoadOp::Clear),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: if target.key.depth
                        == Some(wgpu::TextureFormat::Depth24PlusStencil8)
                    {
                        Some(wgpu::Operations {
                            load: stencil
                                .map_or(wgpu::LoadOp::Load, |s| wgpu::LoadOp::Clear(s as u32)),
                            store: wgpu::StoreOp::Store,
                        })
                    } else {
                        None
                    },
                });
        self.encode(|encoder| {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("miniquad pass"),
                color_attachments: &colors,
                depth_stencil_attachment: depth_stencil,
                ..Default::default()
            });
            f(&mut pass);
        });
    }
    fn allocate_uniform(&self, bytes: &[u8]) -> (wgpu::Buffer, u64, u64) {
        const INITIAL_CAPACITY: u64 = 64 * 1024;
        let binding_size = align_up(bytes.len().max(16) as u64, 16);
        let mut arena = self.uniform_buffer.borrow_mut();
        let offset = arena
            .as_ref()
            .map_or(0, |arena| align_up(arena.offset, self.uniform_alignment));
        let required = offset + binding_size;
        if arena.as_ref().is_none_or(|arena| required > arena.capacity) {
            let capacity = required.max(INITIAL_CAPACITY).next_power_of_two();
            let gpu = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("miniquad frame uniforms"),
                size: capacity,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            *arena = Some(UniformBufferArena {
                gpu,
                capacity,
                offset: 0,
            });
            self.bind_groups.borrow_mut().clear();
            self.last_bind_group.borrow_mut().take();
        }
        let arena_ref = arena.as_mut().expect("uniform buffer arena initialized");
        let offset = align_up(arena_ref.offset, self.uniform_alignment);
        let gpu = arena_ref.gpu.clone();
        arena_ref.offset = offset + binding_size;
        drop(arena);

        let mut encoder = self.encoder.borrow_mut();
        let encoder = encoder.get_or_insert_with(|| {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("miniquad frame"),
                })
        });
        let mut staging_belt = self.staging_belt.borrow_mut();
        let mut staging = staging_belt.write_buffer(
            encoder,
            &gpu,
            offset,
            NonZeroU64::new(binding_size).expect("uniform size is non-zero"),
        );
        if bytes.is_empty() {
            staging.copy_from_slice(&[0; 16]);
        } else {
            staging.copy_from_slice(bytes);
        }
        drop(staging);
        (gpu, offset, binding_size)
    }

    fn bind_group(
        &self,
        shader_id: usize,
        shader: &Shader,
        bindings: &Bindings,
        uniform: &wgpu::Buffer,
        uniform_size: u64,
    ) -> wgpu::BindGroup {
        {
            let last = self.last_bind_group.borrow();
            if let Some(cached) = last.as_ref() {
                if cached.shader == shader_id && cached.images == bindings.images {
                    return cached.group.clone();
                }
            }
        }
        let key = BindGroupKey {
            shader: shader_id,
            images: bindings.images.clone(),
        };
        if let Some(group) = self.bind_groups.borrow().get(&key).cloned() {
            *self.last_bind_group.borrow_mut() = Some(CachedBindGroup {
                shader: shader_id,
                images: bindings.images.clone(),
                group: group.clone(),
            });
            return group;
        }

        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: uniform,
                offset: 0,
                size: Some(NonZeroU64::new(uniform_size).expect("uniform size is non-zero")),
            }),
        }];
        for (i, id) in bindings.images.iter().enumerate() {
            let t = self.texture(*id);
            entries.push(wgpu::BindGroupEntry {
                binding: 1 + i as u32 * 2,
                resource: wgpu::BindingResource::TextureView(&t.view),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 2 + i as u32 * 2,
                resource: wgpu::BindingResource::Sampler(&t.sampler),
            });
        }
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &shader.layout,
            entries: &entries,
        });
        self.bind_groups.borrow_mut().insert(key, group.clone());
        *self.last_bind_group.borrow_mut() = Some(CachedBindGroup {
            shader: shader_id,
            images: bindings.images.clone(),
            group: group.clone(),
        });
        group
    }
}
impl RenderingBackend for WgpuContext {
    fn info(&self) -> ContextInfo {
        ContextInfo {
            backend: Backend::Wgpu,
            gl_version_string: String::new(),
            glsl_support: GlslSupport {
                v100: true,
                ..Default::default()
            },
            features: Features::default(),
        }
    }
    fn compressed_texture_support(&self) -> CompressedTextureSupport {
        self.compressed_texture_support
    }
    fn validate_compressed_texture_params(
        &self,
        params: &CompressedTextureParams,
    ) -> Result<(), TextureError> {
        let (block_width, block_height) = params.format.block_extent();
        if params.width % block_width != 0 || params.height % block_height != 0 {
            return Err(TextureError::UnsupportedDimensions {
                format: params.format,
                width: params.width,
                height: params.height,
            });
        }
        Ok(())
    }
    fn new_shader(
        &mut self,
        source: ShaderSource,
        meta: ShaderMeta,
    ) -> Result<ShaderId, ShaderError> {
        Ok(ShaderId(self.shaders.add(shader::compile(
            &self.device,
            source,
            meta,
        )?)))
    }
    fn new_texture(
        &mut self,
        access: TextureAccess,
        data: TextureSource,
        params: TextureParams,
    ) -> TextureId {
        let texture = Texture::new(&self.device, params, access);
        let id = TextureId(TextureIdInner::Managed(self.textures.add(texture)));
        match data {
            TextureSource::Empty => {}
            TextureSource::Bytes(bytes) => self.texture_update(id, bytes),
            TextureSource::Array(faces) => {
                for (face, mips) in faces.iter().enumerate() {
                    for (mip, bytes) in mips.iter().enumerate() {
                        self.texture(id).upload(
                            &self.queue,
                            face as _,
                            mip as _,
                            0,
                            0,
                            (params.width >> mip).max(1),
                            (params.height >> mip).max(1),
                            bytes,
                        );
                    }
                }
            }
        }
        id
    }
    fn new_compressed_texture(
        &mut self,
        access: TextureAccess,
        source: CompressedTextureSource,
        params: CompressedTextureParams,
    ) -> TextureId {
        let texture = Texture::new_compressed(&self.device, &self.queue, access, source, params);
        TextureId(TextureIdInner::Managed(self.textures.add(texture)))
    }
    fn texture_params(&self, id: TextureId) -> TextureParams {
        self.texture(id).params
    }
    unsafe fn texture_raw_id(&self, _: TextureId) -> RawId {
        panic!("wgpu textures have no OpenGL or Metal raw id")
    }
    fn texture_set_min_filter(&mut self, id: TextureId, filter: FilterMode, mip: MipmapFilterMode) {
        let device = self.device.clone();
        let t = self.texture_mut(id);
        t.params.min_filter = filter;
        t.params.mipmap_filter = mip;
        t.update_sampler(&device);
    }
    fn texture_set_mag_filter(&mut self, id: TextureId, filter: FilterMode) {
        let device = self.device.clone();
        let t = self.texture_mut(id);
        t.params.mag_filter = filter;
        t.update_sampler(&device);
    }
    fn texture_set_wrap(&mut self, id: TextureId, x: TextureWrap, y: TextureWrap) {
        let device = self.device.clone();
        let t = self.texture_mut(id);
        t.params.wrap = x;
        t.wrap_y = y;
        t.update_sampler(&device);
    }
    fn texture_generate_mipmaps(&mut self, id: TextureId) {
        self.texture(id).generate_mipmaps(self);
    }
    fn texture_resize(&mut self, id: TextureId, width: u32, height: u32, bytes: Option<&[u8]>) {
        self.submit();
        let invalidates_target = self.target_pass.is_some_and(|pass| {
            pass.is_some_and(|pass| {
                let pass = &self.passes[pass.0];
                pass.colors.contains(&id) || pass.resolves.contains(&id) || pass.depth == Some(id)
            })
        });
        if invalidates_target {
            assert!(!self.pass_active, "cannot resize an active render target");
            self.target = None;
            self.target_pass = None;
        }
        let old = self.texture(id);
        let mut params = old.params;
        params.width = width;
        params.height = height;
        let t = Texture::new(&self.device, params, old.access);
        *self.texture_mut(id) = t;
        if let Some(bytes) = bytes {
            self.texture_update(id, bytes);
        }
    }
    fn texture_read_pixels(&mut self, id: TextureId, bytes: &mut [u8]) {
        self.submit();
        self.texture(id).read(&self.device, &self.queue, bytes);
    }
    fn texture_update_part(&mut self, id: TextureId, x: i32, y: i32, w: i32, h: i32, bytes: &[u8]) {
        assert!(x >= 0 && y >= 0 && w >= 0 && h >= 0);
        self.submit();
        self.texture(id)
            .upload(&self.queue, 0, 0, x as _, y as _, w as _, h as _, bytes);
    }
    fn new_render_pass_mrt(
        &mut self,
        colors: &[TextureId],
        resolves: Option<&[TextureId]>,
        depth: Option<TextureId>,
    ) -> RenderPass {
        assert!(
            !colors.is_empty() || depth.is_some(),
            "render pass needs an attachment"
        );
        if let Some(resolves) = resolves {
            assert_eq!(resolves.len(), colors.len());
        }
        RenderPass(self.passes.add(Pass {
            colors: colors.to_vec(),
            resolves: resolves.unwrap_or(&[]).to_vec(),
            depth,
        }))
    }
    fn render_pass_color_attachments(&self, id: RenderPass) -> &[TextureId] {
        &self.passes[id.0].colors
    }
    fn delete_render_pass(&mut self, id: RenderPass) {
        if self.target_pass == Some(Some(id)) {
            self.flush_draws();
            self.target = None;
            self.target_pass = None;
        }
        self.passes.remove(id.0);
    }
    fn new_pipeline(
        &mut self,
        layouts: &[BufferLayout],
        attributes: &[VertexAttribute],
        shader: ShaderId,
        params: PipelineParams,
    ) -> Pipeline {
        Pipeline(self.pipelines.add(PipelineState::new(
            layouts,
            attributes,
            shader,
            params,
            &self.shaders[shader.0],
        )))
    }
    fn apply_pipeline(&mut self, id: &Pipeline) {
        self.current_pipeline = Some(*id);
    }
    fn delete_pipeline(&mut self, id: Pipeline) {
        self.pipelines.remove(id.0);
    }
    fn new_buffer(&mut self, kind: BufferType, usage: BufferUsage, data: BufferSource) -> BufferId {
        let (bytes, element_size) = buffer_bytes(data);
        let gpu = make_buffer(&self.device, kind, element_size, &bytes);
        BufferId(self.buffers.add(Buffer {
            gpu,
            gpu_offset: 0,
            bytes,
            element_size,
            kind,
            usage,
        }))
    }
    fn buffer_update(&mut self, id: BufferId, data: BufferSource) {
        // Match the other backends: an empty source reserves capacity only and
        // is not a valid update. This also avoids materialising a zero-filled
        // Vec merely to reject an operation that OpenGL/Metal reject.
        let data = match data {
            BufferSource::Slice(data) => data,
            BufferSource::Empty { .. } => panic!("buffer_update expects BufferSource::slice"),
        };
        let bytes = unsafe { std::slice::from_raw_parts(data.ptr as *const u8, data.size) };
        let element_size = data.element_size;
        let device = self.device.clone();
        let stream = self.buffers[id.0].usage == BufferUsage::Stream;
        if !stream {
            // Dynamic buffers may be updated once and reused indefinitely.
            // Keep their dedicated storage and preserve immediate ordering.
            self.flush_draws();
        }
        let b = &mut self.buffers[id.0];
        assert!(
            bytes.len() <= b.bytes.len(),
            "buffer update exceeds capacity"
        );
        assert!(b.kind != BufferType::IndexBuffer || element_size == b.element_size);
        b.bytes[..bytes.len()].copy_from_slice(bytes);
        // Upload only the slice supplied by the caller. `b.bytes` is the
        // backing allocation/capacity, not the number of elements used by
        // this draw. Uploading it here turns a 26 MiB instance buffer with a
        // few visible notes into a 26 MiB staging allocation every frame.
        // Those allocations remain in flight until Metal retires the command
        // buffer and can easily dominate the post-gameplay footprint.
        if !bytes.is_empty() {
            let upload = buffer_upload_bytes(b.kind, b.element_size, &bytes);
            // wgpu copy commands require the size to be a multiple of 4,
            // while miniquad's slice may contain an odd number of u16
            // indices. Pad only the staging copy; the destination buffer and
            // the logical miniquad byte length remain unchanged.
            let upload = if upload.len().is_multiple_of(4) {
                upload
            } else {
                let aligned_len = (upload.len() + 3) & !3;
                let mut padded = upload.into_owned();
                padded.resize(aligned_len, 0);
                Cow::Owned(padded)
            };
            let mut encoder = self.encoder.borrow_mut();
            let encoder = encoder.get_or_insert_with(|| {
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("miniquad frame"),
                })
            });
            let mut staging_belt = self.staging_belt.borrow_mut();
            if !stream {
                b.gpu_offset = 0;
                let mut staging = staging_belt.write_buffer(
                    encoder,
                    &b.gpu,
                    0,
                    NonZeroU64::new(upload.len() as u64).expect("buffer upload is non-zero"),
                );
                staging.copy_from_slice(&upload);
                return;
            }

            // Stream data is replaced before each use. Give every update a
            // distinct frame-arena range so deferred draws retain a snapshot
            // without forcing a render-pass break before the next update.
            const INITIAL_GEOMETRY_CAPACITY: u64 = 4 * 1024 * 1024;
            let offset = self
                .geometry_buffer
                .as_ref()
                .map_or(0, |arena| align_up(arena.offset, 4));
            let required = offset + upload.len() as u64;
            if self
                .geometry_buffer
                .as_ref()
                .is_none_or(|arena| required > arena.capacity)
            {
                let capacity = required.max(INITIAL_GEOMETRY_CAPACITY).next_power_of_two();
                self.geometry_buffer = Some(GeometryBufferArena {
                    gpu: self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("miniquad frame geometry"),
                        size: capacity,
                        usage: wgpu::BufferUsages::VERTEX
                            | wgpu::BufferUsages::INDEX
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    capacity,
                    offset: 0,
                });
            }
            let arena = self
                .geometry_buffer
                .as_mut()
                .expect("geometry buffer arena initialized");
            let offset = align_up(arena.offset, 4);
            arena.offset = offset + upload.len() as u64;
            b.gpu = arena.gpu.clone();
            b.gpu_offset = offset;
            let mut staging = staging_belt.write_buffer(
                encoder,
                &arena.gpu,
                offset,
                NonZeroU64::new(upload.len() as u64).expect("buffer upload is non-zero"),
            );
            staging.copy_from_slice(&upload);
        }
    }
    fn buffer_size(&mut self, id: BufferId) -> usize {
        self.buffers[id.0].bytes.len()
    }
    fn delete_buffer(&mut self, id: BufferId) {
        self.buffers.remove(id.0);
    }
    fn delete_texture(&mut self, id: TextureId) {
        if let TextureIdInner::Managed(id) = id.0 {
            let texture_id = TextureId(TextureIdInner::Managed(id));
            let invalidates_target = self.target_pass.is_some_and(|pass| {
                pass.is_some_and(|pass| {
                    let pass = &self.passes[pass.0];
                    pass.colors.contains(&texture_id)
                        || pass.resolves.contains(&texture_id)
                        || pass.depth == Some(texture_id)
                })
            });
            if invalidates_target {
                self.flush_draws();
                self.target = None;
                self.target_pass = None;
            }
            self.bind_groups.borrow_mut().clear();
            self.last_bind_group.get_mut().take();
            // Deletion can happen immediately after commit_frame while higher
            // layers still have a cached raw TextureId to flush. Resource ids
            // are monotonic, so retaining the old value cannot alias a new
            // texture. The next successful commit is the lifetime boundary.
            if self.textures.get(id).is_some() {
                let texture = self.textures.remove(id);
                self.retired_textures.insert(id, texture);
            }
        }
    }
    fn delete_shader(&mut self, id: ShaderId) {
        self.bind_groups.borrow_mut().clear();
        self.last_bind_group.get_mut().take();
        self.shaders.remove(id.0);
    }
    fn apply_viewport(&mut self, x: i32, y: i32, w: i32, h: i32) {
        let Some(t) = &self.target else {
            return;
        };
        assert!(x >= 0 && y >= 0 && w >= 0 && h >= 0);
        self.viewport = Some((
            x as f32,
            (t.height as i32 - y - h) as f32,
            w as f32,
            h as f32,
        ));
    }
    fn apply_scissor_rect(&mut self, x: i32, y: i32, w: i32, h: i32) {
        let Some(t) = &self.target else {
            return;
        };
        let left = x.max(0).min(t.width as i32);
        let right = x.saturating_add(w).max(left).min(t.width as i32);
        let bottom = y.max(0).min(t.height as i32);
        let top = y.saturating_add(h).max(bottom).min(t.height as i32);
        self.scissor = Some((
            left as _,
            t.height - top as u32,
            (right - left) as _,
            (top - bottom) as _,
        ));
    }
    fn apply_bindings_from_slice(&mut self, v: &[BufferId], i: BufferId, t: &[TextureId]) {
        // Unlike Metal's encoder API, wgpu needs us to retain this state until
        // draw. Reuse the small vectors instead of allocating on every batch.
        if let Some(bindings) = &mut self.bindings {
            bindings.vertex_buffers.clear();
            bindings.vertex_buffers.extend_from_slice(v);
            bindings.index_buffer = i;
            bindings.images.clear();
            bindings.images.extend_from_slice(t);
        } else {
            self.bindings = Some(Bindings {
                vertex_buffers: v.to_vec(),
                index_buffer: i,
                images: t.to_vec(),
            });
        }
    }
    fn apply_uniforms_from_bytes(&mut self, ptr: *const u8, size: usize) {
        self.uniforms.clear();
        if size != 0 {
            // SAFETY: RenderingBackend consumes uniform input synchronously.
            self.uniforms
                .extend_from_slice(unsafe { std::slice::from_raw_parts(ptr, size) });
        }
    }
    fn clear(
        &mut self,
        color: Option<(f32, f32, f32, f32)>,
        depth: Option<f32>,
        stencil: Option<i32>,
    ) {
        self.flush_draws();
        self.render(
            &PassAction::Clear {
                color,
                depth,
                stencil,
            },
            |_| {},
        );
    }
    fn begin_default_pass(&mut self, action: PassAction) {
        self.begin_pass(None, action);
    }
    fn begin_pass(&mut self, id: Option<RenderPass>, action: PassAction) {
        assert!(!self.pass_active, "end the previous render pass first");
        // A miniquad pass boundary with Load/Store does not need to become a
        // native wgpu pass boundary. Keep adjacent draws for the same target
        // together; this is particularly important for macroquad, which opens
        // one logical pass per DrawCall.
        if self.target_pass != Some(id) || self.target.is_none() {
            self.flush_draws();
            self.target = if let Some(id) = id {
                let p = &self.passes[id.0];
                let first = self.texture(p.colors.first().copied().or(p.depth).unwrap());
                Some(Target {
                    colors: p
                        .colors
                        .iter()
                        .map(|id| self.texture(*id).view.clone())
                        .collect(),
                    resolves: p
                        .resolves
                        .iter()
                        .map(|id| self.texture(*id).view.clone())
                        .collect(),
                    depth: p.depth.map(|id| self.texture(id).view.clone()),
                    key: TargetKey {
                        colors: p
                            .colors
                            .iter()
                            .map(|id| self.texture(*id).gpu.format())
                            .collect(),
                        depth: p.depth.map(|id| self.texture(id).gpu.format()),
                        samples: first.params.sample_count.max(1) as _,
                    },
                    width: first.params.width,
                    height: first.params.height,
                })
            } else {
                self.default_target()
            };
            self.target_pass = self.target.as_ref().map(|_| id);
        }
        self.viewport = None;
        self.scissor = None;
        if matches!(action, PassAction::Clear { .. }) {
            self.flush_draws();
            self.render(&action, |_| {});
        }
        self.pass_active = true;
    }
    fn end_render_pass(&mut self) {
        assert!(self.pass_active, "end_render_pass without begin_pass");
        self.pass_active = false;
    }
    fn commit_frame(&mut self) {
        assert!(!self.pass_active, "end render pass before commit");
        self.submit();
        self.target = None;
        self.target_pass = None;
        let mut state = self.surface.borrow_mut();
        if let Some(frame) = state.frame.take() {
            if let Some(window) = &state.window {
                window.pre_present_notify();
            }
            frame.present();
        }
        self.retired_textures.clear();
        if let Some(arena) = self.uniform_buffer.borrow_mut().as_mut() {
            // Queue writes are ordered after the previous submit, so the
            // same backing allocation can be reused by the next frame.
            arena.offset = 0;
        }
        if let Some(arena) = self.geometry_buffer.as_mut() {
            // Queue submissions execute in order, so reusing the same ranges
            // next frame cannot overtake draws from the preceding frame.
            arena.offset = 0;
        }
    }
    fn draw(&self, base: i32, count: i32, instances: i32) {
        assert!(base >= 0 && count >= 0 && instances >= 0);
        if count == 0 || instances == 0 || self.target.is_none() {
            return;
        }
        if self.scissor.is_some_and(|(_, _, w, h)| w == 0 || h == 0) {
            return;
        }
        let p = &self.pipelines[self
            .current_pipeline
            .expect("apply a pipeline before draw")
            .0];
        let shader = &self.shaders[p.shader.0];
        let bindings = self.bindings.as_ref().expect("apply bindings before draw");
        assert_eq!(bindings.images.len(), shader.images);
        let mut packed_uniform = self.packed_uniforms.borrow_mut();
        shader
            .uniforms
            .pack_into(&self.uniforms, &mut packed_uniform);
        let (uniform, uniform_offset, uniform_size) = self.allocate_uniform(&packed_uniform);
        drop(packed_uniform);
        let group = self.bind_group(p.shader.0, shader, bindings, &uniform, uniform_size);
        let target = self.target.as_ref().unwrap();
        let pipeline = p.get(&self.device, shader, &target.key);
        let mut vertices = self
            .vertex_buffer_pool
            .borrow_mut()
            .pop()
            .unwrap_or_default();
        p.vertex_buffers(
            &self.device,
            &self.buffers,
            &bindings.vertex_buffers,
            &mut vertices,
        );
        let index = &self.buffers[bindings.index_buffer.0];
        assert!((base as usize + count as usize) * index.element_size <= index.bytes.len());
        assert!(uniform_offset <= u64::from(u32::MAX));
        self.pending_draws.borrow_mut().push(DrawCommand {
            pipeline,
            bind_group: group,
            uniform_offset: uniform_offset as u32,
            vertices,
            index: index.gpu.clone(),
            index_offset: index.gpu_offset,
            index_format: if index.element_size == 4 {
                wgpu::IndexFormat::Uint32
            } else {
                wgpu::IndexFormat::Uint16
            },
            base: base as u32,
            count: count as u32,
            instances: instances as u32,
            viewport: self.viewport,
            scissor: self.scissor,
            stencil_reference: p
                .params
                .stencil_test
                .map(|stencil| stencil.front.test_ref as u32),
        });
    }
}

fn align_up(value: u64, alignment: u64) -> u64 {
    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value + alignment - remainder
    }
}

fn buffer_bytes(source: BufferSource) -> (Vec<u8>, usize) {
    match source {
        BufferSource::Empty { size, element_size } => (vec![0; size], element_size),
        BufferSource::Slice(a) => (
            if a.size == 0 {
                vec![]
            } else {
                unsafe { std::slice::from_raw_parts(a.ptr as *const u8, a.size) }.to_vec()
            },
            a.element_size,
        ),
    }
}

fn make_buffer(
    device: &wgpu::Device,
    kind: BufferType,
    element_size: usize,
    bytes: &[u8],
) -> wgpu::Buffer {
    let converted = buffer_upload_bytes(kind, element_size, bytes);
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("miniquad geometry"),
        contents: if converted.is_empty() {
            &[0; 4]
        } else {
            &converted
        },
        usage: if kind == BufferType::IndexBuffer {
            wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST
        } else {
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST
        },
    })
}

fn buffer_upload_bytes<'a>(
    kind: BufferType,
    element_size: usize,
    bytes: &'a [u8],
) -> Cow<'a, [u8]> {
    if kind == BufferType::IndexBuffer && element_size == 1 {
        Cow::Owned(
            bytes
                .iter()
                .flat_map(|byte| (*byte as u16).to_ne_bytes())
                .collect(),
        )
    } else {
        Cow::Borrowed(bytes)
    }
}

#[cfg(test)]
mod tests;
