//! Portable renderer. Public handles stay backend independent; GPU resources are owned here.
use super::*;
use crate::ResourceManager;
use ::wgpu;
use std::{cell::RefCell, collections::HashMap, sync::Arc};
use wgpu::util::DeviceExt;
mod pipeline;
mod shader;
mod texture;
use pipeline::PipelineState;
use shader::Shader;
use texture::Texture;

struct Buffer {
    gpu: wgpu::Buffer,
    bytes: Vec<u8>,
    element_size: usize,
    kind: BufferType,
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

/// The wgpu implementation of [`RenderingBackend`].
/// Created by `window::new_rendering_backend` when `GfxApi::Wgpu` is selected.
pub struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: Option<wgpu::Surface<'static>>,
    config: Option<wgpu::SurfaceConfiguration>,
    window: Option<Arc<winit::window::Window>>,
    frame: Option<wgpu::SurfaceTexture>,
    default_color: Option<wgpu::Texture>,
    default_depth: Option<wgpu::Texture>,
    samples: u32,
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
    current_pipeline: Option<Pipeline>,
    bindings: Option<Bindings>,
    uniforms: Vec<u8>,
    viewport: Option<(f32, f32, f32, f32)>,
    scissor: Option<(u32, u32, u32, u32)>,
}
impl WgpuContext {
    pub(crate) async fn for_window(
        window: Arc<winit::window::Window>,
        conf: &crate::conf::Conf,
    ) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(window.clone())
            .expect("create wgpu surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .expect("no compatible wgpu adapter");
        let mut context = Self::from_adapter(adapter.clone()).await;
        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface configuration");
        // Existing miniquad shaders produce display values, with no implicit sRGB conversion.
        config.format = surface
            .get_capabilities(&adapter)
            .formats
            .into_iter()
            .find(|f| !f.is_srgb())
            .unwrap_or(config.format);
        config.present_mode = if conf.platform.swap_interval == Some(0) {
            wgpu::PresentMode::AutoNoVsync
        } else {
            wgpu::PresentMode::AutoVsync
        };
        surface.configure(&context.device, &config);
        context.surface = Some(surface);
        context.config = Some(config);
        context.window = Some(window);
        context.samples = conf.sample_count.max(1) as u32;
        context
    }
    async fn from_adapter(adapter: wgpu::Adapter) -> Self {
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("miniquad"),
                ..Default::default()
            })
            .await
            .expect("create wgpu device");
        Self {
            device,
            queue,
            surface: None,
            config: None,
            window: None,
            frame: None,
            default_color: None,
            default_depth: None,
            samples: 1,
            shaders: Default::default(),
            textures: Default::default(),
            retired_textures: HashMap::new(),
            buffers: Default::default(),
            pipelines: Default::default(),
            passes: Default::default(),
            encoder: RefCell::new(None),
            target: None,
            current_pipeline: None,
            bindings: None,
            uniforms: vec![],
            viewport: None,
            scissor: None,
        }
    }
    fn submit(&self) {
        if let Some(encoder) = self.encoder.borrow_mut().take() {
            self.queue.submit([encoder.finish()]);
        }
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
            .window
            .as_ref()
            .expect("headless contexts require an offscreen pass");
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return None;
        }
        let config = self.config.as_mut().unwrap();
        if self.frame.is_none() {
            if config.width != size.width || config.height != size.height {
                self.submit();
                let config = self.config.as_mut().unwrap();
                config.width = size.width;
                config.height = size.height;
                self.surface
                    .as_ref()
                    .unwrap()
                    .configure(&self.device, config);
                self.default_color = None;
                self.default_depth = None;
            }

            let surface = self.surface.as_ref().unwrap();
            self.frame = match surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t)
                | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Some(t),
                wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                    surface.configure(&self.device, self.config.as_ref().unwrap());
                    None
                }
                wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                    None
                }
                wgpu::CurrentSurfaceTexture::Validation => panic!("wgpu surface validation failed"),
            };
        }
        let config = self.config.as_ref().unwrap();
        let frame = self.frame.as_ref()?;
        let mut color = frame.texture.create_view(&Default::default());
        let mut resolves = vec![];
        let device = &self.device;
        let samples = self.samples;
        if self.samples > 1 {
            let texture = self.default_color.get_or_insert_with(|| {
                texture::attachment(device, size.width, size.height, config.format, samples)
            });
            resolves.push(color);
            color = texture.create_view(&Default::default());
        }
        let depth = self.default_depth.get_or_insert_with(|| {
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
                colors: vec![config.format],
                depth: Some(wgpu::TextureFormat::Depth24PlusStencil8),
                samples: self.samples,
            },
            width: size.width,
            height: size.height,
        })
    }
    fn render(&self, action: &PassAction, f: impl FnOnce(&mut wgpu::RenderPass<'_>)) {
        let Some(target) = &self.target else {
            return;
        };
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
    fn new_buffer(&mut self, kind: BufferType, _: BufferUsage, data: BufferSource) -> BufferId {
        let (bytes, element_size) = buffer_bytes(data);
        let gpu = make_buffer(&self.device, kind, element_size, &bytes);
        BufferId(self.buffers.add(Buffer {
            gpu,
            bytes,
            element_size,
            kind,
        }))
    }
    fn buffer_update(&mut self, id: BufferId, data: BufferSource) {
        let (bytes, element_size) = buffer_bytes(data);
        let b = &mut self.buffers[id.0];
        assert!(
            bytes.len() <= b.bytes.len(),
            "buffer update exceeds capacity"
        );
        assert!(b.kind != BufferType::IndexBuffer || element_size == b.element_size);
        b.bytes[..bytes.len()].copy_from_slice(&bytes);
        b.gpu = make_buffer(&self.device, b.kind, b.element_size, &b.bytes);
    }
    fn buffer_size(&mut self, id: BufferId) -> usize {
        self.buffers[id.0].bytes.len()
    }
    fn delete_buffer(&mut self, id: BufferId) {
        self.buffers.remove(id.0);
    }
    fn delete_texture(&mut self, id: TextureId) {
        if let TextureIdInner::Managed(id) = id.0 {
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
        self.bindings = Some(Bindings {
            vertex_buffers: v.to_vec(),
            index_buffer: i,
            images: t.to_vec(),
        });
    }
    fn apply_uniforms_from_bytes(&mut self, ptr: *const u8, size: usize) {
        self.uniforms = if size == 0 {
            vec![]
        } else {
            unsafe { std::slice::from_raw_parts(ptr, size) }.to_vec()
        };
    }
    fn clear(
        &mut self,
        color: Option<(f32, f32, f32, f32)>,
        depth: Option<f32>,
        stencil: Option<i32>,
    ) {
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
        assert!(self.target.is_none(), "end the previous render pass first");
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
        self.viewport = None;
        self.scissor = None;
        if matches!(action, PassAction::Clear { .. }) {
            self.render(&action, |_| {});
        }
    }
    fn end_render_pass(&mut self) {
        self.target = None;
    }
    fn commit_frame(&mut self) {
        assert!(self.target.is_none(), "end render pass before commit");
        self.submit();
        if let Some(frame) = self.frame.take() {
            if let Some(window) = &self.window {
                window.pre_present_notify();
            }
            frame.present();
        }
        self.retired_textures.clear();
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
        let uniform = shader.uniforms.pack(&self.uniforms);
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("miniquad draw uniforms"),
                contents: &uniform,
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
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
        let target = self.target.as_ref().unwrap();
        let pipeline = p.get(&self.device, shader, &target.key);
        let vertices = p.vertex_buffers(&self.device, &self.buffers, &bindings.vertex_buffers);
        let index = &self.buffers[bindings.index_buffer.0];
        assert!((base as usize + count as usize) * index.element_size <= index.bytes.len());
        self.render(&PassAction::Nothing, |pass| {
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            for (i, b) in vertices.iter().enumerate() {
                pass.set_vertex_buffer(i as _, b.slice(..));
            }
            pass.set_index_buffer(
                index.gpu.slice(..),
                if index.element_size == 4 {
                    wgpu::IndexFormat::Uint32
                } else {
                    wgpu::IndexFormat::Uint16
                },
            );
            if let Some((x, y, w, h)) = self.viewport {
                pass.set_viewport(x, y, w, h, 0., 1.);
            }
            if let Some((x, y, w, h)) = self.scissor {
                pass.set_scissor_rect(x, y, w, h);
            }
            if let Some(stencil) = p.params.stencil_test {
                pass.set_stencil_reference(stencil.front.test_ref as _);
            }
            pass.draw_indexed(base as u32..(base + count) as u32, 0, 0..instances as u32);
        });
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
    let converted: Vec<u8>;
    let bytes = if kind == BufferType::IndexBuffer && element_size == 1 {
        converted = bytes
            .iter()
            .flat_map(|b| (*b as u16).to_ne_bytes())
            .collect();
        &converted
    } else {
        bytes
    };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("miniquad geometry"),
        contents: if bytes.is_empty() { &[0; 4] } else { bytes },
        usage: if kind == BufferType::IndexBuffer {
            wgpu::BufferUsages::INDEX
        } else {
            wgpu::BufferUsages::VERTEX
        },
    })
}

#[cfg(test)]
mod tests;
