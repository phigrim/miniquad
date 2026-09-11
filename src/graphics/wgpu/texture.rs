use super::*;
pub(super) struct Texture {
    pub gpu: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub params: TextureParams,
    pub wrap_y: TextureWrap,
    pub access: TextureAccess,
}
impl Texture {
    pub fn new(device: &wgpu::Device, params: TextureParams, access: TextureAccess) -> Self {
        assert!(params.width > 0 && params.height > 0);
        let format = format(params.format);
        let samples = params.sample_count.max(1) as u32;
        let levels = if params.allocate_mipmaps {
            32 - params.width.max(params.height).leading_zeros()
        } else {
            1
        };
        // Static textures are sampled/uploaded only. Marking every texture as
        // a render attachment makes Metal place ordinary UI images in the
        // renderable resource pool and inflates its graphics footprint.
        let mut usage = wgpu::TextureUsages::TEXTURE_BINDING;
        if access == TextureAccess::RenderTarget {
            usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
        }
        if samples == 1 {
            usage |= wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST;
        }
        let gpu = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("miniquad texture"),
            size: wgpu::Extent3d {
                width: params.width,
                height: params.height,
                depth_or_array_layers: if params.kind == TextureKind::CubeMap {
                    6
                } else {
                    1
                },
            },
            mip_level_count: levels,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        let view = gpu.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(if params.kind == TextureKind::CubeMap {
                wgpu::TextureViewDimension::Cube
            } else {
                wgpu::TextureViewDimension::D2
            }),
            ..Default::default()
        });
        let sampler = sampler(device, &params, params.wrap);
        Self {
            gpu,
            view,
            sampler,
            params,
            wrap_y: params.wrap,
            access,
        }
    }
    pub fn new_compressed(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        access: TextureAccess,
        source: CompressedTextureSource,
        params: CompressedTextureParams,
    ) -> Self {
        let format = compressed_format(params.format).expect("unsupported wgpu compressed format");
        let levels = match &source {
            CompressedTextureSource::Mipmaps(levels) => levels.len(),
            CompressedTextureSource::CubeMap(faces) => faces[0].len(),
        } as u32;
        let gpu = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("miniquad compressed texture"),
            size: wgpu::Extent3d {
                width: params.width,
                height: params.height,
                depth_or_array_layers: if params.kind == TextureKind::CubeMap {
                    6
                } else {
                    1
                },
            },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = gpu.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(if params.kind == TextureKind::CubeMap {
                wgpu::TextureViewDimension::Cube
            } else {
                wgpu::TextureViewDimension::D2
            }),
            ..Default::default()
        });
        let texture_params = TextureParams {
            kind: params.kind,
            format: TextureFormat::RGBA8,
            wrap: params.wrap,
            min_filter: params.min_filter,
            mag_filter: params.mag_filter,
            mipmap_filter: params.mipmap_filter,
            width: params.width,
            height: params.height,
            allocate_mipmaps: levels > 1,
            sample_count: 1,
        };
        let texture = Self {
            gpu,
            view,
            sampler: sampler(device, &texture_params, params.wrap),
            params: texture_params,
            wrap_y: params.wrap,
            access,
        };
        match source {
            CompressedTextureSource::Mipmaps(levels) => {
                for (mip, bytes) in levels.iter().enumerate() {
                    texture.upload_compressed(
                        queue,
                        params.format,
                        0,
                        mip as u32,
                        params.width,
                        params.height,
                        bytes,
                    );
                }
            }
            CompressedTextureSource::CubeMap(faces) => {
                for (face, levels) in faces.iter().enumerate() {
                    for (mip, bytes) in levels.iter().enumerate() {
                        texture.upload_compressed(
                            queue,
                            params.format,
                            face as u32,
                            mip as u32,
                            params.width,
                            params.height,
                            bytes,
                        );
                    }
                }
            }
        }
        texture
    }
    fn upload_compressed(
        &self,
        queue: &wgpu::Queue,
        format: CompressedTextureFormat,
        layer: u32,
        mip: u32,
        base_width: u32,
        base_height: u32,
        bytes: &[u8],
    ) {
        let width = (base_width >> mip).max(1);
        let height = (base_height >> mip).max(1);
        let (block_width, block_height) = format.block_extent();
        let bytes_per_row = width.div_ceil(block_width) * format.bytes_per_block();
        let rows_per_image = height.div_ceil(block_height);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.gpu,
                mip_level: mip,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(rows_per_image),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }
    pub fn update_sampler(&mut self, device: &wgpu::Device) {
        self.sampler = sampler(device, &self.params, self.wrap_y);
    }
    pub fn upload(
        &self,
        queue: &wgpu::Queue,
        layer: u32,
        mip: u32,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        bytes: &[u8],
    ) {
        assert_eq!(bytes.len(), self.params.format.size(w, h) as usize);
        let converted: Vec<u8>;
        let data = match self.params.format {
            TextureFormat::RGB8 => {
                converted = bytes
                    .chunks_exact(3)
                    .flat_map(|p| [p[0], p[1], p[2], 255])
                    .collect();
                &converted
            }
            TextureFormat::Alpha => {
                converted = bytes.iter().flat_map(|&a| [255, 255, 255, a]).collect();
                &converted
            }
            _ => bytes,
        };
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.gpu,
                mip_level: mip,
                origin: wgpu::Origin3d { x, y, z: layer },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * pixel_size(self.params.format)),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }
    pub fn read(&self, device: &wgpu::Device, queue: &wgpu::Queue, out: &mut [u8]) {
        let p = self.params;
        assert_eq!(out.len(), p.format.size(p.width, p.height) as usize);
        let row = p.width * pixel_size(p.format);
        let padded = shader::align_up(row as usize, 256) as u32;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("miniquad readback"),
            size: padded as u64 * p.height as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            self.gpu.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(p.height),
                },
            },
            wgpu::Extent3d {
                width: p.width,
                height: p.height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("wgpu readback poll");
        rx.recv().unwrap().expect("map readback");
        {
            let mapped = buffer.slice(..).get_mapped_range();
            let output_row = p.format.size(p.width, 1) as usize;
            for (src, dst) in mapped
                .chunks(padded as usize)
                .zip(out.chunks_mut(output_row))
            {
                match p.format {
                    TextureFormat::RGB8 => {
                        for (p, d) in src[..row as usize]
                            .chunks_exact(4)
                            .zip(dst.chunks_exact_mut(3))
                        {
                            d.copy_from_slice(&p[..3]);
                        }
                    }
                    TextureFormat::Alpha => {
                        for (p, d) in src.chunks_exact(4).zip(dst) {
                            *d = p[3];
                        }
                    }
                    _ => dst.copy_from_slice(&src[..row as usize]),
                }
            }
        }
        buffer.unmap();
    }
    pub fn generate_mipmaps(&self, ctx: &WgpuContext) {
        if self.gpu.mip_level_count() == 1 {
            return;
        }
        let source="@group(0) @binding(0) var tex: texture_2d<f32>; @group(0) @binding(1) var smp: sampler; struct V { @builtin(position) p: vec4<f32>, @location(0) uv: vec2<f32> }; @vertex fn vs(@builtin(vertex_index) i:u32)->V { var p=array<vec2<f32>,3>(vec2(-1.,-1.),vec2(3.,-1.),vec2(-1.,3.)); var v:V; v.p=vec4(p[i],0.,1.); v.uv=vec2((p[i].x+1.)*0.5,(1.-p[i].y)*0.5); return v; } @fragment fn fs(v:V)->@location(0) vec4<f32> { return textureSample(tex,smp,v.uv); }";
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("miniquad mipmaps"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: None,
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.gpu.format(),
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });
        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        ctx.encode(|encoder| {
            for layer in 0..self.gpu.depth_or_array_layers() {
                for level in 1..self.gpu.mip_level_count() {
                    let view = |mip| {
                        self.gpu.create_view(&wgpu::TextureViewDescriptor {
                            dimension: Some(wgpu::TextureViewDimension::D2),
                            base_mip_level: mip,
                            mip_level_count: Some(1),
                            base_array_layer: layer,
                            array_layer_count: Some(1),
                            ..Default::default()
                        })
                    };
                    let src = view(level - 1);
                    let dst = view(level);
                    let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &pipeline.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&sampler),
                            },
                        ],
                    });
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &dst,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        ..Default::default()
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.draw(0..3, 0..1);
                }
            }
        });
    }
}
fn sampler(device: &wgpu::Device, p: &TextureParams, y: TextureWrap) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wrap(p.wrap),
        address_mode_v: wrap(y),
        address_mode_w: wrap(p.wrap),
        mag_filter: filter(p.mag_filter),
        min_filter: filter(p.min_filter),
        mipmap_filter: if p.mipmap_filter == MipmapFilterMode::Linear {
            wgpu::MipmapFilterMode::Linear
        } else {
            wgpu::MipmapFilterMode::Nearest
        },
        lod_max_clamp: if p.mipmap_filter == MipmapFilterMode::None {
            0.
        } else {
            32.
        },
        ..Default::default()
    })
}
fn filter(f: FilterMode) -> wgpu::FilterMode {
    if f == FilterMode::Linear {
        wgpu::FilterMode::Linear
    } else {
        wgpu::FilterMode::Nearest
    }
}
fn wrap(w: TextureWrap) -> wgpu::AddressMode {
    match w {
        TextureWrap::Clamp => wgpu::AddressMode::ClampToEdge,
        TextureWrap::Repeat => wgpu::AddressMode::Repeat,
        TextureWrap::Mirror => wgpu::AddressMode::MirrorRepeat,
    }
}
fn format(f: TextureFormat) -> wgpu::TextureFormat {
    match f {
        TextureFormat::RGB8 | TextureFormat::RGBA8 | TextureFormat::Alpha => {
            wgpu::TextureFormat::Rgba8Unorm
        }
        TextureFormat::RGBA16F => wgpu::TextureFormat::Rgba16Float,
        TextureFormat::Depth => wgpu::TextureFormat::Depth16Unorm,
        TextureFormat::Depth32 => wgpu::TextureFormat::Depth32Float,
    }
}
fn pixel_size(f: TextureFormat) -> u32 {
    match f {
        TextureFormat::Depth => 2,
        TextureFormat::RGBA16F => 8,
        _ => 4,
    }
}
pub(super) fn compressed_format(format: CompressedTextureFormat) -> Option<wgpu::TextureFormat> {
    use wgpu::TextureFormat as W;
    use CompressedTextureFormat::*;
    Some(match format {
        Bc1Rgb | Bc1Rgba => W::Bc1RgbaUnorm,
        Bc2 => W::Bc2RgbaUnorm,
        Bc3 => W::Bc3RgbaUnorm,
        Bc4 => W::Bc4RUnorm,
        Bc5 => W::Bc5RgUnorm,
        Bc6hUnsigned => W::Bc6hRgbUfloat,
        Bc6hSigned => W::Bc6hRgbFloat,
        Bc7 => W::Bc7RgbaUnorm,
        Etc2Rgb8 => W::Etc2Rgb8Unorm,
        Etc2Rgb8A1 => W::Etc2Rgb8A1Unorm,
        Etc2Rgba8 => W::Etc2Rgba8Unorm,
        EacR11 => W::EacR11Unorm,
        EacRg11 => W::EacRg11Unorm,
        Astc {
            block_width,
            block_height,
        } => W::Astc {
            block: astc_block(block_width, block_height)?,
            channel: wgpu::AstcChannel::Unorm,
        },
        PvrtcRgb2 | PvrtcRgb4 | PvrtcRgba2 | PvrtcRgba4 => return None,
    })
}
fn astc_block(width: u8, height: u8) -> Option<wgpu::AstcBlock> {
    use wgpu::AstcBlock as B;
    Some(match (width, height) {
        (4, 4) => B::B4x4,
        (5, 4) => B::B5x4,
        (5, 5) => B::B5x5,
        (6, 5) => B::B6x5,
        (6, 6) => B::B6x6,
        (8, 5) => B::B8x5,
        (8, 6) => B::B8x6,
        (8, 8) => B::B8x8,
        (10, 5) => B::B10x5,
        (10, 6) => B::B10x6,
        (10, 8) => B::B10x8,
        (10, 10) => B::B10x10,
        (12, 10) => B::B12x10,
        (12, 12) => B::B12x12,
        _ => return None,
    })
}
pub(super) fn attachment(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    samples: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("miniquad framebuffer"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: samples,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}
