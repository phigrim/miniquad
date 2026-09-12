use super::*;
use std::convert::TryInto;
struct VertexLayout {
    input_stride: usize,
    stride: u64,
    step: wgpu::VertexStepMode,
    rate: usize,
    attributes: Vec<wgpu::VertexAttribute>,
    conversions: Vec<(usize, VertexFormat, bool)>,
}
pub(super) struct PipelineState {
    pub shader: ShaderId,
    pub params: PipelineParams,
    layouts: Vec<VertexLayout>,
    cache: RefCell<HashMap<TargetKey, wgpu::RenderPipeline>>,
}
impl PipelineState {
    pub fn new(
        layouts: &[BufferLayout],
        attributes: &[VertexAttribute],
        shader: ShaderId,
        params: PipelineParams,
        program: &Shader,
    ) -> Self {
        let layouts = layouts
            .iter()
            .enumerate()
            .map(|(buffer, l)| {
                let mut offset = 0;
                let mut input = 0;
                let mut attrs = vec![];
                let mut conversions = vec![];
                for a in attributes.iter().filter(|a| a.buffer_index == buffer) {
                    let size = a.format.size_bytes() as usize;
                    conversions.push((input, a.format, a.gl_pass_as_float));
                    if let Some(&location) = program.attributes.get(a.name) {
                        if a.format == VertexFormat::Mat4 {
                            for column in 0..4 {
                                attrs.push(wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32x4,
                                    offset: offset + column * 16,
                                    shader_location: location + column as u32,
                                });
                            }
                        } else {
                            attrs.push(wgpu::VertexAttribute {
                                format: format(a.format, a.gl_pass_as_float),
                                offset,
                                shader_location: location,
                            });
                        }
                    }
                    offset += a.format.components() as u64 * 4;
                    input += size;
                }
                assert!(
                    l.stride == 0 || l.stride as usize >= input,
                    "vertex stride is smaller than attributes"
                );
                VertexLayout {
                    input_stride: if l.stride == 0 {
                        input
                    } else {
                        l.stride as usize
                    },
                    stride: offset,
                    step: if l.step_func == VertexStep::PerVertex {
                        wgpu::VertexStepMode::Vertex
                    } else {
                        wgpu::VertexStepMode::Instance
                    },
                    rate: l.step_rate.max(1) as usize,
                    attributes: attrs,
                    conversions,
                }
            })
            .collect();
        Self {
            shader,
            params,
            layouts,
            cache: RefCell::new(HashMap::new()),
        }
    }
    pub fn vertex_buffers(
        &self,
        device: &wgpu::Device,
        buffers: &ResourceManager<Buffer>,
        ids: &[BufferId],
        result: &mut Vec<wgpu::Buffer>,
    ) {
        result.clear();
        result.extend(self.layouts.iter().enumerate().map(|(i, l)| {
            let b = &buffers[ids[i].0];
            assert_eq!(b.kind, BufferType::VertexBuffer);
            let needs_conversion = l.conversions.iter().any(|(_, f, _)| {
                !matches!(
                    f,
                    VertexFormat::Float1
                        | VertexFormat::Float2
                        | VertexFormat::Float3
                        | VertexFormat::Float4
                        | VertexFormat::Mat4
                )
            }) || l.input_stride != l.stride as usize
                || l.rate != 1;
            if !needs_conversion {
                return b.gpu.clone();
            }
            let mut data = vec![];
            for v in b.bytes.chunks_exact(l.input_stride) {
                let start = data.len();
                for &(offset, f, as_float) in &l.conversions {
                    let component_size = f.size_bytes() as usize / f.components() as usize;
                    for c in 0..f.components() as usize {
                        let pos = offset + c * component_size;
                        match f {
                            VertexFormat::Float1
                            | VertexFormat::Float2
                            | VertexFormat::Float3
                            | VertexFormat::Float4
                            | VertexFormat::Mat4 => data.extend_from_slice(&v[pos..pos + 4]),
                            _ => {
                                let value = match component_size {
                                    1 => v[pos] as u32,
                                    2 => u16::from_ne_bytes(v[pos..pos + 2].try_into().unwrap())
                                        as u32,
                                    _ => u32::from_ne_bytes(v[pos..pos + 4].try_into().unwrap()),
                                };
                                if as_float {
                                    data.extend_from_slice(&(value as f32).to_ne_bytes());
                                } else {
                                    data.extend_from_slice(&value.to_ne_bytes());
                                }
                            }
                        }
                    }
                }
                let end = data.len();
                if l.step == wgpu::VertexStepMode::Instance {
                    for _ in 1..l.rate {
                        data.extend_from_within(start..end);
                    }
                }
            }
            make_buffer(device, BufferType::VertexBuffer, 1, &data)
        }))
    }
    pub fn get(
        &self,
        device: &wgpu::Device,
        shader: &Shader,
        key: &TargetKey,
    ) -> wgpu::RenderPipeline {
        // Avoid cloning TargetKey on a cache hit. TargetKey owns format Vecs,
        // and this lookup runs for every draw; `entry(key.clone())` therefore
        // caused a heap allocation even after the pipeline was warm.
        if let Some(pipeline) = self.cache.borrow().get(key).cloned() {
            return pipeline;
        }
        self.cache
            .borrow_mut()
            .entry(key.clone())
            .or_insert_with(|| {
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&shader.layout)],
                    immediate_size: 0,
                });
                let buffers: Vec<_> = self
                    .layouts
                    .iter()
                    .map(|l| wgpu::VertexBufferLayout {
                        array_stride: l.stride,
                        step_mode: l.step,
                        attributes: &l.attributes,
                    })
                    .collect();
                let p = self.params;
                let blend = p.color_blend.map(|color| wgpu::BlendState {
                    color: blend_component(color),
                    alpha: blend_component(p.alpha_blend.unwrap_or(color)),
                });
                let (r, g, b, a) = p.color_write;
                let mut write_mask = wgpu::ColorWrites::empty();
                for (enabled, mask) in [
                    (r, wgpu::ColorWrites::RED),
                    (g, wgpu::ColorWrites::GREEN),
                    (b, wgpu::ColorWrites::BLUE),
                    (a, wgpu::ColorWrites::ALPHA),
                ] {
                    if enabled {
                        write_mask |= mask;
                    }
                }
                let targets: Vec<_> = key
                    .colors
                    .iter()
                    .map(|&format| {
                        Some(wgpu::ColorTargetState {
                            format,
                            blend,
                            write_mask,
                        })
                    })
                    .collect();
                let depth_stencil = key.depth.map(|format| wgpu::DepthStencilState {
                    format,
                    depth_write_enabled: Some(p.depth_write),
                    depth_compare: Some(compare(p.depth_test)),
                    stencil: p.stencil_test.map_or(Default::default(), |s| {
                        assert_eq!(
                            s.front.test_ref, s.back.test_ref,
                            "wgpu requires the same front/back stencil reference"
                        );
                        assert_eq!(s.front.test_mask, s.back.test_mask);
                        assert_eq!(s.front.write_mask, s.back.write_mask);
                        wgpu::StencilState {
                            front: stencil_face(s.front),
                            back: stencil_face(s.back),
                            read_mask: s.front.test_mask,
                            write_mask: s.front.write_mask,
                        }
                    }),
                    bias: p
                        .depth_write_offset
                        .map_or(Default::default(), |(factor, units)| wgpu::DepthBiasState {
                            constant: units as i32,
                            slope_scale: factor,
                            clamp: 0.,
                        }),
                });
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("miniquad pipeline"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader.vertex,
                        entry_point: Some(shader.vertex_entry),
                        buffers: &buffers,
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader.fragment,
                        entry_point: Some(shader.fragment_entry),
                        targets: &targets,
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: match p.primitive_type {
                            PrimitiveType::Triangles => wgpu::PrimitiveTopology::TriangleList,
                            PrimitiveType::Lines => wgpu::PrimitiveTopology::LineList,
                            PrimitiveType::Points => wgpu::PrimitiveTopology::PointList,
                        },
                        front_face: if p.front_face_order == FrontFaceOrder::Clockwise {
                            wgpu::FrontFace::Cw
                        } else {
                            wgpu::FrontFace::Ccw
                        },
                        cull_mode: match p.cull_face {
                            CullFace::Nothing => None,
                            CullFace::Front => Some(wgpu::Face::Front),
                            CullFace::Back => Some(wgpu::Face::Back),
                        },
                        ..Default::default()
                    },
                    depth_stencil,
                    multisample: wgpu::MultisampleState {
                        count: key.samples,
                        ..Default::default()
                    },
                    multiview_mask: None,
                    cache: None,
                })
            })
            .clone()
    }
}
fn format(f: VertexFormat, as_float: bool) -> wgpu::VertexFormat {
    use wgpu::VertexFormat as W;
    let float = as_float
        || matches!(
            f,
            VertexFormat::Float1
                | VertexFormat::Float2
                | VertexFormat::Float3
                | VertexFormat::Float4
        );
    match (f.components(), float) {
        (1, true) => W::Float32,
        (2, true) => W::Float32x2,
        (3, true) => W::Float32x3,
        (4, true) => W::Float32x4,
        (1, false) => W::Uint32,
        (2, false) => W::Uint32x2,
        (3, false) => W::Uint32x3,
        (4, false) => W::Uint32x4,
        _ => unreachable!(),
    }
}
fn blend_component(s: BlendState) -> wgpu::BlendComponent {
    wgpu::BlendComponent {
        src_factor: blend_factor(s.sfactor),
        dst_factor: blend_factor(s.dfactor),
        operation: match s.equation {
            Equation::Add => wgpu::BlendOperation::Add,
            Equation::Subtract => wgpu::BlendOperation::Subtract,
            Equation::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
        },
    }
}
fn blend_factor(f: BlendFactor) -> wgpu::BlendFactor {
    use wgpu::BlendFactor as W;
    match f {
        BlendFactor::Zero => W::Zero,
        BlendFactor::One => W::One,
        BlendFactor::SourceAlphaSaturate => W::SrcAlphaSaturated,
        BlendFactor::Value(v) => match v {
            BlendValue::SourceColor => W::Src,
            BlendValue::SourceAlpha => W::SrcAlpha,
            BlendValue::DestinationColor => W::Dst,
            BlendValue::DestinationAlpha => W::DstAlpha,
        },
        BlendFactor::OneMinusValue(v) => match v {
            BlendValue::SourceColor => W::OneMinusSrc,
            BlendValue::SourceAlpha => W::OneMinusSrcAlpha,
            BlendValue::DestinationColor => W::OneMinusDst,
            BlendValue::DestinationAlpha => W::OneMinusDstAlpha,
        },
    }
}
fn compare(c: Comparison) -> wgpu::CompareFunction {
    use wgpu::CompareFunction as W;
    match c {
        Comparison::Always => W::Always,
        Comparison::Never => W::Never,
        Comparison::Less => W::Less,
        Comparison::LessOrEqual => W::LessEqual,
        Comparison::Equal => W::Equal,
        Comparison::NotEqual => W::NotEqual,
        Comparison::Greater => W::Greater,
        Comparison::GreaterOrEqual => W::GreaterEqual,
    }
}
fn stencil_face(s: StencilFaceState) -> wgpu::StencilFaceState {
    wgpu::StencilFaceState {
        compare: match s.test_func {
            CompareFunc::Always => wgpu::CompareFunction::Always,
            CompareFunc::Never => wgpu::CompareFunction::Never,
            CompareFunc::Less => wgpu::CompareFunction::Less,
            CompareFunc::LessOrEqual => wgpu::CompareFunction::LessEqual,
            CompareFunc::Equal => wgpu::CompareFunction::Equal,
            CompareFunc::NotEqual => wgpu::CompareFunction::NotEqual,
            CompareFunc::Greater => wgpu::CompareFunction::Greater,
            CompareFunc::GreaterOrEqual => wgpu::CompareFunction::GreaterEqual,
        },
        fail_op: stencil_op(s.fail_op),
        depth_fail_op: stencil_op(s.depth_fail_op),
        pass_op: stencil_op(s.pass_op),
    }
}
fn stencil_op(s: StencilOp) -> wgpu::StencilOperation {
    use wgpu::StencilOperation as W;
    match s {
        StencilOp::Keep => W::Keep,
        StencilOp::Zero => W::Zero,
        StencilOp::Replace => W::Replace,
        StencilOp::IncrementClamp => W::IncrementClamp,
        StencilOp::DecrementClamp => W::DecrementClamp,
        StencilOp::Invert => W::Invert,
        StencilOp::IncrementWrap => W::IncrementWrap,
        StencilOp::DecrementWrap => W::DecrementWrap,
    }
}
