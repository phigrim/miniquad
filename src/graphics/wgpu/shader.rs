//! GLSL compatibility and the packed miniquad → std140 uniform boundary.
use super::*;
use std::collections::BTreeMap;

pub(super) struct Shader {
    pub vertex: wgpu::ShaderModule,
    pub vertex_entry: &'static str,
    pub fragment_entry: &'static str,
    pub fragment: wgpu::ShaderModule,
    pub layout: wgpu::BindGroupLayout,
    pub attributes: BTreeMap<String, u32>,
    pub uniforms: UniformLayout,
    pub images: usize,
}

pub(super) struct UniformLayout {
    pub size: usize,
    pub packed_size: usize,
    copies: Vec<(usize, usize, usize)>,
}
impl UniformLayout {
    fn push_copy(&mut self, src: usize, dst: usize, len: usize) {
        // Consecutive vec4 arrays (the common large-uniform case) have no
        // std140/WGSL padding between elements. Merge them so packing a
        // 128-element array is one memcpy rather than 128 tiny memcpys.
        if let Some((last_src, last_dst, last_len)) = self.copies.last_mut() {
            if *last_src + *last_len == src && *last_dst + *last_len == dst {
                *last_len += len;
                return;
            }
        }
        self.copies.push((src, dst, len));
    }

    pub fn new(meta: &ShaderMeta) -> Self {
        let mut out = Self {
            size: 0,
            packed_size: 0,
            copies: vec![],
        };
        for u in &meta.uniforms.uniforms {
            let size = u.uniform_type.size();
            let align = if u.array_count > 1 {
                16
            } else {
                match size {
                    4 => 4,
                    8 => 8,
                    _ => 16,
                }
            };
            out.size = align_up(out.size, align);
            let stride = if u.array_count > 1 {
                align_up(size, 16)
            } else {
                size
            };
            for _ in 0..u.array_count {
                out.push_copy(out.packed_size, out.size, size);
                out.packed_size += size;
                out.size += stride;
            }
        }
        out.size = align_up(out.size.max(1), 16);
        out
    }
    /// Pack into caller-owned scratch storage so hot draw paths can reuse the
    /// allocation from the preceding draw.
    pub fn pack_into(&self, input: &[u8], result: &mut Vec<u8>) {
        assert!(
            input.len() >= self.packed_size,
            "uniform block is too small"
        );
        result.clear();
        result.resize(self.size, 0);
        for &(src, dst, len) in &self.copies {
            result[dst..dst + len].copy_from_slice(&input[src..src + len]);
        }
    }
}
pub(super) fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) / alignment * alignment
}

// Tokens, rather than substring substitutions, keep comments and identifier prefixes
// from changing the meaning of user shaders. Preprocessor lines retain their boundaries.
fn tokens(source: &str) -> Vec<String> {
    let chars: Vec<char> = source.chars().collect();
    let mut out = vec![];
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }
        if chars[i] == '/' && chars.get(i + 1) == Some(&'/') {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if chars[i] == '/' && chars.get(i + 1) == Some(&'*') {
            i += 2;
            while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                i += 1;
            }
            i = (i + 2).min(chars.len());
            continue;
        }
        let start = i;
        if chars[i] == '#' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            let line: String = chars[start..i].iter().collect();
            if !line.starts_with("#version") && !line.starts_with("#extension") {
                out.push(format!("\n{}\n", line));
            }
        } else if chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '.' {
            let number_literal = chars[i].is_ascii_digit() || chars[i] == '.';
            i += 1;
            while i < chars.len()
                && (chars[i].is_alphanumeric()
                    || chars[i] == '_'
                    || chars[i] == '.'
                    || (number_literal
                        && matches!(chars[i], '+' | '-')
                        && chars
                            .get(i.wrapping_sub(1))
                            .is_some_and(|previous| matches!(previous, 'e' | 'E'))))
            {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
        } else {
            i += 1;
            if i < chars.len()
                && matches!(
                    (chars[start], chars[i]),
                    ('+', '+')
                        | ('-', '-')
                        | ('+', '=')
                        | ('-', '=')
                        | ('*', '=')
                        | ('/', '=')
                        | ('=', '=')
                        | ('!', '=')
                        | ('<', '=')
                        | ('>', '=')
                        | ('&', '&')
                        | ('|', '|')
                )
            {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
        }
    }
    out
}
fn declarations(tokens: &[String], qualifiers: &[&str]) -> BTreeMap<String, (String, u32)> {
    let mut names = BTreeMap::new();
    for i in 0..tokens.len() {
        if qualifiers.iter().any(|qualifier| tokens[i] == *qualifier) {
            let mut j = i + 1;
            while tokens
                .get(j)
                .is_some_and(|s| matches!(s.as_str(), "lowp" | "mediump" | "highp"))
            {
                j += 1;
            }
            if j + 1 < tokens.len() {
                names.insert(tokens[j + 1].clone(), (tokens[j].clone(), 0));
            }
        }
    }
    let mut location = 0;
    for (ty, loc) in names.values_mut() {
        *loc = location;
        location += if ty == "mat4" { 4 } else { 1 };
    }
    names
}
fn translate(
    source: &[String],
    vertex: bool,
    meta: &ShaderMeta,
    varying: &BTreeMap<String, (String, u32)>,
) -> (String, BTreeMap<String, u32>) {
    let attrs = if vertex {
        declarations(source, &["attribute", "in"])
    } else {
        BTreeMap::new()
    };
    let mut out = String::from("#version 450\n");
    if !meta.uniforms.uniforms.is_empty() {
        out.push_str("layout(set=0,binding=0,std140) uniform MiniquadUniforms {\n");
        for u in &meta.uniforms.uniforms {
            let ty = match u.uniform_type {
                UniformType::Float1 => "float",
                UniformType::Float2 => "vec2",
                UniformType::Float3 => "vec3",
                UniformType::Float4 => "vec4",
                UniformType::Int1 => "int",
                UniformType::Int2 => "ivec2",
                UniformType::Int3 => "ivec3",
                UniformType::Int4 => "ivec4",
                UniformType::Mat4 => "mat4",
            };
            out.push_str(&format!(
                "{} {}{};\n",
                ty,
                u.name,
                if u.array_count > 1 {
                    format!("[{}]", u.array_count)
                } else {
                    String::new()
                }
            ));
        }
        out.push_str("};\n");
    }
    for (i, name) in meta.images.iter().enumerate() {
        let cube = source
            .windows(3)
            .any(|w| w[0] == "uniform" && w[1] == "samplerCube" && &w[2] == name);
        let suffix = if cube { "Cube" } else { "2D" };
        out.push_str(&format!("layout(set=0,binding={}) uniform texture{} mq_tex_{};\nlayout(set=0,binding={}) uniform sampler mq_sampler_{};\n#define {} sampler{}(mq_tex_{},mq_sampler_{})\n",1+i*2,suffix,i,2+i*2,i,name,suffix,i,i));
    }
    if !vertex && source.iter().any(|t| t == "gl_FragColor") {
        out.push_str("layout(location=0) out vec4 mq_frag_color;\n");
    }
    let mut i = 0;
    while i < source.len() {
        let t = &source[i];
        if t == "uniform" || t == "precision" {
            while i < source.len() && source[i] != ";" {
                i += 1;
            }
            i += 1;
            continue;
        }
        if matches!(t.as_str(), "attribute" | "varying" | "in" | "out") {
            let mut j = i + 1;
            while matches!(source[j].as_str(), "lowp" | "mediump" | "highp") {
                j += 1;
            }
            let name = &source[j + 1];
            let (loc, direction) = match t.as_str() {
                "attribute" => (attrs[name].1, "in"),
                "varying" => (varying[name].1, if vertex { "out" } else { "in" }),
                "in" if vertex => (attrs[name].1, "in"),
                "in" => (varying[name].1, "in"),
                // Modern GLSL fragment outputs are the color attachment at
                // location zero. They are not part of the inter-stage varying
                // interface and therefore do not appear in `varying`.
                "out" if !vertex => (0, "out"),
                "out" => (varying[name].1, "out"),
                _ => unreachable!(),
            };
            out.push_str(&format!("layout(location={}) {} ", loc, direction));
            i += 1;
            continue;
        }
        let replacement = match t.as_str() {
            "lowp" | "mediump" | "highp" => "",
            "texture2D" | "textureCube" => "texture",
            "gl_FragColor" => "mq_frag_color",
            "main" if vertex => "mq_main",
            _ => t,
        };
        out.push_str(replacement);
        out.push(' ');
        i += 1;
    }
    if vertex {
        out.push_str(
            "\nvoid main() { mq_main(); gl_Position.z = (gl_Position.z + gl_Position.w) * 0.5; }\n",
        );
    }
    (
        out,
        attrs
            .into_iter()
            .map(|(name, (_, loc))| (name, loc))
            .collect(),
    )
}
#[cfg(test)]
mod tests {
    use super::{declarations, tokens, translate, UniformLayout};
    use crate::graphics::{ShaderMeta, UniformBlockLayout, UniformDesc, UniformType};

    #[test]
    fn uniform_copy_plan_coalesces_contiguous_array_elements() {
        let layout = UniformLayout::new(&ShaderMeta {
            images: vec![],
            uniforms: UniformBlockLayout {
                uniforms: vec![UniformDesc::new("lines", UniformType::Float4).array(128)],
            },
        });
        assert_eq!(layout.copies, vec![(0, 0, 128 * 16)]);
    }

    #[test]
    fn tokens_keep_scientific_float_literals_intact() {
        assert_eq!(
            tokens("float a = 1e-6; float b = .5E+2;"),
            ["float", "a", "=", "1e-6", ";", "float", "b", "=", ".5E+2", ";"]
        );
    }

    #[test]
    fn translate_assigns_locations_to_modern_glsl_interfaces() {
        let vertex = tokens(
            "in vec2 position; out vec2 uv; void main() { gl_Position = vec4(position, 0., 1.); uv = position; }",
        );
        let fragment = tokens(
            "in vec2 uv; out vec4 frag_color; void main() { frag_color = vec4(uv, 0., 1.); }",
        );
        let mut varying = declarations(&vertex, &["out"]);
        for (name, ty) in declarations(&fragment, &["in"]) {
            varying.entry(name).or_insert(ty);
        }
        for (location, (_, slot)) in varying.values_mut().enumerate() {
            *slot = location as u32;
        }
        let meta = ShaderMeta {
            uniforms: UniformBlockLayout { uniforms: vec![] },
            images: vec![],
        };

        let (vertex, attributes) = translate(&vertex, true, &meta, &varying);
        let (fragment, _) = translate(&fragment, false, &meta, &varying);

        assert_eq!(attributes.get("position"), Some(&0));
        assert!(vertex.contains("layout(location=0) in vec2 position"));
        assert!(vertex.contains("layout(location=0) out vec2 uv"));
        assert!(fragment.contains("layout(location=0) in vec2 uv"));
        assert!(fragment.contains("layout(location=0) out vec4 frag_color"));
    }
}
pub(super) fn compile(
    device: &wgpu::Device,
    source: ShaderSource,
    meta: ShaderMeta,
) -> Result<Shader, ShaderError> {
    if let ShaderSource::Wgsl { program } = source {
        return compile_wgsl(device, program, meta);
    }
    let ShaderSource::Glsl { vertex, fragment } = source else {
        return Err(ShaderError::LinkError(
            "wgpu accepts GLSL sources; MSL is specific to Metal".into(),
        ));
    };
    let vt = tokens(vertex);
    let ft = tokens(fragment);
    let mut varying = declarations(&vt, &["varying", "out"]);
    for (name, ty) in declarations(&ft, &["varying", "in"]) {
        varying.entry(name).or_insert(ty);
    }
    let mut location = 0;
    for (ty, loc) in varying.values_mut() {
        *loc = location;
        location += if ty == "mat4" { 4 } else { 1 };
    }
    let (vertex, attributes) = translate(&vt, true, &meta, &varying);
    let (fragment, _) = translate(&ft, false, &meta, &varying);
    let module = |text: String,
                  stage: naga::ShaderStage,
                  shader_type|
     -> Result<wgpu::ShaderModule, ShaderError> {
        let module = naga::front::glsl::Frontend::default()
            .parse(&naga::front::glsl::Options::from(stage), &text)
            .map_err(|e| ShaderError::CompilationError {
                shader_type,
                error_message: e.emit_to_string(&text),
            })?;
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .map_err(|e| ShaderError::CompilationError {
            shader_type,
            error_message: e.to_string(),
        })?;
        Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("miniquad GLSL"),
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
        }))
    };
    let vertex = module(vertex, naga::ShaderStage::Vertex, ShaderType::Vertex)?;
    let fragment = module(fragment, naga::ShaderStage::Fragment, ShaderType::Fragment)?;
    let uniforms = UniformLayout::new(&meta);
    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: None,
        },
        count: None,
    }];
    for (i, name) in meta.images.iter().enumerate() {
        let cube = vt
            .windows(3)
            .chain(ft.windows(3))
            .any(|w| w[0] == "uniform" && w[1] == "samplerCube" && &w[2] == name);
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 1 + i as u32 * 2,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: if cube {
                    wgpu::TextureViewDimension::Cube
                } else {
                    wgpu::TextureViewDimension::D2
                },
                multisampled: false,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 2 + i as u32 * 2,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
    }
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("miniquad shader bindings"),
        entries: &entries,
    });
    Ok(Shader {
        vertex,
        fragment,
        vertex_entry: "main",
        fragment_entry: "main",
        layout,
        attributes,
        uniforms,
        images: meta.images.len(),
    })
}

fn compile_wgsl(
    device: &wgpu::Device,
    source: &str,
    meta: ShaderMeta,
) -> Result<Shader, ShaderError> {
    let error = |message: String| ShaderError::LinkError(message);
    let module =
        naga::front::wgsl::parse_str(source).map_err(|e| error(e.emit_to_string(source)))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .map_err(|e| error(e.to_string()))?;
    let vertex = module
        .entry_points
        .iter()
        .find(|e| e.name == "vs_main" && e.stage == naga::ShaderStage::Vertex)
        .ok_or_else(|| error("WGSL needs @vertex fn vs_main".into()))?;
    if !module
        .entry_points
        .iter()
        .any(|e| e.name == "fs_main" && e.stage == naga::ShaderStage::Fragment)
    {
        return Err(error("WGSL needs @fragment fn fs_main".into()));
    }
    let mut attributes = BTreeMap::new();
    for arg in &vertex.function.arguments {
        if let Some(naga::Binding::Location { location, .. }) = arg.binding {
            if let Some(name) = &arg.name {
                attributes.insert(name.clone(), location);
            }
        } else if let naga::TypeInner::Struct { members, .. } = &module.types[arg.ty].inner {
            for member in members {
                if let Some(naga::Binding::Location { location, .. }) = member.binding {
                    if let Some(name) = &member.name {
                        attributes.insert(name.clone(), location);
                    }
                }
            }
        }
    }
    let block = module.global_variables.iter().find_map(|(_, g)| {
        if g.binding
            == Some(naga::ResourceBinding {
                group: 0,
                binding: 0,
            })
        {
            Some(g.ty)
        } else {
            None
        }
    });
    let mut uniforms = UniformLayout {
        size: 16,
        packed_size: 0,
        copies: vec![],
    };
    if let Some(ty) = block {
        let naga::TypeInner::Struct { members, span } = &module.types[ty].inner else {
            return Err(error("WGSL binding 0 must be a uniform struct".into()));
        };
        uniforms.size = shader::align_up(*span as usize, 16);
        for u in &meta.uniforms.uniforms {
            let size = u.uniform_type.size();
            if let Some(member) = members.iter().find(|m| m.name.as_deref() == Some(&u.name)) {
                let stride = match module.types[member.ty].inner {
                    naga::TypeInner::Array { stride, .. } => stride as usize,
                    _ => size,
                };
                for i in 0..u.array_count {
                    uniforms.push_copy(
                        uniforms.packed_size + i * size,
                        member.offset as usize + i * stride,
                        size,
                    );
                }
            }
            uniforms.packed_size += size * u.array_count;
        }
        if uniforms
            .copies
            .iter()
            .any(|&(_, dst, len)| dst + len > uniforms.size)
        {
            return Err(error(
                "WGSL uniform metadata exceeds the declared block".into(),
            ));
        }
    } else {
        uniforms.packed_size = meta
            .uniforms
            .uniforms
            .iter()
            .map(|u| u.uniform_type.size() * u.array_count)
            .sum();
    }
    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: None,
        },
        count: None,
    }];
    for i in 0..meta.images.len() {
        let binding = 1 + i as u32 * 2;
        let dim = module.global_variables.iter().find_map(|(_, g)| {
            if g.binding != Some(naga::ResourceBinding { group: 0, binding }) {
                return None;
            }
            if let naga::TypeInner::Image { dim, .. } = module.types[g.ty].inner {
                Some(dim)
            } else {
                None
            }
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: if dim == Some(naga::ImageDimension::Cube) {
                    wgpu::TextureViewDimension::Cube
                } else {
                    wgpu::TextureViewDimension::D2
                },
                multisampled: false,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: binding + 1,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
    }
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("miniquad WGSL bindings"),
        entries: &entries,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("miniquad WGSL"),
        source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
    });
    Ok(Shader {
        vertex: module.clone(),
        fragment: module,
        vertex_entry: "vs_main",
        fragment_entry: "fs_main",
        layout,
        attributes,
        uniforms,
        images: meta.images.len(),
    })
}
