use super::*;
const VERTEX: &str = r#"#version 100
attribute vec2 position;
uniform vec4 tint;
varying vec4 color;
void main() { gl_Position = vec4(position, 0.0, 1.0); color = tint; }
"#;
const FRAGMENT: &str = r#"#version 100
precision mediump float;
varying vec4 color;
void main() { gl_FragColor = color; }
"#;
fn meta() -> ShaderMeta {
    ShaderMeta {
        images: vec![],
        uniforms: UniformBlockLayout {
            uniforms: vec![UniformDesc::new("tint", UniformType::Float4)],
        },
    }
}
fn gpu() -> WgpuContext {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("GPU tests require an adapter");
    pollster::block_on(WgpuContext::from_adapter(adapter))
}
fn target(ctx: &mut WgpuContext) -> (TextureId, RenderPass) {
    let t = ctx.new_render_texture(TextureParams {
        width: 32,
        height: 16,
        ..Default::default()
    });
    let p = ctx.new_render_pass(t, None);
    (t, p)
}
fn geometry(ctx: &mut WgpuContext) -> (BufferId, BufferId) {
    let v = ctx.new_buffer(
        BufferType::VertexBuffer,
        BufferUsage::Dynamic,
        BufferSource::slice(&[[-1f32, -1.], [3., -1.], [-1., 3.]]),
    );
    let i = ctx.new_buffer(
        BufferType::IndexBuffer,
        BufferUsage::Immutable,
        BufferSource::slice(&[0u16, 1, 2]),
    );
    (v, i)
}
fn pipeline(ctx: &mut WgpuContext) -> Pipeline {
    let shader = ctx
        .new_shader(
            ShaderSource::Glsl {
                vertex: VERTEX,
                fragment: FRAGMENT,
            },
            meta(),
        )
        .unwrap();
    ctx.new_pipeline(
        &[BufferLayout::default()],
        &[VertexAttribute::new("position", VertexFormat::Float2)],
        shader,
        PipelineParams::default(),
    )
}
#[test]
fn packed_uniforms_are_relocated_without_reading_padding() {
    let meta = ShaderMeta {
        images: vec![],
        uniforms: UniformBlockLayout {
            uniforms: vec![
                UniformDesc::new("scalar", UniformType::Float1),
                UniformDesc::new("vector", UniformType::Float3),
                UniformDesc::new("values", UniformType::Float1).array(2),
                UniformDesc::new("matrix", UniformType::Mat4),
            ],
        },
    };
    let layout = shader::UniformLayout::new(&meta);
    let data: Vec<u8> = (0..layout.packed_size).map(|v| v as u8).collect();
    let packed = layout.pack(&data);
    assert_eq!(layout.packed_size, 88);
    assert_eq!(layout.size, 128);
    assert_eq!(&packed[0..4], &data[0..4]);
    assert_eq!(&packed[16..28], &data[4..16]);
    assert_eq!(&packed[32..36], &data[16..20]);
    assert_eq!(&packed[48..52], &data[20..24]);
    assert_eq!(&packed[64..128], &data[24..88]);
    assert_eq!(&packed[4..16], &[0; 12]);
}

#[test]
fn compression_capabilities_follow_enabled_wgpu_features() {
    let bc = compressed_texture_support(wgpu::Features::TEXTURE_COMPRESSION_BC);
    assert!(bc.supports(CompressedTextureFormat::Bc1Rgb));
    assert!(bc.supports(CompressedTextureFormat::Bc7));
    assert!(!bc.supports(CompressedTextureFormat::Etc2Rgba8));

    let etc2 = compressed_texture_support(wgpu::Features::TEXTURE_COMPRESSION_ETC2);
    assert!(etc2.supports(CompressedTextureFormat::Etc2Rgba8));
    assert!(etc2.supports(CompressedTextureFormat::EacRg11));
    assert!(!etc2.supports(CompressedTextureFormat::Bc7));

    let astc = compressed_texture_support(wgpu::Features::TEXTURE_COMPRESSION_ASTC);
    assert!(astc.supports(CompressedTextureFormat::Astc {
        block_width: 6,
        block_height: 6,
    }));
    assert!(!astc.supports(CompressedTextureFormat::PvrtcRgba4));
}

#[test]
fn compressed_format_mapping_covers_wgpu_families_only() {
    assert_eq!(
        texture::compressed_format(CompressedTextureFormat::Bc7),
        Some(wgpu::TextureFormat::Bc7RgbaUnorm)
    );
    assert_eq!(
        texture::compressed_format(CompressedTextureFormat::Etc2Rgba8),
        Some(wgpu::TextureFormat::Etc2Rgba8Unorm)
    );
    assert!(matches!(
        texture::compressed_format(CompressedTextureFormat::Astc {
            block_width: 6,
            block_height: 6,
        }),
        Some(wgpu::TextureFormat::Astc { .. })
    ));
    assert_eq!(
        texture::compressed_format(CompressedTextureFormat::PvrtcRgba4),
        None
    );
}
#[test]
#[ignore = "requires a native GPU adapter"]
fn gpu_draws_preserve_uniform_snapshots_and_scissor_origin() {
    let mut ctx = gpu();
    let (t, pass) = target(&mut ctx);
    let p = pipeline(&mut ctx);
    let (v, i) = geometry(&mut ctx);
    ctx.begin_pass(Some(pass), PassAction::clear_color(0., 0., 0., 1.));
    ctx.apply_pipeline(&p);
    ctx.apply_bindings_from_slice(&[v], i, &[]);
    ctx.apply_uniforms(UniformsSource::table(&[1f32, 0., 0., 1.]));
    ctx.apply_scissor_rect(0, 0, 16, 16);
    ctx.draw(0, 3, 1);
    ctx.apply_uniforms(UniformsSource::table(&[0f32, 1., 0., 1.]));
    ctx.apply_scissor_rect(16, 8, 16, 8);
    ctx.draw(0, 3, 1);
    ctx.end_render_pass();
    ctx.commit_frame();
    let mut pixels = vec![0; 32 * 16 * 4];
    ctx.texture_read_pixels(t, &mut pixels);
    assert_eq!(&pixels[0..4], &[255, 0, 0, 255]);
    assert_eq!(&pixels[16 * 4..17 * 4], &[0, 255, 0, 255]);
    assert_eq!(
        &pixels[(15 * 32 + 16) * 4..(15 * 32 + 17) * 4],
        &[0, 0, 0, 255]
    );
}
#[test]
#[ignore = "requires a native GPU adapter"]
fn gpu_buffer_updates_do_not_rewrite_prior_draws() {
    let mut ctx = gpu();
    let (t, pass) = target(&mut ctx);
    let p = pipeline(&mut ctx);
    let (v, i) = geometry(&mut ctx);
    ctx.begin_pass(Some(pass), PassAction::clear_color(0., 0., 0., 1.));
    ctx.apply_pipeline(&p);
    ctx.apply_bindings_from_slice(&[v], i, &[]);
    ctx.apply_uniforms(UniformsSource::table(&[1f32, 0., 0., 1.]));
    ctx.draw(0, 3, 1);
    ctx.buffer_update(v, BufferSource::slice(&[[2f32, 2.], [3., 2.], [2., 3.]]));
    ctx.apply_uniforms(UniformsSource::table(&[0f32, 1., 0., 1.]));
    ctx.draw(0, 3, 1);
    ctx.end_render_pass();
    let mut pixels = vec![0; 32 * 16 * 4];
    ctx.texture_read_pixels(t, &mut pixels);
    assert!(pixels.chunks_exact(4).all(|p| p == [255, 0, 0, 255]));
}

#[test]
#[ignore = "requires a native GPU adapter"]
fn deleted_texture_remains_bindable_until_the_next_commit() {
    let mut ctx = gpu();
    let data = [255u8; 4];
    let texture = ctx.new_texture_from_rgba8(1, 1, &data);

    // This is macroquad's scene-release order: present first, collect GPU
    // resources afterwards, then flush a cached draw during the next frame.
    ctx.commit_frame();
    ctx.delete_texture(texture);
    assert_eq!(ctx.texture_size(texture), (1, 1));

    let (target, pass) = target(&mut ctx);
    let shader = ctx
        .new_shader(
            ShaderSource::Glsl {
                vertex: "#version 100\nattribute vec2 position; varying vec2 uv; void main() { gl_Position = vec4(position, 0., 1.); uv = position * .5 + .5; }",
                fragment: "#version 100\nprecision mediump float; varying vec2 uv; uniform sampler2D Texture; void main() { gl_FragColor = texture2D(Texture, uv); }",
            },
            ShaderMeta {
                images: vec!["Texture".into()],
                uniforms: UniformBlockLayout { uniforms: vec![] },
            },
        )
        .unwrap();
    let pipeline = ctx.new_pipeline(
        &[BufferLayout::default()],
        &[VertexAttribute::new("position", VertexFormat::Float2)],
        shader,
        PipelineParams::default(),
    );
    let (vertices, indices) = geometry(&mut ctx);
    ctx.begin_pass(Some(pass), PassAction::default());
    ctx.apply_pipeline(&pipeline);
    ctx.apply_bindings_from_slice(&[vertices], indices, &[texture]);
    ctx.draw(0, 3, 1);
    ctx.end_render_pass();
    ctx.commit_frame();

    let mut pixels = vec![0; 32 * 16 * 4];
    ctx.texture_read_pixels(target, &mut pixels);
    assert!(pixels.chunks_exact(4).all(|pixel| pixel == [255; 4]));
}
#[test]
#[ignore = "requires a native GPU adapter"]
fn gpu_texture_roundtrip_handles_row_padding_and_rgb_alpha() {
    let mut ctx = gpu();
    for format in [
        TextureFormat::RGBA8,
        TextureFormat::RGB8,
        TextureFormat::Alpha,
        TextureFormat::RGBA16F,
    ] {
        let data: Vec<u8> = (0..format.size(7, 3)).map(|i| (i % 251) as u8).collect();
        let t = ctx.new_texture_from_data_and_format(
            &data,
            TextureParams {
                width: 7,
                height: 3,
                format,
                ..Default::default()
            },
        );
        let mut result = vec![0; data.len()];
        ctx.texture_read_pixels(t, &mut result);
        assert_eq!(result, data);
        ctx.texture_resize(t, 5, 2, None);
        assert_eq!(ctx.texture_size(t), (5, 2));
        ctx.delete_texture(t);
    }
}

#[test]
#[ignore = "requires a native GPU adapter with BC compression"]
fn gpu_uploads_compressed_mipmaps_and_cubemaps() {
    let mut ctx = gpu();
    if !ctx
        .compressed_texture_support()
        .supports(CompressedTextureFormat::Bc1Rgba)
    {
        return;
    }
    let level0 = [0_u8; 32]; // 8x8: two by two BC1 blocks
    let level1 = [0_u8; 8]; // 4x4: one BC1 block
    let texture = ctx
        .new_compressed_texture_checked(
            TextureAccess::Static,
            CompressedTextureSource::Mipmaps(&[&level0, &level1]),
            CompressedTextureParams {
                width: 8,
                height: 8,
                format: CompressedTextureFormat::Bc1Rgba,
                mipmap_filter: MipmapFilterMode::Linear,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(ctx.texture_size(texture), (8, 8));

    let face = [&level0[..], &level1[..]];
    let cube = [&face[..]; 6];
    let cube = ctx
        .new_compressed_texture_checked(
            TextureAccess::Static,
            CompressedTextureSource::CubeMap(&cube),
            CompressedTextureParams {
                kind: TextureKind::CubeMap,
                width: 8,
                height: 8,
                format: CompressedTextureFormat::Bc1Rgba,
                mipmap_filter: MipmapFilterMode::Linear,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(ctx.texture_size(cube), (8, 8));
    ctx.delete_texture(texture);
    ctx.delete_texture(cube);
}

#[test]
#[ignore = "requires a native GPU adapter with BC compression"]
fn gpu_rejects_unaligned_compressed_dimensions_before_upload() {
    let mut ctx = gpu();
    if !ctx
        .compressed_texture_support()
        .supports(CompressedTextureFormat::Bc1Rgba)
    {
        return;
    }
    let bytes = [0_u8; 32];
    assert!(matches!(
        ctx.new_compressed_texture_checked(
            TextureAccess::Static,
            CompressedTextureSource::Mipmaps(&[&bytes]),
            CompressedTextureParams {
                width: 7,
                height: 5,
                format: CompressedTextureFormat::Bc1Rgba,
                ..Default::default()
            },
        ),
        Err(TextureError::UnsupportedDimensions { .. })
    ));
}
#[test]
#[ignore = "requires a native GPU adapter"]
fn gpu_msaa_resolve_and_depth_pass() {
    let mut ctx = gpu();
    let (t, _) = target(&mut ctx);
    let msaa = ctx.new_render_texture(TextureParams {
        width: 32,
        height: 16,
        sample_count: 4,
        ..Default::default()
    });
    let depth = ctx.new_render_texture(TextureParams {
        width: 32,
        height: 16,
        sample_count: 4,
        format: TextureFormat::Depth,
        ..Default::default()
    });
    let pass = ctx.new_render_pass_mrt(&[msaa], Some(&[t]), Some(depth));
    let p = pipeline(&mut ctx);
    let (v, i) = geometry(&mut ctx);
    ctx.begin_pass(Some(pass), PassAction::default());
    ctx.apply_pipeline(&p);
    ctx.apply_bindings_from_slice(&[v], i, &[]);
    ctx.apply_uniforms(UniformsSource::table(&[0f32, 0., 1., 1.]));
    ctx.draw(0, 3, 1);
    ctx.end_render_pass();
    let mut bytes = vec![0; 32 * 16 * 4];
    ctx.texture_read_pixels(t, &mut bytes);
    assert!(bytes.chunks_exact(4).all(|p| p == [0, 0, 255, 255]));
}
#[test]
#[ignore = "requires a native GPU adapter"]
fn gpu_wgsl_reflects_member_offsets_and_attribute_names() {
    let mut ctx = gpu();
    let source = r#"
struct U { padding:vec4<f32>, tint:vec4<f32> }; @group(0) @binding(0) var<uniform> u:U;
@vertex fn vs_main(@location(3) position:vec2<f32>)->@builtin(position) vec4<f32>{return vec4(position,0.5,1.);}
@fragment fn fs_main()->@location(0) vec4<f32>{return u.tint;}
"#;
    let shader = ctx
        .new_shader(ShaderSource::Wgsl { program: source }, meta())
        .unwrap();
    let p = ctx.new_pipeline(
        &[BufferLayout::default()],
        &[VertexAttribute::new("position", VertexFormat::Float2)],
        shader,
        Default::default(),
    );
    let (t, pass) = target(&mut ctx);
    let (v, i) = geometry(&mut ctx);
    ctx.begin_pass(Some(pass), PassAction::default());
    ctx.apply_pipeline(&p);
    ctx.apply_bindings_from_slice(&[v], i, &[]);
    ctx.apply_uniforms(UniformsSource::table(&[1f32, 0., 1., 1.]));
    ctx.draw(0, 3, 1);
    ctx.end_render_pass();
    let mut bytes = vec![0; 32 * 16 * 4];
    ctx.texture_read_pixels(t, &mut bytes);
    assert!(bytes.chunks_exact(4).all(|p| p == [255, 0, 255, 255]));
}
