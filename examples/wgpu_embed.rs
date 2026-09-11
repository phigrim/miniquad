//! Minimal caller-owned winit event loop using `WgpuEmbedHost`.
use miniquad::{
    conf::Conf,
    embed::{winit, WgpuEmbedHost},
    window, EventHandler, PassAction, RenderingBackend, TextureFormat, TextureParams,
};
use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc,
};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::WindowId,
};

struct Stage {
    graphics: Box<dyn RenderingBackend>,
    pass: miniquad::RenderPass,
    frames: u8,
    exit_after_smoke: bool,
}

static SMOKE_FRAMES: AtomicU8 = AtomicU8::new(0);

impl Stage {
    fn new() -> Self {
        let mut graphics = window::new_rendering_backend();
        let texture = graphics.new_render_texture(TextureParams {
            width: 64,
            height: 64,
            format: TextureFormat::RGBA8,
            ..Default::default()
        });
        let pass = graphics.new_render_pass(texture, None);
        Self {
            graphics,
            pass,
            frames: 0,
            exit_after_smoke: std::env::var_os("MINIQUAD_WGPU_SMOKE").is_some(),
        }
    }
}

impl EventHandler for Stage {
    fn update(&mut self) {}

    fn draw(&mut self) {
        self.graphics.begin_pass(
            Some(self.pass),
            PassAction::clear_color(0.15, 0.35, 0.7, 1.0),
        );
        self.graphics.end_render_pass();
        self.graphics
            .begin_default_pass(PassAction::clear_color(0.08, 0.08, 0.1, 1.0));
        self.graphics.end_render_pass();
        self.graphics.commit_frame();
        self.frames = self.frames.saturating_add(1);
        if self.exit_after_smoke {
            SMOKE_FRAMES.store(self.frames, Ordering::Release);
            if self.frames == 6 {
                window::quit();
            }
        }
    }
}

struct App {
    host: WgpuEmbedHost,
    smoke_reattached: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(self.host.window_attributes())
                .expect("create window"),
        );
        self.host.attach_window(window).expect("attach window");
    }

    fn suspended(&mut self, _: &ActiveEventLoop) {
        self.host.detach_window();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        self.host.window_event(event_loop, id, event);
    }

    fn device_event(&mut self, _: &ActiveEventLoop, id: DeviceId, event: DeviceEvent) {
        self.host.device_event(id, event);
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if !self.smoke_reattached
            && std::env::var_os("MINIQUAD_WGPU_SMOKE").is_some()
            && SMOKE_FRAMES.load(Ordering::Acquire) >= 3
        {
            self.host.detach_window();
            let window = Arc::new(
                event_loop
                    .create_window(self.host.window_attributes())
                    .expect("recreate window"),
            );
            self.host.attach_window(window).expect("reattach window");
            self.smoke_reattached = true;
        }
        self.host.about_to_wait(event_loop);
    }
}

fn main() {
    let mut conf = Conf::default();
    conf.window_title = "miniquad wgpu embed".into();
    let mut app = App {
        host: WgpuEmbedHost::new(conf, || Box::new(Stage::new())),
        smoke_reattached: false,
    };
    EventLoop::new()
        .expect("create event loop")
        .run_app(&mut app)
        .expect("run event loop");
}
