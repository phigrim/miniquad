//! Default desktop wrapper for [`crate::embed::WgpuEmbedHost`].
use crate::{conf::Conf, embed, EventHandler};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::WindowId,
};

struct Clipboard(Option<arboard::Clipboard>);
impl embed::WgpuEmbedClipboard for Clipboard {
    fn get(&mut self) -> Option<String> {
        self.0.as_mut()?.get_text().ok()
    }
    fn set(&mut self, text: &str) {
        if let Some(clipboard) = &mut self.0 {
            let _ = clipboard.set_text(text);
        }
    }
}

struct App {
    host: embed::WgpuEmbedHost,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.host.is_attached() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(self.host.window_attributes())
                .expect("create wgpu window"),
        );
        self.host.attach_window(window).expect("initialize wgpu");
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
        self.host.about_to_wait(event_loop);
    }
}

pub(crate) fn run<F: 'static + FnOnce() -> Box<dyn EventHandler>>(conf: Conf, factory: F) {
    let event_loop = EventLoop::new().expect("create wgpu event loop");
    let clipboard = Box::new(Clipboard(arboard::Clipboard::new().ok()));
    let mut app = App {
        host: embed::WgpuEmbedHost::with_clipboard(conf, clipboard, factory),
    };
    event_loop.run_app(&mut app).expect("wgpu event loop");
}
