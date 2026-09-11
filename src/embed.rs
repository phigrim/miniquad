//! Embedding support for driving miniquad from a caller-owned winit event loop.
use crate::native::{NativeDisplayData, Request};
use crate::{conf::Conf, *};
use std::{
    cell::RefCell,
    fmt,
    sync::atomic::{AtomicBool, Ordering},
    sync::{mpsc, Arc},
    time::{Duration, Instant},
};
pub use winit;
use winit::{
    dpi::{LogicalSize, PhysicalPosition, PhysicalSize},
    event::{DeviceEvent, ElementState, Ime, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow},
    keyboard::PhysicalKey,
    window::{Window, WindowId},
};

thread_local! {static CONTEXT:RefCell<Option<WgpuContext>>=const{RefCell::new(None)};}
pub(crate) fn take_context() -> Option<WgpuContext> {
    CONTEXT.with(|c| c.borrow_mut().take())
}

static ACTIVE_HOST: AtomicBool = AtomicBool::new(false);

/// Clipboard integration supplied by an embedding application.
pub trait WgpuEmbedClipboard: Send + Sync {
    fn get(&mut self) -> Option<String>;
    fn set(&mut self, text: &str);
}

struct ClipboardAdapter(Box<dyn WgpuEmbedClipboard>);
impl crate::native::Clipboard for ClipboardAdapter {
    fn get(&mut self) -> Option<String> {
        self.0.get()
    }
    fn set(&mut self, text: &str) {
        self.0.set(text)
    }
}

struct EmptyClipboard;
impl WgpuEmbedClipboard for EmptyClipboard {
    fn get(&mut self) -> Option<String> {
        None
    }
    fn set(&mut self, _: &str) {}
}

/// An error produced while attaching a winit window to a wgpu host.
#[derive(Debug)]
pub enum WgpuEmbedError {
    SurfaceCreation(String),
    AdapterUnavailable,
    DeviceCreation(String),
    SurfaceUnsupported,
    AlreadyAttached,
    NotInitialized,
}

impl fmt::Display for WgpuEmbedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SurfaceCreation(message) => write!(f, "failed to create wgpu surface: {message}"),
            Self::AdapterUnavailable => f.write_str("no compatible wgpu adapter is available"),
            Self::DeviceCreation(message) => write!(f, "failed to create wgpu device: {message}"),
            Self::SurfaceUnsupported => {
                f.write_str("the window surface is unsupported by the adapter")
            }
            Self::AlreadyAttached => f.write_str("a window is already attached"),
            Self::NotInitialized => f.write_str("the wgpu embed host is not initialized"),
        }
    }
}
impl std::error::Error for WgpuEmbedError {}

impl From<crate::graphics::wgpu::SurfaceInitError> for WgpuEmbedError {
    fn from(error: crate::graphics::wgpu::SurfaceInitError) -> Self {
        use crate::graphics::wgpu::SurfaceInitError as E;
        match error {
            E::Surface(message) => Self::SurfaceCreation(message),
            E::AdapterUnavailable => Self::AdapterUnavailable,
            E::Device(message) => Self::DeviceCreation(message),
            E::Unsupported => Self::SurfaceUnsupported,
        }
    }
}

/// Drives miniquad from a caller-owned winit event loop and window.
pub struct WgpuEmbedHost {
    conf: Conf,
    factory: Option<Box<dyn FnOnce() -> Box<dyn EventHandler>>>,
    handler: Option<Box<dyn EventHandler>>,
    window: Option<Arc<Window>>,
    surface: Option<crate::graphics::wgpu::SurfaceController>,
    clipboard: Option<Box<dyn WgpuEmbedClipboard>>,
    rx: mpsc::Receiver<Request>,
    tx: mpsc::Sender<Request>,
    mods: KeyMods,
    cursor: (f32, f32),
    start: Instant,
    scheduled: bool,
    owns_active_slot: bool,
}

impl WgpuEmbedHost {
    pub fn new<F>(conf: Conf, factory: F) -> Self
    where
        F: FnOnce() -> Box<dyn EventHandler> + 'static,
    {
        Self::with_clipboard(conf, Box::new(EmptyClipboard), factory)
    }

    pub fn with_clipboard<F>(conf: Conf, clipboard: Box<dyn WgpuEmbedClipboard>, factory: F) -> Self
    where
        F: FnOnce() -> Box<dyn EventHandler> + 'static,
    {
        let (tx, rx) = mpsc::channel();
        Self {
            conf,
            factory: Some(Box::new(factory)),
            handler: None,
            window: None,
            surface: None,
            clipboard: Some(clipboard),
            rx,
            tx,
            mods: Default::default(),
            cursor: (0., 0.),
            start: Instant::now(),
            scheduled: true,
            owns_active_slot: false,
        }
    }

    pub fn window_attributes(&self) -> winit::window::WindowAttributes {
        Window::default_attributes()
            .with_title(&self.conf.window_title)
            .with_inner_size(LogicalSize::new(
                self.conf.window_width,
                self.conf.window_height,
            ))
            .with_resizable(self.conf.window_resizable)
            .with_transparent(self.conf.platform.framebuffer_alpha)
            .with_fullscreen(if self.conf.fullscreen {
                Some(winit::window::Fullscreen::Borderless(None))
            } else {
                None
            })
    }

    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    pub(crate) fn is_attached(&self) -> bool {
        self.window.is_some()
    }

    pub fn attach_window(&mut self, window: Arc<Window>) -> Result<(), WgpuEmbedError> {
        if self.window.is_some() {
            return Err(WgpuEmbedError::AlreadyAttached);
        }
        let first_attach = self.surface.is_none();
        if first_attach {
            if self.factory.is_none() {
                return Err(WgpuEmbedError::NotInitialized);
            }
            if ACTIVE_HOST
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                return Err(WgpuEmbedError::AlreadyAttached);
            }
            self.owns_active_slot = true;
            let initialized =
                pollster::block_on(WgpuContext::for_window(window.clone(), &self.conf));
            let (context, surface) = match initialized {
                Ok(result) => result,
                Err(error) => {
                    self.release_active_slot();
                    return Err(error.into());
                }
            };
            self.surface = Some(surface);
            CONTEXT.with(|slot| *slot.borrow_mut() = Some(context));
        } else {
            self.surface
                .as_ref()
                .ok_or(WgpuEmbedError::NotInitialized)?
                .attach_window(window.clone())?;
        }
        let size = window.inner_size();
        if first_attach {
            #[cfg(target_os = "android")]
            let native_requests = {
                let tx = self.tx.clone();
                Box::new(move |request| {
                    let _ = tx.send(request);
                }) as Box<dyn Fn(Request) + Send>
            };
            #[cfg(not(target_os = "android"))]
            let native_requests = self.tx.clone();
            let mut display = NativeDisplayData::new(
                size.width as _,
                size.height as _,
                native_requests,
                Box::new(ClipboardAdapter(self.clipboard.take().unwrap())),
            );
            display.high_dpi = self.conf.high_dpi;
            display.dpi_scale = window.scale_factor() as f32;
            display.blocking_event_loop = self.conf.platform.blocking_event_loop;
            #[cfg(target_vendor = "apple")]
            {
                display.gfx_api = conf::GfxApi::Wgpu;
            }
            crate::set_or_replace_display(display);
        } else {
            let mut display = crate::native_display().lock().unwrap();
            display.screen_width = size.width as _;
            display.screen_height = size.height as _;
            display.dpi_scale = window.scale_factor() as _;
        }
        window.request_redraw();
        self.window = Some(window);
        if first_attach {
            self.handler = Some(self.factory.take().ok_or(WgpuEmbedError::NotInitialized)?());
        } else if let Some(handler) = &mut self.handler {
            handler.window_restored_event();
        }
        self.scheduled = true;
        Ok(())
    }

    pub fn detach_window(&mut self) {
        if self.window.take().is_none() {
            return;
        }
        if let Some(surface) = &self.surface {
            surface.detach_window();
        }
        if let Some(h) = &mut self.handler {
            h.window_minimized_event();
        }
    }

    pub fn device_event(&mut self, _: winit::event::DeviceId, event: DeviceEvent) {
        if let (Some(h), DeviceEvent::MouseMotion { delta }) = (&mut self.handler, event) {
            h.raw_mouse_motion(delta.0 as _, delta.1 as _);
        }
    }

    pub fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if self
            .window
            .as_ref()
            .map_or(true, |window| window.id() != id)
        {
            return;
        }
        let Some(h) = &mut self.handler else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => {
                crate::native_display().lock().unwrap().quit_requested = true;
            }
            WindowEvent::RedrawRequested => {
                self.scheduled = false;
                h.update();
                h.draw();
            }
            WindowEvent::Resized(size) => {
                {
                    let mut d = crate::native_display().lock().unwrap();
                    d.screen_width = size.width as _;
                    d.screen_height = size.height as _;
                }
                h.resize_event(size.width as _, size.height as _);
                self.scheduled = true;
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                crate::native_display().lock().unwrap().dpi_scale = scale_factor as _
            }
            WindowEvent::Moved(p) => {
                crate::native_display().lock().unwrap().screen_position = (p.x as _, p.y as _)
            }
            WindowEvent::ModifiersChanged(m) => {
                let m = m.state();
                self.mods = KeyMods {
                    shift: m.shift_key(),
                    ctrl: m.control_key(),
                    alt: m.alt_key(),
                    logo: m.super_key(),
                };
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let key = match event.physical_key {
                    PhysicalKey::Code(c) => keycode(c),
                    _ => KeyCode::Unknown,
                };
                if event.state == ElementState::Pressed {
                    h.key_down_event(key, self.mods, event.repeat);
                    if let Some(text) = event.text {
                        for c in text.chars().filter(|c| !c.is_control()) {
                            h.char_event(c, self.mods, event.repeat);
                        }
                    }
                } else {
                    h.key_up_event(key, self.mods);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = (position.x as _, position.y as _);
                h.mouse_motion_event(self.cursor.0, self.cursor.1);
            }
            WindowEvent::CursorEntered { .. } => {
                h.mouse_enter_event(MouseButton::Unknown, self.cursor.0, self.cursor.1)
            }
            WindowEvent::CursorLeft { .. } => h.mouse_leave_event(),
            WindowEvent::MouseInput { state, button, .. } => {
                let button = match button {
                    winit::event::MouseButton::Left => MouseButton::Left,
                    winit::event::MouseButton::Right => MouseButton::Right,
                    winit::event::MouseButton::Middle => MouseButton::Middle,
                    _ => MouseButton::Unknown,
                };
                if state == ElementState::Pressed {
                    h.mouse_button_down_event(button, self.cursor.0, self.cursor.1);
                } else {
                    h.mouse_button_up_event(button, self.cursor.0, self.cursor.1);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let (x, y) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => (x, y),
                    MouseScrollDelta::PixelDelta(p) => (p.x as f32, p.y as f32),
                };
                h.mouse_wheel_event(x, y);
            }
            WindowEvent::Touch(t) => h.touch_event(
                match t.phase {
                    winit::event::TouchPhase::Started => TouchPhase::Started,
                    winit::event::TouchPhase::Moved => TouchPhase::Moved,
                    winit::event::TouchPhase::Ended => TouchPhase::Ended,
                    winit::event::TouchPhase::Cancelled => TouchPhase::Cancelled,
                },
                t.id,
                t.location.x as _,
                t.location.y as _,
                self.start.elapsed().as_secs_f64(),
            ),
            WindowEvent::Ime(Ime::Preedit(text, cursor)) => {
                let cursor = cursor.map_or(0, |c| text[..c.0].encode_utf16().count());
                h.on_ime_preedit(&text, cursor);
            }
            WindowEvent::Ime(Ime::Commit(text)) => h.on_ime_commit(Some(&text)),
            WindowEvent::Focused(true) => h.window_restored_event(),
            WindowEvent::Focused(false) => h.window_minimized_event(),
            WindowEvent::DroppedFile(path) => {
                {
                    let mut d = crate::native_display().lock().unwrap();
                    d.dropped_files.paths = vec![path];
                }
                h.files_dropped_event();
            }
            _ => {}
        }
        self.quit(event_loop);
    }

    pub fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(w) = &self.window else {
            return;
        };
        while let Ok(request) = self.rx.try_recv() {
            match request {
                Request::ScheduleUpdate => self.scheduled = true,
                Request::SetWindowSize {
                    new_width,
                    new_height,
                } => {
                    let _ = w.request_inner_size(PhysicalSize::new(new_width, new_height));
                }
                Request::SetWindowPosition { new_x, new_y } => {
                    w.set_outer_position(PhysicalPosition::new(new_x, new_y))
                }
                Request::SetFullscreen(f) => w.set_fullscreen(if f {
                    Some(winit::window::Fullscreen::Borderless(None))
                } else {
                    None
                }),
                Request::ShowMouse(show) => w.set_cursor_visible(show),
                Request::SetMouseCursor(cursor) => w.set_cursor(cursor_icon(cursor)),
                Request::SetCursorGrab(grab) => {
                    let mode = if grab {
                        winit::window::CursorGrabMode::Locked
                    } else {
                        winit::window::CursorGrabMode::None
                    };
                    if w.set_cursor_grab(mode).is_err() && grab {
                        let _ = w.set_cursor_grab(winit::window::CursorGrabMode::Confined);
                    }
                }
                Request::SetImeEnabled(enabled) | Request::ShowKeyboard(enabled) => {
                    w.set_ime_allowed(enabled)
                }
                Request::SetImePosition { x, y } => {
                    w.set_ime_cursor_area(PhysicalPosition::new(x, y), PhysicalSize::new(1, 1))
                }
                Request::UpdateTextInputState { .. } => {}
            }
        }
        if !self.conf.platform.blocking_event_loop || self.scheduled {
            w.request_redraw();
        }
        // Existing request senders are mpsc; bounded polling also wakes requests sent by worker threads.
        event_loop.set_control_flow(if self.conf.platform.blocking_event_loop {
            ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(8))
        } else {
            ControlFlow::Poll
        });
        self.quit(event_loop);
    }

    fn quit(&mut self, event_loop: &ActiveEventLoop) {
        let requested = crate::native_display().lock().unwrap().quit_requested;
        if requested {
            self.handler.as_mut().unwrap().quit_requested_event();
            let mut d = crate::native_display().lock().unwrap();
            if d.quit_requested {
                d.quit_ordered = true;
            }
        }
        if crate::native_display().lock().unwrap().quit_ordered {
            event_loop.exit();
        }
    }

    fn release_active_slot(&mut self) {
        if self.owns_active_slot {
            ACTIVE_HOST.store(false, Ordering::Release);
            self.owns_active_slot = false;
        }
    }
}

impl Drop for WgpuEmbedHost {
    fn drop(&mut self) {
        self.detach_window();
        CONTEXT.with(|slot| {
            slot.borrow_mut().take();
        });
        self.release_active_slot();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Handler;
    impl EventHandler for Handler {
        fn update(&mut self) {}
        fn draw(&mut self) {}
    }

    fn host() -> WgpuEmbedHost {
        WgpuEmbedHost::new(Conf::default(), || Box::new(Handler))
    }

    #[test]
    fn detach_before_attach_is_idempotent() {
        let mut host = host();
        host.detach_window();
        host.detach_window();
        assert!(host.factory.is_some());
    }

    #[test]
    fn requests_remain_queued_without_a_window() {
        let host = host();
        host.tx.send(Request::ScheduleUpdate).unwrap();
        assert!(matches!(host.rx.try_recv(), Ok(Request::ScheduleUpdate)));
    }
}
fn cursor_icon(c: CursorIcon) -> winit::window::CursorIcon {
    use winit::window::CursorIcon as W;
    match c {
        CursorIcon::Default => W::Default,
        CursorIcon::Help => W::Help,
        CursorIcon::Pointer => W::Pointer,
        CursorIcon::Wait => W::Wait,
        CursorIcon::Crosshair => W::Crosshair,
        CursorIcon::Text => W::Text,
        CursorIcon::Move => W::Move,
        CursorIcon::NotAllowed => W::NotAllowed,
        CursorIcon::EWResize => W::EwResize,
        CursorIcon::NSResize => W::NsResize,
        CursorIcon::NESWResize => W::NeswResize,
        CursorIcon::NWSEResize => W::NwseResize,
    }
}

fn keycode(code: winit::keyboard::KeyCode) -> KeyCode {
    use winit::keyboard::KeyCode as W;
    match code {
        W::Space => KeyCode::Space,
        W::Quote => KeyCode::Apostrophe,
        W::Comma => KeyCode::Comma,
        W::Minus => KeyCode::Minus,
        W::Period => KeyCode::Period,
        W::Slash => KeyCode::Slash,
        W::Digit0 => KeyCode::Key0,
        W::Digit1 => KeyCode::Key1,
        W::Digit2 => KeyCode::Key2,
        W::Digit3 => KeyCode::Key3,
        W::Digit4 => KeyCode::Key4,
        W::Digit5 => KeyCode::Key5,
        W::Digit6 => KeyCode::Key6,
        W::Digit7 => KeyCode::Key7,
        W::Digit8 => KeyCode::Key8,
        W::Digit9 => KeyCode::Key9,
        W::Semicolon => KeyCode::Semicolon,
        W::Equal => KeyCode::Equal,
        W::KeyA => KeyCode::A,
        W::KeyB => KeyCode::B,
        W::KeyC => KeyCode::C,
        W::KeyD => KeyCode::D,
        W::KeyE => KeyCode::E,
        W::KeyF => KeyCode::F,
        W::KeyG => KeyCode::G,
        W::KeyH => KeyCode::H,
        W::KeyI => KeyCode::I,
        W::KeyJ => KeyCode::J,
        W::KeyK => KeyCode::K,
        W::KeyL => KeyCode::L,
        W::KeyM => KeyCode::M,
        W::KeyN => KeyCode::N,
        W::KeyO => KeyCode::O,
        W::KeyP => KeyCode::P,
        W::KeyQ => KeyCode::Q,
        W::KeyR => KeyCode::R,
        W::KeyS => KeyCode::S,
        W::KeyT => KeyCode::T,
        W::KeyU => KeyCode::U,
        W::KeyV => KeyCode::V,
        W::KeyW => KeyCode::W,
        W::KeyX => KeyCode::X,
        W::KeyY => KeyCode::Y,
        W::KeyZ => KeyCode::Z,
        W::BracketLeft => KeyCode::LeftBracket,
        W::Backslash => KeyCode::Backslash,
        W::BracketRight => KeyCode::RightBracket,
        W::Backquote => KeyCode::GraveAccent,
        W::Escape => KeyCode::Escape,
        W::Enter => KeyCode::Enter,
        W::Tab => KeyCode::Tab,
        W::Backspace => KeyCode::Backspace,
        W::Insert => KeyCode::Insert,
        W::Delete => KeyCode::Delete,
        W::ArrowRight => KeyCode::Right,
        W::ArrowLeft => KeyCode::Left,
        W::ArrowDown => KeyCode::Down,
        W::ArrowUp => KeyCode::Up,
        W::PageUp => KeyCode::PageUp,
        W::PageDown => KeyCode::PageDown,
        W::Home => KeyCode::Home,
        W::End => KeyCode::End,
        W::CapsLock => KeyCode::CapsLock,
        W::ScrollLock => KeyCode::ScrollLock,
        W::NumLock => KeyCode::NumLock,
        W::PrintScreen => KeyCode::PrintScreen,
        W::Pause => KeyCode::Pause,
        W::F1 => KeyCode::F1,
        W::F2 => KeyCode::F2,
        W::F3 => KeyCode::F3,
        W::F4 => KeyCode::F4,
        W::F5 => KeyCode::F5,
        W::F6 => KeyCode::F6,
        W::F7 => KeyCode::F7,
        W::F8 => KeyCode::F8,
        W::F9 => KeyCode::F9,
        W::F10 => KeyCode::F10,
        W::F11 => KeyCode::F11,
        W::F12 => KeyCode::F12,
        W::F13 => KeyCode::F13,
        W::F14 => KeyCode::F14,
        W::F15 => KeyCode::F15,
        W::F16 => KeyCode::F16,
        W::F17 => KeyCode::F17,
        W::F18 => KeyCode::F18,
        W::F19 => KeyCode::F19,
        W::F20 => KeyCode::F20,
        W::F21 => KeyCode::F21,
        W::F22 => KeyCode::F22,
        W::F23 => KeyCode::F23,
        W::F24 => KeyCode::F24,
        W::Numpad0 => KeyCode::Kp0,
        W::Numpad1 => KeyCode::Kp1,
        W::Numpad2 => KeyCode::Kp2,
        W::Numpad3 => KeyCode::Kp3,
        W::Numpad4 => KeyCode::Kp4,
        W::Numpad5 => KeyCode::Kp5,
        W::Numpad6 => KeyCode::Kp6,
        W::Numpad7 => KeyCode::Kp7,
        W::Numpad8 => KeyCode::Kp8,
        W::Numpad9 => KeyCode::Kp9,
        W::NumpadDecimal => KeyCode::KpDecimal,
        W::NumpadDivide => KeyCode::KpDivide,
        W::NumpadMultiply => KeyCode::KpMultiply,
        W::NumpadSubtract => KeyCode::KpSubtract,
        W::NumpadAdd => KeyCode::KpAdd,
        W::NumpadEnter => KeyCode::KpEnter,
        W::NumpadEqual => KeyCode::KpEqual,
        W::ShiftLeft => KeyCode::LeftShift,
        W::ControlLeft => KeyCode::LeftControl,
        W::AltLeft => KeyCode::LeftAlt,
        W::SuperLeft => KeyCode::LeftSuper,
        W::ShiftRight => KeyCode::RightShift,
        W::ControlRight => KeyCode::RightControl,
        W::AltRight => KeyCode::RightAlt,
        W::SuperRight => KeyCode::RightSuper,
        W::ContextMenu => KeyCode::Menu,
        _ => KeyCode::Unknown,
    }
}
