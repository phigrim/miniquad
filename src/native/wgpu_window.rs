//! Window/event adapter for the portable renderer. No OpenGL context is created.
use super::{NativeDisplayData, Request};
use crate::{conf::Conf, *};
use std::{
    cell::RefCell,
    sync::{mpsc, Arc},
    time::{Duration, Instant},
};
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalSize, PhysicalPosition, PhysicalSize},
    event::{DeviceEvent, ElementState, Ime, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::PhysicalKey,
    window::{Window, WindowId},
};
thread_local! {static CONTEXT:RefCell<Option<WgpuContext>>=const{RefCell::new(None)};}
pub(crate) fn take_context() -> Option<WgpuContext> {
    CONTEXT.with(|c| c.borrow_mut().take())
}
struct Clipboard(Option<arboard::Clipboard>);
impl super::Clipboard for Clipboard {
    fn get(&mut self) -> Option<String> {
        self.0.as_mut()?.get_text().ok()
    }
    fn set(&mut self, text: &str) {
        if let Some(c) = &mut self.0 {
            let _ = c.set_text(text);
        }
    }
}
struct App<F> {
    conf: Conf,
    factory: Option<F>,
    handler: Option<Box<dyn EventHandler>>,
    window: Option<Arc<Window>>,
    rx: mpsc::Receiver<Request>,
    tx: mpsc::Sender<Request>,
    mods: KeyMods,
    cursor: (f32, f32),
    start: Instant,
    scheduled: bool,
}
impl<F: 'static + FnOnce() -> Box<dyn EventHandler>> ApplicationHandler for App<F> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            if let Some(h) = &mut self.handler {
                h.window_restored_event();
            }
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(&self.conf.window_title)
            .with_inner_size(LogicalSize::new(
                self.conf.window_width,
                self.conf.window_height,
            ))
            .with_resizable(self.conf.window_resizable)
            .with_fullscreen(if self.conf.fullscreen {
                Some(winit::window::Fullscreen::Borderless(None))
            } else {
                None
            });
        let window = Arc::new(event_loop.create_window(attrs).expect("create wgpu window"));
        let context = pollster::block_on(WgpuContext::for_window(window.clone(), &self.conf));
        let size = window.inner_size();
        let mut display = NativeDisplayData::new(
            size.width as _,
            size.height as _,
            self.tx.clone(),
            Box::new(Clipboard(arboard::Clipboard::new().ok())),
        );
        display.high_dpi = self.conf.high_dpi;
        display.dpi_scale = window.scale_factor() as f32;
        display.blocking_event_loop = self.conf.platform.blocking_event_loop;
        #[cfg(target_vendor = "apple")]
        {
            display.gfx_api = conf::GfxApi::Wgpu;
        }
        crate::set_display(display);
        CONTEXT.with(|c| *c.borrow_mut() = Some(context));
        self.window = Some(window);
        self.handler = Some(self.factory.take().unwrap()());
        self.scheduled = true;
    }
    fn suspended(&mut self, _: &ActiveEventLoop) {
        if let Some(h) = &mut self.handler {
            h.window_minimized_event();
        }
    }
    fn device_event(&mut self, _: &ActiveEventLoop, _: winit::event::DeviceId, event: DeviceEvent) {
        if let (Some(h), DeviceEvent::MouseMotion { delta }) = (&mut self.handler, event) {
            h.raw_mouse_motion(delta.0 as _, delta.1 as _);
        }
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
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
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
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
}
impl<F> App<F> {
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
}
pub(crate) fn run<F: 'static + FnOnce() -> Box<dyn EventHandler>>(conf: Conf, factory: F) {
    let event_loop = EventLoop::new().expect("create wgpu event loop");
    let (tx, rx) = mpsc::channel();
    let mut app = App {
        conf,
        factory: Some(factory),
        handler: None,
        window: None,
        tx,
        rx,
        mods: Default::default(),
        cursor: (0., 0.),
        start: Instant::now(),
        scheduled: true,
    };
    event_loop.run_app(&mut app).expect("wgpu event loop");
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
