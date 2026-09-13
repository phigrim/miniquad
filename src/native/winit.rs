//! Shared desktop windowing path for the WGPU renderer.
//!
//! Keeping this separate from the legacy native modules lets the OpenGL and
//! Metal implementations retain their mature platform-specific code while the
//! WGPU backend gets one safe surface owner on Windows, Linux, and macOS.

use std::{
    cell::RefCell,
    sync::mpsc::{self, Receiver},
};

use winit::{
    application::ApplicationHandler,
    dpi::{LogicalPosition, LogicalSize, PhysicalPosition},
    event::{
        DeviceEvent, ElementState, MouseButton as WinitMouseButton, MouseScrollDelta, WindowEvent,
    },
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode as WinitKeyCode, ModifiersState, PhysicalKey},
    window::{
        CursorGrabMode, CursorIcon as WinitCursorIcon, Fullscreen, Window, WindowAttributes,
        WindowId,
    },
};

use crate::{
    conf::Conf,
    event::{EventHandler, KeyCode, KeyMods, MouseButton},
    native::{Clipboard, NativeDisplayData, Request},
    CursorIcon,
};

thread_local! {
    static WGPU_WINDOW: RefCell<Option<std::sync::Arc<Window>>> = const { RefCell::new(None) };
}

/// Returns the current WGPU window. It is intentionally thread-local: winit
/// requires window operations to happen on the event-loop thread.
pub(crate) fn window() -> std::sync::Arc<Window> {
    WGPU_WINDOW.with(|slot| {
        slot.borrow()
            .as_ref()
            .cloned()
            .expect("WGPU window is not initialized")
    })
}

struct SystemClipboard {
    clipboard: Option<arboard::Clipboard>,
    fallback: String,
}

impl SystemClipboard {
    fn new() -> Self {
        Self {
            clipboard: arboard::Clipboard::new().ok(),
            fallback: String::new(),
        }
    }
}

impl Clipboard for SystemClipboard {
    fn get(&mut self) -> Option<String> {
        self.clipboard
            .as_mut()
            .and_then(|clipboard| clipboard.get_text().ok())
            .or_else(|| (!self.fallback.is_empty()).then(|| self.fallback.clone()))
    }

    fn set(&mut self, value: &str) {
        self.fallback.clear();
        self.fallback.push_str(value);
        if let Some(clipboard) = self.clipboard.as_mut() {
            let _ = clipboard.set_text(value);
        }
    }
}

pub fn run<F>(conf: Conf, factory: F)
where
    F: 'static + FnOnce() -> Box<dyn EventHandler>,
{
    let event_loop = EventLoop::new().expect("failed to create winit event loop");
    let mut application = WinitApplication {
        conf,
        factory: Some(Box::new(factory)),
        window: None,
        handler: None,
        requests: None,
        modifiers: ModifiersState::default(),
        cursor_position: (0.0, 0.0),
        update_requested: true,
    };
    event_loop
        .run_app(&mut application)
        .expect("winit event loop terminated unexpectedly");
}

struct WinitApplication {
    conf: Conf,
    factory: Option<Box<dyn FnOnce() -> Box<dyn EventHandler>>>,
    window: Option<std::sync::Arc<Window>>,
    handler: Option<Box<dyn EventHandler>>,
    requests: Option<Receiver<Request>>,
    modifiers: ModifiersState,
    cursor_position: (f32, f32),
    update_requested: bool,
}

impl WinitApplication {
    fn scale_factor(&self) -> f32 {
        self.window
            .as_ref()
            .map(|window| window.scale_factor() as f32)
            .unwrap_or(1.0)
    }

    fn update_display_size(&self, size: winit::dpi::PhysicalSize<u32>) {
        let mut display = crate::native_display().lock().unwrap();
        display.screen_width = size.width.max(1) as i32;
        display.screen_height = size.height.max(1) as i32;
        display.dpi_scale = if self.conf.high_dpi {
            self.scale_factor()
        } else {
            1.0
        };
    }

    fn process_requests(&mut self) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let Some(requests) = self.requests.as_ref() else {
            return;
        };
        while let Ok(request) = requests.try_recv() {
            match request {
                Request::ScheduleUpdate => self.update_requested = true,
                Request::SetCursorGrab(grab) => {
                    let mode = if grab {
                        CursorGrabMode::Confined
                    } else {
                        CursorGrabMode::None
                    };
                    let _ = window.set_cursor_grab(mode);
                }
                Request::ShowMouse(show) => window.set_cursor_visible(show),
                Request::SetMouseCursor(cursor) => window.set_cursor(cursor_icon(cursor)),
                Request::SetWindowSize {
                    new_width,
                    new_height,
                } => {
                    let _ = window.request_inner_size(LogicalSize::new(new_width, new_height));
                }
                Request::SetWindowPosition { new_x, new_y } => {
                    window.set_outer_position(LogicalPosition::new(new_x, new_y));
                }
                Request::SetFullscreen(fullscreen) => {
                    window.set_fullscreen(fullscreen.then(|| Fullscreen::Borderless(None)));
                }
                Request::SetImePosition { x, y } => {
                    window.set_ime_cursor_area(PhysicalPosition::new(x, y), LogicalSize::new(1, 1));
                }
                Request::SetImeEnabled(enabled) => window.set_ime_allowed(enabled),
                Request::ShowKeyboard(_) | Request::UpdateTextInputState { .. } => {}
            }
        }
    }

    fn render(&mut self, event_loop: &ActiveEventLoop) {
        self.process_requests();
        if crate::native_display().lock().unwrap().quit_ordered {
            event_loop.exit();
            return;
        }
        let Some(handler) = self.handler.as_mut() else {
            return;
        };
        if crate::native_display().lock().unwrap().quit_requested {
            handler.quit_requested_event();
            if crate::native_display().lock().unwrap().quit_requested {
                event_loop.exit();
                return;
            }
        }
        handler.update();
        handler.draw();
        self.update_requested = false;
        if crate::native_display().lock().unwrap().quit_ordered {
            event_loop.exit();
        }
    }

    fn event_handler(&mut self) -> &mut dyn EventHandler {
        self.handler
            .as_deref_mut()
            .expect("event handler is not initialized")
    }
}

impl ApplicationHandler for WinitApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        event_loop.set_control_flow(if self.conf.platform.blocking_event_loop {
            ControlFlow::Wait
        } else {
            ControlFlow::Poll
        });
        let attributes = WindowAttributes::default()
            .with_title(self.conf.window_title.clone())
            .with_resizable(self.conf.window_resizable)
            .with_inner_size(LogicalSize::new(
                self.conf.window_width,
                self.conf.window_height,
            ))
            .with_fullscreen(self.conf.fullscreen.then(|| Fullscreen::Borderless(None)))
            .with_visible(!self.conf.headless);
        let attributes = if let Some(icon) = &self.conf.icon {
            attributes.with_window_icon(Some(
                winit::window::Icon::from_rgba(icon.big.to_vec(), 64, 64)
                    .expect("miniquad icon must be 64x64 RGBA"),
            ))
        } else {
            attributes
        };
        let window = std::sync::Arc::new(
            event_loop
                .create_window(attributes)
                .expect("failed to create winit window"),
        );
        let size = window.inner_size();
        let (sender, receiver) = mpsc::channel();
        crate::set_display(NativeDisplayData {
            high_dpi: self.conf.high_dpi,
            dpi_scale: if self.conf.high_dpi {
                window.scale_factor() as f32
            } else {
                1.0
            },
            blocking_event_loop: self.conf.platform.blocking_event_loop,
            sample_count: self.conf.sample_count.max(1) as u32,
            swap_interval: self.conf.platform.swap_interval,
            gfx_api: self.conf.platform.prefer_gfx_api,
            wgpu_backend: self.conf.platform.wgpu_backend,
            ..NativeDisplayData::new(
                size.width.max(1) as i32,
                size.height.max(1) as i32,
                sender,
                Box::new(SystemClipboard::new()),
            )
        });
        WGPU_WINDOW.with(|slot| *slot.borrow_mut() = Some(window.clone()));
        self.handler = Some(self
            .factory
            .take()
            .expect("application factory called twice")());
        self.requests = Some(receiver);
        self.window = Some(window.clone());
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                crate::native_display().lock().unwrap().quit_requested = true;
                self.event_handler().quit_requested_event();
                if crate::native_display().lock().unwrap().quit_requested {
                    event_loop.exit();
                }
            }
            WindowEvent::Resized(size) => {
                self.update_display_size(size);
                self.event_handler()
                    .resize_event(size.width.max(1) as f32, size.height.max(1) as f32);
                self.update_requested = true;
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                if let Some(window) = self.window.as_ref() {
                    self.update_display_size(window.inner_size());
                }
            }
            WindowEvent::RedrawRequested => self.render(event_loop),
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_position = (position.x as f32, position.y as f32);
                let (x, y) = self.cursor_position;
                self.event_handler().mouse_motion_event(x, y);
            }
            WindowEvent::CursorEntered { .. } => {
                let (x, y) = self.cursor_position;
                self.event_handler()
                    .mouse_enter_event(MouseButton::Unknown, x, y);
            }
            WindowEvent::CursorLeft { .. } => self.event_handler().mouse_leave_event(),
            WindowEvent::MouseInput { state, button, .. } => {
                let button = mouse_button(button);
                let (x, y) = self.cursor_position;
                match state {
                    ElementState::Pressed => {
                        self.event_handler().mouse_button_down_event(button, x, y)
                    }
                    ElementState::Released => {
                        self.event_handler().mouse_button_up_event(button, x, y)
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let (x, y) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => (x, y),
                    MouseScrollDelta::PixelDelta(position) => {
                        (position.x as f32, position.y as f32)
                    }
                };
                self.event_handler().mouse_wheel_event(x, y);
            }
            WindowEvent::ModifiersChanged(modifiers) => self.modifiers = modifiers.state(),
            WindowEvent::KeyboardInput { event, .. } => {
                let key = match event.physical_key {
                    PhysicalKey::Code(key) => key_code(key),
                    PhysicalKey::Unidentified(_) => KeyCode::Unknown,
                };
                let modifiers = key_mods(self.modifiers);
                match event.state {
                    ElementState::Pressed => {
                        self.event_handler()
                            .key_down_event(key, modifiers, event.repeat)
                    }
                    ElementState::Released => self.event_handler().key_up_event(key, modifiers),
                }
                if let Some(text) = event.text {
                    for character in text.chars().filter(|character| !character.is_control()) {
                        self.event_handler()
                            .char_event(character, modifiers, event.repeat);
                    }
                }
            }
            WindowEvent::Ime(winit::event::Ime::Preedit(text, cursor)) => {
                self.event_handler()
                    .on_ime_preedit(&text, cursor.map(|(start, _)| start).unwrap_or(0));
            }
            WindowEvent::Ime(winit::event::Ime::Commit(text)) => {
                self.event_handler().on_ime_commit(Some(&text))
            }
            WindowEvent::DroppedFile(path) => {
                crate::native_display()
                    .lock()
                    .unwrap()
                    .dropped_files
                    .paths
                    .push(path);
                self.event_handler().files_dropped_event();
            }
            WindowEvent::Focused(false) => self.event_handler().window_minimized_event(),
            WindowEvent::Focused(true) => self.event_handler().window_restored_event(),
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.event_handler()
                .raw_mouse_motion(delta.0 as f32, delta.1 as f32);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.process_requests();
        if (!self.conf.platform.blocking_event_loop || self.update_requested)
            && self.window.is_some()
        {
            self.window.as_ref().unwrap().request_redraw();
        }
    }
}

fn mouse_button(button: WinitMouseButton) -> MouseButton {
    match button {
        WinitMouseButton::Left => MouseButton::Left,
        WinitMouseButton::Middle => MouseButton::Middle,
        WinitMouseButton::Right => MouseButton::Right,
        _ => MouseButton::Unknown,
    }
}

fn cursor_icon(icon: CursorIcon) -> WinitCursorIcon {
    match icon {
        CursorIcon::Default => WinitCursorIcon::Default,
        CursorIcon::Help => WinitCursorIcon::Help,
        CursorIcon::Pointer => WinitCursorIcon::Pointer,
        CursorIcon::Wait => WinitCursorIcon::Wait,
        CursorIcon::Crosshair => WinitCursorIcon::Crosshair,
        CursorIcon::Text => WinitCursorIcon::Text,
        CursorIcon::Move => WinitCursorIcon::Move,
        CursorIcon::NotAllowed => WinitCursorIcon::NotAllowed,
        CursorIcon::EWResize => WinitCursorIcon::EwResize,
        CursorIcon::NSResize => WinitCursorIcon::NsResize,
        CursorIcon::NESWResize => WinitCursorIcon::NeswResize,
        CursorIcon::NWSEResize => WinitCursorIcon::NwseResize,
    }
}

fn key_mods(modifiers: ModifiersState) -> KeyMods {
    KeyMods {
        shift: modifiers.shift_key(),
        ctrl: modifiers.control_key(),
        alt: modifiers.alt_key(),
        logo: modifiers.super_key(),
    }
}

fn key_code(key: WinitKeyCode) -> KeyCode {
    use WinitKeyCode::*;
    match key {
        KeyA => KeyCode::A,
        KeyB => KeyCode::B,
        KeyC => KeyCode::C,
        KeyD => KeyCode::D,
        KeyE => KeyCode::E,
        KeyF => KeyCode::F,
        KeyG => KeyCode::G,
        KeyH => KeyCode::H,
        KeyI => KeyCode::I,
        KeyJ => KeyCode::J,
        KeyK => KeyCode::K,
        KeyL => KeyCode::L,
        KeyM => KeyCode::M,
        KeyN => KeyCode::N,
        KeyO => KeyCode::O,
        KeyP => KeyCode::P,
        KeyQ => KeyCode::Q,
        KeyR => KeyCode::R,
        KeyS => KeyCode::S,
        KeyT => KeyCode::T,
        KeyU => KeyCode::U,
        KeyV => KeyCode::V,
        KeyW => KeyCode::W,
        KeyX => KeyCode::X,
        KeyY => KeyCode::Y,
        KeyZ => KeyCode::Z,
        Digit0 => KeyCode::Key0,
        Digit1 => KeyCode::Key1,
        Digit2 => KeyCode::Key2,
        Digit3 => KeyCode::Key3,
        Digit4 => KeyCode::Key4,
        Digit5 => KeyCode::Key5,
        Digit6 => KeyCode::Key6,
        Digit7 => KeyCode::Key7,
        Digit8 => KeyCode::Key8,
        Digit9 => KeyCode::Key9,
        Escape => KeyCode::Escape,
        Enter => KeyCode::Enter,
        Tab => KeyCode::Tab,
        Backspace => KeyCode::Backspace,
        Space => KeyCode::Space,
        ArrowUp => KeyCode::Up,
        ArrowDown => KeyCode::Down,
        ArrowLeft => KeyCode::Left,
        ArrowRight => KeyCode::Right,
        ShiftLeft => KeyCode::LeftShift,
        ShiftRight => KeyCode::RightShift,
        ControlLeft => KeyCode::LeftControl,
        ControlRight => KeyCode::RightControl,
        AltLeft => KeyCode::LeftAlt,
        AltRight => KeyCode::RightAlt,
        SuperLeft => KeyCode::LeftSuper,
        SuperRight => KeyCode::RightSuper,
        F1 => KeyCode::F1,
        F2 => KeyCode::F2,
        F3 => KeyCode::F3,
        F4 => KeyCode::F4,
        F5 => KeyCode::F5,
        F6 => KeyCode::F6,
        F7 => KeyCode::F7,
        F8 => KeyCode::F8,
        F9 => KeyCode::F9,
        F10 => KeyCode::F10,
        F11 => KeyCode::F11,
        F12 => KeyCode::F12,
        Insert => KeyCode::Insert,
        Delete => KeyCode::Delete,
        Home => KeyCode::Home,
        End => KeyCode::End,
        PageUp => KeyCode::PageUp,
        PageDown => KeyCode::PageDown,
        CapsLock => KeyCode::CapsLock,
        _ => KeyCode::Unknown,
    }
}
