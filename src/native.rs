#![allow(dead_code)]

#[cfg(not(target_os = "android"))]
use std::sync::mpsc;

#[derive(Default)]
pub(crate) struct DroppedFiles {
    pub paths: Vec<std::path::PathBuf>,
    pub bytes: Vec<Vec<u8>>,
}

#[cfg(any(all(target_os = "android", feature = "wgpu"), test))]
#[derive(Clone, Debug)]
pub(crate) struct SurfaceLifecycle<T> {
    pub generation: u64,
    pub target: Option<T>,
    pub width: u32,
    pub height: u32,
    pub ready: bool,
}

#[cfg(any(all(target_os = "android", feature = "wgpu"), test))]
impl<T> Default for SurfaceLifecycle<T> {
    fn default() -> Self {
        Self {
            generation: 0,
            target: None,
            width: 0,
            height: 0,
            ready: false,
        }
    }
}

#[cfg(any(all(target_os = "android", feature = "wgpu"), test))]
impl<T> SurfaceLifecycle<T> {
    pub fn created(&mut self, target: T) {
        self.generation = self.generation.wrapping_add(1);
        self.target = Some(target);
        self.ready = false;
    }

    pub fn destroyed(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.target = None;
        self.ready = false;
    }

    pub fn resized(&mut self, width: i32, height: i32) {
        self.width = width.max(0) as u32;
        self.height = height.max(0) as u32;
        self.ready = self.target.is_some() && width > 0 && height > 0;
    }
}

#[cfg(test)]
mod surface_lifecycle_tests {
    use super::SurfaceLifecycle;

    #[test]
    fn surface_requires_a_window_and_positive_size() {
        let mut state = SurfaceLifecycle::<u8>::default();
        state.resized(1280, 720);
        assert!(!state.ready);

        state.created(1);
        assert!(!state.ready);
        assert_eq!(state.generation, 1);

        state.resized(0, 720);
        assert!(!state.ready);
        state.resized(1280, 720);
        assert!(state.ready);
        assert_eq!(state.generation, 1, "resize must not replace the surface");
    }

    #[test]
    fn destroy_and_recreate_advance_generation_and_replace_target() {
        let mut state = SurfaceLifecycle::default();
        state.created("first");
        state.resized(640, 480);
        assert!(state.ready);

        state.destroyed();
        assert_eq!(state.generation, 2);
        assert!(state.target.is_none());
        assert!(!state.ready);

        state.created("second");
        assert_eq!(state.generation, 3);
        assert_eq!(state.target, Some("second"));
        assert!(!state.ready);
        state.resized(800, 600);
        assert!(state.ready);
    }
}

pub(crate) struct NativeDisplayData {
    pub screen_width: i32,
    pub screen_height: i32,
    pub screen_position: (u32, u32),
    pub ime_enabled: bool,
    pub dpi_scale: f32,
    pub high_dpi: bool,
    pub quit_requested: bool,
    pub quit_ordered: bool,
    #[cfg(target_os = "android")]
    pub native_requests: Box<dyn Fn(Request) + Send>,
    #[cfg(not(target_os = "android"))]
    pub native_requests: mpsc::Sender<Request>,
    pub clipboard: Box<dyn Clipboard>,
    pub dropped_files: DroppedFiles,
    pub blocking_event_loop: bool,
    pub sample_count: u32,
    pub swap_interval: Option<i32>,

    pub gfx_api: crate::conf::GfxApi,
    #[cfg(feature = "wgpu")]
    pub wgpu_backend: crate::conf::WgpuBackend,
    #[cfg(all(target_os = "android", feature = "wgpu"))]
    pub android_surface_source: std::sync::Arc<crate::native::android::AndroidSurfaceSource>,

    #[cfg(target_vendor = "apple")]
    pub view: crate::native::apple::frameworks::ObjcId,
    #[cfg(target_os = "ios")]
    pub view_ctrl: crate::native::apple::frameworks::ObjcId,
}
#[cfg(target_vendor = "apple")]
unsafe impl Send for NativeDisplayData {}
#[cfg(target_vendor = "apple")]
unsafe impl Sync for NativeDisplayData {}

impl NativeDisplayData {
    pub fn new(
        screen_width: i32,
        screen_height: i32,
        #[cfg(target_os = "android")] native_requests: Box<dyn Fn(Request) + Send>,
        #[cfg(not(target_os = "android"))] native_requests: mpsc::Sender<Request>,
        clipboard: Box<dyn Clipboard>,
    ) -> NativeDisplayData {
        NativeDisplayData {
            screen_width,
            screen_height,
            screen_position: (0, 0),
            ime_enabled: false,
            dpi_scale: 1.,
            high_dpi: false,
            quit_requested: false,
            quit_ordered: false,
            native_requests,
            clipboard,
            dropped_files: Default::default(),
            blocking_event_loop: false,
            sample_count: 1,
            swap_interval: None,
            gfx_api: crate::conf::GfxApi::default(),
            #[cfg(feature = "wgpu")]
            wgpu_backend: crate::conf::WgpuBackend::default(),
            #[cfg(all(target_os = "android", feature = "wgpu"))]
            android_surface_source: std::sync::Arc::default(),
            #[cfg(target_vendor = "apple")]
            view: std::ptr::null_mut(),
            #[cfg(target_os = "ios")]
            view_ctrl: std::ptr::null_mut(),
        }
    }
}

#[derive(Debug)]
pub(crate) enum Request {
    ScheduleUpdate,
    SetCursorGrab(bool),
    ShowMouse(bool),
    SetMouseCursor(crate::CursorIcon),
    SetWindowSize {
        new_width: u32,
        new_height: u32,
    },
    SetWindowPosition {
        new_x: u32,
        new_y: u32,
    },
    SetFullscreen(bool),
    ShowKeyboard(bool),
    SetImePosition {
        x: i32,
        y: i32,
    },
    SetImeEnabled(bool),
    UpdateTextInputState {
        text: String,
        selection_start: usize,
        selection_end: usize,
        is_password: bool,
        is_multiline: bool,
        element_id: u64,
        max_length: usize,
    },
}

pub trait Clipboard: Send + Sync {
    fn get(&mut self) -> Option<String>;
    fn set(&mut self, string: &str);
}

pub mod module;

#[cfg(all(
    feature = "wgpu",
    any(target_os = "windows", target_os = "linux", target_os = "macos")
))]
pub mod winit;

#[cfg(target_env = "ohos")]
pub mod ohos;

#[cfg(target_env = "ohos")]
pub use ohos::*;

#[cfg(all(target_os = "linux", not(target_env = "ohos")))]
pub mod linux_x11;

#[cfg(all(target_os = "linux", not(target_env = "ohos")))]
pub mod linux_wayland;

#[cfg(target_os = "android")]
pub mod android;

#[cfg(target_os = "windows")]
pub mod windows;

#[cfg(target_os = "android")]
pub use android::*;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod apple;

#[cfg(target_os = "macos")]
pub mod macos;

#[cfg(target_os = "ios")]
pub mod ios;

#[cfg(any(target_os = "android", target_os = "linux"))]
pub mod egl;

// there is no glGetProcAddr on webgl, so its impossible to make "gl" module work
// on macos.. well, there is, but way easier to just statically link to gl
#[cfg(not(target_arch = "wasm32"))]
pub mod gl;

#[cfg(target_arch = "wasm32")]
pub use wasm::webgl as gl;

pub mod query_stab;
