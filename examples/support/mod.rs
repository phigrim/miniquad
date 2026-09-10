pub fn configure(conf: &mut miniquad::conf::Conf) {
    #[cfg(not(all(feature = "metal", target_vendor = "apple")))]
    let _ = conf;

    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    if std::env::args().nth(1).as_deref() == Some("metal") {
        conf.platform.prefer_gfx_api = miniquad::conf::GfxApi::Metal;
    }
}
