use chrono::Utc;
use image::{ImageBuffer, RgbaImage};
use std::path::PathBuf;

pub fn paste_image() -> Option<PathBuf> {
    let mut clipboard = arboard::Clipboard::new().ok()?;
    let image_data = clipboard.get_image().ok()?;

    // Convert to image::RgbaImage
    let width = image_data.width as u32;
    let height = image_data.height as u32;
    let img: RgbaImage = ImageBuffer::from_raw(width, height, image_data.bytes.into_owned())?;

    let filename = format!("forge_paste_{}.png", Utc::now().timestamp_millis());
    let path = std::env::temp_dir().join(filename);

    img.save(&path).ok()?;

    // We can return absolute path. Or return a relative path if inside env.cwd
    // For now, absolute path is fine.
    Some(path)
}
