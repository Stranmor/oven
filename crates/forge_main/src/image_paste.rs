use std::path::{Path, PathBuf};

use chrono::Utc;
use image::ImageBuffer;
use url::Url;

pub fn paste_image() -> Vec<PathBuf> {
    let mut clipboard = match arboard::Clipboard::new() {
        Ok(clipboard) => clipboard,
        Err(error) => {
            eprintln!("\n[Forge] Error getting clipboard: {error:?}");
            return Vec::new();
        }
    };

    if let Some(path) = paste_clipboard_pixels(&mut clipboard) {
        return vec![path];
    }

    paste_clipboard_paths(&mut clipboard)
}

fn paste_clipboard_pixels(clipboard: &mut arboard::Clipboard) -> Option<PathBuf> {
    let image_data = clipboard.get_image().ok()?;
    let width = u32::try_from(image_data.width).ok()?;
    let height = u32::try_from(image_data.height).ok()?;
    let img =
        ImageBuffer::<image::Rgba<u8>, _>::from_raw(width, height, image_data.bytes.into_owned())?;

    let images_dir = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(".local/share/forge/images");
    std::fs::create_dir_all(&images_dir).ok()?;

    let filename = format!("forge_paste_{}.png", Utc::now().timestamp_millis());
    let path = images_dir.join(filename);
    img.save(&path).ok()?;

    eprintln!("\n[Forge] Successfully pasted image ({width}x{height}) from clipboard.");
    Some(path)
}

fn paste_clipboard_paths(clipboard: &mut arboard::Clipboard) -> Vec<PathBuf> {
    let Ok(text) = clipboard.get_text() else {
        eprintln!("\n[Forge] Clipboard does not contain an image or valid image paths.");
        return Vec::new();
    };

    let paths: Vec<PathBuf> = text
        .lines()
        .filter_map(|line| parse_image_path(line.trim()))
        .collect();

    if !paths.is_empty() {
        eprintln!(
            "\n[Forge] Successfully pasted {} image file path(s) from clipboard.",
            paths.len()
        );
        return paths;
    }

    parse_image_path(text.trim()).into_iter().collect()
}

fn parse_image_path(value: &str) -> Option<PathBuf> {
    let path = if value.starts_with("file://") {
        Url::parse(value).ok()?.to_file_path().ok()?
    } else if value.starts_with('/') {
        PathBuf::from(value)
    } else {
        return None;
    };

    is_image_file(&path).then_some(path)
}

fn is_image_file(path: &Path) -> bool {
    let Some(ext) = path.extension() else {
        return false;
    };
    let ext = ext.to_string_lossy().to_lowercase();
    path.is_file()
        && matches!(
            ext.as_str(),
            "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp"
        )
}
