use std::path::PathBuf;
use image::{RgbaImage, ImageBuffer};
use chrono::Utc;
use url::Url;

pub fn paste_image() -> Vec<PathBuf> {
    let mut clipboard = match arboard::Clipboard::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("\n[Forge] Error getting clipboard: {:?}", e);
            return vec![];
        }
    };
    
    // 1. Try to get image pixels
    if let Ok(image_data) = clipboard.get_image() {
        let width = image_data.width as u32;
        let height = image_data.height as u32;
        if let Some(img) = ImageBuffer::<image::Rgba<u8>, _>::from_raw(width, height, image_data.bytes.into_owned()) {
            let home_dir = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            let images_dir = PathBuf::from(home_dir).join(".local/share/forge/images");
            if std::fs::create_dir_all(&images_dir).is_ok() {
                let filename = format!("forge_paste_{}.png", Utc::now().timestamp_millis());
                let path = images_dir.join(filename);
                if img.save(&path).is_ok() {
                    eprintln!("\n[Forge] Successfully pasted image ({}x{}) from clipboard!", width, height);
                    return vec![path];
                }
            }
        }
    }
    
    // 2. Fallback: Check if clipboard has text that contains file:// URIs (e.g. copied from file manager)
    if let Ok(text) = clipboard.get_text() {
        let mut paths = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with("file://") {
                if let Ok(url) = Url::parse(line) {
                    if let Ok(path) = url.to_file_path() {
                        // verify it's an image extension roughly
                        if let Some(ext) = path.extension() {
                            let ext_str = ext.to_string_lossy().to_lowercase();
                            if matches!(ext_str.as_str(), "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp") {
                                paths.push(path);
                            }
                        }
                    }
                }
            }
        }
        
        if !paths.is_empty() {
            eprintln!("\n[Forge] Successfully pasted {} image file path(s) from clipboard!", paths.len());
            return paths;
        } else {
            // Also check if it's just a raw absolute path to an image
            let line = text.trim();
            if line.starts_with('/') {
                let path = PathBuf::from(line);
                if path.exists() && path.is_file() {
                    if let Some(ext) = path.extension() {
                        let ext_str = ext.to_string_lossy().to_lowercase();
                        if matches!(ext_str.as_str(), "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp") {
                            eprintln!("\n[Forge] Successfully pasted image file path from clipboard!");
                            return vec![path];
                        }
                    }
                }
            }
        }
        
        eprintln!("\n[Forge] Clipboard text does not contain valid image URIs or paths.");
    } else {
        eprintln!("\n[Forge] Clipboard does not contain an image or valid image paths.");
    }
    
    vec![]
}
