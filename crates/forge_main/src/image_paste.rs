use std::path::{Path, PathBuf};
use std::process::Command;

use chrono::Utc;
use image::ImageBuffer;
use url::Url;

pub fn paste_image() -> Vec<PathBuf> {
    let Some(images_dir) = get_images_dir() else {
        eprintln!("\n[Forge] Failed to create images directory.");
        return Vec::new();
    };

    if let Some(path) = paste_external_png(&images_dir, ClipboardTool::WlPaste) {
        return vec![path];
    }
    if let Some(path) = paste_external_png(&images_dir, ClipboardTool::Xclip) {
        return vec![path];
    }

    let mut clipboard = match arboard::Clipboard::new() {
        Ok(clipboard) => clipboard,
        Err(error) => {
            eprintln!("\n[Forge] Error getting clipboard: {error:?}");
            return Vec::new();
        }
    };

    if let Some(path) = paste_clipboard_pixels(&images_dir, &mut clipboard) {
        return vec![path];
    }

    paste_clipboard_paths(&mut clipboard)
}

enum ClipboardTool {
    WlPaste,
    Xclip,
}

fn get_images_dir() -> Option<PathBuf> {
    let images_dir = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(".local/share/forge/images");
    std::fs::create_dir_all(&images_dir).ok()?;
    Some(images_dir)
}

fn new_image_path(images_dir: &Path) -> PathBuf {
    images_dir.join(format!("forge_paste_{}.png", Utc::now().timestamp_millis()))
}

fn paste_external_png(images_dir: &Path, tool: ClipboardTool) -> Option<PathBuf> {
    let mut command = match tool {
        ClipboardTool::WlPaste => {
            let mut command = Command::new("wl-paste");
            command.args(["-t", "image/png"]);
            command
        }
        ClipboardTool::Xclip => {
            let mut command = Command::new("xclip");
            command.args(["-selection", "clipboard", "-t", "image/png", "-o"]);
            command
        }
    };
    let output = command.output().ok()?;
    if !output.status.success() || output.stdout.is_empty() {
        return None;
    }

    let path = new_image_path(images_dir);
    std::fs::write(&path, output.stdout).ok()?;
    Some(path)
}

fn paste_clipboard_pixels(
    images_dir: &Path,
    clipboard: &mut arboard::Clipboard,
) -> Option<PathBuf> {
    let image_data = clipboard.get_image().ok()?;
    let width = u32::try_from(image_data.width).ok()?;
    let height = u32::try_from(image_data.height).ok()?;
    let img =
        ImageBuffer::<image::Rgba<u8>, _>::from_raw(width, height, image_data.bytes.into_owned())?;

    let path = new_image_path(images_dir);
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
        .filter_map(|line| parse_image_path(unquote(line.trim())))
        .collect();

    if !paths.is_empty() {
        eprintln!(
            "\n[Forge] Successfully pasted {} image file path(s) from clipboard.",
            paths.len()
        );
        return paths;
    }

    parse_image_path(unquote(text.trim())).into_iter().collect()
}

fn unquote(value: &str) -> &str {
    value
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .or_else(|| {
            value
                .strip_prefix('\'')
                .and_then(|rest| rest.strip_suffix('\''))
        })
        .unwrap_or(value)
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
