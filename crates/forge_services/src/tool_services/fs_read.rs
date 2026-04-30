use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use forge_app::{
    Content, EnvironmentInfra, FileInfoInfra, FileReaderInfra as InfraFsReadService, FsReadService,
    ReadOutput, compute_hash,
};
use forge_domain::{FileInfo, Image};
use regex::Regex;

use crate::range::resolve_range;
use crate::utils::assert_absolute_path;

/// Truncates a line to the maximum length if it exceeds the limit
fn truncate_line(line: &str, max_length: usize) -> String {
    if line.len() > max_length {
        // Use char indices to avoid panicking on unicode boundaries
        let truncated = line
            .char_indices()
            .take_while(|(idx, _)| *idx < max_length)
            .map(|(_, ch)| ch)
            .collect::<String>();
        format!(
            "{}... [truncated, line exceeds {} chars]",
            truncated, max_length
        )
    } else {
        line.to_string()
    }
}

/// Detects the MIME type of a file based on extension and content
fn detect_mime_type(path: &Path, content: &[u8]) -> String {
    // Try infer crate first (checks magic numbers)
    if let Some(file_type) = infer::get(content) {
        return file_type.mime_type().to_string();
    }

    // Fallback to extension-based detection
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| match ext.to_lowercase().as_str() {
            "txt" | "md" | "rs" | "toml" | "yaml" | "yml" | "json" | "js" | "ts" | "py" | "sh" => {
                "text/plain"
            }
            "ipynb" => "application/json",
            "pdf" => "application/pdf",
            "jpg" | "jpeg" => "image/jpeg",
            "png" => "image/png",
            "gif" => "image/gif",
            "webp" => "image/webp",
            _ => "text/plain", // Default to text
        })
        .unwrap_or("text/plain")
        .to_string()
}

/// Checks if a MIME type represents visual content (images or PDFs)
fn is_visual_content(mime_type: &str) -> bool {
    mime_type.starts_with("image/") || mime_type == "application/pdf"
}

/// Fetches local Rust dependencies referenced by `mod` declarations and
/// `use crate::` imports in the given file content.
///
/// Only processes `.rs` files. Resolves one level deep (no recursion).
/// Silently skips files that cannot be read or do not exist.
///
/// # Arguments
/// * `content` - The source file content to scan for dependency references
/// * `file_path` - Absolute path of the source file (used to resolve relative
///   `mod` paths and find the crate root)
async fn fetch_local_dependencies(content: &str, file_path: &Path) -> Vec<(PathBuf, String)> {
    if file_path.extension().and_then(|e| e.to_str()) != Some("rs") {
        return Vec::new();
    }

    let mod_re = match Regex::new(r"(?m)^\s*(?:pub\s+)?mod\s+([a-zA-Z0-9_]+)\s*;") {
        Ok(re) => re,
        Err(_) => return Vec::new(),
    };
    let use_re = match Regex::new(r"(?m)^\s*(?:pub\s+)?use\s+crate::([a-zA-Z0-9_:]+)") {
        Ok(re) => re,
        Err(_) => return Vec::new(),
    };

    let mut candidate_paths: Vec<PathBuf> = Vec::new();

    // Resolve `mod name;` → sibling file or sibling directory with mod.rs
    if let Some(parent) = file_path.parent() {
        for cap in mod_re.captures_iter(content) {
            if let Some(m) = cap.get(1) {
                let mod_name = m.as_str();
                candidate_paths.push(parent.join(format!("{mod_name}.rs")));
                candidate_paths.push(parent.join(mod_name).join("mod.rs"));
            }
        }
    }

    // Find crate root (directory containing Cargo.toml) to resolve
    // `use crate::` paths relative to `src/`.
    let src_dir = {
        let mut found = None;
        let mut current = file_path.parent();
        while let Some(dir) = current {
            if tokio::fs::metadata(dir.join("Cargo.toml")).await.is_ok() {
                found = Some(dir.join("src"));
                break;
            }
            current = dir.parent();
        }
        found
    };

    if let Some(ref src) = src_dir {
        for cap in use_re.captures_iter(content) {
            if let Some(m) = cap.get(1) {
                let use_path = m.as_str();
                let parts: Vec<&str> = use_path.split("::").collect();

                // Try progressively shorter prefixes: crate::a::b::c → a/b/c.rs,
                // a/b/c/mod.rs, then a/b.rs, a/b/mod.rs, then a.rs, a/mod.rs
                for depth in (1..=parts.len()).rev() {
                    if let Some(prefix_parts) = parts.get(..depth) {
                        let sub_path = prefix_parts.join("/");
                        candidate_paths.push(src.join(format!("{sub_path}.rs")));
                        candidate_paths.push(src.join(&sub_path).join("mod.rs"));
                    }
                }
            }
        }
    }

    // Deduplicate candidate paths while preserving order
    let mut seen = HashSet::new();
    let mut unique_paths: Vec<PathBuf> = Vec::new();
    for p in candidate_paths {
        if seen.insert(p.clone()) {
            unique_paths.push(p);
        }
    }

    // Read files that actually exist, collecting (path, content) pairs
    let mut deps: Vec<(PathBuf, String)> = Vec::new();
    let mut read_canonical: HashSet<PathBuf> = HashSet::new();
    for p in unique_paths {
        // Canonicalize to avoid reading the same file via different paths
        // (e.g. foo.rs and foo/mod.rs both existing is unlikely, but
        // we also want to skip the source file itself).
        let canonical = match tokio::fs::canonicalize(&p).await {
            Ok(c) => c,
            Err(_) => continue, // file doesn't exist
        };
        if canonical == tokio::fs::canonicalize(file_path).await.unwrap_or_default() {
            continue; // skip self
        }
        if !read_canonical.insert(canonical) {
            continue; // already read via a different path
        }
        if let Ok(c) = tokio::fs::read_to_string(&p).await {
            deps.push((p, c));
        }
    }

    deps
}

/// Validates that file size does not exceed the maximum allowed file size.
///
/// # Arguments
/// * `infra` - The infrastructure instance providing file metadata services
/// * `path` - The file path to check
/// * `max_file_size` - Maximum allowed file size in bytes
///
/// # Returns
/// * `Ok(())` if file size is within limits
/// * `Err(anyhow::Error)` if file exceeds max_file_size
pub async fn assert_file_size<F: FileInfoInfra>(
    infra: &F,
    path: &Path,
    max_file_size: u64,
) -> anyhow::Result<()> {
    let file_size = infra.file_size(path).await?;
    if file_size > max_file_size {
        return Err(anyhow::anyhow!(
            "File size ({file_size} bytes) exceeds the maximum allowed size of {max_file_size} bytes"
        ));
    }
    Ok(())
}

/// Reads file contents from the specified absolute path. Ideal for analyzing
/// code, configuration files, documentation, or textual data. Returns the
/// content as a string. For files larger than 2,000 lines, the tool
/// automatically returns only the first 2,000 lines. You should always rely
/// on this default behavior and avoid specifying custom ranges unless
/// absolutely necessary. If needed, specify a range with the start_line and
/// end_line parameters, ensuring the total range does not exceed 2,000 lines.
/// Specifying a range exceeding this limit will result in an error. Binary
/// files are automatically detected and rejected.
pub struct ForgeFsRead<F> {
    infra: Arc<F>,
}

impl<F> ForgeFsRead<F> {
    pub fn new(infra: Arc<F>) -> Self {
        Self { infra }
    }
}

#[async_trait::async_trait]
impl<F: FileInfoInfra + EnvironmentInfra<Config = forge_config::ForgeConfig> + InfraFsReadService>
    FsReadService for ForgeFsRead<F>
{
    async fn read(
        &self,
        path: String,
        start_line: Option<u64>,
        end_line: Option<u64>,
    ) -> anyhow::Result<ReadOutput> {
        let path = Path::new(&path);
        assert_absolute_path(path)?;

        let config = self.infra.get_config()?;

        // Validate with the larger limit initially since we don't know file type yet
        let initial_size_limit = config.max_file_size_bytes.max(config.max_image_size_bytes);
        assert_file_size(&*self.infra, path, initial_size_limit).await?;

        // Read file content to detect MIME type
        let raw_content = self
            .infra
            .read(path)
            .await
            .with_context(|| format!("Failed to read file from {}", path.display()))?;

        // Detect MIME type
        let mime_type = detect_mime_type(path, &raw_content);

        // Handle visual content (PDFs and images)
        if is_visual_content(&mime_type) {
            // Validate against image-specific size limit (may be different from
            // max_file_size)
            assert_file_size(&*self.infra, path, config.max_image_size_bytes)
                .await
                .with_context(|| {
                    if mime_type == "application/pdf" {
                        "PDF exceeds size limit. Use a smaller PDF or increase FORGE_MAX_IMAGE_SIZE."
                    } else {
                        "Image exceeds size limit. Compress the image or increase FORGE_MAX_IMAGE_SIZE."
                    }
                })?;

            // Convert to base64 image
            let image = Image::new_bytes(raw_content, mime_type.clone());
            let hash = compute_hash(image.url());

            return Ok(ReadOutput {
                content: Content::image(image),
                info: FileInfo::new(0, 0, 0, hash),
            });
        }

        // Handle text content (including Jupyter notebooks)
        // File size already validated above

        let (start_line, end_line) = resolve_range(start_line, end_line, config.max_read_lines);

        // Convert bytes to UTF-8 string
        let full_content = String::from_utf8(raw_content)
            .with_context(|| format!("Failed to read file as UTF-8 from {}", path.display()))?;

        let hash = compute_hash(&full_content);

        // Now extract the requested range from the content we already have
        let lines: Vec<&str> = full_content.lines().collect();
        let total_lines = lines.len() as u64;

        // Convert to 0-based indexing and clamp to valid range
        let start_pos = start_line
            .saturating_sub(1)
            .min(total_lines.saturating_sub(1));
        let end_pos = end_line
            .saturating_sub(1)
            .min(total_lines.saturating_sub(1));

        // Extract requested lines
        let mut content = if start_pos == 0 && end_pos >= total_lines.saturating_sub(1) {
            // Return full content with line truncation
            lines
                .iter()
                .map(|line| truncate_line(line, config.max_line_chars))
                .collect::<Vec<_>>()
                .join("\n")
        } else if total_lines == 0 {
            String::new()
        } else {
            // Return range with line truncation
            let start_pos = usize::try_from(start_pos).unwrap_or(usize::MAX);
            let end_pos = usize::try_from(end_pos).unwrap_or(usize::MAX);
            lines
                .get(start_pos..=end_pos)
                .map(|slice| {
                    slice
                        .iter()
                        .map(|line| truncate_line(line, config.max_line_chars))
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .unwrap_or_default()
        };

        // Append local Rust dependencies (mod declarations, use crate:: imports)
        let deps = fetch_local_dependencies(&full_content, path).await;
        for (dep_path, dep_content) in deps {
            content.push_str(&format!(
                "\n\n--- Local Dependency: {} ---\n{}",
                dep_path.display(),
                dep_content
            ));
        }

        let file_info = FileInfo::new(start_line, end_line, total_lines, hash);

        Ok(ReadOutput { content: Content::file(content), info: file_info })
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use tempfile::NamedTempFile;
    use tokio::fs;

    use super::*;
    use crate::attachment::tests::MockFileService;

    // Helper to create a temporary file with specific content size
    async fn create_test_file_with_size(size: usize) -> anyhow::Result<NamedTempFile> {
        let file = NamedTempFile::new()?;
        let content = "x".repeat(size);
        fs::write(file.path(), content).await?;
        Ok(file)
    }

    #[tokio::test]
    async fn test_assert_file_size_within_limit() {
        let fixture = create_test_file_with_size(13).await.unwrap();
        let infra = MockFileService::new();
        // Add the file to the mock infrastructure
        infra.add_file(fixture.path().to_path_buf(), "x".repeat(13));
        let actual = assert_file_size(&infra, fixture.path(), 20u64).await;
        assert!(actual.is_ok());
    }

    #[tokio::test]
    async fn test_assert_file_size_exactly_at_limit() {
        let fixture = create_test_file_with_size(6).await.unwrap();
        let infra = MockFileService::new();
        infra.add_file(fixture.path().to_path_buf(), "x".repeat(6));
        let actual = assert_file_size(&infra, fixture.path(), 6u64).await;
        assert!(actual.is_ok());
    }

    #[tokio::test]
    async fn test_assert_file_size_exceeds_limit() {
        let fixture = create_test_file_with_size(45).await.unwrap();
        let infra = MockFileService::new();
        infra.add_file(fixture.path().to_path_buf(), "x".repeat(45));
        let actual = assert_file_size(&infra, fixture.path(), 10u64).await;
        assert!(actual.is_err());
    }

    #[tokio::test]
    async fn test_assert_file_size_empty_content() {
        let fixture = create_test_file_with_size(0).await.unwrap();
        let infra = MockFileService::new();
        infra.add_file(fixture.path().to_path_buf(), "".to_string());
        let actual = assert_file_size(&infra, fixture.path(), 100u64).await;
        assert!(actual.is_ok());
    }

    #[tokio::test]
    async fn test_assert_file_size_zero_limit() {
        let fixture = create_test_file_with_size(1).await.unwrap();
        let infra = MockFileService::new();
        infra.add_file(fixture.path().to_path_buf(), "x".to_string());
        let actual = assert_file_size(&infra, fixture.path(), 0u64).await;
        assert!(actual.is_err());
    }

    #[tokio::test]
    async fn test_assert_file_size_large_content() {
        let fixture = create_test_file_with_size(1000).await.unwrap();
        let infra = MockFileService::new();
        infra.add_file(fixture.path().to_path_buf(), "x".repeat(1000));
        let actual = assert_file_size(&infra, fixture.path(), 999u64).await;
        assert!(actual.is_err());
    }

    #[tokio::test]
    async fn test_assert_file_size_large_content_within_limit() {
        let fixture = create_test_file_with_size(1000).await.unwrap();
        let infra = MockFileService::new();
        infra.add_file(fixture.path().to_path_buf(), "x".repeat(1000));
        let actual = assert_file_size(&infra, fixture.path(), 1000u64).await;
        assert!(actual.is_ok());
    }

    #[tokio::test]
    async fn test_assert_file_size_unicode_content() {
        let file = NamedTempFile::new().unwrap();
        fs::write(file.path(), "🚀🚀🚀").await.unwrap(); // Each emoji is 4 bytes in UTF-8 = 12 bytes total
        let infra = MockFileService::new();
        infra.add_file(file.path().to_path_buf(), "🚀🚀🚀".to_string());
        let actual = assert_file_size(&infra, file.path(), 12u64).await;
        assert!(actual.is_ok());
    }

    #[tokio::test]
    async fn test_assert_file_size_unicode_content_exceeds() {
        let file = NamedTempFile::new().unwrap();
        fs::write(file.path(), "🚀🚀🚀🚀").await.unwrap(); // 4 emojis = 16 bytes, exceeds 12 byte limit
        let infra = MockFileService::new();
        infra.add_file(file.path().to_path_buf(), "🚀🚀🚀🚀".to_string());
        let actual = assert_file_size(&infra, file.path(), 12u64).await;
        assert!(actual.is_err());
    }

    #[tokio::test]
    async fn test_assert_file_size_error_message() {
        let file = NamedTempFile::new().unwrap();
        fs::write(file.path(), "too long content").await.unwrap(); // 16 bytes
        let infra = MockFileService::new();
        infra.add_file(file.path().to_path_buf(), "too long content".to_string());
        let actual = assert_file_size(&infra, file.path(), 5u64).await;
        let expected = "File size (16 bytes) exceeds the maximum allowed size of 5 bytes";
        assert!(actual.is_err());
        assert_eq!(actual.unwrap_err().to_string(), expected);
    }

    #[test]
    fn test_detect_mime_type_for_text_files() {
        let path = Path::new("test.txt");
        let content = b"Hello, world!";
        let actual = detect_mime_type(path, content);
        assert_eq!(actual, "text/plain");
    }

    #[test]
    fn test_detect_mime_type_for_ipynb() {
        let path = Path::new("notebook.ipynb");
        let content = b"{\"cells\": []}";
        let actual = detect_mime_type(path, content);
        assert_eq!(actual, "application/json");
    }

    #[test]
    fn test_detect_mime_type_for_png() {
        let path = Path::new("image.png");
        // PNG magic number
        let content = b"\x89PNG\r\n\x1a\n";
        let actual = detect_mime_type(path, content);
        assert_eq!(actual, "image/png");
    }

    #[test]
    fn test_detect_mime_type_for_pdf_with_magic() {
        let path = Path::new("document.pdf");
        // PDF magic number
        let content = b"%PDF-1.4";
        let actual = detect_mime_type(path, content);
        assert_eq!(actual, "application/pdf");
    }

    #[test]
    fn test_detect_mime_type_for_jpeg() {
        let path = Path::new("photo.jpg");
        // JPEG magic number
        let content = b"\xFF\xD8\xFF";
        let actual = detect_mime_type(path, content);
        assert_eq!(actual, "image/jpeg");
    }

    #[test]
    fn test_is_visual_content_for_images() {
        assert!(is_visual_content("image/png"));
        assert!(is_visual_content("image/jpeg"));
        assert!(is_visual_content("image/gif"));
        assert!(is_visual_content("image/webp"));
    }

    #[test]
    fn test_is_visual_content_for_pdf() {
        assert!(is_visual_content("application/pdf"));
    }

    #[test]
    fn test_is_visual_content_for_text() {
        assert!(!is_visual_content("text/plain"));
        assert!(!is_visual_content("application/json"));
        assert!(!is_visual_content("text/html"));
    }

    #[test]
    fn test_truncate_line_short_line() {
        let line = "short line";
        let actual = truncate_line(line, 100);
        assert_eq!(actual, "short line");
    }

    #[test]
    fn test_truncate_line_exact_length() {
        let line = "exactly 17 chars!";
        assert_eq!(line.len(), 17);
        let actual = truncate_line(line, 17);
        assert_eq!(actual, "exactly 17 chars!");
    }

    #[test]
    fn test_truncate_line_long_line() {
        let line = "this is a very long line that exceeds the maximum length";
        let actual = truncate_line(line, 20);
        assert_eq!(actual.len(), 58); // 20 chars + "... [truncated, line exceeds 20 chars]"
        assert!(actual.starts_with("this is a very long"));
        assert!(actual.contains("[truncated"));
        assert!(!actual.contains("exceeds the maximum length"));
    }

    #[test]
    fn test_truncate_line_empty() {
        let line = "";
        let actual = truncate_line(line, 100);
        assert_eq!(actual, "");
    }

    #[test]
    fn test_truncate_line_unicode() {
        let line = "🚀🚀🚀🚀🚀"; // Each emoji is 4 chars, total 20
        let actual = truncate_line(line, 12);
        // Should truncate at byte boundary, not character boundary
        println!("{}", actual);
        assert_eq!(actual.len(), 50); // 12 bytes + truncation message
        assert!(actual.contains("truncated"));
    }

    // ── E2E: fetch_local_dependencies ─────────────────────────────────────────

    /// Creates a minimal fake crate under `root`:
    ///   root/
    ///     Cargo.toml
    ///     src/
    ///       main.rs   ← the file we "read"
    ///       utils.rs  ← declared via `mod utils;`
    ///       models.rs ← imported via `use crate::models::Foo;`
    async fn fixture_rust_crate(root: &std::path::Path) {
        fs::create_dir_all(root.join("src")).await.unwrap();

        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"test-crate\"\n",
        )
        .await
        .unwrap();

        fs::write(
            root.join("src/main.rs"),
            "mod utils;\nuse crate::models::Foo;\nfn main() {}\n",
        )
        .await
        .unwrap();

        fs::write(root.join("src/utils.rs"), "pub fn helper() {}\n")
            .await
            .unwrap();

        fs::write(root.join("src/models.rs"), "pub struct Foo;\n")
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_fetch_local_dependencies_injects_mod_and_use_crate_files() {
        let dir = tempfile::tempdir().unwrap();
        fixture_rust_crate(dir.path()).await;
        let main_path = dir.path().join("src/main.rs");
        let content = fs::read_to_string(&main_path).await.unwrap();

        // Execute
        let actual = fetch_local_dependencies(&content, &main_path).await;

        // Both utils.rs (mod utils;) and models.rs (use crate::models) should be found
        let actual_names: Vec<String> = actual
            .iter()
            .map(|(p, _)| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        let mut actual_names = actual_names;
        actual_names.sort();

        let expected = vec!["models.rs".to_string(), "utils.rs".to_string()];
        assert_eq!(actual_names, expected);
    }

    #[tokio::test]
    async fn test_fetch_local_dependencies_contents_included() {
        let dir = tempfile::tempdir().unwrap();
        fixture_rust_crate(dir.path()).await;
        let main_path = dir.path().join("src/main.rs");
        let content = fs::read_to_string(&main_path).await.unwrap();

        // Execute
        let actual = fetch_local_dependencies(&content, &main_path).await;

        // Verify actual content of dependencies is correct
        let utils_content = actual
            .iter()
            .find(|(p, _)| p.ends_with("utils.rs"))
            .map(|(_, c)| c.as_str())
            .expect("utils.rs should be present");

        let models_content = actual
            .iter()
            .find(|(p, _)| p.ends_with("models.rs"))
            .map(|(_, c)| c.as_str())
            .expect("models.rs should be present");

        assert_eq!(utils_content, "pub fn helper() {}\n");
        assert_eq!(models_content, "pub struct Foo;\n");
    }

    #[tokio::test]
    async fn test_fetch_local_dependencies_skips_non_rs_files() {
        let dir = tempfile::tempdir().unwrap();

        // Write a non-.rs file
        let js_path = dir.path().join("script.js");
        fs::write(&js_path, "console.log('hello');\n")
            .await
            .unwrap();

        let content = "mod utils;\n";
        let actual = fetch_local_dependencies(content, &js_path).await;

        // Non-.rs files should return empty deps
        assert!(
            actual.is_empty(),
            "Non-.rs files should have no dependencies"
        );
    }

    #[tokio::test]
    async fn test_fetch_local_dependencies_does_not_include_self() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("src")).await.unwrap();
        fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"x\"\n")
            .await
            .unwrap();

        // lib.rs imports itself via `use crate::lib` (unlikely, but tested for safety)
        let lib_path = dir.path().join("src/lib.rs");
        fs::write(
            &lib_path,
            "use crate::lib::something;\npub fn something() {}\n",
        )
        .await
        .unwrap();

        let content = fs::read_to_string(&lib_path).await.unwrap();
        let actual = fetch_local_dependencies(&content, &lib_path).await;

        // lib.rs itself must not appear in its own deps
        let contains_self = actual.iter().any(|(p, _)| {
            p.canonicalize()
                .ok()
                .zip(lib_path.canonicalize().ok())
                .map(|(a, b)| a == b)
                .unwrap_or(false)
        });
        assert!(
            !contains_self,
            "A file must not inject itself as a dependency"
        );
    }

    #[tokio::test]
    async fn test_fetch_local_dependencies_missing_files_skipped() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("src")).await.unwrap();
        fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"x\"\n")
            .await
            .unwrap();

        let main_path = dir.path().join("src/main.rs");
        // References a module that doesn't exist on disk
        let content = "mod nonexistent_module;\nfn main() {}\n";
        fs::write(&main_path, content).await.unwrap();

        let actual = fetch_local_dependencies(content, &main_path).await;

        // Missing files should be silently skipped - empty result
        assert!(
            actual.is_empty(),
            "Missing dependency files should be silently skipped"
        );
    }
}
