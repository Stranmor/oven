use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use forge_app::{FileReaderInfra, PdfRenderInfra};
use futures::{StreamExt, stream};
use tokio::process::Command;
use uuid::Uuid;

#[derive(Clone)]
pub struct ForgeFileReadService;

impl Default for ForgeFileReadService {
    fn default() -> Self {
        Self
    }
}

impl ForgeFileReadService {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl FileReaderInfra for ForgeFileReadService {
    async fn read_utf8(&self, path: &Path) -> Result<String> {
        forge_fs::ForgeFS::read_utf8(path).await
    }

    fn read_batch_utf8(
        &self,
        batch_size: usize,
        paths: Vec<PathBuf>,
    ) -> impl futures::Stream<Item = (PathBuf, anyhow::Result<String>)> + Send {
        let batches: Vec<Vec<PathBuf>> = paths
            .chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        stream::iter(batches)
            .then(move |batch| async move {
                let futures = batch.into_iter().map(|path| async move {
                    let result = self.read_utf8(&path).await;
                    (path, result)
                });

                futures::future::join_all(futures).await
            })
            .flat_map(stream::iter)
    }

    async fn read(&self, path: &Path) -> Result<Vec<u8>> {
        forge_fs::ForgeFS::read(path).await
    }

    async fn range_read_utf8(
        &self,
        path: &Path,
        start_line: u64,
        end_line: u64,
    ) -> Result<(String, forge_domain::FileInfo)> {
        forge_fs::ForgeFS::read_range_utf8(path, start_line, end_line).await
    }
}

#[async_trait::async_trait]
impl PdfRenderInfra for ForgeFileReadService {
    async fn render_pdf_first_page_to_png(
        &self,
        path: &Path,
        max_image_size_bytes: u64,
    ) -> anyhow::Result<Vec<u8>> {
        let render_dir = std::env::temp_dir().join(format!("forge-pdf-render-{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&render_dir)
            .await
            .with_context(|| {
                format!(
                    "Failed to create PDF render directory {}",
                    render_dir.display()
                )
            })?;

        let result = async {
            let output_prefix = render_dir.join("page");
            let output = Command::new("pdftoppm")
                .arg("-png")
                .arg("-singlefile")
                .arg("-f")
                .arg("1")
                .arg("-l")
                .arg("1")
                .arg("-r")
                .arg("144")
                .arg(path)
                .arg(&output_prefix)
                .output()
                .await
                .context("Failed to start pdftoppm. Install poppler to inspect PDFs visually.")?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                anyhow::bail!("Failed to render PDF first page with pdftoppm: {stderr}");
            }

            let preview_path = output_prefix.with_extension("png");
            let preview = tokio::fs::read(&preview_path).await.with_context(|| {
                format!(
                    "Failed to read rendered PDF preview {}",
                    preview_path.display()
                )
            })?;

            if u64::try_from(preview.len()).unwrap_or(u64::MAX) > max_image_size_bytes {
                anyhow::bail!(
                    "Rendered PDF preview size ({} bytes) exceeds the maximum allowed size of {} bytes",
                    preview.len(),
                    max_image_size_bytes
                );
            }

            Ok(preview)
        }
        .await;

        let cleanup_result = tokio::fs::remove_dir_all(&render_dir).await;
        match (result, cleanup_result) {
            (Ok(preview), Ok(())) => Ok(preview),
            (Ok(_), Err(error)) => Err(error).with_context(|| {
                format!(
                    "Failed to remove PDF render directory {}",
                    render_dir.display()
                )
            }),
            (Err(error), Ok(())) => Err(error),
            (Err(error), Err(cleanup_error)) => Err(error).with_context(|| {
                format!(
                    "Failed to remove PDF render directory {} after PDF render failure: {}",
                    render_dir.display(),
                    cleanup_error
                )
            }),
        }
    }
}
#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::io::Write;
    use std::path::PathBuf;

    use futures::StreamExt;
    use pretty_assertions::assert_eq;
    use tempfile::NamedTempFile;
    use tokio::fs;

    use super::*;

    static PDF_RENDER_TEMP_DIR_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    fn fixture_pdf_bytes() -> Vec<u8> {
        b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n4 0 obj << /Length 44 >> stream\nBT /F1 18 Tf 20 50 Td (PDF visual test) Tj ET\nendstream endobj\n5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000234 00000 n \n0000000328 00000 n \ntrailer << /Root 1 0 R /Size 6 >>\nstartxref\n398\n%%EOF\n".to_vec()
    }

    fn pdftoppm_available() -> bool {
        std::process::Command::new("pdftoppm")
            .arg("-h")
            .output()
            .is_ok()
    }

    async fn forge_pdf_render_temp_dirs() -> HashSet<PathBuf> {
        let mut entries = fs::read_dir(std::env::temp_dir()).await.unwrap();
        let mut actual = HashSet::new();
        while let Some(entry) = entries.next_entry().await.unwrap() {
            let path = entry.path();
            if path
                .file_name()
                .and_then(|value| value.to_str())
                .map(|value| value.starts_with("forge-pdf-render-"))
                .unwrap_or(false)
            {
                actual.insert(path);
            }
        }
        actual
    }

    #[tokio::test]
    async fn test_render_pdf_first_page_to_png_returns_png_preview() {
        if !pdftoppm_available() {
            return;
        }

        let _guard = PDF_RENDER_TEMP_DIR_LOCK.lock().await;
        let setup = NamedTempFile::with_suffix(".pdf").unwrap();
        fs::write(setup.path(), fixture_pdf_bytes()).await.unwrap();
        let fixture = ForgeFileReadService::new();

        let actual = fixture
            .render_pdf_first_page_to_png(setup.path(), 1024 * 1024)
            .await
            .unwrap();
        let expected = b"\x89PNG\r\n\x1a\n";

        assert_eq!(&actual[..expected.len()], expected);
    }

    #[tokio::test]
    async fn test_render_pdf_first_page_to_png_cleans_temp_dir_after_invalid_pdf() {
        if !pdftoppm_available() {
            return;
        }

        let _guard = PDF_RENDER_TEMP_DIR_LOCK.lock().await;
        let setup = NamedTempFile::with_suffix(".pdf").unwrap();
        fs::write(setup.path(), b"%PDF-1.4\nnot a valid xref\n")
            .await
            .unwrap();
        let fixture = ForgeFileReadService::new();
        let before = forge_pdf_render_temp_dirs().await;

        let actual = fixture
            .render_pdf_first_page_to_png(setup.path(), 1024 * 1024)
            .await;
        let after = forge_pdf_render_temp_dirs().await;
        let expected = before;

        assert!(actual.is_err());
        assert_eq!(after, expected);
    }

    #[tokio::test]
    async fn test_render_pdf_first_page_to_png_rejects_oversized_preview_and_cleans_temp_dir() {
        if !pdftoppm_available() {
            return;
        }

        let _guard = PDF_RENDER_TEMP_DIR_LOCK.lock().await;
        let setup = NamedTempFile::with_suffix(".pdf").unwrap();
        fs::write(setup.path(), fixture_pdf_bytes()).await.unwrap();
        let fixture = ForgeFileReadService::new();
        let before = forge_pdf_render_temp_dirs().await;

        let actual = fixture.render_pdf_first_page_to_png(setup.path(), 0).await;
        let after = forge_pdf_render_temp_dirs().await;
        let expected = before;

        assert!(actual.is_err());
        assert_eq!(after, expected);
    }
    #[tokio::test]
    async fn test_read_batch_utf8() {
        let fixture = ForgeFileReadService::new();

        // Create temporary test files
        let mut file1 = NamedTempFile::new().unwrap();
        let mut file2 = NamedTempFile::new().unwrap();
        let mut file3 = NamedTempFile::new().unwrap();

        writeln!(file1, "content1").unwrap();
        writeln!(file2, "content2").unwrap();
        writeln!(file3, "content3").unwrap();

        let paths = vec![
            file1.path().to_path_buf(),
            file2.path().to_path_buf(),
            file3.path().to_path_buf(),
        ];

        // Read with batch size of 2
        let stream = fixture.read_batch_utf8(2, paths.clone());
        futures::pin_mut!(stream);

        let item1 = stream.next().await.unwrap();
        assert_eq!(item1.0, paths[0]);
        assert_eq!(item1.1.as_deref().unwrap().trim(), "content1");

        let item2 = stream.next().await.unwrap();
        assert_eq!(item2.0, paths[1]);
        assert_eq!(item2.1.as_deref().unwrap().trim(), "content2");

        let item3 = stream.next().await.unwrap();
        assert_eq!(item3.0, paths[2]);
        assert_eq!(item3.1.as_deref().unwrap().trim(), "content3");

        // No more items
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_read_batch_utf8_single_batch() {
        let fixture = ForgeFileReadService::new();

        let mut file1 = NamedTempFile::new().unwrap();
        let mut file2 = NamedTempFile::new().unwrap();

        writeln!(file1, "test1").unwrap();
        writeln!(file2, "test2").unwrap();

        let paths = vec![file1.path().to_path_buf(), file2.path().to_path_buf()];

        // Read with batch size larger than number of files
        let stream = fixture.read_batch_utf8(10, paths.clone());
        futures::pin_mut!(stream);

        let item1 = stream.next().await.unwrap();
        assert_eq!(item1.0, paths[0]);

        let item2 = stream.next().await.unwrap();
        assert_eq!(item2.0, paths[1]);

        // No more items
        assert!(stream.next().await.is_none());
    }
}
