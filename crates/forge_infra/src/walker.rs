use anyhow::Result;
use forge_app::{WalkedFile, WalkedFileStream, Walker};
use futures::StreamExt;

pub struct ForgeWalkerService;

impl ForgeWalkerService {
    pub fn new() -> Self {
        Self
    }

    pub async fn walk(&self, config: Walker) -> Result<Vec<WalkedFile>> {
        let files = build_walker(config).get().await?;
        Ok(files.into_iter().map(to_walked_file).collect())
    }

    /// Streams filesystem entries for the provided walker configuration without
    /// collecting the full traversal result first.
    pub fn walk_stream(&self, config: Walker) -> WalkedFileStream {
        let stream = build_walker(config)
            .stream()
            .map(|result| result.map(to_walked_file));
        Box::pin(stream)
    }
}

fn build_walker(config: Walker) -> forge_walker::Walker {
    // Start from the unlimited representation so every `None` remains unlimited.
    // Explicit `Some` limits below narrow the traversal without reintroducing
    // conservative defaults for unrelated fields.
    let mut walker = forge_walker::Walker::max_all().hidden(true);

    walker = walker.cwd(config.cwd);

    if let Some(depth) = config.max_depth {
        walker = walker.max_depth(depth);
    }
    if let Some(breadth) = config.max_breadth {
        walker = walker.max_breadth(breadth);
    }
    if let Some(file_size) = config.max_file_size {
        walker = walker.max_file_size(file_size);
    }
    if let Some(files) = config.max_files {
        walker = walker.max_files(files);
    }
    if let Some(total_size) = config.max_total_size {
        walker = walker.max_total_size(total_size);
    }
    walker.skip_binary(config.skip_binary)
}

fn to_walked_file(file: forge_walker::File) -> WalkedFile {
    WalkedFile { path: file.path, file_name: file.file_name, size: file.size }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use tempfile::tempdir;

    use super::*;

    #[tokio::test]
    async fn test_walker_service_basic_functionality() -> anyhow::Result<()> {
        let fixture = tempdir()?;
        std::fs::write(fixture.path().join("test.txt"), "test content")?;

        let service = ForgeWalkerService::new();
        let config = Walker::conservative().cwd(fixture.path().to_path_buf());

        let actual = service.walk(config).await?;

        let expected = 1; // Should find the test file
        let file_count = actual.iter().filter(|f| !f.is_dir()).count();
        assert_eq!(file_count, expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_walker_service_unlimited_stream_has_no_default_file_cap() -> anyhow::Result<()> {
        let fixture = tempdir()?;
        for index in 0..150 {
            std::fs::write(
                fixture.path().join(format!("file_{index:03}.txt")),
                "test content",
            )?;
        }

        let service = ForgeWalkerService::new();
        let config = Walker::unlimited().cwd(fixture.path().to_path_buf());

        let actual = service
            .walk_stream(config)
            .filter_map(|result| async move { result.ok() })
            .filter(|file| futures::future::ready(!file.is_dir()))
            .collect::<Vec<_>>()
            .await;

        let expected = 150;
        assert_eq!(actual.len(), expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_walker_service_mixed_config_preserves_unlimited_max_files() -> anyhow::Result<()> {
        let fixture = tempdir()?;
        for index in 0..150 {
            std::fs::write(
                fixture.path().join(format!("file_{index:03}.txt")),
                "test content",
            )?;
        }

        let service = ForgeWalkerService::new();
        let config = Walker::unlimited()
            .cwd(fixture.path().to_path_buf())
            .max_depth(1usize);

        let actual = service
            .walk_stream(config)
            .filter_map(|result| async move { result.ok() })
            .filter(|file| futures::future::ready(!file.is_dir()))
            .collect::<Vec<_>>()
            .await;

        let expected = 150;
        assert_eq!(actual.len(), expected);
        Ok(())
    }
}
