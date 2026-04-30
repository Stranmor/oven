use std::path::PathBuf;

use anyhow::{Context, Result};
use forge_domain::Snapshot;
use forge_fs::ForgeFS;

#[derive(Debug)]
pub struct SnapshotService {
    snapshots_directory: PathBuf,
}

impl SnapshotService {
    pub fn new(snapshot_base_dir: PathBuf) -> Self {
        Self { snapshots_directory: snapshot_base_dir }
    }

    pub async fn create_snapshot(&self, path: PathBuf) -> Result<Snapshot> {
        let snapshot = Snapshot::create(path)?;
        let snapshot_path = snapshot.snapshot_path(Some(self.snapshots_directory.clone()));
        if let Some(parent) = snapshot_path.parent() {
            ForgeFS::create_dir_all(parent).await?;
        }

        let content = ForgeFS::read(&snapshot.path).await?;
        let path = snapshot.snapshot_path(Some(self.snapshots_directory.clone()));
        ForgeFS::write(path, content).await?;
        Ok(snapshot)
    }

    async fn find_recent_snapshot(snapshot_dir: &PathBuf) -> Result<Option<PathBuf>> {
        let mut latest_path = None;
        let mut latest_filename: Option<String> = None;
        let mut dir = ForgeFS::read_dir(snapshot_dir).await?;

        while let Some(entry) = dir.next_entry().await? {
            let filename = entry.file_name().to_string_lossy().to_string();
            if filename.ends_with(".snap")
                && latest_filename
                    .as_ref()
                    .is_none_or(|latest| filename > *latest)
            {
                latest_filename = Some(filename);
                latest_path = Some(entry.path());
            }
        }

        Ok(latest_path)
    }

    pub async fn undo_snapshot(&self, path: PathBuf) -> Result<()> {
        let snapshot = Snapshot::create(path.clone())?;
        let snapshot_dir = self.snapshots_directory.join(snapshot.path_hash());

        if !ForgeFS::exists(&snapshot_dir) {
            return Err(anyhow::anyhow!("No snapshots found for {path:?}"));
        }

        let snapshot_path = Self::find_recent_snapshot(&snapshot_dir)
            .await?
            .context(format!("No valid snapshots found for {path:?}"))?;

        let content = ForgeFS::read(&snapshot_path).await?;
        ForgeFS::write(&path, content).await?;
        ForgeFS::remove_file(&snapshot_path).await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    struct Fixture {
        _temp_dir: TempDir,
        test_file: PathBuf,
        service: SnapshotService,
    }

    impl Fixture {
        async fn new() -> Result<Self> {
            let temp_dir = TempDir::new()?;
            let snapshots_dir = temp_dir.path().join("snapshots");
            let root = temp_dir.path().canonicalize()?;
            let test_file = root.join("test.txt");
            let service = SnapshotService::new(snapshots_dir);
            Ok(Self { _temp_dir: temp_dir, test_file, service })
        }

        async fn write_content(&self, content: &str) -> Result<()> {
            ForgeFS::write(&self.test_file, content.as_bytes()).await
        }

        async fn read_content(&self) -> Result<String> {
            let content = ForgeFS::read(&self.test_file).await?;
            Ok(String::from_utf8(content)?)
        }
    }

    #[tokio::test]
    async fn test_create_snapshot() -> Result<()> {
        let fixture = Fixture::new().await?;
        let expected = "Hello, World!";

        fixture.write_content(expected).await?;
        let snapshot = fixture
            .service
            .create_snapshot(fixture.test_file.clone())
            .await?;
        let actual = String::from_utf8(ForgeFS::read(&snapshot.path).await?)?;

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_undo_snapshot() -> Result<()> {
        let fixture = Fixture::new().await?;
        let expected = "Initial content";

        fixture.write_content(expected).await?;
        fixture
            .service
            .create_snapshot(fixture.test_file.clone())
            .await?;
        fixture.write_content("Modified content").await?;
        fixture
            .service
            .undo_snapshot(fixture.test_file.clone())
            .await?;
        let actual = fixture.read_content().await?;

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_undo_snapshot_no_snapshots() -> Result<()> {
        let fixture = Fixture::new().await?;

        fixture.write_content("test content").await?;
        let result = fixture
            .service
            .undo_snapshot(fixture.test_file.clone())
            .await;
        let actual = result
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        let expected = "No snapshots found";

        assert!(actual.contains(expected));
        Ok(())
    }

    #[tokio::test]
    async fn test_undo_snapshot_after_file_deletion() -> Result<()> {
        let fixture = Fixture::new().await?;
        let expected = "Initial content";

        fixture.write_content(expected).await?;
        fixture
            .service
            .create_snapshot(fixture.test_file.clone())
            .await?;
        ForgeFS::remove_file(&fixture.test_file).await?;
        fixture
            .service
            .undo_snapshot(fixture.test_file.clone())
            .await?;
        let actual = fixture.read_content().await?;

        assert_eq!(actual, expected);
        Ok(())
    }
}
