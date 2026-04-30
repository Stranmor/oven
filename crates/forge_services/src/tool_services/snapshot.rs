use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use forge_app::{
    DirectoryReaderInfra, EnvironmentInfra, FileDirectoryInfra, FileInfoInfra, FileReaderInfra,
    FileRemoverInfra, FileWriterInfra, SnapshotService, SnapshotUndoOutput,
};
use forge_domain::Snapshot;

pub struct ForgeSnapshotService<F> {
    infra: Arc<F>,
}

impl<F> ForgeSnapshotService<F> {
    pub fn new(infra: Arc<F>) -> Self {
        Self { infra }
    }
}

impl<F: DirectoryReaderInfra + Send + Sync> ForgeSnapshotService<F> {
    async fn find_recent_snapshot(infra: &Arc<F>, snapshot_dir: &Path) -> Result<Option<PathBuf>> {
        let mut latest_path = None;
        let mut latest_time = None;

        let entries = infra.list_directory_entries(snapshot_dir).await?;
        for (path, is_dir) in entries {
            if is_dir {
                continue;
            }

            let filename = match path.file_name().and_then(|f| f.to_str()) {
                Some(name) => name,
                None => continue,
            };

            if filename.ends_with(".snap") {
                let time_str = filename.trim_end_matches(".snap");
                if let Ok(time) =
                    chrono::NaiveDateTime::parse_from_str(time_str, "%Y-%m-%d_%H-%M-%S-%f")
                    && latest_time.is_none_or(|lt| time > lt)
                {
                    latest_time = Some(time);
                    latest_path = Some(path);
                }
            }
        }

        Ok(latest_path)
    }
}

#[async_trait::async_trait]
impl<
    F: FileDirectoryInfra
        + FileInfoInfra
        + FileReaderInfra
        + FileWriterInfra
        + FileRemoverInfra
        + DirectoryReaderInfra
        + EnvironmentInfra
        + Send
        + Sync,
> SnapshotService for ForgeSnapshotService<F>
{
    async fn create_snapshot(&self, path: PathBuf) -> Result<Option<Snapshot>> {
        if !self.infra.exists(&path).await? {
            return Ok(None);
        }

        let snapshot = Snapshot::create(path.clone())?;

        let env = self.infra.get_environment();
        let snapshots_directory = env.snapshot_path();

        let snapshot_path = snapshot.snapshot_path(Some(snapshots_directory.clone()));
        if let Some(parent) = snapshot_path.parent() {
            self.infra.create_dirs(parent).await?;
        }

        let content = self.infra.read(Path::new(&snapshot.path)).await?;
        self.infra
            .write(&snapshot_path, bytes::Bytes::from(content))
            .await?;
        Ok(Some(snapshot))
    }

    async fn undo_snapshot(&self, path: PathBuf) -> Result<SnapshotUndoOutput> {
        let mut output = SnapshotUndoOutput::default();
        let snapshot = Snapshot::create(path.clone())?;
        let env = self.infra.get_environment();
        let snapshots_directory = env.snapshot_path();

        let snapshot_dir = snapshots_directory.join(snapshot.path_hash());

        if !self.infra.exists(&snapshot_dir).await? {
            return Err(anyhow::anyhow!("No snapshots found for {:?}", path));
        }

        let snapshot_path = Self::find_recent_snapshot(&self.infra, &snapshot_dir)
            .await?
            .context(format!("No valid snapshots found for {:?}", path))?;

        if self.infra.exists(&path).await? {
            output.before_undo = Some(self.infra.read_utf8(&path).await?);
        }

        let content = self.infra.read(&snapshot_path).await?;
        self.infra.write(&path, bytes::Bytes::from(content)).await?;

        if self.infra.exists(&path).await? {
            output.after_undo = Some(self.infra.read_utf8(&path).await?);
        }

        self.infra.remove(&snapshot_path).await?;

        Ok(output)
    }
}
