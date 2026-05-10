use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use forge_domain::{Environment, Snapshot, SnapshotRepository};

/// Repository adapter that persists file snapshots through the promoted
/// `forge_services::SnapshotService` implementation.
pub struct ForgeFileSnapshotService {
    inner: Arc<forge_services::SnapshotService>,
}

impl ForgeFileSnapshotService {
    /// Creates a file snapshot repository rooted at the environment snapshot
    /// directory.
    pub fn new(env: Environment) -> Self {
        Self {
            inner: Arc::new(forge_services::SnapshotService::new(env.snapshot_path())),
        }
    }
}

#[async_trait::async_trait]
impl SnapshotRepository for ForgeFileSnapshotService {
    async fn insert_snapshot(&self, file_path: &Path) -> Result<Snapshot> {
        self.inner.create_snapshot(file_path.to_path_buf()).await
    }

    async fn undo_snapshot(&self, file_path: &Path) -> Result<()> {
        self.inner.undo_snapshot(file_path.to_path_buf()).await
    }
}
