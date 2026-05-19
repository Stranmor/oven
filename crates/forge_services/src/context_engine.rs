use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use forge_app::{CommandInfra, EnvironmentInfra, FileReaderInfra, WalkerInfra, WorkspaceService};
use forge_domain::{
    AuthCredential, AuthDetails, FileChunk, Node, NodeData, NodeId, ProviderId, ProviderRepository,
    SearchParams, SyncProgress, UserId, WorkspaceContextFreshness,
    WorkspaceContextManifestDiagnostic, WorkspaceId, WorkspaceIndexRepository,
};
use forge_project_model::{
    ContextPack, ContextPackSelection, ProjectIndexer, RetrievalQuery, StaleEvidencePolicy,
    evidence_line_range, local_project_model_dir, local_project_model_manifest, retrieve,
};
use forge_stream::MpscStream;
use futures::future::join_all;
use tracing::info;

use crate::fd::FileDiscovery;
use crate::sync::{WorkspaceSyncEngine, canonicalize_path};

/// Service for indexing workspaces and performing semantic search.
///
/// `F` provides infrastructure capabilities (file I/O, environment, etc.) and
/// `D` is the file-discovery strategy used to enumerate workspace files.
pub struct ForgeWorkspaceService<F, D> {
    infra: Arc<F>,
    discovery: Arc<D>,
}

impl<F, D> Clone for ForgeWorkspaceService<F, D> {
    fn clone(&self) -> Self {
        Self {
            infra: Arc::clone(&self.infra),
            discovery: Arc::clone(&self.discovery),
        }
    }
}

impl<F, D> ForgeWorkspaceService<F, D> {
    /// Creates a new workspace service with the provided infrastructure and
    /// file-discovery strategy.
    pub fn new(infra: Arc<F>, discovery: Arc<D>) -> Self {
        Self { infra, discovery }
    }
}

impl<
    F: 'static
        + ProviderRepository
        + WorkspaceIndexRepository
        + FileReaderInfra
        + EnvironmentInfra<Config = forge_config::ForgeConfig>
        + CommandInfra
        + WalkerInfra,
    D: FileDiscovery + 'static,
> ForgeWorkspaceService<F, D>
{
    /// Internal sync implementation that emits progress events.
    async fn sync_codebase_internal<E, Fut>(&self, path: PathBuf, emit: E) -> Result<()>
    where
        E: Fn(SyncProgress) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = ()> + Send,
    {
        info!(path = %path.display(), "Starting workspace sync");

        emit(SyncProgress::Starting).await;

        let (token, user_id) = self.get_workspace_credentials().await?;
        let batch_size = self.infra.get_config()?.max_file_read_batch_size;
        let path = canonicalize_path(path)?;

        // Find existing workspace - do NOT auto-create
        let workspace = self.get_workspace_by_path(path, &token).await?;
        let workspace_id = workspace.workspace_id.clone();

        // Use the canonical root stored in the workspace record so that file
        // discovery and remote-hash comparison are always relative to the same
        // base, even when `path` is a subdirectory of an ancestor workspace.
        let workspace_root = PathBuf::from(&workspace.working_dir);

        self.write_local_project_model_manifest(&workspace_root)?;

        WorkspaceSyncEngine::new(
            Arc::clone(&self.infra),
            Arc::clone(&self.discovery),
            workspace_root,
            workspace_id,
            user_id,
            token,
            batch_size,
        )
        .run(emit)
        .await
    }

    fn write_local_project_model_manifest(&self, root: &Path) -> Result<PathBuf> {
        let indexer = ProjectIndexer::new(root, local_project_model_dir(root));
        let manifest = indexer.index()?;
        indexer.write_manifest(&manifest)
    }

    /// Gets the ForgeCode services credential and extracts workspace auth
    /// components
    ///
    /// # Errors
    /// Returns an error if the credential is not found, if there's a database
    /// error, or if the credential format is invalid
    async fn get_workspace_credentials(&self) -> Result<(forge_domain::ApiKey, UserId)> {
        let credential = self
            .infra
            .get_credential(&ProviderId::FORGE_SERVICES)
            .await?
            .context("No authentication credentials found. Please authenticate first.")?;

        match &credential.auth_details {
            AuthDetails::ApiKey(token) => {
                // Extract user_id from URL params
                let user_id_str = credential
                    .url_params
                    .get(&"user_id".to_string().into())
                    .ok_or_else(|| {
                        anyhow::anyhow!("Missing user_id in ForgeServices credential")
                    })?;
                let user_id = UserId::from_string(user_id_str.as_str())?;

                Ok((token.clone(), user_id))
            }
            _ => anyhow::bail!("ForgeServices credential must be an API key"),
        }
    }

    /// Finds a workspace by path from remote server, checking for exact match
    /// first, then ancestor workspaces.
    ///
    /// Business logic:
    /// 1. First tries to find an exact match for the given path
    /// 2. If not found, searches for ancestor workspaces
    /// 3. Returns the closest ancestor (longest matching path prefix)
    ///
    /// # Errors
    /// Returns an error if the path cannot be canonicalized or if there's a
    /// server error. Returns Ok(None) if no workspace is found.
    async fn find_workspace_by_path(
        &self,
        path: PathBuf,
        token: &forge_domain::ApiKey,
    ) -> Result<Option<forge_domain::WorkspaceInfo>> {
        let canonical_path = canonicalize_path(path)?;

        // Get all workspaces from remote server
        let workspaces = self.infra.list_workspaces(token).await?;

        let canonical_str = canonical_path.to_string_lossy();

        // Business logic: choose which workspace to use
        // 1. First check for exact match
        if let Some(exact_match) = workspaces.iter().find(|w| w.working_dir == canonical_str) {
            return Ok(Some(exact_match.clone()));
        }

        // 2. Find closest ancestor (longest matching path prefix)
        let mut best_match: Option<(&forge_domain::WorkspaceInfo, usize)> = None;

        for workspace in &workspaces {
            let workspace_path = PathBuf::from(&workspace.working_dir);
            if canonical_path.starts_with(&workspace_path) {
                let path_len = workspace.working_dir.len();
                if best_match.is_none_or(|(_, len)| path_len > len) {
                    best_match = Some((workspace, path_len));
                }
            }
        }

        Ok(best_match.map(|(w, _)| w.clone()))
    }

    /// Looks up the workspace for `path` and returns it, or an error if no
    /// workspace has been indexed for that path.
    ///
    /// # Errors
    ///
    /// Returns an error when the underlying repository lookup fails, or when no
    /// matching workspace is found (i.e. the workspace has not been indexed
    /// yet).
    async fn get_workspace_by_path(
        &self,
        path: PathBuf,
        token: &forge_domain::ApiKey,
    ) -> Result<forge_domain::WorkspaceInfo> {
        self.find_workspace_by_path(path, token)
            .await?
            .context("Workspace not indexed. Please run `forge workspace init` first.")
    }

    async fn _init_workspace(&self, path: PathBuf) -> Result<(bool, WorkspaceId)> {
        let (token, _user_id) = self.get_workspace_credentials().await?;
        let path = canonicalize_path(path)?;

        // Find workspace by exact match or ancestor from remote server
        let workspace = self.find_workspace_by_path(path.clone(), &token).await?;

        let (workspace_id, workspace_path, is_new_workspace) = match workspace {
            Some(workspace_info) => {
                // Found existing workspace - reuse it
                (workspace_info.workspace_id, path.clone(), false)
            }
            None => {
                // No workspace found - create new
                (WorkspaceId::generate(), path.clone(), true)
            }
        };

        let workspace_id = if is_new_workspace {
            // Create workspace on server
            self.infra
                .create_workspace(&workspace_path, &token)
                .await
                .context("Failed to create workspace on server")?
        } else {
            workspace_id
        };

        Ok((is_new_workspace, workspace_id))
    }
    async fn query_local_workspace(
        &self,
        path: PathBuf,
        params: SearchParams<'_>,
    ) -> Result<Vec<Node>> {
        let root = canonicalize_path(path)?;
        let indexer = ProjectIndexer::new(&root, local_project_model_dir(&root));
        let manifest_path = local_project_model_manifest(&root);
        let manifest = indexer.read_manifest().with_context(|| {
            format!(
                "Workspace project model is not indexed at {}. Run project-model indexing first.",
                manifest_path.display()
            )
        })?;
        let retrieval_query = RetrievalQuery {
            text: Some(params.query.to_string()),
            path: None,
            path_prefix: params.starts_with.clone(),
            symbol: None,
            limit: params.limit.unwrap_or(10),
            include_graph_expansion: true,
        };
        let freshness = indexer.evaluate_manifest_freshness(&manifest)?;
        if !freshness.can_inject() {
            anyhow::bail!(
                "Workspace project model is not fresh at {}. Run `forge workspace sync {}` before using project-model context.",
                manifest_path.display(),
                root.display()
            );
        }
        let results = retrieve(&manifest, &retrieval_query);
        let pack = ContextPack::from_selection(
            &manifest,
            ContextPackSelection {
                retrieval_results: results,
                shards: Vec::new(),
                evidence: Vec::new(),
                freshness: freshness.state.clone(),
                stale_policy: StaleEvidencePolicy::Reject,
            },
        )?;
        let mut nodes = Vec::new();
        for evidence in pack.evidence {
            if !matches_path_filters(&evidence.path, &params) {
                continue;
            }
            let Some((start_line, end_line)) = evidence_line_range(&manifest, &evidence.id) else {
                continue;
            };
            let absolute_path = root.join(&evidence.path);
            let (content, _) = self
                .infra
                .range_read_utf8(&absolute_path, u64::from(start_line), u64::from(end_line))
                .await
                .with_context(|| format!("read {}", absolute_path.display()))?;
            nodes.push(Node {
                node_id: NodeId::new(evidence.id),
                node: NodeData::FileChunk(FileChunk {
                    file_path: evidence.path,
                    content,
                    start_line,
                    end_line,
                }),
                relevance: Some(evidence.score),
                distance: None,
            });
        }
        nodes.sort_by(|left, right| {
            right
                .relevance
                .unwrap_or_default()
                .total_cmp(&left.relevance.unwrap_or_default())
                .then_with(|| left.node_id.as_str().cmp(right.node_id.as_str()))
        });
        Ok(nodes)
    }
}

fn matches_path_filters(path: &str, params: &SearchParams<'_>) -> bool {
    if let Some(prefix) = &params.starts_with
        && !path.starts_with(prefix)
    {
        return false;
    }
    if let Some(suffixes) = &params.ends_with
        && !suffixes.iter().any(|suffix| path.ends_with(suffix))
    {
        return false;
    }
    true
}

fn evaluate_project_model_context(path: &Path) -> WorkspaceContextManifestDiagnostic {
    let manifest_path = local_project_model_manifest(path);
    if !path.is_dir() || !manifest_path.is_file() {
        return WorkspaceContextManifestDiagnostic {
            workspace_root: path.to_path_buf(),
            manifest_path,
            manifest_found: false,
            freshness: WorkspaceContextFreshness::Unknown {
                reason: "project-model manifest not found".to_string(),
            },
        };
    }

    let indexer = ProjectIndexer::new(path, local_project_model_dir(path));
    let freshness = match indexer
        .read_manifest()
        .and_then(|manifest| indexer.evaluate_manifest_freshness(&manifest))
    {
        Ok(evaluation) if evaluation.can_inject() => WorkspaceContextFreshness::Fresh,
        Ok(evaluation) if evaluation.state.fresh => WorkspaceContextFreshness::Unknown {
            reason: "project-model freshness checked only indexed files; added-file discovery not proven".to_string(),
        },
        Ok(evaluation) => WorkspaceContextFreshness::Stale {
            changed: evaluation.state.changed,
            deleted: evaluation.state.deleted,
            added: evaluation.state.added,
        },
        Err(error) => WorkspaceContextFreshness::Unknown { reason: error.to_string() },
    };

    WorkspaceContextManifestDiagnostic {
        workspace_root: path.to_path_buf(),
        manifest_path,
        manifest_found: true,
        freshness,
    }
}

#[async_trait]
impl<
    F: ProviderRepository
        + WorkspaceIndexRepository
        + FileReaderInfra
        + EnvironmentInfra<Config = forge_config::ForgeConfig>
        + CommandInfra
        + WalkerInfra
        + 'static,
    D: FileDiscovery + 'static,
> WorkspaceService for ForgeWorkspaceService<F, D>
{
    async fn sync_workspace(&self, path: PathBuf) -> Result<MpscStream<Result<SyncProgress>>> {
        let service = Clone::clone(self);

        let stream = MpscStream::spawn(move |tx| async move {
            // Create emit closure that captures the sender
            let emit = |progress: SyncProgress| {
                let tx = tx.clone();
                async move {
                    let _ = tx.send(Ok(progress)).await;
                }
            };

            // Run the sync and emit progress events
            let result = service.sync_codebase_internal(path, emit).await;

            // If there was an error, send it through the channel
            if let Err(e) = result {
                let _ = tx.send(Err(e)).await;
            }
        });

        Ok(stream)
    }

    /// Performs semantic code search on a workspace.
    async fn query_workspace(
        &self,
        path: PathBuf,
        params: forge_domain::SearchParams<'_>,
    ) -> Result<Vec<forge_domain::Node>> {
        self.query_local_workspace(path, params).await
    }

    /// Lists all workspaces.
    async fn list_workspaces(&self) -> Result<Vec<forge_domain::WorkspaceInfo>> {
        let (token, _) = self.get_workspace_credentials().await?;

        self.infra
            .as_ref()
            .list_workspaces(&token)
            .await
            .context("Failed to list workspaces")
    }

    /// Retrieves workspace information for a specific path.
    async fn get_workspace_info(
        &self,
        path: PathBuf,
    ) -> Result<Option<forge_domain::WorkspaceInfo>> {
        let (token, _user_id) = self.get_workspace_credentials().await?;
        let workspace = self.find_workspace_by_path(path, &token).await?;

        Ok(workspace)
    }

    /// Deletes a workspace from the server.
    async fn delete_workspace(&self, workspace_id: &forge_domain::WorkspaceId) -> Result<()> {
        let (token, _) = self.get_workspace_credentials().await?;

        self.infra
            .as_ref()
            .delete_workspace(workspace_id, &token)
            .await
            .context("Failed to delete workspace from server")?;

        Ok(())
    }

    /// Deletes multiple workspaces in parallel from both the server and local
    /// database.
    async fn delete_workspaces(&self, workspace_ids: &[forge_domain::WorkspaceId]) -> Result<()> {
        // Delete all workspaces in parallel by calling delete_workspace for each
        let delete_tasks: Vec<_> = workspace_ids
            .iter()
            .map(|workspace_id| self.delete_workspace(workspace_id))
            .collect();

        let results = join_all(delete_tasks).await;

        // Collect all errors
        let errors: Vec<_> = results.into_iter().filter_map(|r| r.err()).collect();

        if !errors.is_empty() {
            return Err(anyhow::anyhow!(
                "Failed to delete {} workspace(s): [{}]",
                errors.len(),
                errors
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        Ok(())
    }

    async fn is_indexed(&self, path: &std::path::Path) -> Result<bool> {
        Ok(evaluate_project_model_context(path).can_inject())
    }

    async fn project_model_context_diagnostic(
        &self,
        path: &std::path::Path,
    ) -> Result<WorkspaceContextManifestDiagnostic> {
        Ok(evaluate_project_model_context(path))
    }

    async fn get_workspace_status(&self, path: PathBuf) -> Result<Vec<forge_domain::FileStatus>> {
        let (token, user_id) = self.get_workspace_credentials().await?;

        let workspace = self.get_workspace_by_path(path, &token).await?;

        // Reuse the canonical path already stored in the workspace (resolved during
        // sync), avoiding a redundant canonicalize() IO call.
        let canonical_path = PathBuf::from(&workspace.working_dir);

        let batch_size = self.infra.get_config()?.max_file_read_batch_size;

        WorkspaceSyncEngine::new(
            Arc::clone(&self.infra),
            Arc::clone(&self.discovery),
            canonical_path,
            workspace.workspace_id,
            user_id,
            token,
            batch_size,
        )
        .compute_status()
        .await
    }

    async fn is_authenticated(&self) -> Result<bool> {
        if self
            .infra
            .get_credential(&ProviderId::FORGE_SERVICES)
            .await?
            .is_some()
        {
            return Ok(true);
        }
        let cwd = self.infra.get_environment().cwd;
        if evaluate_project_model_context(&cwd).can_inject() {
            return Ok(true);
        }
        Ok(false)
    }

    async fn init_auth_credentials(&self) -> Result<forge_domain::WorkspaceAuth> {
        // Authenticate with the indexing service
        let auth = self
            .infra
            .authenticate()
            .await
            .context("Failed to authenticate with indexing service")?;

        // Convert to AuthCredential and store
        let mut url_params = HashMap::new();
        url_params.insert(
            "user_id".to_string().into(),
            auth.user_id.to_string().into(),
        );

        let credential = AuthCredential {
            id: ProviderId::FORGE_SERVICES,
            auth_details: auth.clone().into(),
            url_params,
        };

        self.infra
            .upsert_credential(credential)
            .await
            .context("Failed to store authentication credentials")?;

        Ok(auth)
    }

    async fn init_workspace(&self, path: PathBuf) -> Result<WorkspaceId> {
        let (is_new, workspace_id) = self._init_workspace(path).await?;

        if is_new {
            Ok(workspace_id)
        } else {
            Err(forge_domain::Error::WorkspaceAlreadyInitialized(workspace_id).into())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use anyhow::{Result, bail};
    use forge_app::{WalkedFile, WalkedFileStream, Walker};
    use forge_domain::{
        AnyProvider, AuthCredential, CodeSearchQuery, CommandExecutionOutput, CommandOutput,
        ConfigOperation, Environment, FileHash, ProcessId, ProcessReadCursor, ProcessReadOutput,
        ProcessStartOutput, ProcessStatus, ProviderTemplate, ShellHandoffTimeoutSeconds,
        WorkspaceFiles, WorkspaceInfo,
    };
    use forge_project_model::FreshnessState;
    use futures::{Stream, StreamExt};
    use tempfile::TempDir;

    use super::*;
    struct LocalSearchInfra {
        cwd: PathBuf,
        credential: Option<AuthCredential>,
        workspaces: Vec<WorkspaceInfo>,
        remote_search_called: Arc<AtomicBool>,
        range_read_called: Arc<AtomicBool>,
    }

    struct NoopDiscovery;

    #[async_trait]
    impl FileDiscovery for NoopDiscovery {
        async fn discover(&self, _dir_path: &Path) -> Result<Vec<PathBuf>> {
            bail!("unused discovery")
        }
    }

    impl EnvironmentInfra for LocalSearchInfra {
        type Config = forge_config::ForgeConfig;

        fn get_env_var(&self, _key: &str) -> Option<String> {
            None
        }

        fn get_env_vars(&self) -> BTreeMap<String, String> {
            BTreeMap::new()
        }

        fn get_environment(&self) -> Environment {
            Environment {
                os: "test".to_string(),
                cwd: self.cwd.clone(),
                home: None,
                shell: "sh".to_string(),
                base_path: self.cwd.join(".forge"),
            }
        }

        fn get_config(&self) -> Result<Self::Config> {
            Ok(forge_config::ForgeConfig::default())
        }

        async fn update_environment(&self, _ops: Vec<ConfigOperation>) -> Result<()> {
            bail!("unused environment update")
        }
    }

    #[async_trait]
    impl ProviderRepository for LocalSearchInfra {
        async fn get_all_providers(&self) -> Result<Vec<AnyProvider>> {
            bail!("unused provider listing")
        }

        async fn get_provider(&self, _id: ProviderId) -> Result<ProviderTemplate> {
            bail!("unused provider lookup")
        }

        async fn upsert_credential(&self, _credential: AuthCredential) -> Result<()> {
            bail!("unused credential write")
        }

        async fn get_credential(&self, _id: &ProviderId) -> Result<Option<AuthCredential>> {
            Ok(self.credential.clone())
        }

        async fn remove_credential(&self, _id: &ProviderId) -> Result<()> {
            bail!("unused credential removal")
        }

        async fn migrate_env_credentials(&self) -> Result<Option<forge_domain::MigrationResult>> {
            Ok(None)
        }
    }

    #[async_trait]
    impl WorkspaceIndexRepository for LocalSearchInfra {
        async fn authenticate(&self) -> Result<forge_domain::WorkspaceAuth> {
            bail!("unused remote authentication")
        }

        async fn create_workspace(
            &self,
            _working_dir: &Path,
            _auth_token: &forge_domain::ApiKey,
        ) -> Result<WorkspaceId> {
            bail!("unused remote workspace creation")
        }

        async fn upload_files(
            &self,
            _upload: &forge_domain::FileUpload,
            _auth_token: &forge_domain::ApiKey,
        ) -> Result<forge_domain::FileUploadInfo> {
            bail!("unused remote upload")
        }

        async fn search(
            &self,
            _query: &CodeSearchQuery<'_>,
            _auth_token: &forge_domain::ApiKey,
        ) -> Result<Vec<Node>> {
            self.remote_search_called.store(true, Ordering::SeqCst);
            Ok(vec![Node {
                node_id: NodeId::new("remote-search-result"),
                node: NodeData::FileChunk(FileChunk {
                    file_path: "remote.rs".to_string(),
                    content: "remote search should not be used".to_string(),
                    start_line: 1,
                    end_line: 1,
                }),
                relevance: Some(1.0),
                distance: None,
            }])
        }

        async fn list_workspaces(
            &self,
            _auth_token: &forge_domain::ApiKey,
        ) -> Result<Vec<WorkspaceInfo>> {
            Ok(self.workspaces.clone())
        }

        async fn get_workspace(
            &self,
            _workspace_id: &WorkspaceId,
            _auth_token: &forge_domain::ApiKey,
        ) -> Result<Option<WorkspaceInfo>> {
            bail!("unused remote workspace lookup")
        }

        async fn list_workspace_files(
            &self,
            _workspace: &WorkspaceFiles,
            _auth_token: &forge_domain::ApiKey,
        ) -> Result<Vec<FileHash>> {
            bail!("unused remote file listing")
        }

        async fn delete_files(
            &self,
            _deletion: &forge_domain::FileDeletion,
            _auth_token: &forge_domain::ApiKey,
        ) -> Result<()> {
            bail!("unused remote deletion")
        }

        async fn delete_workspace(
            &self,
            _workspace_id: &WorkspaceId,
            _auth_token: &forge_domain::ApiKey,
        ) -> Result<()> {
            bail!("unused remote workspace deletion")
        }
    }

    #[async_trait]
    impl FileReaderInfra for LocalSearchInfra {
        async fn read_utf8(&self, path: &Path) -> Result<String> {
            Ok(fs::read_to_string(path)?)
        }

        fn read_batch_utf8(
            &self,
            _batch_size: usize,
            paths: Vec<PathBuf>,
        ) -> impl Stream<Item = (PathBuf, Result<String>)> + Send {
            futures::stream::iter(paths.into_iter().map(|path| {
                let content = fs::read_to_string(&path).map_err(anyhow::Error::from);
                (path, content)
            }))
        }

        async fn read(&self, path: &Path) -> Result<Vec<u8>> {
            Ok(fs::read(path)?)
        }

        async fn range_read_utf8(
            &self,
            path: &Path,
            start_line: u64,
            end_line: u64,
        ) -> Result<(String, forge_domain::FileInfo)> {
            self.range_read_called.store(true, Ordering::SeqCst);
            let content = fs::read_to_string(path)?;
            let selected = content
                .lines()
                .skip(start_line.saturating_sub(1) as usize)
                .take(end_line.saturating_sub(start_line).saturating_add(1) as usize)
                .collect::<Vec<_>>()
                .join("\n");
            Ok((
                selected,
                forge_domain::FileInfo::new(
                    start_line,
                    end_line,
                    content.lines().count() as u64,
                    String::new(),
                ),
            ))
        }
    }

    #[async_trait]
    impl CommandInfra for LocalSearchInfra {
        async fn execute_command(
            &self,
            _command: String,
            _working_dir: PathBuf,
            _silent: bool,
            _env_vars: Option<Vec<String>>,
            _handoff_timeout: ShellHandoffTimeoutSeconds,
        ) -> Result<CommandExecutionOutput> {
            Ok(CommandExecutionOutput {
                output: CommandOutput {
                    command: String::new(),
                    stdout: String::new(),
                    stderr: String::new(),
                    exit_code: Some(0),
                },
                process: None,
            })
        }

        async fn execute_command_raw(
            &self,
            _command: &str,
            _working_dir: PathBuf,
            _env_vars: Option<Vec<String>>,
        ) -> Result<std::process::ExitStatus> {
            bail!("unused raw command")
        }

        async fn start_process(
            &self,
            _command: String,
            _working_dir: PathBuf,
            _env_vars: Option<Vec<String>>,
        ) -> Result<ProcessStartOutput> {
            bail!("unused process start")
        }

        async fn process_status(
            &self,
            _process_id: ProcessId,
            _wait: Option<forge_domain::ProcessObservationWaitSeconds>,
        ) -> Result<ProcessStatus> {
            bail!("unused process status")
        }

        async fn read_process(
            &self,
            _process_id: ProcessId,
            _cursor: ProcessReadCursor,
            _wait: Option<forge_domain::ProcessObservationWaitSeconds>,
        ) -> Result<ProcessReadOutput> {
            bail!("unused process read")
        }

        async fn list_processes(&self) -> Result<Vec<ProcessStatus>> {
            bail!("unused process list")
        }

        async fn kill_process(&self, _process_id: ProcessId) -> Result<ProcessStatus> {
            bail!("unused process kill")
        }
    }

    #[async_trait]
    impl WalkerInfra for LocalSearchInfra {
        async fn walk(&self, _config: Walker) -> Result<Vec<WalkedFile>> {
            bail!("unused walker")
        }

        async fn walk_stream(&self, _config: Walker) -> Result<WalkedFileStream> {
            let stream = futures::stream::empty::<Result<WalkedFile>>();
            Ok(Pin::from(Box::new(stream)))
        }
    }

    fn fixture_workspace() -> Result<(TempDir, PathBuf)> {
        let fixture = TempDir::new()?;
        let root = fixture.path().join("workspace");
        fs::create_dir_all(root.join("src"))?;
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"runtime_fixture\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )?;
        fs::write(
            root.join("src").join("lib.rs"),
            "pub struct RuntimeNeedle {\n    pub value: usize,\n}\n\npub fn build_runtime_needle() -> RuntimeNeedle {\n    RuntimeNeedle { value: 7 }\n}\n",
        )?;
        Ok((fixture, root))
    }

    fn write_fixture_project_model(root: &Path) -> Result<PathBuf> {
        let setup = ProjectIndexer::new(root, local_project_model_dir(root));
        let manifest = setup.index()?;
        setup.write_manifest(&manifest)
    }

    #[tokio::test]
    async fn query_workspace_uses_local_project_model_and_returns_file_chunks() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        write_fixture_project_model(&root)?;
        let remote_search_called = Arc::new(AtomicBool::new(false));
        let range_read_called = Arc::new(AtomicBool::new(false));
        let setup = ForgeWorkspaceService::new(
            Arc::new(LocalSearchInfra {
                cwd: root.clone(),
                credential: None,
                workspaces: Vec::new(),
                remote_search_called: Arc::clone(&remote_search_called),
                range_read_called: Arc::clone(&range_read_called),
            }),
            Arc::new(NoopDiscovery),
        );
        let params = SearchParams::new("build runtime needle", "runtime integration proof")
            .limit(5usize)
            .ends_with(vec![".rs".to_string()]);
        let actual = WorkspaceService::query_workspace(&setup, root, params).await?;
        let chunk = actual
            .iter()
            .find_map(|node| match &node.node {
                NodeData::FileChunk(chunk) if chunk.content.contains("build_runtime_needle") => {
                    Some((node.node_id.as_str().to_string(), chunk.clone()))
                }
                _ => None,
            })
            .expect("local project-model search should return the Rust function chunk");
        let expected = "src/lib.rs".to_string();

        assert_eq!(chunk.1.file_path, expected);
        assert!(chunk.1.start_line <= 5);
        assert!(chunk.1.end_line >= 7);
        assert!(chunk.0.contains("src/lib.rs"));
        assert!(!remote_search_called.load(Ordering::SeqCst));
        assert!(range_read_called.load(Ordering::SeqCst));
        Ok(())
    }

    #[tokio::test]
    async fn query_workspace_rejects_stale_project_model_manifest() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        write_fixture_project_model(&root)?;
        fs::write(
            root.join("src").join("lib.rs"),
            "pub struct RuntimeNeedle {\n    pub value: usize,\n}\n\npub fn build_runtime_needle() -> RuntimeNeedle {\n    RuntimeNeedle { value: 8 }\n}\n",
        )?;
        let range_read_called = Arc::new(AtomicBool::new(false));
        let setup = ForgeWorkspaceService::new(
            Arc::new(LocalSearchInfra {
                cwd: root.clone(),
                credential: None,
                workspaces: Vec::new(),
                remote_search_called: Arc::new(AtomicBool::new(false)),
                range_read_called: Arc::clone(&range_read_called),
            }),
            Arc::new(NoopDiscovery),
        );
        let params = SearchParams::new("build runtime needle", "runtime integration proof");
        let actual = WorkspaceService::query_workspace(&setup, root, params).await;
        let actual_error = match actual {
            Ok(nodes) => {
                anyhow::bail!("expected stale manifest error, got {} nodes", nodes.len())
            }
            Err(error) => error.to_string(),
        };
        let expected = "Workspace project model is not fresh";

        assert!(actual_error.contains(expected));
        assert!(!range_read_called.load(Ordering::SeqCst));
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_diagnostic_reports_stale_manifest_for_changed_file() -> Result<()>
    {
        let (_fixture, root) = fixture_workspace()?;
        write_fixture_project_model(&root)?;
        let setup = ForgeWorkspaceService::new(
            Arc::new(LocalSearchInfra {
                cwd: root.clone(),
                credential: None,
                workspaces: Vec::new(),
                remote_search_called: Arc::new(AtomicBool::new(false)),
                range_read_called: Arc::new(AtomicBool::new(false)),
            }),
            Arc::new(NoopDiscovery),
        );
        let fresh = WorkspaceService::project_model_context_diagnostic(&setup, &root).await?;
        fs::write(
            root.join("src").join("lib.rs"),
            "pub struct RuntimeNeedle {\n    pub value: usize,\n}\n\npub fn build_runtime_needle() -> RuntimeNeedle {\n    RuntimeNeedle { value: 8 }\n}\n",
        )?;
        let stale = WorkspaceService::project_model_context_diagnostic(&setup, &root).await?;
        let actual = (
            fresh.manifest_found,
            fresh.freshness.label().to_string(),
            stale.manifest_found,
            stale.freshness.label().to_string(),
            stale.can_inject(),
        );
        let expected = (true, "fresh".to_string(), true, "stale".to_string(), false);

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_diagnostic_reports_stale_manifest_for_deleted_file() -> Result<()>
    {
        let (_fixture, root) = fixture_workspace()?;
        write_fixture_project_model(&root)?;
        fs::remove_file(root.join("src").join("lib.rs"))?;
        let setup = ForgeWorkspaceService::new(
            Arc::new(LocalSearchInfra {
                cwd: root.clone(),
                credential: None,
                workspaces: Vec::new(),
                remote_search_called: Arc::new(AtomicBool::new(false)),
                range_read_called: Arc::new(AtomicBool::new(false)),
            }),
            Arc::new(NoopDiscovery),
        );

        let actual = WorkspaceService::project_model_context_diagnostic(&setup, &root).await?;
        let expected = WorkspaceContextFreshness::Stale {
            changed: Vec::new(),
            deleted: vec!["src/lib.rs".to_string()],
            added: Vec::new(),
        };
        assert_eq!(actual.freshness, expected);
        assert_eq!(actual.can_inject(), false);
        Ok(())
    }

    #[tokio::test]
    async fn is_indexed_requires_project_model_manifest_without_remote_credentials() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let remote_search_called = Arc::new(AtomicBool::new(false));
        let range_read_called = Arc::new(AtomicBool::new(false));
        let setup = ForgeWorkspaceService::new(
            Arc::new(LocalSearchInfra {
                cwd: root.clone(),
                credential: None,
                workspaces: Vec::new(),
                remote_search_called,
                range_read_called,
            }),
            Arc::new(NoopDiscovery),
        );
        let actual_before = WorkspaceService::is_indexed(&setup, &root).await?;
        write_fixture_project_model(&root)?;
        let actual_after = WorkspaceService::is_indexed(&setup, &root).await?;
        let expected = (false, true);

        assert_eq!((actual_before, actual_after), expected);
        Ok(())
    }

    fn workspace_auth_credential() -> AuthCredential {
        let mut url_params = HashMap::new();
        url_params.insert(
            "user_id".to_string().into(),
            UserId::generate().to_string().into(),
        );
        AuthCredential {
            id: ProviderId::FORGE_SERVICES,
            auth_details: AuthDetails::ApiKey(forge_domain::ApiKey::from("test-token".to_string())),
            url_params,
        }
    }

    fn remote_workspace(root: &Path) -> WorkspaceInfo {
        WorkspaceInfo {
            workspace_id: WorkspaceId::generate(),
            working_dir: root.to_string_lossy().to_string(),
            node_count: Some(1),
            relation_count: Some(0),
            last_updated: Some(chrono::Utc::now()),
            created_at: chrono::Utc::now(),
        }
    }

    #[tokio::test]
    async fn is_indexed_rejects_remote_workspace_without_local_project_model_manifest() -> Result<()>
    {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ForgeWorkspaceService::new(
            Arc::new(LocalSearchInfra {
                cwd: root.clone(),
                credential: Some(workspace_auth_credential()),
                workspaces: vec![remote_workspace(&root)],
                remote_search_called: Arc::new(AtomicBool::new(false)),
                range_read_called: Arc::new(AtomicBool::new(false)),
            }),
            Arc::new(NoopDiscovery),
        );
        let actual = WorkspaceService::is_indexed(&setup, &root).await?;
        let expected = false;

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn query_workspace_requires_persisted_project_model_manifest() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ForgeWorkspaceService::new(
            Arc::new(LocalSearchInfra {
                cwd: root.clone(),
                credential: None,
                workspaces: Vec::new(),
                remote_search_called: Arc::new(AtomicBool::new(false)),
                range_read_called: Arc::new(AtomicBool::new(false)),
            }),
            Arc::new(NoopDiscovery),
        );
        let params = SearchParams::new("build runtime needle", "runtime integration proof");
        let actual = WorkspaceService::query_workspace(&setup, root, params).await;
        let expected = "Workspace project model is not indexed";
        let actual_error = match actual {
            Ok(nodes) => {
                anyhow::bail!("expected missing manifest error, got {} nodes", nodes.len())
            }
            Err(error) => error.to_string(),
        };

        assert!(actual_error.contains(expected));
        Ok(())
    }

    #[tokio::test]
    async fn sync_workspace_writes_local_project_model_manifest_before_remote_file_sync()
    -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ForgeWorkspaceService::new(
            Arc::new(LocalSearchInfra {
                cwd: root.clone(),
                credential: Some(workspace_auth_credential()),
                workspaces: vec![remote_workspace(&root)],
                remote_search_called: Arc::new(AtomicBool::new(false)),
                range_read_called: Arc::new(AtomicBool::new(false)),
            }),
            Arc::new(NoopDiscovery),
        );
        let mut stream = WorkspaceService::sync_workspace(&setup, root.clone()).await?;
        while let Some(_event) = stream.next().await {}
        let actual = local_project_model_manifest(&root).is_file();
        let expected = true;

        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn context_pack_preserves_retrieval_provenance_for_runtime_fixture() -> Result<()> {
        let (fixture, root) = fixture_workspace()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let query = RetrievalQuery {
            text: Some("build runtime needle".to_string()),
            path: None,
            path_prefix: None,
            symbol: None,
            limit: 5,
            include_graph_expansion: true,
        };
        let results = retrieve(&manifest, &query);
        let pack = ContextPack::from_selection(
            &manifest,
            ContextPackSelection {
                retrieval_results: results,
                shards: Vec::new(),
                evidence: Vec::new(),
                freshness: FreshnessState {
                    changed: Vec::new(),
                    deleted: Vec::new(),
                    added: Vec::new(),
                    unchanged: manifest
                        .files
                        .iter()
                        .map(|file| file.path.clone())
                        .collect(),
                    fresh: true,
                },
                stale_policy: StaleEvidencePolicy::Mark,
            },
        )?;
        let actual = pack
            .evidence
            .iter()
            .find(|evidence| {
                evidence.path == "src/lib.rs" && evidence.provenance.source == "rust-ast"
            })
            .map(|evidence| {
                (
                    evidence.path.clone(),
                    evidence.provenance.path.clone(),
                    evidence.provenance.source.clone(),
                    evidence.provenance.start_line,
                )
            })
            .expect("context pack should include Rust source provenance");
        let expected = (
            "src/lib.rs".to_string(),
            "src/lib.rs".to_string(),
            "rust-ast".to_string(),
            Some(5),
        );

        assert_eq!(actual, expected);
        Ok(())
    }
}
