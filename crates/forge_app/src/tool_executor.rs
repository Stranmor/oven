use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::anyhow;
use forge_domain::{
    CodebaseQueryResult, ToolCallContext, ToolCatalog, ToolOutput, resolve_execution_cwd,
};

use crate::fmt::content::FormatContent;
use crate::operation::{TempContentFiles, ToolOperation};
use crate::services::Services;
use crate::{
    AgentRegistry, ConversationService, EnvironmentInfra, FollowUpService, FsPatchService,
    FsReadService, FsRemoveService, FsSearchService, FsUndoService, FsWriteService,
    ImageReadService, NetFetchService, PlanCreateService, ProviderService, ShellExecuteRequest,
    ShellService, SkillFetchService, WorkspaceService,
};

fn canonicalize_workspace_build_path(
    requested_path: &Path,
    environment_cwd: &Path,
) -> anyhow::Result<PathBuf> {
    let requested = if requested_path.is_absolute() {
        requested_path.to_path_buf()
    } else {
        environment_cwd.join(requested_path)
    };
    let canonical_requested = std::fs::canonicalize(&requested).map_err(|error| {
        anyhow!(
            "workspace_vector_index_build_continuation workspace path '{}' could not be canonicalized: {error}",
            requested.display()
        )
    })?;
    let canonical_allowed = std::fs::canonicalize(environment_cwd).map_err(|error| {
        anyhow!(
            "workspace_vector_index_build_continuation current workspace root '{}' could not be canonicalized: {error}",
            environment_cwd.display()
        )
    })?;
    if canonical_requested != canonical_allowed {
        anyhow::bail!(
            "workspace_vector_index_build_continuation rejected workspace path '{}': canonical path '{}' does not match current workspace root '{}'",
            requested.display(),
            canonical_requested.display(),
            canonical_allowed.display()
        );
    }
    Ok(canonical_requested)
}

pub struct ToolExecutor<S> {
    services: Arc<S>,
}

fn resolve_tool_execution_cwd(requested_cwd: Option<&PathBuf>, environment_cwd: &Path) -> PathBuf {
    resolve_execution_cwd(requested_cwd, environment_cwd)
}

impl<
    S: FsReadService
        + ImageReadService
        + FsWriteService
        + FsSearchService
        + WorkspaceService
        + NetFetchService
        + FsRemoveService
        + FsPatchService
        + FsUndoService
        + ShellService
        + FollowUpService
        + ConversationService
        + EnvironmentInfra<Config = forge_config::ForgeConfig>
        + PlanCreateService
        + SkillFetchService
        + AgentRegistry
        + ProviderService
        + Services,
> ToolExecutor<S>
{
    pub fn new(services: Arc<S>) -> Self {
        Self { services }
    }

    fn require_prior_read(
        &self,
        context: &ToolCallContext,
        raw_path: &str,
        action: &str,
    ) -> anyhow::Result<()> {
        let target_path = self.normalize_path(raw_path.to_string());
        let has_read = context.with_metrics(|metrics| {
            metrics.files_accessed.contains(&target_path)
                || metrics.files_accessed.contains(raw_path)
        })?;

        if has_read {
            Ok(())
        } else {
            Err(anyhow!(
                "You must read the file with the read tool before attempting to {action}.",
                action = action
            ))
        }
    }

    async fn dump_operation(&self, operation: &ToolOperation) -> anyhow::Result<TempContentFiles> {
        match operation {
            ToolOperation::NetFetch { input: _, output } => {
                let config = self.services.get_config()?;
                let original_length = output.content.len();
                let is_truncated = original_length > config.max_fetch_chars;
                let mut files = TempContentFiles::default();

                if is_truncated {
                    files = files.stdout(
                        self.create_temp_file("forge_fetch_", ".txt", &output.content)
                            .await?,
                    );
                }

                Ok(files)
            }
            ToolOperation::Shell { output } => {
                let config = self.services.get_config()?;
                let stdout_lines = output.output.stdout.lines().count();
                let stderr_lines = output.output.stderr.lines().count();
                let stdout_truncated =
                    stdout_lines > config.max_stdout_prefix_lines + config.max_stdout_suffix_lines;
                let stderr_truncated =
                    stderr_lines > config.max_stdout_prefix_lines + config.max_stdout_suffix_lines;

                let mut files = TempContentFiles::default();

                if stdout_truncated {
                    files = files.stdout(
                        self.create_temp_file("forge_shell_stdout_", ".txt", &output.output.stdout)
                            .await?,
                    );
                }
                if stderr_truncated {
                    files = files.stderr(
                        self.create_temp_file("forge_shell_stderr_", ".txt", &output.output.stderr)
                            .await?,
                    );
                }

                Ok(files)
            }
            _ => Ok(TempContentFiles::default()),
        }
    }

    /// Converts a path to absolute by joining it with the current working
    /// directory if it's relative
    fn normalize_path(&self, path: String) -> String {
        let env = self.services.get_environment();
        let path_buf = PathBuf::from(&path);

        if path_buf.is_absolute() {
            path
        } else {
            PathBuf::from(&env.cwd).join(path_buf).display().to_string()
        }
    }

    /// Resolves command execution cwd to the same physical path used for
    /// permission checks.
    fn resolve_execution_cwd(&self, cwd: Option<&PathBuf>) -> PathBuf {
        resolve_tool_execution_cwd(cwd, self.services.get_environment().cwd.as_path())
    }

    async fn create_temp_file(
        &self,
        prefix: &str,
        ext: &str,
        content: &str,
    ) -> anyhow::Result<std::path::PathBuf> {
        let path = tempfile::Builder::new()
            .disable_cleanup(true)
            .prefix(prefix)
            .suffix(ext)
            .tempfile()?
            .into_temp_path()
            .to_path_buf();
        self.services
            .write(
                path.to_string_lossy().to_string(),
                content.to_string(),
                true,
            )
            .await?;
        Ok(path)
    }

    async fn call_internal(
        &self,
        input: ToolCatalog,
        context: &ToolCallContext,
    ) -> anyhow::Result<ToolOperation> {
        Ok(match input {
            ToolCatalog::Read(input) => {
                let normalized_path = self.normalize_path(input.file_path.clone());
                let output = self
                    .services
                    .read(
                        normalized_path,
                        input
                            .range
                            .as_ref()
                            .and_then(|r| r.start_line)
                            .map(|i| i as u64),
                        input
                            .range
                            .as_ref()
                            .and_then(|r| r.end_line)
                            .map(|i| i as u64),
                    )
                    .await?;

                (input, output).into()
            }
            ToolCatalog::Write(input) => {
                let normalized_path = self.normalize_path(input.file_path.clone());
                let output = self
                    .services
                    .write(normalized_path, input.content.clone(), input.overwrite)
                    .await?;
                (input, output).into()
            }
            ToolCatalog::FsSearch(input) => {
                let mut params = input.clone();
                // Normalize path if provided
                if let Some(ref path) = params.path {
                    params.path = Some(self.normalize_path(path.clone()));
                }
                let output = self.services.search(params).await?;
                (input, output).into()
            }
            ToolCatalog::SemSearch(input) => {
                let config = self.services.get_config()?;
                let env = self.services.get_environment();
                let services = self.services.clone();
                let cwd = env.cwd.clone();
                let limit = config.max_sem_search_results;
                let top_k = config.sem_search_top_k as u32;
                let embedding_model_id = config
                    .semantic_embedding_model_id
                    .clone()
                    .filter(|model_id| !model_id.trim().is_empty());
                let readiness = services
                    .sem_search_availability(cwd.clone(), embedding_model_id.clone())
                    .await?;
                readiness.ensure_ready()?;
                let ready_dimension = match readiness {
                    forge_domain::SemSearchAvailability::Ready { dimension, .. } => dimension,
                    forge_domain::SemSearchAvailability::Unsupported { .. }
                    | forge_domain::SemSearchAvailability::Unknown { .. } => {
                        unreachable!("non-ready sem_search state should fail preflight")
                    }
                };
                let embedding_model_id = embedding_model_id.ok_or_else(|| {
                    anyhow!(
                        "semantic search embedding model id is not configured: set semantic_embedding_model_id"
                    )
                })?;
                let mut params = Vec::with_capacity(input.queries.len());
                for search_query in &input.queries {
                    let output = services
                        .embed_workspace_query(
                            search_query.query.clone(),
                            embedding_model_id.clone(),
                        )
                        .await
                        .map_err(|error| {
                            anyhow!(
                                "sem_search provider unavailable after readiness preflight: {error}"
                            )
                        })?;
                    if output.embedding_model_id != embedding_model_id {
                        anyhow::bail!(
                            "semantic search embedding model id mismatch: expected {}, got {}",
                            embedding_model_id,
                            output.embedding_model_id
                        );
                    }
                    if output.vectors.len() != 1 {
                        anyhow::bail!(
                            "semantic search query embedding returned {} vectors, expected 1",
                            output.vectors.len()
                        );
                    }
                    let query_vector = output
                        .vectors
                        .into_iter()
                        .next()
                        .expect("validated semantic query embedding should be present");
                    if query_vector.embedding.is_empty() {
                        anyhow::bail!("semantic search query embedding vector is empty");
                    }
                    if query_vector.embedding.len() != output.dimension {
                        anyhow::bail!(
                            "semantic search query embedding dimension mismatch: expected {}, got {}",
                            output.dimension,
                            query_vector.embedding.len()
                        );
                    }
                    if query_vector.embedding.len() != ready_dimension {
                        anyhow::bail!(
                            "semantic search query embedding dimension mismatch with ready vector artifact: expected {}, got {}",
                            ready_dimension,
                            query_vector.embedding.len()
                        );
                    }
                    if query_vector
                        .embedding
                        .iter()
                        .any(|value| !value.is_finite())
                    {
                        anyhow::bail!(
                            "semantic search query embedding vector contains non-finite values"
                        );
                    }
                    params.push(
                        forge_domain::SearchParams::new(
                            &search_query.query,
                            &search_query.use_case,
                        )
                        .limit(limit)
                        .top_k(top_k)
                        .query_embedding(query_vector.embedding)
                        .embedding_model_id(output.embedding_model_id),
                    );
                }

                // Execute all queries in parallel
                let futures: Vec<_> = params
                    .into_iter()
                    .map(|param| services.query_workspace(cwd.clone(), param))
                    .collect();

                let mut results = futures::future::try_join_all(futures).await?;

                // Deduplicate results across queries
                crate::search_dedup::deduplicate_results(&mut results);

                let output = input
                    .queries
                    .into_iter()
                    .zip(results)
                    .map(|(query, results)| CodebaseQueryResult {
                        query: query.query,
                        use_case: query.use_case,
                        results,
                    })
                    .collect::<Vec<_>>();

                let output = forge_domain::CodebaseSearchResults { queries: output };
                ToolOperation::CodebaseSearch { output }
            }
            ToolCatalog::WorkspaceVectorIndexBuildContinuation(input) => {
                let config = self.services.get_config()?;
                let env = self.services.get_environment();
                let workspace_root = canonicalize_workspace_build_path(
                    input.workspace_path.as_path(),
                    env.cwd.as_path(),
                )?;
                let configured_model_id =
                    config
                        .semantic_embedding_model_id
                        .clone()
                        .and_then(|model_id| {
                            let trimmed = model_id.trim().to_string();
                            (!trimmed.is_empty()).then_some(trimmed)
                        });
                let explicit_model_id = input.embedding_model_id.as_deref().map(str::trim);
                if explicit_model_id.is_some_and(str::is_empty) {
                    anyhow::bail!(
                        "workspace_vector_index_build_continuation rejected embedding_model_id: explicit model is not configured for this build path"
                    );
                }
                let Some(embedding_model_id) = configured_model_id.clone() else {
                    let preflight_diagnostic = self
                        .services
                        .sem_search_diagnostic(workspace_root.clone(), None)
                        .await?;
                    let post_build_diagnostic = preflight_diagnostic.clone();
                    let output = forge_domain::WorkspaceVectorIndexBuildContinuationReport {
                        preflight_diagnostic,
                        build_report: None,
                        post_build_diagnostic,
                        final_status: forge_domain::WorkspaceVectorIndexBuildContinuationStatus::NotBuiltConfigRequired,
                    };
                    return Ok(ToolOperation::WorkspaceVectorIndexBuildContinuation { output });
                };
                if let Some(explicit_model_id) = explicit_model_id
                    && explicit_model_id != embedding_model_id
                {
                    anyhow::bail!(
                        "workspace_vector_index_build_continuation rejected embedding_model_id: explicit model is not configured for this build path"
                    );
                }

                let preflight_diagnostic = self
                    .services
                    .sem_search_diagnostic(workspace_root.clone(), Some(embedding_model_id.clone()))
                    .await?;
                if preflight_diagnostic.status
                    != forge_domain::SemSearchDiagnosticStatus::VectorBuildSuggested
                    || !preflight_diagnostic.safe_to_suggest_build
                {
                    let final_status = forge_domain::WorkspaceVectorIndexBuildContinuationStatus::from_non_build_diagnostic_status(preflight_diagnostic.status);
                    let post_build_diagnostic = preflight_diagnostic.clone();
                    let output = forge_domain::WorkspaceVectorIndexBuildContinuationReport {
                        preflight_diagnostic,
                        build_report: None,
                        post_build_diagnostic,
                        final_status,
                    };
                    return Ok(ToolOperation::WorkspaceVectorIndexBuildContinuation { output });
                }

                let build_report = match self
                    .services
                    .build_workspace_vector_index(
                        workspace_root.clone(),
                        embedding_model_id.clone(),
                    )
                    .await
                {
                    Ok(report) => Some(report),
                    Err(_error) => {
                        let post_build_diagnostic = self
                            .services
                            .sem_search_diagnostic(workspace_root.clone(), Some(embedding_model_id))
                            .await?;
                        let output = forge_domain::WorkspaceVectorIndexBuildContinuationReport {
                            preflight_diagnostic,
                            build_report: None,
                            post_build_diagnostic,
                            final_status: forge_domain::WorkspaceVectorIndexBuildContinuationStatus::BuildFailed,
                        };
                        return Ok(ToolOperation::WorkspaceVectorIndexBuildContinuation { output });
                    }
                };
                let post_build_diagnostic = self
                    .services
                    .sem_search_diagnostic(workspace_root, Some(embedding_model_id))
                    .await?;
                let final_status = match post_build_diagnostic.status {
                    forge_domain::SemSearchDiagnosticStatus::Ready => {
                        forge_domain::WorkspaceVectorIndexBuildContinuationStatus::BuiltReady
                    }
                    status => forge_domain::WorkspaceVectorIndexBuildContinuationStatus::from_non_build_diagnostic_status(status),
                };
                let output = forge_domain::WorkspaceVectorIndexBuildContinuationReport {
                    preflight_diagnostic,
                    build_report,
                    post_build_diagnostic,
                    final_status,
                };
                ToolOperation::WorkspaceVectorIndexBuildContinuation { output }
            }
            ToolCatalog::Remove(input) => {
                let normalized_path = self.normalize_path(input.path.clone());
                let output = self.services.remove(normalized_path).await?;
                (input, output).into()
            }
            ToolCatalog::Patch(input) => {
                let normalized_path = self.normalize_path(input.file_path.clone());
                let output = self
                    .services
                    .patch(
                        normalized_path,
                        input.old_string.clone(),
                        input.new_string.clone(),
                        input.replace_all,
                    )
                    .await?;
                (input, output).into()
            }
            ToolCatalog::MultiPatch(input) => {
                let normalized_path = self.normalize_path(input.file_path.clone());
                let output = self
                    .services
                    .multi_patch(normalized_path, input.edits.clone())
                    .await?;
                (input, output).into()
            }
            ToolCatalog::Undo(input) => {
                let normalized_path = self.normalize_path(input.path.clone());
                let output = self.services.undo(normalized_path).await?;
                (input, output).into()
            }
            ToolCatalog::Shell(input) => {
                let execution_cwd = self.resolve_execution_cwd(input.cwd.as_ref());
                let output = self
                    .services
                    .execute(ShellExecuteRequest {
                        command: input.command.clone(),
                        cwd: execution_cwd,
                        keep_ansi: input.keep_ansi,
                        silent: false,
                        env_vars: input.env.clone(),
                        handoff_timeout: input.handoff_timeout_seconds.unwrap_or_default(),
                        description: input.description.clone(),
                    })
                    .await?;
                output.into()
            }
            ToolCatalog::ProcessStatus(input) => {
                let output = self
                    .services
                    .process_status(
                        forge_domain::ProcessId::parse(input.process_id.clone())?,
                        input.wait_seconds,
                    )
                    .await?;
                ToolOperation::ProcessStatus { output }
            }
            ToolCatalog::ProcessRead(input) => {
                let output = self
                    .services
                    .process_read(
                        forge_domain::ProcessId::parse(input.process_id.clone())?,
                        forge_domain::ProcessReadCursor::new(input.cursor),
                        input.wait_seconds,
                    )
                    .await?;
                ToolOperation::ProcessRead { output }
            }
            ToolCatalog::ProcessList(_input) => {
                let output = self.services.process_list().await?;
                ToolOperation::ProcessList { output }
            }
            ToolCatalog::ProcessKill(input) => {
                let output = self
                    .services
                    .process_kill(forge_domain::ProcessId::parse(input.process_id.clone())?)
                    .await?;
                ToolOperation::ProcessKill { output }
            }
            ToolCatalog::Fetch(input) => {
                let output = self.services.fetch(input.url.clone(), input.raw).await?;
                (input, output).into()
            }
            ToolCatalog::Followup(input) => {
                let output = self
                    .services
                    .follow_up(
                        input.question.clone(),
                        input
                            .option1
                            .clone()
                            .into_iter()
                            .chain(input.option2.clone())
                            .chain(input.option3.clone())
                            .chain(input.option4.clone())
                            .chain(input.option5.clone())
                            .collect(),
                        input.multiple,
                    )
                    .await?;
                output.into()
            }
            ToolCatalog::Plan(input) => {
                let output = self
                    .services
                    .create_plan(
                        input.plan_name.clone(),
                        input.version.clone(),
                        input.content.clone(),
                    )
                    .await?;
                (input, output).into()
            }
            ToolCatalog::Skill(input) => {
                let skill = self.services.fetch_skill(input.name.clone()).await?;
                ToolOperation::Skill { output: skill }
            }
            ToolCatalog::TodoWrite(input) => {
                let before = context.get_todos()?;
                context.update_todos(input.todos.clone())?;
                let after = context.get_todos()?;
                ToolOperation::TodoWrite { before, after }
            }
            ToolCatalog::TodoRead(_input) => {
                let todos = context.get_todos()?;
                ToolOperation::TodoRead { output: todos }
            }
            ToolCatalog::Task(_) => {
                // Task tools are handled in ToolRegistry before reaching here
                unreachable!("Task tool should be handled in ToolRegistry")
            }
        })
    }

    pub async fn execute(
        &self,
        tool_input: ToolCatalog,
        context: &ToolCallContext,
    ) -> anyhow::Result<ToolOutput> {
        let tool_kind = tool_input.kind();
        let env = self.services.get_environment();
        let config = self.services.get_config()?;

        // Enforce read-before-edit for patch operations
        let file_path = match &tool_input {
            ToolCatalog::Patch(input) => Some(&input.file_path),
            ToolCatalog::MultiPatch(input) => Some(&input.file_path),
            _ => None,
        };

        if let Some(path) = file_path {
            self.require_prior_read(context, path, "edit it")?;
        }

        // Enforce read-before-edit for overwrite writes
        if let ToolCatalog::Write(input) = &tool_input
            && input.overwrite
        {
            self.require_prior_read(context, &input.file_path, "overwrite it")?;
        }

        let execution_result = self.call_internal(tool_input.clone(), context).await;

        if let Err(ref error) = execution_result {
            tracing::error!(error = ?error, "Tool execution failed");
        }

        let operation = execution_result?;

        // Send formatted output message
        if let Some(output) = operation.to_content(&env) {
            context.send(output).await?;
        }

        let truncation_path = self.dump_operation(&operation).await?;

        context.with_metrics(|metrics| {
            operation.into_tool_output(tool_kind, truncation_path, &env, &config, metrics)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex as StdMutex};

    use forge_domain::{
        Agent, AgentId, AnyProvider, Attachment, AuthContextRequest, AuthContextResponse,
        AuthCredential, AuthMethod, ChatCompletionMessage, CodebaseQueryResult, ConfigOperation,
        Context, Conversation, ConversationId, ConversationListItem, Effort, Environment, File,
        FileStatus, LearningCaptureMetadata, LearningLedgerAppendOutcome, LearningLedgerEvent,
        LearningLedgerFreshness, LearningRecordId, LearningRecordProjection, LearningReviewOutcome,
        LearningReviewState, McpConfig, McpServers, Metrics, Model, ModelConfig, ModelId, Node,
        NodeData, NodeId, PermissionOperation, ProjectSemanticEmbeddingOutput,
        ProjectSemanticEmbeddingVector, Provider, ProviderId, ResultStream, Scope, SearchParams,
        SemSearchAvailability, SemSearchDiagnosticReport, SemSearchDiagnosticStatus,
        SemSearchUnknownReason, SemSearchUnsupportedReason, Shell, SteerMessage, SubagentTaskId,
        SubagentTaskSession, SubagentTaskSessionFilter, SyncProgress, ToolCallContext,
        ToolCallFull, WorkspaceAuth, WorkspaceContextFreshness, WorkspaceContextManifestDiagnostic,
        WorkspaceEvidenceReplayDiagnostic, WorkspaceEvidenceReplayPreviewDiagnostic,
        WorkspaceExactFactStatusReport, WorkspaceId, WorkspaceInfo,
        WorkspaceSemanticInjectionReadiness, WorkspaceVectorIndexBuildContinuationStatus,
        WorkspaceVectorIndexBuildReport, WorkspaceVectorIndexBuildStatus,
    };
    use pretty_assertions::assert_eq;
    use tokio::sync::Mutex;
    use url::Url;

    use crate::services::{
        AppConfigService, AttachmentService, AuthService, CommandLoaderService,
        CustomInstructionsService, FileDiscoveryService, LearningService, McpConfigManager,
        McpService, PolicyDecision, PolicyService, ProviderAuthService, SteerService,
        TemplateService,
    };

    use super::*;

    struct SemSearchParamSnapshot {
        query: String,
        use_case: String,
        query_embedding: Option<Vec<f32>>,
        embedding_model_id: Option<String>,
    }

    #[derive(Clone)]
    struct SemSearchFixture {
        config: forge_config::ForgeConfig,
        cwd: PathBuf,
        workspace: SemSearchWorkspace,
        unused: SemSearchUnusedService,
    }

    #[derive(Clone, Default)]
    struct SemSearchUnusedService;

    #[derive(Clone)]
    struct SemSearchWorkspace {
        embedding_calls: Arc<Mutex<Vec<(String, String)>>>,
        query_calls: Arc<Mutex<Vec<SemSearchParamSnapshot>>>,
        build_calls: Arc<AtomicUsize>,
        query_error: Option<String>,
        readiness: Arc<StdMutex<SemSearchAvailability>>,
        post_build_readiness: Arc<StdMutex<Option<SemSearchAvailability>>>,
    }

    impl SemSearchFixture {
        fn new(config: forge_config::ForgeConfig) -> Self {
            Self {
                config,
                cwd: PathBuf::from("/workspace"),
                workspace: SemSearchWorkspace {
                    embedding_calls: Arc::new(Mutex::new(Vec::new())),
                    query_calls: Arc::new(Mutex::new(Vec::new())),
                    build_calls: Arc::new(AtomicUsize::new(0)),
                    query_error: None,
                    readiness: Arc::new(StdMutex::new(SemSearchAvailability::Ready {
                        workspace_root: PathBuf::from("/workspace"),
                        manifest_hash: "fixture-manifest".to_string(),
                        vector_artifact_id: "fixture-vector-artifact".to_string(),
                        dimension: 2,
                    })),
                    post_build_readiness: Arc::new(StdMutex::new(None)),
                },
                unused: SemSearchUnusedService,
            }
        }

        fn with_query_error(mut self, error: &str) -> Self {
            self.workspace.query_error = Some(error.to_string());
            self
        }

        fn with_readiness(self, readiness: SemSearchAvailability) -> Self {
            *self.workspace.readiness.lock().unwrap() = readiness;
            self
        }

        fn with_cwd(mut self, cwd: PathBuf) -> Self {
            self.cwd = cwd;
            self
        }

        fn with_post_build_readiness(self, readiness: SemSearchAvailability) -> Self {
            *self.workspace.post_build_readiness.lock().unwrap() = Some(readiness);
            self
        }
    }

    impl EnvironmentInfra for SemSearchFixture {
        type Config = forge_config::ForgeConfig;

        fn get_env_var(&self, _key: &str) -> Option<String> {
            None
        }

        fn get_env_vars(&self) -> std::collections::BTreeMap<String, String> {
            std::collections::BTreeMap::new()
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

        fn get_config(&self) -> anyhow::Result<Self::Config> {
            Ok(self.config.clone())
        }

        async fn update_environment(&self, _ops: Vec<ConfigOperation>) -> anyhow::Result<()> {
            anyhow::bail!("unused environment update")
        }
    }

    #[async_trait::async_trait]
    impl WorkspaceService for SemSearchWorkspace {
        async fn sync_workspace(
            &self,
            _path: PathBuf,
        ) -> anyhow::Result<forge_stream::MpscStream<anyhow::Result<SyncProgress>>> {
            anyhow::bail!("sem_search must not sync workspace")
        }

        async fn produce_workspace_exact_fact_reference(
            &self,
            _path: PathBuf,
        ) -> anyhow::Result<forge_domain::WorkspaceExactFactReferenceReport> {
            anyhow::bail!("unused exact-fact reference")
        }

        async fn workspace_exact_fact_status(
            &self,
            _path: PathBuf,
        ) -> anyhow::Result<WorkspaceExactFactStatusReport> {
            anyhow::bail!("unused exact-fact status")
        }

        async fn workspace_evidence_replay_diagnostic(
            &self,
            _path: PathBuf,
        ) -> anyhow::Result<WorkspaceEvidenceReplayDiagnostic> {
            anyhow::bail!("unused evidence replay diagnostic")
        }

        async fn workspace_evidence_replay_preview_diagnostic(
            &self,
            _path: PathBuf,
        ) -> anyhow::Result<WorkspaceEvidenceReplayPreviewDiagnostic> {
            anyhow::bail!("unused evidence replay preview diagnostic")
        }

        async fn build_workspace_vector_index(
            &self,
            _path: PathBuf,
            embedding_model_id: String,
        ) -> anyhow::Result<WorkspaceVectorIndexBuildReport> {
            self.build_calls.fetch_add(1, Ordering::SeqCst);
            if let Some(readiness) = self.post_build_readiness.lock().unwrap().clone() {
                *self.readiness.lock().unwrap() = readiness;
                return Ok(WorkspaceVectorIndexBuildReport {
                    status: WorkspaceVectorIndexBuildStatus::ArtifactWritten,
                    artifact_path: PathBuf::from(
                        "/workspace/.forge_project_model/vector-indexes/fixture.json",
                    ),
                    artifact_id: "fixture-vector-artifact".to_string(),
                    embedding_model_id,
                    dimension: 2,
                    entry_count: 1,
                    manifest_hash: "fixture-manifest".to_string(),
                });
            }
            anyhow::bail!("sem_search must not build workspace vector indexes")
        }

        async fn embed_workspace_query(
            &self,
            query: String,
            embedding_model_id: String,
        ) -> anyhow::Result<ProjectSemanticEmbeddingOutput> {
            self.embedding_calls
                .lock()
                .await
                .push((query, embedding_model_id.clone()));
            Ok(ProjectSemanticEmbeddingOutput {
                embedding_model_id,
                dimension: 2,
                vectors: vec![ProjectSemanticEmbeddingVector {
                    source_id: "query".to_string(),
                    source_fingerprint: "query".to_string(),
                    embedding: vec![1.0, 0.0],
                }],
            })
        }

        async fn semantic_injection_readiness(
            &self,
            _path: PathBuf,
            _embedding_model_id: Option<String>,
        ) -> anyhow::Result<WorkspaceSemanticInjectionReadiness> {
            Ok(WorkspaceSemanticInjectionReadiness::VectorIndexReady { dimension: 2 })
        }

        async fn sem_search_availability(
            &self,
            _path: PathBuf,
            embedding_model_id: Option<String>,
        ) -> anyhow::Result<SemSearchAvailability> {
            if embedding_model_id
                .as_deref()
                .filter(|model_id| !model_id.trim().is_empty())
                .is_none()
            {
                return Ok(SemSearchAvailability::Unsupported {
                    reason: SemSearchUnsupportedReason::NoModelConfig,
                });
            }
            Ok(self.readiness.lock().unwrap().clone())
        }

        async fn sem_search_diagnostic(
            &self,
            path: PathBuf,
            embedding_model_id: Option<String>,
        ) -> anyhow::Result<SemSearchDiagnosticReport> {
            let availability = self
                .sem_search_availability(path.clone(), embedding_model_id.clone())
                .await?;
            Ok(SemSearchDiagnosticReport::from_availability(
                &availability,
                embedding_model_id.as_deref(),
                &path,
            ))
        }

        async fn query_workspace(
            &self,
            _path: PathBuf,
            params: SearchParams<'_>,
        ) -> anyhow::Result<Vec<Node>> {
            self.query_calls.lock().await.push(SemSearchParamSnapshot {
                query: params.query.to_string(),
                use_case: params.use_case.clone(),
                query_embedding: params.query_embedding.clone(),
                embedding_model_id: params.embedding_model_id.clone(),
            });
            if let Some(error) = &self.query_error {
                anyhow::bail!(error.clone());
            }
            if params.query_embedding.is_none() || params.embedding_model_id.is_none() {
                anyhow::bail!("semantic query parameters were not populated")
            }
            Ok(vec![Node {
                node_id: NodeId::new("semantic-vector-only-hit"),
                node: NodeData::FileChunk(forge_domain::FileChunk {
                    file_path: "src/vector_only.rs".to_string(),
                    content: "pub struct SemanticVectorOnlyHit;".to_string(),
                    start_line: 1,
                    end_line: 1,
                }),
                relevance: Some(1.0),
                distance: None,
            }])
        }

        async fn list_workspaces(&self) -> anyhow::Result<Vec<WorkspaceInfo>> {
            Ok(Vec::new())
        }

        async fn get_workspace_info(
            &self,
            _path: PathBuf,
        ) -> anyhow::Result<Option<WorkspaceInfo>> {
            Ok(None)
        }

        async fn is_indexed(&self, _path: &Path) -> anyhow::Result<bool> {
            Ok(true)
        }

        async fn delete_workspace(&self, _workspace_id: &WorkspaceId) -> anyhow::Result<()> {
            Ok(())
        }

        async fn delete_workspaces(&self, _workspace_ids: &[WorkspaceId]) -> anyhow::Result<()> {
            Ok(())
        }

        async fn project_model_context_diagnostic(
            &self,
            path: &Path,
        ) -> anyhow::Result<WorkspaceContextManifestDiagnostic> {
            Ok(WorkspaceContextManifestDiagnostic {
                workspace_root: path.to_path_buf(),
                manifest_path: path.join(".forge_project_model/project_manifest.json"),
                manifest_found: true,
                freshness: WorkspaceContextFreshness::Fresh,
                exact_fact_readiness: None,
                evidence_readiness: None,
                evidence_ledger_activation: None,
            })
        }

        async fn get_workspace_status(&self, _path: PathBuf) -> anyhow::Result<Vec<FileStatus>> {
            Ok(Vec::new())
        }

        async fn is_authenticated(&self) -> anyhow::Result<bool> {
            Ok(true)
        }

        async fn init_auth_credentials(&self) -> anyhow::Result<WorkspaceAuth> {
            anyhow::bail!("unused workspace auth")
        }

        async fn init_workspace(&self, _path: PathBuf) -> anyhow::Result<WorkspaceId> {
            anyhow::bail!("unused workspace init")
        }
    }

    macro_rules! impl_sem_search_unused_services {
        ($type:ty) => {
            #[async_trait::async_trait]
            impl ProviderService for $type {
                async fn chat(
                    &self,
                    _model_id: &ModelId,
                    _context: Context,
                    _provider: Provider<Url>,
                ) -> ResultStream<ChatCompletionMessage, anyhow::Error> {
                    Ok(Box::pin(tokio_stream::iter(std::iter::empty())))
                }
                async fn models(&self, _provider: Provider<Url>) -> anyhow::Result<Vec<Model>> {
                    Ok(Vec::new())
                }
                async fn get_provider(&self, _id: ProviderId) -> anyhow::Result<Provider<Url>> {
                    anyhow::bail!("unused provider lookup")
                }
                async fn get_all_providers(&self) -> anyhow::Result<Vec<AnyProvider>> {
                    Ok(Vec::new())
                }
                async fn upsert_credential(
                    &self,
                    _credential: AuthCredential,
                ) -> anyhow::Result<()> {
                    anyhow::bail!("unused credential upsert")
                }
                async fn remove_credential(&self, _id: &ProviderId) -> anyhow::Result<()> {
                    anyhow::bail!("unused credential remove")
                }
                async fn migrate_env_credentials(
                    &self,
                ) -> anyhow::Result<Option<forge_domain::MigrationResult>> {
                    Ok(None)
                }
            }

            #[async_trait::async_trait]
            impl AppConfigService for $type {
                async fn get_session_config(&self) -> Option<ModelConfig> {
                    None
                }
                async fn get_commit_config(&self) -> anyhow::Result<Option<ModelConfig>> {
                    Ok(None)
                }
                async fn get_suggest_config(&self) -> anyhow::Result<Option<ModelConfig>> {
                    Ok(None)
                }
                async fn get_reasoning_effort(&self) -> anyhow::Result<Option<Effort>> {
                    Ok(None)
                }
                async fn update_config(&self, _ops: Vec<ConfigOperation>) -> anyhow::Result<()> {
                    anyhow::bail!("unused app config update")
                }
            }

            #[async_trait::async_trait]
            impl ConversationService for $type {
                async fn find_conversation(
                    &self,
                    _id: &ConversationId,
                ) -> anyhow::Result<Option<Conversation>> {
                    Ok(None)
                }
                async fn upsert_conversation(
                    &self,
                    _conversation: Conversation,
                ) -> anyhow::Result<()> {
                    Ok(())
                }
                async fn ensure_delegated_conversation(
                    &self,
                    _id: &ConversationId,
                    _parent_id: Option<ConversationId>,
                ) -> anyhow::Result<Conversation> {
                    anyhow::bail!("unused delegated conversation")
                }
                async fn resolve_root_conversation_id(
                    &self,
                    _parent_id: Option<ConversationId>,
                ) -> anyhow::Result<Option<ConversationId>> {
                    Ok(None)
                }
                async fn modify_conversation<F, T>(
                    &self,
                    _id: &ConversationId,
                    _f: F,
                ) -> anyhow::Result<T>
                where
                    F: FnOnce(&mut Conversation) -> T + Send,
                    T: Send,
                {
                    anyhow::bail!("unused conversation modify")
                }
                async fn list_branch_targets(
                    &self,
                    _conversation_id: &ConversationId,
                ) -> anyhow::Result<Vec<crate::dto::ConversationBranchTarget>> {
                    Ok(Vec::new())
                }
                async fn branch_conversation(
                    &self,
                    _conversation_id: &ConversationId,
                    _target_id: forge_domain::MessageId,
                ) -> anyhow::Result<Conversation> {
                    anyhow::bail!("unused branch conversation")
                }
                async fn get_conversation_list_items_by_query(
                    &self,
                    _query: forge_domain::ConversationListQuery,
                ) -> anyhow::Result<Vec<ConversationListItem>> {
                    Ok(Vec::new())
                }
                async fn get_conversation_list_items_including_agent(
                    &self,
                    _limit: usize,
                ) -> anyhow::Result<Vec<ConversationListItem>> {
                    Ok(Vec::new())
                }
                async fn get_conversation_list_items_by_visibility(
                    &self,
                    _visibility: forge_domain::ConversationVisibilityFilter,
                    _limit: usize,
                ) -> anyhow::Result<Vec<ConversationListItem>> {
                    Ok(Vec::new())
                }
                async fn get_conversations(&self) -> anyhow::Result<Vec<Conversation>> {
                    Ok(Vec::new())
                }
                async fn get_conversations_including_agent(
                    &self,
                ) -> anyhow::Result<Vec<Conversation>> {
                    Ok(Vec::new())
                }
                async fn get_conversations_by_visibility(
                    &self,
                    _visibility: forge_domain::ConversationVisibilityFilter,
                ) -> anyhow::Result<Vec<Conversation>> {
                    Ok(Vec::new())
                }
                async fn get_sub_conversations(
                    &self,
                    _parent_id: &ConversationId,
                ) -> anyhow::Result<Vec<Conversation>> {
                    Ok(Vec::new())
                }
                async fn upsert_subagent_task_session(
                    &self,
                    _session: SubagentTaskSession,
                ) -> anyhow::Result<()> {
                    Ok(())
                }
                async fn get_subagent_task_session(
                    &self,
                    _task_id: &SubagentTaskId,
                ) -> anyhow::Result<Option<SubagentTaskSession>> {
                    Ok(None)
                }
                async fn get_subagent_task_session_by_conversation(
                    &self,
                    _conversation_id: &ConversationId,
                ) -> anyhow::Result<Option<SubagentTaskSession>> {
                    Ok(None)
                }
                async fn list_subagent_task_sessions(
                    &self,
                    _filter: SubagentTaskSessionFilter,
                ) -> anyhow::Result<Vec<SubagentTaskSession>> {
                    Ok(Vec::new())
                }
                async fn last_conversation(&self) -> anyhow::Result<Option<Conversation>> {
                    Ok(None)
                }
                async fn delete_conversation(
                    &self,
                    _conversation_id: &ConversationId,
                ) -> anyhow::Result<()> {
                    Ok(())
                }
            }

            #[async_trait::async_trait]
            impl LearningService for $type {
                async fn capture_candidate_from_conversation(
                    &self,
                    _conversation_id: ConversationId,
                    _source_event_id: String,
                    _summary: String,
                    _metadata: LearningCaptureMetadata,
                ) -> anyhow::Result<LearningLedgerAppendOutcome> {
                    anyhow::bail!("unused learning capture")
                }
                async fn insert_learning_event(
                    &self,
                    _event: LearningLedgerEvent,
                ) -> anyhow::Result<LearningLedgerAppendOutcome> {
                    anyhow::bail!("unused learning insert")
                }
                async fn review_learning_candidate_event(
                    &self,
                    _event: LearningLedgerEvent,
                ) -> anyhow::Result<LearningReviewOutcome> {
                    anyhow::bail!("unused learning review")
                }
                async fn promote_sensor_lesson(
                    &self,
                    _request: forge_domain::SensorLessonPromotionRequest,
                ) -> anyhow::Result<forge_domain::SensorLessonPromotionOutcome> {
                    anyhow::bail!("unused learning promotion")
                }
                async fn get_learning_record(
                    &self,
                    _record_id: LearningRecordId,
                ) -> anyhow::Result<Option<LearningRecordProjection>> {
                    Ok(None)
                }
                async fn list_learning_records(
                    &self,
                    _review_state: Option<LearningReviewState>,
                    _limit: usize,
                ) -> anyhow::Result<Vec<LearningRecordProjection>> {
                    Ok(Vec::new())
                }
                async fn learning_freshness(
                    &self,
                    _review_state: Option<LearningReviewState>,
                ) -> anyhow::Result<LearningLedgerFreshness> {
                    anyhow::bail!("unused learning freshness")
                }
            }

            #[async_trait::async_trait]
            impl SteerService for $type {
                async fn enqueue_steer(
                    &self,
                    _conversation_id: &ConversationId,
                    _message: SteerMessage,
                ) -> anyhow::Result<()> {
                    Ok(())
                }
                async fn clear_steer(
                    &self,
                    _conversation_id: &ConversationId,
                ) -> anyhow::Result<()> {
                    Ok(())
                }
                async fn drain_steer(
                    &self,
                    _conversation_id: &ConversationId,
                ) -> anyhow::Result<Vec<SteerMessage>> {
                    Ok(Vec::new())
                }
            }

            #[async_trait::async_trait]
            impl TemplateService for $type {
                async fn register_template(&self, _path: PathBuf) -> anyhow::Result<()> {
                    Ok(())
                }
                async fn render_template<V: serde::Serialize + Send + Sync>(
                    &self,
                    _template: forge_domain::Template<V>,
                    _object: &V,
                ) -> anyhow::Result<String> {
                    anyhow::bail!("unused template rendering")
                }
            }

            #[async_trait::async_trait]
            impl AttachmentService for $type {
                async fn attachments(&self, _url: &str) -> anyhow::Result<Vec<Attachment>> {
                    Ok(Vec::new())
                }
            }

            #[async_trait::async_trait]
            impl CustomInstructionsService for $type {
                async fn get_custom_instructions(&self) -> Vec<String> {
                    Vec::new()
                }
            }

            #[async_trait::async_trait]
            impl FileDiscoveryService for $type {
                async fn collect_files(&self, _config: crate::Walker) -> anyhow::Result<Vec<File>> {
                    Ok(Vec::new())
                }
                async fn list_current_directory(&self) -> anyhow::Result<Vec<File>> {
                    Ok(Vec::new())
                }
            }

            #[async_trait::async_trait]
            impl McpConfigManager for $type {
                async fn read_mcp_config(
                    &self,
                    _scope: Option<&Scope>,
                ) -> anyhow::Result<McpConfig> {
                    Ok(McpConfig::default())
                }
                async fn write_mcp_config(
                    &self,
                    _config: &McpConfig,
                    _scope: &Scope,
                ) -> anyhow::Result<()> {
                    Ok(())
                }
            }

            #[async_trait::async_trait]
            impl FsWriteService for $type {
                async fn write(
                    &self,
                    _path: String,
                    _content: String,
                    _overwrite: bool,
                ) -> anyhow::Result<crate::FsWriteOutput> {
                    anyhow::bail!("unused fs write")
                }
            }
            #[async_trait::async_trait]
            impl PlanCreateService for $type {
                async fn create_plan(
                    &self,
                    _plan_name: String,
                    _version: String,
                    _content: String,
                ) -> anyhow::Result<crate::PlanCreateOutput> {
                    anyhow::bail!("unused plan create")
                }
            }
            #[async_trait::async_trait]
            impl FsPatchService for $type {
                async fn patch(
                    &self,
                    _path: String,
                    _search: String,
                    _content: String,
                    _replace_all: bool,
                ) -> anyhow::Result<crate::PatchOutput> {
                    anyhow::bail!("unused fs patch")
                }
                async fn multi_patch(
                    &self,
                    _path: String,
                    _edits: Vec<forge_domain::PatchEdit>,
                ) -> anyhow::Result<crate::PatchOutput> {
                    anyhow::bail!("unused fs multi patch")
                }
            }
            #[async_trait::async_trait]
            impl FsReadService for $type {
                async fn read(
                    &self,
                    _path: String,
                    _start_line: Option<u64>,
                    _end_line: Option<u64>,
                ) -> anyhow::Result<crate::ReadOutput> {
                    anyhow::bail!("unused fs read")
                }
            }
            #[async_trait::async_trait]
            impl ImageReadService for $type {
                async fn read_image(&self, _path: String) -> anyhow::Result<forge_domain::Image> {
                    anyhow::bail!("unused image read")
                }
            }
            #[async_trait::async_trait]
            impl FsRemoveService for $type {
                async fn remove(&self, _path: String) -> anyhow::Result<crate::FsRemoveOutput> {
                    anyhow::bail!("unused fs remove")
                }
            }
            #[async_trait::async_trait]
            impl FsSearchService for $type {
                async fn search(
                    &self,
                    _params: forge_domain::FSSearch,
                ) -> anyhow::Result<Option<crate::SearchResult>> {
                    Ok(None)
                }
            }
            #[async_trait::async_trait]
            impl FollowUpService for $type {
                async fn follow_up(
                    &self,
                    _question: String,
                    _options: Vec<String>,
                    _multiple: Option<bool>,
                ) -> anyhow::Result<Option<String>> {
                    Ok(None)
                }
            }
            #[async_trait::async_trait]
            impl FsUndoService for $type {
                async fn undo(&self, _path: String) -> anyhow::Result<crate::FsUndoOutput> {
                    anyhow::bail!("unused fs undo")
                }
            }
            #[async_trait::async_trait]
            impl NetFetchService for $type {
                async fn fetch(
                    &self,
                    _url: String,
                    _raw: Option<bool>,
                ) -> anyhow::Result<crate::HttpResponse> {
                    anyhow::bail!("unused net fetch")
                }
            }
            #[async_trait::async_trait]
            impl ShellService for $type {
                async fn execute(
                    &self,
                    _request: crate::ShellExecuteRequest,
                ) -> anyhow::Result<crate::ShellOutput> {
                    anyhow::bail!("unused shell execute")
                }
            }
            #[async_trait::async_trait]
            impl McpService for $type {
                async fn get_mcp_servers(&self) -> anyhow::Result<McpServers> {
                    Ok(McpServers::default())
                }
                async fn execute_mcp(
                    &self,
                    _call: ToolCallFull,
                ) -> anyhow::Result<forge_domain::ToolOutput> {
                    anyhow::bail!("unused mcp execute")
                }
                async fn reload_mcp(&self) -> anyhow::Result<()> {
                    Ok(())
                }
            }
            #[async_trait::async_trait]
            impl AuthService for $type {
                async fn user_info(&self, _api_key: &str) -> anyhow::Result<crate::user::User> {
                    anyhow::bail!("unused user info")
                }
                async fn user_usage(
                    &self,
                    _api_key: &str,
                ) -> anyhow::Result<crate::user::UserUsage> {
                    anyhow::bail!("unused user usage")
                }
            }
            #[async_trait::async_trait]
            impl AgentRegistry for $type {
                async fn get_active_agent_id(&self) -> anyhow::Result<Option<AgentId>> {
                    Ok(None)
                }
                async fn set_active_agent_id(&self, _agent_id: AgentId) -> anyhow::Result<()> {
                    Ok(())
                }
                async fn get_agents(&self) -> anyhow::Result<Vec<Agent>> {
                    Ok(Vec::new())
                }
                async fn get_agent_infos(&self) -> anyhow::Result<Vec<forge_domain::AgentInfo>> {
                    Ok(Vec::new())
                }
                async fn get_agent(&self, _agent_id: &AgentId) -> anyhow::Result<Option<Agent>> {
                    Ok(None)
                }
                async fn reload_agents(&self) -> anyhow::Result<()> {
                    Ok(())
                }
            }
            #[async_trait::async_trait]
            impl CommandLoaderService for $type {
                async fn get_commands(&self) -> anyhow::Result<Vec<forge_domain::Command>> {
                    Ok(Vec::new())
                }
            }
            #[async_trait::async_trait]
            impl PolicyService for $type {
                async fn check_operation_permission(
                    &self,
                    _operation: &PermissionOperation,
                ) -> anyhow::Result<PolicyDecision> {
                    Ok(PolicyDecision { allowed: true, path: None })
                }
            }
            #[async_trait::async_trait]
            impl ProviderAuthService for $type {
                async fn init_provider_auth(
                    &self,
                    _provider_id: ProviderId,
                    _method: AuthMethod,
                ) -> anyhow::Result<AuthContextRequest> {
                    anyhow::bail!("unused provider auth init")
                }
                async fn complete_provider_auth(
                    &self,
                    _provider_id: ProviderId,
                    _context: AuthContextResponse,
                    _timeout: std::time::Duration,
                ) -> anyhow::Result<()> {
                    anyhow::bail!("unused provider auth complete")
                }
                async fn refresh_provider_credential(
                    &self,
                    provider: Provider<Url>,
                ) -> anyhow::Result<Provider<Url>> {
                    Ok(provider)
                }
            }
            #[async_trait::async_trait]
            impl SkillFetchService for $type {
                async fn fetch_skill(
                    &self,
                    _skill_name: String,
                ) -> anyhow::Result<forge_domain::Skill> {
                    anyhow::bail!("unused skill fetch")
                }
                async fn list_skills(&self) -> anyhow::Result<Vec<forge_domain::Skill>> {
                    Ok(Vec::new())
                }
            }
        };
    }

    impl_sem_search_unused_services!(SemSearchUnusedService);

    impl Services for SemSearchFixture {
        type ProviderService = SemSearchUnusedService;
        type AppConfigService = SemSearchUnusedService;
        type ConversationService = SemSearchUnusedService;
        type LearningService = SemSearchUnusedService;
        type SteerService = SemSearchUnusedService;
        type TemplateService = SemSearchUnusedService;
        type AttachmentService = SemSearchUnusedService;
        type CustomInstructionsService = SemSearchUnusedService;
        type FileDiscoveryService = SemSearchUnusedService;
        type McpConfigManager = SemSearchUnusedService;
        type FsWriteService = SemSearchUnusedService;
        type PlanCreateService = SemSearchUnusedService;
        type FsPatchService = SemSearchUnusedService;
        type FsReadService = SemSearchUnusedService;
        type ImageReadService = SemSearchUnusedService;
        type FsRemoveService = SemSearchUnusedService;
        type FsSearchService = SemSearchUnusedService;
        type FollowUpService = SemSearchUnusedService;
        type FsUndoService = SemSearchUnusedService;
        type NetFetchService = SemSearchUnusedService;
        type ShellService = SemSearchUnusedService;
        type McpService = SemSearchUnusedService;
        type AuthService = SemSearchUnusedService;
        type AgentRegistry = SemSearchUnusedService;
        type CommandLoaderService = SemSearchUnusedService;
        type PolicyService = SemSearchUnusedService;
        type ProviderAuthService = SemSearchUnusedService;
        type WorkspaceService = SemSearchWorkspace;
        type SkillFetchService = SemSearchUnusedService;

        fn provider_service(&self) -> &Self::ProviderService {
            &self.unused
        }
        fn config_service(&self) -> &Self::AppConfigService {
            &self.unused
        }
        fn conversation_service(&self) -> &Self::ConversationService {
            &self.unused
        }
        fn learning_service(&self) -> &Self::LearningService {
            &self.unused
        }
        fn steer_service(&self) -> &Self::SteerService {
            &self.unused
        }
        fn template_service(&self) -> &Self::TemplateService {
            &self.unused
        }
        fn attachment_service(&self) -> &Self::AttachmentService {
            &self.unused
        }
        fn file_discovery_service(&self) -> &Self::FileDiscoveryService {
            &self.unused
        }
        fn mcp_config_manager(&self) -> &Self::McpConfigManager {
            &self.unused
        }
        fn fs_create_service(&self) -> &Self::FsWriteService {
            &self.unused
        }
        fn plan_create_service(&self) -> &Self::PlanCreateService {
            &self.unused
        }
        fn fs_patch_service(&self) -> &Self::FsPatchService {
            &self.unused
        }
        fn fs_read_service(&self) -> &Self::FsReadService {
            &self.unused
        }
        fn image_read_service(&self) -> &Self::ImageReadService {
            &self.unused
        }
        fn fs_remove_service(&self) -> &Self::FsRemoveService {
            &self.unused
        }
        fn fs_search_service(&self) -> &Self::FsSearchService {
            &self.unused
        }
        fn follow_up_service(&self) -> &Self::FollowUpService {
            &self.unused
        }
        fn fs_undo_service(&self) -> &Self::FsUndoService {
            &self.unused
        }
        fn net_fetch_service(&self) -> &Self::NetFetchService {
            &self.unused
        }
        fn shell_service(&self) -> &Self::ShellService {
            &self.unused
        }
        fn mcp_service(&self) -> &Self::McpService {
            &self.unused
        }
        fn custom_instructions_service(&self) -> &Self::CustomInstructionsService {
            &self.unused
        }
        fn auth_service(&self) -> &Self::AuthService {
            &self.unused
        }
        fn agent_registry(&self) -> &Self::AgentRegistry {
            &self.unused
        }
        fn command_loader_service(&self) -> &Self::CommandLoaderService {
            &self.unused
        }
        fn policy_service(&self) -> &Self::PolicyService {
            &self.unused
        }
        fn provider_auth_service(&self) -> &Self::ProviderAuthService {
            &self.unused
        }
        fn workspace_service(&self) -> &Self::WorkspaceService {
            &self.workspace
        }
        fn skill_fetch_service(&self) -> &Self::SkillFetchService {
            &self.unused
        }
    }

    fn sem_search_config(model_id: Option<&str>) -> forge_config::ForgeConfig {
        forge_config::ForgeConfig {
            max_sem_search_results: 7,
            sem_search_top_k: 3,
            semantic_embedding_model_id: model_id.map(str::to_string),
            ..Default::default()
        }
    }

    fn sem_search_tool(query: &str) -> ToolCatalog {
        ToolCatalog::SemSearch(forge_domain::SemanticSearch {
            queries: vec![forge_domain::SearchQuery::new(
                query,
                "Find the struct implementation for semantic vector-only retrieval",
            )],
        })
    }

    fn tool_context() -> ToolCallContext {
        ToolCallContext::new(Metrics::default())
    }

    #[tokio::test]
    async fn sem_search_embeds_each_query_and_passes_vector_params_to_workspace_query()
    -> anyhow::Result<()> {
        let setup = Arc::new(SemSearchFixture::new(sem_search_config(Some(
            "fixture-model",
        ))));
        let executor = ToolExecutor::new(Arc::clone(&setup));
        let input = ToolCatalog::SemSearch(forge_domain::SemanticSearch {
            queries: vec![
                forge_domain::SearchQuery::new(
                    "alpha behavior",
                    "Find the struct implementation for alpha",
                ),
                forge_domain::SearchQuery::new(
                    "beta behavior",
                    "Find the function implementation for beta",
                ),
            ],
        });

        let actual = executor.call_internal(input, &tool_context()).await?;

        match actual {
            ToolOperation::CodebaseSearch { output } => {
                assert_eq!(output.queries.len(), 2);
            }
            _ => panic!("expected semantic codebase search output"),
        }
        let actual_embeddings = setup.workspace.embedding_calls.lock().await.clone();
        let actual_queries = setup.workspace.query_calls.lock().await;
        let expected_embeddings = vec![
            ("alpha behavior".to_string(), "fixture-model".to_string()),
            ("beta behavior".to_string(), "fixture-model".to_string()),
        ];
        assert_eq!(actual_embeddings, expected_embeddings);
        assert_eq!(actual_queries.len(), 2);
        assert_eq!(actual_queries[0].query, "alpha behavior");
        assert_eq!(
            actual_queries[0].use_case,
            "Find the struct implementation for alpha"
        );
        assert_eq!(actual_queries[0].query_embedding, Some(vec![1.0, 0.0]));
        assert_eq!(
            actual_queries[0].embedding_model_id,
            Some("fixture-model".to_string())
        );
        assert_eq!(setup.workspace.build_calls.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[tokio::test]
    async fn sem_search_returns_vector_only_hit_for_lexical_miss() -> anyhow::Result<()> {
        let setup = Arc::new(SemSearchFixture::new(sem_search_config(Some(
            "fixture-model",
        ))));
        let executor = ToolExecutor::new(setup);

        let actual = executor
            .call_internal(
                sem_search_tool("words that do not appear in SemanticVectorOnlyHit"),
                &tool_context(),
            )
            .await?;

        let actual = match actual {
            ToolOperation::CodebaseSearch { output } => output,
            _ => panic!("expected semantic codebase search output"),
        };
        let expected = vec![CodebaseQueryResult {
            query: "words that do not appear in SemanticVectorOnlyHit".to_string(),
            use_case: "Find the struct implementation for semantic vector-only retrieval"
                .to_string(),
            results: vec![Node {
                node_id: NodeId::new("semantic-vector-only-hit"),
                node: NodeData::FileChunk(forge_domain::FileChunk {
                    file_path: "src/vector_only.rs".to_string(),
                    content: "pub struct SemanticVectorOnlyHit;".to_string(),
                    start_line: 1,
                    end_line: 1,
                }),
                relevance: Some(1.0),
                distance: None,
            }],
        }];
        assert_eq!(actual.queries, expected);
        Ok(())
    }

    #[tokio::test]
    async fn sem_search_missing_default_model_fails_before_query_execution() -> anyhow::Result<()> {
        let setup = Arc::new(SemSearchFixture::new(sem_search_config(None)));
        let executor = ToolExecutor::new(Arc::clone(&setup));

        let actual = executor
            .call_internal(sem_search_tool("semantic unavailable"), &tool_context())
            .await;

        assert!(
            actual
                .unwrap_err()
                .to_string()
                .contains("sem_search unavailable: unsupported: no_model_config")
        );
        assert_eq!(setup.workspace.embedding_calls.lock().await.len(), 0);
        assert_eq!(setup.workspace.query_calls.lock().await.len(), 0);
        assert_eq!(setup.workspace.build_calls.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[tokio::test]
    async fn sem_search_propagates_vector_index_unavailability_without_lexical_fallback()
    -> anyhow::Result<()> {
        let setup = Arc::new(
            SemSearchFixture::new(sem_search_config(Some("fixture-model"))).with_query_error(
                "Workspace project model vector retrieval unavailable: AmbiguousVectorIndex",
            ),
        );
        let executor = ToolExecutor::new(Arc::clone(&setup));

        let actual = executor
            .call_internal(
                sem_search_tool("lexical text that could otherwise match"),
                &tool_context(),
            )
            .await;

        assert!(
            actual
                .unwrap_err()
                .to_string()
                .contains("AmbiguousVectorIndex")
        );
        assert_eq!(setup.workspace.embedding_calls.lock().await.len(), 1);
        assert_eq!(setup.workspace.query_calls.lock().await.len(), 1);
        assert_eq!(setup.workspace.build_calls.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[tokio::test]
    async fn sem_search_unknown_readiness_fails_before_embedding_provider() -> anyhow::Result<()> {
        let setup = Arc::new(
            SemSearchFixture::new(sem_search_config(Some("fixture-model"))).with_readiness(
                SemSearchAvailability::Unknown {
                    reason: SemSearchUnknownReason::AmbiguousVectorArtifact,
                },
            ),
        );
        let executor = ToolExecutor::new(Arc::clone(&setup));

        let actual = executor
            .call_internal(sem_search_tool("semantic unavailable"), &tool_context())
            .await;

        assert!(
            actual
                .unwrap_err()
                .to_string()
                .contains("sem_search unavailable: unknown: ambiguous_vector_artifact")
        );
        assert_eq!(setup.workspace.embedding_calls.lock().await.len(), 0);
        assert_eq!(setup.workspace.query_calls.lock().await.len(), 0);
        assert_eq!(setup.workspace.build_calls.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[tokio::test]
    async fn sem_search_ready_dimension_mismatch_fails_after_embedding_before_query()
    -> anyhow::Result<()> {
        let setup = Arc::new(
            SemSearchFixture::new(sem_search_config(Some("fixture-model"))).with_readiness(
                SemSearchAvailability::Ready {
                    workspace_root: PathBuf::from("/workspace"),
                    manifest_hash: "fixture-manifest".to_string(),
                    vector_artifact_id: "fixture-vector-artifact".to_string(),
                    dimension: 3,
                },
            ),
        );
        let executor = ToolExecutor::new(Arc::clone(&setup));

        let actual = executor
            .call_internal(
                sem_search_tool("semantic dimension mismatch"),
                &tool_context(),
            )
            .await;

        assert!(
            actual
                .unwrap_err()
                .to_string()
                .contains("dimension mismatch with ready vector artifact")
        );
        assert_eq!(setup.workspace.embedding_calls.lock().await.len(), 1);
        assert_eq!(setup.workspace.query_calls.lock().await.len(), 0);
        assert_eq!(setup.workspace.build_calls.load(Ordering::SeqCst), 0);
        Ok(())
    }

    fn workspace_vector_build_tool(
        workspace_path: PathBuf,
        embedding_model_id: Option<&str>,
    ) -> ToolCatalog {
        ToolCatalog::WorkspaceVectorIndexBuildContinuation(
            forge_domain::WorkspaceVectorIndexBuildContinuation {
                workspace_path,
                embedding_model_id: embedding_model_id.map(str::to_string),
            },
        )
    }

    #[tokio::test]
    async fn workspace_vector_build_continuation_builds_once_when_diagnostic_is_safe()
    -> anyhow::Result<()> {
        let fixture = tempfile::tempdir()?;
        let workspace = std::fs::canonicalize(fixture.path())?;
        let setup = Arc::new(
            SemSearchFixture::new(sem_search_config(Some("fixture-model")))
                .with_cwd(workspace.clone())
                .with_readiness(SemSearchAvailability::Unsupported {
                    reason: SemSearchUnsupportedReason::VectorArtifactAbsentOrNoMatch,
                })
                .with_post_build_readiness(SemSearchAvailability::Ready {
                    workspace_root: workspace.clone(),
                    manifest_hash: "fixture-manifest".to_string(),
                    vector_artifact_id: "fixture-vector-artifact".to_string(),
                    dimension: 2,
                }),
        );
        let executor = ToolExecutor::new(Arc::clone(&setup));

        let actual = executor
            .call_internal(
                workspace_vector_build_tool(workspace.clone(), None),
                &tool_context(),
            )
            .await?;

        let actual = match actual {
            ToolOperation::WorkspaceVectorIndexBuildContinuation { output } => output,
            _ => panic!("expected workspace vector build continuation output"),
        };
        let expected = WorkspaceVectorIndexBuildContinuationStatus::BuiltReady;
        assert_eq!(actual.final_status, expected);
        assert_eq!(
            actual.preflight_diagnostic.status,
            SemSearchDiagnosticStatus::VectorBuildSuggested
        );
        assert_eq!(
            actual.post_build_diagnostic.status,
            SemSearchDiagnosticStatus::Ready
        );
        assert!(actual.build_report.is_some());
        assert_eq!(setup.workspace.build_calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn workspace_vector_build_continuation_classifies_non_build_safe_without_mutation()
    -> anyhow::Result<()> {
        let fixture = tempfile::tempdir()?;
        let workspace = std::fs::canonicalize(fixture.path())?;
        let setup = Arc::new(
            SemSearchFixture::new(sem_search_config(Some("fixture-model")))
                .with_cwd(workspace.clone())
                .with_readiness(SemSearchAvailability::Unknown {
                    reason: SemSearchUnknownReason::VectorArtifactCorruptOrNotReady,
                }),
        );
        let executor = ToolExecutor::new(Arc::clone(&setup));

        let actual = executor
            .call_internal(
                workspace_vector_build_tool(workspace, None),
                &tool_context(),
            )
            .await?;

        let actual = match actual {
            ToolOperation::WorkspaceVectorIndexBuildContinuation { output } => output,
            _ => panic!("expected workspace vector build continuation output"),
        };
        let expected = WorkspaceVectorIndexBuildContinuationStatus::NotBuiltRepairRequired;
        assert_eq!(actual.final_status, expected);
        assert_eq!(actual.build_report, None);
        assert_eq!(setup.workspace.build_calls.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[tokio::test]
    async fn workspace_vector_build_continuation_does_not_report_built_when_preflight_already_ready()
    -> anyhow::Result<()> {
        let fixture = tempfile::tempdir()?;
        let workspace = std::fs::canonicalize(fixture.path())?;
        let setup = Arc::new(
            SemSearchFixture::new(sem_search_config(Some("fixture-model")))
                .with_cwd(workspace.clone())
                .with_readiness(SemSearchAvailability::Ready {
                    workspace_root: workspace.clone(),
                    manifest_hash: "fixture-manifest".to_string(),
                    vector_artifact_id: "fixture-vector-artifact".to_string(),
                    dimension: 2,
                }),
        );
        let executor = ToolExecutor::new(Arc::clone(&setup));

        let actual = executor
            .call_internal(
                workspace_vector_build_tool(workspace, None),
                &tool_context(),
            )
            .await?;

        let actual = match actual {
            ToolOperation::WorkspaceVectorIndexBuildContinuation { output } => output,
            _ => panic!("expected workspace vector build continuation output"),
        };
        assert!(
            actual.final_status != WorkspaceVectorIndexBuildContinuationStatus::BuiltReady,
            "BuiltReady means a build was performed; preflight-ready continuation must not claim it built"
        );
        assert_eq!(actual.build_report, None);
        assert_eq!(setup.workspace.build_calls.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[tokio::test]
    async fn workspace_vector_build_continuation_rejects_symlink_escape_workspace_identity()
    -> anyhow::Result<()> {
        let fixture = tempfile::tempdir()?;
        let workspace = fixture.path().join("workspace");
        let outside = fixture.path().join("outside");
        let alias = workspace.join("alias");
        std::fs::create_dir_all(&workspace)?;
        std::fs::create_dir_all(&outside)?;
        create_directory_symlink(&outside, &alias)?;
        if !alias.exists() {
            return Ok(());
        }
        let workspace = std::fs::canonicalize(workspace)?;
        let setup = Arc::new(
            SemSearchFixture::new(sem_search_config(Some("fixture-model"))).with_cwd(workspace),
        );
        let executor = ToolExecutor::new(Arc::clone(&setup));

        let actual = executor
            .call_internal(workspace_vector_build_tool(alias, None), &tool_context())
            .await;

        assert!(
            actual
                .unwrap_err()
                .to_string()
                .contains("does not match current workspace root")
        );
        assert_eq!(setup.workspace.build_calls.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[tokio::test]
    async fn workspace_vector_build_continuation_rejects_unconfigured_explicit_model()
    -> anyhow::Result<()> {
        let fixture = tempfile::tempdir()?;
        let workspace = std::fs::canonicalize(fixture.path())?;
        let setup = Arc::new(
            SemSearchFixture::new(sem_search_config(Some("fixture-model")))
                .with_cwd(workspace.clone()),
        );
        let executor = ToolExecutor::new(Arc::clone(&setup));

        let actual = executor
            .call_internal(
                workspace_vector_build_tool(workspace, Some("other-model")),
                &tool_context(),
            )
            .await;

        assert!(
            actual
                .unwrap_err()
                .to_string()
                .contains("explicit model is not configured")
        );
        assert_eq!(setup.workspace.build_calls.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[tokio::test]
    async fn workspace_vector_build_continuation_rejects_blank_explicit_model() -> anyhow::Result<()>
    {
        let fixture = tempfile::tempdir()?;
        let workspace = std::fs::canonicalize(fixture.path())?;
        let setup = Arc::new(
            SemSearchFixture::new(sem_search_config(Some("fixture-model")))
                .with_cwd(workspace.clone()),
        );
        let executor = ToolExecutor::new(Arc::clone(&setup));

        let actual = executor
            .call_internal(
                workspace_vector_build_tool(workspace, Some("   ")),
                &tool_context(),
            )
            .await;

        assert!(
            actual
                .unwrap_err()
                .to_string()
                .contains("explicit model is not configured")
        );
        assert_eq!(setup.workspace.build_calls.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[tokio::test]
    async fn workspace_vector_build_continuation_rejects_blank_explicit_model_without_config()
    -> anyhow::Result<()> {
        let fixture = tempfile::tempdir()?;
        let workspace = std::fs::canonicalize(fixture.path())?;
        let setup =
            Arc::new(SemSearchFixture::new(sem_search_config(None)).with_cwd(workspace.clone()));
        let executor = ToolExecutor::new(Arc::clone(&setup));

        let actual = executor
            .call_internal(
                workspace_vector_build_tool(workspace, Some("   ")),
                &tool_context(),
            )
            .await;

        assert!(
            actual
                .unwrap_err()
                .to_string()
                .contains("explicit model is not configured")
        );
        assert_eq!(setup.workspace.build_calls.load(Ordering::SeqCst), 0);
        Ok(())
    }

    fn create_directory_symlink(physical: &PathBuf, alias: &PathBuf) -> anyhow::Result<()> {
        #[cfg(unix)]
        std::os::unix::fs::symlink(physical, alias)?;
        #[cfg(windows)]
        if let Err(error) = std::os::windows::fs::symlink_dir(physical, alias) {
            if error.kind() == std::io::ErrorKind::PermissionDenied {
                return Ok(());
            }
            return Err(error.into());
        }
        Ok(())
    }

    fn symlink_fixture() -> anyhow::Result<Option<(tempfile::TempDir, PathBuf, PathBuf)>> {
        let fixture = tempfile::tempdir()?;
        let workspace = fixture.path().join("workspace");
        let physical = fixture.path().join("physical");
        let alias = workspace.join("alias");
        std::fs::create_dir_all(&workspace)?;
        std::fs::create_dir_all(&physical)?;
        create_directory_symlink(&physical, &alias)?;
        if !alias.exists() {
            return Ok(None);
        }
        let physical = std::fs::canonicalize(&physical)?;
        Ok(Some((fixture, workspace, physical)))
    }

    #[test]
    fn test_shell_execution_cwd_matches_policy_physical_symlink_resolution() -> anyhow::Result<()> {
        let Some((_fixture, workspace, physical)) = symlink_fixture()? else {
            return Ok(());
        };
        let cwd = PathBuf::from("alias");
        let tool = ToolCatalog::Shell(Shell {
            command: "pwd".to_string(),
            cwd: Some(cwd.clone()),
            ..Default::default()
        });

        let actual = resolve_tool_execution_cwd(Some(&cwd), workspace.as_path());
        let expected = match tool.to_policy_operation(workspace).unwrap() {
            PermissionOperation::Execute { cwd, .. } => cwd,
            _ => unreachable!("shell policy operation must be execute"),
        };
        assert_eq!(actual, physical);
        assert_eq!(actual, expected);
        Ok(())
    }
}
