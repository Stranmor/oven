use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use chrono::Local;
use forge_config::ForgeConfig;
use forge_domain::*;
use forge_project_model::{
    LearningContextPayload, LearningContextRecord,
    LearningLedgerFreshness as ProjectLearningLedgerFreshness,
    LearningProvenance as ProjectLearningProvenance,
    LearningRedactionStatus as ProjectLearningRedactionStatus,
    LearningReviewState as ProjectLearningReviewState,
    LearningSourceKind as ProjectLearningSourceKind, ProjectContextTarget,
    ProjectModelContextRenderBudget, ProjectModelSourceNode, TargetResolutionBudget,
    directory_path_filter, local_project_model_manifest, mentioned_paths,
    render_project_model_context, render_sources_from_nodes,
};
use forge_stream::MpscStream;
use url::Url;

use crate::apply_tunable_parameters::ApplyTunableParameters;
use crate::changed_files::ChangedFiles;
use crate::dto::ToolsOverview;
use crate::dto::openai::ProviderRequestEstimate as OpenAiProviderRequestEstimate;
use crate::hooks::{
    CompactionHandler, DoomLoopDetector, LearningCapture, PendingTodosHandler,
    TitleGenerationHandler, TracingHandler,
};
use crate::init_conversation_metrics::InitConversationMetrics;
use crate::orch::Orchestrator;
use crate::services::{
    AgentRegistry, CustomInstructionsService, ProviderAuthService, SteerService,
};
use crate::set_conversation_id::SetConversationId;
use crate::steer::SteerHandle;
use crate::system_prompt::SystemPrompt;
use crate::tool_registry::ToolRegistry;
use crate::tool_resolver::ToolResolver;
use crate::user_prompt::UserPromptGenerator;
use crate::{
    AgentExt, AgentProviderResolver, ConversationService, EnvironmentInfra, LearningService,
    ProviderService, Services, WorkspaceService,
};

/// Builds a [`TemplateConfig`] from a [`ForgeConfig`].
///
/// Converts the configuration-layer field names into the domain-layer struct
/// expected by [`SystemContext`] for tool description template rendering.
pub(crate) fn build_template_config(config: &ForgeConfig) -> forge_domain::TemplateConfig {
    forge_domain::TemplateConfig {
        max_read_size: config.max_read_lines.try_into().unwrap_or(usize::MAX),
        max_line_length: config.max_line_chars,
        max_image_size: config.max_image_size_bytes.try_into().unwrap_or(usize::MAX),
        stdout_max_prefix_length: config.max_stdout_prefix_lines,
        stdout_max_suffix_length: config.max_stdout_suffix_lines,
        stdout_max_line_length: config.max_stdout_line_chars,
    }
}

struct ProjectContextInjection<S> {
    services: Arc<S>,
    agent: Agent,
}

impl<S: EnvironmentInfra<Config = forge_config::ForgeConfig> + WorkspaceService>
    ProjectContextInjection<S>
{
    const MAX_TARGETS: usize = 4;
    const MAX_EXPLICIT_TARGET_CANDIDATES: usize = 8;
    const MAX_INDEX_PROBES: usize = 32;
    const MAX_LEARNING_RECORDS: usize = 8;
    const MAX_LEARNING_CONTEXT_CHARS: usize = 8_192;

    fn new(services: Arc<S>, agent: Agent) -> Self {
        Self { services, agent }
    }

    async fn inject_learning(&self, mut conversation: Conversation) -> Conversation
    where
        S: LearningService,
    {
        let records = match self
            .services
            .list_learning_records(
                Some(LearningReviewState::Accepted),
                Self::MAX_LEARNING_RECORDS,
            )
            .await
        {
            Ok(records) => records,
            Err(error) => {
                tracing::debug!(error = ?error, "Skipping learning context injection because reviewed query failed");
                return conversation;
            }
        };
        if records.is_empty() {
            return conversation;
        }
        let freshness = match self
            .services
            .learning_freshness(Some(LearningReviewState::Accepted))
            .await
        {
            Ok(freshness) => freshness,
            Err(error) => {
                tracing::debug!(error = ?error, "Skipping learning context injection because freshness query failed");
                return conversation;
            }
        };
        let payload = LearningContextPayload::new(
            Self::learning_freshness_to_project(freshness),
            records
                .into_iter()
                .filter_map(Self::learning_record_to_project)
                .collect(),
        );
        if payload.records.is_empty() {
            return conversation;
        }
        let content = match payload.render() {
            Ok(content) => content,
            Err(error) => {
                tracing::debug!(error = ?error, "Skipping learning context injection because payload violated reviewed-only transport invariants");
                return conversation;
            }
        };
        if content.chars().count() > Self::MAX_LEARNING_CONTEXT_CHARS {
            tracing::debug!(
                actual_chars = content.chars().count(),
                max_chars = Self::MAX_LEARNING_CONTEXT_CHARS,
                "Skipping learning context injection because rendered payload exceeds bounded budget"
            );
            return conversation;
        }
        let mut context = conversation.context.take().unwrap_or_default();
        let message = TextMessage::learning_context(Role::User, content)
            .model(self.agent.model.clone())
            .droppable(true)
            .cacheable(false);
        context = context.add_message(ContextMessage::Text(message));
        conversation.context(context)
    }

    fn learning_freshness_to_project(
        freshness: LearningLedgerFreshness,
    ) -> ProjectLearningLedgerFreshness {
        ProjectLearningLedgerFreshness {
            ledger_cursor: freshness.ledger_cursor,
            projection_version: freshness.projection_version,
            review_state_fingerprint: freshness.review_state_fingerprint,
        }
    }

    fn learning_record_to_project(
        projection: LearningRecordProjection,
    ) -> Option<LearningContextRecord> {
        if projection.review_state != LearningReviewState::Accepted {
            return None;
        }
        Some(LearningContextRecord {
            id: projection.record_id.into_string(),
            summary: projection.summary,
            review_state: ProjectLearningReviewState::Accepted,
            redaction_status: Self::learning_redaction_to_project(projection.redaction_status),
            provenance: Self::learning_provenance_to_project(projection.provenance)?,
        })
    }

    fn learning_redaction_to_project(
        status: LearningRedactionStatus,
    ) -> ProjectLearningRedactionStatus {
        match status {
            LearningRedactionStatus::Clean => ProjectLearningRedactionStatus::Clean,
            LearningRedactionStatus::Redacted => ProjectLearningRedactionStatus::Redacted,
        }
    }

    fn learning_provenance_to_project(
        provenance: LearningProvenance,
    ) -> Option<ProjectLearningProvenance> {
        let source_kind = match provenance.source_kind {
            LearningSourceKind::Conversation => ProjectLearningSourceKind::Conversation,
            LearningSourceKind::Task => ProjectLearningSourceKind::Task,
            LearningSourceKind::Tool => ProjectLearningSourceKind::Tool,
            LearningSourceKind::Eval => ProjectLearningSourceKind::Eval,
        };
        let source_id = provenance.source_id().ok()?;
        Some(ProjectLearningProvenance {
            source_kind,
            source_id,
            source_event_id: Some(provenance.source_event_id),
            source_timestamp: None,
            source_fingerprint: provenance.source_fingerprint,
        })
    }

    async fn inject(&self, mut conversation: Conversation) -> Conversation {
        let environment = self.services.get_environment();
        let Some(query) = Self::query_from_conversation(&conversation) else {
            return conversation;
        };
        let targets = self.resolve_targets(&environment, &query).await;
        if targets.is_empty() {
            return conversation;
        }

        let max_sources = ProjectModelContextRenderBudget::default().max_sources;
        let mut rendered_contexts = Vec::new();
        for target in targets {
            let mut params = SearchParams::new(&query, "automatic project-model context injection")
                .limit(max_sources);
            if let Some(path_filter) = target.path_filter.clone() {
                params = params.starts_with(path_filter);
            }
            let nodes = match self
                .services
                .query_workspace(target.workspace_root.clone(), params)
                .await
            {
                Ok(nodes) => nodes,
                Err(error) => {
                    tracing::debug!(error = ?error, path = %target.workspace_root.display(), "Skipping project-model context target because local retrieval failed");
                    continue;
                }
            };
            if nodes.is_empty() {
                continue;
            }
            rendered_contexts.push(Self::render_context(&target.workspace_root, nodes));
        }
        if rendered_contexts.is_empty() {
            return conversation;
        }

        let mut context = conversation.context.take().unwrap_or_default();
        for content in rendered_contexts {
            let message = TextMessage::project_model_context(Role::User, content)
                .model(self.agent.model.clone())
                .droppable(true)
                .cacheable(false);
            context = context.add_message(ContextMessage::Text(message));
        }
        conversation.context(context)
    }

    async fn explain(&self, query: Option<String>) -> WorkspaceContextExplanation {
        let environment = self.services.get_environment();
        let mut candidates = vec![environment.cwd.clone()];
        if let Some(query) = query.as_deref() {
            candidates.extend(mentioned_paths(
                query,
                &environment.cwd,
                environment.home.as_deref(),
            ));
        }

        let mut budget = TargetResolutionBudget::new(
            Self::MAX_EXPLICIT_TARGET_CANDIDATES,
            Self::MAX_INDEX_PROBES,
        );
        let mut candidate_diagnostics = Vec::new();
        let mut selected_targets = Vec::new();
        let mut target_specs = Vec::new();
        let mut seen = BTreeSet::new();
        for candidate in candidates {
            if !budget.claim_candidate() {
                candidate_diagnostics.push(WorkspaceContextCandidateDiagnostic {
                    candidate_path: candidate,
                    selected_workspace: None,
                    path_filter: None,
                    skip_reason: Some("candidate limit reached".to_string()),
                });
                break;
            }
            let (candidate_diagnostic, manifest_diagnostic, target) =
                self.resolve_target_diagnostic(candidate, &mut budget).await;
            if let (Some(manifest_diagnostic), Some(target)) = (manifest_diagnostic, target) {
                if seen.insert(target.clone()) {
                    selected_targets.push(manifest_diagnostic);
                    target_specs.push(target);
                }
            }
            candidate_diagnostics.push(candidate_diagnostic);
            if target_specs.len() >= Self::MAX_TARGETS {
                break;
            }
        }

        let mut retrieval_empty_targets = Vec::new();
        let mut would_inject = false;
        if let Some(query) = query.as_deref() {
            let max_sources = ProjectModelContextRenderBudget::default().max_sources;
            for target in &target_specs {
                let mut params =
                    SearchParams::new(query, "automatic project-model context injection")
                        .limit(max_sources);
                if let Some(path_filter) = target.path_filter.clone() {
                    params = params.starts_with(path_filter);
                }
                match self
                    .services
                    .query_workspace(target.workspace_root.clone(), params)
                    .await
                {
                    Ok(nodes) if nodes.is_empty() => {
                        retrieval_empty_targets.push(target.workspace_root.clone());
                    }
                    Ok(_) => {
                        would_inject = true;
                    }
                    Err(error) => {
                        retrieval_empty_targets.push(target.workspace_root.clone());
                        tracing::debug!(error = ?error, path = %target.workspace_root.display(), "Explain-context retrieval failed for selected target");
                    }
                }
            }
        }

        let skip_reason = if would_inject {
            None
        } else if query.as_deref().is_none_or(str::is_empty) {
            Some("query not provided; automatic injection needs a latest user message".to_string())
        } else if selected_targets.is_empty() {
            Some("no fresh project-model manifest target selected".to_string())
        } else {
            Some("retrieval returned no usable project-model context".to_string())
        };

        WorkspaceContextExplanation {
            cwd: environment.cwd,
            query,
            candidates: candidate_diagnostics,
            selected_targets,
            retrieval_empty_targets,
            would_inject,
            skip_reason,
        }
    }

    async fn resolve_target_diagnostic(
        &self,
        path: PathBuf,
        budget: &mut TargetResolutionBudget,
    ) -> (
        WorkspaceContextCandidateDiagnostic,
        Option<WorkspaceContextManifestDiagnostic>,
        Option<ProjectContextTarget>,
    ) {
        let candidate_path = path.clone();
        for ancestor in path.ancestors() {
            if !budget.claim_index_probe() {
                return (
                    WorkspaceContextCandidateDiagnostic {
                        candidate_path: candidate_path.clone(),
                        selected_workspace: None,
                        path_filter: None,
                        skip_reason: Some("index freshness probe limit reached".to_string()),
                    },
                    None,
                    None,
                );
            }
            let diagnostic = match self
                .services
                .project_model_context_diagnostic(ancestor)
                .await
            {
                Ok(diagnostic) => diagnostic,
                Err(error) => {
                    return (
                        WorkspaceContextCandidateDiagnostic {
                            candidate_path: candidate_path.clone(),
                            selected_workspace: None,
                            path_filter: None,
                            skip_reason: Some(format!(
                                "freshness check failed for {}: {}",
                                ancestor.display(),
                                error
                            )),
                        },
                        None,
                        None,
                    );
                }
            };
            if !diagnostic.can_inject() {
                if diagnostic.manifest_found {
                    return (
                        WorkspaceContextCandidateDiagnostic {
                            candidate_path: path,
                            selected_workspace: None,
                            path_filter: None,
                            skip_reason: Some(Self::manifest_skip_reason(&diagnostic)),
                        },
                        None,
                        None,
                    );
                }
                continue;
            }
            let workspace_root = ancestor.to_path_buf();
            let path_filter = directory_path_filter(&path, &workspace_root);
            let target = ProjectContextTarget::new(workspace_root.clone(), path_filter.clone());
            return (
                WorkspaceContextCandidateDiagnostic {
                    candidate_path: candidate_path.clone(),
                    selected_workspace: Some(workspace_root),
                    path_filter,
                    skip_reason: None,
                },
                Some(diagnostic),
                Some(target),
            );
        }
        (
            WorkspaceContextCandidateDiagnostic {
                candidate_path: candidate_path.clone(),
                selected_workspace: None,
                path_filter: None,
                skip_reason: Some(
                    "no fresh project-model manifest found in candidate ancestors".to_string(),
                ),
            },
            None,
            None,
        )
    }

    fn manifest_skip_reason(diagnostic: &WorkspaceContextManifestDiagnostic) -> String {
        match &diagnostic.freshness {
            WorkspaceContextFreshness::Fresh => "manifest is fresh".to_string(),
            WorkspaceContextFreshness::Unknown { reason } => format!(
                "project-model manifest freshness unknown at {}: {}",
                diagnostic.manifest_path.display(),
                reason
            ),
            WorkspaceContextFreshness::Stale { changed, deleted, added } => format!(
                "project-model manifest stale at {}: changed=[{}] deleted=[{}] added=[{}]",
                diagnostic.manifest_path.display(),
                changed.join(","),
                deleted.join(","),
                added.join(",")
            ),
        }
    }

    async fn resolve_targets(
        &self,
        environment: &Environment,
        latest_user_message: &str,
    ) -> Vec<ProjectContextTarget> {
        let mut candidates = vec![environment.cwd.clone()];
        candidates.extend(mentioned_paths(
            latest_user_message,
            &environment.cwd,
            environment.home.as_deref(),
        ));

        let mut budget = TargetResolutionBudget::new(
            Self::MAX_EXPLICIT_TARGET_CANDIDATES,
            Self::MAX_INDEX_PROBES,
        );
        let mut targets = Vec::new();
        let mut seen = BTreeSet::new();
        for candidate in candidates {
            if !budget.claim_candidate() {
                break;
            }
            let Some(target) = self.resolve_target(candidate, &mut budget).await else {
                continue;
            };
            if seen.insert(target.clone()) {
                targets.push(target);
            }
            if targets.len() >= Self::MAX_TARGETS {
                break;
            }
        }
        targets
    }

    async fn resolve_target(
        &self,
        path: PathBuf,
        budget: &mut TargetResolutionBudget,
    ) -> Option<ProjectContextTarget> {
        for ancestor in path.ancestors() {
            if !budget.claim_index_probe() {
                return None;
            }
            let diagnostic = match self
                .services
                .project_model_context_diagnostic(ancestor)
                .await
            {
                Ok(diagnostic) => diagnostic,
                Err(error) => {
                    tracing::debug!(error = ?error, path = %ancestor.display(), "Skipping project-model context target because index freshness could not be checked");
                    continue;
                }
            };
            if !diagnostic.can_inject() {
                if diagnostic.manifest_found {
                    tracing::debug!(path = %ancestor.display(), freshness = diagnostic.freshness.label(), "Stopping project-model context target resolution because nearest manifest is not injectable");
                    return None;
                }
                tracing::debug!(path = %ancestor.display(), freshness = diagnostic.freshness.label(), "Skipping project-model context target because manifest is unavailable");
                continue;
            }
            let workspace_root = ancestor.to_path_buf();
            let path_filter = directory_path_filter(&path, &workspace_root);
            return Some(ProjectContextTarget::new(workspace_root, path_filter));
        }
        None
    }

    fn query_from_conversation(conversation: &Conversation) -> Option<String> {
        Self::latest_real_user_message(conversation).map(ToOwned::to_owned)
    }

    fn latest_real_user_message(conversation: &Conversation) -> Option<&str> {
        conversation
            .context
            .as_ref()?
            .messages
            .iter()
            .rev()
            .find(|message| {
                message.has_role(Role::User)
                    && !message.is_droppable()
                    && !matches!(
                        &message.message,
                        ContextMessage::Text(text) if text.is_internal_context()
                    )
            })
            .and_then(|message| message.content())
            .map(str::trim)
            .filter(|content| !content.is_empty())
    }

    fn render_context(workspace_root: &std::path::Path, nodes: Vec<Node>) -> String {
        let manifest_path = local_project_model_manifest(workspace_root);
        let source_nodes = nodes
            .into_iter()
            .map(Self::source_node_from_node)
            .collect::<Vec<_>>();
        let sources = render_sources_from_nodes(source_nodes);
        render_project_model_context(
            &workspace_root.display().to_string(),
            &manifest_path.display().to_string(),
            "local_manifest_available",
            "WorkspaceService::query_workspace",
            &sources,
            &ProjectModelContextRenderBudget::default(),
        )
    }

    fn source_node_from_node(node: Node) -> ProjectModelSourceNode {
        let node_id = node.node_id.as_str().to_string();
        let score = node.relevance;
        match node.node {
            NodeData::FileChunk(chunk) => ProjectModelSourceNode::FileChunk {
                path: chunk.file_path,
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                node_id,
                score,
                content: chunk.content,
            },
            NodeData::File(file) => ProjectModelSourceNode::File {
                path: file.file_path,
                node_id,
                score,
                content_hash: file.hash,
                content: Some(file.content),
            },
            NodeData::FileRef(file_ref) => ProjectModelSourceNode::FileRef {
                path: file_ref.file_path,
                node_id,
                score,
                content_hash: file_ref.file_hash,
            },
            NodeData::Note(note) => {
                ProjectModelSourceNode::Note { node_id, score, content: note.content }
            }
            NodeData::Task(task) => {
                ProjectModelSourceNode::Task { node_id, score, content: task.task }
            }
        }
    }
}

fn provider_request_compaction_estimate(
    estimate: OpenAiProviderRequestEstimate,
    input_budget: Option<usize>,
) -> ProviderRequestEstimate {
    ProviderRequestEstimate::new(estimate.estimated_input_tokens, input_budget)
}

/// ForgeApp handles the core chat functionality by orchestrating various
/// services. It encapsulates the complex logic previously contained in the
/// ForgeAPI chat method.
pub struct ForgeApp<S> {
    services: Arc<S>,
    tool_registry: ToolRegistry<S>,
}

impl<S: Services + EnvironmentInfra<Config = forge_config::ForgeConfig> + SteerService>
    ForgeApp<S>
{
    /// Creates a new ForgeApp instance with the provided services.
    pub fn new(services: Arc<S>) -> Self {
        Self { tool_registry: ToolRegistry::new(services.clone()), services }
    }

    /// Explains whether automatic project-model context would be injected for
    /// the current environment and optional query.
    pub async fn explain_workspace_context(
        &self,
        query: Option<String>,
    ) -> WorkspaceContextExplanation {
        let agent = Agent::new(
            AgentId::new("forge"),
            ProviderId::from("diagnostic-provider".to_string()),
            ModelId::new("diagnostic-model"),
        );
        ProjectContextInjection::new(self.services.clone(), agent)
            .explain(query)
            .await
    }

    /// Accepts a typed steer message for delayed primary-conversation delivery.
    ///
    /// # Arguments
    /// * `request` - The typed steer request to validate and queue.
    ///
    /// # Errors
    /// Returns an error when the conversation is missing or is not primary.
    pub async fn steer(&self, request: SteerRequest) -> anyhow::Result<()> {
        SteerHandle::<S>::new(self.services.clone())
            .accept(request)
            .await
    }

    /// Executes a chat request and returns a stream of responses.
    /// This method contains the core chat logic extracted from ForgeAPI.
    pub async fn chat(
        &self,
        agent_id: AgentId,
        chat: ChatRequest,
    ) -> Result<MpscStream<Result<ChatResponse, anyhow::Error>>> {
        let services = self.services.clone();

        // Get the conversation for the chat request
        let conversation = services
            .find_conversation(&chat.conversation_id)
            .await?
            .ok_or_else(|| forge_domain::Error::ConversationNotFound(chat.conversation_id))?;

        // Discover files using the discovery service
        let app_config = self.services.get_config()?;
        let environment = services.get_environment();

        let custom_instructions = services.get_custom_instructions().await;

        // Prepare agents with user configuration
        let agent_provider_resolver = AgentProviderResolver::new(services.clone());

        // Get agent and apply workflow config
        let agent = self
            .services
            .get_agent(&agent_id)
            .await?
            .ok_or(crate::Error::AgentNotFound(agent_id.clone()))?
            .apply_config(&app_config)
            .set_compact_model_if_none();

        let agent_provider = agent_provider_resolver
            .get_provider(Some(agent.id.clone()))
            .await?;
        let agent_provider = self
            .services
            .provider_auth_service()
            .refresh_provider_credential(agent_provider)
            .await?;

        let models = services.models(agent_provider.clone()).await?;
        let selected_model = models
            .iter()
            .find(|model| model.id == agent.model && model.provider_id == agent.provider)
            .ok_or_else(|| forge_domain::Error::MissingModel(agent.id.clone()))?;
        let agent = agent.compaction_threshold(Some(selected_model));

        // Get system and mcp tool definitions and resolve them for the agent
        let all_tool_definitions = self
            .tool_registry
            .list(&agent.id, selected_model, &agent_provider)
            .await?;
        let tool_resolver = ToolResolver::new(all_tool_definitions);
        let tool_definitions: Vec<ToolDefinition> =
            tool_resolver.resolve(&agent).into_iter().cloned().collect();
        let max_tool_failure_per_turn = agent.max_tool_failure_per_turn.unwrap_or(3);

        let current_time = Local::now();

        // Insert system prompt
        let conversation =
            SystemPrompt::new(self.services.clone(), environment.clone(), agent.clone())
                .custom_instructions(custom_instructions.clone())
                .tool_definitions(tool_definitions.clone())
                .models(models.clone())
                .max_extensions(app_config.max_extensions)
                .template_config(build_template_config(&app_config))
                .add_system_message(conversation)
                .await?;

        // Insert user prompt
        let conversation = UserPromptGenerator::new(
            self.services.clone(),
            agent.clone(),
            chat.event.clone(),
            current_time,
        )
        .add_user_prompt(conversation)
        .await?;

        // Inject local project-model context after the user prompt, before the
        // provider sees the request. This is a best-effort, manifest-gated read
        // path and never triggers hot-path indexing.
        let conversation = ProjectContextInjection::new(self.services.clone(), agent.clone())
            .inject(conversation)
            .await;

        // Inject reviewed learning records as late-bound internal context. This
        // path is reviewed-only, bounded, droppable, and cache-ineligible.
        let conversation = ProjectContextInjection::new(self.services.clone(), agent.clone())
            .inject_learning(conversation)
            .await;

        // Detect and render externally changed files notification
        let conversation = ChangedFiles::new(services.clone(), agent.clone())
            .update_file_stats(conversation)
            .await;

        let conversation = InitConversationMetrics::new(current_time).apply(conversation);
        let conversation = ApplyTunableParameters::new(agent.clone(), tool_definitions.clone())
            .apply(conversation);
        let conversation = SetConversationId.apply(conversation);

        // Create the orchestrator with all necessary dependencies
        let tracing_handler = TracingHandler::new();
        let title_handler = TitleGenerationHandler::new(services.clone());

        // Build the on_end hook, conditionally adding PendingTodosHandler based on
        // config
        let on_end_hook = if app_config.verify_todos {
            tracing_handler
                .clone()
                .and(title_handler.clone())
                .and(PendingTodosHandler::new())
        } else {
            tracing_handler.clone().and(title_handler.clone())
        };

        let hook = Hook::default()
            .on_start(tracing_handler.clone().and(title_handler))
            .on_request(tracing_handler.clone().and(DoomLoopDetector::default()))
            .on_response(
                tracing_handler
                    .clone()
                    .and(CompactionHandler::new(agent.clone(), environment.clone())),
            )
            .on_toolcall_start(tracing_handler.clone())
            .on_toolcall_end(tracing_handler)
            .on_end(on_end_hook);

        let orch = Orchestrator::new(
            services.clone(),
            conversation,
            agent,
            self.services.get_config()?,
        )
        .error_tracker(ToolErrorTracker::new(max_tool_failure_per_turn))
        .tool_definitions(tool_definitions)
        .models(models)
        .active_provider(agent_provider)
        .hook(Arc::new(hook));

        // Create and return the stream
        let stream = MpscStream::spawn(
            |tx: tokio::sync::mpsc::Sender<Result<ChatResponse, anyhow::Error>>| {
                async move {
                    // Execute dispatch and always save conversation afterwards
                    let mut orch = orch.sender(tx.clone());
                    let dispatch_result = orch.run().await;

                    // Always save conversation using get_conversation()
                    let conversation = orch.get_conversation().clone();
                    let save_result = save_conversation_and_capture_learning(
                        services.clone(),
                        conversation.clone(),
                    )
                    .await;

                    // Send any error to the stream (prioritize dispatch error over save error)
                    let final_err = match (dispatch_result, save_result) {
                        (Err(d), Err(s)) => {
                            Some(d.context(format!("Also failed to save conversation: {}", s)))
                        }
                        (Ok(_), Err(s)) => Some(s.context("Failed to save conversation")),
                        (Err(d), Ok(_)) => Some(d),
                        (Ok(_), Ok(_)) => None,
                    };

                    if let Some(err) = final_err {
                        if let Err(e) = tx.send(Err(err)).await {
                            tracing::error!("Failed to send error to stream: {}", e);
                        }
                    }
                }
            },
        );

        Ok(stream)
    }

    fn estimate_compaction_provider_request(
        &self,
        context: Context,
        agent: &Agent,
        models: Vec<Model>,
        provider: &Provider<Url>,
        merge_system_messages: bool,
    ) -> Result<ProviderRequestEstimate> {
        let conversation = Conversation::generate().context(context);
        let orchestrator = Orchestrator::new(
            self.services.clone(),
            conversation,
            agent.clone(),
            forge_config::ForgeConfig { merge_system_messages, ..Default::default() },
        )
        .models(models);
        let context = orchestrator
            .get_conversation()
            .context
            .clone()
            .unwrap_or_default();
        let (estimate, input_budget) =
            orchestrator.estimate_final_provider_request_for_provider(context, provider)?;
        Ok(provider_request_compaction_estimate(estimate, input_budget))
    }

    async fn compaction_provider_request_estimates(
        &self,
        original_context: Context,
        compacted_context: Context,
        agent: &Agent,
        merge_system_messages: bool,
    ) -> Result<(ProviderRequestEstimate, ProviderRequestEstimate)> {
        let agent_provider_resolver = AgentProviderResolver::new(self.services.clone());
        let agent_provider = agent_provider_resolver
            .get_provider(Some(agent.id.clone()))
            .await?;
        let agent_provider = self
            .services
            .provider_auth_service()
            .refresh_provider_credential(agent_provider)
            .await?;
        let models = self.services.models(agent_provider.clone()).await?;
        models
            .iter()
            .find(|model| model.id == agent.model && model.provider_id == agent.provider)
            .ok_or_else(|| forge_domain::Error::MissingModel(agent.id.clone()))?;

        let original_provider_request = self.estimate_compaction_provider_request(
            original_context,
            agent,
            models.clone(),
            &agent_provider,
            merge_system_messages,
        )?;
        let compacted_provider_request = self.estimate_compaction_provider_request(
            compacted_context,
            agent,
            models,
            &agent_provider,
            merge_system_messages,
        )?;
        Ok((original_provider_request, compacted_provider_request))
    }

    /// Compacts the context of the main agent for the given conversation and
    /// persists it. Returns metrics about the compaction (original vs.
    /// compacted tokens and messages).
    pub async fn compact_conversation(
        &self,
        active_agent_id: AgentId,
        conversation_id: &ConversationId,
    ) -> Result<CompactionResult> {
        use crate::compact::Compactor;

        // Get the conversation
        let mut conversation = self
            .services
            .find_conversation(conversation_id)
            .await?
            .ok_or_else(|| forge_domain::Error::ConversationNotFound(*conversation_id))?;

        // Get the context from the conversation
        let context = match conversation.context.take() {
            Some(context) => context,
            None => {
                // No context to compact, return zero metrics
                return Ok(CompactionResult::new(0, 0, 0, 0));
            }
        };

        // Calculate original metrics. User-facing prompt-size metrics deliberately
        // use the current message approximation instead of historical provider
        // usage preserved on compacted summary messages.
        let original_messages = context.messages.len();
        let original_token_count = context.token_count_approx();

        let forge_config = self.services.get_config()?;

        // Get agent and apply workflow config
        let agent = self.services.get_agent(&active_agent_id).await?;

        let Some(agent) = agent else {
            return Err(crate::Error::AgentNotFound(active_agent_id).into());
        };

        // Get compact config from the agent
        let agent = agent
            .apply_config(&forge_config)
            .set_compact_model_if_none();
        let compact = agent.compact.clone();

        // Apply compaction using the Compactor
        let environment = self.services.get_environment();
        let compacted_context =
            Compactor::new(compact, environment).compact(context.clone(), true)?;

        let compacted_messages = compacted_context.messages.len();
        let compacted_tokens = compacted_context.token_count_approx();
        let provider_request_estimates = match self
            .compaction_provider_request_estimates(
                context,
                compacted_context.clone(),
                &agent,
                forge_config.merge_system_messages,
            )
            .await
        {
            Ok(estimates) => Some(estimates),
            Err(error) => {
                tracing::warn!(
                    "Compaction provider request metrics unavailable; compacted context will still be saved: {error:#}"
                );
                None
            }
        };

        // Update the conversation with the compacted context
        conversation.context = Some(compacted_context);

        // Save the updated conversation
        self.services.upsert_conversation(conversation).await?;

        let result = CompactionResult::new(
            original_token_count,
            compacted_tokens,
            original_messages,
            compacted_messages,
        );

        if let Some((original_provider_request, compacted_provider_request)) =
            provider_request_estimates
        {
            Ok(result
                .provider_request_estimates(original_provider_request, compacted_provider_request))
        } else {
            Ok(result)
        }
    }

    pub async fn list_tools(&self) -> Result<ToolsOverview> {
        self.tool_registry.tools_overview().await
    }

    /// Gets available models for the default provider with automatic credential
    /// refresh.
    pub async fn get_models(&self) -> Result<Vec<Model>> {
        let agent_provider_resolver = AgentProviderResolver::new(self.services.clone());
        let provider = agent_provider_resolver.get_provider(None).await?;
        let provider = self
            .services
            .provider_auth_service()
            .refresh_provider_credential(provider)
            .await?;

        self.services.models(provider).await
    }

    /// Gets available models from all configured providers concurrently.
    ///
    /// Returns a list of `ProviderModels` for each configured provider that
    /// successfully returned models. If every configured provider fails (e.g.
    /// due to an invalid API key), the first error encountered is returned so
    /// the caller receives the real underlying cause rather than an empty list.
    pub async fn get_all_provider_models(&self) -> Result<Vec<ProviderModels>> {
        let all_providers = self.services.get_all_providers().await?;

        // Build one future per configured provider, preserving the error on failure.
        let futures: Vec<_> = all_providers
            .into_iter()
            .filter_map(|any_provider| any_provider.into_configured())
            .map(|provider| {
                let provider_id = provider.id.clone();
                let services = self.services.clone();
                async move {
                    let result: Result<ProviderModels> = async {
                        let refreshed = services
                            .provider_auth_service()
                            .refresh_provider_credential(provider)
                            .await?;
                        let models = services.models(refreshed).await?;
                        Ok(ProviderModels { provider_id, models })
                    }
                    .await;
                    result
                }
            })
            .collect();

        // Execute all provider fetches concurrently.
        let results = futures::future::join_all(futures).await;
        let mut successes = Vec::new();
        let mut first_error = None;
        for res in results {
            match res {
                Ok(models) => successes.push(models),
                Err(e) => {
                    tracing::warn!("Failed to fetch models from provider: {}", e);
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
            }
        }
        if successes.is_empty() {
            if let Some(err) = first_error {
                return Err(err);
            }
        }
        Ok(successes)
    }
}

async fn save_conversation_and_capture_learning<S>(
    services: Arc<S>,
    conversation: Conversation,
) -> anyhow::Result<()>
where
    S: ConversationService + LearningService + Send + Sync + 'static,
{
    services.upsert_conversation(conversation.clone()).await?;
    LearningCapture::new(services)
        .capture_saved_conversation(&conversation)
        .await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use anyhow::Result;
    use forge_domain::{
        Agent, AgentId, AnyProvider, AuthContextRequest, AuthContextResponse, AuthMethod,
        ChatCompletionMessage, ChatRequest, Content, Context, ContextMessage, Conversation,
        ConversationId, Environment, Event, FileChunk, FileStatus, FinishReason,
        LEARNING_LEDGER_SCHEMA_VERSION, LearningEventKind, LearningLedgerEvent,
        LearningLedgerFreshness, LearningProvenance, LearningRecordId, LearningRecordProjection,
        LearningRedactionStatus, LearningReviewState, McpConfig, McpServers, Model, ModelId, Node,
        NodeData, NodeId, PermissionOperation, Provider, ProviderId, ProviderType, ResultStream,
        Scope, SearchParams, SteerMessage, SyncProgress, ToolCallContext, ToolCallFull, ToolOutput,
        ToolResult, WorkspaceAuth, WorkspaceContextFreshness, WorkspaceContextManifestDiagnostic,
        WorkspaceId, WorkspaceInfo,
    };
    use futures::StreamExt;
    use pretty_assertions::assert_eq;
    use tempfile::TempDir;
    use tokio::sync::Mutex;

    use super::*;
    use crate::agent::AgentService;
    use crate::orch::Orchestrator;
    use crate::{
        AppConfigService, AttachmentService, AuthService, CommandLoaderService,
        CustomInstructionsService, FileDiscoveryService, FollowUpService, FsPatchService,
        FsReadService, FsRemoveService, FsSearchService, FsUndoService, FsWriteService,
        ImageReadService, McpConfigManager, McpService, NetFetchService, PlanCreateService,
        PolicyService, ProviderAuthService, ShellService, SkillFetchService, TemplateService, User,
        UserUsage, Walker,
    };

    #[derive(Clone)]
    struct ChatFlowLearningHarness {
        state: Arc<ChatFlowLearningState>,
    }

    struct ChatFlowLearningState {
        cwd: PathBuf,
        conversations: Mutex<HashMap<ConversationId, Conversation>>,
        learning_events: Mutex<Vec<LearningLedgerEvent>>,
        learning_records: Mutex<Vec<LearningRecordProjection>>,
        captured_provider_context: Mutex<Option<Context>>,
        agent: Agent,
        model: Model,
        provider: Provider<Url>,
    }

    impl ChatFlowLearningHarness {
        fn new(cwd: PathBuf) -> Arc<Self> {
            let model_id = ModelId::new("runtime-proof-model");
            let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone())
                .tool_supported(false)
                .tools(Vec::<forge_domain::ToolName>::new())
                .max_requests_per_turn(1usize);
            let model = Model::new(ProviderId::OPENAI, model_id).context_length(200_000_u64);
            let provider = Provider {
                id: ProviderId::OPENAI,
                provider_type: ProviderType::Llm,
                response: None,
                url: Url::parse("http://127.0.0.1/runtime-proof").unwrap(),
                models: None,
                auth_methods: Vec::new(),
                url_params: Vec::new(),
                credential: None,
                custom_headers: None,
            };
            Arc::new(Self {
                state: Arc::new(ChatFlowLearningState {
                    cwd,
                    conversations: Mutex::new(HashMap::new()),
                    learning_events: Mutex::new(Vec::new()),
                    learning_records: Mutex::new(Vec::new()),
                    captured_provider_context: Mutex::new(None),
                    agent,
                    model,
                    provider,
                }),
            })
        }

        async fn set_learning_records(&self, records: Vec<LearningRecordProjection>) {
            *self.state.learning_records.lock().await = records;
        }

        async fn upsert_conversation(&self, conversation: Conversation) -> Result<()> {
            self.state.upsert_conversation(conversation).await
        }
    }

    impl EnvironmentInfra for ChatFlowLearningHarness {
        type Config = ForgeConfig;

        fn get_env_var(&self, _key: &str) -> Option<String> {
            None
        }

        fn get_env_vars(&self) -> BTreeMap<String, String> {
            BTreeMap::new()
        }

        fn get_environment(&self) -> Environment {
            Environment {
                os: "test".to_string(),
                cwd: self.state.cwd.clone(),
                home: None,
                shell: "sh".to_string(),
                base_path: self.state.cwd.join(".forge"),
            }
        }

        fn get_config(&self) -> Result<Self::Config> {
            Ok(ForgeConfig { max_parallel_file_reads: 4, ..Default::default() })
        }

        async fn update_environment(&self, _ops: Vec<forge_domain::ConfigOperation>) -> Result<()> {
            anyhow::bail!("unused environment update")
        }
    }

    #[async_trait::async_trait]
    impl ConversationService for ChatFlowLearningState {
        async fn find_conversation(&self, id: &ConversationId) -> Result<Option<Conversation>> {
            Ok(self.conversations.lock().await.get(id).cloned())
        }

        async fn upsert_conversation(&self, conversation: Conversation) -> Result<()> {
            self.conversations
                .lock()
                .await
                .insert(conversation.id, conversation);
            Ok(())
        }

        async fn ensure_delegated_conversation(
            &self,
            id: &ConversationId,
            parent_id: Option<ConversationId>,
        ) -> Result<Conversation> {
            let mut conversations = self.conversations.lock().await;
            let conversation = conversations
                .get_mut(id)
                .ok_or_else(|| forge_domain::Error::ConversationNotFound(*id))?;
            conversation.ensure_delegated(parent_id);
            Ok(conversation.clone())
        }

        async fn resolve_root_conversation_id(
            &self,
            parent_id: Option<ConversationId>,
        ) -> Result<Option<ConversationId>> {
            Ok(parent_id)
        }

        async fn modify_conversation<F, T>(&self, id: &ConversationId, f: F) -> Result<T>
        where
            F: FnOnce(&mut Conversation) -> T + Send,
            T: Send,
        {
            let mut conversations = self.conversations.lock().await;
            let conversation = conversations
                .get_mut(id)
                .ok_or_else(|| forge_domain::Error::ConversationNotFound(*id))?;
            Ok(f(conversation))
        }

        async fn branch_conversation(
            &self,
            _conversation_id: &ConversationId,
            _target_id: forge_domain::MessageId,
        ) -> Result<Conversation> {
            anyhow::bail!("unused branch conversation")
        }

        async fn get_conversations(&self) -> Result<Vec<Conversation>> {
            Ok(self.conversations.lock().await.values().cloned().collect())
        }

        async fn get_conversations_including_agent(&self) -> Result<Vec<Conversation>> {
            self.get_conversations().await
        }

        async fn get_sub_conversations(
            &self,
            parent_id: &ConversationId,
        ) -> Result<Vec<Conversation>> {
            Ok(self
                .conversations
                .lock()
                .await
                .values()
                .filter(|conversation| conversation.parent_id == Some(*parent_id))
                .cloned()
                .collect())
        }

        async fn upsert_subagent_task_session(
            &self,
            _session: forge_domain::SubagentTaskSession,
        ) -> Result<()> {
            Ok(())
        }

        async fn get_subagent_task_session(
            &self,
            _task_id: &forge_domain::SubagentTaskId,
        ) -> Result<Option<forge_domain::SubagentTaskSession>> {
            Ok(None)
        }

        async fn get_subagent_task_session_by_conversation(
            &self,
            _conversation_id: &ConversationId,
        ) -> Result<Option<forge_domain::SubagentTaskSession>> {
            Ok(None)
        }

        async fn list_subagent_task_sessions(
            &self,
            _filter: forge_domain::SubagentTaskSessionFilter,
        ) -> Result<Vec<forge_domain::SubagentTaskSession>> {
            Ok(Vec::new())
        }

        async fn last_conversation(&self) -> Result<Option<Conversation>> {
            Ok(self.conversations.lock().await.values().next().cloned())
        }

        async fn delete_conversation(&self, conversation_id: &ConversationId) -> Result<()> {
            self.conversations.lock().await.remove(conversation_id);
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl LearningService for ChatFlowLearningState {
        async fn capture_candidate_from_conversation(
            &self,
            conversation_id: ConversationId,
            source_event_id: String,
            summary: String,
        ) -> Result<LearningLedgerEvent> {
            let event = LearningLedgerEvent::capture_candidate(
                summary,
                LearningProvenance::conversation(
                    conversation_id,
                    source_event_id,
                    "runtime-proof-source-fingerprint",
                ),
                chrono::Utc::now(),
            )?;
            let mut events = self.learning_events.lock().await;
            let event = events
                .iter()
                .find(|existing| existing.idempotency_key == event.idempotency_key)
                .cloned()
                .unwrap_or_else(|| {
                    events.push(event.clone());
                    event
                });
            Ok(event)
        }

        async fn insert_learning_event(
            &self,
            event: LearningLedgerEvent,
        ) -> Result<LearningLedgerEvent> {
            self.learning_events.lock().await.push(event.clone());
            Ok(event)
        }

        async fn list_learning_records(
            &self,
            review_state: Option<LearningReviewState>,
            limit: usize,
        ) -> Result<Vec<LearningRecordProjection>> {
            let mut records = self.learning_records.lock().await.clone();
            if let Some(review_state) = review_state {
                records.retain(|record| record.review_state == review_state);
            }
            records.truncate(limit);
            Ok(records)
        }

        async fn learning_freshness(
            &self,
            _review_state: Option<LearningReviewState>,
        ) -> Result<LearningLedgerFreshness> {
            Ok(LearningLedgerFreshness {
                ledger_cursor: self.learning_records.lock().await.len() as i64,
                projection_version: 1,
                review_state_fingerprint: "runtime-proof-learning".to_string(),
            })
        }
    }

    #[async_trait::async_trait]
    impl ProviderService for ChatFlowLearningState {
        async fn chat(
            &self,
            _model_id: &ModelId,
            context: Context,
            _provider: Provider<Url>,
        ) -> ResultStream<ChatCompletionMessage, anyhow::Error> {
            *self.captured_provider_context.lock().await = Some(context);
            let message = ChatCompletionMessage::assistant(Content::full("runtime proof response"))
                .finish_reason(FinishReason::Stop);
            Ok(Box::pin(tokio_stream::iter(std::iter::once(Ok(message)))))
        }

        async fn models(&self, _provider: Provider<Url>) -> Result<Vec<Model>> {
            Ok(vec![self.model.clone()])
        }

        async fn get_provider(&self, id: ProviderId) -> Result<Provider<Url>> {
            assert_eq!(id, self.provider.id);
            Ok(self.provider.clone())
        }

        async fn get_all_providers(&self) -> Result<Vec<AnyProvider>> {
            Ok(vec![AnyProvider::Url(self.provider.clone())])
        }

        async fn upsert_credential(&self, _credential: forge_domain::AuthCredential) -> Result<()> {
            anyhow::bail!("unused credential upsert")
        }

        async fn remove_credential(&self, _id: &ProviderId) -> Result<()> {
            anyhow::bail!("unused credential removal")
        }

        async fn migrate_env_credentials(&self) -> Result<Option<forge_domain::MigrationResult>> {
            Ok(None)
        }
    }

    #[async_trait::async_trait]
    impl AppConfigService for ChatFlowLearningState {
        async fn get_session_config(&self) -> Option<forge_domain::ModelConfig> {
            Some(forge_domain::ModelConfig::new(
                self.provider.id.clone(),
                self.model.id.clone(),
            ))
        }

        async fn get_commit_config(&self) -> Result<Option<forge_domain::ModelConfig>> {
            Ok(None)
        }

        async fn get_suggest_config(&self) -> Result<Option<forge_domain::ModelConfig>> {
            Ok(None)
        }

        async fn get_reasoning_effort(&self) -> Result<Option<forge_domain::Effort>> {
            Ok(None)
        }

        async fn update_config(&self, _ops: Vec<forge_domain::ConfigOperation>) -> Result<()> {
            anyhow::bail!("unused config update")
        }
    }

    #[async_trait::async_trait]
    impl AgentRegistry for ChatFlowLearningState {
        async fn get_active_agent_id(&self) -> Result<Option<AgentId>> {
            Ok(Some(self.agent.id.clone()))
        }

        async fn set_active_agent_id(&self, _agent_id: AgentId) -> Result<()> {
            anyhow::bail!("unused active agent update")
        }

        async fn get_agents(&self) -> Result<Vec<Agent>> {
            Ok(vec![self.agent.clone()])
        }

        async fn get_agent_infos(&self) -> Result<Vec<forge_domain::AgentInfo>> {
            Ok(Vec::new())
        }

        async fn get_agent(&self, agent_id: &AgentId) -> Result<Option<Agent>> {
            Ok((agent_id == &self.agent.id).then(|| self.agent.clone()))
        }

        async fn reload_agents(&self) -> Result<()> {
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl ProviderAuthService for ChatFlowLearningState {
        async fn init_provider_auth(
            &self,
            _provider_id: ProviderId,
            _method: AuthMethod,
        ) -> Result<AuthContextRequest> {
            anyhow::bail!("unused provider auth init")
        }

        async fn complete_provider_auth(
            &self,
            _provider_id: ProviderId,
            _context: AuthContextResponse,
            _timeout: std::time::Duration,
        ) -> Result<()> {
            anyhow::bail!("unused provider auth completion")
        }

        async fn refresh_provider_credential(
            &self,
            provider: Provider<Url>,
        ) -> Result<Provider<Url>> {
            Ok(provider)
        }
    }

    #[async_trait::async_trait]
    impl CustomInstructionsService for ChatFlowLearningState {
        async fn get_custom_instructions(&self) -> Vec<String> {
            Vec::new()
        }
    }

    #[async_trait::async_trait]
    impl McpService for ChatFlowLearningState {
        async fn get_mcp_servers(&self) -> Result<McpServers> {
            Ok(McpServers::default())
        }

        async fn execute_mcp(&self, _call: ToolCallFull) -> Result<ToolOutput> {
            anyhow::bail!("unused mcp execution")
        }

        async fn reload_mcp(&self) -> Result<()> {
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl WorkspaceService for ChatFlowLearningState {
        async fn sync_workspace(
            &self,
            _path: PathBuf,
        ) -> Result<forge_stream::MpscStream<Result<SyncProgress>>> {
            anyhow::bail!("unused workspace sync")
        }

        async fn produce_workspace_exact_fact_reference(
            &self,
            _path: PathBuf,
        ) -> Result<WorkspaceExactFactReferenceReport> {
            anyhow::bail!("unused workspace exact-fact reference")
        }

        async fn query_workspace(
            &self,
            _path: PathBuf,
            _params: SearchParams<'_>,
        ) -> Result<Vec<Node>> {
            Ok(Vec::new())
        }

        async fn list_workspaces(&self) -> Result<Vec<WorkspaceInfo>> {
            Ok(Vec::new())
        }

        async fn get_workspace_info(&self, _path: PathBuf) -> Result<Option<WorkspaceInfo>> {
            Ok(None)
        }

        async fn is_indexed(&self, _path: &Path) -> Result<bool> {
            Ok(false)
        }

        async fn delete_workspace(&self, _workspace_id: &WorkspaceId) -> Result<()> {
            Ok(())
        }

        async fn delete_workspaces(&self, _workspace_ids: &[WorkspaceId]) -> Result<()> {
            Ok(())
        }

        async fn project_model_context_diagnostic(
            &self,
            path: &Path,
        ) -> Result<WorkspaceContextManifestDiagnostic> {
            Ok(WorkspaceContextManifestDiagnostic {
                workspace_root: path.to_path_buf(),
                manifest_found: false,
                manifest_path: path.join(".forge_project_model/project_manifest.json"),
                freshness: WorkspaceContextFreshness::Unknown {
                    reason: "runtime proof does not index workspace".to_string(),
                },
            })
        }

        async fn get_workspace_status(&self, _path: PathBuf) -> Result<Vec<FileStatus>> {
            Ok(Vec::new())
        }

        async fn is_authenticated(&self) -> Result<bool> {
            Ok(false)
        }

        async fn init_auth_credentials(&self) -> Result<WorkspaceAuth> {
            anyhow::bail!("unused workspace auth")
        }

        async fn init_workspace(&self, _path: PathBuf) -> Result<WorkspaceId> {
            anyhow::bail!("unused workspace init")
        }
    }

    #[async_trait::async_trait]
    impl SteerService for ChatFlowLearningState {
        async fn enqueue_steer(
            &self,
            _conversation_id: &ConversationId,
            _message: SteerMessage,
        ) -> Result<()> {
            Ok(())
        }

        async fn clear_steer(&self, _conversation_id: &ConversationId) -> Result<()> {
            Ok(())
        }

        async fn drain_steer(
            &self,
            _conversation_id: &ConversationId,
        ) -> Result<Vec<SteerMessage>> {
            Ok(Vec::new())
        }
    }

    macro_rules! impl_unused_service_traits {
        ($type:ty) => {
            #[async_trait::async_trait]
            impl TemplateService for $type {
                async fn register_template(&self, _path: PathBuf) -> Result<()> {
                    anyhow::bail!("unused template registration")
                }

                async fn render_template<V: serde::Serialize + Send + Sync>(
                    &self,
                    _template: forge_domain::Template<V>,
                    _object: &V,
                ) -> Result<String> {
                    anyhow::bail!("unused template rendering")
                }
            }

            #[async_trait::async_trait]
            impl AttachmentService for $type {
                async fn attachments(&self, _url: &str) -> Result<Vec<forge_domain::Attachment>> {
                    Ok(Vec::new())
                }
            }

            #[async_trait::async_trait]
            impl FileDiscoveryService for $type {
                async fn collect_files(&self, _config: Walker) -> Result<Vec<forge_domain::File>> {
                    Ok(Vec::new())
                }

                async fn list_current_directory(&self) -> Result<Vec<forge_domain::File>> {
                    Ok(Vec::new())
                }
            }

            #[async_trait::async_trait]
            impl McpConfigManager for $type {
                async fn read_mcp_config(&self, _scope: Option<&Scope>) -> Result<McpConfig> {
                    anyhow::bail!("unused mcp config read")
                }

                async fn write_mcp_config(
                    &self,
                    _config: &McpConfig,
                    _scope: &Scope,
                ) -> Result<()> {
                    anyhow::bail!("unused mcp config write")
                }
            }

            #[async_trait::async_trait]
            impl FsWriteService for $type {
                async fn write(
                    &self,
                    _path: String,
                    _content: String,
                    _overwrite: bool,
                ) -> Result<crate::FsWriteOutput> {
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
                ) -> Result<crate::PlanCreateOutput> {
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
                ) -> Result<crate::PatchOutput> {
                    anyhow::bail!("unused fs patch")
                }

                async fn multi_patch(
                    &self,
                    _path: String,
                    _edits: Vec<forge_domain::PatchEdit>,
                ) -> Result<crate::PatchOutput> {
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
                ) -> Result<crate::ReadOutput> {
                    anyhow::bail!("unused fs read")
                }
            }

            #[async_trait::async_trait]
            impl ImageReadService for $type {
                async fn read_image(&self, _path: String) -> Result<forge_domain::Image> {
                    anyhow::bail!("unused image read")
                }
            }

            #[async_trait::async_trait]
            impl FsRemoveService for $type {
                async fn remove(&self, _path: String) -> Result<crate::FsRemoveOutput> {
                    anyhow::bail!("unused fs remove")
                }
            }

            #[async_trait::async_trait]
            impl FsSearchService for $type {
                async fn search(
                    &self,
                    _params: forge_domain::FSSearch,
                ) -> Result<Option<crate::SearchResult>> {
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
                ) -> Result<Option<String>> {
                    Ok(None)
                }
            }

            #[async_trait::async_trait]
            impl FsUndoService for $type {
                async fn undo(&self, _path: String) -> Result<crate::FsUndoOutput> {
                    anyhow::bail!("unused fs undo")
                }
            }

            #[async_trait::async_trait]
            impl NetFetchService for $type {
                async fn fetch(
                    &self,
                    _url: String,
                    _raw: Option<bool>,
                ) -> Result<crate::HttpResponse> {
                    anyhow::bail!("unused net fetch")
                }
            }

            #[async_trait::async_trait]
            impl ShellService for $type {
                async fn execute(
                    &self,
                    _request: crate::ShellExecuteRequest,
                ) -> Result<crate::ShellOutput> {
                    anyhow::bail!("unused shell execute")
                }
            }

            #[async_trait::async_trait]
            impl AuthService for $type {
                async fn user_info(&self, _api_key: &str) -> Result<User> {
                    anyhow::bail!("unused auth user info")
                }

                async fn user_usage(&self, _api_key: &str) -> Result<UserUsage> {
                    anyhow::bail!("unused auth user usage")
                }
            }

            #[async_trait::async_trait]
            impl CommandLoaderService for $type {
                async fn get_commands(&self) -> Result<Vec<forge_domain::Command>> {
                    Ok(Vec::new())
                }
            }

            #[async_trait::async_trait]
            impl PolicyService for $type {
                async fn check_operation_permission(
                    &self,
                    _operation: &PermissionOperation,
                ) -> Result<crate::PolicyDecision> {
                    Ok(crate::PolicyDecision { allowed: true, path: None })
                }
            }

            #[async_trait::async_trait]
            impl SkillFetchService for $type {
                async fn fetch_skill(&self, _skill_name: String) -> Result<forge_domain::Skill> {
                    anyhow::bail!("unused skill fetch")
                }

                async fn list_skills(&self) -> Result<Vec<forge_domain::Skill>> {
                    Ok(Vec::new())
                }
            }
        };
    }

    impl_unused_service_traits!(ChatFlowLearningState);

    impl Services for ChatFlowLearningHarness {
        type ProviderService = ChatFlowLearningState;
        type AppConfigService = ChatFlowLearningState;
        type ConversationService = ChatFlowLearningState;
        type LearningService = ChatFlowLearningState;
        type SteerService = ChatFlowLearningState;
        type TemplateService = ChatFlowLearningState;
        type AttachmentService = ChatFlowLearningState;
        type CustomInstructionsService = ChatFlowLearningState;
        type FileDiscoveryService = ChatFlowLearningState;
        type McpConfigManager = ChatFlowLearningState;
        type FsWriteService = ChatFlowLearningState;
        type PlanCreateService = ChatFlowLearningState;
        type FsPatchService = ChatFlowLearningState;
        type FsReadService = ChatFlowLearningState;
        type ImageReadService = ChatFlowLearningState;
        type FsRemoveService = ChatFlowLearningState;
        type FsSearchService = ChatFlowLearningState;
        type FollowUpService = ChatFlowLearningState;
        type FsUndoService = ChatFlowLearningState;
        type NetFetchService = ChatFlowLearningState;
        type ShellService = ChatFlowLearningState;
        type McpService = ChatFlowLearningState;
        type AuthService = ChatFlowLearningState;
        type AgentRegistry = ChatFlowLearningState;
        type CommandLoaderService = ChatFlowLearningState;
        type PolicyService = ChatFlowLearningState;
        type ProviderAuthService = ChatFlowLearningState;
        type WorkspaceService = ChatFlowLearningState;
        type SkillFetchService = ChatFlowLearningState;

        fn provider_service(&self) -> &Self::ProviderService {
            &self.state
        }
        fn config_service(&self) -> &Self::AppConfigService {
            &self.state
        }
        fn conversation_service(&self) -> &Self::ConversationService {
            &self.state
        }
        fn learning_service(&self) -> &Self::LearningService {
            &self.state
        }
        fn steer_service(&self) -> &Self::SteerService {
            &self.state
        }
        fn template_service(&self) -> &Self::TemplateService {
            &self.state
        }
        fn attachment_service(&self) -> &Self::AttachmentService {
            &self.state
        }
        fn file_discovery_service(&self) -> &Self::FileDiscoveryService {
            &self.state
        }
        fn mcp_config_manager(&self) -> &Self::McpConfigManager {
            &self.state
        }
        fn fs_create_service(&self) -> &Self::FsWriteService {
            &self.state
        }
        fn plan_create_service(&self) -> &Self::PlanCreateService {
            &self.state
        }
        fn fs_patch_service(&self) -> &Self::FsPatchService {
            &self.state
        }
        fn fs_read_service(&self) -> &Self::FsReadService {
            &self.state
        }
        fn image_read_service(&self) -> &Self::ImageReadService {
            &self.state
        }
        fn fs_remove_service(&self) -> &Self::FsRemoveService {
            &self.state
        }
        fn fs_search_service(&self) -> &Self::FsSearchService {
            &self.state
        }
        fn follow_up_service(&self) -> &Self::FollowUpService {
            &self.state
        }
        fn fs_undo_service(&self) -> &Self::FsUndoService {
            &self.state
        }
        fn net_fetch_service(&self) -> &Self::NetFetchService {
            &self.state
        }
        fn shell_service(&self) -> &Self::ShellService {
            &self.state
        }
        fn mcp_service(&self) -> &Self::McpService {
            &self.state
        }
        fn custom_instructions_service(&self) -> &Self::CustomInstructionsService {
            &self.state
        }
        fn auth_service(&self) -> &Self::AuthService {
            &self.state
        }
        fn agent_registry(&self) -> &Self::AgentRegistry {
            &self.state
        }
        fn command_loader_service(&self) -> &Self::CommandLoaderService {
            &self.state
        }
        fn policy_service(&self) -> &Self::PolicyService {
            &self.state
        }
        fn provider_auth_service(&self) -> &Self::ProviderAuthService {
            &self.state
        }
        fn workspace_service(&self) -> &Self::WorkspaceService {
            &self.state
        }
        fn skill_fetch_service(&self) -> &Self::SkillFetchService {
            &self.state
        }
    }

    struct ProjectContextHarness {
        cwd: PathBuf,
        empty_paths: Vec<PathBuf>,
        error_paths: Vec<PathBuf>,
        stale_paths: Vec<PathBuf>,
        unknown_paths: Vec<PathBuf>,
        captured_context: Mutex<Option<Context>>,
        workspace_queries: AtomicUsize,
        queried_workspaces: Mutex<Vec<PathBuf>>,
        query_filters: Mutex<Vec<Option<String>>>,
        index_checks: AtomicUsize,
        learning_records: Mutex<Vec<LearningRecordProjection>>,
        learning_freshness: LearningLedgerFreshness,
    }

    impl ProjectContextHarness {
        fn new(cwd: PathBuf) -> Arc<Self> {
            Self::new_with_empty_error_stale_and_unknown_paths(
                cwd,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        }

        fn new_with_empty_paths(cwd: PathBuf, empty_paths: Vec<PathBuf>) -> Arc<Self> {
            Self::new_with_empty_error_stale_and_unknown_paths(
                cwd,
                empty_paths,
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        }

        fn new_with_error_paths(cwd: PathBuf, error_paths: Vec<PathBuf>) -> Arc<Self> {
            Self::new_with_empty_error_stale_and_unknown_paths(
                cwd,
                Vec::new(),
                error_paths,
                Vec::new(),
                Vec::new(),
            )
        }

        fn new_with_stale_paths(cwd: PathBuf, stale_paths: Vec<PathBuf>) -> Arc<Self> {
            Self::new_with_empty_error_stale_and_unknown_paths(
                cwd,
                Vec::new(),
                Vec::new(),
                stale_paths,
                Vec::new(),
            )
        }

        fn new_with_unknown_paths(cwd: PathBuf, unknown_paths: Vec<PathBuf>) -> Arc<Self> {
            Self::new_with_empty_error_stale_and_unknown_paths(
                cwd,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                unknown_paths,
            )
        }

        fn new_with_empty_error_stale_and_unknown_paths(
            cwd: PathBuf,
            empty_paths: Vec<PathBuf>,
            error_paths: Vec<PathBuf>,
            stale_paths: Vec<PathBuf>,
            unknown_paths: Vec<PathBuf>,
        ) -> Arc<Self> {
            Arc::new(Self {
                cwd,
                empty_paths,
                error_paths,
                stale_paths,
                unknown_paths,
                captured_context: Mutex::new(None),
                workspace_queries: AtomicUsize::new(0),
                queried_workspaces: Mutex::new(Vec::new()),
                query_filters: Mutex::new(Vec::new()),
                index_checks: AtomicUsize::new(0),
                learning_records: Mutex::new(Vec::new()),
                learning_freshness: LearningLedgerFreshness {
                    ledger_cursor: 1,
                    projection_version: 1,
                    review_state_fingerprint: "fixture-learning".to_string(),
                },
            })
        }

        async fn set_learning_records(&self, records: Vec<LearningRecordProjection>) {
            *self.learning_records.lock().await = records;
        }
    }

    impl EnvironmentInfra for ProjectContextHarness {
        type Config = ForgeConfig;

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
            Ok(ForgeConfig::default())
        }

        async fn update_environment(&self, _ops: Vec<forge_domain::ConfigOperation>) -> Result<()> {
            anyhow::bail!("unused environment update")
        }
    }

    #[async_trait::async_trait]
    impl WorkspaceService for ProjectContextHarness {
        async fn sync_workspace(
            &self,
            _path: PathBuf,
        ) -> Result<forge_stream::MpscStream<Result<SyncProgress>>> {
            anyhow::bail!("unused workspace sync")
        }

        async fn produce_workspace_exact_fact_reference(
            &self,
            _path: PathBuf,
        ) -> Result<WorkspaceExactFactReferenceReport> {
            anyhow::bail!("unused workspace exact-fact reference")
        }

        async fn query_workspace(
            &self,
            path: PathBuf,
            params: SearchParams<'_>,
        ) -> Result<Vec<Node>> {
            self.workspace_queries.fetch_add(1, Ordering::SeqCst);
            assert!(params.query.contains("automatic injection needle"));
            assert_eq!(params.limit, Some(3));
            self.queried_workspaces.lock().await.push(path.clone());
            self.query_filters
                .lock()
                .await
                .push(params.starts_with.clone());
            if self
                .error_paths
                .iter()
                .any(|error_path| error_path == &path)
            {
                anyhow::bail!("fixture query failure for {}", path.display());
            }
            if self
                .empty_paths
                .iter()
                .any(|empty_path| empty_path == &path)
            {
                return Ok(Vec::new());
            }
            let file_path = params.starts_with.as_deref().unwrap_or("src/lib.rs");
            let long_content = (0..40)
                .map(|index| format!("pub fn long_{index}() -> usize {{ {index} }}"))
                .collect::<Vec<_>>()
                .join("\n");
            Ok(vec![
                Node {
                    node_id: NodeId::new("symbol:src/lib.rs:automatic_injection_needle"),
                    node: NodeData::FileChunk(FileChunk {
                        file_path: file_path.to_string(),
                        content: "pub fn automatic_injection_needle() -> usize { 42 }".to_string(),
                        start_line: 3,
                        end_line: 3,
                    }),
                    relevance: Some(0.875),
                    distance: None,
                },
                Node {
                    node_id: NodeId::new("symbol:src/long.rs:long_block"),
                    node: NodeData::FileChunk(FileChunk {
                        file_path: "src/long.rs".to_string(),
                        content: long_content,
                        start_line: 10,
                        end_line: 80,
                    }),
                    relevance: Some(0.75),
                    distance: None,
                },
                Node {
                    node_id: NodeId::new("file:src/full.rs"),
                    node: NodeData::File(forge_domain::FileNode {
                        file_path: "src/full.rs".to_string(),
                        content: "pub fn full_file_should_not_render() {}".repeat(100),
                        hash: "full-file-hash".to_string(),
                    }),
                    relevance: Some(0.5),
                    distance: None,
                },
                Node {
                    node_id: NodeId::new("symbol:src/extra.rs:extra"),
                    node: NodeData::FileChunk(FileChunk {
                        file_path: "src/extra.rs".to_string(),
                        content: "pub fn extra_should_not_render() {}".to_string(),
                        start_line: 1,
                        end_line: 1,
                    }),
                    relevance: Some(0.25),
                    distance: None,
                },
            ])
        }

        async fn list_workspaces(&self) -> Result<Vec<WorkspaceInfo>> {
            anyhow::bail!("unused workspace list")
        }

        async fn get_workspace_info(&self, _path: PathBuf) -> Result<Option<WorkspaceInfo>> {
            anyhow::bail!("unused workspace info")
        }

        async fn is_indexed(&self, path: &Path) -> Result<bool> {
            self.project_model_context_diagnostic(path)
                .await
                .map(|diagnostic| diagnostic.can_inject())
        }

        async fn delete_workspace(&self, _workspace_id: &WorkspaceId) -> Result<()> {
            anyhow::bail!("unused workspace delete")
        }

        async fn delete_workspaces(&self, _workspace_ids: &[WorkspaceId]) -> Result<()> {
            anyhow::bail!("unused workspace deletes")
        }

        async fn project_model_context_diagnostic(
            &self,
            path: &Path,
        ) -> Result<WorkspaceContextManifestDiagnostic> {
            self.index_checks.fetch_add(1, Ordering::SeqCst);
            let manifest_path = path.join(".forge_project_model/project_manifest.json");
            let manifest_found = manifest_path.is_file();
            let freshness = if self.stale_paths.iter().any(|stale_path| stale_path == path) {
                WorkspaceContextFreshness::Stale {
                    changed: vec!["src/lib.rs".to_string()],
                    deleted: Vec::new(),
                    added: Vec::new(),
                }
            } else if self
                .unknown_paths
                .iter()
                .any(|unknown_path| unknown_path == path)
            {
                WorkspaceContextFreshness::Unknown {
                    reason: "fixture freshness unavailable".to_string(),
                }
            } else if manifest_found {
                WorkspaceContextFreshness::Fresh
            } else {
                WorkspaceContextFreshness::Unknown {
                    reason: "project-model manifest not found".to_string(),
                }
            };
            Ok(WorkspaceContextManifestDiagnostic {
                workspace_root: path.to_path_buf(),
                manifest_found,
                manifest_path,
                freshness,
            })
        }

        async fn get_workspace_status(&self, _path: PathBuf) -> Result<Vec<FileStatus>> {
            anyhow::bail!("unused workspace status")
        }

        async fn is_authenticated(&self) -> Result<bool> {
            Ok(true)
        }

        async fn init_auth_credentials(&self) -> Result<WorkspaceAuth> {
            anyhow::bail!("unused workspace auth")
        }

        async fn init_workspace(&self, _path: PathBuf) -> Result<WorkspaceId> {
            anyhow::bail!("unused workspace init")
        }
    }

    #[async_trait::async_trait]
    impl LearningService for ProjectContextHarness {
        async fn capture_candidate_from_conversation(
            &self,
            _conversation_id: ConversationId,
            _source_event_id: String,
            _summary: String,
        ) -> Result<LearningLedgerEvent> {
            anyhow::bail!("unused learning capture")
        }

        async fn insert_learning_event(
            &self,
            _event: LearningLedgerEvent,
        ) -> Result<LearningLedgerEvent> {
            anyhow::bail!("unused learning insert")
        }

        async fn list_learning_records(
            &self,
            review_state: Option<LearningReviewState>,
            limit: usize,
        ) -> Result<Vec<LearningRecordProjection>> {
            let mut records = self.learning_records.lock().await.clone();
            if let Some(review_state) = review_state {
                records.retain(|record| record.review_state == review_state);
            }
            records.truncate(limit);
            Ok(records)
        }

        async fn learning_freshness(
            &self,
            _review_state: Option<LearningReviewState>,
        ) -> Result<LearningLedgerFreshness> {
            Ok(self.learning_freshness.clone())
        }
    }

    #[async_trait::async_trait]
    impl AgentService for ProjectContextHarness {
        async fn chat_agent(
            &self,
            _id: &ModelId,
            context: Context,
            _provider_id: Option<ProviderId>,
        ) -> ResultStream<ChatCompletionMessage, anyhow::Error> {
            *self.captured_context.lock().await = Some(context);
            let message = ChatCompletionMessage::assistant(Content::full("done"))
                .finish_reason(FinishReason::Stop);
            Ok(Box::pin(tokio_stream::iter(std::iter::once(Ok(message)))))
        }

        async fn call(
            &self,
            _agent: &Agent,
            _context: &ToolCallContext,
            call: ToolCallFull,
        ) -> ToolResult {
            ToolResult::new(call.name)
                .failure(anyhow::anyhow!("tool calls are not expected in this test"))
        }

        async fn update(&self, _conversation: Conversation) -> Result<()> {
            Ok(())
        }

        async fn drain_steer_messages(
            &self,
            _conversation_id: &forge_domain::ConversationId,
        ) -> Result<Vec<SteerMessage>> {
            Ok(Vec::new())
        }
    }

    fn fixture_workspace() -> Result<(TempDir, PathBuf)> {
        let fixture = TempDir::new()?;
        let root = fixture.path().join("workspace");
        create_indexed_workspace(&root)?;
        Ok((fixture, root))
    }

    fn create_indexed_workspace(root: &Path) -> Result<()> {
        fs::create_dir_all(root.join("src"))?;
        fs::create_dir_all(root.join(".forge_project_model"))?;
        fs::write(
            root.join("src/lib.rs"),
            "pub fn unrelated() {}\n\npub fn automatic_injection_needle() -> usize { 42 }\n",
        )?;
        fs::write(
            root.join(".forge_project_model/project_manifest.json"),
            r#"{"version":1,"root":"fixture","files":[],"file_nodes":[],"symbols":[],"edges":[],"shards":[],"manifest_hash":"fixture"}"#,
        )?;
        Ok(())
    }

    fn fixture_learning_projection(
        review_state: LearningReviewState,
        summary: &str,
    ) -> LearningRecordProjection {
        let conversation_id = ConversationId::generate();
        LearningRecordProjection {
            record_id: LearningRecordId::generate(),
            summary: summary.to_string(),
            review_state,
            redaction_status: LearningRedactionStatus::Clean,
            provenance: LearningProvenance::conversation(
                conversation_id,
                "learning-event-1",
                "learning-source-fingerprint",
            ),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            schema_version: LEARNING_LEDGER_SCHEMA_VERSION,
        }
    }

    #[tokio::test]
    async fn chat_flow_saves_conversation_then_captures_learning_candidate() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ChatFlowLearningHarness::new(root);
        let conversation = Conversation::generate();
        let conversation_id = conversation.id;
        setup.upsert_conversation(conversation).await?;
        let app = ForgeApp::new(setup.clone());
        let mut stream = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            app.chat(
                setup.state.agent.id.clone(),
                ChatRequest::new(
                    Event::new("runtime self-learning proof request"),
                    conversation_id,
                ),
            ),
        )
        .await??;

        for _ in 0..32 {
            if !setup.state.learning_events.lock().await.is_empty() {
                break;
            }
            match tokio::time::timeout(std::time::Duration::from_millis(250), stream.next()).await {
                Ok(Some(response)) => {
                    response?;
                }
                Ok(None) => break,
                Err(_) => {}
            }
        }

        let saved = setup
            .find_conversation(&conversation_id)
            .await?
            .expect("conversation should be saved after chat flow");
        let events = setup.state.learning_events.lock().await.clone();
        let actual = events.first().map(|event| {
            (
                events.len(),
                event.event_kind,
                event.provenance.conversation_id,
                event.summary.contains("conversation_saved"),
                event
                    .summary
                    .contains("runtime self-learning proof request"),
                saved
                    .context
                    .as_ref()
                    .is_some_and(|context| context.messages.len() >= 2),
            )
        });
        let expected = Some((
            1usize,
            LearningEventKind::CandidateCaptured,
            Some(conversation_id),
            true,
            true,
            true,
        ));
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn chat_flow_injects_only_accepted_learning_into_provider_context() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ChatFlowLearningHarness::new(root);
        setup
            .set_learning_records(vec![
                fixture_learning_projection(
                    LearningReviewState::Candidate,
                    "candidate runtime learning must stay out",
                ),
                fixture_learning_projection(
                    LearningReviewState::Accepted,
                    "accepted runtime learning reaches provider context",
                ),
            ])
            .await;
        let conversation = Conversation::generate();
        let conversation_id = conversation.id;
        setup.upsert_conversation(conversation).await?;
        let app = ForgeApp::new(setup.clone());
        let mut stream = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            app.chat(
                setup.state.agent.id.clone(),
                ChatRequest::new(
                    Event::new("runtime accepted learning proof"),
                    conversation_id,
                ),
            ),
        )
        .await??;

        for _ in 0..32 {
            if setup.state.captured_provider_context.lock().await.is_some() {
                break;
            }
            match tokio::time::timeout(std::time::Duration::from_millis(250), stream.next()).await {
                Ok(Some(response)) => {
                    response?;
                }
                Ok(None) => break,
                Err(_) => {}
            }
        }

        let captured_context = setup
            .state
            .captured_provider_context
            .lock()
            .await
            .clone()
            .expect("provider context should be captured by fake provider");
        let learning_message = captured_context
            .messages
            .iter()
            .find_map(|message| match &message.message {
                ContextMessage::Text(text) if text.is_learning_context() => Some(text),
                _ => None,
            })
            .expect("accepted learning context should be injected before provider call");
        let actual = vec![
            learning_message
                .content
                .contains("accepted runtime learning reaches provider context"),
            learning_message
                .content
                .contains("candidate runtime learning must stay out"),
            learning_message.droppable,
            learning_message.is_cache_eligible(),
        ];
        let expected = vec![true, false, true, false];
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_context_injection_uses_only_reviewed_accepted_records() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        setup
            .set_learning_records(vec![
                fixture_learning_projection(
                    LearningReviewState::Candidate,
                    "candidate must not inject",
                ),
                fixture_learning_projection(
                    LearningReviewState::Accepted,
                    "accepted reviewed learning",
                ),
            ])
            .await;
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        let actual = ProjectContextInjection::new(setup, agent)
            .inject_learning(conversation)
            .await
            .context
            .unwrap();
        let learning_message = actual
            .messages
            .iter()
            .find_map(|message| match &message.message {
                ContextMessage::Text(text) if text.is_learning_context() => Some(text),
                _ => None,
            })
            .expect("accepted learning context should be injected");

        assert!(
            learning_message
                .content
                .contains("accepted reviewed learning")
        );
        assert!(
            !learning_message
                .content
                .contains("candidate must not inject")
        );
        assert_eq!(learning_message.droppable, true);
        assert_eq!(learning_message.is_cache_eligible(), false);
        Ok(())
    }

    #[tokio::test]
    async fn learning_context_injection_does_not_inject_unreviewed_candidates() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        setup
            .set_learning_records(vec![fixture_learning_projection(
                LearningReviewState::Candidate,
                "candidate must not inject",
            )])
            .await;
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        let actual = ProjectContextInjection::new(setup, agent)
            .inject_learning(conversation)
            .await
            .context
            .unwrap();
        let injected = actual
            .messages
            .iter()
            .any(|message| match &message.message {
                ContextMessage::Text(text) => text.is_learning_context(),
                _ => false,
            });

        assert_eq!(injected, false);
        Ok(())
    }

    #[tokio::test]
    async fn learning_context_injection_skips_payload_over_char_budget() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        setup
            .set_learning_records(vec![fixture_learning_projection(
                LearningReviewState::Accepted,
                &"oversized reviewed learning ".repeat(1_000),
            )])
            .await;
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        let actual = ProjectContextInjection::new(setup, agent)
            .inject_learning(conversation)
            .await
            .context
            .unwrap();
        let injected = actual
            .messages
            .iter()
            .any(|message| match &message.message {
                ContextMessage::Text(text) => text.is_learning_context(),
                _ => false,
            });

        assert_eq!(injected, false);
        Ok(())
    }

    #[test]
    fn compaction_provider_request_metrics_do_not_use_historical_usage() -> Result<()> {
        let setup = OpenAiProviderRequestEstimate {
            estimated_input_tokens: 512,
            serialized_request_bytes: 512,
            media_token_padding: 0,
            output_token_reservation: 3_392,
            message_count: 1,
            tool_count: 0,
            messages_bytes: 128,
            tools_bytes: 0,
        };
        let actual = provider_request_compaction_estimate(setup, Some(3_392));
        let expected = ProviderRequestEstimate::new(512, Some(3_392));

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn project_model_query_ignores_live_runtime_context_message() -> Result<()> {
        let (_fixture, _root) = fixture_workspace()?;
        let model_id = ModelId::new("test-model");
        let conversation = Conversation::generate().context(
            Context::default()
                .add_message(ContextMessage::user(
                    "find automatic injection needle",
                    Some(model_id.clone()),
                ))
                .add_message(ContextMessage::Text(
                    TextMessage::new(
                        Role::User,
                        "<runtime_context freshness=\"live\" cache=\"uncached\">time</runtime_context>",
                    )
                    .model(model_id.clone())
                    .runtime_context()
                    .cacheable(false),
                )),
        );

        let actual = ProjectContextInjection::<ProjectContextHarness>::query_from_conversation(
            &conversation,
        )
        .unwrap();
        let expected = "find automatic injection needle";
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn project_model_query_preserves_real_user_message_containing_runtime_context_text()
    -> Result<()> {
        let (_fixture, _root) = fixture_workspace()?;
        let model_id = ModelId::new("test-model");
        let conversation = Conversation::generate().context(
            Context::default()
                .add_message(ContextMessage::user(
                    "previous automatic injection needle",
                    Some(model_id.clone()),
                ))
                .add_message(ContextMessage::user(
                    "explain the literal <runtime_context tag in prompts",
                    Some(model_id.clone()),
                )),
        );

        let actual = ProjectContextInjection::<ProjectContextHarness>::query_from_conversation(
            &conversation,
        )
        .unwrap();
        let expected = "explain the literal <runtime_context tag in prompts";
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_queries_absolute_file_tag_workspace_outside_cwd() -> Result<()> {
        let fixture = TempDir::new()?;
        let cwd = fixture.path().join("cwd-workspace");
        let other = fixture.path().join("other-workspace");
        create_indexed_workspace(&cwd)?;
        create_indexed_workspace(&other)?;
        let setup = ProjectContextHarness::new(cwd.clone());
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let mentioned_file = other.join("src/lib.rs");
        let conversation =
            Conversation::generate().context(Context::default().add_message(ContextMessage::user(
                format!(
                    "find automatic injection needle in @[{}]",
                    mentioned_file.display()
                ),
                Some(model_id),
            )));

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let expected_workspaces = vec![cwd, other];
        assert_eq!(*setup.queried_workspaces.lock().await, expected_workspaces);
        let expected_filters = vec![None, None];
        assert_eq!(*setup.query_filters.lock().await, expected_filters);
        assert_eq!(
            actual
                .context
                .unwrap()
                .messages
                .iter()
                .filter(|message| message
                    .content()
                    .is_some_and(|content| content.contains("<project_model_context")))
                .count(),
            2usize,
        );
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_queries_backticked_path_inside_cwd_with_filter() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation =
            Conversation::generate().context(Context::default().add_message(ContextMessage::user(
                "find automatic injection needle in `src/lib.rs`",
                Some(model_id),
            )));

        ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let actual = setup.query_filters.lock().await.clone();
        let expected = vec![None];
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_queries_directory_path_inside_cwd_with_safe_filter() -> Result<()>
    {
        let (_fixture, root) = fixture_workspace()?;
        let src_dir = root.join("src");
        let setup = ProjectContextHarness::new(root);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation =
            Conversation::generate().context(Context::default().add_message(ContextMessage::user(
                format!("find automatic injection needle in {}", src_dir.display()),
                Some(model_id),
            )));

        ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let actual = setup.query_filters.lock().await.clone();
        let expected = vec![None, Some("src/".to_string())];
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_exact_file_path_does_not_emit_prefix_filter() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation =
            Conversation::generate().context(Context::default().add_message(ContextMessage::user(
                "find automatic injection needle in `src/lib.rs`",
                Some(model_id),
            )));

        ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let actual = setup.query_filters.lock().await.clone();
        let expected = vec![None];
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_ignores_unindexed_mentioned_path_and_keeps_cwd() -> Result<()> {
        let fixture = TempDir::new()?;
        let cwd = fixture.path().join("workspace");
        create_indexed_workspace(&cwd)?;
        let unindexed_file = fixture.path().join("unindexed/src/lib.rs");
        fs::create_dir_all(unindexed_file.parent().unwrap())?;
        fs::write(&unindexed_file, "pub fn automatic_injection_needle() {}")?;
        let setup = ProjectContextHarness::new(cwd.clone());
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation =
            Conversation::generate().context(Context::default().add_message(ContextMessage::user(
                format!(
                    "find automatic injection needle in {}",
                    unindexed_file.display()
                ),
                Some(model_id),
            )));

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let expected_workspaces = vec![cwd];
        assert_eq!(*setup.queried_workspaces.lock().await, expected_workspaces);
        assert_eq!(setup.workspace_queries.load(Ordering::SeqCst), 1usize);
        assert!(actual.context.unwrap().messages.iter().any(|message| {
            message
                .content()
                .is_some_and(|content| content.contains("<project_model_context"))
        }));
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_queries_cwd_baseline_without_mentioned_path() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root.clone());
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let expected_workspaces = vec![root];
        assert_eq!(*setup.queried_workspaces.lock().await, expected_workspaces);
        let expected_filters = vec![None];
        assert_eq!(*setup.query_filters.lock().await, expected_filters);
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_does_not_inject_stale_manifest() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new_with_stale_paths(root.clone(), vec![root]);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let expected_queries = 0usize;

        assert_eq!(
            setup.workspace_queries.load(Ordering::SeqCst),
            expected_queries
        );
        assert!(!actual.context.unwrap().messages.iter().any(|message| {
            message
                .content()
                .is_some_and(|content| content.contains("<project_model_context"))
        }));
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_nearest_stale_manifest_blocks_parent_fallback() -> Result<()> {
        let fixture = TempDir::new()?;
        let parent = fixture.path().join("workspace");
        create_indexed_workspace(&parent)?;
        let nested = parent.join("nested");
        create_indexed_workspace(&nested)?;
        let setup = ProjectContextHarness::new_with_stale_paths(nested.clone(), vec![nested]);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let expected_queries = 0usize;

        assert_eq!(
            setup.workspace_queries.load(Ordering::SeqCst),
            expected_queries
        );
        assert!(!actual.context.unwrap().messages.iter().any(|message| {
            message
                .content()
                .is_some_and(|content| content.contains("<project_model_context"))
        }));
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_nearest_unknown_manifest_blocks_parent_fallback() -> Result<()> {
        let fixture = TempDir::new()?;
        let parent = fixture.path().join("workspace");
        create_indexed_workspace(&parent)?;
        let nested = parent.join("nested");
        create_indexed_workspace(&nested)?;
        let setup = ProjectContextHarness::new_with_unknown_paths(nested.clone(), vec![nested]);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let expected_queries = 0usize;

        assert_eq!(
            setup.workspace_queries.load(Ordering::SeqCst),
            expected_queries
        );
        assert!(!actual.context.unwrap().messages.iter().any(|message| {
            message
                .content()
                .is_some_and(|content| content.contains("<project_model_context"))
        }));
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_reports_fresh_target_and_injection_decision() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root.clone());
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let expected = (root, true, 1usize, 1usize, None::<String>);

        assert_eq!(
            (
                actual.cwd,
                actual.would_inject,
                actual.candidates.len(),
                actual.selected_targets.len(),
                actual.skip_reason,
            ),
            expected
        );
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_reports_stale_manifest_skip_reason() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new_with_stale_paths(root.clone(), vec![root.clone()]);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let expected = (
            false,
            0usize,
            Some("no fresh project-model manifest target selected".to_string()),
        );

        assert_eq!(
            (
                actual.would_inject,
                actual.selected_targets.len(),
                actual.skip_reason,
            ),
            expected
        );
        assert!(actual.candidates.iter().any(|candidate| {
            candidate
                .skip_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("project-model manifest stale"))
        }));
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_nearest_stale_manifest_blocks_parent_fallback() -> Result<()> {
        let fixture = TempDir::new()?;
        let parent = fixture.path().join("parent-workspace");
        create_indexed_workspace(&parent)?;
        let child = parent.join("nested-child");
        create_indexed_workspace(&child)?;
        let setup = ProjectContextHarness::new_with_stale_paths(child.clone(), vec![child]);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let expected = (
            false,
            Vec::<PathBuf>::new(),
            Some("no fresh project-model manifest target selected".to_string()),
        );

        assert_eq!(
            (
                actual.would_inject,
                actual
                    .selected_targets
                    .iter()
                    .map(|target| target.workspace_root.clone())
                    .collect::<Vec<_>>(),
                actual.skip_reason,
            ),
            expected
        );
        assert!(actual.candidates.iter().any(|candidate| {
            candidate
                .skip_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("project-model manifest stale"))
        }));
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_nearest_unknown_manifest_blocks_parent_fallback() -> Result<()> {
        let fixture = TempDir::new()?;
        let parent = fixture.path().join("parent-workspace");
        create_indexed_workspace(&parent)?;
        let child = parent.join("nested-child");
        create_indexed_workspace(&child)?;
        let setup = ProjectContextHarness::new_with_unknown_paths(child.clone(), vec![child]);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let expected = (
            false,
            Vec::<PathBuf>::new(),
            Some("no fresh project-model manifest target selected".to_string()),
        );

        assert_eq!(
            (
                actual.would_inject,
                actual
                    .selected_targets
                    .iter()
                    .map(|target| target.workspace_root.clone())
                    .collect::<Vec<_>>(),
                actual.skip_reason,
            ),
            expected
        );
        assert!(actual.candidates.iter().any(|candidate| {
            candidate
                .skip_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("project-model manifest freshness unknown"))
        }));
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_reports_stale_manifest_details_for_candidate() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new_with_stale_paths(root.clone(), vec![root]);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let actual_reason = actual
            .candidates
            .iter()
            .find_map(|candidate| candidate.skip_reason.as_deref())
            .unwrap_or_default();

        assert!(
            actual_reason.contains("stale") && actual_reason.contains("src/lib.rs"),
            "explain-context must expose stale manifest details instead of a generic no-fresh-target reason; got {actual_reason:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_path_target_scan_is_bounded_for_unindexed_mentions() -> Result<()>
    {
        let fixture = TempDir::new()?;
        let cwd = fixture.path().join("workspace");
        create_indexed_workspace(&cwd)?;
        let setup = ProjectContextHarness::new(cwd.clone());
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let mentions = (0..64)
            .map(|index| {
                fixture
                    .path()
                    .join(format!("unindexed-{index}/src/lib.rs"))
                    .display()
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join(" ");
        let conversation =
            Conversation::generate().context(Context::default().add_message(ContextMessage::user(
                format!("find automatic injection needle in {mentions}"),
                Some(model_id),
            )));

        ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let actual = setup.index_checks.load(Ordering::SeqCst);
        let expected_maximum = ProjectContextInjection::<ProjectContextHarness>::MAX_INDEX_PROBES;
        assert!(
            actual <= expected_maximum,
            "path-aware injection should bound index checks for untrusted path-like prompt text; got {actual}, expected at most {expected_maximum}"
        );
        let expected_queries = 1usize;
        assert_eq!(
            setup.workspace_queries.load(Ordering::SeqCst),
            expected_queries
        );
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_continues_when_path_target_is_empty() -> Result<()> {
        let fixture = TempDir::new()?;
        let cwd = fixture.path().join("cwd-workspace");
        let other = fixture.path().join("other-workspace");
        create_indexed_workspace(&cwd)?;
        create_indexed_workspace(&other)?;
        let setup = ProjectContextHarness::new_with_empty_paths(cwd.clone(), vec![other.clone()]);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let mentioned_file = other.join("src/lib.rs");
        let conversation =
            Conversation::generate().context(Context::default().add_message(ContextMessage::user(
                format!(
                    "find automatic injection needle in {}",
                    mentioned_file.display()
                ),
                Some(model_id),
            )));

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let expected_workspaces = vec![cwd, other];
        assert_eq!(*setup.queried_workspaces.lock().await, expected_workspaces);
        assert_eq!(
            actual
                .context
                .unwrap()
                .messages
                .iter()
                .filter(|message| message
                    .content()
                    .is_some_and(|content| content.contains("<project_model_context")))
                .count(),
            1usize,
        );
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_continues_when_path_target_query_errors() -> Result<()> {
        let fixture = TempDir::new()?;
        let cwd = fixture.path().join("cwd-workspace");
        let other = fixture.path().join("other-workspace");
        create_indexed_workspace(&cwd)?;
        create_indexed_workspace(&other)?;
        let setup = ProjectContextHarness::new_with_error_paths(cwd.clone(), vec![other.clone()]);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let mentioned_file = other.join("src/lib.rs");
        let conversation =
            Conversation::generate().context(Context::default().add_message(ContextMessage::user(
                format!(
                    "find automatic injection needle in {}",
                    mentioned_file.display()
                ),
                Some(model_id),
            )));

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let expected_workspaces = vec![cwd, other];
        assert_eq!(*setup.queried_workspaces.lock().await, expected_workspaces);
        assert_eq!(
            actual
                .context
                .unwrap()
                .messages
                .iter()
                .filter(|message| message
                    .content()
                    .is_some_and(|content| content.contains("<project_model_context")))
                .count(),
            1usize,
        );
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_is_injected_into_provider_request_without_sem_search_tool_call()
    -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root.clone());
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone())
            .tool_supported(false)
            .max_requests_per_turn(1usize);
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id.clone())),
        ));
        let conversation = ProjectContextInjection::new(setup.clone(), agent.clone())
            .inject(conversation)
            .await;
        let mut orch =
            Orchestrator::new(setup.clone(), conversation, agent, ForgeConfig::default())
                .models(vec![
                    Model::new(ProviderId::OPENAI, model_id).context_length(200_000_u64),
                ])
                .tool_definitions(Vec::new());

        orch.run().await?;
        let captured_context = setup.captured_context.lock().await.clone().unwrap();
        let project_context_message = captured_context
            .messages
            .iter()
            .find(|message| {
                message
                    .content()
                    .is_some_and(|content| content.contains("<project_model_context"))
            })
            .unwrap();
        let actual = project_context_message.content().unwrap().to_string();
        let actual_flags = vec![
            actual.contains("manifest_path"),
            actual.contains("src/lib.rs"),
            actual.contains("start_line=\"3\""),
            actual.contains("score=\"0.875000\""),
            actual.contains("content_digest=\""),
            actual.contains("truncated_reason=\"content_line_budget_exceeded\"")
                || actual.contains("truncated_reason=\"content_char_budget_exceeded\""),
            actual.contains("omitted_reason=\"whole_file_metadata_only\""),
            actual.contains("full_file_should_not_render"),
            actual.contains("extra_should_not_render"),
            project_context_message.is_cache_eligible(),
            captured_context
                .tools
                .iter()
                .any(|tool| tool.name.as_str().eq_ignore_ascii_case("sem_search")),
        ];
        let expected_flags = vec![
            true, true, true, true, true, true, true, false, false, false, false,
        ];
        assert_eq!(actual_flags, expected_flags);
        assert_eq!(actual.matches("<source").count(), 3usize);
        assert_eq!(setup.workspace_queries.load(Ordering::SeqCst), 1usize);
        assert!(
            actual.chars().count() <= ProjectModelContextRenderBudget::default().max_rendered_chars,
            "project-model context should stay inside the typed render budget"
        );
        Ok(())
    }

    #[tokio::test]
    async fn project_model_query_ignores_droppable_user_messages() -> Result<()> {
        let (_fixture, _root) = fixture_workspace()?;
        let model_id = ModelId::new("test-model");
        let conversation = Conversation::generate().context(
            Context::default()
                .add_message(ContextMessage::user(
                    "find automatic injection needle",
                    Some(model_id.clone()),
                ))
                .add_message(ContextMessage::Text(
                    TextMessage::new(Role::User, "ignore automatic injection needle")
                        .model(model_id)
                        .droppable(true),
                )),
        );

        let actual = ProjectContextInjection::<ProjectContextHarness>::query_from_conversation(
            &conversation,
        )
        .unwrap();
        let expected = "find automatic injection needle";
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn project_model_query_ignores_internal_project_model_context_messages() -> Result<()> {
        let (_fixture, _root) = fixture_workspace()?;
        let model_id = ModelId::new("test-model");
        let conversation = Conversation::generate().context(
            Context::default()
                .add_message(ContextMessage::user(
                    "find automatic injection needle",
                    Some(model_id.clone()),
                ))
                .add_message(ContextMessage::Text(
                    TextMessage::project_model_context(
                        Role::User,
                        "<project_model_context>ignore automatic injection needle</project_model_context>",
                    )
                    .model(model_id)
                    .droppable(false),
                )),
        );

        let actual = ProjectContextInjection::<ProjectContextHarness>::query_from_conversation(
            &conversation,
        )
        .unwrap();
        let expected = "find automatic injection needle";
        assert_eq!(actual, expected);
        Ok(())
    }
}
