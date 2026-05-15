use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use chrono::Local;
use forge_config::ForgeConfig;
use forge_domain::*;
use forge_project_model::{
    ProjectModelContextRenderBudget, ProjectModelContextSource, render_project_model_context,
};
use forge_stream::MpscStream;
use url::Url;

use crate::apply_tunable_parameters::ApplyTunableParameters;
use crate::changed_files::ChangedFiles;
use crate::dto::ToolsOverview;
use crate::dto::openai::ProviderRequestEstimate as OpenAiProviderRequestEstimate;
use crate::hooks::{
    CompactionHandler, DoomLoopDetector, PendingTodosHandler, TitleGenerationHandler,
    TracingHandler,
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
    AgentExt, AgentProviderResolver, ConversationService, EnvironmentInfra, ProviderService,
    Services, WorkspaceService,
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ProjectContextTarget {
    workspace_root: PathBuf,
    path_filter: Option<String>,
}

#[derive(Debug, Clone)]
struct TargetResolutionBudget {
    remaining_candidates: usize,
    remaining_index_probes: usize,
}

impl TargetResolutionBudget {
    fn new(explicit_target_candidates: usize, index_probes: usize) -> Self {
        Self {
            remaining_candidates: explicit_target_candidates.saturating_add(1),
            remaining_index_probes: index_probes,
        }
    }

    fn claim_candidate(&mut self) -> bool {
        let Some(remaining) = self.remaining_candidates.checked_sub(1) else {
            return false;
        };
        self.remaining_candidates = remaining;
        true
    }

    fn claim_index_probe(&mut self) -> bool {
        let Some(remaining) = self.remaining_index_probes.checked_sub(1) else {
            return false;
        };
        self.remaining_index_probes = remaining;
        true
    }
}

impl<S: EnvironmentInfra<Config = forge_config::ForgeConfig> + WorkspaceService>
    ProjectContextInjection<S>
{
    const MAX_TARGETS: usize = 4;
    const MAX_EXPLICIT_TARGET_CANDIDATES: usize = 8;
    const MAX_INDEX_PROBES: usize = 32;

    fn new(services: Arc<S>, agent: Agent) -> Self {
        Self { services, agent }
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
            let message = TextMessage::new(Role::User, content)
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
            candidates.extend(Self::mentioned_paths(
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
            let path_filter = Self::directory_path_filter(&path, &workspace_root);
            let target = ProjectContextTarget {
                workspace_root: workspace_root.clone(),
                path_filter: path_filter.clone(),
            };
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
        candidates.extend(Self::mentioned_paths(
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
            let path_filter = Self::directory_path_filter(&path, &workspace_root);
            return Some(ProjectContextTarget { workspace_root, path_filter });
        }
        None
    }

    fn directory_path_filter(path: &Path, workspace_root: &Path) -> Option<String> {
        let relative = path
            .strip_prefix(workspace_root)
            .ok()
            .filter(|relative| !relative.as_os_str().is_empty())?;
        if !path.is_dir() {
            return None;
        }
        let mut filter = relative.to_string_lossy().replace('\\', "/");
        if !filter.ends_with('/') {
            filter.push('/');
        }
        Some(filter)
    }

    fn mentioned_paths(message: &str, cwd: &Path, home: Option<&Path>) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        let mut seen = BTreeSet::new();
        for tag in Attachment::parse_all(message) {
            if let Some(path) = Self::resolve_mentioned_path(&tag.path, cwd, home) {
                if seen.insert(path.clone()) {
                    paths.push(path);
                }
            }
        }
        for token in Self::path_like_tokens(message) {
            if let Some(path) = Self::resolve_mentioned_path(&token, cwd, home) {
                if seen.insert(path.clone()) {
                    paths.push(path);
                }
            }
        }
        paths
    }

    fn path_like_tokens(message: &str) -> Vec<String> {
        message
            .split_whitespace()
            .filter_map(Self::normalize_path_token)
            .collect()
    }

    fn normalize_path_token(raw: &str) -> Option<String> {
        let token = raw.trim_matches(|character: char| {
            matches!(
                character,
                '`' | '\'' | '"' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';'
            )
        });
        let token = token.trim_end_matches(['.', ':', '!']);
        if token.is_empty()
            || token.contains("://")
            || token.starts_with('<')
            || token.ends_with('>')
            || token.starts_with("@[")
        {
            return None;
        }
        let token = Self::trim_line_suffix(token);
        if token.starts_with('/')
            || token.starts_with('~')
            || token.starts_with("./")
            || token.starts_with("../")
            || Self::is_relative_path_token(token)
        {
            Some(token.to_string())
        } else {
            None
        }
    }

    fn is_relative_path_token(token: &str) -> bool {
        if !token.contains('/') {
            return false;
        }
        let components = token
            .split('/')
            .filter(|component| !component.is_empty())
            .collect::<Vec<_>>();
        components.len() > 2
            || components
                .last()
                .is_some_and(|component| component.contains('.'))
    }

    fn trim_line_suffix(token: &str) -> &str {
        let Some((prefix, suffix)) = token.rsplit_once(':') else {
            return token;
        };
        if suffix.chars().all(|character| character.is_ascii_digit()) {
            Self::trim_line_suffix(prefix)
        } else {
            token
        }
    }

    fn resolve_mentioned_path(raw_path: &str, cwd: &Path, home: Option<&Path>) -> Option<PathBuf> {
        if raw_path.is_empty() || raw_path.contains("://") {
            return None;
        }
        let path = if raw_path == "~" {
            home?.to_path_buf()
        } else if let Some(stripped) = raw_path.strip_prefix("~/") {
            home?.join(stripped)
        } else {
            let path = PathBuf::from(raw_path);
            if path.is_absolute() {
                path
            } else {
                cwd.join(path)
            }
        };
        Some(path)
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
                        ContextMessage::Text(text) if text.is_runtime_context()
                    )
            })
            .and_then(|message| message.content())
            .map(str::trim)
            .filter(|content| !content.is_empty())
    }

    fn render_context(workspace_root: &std::path::Path, nodes: Vec<Node>) -> String {
        let manifest_path = workspace_root.join(".forge_project_model/project_manifest.json");
        let sources = nodes
            .into_iter()
            .map(Self::source_from_node)
            .collect::<Vec<_>>();
        render_project_model_context(
            &workspace_root.display().to_string(),
            &manifest_path.display().to_string(),
            "local_manifest_available",
            "WorkspaceService::query_workspace",
            &sources,
            &ProjectModelContextRenderBudget::default(),
        )
    }

    fn source_from_node(node: Node) -> ProjectModelContextSource {
        let node_id = node.node_id.as_str().to_string();
        let score = node.relevance;
        match node.node {
            NodeData::FileChunk(chunk) => ProjectModelContextSource::new(
                chunk.file_path,
                "manifest_snapshot",
                "local_project_model_manifest",
                node_id,
            )
            .line_range(chunk.start_line, chunk.end_line)
            .score(score)
            .content(chunk.content),
            NodeData::File(file) => ProjectModelContextSource::new(
                file.file_path,
                "manifest_snapshot",
                "local_project_model_manifest",
                node_id,
            )
            .score(score)
            .content_hash(file.hash)
            .content(file.content)
            .metadata_only("whole_file_metadata_only"),
            NodeData::FileRef(file_ref) => ProjectModelContextSource::new(
                file_ref.file_path,
                "manifest_snapshot",
                "local_project_model_manifest",
                node_id,
            )
            .score(score)
            .content_hash(file_ref.file_hash)
            .metadata_only("file_reference_metadata_only"),
            NodeData::Note(note) => ProjectModelContextSource::new(
                "note",
                "manifest_snapshot",
                "local_project_model_manifest",
                node_id,
            )
            .score(score)
            .content(note.content),
            NodeData::Task(task) => ProjectModelContextSource::new(
                "task",
                "manifest_snapshot",
                "local_project_model_manifest",
                node_id,
            )
            .score(score)
            .content(task.task),
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
        let forge_config = self.services.get_config()?;
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
            .apply_config(&forge_config)
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
                .max_extensions(forge_config.max_extensions)
                .template_config(build_template_config(&forge_config))
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
        let on_end_hook = if forge_config.verify_todos {
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
                    let save_result = services.upsert_conversation(conversation).await;

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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use anyhow::Result;
    use forge_domain::{
        Agent, AgentId, ChatCompletionMessage, Content, Context, ContextMessage, Conversation,
        Environment, FileChunk, FileStatus, FinishReason, Model, ModelId, Node, NodeData, NodeId,
        ProviderId, ResultStream, SearchParams, SteerMessage, SyncProgress, ToolCallContext,
        ToolCallFull, ToolResult, WorkspaceAuth, WorkspaceContextFreshness,
        WorkspaceContextManifestDiagnostic, WorkspaceId, WorkspaceInfo,
    };
    use pretty_assertions::assert_eq;
    use tempfile::TempDir;
    use tokio::sync::Mutex;

    use super::*;
    use crate::agent::AgentService;
    use crate::orch::Orchestrator;

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
            })
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
}
