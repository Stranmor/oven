use std::sync::Arc;

use anyhow::Result;
use chrono::Local;
use forge_config::ForgeConfig;
use forge_domain::*;
use forge_project_model::{
    ProjectModelContextRenderBudget, ProjectModelContextSource, render_project_model_context,
};
use forge_stream::MpscStream;

use crate::apply_tunable_parameters::ApplyTunableParameters;
use crate::changed_files::ChangedFiles;
use crate::dto::ToolsOverview;
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

impl<S: EnvironmentInfra<Config = forge_config::ForgeConfig> + WorkspaceService>
    ProjectContextInjection<S>
{
    fn new(services: Arc<S>, agent: Agent) -> Self {
        Self { services, agent }
    }

    async fn inject(&self, mut conversation: Conversation) -> Conversation {
        let cwd = self.services.get_environment().cwd;
        let is_indexed = match self.services.is_indexed(&cwd).await {
            Ok(indexed) => indexed,
            Err(error) => {
                tracing::debug!(error = ?error, path = %cwd.display(), "Skipping project-model context injection because index availability could not be checked");
                return conversation;
            }
        };
        if !is_indexed {
            return conversation;
        }

        let Some(query) = Self::query_from_conversation(&conversation) else {
            return conversation;
        };
        let params = SearchParams::new(&query, "automatic project-model context injection")
            .limit(ProjectModelContextRenderBudget::default().max_sources);
        let nodes = match self.services.query_workspace(cwd.clone(), params).await {
            Ok(nodes) => nodes,
            Err(error) => {
                tracing::debug!(error = ?error, path = %cwd.display(), "Skipping project-model context injection because local retrieval failed");
                return conversation;
            }
        };
        if nodes.is_empty() {
            return conversation;
        }

        let content = Self::render_context(&cwd, nodes);
        let mut context = conversation.context.take().unwrap_or_default();
        let message = TextMessage::new(Role::User, content)
            .model(self.agent.model.clone())
            .droppable(true)
            .cacheable(false);
        context = context.add_message(ContextMessage::Text(message));
        conversation.context(context)
    }

    fn query_from_conversation(conversation: &Conversation) -> Option<String> {
        conversation
            .context
            .as_ref()?
            .messages
            .iter()
            .rev()
            .find(|message| message.has_role(Role::User) && !message.is_droppable())
            .and_then(|message| message.content())
            .map(str::trim)
            .filter(|content| !content.is_empty())
            .map(ToOwned::to_owned)
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

        // Calculate original metrics
        let original_messages = context.messages.len();
        let original_token_count = *context.token_count();

        let forge_config = self.services.get_config()?;

        // Get agent and apply workflow config
        let agent = self.services.get_agent(&active_agent_id).await?;

        let Some(agent) = agent else {
            return Err(crate::Error::AgentNotFound(active_agent_id).into());
        };

        // Get compact config from the agent
        let compact = agent
            .apply_config(&forge_config)
            .set_compact_model_if_none()
            .compact;

        // Apply compaction using the Compactor
        let environment = self.services.get_environment();
        let compacted_context = Compactor::new(compact, environment).compact(context, true)?;

        let compacted_messages = compacted_context.messages.len();
        let compacted_tokens = *compacted_context.token_count();

        // Update the conversation with the compacted context
        conversation.context = Some(compacted_context);

        // Save the updated conversation
        self.services.upsert_conversation(conversation).await?;

        Ok(CompactionResult::new(
            original_token_count,
            compacted_tokens,
            original_messages,
            compacted_messages,
        ))
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
        ToolCallFull, ToolResult, WorkspaceAuth, WorkspaceId, WorkspaceInfo,
    };
    use pretty_assertions::assert_eq;
    use tempfile::TempDir;
    use tokio::sync::Mutex;

    use super::*;
    use crate::agent::AgentService;
    use crate::orch::Orchestrator;

    struct ProjectContextHarness {
        cwd: PathBuf,
        captured_context: Mutex<Option<Context>>,
        workspace_queries: AtomicUsize,
    }

    impl ProjectContextHarness {
        fn new(cwd: PathBuf) -> Arc<Self> {
            Arc::new(Self {
                cwd,
                captured_context: Mutex::new(None),
                workspace_queries: AtomicUsize::new(0),
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
            _path: PathBuf,
            params: SearchParams<'_>,
        ) -> Result<Vec<Node>> {
            self.workspace_queries.fetch_add(1, Ordering::SeqCst);
            assert!(params.query.contains("automatic injection needle"));
            assert_eq!(params.limit, Some(3));
            let long_content = (0..40)
                .map(|index| format!("pub fn long_{index}() -> usize {{ {index} }}"))
                .collect::<Vec<_>>()
                .join("\n");
            Ok(vec![
                Node {
                    node_id: NodeId::new("symbol:src/lib.rs:automatic_injection_needle"),
                    node: NodeData::FileChunk(FileChunk {
                        file_path: "src/lib.rs".to_string(),
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

        async fn delete_workspace(&self, _workspace_id: &WorkspaceId) -> Result<()> {
            anyhow::bail!("unused workspace delete")
        }

        async fn delete_workspaces(&self, _workspace_ids: &[WorkspaceId]) -> Result<()> {
            anyhow::bail!("unused workspace deletes")
        }

        async fn is_indexed(&self, path: &Path) -> Result<bool> {
            Ok(path
                .join(".forge_project_model/project_manifest.json")
                .is_file())
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
        Ok((fixture, root))
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
                .models(vec![Model::new(ProviderId::OPENAI, model_id)])
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
}
