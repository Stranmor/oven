use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use forge_app::dto::ToolsOverview;
use forge_app::{
    AgentProviderResolver, AgentRegistry, AppConfigService, AuthService, CommandInfra,
    CommandLoaderService, ConversationService, DataGenerationApp, EnvironmentInfra,
    FileDiscoveryService, ForgeApp, GitApp, GrpcInfra, McpConfigManager, McpService,
    ProviderAuthService, ProviderService, Services, User, UserUsage, Walker, WorkspaceService,
};
use forge_config::ForgeConfig;
use forge_domain::{Agent, ConsoleWriter, *};
use forge_infra::ForgeInfra;
use forge_repo::ForgeRepo;
use forge_services::ForgeServices;
use forge_stream::MpscStream;
use futures::stream::BoxStream;
use url::Url;

use crate::API;

pub struct ForgeAPI<S, F> {
    services: Arc<S>,
    infra: Arc<F>,
}

impl<A, F> ForgeAPI<A, F> {
    pub fn new(services: Arc<A>, infra: Arc<F>) -> Self {
        Self { services, infra }
    }

    /// Creates a ForgeApp instance with the current services and latest config.
    fn app(&self) -> ForgeApp<A>
    where
        A: Services + EnvironmentInfra<Config = forge_config::ForgeConfig>,
        F: EnvironmentInfra<Config = forge_config::ForgeConfig>,
    {
        ForgeApp::new(self.services.clone())
    }
}

impl ForgeAPI<ForgeServices<ForgeRepo<ForgeInfra>>, ForgeRepo<ForgeInfra>> {
    /// Creates a fully-initialized [`ForgeAPI`] from a pre-read configuration.
    ///
    /// # Arguments
    /// * `cwd` - The working directory path for environment and file resolution
    /// * `config` - Pre-read application configuration (from startup)
    /// * `services_url` - Pre-validated URL for the gRPC workspace server
    pub fn init(cwd: PathBuf, config: ForgeConfig) -> anyhow::Result<Self> {
        let infra = Arc::new(ForgeInfra::new(cwd, config));
        let repo = Arc::new(ForgeRepo::new(infra.clone())?);
        let app = Arc::new(ForgeServices::new(repo.clone()));
        Ok(ForgeAPI::new(app, repo))
    }

    pub async fn get_skills_internal(&self) -> Result<Vec<Skill>> {
        use forge_domain::SkillRepository;
        self.infra.load_skills().await
    }
}

impl<A, F> ForgeAPI<A, F>
where
    F: CommandInfra,
{
    async fn execute_shell_command_with_handoff_timeout(
        &self,
        command: &str,
        working_dir: PathBuf,
        handoff_timeout: ShellHandoffTimeoutSeconds,
    ) -> anyhow::Result<CommandExecutionOutput> {
        self.infra
            .execute_command(
                command.to_string(),
                working_dir,
                false,
                None,
                handoff_timeout,
            )
            .await
    }
}

#[async_trait::async_trait]
impl<
    A: Services + EnvironmentInfra<Config = forge_config::ForgeConfig>,
    F: CommandInfra
        + EnvironmentInfra<Config = forge_config::ForgeConfig>
        + SkillRepository
        + GrpcInfra,
> API for ForgeAPI<A, F>
{
    async fn discover(&self) -> Result<Vec<File>> {
        let environment = self.services.get_environment();
        let config = Walker::unlimited().cwd(environment.cwd);
        self.services.collect_files(config).await
    }

    async fn get_tools(&self) -> anyhow::Result<ToolsOverview> {
        self.app().list_tools().await
    }

    async fn get_models(&self) -> Result<Vec<Model>> {
        self.app().get_models().await
    }

    async fn get_all_provider_models(&self) -> Result<Vec<ProviderModels>> {
        self.app().get_all_provider_models().await
    }

    async fn get_agents(&self) -> Result<Vec<Agent>> {
        self.services.get_agents().await
    }

    async fn get_agent_infos(&self) -> Result<Vec<AgentInfo>> {
        self.services.get_agent_infos().await
    }

    async fn get_providers(&self) -> Result<Vec<AnyProvider>> {
        Ok(self.services.get_all_providers().await?)
    }

    #[tracing::instrument(skip(self, diff, additional_context))]
    async fn commit(
        &self,
        preview: bool,
        max_diff_size: Option<usize>,
        diff: Option<String>,
        additional_context: Option<String>,
    ) -> Result<forge_app::CommitResult> {
        let use_forge_committer = self
            .services
            .get_config()
            .context("Failed to read forge config for commit settings")?
            .use_forge_committer;

        let git_app = GitApp::new(self.services.clone());
        let result = git_app
            .commit_message(max_diff_size, diff, additional_context)
            .await?;

        if preview {
            Ok(result)
        } else {
            git_app
                .commit(result.message, result.has_staged_files, use_forge_committer)
                .await
        }
    }

    async fn get_provider(&self, id: &ProviderId) -> Result<AnyProvider> {
        let providers = self.services.get_all_providers().await?;
        Ok(providers
            .into_iter()
            .find(|p| p.id() == *id)
            .ok_or_else(|| Error::provider_not_available(id.clone()))?)
    }

    async fn chat(
        &self,
        chat: ChatRequest,
    ) -> anyhow::Result<MpscStream<Result<ChatResponse, anyhow::Error>>> {
        let agent_id = self
            .services
            .get_active_agent_id()
            .await?
            .ok_or_else(|| anyhow::anyhow!("No active agent configured"))?;
        self.app().chat(agent_id, chat).await
    }

    async fn steer(&self, request: SteerRequest) -> anyhow::Result<()> {
        self.app().steer(request).await
    }

    async fn upsert_conversation(&self, conversation: Conversation) -> anyhow::Result<()> {
        self.services.upsert_conversation(conversation).await
    }

    async fn compact_conversation(
        &self,
        conversation_id: &ConversationId,
    ) -> anyhow::Result<CompactionResult> {
        let agent_id = self
            .services
            .get_active_agent_id()
            .await?
            .ok_or_else(|| anyhow::anyhow!("No active agent configured"))?;
        self.app()
            .compact_conversation(agent_id, conversation_id)
            .await
    }

    fn environment(&self) -> Environment {
        self.services.get_environment().clone()
    }

    async fn conversation(
        &self,
        conversation_id: &ConversationId,
    ) -> anyhow::Result<Option<Conversation>> {
        self.services.find_conversation(conversation_id).await
    }

    async fn list_branch_targets(
        &self,
        conversation_id: &ConversationId,
    ) -> anyhow::Result<Vec<forge_app::dto::ConversationBranchTarget>> {
        self.services.list_branch_targets(conversation_id).await
    }

    async fn branch_conversation(
        &self,
        conversation_id: &ConversationId,
        target_id: forge_domain::MessageId,
    ) -> anyhow::Result<Conversation> {
        self.services
            .branch_conversation(conversation_id, target_id)
            .await
    }

    async fn get_conversation_list_items_by_query(
        &self,
        mut query: ConversationListQuery,
    ) -> anyhow::Result<Vec<ConversationListItem>> {
        query.limit = self.services.get_config()?.max_conversations;
        self.services
            .get_conversation_list_items_by_query(query)
            .await
    }

    async fn get_conversation_list_items_including_agent(
        &self,
    ) -> anyhow::Result<Vec<ConversationListItem>> {
        self.services
            .get_conversation_list_items_including_agent(
                self.services.get_config()?.max_conversations,
            )
            .await
    }

    async fn get_conversation_list_items_by_visibility(
        &self,
        visibility: forge_domain::ConversationVisibilityFilter,
    ) -> anyhow::Result<Vec<ConversationListItem>> {
        self.services
            .get_conversation_list_items_by_visibility(
                visibility,
                self.services.get_config()?.max_conversations,
            )
            .await
    }

    async fn get_conversations(&self) -> anyhow::Result<Vec<Conversation>> {
        let mut conversations = self.services.get_conversations().await?;
        conversations.truncate(self.services.get_config()?.max_conversations);
        Ok(conversations)
    }

    async fn get_conversations_including_agent(&self) -> anyhow::Result<Vec<Conversation>> {
        self.services.get_conversations_including_agent().await
    }

    async fn get_conversations_by_visibility(
        &self,
        visibility: forge_domain::ConversationVisibilityFilter,
    ) -> anyhow::Result<Vec<Conversation>> {
        self.services
            .get_conversations_by_visibility(visibility)
            .await
    }

    async fn get_sub_conversations(
        &self,
        parent_id: &ConversationId,
    ) -> anyhow::Result<Vec<Conversation>> {
        Ok(self.services.get_sub_conversations(parent_id).await?)
    }

    async fn list_subagent_task_sessions(
        &self,
        filter: forge_domain::SubagentTaskSessionFilter,
    ) -> anyhow::Result<Vec<forge_domain::SubagentTaskSession>> {
        self.services.list_subagent_task_sessions(filter).await
    }

    async fn subagent_task_session(
        &self,
        task_id: &forge_domain::SubagentTaskId,
    ) -> anyhow::Result<Option<forge_domain::SubagentTaskSession>> {
        self.services.get_subagent_task_session(task_id).await
    }

    async fn last_conversation(&self) -> anyhow::Result<Option<Conversation>> {
        self.services.last_conversation().await
    }

    async fn delete_conversation(&self, conversation_id: &ConversationId) -> anyhow::Result<()> {
        self.services.delete_conversation(conversation_id).await
    }

    async fn rename_conversation(
        &self,
        conversation_id: &ConversationId,
        title: String,
    ) -> anyhow::Result<()> {
        self.services
            .modify_conversation(conversation_id, |conv| {
                conv.title = Some(title);
            })
            .await
    }

    #[tracing::instrument(skip(self))]
    async fn execute_shell_command(
        &self,
        command: &str,
        working_dir: PathBuf,
    ) -> anyhow::Result<CommandExecutionOutput> {
        self.execute_shell_command_with_handoff_timeout(command, working_dir, Default::default())
            .await
    }
    async fn read_mcp_config(&self, scope: Option<&Scope>) -> Result<McpConfig> {
        self.services
            .read_mcp_config(scope)
            .await
            .map_err(|e| anyhow::anyhow!(e))
    }

    async fn write_mcp_config(&self, scope: &Scope, config: &McpConfig) -> Result<()> {
        self.services
            .write_mcp_config(config, scope)
            .await
            .map_err(|e| anyhow::anyhow!(e))
    }

    async fn execute_shell_command_raw(
        &self,
        command: &str,
    ) -> anyhow::Result<std::process::ExitStatus> {
        let cwd = self.environment().cwd;
        self.infra.execute_command_raw(command, cwd, None).await
    }

    async fn get_agent_provider(&self, agent_id: AgentId) -> anyhow::Result<Provider<Url>> {
        let agent_provider_resolver = AgentProviderResolver::new(self.services.clone());
        agent_provider_resolver.get_provider(Some(agent_id)).await
    }

    #[tracing::instrument(skip(self))]
    async fn update_config(&self, ops: Vec<forge_domain::ConfigOperation>) -> anyhow::Result<()> {
        let needs_agent_reload = ops
            .iter()
            .any(|op| matches!(op, forge_domain::ConfigOperation::SetSessionConfig(_)));

        self.services.update_config(ops).await?;

        if needs_agent_reload {
            self.services.reload_agents().await?;
        }

        Ok(())
    }

    async fn get_commit_config(&self) -> anyhow::Result<Option<ModelConfig>> {
        self.services.get_commit_config().await
    }

    async fn get_suggest_config(&self) -> anyhow::Result<Option<ModelConfig>> {
        self.services.get_suggest_config().await
    }

    async fn get_reasoning_effort(&self) -> anyhow::Result<Option<Effort>> {
        self.services.get_reasoning_effort().await
    }

    async fn user_info(&self) -> Result<Option<User>> {
        let provider = self.get_default_provider().await?;
        if let Some(api_key) = provider.api_key() {
            let user_info = self.services.user_info(api_key.as_str()).await?;
            return Ok(Some(user_info));
        }
        Ok(None)
    }

    async fn user_usage(&self) -> Result<Option<UserUsage>> {
        let provider = self.get_default_provider().await?;
        if let Some(api_key) = provider
            .credential
            .as_ref()
            .and_then(|c| match &c.auth_details {
                forge_domain::AuthDetails::ApiKey(key) => Some(key.as_str()),
                _ => None,
            })
        {
            let user_usage = self.services.user_usage(api_key).await?;
            return Ok(Some(user_usage));
        }
        Ok(None)
    }

    async fn get_active_agent(&self) -> Option<AgentId> {
        self.services.get_active_agent_id().await.ok().flatten()
    }

    async fn set_active_agent(&self, agent_id: AgentId) -> anyhow::Result<()> {
        self.services.set_active_agent_id(agent_id).await
    }

    async fn get_agent_model(&self, agent_id: AgentId) -> Option<ModelId> {
        let agent_provider_resolver = AgentProviderResolver::new(self.services.clone());
        match agent_provider_resolver
            .get_model(Some(agent_id.clone()))
            .await
        {
            Ok(model_id) => Some(model_id),
            Err(error) => {
                tracing::warn!(%agent_id, %error, "failed to resolve agent model");
                None
            }
        }
    }

    async fn reload_mcp(&self) -> Result<()> {
        self.services.mcp_service().reload_mcp().await
    }
    async fn get_commands(&self) -> Result<Vec<Command>> {
        self.services.get_commands().await
    }

    async fn get_skills(&self) -> Result<Vec<Skill>> {
        self.infra.load_skills().await
    }
    async fn generate_command(&self, prompt: UserPrompt) -> Result<String> {
        use forge_app::CommandGenerator;
        let generator = CommandGenerator::new(self.services.clone());
        generator.generate(prompt).await
    }

    async fn init_provider_auth(
        &self,
        provider_id: ProviderId,
        method: AuthMethod,
    ) -> Result<AuthContextRequest> {
        Ok(self
            .services
            .init_provider_auth(provider_id, method)
            .await?)
    }

    async fn complete_provider_auth(
        &self,
        provider_id: ProviderId,
        context: AuthContextResponse,
        timeout: Duration,
    ) -> Result<()> {
        Ok(self
            .services
            .complete_provider_auth(provider_id, context, timeout)
            .await?)
    }

    async fn remove_provider(&self, provider_id: &ProviderId) -> Result<()> {
        self.services.remove_credential(provider_id).await
    }

    async fn sync_workspace(
        &self,
        path: PathBuf,
    ) -> Result<MpscStream<Result<forge_domain::SyncProgress>>> {
        self.services.sync_workspace(path).await
    }

    async fn produce_workspace_exact_fact_reference(
        &self,
        path: PathBuf,
    ) -> Result<forge_domain::WorkspaceExactFactReferenceReport> {
        self.services
            .produce_workspace_exact_fact_reference(path)
            .await
    }

    async fn workspace_exact_fact_status(
        &self,
        path: PathBuf,
    ) -> Result<forge_domain::WorkspaceExactFactStatusReport> {
        self.services.workspace_exact_fact_status(path).await
    }

    async fn build_workspace_vector_index(
        &self,
        path: PathBuf,
        embedding_model_id: String,
    ) -> Result<forge_domain::WorkspaceVectorIndexBuildReport> {
        self.services
            .build_workspace_vector_index(path, embedding_model_id)
            .await
    }

    async fn embed_workspace_query(
        &self,
        query: String,
        embedding_model_id: String,
    ) -> Result<forge_domain::ProjectSemanticEmbeddingOutput> {
        self.services
            .embed_workspace_query(query, embedding_model_id)
            .await
    }

    async fn sem_search_diagnostic(
        &self,
        path: PathBuf,
    ) -> Result<forge_domain::SemSearchDiagnosticReport> {
        let embedding_model_id = self
            .services
            .get_config()
            .context("Failed to read forge config for sem_search diagnostic")?
            .semantic_embedding_model_id;
        self.services
            .sem_search_diagnostic(path, embedding_model_id)
            .await
    }

    async fn query_workspace(
        &self,
        path: PathBuf,
        params: forge_domain::SearchParams<'_>,
    ) -> Result<Vec<forge_domain::Node>> {
        let (_committed_result, nodes) = self
            .services
            .query_workspace_committed(path, params)
            .await?;
        Ok(nodes)
    }

    async fn list_workspaces(&self) -> Result<Vec<forge_domain::WorkspaceInfo>> {
        self.services.list_workspaces().await
    }

    async fn get_workspace_info(
        &self,
        path: PathBuf,
    ) -> Result<Option<forge_domain::WorkspaceInfo>> {
        self.services.get_workspace_info(path).await
    }

    async fn delete_workspaces(&self, workspace_ids: Vec<forge_domain::WorkspaceId>) -> Result<()> {
        self.services.delete_workspaces(&workspace_ids).await
    }

    async fn explain_workspace_context(
        &self,
        query: Option<String>,
    ) -> Result<forge_domain::WorkspaceContextExplanation> {
        Ok(self.app().explain_workspace_context(query).await)
    }

    async fn get_workspace_status(&self, path: PathBuf) -> Result<Vec<forge_domain::FileStatus>> {
        self.services.get_workspace_status(path).await
    }

    async fn is_authenticated(&self) -> Result<bool> {
        self.services.is_authenticated().await
    }

    async fn create_auth_credentials(&self) -> Result<forge_domain::WorkspaceAuth> {
        self.services.init_auth_credentials().await
    }

    async fn init_workspace(&self, path: PathBuf) -> Result<forge_domain::WorkspaceId> {
        self.services.init_workspace(path).await
    }

    async fn migrate_env_credentials(&self) -> Result<Option<forge_domain::MigrationResult>> {
        Ok(self.services.migrate_env_credentials().await?)
    }

    async fn generate_data(
        &self,
        data_parameters: DataGenerationParameters,
    ) -> Result<BoxStream<'static, Result<serde_json::Value, anyhow::Error>>> {
        let app = DataGenerationApp::new(self.services.clone());
        app.execute(data_parameters).await
    }

    async fn get_session_config(&self) -> Option<forge_domain::ModelConfig> {
        self.services.get_session_config().await
    }

    async fn get_default_provider(&self) -> Result<Provider<Url>> {
        let model_config = self
            .services
            .get_session_config()
            .await
            .ok_or_else(|| forge_domain::Error::NoDefaultSession)?;
        self.services.get_provider(model_config.provider).await
    }

    #[tracing::instrument(skip(self))]
    async fn mcp_auth(&self, server_url: &url::Url) -> Result<()> {
        let env = self.services.get_environment().clone();
        forge_infra::mcp_auth(server_url, &env).await
    }

    async fn mcp_logout(&self, server_url: Option<&url::Url>) -> Result<()> {
        let env = self.services.get_environment().clone();
        match server_url {
            Some(url) => forge_infra::mcp_logout(url, &env).await,
            None => forge_infra::mcp_logout_all(&env).await,
        }
    }

    async fn mcp_auth_status(&self, server_url: &url::Url) -> Result<forge_domain::McpAuthStatus> {
        let env = self.services.get_environment().clone();
        Ok(forge_infra::mcp_auth_status(server_url, &env).await)
    }

    fn hydrate_channel(&self) -> Result<()> {
        self.infra.hydrate();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::Path;
    use std::sync::Arc;
    use std::time::Duration;

    use anyhow::{Result, anyhow};
    use forge_app::dto::ConversationBranchTarget;
    use forge_app::*;
    use forge_config::ForgeConfig;
    use forge_domain::*;
    use forge_infra::{ForgeCommandExecutorService, StdConsoleWriter};
    use forge_project_model::{
        ProjectContextCommittedQueryResult, ProjectContextCommittedResultItem,
        ProjectContextEpisodeAppendFailureReason, ProjectContextPersistedEpisodeAppendOutcome,
        ProjectContextReadbackSummary,
    };
    use pretty_assertions::assert_eq;
    use tokio::sync::Mutex;
    use url::Url;

    use super::*;

    fn fixture_environment(cwd: PathBuf) -> Environment {
        Environment {
            os: std::env::consts::OS.to_string(),
            cwd: cwd.clone(),
            home: Some(cwd.clone()),
            shell: std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string()),
            base_path: cwd,
        }
    }

    fn fixture_api(
        cwd: PathBuf,
    ) -> ForgeAPI<(), ForgeCommandExecutorService<std::io::Stdout, std::io::Stderr>> {
        let infra = ForgeCommandExecutorService::new(
            fixture_environment(cwd),
            Arc::new(StdConsoleWriter::default()),
        );
        ForgeAPI::new(Arc::new(()), Arc::new(infra))
    }

    #[derive(Clone)]
    struct QueryWorkspaceServices {
        environment: Environment,
        config: ForgeConfig,
        workspace: Arc<QueryWorkspaceService>,
        noop: Arc<NoopService>,
    }

    impl QueryWorkspaceServices {
        fn new(cwd: PathBuf, workspace: Arc<QueryWorkspaceService>) -> Self {
            Self {
                environment: fixture_environment(cwd),
                config: ForgeConfig::default(),
                workspace,
                noop: Arc::new(NoopService),
            }
        }
    }

    impl EnvironmentInfra for QueryWorkspaceServices {
        type Config = ForgeConfig;

        fn get_env_var(&self, _key: &str) -> Option<String> {
            None
        }

        fn get_env_vars(&self) -> BTreeMap<String, String> {
            BTreeMap::new()
        }

        fn get_environment(&self) -> Environment {
            self.environment.clone()
        }

        fn get_config(&self) -> Result<Self::Config> {
            Ok(self.config.clone())
        }

        async fn update_environment(&self, _ops: Vec<forge_domain::ConfigOperation>) -> Result<()> {
            Ok(())
        }
    }

    impl Services for QueryWorkspaceServices {
        type ProviderService = NoopService;
        type AppConfigService = NoopService;
        type ConversationService = NoopService;
        type LearningService = NoopService;
        type SteerService = NoopService;
        type TemplateService = NoopService;
        type AttachmentService = NoopService;
        type CustomInstructionsService = NoopService;
        type FileDiscoveryService = NoopService;
        type McpConfigManager = NoopService;
        type FsWriteService = NoopService;
        type PlanCreateService = NoopService;
        type FsPatchService = NoopService;
        type FsReadService = NoopService;
        type ImageReadService = NoopService;
        type FsRemoveService = NoopService;
        type FsSearchService = NoopService;
        type FollowUpService = NoopService;
        type FsUndoService = NoopService;
        type NetFetchService = NoopService;
        type ShellService = NoopService;
        type McpService = NoopService;
        type AuthService = NoopService;
        type AgentRegistry = NoopService;
        type CommandLoaderService = NoopService;
        type PolicyService = NoopService;
        type ProviderAuthService = NoopService;
        type WorkspaceService = QueryWorkspaceService;
        type SkillFetchService = NoopService;

        fn provider_service(&self) -> &Self::ProviderService {
            &self.noop
        }
        fn config_service(&self) -> &Self::AppConfigService {
            &self.noop
        }
        fn conversation_service(&self) -> &Self::ConversationService {
            &self.noop
        }
        fn learning_service(&self) -> &Self::LearningService {
            &self.noop
        }
        fn steer_service(&self) -> &Self::SteerService {
            &self.noop
        }
        fn template_service(&self) -> &Self::TemplateService {
            &self.noop
        }
        fn attachment_service(&self) -> &Self::AttachmentService {
            &self.noop
        }
        fn file_discovery_service(&self) -> &Self::FileDiscoveryService {
            &self.noop
        }
        fn mcp_config_manager(&self) -> &Self::McpConfigManager {
            &self.noop
        }
        fn fs_create_service(&self) -> &Self::FsWriteService {
            &self.noop
        }
        fn plan_create_service(&self) -> &Self::PlanCreateService {
            &self.noop
        }
        fn fs_patch_service(&self) -> &Self::FsPatchService {
            &self.noop
        }
        fn fs_read_service(&self) -> &Self::FsReadService {
            &self.noop
        }
        fn image_read_service(&self) -> &Self::ImageReadService {
            &self.noop
        }
        fn fs_remove_service(&self) -> &Self::FsRemoveService {
            &self.noop
        }
        fn fs_search_service(&self) -> &Self::FsSearchService {
            &self.noop
        }
        fn follow_up_service(&self) -> &Self::FollowUpService {
            &self.noop
        }
        fn fs_undo_service(&self) -> &Self::FsUndoService {
            &self.noop
        }
        fn net_fetch_service(&self) -> &Self::NetFetchService {
            &self.noop
        }
        fn shell_service(&self) -> &Self::ShellService {
            &self.noop
        }
        fn mcp_service(&self) -> &Self::McpService {
            &self.noop
        }
        fn custom_instructions_service(&self) -> &Self::CustomInstructionsService {
            &self.noop
        }
        fn auth_service(&self) -> &Self::AuthService {
            &self.noop
        }
        fn agent_registry(&self) -> &Self::AgentRegistry {
            &self.noop
        }
        fn command_loader_service(&self) -> &Self::CommandLoaderService {
            &self.noop
        }
        fn policy_service(&self) -> &Self::PolicyService {
            &self.noop
        }
        fn provider_auth_service(&self) -> &Self::ProviderAuthService {
            &self.noop
        }
        fn workspace_service(&self) -> &Self::WorkspaceService {
            &self.workspace
        }
        fn skill_fetch_service(&self) -> &Self::SkillFetchService {
            &self.noop
        }
    }

    #[derive(Default)]
    struct QueryWorkspaceService {
        committed_result: Mutex<Option<QueryWorkspaceOutcome>>,
        legacy_called: Mutex<bool>,
    }

    enum QueryWorkspaceOutcome {
        Ok(ProjectContextCommittedQueryResult, Vec<Node>),
        Err(String),
    }

    impl QueryWorkspaceService {
        async fn set_committed_result(
            &self,
            result: ProjectContextCommittedQueryResult,
            nodes: Vec<Node>,
        ) {
            *self.committed_result.lock().await = Some(QueryWorkspaceOutcome::Ok(result, nodes));
        }

        async fn set_committed_error(&self, error: impl Into<String>) {
            *self.committed_result.lock().await = Some(QueryWorkspaceOutcome::Err(error.into()));
        }

        async fn legacy_called(&self) -> bool {
            *self.legacy_called.lock().await
        }
    }

    #[async_trait::async_trait]
    impl WorkspaceService for QueryWorkspaceService {
        async fn sync_workspace(&self, _path: PathBuf) -> Result<MpscStream<Result<SyncProgress>>> {
            anyhow::bail!("unused workspace service")
        }
        async fn produce_workspace_exact_fact_reference(
            &self,
            _path: PathBuf,
        ) -> Result<WorkspaceExactFactReferenceReport> {
            anyhow::bail!("unused workspace service")
        }
        async fn workspace_exact_fact_status(
            &self,
            _path: PathBuf,
        ) -> Result<WorkspaceExactFactStatusReport> {
            anyhow::bail!("unused workspace service")
        }
        async fn workspace_evidence_replay_diagnostic(
            &self,
            _path: PathBuf,
        ) -> Result<WorkspaceEvidenceReplayDiagnostic> {
            anyhow::bail!("unused workspace service")
        }
        async fn workspace_evidence_replay_preview_diagnostic(
            &self,
            _path: PathBuf,
        ) -> Result<WorkspaceEvidenceReplayPreviewDiagnostic> {
            anyhow::bail!("unused workspace service")
        }
        async fn build_workspace_vector_index(
            &self,
            _path: PathBuf,
            _embedding_model_id: String,
        ) -> Result<WorkspaceVectorIndexBuildReport> {
            anyhow::bail!("unused workspace service")
        }
        async fn embed_workspace_query(
            &self,
            _query: String,
            _embedding_model_id: String,
        ) -> Result<ProjectSemanticEmbeddingOutput> {
            anyhow::bail!("unused workspace service")
        }
        async fn semantic_injection_readiness(
            &self,
            _path: PathBuf,
            _embedding_model_id: Option<String>,
        ) -> Result<WorkspaceSemanticInjectionReadiness> {
            anyhow::bail!("unused workspace service")
        }
        async fn sem_search_availability(
            &self,
            _path: PathBuf,
            _embedding_model_id: Option<String>,
        ) -> Result<SemSearchAvailability> {
            anyhow::bail!("unused workspace service")
        }
        async fn sem_search_diagnostic(
            &self,
            _path: PathBuf,
            _embedding_model_id: Option<String>,
        ) -> Result<SemSearchDiagnosticReport> {
            anyhow::bail!("unused workspace service")
        }

        async fn query_workspace_committed(
            &self,
            _path: PathBuf,
            _params: SearchParams<'_>,
        ) -> Result<(ProjectContextCommittedQueryResult, Vec<Node>)> {
            match self.committed_result.lock().await.take() {
                Some(QueryWorkspaceOutcome::Ok(result, nodes)) => Ok((result, nodes)),
                Some(QueryWorkspaceOutcome::Err(error)) => Err(anyhow!(error)),
                None => anyhow::bail!("committed query fixture was not configured"),
            }
        }

        async fn query_workspace(
            &self,
            _path: PathBuf,
            _params: SearchParams<'_>,
        ) -> Result<Vec<Node>> {
            *self.legacy_called.lock().await = true;
            anyhow::bail!("legacy query_workspace must not be called")
        }

        async fn list_workspaces(&self) -> Result<Vec<WorkspaceInfo>> {
            anyhow::bail!("unused workspace service")
        }
        async fn get_workspace_info(&self, _path: PathBuf) -> Result<Option<WorkspaceInfo>> {
            anyhow::bail!("unused workspace service")
        }
        async fn is_indexed(&self, _path: &Path) -> Result<bool> {
            anyhow::bail!("unused workspace service")
        }
        async fn delete_workspace(&self, _workspace_id: &WorkspaceId) -> Result<()> {
            anyhow::bail!("unused workspace service")
        }
        async fn delete_workspaces(&self, _workspace_ids: &[WorkspaceId]) -> Result<()> {
            anyhow::bail!("unused workspace service")
        }
        async fn project_model_context_diagnostic(
            &self,
            _path: &Path,
        ) -> Result<WorkspaceContextManifestDiagnostic> {
            anyhow::bail!("unused workspace service")
        }
        async fn get_workspace_status(&self, _path: PathBuf) -> Result<Vec<FileStatus>> {
            anyhow::bail!("unused workspace service")
        }
        async fn is_authenticated(&self) -> Result<bool> {
            anyhow::bail!("unused workspace service")
        }
        async fn init_auth_credentials(&self) -> Result<WorkspaceAuth> {
            anyhow::bail!("unused workspace service")
        }
        async fn init_workspace(&self, _path: PathBuf) -> Result<WorkspaceId> {
            anyhow::bail!("unused workspace service")
        }
    }

    #[derive(Default)]
    struct NoopService;

    #[async_trait::async_trait]
    impl ProviderService for NoopService {
        async fn chat(
            &self,
            _model_id: &ModelId,
            _context: forge_domain::Context,
            _provider: Provider<Url>,
        ) -> ResultStream<ChatCompletionMessage, anyhow::Error> {
            anyhow::bail!("unused provider service")
        }
        async fn models(&self, _provider: Provider<Url>) -> Result<Vec<Model>> {
            anyhow::bail!("unused provider service")
        }
        async fn get_provider(&self, _id: ProviderId) -> Result<Provider<Url>> {
            anyhow::bail!("unused provider service")
        }
        async fn get_all_providers(&self) -> Result<Vec<AnyProvider>> {
            anyhow::bail!("unused provider service")
        }
        async fn upsert_credential(&self, _credential: AuthCredential) -> Result<()> {
            anyhow::bail!("unused provider service")
        }
        async fn remove_credential(&self, _id: &ProviderId) -> Result<()> {
            anyhow::bail!("unused provider service")
        }
        async fn migrate_env_credentials(&self) -> Result<Option<MigrationResult>> {
            anyhow::bail!("unused provider service")
        }
    }

    #[async_trait::async_trait]
    impl AppConfigService for NoopService {
        async fn get_session_config(&self) -> Option<ModelConfig> {
            None
        }
        async fn get_commit_config(&self) -> Result<Option<ModelConfig>> {
            Ok(None)
        }
        async fn get_suggest_config(&self) -> Result<Option<ModelConfig>> {
            Ok(None)
        }
        async fn get_reasoning_effort(&self) -> Result<Option<Effort>> {
            Ok(None)
        }
        async fn update_config(&self, _ops: Vec<ConfigOperation>) -> Result<()> {
            anyhow::bail!("unused config service")
        }
    }

    #[async_trait::async_trait]
    impl ConversationService for NoopService {
        async fn find_conversation(&self, _id: &ConversationId) -> Result<Option<Conversation>> {
            anyhow::bail!("unused conversation service")
        }
        async fn upsert_conversation(&self, _conversation: Conversation) -> Result<()> {
            anyhow::bail!("unused conversation service")
        }
        async fn ensure_delegated_conversation(
            &self,
            _id: &ConversationId,
            _parent_id: Option<ConversationId>,
        ) -> Result<Conversation> {
            anyhow::bail!("unused conversation service")
        }
        async fn resolve_root_conversation_id(
            &self,
            _parent_id: Option<ConversationId>,
        ) -> Result<Option<ConversationId>> {
            anyhow::bail!("unused conversation service")
        }
        async fn modify_conversation<FN, T>(&self, _id: &ConversationId, _f: FN) -> Result<T>
        where
            FN: FnOnce(&mut Conversation) -> T + Send,
            T: Send,
        {
            anyhow::bail!("unused conversation service")
        }
        async fn list_branch_targets(
            &self,
            _conversation_id: &ConversationId,
        ) -> Result<Vec<ConversationBranchTarget>> {
            anyhow::bail!("unused conversation service")
        }
        async fn branch_conversation(
            &self,
            _conversation_id: &ConversationId,
            _target_id: MessageId,
        ) -> Result<Conversation> {
            anyhow::bail!("unused conversation service")
        }
        async fn get_conversation_list_items_by_query(
            &self,
            _query: ConversationListQuery,
        ) -> Result<Vec<ConversationListItem>> {
            anyhow::bail!("unused conversation service")
        }
        async fn get_conversation_list_items_including_agent(
            &self,
            _limit: usize,
        ) -> Result<Vec<ConversationListItem>> {
            anyhow::bail!("unused conversation service")
        }
        async fn get_conversation_list_items_by_visibility(
            &self,
            _visibility: ConversationVisibilityFilter,
            _limit: usize,
        ) -> Result<Vec<ConversationListItem>> {
            anyhow::bail!("unused conversation service")
        }
        async fn get_conversations(&self) -> Result<Vec<Conversation>> {
            anyhow::bail!("unused conversation service")
        }
        async fn get_conversations_including_agent(&self) -> Result<Vec<Conversation>> {
            anyhow::bail!("unused conversation service")
        }
        async fn get_conversations_by_visibility(
            &self,
            _visibility: ConversationVisibilityFilter,
        ) -> Result<Vec<Conversation>> {
            anyhow::bail!("unused conversation service")
        }
        async fn get_sub_conversations(
            &self,
            _parent_id: &ConversationId,
        ) -> Result<Vec<Conversation>> {
            anyhow::bail!("unused conversation service")
        }
        async fn upsert_subagent_task_session(&self, _session: SubagentTaskSession) -> Result<()> {
            anyhow::bail!("unused conversation service")
        }
        async fn get_subagent_task_session(
            &self,
            _task_id: &SubagentTaskId,
        ) -> Result<Option<SubagentTaskSession>> {
            anyhow::bail!("unused conversation service")
        }
        async fn get_subagent_task_session_by_conversation(
            &self,
            _conversation_id: &ConversationId,
        ) -> Result<Option<SubagentTaskSession>> {
            anyhow::bail!("unused conversation service")
        }
        async fn list_subagent_task_sessions(
            &self,
            _filter: SubagentTaskSessionFilter,
        ) -> Result<Vec<SubagentTaskSession>> {
            anyhow::bail!("unused conversation service")
        }
        async fn last_conversation(&self) -> Result<Option<Conversation>> {
            anyhow::bail!("unused conversation service")
        }
        async fn delete_conversation(&self, _conversation_id: &ConversationId) -> Result<()> {
            anyhow::bail!("unused conversation service")
        }
    }

    #[async_trait::async_trait]
    impl LearningService for NoopService {
        async fn capture_candidate_from_conversation(
            &self,
            _conversation_id: ConversationId,
            _source_event_id: String,
            _summary: String,
            _metadata: LearningCaptureMetadata,
        ) -> Result<LearningLedgerAppendOutcome> {
            anyhow::bail!("unused learning service")
        }
        async fn insert_learning_event(
            &self,
            _event: LearningLedgerEvent,
        ) -> Result<LearningLedgerAppendOutcome> {
            anyhow::bail!("unused learning service")
        }
        async fn review_learning_candidate_event(
            &self,
            _event: LearningLedgerEvent,
        ) -> Result<LearningReviewOutcome> {
            anyhow::bail!("unused learning service")
        }
        async fn promote_sensor_lesson(
            &self,
            _request: SensorLessonPromotionRequest,
        ) -> Result<SensorLessonPromotionOutcome> {
            anyhow::bail!("unused learning service")
        }
        async fn get_learning_record(
            &self,
            _record_id: LearningRecordId,
        ) -> Result<Option<LearningRecordProjection>> {
            anyhow::bail!("unused learning service")
        }
        async fn list_learning_records(
            &self,
            _review_state: Option<LearningReviewState>,
            _limit: usize,
        ) -> Result<Vec<LearningRecordProjection>> {
            anyhow::bail!("unused learning service")
        }
        async fn learning_freshness(
            &self,
            _review_state: Option<LearningReviewState>,
        ) -> Result<LearningLedgerFreshness> {
            anyhow::bail!("unused learning service")
        }
    }

    #[async_trait::async_trait]
    impl SteerService for NoopService {
        async fn enqueue_steer(
            &self,
            _conversation_id: &ConversationId,
            _message: SteerMessage,
        ) -> Result<()> {
            anyhow::bail!("unused steer service")
        }
        async fn clear_steer(&self, _conversation_id: &ConversationId) -> Result<()> {
            anyhow::bail!("unused steer service")
        }
        async fn drain_steer(
            &self,
            _conversation_id: &ConversationId,
        ) -> Result<Vec<SteerMessage>> {
            anyhow::bail!("unused steer service")
        }
    }

    #[async_trait::async_trait]
    impl TemplateService for NoopService {
        async fn register_template(&self, _path: PathBuf) -> Result<()> {
            anyhow::bail!("unused template service")
        }
        async fn render_template<V: serde::Serialize + Send + Sync>(
            &self,
            _template: Template<V>,
            _object: &V,
        ) -> Result<String> {
            anyhow::bail!("unused template service")
        }
    }

    #[async_trait::async_trait]
    impl AttachmentService for NoopService {
        async fn attachments(&self, _url: &str) -> Result<Vec<Attachment>> {
            anyhow::bail!("unused attachment service")
        }
    }
    #[async_trait::async_trait]
    impl CustomInstructionsService for NoopService {
        async fn get_custom_instructions(&self) -> Vec<String> {
            Vec::new()
        }
    }
    #[async_trait::async_trait]
    impl FileDiscoveryService for NoopService {
        async fn collect_files(&self, _config: Walker) -> Result<Vec<File>> {
            anyhow::bail!("unused file discovery service")
        }
        async fn list_current_directory(&self) -> Result<Vec<File>> {
            anyhow::bail!("unused file discovery service")
        }
    }
    #[async_trait::async_trait]
    impl McpConfigManager for NoopService {
        async fn read_mcp_config(&self, _scope: Option<&Scope>) -> Result<McpConfig> {
            anyhow::bail!("unused mcp config manager")
        }
        async fn write_mcp_config(&self, _config: &McpConfig, _scope: &Scope) -> Result<()> {
            anyhow::bail!("unused mcp config manager")
        }
    }
    #[async_trait::async_trait]
    impl FsWriteService for NoopService {
        async fn write(
            &self,
            _path: String,
            _content: String,
            _overwrite: bool,
        ) -> Result<FsWriteOutput> {
            anyhow::bail!("unused write service")
        }
    }
    #[async_trait::async_trait]
    impl PlanCreateService for NoopService {
        async fn create_plan(
            &self,
            _plan_name: String,
            _version: String,
            _content: String,
        ) -> Result<PlanCreateOutput> {
            anyhow::bail!("unused plan service")
        }
    }
    #[async_trait::async_trait]
    impl FsPatchService for NoopService {
        async fn patch(
            &self,
            _path: String,
            _search: String,
            _content: String,
            _replace_all: bool,
        ) -> Result<PatchOutput> {
            anyhow::bail!("unused patch service")
        }
        async fn multi_patch(&self, _path: String, _edits: Vec<PatchEdit>) -> Result<PatchOutput> {
            anyhow::bail!("unused patch service")
        }
    }
    #[async_trait::async_trait]
    impl FsReadService for NoopService {
        async fn read(
            &self,
            _path: String,
            _start_line: Option<u64>,
            _end_line: Option<u64>,
        ) -> Result<ReadOutput> {
            anyhow::bail!("unused read service")
        }
    }
    #[async_trait::async_trait]
    impl ImageReadService for NoopService {
        async fn read_image(&self, _path: String) -> Result<Image> {
            anyhow::bail!("unused image service")
        }
    }
    #[async_trait::async_trait]
    impl FsRemoveService for NoopService {
        async fn remove(&self, _path: String) -> Result<FsRemoveOutput> {
            anyhow::bail!("unused remove service")
        }
    }
    #[async_trait::async_trait]
    impl FsSearchService for NoopService {
        async fn search(&self, _params: FSSearch) -> Result<Option<SearchResult>> {
            anyhow::bail!("unused search service")
        }
    }
    #[async_trait::async_trait]
    impl FollowUpService for NoopService {
        async fn follow_up(
            &self,
            _question: String,
            _options: Vec<String>,
            _multiple: Option<bool>,
        ) -> Result<Option<String>> {
            anyhow::bail!("unused follow up service")
        }
    }
    #[async_trait::async_trait]
    impl FsUndoService for NoopService {
        async fn undo(&self, _path: String) -> Result<FsUndoOutput> {
            anyhow::bail!("unused undo service")
        }
    }
    #[async_trait::async_trait]
    impl NetFetchService for NoopService {
        async fn fetch(&self, _url: String, _raw: Option<bool>) -> Result<HttpResponse> {
            anyhow::bail!("unused fetch service")
        }
    }
    #[async_trait::async_trait]
    impl ShellService for NoopService {
        async fn execute(&self, _request: ShellExecuteRequest) -> Result<ShellOutput> {
            anyhow::bail!("unused shell service")
        }
    }
    #[async_trait::async_trait]
    impl McpService for NoopService {
        async fn get_mcp_servers(&self) -> Result<McpServers> {
            anyhow::bail!("unused mcp service")
        }
        async fn execute_mcp(&self, _call: ToolCallFull) -> Result<ToolOutput> {
            anyhow::bail!("unused mcp service")
        }
        async fn reload_mcp(&self) -> Result<()> {
            anyhow::bail!("unused mcp service")
        }
    }
    #[async_trait::async_trait]
    impl AuthService for NoopService {
        async fn user_info(&self, _api_key: &str) -> Result<User> {
            anyhow::bail!("unused auth service")
        }
        async fn user_usage(&self, _api_key: &str) -> Result<UserUsage> {
            anyhow::bail!("unused auth service")
        }
    }
    #[async_trait::async_trait]
    impl AgentRegistry for NoopService {
        async fn get_active_agent_id(&self) -> Result<Option<AgentId>> {
            Ok(None)
        }
        async fn set_active_agent_id(&self, _agent_id: AgentId) -> Result<()> {
            anyhow::bail!("unused agent registry")
        }
        async fn get_agents(&self) -> Result<Vec<Agent>> {
            Ok(Vec::new())
        }
        async fn get_agent_infos(&self) -> Result<Vec<AgentInfo>> {
            Ok(Vec::new())
        }
        async fn get_agent(&self, _agent_id: &AgentId) -> Result<Option<Agent>> {
            Ok(None)
        }
        async fn reload_agents(&self) -> Result<()> {
            Ok(())
        }
    }
    #[async_trait::async_trait]
    impl CommandLoaderService for NoopService {
        async fn get_commands(&self) -> Result<Vec<Command>> {
            Ok(Vec::new())
        }
    }
    #[async_trait::async_trait]
    impl PolicyService for NoopService {
        async fn check_operation_permission(
            &self,
            _operation: &PermissionOperation,
        ) -> Result<PolicyDecision> {
            anyhow::bail!("unused policy service")
        }
    }
    #[async_trait::async_trait]
    impl ProviderAuthService for NoopService {
        async fn init_provider_auth(
            &self,
            _provider_id: ProviderId,
            _method: AuthMethod,
        ) -> Result<AuthContextRequest> {
            anyhow::bail!("unused provider auth service")
        }
        async fn complete_provider_auth(
            &self,
            _provider_id: ProviderId,
            _context: AuthContextResponse,
            _timeout: Duration,
        ) -> Result<()> {
            anyhow::bail!("unused provider auth service")
        }
        async fn refresh_provider_credential(
            &self,
            _provider: Provider<Url>,
        ) -> Result<Provider<Url>> {
            anyhow::bail!("unused provider auth service")
        }
    }
    #[async_trait::async_trait]
    impl SkillFetchService for NoopService {
        async fn fetch_skill(&self, _skill_name: String) -> Result<Skill> {
            anyhow::bail!("unused skill service")
        }
        async fn list_skills(&self) -> Result<Vec<Skill>> {
            anyhow::bail!("unused skill service")
        }
    }

    #[derive(Clone)]
    struct QueryWorkspaceInfra {
        environment: Environment,
        config: ForgeConfig,
    }

    impl QueryWorkspaceInfra {
        fn new(cwd: PathBuf) -> Self {
            Self {
                environment: fixture_environment(cwd),
                config: ForgeConfig::default(),
            }
        }
    }

    impl EnvironmentInfra for QueryWorkspaceInfra {
        type Config = ForgeConfig;

        fn get_env_var(&self, _key: &str) -> Option<String> {
            None
        }
        fn get_env_vars(&self) -> BTreeMap<String, String> {
            BTreeMap::new()
        }
        fn get_environment(&self) -> Environment {
            self.environment.clone()
        }
        fn get_config(&self) -> Result<Self::Config> {
            Ok(self.config.clone())
        }
        async fn update_environment(&self, _ops: Vec<forge_domain::ConfigOperation>) -> Result<()> {
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl CommandInfra for QueryWorkspaceInfra {
        async fn execute_command(
            &self,
            _command: String,
            _working_dir: PathBuf,
            _silent: bool,
            _env_vars: Option<Vec<String>>,
            _handoff_timeout: ShellHandoffTimeoutSeconds,
        ) -> Result<CommandExecutionOutput> {
            Err(anyhow!("unused command infra"))
        }
        async fn execute_command_raw(
            &self,
            _command: &str,
            _working_dir: PathBuf,
            _env_vars: Option<Vec<String>>,
        ) -> Result<std::process::ExitStatus> {
            Err(anyhow!("unused command infra"))
        }
        async fn start_process(
            &self,
            _command: String,
            _working_dir: PathBuf,
            _env_vars: Option<Vec<String>>,
        ) -> Result<ProcessStartOutput> {
            Err(anyhow!("unused command infra"))
        }
        async fn process_status(
            &self,
            _process_id: forge_domain::ProcessId,
            _wait: Option<ProcessObservationWaitSeconds>,
        ) -> Result<ProcessStatus> {
            Err(anyhow!("unused command infra"))
        }
        async fn read_process(
            &self,
            _process_id: forge_domain::ProcessId,
            _cursor: ProcessReadCursor,
            _wait: Option<ProcessObservationWaitSeconds>,
        ) -> Result<ProcessReadOutput> {
            Err(anyhow!("unused command infra"))
        }
        async fn list_processes(&self) -> Result<Vec<ProcessStatus>> {
            Err(anyhow!("unused command infra"))
        }
        async fn kill_process(
            &self,
            _process_id: forge_domain::ProcessId,
        ) -> Result<ProcessStatus> {
            Err(anyhow!("unused command infra"))
        }
    }

    #[async_trait::async_trait]
    impl SkillRepository for QueryWorkspaceInfra {
        async fn load_skills(&self) -> Result<Vec<Skill>> {
            Ok(Vec::new())
        }
    }

    impl GrpcInfra for QueryWorkspaceInfra {
        fn channel(&self) -> Result<tonic::transport::Channel> {
            Err(anyhow!("unused grpc infra"))
        }
        fn hydrate(&self) {}
    }

    fn fixture_query_workspace_api(
        cwd: PathBuf,
        workspace: Arc<QueryWorkspaceService>,
    ) -> ForgeAPI<QueryWorkspaceServices, QueryWorkspaceInfra> {
        let services = Arc::new(QueryWorkspaceServices::new(cwd.clone(), workspace));
        let infra = Arc::new(QueryWorkspaceInfra::new(cwd));
        ForgeAPI::new(services, infra)
    }

    fn fixture_node(node_id: &str, file_path: &str, content: &str) -> Node {
        Node {
            node_id: NodeId::new(node_id),
            node: NodeData::FileChunk(FileChunk {
                file_path: file_path.to_string(),
                content: content.to_string(),
                start_line: 1,
                end_line: 3,
            }),
            relevance: Some(0.99),
            distance: Some(0.01),
        }
    }

    fn fixture_committed_episode_append_failed_result() -> Result<ProjectContextCommittedQueryResult>
    {
        let read_request =
            forge_project_model::ProjectContextReadRequest::new("src/api.rs", "api-node", 1, 3)?;
        let context_pack = forge_project_model::ContextPack {
            version: 1,
            manifest_hash: "api-fixture-manifest".to_string(),
            evidence: vec![forge_project_model::ContextPackEvidence {
                id: "api-node".to_string(),
                path: "src/api.rs".to_string(),
                symbol: None,
                source: forge_project_model::ContextPackEvidenceSource::RetrievalResult,
                freshness: forge_project_model::EvidenceFreshness::Fresh,
                provenance: forge_project_model::Provenance {
                    path: "src/api.rs".to_string(),
                    start_line: Some(1),
                    end_line: Some(3),
                    source: "fixture".to_string(),
                    fingerprint: "api-fixture-fingerprint".to_string(),
                },
                score: 0.99,
            }],
            provenance: vec![forge_project_model::Provenance {
                path: "src/api.rs".to_string(),
                start_line: Some(1),
                end_line: Some(3),
                source: "fixture".to_string(),
                fingerprint: "api-fixture-fingerprint".to_string(),
            }],
        };
        let retrieval_plan = forge_project_model::ProjectContextRetrievalPlan {
            query_diagnostics: forge_project_model::ProjectContextRetrievalQueryDiagnostics {
                query_text: Some("api boundary".to_string()),
                path_prefix: None,
                path_suffixes: Vec::new(),
                limit: 1,
                top_k: Some(1),
                top_k_status: forge_project_model::ProjectContextTopKStatus::Applied {
                    candidate_count: 1,
                },
                use_case: Some("prove committed consumption".to_string()),
                include_graph_expansion: false,
                stale_policy: forge_project_model::StaleEvidencePolicy::Reject,
                freshness_proof_level: forge_project_model::FreshnessProofLevel::FullFilesystem,
                phase_diagnostics:
                    forge_project_model::ProjectContextRetrievalPhaseDiagnostics::default(),
            },
            selected_results: Vec::new(),
            context_pack: Some(context_pack),
            read_requests: vec![read_request.clone()],
            write_decision:
                forge_project_model::ProjectContextWriteDecision::WriteContextPackAfterReadback,
            return_order: Vec::new(),
        };
        let replay_activation = forge_project_model::ReplayActivationBoundary {
            manifest_hash: "api-fixture-manifest".to_string(),
            active_refs: Vec::new(),
            issues: Vec::new(),
            diagnostics: forge_project_model::ReplayActivationDiagnostics::default(),
        };
        let commit = forge_project_model::ProjectContextPackCommit::from_retrieval_plan(
            &retrieval_plan,
            replay_activation,
        )?;
        let commit = match commit.verify_readbacks(vec![
            forge_project_model::ProjectContextReadbackOutcome::succeeded(&read_request),
        ])? {
            forge_project_model::ProjectContextPackReadbackDecision::Write(commit) => commit,
            forge_project_model::ProjectContextPackReadbackDecision::NoWrite(_) => {
                anyhow::bail!("fixture committed query should produce persisted proof")
            }
        };
        let tempdir = tempfile::tempdir()?;
        let indexer = forge_project_model::ProjectIndexer::new(
            tempdir.path(),
            tempdir.path().join(".forge_project_model"),
        );
        let proof = indexer.persist_verified_context_pack(&commit)?;
        Ok(ProjectContextCommittedQueryResult::persisted(
            ProjectContextReadbackSummary::from_outcomes(&[
                forge_project_model::ProjectContextReadbackOutcome::succeeded(&read_request),
            ]),
            proof,
            ProjectContextPersistedEpisodeAppendOutcome::failed(
                ProjectContextEpisodeAppendFailureReason::EpisodeAppendFailed,
            ),
            vec![ProjectContextCommittedResultItem::new(
                "api-node",
                Some(0.99),
            )],
        ))
    }

    #[tokio::test]
    async fn test_query_workspace_uses_committed_boundary_and_returns_only_nodes() {
        let setup = tempfile::tempdir().unwrap();
        let workspace = Arc::new(QueryWorkspaceService::default());
        let expected = vec![fixture_node(
            "api-node",
            "src/api.rs",
            "pub fn api_boundary() {}",
        )];
        workspace
            .set_committed_result(
                fixture_committed_episode_append_failed_result().unwrap(),
                expected.clone(),
            )
            .await;
        let fixture = fixture_query_workspace_api(setup.path().to_path_buf(), workspace.clone());

        let actual = fixture
            .query_workspace(
                setup.path().to_path_buf(),
                SearchParams::new("api boundary", "prove committed consumption"),
            )
            .await
            .unwrap();

        assert_eq!(actual, expected);
        assert!(!workspace.legacy_called().await);
    }

    #[tokio::test]
    async fn test_query_workspace_serializes_public_nodes_without_committed_metadata_leakage() {
        let setup = tempfile::tempdir().unwrap();
        let workspace = Arc::new(QueryWorkspaceService::default());
        workspace
            .set_committed_result(
                fixture_committed_episode_append_failed_result().unwrap(),
                vec![fixture_node(
                    "api-node",
                    "src/api.rs",
                    "public node content stays visible",
                )],
            )
            .await;
        let fixture = fixture_query_workspace_api(setup.path().to_path_buf(), workspace);
        let nodes = fixture
            .query_workspace(
                setup.path().to_path_buf(),
                SearchParams::new("serialization", "prove public api payload"),
            )
            .await
            .unwrap();

        let actual = serde_json::to_string(&nodes).unwrap();

        assert!(!actual.contains("ProjectContextCommittedQueryResult"));
        assert!(!actual.contains("ProjectContextPersistedEpisodeAppendOutcome"));
        assert!(!actual.contains("EpisodeAppendFailed"));
    }

    #[tokio::test]
    async fn test_query_workspace_propagates_committed_boundary_error() {
        let setup = tempfile::tempdir().unwrap();
        let workspace = Arc::new(QueryWorkspaceService::default());
        workspace
            .set_committed_error("committed boundary hard failure")
            .await;
        let fixture = fixture_query_workspace_api(setup.path().to_path_buf(), workspace.clone());

        let actual = fixture
            .query_workspace(
                setup.path().to_path_buf(),
                SearchParams::new("hard error", "prove propagation"),
            )
            .await
            .unwrap_err()
            .to_string();
        let expected = "committed boundary hard failure";

        assert_eq!(actual, expected);
        assert!(!workspace.legacy_called().await);
    }

    #[tokio::test]
    async fn test_api_shell_command_returns_short_output_synchronously() {
        let setup = tempfile::tempdir().unwrap();
        let fixture = fixture_api(setup.path().to_path_buf());

        let actual = fixture
            .execute_shell_command_with_handoff_timeout(
                "printf api-short-output",
                setup.path().to_path_buf(),
                ShellHandoffTimeoutSeconds::new(1).unwrap(),
            )
            .await
            .unwrap();
        let expected = CommandExecutionOutput {
            output: CommandOutput {
                command: "printf api-short-output".to_string(),
                stdout: "api-short-output".to_string(),
                stderr: String::new(),
                exit_code: Some(0),
            },
            process: None,
        };

        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn test_api_shell_command_hands_off_long_output_as_managed_process() {
        let setup = tempfile::tempdir().unwrap();
        let fixture = fixture_api(setup.path().to_path_buf());

        let actual = fixture
            .execute_shell_command_with_handoff_timeout(
                "printf api-before; sleep 1.05; printf api-after; sleep 2",
                setup.path().to_path_buf(),
                ShellHandoffTimeoutSeconds::new(1).unwrap(),
            )
            .await
            .unwrap();
        let process = actual
            .process
            .clone()
            .expect("long-running API command should be handed off");
        let initial_output = fixture
            .infra
            .read_process(
                process.process_id.clone(),
                forge_domain::ProcessReadCursor::new(0),
                None,
            )
            .await
            .unwrap();
        let process_output = fixture
            .infra
            .read_process(
                process.process_id.clone(),
                initial_output.next_cursor,
                Some(ProcessObservationWaitSeconds::new(3).unwrap()),
            )
            .await
            .unwrap();
        let _ = fixture.infra.kill_process(process.process_id).await;

        assert_eq!(actual.output.stdout, "api-before");
        assert_eq!(actual.output.exit_code, None);
        assert!(actual.output.stderr.contains("managed background process"));
        assert!(
            process_output
                .entries
                .iter()
                .any(|entry| entry.content.contains("api-after"))
        );
    }
}

impl<A: Send + Sync, F: ConsoleWriter> ConsoleWriter for ForgeAPI<A, F> {
    fn write(&self, buf: &[u8]) -> std::io::Result<usize> {
        self.infra.write(buf)
    }

    fn write_err(&self, buf: &[u8]) -> std::io::Result<usize> {
        self.infra.write_err(buf)
    }

    fn flush(&self) -> std::io::Result<()> {
        self.infra.flush()
    }

    fn flush_err(&self) -> std::io::Result<()> {
        self.infra.flush_err()
    }
}
