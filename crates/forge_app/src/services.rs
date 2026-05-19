use std::path::{Path, PathBuf};
use std::time::Duration;

use bytes::Bytes;
use derive_setters::Setters;
use forge_domain::{
    AgentId, AnyProvider, Attachment, AuthContextRequest, AuthContextResponse, AuthMethod,
    ChatCompletionMessage, CommandOutput, Context, Conversation, ConversationId, File, FileInfo,
    FileStatus, Image, LearningLedgerEvent, LearningLedgerFreshness, LearningRecordProjection,
    LearningReviewState, McpConfig, McpServers, Model, ModelId, Node, ProcessId, ProcessReadCursor,
    ProcessReadOutput, ProcessStartOutput, ProcessStatus, Provider, ProviderId, ResultStream,
    Scope, SearchParams, SteerMessage, SubagentTaskId, SubagentTaskSession,
    SubagentTaskSessionFilter, SyncProgress, SyntaxError, Template, ToolCallFull, ToolOutput,
    WorkspaceAuth, WorkspaceContextManifestDiagnostic, WorkspaceExactFactReferenceReport,
    WorkspaceExactFactStatusReport, WorkspaceId, WorkspaceInfo,
};
use forge_eventsource::EventSource;
use reqwest::Response;
use reqwest::header::HeaderMap;
use url::Url;

use crate::user::{User, UserUsage};
use crate::{EnvironmentInfra, Walker};

#[derive(Debug, Clone)]
pub struct ShellOutput {
    pub output: CommandOutput,
    pub shell: String,
    pub description: Option<String>,
    /// Present when the shell command exceeded the startup window and was
    /// handed off to the managed background-process session flow.
    pub process: Option<ProcessStartOutput>,
}

#[derive(Debug, Clone)]
pub struct ProcessOutput {
    pub shell: String,
    pub description: Option<String>,
    pub status: ProcessStatus,
}

#[derive(Debug, Clone)]
pub struct ProcessKillServiceOutput {
    pub shell: String,
    pub description: Option<String>,
    pub status: ProcessStatus,
}

#[derive(Debug, Clone)]
pub struct ProcessStartServiceOutput {
    pub shell: String,
    pub description: Option<String>,
    pub output: ProcessStartOutput,
}

#[derive(Debug, Clone)]
pub struct ProcessReadServiceOutput {
    pub shell: String,
    pub output: ProcessReadOutput,
}

#[derive(Debug)]
pub struct PatchOutput {
    pub errors: Vec<SyntaxError>,
    pub before: String,
    pub after: String,
    pub content_hash: String,
}

#[derive(Debug, Setters)]
#[setters(into)]
pub struct ReadOutput {
    pub content: Content,
    pub info: FileInfo,
}

#[derive(Debug)]
pub enum Content {
    File(String),
    Image(Image),
}

impl Content {
    pub fn file<S: Into<String>>(content: S) -> Self {
        Self::File(content.into())
    }

    pub fn image(image: Image) -> Self {
        Self::Image(image)
    }

    pub fn file_content(&self) -> &str {
        match self {
            Self::File(content) => content,
            Self::Image(_) => "",
        }
    }

    pub fn as_image(&self) -> Option<&Image> {
        match self {
            Self::Image(img) => Some(img),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct SearchResult {
    pub matches: Vec<Match>,
}

#[derive(Debug)]
pub struct Match {
    pub path: String,
    pub result: Option<MatchResult>,
}

#[derive(Debug)]
pub enum MatchResult {
    Error(String),
    Found {
        line_number: Option<usize>,
        line: String,
    },
    Count {
        count: usize,
    },
    FileMatch, // For files_with_matches mode
    ContextMatch {
        line_number: Option<usize>,
        line: String,
        before_context: Vec<String>,
        after_context: Vec<String>,
    },
}

#[derive(Debug)]
pub struct HttpResponse {
    pub content: String,
    pub code: u16,
    pub context: ResponseContext,
    pub content_type: String,
}

#[derive(Debug)]
pub enum ResponseContext {
    Parsed,
    Raw,
}

#[derive(Debug)]
pub struct FsWriteOutput {
    pub path: String,
    // Set when the file already exists
    pub before: Option<String>,
    pub errors: Vec<SyntaxError>,
    pub content_hash: String,
}

#[derive(Debug)]
pub struct FsRemoveOutput {
    // Content of the file
    pub content: String,
}

#[derive(Debug)]
pub struct PlanCreateOutput {
    pub path: PathBuf,
    // Set when the file already exists
    pub before: Option<String>,
}

#[derive(Default, Debug, derive_more::From)]
pub struct FsUndoOutput {
    pub before_undo: Option<String>,
    pub after_undo: Option<String>,
}

/// Output from todo_write tool execution
#[derive(Debug)]
pub struct TodoWriteOutput {
    /// List of todos that were saved
    pub todos: Vec<forge_domain::Todo>,
}

#[derive(Debug)]
pub struct PolicyDecision {
    pub allowed: bool,
    pub path: Option<PathBuf>,
}

#[async_trait::async_trait]
pub trait ProviderService: Send + Sync {
    async fn chat(
        &self,
        model_id: &ModelId,
        context: Context,
        provider: Provider<Url>,
    ) -> ResultStream<ChatCompletionMessage, anyhow::Error>;
    async fn models(&self, provider: Provider<Url>) -> anyhow::Result<Vec<Model>>;
    async fn get_provider(&self, id: forge_domain::ProviderId) -> anyhow::Result<Provider<Url>>;
    async fn get_all_providers(&self) -> anyhow::Result<Vec<AnyProvider>>;
    async fn upsert_credential(
        &self,
        credential: forge_domain::AuthCredential,
    ) -> anyhow::Result<()>;
    async fn remove_credential(&self, id: &forge_domain::ProviderId) -> anyhow::Result<()>;
    /// Migrates environment variable-based credentials to file-based
    /// credentials. Returns Some(MigrationResult) if credentials were migrated,
    /// None if file already exists or no credentials to migrate.
    async fn migrate_env_credentials(
        &self,
    ) -> anyhow::Result<Option<forge_domain::MigrationResult>>;
}
/// Manages user preferences for default providers and models.
#[async_trait::async_trait]
pub trait AppConfigService: Send + Sync {
    /// Gets the current session configuration (provider and model pair).
    ///
    /// Returns `None` when no session has been configured yet.
    async fn get_session_config(&self) -> Option<forge_domain::ModelConfig>;

    /// Gets the commit configuration (provider and model for commit message
    /// generation).
    async fn get_commit_config(&self) -> anyhow::Result<Option<forge_domain::ModelConfig>>;

    /// Gets the suggest configuration (provider and model for command
    /// suggestion generation).
    async fn get_suggest_config(&self) -> anyhow::Result<Option<forge_domain::ModelConfig>>;

    /// Gets the current reasoning effort setting.
    async fn get_reasoning_effort(&self) -> anyhow::Result<Option<forge_domain::Effort>>;

    /// Applies one or more configuration mutations atomically.
    ///
    /// Each operation in `ops` is applied in order, and the result is
    /// persisted as a single atomic write. This is the sole write path for
    /// all configuration changes; use [`forge_domain::ConfigOperation`]
    /// variants to describe each mutation.
    async fn update_config(&self, ops: Vec<forge_domain::ConfigOperation>) -> anyhow::Result<()>;
}

#[async_trait::async_trait]
pub trait McpConfigManager: Send + Sync {
    /// Responsible to load the MCP servers from all configuration files.
    /// If scope is provided, only loads from that specific scope (not merged).
    async fn read_mcp_config(&self, scope: Option<&Scope>) -> anyhow::Result<McpConfig>;

    /// Responsible for writing the McpConfig on disk.
    async fn write_mcp_config(&self, config: &McpConfig, scope: &Scope) -> anyhow::Result<()>;
}

#[async_trait::async_trait]
pub trait McpService: Send + Sync {
    async fn get_mcp_servers(&self) -> anyhow::Result<McpServers>;
    async fn execute_mcp(&self, call: ToolCallFull) -> anyhow::Result<ToolOutput>;
    /// Refresh the MCP cache by fetching fresh data
    async fn reload_mcp(&self) -> anyhow::Result<()>;
}

#[async_trait::async_trait]
pub trait ConversationService: Send + Sync {
    /// Finds a conversation by ID.
    ///
    /// # Arguments
    /// * `id` - The conversation ID to retrieve.
    ///
    /// # Errors
    /// Returns an error if the lookup fails.
    async fn find_conversation(&self, id: &ConversationId) -> anyhow::Result<Option<Conversation>>;

    /// Creates or updates a conversation.
    ///
    /// # Arguments
    /// * `conversation` - The conversation to persist.
    ///
    /// # Errors
    /// Returns an error if persistence fails.
    async fn upsert_conversation(&self, conversation: Conversation) -> anyhow::Result<()>;

    /// Marks an existing conversation as delegated agent work and links it to
    /// the current parent conversation when available.
    ///
    /// # Arguments
    /// * `id` - The delegated conversation ID.
    /// * `parent_id` - The parent conversation that owns this delegated
    ///   session.
    ///
    /// # Errors
    /// Returns an error if the conversation is missing or ownership is invalid.
    async fn ensure_delegated_conversation(
        &self,
        id: &ConversationId,
        parent_id: Option<ConversationId>,
    ) -> anyhow::Result<Conversation>;

    /// Resolves the root conversation ID for a delegated parent chain.
    ///
    /// # Arguments
    /// * `parent_id` - The immediate parent conversation, when the task is
    ///   delegated.
    ///
    /// # Errors
    /// Returns an error if reading the persisted parent chain fails.
    async fn resolve_root_conversation_id(
        &self,
        parent_id: Option<ConversationId>,
    ) -> anyhow::Result<Option<ConversationId>>;

    /// This is useful when you want to perform several operations on a
    /// conversation atomically.
    ///
    /// # Arguments
    /// * `id` - The conversation ID to modify.
    /// * `f` - The mutation closure executed before persistence.
    ///
    /// # Errors
    /// Returns an error if the conversation is missing or persistence fails.
    async fn modify_conversation<F, T>(&self, id: &ConversationId, f: F) -> anyhow::Result<T>
    where
        F: FnOnce(&mut Conversation) -> T + Send,
        T: Send;

    /// Creates a branch-only conversation by excluding the selected message and
    /// everything after it. The source conversation is normalized and preserved.
    ///
    /// # Arguments
    /// * `conversation_id` - Source conversation ID.
    /// * `target_id` - Stable selectable message ID that defines the branch boundary.
    ///
    /// # Errors
    /// Returns an error when the source conversation or target message is missing,
    /// not selectable, or persistence fails.
    async fn branch_conversation(
        &self,
        conversation_id: &ConversationId,
        target_id: forge_domain::MessageId,
    ) -> anyhow::Result<Conversation>;

    /// Find primary user conversations.
    ///
    /// # Errors
    /// Returns an error if listing conversations fails.
    async fn get_conversations(&self) -> anyhow::Result<Vec<Conversation>>;

    /// Find root conversations including internal agent sessions for diagnostic
    /// list surfaces.
    ///
    /// # Errors
    /// Returns an error if listing conversations fails.
    async fn get_conversations_including_agent(&self) -> anyhow::Result<Vec<Conversation>>;

    /// Find sub-conversations (subagent chats) for a parent conversation.
    ///
    /// # Arguments
    /// * `parent_id` - The parent conversation ID.
    ///
    /// # Errors
    /// Returns an error if listing sub-conversations fails.
    async fn get_sub_conversations(
        &self,
        parent_id: &ConversationId,
    ) -> anyhow::Result<Vec<Conversation>>;

    /// Creates or updates a durable subagent lifecycle ledger record.
    ///
    /// # Arguments
    /// * `session` - The subagent task-session lifecycle record to persist.
    ///
    /// # Errors
    /// Returns an error if persistence fails.
    async fn upsert_subagent_task_session(
        &self,
        session: SubagentTaskSession,
    ) -> anyhow::Result<()>;

    /// Finds a durable subagent lifecycle ledger record by task ID.
    ///
    /// # Arguments
    /// * `task_id` - The durable task ID to retrieve.
    ///
    /// # Errors
    /// Returns an error if lookup fails.
    async fn get_subagent_task_session(
        &self,
        task_id: &SubagentTaskId,
    ) -> anyhow::Result<Option<SubagentTaskSession>>;

    /// Finds a durable subagent lifecycle ledger record by conversation ID.
    ///
    /// # Arguments
    /// * `conversation_id` - The delegated conversation/session ID to retrieve.
    ///
    /// # Errors
    /// Returns an error if lookup fails.
    async fn get_subagent_task_session_by_conversation(
        &self,
        conversation_id: &ConversationId,
    ) -> anyhow::Result<Option<SubagentTaskSession>>;

    /// Lists durable subagent lifecycle ledger records.
    ///
    /// # Arguments
    /// * `filter` - Selects active-only or all task sessions.
    ///
    /// # Errors
    /// Returns an error if listing task sessions fails.
    async fn list_subagent_task_sessions(
        &self,
        filter: SubagentTaskSessionFilter,
    ) -> anyhow::Result<Vec<SubagentTaskSession>>;

    /// Find the last active conversation.
    ///
    /// # Errors
    /// Returns an error if lookup fails.
    async fn last_conversation(&self) -> anyhow::Result<Option<Conversation>>;

    /// Permanently deletes a conversation.
    ///
    /// # Arguments
    /// * `conversation_id` - The conversation ID to delete.
    ///
    /// # Errors
    /// Returns an error if deletion fails.
    async fn delete_conversation(&self, conversation_id: &ConversationId) -> anyhow::Result<()>;
}

#[async_trait::async_trait]
pub trait LearningService: Send + Sync {
    /// Captures a redacted candidate learning event from conversation evidence.
    ///
    /// # Arguments
    /// * `conversation_id` - Source conversation identifier.
    /// * `source_event_id` - Stable source event identifier.
    /// * `summary` - Candidate summary to redact before persistence.
    ///
    /// # Errors
    /// Returns an error if validation or persistence fails.
    async fn capture_candidate_from_conversation(
        &self,
        conversation_id: ConversationId,
        source_event_id: String,
        summary: String,
    ) -> anyhow::Result<LearningLedgerEvent>;

    /// Inserts an append-only learning event.
    ///
    /// # Arguments
    /// * `event` - Event to append or deduplicate.
    ///
    /// # Errors
    /// Returns an error if validation or persistence fails.
    async fn insert_learning_event(
        &self,
        event: LearningLedgerEvent,
    ) -> anyhow::Result<LearningLedgerEvent>;

    /// Lists projected learning records.
    ///
    /// # Arguments
    /// * `review_state` - Optional review-state filter.
    /// * `limit` - Maximum number of records.
    ///
    /// # Errors
    /// Returns an error if query fails.
    async fn list_learning_records(
        &self,
        review_state: Option<LearningReviewState>,
        limit: usize,
    ) -> anyhow::Result<Vec<LearningRecordProjection>>;

    /// Returns learning ledger freshness for invalidating late-bound context.
    ///
    /// # Arguments
    /// * `review_state` - Optional review-state filter.
    ///
    /// # Errors
    /// Returns an error if query fails.
    async fn learning_freshness(
        &self,
        review_state: Option<LearningReviewState>,
    ) -> anyhow::Result<LearningLedgerFreshness>;
}

#[async_trait::async_trait]
pub trait SteerService: Send + Sync {
    /// Enqueues a typed steer message for delayed main-conversation delivery.
    ///
    /// # Arguments
    /// * `conversation_id` - The primary conversation receiving the message.
    /// * `message` - The typed steer message.
    async fn enqueue_steer(
        &self,
        conversation_id: &ConversationId,
        message: SteerMessage,
    ) -> anyhow::Result<()>;

    /// Clears queued steer messages without delivering them.
    ///
    /// # Arguments
    /// * `conversation_id` - The conversation whose queue should be discarded.
    async fn clear_steer(&self, conversation_id: &ConversationId) -> anyhow::Result<()>;

    /// Drains queued steer messages for a conversation in insertion order.
    ///
    /// # Arguments
    /// * `conversation_id` - The conversation whose queue should be drained.
    async fn drain_steer(
        &self,
        conversation_id: &ConversationId,
    ) -> anyhow::Result<Vec<SteerMessage>>;
}

#[async_trait::async_trait]
pub trait TemplateService: Send + Sync {
    async fn register_template(&self, path: PathBuf) -> anyhow::Result<()>;
    async fn render_template<V: serde::Serialize + Send + Sync>(
        &self,
        template: Template<V>,
        object: &V,
    ) -> anyhow::Result<String>;
}

#[async_trait::async_trait]
pub trait AttachmentService {
    async fn attachments(&self, url: &str) -> anyhow::Result<Vec<Attachment>>;
}

#[async_trait::async_trait]
pub trait CustomInstructionsService: Send + Sync {
    async fn get_custom_instructions(&self) -> Vec<String>;
}

/// Service for indexing workspaces for semantic search
#[async_trait::async_trait]
pub trait WorkspaceService: Send + Sync {
    /// Index the workspace at the given path
    async fn sync_workspace(
        &self,
        path: PathBuf,
    ) -> anyhow::Result<forge_stream::MpscStream<anyhow::Result<SyncProgress>>>;

    /// Produces one explicit bounded workspace exact-fact reference artifact.
    async fn produce_workspace_exact_fact_reference(
        &self,
        path: PathBuf,
    ) -> anyhow::Result<WorkspaceExactFactReferenceReport>;

    /// Reads persisted exact-fact status without producing or mutating artifacts.
    async fn workspace_exact_fact_status(
        &self,
        path: PathBuf,
    ) -> anyhow::Result<WorkspaceExactFactStatusReport>;

    /// Query the indexed workspace with semantic search
    async fn query_workspace(
        &self,
        path: PathBuf,
        params: SearchParams<'_>,
    ) -> anyhow::Result<Vec<Node>>;

    /// List all workspaces indexed by the user
    async fn list_workspaces(&self) -> anyhow::Result<Vec<WorkspaceInfo>>;

    /// Get workspace information for a specific path
    async fn get_workspace_info(&self, path: PathBuf) -> anyhow::Result<Option<WorkspaceInfo>>;

    /// Checks whether a path belongs to an indexed workspace.
    async fn is_indexed(&self, path: &Path) -> anyhow::Result<bool>;

    /// Delete a workspace and all its indexed data
    async fn delete_workspace(&self, workspace_id: &WorkspaceId) -> anyhow::Result<()>;

    /// Delete multiple workspaces in parallel and all their indexed data
    async fn delete_workspaces(&self, workspace_ids: &[WorkspaceId]) -> anyhow::Result<()>;

    /// Checks if workspace project-model context is available and fresh.
    async fn project_model_context_diagnostic(
        &self,
        path: &Path,
    ) -> anyhow::Result<WorkspaceContextManifestDiagnostic>;

    /// Get sync status for all files in workspace
    async fn get_workspace_status(&self, path: PathBuf) -> anyhow::Result<Vec<FileStatus>>;

    /// Check if authentication credentials exist
    async fn is_authenticated(&self) -> anyhow::Result<bool>;

    /// Create new authentication credentials
    async fn init_auth_credentials(&self) -> anyhow::Result<WorkspaceAuth>;

    /// Initialize a workspace without syncing files
    async fn init_workspace(&self, path: PathBuf) -> anyhow::Result<WorkspaceId>;
}

#[async_trait::async_trait]
pub trait FileDiscoveryService: Send + Sync {
    async fn collect_files(&self, config: Walker) -> anyhow::Result<Vec<File>>;

    /// Lists all entries (files and directories) in the current directory
    /// Returns a sorted vector of File entries with directories first
    async fn list_current_directory(&self) -> anyhow::Result<Vec<File>>;
}

#[async_trait::async_trait]
pub trait FsWriteService: Send + Sync {
    /// Create a file at the specified path with the given content.
    async fn write(
        &self,
        path: String,
        content: String,
        overwrite: bool,
    ) -> anyhow::Result<FsWriteOutput>;
}

#[async_trait::async_trait]
pub trait PlanCreateService: Send + Sync {
    /// Create a plan file with the specified name and version.
    async fn create_plan(
        &self,
        plan_name: String,
        version: String,
        content: String,
    ) -> anyhow::Result<PlanCreateOutput>;
}

#[async_trait::async_trait]
pub trait FsPatchService: Send + Sync {
    /// Patches a file at the specified path with the given content.
    async fn patch(
        &self,
        path: String,
        search: String,
        content: String,
        replace_all: bool,
    ) -> anyhow::Result<PatchOutput>;

    /// Applies multiple patches to a single file in sequence
    async fn multi_patch(
        &self,
        path: String,
        edits: Vec<forge_domain::PatchEdit>,
    ) -> anyhow::Result<PatchOutput>;
}

#[async_trait::async_trait]
pub trait FsReadService: Send + Sync {
    /// Reads a file at the specified path and returns its content.
    async fn read(
        &self,
        path: String,
        start_line: Option<u64>,
        end_line: Option<u64>,
    ) -> anyhow::Result<ReadOutput>;
}

#[async_trait::async_trait]
pub trait ImageReadService: Send + Sync {
    /// Reads an image file at the specified path and returns its content.
    async fn read_image(&self, path: String) -> anyhow::Result<forge_domain::Image>;
}

#[async_trait::async_trait]
pub trait FsRemoveService: Send + Sync {
    /// Removes a file at the specified path.
    async fn remove(&self, path: String) -> anyhow::Result<FsRemoveOutput>;
}

#[async_trait::async_trait]
pub trait FsSearchService: Send + Sync {
    /// Searches for files and content based on the provided parameters.
    ///
    /// # Arguments
    /// * `params` - Search parameters including pattern, path, output mode,
    ///   etc.
    ///
    /// # Returns
    /// * `Ok(Some(SearchResult))` - Matches found
    /// * `Ok(None)` - No matches found
    /// * `Err(_)` - Search error
    async fn search(&self, params: forge_domain::FSSearch) -> anyhow::Result<Option<SearchResult>>;
}

#[async_trait::async_trait]
pub trait FollowUpService: Send + Sync {
    /// Follows up on a tool call with the given context.
    async fn follow_up(
        &self,
        question: String,
        options: Vec<String>,
        multiple: Option<bool>,
    ) -> anyhow::Result<Option<String>>;
}

#[async_trait::async_trait]
pub trait FsUndoService: Send + Sync {
    /// Undoes the last file operation at the specified path.
    /// And returns the content of the undone file.
    async fn undo(&self, path: String) -> anyhow::Result<FsUndoOutput>;
}

#[async_trait::async_trait]
pub trait NetFetchService: Send + Sync {
    /// Fetches content from a URL and returns it as a string.
    async fn fetch(&self, url: String, raw: Option<bool>) -> anyhow::Result<HttpResponse>;
}

/// Typed request for executing a shell command through the shell service.
pub struct ShellExecuteRequest {
    /// Shell command text to execute.
    pub command: String,
    /// Working directory for command execution.
    pub cwd: PathBuf,
    /// Whether ANSI escape codes should be preserved.
    pub keep_ansi: bool,
    /// Whether command output should be suppressed from console display.
    pub silent: bool,
    /// Environment variable names copied from the current process.
    pub env_vars: Option<Vec<String>>,
    /// Synchronous wait window before background process handoff.
    pub handoff_timeout: forge_domain::ShellHandoffTimeoutSeconds,
    /// Human-readable command description.
    pub description: Option<String>,
}

#[async_trait::async_trait]
pub trait ShellService: Send + Sync {
    /// Executes a shell command and returns the output.
    async fn execute(&self, request: ShellExecuteRequest) -> anyhow::Result<ShellOutput>;

    /// Starts a managed background process and returns its handle immediately.
    async fn process_start(
        &self,
        _command: String,
        _cwd: PathBuf,
        _env_vars: Option<Vec<String>>,
        _description: Option<String>,
    ) -> anyhow::Result<ProcessStartServiceOutput> {
        anyhow::bail!("Managed background processes are not supported by this shell service")
    }

    /// Returns status for a managed background process.
    async fn process_status(
        &self,
        _process_id: ProcessId,
        _wait: Option<forge_domain::ProcessObservationWaitSeconds>,
    ) -> anyhow::Result<ProcessOutput> {
        anyhow::bail!("Managed background processes are not supported by this shell service")
    }

    /// Reads captured output from a managed background process.
    async fn process_read(
        &self,
        _process_id: ProcessId,
        _cursor: ProcessReadCursor,
        _wait: Option<forge_domain::ProcessObservationWaitSeconds>,
    ) -> anyhow::Result<ProcessReadServiceOutput> {
        anyhow::bail!("Managed background processes are not supported by this shell service")
    }

    /// Lists currently running managed background process statuses.
    async fn process_list(&self) -> anyhow::Result<Vec<ProcessStatus>> {
        anyhow::bail!("Managed background processes are not supported by this shell service")
    }

    /// Stops a managed background process.
    async fn process_kill(
        &self,
        _process_id: ProcessId,
    ) -> anyhow::Result<ProcessKillServiceOutput> {
        anyhow::bail!("Managed background processes are not supported by this shell service")
    }
}

#[async_trait::async_trait]
pub trait AuthService: Send + Sync {
    async fn user_info(&self, api_key: &str) -> anyhow::Result<User>;
    async fn user_usage(&self, api_key: &str) -> anyhow::Result<UserUsage>;
}

#[async_trait::async_trait]
pub trait AgentRegistry: Send + Sync {
    /// Get the active agent ID
    async fn get_active_agent_id(&self) -> anyhow::Result<Option<AgentId>>;

    /// Set the active agent ID
    async fn set_active_agent_id(&self, agent_id: AgentId) -> anyhow::Result<()>;

    /// Get all agents from the registry store
    async fn get_agents(&self) -> anyhow::Result<Vec<forge_domain::Agent>>;

    /// Get lightweight metadata for all agents without requiring a configured
    /// provider or model
    async fn get_agent_infos(&self) -> anyhow::Result<Vec<forge_domain::AgentInfo>>;

    /// Get agent by ID (from registry store)
    async fn get_agent(&self, agent_id: &AgentId) -> anyhow::Result<Option<forge_domain::Agent>>;

    /// Reload agents by invalidating the cache
    async fn reload_agents(&self) -> anyhow::Result<()>;
}

#[async_trait::async_trait]
pub trait CommandLoaderService: Send + Sync {
    /// Load all command definitions from the forge/commands directory
    async fn get_commands(&self) -> anyhow::Result<Vec<forge_domain::Command>>;
}

#[async_trait::async_trait]
pub trait PolicyService: Send + Sync {
    /// Check if an operation is allowed and handle user confirmation if needed
    /// Returns PolicyDecision with allowed flag and optional policy file path
    /// (only when created)
    async fn check_operation_permission(
        &self,
        operation: &forge_domain::PermissionOperation,
    ) -> anyhow::Result<PolicyDecision>;
}

/// Skill fetch service
#[async_trait::async_trait]
pub trait SkillFetchService: Send + Sync {
    /// Fetches a skill by name
    ///
    /// # Errors
    ///
    /// Returns an error if the skill is not found or cannot be loaded
    async fn fetch_skill(&self, skill_name: String) -> anyhow::Result<forge_domain::Skill>;

    /// Lists all available skills
    ///
    /// # Errors
    ///
    /// Returns an error if skills cannot be loaded
    async fn list_skills(&self) -> anyhow::Result<Vec<forge_domain::Skill>>;
}

/// Provider authentication service
#[async_trait::async_trait]
pub trait ProviderAuthService: Send + Sync {
    async fn init_provider_auth(
        &self,
        provider_id: ProviderId,
        method: AuthMethod,
    ) -> anyhow::Result<AuthContextRequest>;
    async fn complete_provider_auth(
        &self,
        provider_id: ProviderId,
        context: AuthContextResponse,
        timeout: Duration,
    ) -> anyhow::Result<()>;

    /// Refreshes provider credentials if they're about to expire.
    /// Checks if credential needs refresh (5 minute buffer before expiry),
    /// iterates through provider's auth methods, and attempts to refresh.
    /// Returns the provider with updated credentials, or original if refresh
    /// fails or isn't needed.
    async fn refresh_provider_credential(
        &self,
        provider: Provider<Url>,
    ) -> anyhow::Result<Provider<Url>>;
}

pub trait Services: Send + Sync + 'static + Clone + EnvironmentInfra {
    type ProviderService: ProviderService;
    type AppConfigService: AppConfigService;
    type ConversationService: ConversationService;
    type LearningService: LearningService;
    type SteerService: SteerService;
    type TemplateService: TemplateService;
    type AttachmentService: AttachmentService;
    type CustomInstructionsService: CustomInstructionsService;
    type FileDiscoveryService: FileDiscoveryService;
    type McpConfigManager: McpConfigManager;
    type FsWriteService: FsWriteService;
    type PlanCreateService: PlanCreateService;
    type FsPatchService: FsPatchService;
    type FsReadService: FsReadService;
    type ImageReadService: ImageReadService;
    type FsRemoveService: FsRemoveService;
    type FsSearchService: FsSearchService;
    type FollowUpService: FollowUpService;
    type FsUndoService: FsUndoService;
    type NetFetchService: NetFetchService;
    type ShellService: ShellService;
    type McpService: McpService;
    type AuthService: AuthService;
    type AgentRegistry: AgentRegistry;
    type CommandLoaderService: CommandLoaderService;
    type PolicyService: PolicyService;
    type ProviderAuthService: ProviderAuthService;
    type WorkspaceService: WorkspaceService;
    type SkillFetchService: SkillFetchService;

    fn provider_service(&self) -> &Self::ProviderService;
    fn config_service(&self) -> &Self::AppConfigService;
    fn conversation_service(&self) -> &Self::ConversationService;
    fn learning_service(&self) -> &Self::LearningService;
    fn steer_service(&self) -> &Self::SteerService;
    fn template_service(&self) -> &Self::TemplateService;
    fn attachment_service(&self) -> &Self::AttachmentService;
    fn file_discovery_service(&self) -> &Self::FileDiscoveryService;
    fn mcp_config_manager(&self) -> &Self::McpConfigManager;
    fn fs_create_service(&self) -> &Self::FsWriteService;
    fn plan_create_service(&self) -> &Self::PlanCreateService;
    fn fs_patch_service(&self) -> &Self::FsPatchService;
    fn fs_read_service(&self) -> &Self::FsReadService;
    fn image_read_service(&self) -> &Self::ImageReadService;
    fn fs_remove_service(&self) -> &Self::FsRemoveService;
    fn fs_search_service(&self) -> &Self::FsSearchService;
    fn follow_up_service(&self) -> &Self::FollowUpService;
    fn fs_undo_service(&self) -> &Self::FsUndoService;
    fn net_fetch_service(&self) -> &Self::NetFetchService;
    fn shell_service(&self) -> &Self::ShellService;
    fn mcp_service(&self) -> &Self::McpService;
    fn custom_instructions_service(&self) -> &Self::CustomInstructionsService;
    fn auth_service(&self) -> &Self::AuthService;
    fn agent_registry(&self) -> &Self::AgentRegistry;
    fn command_loader_service(&self) -> &Self::CommandLoaderService;
    fn policy_service(&self) -> &Self::PolicyService;
    fn provider_auth_service(&self) -> &Self::ProviderAuthService;
    fn workspace_service(&self) -> &Self::WorkspaceService;
    fn skill_fetch_service(&self) -> &Self::SkillFetchService;
}

#[async_trait::async_trait]
impl<I: Services> ConversationService for I {
    async fn find_conversation(&self, id: &ConversationId) -> anyhow::Result<Option<Conversation>> {
        self.conversation_service().find_conversation(id).await
    }

    async fn upsert_conversation(&self, conversation: Conversation) -> anyhow::Result<()> {
        self.conversation_service()
            .upsert_conversation(conversation)
            .await
    }

    async fn ensure_delegated_conversation(
        &self,
        id: &ConversationId,
        parent_id: Option<ConversationId>,
    ) -> anyhow::Result<Conversation> {
        let conversation = self
            .conversation_service()
            .ensure_delegated_conversation(id, parent_id)
            .await?;
        self.steer_service().clear_steer(id).await?;
        Ok(conversation)
    }

    async fn resolve_root_conversation_id(
        &self,
        parent_id: Option<ConversationId>,
    ) -> anyhow::Result<Option<ConversationId>> {
        self.conversation_service()
            .resolve_root_conversation_id(parent_id)
            .await
    }

    async fn modify_conversation<F, T>(&self, id: &ConversationId, f: F) -> anyhow::Result<T>
    where
        F: FnOnce(&mut Conversation) -> T + Send,
        T: Send,
    {
        self.conversation_service().modify_conversation(id, f).await
    }

    async fn branch_conversation(
        &self,
        conversation_id: &ConversationId,
        target_id: forge_domain::MessageId,
    ) -> anyhow::Result<Conversation> {
        self.conversation_service()
            .branch_conversation(conversation_id, target_id)
            .await
    }

    async fn get_conversations(&self) -> anyhow::Result<Vec<Conversation>> {
        self.conversation_service().get_conversations().await
    }

    async fn get_conversations_including_agent(&self) -> anyhow::Result<Vec<Conversation>> {
        self.conversation_service()
            .get_conversations_including_agent()
            .await
    }

    async fn get_sub_conversations(
        &self,
        parent_id: &ConversationId,
    ) -> anyhow::Result<Vec<Conversation>> {
        self.conversation_service()
            .get_sub_conversations(parent_id)
            .await
    }

    async fn upsert_subagent_task_session(
        &self,
        session: SubagentTaskSession,
    ) -> anyhow::Result<()> {
        self.conversation_service()
            .upsert_subagent_task_session(session)
            .await
    }

    async fn get_subagent_task_session(
        &self,
        task_id: &SubagentTaskId,
    ) -> anyhow::Result<Option<SubagentTaskSession>> {
        self.conversation_service()
            .get_subagent_task_session(task_id)
            .await
    }

    async fn get_subagent_task_session_by_conversation(
        &self,
        conversation_id: &ConversationId,
    ) -> anyhow::Result<Option<SubagentTaskSession>> {
        self.conversation_service()
            .get_subagent_task_session_by_conversation(conversation_id)
            .await
    }

    async fn list_subagent_task_sessions(
        &self,
        filter: SubagentTaskSessionFilter,
    ) -> anyhow::Result<Vec<SubagentTaskSession>> {
        self.conversation_service()
            .list_subagent_task_sessions(filter)
            .await
    }

    async fn last_conversation(&self) -> anyhow::Result<Option<Conversation>> {
        self.conversation_service().last_conversation().await
    }

    async fn delete_conversation(&self, conversation_id: &ConversationId) -> anyhow::Result<()> {
        self.conversation_service()
            .delete_conversation(conversation_id)
            .await
    }
}

#[async_trait::async_trait]
impl<I: Services> LearningService for I {
    async fn capture_candidate_from_conversation(
        &self,
        conversation_id: ConversationId,
        source_event_id: String,
        summary: String,
    ) -> anyhow::Result<LearningLedgerEvent> {
        self.learning_service()
            .capture_candidate_from_conversation(conversation_id, source_event_id, summary)
            .await
    }

    async fn insert_learning_event(
        &self,
        event: LearningLedgerEvent,
    ) -> anyhow::Result<LearningLedgerEvent> {
        self.learning_service().insert_learning_event(event).await
    }

    async fn list_learning_records(
        &self,
        review_state: Option<LearningReviewState>,
        limit: usize,
    ) -> anyhow::Result<Vec<LearningRecordProjection>> {
        self.learning_service()
            .list_learning_records(review_state, limit)
            .await
    }

    async fn learning_freshness(
        &self,
        review_state: Option<LearningReviewState>,
    ) -> anyhow::Result<LearningLedgerFreshness> {
        self.learning_service()
            .learning_freshness(review_state)
            .await
    }
}
#[async_trait::async_trait]
impl<I: Services> SteerService for I {
    async fn enqueue_steer(
        &self,
        conversation_id: &ConversationId,
        message: SteerMessage,
    ) -> anyhow::Result<()> {
        self.steer_service()
            .enqueue_steer(conversation_id, message)
            .await
    }

    async fn clear_steer(&self, conversation_id: &ConversationId) -> anyhow::Result<()> {
        self.steer_service().clear_steer(conversation_id).await
    }

    async fn drain_steer(
        &self,
        conversation_id: &ConversationId,
    ) -> anyhow::Result<Vec<SteerMessage>> {
        self.steer_service().drain_steer(conversation_id).await
    }
}

#[async_trait::async_trait]
impl<I: Services> ProviderService for I {
    async fn chat(
        &self,
        model_id: &ModelId,
        context: Context,
        provider: Provider<Url>,
    ) -> ResultStream<ChatCompletionMessage, anyhow::Error> {
        self.provider_service()
            .chat(model_id, context, provider)
            .await
    }

    async fn models(&self, provider: Provider<Url>) -> anyhow::Result<Vec<Model>> {
        self.provider_service().models(provider).await
    }

    async fn get_provider(&self, id: forge_domain::ProviderId) -> anyhow::Result<Provider<Url>> {
        self.provider_service().get_provider(id).await
    }

    async fn get_all_providers(&self) -> anyhow::Result<Vec<AnyProvider>> {
        self.provider_service().get_all_providers().await
    }

    async fn upsert_credential(
        &self,
        credential: forge_domain::AuthCredential,
    ) -> anyhow::Result<()> {
        self.provider_service().upsert_credential(credential).await
    }

    async fn remove_credential(&self, id: &forge_domain::ProviderId) -> anyhow::Result<()> {
        self.provider_service().remove_credential(id).await
    }

    async fn migrate_env_credentials(
        &self,
    ) -> anyhow::Result<Option<forge_domain::MigrationResult>> {
        self.provider_service().migrate_env_credentials().await
    }
}

#[async_trait::async_trait]
impl<I: Services> McpConfigManager for I {
    async fn read_mcp_config(&self, scope: Option<&Scope>) -> anyhow::Result<McpConfig> {
        self.mcp_config_manager().read_mcp_config(scope).await
    }

    async fn write_mcp_config(&self, config: &McpConfig, scope: &Scope) -> anyhow::Result<()> {
        self.mcp_config_manager()
            .write_mcp_config(config, scope)
            .await
    }
}

#[async_trait::async_trait]
impl<I: Services> McpService for I {
    async fn get_mcp_servers(&self) -> anyhow::Result<McpServers> {
        self.mcp_service().get_mcp_servers().await
    }

    async fn execute_mcp(&self, call: ToolCallFull) -> anyhow::Result<ToolOutput> {
        self.mcp_service().execute_mcp(call).await
    }

    async fn reload_mcp(&self) -> anyhow::Result<()> {
        self.mcp_service().reload_mcp().await
    }
}

#[async_trait::async_trait]
impl<I: Services> TemplateService for I {
    async fn register_template(&self, path: PathBuf) -> anyhow::Result<()> {
        self.template_service().register_template(path).await
    }

    async fn render_template<V: serde::Serialize + Send + Sync>(
        &self,
        template: Template<V>,
        object: &V,
    ) -> anyhow::Result<String> {
        self.template_service()
            .render_template(template, object)
            .await
    }
}

#[async_trait::async_trait]
impl<I: Services> AttachmentService for I {
    async fn attachments(&self, url: &str) -> anyhow::Result<Vec<Attachment>> {
        self.attachment_service().attachments(url).await
    }
}

#[async_trait::async_trait]
impl<I: Services> FileDiscoveryService for I {
    async fn collect_files(&self, config: Walker) -> anyhow::Result<Vec<File>> {
        self.file_discovery_service().collect_files(config).await
    }

    async fn list_current_directory(&self) -> anyhow::Result<Vec<File>> {
        self.file_discovery_service().list_current_directory().await
    }
}

#[async_trait::async_trait]
impl<I: Services> FsWriteService for I {
    async fn write(
        &self,
        path: String,
        content: String,
        overwrite: bool,
    ) -> anyhow::Result<FsWriteOutput> {
        self.fs_create_service()
            .write(path, content, overwrite)
            .await
    }
}

#[async_trait::async_trait]
impl<I: Services> PlanCreateService for I {
    async fn create_plan(
        &self,
        plan_name: String,
        version: String,
        content: String,
    ) -> anyhow::Result<PlanCreateOutput> {
        self.plan_create_service()
            .create_plan(plan_name, version, content)
            .await
    }
}

#[async_trait::async_trait]
impl<I: Services> FsPatchService for I {
    async fn patch(
        &self,
        path: String,
        search: String,
        content: String,
        replace_all: bool,
    ) -> anyhow::Result<PatchOutput> {
        self.fs_patch_service()
            .patch(path, search, content, replace_all)
            .await
    }

    async fn multi_patch(
        &self,
        path: String,
        edits: Vec<forge_domain::PatchEdit>,
    ) -> anyhow::Result<PatchOutput> {
        self.fs_patch_service().multi_patch(path, edits).await
    }
}

#[async_trait::async_trait]
impl<I: Services> FsReadService for I {
    async fn read(
        &self,
        path: String,
        start_line: Option<u64>,
        end_line: Option<u64>,
    ) -> anyhow::Result<ReadOutput> {
        self.fs_read_service()
            .read(path, start_line, end_line)
            .await
    }
}
#[async_trait::async_trait]
impl<I: Services> ImageReadService for I {
    async fn read_image(&self, path: String) -> anyhow::Result<Image> {
        self.image_read_service().read_image(path).await
    }
}

#[async_trait::async_trait]
impl<I: Services> FsRemoveService for I {
    async fn remove(&self, path: String) -> anyhow::Result<FsRemoveOutput> {
        self.fs_remove_service().remove(path).await
    }
}

#[async_trait::async_trait]
impl<I: Services> FsSearchService for I {
    async fn search(&self, params: forge_domain::FSSearch) -> anyhow::Result<Option<SearchResult>> {
        self.fs_search_service().search(params).await
    }
}

#[async_trait::async_trait]
impl<I: Services> FollowUpService for I {
    async fn follow_up(
        &self,
        question: String,
        options: Vec<String>,
        multiple: Option<bool>,
    ) -> anyhow::Result<Option<String>> {
        self.follow_up_service()
            .follow_up(question, options, multiple)
            .await
    }
}

#[async_trait::async_trait]
impl<I: Services> FsUndoService for I {
    async fn undo(&self, path: String) -> anyhow::Result<FsUndoOutput> {
        self.fs_undo_service().undo(path).await
    }
}

#[async_trait::async_trait]
impl<I: Services> NetFetchService for I {
    async fn fetch(&self, url: String, raw: Option<bool>) -> anyhow::Result<HttpResponse> {
        self.net_fetch_service().fetch(url, raw).await
    }
}

#[async_trait::async_trait]
impl<I: Services> ShellService for I {
    async fn execute(&self, request: ShellExecuteRequest) -> anyhow::Result<ShellOutput> {
        self.shell_service().execute(request).await
    }

    async fn process_start(
        &self,
        command: String,
        cwd: PathBuf,
        env_vars: Option<Vec<String>>,
        description: Option<String>,
    ) -> anyhow::Result<ProcessStartServiceOutput> {
        self.shell_service()
            .process_start(command, cwd, env_vars, description)
            .await
    }

    async fn process_status(
        &self,
        process_id: ProcessId,
        wait: Option<forge_domain::ProcessObservationWaitSeconds>,
    ) -> anyhow::Result<ProcessOutput> {
        self.shell_service().process_status(process_id, wait).await
    }

    async fn process_read(
        &self,
        process_id: ProcessId,
        cursor: ProcessReadCursor,
        wait: Option<forge_domain::ProcessObservationWaitSeconds>,
    ) -> anyhow::Result<ProcessReadServiceOutput> {
        self.shell_service()
            .process_read(process_id, cursor, wait)
            .await
    }

    async fn process_list(&self) -> anyhow::Result<Vec<ProcessStatus>> {
        self.shell_service().process_list().await
    }

    async fn process_kill(
        &self,
        process_id: ProcessId,
    ) -> anyhow::Result<ProcessKillServiceOutput> {
        self.shell_service().process_kill(process_id).await
    }
}

#[async_trait::async_trait]
impl<I: Services> CustomInstructionsService for I {
    async fn get_custom_instructions(&self) -> Vec<String> {
        self.custom_instructions_service()
            .get_custom_instructions()
            .await
    }
}

#[async_trait::async_trait]
impl<I: Services> AuthService for I {
    async fn user_info(&self, api_key: &str) -> anyhow::Result<User> {
        self.auth_service().user_info(api_key).await
    }

    async fn user_usage(&self, api_key: &str) -> anyhow::Result<UserUsage> {
        self.auth_service().user_usage(api_key).await
    }
}

/// HTTP service trait for making HTTP requests
#[async_trait::async_trait]
pub trait HttpClientService: Send + Sync + 'static {
    async fn get(&self, url: &Url, headers: Option<HeaderMap>) -> anyhow::Result<Response>;
    async fn post(&self, url: &Url, body: bytes::Bytes) -> anyhow::Result<Response>;
    async fn delete(&self, url: &Url) -> anyhow::Result<Response>;

    /// Posts JSON data and returns a server-sent events stream
    async fn eventsource(
        &self,
        url: &Url,
        headers: Option<HeaderMap>,
        body: Bytes,
    ) -> anyhow::Result<EventSource>;
}

#[async_trait::async_trait]
impl<I: Services> AgentRegistry for I {
    async fn get_active_agent_id(&self) -> anyhow::Result<Option<AgentId>> {
        self.agent_registry().get_active_agent_id().await
    }

    async fn set_active_agent_id(&self, agent_id: AgentId) -> anyhow::Result<()> {
        self.agent_registry().set_active_agent_id(agent_id).await
    }

    async fn get_agents(&self) -> anyhow::Result<Vec<forge_domain::Agent>> {
        self.agent_registry().get_agents().await
    }

    async fn get_agent_infos(&self) -> anyhow::Result<Vec<forge_domain::AgentInfo>> {
        self.agent_registry().get_agent_infos().await
    }

    async fn get_agent(&self, agent_id: &AgentId) -> anyhow::Result<Option<forge_domain::Agent>> {
        self.agent_registry().get_agent(agent_id).await
    }

    async fn reload_agents(&self) -> anyhow::Result<()> {
        self.agent_registry().reload_agents().await
    }
}

#[async_trait::async_trait]
impl<I: Services> CommandLoaderService for I {
    async fn get_commands(&self) -> anyhow::Result<Vec<forge_domain::Command>> {
        self.command_loader_service().get_commands().await
    }
}

#[async_trait::async_trait]
impl<I: Services> PolicyService for I {
    async fn check_operation_permission(
        &self,
        operation: &forge_domain::PermissionOperation,
    ) -> anyhow::Result<PolicyDecision> {
        self.policy_service()
            .check_operation_permission(operation)
            .await
    }
}

#[async_trait::async_trait]
impl<I: Services> AppConfigService for I {
    async fn get_session_config(&self) -> Option<forge_domain::ModelConfig> {
        self.config_service().get_session_config().await
    }

    async fn get_commit_config(&self) -> anyhow::Result<Option<forge_domain::ModelConfig>> {
        self.config_service().get_commit_config().await
    }

    async fn get_suggest_config(&self) -> anyhow::Result<Option<forge_domain::ModelConfig>> {
        self.config_service().get_suggest_config().await
    }

    async fn get_reasoning_effort(&self) -> anyhow::Result<Option<forge_domain::Effort>> {
        self.config_service().get_reasoning_effort().await
    }

    async fn update_config(&self, ops: Vec<forge_domain::ConfigOperation>) -> anyhow::Result<()> {
        self.config_service().update_config(ops).await
    }
}

#[async_trait::async_trait]
impl<I: Services> SkillFetchService for I {
    async fn fetch_skill(&self, skill_name: String) -> anyhow::Result<forge_domain::Skill> {
        self.skill_fetch_service().fetch_skill(skill_name).await
    }

    async fn list_skills(&self) -> anyhow::Result<Vec<forge_domain::Skill>> {
        self.skill_fetch_service().list_skills().await
    }
}

#[async_trait::async_trait]
impl<I: Services> ProviderAuthService for I {
    async fn init_provider_auth(
        &self,
        provider_id: ProviderId,
        method: AuthMethod,
    ) -> anyhow::Result<AuthContextRequest> {
        self.provider_auth_service()
            .init_provider_auth(provider_id, method)
            .await
    }
    async fn complete_provider_auth(
        &self,
        provider_id: ProviderId,
        context: AuthContextResponse,
        timeout: Duration,
    ) -> anyhow::Result<()> {
        self.provider_auth_service()
            .complete_provider_auth(provider_id, context, timeout)
            .await
    }
    async fn refresh_provider_credential(
        &self,
        provider: Provider<Url>,
    ) -> anyhow::Result<Provider<Url>> {
        self.provider_auth_service()
            .refresh_provider_credential(provider)
            .await
    }
}

#[async_trait::async_trait]
impl<I: Services> WorkspaceService for I {
    async fn sync_workspace(
        &self,
        path: PathBuf,
    ) -> anyhow::Result<forge_stream::MpscStream<anyhow::Result<SyncProgress>>> {
        self.workspace_service().sync_workspace(path).await
    }

    async fn produce_workspace_exact_fact_reference(
        &self,
        path: PathBuf,
    ) -> anyhow::Result<WorkspaceExactFactReferenceReport> {
        self.workspace_service()
            .produce_workspace_exact_fact_reference(path)
            .await
    }

    async fn workspace_exact_fact_status(
        &self,
        path: PathBuf,
    ) -> anyhow::Result<WorkspaceExactFactStatusReport> {
        self.workspace_service()
            .workspace_exact_fact_status(path)
            .await
    }

    async fn query_workspace(
        &self,
        path: PathBuf,
        params: SearchParams<'_>,
    ) -> anyhow::Result<Vec<Node>> {
        self.workspace_service().query_workspace(path, params).await
    }

    async fn list_workspaces(&self) -> anyhow::Result<Vec<WorkspaceInfo>> {
        self.workspace_service().list_workspaces().await
    }

    async fn get_workspace_info(&self, path: PathBuf) -> anyhow::Result<Option<WorkspaceInfo>> {
        self.workspace_service().get_workspace_info(path).await
    }

    async fn is_indexed(&self, path: &Path) -> anyhow::Result<bool> {
        self.workspace_service().is_indexed(path).await
    }

    async fn delete_workspace(&self, workspace_id: &WorkspaceId) -> anyhow::Result<()> {
        self.workspace_service()
            .delete_workspace(workspace_id)
            .await
    }

    async fn delete_workspaces(&self, workspace_ids: &[WorkspaceId]) -> anyhow::Result<()> {
        self.workspace_service()
            .delete_workspaces(workspace_ids)
            .await
    }

    async fn project_model_context_diagnostic(
        &self,
        path: &Path,
    ) -> anyhow::Result<WorkspaceContextManifestDiagnostic> {
        self.workspace_service()
            .project_model_context_diagnostic(path)
            .await
    }

    async fn get_workspace_status(&self, path: PathBuf) -> anyhow::Result<Vec<FileStatus>> {
        self.workspace_service().get_workspace_status(path).await
    }

    async fn is_authenticated(&self) -> anyhow::Result<bool> {
        self.workspace_service().is_authenticated().await
    }

    async fn init_auth_credentials(&self) -> anyhow::Result<WorkspaceAuth> {
        self.workspace_service().init_auth_credentials().await
    }

    async fn init_workspace(&self, path: PathBuf) -> anyhow::Result<WorkspaceId> {
        self.workspace_service().init_workspace(path).await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Arc;

    use forge_domain::{Context, Initiator, SteerQueue};
    use pretty_assertions::assert_eq;
    use tokio::sync::Mutex;

    use super::*;

    #[derive(Default)]
    struct RawConversationService {
        conversations: Mutex<HashMap<ConversationId, Conversation>>,
    }

    #[async_trait::async_trait]
    impl ConversationService for RawConversationService {
        async fn find_conversation(
            &self,
            id: &ConversationId,
        ) -> anyhow::Result<Option<Conversation>> {
            Ok(self.conversations.lock().await.get(id).cloned())
        }

        async fn upsert_conversation(&self, conversation: Conversation) -> anyhow::Result<()> {
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
        ) -> anyhow::Result<Conversation> {
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
        ) -> anyhow::Result<Option<ConversationId>> {
            let Some(mut current_id) = parent_id else {
                return Ok(None);
            };
            let mut root_id = current_id;
            let mut seen = std::collections::HashSet::new();
            while seen.insert(current_id) {
                let conversations = self.conversations.lock().await;
                let Some(parent) = conversations.get(&current_id) else {
                    break;
                };
                let Some(next_parent_id) = parent.parent_id else {
                    break;
                };
                drop(conversations);
                root_id = next_parent_id;
                current_id = next_parent_id;
            }
            Ok(Some(root_id))
        }

        async fn modify_conversation<F, T>(&self, id: &ConversationId, f: F) -> anyhow::Result<T>
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
        ) -> anyhow::Result<Conversation> {
            anyhow::bail!("branch conversation is not implemented for raw fixture service")
        }

        async fn get_conversations(&self) -> anyhow::Result<Vec<Conversation>> {
            Ok(self.conversations.lock().await.values().cloned().collect())
        }

        async fn get_conversations_including_agent(&self) -> anyhow::Result<Vec<Conversation>> {
            self.get_conversations().await
        }

        async fn get_sub_conversations(
            &self,
            parent_id: &ConversationId,
        ) -> anyhow::Result<Vec<Conversation>> {
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
            Ok(self.conversations.lock().await.values().next().cloned())
        }

        async fn delete_conversation(
            &self,
            conversation_id: &ConversationId,
        ) -> anyhow::Result<()> {
            self.conversations.lock().await.remove(conversation_id);
            Ok(())
        }
    }

    #[derive(Default)]
    struct RawSteerService {
        queues: Mutex<HashMap<ConversationId, SteerQueue>>,
        clear_count: Mutex<usize>,
    }

    #[async_trait::async_trait]
    impl SteerService for RawSteerService {
        async fn enqueue_steer(
            &self,
            conversation_id: &ConversationId,
            message: SteerMessage,
        ) -> anyhow::Result<()> {
            self.queues
                .lock()
                .await
                .entry(*conversation_id)
                .or_default()
                .push(message);
            Ok(())
        }

        async fn clear_steer(&self, conversation_id: &ConversationId) -> anyhow::Result<()> {
            *self.clear_count.lock().await += 1;
            self.queues.lock().await.remove(conversation_id);
            Ok(())
        }

        async fn drain_steer(
            &self,
            conversation_id: &ConversationId,
        ) -> anyhow::Result<Vec<SteerMessage>> {
            Ok(self
                .queues
                .lock()
                .await
                .remove(conversation_id)
                .map(|mut queue| queue.drain().collect())
                .unwrap_or_default())
        }
    }

    #[derive(Default)]
    struct NoopService;

    #[async_trait::async_trait]
    impl LearningService for NoopService {
        async fn capture_candidate_from_conversation(
            &self,
            _conversation_id: ConversationId,
            _source_event_id: String,
            _summary: String,
        ) -> anyhow::Result<LearningLedgerEvent> {
            anyhow::bail!("unused learning service")
        }

        async fn insert_learning_event(
            &self,
            _event: LearningLedgerEvent,
        ) -> anyhow::Result<LearningLedgerEvent> {
            anyhow::bail!("unused learning service")
        }

        async fn list_learning_records(
            &self,
            _review_state: Option<LearningReviewState>,
            _limit: usize,
        ) -> anyhow::Result<Vec<LearningRecordProjection>> {
            anyhow::bail!("unused learning service")
        }

        async fn learning_freshness(
            &self,
            _review_state: Option<LearningReviewState>,
        ) -> anyhow::Result<LearningLedgerFreshness> {
            anyhow::bail!("unused learning service")
        }
    }

    #[async_trait::async_trait]
    impl ProviderService for NoopService {
        async fn chat(
            &self,
            _model_id: &ModelId,
            _context: Context,
            _provider: Provider<Url>,
        ) -> ResultStream<ChatCompletionMessage, anyhow::Error> {
            Ok(Box::pin(tokio_stream::iter(std::iter::empty())))
        }

        async fn models(&self, _provider: Provider<Url>) -> anyhow::Result<Vec<Model>> {
            anyhow::bail!("unused provider service")
        }

        async fn get_provider(&self, _id: ProviderId) -> anyhow::Result<Provider<Url>> {
            anyhow::bail!("unused provider service")
        }

        async fn get_all_providers(&self) -> anyhow::Result<Vec<AnyProvider>> {
            anyhow::bail!("unused provider service")
        }

        async fn upsert_credential(
            &self,
            _credential: forge_domain::AuthCredential,
        ) -> anyhow::Result<()> {
            anyhow::bail!("unused provider service")
        }

        async fn remove_credential(&self, _id: &ProviderId) -> anyhow::Result<()> {
            anyhow::bail!("unused provider service")
        }

        async fn migrate_env_credentials(
            &self,
        ) -> anyhow::Result<Option<forge_domain::MigrationResult>> {
            anyhow::bail!("unused provider service")
        }
    }

    #[async_trait::async_trait]
    impl AppConfigService for NoopService {
        async fn get_session_config(&self) -> Option<forge_domain::ModelConfig> {
            None
        }

        async fn get_commit_config(&self) -> anyhow::Result<Option<forge_domain::ModelConfig>> {
            Ok(None)
        }

        async fn get_suggest_config(&self) -> anyhow::Result<Option<forge_domain::ModelConfig>> {
            Ok(None)
        }

        async fn get_reasoning_effort(&self) -> anyhow::Result<Option<forge_domain::Effort>> {
            Ok(None)
        }

        async fn update_config(
            &self,
            _ops: Vec<forge_domain::ConfigOperation>,
        ) -> anyhow::Result<()> {
            anyhow::bail!("unused config service")
        }
    }

    #[async_trait::async_trait]
    impl TemplateService for NoopService {
        async fn register_template(&self, _path: PathBuf) -> anyhow::Result<()> {
            anyhow::bail!("unused template service")
        }

        async fn render_template<V: serde::Serialize + Send + Sync>(
            &self,
            _template: Template<V>,
            _object: &V,
        ) -> anyhow::Result<String> {
            anyhow::bail!("unused template service")
        }
    }

    #[async_trait::async_trait]
    impl AttachmentService for NoopService {
        async fn attachments(&self, _url: &str) -> anyhow::Result<Vec<Attachment>> {
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
        async fn collect_files(&self, _config: Walker) -> anyhow::Result<Vec<File>> {
            anyhow::bail!("unused file discovery service")
        }

        async fn list_current_directory(&self) -> anyhow::Result<Vec<File>> {
            anyhow::bail!("unused file discovery service")
        }
    }

    #[async_trait::async_trait]
    impl McpConfigManager for NoopService {
        async fn read_mcp_config(&self, _scope: Option<&Scope>) -> anyhow::Result<McpConfig> {
            anyhow::bail!("unused mcp config manager")
        }

        async fn write_mcp_config(
            &self,
            _config: &McpConfig,
            _scope: &Scope,
        ) -> anyhow::Result<()> {
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
        ) -> anyhow::Result<FsWriteOutput> {
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
        ) -> anyhow::Result<PlanCreateOutput> {
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
        ) -> anyhow::Result<PatchOutput> {
            anyhow::bail!("unused patch service")
        }

        async fn multi_patch(
            &self,
            _path: String,
            _edits: Vec<forge_domain::PatchEdit>,
        ) -> anyhow::Result<PatchOutput> {
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
        ) -> anyhow::Result<ReadOutput> {
            anyhow::bail!("unused read service")
        }
    }

    #[async_trait::async_trait]
    impl ImageReadService for NoopService {
        async fn read_image(&self, _path: String) -> anyhow::Result<Image> {
            anyhow::bail!("unused image service")
        }
    }

    #[async_trait::async_trait]
    impl FsRemoveService for NoopService {
        async fn remove(&self, _path: String) -> anyhow::Result<FsRemoveOutput> {
            anyhow::bail!("unused remove service")
        }
    }

    #[async_trait::async_trait]
    impl FsSearchService for NoopService {
        async fn search(
            &self,
            _params: forge_domain::FSSearch,
        ) -> anyhow::Result<Option<SearchResult>> {
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
        ) -> anyhow::Result<Option<String>> {
            anyhow::bail!("unused follow up service")
        }
    }

    #[async_trait::async_trait]
    impl FsUndoService for NoopService {
        async fn undo(&self, _path: String) -> anyhow::Result<FsUndoOutput> {
            anyhow::bail!("unused undo service")
        }
    }

    #[async_trait::async_trait]
    impl NetFetchService for NoopService {
        async fn fetch(&self, _url: String, _raw: Option<bool>) -> anyhow::Result<HttpResponse> {
            anyhow::bail!("unused fetch service")
        }
    }

    #[async_trait::async_trait]
    impl ShellService for NoopService {
        async fn execute(&self, _request: ShellExecuteRequest) -> anyhow::Result<ShellOutput> {
            anyhow::bail!("unused shell service")
        }
    }

    #[async_trait::async_trait]
    impl McpService for NoopService {
        async fn get_mcp_servers(&self) -> anyhow::Result<McpServers> {
            anyhow::bail!("unused mcp service")
        }

        async fn execute_mcp(&self, _call: ToolCallFull) -> anyhow::Result<ToolOutput> {
            anyhow::bail!("unused mcp service")
        }

        async fn reload_mcp(&self) -> anyhow::Result<()> {
            anyhow::bail!("unused mcp service")
        }
    }

    #[async_trait::async_trait]
    impl AuthService for NoopService {
        async fn user_info(&self, _api_key: &str) -> anyhow::Result<User> {
            anyhow::bail!("unused auth service")
        }

        async fn user_usage(&self, _api_key: &str) -> anyhow::Result<UserUsage> {
            anyhow::bail!("unused auth service")
        }
    }

    #[async_trait::async_trait]
    impl AgentRegistry for NoopService {
        async fn get_active_agent_id(&self) -> anyhow::Result<Option<AgentId>> {
            Ok(None)
        }

        async fn set_active_agent_id(&self, _agent_id: AgentId) -> anyhow::Result<()> {
            anyhow::bail!("unused agent registry")
        }

        async fn get_agents(&self) -> anyhow::Result<Vec<forge_domain::Agent>> {
            Ok(Vec::new())
        }

        async fn get_agent_infos(&self) -> anyhow::Result<Vec<forge_domain::AgentInfo>> {
            Ok(Vec::new())
        }

        async fn get_agent(
            &self,
            _agent_id: &AgentId,
        ) -> anyhow::Result<Option<forge_domain::Agent>> {
            Ok(None)
        }

        async fn reload_agents(&self) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl CommandLoaderService for NoopService {
        async fn get_commands(&self) -> anyhow::Result<Vec<forge_domain::Command>> {
            Ok(Vec::new())
        }
    }

    #[async_trait::async_trait]
    impl PolicyService for NoopService {
        async fn check_operation_permission(
            &self,
            _operation: &forge_domain::PermissionOperation,
        ) -> anyhow::Result<PolicyDecision> {
            anyhow::bail!("unused policy service")
        }
    }

    #[async_trait::async_trait]
    impl ProviderAuthService for NoopService {
        async fn init_provider_auth(
            &self,
            _provider_id: ProviderId,
            _method: AuthMethod,
        ) -> anyhow::Result<AuthContextRequest> {
            anyhow::bail!("unused provider auth service")
        }

        async fn complete_provider_auth(
            &self,
            _provider_id: ProviderId,
            _context: AuthContextResponse,
            _timeout: Duration,
        ) -> anyhow::Result<()> {
            anyhow::bail!("unused provider auth service")
        }

        async fn refresh_provider_credential(
            &self,
            _provider: Provider<Url>,
        ) -> anyhow::Result<Provider<Url>> {
            anyhow::bail!("unused provider auth service")
        }
    }

    #[async_trait::async_trait]
    impl WorkspaceService for NoopService {
        async fn sync_workspace(
            &self,
            _path: PathBuf,
        ) -> anyhow::Result<forge_stream::MpscStream<anyhow::Result<SyncProgress>>> {
            anyhow::bail!("unused workspace service")
        }

        async fn produce_workspace_exact_fact_reference(
            &self,
            _path: PathBuf,
        ) -> anyhow::Result<WorkspaceExactFactReferenceReport> {
            anyhow::bail!("unused workspace service")
        }

        async fn workspace_exact_fact_status(
            &self,
            _path: PathBuf,
        ) -> anyhow::Result<WorkspaceExactFactStatusReport> {
            anyhow::bail!("unused workspace service")
        }

        async fn query_workspace(
            &self,
            _path: PathBuf,
            _params: SearchParams<'_>,
        ) -> anyhow::Result<Vec<Node>> {
            anyhow::bail!("unused workspace service")
        }

        async fn list_workspaces(&self) -> anyhow::Result<Vec<WorkspaceInfo>> {
            anyhow::bail!("unused workspace service")
        }

        async fn get_workspace_info(
            &self,
            _path: PathBuf,
        ) -> anyhow::Result<Option<WorkspaceInfo>> {
            anyhow::bail!("unused workspace service")
        }

        async fn is_indexed(&self, _path: &Path) -> anyhow::Result<bool> {
            anyhow::bail!("unused workspace service")
        }

        async fn delete_workspace(&self, _workspace_id: &WorkspaceId) -> anyhow::Result<()> {
            anyhow::bail!("unused workspace service")
        }

        async fn delete_workspaces(&self, _workspace_ids: &[WorkspaceId]) -> anyhow::Result<()> {
            anyhow::bail!("unused workspace service")
        }

        async fn project_model_context_diagnostic(
            &self,
            _path: &Path,
        ) -> anyhow::Result<WorkspaceContextManifestDiagnostic> {
            anyhow::bail!("unused workspace service")
        }

        async fn get_workspace_status(&self, _path: PathBuf) -> anyhow::Result<Vec<FileStatus>> {
            anyhow::bail!("unused workspace service")
        }

        async fn is_authenticated(&self) -> anyhow::Result<bool> {
            anyhow::bail!("unused workspace service")
        }

        async fn init_auth_credentials(&self) -> anyhow::Result<WorkspaceAuth> {
            anyhow::bail!("unused workspace service")
        }

        async fn init_workspace(&self, _path: PathBuf) -> anyhow::Result<WorkspaceId> {
            anyhow::bail!("unused workspace service")
        }
    }

    #[async_trait::async_trait]
    impl SkillFetchService for NoopService {
        async fn fetch_skill(&self, _skill_name: String) -> anyhow::Result<forge_domain::Skill> {
            anyhow::bail!("unused skill service")
        }

        async fn list_skills(&self) -> anyhow::Result<Vec<forge_domain::Skill>> {
            anyhow::bail!("unused skill service")
        }
    }

    #[derive(Clone, Default)]
    struct FacadeFixture {
        conversation: Arc<RawConversationService>,
        steer: Arc<RawSteerService>,
        noop: Arc<NoopService>,
    }

    impl EnvironmentInfra for FacadeFixture {
        type Config = forge_config::ForgeConfig;

        fn get_env_var(&self, _key: &str) -> Option<String> {
            None
        }

        fn get_env_vars(&self) -> BTreeMap<String, String> {
            BTreeMap::new()
        }

        fn get_environment(&self) -> forge_domain::Environment {
            forge_domain::Environment {
                os: "test".to_string(),
                cwd: PathBuf::from("/tmp"),
                home: None,
                shell: "sh".to_string(),
                base_path: PathBuf::from("/tmp/.forge"),
            }
        }

        fn get_config(&self) -> anyhow::Result<Self::Config> {
            Ok(forge_config::ForgeConfig::default())
        }

        async fn update_environment(
            &self,
            _ops: Vec<forge_domain::ConfigOperation>,
        ) -> anyhow::Result<()> {
            anyhow::bail!("unused environment infra")
        }
    }

    impl Services for FacadeFixture {
        type ProviderService = NoopService;
        type AppConfigService = NoopService;
        type ConversationService = RawConversationService;
        type LearningService = NoopService;
        type SteerService = RawSteerService;
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
        type WorkspaceService = NoopService;
        type SkillFetchService = NoopService;

        fn provider_service(&self) -> &Self::ProviderService {
            &self.noop
        }

        fn config_service(&self) -> &Self::AppConfigService {
            &self.noop
        }

        fn conversation_service(&self) -> &Self::ConversationService {
            &self.conversation
        }

        fn learning_service(&self) -> &Self::LearningService {
            &self.noop
        }

        fn steer_service(&self) -> &Self::SteerService {
            &self.steer
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
            &self.noop
        }

        fn skill_fetch_service(&self) -> &Self::SkillFetchService {
            &self.noop
        }
    }

    #[tokio::test]
    async fn test_services_facade_clears_steer_queue_when_promoting_delegated_conversation() {
        let setup = FacadeFixture::default();
        let conversation = Conversation::generate().context(Context::default());
        let parent_id = ConversationId::generate();
        ConversationService::upsert_conversation(&setup, conversation.clone())
            .await
            .unwrap();
        SteerService::enqueue_steer(
            &setup,
            &conversation.id,
            SteerMessage::new("stale main steer").unwrap(),
        )
        .await
        .unwrap();

        let promoted = ConversationService::ensure_delegated_conversation(
            &setup,
            &conversation.id,
            Some(parent_id),
        )
        .await
        .unwrap();
        let actual = (
            (promoted.initiator, promoted.parent_id),
            *setup.steer.clear_count.lock().await,
            SteerService::drain_steer(&setup, &conversation.id)
                .await
                .unwrap(),
        );
        let expected = ((Initiator::Agent, Some(parent_id)), 1, Vec::new());

        assert_eq!(actual, expected);
    }
}
