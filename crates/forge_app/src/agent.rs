use std::sync::Arc;

use forge_config::ForgeConfig;
use forge_domain::{
    Agent, ChatCompletionMessage, Compact, Context, Conversation, Effort, MaxTokens, ModelId,
    ProviderId, ReasoningConfig, ResultStream, SteerMessage, Temperature, ToolCallContext,
    ToolCallFull, ToolResult, TopK, TopP,
};
use merge::Merge;

use crate::services::{AppConfigService, SteerService};
use crate::tool_registry::ToolRegistry;
use crate::{ConversationService, EnvironmentInfra, ProviderService, Services};

/// Agent service trait that provides core chat and tool call functionality.
/// This trait abstracts the essential operations needed by the Orchestrator.
#[async_trait::async_trait]
pub trait AgentService: Send + Sync + 'static {
    /// Execute a chat completion request
    async fn chat_agent(
        &self,
        id: &ModelId,
        context: Context,
        provider_id: Option<ProviderId>,
    ) -> ResultStream<ChatCompletionMessage, anyhow::Error>;

    /// Execute a tool call
    async fn call(
        &self,
        agent: &Agent,
        context: &ToolCallContext,
        call: ToolCallFull,
    ) -> ToolResult;

    /// Synchronize the on-going conversation
    async fn update(&self, conversation: Conversation) -> anyhow::Result<()>;

    /// Returns whether a conversation is currently primary.
    ///
    /// # Arguments
    /// * `conversation_id` - The conversation to inspect.
    async fn is_primary_conversation(
        &self,
        conversation_id: &forge_domain::ConversationId,
    ) -> anyhow::Result<bool> {
        let _ = conversation_id;
        Ok(true)
    }

    /// Drains typed steer messages for the current conversation.
    ///
    /// # Arguments
    /// * `conversation_id` - The conversation whose steer queue should be
    ///   drained.
    async fn drain_steer_messages(
        &self,
        conversation_id: &forge_domain::ConversationId,
    ) -> anyhow::Result<Vec<SteerMessage>> {
        let _ = conversation_id;
        Ok(Vec::new())
    }
}

/// Blanket implementation of AgentService for any type that implements Services
#[async_trait::async_trait]
impl<T: Services + EnvironmentInfra<Config = forge_config::ForgeConfig>> AgentService for T {
    async fn chat_agent(
        &self,
        id: &ModelId,
        context: Context,
        provider_id: Option<ProviderId>,
    ) -> ResultStream<ChatCompletionMessage, anyhow::Error> {
        let provider_id = if let Some(provider_id) = provider_id {
            provider_id
        } else {
            self.get_session_config()
                .await
                .map(|c| c.provider)
                .ok_or_else(|| forge_domain::Error::NoDefaultSession)?
        };
        let provider = self.get_provider(provider_id).await?;
        let models = self.models(provider.clone()).await?;
        let selected_model = models
            .iter()
            .find(|model| model.id == *id && model.provider_id == provider.id)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Selected model '{}' for provider '{}' is missing from resolved provider metadata; context-window safety cannot be proven before provider dispatch.",
                    id,
                    provider.id
                )
            })?;
        let context_length = selected_model.context_length.ok_or_else(|| {
            anyhow::anyhow!(
                "Selected model '{}' for provider '{}' does not expose a configured context_length; context-window safety cannot be proven before provider dispatch. Add context_length to the model metadata or select a model with known context window.",
                id,
                provider.id
            )
        })?;
        let context = context.model_context_length(context_length);

        self.chat(id, context, provider).await
    }

    async fn call(
        &self,
        agent: &Agent,
        context: &ToolCallContext,
        call: ToolCallFull,
    ) -> ToolResult {
        let registry = ToolRegistry::new(Arc::new(self.clone()));
        registry.call(agent, context, call).await
    }

    async fn update(&self, conversation: Conversation) -> anyhow::Result<()> {
        self.upsert_conversation(conversation).await
    }

    async fn is_primary_conversation(
        &self,
        conversation_id: &forge_domain::ConversationId,
    ) -> anyhow::Result<bool> {
        Ok(self
            .find_conversation(conversation_id)
            .await?
            .is_some_and(|conversation| conversation.is_primary_user_conversation()))
    }

    async fn drain_steer_messages(
        &self,
        conversation_id: &forge_domain::ConversationId,
    ) -> anyhow::Result<Vec<SteerMessage>> {
        self.drain_steer(conversation_id).await
    }
}

/// Extension trait for applying workflow-level configuration overrides to an
/// [`Agent`].
///
/// This lives in the application layer because the configuration is built
/// from [`ForgeConfig`] and applied to domain agents at runtime.
pub trait AgentExt {
    /// Applies workflow-level configuration overrides to this agent.
    ///
    /// Fields in `config` always win over agent defaults, except for
    /// `max_tool_failure_per_turn` and `max_requests_per_turn` where the
    /// agent's own value takes priority (i.e. the workflow value is only
    /// applied when the agent has no value set).
    ///
    /// # Arguments
    /// * `config` - The top-level Forge configuration.
    fn apply_config(self, config: &ForgeConfig) -> Agent;
}

impl AgentExt for Agent {
    fn apply_config(self, config: &ForgeConfig) -> Agent {
        let mut agent = self;

        if let Some(temperature) = config
            .temperature
            .and_then(|d| Temperature::new(d.0 as f32).ok())
        {
            agent.temperature = Some(temperature);
        }

        if let Some(top_p) = config.top_p.and_then(|d| TopP::new(d.0 as f32).ok()) {
            agent.top_p = Some(top_p);
        }

        if let Some(top_k) = config.top_k.and_then(|k| TopK::new(k).ok()) {
            agent.top_k = Some(top_k);
        }

        if let Some(max_tokens) = config.max_tokens.and_then(|m| MaxTokens::new(m).ok()) {
            agent.max_tokens = Some(max_tokens);
        }

        if agent.max_tool_failure_per_turn.is_none()
            && let Some(max_tool_failure_per_turn) = config.max_tool_failure_per_turn
        {
            agent.max_tool_failure_per_turn = Some(max_tool_failure_per_turn);
        }

        agent.tool_supported = Some(config.tool_supported);

        if agent.max_requests_per_turn.is_none()
            && let Some(max_requests_per_turn) = config.max_requests_per_turn
        {
            agent.max_requests_per_turn = Some(max_requests_per_turn);
        }

        // Apply workflow compact configuration to agents. Workflow values fill
        // the agent's compact defaults, and only explicitly configured agent
        // compact fields override them.
        if let Some(ref workflow_compact) = config.compact {
            let workflow_compact = Compact {
                retention_window: Some(workflow_compact.retention_window),
                eviction_window: Some(workflow_compact.eviction_window.value()),
                max_tokens: workflow_compact.max_tokens,
                token_threshold: workflow_compact.token_threshold,
                token_threshold_percentage: workflow_compact
                    .token_threshold_percentage
                    .map(|percentage| percentage.value()),
                turn_threshold: workflow_compact.turn_threshold,
                message_threshold: workflow_compact.message_threshold,
                model: workflow_compact.model.as_deref().map(ModelId::new),
                on_turn_end: workflow_compact.on_turn_end,
            };
            let mut merged_compact = workflow_compact;
            merged_compact.merge(agent.compact.clone());
            agent.compact = merged_compact;
        }

        // Apply workflow reasoning configuration to agents.
        // Agent-level fields take priority; config fills in any unset fields.
        // Exception: config `enabled = false` always wins — it is an explicit
        // global disable that must override any per-agent setting.
        if let Some(ref config_reasoning) = config.reasoning {
            use forge_config::Effort as ConfigEffort;
            let config_as_domain = ReasoningConfig {
                effort: config_reasoning.effort.as_ref().map(|e| match e {
                    ConfigEffort::None => Effort::None,
                    ConfigEffort::Minimal => Effort::Minimal,
                    ConfigEffort::Low => Effort::Low,
                    ConfigEffort::Medium => Effort::Medium,
                    ConfigEffort::High => Effort::High,
                    ConfigEffort::XHigh => Effort::XHigh,
                    ConfigEffort::Max => Effort::Max,
                }),
                max_tokens: config_reasoning.max_tokens,
                exclude: config_reasoning.exclude,
                enabled: config_reasoning.enabled,
            };
            // Start from the agent's own settings and fill unset fields from config.
            let mut merged = agent.reasoning.clone().unwrap_or_default();
            merged.merge(config_as_domain);
            // If the config explicitly disables reasoning, honour that override
            // regardless of what the agent definition says.
            if config_reasoning.enabled == Some(false) {
                merged.enabled = Some(false);
            }
            agent.reasoning = Some(merged);
        }

        agent
    }
}

#[cfg(test)]
mod tests {
    use forge_config::{Effort as ConfigEffort, ReasoningConfig as ConfigReasoningConfig};
    use forge_domain::{AgentId, Effort, ModelId, ProviderId, ReasoningConfig};
    use pretty_assertions::assert_eq;

    use super::*;

    fn fixture_agent() -> Agent {
        Agent::new(
            AgentId::new("test"),
            ProviderId::ANTHROPIC,
            ModelId::new("claude-3-5-sonnet-20241022"),
        )
    }

    /// When the agent has no reasoning config, the config's reasoning is
    /// applied in full.
    #[test]
    fn test_reasoning_applied_from_config_when_agent_has_none() {
        let config = ForgeConfig::default().reasoning(
            ConfigReasoningConfig::default()
                .enabled(true)
                .effort(ConfigEffort::Medium),
        );

        let actual = fixture_agent().apply_config(&config).reasoning;

        let expected = Some(
            ReasoningConfig::default()
                .enabled(true)
                .effort(Effort::Medium),
        );

        assert_eq!(actual, expected);
    }

    /// When the agent already has reasoning fields set, those fields take
    /// priority; config only fills in fields the agent left unset.
    #[test]
    fn test_reasoning_agent_fields_take_priority_over_config() {
        let config = ForgeConfig::default().reasoning(
            ConfigReasoningConfig::default()
                .enabled(true)
                .effort(ConfigEffort::Low)
                .max_tokens(1024_usize),
        );

        // Agent overrides effort but leaves enabled and max_tokens unset.
        let agent = fixture_agent().reasoning(ReasoningConfig::default().effort(Effort::High));

        let actual = agent.apply_config(&config).reasoning;

        let expected = Some(
            ReasoningConfig::default()
                .effort(Effort::High) // agent's value wins
                .enabled(true) // filled in from config
                .max_tokens(1024_usize), // filled in from config
        );

        assert_eq!(actual, expected);
    }

    /// When config sets `enabled = false`, it must override the agent's
    /// `enabled = true`. This prevents reasoning parameters from being sent to
    /// models that don't support them (e.g. claude-haiku with effort set).
    #[test]
    fn test_config_disabled_overrides_agent_enabled() {
        let config = ForgeConfig::default().reasoning(
            ConfigReasoningConfig::default()
                .enabled(false)
                .effort(ConfigEffort::None),
        );

        // Agent has reasoning explicitly enabled.
        let agent = fixture_agent().reasoning(
            ReasoningConfig::default()
                .enabled(true)
                .effort(Effort::High),
        );

        let actual = agent.apply_config(&config).reasoning;

        // enabled must be false even though the agent said true.
        assert_eq!(actual.as_ref().and_then(|r| r.enabled), Some(false));
    }

    /// Tests compact merging where workflow config fills default agent compact
    /// settings while explicit agent compact fields retain priority.
    #[test]
    fn test_compact_agent_settings_take_priority_over_workflow_config() {
        use forge_config::Percentage;

        // Workflow config with custom compact settings (from .forge.toml)
        let workflow_compact = forge_config::Compact::default()
            .retention_window(10_usize)
            .eviction_window(Percentage::new(0.3).unwrap())
            .max_tokens(5000_usize)
            .token_threshold(80000_usize)
            .token_threshold_percentage(0.65_f64);

        let config = ForgeConfig::default().compact(workflow_compact);

        // Agent with default compact config - no explicit compact fields.
        let agent = fixture_agent();

        let actual = agent.apply_config(&config).compact;

        assert_eq!(
            actual.retention_window,
            Some(10),
            "Workflow retention_window applies because the agent has no explicit compact override"
        );

        // Agent default has token_threshold=None, workflow's 80000 should apply
        assert_eq!(
            actual.token_threshold,
            Some(80000),
            "Workflow token_threshold applies because agent default has None"
        );
        assert_eq!(
            actual.token_threshold_percentage,
            Some(0.65),
            "Workflow context-window percentage applies because agent default has None"
        );
    }

    #[test]
    fn test_compact_explicit_agent_zero_overrides_workflow_retention() {
        use forge_config::Percentage;
        use forge_domain::Compact as DomainCompact;

        let workflow_compact = forge_config::Compact::default()
            .retention_window(15_usize)
            .eviction_window(Percentage::new(0.25).unwrap());
        let config = ForgeConfig::default().compact(workflow_compact);
        let agent = fixture_agent().compact(DomainCompact::new().retention_window(0_usize));

        let actual = agent.apply_config(&config).compact;
        let expected = Some(0);

        assert_eq!(actual.retention_window, expected);
    }

    /// Tests that explicit agent compact fields override workflow compact
    /// values while unset fields are filled from workflow config.
    #[test]
    fn test_compact_partial_agent_settings_override_workflow_values() {
        use forge_config::Percentage;
        use forge_domain::Compact as DomainCompact;

        // Workflow config with ALL settings
        let workflow_compact = forge_config::Compact::default()
            .retention_window(15_usize)
            .eviction_window(Percentage::new(0.25).unwrap())
            .max_tokens(6000_usize)
            .token_threshold(90000_usize)
            .token_threshold_percentage(0.4_f64)
            .turn_threshold(20_usize);

        let config = ForgeConfig::default().compact(workflow_compact);

        // Agent with PARTIAL compact config (only retention_window and
        // token_threshold_percentage set).
        let agent = fixture_agent().compact(
            DomainCompact::new()
                .retention_window(5_usize)
                .token_threshold_percentage(0.25_f64),
        );

        let actual = agent.apply_config(&config).compact;

        assert_eq!(
            actual.retention_window,
            Some(5),
            "Explicit agent retention_window takes priority over workflow retention_window"
        );

        // Fields where agent had None get workflow values
        assert_eq!(
            actual.token_threshold,
            Some(90000),
            "Workflow token_threshold applies (agent had None)"
        );
        assert_eq!(
            actual.token_threshold_percentage,
            Some(0.25),
            "Agent's context-window percentage takes priority over workflow's 0.4"
        );
        assert_eq!(
            actual.turn_threshold,
            Some(20),
            "Workflow turn_threshold applies (agent had None)"
        );
    }
}
