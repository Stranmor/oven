use std::borrow::Cow;

use derive_more::derive::Display;
use derive_setters::Setters;
use merge::Merge;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use strum_macros::{Display as StrumDisplay, EnumString};

use crate::{
    Compact, ContextWindowBudget, Error, EventContext, MaxTokens, Model, ModelId, ProviderId,
    Result, SystemContext, Temperature, Template, ToolDefinition, ToolName, TopK, TopP,
};

// Unique identifier for an agent
#[derive(Debug, Display, Eq, PartialEq, Hash, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
pub struct AgentId(Cow<'static, str>);

impl From<&str> for AgentId {
    fn from(value: &str) -> Self {
        AgentId(Cow::Owned(value.to_string()))
    }
}

impl AgentId {
    // Creates a new agent ID from a string-like value
    pub fn new(id: impl ToString) -> Self {
        Self(Cow::Owned(id.to_string()))
    }

    // Returns the agent ID as a string reference
    pub fn as_str(&self) -> &str {
        self.0.as_ref()
    }

    pub const FORGE: AgentId = AgentId(Cow::Borrowed("forge"));
    pub const MUSE: AgentId = AgentId(Cow::Borrowed("muse"));
    pub const SAGE: AgentId = AgentId(Cow::Borrowed("sage"));
}

impl Default for AgentId {
    fn default() -> Self {
        AgentId::FORGE
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize, Merge, Setters, JsonSchema, PartialEq)]
#[setters(strip_option)]
#[merge(strategy = merge::option::overwrite_none)]
pub struct ReasoningConfig {
    /// Controls the effort level of the agent's reasoning
    /// supported by openrouter and forge provider
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<Effort>,

    /// Controls how many tokens the model can spend thinking.
    /// supported by openrouter, anthropic and forge provider
    /// should be greater then 1024 but less than overall max_tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,

    /// Model thinks deeply, but the reasoning is hidden from you.
    /// supported by openrouter and forge provider
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude: Option<bool>,

    /// Enables reasoning at the "medium" effort level with no exclusions.
    /// supported by openrouter, anthropic and forge provider
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, StrumDisplay, EnumString)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase", ascii_case_insensitive)]
pub enum Effort {
    /// No reasoning; skips the thinking step entirely.
    None,
    /// Minimal reasoning; fastest and cheapest.
    Minimal,
    /// Low reasoning effort.
    Low,
    /// Medium reasoning effort; the default for most providers.
    Medium,
    /// High reasoning effort.
    High,
    /// Extra-high reasoning effort (OpenAI / OpenRouter).
    XHigh,
    /// Maximum reasoning effort; only available on select Anthropic models.
    Max,
}

/// Estimates the token count from a string representation
/// This is a simple estimation that should be replaced with a more accurate
/// tokenizer
/// Estimates token count from a string representation
/// Re-exported for compaction reporting
pub fn estimate_token_count(count: usize) -> usize {
    // A very rough estimation that assumes ~4 characters per token on average
    // In a real implementation, this should use a proper LLM-specific tokenizer
    count / 4
}

/// Runtime agent representation with required model and provider
#[derive(Debug, Clone, PartialEq, Setters, Serialize, Deserialize, JsonSchema)]
#[setters(strip_option, into)]
pub struct Agent {
    /// Unique identifier for the agent
    pub id: AgentId,

    /// Human-readable title for the agent
    pub title: Option<String>,

    /// Human-readable description of the agent's purpose
    pub description: Option<String>,

    /// Flag to enable/disable tool support for this agent.
    pub tool_supported: Option<bool>,

    /// Path to the agent definition file, if loaded from a file
    pub path: Option<String>,

    /// Required provider for the agent
    pub provider: ProviderId,

    /// Required language model ID to be used by this agent
    pub model: ModelId,

    /// Template for the system prompt provided to the agent
    pub system_prompt: Option<Template<SystemContext>>,

    /// Template for the user prompt provided to the agent
    pub user_prompt: Option<Template<EventContext>>,

    /// Tools that the agent can use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolName>>,

    /// Maximum number of turns the agent can take
    pub max_turns: Option<u64>,

    /// Configuration for automatic context compaction
    pub compact: Compact,

    /// A set of custom rules that the agent should follow
    pub custom_rules: Option<String>,

    /// Temperature used for agent
    pub temperature: Option<Temperature>,

    /// Top-p (nucleus sampling) used for agent
    pub top_p: Option<TopP>,

    /// Top-k used for agent
    pub top_k: Option<TopK>,

    /// Maximum number of tokens the model can generate
    pub max_tokens: Option<MaxTokens>,

    /// Reasoning configuration for the agent.
    pub reasoning: Option<ReasoningConfig>,

    /// Maximum number of times a tool can fail before sending the response back
    pub max_tool_failure_per_turn: Option<usize>,

    /// Maximum number of requests that can be made in a single turn
    pub max_requests_per_turn: Option<usize>,

    /// Penalizes tokens based on how frequently they have appeared, preventing
    /// repetitive degeneration loops. Range: -2.0 to 2.0
    pub frequency_penalty: Option<f64>,

    /// Penalizes tokens that have appeared at least once, encouraging
    /// diversity. Range: -2.0 to 2.0
    pub presence_penalty: Option<f64>,
}

/// Lightweight metadata about an agent, used for listing without requiring a
/// configured provider or model.
#[derive(Debug, Default, Clone, PartialEq, Setters, Serialize, Deserialize, JsonSchema)]
#[setters(strip_option, into)]
pub struct AgentInfo {
    /// Unique identifier for the agent
    pub id: AgentId,

    /// Human-readable title for the agent
    pub title: Option<String>,

    /// Human-readable description of the agent's purpose
    pub description: Option<String>,
}

impl Agent {
    /// Create a new Agent with required provider and model
    pub fn new(id: impl Into<AgentId>, provider: ProviderId, model: ModelId) -> Self {
        Self {
            id: id.into(),
            title: Default::default(),
            description: Default::default(),
            provider,
            model,
            tool_supported: Default::default(),
            system_prompt: Default::default(),
            user_prompt: Default::default(),
            tools: Default::default(),
            max_turns: Default::default(),
            compact: Compact::default(),
            custom_rules: Default::default(),
            temperature: Default::default(),
            top_p: Default::default(),
            top_k: Default::default(),
            max_tokens: Default::default(),
            reasoning: Default::default(),
            max_tool_failure_per_turn: Default::default(),
            max_requests_per_turn: Default::default(),
            frequency_penalty: Default::default(),
            presence_penalty: Default::default(),
            path: Default::default(),
        }
    }

    /// Creates a ToolDefinition from this agent
    ///
    /// # Errors
    ///
    /// Returns an error if the agent has no description
    pub fn tool_definition(&self) -> Result<ToolDefinition> {
        if self.description.is_none() || self.description.as_ref().is_none_or(|d| d.is_empty()) {
            return Err(Error::MissingAgentDescription(self.id.clone()));
        }
        Ok(self.clone().into())
    }

    /// Sets the model in compaction config if not already set
    pub fn set_compact_model_if_none(mut self) -> Self {
        if self.compact.model.is_none() {
            self.compact.model = Some(self.model.clone());
        }
        self
    }

    /// Applies the effective `token_threshold` for automatic compaction.
    ///
    /// Explicit `compact.token_threshold` values are capped by the selected
    /// model's effective input budget when the model context window is known.
    /// The effective budget reserves the configured output tokens plus the
    /// same conservative provider safety margin used by preflight, so an
    /// oversized absolute threshold cannot delay compaction until provider
    /// dispatch is already unsafe. When `compact.token_threshold_percentage` is
    /// configured, it also participates as a percentage-derived cap and the
    /// lowest available threshold wins. When the absolute threshold is absent,
    /// Forge derives a default from the model context window using the
    /// configured percentage, or 70% when no percentage is configured, then
    /// applies the effective-input-budget cap.
    ///
    /// When the model's `context_length` is unknown, no context-window cap can
    /// be computed, so the explicit threshold or the default threshold is used
    /// as-is.
    ///
    /// # Arguments
    /// * `selected_model` - The model that will be used for this agent
    ///
    /// # Returns
    /// The agent with an effective token_threshold configured
    pub fn compaction_threshold(mut self, selected_model: Option<&Model>) -> Self {
        const DEFAULT_TOKEN_THRESHOLD: usize = 100_000;
        const DEFAULT_CONTEXT_WINDOW_PERCENTAGE: f64 = 0.7;

        let configured_threshold = self.compact.token_threshold;

        let known_context_window = selected_model
            .and_then(|model| model.context_length)
            .and_then(|cl| usize::try_from(cl).ok());

        let output_reservation = self
            .max_tokens
            .map(|max_tokens| max_tokens.value() as usize)
            .unwrap_or(ContextWindowBudget::DEFAULT_OUTPUT_TOKEN_RESERVATION);
        let effective_input_budget = known_context_window.and_then(|context_window| {
            ContextWindowBudget::new(context_window, output_reservation).effective_input_budget()
        });

        let percentage_threshold = known_context_window.and_then(|context_window| {
            self.compact
                .token_threshold_percentage
                .map(|percentage| ((context_window as f64) * percentage).floor() as usize)
        });

        let default_threshold = || {
            known_context_window
                .map(|context_window| {
                    let context_window_threshold = ((context_window as f64)
                        * DEFAULT_CONTEXT_WINDOW_PERCENTAGE)
                        .floor() as usize;
                    DEFAULT_TOKEN_THRESHOLD.min(context_window_threshold)
                })
                .unwrap_or(DEFAULT_TOKEN_THRESHOLD)
        };

        let mut candidates = Vec::new();
        match configured_threshold {
            Some(threshold) => candidates.push(threshold),
            None => candidates.push(default_threshold()),
        }
        if let Some(percentage_threshold) = percentage_threshold {
            candidates.push(percentage_threshold);
        }
        if let Some(effective_input_budget) = effective_input_budget {
            candidates.push(effective_input_budget);
        }
        let final_threshold = candidates
            .into_iter()
            .min()
            .expect("at least one compaction threshold candidate should exist");

        self.compact.token_threshold = Some(final_threshold);

        self
    }

    /// Gets the tool ordering for this agent, derived from the tools list
    pub fn tool_order(&self) -> crate::ToolOrder {
        self.tools
            .as_ref()
            .map(|tools| crate::ToolOrder::from_tool_list(tools))
            .unwrap_or_default()
    }
}

impl From<Agent> for ToolDefinition {
    fn from(value: Agent) -> Self {
        let description = value.description.unwrap_or_default();
        let name = ToolName::new(value.id);
        ToolDefinition {
            name,
            description,
            input_schema: schemars::schema_for!(crate::AgentInput),
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::{InputModality, Model, ProviderId, ToolName};

    fn model_fixture(id: &str, context_length: Option<u64>) -> Model {
        Model {
            id: ModelId::new(id),
            provider_id: ProviderId::FORGE,
            name: Some(id.to_string()),
            description: None,
            context_length,
            tools_supported: Some(true),
            supports_parallel_tool_calls: Some(true),
            supports_reasoning: Some(true),
            input_modalities: vec![InputModality::Text],
        }
    }

    #[test]
    fn test_agent_tool_definition_uses_agent_input_object_schema() {
        let fixture = Agent::new(
            AgentId::new("arch-sentinel"),
            ProviderId::OPENAI,
            ModelId::new("gpt-test"),
        )
        .description("Architecture critic");

        let actual = fixture.tool_definition().unwrap();
        let actual_schema = serde_json::to_value(actual.input_schema).unwrap();
        let expected_type = serde_json::json!("object");
        let expected_tasks_type = serde_json::json!("array");

        assert_eq!(actual.name, ToolName::new("arch-sentinel"));
        assert_eq!(actual_schema["type"], expected_type);
        assert_eq!(
            actual_schema["properties"]["tasks"]["type"],
            expected_tasks_type
        );
    }

    #[test]
    fn test_compaction_threshold_caps_explicit_threshold_without_percentage() {
        let fixture = Agent::new(
            AgentId::new("test"),
            ProviderId::OPENAI,
            ModelId::new("selected-model"),
        )
        .compact(Compact::new().token_threshold(100_000_usize));

        let selected_model = model_fixture("selected-model", Some(80_000));

        let actual = fixture.compaction_threshold(Some(&selected_model));
        let expected = Some(59_904);

        assert_eq!(actual.compact.token_threshold, expected);
    }

    #[test]
    fn test_compaction_threshold_preserves_explicit_threshold_within_default_margin() {
        let fixture = Agent::new(
            AgentId::new("test"),
            ProviderId::OPENAI,
            ModelId::new("selected-model"),
        )
        .compact(Compact::new().token_threshold(60_000_usize));

        let selected_model = model_fixture("selected-model", Some(80_000));

        let actual = fixture.compaction_threshold(Some(&selected_model));
        // The explicit threshold is capped to the effective input budget.
        let expected = Some(59_904);

        assert_eq!(actual.compact.token_threshold, expected);
    }

    #[test]
    fn test_compaction_threshold_explicit_percentage_caps_explicit_threshold() {
        let fixture = Agent::new(
            AgentId::new("test"),
            ProviderId::OPENAI,
            ModelId::new("selected-model"),
        )
        .compact(
            Compact::new()
                .token_threshold(100_000_usize)
                .token_threshold_percentage(0.5_f64),
        );

        let selected_model = model_fixture("selected-model", Some(80_000));

        let actual = fixture.compaction_threshold(Some(&selected_model));
        // The explicit 0.5 percentage caps the explicit absolute threshold.
        let expected = Some(40_000);

        assert_eq!(actual.compact.token_threshold, expected);
    }

    #[test]
    fn test_compaction_threshold_explicit_percentage_derives_default_threshold() {
        let fixture = Agent::new(
            AgentId::new("test"),
            ProviderId::OPENAI,
            ModelId::new("selected-model"),
        )
        .compact(Compact::new().token_threshold_percentage(0.5_f64));

        let selected_model = model_fixture("selected-model", Some(80_000));

        let actual = fixture.compaction_threshold(Some(&selected_model));
        let expected = Some(40_000);

        assert_eq!(actual.compact.token_threshold, expected);
    }

    #[test]
    fn test_compaction_threshold_uses_hardcoded_cap_when_context_window_cap_is_higher() {
        let fixture = Agent::new(
            AgentId::new("test"),
            ProviderId::OPENAI,
            ModelId::new("selected-model"),
        );

        let selected_model = model_fixture("selected-model", Some(200_000));

        let actual = fixture.compaction_threshold(Some(&selected_model));
        let expected = Some(100_000);

        assert_eq!(actual.compact.token_threshold, expected);
    }

    #[test]
    fn test_cap_compact_token_threshold_uses_configured_when_selected_model_is_missing() {
        // When the model is not found, the configured threshold is trusted as-is.
        // We can't meaningfully clamp against an unknown context window.
        let fixture = Agent::new(
            AgentId::new("test"),
            ProviderId::OPENAI,
            ModelId::new("selected-model"),
        )
        .compact(Compact::new().token_threshold(100_000_usize));

        let actual = fixture.compaction_threshold(None);
        let expected = Some(100_000);

        assert_eq!(actual.compact.token_threshold, expected);
    }

    #[test]
    fn test_compaction_threshold_caps_antigravity_250k_threshold_by_effective_input_budget() {
        let fixture = Agent::new(
            AgentId::new("test"),
            ProviderId::OPENAI,
            ModelId::new("gpt-5.5"),
        )
        .max_tokens(MaxTokens::new(100_000).unwrap())
        .compact(Compact::new().token_threshold(250_000_usize));
        let selected_model = model_fixture("gpt-5.5", Some(266_300));

        let actual = fixture.compaction_threshold(Some(&selected_model));
        let expected = Some(133_532);

        assert_eq!(actual.compact.token_threshold, expected);
    }

    /// When no token_threshold is configured, the default (100K) is used.
    /// If the model's context window is known, that default is capped to
    /// 70% of the context window.
    #[test]
    fn test_compaction_threshold_should_set_default_when_token_threshold_is_none() {
        // Agent with NO token_threshold set (default Compact)
        let fixture = Agent::new(
            AgentId::new("test"),
            ProviderId::OPENAI,
            ModelId::new("gpt-5.3-codex-spark"),
        );
        // Verify default has no threshold
        assert_eq!(fixture.compact.token_threshold, None);

        let selected_model = model_fixture("gpt-5.3-codex-spark", Some(128_000));

        let actual = fixture.compaction_threshold(Some(&selected_model));

        // default 100K capped to 70% of 128K = 89.6K
        let expected_threshold = Some(89_600);
        assert_eq!(actual.compact.token_threshold, expected_threshold);
    }

    /// Explicit token_threshold is capped by the effective input budget for
    /// known model windows. Provider preflight remains the final request-size
    /// guard, while automatic compaction now starts before the configured
    /// threshold can exceed the safe prompt budget.
    #[test]
    fn test_compaction_threshold_caps_explicit_threshold_for_codex_spark() {
        let fixture = Agent::new(
            AgentId::new("test"),
            ProviderId::OPENAI,
            ModelId::new("gpt-5.3-codex-spark"),
        )
        .compact(Compact::new().token_threshold(100_000_usize));

        let selected_model = model_fixture("gpt-5.3-codex-spark", Some(128_000));

        let actual = fixture.compaction_threshold(Some(&selected_model));
        let expected = Some(98_304);

        assert_eq!(actual.compact.token_threshold, expected);
    }

    /// When model is found but has no context_length, the default threshold
    /// (100K) is used without clamping.
    #[test]
    fn test_compaction_threshold_no_model_context_length_uses_default() {
        // Agent with no compact config
        let fixture = Agent::new(
            AgentId::new("test"),
            ProviderId::OPENAI,
            ModelId::new("unknown-model"),
        );

        // Model with NO context_length info
        let selected_model = model_fixture("unknown-model", None);

        let actual = fixture.compaction_threshold(Some(&selected_model));

        // No context_length → no clamping, default 100K is used as-is
        assert_eq!(actual.compact.token_threshold, Some(100_000));
    }

    /// When user explicitly sets a large threshold and model has no
    /// context_length, the user's threshold is preserved.
    #[test]
    fn test_compaction_threshold_preserves_user_config_when_context_unknown() {
        let fixture = Agent::new(
            AgentId::new("test"),
            ProviderId::OPENAI,
            ModelId::new("gpt-5.4"),
        )
        .compact(Compact::new().token_threshold(400_000_usize));

        // Model not found at all
        let actual = fixture.compaction_threshold(None);

        // User's 400K threshold must be preserved — no clamping against
        // an unknown context window.
        assert_eq!(actual.compact.token_threshold, Some(400_000));
    }
}
