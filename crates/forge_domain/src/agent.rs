use std::borrow::Cow;

use derive_more::derive::Display;
use derive_setters::Setters;
use merge::Merge;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use strum_macros::{Display as StrumDisplay, EnumString};

use crate::{
    Compact, Error, EventContext, MaxTokens, Model, ModelId, ProviderId, Result, SystemContext,
    Temperature, Template, ToolDefinition, ToolName, TopK, TopP,
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

    /// Penalizes tokens that have appeared at least once, encouraging diversity.
    /// Range: -2.0 to 2.0
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
        Ok(ToolDefinition::new(self.id.as_str().to_string())
            .description(self.description.clone().unwrap()))
    }

    /// Sets the model in compaction config if not already set
    pub fn set_compact_model_if_none(mut self) -> Self {
        if self.compact.model.is_none() {
            self.compact.model = Some(self.model.clone());
        }
        self
    }

    /// Applies a safe `token_threshold` based on the model's context window.
    ///
    /// When the model's `context_length` is **known**, the threshold is capped
    /// to the lower of the configured absolute threshold and a percentage of
    /// the context window (default 70%). This preserves headroom for tool
    /// outputs and follow-up messages.
    ///
    /// When the model's `context_length` is **unknown** (model not found in
    /// the provider's list, or the provider doesn't report context length),
    /// the configured threshold is used as-is without clamping. Applying a
    /// percentage cap against a guessed default would silently override
    /// explicit user configuration.
    ///
    /// # Arguments
    /// * `selected_model` - The model that will be used for this agent
    ///
    /// # Returns
    /// The agent with a safe token_threshold configured
    pub fn compaction_threshold(mut self, selected_model: Option<&Model>) -> Self {
        const DEFAULT_TOKEN_THRESHOLD: usize = 100_000;
        const DEFAULT_CONTEXT_WINDOW_PERCENTAGE: f64 = 0.7;

        let configured_threshold = self
            .compact
            .token_threshold
            .unwrap_or(DEFAULT_TOKEN_THRESHOLD);

        let known_context_window = selected_model
            .and_then(|model| model.context_length)
            .and_then(|cl| usize::try_from(cl).ok());

        let final_threshold = if let Some(context_window) = known_context_window {
            // Model context window is known — clamp to a safe percentage.
            let percentage = self
                .compact
                .token_threshold_percentage
                .unwrap_or(DEFAULT_CONTEXT_WINDOW_PERCENTAGE);
            let context_window_threshold = ((context_window as f64) * percentage).floor() as usize;
            configured_threshold.min(context_window_threshold)
        } else {
            // Model context window is unknown — trust the configured threshold.
            configured_threshold
        };

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
    use crate::{InputModality, Model, ProviderId};

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
    fn test_cap_compact_token_threshold_by_context_window_caps_when_threshold_exceeds_context_window()
     {
        let fixture = Agent::new(
            AgentId::new("test"),
            ProviderId::OPENAI,
            ModelId::new("selected-model"),
        )
        .compact(Compact::new().token_threshold(100_000_usize));

        let selected_model = model_fixture("selected-model", Some(80_000));

        let actual = fixture.compaction_threshold(Some(&selected_model));
        let expected = Some(56_000);

        assert_eq!(actual.compact.token_threshold, expected);
    }

    #[test]
    fn test_cap_compact_token_threshold_caps_to_safe_margin_when_within_context_window() {
        // With the fix, thresholds are capped to 70% of context window for safety
        // even when they're technically "within" the context window
        let fixture = Agent::new(
            AgentId::new("test"),
            ProviderId::OPENAI,
            ModelId::new("selected-model"),
        )
        .compact(Compact::new().token_threshold(60_000_usize));

        let selected_model = model_fixture("selected-model", Some(80_000));

        let actual = fixture.compaction_threshold(Some(&selected_model));
        // 70% of 80K = 56K, so 60K threshold gets capped to 56K
        let expected = Some(56_000);

        assert_eq!(actual.compact.token_threshold, expected);
    }

    #[test]
    fn test_compaction_threshold_uses_configured_context_window_percentage_cap() {
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

    /// BUG 2: With default token_threshold of 100000 and codex-spark's 128000
    /// window, the threshold leaves only 28K headroom. When context grows
    /// to ~110K tokens, compaction won't trigger (below 100K threshold),
    /// but the API call will fail because the context (110K + tool outputs)
    /// exceeds 128K limit.
    #[test]
    fn test_compaction_threshold_insufficient_headroom_for_codex_spark() {
        // Simulates the embedded default config: token_threshold = 100000
        let fixture = Agent::new(
            AgentId::new("test"),
            ProviderId::OPENAI,
            ModelId::new("gpt-5.3-codex-spark"),
        )
        .compact(Compact::new().token_threshold(100_000_usize));

        let selected_model = model_fixture("gpt-5.3-codex-spark", Some(128_000));

        let actual = fixture.compaction_threshold(Some(&selected_model));

        // The current logic keeps 100000 because 100000 < 128000
        // But this leaves only 28000 tokens of headroom for tool outputs and new
        // messages When context is at 105000 tokens, compaction won't trigger
        // (below 100K threshold) But adding tool outputs (5000 tokens) + new
        // user message (2000 tokens) = 112000 API request with 112000 tokens
        // succeeds Next turn: context at 112000, still below 100K threshold
        // Adding more tool outputs: 112000 + 20000 = 132000 > 128000 limit →
        // context_length_exceeded!

        // EXPECTED: Threshold should be capped to provide safety margin (70% = 89600)
        // ACTUAL BUG: Threshold stays at 100000, causing eventual overflow
        let expected_safe_threshold = Some(89_600);
        assert_eq!(
            actual.compact.token_threshold, expected_safe_threshold,
            "BUG: With codex-spark (128K context), token_threshold of 100K leaves insufficient headroom. \
             Context can grow to 105K without compaction, then adding tool outputs pushes it over 128K limit. \
             Threshold should be capped to 70% of context window (89600) for safety."
        );
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
