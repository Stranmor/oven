use derive_more::derive::Display;
use derive_setters::Setters;
use fake::Dummy;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use strum_macros::EnumString;

use crate::ProviderId;

/// Conservative token budget used to prove that a model request fits inside a
/// known context window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextWindowBudget {
    context_window: usize,
    output_reservation: usize,
}

impl ContextWindowBudget {
    /// Default output-token reservation used when a request does not specify an
    /// explicit response budget.
    pub const DEFAULT_OUTPUT_TOKEN_RESERVATION: usize = 4_096;

    /// Creates a budget for a known model context window and reserved output
    /// budget.
    ///
    /// # Arguments
    /// * `context_window` - Maximum model context window in tokens.
    /// * `output_reservation` - Tokens reserved for model output.
    pub fn new(context_window: usize, output_reservation: usize) -> Self {
        Self { context_window, output_reservation }
    }

    /// Returns the known model context window in tokens.
    pub fn context_window(&self) -> usize {
        self.context_window
    }

    /// Returns the reserved output budget in tokens.
    pub fn output_reservation(&self) -> usize {
        self.output_reservation
    }

    /// Returns a conservative provider context-window safety margin.
    pub fn safety_margin(&self) -> usize {
        Self::context_window_safety_margin(self.context_window)
    }

    /// Returns the maximum input token budget after reserving output and
    /// margin.
    pub fn effective_input_budget(&self) -> Option<usize> {
        self.context_window
            .checked_sub(self.output_reservation)?
            .checked_sub(self.safety_margin())
    }

    /// Returns a conservative provider context-window safety margin.
    ///
    /// # Arguments
    /// * `context_window` - Maximum model context window in tokens.
    pub fn context_window_safety_margin(context_window: usize) -> usize {
        const MIN_MARGIN: usize = 4_096;
        const MAX_MARGIN: usize = 32_768;
        const PERCENTAGE: usize = 20;

        context_window
            .saturating_mul(PERCENTAGE)
            .saturating_div(100)
            .clamp(MIN_MARGIN, MAX_MARGIN)
    }
}

/// Represents input modalities that a model can accept
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, EnumString, JsonSchema, Dummy,
)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase", ascii_case_insensitive)]
pub enum InputModality {
    /// Text input (all models support this)
    Text,
    /// Image input (vision-capable models)
    Image,
}

/// Default input modalities when not specified (text-only)
fn default_input_modalities() -> Vec<InputModality> {
    vec![InputModality::Text]
}

/// Describes a model exposed by a specific provider.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, Setters, JsonSchema, Dummy)]
#[setters(strip_option)]
pub struct Model {
    /// Provider-local model identifier used in requests.
    pub id: ModelId,
    /// Provider that owns this model metadata.
    pub provider_id: ProviderId,
    /// Optional human-readable model name.
    pub name: Option<String>,
    /// Optional provider-supplied model description.
    pub description: Option<String>,
    /// Optional maximum context window reported by the provider.
    pub context_length: Option<u64>,
    /// Optional flag indicating whether tools are supported.
    pub tools_supported: Option<bool>,
    /// Whether the model supports parallel tool calls
    pub supports_parallel_tool_calls: Option<bool>,
    /// Whether the model supports reasoning
    pub supports_reasoning: Option<bool>,
    /// Input modalities supported by the model (defaults to text-only)
    #[serde(default = "default_input_modalities")]
    pub input_modalities: Vec<InputModality>,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Parameters {
    pub tool_supported: bool,
}

impl Parameters {
    pub fn new(tool_supported: bool) -> Self {
        Self { tool_supported }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, Hash, Eq, Display, JsonSchema, Dummy)]
#[serde(transparent)]
pub struct ModelId(String);

impl ModelId {
    pub fn new<T: Into<String>>(id: T) -> Self {
        Self(id.into())
    }
}

impl Model {
    /// Creates a new `Model` for the given provider with default values for
    /// all optional metadata fields.
    pub fn new(provider_id: impl Into<ProviderId>, id: impl Into<ModelId>) -> Self {
        Self {
            id: id.into(),
            provider_id: provider_id.into(),
            name: None,
            description: None,
            context_length: None,
            tools_supported: None,
            supports_parallel_tool_calls: None,
            supports_reasoning: None,
            input_modalities: default_input_modalities(),
        }
    }
}

impl From<String> for ModelId {
    fn from(value: String) -> Self {
        ModelId(value)
    }
}

impl From<&str> for ModelId {
    fn from(value: &str) -> Self {
        ModelId(value.to_string())
    }
}

impl ModelId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::str::FromStr for ModelId {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(ModelId(s.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_provider_id_is_required() {
        let json = r#"{"id": "test-model"}"#;
        let result = serde_json::from_str::<Model>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_context_window_budget_subtracts_output_reservation_and_margin() {
        let fixture = ContextWindowBudget::new(266_300, 60_000);

        let actual = fixture.effective_input_budget();
        let expected = Some(173_532);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_effective_input_budget_caps_266k_window_with_120k_output_reservation() {
        let fixture = ContextWindowBudget::new(266_300, 120_000);

        let actual = fixture.effective_input_budget();
        let expected = Some(113_532);

        assert_eq!(actual, expected);
        assert!(actual.expect("budget should fit") < 250_000);
    }

    #[test]
    fn test_context_window_budget_safety_margin_is_capped_for_large_windows() {
        let fixture = 266_300;

        let actual = ContextWindowBudget::context_window_safety_margin(fixture);
        let expected = 32_768;

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_context_window_budget_returns_none_when_output_consumes_window() {
        let fixture = ContextWindowBudget::new(8_000, 7_000);

        let actual = fixture.effective_input_budget();
        let expected = None;

        assert_eq!(actual, expected);
    }
}
