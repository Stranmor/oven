use derive_more::derive::Display;
use derive_setters::Setters;
use fake::Dummy;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use strum_macros::EnumString;

use crate::ProviderId;

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
}
