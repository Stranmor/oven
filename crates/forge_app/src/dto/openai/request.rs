use std::vec;

use derive_more::derive::Display;
use derive_setters::Setters;
use forge_json_repair::coerce_to_schema;
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;
use url::Url;

use super::response::{ExtraContent, FunctionCall, ToolCall};
use super::tool_choice::ToolChoice;
use crate::domain::{
    Context, ContextMessage, ContextWindowBudget, ModelId, Provider, ToolCallFull, ToolCallId,
    ToolCatalog, ToolDefinition, ToolName, ToolResult, ToolValue, Transformer,
};
use crate::dto::openai::ReasoningDetail;
use crate::dto::openai::transformers::ProviderPipeline;

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Message {
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<ToolName>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<ToolCallId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_details: Option<Vec<ReasoningDetail>>,
    // GitHub Copilot format (flat fields instead of array)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_opaque: Option<String>,
    // kimi_k2 uses reasoning_content as flat string (similar to reasoning_text but aliased)
    #[serde(skip_serializing_if = "Option::is_none", rename = "reasoning_content")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_content: Option<ExtraContent>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    pub fn cached(self, enable_cache: bool) -> Self {
        let cache_control =
            enable_cache.then_some(CacheControl { type_: CacheControlType::Ephemeral });

        match self {
            MessageContent::Text(text) => {
                if let Some(cc) = cache_control {
                    MessageContent::Parts(vec![ContentPart::Text { text, cache_control: Some(cc) }])
                } else {
                    MessageContent::Text(text)
                }
            }
            MessageContent::Parts(mut parts) => {
                parts.iter_mut().for_each(ContentPart::reset_cache);
                match cache_control {
                    Some(_) => {
                        // cache the last part of the message
                        if let Some(part) = parts.last_mut() {
                            part.cached(enable_cache)
                        }
                        MessageContent::Parts(parts)
                    }
                    None => MessageContent::Parts(parts),
                }
            }
        }
    }

    pub fn is_cached(&self) -> bool {
        match self {
            MessageContent::Text(_) => false,
            MessageContent::Parts(parts) => parts.iter().any(|part| {
                if let ContentPart::Text { cache_control, .. } = part {
                    cache_control.is_some()
                } else {
                    false
                }
            }),
        }
    }

    fn media_token_padding(&self) -> usize {
        match self {
            MessageContent::Text(_) => 0,
            MessageContent::Parts(parts) => {
                parts.iter().map(ContentPart::media_token_padding).sum()
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ImageUrl {
        image_url: ImageUrl,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

impl ContentPart {
    pub fn reset_cache(&mut self) {
        match self {
            ContentPart::Text { cache_control, .. } => {
                *cache_control = None;
            }
            ContentPart::ImageUrl { cache_control, .. } => {
                *cache_control = None;
            }
        }
    }

    pub fn cached(&mut self, enable_cache: bool) {
        let src_cache_control =
            enable_cache.then_some(CacheControl { type_: CacheControlType::Ephemeral });
        match self {
            ContentPart::Text { cache_control, .. } => {
                *cache_control = src_cache_control;
            }
            ContentPart::ImageUrl { cache_control, .. } => {
                *cache_control = src_cache_control;
            }
        }
    }

    fn media_token_padding(&self) -> usize {
        match self {
            ContentPart::Text { .. } => 0,
            ContentPart::ImageUrl { image_url, .. } => {
                if image_url.url.trim_start().starts_with("data:") {
                    image_url.url.len().div_ceil(3)
                } else {
                    2_048
                }
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub type_: CacheControlType,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CacheControlType {
    Ephemeral,
}

/// Describes an OpenAI-compatible function tool.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct FunctionDescription {
    /// Optional natural-language function description sent to the provider.
    pub description: Option<String>,
    /// Function name used by the model in tool calls.
    pub name: ToolName,
    /// JSON schema for accepted function arguments.
    pub parameters: schemars::Schema,
}

/// Tool definition encoded for OpenAI-compatible chat/completions requests.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type")]
pub enum Tool {
    /// Function tool with a JSON-schema argument contract.
    #[serde(rename = "function")]
    Function { function: FunctionDescription },
    /// Provider-native code interpreter tool marker.
    #[serde(rename = "code_interpreter")]
    CodeInterpreter,
}

/// Response format configuration for OpenAI API
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type", content = "json_schema")]
pub enum ResponseFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_schema")]
    JsonSchema {
        name: String,
        schema: Box<schemars::Schema>,
    },
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct Prediction {
    pub r#type: String,
    pub content: String,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ProviderPreferences {
    // Define fields as necessary
}

/// Z.ai-specific thinking type
///
/// Represents the state of thinking for z.ai providers
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingType {
    Enabled,
    Disabled,
}

/// Z.ai-specific thinking configuration structure
///
/// Z.ai uses a different format than standard OpenAI reasoning configuration.
/// This struct represents z.ai's thinking format: `{"type": "enabled"}` or
/// `{"type": "disabled"}`
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ThinkingConfig {
    /// Type of thinking configuration - enabled or disabled
    #[serde(rename = "type")]
    pub r#type: ThinkingType,
}

/// Diagnostic estimate for a final serialized provider request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderRequestEstimate {
    /// Conservative input token estimate used by local context-window guards.
    pub estimated_input_tokens: usize,
    /// Final JSON payload byte length before media padding.
    pub serialized_request_bytes: usize,
    /// Additional token padding for media payloads.
    pub media_token_padding: usize,
    /// Output tokens reserved by the final provider request.
    pub output_token_reservation: usize,
    /// Number of provider messages in the final request.
    pub message_count: usize,
    /// Number of provider tools in the final request.
    pub tool_count: usize,
    /// Serialized bytes of the messages field.
    pub messages_bytes: usize,
    /// Serialized bytes of the tools field.
    pub tools_bytes: usize,
}

impl ProviderRequestEstimate {
    /// Builds a final provider request estimate from a serialized request body.
    ///
    /// # Arguments
    /// * `serialized_request` - Final JSON payload bytes before media padding.
    /// * `media_token_padding` - Additional media padding included in the input estimate.
    /// * `output_token_reservation` - Output tokens reserved by the final provider request.
    /// * `message_count` - Number of provider messages in the final request.
    /// * `tool_count` - Number of provider tools in the final request.
    /// * `messages_bytes` - Serialized byte length of the provider messages field.
    /// * `tools_bytes` - Serialized byte length of the provider tools field.
    pub fn from_serialized_request(
        serialized_request: &[u8],
        media_token_padding: usize,
        output_token_reservation: usize,
        message_count: usize,
        tool_count: usize,
        messages_bytes: usize,
        tools_bytes: usize,
    ) -> Self {
        let serialized_request_bytes = serialized_request.len();
        Self {
            estimated_input_tokens: estimate_serialized_text_tokens(serialized_request)
                .saturating_add(media_token_padding),
            serialized_request_bytes,
            media_token_padding,
            output_token_reservation,
            message_count,
            tool_count,
            messages_bytes,
            tools_bytes,
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, Setters, Default)]
#[setters(strip_option)]
pub struct Request {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<Message>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<std::collections::HashMap<u32, f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_a: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<Prediction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transforms: Option<Vec<Transform>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderPreferences>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Indicates whether the request is user- or agent-initiated.
    #[serde(skip_serializing)]
    pub initiator: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<forge_domain::ReasoningConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    #[serde(skip)]
    pub message_cache_eligibility: Vec<bool>,
    #[serde(skip)]
    pub context_window: Option<u64>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct StreamOptions {
    pub include_usage: Option<bool>,
}

impl Request {
    /// Builds the exact OpenAI-compatible request shape used by provider dispatch.
    ///
    /// # Arguments
    /// * `context` - Domain context to convert into an OpenAI-compatible request.
    /// * `model` - Provider-local model identifier attached to the request.
    /// * `provider` - Provider metadata driving compatibility transformations.
    /// * `merge_system_messages` - Whether system messages should be merged before dispatch.
    ///
    /// # Errors
    /// Returns an error when image payloads cannot be canonicalized or when a
    /// function tool parameter schema cannot be proven to have an object root
    /// accepted by OpenAI-compatible providers.
    pub fn from_context_for_provider(
        context: Context,
        model: &ModelId,
        provider: &Provider<Url>,
        merge_system_messages: bool,
    ) -> anyhow::Result<Self> {
        let mut request = Request::from(context).model(model.clone());
        request.validate_and_canonicalize_images()?;
        let mut pipeline = ProviderPipeline::new(provider, merge_system_messages);
        let mut request = pipeline.transform(request);
        request.validate_and_canonicalize_images()?;
        request.validate_tool_parameter_schemas()?;
        Ok(request)
    }

    /// Estimates input tokens and serialized payload bytes from the final provider request.
    ///
    /// # Arguments
    /// * `context` - Domain context to convert into the provider request.
    /// * `model` - Provider-local model identifier attached to the request.
    /// * `provider` - Provider metadata driving compatibility transformations.
    /// * `merge_system_messages` - Whether system messages should be merged before dispatch.
    ///
    /// # Errors
    /// Returns an error when request conversion or serialization fails.
    pub fn estimate_provider_request(
        context: Context,
        model: &ModelId,
        provider: &Provider<Url>,
        merge_system_messages: bool,
    ) -> anyhow::Result<ProviderRequestEstimate> {
        let request =
            Self::from_context_for_provider(context, model, provider, merge_system_messages)?;
        let serialized_request = serde_json::to_vec(&request)?;
        Ok(ProviderRequestEstimate::from_serialized_request(
            &serialized_request,
            request.media_token_padding(),
            request.output_token_reservation(),
            request.message_count(),
            request.tools.as_ref().map(|tools| tools.len()).unwrap_or(0),
            serialized_optional_bytes(&request.messages)?,
            serialized_optional_bytes(&request.tools)?,
        ))
    }

    /// Estimates input tokens from the final serialized OpenAI-compatible provider request.
    ///
    /// # Arguments
    /// * `context` - Domain context to convert into the provider request.
    /// * `model` - Provider-local model identifier attached to the request.
    /// * `provider` - Provider metadata driving compatibility transformations.
    /// * `merge_system_messages` - Whether system messages should be merged before dispatch.
    ///
    /// # Errors
    /// Returns an error when request conversion or serialization fails.
    pub fn estimate_provider_input_tokens(
        context: Context,
        model: &ModelId,
        provider: &Provider<Url>,
        merge_system_messages: bool,
    ) -> anyhow::Result<usize> {
        Ok(
            Self::estimate_provider_request(context, model, provider, merge_system_messages)?
                .estimated_input_tokens,
        )
    }

    pub fn validate_and_canonicalize_images(&mut self) -> anyhow::Result<()> {
        for message in self.messages.iter_mut().flatten() {
            if let Some(MessageContent::Parts(parts)) = &mut message.content {
                for part in parts {
                    if let ContentPart::ImageUrl { image_url, .. } = part {
                        let trimmed_url = image_url.url.trim();
                        if trimmed_url
                            .get(.."data:".len())
                            .is_some_and(|scheme| scheme.eq_ignore_ascii_case("data:"))
                        {
                            image_url.url =
                                crate::domain::Image::canonicalize_data_url(&image_url.url)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_tool_parameter_schemas(&self) -> anyhow::Result<()> {
        for tool in self.tools.iter().flatten() {
            let Tool::Function { function } = tool else {
                continue;
            };
            let schema_value = serde_json::to_value(&function.parameters)?;
            let root_type = schema_value
                .as_object()
                .and_then(|schema| schema.get("type"))
                .and_then(|schema_type| schema_type.as_str());
            if root_type != Some("object") {
                anyhow::bail!(
                    "OpenAI-compatible provider request blocked before dispatch: tool function '{}' has parameters root schema type '{}', but function parameters must be a JSON Schema object. Fix the ToolDefinition.input_schema at its source.",
                    function.name.as_str(),
                    root_type.unwrap_or("<missing-or-non-string>")
                );
            }
        }
        Ok(())
    }

    pub fn message_count(&self) -> usize {
        self.messages
            .as_ref()
            .map(|messages| messages.len())
            .unwrap_or(0)
    }

    pub fn message_cache_count(&self) -> usize {
        self.messages
            .iter()
            .flatten()
            .flat_map(|a| a.content.as_ref())
            .enumerate()
            .map(|(i, _)| i)
            .max()
            .unwrap_or(0)
    }

    /// Returns whether the provider message at `index` may receive a
    /// prompt-cache marker.
    pub fn is_message_cache_eligible(&self, index: usize) -> bool {
        self.message_cache_eligibility
            .get(index)
            .copied()
            .unwrap_or(true)
    }

    /// Returns the output-token reservation encoded by this provider request.
    pub fn output_token_reservation(&self) -> usize {
        self.max_completion_tokens
            .or(self.max_tokens)
            .and_then(|tokens| usize::try_from(tokens).ok())
            .unwrap_or(ContextWindowBudget::DEFAULT_OUTPUT_TOKEN_RESERVATION)
    }

    /// Returns a conservative token estimate for the request input after
    /// provider pipeline transformations and JSON serialization.
    ///
    /// # Arguments
    /// * `serialized_request` - Final JSON payload bytes that would be sent to
    ///   the provider.
    pub fn estimated_input_tokens_from_serialized(&self, serialized_request: &[u8]) -> usize {
        estimate_serialized_text_tokens(serialized_request)
            .saturating_add(self.media_token_padding())
    }

    /// Validates that the final serialized provider request fits the known
    /// model context window.
    ///
    /// # Arguments
    /// * `serialized_request` - Final JSON payload bytes that would be sent to
    ///   the provider.
    ///
    /// # Errors
    /// Returns a local actionable error when context-window safety cannot be
    /// proven or the final request is too large.
    pub fn validate_context_window(&self, serialized_request: &[u8]) -> anyhow::Result<()> {
        let context_window = self
            .context_window
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "OpenAI-compatible context-window guard cannot prove safety for model '{}' because context_length metadata is missing. Add context_length to the model metadata or select a model with known context window.",
                    self.model
                        .as_ref()
                        .map(|model| model.as_str())
                        .unwrap_or("<unknown>")
                )
            })?;

        let output_reservation = self.output_token_reservation();
        let context_budget = ContextWindowBudget::new(context_window, output_reservation);
        let input_budget = context_budget.effective_input_budget().ok_or_else(|| {
            anyhow::anyhow!(
                "OpenAI-compatible context-window guard blocked request for model '{}'. Context window is {} tokens, reserved output is {} tokens, and safety margin is {} tokens, leaving no safe prompt budget. Lower max_tokens/max_completion_tokens or select a larger-context model.",
                self.model
                    .as_ref()
                    .map(|model| model.as_str())
                    .unwrap_or("<unknown>"),
                context_budget.context_window(),
                context_budget.output_reservation(),
                context_budget.safety_margin()
            )
        })?;
        let estimated_input = self.estimated_input_tokens_from_serialized(serialized_request);

        if estimated_input <= input_budget {
            return Ok(());
        }

        anyhow::bail!(
            "OpenAI-compatible context-window guard blocked an oversized request before HTTP dispatch. Model '{}' has context window {} tokens; reserved output is {} tokens; safety margin is {} tokens; effective input budget is {} tokens; final serialized request estimate is {} tokens. Reduce context, lower max_tokens/max_completion_tokens, or select a larger-context model.",
            self.model
                .as_ref()
                .map(|model| model.as_str())
                .unwrap_or("<unknown>"),
            context_budget.context_window(),
            context_budget.output_reservation(),
            context_budget.safety_margin(),
            input_budget,
            estimated_input
        )
    }

    fn media_token_padding(&self) -> usize {
        self.messages
            .iter()
            .flatten()
            .flat_map(|message| message.content.as_ref())
            .map(MessageContent::media_token_padding)
            .sum()
    }
}

/// ref: https://openrouter.ai/docs/transforms
#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Transform {
    #[default]
    #[serde(rename = "middle-out")]
    MiddleOut,
}

impl From<ToolDefinition> for Tool {
    fn from(value: ToolDefinition) -> Self {
        Tool::Function {
            function: FunctionDescription {
                description: Some(value.description),
                name: value.name,
                parameters: {
                    let mut params = value.input_schema;
                    if let Some(obj) = params.as_object_mut()
                        && obj.get("type") == Some(&serde_json::Value::String("object".to_string()))
                        && !obj.contains_key("properties")
                    {
                        obj.insert(
                            "properties".to_string(),
                            serde_json::Value::Object(serde_json::Map::new()),
                        );
                    }
                    params
                },
            },
        }
    }
}

impl From<Context> for Request {
    fn from(context: Context) -> Self {
        let message_cache_eligibility = context
            .messages
            .iter()
            .map(|message| message.is_cache_eligible())
            .collect::<Vec<_>>();

        Request {
            messages: {
                let messages = context
                    .messages
                    .into_iter()
                    .map(|msg| Message::from(msg.message))
                    .collect::<Vec<_>>();

                Some(messages)
            },
            tools: {
                let tools = context
                    .tools
                    .into_iter()
                    .map(Tool::from)
                    .collect::<Vec<_>>();
                if tools.is_empty() { None } else { Some(tools) }
            },
            model: None,
            prompt: Default::default(),
            response_format: context.response_format.map(|rf| match rf {
                forge_domain::ResponseFormat::Text => ResponseFormat::Text,
                forge_domain::ResponseFormat::JsonSchema(schema) => {
                    // Extract name from schema title
                    let name = schema
                        .as_value()
                        .as_object()
                        .and_then(|obj| obj.get("title"))
                        .and_then(|t| t.as_str())
                        .map(String::from)
                        .unwrap_or_else(|| "schema".to_string());

                    ResponseFormat::JsonSchema { name, schema }
                }
            }),
            stop: Default::default(),
            stream: Some(context.stream.unwrap_or(true)),
            max_tokens: context.max_tokens.map(|t| t as u32),
            temperature: context.temperature.map(|t| t.value()),
            tool_choice: context.tool_choice.map(|tc| tc.into()),
            seed: Default::default(),
            top_p: context.top_p.map(|t| t.value()),
            top_k: context.top_k.map(|t| t.value()),
            frequency_penalty: Default::default(),
            presence_penalty: Default::default(),
            repetition_penalty: Default::default(),
            logit_bias: Default::default(),
            top_logprobs: Default::default(),
            min_p: Default::default(),
            top_a: Default::default(),
            prediction: Default::default(),
            // Since compaction is support on the client we don't need middle-out transforms any
            // more
            transforms: Default::default(),
            models: Default::default(),
            route: Default::default(),
            provider: Default::default(),
            parallel_tool_calls: Some(true), /* Default to true, transformers will adjust based
                                              * on model capabilities */
            stream_options: Some(StreamOptions { include_usage: Some(true) }),
            session_id: context.conversation_id.map(|id| id.to_string()),
            initiator: context.initiator.map(|initiator| match initiator {
                forge_domain::Initiator::User => "user".to_string(),
                forge_domain::Initiator::Agent => "agent".to_string(),
            }),
            reasoning: context.reasoning,
            reasoning_effort: Default::default(),
            max_completion_tokens: Default::default(),
            thinking: Default::default(),
            message_cache_eligibility,
            context_window: context.model_context_length,
        }
    }
}

fn estimate_serialized_text_tokens(serialized_request: &[u8]) -> usize {
    std::str::from_utf8(serialized_request)
        .map(|text| text.chars().count().div_ceil(4))
        .unwrap_or(serialized_request.len())
}

fn serialized_optional_bytes<T: Serialize>(value: &Option<T>) -> anyhow::Result<usize> {
    value
        .as_ref()
        .map(serde_json::to_vec)
        .transpose()
        .map(|bytes| bytes.map(|bytes| bytes.len()).unwrap_or(0))
        .map_err(Into::into)
}

fn serialize_tool_call_arguments(tool_call: &ToolCallFull) -> String {
    let serialized_arguments =
        || serde_json::to_string(&tool_call.arguments).unwrap_or_else(|_| "{}".to_string());

    let Ok(parsed_arguments) = tool_call.arguments.parse() else {
        return serialized_arguments();
    };

    let normalized_arguments = ToolCatalog::iter()
        .find(|tool| tool.definition().name == tool_call.name)
        .map(|tool| coerce_to_schema(parsed_arguments.clone(), &tool.definition().input_schema))
        .unwrap_or(parsed_arguments);

    serde_json::to_string(&normalized_arguments).unwrap_or_else(|_| serialized_arguments())
}

impl From<ToolCallFull> for ToolCall {
    fn from(value: ToolCallFull) -> Self {
        let arguments = serialize_tool_call_arguments(&value);
        let extra_content = value.thought_signature.map(ExtraContent::from);

        Self::Function {
            id: value.call_id,
            function: FunctionCall { arguments, name: Some(value.name) },
            extra_content,
        }
    }
}

impl From<ContextMessage> for Message {
    fn from(value: ContextMessage) -> Self {
        match value {
            ContextMessage::Text(chat_message) => Message {
                role: chat_message.role.into(),
                content: Some(MessageContent::Text(chat_message.content)),
                name: None,
                tool_call_id: None,
                tool_calls: chat_message
                    .tool_calls
                    .map(|tool_calls| tool_calls.into_iter().map(ToolCall::from).collect()),
                reasoning_details: chat_message.reasoning_details.map(|details| {
                    details
                        .into_iter()
                        .map(|detail| ReasoningDetail {
                            r#type: detail
                                .type_of
                                .unwrap_or_else(|| "reasoning.text".to_string()),
                            text: detail.text,
                            signature: detail.signature,
                            data: detail.data,
                            id: detail.id,
                            format: detail.format,
                            index: detail.index,
                        })
                        .collect::<Vec<ReasoningDetail>>()
                }),
                reasoning_text: None,
                reasoning_opaque: None,
                reasoning_content: None,
                extra_content: chat_message.thought_signature.map(ExtraContent::from),
            },
            ContextMessage::Tool(tool_result) => Message {
                role: Role::Tool,
                tool_call_id: tool_result.call_id.clone(),
                name: Some(tool_result.name.clone()),
                content: Some(tool_result.into()),
                tool_calls: None,
                reasoning_details: None,
                reasoning_text: None,
                reasoning_opaque: None,
                reasoning_content: None,
                extra_content: None,
            },
            ContextMessage::Image(img) => {
                let content = vec![ContentPart::ImageUrl {
                    image_url: ImageUrl { url: img.url().clone(), detail: None },
                    cache_control: None,
                }];
                Message {
                    role: Role::User,
                    content: Some(MessageContent::Parts(content)),
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                    reasoning_details: None,
                    reasoning_text: None,
                    reasoning_opaque: None,
                    reasoning_content: None,
                    extra_content: None,
                }
            }
        }
    }
}

impl From<ToolResult> for MessageContent {
    fn from(result: ToolResult) -> Self {
        if result.output.values.len() == 1
            && let Some(text) = result.output.as_str()
        {
            return MessageContent::Text(text.to_string());
        }
        let mut parts = Vec::new();
        for value in result.output.values.into_iter() {
            match value {
                ToolValue::Text(text) => {
                    parts.push(ContentPart::Text { text, cache_control: None });
                }
                ToolValue::Image(img) => {
                    let content = ContentPart::ImageUrl {
                        image_url: ImageUrl { url: img.url().clone(), detail: None },
                        cache_control: None,
                    };
                    parts.push(content);
                }
                ToolValue::Empty => {
                    // Handle empty case if needed
                }
                ToolValue::AI { value, .. } => {
                    parts.push(ContentPart::Text { text: value, cache_control: None })
                }
            }
        }

        MessageContent::Parts(parts)
    }
}

impl From<forge_domain::Role> for Role {
    fn from(role: forge_domain::Role) -> Self {
        match role {
            forge_domain::Role::System => Role::System,
            forge_domain::Role::User => Role::User,
            forge_domain::Role::Assistant => Role::Assistant,
        }
    }
}

#[derive(Debug, Deserialize, Display, Serialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_context_window_guard_counts_max_completion_tokens_after_pipeline_mapping() {
        let fixture = Request {
            model: Some(ModelId::new("context-guard-model")),
            messages: Some(vec![Message {
                role: super::Role::User,
                content: Some(MessageContent::Text("short".to_string())),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                reasoning_details: None,
                reasoning_text: None,
                reasoning_opaque: None,
                reasoning_content: None,
                extra_content: None,
            }]),
            max_completion_tokens: Some(7_000),
            context_window: Some(8_000),
            ..Default::default()
        };
        let serialized = serde_json::to_vec(&fixture).unwrap();

        let actual = fixture
            .validate_context_window(&serialized)
            .unwrap_err()
            .to_string();
        let expected = true;

        assert_eq!(actual.contains("leaving no safe prompt budget"), expected);
    }

    #[test]
    fn test_context_window_guard_blocks_high_output_budget_oversized_request() {
        let fixture = Request {
            model: Some(ModelId::new("context-guard-model")),
            messages: Some(vec![Message {
                role: super::Role::User,
                content: Some(MessageContent::Text("x".repeat(920_000))),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                reasoning_details: None,
                reasoning_text: None,
                reasoning_opaque: None,
                reasoning_content: None,
                extra_content: None,
            }]),
            max_completion_tokens: Some(60_000),
            context_window: Some(266_300),
            ..Default::default()
        };
        let serialized = serde_json::to_vec(&fixture).unwrap();

        let actual = fixture.validate_context_window(&serialized);

        assert!(actual.is_err());
    }

    #[test]
    fn test_context_window_guard_estimates_ascii_json_as_tokens_not_bytes() {
        let content = "x".repeat(1_000_000);
        let fixture = Request {
            model: Some(ModelId::new("context-guard-model")),
            messages: Some(vec![Message {
                role: super::Role::User,
                content: Some(MessageContent::Text(content)),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                reasoning_details: None,
                reasoning_text: None,
                reasoning_opaque: None,
                reasoning_content: None,
                extra_content: None,
            }]),
            context_window: Some(1_048_576),
            max_completion_tokens: Some(10_444),
            ..Default::default()
        };
        let serialized = serde_json::to_vec(&fixture).unwrap();
        let actual = fixture.estimated_input_tokens_from_serialized(&serialized);

        assert!(
            actual < serialized.len() / 2,
            "estimated tokens must be a token approximation, not serialized bytes: actual={actual}, bytes={}",
            serialized.len()
        );
        assert!(
            actual > 240_000,
            "large JSON text should still have a substantial estimate"
        );
    }

    #[test]
    fn test_context_window_guard_allows_incident_sized_text_heavy_request() {
        let fixture = Request {
            model: Some(ModelId::new("gpt-5.5")),
            messages: Some(vec![Message {
                role: super::Role::User,
                content: Some(MessageContent::Text("x".repeat(1_020_000))),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                reasoning_details: None,
                reasoning_text: None,
                reasoning_opaque: None,
                reasoning_content: None,
                extra_content: None,
            }]),
            tools: Some(
                (0..12)
                    .map(|index| Tool::Function {
                        function: FunctionDescription {
                            description: Some(format!(
                                "tool {index} {}",
                                "description ".repeat(400)
                            )),
                            name: ToolName::new(format!("tool_{index}")),
                            parameters: schemars::schema_for!(serde_json::Value),
                        },
                    })
                    .collect(),
            ),
            max_completion_tokens: Some(10_444),
            context_window: Some(1_048_576),
            ..Default::default()
        };
        let serialized = serde_json::to_vec(&fixture).unwrap();

        assert!(serialized.len() > 1_020_000);
        assert!(fixture.validate_context_window(&serialized).is_ok());
    }

    #[test]
    fn test_context_window_guard_estimates_tool_schemas_by_tokens_not_bytes() {
        let fixture = Request {
            model: Some(ModelId::new("context-guard-model")),
            messages: Some(vec![Message {
                role: super::Role::User,
                content: Some(MessageContent::Text("short".to_string())),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                reasoning_details: None,
                reasoning_text: None,
                reasoning_opaque: None,
                reasoning_content: None,
                extra_content: None,
            }]),
            tools: Some(
                (0..100)
                    .map(|index| Tool::Function {
                        function: FunctionDescription {
                            description: Some("large schema description ".repeat(100)),
                            name: ToolName::new(format!("tool_{index}")),
                            parameters: schemars::schema_for!(serde_json::Value),
                        },
                    })
                    .collect(),
            ),
            context_window: Some(128_000),
            ..Default::default()
        };
        let serialized = serde_json::to_vec(&fixture).unwrap();
        let actual = fixture.estimated_input_tokens_from_serialized(&serialized);

        assert!(actual < serialized.len() / 2);
        assert!(actual > 50_000);
    }

    #[test]
    fn test_context_window_guard_includes_media_padding_in_token_estimate() {
        let request_without_media = Request {
            model: Some(ModelId::new("context-guard-model")),
            messages: Some(vec![Message {
                role: super::Role::User,
                content: Some(MessageContent::Text("short".to_string())),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                reasoning_details: None,
                reasoning_text: None,
                reasoning_opaque: None,
                reasoning_content: None,
                extra_content: None,
            }]),
            tools: Some(vec![Tool::Function {
                function: FunctionDescription {
                    description: Some("x".repeat(12_000)),
                    name: ToolName::new("large_tool"),
                    parameters: schemars::schema_for!(()),
                },
            }]),
            context_window: Some(64_000),
            ..Default::default()
        };
        let fixture = Request {
            messages: Some(vec![Message {
                role: super::Role::User,
                content: Some(MessageContent::Parts(vec![ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/image.png".to_string(),
                        detail: None,
                    },
                    cache_control: None,
                }])),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                reasoning_details: None,
                reasoning_text: None,
                reasoning_opaque: None,
                reasoning_content: None,
                extra_content: None,
            }]),
            ..request_without_media.clone()
        };
        let serialized_without_media = serde_json::to_vec(&request_without_media).unwrap();
        let serialized_with_media = serde_json::to_vec(&fixture).unwrap();

        let actual = fixture.estimated_input_tokens_from_serialized(&serialized_with_media)
            > request_without_media
                .estimated_input_tokens_from_serialized(&serialized_without_media);
        let expected = true;

        assert_eq!(actual, expected);
    }

    #[test]
    fn validate_and_canonicalize_images_rejects_unsupported_mime() {
        let mut fixture = Request {
            messages: Some(vec![Message {
                role: super::Role::User,
                content: Some(MessageContent::Parts(vec![ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "data:image/bmp;base64,AAAA".to_string(),
                        detail: None,
                    },
                    cache_control: None,
                }])),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                reasoning_details: None,
                reasoning_text: None,
                reasoning_opaque: None,
                reasoning_content: None,
                extra_content: None,
            }]),
            ..Default::default()
        };
        let actual = fixture
            .validate_and_canonicalize_images()
            .unwrap_err()
            .to_string();
        let expected = "Unsupported image MIME type: image/bmp".to_string();
        assert_eq!(actual, expected);
    }

    #[test]
    fn validate_and_canonicalize_images_rejects_uppercase_unsupported_data_uri() {
        let mut fixture = Request {
            messages: Some(vec![Message {
                role: super::Role::User,
                content: Some(MessageContent::Parts(vec![ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "DATA:image/bmp;base64,AAAA".to_string(),
                        detail: None,
                    },
                    cache_control: None,
                }])),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                reasoning_details: None,
                reasoning_text: None,
                reasoning_opaque: None,
                reasoning_content: None,
                extra_content: None,
            }]),
            ..Default::default()
        };
        assert!(fixture.validate_and_canonicalize_images().is_err());
    }

    #[test]
    fn validate_and_canonicalize_images_normalizes_payload() {
        let mut fixture = Request {
            messages: Some(vec![Message {
                role: super::Role::User,
                content: Some(MessageContent::Parts(vec![ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "data:image/png;base64,iVBO\nRw0KGgo=".to_string(),
                        detail: None,
                    },
                    cache_control: None,
                }])),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                reasoning_details: None,
                reasoning_text: None,
                reasoning_opaque: None,
                reasoning_content: None,
                extra_content: None,
            }]),
            ..Default::default()
        };
        fixture.validate_and_canonicalize_images().unwrap();
        let actual = fixture.messages.unwrap()[0].content.clone().unwrap();
        let expected = MessageContent::Parts(vec![ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "data:image/png;base64,iVBORw0KGgo=".to_string(),
                detail: None,
            },
            cache_control: None,
        }]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_cached_text_true() {
        let fixture = MessageContent::Text("hello".to_string());
        let actual = fixture.cached(true);
        let expected = MessageContent::Parts(vec![ContentPart::Text {
            text: "hello".to_string(),
            cache_control: Some(CacheControl { type_: CacheControlType::Ephemeral }),
        }]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_cached_text_false() {
        let fixture = MessageContent::Text("hello".to_string());
        let actual = fixture.cached(false);
        let expected = MessageContent::Text("hello".to_string());
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_cached_parts_true() {
        let fixture = MessageContent::Parts(vec![
            ContentPart::Text { text: "a".to_string(), cache_control: None },
            ContentPart::ImageUrl {
                image_url: ImageUrl { url: "http://example.com/a.png".to_string(), detail: None },
                cache_control: Some(CacheControl { type_: CacheControlType::Ephemeral }),
            },
        ]);
        let actual = fixture.cached(true);
        let expected = MessageContent::Parts(vec![
            ContentPart::Text { text: "a".to_string(), cache_control: None },
            ContentPart::ImageUrl {
                image_url: ImageUrl { url: "http://example.com/a.png".to_string(), detail: None },
                cache_control: Some(CacheControl { type_: CacheControlType::Ephemeral }),
            },
        ]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_cached_parts_multi_false() {
        let fixture = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "a".to_string(),
                cache_control: Some(CacheControl { type_: CacheControlType::Ephemeral }),
            },
            ContentPart::Text {
                text: "b".to_string(),
                cache_control: Some(CacheControl { type_: CacheControlType::Ephemeral }),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl { url: "http://example.com/a.png".to_string(), detail: None },
                cache_control: Some(CacheControl { type_: CacheControlType::Ephemeral }),
            },
        ]);
        let actual = fixture.cached(false);
        let expected = MessageContent::Parts(vec![
            ContentPart::Text { text: "a".to_string(), cache_control: None },
            ContentPart::Text { text: "b".to_string(), cache_control: None },
            ContentPart::ImageUrl {
                image_url: ImageUrl { url: "http://example.com/a.png".to_string(), detail: None },
                cache_control: None,
            },
        ]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_cached_parts_already_true() {
        let fixture = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "a".to_string(),
                cache_control: Some(CacheControl { type_: CacheControlType::Ephemeral }),
            },
            ContentPart::Text { text: "b".to_string(), cache_control: None },
            ContentPart::ImageUrl {
                image_url: ImageUrl { url: "http://example.com/a.png".to_string(), detail: None },
                cache_control: None,
            },
        ]);
        let actual = fixture.cached(true);
        let expected = MessageContent::Parts(vec![
            ContentPart::Text { text: "a".to_string(), cache_control: None },
            ContentPart::Text { text: "b".to_string(), cache_control: None },
            ContentPart::ImageUrl {
                image_url: ImageUrl { url: "http://example.com/a.png".to_string(), detail: None },
                cache_control: Some(CacheControl { type_: CacheControlType::Ephemeral }),
            },
        ]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_cached_parts_multi_true() {
        let fixture = MessageContent::Parts(vec![
            ContentPart::Text { text: "a".to_string(), cache_control: None },
            ContentPart::Text { text: "b".to_string(), cache_control: None },
            ContentPart::ImageUrl {
                image_url: ImageUrl { url: "http://example.com/a.png".to_string(), detail: None },
                cache_control: None,
            },
        ]);
        let actual = fixture.cached(true);
        let expected = MessageContent::Parts(vec![
            ContentPart::Text { text: "a".to_string(), cache_control: None },
            ContentPart::Text { text: "b".to_string(), cache_control: None },
            ContentPart::ImageUrl {
                image_url: ImageUrl { url: "http://example.com/a.png".to_string(), detail: None },
                cache_control: Some(CacheControl { type_: CacheControlType::Ephemeral }),
            },
        ]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_cached_parts_false() {
        let fixture = MessageContent::Parts(vec![
            ContentPart::Text { text: "a".to_string(), cache_control: None },
            ContentPart::ImageUrl {
                image_url: ImageUrl { url: "http://example.com/a.png".to_string(), detail: None },
                cache_control: None,
            },
        ]);
        let actual = fixture.cached(false);
        let expected = MessageContent::Parts(vec![
            ContentPart::Text { text: "a".to_string(), cache_control: None },
            ContentPart::ImageUrl {
                image_url: ImageUrl { url: "http://example.com/a.png".to_string(), detail: None },
                cache_control: None,
            },
        ]);
        assert_eq!(actual, expected);
    }

    use forge_domain::{
        Agent, AgentId, ContextMessage, ProviderId, Role, TextMessage, ToolCallFull, ToolCallId,
        ToolCatalog, ToolDefinition, ToolName, ToolResult,
    };
    use insta::assert_json_snapshot;

    #[test]
    fn test_dynamic_agent_tool_serializes_object_schema_with_tasks() {
        let fixture = Agent::new(
            AgentId::new("arch-sentinel"),
            ProviderId::OPENAI,
            ModelId::new("gpt-test"),
        )
        .description("Architecture critic");
        let tool_definition: ToolDefinition = fixture.into();

        let actual = Tool::from(tool_definition);
        let Tool::Function { function } = actual else {
            panic!("Expected function tool")
        };
        let actual_schema = serde_json::to_value(function.parameters).unwrap();
        let expected_type = serde_json::json!("object");
        let expected_tasks_type = serde_json::json!("array");

        assert_eq!(function.name, ToolName::new("arch-sentinel"));
        assert_eq!(actual_schema["type"], expected_type);
        assert_eq!(
            actual_schema["properties"]["tasks"]["type"],
            expected_tasks_type
        );
    }

    #[test]
    fn test_provider_request_rejects_root_string_tool_schema_before_dispatch() {
        let fixture = Request::default().tools(vec![Tool::Function {
            function: FunctionDescription {
                name: ToolName::new("arch-sentinel"),
                description: Some("Architecture critic".to_string()),
                parameters: serde_json::from_value(serde_json::json!({
                    "type": "string"
                }))
                .unwrap(),
            },
        }]);

        let actual = fixture
            .validate_tool_parameter_schemas()
            .unwrap_err()
            .to_string();
        let expected = true;

        assert_eq!(actual.contains("arch-sentinel"), expected);
        assert_eq!(actual.contains("root schema type 'string'"), expected);
    }

    #[test]
    fn test_provider_request_accepts_empty_object_tool_schema() {
        let fixture = Request::default().tools(vec![Tool::from(ToolDefinition::new("empty_tool"))]);

        let actual = fixture.validate_tool_parameter_schemas();

        assert!(actual.is_ok());
    }

    #[test]
    fn test_user_message_conversion() {
        let user_message = ContextMessage::Text(
            TextMessage::new(Role::User, "Hello").model(ModelId::new("gpt-3.5-turbo")),
        );
        let router_message = Message::from(user_message);
        assert_json_snapshot!(router_message);
    }

    #[test]
    fn test_message_with_special_chars() {
        let xml_content = r#"Here's some XML content:
<task>
    <id>123</id>
    <description>Test <special> characters</description>
    <data key="value">
        <item>1</item>
        <item>2</item>
    </data>
</task>"#;

        let message = ContextMessage::Text(
            TextMessage::new(Role::User, xml_content).model(ModelId::new("gpt-3.5-turbo")),
        );
        let router_message = Message::from(message);
        assert_json_snapshot!(router_message);
    }

    #[test]
    fn test_assistant_message_with_tool_call_conversion() {
        let tool_call = ToolCallFull {
            call_id: Some(ToolCallId::new("123")),
            name: ToolName::new("test_tool"),
            arguments: serde_json::json!({"key": "value"}).into(),
            thought_signature: None,
        };

        let assistant_message = ContextMessage::Text(
            TextMessage::new(Role::Assistant, "Using tool")
                .tool_calls(vec![tool_call])
                .model(ModelId::new("gpt-3.5-turbo")),
        );
        let router_message = Message::from(assistant_message);
        assert_json_snapshot!(router_message);
    }

    #[test]
    fn test_assistant_message_with_dump_style_tool_call_arguments_conversion() {
        let fixture = ToolCatalog::tool_call_patch(
            "/tmp/file.txt",
            "new text",
            "old text",
            false,
        )
        .arguments(
            serde_json::from_str::<forge_domain::ToolCallArguments>(
                r#""{\"file_path\":\"/tmp/file.txt\",\"old_string\":\"old text\",\"new_string\":\"new text\",\"replace_all\":false}""#,
            )
            .unwrap(),
        )
        .call_id(ToolCallId::new("123"));

        let assistant_message = ContextMessage::Text(
            TextMessage::new(Role::Assistant, "Using tool")
                .tool_calls(vec![fixture])
                .model(ModelId::new("gpt-3.5-turbo")),
        );
        let actual = Message::from(assistant_message);
        let actual =
            serde_json::to_value(actual.tool_calls.expect("Tool calls should exist")).unwrap();
        let expected = serde_json::json!([
            {
                "id": "123",
                "type": "function",
                "function": {
                    "arguments": "{\"file_path\":\"/tmp/file.txt\",\"new_string\":\"new text\",\"old_string\":\"old text\",\"replace_all\":false}",
                    "name": "patch"
                }
            }
        ]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_tool_message_conversion() {
        let tool_result = ToolResult::new(ToolName::new("test_tool"))
            .call_id(ToolCallId::new("123"))
            .success(
                r#"{
               "user": "John",
               "age": 30,
               "address": [{"city": "New York"}, {"city": "San Francisco"}]
            }"#,
            );

        let tool_message = ContextMessage::Tool(tool_result);
        let router_message = Message::from(tool_message);
        assert_json_snapshot!(router_message);
    }

    #[test]
    fn test_tool_message_with_special_chars() {
        let tool_result = ToolResult::new(ToolName::new("html_tool"))
            .call_id(ToolCallId::new("456"))
            .success(
                r#"{
                "html": "<div class=\"container\"><p>Hello <World></p></div>",
                "elements": ["<span>", "<br/>", "<hr>"],
                "attributes": {
                    "style": "color: blue; font-size: 12px;",
                    "data-test": "<test>&value</test>"
                }
            }"#,
            );

        let tool_message = ContextMessage::Tool(tool_result);
        let router_message = Message::from(tool_message);
        assert_json_snapshot!(router_message);
    }

    #[test]
    fn test_tool_message_typescript_code() {
        let tool_result = ToolResult::new(ToolName::new("rust_tool"))
            .call_id(ToolCallId::new("456"))
            .success(r#"{ "code": "fn main<T>(gt: T) {let b = &gt; }"}"#);

        let tool_message = ContextMessage::Tool(tool_result);
        let router_message = Message::from(tool_message);
        assert_json_snapshot!(router_message);
    }

    #[test]
    fn test_tool_call_supports_code_interpreter() {
        let json = r#"{"id":"call_123","type":"code_interpreter","function":{"name":"python","arguments":"{}"}}"#;
        let result: Result<super::ToolCall, _> = serde_json::from_str(json);
        assert!(
            result.is_ok(),
            "ToolCall should support code_interpreter type: {:?}",
            result.unwrap_err()
        );
    }

    #[test]
    fn test_transform_display() {
        assert_eq!(
            serde_json::to_string(&Transform::MiddleOut).unwrap(),
            "\"middle-out\""
        );
    }
    #[test]
    fn test_tool_definition_conversion_missing_properties() {
        // Test case where input_schema is an object type but missing properties field
        let fixture = {
            // In schemars 1.0, Schema wraps serde_json::Value, so we create JSON directly
            let schema_value = serde_json::json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "title": "Null",
                "type": "object"
            });
            let schema = schemars::Schema::try_from(schema_value).unwrap();

            ToolDefinition::new("test_tool")
                .description("Test tool")
                .input_schema(schema)
        };

        let actual = Tool::from(fixture);

        let expected = Tool::Function {
            function: FunctionDescription {
                description: Some("Test tool".to_string()),
                name: ToolName::new("test_tool"),
                parameters: serde_json::from_value(serde_json::json!({
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "properties": {},
                    "title": "Null",
                    "type": "object"
                }))
                .unwrap(),
            },
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_context_conversion_stream_defaults_to_true() {
        let fixture = forge_domain::Context::default();
        let actual = Request::from(fixture);

        assert_eq!(actual.stream, Some(true));
    }

    #[test]
    fn test_context_conversion_maps_agent_initiator() {
        let fixture = forge_domain::Context::default().initiator(forge_domain::Initiator::Agent);
        let actual = Request::from(fixture);

        assert_eq!(actual.initiator, Some("agent".to_string()));
    }

    #[test]
    fn test_context_conversion_maps_user_initiator() {
        let fixture = forge_domain::Context::default().initiator(forge_domain::Initiator::User);
        let actual = Request::from(fixture);

        assert_eq!(actual.initiator, Some("user".to_string()));
    }

    #[test]
    fn test_context_conversion_stream_explicit_true() {
        let fixture = forge_domain::Context::default().stream(true);
        let actual = Request::from(fixture);

        assert_eq!(actual.stream, Some(true));
    }

    #[test]
    fn test_context_conversion_stream_explicit_false() {
        let fixture = forge_domain::Context::default().stream(false);
        let actual = Request::from(fixture);

        assert_eq!(actual.stream, Some(false));
    }

    #[test]
    fn test_response_format_json_schema_serialization() {
        use schemars::JsonSchema;
        use serde::Deserialize;

        #[derive(Deserialize, JsonSchema)]
        #[allow(dead_code)]
        #[schemars(title = "test_response")]
        struct TestResponse {
            message: String,
        }

        let schema = schemars::schema_for!(TestResponse);
        let fixture = forge_domain::Context::default()
            .response_format(forge_domain::ResponseFormat::JsonSchema(Box::new(schema)));

        let actual = Request::from(fixture);

        assert!(actual.response_format.is_some());
        let rf = actual.response_format.unwrap();

        // Serialize to JSON to verify the format
        let json = serde_json::to_string(&rf).unwrap();
        println!("Serialized response_format: {}", json);

        // Should contain type and json_schema fields
        assert!(json.contains("\"type\":\"json_schema\""));
        assert!(json.contains("\"json_schema\""));
    }
}
