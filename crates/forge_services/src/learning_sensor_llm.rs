use std::sync::{Arc, Mutex};

use anyhow::Result;
use forge_app::domain::{
    ChatCompletionMessage, Context, ContextMessage, FinishReason,
    LEARNING_SENSOR_REVIEW_SCHEMA_VERSION, LearningSensorDecisionKind, LearningSensorReviewInput,
    LearningSensorReviewOutput, LearningSensorReviewerIdentity, ResponseFormat,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

const LEARNING_SENSOR_LLM_SYSTEM_INSTRUCTION: &str = "You are a read-only learning Sensor. Review only the provided sanitized LearningSensorReviewInput JSON. Return exactly one JSON object matching the schema. Do not request tools. Do not infer from transcripts, files, repositories, secrets, or external state. Normal runtime conversation_metadata evidence may only be pending or reject. ProposeLesson is allowed only when the input evidence is sanctioned_sanitized_chat_observation with runtime_sanitized_chat_observation provenance, or typed_fixture_observation with fake_reviewer_fixture provenance.";
const LEARNING_SENSOR_LLM_MAX_TOKENS: usize = 768;

/// Narrow Sensor-safe port for live LLM JSON-schema calls.
#[async_trait::async_trait]
pub trait LearningSensorLlmClient: Send + Sync + 'static {
    /// Submits a prebuilt safe learning Sensor request and returns one raw model response.
    ///
    /// # Arguments
    /// * `request` - Structurally safe request DTO produced by the Sensor adapter.
    ///
    /// # Errors
    /// Returns a typed provider error for timeout or infrastructure failures.
    async fn review_json(
        &self,
        request: LearningSensorLlmRequest,
    ) -> std::result::Result<LearningSensorLlmResponse, LearningSensorLlmClientError>;
}

/// Provider failure class visible through the narrow Sensor-safe LLM port.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LearningSensorLlmClientError {
    /// Provider call timed out before completion.
    #[error("learning sensor llm provider timeout")]
    Timeout,
    /// Provider infrastructure failed before a model-level completion was available.
    #[error("learning sensor llm provider infrastructure failure: {0}")]
    ProviderInfra(String),
}

/// Safe request DTO for a learning Sensor LLM JSON-schema call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearningSensorLlmRequest {
    /// Static system instruction for the bounded Sensor role.
    pub system_instruction: String,
    /// Serialized sanitized Sensor input JSON.
    pub input_json: String,
    /// Streaming is always disabled for unambiguous full JSON extraction.
    pub stream: bool,
    /// Bounded output budget.
    pub max_tokens: usize,
    /// JSON schema response-format payload.
    pub response_format: serde_json::Value,
}

impl LearningSensorLlmRequest {
    /// Builds the safe request DTO from sanitized Sensor input.
    ///
    /// # Arguments
    /// * `input` - Sanitized learning Sensor input.
    ///
    /// # Errors
    /// Returns an error when input validation or serialization fails.
    pub fn from_input(input: &LearningSensorReviewInput) -> Result<Self> {
        input.validate()?;
        Ok(Self {
            system_instruction: LEARNING_SENSOR_LLM_SYSTEM_INSTRUCTION.to_string(),
            input_json: serde_json::to_string(input)?,
            stream: false,
            max_tokens: LEARNING_SENSOR_LLM_MAX_TOKENS,
            response_format: learning_sensor_response_format_schema(),
        })
    }

    /// Builds an application context without tools, tool choice, or conversation identity.
    ///
    /// # Errors
    /// Returns an error when the safe request cannot be represented as context.
    pub fn into_context(&self) -> Result<Context> {
        Ok(Context::default()
            .add_message(ContextMessage::system(self.system_instruction.clone()))
            .add_message(ContextMessage::user(self.input_json.clone(), None))
            .stream(false)
            .max_tokens(self.max_tokens)
            .response_format(ResponseFormat::JsonSchema(Box::new(
                serde_json::from_value(
                    self.response_format
                        .get("json_schema")
                        .and_then(|value| value.get("schema"))
                        .cloned()
                        .ok_or_else(|| {
                            anyhow::anyhow!("learning sensor response schema missing")
                        })?,
                )?,
            ))))
    }
}

/// Raw provider response visible through the narrow learning Sensor LLM port.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LearningSensorLlmResponse {
    /// Provider finish reason.
    pub finish_reason: Option<FinishReason>,
    /// Provider content blocks after collection.
    pub content: Vec<LearningSensorLlmResponseContent>,
}

impl LearningSensorLlmResponse {
    /// Creates a successful complete JSON response fixture.
    ///
    /// # Arguments
    /// * `json_content` - Complete JSON object content.
    pub fn complete_json(json_content: impl Into<String>) -> Self {
        Self {
            finish_reason: Some(FinishReason::Stop),
            content: vec![LearningSensorLlmResponseContent::CompleteJson(
                json_content.into(),
            )],
        }
    }
}

impl From<ChatCompletionMessage> for LearningSensorLlmResponse {
    fn from(value: ChatCompletionMessage) -> Self {
        let mut content = Vec::new();
        if let Some(message_content) = value.content {
            let text = message_content.as_str().to_string();
            if message_content.is_part() {
                content.push(LearningSensorLlmResponseContent::PartialText(text));
            } else {
                content.push(LearningSensorLlmResponseContent::CompleteJson(text));
            }
        }
        if value.reasoning.is_some() || value.reasoning_details.is_some() {
            content.push(LearningSensorLlmResponseContent::ReasoningOnly);
        }
        if !value.tool_calls.is_empty() {
            content.push(LearningSensorLlmResponseContent::ToolCall);
        }
        Self { finish_reason: value.finish_reason, content }
    }
}

/// Provider content block visible through the narrow learning Sensor LLM port.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LearningSensorLlmResponseContent {
    /// One complete text content block that must contain only a JSON object.
    CompleteJson(String),
    /// Partial or streaming text is ambiguous and rejected.
    PartialText(String),
    /// Reasoning-only output is ambiguous and rejected.
    ReasoningOnly,
    /// Tool calls are forbidden for Sensor review.
    ToolCall,
}

/// Live learning Sensor reviewer adapter backed by a narrow JSON-schema LLM port.
#[derive(Clone)]
pub struct LiveLearningSensorReviewer<C> {
    client: Arc<C>,
    identity: LearningSensorReviewerIdentity,
    circuit_breaker: Arc<Mutex<LearningSensorCircuitBreakerState>>,
}

impl<C> LiveLearningSensorReviewer<C> {
    /// Creates a live learning Sensor reviewer adapter.
    ///
    /// # Arguments
    /// * `client` - Narrow Sensor-safe LLM client.
    /// * `identity` - Reviewer identity expected in model output.
    pub fn new(client: Arc<C>, identity: LearningSensorReviewerIdentity) -> Self {
        Self {
            client,
            identity,
            circuit_breaker: Arc::new(Mutex::new(Default::default())),
        }
    }

    /// Explicitly resets this adapter instance circuit breaker.
    pub fn reset_circuit_breaker(&self) {
        self.circuit_breaker
            .lock()
            .expect("circuit breaker mutex poisoned")
            .reset();
    }

    /// Returns whether this adapter instance circuit breaker is open.
    pub fn is_circuit_open(&self) -> bool {
        self.circuit_breaker
            .lock()
            .expect("circuit breaker mutex poisoned")
            .is_open()
    }
}

impl<C: LearningSensorLlmClient> LiveLearningSensorReviewer<C> {
    /// Reviews sanitized input through the live Sensor-safe LLM adapter.
    ///
    /// # Arguments
    /// * `input` - Sanitized learning Sensor input.
    pub async fn review(&self, input: LearningSensorReviewInput) -> LearningSensorReviewOutput {
        if self.is_circuit_open() {
            return self.pending_output(&input, "provider_circuit_open");
        }
        let request = match LearningSensorLlmRequest::from_input(&input) {
            Ok(request) => request,
            Err(_) => return self.reject_output(&input, "invalid_safe_sensor_input"),
        };
        match self.client.review_json(request).await {
            Ok(response) => {
                self.circuit_breaker
                    .lock()
                    .expect("circuit breaker mutex poisoned")
                    .reset();
                self.extract_validated_output(&input, response)
            }
            Err(error) => {
                self.circuit_breaker
                    .lock()
                    .expect("circuit breaker mutex poisoned")
                    .record_infra_failure();
                let reason = match error {
                    LearningSensorLlmClientError::Timeout => "provider_timeout",
                    LearningSensorLlmClientError::ProviderInfra(_) => "provider_infra_failure",
                };
                self.pending_output(&input, reason)
            }
        }
    }

    fn extract_validated_output(
        &self,
        input: &LearningSensorReviewInput,
        response: LearningSensorLlmResponse,
    ) -> LearningSensorReviewOutput {
        if response.finish_reason != Some(FinishReason::Stop) {
            return self.reject_output(input, "non_stop_finish_reason");
        }
        let [content] = response.content.as_slice() else {
            return self.reject_output(input, "ambiguous_content_blocks");
        };
        let LearningSensorLlmResponseContent::CompleteJson(text) = content else {
            return self.reject_output(input, "non_json_content_block");
        };
        let trimmed = text.trim();
        if trimmed.is_empty() || !trimmed.starts_with('{') || !trimmed.ends_with('}') {
            return self.reject_output(input, "mixed_or_empty_json_content");
        }
        let parsed = match serde_json::from_str::<LearningSensorReviewOutput>(trimmed) {
            Ok(output) => output,
            Err(_) => return self.reject_output(input, "invalid_json_or_schema"),
        };
        if parsed
            .validate_against_identity(input, &self.identity)
            .is_err()
        {
            return self.reject_output(input, "invalid_sensor_output_contract");
        }
        parsed
    }

    fn pending_output(
        &self,
        input: &LearningSensorReviewInput,
        reason_code: &str,
    ) -> LearningSensorReviewOutput {
        self.output(input, LearningSensorDecisionKind::Pending, reason_code)
    }

    fn reject_output(
        &self,
        input: &LearningSensorReviewInput,
        reason_code: &str,
    ) -> LearningSensorReviewOutput {
        self.output(input, LearningSensorDecisionKind::Reject, reason_code)
    }

    fn output(
        &self,
        input: &LearningSensorReviewInput,
        decision: LearningSensorDecisionKind,
        reason_code: &str,
    ) -> LearningSensorReviewOutput {
        LearningSensorReviewOutput {
            schema_version: LEARNING_SENSOR_REVIEW_SCHEMA_VERSION,
            reviewer_id: self.identity.reviewer_id.clone(),
            reviewer_version: self.identity.reviewer_version,
            input_fingerprint: input
                .fingerprint()
                .unwrap_or_else(|_| "invalid-input-fingerprint".to_string()),
            decision,
            reason_code: reason_code.to_string(),
            proposal_title: None,
            proposal_body: None,
        }
    }
}

#[derive(Debug, Default)]
struct LearningSensorCircuitBreakerState {
    consecutive_infra_failures: u8,
    open: bool,
}

impl LearningSensorCircuitBreakerState {
    fn record_infra_failure(&mut self) {
        self.consecutive_infra_failures = self.consecutive_infra_failures.saturating_add(1);
        if self.consecutive_infra_failures >= 2 {
            self.open = true;
        }
    }

    fn reset(&mut self) {
        self.consecutive_infra_failures = 0;
        self.open = false;
    }

    fn is_open(&self) -> bool {
        self.open
    }
}

fn learning_sensor_response_format_schema() -> serde_json::Value {
    json!({
        "type": "json_schema",
        "json_schema": {
            "name": "learning_sensor_review_output",
            "strict": true,
            "schema": {
                "type": "object",
                "additionalProperties": false,
                "required": [
                    "schema_version",
                    "reviewer_id",
                    "reviewer_version",
                    "input_fingerprint",
                    "decision",
                    "reason_code",
                    "proposal_title",
                    "proposal_body"
                ],
                "properties": {
                    "schema_version": {"type": "integer", "const": LEARNING_SENSOR_REVIEW_SCHEMA_VERSION},
                    "reviewer_id": {"type": "string", "minLength": 1, "maxLength": 128},
                    "reviewer_version": {"type": "integer", "minimum": 1},
                    "input_fingerprint": {"type": "string", "minLength": 1, "maxLength": 128},
                    "decision": {"type": "string", "enum": ["propose_lesson", "pending", "reject"]},
                    "reason_code": {"type": "string", "minLength": 1, "maxLength": 128, "pattern": "^[a-z0-9_]+$"},
                    "proposal_title": {"type": ["string", "null"], "enum": ["typed_fixture_observation", "sanctioned_sanitized_observation", null]},
                    "proposal_body": {"type": ["string", "null"], "enum": ["typed_fixture_substantive_pattern", "validated_counters_and_fingerprints", null]}
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use pretty_assertions::assert_eq;

    use super::*;
    use forge_app::domain::{
        ConversationId, LearningCaptureMetadata, LearningProvenance, LearningRecordId,
        LearningRecordProjection, LearningRedactionStatus, LearningReviewState,
        SanitizedChatLessonObservation, SanitizedChatObservationKind,
        SanitizedObservationCountBucket, SanitizedObservationSeverity, learning_digest_hex,
    };

    #[derive(Default)]
    struct FixtureLearningSensorLlmClient {
        response: Mutex<
            Option<std::result::Result<LearningSensorLlmResponse, LearningSensorLlmClientError>>,
        >,
        calls: AtomicUsize,
        captured_request: Mutex<Option<LearningSensorLlmRequest>>,
    }

    #[async_trait::async_trait]
    impl LearningSensorLlmClient for FixtureLearningSensorLlmClient {
        async fn review_json(
            &self,
            request: LearningSensorLlmRequest,
        ) -> std::result::Result<LearningSensorLlmResponse, LearningSensorLlmClientError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            *self.captured_request.lock().unwrap() = Some(request);
            self.response
                .lock()
                .unwrap()
                .take()
                .unwrap_or_else(|| Ok(LearningSensorLlmResponse::complete_json("{}")))
        }
    }

    impl FixtureLearningSensorLlmClient {
        fn with_response(
            response: std::result::Result<LearningSensorLlmResponse, LearningSensorLlmClientError>,
        ) -> Self {
            Self {
                response: Mutex::new(Some(response)),
                calls: AtomicUsize::new(0),
                captured_request: Mutex::new(None),
            }
        }
    }

    #[test]
    fn learning_sensor_llm_request_has_safe_context_shape() -> Result<()> {
        let fixture = fixture_sensor_input(false);
        let request = LearningSensorLlmRequest::from_input(&fixture)?;
        let context = request.into_context()?;
        let serialized = serde_json::to_string(&request)?.to_ascii_lowercase();

        let actual = (
            context.conversation_id.is_none(),
            context.tools.is_empty(),
            context.tool_choice.is_none(),
            context.stream,
            context.max_tokens,
            context.response_format.is_some(),
            serialized.contains("json_schema"),
            serialized.contains("raw_transcript")
                || serialized.contains("tool_payload")
                || serialized.contains("conversation_id"),
        );
        let expected = (
            true,
            true,
            true,
            Some(false),
            Some(768usize),
            true,
            true,
            false,
        );

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn fake_client_success_returns_valid_typed_output() -> Result<()> {
        let input = fixture_sensor_input(false);
        let output = fixture_output(&input, LearningSensorDecisionKind::Pending);
        let client = Arc::new(FixtureLearningSensorLlmClient::with_response(Ok(
            LearningSensorLlmResponse::complete_json(serde_json::to_string(&output)?),
        )));
        let reviewer =
            LiveLearningSensorReviewer::new(client, LearningSensorReviewerIdentity::fake());

        let actual = reviewer.review(input.clone()).await;
        let expected = output;

        actual.validate_against_identity(&input, &LearningSensorReviewerIdentity::fake())?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn invalid_schema_identity_and_runtime_proposal_map_to_reject() -> Result<()> {
        let input = fixture_sensor_input(false);
        let cases = vec![
            "not json".to_string(),
            format!(
                r#"{{"schema_version":{},"reviewer_id":"{}","reviewer_version":{},"input_fingerprint":"{}","decision":"pending","reason_code":"ok","proposal_title":null,"proposal_body":null,"extra":"blocked"}}"#,
                LEARNING_SENSOR_REVIEW_SCHEMA_VERSION,
                LearningSensorReviewerIdentity::fake().reviewer_id,
                LearningSensorReviewerIdentity::fake().reviewer_version,
                input.fingerprint()?
            ),
            format!(
                r#"{{"schema_version":{},"reviewer_id":"{}","reviewer_version":{},"input_fingerprint":"{}","decision":"accepted","reason_code":"bad","proposal_title":null,"proposal_body":null}}"#,
                LEARNING_SENSOR_REVIEW_SCHEMA_VERSION,
                LearningSensorReviewerIdentity::fake().reviewer_id,
                LearningSensorReviewerIdentity::fake().reviewer_version,
                input.fingerprint()?
            ),
            serde_json::to_string(&LearningSensorReviewOutput {
                reviewer_id: "other".to_string(),
                ..fixture_output(&input, LearningSensorDecisionKind::Pending)
            })?,
            serde_json::to_string(&LearningSensorReviewOutput {
                input_fingerprint: "wrong".to_string(),
                ..fixture_output(&input, LearningSensorDecisionKind::Pending)
            })?,
            serde_json::to_string(&LearningSensorReviewOutput {
                decision: LearningSensorDecisionKind::ProposeLesson,
                proposal_title: Some("Runtime proposal".to_string()),
                proposal_body: Some(
                    "Runtime metadata is not sanctioned fixture evidence".to_string(),
                ),
                ..fixture_output(&input, LearningSensorDecisionKind::Pending)
            })?,
        ];

        let mut actual = Vec::new();
        for case in cases {
            let client = Arc::new(FixtureLearningSensorLlmClient::with_response(Ok(
                LearningSensorLlmResponse::complete_json(case),
            )));
            let reviewer =
                LiveLearningSensorReviewer::new(client, LearningSensorReviewerIdentity::fake());
            actual.push(reviewer.review(input.clone()).await.decision);
        }
        let expected = vec![LearningSensorDecisionKind::Reject; 6];

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn ambiguous_content_shapes_map_to_reject() -> Result<()> {
        let input = fixture_sensor_input(false);
        let valid =
            serde_json::to_string(&fixture_output(&input, LearningSensorDecisionKind::Pending))?;
        let cases = vec![
            LearningSensorLlmResponse {
                finish_reason: Some(FinishReason::Stop),
                content: vec![LearningSensorLlmResponseContent::ToolCall],
            },
            LearningSensorLlmResponse {
                finish_reason: Some(FinishReason::Stop),
                content: vec![
                    LearningSensorLlmResponseContent::CompleteJson(valid.clone()),
                    LearningSensorLlmResponseContent::CompleteJson(valid.clone()),
                ],
            },
            LearningSensorLlmResponse {
                finish_reason: Some(FinishReason::Stop),
                content: Vec::new(),
            },
            LearningSensorLlmResponse {
                finish_reason: Some(FinishReason::Length),
                content: vec![LearningSensorLlmResponseContent::CompleteJson(
                    valid.clone(),
                )],
            },
            LearningSensorLlmResponse {
                finish_reason: Some(FinishReason::Stop),
                content: vec![LearningSensorLlmResponseContent::ReasoningOnly],
            },
            LearningSensorLlmResponse {
                finish_reason: Some(FinishReason::Stop),
                content: vec![LearningSensorLlmResponseContent::CompleteJson(format!(
                    "prefix {valid}"
                ))],
            },
            LearningSensorLlmResponse {
                finish_reason: Some(FinishReason::Stop),
                content: vec![LearningSensorLlmResponseContent::PartialText(valid)],
            },
        ];

        let mut actual = Vec::new();
        for case in cases {
            let client = Arc::new(FixtureLearningSensorLlmClient::with_response(Ok(case)));
            let reviewer =
                LiveLearningSensorReviewer::new(client, LearningSensorReviewerIdentity::fake());
            actual.push(reviewer.review(input.clone()).await.decision);
        }
        let expected = vec![LearningSensorDecisionKind::Reject; 7];

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn provider_failures_are_pending_and_open_instance_circuit_breaker() -> Result<()> {
        let input = fixture_sensor_input(false);
        let client = Arc::new(FixtureLearningSensorLlmClient::with_response(Err(
            LearningSensorLlmClientError::Timeout,
        )));
        let reviewer =
            LiveLearningSensorReviewer::new(client.clone(), LearningSensorReviewerIdentity::fake());

        let first = reviewer.review(input.clone()).await.decision;
        *client.response.lock().unwrap() = Some(Err(LearningSensorLlmClientError::ProviderInfra(
            "bad gateway".to_string(),
        )));
        let second = reviewer.review(input.clone()).await.decision;
        let open_after_two = reviewer.is_circuit_open();
        *client.response.lock().unwrap() = Some(Ok(LearningSensorLlmResponse::complete_json("{}")));
        let third = reviewer.review(input).await.decision;
        let calls = client.calls.load(Ordering::SeqCst);

        let actual = (first, second, open_after_two, third, calls);
        let expected = (
            LearningSensorDecisionKind::Pending,
            LearningSensorDecisionKind::Pending,
            true,
            LearningSensorDecisionKind::Pending,
            2usize,
        );

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn schema_failures_do_not_open_circuit_and_success_resets_infra_failures() -> Result<()> {
        let input = fixture_sensor_input(false);
        let client = Arc::new(FixtureLearningSensorLlmClient::with_response(Ok(
            LearningSensorLlmResponse::complete_json("{}"),
        )));
        let reviewer =
            LiveLearningSensorReviewer::new(client.clone(), LearningSensorReviewerIdentity::fake());

        let schema_failure = reviewer.review(input.clone()).await.decision;
        let open_after_schema = reviewer.is_circuit_open();
        *client.response.lock().unwrap() = Some(Err(LearningSensorLlmClientError::Timeout));
        let infra_failure = reviewer.review(input.clone()).await.decision;
        let output = fixture_output(&input, LearningSensorDecisionKind::Pending);
        *client.response.lock().unwrap() = Some(Ok(LearningSensorLlmResponse::complete_json(
            serde_json::to_string(&output)?,
        )));
        let success = reviewer.review(input).await.decision;
        let open_after_success = reviewer.is_circuit_open();

        let actual = (
            schema_failure,
            open_after_schema,
            infra_failure,
            success,
            open_after_success,
        );
        let expected = (
            LearningSensorDecisionKind::Reject,
            false,
            LearningSensorDecisionKind::Pending,
            LearningSensorDecisionKind::Pending,
            false,
        );

        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn adapter_depends_on_narrow_client_trait_not_agent_service() {
        fn accepts_client<T: LearningSensorLlmClient>() {}
        accepts_client::<FixtureLearningSensorLlmClient>();

        let actual =
            std::any::type_name::<LiveLearningSensorReviewer<FixtureLearningSensorLlmClient>>()
                .contains("AgentService");
        let expected = false;

        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn live_reviewer_allows_proposal_from_sanctioned_sanitized_observation() -> Result<()> {
        let input = fixture_sanitized_observation_input();
        let output = LearningSensorReviewOutput {
            schema_version: LEARNING_SENSOR_REVIEW_SCHEMA_VERSION,
            reviewer_id: LearningSensorReviewerIdentity::fake().reviewer_id,
            reviewer_version: LearningSensorReviewerIdentity::fake().reviewer_version,
            input_fingerprint: input.fingerprint()?,
            decision: LearningSensorDecisionKind::ProposeLesson,
            reason_code: "sanctioned_sanitized_chat_observation".to_string(),
            proposal_title: Some("sanctioned_sanitized_observation".to_string()),
            proposal_body: Some("validated_counters_and_fingerprints".to_string()),
        };
        let client = Arc::new(FixtureLearningSensorLlmClient::with_response(Ok(
            LearningSensorLlmResponse::complete_json(serde_json::to_string(&output)?),
        )));
        let reviewer =
            LiveLearningSensorReviewer::new(client, LearningSensorReviewerIdentity::fake());

        let actual = reviewer.review(input.clone()).await;
        let expected = LearningSensorDecisionKind::ProposeLesson;

        actual.validate_against_identity(&input, &LearningSensorReviewerIdentity::fake())?;
        assert_eq!(actual.decision, expected);
        Ok(())
    }

    #[test]
    fn system_instruction_keeps_metadata_pending_and_sanctioned_gate_explicit() -> Result<()> {
        let input = fixture_sanitized_observation_input();
        let request = LearningSensorLlmRequest::from_input(&input)?;
        let actual = (
            request
                .system_instruction
                .contains("conversation_metadata evidence may only be pending or reject"),
            request
                .system_instruction
                .contains("sanctioned_sanitized_chat_observation"),
            request
                .system_instruction
                .contains("typed_fixture_observation"),
        );
        let expected = (true, true, true);

        assert_eq!(actual, expected);
        Ok(())
    }

    fn fixture_sanitized_observation_input() -> LearningSensorReviewInput {
        LearningSensorReviewInput::from_sanitized_chat_observation(
            &fixture_learning_projection(),
            SanitizedChatLessonObservation::new(
                SanitizedChatObservationKind::ReviewerIdentifiedGap,
                SanitizedObservationCountBucket::Two,
                SanitizedObservationSeverity::Medium,
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            )
            .unwrap()
            .validate()
            .unwrap(),
        )
    }

    fn fixture_sensor_input(fixture_evidence: bool) -> LearningSensorReviewInput {
        let projection = fixture_learning_projection();
        if fixture_evidence {
            LearningSensorReviewInput::fake_fixture(
                &projection,
                "Durable typed observation",
                "A recurring typed fixture observation exists",
            )
        } else {
            LearningSensorReviewInput::from_candidate_projection(&projection)
        }
    }

    fn fixture_output(
        input: &LearningSensorReviewInput,
        decision: LearningSensorDecisionKind,
    ) -> LearningSensorReviewOutput {
        LearningSensorReviewOutput {
            schema_version: LEARNING_SENSOR_REVIEW_SCHEMA_VERSION,
            reviewer_id: LearningSensorReviewerIdentity::fake().reviewer_id,
            reviewer_version: LearningSensorReviewerIdentity::fake().reviewer_version,
            input_fingerprint: input.fingerprint().unwrap(),
            decision,
            reason_code: "fixture_reason".to_string(),
            proposal_title: None,
            proposal_body: None,
        }
    }

    fn fixture_learning_projection() -> LearningRecordProjection {
        let provenance = LearningProvenance::conversation(
            ConversationId::generate(),
            "source-event",
            "safe-source-fingerprint",
        );
        LearningRecordProjection {
            record_id: LearningRecordId::generate(),
            summary: "conversation_saved message_count=2 user_message_count=1 context_fingerprint=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            review_state: LearningReviewState::Candidate,
            redaction_status: LearningRedactionStatus::Clean,
            provenance,
            capture_metadata: Some(LearningCaptureMetadata::conversation_save(
                2,
                1,
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                learning_digest_hex("conversation_saved message_count=2 user_message_count=1 context_fingerprint=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            )),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            schema_version: forge_app::domain::LEARNING_LEDGER_SCHEMA_VERSION,
        }
    }
}
