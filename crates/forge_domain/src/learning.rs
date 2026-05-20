use std::str::FromStr;

use chrono::{DateTime, Utc};
use derive_more::derive::Display;
use derive_setters::Setters;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use strum_macros::{Display as StrumDisplay, EnumString};
use uuid::Uuid;

use crate::{ConversationId, Error, Result, SubagentTaskId};

/// Current learning ledger schema version.
pub const LEARNING_LEDGER_SCHEMA_VERSION: i32 = 2;

/// Current schema version for sanitized learning sensor review DTOs.
pub const LEARNING_SENSOR_REVIEW_SCHEMA_VERSION: i32 = 1;

/// Deterministic fake sensor reviewer identity used by the first self-learning slice.
pub const FAKE_LEARNING_SENSOR_REVIEWER_ID: &str = "fake_learning_sensor_reviewer";

/// Deterministic fake sensor reviewer version used by the first self-learning slice.
pub const FAKE_LEARNING_SENSOR_REVIEWER_VERSION: i32 = 1;

/// Current deterministic conversation-save capture version.
pub const CONVERSATION_SAVE_CAPTURE_VERSION: i32 = 1;

/// Deterministic reviewer identity for safe conversation-save auto-review.
pub const DETERMINISTIC_CONVERSATION_SAVE_AUTO_REVIEWER_V1: &str =
    "deterministic_conversation_save_auto_review_v1";

/// Machine-readable reason code used for deterministic conversation-save acceptance.
pub const DETERMINISTIC_CONVERSATION_SAVE_AUTO_ACCEPT_REASON: &str =
    "eligible_clean_current_conversation_save_capture";

/// Stable workspace-scoped identifier for a learning ledger record.
#[derive(Debug, Default, Display, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash)]
#[serde(transparent)]
pub struct LearningRecordId(Uuid);

impl LearningRecordId {
    /// Generates a new learning ledger record identifier.
    pub fn generate() -> Self {
        Self(Uuid::new_v4())
    }

    /// Converts this identifier into canonical text.
    pub fn into_string(&self) -> String {
        self.0.to_string()
    }

    /// Parses a learning ledger record identifier from text.
    ///
    /// # Arguments
    /// * `value` - Textual UUID representation.
    ///
    /// # Errors
    /// Returns an error when `value` is not a UUID.
    pub fn parse(value: impl ToString) -> Result<Self> {
        Ok(Self(
            Uuid::parse_str(&value.to_string()).map_err(Error::ConversationId)?,
        ))
    }
}

impl FromStr for LearningRecordId {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

/// Append-only event identifier for a learning ledger event.
#[derive(Debug, Default, Display, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash)]
#[serde(transparent)]
pub struct LearningEventId(Uuid);

impl LearningEventId {
    /// Generates a new learning ledger event identifier.
    pub fn generate() -> Self {
        Self(Uuid::new_v4())
    }

    /// Converts this identifier into canonical text.
    pub fn into_string(&self) -> String {
        self.0.to_string()
    }

    /// Parses a learning ledger event identifier from text.
    ///
    /// # Arguments
    /// * `value` - Textual UUID representation.
    ///
    /// # Errors
    /// Returns an error when `value` is not a UUID.
    pub fn parse(value: impl ToString) -> Result<Self> {
        Ok(Self(
            Uuid::parse_str(&value.to_string()).map_err(Error::ConversationId)?,
        ))
    }
}

impl FromStr for LearningEventId {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

/// Learning ledger source kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, StrumDisplay, EnumString)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum LearningSourceKind {
    /// Conversation-derived source.
    Conversation,
    /// Delegated task source.
    Task,
    /// Tool execution source.
    Tool,
    /// Evaluation or regression source.
    Eval,
}

/// Learning ledger event kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, StrumDisplay, EnumString)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum LearningEventKind {
    /// Candidate captured from redacted evidence.
    CandidateCaptured,
    /// Review accepted this candidate for future injection.
    ReviewAccepted,
    /// Review rejected this candidate.
    ReviewRejected,
    /// Non-injection sensor reviewer proposed a lesson candidate.
    SensorLessonProposed,
    /// Non-injection sensor reviewer could not decide because evidence is insufficient.
    SensorReviewPending,
    /// Non-injection sensor reviewer rejected the sanitized evidence.
    SensorReviewRejected,
    /// Record was superseded by a newer event.
    Superseded,
}

/// Review state projected from append-only events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, StrumDisplay, EnumString)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum LearningReviewState {
    /// Candidate captured but not reviewed.
    Candidate,
    /// Candidate rejected by review.
    Rejected,
    /// Candidate accepted for bounded injection.
    Accepted,
    /// Candidate superseded by later evidence.
    Superseded,
}

/// Review decision for a captured learning candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, StrumDisplay, EnumString)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum LearningReviewDecision {
    /// Accept the candidate for bounded future injection.
    Accept,
    /// Reject the candidate and keep it excluded from injection.
    Reject,
}

impl LearningReviewDecision {
    /// Returns the append-only ledger event kind for this review decision.
    pub fn event_kind(&self) -> LearningEventKind {
        match self {
            Self::Accept => LearningEventKind::ReviewAccepted,
            Self::Reject => LearningEventKind::ReviewRejected,
        }
    }

    /// Returns the projected review state produced by this decision.
    pub fn review_state(&self) -> LearningReviewState {
        match self {
            Self::Accept => LearningReviewState::Accepted,
            Self::Reject => LearningReviewState::Rejected,
        }
    }
}

/// Redaction status for persisted learning records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, StrumDisplay, EnumString)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum LearningRedactionStatus {
    /// Source required no redaction.
    Clean,
    /// Sensitive-looking data was redacted before persistence.
    Redacted,
}

/// Fingerprinted source provenance for a learning event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Setters)]
#[setters(into, strip_option)]
pub struct LearningProvenance {
    /// Kind of source that produced the evidence.
    pub source_kind: LearningSourceKind,
    /// Source conversation ID, when available.
    pub conversation_id: Option<ConversationId>,
    /// Source delegated task ID, when available.
    pub task_id: Option<SubagentTaskId>,
    /// Source tool name or identifier, when available.
    pub tool_name: Option<String>,
    /// Source eval case identifier, when available.
    pub eval_id: Option<String>,
    /// Source event identifier, when available.
    pub source_event_id: String,
    /// Redacted source fingerprint.
    pub source_fingerprint: String,
}

impl LearningProvenance {
    /// Builds provenance for conversation-derived evidence.
    ///
    /// # Arguments
    /// * `conversation_id` - Source conversation identifier.
    /// * `source_event_id` - Stable source event identifier.
    /// * `source_fingerprint` - Redacted source fingerprint.
    pub fn conversation(
        conversation_id: ConversationId,
        source_event_id: impl Into<String>,
        source_fingerprint: impl Into<String>,
    ) -> Self {
        Self {
            source_kind: LearningSourceKind::Conversation,
            conversation_id: Some(conversation_id),
            task_id: None,
            tool_name: None,
            eval_id: None,
            source_event_id: source_event_id.into(),
            source_fingerprint: source_fingerprint.into(),
        }
    }

    /// Builds provenance for deterministic evaluation or reviewer evidence.
    ///
    /// # Arguments
    /// * `eval_id` - Stable reviewer or evaluation identity.
    /// * `source_event_id` - Stable source event identifier.
    /// * `source_fingerprint` - Redacted source fingerprint.
    pub fn eval(
        eval_id: impl Into<String>,
        source_event_id: impl Into<String>,
        source_fingerprint: impl Into<String>,
    ) -> Self {
        Self {
            source_kind: LearningSourceKind::Eval,
            conversation_id: None,
            task_id: None,
            tool_name: None,
            eval_id: Some(eval_id.into()),
            source_event_id: source_event_id.into(),
            source_fingerprint: source_fingerprint.into(),
        }
    }

    /// Returns the stable source identifier for persistence and projections.
    ///
    /// # Errors
    /// Returns an error when the source kind lacks its required typed source ID.
    pub fn source_id(&self) -> anyhow::Result<String> {
        match self.source_kind {
            LearningSourceKind::Conversation => self
                .conversation_id
                .map(|id| id.into_string())
                .ok_or_else(|| anyhow::anyhow!("conversation provenance requires conversation_id")),
            LearningSourceKind::Task => self
                .task_id
                .map(|id| id.into_string())
                .ok_or_else(|| anyhow::anyhow!("task provenance requires task_id")),
            LearningSourceKind::Tool => self
                .tool_name
                .clone()
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| anyhow::anyhow!("tool provenance requires tool_name")),
            LearningSourceKind::Eval => self
                .eval_id
                .clone()
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| anyhow::anyhow!("eval provenance requires eval_id")),
        }
    }

    /// Validates source identity and fingerprint completeness.
    ///
    /// # Errors
    /// Returns an error when required fields are missing.
    pub fn validate(&self) -> anyhow::Result<()> {
        self.source_id()?;
        if self.source_event_id.trim().is_empty() {
            anyhow::bail!("learning provenance source_event_id is required");
        }
        if self.source_fingerprint.trim().is_empty() {
            anyhow::bail!("learning provenance source_fingerprint is required");
        }
        Ok(())
    }
}

/// Source path that generated a learning candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, StrumDisplay, EnumString)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum LearningCaptureSource {
    /// Candidate was generated by the conversation-save capture hook.
    ConversationSave,
}

/// Typed deterministic metadata for machine-generated learning capture.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Setters)]
#[setters(into)]
pub struct LearningCaptureMetadata {
    /// Source capture path that generated the candidate.
    pub source: LearningCaptureSource,
    /// Capture implementation version.
    pub capture_version: i32,
    /// Number of context messages in the saved conversation.
    pub message_count: i32,
    /// Number of external user messages in the saved conversation.
    pub user_message_count: i32,
    /// Fingerprint of the saved conversation context.
    pub context_fingerprint: String,
    /// Fingerprint of the deterministic summary rendering.
    pub summary_fingerprint: String,
}

impl LearningCaptureMetadata {
    /// Creates typed metadata for the deterministic conversation-save capture path.
    ///
    /// # Arguments
    /// * `message_count` - Number of context messages in the saved conversation.
    /// * `user_message_count` - Number of external user messages in the saved conversation.
    /// * `context_fingerprint` - Stable context fingerprint for the saved conversation.
    /// * `summary_fingerprint` - Fingerprint of the deterministic summary rendering.
    pub fn conversation_save(
        message_count: i32,
        user_message_count: i32,
        context_fingerprint: impl Into<String>,
        summary_fingerprint: impl Into<String>,
    ) -> Self {
        Self {
            source: LearningCaptureSource::ConversationSave,
            capture_version: CONVERSATION_SAVE_CAPTURE_VERSION,
            message_count,
            user_message_count,
            context_fingerprint: context_fingerprint.into(),
            summary_fingerprint: summary_fingerprint.into(),
        }
    }

    /// Validates metadata completeness and current version compatibility.
    ///
    /// # Errors
    /// Returns an error when metadata is incomplete or version-incompatible.
    pub fn validate_current(&self) -> anyhow::Result<()> {
        if self.source != LearningCaptureSource::ConversationSave {
            anyhow::bail!("unsupported learning capture source {}", self.source);
        }
        if self.capture_version != CONVERSATION_SAVE_CAPTURE_VERSION {
            anyhow::bail!(
                "unsupported learning capture version {}",
                self.capture_version
            );
        }
        if self.message_count <= 0 {
            anyhow::bail!("learning capture message_count must be positive");
        }
        if self.user_message_count < 0 {
            anyhow::bail!("learning capture user_message_count cannot be negative");
        }
        if self.context_fingerprint.trim().is_empty() {
            anyhow::bail!("learning capture context_fingerprint is required");
        }
        if self.summary_fingerprint.trim().is_empty() {
            anyhow::bail!("learning capture summary_fingerprint is required");
        }
        Ok(())
    }
}

/// Sanitized evidence class exposed to pure learning Sensor reviewers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, StrumDisplay, EnumString)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum LearningSensorEvidenceKind {
    /// Metadata-only conversation-save candidate.
    ConversationMetadata,
    /// Runtime sanitized chat observation with closed evidence fields only.
    SanctionedSanitizedChatObservation,
    /// Typed fixture evidence available only from regression tests or explicit fixture paths.
    TypedFixtureObservation,
}

/// Provenance marker for sanitized Sensor inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, StrumDisplay, EnumString)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum LearningSensorProvenanceMarker {
    /// Normal runtime metadata-only capture path.
    RuntimeConversationSaved,
    /// Runtime sanctioned sanitized chat observation path.
    RuntimeSanitizedChatObservation,
    /// Explicit fake-reviewer fixture marker unavailable from runtime conversation-save metadata.
    FakeReviewerFixture,
}

/// Project-local opaque digest used for non-reversible sanitized chat evidence.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct OpaqueLearningFingerprint(String);

impl OpaqueLearningFingerprint {
    /// Creates a validated opaque lowercase SHA-256-style fingerprint.
    ///
    /// # Arguments
    /// * `value` - Fixed-length lowercase hex digest.
    ///
    /// # Errors
    /// Returns an error when the value is empty, not 64 characters, or not lowercase hex.
    pub fn new(value: impl Into<String>) -> anyhow::Result<Self> {
        let value = value.into();
        if value.len() != 64 {
            anyhow::bail!(
                "opaque learning fingerprint must be exactly 64 lowercase hex characters"
            );
        }
        if !value
            .chars()
            .all(|ch| ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase())
        {
            anyhow::bail!("opaque learning fingerprint must contain only lowercase hex characters");
        }
        Ok(Self(value))
    }

    /// Returns the validated digest text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for OpaqueLearningFingerprint {
    type Error = anyhow::Error;

    fn try_from(value: String) -> anyhow::Result<Self> {
        Self::new(value)
    }
}

impl From<OpaqueLearningFingerprint> for String {
    fn from(value: OpaqueLearningFingerprint) -> Self {
        value.0
    }
}

/// Closed runtime observation class for sanitized chat evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, StrumDisplay, EnumString)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum SanitizedChatObservationKind {
    /// Repeated agent behavior created a candidate self-learning opportunity.
    RepeatedAgentBehavior,
    /// A verifier or reviewer found a recurring same-path quality issue.
    ReviewerIdentifiedGap,
    /// A safety invariant near miss was observed through sanitized counters.
    SafetyInvariantNearMiss,
}

/// Closed count bucket for sanitized chat evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, StrumDisplay, EnumString)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum SanitizedObservationCountBucket {
    /// One observed instance.
    One,
    /// Two observed instances.
    Two,
    /// Three to five observed instances.
    ThreeToFive,
    /// More than five observed instances.
    MoreThanFive,
}

/// Closed severity class for sanitized chat evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, StrumDisplay, EnumString)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum SanitizedObservationSeverity {
    /// Low-risk improvement opportunity.
    Low,
    /// Structural or repeated quality risk.
    Medium,
    /// Safety-relevant or architecture-relevant risk.
    High,
}

/// Runtime sanitized chat observation DTO containing only closed enums, counters, and opaque digests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SanitizedChatLessonObservation {
    /// Observation class.
    pub observation_kind: SanitizedChatObservationKind,
    /// Count bucket for repeated evidence.
    pub occurrence_bucket: SanitizedObservationCountBucket,
    /// Sanitized impact severity.
    pub severity: SanitizedObservationSeverity,
    /// Project-local non-reversible fingerprint of the observation source cluster.
    pub source_cluster_fingerprint: OpaqueLearningFingerprint,
    /// Project-local non-reversible fingerprint of the behavioral pattern.
    pub behavior_pattern_fingerprint: OpaqueLearningFingerprint,
}

impl SanitizedChatLessonObservation {
    /// Creates a validated sanitized chat observation DTO.
    ///
    /// # Arguments
    /// * `observation_kind` - Closed observation class.
    /// * `occurrence_bucket` - Closed occurrence bucket.
    /// * `severity` - Closed severity class.
    /// * `source_cluster_fingerprint` - Non-reversible source-cluster digest.
    /// * `behavior_pattern_fingerprint` - Non-reversible behavior-pattern digest.
    ///
    /// # Errors
    /// Returns an error when any opaque fingerprint is malformed.
    pub fn new(
        observation_kind: SanitizedChatObservationKind,
        occurrence_bucket: SanitizedObservationCountBucket,
        severity: SanitizedObservationSeverity,
        source_cluster_fingerprint: impl Into<String>,
        behavior_pattern_fingerprint: impl Into<String>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            observation_kind,
            occurrence_bucket,
            severity,
            source_cluster_fingerprint: OpaqueLearningFingerprint::new(source_cluster_fingerprint)?,
            behavior_pattern_fingerprint: OpaqueLearningFingerprint::new(
                behavior_pattern_fingerprint,
            )?,
        })
    }

    /// Validates this DTO and returns a typestate wrapper for proposal-capable use.
    ///
    /// # Errors
    /// Returns an error when any field is not sanctioned sanitized evidence.
    pub fn validate(self) -> anyhow::Result<ValidatedSanitizedChatLessonObservation> {
        Ok(ValidatedSanitizedChatLessonObservation(self))
    }

    /// Returns a stable non-reversible observation fingerprint.
    pub fn fingerprint(&self) -> String {
        learning_digest_hex(format!(
            "{}:{}:{}:{}:{}",
            self.observation_kind,
            self.occurrence_bucket,
            self.severity,
            self.source_cluster_fingerprint.as_str(),
            self.behavior_pattern_fingerprint.as_str()
        ))
    }
}

/// Typestate wrapper proving a sanitized chat observation passed validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedSanitizedChatLessonObservation(SanitizedChatLessonObservation);

impl ValidatedSanitizedChatLessonObservation {
    /// Returns the validated sanitized observation DTO.
    pub fn observation(&self) -> &SanitizedChatLessonObservation {
        &self.0
    }
}

/// Separate untrusted Sensor decision enum; it cannot encode Accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, StrumDisplay, EnumString)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum LearningSensorDecisionKind {
    /// Sensor proposes a lesson for audit only; it is not accepted or injected.
    ProposeLesson,
    /// Sensor needs more substantive evidence.
    Pending,
    /// Sensor rejected the sanitized evidence.
    Reject,
}

impl LearningSensorDecisionKind {
    /// Returns the non-injection append-only event kind for this Sensor decision.
    pub fn event_kind(&self) -> LearningEventKind {
        match self {
            Self::ProposeLesson => LearningEventKind::SensorLessonProposed,
            Self::Pending => LearningEventKind::SensorReviewPending,
            Self::Reject => LearningEventKind::SensorReviewRejected,
        }
    }
}

/// Sanitized bounded learning Sensor review input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Setters)]
#[serde(deny_unknown_fields)]
#[setters(into, strip_option)]
pub struct LearningSensorReviewInput {
    /// Sensor DTO schema version.
    pub schema_version: i32,
    /// Candidate record under review.
    pub candidate_id: LearningRecordId,
    /// Hash of the current candidate projection at input construction time.
    pub sanitized_projection_hash: String,
    /// Sanitized candidate summary.
    pub sanitized_summary: String,
    /// Sanitized source fingerprint.
    pub sanitized_source_fingerprint: String,
    /// Sanitized evidence kind.
    pub evidence_kind: LearningSensorEvidenceKind,
    /// Sanitized provenance marker.
    pub provenance_marker: LearningSensorProvenanceMarker,
    /// Optional sanitized fixture title.
    pub fixture_title: Option<String>,
    /// Optional sanitized fixture observation body.
    pub fixture_observation: Option<String>,
    /// Optional runtime sanitized chat observation payload.
    pub sanitized_chat_observation: Option<SanitizedChatLessonObservation>,
}

impl LearningSensorReviewInput {
    /// Builds metadata-only Sensor input from a projected candidate.
    ///
    /// # Arguments
    /// * `projection` - Current candidate projection.
    pub fn from_candidate_projection(projection: &LearningRecordProjection) -> Self {
        Self {
            schema_version: LEARNING_SENSOR_REVIEW_SCHEMA_VERSION,
            candidate_id: projection.record_id,
            sanitized_projection_hash: learning_projection_hash(projection),
            sanitized_summary: projection.summary.clone(),
            sanitized_source_fingerprint: projection.provenance.source_fingerprint.clone(),
            evidence_kind: LearningSensorEvidenceKind::ConversationMetadata,
            provenance_marker: LearningSensorProvenanceMarker::RuntimeConversationSaved,
            fixture_title: None,
            fixture_observation: None,
            sanitized_chat_observation: None,
        }
    }

    /// Builds runtime sanctioned sanitized chat observation input from a projected candidate.
    ///
    /// # Arguments
    /// * `projection` - Current candidate projection.
    /// * `observation` - Validated sanitized chat observation typestate.
    pub fn from_sanitized_chat_observation(
        projection: &LearningRecordProjection,
        observation: ValidatedSanitizedChatLessonObservation,
    ) -> Self {
        let observation = observation.observation().clone();
        Self {
            schema_version: LEARNING_SENSOR_REVIEW_SCHEMA_VERSION,
            candidate_id: projection.record_id,
            sanitized_projection_hash: learning_projection_hash(projection),
            sanitized_summary: projection.summary.clone(),
            sanitized_source_fingerprint: observation.fingerprint(),
            evidence_kind: LearningSensorEvidenceKind::SanctionedSanitizedChatObservation,
            provenance_marker: LearningSensorProvenanceMarker::RuntimeSanitizedChatObservation,
            fixture_title: None,
            fixture_observation: None,
            sanitized_chat_observation: Some(observation),
        }
    }

    /// Builds explicit fake Sensor fixture input from a projected candidate.
    ///
    /// # Arguments
    /// * `projection` - Current candidate projection.
    /// * `title` - Sanitized fixture title.
    /// * `observation` - Sanitized fixture observation.
    pub fn fake_fixture(
        projection: &LearningRecordProjection,
        title: impl Into<String>,
        observation: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: LEARNING_SENSOR_REVIEW_SCHEMA_VERSION,
            candidate_id: projection.record_id,
            sanitized_projection_hash: learning_projection_hash(projection),
            sanitized_summary: projection.summary.clone(),
            sanitized_source_fingerprint: projection.provenance.source_fingerprint.clone(),
            evidence_kind: LearningSensorEvidenceKind::TypedFixtureObservation,
            provenance_marker: LearningSensorProvenanceMarker::FakeReviewerFixture,
            fixture_title: Some(title.into()),
            fixture_observation: Some(observation.into()),
            sanitized_chat_observation: None,
        }
    }

    /// Returns a deterministic fingerprint of the sanitized Sensor input.
    ///
    /// # Errors
    /// Returns an error when serialization fails.
    pub fn fingerprint(&self) -> anyhow::Result<String> {
        Ok(learning_digest_hex(serde_json::to_vec(self)?))
    }

    /// Validates that the input is bounded and contains no action/control fields.
    ///
    /// # Errors
    /// Returns an error when the input is invalid or unsafe for Sensor review.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.schema_version != LEARNING_SENSOR_REVIEW_SCHEMA_VERSION {
            anyhow::bail!("learning sensor input schema version mismatch");
        }
        ensure_learning_sensor_text("sanitized_summary", &self.sanitized_summary, 512)?;
        ensure_learning_sensor_text(
            "sanitized_source_fingerprint",
            &self.sanitized_source_fingerprint,
            128,
        )?;
        ensure_learning_sensor_text(
            "sanitized_projection_hash",
            &self.sanitized_projection_hash,
            128,
        )?;
        if let Some(title) = &self.fixture_title {
            ensure_learning_sensor_text("fixture_title", title, 160)?;
        }
        if let Some(observation) = &self.fixture_observation {
            ensure_learning_sensor_text("fixture_observation", observation, 1_024)?;
        }
        if self.evidence_kind == LearningSensorEvidenceKind::ConversationMetadata {
            if self.provenance_marker != LearningSensorProvenanceMarker::RuntimeConversationSaved {
                anyhow::bail!("conversation metadata input requires runtime provenance marker");
            }
            if self.fixture_title.is_some()
                || self.fixture_observation.is_some()
                || self.sanitized_chat_observation.is_some()
            {
                anyhow::bail!("conversation metadata input cannot include observation payload");
            }
        }
        if self.evidence_kind == LearningSensorEvidenceKind::SanctionedSanitizedChatObservation {
            if self.provenance_marker
                != LearningSensorProvenanceMarker::RuntimeSanitizedChatObservation
            {
                anyhow::bail!(
                    "sanitized chat observation requires runtime sanitized observation marker"
                );
            }
            if self.fixture_title.is_some() || self.fixture_observation.is_some() {
                anyhow::bail!("sanitized chat observation cannot include fixture payload");
            }
            let observation = self.sanitized_chat_observation.clone().ok_or_else(|| {
                anyhow::anyhow!("sanitized chat observation input requires observation payload")
            })?;
            observation.validate()?;
        }
        if self.evidence_kind == LearningSensorEvidenceKind::TypedFixtureObservation {
            if self.provenance_marker != LearningSensorProvenanceMarker::FakeReviewerFixture {
                anyhow::bail!("fixture observation requires fake reviewer fixture marker");
            }
            if self.sanitized_chat_observation.is_some() {
                anyhow::bail!("fixture observation cannot include sanitized chat payload");
            }
            if self
                .fixture_title
                .as_deref()
                .unwrap_or_default()
                .trim()
                .is_empty()
                || self
                    .fixture_observation
                    .as_deref()
                    .unwrap_or_default()
                    .trim()
                    .is_empty()
            {
                anyhow::bail!("fixture observation input requires title and observation");
            }
        }
        Ok(())
    }

    fn is_proposal_capable_validated_evidence(&self) -> bool {
        match (self.evidence_kind, self.provenance_marker) {
            (
                LearningSensorEvidenceKind::SanctionedSanitizedChatObservation,
                LearningSensorProvenanceMarker::RuntimeSanitizedChatObservation,
            ) => self
                .sanitized_chat_observation
                .clone()
                .is_some_and(|observation| observation.validate().is_ok()),
            (
                LearningSensorEvidenceKind::TypedFixtureObservation,
                LearningSensorProvenanceMarker::FakeReviewerFixture,
            ) => self.fixture_title.is_some() && self.fixture_observation.is_some(),
            _ => false,
        }
    }
}

/// Reviewer identity accepted by the bounded learning Sensor validation path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Setters)]
#[serde(deny_unknown_fields)]
#[setters(into)]
pub struct LearningSensorReviewerIdentity {
    /// Stable reviewer identity.
    pub reviewer_id: String,
    /// Reviewer implementation version.
    pub reviewer_version: i32,
}

impl LearningSensorReviewerIdentity {
    /// Creates a learning Sensor reviewer identity.
    ///
    /// # Arguments
    /// * `reviewer_id` - Stable reviewer identity.
    /// * `reviewer_version` - Reviewer implementation version.
    pub fn new(reviewer_id: impl Into<String>, reviewer_version: i32) -> Self {
        Self { reviewer_id: reviewer_id.into(), reviewer_version }
    }

    /// Returns the deterministic fake reviewer identity.
    pub fn fake() -> Self {
        Self::new(
            FAKE_LEARNING_SENSOR_REVIEWER_ID,
            FAKE_LEARNING_SENSOR_REVIEWER_VERSION,
        )
    }

    /// Validates identity fields for Sensor event construction.
    ///
    /// # Errors
    /// Returns an error when the identity is empty, oversized, or version-invalid.
    pub fn validate(&self) -> anyhow::Result<()> {
        ensure_learning_sensor_text("reviewer_id", &self.reviewer_id, 128)?;
        if self.reviewer_version <= 0 {
            anyhow::bail!("learning sensor reviewer_version must be positive");
        }
        Ok(())
    }
}

/// Untrusted output from the pure learning Sensor reviewer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Setters)]
#[serde(deny_unknown_fields)]
#[setters(into, strip_option)]
pub struct LearningSensorReviewOutput {
    /// Sensor DTO schema version.
    pub schema_version: i32,
    /// Reviewer identity.
    pub reviewer_id: String,
    /// Reviewer implementation version.
    pub reviewer_version: i32,
    /// Fingerprint of the sanitized input reviewed by the Sensor.
    pub input_fingerprint: String,
    /// Untrusted non-accepting Sensor decision.
    pub decision: LearningSensorDecisionKind,
    /// Sanitized reason code.
    pub reason_code: String,
    /// Optional sanitized proposal title.
    pub proposal_title: Option<String>,
    /// Optional sanitized proposal body.
    pub proposal_body: Option<String>,
}

impl LearningSensorReviewOutput {
    /// Returns a deterministic fingerprint of the normalized proposal or reason payload.
    pub fn normalized_payload_fingerprint(&self) -> String {
        let normalized = format!(
            "{}:{}:{}:{}",
            self.decision,
            self.reason_code.trim().to_ascii_lowercase(),
            self.proposal_title
                .as_deref()
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase(),
            self.proposal_body
                .as_deref()
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
        );
        learning_digest_hex(normalized)
    }

    /// Validates untrusted Sensor output against a previously built sanitized input.
    ///
    /// # Arguments
    /// * `input` - Sanitized input that was sent to the Sensor.
    /// * `reviewer_id` - Expected reviewer identity.
    /// * `reviewer_version` - Expected reviewer version.
    ///
    /// # Errors
    /// Returns an error when the output is mismatched or unsafe to append.
    pub fn validate_against(
        &self,
        input: &LearningSensorReviewInput,
        reviewer_id: &str,
        reviewer_version: i32,
    ) -> anyhow::Result<()> {
        self.validate_against_identity(
            input,
            &LearningSensorReviewerIdentity::new(reviewer_id, reviewer_version),
        )
    }

    /// Validates untrusted Sensor output against a typed reviewer identity and input.
    ///
    /// # Arguments
    /// * `input` - Sanitized input that was sent to the Sensor.
    /// * `identity` - Expected reviewer identity.
    ///
    /// # Errors
    /// Returns an error when the output is mismatched or unsafe to append.
    pub fn validate_against_identity(
        &self,
        input: &LearningSensorReviewInput,
        identity: &LearningSensorReviewerIdentity,
    ) -> anyhow::Result<()> {
        input.validate()?;
        identity.validate()?;
        if self.schema_version != LEARNING_SENSOR_REVIEW_SCHEMA_VERSION {
            anyhow::bail!("learning sensor output schema version mismatch");
        }
        if self.reviewer_id != identity.reviewer_id
            || self.reviewer_version != identity.reviewer_version
        {
            anyhow::bail!("learning sensor reviewer identity mismatch");
        }
        if self.input_fingerprint != input.fingerprint()? {
            anyhow::bail!("learning sensor input fingerprint mismatch");
        }
        ensure_learning_sensor_code("reason_code", &self.reason_code, 128)?;
        if let Some(title) = &self.proposal_title {
            ensure_learning_sensor_proposal_code("proposal_title", title, 160)?;
        }
        if let Some(body) = &self.proposal_body {
            ensure_learning_sensor_proposal_code("proposal_body", body, 1_024)?;
        }
        if self.decision != LearningSensorDecisionKind::ProposeLesson
            && (self.proposal_title.is_some() || self.proposal_body.is_some())
        {
            anyhow::bail!("learning sensor non-proposal decision cannot include proposal payload");
        }
        if self.decision == LearningSensorDecisionKind::ProposeLesson {
            if !input.is_proposal_capable_validated_evidence() {
                anyhow::bail!(
                    "learning sensor proposal requires validated sanitized observation or typed fake fixture evidence"
                );
            }
            if self
                .proposal_title
                .as_deref()
                .unwrap_or_default()
                .trim()
                .is_empty()
                || self
                    .proposal_body
                    .as_deref()
                    .unwrap_or_default()
                    .trim()
                    .is_empty()
            {
                anyhow::bail!("learning sensor proposal requires title and body");
            }
            ensure_not_forbidden_learning_sensor_scope(self)?;
        }
        Ok(())
    }

    /// Builds the append-only non-injection ledger event for this output.
    ///
    /// # Arguments
    /// * `input` - Sanitized input reviewed by the Sensor.
    /// * `created_at` - Event timestamp.
    ///
    /// # Errors
    /// Returns an error when validation fails.
    pub fn into_sensor_event(
        &self,
        input: &LearningSensorReviewInput,
        created_at: DateTime<Utc>,
    ) -> anyhow::Result<LearningLedgerEvent> {
        self.into_sensor_event_with_identity(
            input,
            &LearningSensorReviewerIdentity::fake(),
            created_at,
        )
    }

    /// Builds the append-only non-injection ledger event using a typed reviewer identity.
    ///
    /// # Arguments
    /// * `input` - Sanitized input reviewed by the Sensor.
    /// * `identity` - Expected reviewer identity.
    /// * `created_at` - Event timestamp.
    ///
    /// # Errors
    /// Returns an error when validation fails.
    pub fn into_sensor_event_with_identity(
        &self,
        input: &LearningSensorReviewInput,
        identity: &LearningSensorReviewerIdentity,
        created_at: DateTime<Utc>,
    ) -> anyhow::Result<LearningLedgerEvent> {
        self.validate_against_identity(input, identity)?;
        let input_fingerprint = input.fingerprint()?;
        let payload_fingerprint = self.normalized_payload_fingerprint();
        let event_kind = self.decision.event_kind();
        let summary = match self.decision {
            LearningSensorDecisionKind::ProposeLesson => format!(
                "sensor_proposal reviewer={} version={} reason={} title={} body={}",
                self.reviewer_id,
                self.reviewer_version,
                self.reason_code,
                self.proposal_title.as_deref().unwrap_or_default(),
                self.proposal_body.as_deref().unwrap_or_default()
            ),
            LearningSensorDecisionKind::Pending => format!(
                "sensor_pending reviewer={} version={} reason={}",
                self.reviewer_id, self.reviewer_version, self.reason_code
            ),
            LearningSensorDecisionKind::Reject => format!(
                "sensor_reject reviewer={} version={} reason={}",
                self.reviewer_id, self.reviewer_version, self.reason_code
            ),
        };
        let source_event_id = format!(
            "sensor:{}:candidate:{}:input:{}:decision:{}:payload:{}",
            self.reviewer_id,
            input.candidate_id.into_string(),
            input_fingerprint,
            self.decision,
            payload_fingerprint
        );
        let source_fingerprint = learning_digest_hex(format!(
            "schema:{}:candidate:{}:input:{}:reviewer:{}:{}:decision:{}:payload:{}",
            LEARNING_SENSOR_REVIEW_SCHEMA_VERSION,
            input.candidate_id.into_string(),
            input_fingerprint,
            self.reviewer_id,
            self.reviewer_version,
            self.decision,
            payload_fingerprint
        ));
        let mut event = LearningLedgerEvent::review(
            input.candidate_id,
            event_kind,
            summary,
            LearningProvenance::eval(&self.reviewer_id, source_event_id, source_fingerprint),
            created_at,
        )?;
        event.idempotency_key = learning_digest_hex(format!(
            "sensor-event:schema={}:candidate={}:input={}:reviewer={}:version={}:decision={}:payload={}",
            LEARNING_SENSOR_REVIEW_SCHEMA_VERSION,
            input.candidate_id.into_string(),
            input_fingerprint,
            self.reviewer_id,
            self.reviewer_version,
            self.decision,
            payload_fingerprint
        ));
        Ok(event)
    }
}

/// Pure side-effect-free Sensor reviewer interface.
pub trait LearningSensorReviewer {
    /// Reviews sanitized typed evidence and returns only an untrusted Sensor output.
    ///
    /// # Arguments
    /// * `input` - Sanitized evidence projection.
    ///
    /// # Errors
    /// Returns an error when the reviewer cannot produce a typed output.
    fn review(
        &self,
        input: LearningSensorReviewInput,
    ) -> anyhow::Result<LearningSensorReviewOutput>;
}

/// Deterministic fake Sensor reviewer used before any live LLM adapter exists.
#[derive(Debug, Clone, Copy, Default)]
pub struct FakeLearningSensorReviewer;

impl LearningSensorReviewer for FakeLearningSensorReviewer {
    fn review(
        &self,
        input: LearningSensorReviewInput,
    ) -> anyhow::Result<LearningSensorReviewOutput> {
        input.validate()?;
        let input_fingerprint = input.fingerprint()?;
        let (decision, reason_code, proposal_title, proposal_body) = match (
            input.evidence_kind,
            input.provenance_marker,
            input.fixture_title.clone(),
            input.fixture_observation.clone(),
        ) {
            (
                LearningSensorEvidenceKind::TypedFixtureObservation,
                LearningSensorProvenanceMarker::FakeReviewerFixture,
                Some(_),
                Some(_),
            ) => (
                LearningSensorDecisionKind::ProposeLesson,
                "typed_fixture_substantive_evidence".to_string(),
                Some("typed_fixture_observation".to_string()),
                Some("typed_fixture_substantive_pattern".to_string()),
            ),
            (
                LearningSensorEvidenceKind::SanctionedSanitizedChatObservation,
                LearningSensorProvenanceMarker::RuntimeSanitizedChatObservation,
                _,
                _,
            ) => (
                LearningSensorDecisionKind::ProposeLesson,
                "sanctioned_sanitized_chat_observation".to_string(),
                Some("sanctioned_sanitized_observation".to_string()),
                Some("validated_counters_and_fingerprints".to_string()),
            ),
            _ => (
                LearningSensorDecisionKind::Pending,
                "insufficient_substantive_evidence".to_string(),
                None,
                None,
            ),
        };
        Ok(LearningSensorReviewOutput {
            schema_version: LEARNING_SENSOR_REVIEW_SCHEMA_VERSION,
            reviewer_id: FAKE_LEARNING_SENSOR_REVIEWER_ID.to_string(),
            reviewer_version: FAKE_LEARNING_SENSOR_REVIEWER_VERSION,
            input_fingerprint,
            decision,
            reason_code,
            proposal_title,
            proposal_body,
        })
    }
}

/// Hashes the sanitized candidate projection used by Sensor stale-state checks.
///
/// # Arguments
/// * `projection` - Projection to fingerprint.
pub fn learning_projection_hash(projection: &LearningRecordProjection) -> String {
    learning_digest_hex(format!(
        "{}:{}:{}:{}:{}:{}",
        projection.record_id.into_string(),
        projection.summary,
        projection.review_state,
        projection.redaction_status,
        projection.provenance.source_fingerprint,
        projection.schema_version
    ))
}

fn ensure_learning_sensor_proposal_code(
    name: &str,
    value: &str,
    max_chars: usize,
) -> anyhow::Result<()> {
    ensure_learning_sensor_code(name, value, max_chars)?;
    const ALLOWED_PROPOSAL_CODES: &[&str] = &[
        "typed_fixture_observation",
        "typed_fixture_substantive_pattern",
        "sanctioned_sanitized_observation",
        "validated_counters_and_fingerprints",
    ];
    if !ALLOWED_PROPOSAL_CODES.contains(&value.trim()) {
        anyhow::bail!("learning sensor {name} is not an allowed audit-only proposal code");
    }
    Ok(())
}

fn ensure_learning_sensor_code(name: &str, value: &str, max_chars: usize) -> anyhow::Result<()> {
    let value = value.trim();
    if value.is_empty() {
        anyhow::bail!("learning sensor {name} cannot be empty");
    }
    if value.chars().count() > max_chars {
        anyhow::bail!("learning sensor {name} exceeds max length {max_chars}");
    }
    if !value
        .chars()
        .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
    {
        anyhow::bail!("learning sensor {name} must be a bounded safe reason code");
    }
    ensure_learning_sensor_text(name, value, max_chars)
}

fn ensure_learning_sensor_text(name: &str, value: &str, max_chars: usize) -> anyhow::Result<()> {
    let value = value.trim();
    if value.is_empty() {
        anyhow::bail!("learning sensor {name} cannot be empty");
    }
    if value.chars().count() > max_chars {
        anyhow::bail!("learning sensor {name} exceeds max length {max_chars}");
    }
    let lower = value.to_ascii_lowercase();
    let forbidden = [
        "token",
        "password",
        "bearer",
        "secret",
        "api_key",
        "apikey",
        "tool_call",
        "tool payload",
        "raw_transcript",
        "patch",
        "diff --git",
        "action",
        "mutation_target",
        "file_path",
        "../",
        "/home/",
        "/etc/",
        "http://",
        "https://",
        "www.",
        "rules",
        "skills",
        "system prompt",
        "agent definition",
        "source code",
        "provider config",
        "tool access",
        "public publish",
        "credential",
    ];
    if forbidden.iter().any(|needle| lower.contains(needle)) {
        anyhow::bail!("learning sensor {name} contains forbidden control or secret-shaped text");
    }
    if lower.contains('@')
        || lower.contains("cargo ")
        || lower.contains("git ")
        || lower.contains("curl ")
        || lower.contains("rm ")
        || lower.contains("sudo ")
        || lower.contains("ssh ")
    {
        anyhow::bail!("learning sensor {name} contains raw-looking payload text");
    }
    Ok(())
}

fn ensure_not_forbidden_learning_sensor_scope(
    output: &LearningSensorReviewOutput,
) -> anyhow::Result<()> {
    if output
        .proposal_title
        .iter()
        .chain(output.proposal_body.iter())
        .any(|value| {
            let lower = value.to_ascii_lowercase();
            [
                "rules",
                "skills",
                "system prompt",
                "agent definition",
                "source code",
                "provider config",
                "tool access",
                "public publish",
                "credential",
            ]
            .iter()
            .any(|needle| lower.contains(needle))
        })
    {
        anyhow::bail!("learning sensor proposal targets a forbidden self-mutation scope");
    }
    Ok(())
}

/// Append-only learning ledger event persisted by repository implementations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Setters)]
#[setters(into)]
pub struct LearningLedgerEvent {
    /// Durable event identifier.
    pub event_id: LearningEventId,
    /// Stable record identifier receiving this event.
    pub record_id: LearningRecordId,
    /// Stable idempotency key for retry-safe insertion.
    pub idempotency_key: String,
    /// Event kind.
    pub event_kind: LearningEventKind,
    /// Redacted summary or review note.
    pub summary: String,
    /// Redacted source fingerprint.
    pub content_fingerprint: String,
    /// Redaction status of the persisted summary.
    pub redaction_status: LearningRedactionStatus,
    /// Complete typed provenance.
    pub provenance: LearningProvenance,
    /// Optional typed capture metadata for machine-generated candidates.
    pub capture_metadata: Option<LearningCaptureMetadata>,
    /// Event timestamp.
    pub created_at: DateTime<Utc>,
    /// Schema version for forward migrations.
    pub schema_version: i32,
}

impl LearningLedgerEvent {
    /// Builds a retry-safe candidate capture event from redacted input.
    ///
    /// # Arguments
    /// * `summary` - Candidate summary. Sensitive-looking fragments are
    ///   redacted before persistence.
    /// * `provenance` - Typed source provenance.
    /// * `created_at` - Capture timestamp.
    ///
    /// # Errors
    /// Returns an error when provenance is incomplete.
    pub fn capture_candidate(
        summary: impl Into<String>,
        provenance: LearningProvenance,
        created_at: DateTime<Utc>,
    ) -> anyhow::Result<Self> {
        provenance.validate()?;
        let redacted = RedactedLearningSummary::from_raw(summary.into());
        let source_id = provenance.source_id()?;
        let idempotency_key = stable_learning_key(
            LearningEventKind::CandidateCaptured,
            provenance.source_kind,
            &source_id,
            &provenance.source_event_id,
            &redacted.fingerprint,
        );
        Ok(Self {
            event_id: LearningEventId::generate(),
            record_id: LearningRecordId::generate(),
            idempotency_key,
            event_kind: LearningEventKind::CandidateCaptured,
            summary: redacted.summary,
            content_fingerprint: redacted.fingerprint,
            redaction_status: redacted.status,
            provenance,
            capture_metadata: None,
            created_at,
            schema_version: LEARNING_LEDGER_SCHEMA_VERSION,
        })
    }

    /// Builds a review event for an existing learning record.
    ///
    /// # Arguments
    /// * `record_id` - Record receiving this review event.
    /// * `event_kind` - Review event kind.
    /// * `summary` - Redacted review note.
    /// * `provenance` - Typed source provenance.
    /// * `created_at` - Review timestamp.
    ///
    /// # Errors
    /// Returns an error when provenance is incomplete or event kind is not a
    /// review/projection event.
    pub fn review(
        record_id: LearningRecordId,
        event_kind: LearningEventKind,
        summary: impl Into<String>,
        provenance: LearningProvenance,
        created_at: DateTime<Utc>,
    ) -> anyhow::Result<Self> {
        if event_kind == LearningEventKind::CandidateCaptured {
            anyhow::bail!("review event kind cannot be candidate capture");
        }
        provenance.validate()?;
        let redacted = RedactedLearningSummary::from_raw(summary.into());
        let source_id = provenance.source_id()?;
        let idempotency_key = stable_review_key(
            record_id,
            event_kind,
            provenance.source_kind,
            &source_id,
            &provenance.source_event_id,
            &redacted.fingerprint,
        );
        Ok(Self {
            event_id: LearningEventId::generate(),
            record_id,
            idempotency_key,
            event_kind,
            summary: redacted.summary,
            content_fingerprint: redacted.fingerprint,
            redaction_status: redacted.status,
            provenance,
            capture_metadata: None,
            created_at,
            schema_version: LEARNING_LEDGER_SCHEMA_VERSION,
        })
    }
}

/// Freshness of an idempotent learning ledger append.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LearningLedgerEventFreshness {
    /// Event was inserted by the current persistence call.
    Inserted,
    /// Event already existed and was returned as an idempotency replay.
    Existing,
}

/// Typed outcome of an idempotent learning ledger append.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LearningLedgerAppendOutcome {
    /// Event inserted or replayed by the ledger.
    pub event: LearningLedgerEvent,
    /// Whether this call inserted the event or returned an existing one.
    pub freshness: LearningLedgerEventFreshness,
}

impl LearningLedgerAppendOutcome {
    /// Creates an append outcome for an event inserted by the current call.
    ///
    /// # Arguments
    /// * `event` - Event inserted by the current persistence call.
    pub fn inserted(event: LearningLedgerEvent) -> Self {
        Self { event, freshness: LearningLedgerEventFreshness::Inserted }
    }

    /// Creates an append outcome for an existing idempotency replay.
    ///
    /// # Arguments
    /// * `event` - Event returned from a previous persistence call.
    pub fn existing(event: LearningLedgerEvent) -> Self {
        Self { event, freshness: LearningLedgerEventFreshness::Existing }
    }
}

/// Typed request to review one captured learning candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Setters)]
#[setters(into)]
pub struct LearningReviewRequest {
    /// Candidate record to review.
    pub record_id: LearningRecordId,
    /// Conservative review decision.
    pub decision: LearningReviewDecision,
    /// Redacted reviewer note used only as event evidence.
    pub review_note: String,
    /// Typed review provenance.
    pub provenance: LearningProvenance,
}

impl LearningReviewRequest {
    /// Creates a typed learning review request.
    ///
    /// # Arguments
    /// * `record_id` - Candidate record identifier.
    /// * `decision` - Review decision to append.
    /// * `review_note` - Redacted note explaining the review.
    /// * `provenance` - Typed review provenance.
    pub fn new(
        record_id: LearningRecordId,
        decision: LearningReviewDecision,
        review_note: impl Into<String>,
        provenance: LearningProvenance,
    ) -> Self {
        Self {
            record_id,
            decision,
            review_note: review_note.into(),
            provenance,
        }
    }
}

/// Result of reviewing one learning candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LearningReviewOutcome {
    /// Review event appended or deduplicated by the ledger.
    pub event: LearningLedgerEvent,
    /// Projection after the review event was applied.
    pub projection: LearningRecordProjection,
}

/// Projected learning record returned by repository queries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LearningRecordProjection {
    /// Stable record identifier.
    pub record_id: LearningRecordId,
    /// Redacted candidate summary.
    pub summary: String,
    /// Current projected review state.
    pub review_state: LearningReviewState,
    /// Redaction status for the candidate summary.
    pub redaction_status: LearningRedactionStatus,
    /// Typed source provenance.
    pub provenance: LearningProvenance,
    /// Optional typed capture metadata for machine-generated candidates.
    pub capture_metadata: Option<LearningCaptureMetadata>,
    /// Candidate capture timestamp.
    pub created_at: DateTime<Utc>,
    /// Last event timestamp in this projection.
    pub updated_at: DateTime<Utc>,
    /// Schema version.
    pub schema_version: i32,
}

/// Cursor describing ledger/projection freshness for a workspace.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LearningLedgerFreshness {
    /// Highest append-only event sequence for the workspace.
    pub ledger_cursor: i64,
    /// Projection version derived from ledger cursor and query state.
    pub projection_version: i64,
    /// Review-state fingerprint for invalidating reviewed-only context.
    pub review_state_fingerprint: String,
}

/// Redacted summary and fingerprint generated before persistence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedactedLearningSummary {
    /// Redacted summary safe for persistence.
    pub summary: String,
    /// Fingerprint of the redacted summary.
    pub fingerprint: String,
    /// Redaction status.
    pub status: LearningRedactionStatus,
}

impl RedactedLearningSummary {
    /// Redacts sensitive-looking input and computes a stable fingerprint.
    ///
    /// # Arguments
    /// * `raw` - Untrusted candidate text. It is not persisted as-is.
    pub fn from_raw(raw: impl AsRef<str>) -> Self {
        let mut status = LearningRedactionStatus::Clean;
        let mut redacted_words = Vec::new();
        let mut redact_next = false;
        for word in raw.as_ref().split_whitespace() {
            if redact_next || looks_sensitive(word) {
                status = LearningRedactionStatus::Redacted;
                redacted_words.push("[REDACTED]".to_string());
                redact_next = !redact_next && introduces_secret_value(word);
            } else {
                redact_next = introduces_secret_value(word);
                redacted_words.push(word.to_string());
            }
        }
        let summary = redacted_words.join(" ");
        let fingerprint = digest_hex(&summary);
        Self { summary, fingerprint, status }
    }
}

/// Returns the stable digest used by learning summaries and metadata.
///
/// # Arguments
/// * `value` - Value to hash.
pub fn learning_digest_hex(value: impl AsRef<[u8]>) -> String {
    digest_hex(value)
}

fn introduces_secret_value(word: &str) -> bool {
    let lower = word
        .trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
        .to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "secret" | "token" | "password" | "apikey" | "api_key" | "bearer"
    )
}

fn looks_sensitive(word: &str) -> bool {
    if is_known_learning_fingerprint_word(word) {
        return false;
    }
    let lower = word.to_ascii_lowercase();
    lower.contains("secret")
        || lower.contains("token")
        || lower.contains("password")
        || lower.contains("apikey")
        || lower.contains("api_key")
        || lower.contains("bearer")
        || word.len() >= 24 && word.chars().any(|ch| ch.is_ascii_digit())
}

fn is_known_learning_fingerprint_word(word: &str) -> bool {
    let Some((key, value)) = word.split_once('=') else {
        return false;
    };
    key == "context_fingerprint"
        && value.len() == 64
        && value.chars().all(|ch| ch.is_ascii_hexdigit())
}

fn stable_learning_key(
    event_kind: LearningEventKind,
    source_kind: LearningSourceKind,
    source_id: &str,
    source_event_id: &str,
    content_fingerprint: &str,
) -> String {
    digest_hex(format!(
        "{}:{}:{}:{}:{}",
        event_kind, source_kind, source_id, source_event_id, content_fingerprint
    ))
}

fn stable_review_key(
    record_id: LearningRecordId,
    event_kind: LearningEventKind,
    source_kind: LearningSourceKind,
    source_id: &str,
    source_event_id: &str,
    content_fingerprint: &str,
) -> String {
    digest_hex(format!(
        "{}:{}",
        record_id.into_string(),
        stable_learning_key(
            event_kind,
            source_kind,
            source_id,
            source_event_id,
            content_fingerprint
        )
    ))
}

fn digest_hex(value: impl AsRef<[u8]>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_ref());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn capture_candidate_redacts_secret_bearing_input_and_builds_stable_key() {
        let conversation_id = ConversationId::generate();
        let provenance =
            LearningProvenance::conversation(conversation_id, "event-1", "source-fingerprint-1");
        let created_at = Utc::now();
        let left = LearningLedgerEvent::capture_candidate(
            "Do not store token sk-live-123456789012345678901234",
            provenance.clone(),
            created_at,
        )
        .unwrap();
        let right = LearningLedgerEvent::capture_candidate(
            "Do not store token sk-live-123456789012345678901234",
            provenance,
            created_at,
        )
        .unwrap();

        let actual = (
            left.summary.contains("sk-live"),
            left.redaction_status,
            left.idempotency_key == right.idempotency_key,
            left.schema_version,
        );
        let expected = (
            false,
            LearningRedactionStatus::Redacted,
            true,
            LEARNING_LEDGER_SCHEMA_VERSION,
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn review_idempotency_is_record_scoped_for_identical_review_evidence() {
        let conversation_id = ConversationId::generate();
        let provenance =
            LearningProvenance::conversation(conversation_id, "review-event", "review-fingerprint");
        let created_at = Utc::now();
        let left = LearningLedgerEvent::review(
            LearningRecordId::generate(),
            LearningEventKind::ReviewAccepted,
            "accepted by deterministic policy",
            provenance.clone(),
            created_at,
        )
        .unwrap();
        let right = LearningLedgerEvent::review(
            LearningRecordId::generate(),
            LearningEventKind::ReviewAccepted,
            "accepted by deterministic policy",
            provenance,
            created_at,
        )
        .unwrap();

        let actual = left.idempotency_key == right.idempotency_key;
        let expected = false;

        assert_eq!(actual, expected);
    }

    #[test]
    fn sensor_reviewer_contract_is_pure_input_to_output_shape() {
        let fixture: fn(
            &FakeLearningSensorReviewer,
            LearningSensorReviewInput,
        ) -> anyhow::Result<LearningSensorReviewOutput> =
            <FakeLearningSensorReviewer as LearningSensorReviewer>::review;

        let actual = std::any::type_name_of_val(&fixture).contains("LearningSensorReviewInput");
        let expected = true;

        assert_eq!(actual, expected);
    }

    #[test]
    fn metadata_only_sensor_input_produces_pending_not_proposal() {
        let projection = fixture_learning_projection(LearningReviewState::Candidate);
        let input = LearningSensorReviewInput::from_candidate_projection(&projection);
        let actual = FakeLearningSensorReviewer.review(input).unwrap().decision;
        let expected = LearningSensorDecisionKind::Pending;

        assert_eq!(actual, expected);
    }

    #[test]
    fn fake_fixture_sensor_input_produces_proposal() {
        let projection = fixture_learning_projection(LearningReviewState::Candidate);
        let input = LearningSensorReviewInput::fake_fixture(
            &projection,
            "Capture repeated failure pattern",
            "A substantive typed fixture observation requires a durable principle",
        );
        let actual = FakeLearningSensorReviewer.review(input).unwrap().decision;
        let expected = LearningSensorDecisionKind::ProposeLesson;

        assert_eq!(actual, expected);
    }

    #[test]
    fn sensor_output_validation_rejects_mismatched_identity_and_fingerprint() {
        let projection = fixture_learning_projection(LearningReviewState::Candidate);
        let input = LearningSensorReviewInput::fake_fixture(
            &projection,
            "Capture repeated failure pattern",
            "A substantive typed fixture observation requires a durable principle",
        );
        let mut output = FakeLearningSensorReviewer.review(input.clone()).unwrap();
        output.reviewer_id = "other_reviewer".to_string();
        let identity = output.validate_against(
            &input,
            FAKE_LEARNING_SENSOR_REVIEWER_ID,
            FAKE_LEARNING_SENSOR_REVIEWER_VERSION,
        );
        output.reviewer_id = FAKE_LEARNING_SENSOR_REVIEWER_ID.to_string();
        output.input_fingerprint = "wrong".to_string();
        let fingerprint = output.validate_against(
            &input,
            FAKE_LEARNING_SENSOR_REVIEWER_ID,
            FAKE_LEARNING_SENSOR_REVIEWER_VERSION,
        );
        let actual = (identity.is_err(), fingerprint.is_err());
        let expected = (true, true);

        assert_eq!(actual, expected);
    }

    #[test]
    fn sensor_output_rejects_unknown_json_and_forbidden_mutation_targets() {
        let projection = fixture_learning_projection(LearningReviewState::Candidate);
        let input = LearningSensorReviewInput::fake_fixture(
            &projection,
            "Capture repeated failure pattern",
            "A substantive typed fixture observation requires a durable principle",
        );
        let json = format!(
            r#"{{"schema_version":{},"reviewer_id":"{}","reviewer_version":{},"input_fingerprint":"{}","decision":"pending","reason_code":"insufficient_substantive_evidence","proposal_title":null,"proposal_body":null,"extra":"blocked"}}"#,
            LEARNING_SENSOR_REVIEW_SCHEMA_VERSION,
            FAKE_LEARNING_SENSOR_REVIEWER_ID,
            FAKE_LEARNING_SENSOR_REVIEWER_VERSION,
            input.fingerprint().unwrap()
        );
        let unknown = serde_json::from_str::<LearningSensorReviewOutput>(&json);
        let blocked_scopes = [
            "rules",
            "skills",
            "system prompt",
            "agent definition",
            "source code",
            "provider config",
            "tool access",
            "public publish",
            "credential",
            "/home/stranmor/.forge/AGENTS.md",
            "tool_action sendMessage",
        ];
        let blocked_count = blocked_scopes
            .iter()
            .filter(|scope| {
                let output = LearningSensorReviewOutput {
                    schema_version: LEARNING_SENSOR_REVIEW_SCHEMA_VERSION,
                    reviewer_id: FAKE_LEARNING_SENSOR_REVIEWER_ID.to_string(),
                    reviewer_version: FAKE_LEARNING_SENSOR_REVIEWER_VERSION,
                    input_fingerprint: input.fingerprint().unwrap(),
                    decision: LearningSensorDecisionKind::ProposeLesson,
                    reason_code: "typed_fixture_substantive_evidence".to_string(),
                    proposal_title: Some("Unsafe scope".to_string()),
                    proposal_body: Some(format!("mutate {scope}")),
                };
                output
                    .validate_against(
                        &input,
                        FAKE_LEARNING_SENSOR_REVIEWER_ID,
                        FAKE_LEARNING_SENSOR_REVIEWER_VERSION,
                    )
                    .is_err()
            })
            .count();
        let oversized = LearningSensorReviewOutput {
            schema_version: LEARNING_SENSOR_REVIEW_SCHEMA_VERSION,
            reviewer_id: FAKE_LEARNING_SENSOR_REVIEWER_ID.to_string(),
            reviewer_version: FAKE_LEARNING_SENSOR_REVIEWER_VERSION,
            input_fingerprint: input.fingerprint().unwrap(),
            decision: LearningSensorDecisionKind::ProposeLesson,
            reason_code: "typed_fixture_substantive_evidence".to_string(),
            proposal_title: Some("Oversized".to_string()),
            proposal_body: Some("x".repeat(1_025)),
        };
        let actual = (
            unknown.is_err(),
            blocked_count,
            oversized
                .validate_against(
                    &input,
                    FAKE_LEARNING_SENSOR_REVIEWER_ID,
                    FAKE_LEARNING_SENSOR_REVIEWER_VERSION,
                )
                .is_err(),
        );
        let expected = (true, blocked_scopes.len(), true);

        assert_eq!(actual, expected);
    }

    #[test]
    fn sensor_input_rejects_unknown_json_fields_before_validation() {
        let projection = fixture_learning_projection(LearningReviewState::Candidate);
        let input = LearningSensorReviewInput::from_candidate_projection(&projection);
        let json = format!(
            r#"{{"schema_version":{},"candidate_id":"{}","sanitized_projection_hash":"{}","sanitized_summary":"{}","sanitized_source_fingerprint":"{}","evidence_kind":"conversation_metadata","provenance_marker":"runtime_conversation_saved","fixture_title":null,"fixture_observation":null,"raw_transcript":"blocked","tool_action":"blocked","file_path":"/home/blocked","secret":"blocked"}}"#,
            input.schema_version,
            input.candidate_id.into_string(),
            input.sanitized_projection_hash,
            input.sanitized_summary,
            input.sanitized_source_fingerprint,
        );

        let actual = serde_json::from_str::<LearningSensorReviewInput>(&json).is_err();
        let expected = true;

        assert_eq!(actual, expected);
    }

    #[test]
    fn sensor_output_rejects_non_proposal_payload_for_pending_and_reject_decisions() {
        let projection = fixture_learning_projection(LearningReviewState::Candidate);
        let input = LearningSensorReviewInput::from_candidate_projection(&projection);
        let pending_with_payload = LearningSensorReviewOutput {
            schema_version: LEARNING_SENSOR_REVIEW_SCHEMA_VERSION,
            reviewer_id: FAKE_LEARNING_SENSOR_REVIEWER_ID.to_string(),
            reviewer_version: FAKE_LEARNING_SENSOR_REVIEWER_VERSION,
            input_fingerprint: input.fingerprint().unwrap(),
            decision: LearningSensorDecisionKind::Pending,
            reason_code: "insufficient_substantive_evidence".to_string(),
            proposal_title: Some("Hidden harmless title".to_string()),
            proposal_body: Some("Hidden harmless body changes audit identity".to_string()),
        };
        let reject_with_payload = LearningSensorReviewOutput {
            decision: LearningSensorDecisionKind::Reject,
            ..pending_with_payload.clone()
        };

        let actual = (
            pending_with_payload
                .validate_against(
                    &input,
                    FAKE_LEARNING_SENSOR_REVIEWER_ID,
                    FAKE_LEARNING_SENSOR_REVIEWER_VERSION,
                )
                .is_err(),
            reject_with_payload
                .validate_against(
                    &input,
                    FAKE_LEARNING_SENSOR_REVIEWER_ID,
                    FAKE_LEARNING_SENSOR_REVIEWER_VERSION,
                )
                .is_err(),
        );
        let expected = (true, true);

        assert_eq!(actual, expected);
    }

    #[test]
    fn conversation_metadata_sensor_input_rejects_fixture_payload_smuggling() {
        let projection = fixture_learning_projection(LearningReviewState::Candidate);
        let mut input = LearningSensorReviewInput::from_candidate_projection(&projection);
        input.fixture_title = Some("Hidden fixture title".to_string());
        input.fixture_observation = Some("Hidden fixture observation".to_string());

        let actual = input.validate().is_err();
        let expected = true;

        assert_eq!(actual, expected);
    }

    #[test]
    fn sensor_input_serialization_contains_no_raw_secret_or_tool_payload() {
        let projection = fixture_learning_projection(LearningReviewState::Candidate);
        let input = LearningSensorReviewInput::fake_fixture(
            &projection,
            "Durable typed observation",
            "The sanitized evidence has substantive recurring behavior",
        );

        let serialized = serde_json::to_string(&input).unwrap().to_ascii_lowercase();
        let actual = ["token", "password", "bearer", "tool payload", "tool_call"]
            .iter()
            .any(|needle| serialized.contains(needle));
        let expected = false;

        assert_eq!(actual, expected);
    }

    #[test]
    fn secret_shaped_sensor_input_is_rejected_before_review() {
        let projection = fixture_learning_projection(LearningReviewState::Candidate);
        let mut input = LearningSensorReviewInput::from_candidate_projection(&projection);
        input.fixture_observation = Some("bearer token should never be reviewed".to_string());

        let actual = input.validate().is_err();
        let expected = true;

        assert_eq!(actual, expected);
    }

    #[test]
    fn sensor_event_idempotency_includes_required_review_factors() {
        let projection = fixture_learning_projection(LearningReviewState::Candidate);
        let input = LearningSensorReviewInput::fake_fixture(
            &projection,
            "Durable typed observation",
            "The sanitized evidence has substantive recurring behavior",
        );
        let mut output = FakeLearningSensorReviewer.review(input.clone()).unwrap();
        let left = output
            .into_sensor_event(&input, Utc::now())
            .unwrap()
            .idempotency_key;
        output.reason_code = "different_reason".to_string();
        let reason_changed = output
            .into_sensor_event(&input, Utc::now())
            .unwrap()
            .idempotency_key;
        output.reason_code = "typed_fixture_substantive_evidence".to_string();
        output.reviewer_version = FAKE_LEARNING_SENSOR_REVIEWER_VERSION + 1;
        let invalid_version = output.into_sensor_event(&input, Utc::now());
        let actual = (left != reason_changed, invalid_version.is_err());
        let expected = (true, true);

        assert_eq!(actual, expected);
    }

    #[test]
    fn sensor_reviewer_identity_generalizes_fake_conversion_to_non_injection_events() {
        let projection = fixture_learning_projection(LearningReviewState::Candidate);
        let input = LearningSensorReviewInput::fake_fixture(
            &projection,
            "Durable typed observation",
            "The sanitized evidence has substantive recurring behavior",
        );
        let output = FakeLearningSensorReviewer.review(input.clone()).unwrap();
        let identity = LearningSensorReviewerIdentity::fake();

        output.validate_against_identity(&input, &identity).unwrap();
        let actual = output
            .into_sensor_event_with_identity(&input, &identity, Utc::now())
            .unwrap()
            .event_kind;
        let expected = LearningEventKind::SensorLessonProposed;

        assert_eq!(actual, expected);
        assert_ne!(actual, LearningEventKind::ReviewAccepted);
    }

    #[test]
    fn sensor_output_rejects_accept_like_enum_values() {
        let projection = fixture_learning_projection(LearningReviewState::Candidate);
        let input = LearningSensorReviewInput::from_candidate_projection(&projection);
        let fixture = |decision: &str| {
            format!(
                r#"{{"schema_version":{},"reviewer_id":"{}","reviewer_version":{},"input_fingerprint":"{}","decision":"{}","reason_code":"blocked_accept_path","proposal_title":null,"proposal_body":null}}"#,
                LEARNING_SENSOR_REVIEW_SCHEMA_VERSION,
                FAKE_LEARNING_SENSOR_REVIEWER_ID,
                FAKE_LEARNING_SENSOR_REVIEWER_VERSION,
                input.fingerprint().unwrap(),
                decision
            )
        };

        let actual = ["accept", "accepted", "review_accepted"]
            .iter()
            .map(|decision| {
                serde_json::from_str::<LearningSensorReviewOutput>(&fixture(decision)).is_err()
            })
            .collect::<Vec<_>>();
        let expected = vec![true, true, true];

        assert_eq!(actual, expected);
    }

    #[test]
    fn sanitized_chat_observation_validates_and_builds_sensor_input() {
        let projection = fixture_learning_projection(LearningReviewState::Candidate);
        let observation = fixture_sanitized_observation().validate().unwrap();

        let actual =
            LearningSensorReviewInput::from_sanitized_chat_observation(&projection, observation);
        let expected = (
            LearningSensorEvidenceKind::SanctionedSanitizedChatObservation,
            LearningSensorProvenanceMarker::RuntimeSanitizedChatObservation,
            true,
            true,
        );

        assert_eq!(
            (
                actual.evidence_kind,
                actual.provenance_marker,
                actual.sanitized_chat_observation.is_some(),
                actual.validate().is_ok(),
            ),
            expected
        );
    }

    #[test]
    fn sanitized_chat_observation_rejects_unknown_missing_and_unknown_enum_fields() {
        let valid = fixture_sanitized_observation();
        let valid_json = serde_json::to_string(&valid).unwrap();
        let unknown_json = valid_json.replace('}', r#",\"raw_text\":\"blocked\"}"#);
        let missing_json = r#"{"observation_kind":"repeated_agent_behavior","occurrence_bucket":"two","severity":"medium","source_cluster_fingerprint":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}"#
            .to_string();
        let unknown_enum_json = valid_json.replace("repeated_agent_behavior", "unknown_behavior");

        let actual = (
            serde_json::from_str::<SanitizedChatLessonObservation>(&unknown_json).is_err(),
            serde_json::from_str::<SanitizedChatLessonObservation>(&missing_json).is_err(),
            serde_json::from_str::<SanitizedChatLessonObservation>(&unknown_enum_json).is_err(),
        );
        let expected = (true, true, true);

        assert_eq!(actual, expected);
    }

    #[test]
    fn sanitized_chat_observation_fingerprint_constraints_are_enforced() {
        let cases = vec![
            "".to_string(),
            "a".repeat(63),
            "a".repeat(65),
            "A".repeat(64),
            "g".repeat(64),
        ];

        let actual = cases
            .into_iter()
            .map(|value| OpaqueLearningFingerprint::new(value).is_err())
            .collect::<Vec<_>>();
        let expected = vec![true, true, true, true, true];

        assert_eq!(actual, expected);
    }

    #[test]
    fn sanitized_chat_observation_rejects_mismatched_gate_payloads() {
        let projection = fixture_learning_projection(LearningReviewState::Candidate);
        let mut observation_input = LearningSensorReviewInput::from_sanitized_chat_observation(
            &projection,
            fixture_sanitized_observation().validate().unwrap(),
        );
        observation_input.provenance_marker =
            LearningSensorProvenanceMarker::RuntimeConversationSaved;
        let mut fixture_input = LearningSensorReviewInput::fake_fixture(
            &projection,
            "Durable typed observation",
            "The sanitized evidence has substantive recurring behavior",
        );
        fixture_input.sanitized_chat_observation = Some(fixture_sanitized_observation());
        let mut metadata_input = LearningSensorReviewInput::from_candidate_projection(&projection);
        metadata_input.sanitized_chat_observation = Some(fixture_sanitized_observation());

        let actual = (
            observation_input.validate().is_err(),
            fixture_input.validate().is_err(),
            metadata_input.validate().is_err(),
        );
        let expected = (true, true, true);

        assert_eq!(actual, expected);
    }

    #[test]
    fn conversation_metadata_still_cannot_propose_lesson_or_accept() {
        let projection = fixture_learning_projection(LearningReviewState::Candidate);
        let input = LearningSensorReviewInput::from_candidate_projection(&projection);
        let proposal = LearningSensorReviewOutput {
            schema_version: LEARNING_SENSOR_REVIEW_SCHEMA_VERSION,
            reviewer_id: FAKE_LEARNING_SENSOR_REVIEWER_ID.to_string(),
            reviewer_version: FAKE_LEARNING_SENSOR_REVIEWER_VERSION,
            input_fingerprint: input.fingerprint().unwrap(),
            decision: LearningSensorDecisionKind::ProposeLesson,
            reason_code: "blocked_runtime_metadata".to_string(),
            proposal_title: Some("Metadata proposal".to_string()),
            proposal_body: Some("Metadata cannot produce proposal".to_string()),
        };
        let accept_json = serde_json::to_string(&proposal)
            .unwrap()
            .replace("propose_lesson", "accept");

        let actual = (
            proposal
                .validate_against(
                    &input,
                    FAKE_LEARNING_SENSOR_REVIEWER_ID,
                    FAKE_LEARNING_SENSOR_REVIEWER_VERSION,
                )
                .is_err(),
            serde_json::from_str::<LearningSensorReviewOutput>(&accept_json).is_err(),
        );
        let expected = (true, true);

        assert_eq!(actual, expected);
    }

    #[test]
    fn sanitized_observation_can_validate_proposal_into_sensor_audit_event() {
        let projection = fixture_learning_projection(LearningReviewState::Candidate);
        let input = LearningSensorReviewInput::from_sanitized_chat_observation(
            &projection,
            fixture_sanitized_observation().validate().unwrap(),
        );
        let output = FakeLearningSensorReviewer.review(input.clone()).unwrap();

        let actual = output
            .into_sensor_event(&input, Utc::now())
            .unwrap()
            .event_kind;
        let expected = LearningEventKind::SensorLessonProposed;

        assert_eq!(actual, expected);
        assert_ne!(actual, LearningEventKind::ReviewAccepted);
    }

    #[test]
    fn stage_two_output_cannot_carry_commands_patches_rule_text_or_freeform_payloads() {
        let projection = fixture_learning_projection(LearningReviewState::Candidate);
        let input = LearningSensorReviewInput::from_sanitized_chat_observation(
            &projection,
            fixture_sanitized_observation().validate().unwrap(),
        );
        let blocked = [
            "cargo test -p forge_domain",
            "git apply patch",
            "https://example.com",
            "user@example.com",
            "/home/stranmor/file",
            "token abcdefghijklmnopqrstuvwxyz",
            "system prompt update",
            "source code mutation",
        ];
        let actual = blocked
            .iter()
            .map(|payload| {
                LearningSensorReviewOutput {
                    schema_version: LEARNING_SENSOR_REVIEW_SCHEMA_VERSION,
                    reviewer_id: FAKE_LEARNING_SENSOR_REVIEWER_ID.to_string(),
                    reviewer_version: FAKE_LEARNING_SENSOR_REVIEWER_VERSION,
                    input_fingerprint: input.fingerprint().unwrap(),
                    decision: LearningSensorDecisionKind::ProposeLesson,
                    reason_code: "sanctioned_sanitized_chat_observation".to_string(),
                    proposal_title: Some("Blocked payload".to_string()),
                    proposal_body: Some((*payload).to_string()),
                }
                .validate_against(
                    &input,
                    FAKE_LEARNING_SENSOR_REVIEWER_ID,
                    FAKE_LEARNING_SENSOR_REVIEWER_VERSION,
                )
                .is_err()
            })
            .collect::<Vec<_>>();
        let expected = vec![true; blocked.len()];

        assert_eq!(actual, expected);
    }

    #[test]
    fn stage_two_output_rejects_command_shaped_snake_case_tokens() {
        let projection = fixture_learning_projection(LearningReviewState::Candidate);
        let input = LearningSensorReviewInput::from_sanitized_chat_observation(
            &projection,
            fixture_sanitized_observation().validate().unwrap(),
        );
        let output = LearningSensorReviewOutput {
            schema_version: LEARNING_SENSOR_REVIEW_SCHEMA_VERSION,
            reviewer_id: FAKE_LEARNING_SENSOR_REVIEWER_ID.to_string(),
            reviewer_version: FAKE_LEARNING_SENSOR_REVIEWER_VERSION,
            input_fingerprint: input.fingerprint().unwrap(),
            decision: LearningSensorDecisionKind::ProposeLesson,
            reason_code: "sanctioned_sanitized_chat_observation".to_string(),
            proposal_title: Some("safe_title".to_string()),
            proposal_body: Some("run_shell".to_string()),
        };

        let actual = output
            .validate_against(
                &input,
                FAKE_LEARNING_SENSOR_REVIEWER_ID,
                FAKE_LEARNING_SENSOR_REVIEWER_VERSION,
            )
            .is_err();
        let expected = true;

        assert_eq!(actual, expected);
    }

    fn fixture_sanitized_observation() -> SanitizedChatLessonObservation {
        SanitizedChatLessonObservation::new(
            SanitizedChatObservationKind::RepeatedAgentBehavior,
            SanitizedObservationCountBucket::Two,
            SanitizedObservationSeverity::Medium,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        )
        .unwrap()
    }

    fn fixture_learning_projection(review_state: LearningReviewState) -> LearningRecordProjection {
        let provenance = LearningProvenance::conversation(
            ConversationId::generate(),
            "source-event",
            "safe-source-fingerprint",
        );
        LearningRecordProjection {
            record_id: LearningRecordId::generate(),
            summary: "conversation_saved message_count=2 user_message_count=1 context_fingerprint=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            review_state,
            redaction_status: LearningRedactionStatus::Clean,
            provenance,
            capture_metadata: Some(LearningCaptureMetadata::conversation_save(
                2,
                1,
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                learning_digest_hex("conversation_saved message_count=2 user_message_count=1 context_fingerprint=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            )),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            schema_version: LEARNING_LEDGER_SCHEMA_VERSION,
        }
    }
}
