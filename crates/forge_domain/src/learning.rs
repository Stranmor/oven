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
pub const LEARNING_LEDGER_SCHEMA_VERSION: i32 = 1;

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
            created_at,
            schema_version: LEARNING_LEDGER_SCHEMA_VERSION,
        })
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
    let lower = word.to_ascii_lowercase();
    lower.contains("secret")
        || lower.contains("token")
        || lower.contains("password")
        || lower.contains("apikey")
        || lower.contains("api_key")
        || lower.contains("bearer")
        || word.len() >= 24 && word.chars().any(|ch| ch.is_ascii_digit())
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
    fn provenance_requires_typed_source_identifier() {
        let fixture = LearningProvenance {
            source_kind: LearningSourceKind::Conversation,
            conversation_id: None,
            task_id: None,
            tool_name: None,
            eval_id: None,
            source_event_id: "event-1".to_string(),
            source_fingerprint: "source-fingerprint-1".to_string(),
        };

        let actual = fixture.validate().is_err();
        let expected = true;

        assert_eq!(actual, expected);
    }
}
