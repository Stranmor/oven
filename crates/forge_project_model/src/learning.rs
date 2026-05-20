use std::collections::BTreeSet;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::{
    EdgeConfidence, GraphEdgeKind, KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode,
    KnowledgeGraphNodeId, Provenance, RetrievedEvidenceGraphNode,
};

/// Typed summary projection for accepted sensor-promoted lessons.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct AcceptedLearningSummary(String);

impl AcceptedLearningSummary {
    /// Creates a validated accepted summary projection.
    ///
    /// # Arguments
    /// * `value` - Summary text safe for accepted context injection.
    ///
    /// # Errors
    /// Returns an error when `value` is not in the closed promotion allowlist.
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value != "sanctioned_sanitized_observation:validated_counters_and_fingerprints" {
            bail!("accepted learning summary is not allowlisted");
        }
        Ok(Self(value))
    }

    /// Returns the accepted summary text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for AcceptedLearningSummary {
    type Error = anyhow::Error;

    fn try_from(value: String) -> Result<Self> {
        Self::new(value)
    }
}

impl From<AcceptedLearningSummary> for String {
    fn from(value: AcceptedLearningSummary) -> Self {
        value.0
    }
}

/// Transport invariants for rendered self-learning context payloads.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LearningContextTransport {
    /// Whether this payload may be dropped by context compaction.
    pub droppable: bool,
    /// Whether this payload may receive provider prompt-cache markers.
    pub cacheable: bool,
}

impl Default for LearningContextTransport {
    fn default() -> Self {
        Self { droppable: true, cacheable: false }
    }
}

/// Ledger freshness cursor used to invalidate late-bound learning context.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LearningLedgerFreshness {
    /// Highest observed append-only ledger sequence.
    pub ledger_cursor: i64,
    /// Projection version computed by the persistence layer.
    pub projection_version: i64,
    /// Stable review-state fingerprint for the selected projection.
    pub review_state_fingerprint: String,
}

/// Bounded renderable learning context built from accepted ledger records.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LearningContextPayload {
    /// Transport invariants enforced by application injection.
    pub transport: LearningContextTransport,
    /// Freshness cursor for this payload.
    pub freshness: LearningLedgerFreshness,
    /// Accepted record summaries safe for prompt injection.
    pub records: Vec<LearningContextRecord>,
}

impl LearningContextPayload {
    /// Creates a renderable learning context payload.
    ///
    /// # Arguments
    /// * `freshness` - Ledger/projection freshness cursor.
    /// * `records` - Accepted records included in this payload.
    pub fn new(freshness: LearningLedgerFreshness, records: Vec<LearningContextRecord>) -> Self {
        Self {
            transport: LearningContextTransport::default(),
            freshness,
            records,
        }
    }

    /// Renders this bounded payload for late-bound user-scoped injection.
    ///
    /// # Errors
    /// Returns an error if any record is not accepted for injection or if the
    /// payload violates transport invariants.
    pub fn render(&self) -> Result<String> {
        if !self.transport.droppable || self.transport.cacheable {
            bail!("learning context transport must be droppable and cache-ineligible");
        }
        for record in &self.records {
            record.validate_for_render()?;
        }
        serde_json::to_string(self).map_err(Into::into)
    }
}

/// One accepted learning record exposed to late-bound context rendering.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LearningContextRecord {
    /// Stable ledger record identifier.
    pub id: String,
    /// Redacted human-safe summary.
    pub summary: String,
    /// Review state used to gate injection.
    pub review_state: LearningReviewState,
    /// Redaction status for the persisted record.
    pub redaction_status: LearningRedactionStatus,
    /// Source provenance for this learning item.
    pub provenance: LearningProvenance,
}

impl LearningContextRecord {
    /// Validates this accepted record before rendering into model context.
    ///
    /// # Errors
    /// Returns an error when the record is not accepted, lacks provenance, or
    /// carries a sensor-promotion-shaped raw summary instead of an accepted projection.
    pub fn validate_for_render(&self) -> Result<()> {
        if self.review_state != LearningReviewState::Accepted {
            bail!("learning context can only render accepted records");
        }
        self.provenance.validate()?;
        let summary = self.summary.trim();
        if summary.is_empty() {
            bail!("learning context summary is required");
        }
        if summary.starts_with("sensor_proposal ") || summary.contains("body=") {
            bail!("learning context cannot render raw sensor proposal summaries");
        }
        if summary.starts_with("sanctioned_sanitized_observation:") {
            AcceptedLearningSummary::new(summary.to_string())?;
        }
        Ok(())
    }
}

/// Review state projected from append-only learning ledger events.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearningReviewState {
    /// Candidate captured but not reviewed.
    Candidate,
    /// Candidate rejected by reviewer or policy.
    Rejected,
    /// Candidate reviewed and accepted for bounded context injection.
    Accepted,
    /// Candidate superseded by another accepted or rejected record.
    Superseded,
}

/// Redaction state for persisted learning summaries and fingerprints.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearningRedactionStatus {
    /// Input required no secret redaction.
    Clean,
    /// Input contained sensitive-looking material and was redacted before save.
    Redacted,
}

/// Source type for captured learning provenance.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

/// Typed provenance for a learning ledger event or projection item.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LearningProvenance {
    /// Source kind that produced the learning candidate or review event.
    pub source_kind: LearningSourceKind,
    /// Stable source identifier, such as conversation ID or task ID.
    pub source_id: String,
    /// Optional source event identifier.
    pub source_event_id: Option<String>,
    /// Optional source timestamp in RFC3339 format.
    pub source_timestamp: Option<String>,
    /// Fingerprint of redacted source evidence.
    pub source_fingerprint: String,
}

impl LearningProvenance {
    /// Validates provenance completeness without reading raw source payloads.
    ///
    /// # Errors
    /// Returns an error when required provenance fields are empty.
    pub fn validate(&self) -> Result<()> {
        if self.source_id.trim().is_empty() {
            bail!("learning provenance source_id is required");
        }
        if self.source_fingerprint.trim().is_empty() {
            bail!("learning provenance source_fingerprint is required");
        }
        Ok(())
    }
}

impl LearningContextRecord {
    /// Converts this record into a graph evidence node.
    ///
    /// # Errors
    /// Returns an error when provenance is incomplete.
    pub fn to_graph_node(&self) -> Result<KnowledgeGraphNode> {
        self.provenance.validate()?;
        Ok(KnowledgeGraphNode::RetrievedEvidence(
            RetrievedEvidenceGraphNode {
                id: KnowledgeGraphNodeId::RetrievedEvidence(self.id.clone()),
                evidence_id: self.id.clone(),
                path: format!("learning/{}", self.provenance.source_kind.as_str()),
                freshness: crate::EvidenceFreshness::Fresh,
                provenance: Provenance {
                    path: format!("learning/{}", self.provenance.source_kind.as_str()),
                    start_line: None,
                    end_line: None,
                    source: self.provenance.source_id.clone(),
                    fingerprint: self.provenance.source_fingerprint.clone(),
                },
            },
        ))
    }
}

impl LearningSourceKind {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Conversation => "conversation",
            Self::Task => "task",
            Self::Tool => "tool",
            Self::Eval => "eval",
        }
    }
}

/// Builds a validated graph surface for learning context records.
///
/// # Arguments
/// * `records` - Learning records to expose as graph evidence.
///
/// # Errors
/// Returns an error when graph endpoint validation or provenance validation
/// fails.
pub fn learning_records_to_graph(records: &[LearningContextRecord]) -> Result<KnowledgeGraph> {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut seen_sources = BTreeSet::new();
    for record in records {
        let node = record.to_graph_node()?;
        let node_id = node.id().clone();
        let source_id = KnowledgeGraphNodeId::Task(record.provenance.source_id.clone());
        if seen_sources.insert(source_id.clone()) {
            nodes.push(KnowledgeGraphNode::Task(crate::TaskGraphNode {
                id: source_id.clone(),
                title: record.provenance.source_id.clone(),
                status: "source".to_string(),
                provenance: Provenance {
                    path: format!("learning/{}", record.provenance.source_kind.as_str()),
                    start_line: None,
                    end_line: None,
                    source: record.provenance.source_id.clone(),
                    fingerprint: record.provenance.source_fingerprint.clone(),
                },
            }));
        }
        edges.push(KnowledgeGraphEdge {
            from: source_id,
            to: node_id,
            kind: GraphEdgeKind::EvidenceCites,
            confidence: 1.0,
            confidence_kind: EdgeConfidence::ExactCompiler,
            provenance: Provenance {
                path: format!("learning/{}", record.provenance.source_kind.as_str()),
                start_line: None,
                end_line: None,
                source: record.provenance.source_id.clone(),
                fingerprint: record.provenance.source_fingerprint.clone(),
            },
        });
        nodes.push(node);
    }
    KnowledgeGraph::new(nodes, edges)
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use pretty_assertions::assert_eq;

    use super::*;

    fn fixture_record(id: &str, review_state: LearningReviewState) -> LearningContextRecord {
        LearningContextRecord {
            id: id.to_string(),
            summary: "Use typed append-only events".to_string(),
            review_state,
            redaction_status: LearningRedactionStatus::Clean,
            provenance: LearningProvenance {
                source_kind: LearningSourceKind::Conversation,
                source_id: "conversation-1".to_string(),
                source_event_id: Some("event-1".to_string()),
                source_timestamp: Some("2026-05-19T10:00:00Z".to_string()),
                source_fingerprint: "fingerprint-1".to_string(),
            },
        }
    }

    #[test]
    fn learning_context_payload_renders_only_accepted_records() -> Result<()> {
        let fixture = LearningContextPayload::new(
            LearningLedgerFreshness {
                ledger_cursor: 1,
                projection_version: 1,
                review_state_fingerprint: "accepted".to_string(),
            },
            vec![fixture_record("learning-1", LearningReviewState::Accepted)],
        );

        let actual = fixture.render()?.contains("Use typed append-only events");
        let expected = true;

        assert_eq!(actual, expected);
        assert_eq!(fixture.transport.droppable, true);
        assert_eq!(fixture.transport.cacheable, false);
        Ok(())
    }

    #[test]
    fn learning_context_payload_rejects_unreviewed_candidates() {
        let fixture = LearningContextPayload::new(
            LearningLedgerFreshness::default(),
            vec![fixture_record("learning-1", LearningReviewState::Candidate)],
        );

        let actual = fixture.render().is_err();
        let expected = true;

        assert_eq!(actual, expected);
    }

    #[test]
    fn learning_context_payload_rejects_raw_sensor_proposal_summary() {
        let mut fixture = fixture_record("learning-1", LearningReviewState::Accepted);
        fixture.summary =
            "sensor_proposal reviewer=fake body=validated_counters_and_fingerprints".to_string();
        let payload =
            LearningContextPayload::new(LearningLedgerFreshness::default(), vec![fixture]);

        let actual = payload.render().is_err();
        let expected = true;

        assert_eq!(actual, expected);
    }

    #[test]
    fn accepted_learning_summary_roundtrips_through_json() -> Result<()> {
        let fixture = AcceptedLearningSummary::new(
            "sanctioned_sanitized_observation:validated_counters_and_fingerprints",
        )?;

        let serialized = serde_json::to_string(&fixture)?;
        let actual: AcceptedLearningSummary = serde_json::from_str(&serialized)?;
        let expected = fixture;

        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn learning_records_to_graph_validates_provenance_and_edges() -> Result<()> {
        let fixture = vec![fixture_record("learning-1", LearningReviewState::Accepted)];

        let actual = learning_records_to_graph(&fixture)?;
        let expected = (2usize, 1usize);

        assert_eq!((actual.nodes.len(), actual.edges.len()), expected);
        Ok(())
    }

    #[test]
    fn learning_records_to_graph_rejects_incomplete_provenance() {
        let mut fixture = fixture_record("learning-1", LearningReviewState::Accepted);
        fixture.provenance.source_fingerprint = String::new();

        let actual = learning_records_to_graph(&[fixture]).is_err();
        let expected = true;

        assert_eq!(actual, expected);
    }
}
