use std::collections::BTreeMap;
use std::path::PathBuf;

use derive_more::Display;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Workspace identifier (UUID) from workspace server.
///
/// Generated locally and sent to server during CreateWorkspace.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Display)]
#[display("{}", _0)]
pub struct WorkspaceId(Uuid);

impl WorkspaceId {
    /// Generate a new random workspace ID
    pub fn generate() -> Self {
        Self(Uuid::new_v4())
    }

    /// Parse a workspace ID from a string
    ///
    /// # Errors
    /// Returns an error if the string is not a valid UUID
    pub fn from_string(s: &str) -> anyhow::Result<Self> {
        Ok(Self(Uuid::parse_str(s)?))
    }

    /// Get the inner UUID
    pub fn inner(&self) -> Uuid {
        self.0
    }
}

/// Freshness state for the local project-model manifest used by automatic
/// context injection and workspace diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkspaceContextFreshness {
    /// The local project-model manifest exists and matches the current
    /// filesystem view.
    Fresh,
    /// The local project-model manifest exists but no longer matches the
    /// current filesystem view.
    Stale {
        /// Files changed since the manifest was written.
        changed: Vec<String>,
        /// Files deleted since the manifest was written.
        deleted: Vec<String>,
        /// Files added since the manifest was written.
        added: Vec<String>,
    },
    /// Freshness could not be proven, so callers must treat the manifest as
    /// unavailable for injection.
    Unknown {
        /// Redaction-safe reason freshness could not be evaluated.
        reason: String,
    },
}

impl WorkspaceContextFreshness {
    /// Returns true only when the manifest is proven fresh.
    pub fn is_fresh(&self) -> bool {
        matches!(self, Self::Fresh)
    }

    /// Returns a stable diagnostic label for this freshness state.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Fresh => "fresh",
            Self::Stale { .. } => "stale",
            Self::Unknown { .. } => "unknown",
        }
    }
}

/// Local project-model exact-fact readiness diagnostic for a workspace candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceExactFactReadinessDiagnostic {
    /// Stable readiness status label.
    pub status_label: String,
    /// Whether persisted exact facts are active for this workspace.
    pub exact_facts_active: bool,
    /// Total redaction-safe issue count before summary capping.
    pub issue_count: usize,
    /// Deterministically capped redaction-safe issue summaries.
    pub issue_summaries: Vec<String>,
    /// Persisted manifest hash when available.
    pub manifest_hash: Option<String>,
    /// Manifest external-facts fingerprint when available.
    pub manifest_external_facts_fingerprint: Option<String>,
    /// Graph-visible reference edge count.
    pub reference_edge_count: usize,
    /// Graph-visible exact compiler reference edge count.
    pub exact_compiler_reference_edge_count: usize,
}

/// Local project-model evidence-ledger graph metadata for context-pack artifacts and tool episodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceEvidenceLedgerGraphMetadata {
    /// Total typed graph node count.
    pub node_count: usize,
    /// Total typed graph edge count.
    pub edge_count: usize,
    /// Node counts keyed by stable node-kind label.
    pub node_kind_counts: std::collections::BTreeMap<String, usize>,
    /// Edge counts keyed by stable edge-kind label.
    pub edge_kind_counts: std::collections::BTreeMap<String, usize>,
}

/// Local project-model evidence-ledger activation summary for context-pack artifacts and tool episodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceEvidenceLedgerActivationSummary {
    /// Number of context-pack artifact candidates inspected under budget.
    pub context_pack_artifact_count: usize,
    /// Number of inspected context-pack artifacts that were readable.
    pub readable_context_pack_count: usize,
    /// Number of valid tool episodes inspected under budget.
    pub tool_episode_count: usize,
    /// Number of inspected tool episodes linked to a readable context-pack artifact.
    pub linked_episode_count: usize,
    /// Number of linkage issues or missing context-pack artifact references.
    pub missing_link_count: usize,
    /// Graph node count computed from metadata-only activation graph construction.
    pub graph_node_count: usize,
    /// Graph edge count computed from metadata-only activation graph construction.
    pub graph_edge_count: usize,
    /// Worst-case freshness across readable context-pack artifacts.
    pub worst_case_freshness: Option<String>,
    /// Total redaction-safe issue count before summary capping.
    pub issue_count: usize,
    /// Deterministically capped stable issue labels.
    pub issue_summaries: Vec<String>,
    /// Whether any activation budget omitted data or graph metadata.
    pub truncated: bool,
}

/// Local project-model evidence-ledger activation diagnostic for context-pack artifacts and tool episodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceEvidenceLedgerActivationDiagnostic {
    /// Compact counters and redaction-safe proof labels.
    pub summary: WorkspaceEvidenceLedgerActivationSummary,
    /// Optional metadata-only graph proof omitted when graph budgets are exceeded.
    pub graph: Option<WorkspaceEvidenceLedgerGraphMetadata>,
}

/// Local project-model evidence readiness diagnostic for context-pack artifacts and tool episodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceEvidenceReadinessDiagnostic {
    /// Number of context-pack artifacts inspected under the bounded diagnostic budget.
    pub context_pack_artifact_count: usize,
    /// Whether inspected context-pack artifacts were readable and structurally valid.
    pub context_pack_valid: bool,
    /// Total redaction-safe context-pack issue count before summary capping.
    pub context_pack_issue_count: usize,
    /// Number of valid tool episodes inspected under the bounded diagnostic budget.
    pub tool_episode_count: usize,
    /// Whether inspected tool episodes were readable and structurally valid.
    pub tool_episode_valid: bool,
    /// Total redaction-safe tool-episode issue count before summary capping.
    pub tool_episode_issue_count: usize,
    /// Whether inspected tool episodes link only to existing context-pack artifacts.
    pub episode_artifact_link_valid: bool,
    /// Number of inspected tool episodes linked to an existing context-pack artifact.
    pub linked_episode_count: usize,
    /// Number of linkage issues or missing context-pack artifact references.
    pub missing_link_count: usize,
    /// Worst-case freshness across readable context-pack artifacts.
    pub worst_case_freshness: Option<String>,
    /// Deterministically capped redaction-safe issue summaries.
    pub issue_summaries: Vec<String>,
    /// Whether inspection exceeded configured diagnostic budgets.
    pub truncated: bool,
}

/// Local project-model manifest diagnostic for a workspace candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceContextManifestDiagnostic {
    /// Workspace root path being evaluated.
    pub workspace_root: PathBuf,
    /// Expected local project-model manifest path.
    pub manifest_path: PathBuf,
    /// Whether the manifest file exists at the expected path.
    pub manifest_found: bool,
    /// Freshness classification for the manifest when present.
    pub freshness: WorkspaceContextFreshness,
    /// Read-only exact-fact readiness for this manifest root, when evaluated.
    pub exact_fact_readiness: Option<WorkspaceExactFactReadinessDiagnostic>,
    /// Read-only evidence readiness for context-pack artifacts and tool episodes, when evaluated.
    pub evidence_readiness: Option<WorkspaceEvidenceReadinessDiagnostic>,
    /// Read-only evidence-ledger activation proof for context-pack artifacts and tool episodes, when evaluated.
    pub evidence_ledger_activation: Option<WorkspaceEvidenceLedgerActivationDiagnostic>,
}

impl WorkspaceContextManifestDiagnostic {
    /// Returns true only when the manifest is present and proven fresh.
    pub fn can_inject(&self) -> bool {
        self.manifest_found && self.freshness.is_fresh()
    }
}

/// Candidate path considered while explaining project-model context injection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceContextCandidateDiagnostic {
    /// Candidate path before ancestor scanning.
    pub candidate_path: PathBuf,
    /// Selected workspace root for this candidate, when a fresh manifest is
    /// found.
    pub selected_workspace: Option<PathBuf>,
    /// Path filter that would be applied to retrieval for this candidate.
    pub path_filter: Option<String>,
    /// Exact reason this candidate was not selected.
    pub skip_reason: Option<String>,
}

/// Query-specific read-only retrieval-plan diagnostic for explain-context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkspaceRetrievalPlanDiagnostic {
    /// Stable redaction-safe label for the canonical workspace root.
    pub workspace_root_label: String,
    /// Stable redaction-safe label for the project-model manifest.
    pub manifest_label: String,
    /// Whether the project-model planner produced an executable plan.
    pub planned: bool,
    /// Stable machine-readable refusal code when planning was refused.
    pub refusal_code: Option<String>,
    /// Human-readable redaction-safe refusal detail when planning was refused.
    pub refusal_detail: Option<String>,
    /// Number of retrieval results selected by the planner.
    pub selected_result_count: usize,
    /// Number of validated read requests planned before readback.
    pub read_request_count: usize,
    /// Deterministic write decision label when planning succeeded.
    pub write_decision: Option<String>,
    /// Bounded metadata-only summaries of selected retrieval results.
    pub selected_summaries: Vec<WorkspaceRetrievalPlanSelectedSummary>,
    /// Bounded metadata-only summaries of planned read requests.
    pub read_request_summaries: Vec<WorkspaceRetrievalPlanReadRequestSummary>,
    /// Whether retrieval selected no evidence.
    pub retrieval_empty: bool,
    /// Whether selected or read-request summaries were truncated.
    pub truncated: bool,
}

/// Metadata-only selected-result summary for explain-context planner diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkspaceRetrievalPlanSelectedSummary {
    /// Evidence identifier selected by retrieval.
    pub evidence_id: String,
    /// Manifest-relative path associated with the selected result.
    pub path: String,
    /// Optional one-based inclusive start line.
    pub start_line: Option<u32>,
    /// Optional one-based inclusive end line.
    pub end_line: Option<u32>,
    /// Planner relevance score.
    pub relevance: f32,
}

/// Metadata-only read-request summary for explain-context planner diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkspaceRetrievalPlanReadRequestSummary {
    /// Evidence identifier planned for readback.
    pub evidence_id: String,
    /// Manifest-relative path planned for readback.
    pub path: String,
    /// One-based inclusive start line.
    pub start_line: u32,
    /// One-based inclusive end line.
    pub end_line: u32,
}

/// User-facing explanation for automatic project-model context injection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkspaceContextExplanation {
    /// Current working directory used for target resolution.
    pub cwd: PathBuf,
    /// Latest user query or explicit diagnostic query being explained.
    pub query: Option<String>,
    /// Candidate paths extracted from cwd and path mentions.
    pub candidates: Vec<WorkspaceContextCandidateDiagnostic>,
    /// Fresh selected targets that would be queried for context.
    pub selected_targets: Vec<WorkspaceContextManifestDiagnostic>,
    /// Nearest manifest candidates skipped because manifest readiness blocked injection.
    pub nearest_skipped_manifest_candidates: Vec<WorkspaceContextManifestDiagnostic>,
    /// Selected target roots whose retrieval returned no usable nodes for the
    /// explained query. This legacy field is preserved for porcelain
    /// compatibility; read-only explain no longer performs retrieval dry-runs.
    pub retrieval_empty_targets: Vec<PathBuf>,
    /// Query-specific read-only retrieval-plan diagnostics. These diagnostics
    /// are planner-derived, pre-readback, and never include source content or
    /// persisted context-pack bodies.
    #[serde(default)]
    pub retrieval_plan_diagnostics: Vec<WorkspaceRetrievalPlanDiagnostic>,
    /// Read-only replay-derived preview diagnostics from the existing evidence
    /// ledger. These diagnostics are non-query-specific and do not predict
    /// query_workspace retrieval output.
    #[serde(default)]
    pub replay_preview_diagnostics: Vec<WorkspaceEvidenceReplayPreviewDiagnostic>,
    /// Whether automatic project-model context would pass the manifest/query gate.
    /// This does not predict query-specific retrieval output.
    pub would_inject: bool,
    /// Exact top-level reason context would not be injected.
    pub skip_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_context_explanation_deserializes_legacy_payload_without_replay_preview() {
        let fixture = r#"{
            "cwd":"/workspace",
            "query":"needle",
            "candidates":[],
            "selected_targets":[],
            "nearest_skipped_manifest_candidates":[],
            "retrieval_empty_targets":[],
            "would_inject":false,
            "skip_reason":"legacy"
        }"#;
        let actual: WorkspaceContextExplanation = serde_json::from_str(fixture).unwrap();
        let expected = (
            Vec::<WorkspaceEvidenceReplayPreviewDiagnostic>::new(),
            Vec::<WorkspaceRetrievalPlanDiagnostic>::new(),
        );

        assert_eq!(
            (
                actual.replay_preview_diagnostics,
                actual.retrieval_plan_diagnostics,
            ),
            expected,
        );
    }
}

/// Redaction-safe diagnostic-only workspace evidence replay status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkspaceEvidenceReplayStatus {
    /// Project-model manifest was not found at the expected workspace path.
    ManifestMissing,
    /// Manifest exists but is stale relative to the current filesystem view.
    ManifestStale,
    /// Manifest exists but freshness could not be proven.
    ManifestUnknown,
    /// Manifest is injectable, replay ran, and no selected references or issues were found.
    ReplayedEmpty,
    /// Manifest is injectable and replay selected bounded reference-only evidence.
    ReplayedWithSelection,
    /// Manifest is injectable and replay found typed issues.
    ReplayedWithIssues,
}

impl WorkspaceEvidenceReplayStatus {
    /// Returns the stable status label used by diagnostics.
    pub fn label(&self) -> &'static str {
        match self {
            Self::ManifestMissing => "not_replayed_manifest_missing",
            Self::ManifestStale => "not_replayed_manifest_stale",
            Self::ManifestUnknown => "not_replayed_manifest_unknown",
            Self::ReplayedEmpty => "replayed_empty",
            Self::ReplayedWithSelection => "replayed_with_selection",
            Self::ReplayedWithIssues => "replayed_with_issues",
        }
    }
}

/// Redaction-safe read-only replay-preview status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkspaceEvidenceReplayPreviewStatus {
    /// Project-model manifest was not found, so preview did not inspect replay artifacts.
    NotPreviewedManifestMissing,
    /// Project-model manifest is stale, so preview did not inspect replay artifacts.
    NotPreviewedManifestStale,
    /// Project-model manifest freshness could not be proven, so preview did not inspect replay artifacts.
    NotPreviewedManifestUnknown,
    /// Replay ran but selected no previewable references and emitted no visible issues.
    NotPreviewedEmptyReplay,
    /// Replay preview rendered selected metadata-only references.
    PreviewedWithSelection,
    /// Replay preview rendered metadata-only issue evidence.
    PreviewedWithIssues,
    /// Replay preview adapter refused the selected report as invalid for the current manifest.
    PreviewRefused,
    /// Replay preview was produced, but replay selection reported budget truncation.
    PreviewTruncated,
    /// Replay preview rendering exceeded the render budget and fell back to omission metadata.
    PreviewBudgetExceeded,
}

impl WorkspaceEvidenceReplayPreviewStatus {
    /// Returns the stable status label used by read-only preview diagnostics.
    pub fn label(&self) -> &'static str {
        match self {
            Self::NotPreviewedManifestMissing => "not_previewed_manifest_missing",
            Self::NotPreviewedManifestStale => "not_previewed_manifest_stale",
            Self::NotPreviewedManifestUnknown => "not_previewed_manifest_unknown",
            Self::NotPreviewedEmptyReplay => "not_previewed_empty_replay",
            Self::PreviewedWithSelection => "previewed_with_selection",
            Self::PreviewedWithIssues => "previewed_with_issues",
            Self::PreviewRefused => "preview_refused",
            Self::PreviewTruncated => "preview_truncated",
            Self::PreviewBudgetExceeded => "preview_budget_exceeded",
        }
    }
}

/// Redaction-safe rendered evidence replay preview for diagnostics only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkspaceEvidenceReplayPreviewDiagnostic {
    /// Stable preview status.
    pub status: WorkspaceEvidenceReplayPreviewStatus,
    /// Stable redaction-safe label for the canonical workspace root.
    pub workspace_root_label: String,
    /// Stable redaction-safe label for the project-model manifest.
    pub manifest_label: String,
    /// Whether the manifest file exists at the expected path.
    pub manifest_found: bool,
    /// Manifest freshness classification label.
    pub manifest_freshness: String,
    /// Redaction-safe skip or refusal reason.
    pub not_previewed_reason: Option<String>,
    /// Manifest hash used by replay after freshness was proven.
    pub manifest_hash: Option<String>,
    /// Content policy label; always reference-only when replay runs.
    pub content_policy: Option<String>,
    /// Stale policy label when replay runs.
    pub stale_policy: Option<String>,
    /// Changed evidence references excluded by policy.
    pub changed_excluded: usize,
    /// Deleted evidence references excluded by policy.
    pub deleted_excluded: usize,
    /// Deterministic budget and selection audit when replay runs.
    pub budget: Option<WorkspaceEvidenceReplayBudgetSummary>,
    /// Reference-only selected evidence items.
    pub selected: Vec<WorkspaceEvidenceReplayReference>,
    /// Typed redaction-safe issue summaries.
    pub issues: Vec<WorkspaceEvidenceReplayIssueSummary>,
    /// Rendered metadata-only preview using the canonical project-model context renderer.
    pub rendered_preview: Option<String>,
}

/// Reference-only selected evidence item for workspace replay diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkspaceEvidenceReplayReference {
    /// Context-pack artifact identifier.
    pub artifact_id: String,
    /// Context-pack artifact path under the model metadata root.
    pub artifact_path: String,
    /// Evidence identifier inside the artifact.
    pub evidence_id: String,
    /// Evidence source path relative to the project root.
    pub evidence_path: String,
    /// Optional one-based inclusive start line.
    pub start_line: Option<u32>,
    /// Optional one-based inclusive end line.
    pub end_line: Option<u32>,
    /// Stable score class label.
    pub score_kind: String,
    /// Redaction-safe numeric priority within the score class.
    pub score: f32,
    /// Evidence provenance path.
    pub provenance_path: String,
    /// Optional provenance one-based inclusive start line.
    pub provenance_start_line: Option<u32>,
    /// Optional provenance one-based inclusive end line.
    pub provenance_end_line: Option<u32>,
    /// Evidence provenance source label.
    pub provenance_source: String,
    /// Redaction-safe provenance fingerprint.
    pub provenance_fingerprint: String,
    /// Evidence freshness label.
    pub freshness: String,
    /// Number of valid tool episodes linking this artifact.
    pub linked_episode_count: usize,
    /// Number of dangling or invalid linkage proofs seen for this artifact.
    pub link_issue_count: usize,
}

/// Redaction-safe typed issue emitted by workspace evidence replay diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceEvidenceReplayIssueSummary {
    /// Machine-readable issue code.
    pub code: String,
    /// Optional context-pack artifact identifier.
    pub artifact_id: Option<String>,
    /// Optional evidence identifier.
    pub evidence_id: Option<String>,
    /// Optional tool-episode fingerprint.
    pub episode_fingerprint: Option<String>,
    /// Redaction-safe path or storage label.
    pub path: Option<String>,
}

/// Deterministic budget summary for workspace evidence replay diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceEvidenceReplayBudgetSummary {
    /// Number of artifact candidates before inspection budget truncation.
    pub original_candidate_count: usize,
    /// Number of selected reference-only evidence items.
    pub selected_count: usize,
    /// Total excluded candidate/evidence/link count.
    pub excluded_count: usize,
    /// Excluded counts keyed by typed issue code label.
    pub excluded_by_reason: BTreeMap<String, usize>,
    /// Whether any artifact, episode, or selected output was truncated by budget.
    pub truncated: bool,
    /// Maximum artifact candidates inspected.
    pub max_artifacts: usize,
    /// Maximum episode lines inspected.
    pub max_episode_lines: usize,
    /// Maximum selected references returned.
    pub max_selected: usize,
    /// Stable ordering and tie-break contract used by replay.
    pub stable_ordering: String,
}

/// Redaction-safe diagnostic-only workspace evidence replay surface.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkspaceEvidenceReplayDiagnostic {
    /// Stable replay status.
    pub status: WorkspaceEvidenceReplayStatus,
    /// Canonical workspace root evaluated for replay.
    pub workspace_root: PathBuf,
    /// Expected local project-model manifest path.
    pub manifest_path: PathBuf,
    /// Whether the manifest file exists at the expected path.
    pub manifest_found: bool,
    /// Manifest freshness classification label.
    pub manifest_freshness: String,
    /// Redaction-safe freshness issue or skip reason when replay is not allowed.
    pub not_replayed_reason: Option<String>,
    /// Manifest hash used by replay after freshness was proven.
    pub manifest_hash: Option<String>,
    /// Content policy label; always reference-only when replay runs.
    pub content_policy: Option<String>,
    /// Stale policy label when replay runs.
    pub stale_policy: Option<String>,
    /// Changed evidence references excluded by policy.
    pub changed_excluded: usize,
    /// Deleted evidence references excluded by policy.
    pub deleted_excluded: usize,
    /// Deterministic budget and selection audit when replay runs.
    pub budget: Option<WorkspaceEvidenceReplayBudgetSummary>,
    /// Reference-only selected evidence items.
    pub selected: Vec<WorkspaceEvidenceReplayReference>,
    /// Typed redaction-safe issue summaries.
    pub issues: Vec<WorkspaceEvidenceReplayIssueSummary>,
}

/// Redaction-safe transport report for read-only exact-fact workspace status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceExactFactStatusReport {
    /// Stable status label.
    pub status: String,
    /// Canonical manifest path.
    pub manifest_path: PathBuf,
    /// Persisted manifest hash when the manifest is readable.
    pub manifest_hash: Option<String>,
    /// Manifest freshness or proof-level label.
    pub manifest_freshness_proof_level: Option<String>,
    /// Canonical persisted ingestion-report path.
    pub ingestion_report_path: PathBuf,
    /// Read-only artifact-store metadata state label.
    pub artifact_store_state: String,
    /// Count of artifact candidates inspected by the persisted ingestion report.
    pub inspected_artifact_count: usize,
    /// Count of accepted artifacts in the persisted ingestion report.
    pub accepted_artifact_count: usize,
    /// Accepted batch fingerprints in deterministic persisted report order.
    pub accepted_batch_fingerprints: Vec<String>,
    /// Accepted external fact batch count persisted in the manifest.
    pub manifest_external_fact_batch_count: usize,
    /// Manifest external facts fingerprint when the manifest is readable.
    pub manifest_external_facts_fingerprint: Option<String>,
    /// Graph-visible reference edge count.
    pub reference_edge_count: usize,
    /// Graph-visible exact compiler reference edge count.
    pub exact_compiler_reference_edge_count: usize,
    /// Redaction-safe issue count.
    pub issue_count: usize,
    /// Redaction-safe status and ingestion issue summaries.
    pub issue_summaries: Vec<String>,
    /// Whether exact facts are active.
    pub exact_facts_active: bool,
}

/// Stable status for explicit workspace exact-fact reference production.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkspaceExactFactReferenceStatus {
    /// A typed external fact artifact was written.
    ArtifactWritten,
    /// No eligible manifest-owned endpoint was available for one bounded request.
    NoEligibleEndpoint,
    /// rust-analyzer was unavailable or failed its capability probe.
    RustAnalyzerUnavailable,
    /// The capability probe or production request timed out.
    Timeout,
    /// The producer ran successfully but returned no reference facts.
    NoFacts,
    /// Request validation or typed normalization failed.
    Failed,
}

impl WorkspaceExactFactReferenceStatus {
    /// Returns the stable lowercase status label used by human output.
    pub fn label(&self) -> &'static str {
        match self {
            Self::ArtifactWritten => "artifact_written",
            Self::NoEligibleEndpoint => "no_eligible_endpoint",
            Self::RustAnalyzerUnavailable => "rust_analyzer_unavailable",
            Self::Timeout => "timeout",
            Self::NoFacts => "no_facts",
            Self::Failed => "failed",
        }
    }
}

/// Bounded-loss summary for a single explicit native LSP reference request.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceExactFactBoundedLoss {
    /// Eligible endpoint positions omitted because request bounds were reached.
    pub omitted_endpoint_positions: usize,
    /// Manifest-owned source files omitted from didOpen because request bounds were reached.
    pub omitted_open_files: usize,
}

/// Redaction-safe issue emitted by exact-fact reference production or ingestion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceExactFactIssue {
    /// Stable issue code.
    pub code: String,
    /// Optional typed endpoint involved in the issue.
    pub endpoint: Option<String>,
    /// Redaction-safe detail without raw source, JSON-RPC, stdout, or stderr.
    pub detail: String,
}

/// Compact ingestion summary after refreshing the manifest from typed artifacts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceExactFactIngestionSummary {
    /// Number of external fact artifacts inspected.
    pub inspected_artifacts: usize,
    /// Number of external fact artifacts accepted.
    pub accepted_artifacts: usize,
    /// Accepted batch fingerprints in deterministic ingestion order.
    pub accepted_batch_fingerprints: Vec<String>,
    /// Number of ingestion issues surfaced across artifacts.
    pub issue_count: usize,
}

/// Redaction-safe command report for explicit workspace exact-fact references.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceExactFactReferenceReport {
    /// Explicit command status.
    pub status: WorkspaceExactFactReferenceStatus,
    /// Persisted artifact path when exactly one typed artifact was written.
    pub artifact_path: Option<PathBuf>,
    /// Batch fingerprint when an artifact was written and accepted by validation.
    pub batch_fingerprint: Option<String>,
    /// Number of produced typed reference facts.
    pub produced_reference_count: usize,
    /// Bounded loss marker for the single native LSP request.
    pub bounded_loss: WorkspaceExactFactBoundedLoss,
    /// Manifest hash used as the frozen production baseline.
    pub manifest_hash_input: String,
    /// Redaction-safe command and validation issues.
    pub issues: Vec<WorkspaceExactFactIssue>,
    /// Summary of the manifest refresh ingestion pass.
    pub ingestion_summary: WorkspaceExactFactIngestionSummary,
    /// Path of the refreshed project manifest written after re-ingestion.
    pub manifest_path: PathBuf,
    /// Path of the refreshed external fact ingestion report.
    pub ingestion_report_path: PathBuf,
}
