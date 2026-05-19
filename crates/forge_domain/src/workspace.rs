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

/// User-facing explanation for automatic project-model context injection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    /// explained query.
    pub retrieval_empty_targets: Vec<PathBuf>,
    /// Whether automatic project-model context would be injected.
    pub would_inject: bool,
    /// Exact top-level reason context would not be injected.
    pub skip_reason: Option<String>,
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
