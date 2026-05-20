use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use derive_more::Display;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Typed manifest-derived input for an external project semantic embedding boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectSemanticEmbeddingInput {
    /// Manifest-owned source identifier that must be echoed by the boundary.
    pub source_id: String,
    /// Manifest-owned source fingerprint that must be echoed by the boundary.
    pub source_fingerprint: String,
    /// Bounded text derived from project-model evidence for embedding.
    pub text: String,
}

/// Provider-neutral semantic embedding request for project-model vector indexing or query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectSemanticEmbeddingRequest {
    /// External embedding model identity chosen by the caller.
    pub embedding_model_id: String,
    /// Ordered bounded embedding inputs.
    pub inputs: Vec<ProjectSemanticEmbeddingInput>,
}

/// Provider-neutral semantic vector returned by an embedding boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectSemanticEmbeddingVector {
    /// Manifest-owned source identifier echoed from the request.
    pub source_id: String,
    /// Manifest-owned source fingerprint echoed from the request.
    pub source_fingerprint: String,
    /// Provider-neutral embedding values.
    pub embedding: Vec<f32>,
}

/// Provider-neutral ordered semantic embedding output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectSemanticEmbeddingOutput {
    /// External embedding model identity echoed by the boundary.
    pub embedding_model_id: String,
    /// Fixed embedding dimension for every vector.
    pub dimension: usize,
    /// Ordered provider-neutral vectors matching the request inputs.
    pub vectors: Vec<ProjectSemanticEmbeddingVector>,
}

/// Stable unsupported reason for direct `sem_search` availability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SemSearchUnsupportedReason {
    /// Semantic embedding model configuration is absent or blank.
    NoModelConfig,
    /// Project-model manifest is absent for the workspace root.
    ManifestMissing,
    /// No durable vector artifact matches the fresh manifest and configured model.
    VectorArtifactAbsentOrNoMatch,
}

impl SemSearchUnsupportedReason {
    /// Returns the stable cache/user diagnostic label for this reason.
    pub fn label(&self) -> &'static str {
        match self {
            Self::NoModelConfig => "no_model_config",
            Self::ManifestMissing => "manifest_missing",
            Self::VectorArtifactAbsentOrNoMatch => "vector_artifact_absent_or_no_match",
        }
    }
}

/// Stable unknown reason for direct `sem_search` availability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SemSearchUnknownReason {
    /// Workspace root could not be canonicalized or inspected.
    WorkspaceProbeFailed,
    /// Project-model manifest exists but could not be read as typed data.
    ManifestUnreadable,
    /// Project-model manifest exists but is stale relative to the filesystem.
    StaleManifest,
    /// Project-model manifest freshness could not be proven.
    ManifestFreshnessUnknown,
    /// Durable vector artifact listing failed.
    VectorArtifactListingFailed,
    /// Durable vector artifact is unreadable, corrupt, or structurally not ready.
    VectorArtifactCorruptOrNotReady,
    /// Multiple durable vector artifacts match the same fresh manifest and model.
    AmbiguousVectorArtifact,
    /// Read-only availability adapter received an unexpected legacy readiness state.
    UnknownProbeFailure,
}

impl SemSearchUnknownReason {
    /// Returns the stable cache/user diagnostic label for this reason.
    pub fn label(&self) -> &'static str {
        match self {
            Self::WorkspaceProbeFailed => "workspace_probe_failed",
            Self::ManifestUnreadable => "manifest_unreadable",
            Self::StaleManifest => "stale_manifest",
            Self::ManifestFreshnessUnknown => "manifest_freshness_unknown",
            Self::VectorArtifactListingFailed => "vector_artifact_listing_failed",
            Self::VectorArtifactCorruptOrNotReady => "vector_artifact_corrupt_or_not_ready",
            Self::AmbiguousVectorArtifact => "ambiguous_vector_artifact",
            Self::UnknownProbeFailure => "unknown_probe_failure",
        }
    }
}

/// Typed read-only readiness classification for direct `sem_search` availability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SemSearchAvailability {
    /// A fresh manifest and exactly one matching durable vector artifact are ready.
    Ready {
        /// Canonical workspace root used for semantic retrieval.
        workspace_root: PathBuf,
        /// Fresh project-model manifest hash.
        manifest_hash: String,
        /// Deterministic durable vector artifact identifier.
        vector_artifact_id: String,
        /// Fixed vector dimension required by the selected artifact.
        dimension: usize,
    },
    /// Semantic search is intentionally unavailable and should normally not be advertised.
    Unsupported {
        /// Closed stable reason class.
        reason: SemSearchUnsupportedReason,
    },
    /// Readiness could not be proven; direct calls must fail during typed preflight.
    Unknown {
        /// Closed stable reason class.
        reason: SemSearchUnknownReason,
    },
}

impl SemSearchAvailability {
    /// Returns true when the tool may be advertised to expose a direct-call diagnostic path.
    pub fn should_advertise(&self) -> bool {
        matches!(self, Self::Ready { .. } | Self::Unknown { .. })
    }

    /// Returns the stable reason label for this availability classification.
    pub fn reason_label(&self) -> &'static str {
        match self {
            Self::Ready { .. } => "ready",
            Self::Unsupported { reason } => reason.label(),
            Self::Unknown { reason } => reason.label(),
        }
    }

    /// Returns the minimal stable fingerprint for tool-definition cache invalidation.
    pub fn semantic_fingerprint(&self) -> String {
        match self {
            Self::Ready { manifest_hash, vector_artifact_id, dimension, .. } => {
                format!("ready:{manifest_hash}:{vector_artifact_id}:{dimension}")
            }
            Self::Unsupported { reason } => format!("unsupported:{}", reason.label()),
            Self::Unknown { reason } => format!("unknown:{}", reason.label()),
        }
    }

    /// Converts non-ready states into a typed preflight error.
    ///
    /// # Errors
    /// Returns an error when semantic search is unsupported or unknown.
    pub fn ensure_ready(&self) -> anyhow::Result<()> {
        match self {
            Self::Ready { .. } => Ok(()),
            Self::Unsupported { reason } => {
                anyhow::bail!("sem_search unavailable: unsupported: {}", reason.label())
            }
            Self::Unknown { reason } => {
                anyhow::bail!("sem_search unavailable: unknown: {}", reason.label())
            }
        }
    }
}

/// Stable status for the read-only `sem_search` build/update diagnostic report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemSearchDiagnosticStatus {
    /// A fresh manifest and exactly one matching durable vector artifact are ready.
    Ready,
    /// Semantic embedding model configuration is absent or blank.
    ConfigRequired,
    /// Project-model manifest is absent for the workspace root.
    ManifestRequired,
    /// Project-model manifest exists but must be refreshed before vector work is suggested.
    ManifestRefreshRequired,
    /// A fresh manifest exists and a vector build is the safe next action.
    VectorBuildSuggested,
    /// Vector artifacts must be inspected or repaired before retrying.
    VectorArtifactRepairRequired,
    /// Readiness could not be classified into a safe specific build or repair action.
    ProbeUnknown,
}

/// Closed suggested action for the read-only `sem_search` diagnostic report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemSearchSuggestedAction {
    /// No follow-up command is suggested.
    None,
    /// Configure `semantic_embedding_model_id` before semantic search can be used.
    ConfigureEmbeddingModel,
    /// Build or refresh the project-model manifest before vector work.
    RefreshManifest,
    /// Build the durable vector artifact for the configured embedding model.
    BuildVectorIndex,
    /// Inspect or repair existing vector artifacts without destructive cleanup.
    RepairVectorArtifact,
    /// Run a read-only probe or inspect diagnostics before choosing a mutation.
    ProbeReadiness,
}

/// Redaction-safe embedding model identity in a `sem_search` diagnostic report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemSearchEmbeddingModelDiagnostic {
    /// Configured embedding model identifier when present and non-blank.
    pub configured_model_id: Option<String>,
}

/// Optional typed manifest identity exposed only when a probe already produced it safely.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemSearchManifestIdentity {
    /// Workspace root identity associated with the manifest.
    pub workspace_root: PathBuf,
    /// Fresh project-model manifest hash.
    pub manifest_hash: String,
}

/// Optional typed vector identity exposed only when the vector artifact is ready.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemSearchVectorIdentity {
    /// Deterministic durable vector artifact identifier.
    pub vector_artifact_id: String,
    /// Fixed vector dimension required by the selected artifact.
    pub dimension: usize,
}

/// Structured command suggestion with lossless argv and safe display text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemSearchDiagnosticCommand {
    /// Command argv without shell parsing.
    pub argv: Vec<String>,
    /// Display-only shell command string with conservative quoting.
    pub display: String,
}

impl SemSearchDiagnosticCommand {
    /// Creates a structured command and a display-only safely quoted representation.
    pub fn new(argv: Vec<String>) -> Self {
        let display = argv
            .iter()
            .map(|arg| shell_quote_display(arg))
            .collect::<Vec<_>>()
            .join(" ");
        Self { argv, display }
    }
}

/// Read-only `sem_search` build/update diagnostic report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemSearchDiagnosticReport {
    /// Closed diagnostic status.
    pub status: SemSearchDiagnosticStatus,
    /// Stable reason label from the shared availability classification source.
    pub reason_label: String,
    /// Configured embedding model identity without raw provider/config dumps.
    pub embedding_model: SemSearchEmbeddingModelDiagnostic,
    /// Manifest identity when safely available from the existing typed probe.
    pub manifest_identity: Option<SemSearchManifestIdentity>,
    /// Vector identity only when the vector artifact is ready.
    pub vector_identity: Option<SemSearchVectorIdentity>,
    /// Closed suggested next action.
    pub suggested_action: SemSearchSuggestedAction,
    /// Whether suggesting a vector build is safe for this classification.
    pub safe_to_suggest_build: bool,
    /// Optional structured command for the suggested action.
    pub command: Option<SemSearchDiagnosticCommand>,
}

impl SemSearchDiagnosticReport {
    /// Builds a read-only diagnostic report from the shared `sem_search` availability probe.
    pub fn from_availability(
        availability: &SemSearchAvailability,
        embedding_model_id: Option<&str>,
        workspace_path: &Path,
    ) -> Self {
        let configured_model_id = embedding_model_id.and_then(|model_id| {
            let trimmed = model_id.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        });
        let mut report = match availability {
            SemSearchAvailability::Ready {
                workspace_root,
                manifest_hash,
                vector_artifact_id,
                dimension,
            } => Self {
                status: SemSearchDiagnosticStatus::Ready,
                reason_label: availability.reason_label().to_string(),
                embedding_model: SemSearchEmbeddingModelDiagnostic { configured_model_id },
                manifest_identity: Some(SemSearchManifestIdentity {
                    workspace_root: workspace_root.clone(),
                    manifest_hash: manifest_hash.clone(),
                }),
                vector_identity: Some(SemSearchVectorIdentity {
                    vector_artifact_id: vector_artifact_id.clone(),
                    dimension: *dimension,
                }),
                suggested_action: SemSearchSuggestedAction::None,
                safe_to_suggest_build: false,
                command: None,
            },
            SemSearchAvailability::Unsupported { reason } => {
                let (status, suggested_action, safe_to_suggest_build) = match reason {
                    SemSearchUnsupportedReason::NoModelConfig => (
                        SemSearchDiagnosticStatus::ConfigRequired,
                        SemSearchSuggestedAction::ConfigureEmbeddingModel,
                        false,
                    ),
                    SemSearchUnsupportedReason::ManifestMissing => (
                        SemSearchDiagnosticStatus::ManifestRequired,
                        SemSearchSuggestedAction::RefreshManifest,
                        false,
                    ),
                    SemSearchUnsupportedReason::VectorArtifactAbsentOrNoMatch => (
                        SemSearchDiagnosticStatus::VectorBuildSuggested,
                        SemSearchSuggestedAction::BuildVectorIndex,
                        true,
                    ),
                };
                Self {
                    status,
                    reason_label: reason.label().to_string(),
                    embedding_model: SemSearchEmbeddingModelDiagnostic { configured_model_id },
                    manifest_identity: None,
                    vector_identity: None,
                    suggested_action,
                    safe_to_suggest_build,
                    command: None,
                }
            }
            SemSearchAvailability::Unknown { reason } => {
                let (status, suggested_action) = match reason {
                    SemSearchUnknownReason::StaleManifest => (
                        SemSearchDiagnosticStatus::ManifestRefreshRequired,
                        SemSearchSuggestedAction::RefreshManifest,
                    ),
                    SemSearchUnknownReason::ManifestUnreadable => (
                        SemSearchDiagnosticStatus::ManifestRefreshRequired,
                        SemSearchSuggestedAction::RefreshManifest,
                    ),
                    SemSearchUnknownReason::VectorArtifactListingFailed
                    | SemSearchUnknownReason::VectorArtifactCorruptOrNotReady => (
                        SemSearchDiagnosticStatus::VectorArtifactRepairRequired,
                        SemSearchSuggestedAction::RepairVectorArtifact,
                    ),
                    SemSearchUnknownReason::WorkspaceProbeFailed
                    | SemSearchUnknownReason::ManifestFreshnessUnknown
                    | SemSearchUnknownReason::AmbiguousVectorArtifact
                    | SemSearchUnknownReason::UnknownProbeFailure => (
                        SemSearchDiagnosticStatus::ProbeUnknown,
                        SemSearchSuggestedAction::ProbeReadiness,
                    ),
                };
                Self {
                    status,
                    reason_label: reason.label().to_string(),
                    embedding_model: SemSearchEmbeddingModelDiagnostic { configured_model_id },
                    manifest_identity: None,
                    vector_identity: None,
                    suggested_action,
                    safe_to_suggest_build: false,
                    command: None,
                }
            }
        };

        if report.safe_to_suggest_build
            && let Some(model_id) = report.embedding_model.configured_model_id.clone()
        {
            report.command = Some(SemSearchDiagnosticCommand::new(vec![
                "forge".to_string(),
                "workspace".to_string(),
                "vector-index".to_string(),
                "build".to_string(),
                "--embedding-model-id".to_string(),
                model_id,
                workspace_path.display().to_string(),
            ]));
        }
        report
    }
}

fn shell_quote_display(value: &str) -> String {
    if value.is_empty() {
        return "''".to_string();
    }
    if value.chars().all(|character| {
        character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.' | '/' | ':' | '=')
    }) {
        return value.to_string();
    }
    format!("'{}'", value.replace('\'', "'\\''"))
}

/// Typed readiness classification for automatic semantic project-context injection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkspaceSemanticInjectionReadiness {
    /// Semantic injection is disabled because no embedding model is configured.
    SemanticDisabledNoModelConfig,
    /// No durable vector index matches the current manifest and embedding model.
    VectorIndexAbsentOrNoMatch,
    /// Exactly one durable vector index matches the current manifest and embedding model.
    VectorIndexReady {
        /// Fixed vector dimension expected by the durable index.
        dimension: usize,
    },
    /// More than one durable vector index matches, so no random artifact may be selected.
    VectorIndexAmbiguous,
    /// Vector artifact state is corrupt or structurally not ready.
    VectorIndexCorruptOrNotReady,
    /// Query embedding dimension is incompatible with the selected durable vector index.
    VectorDimensionMismatch {
        /// Durable index dimension.
        expected: usize,
        /// Query vector dimension.
        actual: usize,
    },
    /// Embedding provider was unavailable during the bounded semantic attempt.
    EmbeddingProviderUnavailable,
    /// Embedding provider timed out during the bounded semantic attempt.
    EmbeddingProviderTimeout,
}

/// Stable status for explicit workspace vector-index production.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkspaceVectorIndexBuildStatus {
    /// Exactly one durable vector artifact was written and validated by readback.
    ArtifactWritten,
}

impl WorkspaceVectorIndexBuildStatus {
    /// Returns the stable lowercase status label used by human output.
    pub fn label(&self) -> &'static str {
        match self {
            Self::ArtifactWritten => "artifact_written",
        }
    }
}

/// Redaction-safe command report for explicit workspace vector-index production.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceVectorIndexBuildReport {
    /// Explicit command status.
    pub status: WorkspaceVectorIndexBuildStatus,
    /// Persisted artifact path.
    pub artifact_path: PathBuf,
    /// Deterministic vector artifact identifier.
    pub artifact_id: String,
    /// Embedding model identity used for the artifact.
    pub embedding_model_id: String,
    /// Fixed embedding dimension.
    pub dimension: usize,
    /// Number of persisted vector entries.
    pub entry_count: usize,
    /// Manifest hash used as the source baseline.
    pub manifest_hash: String,
}

/// Closed status returned by the explicit agent-invoked vector build continuation tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceVectorIndexBuildContinuationStatus {
    /// A build was performed and the post-build diagnostic proved sem_search readiness.
    BuiltReady,
    /// No build was performed because embedding model configuration is required.
    NotBuiltConfigRequired,
    /// No build was performed because a project-model manifest is required.
    NotBuiltManifestRequired,
    /// No build was performed because the manifest must be refreshed first.
    NotBuiltManifestRefreshRequired,
    /// No build was performed because vector artifacts require repair or inspection.
    NotBuiltRepairRequired,
    /// No build was performed because readiness probing is unknown or ambiguous.
    NotBuiltProbeUnknown,
    /// A build was attempted through the typed service boundary and failed.
    BuildFailed,
}

/// Closed status returned by the explicit agent-invoked exact-fact continuation tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceExactFactReferenceContinuationStatus {
    /// Preflight status already proved exact facts are active; no producer was called.
    AlreadyActive,
    /// The producer was called once and postflight status proved exact facts are active.
    ProducedActive,
    /// No producer call was made because the project-model manifest is missing.
    NotProducedManifestMissing,
    /// No producer call was made because the manifest is stale.
    NotProducedManifestStale,
    /// No producer call was made or trusted because status could not be read safely.
    NotProducedStatusUnreadable,
    /// The producer was called once and failed through the service boundary.
    ProducerFailed,
    /// The producer returned success but postflight status did not prove active exact facts.
    ProducedButInactive,
    /// No producer call was made or no active facts were expected for this safe terminal state.
    NotProducedNoEligibleProductionState,
}

/// Typed output for the explicit agent-invoked exact-fact reference continuation tool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceExactFactReferenceContinuationReport {
    /// Status captured immediately before any possible producer mutation.
    pub preflight_status: Option<WorkspaceExactFactStatusReport>,
    /// Service-owned producer report when a producer call was safely attempted and succeeded.
    pub producer_report: Option<WorkspaceExactFactReferenceReport>,
    /// Redaction-safe marker that the producer failed without exposing raw errors.
    pub producer_failed: bool,
    /// Status captured after no-production classification or attempted production.
    pub postflight_status: Option<WorkspaceExactFactStatusReport>,
    /// Redaction-safe diagnostic when preflight or postflight status was unreadable.
    pub status_unreadable_diagnostic: Option<String>,
    /// Closed final status for agent control flow.
    pub final_status: WorkspaceExactFactReferenceContinuationStatus,
}

impl WorkspaceVectorIndexBuildContinuationStatus {
    /// Maps non-build-safe diagnostic statuses into closed continuation statuses.
    pub fn from_non_build_diagnostic_status(status: SemSearchDiagnosticStatus) -> Self {
        match status {
            SemSearchDiagnosticStatus::Ready => Self::NotBuiltProbeUnknown,
            SemSearchDiagnosticStatus::ConfigRequired => Self::NotBuiltConfigRequired,
            SemSearchDiagnosticStatus::ManifestRequired => Self::NotBuiltManifestRequired,
            SemSearchDiagnosticStatus::ManifestRefreshRequired => {
                Self::NotBuiltManifestRefreshRequired
            }
            SemSearchDiagnosticStatus::VectorBuildSuggested => Self::NotBuiltProbeUnknown,
            SemSearchDiagnosticStatus::VectorArtifactRepairRequired => Self::NotBuiltRepairRequired,
            SemSearchDiagnosticStatus::ProbeUnknown => Self::NotBuiltProbeUnknown,
        }
    }
}

/// Typed output for the explicit agent-invoked vector build continuation tool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceVectorIndexBuildContinuationReport {
    /// Diagnostic captured immediately before any possible mutation.
    pub preflight_diagnostic: SemSearchDiagnosticReport,
    /// Service-owned build report when a build was safely attempted and succeeded.
    pub build_report: Option<WorkspaceVectorIndexBuildReport>,
    /// Diagnostic captured after the no-build classification or attempted build.
    pub post_build_diagnostic: SemSearchDiagnosticReport,
    /// Closed final status for agent control flow.
    pub final_status: WorkspaceVectorIndexBuildContinuationStatus,
}

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
    /// Fresh manifest hash when freshness is proven.
    pub manifest_hash: Option<String>,
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

/// Typed semantic readiness diagnostic for read-only workspace context explanations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceSemanticReadinessDiagnostic {
    /// Stable redaction-safe label for the workspace root being evaluated.
    pub workspace_root_label: String,
    /// Whether local semantic readiness was evaluated for this target.
    pub evaluated: bool,
    /// Semantic readiness status when evaluation completed.
    pub status: WorkspaceSemanticReadinessStatus,
    /// Fixed vector dimension when a current durable vector artifact is usable.
    pub dimension: Option<usize>,
    /// Redaction-safe reason when readiness was intentionally not evaluated.
    pub not_evaluated_reason: Option<String>,
}

/// Stable semantic readiness status for read-only workspace context explanations.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkspaceSemanticReadinessStatus {
    /// Semantic retrieval is disabled because no embedding model is configured.
    SemanticDisabledNoModelConfig,
    /// No durable vector index matches the current manifest and embedding model.
    VectorIndexAbsentOrNoMatch,
    /// Exactly one current durable vector artifact is usable.
    VectorIndexReady,
    /// More than one current durable vector artifact matched.
    VectorIndexAmbiguous,
    /// Durable vector artifact state is corrupt, unreadable, or structurally not ready.
    VectorIndexCorruptOrNotReady,
    /// Query/runtime vector dimension does not match the durable index.
    VectorDimensionMismatch,
    /// Embedding provider is unavailable; included only when a runtime semantic attempt reported it.
    EmbeddingProviderUnavailable,
    /// Embedding provider timed out; included only when a runtime semantic attempt reported it.
    EmbeddingProviderTimeout,
    /// Target was not evaluated by the read-only diagnostic bridge.
    #[default]
    NotEvaluated,
}

/// Typed redaction-safe phase diagnostics for explain-context retrieval planning.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceRetrievalPhaseDiagnostics {
    /// Lexical retrieval phase status.
    pub lexical: WorkspaceRetrievalPhaseStatus,
    /// Graph retrieval phase status.
    pub graph: WorkspaceRetrievalPhaseStatus,
    /// Vector retrieval phase status.
    pub vector: WorkspaceRetrievalPhaseStatus,
    /// Rerank phase status.
    pub rerank: WorkspaceRetrievalPhaseStatus,
}

/// Stable phase status for explain-context retrieval planning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkspaceRetrievalPhaseStatus {
    /// Phase was active in the read-only plan projection.
    Active {
        /// Number of selected results carrying this phase score.
        result_count: usize,
    },
    /// Phase was intentionally skipped by request shape.
    Skipped {
        /// Stable skip reason.
        reason: WorkspaceRetrievalPhaseSkipReason,
    },
    /// Phase could not run because runtime input or boundary was missing/not ready.
    Unavailable {
        /// Stable unavailable reason.
        reason: WorkspaceRetrievalPhaseUnavailableReason,
    },
    /// Phase input was present but invalid.
    Invalid {
        /// Stable invalid reason.
        reason: WorkspaceRetrievalPhaseInvalidReason,
    },
}

impl Default for WorkspaceRetrievalPhaseStatus {
    fn default() -> Self {
        Self::Skipped { reason: WorkspaceRetrievalPhaseSkipReason::EmptyQueryText }
    }
}

/// Stable skip reason for explain-context retrieval planning phases.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkspaceRetrievalPhaseSkipReason {
    /// Query text is empty.
    #[default]
    EmptyQueryText,
    /// Graph expansion was not requested.
    GraphExpansionDisabled,
}

/// Stable unavailable reason for explain-context retrieval planning phases.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkspaceRetrievalPhaseUnavailableReason {
    /// Vector query embedding was not supplied because explain-context is read-only.
    MissingQueryEmbedding,
    /// Vector index boundary was not supplied.
    MissingVectorIndex,
    /// Vector index boundary reported not-ready status.
    VectorIndexNotReady,
    /// Reranker runtime boundary was not supplied.
    MissingReranker,
    /// Reranker runtime boundary reported not-ready status.
    RerankerNotReady,
    /// No durable vector index matched.
    NoMatchingVectorIndex,
    /// More than one durable vector index matched.
    AmbiguousVectorIndex,
}

/// Stable invalid reason for explain-context retrieval planning phases.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkspaceRetrievalPhaseInvalidReason {
    /// Query vector dimension differs from index dimension.
    VectorDimensionMismatch {
        /// Query embedding dimension.
        query_dimension: usize,
        /// Durable index dimension.
        index_dimension: usize,
    },
    /// Ready vector index reported zero dimensions.
    VectorIndexZeroDimension,
    /// Query embedding was empty.
    EmptyQueryEmbedding,
    /// Query embedding contains a non-finite value.
    NonFiniteQueryEmbedding,
    /// Query embedding has zero norm.
    ZeroQueryEmbeddingNorm,
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
    /// Typed phase diagnostics derived from the read-only planner only.
    #[serde(default)]
    pub phase_diagnostics: WorkspaceRetrievalPhaseDiagnostics,
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
    /// Typed semantic readiness diagnostics for current durable vector artifacts.
    /// This is separate from read-only retrieval phase participation.
    #[serde(default)]
    pub semantic_readiness: Vec<WorkspaceSemanticReadinessDiagnostic>,
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
            Vec::<WorkspaceSemanticReadinessDiagnostic>::new(),
        );

        assert_eq!(
            (
                actual.replay_preview_diagnostics,
                actual.retrieval_plan_diagnostics,
                actual.semantic_readiness,
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
