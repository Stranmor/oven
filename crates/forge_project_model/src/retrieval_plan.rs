//! Pure retrieval execution planning for project-model workspace queries.

use std::cmp::Ordering;
use std::path::{Component, Path};

use anyhow::{Result, bail};

use crate::context_adapter::{ManifestEvidenceTarget, resolve_manifest_evidence_target};
use crate::retrieval::retrieve_with_boundaries_and_rerank_intent_diagnostics;
use crate::types::{
    ContextPack, ContextPackEvidence, ContextPackSelection, EdgeConfidence, FreshnessProofLevel,
    GraphEdge, GraphEdgeKind, ManifestFreshnessEvaluation, ProjectManifest, RerankIntent,
    RerankIntentSource, RetrievalQuery, RetrievalResult, RetrievalScoringWeights,
    StaleEvidencePolicy, VectorQuery,
};
use crate::vector::{Reranker, VectorIndex};
use crate::{
    ExactFactStatus, ExactFactStatusReport, OfflineRerankApplicability, ReplayActivationBoundary,
};

const MAX_DIAGNOSTIC_SUMMARIES: usize = 8;
const EXACT_FACT_REFERENCE_FANOUT_CAP: usize = 8;
const EXACT_FACT_REFERENCE_SCORE: f32 = 25.0;
const EXACT_FACT_REFERENCE_SCORE_PART: &str = "exact_fact_reference";

/// Query request accepted by the project-model retrieval planner.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProjectContextRetrievalRequest {
    /// Free-form retrieval text.
    pub query_text: String,
    /// Maximum number of returned retrieval results.
    pub limit: usize,
    /// Optional full path scope applied before truncation.
    pub path_scope: ProjectContextPathScope,
    /// Whether graph expansion should participate in deterministic retrieval.
    pub include_graph_expansion: bool,
    /// Optional caller use-case preserved as metadata/provenance diagnostics.
    pub use_case: Option<String>,
    /// Typed caller policy used to select reranker intent.
    pub rerank_intent_source: ProjectContextRerankIntentSourceSelection,
    /// Optional typed retrieval candidate budget requested by the caller.
    pub top_k: Option<usize>,
}

impl ProjectContextRetrievalRequest {
    /// Builds a project-context retrieval request.
    ///
    /// # Arguments
    ///
    /// * `query_text` - Free-form retrieval text.
    /// * `limit` - Maximum number of retrieval results.
    /// * `path_scope` - Full path scope applied before truncation.
    /// * `include_graph_expansion` - Whether graph expansion is enabled.
    pub fn new(
        query_text: impl Into<String>,
        limit: usize,
        path_scope: ProjectContextPathScope,
        include_graph_expansion: bool,
    ) -> Self {
        Self {
            query_text: query_text.into(),
            limit,
            path_scope,
            include_graph_expansion,
            use_case: None,
            rerank_intent_source: ProjectContextRerankIntentSourceSelection::Default,
            top_k: None,
        }
    }

    /// Adds use-case metadata to the retrieval request.
    ///
    /// # Arguments
    ///
    /// * `use_case` - Caller-supplied use-case text preserved as typed metadata.
    pub fn with_use_case(mut self, use_case: impl Into<String>) -> Self {
        self.use_case = Some(use_case.into());
        self
    }

    /// Marks this request as automatic project-model context injection.
    pub fn automatic_injection(mut self) -> Self {
        self.rerank_intent_source = ProjectContextRerankIntentSourceSelection::AutomaticInjection;
        self
    }

    /// Adds an explicit retrieval candidate budget to the request.
    ///
    /// # Arguments
    ///
    /// * `top_k` - Candidate budget requested by the caller.
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }
}

/// Typed caller policy for selecting reranker intent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProjectContextRerankIntentSourceSelection {
    /// Normal sem_search behavior: explicit non-empty use-case wins, otherwise query text.
    #[default]
    Default,
    /// Automatic project-model context injection deliberately reranks by actual query text.
    AutomaticInjection,
}

/// Full project-model path scope applied before retrieval truncation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProjectContextPathScope {
    /// Optional manifest-relative path prefix.
    pub starts_with: Option<String>,
    /// Optional manifest-relative path suffix set.
    pub ends_with: Vec<String>,
}

impl ProjectContextPathScope {
    /// Builds a path scope from optional prefix and suffix filters.
    ///
    /// # Arguments
    ///
    /// * `starts_with` - Optional manifest-relative prefix.
    /// * `ends_with` - Optional suffix filters.
    pub fn new(starts_with: Option<String>, ends_with: Vec<String>) -> Self {
        Self { starts_with, ends_with }
    }

    fn matches(&self, path: &str) -> bool {
        if let Some(prefix) = &self.starts_with
            && !path.starts_with(prefix)
        {
            return false;
        }
        if !self.ends_with.is_empty() && !self.ends_with.iter().any(|suffix| path.ends_with(suffix))
        {
            return false;
        }
        true
    }
}

/// Optional pure retrieval integration boundaries supplied by the caller.
#[derive(Clone, Copy, Default)]
pub struct ProjectContextRetrievalOptions<'a> {
    /// Optional precomputed vector query generated outside this crate.
    pub vector_query: Option<&'a VectorQuery>,
    /// Optional validated vector index boundary.
    pub vector_index: Option<ProjectContextVectorIndexBoundary<'a>>,
    /// Optional reranker boundary and readiness metadata.
    pub reranker: Option<ProjectContextRerankerBoundary<'a>>,
    /// Optional semantic unavailable reason from a runtime artifact selector.
    pub vector_unavailable_reason: Option<ProjectContextVectorUnavailableReason>,
    /// Optional exact-fact activation status supplied by a read-only status boundary.
    pub exact_fact_status: Option<&'a ExactFactStatusReport>,
    /// Optional read-only replay activation boundary supplied by project-model activation.
    pub replay_activation: Option<&'a ReplayActivationBoundary>,
}

/// Validated vector index boundary plus redaction-safe metadata.
#[derive(Clone, Copy)]
pub struct ProjectContextVectorIndexBoundary<'a> {
    /// Vector index implementation used by pure hybrid retrieval.
    pub index: &'a dyn VectorIndex,
    /// Redaction-safe identity for diagnostics.
    pub identity: ProjectContextIntegrationIdentity,
    /// Explicit readiness metadata supplied by the integration boundary.
    pub readiness: ProjectContextVectorReadiness,
}

/// Reranker boundary plus redaction-safe metadata.
#[derive(Clone, Copy)]
pub struct ProjectContextRerankerBoundary<'a> {
    /// Reranker implementation used by pure hybrid retrieval.
    pub reranker: &'a dyn Reranker,
    /// Redaction-safe identity for diagnostics.
    pub identity: ProjectContextIntegrationIdentity,
    /// Explicit readiness metadata supplied by the integration boundary.
    pub readiness: ProjectContextRerankerReadiness,
}

/// Redaction-safe integration identity metadata.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProjectContextIntegrationIdentity {
    /// Stable provider or subsystem label with no secrets or raw source text.
    pub provider: &'static str,
    /// Stable model or artifact label with no secrets or raw source text.
    pub artifact: &'static str,
}

/// Explicit vector boundary readiness metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectContextVectorReadiness {
    /// Vector index is ready for queries with the given embedding dimension.
    Ready { dimension: usize },
    /// Vector index is intentionally unavailable.
    Unavailable(ProjectContextVectorUnavailableReason),
}

/// Explicit reranker boundary readiness metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectContextRerankerReadiness {
    /// Reranker is ready for use.
    Ready,
    /// Reranker is intentionally unavailable.
    Unavailable(ProjectContextRerankerUnavailableReason),
}

/// Redaction-safe vector unavailability reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectContextVectorUnavailableReason {
    /// No query embedding was supplied by the caller.
    MissingQueryEmbedding,
    /// No vector index boundary was supplied by the caller.
    MissingVectorIndex,
    /// Vector index metadata says the index is not ready.
    IndexNotReady,
    /// No valid durable vector index matched the query boundary.
    NoMatchingVectorIndex,
    /// Multiple vector index artifacts matched the same query boundary.
    AmbiguousVectorIndex,
}

/// Redaction-safe reranker unavailability reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectContextRerankerUnavailableReason {
    /// No reranker boundary was supplied by the caller.
    MissingReranker,
    /// Reranker metadata says the boundary is not ready.
    RerankerNotReady,
}

/// Redaction-safe invalid vector boundary reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectContextVectorInvalidReason {
    /// Query embedding dimension differs from the ready vector index dimension.
    DimensionMismatch {
        query_dimension: usize,
        index_dimension: usize,
    },
    /// Ready vector index reported an invalid zero dimension.
    ZeroIndexDimension,
    /// Query embedding is empty while vector retrieval was requested.
    EmptyQueryEmbedding,
    /// Query embedding contains a non-finite value.
    NonFiniteQueryEmbedding,
    /// Query embedding has zero norm.
    ZeroQueryEmbeddingNorm,
}

/// Typed redaction-safe phase diagnostics for pure retrieval planning.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProjectContextRetrievalPhaseDiagnostics {
    /// Lexical retrieval phase status.
    pub lexical: ProjectContextRetrievalPhaseStatus,
    /// Graph expansion phase status.
    pub graph: ProjectContextRetrievalPhaseStatus,
    /// Vector retrieval phase status.
    pub vector: ProjectContextRetrievalPhaseStatus,
    /// Reranking phase status.
    pub rerank: ProjectContextRetrievalPhaseStatus,
    /// Exact compiler reference activation phase status.
    pub exact_fact: ProjectContextExactFactPhaseStatus,
}

/// Typed redaction-safe exact-fact retrieval phase status.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectContextExactFactPhaseStatus {
    /// Exact-fact phase selected eligible exact compiler references.
    Active(ProjectContextExactFactActiveSummary),
    /// Exact-fact phase was inactive or unsafe and selected nothing.
    Inactive(ProjectContextExactFactInactiveReason),
}

impl Default for ProjectContextExactFactPhaseStatus {
    fn default() -> Self {
        Self::Inactive(ProjectContextExactFactInactiveReason::StatusUnavailable)
    }
}

/// Bounded exact-fact activation summary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectContextExactFactActiveSummary {
    /// Count of exact compiler reference edges considered before eligibility filters.
    pub candidate_count: usize,
    /// Count of eligible exact references selected under deterministic cap.
    pub selected_count: usize,
    /// Deterministic fanout cap for exact references.
    pub fanout_cap: usize,
}

/// Deterministic redaction-safe reason exact facts did not participate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectContextExactFactInactiveReason {
    /// No exact-fact status report was supplied to retrieval.
    StatusUnavailable,
    /// Status report says exact facts are inactive.
    StatusInactive(ExactFactStatus),
    /// Status report says exact facts are active but manifest hash readback is absent.
    MissingManifestHash,
    /// Status report was produced for a different manifest snapshot.
    WorkspaceSnapshotMismatch,
    /// Manifest freshness was not proven for full injection/readback.
    ManifestNotFresh,
    /// Active status contains no eligible exact compiler reference evidence.
    NoEligibleEvidence,
    /// Eligible exact evidence existed, but ranking/path filters selected zero.
    SelectedZero,
}

/// Exact compiler reference evidence constructed only after all activation guards pass.
#[derive(Clone, Debug, PartialEq)]
pub struct ExactCompilerReferenceEvidence {
    edge: GraphEdge,
    read_request: ProjectContextReadRequest,
}

impl ExactCompilerReferenceEvidence {
    /// Constructs exact compiler reference evidence behind the activation boundary.
    ///
    /// # Arguments
    ///
    /// * `manifest` - Project manifest owning the graph edge and readback target.
    /// * `freshness` - Freshness evaluation for injection/readback eligibility.
    /// * `status` - Read-only exact-fact status report.
    /// * `edge` - Candidate graph edge.
    ///
    /// # Errors
    ///
    /// Returns an error when the edge or activation state is not exact, active,
    /// fresh, snapshot-identical, and readback-eligible.
    pub fn new(
        manifest: &ProjectManifest,
        freshness: &ManifestFreshnessEvaluation,
        status: &ExactFactStatusReport,
        edge: &GraphEdge,
    ) -> Result<Self> {
        if status.status != ExactFactStatus::Active || !status.exact_facts_active {
            bail!("exact fact status is not active");
        }
        if status.manifest_hash.as_deref() != Some(manifest.manifest_hash.as_str()) {
            bail!("exact fact manifest snapshot mismatch");
        }
        if !freshness.can_inject() {
            bail!("exact fact manifest is not fresh for injection");
        }
        if edge.kind != GraphEdgeKind::References {
            bail!("exact fact edge kind is not references");
        }
        if edge.confidence_kind != EdgeConfidence::ExactCompiler {
            bail!("exact fact edge confidence is not exact compiler");
        }
        let target = resolve_manifest_evidence_target(manifest, &edge.to)
            .ok_or_else(|| anyhow::anyhow!("exact fact referenced evidence target is missing"))?;
        let (start_line, end_line) = target
            .line_range
            .ok_or_else(|| anyhow::anyhow!("exact fact evidence line range is missing"))?;
        let read_request =
            ProjectContextReadRequest::new(target.path, edge.to.clone(), start_line, end_line)?;
        Ok(Self { edge: edge.clone(), read_request })
    }

    fn into_retrieval_result(self) -> RetrievalResult {
        let mut score_parts = std::collections::BTreeMap::new();
        score_parts.insert(
            EXACT_FACT_REFERENCE_SCORE_PART.to_string(),
            EXACT_FACT_REFERENCE_SCORE,
        );
        RetrievalResult {
            id: self.edge.to,
            path: self.read_request.relative_manifest_path().to_string(),
            symbol: None,
            score: EXACT_FACT_REFERENCE_SCORE,
            score_parts,
            provenance: self.edge.provenance,
        }
    }
}

/// Typed redaction-safe status for one retrieval phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectContextRetrievalPhaseStatus {
    /// Phase was active; result count is a phase participation count, not an
    /// availability proof.
    Active { result_count: usize },
    /// Phase was valid but intentionally not used for this request.
    Skipped(ProjectContextRetrievalPhaseSkipReason),
    /// Phase could not run because a boundary or input was absent/not ready.
    Unavailable(ProjectContextRetrievalPhaseUnavailableReason),
    /// Phase input was present but invalid.
    Invalid(ProjectContextRetrievalPhaseInvalidReason),
}

impl Default for ProjectContextRetrievalPhaseStatus {
    fn default() -> Self {
        Self::Skipped(ProjectContextRetrievalPhaseSkipReason::EmptyQueryText)
    }
}

/// Redaction-safe reason for a skipped phase.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProjectContextRetrievalPhaseSkipReason {
    /// Query text is empty, so lexical/rerank text matching is skipped.
    #[default]
    EmptyQueryText,
    /// Reranker boundary was ready, but no non-empty intent text was selected.
    EmptyRerankIntent,
    /// Graph expansion was not requested.
    GraphExpansionDisabled,
}

/// Redaction-safe reason for an unavailable phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectContextRetrievalPhaseUnavailableReason {
    /// Vector query embedding was not supplied.
    MissingQueryEmbedding,
    /// Vector index boundary was not supplied.
    MissingVectorIndex,
    /// Vector index boundary reported not-ready status.
    VectorIndexNotReady,
    /// Reranker boundary was not supplied.
    MissingReranker,
    /// Reranker boundary reported not-ready status.
    RerankerNotReady,
    /// No valid durable vector index matched the query boundary.
    NoMatchingVectorIndex,
    /// Multiple vector index artifacts matched the same query boundary.
    AmbiguousVectorIndex,
}

/// Redaction-safe reason for an invalid phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectContextRetrievalPhaseInvalidReason {
    /// Query vector dimension differs from index dimension.
    VectorDimensionMismatch {
        query_dimension: usize,
        index_dimension: usize,
    },
    /// Ready index reported zero dimensions.
    VectorIndexZeroDimension,
    /// Query embedding was empty.
    EmptyQueryEmbedding,
    /// Query embedding contains a non-finite value.
    NonFiniteQueryEmbedding,
    /// Query embedding has zero norm.
    ZeroQueryEmbeddingNorm,
}

/// Pure planner refusal for project-context retrieval.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectContextRetrievalRefusal {
    /// Stable machine-readable refusal code.
    pub code: ProjectContextRetrievalRefusalCode,
    /// Human-readable redaction-safe refusal detail.
    pub detail: String,
}

/// Stable project-context retrieval refusal code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectContextRetrievalRefusalCode {
    /// Manifest freshness is not strong enough for injection/query retrieval.
    ManifestNotInjectable,
    /// Context pack construction rejected stale evidence.
    StaleEvidenceRejected,
    /// Evidence line range could not be resolved from the manifest.
    EvidenceRangeMissing,
    /// Evidence path failed read-request validation.
    InvalidReadRequestPath,
}

/// Pure retrieval planning result.
#[derive(Clone, Debug, PartialEq)]
pub enum ProjectContextRetrievalPlanningOutcome {
    /// Retrieval is refused before any read or write side effect.
    Refusal(ProjectContextRetrievalRefusal),
    /// Retrieval is planned and safe for IO execution by a service boundary.
    Plan(Box<ProjectContextRetrievalPlan>),
}

/// Redaction-safe pure diagnostic projection for a retrieval planning outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectContextRetrievalPlanDiagnostic {
    /// Whether the planner produced an executable plan.
    pub planned: bool,
    /// Stable machine-readable refusal code when planning was refused.
    pub refusal_code: Option<ProjectContextRetrievalRefusalCode>,
    /// Human-readable redaction-safe refusal detail when planning was refused.
    pub refusal_detail: Option<String>,
    /// Number of retrieval results selected by the planner.
    pub selected_result_count: usize,
    /// Number of validated read requests planned before readback.
    pub read_request_count: usize,
    /// Deterministic write decision when planning succeeded.
    pub write_decision: Option<ProjectContextWriteDecision>,
    /// Bounded metadata-only summaries of selected retrieval results.
    pub selected_summaries: Vec<ProjectContextRetrievalSelectedSummary>,
    /// Bounded metadata-only summaries of planned read requests.
    pub read_request_summaries: Vec<ProjectContextRetrievalReadRequestSummary>,
    /// Typed redaction-safe retrieval phase diagnostics.
    pub phase_diagnostics: ProjectContextRetrievalPhaseDiagnostics,
    /// Typed redaction-safe selected rerank intent source.
    pub rerank_intent_source: Option<RerankIntentSource>,
    /// Fingerprint of the selected rerank intent text without raw payload duplication.
    pub rerank_intent_fingerprint: Option<String>,
    /// Normalized selected rerank intent length.
    pub rerank_intent_len: Option<usize>,
    /// Redaction-safe offline rerank applicability for this runtime request, if available.
    pub offline_rerank_applicability: Option<OfflineRerankApplicability>,
    /// Whether retrieval selected no evidence.
    pub retrieval_empty: bool,
    /// Whether selected or read-request summaries were truncated.
    pub truncated: bool,
}

impl ProjectContextRetrievalPlanDiagnostic {
    /// Builds a redaction-safe diagnostic projection from a pure planning outcome.
    ///
    /// # Arguments
    ///
    /// * `outcome` - Pure retrieval planning outcome to project into bounded diagnostics.
    pub fn from_outcome(outcome: &ProjectContextRetrievalPlanningOutcome) -> Self {
        match outcome {
            ProjectContextRetrievalPlanningOutcome::Refusal(refusal) => Self {
                planned: false,
                refusal_code: Some(refusal.code.clone()),
                refusal_detail: Some(refusal.detail.clone()),
                selected_result_count: 0,
                read_request_count: 0,
                write_decision: None,
                selected_summaries: Vec::new(),
                read_request_summaries: Vec::new(),
                phase_diagnostics: ProjectContextRetrievalPhaseDiagnostics::default(),
                rerank_intent_source: None,
                rerank_intent_fingerprint: None,
                rerank_intent_len: None,
                offline_rerank_applicability: None,
                retrieval_empty: false,
                truncated: false,
            },
            ProjectContextRetrievalPlanningOutcome::Plan(plan) => {
                let selected_result_count = plan.selected_results.len();
                let read_request_count = plan.read_requests.len();
                let selected_summaries = plan
                    .selected_results
                    .iter()
                    .take(MAX_DIAGNOSTIC_SUMMARIES)
                    .map(ProjectContextRetrievalSelectedSummary::from_result)
                    .collect::<Vec<_>>();
                let read_request_summaries = plan
                    .read_requests
                    .iter()
                    .take(MAX_DIAGNOSTIC_SUMMARIES)
                    .map(ProjectContextRetrievalReadRequestSummary::from_request)
                    .collect::<Vec<_>>();
                Self {
                    planned: true,
                    refusal_code: None,
                    refusal_detail: None,
                    selected_result_count,
                    read_request_count,
                    write_decision: Some(plan.write_decision.clone()),
                    selected_summaries,
                    read_request_summaries,
                    phase_diagnostics: plan.query_diagnostics.phase_diagnostics.clone(),
                    rerank_intent_source: plan.query_diagnostics.rerank_intent_source,
                    rerank_intent_fingerprint: plan
                        .query_diagnostics
                        .rerank_intent_fingerprint
                        .clone(),
                    rerank_intent_len: plan.query_diagnostics.rerank_intent_len,
                    offline_rerank_applicability: plan
                        .query_diagnostics
                        .offline_rerank_applicability
                        .clone(),
                    retrieval_empty: selected_result_count == 0,
                    truncated: selected_result_count > MAX_DIAGNOSTIC_SUMMARIES
                        || read_request_count > MAX_DIAGNOSTIC_SUMMARIES,
                }
            }
        }
    }
}

/// Metadata-only selected-result summary for retrieval diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectContextRetrievalSelectedSummary {
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

impl ProjectContextRetrievalSelectedSummary {
    fn from_result(result: &RetrievalResult) -> Self {
        Self {
            evidence_id: result.id.clone(),
            path: result.path.clone(),
            start_line: result.provenance.start_line,
            end_line: result.provenance.end_line,
            relevance: result.score,
        }
    }
}

/// Metadata-only read-request summary for retrieval diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectContextRetrievalReadRequestSummary {
    /// Evidence identifier planned for readback.
    pub evidence_id: String,
    /// Manifest-relative path planned for readback.
    pub path: String,
    /// One-based inclusive start line.
    pub start_line: u32,
    /// One-based inclusive end line.
    pub end_line: u32,
}

impl ProjectContextRetrievalReadRequestSummary {
    fn from_request(request: &ProjectContextReadRequest) -> Self {
        Self {
            evidence_id: request.evidence_id.clone(),
            path: request.relative_manifest_path().to_string(),
            start_line: request.start_line,
            end_line: request.end_line,
        }
    }
}

/// Pure project-context retrieval execution plan.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectContextRetrievalPlan {
    /// Diagnostics for the retrieval query created by the planner.
    pub query_diagnostics: ProjectContextRetrievalQueryDiagnostics,
    /// Selected retrieval results before context-pack ordering.
    pub selected_results: Vec<RetrievalResult>,
    /// Deterministic context pack intended for persistence after readback.
    pub context_pack: Option<ContextPack>,
    /// Validated read requests to execute before writing the pack.
    pub read_requests: Vec<ProjectContextReadRequest>,
    /// Deterministic write decision.
    pub write_decision: ProjectContextWriteDecision,
    /// Stable return ordering independent of context-pack evidence order.
    pub return_order: Vec<ProjectContextReturnOrderItem>,
}

/// Diagnostics for a project-context retrieval query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectContextRetrievalQueryDiagnostics {
    /// Query text supplied to retrieval.
    pub query_text: Option<String>,
    /// Optional prefix scope supplied to retrieval.
    pub path_prefix: Option<String>,
    /// Suffix scope applied before truncation by the planner.
    pub path_suffixes: Vec<String>,
    /// Effective retrieval limit.
    pub limit: usize,
    /// Candidate budget metadata supplied by the caller.
    pub top_k: Option<usize>,
    /// Status of top-k support for this query.
    pub top_k_status: ProjectContextTopKStatus,
    /// Redaction-safe use-case metadata supplied by the caller.
    pub use_case: Option<String>,
    /// Selected reranker intent source.
    pub rerank_intent_source: Option<RerankIntentSource>,
    /// Fingerprint of the normalized reranker intent when selected.
    pub rerank_intent_fingerprint: Option<String>,
    /// Character length of the normalized reranker intent when selected.
    pub rerank_intent_len: Option<usize>,
    /// Redaction-safe offline rerank applicability for this runtime request, if available.
    pub offline_rerank_applicability: Option<OfflineRerankApplicability>,
    /// Whether graph expansion was requested.
    pub include_graph_expansion: bool,
    /// Fixed stale policy used for query path injection.
    pub stale_policy: StaleEvidencePolicy,
    /// Freshness proof level used for injection gating.
    pub freshness_proof_level: FreshnessProofLevel,
    /// Typed redaction-safe retrieval phase diagnostics.
    pub phase_diagnostics: ProjectContextRetrievalPhaseDiagnostics,
}

/// Typed top-k handling status for project-context retrieval.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ProjectContextTopKStatus {
    /// No top-k candidate budget was supplied.
    #[default]
    NotRequested,
    /// Candidate retrieval used the supplied top-k budget before final limit truncation.
    Applied { candidate_count: usize },
}

/// Runtime semantic options passed from service/integration boundaries.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProjectContextSemanticQueryOptions {
    /// Optional provider-neutral query embedding generated outside this crate.
    pub query_embedding: Option<Vec<f32>>,
    /// Optional external embedding model identity used for durable index selection.
    pub embedding_model_id: Option<String>,
}

impl ProjectContextSemanticQueryOptions {
    /// Builds semantic query options from optional embedding data.
    ///
    /// # Arguments
    ///
    /// * `query_embedding` - Optional provider-neutral query vector.
    /// * `embedding_model_id` - Optional external embedding model identity.
    pub fn new(query_embedding: Option<Vec<f32>>, embedding_model_id: Option<String>) -> Self {
        Self { query_embedding, embedding_model_id }
    }
}

/// Deterministic write decision for project-context retrieval.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectContextWriteDecision {
    /// No readback or pack write should occur because retrieval selected no
    /// evidence.
    NoWriteEmptyRetrieval,
    /// Persist the context pack after every readback succeeds.
    WriteContextPackAfterReadback,
}

/// Stable metadata item describing service return order.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectContextReturnOrderItem {
    /// Evidence identifier returned to the service as a node id.
    pub evidence_id: String,
    /// Relevance score used for stable ordering.
    pub relevance: f32,
}

/// Validated manifest-relative read request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectContextReadRequest {
    relative_manifest_path: String,
    /// Evidence identifier read from this path.
    pub evidence_id: String,
    /// One-based inclusive start line.
    pub start_line: u32,
    /// One-based inclusive end line.
    pub end_line: u32,
}

impl ProjectContextReadRequest {
    /// Builds a validated manifest-relative read request.
    ///
    /// # Arguments
    ///
    /// * `relative_manifest_path` - Manifest-relative path using safe normal
    ///   path components.
    /// * `evidence_id` - Evidence identifier to associate with the readback.
    /// * `start_line` - One-based inclusive start line.
    /// * `end_line` - One-based inclusive end line.
    ///
    /// # Errors
    ///
    /// Returns an error for absolute paths, empty paths, parent components,
    /// traversal components, platform prefixes, or invalid line ranges.
    pub fn new(
        relative_manifest_path: impl Into<String>,
        evidence_id: impl Into<String>,
        start_line: u32,
        end_line: u32,
    ) -> Result<Self> {
        let relative_manifest_path = relative_manifest_path.into();
        validate_manifest_relative_path(&relative_manifest_path)?;
        if start_line == 0 || end_line < start_line {
            bail!("project context read request line range is invalid");
        }
        Ok(Self {
            relative_manifest_path,
            evidence_id: evidence_id.into(),
            start_line,
            end_line,
        })
    }

    /// Returns the validated manifest-relative path.
    pub fn relative_manifest_path(&self) -> &str {
        &self.relative_manifest_path
    }
}

/// Plans project-context retrieval without performing IO.
///
/// # Arguments
///
/// * `manifest` - Project manifest used for retrieval and evidence resolution.
/// * `freshness` - Freshness evaluation used for injection gating and stale
///   evidence checks.
/// * `request` - Typed retrieval request.
pub fn plan_project_context_retrieval(
    manifest: &ProjectManifest,
    freshness: &ManifestFreshnessEvaluation,
    request: ProjectContextRetrievalRequest,
) -> ProjectContextRetrievalPlanningOutcome {
    plan_project_context_retrieval_with_options(
        manifest,
        freshness,
        request,
        ProjectContextRetrievalOptions::default(),
    )
}

/// Plans project-context retrieval with optional pure semantic boundaries.
///
/// # Arguments
///
/// * `manifest` - Project manifest used for retrieval and evidence resolution.
/// * `freshness` - Freshness evaluation used for injection gating and stale
///   evidence checks.
/// * `request` - Typed retrieval request.
/// * `options` - Optional vector/reranker boundaries supplied by the caller.
pub fn plan_project_context_retrieval_with_options(
    manifest: &ProjectManifest,
    freshness: &ManifestFreshnessEvaluation,
    request: ProjectContextRetrievalRequest,
    options: ProjectContextRetrievalOptions<'_>,
) -> ProjectContextRetrievalPlanningOutcome {
    let effective_limit = if request.limit == 0 {
        10
    } else {
        request.limit
    };
    let candidate_limit = request
        .top_k
        .unwrap_or(effective_limit)
        .max(effective_limit);
    let top_k_status = match request.top_k {
        Some(top_k) => {
            ProjectContextTopKStatus::Applied { candidate_count: top_k.max(effective_limit) }
        }
        None => ProjectContextTopKStatus::NotRequested,
    };
    let selected_rerank_intent = select_rerank_intent_for_request(&request);
    let phase_diagnostics = ProjectContextRetrievalPhaseDiagnostics {
        lexical: lexical_phase_status(&request.query_text),
        graph: graph_phase_status(request.include_graph_expansion),
        vector: ProjectContextRetrievalPhaseStatus::default(),
        rerank: ProjectContextRetrievalPhaseStatus::default(),
        exact_fact: ProjectContextExactFactPhaseStatus::default(),
    };
    let diagnostics = ProjectContextRetrievalQueryDiagnostics {
        query_text: Some(request.query_text.clone()),
        path_prefix: request.path_scope.starts_with.clone(),
        path_suffixes: request.path_scope.ends_with.clone(),
        limit: effective_limit,
        top_k: request.top_k,
        top_k_status,
        use_case: normalized_use_case(request.use_case.as_deref()),
        rerank_intent_source: selected_rerank_intent.as_ref().map(|intent| intent.source),
        rerank_intent_fingerprint: selected_rerank_intent
            .as_ref()
            .map(|intent| crate::fingerprint(&intent.text)),
        rerank_intent_len: selected_rerank_intent
            .as_ref()
            .map(|intent| intent.text.chars().count()),
        offline_rerank_applicability: None,
        include_graph_expansion: request.include_graph_expansion,
        stale_policy: StaleEvidencePolicy::Reject,
        freshness_proof_level: freshness.proof_level.clone(),
        phase_diagnostics,
    };
    if !freshness.can_inject() {
        return ProjectContextRetrievalPlanningOutcome::Refusal(ProjectContextRetrievalRefusal {
            code: ProjectContextRetrievalRefusalCode::ManifestNotInjectable,
            detail: "project-model manifest is not fully fresh for injection".to_string(),
        });
    }

    let query = RetrievalQuery {
        text: diagnostics.query_text.clone(),
        path: None,
        path_prefix: diagnostics.path_prefix.clone(),
        symbol: None,
        limit: candidate_limit,
        include_graph_expansion: diagnostics.include_graph_expansion,
    };
    let vector_activation = vector_activation(
        options.vector_query,
        options.vector_index,
        options.vector_unavailable_reason,
    );
    let reranker_activation =
        reranker_activation(options.reranker, selected_rerank_intent.as_ref());
    let retrieval_execution = retrieve_with_boundaries_and_rerank_intent_diagnostics(
        manifest,
        &query,
        vector_activation.vector_query,
        vector_activation.vector_index,
        reranker_activation.reranker,
        selected_rerank_intent.as_ref(),
        &RetrievalScoringWeights::default(),
    );
    let mut selected_results = retrieval_execution.results;
    let exact_fact_activation =
        exact_fact_activation(manifest, freshness, options.exact_fact_status);
    selected_results.extend(exact_fact_activation.results.iter().cloned());
    if selected_results.is_empty()
        && let Some(replay_activation) = options.replay_activation
    {
        selected_results.extend(
            replay_activation
                .active_refs
                .iter()
                .map(|reference| reference.to_retrieval_result()),
        );
    }
    selected_results = selected_results
        .into_iter()
        .filter(|result| request.path_scope.matches(&result.path))
        .collect::<Vec<_>>();
    selected_results.sort_by(compare_retrieval_results_for_return);
    selected_results.dedup_by(|left, right| left.id == right.id);
    selected_results.truncate(effective_limit);

    let mut diagnostics = diagnostics;
    diagnostics.offline_rerank_applicability = retrieval_execution.offline_rerank_applicability;
    diagnostics.phase_diagnostics.vector = vector_activation.status(&selected_results);
    diagnostics.phase_diagnostics.rerank = reranker_activation.status(&selected_results);
    diagnostics.phase_diagnostics.lexical =
        lexical_phase_status_with_results(&request.query_text, &selected_results);
    diagnostics.phase_diagnostics.graph =
        graph_phase_status_with_results(request.include_graph_expansion, &selected_results);
    diagnostics.phase_diagnostics.exact_fact = exact_fact_activation.status(&selected_results);

    if selected_results.is_empty() {
        return ProjectContextRetrievalPlanningOutcome::Plan(Box::new(
            ProjectContextRetrievalPlan {
                query_diagnostics: diagnostics,
                selected_results,
                context_pack: None,
                read_requests: Vec::new(),
                write_decision: ProjectContextWriteDecision::NoWriteEmptyRetrieval,
                return_order: Vec::new(),
            },
        ));
    }

    let context_pack = match ContextPack::from_selection(
        manifest,
        ContextPackSelection {
            retrieval_results: selected_results.clone(),
            shards: Vec::new(),
            evidence: Vec::new(),
            freshness: freshness.state.clone(),
            stale_policy: StaleEvidencePolicy::Reject,
        },
    ) {
        Ok(context_pack) => context_pack,
        Err(error) => {
            return ProjectContextRetrievalPlanningOutcome::Refusal(
                ProjectContextRetrievalRefusal {
                    code: ProjectContextRetrievalRefusalCode::StaleEvidenceRejected,
                    detail: error.to_string(),
                },
            );
        }
    };

    let mut read_requests = Vec::new();
    for evidence in &context_pack.evidence {
        let target = match resolve_readback_evidence_target(manifest, evidence) {
            Some(target) => target,
            None => {
                return ProjectContextRetrievalPlanningOutcome::Refusal(
                    ProjectContextRetrievalRefusal {
                        code: ProjectContextRetrievalRefusalCode::EvidenceRangeMissing,
                        detail: format!(
                            "project-model evidence line range is missing: {}",
                            evidence.id
                        ),
                    },
                );
            }
        };
        let (start_line, end_line) = match target.line_range {
            Some(line_range) => line_range,
            None => {
                return ProjectContextRetrievalPlanningOutcome::Refusal(
                    ProjectContextRetrievalRefusal {
                        code: ProjectContextRetrievalRefusalCode::EvidenceRangeMissing,
                        detail: format!(
                            "project-model evidence line range is missing: {}",
                            evidence.id
                        ),
                    },
                );
            }
        };
        match ProjectContextReadRequest::new(target.path, evidence.id.clone(), start_line, end_line)
        {
            Ok(read_request) => read_requests.push(read_request),
            Err(error) => {
                return ProjectContextRetrievalPlanningOutcome::Refusal(
                    ProjectContextRetrievalRefusal {
                        code: ProjectContextRetrievalRefusalCode::InvalidReadRequestPath,
                        detail: error.to_string(),
                    },
                );
            }
        }
    }
    let return_order = stable_return_order(&context_pack.evidence);

    ProjectContextRetrievalPlanningOutcome::Plan(Box::new(ProjectContextRetrievalPlan {
        query_diagnostics: diagnostics,
        selected_results,
        context_pack: Some(context_pack),
        read_requests,
        write_decision: ProjectContextWriteDecision::WriteContextPackAfterReadback,
        return_order,
    }))
}

struct ProjectContextExactFactActivation {
    results: Vec<RetrievalResult>,
    initial_status: ProjectContextExactFactPhaseStatus,
}

impl ProjectContextExactFactActivation {
    fn status(&self, selected_results: &[RetrievalResult]) -> ProjectContextExactFactPhaseStatus {
        match &self.initial_status {
            ProjectContextExactFactPhaseStatus::Active(summary) => {
                let selected_count = selected_results
                    .iter()
                    .filter(|result| {
                        result
                            .score_parts
                            .contains_key(EXACT_FACT_REFERENCE_SCORE_PART)
                    })
                    .count();
                if selected_count == 0 {
                    ProjectContextExactFactPhaseStatus::Inactive(
                        ProjectContextExactFactInactiveReason::SelectedZero,
                    )
                } else {
                    ProjectContextExactFactPhaseStatus::Active(
                        ProjectContextExactFactActiveSummary {
                            candidate_count: summary.candidate_count,
                            selected_count,
                            fanout_cap: summary.fanout_cap,
                        },
                    )
                }
            }
            status => status.clone(),
        }
    }
}

fn exact_fact_activation(
    manifest: &ProjectManifest,
    freshness: &ManifestFreshnessEvaluation,
    status: Option<&ExactFactStatusReport>,
) -> ProjectContextExactFactActivation {
    let Some(status) = status else {
        return inactive_exact_fact(ProjectContextExactFactInactiveReason::StatusUnavailable);
    };
    if status.status != ExactFactStatus::Active || !status.exact_facts_active {
        return inactive_exact_fact(ProjectContextExactFactInactiveReason::StatusInactive(
            status.status.clone(),
        ));
    }
    let Some(status_manifest_hash) = status.manifest_hash.as_deref() else {
        return inactive_exact_fact(ProjectContextExactFactInactiveReason::MissingManifestHash);
    };
    if status_manifest_hash != manifest.manifest_hash {
        return inactive_exact_fact(
            ProjectContextExactFactInactiveReason::WorkspaceSnapshotMismatch,
        );
    }
    if !freshness.can_inject()
        || !freshness.state.changed.is_empty()
        || !freshness.state.deleted.is_empty()
        || !freshness.state.added.is_empty()
    {
        return inactive_exact_fact(ProjectContextExactFactInactiveReason::ManifestNotFresh);
    }

    let candidate_count = manifest
        .edges
        .iter()
        .filter(|edge| {
            edge.kind == GraphEdgeKind::References
                && edge.confidence_kind == EdgeConfidence::ExactCompiler
        })
        .count();
    let mut results = manifest
        .edges
        .iter()
        .filter_map(|edge| {
            ExactCompilerReferenceEvidence::new(manifest, freshness, status, edge)
                .ok()
                .map(ExactCompilerReferenceEvidence::into_retrieval_result)
        })
        .collect::<Vec<_>>();
    results.sort_by(compare_retrieval_results_for_return);
    results.dedup_by(|left, right| left.id == right.id);
    results.truncate(EXACT_FACT_REFERENCE_FANOUT_CAP);
    if results.is_empty() {
        return inactive_exact_fact(ProjectContextExactFactInactiveReason::NoEligibleEvidence);
    }
    let selected_count = results.len();
    ProjectContextExactFactActivation {
        results,
        initial_status: ProjectContextExactFactPhaseStatus::Active(
            ProjectContextExactFactActiveSummary {
                candidate_count,
                selected_count,
                fanout_cap: EXACT_FACT_REFERENCE_FANOUT_CAP,
            },
        ),
    }
}

fn inactive_exact_fact(
    reason: ProjectContextExactFactInactiveReason,
) -> ProjectContextExactFactActivation {
    ProjectContextExactFactActivation {
        results: Vec::new(),
        initial_status: ProjectContextExactFactPhaseStatus::Inactive(reason),
    }
}

fn normalized_use_case(use_case: Option<&str>) -> Option<String> {
    use_case
        .map(str::trim)
        .filter(|use_case| !use_case.is_empty())
        .map(ToString::to_string)
}

fn select_rerank_intent_for_request(
    request: &ProjectContextRetrievalRequest,
) -> Option<RerankIntent> {
    match request.rerank_intent_source {
        ProjectContextRerankIntentSourceSelection::Default => {
            normalized_use_case(request.use_case.as_deref())
                .and_then(|use_case| {
                    RerankIntent::new(use_case, RerankIntentSource::ExplicitUseCase)
                })
                .or_else(|| {
                    RerankIntent::new(
                        request.query_text.as_str(),
                        RerankIntentSource::QueryTextFallback,
                    )
                })
        }
        ProjectContextRerankIntentSourceSelection::AutomaticInjection => RerankIntent::new(
            request.query_text.as_str(),
            RerankIntentSource::AutomaticInjectionQueryFallback,
        ),
    }
}

fn lexical_phase_status(query_text: &str) -> ProjectContextRetrievalPhaseStatus {
    if query_text.trim().is_empty() {
        ProjectContextRetrievalPhaseStatus::Skipped(
            ProjectContextRetrievalPhaseSkipReason::EmptyQueryText,
        )
    } else {
        ProjectContextRetrievalPhaseStatus::Active { result_count: 0 }
    }
}

fn lexical_phase_status_with_results(
    query_text: &str,
    selected_results: &[RetrievalResult],
) -> ProjectContextRetrievalPhaseStatus {
    if query_text.trim().is_empty() {
        ProjectContextRetrievalPhaseStatus::Skipped(
            ProjectContextRetrievalPhaseSkipReason::EmptyQueryText,
        )
    } else {
        ProjectContextRetrievalPhaseStatus::Active {
            result_count: selected_results
                .iter()
                .filter(|result| result.score_parts.contains_key("lexical"))
                .count(),
        }
    }
}

fn graph_phase_status(include_graph_expansion: bool) -> ProjectContextRetrievalPhaseStatus {
    if include_graph_expansion {
        ProjectContextRetrievalPhaseStatus::Active { result_count: 0 }
    } else {
        ProjectContextRetrievalPhaseStatus::Skipped(
            ProjectContextRetrievalPhaseSkipReason::GraphExpansionDisabled,
        )
    }
}

fn graph_phase_status_with_results(
    include_graph_expansion: bool,
    selected_results: &[RetrievalResult],
) -> ProjectContextRetrievalPhaseStatus {
    if include_graph_expansion {
        ProjectContextRetrievalPhaseStatus::Active {
            result_count: selected_results
                .iter()
                .filter(|result| result.score_parts.contains_key("graph"))
                .count(),
        }
    } else {
        ProjectContextRetrievalPhaseStatus::Skipped(
            ProjectContextRetrievalPhaseSkipReason::GraphExpansionDisabled,
        )
    }
}

fn vector_activation<'a>(
    vector_query: Option<&'a VectorQuery>,
    vector_index: Option<ProjectContextVectorIndexBoundary<'a>>,
    unavailable_reason: Option<ProjectContextVectorUnavailableReason>,
) -> ProjectContextVectorActivation<'a> {
    match (vector_query, vector_index) {
        (Some(query), Some(boundary)) => match boundary.readiness {
            ProjectContextVectorReadiness::Ready { dimension } => {
                if dimension == 0 {
                    ProjectContextVectorActivation::invalid(
                        ProjectContextRetrievalPhaseInvalidReason::VectorIndexZeroDimension,
                    )
                } else if query.embedding.is_empty() {
                    ProjectContextVectorActivation::invalid(
                        ProjectContextRetrievalPhaseInvalidReason::EmptyQueryEmbedding,
                    )
                } else if query.embedding.len() != dimension {
                    ProjectContextVectorActivation::invalid(
                        ProjectContextRetrievalPhaseInvalidReason::VectorDimensionMismatch {
                            query_dimension: query.embedding.len(),
                            index_dimension: dimension,
                        },
                    )
                } else if let Some(reason) = invalid_query_embedding_value_reason(query) {
                    ProjectContextVectorActivation::invalid(reason)
                } else {
                    ProjectContextVectorActivation {
                        vector_query: Some(query),
                        vector_index: Some(boundary.index),
                        initial_status: None,
                    }
                }
            }
            ProjectContextVectorReadiness::Unavailable(reason) => {
                ProjectContextVectorActivation::unavailable(vector_unavailable_phase_reason(reason))
            }
        },
        (None, _) => ProjectContextVectorActivation::unavailable(
            ProjectContextRetrievalPhaseUnavailableReason::MissingQueryEmbedding,
        ),
        (Some(query), None) if query.embedding.is_empty() => {
            ProjectContextVectorActivation::invalid(
                ProjectContextRetrievalPhaseInvalidReason::EmptyQueryEmbedding,
            )
        }
        (Some(query), None) => match invalid_query_embedding_value_reason(query) {
            Some(reason) => ProjectContextVectorActivation::invalid(reason),
            None => ProjectContextVectorActivation::unavailable(
                unavailable_reason
                    .map(vector_unavailable_phase_reason)
                    .unwrap_or(ProjectContextRetrievalPhaseUnavailableReason::MissingVectorIndex),
            ),
        },
    }
}

fn invalid_query_embedding_value_reason(
    query: &VectorQuery,
) -> Option<ProjectContextRetrievalPhaseInvalidReason> {
    if query.embedding.iter().any(|value| !value.is_finite()) {
        return Some(ProjectContextRetrievalPhaseInvalidReason::NonFiniteQueryEmbedding);
    }

    let mut squared_norm = 0.0f32;
    for value in &query.embedding {
        let square = value * value;
        if !square.is_finite() {
            return Some(ProjectContextRetrievalPhaseInvalidReason::NonFiniteQueryEmbedding);
        }
        squared_norm += square;
        if !squared_norm.is_finite() {
            return Some(ProjectContextRetrievalPhaseInvalidReason::NonFiniteQueryEmbedding);
        }
    }

    if squared_norm <= 0.0 {
        Some(ProjectContextRetrievalPhaseInvalidReason::ZeroQueryEmbeddingNorm)
    } else {
        None
    }
}

fn vector_unavailable_phase_reason(
    reason: ProjectContextVectorUnavailableReason,
) -> ProjectContextRetrievalPhaseUnavailableReason {
    match reason {
        ProjectContextVectorUnavailableReason::MissingQueryEmbedding => {
            ProjectContextRetrievalPhaseUnavailableReason::MissingQueryEmbedding
        }
        ProjectContextVectorUnavailableReason::MissingVectorIndex => {
            ProjectContextRetrievalPhaseUnavailableReason::MissingVectorIndex
        }
        ProjectContextVectorUnavailableReason::IndexNotReady => {
            ProjectContextRetrievalPhaseUnavailableReason::VectorIndexNotReady
        }
        ProjectContextVectorUnavailableReason::NoMatchingVectorIndex => {
            ProjectContextRetrievalPhaseUnavailableReason::NoMatchingVectorIndex
        }
        ProjectContextVectorUnavailableReason::AmbiguousVectorIndex => {
            ProjectContextRetrievalPhaseUnavailableReason::AmbiguousVectorIndex
        }
    }
}

fn reranker_activation<'a>(
    reranker: Option<ProjectContextRerankerBoundary<'a>>,
    rerank_intent: Option<&RerankIntent>,
) -> ProjectContextRerankerActivation<'a> {
    if rerank_intent.is_none() {
        return ProjectContextRerankerActivation::skipped(
            ProjectContextRetrievalPhaseSkipReason::EmptyRerankIntent,
        );
    }
    match reranker {
        Some(boundary) => match boundary.readiness {
            ProjectContextRerankerReadiness::Ready => ProjectContextRerankerActivation {
                reranker: Some(boundary.reranker),
                initial_status: None,
            },
            ProjectContextRerankerReadiness::Unavailable(_) => {
                ProjectContextRerankerActivation::unavailable(
                    ProjectContextRetrievalPhaseUnavailableReason::RerankerNotReady,
                )
            }
        },
        None => ProjectContextRerankerActivation::unavailable(
            ProjectContextRetrievalPhaseUnavailableReason::MissingReranker,
        ),
    }
}

struct ProjectContextVectorActivation<'a> {
    vector_query: Option<&'a VectorQuery>,
    vector_index: Option<&'a dyn VectorIndex>,
    initial_status: Option<ProjectContextRetrievalPhaseStatus>,
}

impl ProjectContextVectorActivation<'_> {
    fn unavailable(reason: ProjectContextRetrievalPhaseUnavailableReason) -> Self {
        Self {
            vector_query: None,
            vector_index: None,
            initial_status: Some(ProjectContextRetrievalPhaseStatus::Unavailable(reason)),
        }
    }

    fn invalid(reason: ProjectContextRetrievalPhaseInvalidReason) -> Self {
        Self {
            vector_query: None,
            vector_index: None,
            initial_status: Some(ProjectContextRetrievalPhaseStatus::Invalid(reason)),
        }
    }

    fn status(&self, selected_results: &[RetrievalResult]) -> ProjectContextRetrievalPhaseStatus {
        self.initial_status
            .clone()
            .unwrap_or_else(|| ProjectContextRetrievalPhaseStatus::Active {
                result_count: selected_results
                    .iter()
                    .filter(|result| result.score_parts.contains_key("vector"))
                    .count(),
            })
    }
}

struct ProjectContextRerankerActivation<'a> {
    reranker: Option<&'a dyn Reranker>,
    initial_status: Option<ProjectContextRetrievalPhaseStatus>,
}

impl ProjectContextRerankerActivation<'_> {
    fn unavailable(reason: ProjectContextRetrievalPhaseUnavailableReason) -> Self {
        Self {
            reranker: None,
            initial_status: Some(ProjectContextRetrievalPhaseStatus::Unavailable(reason)),
        }
    }

    fn skipped(reason: ProjectContextRetrievalPhaseSkipReason) -> Self {
        Self {
            reranker: None,
            initial_status: Some(ProjectContextRetrievalPhaseStatus::Skipped(reason)),
        }
    }

    fn status(&self, selected_results: &[RetrievalResult]) -> ProjectContextRetrievalPhaseStatus {
        self.initial_status
            .clone()
            .unwrap_or_else(|| ProjectContextRetrievalPhaseStatus::Active {
                result_count: selected_results
                    .iter()
                    .filter(|result| result.score_parts.contains_key("rerank"))
                    .count(),
            })
    }
}

fn resolve_readback_evidence_target(
    manifest: &ProjectManifest,
    evidence: &ContextPackEvidence,
) -> Option<ManifestEvidenceTarget> {
    if let Some(target) = resolve_manifest_evidence_target(manifest, &evidence.id) {
        return Some(target);
    }
    let start_line = evidence.provenance.start_line?;
    let end_line = evidence.provenance.end_line?;
    if start_line == 0 || start_line > end_line {
        return None;
    }
    let file = manifest
        .files
        .iter()
        .find(|file| file.path == evidence.provenance.path)?;
    if end_line > file.lines {
        return None;
    }
    Some(ManifestEvidenceTarget {
        path: file.path.clone(),
        line_range: Some((start_line, end_line)),
        content_hash: file.content_hash.clone(),
    })
}

fn stable_return_order(evidence: &[ContextPackEvidence]) -> Vec<ProjectContextReturnOrderItem> {
    let mut items = evidence
        .iter()
        .map(|evidence| ProjectContextReturnOrderItem {
            evidence_id: evidence.id.clone(),
            relevance: evidence.score,
        })
        .collect::<Vec<_>>();
    items.sort_by(|left, right| {
        right
            .relevance
            .total_cmp(&left.relevance)
            .then_with(|| left.evidence_id.cmp(&right.evidence_id))
    });
    items
}

fn compare_retrieval_results_for_return(
    left: &RetrievalResult,
    right: &RetrievalResult,
) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.id.cmp(&right.id))
}

fn validate_manifest_relative_path(path: &str) -> Result<()> {
    if path.is_empty() || path.trim().is_empty() {
        bail!("project context read path must not be empty");
    }
    if path.contains('\\') || path.contains('\0') {
        bail!("project context read path contains unsupported separators");
    }
    let parsed = Path::new(path);
    if parsed.is_absolute() {
        bail!("project context read path must be relative");
    }
    for segment in path.split('/') {
        if segment.is_empty() {
            bail!("project context read path contains empty component");
        }
        if segment == "." || segment == ".." {
            bail!("project context read path contains traversal");
        }
    }
    let mut has_component = false;
    for component in parsed.components() {
        match component {
            Component::Normal(value) if !value.is_empty() => {
                has_component = true;
            }
            Component::Normal(_) => {
                bail!("project context read path contains empty component");
            }
            Component::CurDir
            | Component::ParentDir
            | Component::RootDir
            | Component::Prefix(_) => {
                bail!("project context read path contains traversal");
            }
        }
    }
    if !has_component {
        bail!("project context read path must contain a normal component");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use anyhow::Result;
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::indexer::tests::fixture_project;
    use crate::types::{
        CargoDependencyDeclaration, CargoDependencyKind, CargoPackageDependency,
        CargoPackageMetadata, CargoTargetDeclaration, CargoTargetKind, CargoTargetMetadata,
        FreshnessState, Language, Provenance, RerankCandidate, RerankScore, SourceFile,
    };
    use crate::vector::DeterministicVectorIndex;
    use crate::{
        ProjectIndexer, ReplayActivatedEvidenceRef, ReplayActivationBoundary, ReplayActivationCaps,
        ReplayActivationDiagnostics, ReplayActivationFingerprintInputs,
        ReplayEvidenceReadbackStatus, ReplayEvidenceTargetKind, ShardManifest, SymbolKind,
        SymbolNode, fingerprint,
    };

    #[test]
    fn planner_builds_whole_file_read_request_for_cargo_metadata_evidence() {
        let setup = cargo_plan_manifest();
        let request = ProjectContextRetrievalRequest::new(
            "fixture_app package",
            1,
            ProjectContextPathScope::default(),
            false,
        );

        let actual = plan_project_context_retrieval(&setup, &freshness(&setup), request);
        let expected = Some(("Cargo.toml".to_string(), 1u32, 16u32));
        assert_eq!(
            match actual {
                ProjectContextRetrievalPlanningOutcome::Plan(plan) =>
                    plan.read_requests.first().map(|request| {
                        (
                            request.relative_manifest_path().to_string(),
                            request.start_line,
                            request.end_line,
                        )
                    }),
                ProjectContextRetrievalPlanningOutcome::Refusal(refusal) => {
                    panic!("unexpected refusal: {:?}", refusal)
                }
            },
            expected,
        );
    }

    #[test]
    fn planner_consumes_only_activation_boundary_for_replay_evidence() {
        let setup = cargo_plan_manifest();
        let source = setup.files.first().expect("fixture file should exist");
        let boundary = ReplayActivationBoundary {
            manifest_hash: setup.manifest_hash.clone(),
            active_refs: vec![ReplayActivatedEvidenceRef {
                artifact_id: "artifact".to_string(),
                evidence_id: "Cargo.toml".to_string(),
                evidence_path: "Cargo.toml".to_string(),
                start_line: 1,
                end_line: 16,
                score_kind: crate::EvidenceReplayScoreKind::DirectEvidence,
                score: 42.0,
                provenance_source: "indexer".to_string(),
                target_kind: ReplayEvidenceTargetKind::ManifestSource,
                canonical_target_id: fingerprint("target"),
                fingerprint_inputs: ReplayActivationFingerprintInputs {
                    manifest_hash: setup.manifest_hash.clone(),
                    source_content_hash: source.content_hash.clone(),
                    line_range_fingerprint: fingerprint("Cargo.toml:1-16"),
                    target_kind: ReplayEvidenceTargetKind::ManifestSource,
                    target_id: fingerprint("target"),
                },
                readback_status: ReplayEvidenceReadbackStatus::Verified,
            }],
            issues: Vec::new(),
            diagnostics: ReplayActivationDiagnostics {
                candidate_count: 1,
                active_count: 1,
                excluded_count: 0,
                excluded_by_reason: BTreeMap::new(),
                caps: ReplayActivationCaps::default(),
                stable_ordering: "test".to_string(),
                activation_fingerprint: fingerprint("activation"),
            },
        };
        let request = ProjectContextRetrievalRequest::new(
            "no lexical match for replay-only boundary",
            1,
            ProjectContextPathScope::default(),
            false,
        );

        let actual = plan_project_context_retrieval_with_options(
            &setup,
            &freshness(&setup),
            request,
            ProjectContextRetrievalOptions {
                replay_activation: Some(&boundary),
                ..ProjectContextRetrievalOptions::default()
            },
        );
        let expected = vec!["Cargo.toml".to_string()];

        assert_eq!(planned_paths(actual), expected);
    }

    #[test]
    fn planner_preserves_replay_route_identity_for_same_evidence_id() {
        let setup = cargo_plan_manifest();
        let source = setup.files.first().expect("fixture file should exist");
        let boundary = ReplayActivationBoundary {
            manifest_hash: setup.manifest_hash.clone(),
            active_refs: vec![
                ReplayActivatedEvidenceRef {
                    artifact_id: "unlinked-artifact".to_string(),
                    evidence_id: "Cargo.toml".to_string(),
                    evidence_path: "Cargo.toml".to_string(),
                    start_line: 1,
                    end_line: 16,
                    score_kind: crate::EvidenceReplayScoreKind::DirectEvidence,
                    score: 2.0,
                    provenance_source: "indexer".to_string(),
                    target_kind: ReplayEvidenceTargetKind::ManifestSource,
                    canonical_target_id: fingerprint("manifest-source-target"),
                    fingerprint_inputs: ReplayActivationFingerprintInputs {
                        manifest_hash: setup.manifest_hash.clone(),
                        source_content_hash: source.content_hash.clone(),
                        line_range_fingerprint: fingerprint("Cargo.toml:1-16"),
                        target_kind: ReplayEvidenceTargetKind::ManifestSource,
                        target_id: fingerprint("manifest-source-target"),
                    },
                    readback_status: ReplayEvidenceReadbackStatus::Verified,
                },
                ReplayActivatedEvidenceRef {
                    artifact_id: "linked-artifact".to_string(),
                    evidence_id: "Cargo.toml".to_string(),
                    evidence_path: "Cargo.toml".to_string(),
                    start_line: 1,
                    end_line: 16,
                    score_kind: crate::EvidenceReplayScoreKind::DirectEvidence,
                    score: 1.0,
                    provenance_source: "indexer".to_string(),
                    target_kind: ReplayEvidenceTargetKind::ToolObservation,
                    canonical_target_id: fingerprint("tool-observation-target"),
                    fingerprint_inputs: ReplayActivationFingerprintInputs {
                        manifest_hash: setup.manifest_hash.clone(),
                        source_content_hash: source.content_hash.clone(),
                        line_range_fingerprint: fingerprint("Cargo.toml:1-16"),
                        target_kind: ReplayEvidenceTargetKind::ToolObservation,
                        target_id: fingerprint("tool-observation-target"),
                    },
                    readback_status: ReplayEvidenceReadbackStatus::Verified,
                },
            ],
            issues: Vec::new(),
            diagnostics: ReplayActivationDiagnostics {
                candidate_count: 2,
                active_count: 2,
                excluded_count: 0,
                excluded_by_reason: BTreeMap::new(),
                caps: ReplayActivationCaps::default(),
                stable_ordering: "test".to_string(),
                activation_fingerprint: fingerprint("activation"),
            },
        };
        let request = ProjectContextRetrievalRequest::new(
            "no lexical match for replay-only route identity boundary",
            2,
            ProjectContextPathScope::default(),
            false,
        );

        let actual = expect_plan(plan_project_context_retrieval_with_options(
            &setup,
            &freshness(&setup),
            request,
            ProjectContextRetrievalOptions {
                replay_activation: Some(&boundary),
                ..ProjectContextRetrievalOptions::default()
            },
        ));
        let expected = 2usize;

        assert_eq!(actual.selected_results.len(), expected);
    }

    #[test]
    fn artifact_queries_surface_lexical_candidates_and_validated_readback_requests() -> Result<()> {
        let fixture = tempfile::TempDir::new()?;
        let root = fixture.path().join("project");
        std::fs::create_dir_all(root.join("src"))?;
        std::fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"context_engine\"\nversion = \"0.1.0\"\n",
        )?;
        std::fs::write(
            root.join("AGENTS.md"),
            "# Target goal\nworkspace context engine\n",
        )?;
        std::fs::write(root.join("settings.json"), "{\"schema\":true}\n")?;
        std::fs::write(root.join("src").join("ui.rs"), "pub fn renderer() {}\n")?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let queries = [
            ("AGENTS target goal", "AGENTS.md"),
            ("workspace context engine", "Cargo.toml"),
            ("TUI renderer", "src/ui.rs"),
            ("config schema", "settings.json"),
        ];

        let actual = queries
            .iter()
            .map(|(query, expected_path)| {
                let request = ProjectContextRetrievalRequest::new(
                    (*query).to_string(),
                    5,
                    ProjectContextPathScope::default(),
                    false,
                );
                let plan = expect_plan(plan_project_context_retrieval(
                    &manifest,
                    &freshness(&manifest),
                    request,
                ));
                (
                    query.to_string(),
                    plan.selected_results.iter().any(|result| {
                        result.id.starts_with("artifact:") && result.path == *expected_path
                    }),
                    plan.read_requests.iter().any(|request| {
                        request.evidence_id.starts_with("artifact:")
                            && request.relative_manifest_path() == *expected_path
                            && request.start_line == 1
                            && request.end_line >= request.start_line
                    }),
                )
            })
            .collect::<Vec<_>>();
        let expected = queries
            .iter()
            .map(|(query, _)| (query.to_string(), true, true))
            .collect::<Vec<_>>();

        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn empty_artifact_files_still_produce_validated_readback_requests() -> Result<()> {
        let fixture = tempfile::TempDir::new()?;
        let root = fixture.path().join("project");
        std::fs::create_dir_all(root.join("src"))?;
        std::fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"empty-artifact\"\nversion = \"0.1.0\"\n",
        )?;
        std::fs::write(root.join("settings.json"), "")?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let request = ProjectContextRetrievalRequest::new(
            "config schema".to_string(),
            5,
            ProjectContextPathScope::default(),
            false,
        );

        let actual = expect_plan(plan_project_context_retrieval(
            &manifest,
            &freshness(&manifest),
            request,
        ));
        let expected = Some(("settings.json".to_string(), 1u32, 1u32));

        assert_eq!(
            actual
                .read_requests
                .iter()
                .find(|request| {
                    request.evidence_id.starts_with("artifact:")
                        && request.relative_manifest_path() == "settings.json"
                })
                .map(|request| {
                    (
                        request.relative_manifest_path().to_string(),
                        request.start_line,
                        request.end_line,
                    )
                }),
            expected,
        );
        Ok(())
    }

    #[test]
    fn invalid_or_suspicious_cargo_metadata_ids_do_not_produce_read_requests() {
        let setup = cargo_plan_manifest();
        let candidates = [
            "cargo:dependency:../x",
            "cargo:v1:dependency:../x",
            "cargo:v1:unknown:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "cargo:v1:dependency:raw:separator:collision",
        ];
        let actual = candidates
            .into_iter()
            .map(|id| resolve_manifest_evidence_target(&setup, id).is_none())
            .collect::<Vec<_>>();
        let expected = vec![true, true, true, true];
        assert_eq!(actual, expected);
    }

    #[test]
    fn stale_freshness_refusal_is_unchanged_for_cargo_metadata_hits() {
        let setup = cargo_plan_manifest();
        let evaluation = ManifestFreshnessEvaluation {
            state: FreshnessState {
                changed: vec!["Cargo.toml".to_string()],
                fresh: true,
                ..fresh_state(&setup)
            },
            proof_level: FreshnessProofLevel::FullFilesystem,
        };
        let request = ProjectContextRetrievalRequest::new(
            "fixture_app package",
            1,
            ProjectContextPathScope::default(),
            false,
        );

        let actual = plan_project_context_retrieval(&setup, &evaluation, request);
        let expected = Some(ProjectContextRetrievalRefusalCode::StaleEvidenceRejected);
        assert_eq!(refusal_code(actual), expected);
    }

    #[test]
    fn cargo_like_normal_ids_cannot_shadow_valid_metadata_ids() {
        let mut setup = cargo_plan_manifest();
        setup.files.push(SourceFile {
            path: "src/cargo.rs".to_string(),
            language: Language::Rust,
            bytes: 10,
            lines: 2,
            content_hash: fingerprint("cargo-like-source"),
            provenance: provenance("src/cargo.rs", Some(1), Some(2), "test", "file"),
        });
        setup.symbols.push(SymbolNode {
            id: "symbol:src/cargo.rs:Function:cargo:v1:package:fake".to_string(),
            name: "cargo:v1:package:fake".to_string(),
            kind: SymbolKind::Function,
            path: "src/cargo.rs".to_string(),
            parent: None,
            start_line: 1,
            end_line: 2,
            provenance: provenance("src/cargo.rs", Some(1), Some(2), "test", "symbol"),
        });
        setup.shards.push(ShardManifest {
            id: "shard:cargo:v1:package:fake".to_string(),
            path: "src/cargo.rs".to_string(),
            start_line: 1,
            end_line: 2,
            content_hash: fingerprint("cargo-like-shard"),
            symbol_ids: Vec::new(),
            provenance: provenance("src/cargo.rs", Some(1), Some(2), "test", "shard"),
        });

        let actual = (
            resolve_manifest_evidence_target(&setup, "cargo:v1:package:fake").is_none(),
            resolve_manifest_evidence_target(
                &setup,
                "symbol:src/cargo.rs:Function:cargo:v1:package:fake",
            )
            .map(|target| target.path),
            resolve_manifest_evidence_target(&setup, "shard:cargo:v1:package:fake")
                .map(|target| target.path),
        );
        let expected = (
            true,
            Some("src/cargo.rs".to_string()),
            Some("src/cargo.rs".to_string()),
        );
        assert_eq!(actual, expected);
    }
    #[test]
    fn planner_applies_prefix_and_suffix_before_truncation_without_underfill() -> Result<()> {
        let fixture = tempfile::TempDir::new()?;
        let root = fixture.path().join("project");
        std::fs::create_dir_all(root.join("src").join("in"))?;
        std::fs::create_dir_all(root.join("src").join("out"))?;
        std::fs::write(
            root.join("src").join("in").join("target.rs"),
            "pub fn target() {\n    let _ = \"scopedneedle\";\n}\n",
        )?;
        std::fs::write(
            root.join("src").join("out").join("loud.rs"),
            "pub fn loud() {\n    let _ = \"scopedneedle scopedneedle scopedneedle scopedneedle scopedneedle\";\n}\n",
        )?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let request = ProjectContextRetrievalRequest::new(
            "target Function",
            1,
            ProjectContextPathScope::new(Some("src/in/".to_string()), vec![".rs".to_string()]),
            true,
        );

        let actual = plan_project_context_retrieval(&manifest, &freshness(&manifest), request);
        let expected = vec!["src/in/target.rs".to_string()];
        assert_eq!(planned_paths(actual), expected);
        Ok(())
    }

    #[test]
    fn planner_preserves_limit_zero_as_retrieval_default_limit() -> Result<()> {
        let (_fixture, root, manifest) = indexed_fixture()?;
        let request = ProjectContextRetrievalRequest::new(
            "Root",
            0,
            ProjectContextPathScope::default(),
            true,
        );

        let actual = plan_project_context_retrieval(&manifest, &freshness(&manifest), request);
        let expected = (
            ProjectContextWriteDecision::WriteContextPackAfterReadback,
            10usize,
            false,
        );
        let actual = match actual {
            ProjectContextRetrievalPlanningOutcome::Plan(plan) => {
                assert!(plan.selected_results.len() <= 10);
                assert!(!plan.read_requests.is_empty());
                (
                    plan.write_decision,
                    plan.query_diagnostics.limit,
                    plan.context_pack.is_none(),
                )
            }
            ProjectContextRetrievalPlanningOutcome::Refusal(refusal) => {
                panic!("unexpected refusal: {:?}", refusal)
            }
        };
        assert_eq!(actual, expected);
        assert!(root.is_dir());
        Ok(())
    }

    #[test]
    fn planner_refuses_indexed_files_only_freshness_even_when_state_is_fresh() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let evaluation = ManifestFreshnessEvaluation {
            state: fresh_state(&manifest),
            proof_level: FreshnessProofLevel::IndexedFilesOnly,
        };
        let request = ProjectContextRetrievalRequest::new(
            "Root",
            3,
            ProjectContextPathScope::default(),
            true,
        );

        let actual = plan_project_context_retrieval(&manifest, &evaluation, request);
        let expected = Some(ProjectContextRetrievalRefusalCode::ManifestNotInjectable);
        assert_eq!(refusal_code(actual), expected);
        Ok(())
    }

    #[test]
    fn planner_refuses_changed_stale_evidence_under_reject_policy() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let evaluation = ManifestFreshnessEvaluation {
            state: FreshnessState {
                changed: vec!["src/lib.rs".to_string()],
                fresh: true,
                ..fresh_state(&manifest)
            },
            proof_level: FreshnessProofLevel::FullFilesystem,
        };
        let request = ProjectContextRetrievalRequest::new(
            "Root",
            3,
            ProjectContextPathScope::default(),
            true,
        );

        let actual = plan_project_context_retrieval(&manifest, &evaluation, request);
        let expected = Some(ProjectContextRetrievalRefusalCode::StaleEvidenceRejected);
        assert_eq!(refusal_code(actual), expected);
        Ok(())
    }

    #[test]
    fn planner_refuses_deleted_stale_evidence_under_reject_policy() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let evaluation = ManifestFreshnessEvaluation {
            state: FreshnessState {
                deleted: vec!["src/lib.rs".to_string()],
                fresh: true,
                ..fresh_state(&manifest)
            },
            proof_level: FreshnessProofLevel::FullFilesystem,
        };
        let request = ProjectContextRetrievalRequest::new(
            "Root",
            3,
            ProjectContextPathScope::default(),
            true,
        );

        let actual = plan_project_context_retrieval(&manifest, &evaluation, request);
        let expected = Some(ProjectContextRetrievalRefusalCode::StaleEvidenceRejected);
        assert_eq!(refusal_code(actual), expected);
        Ok(())
    }

    #[test]
    fn planner_empty_retrieval_gives_no_write_no_read_plan() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let request = ProjectContextRetrievalRequest::new(
            "absent-token-for-empty-plan",
            5,
            ProjectContextPathScope::new(
                Some("absent/scope/".to_string()),
                vec![".rs".to_string()],
            ),
            true,
        );

        let actual = plan_project_context_retrieval(&manifest, &freshness(&manifest), request);
        let expected = (
            ProjectContextWriteDecision::NoWriteEmptyRetrieval,
            0usize,
            true,
        );
        assert_eq!(
            match actual {
                ProjectContextRetrievalPlanningOutcome::Plan(plan) => (
                    plan.write_decision,
                    plan.read_requests.len(),
                    plan.context_pack.is_none(),
                ),
                ProjectContextRetrievalPlanningOutcome::Refusal(refusal) => {
                    panic!("unexpected refusal: {:?}", refusal)
                }
            },
            expected,
        );
        Ok(())
    }

    #[test]
    fn default_options_preserve_no_vector_planner_fixture() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let request = ProjectContextRetrievalRequest::new(
            "Root model",
            5,
            ProjectContextPathScope::default(),
            true,
        );

        let actual_default =
            plan_project_context_retrieval(&manifest, &freshness(&manifest), request.clone());
        let actual_options = plan_project_context_retrieval_with_options(
            &manifest,
            &freshness(&manifest),
            request,
            ProjectContextRetrievalOptions::default(),
        );
        let actual = plan_snapshot(actual_options);
        let expected = plan_snapshot(actual_default);
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn vector_hits_participate_when_query_vector_and_ready_index_are_supplied() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let widget_symbol = manifest
            .symbols
            .iter()
            .find(|symbol| symbol.name == "Widget")
            .expect("fixture should include Widget symbol");
        let vector_index = DeterministicVectorIndex::new(BTreeMap::from([(
            widget_symbol.id.clone(),
            vec![1.0, 0.0],
        )]));
        let vector_query = VectorQuery { embedding: vec![1.0, 0.0] };
        let request = ProjectContextRetrievalRequest::new(
            "lexicalmissneedle",
            1,
            ProjectContextPathScope::default(),
            false,
        );

        let actual = plan_project_context_retrieval_with_options(
            &manifest,
            &freshness(&manifest),
            request,
            ProjectContextRetrievalOptions {
                vector_query: Some(&vector_query),
                vector_index: Some(ready_vector_boundary(&vector_index, 2)),
                reranker: None,
                vector_unavailable_reason: None,
                exact_fact_status: None,
                replay_activation: None,
            },
        );
        let plan = expect_plan(actual);
        let expected = (
            Some(widget_symbol.id.clone()),
            ProjectContextRetrievalPhaseStatus::Active { result_count: 1 },
            Some(1.0),
        );
        assert_eq!(
            (
                plan.selected_results
                    .first()
                    .map(|result| result.id.clone()),
                plan.query_diagnostics.phase_diagnostics.vector,
                plan.selected_results
                    .first()
                    .and_then(|result| result.score_parts.get("vector"))
                    .copied(),
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn missing_query_embedding_reports_unavailable_and_preserves_fallback() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let vector_index = DeterministicVectorIndex::default();
        let request = ProjectContextRetrievalRequest::new(
            "Root model",
            3,
            ProjectContextPathScope::default(),
            true,
        );

        let fallback =
            plan_project_context_retrieval(&manifest, &freshness(&manifest), request.clone());
        let actual = plan_project_context_retrieval_with_options(
            &manifest,
            &freshness(&manifest),
            request,
            ProjectContextRetrievalOptions {
                vector_query: None,
                vector_index: Some(ready_vector_boundary(&vector_index, 2)),
                reranker: None,
                vector_unavailable_reason: None,
                exact_fact_status: None,
                replay_activation: None,
            },
        );
        let actual_plan = expect_plan(actual);
        let expected = (
            plan_snapshot(fallback),
            ProjectContextRetrievalPhaseStatus::Unavailable(
                ProjectContextRetrievalPhaseUnavailableReason::MissingQueryEmbedding,
            ),
        );
        assert_eq!(
            (
                plan_snapshot(ProjectContextRetrievalPlanningOutcome::Plan(
                    actual_plan.clone()
                )),
                actual_plan.query_diagnostics.phase_diagnostics.vector,
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn missing_vector_index_reports_unavailable_and_preserves_fallback() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let vector_query = VectorQuery { embedding: vec![1.0, 0.0] };
        let request = ProjectContextRetrievalRequest::new(
            "Root model",
            3,
            ProjectContextPathScope::default(),
            true,
        );

        let fallback =
            plan_project_context_retrieval(&manifest, &freshness(&manifest), request.clone());
        let actual = plan_project_context_retrieval_with_options(
            &manifest,
            &freshness(&manifest),
            request,
            ProjectContextRetrievalOptions {
                vector_query: Some(&vector_query),
                vector_index: None,
                reranker: None,
                vector_unavailable_reason: None,
                exact_fact_status: None,
                replay_activation: None,
            },
        );
        let actual_plan = expect_plan(actual);
        let expected = (
            plan_snapshot(fallback),
            ProjectContextRetrievalPhaseStatus::Unavailable(
                ProjectContextRetrievalPhaseUnavailableReason::MissingVectorIndex,
            ),
        );
        assert_eq!(
            (
                plan_snapshot(ProjectContextRetrievalPlanningOutcome::Plan(
                    actual_plan.clone()
                )),
                actual_plan.query_diagnostics.phase_diagnostics.vector,
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn vector_dimension_mismatch_reports_invalid_and_does_not_succeed_silently() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let vector_index = DeterministicVectorIndex::default();
        let vector_query = VectorQuery { embedding: vec![1.0, 0.0] };
        let request = ProjectContextRetrievalRequest::new(
            "Root model",
            3,
            ProjectContextPathScope::default(),
            true,
        );

        let fallback =
            plan_project_context_retrieval(&manifest, &freshness(&manifest), request.clone());
        let actual = plan_project_context_retrieval_with_options(
            &manifest,
            &freshness(&manifest),
            request,
            ProjectContextRetrievalOptions {
                vector_query: Some(&vector_query),
                vector_index: Some(ready_vector_boundary(&vector_index, 3)),
                reranker: None,
                vector_unavailable_reason: None,
                exact_fact_status: None,
                replay_activation: None,
            },
        );
        let actual_plan = expect_plan(actual);
        let expected = (
            plan_snapshot(fallback),
            ProjectContextRetrievalPhaseStatus::Invalid(
                ProjectContextRetrievalPhaseInvalidReason::VectorDimensionMismatch {
                    query_dimension: 2,
                    index_dimension: 3,
                },
            ),
            false,
        );
        assert_eq!(
            (
                plan_snapshot(ProjectContextRetrievalPlanningOutcome::Plan(
                    actual_plan.clone()
                )),
                actual_plan.query_diagnostics.phase_diagnostics.vector,
                actual_plan
                    .selected_results
                    .iter()
                    .any(|result| result.score_parts.contains_key("vector")),
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn empty_query_embedding_reports_invalid_even_without_vector_index() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let vector_query = VectorQuery { embedding: Vec::new() };
        let request = ProjectContextRetrievalRequest::new(
            "Root model",
            3,
            ProjectContextPathScope::default(),
            true,
        );

        let actual = plan_project_context_retrieval_with_options(
            &manifest,
            &freshness(&manifest),
            request,
            ProjectContextRetrievalOptions {
                vector_query: Some(&vector_query),
                vector_index: None,
                reranker: None,
                vector_unavailable_reason: None,
                exact_fact_status: None,
                replay_activation: None,
            },
        );
        let actual_plan = expect_plan(actual);
        let expected = ProjectContextRetrievalPhaseStatus::Invalid(
            ProjectContextRetrievalPhaseInvalidReason::EmptyQueryEmbedding,
        );
        assert_eq!(
            actual_plan.query_diagnostics.phase_diagnostics.vector,
            expected
        );
        Ok(())
    }

    #[test]
    fn non_finite_query_embedding_is_rejected_instead_of_reported_active() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let root_symbol = manifest
            .symbols
            .iter()
            .find(|symbol| symbol.name == "Root")
            .expect("fixture should include Root symbol");
        let vector_index = DeterministicVectorIndex::new(BTreeMap::from([(
            root_symbol.id.clone(),
            vec![1.0, 0.0],
        )]));
        let vector_query = VectorQuery { embedding: vec![f32::NAN, 0.0] };
        let request = ProjectContextRetrievalRequest::new(
            "Root model",
            3,
            ProjectContextPathScope::default(),
            true,
        );

        let actual = plan_project_context_retrieval_with_options(
            &manifest,
            &freshness(&manifest),
            request,
            ProjectContextRetrievalOptions {
                vector_query: Some(&vector_query),
                vector_index: Some(ready_vector_boundary(&vector_index, 2)),
                reranker: None,
                vector_unavailable_reason: None,
                exact_fact_status: None,
                replay_activation: None,
            },
        );
        let actual_plan = expect_plan(actual);
        let expected = ProjectContextRetrievalPhaseStatus::Invalid(
            ProjectContextRetrievalPhaseInvalidReason::NonFiniteQueryEmbedding,
        );

        assert_eq!(
            actual_plan.query_diagnostics.phase_diagnostics.vector, expected,
            "non-finite vector queries must be reported as invalid, not active with empty hits"
        );
        Ok(())
    }

    #[test]
    fn overflow_query_embedding_norm_is_rejected_instead_of_reported_active() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let root_symbol = manifest
            .symbols
            .iter()
            .find(|symbol| symbol.name == "Root")
            .expect("fixture should include Root symbol");
        let vector_index = DeterministicVectorIndex::new(BTreeMap::from([(
            root_symbol.id.clone(),
            vec![1.0, 0.0],
        )]));
        let vector_query = VectorQuery { embedding: vec![f32::MAX, 0.0] };
        let request = ProjectContextRetrievalRequest::new(
            "Root model",
            3,
            ProjectContextPathScope::default(),
            true,
        );

        let actual = plan_project_context_retrieval_with_options(
            &manifest,
            &freshness(&manifest),
            request,
            ProjectContextRetrievalOptions {
                vector_query: Some(&vector_query),
                vector_index: Some(ready_vector_boundary(&vector_index, 2)),
                reranker: None,
                vector_unavailable_reason: None,
                exact_fact_status: None,
                replay_activation: None,
            },
        );
        let actual_plan = expect_plan(actual);
        let expected = ProjectContextRetrievalPhaseStatus::Invalid(
            ProjectContextRetrievalPhaseInvalidReason::NonFiniteQueryEmbedding,
        );

        assert_eq!(
            actual_plan.query_diagnostics.phase_diagnostics.vector, expected,
            "finite values whose norm arithmetic overflows must be invalid, not active with empty hits"
        );
        Ok(())
    }

    #[test]
    fn zero_norm_query_embedding_is_rejected_instead_of_reported_active() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let root_symbol = manifest
            .symbols
            .iter()
            .find(|symbol| symbol.name == "Root")
            .expect("fixture should include Root symbol");
        let vector_index = DeterministicVectorIndex::new(BTreeMap::from([(
            root_symbol.id.clone(),
            vec![1.0, 0.0],
        )]));
        let vector_query = VectorQuery { embedding: vec![0.0, -0.0] };
        let request = ProjectContextRetrievalRequest::new(
            "Root model",
            3,
            ProjectContextPathScope::default(),
            true,
        );

        let actual = plan_project_context_retrieval_with_options(
            &manifest,
            &freshness(&manifest),
            request,
            ProjectContextRetrievalOptions {
                vector_query: Some(&vector_query),
                vector_index: Some(ready_vector_boundary(&vector_index, 2)),
                reranker: None,
                vector_unavailable_reason: None,
                exact_fact_status: None,
                replay_activation: None,
            },
        );
        let actual_plan = expect_plan(actual);
        let expected = ProjectContextRetrievalPhaseStatus::Invalid(
            ProjectContextRetrievalPhaseInvalidReason::ZeroQueryEmbeddingNorm,
        );

        assert_eq!(
            actual_plan.query_diagnostics.phase_diagnostics.vector,
            expected
        );
        Ok(())
    }

    #[test]
    fn vector_selector_unavailable_reason_is_preserved_as_typed_phase() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let vector_query = VectorQuery { embedding: vec![1.0, 0.0] };
        let request = ProjectContextRetrievalRequest::new(
            "Root model",
            3,
            ProjectContextPathScope::default(),
            true,
        );

        let actual = plan_project_context_retrieval_with_options(
            &manifest,
            &freshness(&manifest),
            request,
            ProjectContextRetrievalOptions {
                vector_query: Some(&vector_query),
                vector_index: None,
                reranker: None,
                vector_unavailable_reason: Some(
                    ProjectContextVectorUnavailableReason::AmbiguousVectorIndex,
                ),
                exact_fact_status: None,
                replay_activation: None,
            },
        );
        let actual_plan = expect_plan(actual);
        let expected = ProjectContextRetrievalPhaseStatus::Unavailable(
            ProjectContextRetrievalPhaseUnavailableReason::AmbiguousVectorIndex,
        );
        assert_eq!(
            actual_plan.query_diagnostics.phase_diagnostics.vector,
            expected
        );
        Ok(())
    }

    #[test]
    fn planner_preserves_use_case_and_applies_top_k_candidate_budget() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let request = ProjectContextRetrievalRequest::new(
            "Root model",
            1,
            ProjectContextPathScope::default(),
            true,
        )
        .with_use_case("ranked caller proof")
        .with_top_k(4);

        let actual = plan_project_context_retrieval_with_options(
            &manifest,
            &freshness(&manifest),
            request,
            ProjectContextRetrievalOptions::default(),
        );
        let actual_plan = expect_plan(actual);
        let expected = (
            Some("ranked caller proof".to_string()),
            Some(4usize),
            ProjectContextTopKStatus::Applied { candidate_count: 4 },
            1usize,
        );
        assert_eq!(
            (
                actual_plan.query_diagnostics.use_case,
                actual_plan.query_diagnostics.top_k,
                actual_plan.query_diagnostics.top_k_status,
                actual_plan.selected_results.len(),
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn planner_selects_explicit_use_case_as_redaction_safe_rerank_intent() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let request = ProjectContextRetrievalRequest::new(
            "Root model",
            3,
            ProjectContextPathScope::default(),
            true,
        )
        .with_use_case("  ranked caller proof  ");

        let actual = plan_project_context_retrieval_with_options(
            &manifest,
            &freshness(&manifest),
            request,
            ProjectContextRetrievalOptions {
                vector_query: None,
                vector_index: None,
                reranker: Some(ready_reranker_boundary(&IntentOrderReranker)),
                vector_unavailable_reason: None,
                exact_fact_status: None,
                replay_activation: None,
            },
        );
        let actual_plan = expect_plan(actual);
        let expected = (
            Some("ranked caller proof".to_string()),
            Some(RerankIntentSource::ExplicitUseCase),
            Some(fingerprint("ranked caller proof")),
            Some("ranked caller proof".len()),
        );
        assert_eq!(
            (
                actual_plan.query_diagnostics.use_case,
                actual_plan.query_diagnostics.rerank_intent_source,
                actual_plan.query_diagnostics.rerank_intent_fingerprint,
                actual_plan.query_diagnostics.rerank_intent_len,
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn whitespace_use_case_falls_back_to_query_text_rerank_intent() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let request = ProjectContextRetrievalRequest::new(
            "Root model",
            3,
            ProjectContextPathScope::default(),
            true,
        )
        .with_use_case("   \n\t ");

        let actual = plan_project_context_retrieval_with_options(
            &manifest,
            &freshness(&manifest),
            request,
            ProjectContextRetrievalOptions {
                vector_query: None,
                vector_index: None,
                reranker: Some(ready_reranker_boundary(&IntentOrderReranker)),
                vector_unavailable_reason: None,
                exact_fact_status: None,
                replay_activation: None,
            },
        );
        let actual_plan = expect_plan(actual);
        let expected = (
            None,
            Some(RerankIntentSource::QueryTextFallback),
            Some(fingerprint("Root model")),
            ProjectContextRetrievalPhaseStatus::Active { result_count: 3 },
        );
        assert_eq!(
            (
                actual_plan.query_diagnostics.use_case,
                actual_plan.query_diagnostics.rerank_intent_source,
                actual_plan.query_diagnostics.rerank_intent_fingerprint,
                actual_plan.query_diagnostics.phase_diagnostics.rerank,
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn empty_query_and_empty_use_case_skip_reranker_with_typed_reason() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let request =
            ProjectContextRetrievalRequest::new("   ", 3, ProjectContextPathScope::default(), true)
                .with_use_case("   ");

        let actual = plan_project_context_retrieval_with_options(
            &manifest,
            &freshness(&manifest),
            request,
            ProjectContextRetrievalOptions {
                vector_query: None,
                vector_index: None,
                reranker: Some(ready_reranker_boundary(&IntentOrderReranker)),
                vector_unavailable_reason: None,
                exact_fact_status: None,
                replay_activation: None,
            },
        );
        let actual_plan = expect_plan(actual);
        let expected = (
            None,
            None,
            ProjectContextRetrievalPhaseStatus::Skipped(
                ProjectContextRetrievalPhaseSkipReason::EmptyRerankIntent,
            ),
        );
        assert_eq!(
            (
                actual_plan.query_diagnostics.use_case,
                actual_plan.query_diagnostics.rerank_intent_source,
                actual_plan.query_diagnostics.phase_diagnostics.rerank,
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn automatic_injection_use_case_does_not_override_actual_query_intent() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let request = ProjectContextRetrievalRequest::new(
            "Root model",
            3,
            ProjectContextPathScope::default(),
            true,
        )
        .with_use_case("automatic project-model context injection")
        .automatic_injection();

        let actual = plan_project_context_retrieval_with_options(
            &manifest,
            &freshness(&manifest),
            request,
            ProjectContextRetrievalOptions {
                vector_query: None,
                vector_index: None,
                reranker: Some(ready_reranker_boundary(&IntentOrderReranker)),
                vector_unavailable_reason: None,
                exact_fact_status: None,
                replay_activation: None,
            },
        );
        let actual_plan = expect_plan(actual);
        let expected = (
            Some("automatic project-model context injection".to_string()),
            Some(RerankIntentSource::AutomaticInjectionQueryFallback),
            Some(fingerprint("Root model")),
        );
        assert_eq!(
            (
                actual_plan.query_diagnostics.use_case,
                actual_plan.query_diagnostics.rerank_intent_source,
                actual_plan.query_diagnostics.rerank_intent_fingerprint,
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn freshness_refusal_still_blocks_semantic_boundaries_before_reads_or_writes() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let vector_query = VectorQuery { embedding: vec![1.0, 0.0] };
        let root_symbol = manifest
            .symbols
            .iter()
            .find(|symbol| symbol.name == "Root")
            .expect("fixture should include Root symbol");
        let vector_index = DeterministicVectorIndex::new(BTreeMap::from([(
            root_symbol.id.clone(),
            vec![1.0, 0.0],
        )]));
        let evaluation = ManifestFreshnessEvaluation {
            state: FreshnessState {
                changed: vec!["src/lib.rs".to_string()],
                fresh: true,
                ..fresh_state(&manifest)
            },
            proof_level: FreshnessProofLevel::FullFilesystem,
        };
        let request = ProjectContextRetrievalRequest::new(
            "Root model",
            3,
            ProjectContextPathScope::default(),
            true,
        );

        let actual = plan_project_context_retrieval_with_options(
            &manifest,
            &evaluation,
            request,
            ProjectContextRetrievalOptions {
                vector_query: Some(&vector_query),
                vector_index: Some(ready_vector_boundary(&vector_index, 2)),
                reranker: None,
                vector_unavailable_reason: None,
                exact_fact_status: None,
                replay_activation: None,
            },
        );
        let actual_plan = expect_plan(actual);
        let expected = ProjectContextRetrievalPhaseStatus::Active { result_count: 0 };
        assert_eq!(
            actual_plan.query_diagnostics.phase_diagnostics.vector,
            expected
        );
        Ok(())
    }
    #[test]
    fn reranker_absence_reports_diagnostic_without_changing_fallback() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let request = ProjectContextRetrievalRequest::new(
            "Root model",
            3,
            ProjectContextPathScope::default(),
            true,
        );

        let fallback =
            plan_project_context_retrieval(&manifest, &freshness(&manifest), request.clone());
        let actual = plan_project_context_retrieval_with_options(
            &manifest,
            &freshness(&manifest),
            request,
            ProjectContextRetrievalOptions::default(),
        );
        let actual_plan = expect_plan(actual);
        let expected = (
            plan_snapshot(fallback),
            ProjectContextRetrievalPhaseStatus::Unavailable(
                ProjectContextRetrievalPhaseUnavailableReason::MissingReranker,
            ),
        );
        assert_eq!(
            (
                plan_snapshot(ProjectContextRetrievalPlanningOutcome::Plan(
                    actual_plan.clone()
                )),
                actual_plan.query_diagnostics.phase_diagnostics.rerank,
            ),
            expected,
        );
        Ok(())
    }
    #[test]
    fn read_request_plans_symbol_shard_and_file_evidence_ranges() -> Result<()> {
        let mut manifest = manual_manifest();
        manifest.symbols.push(SymbolNode {
            id: "symbol:src/lib.rs:Function:target".to_string(),
            name: "target".to_string(),
            kind: SymbolKind::Function,
            path: "src/lib.rs".to_string(),
            parent: None,
            start_line: 2,
            end_line: 4,
            provenance: provenance("src/lib.rs", Some(2), Some(4), "test", "symbol"),
        });
        manifest.shards.push(ShardManifest {
            id: "shard:src/lib.rs:5-7".to_string(),
            path: "src/lib.rs".to_string(),
            start_line: 5,
            end_line: 7,
            content_hash: fingerprint("shard"),
            symbol_ids: Vec::new(),
            provenance: provenance("src/lib.rs", Some(5), Some(7), "test", "shard"),
        });
        let selection = ContextPackSelection {
            retrieval_results: Vec::new(),
            shards: manifest.shards.clone(),
            evidence: vec![
                ContextPackEvidence {
                    id: "src/lib.rs".to_string(),
                    path: "src/lib.rs".to_string(),
                    symbol: None,
                    source: crate::ContextPackEvidenceSource::DirectEvidence,
                    freshness: crate::EvidenceFreshness::Fresh,
                    provenance: provenance("src/lib.rs", Some(1), Some(9), "test", "file"),
                    score: 1.0,
                },
                ContextPackEvidence {
                    id: "symbol:src/lib.rs:Function:target".to_string(),
                    path: "src/lib.rs".to_string(),
                    symbol: Some("target".to_string()),
                    source: crate::ContextPackEvidenceSource::DirectEvidence,
                    freshness: crate::EvidenceFreshness::Fresh,
                    provenance: provenance("src/lib.rs", Some(2), Some(4), "test", "symbol-direct"),
                    score: 3.0,
                },
            ],
            freshness: fresh_state(&manifest),
            stale_policy: StaleEvidencePolicy::Reject,
        };
        let pack = ContextPack::from_selection(&manifest, selection)?;
        let actual = pack
            .evidence
            .iter()
            .map(|evidence| {
                let target = resolve_manifest_evidence_target(&manifest, &evidence.id).unwrap();
                let (start, end) = target.line_range.unwrap();
                ProjectContextReadRequest::new(target.path, evidence.id.clone(), start, end)
                    .unwrap()
            })
            .map(|request| (request.evidence_id, request.start_line, request.end_line))
            .collect::<Vec<_>>();
        let expected = vec![
            ("shard:src/lib.rs:5-7".to_string(), 5, 7),
            ("src/lib.rs".to_string(), 1, 9),
            ("symbol:src/lib.rs:Function:target".to_string(), 2, 4),
        ];
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn read_request_rejects_path_traversal_and_absolute_or_empty_paths() {
        let setup = [
            "../secret",
            "/tmp/secret",
            "",
            ".",
            "safe/./secret",
            "safe/../secret",
            "safe\\..\\secret",
        ];
        let actual = setup
            .into_iter()
            .map(|path| ProjectContextReadRequest::new(path, "id", 1, 1).is_err())
            .collect::<Vec<_>>();
        let expected = vec![true, true, true, true, true, true, true];
        assert_eq!(actual, expected);
    }

    #[test]
    fn planner_return_order_is_stable_independent_of_pack_order() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let request = ProjectContextRetrievalRequest::new(
            "Root model",
            5,
            ProjectContextPathScope::default(),
            true,
        );

        let actual = plan_project_context_retrieval(&manifest, &freshness(&manifest), request);
        let plan = match actual {
            ProjectContextRetrievalPlanningOutcome::Plan(plan) => plan,
            ProjectContextRetrievalPlanningOutcome::Refusal(refusal) => {
                panic!("unexpected refusal: {:?}", refusal)
            }
        };
        let expected_return = {
            let mut values = plan
                .context_pack
                .as_ref()
                .unwrap()
                .evidence
                .iter()
                .map(|evidence| (evidence.id.clone(), evidence.score))
                .collect::<Vec<_>>();
            values.sort_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            values
        };
        let actual_return = plan
            .return_order
            .iter()
            .map(|item| (item.evidence_id.clone(), item.relevance))
            .collect::<Vec<_>>();
        let pack_order = plan
            .context_pack
            .as_ref()
            .unwrap()
            .evidence
            .iter()
            .map(|evidence| evidence.id.clone())
            .collect::<Vec<_>>();
        let return_ids = plan
            .return_order
            .iter()
            .map(|item| item.evidence_id.clone())
            .collect::<Vec<_>>();
        assert_eq!(actual_return, expected_return);
        assert_ne!(return_ids, pack_order);
        Ok(())
    }

    #[test]
    fn active_fresh_exact_status_selects_only_eligible_exact_compiler_reference_evidence() {
        let setup = exact_fact_manifest();
        let request = ProjectContextRetrievalRequest::new(
            "lexicalmissneedle",
            5,
            ProjectContextPathScope::default(),
            false,
        );

        let actual = plan_project_context_retrieval_with_options(
            &setup,
            &freshness(&setup),
            request,
            ProjectContextRetrievalOptions {
                exact_fact_status: Some(&exact_status(&setup, ExactFactStatus::Active)),
                replay_activation: None,
                ..ProjectContextRetrievalOptions::default()
            },
        );
        let plan = expect_plan(actual);
        let actual = (
            plan.selected_results
                .iter()
                .map(|result| result.id.clone())
                .collect::<Vec<_>>(),
            plan.query_diagnostics.phase_diagnostics.exact_fact,
            plan.read_requests
                .iter()
                .map(|request| request.evidence_id.clone())
                .collect::<Vec<_>>(),
        );
        let expected = (
            vec!["symbol:src/exact.rs:Function:eligible".to_string()],
            ProjectContextExactFactPhaseStatus::Active(ProjectContextExactFactActiveSummary {
                candidate_count: 1,
                selected_count: 1,
                fanout_cap: EXACT_FACT_REFERENCE_FANOUT_CAP,
            }),
            vec!["symbol:src/exact.rs:Function:eligible".to_string()],
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn exact_fact_inactive_statuses_emit_deterministic_diagnostics_and_select_nothing() {
        let setup = exact_fact_manifest();
        let cases = vec![
            (
                None,
                ProjectContextExactFactPhaseStatus::Inactive(
                    ProjectContextExactFactInactiveReason::StatusUnavailable,
                ),
            ),
            (
                Some(exact_status(&setup, ExactFactStatus::NoArtifactStore)),
                ProjectContextExactFactPhaseStatus::Inactive(
                    ProjectContextExactFactInactiveReason::StatusInactive(
                        ExactFactStatus::NoArtifactStore,
                    ),
                ),
            ),
            (
                Some(exact_status(
                    &setup,
                    ExactFactStatus::ReportMissingOrCorrupt,
                )),
                ProjectContextExactFactPhaseStatus::Inactive(
                    ProjectContextExactFactInactiveReason::StatusInactive(
                        ExactFactStatus::ReportMissingOrCorrupt,
                    ),
                ),
            ),
            (
                Some(exact_status(&setup, ExactFactStatus::StaleManifest)),
                ProjectContextExactFactPhaseStatus::Inactive(
                    ProjectContextExactFactInactiveReason::StatusInactive(
                        ExactFactStatus::StaleManifest,
                    ),
                ),
            ),
            (
                Some(exact_status(
                    &setup,
                    ExactFactStatus::ArtifactsPresentNoneAccepted,
                )),
                ProjectContextExactFactPhaseStatus::Inactive(
                    ProjectContextExactFactInactiveReason::StatusInactive(
                        ExactFactStatus::ArtifactsPresentNoneAccepted,
                    ),
                ),
            ),
            (
                Some(exact_status(
                    &setup,
                    ExactFactStatus::AcceptedButNoGraphEdges,
                )),
                ProjectContextExactFactPhaseStatus::Inactive(
                    ProjectContextExactFactInactiveReason::StatusInactive(
                        ExactFactStatus::AcceptedButNoGraphEdges,
                    ),
                ),
            ),
        ];

        let actual = cases
            .into_iter()
            .map(|(status, expected_status)| {
                let request = ProjectContextRetrievalRequest::new(
                    "lexicalmissneedle",
                    5,
                    ProjectContextPathScope::default(),
                    false,
                );
                let outcome = plan_project_context_retrieval_with_options(
                    &setup,
                    &freshness(&setup),
                    request,
                    ProjectContextRetrievalOptions {
                        exact_fact_status: status.as_ref(),
                        replay_activation: None,
                        ..ProjectContextRetrievalOptions::default()
                    },
                );
                let plan = expect_plan(outcome);
                (
                    plan.selected_results.iter().any(|result| {
                        result
                            .score_parts
                            .contains_key(EXACT_FACT_REFERENCE_SCORE_PART)
                    }),
                    plan.query_diagnostics.phase_diagnostics.exact_fact,
                    expected_status,
                )
            })
            .collect::<Vec<_>>();
        let expected = actual
            .iter()
            .map(|(_, _, expected_status)| {
                (false, expected_status.clone(), expected_status.clone())
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn exact_fact_active_but_unsafe_branches_are_diagnostic_only() {
        let setup = exact_fact_manifest();
        let mut missing_hash = exact_status(&setup, ExactFactStatus::Active);
        missing_hash.manifest_hash = None;
        let mut mismatch = exact_status(&setup, ExactFactStatus::Active);
        mismatch.manifest_hash = Some(fingerprint("different-snapshot"));
        let mut no_eligible = exact_fact_manifest();
        no_eligible.edges.clear();
        let selected_zero_request = ProjectContextRetrievalRequest::new(
            "lexicalmissneedle",
            5,
            ProjectContextPathScope::new(Some("src/other/".to_string()), Vec::new()),
            false,
        );
        let stale_status = exact_status(&setup, ExactFactStatus::StaleManifest);
        let stale_status_plan = expect_plan(plan_project_context_retrieval_with_options(
            &setup,
            &freshness(&setup),
            ProjectContextRetrievalRequest::new(
                "lexicalmissneedle",
                5,
                ProjectContextPathScope::default(),
                false,
            ),
            ProjectContextRetrievalOptions {
                exact_fact_status: Some(&stale_status),
                replay_activation: None,
                ..ProjectContextRetrievalOptions::default()
            },
        ));
        assert_eq!(
            stale_status_plan
                .query_diagnostics
                .phase_diagnostics
                .exact_fact,
            ProjectContextExactFactPhaseStatus::Inactive(
                ProjectContextExactFactInactiveReason::StatusInactive(
                    ExactFactStatus::StaleManifest
                ),
            ),
        );

        let stale = ManifestFreshnessEvaluation {
            state: FreshnessState {
                changed: vec!["src/exact.rs".to_string()],
                fresh: true,
                ..fresh_state(&setup)
            },
            proof_level: FreshnessProofLevel::FullFilesystem,
        };
        let cases = vec![
            (
                &setup,
                freshness(&setup),
                ProjectContextRetrievalRequest::new(
                    "lexicalmissneedle",
                    5,
                    ProjectContextPathScope::default(),
                    false,
                ),
                missing_hash,
                ProjectContextExactFactInactiveReason::MissingManifestHash,
            ),
            (
                &setup,
                freshness(&setup),
                ProjectContextRetrievalRequest::new(
                    "lexicalmissneedle",
                    5,
                    ProjectContextPathScope::default(),
                    false,
                ),
                mismatch,
                ProjectContextExactFactInactiveReason::WorkspaceSnapshotMismatch,
            ),
            (
                &setup,
                stale,
                ProjectContextRetrievalRequest::new(
                    "lexicalmissneedle",
                    5,
                    ProjectContextPathScope::default(),
                    false,
                ),
                exact_status(&setup, ExactFactStatus::Active),
                ProjectContextExactFactInactiveReason::ManifestNotFresh,
            ),
            (
                &no_eligible,
                freshness(&no_eligible),
                ProjectContextRetrievalRequest::new(
                    "lexicalmissneedle",
                    5,
                    ProjectContextPathScope::default(),
                    false,
                ),
                exact_status(&no_eligible, ExactFactStatus::Active),
                ProjectContextExactFactInactiveReason::NoEligibleEvidence,
            ),
            (
                &setup,
                freshness(&setup),
                selected_zero_request,
                exact_status(&setup, ExactFactStatus::Active),
                ProjectContextExactFactInactiveReason::SelectedZero,
            ),
        ];

        let actual = cases
            .into_iter()
            .map(|(manifest, freshness, request, status, expected_reason)| {
                let plan = expect_plan(plan_project_context_retrieval_with_options(
                    manifest,
                    &freshness,
                    request,
                    ProjectContextRetrievalOptions {
                        exact_fact_status: Some(&status),
                        replay_activation: None,
                        ..ProjectContextRetrievalOptions::default()
                    },
                ));
                (
                    plan.selected_results.iter().any(|result| {
                        result
                            .score_parts
                            .contains_key(EXACT_FACT_REFERENCE_SCORE_PART)
                    }),
                    plan.query_diagnostics.phase_diagnostics.exact_fact,
                    ProjectContextExactFactPhaseStatus::Inactive(expected_reason),
                )
            })
            .collect::<Vec<_>>();
        let expected = actual
            .iter()
            .map(|(_, _, expected_status)| {
                (false, expected_status.clone(), expected_status.clone())
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn exact_fact_constructor_enforces_references_exact_compiler_and_readback_eligibility() {
        let setup = exact_fact_manifest();
        let status = exact_status(&setup, ExactFactStatus::Active);
        let actual = setup
            .edges
            .iter()
            .map(|edge| {
                ExactCompilerReferenceEvidence::new(&setup, &freshness(&setup), &status, edge)
                    .is_ok()
            })
            .collect::<Vec<_>>();
        let expected = vec![false, false, true];
        assert_eq!(actual, expected);
    }

    #[test]
    fn exact_fact_edges_with_unresolvable_reference_target_are_diagnostic_only() {
        let mut setup = exact_fact_manifest();
        setup.edges = vec![GraphEdge {
            from: "symbol:src/exact.rs:Function:root".to_string(),
            to: "symbol:src/exact.rs:Function:missing".to_string(),
            kind: GraphEdgeKind::References,
            confidence: 1.0,
            confidence_kind: EdgeConfidence::ExactCompiler,
            provenance: provenance("src/exact.rs", Some(1), Some(1), "exact-test", "missing"),
        }];
        let request = ProjectContextRetrievalRequest::new(
            "lexicalmissneedle",
            5,
            ProjectContextPathScope::default(),
            false,
        );

        let actual = plan_project_context_retrieval_with_options(
            &setup,
            &freshness(&setup),
            request,
            ProjectContextRetrievalOptions {
                exact_fact_status: Some(&exact_status(&setup, ExactFactStatus::Active)),
                replay_activation: None,
                ..ProjectContextRetrievalOptions::default()
            },
        );
        let plan = expect_plan(actual);
        let expected = (
            Vec::<String>::new(),
            ProjectContextExactFactPhaseStatus::Inactive(
                ProjectContextExactFactInactiveReason::NoEligibleEvidence,
            ),
            ProjectContextWriteDecision::NoWriteEmptyRetrieval,
        );

        assert_eq!(
            (
                plan.selected_results
                    .iter()
                    .map(|result| result.id.clone())
                    .collect::<Vec<_>>(),
                plan.query_diagnostics.phase_diagnostics.exact_fact,
                plan.write_decision,
            ),
            expected,
        );
    }

    #[test]
    fn exact_fact_fanout_caps_and_ordering_are_deterministic() {
        let setup = capped_exact_fact_manifest();
        let request = ProjectContextRetrievalRequest::new(
            "lexicalmissneedle",
            20,
            ProjectContextPathScope::default(),
            false,
        );

        let plan = expect_plan(plan_project_context_retrieval_with_options(
            &setup,
            &freshness(&setup),
            request,
            ProjectContextRetrievalOptions {
                exact_fact_status: Some(&exact_status(&setup, ExactFactStatus::Active)),
                replay_activation: None,
                ..ProjectContextRetrievalOptions::default()
            },
        ));
        let actual = (
            plan.selected_results
                .iter()
                .filter(|result| {
                    result
                        .score_parts
                        .contains_key(EXACT_FACT_REFERENCE_SCORE_PART)
                })
                .map(|result| result.id.clone())
                .collect::<Vec<_>>(),
            plan.query_diagnostics.phase_diagnostics.exact_fact,
        );
        let expected = (
            (0..EXACT_FACT_REFERENCE_FANOUT_CAP)
                .map(|index| format!("symbol:src/exact.rs:Function:target_{index:02}"))
                .collect::<Vec<_>>(),
            ProjectContextExactFactPhaseStatus::Active(ProjectContextExactFactActiveSummary {
                candidate_count: 12,
                selected_count: EXACT_FACT_REFERENCE_FANOUT_CAP,
                fanout_cap: EXACT_FACT_REFERENCE_FANOUT_CAP,
            }),
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn exact_fact_activation_preserves_lexical_vector_graph_fallback_baseline() {
        let setup = exact_fact_manifest();
        let request = ProjectContextRetrievalRequest::new(
            "eligible Function",
            5,
            ProjectContextPathScope::default(),
            true,
        );

        let fallback = plan_project_context_retrieval(&setup, &freshness(&setup), request.clone());
        let active = plan_project_context_retrieval_with_options(
            &setup,
            &freshness(&setup),
            request,
            ProjectContextRetrievalOptions {
                exact_fact_status: Some(&exact_status(&setup, ExactFactStatus::Active)),
                replay_activation: None,
                ..ProjectContextRetrievalOptions::default()
            },
        );
        let fallback_ids = expect_plan(fallback)
            .selected_results
            .iter()
            .map(|result| result.id.clone())
            .collect::<Vec<_>>();
        let active_ids = expect_plan(active)
            .selected_results
            .iter()
            .map(|result| result.id.clone())
            .collect::<Vec<_>>();
        assert!(fallback_ids.iter().all(|id| active_ids.contains(id)));
    }

    #[test]
    fn retrieval_plan_path_has_no_exact_fact_producer_invocation_surface() {
        let setup = std::include_str!("retrieval_plan.rs");
        let forbidden = [
            concat!("produce_workspace_exact_fact_reference", "_with_driver("),
            concat!("NativeLspReference", "Producer"),
            concat!("derive_native_lsp", "_reference_request"),
            concat!("RustAnalyzerCapability", "Probe"),
            concat!("write_external_fact", "_artifact("),
            concat!("ingest_external_fact", "_artifacts("),
        ];
        let actual = forbidden
            .iter()
            .filter(|pattern| setup.contains(**pattern))
            .copied()
            .collect::<Vec<_>>();
        let expected: Vec<&str> = Vec::new();
        assert_eq!(actual, expected);
    }

    fn expect_plan(
        outcome: ProjectContextRetrievalPlanningOutcome,
    ) -> Box<ProjectContextRetrievalPlan> {
        match outcome {
            ProjectContextRetrievalPlanningOutcome::Plan(plan) => plan,
            ProjectContextRetrievalPlanningOutcome::Refusal(refusal) => {
                panic!("unexpected refusal: {:?}", refusal)
            }
        }
    }

    fn plan_snapshot(
        outcome: ProjectContextRetrievalPlanningOutcome,
    ) -> (
        Vec<(String, String, Option<String>)>,
        Vec<(String, String, u32, u32)>,
        ProjectContextWriteDecision,
    ) {
        let plan = expect_plan(outcome);
        (
            plan.selected_results
                .iter()
                .map(|result| {
                    (
                        result.id.clone(),
                        result.path.clone(),
                        result.symbol.clone(),
                    )
                })
                .collect(),
            plan.read_requests
                .iter()
                .map(|request| {
                    (
                        request.evidence_id.clone(),
                        request.relative_manifest_path().to_string(),
                        request.start_line,
                        request.end_line,
                    )
                })
                .collect(),
            plan.write_decision,
        )
    }

    fn ready_vector_boundary<'a>(
        index: &'a DeterministicVectorIndex,
        dimension: usize,
    ) -> ProjectContextVectorIndexBoundary<'a> {
        ProjectContextVectorIndexBoundary {
            index,
            identity: ProjectContextIntegrationIdentity {
                provider: "fixture",
                artifact: "deterministic-vector-index",
            },
            readiness: ProjectContextVectorReadiness::Ready { dimension },
        }
    }

    fn ready_reranker_boundary<'a>(
        reranker: &'a dyn Reranker,
    ) -> ProjectContextRerankerBoundary<'a> {
        ProjectContextRerankerBoundary {
            reranker,
            identity: ProjectContextIntegrationIdentity {
                provider: "fixture",
                artifact: "deterministic-reranker",
            },
            readiness: ProjectContextRerankerReadiness::Ready,
        }
    }

    struct IntentOrderReranker;

    impl Reranker for IntentOrderReranker {
        fn rerank(&self, query: &str, candidates: &[RerankCandidate]) -> Vec<RerankScore> {
            candidates
                .iter()
                .map(|candidate| RerankScore {
                    id: candidate.id.clone(),
                    score: if candidate.text.contains(query) {
                        1.0
                    } else {
                        0.0
                    },
                })
                .collect()
        }
    }

    fn cargo_plan_manifest() -> ProjectManifest {
        ProjectManifest {
            version: 1,
            root: "/workspace".into(),
            files: vec![SourceFile {
                path: "Cargo.toml".to_string(),
                language: Language::Toml,
                bytes: 200,
                lines: 16,
                content_hash: fingerprint("cargo-plan-toml"),
                provenance: provenance("Cargo.toml", Some(1), Some(16), "test", "file"),
            }],
            cargo_packages: vec![CargoPackageMetadata {
                manifest_path: "Cargo.toml".to_string(),
                package_root: "".to_string(),
                name: "fixture_app".to_string(),
                version: Some("0.1.0".to_string()),
                edition: Some("2021".to_string()),
                targets: vec![CargoTargetMetadata {
                    name: "fixture_bin".to_string(),
                    kind: CargoTargetKind::Bin,
                    path: "src/main.rs".to_string(),
                    declaration: CargoTargetDeclaration::Declared,
                    provenance: provenance("Cargo.toml", Some(8), Some(10), "test", "target"),
                }],
                features: Vec::new(),
                provenance: provenance("Cargo.toml", Some(1), Some(5), "test", "package"),
            }],
            cargo_package_dependencies: vec![CargoPackageDependency {
                manifest_path: "Cargo.toml".to_string(),
                declaring_package: Some("fixture_app".to_string()),
                dependency_key: "serde_alias".to_string(),
                package_name: "serde".to_string(),
                kind: CargoDependencyKind::Normal,
                target: None,
                version: Some("1".to_string()),
                path: None,
                optional: false,
                features: Vec::new(),
                declaration: CargoDependencyDeclaration::DeclaredExternal,
                linked_package_manifest_path: None,
                provenance: provenance("Cargo.toml", Some(12), Some(12), "test", "dependency"),
            }],
            manifest_hash: fingerprint("cargo-plan"),
            ..ProjectManifest::default()
        }
    }

    fn indexed_fixture() -> Result<(tempfile::TempDir, std::path::PathBuf, ProjectManifest)> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        Ok((fixture, root, manifest))
    }

    fn planned_paths(outcome: ProjectContextRetrievalPlanningOutcome) -> Vec<String> {
        match outcome {
            ProjectContextRetrievalPlanningOutcome::Plan(plan) => plan
                .read_requests
                .into_iter()
                .map(|request| request.relative_manifest_path().to_string())
                .collect(),
            ProjectContextRetrievalPlanningOutcome::Refusal(refusal) => {
                panic!("unexpected refusal: {:?}", refusal)
            }
        }
    }

    fn refusal_code(
        outcome: ProjectContextRetrievalPlanningOutcome,
    ) -> Option<ProjectContextRetrievalRefusalCode> {
        match outcome {
            ProjectContextRetrievalPlanningOutcome::Refusal(refusal) => Some(refusal.code),
            ProjectContextRetrievalPlanningOutcome::Plan(_) => None,
        }
    }

    fn freshness(manifest: &ProjectManifest) -> ManifestFreshnessEvaluation {
        ManifestFreshnessEvaluation {
            state: fresh_state(manifest),
            proof_level: FreshnessProofLevel::FullFilesystem,
        }
    }

    fn fresh_state(manifest: &ProjectManifest) -> FreshnessState {
        FreshnessState {
            changed: Vec::new(),
            deleted: Vec::new(),
            added: Vec::new(),
            unchanged: manifest
                .files
                .iter()
                .map(|file| file.path.clone())
                .collect(),
            fresh: true,
        }
    }

    fn manual_manifest() -> ProjectManifest {
        ProjectManifest {
            version: 1,
            root: std::path::PathBuf::from("/workspace"),
            files: vec![SourceFile {
                path: "src/lib.rs".to_string(),
                language: crate::Language::Rust,
                bytes: 100,
                lines: 9,
                content_hash: fingerprint("file"),
                provenance: provenance("src/lib.rs", Some(1), Some(9), "test", "file-provenance"),
            }],
            manifest_hash: fingerprint("manifest"),
            ..ProjectManifest::default()
        }
    }

    fn exact_status(manifest: &ProjectManifest, status: ExactFactStatus) -> ExactFactStatusReport {
        ExactFactStatusReport {
            exact_facts_active: status == ExactFactStatus::Active,
            status,
            manifest_path: std::path::PathBuf::from("project_model_manifest"),
            manifest_hash: Some(manifest.manifest_hash.clone()),
            manifest_freshness_proof_level: Some(FreshnessProofLevel::FullFilesystem),
            ingestion_report_path: std::path::PathBuf::from("external_fact_report"),
            artifact_store_state: crate::ExactFactArtifactStoreState::Present,
            inspected_artifact_count: 1,
            accepted_artifact_count: 1,
            accepted_batch_fingerprints: vec![fingerprint("batch")],
            manifest_external_fact_batch_count: 1,
            manifest_external_facts_fingerprint: Some(fingerprint("external-facts")),
            reference_edge_count: manifest
                .edges
                .iter()
                .filter(|edge| edge.kind == GraphEdgeKind::References)
                .count(),
            exact_compiler_reference_edge_count: manifest
                .edges
                .iter()
                .filter(|edge| {
                    edge.kind == GraphEdgeKind::References
                        && edge.confidence_kind == EdgeConfidence::ExactCompiler
                })
                .count(),
            issue_summaries: Vec::new(),
        }
    }

    fn exact_fact_manifest() -> ProjectManifest {
        let mut manifest = manual_manifest();
        manifest.files = vec![SourceFile {
            path: "src/exact.rs".to_string(),
            language: Language::Rust,
            bytes: 100,
            lines: 20,
            content_hash: fingerprint("exact-file"),
            provenance: provenance("src/exact.rs", Some(1), Some(20), "test", "exact-file"),
        }];
        manifest.symbols = vec![
            exact_symbol("root", 1),
            exact_symbol("calls_exact", 2),
            exact_symbol("heuristic", 3),
            exact_symbol("eligible", 4),
        ];
        manifest.edges = vec![
            exact_edge(
                "root",
                "calls_exact",
                GraphEdgeKind::Calls,
                EdgeConfidence::ExactCompiler,
            ),
            exact_edge(
                "root",
                "heuristic",
                GraphEdgeKind::References,
                EdgeConfidence::HeuristicHigh,
            ),
            exact_edge(
                "root",
                "eligible",
                GraphEdgeKind::References,
                EdgeConfidence::ExactCompiler,
            ),
        ];
        manifest.manifest_hash = fingerprint("exact-manifest");
        manifest
    }

    fn capped_exact_fact_manifest() -> ProjectManifest {
        let mut manifest = manual_manifest();
        manifest.files = vec![SourceFile {
            path: "src/exact.rs".to_string(),
            language: Language::Rust,
            bytes: 100,
            lines: 40,
            content_hash: fingerprint("capped-exact-file"),
            provenance: provenance("src/exact.rs", Some(1), Some(40), "test", "capped-file"),
        }];
        manifest.symbols.push(exact_symbol("root", 1));
        for index in (0..12).rev() {
            manifest
                .symbols
                .push(exact_symbol(&format!("target_{index:02}"), index + 2));
            manifest.edges.push(exact_edge(
                "root",
                &format!("target_{index:02}"),
                GraphEdgeKind::References,
                EdgeConfidence::ExactCompiler,
            ));
        }
        manifest.manifest_hash = fingerprint("capped-exact-manifest");
        manifest
    }

    fn exact_symbol(name: &str, line: usize) -> SymbolNode {
        SymbolNode {
            id: format!("symbol:src/exact.rs:Function:{name}"),
            name: name.to_string(),
            kind: SymbolKind::Function,
            path: "src/exact.rs".to_string(),
            parent: None,
            start_line: u32::try_from(line).unwrap(),
            end_line: u32::try_from(line).unwrap(),
            provenance: provenance(
                "src/exact.rs",
                Some(u32::try_from(line).unwrap()),
                Some(u32::try_from(line).unwrap()),
                "test",
                name,
            ),
        }
    }

    fn exact_edge(
        from: &str,
        to: &str,
        kind: GraphEdgeKind,
        confidence_kind: EdgeConfidence,
    ) -> GraphEdge {
        GraphEdge {
            from: format!("symbol:src/exact.rs:Function:{from}"),
            to: format!("symbol:src/exact.rs:Function:{to}"),
            kind,
            confidence: 1.0,
            confidence_kind,
            provenance: provenance("src/exact.rs", Some(1), Some(1), "exact-test", to),
        }
    }

    fn provenance(
        path: &str,
        start_line: Option<u32>,
        end_line: Option<u32>,
        source: &str,
        seed: &str,
    ) -> Provenance {
        Provenance {
            path: path.to_string(),
            start_line,
            end_line,
            source: source.to_string(),
            fingerprint: fingerprint(seed),
        }
    }
}
