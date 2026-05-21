//! Project model primitives, indexing, retrieval, graph, and episodic storage.

mod cache_partition;
mod commit_boundary;
mod context_adapter;
mod durable_vector_index;
mod eval;
mod evidence_replay;
mod extraction;
mod freshness;
mod indexer;
mod ingestion;
mod learning;
mod lexical;
mod policy;
mod producer;
mod render;
mod retrieval;
mod retrieval_plan;
mod status;
mod types;
mod util;
mod vector;

pub use cache_partition::{
    CACHE_PARTITION_SCHEMA_VERSION, CachePartitionError, CachePartitionManifestKnown,
    CachePartitionReadbackVerified, CachePartitionSourcesSelected,
    CachePartitionStablePayloadSealed, CachePartitionVolatileSidecarAttached,
    PROJECT_MODEL_CONTEXT_RENDERER_TEMPLATE_VERSION, PROJECT_MODEL_CONTEXT_RETRIEVAL_PLAN_VERSION,
    PROJECT_MODEL_CONTEXT_TRUNCATION_POLICY, ProjectModelCachePartitionIdentity,
    ProjectModelCachePartitionInput, ProjectModelCachePartitionSource, ProjectModelContextEnvelope,
    ProjectModelContextEnvelopeInput, ProjectModelContextEnvelopeRefusal,
    ProjectModelEnvelopeCacheClass, ProjectModelManifestFreshnessProof, ProjectModelStablePayload,
    ProjectModelStablePayloadWhitelistedFields, ProjectModelVolatileSidecar,
    ProjectModelVolatileSidecarInput, StableProjectModelContextMessage,
    build_project_model_context_envelope, stable_cache_partition_sources_from_nodes,
};
pub use commit_boundary::{
    ProjectContextCommitOutcome, ProjectContextCommittedQueryResult,
    ProjectContextCommittedResultItem, ProjectContextEpisodeAppendFailureReason,
    ProjectContextEpisodeAppendInstruction, ProjectContextEpisodeAppendNotAttemptedReason,
    ProjectContextEpisodeAppendOutcome, ProjectContextPackCommit, ProjectContextPackCommitError,
    ProjectContextPackNoWrite, ProjectContextPackNoWriteReason, ProjectContextPackPersistedProof,
    ProjectContextPackReadbackDecision, ProjectContextPackWriteInstruction,
    ProjectContextPersistedEpisodeAppendOutcome, ProjectContextReadbackEvidence,
    ProjectContextReadbackOutcome, ProjectContextReadbackStatus, ProjectContextReadbackSummary,
    ProjectModelSearchEpisodeInput, ReadRequestsSelected, ReadbackVerified,
};
pub use context_adapter::{
    EvidenceReplayPreviewError, ManifestEvidenceTarget, ProjectModelContextRenderRoot,
    ProjectModelSourceNode, cargo_dependency_evidence_id, cargo_feature_evidence_id,
    cargo_metadata_evidence_id, cargo_package_evidence_id, cargo_target_evidence_id,
    cargo_workspace_evidence_id, evidence_line_range, is_reserved_artifact_evidence_id,
    is_reserved_cargo_evidence_id, redaction_safe_issue_path_label,
    redaction_safe_provenance_source_label, redaction_safe_replay_path_label,
    render_source_from_evidence, render_sources_from_context_pack,
    render_sources_from_evidence_replay, render_sources_from_nodes,
    resolve_manifest_evidence_target, resolve_project_artifact_evidence,
};
pub use durable_vector_index::{
    DURABLE_VECTOR_INDEX_VERSION, DurableVectorIndex, VectorIndexArtifact, VectorIndexArtifactId,
    VectorIndexEntry, VectorSourceKind, vector_entries_from_manifest_embeddings,
};
pub use eval::{
    EvidenceLedgerActivationBudget, EvidenceReadinessDiagnosticBudget,
    context_pack_worst_case_freshness, diagnose_evidence_readiness,
    evaluate_context_pack_artifacts, evaluate_context_pack_artifacts_by_id,
    evaluate_episode_artifact_links, evaluate_freshness, evaluate_graph_coverage,
    evaluate_provenance_completeness, evaluate_retrieval, evaluate_tool_episodes,
    load_evidence_ledger_activation, tool_episode_graph_id, tool_episodes_to_graph,
};
pub use evidence_replay::{
    EvidenceLedgerReplayReport, EvidenceLedgerReplayRequest, EvidenceReplayBudget,
    EvidenceReplayBudgetReport, EvidenceReplayContentPolicy, EvidenceReplayFreshnessPolicy,
    EvidenceReplayIssue, EvidenceReplayIssueCode, EvidenceReplayManifestReference,
    EvidenceReplayReference, EvidenceReplayScoreKind, EvidenceReplaySelectionPolicy,
    EvidenceReplayStalePolicyReport, ReplayActivatedEvidenceRef, ReplayActivationBoundary,
    ReplayActivationCaps, ReplayActivationDiagnostics, ReplayActivationFingerprintInputs,
    ReplayActivationIssue, ReplayActivationRequest, ReplayEvidenceReadbackStatus,
    ReplayEvidenceTargetKind, activate_evidence_ledger_replay, apply_replay_readback_results,
    select_evidence_ledger_replay,
};

pub use extraction::{
    RustExtraction, extract_cargo_dependency_edges, extract_rust_call_edges,
    extract_rust_import_edges, extract_rust_symbols,
};
pub use freshness::compare_freshness;
pub use indexer::ProjectIndexer;
pub use ingestion::{
    external_fact_artifact_fingerprint, external_fact_batch_fingerprint,
    ingest_external_fact_artifacts, ingest_external_fact_batch, ingest_external_facts,
    ingest_typed_external_facts, prepared_external_fact_artifact_batch,
    validate_external_fact_batch, write_external_fact_artifact,
};
pub use learning::{
    AcceptedLearningSummary, LearningContextPayload, LearningContextRecord,
    LearningContextTransport, LearningLedgerFreshness, LearningProvenance, LearningRedactionStatus,
    LearningReviewState, LearningSourceKind, learning_records_to_graph,
};
pub use lexical::{LexicalIndex, documents_from_manifest};
pub use policy::{
    ProjectContextTarget, TargetResolutionBudget, directory_path_filter, local_project_model_dir,
    local_project_model_external_fact_report, local_project_model_manifest, mentioned_paths,
    resolve_mentioned_path,
};
pub use producer::{
    BoundedLspReferenceParser, ExternalFactProducer, ExternalFactProducerCapability,
    ExternalFactProducerProbe, ExternalFactProductionReport, ExternalFactProductionRequest,
    ExternalFactProductionStatus, LspFixtureExactFactProducer, LspReferenceFact, LspTransport,
    NativeLspBoundedLoss, NativeLspEndpointPosition, NativeLspFileOpenPlan,
    NativeLspNoEligibleEndpoint, NativeLspReferenceNormalizationRequest,
    NativeLspReferenceNormalizer, NativeLspReferenceProducer, NativeLspReferenceRequest,
    NativeLspReferenceRequestDerivation, NativeLspSensor, NativeLspSourceEndpoint,
    RustAnalyzerBounds, RustAnalyzerCapability, RustAnalyzerCapabilityProbe,
    RustAnalyzerCapabilityStatus, RustAnalyzerProbe, RustAnalyzerProcess,
    RustAnalyzerProcessOutput, RustAnalyzerReferenceProducer, RustAnalyzerReferenceRequest,
    StdRustAnalyzerProcess, derive_native_lsp_reference_request,
};
pub use render::{
    DEFAULT_RENDERED_SOURCE_LIMIT, ProjectModelContextReadinessMetadata,
    ProjectModelContextRenderBudget, ProjectModelContextRenderOverflow, ProjectModelContextSource,
    ProjectModelEvidenceLedgerActivationMetadata, ProjectModelEvidenceReadinessMetadata,
    ProjectModelExactFactReadinessMetadata, render_project_model_context,
    render_project_model_context_checked,
};
pub use retrieval::{plan_retrieval, retrieve, retrieve_with_boundaries};
pub use retrieval_plan::{
    ExactCompilerReferenceEvidence, ProjectContextExactFactActiveSummary,
    ProjectContextExactFactInactiveReason, ProjectContextExactFactPhaseStatus,
    ProjectContextIntegrationIdentity, ProjectContextPathScope, ProjectContextReadRequest,
    ProjectContextRerankerBoundary, ProjectContextRerankerReadiness,
    ProjectContextRerankerUnavailableReason, ProjectContextRetrievalOptions,
    ProjectContextRetrievalPhaseDiagnostics, ProjectContextRetrievalPhaseInvalidReason,
    ProjectContextRetrievalPhaseSkipReason, ProjectContextRetrievalPhaseStatus,
    ProjectContextRetrievalPhaseUnavailableReason, ProjectContextRetrievalPlan,
    ProjectContextRetrievalPlanDiagnostic, ProjectContextRetrievalPlanningOutcome,
    ProjectContextRetrievalQueryDiagnostics, ProjectContextRetrievalReadRequestSummary,
    ProjectContextRetrievalRefusal, ProjectContextRetrievalRefusalCode,
    ProjectContextRetrievalRequest, ProjectContextRetrievalSelectedSummary,
    ProjectContextReturnOrderItem, ProjectContextSemanticQueryOptions, ProjectContextTopKStatus,
    ProjectContextVectorIndexBoundary, ProjectContextVectorInvalidReason,
    ProjectContextVectorReadiness, ProjectContextVectorUnavailableReason,
    ProjectContextWriteDecision, plan_project_context_retrieval,
    plan_project_context_retrieval_with_options,
};
pub use status::{
    ExactFactArtifactStoreMetadata, ExactFactArtifactStoreState, ExactFactStatus,
    ExactFactStatusReport, read_exact_fact_artifact_store_metadata, read_exact_fact_status,
};
pub use types::{
    ArtifactGraphNode, CargoDependencyDeclaration, CargoDependencyKind, CargoFeatureMetadata,
    CargoPackageDependency, CargoPackageMetadata, CargoTargetDeclaration, CargoTargetKind,
    CargoTargetMetadata, CargoWorkspaceMetadata, ContextPack, ContextPackArtifactEvalReport,
    ContextPackArtifactId, ContextPackEvidence, ContextPackEvidenceSource, ContextPackSelection,
    DecisionGraphNode, EdgeConfidence, EvalCaseGraphNode, EvidenceFreshness,
    EvidenceLedgerActivation, EvidenceLedgerActivationSummary, EvidenceLedgerEvalIssue,
    EvidenceLedgerEvalIssueCode, EvidenceLedgerGraphMetadata, EvidenceLedgerLinkageReport,
    EvidenceReadinessDiagnostic, ExternalFactArtifactIngestionReport, ExternalFactArtifactReport,
    ExternalFactBatch, ExternalFactBatchMetadata, ExternalFactIngestionIssue,
    ExternalFactIngestionIssueCode, ExternalFactIngestionReport, ExternalFactProductionBaseline,
    ExternalFactSource, ExternalFacts, ExternalReferenceFact, ExternalSymbolFact, FileGraphNode,
    FileNode, FileNodeKind, FreshnessEvalReport, FreshnessProofLevel, FreshnessState,
    FutureVectorRetrievalScaffold, GraphCoverageReport, GraphEdge, GraphEdgeKind, KnowledgeGraph,
    KnowledgeGraphEdge, KnowledgeGraphNode, KnowledgeGraphNodeId, KnowledgeGraphNodeKind, Language,
    LexicalDocument, LexicalDocumentKind, LexicalSearchHit, ManifestFreshnessEvaluation,
    ProjectArtifact, ProjectArtifactConfigFormat, ProjectArtifactKind, ProjectManifest, Provenance,
    ProvenanceCompletenessReport, RerankCandidate, RerankScore, RetrievalEvalCase,
    RetrievalEvalReport, RetrievalQuery, RetrievalResult, RetrievalScoringPlan,
    RetrievalScoringWeights, RetrievedEvidenceGraphNode, ShardGraphNode, ShardManifest,
    ShardStrategy, SourceFile, StaleEvidencePolicy, SymbolGraphNode, SymbolKind, SymbolNode,
    TaskGraphNode, ToolEpisode, ToolEpisodeEvalReport, ToolEpisodeGraphNode, TypedExternalFacts,
    TypedExternalReferenceFact, TypedExternalSymbolFact, VectorQuery, VectorSearchHit,
    classify_evidence_freshness,
};
pub use util::fingerprint;
pub use vector::{DeterministicReranker, DeterministicVectorIndex, Reranker, VectorIndex};
