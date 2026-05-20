//! Project model primitives, indexing, retrieval, graph, and episodic storage.

mod context_adapter;
mod durable_vector_index;
mod eval;
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
mod status;
mod types;
mod util;
mod vector;

pub use context_adapter::{
    ProjectModelContextRenderRoot, ProjectModelSourceNode, evidence_line_range,
    render_source_from_evidence, render_sources_from_context_pack, render_sources_from_nodes,
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
    LearningContextPayload, LearningContextRecord, LearningContextTransport,
    LearningLedgerFreshness, LearningProvenance, LearningRedactionStatus, LearningReviewState,
    LearningSourceKind, learning_records_to_graph,
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
    ProjectModelContextRenderBudget, ProjectModelContextSource,
    ProjectModelEvidenceLedgerActivationMetadata, ProjectModelEvidenceReadinessMetadata,
    ProjectModelExactFactReadinessMetadata, render_project_model_context,
};
pub use retrieval::{plan_retrieval, retrieve, retrieve_with_boundaries};
pub use status::{
    ExactFactArtifactStoreMetadata, ExactFactArtifactStoreState, ExactFactStatus,
    ExactFactStatusReport, read_exact_fact_artifact_store_metadata, read_exact_fact_status,
};
pub use types::{
    CargoDependencyDeclaration, CargoDependencyKind, CargoFeatureMetadata, CargoPackageDependency,
    CargoPackageMetadata, CargoTargetDeclaration, CargoTargetKind, CargoTargetMetadata,
    CargoWorkspaceMetadata, ContextPack, ContextPackArtifactEvalReport, ContextPackArtifactId,
    ContextPackEvidence, ContextPackEvidenceSource, ContextPackSelection, DecisionGraphNode,
    EdgeConfidence, EvalCaseGraphNode, EvidenceFreshness, EvidenceLedgerActivation,
    EvidenceLedgerActivationSummary, EvidenceLedgerEvalIssue, EvidenceLedgerEvalIssueCode,
    EvidenceLedgerGraphMetadata, EvidenceLedgerLinkageReport, EvidenceReadinessDiagnostic,
    ExternalFactArtifactIngestionReport, ExternalFactArtifactReport, ExternalFactBatch,
    ExternalFactBatchMetadata, ExternalFactIngestionIssue, ExternalFactIngestionIssueCode,
    ExternalFactIngestionReport, ExternalFactProductionBaseline, ExternalFactSource, ExternalFacts,
    ExternalReferenceFact, ExternalSymbolFact, FileGraphNode, FileNode, FileNodeKind,
    FreshnessEvalReport, FreshnessProofLevel, FreshnessState, FutureVectorRetrievalScaffold,
    GraphCoverageReport, GraphEdge, GraphEdgeKind, KnowledgeGraph, KnowledgeGraphEdge,
    KnowledgeGraphNode, KnowledgeGraphNodeId, KnowledgeGraphNodeKind, Language, LexicalDocument,
    LexicalDocumentKind, LexicalSearchHit, ManifestFreshnessEvaluation, ProjectManifest,
    Provenance, ProvenanceCompletenessReport, RerankCandidate, RerankScore, RetrievalEvalCase,
    RetrievalEvalReport, RetrievalQuery, RetrievalResult, RetrievalScoringPlan,
    RetrievalScoringWeights, RetrievedEvidenceGraphNode, ShardGraphNode, ShardManifest,
    ShardStrategy, SourceFile, StaleEvidencePolicy, SymbolGraphNode, SymbolKind, SymbolNode,
    TaskGraphNode, ToolEpisode, ToolEpisodeEvalReport, ToolEpisodeGraphNode, TypedExternalFacts,
    TypedExternalReferenceFact, TypedExternalSymbolFact, VectorQuery, VectorSearchHit,
    classify_evidence_freshness,
};
pub use util::fingerprint;
pub use vector::{DeterministicReranker, DeterministicVectorIndex, Reranker, VectorIndex};
