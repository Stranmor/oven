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
mod render;
mod retrieval;
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
    context_pack_worst_case_freshness, evaluate_context_pack_artifacts,
    evaluate_context_pack_artifacts_by_id, evaluate_episode_artifact_links, evaluate_freshness,
    evaluate_graph_coverage, evaluate_provenance_completeness, evaluate_retrieval,
    evaluate_tool_episodes, tool_episode_graph_id, tool_episodes_to_graph,
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
    ingest_typed_external_facts, validate_external_fact_batch,
};
pub use learning::{
    LearningContextPayload, LearningContextRecord, LearningContextTransport,
    LearningLedgerFreshness, LearningProvenance, LearningRedactionStatus, LearningReviewState,
    LearningSourceKind, learning_records_to_graph,
};
pub use lexical::{LexicalIndex, documents_from_manifest};
pub use policy::{
    ProjectContextTarget, TargetResolutionBudget, directory_path_filter, local_project_model_dir,
    local_project_model_manifest, mentioned_paths, resolve_mentioned_path,
};
pub use render::{
    DEFAULT_RENDERED_SOURCE_LIMIT, ProjectModelContextRenderBudget, ProjectModelContextSource,
    render_project_model_context,
};
pub use retrieval::{plan_retrieval, retrieve, retrieve_with_boundaries};
pub use types::{
    CargoDependencyDeclaration, CargoDependencyKind, CargoFeatureMetadata, CargoPackageDependency,
    CargoPackageMetadata, CargoTargetDeclaration, CargoTargetKind, CargoTargetMetadata,
    CargoWorkspaceMetadata, ContextPack, ContextPackArtifactEvalReport, ContextPackArtifactId,
    ContextPackEvidence, ContextPackEvidenceSource, ContextPackSelection, DecisionGraphNode,
    EdgeConfidence, EvalCaseGraphNode, EvidenceFreshness, EvidenceLedgerEvalIssue,
    EvidenceLedgerEvalIssueCode, EvidenceLedgerLinkageReport, ExternalFactArtifactIngestionReport,
    ExternalFactArtifactReport, ExternalFactBatch, ExternalFactBatchMetadata,
    ExternalFactIngestionIssue, ExternalFactIngestionIssueCode, ExternalFactIngestionReport,
    ExternalFactSource, ExternalFacts, ExternalReferenceFact, ExternalSymbolFact, FileGraphNode,
    FileNode, FileNodeKind, FreshnessEvalReport, FreshnessProofLevel, FreshnessState,
    FutureVectorRetrievalScaffold, GraphCoverageReport, GraphEdge, GraphEdgeKind, KnowledgeGraph,
    KnowledgeGraphEdge, KnowledgeGraphNode, KnowledgeGraphNodeId, KnowledgeGraphNodeKind, Language,
    LexicalDocument, LexicalDocumentKind, LexicalSearchHit, ManifestFreshnessEvaluation,
    ProjectManifest, Provenance, ProvenanceCompletenessReport, RerankCandidate, RerankScore,
    RetrievalEvalCase, RetrievalEvalReport, RetrievalQuery, RetrievalResult, RetrievalScoringPlan,
    RetrievalScoringWeights, RetrievedEvidenceGraphNode, ShardGraphNode, ShardManifest,
    ShardStrategy, SourceFile, StaleEvidencePolicy, SymbolGraphNode, SymbolKind, SymbolNode,
    TaskGraphNode, ToolEpisode, ToolEpisodeEvalReport, ToolEpisodeGraphNode, TypedExternalFacts,
    TypedExternalReferenceFact, TypedExternalSymbolFact, VectorQuery, VectorSearchHit,
    classify_evidence_freshness,
};
pub use util::fingerprint;
pub use vector::{DeterministicReranker, DeterministicVectorIndex, Reranker, VectorIndex};
