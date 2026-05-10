//! Project model primitives, indexing, retrieval, graph, and episodic storage.

mod eval;
mod extraction;
mod freshness;
mod indexer;
mod ingestion;
mod lexical;
mod retrieval;
mod types;
mod util;
mod vector;

pub use eval::{
    evaluate_freshness, evaluate_graph_coverage, evaluate_provenance_completeness,
    evaluate_retrieval,
};
pub use extraction::{
    RustExtraction, extract_cargo_dependency_edges, extract_rust_call_edges,
    extract_rust_import_edges, extract_rust_symbols,
};
pub use freshness::compare_freshness;
pub use indexer::ProjectIndexer;
pub use ingestion::{ingest_external_facts, ingest_typed_external_facts};
pub use lexical::{LexicalIndex, documents_from_manifest};
pub use retrieval::{retrieve, retrieve_with_boundaries};
pub use types::{
    ContextPack, ContextPackEvidence, ContextPackEvidenceSource, ContextPackSelection,
    DecisionGraphNode, EdgeConfidence, EvalCaseGraphNode, EvidenceFreshness, ExternalFactSource,
    ExternalFacts, ExternalReferenceFact, ExternalSymbolFact, FileGraphNode, FileNode,
    FileNodeKind, FreshnessEvalReport, FreshnessState, FutureVectorRetrievalScaffold,
    GraphCoverageReport, GraphEdge, GraphEdgeKind, KnowledgeGraph, KnowledgeGraphEdge,
    KnowledgeGraphNode, KnowledgeGraphNodeId, KnowledgeGraphNodeKind, Language, LexicalDocument,
    LexicalDocumentKind, LexicalSearchHit, ProjectManifest, Provenance,
    ProvenanceCompletenessReport, RerankCandidate, RerankScore, RetrievalEvalCase,
    RetrievalEvalReport, RetrievalQuery, RetrievalResult, RetrievedEvidenceGraphNode,
    ShardGraphNode, ShardManifest, SourceFile, StaleEvidencePolicy, SymbolGraphNode, SymbolKind,
    SymbolNode, TaskGraphNode, ToolEpisode, ToolEpisodeGraphNode, TypedExternalFacts,
    TypedExternalReferenceFact, TypedExternalSymbolFact, VectorQuery, VectorSearchHit,
    classify_evidence_freshness,
};
pub use util::fingerprint;
pub use vector::{DeterministicReranker, DeterministicVectorIndex, Reranker, VectorIndex};
