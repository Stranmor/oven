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
pub use ingestion::ingest_external_facts;
pub use lexical::{LexicalIndex, documents_from_manifest};
pub use retrieval::{retrieve, retrieve_with_boundaries};
pub use types::{
    EdgeConfidence, ExternalFacts, ExternalReferenceFact, ExternalSymbolFact, FileNode,
    FileNodeKind, FreshnessEvalReport, FreshnessState, FutureVectorRetrievalScaffold,
    GraphCoverageReport, GraphEdge, GraphEdgeKind, Language, LexicalDocument, LexicalDocumentKind,
    LexicalSearchHit, ProjectManifest, Provenance, ProvenanceCompletenessReport, RerankCandidate,
    RerankScore, RetrievalEvalCase, RetrievalEvalReport, RetrievalQuery, RetrievalResult,
    ShardManifest, SourceFile, SymbolKind, SymbolNode, ToolEpisode, VectorQuery, VectorSearchHit,
};
pub use util::fingerprint;
pub use vector::{DeterministicReranker, DeterministicVectorIndex, Reranker, VectorIndex};
