//! Project model primitives, indexing, retrieval, graph, and episodic storage.

mod extraction;
mod freshness;
mod indexer;
mod retrieval;
mod types;
mod util;

pub use extraction::{
    RustExtraction, extract_cargo_dependency_edges, extract_rust_import_edges, extract_rust_symbols,
};
pub use freshness::compare_freshness;
pub use indexer::ProjectIndexer;
pub use retrieval::retrieve;
pub use types::{
    FileNode, FileNodeKind, FreshnessState, FutureVectorRetrievalScaffold, GraphEdge,
    GraphEdgeKind, Language, ProjectManifest, Provenance, RetrievalQuery, RetrievalResult,
    ShardManifest, SourceFile, SymbolKind, SymbolNode, ToolEpisode,
};
pub use util::fingerprint;
