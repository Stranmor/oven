//! Public project-model DTOs and type surfaces.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

/// A deterministic project manifest generated from a workspace root.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProjectManifest {
    /// Manifest format version.
    pub version: u32,
    /// Project root path used to compute relative paths.
    pub root: PathBuf,
    /// Indexed source files keyed by relative path ordering.
    pub files: Vec<SourceFile>,
    /// Hierarchical file nodes derived from indexed files.
    pub file_nodes: Vec<FileNode>,
    /// Rust symbols extracted from source files.
    pub symbols: Vec<SymbolNode>,
    /// Typed knowledge and dependency edges.
    pub edges: Vec<GraphEdge>,
    /// Content shards used by retrieval.
    pub shards: Vec<ShardManifest>,
    /// Manifest-level content hash over deterministic file hashes.
    pub manifest_hash: String,
}

/// A source file known to the project model.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceFile {
    /// Path relative to the project root using `/` separators.
    pub path: String,
    /// Detected implementation language.
    pub language: Language,
    /// UTF-8 byte length of the file content.
    pub bytes: u64,
    /// Number of lines in the file.
    pub lines: u32,
    /// SHA-256 content hash encoded as lowercase hex.
    pub content_hash: String,
    /// Provenance for the indexed file.
    pub provenance: Provenance,
}

/// A file tree node represented as a stable model object.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileNode {
    /// Relative path for this file or directory node.
    pub path: String,
    /// Node kind.
    pub kind: FileNodeKind,
    /// Optional parent relative path.
    pub parent: Option<String>,
    /// Provenance for the file-tree observation.
    pub provenance: Provenance,
}

/// Kind of file-tree node.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileNodeKind {
    /// Directory node.
    Directory,
    /// Regular file node.
    #[default]
    File,
}

/// A typed symbol extracted from source code.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolNode {
    /// Stable symbol identifier.
    pub id: String,
    /// Symbol display name.
    pub name: String,
    /// Symbol kind.
    pub kind: SymbolKind,
    /// Containing relative file path.
    pub path: String,
    /// Optional enclosing symbol identifier.
    pub parent: Option<String>,
    /// One-based inclusive start line.
    pub start_line: u32,
    /// One-based inclusive end line.
    pub end_line: u32,
    /// Provenance for the extraction.
    pub provenance: Provenance,
}

/// Symbol classes supported by the first project-model slice.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SymbolKind {
    /// Rust struct item.
    Struct,
    /// Rust enum item.
    Enum,
    /// Rust trait item.
    Trait,
    /// Rust implementation block.
    Impl,
    /// Rust free function.
    Function,
    /// Rust method inside trait or impl blocks.
    Method,
    /// Rust test function.
    Test,
    /// Rust module item.
    Module,
    /// Unknown or future symbol kind.
    #[default]
    Unknown,
}

/// A typed edge in the project knowledge graph.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Stable source node identifier.
    pub from: String,
    /// Stable target node identifier.
    pub to: String,
    /// Edge kind.
    pub kind: GraphEdgeKind,
    /// Confidence from 0.0 to 1.0.
    pub confidence: f32,
    /// Semantic confidence class used to prevent overclaiming heuristic facts.
    pub confidence_kind: EdgeConfidence,
    /// Provenance for the edge.
    pub provenance: Provenance,
}

/// Semantic confidence carried by graph edges.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EdgeConfidence {
    /// Compiler, LSP, or SCIP fact imported from an authoritative external source.
    ExactCompiler,
    /// High-confidence syntax heuristic produced without type resolution.
    #[default]
    HeuristicHigh,
    /// Low-confidence syntax heuristic that may require later compiler validation.
    HeuristicLow,
}

/// Supported graph relationships.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GraphEdgeKind {
    /// File contains symbol.
    Contains,
    /// Symbol is a child of another symbol.
    ChildOf,
    /// Rust use or pub use import.
    Imports,
    /// Rust module declaration.
    ModuleDeclares,
    /// Rust extern crate declaration.
    ExternCrate,
    /// Cargo dependency declared in Cargo.toml.
    CargoDependency,
    /// Symbol or callable invokes another callable by name or imported compiler fact.
    Calls,
    /// Symbol or file references another symbol.
    References,
    /// Retrieval expansion relationship.
    #[default]
    Related,
}

/// A deterministic retrieval shard descriptor.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardManifest {
    /// Stable shard identifier.
    pub id: String,
    /// File path backing the shard.
    pub path: String,
    /// One-based inclusive start line.
    pub start_line: u32,
    /// One-based inclusive end line.
    pub end_line: u32,
    /// SHA-256 hash of shard content.
    pub content_hash: String,
    /// Symbols overlapping this shard.
    pub symbol_ids: Vec<String>,
    /// Provenance for the shard.
    pub provenance: Provenance,
}

/// Retrieval query supporting exact, lexical, and graph expansion phases.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetrievalQuery {
    /// Free-form lexical text.
    pub text: Option<String>,
    /// Exact path filter.
    pub path: Option<String>,
    /// Exact symbol name or identifier filter.
    pub symbol: Option<String>,
    /// Maximum number of results.
    pub limit: usize,
    /// Whether graph neighbors should be expanded into the result set.
    pub include_graph_expansion: bool,
}

/// Retrieval result with fused exact, lexical, and graph scores.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RetrievalResult {
    /// Result node or shard identifier.
    pub id: String,
    /// Relative file path.
    pub path: String,
    /// Optional symbol name.
    pub symbol: Option<String>,
    /// Fused deterministic score.
    pub score: f32,
    /// Score components for diagnostics and future reranking.
    pub score_parts: BTreeMap<String, f32>,
    /// Provenance for the result.
    pub provenance: Provenance,
}

/// Freshness state for a file relative to a previous manifest.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshnessState {
    /// Changed files where path exists in both manifests with different content hashes.
    pub changed: Vec<String>,
    /// Deleted files from the previous manifest.
    pub deleted: Vec<String>,
    /// Added files not present in the previous manifest.
    pub added: Vec<String>,
    /// Unchanged files with identical content hashes.
    pub unchanged: Vec<String>,
    /// True when no indexed content changed.
    pub fresh: bool,
}

/// Provenance carried by every project-model observation.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// Relative path or storage file that produced the observation.
    pub path: String,
    /// Optional one-based inclusive start line.
    pub start_line: Option<u32>,
    /// Optional one-based inclusive end line.
    pub end_line: Option<u32>,
    /// Extraction source or subsystem.
    pub source: String,
    /// Redaction-safe fingerprint for the underlying content or episode.
    pub fingerprint: String,
}

/// Redaction-safe tool episode persisted as JSONL.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolEpisode {
    /// Stable timestamp or caller-provided event time.
    pub timestamp: String,
    /// Tool name or capability label.
    pub tool: String,
    /// Redaction-safe input fingerprint.
    pub input_fingerprint: String,
    /// Redaction-safe output fingerprint.
    pub output_fingerprint: String,
    /// Optional status label.
    pub status: String,
    /// Episode provenance.
    pub provenance: Provenance,
}

/// Detected source language.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Language {
    /// Rust source file.
    Rust,
    /// TOML configuration.
    Toml,
    /// Markdown document.
    Markdown,
    /// JSON document.
    Json,
    /// Unknown textual file.
    #[default]
    Unknown,
}

/// A provider-neutral lexical document accepted by the in-crate lexical index.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexicalDocument {
    /// Stable document identifier.
    pub id: String,
    /// Relative file path associated with the document.
    pub path: String,
    /// Optional symbol name associated with the document.
    pub symbol: Option<String>,
    /// Document kind used for deterministic weighting.
    pub kind: LexicalDocumentKind,
    /// Tokenized text surface.
    pub text: String,
    /// Provenance for the searchable text surface.
    pub provenance: Provenance,
}

/// Searchable lexical document kind.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LexicalDocumentKind {
    /// File path and metadata document.
    #[default]
    File,
    /// Source shard metadata document.
    Shard,
    /// Symbol metadata document.
    Symbol,
}

/// BM25-like lexical search hit.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LexicalSearchHit {
    /// Stable matched document identifier.
    pub id: String,
    /// Relative file path for the hit.
    pub path: String,
    /// Optional symbol name for the hit.
    pub symbol: Option<String>,
    /// Deterministic lexical score.
    pub score: f32,
    /// Query tokens matched in this document.
    pub matched_terms: Vec<String>,
    /// Provenance for the hit.
    pub provenance: Provenance,
}

/// Provider-neutral vector query supplied by an external embedding boundary.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct VectorQuery {
    /// Embedding vector generated outside this crate.
    pub embedding: Vec<f32>,
}

/// Vector search hit returned by a typed vector index implementation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct VectorSearchHit {
    /// Result identifier matching a manifest node or lexical document.
    pub id: String,
    /// Similarity score where larger is better.
    pub score: f32,
}

/// Candidate passed through the reranking boundary.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RerankCandidate {
    /// Candidate identifier.
    pub id: String,
    /// Candidate text surface supplied by the caller.
    pub text: String,
}

/// Reranked candidate score returned by a typed reranker implementation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RerankScore {
    /// Candidate identifier.
    pub id: String,
    /// Reranker score where larger is better.
    pub score: f32,
}

/// External compiler, LSP, or SCIP symbol fact accepted by the typed importer.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalSymbolFact {
    /// Stable external symbol identifier.
    pub id: String,
    /// Symbol display name.
    pub name: String,
    /// Symbol kind.
    pub kind: SymbolKind,
    /// Relative file path containing the symbol.
    pub path: String,
    /// One-based inclusive start line.
    pub start_line: u32,
    /// One-based inclusive end line.
    pub end_line: u32,
    /// Source system label, such as `lsp` or `scip`.
    pub source: String,
}

/// External compiler, LSP, or SCIP relationship fact accepted by the typed importer.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalReferenceFact {
    /// Stable source node identifier.
    pub from: String,
    /// Stable target node identifier.
    pub to: String,
    /// Relationship kind.
    pub kind: GraphEdgeKind,
    /// Relative file path containing the reference.
    pub path: String,
    /// Optional one-based inclusive start line.
    pub start_line: Option<u32>,
    /// Optional one-based inclusive end line.
    pub end_line: Option<u32>,
    /// Source system label, such as `lsp` or `scip`.
    pub source: String,
}

/// External facts bundle imported through a compiler/LSP/SCIP boundary.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalFacts {
    /// Symbol facts to merge into the project model.
    pub symbols: Vec<ExternalSymbolFact>,
    /// Reference or call facts to merge into the graph.
    pub references: Vec<ExternalReferenceFact>,
}

/// Retrieval evaluation query with expected relevant identifiers.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetrievalEvalCase {
    /// Query under evaluation.
    pub query: RetrievalQuery,
    /// Relevant result identifiers for this query.
    pub relevant_ids: BTreeSet<String>,
}

/// Aggregated retrieval metrics.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RetrievalEvalReport {
    /// Precision at the evaluated cutoff.
    pub precision_at_k: f32,
    /// Recall at the evaluated cutoff.
    pub recall_at_k: f32,
    /// Mean reciprocal rank across cases.
    pub mean_reciprocal_rank: f32,
}

/// Graph edge coverage report.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphCoverageReport {
    /// Number of expected edges.
    pub expected_edges: usize,
    /// Number of expected edges present in the graph.
    pub covered_edges: usize,
    /// Coverage ratio from 0.0 to 1.0.
    pub coverage: f32,
}

/// Provenance completeness report for manifest model objects.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProvenanceCompletenessReport {
    /// Total checked provenance records.
    pub total: usize,
    /// Records with non-empty source, path, and fingerprint.
    pub complete: usize,
    /// Completeness ratio from 0.0 to 1.0.
    pub completeness: f32,
}

/// Freshness evaluation report for two manifests.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshnessEvalReport {
    /// Freshness state produced by deterministic comparison.
    pub state: FreshnessState,
    /// Whether every manifest file has complete provenance.
    pub provenance_complete: bool,
}

/// Future vector and reranking integration point without provider coupling.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FutureVectorRetrievalScaffold {
    /// Provider-neutral embedding model identifier supplied by future callers.
    pub embedding_model: Option<String>,
    /// Provider-neutral reranker identifier supplied by future callers.
    pub reranker_model: Option<String>,
    /// Whether vector lookup was requested by an outer layer.
    pub requested: bool,
}
