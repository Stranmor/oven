//! Public project-model DTOs and type surfaces.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
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
    /// Provenance for the edge.
    pub provenance: Provenance,
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
