//! Public project-model DTOs and type surfaces.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::PathBuf;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

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
    /// Accepted external exact fact batch provenance.
    #[serde(default)]
    pub external_fact_batches: Vec<ExternalFactBatchMetadata>,
    /// Manifest-level fingerprint for accepted external exact fact batches.
    #[serde(default)]
    pub external_facts_fingerprint: String,
    /// Content shards used by retrieval.
    pub shards: Vec<ShardManifest>,
    /// Manifest-level content hash over deterministic file hashes and accepted
    /// external fact fingerprints.
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
    /// Exact fact accepted from a typed external batch whose source contract and
    /// manifest baseline were validated before ingestion.
    ExactCompiler,
    /// High-confidence syntax heuristic produced without type resolution.
    #[default]
    HeuristicHigh,
    /// Low-confidence syntax heuristic that may require later compiler
    /// validation.
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
    /// Symbol or callable invokes another callable by name or imported compiler
    /// fact.
    Calls,
    /// Symbol or file references another symbol.
    References,
    /// Task depends on file, symbol, shard, decision, retrieved evidence, tool
    /// episode, or eval evidence.
    TaskDependsOn,
    /// Decision is supported by file, symbol, shard, task, retrieved evidence,
    /// tool episode, or eval evidence.
    DecisionSupportedBy,
    /// Retrieved evidence cites a file, symbol, or shard.
    EvidenceCites,
    /// Tool episode produced or consumed a graph node.
    ToolEpisodeRelates,
    /// Eval case covers a graph node or relationship.
    EvalCovers,
    /// Retrieval expansion relationship.
    #[default]
    Related,
}

/// Stable typed identifier for a knowledge graph node.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum KnowledgeGraphNodeId {
    /// Source file node keyed by relative path.
    File(String),
    /// Source symbol node keyed by stable symbol identifier.
    Symbol(String),
    /// Retrieval shard node keyed by stable shard identifier.
    Shard(String),
    /// Agent task node keyed by durable task identifier.
    Task(String),
    /// Architecture or product decision node keyed by durable decision
    /// identifier.
    Decision(String),
    /// Retrieved external or internal evidence node keyed by durable evidence
    /// identifier.
    RetrievedEvidence(String),
    /// Tool episode node keyed by durable tool episode identifier.
    ToolEpisode(String),
    /// Evaluation case node keyed by durable eval case identifier.
    EvalCase(String),
}

impl KnowledgeGraphNodeId {
    /// Returns a stable string representation suitable for legacy edge interop.
    pub fn as_legacy_id(&self) -> String {
        match self {
            Self::File(value) => value.clone(),
            Self::Symbol(value)
            | Self::Shard(value)
            | Self::Task(value)
            | Self::Decision(value)
            | Self::RetrievedEvidence(value)
            | Self::ToolEpisode(value)
            | Self::EvalCase(value) => value.clone(),
        }
    }
}

impl Default for KnowledgeGraphNodeId {
    fn default() -> Self {
        Self::File(String::new())
    }
}

/// Typed knowledge graph node kind.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum KnowledgeGraphNodeKind {
    /// Source file node.
    #[default]
    File,
    /// Source symbol node.
    Symbol,
    /// Retrieval shard node.
    Shard,
    /// Agent task node.
    Task,
    /// Architecture or product decision node.
    Decision,
    /// Retrieved evidence node.
    RetrievedEvidence,
    /// Tool episode node.
    ToolEpisode,
    /// Evaluation case node.
    EvalCase,
}

/// Stable file node payload for the knowledge graph.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileGraphNode {
    /// Stable typed node identifier.
    pub id: KnowledgeGraphNodeId,
    /// Relative source path.
    pub path: String,
    /// File content hash at graph construction time.
    pub content_hash: String,
    /// Provenance for the file evidence.
    pub provenance: Provenance,
}

/// Stable symbol node payload for the knowledge graph.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolGraphNode {
    /// Stable typed node identifier.
    pub id: KnowledgeGraphNodeId,
    /// Stable symbol identifier.
    pub symbol_id: String,
    /// Symbol display name.
    pub name: String,
    /// Containing relative source path.
    pub path: String,
    /// Provenance for the symbol evidence.
    pub provenance: Provenance,
}

/// Stable shard node payload for the knowledge graph.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardGraphNode {
    /// Stable typed node identifier.
    pub id: KnowledgeGraphNodeId,
    /// Stable shard identifier.
    pub shard_id: String,
    /// Relative source path.
    pub path: String,
    /// Shard content hash at graph construction time.
    pub content_hash: String,
    /// Provenance for the shard evidence.
    pub provenance: Provenance,
}

/// Agent task node payload for the knowledge graph.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskGraphNode {
    /// Stable typed node identifier.
    pub id: KnowledgeGraphNodeId,
    /// Human-readable task title.
    pub title: String,
    /// Current task status label supplied by the caller.
    pub status: String,
    /// Provenance for the task record.
    pub provenance: Provenance,
}

/// Decision node payload for the knowledge graph.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionGraphNode {
    /// Stable typed node identifier.
    pub id: KnowledgeGraphNodeId,
    /// Human-readable decision title.
    pub title: String,
    /// Decision outcome or selected option.
    pub outcome: String,
    /// Provenance for the decision record.
    pub provenance: Provenance,
}

/// Retrieved evidence node payload for the knowledge graph.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetrievedEvidenceGraphNode {
    /// Stable typed node identifier.
    pub id: KnowledgeGraphNodeId,
    /// Retrieval result identifier or external evidence key.
    pub evidence_id: String,
    /// Relative source path associated with the evidence.
    pub path: String,
    /// Evidence freshness state at graph construction time.
    pub freshness: EvidenceFreshness,
    /// Provenance for the retrieved evidence.
    pub provenance: Provenance,
}

/// Tool episode node payload for the knowledge graph.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolEpisodeGraphNode {
    /// Stable typed node identifier.
    pub id: KnowledgeGraphNodeId,
    /// Tool name or capability label.
    pub tool: String,
    /// Tool episode status label.
    pub status: String,
    /// Provenance for the tool episode.
    pub provenance: Provenance,
}

/// Evaluation case node payload for the knowledge graph.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalCaseGraphNode {
    /// Stable typed node identifier.
    pub id: KnowledgeGraphNodeId,
    /// Evaluation case title.
    pub title: String,
    /// Stable expected identifiers covered by this case.
    pub expected_ids: BTreeSet<String>,
    /// Provenance for the eval case.
    pub provenance: Provenance,
}

/// Typed knowledge graph node payload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum KnowledgeGraphNode {
    /// Source file node.
    File(FileGraphNode),
    /// Source symbol node.
    Symbol(SymbolGraphNode),
    /// Retrieval shard node.
    Shard(ShardGraphNode),
    /// Agent task node.
    Task(TaskGraphNode),
    /// Architecture or product decision node.
    Decision(DecisionGraphNode),
    /// Retrieved evidence node.
    RetrievedEvidence(RetrievedEvidenceGraphNode),
    /// Tool episode node.
    ToolEpisode(ToolEpisodeGraphNode),
    /// Evaluation case node.
    EvalCase(EvalCaseGraphNode),
}

impl KnowledgeGraphNode {
    /// Returns the stable typed identifier for this node.
    pub fn id(&self) -> &KnowledgeGraphNodeId {
        match self {
            Self::File(node) => &node.id,
            Self::Symbol(node) => &node.id,
            Self::Shard(node) => &node.id,
            Self::Task(node) => &node.id,
            Self::Decision(node) => &node.id,
            Self::RetrievedEvidence(node) => &node.id,
            Self::ToolEpisode(node) => &node.id,
            Self::EvalCase(node) => &node.id,
        }
    }

    /// Returns the typed node kind for this node.
    pub fn kind(&self) -> KnowledgeGraphNodeKind {
        match self {
            Self::File(_) => KnowledgeGraphNodeKind::File,
            Self::Symbol(_) => KnowledgeGraphNodeKind::Symbol,
            Self::Shard(_) => KnowledgeGraphNodeKind::Shard,
            Self::Task(_) => KnowledgeGraphNodeKind::Task,
            Self::Decision(_) => KnowledgeGraphNodeKind::Decision,
            Self::RetrievedEvidence(_) => KnowledgeGraphNodeKind::RetrievedEvidence,
            Self::ToolEpisode(_) => KnowledgeGraphNodeKind::ToolEpisode,
            Self::EvalCase(_) => KnowledgeGraphNodeKind::EvalCase,
        }
    }

    /// Returns provenance for this node.
    pub fn provenance(&self) -> &Provenance {
        match self {
            Self::File(node) => &node.provenance,
            Self::Symbol(node) => &node.provenance,
            Self::Shard(node) => &node.provenance,
            Self::Task(node) => &node.provenance,
            Self::Decision(node) => &node.provenance,
            Self::RetrievedEvidence(node) => &node.provenance,
            Self::ToolEpisode(node) => &node.provenance,
            Self::EvalCase(node) => &node.provenance,
        }
    }
}

/// Typed edge connecting knowledge graph nodes.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeGraphEdge {
    /// Stable typed source node identifier.
    pub from: KnowledgeGraphNodeId,
    /// Stable typed target node identifier.
    pub to: KnowledgeGraphNodeId,
    /// Edge kind.
    pub kind: GraphEdgeKind,
    /// Confidence from 0.0 to 1.0.
    pub confidence: f32,
    /// Semantic confidence class used to prevent overclaiming heuristic facts.
    pub confidence_kind: EdgeConfidence,
    /// Provenance for the edge.
    pub provenance: Provenance,
}

/// Typed knowledge graph with validated node endpoints and deterministic
/// ordering.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    /// Graph nodes in deterministic identifier order.
    pub nodes: Vec<KnowledgeGraphNode>,
    /// Graph edges in deterministic source-target-kind-confidence-provenance
    /// order.
    pub edges: Vec<KnowledgeGraphEdge>,
}

impl KnowledgeGraph {
    /// Builds a validated knowledge graph from typed nodes and edges.
    ///
    /// # Arguments
    ///
    /// * `nodes` - Candidate typed graph nodes.
    /// * `edges` - Candidate typed graph edges.
    ///
    /// # Errors
    ///
    /// Returns an error when an edge endpoint is absent from the node set, a
    /// node identifier is duplicated, or an edge confidence is outside the
    /// closed 0.0 to 1.0 range.
    pub fn new(
        mut nodes: Vec<KnowledgeGraphNode>,
        mut edges: Vec<KnowledgeGraphEdge>,
    ) -> Result<Self> {
        nodes.sort_by(|left, right| left.id().cmp(right.id()));
        for pair in nodes.windows(2) {
            if let [left, right] = pair
                && left.id() == right.id()
            {
                bail!("knowledge graph node id is duplicated: {:?}", left.id());
            }
        }
        let node_ids = nodes
            .iter()
            .map(|node| node.id().clone())
            .collect::<BTreeSet<_>>();
        for edge in &edges {
            if !(0.0..=1.0).contains(&edge.confidence) || !edge.confidence.is_finite() {
                bail!(
                    "knowledge graph edge confidence is invalid: {}",
                    edge.confidence
                );
            }
            if !node_ids.contains(&edge.from) {
                bail!("knowledge graph edge source is missing: {:?}", edge.from);
            }
            if !node_ids.contains(&edge.to) {
                bail!("knowledge graph edge target is missing: {:?}", edge.to);
            }
        }
        edges.sort_by(compare_knowledge_graph_edges);
        Ok(Self { nodes, edges })
    }

    /// Builds file, symbol, shard, and legacy graph edges from a project
    /// manifest.
    ///
    /// # Arguments
    ///
    /// * `manifest` - Project manifest used as source evidence.
    ///
    /// # Errors
    ///
    /// Returns an error when manifest edges point to nodes absent from the
    /// graph surface.
    pub fn from_manifest(manifest: &ProjectManifest) -> Result<Self> {
        let mut nodes = Vec::new();
        for file in &manifest.files {
            nodes.push(KnowledgeGraphNode::File(FileGraphNode {
                id: KnowledgeGraphNodeId::File(file.path.clone()),
                path: file.path.clone(),
                content_hash: file.content_hash.clone(),
                provenance: file.provenance.clone(),
            }));
        }
        for symbol in &manifest.symbols {
            nodes.push(KnowledgeGraphNode::Symbol(SymbolGraphNode {
                id: KnowledgeGraphNodeId::Symbol(symbol.id.clone()),
                symbol_id: symbol.id.clone(),
                name: symbol.name.clone(),
                path: symbol.path.clone(),
                provenance: symbol.provenance.clone(),
            }));
        }
        for shard in &manifest.shards {
            nodes.push(KnowledgeGraphNode::Shard(ShardGraphNode {
                id: KnowledgeGraphNodeId::Shard(shard.id.clone()),
                shard_id: shard.id.clone(),
                path: shard.path.clone(),
                content_hash: shard.content_hash.clone(),
                provenance: shard.provenance.clone(),
            }));
        }
        let mut node_ids = nodes
            .iter()
            .map(|node| node.id().clone())
            .collect::<BTreeSet<_>>();
        let mut external_nodes = BTreeMap::<KnowledgeGraphNodeId, KnowledgeGraphNode>::new();
        let edges = manifest
            .edges
            .iter()
            .map(|edge| {
                let from = typed_legacy_node_id_or_external(&edge.from, &node_ids);
                let to = typed_legacy_node_id_or_external(&edge.to, &node_ids);
                for node_id in [from.clone(), to.clone()] {
                    if !node_ids.contains(&node_id) {
                        external_nodes.insert(
                            node_id.clone(),
                            KnowledgeGraphNode::RetrievedEvidence(RetrievedEvidenceGraphNode {
                                id: node_id.clone(),
                                evidence_id: node_id.as_legacy_id(),
                                path: edge.provenance.path.clone(),
                                freshness: EvidenceFreshness::Fresh,
                                provenance: edge.provenance.clone(),
                            }),
                        );
                        node_ids.insert(node_id);
                    }
                }
                KnowledgeGraphEdge {
                    from,
                    to,
                    kind: edge.kind.clone(),
                    confidence: edge.confidence,
                    confidence_kind: edge.confidence_kind.clone(),
                    provenance: edge.provenance.clone(),
                }
            })
            .collect();
        nodes.extend(external_nodes.into_values());
        Self::new(nodes, edges)
    }
}

fn typed_legacy_node_id_or_external(
    value: &str,
    known_ids: &BTreeSet<KnowledgeGraphNodeId>,
) -> KnowledgeGraphNodeId {
    typed_legacy_node_id(value, known_ids)
        .unwrap_or_else(|| KnowledgeGraphNodeId::RetrievedEvidence(value.to_string()))
}

fn compare_knowledge_graph_edges(
    left: &KnowledgeGraphEdge,
    right: &KnowledgeGraphEdge,
) -> Ordering {
    left.from
        .cmp(&right.from)
        .then_with(|| left.to.cmp(&right.to))
        .then_with(|| left.kind.cmp(&right.kind))
        .then_with(|| left.confidence_kind.cmp(&right.confidence_kind))
        .then_with(|| compare_f32_total(left.confidence, right.confidence))
        .then_with(|| compare_provenance(&left.provenance, &right.provenance))
}

fn compare_context_pack_evidence(
    left: &ContextPackEvidence,
    right: &ContextPackEvidence,
) -> Ordering {
    left.freshness
        .cmp(&right.freshness)
        .then_with(|| left.path.cmp(&right.path))
        .then_with(|| left.id.cmp(&right.id))
        .then_with(|| left.source.cmp(&right.source))
        .then_with(|| compare_f32_total(left.score, right.score))
        .then_with(|| compare_provenance(&left.provenance, &right.provenance))
}

fn compare_provenance(left: &Provenance, right: &Provenance) -> Ordering {
    left.path
        .cmp(&right.path)
        .then_with(|| left.start_line.cmp(&right.start_line))
        .then_with(|| left.end_line.cmp(&right.end_line))
        .then_with(|| left.source.cmp(&right.source))
        .then_with(|| left.fingerprint.cmp(&right.fingerprint))
}

fn compare_f32_total(left: f32, right: f32) -> Ordering {
    left.total_cmp(&right)
}

fn typed_legacy_node_id(
    value: &str,
    known_ids: &BTreeSet<KnowledgeGraphNodeId>,
) -> Option<KnowledgeGraphNodeId> {
    let candidates = [
        KnowledgeGraphNodeId::File(value.to_string()),
        KnowledgeGraphNodeId::Symbol(value.to_string()),
        KnowledgeGraphNodeId::Shard(value.to_string()),
    ];
    candidates
        .into_iter()
        .find(|candidate| known_ids.contains(candidate))
}

/// Freshness classification for evidence included in context packaging or graph
/// evidence.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EvidenceFreshness {
    /// Evidence path is absent from the freshness state or explicitly
    /// unchanged.
    #[default]
    Fresh,
    /// Evidence path was added after the baseline and should be treated as
    /// fresh current evidence.
    Added,
    /// Evidence path changed relative to the baseline and must be reviewed
    /// before use.
    Changed,
    /// Evidence path was deleted relative to the baseline and cannot be used as
    /// current evidence.
    Deleted,
}

/// Policy for stale evidence during context pack construction.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum StaleEvidencePolicy {
    /// Reject context pack construction if stale evidence is selected.
    Reject,
    /// Include selected evidence while marking stale state explicitly.
    #[default]
    Mark,
}

/// Selected retrieval result, shard, or ad-hoc evidence used to build a context
/// pack.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ContextPackSelection {
    /// Retrieved results selected by the retrieval layer.
    pub retrieval_results: Vec<RetrievalResult>,
    /// Shard manifests selected by structural or graph expansion.
    pub shards: Vec<ShardManifest>,
    /// Additional typed evidence selected by external evaluators or
    /// integrations.
    pub evidence: Vec<ContextPackEvidence>,
    /// Freshness state used to classify included paths.
    pub freshness: FreshnessState,
    /// Policy for stale evidence.
    pub stale_policy: StaleEvidencePolicy,
}

/// Evidence item included in a context pack.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ContextPackEvidence {
    /// Stable evidence identifier.
    pub id: String,
    /// Relative file path associated with the evidence.
    pub path: String,
    /// Optional symbol display name associated with the evidence.
    pub symbol: Option<String>,
    /// Evidence source class.
    pub source: ContextPackEvidenceSource,
    /// Freshness classification for this evidence.
    pub freshness: EvidenceFreshness,
    /// Provenance for the evidence.
    pub provenance: Provenance,
    /// Deterministic score or priority supplied by retrieval.
    pub score: f32,
}

/// Source class for context-pack evidence.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ContextPackEvidenceSource {
    /// Evidence derived from a retrieval result.
    #[default]
    RetrievalResult,
    /// Evidence derived from a structural shard.
    Shard,
    /// Evidence supplied directly by a typed caller.
    DirectEvidence,
}

/// Hash-only deterministic identifier for a persisted context pack artifact.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ContextPackArtifactId(String);

impl ContextPackArtifactId {
    /// Builds a context pack artifact identifier from a 64-character hex hash.
    ///
    /// # Arguments
    ///
    /// * `value` - Candidate lowercase SHA-256 hex string.
    ///
    /// # Errors
    ///
    /// Returns an error when the value is not a hash-only artifact identifier.
    pub fn new(value: String) -> Result<Self> {
        if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            bail!("context pack artifact id must be a 64-character hex hash");
        }
        Ok(Self(value.to_ascii_lowercase()))
    }

    /// Returns the validated hash-only artifact identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ContextPackArtifactId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Deterministic context package consumed by model-context assembly layers.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ContextPack {
    /// Format version for deterministic serialization.
    pub version: u32,
    /// Manifest hash used to build this context pack.
    pub manifest_hash: String,
    /// Evidence entries sorted deterministically by freshness, path,
    /// identifier, source, score, and provenance.
    pub evidence: Vec<ContextPackEvidence>,
    /// All provenance records required to audit the pack.
    pub provenance: Vec<Provenance>,
}

impl ContextPack {
    /// Builds a deterministic context pack from selected retrieval, shard, and
    /// direct evidence.
    ///
    /// # Arguments
    ///
    /// * `manifest` - Manifest that owns current project evidence.
    /// * `selection` - Selected evidence and freshness policy.
    ///
    /// # Errors
    ///
    /// Returns an error when provenance is incomplete, a score is not finite,
    /// or stale evidence is rejected by policy.
    pub fn from_selection(
        manifest: &ProjectManifest,
        selection: ContextPackSelection,
    ) -> Result<Self> {
        let mut evidence = Vec::new();
        for result in selection.retrieval_results {
            evidence.push(ContextPackEvidence {
                id: result.id,
                path: result.path.clone(),
                symbol: result.symbol,
                source: ContextPackEvidenceSource::RetrievalResult,
                freshness: classify_evidence_freshness(&result.path, &selection.freshness),
                provenance: result.provenance,
                score: result.score,
            });
        }
        for shard in selection.shards {
            evidence.push(ContextPackEvidence {
                id: shard.id,
                path: shard.path.clone(),
                symbol: None,
                source: ContextPackEvidenceSource::Shard,
                freshness: classify_evidence_freshness(&shard.path, &selection.freshness),
                provenance: shard.provenance,
                score: 0.0,
            });
        }
        for mut direct in selection.evidence {
            direct.freshness = classify_evidence_freshness(&direct.path, &selection.freshness);
            evidence.push(direct);
        }
        for item in &evidence {
            if !item.score.is_finite() {
                bail!("context pack evidence score is invalid: {}", item.id);
            }
            if !item.provenance.is_complete() {
                bail!(
                    "context pack evidence has incomplete provenance: {}",
                    item.id
                );
            }
            if selection.stale_policy == StaleEvidencePolicy::Reject
                && matches!(
                    item.freshness,
                    EvidenceFreshness::Changed | EvidenceFreshness::Deleted
                )
            {
                bail!("context pack evidence is stale: {}", item.id);
            }
        }
        evidence.sort_by(compare_context_pack_evidence);
        let mut provenance = evidence
            .iter()
            .map(|item| item.provenance.clone())
            .collect::<Vec<_>>();
        provenance.sort_by(compare_provenance);
        Ok(Self {
            version: 1,
            manifest_hash: manifest.manifest_hash.clone(),
            evidence,
            provenance,
        })
    }

    /// Serializes this context pack as stable pretty JSON.
    ///
    /// # Errors
    ///
    /// Returns an error when JSON serialization fails.
    pub fn to_stable_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

impl Provenance {
    /// Returns true when the provenance contains source, path, and fingerprint.
    pub fn is_complete(&self) -> bool {
        !self.path.is_empty() && !self.source.is_empty() && !self.fingerprint.is_empty()
    }
}

/// Classifies a path against a typed freshness state.
///
/// # Arguments
///
/// * `path` - Relative path to classify.
/// * `freshness` - Freshness state generated from manifest comparison.
pub fn classify_evidence_freshness(path: &str, freshness: &FreshnessState) -> EvidenceFreshness {
    if freshness.deleted.iter().any(|value| value == path) {
        EvidenceFreshness::Deleted
    } else if freshness.changed.iter().any(|value| value == path) {
        EvidenceFreshness::Changed
    } else if freshness.added.iter().any(|value| value == path) {
        EvidenceFreshness::Added
    } else {
        EvidenceFreshness::Fresh
    }
}

/// Strategy used to construct deterministic context shards.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardStrategy {
    /// Use semantic Rust symbol ranges when available and line fallback for
    /// files without supported semantic ranges.
    #[default]
    RustSemanticWithLineFallback,
    /// Use fixed line chunks for all files.
    FixedLineChunks {
        /// Maximum lines per fallback chunk.
        chunk_size: usize,
    },
}

impl ShardStrategy {
    /// Returns the default fallback chunk size.
    pub fn default_chunk_size(&self) -> usize {
        match self {
            Self::RustSemanticWithLineFallback => 80,
            Self::FixedLineChunks { chunk_size } => *chunk_size,
        }
    }
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

/// Retrieval scoring weights used by the planner.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetrievalScoringWeights {
    /// Score assigned to exact path matches.
    pub exact_path: f32,
    /// Score assigned to exact symbol matches.
    pub exact_symbol: f32,
    /// Multiplier applied to lexical scores.
    pub lexical: f32,
    /// Multiplier applied to vector scores.
    pub vector: f32,
    /// Multiplier applied to graph edge confidence.
    pub graph: f32,
    /// Multiplier applied to reranker scores.
    pub rerank: f32,
}

impl Default for RetrievalScoringWeights {
    fn default() -> Self {
        Self {
            exact_path: 100.0,
            exact_symbol: 100.0,
            lexical: 1.0,
            vector: 1.0,
            graph: 10.0,
            rerank: 1.0,
        }
    }
}

impl RetrievalScoringWeights {
    /// Validates that all weights are finite and non-negative.
    ///
    /// # Errors
    ///
    /// Returns an error when any weight is negative, NaN, or infinite.
    pub fn validate(&self) -> Result<()> {
        let weights = [
            self.exact_path,
            self.exact_symbol,
            self.lexical,
            self.vector,
            self.graph,
            self.rerank,
        ];
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            bail!("retrieval scoring weights must be finite non-negative values");
        }
        Ok(())
    }
}

/// Planned retrieval phases derived from a query and integration boundaries.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetrievalScoringPlan {
    /// Whether exact path/symbol matching is active.
    pub exact: bool,
    /// Whether lexical retrieval is active.
    pub lexical: bool,
    /// Whether vector retrieval is active.
    pub vector: bool,
    /// Whether graph expansion is active.
    pub graph: bool,
    /// Whether reranking is active.
    pub rerank: bool,
}

/// Retrieval query supporting exact, lexical, and graph expansion phases.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetrievalQuery {
    /// Free-form lexical text.
    pub text: Option<String>,
    /// Exact path filter.
    pub path: Option<String>,
    /// Directory or path prefix scope applied before scoring and truncation.
    pub path_prefix: Option<String>,
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
    /// Changed files where path exists in both manifests with different content
    /// hashes.
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

/// Freshness proof strength for a manifest compared against the current filesystem.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FreshnessProofLevel {
    /// All ignore-aware filesystem inputs were scanned, including added files.
    FullFilesystem,
    /// Only files already persisted in the manifest were checked.
    #[default]
    IndexedFilesOnly,
}

/// Freshness evaluation for a persisted manifest against the current filesystem.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestFreshnessEvaluation {
    /// Deterministic freshness state.
    pub state: FreshnessState,
    /// Proof strength used to produce the freshness state.
    pub proof_level: FreshnessProofLevel,
}

impl ManifestFreshnessEvaluation {
    /// Returns true only when the manifest is proven fresh against the full filesystem.
    pub fn can_inject(&self) -> bool {
        self.state.fresh && self.proof_level == FreshnessProofLevel::FullFilesystem
    }
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

/// External exact fact batch metadata persisted into manifests after validated
/// ingestion.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ExternalFactBatchMetadata {
    /// Typed source boundary for the external fact producer.
    pub source: ExternalFactSource,
    /// Human-readable source label or tool name.
    pub source_label: String,
    /// Optional producer version when supplied by the caller.
    pub tool_version: Option<String>,
    /// Workspace root identity used by the producer.
    pub workspace_root: String,
    /// Redaction-safe fingerprint for the source artifact or source snapshot.
    pub source_artifact_fingerprint: String,
    /// Manifest baseline identity the batch was produced against.
    pub manifest_hash_input: String,
    /// Deterministic fingerprint over batch metadata and fact payloads.
    pub batch_fingerprint: String,
}

/// External exact fact batch imported through the safe fixture-backed boundary.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalFactBatch {
    /// Batch-level metadata and deterministic fingerprint.
    pub metadata: ExternalFactBatchMetadata,
    /// Symbol and reference facts carried by this batch.
    pub facts: TypedExternalFacts,
}

/// Stable issue code emitted by external exact fact ingestion.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ExternalFactIngestionIssueCode {
    /// Batch metadata is missing source, artifact, workspace, or baseline data.
    IncompleteBatchMetadata,
    /// Batch workspace identity does not match the current manifest workspace.
    WorkspaceRootMismatch,
    /// Batch metadata baseline does not match the current manifest baseline.
    ManifestBaselineMismatch,
    /// Batch fingerprint does not match deterministic recomputation.
    BatchFingerprintMismatch,
    /// A symbol fact repeats a symbol identifier inside the same batch.
    DuplicateSymbolId,
    /// A symbol fact would overwrite an existing manifest-owned symbol identifier.
    ConflictingManifestSymbolId,
    /// A symbol fact points at a file not present in the manifest.
    MissingSymbolFileEndpoint,
    /// A reference provenance path is absent from the manifest file set.
    MissingReferenceFileEndpoint,
    /// A reference or call edge source is absent from accepted file, symbol, or shard endpoints.
    MissingReferenceSourceEndpoint,
    /// A reference or call edge target is absent from accepted file, symbol, or shard endpoints.
    MissingReferenceTargetEndpoint,
    /// A fact cannot be accepted as exact under the batch source contract.
    InvalidExactSourceContract,
}

/// Redaction-safe issue emitted by external exact fact ingestion.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalFactIngestionIssue {
    /// Stable machine-readable issue code.
    pub code: ExternalFactIngestionIssueCode,
    /// Optional rejected endpoint or fact identifier.
    pub endpoint: Option<String>,
    /// Redaction-safe detail containing only labels, paths, and fingerprints.
    pub detail: String,
}

/// Result of applying one validated external exact fact batch.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalFactIngestionReport {
    /// Number of accepted symbol facts.
    pub accepted_symbols: usize,
    /// Number of accepted reference or call edge facts.
    pub accepted_edges: usize,
    /// Number of duplicate accepted edge facts removed deterministically.
    pub deduplicated_edges: usize,
    /// Batch metadata persisted into the manifest.
    pub batch_metadata: ExternalFactBatchMetadata,
}

/// Typed external fact source accepted at the exact fact ingestion boundary.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ExternalFactSource {
    /// Language Server Protocol facts.
    Lsp,
    /// Sourcegraph SCIP facts.
    Scip,
    /// Rust compiler or rust-analyzer compiler-derived facts.
    Compiler,
    /// Custom compatibility source label retained from older public DTOs.
    Custom(String),
    /// Unknown typed source with no legacy label.
    #[default]
    Unknown,
}

impl ExternalFactSource {
    /// Converts a legacy source label into a typed source boundary.
    ///
    /// # Arguments
    ///
    /// * `label` - Legacy external source label.
    pub fn from_label(label: &str) -> Self {
        match label.to_ascii_lowercase().as_str() {
            "lsp" | "rust-analyzer" | "rust_analyzer" => Self::Lsp,
            "scip" => Self::Scip,
            "compiler" | "rustc" => Self::Compiler,
            _ if label.is_empty() => Self::Unknown,
            _ => Self::Custom(label.to_string()),
        }
    }

    /// Returns a stable provenance label for this typed source.
    pub fn provenance_label(&self) -> String {
        match self {
            Self::Lsp => "lsp".to_string(),
            Self::Scip => "scip".to_string(),
            Self::Compiler => "compiler".to_string(),
            Self::Custom(label) => label.clone(),
            Self::Unknown => "external-unknown".to_string(),
        }
    }
}

/// External symbol fact accepted by the typed importer only as part of a
/// validated external fact batch.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypedExternalSymbolFact {
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
    /// Typed external source boundary.
    pub source: ExternalFactSource,
}

/// External relationship fact accepted by the typed importer only as part of a
/// validated external fact batch.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypedExternalReferenceFact {
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
    /// Typed external source boundary.
    pub source: ExternalFactSource,
}

/// Typed external facts bundle imported through a validated batch boundary.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypedExternalFacts {
    /// Symbol facts to merge into the project model.
    pub symbols: Vec<TypedExternalSymbolFact>,
    /// Reference or call facts to merge into the graph.
    pub references: Vec<TypedExternalReferenceFact>,
}

/// Legacy external compiler, LSP, or SCIP symbol fact accepted for public API
/// compatibility.
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

/// Legacy external compiler, LSP, or SCIP relationship fact accepted for public
/// API compatibility.
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

/// Legacy external facts bundle imported through a compiler/LSP/SCIP boundary.
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

/// Stable issue code emitted by artifact, episode, and linkage evaluators.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EvidenceLedgerEvalIssueCode {
    /// A referenced artifact does not exist in the persisted context-pack store.
    MissingArtifact,
    /// A persisted artifact could not be decoded or failed its hash readback.
    CorruptArtifact,
    /// A context-pack artifact contains no evidence.
    EmptyArtifactEvidence,
    /// A context-pack artifact has incomplete provenance.
    IncompleteArtifactProvenance,
    /// A context-pack artifact contains changed or deleted evidence.
    StaleArtifactEvidence,
    /// A tool episode is missing an input fingerprint.
    EmptyEpisodeInputFingerprint,
    /// A tool episode is missing an output fingerprint.
    EmptyEpisodeOutputFingerprint,
    /// A tool episode has incomplete provenance.
    IncompleteEpisodeProvenance,
    /// Multiple tool episodes resolve to the same deterministic identity.
    DuplicateEpisodeIdentity,
    /// A tool episode references no context-pack artifact.
    MissingEpisodeArtifactReference,
    /// A tool episode references a context-pack artifact that is absent.
    MissingLinkedArtifact,
}

/// Redaction-safe issue emitted by artifact, episode, and linkage evaluators.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceLedgerEvalIssue {
    /// Stable machine-readable issue code.
    pub code: EvidenceLedgerEvalIssueCode,
    /// Optional context-pack artifact identifier associated with the issue.
    pub artifact_id: Option<String>,
    /// Optional redaction-safe deterministic episode fingerprint associated with the issue.
    pub episode_fingerprint: Option<String>,
    /// Redaction-safe detail string containing only codes, hashes, and storage labels.
    pub detail: String,
}

/// Evaluation report for persisted context-pack artifacts.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextPackArtifactEvalReport {
    /// Number of artifact identifiers checked.
    pub checked: usize,
    /// True when all checked artifacts satisfy structural invariants.
    pub valid: bool,
    /// Redaction-safe issues discovered during artifact evaluation.
    pub issues: Vec<EvidenceLedgerEvalIssue>,
}

/// Evaluation report for persisted tool episodes.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolEpisodeEvalReport {
    /// Number of tool episodes checked.
    pub checked: usize,
    /// True when all checked episodes satisfy structural invariants.
    pub valid: bool,
    /// Redaction-safe issues discovered during episode evaluation.
    pub issues: Vec<EvidenceLedgerEvalIssue>,
}

/// Evaluation report for context-pack artifact and tool-episode linkage.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceLedgerLinkageReport {
    /// Number of context-pack artifact identifiers checked.
    pub artifact_count: usize,
    /// Number of tool episodes checked.
    pub episode_count: usize,
    /// Number of episodes linked to an existing context-pack artifact.
    pub linked_count: usize,
    /// Redaction-safe issues discovered during linkage evaluation.
    pub issues: Vec<EvidenceLedgerEvalIssue>,
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

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::indexer::tests::fixture_project;
    use crate::util::{fingerprint, provenance};
    use crate::{ProjectIndexer, retrieve};

    #[test]
    fn knowledge_graph_connects_tasks_decisions_and_retrieved_evidence_to_code_evidence()
    -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let file = manifest
            .files
            .iter()
            .find(|file| file.path == "src/lib.rs")
            .expect("fixture should include src/lib.rs");
        let symbol = manifest
            .symbols
            .iter()
            .find(|symbol| symbol.name == "Root")
            .expect("fixture should include Root symbol");
        let nodes = vec![
            KnowledgeGraphNode::File(FileGraphNode {
                id: KnowledgeGraphNodeId::File(file.path.clone()),
                path: file.path.clone(),
                content_hash: file.content_hash.clone(),
                provenance: file.provenance.clone(),
            }),
            KnowledgeGraphNode::Symbol(SymbolGraphNode {
                id: KnowledgeGraphNodeId::Symbol(symbol.id.clone()),
                symbol_id: symbol.id.clone(),
                name: symbol.name.clone(),
                path: symbol.path.clone(),
                provenance: symbol.provenance.clone(),
            }),
            KnowledgeGraphNode::Task(TaskGraphNode {
                id: KnowledgeGraphNodeId::Task("task:final-gate".to_string()),
                title: "Implement final gate fixes".to_string(),
                status: "completed".to_string(),
                provenance: provenance("gate", Some(1), Some(1), "test", "task"),
            }),
            KnowledgeGraphNode::Decision(DecisionGraphNode {
                id: KnowledgeGraphNodeId::Decision("decision:typed-context-pack".to_string()),
                title: "Use typed context packaging".to_string(),
                outcome: "ContextPack".to_string(),
                provenance: provenance("gate", Some(2), Some(2), "test", "decision"),
            }),
            KnowledgeGraphNode::RetrievedEvidence(RetrievedEvidenceGraphNode {
                id: KnowledgeGraphNodeId::RetrievedEvidence("evidence:root-symbol".to_string()),
                evidence_id: symbol.id.clone(),
                path: symbol.path.clone(),
                freshness: EvidenceFreshness::Fresh,
                provenance: symbol.provenance.clone(),
            }),
        ];
        let edges = vec![
            KnowledgeGraphEdge {
                from: KnowledgeGraphNodeId::Task("task:final-gate".to_string()),
                to: KnowledgeGraphNodeId::File(file.path.clone()),
                kind: GraphEdgeKind::TaskDependsOn,
                confidence: 1.0,
                confidence_kind: EdgeConfidence::ExactCompiler,
                provenance: provenance("gate", Some(3), Some(3), "test", "task-file-edge"),
            },
            KnowledgeGraphEdge {
                from: KnowledgeGraphNodeId::Decision("decision:typed-context-pack".to_string()),
                to: KnowledgeGraphNodeId::Symbol(symbol.id.clone()),
                kind: GraphEdgeKind::DecisionSupportedBy,
                confidence: 1.0,
                confidence_kind: EdgeConfidence::ExactCompiler,
                provenance: provenance("gate", Some(4), Some(4), "test", "decision-symbol-edge"),
            },
            KnowledgeGraphEdge {
                from: KnowledgeGraphNodeId::RetrievedEvidence("evidence:root-symbol".to_string()),
                to: KnowledgeGraphNodeId::Symbol(symbol.id.clone()),
                kind: GraphEdgeKind::EvidenceCites,
                confidence: 1.0,
                confidence_kind: EdgeConfidence::ExactCompiler,
                provenance: provenance("gate", Some(5), Some(5), "test", "evidence-symbol-edge"),
            },
        ];

        let actual = KnowledgeGraph::new(nodes, edges)?;
        let expected = BTreeSet::from([
            GraphEdgeKind::DecisionSupportedBy,
            GraphEdgeKind::EvidenceCites,
            GraphEdgeKind::TaskDependsOn,
        ]);
        assert_eq!(
            actual
                .edges
                .iter()
                .map(|edge| edge.kind.clone())
                .collect::<BTreeSet<_>>(),
            expected
        );
        assert_eq!(
            actual
                .edges
                .iter()
                .all(|edge| edge.provenance.is_complete()),
            true
        );
        Ok(())
    }

    #[test]
    fn knowledge_graph_promotes_manifest_edges_without_silent_drops() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;

        let actual = KnowledgeGraph::from_manifest(&manifest)?;
        let expected = manifest.edges.len();
        assert_eq!(actual.edges.len(), expected);
        assert_eq!(
            actual
                .nodes
                .iter()
                .all(|node| node.provenance().is_complete()),
            true
        );
        Ok(())
    }

    #[test]
    fn knowledge_graph_rejects_missing_typed_edge_endpoint() {
        let nodes = vec![KnowledgeGraphNode::Task(TaskGraphNode {
            id: KnowledgeGraphNodeId::Task("task:only".to_string()),
            title: "Only task".to_string(),
            status: "open".to_string(),
            provenance: provenance("task", None, None, "test", "task"),
        })];
        let edges = vec![KnowledgeGraphEdge {
            from: KnowledgeGraphNodeId::Task("task:only".to_string()),
            to: KnowledgeGraphNodeId::File("src/lib.rs".to_string()),
            kind: GraphEdgeKind::TaskDependsOn,
            confidence: 1.0,
            confidence_kind: EdgeConfidence::HeuristicHigh,
            provenance: provenance("task", None, None, "test", "missing"),
        }];

        let actual = KnowledgeGraph::new(nodes, edges).is_err();
        let expected = true;
        assert_eq!(actual, expected);
    }

    #[test]
    fn knowledge_graph_rejects_duplicate_node_ids() {
        let nodes = vec![
            KnowledgeGraphNode::Task(TaskGraphNode {
                id: KnowledgeGraphNodeId::Task("task:duplicate".to_string()),
                title: "First task".to_string(),
                status: "open".to_string(),
                provenance: provenance("task", None, None, "test", "first"),
            }),
            KnowledgeGraphNode::Task(TaskGraphNode {
                id: KnowledgeGraphNodeId::Task("task:duplicate".to_string()),
                title: "Second task".to_string(),
                status: "open".to_string(),
                provenance: provenance("task", None, None, "test", "second"),
            }),
        ];

        let actual = KnowledgeGraph::new(nodes, Vec::new()).is_err();
        let expected = true;
        assert_eq!(actual, expected);
    }

    #[test]
    fn knowledge_graph_rejects_invalid_confidence() {
        let nodes = vec![
            KnowledgeGraphNode::Task(TaskGraphNode {
                id: KnowledgeGraphNodeId::Task("task:source".to_string()),
                title: "Source task".to_string(),
                status: "open".to_string(),
                provenance: provenance("task", None, None, "test", "source"),
            }),
            KnowledgeGraphNode::Decision(DecisionGraphNode {
                id: KnowledgeGraphNodeId::Decision("decision:target".to_string()),
                title: "Target decision".to_string(),
                outcome: "target".to_string(),
                provenance: provenance("decision", None, None, "test", "target"),
            }),
        ];
        let edges = vec![KnowledgeGraphEdge {
            from: KnowledgeGraphNodeId::Task("task:source".to_string()),
            to: KnowledgeGraphNodeId::Decision("decision:target".to_string()),
            kind: GraphEdgeKind::TaskDependsOn,
            confidence: f32::NAN,
            confidence_kind: EdgeConfidence::HeuristicHigh,
            provenance: provenance("task", None, None, "test", "nan"),
        }];

        let actual = KnowledgeGraph::new(nodes, edges).is_err();
        let expected = true;
        assert_eq!(actual, expected);
    }

    #[test]
    fn knowledge_graph_sorts_edges_deterministically_for_same_endpoints_and_kind() -> Result<()> {
        let nodes = vec![
            KnowledgeGraphNode::Task(TaskGraphNode {
                id: KnowledgeGraphNodeId::Task("task:source".to_string()),
                title: "Source task".to_string(),
                status: "open".to_string(),
                provenance: provenance("task", None, None, "test", "source"),
            }),
            KnowledgeGraphNode::Decision(DecisionGraphNode {
                id: KnowledgeGraphNodeId::Decision("decision:target".to_string()),
                title: "Target decision".to_string(),
                outcome: "target".to_string(),
                provenance: provenance("decision", None, None, "test", "target"),
            }),
        ];
        let edges = vec![
            KnowledgeGraphEdge {
                from: KnowledgeGraphNodeId::Task("task:source".to_string()),
                to: KnowledgeGraphNodeId::Decision("decision:target".to_string()),
                kind: GraphEdgeKind::TaskDependsOn,
                confidence: 0.7,
                confidence_kind: EdgeConfidence::HeuristicLow,
                provenance: provenance("task", None, None, "test", "b"),
            },
            KnowledgeGraphEdge {
                from: KnowledgeGraphNodeId::Task("task:source".to_string()),
                to: KnowledgeGraphNodeId::Decision("decision:target".to_string()),
                kind: GraphEdgeKind::TaskDependsOn,
                confidence: 0.9,
                confidence_kind: EdgeConfidence::HeuristicHigh,
                provenance: provenance("task", None, None, "test", "a"),
            },
        ];

        let actual = KnowledgeGraph::new(nodes, edges)?;
        let expected = vec![EdgeConfidence::HeuristicHigh, EdgeConfidence::HeuristicLow];
        assert_eq!(
            actual
                .edges
                .iter()
                .map(|edge| edge.confidence_kind.clone())
                .collect::<Vec<_>>(),
            expected
        );
        Ok(())
    }

    #[test]
    fn context_pack_constructs_from_retrieval_shards_and_direct_evidence_with_freshness()
    -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let results = retrieve(
            &manifest,
            &RetrievalQuery {
                text: Some("Root".to_string()),
                path: None,
                path_prefix: None,
                symbol: None,
                limit: 2,
                include_graph_expansion: false,
            },
        );
        let shard = manifest
            .shards
            .iter()
            .find(|shard| shard.path == "src/model.rs")
            .expect("fixture should include model shard")
            .clone();
        let selection = ContextPackSelection {
            retrieval_results: results,
            shards: vec![shard],
            evidence: vec![ContextPackEvidence {
                id: "evidence:decision".to_string(),
                path: "src/lib.rs".to_string(),
                symbol: Some("Root".to_string()),
                source: ContextPackEvidenceSource::DirectEvidence,
                freshness: EvidenceFreshness::Fresh,
                provenance: provenance("src/lib.rs", Some(1), Some(1), "test", "direct"),
                score: 10.0,
            }],
            freshness: FreshnessState {
                changed: vec!["src/model.rs".to_string()],
                deleted: Vec::new(),
                added: Vec::new(),
                unchanged: vec!["src/lib.rs".to_string()],
                fresh: false,
            },
            stale_policy: StaleEvidencePolicy::Mark,
        };

        let actual = ContextPack::from_selection(&manifest, selection)?;
        let expected = true;
        assert_eq!(
            actual
                .evidence
                .iter()
                .any(|evidence| evidence.path == "src/model.rs"
                    && evidence.freshness == EvidenceFreshness::Changed),
            expected
        );
        assert_eq!(
            actual.provenance.iter().all(Provenance::is_complete),
            expected
        );
        assert_eq!(actual.manifest_hash, manifest.manifest_hash);
        Ok(())
    }

    #[test]
    fn context_pack_rejects_stale_evidence_when_policy_requires_current_evidence() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let selection = ContextPackSelection {
            retrieval_results: Vec::new(),
            shards: Vec::new(),
            evidence: vec![ContextPackEvidence {
                id: "evidence:stale".to_string(),
                path: "src/lib.rs".to_string(),
                symbol: None,
                source: ContextPackEvidenceSource::DirectEvidence,
                freshness: EvidenceFreshness::Fresh,
                provenance: provenance("src/lib.rs", Some(1), Some(1), "test", "stale"),
                score: 1.0,
            }],
            freshness: FreshnessState {
                changed: vec!["src/lib.rs".to_string()],
                deleted: Vec::new(),
                added: Vec::new(),
                unchanged: Vec::new(),
                fresh: false,
            },
            stale_policy: StaleEvidencePolicy::Reject,
        };

        let actual = ContextPack::from_selection(&manifest, selection).is_err();
        let expected = true;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn context_pack_rejects_incomplete_provenance() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let selection = ContextPackSelection {
            retrieval_results: Vec::new(),
            shards: Vec::new(),
            evidence: vec![ContextPackEvidence {
                id: "evidence:incomplete".to_string(),
                path: "src/lib.rs".to_string(),
                symbol: None,
                source: ContextPackEvidenceSource::DirectEvidence,
                freshness: EvidenceFreshness::Fresh,
                provenance: Provenance {
                    path: "src/lib.rs".to_string(),
                    start_line: Some(1),
                    end_line: Some(1),
                    source: String::new(),
                    fingerprint: fingerprint("incomplete"),
                },
                score: 1.0,
            }],
            freshness: FreshnessState::default(),
            stale_policy: StaleEvidencePolicy::Mark,
        };

        let actual = ContextPack::from_selection(&manifest, selection).is_err();
        let expected = true;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn context_pack_rejects_non_finite_scores() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let selection = ContextPackSelection {
            retrieval_results: Vec::new(),
            shards: Vec::new(),
            evidence: vec![ContextPackEvidence {
                id: "evidence:nan".to_string(),
                path: "src/lib.rs".to_string(),
                symbol: None,
                source: ContextPackEvidenceSource::DirectEvidence,
                freshness: EvidenceFreshness::Fresh,
                provenance: provenance("src/lib.rs", Some(1), Some(1), "test", "nan"),
                score: f32::NAN,
            }],
            freshness: FreshnessState::default(),
            stale_policy: StaleEvidencePolicy::Mark,
        };

        let actual = ContextPack::from_selection(&manifest, selection).is_err();
        let expected = true;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn context_pack_serializes_deterministically_with_stable_ordering() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let left = context_pack_ordering_fixture(&manifest, false)?;
        let right = context_pack_ordering_fixture(&manifest, true)?;

        let actual = left.to_stable_json()?;
        let expected = right.to_stable_json()?;
        assert_eq!(actual, expected);
        assert_eq!(
            left.evidence
                .iter()
                .map(|evidence| evidence.id.clone())
                .collect::<Vec<_>>(),
            vec!["a".to_string(), "b".to_string()]
        );
        Ok(())
    }

    fn context_pack_ordering_fixture(
        manifest: &ProjectManifest,
        reversed: bool,
    ) -> Result<ContextPack> {
        let mut evidence = vec![
            ContextPackEvidence {
                id: "b".to_string(),
                path: "src/model.rs".to_string(),
                symbol: None,
                source: ContextPackEvidenceSource::DirectEvidence,
                freshness: EvidenceFreshness::Fresh,
                provenance: provenance("src/model.rs", Some(1), Some(1), "test", "b"),
                score: 1.0,
            },
            ContextPackEvidence {
                id: "a".to_string(),
                path: "src/lib.rs".to_string(),
                symbol: None,
                source: ContextPackEvidenceSource::DirectEvidence,
                freshness: EvidenceFreshness::Fresh,
                provenance: provenance("src/lib.rs", Some(1), Some(1), "test", "a"),
                score: 1.0,
            },
        ];
        if reversed {
            evidence.reverse();
        }
        ContextPack::from_selection(
            manifest,
            ContextPackSelection {
                retrieval_results: Vec::new(),
                shards: Vec::new(),
                evidence,
                freshness: FreshnessState::default(),
                stale_policy: StaleEvidencePolicy::Mark,
            },
        )
    }
}
