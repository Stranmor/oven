//! Typed adapters for project-model context rendering and pack evidence.

use crate::render::ProjectModelContextSource;
use crate::types::{
    ContextPack, ContextPackEvidence, EvidenceFreshness, ProjectManifest, ShardManifest,
    SourceFile, SymbolNode,
};

/// Typed render root metadata for a project-model context payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectModelContextRenderRoot {
    /// Display path for the workspace root.
    pub workspace_root: String,
    /// Display path for the manifest backing the rendered context.
    pub manifest_path: String,
    /// Freshness label for the manifest/root.
    pub freshness: String,
    /// Provenance label for the render operation.
    pub provenance: String,
}

impl ProjectModelContextRenderRoot {
    /// Creates render-root metadata for project-model context rendering.
    ///
    /// # Arguments
    ///
    /// * `workspace_root` - Display path for the workspace root.
    /// * `manifest_path` - Display path for the manifest.
    /// * `freshness` - Freshness label.
    /// * `provenance` - Provenance label.
    pub fn new(
        workspace_root: impl Into<String>,
        manifest_path: impl Into<String>,
        freshness: impl Into<String>,
        provenance: impl Into<String>,
    ) -> Self {
        Self {
            workspace_root: workspace_root.into(),
            manifest_path: manifest_path.into(),
            freshness: freshness.into(),
            provenance: provenance.into(),
        }
    }
}

/// Typed source-node-like input owned by project-model adapters.
#[derive(Clone, Debug, PartialEq)]
pub enum ProjectModelSourceNode {
    /// Retrieved chunk or symbol-like source with line-bounded content.
    FileChunk {
        /// Relative source file path.
        path: String,
        /// One-based inclusive start line.
        start_line: u32,
        /// One-based inclusive end line.
        end_line: u32,
        /// Stable node identifier.
        node_id: String,
        /// Optional relevance score.
        score: Option<f32>,
        /// Source content.
        content: String,
    },
    /// Whole-file source that should be represented as metadata-only by default.
    File {
        /// Relative source file path.
        path: String,
        /// Stable node identifier.
        node_id: String,
        /// Optional relevance score.
        score: Option<f32>,
        /// Source content hash.
        content_hash: String,
        /// Optional full content.
        content: Option<String>,
    },
    /// File reference with no inline content.
    FileRef {
        /// Relative source file path.
        path: String,
        /// Stable node identifier.
        node_id: String,
        /// Optional relevance score.
        score: Option<f32>,
        /// Source content hash.
        content_hash: String,
    },
    /// Note evidence source.
    Note {
        /// Stable node identifier.
        node_id: String,
        /// Optional relevance score.
        score: Option<f32>,
        /// Note content.
        content: String,
    },
    /// Task evidence source.
    Task {
        /// Stable node identifier.
        node_id: String,
        /// Optional relevance score.
        score: Option<f32>,
        /// Task content.
        content: String,
    },
}

impl ProjectModelSourceNode {
    /// Converts a source-node-like value into a render source.
    pub fn into_render_source(self) -> ProjectModelContextSource {
        match self {
            Self::FileChunk { path, start_line, end_line, node_id, score, content } => {
                ProjectModelContextSource::new(
                    path,
                    "manifest_snapshot",
                    "local_project_model_manifest",
                    node_id,
                )
                .line_range(start_line, end_line)
                .score(score)
                .content(content)
            }
            Self::File { path, node_id, score, content_hash, content } => {
                let mut source = ProjectModelContextSource::new(
                    path,
                    "manifest_snapshot",
                    "local_project_model_manifest",
                    node_id,
                )
                .score(score)
                .content_hash(content_hash)
                .metadata_only("whole_file_metadata_only");
                if let Some(content) = content {
                    source = source.content(content);
                }
                source
            }
            Self::FileRef { path, node_id, score, content_hash } => ProjectModelContextSource::new(
                path,
                "manifest_snapshot",
                "local_project_model_manifest",
                node_id,
            )
            .score(score)
            .content_hash(content_hash)
            .metadata_only("file_reference_metadata_only"),
            Self::Note { node_id, score, content } => ProjectModelContextSource::new(
                "note",
                "manifest_snapshot",
                "local_project_model_manifest",
                node_id,
            )
            .score(score)
            .content(content),
            Self::Task { node_id, score, content } => ProjectModelContextSource::new(
                "task",
                "manifest_snapshot",
                "local_project_model_manifest",
                node_id,
            )
            .score(score)
            .content(content),
        }
    }
}

/// Converts source-node-like values into render sources.
///
/// # Arguments
///
/// * `nodes` - Typed source nodes supplied by an adapter.
pub fn render_sources_from_nodes(
    nodes: impl IntoIterator<Item = ProjectModelSourceNode>,
) -> Vec<ProjectModelContextSource> {
    nodes
        .into_iter()
        .map(ProjectModelSourceNode::into_render_source)
        .collect()
}

/// Converts a context pack into render sources using manifest metadata when
/// available.
///
/// # Arguments
///
/// * `manifest` - Project manifest that owns source metadata.
/// * `pack` - Context pack evidence to surface for rendering.
pub fn render_sources_from_context_pack(
    manifest: &ProjectManifest,
    pack: &ContextPack,
) -> Vec<ProjectModelContextSource> {
    pack.evidence
        .iter()
        .map(|evidence| render_source_from_evidence(manifest, evidence))
        .collect()
}

/// Converts a context-pack evidence item into a render source.
///
/// # Arguments
///
/// * `manifest` - Project manifest that owns source metadata.
/// * `evidence` - Context-pack evidence item.
pub fn render_source_from_evidence(
    manifest: &ProjectManifest,
    evidence: &ContextPackEvidence,
) -> ProjectModelContextSource {
    let mut source = ProjectModelContextSource::new(
        evidence.path.clone(),
        evidence_freshness_label(&evidence.freshness),
        evidence.provenance.source.clone(),
        evidence.id.clone(),
    )
    .score(Some(evidence.score));
    if let Some((start_line, end_line)) = evidence_line_range(manifest, &evidence.id) {
        source = source.line_range(start_line, end_line);
    }
    if let Some(content_hash) = evidence_content_hash(manifest, &evidence.id, &evidence.path) {
        source = source.content_hash(content_hash);
    }
    source
}

/// Returns the one-based inclusive line range for evidence owned by a manifest.
///
/// # Arguments
///
/// * `manifest` - Manifest searched for matching evidence.
/// * `evidence_id` - Retrieval, symbol, shard, or file identifier.
pub fn evidence_line_range(manifest: &ProjectManifest, evidence_id: &str) -> Option<(u32, u32)> {
    manifest
        .symbols
        .iter()
        .find(|symbol| symbol.id == evidence_id)
        .map(symbol_line_range)
        .or_else(|| {
            manifest
                .shards
                .iter()
                .find(|shard| shard.id == evidence_id)
                .map(shard_line_range)
        })
        .or_else(|| {
            manifest
                .files
                .iter()
                .find(|file| file.path == evidence_id)
                .map(file_line_range)
        })
}

fn evidence_content_hash(
    manifest: &ProjectManifest,
    evidence_id: &str,
    evidence_path: &str,
) -> Option<String> {
    manifest
        .shards
        .iter()
        .find(|shard| shard.id == evidence_id)
        .map(|shard| shard.content_hash.clone())
        .or_else(|| {
            manifest
                .files
                .iter()
                .find(|file| file.path == evidence_id || file.path == evidence_path)
                .map(|file| file.content_hash.clone())
        })
}

fn symbol_line_range(symbol: &SymbolNode) -> (u32, u32) {
    (symbol.start_line, symbol.end_line)
}

fn shard_line_range(shard: &ShardManifest) -> (u32, u32) {
    (shard.start_line, shard.end_line)
}

fn file_line_range(file: &SourceFile) -> (u32, u32) {
    (1, file.lines)
}

fn evidence_freshness_label(freshness: &EvidenceFreshness) -> &'static str {
    match freshness {
        EvidenceFreshness::Fresh => "fresh",
        EvidenceFreshness::Added => "added",
        EvidenceFreshness::Changed => "changed",
        EvidenceFreshness::Deleted => "deleted",
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::indexer::tests::fixture_project;
    use crate::{ContextPackSelection, FreshnessState, ProjectIndexer, RetrievalQuery, retrieve};

    #[test]
    fn converts_source_node_to_metadata_only_file_render_source() {
        let setup = ProjectModelSourceNode::File {
            path: "src/lib.rs".to_string(),
            node_id: "src/lib.rs".to_string(),
            score: Some(1.0),
            content_hash: "hash".to_string(),
            content: Some("full file".to_string()),
        };
        let actual = setup.into_render_source();
        let expected = (
            "src/lib.rs".to_string(),
            Some(1.0),
            Some("hash".to_string()),
            Some("whole_file_metadata_only".to_string()),
            Some("full file".to_string()),
        );
        assert_eq!(
            (
                actual.path,
                actual.score,
                actual.content_hash,
                actual.metadata_only_reason,
                actual.content,
            ),
            expected,
        );
    }

    #[test]
    fn context_pack_adapter_preserves_line_ranges_and_hashes() -> anyhow::Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let root_symbol = manifest
            .symbols
            .iter()
            .find(|symbol| symbol.name == "Root")
            .expect("fixture should include Root symbol");
        let query = RetrievalQuery {
            text: Some("Root".to_string()),
            path: None,
            path_prefix: None,
            symbol: Some("Root".to_string()),
            limit: 1,
            include_graph_expansion: false,
        };
        let pack = ContextPack::from_selection(
            &manifest,
            ContextPackSelection {
                retrieval_results: retrieve(&manifest, &query),
                shards: Vec::new(),
                evidence: Vec::new(),
                freshness: FreshnessState {
                    unchanged: manifest
                        .files
                        .iter()
                        .map(|file| file.path.clone())
                        .collect(),
                    fresh: true,
                    ..FreshnessState::default()
                },
                stale_policy: crate::StaleEvidencePolicy::Mark,
            },
        )?;
        let actual = render_sources_from_context_pack(&manifest, &pack);
        let expected = Some((
            root_symbol.path.clone(),
            Some(root_symbol.start_line),
            Some(root_symbol.end_line),
            "fresh".to_string(),
        ));
        assert_eq!(
            actual.first().map(|source| (
                source.path.clone(),
                source.start_line,
                source.end_line,
                source.freshness.clone(),
            )),
            expected,
        );
        Ok(())
    }
}
