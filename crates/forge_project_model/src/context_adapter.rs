//! Typed adapters for project-model context rendering and pack evidence.

use std::path::{Component, Path};

use crate::evidence_replay::{EvidenceLedgerReplayReport, EvidenceReplayIssueCode};
use crate::render::ProjectModelContextSource;
use crate::types::{
    CargoFeatureMetadata, CargoPackageDependency, CargoPackageMetadata, CargoTargetMetadata,
    CargoWorkspaceMetadata, ContextPack, ContextPackEvidence, EvidenceFreshness, ProjectArtifact,
    ProjectManifest, ShardManifest, SourceFile, SymbolNode,
};

const EVIDENCE_REPLAY_METADATA_ONLY_REASON: &str = "evidence_replay_reference_only";

/// Typed refusal returned by evidence-replay preview rendering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvidenceReplayPreviewError {
    /// Replay report was produced for a different manifest hash than the current manifest.
    ManifestHashMismatch {
        /// Current manifest hash supplied by the manifest.
        manifest_hash: String,
        /// Manifest hash recorded by the replay report.
        report_manifest_hash: String,
    },
}

impl std::fmt::Display for EvidenceReplayPreviewError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ManifestHashMismatch { manifest_hash, report_manifest_hash } => write!(
                formatter,
                "evidence replay manifest hash mismatch: manifest={manifest_hash}, report={report_manifest_hash}"
            ),
        }
    }
}

impl std::error::Error for EvidenceReplayPreviewError {}

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

/// Converts a narrowed evidence-ledger replay report into metadata-only render sources.
///
/// This adapter is intentionally read-only and never exposes source content or
/// tool payloads. It only reflects selected fresh/additional references that are
/// bound to the current manifest and safe relative paths.
///
/// # Arguments
///
/// * `manifest` - Current project manifest used for hash and path validation.
/// * `report` - Read-only evidence-ledger replay report to preview.
///
/// # Errors
///
/// Returns a typed refusal when the replay report does not match the current
/// manifest hash.
pub fn render_sources_from_evidence_replay(
    manifest: &ProjectManifest,
    report: &EvidenceLedgerReplayReport,
) -> Result<Vec<ProjectModelContextSource>, EvidenceReplayPreviewError> {
    if manifest.manifest_hash != report.manifest_hash {
        return Err(EvidenceReplayPreviewError::ManifestHashMismatch {
            manifest_hash: manifest.manifest_hash.clone(),
            report_manifest_hash: report.manifest_hash.clone(),
        });
    }

    Ok(report
        .selected
        .iter()
        .filter(|reference| evidence_replay_reference_is_allowed(&reference.freshness))
        .filter_map(|reference| {
            let target = resolve_manifest_evidence_target(manifest, &reference.evidence_id)?;
            if !path_is_safe_relative_label(&reference.evidence_path)
                || reference.evidence_path != target.path
            {
                return None;
            }
            let mut source = ProjectModelContextSource::new(
                target.path,
                evidence_freshness_label(&reference.freshness),
                redaction_safe_provenance_source_label(&reference.provenance.source),
                reference.evidence_id.clone(),
            )
            .score(Some(reference.score))
            .content_hash(target.content_hash)
            .metadata_only(EVIDENCE_REPLAY_METADATA_ONLY_REASON);
            if let Some((start_line, end_line)) = target.line_range {
                source = source.line_range(start_line, end_line);
            }
            Some(source)
        })
        .collect())
}

/// Returns a redaction-safe label for replay diagnostic paths.
///
/// # Arguments
///
/// * `path` - Candidate path or storage label to sanitize.
pub fn redaction_safe_replay_path_label(path: &str) -> String {
    if path == "tool_episodes.jsonl" {
        return path.to_string();
    }
    if path
        .strip_prefix("context_packs/")
        .and_then(|value| value.strip_suffix(".json"))
        .is_some_and(|id| id.len() == 64 && id.chars().all(|ch| ch.is_ascii_hexdigit()))
    {
        return path.to_string();
    }
    if path_is_safe_relative_label(path) {
        return path.to_string();
    }
    "redacted_path".to_string()
}

/// Returns a redaction-safe issue-path label for replay diagnostics.
///
/// # Arguments
///
/// * `code` - Issue code that determines whether tool-episode provenance must be hidden.
/// * `path` - Optional path or storage label to sanitize.
pub fn redaction_safe_issue_path_label(
    code: &EvidenceReplayIssueCode,
    path: Option<&str>,
) -> Option<String> {
    if matches!(
        code,
        EvidenceReplayIssueCode::CorruptEpisode
            | EvidenceReplayIssueCode::Duplicate
            | EvidenceReplayIssueCode::UnlinkedEpisode
            | EvidenceReplayIssueCode::DanglingEpisodeLink
    ) {
        return Some("tool_episode_provenance".to_string());
    }
    path.map(redaction_safe_replay_path_label)
}

/// Returns a stable redaction-safe provenance source label.
///
/// # Arguments
///
/// * `source` - Raw provenance source label to classify.
pub fn redaction_safe_provenance_source_label(source: &str) -> String {
    match source {
        "WorkspaceService::query_workspace" => "workspace_service_query".to_string(),
        "build-dependencies" => "cargo_build_dependencies".to_string(),
        "dependencies" => "cargo_dependencies".to_string(),
        "dev-dependencies" => "cargo_dev_dependencies".to_string(),
        "extern crate" => "rust_extern_crate".to_string(),
        "file-tree" => "file_tree".to_string(),
        "indexer" => "project_indexer".to_string(),
        "mod" => "rust_module".to_string(),
        "rust-ast" => "rust_ast".to_string(),
        source if source.starts_with("cargo_metadata:") => "cargo_metadata".to_string(),
        source if source.starts_with("call_graph:") => "rust_call_graph".to_string(),
        source if source.starts_with("dependency:") => "rust_dependency".to_string(),
        source if source.starts_with("external_fact:") => "external_fact".to_string(),
        _ => "redacted_provenance_source".to_string(),
    }
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
    let target = resolve_manifest_evidence_target(manifest, &evidence.id);
    let path = target
        .as_ref()
        .map(|target| target.path.clone())
        .unwrap_or_else(|| evidence.path.clone());
    let mut source = ProjectModelContextSource::new(
        path,
        evidence_freshness_label(&evidence.freshness),
        evidence.provenance.source.clone(),
        evidence.id.clone(),
    )
    .score(Some(evidence.score));
    if let Some(target) = target {
        if let Some((start_line, end_line)) = target.line_range {
            source = source.line_range(start_line, end_line);
        }
        source = source.content_hash(target.content_hash);
    } else if let Some(content_hash) = evidence_content_hash(manifest, &evidence.id, &evidence.path)
    {
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
    resolve_manifest_evidence_target(manifest, evidence_id).and_then(|target| target.line_range)
}

fn evidence_replay_reference_is_allowed(freshness: &EvidenceFreshness) -> bool {
    matches!(
        freshness,
        EvidenceFreshness::Fresh | EvidenceFreshness::Added
    )
}

/// Resolved manifest-owned evidence target for readback and rendering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestEvidenceTarget {
    /// Manifest-relative owner path for this evidence.
    pub path: String,
    /// One-based inclusive line range for source readback.
    pub line_range: Option<(u32, u32)>,
    /// Content hash of the owning file or shard.
    pub content_hash: String,
}

/// Returns true when an evidence identifier uses the reserved Cargo namespace.
///
/// # Arguments
///
/// * `evidence_id` - Candidate evidence identifier.
pub fn is_reserved_cargo_evidence_id(evidence_id: &str) -> bool {
    evidence_id.starts_with("cargo:")
}

/// Returns true when an evidence identifier uses the reserved artifact namespace.
///
/// # Arguments
///
/// * `evidence_id` - Candidate evidence identifier.
pub fn is_reserved_artifact_evidence_id(evidence_id: &str) -> bool {
    evidence_id.starts_with("artifact:")
}

/// Resolves a manifest-owned project artifact by stable identifier.
///
/// # Arguments
///
/// * `manifest` - Manifest owning the artifact inventory.
/// * `evidence_id` - Candidate artifact identifier.
pub fn resolve_project_artifact_evidence<'a>(
    manifest: &'a ProjectManifest,
    evidence_id: &str,
) -> Option<&'a ProjectArtifact> {
    if !is_reserved_artifact_evidence_id(evidence_id) {
        return None;
    }
    manifest
        .artifacts
        .iter()
        .find(|artifact| artifact.id == evidence_id)
}

/// Builds a stable opaque Cargo evidence identifier for a typed manifest item.
///
/// # Arguments
///
/// * `kind` - Stable Cargo metadata evidence kind.
/// * `fields` - Length-delimited stable fields describing the typed DTO item.
pub fn cargo_metadata_evidence_id(kind: &str, fields: &[&str]) -> String {
    let mut seed = format!("cargo-evidence-v1\0kind:{}:{}", kind.len(), kind);
    for field in fields {
        seed.push('\0');
        seed.push_str(&field.len().to_string());
        seed.push(':');
        seed.push_str(field);
    }
    format!("cargo:v1:{kind}:{}", crate::util::fingerprint(&seed))
}

/// Resolves manifest-owned evidence to the path, range, and content hash that
/// should be used for service readback and metadata rendering.
///
/// # Arguments
///
/// * `manifest` - Manifest owning the evidence namespace.
/// * `evidence_id` - File, symbol, shard, or Cargo metadata evidence id.
pub fn resolve_manifest_evidence_target(
    manifest: &ProjectManifest,
    evidence_id: &str,
) -> Option<ManifestEvidenceTarget> {
    if let Some(symbol) = manifest
        .symbols
        .iter()
        .find(|symbol| symbol.id == evidence_id)
    {
        let file = manifest
            .files
            .iter()
            .find(|file| file.path == symbol.path)?;
        return Some(ManifestEvidenceTarget {
            path: symbol.path.clone(),
            line_range: Some(symbol_line_range(symbol)),
            content_hash: file.content_hash.clone(),
        });
    }
    if let Some(shard) = manifest.shards.iter().find(|shard| shard.id == evidence_id) {
        return Some(ManifestEvidenceTarget {
            path: shard.path.clone(),
            line_range: Some(shard_line_range(shard)),
            content_hash: shard.content_hash.clone(),
        });
    }
    if let Some(file) = manifest.files.iter().find(|file| file.path == evidence_id) {
        return Some(ManifestEvidenceTarget {
            path: file.path.clone(),
            line_range: Some(file_line_range(file)),
            content_hash: file.content_hash.clone(),
        });
    }
    if let Some(artifact) = resolve_project_artifact_evidence(manifest, evidence_id) {
        let file = manifest
            .files
            .iter()
            .find(|file| file.path == artifact.path)?;
        return Some(ManifestEvidenceTarget {
            path: artifact.path.clone(),
            line_range: Some(file_line_range(file)),
            content_hash: file.content_hash.clone(),
        });
    }
    resolve_cargo_metadata_evidence_target(manifest, evidence_id)
}

fn resolve_cargo_metadata_evidence_target(
    manifest: &ProjectManifest,
    evidence_id: &str,
) -> Option<ManifestEvidenceTarget> {
    if !is_reserved_cargo_evidence_id(evidence_id) {
        return None;
    }
    if let Some(workspace) = &manifest.cargo_workspace
        && cargo_workspace_evidence_id(workspace) == evidence_id
    {
        return cargo_manifest_target(manifest, &workspace.manifest_path);
    }
    for package in &manifest.cargo_packages {
        if cargo_package_evidence_id(package) == evidence_id {
            return cargo_manifest_target(manifest, &package.manifest_path);
        }
        for target in &package.targets {
            if cargo_target_evidence_id(package, target) == evidence_id {
                return cargo_manifest_target(manifest, &package.manifest_path);
            }
        }
        for feature in &package.features {
            if cargo_feature_evidence_id(package, feature) == evidence_id {
                return cargo_manifest_target(manifest, &package.manifest_path);
            }
        }
    }
    for dependency in &manifest.cargo_package_dependencies {
        if cargo_dependency_evidence_id(dependency) == evidence_id {
            return cargo_manifest_target(manifest, &dependency.manifest_path);
        }
    }
    None
}

fn cargo_manifest_target(
    manifest: &ProjectManifest,
    manifest_path: &str,
) -> Option<ManifestEvidenceTarget> {
    manifest
        .files
        .iter()
        .find(|file| file.path == manifest_path)
        .map(|file| ManifestEvidenceTarget {
            path: file.path.clone(),
            line_range: Some(file_line_range(file)),
            content_hash: file.content_hash.clone(),
        })
}

/// Builds the stable Cargo workspace evidence id.
///
/// # Arguments
///
/// * `workspace` - Manifest-owned Cargo workspace metadata.
pub fn cargo_workspace_evidence_id(workspace: &CargoWorkspaceMetadata) -> String {
    let members = workspace.members.join("\0");
    let package_manifest_paths = workspace.package_manifest_paths.join("\0");
    cargo_metadata_evidence_id(
        "workspace",
        &[
            &workspace.manifest_path,
            &workspace.root_path,
            &members,
            &package_manifest_paths,
        ],
    )
}

/// Builds the stable Cargo package evidence id.
///
/// # Arguments
///
/// * `package` - Manifest-owned Cargo package metadata.
pub fn cargo_package_evidence_id(package: &CargoPackageMetadata) -> String {
    cargo_metadata_evidence_id(
        "package",
        &[
            &package.manifest_path,
            &package.package_root,
            &package.name,
            package.version.as_deref().unwrap_or_default(),
            package.edition.as_deref().unwrap_or_default(),
        ],
    )
}

/// Builds the stable Cargo target evidence id.
///
/// # Arguments
///
/// * `package` - Package owning the target.
/// * `target` - Manifest-owned Cargo target metadata.
pub fn cargo_target_evidence_id(
    package: &CargoPackageMetadata,
    target: &CargoTargetMetadata,
) -> String {
    cargo_metadata_evidence_id(
        "target",
        &[
            &package.manifest_path,
            &package.name,
            &target.name,
            &format!("{:?}", target.kind),
            &target.path,
            &format!("{:?}", target.declaration),
        ],
    )
}

/// Builds the stable Cargo feature evidence id.
///
/// # Arguments
///
/// * `package` - Package owning the feature.
/// * `feature` - Manifest-owned Cargo feature metadata.
pub fn cargo_feature_evidence_id(
    package: &CargoPackageMetadata,
    feature: &CargoFeatureMetadata,
) -> String {
    let members = feature.members.join("\0");
    cargo_metadata_evidence_id(
        "feature",
        &[
            &package.manifest_path,
            &package.name,
            &feature.name,
            &members,
        ],
    )
}

/// Builds the stable Cargo dependency evidence id.
///
/// # Arguments
///
/// * `dependency` - Manifest-owned Cargo dependency metadata.
pub fn cargo_dependency_evidence_id(dependency: &CargoPackageDependency) -> String {
    let features = dependency.features.join("\0");
    cargo_metadata_evidence_id(
        "dependency",
        &[
            &dependency.manifest_path,
            dependency.declaring_package.as_deref().unwrap_or_default(),
            &dependency.dependency_key,
            &dependency.package_name,
            &format!("{:?}", dependency.kind),
            dependency.target.as_deref().unwrap_or_default(),
            dependency.version.as_deref().unwrap_or_default(),
            dependency.path.as_deref().unwrap_or_default(),
            if dependency.optional {
                "optional"
            } else {
                "required"
            },
            &features,
            &format!("{:?}", dependency.declaration),
            dependency
                .linked_package_manifest_path
                .as_deref()
                .unwrap_or_default(),
        ],
    )
}

fn path_is_safe_relative_label(path: &str) -> bool {
    if path.is_empty() || Path::new(path).is_absolute() {
        return false;
    }
    path.chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '/' | '.' | '_' | '-'))
        && Path::new(path)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn evidence_content_hash(
    manifest: &ProjectManifest,
    evidence_id: &str,
    evidence_path: &str,
) -> Option<String> {
    resolve_manifest_evidence_target(manifest, evidence_id)
        .filter(|target| target.path == evidence_path)
        .map(|target| target.content_hash)
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
    (1, file.lines.max(1))
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
    use std::collections::BTreeMap;

    use pretty_assertions::assert_eq;

    use super::*;
    use crate::evidence_replay::{
        EvidenceLedgerReplayReport, EvidenceReplayBudget, EvidenceReplayBudgetReport,
        EvidenceReplayContentPolicy, EvidenceReplayFreshnessPolicy, EvidenceReplayReference,
        EvidenceReplayScoreKind, EvidenceReplayStalePolicyReport,
    };
    use crate::indexer::tests::fixture_project;
    use crate::{ContextPackSelection, FreshnessState, ProjectIndexer, RetrievalQuery, retrieve};

    fn fixture_replay_manifest() -> ProjectManifest {
        ProjectManifest {
            version: 1,
            root: "/workspace".into(),
            files: vec![SourceFile {
                path: "src/lib.rs".to_string(),
                language: crate::Language::Rust,
                bytes: 128,
                lines: 12,
                content_hash: "file-hash".to_string(),
                provenance: fixture_provenance("indexer"),
            }],
            symbols: vec![SymbolNode {
                id: "symbol:root".to_string(),
                name: "Root".to_string(),
                kind: crate::SymbolKind::Struct,
                path: "src/lib.rs".to_string(),
                parent: None,
                start_line: 3,
                end_line: 7,
                provenance: fixture_provenance("rust-ast"),
            }],
            shards: vec![ShardManifest {
                id: "shard:src/lib.rs:1-12".to_string(),
                path: "src/lib.rs".to_string(),
                start_line: 1,
                end_line: 12,
                content_hash: "shard-hash".to_string(),
                symbol_ids: vec!["symbol:root".to_string()],
                provenance: fixture_provenance("indexer"),
            }],
            manifest_hash: "manifest-hash".to_string(),
            ..ProjectManifest::default()
        }
    }

    fn fixture_provenance(source: &str) -> crate::Provenance {
        crate::Provenance {
            path: "src/lib.rs".to_string(),
            start_line: Some(1),
            end_line: Some(1),
            source: source.to_string(),
            fingerprint: "fingerprint".to_string(),
        }
    }

    fn fixture_replay_reference(
        evidence_id: impl Into<String>,
        evidence_path: impl Into<String>,
        freshness: EvidenceFreshness,
        provenance_source: impl Into<String>,
    ) -> EvidenceReplayReference {
        EvidenceReplayReference {
            artifact_id: "artifact".to_string(),
            artifact_path: "context_packs/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.json".to_string(),
            evidence_id: evidence_id.into(),
            evidence_path: evidence_path.into(),
            start_line: Some(3),
            end_line: Some(7),
            score_kind: EvidenceReplayScoreKind::RetrievalResult,
            score: 0.875,
            provenance: crate::Provenance {
                source: provenance_source.into(),
                ..fixture_provenance("indexer")
            },
            freshness,
            source_content_hash: "source-hash".to_string(),
            line_range_fingerprint: "range-fingerprint".to_string(),
            linked_episode_count: 1,
            link_issue_count: 0,
        }
    }

    fn fixture_replay_report(
        manifest_hash: impl Into<String>,
        selected: Vec<EvidenceReplayReference>,
    ) -> EvidenceLedgerReplayReport {
        let selected_count = selected.len();
        EvidenceLedgerReplayReport {
            manifest_hash: manifest_hash.into(),
            content_policy: EvidenceReplayContentPolicy::ReferenceOnly,
            stale_policy: EvidenceReplayStalePolicyReport {
                policy: EvidenceReplayFreshnessPolicy::ExcludeChangedAndDeleted,
                changed_excluded: 2,
                deleted_excluded: 1,
            },
            selected,
            issues: Vec::new(),
            budget: EvidenceReplayBudgetReport {
                original_candidate_count: selected_count,
                selected_count,
                excluded_count: 3,
                excluded_by_reason: BTreeMap::new(),
                truncated: false,
                budget: EvidenceReplayBudget::default(),
                stable_ordering: "fixture".to_string(),
            },
        }
    }

    #[test]
    fn evidence_replay_adapter_converts_fresh_reference_to_metadata_only_source() {
        let setup = fixture_replay_manifest();
        let report = fixture_replay_report(
            setup.manifest_hash.clone(),
            vec![fixture_replay_reference(
                "symbol:root",
                "src/lib.rs",
                EvidenceFreshness::Fresh,
                "WorkspaceService::query_workspace",
            )],
        );

        let actual = render_sources_from_evidence_replay(&setup, &report).unwrap();
        let actual = actual.first().expect("fixture should render one source");
        let expected = (
            "src/lib.rs".to_string(),
            None,
            Some("evidence_replay_reference_only".to_string()),
            "workspace_service_query".to_string(),
            Some(3),
            Some(7),
            Some(0.875),
            "fresh".to_string(),
            "symbol:root".to_string(),
            Some("file-hash".to_string()),
        );

        assert_eq!(
            (
                actual.path.clone(),
                actual.content.clone(),
                actual.metadata_only_reason.clone(),
                actual.provenance.clone(),
                actual.start_line,
                actual.end_line,
                actual.score,
                actual.freshness.clone(),
                actual.node_id.clone(),
                actual.content_hash.clone(),
            ),
            expected,
        );
    }

    #[test]
    fn evidence_replay_adapter_refuses_manifest_hash_mismatch_without_sources() {
        let setup = fixture_replay_manifest();
        let report = fixture_replay_report(
            "different-hash",
            vec![fixture_replay_reference(
                "symbol:root",
                "src/lib.rs",
                EvidenceFreshness::Fresh,
                "indexer",
            )],
        );

        let actual = render_sources_from_evidence_replay(&setup, &report);
        let expected = Err(EvidenceReplayPreviewError::ManifestHashMismatch {
            manifest_hash: "manifest-hash".to_string(),
            report_manifest_hash: "different-hash".to_string(),
        });

        assert_eq!(actual, expected);
    }

    #[test]
    fn evidence_replay_adapter_redacts_unknown_raw_provenance_source() {
        let setup = fixture_replay_manifest();
        let report = fixture_replay_report(
            setup.manifest_hash.clone(),
            vec![fixture_replay_reference(
                "symbol:root",
                "src/lib.rs",
                EvidenceFreshness::Fresh,
                "https://secret.example/raw/source",
            )],
        );

        let actual = render_sources_from_evidence_replay(&setup, &report).unwrap();
        let expected = Some("redacted_provenance_source".to_string());

        assert_eq!(
            actual.first().map(|source| source.provenance.clone()),
            expected
        );
    }

    #[test]
    fn evidence_replay_adapter_skips_absolute_traversal_unmanifested_and_stale_paths() {
        let setup = fixture_replay_manifest();
        let report = fixture_replay_report(
            setup.manifest_hash.clone(),
            vec![
                fixture_replay_reference(
                    "absolute",
                    "/tmp/secret.rs",
                    EvidenceFreshness::Fresh,
                    "indexer",
                ),
                fixture_replay_reference(
                    "traversal",
                    "../secret.rs",
                    EvidenceFreshness::Fresh,
                    "indexer",
                ),
                fixture_replay_reference(
                    "unmanifested",
                    "src/unknown.rs",
                    EvidenceFreshness::Fresh,
                    "indexer",
                ),
                fixture_replay_reference(
                    "changed",
                    "src/lib.rs",
                    EvidenceFreshness::Changed,
                    "indexer",
                ),
                fixture_replay_reference(
                    "deleted",
                    "src/lib.rs",
                    EvidenceFreshness::Deleted,
                    "indexer",
                ),
                fixture_replay_reference(
                    "symbol:root",
                    "src/lib.rs",
                    EvidenceFreshness::Added,
                    "indexer",
                ),
            ],
        );

        let actual = render_sources_from_evidence_replay(&setup, &report).unwrap();
        let expected = vec!["src/lib.rs".to_string()];

        assert_eq!(
            actual
                .into_iter()
                .map(|source| source.path)
                .collect::<Vec<_>>(),
            expected,
        );
    }

    #[test]
    fn evidence_replay_adapter_rejects_reference_when_evidence_id_path_mismatches_manifest_path() {
        let mut setup = fixture_replay_manifest();
        setup.files.push(SourceFile {
            path: "src/other.rs".to_string(),
            language: crate::Language::Rust,
            bytes: 64,
            lines: 4,
            content_hash: "other-file-hash".to_string(),
            provenance: fixture_provenance("indexer"),
        });
        let report = fixture_replay_report(
            setup.manifest_hash.clone(),
            vec![fixture_replay_reference(
                "symbol:root",
                "src/other.rs",
                EvidenceFreshness::Fresh,
                "indexer",
            )],
        );

        let actual = render_sources_from_evidence_replay(&setup, &report).unwrap();
        let expected = Vec::<ProjectModelContextSource>::new();

        assert_eq!(actual, expected);
    }

    #[test]
    fn evidence_replay_adapter_never_includes_source_content_or_tool_payload() {
        let setup = fixture_replay_manifest();
        let report = fixture_replay_report(
            setup.manifest_hash.clone(),
            vec![fixture_replay_reference(
                "shard:src/lib.rs:1-12",
                "src/lib.rs",
                EvidenceFreshness::Fresh,
                "indexer",
            )],
        );

        let actual = render_sources_from_evidence_replay(&setup, &report).unwrap();
        let expected = Some((None, Some("evidence_replay_reference_only".to_string())));

        assert_eq!(
            actual
                .first()
                .map(|source| (source.content.clone(), source.metadata_only_reason.clone())),
            expected,
        );
    }

    #[test]
    fn evidence_replay_adapter_is_pure_and_does_not_mutate_fixture_manifest_or_report() {
        let setup = fixture_replay_manifest();
        let report = fixture_replay_report(
            setup.manifest_hash.clone(),
            vec![fixture_replay_reference(
                "symbol:root",
                "src/lib.rs",
                EvidenceFreshness::Fresh,
                "indexer",
            )],
        );
        let expected = (setup.clone(), report.clone());

        let _actual = render_sources_from_evidence_replay(&setup, &report).unwrap();

        assert_eq!((setup, report), expected);
    }

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
