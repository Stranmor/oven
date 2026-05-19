//! Rust-native evaluation harness for retrieval, graph, freshness, and
//! provenance.

use std::collections::{BTreeMap, BTreeSet};

use crate::ProjectIndexer;
use crate::freshness::compare_freshness;
use crate::retrieval::retrieve;
use crate::types::{
    ContextPack, ContextPackArtifactEvalReport, ContextPackArtifactId, EdgeConfidence,
    EvidenceFreshness, EvidenceLedgerEvalIssue, EvidenceLedgerEvalIssueCode,
    EvidenceLedgerLinkageReport, FreshnessEvalReport, GraphCoverageReport, GraphEdge,
    GraphEdgeKind, KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode, KnowledgeGraphNodeId,
    ProjectManifest, Provenance, ProvenanceCompletenessReport, RetrievalEvalCase,
    RetrievalEvalReport, RetrievedEvidenceGraphNode, ToolEpisode, ToolEpisodeEvalReport,
    ToolEpisodeGraphNode,
};
use crate::util::fingerprint;

/// Evaluates retrieval precision@k, recall@k, and mean reciprocal rank.
///
/// # Arguments
///
/// * `manifest` - Manifest searched by the deterministic retrieval
///   implementation.
/// * `cases` - Evaluation cases with relevant result identifiers.
/// * `k` - Cutoff used for precision and recall.
pub fn evaluate_retrieval(
    manifest: &ProjectManifest,
    cases: &[RetrievalEvalCase],
    k: usize,
) -> RetrievalEvalReport {
    if cases.is_empty() || k == 0 {
        return RetrievalEvalReport::default();
    }
    let mut precision_sum = 0.0f32;
    let mut recall_sum = 0.0f32;
    let mut reciprocal_rank_sum = 0.0f32;
    for case in cases {
        let mut query = case.query.clone();
        query.limit = k;
        let results = retrieve(manifest, &query);
        let top_ids = results
            .iter()
            .take(k)
            .map(|result| result.id.clone())
            .collect::<Vec<_>>();
        let top_set = top_ids.iter().cloned().collect::<BTreeSet<_>>();
        let relevant_hits = top_set.intersection(&case.relevant_ids).count();
        precision_sum += relevant_hits as f32 / k as f32;
        if !case.relevant_ids.is_empty() {
            recall_sum += relevant_hits as f32 / case.relevant_ids.len() as f32;
        }
        reciprocal_rank_sum += top_ids
            .iter()
            .position(|id| case.relevant_ids.contains(id))
            .map(|index| 1.0 / (index.saturating_add(1) as f32))
            .unwrap_or_default();
    }
    let denominator = cases.len() as f32;
    RetrievalEvalReport {
        precision_at_k: precision_sum / denominator,
        recall_at_k: recall_sum / denominator,
        mean_reciprocal_rank: reciprocal_rank_sum / denominator,
    }
}

/// Evaluates graph edge coverage against expected edges.
///
/// # Arguments
///
/// * `manifest` - Manifest containing actual edges.
/// * `expected` - Expected edges whose identity is `from`, `to`, and `kind`.
pub fn evaluate_graph_coverage(
    manifest: &ProjectManifest,
    expected: &[GraphEdge],
) -> GraphCoverageReport {
    let actual = manifest
        .edges
        .iter()
        .map(|edge| (edge.from.as_str(), edge.to.as_str(), &edge.kind))
        .collect::<BTreeSet<_>>();
    let covered_edges = expected
        .iter()
        .filter(|edge| actual.contains(&(edge.from.as_str(), edge.to.as_str(), &edge.kind)))
        .count();
    let coverage = if expected.is_empty() {
        1.0
    } else {
        covered_edges as f32 / expected.len() as f32
    };
    GraphCoverageReport { expected_edges: expected.len(), covered_edges, coverage }
}

/// Evaluates provenance completeness across manifest objects.
///
/// # Arguments
///
/// * `manifest` - Manifest whose provenance fields are checked.
pub fn evaluate_provenance_completeness(
    manifest: &ProjectManifest,
) -> ProvenanceCompletenessReport {
    let mut total = 0usize;
    let mut complete = 0usize;
    for provenance in manifest_provenance(manifest) {
        total = total.saturating_add(1);
        if is_complete(provenance) {
            complete = complete.saturating_add(1);
        }
    }
    let completeness = if total == 0 {
        1.0
    } else {
        complete as f32 / total as f32
    };
    ProvenanceCompletenessReport { total, complete, completeness }
}

/// Evaluates freshness and current manifest file provenance completeness.
///
/// # Arguments
///
/// * `previous` - Baseline manifest.
/// * `current` - Current manifest.
pub fn evaluate_freshness(
    previous: &ProjectManifest,
    current: &ProjectManifest,
) -> FreshnessEvalReport {
    let state = compare_freshness(previous, current);
    let provenance_complete = current
        .files
        .iter()
        .all(|file| is_complete(&file.provenance));
    FreshnessEvalReport { state, provenance_complete }
}

/// Evaluates persisted context-pack artifacts from a project indexer.
///
/// # Arguments
///
/// * `indexer` - Project indexer whose artifact directory is evaluated.
///
/// # Errors
///
/// Returns an error when the artifact directory cannot be listed.
pub fn evaluate_context_pack_artifacts(
    indexer: &ProjectIndexer,
) -> anyhow::Result<ContextPackArtifactEvalReport> {
    let artifact_ids = indexer.list_context_pack_artifacts()?;
    evaluate_context_pack_artifacts_by_id(indexer, &artifact_ids)
}

/// Evaluates explicit context-pack artifact identifiers from a project indexer.
///
/// # Arguments
///
/// * `indexer` - Project indexer used to read persisted artifacts.
/// * `artifact_ids` - Artifact identifiers to validate.
///
/// # Errors
///
/// Returns an error only when directory-level artifact discovery is impossible;
/// individual corrupt or missing artifacts are reported as typed issues.
pub fn evaluate_context_pack_artifacts_by_id(
    indexer: &ProjectIndexer,
    artifact_ids: &[ContextPackArtifactId],
) -> anyhow::Result<ContextPackArtifactEvalReport> {
    let mut issues = Vec::new();
    for artifact_id in artifact_ids {
        match indexer.read_context_pack(artifact_id) {
            Ok(pack) => evaluate_context_pack_artifact(artifact_id, &pack, &mut issues),
            Err(error) => {
                let (code, reason) = if error_is_not_found(&error) {
                    (EvidenceLedgerEvalIssueCode::MissingArtifact, "missing")
                } else {
                    (EvidenceLedgerEvalIssueCode::CorruptArtifact, "corrupt")
                };
                issues.push(issue(
                    code,
                    Some(artifact_id),
                    None,
                    format!("artifact_read_failed:{artifact_id}:{reason}"),
                ));
            }
        }
    }
    Ok(ContextPackArtifactEvalReport {
        checked: artifact_ids.len(),
        valid: issues.is_empty(),
        issues,
    })
}

/// Evaluates redaction-safe tool episodes.
///
/// # Arguments
///
/// * `episodes` - Tool episodes loaded from JSONL storage.
pub fn evaluate_tool_episodes(episodes: &[ToolEpisode]) -> ToolEpisodeEvalReport {
    let mut issues = Vec::new();
    let mut identities = BTreeSet::new();
    for episode in episodes {
        let episode_id = tool_episode_graph_id(episode);
        if episode.input_fingerprint.is_empty() {
            issues.push(issue(
                EvidenceLedgerEvalIssueCode::EmptyEpisodeInputFingerprint,
                None,
                Some(episode_id.clone()),
                "episode_input_fingerprint_empty".to_string(),
            ));
        }
        if episode.output_fingerprint.is_empty() {
            issues.push(issue(
                EvidenceLedgerEvalIssueCode::EmptyEpisodeOutputFingerprint,
                None,
                Some(episode_id.clone()),
                "episode_output_fingerprint_empty".to_string(),
            ));
        }
        if !episode.provenance.is_complete() {
            issues.push(issue(
                EvidenceLedgerEvalIssueCode::IncompleteEpisodeProvenance,
                None,
                Some(episode_id.clone()),
                "episode_provenance_incomplete".to_string(),
            ));
        }
        if !identities.insert(episode_id.clone()) {
            issues.push(issue(
                EvidenceLedgerEvalIssueCode::DuplicateEpisodeIdentity,
                None,
                Some(episode_id),
                "episode_identity_duplicated".to_string(),
            ));
        }
    }
    ToolEpisodeEvalReport { checked: episodes.len(), valid: issues.is_empty(), issues }
}

/// Evaluates whether tool episodes link to existing context-pack artifacts.
///
/// # Arguments
///
/// * `episodes` - Tool episodes loaded from JSONL storage.
/// * `artifact_ids` - Existing context-pack artifact identifiers.
pub fn evaluate_episode_artifact_links(
    episodes: &[ToolEpisode],
    artifact_ids: &[ContextPackArtifactId],
) -> EvidenceLedgerLinkageReport {
    let artifact_set = artifact_ids
        .iter()
        .map(|artifact_id| artifact_id.as_str().to_string())
        .collect::<BTreeSet<_>>();
    let mut linked_count = 0usize;
    let mut issues = Vec::new();
    for episode in episodes {
        let episode_id = tool_episode_graph_id(episode);
        let Some(artifact_id) = episode_context_pack_artifact_id(episode) else {
            issues.push(issue(
                EvidenceLedgerEvalIssueCode::MissingEpisodeArtifactReference,
                None,
                Some(episode_id),
                "episode_context_pack_reference_missing".to_string(),
            ));
            continue;
        };
        if artifact_set.contains(artifact_id.as_str()) {
            linked_count = linked_count.saturating_add(1);
        } else {
            issues.push(issue(
                EvidenceLedgerEvalIssueCode::MissingLinkedArtifact,
                Some(&artifact_id),
                Some(episode_id),
                format!("episode_context_pack_missing:{artifact_id}"),
            ));
        }
    }
    EvidenceLedgerLinkageReport {
        artifact_count: artifact_ids.len(),
        episode_count: episodes.len(),
        linked_count,
        issues,
    }
}

/// Builds a knowledge graph for tool episodes and context-pack artifacts.
///
/// # Arguments
///
/// * `episodes` - Tool episodes to represent as graph nodes.
/// * `artifacts` - Context-pack artifacts to represent as evidence nodes.
///
/// # Errors
///
/// Returns an error when graph validation finds duplicate nodes or invalid edge endpoints.
pub fn tool_episodes_to_graph(
    episodes: &[ToolEpisode],
    artifacts: &[(ContextPackArtifactId, ContextPack)],
) -> anyhow::Result<KnowledgeGraph> {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut artifact_nodes = BTreeMap::new();
    for (artifact_id, pack) in artifacts {
        let node_id = KnowledgeGraphNodeId::RetrievedEvidence(context_pack_graph_id(artifact_id));
        let provenance = context_pack_artifact_provenance(artifact_id);
        artifact_nodes.insert(artifact_id.as_str().to_string(), node_id.clone());
        nodes.push(KnowledgeGraphNode::RetrievedEvidence(
            RetrievedEvidenceGraphNode {
                id: node_id,
                evidence_id: artifact_id.as_str().to_string(),
                path: format!("context_packs/{}.json", artifact_id.as_str()),
                freshness: context_pack_worst_case_freshness(pack),
                provenance,
            },
        ));
    }
    for episode in episodes {
        let episode_node_id = KnowledgeGraphNodeId::ToolEpisode(tool_episode_graph_id(episode));
        nodes.push(KnowledgeGraphNode::ToolEpisode(ToolEpisodeGraphNode {
            id: episode_node_id.clone(),
            tool: episode.tool.clone(),
            status: episode.status.clone(),
            provenance: episode.provenance.clone(),
        }));
        if let Some(artifact_id) = episode_context_pack_artifact_id(episode)
            && let Some(artifact_node_id) = artifact_nodes.get(artifact_id.as_str())
        {
            edges.push(KnowledgeGraphEdge {
                from: episode_node_id,
                to: artifact_node_id.clone(),
                kind: GraphEdgeKind::ToolEpisodeRelates,
                confidence: 0.9,
                confidence_kind: EdgeConfidence::HeuristicHigh,
                provenance: episode.provenance.clone(),
            });
        }
    }
    KnowledgeGraph::new(nodes, edges)
}

/// Aggregates context-pack evidence freshness using the worst-case state.
///
/// # Arguments
///
/// * `pack` - Context pack whose evidence freshness is aggregated.
pub fn context_pack_worst_case_freshness(pack: &ContextPack) -> EvidenceFreshness {
    pack.evidence
        .iter()
        .map(|evidence| evidence.freshness.clone())
        .max_by_key(freshness_rank)
        .unwrap_or(EvidenceFreshness::Fresh)
}

/// Returns the deterministic graph identifier for a tool episode.
///
/// # Arguments
///
/// * `episode` - Redaction-safe tool episode.
pub fn tool_episode_graph_id(episode: &ToolEpisode) -> String {
    let identity = [
        episode.tool.as_str(),
        episode.timestamp.as_str(),
        episode.input_fingerprint.as_str(),
        episode.output_fingerprint.as_str(),
        episode.provenance.fingerprint.as_str(),
    ];
    fingerprint(&length_prefixed_fields(&identity))
}

fn freshness_rank(freshness: &EvidenceFreshness) -> u8 {
    match freshness {
        EvidenceFreshness::Fresh => 0,
        EvidenceFreshness::Added => 1,
        EvidenceFreshness::Changed => 2,
        EvidenceFreshness::Deleted => 3,
    }
}

fn length_prefixed_fields(fields: &[&str]) -> String {
    let mut encoded = String::new();
    for field in fields {
        encoded.push_str(&field.len().to_string());
        encoded.push(':');
        encoded.push_str(field);
        encoded.push(';');
    }
    encoded
}

fn evaluate_context_pack_artifact(
    artifact_id: &ContextPackArtifactId,
    pack: &ContextPack,
    issues: &mut Vec<EvidenceLedgerEvalIssue>,
) {
    if pack.evidence.is_empty() {
        issues.push(issue(
            EvidenceLedgerEvalIssueCode::EmptyArtifactEvidence,
            Some(artifact_id),
            None,
            format!("artifact_evidence_empty:{artifact_id}"),
        ));
    }
    if pack.provenance.is_empty() {
        issues.push(issue(
            EvidenceLedgerEvalIssueCode::IncompleteArtifactProvenance,
            Some(artifact_id),
            None,
            format!("artifact_provenance_empty:{artifact_id}"),
        ));
    }
    for provenance in &pack.provenance {
        if !provenance.is_complete() {
            issues.push(issue(
                EvidenceLedgerEvalIssueCode::IncompleteArtifactProvenance,
                Some(artifact_id),
                None,
                format!("artifact_provenance_incomplete:{artifact_id}"),
            ));
        }
    }
    for evidence in &pack.evidence {
        if !evidence.provenance.is_complete() {
            issues.push(issue(
                EvidenceLedgerEvalIssueCode::IncompleteArtifactProvenance,
                Some(artifact_id),
                None,
                format!(
                    "artifact_evidence_provenance_incomplete:{artifact_id}:{}",
                    evidence.id
                ),
            ));
        }
        if matches!(
            evidence.freshness,
            EvidenceFreshness::Changed | EvidenceFreshness::Deleted
        ) {
            issues.push(issue(
                EvidenceLedgerEvalIssueCode::StaleArtifactEvidence,
                Some(artifact_id),
                None,
                format!("artifact_evidence_stale:{artifact_id}:{}", evidence.id),
            ));
        }
    }
}

fn episode_context_pack_artifact_id(episode: &ToolEpisode) -> Option<ContextPackArtifactId> {
    let id = episode
        .provenance
        .path
        .strip_prefix("context_packs/")?
        .strip_suffix(".json")?;
    ContextPackArtifactId::new(id.to_string()).ok()
}

fn context_pack_graph_id(artifact_id: &ContextPackArtifactId) -> String {
    format!("context_pack_artifact:{}", artifact_id.as_str())
}

fn context_pack_artifact_provenance(artifact_id: &ContextPackArtifactId) -> Provenance {
    Provenance {
        path: format!("context_packs/{}.json", artifact_id.as_str()),
        start_line: None,
        end_line: None,
        source: "context_pack_artifact".to_string(),
        fingerprint: fingerprint(&format!("context_pack_artifact:{}", artifact_id.as_str())),
    }
}

fn issue(
    code: EvidenceLedgerEvalIssueCode,
    artifact_id: Option<&ContextPackArtifactId>,
    episode_fingerprint: Option<String>,
    detail: String,
) -> EvidenceLedgerEvalIssue {
    EvidenceLedgerEvalIssue {
        code,
        artifact_id: artifact_id.map(|value| value.as_str().to_string()),
        episode_fingerprint,
        detail,
    }
}

fn error_is_not_found(error: &anyhow::Error) -> bool {
    error
        .chain()
        .filter_map(|cause| cause.downcast_ref::<std::io::Error>())
        .any(|cause| cause.kind() == std::io::ErrorKind::NotFound)
}

fn manifest_provenance(manifest: &ProjectManifest) -> Vec<&crate::types::Provenance> {
    manifest
        .files
        .iter()
        .map(|file| &file.provenance)
        .chain(manifest.file_nodes.iter().map(|node| &node.provenance))
        .chain(manifest.symbols.iter().map(|symbol| &symbol.provenance))
        .chain(manifest.edges.iter().map(|edge| &edge.provenance))
        .chain(manifest.shards.iter().map(|shard| &shard.provenance))
        .collect()
}

fn is_complete(provenance: &crate::types::Provenance) -> bool {
    !provenance.path.is_empty()
        && !provenance.source.is_empty()
        && !provenance.fingerprint.is_empty()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use anyhow::Result;
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::indexer::tests::fixture_project;
    use crate::{
        ContextPackSelection, EvidenceLedgerEvalIssueCode, FreshnessState, GraphEdgeKind,
        ProjectIndexer, RetrievalQuery, StaleEvidencePolicy,
    };

    fn write_fixture_context_pack(
        indexer: &ProjectIndexer,
        manifest: &ProjectManifest,
    ) -> Result<(ContextPackArtifactId, ContextPack)> {
        let result = retrieve(
            manifest,
            &RetrievalQuery {
                text: Some("Root".to_string()),
                path: None,
                path_prefix: None,
                symbol: None,
                limit: 1,
                include_graph_expansion: false,
            },
        )
        .into_iter()
        .next()
        .expect("fixture should retrieve Root evidence");
        let pack = ContextPack::from_selection(
            manifest,
            ContextPackSelection {
                retrieval_results: vec![result],
                shards: Vec::new(),
                evidence: Vec::new(),
                freshness: FreshnessState { fresh: true, ..Default::default() },
                stale_policy: StaleEvidencePolicy::Reject,
            },
        )?;
        indexer.write_context_pack(&pack)?;
        let id = indexer.context_pack_artifact_id(&pack)?;
        Ok((id, pack))
    }

    fn fixture_episode(artifact_id: &ContextPackArtifactId) -> ToolEpisode {
        ToolEpisode {
            timestamp: "2026-05-19T14:42:31+03:00".to_string(),
            tool: "project_model_search".to_string(),
            input_fingerprint: crate::fingerprint("input"),
            output_fingerprint: crate::fingerprint("output"),
            status: "success".to_string(),
            provenance: Provenance {
                path: format!("context_packs/{}.json", artifact_id.as_str()),
                start_line: None,
                end_line: None,
                source: "WorkspaceService::query_workspace".to_string(),
                fingerprint: crate::fingerprint("episode-provenance"),
            },
        }
    }

    fn artifact_path(model_root: &std::path::Path, artifact_id: &ContextPackArtifactId) -> PathBuf {
        model_root
            .join("model")
            .join("context_packs")
            .join(format!("{}.json", artifact_id.as_str()))
    }

    #[test]
    fn evaluates_context_pack_episode_linkage_and_graph_happy_path() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let indexer = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = indexer.index()?;
        let (artifact_id, pack) = write_fixture_context_pack(&indexer, &manifest)?;
        let artifacts = vec![(artifact_id.clone(), pack)];
        let episode = fixture_episode(&artifact_id);

        let artifact_report = evaluate_context_pack_artifacts(&indexer)?;
        let episode_report = evaluate_tool_episodes(std::slice::from_ref(&episode));
        let linkage_report = evaluate_episode_artifact_links(
            std::slice::from_ref(&episode),
            std::slice::from_ref(&artifact_id),
        );
        let graph = tool_episodes_to_graph(std::slice::from_ref(&episode), &artifacts)?;
        let graph_again = tool_episodes_to_graph(std::slice::from_ref(&episode), &artifacts)?;

        assert_eq!(artifact_report.checked, 1usize);
        assert_eq!(artifact_report.valid, true);
        assert_eq!(episode_report.checked, 1usize);
        assert_eq!(episode_report.valid, true);
        assert_eq!(linkage_report.linked_count, 1usize);
        assert_eq!(linkage_report.issues, Vec::new());
        assert_eq!(graph, graph_again);
        assert_eq!(graph.nodes.len(), 2usize);
        assert_eq!(graph.edges.len(), 1usize);
        let actual_freshness = graph
            .nodes
            .iter()
            .find_map(|node| match node {
                KnowledgeGraphNode::RetrievedEvidence(evidence) => Some(evidence.freshness.clone()),
                _ => None,
            })
            .expect("graph should include artifact evidence node");
        assert_eq!(actual_freshness, EvidenceFreshness::Fresh);
        let actual_edge = graph.edges.first().expect("graph should include one edge");
        assert_eq!(actual_edge.kind, GraphEdgeKind::ToolEpisodeRelates);
        assert_eq!(actual_edge.confidence_kind, EdgeConfidence::HeuristicHigh);
        assert_eq!(
            graph
                .edges
                .iter()
                .any(|edge| edge.confidence_kind == EdgeConfidence::ExactCompiler),
            false
        );
        let report_json = serde_json::to_string(&artifact_report)?;
        assert!(!report_json.contains("pub struct"));
        assert!(!report_json.contains("capture a deterministic"));
        assert!(!report_json.contains("<project_model_context>"));
        Ok(())
    }

    #[test]
    fn reports_missing_and_corrupt_context_pack_artifacts() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let indexer = ProjectIndexer::new(&root, fixture.path().join("model"));
        let missing_id = ContextPackArtifactId::new("a".repeat(64))?;
        let corrupt_id = ContextPackArtifactId::new("b".repeat(64))?;
        fs::create_dir_all(fixture.path().join("model").join("context_packs"))?;
        fs::write(artifact_path(fixture.path(), &corrupt_id), "not json")?;

        let actual = evaluate_context_pack_artifacts_by_id(
            &indexer,
            &[missing_id.clone(), corrupt_id.clone()],
        )?;
        let actual_codes = actual
            .issues
            .iter()
            .map(|issue| issue.code.clone())
            .collect::<BTreeSet<_>>();
        let expected = BTreeSet::from([
            EvidenceLedgerEvalIssueCode::MissingArtifact,
            EvidenceLedgerEvalIssueCode::CorruptArtifact,
        ]);

        assert_eq!(actual.checked, 2usize);
        assert_eq!(actual.valid, false);
        assert_eq!(actual_codes, expected);
        Ok(())
    }

    #[test]
    fn reports_stale_and_incomplete_context_pack_artifact() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let indexer = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = indexer.index()?;
        let (_id, mut pack) = write_fixture_context_pack(&indexer, &manifest)?;
        let evidence = pack
            .evidence
            .first_mut()
            .expect("fixture context pack should include evidence");
        evidence.freshness = EvidenceFreshness::Changed;
        evidence.provenance.fingerprint.clear();
        pack.provenance.clear();
        let stale_id = indexer.context_pack_artifact_id(&pack)?;
        fs::write(
            artifact_path(fixture.path(), &stale_id),
            pack.to_stable_json()?,
        )?;

        let actual = evaluate_context_pack_artifacts_by_id(&indexer, &[stale_id])?;
        let actual_codes = actual
            .issues
            .iter()
            .map(|issue| issue.code.clone())
            .collect::<BTreeSet<_>>();
        let expected = BTreeSet::from([
            EvidenceLedgerEvalIssueCode::IncompleteArtifactProvenance,
            EvidenceLedgerEvalIssueCode::StaleArtifactEvidence,
        ]);

        assert_eq!(actual.valid, false);
        assert_eq!(actual_codes, expected);
        Ok(())
    }

    #[test]
    fn reports_episode_fingerprint_provenance_and_duplicate_failures() -> Result<()> {
        let artifact_id = ContextPackArtifactId::new("c".repeat(64))?;
        let mut episode = fixture_episode(&artifact_id);
        episode.input_fingerprint.clear();
        episode.output_fingerprint.clear();
        episode.provenance.fingerprint.clear();
        let actual = evaluate_tool_episodes(&[episode.clone(), episode]);
        let actual_codes = actual
            .issues
            .iter()
            .map(|issue| issue.code.clone())
            .collect::<BTreeSet<_>>();
        let expected = BTreeSet::from([
            EvidenceLedgerEvalIssueCode::DuplicateEpisodeIdentity,
            EvidenceLedgerEvalIssueCode::EmptyEpisodeInputFingerprint,
            EvidenceLedgerEvalIssueCode::EmptyEpisodeOutputFingerprint,
            EvidenceLedgerEvalIssueCode::IncompleteEpisodeProvenance,
        ]);

        assert_eq!(actual.checked, 2usize);
        assert_eq!(actual.valid, false);
        assert_eq!(actual_codes, expected);
        Ok(())
    }

    #[test]
    fn episode_identity_uses_length_prefixed_fields() -> Result<()> {
        let artifact_id = ContextPackArtifactId::new("e".repeat(64))?;
        let mut left = fixture_episode(&artifact_id);
        left.tool = "tool=a;timestamp=b".to_string();
        left.timestamp = "c".to_string();
        let mut right = fixture_episode(&artifact_id);
        right.tool = "tool=a".to_string();
        right.timestamp = "b;timestamp=c".to_string();

        let left_id = tool_episode_graph_id(&left);
        let right_id = tool_episode_graph_id(&right);
        let report = evaluate_tool_episodes(&[left, right]);

        assert_ne!(left_id, right_id);
        assert_eq!(report.valid, true);
        Ok(())
    }

    #[test]
    fn reports_missing_episode_artifact_links_and_rejects_duplicate_graph_nodes() -> Result<()> {
        let artifact_id = ContextPackArtifactId::new("d".repeat(64))?;
        let mut missing_reference = fixture_episode(&artifact_id);
        missing_reference.provenance.path = "episodes/local.jsonl".to_string();
        let missing_artifact = fixture_episode(&artifact_id);
        let linkage = evaluate_episode_artifact_links(&[missing_reference, missing_artifact], &[]);
        let actual_codes = linkage
            .issues
            .iter()
            .map(|issue| issue.code.clone())
            .collect::<BTreeSet<_>>();
        let expected = BTreeSet::from([
            EvidenceLedgerEvalIssueCode::MissingEpisodeArtifactReference,
            EvidenceLedgerEvalIssueCode::MissingLinkedArtifact,
        ]);
        let duplicate = fixture_episode(&artifact_id);
        let pack = ContextPack {
            version: 1,
            manifest_hash: "test".to_string(),
            evidence: Vec::new(),
            provenance: vec![context_pack_artifact_provenance(&artifact_id)],
        };
        let artifacts = vec![(artifact_id.clone(), pack)];
        let graph_error = tool_episodes_to_graph(&[duplicate.clone(), duplicate], &artifacts)
            .expect_err("duplicate episode graph nodes should fail validation")
            .to_string();

        assert_eq!(linkage.linked_count, 0usize);
        assert_eq!(actual_codes, expected);
        assert!(graph_error.contains("duplicated"));
        Ok(())
    }

    #[test]
    fn graph_helper_does_not_mark_stale_artifact_as_fresh() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let indexer = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = indexer.index()?;
        let (_id, mut pack) = write_fixture_context_pack(&indexer, &manifest)?;
        pack.evidence
            .first_mut()
            .expect("fixture context pack should include evidence")
            .freshness = EvidenceFreshness::Changed;
        let stale_id = indexer.context_pack_artifact_id(&pack)?;
        fs::write(
            artifact_path(fixture.path(), &stale_id),
            pack.to_stable_json()?,
        )?;
        let episode = fixture_episode(&stale_id);
        let artifacts = vec![(stale_id.clone(), pack)];

        let graph = tool_episodes_to_graph(&[episode], &artifacts)?;
        let actual = graph
            .nodes
            .iter()
            .find_map(|node| match node {
                KnowledgeGraphNode::RetrievedEvidence(evidence) => Some(evidence.freshness.clone()),
                _ => None,
            })
            .expect("graph should include artifact evidence node");
        let expected = EvidenceFreshness::Changed;

        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn context_pack_worst_case_freshness_prefers_deleted_over_changed_added_fresh() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let indexer = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = indexer.index()?;
        let (_id, mut pack) = write_fixture_context_pack(&indexer, &manifest)?;
        let fixture_evidence = pack
            .evidence
            .first()
            .expect("fixture context pack should include evidence")
            .clone();
        let mut added = fixture_evidence.clone();
        added.id = "added".to_string();
        added.freshness = EvidenceFreshness::Added;
        let mut changed = fixture_evidence.clone();
        changed.id = "changed".to_string();
        changed.freshness = EvidenceFreshness::Changed;
        let mut deleted = fixture_evidence;
        deleted.id = "deleted".to_string();
        deleted.freshness = EvidenceFreshness::Deleted;
        pack.evidence
            .first_mut()
            .expect("fixture context pack should include evidence")
            .freshness = EvidenceFreshness::Fresh;
        pack.evidence.extend([added, changed, deleted]);

        let actual = context_pack_worst_case_freshness(&pack);
        let expected = EvidenceFreshness::Deleted;

        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn evaluates_retrieval_metrics_on_fixture() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let root_symbol = manifest
            .symbols
            .iter()
            .find(|symbol| symbol.name == "Root")
            .expect("fixture should include Root symbol");
        let cases = vec![RetrievalEvalCase {
            query: RetrievalQuery {
                text: Some("Root".to_string()),
                path: None,
                path_prefix: None,
                symbol: None,
                limit: 3,
                include_graph_expansion: false,
            },
            relevant_ids: BTreeSet::from([root_symbol.id.clone()]),
        }];
        let actual = evaluate_retrieval(&manifest, &cases, 3);
        assert_eq!(actual.recall_at_k, 1.0);
        assert_eq!(actual.mean_reciprocal_rank > 0.0, true);
        Ok(())
    }

    #[test]
    fn evaluates_graph_coverage_and_provenance() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let expected = manifest
            .edges
            .iter()
            .filter(|edge| edge.kind == GraphEdgeKind::Contains)
            .take(1)
            .cloned()
            .collect::<Vec<_>>();
        let actual = evaluate_graph_coverage(&manifest, &expected);
        assert_eq!(actual.coverage, 1.0);
        assert_eq!(
            evaluate_provenance_completeness(&manifest).completeness,
            1.0
        );
        Ok(())
    }

    #[test]
    fn evaluates_freshness_with_provenance_completeness() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let previous = setup.index()?;
        fs::write(root.join("src").join("added.rs"), "pub fn added() {}\n")?;
        let current = setup.index()?;
        let actual = evaluate_freshness(&previous, &current);
        assert_eq!(actual.state.added, vec!["src/added.rs".to_string()]);
        assert_eq!(actual.provenance_complete, true);
        Ok(())
    }
}
