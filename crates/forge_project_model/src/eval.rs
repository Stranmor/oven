//! Rust-native evaluation harness for retrieval, graph, freshness, and
//! provenance.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::ProjectIndexer;
use crate::freshness::compare_freshness;
use crate::retrieval::retrieve;
use crate::types::{
    ContextPack, ContextPackArtifactEvalReport, ContextPackArtifactId, EdgeConfidence,
    EvidenceFreshness, EvidenceLedgerActivation, EvidenceLedgerActivationSummary,
    EvidenceLedgerEvalIssue, EvidenceLedgerEvalIssueCode, EvidenceLedgerGraphMetadata,
    EvidenceLedgerLinkageReport, EvidenceReadinessDiagnostic, FreshnessEvalReport,
    GraphCoverageReport, GraphEdge, GraphEdgeKind, KnowledgeGraph, KnowledgeGraphEdge,
    KnowledgeGraphNode, KnowledgeGraphNodeId, ProjectManifest, Provenance,
    ProvenanceCompletenessReport, RetrievalEvalCase, RetrievalEvalReport,
    RetrievedEvidenceGraphNode, ToolEpisode, ToolEpisodeEvalReport, ToolEpisodeGraphNode,
};
use crate::util::fingerprint;

/// Bounded budgets for read-only evidence readiness diagnostics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceReadinessDiagnosticBudget {
    /// Maximum context-pack artifact files to inspect.
    pub max_artifacts: usize,
    /// Maximum non-empty tool episode JSONL lines to inspect.
    pub max_episode_lines: usize,
    /// Maximum redaction-safe issue summaries to retain.
    pub max_issue_summaries: usize,
}

impl Default for EvidenceReadinessDiagnosticBudget {
    fn default() -> Self {
        Self {
            max_artifacts: 128,
            max_episode_lines: 512,
            max_issue_summaries: 16,
        }
    }
}

/// Bounded budgets for read-only evidence-ledger activation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceLedgerActivationBudget {
    /// Maximum context-pack artifact files to inspect.
    pub max_artifacts: usize,
    /// Maximum non-empty tool episode JSONL lines to inspect.
    pub max_episode_lines: usize,
    /// Maximum graph nodes allowed before graph metadata is omitted.
    pub max_nodes: usize,
    /// Maximum graph edges allowed before graph metadata is omitted.
    pub max_edges: usize,
    /// Maximum redaction-safe issue summaries to retain.
    pub max_issue_summaries: usize,
}

impl Default for EvidenceLedgerActivationBudget {
    fn default() -> Self {
        Self {
            max_artifacts: 128,
            max_episode_lines: 512,
            max_nodes: 512,
            max_edges: 512,
            max_issue_summaries: 16,
        }
    }
}

/// Builds a read-only bounded evidence-ledger activation from persisted artifacts only.
///
/// This API reads existing context-pack artifacts and tool episodes, then returns
/// redaction-safe counters and graph metadata. It does not index, sync, write,
/// append, repair, invoke producers, or expose source/tool payload content.
///
/// # Arguments
///
/// * `indexer` - Project indexer whose model storage is inspected.
/// * `budget` - Hard caps for artifact, episode, graph, and issue-summary inspection.
///
/// # Errors
///
/// Returns an error only when typed graph metadata construction fails for the
/// already-readable artifacts and episodes.
pub fn load_evidence_ledger_activation(
    indexer: &ProjectIndexer,
    budget: &EvidenceLedgerActivationBudget,
) -> anyhow::Result<EvidenceLedgerActivation> {
    let readiness_budget = EvidenceReadinessDiagnosticBudget {
        max_artifacts: budget.max_artifacts,
        max_episode_lines: budget.max_episode_lines,
        max_issue_summaries: budget.max_issue_summaries,
    };
    let readiness = diagnose_evidence_readiness(indexer, &readiness_budget);
    let mut builder = EvidenceReadinessDiagnosticBuilder::new(budget.max_issue_summaries);
    let artifact_paths =
        context_pack_artifact_paths(indexer.model_dir(), &readiness_budget, &mut builder);
    let mut readable_artifacts = Vec::new();
    for (artifact_id, _path) in &artifact_paths {
        if let Ok(pack) = indexer.read_context_pack(artifact_id) {
            readable_artifacts.push((artifact_id.clone(), pack));
        }
    }
    let episode_read =
        read_tool_episode_lines(indexer.model_dir(), &readiness_budget, &mut builder);
    let readable_artifact_ids = readable_artifacts
        .iter()
        .map(|(artifact_id, _pack)| artifact_id.clone())
        .collect::<Vec<_>>();
    let activation_linkage =
        evaluate_episode_artifact_links(&episode_read.episodes, &readable_artifact_ids);
    let (issue_summaries, activation_issues_truncated) = activation_issue_summaries(
        &readiness.issue_summaries,
        &activation_linkage.issues,
        budget.max_issue_summaries,
    );
    let graph = tool_episodes_to_graph(&episode_read.episodes, &readable_artifacts)?;
    let graph_metadata = EvidenceLedgerGraphMetadata::from_graph(&graph);
    let graph_over_budget = graph_metadata.node_count > budget.max_nodes
        || graph_metadata.edge_count > budget.max_edges;
    let graph = (!graph_over_budget).then_some(graph_metadata.clone());
    let issue_count = readiness
        .context_pack_issue_count
        .saturating_add(readiness.tool_episode_issue_count)
        .saturating_add(activation_linkage.issues.len());
    let summary = EvidenceLedgerActivationSummary {
        context_pack_artifact_count: readiness.context_pack_artifact_count,
        readable_context_pack_count: readable_artifacts.len(),
        tool_episode_count: readiness.tool_episode_count,
        linked_episode_count: activation_linkage.linked_count,
        missing_link_count: activation_linkage.issues.len(),
        graph_node_count: graph_metadata.node_count,
        graph_edge_count: graph_metadata.edge_count,
        worst_case_freshness: readiness.worst_case_freshness.clone(),
        issue_count,
        issue_summaries,
        truncated: readiness.truncated || graph_over_budget || activation_issues_truncated,
    };
    Ok(EvidenceLedgerActivation { summary, readiness, graph })
}

/// Builds a read-only bounded evidence readiness diagnostic for context packs and tool episodes.
///
/// Missing storage files are reported as valid zero-count diagnostics. Corrupt
/// artifacts and malformed JSONL lines are counted as redaction-safe issues
/// without aborting the whole report.
///
/// # Arguments
///
/// * `indexer` - Project indexer whose model storage is inspected.
/// * `budget` - Hard caps for artifact, episode, and issue-summary inspection.
pub fn diagnose_evidence_readiness(
    indexer: &ProjectIndexer,
    budget: &EvidenceReadinessDiagnosticBudget,
) -> EvidenceReadinessDiagnostic {
    let mut builder = EvidenceReadinessDiagnosticBuilder::new(budget.max_issue_summaries);
    let artifact_paths = context_pack_artifact_paths(indexer.model_dir(), budget, &mut builder);
    let mut artifact_ids = Vec::new();
    let mut readable_packs = Vec::new();

    for (artifact_id, _path) in &artifact_paths {
        artifact_ids.push(artifact_id.clone());
        if let Ok(pack) = indexer.read_context_pack(artifact_id) {
            readable_packs.push(pack);
        }
    }

    let artifact_report = evaluate_context_pack_artifacts_by_id(indexer, &artifact_ids)
        .unwrap_or_else(|error| {
            builder.push_issue(format!("context_pack_artifact_eval_failed:{error}"));
            ContextPackArtifactEvalReport {
                checked: artifact_ids.len(),
                valid: false,
                issues: Vec::new(),
            }
        });
    builder.push_eval_issues("context_pack", &artifact_report.issues);

    let episode_read = read_tool_episode_lines(indexer.model_dir(), budget, &mut builder);
    let episode_report = evaluate_tool_episodes(&episode_read.episodes);
    builder.push_eval_issues("tool_episode", &episode_report.issues);

    let linkage_report = evaluate_episode_artifact_links(&episode_read.episodes, &artifact_ids);
    builder.push_eval_issues("episode_artifact_link", &linkage_report.issues);

    let worst_case_freshness = readable_packs
        .iter()
        .map(context_pack_worst_case_freshness)
        .max_by_key(freshness_rank)
        .map(|freshness| evidence_freshness_label(&freshness).to_string());

    let tool_episode_issue_count = episode_report
        .issues
        .len()
        .saturating_add(episode_read.malformed_line_count)
        .saturating_add(episode_read.read_issue_count);
    let context_pack_issue_count = artifact_report
        .issues
        .len()
        .saturating_add(builder.context_pack_reader_issue_count);

    EvidenceReadinessDiagnostic {
        context_pack_artifact_count: artifact_report.checked,
        context_pack_valid: artifact_report.valid && builder.context_pack_reader_issue_count == 0,
        context_pack_issue_count,
        tool_episode_count: episode_report.checked,
        tool_episode_valid: episode_report.valid
            && episode_read.malformed_line_count == 0
            && episode_read.read_issue_count == 0,
        tool_episode_issue_count,
        episode_artifact_link_valid: linkage_report.issues.is_empty(),
        linked_episode_count: linkage_report.linked_count,
        missing_link_count: linkage_report.issues.len(),
        worst_case_freshness,
        issue_summaries: builder.issue_summaries,
        truncated: builder.truncated || episode_read.truncated,
    }
}

fn activation_issue_summaries(
    readiness_summaries: &[String],
    linkage_issues: &[EvidenceLedgerEvalIssue],
    max_issue_summaries: usize,
) -> (Vec<String>, bool) {
    let mut summaries = Vec::new();
    let mut truncated = false;
    for summary in readiness_summaries {
        push_activation_issue_summary(
            &mut summaries,
            &mut truncated,
            max_issue_summaries,
            summary.clone(),
        );
    }
    for issue in linkage_issues {
        push_activation_issue_summary(
            &mut summaries,
            &mut truncated,
            max_issue_summaries,
            format!("episode_artifact_link:{:?}", issue.code),
        );
    }
    (summaries, truncated)
}

fn push_activation_issue_summary(
    summaries: &mut Vec<String>,
    truncated: &mut bool,
    max_issue_summaries: usize,
    summary: String,
) {
    if summaries.contains(&summary) {
        return;
    }
    if summaries.len() < max_issue_summaries {
        summaries.push(summary);
    } else {
        *truncated = true;
    }
}

struct EvidenceReadinessDiagnosticBuilder {
    max_issue_summaries: usize,
    issue_summaries: Vec<String>,
    truncated: bool,
    context_pack_reader_issue_count: usize,
}

impl EvidenceReadinessDiagnosticBuilder {
    fn new(max_issue_summaries: usize) -> Self {
        Self {
            max_issue_summaries,
            issue_summaries: Vec::new(),
            truncated: false,
            context_pack_reader_issue_count: 0,
        }
    }

    fn push_issue(&mut self, summary: String) {
        if summary.starts_with("context_pack_") {
            self.context_pack_reader_issue_count =
                self.context_pack_reader_issue_count.saturating_add(1);
        }
        if self.issue_summaries.len() < self.max_issue_summaries {
            self.issue_summaries.push(summary);
        } else {
            self.truncated = true;
        }
    }

    fn push_eval_issues(&mut self, prefix: &str, issues: &[EvidenceLedgerEvalIssue]) {
        for issue in issues {
            self.push_issue(format!("{prefix}:{:?}", issue.code));
        }
    }
}

struct EpisodeReadResult {
    episodes: Vec<ToolEpisode>,
    malformed_line_count: usize,
    read_issue_count: usize,
    truncated: bool,
}

fn context_pack_artifact_paths(
    model_dir: &Path,
    budget: &EvidenceReadinessDiagnosticBudget,
    builder: &mut EvidenceReadinessDiagnosticBuilder,
) -> Vec<(ContextPackArtifactId, std::path::PathBuf)> {
    let directory = model_dir.join("context_packs");
    let Ok(entries) = fs::read_dir(&directory) else {
        if directory.exists() {
            builder.push_issue("context_pack_dir_unreadable".to_string());
        }
        return Vec::new();
    };
    let mut artifact_paths = Vec::new();
    for entry in entries {
        let Ok(entry) = entry else {
            builder.push_issue("context_pack_dir_entry_unreadable".to_string());
            continue;
        };
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|value| value.to_str()) else {
            builder.push_issue("context_pack_filename_non_utf8".to_string());
            continue;
        };
        let Some(raw_id) = file_name.strip_suffix(".json") else {
            continue;
        };
        let Ok(artifact_id) = ContextPackArtifactId::new(raw_id.to_string()) else {
            builder.push_issue("context_pack_filename_invalid".to_string());
            continue;
        };
        if artifact_paths.len() >= budget.max_artifacts {
            builder.truncated = true;
            break;
        }
        artifact_paths.push((artifact_id, path));
    }
    artifact_paths.sort_by(|left, right| left.0.cmp(&right.0));
    artifact_paths
}

fn read_tool_episode_lines(
    model_dir: &Path,
    budget: &EvidenceReadinessDiagnosticBudget,
    builder: &mut EvidenceReadinessDiagnosticBuilder,
) -> EpisodeReadResult {
    let path = model_dir.join("tool_episodes.jsonl");
    let Ok(file) = File::open(&path) else {
        if path.exists() {
            builder.push_issue("tool_episode_file_unreadable".to_string());
            return EpisodeReadResult {
                episodes: Vec::new(),
                malformed_line_count: 0,
                read_issue_count: 1,
                truncated: false,
            };
        }
        return EpisodeReadResult {
            episodes: Vec::new(),
            malformed_line_count: 0,
            read_issue_count: 0,
            truncated: false,
        };
    };
    let mut episodes = Vec::new();
    let mut malformed_line_count = 0usize;
    let mut read_issue_count = 0usize;
    let mut truncated = false;
    for line in BufReader::new(file).lines() {
        let line = match line {
            Ok(line) => line,
            Err(_) => {
                read_issue_count = read_issue_count.saturating_add(1);
                builder.push_issue("tool_episode_line_unreadable".to_string());
                continue;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        if episodes.len().saturating_add(malformed_line_count) >= budget.max_episode_lines {
            truncated = true;
            break;
        }
        match serde_json::from_str::<ToolEpisode>(&line) {
            Ok(episode) => episodes.push(episode),
            Err(_) => {
                malformed_line_count = malformed_line_count.saturating_add(1);
                builder.push_issue("tool_episode_line_malformed".to_string());
            }
        }
    }
    EpisodeReadResult { episodes, malformed_line_count, read_issue_count, truncated }
}

fn evidence_freshness_label(freshness: &EvidenceFreshness) -> &'static str {
    match freshness {
        EvidenceFreshness::Fresh => "fresh",
        EvidenceFreshness::Added => "added",
        EvidenceFreshness::Changed => "changed",
        EvidenceFreshness::Deleted => "deleted",
    }
}

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
        .chain(
            manifest
                .cargo_workspace
                .iter()
                .map(|workspace| &workspace.provenance),
        )
        .chain(
            manifest
                .cargo_packages
                .iter()
                .map(|package| &package.provenance),
        )
        .chain(
            manifest
                .cargo_packages
                .iter()
                .flat_map(|package| package.targets.iter().map(|target| &target.provenance)),
        )
        .chain(
            manifest
                .cargo_packages
                .iter()
                .flat_map(|package| package.features.iter().map(|feature| &feature.provenance)),
        )
        .chain(
            manifest
                .cargo_package_dependencies
                .iter()
                .map(|dependency| &dependency.provenance),
        )
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
    fn evidence_readiness_missing_context_pack_dir_and_episode_file_is_valid_zero_count()
    -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let model_dir = fixture.path().join("model");
        let setup = ProjectIndexer::new(&root, &model_dir);

        let actual =
            diagnose_evidence_readiness(&setup, &EvidenceReadinessDiagnosticBudget::default());
        let expected = EvidenceReadinessDiagnostic {
            context_pack_artifact_count: 0,
            context_pack_valid: true,
            context_pack_issue_count: 0,
            tool_episode_count: 0,
            tool_episode_valid: true,
            tool_episode_issue_count: 0,
            episode_artifact_link_valid: true,
            linked_episode_count: 0,
            missing_link_count: 0,
            worst_case_freshness: None,
            issue_summaries: Vec::new(),
            truncated: false,
        };

        assert_eq!(actual, expected);
        assert_eq!(model_dir.exists(), false);
        Ok(())
    }

    #[test]
    fn evidence_readiness_counts_corrupt_context_pack_artifact_without_panic() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let artifact_id = ContextPackArtifactId::new("b".repeat(64))?;
        fs::create_dir_all(fixture.path().join("model").join("context_packs"))?;
        fs::write(artifact_path(fixture.path(), &artifact_id), "not json")?;

        let actual =
            diagnose_evidence_readiness(&setup, &EvidenceReadinessDiagnosticBudget::default());
        let expected = (1usize, false, 1usize, true);

        assert_eq!(
            (
                actual.context_pack_artifact_count,
                actual.context_pack_valid,
                actual.context_pack_issue_count,
                actual
                    .issue_summaries
                    .iter()
                    .any(|summary| summary == "context_pack:CorruptArtifact"),
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn evidence_readiness_counts_corrupt_episode_line_without_aborting_report() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        fs::create_dir_all(fixture.path().join("model"))?;
        fs::write(
            fixture.path().join("model").join("tool_episodes.jsonl"),
            "not json\n",
        )?;

        let actual =
            diagnose_evidence_readiness(&setup, &EvidenceReadinessDiagnosticBudget::default());
        let expected = (0usize, false, 1usize, true);

        assert_eq!(
            (
                actual.tool_episode_count,
                actual.tool_episode_valid,
                actual.tool_episode_issue_count,
                actual
                    .issue_summaries
                    .iter()
                    .any(|summary| summary == "tool_episode_line_malformed"),
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn evidence_readiness_reports_episode_link_to_missing_context_pack_artifact() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let artifact_id = ContextPackArtifactId::new("a".repeat(64))?;
        setup.append_episode(&fixture_episode(&artifact_id))?;

        let actual =
            diagnose_evidence_readiness(&setup, &EvidenceReadinessDiagnosticBudget::default());
        let expected = (1usize, false, 0usize, 1usize);

        assert_eq!(
            (
                actual.tool_episode_count,
                actual.episode_artifact_link_valid,
                actual.linked_episode_count,
                actual.missing_link_count,
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn evidence_readiness_aggregates_worst_case_freshness_for_readable_packs() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let (_id, mut changed_pack) = write_fixture_context_pack(&setup, &manifest)?;
        changed_pack
            .evidence
            .first_mut()
            .expect("fixture context pack should include evidence")
            .freshness = EvidenceFreshness::Changed;
        let changed_id = setup.context_pack_artifact_id(&changed_pack)?;
        fs::write(
            artifact_path(fixture.path(), &changed_id),
            changed_pack.to_stable_json()?,
        )?;
        let (_id, mut deleted_pack) = write_fixture_context_pack(&setup, &manifest)?;
        deleted_pack
            .evidence
            .first_mut()
            .expect("fixture context pack should include evidence")
            .freshness = EvidenceFreshness::Deleted;
        let deleted_id = setup.context_pack_artifact_id(&deleted_pack)?;
        fs::write(
            artifact_path(fixture.path(), &deleted_id),
            deleted_pack.to_stable_json()?,
        )?;

        let actual =
            diagnose_evidence_readiness(&setup, &EvidenceReadinessDiagnosticBudget::default());
        let expected = Some("deleted".to_string());

        assert_eq!(actual.worst_case_freshness, expected);
        Ok(())
    }

    #[test]
    fn evidence_readiness_reports_truncated_when_budgets_are_exceeded() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let (_id, pack) = write_fixture_context_pack(&setup, &manifest)?;
        let first_id = setup.context_pack_artifact_id(&pack)?;
        let second_id = ContextPackArtifactId::new("f".repeat(64))?;
        fs::write(
            artifact_path(fixture.path(), &second_id),
            pack.to_stable_json()?,
        )?;

        let actual = diagnose_evidence_readiness(
            &setup,
            &EvidenceReadinessDiagnosticBudget {
                max_artifacts: 1,
                max_episode_lines: 1,
                max_issue_summaries: 1,
            },
        );
        let expected = (1usize, true);

        assert_eq!(
            (actual.context_pack_artifact_count, actual.truncated),
            expected
        );
        assert_eq!(first_id.as_str().len(), 64usize);
        Ok(())
    }
    #[test]
    fn evidence_ledger_activation_builds_summary_from_valid_context_pack_and_linked_episode()
    -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let (artifact_id, _pack) = write_fixture_context_pack(&setup, &manifest)?;
        setup.append_episode(&fixture_episode(&artifact_id))?;

        let actual =
            load_evidence_ledger_activation(&setup, &EvidenceLedgerActivationBudget::default())?;
        let expected = (
            1usize, 1usize, 1usize, 1usize, 0usize, 2usize, 1usize, false,
        );

        assert_eq!(
            (
                actual.summary.context_pack_artifact_count,
                actual.summary.readable_context_pack_count,
                actual.summary.tool_episode_count,
                actual.summary.linked_episode_count,
                actual.summary.missing_link_count,
                actual.summary.graph_node_count,
                actual.summary.graph_edge_count,
                actual.summary.truncated,
            ),
            expected,
        );
        assert_eq!(
            actual.summary.worst_case_freshness,
            Some("fresh".to_string())
        );
        let actual_graph = actual
            .graph
            .as_ref()
            .expect("fixture activation should include graph metadata");
        assert_eq!(
            actual_graph.node_kind_counts.get("retrieved_evidence"),
            Some(&1)
        );
        assert_eq!(actual_graph.node_kind_counts.get("tool_episode"), Some(&1));
        Ok(())
    }

    #[test]
    fn evidence_ledger_activation_counts_malformed_inputs_without_raw_payload_echo() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let artifact_id = ContextPackArtifactId::new("b".repeat(64))?;
        fs::create_dir_all(fixture.path().join("model").join("context_packs"))?;
        fs::write(
            artifact_path(fixture.path(), &artifact_id),
            "not json secret payload",
        )?;
        fs::write(
            fixture.path().join("model").join("tool_episodes.jsonl"),
            "not json raw tool payload\n",
        )?;

        let actual =
            load_evidence_ledger_activation(&setup, &EvidenceLedgerActivationBudget::default())?;
        let actual_json = serde_json::to_string(&actual)?;
        let expected = (1usize, 0usize, 0usize, 2usize, true, true);

        assert_eq!(
            (
                actual.summary.context_pack_artifact_count,
                actual.summary.readable_context_pack_count,
                actual.summary.tool_episode_count,
                actual.summary.issue_count,
                actual
                    .summary
                    .issue_summaries
                    .contains(&"context_pack:CorruptArtifact".to_string()),
                actual
                    .summary
                    .issue_summaries
                    .contains(&"tool_episode_line_malformed".to_string()),
            ),
            expected,
        );
        assert!(!actual_json.contains("not json"));
        assert!(!actual_json.contains("secret payload"));
        assert!(!actual_json.contains("raw tool payload"));
        Ok(())
    }

    #[test]
    fn evidence_ledger_activation_does_not_count_corrupt_artifact_as_readable_link() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let artifact_id = ContextPackArtifactId::new("c".repeat(64))?;
        fs::create_dir_all(fixture.path().join("model").join("context_packs"))?;
        fs::write(
            artifact_path(fixture.path(), &artifact_id),
            "not json secret payload",
        )?;
        setup.append_episode(&fixture_episode(&artifact_id))?;

        let actual =
            load_evidence_ledger_activation(&setup, &EvidenceLedgerActivationBudget::default())?;
        let expected = (0usize, 0usize, 1usize, 0usize, 2usize, true, true);

        assert_eq!(
            (
                actual.summary.readable_context_pack_count,
                actual.summary.linked_episode_count,
                actual.summary.missing_link_count,
                actual.summary.graph_edge_count,
                actual.summary.issue_count,
                actual
                    .summary
                    .issue_summaries
                    .contains(&"context_pack:CorruptArtifact".to_string()),
                actual
                    .summary
                    .issue_summaries
                    .contains(&"episode_artifact_link:MissingLinkedArtifact".to_string()),
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn evidence_ledger_activation_graph_budget_omits_graph_and_preserves_summary() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let (artifact_id, _pack) = write_fixture_context_pack(&setup, &manifest)?;
        setup.append_episode(&fixture_episode(&artifact_id))?;

        let actual = load_evidence_ledger_activation(
            &setup,
            &EvidenceLedgerActivationBudget {
                max_artifacts: 8,
                max_episode_lines: 8,
                max_nodes: 1,
                max_edges: 8,
                max_issue_summaries: 8,
            },
        )?;
        let expected = (2usize, 1usize, true);

        assert_eq!(
            (
                actual.summary.graph_node_count,
                actual.summary.graph_edge_count,
                actual.summary.truncated,
            ),
            expected,
        );
        assert_eq!(actual.graph, None);
        Ok(())
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
    fn provenance_completeness_checks_static_cargo_metadata() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let mut manifest = setup.index()?;
        manifest
            .cargo_package_dependencies
            .first_mut()
            .expect("fixture should include Cargo dependency metadata")
            .provenance
            .fingerprint
            .clear();

        let actual = evaluate_provenance_completeness(&manifest).completeness < 1.0;
        let expected = true;
        assert_eq!(actual, expected);
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
