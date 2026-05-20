//! Bounded read-only evidence-ledger replay and selection surface.
//!
//! This module inspects only existing context-pack artifacts and existing
//! redaction-safe tool episodes. It never indexes, generates context packs,
//! records episodes, normalizes artifacts, repairs links, refreshes caches, or
//! writes diagnostics.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};

use crate::ProjectIndexer;
use crate::eval::tool_episode_graph_id;
use crate::types::{
    ContextPack, ContextPackArtifactId, ContextPackEvidenceSource, EvidenceFreshness,
    ProjectManifest, Provenance, ToolEpisode,
};

/// Manifest identity expected by a read-only evidence-ledger replay request.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceReplayManifestReference {
    /// Current manifest hash that context-pack artifacts must match exactly.
    pub manifest_hash: String,
}

/// Deterministic replay budget for read-only evidence-ledger selection.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceReplayBudget {
    /// Maximum artifact candidates to inspect after stable artifact ordering.
    pub max_artifacts: usize,
    /// Maximum tool-episode lines to inspect after JSONL ordering by file line.
    pub max_episode_lines: usize,
    /// Maximum selected evidence references to return.
    pub max_selected: usize,
}

impl Default for EvidenceReplayBudget {
    fn default() -> Self {
        Self { max_artifacts: 128, max_episode_lines: 512, max_selected: 32 }
    }
}

/// Content exposure policy for evidence-ledger replay.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceReplayContentPolicy {
    /// Return identifiers, paths, ranges, score kind, provenance, freshness, and
    /// linkage counters only.
    #[default]
    ReferenceOnly,
}

/// Stale changed/deleted evidence selection policy.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceReplayFreshnessPolicy {
    /// Select only fresh or added evidence. Changed and deleted evidence are
    /// excluded and reported with typed issue codes.
    #[default]
    ExcludeChangedAndDeleted,
    /// Allow changed evidence while still excluding deleted evidence.
    AllowChangedExcludeDeleted,
}

/// Artifact selection policy for readable context-pack artifacts.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceReplaySelectionPolicy {
    /// Prefer artifacts linked by valid tool episodes, then stable artifact and
    /// evidence identity tie-breakers.
    #[default]
    PreferLinkedReadable,
}

/// Small typed input for read-only evidence-ledger replay.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceLedgerReplayRequest {
    /// Manifest identity expected by the caller.
    pub manifest: EvidenceReplayManifestReference,
    /// Deterministic inspection and selection budget.
    pub budget: EvidenceReplayBudget,
    /// Selection ordering policy.
    pub selection_policy: EvidenceReplaySelectionPolicy,
    /// Freshness policy for stale changed/deleted evidence.
    pub freshness_policy: EvidenceReplayFreshnessPolicy,
    /// Content exposure policy. Defaults to reference-only.
    pub content_policy: EvidenceReplayContentPolicy,
}

impl EvidenceLedgerReplayRequest {
    /// Builds a reference-only replay request for the current manifest.
    ///
    /// # Arguments
    ///
    /// * `manifest` - Current manifest whose hash must match selected artifacts.
    pub fn reference_only(manifest: &ProjectManifest) -> Self {
        Self {
            manifest: EvidenceReplayManifestReference {
                manifest_hash: manifest.manifest_hash.clone(),
            },
            budget: EvidenceReplayBudget::default(),
            selection_policy: EvidenceReplaySelectionPolicy::PreferLinkedReadable,
            freshness_policy: EvidenceReplayFreshnessPolicy::ExcludeChangedAndDeleted,
            content_policy: EvidenceReplayContentPolicy::ReferenceOnly,
        }
    }
}

/// Stable score class used by replay references.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EvidenceReplayScoreKind {
    /// Score came from a retrieval result in a context-pack artifact.
    RetrievalResult,
    /// Score came from a structural shard in a context-pack artifact.
    Shard,
    /// Score came from caller-supplied direct evidence in a context-pack artifact.
    DirectEvidence,
}

impl From<&ContextPackEvidenceSource> for EvidenceReplayScoreKind {
    fn from(source: &ContextPackEvidenceSource) -> Self {
        match source {
            ContextPackEvidenceSource::RetrievalResult => Self::RetrievalResult,
            ContextPackEvidenceSource::Shard => Self::Shard,
            ContextPackEvidenceSource::DirectEvidence => Self::DirectEvidence,
        }
    }
}

/// Explicit stale evidence policy reported with replay results.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceReplayStalePolicyReport {
    /// Configured changed/deleted evidence policy.
    pub policy: EvidenceReplayFreshnessPolicy,
    /// Number of changed evidence references excluded by policy.
    pub changed_excluded: usize,
    /// Number of deleted evidence references excluded by policy.
    pub deleted_excluded: usize,
}

/// Typed issue code emitted by read-only evidence-ledger replay.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EvidenceReplayIssueCode {
    /// Context-pack store is absent; this is a typed empty success condition.
    EmptyStore,
    /// Artifact store metadata root exists but cannot be listed.
    StoreAccessFailure,
    /// Current manifest identity is unavailable or empty.
    ManifestUnavailable,
    /// Request manifest hash differs from the current manifest hash.
    ManifestMismatch,
    /// Artifact provenance is missing, unknown, or does not match current manifest hash.
    ArtifactProvenanceMismatch,
    /// Referenced context-pack artifact is missing.
    MissingArtifact,
    /// Context-pack artifact cannot be decoded or hash-validated.
    CorruptArtifact,
    /// Context-pack artifact file exists but cannot be read.
    UnreadableArtifact,
    /// Artifact path or filename is invalid.
    InvalidArtifactPath,
    /// Artifact path escapes the allowed metadata root.
    ArtifactPathEscape,
    /// Evidence source path is absolute, traverses parents, or escapes the project root.
    PathEscape,
    /// Evidence range is invalid for one-based inclusive line semantics.
    InvalidRange,
    /// Evidence points at a source file no longer present in the current manifest.
    DeletedEvidence,
    /// Evidence is changed relative to the current replay policy.
    StaleEvidenceChanged,
    /// Tool episode is malformed and cannot be decoded.
    CorruptEpisode,
    /// Tool episode JSONL file exists but cannot be read.
    UnreadableEpisodeStore,
    /// Tool episode does not contain a context-pack artifact link.
    UnlinkedEpisode,
    /// Tool episode links to an artifact that is absent or unreadable.
    DanglingEpisodeLink,
    /// Duplicate artifact or evidence identity was observed.
    Duplicate,
}

/// Redaction-safe typed issue emitted during replay selection.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceReplayIssue {
    /// Machine-readable issue code.
    pub code: EvidenceReplayIssueCode,
    /// Optional context-pack artifact identifier.
    pub artifact_id: Option<String>,
    /// Optional evidence identifier.
    pub evidence_id: Option<String>,
    /// Optional tool-episode fingerprint.
    pub episode_fingerprint: Option<String>,
    /// Redaction-safe path or storage label.
    pub path: Option<String>,
}

/// Reference-only selected evidence item.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvidenceReplayReference {
    /// Context-pack artifact identifier.
    pub artifact_id: String,
    /// Context-pack artifact path under the model metadata root.
    pub artifact_path: String,
    /// Evidence identifier inside the artifact.
    pub evidence_id: String,
    /// Evidence source path relative to the project root.
    pub evidence_path: String,
    /// Optional one-based inclusive start line.
    pub start_line: Option<u32>,
    /// Optional one-based inclusive end line.
    pub end_line: Option<u32>,
    /// Score class; raw trust floats are intentionally not exposed as authority.
    pub score_kind: EvidenceReplayScoreKind,
    /// Redaction-safe numeric priority within the score class.
    pub score: f32,
    /// Evidence provenance.
    pub provenance: Provenance,
    /// Evidence freshness state.
    pub freshness: EvidenceFreshness,
    /// Number of valid tool episodes linking this artifact.
    pub linked_episode_count: usize,
    /// Number of dangling or invalid linkage proofs seen for this artifact.
    pub link_issue_count: usize,
}

/// Deterministic/auditable budget summary for replay selection.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceReplayBudgetReport {
    /// Number of artifact candidates before inspection budget truncation.
    pub original_candidate_count: usize,
    /// Number of selected reference-only evidence items.
    pub selected_count: usize,
    /// Total excluded candidate/evidence/link count.
    pub excluded_count: usize,
    /// Excluded counts keyed by typed issue code label.
    pub excluded_by_reason: BTreeMap<EvidenceReplayIssueCode, usize>,
    /// Whether any artifact, episode, or selected output was truncated by budget.
    pub truncated: bool,
    /// Budget applied to this replay run.
    pub budget: EvidenceReplayBudget,
    /// Stable ordering and tie-break contract used by this replay run.
    pub stable_ordering: String,
}

/// Read-only evidence-ledger replay result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvidenceLedgerReplayReport {
    /// Manifest hash used for compatibility checks.
    pub manifest_hash: String,
    /// Content exposure policy actually applied.
    pub content_policy: EvidenceReplayContentPolicy,
    /// Explicit stale evidence policy and stale exclusion counters.
    pub stale_policy: EvidenceReplayStalePolicyReport,
    /// Reference-only selected evidence items.
    pub selected: Vec<EvidenceReplayReference>,
    /// Typed redaction-safe issues.
    pub issues: Vec<EvidenceReplayIssue>,
    /// Deterministic budget and selection audit.
    pub budget: EvidenceReplayBudgetReport,
}

/// Selects bounded reference-only evidence from existing evidence-ledger artifacts.
///
/// This is a read-only replay/selection surface. It reads existing context-pack
/// artifacts and existing tool episodes, validates manifest compatibility and
/// path boundaries, and returns reference-only evidence selections plus typed
/// issues. It performs no writes, repairs, indexing, cache refreshes, artifact
/// normalization, episode backfill, or schema migration.
///
/// # Arguments
///
/// * `indexer` - Project indexer whose existing model storage is inspected.
/// * `current_manifest` - Current manifest identity and source boundary.
/// * `request` - Replay budget, manifest reference, and selection policies.
pub fn select_evidence_ledger_replay(
    indexer: &ProjectIndexer,
    current_manifest: &ProjectManifest,
    request: &EvidenceLedgerReplayRequest,
) -> EvidenceLedgerReplayReport {
    let mut builder = ReplayBuilder::new(current_manifest, request);
    if current_manifest.manifest_hash.is_empty() {
        builder.push_issue(
            EvidenceReplayIssueCode::ManifestUnavailable,
            None,
            None,
            None,
            None,
        );
        return builder.finish(Vec::new(), 0, false);
    }
    if request.manifest.manifest_hash != current_manifest.manifest_hash {
        builder.push_issue(
            EvidenceReplayIssueCode::ManifestMismatch,
            None,
            None,
            None,
            None,
        );
        return builder.finish(Vec::new(), 0, false);
    }

    let artifact_candidates = list_artifact_candidates(indexer.model_dir(), request, &mut builder);
    let original_candidate_count = artifact_candidates.original_count;
    let episode_links = read_episode_links(indexer.model_dir(), request, &mut builder);
    let mut readable_artifact_ids = BTreeSet::new();
    let mut duplicate_artifact_ids = BTreeSet::new();
    let mut candidate_refs = Vec::new();

    for candidate in artifact_candidates.paths {
        if !readable_artifact_ids.insert(candidate.id.as_str().to_string()) {
            duplicate_artifact_ids.insert(candidate.id.as_str().to_string());
            builder.push_issue(
                EvidenceReplayIssueCode::Duplicate,
                Some(candidate.id.as_str()),
                None,
                None,
                Some(candidate.storage_path.as_str()),
            );
            continue;
        }
        match indexer.read_context_pack(&candidate.id) {
            Ok(pack) => {
                let mut collector = PackReferenceCollector {
                    indexer,
                    current_manifest,
                    request,
                    builder: &mut builder,
                    episode_links: &episode_links,
                    candidate: &candidate,
                    pack: &pack,
                };
                collector.collect_into(&mut candidate_refs);
            }
            Err(error) => {
                let code = if error_is_not_found(&error) {
                    EvidenceReplayIssueCode::MissingArtifact
                } else if error_is_unreadable(&error) {
                    EvidenceReplayIssueCode::UnreadableArtifact
                } else {
                    EvidenceReplayIssueCode::CorruptArtifact
                };
                builder.push_issue(
                    code,
                    Some(candidate.id.as_str()),
                    None,
                    None,
                    Some(candidate.storage_path.as_str()),
                );
            }
        }
    }

    for (artifact_id, episode_fingerprints) in &episode_links.by_artifact_id {
        if !readable_artifact_ids.contains(artifact_id) {
            for episode_fingerprint in episode_fingerprints {
                builder.push_issue(
                    EvidenceReplayIssueCode::DanglingEpisodeLink,
                    Some(artifact_id),
                    None,
                    Some(episode_fingerprint.as_str()),
                    Some(&format!("context_packs/{artifact_id}.json")),
                );
            }
        }
    }

    candidate_refs.sort_by(compare_reference_candidates);
    let truncated_by_selection = candidate_refs.len() > request.budget.max_selected;
    let selected = candidate_refs
        .into_iter()
        .take(request.budget.max_selected)
        .map(|candidate| candidate.reference)
        .collect::<Vec<_>>();
    let truncated =
        artifact_candidates.truncated || episode_links.truncated || truncated_by_selection;
    builder.finish(selected, original_candidate_count, truncated)
}

struct ReplayBuilder<'a> {
    current_manifest: &'a ProjectManifest,
    request: &'a EvidenceLedgerReplayRequest,
    issues: Vec<EvidenceReplayIssue>,
    excluded_by_reason: BTreeMap<EvidenceReplayIssueCode, usize>,
    changed_excluded: usize,
    deleted_excluded: usize,
}

impl<'a> ReplayBuilder<'a> {
    fn new(
        current_manifest: &'a ProjectManifest,
        request: &'a EvidenceLedgerReplayRequest,
    ) -> Self {
        Self {
            current_manifest,
            request,
            issues: Vec::new(),
            excluded_by_reason: BTreeMap::new(),
            changed_excluded: 0,
            deleted_excluded: 0,
        }
    }

    fn push_issue(
        &mut self,
        code: EvidenceReplayIssueCode,
        artifact_id: Option<&str>,
        evidence_id: Option<&str>,
        episode_fingerprint: Option<&str>,
        path: Option<&str>,
    ) {
        let count = self.excluded_by_reason.entry(code.clone()).or_default();
        *count = count.saturating_add(1);
        if code == EvidenceReplayIssueCode::StaleEvidenceChanged {
            self.changed_excluded = self.changed_excluded.saturating_add(1);
        }
        if code == EvidenceReplayIssueCode::DeletedEvidence {
            self.deleted_excluded = self.deleted_excluded.saturating_add(1);
        }
        self.issues.push(EvidenceReplayIssue {
            code,
            artifact_id: artifact_id.map(ToString::to_string),
            evidence_id: evidence_id.map(ToString::to_string),
            episode_fingerprint: episode_fingerprint.map(ToString::to_string),
            path: path.map(ToString::to_string),
        });
    }

    fn finish(
        self,
        selected: Vec<EvidenceReplayReference>,
        original_candidate_count: usize,
        truncated: bool,
    ) -> EvidenceLedgerReplayReport {
        let selected_count = selected.len();
        let excluded_count = self.excluded_by_reason.values().copied().sum();
        EvidenceLedgerReplayReport {
            manifest_hash: self.current_manifest.manifest_hash.clone(),
            content_policy: self.request.content_policy.clone(),
            stale_policy: EvidenceReplayStalePolicyReport {
                policy: self.request.freshness_policy.clone(),
                changed_excluded: self.changed_excluded,
                deleted_excluded: self.deleted_excluded,
            },
            selected,
            issues: self.issues,
            budget: EvidenceReplayBudgetReport {
                original_candidate_count,
                selected_count,
                excluded_count,
                excluded_by_reason: self.excluded_by_reason,
                truncated,
                budget: self.request.budget.clone(),
                stable_ordering: "linked_episode_count_desc:score_kind:score_desc:freshness:path:evidence_id:artifact_id".to_string(),
            },
        }
    }
}

struct ArtifactCandidates {
    paths: Vec<ArtifactCandidate>,
    original_count: usize,
    truncated: bool,
}

struct ArtifactCandidate {
    id: ContextPackArtifactId,
    storage_path: String,
}

fn list_artifact_candidates(
    model_dir: &Path,
    request: &EvidenceLedgerReplayRequest,
    builder: &mut ReplayBuilder<'_>,
) -> ArtifactCandidates {
    let directory = model_dir.join("context_packs");
    let Ok(entries) = fs::read_dir(&directory) else {
        if directory.exists() {
            builder.push_issue(
                EvidenceReplayIssueCode::StoreAccessFailure,
                None,
                None,
                None,
                Some("context_packs"),
            );
        } else {
            builder.push_issue(
                EvidenceReplayIssueCode::EmptyStore,
                None,
                None,
                None,
                Some("context_packs"),
            );
        }
        return ArtifactCandidates { paths: Vec::new(), original_count: 0, truncated: false };
    };
    let mut paths = Vec::new();
    for entry in entries {
        let Ok(entry) = entry else {
            builder.push_issue(
                EvidenceReplayIssueCode::StoreAccessFailure,
                None,
                None,
                None,
                Some("context_packs"),
            );
            continue;
        };
        let path = entry.path();
        let storage_path = match metadata_relative_path(model_dir, &path) {
            Some(value) => value,
            None => {
                builder.push_issue(
                    EvidenceReplayIssueCode::ArtifactPathEscape,
                    None,
                    None,
                    None,
                    None,
                );
                continue;
            }
        };
        if !artifact_path_stays_inside_store(&directory, &path) {
            builder.push_issue(
                EvidenceReplayIssueCode::ArtifactPathEscape,
                None,
                None,
                None,
                Some(storage_path.as_str()),
            );
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|value| value.to_str()) else {
            builder.push_issue(
                EvidenceReplayIssueCode::InvalidArtifactPath,
                None,
                None,
                None,
                Some(storage_path.as_str()),
            );
            continue;
        };
        let Some(raw_id) = file_name.strip_suffix(".json") else {
            continue;
        };
        let Ok(id) = ContextPackArtifactId::new(raw_id.to_string()) else {
            builder.push_issue(
                EvidenceReplayIssueCode::InvalidArtifactPath,
                None,
                None,
                None,
                Some(storage_path.as_str()),
            );
            continue;
        };
        paths.push(ArtifactCandidate { id, storage_path });
    }
    paths.sort_by(|left, right| left.id.cmp(&right.id));
    let original_count = paths.len();
    let truncated = paths.len() > request.budget.max_artifacts;
    paths.truncate(request.budget.max_artifacts);
    ArtifactCandidates { paths, original_count, truncated }
}

#[derive(Default)]
struct EpisodeLinks {
    by_artifact_id: BTreeMap<String, BTreeSet<String>>,
    invalid_link_count_by_artifact_id: BTreeMap<String, usize>,
    truncated: bool,
}

fn read_episode_links(
    model_dir: &Path,
    request: &EvidenceLedgerReplayRequest,
    builder: &mut ReplayBuilder<'_>,
) -> EpisodeLinks {
    let path = model_dir.join("tool_episodes.jsonl");
    let Ok(file) = File::open(&path) else {
        if path.exists() {
            builder.push_issue(
                EvidenceReplayIssueCode::UnreadableEpisodeStore,
                None,
                None,
                None,
                Some("tool_episodes.jsonl"),
            );
        }
        return EpisodeLinks::default();
    };
    let mut links = EpisodeLinks::default();
    let mut inspected = 0usize;
    for line in BufReader::new(file).lines() {
        if inspected >= request.budget.max_episode_lines {
            links.truncated = true;
            break;
        }
        inspected = inspected.saturating_add(1);
        let line = match line {
            Ok(line) => line,
            Err(_) => {
                builder.push_issue(
                    EvidenceReplayIssueCode::UnreadableEpisodeStore,
                    None,
                    None,
                    None,
                    Some("tool_episodes.jsonl"),
                );
                continue;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let episode = match serde_json::from_str::<ToolEpisode>(&line) {
            Ok(episode) => episode,
            Err(_) => {
                builder.push_issue(
                    EvidenceReplayIssueCode::CorruptEpisode,
                    None,
                    None,
                    None,
                    Some("tool_episodes.jsonl"),
                );
                continue;
            }
        };
        let episode_fingerprint = tool_episode_graph_id(&episode);
        let Some(artifact_id) = episode_context_pack_artifact_id(&episode) else {
            builder.push_issue(
                EvidenceReplayIssueCode::UnlinkedEpisode,
                None,
                None,
                Some(episode_fingerprint.as_str()),
                Some(episode.provenance.path.as_str()),
            );
            continue;
        };
        let artifact_id = artifact_id.as_str().to_string();
        let fingerprints = links.by_artifact_id.entry(artifact_id.clone()).or_default();
        if !fingerprints.insert(episode_fingerprint.clone()) {
            builder.push_issue(
                EvidenceReplayIssueCode::Duplicate,
                Some(artifact_id.as_str()),
                None,
                Some(episode_fingerprint.as_str()),
                Some(episode.provenance.path.as_str()),
            );
            continue;
        }
    }
    links
}

struct PackReferenceCollector<'a, 'b> {
    indexer: &'a ProjectIndexer,
    current_manifest: &'a ProjectManifest,
    request: &'a EvidenceLedgerReplayRequest,
    builder: &'a mut ReplayBuilder<'b>,
    episode_links: &'a EpisodeLinks,
    candidate: &'a ArtifactCandidate,
    pack: &'a ContextPack,
}

impl PackReferenceCollector<'_, '_> {
    fn collect_into(&mut self, candidate_refs: &mut Vec<ReferenceCandidate>) {
        if self.pack.manifest_hash != self.current_manifest.manifest_hash {
            self.builder.push_issue(
                EvidenceReplayIssueCode::ArtifactProvenanceMismatch,
                Some(self.candidate.id.as_str()),
                None,
                None,
                Some(self.candidate.storage_path.as_str()),
            );
            return;
        }
        let linked_episode_count = self
            .episode_links
            .by_artifact_id
            .get(self.candidate.id.as_str())
            .map(BTreeSet::len)
            .unwrap_or_default();
        let link_issue_count = self
            .episode_links
            .invalid_link_count_by_artifact_id
            .get(self.candidate.id.as_str())
            .copied()
            .unwrap_or_default();
        let mut seen_evidence = BTreeSet::new();
        for evidence in &self.pack.evidence {
            if !seen_evidence.insert(evidence.id.clone()) {
                self.push_evidence_issue(EvidenceReplayIssueCode::Duplicate, evidence);
                continue;
            }
            if validate_source_path(self.indexer.root(), &evidence.path).is_err()
                || validate_source_path(self.indexer.root(), &evidence.provenance.path).is_err()
            {
                self.push_evidence_issue(EvidenceReplayIssueCode::PathEscape, evidence);
                continue;
            }
            let Some(source_file) = self
                .current_manifest
                .files
                .iter()
                .find(|file| file.path == evidence.path)
            else {
                self.push_evidence_issue(EvidenceReplayIssueCode::DeletedEvidence, evidence);
                continue;
            };
            let Some(provenance_file) = self
                .current_manifest
                .files
                .iter()
                .find(|file| file.path == evidence.provenance.path)
            else {
                self.push_evidence_issue(EvidenceReplayIssueCode::DeletedEvidence, evidence);
                continue;
            };
            if invalid_range(
                evidence.provenance.start_line,
                evidence.provenance.end_line,
                source_file.lines,
            ) || invalid_range(
                evidence.provenance.start_line,
                evidence.provenance.end_line,
                provenance_file.lines,
            ) {
                self.push_evidence_issue(EvidenceReplayIssueCode::InvalidRange, evidence);
                continue;
            }
            if evidence.freshness == EvidenceFreshness::Deleted {
                self.push_evidence_issue(EvidenceReplayIssueCode::DeletedEvidence, evidence);
                continue;
            }
            if evidence.freshness == EvidenceFreshness::Changed
                && self.request.freshness_policy
                    == EvidenceReplayFreshnessPolicy::ExcludeChangedAndDeleted
            {
                self.push_evidence_issue(EvidenceReplayIssueCode::StaleEvidenceChanged, evidence);
                continue;
            }
            if !evidence.score.is_finite() {
                self.push_evidence_issue(EvidenceReplayIssueCode::CorruptArtifact, evidence);
                continue;
            }
            candidate_refs.push(ReferenceCandidate {
                linked_episode_count,
                score_kind: EvidenceReplayScoreKind::from(&evidence.source),
                score: evidence.score,
                reference: EvidenceReplayReference {
                    artifact_id: self.candidate.id.as_str().to_string(),
                    artifact_path: self.candidate.storage_path.clone(),
                    evidence_id: evidence.id.clone(),
                    evidence_path: evidence.path.clone(),
                    start_line: evidence.provenance.start_line,
                    end_line: evidence.provenance.end_line,
                    score_kind: EvidenceReplayScoreKind::from(&evidence.source),
                    score: evidence.score,
                    provenance: evidence.provenance.clone(),
                    freshness: evidence.freshness.clone(),
                    linked_episode_count,
                    link_issue_count,
                },
            });
        }
    }

    fn push_evidence_issue(
        &mut self,
        code: EvidenceReplayIssueCode,
        evidence: &crate::types::ContextPackEvidence,
    ) {
        self.builder.push_issue(
            code,
            Some(self.candidate.id.as_str()),
            Some(evidence.id.as_str()),
            None,
            Some(evidence.path.as_str()),
        );
    }
}

struct ReferenceCandidate {
    linked_episode_count: usize,
    score_kind: EvidenceReplayScoreKind,
    score: f32,
    reference: EvidenceReplayReference,
}

fn compare_reference_candidates(
    left: &ReferenceCandidate,
    right: &ReferenceCandidate,
) -> std::cmp::Ordering {
    right
        .linked_episode_count
        .cmp(&left.linked_episode_count)
        .then_with(|| left.score_kind.cmp(&right.score_kind))
        .then_with(|| right.score.total_cmp(&left.score))
        .then_with(|| left.reference.freshness.cmp(&right.reference.freshness))
        .then_with(|| {
            left.reference
                .evidence_path
                .cmp(&right.reference.evidence_path)
        })
        .then_with(|| left.reference.evidence_id.cmp(&right.reference.evidence_id))
        .then_with(|| left.reference.artifact_id.cmp(&right.reference.artifact_id))
}

fn metadata_relative_path(root: &Path, path: &Path) -> Option<String> {
    let relative = path.strip_prefix(root).ok()?;
    if relative
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return None;
    }
    Some(relative.to_string_lossy().replace('\\', "/"))
}

fn artifact_path_stays_inside_store(directory: &Path, path: &Path) -> bool {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return false;
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return false;
    }
    let Ok(canonical_directory) = directory.canonicalize() else {
        return false;
    };
    let Ok(canonical_path) = path.canonicalize() else {
        return false;
    };
    canonical_path.starts_with(canonical_directory)
}

fn validate_source_path(root: &Path, relative: &str) -> anyhow::Result<()> {
    let candidate = Path::new(relative);
    if relative.is_empty() || candidate.is_absolute() {
        anyhow::bail!("unsafe source path");
    }
    for component in candidate.components() {
        if !matches!(component, Component::Normal(_)) {
            anyhow::bail!("unsafe source path");
        }
    }
    let joined = root.join(candidate);
    let canonical_root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    if let Ok(canonical_path) = joined.canonicalize()
        && !canonical_path.starts_with(canonical_root)
    {
        anyhow::bail!("source path escapes root");
    }
    Ok(())
}

fn invalid_range(start_line: Option<u32>, end_line: Option<u32>, file_lines: u32) -> bool {
    match (start_line, end_line) {
        (Some(0), _) | (_, Some(0)) => true,
        (Some(start), Some(end)) => end < start || start > file_lines || end > file_lines,
        (Some(start), None) => start > file_lines,
        (None, Some(_)) => true,
        _ => false,
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

fn error_is_not_found(error: &anyhow::Error) -> bool {
    error
        .chain()
        .filter_map(|cause| cause.downcast_ref::<std::io::Error>())
        .any(|cause| cause.kind() == std::io::ErrorKind::NotFound)
}

fn error_is_unreadable(error: &anyhow::Error) -> bool {
    error
        .chain()
        .filter_map(|cause| cause.downcast_ref::<std::io::Error>())
        .any(|cause| matches!(cause.kind(), std::io::ErrorKind::PermissionDenied))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::SystemTime;

    use anyhow::Result;
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::indexer::tests::fixture_project;
    use crate::{
        ContextPackSelection, FreshnessState, RetrievalQuery, StaleEvidencePolicy, fingerprint,
        retrieve,
    };

    fn write_fixture_context_pack(
        indexer: &ProjectIndexer,
        manifest: &ProjectManifest,
        freshness: EvidenceFreshness,
        score: f32,
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
        let mut pack = ContextPack::from_selection(
            manifest,
            ContextPackSelection {
                retrieval_results: vec![result],
                shards: Vec::new(),
                evidence: Vec::new(),
                freshness: FreshnessState { fresh: true, ..Default::default() },
                stale_policy: StaleEvidencePolicy::Mark,
            },
        )?;
        let evidence = pack
            .evidence
            .first_mut()
            .expect("fixture context pack should include evidence");
        evidence.freshness = freshness;
        evidence.score = score;
        indexer.write_context_pack(&pack)?;
        let id = indexer.context_pack_artifact_id(&pack)?;
        Ok((id, pack))
    }

    fn fixture_request(manifest: &ProjectManifest) -> EvidenceLedgerReplayRequest {
        EvidenceLedgerReplayRequest::reference_only(manifest)
    }

    fn fixture_episode(artifact_id: &ContextPackArtifactId) -> ToolEpisode {
        ToolEpisode {
            timestamp: "2026-05-20T03:00:00+03:00".to_string(),
            tool: "project_model_replay".to_string(),
            input_fingerprint: fingerprint("input"),
            output_fingerprint: fingerprint("output"),
            status: "success".to_string(),
            provenance: Provenance {
                path: format!("context_packs/{}.json", artifact_id.as_str()),
                start_line: None,
                end_line: None,
                source: "test".to_string(),
                fingerprint: fingerprint("episode"),
            },
        }
    }

    fn artifact_path(model_root: &Path, artifact_id: &ContextPackArtifactId) -> PathBuf {
        model_root
            .join("model")
            .join("context_packs")
            .join(format!("{}.json", artifact_id.as_str()))
    }

    fn issue_codes(report: &EvidenceLedgerReplayReport) -> BTreeSet<EvidenceReplayIssueCode> {
        report
            .issues
            .iter()
            .map(|issue| issue.code.clone())
            .collect()
    }

    #[test]
    fn replay_empty_store_returns_typed_empty_success_without_creating_model_dir() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let indexer = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = indexer.index()?;
        let setup = fixture_request(&manifest);

        let actual = select_evidence_ledger_replay(&indexer, &manifest, &setup);
        let expected = (
            0usize,
            0usize,
            Some(EvidenceReplayIssueCode::EmptyStore),
            false,
        );

        assert_eq!(
            (
                actual.budget.original_candidate_count,
                actual.selected.len(),
                actual.issues.first().map(|issue| issue.code.clone()),
                fixture.path().join("model").exists(),
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn replay_excludes_corrupt_artifact_and_reports_corrupt_issue() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let indexer = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = indexer.index()?;
        let artifact_id = ContextPackArtifactId::new("b".repeat(64))?;
        fs::create_dir_all(fixture.path().join("model").join("context_packs"))?;
        fs::write(artifact_path(fixture.path(), &artifact_id), "not json")?;

        let actual =
            select_evidence_ledger_replay(&indexer, &manifest, &fixture_request(&manifest));
        let expected = (0usize, true);

        assert_eq!(
            (
                actual.selected.len(),
                actual
                    .issues
                    .iter()
                    .any(|issue| issue.code == EvidenceReplayIssueCode::CorruptArtifact),
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn replay_reports_missing_artifact_separately_from_corrupt_when_episode_dangles() -> Result<()>
    {
        let (fixture, root) = fixture_project()?;
        let indexer = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = indexer.index()?;
        let missing_id = ContextPackArtifactId::new("a".repeat(64))?;
        let corrupt_id = ContextPackArtifactId::new("b".repeat(64))?;
        fs::create_dir_all(fixture.path().join("model").join("context_packs"))?;
        fs::write(artifact_path(fixture.path(), &corrupt_id), "not json")?;
        indexer.append_episode(&fixture_episode(&missing_id))?;

        let actual =
            select_evidence_ledger_replay(&indexer, &manifest, &fixture_request(&manifest));
        let expected = BTreeSet::from([
            EvidenceReplayIssueCode::CorruptArtifact,
            EvidenceReplayIssueCode::DanglingEpisodeLink,
        ]);

        assert_eq!(issue_codes(&actual), expected);
        Ok(())
    }

    #[test]
    fn replay_selects_linked_readable_evidence_ahead_of_unlinked_readable() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let indexer = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = indexer.index()?;
        let (_unlinked_id, _unlinked_pack) =
            write_fixture_context_pack(&indexer, &manifest, EvidenceFreshness::Fresh, 100.0)?;
        let (linked_id, _linked_pack) =
            write_fixture_context_pack(&indexer, &manifest, EvidenceFreshness::Fresh, 1.0)?;
        indexer.append_episode(&fixture_episode(&linked_id))?;

        let actual =
            select_evidence_ledger_replay(&indexer, &manifest, &fixture_request(&manifest));
        let expected = Some(linked_id.as_str().to_string());

        assert_eq!(
            actual.selected.first().map(|item| item.artifact_id.clone()),
            expected
        );
        Ok(())
    }

    #[test]
    fn replay_deduplicates_repeated_episode_links_before_scoring() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let indexer = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = indexer.index()?;
        let (linked_id, _linked_pack) =
            write_fixture_context_pack(&indexer, &manifest, EvidenceFreshness::Fresh, 1.0)?;
        let setup = fixture_episode(&linked_id);
        indexer.append_episode(&setup)?;
        indexer.append_episode(&setup)?;

        let actual =
            select_evidence_ledger_replay(&indexer, &manifest, &fixture_request(&manifest));
        let expected = (Some(1usize), 1usize, true);

        assert_eq!(
            (
                actual
                    .selected
                    .first()
                    .map(|reference| reference.linked_episode_count),
                actual
                    .issues
                    .iter()
                    .filter(|issue| issue.code == EvidenceReplayIssueCode::Duplicate)
                    .count(),
                actual.issues.iter().any(|issue| {
                    issue.code == EvidenceReplayIssueCode::Duplicate
                        && issue.artifact_id.as_deref() == Some(linked_id.as_str())
                        && issue.episode_fingerprint.as_deref()
                            == Some(tool_episode_graph_id(&setup).as_str())
                }),
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn replay_rejects_artifact_manifest_mismatch_without_selection() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let indexer = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = indexer.index()?;
        let (original_id, mut pack) =
            write_fixture_context_pack(&indexer, &manifest, EvidenceFreshness::Fresh, 1.0)?;
        fs::remove_file(artifact_path(fixture.path(), &original_id))?;
        pack.manifest_hash = "different".to_string();
        let mismatched_id = indexer.context_pack_artifact_id(&pack)?;
        fs::write(
            artifact_path(fixture.path(), &mismatched_id),
            pack.to_stable_json()?,
        )?;

        let actual =
            select_evidence_ledger_replay(&indexer, &manifest, &fixture_request(&manifest));
        let expected = (0usize, true);

        assert_eq!(
            (
                actual.selected.len(),
                actual
                    .issues
                    .iter()
                    .any(|issue| issue.code == EvidenceReplayIssueCode::ArtifactProvenanceMismatch),
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn replay_reports_deleted_and_changed_source_policy_explicitly() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let indexer = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = indexer.index()?;
        let (_changed_id, _changed_pack) =
            write_fixture_context_pack(&indexer, &manifest, EvidenceFreshness::Changed, 1.0)?;
        let (original_deleted_id, mut deleted_pack) =
            write_fixture_context_pack(&indexer, &manifest, EvidenceFreshness::Deleted, 2.0)?;
        fs::remove_file(artifact_path(fixture.path(), &original_deleted_id))?;
        deleted_pack
            .evidence
            .first_mut()
            .expect("fixture evidence should exist")
            .path = "src/deleted.rs".to_string();
        let deleted_id = indexer.context_pack_artifact_id(&deleted_pack)?;
        fs::write(
            artifact_path(fixture.path(), &deleted_id),
            deleted_pack.to_stable_json()?,
        )?;

        let actual =
            select_evidence_ledger_replay(&indexer, &manifest, &fixture_request(&manifest));
        let expected = (1usize, 1usize, true, true);

        assert_eq!(
            (
                actual.stale_policy.changed_excluded,
                actual.stale_policy.deleted_excluded,
                actual
                    .issues
                    .iter()
                    .any(|issue| issue.code == EvidenceReplayIssueCode::StaleEvidenceChanged),
                actual
                    .issues
                    .iter()
                    .any(|issue| issue.code == EvidenceReplayIssueCode::DeletedEvidence),
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn replay_invalid_episode_link_reports_issue_without_raw_payload() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let indexer = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = indexer.index()?;
        fs::create_dir_all(fixture.path().join("model"))?;
        fs::write(
            fixture.path().join("model").join("tool_episodes.jsonl"),
            "raw secret payload not json\n",
        )?;

        let actual =
            select_evidence_ledger_replay(&indexer, &manifest, &fixture_request(&manifest));
        let actual_json = serde_json::to_string(&actual)?;
        let expected = true;

        assert_eq!(
            actual
                .issues
                .iter()
                .any(|issue| issue.code == EvidenceReplayIssueCode::CorruptEpisode),
            expected
        );
        assert!(!actual_json.contains("raw secret payload"));
        Ok(())
    }

    #[test]
    fn replay_budget_is_deterministic_and_reports_truncation() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let indexer = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = indexer.index()?;
        write_fixture_context_pack(&indexer, &manifest, EvidenceFreshness::Fresh, 1.0)?;
        write_fixture_context_pack(&indexer, &manifest, EvidenceFreshness::Fresh, 2.0)?;
        let mut request = fixture_request(&manifest);
        request.budget.max_artifacts = 8;
        request.budget.max_selected = 1;

        let actual = select_evidence_ledger_replay(&indexer, &manifest, &request);
        let expected = (1usize, true, "linked_episode_count_desc:score_kind:score_desc:freshness:path:evidence_id:artifact_id".to_string());

        assert_eq!(
            (
                actual.budget.selected_count,
                actual.budget.truncated,
                actual.budget.stable_ordering,
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn replay_does_not_write_or_touch_existing_files() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let indexer = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = indexer.index()?;
        let (artifact_id, _pack) =
            write_fixture_context_pack(&indexer, &manifest, EvidenceFreshness::Fresh, 1.0)?;
        let path = artifact_path(fixture.path(), &artifact_id);
        let setup = fs::metadata(&path)?
            .modified()
            .unwrap_or(SystemTime::UNIX_EPOCH);

        let _actual =
            select_evidence_ledger_replay(&indexer, &manifest, &fixture_request(&manifest));
        let expected = setup;

        assert_eq!(
            fs::metadata(&path)?
                .modified()
                .unwrap_or(SystemTime::UNIX_EPOCH),
            expected
        );
        Ok(())
    }

    #[test]
    fn replay_output_is_reference_only_without_source_or_episode_payloads() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let indexer = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = indexer.index()?;
        let (artifact_id, _pack) =
            write_fixture_context_pack(&indexer, &manifest, EvidenceFreshness::Fresh, 1.0)?;
        indexer.append_episode(&fixture_episode(&artifact_id))?;

        let actual =
            select_evidence_ledger_replay(&indexer, &manifest, &fixture_request(&manifest));
        let actual_json = serde_json::to_string(&actual)?;
        let expected = EvidenceReplayContentPolicy::ReferenceOnly;

        assert_eq!(actual.content_policy, expected);
        assert!(!actual_json.contains("pub struct Root"));
        assert!(!actual_json.contains("input_fingerprint"));
        assert!(!actual_json.contains("output_fingerprint"));
        Ok(())
    }

    #[test]
    fn replay_ordering_is_stable_for_equal_candidates() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let indexer = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = indexer.index()?;
        write_fixture_context_pack(&indexer, &manifest, EvidenceFreshness::Fresh, 1.0)?;
        write_fixture_context_pack(&indexer, &manifest, EvidenceFreshness::Fresh, 1.0)?;
        let setup = fixture_request(&manifest);

        let actual = select_evidence_ledger_replay(&indexer, &manifest, &setup);
        let expected = select_evidence_ledger_replay(&indexer, &manifest, &setup);

        assert_eq!(actual.selected, expected.selected);
        Ok(())
    }

    #[test]
    fn replay_rejects_path_boundary_escape() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let indexer = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = indexer.index()?;
        let (original_id, mut pack) =
            write_fixture_context_pack(&indexer, &manifest, EvidenceFreshness::Fresh, 1.0)?;
        fs::remove_file(artifact_path(fixture.path(), &original_id))?;
        let evidence = pack
            .evidence
            .first_mut()
            .expect("fixture evidence should exist");
        evidence.path = "../escape.rs".to_string();
        evidence.provenance.path = "../escape.rs".to_string();
        let escape_id = indexer.context_pack_artifact_id(&pack)?;
        fs::write(
            artifact_path(fixture.path(), &escape_id),
            pack.to_stable_json()?,
        )?;

        let actual =
            select_evidence_ledger_replay(&indexer, &manifest, &fixture_request(&manifest));
        let expected = (0usize, true);

        assert_eq!(
            (
                actual.selected.len(),
                actual
                    .issues
                    .iter()
                    .any(|issue| issue.code == EvidenceReplayIssueCode::PathEscape),
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn replay_rejects_range_beyond_current_manifest_file_lines() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let indexer = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = indexer.index()?;
        let (original_id, mut pack) =
            write_fixture_context_pack(&indexer, &manifest, EvidenceFreshness::Fresh, 1.0)?;
        fs::remove_file(artifact_path(fixture.path(), &original_id))?;
        let evidence = pack
            .evidence
            .first_mut()
            .expect("fixture evidence should exist");
        evidence.provenance.end_line = Some(u32::MAX);
        let out_of_bounds_id = indexer.context_pack_artifact_id(&pack)?;
        fs::write(
            artifact_path(fixture.path(), &out_of_bounds_id),
            pack.to_stable_json()?,
        )?;

        let actual =
            select_evidence_ledger_replay(&indexer, &manifest, &fixture_request(&manifest));
        let expected = (0usize, true);

        assert_eq!(
            (
                actual.selected.len(),
                actual
                    .issues
                    .iter()
                    .any(|issue| issue.code == EvidenceReplayIssueCode::InvalidRange),
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn replay_rejects_range_out_of_bounds_for_distinct_provenance_path() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        fs::write(
            root.join("src").join("long.rs"),
            "pub struct Long;\n".repeat(64),
        )?;
        let indexer = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = indexer.index()?;
        let (original_id, mut pack) =
            write_fixture_context_pack(&indexer, &manifest, EvidenceFreshness::Fresh, 1.0)?;
        fs::remove_file(artifact_path(fixture.path(), &original_id))?;
        let evidence = pack
            .evidence
            .first_mut()
            .expect("fixture evidence should exist");
        evidence.path = "src/long.rs".to_string();
        evidence.provenance.path = "src/model.rs".to_string();
        evidence.provenance.start_line = Some(32);
        evidence.provenance.end_line = Some(32);
        let out_of_bounds_id = indexer.context_pack_artifact_id(&pack)?;
        fs::write(
            artifact_path(fixture.path(), &out_of_bounds_id),
            pack.to_stable_json()?,
        )?;

        let actual =
            select_evidence_ledger_replay(&indexer, &manifest, &fixture_request(&manifest));
        let expected = (0usize, true);

        assert_eq!(
            (
                actual.selected.len(),
                actual
                    .issues
                    .iter()
                    .any(|issue| issue.code == EvidenceReplayIssueCode::InvalidRange),
            ),
            expected,
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn replay_rejects_context_pack_symlink_escape_from_metadata_root() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let indexer = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = indexer.index()?;
        let (artifact_id, pack) =
            write_fixture_context_pack(&indexer, &manifest, EvidenceFreshness::Fresh, 1.0)?;
        let internal_path = artifact_path(fixture.path(), &artifact_id);
        fs::remove_file(&internal_path)?;
        let outside_directory = fixture.path().join("outside-context-packs");
        fs::create_dir_all(&outside_directory)?;
        let outside_path = outside_directory.join(format!("{}.json", artifact_id.as_str()));
        fs::write(&outside_path, pack.to_stable_json()?)?;
        std::os::unix::fs::symlink(&outside_path, &internal_path)?;

        let actual =
            select_evidence_ledger_replay(&indexer, &manifest, &fixture_request(&manifest));
        let expected = (0usize, true);

        assert_eq!(
            (
                actual.selected.len(),
                actual.issues.iter().any(|issue| matches!(
                    issue.code,
                    EvidenceReplayIssueCode::ArtifactPathEscape
                        | EvidenceReplayIssueCode::InvalidArtifactPath
                )),
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn replay_request_manifest_mismatch_is_not_broad_replacement_path() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let indexer = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = indexer.index()?;
        write_fixture_context_pack(&indexer, &manifest, EvidenceFreshness::Fresh, 1.0)?;
        let mut request = fixture_request(&manifest);
        request.manifest.manifest_hash = "old".to_string();

        let actual = select_evidence_ledger_replay(&indexer, &manifest, &request);
        let expected = (
            0usize,
            BTreeSet::from([EvidenceReplayIssueCode::ManifestMismatch]),
        );

        assert_eq!((actual.selected.len(), issue_codes(&actual)), expected);
        Ok(())
    }
}
