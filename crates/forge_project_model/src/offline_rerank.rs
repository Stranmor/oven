//! Provider-neutral offline rerank-score artifacts and exact-match boundaries.

use std::collections::BTreeSet;
use std::fmt;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::types::{ProjectManifest, RerankCandidate, RerankScore};
use crate::util::hash_text;

/// Redaction-safe applicability of one offline rerank artifact key to a live request key.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub enum OfflineRerankApplicability {
    /// Artifact key exactly matches the current rerank request key.
    ExactMatch,
    /// Artifact key is valid but not applicable to this current request key.
    Mismatch {
        /// Deterministically ordered redaction-safe mismatch reasons.
        reasons: Vec<OfflineRerankApplicabilityMismatch>,
    },
}

/// Deterministic redaction-safe reason an offline rerank artifact key is not applicable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub enum OfflineRerankApplicabilityMismatch {
    /// Artifact was produced for a different project manifest snapshot.
    ManifestHashMismatch,
    /// Artifact was produced for a different rerank intent fingerprint.
    RerankIntentFingerprintMismatch,
    /// Artifact candidate identifiers or their order differ from the current request.
    CandidateIdsOrderMismatch,
    /// Artifact candidate content fingerprints differ from the current request.
    CandidateContentFingerprintMismatch,
    /// Artifact top-k scope differs from the current request.
    TopKScopeMismatch,
    /// Artifact producer identity or ordering policy differs from the current request.
    ProducerIdentityPolicyMismatch,
    /// Artifact score schema version differs from the current request.
    ScoreArtifactVersionMismatch,
}

/// Compares a validated artifact key with the current live request key without exposing raw payloads.
///
/// # Arguments
///
/// * `artifact_key` - Redaction-safe key persisted in the offline rerank artifact.
/// * `request_key` - Redaction-safe exact key computed for the current runtime request.
pub fn offline_rerank_applicability(
    artifact_key: &OfflineRerankScoreKey,
    request_key: &OfflineRerankScoreKey,
) -> OfflineRerankApplicability {
    let mut reasons = Vec::new();
    if artifact_key.manifest_hash != request_key.manifest_hash {
        reasons.push(OfflineRerankApplicabilityMismatch::ManifestHashMismatch);
    }
    if artifact_key.rerank_intent_fingerprint != request_key.rerank_intent_fingerprint {
        reasons.push(OfflineRerankApplicabilityMismatch::RerankIntentFingerprintMismatch);
    }
    if artifact_key.ordered_candidate_ids != request_key.ordered_candidate_ids {
        reasons.push(OfflineRerankApplicabilityMismatch::CandidateIdsOrderMismatch);
    }
    if artifact_key.ordered_candidate_content_fingerprints
        != request_key.ordered_candidate_content_fingerprints
    {
        reasons.push(OfflineRerankApplicabilityMismatch::CandidateContentFingerprintMismatch);
    }
    if artifact_key.top_k_scope != request_key.top_k_scope {
        reasons.push(OfflineRerankApplicabilityMismatch::TopKScopeMismatch);
    }
    if artifact_key.producer_identity != request_key.producer_identity
        || artifact_key.ordering_policy != request_key.ordering_policy
    {
        reasons.push(OfflineRerankApplicabilityMismatch::ProducerIdentityPolicyMismatch);
    }
    if artifact_key.score_artifact_version != request_key.score_artifact_version {
        reasons.push(OfflineRerankApplicabilityMismatch::ScoreArtifactVersionMismatch);
    }

    if reasons.is_empty() {
        OfflineRerankApplicability::ExactMatch
    } else {
        OfflineRerankApplicability::Mismatch { reasons }
    }
}

/// Current offline rerank-score artifact format version.
pub const OFFLINE_RERANK_SCORE_ARTIFACT_VERSION: u32 = 1;

/// Ordering policy used when the score artifact was produced.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OfflineRerankOrderingPolicy {
    /// Candidate order is the exact order supplied to the reranker boundary.
    #[default]
    InputOrder,
}

/// Top-k candidate scope used when the score artifact was produced.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OfflineRerankTopKScope {
    /// Optional producer-side candidate cap.
    pub top_k: Option<usize>,
}

impl OfflineRerankTopKScope {
    /// Creates a top-k scope from an optional producer-side cap.
    ///
    /// # Arguments
    ///
    /// * `top_k` - Optional number of candidates included by the producer.
    pub fn new(top_k: Option<usize>) -> Self {
        Self { top_k }
    }
}

/// Redaction-safe producer identity for offline rerank-score artifacts.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OfflineRerankProducerIdentity {
    /// Provider, model, or offline subsystem label.
    pub producer: String,
    /// Provider model id or offline producer version label.
    pub model_or_version: String,
}

impl OfflineRerankProducerIdentity {
    /// Creates a producer identity.
    ///
    /// # Arguments
    ///
    /// * `producer` - Provider, model, or offline subsystem label.
    /// * `model_or_version` - Provider model id or offline producer version label.
    pub fn new(producer: impl Into<String>, model_or_version: impl Into<String>) -> Self {
        Self {
            producer: producer.into(),
            model_or_version: model_or_version.into(),
        }
    }

    fn validate(&self) -> Result<()> {
        if self.producer.trim().is_empty() {
            bail!("offline rerank producer identity must be non-empty");
        }
        if self.model_or_version.trim().is_empty() {
            bail!("offline rerank model or producer version must be non-empty");
        }
        Ok(())
    }
}

/// Exact redaction-safe key for a cached offline rerank-score artifact.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OfflineRerankScoreKey {
    /// Rerank intent fingerprint; raw query or use-case text is never stored here.
    pub rerank_intent_fingerprint: String,
    /// Manifest hash for the project-model snapshot used by the producer.
    pub manifest_hash: String,
    /// Ordered candidate identifiers accepted by the producer.
    pub ordered_candidate_ids: Vec<String>,
    /// Ordered candidate text/content fingerprints accepted by the producer.
    pub ordered_candidate_content_fingerprints: Vec<String>,
    /// Provider/model or offline producer identity.
    pub producer_identity: OfflineRerankProducerIdentity,
    /// Score artifact schema version.
    pub score_artifact_version: u32,
    /// Candidate ordering policy.
    pub ordering_policy: OfflineRerankOrderingPolicy,
    /// Candidate top-k scope.
    pub top_k_scope: OfflineRerankTopKScope,
}

impl OfflineRerankScoreKey {
    /// Builds the exact redaction-safe key for a rerank request.
    ///
    /// # Arguments
    ///
    /// * `manifest` - Current project manifest that owns the candidate snapshot.
    /// * `rerank_intent` - Raw rerank intent text used only to compute a fingerprint.
    /// * `candidates` - Ordered rerank candidates used only for ids and text fingerprints.
    /// * `producer_identity` - Provider/model or offline producer identity.
    /// * `ordering_policy` - Candidate ordering policy.
    /// * `top_k_scope` - Candidate top-k scope.
    pub fn from_request(
        manifest: &ProjectManifest,
        rerank_intent: &str,
        candidates: &[RerankCandidate],
        producer_identity: OfflineRerankProducerIdentity,
        ordering_policy: OfflineRerankOrderingPolicy,
        top_k_scope: OfflineRerankTopKScope,
    ) -> Self {
        Self::from_manifest_hash(
            &manifest.manifest_hash,
            rerank_intent,
            candidates,
            producer_identity,
            ordering_policy,
            top_k_scope,
        )
    }

    /// Builds the exact redaction-safe key for a known manifest hash.
    ///
    /// # Arguments
    ///
    /// * `manifest_hash` - Manifest hash for the project-model snapshot.
    /// * `rerank_intent` - Raw rerank intent text used only to compute a fingerprint.
    /// * `candidates` - Ordered rerank candidates used only for ids and text fingerprints.
    /// * `producer_identity` - Provider/model or offline producer identity.
    /// * `ordering_policy` - Candidate ordering policy.
    /// * `top_k_scope` - Candidate top-k scope.
    pub fn from_manifest_hash(
        manifest_hash: &str,
        rerank_intent: &str,
        candidates: &[RerankCandidate],
        producer_identity: OfflineRerankProducerIdentity,
        ordering_policy: OfflineRerankOrderingPolicy,
        top_k_scope: OfflineRerankTopKScope,
    ) -> Self {
        Self {
            rerank_intent_fingerprint: hash_text(rerank_intent.trim()),
            manifest_hash: manifest_hash.to_string(),
            ordered_candidate_ids: candidates
                .iter()
                .map(|candidate| candidate.id.clone())
                .collect(),
            ordered_candidate_content_fingerprints: candidates
                .iter()
                .map(|candidate| hash_text(&candidate.text))
                .collect(),
            producer_identity,
            score_artifact_version: OFFLINE_RERANK_SCORE_ARTIFACT_VERSION,
            ordering_policy,
            top_k_scope,
        }
    }

    /// Validates this key against schema and candidate identity invariants.
    ///
    /// # Errors
    ///
    /// Returns an error when required identity fields are absent or inconsistent.
    pub fn validate(&self) -> Result<()> {
        if self.score_artifact_version != OFFLINE_RERANK_SCORE_ARTIFACT_VERSION {
            bail!("offline rerank score artifact version is unsupported");
        }
        if self.rerank_intent_fingerprint.trim().is_empty() {
            bail!("offline rerank intent fingerprint must be non-empty");
        }
        if self.manifest_hash.trim().is_empty() {
            bail!("offline rerank manifest hash must be non-empty");
        }
        self.producer_identity.validate()?;
        if self.ordered_candidate_ids.is_empty() {
            bail!("offline rerank key requires at least one candidate id");
        }
        if self.ordered_candidate_ids.len() != self.ordered_candidate_content_fingerprints.len() {
            bail!("offline rerank candidate id and fingerprint counts differ");
        }
        if self
            .ordered_candidate_content_fingerprints
            .iter()
            .any(|fingerprint| fingerprint.trim().is_empty())
        {
            bail!("offline rerank candidate fingerprints must be non-empty");
        }
        let mut unique_ids = BTreeSet::new();
        for candidate_id in &self.ordered_candidate_ids {
            if candidate_id.trim().is_empty() {
                bail!("offline rerank candidate ids must be non-empty");
            }
            if !unique_ids.insert(candidate_id) {
                bail!("offline rerank candidate ids must be unique");
            }
        }
        Ok(())
    }
}

/// Durable offline rerank-score artifact containing scores for one exact key only.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OfflineRerankScoreArtifact {
    /// Artifact format version.
    pub version: u32,
    /// Exact redaction-safe artifact key.
    pub key: OfflineRerankScoreKey,
    /// Scores for the exact candidate set.
    pub scores: Vec<RerankScore>,
    /// Deterministic fingerprint over artifact fields excluding this fingerprint.
    pub score_fingerprint: String,
}

impl OfflineRerankScoreArtifact {
    /// Builds and validates an offline rerank-score artifact.
    ///
    /// # Arguments
    ///
    /// * `key` - Exact redaction-safe score artifact key.
    /// * `scores` - Scores for the exact candidate set.
    ///
    /// # Errors
    ///
    /// Returns an error when the key or scores are invalid.
    pub fn new(key: OfflineRerankScoreKey, scores: Vec<RerankScore>) -> Result<Self> {
        let mut artifact = Self {
            version: OFFLINE_RERANK_SCORE_ARTIFACT_VERSION,
            key,
            scores,
            score_fingerprint: String::new(),
        };
        artifact.score_fingerprint = artifact.compute_score_fingerprint()?;
        artifact.validate()?;
        Ok(artifact)
    }

    /// Validates this artifact without matching it to a live request.
    ///
    /// # Errors
    ///
    /// Returns an error when artifact schema, key, score, or fingerprint invariants fail.
    pub fn validate(&self) -> Result<()> {
        if self.version != OFFLINE_RERANK_SCORE_ARTIFACT_VERSION {
            bail!("offline rerank artifact version is unsupported");
        }
        self.key.validate()?;
        if self.scores.len() != self.key.ordered_candidate_ids.len() {
            bail!("offline rerank score count must match candidate count");
        }
        let candidate_ids = self
            .key
            .ordered_candidate_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let mut score_ids = BTreeSet::new();
        for score in &self.scores {
            if !score.score.is_finite() {
                bail!("offline rerank scores must be finite");
            }
            if !candidate_ids.contains(&score.id) {
                bail!("offline rerank score id is absent from candidate key");
            }
            if !score_ids.insert(score.id.clone()) {
                bail!("offline rerank score ids must be unique");
            }
        }
        let expected = self.compute_score_fingerprint()?;
        if self.score_fingerprint != expected {
            bail!("offline rerank score artifact fingerprint mismatch");
        }
        Ok(())
    }

    /// Returns scores only when the current request key exactly matches the artifact key.
    ///
    /// # Arguments
    ///
    /// * `request_key` - Exact request key computed at the reranker boundary.
    pub fn scores_for_exact_key(&self, request_key: &OfflineRerankScoreKey) -> Vec<RerankScore> {
        if self.validate().is_err() || &self.key != request_key {
            Vec::new()
        } else {
            sorted_scores(self.scores.clone())
        }
    }

    /// Serializes this artifact as stable pretty JSON.
    ///
    /// # Errors
    ///
    /// Returns an error when JSON serialization fails.
    pub fn to_stable_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    fn compute_score_fingerprint(&self) -> Result<String> {
        let payload = OfflineRerankScoreFingerprintPayload {
            version: self.version,
            key: &self.key,
            scores: &self.scores,
        };
        Ok(hash_text(&serde_json::to_string_pretty(&payload)?))
    }
}

#[derive(Serialize)]
struct OfflineRerankScoreFingerprintPayload<'a> {
    version: u32,
    key: &'a OfflineRerankScoreKey,
    scores: &'a [RerankScore],
}

/// Validated in-memory offline reranker for one cached score artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct OfflineRerankScoreArtifactReranker {
    manifest_hash: String,
    producer_identity: OfflineRerankProducerIdentity,
    ordering_policy: OfflineRerankOrderingPolicy,
    top_k_scope: OfflineRerankTopKScope,
    artifact: OfflineRerankScoreArtifact,
}

impl OfflineRerankScoreArtifactReranker {
    /// Builds a reranker from a validated offline score artifact.
    ///
    /// # Arguments
    ///
    /// * `manifest` - Current project manifest that owns the candidate snapshot.
    /// * `artifact` - Offline score artifact to validate.
    ///
    /// # Errors
    ///
    /// Returns an error when the artifact is corrupt or stale for the manifest.
    pub fn new(manifest: &ProjectManifest, artifact: OfflineRerankScoreArtifact) -> Result<Self> {
        artifact
            .validate()
            .context("offline rerank artifact is invalid")?;
        if artifact.key.manifest_hash != manifest.manifest_hash {
            bail!("offline rerank artifact manifest hash mismatch");
        }
        Ok(Self {
            manifest_hash: artifact.key.manifest_hash.clone(),
            producer_identity: artifact.key.producer_identity.clone(),
            ordering_policy: artifact.key.ordering_policy.clone(),
            top_k_scope: artifact.key.top_k_scope.clone(),
            artifact,
        })
    }

    /// Returns the artifact producer identity.
    pub fn producer_identity(&self) -> &OfflineRerankProducerIdentity {
        &self.producer_identity
    }
    /// Returns redaction-safe applicability diagnostics for the current request key.
    ///
    /// # Arguments
    ///
    /// * `query` - Raw rerank intent text used only to compute a fingerprint.
    /// * `candidates` - Ordered candidates used only for ids and content fingerprints.
    pub fn applicability_for_request(
        &self,
        query: &str,
        candidates: &[RerankCandidate],
    ) -> OfflineRerankApplicability {
        let request_key = self.request_key(query, candidates);
        offline_rerank_applicability(&self.artifact.key, &request_key)
    }

    fn request_key(&self, query: &str, candidates: &[RerankCandidate]) -> OfflineRerankScoreKey {
        OfflineRerankScoreKey::from_manifest_hash(
            &self.manifest_hash,
            query,
            candidates,
            self.producer_identity.clone(),
            self.ordering_policy.clone(),
            self.top_k_scope.clone(),
        )
    }
}

impl crate::vector::Reranker for OfflineRerankScoreArtifactReranker {
    fn rerank(&self, query: &str, candidates: &[RerankCandidate]) -> Vec<RerankScore> {
        let request_key = self.request_key(query, candidates);
        self.artifact.scores_for_exact_key(&request_key)
    }

    fn offline_applicability(
        &self,
        query: &str,
        candidates: &[RerankCandidate],
    ) -> Option<OfflineRerankApplicability> {
        Some(self.applicability_for_request(query, candidates))
    }
}

fn sorted_scores(mut scores: Vec<RerankScore>) -> Vec<RerankScore> {
    scores.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.id.cmp(&right.id))
    });
    scores
}

impl fmt::Display for OfflineRerankProducerIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:{}", self.producer, self.model_or_version)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::ProjectIndexer;
    use crate::indexer::tests::fixture_project;
    use crate::vector::Reranker;

    fn fixture_manifest() -> Result<(tempfile::TempDir, ProjectManifest)> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        Ok((fixture, manifest))
    }

    fn fixture_candidates(manifest: &ProjectManifest) -> Vec<RerankCandidate> {
        manifest
            .shards
            .iter()
            .take(2)
            .map(|shard| RerankCandidate {
                id: shard.id.clone(),
                text: format!("{} {}", shard.path, shard.content_hash),
            })
            .collect()
    }

    fn fixture_identity() -> OfflineRerankProducerIdentity {
        OfflineRerankProducerIdentity::new("offline-fixture", "model-v1")
    }

    fn fixture_key(manifest: &ProjectManifest) -> OfflineRerankScoreKey {
        let candidates = fixture_candidates(manifest);
        OfflineRerankScoreKey::from_request(
            manifest,
            "intent text",
            &candidates,
            fixture_identity(),
            OfflineRerankOrderingPolicy::InputOrder,
            OfflineRerankTopKScope::new(Some(candidates.len())),
        )
    }

    fn fixture_artifact(manifest: &ProjectManifest) -> Result<OfflineRerankScoreArtifact> {
        let candidates = fixture_candidates(manifest);
        let key = fixture_key(manifest);
        let scores = candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| RerankScore {
                id: candidate.id.clone(),
                score: index as f32 + 1.0,
            })
            .collect();
        OfflineRerankScoreArtifact::new(key, scores)
    }

    #[test]
    fn offline_rerank_applicability_exact_match_returns_exact_match() -> Result<()> {
        let (_fixture, manifest) = fixture_manifest()?;
        let setup = fixture_key(&manifest);

        let actual = offline_rerank_applicability(&setup, &setup);
        let expected = OfflineRerankApplicability::ExactMatch;

        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn offline_rerank_applicability_returns_each_first_slice_reason_independently() -> Result<()> {
        let (_fixture, manifest) = fixture_manifest()?;
        let setup = fixture_key(&manifest);
        let mut manifest_mismatch = setup.clone();
        manifest_mismatch.manifest_hash = hash_text("different manifest");
        let mut intent_mismatch = setup.clone();
        intent_mismatch.rerank_intent_fingerprint = hash_text("different intent");
        let mut candidate_ids_mismatch = setup.clone();
        candidate_ids_mismatch.ordered_candidate_ids.reverse();
        let mut content_mismatch = setup.clone();
        content_mismatch.ordered_candidate_content_fingerprints[0] = hash_text("different content");
        let mut top_k_mismatch = setup.clone();
        top_k_mismatch.top_k_scope = OfflineRerankTopKScope::new(Some(99));
        let mut producer_mismatch = setup.clone();
        producer_mismatch.producer_identity =
            OfflineRerankProducerIdentity::new("different", "model-v1");
        let mut version_mismatch = setup.clone();
        version_mismatch.score_artifact_version = OFFLINE_RERANK_SCORE_ARTIFACT_VERSION + 1;

        let actual = vec![
            mismatch_reasons(&manifest_mismatch, &setup),
            mismatch_reasons(&intent_mismatch, &setup),
            mismatch_reasons(&candidate_ids_mismatch, &setup),
            mismatch_reasons(&content_mismatch, &setup),
            mismatch_reasons(&top_k_mismatch, &setup),
            mismatch_reasons(&producer_mismatch, &setup),
            mismatch_reasons(&version_mismatch, &setup),
        ];
        let expected = vec![
            vec![OfflineRerankApplicabilityMismatch::ManifestHashMismatch],
            vec![OfflineRerankApplicabilityMismatch::RerankIntentFingerprintMismatch],
            vec![OfflineRerankApplicabilityMismatch::CandidateIdsOrderMismatch],
            vec![OfflineRerankApplicabilityMismatch::CandidateContentFingerprintMismatch],
            vec![OfflineRerankApplicabilityMismatch::TopKScopeMismatch],
            vec![OfflineRerankApplicabilityMismatch::ProducerIdentityPolicyMismatch],
            vec![OfflineRerankApplicabilityMismatch::ScoreArtifactVersionMismatch],
        ];

        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn offline_rerank_applicability_returns_multiple_reasons_in_deterministic_order() -> Result<()>
    {
        let (_fixture, manifest) = fixture_manifest()?;
        let setup = fixture_key(&manifest);
        let mut mismatch = setup.clone();
        mismatch.manifest_hash = hash_text("different manifest");
        mismatch.rerank_intent_fingerprint = hash_text("different intent");
        mismatch.ordered_candidate_ids.reverse();
        mismatch.ordered_candidate_content_fingerprints[0] = hash_text("different content");
        mismatch.top_k_scope = OfflineRerankTopKScope::new(Some(99));
        mismatch.producer_identity = OfflineRerankProducerIdentity::new("different", "model-v1");
        mismatch.score_artifact_version = OFFLINE_RERANK_SCORE_ARTIFACT_VERSION + 1;

        let actual = mismatch_reasons(&mismatch, &setup);
        let expected = vec![
            OfflineRerankApplicabilityMismatch::ManifestHashMismatch,
            OfflineRerankApplicabilityMismatch::RerankIntentFingerprintMismatch,
            OfflineRerankApplicabilityMismatch::CandidateIdsOrderMismatch,
            OfflineRerankApplicabilityMismatch::CandidateContentFingerprintMismatch,
            OfflineRerankApplicabilityMismatch::TopKScopeMismatch,
            OfflineRerankApplicabilityMismatch::ProducerIdentityPolicyMismatch,
            OfflineRerankApplicabilityMismatch::ScoreArtifactVersionMismatch,
        ];

        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn offline_rerank_applicability_serialization_is_redaction_safe() -> Result<()> {
        let (_fixture, manifest) = fixture_manifest()?;
        let artifact_candidates = vec![RerankCandidate {
            id: "candidate-a".to_string(),
            text: "raw artifact candidate secret".to_string(),
        }];
        let request_candidates = vec![RerankCandidate {
            id: "candidate-a".to_string(),
            text: "raw request candidate secret".to_string(),
        }];
        let artifact_key = OfflineRerankScoreKey::from_request(
            &manifest,
            "raw artifact query secret",
            &artifact_candidates,
            fixture_identity(),
            OfflineRerankOrderingPolicy::InputOrder,
            OfflineRerankTopKScope::new(Some(1)),
        );
        let request_key = OfflineRerankScoreKey::from_request(
            &manifest,
            "raw request query secret",
            &request_candidates,
            fixture_identity(),
            OfflineRerankOrderingPolicy::InputOrder,
            OfflineRerankTopKScope::new(Some(1)),
        );

        let actual = format!(
            "{:?} {}",
            offline_rerank_applicability(&artifact_key, &request_key),
            serde_json::to_string(&offline_rerank_applicability(&artifact_key, &request_key))?
        );
        let expected = vec![false, false, false, false, false, false];

        assert_eq!(
            vec![
                actual.contains("raw artifact query secret"),
                actual.contains("raw request query secret"),
                actual.contains("raw artifact candidate secret"),
                actual.contains("raw request candidate secret"),
                actual.contains("/tmp/offline_rerank_scores.json"),
                actual.contains("configured/path/offline_rerank_scores.json"),
            ],
            expected,
        );
        Ok(())
    }

    fn mismatch_reasons(
        artifact_key: &OfflineRerankScoreKey,
        request_key: &OfflineRerankScoreKey,
    ) -> Vec<OfflineRerankApplicabilityMismatch> {
        match offline_rerank_applicability(artifact_key, request_key) {
            OfflineRerankApplicability::ExactMatch => Vec::new(),
            OfflineRerankApplicability::Mismatch { reasons } => reasons,
        }
    }

    #[test]
    fn offline_rerank_key_uses_fingerprints_without_raw_query_or_candidate_text() -> Result<()> {
        let (_fixture, manifest) = fixture_manifest()?;
        let candidates = vec![RerankCandidate {
            id: "candidate-a".to_string(),
            text: "raw candidate secret text".to_string(),
        }];
        let setup = OfflineRerankScoreKey::from_request(
            &manifest,
            "raw query secret",
            &candidates,
            fixture_identity(),
            OfflineRerankOrderingPolicy::InputOrder,
            OfflineRerankTopKScope::new(Some(1)),
        );

        let actual = serde_json::to_string(&setup)?;
        let expected = false;

        assert_eq!(actual.contains("raw query secret"), expected);
        assert_eq!(actual.contains("raw candidate secret text"), expected);
        assert_eq!(setup.ordered_candidate_ids, vec!["candidate-a".to_string()]);
        assert_eq!(setup.ordered_candidate_content_fingerprints.len(), 1usize);
        Ok(())
    }

    #[test]
    fn offline_rerank_artifact_rejects_stale_duplicate_corrupt_and_invalid_scores() -> Result<()> {
        let (_fixture, manifest) = fixture_manifest()?;
        let artifact = fixture_artifact(&manifest)?;
        let mut stale = artifact.clone();
        stale.key.manifest_hash = hash_text("stale-manifest");
        stale.score_fingerprint = stale.compute_score_fingerprint()?;

        let mut duplicate = artifact.clone();
        duplicate.key.ordered_candidate_ids[1] = duplicate.key.ordered_candidate_ids[0].clone();
        duplicate.score_fingerprint = duplicate.compute_score_fingerprint()?;

        let mut corrupt = artifact.clone();
        corrupt.score_fingerprint = hash_text("corrupt");

        let mut invalid_score = artifact.clone();
        invalid_score.scores[0].score = f32::NAN;
        invalid_score.score_fingerprint = invalid_score.compute_score_fingerprint()?;

        let actual = vec![
            OfflineRerankScoreArtifactReranker::new(&manifest, stale).is_err(),
            duplicate.validate().is_err(),
            corrupt.validate().is_err(),
            invalid_score.validate().is_err(),
        ];
        let expected = vec![true, true, true, true];

        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn offline_rerank_artifact_rejects_unknown_raw_text_fields() -> Result<()> {
        let (_fixture, manifest) = fixture_manifest()?;
        let artifact = fixture_artifact(&manifest)?;
        let mut setup: serde_json::Value = serde_json::from_str(&artifact.to_stable_json()?)?;
        setup["raw_query_text"] = serde_json::Value::String("raw query secret".to_string());
        setup["key"]["raw_candidate_text"] =
            serde_json::Value::String("raw candidate secret text".to_string());

        let actual = serde_json::from_value::<OfflineRerankScoreArtifact>(setup).is_err();
        let expected = true;

        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn offline_rerank_artifact_returns_scores_only_for_exact_key_match() -> Result<()> {
        let (_fixture, manifest) = fixture_manifest()?;
        let candidates = fixture_candidates(&manifest);
        let setup =
            OfflineRerankScoreArtifactReranker::new(&manifest, fixture_artifact(&manifest)?)?;

        let actual = (
            setup.rerank("intent text", &candidates).len(),
            setup.rerank("different intent", &candidates).len(),
            setup
                .rerank(
                    "intent text",
                    &[RerankCandidate {
                        id: candidates[0].id.clone(),
                        text: "changed content".to_string(),
                    }],
                )
                .len(),
        );
        let expected = (2usize, 0usize, 0usize);

        assert_eq!(actual, expected);
        Ok(())
    }
}
