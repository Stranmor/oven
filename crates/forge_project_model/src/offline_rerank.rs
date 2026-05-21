//! Provider-neutral offline rerank-score artifacts and exact-match boundaries.

use std::collections::BTreeSet;
use std::fmt;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::types::{ProjectManifest, RerankCandidate, RerankScore};
use crate::util::hash_text;

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
}

impl crate::vector::Reranker for OfflineRerankScoreArtifactReranker {
    fn rerank(&self, query: &str, candidates: &[RerankCandidate]) -> Vec<RerankScore> {
        let request_key = OfflineRerankScoreKey::from_manifest_hash(
            &self.manifest_hash,
            query,
            candidates,
            self.producer_identity.clone(),
            self.ordering_policy.clone(),
            self.top_k_scope.clone(),
        );
        self.artifact.scores_for_exact_key(&request_key)
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

    fn fixture_artifact(manifest: &ProjectManifest) -> Result<OfflineRerankScoreArtifact> {
        let candidates = fixture_candidates(manifest);
        let key = OfflineRerankScoreKey::from_request(
            manifest,
            "intent text",
            &candidates,
            fixture_identity(),
            OfflineRerankOrderingPolicy::InputOrder,
            OfflineRerankTopKScope::new(Some(candidates.len())),
        );
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
