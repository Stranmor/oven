//! Provider-neutral vector search and reranking boundaries.

use std::collections::{BTreeMap, BTreeSet};

use crate::lexical::tokenize;
use crate::types::{RerankCandidate, RerankScore, VectorQuery, VectorSearchHit};

/// Typed vector search boundary implemented by external embedding integrations.
pub trait VectorIndex {
    /// Searches by an already-computed embedding vector.
    ///
    /// # Arguments
    ///
    /// * `query` - Provider-neutral vector query produced outside this crate.
    fn search(&self, query: &VectorQuery) -> Vec<VectorSearchHit>;
}

/// Typed reranker boundary implemented by external reranking integrations.
pub trait Reranker {
    /// Reranks candidate text surfaces for a query.
    ///
    /// # Arguments
    ///
    /// * `query` - Free-form query text.
    /// * `candidates` - Candidate identifiers and text surfaces.
    fn rerank(&self, query: &str, candidates: &[RerankCandidate]) -> Vec<RerankScore>;
}

/// Deterministic in-memory vector index for tests and offline evaluation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeterministicVectorIndex {
    vectors: BTreeMap<String, Vec<f32>>,
}

impl DeterministicVectorIndex {
    /// Creates a deterministic vector index from precomputed vectors.
    ///
    /// # Arguments
    ///
    /// * `vectors` - Mapping from result identifiers to embedding vectors.
    pub fn new(vectors: BTreeMap<String, Vec<f32>>) -> Self {
        Self { vectors }
    }
}

impl VectorIndex for DeterministicVectorIndex {
    fn search(&self, query: &VectorQuery) -> Vec<VectorSearchHit> {
        let mut hits = self
            .vectors
            .iter()
            .filter_map(|(id, vector)| {
                cosine_similarity(&query.embedding, vector).map(|score| (id, score))
            })
            .map(|(id, score)| VectorSearchHit { id: id.clone(), score })
            .collect::<Vec<_>>();
        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.id.cmp(&right.id))
        });
        hits
    }
}

/// Deterministic token-overlap reranker for tests and offline evaluation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeterministicReranker;

impl Reranker for DeterministicReranker {
    fn rerank(&self, query: &str, candidates: &[RerankCandidate]) -> Vec<RerankScore> {
        let query_terms = tokenize(query).into_iter().collect::<BTreeSet<_>>();
        let mut scores = candidates
            .iter()
            .map(|candidate| {
                let candidate_terms = tokenize(&candidate.text)
                    .into_iter()
                    .collect::<BTreeSet<_>>();
                let overlap = query_terms.intersection(&candidate_terms).count() as f32;
                let union = query_terms.union(&candidate_terms).count() as f32;
                let score = if union > 0.0 { overlap / union } else { 0.0 };
                RerankScore { id: candidate.id.clone(), score }
            })
            .collect::<Vec<_>>();
        scores.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.id.cmp(&right.id))
        });
        scores
    }
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    let mut dot = 0.0f32;
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;
    for (left_value, right_value) in left.iter().zip(right) {
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }
    let denominator = left_norm.sqrt() * right_norm.sqrt();
    if denominator > 0.0 {
        Some(dot / denominator)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn deterministic_vector_index_returns_cosine_ranked_hits() {
        let setup = DeterministicVectorIndex::new(BTreeMap::from([
            ("near".to_string(), vec![1.0, 0.0]),
            ("far".to_string(), vec![0.0, 1.0]),
        ]));
        let actual = setup.search(&VectorQuery { embedding: vec![1.0, 0.0] });
        let expected = vec!["near".to_string(), "far".to_string()];
        assert_eq!(
            actual.into_iter().map(|hit| hit.id).collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn deterministic_reranker_scores_token_overlap() {
        let setup = DeterministicReranker;
        let actual = setup.rerank(
            "project model",
            &[
                RerankCandidate {
                    id: "a".to_string(),
                    text: "project model context".to_string(),
                },
                RerankCandidate { id: "b".to_string(), text: "provider dto".to_string() },
            ],
        );
        let expected = "a".to_string();
        assert_eq!(
            actual
                .first()
                .map(|score| score.id.clone())
                .expect("reranker should return at least one score"),
            expected
        );
    }
}
