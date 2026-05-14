//! Rust-native evaluation harness for retrieval, graph, freshness, and
//! provenance.

use std::collections::BTreeSet;

use crate::freshness::compare_freshness;
use crate::retrieval::retrieve;
use crate::types::{
    FreshnessEvalReport, GraphCoverageReport, GraphEdge, ProjectManifest,
    ProvenanceCompletenessReport, RetrievalEvalCase, RetrievalEvalReport,
};

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

    use anyhow::Result;
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::indexer::tests::fixture_project;
    use crate::{GraphEdgeKind, ProjectIndexer, RetrievalQuery};

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
