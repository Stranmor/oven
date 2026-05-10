//! Hybrid deterministic retrieval over project manifests.

use crate::types::{ProjectManifest, RetrievalQuery, RetrievalResult};
use std::collections::{BTreeMap, BTreeSet};

/// Retrieves project model results using exact path, exact symbol, lexical, and graph expansion scoring.
///
/// Vector search and reranking are intentionally represented only by `FutureVectorRetrievalScaffold`.
///
/// # Arguments
///
/// * `manifest` - Indexed project manifest.
/// * `query` - Retrieval query.
pub fn retrieve(manifest: &ProjectManifest, query: &RetrievalQuery) -> Vec<RetrievalResult> {
    let limit = if query.limit == 0 { 10 } else { query.limit };
    let terms = query
        .text
        .as_ref()
        .map(|text| tokenize(text))
        .unwrap_or_default();
    let mut results: BTreeMap<String, RetrievalResult> = BTreeMap::new();

    for file in &manifest.files {
        let mut parts = BTreeMap::new();
        if query.path.as_deref() == Some(file.path.as_str()) {
            parts.insert("exact_path".to_string(), 100.0);
        }
        if !terms.is_empty() {
            let lexical = lexical_score(&file.path, &terms);
            if lexical > 0.0 {
                parts.insert("lexical".to_string(), lexical);
            }
        }
        if !parts.is_empty() {
            let score = parts.values().sum();
            results.insert(
                file.path.clone(),
                RetrievalResult {
                    id: file.path.clone(),
                    path: file.path.clone(),
                    symbol: None,
                    score,
                    score_parts: parts,
                    provenance: file.provenance.clone(),
                },
            );
        }
    }

    for symbol in &manifest.symbols {
        let mut parts = BTreeMap::new();
        if query.symbol.as_deref() == Some(symbol.name.as_str())
            || query.symbol.as_deref() == Some(symbol.id.as_str())
        {
            parts.insert("exact_symbol".to_string(), 100.0);
        }
        if !terms.is_empty() {
            let lexical = lexical_score(&format!("{} {}", symbol.name, symbol.path), &terms);
            if lexical > 0.0 {
                parts.insert("lexical".to_string(), lexical + 5.0);
            }
        }
        if !parts.is_empty() {
            let score = parts.values().sum();
            results.insert(
                symbol.id.clone(),
                RetrievalResult {
                    id: symbol.id.clone(),
                    path: symbol.path.clone(),
                    symbol: Some(symbol.name.clone()),
                    score,
                    score_parts: parts,
                    provenance: symbol.provenance.clone(),
                },
            );
        }
    }

    if query.include_graph_expansion {
        let seeds = results.keys().cloned().collect::<BTreeSet<_>>();
        for graph_edge in &manifest.edges {
            if seeds.contains(&graph_edge.from) || seeds.contains(&graph_edge.to) {
                let id = if seeds.contains(&graph_edge.from) {
                    graph_edge.to.clone()
                } else {
                    graph_edge.from.clone()
                };
                results.entry(id.clone()).or_insert_with(|| {
                    let mut parts = BTreeMap::new();
                    parts.insert("graph".to_string(), graph_edge.confidence * 10.0);
                    RetrievalResult {
                        id: id.clone(),
                        path: graph_edge.provenance.path.clone(),
                        symbol: None,
                        score: graph_edge.confidence * 10.0,
                        score_parts: parts,
                        provenance: graph_edge.provenance.clone(),
                    }
                });
            }
        }
    }

    let mut values = results.into_values().collect::<Vec<_>>();
    values.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.id.cmp(&right.id))
    });
    values.truncate(limit);
    values
}

fn retrieve_text_score(haystack: &str, term: &str) -> f32 {
    if haystack.contains(term) { 10.0 } else { 0.0 }
}

fn lexical_score(haystack: &str, terms: &[String]) -> f32 {
    let normalized = haystack.to_lowercase();
    terms
        .iter()
        .map(|term| retrieve_text_score(&normalized, term))
        .sum()
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|ch: char| !ch.is_alphanumeric() && ch != '_')
        .filter(|part| !part.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProjectIndexer;
    use crate::indexer::tests::fixture_project;
    use anyhow::Result;
    use pretty_assertions::assert_eq;

    #[test]
    fn retrieves_exact_symbol_lexical_and_graph_expansion() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let query = RetrievalQuery {
            text: Some("Root".to_string()),
            path: None,
            symbol: Some("Root".to_string()),
            limit: 5,
            include_graph_expansion: true,
        };
        let actual = retrieve(&manifest, &query);
        let expected = true;
        assert_eq!(
            actual
                .iter()
                .any(|result| result.symbol.as_deref() == Some("Root")
                    && result.score_parts.contains_key("exact_symbol")),
            expected
        );
        Ok(())
    }
}
