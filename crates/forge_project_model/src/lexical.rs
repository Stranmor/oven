//! Deterministic lexical retrieval index with BM25-like scoring.

use std::collections::{BTreeMap, BTreeSet};

use crate::types::{LexicalDocument, LexicalDocumentKind, LexicalSearchHit, ProjectManifest};

const K1: f32 = 1.2;
const B: f32 = 0.75;

/// In-memory deterministic lexical index over files, shards, and symbols.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LexicalIndex {
    documents: Vec<IndexedDocument>,
    document_frequency: BTreeMap<String, usize>,
    average_length: f32,
}

impl LexicalIndex {
    /// Builds a lexical index from typed documents.
    ///
    /// # Arguments
    ///
    /// * `documents` - Searchable documents with deterministic identifiers and
    ///   display text.
    pub fn new(documents: Vec<LexicalDocument>) -> Self {
        let mut indexed_documents = Vec::new();
        let mut document_frequency = BTreeMap::<String, usize>::new();
        let mut total_length = 0usize;

        for document in documents {
            let tokens = tokenize(&document.text);
            let mut frequencies = BTreeMap::<String, usize>::new();
            for token in &tokens {
                let count = frequencies.entry(token.clone()).or_default();
                *count = count.saturating_add(1);
            }
            for token in frequencies.keys().cloned().collect::<BTreeSet<_>>() {
                let count = document_frequency.entry(token).or_default();
                *count = count.saturating_add(1);
            }
            total_length = total_length.saturating_add(tokens.len());
            indexed_documents.push(IndexedDocument { document, frequencies, length: tokens.len() });
        }

        let average_length = if indexed_documents.is_empty() {
            0.0
        } else {
            total_length as f32 / indexed_documents.len() as f32
        };
        Self {
            documents: indexed_documents,
            document_frequency,
            average_length,
        }
    }

    /// Builds a lexical index from a project manifest.
    ///
    /// # Arguments
    ///
    /// * `manifest` - Manifest whose path, symbol, and shard metadata becomes
    ///   searchable.
    pub fn from_manifest(manifest: &ProjectManifest) -> Self {
        Self::new(documents_from_manifest(manifest))
    }

    /// Searches documents with deterministic BM25-like scoring.
    ///
    /// # Arguments
    ///
    /// * `query` - Free-form search text.
    pub fn search(&self, query: &str) -> Vec<LexicalSearchHit> {
        let query_terms = tokenize(query);
        if query_terms.is_empty() || self.documents.is_empty() {
            return Vec::new();
        }
        let mut hits = Vec::new();
        for document in &self.documents {
            let mut score = 0.0f32;
            let mut matched_terms = Vec::new();
            for term in query_terms.iter().collect::<BTreeSet<_>>() {
                let frequency = document
                    .frequencies
                    .get(term.as_str())
                    .copied()
                    .unwrap_or_default();
                if frequency == 0 {
                    continue;
                }
                matched_terms.push((*term).clone());
                let idf = self.idf(term);
                let term_frequency = frequency as f32;
                let length_norm = if self.average_length > 0.0 {
                    (1.0 - B) + B * (document.length as f32 / self.average_length)
                } else {
                    1.0
                };
                score += idf * (term_frequency * (K1 + 1.0)) / (term_frequency + K1 * length_norm);
            }
            if score > 0.0 {
                let kind_weight = match document.document.kind {
                    LexicalDocumentKind::Symbol => 1.25,
                    LexicalDocumentKind::Shard => 1.0,
                    LexicalDocumentKind::File => 0.8,
                };
                hits.push(LexicalSearchHit {
                    id: document.document.id.clone(),
                    path: document.document.path.clone(),
                    symbol: document.document.symbol.clone(),
                    score: score * kind_weight,
                    matched_terms,
                    provenance: document.document.provenance.clone(),
                });
            }
        }
        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.id.cmp(&right.id))
        });
        hits
    }

    fn idf(&self, term: &str) -> f32 {
        let document_count = self.documents.len() as f32;
        let frequency = self
            .document_frequency
            .get(term)
            .copied()
            .unwrap_or_default() as f32;
        ((document_count - frequency + 0.5) / (frequency + 0.5) + 1.0).ln()
    }
}

#[derive(Clone, Debug, PartialEq)]
struct IndexedDocument {
    document: LexicalDocument,
    frequencies: BTreeMap<String, usize>,
    length: usize,
}

/// Converts manifest metadata into deterministic lexical documents.
///
/// # Arguments
///
/// * `manifest` - Manifest to expose to lexical retrieval.
pub fn documents_from_manifest(manifest: &ProjectManifest) -> Vec<LexicalDocument> {
    let mut documents = Vec::new();
    for file in &manifest.files {
        documents.push(LexicalDocument {
            id: file.path.clone(),
            path: file.path.clone(),
            symbol: None,
            kind: LexicalDocumentKind::File,
            text: file.path.replace(['/', '.', '_', '-'], " "),
            provenance: file.provenance.clone(),
        });
    }
    for symbol in &manifest.symbols {
        documents.push(LexicalDocument {
            id: symbol.id.clone(),
            path: symbol.path.clone(),
            symbol: Some(symbol.name.clone()),
            kind: LexicalDocumentKind::Symbol,
            text: format!("{} {:?} {}", symbol.name, symbol.kind, symbol.path),
            provenance: symbol.provenance.clone(),
        });
    }
    for shard in &manifest.shards {
        documents.push(LexicalDocument {
            id: shard.id.clone(),
            path: shard.path.clone(),
            symbol: None,
            kind: LexicalDocumentKind::Shard,
            text: shard_search_text(manifest, shard),
            provenance: shard.provenance.clone(),
        });
    }
    documents.sort_by(|left, right| left.id.cmp(&right.id));
    documents
}

fn shard_search_text(manifest: &ProjectManifest, shard: &crate::types::ShardManifest) -> String {
    let symbols_by_id = manifest
        .symbols
        .iter()
        .map(|symbol| (symbol.id.as_str(), symbol))
        .collect::<BTreeMap<_, _>>();
    let unique_symbol_ids = shard.symbol_ids.iter().collect::<BTreeSet<_>>();
    let mut surfaces = vec![
        "shard".to_string(),
        shard.id.clone(),
        shard.path.clone(),
        format!("start_line_{}", shard.start_line),
        format!("end_line_{}", shard.end_line),
        format!("line_{}", shard.start_line),
        format!("line_{}", shard.end_line),
        format!("lines_{}_{}", shard.start_line, shard.end_line),
        format!("range_{}_{}", shard.start_line, shard.end_line),
    ];

    for symbol_id in unique_symbol_ids {
        surfaces.push(symbol_id.clone());
        if let Some(symbol) = symbols_by_id.get(symbol_id.as_str()) {
            surfaces.push("symbol".to_string());
            surfaces.push(symbol.id.clone());
            surfaces.push(symbol.name.clone());
            surfaces.push(format!("{:?}", symbol.kind));
            surfaces.push(symbol.path.clone());
            surfaces.push(format!("symbol_start_line_{}", symbol.start_line));
            surfaces.push(format!("symbol_end_line_{}", symbol.end_line));
            surfaces.push(format!(
                "symbol_lines_{}_{}",
                symbol.start_line, symbol.end_line
            ));
        } else {
            surfaces.push("missing_symbol".to_string());
        }
    }

    surfaces.join(" ")
}

pub(crate) fn tokenize(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_alphanumeric() && ch != '_')
        .filter(|part| !part.is_empty())
        .flat_map(split_identifier)
        .map(|token| token.to_lowercase())
        .collect()
}

fn split_identifier(part: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut previous_lowercase_or_digit = false;
    for ch in part.chars() {
        if ch == '_' {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            previous_lowercase_or_digit = false;
            continue;
        }
        if ch.is_uppercase() && previous_lowercase_or_digit && !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
        previous_lowercase_or_digit = ch.is_lowercase() || ch.is_ascii_digit();
        current.push(ch);
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::indexer::tests::fixture_project;
    use crate::{ProjectIndexer, SymbolKind};

    #[test]
    fn shard_documents_include_resolved_symbol_terms() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let root_symbol = manifest
            .symbols
            .iter()
            .find(|symbol| symbol.name == "Root")
            .expect("fixture should include Root symbol");

        let actual = LexicalIndex::from_manifest(&manifest).search("Root Struct");
        let expected = true;
        assert_eq!(
            actual.iter().any(|hit| hit.id != root_symbol.id
                && hit.id.starts_with("shard:")
                && hit.path == root_symbol.path),
            expected
        );
        Ok(())
    }

    #[test]
    fn shard_documents_include_missing_symbol_marker_and_deduplicate_symbol_ids() {
        let known_symbol = "symbol:src/lib.rs:Struct:Known".to_string();
        let missing_symbol = "symbol:src/lib.rs:Function:Missing".to_string();
        let setup = ProjectManifest {
            symbols: vec![crate::SymbolNode {
                id: known_symbol.clone(),
                name: "Known".to_string(),
                kind: SymbolKind::Struct,
                path: "src/lib.rs".to_string(),
                parent: None,
                start_line: 5,
                end_line: 9,
                provenance: Default::default(),
            }],
            shards: vec![crate::ShardManifest {
                id: "shard:src/lib.rs:1-10".to_string(),
                path: "src/lib.rs".to_string(),
                start_line: 1,
                end_line: 10,
                content_hash: "must_not_be_indexed".to_string(),
                symbol_ids: vec![
                    missing_symbol.clone(),
                    known_symbol.clone(),
                    known_symbol.clone(),
                ],
                provenance: Default::default(),
            }],
            ..ProjectManifest::default()
        };

        let actual = documents_from_manifest(&setup)
            .into_iter()
            .find(|document| document.kind == LexicalDocumentKind::Shard)
            .map(|document| document.text)
            .expect("fixture should include shard document");
        let expected = "shard shard:src/lib.rs:1-10 src/lib.rs start_line_1 end_line_10 line_1 line_10 lines_1_10 range_1_10 symbol:src/lib.rs:Function:Missing missing_symbol symbol:src/lib.rs:Struct:Known symbol symbol:src/lib.rs:Struct:Known Known Struct src/lib.rs symbol_start_line_5 symbol_end_line_9 symbol_lines_5_9".to_string();
        assert_eq!(actual, expected);
        assert_eq!(actual.contains("must_not_be_indexed"), false);
    }

    #[test]
    fn lexical_index_scores_symbols_above_file_path_matches() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let index = LexicalIndex::from_manifest(&manifest);
        let actual = index.search("Root");
        let expected = Some(SymbolKind::Struct);
        let first_symbol_kind = actual
            .first()
            .and_then(|hit| manifest.symbols.iter().find(|symbol| symbol.id == hit.id))
            .map(|symbol| symbol.kind.clone());
        assert_eq!(first_symbol_kind, expected);
        Ok(())
    }

    #[test]
    fn lexical_index_is_tokenized_not_substring_only() {
        let setup = LexicalIndex::new(vec![LexicalDocument {
            id: "doc".to_string(),
            path: "src/lib.rs".to_string(),
            symbol: None,
            kind: LexicalDocumentKind::File,
            text: "catalog".to_string(),
            provenance: Default::default(),
        }]);
        let actual = setup.search("cat");
        let expected: Vec<LexicalSearchHit> = Vec::new();
        assert_eq!(actual, expected);
    }

    #[test]
    fn lexical_index_uses_term_frequency_for_repeated_tokens() {
        let setup = LexicalIndex::new(vec![
            LexicalDocument {
                id: "frequent".to_string(),
                path: "src/frequent.rs".to_string(),
                symbol: None,
                kind: LexicalDocumentKind::Shard,
                text: "cache cache cache invalidation".to_string(),
                provenance: Default::default(),
            },
            LexicalDocument {
                id: "single".to_string(),
                path: "src/single.rs".to_string(),
                symbol: None,
                kind: LexicalDocumentKind::Shard,
                text: "cache boundary unrelated tokens".to_string(),
                provenance: Default::default(),
            },
        ]);
        let actual = setup.search("cache");
        let expected = vec!["frequent".to_string(), "single".to_string()];
        assert_eq!(
            actual.into_iter().map(|hit| hit.id).collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn lexical_index_uses_idf_to_rank_rare_terms_over_common_only_matches() {
        let setup = LexicalIndex::new(vec![
            LexicalDocument {
                id: "rare".to_string(),
                path: "src/rare.rs".to_string(),
                symbol: None,
                kind: LexicalDocumentKind::Shard,
                text: "context compiler".to_string(),
                provenance: Default::default(),
            },
            LexicalDocument {
                id: "common-a".to_string(),
                path: "src/a.rs".to_string(),
                symbol: None,
                kind: LexicalDocumentKind::Shard,
                text: "context repeated".to_string(),
                provenance: Default::default(),
            },
            LexicalDocument {
                id: "common-b".to_string(),
                path: "src/b.rs".to_string(),
                symbol: None,
                kind: LexicalDocumentKind::Shard,
                text: "context another".to_string(),
                provenance: Default::default(),
            },
        ]);
        let actual = setup.search("context compiler");
        let expected = Some("rare".to_string());
        assert_eq!(actual.first().map(|hit| hit.id.clone()), expected);
    }

    #[test]
    fn lexical_tokenizer_is_unicode_case_insensitive() {
        let setup = LexicalIndex::new(vec![LexicalDocument {
            id: "unicode".to_string(),
            path: "src/unicode.rs".to_string(),
            symbol: None,
            kind: LexicalDocumentKind::Shard,
            text: "Приветствие".to_string(),
            provenance: Default::default(),
        }]);
        let actual = setup
            .search("приветствие")
            .into_iter()
            .map(|hit| hit.id)
            .collect::<Vec<_>>();
        let expected = vec!["unicode".to_string()];
        assert_eq!(actual, expected);
    }
}
