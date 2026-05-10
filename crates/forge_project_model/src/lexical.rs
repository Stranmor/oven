//! Deterministic lexical retrieval index with BM25-like scoring.

use crate::types::{LexicalDocument, LexicalDocumentKind, LexicalSearchHit, ProjectManifest};
use std::collections::{BTreeMap, BTreeSet};

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
    /// * `documents` - Searchable documents with deterministic identifiers and display text.
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
    /// * `manifest` - Manifest whose path, symbol, and shard metadata becomes searchable.
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
            text: format!("{} {}", shard.path, shard.symbol_ids.join(" ")),
            provenance: shard.provenance.clone(),
        });
    }
    documents.sort_by(|left, right| left.id.cmp(&right.id));
    documents
}

pub(crate) fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|ch: char| !ch.is_alphanumeric() && ch != '_')
        .filter(|part| !part.is_empty())
        .flat_map(split_identifier)
        .collect()
}

fn split_identifier(part: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in part.chars() {
        if ch == '_' {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            continue;
        }
        current.push(ch);
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::tests::fixture_project;
    use crate::{ProjectIndexer, SymbolKind};
    use anyhow::Result;
    use pretty_assertions::assert_eq;

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
}
