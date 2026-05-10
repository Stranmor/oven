//! Typed ingestion boundary for external compiler, LSP, or SCIP facts.

use crate::types::{
    EdgeConfidence, ExternalFactSource, ExternalFacts, GraphEdge, ProjectManifest, Provenance,
    SymbolNode, TypedExternalFacts, TypedExternalReferenceFact, TypedExternalSymbolFact,
};
use crate::util::{edge, edge_sort_key, fingerprint, provenance};
use std::collections::BTreeSet;

/// Imports legacy external compiler/LSP/SCIP facts into an in-memory project manifest.
///
/// # Arguments
///
/// * `manifest` - Manifest updated in place.
/// * `facts` - Legacy external facts to merge.
pub fn ingest_external_facts(manifest: &mut ProjectManifest, facts: ExternalFacts) {
    ingest_typed_external_facts(manifest, facts.into());
}

/// Imports typed external compiler/LSP/SCIP facts into an in-memory project manifest.
///
/// # Arguments
///
/// * `manifest` - Manifest updated in place.
/// * `facts` - Typed external facts to merge.
pub fn ingest_typed_external_facts(manifest: &mut ProjectManifest, facts: TypedExternalFacts) {
    for symbol in facts.symbols {
        let node = SymbolNode {
            id: symbol.id.clone(),
            name: symbol.name,
            kind: symbol.kind,
            path: symbol.path.clone(),
            parent: None,
            start_line: symbol.start_line,
            end_line: symbol.end_line,
            provenance: provenance(
                &symbol.path,
                Some(symbol.start_line),
                Some(symbol.end_line),
                &symbol.source.provenance_label(),
                &symbol.id,
            ),
        };
        upsert_symbol(&mut manifest.symbols, node);
    }

    for reference in facts.references {
        let graph_edge = edge(
            &reference.from,
            &reference.to,
            reference.kind,
            1.0,
            EdgeConfidence::ExactCompiler,
            Provenance {
                path: reference.path.clone(),
                start_line: reference.start_line,
                end_line: reference.end_line,
                source: reference.source.provenance_label(),
                fingerprint: fingerprint(&format!(
                    "{}:{}:{}:{:?}:{:?}",
                    reference.from,
                    reference.to,
                    reference.path,
                    reference.start_line,
                    reference.end_line
                )),
            },
        );
        manifest.edges.push(graph_edge);
    }
    manifest
        .symbols
        .sort_by(|left, right| left.id.cmp(&right.id));
    manifest.edges.sort_by_key(edge_sort_key);
    deduplicate_edges(&mut manifest.edges);
}

fn upsert_symbol(symbols: &mut Vec<SymbolNode>, incoming: SymbolNode) {
    if let Some(existing) = symbols.iter_mut().find(|symbol| symbol.id == incoming.id) {
        *existing = incoming;
    } else {
        symbols.push(incoming);
    }
}

fn deduplicate_edges(edges: &mut Vec<GraphEdge>) {
    let mut seen = BTreeSet::new();
    edges.retain(|edge| {
        seen.insert((
            edge.from.clone(),
            edge.to.clone(),
            edge.kind.clone(),
            edge.confidence_kind.clone(),
            edge.provenance.source.clone(),
        ))
    });
}

impl From<ExternalFacts> for TypedExternalFacts {
    fn from(facts: ExternalFacts) -> Self {
        Self {
            symbols: facts.symbols.into_iter().map(Into::into).collect(),
            references: facts.references.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<crate::types::ExternalSymbolFact> for TypedExternalSymbolFact {
    fn from(fact: crate::types::ExternalSymbolFact) -> Self {
        Self {
            id: fact.id,
            name: fact.name,
            kind: fact.kind,
            path: fact.path,
            start_line: fact.start_line,
            end_line: fact.end_line,
            source: ExternalFactSource::from_label(&fact.source),
        }
    }
}

impl From<crate::types::ExternalReferenceFact> for TypedExternalReferenceFact {
    fn from(fact: crate::types::ExternalReferenceFact) -> Self {
        Self {
            from: fact.from,
            to: fact.to,
            kind: fact.kind,
            path: fact.path,
            start_line: fact.start_line,
            end_line: fact.end_line,
            source: ExternalFactSource::from_label(&fact.source),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::tests::fixture_project;
    use crate::{
        ExternalFactSource, ExternalReferenceFact, GraphEdgeKind, ProjectIndexer, SymbolKind,
        TypedExternalFacts, TypedExternalReferenceFact, TypedExternalSymbolFact,
    };
    use anyhow::Result;
    use pretty_assertions::assert_eq;

    #[test]
    fn imports_typed_lsp_and_scip_facts_as_exact_compiler_edges() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let mut manifest = setup.index()?;
        let facts = TypedExternalFacts {
            symbols: vec![TypedExternalSymbolFact {
                id: "lsp:src/lib.rs:Root::new".to_string(),
                name: "new".to_string(),
                kind: SymbolKind::Method,
                path: "src/lib.rs".to_string(),
                start_line: 10,
                end_line: 12,
                source: ExternalFactSource::Lsp,
            }],
            references: vec![TypedExternalReferenceFact {
                from: "lsp:src/lib.rs:Root::new".to_string(),
                to: "symbol:src/lib.rs:Struct:Root".to_string(),
                kind: GraphEdgeKind::References,
                path: "src/lib.rs".to_string(),
                start_line: Some(10),
                end_line: Some(10),
                source: ExternalFactSource::Scip,
            }],
        };
        ingest_typed_external_facts(&mut manifest, facts);
        let actual = manifest
            .edges
            .iter()
            .find(|edge| edge.from == "lsp:src/lib.rs:Root::new")
            .map(|edge| (edge.confidence_kind.clone(), edge.provenance.source.clone()));
        let expected = Some((EdgeConfidence::ExactCompiler, "scip".to_string()));
        assert_eq!(actual, expected);
        assert_eq!(
            manifest
                .symbols
                .iter()
                .find(|symbol| symbol.id == "lsp:src/lib.rs:Root::new")
                .map(|symbol| symbol.provenance.source.clone()),
            Some("lsp".to_string())
        );
        Ok(())
    }

    #[test]
    fn legacy_external_facts_preserve_unknown_source_labels() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let mut manifest = setup.index()?;
        let facts = ExternalFacts {
            symbols: Vec::new(),
            references: vec![ExternalReferenceFact {
                from: "legacy:custom:caller".to_string(),
                to: "legacy:custom:callee".to_string(),
                kind: GraphEdgeKind::Calls,
                path: "src/lib.rs".to_string(),
                start_line: Some(1),
                end_line: Some(1),
                source: "bespoke-indexer".to_string(),
            }],
        };
        ingest_external_facts(&mut manifest, facts);
        let actual = manifest
            .edges
            .iter()
            .find(|edge| edge.from == "legacy:custom:caller")
            .map(|edge| edge.provenance.source.clone());
        let expected = Some("bespoke-indexer".to_string());
        assert_eq!(actual, expected);
        Ok(())
    }
}
