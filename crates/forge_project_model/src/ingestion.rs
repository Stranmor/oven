//! Typed ingestion boundary for external compiler, LSP, or SCIP facts.

use crate::types::{
    EdgeConfidence, ExternalFacts, GraphEdge, ProjectManifest, Provenance, SymbolNode,
};
use crate::util::{edge, edge_sort_key, fingerprint, provenance};
use std::collections::BTreeSet;

/// Imports external compiler/LSP/SCIP facts into an in-memory project manifest.
///
/// # Arguments
///
/// * `manifest` - Manifest updated in place.
/// * `facts` - Typed external facts to merge.
pub fn ingest_external_facts(manifest: &mut ProjectManifest, facts: ExternalFacts) {
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
                &symbol.source,
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
                source: reference.source.clone(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::tests::fixture_project;
    use crate::{
        ExternalReferenceFact, ExternalSymbolFact, GraphEdgeKind, ProjectIndexer, SymbolKind,
    };
    use anyhow::Result;
    use pretty_assertions::assert_eq;

    #[test]
    fn imports_external_symbols_and_exact_compiler_edges() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let mut manifest = setup.index()?;
        let facts = ExternalFacts {
            symbols: vec![ExternalSymbolFact {
                id: "scip:src/lib.rs:Root::run".to_string(),
                name: "run".to_string(),
                kind: SymbolKind::Method,
                path: "src/lib.rs".to_string(),
                start_line: 10,
                end_line: 12,
                source: "scip".to_string(),
            }],
            references: vec![ExternalReferenceFact {
                from: "scip:src/lib.rs:Root::run".to_string(),
                to: "symbol:src/lib.rs:Function:helper".to_string(),
                kind: GraphEdgeKind::Calls,
                path: "src/lib.rs".to_string(),
                start_line: Some(11),
                end_line: Some(11),
                source: "scip".to_string(),
            }],
        };
        ingest_external_facts(&mut manifest, facts);
        let actual = manifest
            .edges
            .iter()
            .find(|edge| edge.from == "scip:src/lib.rs:Root::run")
            .map(|edge| edge.confidence_kind.clone());
        let expected = Some(EdgeConfidence::ExactCompiler);
        assert_eq!(actual, expected);
        assert_eq!(
            manifest
                .symbols
                .iter()
                .any(|symbol| symbol.id == "scip:src/lib.rs:Root::run"),
            true
        );
        Ok(())
    }
}
