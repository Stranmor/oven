//! Typed ingestion boundary for validated external exact facts.

use std::collections::BTreeSet;

use anyhow::{Result, bail};

use crate::types::{
    EdgeConfidence, ExternalFactBatch, ExternalFactBatchMetadata, ExternalFactIngestionIssue,
    ExternalFactIngestionIssueCode, ExternalFactIngestionReport, ExternalFactSource, ExternalFacts,
    GraphEdge, ProjectManifest, Provenance, SymbolNode, TypedExternalFacts,
    TypedExternalReferenceFact, TypedExternalSymbolFact,
};
use crate::util::{
    edge, edge_sort_key, external_facts_fingerprint, fingerprint, manifest_hash, provenance,
};

/// Imports legacy external facts through a validated synthetic batch.
///
/// # Arguments
///
/// * `manifest` - Manifest updated in place.
/// * `facts` - Legacy external facts to merge.
///
/// # Errors
///
/// Returns an error when batch validation fails or any exact edge endpoint is
/// absent from the manifest graph surface.
pub fn ingest_external_facts(
    manifest: &mut ProjectManifest,
    facts: ExternalFacts,
) -> Result<ExternalFactIngestionReport> {
    ingest_typed_external_facts(manifest, facts.into())
}

/// Imports typed external facts through a validated synthetic batch.
///
/// # Arguments
///
/// * `manifest` - Manifest updated in place.
/// * `facts` - Typed external facts to merge.
///
/// # Errors
///
/// Returns an error when batch validation fails or any exact edge endpoint is
/// absent from the manifest graph surface.
pub fn ingest_typed_external_facts(
    manifest: &mut ProjectManifest,
    facts: TypedExternalFacts,
) -> Result<ExternalFactIngestionReport> {
    let source = batch_source_from_facts(&facts);
    let metadata = ExternalFactBatchMetadata {
        source: source.clone(),
        source_label: source.provenance_label(),
        tool_version: None,
        workspace_root: manifest.root.to_string_lossy().to_string(),
        source_artifact_fingerprint: fingerprint(&legacy_fact_payload_identity(&facts)),
        manifest_hash_input: manifest.manifest_hash.clone(),
        batch_fingerprint: String::new(),
    };
    let batch = ExternalFactBatch::new(metadata, facts);
    ingest_external_fact_batch(manifest, batch)
}

/// Imports one validated external exact fact batch into an in-memory manifest.
///
/// The safe first slice accepts only caller-supplied fixture or precomputed facts;
/// it does not spawn LSP, rust-analyzer, SCIP, or compiler processes.
///
/// # Arguments
///
/// * `manifest` - Manifest updated in place.
/// * `batch` - External exact fact batch with deterministic source metadata.
///
/// # Errors
///
/// Returns an error when batch metadata is incomplete, the manifest baseline is
/// stale, the batch fingerprint is not authoritative, source contracts are not
/// explicit, or an edge endpoint cannot be resolved to a file, symbol, or shard
/// surface used by retrieval.
pub fn ingest_external_fact_batch(
    manifest: &mut ProjectManifest,
    batch: ExternalFactBatch,
) -> Result<ExternalFactIngestionReport> {
    let issues = validate_external_fact_batch(manifest, &batch);
    if !issues.is_empty() {
        bail!("external fact batch rejected: {}", issue_summary(&issues));
    }

    let ExternalFactBatch { metadata, facts } = batch;
    let accepted_symbols = facts.symbols.len();
    let accepted_edges = facts.references.len();

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
                &metadata.source_label,
                &format!("{}:{}", metadata.batch_fingerprint, symbol.id),
            ),
        };
        upsert_symbol(&mut manifest.symbols, node);
    }

    for reference in facts.references {
        manifest
            .edges
            .push(exact_reference_edge(&metadata, reference));
    }
    manifest
        .symbols
        .sort_by(|left, right| left.id.cmp(&right.id));
    manifest.edges.sort_by_key(edge_sort_key);
    let removed_edges = deduplicate_edges(&mut manifest.edges);
    upsert_batch_metadata(&mut manifest.external_fact_batches, metadata.clone());
    manifest.external_fact_batches.sort();
    manifest.external_facts_fingerprint =
        external_facts_fingerprint(&manifest.external_fact_batches);
    manifest.manifest_hash = manifest_hash(
        &manifest.files,
        &manifest.external_fact_batches,
        &manifest.external_facts_fingerprint,
    );

    Ok(ExternalFactIngestionReport {
        accepted_symbols,
        accepted_edges,
        deduplicated_edges: removed_edges,
        batch_metadata: metadata,
    })
}

/// Validates one external exact fact batch against the current manifest graph
/// surface without mutating the manifest.
///
/// # Arguments
///
/// * `manifest` - Manifest whose file, symbol, and shard endpoints are accepted.
/// * `batch` - Candidate external fact batch.
pub fn validate_external_fact_batch(
    manifest: &ProjectManifest,
    batch: &ExternalFactBatch,
) -> Vec<ExternalFactIngestionIssue> {
    let mut issues = Vec::new();
    validate_batch_metadata(manifest, batch, &mut issues);
    validate_source_contract(batch, &mut issues);
    let mut endpoints = manifest_endpoints(manifest);
    for symbol in &batch.facts.symbols {
        if !endpoints.contains(&symbol.path) {
            issues.push(issue(
                ExternalFactIngestionIssueCode::MissingSymbolFileEndpoint,
                Some(symbol.path.clone()),
                format!("symbol_file_missing:{}", symbol.id),
            ));
        }
        endpoints.insert(symbol.id.clone());
    }
    for reference in &batch.facts.references {
        if !endpoints.contains(&reference.from) {
            issues.push(issue(
                ExternalFactIngestionIssueCode::MissingReferenceSourceEndpoint,
                Some(reference.from.clone()),
                format!("reference_source_missing:{}", reference.path),
            ));
        }
        if !endpoints.contains(&reference.to) {
            issues.push(issue(
                ExternalFactIngestionIssueCode::MissingReferenceTargetEndpoint,
                Some(reference.to.clone()),
                format!("reference_target_missing:{}", reference.path),
            ));
        }
    }
    issues.sort_by(|left, right| {
        left.code
            .cmp(&right.code)
            .then_with(|| left.endpoint.cmp(&right.endpoint))
            .then_with(|| left.detail.cmp(&right.detail))
    });
    issues.dedup();
    issues
}

impl ExternalFactBatch {
    /// Builds a batch and fills its deterministic batch fingerprint.
    ///
    /// # Arguments
    ///
    /// * `metadata` - Batch metadata with all fields except `batch_fingerprint`
    ///   already populated.
    /// * `facts` - Exact facts carried by the batch.
    pub fn new(mut metadata: ExternalFactBatchMetadata, facts: TypedExternalFacts) -> Self {
        metadata.batch_fingerprint = external_fact_batch_fingerprint(&metadata, &facts);
        Self { metadata, facts }
    }
}

/// Computes the deterministic fingerprint for one external exact fact batch.
///
/// # Arguments
///
/// * `metadata` - Batch metadata; `batch_fingerprint` is ignored.
/// * `facts` - Fact payload included in the fingerprint.
pub fn external_fact_batch_fingerprint(
    metadata: &ExternalFactBatchMetadata,
    facts: &TypedExternalFacts,
) -> String {
    let mut content = String::new();
    content.push_str(&format!("source:{:?}\n", metadata.source));
    content.push_str(&format!("source_label:{}\n", metadata.source_label));
    content.push_str(&format!(
        "tool_version:{}\n",
        metadata.tool_version.as_deref().unwrap_or_default()
    ));
    content.push_str(&format!("workspace_root:{}\n", metadata.workspace_root));
    content.push_str(&format!(
        "source_artifact_fingerprint:{}\n",
        metadata.source_artifact_fingerprint
    ));
    content.push_str(&format!(
        "manifest_hash_input:{}\n",
        metadata.manifest_hash_input
    ));
    let mut symbols = facts.symbols.clone();
    symbols.sort_by(|left, right| left.id.cmp(&right.id));
    for symbol in symbols {
        content.push_str(&format!(
            "symbol:{}\0{}\0{:?}\0{}\0{}\0{}\0{:?}\n",
            symbol.id,
            symbol.name,
            symbol.kind,
            symbol.path,
            symbol.start_line,
            symbol.end_line,
            symbol.source
        ));
    }
    let mut references = facts.references.clone();
    references.sort_by(|left, right| {
        left.from
            .cmp(&right.from)
            .then_with(|| left.to.cmp(&right.to))
            .then_with(|| left.kind.cmp(&right.kind))
            .then_with(|| left.path.cmp(&right.path))
            .then_with(|| left.start_line.cmp(&right.start_line))
            .then_with(|| left.end_line.cmp(&right.end_line))
            .then_with(|| left.source.cmp(&right.source))
    });
    for reference in references {
        content.push_str(&format!(
            "reference:{}\0{}\0{:?}\0{}\0{:?}\0{:?}\0{:?}\n",
            reference.from,
            reference.to,
            reference.kind,
            reference.path,
            reference.start_line,
            reference.end_line,
            reference.source
        ));
    }
    fingerprint(&content)
}

fn validate_batch_metadata(
    manifest: &ProjectManifest,
    batch: &ExternalFactBatch,
    issues: &mut Vec<ExternalFactIngestionIssue>,
) {
    let metadata = &batch.metadata;
    if metadata.source == ExternalFactSource::Unknown
        || metadata.source_label.trim().is_empty()
        || metadata.workspace_root.trim().is_empty()
        || metadata.source_artifact_fingerprint.trim().is_empty()
        || metadata.manifest_hash_input.trim().is_empty()
    {
        issues.push(issue(
            ExternalFactIngestionIssueCode::IncompleteBatchMetadata,
            None,
            "batch_metadata_incomplete".to_string(),
        ));
    }
    if metadata.manifest_hash_input != manifest.manifest_hash {
        issues.push(issue(
            ExternalFactIngestionIssueCode::ManifestBaselineMismatch,
            Some(metadata.manifest_hash_input.clone()),
            format!(
                "manifest_baseline_mismatch:current={}",
                manifest.manifest_hash
            ),
        ));
    }
    let expected = external_fact_batch_fingerprint(metadata, &batch.facts);
    if metadata.batch_fingerprint != expected {
        issues.push(issue(
            ExternalFactIngestionIssueCode::BatchFingerprintMismatch,
            Some(metadata.batch_fingerprint.clone()),
            format!("batch_fingerprint_expected:{expected}"),
        ));
    }
}

fn validate_source_contract(
    batch: &ExternalFactBatch,
    issues: &mut Vec<ExternalFactIngestionIssue>,
) {
    for symbol in &batch.facts.symbols {
        if symbol.source != batch.metadata.source || symbol.source == ExternalFactSource::Unknown {
            issues.push(issue(
                ExternalFactIngestionIssueCode::InvalidExactSourceContract,
                Some(symbol.id.clone()),
                format!(
                    "symbol_source_contract_invalid:{}",
                    batch.metadata.source_label
                ),
            ));
        }
    }
    for reference in &batch.facts.references {
        if reference.source != batch.metadata.source
            || reference.source == ExternalFactSource::Unknown
        {
            issues.push(issue(
                ExternalFactIngestionIssueCode::InvalidExactSourceContract,
                Some(format!("{}->{}", reference.from, reference.to)),
                format!(
                    "reference_source_contract_invalid:{}",
                    batch.metadata.source_label
                ),
            ));
        }
    }
}

fn manifest_endpoints(manifest: &ProjectManifest) -> BTreeSet<String> {
    manifest
        .files
        .iter()
        .map(|file| file.path.clone())
        .chain(manifest.symbols.iter().map(|symbol| symbol.id.clone()))
        .chain(manifest.shards.iter().map(|shard| shard.id.clone()))
        .collect()
}

fn exact_reference_edge(
    metadata: &ExternalFactBatchMetadata,
    reference: TypedExternalReferenceFact,
) -> GraphEdge {
    edge(
        &reference.from,
        &reference.to,
        reference.kind,
        1.0,
        EdgeConfidence::ExactCompiler,
        Provenance {
            path: reference.path.clone(),
            start_line: reference.start_line,
            end_line: reference.end_line,
            source: metadata.source_label.clone(),
            fingerprint: metadata.batch_fingerprint.clone(),
        },
    )
}

fn upsert_symbol(symbols: &mut Vec<SymbolNode>, incoming: SymbolNode) {
    if let Some(existing) = symbols.iter_mut().find(|symbol| symbol.id == incoming.id) {
        *existing = incoming;
    } else {
        symbols.push(incoming);
    }
}

fn upsert_batch_metadata(
    batches: &mut Vec<ExternalFactBatchMetadata>,
    incoming: ExternalFactBatchMetadata,
) {
    if let Some(existing) = batches
        .iter_mut()
        .find(|batch| batch.batch_fingerprint == incoming.batch_fingerprint)
    {
        *existing = incoming;
    } else {
        batches.push(incoming);
    }
}

fn deduplicate_edges(edges: &mut Vec<GraphEdge>) -> usize {
    let before = edges.len();
    let mut seen = BTreeSet::new();
    edges.retain(|edge| {
        seen.insert((
            edge.from.clone(),
            edge.to.clone(),
            edge.kind.clone(),
            edge.confidence_kind.clone(),
            edge.provenance.source.clone(),
            edge.provenance.fingerprint.clone(),
        ))
    });
    before.saturating_sub(edges.len())
}

fn batch_source_from_facts(facts: &TypedExternalFacts) -> ExternalFactSource {
    facts
        .symbols
        .first()
        .map(|symbol| symbol.source.clone())
        .or_else(|| {
            facts
                .references
                .first()
                .map(|reference| reference.source.clone())
        })
        .unwrap_or(ExternalFactSource::Unknown)
}

fn legacy_fact_payload_identity(facts: &TypedExternalFacts) -> String {
    format!("{:?}:{:?}", facts.symbols, facts.references)
}

fn issue(
    code: ExternalFactIngestionIssueCode,
    endpoint: Option<String>,
    detail: String,
) -> ExternalFactIngestionIssue {
    ExternalFactIngestionIssue { code, endpoint, detail }
}

fn issue_summary(issues: &[ExternalFactIngestionIssue]) -> String {
    issues
        .iter()
        .map(|issue| format!("{:?}:{}", issue.code, issue.detail))
        .collect::<Vec<_>>()
        .join(",")
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
    use anyhow::Result;
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::indexer::tests::fixture_project;
    use crate::{
        ExternalFactSource, ExternalReferenceFact, GraphEdgeKind, ProjectIndexer, RetrievalQuery,
        SymbolKind, retrieve,
    };

    fn fixture_batch(
        manifest: &ProjectManifest,
        source_artifact_fingerprint: &str,
    ) -> ExternalFactBatch {
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
            references: vec![
                TypedExternalReferenceFact {
                    from: "lsp:src/lib.rs:Root::new".to_string(),
                    to: "symbol:src/lib.rs:Struct:Root".to_string(),
                    kind: GraphEdgeKind::References,
                    path: "src/lib.rs".to_string(),
                    start_line: Some(10),
                    end_line: Some(10),
                    source: ExternalFactSource::Lsp,
                },
                TypedExternalReferenceFact {
                    from: "lsp:src/lib.rs:Root::new".to_string(),
                    to: "symbol:src/lib.rs:Struct:Root".to_string(),
                    kind: GraphEdgeKind::References,
                    path: "src/lib.rs".to_string(),
                    start_line: Some(10),
                    end_line: Some(10),
                    source: ExternalFactSource::Lsp,
                },
            ],
        };
        ExternalFactBatch::new(
            ExternalFactBatchMetadata {
                source: ExternalFactSource::Lsp,
                source_label: "rust-analyzer".to_string(),
                tool_version: Some("fixture-1".to_string()),
                workspace_root: manifest.root.to_string_lossy().to_string(),
                source_artifact_fingerprint: source_artifact_fingerprint.to_string(),
                manifest_hash_input: manifest.manifest_hash.clone(),
                batch_fingerprint: String::new(),
            },
            facts,
        )
    }

    #[test]
    fn external_batch_metadata_fingerprint_is_deterministic_and_changes_on_snapshot() -> Result<()>
    {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let left = fixture_batch(&manifest, &fingerprint("snapshot-a"));
        let right = fixture_batch(&manifest, &fingerprint("snapshot-a"));
        let changed = fixture_batch(&manifest, &fingerprint("snapshot-b"));

        assert_eq!(
            left.metadata.batch_fingerprint,
            right.metadata.batch_fingerprint
        );
        assert_ne!(
            left.metadata.batch_fingerprint,
            changed.metadata.batch_fingerprint
        );
        Ok(())
    }

    #[test]
    fn accepted_batch_changes_manifest_identity_and_deduplicates_edges() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let mut manifest = setup.index()?;
        let before = manifest.manifest_hash.clone();
        let batch = fixture_batch(&manifest, &fingerprint("snapshot-a"));

        let actual = ingest_external_fact_batch(&mut manifest, batch)?;

        assert_ne!(manifest.manifest_hash, before);
        assert_eq!(actual.accepted_symbols, 1usize);
        assert_eq!(actual.accepted_edges, 2usize);
        assert_eq!(actual.deduplicated_edges, 1usize);
        assert_eq!(manifest.external_fact_batches.len(), 1usize);
        assert_eq!(manifest.external_facts_fingerprint.len(), 64usize);
        Ok(())
    }

    #[test]
    fn different_accepted_external_facts_produce_different_manifest_identities() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let mut left = setup.index()?;
        let mut right = setup.index()?;
        let left_batch = fixture_batch(&left, &fingerprint("snapshot-a"));
        let right_batch = fixture_batch(&right, &fingerprint("snapshot-b"));

        ingest_external_fact_batch(&mut left, left_batch)?;
        ingest_external_fact_batch(&mut right, right_batch)?;

        assert_ne!(
            left.external_facts_fingerprint,
            right.external_facts_fingerprint
        );
        assert_ne!(left.manifest_hash, right.manifest_hash);
        Ok(())
    }

    #[test]
    fn accepted_exact_edge_resolves_and_retrieval_graph_expansion_surfaces_neighbor() -> Result<()>
    {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let mut manifest = setup.index()?;
        let batch = fixture_batch(&manifest, &fingerprint("snapshot-a"));
        ingest_external_fact_batch(&mut manifest, batch)?;
        let query = RetrievalQuery {
            text: None,
            path: None,
            path_prefix: None,
            symbol: Some("new".to_string()),
            limit: 5,
            include_graph_expansion: true,
        };

        let actual = retrieve(&manifest, &query);
        let actual_ids = actual
            .into_iter()
            .map(|result| result.id)
            .collect::<BTreeSet<_>>();
        let expected = true;

        assert_eq!(
            actual_ids.contains("symbol:src/lib.rs:Struct:Root"),
            expected
        );
        Ok(())
    }

    #[test]
    fn unresolved_external_endpoint_is_reported_and_rejected() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let mut manifest = setup.index()?;
        let mut batch = fixture_batch(&manifest, &fingerprint("snapshot-a"));
        batch.facts.references[0].to = "missing:symbol".to_string();
        batch.metadata.batch_fingerprint =
            external_fact_batch_fingerprint(&batch.metadata, &batch.facts);
        let before_edges = manifest.edges.len();

        let issues = validate_external_fact_batch(&manifest, &batch);
        let actual = ingest_external_fact_batch(&mut manifest, batch).is_err();

        assert_eq!(actual, true);
        assert_eq!(manifest.edges.len(), before_edges);
        assert_eq!(
            issues.iter().any(|issue| issue.code
                == ExternalFactIngestionIssueCode::MissingReferenceTargetEndpoint),
            true
        );
        Ok(())
    }

    #[test]
    fn persisted_manifest_readback_preserves_external_batch_metadata() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let mut manifest = setup.index()?;
        let batch = fixture_batch(&manifest, &fingerprint("snapshot-a"));
        let expected = batch.metadata.clone();
        ingest_external_fact_batch(&mut manifest, batch)?;
        setup.write_manifest(&manifest)?;

        let actual = setup.read_manifest()?;

        assert_eq!(actual.external_fact_batches, vec![expected]);
        assert_eq!(
            actual.external_facts_fingerprint,
            manifest.external_facts_fingerprint
        );
        assert_eq!(actual.manifest_hash, manifest.manifest_hash);
        Ok(())
    }

    #[test]
    fn heuristic_edges_remain_heuristic_after_external_ingestion() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let mut manifest = setup.index()?;
        let heuristic_before = manifest
            .edges
            .iter()
            .filter(|edge| edge.confidence_kind == EdgeConfidence::HeuristicHigh)
            .count();
        let batch = fixture_batch(&manifest, &fingerprint("snapshot-a"));
        ingest_external_fact_batch(&mut manifest, batch)?;
        let heuristic_after = manifest
            .edges
            .iter()
            .filter(|edge| edge.confidence_kind == EdgeConfidence::HeuristicHigh)
            .count();
        let exact_after = manifest
            .edges
            .iter()
            .filter(|edge| edge.confidence_kind == EdgeConfidence::ExactCompiler)
            .count();

        assert_eq!(heuristic_after, heuristic_before);
        assert_eq!(exact_after > 0, true);
        Ok(())
    }

    #[test]
    fn legacy_external_facts_reject_unresolved_endpoints() -> Result<()> {
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

        let actual = ingest_external_facts(&mut manifest, facts).is_err();
        let expected = true;
        assert_eq!(actual, expected);
        Ok(())
    }
}
