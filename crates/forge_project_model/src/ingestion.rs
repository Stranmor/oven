//! Typed ingestion boundary for validated external exact facts.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

use crate::types::{
    EdgeConfidence, ExternalFactArtifactIngestionReport, ExternalFactArtifactReport,
    ExternalFactBatch, ExternalFactBatchMetadata, ExternalFactIngestionIssue,
    ExternalFactIngestionIssueCode, ExternalFactIngestionReport, ExternalFactSource, ExternalFacts,
    GraphEdge, ProjectManifest, Provenance, SymbolNode, TypedExternalFacts,
    TypedExternalReferenceFact, TypedExternalSymbolFact,
};
use crate::util::{
    edge, edge_sort_key, external_facts_fingerprint, fingerprint, manifest_hash, provenance,
};

const MAX_EXTERNAL_FACT_ARTIFACT_BYTES: u64 = 8 * 1024 * 1024;

/// Writes one bounded external exact-fact artifact into a model-dir scoped store.
///
/// This helper is a serialization scaffold only: it accepts an already typed
/// batch, computes deterministic artifact and batch fingerprints, validates the
/// result against a frozen manifest, writes pretty JSON atomically, and verifies
/// readback before returning. It never spawns producer tools or reads raw tool
/// output.
///
/// # Arguments
///
/// * `model_dir` - Project-model directory that owns the `external_facts` store.
/// * `frozen_manifest` - Manifest baseline the facts were produced against.
/// * `batch` - Already typed external facts and source metadata to serialize.
///
/// # Errors
///
/// Returns an error when source metadata is not explicit, validation fails, the
/// artifact cannot be written atomically, or readback validation fails.
pub fn write_external_fact_artifact(
    model_dir: &Path,
    frozen_manifest: &ProjectManifest,
    batch: ExternalFactBatch,
) -> Result<PathBuf> {
    let batch = prepared_external_fact_artifact_batch(frozen_manifest, batch)?;
    let issues = validate_external_fact_batch(frozen_manifest, &batch);
    if !issues.is_empty() {
        bail!(
            "external fact artifact rejected before write: {}",
            issue_summary(&issues)
        );
    }

    let directory = model_dir.join("external_facts");
    fs::create_dir_all(&directory)?;
    let filename = external_fact_artifact_filename(&batch);
    let path = directory.join(filename);
    let temp_path = directory.join(format!(".{}.tmp", batch.metadata.batch_fingerprint));
    let json = serde_json::to_string_pretty(&batch)?;
    write_json_atomically(&temp_path, &path, json.as_bytes())?;

    let readback_json = fs::read_to_string(&path)?;
    let readback: ExternalFactBatch = serde_json::from_str(&readback_json)?;
    let readback_issues = validate_external_fact_batch(frozen_manifest, &readback);
    if !readback_issues.is_empty() || readback != batch {
        bail!(
            "external fact artifact readback rejected: {}",
            issue_summary(&readback_issues)
        );
    }
    Ok(path)
}

/// Prepares deterministic metadata for one external fact artifact without writing it.
///
/// # Arguments
///
/// * `frozen_manifest` - Manifest baseline used for workspace and hash identity.
/// * `batch` - Batch whose source and fact payload are already typed.
///
/// # Errors
///
/// Returns an error when the artifact source is unknown or uses an implicit label.
pub fn prepared_external_fact_artifact_batch(
    frozen_manifest: &ProjectManifest,
    mut batch: ExternalFactBatch,
) -> Result<ExternalFactBatch> {
    if batch.metadata.source == ExternalFactSource::Unknown {
        bail!("external fact artifact source must be explicit");
    }
    if batch.metadata.source_label.trim().is_empty()
        || batch.metadata.source_label == ExternalFactSource::Unknown.provenance_label()
    {
        bail!("external fact artifact source label must be explicit");
    }
    batch.metadata.workspace_root = frozen_manifest.root.to_string_lossy().to_string();
    batch.metadata.manifest_hash_input = frozen_manifest.manifest_hash.clone();
    batch.metadata.source_artifact_fingerprint.clear();
    batch.metadata.batch_fingerprint.clear();
    batch.metadata.source_artifact_fingerprint = external_fact_artifact_fingerprint(&batch);
    batch.metadata.batch_fingerprint =
        external_fact_batch_fingerprint(&batch.metadata, &batch.facts);
    Ok(batch)
}

/// Loads and applies durable precomputed external exact-fact artifacts from a
/// model-dir scoped store.
///
/// The loader is intentionally a consumer only: it never spawns language
/// servers, SCIP, Cargo, rustc, or any other producer. All candidates are parsed
/// and validated against the frozen base manifest before any accepted batch is
/// applied, preventing mutable-baseline acceptance bugs.
///
/// # Arguments
///
/// * `manifest` - Frozen-base manifest that accepted artifacts may extend.
/// * `external_facts_dir` - Model-dir scoped `external_facts` storage path.
///
/// # Errors
///
/// Returns an error only when the artifact store itself cannot be listed or a
/// previously accepted batch fails to apply after successful dry-run validation.
pub fn ingest_external_fact_artifacts(
    manifest: &mut ProjectManifest,
    external_facts_dir: &Path,
) -> Result<ExternalFactArtifactIngestionReport> {
    if !external_facts_dir.exists() {
        return Ok(ExternalFactArtifactIngestionReport {
            store_path: "external_facts".to_string(),
            inspected_artifacts: 0,
            accepted_artifacts: 0,
            artifacts: Vec::new(),
            accepted_batches: Vec::new(),
        });
    }

    let frozen_base = manifest.clone();
    let mut candidates = list_artifact_candidates(external_facts_dir)?;
    let mut accepted = Vec::<AcceptedArtifact>::new();
    let mut reports = Vec::<ExternalFactArtifactReport>::new();

    for candidate in &mut candidates {
        let report = parse_artifact_candidate(candidate, &frozen_base);
        if let Some(batch) = report.batch {
            accepted
                .push(AcceptedArtifact { artifact_path: candidate.relative_path.clone(), batch });
        }
        reports.push(ExternalFactArtifactReport {
            artifact_path: candidate.relative_path.clone(),
            artifact_fingerprint: report.artifact_fingerprint,
            accepted_batch: None,
            issues: report.issues,
        });
    }

    let duplicate_issues = duplicate_batch_issues(&accepted);
    for (artifact_path, issue) in duplicate_issues {
        if let Some(report) = reports
            .iter_mut()
            .find(|report| report.artifact_path == artifact_path)
        {
            report.issues.push(issue);
        }
    }

    let mut seen_batch_fingerprints = BTreeSet::new();
    let mut seen_symbol_ids = BTreeSet::new();
    let mut accepted_filtered = Vec::new();
    for artifact in accepted {
        let has_report_issues = reports
            .iter()
            .find(|report| report.artifact_path == artifact.artifact_path)
            .is_some_and(|report| !report.issues.is_empty());
        if has_report_issues {
            continue;
        }

        let duplicate_symbols = artifact
            .batch
            .facts
            .symbols
            .iter()
            .filter(|symbol| seen_symbol_ids.contains(&symbol.id))
            .map(|symbol| symbol.id.clone())
            .collect::<Vec<_>>();
        if !duplicate_symbols.is_empty() {
            if let Some(report) = reports
                .iter_mut()
                .find(|report| report.artifact_path == artifact.artifact_path)
            {
                for symbol_id in duplicate_symbols {
                    report.issues.push(issue(
                        ExternalFactIngestionIssueCode::DuplicateAcceptedSymbolId,
                        Some(symbol_id.clone()),
                        format!("duplicate_accepted_symbol_id:{symbol_id}"),
                    ));
                }
                sort_issues(&mut report.issues);
            }
            continue;
        }

        if !seen_batch_fingerprints.insert(artifact.batch.metadata.batch_fingerprint.clone()) {
            continue;
        }
        for symbol in &artifact.batch.facts.symbols {
            seen_symbol_ids.insert(symbol.id.clone());
        }
        accepted_filtered.push(artifact);
    }

    accepted_filtered.sort_by(|left, right| {
        left.batch
            .metadata
            .batch_fingerprint
            .cmp(&right.batch.metadata.batch_fingerprint)
            .then_with(|| {
                left.batch
                    .metadata
                    .source_label
                    .cmp(&right.batch.metadata.source_label)
            })
            .then_with(|| {
                left.batch
                    .metadata
                    .source_artifact_fingerprint
                    .cmp(&right.batch.metadata.source_artifact_fingerprint)
            })
            .then_with(|| left.artifact_path.cmp(&right.artifact_path))
    });

    let mut accepted_batches = Vec::new();
    for artifact in accepted_filtered {
        let metadata = artifact.batch.metadata.clone();
        let ingestion = apply_external_fact_batch(manifest, artifact.batch);
        if let Some(report) = reports
            .iter_mut()
            .find(|report| report.artifact_path == artifact.artifact_path)
        {
            report.accepted_batch = Some(metadata);
        }
        accepted_batches.push(ingestion);
    }

    reports.sort_by(|left, right| left.artifact_path.cmp(&right.artifact_path));
    Ok(ExternalFactArtifactIngestionReport {
        store_path: "external_facts".to_string(),
        inspected_artifacts: reports.len(),
        accepted_artifacts: accepted_batches.len(),
        artifacts: reports,
        accepted_batches,
    })
}

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

    Ok(apply_external_fact_batch(manifest, batch))
}

fn apply_external_fact_batch(
    manifest: &mut ProjectManifest,
    batch: ExternalFactBatch,
) -> ExternalFactIngestionReport {
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

    ExternalFactIngestionReport {
        accepted_symbols,
        accepted_edges,
        deduplicated_edges: removed_edges,
        batch_metadata: metadata,
    }
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
    let file_endpoints = manifest_file_endpoints(manifest);
    let file_line_counts = manifest_file_line_counts(manifest);
    let manifest_symbol_ids = manifest_symbol_ids(manifest);
    let mut batch_symbol_ids = BTreeSet::new();
    for symbol in &batch.facts.symbols {
        if !batch_symbol_ids.insert(symbol.id.clone()) {
            issues.push(issue(
                ExternalFactIngestionIssueCode::DuplicateSymbolId,
                Some(symbol.id.clone()),
                format!("duplicate_symbol_id:{}", symbol.id),
            ));
        }
        if manifest_symbol_ids.contains(&symbol.id) {
            issues.push(issue(
                ExternalFactIngestionIssueCode::ConflictingManifestSymbolId,
                Some(symbol.id.clone()),
                format!("manifest_symbol_id_conflict:{}", symbol.id),
            ));
        }
        if !file_endpoints.contains(&symbol.path) {
            issues.push(issue(
                ExternalFactIngestionIssueCode::MissingSymbolFileEndpoint,
                Some(symbol.path.clone()),
                format!("symbol_file_missing:{}", symbol.id),
            ));
        }
        if !valid_required_line_range(symbol.start_line, symbol.end_line)
            || !required_line_range_within_file(
                &file_line_counts,
                &symbol.path,
                symbol.start_line,
                symbol.end_line,
            )
        {
            issues.push(issue(
                ExternalFactIngestionIssueCode::InvalidSymbolLineRange,
                Some(symbol.id.clone()),
                format!(
                    "symbol_line_range_invalid:{}:{}-{}",
                    symbol.id, symbol.start_line, symbol.end_line
                ),
            ));
        }
        endpoints.insert(symbol.id.clone());
    }
    for reference in &batch.facts.references {
        if !valid_optional_line_range(reference.start_line, reference.end_line)
            || !optional_line_range_within_file(
                &file_line_counts,
                &reference.path,
                reference.start_line,
                reference.end_line,
            )
        {
            issues.push(issue(
                ExternalFactIngestionIssueCode::InvalidReferenceLineRange,
                Some(reference.path.clone()),
                format!(
                    "reference_line_range_invalid:{}->{}:{:?}-{:?}",
                    reference.from, reference.to, reference.start_line, reference.end_line
                ),
            ));
        }
        if !file_endpoints.contains(&reference.path) {
            issues.push(issue(
                ExternalFactIngestionIssueCode::MissingReferenceFileEndpoint,
                Some(reference.path.clone()),
                format!(
                    "reference_file_missing:{}->{}",
                    reference.from, reference.to
                ),
            ));
        }
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

fn valid_required_line_range(start_line: u32, end_line: u32) -> bool {
    start_line >= 1 && end_line >= start_line
}

fn valid_optional_line_range(start_line: Option<u32>, end_line: Option<u32>) -> bool {
    match (start_line, end_line) {
        (Some(start_line), Some(end_line)) => valid_required_line_range(start_line, end_line),
        (None, None) => true,
        _ => false,
    }
}

fn required_line_range_within_file(
    file_line_counts: &BTreeMap<String, u32>,
    path: &str,
    start_line: u32,
    end_line: u32,
) -> bool {
    file_line_counts
        .get(path)
        .is_none_or(|line_count| start_line <= *line_count && end_line <= *line_count)
}

fn optional_line_range_within_file(
    file_line_counts: &BTreeMap<String, u32>,
    path: &str,
    start_line: Option<u32>,
    end_line: Option<u32>,
) -> bool {
    match (start_line, end_line) {
        (Some(start_line), Some(end_line)) => {
            required_line_range_within_file(file_line_counts, path, start_line, end_line)
        }
        (None, None) => true,
        _ => false,
    }
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

/// Computes the deterministic source-artifact fingerprint for one external fact artifact.
///
/// The fingerprint intentionally excludes both stored fingerprint fields so a
/// producer can write placeholder values first and callers can recompute the
/// authoritative identifiers from the decoded artifact payload.
///
/// # Arguments
///
/// * `batch` - Decoded artifact batch payload.
pub fn external_fact_artifact_fingerprint(batch: &ExternalFactBatch) -> String {
    let mut metadata = batch.metadata.clone();
    metadata.source_artifact_fingerprint.clear();
    metadata.batch_fingerprint.clear();
    let canonical = serde_json::to_string(&(&metadata, &batch.facts))
        .expect("external fact artifact fingerprint serialization should be infallible");
    fingerprint(&canonical)
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

fn write_json_atomically(temp_path: &Path, path: &Path, bytes: &[u8]) -> Result<()> {
    if temp_path.exists() {
        let _ = fs::remove_file(temp_path);
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(temp_path)?;
    if let Err(error) = file.write_all(bytes).and_then(|_| file.sync_all()) {
        let _ = fs::remove_file(temp_path);
        return Err(error.into());
    }
    drop(file);
    if let Err(error) = fs::rename(temp_path, path) {
        let _ = fs::remove_file(temp_path);
        return Err(error.into());
    }
    if let Some(parent) = path.parent()
        && let Ok(directory) = File::open(parent)
    {
        directory.sync_all()?;
    }
    Ok(())
}

fn external_fact_artifact_filename(batch: &ExternalFactBatch) -> String {
    let label = sanitized_source_label(&batch.metadata.source_label);
    format!(
        "{}-{}-{}.json",
        label,
        fingerprint_prefix(&batch.metadata.batch_fingerprint),
        fingerprint_prefix(&batch.metadata.source_artifact_fingerprint)
    )
}

fn sanitized_source_label(label: &str) -> String {
    let sanitized = label
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string();
    if sanitized.is_empty() {
        "external-facts".to_string()
    } else {
        sanitized
    }
}

fn fingerprint_prefix(fingerprint: &str) -> &str {
    fingerprint.get(..16).unwrap_or(fingerprint)
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
    let current_workspace_root = manifest.root.to_string_lossy().to_string();
    if !metadata.workspace_root.trim().is_empty()
        && metadata.workspace_root != current_workspace_root
    {
        issues.push(issue(
            ExternalFactIngestionIssueCode::WorkspaceRootMismatch,
            Some(metadata.workspace_root.clone()),
            format!("workspace_root_mismatch:current={current_workspace_root}"),
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
    manifest_file_endpoints(manifest)
        .into_iter()
        .chain(manifest.symbols.iter().map(|symbol| symbol.id.clone()))
        .chain(manifest.shards.iter().map(|shard| shard.id.clone()))
        .collect()
}

fn manifest_symbol_ids(manifest: &ProjectManifest) -> BTreeSet<String> {
    manifest
        .symbols
        .iter()
        .map(|symbol| symbol.id.clone())
        .collect()
}

fn manifest_file_endpoints(manifest: &ProjectManifest) -> BTreeSet<String> {
    manifest
        .files
        .iter()
        .map(|file| file.path.clone())
        .collect()
}

fn manifest_file_line_counts(manifest: &ProjectManifest) -> BTreeMap<String, u32> {
    manifest
        .files
        .iter()
        .map(|file| (file.path.clone(), file.lines))
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
    edges.sort_by(canonical_edge_order);
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

fn canonical_edge_order(left: &GraphEdge, right: &GraphEdge) -> Ordering {
    left.from
        .cmp(&right.from)
        .then_with(|| left.to.cmp(&right.to))
        .then_with(|| left.kind.cmp(&right.kind))
        .then_with(|| left.confidence_kind.cmp(&right.confidence_kind))
        .then_with(|| left.confidence.total_cmp(&right.confidence))
        .then_with(|| left.provenance.source.cmp(&right.provenance.source))
        .then_with(|| {
            left.provenance
                .fingerprint
                .cmp(&right.provenance.fingerprint)
        })
        .then_with(|| left.provenance.path.cmp(&right.provenance.path))
        .then_with(|| left.provenance.start_line.cmp(&right.provenance.start_line))
        .then_with(|| left.provenance.end_line.cmp(&right.provenance.end_line))
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

#[derive(Clone, Debug)]
struct ArtifactCandidate {
    relative_path: String,
    absolute_path: PathBuf,
    file_type: fs::FileType,
}

#[derive(Clone, Debug)]
struct ArtifactParseReport {
    artifact_fingerprint: Option<String>,
    batch: Option<ExternalFactBatch>,
    issues: Vec<ExternalFactIngestionIssue>,
}

#[derive(Clone, Debug)]
struct AcceptedArtifact {
    artifact_path: String,
    batch: ExternalFactBatch,
}

fn list_artifact_candidates(external_facts_dir: &Path) -> Result<Vec<ArtifactCandidate>> {
    let mut candidates = Vec::new();
    for entry in fs::read_dir(external_facts_dir)? {
        let entry = entry?;
        let absolute_path = entry.path();
        let file_type = entry.file_type()?;
        let relative_path = absolute_path
            .strip_prefix(external_facts_dir)
            .map(normalized_artifact_path)
            .unwrap_or_else(|_| normalized_artifact_path(&absolute_path));
        candidates.push(ArtifactCandidate { relative_path, absolute_path, file_type });
    }
    candidates.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    Ok(candidates)
}

fn parse_artifact_candidate(
    candidate: &ArtifactCandidate,
    frozen_base: &ProjectManifest,
) -> ArtifactParseReport {
    let mut issues = artifact_file_issues(candidate);
    if !issues.is_empty() {
        sort_issues(&mut issues);
        return ArtifactParseReport { artifact_fingerprint: None, batch: None, issues };
    }

    let metadata = match fs::symlink_metadata(&candidate.absolute_path) {
        Ok(metadata) => metadata,
        Err(error) => {
            issues.push(issue(
                ExternalFactIngestionIssueCode::ArtifactReadFailed,
                Some(candidate.relative_path.clone()),
                format!("artifact_metadata_failed:{}", redacted_io_error(&error)),
            ));
            sort_issues(&mut issues);
            return ArtifactParseReport { artifact_fingerprint: None, batch: None, issues };
        }
    };
    if metadata.len() > MAX_EXTERNAL_FACT_ARTIFACT_BYTES {
        issues.push(issue(
            ExternalFactIngestionIssueCode::ArtifactTooLarge,
            Some(candidate.relative_path.clone()),
            format!(
                "artifact_too_large:{}>{}",
                metadata.len(),
                MAX_EXTERNAL_FACT_ARTIFACT_BYTES
            ),
        ));
        sort_issues(&mut issues);
        return ArtifactParseReport { artifact_fingerprint: None, batch: None, issues };
    }

    let bytes = match fs::read(&candidate.absolute_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            issues.push(issue(
                ExternalFactIngestionIssueCode::ArtifactReadFailed,
                Some(candidate.relative_path.clone()),
                format!("artifact_read_failed:{}", redacted_io_error(&error)),
            ));
            sort_issues(&mut issues);
            return ArtifactParseReport { artifact_fingerprint: None, batch: None, issues };
        }
    };
    let json = match String::from_utf8(bytes) {
        Ok(json) => json,
        Err(error) => {
            issues.push(issue(
                ExternalFactIngestionIssueCode::ArtifactParseFailed,
                Some(candidate.relative_path.clone()),
                format!("artifact_utf8_failed:{}", error.utf8_error()),
            ));
            sort_issues(&mut issues);
            return ArtifactParseReport { artifact_fingerprint: None, batch: None, issues };
        }
    };
    let mut batch = match serde_json::from_str::<ExternalFactBatch>(&json) {
        Ok(batch) => batch,
        Err(error) => {
            let artifact_fingerprint = fingerprint(&json);
            issues.push(issue(
                ExternalFactIngestionIssueCode::ArtifactParseFailed,
                Some(candidate.relative_path.clone()),
                format!("artifact_json_failed:{error}"),
            ));
            sort_issues(&mut issues);
            return ArtifactParseReport {
                artifact_fingerprint: Some(artifact_fingerprint),
                batch: None,
                issues,
            };
        }
    };
    let artifact_fingerprint = external_fact_artifact_fingerprint(&batch);
    if batch.metadata.source_artifact_fingerprint != artifact_fingerprint {
        issues.push(issue(
            ExternalFactIngestionIssueCode::SourceArtifactFingerprintMismatch,
            Some(batch.metadata.source_artifact_fingerprint.clone()),
            format!("source_artifact_fingerprint_expected:{artifact_fingerprint}"),
        ));
    }
    batch.metadata.source_artifact_fingerprint = artifact_fingerprint.clone();
    let expected_batch_fingerprint = external_fact_batch_fingerprint(&batch.metadata, &batch.facts);
    if batch.metadata.batch_fingerprint != expected_batch_fingerprint {
        issues.push(issue(
            ExternalFactIngestionIssueCode::BatchFingerprintMismatch,
            Some(batch.metadata.batch_fingerprint.clone()),
            format!("batch_fingerprint_expected:{expected_batch_fingerprint}"),
        ));
    }
    batch.metadata.batch_fingerprint = expected_batch_fingerprint;
    issues.extend(validate_external_fact_batch(frozen_base, &batch));
    sort_issues(&mut issues);
    let accepted_batch = if issues.is_empty() { Some(batch) } else { None };
    ArtifactParseReport {
        artifact_fingerprint: Some(artifact_fingerprint),
        batch: accepted_batch,
        issues,
    }
}

fn artifact_file_issues(candidate: &ArtifactCandidate) -> Vec<ExternalFactIngestionIssue> {
    let mut issues = Vec::new();
    if candidate.file_type.is_symlink() {
        issues.push(issue(
            ExternalFactIngestionIssueCode::SymlinkArtifact,
            Some(candidate.relative_path.clone()),
            "artifact_symlink_ignored".to_string(),
        ));
        return issues;
    }
    if !candidate.file_type.is_file() {
        issues.push(issue(
            ExternalFactIngestionIssueCode::NonFileArtifact,
            Some(candidate.relative_path.clone()),
            "artifact_non_file_ignored".to_string(),
        ));
        return issues;
    }
    if !candidate.relative_path.ends_with(".json") {
        issues.push(issue(
            ExternalFactIngestionIssueCode::NonJsonArtifact,
            Some(candidate.relative_path.clone()),
            "artifact_non_json_ignored".to_string(),
        ));
    }
    issues
}

fn duplicate_batch_issues(
    accepted: &[AcceptedArtifact],
) -> BTreeMap<String, ExternalFactIngestionIssue> {
    let mut by_fingerprint = BTreeMap::<String, Vec<String>>::new();
    for artifact in accepted {
        by_fingerprint
            .entry(artifact.batch.metadata.batch_fingerprint.clone())
            .or_default()
            .push(artifact.artifact_path.clone());
    }
    let mut issues = BTreeMap::new();
    for (fingerprint, mut paths) in by_fingerprint {
        if paths.len() < 2 {
            continue;
        }
        paths.sort();
        for path in paths {
            issues.insert(
                path,
                issue(
                    ExternalFactIngestionIssueCode::DuplicateBatchFingerprint,
                    Some(fingerprint.clone()),
                    "duplicate_batch_fingerprint".to_string(),
                ),
            );
        }
    }
    issues
}

fn normalized_artifact_path(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

fn redacted_io_error(error: &std::io::Error) -> String {
    error.kind().to_string().replace(char::is_whitespace, "_")
}

fn sort_issues(issues: &mut Vec<ExternalFactIngestionIssue>) {
    issues.sort_by(|left, right| {
        left.code
            .cmp(&right.code)
            .then_with(|| left.endpoint.cmp(&right.endpoint))
            .then_with(|| left.detail.cmp(&right.detail))
    });
    issues.dedup();
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
    fn duplicate_external_edges_deduplicate_independently_of_input_order() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let mut left = setup.index()?;
        let mut right = setup.index()?;
        let left_batch = duplicate_line_batch(&left, false);
        let right_batch = duplicate_line_batch(&right, true);
        assert_eq!(
            left_batch.metadata.batch_fingerprint,
            right_batch.metadata.batch_fingerprint
        );

        ingest_external_fact_batch(&mut left, left_batch)?;
        ingest_external_fact_batch(&mut right, right_batch)?;

        let actual = left
            .edges
            .iter()
            .filter(|edge| edge.from == "lsp:src/lib.rs:Root::new")
            .collect::<Vec<_>>();
        let expected = right
            .edges
            .iter()
            .filter(|edge| edge.from == "lsp:src/lib.rs:Root::new")
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        Ok(())
    }

    fn duplicate_line_batch(manifest: &ProjectManifest, reversed: bool) -> ExternalFactBatch {
        let mut references = vec![
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
                start_line: Some(11),
                end_line: Some(11),
                source: ExternalFactSource::Lsp,
            },
        ];
        if reversed {
            references.reverse();
        }
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
            references,
        };
        ExternalFactBatch::new(
            ExternalFactBatchMetadata {
                source: ExternalFactSource::Lsp,
                source_label: "rust-analyzer".to_string(),
                tool_version: Some("fixture-1".to_string()),
                workspace_root: manifest.root.to_string_lossy().to_string(),
                source_artifact_fingerprint: fingerprint("duplicate-lines"),
                manifest_hash_input: manifest.manifest_hash.clone(),
                batch_fingerprint: String::new(),
            },
            facts,
        )
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
        batch
            .facts
            .references
            .first_mut()
            .expect("fixture batch should include references")
            .to = "missing:symbol".to_string();
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
    fn external_reference_path_outside_manifest_is_rejected() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let mut manifest = setup.index()?;
        let before_edges = manifest.edges.clone();
        let mut batch = fixture_batch(&manifest, &fingerprint("snapshot-a"));
        batch
            .facts
            .references
            .first_mut()
            .expect("fixture batch should include references")
            .path = "src/missing.rs".to_string();
        batch.metadata.batch_fingerprint =
            external_fact_batch_fingerprint(&batch.metadata, &batch.facts);

        let actual = ingest_external_fact_batch(&mut manifest, batch).is_err();
        let expected = true;

        assert_eq!(actual, expected);
        assert_eq!(manifest.edges, before_edges);
        Ok(())
    }

    #[test]
    fn external_symbol_with_invalid_line_range_is_rejected_before_manifest_mutation() -> Result<()>
    {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let mut manifest = setup.index()?;
        let before_symbols = manifest.symbols.clone();
        let mut batch = fixture_batch(&manifest, &fingerprint("invalid-line-range"));
        let symbol = batch
            .facts
            .symbols
            .first_mut()
            .expect("fixture batch should include symbols");
        symbol.start_line = 0;
        symbol.end_line = 0;
        batch.metadata.batch_fingerprint =
            external_fact_batch_fingerprint(&batch.metadata, &batch.facts);

        let actual = ingest_external_fact_batch(&mut manifest, batch).is_err();
        let expected = true;

        assert_eq!(actual, expected);
        assert_eq!(manifest.symbols, before_symbols);
        Ok(())
    }

    #[test]
    fn external_symbol_line_range_past_file_end_is_rejected_before_manifest_mutation() -> Result<()>
    {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let mut manifest = setup.index()?;
        let before_symbols = manifest.symbols.clone();
        let mut batch = fixture_batch(&manifest, &fingerprint("out-of-bounds-line-range"));
        let symbol = batch
            .facts
            .symbols
            .first_mut()
            .expect("fixture batch should include symbols");
        symbol.start_line = 10_000;
        symbol.end_line = 10_001;
        batch.metadata.batch_fingerprint =
            external_fact_batch_fingerprint(&batch.metadata, &batch.facts);

        let actual = ingest_external_fact_batch(&mut manifest, batch).is_err();
        let expected = true;

        assert_eq!(actual, expected);
        assert_eq!(manifest.symbols, before_symbols);
        Ok(())
    }

    #[test]
    fn external_reference_line_range_past_file_end_is_rejected_before_manifest_mutation()
    -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let mut manifest = setup.index()?;
        let before_edges = manifest.edges.clone();
        let mut batch = fixture_batch(&manifest, &fingerprint("out-of-bounds-reference-range"));
        let reference = batch
            .facts
            .references
            .first_mut()
            .expect("fixture batch should include references");
        reference.start_line = Some(10_000);
        reference.end_line = Some(10_001);
        batch.metadata.batch_fingerprint =
            external_fact_batch_fingerprint(&batch.metadata, &batch.facts);

        let issues = validate_external_fact_batch(&manifest, &batch);
        let actual = ingest_external_fact_batch(&mut manifest, batch).is_err();
        let expected = true;

        assert_eq!(actual, expected);
        assert_eq!(
            issues.iter().any(
                |issue| issue.code == ExternalFactIngestionIssueCode::InvalidReferenceLineRange
            ),
            true
        );
        assert_eq!(manifest.edges, before_edges);
        Ok(())
    }

    #[test]
    fn external_batch_workspace_root_mismatch_is_rejected() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let mut manifest = setup.index()?;
        let before_edges = manifest.edges.clone();
        let mut batch = fixture_batch(&manifest, &fingerprint("snapshot-a"));
        batch.metadata.workspace_root = "/different/workspace".to_string();
        batch.metadata.batch_fingerprint =
            external_fact_batch_fingerprint(&batch.metadata, &batch.facts);

        let issues = validate_external_fact_batch(&manifest, &batch);
        let actual = ingest_external_fact_batch(&mut manifest, batch).is_err();
        let expected = true;

        assert_eq!(actual, expected);
        assert_eq!(
            issues
                .iter()
                .any(|issue| issue.code == ExternalFactIngestionIssueCode::WorkspaceRootMismatch),
            true
        );
        assert_eq!(manifest.edges, before_edges);
        Ok(())
    }

    #[test]
    fn duplicate_external_symbol_ids_are_rejected_before_manifest_mutation() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let mut manifest = setup.index()?;
        let before_symbols = manifest.symbols.clone();
        let facts = TypedExternalFacts {
            symbols: vec![
                TypedExternalSymbolFact {
                    id: "lsp:src/lib.rs:Root::new".to_string(),
                    name: "new".to_string(),
                    kind: SymbolKind::Method,
                    path: "src/lib.rs".to_string(),
                    start_line: 10,
                    end_line: 12,
                    source: ExternalFactSource::Lsp,
                },
                TypedExternalSymbolFact {
                    id: "lsp:src/lib.rs:Root::new".to_string(),
                    name: "conflicting_new".to_string(),
                    kind: SymbolKind::Method,
                    path: "src/lib.rs".to_string(),
                    start_line: 11,
                    end_line: 13,
                    source: ExternalFactSource::Lsp,
                },
            ],
            references: Vec::new(),
        };
        let batch = ExternalFactBatch::new(
            ExternalFactBatchMetadata {
                source: ExternalFactSource::Lsp,
                source_label: "rust-analyzer".to_string(),
                tool_version: Some("fixture-1".to_string()),
                workspace_root: manifest.root.to_string_lossy().to_string(),
                source_artifact_fingerprint: fingerprint("duplicate-symbol-id"),
                manifest_hash_input: manifest.manifest_hash.clone(),
                batch_fingerprint: String::new(),
            },
            facts,
        );

        let actual = ingest_external_fact_batch(&mut manifest, batch).is_err();
        let expected = true;

        assert_eq!(actual, expected);
        assert_eq!(manifest.symbols, before_symbols);
        Ok(())
    }

    #[test]
    fn external_symbol_id_conflicting_with_manifest_symbol_is_rejected_before_manifest_mutation()
    -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let mut manifest = setup.index()?;
        let existing_symbol = manifest
            .symbols
            .iter()
            .find(|symbol| symbol.id == "symbol:src/lib.rs:Struct:Root")
            .expect("fixture should include Root symbol")
            .clone();
        let before_symbols = manifest.symbols.clone();
        let facts = TypedExternalFacts {
            symbols: vec![TypedExternalSymbolFact {
                id: existing_symbol.id.clone(),
                name: "conflicting_external_root".to_string(),
                kind: SymbolKind::Method,
                path: existing_symbol.path.clone(),
                start_line: existing_symbol.start_line,
                end_line: existing_symbol.end_line,
                source: ExternalFactSource::Lsp,
            }],
            references: Vec::new(),
        };
        let batch = ExternalFactBatch::new(
            ExternalFactBatchMetadata {
                source: ExternalFactSource::Lsp,
                source_label: "rust-analyzer".to_string(),
                tool_version: Some("fixture-1".to_string()),
                workspace_root: manifest.root.to_string_lossy().to_string(),
                source_artifact_fingerprint: fingerprint("manifest-symbol-id-conflict"),
                manifest_hash_input: manifest.manifest_hash.clone(),
                batch_fingerprint: String::new(),
            },
            facts,
        );

        let issues = validate_external_fact_batch(&manifest, &batch);
        let actual = ingest_external_fact_batch(&mut manifest, batch).is_err();
        let expected = true;

        assert_eq!(actual, expected);
        assert_eq!(
            issues
                .iter()
                .any(|issue| issue.code
                    == ExternalFactIngestionIssueCode::ConflictingManifestSymbolId),
            true
        );
        assert_eq!(manifest.symbols, before_symbols);
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
