//! Read-only exact-fact status auditing for persisted project-model state.

use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::policy::{
    local_project_model_dir, local_project_model_external_fact_report, local_project_model_manifest,
};
use crate::types::{
    EdgeConfidence, ExternalFactArtifactIngestionReport, ExternalFactBatchMetadata,
    ExternalFactIngestionIssue, FreshnessProofLevel, GraphEdgeKind, ProjectManifest,
};
use crate::util::{external_facts_fingerprint, hash_text, manifest_hash};

/// Stable state of the read-only exact-fact artifact store inspection.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExactFactArtifactStoreState {
    /// The typed artifact metadata API proved that the store directory is absent.
    Absent,
    /// The store directory exists and contains no regular candidate files.
    Empty,
    /// The store directory exists and contains at least one regular candidate file.
    Present,
    /// The store directory exists but metadata could not be read safely.
    Unreadable,
}

impl ExactFactArtifactStoreState {
    /// Returns the stable lowercase label used by transport DTOs.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Absent => "absent",
            Self::Empty => "empty",
            Self::Present => "present",
            Self::Unreadable => "unreadable",
        }
    }
}

/// Stable read-only exact-fact status.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExactFactStatus {
    /// The persisted project manifest file is absent.
    NotIndexed,
    /// Required persisted status data is missing, corrupt, or unreadable.
    ReportMissingOrCorrupt,
    /// The manifest is readable but not sufficiently fresh for active truth.
    StaleManifest,
    /// The artifact store directory is proven absent.
    NoArtifactStore,
    /// Artifact candidates or report entries exist, but none are accepted.
    ArtifactsPresentNoneAccepted,
    /// Accepted batch evidence exists without graph-visible exact reference edges.
    AcceptedButNoGraphEdges,
    /// Fresh persisted evidence proves active exact compiler reference facts.
    Active,
}

impl ExactFactStatus {
    /// Returns the stable lowercase label used by transport DTOs.
    pub fn label(&self) -> &'static str {
        match self {
            Self::NotIndexed => "not_indexed",
            Self::ReportMissingOrCorrupt => "report_missing_or_corrupt",
            Self::StaleManifest => "stale_manifest",
            Self::NoArtifactStore => "no_artifact_store",
            Self::ArtifactsPresentNoneAccepted => "artifacts_present_none_accepted",
            Self::AcceptedButNoGraphEdges => "accepted_but_no_graph_edges",
            Self::Active => "active",
        }
    }
}

/// Read-only exact-fact artifact-store metadata.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExactFactArtifactStoreMetadata {
    /// Artifact store directory path.
    pub path: PathBuf,
    /// Stable store state.
    pub state: ExactFactArtifactStoreState,
    /// Count of regular candidate files observed without parsing payloads.
    pub candidate_file_count: usize,
}

/// Redaction-safe read-only exact-fact status report.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExactFactStatusReport {
    /// Stable status.
    pub status: ExactFactStatus,
    /// Canonical manifest path.
    pub manifest_path: PathBuf,
    /// Manifest hash from the persisted manifest, when readable.
    pub manifest_hash: Option<String>,
    /// Manifest freshness proof label.
    pub manifest_freshness_proof_level: Option<FreshnessProofLevel>,
    /// Canonical persisted ingestion-report path.
    pub ingestion_report_path: PathBuf,
    /// Artifact store metadata state label.
    pub artifact_store_state: ExactFactArtifactStoreState,
    /// Count of artifact candidates inspected by the persisted ingestion report.
    pub inspected_artifact_count: usize,
    /// Count of accepted artifacts in the persisted ingestion report.
    pub accepted_artifact_count: usize,
    /// Accepted batch fingerprints in deterministic report order.
    pub accepted_batch_fingerprints: Vec<String>,
    /// Accepted external fact batch count persisted in the manifest.
    pub manifest_external_fact_batch_count: usize,
    /// Manifest external facts fingerprint.
    pub manifest_external_facts_fingerprint: Option<String>,
    /// Graph-visible reference edge count.
    pub reference_edge_count: usize,
    /// Graph-visible exact compiler reference edge count.
    pub exact_compiler_reference_edge_count: usize,
    /// Redaction-safe status issue summaries.
    pub issue_summaries: Vec<String>,
    /// Whether exact facts are active.
    pub exact_facts_active: bool,
}

/// Reads artifact-store metadata without parsing or mutating artifact payloads.
///
/// # Arguments
///
/// * `model_dir` - Project-model directory that owns the external fact store.
///
/// # Errors
///
/// Returns an error when the store exists but cannot be listed safely.
pub fn read_exact_fact_artifact_store_metadata(
    model_dir: &Path,
) -> anyhow::Result<ExactFactArtifactStoreMetadata> {
    let path = model_dir.join("external_facts");
    if !path.exists() {
        return Ok(ExactFactArtifactStoreMetadata {
            path,
            state: ExactFactArtifactStoreState::Absent,
            candidate_file_count: 0,
        });
    }
    if !path.is_dir() {
        return Ok(ExactFactArtifactStoreMetadata {
            path,
            state: ExactFactArtifactStoreState::Unreadable,
            candidate_file_count: 0,
        });
    }

    let mut candidate_file_count = 0usize;
    for entry in fs::read_dir(&path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_file() {
            candidate_file_count = candidate_file_count.saturating_add(1);
        }
    }
    let state = if candidate_file_count == 0 {
        ExactFactArtifactStoreState::Empty
    } else {
        ExactFactArtifactStoreState::Present
    };
    Ok(ExactFactArtifactStoreMetadata { path, state, candidate_file_count })
}

/// Audits persisted exact-fact health without invoking producers or writers.
///
/// # Arguments
///
/// * `workspace_root` - Workspace root that owns local project-model storage.
///
/// # Errors
///
/// Returns an error only when the artifact-store metadata API itself fails before
/// a typed diagnostic report can be constructed.
pub fn read_exact_fact_status(workspace_root: &Path) -> anyhow::Result<ExactFactStatusReport> {
    let model_dir = local_project_model_dir(workspace_root);
    let manifest_path = local_project_model_manifest(workspace_root);
    let ingestion_report_path = local_project_model_external_fact_report(workspace_root);

    if !manifest_path.exists() {
        return Ok(base_report(
            ExactFactStatus::NotIndexed,
            manifest_path,
            ingestion_report_path,
            ExactFactArtifactStoreMetadata {
                path: model_dir.join("external_facts"),
                state: ExactFactArtifactStoreState::Absent,
                candidate_file_count: 0,
            },
            vec!["manifest_absent".to_string()],
        ));
    }

    let artifact_store = read_exact_fact_artifact_store_metadata(&model_dir).unwrap_or_else(|_| {
        ExactFactArtifactStoreMetadata {
            path: model_dir.join("external_facts"),
            state: ExactFactArtifactStoreState::Unreadable,
            candidate_file_count: 0,
        }
    });

    let (manifest, mut issues) = match read_manifest_status_data(&manifest_path) {
        Ok(manifest) => (manifest, Vec::new()),
        Err(issue) => {
            return Ok(base_report(
                ExactFactStatus::ReportMissingOrCorrupt,
                manifest_path,
                ingestion_report_path,
                artifact_store,
                vec![issue],
            ));
        }
    };

    let report = match read_ingestion_report_status_data(&ingestion_report_path) {
        Ok(report) => report,
        Err(issue) => {
            return Ok(report_from_parts(
                ExactFactStatus::ReportMissingOrCorrupt,
                manifest_path,
                ingestion_report_path,
                artifact_store,
                &manifest,
                None,
                vec![issue],
            ));
        }
    };

    if artifact_store.state == ExactFactArtifactStoreState::Unreadable {
        issues.push("artifact_store_metadata_unreadable".to_string());
        return Ok(report_from_parts(
            ExactFactStatus::ReportMissingOrCorrupt,
            manifest_path,
            ingestion_report_path,
            artifact_store,
            &manifest,
            Some(&report),
            issues,
        ));
    }

    let freshness = persisted_manifest_freshness(&manifest);
    if freshness.is_err() {
        issues.push("manifest_self_fingerprint_mismatch".to_string());
        return Ok(report_from_parts(
            ExactFactStatus::StaleManifest,
            manifest_path,
            ingestion_report_path,
            artifact_store,
            &manifest,
            Some(&report),
            issues,
        ));
    }

    if !workspace_manifest_sources_fresh(workspace_root, &manifest)? {
        issues.push("manifest_source_file_changed_or_missing".to_string());
        return Ok(report_from_parts(
            ExactFactStatus::StaleManifest,
            manifest_path,
            ingestion_report_path,
            artifact_store,
            &manifest,
            Some(&report),
            issues,
        ));
    }

    if !accepted_batch_metadata_consistent(&manifest, &report) {
        issues.push("accepted_batch_metadata_mismatch".to_string());
        return Ok(report_from_parts(
            ExactFactStatus::ReportMissingOrCorrupt,
            manifest_path,
            ingestion_report_path,
            artifact_store,
            &manifest,
            Some(&report),
            issues,
        ));
    }

    let accepted_count = report.accepted_artifacts;
    let exact_edges = exact_compiler_reference_edge_count(&manifest);
    let status = if artifact_store.state == ExactFactArtifactStoreState::Absent {
        ExactFactStatus::NoArtifactStore
    } else if accepted_count == 0 {
        ExactFactStatus::ArtifactsPresentNoneAccepted
    } else if exact_edges == 0 {
        ExactFactStatus::AcceptedButNoGraphEdges
    } else {
        ExactFactStatus::Active
    };

    Ok(report_from_parts(
        status,
        manifest_path,
        ingestion_report_path,
        artifact_store,
        &manifest,
        Some(&report),
        issues,
    ))
}

fn read_manifest_status_data(path: &Path) -> Result<ProjectManifest, String> {
    let json = fs::read_to_string(path).map_err(|_| "manifest_unreadable".to_string())?;
    serde_json::from_str(&json).map_err(|_| "manifest_corrupt".to_string())
}

fn read_ingestion_report_status_data(
    path: &Path,
) -> Result<ExternalFactArtifactIngestionReport, String> {
    let json = fs::read_to_string(path)
        .map_err(|_| "ingestion_report_missing_or_unreadable".to_string())?;
    serde_json::from_str(&json).map_err(|_| "ingestion_report_corrupt".to_string())
}

fn persisted_manifest_freshness(manifest: &ProjectManifest) -> Result<(), ()> {
    let expected_external_facts_fingerprint =
        external_facts_fingerprint(&manifest.external_fact_batches);
    if manifest.external_facts_fingerprint != expected_external_facts_fingerprint {
        return Err(());
    }
    let expected_manifest_hash = manifest_hash(
        &manifest.files,
        &manifest.external_fact_batches,
        &manifest.external_facts_fingerprint,
    );
    if manifest.manifest_hash != expected_manifest_hash {
        return Err(());
    }
    Ok(())
}

fn workspace_manifest_sources_fresh(
    workspace_root: &Path,
    manifest: &ProjectManifest,
) -> anyhow::Result<bool> {
    let root = workspace_root
        .canonicalize()
        .unwrap_or_else(|_| workspace_root.to_path_buf());
    for file in &manifest.files {
        if !manifest_relative_path_is_safe(&file.path) {
            return Ok(false);
        }
        let path = workspace_root.join(&file.path);
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(error.into()),
        };
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Ok(false);
        }
        let canonical_path = match path.canonicalize() {
            Ok(canonical_path) => canonical_path,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(error.into()),
        };
        if !canonical_path.starts_with(&root) {
            return Ok(false);
        }
        let bytes = fs::read(&canonical_path)?;
        let content = match String::from_utf8(bytes) {
            Ok(content) => content,
            Err(_) => return Ok(false),
        };
        if hash_text(&content) != file.content_hash {
            return Ok(false);
        }
    }
    Ok(true)
}

fn manifest_relative_path_is_safe(path: &str) -> bool {
    !path.is_empty()
        && !Path::new(path).is_absolute()
        && Path::new(path)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn accepted_batch_metadata_consistent(
    manifest: &ProjectManifest,
    report: &ExternalFactArtifactIngestionReport,
) -> bool {
    let report_batches = report
        .accepted_batches
        .iter()
        .map(|batch| batch.batch_metadata.clone())
        .collect::<Vec<_>>();
    let artifact_batches = report
        .artifacts
        .iter()
        .filter_map(|artifact| artifact.accepted_batch.clone())
        .collect::<Vec<_>>();
    if report.accepted_artifacts > 0 && artifact_batches != report_batches {
        return false;
    }
    manifest.external_fact_batches == report_batches
        && report.accepted_artifacts == report.accepted_batches.len()
        && manifest.external_facts_fingerprint == external_facts_fingerprint(&report_batches)
}

fn reference_edge_count(manifest: &ProjectManifest) -> usize {
    manifest
        .edges
        .iter()
        .filter(|edge| edge.kind == GraphEdgeKind::References)
        .count()
}

fn exact_compiler_reference_edge_count(manifest: &ProjectManifest) -> usize {
    manifest
        .edges
        .iter()
        .filter(|edge| {
            edge.kind == GraphEdgeKind::References
                && edge.confidence_kind == EdgeConfidence::ExactCompiler
        })
        .count()
}

fn report_from_parts(
    status: ExactFactStatus,
    manifest_path: PathBuf,
    ingestion_report_path: PathBuf,
    artifact_store: ExactFactArtifactStoreMetadata,
    manifest: &ProjectManifest,
    report: Option<&ExternalFactArtifactIngestionReport>,
    mut issue_summaries: Vec<String>,
) -> ExactFactStatusReport {
    if let Some(report) = report {
        issue_summaries.extend(redaction_safe_ingestion_issue_summaries(report));
    }
    issue_summaries.sort();
    issue_summaries.dedup();
    let exact_facts_active = status == ExactFactStatus::Active;
    ExactFactStatusReport {
        status,
        manifest_path,
        manifest_hash: Some(manifest.manifest_hash.clone()),
        manifest_freshness_proof_level: Some(FreshnessProofLevel::IndexedFilesOnly),
        ingestion_report_path,
        artifact_store_state: artifact_store.state,
        inspected_artifact_count: report.map_or(0, |report| report.inspected_artifacts),
        accepted_artifact_count: report.map_or(0, |report| report.accepted_artifacts),
        accepted_batch_fingerprints: report.map_or_else(Vec::new, accepted_batch_fingerprints),
        manifest_external_fact_batch_count: manifest.external_fact_batches.len(),
        manifest_external_facts_fingerprint: Some(manifest.external_facts_fingerprint.clone()),
        reference_edge_count: reference_edge_count(manifest),
        exact_compiler_reference_edge_count: exact_compiler_reference_edge_count(manifest),
        issue_summaries,
        exact_facts_active,
    }
}

fn base_report(
    status: ExactFactStatus,
    manifest_path: PathBuf,
    ingestion_report_path: PathBuf,
    artifact_store: ExactFactArtifactStoreMetadata,
    issue_summaries: Vec<String>,
) -> ExactFactStatusReport {
    ExactFactStatusReport {
        status,
        manifest_path,
        manifest_hash: None,
        manifest_freshness_proof_level: None,
        ingestion_report_path,
        artifact_store_state: artifact_store.state,
        inspected_artifact_count: 0,
        accepted_artifact_count: 0,
        accepted_batch_fingerprints: Vec::new(),
        manifest_external_fact_batch_count: 0,
        manifest_external_facts_fingerprint: None,
        reference_edge_count: 0,
        exact_compiler_reference_edge_count: 0,
        issue_summaries,
        exact_facts_active: false,
    }
}

fn accepted_batch_fingerprints(report: &ExternalFactArtifactIngestionReport) -> Vec<String> {
    report
        .accepted_batches
        .iter()
        .map(|batch| batch.batch_metadata.batch_fingerprint.clone())
        .collect()
}

fn redaction_safe_ingestion_issue_summaries(
    report: &ExternalFactArtifactIngestionReport,
) -> Vec<String> {
    report
        .artifacts
        .iter()
        .flat_map(|artifact| artifact.issues.iter().map(redaction_safe_issue_summary))
        .collect()
}

fn redaction_safe_issue_summary(issue: &ExternalFactIngestionIssue) -> String {
    let endpoint = issue.endpoint.as_deref().unwrap_or("none");
    format!("{:?}:{}:{}", issue.code, endpoint, hash_text(&issue.detail))
}

#[allow(dead_code)]
fn _assert_no_producer_path_symbols(
    _metadata: &ExactFactArtifactStoreMetadata,
    _status: &ExactFactStatusReport,
    _batch: &ExternalFactBatchMetadata,
) {
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use anyhow::Result;
    use pretty_assertions::assert_eq;
    use tempfile::TempDir;

    use super::*;
    use crate::ingestion::{ingest_external_fact_batch, prepared_external_fact_artifact_batch};
    use crate::types::{
        ExternalFactArtifactReport, ExternalFactBatch, ExternalFactIngestionReport,
        ExternalFactSource, GraphEdgeKind, SourceFile, SymbolKind, TypedExternalFacts,
        TypedExternalReferenceFact, TypedExternalSymbolFact,
    };
    use crate::util::{fingerprint, provenance};

    struct Fixture {
        _temp: TempDir,
        root: PathBuf,
        model_dir: PathBuf,
    }

    impl Fixture {
        fn new() -> Result<Self> {
            let temp = TempDir::new()?;
            let root = temp.path().join("workspace");
            let model_dir = local_project_model_dir(&root);
            fs::create_dir_all(&model_dir)?;
            Ok(Self { _temp: temp, root, model_dir })
        }

        fn write_manifest(&self, manifest: &ProjectManifest) -> Result<()> {
            fs::write(
                local_project_model_manifest(&self.root),
                serde_json::to_string_pretty(manifest)?,
            )?;
            Ok(())
        }

        fn write_report(&self, report: &ExternalFactArtifactIngestionReport) -> Result<()> {
            fs::write(
                local_project_model_external_fact_report(&self.root),
                serde_json::to_string_pretty(report)?,
            )?;
            Ok(())
        }

        fn create_store_file(&self, name: &str) -> Result<()> {
            let store = self.model_dir.join("external_facts");
            fs::create_dir_all(&store)?;
            fs::write(store.join(name), b"{}")?;
            Ok(())
        }

        fn write_source_file(&self) -> Result<()> {
            fs::create_dir_all(self.root.join("src"))?;
            fs::write(self.root.join("src/lib.rs"), b"source")?;
            Ok(())
        }
    }

    fn base_manifest(root: &Path) -> ProjectManifest {
        let files = vec![SourceFile {
            path: "src/lib.rs".to_string(),
            language: crate::types::Language::Rust,
            bytes: 10,
            lines: 2,
            content_hash: fingerprint("source"),
            provenance: provenance("src/lib.rs", None, None, "test", "source"),
        }];
        let external_fact_batches = Vec::new();
        let external_facts_fingerprint = external_facts_fingerprint(&external_fact_batches);
        let manifest_hash =
            manifest_hash(&files, &external_fact_batches, &external_facts_fingerprint);
        ProjectManifest {
            version: 1,
            root: root.to_path_buf(),
            files,
            file_nodes: Vec::new(),
            symbols: Vec::new(),
            cargo_workspace: None,
            cargo_packages: Vec::new(),
            cargo_package_dependencies: Vec::new(),
            edges: Vec::new(),
            external_fact_batches,
            external_facts_fingerprint,
            shards: Vec::new(),
            manifest_hash,
        }
    }

    fn active_manifest(
        root: &Path,
    ) -> Result<(ProjectManifest, ExternalFactArtifactIngestionReport)> {
        let mut manifest = base_manifest(root);
        manifest.symbols.push(crate::types::SymbolNode {
            id: "symbol:root".to_string(),
            name: "root".to_string(),
            kind: SymbolKind::Function,
            path: "src/lib.rs".to_string(),
            parent: None,
            start_line: 1,
            end_line: 1,
            provenance: provenance("src/lib.rs", Some(1), Some(1), "test", "symbol"),
        });
        let batch = fixture_batch(&manifest);
        let batch_metadata = batch.metadata.clone();
        ingest_external_fact_batch(&mut manifest, batch)?;
        let report = ExternalFactArtifactIngestionReport {
            store_path: "external_facts".to_string(),
            inspected_artifacts: 1,
            accepted_artifacts: 1,
            artifacts: vec![ExternalFactArtifactReport {
                artifact_path: "fixture.json".to_string(),
                artifact_fingerprint: Some(batch_metadata.source_artifact_fingerprint.clone()),
                accepted_batch: Some(batch_metadata.clone()),
                issues: Vec::new(),
            }],
            accepted_batches: vec![ExternalFactIngestionReport {
                accepted_symbols: 1,
                accepted_edges: 1,
                deduplicated_edges: 0,
                batch_metadata,
            }],
        };
        Ok((manifest, report))
    }

    fn fixture_batch(manifest: &ProjectManifest) -> ExternalFactBatch {
        let batch = ExternalFactBatch {
            metadata: ExternalFactBatchMetadata {
                source: ExternalFactSource::Lsp,
                source_label: "fixture-lsp".to_string(),
                tool_version: Some("test".to_string()),
                producer_snapshot_fingerprint: fingerprint("producer"),
                workspace_root: manifest.root.to_string_lossy().to_string(),
                source_artifact_fingerprint: String::new(),
                manifest_hash_input: manifest.manifest_hash.clone(),
                batch_fingerprint: String::new(),
            },
            facts: TypedExternalFacts {
                symbols: vec![TypedExternalSymbolFact {
                    id: "external:symbol".to_string(),
                    name: "external".to_string(),
                    kind: SymbolKind::Function,
                    path: "src/lib.rs".to_string(),
                    start_line: 2,
                    end_line: 2,
                    source: ExternalFactSource::Lsp,
                }],
                references: vec![TypedExternalReferenceFact {
                    from: "symbol:root".to_string(),
                    to: "external:symbol".to_string(),
                    kind: GraphEdgeKind::References,
                    path: "src/lib.rs".to_string(),
                    start_line: Some(1),
                    end_line: Some(1),
                    source: ExternalFactSource::Lsp,
                }],
            },
        };
        prepared_external_fact_artifact_batch(manifest, batch)
            .expect("fixture batch should prepare")
    }

    #[test]
    fn manifest_absent_returns_not_indexed_without_creating_model_dir() -> Result<()> {
        let fixture = TempDir::new()?;
        let setup = fixture.path().join("workspace");

        let actual = read_exact_fact_status(&setup)?;

        assert_eq!(actual.status, ExactFactStatus::NotIndexed);
        assert_eq!(setup.join(".forge_project_model").exists(), false);
        Ok(())
    }

    #[test]
    fn manifest_present_without_external_facts_returns_typed_inactive_noop() -> Result<()> {
        let setup = Fixture::new()?;
        setup.write_source_file()?;
        let manifest = base_manifest(&setup.root);
        setup.write_manifest(&manifest)?;
        setup.write_report(&ExternalFactArtifactIngestionReport::default())?;
        fs::create_dir_all(setup.model_dir.join("external_facts"))?;

        let actual = read_exact_fact_status(&setup.root)?;

        assert_eq!(actual.status, ExactFactStatus::ArtifactsPresentNoneAccepted);
        assert_eq!(actual.exact_facts_active, false);
        Ok(())
    }

    #[test]
    fn accepted_fixture_external_facts_produce_active_status() -> Result<()> {
        let setup = Fixture::new()?;
        setup.write_source_file()?;
        let (manifest, report) = active_manifest(&setup.root)?;
        setup.write_manifest(&manifest)?;
        setup.write_report(&report)?;
        setup.create_store_file("fixture.json")?;

        let actual = read_exact_fact_status(&setup.root)?;

        assert_eq!(actual.status, ExactFactStatus::Active);
        assert_eq!(actual.exact_facts_active, true);
        assert_eq!(actual.exact_compiler_reference_edge_count, 1);
        Ok(())
    }

    #[test]
    fn accepted_batch_with_zero_graph_visible_reference_edges_is_inactive() -> Result<()> {
        let setup = Fixture::new()?;
        setup.write_source_file()?;
        let (mut manifest, report) = active_manifest(&setup.root)?;
        manifest.edges.clear();
        manifest.manifest_hash = manifest_hash(
            &manifest.files,
            &manifest.external_fact_batches,
            &manifest.external_facts_fingerprint,
        );
        setup.write_manifest(&manifest)?;
        setup.write_report(&report)?;
        setup.create_store_file("fixture.json")?;

        let actual = read_exact_fact_status(&setup.root)?;

        assert_eq!(actual.status, ExactFactStatus::AcceptedButNoGraphEdges);
        assert_eq!(actual.exact_facts_active, false);
        Ok(())
    }

    #[test]
    fn stale_manifest_blocks_active_status() -> Result<()> {
        let setup = Fixture::new()?;
        let (mut manifest, report) = active_manifest(&setup.root)?;
        manifest.external_facts_fingerprint = fingerprint("stale");
        setup.write_manifest(&manifest)?;
        setup.write_report(&report)?;
        setup.create_store_file("fixture.json")?;

        let actual = read_exact_fact_status(&setup.root)?;

        assert_eq!(actual.status, ExactFactStatus::StaleManifest);
        assert_eq!(actual.exact_facts_active, false);
        Ok(())
    }

    #[test]
    fn changed_workspace_file_blocks_active_status() -> Result<()> {
        let setup = Fixture::new()?;
        fs::create_dir_all(setup.root.join("src"))?;
        fs::write(setup.root.join("src/lib.rs"), b"mutated source")?;
        let (manifest, report) = active_manifest(&setup.root)?;
        setup.write_manifest(&manifest)?;
        setup.write_report(&report)?;
        setup.create_store_file("fixture.json")?;

        let actual = read_exact_fact_status(&setup.root)?;

        assert_eq!(actual.status, ExactFactStatus::StaleManifest);
        assert_eq!(actual.exact_facts_active, false);
        Ok(())
    }

    #[test]
    fn missing_workspace_file_blocks_active_status() -> Result<()> {
        let setup = Fixture::new()?;
        let (manifest, report) = active_manifest(&setup.root)?;
        setup.write_manifest(&manifest)?;
        setup.write_report(&report)?;
        setup.create_store_file("fixture.json")?;

        let actual = read_exact_fact_status(&setup.root)?;

        assert_eq!(actual.status, ExactFactStatus::StaleManifest);
        assert_eq!(actual.exact_facts_active, false);
        assert_eq!(
            actual.issue_summaries,
            vec!["manifest_source_file_changed_or_missing"]
        );
        Ok(())
    }

    #[test]
    fn missing_or_corrupt_ingestion_report_is_redaction_safe_recoverable_status() -> Result<()> {
        let setup = Fixture::new()?;
        let manifest = base_manifest(&setup.root);
        setup.write_manifest(&manifest)?;

        let actual = read_exact_fact_status(&setup.root)?;

        assert_eq!(actual.status, ExactFactStatus::ReportMissingOrCorrupt);
        assert_eq!(
            actual.issue_summaries,
            vec!["ingestion_report_missing_or_unreadable"]
        );
        Ok(())
    }

    #[test]
    fn artifact_store_absent_empty_and_present_are_distinguished_by_metadata_api() -> Result<()> {
        let setup = Fixture::new()?;
        let absent = read_exact_fact_artifact_store_metadata(&setup.model_dir)?;
        fs::create_dir_all(setup.model_dir.join("external_facts"))?;
        let empty = read_exact_fact_artifact_store_metadata(&setup.model_dir)?;
        setup.create_store_file("fixture.json")?;
        let present = read_exact_fact_artifact_store_metadata(&setup.model_dir)?;

        assert_eq!(absent.state, ExactFactArtifactStoreState::Absent);
        assert_eq!(empty.state, ExactFactArtifactStoreState::Empty);
        assert_eq!(present.state, ExactFactArtifactStoreState::Present);
        Ok(())
    }

    #[test]
    fn status_reads_do_not_mutate_manifest_report_or_artifacts() -> Result<()> {
        let setup = Fixture::new()?;
        setup.write_source_file()?;
        let (manifest, report) = active_manifest(&setup.root)?;
        setup.write_manifest(&manifest)?;
        setup.write_report(&report)?;
        setup.create_store_file("fixture.json")?;
        let paths = vec![
            local_project_model_manifest(&setup.root),
            local_project_model_external_fact_report(&setup.root),
            setup.model_dir.join("external_facts/fixture.json"),
        ];
        let before = snapshot(&paths)?;

        let actual = read_exact_fact_status(&setup.root)?;
        let after = snapshot(&paths)?;

        assert_eq!(actual.status, ExactFactStatus::Active);
        assert_eq!(after, before);
        Ok(())
    }

    #[test]
    fn deterministic_precedence_prefers_report_corrupt_before_stale_manifest() -> Result<()> {
        let setup = Fixture::new()?;
        let mut manifest = base_manifest(&setup.root);
        manifest.manifest_hash = fingerprint("stale");
        setup.write_manifest(&manifest)?;
        fs::write(
            local_project_model_external_fact_report(&setup.root),
            b"not-json",
        )?;

        let actual = read_exact_fact_status(&setup.root)?;

        assert_eq!(actual.status, ExactFactStatus::ReportMissingOrCorrupt);
        assert_eq!(actual.issue_summaries, vec!["ingestion_report_corrupt"]);
        Ok(())
    }

    #[test]
    fn accepted_report_without_per_artifact_batch_metadata_is_corrupt() -> Result<()> {
        let setup = Fixture::new()?;
        setup.write_source_file()?;
        let (manifest, mut report) = active_manifest(&setup.root)?;
        report
            .artifacts
            .first_mut()
            .expect("fixture report should contain one artifact")
            .accepted_batch = None;
        setup.write_manifest(&manifest)?;
        setup.write_report(&report)?;
        setup.create_store_file("fixture.json")?;

        let actual = read_exact_fact_status(&setup.root)?;

        assert_eq!(actual.status, ExactFactStatus::ReportMissingOrCorrupt);
        assert_eq!(actual.exact_facts_active, false);
        assert_eq!(
            actual.issue_summaries,
            vec!["accepted_batch_metadata_mismatch"]
        );
        Ok(())
    }

    #[test]
    fn status_path_has_no_producer_driver_or_probe_dependencies() {
        let setup = std::include_str!("status.rs");
        let forbidden = [
            concat!("produce_workspace_exact_fact_reference", "_with_driver("),
            concat!("NativeLspReference", "Producer"),
            concat!("derive_native_lsp", "_reference_request"),
            concat!("RustAnalyzerCapability", "Probe"),
        ];
        let actual = forbidden
            .iter()
            .filter(|pattern| setup.contains(**pattern))
            .copied()
            .collect::<Vec<_>>();
        let expected: Vec<&str> = Vec::new();

        assert_eq!(actual, expected);
    }

    fn snapshot(paths: &[PathBuf]) -> Result<BTreeMap<PathBuf, (u64, String)>> {
        let mut snapshot = BTreeMap::new();
        for path in paths {
            let bytes = fs::read(path)?;
            let content_hash = hash_text(std::str::from_utf8(&bytes)?);
            snapshot.insert(path.clone(), (u64::try_from(bytes.len())?, content_hash));
        }
        Ok(snapshot)
    }
}
