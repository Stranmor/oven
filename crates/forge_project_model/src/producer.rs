//! Typed exact-fact producer boundary for fixture-backed LSP facts.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

use crate::ingestion::{
    prepared_external_fact_artifact_batch, validate_external_fact_batch,
    write_external_fact_artifact,
};
use crate::types::{
    ExternalFactBatch, ExternalFactBatchMetadata, ExternalFactIngestionIssue, ExternalFactSource,
    GraphEdgeKind, ProjectManifest, TypedExternalFacts, TypedExternalReferenceFact,
};
use crate::util::fingerprint;

/// Capability-probed exact-fact producer boundary.
pub trait ExternalFactProducer {
    /// Probes producer capability without emitting facts or writing artifacts.
    ///
    /// # Arguments
    ///
    /// * `request` - Bounded request describing the desired exact-fact slice.
    fn probe(&self, request: &ExternalFactProductionRequest) -> ExternalFactProducerProbe;

    /// Produces a typed external fact artifact through the project-model writer.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Project-model directory that owns external fact storage.
    /// * `frozen_manifest` - Immutable manifest baseline used for endpoint validation.
    /// * `request` - Explicit bounded production request.
    ///
    /// # Errors
    ///
    /// Returns an error when production exceeds the request bound, endpoint
    /// validation fails, or artifact persistence fails.
    fn produce(
        &self,
        model_dir: &Path,
        frozen_manifest: &ProjectManifest,
        request: &ExternalFactProductionRequest,
    ) -> Result<ExternalFactProductionReport>;
}

/// Explicit bounded request for an external exact-fact producer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalFactProductionRequest {
    /// Human-readable source label persisted in fact provenance.
    pub source_label: String,
    /// Optional producer version metadata.
    pub tool_version: Option<String>,
    /// Maximum number of reference facts this request may emit.
    pub max_reference_facts: usize,
}

impl ExternalFactProductionRequest {
    /// Creates a bounded request for reference-only exact fact production.
    ///
    /// # Arguments
    ///
    /// * `source_label` - Explicit producer source label.
    /// * `tool_version` - Optional producer version metadata.
    /// * `max_reference_facts` - Maximum reference facts allowed in one batch.
    pub fn new(
        source_label: impl Into<String>,
        tool_version: Option<String>,
        max_reference_facts: usize,
    ) -> Self {
        Self {
            source_label: source_label.into(),
            tool_version,
            max_reference_facts,
        }
    }
}

/// Capability probe emitted before exact fact production.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalFactProducerProbe {
    /// Typed producer source.
    pub source: ExternalFactSource,
    /// Capability checked by the producer.
    pub capability: ExternalFactProducerCapability,
    /// Explicit source label for reports and provenance.
    pub source_label: String,
    /// Optional producer version metadata.
    pub tool_version: Option<String>,
    /// Whether the capability is available.
    pub available: bool,
    /// Redaction-safe unavailable reason when the probe fails.
    pub unavailable_reason: Option<String>,
}

/// Producer capability supported by the first exact-fact slice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExternalFactProducerCapability {
    /// Reference-only LSP endpoint facts.
    LspReferenceFacts,
}

/// Status of one producer run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExternalFactProductionStatus {
    /// Probe failed and no fact artifact was written.
    Unavailable,
    /// Probe succeeded but the typed producer had no facts to emit.
    NoFacts,
    /// A typed artifact was written through the external fact writer.
    ArtifactWritten,
}

/// Redaction-safe exact-fact production report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalFactProductionReport {
    /// Probe result used to authorize or reject production.
    pub probe: ExternalFactProducerProbe,
    /// Production status.
    pub status: ExternalFactProductionStatus,
    /// Manifest hash used as the frozen production baseline.
    pub manifest_hash_input: String,
    /// Number of reference facts emitted into the artifact.
    pub produced_reference_facts: usize,
    /// Persisted artifact path when a typed artifact was written.
    pub artifact_path: Option<PathBuf>,
    /// Accepted batch metadata when a typed artifact was written.
    pub batch_metadata: Option<ExternalFactBatchMetadata>,
    /// Redaction-safe validation issues when production stopped before write.
    pub issues: Vec<ExternalFactIngestionIssue>,
}

impl ExternalFactProductionReport {
    fn unavailable(probe: ExternalFactProducerProbe, manifest: &ProjectManifest) -> Self {
        Self {
            probe,
            status: ExternalFactProductionStatus::Unavailable,
            manifest_hash_input: manifest.manifest_hash.clone(),
            produced_reference_facts: 0,
            artifact_path: None,
            batch_metadata: None,
            issues: Vec::new(),
        }
    }

    fn no_facts(probe: ExternalFactProducerProbe, manifest: &ProjectManifest) -> Self {
        Self {
            probe,
            status: ExternalFactProductionStatus::NoFacts,
            manifest_hash_input: manifest.manifest_hash.clone(),
            produced_reference_facts: 0,
            artifact_path: None,
            batch_metadata: None,
            issues: Vec::new(),
        }
    }
}

/// Typed fixture-backed LSP reference fact.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LspReferenceFact {
    /// Already-known source endpoint from the manifest graph surface.
    pub from: String,
    /// Already-known target endpoint from the manifest graph surface.
    pub to: String,
    /// Reference relationship kind.
    pub kind: GraphEdgeKind,
    /// Manifest-owned path containing the reference.
    pub path: String,
    /// Optional one-based inclusive start line.
    pub start_line: Option<u32>,
    /// Optional one-based inclusive end line.
    pub end_line: Option<u32>,
}

impl LspReferenceFact {
    /// Creates a typed LSP reference fact over already-known manifest endpoints.
    ///
    /// # Arguments
    ///
    /// * `from` - Source endpoint from the manifest graph surface.
    /// * `to` - Target endpoint from the manifest graph surface.
    /// * `kind` - Reference relationship kind.
    /// * `path` - Manifest-owned source path containing the reference.
    /// * `start_line` - Optional one-based inclusive start line.
    /// * `end_line` - Optional one-based inclusive end line.
    pub fn new(
        from: impl Into<String>,
        to: impl Into<String>,
        kind: GraphEdgeKind,
        path: impl Into<String>,
        start_line: Option<u32>,
        end_line: Option<u32>,
    ) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            kind,
            path: path.into(),
            start_line,
            end_line,
        }
    }
}

/// Fixture-backed LSP producer for the narrowed reference-only first slice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LspFixtureExactFactProducer {
    available: bool,
    unavailable_reason: Option<String>,
    snapshot_label: String,
    references: Vec<LspReferenceFact>,
}

impl LspFixtureExactFactProducer {
    /// Creates an available fixture-backed LSP producer.
    ///
    /// # Arguments
    ///
    /// * `snapshot_label` - Redaction-safe fixture or endpoint snapshot label.
    /// * `references` - Typed reference facts over already-known manifest endpoints.
    pub fn available(snapshot_label: impl Into<String>, references: Vec<LspReferenceFact>) -> Self {
        Self {
            available: true,
            unavailable_reason: None,
            snapshot_label: snapshot_label.into(),
            references,
        }
    }

    /// Creates an unavailable fixture-backed LSP producer.
    ///
    /// # Arguments
    ///
    /// * `reason` - Redaction-safe unavailable reason.
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            available: false,
            unavailable_reason: Some(reason.into()),
            snapshot_label: String::new(),
            references: Vec::new(),
        }
    }

    /// Builds a validated reference-only batch without writing it.
    ///
    /// # Arguments
    ///
    /// * `frozen_manifest` - Immutable manifest baseline used for endpoint validation.
    /// * `request` - Explicit bounded production request.
    ///
    /// # Errors
    ///
    /// Returns an error when the producer is unavailable, the request bound is
    /// exceeded, source metadata is incomplete, or validation rejects endpoints.
    pub fn produce_batch(
        &self,
        frozen_manifest: &ProjectManifest,
        request: &ExternalFactProductionRequest,
    ) -> Result<ExternalFactBatch> {
        let probe = self.probe(request);
        if !probe.available {
            bail!("external fact producer unavailable");
        }
        if request.source_label.trim().is_empty() {
            bail!("external fact producer source label must be explicit");
        }
        if self.references.len() > request.max_reference_facts {
            bail!(
                "external fact producer emitted {} references above request bound {}",
                self.references.len(),
                request.max_reference_facts
            );
        }
        let references = self
            .references
            .iter()
            .cloned()
            .map(|reference| TypedExternalReferenceFact {
                from: reference.from,
                to: reference.to,
                kind: reference.kind,
                path: reference.path,
                start_line: reference.start_line,
                end_line: reference.end_line,
                source: ExternalFactSource::Lsp,
            })
            .collect::<Vec<_>>();
        let facts = TypedExternalFacts { symbols: Vec::new(), references };
        let metadata = ExternalFactBatchMetadata {
            source: ExternalFactSource::Lsp,
            source_label: request.source_label.clone(),
            tool_version: request.tool_version.clone(),
            producer_snapshot_fingerprint: self.snapshot_fingerprint(),
            workspace_root: frozen_manifest.root.to_string_lossy().to_string(),
            source_artifact_fingerprint: String::new(),
            manifest_hash_input: frozen_manifest.manifest_hash.clone(),
            batch_fingerprint: String::new(),
        };
        let batch = prepared_external_fact_artifact_batch(
            frozen_manifest,
            ExternalFactBatch { metadata, facts },
        )?;
        let issues = validate_external_fact_batch(frozen_manifest, &batch);
        if !issues.is_empty() {
            bail!("external fact producer emitted invalid endpoints");
        }
        Ok(batch)
    }

    fn snapshot_fingerprint(&self) -> String {
        let mut references = self.references.clone();
        references.sort();
        let mut content = format!("snapshot:{}\n", self.snapshot_label);
        for reference in references {
            content.push_str(&format!(
                "reference:{}\0{}\0{:?}\0{}\0{:?}\0{:?}\n",
                reference.from,
                reference.to,
                reference.kind,
                reference.path,
                reference.start_line,
                reference.end_line
            ));
        }
        fingerprint(&content)
    }
}

impl ExternalFactProducer for LspFixtureExactFactProducer {
    fn probe(&self, request: &ExternalFactProductionRequest) -> ExternalFactProducerProbe {
        ExternalFactProducerProbe {
            source: ExternalFactSource::Lsp,
            capability: ExternalFactProducerCapability::LspReferenceFacts,
            source_label: request.source_label.clone(),
            tool_version: request.tool_version.clone(),
            available: self.available,
            unavailable_reason: self.unavailable_reason.clone(),
        }
    }

    fn produce(
        &self,
        model_dir: &Path,
        frozen_manifest: &ProjectManifest,
        request: &ExternalFactProductionRequest,
    ) -> Result<ExternalFactProductionReport> {
        let probe = self.probe(request);
        if !probe.available {
            return Ok(ExternalFactProductionReport::unavailable(
                probe,
                frozen_manifest,
            ));
        }
        if self.references.is_empty() {
            return Ok(ExternalFactProductionReport::no_facts(
                probe,
                frozen_manifest,
            ));
        }
        let batch = self.produce_batch(frozen_manifest, request)?;
        let produced_reference_facts = batch.facts.references.len();
        let batch_metadata = batch.metadata.clone();
        let artifact_path = write_external_fact_artifact(model_dir, frozen_manifest, batch)?;
        Ok(ExternalFactProductionReport {
            probe,
            status: ExternalFactProductionStatus::ArtifactWritten,
            manifest_hash_input: frozen_manifest.manifest_hash.clone(),
            produced_reference_facts,
            artifact_path: Some(artifact_path),
            batch_metadata: Some(batch_metadata),
            issues: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::fs;

    use anyhow::Result;
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::indexer::tests::fixture_project;
    use crate::{
        EdgeConfidence, ExternalFactIngestionIssueCode, ProjectIndexer, RetrievalQuery, retrieve,
    };

    fn fixture_request() -> ExternalFactProductionRequest {
        ExternalFactProductionRequest::new(
            "rust-analyzer-fixture",
            Some("fixture-1".to_string()),
            8,
        )
    }

    fn fixture_reference() -> LspReferenceFact {
        LspReferenceFact::new(
            "symbol:src/lib.rs:Struct:Root",
            "symbol:src/model.rs:Enum:Widget",
            GraphEdgeKind::References,
            "src/lib.rs",
            Some(6),
            Some(6),
        )
    }

    #[test]
    fn producer_probe_failure_returns_zero_facts_and_writes_no_artifact() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let producer = LspFixtureExactFactProducer::unavailable("lsp-disabled");

        let actual = producer.produce(setup.model_dir(), &manifest, &fixture_request())?;

        assert_eq!(actual.status, ExternalFactProductionStatus::Unavailable);
        assert_eq!(actual.produced_reference_facts, 0usize);
        assert_eq!(actual.artifact_path, None);
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn successful_lsp_producer_writes_lsp_artifact_with_metadata_and_fingerprint() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let producer =
            LspFixtureExactFactProducer::available("snapshot-a", vec![fixture_reference()]);

        let actual = producer.produce(setup.model_dir(), &manifest, &fixture_request())?;
        let metadata = actual
            .batch_metadata
            .as_ref()
            .expect("successful production should include metadata");

        assert_eq!(actual.status, ExternalFactProductionStatus::ArtifactWritten);
        assert_eq!(actual.produced_reference_facts, 1usize);
        assert_eq!(metadata.source, ExternalFactSource::Lsp);
        assert_eq!(metadata.source_label, "rust-analyzer-fixture");
        assert_eq!(metadata.tool_version, Some("fixture-1".to_string()));
        assert_eq!(metadata.manifest_hash_input, manifest.manifest_hash);
        assert_eq!(metadata.producer_snapshot_fingerprint.len(), 64usize);
        assert_eq!(metadata.batch_fingerprint.len(), 64usize);
        assert!(
            actual
                .artifact_path
                .as_ref()
                .expect("artifact path should be present")
                .is_file()
        );
        Ok(())
    }

    #[test]
    fn writer_rejects_batch_when_manifest_hash_changes_between_production_and_write() -> Result<()>
    {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let producer =
            LspFixtureExactFactProducer::available("snapshot-a", vec![fixture_reference()]);
        let batch = producer.produce_batch(&manifest, &fixture_request())?;
        fs::write(root.join("src").join("lib.rs"), "pub struct Changed;\n")?;
        let changed_manifest = setup.index()?;

        let actual =
            write_external_fact_artifact(setup.model_dir(), &changed_manifest, batch).is_err();
        let expected = true;

        assert_eq!(actual, expected);
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn writer_rejects_batch_when_manifest_hash_changes_but_endpoints_remain_valid() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let producer =
            LspFixtureExactFactProducer::available("snapshot-a", vec![fixture_reference()]);
        let batch = producer.produce_batch(&manifest, &fixture_request())?;
        fs::write(
            root.join("src").join("lib.rs"),
            "use serde::{Serialize, Deserialize};\npub use crate::model::Widget;\nmod model;\nextern crate core;\n\npub struct Root {\n    value: usize,\n}\n\nimpl Root {\n    pub fn new() -> Self {\n        Self { value: 1 }\n    }\n}\n",
        )?;
        let changed_manifest = setup.index()?;

        let actual =
            write_external_fact_artifact(setup.model_dir(), &changed_manifest, batch).is_err();
        let expected = true;

        assert_eq!(actual, expected);
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn unknown_from_endpoint_is_rejected_without_artifact() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let producer = LspFixtureExactFactProducer::available(
            "snapshot-a",
            vec![LspReferenceFact::new(
                "symbol:missing:from",
                "symbol:src/model.rs:Enum:Widget",
                GraphEdgeKind::References,
                "src/lib.rs",
                Some(6),
                Some(6),
            )],
        );

        let actual = producer
            .produce(setup.model_dir(), &manifest, &fixture_request())
            .is_err();
        let issues = validate_external_fact_batch(
            &manifest,
            &ExternalFactBatch {
                metadata: ExternalFactBatchMetadata {
                    source: ExternalFactSource::Lsp,
                    source_label: "rust-analyzer-fixture".to_string(),
                    tool_version: Some("fixture-1".to_string()),
                    producer_snapshot_fingerprint: fingerprint("manual-invalid-from"),
                    workspace_root: manifest.root.to_string_lossy().to_string(),
                    source_artifact_fingerprint: fingerprint("manual-invalid-from"),
                    manifest_hash_input: manifest.manifest_hash.clone(),
                    batch_fingerprint: String::new(),
                },
                facts: TypedExternalFacts {
                    symbols: Vec::new(),
                    references: vec![TypedExternalReferenceFact {
                        from: "symbol:missing:from".to_string(),
                        to: "symbol:src/model.rs:Enum:Widget".to_string(),
                        kind: GraphEdgeKind::References,
                        path: "src/lib.rs".to_string(),
                        start_line: Some(6),
                        end_line: Some(6),
                        source: ExternalFactSource::Lsp,
                    }],
                },
            },
        );

        assert_eq!(actual, true);
        assert_eq!(
            issues.iter().any(|issue| issue.code
                == ExternalFactIngestionIssueCode::MissingReferenceSourceEndpoint),
            true
        );
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn unknown_to_endpoint_is_rejected_without_artifact() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let producer = LspFixtureExactFactProducer::available(
            "snapshot-a",
            vec![LspReferenceFact::new(
                "symbol:src/lib.rs:Struct:Root",
                "symbol:missing:to",
                GraphEdgeKind::References,
                "src/lib.rs",
                Some(6),
                Some(6),
            )],
        );

        let actual = producer
            .produce(setup.model_dir(), &manifest, &fixture_request())
            .is_err();
        let expected = true;

        assert_eq!(actual, expected);
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn path_outside_manifest_files_is_rejected_without_artifact() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let producer = LspFixtureExactFactProducer::available(
            "snapshot-a",
            vec![LspReferenceFact::new(
                "symbol:src/lib.rs:Struct:Root",
                "symbol:src/model.rs:Enum:Widget",
                GraphEdgeKind::References,
                "src/missing.rs",
                Some(1),
                Some(1),
            )],
        );

        let actual = producer
            .produce(setup.model_dir(), &manifest, &fixture_request())
            .is_err();
        let expected = true;

        assert_eq!(actual, expected);
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn producer_does_not_ingest_or_mutate_manifest() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let expected = manifest.clone();
        let producer =
            LspFixtureExactFactProducer::available("snapshot-a", vec![fixture_reference()]);

        let _actual = producer.produce(setup.model_dir(), &manifest, &fixture_request())?;

        assert_eq!(manifest, expected);
        assert_eq!(manifest.external_fact_batches.is_empty(), true);
        Ok(())
    }

    #[test]
    fn indexing_ingests_produced_artifact_and_retrieval_graph_expands_exact_edge() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let base = setup.index()?;
        let producer =
            LspFixtureExactFactProducer::available("snapshot-a", vec![fixture_reference()]);
        producer.produce(setup.model_dir(), &base, &fixture_request())?;

        let manifest = setup.index()?;
        let query = RetrievalQuery {
            text: None,
            path: None,
            path_prefix: None,
            symbol: Some("Root".to_string()),
            limit: 5,
            include_graph_expansion: true,
        };
        let actual = retrieve(&manifest, &query)
            .into_iter()
            .map(|result| result.id)
            .collect::<BTreeSet<_>>();

        assert_eq!(manifest.external_fact_batches.len(), 1usize);
        assert_eq!(
            manifest
                .edges
                .iter()
                .any(|edge| edge.from == "symbol:src/lib.rs:Struct:Root"
                    && edge.to == "symbol:src/model.rs:Enum:Widget"
                    && edge.confidence_kind == EdgeConfidence::ExactCompiler),
            true
        );
        assert_eq!(actual.contains("symbol:src/model.rs:Enum:Widget"), true);
        Ok(())
    }

    #[test]
    fn deterministic_artifact_filename_and_fingerprint_follow_snapshot_and_facts() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let left_setup = ProjectIndexer::new(&root, fixture.path().join("model-left"));
        let right_setup = ProjectIndexer::new(&root, fixture.path().join("model-right"));
        let changed_setup = ProjectIndexer::new(&root, fixture.path().join("model-changed"));
        let manifest = left_setup.index()?;
        let request = fixture_request();
        let left = LspFixtureExactFactProducer::available("snapshot-a", vec![fixture_reference()]);
        let right = LspFixtureExactFactProducer::available("snapshot-a", vec![fixture_reference()]);
        let changed =
            LspFixtureExactFactProducer::available("snapshot-b", vec![fixture_reference()]);

        let left_report = left.produce(left_setup.model_dir(), &manifest, &request)?;
        let right_report = right.produce(right_setup.model_dir(), &manifest, &request)?;
        let changed_report = changed.produce(changed_setup.model_dir(), &manifest, &request)?;
        let left_file = left_report
            .artifact_path
            .as_ref()
            .and_then(|path| path.file_name())
            .expect("left artifact should have filename")
            .to_string_lossy()
            .to_string();
        let right_file = right_report
            .artifact_path
            .as_ref()
            .and_then(|path| path.file_name())
            .expect("right artifact should have filename")
            .to_string_lossy()
            .to_string();
        let changed_file = changed_report
            .artifact_path
            .as_ref()
            .and_then(|path| path.file_name())
            .expect("changed artifact should have filename")
            .to_string_lossy()
            .to_string();

        assert_eq!(left_file, right_file);
        assert_eq!(
            left_report
                .batch_metadata
                .as_ref()
                .map(|metadata| metadata.batch_fingerprint.clone()),
            right_report
                .batch_metadata
                .as_ref()
                .map(|metadata| metadata.batch_fingerprint.clone())
        );
        assert_ne!(left_file, changed_file);
        assert_ne!(
            left_report
                .batch_metadata
                .as_ref()
                .map(|metadata| metadata.batch_fingerprint.clone()),
            changed_report
                .batch_metadata
                .as_ref()
                .map(|metadata| metadata.batch_fingerprint.clone())
        );
        Ok(())
    }
}
