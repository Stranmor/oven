//! Typed exact-fact producer boundary for fixture-backed LSP facts.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use serde_json::Value;

use crate::ingestion::{
    prepared_external_fact_artifact_batch, validate_external_fact_batch,
    write_external_fact_artifact,
};
use crate::types::{
    ExternalFactBatch, ExternalFactBatchMetadata, ExternalFactIngestionIssue,
    ExternalFactIngestionIssueCode, ExternalFactSource, GraphEdgeKind, ProjectManifest,
    TypedExternalFacts, TypedExternalReferenceFact,
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

    /// Produces a typed external fact artifact through the project-model
    /// writer.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Project-model directory that owns external fact storage.
    /// * `frozen_manifest` - Immutable manifest baseline used for endpoint
    ///   validation.
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

/// Hard limits for bounded rust-analyzer transport and JSON-RPC parsing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RustAnalyzerBounds {
    /// Maximum source endpoints requested in one production run.
    pub max_files: usize,
    /// Maximum target endpoints requested in one production run.
    pub max_endpoints: usize,
    /// Maximum accepted JSON bytes per message.
    pub max_json_bytes_per_message: usize,
    /// Maximum accepted JSON-RPC messages in one transcript.
    pub max_messages: usize,
    /// Maximum typed reference candidates accepted before writing.
    pub max_references: usize,
    /// Maximum process/probe lifetime.
    pub process_timeout: Duration,
}

impl Default for RustAnalyzerBounds {
    fn default() -> Self {
        Self {
            max_files: 8,
            max_endpoints: 32,
            max_json_bytes_per_message: 64 * 1024,
            max_messages: 32,
            max_references: 128,
            process_timeout: Duration::from_secs(5),
        }
    }
}

/// Bounded rust-analyzer reference production request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RustAnalyzerReferenceRequest {
    /// Base external fact production metadata.
    pub production: ExternalFactProductionRequest,
    /// Manifest-owned files that may be requested from rust-analyzer.
    pub files: Vec<String>,
    /// Manifest-owned source endpoints that may emit references.
    pub from_endpoints: Vec<String>,
    /// Manifest-owned target endpoints that may be referenced.
    pub to_endpoints: Vec<String>,
    /// Transport and parser bounds.
    pub bounds: RustAnalyzerBounds,
}

impl RustAnalyzerReferenceRequest {
    /// Creates a bounded rust-analyzer reference request.
    ///
    /// # Arguments
    ///
    /// * `production` - External fact metadata and reference fact bound.
    /// * `files` - Manifest-owned files that may be requested.
    /// * `from_endpoints` - Known manifest endpoints allowed as reference
    ///   sources.
    /// * `to_endpoints` - Known manifest endpoints allowed as reference
    ///   targets.
    /// * `bounds` - Transport and parser limits.
    pub fn new(
        production: ExternalFactProductionRequest,
        files: Vec<String>,
        from_endpoints: Vec<String>,
        to_endpoints: Vec<String>,
        bounds: RustAnalyzerBounds,
    ) -> Self {
        Self { production, files, from_endpoints, to_endpoints, bounds }
    }
}

/// Rust-analyzer probe capability checked before production.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RustAnalyzerCapability {
    /// Executable and version probe for reference requests.
    References,
}

/// Redaction-safe rust-analyzer capability status.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RustAnalyzerCapabilityStatus {
    /// Capability is usable.
    Available,
    /// Executable was not found or could not be invoked.
    Unavailable,
    /// Executable version was unsupported for this bounded scaffold.
    BadVersion,
    /// Requested capability is not supported by this scaffold.
    UnsupportedCapability,
    /// Initialization or probe failed.
    InitializationFailed,
    /// Probe exceeded the configured timeout and was canceled.
    Timeout,
}

/// Redaction-safe rust-analyzer availability and capability report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RustAnalyzerProbe {
    /// Whether the rust-analyzer executable was available.
    pub executable_available: bool,
    /// Redaction-safe version string when available.
    pub version: Option<String>,
    /// Capability that was probed.
    pub capability: RustAnalyzerCapability,
    /// Capability status.
    pub status: RustAnalyzerCapabilityStatus,
    /// Whether the probe timed out.
    pub timed_out: bool,
    /// Redaction-safe failure reason without raw stdout/stderr payloads.
    pub failure_reason: Option<String>,
}

impl RustAnalyzerProbe {
    fn available(version: String, capability: RustAnalyzerCapability) -> Self {
        Self {
            executable_available: true,
            version: Some(redact_version(&version)),
            capability,
            status: RustAnalyzerCapabilityStatus::Available,
            timed_out: false,
            failure_reason: None,
        }
    }

    fn failure(
        capability: RustAnalyzerCapability,
        status: RustAnalyzerCapabilityStatus,
        timed_out: bool,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            executable_available: !matches!(status, RustAnalyzerCapabilityStatus::Unavailable),
            version: None,
            capability,
            status,
            timed_out,
            failure_reason: Some(reason.into()),
        }
    }
}

/// Process output produced by the read-only rust-analyzer transport boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RustAnalyzerProcessOutput {
    /// Process exit code when the process exited normally.
    pub exit_code: Option<i32>,
    /// Captured stdout bytes, bounded by caller limits before parsing.
    pub stdout: Vec<u8>,
    /// Captured stderr bytes, used only for redaction-safe failure
    /// classification.
    pub stderr: Vec<u8>,
    /// Whether the process timed out and was canceled.
    pub timed_out: bool,
}

/// Raw-data sensor boundary for rust-analyzer execution.
pub trait RustAnalyzerProcess {
    /// Runs `rust-analyzer --version` or equivalent without writing artifacts.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum probe lifetime.
    fn version(&self, timeout: Duration) -> RustAnalyzerProcessOutput;

    /// Runs a bounded LSP transcript request without writing artifacts.
    ///
    /// # Arguments
    ///
    /// * `request` - Bounded reference production request.
    fn references(&self, request: &RustAnalyzerReferenceRequest) -> RustAnalyzerProcessOutput;
}

/// JSON-RPC transport abstraction for fake transcript and live process sensors.
pub trait LspTransport {
    /// Captures bounded JSON-RPC bytes without writing artifacts.
    ///
    /// # Arguments
    ///
    /// * `request` - Bounded reference production request.
    fn capture(&self, request: &RustAnalyzerReferenceRequest) -> RustAnalyzerProcessOutput;
}

impl<T: RustAnalyzerProcess> LspTransport for T {
    fn capture(&self, request: &RustAnalyzerReferenceRequest) -> RustAnalyzerProcessOutput {
        self.references(request)
    }
}

/// Minimal live rust-analyzer process sensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StdRustAnalyzerProcess {
    executable: PathBuf,
}

impl StdRustAnalyzerProcess {
    /// Creates a live process sensor for the supplied executable path.
    ///
    /// # Arguments
    ///
    /// * `executable` - rust-analyzer executable path or command name.
    pub fn new(executable: impl Into<PathBuf>) -> Self {
        Self { executable: executable.into() }
    }
}

impl RustAnalyzerProcess for StdRustAnalyzerProcess {
    fn version(&self, timeout: Duration) -> RustAnalyzerProcessOutput {
        run_command_with_timeout(&self.executable, &["--version"], timeout)
    }

    fn references(&self, _request: &RustAnalyzerReferenceRequest) -> RustAnalyzerProcessOutput {
        RustAnalyzerProcessOutput {
            exit_code: None,
            stdout: Vec::new(),
            stderr: Vec::new(),
            timed_out: false,
        }
    }
}

/// Probe helper for a rust-analyzer process sensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RustAnalyzerCapabilityProbe<P> {
    process: P,
}

impl<P> RustAnalyzerCapabilityProbe<P> {
    /// Creates a rust-analyzer capability probe.
    ///
    /// # Arguments
    ///
    /// * `process` - Read-only process sensor.
    pub fn new(process: P) -> Self {
        Self { process }
    }
}

impl<P: RustAnalyzerProcess> RustAnalyzerCapabilityProbe<P> {
    /// Probes executable availability and version without writing artifacts.
    ///
    /// # Arguments
    ///
    /// * `capability` - Capability to report.
    /// * `timeout` - Maximum process lifetime.
    pub fn probe(
        &self,
        capability: RustAnalyzerCapability,
        timeout: Duration,
    ) -> RustAnalyzerProbe {
        let output = self.process.version(timeout);
        if output.timed_out {
            return RustAnalyzerProbe::failure(
                capability,
                RustAnalyzerCapabilityStatus::Timeout,
                true,
                "rust_analyzer_probe_timeout",
            );
        }
        if output.exit_code != Some(0) {
            return RustAnalyzerProbe::failure(
                capability,
                RustAnalyzerCapabilityStatus::Unavailable,
                false,
                redaction_safe_process_failure(&output),
            );
        }
        let version = match std::str::from_utf8(&output.stdout) {
            Ok(value) => value.trim().to_string(),
            Err(_) => {
                return RustAnalyzerProbe::failure(
                    capability,
                    RustAnalyzerCapabilityStatus::BadVersion,
                    false,
                    "rust_analyzer_version_not_utf8",
                );
            }
        };
        if !version.to_ascii_lowercase().contains("rust-analyzer") {
            return RustAnalyzerProbe::failure(
                capability,
                RustAnalyzerCapabilityStatus::BadVersion,
                false,
                "rust_analyzer_version_unrecognized",
            );
        }
        RustAnalyzerProbe::available(version, capability)
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
    /// Production failed after probing and no fact artifact was written.
    Failed,
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

    fn failed(
        probe: ExternalFactProducerProbe,
        manifest: &ProjectManifest,
        issues: Vec<ExternalFactIngestionIssue>,
    ) -> Self {
        Self {
            probe,
            status: ExternalFactProductionStatus::Failed,
            manifest_hash_input: manifest.manifest_hash.clone(),
            produced_reference_facts: 0,
            artifact_path: None,
            batch_metadata: None,
            issues,
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
    /// Creates a typed LSP reference fact over already-known manifest
    /// endpoints.
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

/// Explicit manifest-owned position-to-endpoint mapping for native LSP
/// responses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeLspEndpointPosition {
    /// Manifest-owned path for the mapped position.
    pub path: String,
    /// Zero-based LSP line for the mapped position.
    pub line: u32,
    /// Zero-based LSP character for the mapped position.
    pub character: u32,
    /// Manifest-owned endpoint at this exact position.
    pub endpoint: String,
}

impl NativeLspEndpointPosition {
    /// Creates an explicit native LSP endpoint position mapping.
    ///
    /// # Arguments
    ///
    /// * `path` - Manifest-owned path for the mapped position.
    /// * `line` - Zero-based LSP line for the mapped position.
    /// * `character` - Zero-based LSP character for the mapped position.
    /// * `endpoint` - Manifest-owned endpoint at this exact position.
    pub fn new(
        path: impl Into<String>,
        line: u32,
        character: u32,
        endpoint: impl Into<String>,
    ) -> Self {
        Self {
            path: path.into(),
            line,
            character,
            endpoint: endpoint.into(),
        }
    }
}

/// Bounded native JSON-RPC response normalization request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeLspReferenceNormalizationRequest {
    /// Expected JSON-RPC response identifier.
    pub response_id: u64,
    /// Manifest-owned endpoint that requested references.
    pub from_endpoint: String,
    /// Explicit manifest-owned mapping from native response locations to
    /// endpoints.
    pub endpoint_positions: Vec<NativeLspEndpointPosition>,
    /// Parser and result bounds.
    pub bounds: RustAnalyzerBounds,
}

impl NativeLspReferenceNormalizationRequest {
    /// Creates a bounded native LSP reference normalization request.
    ///
    /// # Arguments
    ///
    /// * `response_id` - Expected JSON-RPC response identifier.
    /// * `from_endpoint` - Manifest-owned endpoint that requested references.
    /// * `endpoint_positions` - Explicit manifest-owned endpoint position
    ///   mapping.
    /// * `bounds` - Parser and result bounds.
    pub fn new(
        response_id: u64,
        from_endpoint: impl Into<String>,
        endpoint_positions: Vec<NativeLspEndpointPosition>,
        bounds: RustAnalyzerBounds,
    ) -> Self {
        Self {
            response_id,
            from_endpoint: from_endpoint.into(),
            endpoint_positions,
            bounds,
        }
    }
}

/// Pure normalizer for bounded native LSP `Location[]` JSON-RPC responses.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NativeLspReferenceNormalizer;

impl NativeLspReferenceNormalizer {
    /// Normalizes a bounded native LSP response into typed reference facts.
    ///
    /// The normalizer accepts one Content-Length framed JSON-RPC response whose
    /// `result` is a native `Location[]`. Raw JSON remains local to parsing and
    /// all cross-boundary output is typed `LspReferenceFact`.
    ///
    /// # Arguments
    ///
    /// * `manifest` - Frozen manifest that owns every path and endpoint.
    /// * `bytes` - Native LSP response bytes with `Content-Length` framing.
    /// * `request` - Expected response id and explicit endpoint-position
    ///   mapping.
    ///
    /// # Errors
    ///
    /// Returns an error when framing, size, id, path, endpoint, position
    /// mapping, or reference-count validation fails.
    pub fn normalize(
        manifest: &ProjectManifest,
        bytes: &[u8],
        request: &NativeLspReferenceNormalizationRequest,
    ) -> Result<Vec<LspReferenceFact>> {
        validate_native_lsp_mapping(manifest, request)?;
        let message = parse_single_content_length_message(bytes, &request.bounds)?;
        if message.get("error").is_some() {
            bail!("native lsp response returned error");
        }
        let actual_id = message
            .get("id")
            .and_then(Value::as_u64)
            .ok_or_else(|| anyhow::anyhow!("native lsp response id is missing"))?;
        if actual_id != request.response_id {
            bail!("native lsp response id mismatch");
        }
        let items = message
            .get("result")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("native lsp response result is not Location[]"))?;
        if items.len() > request.bounds.max_references {
            bail!("native lsp reference result exceeds reference bound");
        }
        let mut references = Vec::with_capacity(items.len());
        for item in items {
            let (path, start_line, start_character, end_line) =
                parse_native_location(manifest, item)?;
            let endpoint = request
                .endpoint_positions
                .iter()
                .find(|position| {
                    position.path == path
                        && position.line == start_line
                        && position.character == start_character
                })
                .ok_or_else(|| {
                    anyhow::anyhow!("native lsp location has no explicit endpoint mapping")
                })?;
            references.push(LspReferenceFact::new(
                request.from_endpoint.clone(),
                endpoint.endpoint.clone(),
                GraphEdgeKind::References,
                path,
                Some(start_line.saturating_add(1)),
                Some(end_line.saturating_add(1)),
            ));
        }
        Ok(references)
    }
}

/// Bounded JSON-RPC parser for typed LSP reference candidates.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BoundedLspReferenceParser;

impl BoundedLspReferenceParser {
    /// Parses bounded JSON-RPC bytes into typed reference candidates.
    ///
    /// The parser accepts newline-delimited JSON-RPC response objects whose
    /// `result.references` array contains typed candidate fields. It never
    /// persists or returns raw JSON payloads.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Captured JSON-RPC response bytes.
    /// * `bounds` - Parser byte, message, and reference limits.
    ///
    /// # Errors
    ///
    /// Returns an error when bytes, message count, JSON syntax, or reference
    /// count exceed the bounded contract.
    pub fn parse(bytes: &[u8], bounds: &RustAnalyzerBounds) -> Result<Vec<LspReferenceFact>> {
        if bytes.len()
            > bounds
                .max_json_bytes_per_message
                .saturating_mul(bounds.max_messages)
        {
            bail!("lsp transcript exceeds bounded byte budget");
        }
        let text = std::str::from_utf8(bytes)
            .map_err(|_| anyhow::anyhow!("lsp transcript is not utf-8"))?;
        let mut references = Vec::new();
        let mut messages = 0usize;
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            messages = messages.saturating_add(1);
            if messages > bounds.max_messages {
                bail!("lsp transcript exceeds message bound");
            }
            if line.len() > bounds.max_json_bytes_per_message {
                bail!("lsp json-rpc message exceeds byte bound");
            }
            let message: Value = serde_json::from_str(line)
                .map_err(|_| anyhow::anyhow!("lsp json-rpc message is malformed"))?;
            if message.get("error").is_some() {
                bail!("lsp json-rpc response returned error");
            }
            let Some(items) = message
                .get("result")
                .and_then(|result| result.get("references"))
                .and_then(Value::as_array)
            else {
                continue;
            };
            for item in items {
                if references.len() >= bounds.max_references {
                    bail!("lsp reference result exceeds reference bound");
                }
                references.push(parse_reference_item(item)?);
            }
        }
        Ok(references)
    }
}

/// Thin rust-analyzer live/fake producer that writes only typed artifacts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RustAnalyzerReferenceProducer<T> {
    transport: T,
    probe: RustAnalyzerProbe,
}

impl<T> RustAnalyzerReferenceProducer<T> {
    /// Creates a rust-analyzer reference producer from a sensor and probe.
    ///
    /// # Arguments
    ///
    /// * `transport` - Read-only LSP transport.
    /// * `probe` - Redaction-safe capability probe result.
    pub fn new(transport: T, probe: RustAnalyzerProbe) -> Self {
        Self { transport, probe }
    }
}

impl<T: LspTransport> RustAnalyzerReferenceProducer<T> {
    /// Produces a typed external fact artifact from bounded LSP bytes.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Project-model directory that owns external fact storage.
    /// * `frozen_manifest` - Immutable manifest baseline used for validation.
    /// * `request` - Bounded rust-analyzer request.
    ///
    /// # Errors
    ///
    /// Returns an error only when typed artifact writing fails after successful
    /// probe, parsing, and validation.
    pub fn produce(
        &self,
        model_dir: &Path,
        frozen_manifest: &ProjectManifest,
        request: &RustAnalyzerReferenceRequest,
    ) -> Result<ExternalFactProductionReport> {
        let probe = external_probe_from_rust_analyzer(&self.probe, request);
        if self.probe.status != RustAnalyzerCapabilityStatus::Available {
            return Ok(ExternalFactProductionReport::unavailable(
                probe,
                frozen_manifest,
            ));
        }
        let issue = validate_rust_analyzer_request(frozen_manifest, request);
        if let Some(issue) = issue {
            return Ok(ExternalFactProductionReport::failed(
                probe,
                frozen_manifest,
                vec![issue],
            ));
        }
        let output = self.transport.capture(request);
        if output.timed_out {
            return Ok(ExternalFactProductionReport::failed(
                probe,
                frozen_manifest,
                vec![producer_issue("rust_analyzer_timeout")],
            ));
        }
        if output.exit_code != Some(0) {
            return Ok(ExternalFactProductionReport::failed(
                probe,
                frozen_manifest,
                vec![producer_issue(redaction_safe_process_failure(&output))],
            ));
        }
        let references = match BoundedLspReferenceParser::parse(&output.stdout, &request.bounds) {
            Ok(references) => references,
            Err(error) => {
                return Ok(ExternalFactProductionReport::failed(
                    probe,
                    frozen_manifest,
                    vec![producer_issue(error.to_string())],
                ));
            }
        };
        if references.is_empty() {
            return Ok(ExternalFactProductionReport::no_facts(
                probe,
                frozen_manifest,
            ));
        }
        if references.len() > request.production.max_reference_facts {
            return Ok(ExternalFactProductionReport::failed(
                probe,
                frozen_manifest,
                vec![producer_issue(
                    "rust_analyzer_reference_fact_bound_exceeded",
                )],
            ));
        }
        let fixture = LspFixtureExactFactProducer::available(
            rust_analyzer_snapshot_fingerprint(&self.probe, request, &references),
            references,
        );
        let batch = match fixture.produce_batch(frozen_manifest, &request.production) {
            Ok(batch) => batch,
            Err(error) => {
                return Ok(ExternalFactProductionReport::failed(
                    probe,
                    frozen_manifest,
                    vec![producer_issue(error.to_string())],
                ));
            }
        };
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

fn parse_single_content_length_message(bytes: &[u8], bounds: &RustAnalyzerBounds) -> Result<Value> {
    if bounds.max_messages == 0 {
        bail!("native lsp response exceeds message bound");
    }
    if bytes.len()
        > bounds
            .max_json_bytes_per_message
            .saturating_mul(bounds.max_messages)
    {
        bail!("native lsp response exceeds bounded byte budget");
    }
    let header_end = bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| anyhow::anyhow!("native lsp response header is malformed"))?;
    let headers = std::str::from_utf8(&bytes[..header_end])
        .map_err(|_| anyhow::anyhow!("native lsp response header is not utf-8"))?;
    let mut content_length = None;
    for line in headers.split("\r\n") {
        let Some((name, value)) = line.split_once(':') else {
            bail!("native lsp response header is malformed");
        };
        if name.eq_ignore_ascii_case("content-length") {
            content_length = Some(
                value
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| anyhow::anyhow!("native lsp content length is malformed"))?,
            );
        }
    }
    let content_length =
        content_length.ok_or_else(|| anyhow::anyhow!("native lsp content length is missing"))?;
    if content_length > bounds.max_json_bytes_per_message {
        bail!("native lsp json-rpc message exceeds byte bound");
    }
    let body_start = header_end + 4;
    let body_end = body_start
        .checked_add(content_length)
        .ok_or_else(|| anyhow::anyhow!("native lsp content length overflow"))?;
    if body_end != bytes.len() {
        bail!("native lsp response frame length mismatch");
    }
    serde_json::from_slice(&bytes[body_start..body_end])
        .map_err(|_| anyhow::anyhow!("native lsp json-rpc message is malformed"))
}

fn parse_native_location(
    manifest: &ProjectManifest,
    item: &Value,
) -> Result<(String, u32, u32, u32)> {
    let uri = item
        .get("uri")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("native lsp location uri is missing"))?;
    let path = manifest_relative_path_from_uri(manifest, uri)?;
    let range = item
        .get("range")
        .ok_or_else(|| anyhow::anyhow!("native lsp location range is missing"))?;
    let start = range
        .get("start")
        .ok_or_else(|| anyhow::anyhow!("native lsp location start is missing"))?;
    let end = range
        .get("end")
        .ok_or_else(|| anyhow::anyhow!("native lsp location end is missing"))?;
    let start_line = native_u32_field(start, "line")?;
    let start_character = native_u32_field(start, "character")?;
    let end_line = native_u32_field(end, "line")?;
    let _end_character = native_u32_field(end, "character")?;
    if end_line < start_line {
        bail!("native lsp location range is invalid");
    }
    if !manifest.files.iter().any(|file| file.path == path) {
        bail!("native lsp location path is not manifest-owned");
    }
    Ok((path, start_line, start_character, end_line))
}

fn native_u32_field(item: &Value, field: &str) -> Result<u32> {
    item.get(field)
        .and_then(Value::as_u64)
        .and_then(|number| u32::try_from(number).ok())
        .ok_or_else(|| anyhow::anyhow!("native lsp position field is invalid: {field}"))
}

fn manifest_relative_path_from_uri(manifest: &ProjectManifest, uri: &str) -> Result<String> {
    let Some(raw_path) = uri.strip_prefix("file://") else {
        bail!("native lsp location uri is not file scheme");
    };
    let absolute = PathBuf::from(percent_decode_file_uri_path(raw_path)?);
    let relative = absolute
        .strip_prefix(&manifest.root)
        .map_err(|_| anyhow::anyhow!("native lsp location uri is outside manifest root"))?;
    Ok(relative.to_string_lossy().replace('\\', "/"))
}

fn percent_decode_file_uri_path(raw_path: &str) -> Result<String> {
    let bytes = raw_path.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let high = bytes
                .get(index + 1)
                .copied()
                .and_then(hex_value)
                .ok_or_else(|| anyhow::anyhow!("native lsp uri percent escape is malformed"))?;
            let low = bytes
                .get(index + 2)
                .copied()
                .and_then(hex_value)
                .ok_or_else(|| anyhow::anyhow!("native lsp uri percent escape is malformed"))?;
            decoded.push((high << 4) | low);
            index += 3;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(decoded).map_err(|_| anyhow::anyhow!("native lsp uri path is not utf-8"))
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn validate_native_lsp_mapping(
    manifest: &ProjectManifest,
    request: &NativeLspReferenceNormalizationRequest,
) -> Result<()> {
    if request.endpoint_positions.len() > request.bounds.max_endpoints {
        bail!("native lsp endpoint mapping exceeds endpoint bound");
    }
    ensure_manifest_endpoint(manifest, &request.from_endpoint)?;
    for position in &request.endpoint_positions {
        if !manifest.files.iter().any(|file| file.path == position.path) {
            bail!("native lsp mapping path is not manifest-owned");
        }
        ensure_manifest_endpoint(manifest, &position.endpoint)?;
        ensure_endpoint_owns_position(manifest, &position.endpoint, &position.path, position.line)?;
    }
    Ok(())
}

fn ensure_manifest_endpoint(manifest: &ProjectManifest, endpoint: &str) -> Result<()> {
    if manifest_endpoint_strings(manifest)
        .iter()
        .any(|known| *known == endpoint)
    {
        Ok(())
    } else {
        bail!("native lsp endpoint is not manifest-owned")
    }
}

fn ensure_endpoint_owns_position(
    manifest: &ProjectManifest,
    endpoint: &str,
    path: &str,
    zero_based_line: u32,
) -> Result<()> {
    if endpoint == path {
        return Ok(());
    }
    let one_based_line = zero_based_line.saturating_add(1);
    let symbol = manifest
        .symbols
        .iter()
        .find(|symbol| symbol.id == endpoint)
        .ok_or_else(|| anyhow::anyhow!("native lsp mapped endpoint is not a symbol"))?;
    if symbol.path == path
        && symbol.start_line <= one_based_line
        && symbol.end_line >= one_based_line
    {
        Ok(())
    } else {
        bail!("native lsp mapped endpoint does not own position")
    }
}

fn parse_reference_item(item: &Value) -> Result<LspReferenceFact> {
    let from = string_field(item, "from")?;
    let to = string_field(item, "to")?;
    let path = string_field(item, "path")?;
    let kind = match item
        .get("kind")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("lsp reference field is missing: kind"))?
    {
        "Calls" | "calls" => GraphEdgeKind::Calls,
        "References" | "references" => GraphEdgeKind::References,
        _ => bail!("lsp reference kind is unsupported"),
    };
    let start_line = optional_u32_field(item, "start_line")?;
    let end_line = optional_u32_field(item, "end_line")?;
    Ok(LspReferenceFact::new(
        from, to, kind, path, start_line, end_line,
    ))
}

fn string_field(item: &Value, field: &str) -> Result<String> {
    item.get(field)
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("lsp reference field is missing: {field}"))
}

fn optional_u32_field(item: &Value, field: &str) -> Result<Option<u32>> {
    item.get(field)
        .map(|value| {
            value
                .as_u64()
                .and_then(|number| u32::try_from(number).ok())
                .ok_or_else(|| anyhow::anyhow!("lsp reference line field is invalid: {field}"))
        })
        .transpose()
}

fn validate_rust_analyzer_request(
    manifest: &ProjectManifest,
    request: &RustAnalyzerReferenceRequest,
) -> Option<ExternalFactIngestionIssue> {
    if request.files.len() > request.bounds.max_files
        || request.from_endpoints.len() > request.bounds.max_endpoints
        || request.to_endpoints.len() > request.bounds.max_endpoints
        || request.production.max_reference_facts > request.bounds.max_references
    {
        return Some(producer_issue("rust_analyzer_request_bound_exceeded"));
    }
    let files = manifest
        .files
        .iter()
        .map(|file| file.path.as_str())
        .collect::<Vec<_>>();
    let endpoints = manifest_endpoint_strings(manifest);
    for path in &request.files {
        if !files.iter().any(|known| known == path) {
            return Some(producer_issue("rust_analyzer_unknown_path"));
        }
    }
    for endpoint in request
        .from_endpoints
        .iter()
        .chain(request.to_endpoints.iter())
    {
        if !endpoints.iter().any(|known| known == endpoint) {
            return Some(producer_issue("rust_analyzer_unknown_endpoint"));
        }
    }
    None
}

fn manifest_endpoint_strings(manifest: &ProjectManifest) -> Vec<&str> {
    manifest
        .files
        .iter()
        .map(|file| file.path.as_str())
        .chain(manifest.symbols.iter().map(|symbol| symbol.id.as_str()))
        .chain(manifest.shards.iter().map(|shard| shard.id.as_str()))
        .collect()
}

fn external_probe_from_rust_analyzer(
    probe: &RustAnalyzerProbe,
    request: &RustAnalyzerReferenceRequest,
) -> ExternalFactProducerProbe {
    ExternalFactProducerProbe {
        source: ExternalFactSource::Lsp,
        capability: ExternalFactProducerCapability::LspReferenceFacts,
        source_label: request.production.source_label.clone(),
        tool_version: probe
            .version
            .clone()
            .or_else(|| request.production.tool_version.clone()),
        available: probe.status == RustAnalyzerCapabilityStatus::Available,
        unavailable_reason: probe.failure_reason.clone(),
    }
}

fn producer_issue(detail: impl Into<String>) -> ExternalFactIngestionIssue {
    ExternalFactIngestionIssue {
        code: ExternalFactIngestionIssueCode::InvalidExactSourceContract,
        endpoint: None,
        detail: detail.into(),
    }
}

fn redaction_safe_process_failure(output: &RustAnalyzerProcessOutput) -> String {
    if output.timed_out {
        "rust_analyzer_process_timeout".to_string()
    } else if let Some(code) = output.exit_code {
        format!("rust_analyzer_process_exit:{code}")
    } else {
        "rust_analyzer_process_unavailable".to_string()
    }
}

fn redact_version(version: &str) -> String {
    version
        .lines()
        .next()
        .unwrap_or_default()
        .chars()
        .take(120)
        .collect()
}

fn rust_analyzer_snapshot_fingerprint(
    probe: &RustAnalyzerProbe,
    request: &RustAnalyzerReferenceRequest,
    references: &[LspReferenceFact],
) -> String {
    let mut content = format!(
        "probe:{:?}:{:?}\nversion:{}\nmanifest-request:{}:{}:{}\n",
        probe.capability,
        probe.status,
        probe.version.as_deref().unwrap_or_default(),
        request.files.join("\0"),
        request.from_endpoints.join("\0"),
        request.to_endpoints.join("\0")
    );
    let mut sorted = references.to_vec();
    sorted.sort();
    for reference in sorted {
        content.push_str(&format!(
            "{}\0{}\0{:?}\0{}\0{:?}\0{:?}\n",
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

fn run_command_with_timeout(
    executable: &Path,
    args: &[&str],
    timeout: Duration,
) -> RustAnalyzerProcessOutput {
    let mut child = match Command::new(executable)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => {
            return RustAnalyzerProcessOutput {
                exit_code: None,
                stdout: Vec::new(),
                stderr: Vec::new(),
                timed_out: false,
            };
        }
    };
    let deadline = Instant::now()
        .checked_add(timeout)
        .unwrap_or_else(Instant::now);
    loop {
        match child.try_wait() {
            Ok(Some(_status)) => match child.wait_with_output() {
                Ok(output) => {
                    return RustAnalyzerProcessOutput {
                        exit_code: output.status.code(),
                        stdout: output.stdout,
                        stderr: output.stderr,
                        timed_out: false,
                    };
                }
                Err(_) => {
                    return RustAnalyzerProcessOutput {
                        exit_code: None,
                        stdout: Vec::new(),
                        stderr: Vec::new(),
                        timed_out: false,
                    };
                }
            },
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return RustAnalyzerProcessOutput {
                    exit_code: None,
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                    timed_out: true,
                };
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(10)),
            Err(_) => {
                return RustAnalyzerProcessOutput {
                    exit_code: None,
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                    timed_out: false,
                };
            }
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
    /// * `references` - Typed reference facts over already-known manifest
    ///   endpoints.
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
    /// * `frozen_manifest` - Immutable manifest baseline used for endpoint
    ///   validation.
    /// * `request` - Explicit bounded production request.
    ///
    /// # Errors
    ///
    /// Returns an error when the producer is unavailable, the request bound is
    /// exceeded, source metadata is incomplete, or validation rejects
    /// endpoints.
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
    use std::time::Duration;

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

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct FakeRustAnalyzerProcess {
        version: RustAnalyzerProcessOutput,
        references: RustAnalyzerProcessOutput,
    }

    impl RustAnalyzerProcess for FakeRustAnalyzerProcess {
        fn version(&self, _timeout: Duration) -> RustAnalyzerProcessOutput {
            self.version.clone()
        }

        fn references(&self, _request: &RustAnalyzerReferenceRequest) -> RustAnalyzerProcessOutput {
            self.references.clone()
        }
    }

    fn process_output(stdout: impl AsRef<[u8]>) -> RustAnalyzerProcessOutput {
        RustAnalyzerProcessOutput {
            exit_code: Some(0),
            stdout: stdout.as_ref().to_vec(),
            stderr: Vec::new(),
            timed_out: false,
        }
    }

    fn timed_out_output() -> RustAnalyzerProcessOutput {
        RustAnalyzerProcessOutput {
            exit_code: None,
            stdout: Vec::new(),
            stderr: Vec::new(),
            timed_out: true,
        }
    }

    fn unavailable_output() -> RustAnalyzerProcessOutput {
        RustAnalyzerProcessOutput {
            exit_code: None,
            stdout: Vec::new(),
            stderr: Vec::new(),
            timed_out: false,
        }
    }

    fn fake_request(bounds: RustAnalyzerBounds) -> RustAnalyzerReferenceRequest {
        RustAnalyzerReferenceRequest::new(
            fixture_request(),
            vec!["src/lib.rs".to_string()],
            vec!["symbol:src/lib.rs:Struct:Root".to_string()],
            vec!["symbol:src/model.rs:Enum:Widget".to_string()],
            bounds,
        )
    }

    fn valid_lsp_response() -> Vec<u8> {
        br#"{"jsonrpc":"2.0","id":1,"result":{"references":[{"from":"symbol:src/lib.rs:Struct:Root","to":"symbol:src/model.rs:Enum:Widget","kind":"References","path":"src/lib.rs","start_line":6,"end_line":6}]}}
"#.to_vec()
    }

    fn empty_lsp_response() -> Vec<u8> {
        br#"{"jsonrpc":"2.0","id":1,"result":{"references":[]}}
"#
        .to_vec()
    }

    fn available_probe() -> RustAnalyzerProbe {
        RustAnalyzerProbe::available(
            "rust-analyzer 1.2.3".to_string(),
            RustAnalyzerCapability::References,
        )
    }

    fn native_response(body: &str) -> Vec<u8> {
        format!("Content-Length: {}\r\n\r\n{}", body.len(), body).into_bytes()
    }

    fn native_location_response(root: &Path, response_id: u64, relative_path: &str) -> Vec<u8> {
        let uri_path = root
            .join(relative_path)
            .to_string_lossy()
            .replace(' ', "%20");
        native_response(&format!(
            r#"{{"jsonrpc":"2.0","id":{response_id},"result":[{{"uri":"file://{uri_path}","range":{{"start":{{"line":0,"character":0}},"end":{{"line":0,"character":6}}}}}}]}}"#,
        ))
    }

    fn native_request(bounds: RustAnalyzerBounds) -> NativeLspReferenceNormalizationRequest {
        NativeLspReferenceNormalizationRequest::new(
            7,
            "symbol:src/lib.rs:Struct:Root",
            vec![NativeLspEndpointPosition::new(
                "src/model.rs",
                0,
                0,
                "symbol:src/model.rs:Enum:Widget",
            )],
            bounds,
        )
    }

    #[test]
    fn rust_analyzer_version_probe_timeout_returns_timeout_without_artifact_write() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let process = FakeRustAnalyzerProcess {
            version: timed_out_output(),
            references: process_output(valid_lsp_response()),
        };

        let actual = RustAnalyzerCapabilityProbe::new(process)
            .probe(RustAnalyzerCapability::References, Duration::from_secs(1));

        assert_eq!(actual.status, RustAnalyzerCapabilityStatus::Timeout);
        assert_eq!(actual.timed_out, true);
        assert_eq!(
            actual.failure_reason,
            Some("rust_analyzer_probe_timeout".to_string())
        );
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn rust_analyzer_probe_failure_redacts_raw_stderr_from_report_details() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let process = FakeRustAnalyzerProcess {
            version: RustAnalyzerProcessOutput {
                exit_code: Some(1),
                stdout: Vec::new(),
                stderr: b"token=secret-native-json".to_vec(),
                timed_out: false,
            },
            references: process_output(valid_lsp_response()),
        };
        let probe = RustAnalyzerCapabilityProbe::new(process.clone())
            .probe(RustAnalyzerCapability::References, Duration::from_secs(1));
        let producer = RustAnalyzerReferenceProducer::new(process, probe.clone());

        let actual = producer.produce(
            setup.model_dir(),
            &manifest,
            &fake_request(Default::default()),
        )?;

        assert_eq!(
            probe.failure_reason,
            Some("rust_analyzer_process_exit:1".to_string())
        );
        assert_eq!(
            actual.probe.unavailable_reason,
            Some("rust_analyzer_process_exit:1".to_string())
        );
        assert_eq!(actual.status, ExternalFactProductionStatus::Unavailable);
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn native_lsp_location_array_maps_only_through_explicit_manifest_mapping() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let request = native_request(Default::default());

        let actual = NativeLspReferenceNormalizer::normalize(
            &manifest,
            &native_location_response(&root, 7, "src/model.rs"),
            &request,
        )?;
        let expected = vec![LspReferenceFact::new(
            "symbol:src/lib.rs:Struct:Root",
            "symbol:src/model.rs:Enum:Widget",
            GraphEdgeKind::References,
            "src/model.rs",
            Some(1),
            Some(1),
        )];

        assert_eq!(actual, expected);
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn native_lsp_unknown_position_mapping_fails_without_artifact() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let request = NativeLspReferenceNormalizationRequest::new(
            7,
            "symbol:src/lib.rs:Struct:Root",
            vec![NativeLspEndpointPosition::new(
                "src/model.rs",
                1,
                0,
                "symbol:src/model.rs:Enum:Widget",
            )],
            Default::default(),
        );

        let actual = NativeLspReferenceNormalizer::normalize(
            &manifest,
            &native_location_response(&root, 7, "src/model.rs"),
            &request,
        )
        .is_err();

        assert_eq!(actual, true);
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn native_lsp_unknown_mapping_endpoint_fails_without_artifact() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let request = NativeLspReferenceNormalizationRequest::new(
            7,
            "symbol:src/lib.rs:Struct:Root",
            vec![NativeLspEndpointPosition::new(
                "src/model.rs",
                0,
                0,
                "symbol:missing:Endpoint",
            )],
            Default::default(),
        );

        let actual = NativeLspReferenceNormalizer::normalize(
            &manifest,
            &native_location_response(&root, 7, "src/model.rs"),
            &request,
        )
        .is_err();

        assert_eq!(actual, true);
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn native_lsp_response_id_mismatch_fails_without_artifact() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;

        let actual = NativeLspReferenceNormalizer::normalize(
            &manifest,
            &native_location_response(&root, 99, "src/model.rs"),
            &native_request(Default::default()),
        )
        .is_err();

        assert_eq!(actual, true);
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn native_lsp_malformed_header_fails_without_artifact() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;

        let actual = NativeLspReferenceNormalizer::normalize(
            &manifest,
            b"Content-Length nope\r\n\r\n{}",
            &native_request(Default::default()),
        )
        .is_err();

        assert_eq!(actual, true);
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn native_lsp_oversized_frame_and_reference_count_fail_without_artifact() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let oversized = NativeLspReferenceNormalizer::normalize(
            &manifest,
            &native_location_response(&root, 7, "src/model.rs"),
            &native_request(RustAnalyzerBounds {
                max_json_bytes_per_message: 8,
                ..Default::default()
            }),
        )
        .is_err();
        let reference_bound = NativeLspReferenceNormalizer::normalize(
            &manifest,
            &native_location_response(&root, 7, "src/model.rs"),
            &native_request(RustAnalyzerBounds { max_references: 0, ..Default::default() }),
        )
        .is_err();

        assert_eq!(oversized, true);
        assert_eq!(reference_bound, true);
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn native_lsp_raw_json_is_not_persisted_into_report_details() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let process = FakeRustAnalyzerProcess {
            version: process_output("rust-analyzer 1.2.3\n"),
            references: process_output(native_location_response(&root, 99, "src/model.rs")),
        };
        let producer = RustAnalyzerReferenceProducer::new(process, available_probe());

        let actual = producer.produce(
            setup.model_dir(),
            &manifest,
            &fake_request(Default::default()),
        )?;

        assert_eq!(actual.status, ExternalFactProductionStatus::Failed);
        assert_eq!(
            actual
                .issues
                .first()
                .expect("native json report should include redacted issue")
                .detail
                .contains("file://"),
            false
        );
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn rust_analyzer_process_unavailable_returns_unavailable_and_writes_no_artifact() -> Result<()>
    {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let process = FakeRustAnalyzerProcess {
            version: unavailable_output(),
            references: process_output(valid_lsp_response()),
        };
        let probe = RustAnalyzerCapabilityProbe::new(process.clone())
            .probe(RustAnalyzerCapability::References, Duration::from_secs(1));
        let producer = RustAnalyzerReferenceProducer::new(process, probe);

        let actual = producer.produce(
            setup.model_dir(),
            &manifest,
            &fake_request(Default::default()),
        )?;

        assert_eq!(actual.status, ExternalFactProductionStatus::Unavailable);
        assert_eq!(actual.artifact_path, None);
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn rust_analyzer_version_capability_probe_succeeds_without_artifact_write() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let process = FakeRustAnalyzerProcess {
            version: process_output("rust-analyzer 1.2.3\n"),
            references: process_output(valid_lsp_response()),
        };

        let actual = RustAnalyzerCapabilityProbe::new(process)
            .probe(RustAnalyzerCapability::References, Duration::from_secs(1));

        assert_eq!(actual.status, RustAnalyzerCapabilityStatus::Available);
        assert_eq!(actual.version, Some("rust-analyzer 1.2.3".to_string()));
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn rust_analyzer_initialization_timeout_returns_non_writing_report() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let process = FakeRustAnalyzerProcess {
            version: process_output("rust-analyzer 1.2.3\n"),
            references: timed_out_output(),
        };
        let producer = RustAnalyzerReferenceProducer::new(process, available_probe());

        let actual = producer.produce(
            setup.model_dir(),
            &manifest,
            &fake_request(Default::default()),
        )?;

        assert_eq!(actual.status, ExternalFactProductionStatus::Failed);
        assert_eq!(actual.artifact_path, None);
        assert_eq!(
            actual
                .issues
                .first()
                .expect("timeout report should include issue")
                .detail,
            "rust_analyzer_timeout"
        );
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn malformed_json_rpc_returns_typed_failure_without_raw_payload_persistence() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let process = FakeRustAnalyzerProcess {
            version: process_output("rust-analyzer 1.2.3\n"),
            references: process_output(b"{not-json"),
        };
        let producer = RustAnalyzerReferenceProducer::new(process, available_probe());

        let actual = producer.produce(
            setup.model_dir(),
            &manifest,
            &fake_request(Default::default()),
        )?;

        assert_eq!(actual.status, ExternalFactProductionStatus::Failed);
        assert_eq!(actual.artifact_path, None);
        assert_eq!(
            actual
                .issues
                .first()
                .expect("malformed report should include issue")
                .detail,
            "lsp json-rpc message is malformed"
        );
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn missing_json_rpc_reference_kind_returns_typed_failure_without_artifact() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let response = br#"{"jsonrpc":"2.0","id":1,"result":{"references":[{"from":"symbol:src/lib.rs:Struct:Root","to":"symbol:src/model.rs:Enum:Widget","path":"src/lib.rs","start_line":6,"end_line":6}]}}
"#;
        let process = FakeRustAnalyzerProcess {
            version: process_output("rust-analyzer 1.2.3\n"),
            references: process_output(response),
        };
        let producer = RustAnalyzerReferenceProducer::new(process, available_probe());

        let actual = producer.produce(
            setup.model_dir(),
            &manifest,
            &fake_request(Default::default()),
        )?;

        assert_eq!(actual.status, ExternalFactProductionStatus::Failed);
        assert_eq!(actual.artifact_path, None);
        assert_eq!(
            actual
                .issues
                .first()
                .expect("missing kind report should include issue")
                .detail,
            "lsp reference field is missing: kind"
        );
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn valid_json_rpc_with_zero_refs_returns_no_facts_without_artifact() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let process = FakeRustAnalyzerProcess {
            version: process_output("rust-analyzer 1.2.3\n"),
            references: process_output(empty_lsp_response()),
        };
        let producer = RustAnalyzerReferenceProducer::new(process, available_probe());

        let actual = producer.produce(
            setup.model_dir(),
            &manifest,
            &fake_request(Default::default()),
        )?;

        assert_eq!(actual.status, ExternalFactProductionStatus::NoFacts);
        assert_eq!(actual.artifact_path, None);
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn valid_json_rpc_references_over_known_manifest_endpoints_writes_one_artifact() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let process = FakeRustAnalyzerProcess {
            version: process_output("rust-analyzer 1.2.3\n"),
            references: process_output(valid_lsp_response()),
        };
        let producer = RustAnalyzerReferenceProducer::new(process, available_probe());

        let actual = producer.produce(
            setup.model_dir(),
            &manifest,
            &fake_request(Default::default()),
        )?;

        assert_eq!(actual.status, ExternalFactProductionStatus::ArtifactWritten);
        assert_eq!(actual.produced_reference_facts, 1usize);
        assert_eq!(
            actual
                .artifact_path
                .as_ref()
                .is_some_and(|path| path.is_file()),
            true
        );
        Ok(())
    }

    #[test]
    fn rust_analyzer_unknown_path_or_endpoint_rejected_without_artifact() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let process = FakeRustAnalyzerProcess {
            version: process_output("rust-analyzer 1.2.3\n"),
            references: process_output(valid_lsp_response()),
        };
        let request = RustAnalyzerReferenceRequest::new(
            fixture_request(),
            vec!["src/missing.rs".to_string()],
            vec!["symbol:src/lib.rs:Struct:Root".to_string()],
            vec!["symbol:src/model.rs:Enum:Widget".to_string()],
            Default::default(),
        );
        let producer = RustAnalyzerReferenceProducer::new(process, available_probe());

        let actual = producer.produce(setup.model_dir(), &manifest, &request)?;

        assert_eq!(actual.status, ExternalFactProductionStatus::Failed);
        assert_eq!(actual.artifact_path, None);
        assert_eq!(
            actual
                .issues
                .first()
                .expect("unknown path report should include issue")
                .detail,
            "rust_analyzer_unknown_path"
        );
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn rust_analyzer_unknown_response_endpoint_rejected_without_artifact() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let response = br#"{"jsonrpc":"2.0","id":1,"result":{"references":[{"from":"symbol:missing:from","to":"symbol:src/model.rs:Enum:Widget","kind":"References","path":"src/lib.rs","start_line":6,"end_line":6}]}}
"#;
        let process = FakeRustAnalyzerProcess {
            version: process_output("rust-analyzer 1.2.3\n"),
            references: process_output(response),
        };
        let producer = RustAnalyzerReferenceProducer::new(process, available_probe());

        let actual = producer.produce(
            setup.model_dir(),
            &manifest,
            &fake_request(Default::default()),
        )?;

        assert_eq!(actual.status, ExternalFactProductionStatus::Failed);
        assert_eq!(actual.artifact_path, None);
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn rust_analyzer_unknown_request_path_and_endpoint_are_redacted_in_report_details() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let process = FakeRustAnalyzerProcess {
            version: process_output("rust-analyzer 1.2.3\n"),
            references: process_output(valid_lsp_response()),
        };
        let unknown_path_request = RustAnalyzerReferenceRequest::new(
            fixture_request(),
            vec!["src/token=secret-native-json.rs".to_string()],
            vec!["symbol:src/lib.rs:Struct:Root".to_string()],
            vec!["symbol:src/model.rs:Enum:Widget".to_string()],
            Default::default(),
        );
        let unknown_endpoint_request = RustAnalyzerReferenceRequest::new(
            fixture_request(),
            vec!["src/lib.rs".to_string()],
            vec!["symbol:token=secret-native-json:Struct:Root".to_string()],
            vec!["symbol:src/model.rs:Enum:Widget".to_string()],
            Default::default(),
        );
        let producer = RustAnalyzerReferenceProducer::new(process, available_probe());

        let path_report = producer.produce(setup.model_dir(), &manifest, &unknown_path_request)?;
        let endpoint_report = producer.produce(setup.model_dir(), &manifest, &unknown_endpoint_request)?;

        assert_eq!(path_report.status, ExternalFactProductionStatus::Failed);
        assert_eq!(endpoint_report.status, ExternalFactProductionStatus::Failed);
        assert_eq!(
            path_report
                .issues
                .first()
                .expect("unknown path report should include issue")
                .detail
                .contains("secret-native-json"),
            false
        );
        assert_eq!(
            endpoint_report
                .issues
                .first()
                .expect("unknown endpoint report should include issue")
                .detail
                .contains("secret-native-json"),
            false
        );
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn rust_analyzer_bounds_are_enforced_without_artifact() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let process = FakeRustAnalyzerProcess {
            version: process_output("rust-analyzer 1.2.3\n"),
            references: process_output(valid_lsp_response()),
        };
        let producer = RustAnalyzerReferenceProducer::new(process, available_probe());
        let mut bounds = RustAnalyzerBounds { max_files: 0, ..Default::default() };
        let file_bound =
            producer.produce(setup.model_dir(), &manifest, &fake_request(bounds.clone()))?;
        bounds = RustAnalyzerBounds { max_json_bytes_per_message: 8, ..Default::default() };
        let message_bound =
            producer.produce(setup.model_dir(), &manifest, &fake_request(bounds.clone()))?;
        bounds = RustAnalyzerBounds { max_messages: 0, ..Default::default() };
        let message_count_bound =
            producer.produce(setup.model_dir(), &manifest, &fake_request(bounds.clone()))?;
        bounds = RustAnalyzerBounds { max_references: 0, ..Default::default() };
        let reference_bound =
            producer.produce(setup.model_dir(), &manifest, &fake_request(bounds))?;

        assert_eq!(file_bound.status, ExternalFactProductionStatus::Failed);
        assert_eq!(message_bound.status, ExternalFactProductionStatus::Failed);
        assert_eq!(
            message_count_bound.status,
            ExternalFactProductionStatus::Failed
        );
        assert_eq!(reference_bound.status, ExternalFactProductionStatus::Failed);
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn project_indexer_index_still_does_not_spawn_any_process() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));

        let actual = setup.index()?;

        assert_eq!(actual.external_fact_batches.is_empty(), true);
        assert_eq!(setup.model_dir().join("external_facts").exists(), false);
        Ok(())
    }

    #[test]
    fn same_fake_transcript_and_manifest_produce_identical_artifact_identity() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let left_setup = ProjectIndexer::new(&root, fixture.path().join("model-left-ra"));
        let right_setup = ProjectIndexer::new(&root, fixture.path().join("model-right-ra"));
        let manifest = left_setup.index()?;
        let left_process = FakeRustAnalyzerProcess {
            version: process_output("rust-analyzer 1.2.3\n"),
            references: process_output(valid_lsp_response()),
        };
        let right_process = left_process.clone();
        let request = fake_request(Default::default());

        let left = RustAnalyzerReferenceProducer::new(left_process, available_probe()).produce(
            left_setup.model_dir(),
            &manifest,
            &request,
        )?;
        let right = RustAnalyzerReferenceProducer::new(right_process, available_probe()).produce(
            right_setup.model_dir(),
            &manifest,
            &request,
        )?;

        assert_eq!(
            left.artifact_path
                .as_ref()
                .and_then(|path| path.file_name()),
            right
                .artifact_path
                .as_ref()
                .and_then(|path| path.file_name())
        );
        assert_eq!(
            left.batch_metadata
                .as_ref()
                .map(|metadata| metadata.batch_fingerprint.clone()),
            right
                .batch_metadata
                .as_ref()
                .map(|metadata| metadata.batch_fingerprint.clone())
        );
        Ok(())
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
