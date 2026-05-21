use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use uuid::Uuid;

pub const SCHEMA_VERSION: u32 = 1;
pub const MAX_PAYLOAD_BYTES: usize = 64 * 1024;
pub const MAX_JSON_DEPTH: usize = 24;
pub const MAX_EVIDENCE_AGE_SECONDS: i64 = 24 * 60 * 60;
const TRACE_FILE_NAME: &str = "quality-trace.jsonl";
const TRACE_LOCK_FILE_NAME: &str = "quality-trace.lock";
pub const SECRET_REDACTION: &str = "[REDACTED]";

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum QualityError {
    #[error("artifact type is unknown")]
    UnknownArtifactType,
    #[error("configured project root is not canonical or does not match requested project root")]
    ProjectRootRejected,
    #[error("trace payload exceeds bounded size or depth")]
    PayloadRejected,
    #[error("trace payload contains token-like secret material")]
    SecretRejected,
    #[error("trace idempotency key conflicts with existing record")]
    IdempotencyConflict,
    #[error("trace sequence is not monotonic")]
    NonMonotonicSequence,
    #[error("trace record digest chain is invalid")]
    DigestChainInvalid,
    #[error("release request is malformed")]
    MalformedReleaseRequest,
    #[error("io error: {0}")]
    Io(String),
    #[error("json error: {0}")]
    Json(String),
}

impl From<std::io::Error> for QualityError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

impl From<serde_json::Error> for QualityError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value.to_string())
    }
}

pub type QualityResult<T> = Result<T, QualityError>;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactClass {
    CodeMcpToolSurface,
    RenderedUserFacing,
    PublicClientFacing,
    DeploymentRuntime,
    Data,
    StrategyJudgment,
}

impl ArtifactClass {
    pub fn parse(value: &str) -> QualityResult<Self> {
        match value {
            "code" | "mcp_tool_surface" | "code_mcp_tool_surface" => Ok(Self::CodeMcpToolSurface),
            "rendered" | "user_facing" | "rendered_user_facing" => Ok(Self::RenderedUserFacing),
            "public" | "client_facing" | "public_client_facing" => Ok(Self::PublicClientFacing),
            "deployment" | "runtime" | "deployment_runtime" => Ok(Self::DeploymentRuntime),
            "data" => Ok(Self::Data),
            "strategy" | "judgment" | "strategy_judgment" => Ok(Self::StrategyJudgment),
            _ => Err(QualityError::UnknownArtifactType),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum QualityDimension {
    PurposeFit,
    DomainCorrectness,
    TypeSafety,
    McpSchemaSafety,
    SecurityPrivacy,
    AudienceFit,
    PlatformFit,
    StyleBrandFit,
    UsabilityReadability,
    FinalStateSimulation,
    FreshRecipientPerception,
    PersistenceRecovery,
    RuntimeProof,
    DataLineage,
    DataSchema,
    StrategyEvidence,
    IndependentCritic,
    PublicApprovalBoundary,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GateKind {
    DeterministicCheck,
    AdversarialReview,
    IndependentObserver,
    SecurityReview,
    PersistenceReadback,
    RuntimeProbe,
    LineageCheck,
    StrategyCritic,
    PublicApprovalBoundary,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct GateSpec {
    pub gate_id: String,
    pub kind: GateKind,
    pub required: bool,
    pub independence_required: bool,
    pub dimensions: BTreeSet<QualityDimension>,
    pub evidence_required: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct GateGraph {
    pub schema_version: u32,
    pub graph_id: String,
    pub artifact_class: ArtifactClass,
    pub required_gates: Vec<GateSpec>,
    pub required_dimensions: BTreeSet<QualityDimension>,
    pub unknown_policy: UnknownPolicy,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum UnknownPolicy {
    FailClosedBlocked,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ArtifactReport {
    pub schema_version: u32,
    pub artifact_id: String,
    pub artifact_class: ArtifactClass,
    pub artifact_version: String,
    pub artifact_hash: String,
    pub producer_id: String,
    pub owner: Option<String>,
    pub claim: ReleaseClaim,
    pub non_claims: Vec<String>,
    pub dimensions_not_checked: BTreeSet<QualityDimension>,
    pub publish_approval_boundary: Option<PublicApprovalBoundary>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReleaseClaim {
    LocalSmoke,
    InternalReady,
    UserFacingReady,
    PublishAdjacent,
    PublicRelease,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PublicApprovalBoundary {
    pub approval_required: bool,
    pub approval_present: bool,
    pub approval_reference: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct EvidenceRef {
    pub evidence_id: String,
    pub kind: EvidenceKind,
    pub uri: String,
    pub artifact_hash: String,
    pub produced_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    CommandOutput,
    TestResult,
    RenderedPreview,
    RuntimeProbe,
    TraceReadback,
    CriticReport,
    ApprovalRecord,
    SchemaValidation,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct EvaluatorRef {
    pub evaluator_id: String,
    pub role: EvaluatorRole,
    pub independent_from_producer: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EvaluatorRole {
    Producer,
    Critic,
    Observer,
    SecurityReviewer,
    RuntimeProbe,
    ApprovalAuthority,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GateStatus {
    Pass,
    Warn,
    Fail,
    Blocked,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct GateResult {
    pub schema_version: u32,
    pub gate_id: String,
    pub status: GateStatus,
    pub verdict: Option<VectorVerdict>,
    pub evaluator: EvaluatorRef,
    pub evidence: Vec<EvidenceRef>,
    pub checked_dimensions: BTreeSet<QualityDimension>,
    pub dimensions_not_checked: BTreeSet<QualityDimension>,
    pub summary: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct VectorVerdict {
    pub dimensions: BTreeMap<QualityDimension, GateStatus>,
    pub max_mode: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct QualityProfile {
    pub schema_version: u32,
    pub profile_id: String,
    pub artifact: ArtifactReport,
    pub gate_graph: GateGraph,
    pub compiled_at: DateTime<Utc>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReleaseDecision {
    pub schema_version: u32,
    pub decision_id: String,
    pub artifact_id: String,
    pub decision: ReleaseDecisionStatus,
    pub evaluated_at: DateTime<Utc>,
    pub blockers: Vec<ReleaseBlocker>,
    pub passed_dimensions: BTreeSet<QualityDimension>,
    pub missing_dimensions: BTreeSet<QualityDimension>,
    pub non_claims: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReleaseDecisionStatus {
    Pass,
    Blocked,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReleaseBlocker {
    pub code: ReleaseBlockerCode,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReleaseBlockerCode {
    MissingOwner,
    MissingRequiredGate,
    MissingEvidence,
    ScalarOnlyVerdict,
    UntestedRequiredDimension,
    WarnInMaxMode,
    SameProducerSelfReview,
    MissingPublicApprovalBoundary,
    StaleOrMissingEvidence,
    MalformedOrEmptyVerdict,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct TraceEvent {
    pub schema_version: u32,
    pub sequence: u64,
    pub event_id: String,
    pub idempotency_key: String,
    pub timestamp: DateTime<Utc>,
    pub project_root: PathBuf,
    pub event_kind: TraceEventKind,
    pub payload: serde_json::Value,
    pub previous_digest: Option<String>,
    pub record_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TraceEventKind {
    QualityProfileCompiled,
    GateRecorded,
    EvidenceRecorded,
    ReleaseDecisionEvaluated,
    RuntimeStatus,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct TraceAppendRequest {
    pub project_root: PathBuf,
    pub idempotency_key: String,
    pub event_kind: TraceEventKind,
    pub payload: serde_json::Value,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct TraceQuery {
    pub project_root: PathBuf,
    pub limit: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RuntimeStatus {
    pub schema_version: u32,
    pub runtime_id: String,
    pub status: String,
    pub supported_artifact_classes: Vec<ArtifactClass>,
    pub tools: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct QualityProfileCompileRequest {
    pub artifact: ArtifactReport,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReleaseDecisionEvaluateRequest {
    pub profile: QualityProfile,
    pub gate_results: Vec<GateResult>,
    pub now: Option<DateTime<Utc>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct TraceStoreConfig {
    pub project_root: PathBuf,
    pub trace_dir: PathBuf,
}

#[derive(Clone, Debug)]
pub struct TraceStore {
    project_root: PathBuf,
    trace_dir: PathBuf,
    trace_file: PathBuf,
    lock_file: PathBuf,
}

impl TraceStore {
    pub fn new(config: TraceStoreConfig) -> QualityResult<Self> {
        let project_root = canonicalize_existing_dir(&config.project_root)?;
        let trace_dir = if config.trace_dir.exists() {
            canonicalize_existing_dir(&config.trace_dir)?
        } else {
            let parent = config
                .trace_dir
                .parent()
                .ok_or(QualityError::ProjectRootRejected)?;
            let canonical_parent = canonicalize_existing_dir(parent)?;
            if !canonical_parent.starts_with(&project_root) {
                return Err(QualityError::ProjectRootRejected);
            }
            fs::create_dir_all(&config.trace_dir)?;
            canonicalize_existing_dir(&config.trace_dir)?
        };

        if !trace_dir.starts_with(&project_root) {
            return Err(QualityError::ProjectRootRejected);
        }

        Ok(Self {
            project_root,
            trace_file: trace_dir.join(TRACE_FILE_NAME),
            lock_file: trace_dir.join(TRACE_LOCK_FILE_NAME),
            trace_dir,
        })
    }

    pub fn runtime_status(&self) -> RuntimeStatus {
        RuntimeStatus {
            schema_version: SCHEMA_VERSION,
            runtime_id: digest_text(self.project_root.to_string_lossy().as_ref()),
            status: "ready".to_string(),
            supported_artifact_classes: vec![
                ArtifactClass::CodeMcpToolSurface,
                ArtifactClass::RenderedUserFacing,
                ArtifactClass::PublicClientFacing,
                ArtifactClass::DeploymentRuntime,
                ArtifactClass::Data,
                ArtifactClass::StrategyJudgment,
            ],
            tools: vec![
                "runtime_status".to_string(),
                "quality_profile_compile".to_string(),
                "trace_append".to_string(),
                "trace_get".to_string(),
                "trace_query".to_string(),
                "gate_record".to_string(),
                "release_decision_evaluate".to_string(),
                "release_decision_get".to_string(),
            ],
        }
    }

    pub fn append(&self, request: TraceAppendRequest) -> QualityResult<TraceEvent> {
        self.ensure_project_root(&request.project_root)?;
        validate_payload(&request.payload)?;
        reject_secrets(&request.payload)?;

        fs::create_dir_all(&self.trace_dir)?;
        let _lock = self.lock_exclusive()?;
        let existing = self.read_all_unlocked()?;
        if let Some(prior) = existing
            .iter()
            .find(|event| event.idempotency_key == request.idempotency_key)
        {
            if prior.event_kind == request.event_kind && prior.payload == request.payload {
                return Ok(prior.clone());
            }
            return Err(QualityError::IdempotencyConflict);
        }

        let next_sequence = existing
            .last()
            .map_or(Some(1), |event| event.sequence.checked_add(1))
            .ok_or(QualityError::NonMonotonicSequence)?;
        let previous_digest = existing.last().map(|event| event.record_digest.clone());
        let unsigned = UnsignedTraceEvent {
            schema_version: SCHEMA_VERSION,
            sequence: next_sequence,
            event_id: Uuid::new_v4().to_string(),
            idempotency_key: request.idempotency_key,
            timestamp: Utc::now(),
            project_root: self.project_root.clone(),
            event_kind: request.event_kind,
            payload: request.payload,
            previous_digest,
        };
        let record_digest = digest_json(&unsigned)?;
        let event = TraceEvent {
            schema_version: unsigned.schema_version,
            sequence: unsigned.sequence,
            event_id: unsigned.event_id,
            idempotency_key: unsigned.idempotency_key,
            timestamp: unsigned.timestamp,
            project_root: unsigned.project_root,
            event_kind: unsigned.event_kind,
            payload: unsigned.payload,
            previous_digest: unsigned.previous_digest,
            record_digest,
        };
        let line = serde_json::to_string(&event)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.trace_file)?;
        writeln!(file, "{line}")?;
        file.sync_data()?;
        let readback = self
            .read_all_unlocked()?
            .into_iter()
            .find(|candidate| candidate.event_id == event.event_id)
            .ok_or(QualityError::DigestChainInvalid)?;
        Ok(readback)
    }

    pub fn read_all(&self) -> QualityResult<Vec<TraceEvent>> {
        let _lock = self.lock_shared()?;
        let events = self.read_all_unlocked()?;
        Ok(events)
    }

    fn read_all_unlocked(&self) -> QualityResult<Vec<TraceEvent>> {
        if !self.trace_file.exists() {
            return Ok(Vec::new());
        }
        let file = OpenOptions::new().read(true).open(&self.trace_file)?;
        let reader = BufReader::new(file);
        let mut events = Vec::new();
        let mut previous_sequence = 0;
        let mut previous_digest: Option<String> = None;
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let event: TraceEvent = serde_json::from_str(&line)?;
            self.ensure_project_root(&event.project_root)?;
            if event.sequence <= previous_sequence {
                return Err(QualityError::NonMonotonicSequence);
            }
            if event.previous_digest != previous_digest {
                return Err(QualityError::DigestChainInvalid);
            }
            let unsigned = UnsignedTraceEvent::from_event(&event);
            if digest_json(&unsigned)? != event.record_digest {
                return Err(QualityError::DigestChainInvalid);
            }
            previous_sequence = event.sequence;
            previous_digest = Some(event.record_digest.clone());
            events.push(event);
        }
        Ok(events)
    }

    pub fn query(&self, query: TraceQuery) -> QualityResult<Vec<TraceEvent>> {
        self.ensure_project_root(&query.project_root)?;
        let mut events = self.read_all()?;
        if let Some(limit) = query.limit {
            let keep_from = events.len().saturating_sub(limit);
            events = events.split_off(keep_from);
        }
        Ok(events)
    }

    fn lock_exclusive(&self) -> QualityResult<FileLock> {
        self.lock(libc::LOCK_EX)
    }

    fn lock_shared(&self) -> QualityResult<FileLock> {
        self.lock(libc::LOCK_SH)
    }

    fn lock(&self, operation: libc::c_int) -> QualityResult<FileLock> {
        fs::create_dir_all(&self.trace_dir)?;
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&self.lock_file)?;
        let rc = unsafe { libc::flock(file.as_raw_fd(), operation) };
        if rc == 0 {
            Ok(FileLock { file })
        } else {
            Err(QualityError::Io(
                std::io::Error::last_os_error().to_string(),
            ))
        }
    }

    fn ensure_project_root(&self, requested: &Path) -> QualityResult<()> {
        let requested = canonicalize_existing_dir(requested)?;
        if requested == self.project_root {
            Ok(())
        } else {
            Err(QualityError::ProjectRootRejected)
        }
    }
}

struct FileLock {
    file: File,
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
    }
}

#[derive(Clone, Debug, Serialize)]
struct UnsignedTraceEvent {
    schema_version: u32,
    sequence: u64,
    event_id: String,
    idempotency_key: String,
    timestamp: DateTime<Utc>,
    project_root: PathBuf,
    event_kind: TraceEventKind,
    payload: serde_json::Value,
    previous_digest: Option<String>,
}

impl UnsignedTraceEvent {
    fn from_event(event: &TraceEvent) -> Self {
        Self {
            schema_version: event.schema_version,
            sequence: event.sequence,
            event_id: event.event_id.clone(),
            idempotency_key: event.idempotency_key.clone(),
            timestamp: event.timestamp,
            project_root: event.project_root.clone(),
            event_kind: event.event_kind.clone(),
            payload: event.payload.clone(),
            previous_digest: event.previous_digest.clone(),
        }
    }
}

pub fn compile_quality_profile(artifact: ArtifactReport) -> QualityResult<QualityProfile> {
    let gate_graph = compile_gate_graph(&artifact.artifact_class)?;
    let compiled_at = DateTime::<Utc>::from_timestamp(0, 0)
        .expect("unix epoch must be representable for deterministic profiles");
    let profile_id = stable_profile_id(&artifact, &gate_graph)?;
    Ok(QualityProfile {
        schema_version: SCHEMA_VERSION,
        profile_id,
        artifact,
        gate_graph,
        compiled_at,
    })
}

pub fn compile_gate_graph(artifact_class: &ArtifactClass) -> QualityResult<GateGraph> {
    let required_gates = match artifact_class {
        ArtifactClass::CodeMcpToolSurface => vec![
            gate(
                "code_tests",
                GateKind::DeterministicCheck,
                false,
                &[
                    QualityDimension::DomainCorrectness,
                    QualityDimension::TypeSafety,
                ],
            ),
            gate(
                "mcp_schema_safety",
                GateKind::SecurityReview,
                true,
                &[
                    QualityDimension::McpSchemaSafety,
                    QualityDimension::SecurityPrivacy,
                ],
            ),
            gate(
                "independent_code_critic",
                GateKind::AdversarialReview,
                true,
                &[QualityDimension::IndependentCritic],
            ),
        ],
        ArtifactClass::RenderedUserFacing => vec![
            gate(
                "rendered_final_state",
                GateKind::IndependentObserver,
                true,
                &[
                    QualityDimension::FinalStateSimulation,
                    QualityDimension::FreshRecipientPerception,
                    QualityDimension::PlatformFit,
                    QualityDimension::UsabilityReadability,
                ],
            ),
            gate(
                "domain_review",
                GateKind::AdversarialReview,
                true,
                &[
                    QualityDimension::DomainCorrectness,
                    QualityDimension::AudienceFit,
                ],
            ),
        ],
        ArtifactClass::PublicClientFacing => vec![
            gate(
                "public_approval_boundary",
                GateKind::PublicApprovalBoundary,
                true,
                &[QualityDimension::PublicApprovalBoundary],
            ),
            gate(
                "cold_recipient_review",
                GateKind::IndependentObserver,
                true,
                &[
                    QualityDimension::FreshRecipientPerception,
                    QualityDimension::AudienceFit,
                    QualityDimension::StyleBrandFit,
                    QualityDimension::PlatformFit,
                ],
            ),
            gate(
                "risk_scan",
                GateKind::SecurityReview,
                true,
                &[QualityDimension::SecurityPrivacy],
            ),
        ],
        ArtifactClass::DeploymentRuntime => vec![
            gate(
                "runtime_probe",
                GateKind::RuntimeProbe,
                false,
                &[QualityDimension::RuntimeProof],
            ),
            gate(
                "security_review",
                GateKind::SecurityReview,
                true,
                &[QualityDimension::SecurityPrivacy],
            ),
            gate(
                "persistence_recovery",
                GateKind::PersistenceReadback,
                false,
                &[QualityDimension::PersistenceRecovery],
            ),
        ],
        ArtifactClass::Data => vec![
            gate(
                "schema_lineage",
                GateKind::LineageCheck,
                false,
                &[QualityDimension::DataSchema, QualityDimension::DataLineage],
            ),
            gate(
                "consumer_readiness",
                GateKind::AdversarialReview,
                true,
                &[
                    QualityDimension::PurposeFit,
                    QualityDimension::UsabilityReadability,
                ],
            ),
        ],
        ArtifactClass::StrategyJudgment => vec![
            gate(
                "strategy_evidence",
                GateKind::StrategyCritic,
                true,
                &[
                    QualityDimension::StrategyEvidence,
                    QualityDimension::IndependentCritic,
                ],
            ),
            gate(
                "decision_readiness",
                GateKind::AdversarialReview,
                true,
                &[
                    QualityDimension::PurposeFit,
                    QualityDimension::DomainCorrectness,
                ],
            ),
        ],
    };
    let required_dimensions = required_gates
        .iter()
        .flat_map(|gate| gate.dimensions.iter().cloned())
        .collect();
    Ok(GateGraph {
        schema_version: SCHEMA_VERSION,
        graph_id: format!("gate_graph:{}", stable_class_name(artifact_class)),
        artifact_class: artifact_class.clone(),
        required_gates,
        required_dimensions,
        unknown_policy: UnknownPolicy::FailClosedBlocked,
    })
}

pub fn evaluate_release(
    profile: &QualityProfile,
    gate_results: &[GateResult],
    now: DateTime<Utc>,
) -> QualityResult<ReleaseDecision> {
    if profile.artifact.artifact_id.trim().is_empty()
        || profile.artifact.producer_id.trim().is_empty()
    {
        return Err(QualityError::MalformedReleaseRequest);
    }

    let mut blockers = Vec::new();
    let mut passed_dimensions = BTreeSet::new();
    let mut checked_dimensions = BTreeSet::new();

    if profile
        .artifact
        .owner
        .as_deref()
        .unwrap_or_default()
        .trim()
        .is_empty()
    {
        blockers.push(blocker(
            ReleaseBlockerCode::MissingOwner,
            "release owner is required",
        ));
    }

    if matches!(
        profile.artifact.claim,
        ReleaseClaim::PublishAdjacent | ReleaseClaim::PublicRelease
    ) && !approval_boundary_present(&profile.artifact.publish_approval_boundary)
    {
        blockers.push(blocker(
            ReleaseBlockerCode::MissingPublicApprovalBoundary,
            "public or publish-adjacent claim requires explicit approval boundary",
        ));
    }

    let by_gate: BTreeMap<&str, &GateResult> = gate_results
        .iter()
        .map(|result| (result.gate_id.as_str(), result))
        .collect();

    for required_gate in &profile.gate_graph.required_gates {
        let Some(result) = by_gate.get(required_gate.gate_id.as_str()) else {
            blockers.push(blocker(
                ReleaseBlockerCode::MissingRequiredGate,
                format!("required gate '{}' is missing", required_gate.gate_id),
            ));
            continue;
        };

        if result.summary.trim().is_empty()
            || matches!(result.status, GateStatus::Blocked | GateStatus::Fail)
        {
            blockers.push(blocker(
                ReleaseBlockerCode::MalformedOrEmptyVerdict,
                format!(
                    "gate '{}' has malformed, empty, fail, or blocked verdict",
                    result.gate_id
                ),
            ));
        }

        if required_gate.independence_required
            && (!result.evaluator.independent_from_producer
                || result.evaluator.evaluator_id == profile.artifact.producer_id
                || matches!(result.evaluator.role, EvaluatorRole::Producer))
        {
            blockers.push(blocker(
                ReleaseBlockerCode::SameProducerSelfReview,
                format!("gate '{}' requires independent evaluator", result.gate_id),
            ));
        }

        if required_gate.evidence_required && result.evidence.is_empty() {
            blockers.push(blocker(
                ReleaseBlockerCode::MissingEvidence,
                format!("gate '{}' requires evidence", result.gate_id),
            ));
        }

        if result.evidence.is_empty()
            || result.evidence.iter().any(|evidence| {
                evidence_is_stale(evidence, now)
                    || evidence.digest.trim().is_empty()
                    || evidence.artifact_hash != profile.artifact.artifact_hash
            })
        {
            blockers.push(blocker(
                ReleaseBlockerCode::StaleOrMissingEvidence,
                format!("gate '{}' has stale or missing evidence", result.gate_id),
            ));
        }

        let Some(verdict) = &result.verdict else {
            blockers.push(blocker(
                ReleaseBlockerCode::ScalarOnlyVerdict,
                format!("gate '{}' has scalar-only verdict", result.gate_id),
            ));
            continue;
        };

        if verdict.dimensions.is_empty() {
            blockers.push(blocker(
                ReleaseBlockerCode::MalformedOrEmptyVerdict,
                format!("gate '{}' has empty vector verdict", result.gate_id),
            ));
        }

        for dimension in &required_gate.dimensions {
            if result.dimensions_not_checked.contains(dimension)
                || !result.checked_dimensions.contains(dimension)
            {
                blockers.push(blocker(
                    ReleaseBlockerCode::UntestedRequiredDimension,
                    format!("dimension '{dimension:?}' was declared not checked"),
                ));
                continue;
            }
            match verdict.dimensions.get(dimension) {
                Some(GateStatus::Pass) => {
                    passed_dimensions.insert(dimension.clone());
                    checked_dimensions.insert(dimension.clone());
                }
                Some(GateStatus::Warn) if verdict.max_mode => {
                    checked_dimensions.insert(dimension.clone());
                    blockers.push(blocker(
                        ReleaseBlockerCode::WarnInMaxMode,
                        format!("dimension '{dimension:?}' is WARN in max mode"),
                    ));
                }
                Some(_) => {
                    checked_dimensions.insert(dimension.clone());
                    blockers.push(blocker(
                        ReleaseBlockerCode::UntestedRequiredDimension,
                        format!("dimension '{dimension:?}' did not pass"),
                    ));
                }
                None => blockers.push(blocker(
                    ReleaseBlockerCode::UntestedRequiredDimension,
                    format!("dimension '{dimension:?}' was not checked"),
                )),
            }
        }
    }

    for dimension in &profile.artifact.dimensions_not_checked {
        if profile.gate_graph.required_dimensions.contains(dimension) {
            blockers.push(blocker(
                ReleaseBlockerCode::UntestedRequiredDimension,
                format!("required dimension '{dimension:?}' is declared not checked"),
            ));
        }
    }

    let missing_dimensions = profile
        .gate_graph
        .required_dimensions
        .difference(&checked_dimensions)
        .cloned()
        .collect::<BTreeSet<_>>();
    for dimension in &missing_dimensions {
        blockers.push(blocker(
            ReleaseBlockerCode::UntestedRequiredDimension,
            format!("required dimension '{dimension:?}' has no result"),
        ));
    }

    Ok(ReleaseDecision {
        schema_version: SCHEMA_VERSION,
        decision_id: Uuid::new_v4().to_string(),
        artifact_id: profile.artifact.artifact_id.clone(),
        decision: if blockers.is_empty() {
            ReleaseDecisionStatus::Pass
        } else {
            ReleaseDecisionStatus::Blocked
        },
        evaluated_at: now,
        blockers,
        passed_dimensions,
        missing_dimensions,
        non_claims: profile.artifact.non_claims.clone(),
    })
}

pub fn digest_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hex::encode(hasher.finalize())
}

pub fn digest_json<T: Serialize>(value: &T) -> QualityResult<String> {
    let bytes = serde_json::to_vec(value)?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(hex::encode(hasher.finalize()))
}

pub fn redact_secrets(value: &mut serde_json::Value) {
    redact_secrets_for_key(None, value);
}

fn redact_secrets_for_key(parent_key: Option<&str>, value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, child) in map.iter_mut() {
                if is_secret_key(key) || is_secret_value_for_key(key, child) {
                    *child = serde_json::Value::String(SECRET_REDACTION.to_string());
                } else {
                    redact_secrets_for_key(Some(key), child);
                }
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                redact_secrets_for_key(parent_key, item);
            }
        }
        serde_json::Value::String(text)
            if !parent_key.is_some_and(is_safe_digest_key) && looks_like_secret(text) =>
        {
            *value = serde_json::Value::String(SECRET_REDACTION.to_string());
        }
        _ => {}
    }
}

pub fn reject_secrets(value: &serde_json::Value) -> QualityResult<()> {
    reject_secrets_for_key(None, value)
}

fn reject_secrets_for_key(
    parent_key: Option<&str>,
    value: &serde_json::Value,
) -> QualityResult<()> {
    match value {
        serde_json::Value::Object(map) => {
            for (key, child) in map {
                if is_secret_key(key) || is_secret_value_for_key(key, child) {
                    return Err(QualityError::SecretRejected);
                }
                reject_secrets_for_key(Some(key), child)?;
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                reject_secrets_for_key(parent_key, item)?;
            }
        }
        serde_json::Value::String(text)
            if !parent_key.is_some_and(is_safe_digest_key) && looks_like_secret(text) =>
        {
            return Err(QualityError::SecretRejected);
        }
        _ => {}
    }
    Ok(())
}

pub fn validate_payload(value: &serde_json::Value) -> QualityResult<()> {
    let bytes = serde_json::to_vec(value)?.len();
    if bytes > MAX_PAYLOAD_BYTES || json_depth(value) > MAX_JSON_DEPTH {
        Err(QualityError::PayloadRejected)
    } else {
        Ok(())
    }
}

fn gate(
    gate_id: &str,
    kind: GateKind,
    independence_required: bool,
    dimensions: &[QualityDimension],
) -> GateSpec {
    GateSpec {
        gate_id: gate_id.to_string(),
        kind,
        required: true,
        independence_required,
        dimensions: dimensions.iter().cloned().collect(),
        evidence_required: true,
    }
}

fn blocker(code: ReleaseBlockerCode, message: impl Into<String>) -> ReleaseBlocker {
    ReleaseBlocker { code, message: message.into() }
}

fn approval_boundary_present(boundary: &Option<PublicApprovalBoundary>) -> bool {
    matches!(
        boundary,
        Some(PublicApprovalBoundary {
            approval_required: true,
            approval_present: true,
            approval_reference: Some(reference),
        }) if !reference.trim().is_empty()
    )
}

fn evidence_is_stale(evidence: &EvidenceRef, now: DateTime<Utc>) -> bool {
    if let Some(expires_at) = evidence.expires_at
        && expires_at < now
    {
        return true;
    }
    now.signed_duration_since(evidence.produced_at)
        .num_seconds()
        > MAX_EVIDENCE_AGE_SECONDS
}

fn stable_profile_id(artifact: &ArtifactReport, gate_graph: &GateGraph) -> QualityResult<String> {
    #[derive(Serialize)]
    struct StableProfileIdInput<'a> {
        schema_version: u32,
        artifact: &'a ArtifactReport,
        gate_graph: &'a GateGraph,
    }

    Ok(format!(
        "quality_profile:{}",
        digest_json(&StableProfileIdInput {
            schema_version: SCHEMA_VERSION,
            artifact,
            gate_graph,
        })?
    ))
}

fn stable_class_name(artifact_class: &ArtifactClass) -> &'static str {
    match artifact_class {
        ArtifactClass::CodeMcpToolSurface => "code_mcp_tool_surface",
        ArtifactClass::RenderedUserFacing => "rendered_user_facing",
        ArtifactClass::PublicClientFacing => "public_client_facing",
        ArtifactClass::DeploymentRuntime => "deployment_runtime",
        ArtifactClass::Data => "data",
        ArtifactClass::StrategyJudgment => "strategy_judgment",
    }
}

fn canonicalize_existing_dir(path: &Path) -> QualityResult<PathBuf> {
    let canonical = path.canonicalize()?;
    if canonical.is_dir() {
        Ok(canonical)
    } else {
        Err(QualityError::ProjectRootRejected)
    }
}

fn json_depth(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Array(items) => items
            .iter()
            .map(json_depth)
            .max()
            .unwrap_or(0)
            .saturating_add(1),
        serde_json::Value::Object(map) => map
            .values()
            .map(json_depth)
            .max()
            .unwrap_or(0)
            .saturating_add(1),
        _ => 1,
    }
}

fn is_secret_key(key: &str) -> bool {
    let lowered = key.to_ascii_lowercase();
    lowered.contains("token")
        || lowered.contains("secret")
        || lowered.contains("password")
        || lowered.contains("api_key")
        || lowered.contains("apikey")
        || lowered.contains("authorization")
        || lowered.contains("bearer")
}

fn is_secret_value_for_key(key: &str, value: &serde_json::Value) -> bool {
    if is_safe_digest_key(key) && is_scalar_digest_like_value(value) {
        false
    } else {
        matches!(value, serde_json::Value::String(text) if looks_like_secret(text))
    }
}

fn is_scalar_digest_like_value(value: &serde_json::Value) -> bool {
    matches!(
        value,
        serde_json::Value::Null
            | serde_json::Value::Bool(_)
            | serde_json::Value::Number(_)
            | serde_json::Value::String(_)
    )
}

fn is_safe_digest_key(key: &str) -> bool {
    let lowered = key.to_ascii_lowercase();
    lowered.ends_with("_id")
        || lowered.ends_with("_hash")
        || lowered.ends_with("_digest")
        || lowered == "digest"
        || lowered == "hash"
}

fn looks_like_secret(text: &str) -> bool {
    let lowered = text.to_ascii_lowercase();
    lowered.starts_with("bearer ")
        || lowered.starts_with("sk-")
        || lowered.contains("xoxb-")
        || lowered.contains("ghp_")
        || lowered.contains("github_pat_")
        || (text.len() >= 40
            && text
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'))
}
