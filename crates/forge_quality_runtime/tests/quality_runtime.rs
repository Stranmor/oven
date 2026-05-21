use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use chrono::{Duration, Utc};
use forge_quality_runtime::*;
use pretty_assertions::assert_eq;
use serde_json::json;
use tempfile::TempDir;

fn artifact(class: ArtifactClass) -> ArtifactReport {
    ArtifactReport {
        schema_version: SCHEMA_VERSION,
        artifact_id: "artifact-1".to_string(),
        artifact_class: class,
        artifact_version: "v1".to_string(),
        artifact_hash: digest_text("artifact"),
        producer_id: "producer".to_string(),
        owner: Some("owner".to_string()),
        claim: ReleaseClaim::InternalReady,
        non_claims: vec!["does not claim public publishing".to_string()],
        dimensions_not_checked: BTreeSet::new(),
        publish_approval_boundary: None,
    }
}

fn evidence() -> EvidenceRef {
    EvidenceRef {
        evidence_id: "evidence-1".to_string(),
        kind: EvidenceKind::TestResult,
        uri: "memory://test".to_string(),
        artifact_hash: digest_text("artifact"),
        produced_at: Utc::now(),
        expires_at: Some(
            Utc::now()
                .checked_add_signed(Duration::hours(1))
                .expect("test evidence expiry must be representable"),
        ),
        digest: digest_text("evidence"),
    }
}

fn gate_result(profile: &QualityProfile, gate: &GateSpec) -> GateResult {
    let dimensions = gate
        .dimensions
        .iter()
        .map(|dimension| (dimension.clone(), GateStatus::Pass))
        .collect::<BTreeMap<_, _>>();
    GateResult {
        schema_version: SCHEMA_VERSION,
        gate_id: gate.gate_id.clone(),
        status: GateStatus::Pass,
        verdict: Some(VectorVerdict { dimensions, max_mode: true }),
        evaluator: EvaluatorRef {
            evaluator_id: format!("critic-{}", gate.gate_id),
            role: EvaluatorRole::Critic,
            independent_from_producer: true,
        },
        evidence: vec![evidence()],
        checked_dimensions: gate.dimensions.clone(),
        dimensions_not_checked: BTreeSet::new(),
        summary: format!(
            "{} passed for {}",
            gate.gate_id, profile.artifact.artifact_id
        ),
    }
}

fn passing_results(profile: &QualityProfile) -> Vec<GateResult> {
    profile
        .gate_graph
        .required_gates
        .iter()
        .map(|gate| gate_result(profile, gate))
        .collect()
}

fn blocker_codes(decision: &ReleaseDecision) -> BTreeSet<ReleaseBlockerCode> {
    decision
        .blockers
        .iter()
        .map(|blocker| blocker.code.clone())
        .collect()
}

fn compile(class: ArtifactClass) -> QualityProfile {
    compile_quality_profile(artifact(class)).expect("quality profile should compile")
}

fn evaluate(profile: &QualityProfile, results: &[GateResult]) -> ReleaseDecision {
    evaluate_release(profile, results, Utc::now()).expect("release evaluation should run")
}

fn first_result_mut(results: &mut [GateResult]) -> &mut GateResult {
    results
        .first_mut()
        .expect("compiled profile should include at least one gate")
}

#[test]
fn quality_profile_compilation_is_deterministic() {
    let report = artifact(ArtifactClass::CodeMcpToolSurface);
    let first = compile_quality_profile(report.clone()).expect("first profile should compile");
    let second = compile_quality_profile(report).expect("second profile should compile");
    assert_eq!(first, second);
}

#[test]
fn missing_release_owner_blocks() {
    let mut artifact = artifact(ArtifactClass::CodeMcpToolSurface);
    artifact.owner = None;
    let profile = compile_quality_profile(artifact).expect("profile should compile");
    let decision = evaluate(&profile, &passing_results(&profile));
    assert_eq!(decision.decision, ReleaseDecisionStatus::Blocked);
    assert!(blocker_codes(&decision).contains(&ReleaseBlockerCode::MissingOwner));
}

#[test]
fn missing_evidence_blocks() {
    let profile = compile(ArtifactClass::CodeMcpToolSurface);
    let mut results = passing_results(&profile);
    first_result_mut(&mut results).evidence.clear();
    let decision = evaluate(&profile, &results);
    let codes = blocker_codes(&decision);
    assert!(codes.contains(&ReleaseBlockerCode::MissingEvidence));
    assert!(codes.contains(&ReleaseBlockerCode::StaleOrMissingEvidence));
}

#[test]
fn scalar_pass_blocks() {
    let profile = compile(ArtifactClass::CodeMcpToolSurface);
    let mut results = passing_results(&profile);
    first_result_mut(&mut results).verdict = None;
    let decision = evaluate(&profile, &results);
    assert!(blocker_codes(&decision).contains(&ReleaseBlockerCode::ScalarOnlyVerdict));
}

#[test]
fn untested_visual_platform_dimension_blocks() {
    let profile = compile(ArtifactClass::RenderedUserFacing);
    let mut results = passing_results(&profile);
    let visual_gate = results
        .iter_mut()
        .find(|result| result.gate_id == "rendered_final_state")
        .expect("rendered profile should include final-state gate");
    visual_gate
        .verdict
        .as_mut()
        .expect("visual gate should have vector verdict")
        .dimensions
        .remove(&QualityDimension::PlatformFit);
    let decision = evaluate(&profile, &results);
    assert!(blocker_codes(&decision).contains(&ReleaseBlockerCode::UntestedRequiredDimension));
}

#[test]
fn same_evaluator_as_producer_blocks() {
    let profile = compile(ArtifactClass::CodeMcpToolSurface);
    let mut results = passing_results(&profile);
    let independent_gate = results
        .iter_mut()
        .find(|result| result.gate_id == "independent_code_critic")
        .expect("code profile should include independent critic gate");
    independent_gate.evaluator = EvaluatorRef {
        evaluator_id: "producer".to_string(),
        role: EvaluatorRole::Producer,
        independent_from_producer: false,
    };
    let decision = evaluate(&profile, &results);
    assert!(blocker_codes(&decision).contains(&ReleaseBlockerCode::SameProducerSelfReview));
}

#[test]
fn public_artifact_without_approval_boundary_blocks() {
    let mut artifact = artifact(ArtifactClass::PublicClientFacing);
    artifact.claim = ReleaseClaim::PublishAdjacent;
    artifact.publish_approval_boundary = Some(PublicApprovalBoundary {
        approval_required: true,
        approval_present: false,
        approval_reference: None,
    });
    let profile = compile_quality_profile(artifact).expect("profile should compile");
    let decision = evaluate(&profile, &passing_results(&profile));
    assert!(blocker_codes(&decision).contains(&ReleaseBlockerCode::MissingPublicApprovalBoundary));
}

#[test]
fn malformed_empty_verdict_blocks() {
    let profile = compile(ArtifactClass::CodeMcpToolSurface);
    let mut results = passing_results(&profile);
    let first = first_result_mut(&mut results);
    first.verdict = Some(VectorVerdict { dimensions: BTreeMap::new(), max_mode: true });
    first.summary.clear();
    let decision = evaluate(&profile, &results);
    assert!(blocker_codes(&decision).contains(&ReleaseBlockerCode::MalformedOrEmptyVerdict));
}

#[test]
fn unknown_artifact_type_fails_closed() {
    assert_eq!(
        ArtifactClass::parse("unknown-webthing"),
        Err(QualityError::UnknownArtifactType)
    );
}

#[test]
fn trace_secret_rejection_and_redaction_work() {
    let mut payload = json!({
        "nested": {
            "api_token": "sk-secret-value-that-must-not-survive"
        }
    });
    assert_eq!(reject_secrets(&payload), Err(QualityError::SecretRejected));
    redact_secrets(&mut payload);
    let redacted = payload
        .pointer("/nested/api_token")
        .expect("redacted token path should exist");
    assert_eq!(redacted, SECRET_REDACTION);
}

#[test]
fn trace_append_readback_and_idempotency_work() {
    let temp = TempDir::new().expect("tempdir should be created");
    let trace_dir = temp.path().join("trace");
    let store =
        TraceStore::new(TraceStoreConfig { project_root: temp.path().to_path_buf(), trace_dir })
            .expect("trace store should initialize");
    let request = TraceAppendRequest {
        project_root: temp.path().to_path_buf(),
        idempotency_key: "same-key".to_string(),
        event_kind: TraceEventKind::RuntimeStatus,
        payload: json!({"status":"ok"}),
    };
    let first = store
        .append(request.clone())
        .expect("first append should work");
    let second = store
        .append(request)
        .expect("idempotent append should work");
    assert_eq!(first.event_id, second.event_id);
    let all_events = store.read_all().expect("trace readback should work");
    assert_eq!(all_events.len(), 1);
    assert_eq!(
        all_events
            .first()
            .expect("trace readback should include event")
            .sequence,
        1
    );
}

#[test]
fn trace_rejects_outside_project_root() {
    let temp = TempDir::new().expect("tempdir should be created");
    let outside = TempDir::new().expect("outside tempdir should be created");
    let trace_dir = temp.path().join("trace");
    let store =
        TraceStore::new(TraceStoreConfig { project_root: temp.path().to_path_buf(), trace_dir })
            .expect("trace store should initialize");
    let result = store.append(TraceAppendRequest {
        project_root: outside.path().to_path_buf(),
        idempotency_key: "outside".to_string(),
        event_kind: TraceEventKind::RuntimeStatus,
        payload: json!({"status":"ok"}),
    });
    assert_eq!(
        result.expect_err("outside root must be rejected"),
        QualityError::ProjectRootRejected
    );
}

#[test]
fn release_blocks_warn_in_max_mode() {
    let profile = compile(ArtifactClass::CodeMcpToolSurface);
    let mut results = passing_results(&profile);
    let first_result = first_result_mut(&mut results);
    let first_dimension = first_result
        .checked_dimensions
        .iter()
        .next()
        .expect("first gate should check at least one dimension")
        .clone();
    first_result
        .verdict
        .as_mut()
        .expect("first gate should have vector verdict")
        .dimensions
        .insert(first_dimension, GateStatus::Warn);
    let decision = evaluate(&profile, &results);
    assert!(blocker_codes(&decision).contains(&ReleaseBlockerCode::WarnInMaxMode));
}

#[test]
fn golden_path_passes() {
    let profile = compile(ArtifactClass::CodeMcpToolSurface);
    let decision = evaluate(&profile, &passing_results(&profile));
    assert_eq!(decision.decision, ReleaseDecisionStatus::Pass);
    assert!(decision.blockers.is_empty());
    assert_eq!(
        decision.passed_dimensions,
        profile.gate_graph.required_dimensions
    );
}

#[test]
fn trace_rejects_excessive_payload_depth() {
    let temp = TempDir::new().expect("tempdir should be created");
    let store = TraceStore::new(TraceStoreConfig {
        project_root: temp.path().to_path_buf(),
        trace_dir: temp.path().join("trace"),
    })
    .expect("trace store should initialize");
    let mut payload = json!("leaf");
    for _ in 0..MAX_JSON_DEPTH {
        payload = json!({"next": payload});
    }
    let result = store.append(TraceAppendRequest {
        project_root: PathBuf::from(temp.path()),
        idempotency_key: "deep".to_string(),
        event_kind: TraceEventKind::RuntimeStatus,
        payload,
    });
    assert_eq!(
        result.expect_err("excessive depth must be rejected"),
        QualityError::PayloadRejected
    );
}

#[test]
fn trace_idempotency_conflict_blocks() {
    let temp = TempDir::new().expect("tempdir should be created");
    let store = TraceStore::new(TraceStoreConfig {
        project_root: temp.path().to_path_buf(),
        trace_dir: temp.path().join("trace"),
    })
    .expect("trace store should initialize");
    let first = TraceAppendRequest {
        project_root: temp.path().to_path_buf(),
        idempotency_key: "conflict-key".to_string(),
        event_kind: TraceEventKind::RuntimeStatus,
        payload: json!({"status":"one"}),
    };
    let second = TraceAppendRequest {
        project_root: temp.path().to_path_buf(),
        idempotency_key: "conflict-key".to_string(),
        event_kind: TraceEventKind::RuntimeStatus,
        payload: json!({"status":"two"}),
    };
    store.append(first).expect("first append should work");
    assert_eq!(
        store
            .append(second)
            .expect_err("conflicting idempotency key must be rejected"),
        QualityError::IdempotencyConflict
    );
}

#[test]
fn trace_detects_digest_tampering() {
    let temp = TempDir::new().expect("tempdir should be created");
    let trace_dir = temp.path().join("trace");
    let store = TraceStore::new(TraceStoreConfig {
        project_root: temp.path().to_path_buf(),
        trace_dir: trace_dir.clone(),
    })
    .expect("trace store should initialize");
    store
        .append(TraceAppendRequest {
            project_root: temp.path().to_path_buf(),
            idempotency_key: "tamper".to_string(),
            event_kind: TraceEventKind::RuntimeStatus,
            payload: json!({"status":"ok"}),
        })
        .expect("append should work");
    let trace_file = trace_dir.join("quality-trace.jsonl");
    let content = fs::read_to_string(&trace_file).expect("trace file should be readable");
    fs::write(
        trace_file,
        content.replace("\"status\":\"ok\"", "\"status\":\"bad\""),
    )
    .expect("tamper write should work");
    assert_eq!(
        store
            .read_all()
            .expect_err("tampered digest chain must be rejected"),
        QualityError::DigestChainInvalid
    );
}

#[test]
fn trace_rejects_truncated_malformed_jsonl() {
    let temp = TempDir::new().expect("tempdir should be created");
    let trace_dir = temp.path().join("trace");
    let store = TraceStore::new(TraceStoreConfig {
        project_root: temp.path().to_path_buf(),
        trace_dir: trace_dir.clone(),
    })
    .expect("trace store should initialize");
    store
        .append(TraceAppendRequest {
            project_root: temp.path().to_path_buf(),
            idempotency_key: "malformed".to_string(),
            event_kind: TraceEventKind::RuntimeStatus,
            payload: json!({"status":"ok"}),
        })
        .expect("append should work");
    fs::write(
        trace_dir.join("quality-trace.jsonl"),
        "{\"schema_version\":1",
    )
    .expect("malformed trace write should work");
    assert!(matches!(
        store
            .read_all()
            .expect_err("malformed trace record must fail closed"),
        QualityError::Json(_)
    ));
}

#[test]
fn trace_rejects_secret_nested_under_digest_like_key() {
    let payload = json!({
        "evidence_id": {
            "api_token": "sk-secret-value-that-must-not-survive"
        }
    });
    assert_eq!(
        reject_secrets(&payload)
            .expect_err("nested secrets under id/digest-like keys must be rejected"),
        QualityError::SecretRejected
    );
}

#[test]
fn release_blocks_required_dimension_declared_not_checked_by_gate_result() {
    let profile = compile(ArtifactClass::CodeMcpToolSurface);
    let mut results = passing_results(&profile);
    let first = first_result_mut(&mut results);
    let dimension = first
        .checked_dimensions
        .iter()
        .next()
        .expect("gate should have a checked dimension")
        .clone();
    first.checked_dimensions.remove(&dimension);
    first.dimensions_not_checked.insert(dimension);
    let decision = evaluate(&profile, &results);
    assert!(blocker_codes(&decision).contains(&ReleaseBlockerCode::UntestedRequiredDimension));
}

#[test]
fn release_blocks_evidence_for_different_artifact_hash() {
    let profile = compile(ArtifactClass::CodeMcpToolSurface);
    let mut results = passing_results(&profile);
    first_result_mut(&mut results)
        .evidence
        .first_mut()
        .expect("gate should have evidence")
        .artifact_hash = digest_text("different-artifact");
    let decision = evaluate(&profile, &results);
    assert!(blocker_codes(&decision).contains(&ReleaseBlockerCode::StaleOrMissingEvidence));
}

#[test]
fn trace_rejects_excessive_payload_size() {
    let oversized = "x".repeat(MAX_PAYLOAD_BYTES.saturating_add(1));
    assert_eq!(
        validate_payload(&json!({"large": oversized}))
            .expect_err("oversized payload must be rejected"),
        QualityError::PayloadRejected
    );
}

#[test]
fn concurrent_appends_preserve_monotonic_sequence() {
    let temp = TempDir::new().expect("tempdir should be created");
    let store = Arc::new(
        TraceStore::new(TraceStoreConfig {
            project_root: temp.path().to_path_buf(),
            trace_dir: temp.path().join("trace"),
        })
        .expect("trace store should initialize"),
    );
    let mut handles = Vec::new();
    for index in 0_u8..8 {
        let store = Arc::clone(&store);
        let project_root = temp.path().to_path_buf();
        handles.push(thread::spawn(move || {
            store
                .append(TraceAppendRequest {
                    project_root,
                    idempotency_key: format!("concurrent-{index}"),
                    event_kind: TraceEventKind::RuntimeStatus,
                    payload: json!({"index": index}),
                })
                .expect("concurrent append should work");
        }));
    }
    for handle in handles {
        handle.join().expect("append thread should not panic");
    }
    let events = store.read_all().expect("trace readback should work");
    assert_eq!(events.len(), 8);
    for (offset, event) in events.iter().enumerate() {
        let expected = u64::try_from(offset)
            .expect("test offset should fit u64")
            .checked_add(1)
            .expect("test sequence should not overflow");
        assert_eq!(event.sequence, expected);
    }
}

#[test]
fn stale_evidence_blocks() {
    let profile = compile(ArtifactClass::CodeMcpToolSurface);
    let mut results = passing_results(&profile);
    first_result_mut(&mut results)
        .evidence
        .first_mut()
        .expect("gate should have evidence")
        .produced_at = Utc::now()
        .checked_sub_signed(Duration::hours(48))
        .expect("test timestamp should be representable");
    let decision = evaluate(&profile, &results);
    assert!(blocker_codes(&decision).contains(&ReleaseBlockerCode::StaleOrMissingEvidence));
}

#[test]
fn public_artifact_with_approval_boundary_can_pass() {
    let mut report = artifact(ArtifactClass::PublicClientFacing);
    report.claim = ReleaseClaim::PublishAdjacent;
    report.publish_approval_boundary = Some(PublicApprovalBoundary {
        approval_required: true,
        approval_present: true,
        approval_reference: Some("current-session-private-approval-boundary".to_string()),
    });
    let profile = compile_quality_profile(report).expect("profile should compile");
    let decision = evaluate(&profile, &passing_results(&profile));
    assert_eq!(decision.decision, ReleaseDecisionStatus::Pass);
}
