//! Typestate boundary for readback-verified context-pack commits.

use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;

use thiserror::Error;

use crate::evidence_replay::{
    ReplayActivationBoundary, ReplayEvidenceReadbackStatus, apply_replay_readback_results,
};
use crate::retrieval_plan::{
    ProjectContextReadRequest, ProjectContextRetrievalPlan, ProjectContextWriteDecision,
};
use crate::types::{ContextPack, ContextPackArtifactId, Provenance, ToolEpisode};
use crate::util::fingerprint;

/// Redaction-safe committed query boundary returned by project-model-owned search.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectContextCommittedQueryResult {
    readback: ProjectContextReadbackSummary,
    commit: ProjectContextCommitOutcome,
    episode_append: ProjectContextEpisodeAppendOutcome,
    result_order: Vec<ProjectContextCommittedResultItem>,
}

impl ProjectContextCommittedQueryResult {
    /// Builds a committed query result for a no-write commit outcome.
    ///
    /// # Arguments
    ///
    /// * `readback` - Redaction-safe summary of executed readback requests.
    /// * `reason` - Typed reason that the context pack was not written.
    /// * `result_order` - Redaction-safe legacy result order metadata.
    pub fn no_write(
        readback: ProjectContextReadbackSummary,
        reason: ProjectContextPackNoWriteReason,
        result_order: Vec<ProjectContextCommittedResultItem>,
    ) -> Self {
        Self {
            readback,
            commit: ProjectContextCommitOutcome::NoWrite(reason),
            episode_append: ProjectContextEpisodeAppendOutcome::NotAttempted {
                reason: ProjectContextEpisodeAppendNotAttemptedReason::NoPersistedContextPack,
            },
            result_order,
        }
    }

    /// Builds a committed query result for a persisted context-pack outcome.
    ///
    /// # Arguments
    ///
    /// * `readback` - Redaction-safe summary of executed readback requests.
    /// * `proof` - Persisted proof produced by verified context-pack persistence.
    /// * `episode_append` - Typed persisted-pack episode append outcome.
    /// * `result_order` - Redaction-safe legacy result order metadata.
    pub fn persisted(
        readback: ProjectContextReadbackSummary,
        proof: ProjectContextPackPersistedProof,
        episode_append: ProjectContextPersistedEpisodeAppendOutcome,
        result_order: Vec<ProjectContextCommittedResultItem>,
    ) -> Self {
        Self {
            readback,
            commit: ProjectContextCommitOutcome::Persisted(proof),
            episode_append: episode_append.into(),
            result_order,
        }
    }

    /// Returns redaction-safe readback metadata.
    pub fn readback(&self) -> &ProjectContextReadbackSummary {
        &self.readback
    }

    /// Returns the context-pack commit outcome.
    pub fn commit(&self) -> &ProjectContextCommitOutcome {
        &self.commit
    }

    /// Returns the episode append outcome.
    pub fn episode_append(&self) -> &ProjectContextEpisodeAppendOutcome {
        &self.episode_append
    }

    /// Returns redaction-safe result ordering metadata.
    pub fn result_order(&self) -> &[ProjectContextCommittedResultItem] {
        &self.result_order
    }
}

/// Redaction-safe context-pack commit outcome for a committed project-model query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectContextCommitOutcome {
    /// Readback verification or retrieval policy proved that no pack may be written.
    NoWrite(ProjectContextPackNoWriteReason),
    /// Readback-verified context pack was persisted through the commit typestate boundary.
    Persisted(ProjectContextPackPersistedProof),
}

/// Redaction-safe episode append status for a committed project-model query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectContextEpisodeAppendOutcome {
    /// Episode append was not attempted because no persisted pack exists.
    NotAttempted {
        /// Typed no-attempt reason.
        reason: ProjectContextEpisodeAppendNotAttemptedReason,
    },
    /// Episode append completed successfully.
    Appended {
        /// Stable redaction-safe episode fingerprint.
        episode_fingerprint: String,
    },
    /// Episode append failed after the context pack was already persisted.
    Failed {
        /// Stable redaction-safe failure classification.
        reason_code: ProjectContextEpisodeAppendFailureReason,
    },
}

/// Episode append status that is valid only after a context pack was persisted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectContextPersistedEpisodeAppendOutcome {
    /// Episode append completed successfully.
    Appended {
        /// Stable redaction-safe episode fingerprint.
        episode_fingerprint: String,
    },
    /// Episode append failed after the context pack was already persisted.
    Failed {
        /// Stable redaction-safe failure classification.
        reason_code: ProjectContextEpisodeAppendFailureReason,
    },
}

impl ProjectContextPersistedEpisodeAppendOutcome {
    /// Builds a persisted-pack append success outcome.
    ///
    /// # Arguments
    ///
    /// * `episode_fingerprint` - Stable redaction-safe episode fingerprint.
    pub fn appended(episode_fingerprint: impl Into<String>) -> Self {
        Self::Appended { episode_fingerprint: episode_fingerprint.into() }
    }

    /// Builds a persisted-pack append failure outcome.
    ///
    /// # Arguments
    ///
    /// * `reason_code` - Stable redaction-safe failure classification.
    pub fn failed(reason_code: ProjectContextEpisodeAppendFailureReason) -> Self {
        Self::Failed { reason_code }
    }
}

impl From<ProjectContextPersistedEpisodeAppendOutcome> for ProjectContextEpisodeAppendOutcome {
    fn from(outcome: ProjectContextPersistedEpisodeAppendOutcome) -> Self {
        match outcome {
            ProjectContextPersistedEpisodeAppendOutcome::Appended { episode_fingerprint } => {
                Self::Appended { episode_fingerprint }
            }
            ProjectContextPersistedEpisodeAppendOutcome::Failed { reason_code } => {
                Self::Failed { reason_code }
            }
        }
    }
}

/// Typed reason that episode append was not attempted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectContextEpisodeAppendNotAttemptedReason {
    /// No persisted context pack exists for this committed query.
    NoPersistedContextPack,
}

/// Redaction-safe episode append failure classification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectContextEpisodeAppendFailureReason {
    /// The episode ledger append operation failed.
    EpisodeAppendFailed,
}

/// Redaction-safe readback summary for planned project-model evidence requests.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProjectContextReadbackSummary {
    requested_count: usize,
    succeeded_count: usize,
    failed_count: usize,
    evidence: Vec<ProjectContextReadbackEvidence>,
}

impl ProjectContextReadbackSummary {
    /// Builds a readback summary from typed readback outcomes.
    ///
    /// # Arguments
    ///
    /// * `outcomes` - Readback outcomes produced from validated read requests.
    pub fn from_outcomes(outcomes: &[ProjectContextReadbackOutcome]) -> Self {
        let succeeded_count = outcomes
            .iter()
            .filter(|outcome| outcome.status == ProjectContextReadbackStatus::Succeeded)
            .count();
        let failed_count = outcomes.len().saturating_sub(succeeded_count);
        Self {
            requested_count: outcomes.len(),
            succeeded_count,
            failed_count,
            evidence: outcomes
                .iter()
                .map(ProjectContextReadbackEvidence::from_outcome)
                .collect(),
        }
    }

    /// Returns the number of planned readbacks executed by the service.
    pub fn requested_count(&self) -> usize {
        self.requested_count
    }

    /// Returns the number of successful readbacks.
    pub fn succeeded_count(&self) -> usize {
        self.succeeded_count
    }

    /// Returns the number of failed readbacks.
    pub fn failed_count(&self) -> usize {
        self.failed_count
    }

    /// Returns evidence identifiers and ranges validated through read requests.
    pub fn evidence(&self) -> &[ProjectContextReadbackEvidence] {
        &self.evidence
    }
}

/// Redaction-safe evidence id/range label derived from a validated read request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectContextReadbackEvidence {
    evidence_id: String,
    relative_manifest_path: String,
    start_line: u32,
    end_line: u32,
    status: ProjectContextReadbackStatus,
}

impl ProjectContextReadbackEvidence {
    fn from_outcome(outcome: &ProjectContextReadbackOutcome) -> Self {
        Self {
            evidence_id: outcome.evidence_id.clone(),
            relative_manifest_path: outcome.relative_manifest_path.clone(),
            start_line: outcome.start_line,
            end_line: outcome.end_line,
            status: outcome.status.clone(),
        }
    }

    /// Returns the validated evidence identifier.
    pub fn evidence_id(&self) -> &str {
        &self.evidence_id
    }

    /// Returns the validated manifest-relative path label.
    pub fn relative_manifest_path(&self) -> &str {
        &self.relative_manifest_path
    }

    /// Returns the one-based inclusive start line.
    pub fn start_line(&self) -> u32 {
        self.start_line
    }

    /// Returns the one-based inclusive end line.
    pub fn end_line(&self) -> u32 {
        self.end_line
    }

    /// Returns the typed readback status.
    pub fn status(&self) -> &ProjectContextReadbackStatus {
        &self.status
    }
}

/// Redaction-safe result ordering item for legacy adapters.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectContextCommittedResultItem {
    evidence_id: String,
    relevance: Option<f32>,
}

impl ProjectContextCommittedResultItem {
    /// Builds a result ordering item from redaction-safe metadata.
    ///
    /// # Arguments
    ///
    /// * `evidence_id` - Evidence identifier returned by the planner.
    /// * `relevance` - Optional redaction-safe relevance score.
    pub fn new(evidence_id: impl Into<String>, relevance: Option<f32>) -> Self {
        Self { evidence_id: evidence_id.into(), relevance }
    }

    /// Returns the evidence identifier.
    pub fn evidence_id(&self) -> &str {
        &self.evidence_id
    }

    /// Returns the optional relevance score.
    pub fn relevance(&self) -> Option<f32> {
        self.relevance
    }
}

const PROJECT_MODEL_SEARCH_TOOL: &str = "project_model_search";
const PROJECT_MODEL_SEARCH_SUCCESS: &str = "success";
const PROJECT_MODEL_SEARCH_PROVENANCE_SOURCE: &str = "WorkspaceService::query_workspace";

/// Typestate marker for a selected plan whose read requests have not yet been verified.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadRequestsSelected;

/// Typestate marker for a context-pack commit whose readback evidence has been verified.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadbackVerified;

/// Typestate boundary for the project-model context-pack commit lifecycle.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectContextPackCommit<State> {
    state: ProjectContextPackCommitState,
    _state: PhantomData<State>,
}

#[derive(Clone, Debug, PartialEq)]
enum ProjectContextPackCommitState {
    Selected(Box<ProjectContextPackCommitSelected>),
    Verified(ProjectContextPackCommitVerified),
}

#[derive(Clone, Debug, PartialEq)]
struct ProjectContextPackCommitSelected {
    context_pack: Option<ContextPack>,
    read_requests: Vec<ProjectContextReadRequest>,
    write_decision: ProjectContextWriteDecision,
    replay_activation: ReplayActivationBoundary,
}

#[derive(Clone, Debug, PartialEq)]
struct ProjectContextPackCommitVerified {
    context_pack: ContextPack,
}

/// Typed readback status produced by the service IO boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectContextReadbackStatus {
    /// The service successfully read the planned path/range.
    Succeeded,
    /// The service failed to read the planned path/range.
    Failed,
}

/// Typed readback outcome for exactly one planned read request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectContextReadbackOutcome {
    evidence_id: String,
    relative_manifest_path: String,
    start_line: u32,
    end_line: u32,
    status: ProjectContextReadbackStatus,
}

impl ProjectContextReadbackOutcome {
    /// Builds a successful readback outcome from the executed request.
    ///
    /// # Arguments
    ///
    /// * `request` - Planned read request that was executed successfully.
    pub fn succeeded(request: &ProjectContextReadRequest) -> Self {
        Self::from_request(request, ProjectContextReadbackStatus::Succeeded)
    }

    /// Builds a failed readback outcome from the executed request.
    ///
    /// # Arguments
    ///
    /// * `request` - Planned read request that failed at the service IO boundary.
    pub fn failed(request: &ProjectContextReadRequest) -> Self {
        Self::from_request(request, ProjectContextReadbackStatus::Failed)
    }

    /// Builds a readback outcome from explicit typed metadata.
    ///
    /// # Arguments
    ///
    /// * `evidence_id` - Evidence identifier reported by readback.
    /// * `relative_manifest_path` - Manifest-relative path reported by readback.
    /// * `start_line` - One-based inclusive start line reported by readback.
    /// * `end_line` - One-based inclusive end line reported by readback.
    /// * `status` - Readback status reported by the IO boundary.
    ///
    /// # Errors
    ///
    /// Returns an error when the readback path or line range is invalid.
    pub fn new(
        evidence_id: impl Into<String>,
        relative_manifest_path: impl Into<String>,
        start_line: u32,
        end_line: u32,
        status: ProjectContextReadbackStatus,
    ) -> anyhow::Result<Self> {
        let request = ProjectContextReadRequest::new(
            relative_manifest_path,
            evidence_id,
            start_line,
            end_line,
        )?;
        Ok(Self::from_request(&request, status))
    }

    fn from_request(
        request: &ProjectContextReadRequest,
        status: ProjectContextReadbackStatus,
    ) -> Self {
        Self {
            evidence_id: request.evidence_id.clone(),
            relative_manifest_path: request.relative_manifest_path().to_string(),
            start_line: request.start_line,
            end_line: request.end_line,
            status,
        }
    }

    /// Returns the evidence identifier carried by this outcome.
    pub fn evidence_id(&self) -> &str {
        &self.evidence_id
    }

    /// Returns the readback status.
    pub fn status(&self) -> &ProjectContextReadbackStatus {
        &self.status
    }
}

/// Deterministic no-write reason produced by the commit lifecycle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectContextPackNoWriteReason {
    /// Retrieval selected no evidence.
    EmptyEvidence,
    /// A non-replay readback failed and therefore blocks persistence.
    RequiredReadbackFailed,
    /// Replay filtering left no verified evidence to persist.
    NoVerifiedEvidenceAfterReplayFiltering,
}

/// Typed no-write state that cannot produce a write instruction or episode payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectContextPackNoWrite {
    reason: ProjectContextPackNoWriteReason,
}

impl ProjectContextPackNoWrite {
    /// Returns the typed no-write reason.
    pub fn reason(&self) -> &ProjectContextPackNoWriteReason {
        &self.reason
    }
}

/// Commit decision after typed readback verification.
#[derive(Clone, Debug, PartialEq)]
pub enum ProjectContextPackReadbackDecision {
    /// No context pack may be written.
    NoWrite(ProjectContextPackNoWrite),
    /// A readback-verified context pack may be persisted.
    Write(Box<ProjectContextPackCommit<ReadbackVerified>>),
}

/// Typed context-pack write instruction produced only by verified readback state.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectContextPackWriteInstruction {
    context_pack: ContextPack,
}

impl ProjectContextPackWriteInstruction {
    /// Returns the verified context pack that must be persisted.
    pub fn context_pack(&self) -> &ContextPack {
        &self.context_pack
    }
}

/// Readback-verified proof produced after durable context-pack persistence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectContextPackPersistedProof {
    artifact_id: ContextPackArtifactId,
    artifact_path: String,
    manifest_hash: String,
}

impl ProjectContextPackPersistedProof {
    fn from_persisted_commit(
        artifact_id: ContextPackArtifactId,
        artifact_path: impl Into<String>,
        context_pack: &ContextPack,
    ) -> Self {
        Self {
            artifact_id,
            artifact_path: artifact_path.into(),
            manifest_hash: context_pack.manifest_hash.clone(),
        }
    }

    /// Returns the persisted context-pack artifact id.
    pub fn artifact_id(&self) -> &ContextPackArtifactId {
        &self.artifact_id
    }

    /// Returns the redaction-safe artifact path label.
    pub fn artifact_path(&self) -> &str {
        &self.artifact_path
    }

    /// Returns the manifest hash proven by the persisted pack.
    pub fn manifest_hash(&self) -> &str {
        &self.manifest_hash
    }
}

/// Redaction-safe input used to build a project-model search episode.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProjectModelSearchEpisodeInput {
    /// Search query text used only as fingerprint input.
    pub query: String,
    /// Search use-case text used only as fingerprint input.
    pub use_case: String,
    /// Optional result limit used only as fingerprint input.
    pub limit: Option<usize>,
    /// Optional top-k candidate budget used only as fingerprint input.
    pub top_k: Option<u32>,
    /// Optional path prefix scope used only as fingerprint input.
    pub starts_with: Option<String>,
    /// Path suffix scope used only as fingerprint input.
    pub ends_with: Vec<String>,
    /// Returned node identifiers used only as output fingerprint input.
    pub node_ids: Vec<String>,
    /// Event timestamp supplied by the service boundary.
    pub timestamp: String,
}

/// Typed episode append instruction produced only from a persisted context-pack proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectContextEpisodeAppendInstruction {
    episode: ToolEpisode,
}

impl ProjectContextEpisodeAppendInstruction {
    /// Returns the redaction-safe episode payload to append.
    pub fn episode(&self) -> &ToolEpisode {
        &self.episode
    }
}

/// Typed errors emitted by the context-pack commit boundary.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ProjectContextPackCommitError {
    /// The retrieval plan requested a write but did not contain a context pack.
    #[error("project-model retrieval plan requested write without context pack")]
    MissingContextPackForWrite,
    /// A readback outcome referenced evidence that was not planned.
    #[error("unknown project-model readback outcome: {evidence_id}")]
    UnknownReadbackOutcome { evidence_id: String },
    /// A planned read request did not receive any readback outcome.
    #[error("missing required project-model readback outcome: {evidence_id}")]
    MissingRequiredReadback { evidence_id: String },
    /// A readback outcome did not match the planned path/range for its evidence id.
    #[error("project-model readback path/range mismatch: {evidence_id}")]
    ReadbackPathRangeMismatch { evidence_id: String },
    /// Two planned read requests share an evidence id and cannot be verified independently.
    #[error("duplicate planned project-model read request: {evidence_id}")]
    DuplicatePlannedReadRequest { evidence_id: String },
    /// Two readback outcomes reported the same evidence id.
    #[error("duplicate project-model readback outcome: {evidence_id}")]
    DuplicateReadbackOutcome { evidence_id: String },
    /// A context-pack evidence entry was not covered by a planned read request.
    #[error("unplanned project-model context-pack evidence: {evidence_id}")]
    UnplannedContextPackEvidence { evidence_id: String },
    /// Two context-pack evidence entries share an evidence id and cannot be verified independently.
    #[error("duplicate project-model context-pack evidence: {evidence_id}")]
    DuplicateContextPackEvidence { evidence_id: String },
}

impl ProjectContextPackCommit<ReadRequestsSelected> {
    /// Builds the selected-read-requests typestate from a retrieval plan.
    ///
    /// # Arguments
    ///
    /// * `plan` - Pure retrieval plan produced by the project-model planner.
    /// * `replay_activation` - Pending replay activation used to classify replay readback failures.
    ///
    /// # Errors
    ///
    /// Returns an error when the plan requests a write without a context pack.
    pub fn from_retrieval_plan(
        plan: &ProjectContextRetrievalPlan,
        replay_activation: ReplayActivationBoundary,
    ) -> Result<Self, ProjectContextPackCommitError> {
        if plan.write_decision == ProjectContextWriteDecision::WriteContextPackAfterReadback
            && plan.context_pack.is_none()
        {
            return Err(ProjectContextPackCommitError::MissingContextPackForWrite);
        }
        Ok(Self {
            state: ProjectContextPackCommitState::Selected(Box::new(
                ProjectContextPackCommitSelected {
                    context_pack: plan.context_pack.clone(),
                    read_requests: plan.read_requests.clone(),
                    write_decision: plan.write_decision.clone(),
                    replay_activation,
                },
            )),
            _state: PhantomData,
        })
    }

    /// Returns the validated read requests that the service must execute.
    pub fn read_requests(&self) -> &[ProjectContextReadRequest] {
        match &self.state {
            ProjectContextPackCommitState::Selected(selected) => &selected.read_requests,
            ProjectContextPackCommitState::Verified(_) => {
                unreachable!("selected typestate cannot hold verified state")
            }
        }
    }

    /// Verifies typed readback outcomes and returns the write/no-write decision.
    ///
    /// # Arguments
    ///
    /// * `outcomes` - Typed readback outcomes produced by the service IO boundary.
    ///
    /// # Errors
    ///
    /// Returns an error for missing, duplicate, unknown, or path/range-mismatched outcomes.
    pub fn verify_readbacks(
        self,
        outcomes: Vec<ProjectContextReadbackOutcome>,
    ) -> Result<ProjectContextPackReadbackDecision, ProjectContextPackCommitError> {
        let selected = match self.state {
            ProjectContextPackCommitState::Selected(selected) => selected,
            ProjectContextPackCommitState::Verified(_) => {
                unreachable!("selected typestate cannot hold verified state")
            }
        };
        if selected.write_decision == ProjectContextWriteDecision::NoWriteEmptyRetrieval {
            return Ok(ProjectContextPackReadbackDecision::NoWrite(
                ProjectContextPackNoWrite {
                    reason: ProjectContextPackNoWriteReason::EmptyEvidence,
                },
            ));
        }
        let Some(context_pack) = selected.context_pack else {
            return Err(ProjectContextPackCommitError::MissingContextPackForWrite);
        };
        if context_pack.evidence.is_empty() {
            return Ok(ProjectContextPackReadbackDecision::NoWrite(
                ProjectContextPackNoWrite {
                    reason: ProjectContextPackNoWriteReason::EmptyEvidence,
                },
            ));
        }

        let mut planned_by_id = BTreeMap::new();
        for request in &selected.read_requests {
            if planned_by_id
                .insert(request.evidence_id.clone(), request)
                .is_some()
            {
                return Err(ProjectContextPackCommitError::DuplicatePlannedReadRequest {
                    evidence_id: request.evidence_id.clone(),
                });
            }
        }
        let mut context_pack_evidence_ids = BTreeSet::new();
        for evidence in &context_pack.evidence {
            if !context_pack_evidence_ids.insert(evidence.id.clone()) {
                return Err(
                    ProjectContextPackCommitError::DuplicateContextPackEvidence {
                        evidence_id: evidence.id.clone(),
                    },
                );
            }
            if !planned_by_id.contains_key(&evidence.id) {
                return Err(
                    ProjectContextPackCommitError::UnplannedContextPackEvidence {
                        evidence_id: evidence.id.clone(),
                    },
                );
            }
        }
        let mut outcome_by_id = BTreeMap::new();
        for outcome in outcomes {
            let evidence_id = outcome.evidence_id.clone();
            let Some(planned) = planned_by_id.get(&evidence_id) else {
                return Err(ProjectContextPackCommitError::UnknownReadbackOutcome { evidence_id });
            };
            if outcome_by_id
                .insert(evidence_id.clone(), outcome.clone())
                .is_some()
            {
                return Err(ProjectContextPackCommitError::DuplicateReadbackOutcome {
                    evidence_id,
                });
            }
            if planned.relative_manifest_path() != outcome.relative_manifest_path
                || planned.start_line != outcome.start_line
                || planned.end_line != outcome.end_line
            {
                return Err(ProjectContextPackCommitError::ReadbackPathRangeMismatch {
                    evidence_id,
                });
            }
        }
        for evidence_id in &context_pack_evidence_ids {
            if !outcome_by_id.contains_key(evidence_id) {
                return Err(ProjectContextPackCommitError::MissingRequiredReadback {
                    evidence_id: evidence_id.clone(),
                });
            }
        }
        for request in &selected.read_requests {
            if !outcome_by_id.contains_key(&request.evidence_id) {
                return Err(ProjectContextPackCommitError::MissingRequiredReadback {
                    evidence_id: request.evidence_id.clone(),
                });
            }
        }

        let replay_ids = selected
            .replay_activation
            .active_refs
            .iter()
            .map(|reference| reference.canonical_target_id.clone())
            .filter(|evidence_id| planned_by_id.contains_key(evidence_id))
            .collect::<BTreeSet<_>>();
        let readback_results = outcome_by_id
            .iter()
            .filter(|(evidence_id, _)| replay_ids.contains(*evidence_id))
            .map(|(evidence_id, outcome)| {
                (
                    evidence_id.clone(),
                    outcome.status == ProjectContextReadbackStatus::Succeeded,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let verified_replay_activation =
            apply_replay_readback_results(&selected.replay_activation, &readback_results);
        let verified_replay_ids = verified_replay_activation
            .active_refs
            .iter()
            .filter(|reference| reference.readback_status == ReplayEvidenceReadbackStatus::Verified)
            .map(|reference| reference.canonical_target_id.clone())
            .collect::<BTreeSet<_>>();
        let failed_non_replay = outcome_by_id.iter().any(|(evidence_id, outcome)| {
            outcome.status == ProjectContextReadbackStatus::Failed
                && !replay_ids.contains(evidence_id)
        });
        if failed_non_replay {
            return Ok(ProjectContextPackReadbackDecision::NoWrite(
                ProjectContextPackNoWrite {
                    reason: ProjectContextPackNoWriteReason::RequiredReadbackFailed,
                },
            ));
        }

        let failed_replay_ids = replay_ids
            .difference(&verified_replay_ids)
            .cloned()
            .collect::<BTreeSet<_>>();
        let has_stale_context_pack_evidence = context_pack
            .evidence
            .iter()
            .any(|evidence| is_stale_context_pack_freshness(&evidence.freshness));
        if has_stale_context_pack_evidence {
            return Ok(ProjectContextPackReadbackDecision::NoWrite(
                ProjectContextPackNoWrite {
                    reason: ProjectContextPackNoWriteReason::RequiredReadbackFailed,
                },
            ));
        }
        let filtered_context_pack = filter_context_pack_to_verified_evidence(
            context_pack,
            &context_pack_evidence_ids,
            &failed_replay_ids,
        );
        if filtered_context_pack.evidence.is_empty() {
            return Ok(ProjectContextPackReadbackDecision::NoWrite(
                ProjectContextPackNoWrite {
                    reason: ProjectContextPackNoWriteReason::NoVerifiedEvidenceAfterReplayFiltering,
                },
            ));
        }
        Ok(ProjectContextPackReadbackDecision::Write(Box::new(
            ProjectContextPackCommit {
                state: ProjectContextPackCommitState::Verified(ProjectContextPackCommitVerified {
                    context_pack: filtered_context_pack,
                }),
                _state: PhantomData,
            },
        )))
    }
}

impl ProjectContextPackCommit<ReadbackVerified> {
    /// Builds a context-pack write instruction from readback-verified state.
    pub fn write_instruction(&self) -> ProjectContextPackWriteInstruction {
        let verified = match &self.state {
            ProjectContextPackCommitState::Verified(verified) => verified,
            ProjectContextPackCommitState::Selected(_) => {
                unreachable!("verified typestate cannot hold selected state")
            }
        };
        ProjectContextPackWriteInstruction { context_pack: verified.context_pack.clone() }
    }

    /// Builds a persisted proof after storage has written and read back this verified pack.
    ///
    /// # Arguments
    ///
    /// * `artifact_id` - Deterministic persisted context-pack artifact id returned by storage.
    /// * `artifact_path` - Redaction-safe persisted artifact path label returned by storage.
    pub(crate) fn persisted_proof(
        &self,
        artifact_id: ContextPackArtifactId,
        artifact_path: impl Into<String>,
    ) -> ProjectContextPackPersistedProof {
        let verified = match &self.state {
            ProjectContextPackCommitState::Verified(verified) => verified,
            ProjectContextPackCommitState::Selected(_) => {
                unreachable!("verified typestate cannot hold selected state")
            }
        };
        ProjectContextPackPersistedProof::from_persisted_commit(
            artifact_id,
            artifact_path,
            &verified.context_pack,
        )
    }
}

impl ProjectContextPackPersistedProof {
    /// Builds a redaction-safe episode append instruction from persisted context-pack proof.
    ///
    /// # Arguments
    ///
    /// * `input` - Redaction-safe episode fingerprint inputs.
    pub fn project_model_search_episode_instruction(
        &self,
        mut input: ProjectModelSearchEpisodeInput,
    ) -> ProjectContextEpisodeAppendInstruction {
        input.node_ids.sort();
        let input_fingerprint = fingerprint(&format!(
            "query={};use_case={};limit={:?};top_k={:?};starts_with={:?};ends_with={:?}",
            input.query,
            input.use_case,
            input.limit,
            input.top_k,
            input.starts_with,
            input.ends_with
        ));
        let output_seed = format!(
            "artifact={};manifest={};nodes={}",
            self.artifact_id.as_str(),
            self.manifest_hash,
            input.node_ids.join("\0")
        );
        let output_fingerprint = fingerprint(&output_seed);
        ProjectContextEpisodeAppendInstruction {
            episode: ToolEpisode {
                timestamp: input.timestamp,
                tool: PROJECT_MODEL_SEARCH_TOOL.to_string(),
                input_fingerprint,
                output_fingerprint,
                status: PROJECT_MODEL_SEARCH_SUCCESS.to_string(),
                provenance: Provenance {
                    path: format!("context_packs/{}.json", self.artifact_id.as_str()),
                    start_line: None,
                    end_line: None,
                    source: PROJECT_MODEL_SEARCH_PROVENANCE_SOURCE.to_string(),
                    fingerprint: fingerprint(&output_seed),
                },
            },
        }
    }
}

fn is_stale_context_pack_freshness(freshness: &crate::types::EvidenceFreshness) -> bool {
    matches!(
        freshness,
        crate::types::EvidenceFreshness::Changed | crate::types::EvidenceFreshness::Deleted
    )
}

fn filter_context_pack_to_verified_evidence(
    mut context_pack: ContextPack,
    verified_evidence_ids: &BTreeSet<String>,
    failed_replay_ids: &BTreeSet<String>,
) -> ContextPack {
    context_pack.evidence.retain(|evidence| {
        verified_evidence_ids.contains(&evidence.id) && !failed_replay_ids.contains(&evidence.id)
    });
    context_pack.provenance = context_pack
        .evidence
        .iter()
        .map(|evidence| evidence.provenance.clone())
        .collect::<Vec<_>>();
    context_pack
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use pretty_assertions::assert_eq;

    use super::*;
    use crate::types::{ContextPackEvidence, ContextPackEvidenceSource, EvidenceFreshness};
    use crate::{
        EvidenceReplayScoreKind, ProjectContextRetrievalPhaseDiagnostics,
        ProjectContextRetrievalQueryDiagnostics, ProjectContextTopKStatus,
        ReplayActivatedEvidenceRef, ReplayActivationCaps, ReplayActivationDiagnostics,
        ReplayActivationFingerprintInputs, ReplayEvidenceTargetKind,
    };

    #[test]
    fn empty_evidence_returns_no_write_state() {
        let setup = plan(
            None,
            Vec::new(),
            ProjectContextWriteDecision::NoWriteEmptyRetrieval,
        );
        let actual =
            ProjectContextPackCommit::from_retrieval_plan(&setup, replay_boundary(Vec::new()))
                .unwrap()
                .verify_readbacks(Vec::new())
                .unwrap();
        let expected = ProjectContextPackNoWriteReason::EmptyEvidence;

        assert_eq!(no_write_reason(actual), expected);
    }

    #[test]
    fn missing_required_readback_is_rejected_distinctly() {
        let setup = plan(
            Some(pack(vec![evidence("main", "src/main.rs")])),
            vec![request("main", "src/main.rs")],
            ProjectContextWriteDecision::WriteContextPackAfterReadback,
        );
        let actual =
            ProjectContextPackCommit::from_retrieval_plan(&setup, replay_boundary(Vec::new()))
                .unwrap()
                .verify_readbacks(Vec::new())
                .unwrap_err();
        let expected = ProjectContextPackCommitError::MissingRequiredReadback {
            evidence_id: "main".to_string(),
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn unknown_readback_outcome_is_rejected() {
        let setup = plan(
            Some(pack(vec![evidence("main", "src/main.rs")])),
            vec![request("main", "src/main.rs")],
            ProjectContextWriteDecision::WriteContextPackAfterReadback,
        );
        let actual =
            ProjectContextPackCommit::from_retrieval_plan(&setup, replay_boundary(Vec::new()))
                .unwrap()
                .verify_readbacks(vec![
                    ProjectContextReadbackOutcome::new(
                        "other",
                        "src/lib.rs",
                        1,
                        3,
                        ProjectContextReadbackStatus::Succeeded,
                    )
                    .unwrap(),
                ])
                .unwrap_err();
        let expected = ProjectContextPackCommitError::UnknownReadbackOutcome {
            evidence_id: "other".to_string(),
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn readback_path_range_must_match_planned_request() {
        let setup = plan(
            Some(pack(vec![evidence("main", "src/main.rs")])),
            vec![request("main", "src/main.rs")],
            ProjectContextWriteDecision::WriteContextPackAfterReadback,
        );
        let actual =
            ProjectContextPackCommit::from_retrieval_plan(&setup, replay_boundary(Vec::new()))
                .unwrap()
                .verify_readbacks(vec![
                    ProjectContextReadbackOutcome::new(
                        "main",
                        "src/main.rs",
                        2,
                        3,
                        ProjectContextReadbackStatus::Succeeded,
                    )
                    .unwrap(),
                ])
                .unwrap_err();
        let expected = ProjectContextPackCommitError::ReadbackPathRangeMismatch {
            evidence_id: "main".to_string(),
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn non_replay_readback_failure_yields_no_write_or_episode_instruction() {
        let setup = plan(
            Some(pack(vec![evidence("main", "src/main.rs")])),
            vec![request("main", "src/main.rs")],
            ProjectContextWriteDecision::WriteContextPackAfterReadback,
        );
        let actual =
            ProjectContextPackCommit::from_retrieval_plan(&setup, replay_boundary(Vec::new()))
                .unwrap()
                .verify_readbacks(vec![ProjectContextReadbackOutcome::failed(
                    &setup.read_requests[0],
                )])
                .unwrap();
        let expected = ProjectContextPackNoWriteReason::RequiredReadbackFailed;

        assert_eq!(no_write_reason(actual), expected);
    }

    #[test]
    fn replay_readback_failure_filters_only_failed_replay_evidence() {
        let setup = plan(
            Some(pack(vec![
                evidence("stable", "src/stable.rs"),
                evidence("replay_failed", "src/replay_failed.rs"),
                evidence("replay_ok", "src/replay_ok.rs"),
            ])),
            vec![
                request("stable", "src/stable.rs"),
                request("replay_failed", "src/replay_failed.rs"),
                request("replay_ok", "src/replay_ok.rs"),
            ],
            ProjectContextWriteDecision::WriteContextPackAfterReadback,
        );
        let actual = ProjectContextPackCommit::from_retrieval_plan(
            &setup,
            replay_boundary(vec!["replay_failed", "replay_ok"]),
        )
        .unwrap()
        .verify_readbacks(vec![
            ProjectContextReadbackOutcome::succeeded(&setup.read_requests[0]),
            ProjectContextReadbackOutcome::failed(&setup.read_requests[1]),
            ProjectContextReadbackOutcome::succeeded(&setup.read_requests[2]),
        ])
        .unwrap();
        let expected = vec!["replay_ok".to_string(), "stable".to_string()];

        let ProjectContextPackReadbackDecision::Write(verified) = actual else {
            panic!("expected write decision")
        };
        let write = verified.write_instruction();
        let mut actual = write
            .context_pack()
            .evidence
            .iter()
            .map(|evidence| evidence.id.clone())
            .collect::<Vec<_>>();
        actual.sort();
        assert_eq!(actual, expected);
    }

    #[test]
    fn duplicate_planned_evidence_id_cannot_hide_unverified_read_request() {
        let setup = plan(
            Some(pack(vec![evidence("main", "src/second.rs")])),
            vec![
                request("main", "src/first.rs"),
                request("main", "src/second.rs"),
            ],
            ProjectContextWriteDecision::WriteContextPackAfterReadback,
        );
        let actual =
            ProjectContextPackCommit::from_retrieval_plan(&setup, replay_boundary(Vec::new()))
                .unwrap()
                .verify_readbacks(vec![ProjectContextReadbackOutcome::succeeded(
                    &setup.read_requests[1],
                )])
                .unwrap_err();
        let expected = ProjectContextPackCommitError::DuplicatePlannedReadRequest {
            evidence_id: "main".to_string(),
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn context_pack_evidence_without_planned_read_request_is_rejected() {
        let setup = plan(
            Some(pack(vec![
                evidence("verified", "src/verified.rs"),
                evidence("unverified", "src/unverified.rs"),
            ])),
            vec![request("verified", "src/verified.rs")],
            ProjectContextWriteDecision::WriteContextPackAfterReadback,
        );
        let actual =
            ProjectContextPackCommit::from_retrieval_plan(&setup, replay_boundary(Vec::new()))
                .unwrap()
                .verify_readbacks(vec![ProjectContextReadbackOutcome::succeeded(
                    &setup.read_requests[0],
                )])
                .unwrap_err();
        let expected = ProjectContextPackCommitError::UnplannedContextPackEvidence {
            evidence_id: "unverified".to_string(),
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn context_pack_stale_evidence_is_rejected_at_commit_boundary() {
        let mut stale = evidence("stale", "src/stale.rs");
        stale.freshness = EvidenceFreshness::Changed;
        let setup = plan(
            Some(pack(vec![stale])),
            vec![request("stale", "src/stale.rs")],
            ProjectContextWriteDecision::WriteContextPackAfterReadback,
        );
        let actual =
            ProjectContextPackCommit::from_retrieval_plan(&setup, replay_boundary(Vec::new()))
                .unwrap()
                .verify_readbacks(vec![ProjectContextReadbackOutcome::succeeded(
                    &setup.read_requests[0],
                )])
                .unwrap();
        let ProjectContextPackReadbackDecision::NoWrite(actual) = actual else {
            panic!("stale context-pack evidence must not produce a write decision")
        };
        let expected = ProjectContextPackNoWriteReason::RequiredReadbackFailed;

        assert_eq!(actual.reason(), &expected);
    }

    #[test]
    fn successful_persisted_context_pack_proof_required_before_episode_payload_construction() {
        let setup_pack = pack(vec![evidence("main", "src/main.rs")]);
        let proof = verified_commit(&setup_pack).persisted_proof(
            crate::ContextPackArtifactId::new("a".repeat(64)).unwrap(),
            "context_packs/proof.json",
        );
        let actual =
            proof.project_model_search_episode_instruction(ProjectModelSearchEpisodeInput {
                query: "needle".to_string(),
                use_case: "proof".to_string(),
                limit: Some(1),
                top_k: Some(1),
                starts_with: Some("src".to_string()),
                ends_with: vec![".rs".to_string()],
                node_ids: vec!["main".to_string()],
                timestamp: "2026-01-01T00:00:00Z".to_string(),
            });
        let expected = "project_model_search";

        assert_eq!(actual.episode().tool, expected);
        assert_eq!(
            actual.episode().provenance.path,
            format!("context_packs/{}.json", proof.artifact_id().as_str())
        );
    }

    #[test]
    fn committed_no_write_result_cannot_contain_proof_or_episode_success() {
        let setup =
            ProjectContextReadbackSummary::from_outcomes(&[ProjectContextReadbackOutcome::new(
                "main",
                "src/main.rs",
                1,
                3,
                ProjectContextReadbackStatus::Failed,
            )
            .unwrap()]);
        let actual = ProjectContextCommittedQueryResult::no_write(
            setup,
            ProjectContextPackNoWriteReason::RequiredReadbackFailed,
            vec![ProjectContextCommittedResultItem::new("main", Some(1.0))],
        );
        let expected = (
            ProjectContextCommitOutcome::NoWrite(
                ProjectContextPackNoWriteReason::RequiredReadbackFailed,
            ),
            ProjectContextEpisodeAppendOutcome::NotAttempted {
                reason: ProjectContextEpisodeAppendNotAttemptedReason::NoPersistedContextPack,
            },
            1usize,
        );

        assert_eq!(
            (
                actual.commit().clone(),
                actual.episode_append().clone(),
                actual.readback().failed_count(),
            ),
            expected,
        );
    }

    #[test]
    fn committed_persisted_result_requires_persisted_proof_typestate() {
        let setup_pack = pack(vec![evidence("main", "src/main.rs")]);
        let setup_proof = verified_commit(&setup_pack).persisted_proof(
            crate::ContextPackArtifactId::new("b".repeat(64)).unwrap(),
            "context_packs/proof.json",
        );
        let actual = ProjectContextCommittedQueryResult::persisted(
            ProjectContextReadbackSummary::from_outcomes(&[
                ProjectContextReadbackOutcome::succeeded(&request("main", "src/main.rs")),
            ]),
            setup_proof.clone(),
            ProjectContextPersistedEpisodeAppendOutcome::appended("episode-fingerprint"),
            vec![ProjectContextCommittedResultItem::new("main", Some(1.0))],
        );
        let expected = ProjectContextCommitOutcome::Persisted(setup_proof);

        assert_eq!(actual.commit(), &expected);
    }
    #[test]
    fn committed_persisted_result_requires_persisted_episode_append_outcome() {
        let setup_pack = pack(vec![evidence("main", "src/main.rs")]);
        let setup_proof = verified_commit(&setup_pack).persisted_proof(
            crate::ContextPackArtifactId::new("c".repeat(64)).unwrap(),
            "context_packs/proof.json",
        );
        let actual = ProjectContextCommittedQueryResult::persisted(
            ProjectContextReadbackSummary::from_outcomes(&[
                ProjectContextReadbackOutcome::succeeded(&request("main", "src/main.rs")),
            ]),
            setup_proof,
            ProjectContextPersistedEpisodeAppendOutcome::failed(
                ProjectContextEpisodeAppendFailureReason::EpisodeAppendFailed,
            ),
            vec![ProjectContextCommittedResultItem::new("main", Some(1.0))],
        );
        let expected = ProjectContextEpisodeAppendOutcome::Failed {
            reason_code: ProjectContextEpisodeAppendFailureReason::EpisodeAppendFailed,
        };

        assert_eq!(actual.episode_append(), &expected);
    }

    fn no_write_reason(
        decision: ProjectContextPackReadbackDecision,
    ) -> ProjectContextPackNoWriteReason {
        match decision {
            ProjectContextPackReadbackDecision::NoWrite(no_write) => no_write.reason().clone(),
            ProjectContextPackReadbackDecision::Write(_) => panic!("expected no-write decision"),
        }
    }

    fn plan(
        context_pack: Option<ContextPack>,
        read_requests: Vec<ProjectContextReadRequest>,
        write_decision: ProjectContextWriteDecision,
    ) -> ProjectContextRetrievalPlan {
        ProjectContextRetrievalPlan {
            query_diagnostics: query_diagnostics(),
            selected_results: Vec::new(),
            context_pack,
            read_requests,
            write_decision,
            return_order: Vec::new(),
        }
    }

    fn query_diagnostics() -> ProjectContextRetrievalQueryDiagnostics {
        ProjectContextRetrievalQueryDiagnostics {
            query_text: Some("test".to_string()),
            path_prefix: None,
            path_suffixes: Vec::new(),
            limit: 10,
            top_k: None,
            top_k_status: ProjectContextTopKStatus::NotRequested,
            use_case: None,
            rerank_intent_source: None,
            rerank_intent_fingerprint: None,
            rerank_intent_len: None,
            include_graph_expansion: false,
            stale_policy: crate::StaleEvidencePolicy::Reject,
            freshness_proof_level: crate::FreshnessProofLevel::FullFilesystem,
            phase_diagnostics: ProjectContextRetrievalPhaseDiagnostics::default(),
        }
    }

    fn request(evidence_id: &str, path: &str) -> ProjectContextReadRequest {
        ProjectContextReadRequest::new(path, evidence_id, 1, 3).unwrap()
    }

    fn pack(evidence: Vec<ContextPackEvidence>) -> ContextPack {
        ContextPack {
            version: 1,
            manifest_hash: "manifest".to_string(),
            provenance: evidence
                .iter()
                .map(|evidence| evidence.provenance.clone())
                .collect(),
            evidence,
        }
    }

    fn evidence(id: &str, path: &str) -> ContextPackEvidence {
        ContextPackEvidence {
            id: id.to_string(),
            path: path.to_string(),
            symbol: None,
            source: ContextPackEvidenceSource::RetrievalResult,
            freshness: EvidenceFreshness::Fresh,
            provenance: Provenance {
                path: path.to_string(),
                start_line: Some(1),
                end_line: Some(3),
                source: "test".to_string(),
                fingerprint: fingerprint(&format!("{path}:1-3")),
            },
            score: 1.0,
        }
    }

    fn verified_commit(context_pack: &ContextPack) -> ProjectContextPackCommit<ReadbackVerified> {
        let setup = plan(
            Some(context_pack.clone()),
            context_pack
                .evidence
                .iter()
                .map(|evidence| request(&evidence.id, &evidence.path))
                .collect(),
            ProjectContextWriteDecision::WriteContextPackAfterReadback,
        );
        match ProjectContextPackCommit::from_retrieval_plan(&setup, replay_boundary(Vec::new()))
            .unwrap()
            .verify_readbacks(
                setup
                    .read_requests
                    .iter()
                    .map(ProjectContextReadbackOutcome::succeeded)
                    .collect(),
            )
            .unwrap()
        {
            ProjectContextPackReadbackDecision::Write(commit) => *commit,
            ProjectContextPackReadbackDecision::NoWrite(_) => panic!("expected write decision"),
        }
    }

    fn replay_boundary(ids: Vec<&str>) -> ReplayActivationBoundary {
        ReplayActivationBoundary {
            manifest_hash: "manifest".to_string(),
            active_refs: ids
                .into_iter()
                .map(|id| ReplayActivatedEvidenceRef {
                    artifact_id: "artifact".to_string(),
                    evidence_id: id.to_string(),
                    evidence_path: format!("src/{id}.rs"),
                    start_line: 1,
                    end_line: 3,
                    score_kind: EvidenceReplayScoreKind::RetrievalResult,
                    score: 1.0,
                    provenance_source: "test".to_string(),
                    target_kind: ReplayEvidenceTargetKind::ManifestSource,
                    canonical_target_id: id.to_string(),
                    fingerprint_inputs: ReplayActivationFingerprintInputs {
                        manifest_hash: "manifest".to_string(),
                        source_content_hash: fingerprint(id),
                        line_range_fingerprint: fingerprint(&format!("{id}:1-3")),
                        target_kind: ReplayEvidenceTargetKind::ManifestSource,
                        target_id: id.to_string(),
                    },
                    readback_status: ReplayEvidenceReadbackStatus::Pending,
                })
                .collect(),
            issues: Vec::new(),
            diagnostics: ReplayActivationDiagnostics {
                candidate_count: 0,
                active_count: 0,
                excluded_count: 0,
                excluded_by_reason: BTreeMap::new(),
                caps: ReplayActivationCaps::default(),
                stable_ordering: "test".to_string(),
                activation_fingerprint: fingerprint("activation"),
            },
        }
    }
}
