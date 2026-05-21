use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use chrono::Local;
use forge_config::ForgeConfig;
use forge_domain::*;
use forge_project_model::{
    AcceptedLearningSummary as ProjectAcceptedLearningSummary, LearningContextPayload,
    LearningContextRecord, LearningLedgerFreshness as ProjectLearningLedgerFreshness,
    LearningProvenance as ProjectLearningProvenance,
    LearningRedactionStatus as ProjectLearningRedactionStatus,
    LearningReviewState as ProjectLearningReviewState,
    LearningSourceKind as ProjectLearningSourceKind, ProjectContextPathScope,
    ProjectContextRetrievalPhaseDiagnostics, ProjectContextRetrievalPhaseInvalidReason,
    ProjectContextRetrievalPhaseSkipReason, ProjectContextRetrievalPhaseStatus,
    ProjectContextRetrievalPhaseUnavailableReason, ProjectContextRetrievalPlanDiagnostic,
    ProjectContextRetrievalReadRequestSummary, ProjectContextRetrievalRequest,
    ProjectContextRetrievalSelectedSummary, ProjectContextTarget, ProjectContextWriteDecision,
    ProjectIndexer, ProjectModelContextEnvelopeInput, ProjectModelContextReadinessMetadata,
    ProjectModelContextRenderBudget, ProjectModelContextRenderRoot,
    ProjectModelEvidenceLedgerActivationMetadata, ProjectModelEvidenceReadinessMetadata,
    ProjectModelExactFactReadinessMetadata, ProjectModelManifestFreshnessProof,
    ProjectModelSourceNode, ProjectModelVolatileSidecarInput, TargetResolutionBudget,
    build_project_model_context_envelope, directory_path_filter, local_project_model_dir,
    local_project_model_manifest, mentioned_paths, plan_project_context_retrieval,
};
use forge_stream::MpscStream;
use url::Url;

use crate::apply_tunable_parameters::ApplyTunableParameters;
use crate::changed_files::ChangedFiles;
use crate::dto::ToolsOverview;
use crate::dto::openai::ProviderRequestEstimate as OpenAiProviderRequestEstimate;
use crate::hooks::{
    CompactionHandler, DoomLoopDetector, LearningCapture, PendingTodosHandler,
    TitleGenerationHandler, TracingHandler,
};
use crate::init_conversation_metrics::InitConversationMetrics;
use crate::orch::Orchestrator;
use crate::services::{
    AgentRegistry, CustomInstructionsService, ProviderAuthService, SteerService,
};
use crate::set_conversation_id::SetConversationId;
use crate::steer::SteerHandle;
use crate::system_prompt::SystemPrompt;
use crate::tool_registry::ToolRegistry;
use crate::tool_resolver::ToolResolver;
use crate::user_prompt::UserPromptGenerator;
use crate::{
    AgentExt, AgentProviderResolver, ConversationService, EnvironmentInfra, LearningService,
    ProviderService, Services, WorkspaceService,
};

/// Builds a [`TemplateConfig`] from a [`ForgeConfig`].
///
/// Converts the configuration-layer field names into the domain-layer struct
/// expected by [`SystemContext`] for tool description template rendering.
pub(crate) fn build_template_config(config: &ForgeConfig) -> forge_domain::TemplateConfig {
    forge_domain::TemplateConfig {
        max_read_size: config.max_read_lines.try_into().unwrap_or(usize::MAX),
        max_line_length: config.max_line_chars,
        max_image_size: config.max_image_size_bytes.try_into().unwrap_or(usize::MAX),
        stdout_max_prefix_length: config.max_stdout_prefix_lines,
        stdout_max_suffix_length: config.max_stdout_suffix_lines,
        stdout_max_line_length: config.max_stdout_line_chars,
    }
}

const AUTOMATIC_CONTEXT_QUERY_EMBEDDING_TEXT_LIMIT: usize = 4096;

fn bounded_semantic_embedding_query(query: &str) -> String {
    query
        .chars()
        .take(AUTOMATIC_CONTEXT_QUERY_EMBEDDING_TEXT_LIMIT)
        .collect()
}

struct ProjectContextTargetDiagnostic {
    target: ProjectContextTarget,
    diagnostic: WorkspaceContextManifestDiagnostic,
}

struct ProjectContextRenderedPartition {
    stable_payload: String,
    volatile_sidecar: String,
    stable_identity: String,
}

struct ProjectContextInjection<S> {
    services: Arc<S>,
    agent: Agent,
}

impl<S: EnvironmentInfra<Config = forge_config::ForgeConfig> + WorkspaceService>
    ProjectContextInjection<S>
{
    const MAX_TARGETS: usize = 4;
    const MAX_EXPLICIT_TARGET_CANDIDATES: usize = 8;
    const MAX_INDEX_PROBES: usize = 32;
    const MAX_LEARNING_RECORDS: usize = 8;
    const MAX_LEARNING_CONTEXT_CHARS: usize = 8_192;
    const SEMANTIC_EMBEDDING_TIMEOUT: Duration = Duration::from_secs(5);

    fn new(services: Arc<S>, agent: Agent) -> Self {
        Self { services, agent }
    }

    async fn inject_learning(&self, mut conversation: Conversation) -> Conversation
    where
        S: LearningService,
    {
        let records = match self
            .services
            .list_learning_records(
                Some(LearningReviewState::Accepted),
                Self::MAX_LEARNING_RECORDS,
            )
            .await
        {
            Ok(records) => records,
            Err(error) => {
                tracing::debug!(error = ?error, "Skipping learning context injection because reviewed query failed");
                return conversation;
            }
        };
        if records.is_empty() {
            return conversation;
        }
        let freshness = match self
            .services
            .learning_freshness(Some(LearningReviewState::Accepted))
            .await
        {
            Ok(freshness) => freshness,
            Err(error) => {
                tracing::debug!(error = ?error, "Skipping learning context injection because freshness query failed");
                return conversation;
            }
        };
        let payload = LearningContextPayload::new(
            Self::learning_freshness_to_project(freshness),
            records
                .into_iter()
                .filter_map(Self::learning_record_to_project)
                .collect(),
        );
        if payload.records.is_empty() {
            return conversation;
        }
        let content = match payload.render() {
            Ok(content) => content,
            Err(error) => {
                tracing::debug!(error = ?error, "Skipping learning context injection because payload violated reviewed-only transport invariants");
                return conversation;
            }
        };
        if content.chars().count() > Self::MAX_LEARNING_CONTEXT_CHARS {
            tracing::debug!(
                actual_chars = content.chars().count(),
                max_chars = Self::MAX_LEARNING_CONTEXT_CHARS,
                "Skipping learning context injection because rendered payload exceeds bounded budget"
            );
            return conversation;
        }
        let mut context = conversation.context.take().unwrap_or_default();
        let message = TextMessage::learning_context(Role::User, content)
            .model(self.agent.model.clone())
            .droppable(true)
            .cacheable(false);
        context = context.add_message(ContextMessage::Text(message));
        conversation.context(context)
    }

    fn learning_freshness_to_project(
        freshness: LearningLedgerFreshness,
    ) -> ProjectLearningLedgerFreshness {
        ProjectLearningLedgerFreshness {
            ledger_cursor: freshness.ledger_cursor,
            projection_version: freshness.projection_version,
            review_state_fingerprint: freshness.review_state_fingerprint,
        }
    }

    fn is_deterministic_conversation_save_accepted(projection: &LearningRecordProjection) -> bool {
        projection.provenance.source_kind == LearningSourceKind::Conversation
            && projection.provenance.conversation_id.is_some()
            && projection
                .capture_metadata
                .as_ref()
                .is_some_and(|metadata| {
                    metadata.validate_current().is_ok()
                        && projection.summary.starts_with("conversation_saved ")
                })
            && projection.review_state == LearningReviewState::Accepted
    }

    fn learning_record_to_project(
        projection: LearningRecordProjection,
    ) -> Option<LearningContextRecord> {
        if projection.review_state != LearningReviewState::Accepted {
            return None;
        }
        let summary = match projection.accepted_summary {
            Some(accepted_summary) => ProjectAcceptedLearningSummary::new(accepted_summary)
                .ok()?
                .as_str()
                .to_string(),
            None if Self::is_deterministic_conversation_save_accepted(&projection) => {
                projection.summary
            }
            None => return None,
        };
        Some(LearningContextRecord {
            id: projection.record_id.into_string(),
            summary,
            review_state: ProjectLearningReviewState::Accepted,
            redaction_status: Self::learning_redaction_to_project(projection.redaction_status),
            provenance: Self::learning_provenance_to_project(projection.provenance)?,
        })
    }

    fn learning_redaction_to_project(
        status: LearningRedactionStatus,
    ) -> ProjectLearningRedactionStatus {
        match status {
            LearningRedactionStatus::Clean => ProjectLearningRedactionStatus::Clean,
            LearningRedactionStatus::Redacted => ProjectLearningRedactionStatus::Redacted,
        }
    }

    fn learning_provenance_to_project(
        provenance: LearningProvenance,
    ) -> Option<ProjectLearningProvenance> {
        let source_kind = match provenance.source_kind {
            LearningSourceKind::Conversation => ProjectLearningSourceKind::Conversation,
            LearningSourceKind::Task => ProjectLearningSourceKind::Task,
            LearningSourceKind::Tool => ProjectLearningSourceKind::Tool,
            LearningSourceKind::Eval => ProjectLearningSourceKind::Eval,
        };
        let source_id = provenance.source_id().ok()?;
        Some(ProjectLearningProvenance {
            source_kind,
            source_id,
            source_event_id: Some(provenance.source_event_id),
            source_timestamp: None,
            source_fingerprint: provenance.source_fingerprint,
        })
    }

    async fn inject(&self, mut conversation: Conversation) -> Conversation {
        let environment = self.services.get_environment();
        let Some(query) = Self::query_from_conversation(&conversation) else {
            return conversation;
        };
        let targets = self.resolve_targets(&environment, &query).await;
        if targets.is_empty() {
            return conversation;
        }

        let max_sources = ProjectModelContextRenderBudget::default().max_sources;
        let config = self.services.get_config().ok();
        let embedding_model_id = config
            .as_ref()
            .and_then(|config| config.semantic_embedding_model_id.clone())
            .filter(|model_id| !model_id.trim().is_empty());
        let semantic_top_k = config
            .as_ref()
            .map(|config| config.sem_search_top_k)
            .filter(|top_k| *top_k > 0);
        let mut rendered_contexts = Vec::new();
        for target_diagnostic in targets {
            let target = target_diagnostic.target;
            let mut params = SearchParams::new(&query, "automatic project-model context injection")
                .automatic_injection()
                .limit(max_sources);
            if let Some(path_filter) = target.path_filter.clone() {
                params = params.starts_with(path_filter);
            }
            if let Some(top_k) = semantic_top_k {
                params = params.top_k(top_k as u32);
            }
            let (params, semantic_diagnostic) = self
                .semantic_params_for_target(
                    &target.workspace_root,
                    &query,
                    params,
                    embedding_model_id.clone(),
                )
                .await;
            let nodes = match self
                .services
                .query_workspace_committed(target.workspace_root.clone(), params)
                .await
            {
                Ok((_committed_result, nodes)) => nodes,
                Err(error) => {
                    tracing::debug!(error = ?error, path = %target.workspace_root.display(), "Skipping project-model context target because committed local retrieval failed");
                    continue;
                }
            };
            if nodes.is_empty() {
                continue;
            }
            if let Some(rendered_context) = Self::render_context(
                &target.workspace_root,
                &target_diagnostic.diagnostic,
                semantic_diagnostic.as_deref(),
                nodes,
            ) {
                rendered_contexts.push(rendered_context);
            }
        }
        if rendered_contexts.is_empty() {
            return conversation;
        }

        let mut context = conversation.context.take().unwrap_or_default();
        for rendered_context in rendered_contexts {
            let stable_message = TextMessage::stable_project_model_context(
                Role::User,
                rendered_context.stable_payload,
            )
            .model(self.agent.model.clone())
            .droppable(true)
            .cacheable(true);
            tracing::debug!(stable_identity = %rendered_context.stable_identity, "Injecting stable project-model context envelope");
            context = context.add_message(ContextMessage::Text(stable_message));
            let sidecar_message = TextMessage::project_model_volatile_sidecar(
                Role::User,
                rendered_context.volatile_sidecar,
            )
            .model(self.agent.model.clone())
            .droppable(true)
            .cacheable(false);
            context = context.add_message(ContextMessage::Text(sidecar_message));
        }
        conversation.context(context)
    }

    async fn semantic_params_for_target<'a>(
        &self,
        workspace_root: &Path,
        query: &'a str,
        params: SearchParams<'a>,
        embedding_model_id: Option<String>,
    ) -> (SearchParams<'a>, Option<String>) {
        let readiness = match self
            .services
            .semantic_injection_readiness(workspace_root.to_path_buf(), embedding_model_id.clone())
            .await
        {
            Ok(readiness) => readiness,
            Err(error) => {
                tracing::debug!(error = ?error, path = %workspace_root.display(), "Automatic semantic project-context readiness check failed; using lexical fallback");
                return (
                    params,
                    Some("semantic_vector_state=VectorIndexCorruptOrNotReady".to_string()),
                );
            }
        };
        match readiness {
            WorkspaceSemanticInjectionReadiness::SemanticDisabledNoModelConfig
            | WorkspaceSemanticInjectionReadiness::VectorIndexAbsentOrNoMatch => (params, None),
            WorkspaceSemanticInjectionReadiness::VectorIndexReady { dimension } => {
                let Some(embedding_model_id) = embedding_model_id else {
                    return (params, None);
                };
                let output = match tokio::time::timeout(
                    Self::SEMANTIC_EMBEDDING_TIMEOUT,
                    self.services.embed_workspace_query(
                        bounded_semantic_embedding_query(query),
                        embedding_model_id.clone(),
                    ),
                )
                .await
                {
                    Ok(Ok(output)) => output,
                    Ok(Err(error)) => {
                        tracing::debug!(error = ?error, path = %workspace_root.display(), "Automatic semantic project-context embedding failed; using lexical fallback");
                        return (
                            params,
                            Some(
                                "semantic_vector_state=EmbeddingProviderUnavailable lexical_fallback=true"
                                    .to_string(),
                            ),
                        );
                    }
                    Err(_elapsed) => {
                        tracing::debug!(path = %workspace_root.display(), timeout_ms = Self::SEMANTIC_EMBEDDING_TIMEOUT.as_millis(), "Automatic semantic project-context embedding timed out; using lexical fallback");
                        return (
                            params,
                            Some(
                                "semantic_vector_state=EmbeddingProviderTimeout lexical_fallback=true"
                                    .to_string(),
                            ),
                        );
                    }
                };
                let Some(vector) = output.vectors.into_iter().next() else {
                    return (
                        params,
                        Some(
                            "semantic_vector_state=EmbeddingProviderUnavailable lexical_fallback=true"
                                .to_string(),
                        ),
                    );
                };
                if output.embedding_model_id != embedding_model_id {
                    return (
                        params,
                        Some(
                            "semantic_vector_state=EmbeddingProviderUnavailable lexical_fallback=true"
                                .to_string(),
                        ),
                    );
                }
                if vector.embedding.len() != dimension {
                    tracing::debug!(expected = dimension, actual = vector.embedding.len(), path = %workspace_root.display(), "Automatic semantic project-context embedding dimension mismatched durable vector index; using lexical fallback");
                    return (
                        params,
                        Some(format!(
                            "semantic_vector_state=VectorDimensionMismatch expected={} actual={} lexical_fallback=true",
                            dimension,
                            vector.embedding.len()
                        )),
                    );
                }
                (
                    params
                        .query_embedding(vector.embedding)
                        .embedding_model_id(embedding_model_id),
                    None,
                )
            }
            WorkspaceSemanticInjectionReadiness::VectorIndexAmbiguous
            | WorkspaceSemanticInjectionReadiness::VectorIndexCorruptOrNotReady
            | WorkspaceSemanticInjectionReadiness::VectorDimensionMismatch { .. } => {
                let diagnostic = Self::semantic_invalid_diagnostic(readiness);
                tracing::debug!(diagnostic = %diagnostic, path = %workspace_root.display(), "Automatic semantic project-context disabled because vector state is invalid; using lexical fallback");
                (params, Some(diagnostic))
            }
            WorkspaceSemanticInjectionReadiness::EmbeddingProviderUnavailable
            | WorkspaceSemanticInjectionReadiness::EmbeddingProviderTimeout => {
                let diagnostic = Self::semantic_invalid_diagnostic(readiness);
                tracing::debug!(diagnostic = %diagnostic, path = %workspace_root.display(), "Automatic semantic project-context provider unavailable; using lexical fallback");
                (params, Some(diagnostic))
            }
        }
    }

    fn semantic_invalid_diagnostic(readiness: WorkspaceSemanticInjectionReadiness) -> String {
        match readiness {
            WorkspaceSemanticInjectionReadiness::SemanticDisabledNoModelConfig => {
                "semantic_vector_state=SemanticDisabledNoModelConfig".to_string()
            }
            WorkspaceSemanticInjectionReadiness::VectorIndexAbsentOrNoMatch => {
                "semantic_vector_state=VectorIndexAbsentOrNoMatch".to_string()
            }
            WorkspaceSemanticInjectionReadiness::VectorIndexReady { dimension } => {
                format!("semantic_vector_state=VectorIndexReady dimension={dimension}")
            }
            WorkspaceSemanticInjectionReadiness::VectorIndexAmbiguous => {
                "semantic_vector_state=VectorIndexAmbiguous lexical_fallback=true".to_string()
            }
            WorkspaceSemanticInjectionReadiness::VectorIndexCorruptOrNotReady => {
                "semantic_vector_state=VectorIndexCorruptOrNotReady lexical_fallback=true"
                    .to_string()
            }
            WorkspaceSemanticInjectionReadiness::VectorDimensionMismatch { expected, actual } => {
                format!(
                    "semantic_vector_state=VectorDimensionMismatch expected={expected} actual={actual} lexical_fallback=true"
                )
            }
            WorkspaceSemanticInjectionReadiness::EmbeddingProviderUnavailable => {
                "semantic_vector_state=EmbeddingProviderUnavailable lexical_fallback=true"
                    .to_string()
            }
            WorkspaceSemanticInjectionReadiness::EmbeddingProviderTimeout => {
                "semantic_vector_state=EmbeddingProviderTimeout lexical_fallback=true".to_string()
            }
        }
    }

    async fn explain(&self, query: Option<String>) -> WorkspaceContextExplanation {
        let environment = self.services.get_environment();
        let mut candidates = vec![environment.cwd.clone()];
        if let Some(query) = query.as_deref() {
            candidates.extend(mentioned_paths(
                query,
                &environment.cwd,
                environment.home.as_deref(),
            ));
        }

        let mut budget = TargetResolutionBudget::new(
            Self::MAX_EXPLICIT_TARGET_CANDIDATES,
            Self::MAX_INDEX_PROBES,
        );
        let mut candidate_diagnostics = Vec::new();
        let mut selected_targets = Vec::new();
        let mut nearest_skipped_manifest_candidates = Vec::new();
        let mut target_specs = Vec::new();
        let mut seen = BTreeSet::new();
        for candidate in candidates {
            if !budget.claim_candidate() {
                candidate_diagnostics.push(WorkspaceContextCandidateDiagnostic {
                    candidate_path: candidate,
                    selected_workspace: None,
                    path_filter: None,
                    skip_reason: Some("candidate limit reached".to_string()),
                });
                break;
            }
            let (candidate_diagnostic, manifest_diagnostic, skipped_manifest_diagnostic, target) =
                self.resolve_target_diagnostic(candidate, &mut budget).await;
            if let (Some(manifest_diagnostic), Some(target)) = (manifest_diagnostic, target) {
                if seen.insert(target.clone()) {
                    selected_targets.push(manifest_diagnostic.clone());
                    target_specs.push(ProjectContextTargetDiagnostic {
                        target,
                        diagnostic: manifest_diagnostic,
                    });
                }
            } else if let Some(skipped_manifest_diagnostic) = skipped_manifest_diagnostic {
                nearest_skipped_manifest_candidates.push(skipped_manifest_diagnostic);
            }
            candidate_diagnostics.push(candidate_diagnostic);
            if target_specs.len() >= Self::MAX_TARGETS {
                break;
            }
        }

        let mut replay_preview_targets = target_specs
            .iter()
            .map(|target_diagnostic| target_diagnostic.diagnostic.clone())
            .chain(nearest_skipped_manifest_candidates.iter().cloned())
            .collect::<Vec<_>>();
        if replay_preview_targets.is_empty() {
            replay_preview_targets.push(WorkspaceContextManifestDiagnostic {
                workspace_root: environment.cwd.clone(),
                manifest_found: false,
                manifest_path: local_project_model_manifest(&environment.cwd),
                freshness: WorkspaceContextFreshness::Unknown {
                    reason: "project-model manifest not found".to_string(),
                },
                manifest_hash: None,
                exact_fact_readiness: None,
                evidence_readiness: None,
                evidence_ledger_activation: None,
            });
        }
        let mut replay_preview_diagnostics = Vec::new();
        for manifest_diagnostic in replay_preview_targets.iter().take(Self::MAX_TARGETS) {
            match self
                .services
                .workspace_evidence_replay_preview_diagnostic(
                    manifest_diagnostic.workspace_root.clone(),
                )
                .await
            {
                Ok(diagnostic) => replay_preview_diagnostics.push(diagnostic),
                Err(error) => {
                    tracing::debug!(error = ?error, path = %manifest_diagnostic.workspace_root.display(), "Explain-context replay preview diagnostic failed for target");
                    replay_preview_diagnostics.push(Self::failed_replay_preview_diagnostic(
                        manifest_diagnostic,
                        format!("replay preview diagnostic failed: {error}"),
                    ));
                }
            }
        }

        let semantic_readiness = self
            .semantic_readiness_diagnostics(&target_specs, &nearest_skipped_manifest_candidates)
            .await;
        let retrieval_empty_targets = Vec::new();
        let has_query = query.as_deref().is_some_and(|query| !query.is_empty());
        let rerank_runtime = self
            .services
            .project_context_reranker_diagnostic()
            .await
            .ok();
        let retrieval_plan_diagnostics = if has_query {
            self.retrieval_plan_diagnostics(
                &target_specs,
                &nearest_skipped_manifest_candidates,
                query.as_deref().unwrap_or_default(),
                rerank_runtime,
            )
        } else {
            Vec::new()
        };
        let would_inject = has_query && !selected_targets.is_empty();

        let skip_reason = if would_inject {
            None
        } else if query.as_deref().is_none_or(str::is_empty) {
            Some("query not provided; automatic injection needs a latest user message".to_string())
        } else if selected_targets.is_empty() {
            Some("no fresh project-model manifest target selected".to_string())
        } else {
            Some("read-only explain does not run query-specific retrieval; replay preview is existing ledger evidence only".to_string())
        };

        WorkspaceContextExplanation {
            cwd: environment.cwd,
            query,
            candidates: candidate_diagnostics,
            selected_targets,
            nearest_skipped_manifest_candidates,
            retrieval_empty_targets,
            semantic_readiness,
            retrieval_plan_diagnostics,
            replay_preview_diagnostics,
            would_inject,
            skip_reason,
        }
    }

    async fn semantic_readiness_diagnostics(
        &self,
        target_specs: &[ProjectContextTargetDiagnostic],
        skipped_manifest_candidates: &[WorkspaceContextManifestDiagnostic],
    ) -> Vec<WorkspaceSemanticReadinessDiagnostic> {
        let embedding_model_id = self
            .services
            .get_config()
            .ok()
            .and_then(|config| config.semantic_embedding_model_id);
        let mut diagnostics = Vec::new();
        for target_diagnostic in target_specs.iter().take(Self::MAX_TARGETS) {
            diagnostics.push(
                self.semantic_readiness_diagnostic_for_workspace(
                    &target_diagnostic.target.workspace_root,
                    embedding_model_id.clone(),
                )
                .await,
            );
        }
        let remaining = Self::MAX_TARGETS.saturating_sub(diagnostics.len());
        for manifest_diagnostic in skipped_manifest_candidates.iter().take(remaining) {
            if manifest_diagnostic.manifest_found {
                diagnostics.push(
                    self.semantic_readiness_diagnostic_for_workspace(
                        &manifest_diagnostic.workspace_root,
                        embedding_model_id.clone(),
                    )
                    .await,
                );
            } else {
                diagnostics.push(WorkspaceSemanticReadinessDiagnostic {
                    workspace_root_label: "workspace_root".to_string(),
                    evaluated: false,
                    status: WorkspaceSemanticReadinessStatus::NotEvaluated,
                    dimension: None,
                    not_evaluated_reason: Some("project-model manifest not found".to_string()),
                });
            }
        }
        diagnostics
    }

    async fn semantic_readiness_diagnostic_for_workspace(
        &self,
        workspace_root: &Path,
        embedding_model_id: Option<String>,
    ) -> WorkspaceSemanticReadinessDiagnostic {
        match self
            .services
            .semantic_injection_readiness(workspace_root.to_path_buf(), embedding_model_id)
            .await
        {
            Ok(readiness) => Self::semantic_readiness_diagnostic_from_readiness(readiness),
            Err(error) => {
                tracing::debug!(error = ?error, path = %workspace_root.display(), "Explain-context semantic readiness diagnostic failed for target");
                WorkspaceSemanticReadinessDiagnostic {
                    workspace_root_label: "workspace_root".to_string(),
                    evaluated: true,
                    status: WorkspaceSemanticReadinessStatus::VectorIndexCorruptOrNotReady,
                    dimension: None,
                    not_evaluated_reason: None,
                }
            }
        }
    }

    fn semantic_readiness_diagnostic_from_readiness(
        readiness: WorkspaceSemanticInjectionReadiness,
    ) -> WorkspaceSemanticReadinessDiagnostic {
        let (status, dimension) = match readiness {
            WorkspaceSemanticInjectionReadiness::SemanticDisabledNoModelConfig => (
                WorkspaceSemanticReadinessStatus::SemanticDisabledNoModelConfig,
                None,
            ),
            WorkspaceSemanticInjectionReadiness::VectorIndexAbsentOrNoMatch => (
                WorkspaceSemanticReadinessStatus::VectorIndexAbsentOrNoMatch,
                None,
            ),
            WorkspaceSemanticInjectionReadiness::VectorIndexReady { dimension } => (
                WorkspaceSemanticReadinessStatus::VectorIndexReady,
                Some(dimension),
            ),
            WorkspaceSemanticInjectionReadiness::VectorIndexAmbiguous => {
                (WorkspaceSemanticReadinessStatus::VectorIndexAmbiguous, None)
            }
            WorkspaceSemanticInjectionReadiness::VectorIndexCorruptOrNotReady => (
                WorkspaceSemanticReadinessStatus::VectorIndexCorruptOrNotReady,
                None,
            ),
            WorkspaceSemanticInjectionReadiness::VectorDimensionMismatch { .. } => (
                WorkspaceSemanticReadinessStatus::VectorDimensionMismatch,
                None,
            ),
            WorkspaceSemanticInjectionReadiness::EmbeddingProviderUnavailable => (
                WorkspaceSemanticReadinessStatus::EmbeddingProviderUnavailable,
                None,
            ),
            WorkspaceSemanticInjectionReadiness::EmbeddingProviderTimeout => (
                WorkspaceSemanticReadinessStatus::EmbeddingProviderTimeout,
                None,
            ),
        };
        WorkspaceSemanticReadinessDiagnostic {
            workspace_root_label: "workspace_root".to_string(),
            evaluated: true,
            status,
            dimension,
            not_evaluated_reason: None,
        }
    }

    fn retrieval_plan_diagnostics(
        &self,
        target_specs: &[ProjectContextTargetDiagnostic],
        skipped_manifest_candidates: &[WorkspaceContextManifestDiagnostic],
        query: &str,
        rerank_runtime: Option<WorkspaceRerankRuntimeDiagnostic>,
    ) -> Vec<WorkspaceRetrievalPlanDiagnostic> {
        let max_sources = ProjectModelContextRenderBudget::default().max_sources;
        let mut diagnostics = target_specs
            .iter()
            .take(Self::MAX_TARGETS)
            .filter_map(|target_diagnostic| {
                Self::retrieval_plan_diagnostic_for_workspace(
                    &target_diagnostic.target.workspace_root,
                    target_diagnostic.target.path_filter.clone(),
                    query,
                    max_sources,
                    rerank_runtime.clone(),
                )
            })
            .collect::<Vec<_>>();
        diagnostics.extend(
            skipped_manifest_candidates
                .iter()
                .take(Self::MAX_TARGETS.saturating_sub(diagnostics.len()))
                .filter(|manifest_diagnostic| manifest_diagnostic.manifest_found)
                .filter_map(|manifest_diagnostic| {
                    Self::retrieval_plan_diagnostic_for_workspace(
                        &manifest_diagnostic.workspace_root,
                        None,
                        query,
                        max_sources,
                        rerank_runtime.clone(),
                    )
                }),
        );
        diagnostics
    }

    fn retrieval_plan_diagnostic_for_workspace(
        workspace_root: &PathBuf,
        path_filter: Option<String>,
        query: &str,
        max_sources: usize,
        rerank_runtime: Option<WorkspaceRerankRuntimeDiagnostic>,
    ) -> Option<WorkspaceRetrievalPlanDiagnostic> {
        let indexer = ProjectIndexer::new(workspace_root, local_project_model_dir(workspace_root));
        let manifest = match indexer.read_manifest() {
            Ok(manifest) => manifest,
            Err(error) => {
                tracing::debug!(error = ?error, path = %workspace_root.display(), "Explain-context retrieval-plan diagnostic could not read manifest");
                return None;
            }
        };
        let freshness = match indexer.evaluate_manifest_freshness(&manifest) {
            Ok(freshness) => freshness,
            Err(error) => {
                tracing::debug!(error = ?error, path = %workspace_root.display(), "Explain-context retrieval-plan diagnostic could not evaluate freshness");
                return None;
            }
        };
        let request = ProjectContextRetrievalRequest::new(
            query.to_string(),
            max_sources,
            ProjectContextPathScope::new(path_filter, Vec::new()),
            true,
        );
        let outcome = plan_project_context_retrieval(&manifest, &freshness, request);
        let mut diagnostic = Self::retrieval_plan_diagnostic_to_domain(
            ProjectContextRetrievalPlanDiagnostic::from_outcome(&outcome),
        );
        if let Some(rerank_runtime) = rerank_runtime {
            diagnostic.phase_diagnostics.rerank =
                rerank_runtime.project_phase_status(diagnostic.rerank_intent_len);
            diagnostic.rerank_runtime = Some(rerank_runtime);
        }
        Some(diagnostic)
    }

    fn retrieval_plan_diagnostic_to_domain(
        diagnostic: ProjectContextRetrievalPlanDiagnostic,
    ) -> WorkspaceRetrievalPlanDiagnostic {
        WorkspaceRetrievalPlanDiagnostic {
            workspace_root_label: "workspace_root".to_string(),
            manifest_label: "project_model_manifest".to_string(),
            planned: diagnostic.planned,
            refusal_code: diagnostic.refusal_code.map(|code| format!("{code:?}")),
            refusal_detail: diagnostic.refusal_detail,
            selected_result_count: diagnostic.selected_result_count,
            read_request_count: diagnostic.read_request_count,
            write_decision: diagnostic.write_decision.map(Self::write_decision_label),
            selected_summaries: diagnostic
                .selected_summaries
                .into_iter()
                .map(Self::selected_summary_to_domain)
                .collect(),
            read_request_summaries: diagnostic
                .read_request_summaries
                .into_iter()
                .map(Self::read_request_summary_to_domain)
                .collect(),
            phase_diagnostics: Self::phase_diagnostics_to_domain(diagnostic.phase_diagnostics),
            rerank_intent_source: diagnostic
                .rerank_intent_source
                .map(|source| format!("{source:?}")),
            rerank_intent_fingerprint: diagnostic.rerank_intent_fingerprint,
            rerank_intent_len: diagnostic.rerank_intent_len,
            offline_rerank_applicability: diagnostic
                .offline_rerank_applicability
                .map(Self::offline_rerank_applicability_to_domain),
            rerank_runtime: None,
            retrieval_empty: diagnostic.retrieval_empty,
            truncated: diagnostic.truncated,
        }
    }

    fn offline_rerank_applicability_to_domain(
        applicability: forge_project_model::OfflineRerankApplicability,
    ) -> WorkspaceOfflineRerankApplicability {
        match applicability {
            forge_project_model::OfflineRerankApplicability::ExactMatch => {
                WorkspaceOfflineRerankApplicability::ExactMatch
            }
            forge_project_model::OfflineRerankApplicability::Mismatch { reasons } => {
                WorkspaceOfflineRerankApplicability::Mismatch {
                    reasons: reasons
                        .into_iter()
                        .map(Self::offline_rerank_mismatch_to_domain)
                        .collect(),
                }
            }
        }
    }

    fn offline_rerank_mismatch_to_domain(
        mismatch: forge_project_model::OfflineRerankApplicabilityMismatch,
    ) -> WorkspaceOfflineRerankApplicabilityMismatch {
        match mismatch {
            forge_project_model::OfflineRerankApplicabilityMismatch::ManifestHashMismatch => {
                WorkspaceOfflineRerankApplicabilityMismatch::ManifestHashMismatch
            }
            forge_project_model::OfflineRerankApplicabilityMismatch::RerankIntentFingerprintMismatch => {
                WorkspaceOfflineRerankApplicabilityMismatch::RerankIntentFingerprintMismatch
            }
            forge_project_model::OfflineRerankApplicabilityMismatch::CandidateIdsOrderMismatch => {
                WorkspaceOfflineRerankApplicabilityMismatch::CandidateIdsOrderMismatch
            }
            forge_project_model::OfflineRerankApplicabilityMismatch::CandidateContentFingerprintMismatch => {
                WorkspaceOfflineRerankApplicabilityMismatch::CandidateContentFingerprintMismatch
            }
            forge_project_model::OfflineRerankApplicabilityMismatch::TopKScopeMismatch => {
                WorkspaceOfflineRerankApplicabilityMismatch::TopKScopeMismatch
            }
            forge_project_model::OfflineRerankApplicabilityMismatch::ProducerIdentityPolicyMismatch => {
                WorkspaceOfflineRerankApplicabilityMismatch::ProducerIdentityPolicyMismatch
            }
            forge_project_model::OfflineRerankApplicabilityMismatch::ScoreArtifactVersionMismatch => {
                WorkspaceOfflineRerankApplicabilityMismatch::ScoreArtifactVersionMismatch
            }
        }
    }

    fn phase_diagnostics_to_domain(
        diagnostics: ProjectContextRetrievalPhaseDiagnostics,
    ) -> WorkspaceRetrievalPhaseDiagnostics {
        WorkspaceRetrievalPhaseDiagnostics {
            lexical: Self::phase_status_to_domain(diagnostics.lexical),
            graph: Self::phase_status_to_domain(diagnostics.graph),
            vector: Self::phase_status_to_domain(diagnostics.vector),
            rerank: Self::phase_status_to_domain(diagnostics.rerank),
        }
    }

    fn phase_status_to_domain(
        status: ProjectContextRetrievalPhaseStatus,
    ) -> WorkspaceRetrievalPhaseStatus {
        match status {
            ProjectContextRetrievalPhaseStatus::Active { result_count } => {
                WorkspaceRetrievalPhaseStatus::Active { result_count }
            }
            ProjectContextRetrievalPhaseStatus::Skipped(reason) => {
                WorkspaceRetrievalPhaseStatus::Skipped {
                    reason: Self::phase_skip_reason_to_domain(reason),
                }
            }
            ProjectContextRetrievalPhaseStatus::Unavailable(reason) => {
                WorkspaceRetrievalPhaseStatus::Unavailable {
                    reason: Self::phase_unavailable_reason_to_domain(reason),
                }
            }
            ProjectContextRetrievalPhaseStatus::Invalid(reason) => {
                WorkspaceRetrievalPhaseStatus::Invalid {
                    reason: Self::phase_invalid_reason_to_domain(reason),
                }
            }
        }
    }

    fn phase_skip_reason_to_domain(
        reason: ProjectContextRetrievalPhaseSkipReason,
    ) -> WorkspaceRetrievalPhaseSkipReason {
        match reason {
            ProjectContextRetrievalPhaseSkipReason::EmptyQueryText => {
                WorkspaceRetrievalPhaseSkipReason::EmptyQueryText
            }
            ProjectContextRetrievalPhaseSkipReason::EmptyRerankIntent => {
                WorkspaceRetrievalPhaseSkipReason::EmptyRerankIntent
            }
            ProjectContextRetrievalPhaseSkipReason::GraphExpansionDisabled => {
                WorkspaceRetrievalPhaseSkipReason::GraphExpansionDisabled
            }
        }
    }

    fn phase_unavailable_reason_to_domain(
        reason: ProjectContextRetrievalPhaseUnavailableReason,
    ) -> WorkspaceRetrievalPhaseUnavailableReason {
        match reason {
            ProjectContextRetrievalPhaseUnavailableReason::MissingQueryEmbedding => {
                WorkspaceRetrievalPhaseUnavailableReason::MissingQueryEmbedding
            }
            ProjectContextRetrievalPhaseUnavailableReason::MissingVectorIndex => {
                WorkspaceRetrievalPhaseUnavailableReason::MissingVectorIndex
            }
            ProjectContextRetrievalPhaseUnavailableReason::VectorIndexNotReady => {
                WorkspaceRetrievalPhaseUnavailableReason::VectorIndexNotReady
            }
            ProjectContextRetrievalPhaseUnavailableReason::MissingReranker => {
                WorkspaceRetrievalPhaseUnavailableReason::MissingReranker
            }
            ProjectContextRetrievalPhaseUnavailableReason::RerankerNotReady => {
                WorkspaceRetrievalPhaseUnavailableReason::RerankerNotReady
            }
            ProjectContextRetrievalPhaseUnavailableReason::NoMatchingVectorIndex => {
                WorkspaceRetrievalPhaseUnavailableReason::NoMatchingVectorIndex
            }
            ProjectContextRetrievalPhaseUnavailableReason::AmbiguousVectorIndex => {
                WorkspaceRetrievalPhaseUnavailableReason::AmbiguousVectorIndex
            }
        }
    }

    fn phase_invalid_reason_to_domain(
        reason: ProjectContextRetrievalPhaseInvalidReason,
    ) -> WorkspaceRetrievalPhaseInvalidReason {
        match reason {
            ProjectContextRetrievalPhaseInvalidReason::VectorDimensionMismatch {
                query_dimension,
                index_dimension,
            } => WorkspaceRetrievalPhaseInvalidReason::VectorDimensionMismatch {
                query_dimension,
                index_dimension,
            },
            ProjectContextRetrievalPhaseInvalidReason::VectorIndexZeroDimension => {
                WorkspaceRetrievalPhaseInvalidReason::VectorIndexZeroDimension
            }
            ProjectContextRetrievalPhaseInvalidReason::EmptyQueryEmbedding => {
                WorkspaceRetrievalPhaseInvalidReason::EmptyQueryEmbedding
            }
            ProjectContextRetrievalPhaseInvalidReason::NonFiniteQueryEmbedding => {
                WorkspaceRetrievalPhaseInvalidReason::NonFiniteQueryEmbedding
            }
            ProjectContextRetrievalPhaseInvalidReason::ZeroQueryEmbeddingNorm => {
                WorkspaceRetrievalPhaseInvalidReason::ZeroQueryEmbeddingNorm
            }
        }
    }

    fn write_decision_label(decision: ProjectContextWriteDecision) -> String {
        match decision {
            ProjectContextWriteDecision::NoWriteEmptyRetrieval => {
                "NoWriteEmptyRetrieval".to_string()
            }
            ProjectContextWriteDecision::WriteContextPackAfterReadback => {
                "WriteContextPackAfterReadback".to_string()
            }
        }
    }

    fn selected_summary_to_domain(
        summary: ProjectContextRetrievalSelectedSummary,
    ) -> WorkspaceRetrievalPlanSelectedSummary {
        WorkspaceRetrievalPlanSelectedSummary {
            evidence_id: summary.evidence_id,
            path: summary.path,
            start_line: summary.start_line,
            end_line: summary.end_line,
            relevance: summary.relevance,
        }
    }

    fn read_request_summary_to_domain(
        summary: ProjectContextRetrievalReadRequestSummary,
    ) -> WorkspaceRetrievalPlanReadRequestSummary {
        WorkspaceRetrievalPlanReadRequestSummary {
            evidence_id: summary.evidence_id,
            path: summary.path,
            start_line: summary.start_line,
            end_line: summary.end_line,
        }
    }

    fn failed_replay_preview_diagnostic(
        manifest_diagnostic: &WorkspaceContextManifestDiagnostic,
        reason: String,
    ) -> WorkspaceEvidenceReplayPreviewDiagnostic {
        WorkspaceEvidenceReplayPreviewDiagnostic {
            status: WorkspaceEvidenceReplayPreviewStatus::NotPreviewedManifestUnknown,
            workspace_root_label: "workspace_root".to_string(),
            manifest_label: "project_model_manifest".to_string(),
            manifest_found: manifest_diagnostic.manifest_found,
            manifest_freshness: manifest_diagnostic.freshness.label().to_string(),
            not_previewed_reason: Some(reason),
            manifest_hash: None,
            content_policy: None,
            stale_policy: None,
            changed_excluded: 0,
            deleted_excluded: 0,
            budget: None,
            selected: Vec::new(),
            issues: Vec::new(),
            rendered_preview: None,
        }
    }

    async fn resolve_target_diagnostic(
        &self,
        path: PathBuf,
        budget: &mut TargetResolutionBudget,
    ) -> (
        WorkspaceContextCandidateDiagnostic,
        Option<WorkspaceContextManifestDiagnostic>,
        Option<WorkspaceContextManifestDiagnostic>,
        Option<ProjectContextTarget>,
    ) {
        let candidate_path = path.clone();
        for ancestor in path.ancestors() {
            if !budget.claim_index_probe() {
                return (
                    WorkspaceContextCandidateDiagnostic {
                        candidate_path: candidate_path.clone(),
                        selected_workspace: None,
                        path_filter: None,
                        skip_reason: Some("index freshness probe limit reached".to_string()),
                    },
                    None,
                    None,
                    None,
                );
            }
            let diagnostic = match self
                .services
                .project_model_context_diagnostic(ancestor)
                .await
            {
                Ok(diagnostic) => diagnostic,
                Err(error) => {
                    return (
                        WorkspaceContextCandidateDiagnostic {
                            candidate_path: candidate_path.clone(),
                            selected_workspace: None,
                            path_filter: None,
                            skip_reason: Some(format!(
                                "freshness check failed for {}: {}",
                                ancestor.display(),
                                error
                            )),
                        },
                        None,
                        None,
                        None,
                    );
                }
            };
            if !diagnostic.can_inject() {
                if diagnostic.manifest_found {
                    return (
                        WorkspaceContextCandidateDiagnostic {
                            candidate_path: path,
                            selected_workspace: None,
                            path_filter: None,
                            skip_reason: Some(Self::manifest_skip_reason(&diagnostic)),
                        },
                        None,
                        Some(diagnostic),
                        None,
                    );
                }
                continue;
            }
            let workspace_root = ancestor.to_path_buf();
            let path_filter = directory_path_filter(&path, &workspace_root);
            let target = ProjectContextTarget::new(workspace_root.clone(), path_filter.clone());
            return (
                WorkspaceContextCandidateDiagnostic {
                    candidate_path: candidate_path.clone(),
                    selected_workspace: Some(workspace_root),
                    path_filter,
                    skip_reason: None,
                },
                Some(diagnostic),
                None,
                Some(target),
            );
        }
        (
            WorkspaceContextCandidateDiagnostic {
                candidate_path: candidate_path.clone(),
                selected_workspace: None,
                path_filter: None,
                skip_reason: Some(
                    "no fresh project-model manifest found in candidate ancestors".to_string(),
                ),
            },
            None,
            None,
            None,
        )
    }

    fn manifest_skip_reason(diagnostic: &WorkspaceContextManifestDiagnostic) -> String {
        match &diagnostic.freshness {
            WorkspaceContextFreshness::Fresh => "manifest is fresh".to_string(),
            WorkspaceContextFreshness::Unknown { reason } => format!(
                "project-model manifest freshness unknown at {}: {}",
                diagnostic.manifest_path.display(),
                reason
            ),
            WorkspaceContextFreshness::Stale { changed, deleted, added } => format!(
                "project-model manifest stale at {}: changed=[{}] deleted=[{}] added=[{}]",
                diagnostic.manifest_path.display(),
                changed.join(","),
                deleted.join(","),
                added.join(",")
            ),
        }
    }

    async fn resolve_targets(
        &self,
        environment: &Environment,
        latest_user_message: &str,
    ) -> Vec<ProjectContextTargetDiagnostic> {
        let mut candidates = vec![environment.cwd.clone()];
        candidates.extend(mentioned_paths(
            latest_user_message,
            &environment.cwd,
            environment.home.as_deref(),
        ));

        let mut budget = TargetResolutionBudget::new(
            Self::MAX_EXPLICIT_TARGET_CANDIDATES,
            Self::MAX_INDEX_PROBES,
        );
        let mut targets = Vec::new();
        let mut seen = BTreeSet::new();
        for candidate in candidates {
            if !budget.claim_candidate() {
                break;
            }
            let Some((target, diagnostic)) = self.resolve_target(candidate, &mut budget).await
            else {
                continue;
            };
            if seen.insert(target.clone()) {
                targets.push(ProjectContextTargetDiagnostic { target, diagnostic });
            }
            if targets.len() >= Self::MAX_TARGETS {
                break;
            }
        }
        targets
    }

    async fn resolve_target(
        &self,
        path: PathBuf,
        budget: &mut TargetResolutionBudget,
    ) -> Option<(ProjectContextTarget, WorkspaceContextManifestDiagnostic)> {
        for ancestor in path.ancestors() {
            if !budget.claim_index_probe() {
                return None;
            }
            let diagnostic = match self
                .services
                .project_model_context_diagnostic(ancestor)
                .await
            {
                Ok(diagnostic) => diagnostic,
                Err(error) => {
                    tracing::debug!(error = ?error, path = %ancestor.display(), "Skipping project-model context target because index freshness could not be checked");
                    continue;
                }
            };
            if !diagnostic.can_inject() {
                if diagnostic.manifest_found {
                    tracing::debug!(path = %ancestor.display(), freshness = diagnostic.freshness.label(), "Stopping project-model context target resolution because nearest manifest is not injectable");
                    return None;
                }
                tracing::debug!(path = %ancestor.display(), freshness = diagnostic.freshness.label(), "Skipping project-model context target because manifest is unavailable");
                continue;
            }
            let workspace_root = ancestor.to_path_buf();
            let path_filter = directory_path_filter(&path, &workspace_root);
            return Some((
                ProjectContextTarget::new(workspace_root, path_filter),
                diagnostic,
            ));
        }
        None
    }

    fn query_from_conversation(conversation: &Conversation) -> Option<String> {
        Self::latest_real_user_message(conversation).map(ToOwned::to_owned)
    }

    fn latest_real_user_message(conversation: &Conversation) -> Option<&str> {
        conversation
            .context
            .as_ref()?
            .messages
            .iter()
            .rev()
            .find(|message| {
                message.has_role(Role::User)
                    && !message.is_droppable()
                    && !matches!(
                        &message.message,
                        ContextMessage::Text(text) if text.is_internal_context()
                    )
            })
            .and_then(|message| message.content())
            .map(str::trim)
            .filter(|content| !content.is_empty())
    }

    fn render_context(
        workspace_root: &std::path::Path,
        diagnostic: &WorkspaceContextManifestDiagnostic,
        semantic_diagnostic: Option<&str>,
        nodes: Vec<Node>,
    ) -> Option<ProjectContextRenderedPartition> {
        let manifest_path = local_project_model_manifest(workspace_root);
        let source_nodes = nodes
            .into_iter()
            .map(Self::source_node_from_node)
            .collect::<Vec<_>>();
        let exact_fact_readiness = diagnostic
            .exact_fact_readiness
            .as_ref()
            .map(Self::exact_fact_readiness_metadata);
        let evidence_readiness = diagnostic
            .evidence_readiness
            .as_ref()
            .map(Self::evidence_readiness_metadata);
        let evidence_ledger_activation = diagnostic
            .evidence_ledger_activation
            .as_ref()
            .map(Self::evidence_ledger_activation_metadata);
        let readiness = ProjectModelContextReadinessMetadata {
            exact_fact_readiness,
            evidence_readiness,
            evidence_ledger_activation,
        };
        let mut diagnostics = Vec::new();
        diagnostics.push(format!("freshness={}", diagnostic.freshness.label()));
        if let Some(semantic_diagnostic) = semantic_diagnostic {
            diagnostics.push(semantic_diagnostic.to_string());
        }
        if let Some(readiness) = &readiness.exact_fact_readiness {
            diagnostics.push(format!("exact_fact_status={}", readiness.status_label));
            if let Some(fingerprint) = &readiness.manifest_external_facts_fingerprint {
                diagnostics.push(format!("manifest_external_facts_fingerprint={fingerprint}"));
            }
            diagnostics.extend(readiness.issue_summaries.clone());
        }
        if let Some(readiness) = &readiness.evidence_readiness {
            diagnostics.push(format!(
                "context_pack_issue_count={}",
                readiness.context_pack_issue_count
            ));
            diagnostics.push(format!(
                "tool_episode_issue_count={}",
                readiness.tool_episode_issue_count
            ));
            diagnostics.push(format!(
                "episode_artifact_link_valid={}",
                readiness.episode_artifact_link_valid
            ));
            diagnostics.extend(readiness.issue_summaries.clone());
        }
        if let Some(activation) = &readiness.evidence_ledger_activation {
            diagnostics.extend(activation.issue_summaries.clone());
        }
        let manifest_freshness = diagnostic.manifest_hash.clone().map_or_else(
            || ProjectModelManifestFreshnessProof::Unknown {
                reason: "manifest hash is absent".to_string(),
            },
            |manifest_hash| {
                if diagnostic.freshness.is_fresh() {
                    ProjectModelManifestFreshnessProof::KnownFresh {
                        schema_version: 1,
                        manifest_hash,
                        freshness_label: diagnostic.freshness.label().to_string(),
                    }
                } else {
                    ProjectModelManifestFreshnessProof::Stale {
                        manifest_hash: Some(manifest_hash),
                        reason: format!("manifest freshness is {}", diagnostic.freshness.label()),
                    }
                }
            },
        );
        let envelope = match build_project_model_context_envelope(
            ProjectModelContextEnvelopeInput {
                render_root: ProjectModelContextRenderRoot::new(
                    workspace_root.display().to_string(),
                    manifest_path.display().to_string(),
                    diagnostic.freshness.label().to_string(),
                    "WorkspaceService::query_workspace",
                ),
                manifest_freshness,
                render_budget: ProjectModelContextRenderBudget::default(),
                source_nodes,
                readiness,
                semantic_diagnostics: Vec::new(),
                volatile: ProjectModelVolatileSidecarInput {
                    diagnostics,
                    readiness_warnings: Vec::new(),
                    mutable_file_freshness: Vec::new(),
                    transient_errors: Vec::new(),
                    ..Default::default()
                },
                agents_project_rules_digest: None,
            },
        ) {
            Ok(envelope) => envelope,
            Err(refusal) => {
                tracing::debug!(refusal = ?refusal, path = %workspace_root.display(), "Skipping project-model context injection because envelope construction refused");
                return None;
            }
        };
        Some(ProjectContextRenderedPartition {
            stable_payload: envelope.stable_provider_visible_message,
            volatile_sidecar: envelope.volatile_sidecar_message,
            stable_identity: envelope.stable_identity,
        })
    }

    fn exact_fact_readiness_metadata(
        readiness: &WorkspaceExactFactReadinessDiagnostic,
    ) -> ProjectModelExactFactReadinessMetadata {
        ProjectModelExactFactReadinessMetadata {
            status_label: readiness.status_label.clone(),
            exact_facts_active: readiness.exact_facts_active,
            issue_count: readiness.issue_count,
            issue_summaries: readiness.issue_summaries.clone(),
            manifest_hash: readiness.manifest_hash.clone(),
            manifest_external_facts_fingerprint: readiness
                .manifest_external_facts_fingerprint
                .clone(),
            reference_edge_count: readiness.reference_edge_count,
            exact_compiler_reference_edge_count: readiness.exact_compiler_reference_edge_count,
        }
    }

    fn evidence_readiness_metadata(
        readiness: &WorkspaceEvidenceReadinessDiagnostic,
    ) -> ProjectModelEvidenceReadinessMetadata {
        ProjectModelEvidenceReadinessMetadata {
            context_pack_artifact_count: readiness.context_pack_artifact_count,
            context_pack_valid: readiness.context_pack_valid,
            context_pack_issue_count: readiness.context_pack_issue_count,
            tool_episode_count: readiness.tool_episode_count,
            tool_episode_valid: readiness.tool_episode_valid,
            tool_episode_issue_count: readiness.tool_episode_issue_count,
            episode_artifact_link_valid: readiness.episode_artifact_link_valid,
            linked_episode_count: readiness.linked_episode_count,
            missing_link_count: readiness.missing_link_count,
            worst_case_freshness: readiness.worst_case_freshness.clone(),
            issue_summaries: readiness.issue_summaries.clone(),
            truncated: readiness.truncated,
        }
    }

    fn evidence_ledger_activation_metadata(
        activation: &WorkspaceEvidenceLedgerActivationDiagnostic,
    ) -> ProjectModelEvidenceLedgerActivationMetadata {
        ProjectModelEvidenceLedgerActivationMetadata {
            context_pack_artifact_count: activation.summary.context_pack_artifact_count,
            readable_context_pack_count: activation.summary.readable_context_pack_count,
            tool_episode_count: activation.summary.tool_episode_count,
            linked_episode_count: activation.summary.linked_episode_count,
            missing_link_count: activation.summary.missing_link_count,
            graph_node_count: activation.summary.graph_node_count,
            graph_edge_count: activation.summary.graph_edge_count,
            worst_case_freshness: activation.summary.worst_case_freshness.clone(),
            issue_count: activation.summary.issue_count,
            issue_summaries: activation.summary.issue_summaries.clone(),
            truncated: activation.summary.truncated,
        }
    }

    fn source_node_from_node(node: Node) -> ProjectModelSourceNode {
        let node_id = node.node_id.as_str().to_string();
        let score = node.relevance;
        match node.node {
            NodeData::FileChunk(chunk) => ProjectModelSourceNode::FileChunk {
                path: chunk.file_path,
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                node_id,
                score,
                content: chunk.content,
            },
            NodeData::File(file) => ProjectModelSourceNode::File {
                path: file.file_path,
                node_id,
                score,
                content_hash: file.hash,
                content: Some(file.content),
            },
            NodeData::FileRef(file_ref) => ProjectModelSourceNode::FileRef {
                path: file_ref.file_path,
                node_id,
                score,
                content_hash: file_ref.file_hash,
            },
            NodeData::Note(note) => {
                ProjectModelSourceNode::Note { node_id, score, content: note.content }
            }
            NodeData::Task(task) => {
                ProjectModelSourceNode::Task { node_id, score, content: task.task }
            }
        }
    }
}

fn provider_request_compaction_estimate(
    estimate: OpenAiProviderRequestEstimate,
    input_budget: Option<usize>,
) -> ProviderRequestEstimate {
    ProviderRequestEstimate::new(estimate.estimated_input_tokens, input_budget)
}

/// ForgeApp handles the core chat functionality by orchestrating various
/// services. It encapsulates the complex logic previously contained in the
/// ForgeAPI chat method.
pub struct ForgeApp<S> {
    services: Arc<S>,
    tool_registry: ToolRegistry<S>,
}

impl<S: Services + EnvironmentInfra<Config = forge_config::ForgeConfig> + SteerService>
    ForgeApp<S>
{
    /// Creates a new ForgeApp instance with the provided services.
    pub fn new(services: Arc<S>) -> Self {
        Self { tool_registry: ToolRegistry::new(services.clone()), services }
    }

    /// Explains whether automatic project-model context would be injected for
    /// the current environment and optional query.
    pub async fn explain_workspace_context(
        &self,
        query: Option<String>,
    ) -> WorkspaceContextExplanation {
        let agent = Agent::new(
            AgentId::new("forge"),
            ProviderId::from("diagnostic-provider".to_string()),
            ModelId::new("diagnostic-model"),
        );
        ProjectContextInjection::new(self.services.clone(), agent)
            .explain(query)
            .await
    }

    /// Accepts a typed steer message for delayed primary-conversation delivery.
    ///
    /// # Arguments
    /// * `request` - The typed steer request to validate and queue.
    ///
    /// # Errors
    /// Returns an error when the conversation is missing or is not primary.
    pub async fn steer(&self, request: SteerRequest) -> anyhow::Result<()> {
        SteerHandle::<S>::new(self.services.clone())
            .accept(request)
            .await
    }

    /// Reviews a captured learning candidate through the typed append-only learning ledger.
    ///
    /// # Arguments
    /// * `request` - Typed review request with candidate ID, decision, note, and provenance.
    ///
    /// # Errors
    /// Returns an error when the candidate does not exist, is not reviewable, or persistence fails.
    pub async fn review_learning_candidate(
        &self,
        request: LearningReviewRequest,
    ) -> anyhow::Result<LearningReviewOutcome>
    where
        S: LearningService,
    {
        self.services.review_learning_candidate(request).await
    }

    /// Executes a chat request and returns a stream of responses.
    /// This method contains the core chat logic extracted from ForgeAPI.
    pub async fn chat(
        &self,
        agent_id: AgentId,
        chat: ChatRequest,
    ) -> Result<MpscStream<Result<ChatResponse, anyhow::Error>>> {
        let services = self.services.clone();

        // Get the conversation for the chat request
        let conversation = services
            .find_conversation(&chat.conversation_id)
            .await?
            .ok_or_else(|| forge_domain::Error::ConversationNotFound(chat.conversation_id))?;

        // Discover files using the discovery service
        let app_config = self.services.get_config()?;
        let environment = services.get_environment();

        let custom_instructions = services.get_custom_instructions().await;

        // Prepare agents with user configuration
        let agent_provider_resolver = AgentProviderResolver::new(services.clone());

        // Get agent and apply workflow config
        let agent = self
            .services
            .get_agent(&agent_id)
            .await?
            .ok_or(crate::Error::AgentNotFound(agent_id.clone()))?
            .apply_config(&app_config)
            .set_compact_model_if_none();

        let agent_provider = agent_provider_resolver
            .get_provider(Some(agent.id.clone()))
            .await?;
        let agent_provider = self
            .services
            .provider_auth_service()
            .refresh_provider_credential(agent_provider)
            .await?;

        let models = services.models(agent_provider.clone()).await?;
        let selected_model = models
            .iter()
            .find(|model| model.id == agent.model && model.provider_id == agent.provider)
            .ok_or_else(|| forge_domain::Error::MissingModel(agent.id.clone()))?;
        let agent = agent.compaction_threshold(Some(selected_model));

        // Get system and mcp tool definitions and resolve them for the agent
        let all_tool_definitions = self
            .tool_registry
            .list(&agent.id, selected_model, &agent_provider)
            .await?;
        let tool_resolver = ToolResolver::new(all_tool_definitions);
        let tool_definitions: Vec<ToolDefinition> =
            tool_resolver.resolve(&agent).into_iter().cloned().collect();
        let max_tool_failure_per_turn = agent.max_tool_failure_per_turn.unwrap_or(3);

        let current_time = Local::now();

        // Insert system prompt
        let conversation =
            SystemPrompt::new(self.services.clone(), environment.clone(), agent.clone())
                .custom_instructions(custom_instructions.clone())
                .tool_definitions(tool_definitions.clone())
                .models(models.clone())
                .max_extensions(app_config.max_extensions)
                .template_config(build_template_config(&app_config))
                .add_system_message(conversation)
                .await?;

        // Insert user prompt
        let conversation = UserPromptGenerator::new(
            self.services.clone(),
            agent.clone(),
            chat.event.clone(),
            current_time,
        )
        .add_user_prompt(conversation)
        .await?;

        // Inject local project-model context after the user prompt, before the
        // provider sees the request. This is a best-effort, manifest-gated read
        // path and never triggers hot-path indexing.
        let conversation = ProjectContextInjection::new(self.services.clone(), agent.clone())
            .inject(conversation)
            .await;

        // Inject reviewed learning records as late-bound internal context. This
        // path is reviewed-only, bounded, droppable, and cache-ineligible.
        let conversation = ProjectContextInjection::new(self.services.clone(), agent.clone())
            .inject_learning(conversation)
            .await;

        // Detect and render externally changed files notification
        let conversation = ChangedFiles::new(services.clone(), agent.clone())
            .update_file_stats(conversation)
            .await;

        let conversation = InitConversationMetrics::new(current_time).apply(conversation);
        let conversation = ApplyTunableParameters::new(agent.clone(), tool_definitions.clone())
            .apply(conversation);
        let conversation = SetConversationId.apply(conversation);

        // Create the orchestrator with all necessary dependencies
        let tracing_handler = TracingHandler::new();
        let title_handler = TitleGenerationHandler::new(services.clone());

        // Build the on_end hook, conditionally adding PendingTodosHandler based on
        // config
        let on_end_hook = if app_config.verify_todos {
            tracing_handler
                .clone()
                .and(title_handler.clone())
                .and(PendingTodosHandler::new())
        } else {
            tracing_handler.clone().and(title_handler.clone())
        };

        let hook = Hook::default()
            .on_start(tracing_handler.clone().and(title_handler))
            .on_request(tracing_handler.clone().and(DoomLoopDetector::default()))
            .on_response(
                tracing_handler
                    .clone()
                    .and(CompactionHandler::new(agent.clone(), environment.clone())),
            )
            .on_toolcall_start(tracing_handler.clone())
            .on_toolcall_end(tracing_handler)
            .on_end(on_end_hook);

        let orch = Orchestrator::new(
            services.clone(),
            conversation,
            agent,
            self.services.get_config()?,
        )
        .error_tracker(ToolErrorTracker::new(max_tool_failure_per_turn))
        .tool_definitions(tool_definitions)
        .models(models)
        .active_provider(agent_provider)
        .hook(Arc::new(hook));

        // Create and return the stream
        let stream = MpscStream::spawn(
            |tx: tokio::sync::mpsc::Sender<Result<ChatResponse, anyhow::Error>>| {
                async move {
                    // Execute dispatch and always save conversation afterwards
                    let mut orch = orch.sender(tx.clone());
                    let dispatch_result = orch.run().await;

                    // Always save conversation using get_conversation()
                    let conversation = orch.get_conversation().clone();
                    let save_result = save_conversation_and_capture_learning(
                        services.clone(),
                        conversation.clone(),
                    )
                    .await;

                    // Send any error to the stream (prioritize dispatch error over save error)
                    let final_err = match (dispatch_result, save_result) {
                        (Err(d), Err(s)) => {
                            Some(d.context(format!("Also failed to save conversation: {}", s)))
                        }
                        (Ok(_), Err(s)) => Some(s.context("Failed to save conversation")),
                        (Err(d), Ok(_)) => Some(d),
                        (Ok(_), Ok(_)) => None,
                    };

                    if let Some(err) = final_err {
                        if let Err(e) = tx.send(Err(err)).await {
                            tracing::error!("Failed to send error to stream: {}", e);
                        }
                    }
                }
            },
        );

        Ok(stream)
    }

    fn estimate_compaction_provider_request(
        &self,
        context: Context,
        agent: &Agent,
        models: Vec<Model>,
        provider: &Provider<Url>,
        merge_system_messages: bool,
    ) -> Result<ProviderRequestEstimate> {
        let conversation = Conversation::generate().context(context);
        let orchestrator = Orchestrator::new(
            self.services.clone(),
            conversation,
            agent.clone(),
            forge_config::ForgeConfig { merge_system_messages, ..Default::default() },
        )
        .models(models);
        let context = orchestrator
            .get_conversation()
            .context
            .clone()
            .unwrap_or_default();
        let (estimate, input_budget) =
            orchestrator.estimate_final_provider_request_for_provider(context, provider)?;
        Ok(provider_request_compaction_estimate(estimate, input_budget))
    }

    async fn compaction_provider_request_estimates(
        &self,
        original_context: Context,
        compacted_context: Context,
        agent: &Agent,
        merge_system_messages: bool,
    ) -> Result<(ProviderRequestEstimate, ProviderRequestEstimate)> {
        let agent_provider_resolver = AgentProviderResolver::new(self.services.clone());
        let agent_provider = agent_provider_resolver
            .get_provider(Some(agent.id.clone()))
            .await?;
        let agent_provider = self
            .services
            .provider_auth_service()
            .refresh_provider_credential(agent_provider)
            .await?;
        let models = self.services.models(agent_provider.clone()).await?;
        models
            .iter()
            .find(|model| model.id == agent.model && model.provider_id == agent.provider)
            .ok_or_else(|| forge_domain::Error::MissingModel(agent.id.clone()))?;

        let original_provider_request = self.estimate_compaction_provider_request(
            original_context,
            agent,
            models.clone(),
            &agent_provider,
            merge_system_messages,
        )?;
        let compacted_provider_request = self.estimate_compaction_provider_request(
            compacted_context,
            agent,
            models,
            &agent_provider,
            merge_system_messages,
        )?;
        Ok((original_provider_request, compacted_provider_request))
    }

    /// Compacts the context of the main agent for the given conversation and
    /// persists it. Returns metrics about the compaction (original vs.
    /// compacted tokens and messages).
    pub async fn compact_conversation(
        &self,
        active_agent_id: AgentId,
        conversation_id: &ConversationId,
    ) -> Result<CompactionResult> {
        use crate::compact::Compactor;

        // Get the conversation
        let mut conversation = self
            .services
            .find_conversation(conversation_id)
            .await?
            .ok_or_else(|| forge_domain::Error::ConversationNotFound(*conversation_id))?;

        // Get the context from the conversation
        let context = match conversation.context.take() {
            Some(context) => context,
            None => {
                // No context to compact, return zero metrics
                return Ok(CompactionResult::new(0, 0, 0, 0));
            }
        };

        // Calculate original metrics. User-facing prompt-size metrics deliberately
        // use the current message approximation instead of historical provider
        // usage preserved on compacted summary messages.
        let original_messages = context.messages.len();
        let original_token_count = context.token_count_approx();

        let forge_config = self.services.get_config()?;

        // Get agent and apply workflow config
        let agent = self.services.get_agent(&active_agent_id).await?;

        let Some(agent) = agent else {
            return Err(crate::Error::AgentNotFound(active_agent_id).into());
        };

        // Get compact config from the agent
        let agent = agent
            .apply_config(&forge_config)
            .set_compact_model_if_none();
        let compact = agent.compact.clone();

        // Apply compaction using the Compactor
        let environment = self.services.get_environment();
        let compacted_context =
            Compactor::new(compact, environment).compact(context.clone(), true)?;

        let compacted_messages = compacted_context.messages.len();
        let compacted_tokens = compacted_context.token_count_approx();
        let provider_request_estimates = match self
            .compaction_provider_request_estimates(
                context,
                compacted_context.clone(),
                &agent,
                forge_config.merge_system_messages,
            )
            .await
        {
            Ok(estimates) => Some(estimates),
            Err(error) => {
                tracing::warn!(
                    "Compaction provider request metrics unavailable; compacted context will still be saved: {error:#}"
                );
                None
            }
        };

        // Update the conversation with the compacted context
        conversation.context = Some(compacted_context);

        // Save the updated conversation
        self.services.upsert_conversation(conversation).await?;

        let result = CompactionResult::new(
            original_token_count,
            compacted_tokens,
            original_messages,
            compacted_messages,
        );

        if let Some((original_provider_request, compacted_provider_request)) =
            provider_request_estimates
        {
            Ok(result
                .provider_request_estimates(original_provider_request, compacted_provider_request))
        } else {
            Ok(result)
        }
    }

    pub async fn list_tools(&self) -> Result<ToolsOverview> {
        self.tool_registry.tools_overview().await
    }

    /// Gets available models for the default provider with automatic credential
    /// refresh.
    pub async fn get_models(&self) -> Result<Vec<Model>> {
        let agent_provider_resolver = AgentProviderResolver::new(self.services.clone());
        let provider = agent_provider_resolver.get_provider(None).await?;
        let provider = self
            .services
            .provider_auth_service()
            .refresh_provider_credential(provider)
            .await?;

        self.services.models(provider).await
    }

    /// Gets available models from all configured providers concurrently.
    ///
    /// Returns a list of `ProviderModels` for each configured provider that
    /// successfully returned models. If every configured provider fails (e.g.
    /// due to an invalid API key), the first error encountered is returned so
    /// the caller receives the real underlying cause rather than an empty list.
    pub async fn get_all_provider_models(&self) -> Result<Vec<ProviderModels>> {
        let all_providers = self.services.get_all_providers().await?;

        // Build one future per configured provider, preserving the error on failure.
        let futures: Vec<_> = all_providers
            .into_iter()
            .filter_map(|any_provider| any_provider.into_configured())
            .map(|provider| {
                let provider_id = provider.id.clone();
                let services = self.services.clone();
                async move {
                    let result: Result<ProviderModels> = async {
                        let refreshed = services
                            .provider_auth_service()
                            .refresh_provider_credential(provider)
                            .await?;
                        let models = services.models(refreshed).await?;
                        Ok(ProviderModels { provider_id, models })
                    }
                    .await;
                    result
                }
            })
            .collect();

        // Execute all provider fetches concurrently.
        let results = futures::future::join_all(futures).await;
        let mut successes = Vec::new();
        let mut first_error = None;
        for res in results {
            match res {
                Ok(models) => successes.push(models),
                Err(e) => {
                    tracing::warn!("Failed to fetch models from provider: {}", e);
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
            }
        }
        if successes.is_empty() {
            if let Some(err) = first_error {
                return Err(err);
            }
        }
        Ok(successes)
    }
}

async fn save_conversation_and_capture_learning<S>(
    services: Arc<S>,
    conversation: Conversation,
) -> anyhow::Result<()>
where
    S: ConversationService + LearningService + Send + Sync + 'static,
{
    services.upsert_conversation(conversation.clone()).await?;
    LearningCapture::new(services)
        .capture_saved_conversation(&conversation)
        .await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use anyhow::Result;
    use forge_domain::{
        Agent, AgentId, AnyProvider, AuthContextRequest, AuthContextResponse, AuthMethod,
        ChatCompletionMessage, ChatRequest, Content, Context, ContextMessage, Conversation,
        ConversationId, Environment, Event, FileChunk, FileStatus, FinishReason,
        LEARNING_LEDGER_SCHEMA_VERSION, LearningCaptureMetadata, LearningEventKind,
        LearningLedgerAppendOutcome, LearningLedgerEvent, LearningLedgerFreshness,
        LearningProvenance, LearningRecordId, LearningRecordProjection, LearningRedactionStatus,
        LearningReviewState, McpConfig, McpServers, Model, ModelId, Node, NodeData, NodeId,
        PermissionOperation, ProjectSemanticEmbeddingVector, Provider, ProviderId, ProviderType,
        ResultStream, Scope, SearchParams, SemSearchAvailability, SemSearchDiagnosticReport,
        SteerMessage, SyncProgress, ToolCallContext, ToolCallFull, ToolOutput, ToolResult,
        WorkspaceAuth, WorkspaceContextFreshness, WorkspaceContextManifestDiagnostic,
        WorkspaceEvidenceReadinessDiagnostic, WorkspaceId, WorkspaceInfo,
        WorkspaceSemanticInjectionReadiness,
    };
    use forge_project_model::{
        ProjectContextCommittedQueryResult, ProjectContextPackNoWriteReason,
    };
    use futures::StreamExt;
    use pretty_assertions::assert_eq;
    use tempfile::TempDir;
    use tokio::sync::Mutex;

    use super::*;
    use crate::agent::AgentService;
    use crate::orch::Orchestrator;
    use crate::{
        AppConfigService, AttachmentService, AuthService, CommandLoaderService,
        CustomInstructionsService, FileDiscoveryService, FollowUpService, FsPatchService,
        FsReadService, FsRemoveService, FsSearchService, FsUndoService, FsWriteService,
        ImageReadService, McpConfigManager, McpService, NetFetchService, PlanCreateService,
        PolicyService, ProviderAuthService, ShellService, SkillFetchService, TemplateService, User,
        UserUsage, Walker,
    };

    #[test]
    fn learning_context_rejects_generic_accepted_raw_summary_without_closed_projection() {
        let projection = LearningRecordProjection {
            record_id: LearningRecordId::generate(),
            summary: "generic raw accepted text".to_string(),
            accepted_summary: None,
            review_state: LearningReviewState::Accepted,
            redaction_status: LearningRedactionStatus::Clean,
            provenance: LearningProvenance::conversation(
                ConversationId::generate(),
                "raw-accepted",
                "raw-accepted-fingerprint",
            ),
            capture_metadata: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            schema_version: LEARNING_LEDGER_SCHEMA_VERSION,
        };

        let actual =
            ProjectContextInjection::<ChatFlowLearningHarness>::learning_record_to_project(
                projection,
            )
            .is_none();
        let expected = true;

        assert_eq!(actual, expected);
    }

    #[derive(Clone)]
    struct ChatFlowLearningHarness {
        state: Arc<ChatFlowLearningState>,
    }

    struct ChatFlowLearningState {
        cwd: PathBuf,
        conversations: Mutex<HashMap<ConversationId, Conversation>>,
        learning_events: Mutex<Vec<LearningLedgerEvent>>,
        learning_records: Mutex<Vec<LearningRecordProjection>>,
        captured_provider_context: Mutex<Option<Context>>,
        agent: Agent,
        model: Model,
        provider: Provider<Url>,
    }

    impl ChatFlowLearningHarness {
        fn new(cwd: PathBuf) -> Arc<Self> {
            let model_id = ModelId::new("runtime-proof-model");
            let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone())
                .tool_supported(false)
                .tools(Vec::<forge_domain::ToolName>::new())
                .max_requests_per_turn(1usize);
            let model = Model::new(ProviderId::OPENAI, model_id).context_length(200_000_u64);
            let provider = Provider {
                id: ProviderId::OPENAI,
                provider_type: ProviderType::Llm,
                response: None,
                url: Url::parse("http://127.0.0.1/runtime-proof").unwrap(),
                models: None,
                auth_methods: Vec::new(),
                url_params: Vec::new(),
                credential: None,
                custom_headers: None,
            };
            Arc::new(Self {
                state: Arc::new(ChatFlowLearningState {
                    cwd,
                    conversations: Mutex::new(HashMap::new()),
                    learning_events: Mutex::new(Vec::new()),
                    learning_records: Mutex::new(Vec::new()),
                    captured_provider_context: Mutex::new(None),
                    agent,
                    model,
                    provider,
                }),
            })
        }

        async fn set_learning_records(&self, records: Vec<LearningRecordProjection>) {
            *self.state.learning_records.lock().await = records;
        }

        async fn upsert_conversation(&self, conversation: Conversation) -> Result<()> {
            self.state.upsert_conversation(conversation).await
        }
    }

    impl EnvironmentInfra for ChatFlowLearningHarness {
        type Config = ForgeConfig;

        fn get_env_var(&self, _key: &str) -> Option<String> {
            None
        }

        fn get_env_vars(&self) -> BTreeMap<String, String> {
            BTreeMap::new()
        }

        fn get_environment(&self) -> Environment {
            Environment {
                os: "test".to_string(),
                cwd: self.state.cwd.clone(),
                home: None,
                shell: "sh".to_string(),
                base_path: self.state.cwd.join(".forge"),
            }
        }

        fn get_config(&self) -> Result<Self::Config> {
            Ok(ForgeConfig { max_parallel_file_reads: 4, ..Default::default() })
        }

        async fn update_environment(&self, _ops: Vec<forge_domain::ConfigOperation>) -> Result<()> {
            anyhow::bail!("unused environment update")
        }
    }

    impl ChatFlowLearningState {
        async fn apply_learning_event(&self, event: &LearningLedgerEvent) {
            let mut records = self.learning_records.lock().await;
            match event.event_kind {
                LearningEventKind::CandidateCaptured => {
                    if records
                        .iter()
                        .any(|record| record.record_id == event.record_id)
                    {
                        return;
                    }
                    records.push(LearningRecordProjection {
                        record_id: event.record_id,
                        summary: event.summary.clone(),
                        accepted_summary: None,
                        review_state: LearningReviewState::Candidate,
                        redaction_status: event.redaction_status,
                        provenance: event.provenance.clone(),
                        capture_metadata: event.capture_metadata.clone(),
                        created_at: event.created_at,
                        updated_at: event.created_at,
                        schema_version: event.schema_version,
                    });
                }
                LearningEventKind::ReviewAccepted => {
                    if let Some(record) = records
                        .iter_mut()
                        .find(|record| record.record_id == event.record_id)
                    {
                        record.review_state = LearningReviewState::Accepted;
                        record.updated_at = event.created_at;
                    }
                }
                LearningEventKind::ReviewRejected => {
                    if let Some(record) = records
                        .iter_mut()
                        .find(|record| record.record_id == event.record_id)
                    {
                        record.review_state = LearningReviewState::Rejected;
                        record.updated_at = event.created_at;
                    }
                }
                LearningEventKind::SensorLessonProposed
                | LearningEventKind::SensorReviewPending
                | LearningEventKind::SensorReviewRejected
                | LearningEventKind::PromotionAudit => {
                    if let Some(record) = records
                        .iter_mut()
                        .find(|record| record.record_id == event.record_id)
                    {
                        record.updated_at = event.created_at;
                    }
                }
                LearningEventKind::Superseded => {
                    if let Some(record) = records
                        .iter_mut()
                        .find(|record| record.record_id == event.record_id)
                    {
                        record.review_state = LearningReviewState::Superseded;
                        record.updated_at = event.created_at;
                    }
                }
            }
        }
    }

    #[async_trait::async_trait]
    impl ConversationService for ChatFlowLearningState {
        async fn find_conversation(&self, id: &ConversationId) -> Result<Option<Conversation>> {
            Ok(self.conversations.lock().await.get(id).cloned())
        }

        async fn upsert_conversation(&self, conversation: Conversation) -> Result<()> {
            self.conversations
                .lock()
                .await
                .insert(conversation.id, conversation);
            Ok(())
        }

        async fn ensure_delegated_conversation(
            &self,
            id: &ConversationId,
            parent_id: Option<ConversationId>,
        ) -> Result<Conversation> {
            let mut conversations = self.conversations.lock().await;
            let conversation = conversations
                .get_mut(id)
                .ok_or_else(|| forge_domain::Error::ConversationNotFound(*id))?;
            conversation.ensure_delegated(parent_id);
            Ok(conversation.clone())
        }

        async fn resolve_root_conversation_id(
            &self,
            parent_id: Option<ConversationId>,
        ) -> Result<Option<ConversationId>> {
            Ok(parent_id)
        }

        async fn list_branch_targets(
            &self,
            conversation_id: &ConversationId,
        ) -> Result<Vec<crate::dto::ConversationBranchTarget>> {
            let conversations = self.conversations.lock().await;
            let source = conversations
                .get(conversation_id)
                .ok_or_else(|| forge_domain::Error::ConversationNotFound(*conversation_id))?;
            let mut context = source
                .context
                .clone()
                .ok_or_else(|| anyhow::anyhow!("Conversation {conversation_id} has no context"))?;
            context.conversation_id = Some(source.id);
            Ok(crate::dto::ConversationBranchTarget::list_from_context(
                source.id, &context,
            ))
        }

        async fn modify_conversation<F, T>(&self, id: &ConversationId, f: F) -> Result<T>
        where
            F: FnOnce(&mut Conversation) -> T + Send,
            T: Send,
        {
            let mut conversations = self.conversations.lock().await;
            let conversation = conversations
                .get_mut(id)
                .ok_or_else(|| forge_domain::Error::ConversationNotFound(*id))?;
            Ok(f(conversation))
        }

        async fn try_modify_conversation<F, T>(&self, id: &ConversationId, f: F) -> Result<T>
        where
            F: FnOnce(&mut Conversation) -> Result<T> + Send,
            T: Send,
        {
            let mut conversations = self.conversations.lock().await;
            let conversation = conversations
                .get_mut(id)
                .ok_or_else(|| forge_domain::Error::ConversationNotFound(*id))?;
            f(conversation)
        }

        async fn branch_conversation(
            &self,
            _conversation_id: &ConversationId,
            _target_id: forge_domain::MessageId,
        ) -> Result<Conversation> {
            anyhow::bail!("unused branch conversation")
        }

        async fn get_conversation_list_items_by_query(
            &self,
            _query: forge_domain::ConversationListQuery,
        ) -> Result<Vec<ConversationListItem>> {
            Ok(Vec::new())
        }

        async fn get_conversation_list_items_including_agent(
            &self,
            _limit: usize,
        ) -> Result<Vec<ConversationListItem>> {
            Ok(Vec::new())
        }

        async fn get_conversation_list_items_by_visibility(
            &self,
            _visibility: forge_domain::ConversationVisibilityFilter,
            _limit: usize,
        ) -> Result<Vec<ConversationListItem>> {
            Ok(Vec::new())
        }

        async fn get_conversations(&self) -> Result<Vec<Conversation>> {
            Ok(self.conversations.lock().await.values().cloned().collect())
        }

        async fn get_conversations_including_agent(&self) -> Result<Vec<Conversation>> {
            self.get_conversations().await
        }

        async fn get_conversations_by_visibility(
            &self,
            visibility: forge_domain::ConversationVisibilityFilter,
        ) -> Result<Vec<Conversation>> {
            let conversations = self.get_conversations_including_agent().await?;
            Ok(conversations
                .into_iter()
                .filter(|conversation| match visibility {
                    forge_domain::ConversationVisibilityFilter::Normal => {
                        conversation.is_normal_visibility()
                    }
                    forge_domain::ConversationVisibilityFilter::Background => {
                        conversation.is_background()
                    }
                    forge_domain::ConversationVisibilityFilter::All => true,
                })
                .collect())
        }

        async fn get_sub_conversations(
            &self,
            parent_id: &ConversationId,
        ) -> Result<Vec<Conversation>> {
            Ok(self
                .conversations
                .lock()
                .await
                .values()
                .filter(|conversation| conversation.parent_id == Some(*parent_id))
                .cloned()
                .collect())
        }

        async fn upsert_subagent_task_session(
            &self,
            _session: forge_domain::SubagentTaskSession,
        ) -> Result<()> {
            Ok(())
        }

        async fn get_subagent_task_session(
            &self,
            _task_id: &forge_domain::SubagentTaskId,
        ) -> Result<Option<forge_domain::SubagentTaskSession>> {
            Ok(None)
        }

        async fn get_subagent_task_session_by_conversation(
            &self,
            _conversation_id: &ConversationId,
        ) -> Result<Option<forge_domain::SubagentTaskSession>> {
            Ok(None)
        }

        async fn list_subagent_task_sessions(
            &self,
            _filter: forge_domain::SubagentTaskSessionFilter,
        ) -> Result<Vec<forge_domain::SubagentTaskSession>> {
            Ok(Vec::new())
        }

        async fn last_conversation(&self) -> Result<Option<Conversation>> {
            Ok(self.conversations.lock().await.values().next().cloned())
        }

        async fn delete_conversation(&self, conversation_id: &ConversationId) -> Result<()> {
            self.conversations.lock().await.remove(conversation_id);
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl LearningService for ChatFlowLearningState {
        async fn capture_candidate_from_conversation(
            &self,
            conversation_id: ConversationId,
            source_event_id: String,
            summary: String,
            metadata: LearningCaptureMetadata,
        ) -> Result<LearningLedgerAppendOutcome> {
            let mut event = LearningLedgerEvent::capture_candidate(
                summary,
                LearningProvenance::conversation(
                    conversation_id,
                    source_event_id,
                    metadata.context_fingerprint.clone(),
                ),
                chrono::Utc::now(),
            )?;
            event.capture_metadata = Some(metadata);
            let mut events = self.learning_events.lock().await;
            let key = event.idempotency_key.clone();
            let outcome = if let Some(existing) = events
                .iter()
                .find(|existing| existing.idempotency_key == key)
                .cloned()
            {
                LearningLedgerAppendOutcome::existing(existing)
            } else {
                events.push(event.clone());
                LearningLedgerAppendOutcome::inserted(event)
            };
            let event = outcome.event.clone();
            drop(events);
            self.apply_learning_event(&event).await;
            Ok(outcome)
        }

        async fn insert_learning_event(
            &self,
            event: LearningLedgerEvent,
        ) -> Result<LearningLedgerAppendOutcome> {
            let mut events = self.learning_events.lock().await;
            let key = event.idempotency_key.clone();
            let outcome = if let Some(existing) = events
                .iter()
                .find(|existing| existing.idempotency_key == key)
                .cloned()
            {
                LearningLedgerAppendOutcome::existing(existing)
            } else {
                events.push(event.clone());
                LearningLedgerAppendOutcome::inserted(event)
            };
            let event = outcome.event.clone();
            drop(events);
            self.apply_learning_event(&event).await;
            Ok(outcome)
        }

        async fn review_learning_candidate_event(
            &self,
            event: LearningLedgerEvent,
        ) -> Result<LearningReviewOutcome> {
            let target_state = match event.event_kind {
                LearningEventKind::ReviewAccepted => LearningReviewState::Accepted,
                LearningEventKind::ReviewRejected => LearningReviewState::Rejected,
                LearningEventKind::Superseded => LearningReviewState::Superseded,
                LearningEventKind::CandidateCaptured
                | LearningEventKind::SensorLessonProposed
                | LearningEventKind::SensorReviewPending
                | LearningEventKind::SensorReviewRejected
                | LearningEventKind::PromotionAudit => {
                    anyhow::bail!(
                        "event kind {} cannot review learning record",
                        event.event_kind
                    )
                }
            };
            let before = self
                .learning_records
                .lock()
                .await
                .iter()
                .find(|record| record.record_id == event.record_id)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("learning candidate record not found"))?;
            if before.review_state == target_state {
                let existing = self
                    .learning_events
                    .lock()
                    .await
                    .iter()
                    .find(|existing| {
                        existing.record_id == event.record_id
                            && existing.event_kind == event.event_kind
                    })
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("learning review event not found"))?;
                return Ok(LearningReviewOutcome { event: existing, projection: before });
            }
            if before.review_state != LearningReviewState::Candidate {
                anyhow::bail!(
                    "learning record cannot be reviewed from state {}",
                    before.review_state
                );
            }
            let event = self.insert_learning_event(event).await?.event;
            let projection = self
                .learning_records
                .lock()
                .await
                .iter()
                .find(|record| record.record_id == event.record_id)
                .cloned()
                .ok_or_else(|| {
                    anyhow::anyhow!("learning review projection not found after append")
                })?;
            Ok(LearningReviewOutcome { event, projection })
        }

        async fn get_learning_record(
            &self,
            record_id: LearningRecordId,
        ) -> Result<Option<LearningRecordProjection>> {
            Ok(self
                .learning_records
                .lock()
                .await
                .iter()
                .find(|record| record.record_id == record_id)
                .cloned())
        }

        async fn list_learning_records(
            &self,
            review_state: Option<LearningReviewState>,
            limit: usize,
        ) -> Result<Vec<LearningRecordProjection>> {
            let mut records = self.learning_records.lock().await.clone();
            if let Some(review_state) = review_state {
                records.retain(|record| record.review_state == review_state);
            }
            records.truncate(limit);
            Ok(records)
        }

        async fn learning_freshness(
            &self,
            _review_state: Option<LearningReviewState>,
        ) -> Result<LearningLedgerFreshness> {
            Ok(LearningLedgerFreshness {
                ledger_cursor: self.learning_records.lock().await.len() as i64,
                projection_version: 1,
                review_state_fingerprint: "runtime-proof-learning".to_string(),
            })
        }

        async fn promote_sensor_lesson(
            &self,
            _request: SensorLessonPromotionRequest,
        ) -> Result<SensorLessonPromotionOutcome> {
            anyhow::bail!("unused learning promotion")
        }
    }

    #[async_trait::async_trait]
    impl ProviderService for ChatFlowLearningState {
        async fn chat(
            &self,
            _model_id: &ModelId,
            context: Context,
            _provider: Provider<Url>,
        ) -> ResultStream<ChatCompletionMessage, anyhow::Error> {
            *self.captured_provider_context.lock().await = Some(context);
            let message = ChatCompletionMessage::assistant(Content::full("runtime proof response"))
                .finish_reason(FinishReason::Stop);
            Ok(Box::pin(tokio_stream::iter(std::iter::once(Ok(message)))))
        }

        async fn models(&self, _provider: Provider<Url>) -> Result<Vec<Model>> {
            Ok(vec![self.model.clone()])
        }

        async fn get_provider(&self, id: ProviderId) -> Result<Provider<Url>> {
            assert_eq!(id, self.provider.id);
            Ok(self.provider.clone())
        }

        async fn get_all_providers(&self) -> Result<Vec<AnyProvider>> {
            Ok(vec![AnyProvider::Url(self.provider.clone())])
        }

        async fn upsert_credential(&self, _credential: forge_domain::AuthCredential) -> Result<()> {
            anyhow::bail!("unused credential upsert")
        }

        async fn remove_credential(&self, _id: &ProviderId) -> Result<()> {
            anyhow::bail!("unused credential removal")
        }

        async fn migrate_env_credentials(&self) -> Result<Option<forge_domain::MigrationResult>> {
            Ok(None)
        }
    }

    #[async_trait::async_trait]
    impl AppConfigService for ChatFlowLearningState {
        async fn get_session_config(&self) -> Option<forge_domain::ModelConfig> {
            Some(forge_domain::ModelConfig::new(
                self.provider.id.clone(),
                self.model.id.clone(),
            ))
        }

        async fn get_commit_config(&self) -> Result<Option<forge_domain::ModelConfig>> {
            Ok(None)
        }

        async fn get_suggest_config(&self) -> Result<Option<forge_domain::ModelConfig>> {
            Ok(None)
        }

        async fn get_reasoning_effort(&self) -> Result<Option<forge_domain::Effort>> {
            Ok(None)
        }

        async fn update_config(&self, _ops: Vec<forge_domain::ConfigOperation>) -> Result<()> {
            anyhow::bail!("unused config update")
        }
    }

    #[async_trait::async_trait]
    impl AgentRegistry for ChatFlowLearningState {
        async fn get_active_agent_id(&self) -> Result<Option<AgentId>> {
            Ok(Some(self.agent.id.clone()))
        }

        async fn set_active_agent_id(&self, _agent_id: AgentId) -> Result<()> {
            anyhow::bail!("unused active agent update")
        }

        async fn get_agents(&self) -> Result<Vec<Agent>> {
            Ok(vec![self.agent.clone()])
        }

        async fn get_agent_infos(&self) -> Result<Vec<forge_domain::AgentInfo>> {
            Ok(Vec::new())
        }

        async fn get_agent(&self, agent_id: &AgentId) -> Result<Option<Agent>> {
            Ok((agent_id == &self.agent.id).then(|| self.agent.clone()))
        }

        async fn reload_agents(&self) -> Result<()> {
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl ProviderAuthService for ChatFlowLearningState {
        async fn init_provider_auth(
            &self,
            _provider_id: ProviderId,
            _method: AuthMethod,
        ) -> Result<AuthContextRequest> {
            anyhow::bail!("unused provider auth init")
        }

        async fn complete_provider_auth(
            &self,
            _provider_id: ProviderId,
            _context: AuthContextResponse,
            _timeout: std::time::Duration,
        ) -> Result<()> {
            anyhow::bail!("unused provider auth completion")
        }

        async fn refresh_provider_credential(
            &self,
            provider: Provider<Url>,
        ) -> Result<Provider<Url>> {
            Ok(provider)
        }
    }

    #[async_trait::async_trait]
    impl CustomInstructionsService for ChatFlowLearningState {
        async fn get_custom_instructions(&self) -> Vec<String> {
            Vec::new()
        }
    }

    #[async_trait::async_trait]
    impl McpService for ChatFlowLearningState {
        async fn get_mcp_servers(&self) -> Result<McpServers> {
            Ok(McpServers::default())
        }

        async fn execute_mcp(&self, _call: ToolCallFull) -> Result<ToolOutput> {
            anyhow::bail!("unused mcp execution")
        }

        async fn reload_mcp(&self) -> Result<()> {
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl WorkspaceService for ChatFlowLearningState {
        async fn sync_workspace(
            &self,
            _path: PathBuf,
        ) -> Result<forge_stream::MpscStream<Result<SyncProgress>>> {
            anyhow::bail!("unused workspace sync")
        }

        async fn produce_workspace_exact_fact_reference(
            &self,
            _path: PathBuf,
        ) -> Result<WorkspaceExactFactReferenceReport> {
            anyhow::bail!("unused workspace exact-fact reference")
        }

        async fn workspace_exact_fact_status(
            &self,
            _path: PathBuf,
        ) -> Result<WorkspaceExactFactStatusReport> {
            anyhow::bail!("unused workspace exact-fact status")
        }

        async fn workspace_evidence_replay_diagnostic(
            &self,
            _path: PathBuf,
        ) -> Result<WorkspaceEvidenceReplayDiagnostic> {
            anyhow::bail!("unused workspace evidence replay diagnostic")
        }

        async fn workspace_evidence_replay_preview_diagnostic(
            &self,
            _path: PathBuf,
        ) -> Result<WorkspaceEvidenceReplayPreviewDiagnostic> {
            anyhow::bail!("unused workspace evidence replay preview diagnostic")
        }

        async fn build_workspace_vector_index(
            &self,
            _path: PathBuf,
            _embedding_model_id: String,
        ) -> Result<WorkspaceVectorIndexBuildReport> {
            anyhow::bail!("unused workspace vector index build")
        }

        async fn embed_workspace_query(
            &self,
            _query: String,
            _embedding_model_id: String,
        ) -> Result<ProjectSemanticEmbeddingOutput> {
            anyhow::bail!("unused workspace query embedding")
        }

        async fn semantic_injection_readiness(
            &self,
            _path: PathBuf,
            _embedding_model_id: Option<String>,
        ) -> Result<WorkspaceSemanticInjectionReadiness> {
            Ok(WorkspaceSemanticInjectionReadiness::VectorIndexAbsentOrNoMatch)
        }

        async fn sem_search_availability(
            &self,
            _path: PathBuf,
            _embedding_model_id: Option<String>,
        ) -> Result<SemSearchAvailability> {
            Ok(SemSearchAvailability::Unsupported {
                reason: SemSearchUnsupportedReason::VectorArtifactAbsentOrNoMatch,
            })
        }

        async fn sem_search_diagnostic(
            &self,
            path: PathBuf,
            embedding_model_id: Option<String>,
        ) -> Result<SemSearchDiagnosticReport> {
            let availability = self
                .sem_search_availability(path.clone(), embedding_model_id.clone())
                .await?;
            Ok(SemSearchDiagnosticReport::from_availability(
                &availability,
                embedding_model_id.as_deref(),
                &path,
            ))
        }

        async fn project_context_reranker_diagnostic(
            &self,
        ) -> Result<WorkspaceRerankRuntimeDiagnostic> {
            Ok(WorkspaceRerankRuntimeDiagnostic::missing_config())
        }

        async fn query_workspace_committed(
            &self,
            _path: PathBuf,
            _params: SearchParams<'_>,
        ) -> Result<(ProjectContextCommittedQueryResult, Vec<Node>)> {
            Ok((
                ProjectContextCommittedQueryResult::no_write(
                    Default::default(),
                    ProjectContextPackNoWriteReason::EmptyEvidence,
                    Vec::new(),
                ),
                Vec::new(),
            ))
        }

        async fn query_workspace(
            &self,
            _path: PathBuf,
            _params: SearchParams<'_>,
        ) -> Result<Vec<Node>> {
            Ok(Vec::new())
        }

        async fn list_workspaces(&self) -> Result<Vec<WorkspaceInfo>> {
            Ok(Vec::new())
        }

        async fn get_workspace_info(&self, _path: PathBuf) -> Result<Option<WorkspaceInfo>> {
            Ok(None)
        }

        async fn is_indexed(&self, _path: &Path) -> Result<bool> {
            Ok(false)
        }

        async fn delete_workspace(&self, _workspace_id: &WorkspaceId) -> Result<()> {
            Ok(())
        }

        async fn delete_workspaces(&self, _workspace_ids: &[WorkspaceId]) -> Result<()> {
            Ok(())
        }

        async fn project_model_context_diagnostic(
            &self,
            path: &Path,
        ) -> Result<WorkspaceContextManifestDiagnostic> {
            Ok(WorkspaceContextManifestDiagnostic {
                workspace_root: path.to_path_buf(),
                manifest_found: false,
                manifest_path: path.join(".forge_project_model/project_manifest.json"),
                freshness: WorkspaceContextFreshness::Unknown {
                    reason: "runtime proof does not index workspace".to_string(),
                },
                manifest_hash: None,
                exact_fact_readiness: None,
                evidence_readiness: None,
                evidence_ledger_activation: None,
            })
        }

        async fn get_workspace_status(&self, _path: PathBuf) -> Result<Vec<FileStatus>> {
            Ok(Vec::new())
        }

        async fn is_authenticated(&self) -> Result<bool> {
            Ok(false)
        }

        async fn init_auth_credentials(&self) -> Result<WorkspaceAuth> {
            anyhow::bail!("unused workspace auth")
        }

        async fn init_workspace(&self, _path: PathBuf) -> Result<WorkspaceId> {
            anyhow::bail!("unused workspace init")
        }
    }

    #[async_trait::async_trait]
    impl SteerService for ChatFlowLearningState {
        async fn enqueue_steer(
            &self,
            _conversation_id: &ConversationId,
            _message: SteerMessage,
        ) -> Result<()> {
            Ok(())
        }

        async fn clear_steer(&self, _conversation_id: &ConversationId) -> Result<()> {
            Ok(())
        }

        async fn drain_steer(
            &self,
            _conversation_id: &ConversationId,
        ) -> Result<Vec<SteerMessage>> {
            Ok(Vec::new())
        }
    }

    macro_rules! impl_unused_service_traits {
        ($type:ty) => {
            #[async_trait::async_trait]
            impl TemplateService for $type {
                async fn register_template(&self, _path: PathBuf) -> Result<()> {
                    anyhow::bail!("unused template registration")
                }

                async fn render_template<V: serde::Serialize + Send + Sync>(
                    &self,
                    _template: forge_domain::Template<V>,
                    _object: &V,
                ) -> Result<String> {
                    anyhow::bail!("unused template rendering")
                }
            }

            #[async_trait::async_trait]
            impl AttachmentService for $type {
                async fn attachments(&self, _url: &str) -> Result<Vec<forge_domain::Attachment>> {
                    Ok(Vec::new())
                }
            }

            #[async_trait::async_trait]
            impl FileDiscoveryService for $type {
                async fn collect_files(&self, _config: Walker) -> Result<Vec<forge_domain::File>> {
                    Ok(Vec::new())
                }

                async fn list_current_directory(&self) -> Result<Vec<forge_domain::File>> {
                    Ok(Vec::new())
                }
            }

            #[async_trait::async_trait]
            impl McpConfigManager for $type {
                async fn read_mcp_config(&self, _scope: Option<&Scope>) -> Result<McpConfig> {
                    anyhow::bail!("unused mcp config read")
                }

                async fn write_mcp_config(
                    &self,
                    _config: &McpConfig,
                    _scope: &Scope,
                ) -> Result<()> {
                    anyhow::bail!("unused mcp config write")
                }
            }

            #[async_trait::async_trait]
            impl FsWriteService for $type {
                async fn write(
                    &self,
                    _path: String,
                    _content: String,
                    _overwrite: bool,
                ) -> Result<crate::FsWriteOutput> {
                    anyhow::bail!("unused fs write")
                }
            }

            #[async_trait::async_trait]
            impl PlanCreateService for $type {
                async fn create_plan(
                    &self,
                    _plan_name: String,
                    _version: String,
                    _content: String,
                ) -> Result<crate::PlanCreateOutput> {
                    anyhow::bail!("unused plan create")
                }
            }

            #[async_trait::async_trait]
            impl FsPatchService for $type {
                async fn patch(
                    &self,
                    _path: String,
                    _search: String,
                    _content: String,
                    _replace_all: bool,
                ) -> Result<crate::PatchOutput> {
                    anyhow::bail!("unused fs patch")
                }

                async fn multi_patch(
                    &self,
                    _path: String,
                    _edits: Vec<forge_domain::PatchEdit>,
                ) -> Result<crate::PatchOutput> {
                    anyhow::bail!("unused fs multi patch")
                }

                async fn apply_patch(&self, _patch: String) -> Result<crate::ApplyPatchOutput> {
                    anyhow::bail!("unused apply patch")
                }
            }

            #[async_trait::async_trait]
            impl FsReadService for $type {
                async fn read(
                    &self,
                    _path: String,
                    _start_line: Option<u64>,
                    _end_line: Option<u64>,
                ) -> Result<crate::ReadOutput> {
                    anyhow::bail!("unused fs read")
                }
            }

            #[async_trait::async_trait]
            impl ImageReadService for $type {
                async fn read_image(&self, _path: String) -> Result<forge_domain::Image> {
                    anyhow::bail!("unused image read")
                }
            }

            #[async_trait::async_trait]
            impl FsRemoveService for $type {
                async fn remove(&self, _path: String) -> Result<crate::FsRemoveOutput> {
                    anyhow::bail!("unused fs remove")
                }
            }

            #[async_trait::async_trait]
            impl FsSearchService for $type {
                async fn search(
                    &self,
                    _params: forge_domain::FSSearch,
                ) -> Result<Option<crate::SearchResult>> {
                    Ok(None)
                }
            }

            #[async_trait::async_trait]
            impl FollowUpService for $type {
                async fn follow_up(
                    &self,
                    _question: String,
                    _options: Vec<String>,
                    _multiple: Option<bool>,
                ) -> Result<Option<String>> {
                    Ok(None)
                }
            }

            #[async_trait::async_trait]
            impl FsUndoService for $type {
                async fn undo(&self, _path: String) -> Result<crate::FsUndoOutput> {
                    anyhow::bail!("unused fs undo")
                }
            }

            #[async_trait::async_trait]
            impl NetFetchService for $type {
                async fn fetch(
                    &self,
                    _url: String,
                    _raw: Option<bool>,
                ) -> Result<crate::HttpResponse> {
                    anyhow::bail!("unused net fetch")
                }
            }

            #[async_trait::async_trait]
            impl ShellService for $type {
                async fn execute(
                    &self,
                    _request: crate::ShellExecuteRequest,
                ) -> Result<crate::ShellOutput> {
                    anyhow::bail!("unused shell execute")
                }
            }

            #[async_trait::async_trait]
            impl AuthService for $type {
                async fn user_info(&self, _api_key: &str) -> Result<User> {
                    anyhow::bail!("unused auth user info")
                }

                async fn user_usage(&self, _api_key: &str) -> Result<UserUsage> {
                    anyhow::bail!("unused auth user usage")
                }
            }

            #[async_trait::async_trait]
            impl CommandLoaderService for $type {
                async fn get_commands(&self) -> Result<Vec<forge_domain::Command>> {
                    Ok(Vec::new())
                }
            }

            #[async_trait::async_trait]
            impl PolicyService for $type {
                async fn check_operation_permission(
                    &self,
                    _operation: &PermissionOperation,
                ) -> Result<crate::PolicyDecision> {
                    Ok(crate::PolicyDecision { allowed: true, path: None })
                }
            }

            #[async_trait::async_trait]
            impl SkillFetchService for $type {
                async fn fetch_skill(&self, _skill_name: String) -> Result<forge_domain::Skill> {
                    anyhow::bail!("unused skill fetch")
                }

                async fn list_skills(&self) -> Result<Vec<forge_domain::Skill>> {
                    Ok(Vec::new())
                }
            }
        };
    }

    impl_unused_service_traits!(ChatFlowLearningState);

    impl Services for ChatFlowLearningHarness {
        type ProviderService = ChatFlowLearningState;
        type AppConfigService = ChatFlowLearningState;
        type ConversationService = ChatFlowLearningState;
        type LearningService = ChatFlowLearningState;
        type SteerService = ChatFlowLearningState;
        type TemplateService = ChatFlowLearningState;
        type AttachmentService = ChatFlowLearningState;
        type CustomInstructionsService = ChatFlowLearningState;
        type FileDiscoveryService = ChatFlowLearningState;
        type McpConfigManager = ChatFlowLearningState;
        type FsWriteService = ChatFlowLearningState;
        type PlanCreateService = ChatFlowLearningState;
        type FsPatchService = ChatFlowLearningState;
        type FsReadService = ChatFlowLearningState;
        type ImageReadService = ChatFlowLearningState;
        type FsRemoveService = ChatFlowLearningState;
        type FsSearchService = ChatFlowLearningState;
        type FollowUpService = ChatFlowLearningState;
        type FsUndoService = ChatFlowLearningState;
        type NetFetchService = ChatFlowLearningState;
        type ShellService = ChatFlowLearningState;
        type McpService = ChatFlowLearningState;
        type AuthService = ChatFlowLearningState;
        type AgentRegistry = ChatFlowLearningState;
        type CommandLoaderService = ChatFlowLearningState;
        type PolicyService = ChatFlowLearningState;
        type ProviderAuthService = ChatFlowLearningState;
        type WorkspaceService = ChatFlowLearningState;
        type SkillFetchService = ChatFlowLearningState;

        fn provider_service(&self) -> &Self::ProviderService {
            &self.state
        }
        fn config_service(&self) -> &Self::AppConfigService {
            &self.state
        }
        fn conversation_service(&self) -> &Self::ConversationService {
            &self.state
        }
        fn learning_service(&self) -> &Self::LearningService {
            &self.state
        }
        fn steer_service(&self) -> &Self::SteerService {
            &self.state
        }
        fn template_service(&self) -> &Self::TemplateService {
            &self.state
        }
        fn attachment_service(&self) -> &Self::AttachmentService {
            &self.state
        }
        fn file_discovery_service(&self) -> &Self::FileDiscoveryService {
            &self.state
        }
        fn mcp_config_manager(&self) -> &Self::McpConfigManager {
            &self.state
        }
        fn fs_create_service(&self) -> &Self::FsWriteService {
            &self.state
        }
        fn plan_create_service(&self) -> &Self::PlanCreateService {
            &self.state
        }
        fn fs_patch_service(&self) -> &Self::FsPatchService {
            &self.state
        }
        fn fs_read_service(&self) -> &Self::FsReadService {
            &self.state
        }
        fn image_read_service(&self) -> &Self::ImageReadService {
            &self.state
        }
        fn fs_remove_service(&self) -> &Self::FsRemoveService {
            &self.state
        }
        fn fs_search_service(&self) -> &Self::FsSearchService {
            &self.state
        }
        fn follow_up_service(&self) -> &Self::FollowUpService {
            &self.state
        }
        fn fs_undo_service(&self) -> &Self::FsUndoService {
            &self.state
        }
        fn net_fetch_service(&self) -> &Self::NetFetchService {
            &self.state
        }
        fn shell_service(&self) -> &Self::ShellService {
            &self.state
        }
        fn mcp_service(&self) -> &Self::McpService {
            &self.state
        }
        fn custom_instructions_service(&self) -> &Self::CustomInstructionsService {
            &self.state
        }
        fn auth_service(&self) -> &Self::AuthService {
            &self.state
        }
        fn agent_registry(&self) -> &Self::AgentRegistry {
            &self.state
        }
        fn command_loader_service(&self) -> &Self::CommandLoaderService {
            &self.state
        }
        fn policy_service(&self) -> &Self::PolicyService {
            &self.state
        }
        fn provider_auth_service(&self) -> &Self::ProviderAuthService {
            &self.state
        }
        fn workspace_service(&self) -> &Self::WorkspaceService {
            &self.state
        }
        fn skill_fetch_service(&self) -> &Self::SkillFetchService {
            &self.state
        }
    }

    struct ProjectContextHarness {
        cwd: PathBuf,
        empty_paths: Vec<PathBuf>,
        error_paths: Vec<PathBuf>,
        stale_paths: Vec<PathBuf>,
        unknown_paths: Vec<PathBuf>,
        inactive_exact_fact_paths: Vec<PathBuf>,
        #[allow(dead_code)]
        replay_preview_empty_paths: Vec<PathBuf>,
        captured_context: Mutex<Option<Context>>,
        workspace_queries: AtomicUsize,
        committed_workspace_queries: AtomicUsize,
        committed_result: Mutex<Option<ProjectContextCommittedQueryResult>>,
        queried_workspaces: Mutex<Vec<PathBuf>>,
        query_filters: Mutex<Vec<Option<String>>>,
        query_embeddings: Mutex<Vec<Option<Vec<f32>>>>,
        query_rerank_sources: Mutex<Vec<forge_domain::SearchRerankIntentSource>>,
        index_checks: AtomicUsize,
        learning_records: Mutex<Vec<LearningRecordProjection>>,
        learning_freshness: LearningLedgerFreshness,
        config: ForgeConfig,
        semantic_model_disabled: AtomicBool,
        semantic_readiness: Mutex<HashMap<PathBuf, WorkspaceSemanticInjectionReadiness>>,
        embedding_calls: AtomicUsize,
        embedding_inputs: Mutex<Vec<String>>,
        embedding_failure: AtomicBool,
        embedding_pending: AtomicBool,
        embedding_dimension: AtomicUsize,
        rerank_runtime: Mutex<WorkspaceRerankRuntimeDiagnostic>,
        rerank_runtime_diagnostic_calls: AtomicUsize,
    }

    impl ProjectContextHarness {
        fn new(cwd: PathBuf) -> Arc<Self> {
            Self::new_with_empty_error_stale_unknown_inactive_exact_fact_and_replay_preview_empty_paths(
                cwd,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        }

        fn new_with_empty_paths(cwd: PathBuf, empty_paths: Vec<PathBuf>) -> Arc<Self> {
            Self::new_with_empty_error_stale_unknown_inactive_exact_fact_and_replay_preview_empty_paths(
                cwd,
                empty_paths,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        }

        fn new_with_error_paths(cwd: PathBuf, error_paths: Vec<PathBuf>) -> Arc<Self> {
            Self::new_with_empty_error_stale_unknown_inactive_exact_fact_and_replay_preview_empty_paths(
                cwd,
                Vec::new(),
                error_paths,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        }

        fn new_with_stale_paths(cwd: PathBuf, stale_paths: Vec<PathBuf>) -> Arc<Self> {
            Self::new_with_empty_error_stale_unknown_inactive_exact_fact_and_replay_preview_empty_paths(
                cwd,
                Vec::new(),
                Vec::new(),
                stale_paths,
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        }

        fn new_with_unknown_paths(cwd: PathBuf, unknown_paths: Vec<PathBuf>) -> Arc<Self> {
            Self::new_with_empty_error_stale_unknown_inactive_exact_fact_and_replay_preview_empty_paths(
                cwd,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                unknown_paths,
                Vec::new(),
                Vec::new(),
            )
        }

        fn new_with_inactive_exact_fact_paths(
            cwd: PathBuf,
            inactive_exact_fact_paths: Vec<PathBuf>,
        ) -> Arc<Self> {
            Self::new_with_empty_error_stale_unknown_inactive_exact_fact_and_replay_preview_empty_paths(
                cwd,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                inactive_exact_fact_paths,
                Vec::new(),
            )
        }

        #[allow(dead_code)]
        fn new_with_replay_preview_empty_paths(
            cwd: PathBuf,
            replay_preview_empty_paths: Vec<PathBuf>,
        ) -> Arc<Self> {
            Self::new_with_empty_error_stale_unknown_inactive_exact_fact_and_replay_preview_empty_paths(
                cwd,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                replay_preview_empty_paths,
            )
        }

        fn new_with_empty_error_stale_unknown_inactive_exact_fact_and_replay_preview_empty_paths(
            cwd: PathBuf,
            empty_paths: Vec<PathBuf>,
            error_paths: Vec<PathBuf>,
            stale_paths: Vec<PathBuf>,
            unknown_paths: Vec<PathBuf>,
            inactive_exact_fact_paths: Vec<PathBuf>,
            replay_preview_empty_paths: Vec<PathBuf>,
        ) -> Arc<Self> {
            let config = ForgeConfig {
                semantic_embedding_model_id: Some("fixture-embedding-model".to_string()),
                ..Default::default()
            };
            Arc::new(Self {
                cwd,
                empty_paths,
                error_paths,
                stale_paths,
                unknown_paths,
                inactive_exact_fact_paths,
                replay_preview_empty_paths,
                captured_context: Mutex::new(None),
                workspace_queries: AtomicUsize::new(0),
                committed_workspace_queries: AtomicUsize::new(0),
                committed_result: Mutex::new(None),
                queried_workspaces: Mutex::new(Vec::new()),
                query_filters: Mutex::new(Vec::new()),
                query_embeddings: Mutex::new(Vec::new()),
                query_rerank_sources: Mutex::new(Vec::new()),
                index_checks: AtomicUsize::new(0),
                learning_records: Mutex::new(Vec::new()),
                learning_freshness: LearningLedgerFreshness {
                    ledger_cursor: 1,
                    projection_version: 1,
                    review_state_fingerprint: "fixture-learning".to_string(),
                },
                config,
                semantic_model_disabled: AtomicBool::new(false),
                semantic_readiness: Mutex::new(HashMap::new()),
                embedding_calls: AtomicUsize::new(0),
                embedding_inputs: Mutex::new(Vec::new()),
                embedding_failure: AtomicBool::new(false),
                embedding_pending: AtomicBool::new(false),
                embedding_dimension: AtomicUsize::new(2),
                rerank_runtime: Mutex::new(WorkspaceRerankRuntimeDiagnostic::missing_config()),
                rerank_runtime_diagnostic_calls: AtomicUsize::new(0),
            })
        }

        async fn set_learning_records(&self, records: Vec<LearningRecordProjection>) {
            *self.learning_records.lock().await = records;
        }

        async fn set_semantic_readiness(
            &self,
            path: PathBuf,
            readiness: WorkspaceSemanticInjectionReadiness,
        ) {
            let path = fs::canonicalize(&path).unwrap_or(path);
            self.semantic_readiness.lock().await.insert(path, readiness);
        }

        async fn set_rerank_runtime(&self, diagnostic: WorkspaceRerankRuntimeDiagnostic) {
            *self.rerank_runtime.lock().await = diagnostic;
        }

        fn rerank_runtime_diagnostic_calls(&self) -> usize {
            self.rerank_runtime_diagnostic_calls.load(Ordering::SeqCst)
        }

        fn disable_semantic_embedding_model(&self) {
            self.semantic_model_disabled.store(true, Ordering::SeqCst);
        }

        fn fail_embedding(&self) {
            self.embedding_failure.store(true, Ordering::SeqCst);
        }

        fn pending_embedding(&self) {
            self.embedding_pending.store(true, Ordering::SeqCst);
        }

        fn set_embedding_dimension(&self, dimension: usize) {
            self.embedding_dimension.store(dimension, Ordering::SeqCst);
        }

        async fn set_committed_result(&self, result: ProjectContextCommittedQueryResult) {
            *self.committed_result.lock().await = Some(result);
        }
    }

    impl EnvironmentInfra for ProjectContextHarness {
        type Config = ForgeConfig;

        fn get_env_var(&self, _key: &str) -> Option<String> {
            None
        }

        fn get_env_vars(&self) -> BTreeMap<String, String> {
            BTreeMap::new()
        }

        fn get_environment(&self) -> Environment {
            Environment {
                os: "test".to_string(),
                cwd: self.cwd.clone(),
                home: None,
                shell: "sh".to_string(),
                base_path: self.cwd.join(".forge"),
            }
        }

        fn get_config(&self) -> Result<Self::Config> {
            let mut config = self.config.clone();
            if self.semantic_model_disabled.load(Ordering::SeqCst) {
                config.semantic_embedding_model_id = None;
            }
            Ok(config)
        }

        async fn update_environment(&self, _ops: Vec<forge_domain::ConfigOperation>) -> Result<()> {
            anyhow::bail!("unused environment update")
        }
    }

    #[async_trait::async_trait]
    impl WorkspaceService for ProjectContextHarness {
        async fn sync_workspace(
            &self,
            _path: PathBuf,
        ) -> Result<forge_stream::MpscStream<Result<SyncProgress>>> {
            anyhow::bail!("unused workspace sync")
        }

        async fn produce_workspace_exact_fact_reference(
            &self,
            _path: PathBuf,
        ) -> Result<WorkspaceExactFactReferenceReport> {
            anyhow::bail!("unused workspace exact-fact reference")
        }

        async fn workspace_exact_fact_status(
            &self,
            _path: PathBuf,
        ) -> Result<WorkspaceExactFactStatusReport> {
            anyhow::bail!("unused workspace exact-fact status")
        }

        async fn workspace_evidence_replay_diagnostic(
            &self,
            _path: PathBuf,
        ) -> Result<WorkspaceEvidenceReplayDiagnostic> {
            anyhow::bail!("unused workspace evidence replay diagnostic")
        }

        async fn workspace_evidence_replay_preview_diagnostic(
            &self,
            path: PathBuf,
        ) -> Result<WorkspaceEvidenceReplayPreviewDiagnostic> {
            let manifest_path = local_project_model_manifest(&path);
            let manifest_found = manifest_path.is_file();
            let freshness = if self
                .stale_paths
                .iter()
                .any(|stale_path| stale_path == &path)
            {
                WorkspaceContextFreshness::Stale {
                    changed: vec!["src/lib.rs".to_string()],
                    deleted: Vec::new(),
                    added: Vec::new(),
                }
            } else if self
                .unknown_paths
                .iter()
                .any(|unknown_path| unknown_path == &path)
            {
                WorkspaceContextFreshness::Unknown {
                    reason: "fixture freshness unavailable".to_string(),
                }
            } else if manifest_found {
                WorkspaceContextFreshness::Fresh
            } else {
                WorkspaceContextFreshness::Unknown {
                    reason: "project-model manifest not found".to_string(),
                }
            };
            if !freshness.is_fresh() {
                let status = match &freshness {
                    WorkspaceContextFreshness::Stale { .. } => {
                        WorkspaceEvidenceReplayPreviewStatus::NotPreviewedManifestStale
                    }
                    WorkspaceContextFreshness::Unknown { .. } if manifest_found => {
                        WorkspaceEvidenceReplayPreviewStatus::NotPreviewedManifestUnknown
                    }
                    WorkspaceContextFreshness::Unknown { .. } => {
                        WorkspaceEvidenceReplayPreviewStatus::NotPreviewedManifestMissing
                    }
                    WorkspaceContextFreshness::Fresh => {
                        WorkspaceEvidenceReplayPreviewStatus::NotPreviewedManifestUnknown
                    }
                };
                let not_previewed_reason = match &freshness {
                    WorkspaceContextFreshness::Stale { changed, deleted, added } => Some(format!(
                        "manifest stale: changed={}, deleted={}, added={}",
                        changed.len(),
                        deleted.len(),
                        added.len()
                    )),
                    WorkspaceContextFreshness::Unknown { reason } => Some(reason.clone()),
                    WorkspaceContextFreshness::Fresh => None,
                };
                return Ok(WorkspaceEvidenceReplayPreviewDiagnostic {
                    status,
                    workspace_root_label: "workspace_root".to_string(),
                    manifest_label: "project_model_manifest".to_string(),
                    manifest_found,
                    manifest_freshness: freshness.label().to_string(),
                    not_previewed_reason,
                    manifest_hash: None,
                    content_policy: None,
                    stale_policy: None,
                    changed_excluded: 0,
                    deleted_excluded: 0,
                    budget: None,
                    selected: Vec::new(),
                    issues: Vec::new(),
                    rendered_preview: None,
                });
            }
            if self
                .replay_preview_empty_paths
                .iter()
                .any(|empty_path| empty_path == &path)
            {
                return Ok(WorkspaceEvidenceReplayPreviewDiagnostic {
                    status: WorkspaceEvidenceReplayPreviewStatus::NotPreviewedEmptyReplay,
                    workspace_root_label: "workspace_root".to_string(),
                    manifest_label: "project_model_manifest".to_string(),
                    manifest_found: true,
                    manifest_freshness: WorkspaceContextFreshness::Fresh.label().to_string(),
                    not_previewed_reason: Some("no previewable ledger evidence".to_string()),
                    manifest_hash: Some("fixture-manifest-hash".to_string()),
                    content_policy: Some("reference_only".to_string()),
                    stale_policy: Some("exclude_changed_or_deleted".to_string()),
                    changed_excluded: 0,
                    deleted_excluded: 0,
                    budget: Some(WorkspaceEvidenceReplayBudgetSummary {
                        original_candidate_count: 0,
                        selected_count: 0,
                        excluded_count: 0,
                        excluded_by_reason: BTreeMap::new(),
                        truncated: false,
                        max_artifacts: 8,
                        max_episode_lines: 64,
                        max_selected: 3,
                        stable_ordering: "fixture-stable-ordering".to_string(),
                    }),
                    selected: Vec::new(),
                    issues: Vec::new(),
                    rendered_preview: None,
                });
            }
            Ok(WorkspaceEvidenceReplayPreviewDiagnostic {
                status: WorkspaceEvidenceReplayPreviewStatus::PreviewedWithSelection,
                workspace_root_label: "workspace_root".to_string(),
                manifest_label: "project_model_manifest".to_string(),
                manifest_found: true,
                manifest_freshness: WorkspaceContextFreshness::Fresh.label().to_string(),
                not_previewed_reason: None,
                manifest_hash: Some("fixture-manifest-hash".to_string()),
                content_policy: Some("reference_only".to_string()),
                stale_policy: Some("exclude_changed_or_deleted".to_string()),
                changed_excluded: 1,
                deleted_excluded: 1,
                budget: Some(WorkspaceEvidenceReplayBudgetSummary {
                    original_candidate_count: 2,
                    selected_count: 1,
                    excluded_count: 1,
                    excluded_by_reason: BTreeMap::from([("changed_evidence_excluded".to_string(), 1)]),
                    truncated: false,
                    max_artifacts: 8,
                    max_episode_lines: 64,
                    max_selected: 3,
                    stable_ordering: "fixture-stable-ordering".to_string(),
                }),
                selected: vec![WorkspaceEvidenceReplayReference {
                    artifact_id: "artifact-fixture".to_string(),
                    artifact_path: "context_packs/artifact-fixture.json".to_string(),
                    evidence_id: "evidence-fixture".to_string(),
                    evidence_path: "src/lib.rs".to_string(),
                    start_line: Some(3),
                    end_line: Some(3),
                    score_kind: "exact_fact".to_string(),
                    score: 1.0,
                    provenance_path: "tool_episodes/episode-fixture.jsonl".to_string(),
                    provenance_start_line: Some(1),
                    provenance_end_line: Some(1),
                    provenance_source: "tool_episode".to_string(),
                    provenance_fingerprint: "fixture-fingerprint".to_string(),
                    freshness: "fresh".to_string(),
                    linked_episode_count: 1,
                    link_issue_count: 0,
                }],
                issues: Vec::new(),
                rendered_preview: Some(
                    "<project_model_context source=\"evidence_replay_preview\"><source path=\"src/lib.rs\" start_line=\"3\" end_line=\"3\" content_digest=\"fixture-digest\" /></project_model_context>".to_string(),
                ),
            })
        }

        async fn build_workspace_vector_index(
            &self,
            _path: PathBuf,
            _embedding_model_id: String,
        ) -> Result<WorkspaceVectorIndexBuildReport> {
            anyhow::bail!("unused workspace vector index build")
        }

        async fn embed_workspace_query(
            &self,
            query: String,
            embedding_model_id: String,
        ) -> Result<ProjectSemanticEmbeddingOutput> {
            self.embedding_calls.fetch_add(1, Ordering::SeqCst);
            self.embedding_inputs.lock().await.push(query);
            if self.embedding_pending.load(Ordering::SeqCst) {
                std::future::pending::<()>().await;
            }
            if self.embedding_failure.load(Ordering::SeqCst) {
                anyhow::bail!("fixture embedding provider unavailable")
            }
            let dimension = self.embedding_dimension.load(Ordering::SeqCst);
            Ok(ProjectSemanticEmbeddingOutput {
                embedding_model_id,
                dimension,
                vectors: vec![ProjectSemanticEmbeddingVector {
                    source_id: "query".to_string(),
                    source_fingerprint: "query".to_string(),
                    embedding: vec![1.0; dimension],
                }],
            })
        }

        async fn semantic_injection_readiness(
            &self,
            path: PathBuf,
            embedding_model_id: Option<String>,
        ) -> Result<WorkspaceSemanticInjectionReadiness> {
            if embedding_model_id.is_none() {
                return Ok(WorkspaceSemanticInjectionReadiness::SemanticDisabledNoModelConfig);
            }
            let readiness = self.semantic_readiness.lock().await;
            if let Some(readiness) = readiness.get(&path) {
                return Ok(readiness.clone());
            }
            if let Ok(canonical_path) = fs::canonicalize(&path)
                && let Some(readiness) = readiness.get(&canonical_path)
            {
                return Ok(readiness.clone());
            }
            Ok(WorkspaceSemanticInjectionReadiness::VectorIndexAbsentOrNoMatch)
        }

        async fn sem_search_availability(
            &self,
            path: PathBuf,
            embedding_model_id: Option<String>,
        ) -> Result<SemSearchAvailability> {
            match self
                .semantic_injection_readiness(path.clone(), embedding_model_id)
                .await?
            {
                WorkspaceSemanticInjectionReadiness::SemanticDisabledNoModelConfig => {
                    Ok(SemSearchAvailability::Unsupported {
                        reason: SemSearchUnsupportedReason::NoModelConfig,
                    })
                }
                WorkspaceSemanticInjectionReadiness::VectorIndexAbsentOrNoMatch => {
                    Ok(SemSearchAvailability::Unsupported {
                        reason: SemSearchUnsupportedReason::VectorArtifactAbsentOrNoMatch,
                    })
                }
                WorkspaceSemanticInjectionReadiness::VectorIndexReady { dimension } => {
                    Ok(SemSearchAvailability::Ready {
                        workspace_root: path,
                        manifest_hash: "fixture-manifest-hash".to_string(),
                        vector_artifact_id: "fixture-vector-artifact".to_string(),
                        dimension,
                    })
                }
                WorkspaceSemanticInjectionReadiness::VectorIndexAmbiguous => {
                    Ok(SemSearchAvailability::Unknown {
                        reason: SemSearchUnknownReason::AmbiguousVectorArtifact,
                    })
                }
                WorkspaceSemanticInjectionReadiness::VectorIndexCorruptOrNotReady => {
                    Ok(SemSearchAvailability::Unknown {
                        reason: SemSearchUnknownReason::VectorArtifactCorruptOrNotReady,
                    })
                }
                WorkspaceSemanticInjectionReadiness::VectorDimensionMismatch { .. } => {
                    Ok(SemSearchAvailability::Unknown {
                        reason: SemSearchUnknownReason::UnknownProbeFailure,
                    })
                }
                WorkspaceSemanticInjectionReadiness::EmbeddingProviderUnavailable => {
                    Ok(SemSearchAvailability::Unknown {
                        reason: SemSearchUnknownReason::UnknownProbeFailure,
                    })
                }
                WorkspaceSemanticInjectionReadiness::EmbeddingProviderTimeout => {
                    Ok(SemSearchAvailability::Unknown {
                        reason: SemSearchUnknownReason::UnknownProbeFailure,
                    })
                }
            }
        }

        async fn sem_search_diagnostic(
            &self,
            path: PathBuf,
            embedding_model_id: Option<String>,
        ) -> Result<SemSearchDiagnosticReport> {
            let availability = self
                .sem_search_availability(path.clone(), embedding_model_id.clone())
                .await?;
            Ok(SemSearchDiagnosticReport::from_availability(
                &availability,
                embedding_model_id.as_deref(),
                &path,
            ))
        }

        async fn project_context_reranker_diagnostic(
            &self,
        ) -> Result<WorkspaceRerankRuntimeDiagnostic> {
            self.rerank_runtime_diagnostic_calls
                .fetch_add(1, Ordering::SeqCst);
            Ok(self.rerank_runtime.lock().await.clone())
        }

        async fn query_workspace_committed(
            &self,
            path: PathBuf,
            params: SearchParams<'_>,
        ) -> Result<(ProjectContextCommittedQueryResult, Vec<Node>)> {
            self.committed_workspace_queries
                .fetch_add(1, Ordering::SeqCst);
            let nodes = self.fixture_query_workspace_nodes(path, params).await?;
            let result = self
                .committed_result
                .lock()
                .await
                .clone()
                .unwrap_or_else(|| {
                    ProjectContextCommittedQueryResult::no_write(
                        Default::default(),
                        ProjectContextPackNoWriteReason::EmptyEvidence,
                        Vec::new(),
                    )
                });
            Ok((result, nodes))
        }

        async fn query_workspace(
            &self,
            path: PathBuf,
            params: SearchParams<'_>,
        ) -> Result<Vec<Node>> {
            self.workspace_queries.fetch_add(1, Ordering::SeqCst);
            self.fixture_query_workspace_nodes(path, params).await
        }

        async fn list_workspaces(&self) -> Result<Vec<WorkspaceInfo>> {
            anyhow::bail!("unused workspace list")
        }

        async fn get_workspace_info(&self, _path: PathBuf) -> Result<Option<WorkspaceInfo>> {
            anyhow::bail!("unused workspace info")
        }

        async fn is_indexed(&self, path: &Path) -> Result<bool> {
            self.project_model_context_diagnostic(path)
                .await
                .map(|diagnostic| diagnostic.can_inject())
        }

        async fn delete_workspace(&self, _workspace_id: &WorkspaceId) -> Result<()> {
            anyhow::bail!("unused workspace delete")
        }

        async fn delete_workspaces(&self, _workspace_ids: &[WorkspaceId]) -> Result<()> {
            anyhow::bail!("unused workspace deletes")
        }

        async fn project_model_context_diagnostic(
            &self,
            path: &Path,
        ) -> Result<WorkspaceContextManifestDiagnostic> {
            self.index_checks.fetch_add(1, Ordering::SeqCst);
            let manifest_path = local_project_model_manifest(path);
            let manifest_found = manifest_path.is_file();
            let freshness = if self.stale_paths.iter().any(|stale_path| stale_path == path) {
                WorkspaceContextFreshness::Stale {
                    changed: vec!["src/lib.rs".to_string()],
                    deleted: Vec::new(),
                    added: Vec::new(),
                }
            } else if self
                .unknown_paths
                .iter()
                .any(|unknown_path| unknown_path == path)
            {
                WorkspaceContextFreshness::Unknown {
                    reason: "fixture freshness unavailable".to_string(),
                }
            } else if manifest_found {
                WorkspaceContextFreshness::Fresh
            } else {
                WorkspaceContextFreshness::Unknown {
                    reason: "project-model manifest not found".to_string(),
                }
            };
            let exact_fact_readiness = manifest_found.then(|| {
                if self
                    .inactive_exact_fact_paths
                    .iter()
                    .any(|inactive_path| inactive_path == path)
                {
                    WorkspaceExactFactReadinessDiagnostic {
                        status_label: "accepted_but_no_graph_edges".to_string(),
                        exact_facts_active: false,
                        issue_count: 2,
                        issue_summaries: vec![
                            "accepted_but_no_graph_edges".to_string(),
                            "redaction_safe_fixture_issue".to_string(),
                        ],
                        manifest_hash: Some("fixture-manifest-hash".to_string()),
                        manifest_external_facts_fingerprint: Some(
                            "fixture-external-facts".to_string(),
                        ),
                        reference_edge_count: 0,
                        exact_compiler_reference_edge_count: 0,
                    }
                } else {
                    WorkspaceExactFactReadinessDiagnostic {
                        status_label: "active".to_string(),
                        exact_facts_active: true,
                        issue_count: 0,
                        issue_summaries: Vec::new(),
                        manifest_hash: Some("fixture-manifest-hash".to_string()),
                        manifest_external_facts_fingerprint: Some(
                            "fixture-external-facts".to_string(),
                        ),
                        reference_edge_count: 2,
                        exact_compiler_reference_edge_count: 1,
                    }
                }
            });
            let evidence_readiness = manifest_found.then(|| WorkspaceEvidenceReadinessDiagnostic {
                context_pack_artifact_count: 1,
                context_pack_valid: true,
                context_pack_issue_count: 0,
                tool_episode_count: 1,
                tool_episode_valid: true,
                tool_episode_issue_count: 0,
                episode_artifact_link_valid: true,
                linked_episode_count: 1,
                missing_link_count: 0,
                worst_case_freshness: Some("fresh".to_string()),
                issue_summaries: Vec::new(),
                truncated: false,
            });
            Ok(WorkspaceContextManifestDiagnostic {
                workspace_root: path.to_path_buf(),
                manifest_found,
                manifest_path,
                freshness,
                manifest_hash: exact_fact_readiness
                    .as_ref()
                    .and_then(|readiness| readiness.manifest_hash.clone())
                    .or_else(|| manifest_found.then_some("fixture-manifest-hash".to_string())),
                exact_fact_readiness,
                evidence_readiness: evidence_readiness.clone(),
                evidence_ledger_activation: evidence_readiness.as_ref().map(|readiness| {
                    WorkspaceEvidenceLedgerActivationDiagnostic {
                        summary: WorkspaceEvidenceLedgerActivationSummary {
                            context_pack_artifact_count: readiness.context_pack_artifact_count,
                            readable_context_pack_count: readiness.context_pack_artifact_count,
                            tool_episode_count: readiness.tool_episode_count,
                            linked_episode_count: readiness.linked_episode_count,
                            missing_link_count: readiness.missing_link_count,
                            graph_node_count: 2,
                            graph_edge_count: readiness.linked_episode_count,
                            worst_case_freshness: readiness.worst_case_freshness.clone(),
                            issue_count: readiness
                                .context_pack_issue_count
                                .saturating_add(readiness.tool_episode_issue_count)
                                .saturating_add(readiness.missing_link_count),
                            issue_summaries: readiness.issue_summaries.clone(),
                            truncated: readiness.truncated,
                        },
                        graph: Some(WorkspaceEvidenceLedgerGraphMetadata {
                            node_count: 2,
                            edge_count: readiness.linked_episode_count,
                            node_kind_counts: BTreeMap::from([
                                ("retrieved_evidence".to_string(), 1),
                                ("tool_episode".to_string(), 1),
                            ]),
                            edge_kind_counts: BTreeMap::from([(
                                "tool_episode_relates".to_string(),
                                readiness.linked_episode_count,
                            )]),
                        }),
                    }
                }),
            })
        }

        async fn get_workspace_status(&self, _path: PathBuf) -> Result<Vec<FileStatus>> {
            anyhow::bail!("unused workspace status")
        }

        async fn is_authenticated(&self) -> Result<bool> {
            Ok(true)
        }

        async fn init_auth_credentials(&self) -> Result<WorkspaceAuth> {
            anyhow::bail!("unused workspace auth")
        }

        async fn init_workspace(&self, _path: PathBuf) -> Result<WorkspaceId> {
            anyhow::bail!("unused workspace init")
        }
    }

    impl ProjectContextHarness {
        async fn fixture_query_workspace_nodes(
            &self,
            path: PathBuf,
            params: SearchParams<'_>,
        ) -> Result<Vec<Node>> {
            assert!(params.query.contains("automatic injection needle"));
            assert_eq!(params.limit, Some(3));
            self.queried_workspaces.lock().await.push(path.clone());
            self.query_filters
                .lock()
                .await
                .push(params.starts_with.clone());
            self.query_embeddings
                .lock()
                .await
                .push(params.query_embedding.clone());
            self.query_rerank_sources
                .lock()
                .await
                .push(params.rerank_intent_source);
            if self
                .error_paths
                .iter()
                .any(|error_path| error_path == &path)
            {
                anyhow::bail!("fixture query failure for {}", path.display());
            }
            if self
                .empty_paths
                .iter()
                .any(|empty_path| empty_path == &path)
            {
                return Ok(Vec::new());
            }
            if params.query_embedding.is_some() {
                return Ok(vec![Node {
                    node_id: NodeId::new("symbol:src/vector_only.rs:SemanticVectorOnlyHit"),
                    node: NodeData::FileChunk(FileChunk {
                        file_path: "src/vector_only.rs".to_string(),
                        content: "pub struct SemanticVectorOnlyHit;".to_string(),
                        start_line: 7,
                        end_line: 7,
                    }),
                    relevance: Some(0.99),
                    distance: None,
                }]);
            }
            let file_path = params.starts_with.as_deref().unwrap_or("src/lib.rs");
            let long_content = (0..40)
                .map(|index| format!("pub fn long_{index}() -> usize {{ {index} }}"))
                .collect::<Vec<_>>()
                .join("\n");
            Ok(vec![
                Node {
                    node_id: NodeId::new("symbol:src/lib.rs:automatic_injection_needle"),
                    node: NodeData::FileChunk(FileChunk {
                        file_path: file_path.to_string(),
                        content: "pub fn automatic_injection_needle() -> usize { 42 }".to_string(),
                        start_line: 3,
                        end_line: 3,
                    }),
                    relevance: Some(0.875),
                    distance: None,
                },
                Node {
                    node_id: NodeId::new("symbol:src/long.rs:long_block"),
                    node: NodeData::FileChunk(FileChunk {
                        file_path: "src/long.rs".to_string(),
                        content: long_content,
                        start_line: 10,
                        end_line: 80,
                    }),
                    relevance: Some(0.75),
                    distance: None,
                },
                Node {
                    node_id: NodeId::new("file:src/full.rs"),
                    node: NodeData::File(forge_domain::FileNode {
                        file_path: "src/full.rs".to_string(),
                        content: "pub fn full_file_should_not_render() {}".repeat(100),
                        hash: "full-file-hash".to_string(),
                    }),
                    relevance: Some(0.5),
                    distance: None,
                },
                Node {
                    node_id: NodeId::new("symbol:src/extra.rs:extra"),
                    node: NodeData::FileChunk(FileChunk {
                        file_path: "src/extra.rs".to_string(),
                        content: "pub fn extra_should_not_render() {}".to_string(),
                        start_line: 1,
                        end_line: 1,
                    }),
                    relevance: Some(0.25),
                    distance: None,
                },
            ])
        }
    }

    #[async_trait::async_trait]
    impl LearningService for ProjectContextHarness {
        async fn capture_candidate_from_conversation(
            &self,
            _conversation_id: ConversationId,
            _source_event_id: String,
            _summary: String,
            _metadata: LearningCaptureMetadata,
        ) -> Result<LearningLedgerAppendOutcome> {
            anyhow::bail!("unused learning capture")
        }

        async fn insert_learning_event(
            &self,
            _event: LearningLedgerEvent,
        ) -> Result<LearningLedgerAppendOutcome> {
            anyhow::bail!("unused learning insert")
        }

        async fn review_learning_candidate_event(
            &self,
            _event: LearningLedgerEvent,
        ) -> Result<LearningReviewOutcome> {
            anyhow::bail!("unused learning review")
        }

        async fn promote_sensor_lesson(
            &self,
            _request: SensorLessonPromotionRequest,
        ) -> Result<SensorLessonPromotionOutcome> {
            anyhow::bail!("unused learning promotion")
        }

        async fn get_learning_record(
            &self,
            _record_id: LearningRecordId,
        ) -> Result<Option<LearningRecordProjection>> {
            Ok(None)
        }

        async fn list_learning_records(
            &self,
            review_state: Option<LearningReviewState>,
            limit: usize,
        ) -> Result<Vec<LearningRecordProjection>> {
            let mut records = self.learning_records.lock().await.clone();
            if let Some(review_state) = review_state {
                records.retain(|record| record.review_state == review_state);
            }
            records.truncate(limit);
            Ok(records)
        }

        async fn learning_freshness(
            &self,
            _review_state: Option<LearningReviewState>,
        ) -> Result<LearningLedgerFreshness> {
            Ok(self.learning_freshness.clone())
        }
    }

    #[async_trait::async_trait]
    impl AgentService for ProjectContextHarness {
        async fn chat_agent(
            &self,
            _id: &ModelId,
            context: Context,
            _provider_id: Option<ProviderId>,
        ) -> ResultStream<ChatCompletionMessage, anyhow::Error> {
            *self.captured_context.lock().await = Some(context);
            let message = ChatCompletionMessage::assistant(Content::full("done"))
                .finish_reason(FinishReason::Stop);
            Ok(Box::pin(tokio_stream::iter(std::iter::once(Ok(message)))))
        }

        async fn call(
            &self,
            _agent: &Agent,
            _context: &ToolCallContext,
            call: ToolCallFull,
        ) -> ToolResult {
            ToolResult::new(call.name)
                .failure(anyhow::anyhow!("tool calls are not expected in this test"))
        }

        async fn update(&self, _conversation: Conversation) -> Result<()> {
            Ok(())
        }

        async fn drain_steer_messages(
            &self,
            _conversation_id: &forge_domain::ConversationId,
        ) -> Result<Vec<SteerMessage>> {
            Ok(Vec::new())
        }
    }

    fn fixture_workspace() -> Result<(TempDir, PathBuf)> {
        let fixture = TempDir::new()?;
        let root = fixture.path().join("workspace");
        create_indexed_workspace(&root)?;
        Ok((fixture, root))
    }

    fn create_indexed_workspace(root: &Path) -> Result<()> {
        fs::create_dir_all(root.join("src"))?;
        fs::write(
            root.join("src/lib.rs"),
            "pub fn unrelated() {}\n\npub fn automatic_injection_needle() -> usize { 42 }\n",
        )?;
        let indexer = ProjectIndexer::new(root, local_project_model_dir(root));
        let manifest = indexer.index()?;
        indexer.write_manifest(&manifest)?;
        Ok(())
    }

    fn committed_persisted_episode_failed_result() -> Result<ProjectContextCommittedQueryResult> {
        let read_request = forge_project_model::ProjectContextReadRequest::new(
            "src/committed.rs",
            "committed-node",
            1,
            1,
        )?;
        let context_pack = forge_project_model::ContextPack {
            version: 1,
            manifest_hash: "fixture-manifest".to_string(),
            evidence: vec![forge_project_model::ContextPackEvidence {
                id: "committed-node".to_string(),
                path: "src/committed.rs".to_string(),
                symbol: None,
                source: forge_project_model::ContextPackEvidenceSource::RetrievalResult,
                freshness: forge_project_model::EvidenceFreshness::Fresh,
                provenance: forge_project_model::Provenance {
                    path: "src/committed.rs".to_string(),
                    start_line: Some(1),
                    end_line: Some(1),
                    source: "fixture".to_string(),
                    fingerprint: "fixture-fingerprint".to_string(),
                },
                score: 1.0,
            }],
            provenance: vec![forge_project_model::Provenance {
                path: "src/committed.rs".to_string(),
                start_line: Some(1),
                end_line: Some(1),
                source: "fixture".to_string(),
                fingerprint: "fixture-fingerprint".to_string(),
            }],
        };
        let retrieval_plan = forge_project_model::ProjectContextRetrievalPlan {
            query_diagnostics: forge_project_model::ProjectContextRetrievalQueryDiagnostics {
                query_text: Some("committed query".to_string()),
                path_prefix: None,
                path_suffixes: Vec::new(),
                limit: 1,
                top_k: Some(1),
                top_k_status: forge_project_model::ProjectContextTopKStatus::Applied {
                    candidate_count: 1,
                },
                use_case: Some("committed use case".to_string()),
                rerank_intent_source: None,
                rerank_intent_fingerprint: None,
                rerank_intent_len: None,
                offline_rerank_applicability: None,
                include_graph_expansion: false,
                stale_policy: forge_project_model::StaleEvidencePolicy::Reject,
                freshness_proof_level: forge_project_model::FreshnessProofLevel::FullFilesystem,
                phase_diagnostics:
                    forge_project_model::ProjectContextRetrievalPhaseDiagnostics::default(),
            },
            selected_results: Vec::new(),
            context_pack: Some(context_pack),
            read_requests: vec![read_request.clone()],
            write_decision:
                forge_project_model::ProjectContextWriteDecision::WriteContextPackAfterReadback,
            return_order: Vec::new(),
        };
        let replay_activation = forge_project_model::ReplayActivationBoundary {
            manifest_hash: "fixture-manifest".to_string(),
            active_refs: Vec::new(),
            issues: Vec::new(),
            diagnostics: forge_project_model::ReplayActivationDiagnostics::default(),
        };
        let commit = forge_project_model::ProjectContextPackCommit::from_retrieval_plan(
            &retrieval_plan,
            replay_activation,
        )?;
        let commit = match commit.verify_readbacks(vec![
            forge_project_model::ProjectContextReadbackOutcome::succeeded(&read_request),
        ])? {
            forge_project_model::ProjectContextPackReadbackDecision::Write(commit) => commit,
            forge_project_model::ProjectContextPackReadbackDecision::NoWrite(_) => {
                anyhow::bail!("fixture committed query should produce persisted proof")
            }
        };
        let tempdir = tempfile::tempdir()?;
        let indexer = forge_project_model::ProjectIndexer::new(
            tempdir.path(),
            tempdir.path().join(".forge_project_model"),
        );
        let proof = indexer.persist_verified_context_pack(&commit)?;
        Ok(ProjectContextCommittedQueryResult::persisted(
            forge_project_model::ProjectContextReadbackSummary::from_outcomes(&[
                forge_project_model::ProjectContextReadbackOutcome::succeeded(&read_request),
            ]),
            proof,
            forge_project_model::ProjectContextPersistedEpisodeAppendOutcome::failed(
                forge_project_model::ProjectContextEpisodeAppendFailureReason::EpisodeAppendFailed,
            ),
            vec![forge_project_model::ProjectContextCommittedResultItem::new(
                "committed-node",
                Some(1.0),
            )],
        ))
    }

    fn fixture_learning_projection(
        review_state: LearningReviewState,
        summary: &str,
    ) -> LearningRecordProjection {
        let conversation_id = ConversationId::generate();
        LearningRecordProjection {
            record_id: LearningRecordId::generate(),
            summary: summary.to_string(),
            accepted_summary: if review_state == LearningReviewState::Accepted {
                Some(
                    "sanctioned_sanitized_observation:validated_counters_and_fingerprints"
                        .to_string(),
                )
            } else {
                None
            },
            review_state,
            redaction_status: LearningRedactionStatus::Clean,
            provenance: LearningProvenance::conversation(
                conversation_id,
                "learning-event-1",
                "learning-source-fingerprint",
            ),
            capture_metadata: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            schema_version: LEARNING_LEDGER_SCHEMA_VERSION,
        }
    }

    #[tokio::test]
    async fn chat_flow_saves_conversation_then_captures_learning_candidate() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ChatFlowLearningHarness::new(root);
        let conversation = Conversation::generate();
        let conversation_id = conversation.id;
        setup.upsert_conversation(conversation).await?;
        let app = ForgeApp::new(setup.clone());
        let mut stream = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            app.chat(
                setup.state.agent.id.clone(),
                ChatRequest::new(
                    Event::new("runtime self-learning proof request"),
                    conversation_id,
                ),
            ),
        )
        .await??;

        for _ in 0..32 {
            if !setup.state.learning_events.lock().await.is_empty() {
                break;
            }
            match tokio::time::timeout(std::time::Duration::from_millis(250), stream.next()).await {
                Ok(Some(response)) => {
                    response?;
                }
                Ok(None) => break,
                Err(_) => {}
            }
        }
        while let Some(response) = stream.next().await {
            response?;
        }

        let saved = setup
            .find_conversation(&conversation_id)
            .await?
            .expect("conversation should be saved after chat flow");
        let events = setup.state.learning_events.lock().await.clone();
        let records = setup.state.learning_records.lock().await.clone();
        let actual = events
            .iter()
            .find(|event| event.event_kind == LearningEventKind::CandidateCaptured)
            .map(|event| {
                (
                    events.len(),
                    event.event_kind,
                    event.provenance.conversation_id,
                    event.summary.contains("conversation_saved"),
                    event.summary.contains("user_message_count=1"),
                    event
                        .summary
                        .contains("runtime self-learning proof request"),
                    records.iter().any(|record| {
                        record.record_id == event.record_id
                            && record.review_state == LearningReviewState::Accepted
                    }),
                    saved
                        .context
                        .as_ref()
                        .is_some_and(|context| context.messages.len() >= 2),
                )
            });
        let expected = Some((
            2usize,
            LearningEventKind::CandidateCaptured,
            Some(conversation_id),
            true,
            true,
            false,
            true,
            true,
        ));
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn chat_flow_promotes_captured_candidate_then_injects_next_chat_learning() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ChatFlowLearningHarness::new(root);
        let conversation = Conversation::generate();
        let conversation_id = conversation.id;
        setup.upsert_conversation(conversation).await?;
        let app = ForgeApp::new(setup.clone());
        let mut stream = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            app.chat(
                setup.state.agent.id.clone(),
                ChatRequest::new(
                    Event::new("turnkey self-learning promotion proof"),
                    conversation_id,
                ),
            ),
        )
        .await??;

        for _ in 0..32 {
            if !setup.state.learning_events.lock().await.is_empty() {
                break;
            }
            match tokio::time::timeout(std::time::Duration::from_millis(250), stream.next()).await {
                Ok(Some(response)) => {
                    response?;
                }
                Ok(None) => break,
                Err(_) => {}
            }
        }
        while let Some(response) = stream.next().await {
            response?;
        }

        let accepted = setup
            .state
            .learning_records
            .lock()
            .await
            .iter()
            .find(|record| record.review_state == LearningReviewState::Accepted)
            .cloned()
            .expect("chat save should auto-accept a safe current capture");
        let review_event = setup
            .state
            .learning_events
            .lock()
            .await
            .iter()
            .find(|event| event.event_kind == LearningEventKind::ReviewAccepted)
            .cloned()
            .expect("auto-review should append an accepted review event");
        let review_event_count = setup.state.learning_events.lock().await.len();
        assert_eq!(
            (accepted.record_id, review_event_count),
            (review_event.record_id, 2usize)
        );
        let saved = setup
            .find_conversation(&conversation_id)
            .await?
            .expect("conversation should remain saved after chat flow");
        save_conversation_and_capture_learning(setup.clone(), saved).await?;
        assert_eq!(setup.state.learning_events.lock().await.len(), 2usize);
        let next_conversation = Conversation::generate();
        let next_conversation_id = next_conversation.id;
        setup.upsert_conversation(next_conversation).await?;
        setup.state.captured_provider_context.lock().await.take();
        let mut next_stream = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            app.chat(
                setup.state.agent.id.clone(),
                ChatRequest::new(
                    Event::new("next chat should use accepted learning"),
                    next_conversation_id,
                ),
            ),
        )
        .await??;

        for _ in 0..32 {
            if setup.state.captured_provider_context.lock().await.is_some() {
                break;
            }
            match tokio::time::timeout(std::time::Duration::from_millis(250), next_stream.next())
                .await
            {
                Ok(Some(response)) => {
                    response?;
                }
                Ok(None) => break,
                Err(_) => {}
            }
        }

        let captured_context = setup
            .state
            .captured_provider_context
            .lock()
            .await
            .clone()
            .expect("provider context should be captured by fake provider");
        let learning_message = captured_context
            .messages
            .iter()
            .find_map(|message| match &message.message {
                ContextMessage::Text(text) if text.is_learning_context() => Some(text),
                _ => None,
            })
            .expect("accepted learning context should be injected into the next provider call");
        let actual = (
            review_event.event_kind,
            accepted.review_state,
            learning_message.content.contains("conversation_saved"),
            learning_message
                .content
                .contains("turnkey self-learning promotion proof"),
        );
        let expected = (
            LearningEventKind::ReviewAccepted,
            LearningReviewState::Accepted,
            true,
            false,
        );
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn chat_flow_rejected_candidate_remains_excluded_from_next_chat_learning() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ChatFlowLearningHarness::new(root);
        setup
            .set_learning_records(vec![fixture_learning_projection(
                LearningReviewState::Rejected,
                "rejected runtime learning must stay out",
            )])
            .await;
        let app = ForgeApp::new(setup.clone());
        let next_conversation = Conversation::generate();
        let next_conversation_id = next_conversation.id;
        setup.upsert_conversation(next_conversation).await?;
        setup.state.captured_provider_context.lock().await.take();
        let mut next_stream = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            app.chat(
                setup.state.agent.id.clone(),
                ChatRequest::new(
                    Event::new("next chat excludes rejected learning"),
                    next_conversation_id,
                ),
            ),
        )
        .await??;

        for _ in 0..32 {
            if setup.state.captured_provider_context.lock().await.is_some() {
                break;
            }
            match tokio::time::timeout(std::time::Duration::from_millis(250), next_stream.next())
                .await
            {
                Ok(Some(response)) => {
                    response?;
                }
                Ok(None) => break,
                Err(_) => {}
            }
        }

        let captured_context = setup
            .state
            .captured_provider_context
            .lock()
            .await
            .clone()
            .expect("provider context should be captured by fake provider");
        let injected = captured_context
            .messages
            .iter()
            .any(|message| match &message.message {
                ContextMessage::Text(text) => text.is_learning_context(),
                _ => false,
            });
        let actual = injected;
        let expected = false;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn chat_flow_injects_only_accepted_learning_into_provider_context() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ChatFlowLearningHarness::new(root);
        setup
            .set_learning_records(vec![
                fixture_learning_projection(
                    LearningReviewState::Candidate,
                    "candidate runtime learning must stay out",
                ),
                fixture_learning_projection(
                    LearningReviewState::Accepted,
                    "accepted runtime learning reaches provider context",
                ),
            ])
            .await;
        let conversation = Conversation::generate();
        let conversation_id = conversation.id;
        setup.upsert_conversation(conversation).await?;
        let app = ForgeApp::new(setup.clone());
        let mut stream = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            app.chat(
                setup.state.agent.id.clone(),
                ChatRequest::new(
                    Event::new("runtime accepted learning proof"),
                    conversation_id,
                ),
            ),
        )
        .await??;

        for _ in 0..32 {
            if setup.state.captured_provider_context.lock().await.is_some() {
                break;
            }
            match tokio::time::timeout(std::time::Duration::from_millis(250), stream.next()).await {
                Ok(Some(response)) => {
                    response?;
                }
                Ok(None) => break,
                Err(_) => {}
            }
        }

        let captured_context = setup
            .state
            .captured_provider_context
            .lock()
            .await
            .clone()
            .expect("provider context should be captured by fake provider");
        let learning_message = captured_context
            .messages
            .iter()
            .find_map(|message| match &message.message {
                ContextMessage::Text(text) if text.is_learning_context() => Some(text),
                _ => None,
            })
            .expect("accepted learning context should be injected before provider call");
        let actual = vec![
            learning_message
                .content
                .contains("sanctioned_sanitized_observation:validated_counters_and_fingerprints"),
            learning_message
                .content
                .contains("candidate runtime learning must stay out"),
            learning_message.droppable,
            learning_message.is_cache_eligible(),
        ];
        let expected = vec![true, false, true, false];
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_context_injection_uses_only_reviewed_accepted_records() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        setup
            .set_learning_records(vec![
                fixture_learning_projection(
                    LearningReviewState::Candidate,
                    "candidate must not inject",
                ),
                fixture_learning_projection(
                    LearningReviewState::Accepted,
                    "accepted reviewed learning",
                ),
            ])
            .await;
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        let actual = ProjectContextInjection::new(setup, agent)
            .inject_learning(conversation)
            .await
            .context
            .unwrap();
        let learning_message = actual
            .messages
            .iter()
            .find_map(|message| match &message.message {
                ContextMessage::Text(text) if text.is_learning_context() => Some(text),
                _ => None,
            })
            .expect("accepted learning context should be injected");

        assert!(
            learning_message
                .content
                .contains("sanctioned_sanitized_observation:validated_counters_and_fingerprints")
        );
        assert!(
            !learning_message
                .content
                .contains("accepted reviewed learning")
        );
        assert!(
            !learning_message
                .content
                .contains("candidate must not inject")
        );
        assert_eq!(learning_message.droppable, true);
        assert_eq!(learning_message.is_cache_eligible(), false);
        Ok(())
    }

    #[tokio::test]
    async fn learning_context_injection_excludes_sensor_proposal_pending_and_reject_audit_events()
    -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        setup
            .set_learning_records(vec![
                fixture_learning_projection(
                    LearningReviewState::Candidate,
                    "sensor_proposal reviewer=fake_learning_sensor_reviewer title=proposal must not inject",
                ),
                fixture_learning_projection(
                    LearningReviewState::Candidate,
                    "sensor_pending reviewer=fake_learning_sensor_reviewer reason=insufficient_substantive_evidence",
                ),
                fixture_learning_projection(
                    LearningReviewState::Rejected,
                    "sensor_reject reviewer=fake_learning_sensor_reviewer reason=invalid_output",
                ),
                fixture_learning_projection(
                    LearningReviewState::Accepted,
                    "accepted reviewed learning remains the only injectable record",
                ),
            ])
            .await;
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        let actual = ProjectContextInjection::new(setup, agent)
            .inject_learning(conversation)
            .await
            .context
            .unwrap();
        let learning_message = actual
            .messages
            .iter()
            .find_map(|message| match &message.message {
                ContextMessage::Text(text) if text.is_learning_context() => Some(text),
                _ => None,
            })
            .expect("accepted learning context should be injected");
        let actual = vec![
            learning_message
                .content
                .contains("sanctioned_sanitized_observation:validated_counters_and_fingerprints"),
            learning_message.content.contains("sensor_proposal"),
            learning_message.content.contains("sensor_pending"),
            learning_message.content.contains("sensor_reject"),
        ];
        let expected = vec![true, false, false, false];

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_context_injection_does_not_inject_unreviewed_candidates() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        setup
            .set_learning_records(vec![fixture_learning_projection(
                LearningReviewState::Candidate,
                "candidate must not inject",
            )])
            .await;
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        let actual = ProjectContextInjection::new(setup, agent)
            .inject_learning(conversation)
            .await
            .context
            .unwrap();
        let injected = actual
            .messages
            .iter()
            .any(|message| match &message.message {
                ContextMessage::Text(text) => text.is_learning_context(),
                _ => false,
            });

        assert_eq!(injected, false);
        Ok(())
    }

    #[tokio::test]
    async fn learning_context_injection_skips_payload_over_char_budget() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        let mut oversized = fixture_learning_projection(
            LearningReviewState::Accepted,
            &format!(
                "conversation_saved {}",
                "oversized reviewed learning ".repeat(1_000)
            ),
        );
        oversized.accepted_summary = None;
        oversized.capture_metadata = Some(LearningCaptureMetadata::conversation_save(
            1,
            1,
            "oversized-context-fingerprint",
            "oversized-summary-fingerprint",
        ));
        setup.set_learning_records(vec![oversized]).await;
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        let actual = ProjectContextInjection::new(setup, agent)
            .inject_learning(conversation)
            .await
            .context
            .unwrap();
        let injected = actual
            .messages
            .iter()
            .any(|message| match &message.message {
                ContextMessage::Text(text) => text.is_learning_context(),
                _ => false,
            });

        assert_eq!(injected, false);
        Ok(())
    }

    #[test]
    fn compaction_provider_request_metrics_do_not_use_historical_usage() -> Result<()> {
        let setup = OpenAiProviderRequestEstimate {
            estimated_input_tokens: 512,
            serialized_request_bytes: 512,
            media_token_padding: 0,
            output_token_reservation: 3_392,
            message_count: 1,
            tool_count: 0,
            messages_bytes: 128,
            tools_bytes: 0,
        };
        let actual = provider_request_compaction_estimate(setup, Some(3_392));
        let expected = ProviderRequestEstimate::new(512, Some(3_392));

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn project_model_query_ignores_live_runtime_context_message() -> Result<()> {
        let (_fixture, _root) = fixture_workspace()?;
        let model_id = ModelId::new("test-model");
        let conversation = Conversation::generate().context(
            Context::default()
                .add_message(ContextMessage::user(
                    "find automatic injection needle",
                    Some(model_id.clone()),
                ))
                .add_message(ContextMessage::Text(
                    TextMessage::new(
                        Role::User,
                        "<runtime_context freshness=\"live\" cache=\"uncached\">time</runtime_context>",
                    )
                    .model(model_id.clone())
                    .runtime_context()
                    .cacheable(false),
                )),
        );

        let actual = ProjectContextInjection::<ProjectContextHarness>::query_from_conversation(
            &conversation,
        )
        .unwrap();
        let expected = "find automatic injection needle";
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn project_model_query_preserves_real_user_message_containing_runtime_context_text()
    -> Result<()> {
        let (_fixture, _root) = fixture_workspace()?;
        let model_id = ModelId::new("test-model");
        let conversation = Conversation::generate().context(
            Context::default()
                .add_message(ContextMessage::user(
                    "previous automatic injection needle",
                    Some(model_id.clone()),
                ))
                .add_message(ContextMessage::user(
                    "explain the literal <runtime_context tag in prompts",
                    Some(model_id.clone()),
                )),
        );

        let actual = ProjectContextInjection::<ProjectContextHarness>::query_from_conversation(
            &conversation,
        )
        .unwrap();
        let expected = "explain the literal <runtime_context tag in prompts";
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_queries_absolute_file_tag_workspace_outside_cwd() -> Result<()> {
        let fixture = TempDir::new()?;
        let cwd = fixture.path().join("cwd-workspace");
        let other = fixture.path().join("other-workspace");
        create_indexed_workspace(&cwd)?;
        create_indexed_workspace(&other)?;
        let setup = ProjectContextHarness::new(cwd.clone());
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let mentioned_file = other.join("src/lib.rs");
        let conversation =
            Conversation::generate().context(Context::default().add_message(ContextMessage::user(
                format!(
                    "find automatic injection needle in @[{}]",
                    mentioned_file.display()
                ),
                Some(model_id),
            )));

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let expected_workspaces = vec![cwd, other];
        assert_eq!(*setup.queried_workspaces.lock().await, expected_workspaces);
        let expected_filters = vec![None, None];
        assert_eq!(*setup.query_filters.lock().await, expected_filters);
        assert_eq!(
            actual
                .context
                .unwrap()
                .messages
                .iter()
                .filter(|message| message
                    .content()
                    .is_some_and(|content| content.contains("<project_model_context")))
                .count(),
            2usize,
        );
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_queries_backticked_path_inside_cwd_with_filter() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation =
            Conversation::generate().context(Context::default().add_message(ContextMessage::user(
                "find automatic injection needle in `src/lib.rs`",
                Some(model_id),
            )));

        ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let actual = setup.query_filters.lock().await.clone();
        let expected = vec![None];
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_queries_directory_path_inside_cwd_with_safe_filter() -> Result<()>
    {
        let (_fixture, root) = fixture_workspace()?;
        let src_dir = root.join("src");
        let setup = ProjectContextHarness::new(root);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation =
            Conversation::generate().context(Context::default().add_message(ContextMessage::user(
                format!("find automatic injection needle in {}", src_dir.display()),
                Some(model_id),
            )));

        ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let actual = setup.query_filters.lock().await.clone();
        let expected = vec![None, Some("src/".to_string())];
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_exact_file_path_does_not_emit_prefix_filter() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation =
            Conversation::generate().context(Context::default().add_message(ContextMessage::user(
                "find automatic injection needle in `src/lib.rs`",
                Some(model_id),
            )));

        ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let actual = setup.query_filters.lock().await.clone();
        let expected = vec![None];
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_ignores_unindexed_mentioned_path_and_keeps_cwd() -> Result<()> {
        let fixture = TempDir::new()?;
        let cwd = fixture.path().join("workspace");
        create_indexed_workspace(&cwd)?;
        let unindexed_file = fixture.path().join("unindexed/src/lib.rs");
        fs::create_dir_all(unindexed_file.parent().unwrap())?;
        fs::write(&unindexed_file, "pub fn automatic_injection_needle() {}")?;
        let setup = ProjectContextHarness::new(cwd.clone());
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation =
            Conversation::generate().context(Context::default().add_message(ContextMessage::user(
                format!(
                    "find automatic injection needle in {}",
                    unindexed_file.display()
                ),
                Some(model_id),
            )));

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let expected_workspaces = vec![cwd];
        assert_eq!(*setup.queried_workspaces.lock().await, expected_workspaces);
        assert_eq!(setup.workspace_queries.load(Ordering::SeqCst), 0usize);
        assert_eq!(
            setup.committed_workspace_queries.load(Ordering::SeqCst),
            1usize
        );
        assert!(actual.context.unwrap().messages.iter().any(|message| {
            message
                .content()
                .is_some_and(|content| content.contains("<project_model_context"))
        }));
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_queries_cwd_baseline_without_mentioned_path() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root.clone());
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let expected_workspaces = vec![root];
        assert_eq!(*setup.queried_workspaces.lock().await, expected_workspaces);
        let expected_filters = vec![None];
        assert_eq!(*setup.query_filters.lock().await, expected_filters);
        let expected_sources = vec![forge_domain::SearchRerankIntentSource::AutomaticInjection];
        assert_eq!(*setup.query_rerank_sources.lock().await, expected_sources);
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_vector_ready_injects_vector_only_source() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root.clone());
        setup
            .set_semantic_readiness(
                root.clone(),
                WorkspaceSemanticInjectionReadiness::VectorIndexReady { dimension: 2 },
            )
            .await;
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation =
            Conversation::generate().context(Context::default().add_message(ContextMessage::user(
                "find automatic injection needle lexicalmiss",
                Some(model_id),
            )));

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let content = actual
            .context
            .unwrap()
            .messages
            .iter()
            .filter_map(|message| message.content())
            .collect::<Vec<_>>()
            .join("\n");
        let expected = (1usize, vec![Some(vec![1.0, 1.0])], true, false);

        assert_eq!(
            (
                setup.embedding_calls.load(Ordering::SeqCst),
                setup.query_embeddings.lock().await.clone(),
                content.contains("SemanticVectorOnlyHit"),
                content.contains("automatic_injection_needle"),
            ),
            expected,
        );
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_vector_ready_bounds_query_embedding_input() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root.clone());
        setup
            .set_semantic_readiness(
                root.clone(),
                WorkspaceSemanticInjectionReadiness::VectorIndexReady { dimension: 2 },
            )
            .await;
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let long_query = format!("find automatic injection needle {}", "x".repeat(10_000));
        let conversation = Conversation::generate().context(
            Context::default().add_message(ContextMessage::user(long_query, Some(model_id))),
        );

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let content = actual
            .context
            .unwrap()
            .messages
            .iter()
            .filter_map(|message| message.content())
            .collect::<Vec<_>>()
            .join("\n");
        let embedding_inputs = setup.embedding_inputs.lock().await.clone();
        let expected = (1usize, true, true);

        assert_eq!(
            (
                embedding_inputs.len(),
                embedding_inputs[0].chars().count() <= AUTOMATIC_CONTEXT_QUERY_EMBEDDING_TEXT_LIMIT,
                content.contains("SemanticVectorOnlyHit"),
            ),
            expected,
        );
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_no_vector_keeps_lexical_and_skips_embedding() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let content = actual
            .context
            .unwrap()
            .messages
            .iter()
            .filter_map(|message| message.content())
            .collect::<Vec<_>>()
            .join("\n");
        let expected = (0usize, vec![None], true);

        assert_eq!(
            (
                setup.embedding_calls.load(Ordering::SeqCst),
                setup.query_embeddings.lock().await.clone(),
                content.contains("automatic_injection_needle"),
            ),
            expected,
        );
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_committed_episode_append_failure_with_nodes_still_injects()
    -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        setup
            .set_committed_result(committed_persisted_episode_failed_result()?)
            .await;
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let content = actual
            .context
            .unwrap()
            .messages
            .iter()
            .filter_map(|message| message.content())
            .collect::<Vec<_>>()
            .join("\n");
        let expected = (1usize, 0usize, true, false);

        assert_eq!(
            (
                setup.committed_workspace_queries.load(Ordering::SeqCst),
                setup.workspace_queries.load(Ordering::SeqCst),
                content.contains("automatic_injection_needle"),
                content.contains("EpisodeAppendFailed"),
            ),
            expected,
        );
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_invalid_vector_state_marks_diagnostic_and_uses_lexical()
    -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root.clone());
        setup
            .set_semantic_readiness(
                root.clone(),
                WorkspaceSemanticInjectionReadiness::VectorIndexAmbiguous,
            )
            .await;
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let content = actual
            .context
            .unwrap()
            .messages
            .iter()
            .filter_map(|message| message.content())
            .collect::<Vec<_>>()
            .join("\n");
        let expected = (0usize, vec![None], true, true, false);

        assert_eq!(
            (
                setup.embedding_calls.load(Ordering::SeqCst),
                setup.query_embeddings.lock().await.clone(),
                content.contains("automatic_injection_needle"),
                content.contains("semantic_vector_state=VectorIndexAmbiguous"),
                content.contains("SemanticVectorOnlyHit"),
            ),
            expected,
        );
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_corrupt_vector_state_marks_diagnostic_and_uses_lexical()
    -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root.clone());
        setup
            .set_semantic_readiness(
                root.clone(),
                WorkspaceSemanticInjectionReadiness::VectorIndexCorruptOrNotReady,
            )
            .await;
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let content = actual
            .context
            .unwrap()
            .messages
            .iter()
            .filter_map(|message| message.content())
            .collect::<Vec<_>>()
            .join("\n");
        let expected = (0usize, vec![None], true, true, false);

        assert_eq!(
            (
                setup.embedding_calls.load(Ordering::SeqCst),
                setup.query_embeddings.lock().await.clone(),
                content.contains("automatic_injection_needle"),
                content.contains("semantic_vector_state=VectorIndexCorruptOrNotReady"),
                content.contains("SemanticVectorOnlyHit"),
            ),
            expected,
        );
        Ok(())
    }

    #[tokio::test(start_paused = true)]
    async fn project_model_context_embedding_timeout_keeps_main_request_lexical() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root.clone());
        setup.pending_embedding();
        setup
            .set_semantic_readiness(
                root.clone(),
                WorkspaceSemanticInjectionReadiness::VectorIndexReady { dimension: 2 },
            )
            .await;
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let content = actual
            .context
            .unwrap()
            .messages
            .iter()
            .filter_map(|message| message.content())
            .collect::<Vec<_>>()
            .join("\n");
        let expected = (1usize, vec![None], true, true);

        assert_eq!(
            (
                setup.embedding_calls.load(Ordering::SeqCst),
                setup.query_embeddings.lock().await.clone(),
                content.contains("automatic_injection_needle"),
                content.contains("semantic_vector_state=EmbeddingProviderTimeout"),
            ),
            expected,
        );
        Ok(())
    }
    #[tokio::test]
    async fn project_model_context_embedding_failure_keeps_main_request_lexical() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root.clone());
        setup.fail_embedding();
        setup
            .set_semantic_readiness(
                root.clone(),
                WorkspaceSemanticInjectionReadiness::VectorIndexReady { dimension: 2 },
            )
            .await;
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let content = actual
            .context
            .unwrap()
            .messages
            .iter()
            .filter_map(|message| message.content())
            .collect::<Vec<_>>()
            .join("\n");
        let expected = (1usize, vec![None], true, true);

        assert_eq!(
            (
                setup.embedding_calls.load(Ordering::SeqCst),
                setup.query_embeddings.lock().await.clone(),
                content.contains("automatic_injection_needle"),
                content.contains("semantic_vector_state=EmbeddingProviderUnavailable"),
            ),
            expected,
        );
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_dimension_mismatch_disables_semantic_without_query_vector()
    -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root.clone());
        setup.set_embedding_dimension(3);
        setup
            .set_semantic_readiness(
                root.clone(),
                WorkspaceSemanticInjectionReadiness::VectorIndexReady { dimension: 2 },
            )
            .await;
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let content = actual
            .context
            .unwrap()
            .messages
            .iter()
            .filter_map(|message| message.content())
            .collect::<Vec<_>>()
            .join("\n");
        let expected = (1usize, vec![None], true, true);

        assert_eq!(
            (
                setup.embedding_calls.load(Ordering::SeqCst),
                setup.query_embeddings.lock().await.clone(),
                content.contains("automatic_injection_needle"),
                content.contains("semantic_vector_state=VectorDimensionMismatch"),
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn project_model_context_semantic_diagnostic_does_not_activate_reranker_config() {
        let actual = ProjectContextInjection::<ProjectContextHarness>::semantic_invalid_diagnostic(
            WorkspaceSemanticInjectionReadiness::VectorIndexAmbiguous,
        );
        let expected = false;

        assert_eq!(actual.contains("reranker"), expected);
    }
    #[tokio::test]
    async fn project_model_context_does_not_inject_stale_manifest() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new_with_stale_paths(root.clone(), vec![root]);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let expected_queries = 0usize;

        assert_eq!(
            setup.workspace_queries.load(Ordering::SeqCst),
            expected_queries
        );
        assert!(!actual.context.unwrap().messages.iter().any(|message| {
            message
                .content()
                .is_some_and(|content| content.contains("<project_model_context"))
        }));
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_nearest_stale_manifest_blocks_parent_fallback() -> Result<()> {
        let fixture = TempDir::new()?;
        let parent = fixture.path().join("workspace");
        create_indexed_workspace(&parent)?;
        let nested = parent.join("nested");
        create_indexed_workspace(&nested)?;
        let setup = ProjectContextHarness::new_with_stale_paths(nested.clone(), vec![nested]);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let expected_queries = 0usize;

        assert_eq!(
            setup.workspace_queries.load(Ordering::SeqCst),
            expected_queries
        );
        assert!(!actual.context.unwrap().messages.iter().any(|message| {
            message
                .content()
                .is_some_and(|content| content.contains("<project_model_context"))
        }));
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_nearest_unknown_manifest_blocks_parent_fallback() -> Result<()> {
        let fixture = TempDir::new()?;
        let parent = fixture.path().join("workspace");
        create_indexed_workspace(&parent)?;
        let nested = parent.join("nested");
        create_indexed_workspace(&nested)?;
        let setup = ProjectContextHarness::new_with_unknown_paths(nested.clone(), vec![nested]);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let expected_queries = 0usize;

        assert_eq!(
            setup.workspace_queries.load(Ordering::SeqCst),
            expected_queries
        );
        assert!(!actual.context.unwrap().messages.iter().any(|message| {
            message
                .content()
                .is_some_and(|content| content.contains("<project_model_context"))
        }));
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_reports_fresh_target_and_injection_decision() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root.clone());
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let expected = (root, true, 1usize, 1usize, None::<String>, 0usize, 0usize);

        assert_eq!(
            (
                actual.cwd,
                actual.would_inject,
                actual.candidates.len(),
                actual.selected_targets.len(),
                actual.skip_reason,
                setup.workspace_queries.load(Ordering::SeqCst),
                setup.committed_workspace_queries.load(Ordering::SeqCst),
            ),
            expected
        );
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_with_query_is_read_only_and_does_not_query_workspace() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let actual_plan = actual
            .retrieval_plan_diagnostics
            .first()
            .expect("explain should include query-specific retrieval plan diagnostic");
        let expected = (
            0usize,
            0usize,
            Vec::<PathBuf>::new(),
            true,
            1usize,
            true,
            false,
        );

        assert_eq!(
            (
                setup.workspace_queries.load(Ordering::SeqCst),
                setup.committed_workspace_queries.load(Ordering::SeqCst),
                actual.retrieval_empty_targets,
                actual.would_inject,
                actual.replay_preview_diagnostics.len(),
                actual_plan.planned,
                actual_plan.retrieval_empty,
            ),
            expected,
        );
        assert!(actual_plan.selected_result_count > 0);
        assert!(actual_plan.read_request_count > 0);
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_with_query_does_not_call_embedding_provider() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root.clone());
        setup
            .set_semantic_readiness(
                root,
                WorkspaceSemanticInjectionReadiness::VectorIndexReady { dimension: 2 },
            )
            .await;
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let actual_readiness = actual
            .semantic_readiness
            .first()
            .expect("semantic readiness diagnostic should be present");
        let actual_plan = actual
            .retrieval_plan_diagnostics
            .first()
            .expect("retrieval plan diagnostic should be present");
        let expected = (
            0usize,
            WorkspaceSemanticReadinessStatus::VectorIndexReady,
            Some(2usize),
            WorkspaceRetrievalPhaseStatus::Unavailable {
                reason: WorkspaceRetrievalPhaseUnavailableReason::MissingQueryEmbedding,
            },
        );

        assert_eq!(
            (
                setup.embedding_calls.load(Ordering::SeqCst),
                actual_readiness.status.clone(),
                actual_readiness.dimension,
                actual_plan.phase_diagnostics.vector.clone(),
            ),
            expected,
        );
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_semantic_no_model_config_reports_disabled_diagnostic() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        setup.disable_semantic_embedding_model();
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let actual = actual
            .semantic_readiness
            .first()
            .expect("semantic readiness diagnostic should be present");
        let expected = (
            true,
            WorkspaceSemanticReadinessStatus::SemanticDisabledNoModelConfig,
            None,
        );

        assert_eq!(
            (actual.evaluated, actual.status.clone(), actual.dimension),
            expected
        );
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_semantic_absent_no_match_reports_absent_diagnostic() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let actual = actual
            .semantic_readiness
            .first()
            .expect("semantic readiness diagnostic should be present");
        let expected = WorkspaceSemanticReadinessStatus::VectorIndexAbsentOrNoMatch;

        assert_eq!(actual.status, expected);
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_semantic_ambiguous_reports_invalid_ambiguous_diagnostic() -> Result<()>
    {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root.clone());
        setup
            .set_semantic_readiness(
                root,
                WorkspaceSemanticInjectionReadiness::VectorIndexAmbiguous,
            )
            .await;
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let actual = actual
            .semantic_readiness
            .first()
            .expect("semantic readiness diagnostic should be present");
        let expected = WorkspaceSemanticReadinessStatus::VectorIndexAmbiguous;

        assert_eq!(actual.status, expected);
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_semantic_corrupt_reports_invalid_not_ready_diagnostic() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root.clone());
        setup
            .set_semantic_readiness(
                root,
                WorkspaceSemanticInjectionReadiness::VectorIndexCorruptOrNotReady,
            )
            .await;
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let actual = actual
            .semantic_readiness
            .first()
            .expect("semantic readiness diagnostic should be present");
        let expected = WorkspaceSemanticReadinessStatus::VectorIndexCorruptOrNotReady;

        assert_eq!(actual.status, expected);
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_reranker_phase_reports_missing_runtime_without_activation()
    -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let actual = actual
            .retrieval_plan_diagnostics
            .first()
            .expect("retrieval plan diagnostic should be present");
        let expected = WorkspaceRetrievalPhaseStatus::Unavailable {
            reason: WorkspaceRetrievalPhaseUnavailableReason::MissingReranker,
        };

        assert_eq!(actual.phase_diagnostics.rerank, expected);
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_reranker_runtime_reports_configured_ready_without_activation()
    -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        let fake_reranker_calls = AtomicUsize::new(0);
        setup
            .set_rerank_runtime(WorkspaceRerankRuntimeDiagnostic::configured_ready(
                "project-context-offline-rerank",
                "offline-rerank-score-artifact",
            ))
            .await;
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let actual = actual
            .retrieval_plan_diagnostics
            .first()
            .expect("retrieval plan diagnostic should be present");
        let actual_runtime = actual
            .rerank_runtime
            .clone()
            .expect("rerank runtime diagnostic should be present");
        let expected = (
            WorkspaceRerankRuntimeState::ConfiguredReady,
            true,
            WorkspaceRetrievalPhaseStatus::Active { result_count: 0 },
            false,
            0usize,
            1usize,
        );

        assert_eq!(
            (
                actual_runtime.state,
                actual_runtime.rerank_available,
                actual.phase_diagnostics.rerank.clone(),
                matches!(
                    actual.phase_diagnostics.rerank,
                    WorkspaceRetrievalPhaseStatus::Unavailable {
                        reason: WorkspaceRetrievalPhaseUnavailableReason::MissingReranker
                    }
                ),
                fake_reranker_calls.load(Ordering::SeqCst),
                setup.rerank_runtime_diagnostic_calls(),
            ),
            expected,
        );
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_reranker_runtime_configured_not_ready_projects_phase_not_ready()
    -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        setup
            .set_rerank_runtime(WorkspaceRerankRuntimeDiagnostic::configured_not_ready(
                "project-context-offline-rerank",
                "offline-rerank-score-artifact",
                OfflineRerankScoreArtifactReadinessIssue::ArtifactUnreadable,
            ))
            .await;
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let actual = actual
            .retrieval_plan_diagnostics
            .first()
            .expect("retrieval plan diagnostic should be present");
        let expected = WorkspaceRetrievalPhaseStatus::Unavailable {
            reason: WorkspaceRetrievalPhaseUnavailableReason::RerankerNotReady,
        };

        assert_eq!(actual.phase_diagnostics.rerank, expected);
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_reranker_runtime_reports_missing_config() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let actual = actual
            .retrieval_plan_diagnostics
            .first()
            .expect("retrieval plan diagnostic should be present");
        let actual_runtime = actual
            .rerank_runtime
            .clone()
            .expect("rerank runtime diagnostic should be present");
        let expected = (
            WorkspaceRerankRuntimeState::MissingConfig,
            false,
            WorkspaceRetrievalPhaseStatus::Unavailable {
                reason: WorkspaceRetrievalPhaseUnavailableReason::MissingReranker,
            },
        );

        assert_eq!(
            (
                actual_runtime.state,
                actual_runtime.rerank_available,
                actual.phase_diagnostics.rerank.clone(),
            ),
            expected,
        );
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_reranker_runtime_redacts_rejected_malicious_artifact() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        setup
            .set_rerank_runtime(WorkspaceRerankRuntimeDiagnostic::configured_not_ready(
                "project-context-offline-rerank",
                "offline-rerank-score-artifact",
                OfflineRerankScoreArtifactReadinessIssue::ArtifactPathRejected,
            ))
            .await;
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some(
                "https://evil.invalid/raw-query candidate secret artifact content".to_string(),
            ))
            .await;
        let diagnostic = actual
            .retrieval_plan_diagnostics
            .first()
            .and_then(|diagnostic| diagnostic.rerank_runtime.clone())
            .expect("rerank runtime diagnostic should be present");
        let diagnostic_json = serde_json::to_string(&diagnostic)?;
        let expected = (
            WorkspaceRerankRuntimeState::ConfiguredNotReady {
                issue: OfflineRerankScoreArtifactReadinessIssue::ArtifactPathRejected,
            },
            Some("artifact_path_rejected".to_string()),
            false,
            false,
            false,
            false,
        );

        assert_eq!(
            (
                diagnostic.state.clone(),
                diagnostic.issue_label().map(str::to_string),
                diagnostic_json.contains("evil.invalid"),
                diagnostic_json.contains("candidate secret"),
                diagnostic_json.contains("artifact content"),
                diagnostic_json.contains("/secret/"),
            ),
            expected,
        );
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_porcelain_json_contains_no_raw_embeddings_snippets_or_provider_payloads()
    -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root.clone());
        setup
            .set_semantic_readiness(
                root,
                WorkspaceSemanticInjectionReadiness::VectorIndexReady { dimension: 2 },
            )
            .await;
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let actual = serde_json::to_string(&actual)?;
        let expected = (false, false, false, false, true);

        assert_eq!(
            (
                actual.contains("embedding\":"),
                actual.contains("pub fn automatic_injection_needle"),
                actual.contains("provider_request"),
                actual.contains("provider_response"),
                actual.contains("semantic_readiness"),
            ),
            expected,
        );
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_does_not_perform_workspace_query_writes_or_episode_appends()
    -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let expected = (0usize, 0usize, 0usize, true, true);

        assert_eq!(
            (
                setup.workspace_queries.load(Ordering::SeqCst),
                setup.committed_workspace_queries.load(Ordering::SeqCst),
                setup.learning_records.lock().await.len(),
                actual
                    .retrieval_plan_diagnostics
                    .first()
                    .is_some_and(|diagnostic| diagnostic.write_decision.as_deref()
                        == Some("WriteContextPackAfterReadback")),
                actual.replay_preview_diagnostics.len() == 1,
            ),
            expected,
        );
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_absent_query_reports_empty_read_only_plan() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .explain(Some("absent-token-for-empty-retrieval-plan".to_string()))
            .await;
        let actual_plan = actual
            .retrieval_plan_diagnostics
            .first()
            .expect("explain should include empty retrieval plan diagnostic");
        let expected = (0usize, true, 0usize, 0usize, Some("NoWriteEmptyRetrieval"));

        assert_eq!(
            (
                setup.workspace_queries.load(Ordering::SeqCst),
                actual_plan.planned,
                actual_plan.selected_result_count,
                actual_plan.read_request_count,
                actual_plan.write_decision.as_deref(),
            ),
            expected,
        );
        assert!(actual_plan.retrieval_empty);
        Ok(())
    }

    #[test]
    fn retrieval_plan_diagnostic_preserves_offline_rerank_applicability() -> Result<()> {
        let diagnostic = forge_project_model::ProjectContextRetrievalPlanDiagnostic {
            planned: true,
            refusal_code: None,
            refusal_detail: None,
            selected_result_count: 1,
            read_request_count: 0,
            write_decision: Some(forge_project_model::ProjectContextWriteDecision::NoWriteEmptyRetrieval),
            selected_summaries: Vec::new(),
            read_request_summaries: Vec::new(),
            phase_diagnostics: forge_project_model::ProjectContextRetrievalPhaseDiagnostics::default(),
            rerank_intent_source: None,
            rerank_intent_fingerprint: Some(forge_project_model::fingerprint("raw query secret")),
            rerank_intent_len: Some(16),
            offline_rerank_applicability: Some(
                forge_project_model::OfflineRerankApplicability::Mismatch {
                    reasons: vec![
                        forge_project_model::OfflineRerankApplicabilityMismatch::RerankIntentFingerprintMismatch,
                    ],
                },
            ),
            retrieval_empty: false,
            truncated: false,
        };

        let actual =
            ProjectContextInjection::<ProjectContextHarness>::retrieval_plan_diagnostic_to_domain(
                diagnostic,
            );
        let serialized = serde_json::to_string(&actual)?;
        let expected = (
            Some(forge_domain::WorkspaceOfflineRerankApplicability::Mismatch {
                reasons: vec![
                    forge_domain::WorkspaceOfflineRerankApplicabilityMismatch::RerankIntentFingerprintMismatch,
                ],
            }),
            true,
            false,
            false,
        );

        assert_eq!(
            (
                actual.offline_rerank_applicability,
                serialized.contains("rerank_intent_fingerprint_mismatch"),
                serialized.contains("raw query secret"),
                serialized.contains("raw candidate secret"),
            ),
            expected,
        );
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_stale_manifest_reports_planner_refusal_without_io_plan() -> Result<()>
    {
        let (_fixture, root) = fixture_workspace()?;
        fs::write(
            root.join("src/lib.rs"),
            "pub fn unrelated() {}\n\npub fn automatic_injection_needle() -> usize { 43 }\n",
        )?;
        let setup = ProjectContextHarness::new_with_stale_paths(root.clone(), vec![root]);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let actual_plan = actual
            .retrieval_plan_diagnostics
            .first()
            .expect("stale manifest should include planner refusal diagnostic");
        let expected = (
            0usize,
            false,
            Some("ManifestNotInjectable"),
            0usize,
            None::<&str>,
            false,
        );

        assert_eq!(
            (
                setup.workspace_queries.load(Ordering::SeqCst),
                actual_plan.planned,
                actual_plan.refusal_code.as_deref(),
                actual_plan.read_request_count,
                actual_plan.write_decision.as_deref(),
                actual_plan.retrieval_empty,
            ),
            expected,
        );
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_retrieval_plan_diagnostic_is_metadata_only_and_query_not_repeated()
    -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root.clone());
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let actual = serde_json::to_string(&actual.retrieval_plan_diagnostics)?;
        let expected = (false, false, false, true, true);

        assert_eq!(
            (
                actual.contains("pub fn automatic_injection_needle"),
                actual.contains("project_model_context"),
                actual.contains("find automatic injection needle"),
                actual.contains("workspace_root"),
                actual.contains("project_model_manifest"),
            ),
            expected,
        );
        assert!(!actual.contains(&root.display().to_string()));
        Ok(())
    }

    #[tokio::test]
    async fn inject_still_queries_workspace_for_representative_query() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let actual = setup.committed_workspace_queries.load(Ordering::SeqCst);
        let expected = 1usize;

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_includes_replay_derived_non_query_specific_preview() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("unrelated query text".to_string()))
            .await;
        let actual_preview = actual
            .replay_preview_diagnostics
            .first()
            .expect("explain should include replay-derived preview diagnostic");
        let expected = (
            WorkspaceEvidenceReplayPreviewStatus::PreviewedWithSelection,
            Some("reference_only"),
            Some(1usize),
            1usize,
            true,
        );

        assert_eq!(
            (
                actual_preview.status.clone(),
                actual_preview.content_policy.as_deref(),
                actual_preview
                    .budget
                    .as_ref()
                    .map(|budget| budget.selected_count),
                actual_preview.selected.len(),
                actual_preview
                    .rendered_preview
                    .as_ref()
                    .is_some_and(|preview| preview.contains("evidence_replay_preview")),
            ),
            expected,
        );
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_empty_replay_preview_is_not_injection_impossible() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup =
            ProjectContextHarness::new_with_replay_preview_empty_paths(root.clone(), vec![root]);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let actual_preview = actual
            .replay_preview_diagnostics
            .first()
            .expect("explain should include empty replay diagnostic");
        let expected = (
            true,
            WorkspaceEvidenceReplayPreviewStatus::NotPreviewedEmptyReplay,
            Some("no previewable ledger evidence"),
            false,
            0usize,
        );

        assert_eq!(
            (
                actual.would_inject,
                actual_preview.status.clone(),
                actual_preview.not_previewed_reason.as_deref(),
                actual_preview.rendered_preview.is_some(),
                actual_preview.selected.len(),
            ),
            expected,
        );
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_stale_manifest_is_diagnostic_state_without_writes() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new_with_stale_paths(root.clone(), vec![root]);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let actual_preview = actual
            .replay_preview_diagnostics
            .first()
            .expect("stale manifest should still produce read-only replay diagnostic state");
        let expected = (
            false,
            0usize,
            WorkspaceEvidenceReplayPreviewStatus::NotPreviewedManifestStale,
            "stale",
        );

        assert_eq!(
            (
                actual.would_inject,
                setup.workspace_queries.load(Ordering::SeqCst),
                actual_preview.status.clone(),
                actual_preview.manifest_freshness.as_str(),
            ),
            expected,
        );
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_replay_preview_is_metadata_only_and_redacted() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root.clone());
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let actual = serde_json::to_string(&actual.replay_preview_diagnostics)?;
        let expected = (false, false, false, true, true);

        assert_eq!(
            (
                actual.contains("pub fn automatic_injection_needle"),
                actual.contains("tool payload"),
                actual.contains(&root.display().to_string()),
                actual.contains("workspace_root"),
                actual.contains("project_model_manifest"),
            ),
            expected,
        );
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_reports_exact_fact_active_for_selected_target() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root.clone());
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let actual_readiness = actual.selected_targets[0]
            .exact_fact_readiness
            .as_ref()
            .unwrap();
        let expected = (true, "active", 2usize, 1usize);

        assert_eq!(
            (
                actual_readiness.exact_facts_active,
                actual_readiness.status_label.as_str(),
                actual_readiness.reference_edge_count,
                actual_readiness.exact_compiler_reference_edge_count,
            ),
            expected,
        );
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_reports_evidence_readiness_for_selected_target() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root.clone());
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let actual_readiness = actual.selected_targets[0]
            .evidence_readiness
            .as_ref()
            .unwrap();
        let expected = (1usize, true, 1usize, true, true, Some("fresh"));

        assert_eq!(
            (
                actual_readiness.context_pack_artifact_count,
                actual_readiness.context_pack_valid,
                actual_readiness.tool_episode_count,
                actual_readiness.tool_episode_valid,
                actual_readiness.episode_artifact_link_valid,
                actual_readiness.worst_case_freshness.as_deref(),
            ),
            expected,
        );
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_reports_exact_fact_inactive_without_blocking_manifest_injection()
    -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new_with_inactive_exact_fact_paths(
            root.clone(),
            vec![root.clone()],
        );
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let actual_readiness = actual.selected_targets[0]
            .exact_fact_readiness
            .as_ref()
            .unwrap();
        let expected = (true, false, "accepted_but_no_graph_edges");

        assert_eq!(
            (
                actual.would_inject,
                actual_readiness.exact_facts_active,
                actual_readiness.status_label.as_str(),
            ),
            expected,
        );
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_injection_renders_exact_fact_readiness_metadata() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root.clone());
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id)),
        ));

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let messages = actual.context.unwrap().messages;
        let actual_context = messages
            .iter()
            .filter_map(|message| message.content().map(str::to_string))
            .find(|content| content.contains("<project_model_context"))
            .unwrap();
        let actual_sidecar = messages
            .iter()
            .filter_map(|message| message.content().map(str::to_string))
            .find(|content| content.contains("<project_model_volatile_sidecar"))
            .unwrap();
        let expected = (true, true, true, true, true, true, true, true, false);

        assert_eq!(
            (
                actual_sidecar.contains("exact_fact_status=active"),
                actual_sidecar.contains("fixture-external-facts"),
                actual_sidecar.contains("context_pack_issue_count=0"),
                actual_sidecar.contains("tool_episode_issue_count=0"),
                actual_sidecar.contains("episode_artifact_link_valid=true"),
                actual_context.contains("freshness=\"fresh\""),
                actual_context.contains("exact_fact_readiness=\"evaluated\""),
                actual_context.contains("cache=\"stable\""),
                actual_context.contains("rendered_context_budget_exceeded"),
            ),
            expected,
        );
        Ok(())
    }

    #[test]
    fn project_model_context_exact_fact_readiness_does_not_change_can_inject() {
        let setup = WorkspaceContextManifestDiagnostic {
            workspace_root: PathBuf::from("/workspace"),
            manifest_path: PathBuf::from("/workspace/.forge_project_model/project_manifest.json"),
            manifest_found: true,
            freshness: WorkspaceContextFreshness::Fresh,
            manifest_hash: Some("hash".to_string()),
            exact_fact_readiness: Some(WorkspaceExactFactReadinessDiagnostic {
                status_label: "accepted_but_no_graph_edges".to_string(),
                exact_facts_active: false,
                issue_count: 1,
                issue_summaries: vec!["inactive".to_string()],
                manifest_hash: Some("hash".to_string()),
                manifest_external_facts_fingerprint: Some("fingerprint".to_string()),
                reference_edge_count: 0,
                exact_compiler_reference_edge_count: 0,
            }),
            evidence_readiness: Some(WorkspaceEvidenceReadinessDiagnostic {
                context_pack_artifact_count: 1,
                context_pack_valid: false,
                context_pack_issue_count: 1,
                tool_episode_count: 1,
                tool_episode_valid: true,
                tool_episode_issue_count: 0,
                episode_artifact_link_valid: true,
                linked_episode_count: 1,
                missing_link_count: 0,
                worst_case_freshness: Some("changed".to_string()),
                issue_summaries: vec!["context_pack:StaleArtifactEvidence".to_string()],
                truncated: false,
            }),
            evidence_ledger_activation: Some(WorkspaceEvidenceLedgerActivationDiagnostic {
                summary: WorkspaceEvidenceLedgerActivationSummary {
                    context_pack_artifact_count: 1,
                    readable_context_pack_count: 1,
                    tool_episode_count: 1,
                    linked_episode_count: 1,
                    missing_link_count: 0,
                    graph_node_count: 2,
                    graph_edge_count: 1,
                    worst_case_freshness: Some("changed".to_string()),
                    issue_count: 1,
                    issue_summaries: vec!["context_pack:StaleArtifactEvidence".to_string()],
                    truncated: false,
                },
                graph: Some(WorkspaceEvidenceLedgerGraphMetadata {
                    node_count: 2,
                    edge_count: 1,
                    node_kind_counts: BTreeMap::from([
                        ("retrieved_evidence".to_string(), 1),
                        ("tool_episode".to_string(), 1),
                    ]),
                    edge_kind_counts: BTreeMap::from([("tool_episode_relates".to_string(), 1)]),
                }),
            }),
        };
        let actual = setup.can_inject();
        let expected = true;

        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn exact_fact_readiness_checks_stay_inside_existing_target_resolution_budget()
    -> Result<()> {
        let fixture = TempDir::new()?;
        let cwd = fixture.path().join("workspace");
        create_indexed_workspace(&cwd)?;
        let setup = ProjectContextHarness::new(cwd.clone());
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let mentions = (0..64)
            .map(|index| {
                fixture
                    .path()
                    .join(format!("unindexed-{index}/src/lib.rs"))
                    .display()
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join(" ");

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .explain(Some(format!(
                "find automatic injection needle in {mentions}"
            )))
            .await;
        let expected = (32usize, 1usize, 1usize);

        assert_eq!(
            (
                setup.index_checks.load(Ordering::SeqCst),
                actual.selected_targets.len(),
                actual
                    .candidates
                    .iter()
                    .filter(|candidate| candidate.skip_reason.as_deref()
                        == Some("candidate limit reached"))
                    .count(),
            ),
            expected,
        );
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_reports_stale_manifest_skip_reason() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new_with_stale_paths(root.clone(), vec![root.clone()]);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let expected = (
            false,
            0usize,
            Some("no fresh project-model manifest target selected".to_string()),
        );

        assert_eq!(
            (
                actual.would_inject,
                actual.selected_targets.len(),
                actual.skip_reason,
            ),
            expected
        );
        assert!(actual.candidates.iter().any(|candidate| {
            candidate
                .skip_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("project-model manifest stale"))
        }));
        let actual_readiness = actual.nearest_skipped_manifest_candidates[0]
            .evidence_readiness
            .as_ref()
            .unwrap();
        assert_eq!(actual_readiness.context_pack_valid, true);
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_nearest_stale_manifest_blocks_parent_fallback() -> Result<()> {
        let fixture = TempDir::new()?;
        let parent = fixture.path().join("parent-workspace");
        create_indexed_workspace(&parent)?;
        let child = parent.join("nested-child");
        create_indexed_workspace(&child)?;
        let setup = ProjectContextHarness::new_with_stale_paths(child.clone(), vec![child]);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let expected = (
            false,
            Vec::<PathBuf>::new(),
            Some("no fresh project-model manifest target selected".to_string()),
        );

        assert_eq!(
            (
                actual.would_inject,
                actual
                    .selected_targets
                    .iter()
                    .map(|target| target.workspace_root.clone())
                    .collect::<Vec<_>>(),
                actual.skip_reason,
            ),
            expected
        );
        assert!(actual.candidates.iter().any(|candidate| {
            candidate
                .skip_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("project-model manifest stale"))
        }));
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_nearest_unknown_manifest_blocks_parent_fallback() -> Result<()> {
        let fixture = TempDir::new()?;
        let parent = fixture.path().join("parent-workspace");
        create_indexed_workspace(&parent)?;
        let child = parent.join("nested-child");
        create_indexed_workspace(&child)?;
        let setup = ProjectContextHarness::new_with_unknown_paths(child.clone(), vec![child]);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let expected = (
            false,
            Vec::<PathBuf>::new(),
            Some("no fresh project-model manifest target selected".to_string()),
        );

        assert_eq!(
            (
                actual.would_inject,
                actual
                    .selected_targets
                    .iter()
                    .map(|target| target.workspace_root.clone())
                    .collect::<Vec<_>>(),
                actual.skip_reason,
            ),
            expected
        );
        assert!(actual.candidates.iter().any(|candidate| {
            candidate
                .skip_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("project-model manifest freshness unknown"))
        }));
        Ok(())
    }

    #[tokio::test]
    async fn explain_context_reports_stale_manifest_details_for_candidate() -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new_with_stale_paths(root.clone(), vec![root]);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id);

        let actual = ProjectContextInjection::new(setup, agent)
            .explain(Some("find automatic injection needle".to_string()))
            .await;
        let actual_reason = actual
            .candidates
            .iter()
            .find_map(|candidate| candidate.skip_reason.as_deref())
            .unwrap_or_default();

        assert!(
            actual_reason.contains("stale") && actual_reason.contains("src/lib.rs"),
            "explain-context must expose stale manifest details instead of a generic no-fresh-target reason; got {actual_reason:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_path_target_scan_is_bounded_for_unindexed_mentions() -> Result<()>
    {
        let fixture = TempDir::new()?;
        let cwd = fixture.path().join("workspace");
        create_indexed_workspace(&cwd)?;
        let setup = ProjectContextHarness::new(cwd.clone());
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let mentions = (0..64)
            .map(|index| {
                fixture
                    .path()
                    .join(format!("unindexed-{index}/src/lib.rs"))
                    .display()
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join(" ");
        let conversation =
            Conversation::generate().context(Context::default().add_message(ContextMessage::user(
                format!("find automatic injection needle in {mentions}"),
                Some(model_id),
            )));

        ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let actual = setup.index_checks.load(Ordering::SeqCst);
        let expected_maximum = ProjectContextInjection::<ProjectContextHarness>::MAX_INDEX_PROBES;
        assert!(
            actual <= expected_maximum,
            "path-aware injection should bound index checks for untrusted path-like prompt text; got {actual}, expected at most {expected_maximum}"
        );
        assert_eq!(setup.workspace_queries.load(Ordering::SeqCst), 0usize);
        let expected_queries = 1usize;
        assert_eq!(
            setup.committed_workspace_queries.load(Ordering::SeqCst),
            expected_queries
        );
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_continues_when_path_target_is_empty() -> Result<()> {
        let fixture = TempDir::new()?;
        let cwd = fixture.path().join("cwd-workspace");
        let other = fixture.path().join("other-workspace");
        create_indexed_workspace(&cwd)?;
        create_indexed_workspace(&other)?;
        let setup = ProjectContextHarness::new_with_empty_paths(cwd.clone(), vec![other.clone()]);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let mentioned_file = other.join("src/lib.rs");
        let conversation =
            Conversation::generate().context(Context::default().add_message(ContextMessage::user(
                format!(
                    "find automatic injection needle in {}",
                    mentioned_file.display()
                ),
                Some(model_id),
            )));

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let expected_workspaces = vec![cwd, other];
        assert_eq!(*setup.queried_workspaces.lock().await, expected_workspaces);
        assert_eq!(
            actual
                .context
                .unwrap()
                .messages
                .iter()
                .filter(|message| message
                    .content()
                    .is_some_and(|content| content.contains("<project_model_context")))
                .count(),
            1usize,
        );
        Ok(())
    }

    #[tokio::test]
    async fn project_model_context_continues_when_path_target_query_errors() -> Result<()> {
        let fixture = TempDir::new()?;
        let cwd = fixture.path().join("cwd-workspace");
        let other = fixture.path().join("other-workspace");
        create_indexed_workspace(&cwd)?;
        create_indexed_workspace(&other)?;
        let setup = ProjectContextHarness::new_with_error_paths(cwd.clone(), vec![other.clone()]);
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone());
        let mentioned_file = other.join("src/lib.rs");
        let conversation =
            Conversation::generate().context(Context::default().add_message(ContextMessage::user(
                format!(
                    "find automatic injection needle in {}",
                    mentioned_file.display()
                ),
                Some(model_id),
            )));

        let actual = ProjectContextInjection::new(setup.clone(), agent)
            .inject(conversation)
            .await;
        let expected_workspaces = vec![cwd, other];
        assert_eq!(*setup.queried_workspaces.lock().await, expected_workspaces);
        assert_eq!(
            actual
                .context
                .unwrap()
                .messages
                .iter()
                .filter(|message| message
                    .content()
                    .is_some_and(|content| content.contains("<project_model_context")))
                .count(),
            1usize,
        );
        Ok(())
    }

    #[test]
    fn project_model_context_stable_identity_changes_when_rendered_stable_bytes_change() {
        let fixture = WorkspaceContextManifestDiagnostic {
            workspace_root: PathBuf::from("/workspace"),
            manifest_path: PathBuf::from("/workspace/.forge_project_model/project_manifest.json"),
            manifest_found: true,
            freshness: WorkspaceContextFreshness::Fresh,
            manifest_hash: Some("fixture-manifest-hash".to_string()),
            exact_fact_readiness: None,
            evidence_readiness: None,
            evidence_ledger_activation: None,
        };
        let setup_nodes = |score| {
            vec![Node {
                node_id: NodeId::new("symbol:src/lib.rs:automatic_injection_needle"),
                node: NodeData::FileChunk(FileChunk {
                    file_path: "src/lib.rs".to_string(),
                    content: "pub fn automatic_injection_needle() -> usize { 42 }".to_string(),
                    start_line: 3,
                    end_line: 3,
                }),
                relevance: Some(score),
                distance: None,
            }]
        };

        let actual_a = ProjectContextInjection::<ProjectContextHarness>::render_context(
            Path::new("/workspace"),
            &fixture,
            None,
            setup_nodes(0.875),
        )
        .unwrap();
        let actual_b = ProjectContextInjection::<ProjectContextHarness>::render_context(
            Path::new("/workspace"),
            &fixture,
            None,
            setup_nodes(0.500),
        )
        .unwrap();
        let identity_a = actual_a
            .stable_payload
            .split("stable_identity=\"")
            .nth(1)
            .unwrap()
            .split('"')
            .next()
            .unwrap();
        let identity_b = actual_b
            .stable_payload
            .split("stable_identity=\"")
            .nth(1)
            .unwrap()
            .split('"')
            .next()
            .unwrap();

        assert_ne!(actual_a.stable_payload, actual_b.stable_payload);
        assert_ne!(identity_a, identity_b);
    }

    #[tokio::test]
    async fn project_model_context_is_injected_into_provider_request_without_sem_search_tool_call()
    -> Result<()> {
        let (_fixture, root) = fixture_workspace()?;
        let setup = ProjectContextHarness::new(root.clone());
        let model_id = ModelId::new("test-model");
        let agent = Agent::new(AgentId::new("forge"), ProviderId::OPENAI, model_id.clone())
            .tool_supported(false)
            .max_requests_per_turn(1usize);
        let conversation = Conversation::generate().context(Context::default().add_message(
            ContextMessage::user("find automatic injection needle", Some(model_id.clone())),
        ));
        let conversation = ProjectContextInjection::new(setup.clone(), agent.clone())
            .inject(conversation)
            .await;
        let mut orch =
            Orchestrator::new(setup.clone(), conversation, agent, ForgeConfig::default())
                .models(vec![
                    Model::new(ProviderId::OPENAI, model_id).context_length(200_000_u64),
                ])
                .tool_definitions(Vec::new());

        orch.run().await?;
        let captured_context = setup.captured_context.lock().await.clone().unwrap();
        let project_context_message = captured_context
            .messages
            .iter()
            .find(|message| {
                message
                    .content()
                    .is_some_and(|content| content.contains("<project_model_context"))
            })
            .unwrap();
        let actual = project_context_message.content().unwrap().to_string();
        let actual_flags = vec![
            actual.contains("manifest_path"),
            actual.contains("src/lib.rs"),
            actual.contains("start_line=\"3\""),
            actual.contains("score=\"0.875000\""),
            actual.contains("content_digest=\""),
            actual.contains("truncated_reason=\"content_line_budget_exceeded\"")
                || actual.contains("truncated_reason=\"content_char_budget_exceeded\""),
            actual.contains("omitted_reason=\"whole_file_metadata_only\""),
            actual.contains("full_file_should_not_render"),
            actual.contains("extra_should_not_render"),
            project_context_message.is_cache_eligible(),
            captured_context
                .tools
                .iter()
                .any(|tool| tool.name.as_str().eq_ignore_ascii_case("sem_search")),
        ];
        let expected_flags = vec![
            true, true, true, true, true, true, true, false, false, true, false,
        ];
        assert_eq!(actual_flags, expected_flags);
        assert_eq!(actual.matches("<source").count(), 3usize);
        assert_eq!(setup.workspace_queries.load(Ordering::SeqCst), 0usize);
        assert_eq!(
            setup.committed_workspace_queries.load(Ordering::SeqCst),
            1usize
        );
        assert!(
            actual.chars().count() <= ProjectModelContextRenderBudget::default().max_rendered_chars,
            "project-model context should stay inside the typed render budget"
        );
        Ok(())
    }

    #[tokio::test]
    async fn project_model_query_ignores_droppable_user_messages() -> Result<()> {
        let (_fixture, _root) = fixture_workspace()?;
        let model_id = ModelId::new("test-model");
        let conversation = Conversation::generate().context(
            Context::default()
                .add_message(ContextMessage::user(
                    "find automatic injection needle",
                    Some(model_id.clone()),
                ))
                .add_message(ContextMessage::Text(
                    TextMessage::new(Role::User, "ignore automatic injection needle")
                        .model(model_id)
                        .droppable(true),
                )),
        );

        let actual = ProjectContextInjection::<ProjectContextHarness>::query_from_conversation(
            &conversation,
        )
        .unwrap();
        let expected = "find automatic injection needle";
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn project_model_query_ignores_internal_project_model_context_messages() -> Result<()> {
        let (_fixture, _root) = fixture_workspace()?;
        let model_id = ModelId::new("test-model");
        let conversation = Conversation::generate().context(
            Context::default()
                .add_message(ContextMessage::user(
                    "find automatic injection needle",
                    Some(model_id.clone()),
                ))
                .add_message(ContextMessage::Text(
                    TextMessage::project_model_context(
                        Role::User,
                        "<project_model_context>ignore automatic injection needle</project_model_context>",
                    )
                    .model(model_id)
                    .droppable(false),
                )),
        );

        let actual = ProjectContextInjection::<ProjectContextHarness>::query_from_conversation(
            &conversation,
        )
        .unwrap();
        let expected = "find automatic injection needle";
        assert_eq!(actual, expected);
        Ok(())
    }
}
