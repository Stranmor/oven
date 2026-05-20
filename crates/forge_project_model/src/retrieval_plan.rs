//! Pure retrieval execution planning for project-model workspace queries.

use std::cmp::Ordering;
use std::path::{Component, Path};

use anyhow::{Result, bail};

use crate::context_adapter::resolve_manifest_evidence_target;
use crate::retrieval::retrieve;
use crate::types::{
    ContextPack, ContextPackEvidence, ContextPackSelection, FreshnessProofLevel,
    ManifestFreshnessEvaluation, ProjectManifest, RetrievalQuery, RetrievalResult,
    StaleEvidencePolicy,
};

const MAX_DIAGNOSTIC_SUMMARIES: usize = 8;

/// Query request accepted by the project-model retrieval planner.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProjectContextRetrievalRequest {
    /// Free-form retrieval text.
    pub query_text: String,
    /// Maximum number of returned retrieval results.
    pub limit: usize,
    /// Optional full path scope applied before truncation.
    pub path_scope: ProjectContextPathScope,
    /// Whether graph expansion should participate in deterministic retrieval.
    pub include_graph_expansion: bool,
}

impl ProjectContextRetrievalRequest {
    /// Builds a project-context retrieval request.
    ///
    /// # Arguments
    ///
    /// * `query_text` - Free-form retrieval text.
    /// * `limit` - Maximum number of retrieval results.
    /// * `path_scope` - Full path scope applied before truncation.
    /// * `include_graph_expansion` - Whether graph expansion is enabled.
    pub fn new(
        query_text: impl Into<String>,
        limit: usize,
        path_scope: ProjectContextPathScope,
        include_graph_expansion: bool,
    ) -> Self {
        Self {
            query_text: query_text.into(),
            limit,
            path_scope,
            include_graph_expansion,
        }
    }
}

/// Full project-model path scope applied before retrieval truncation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProjectContextPathScope {
    /// Optional manifest-relative path prefix.
    pub starts_with: Option<String>,
    /// Optional manifest-relative path suffix set.
    pub ends_with: Vec<String>,
}

impl ProjectContextPathScope {
    /// Builds a path scope from optional prefix and suffix filters.
    ///
    /// # Arguments
    ///
    /// * `starts_with` - Optional manifest-relative prefix.
    /// * `ends_with` - Optional suffix filters.
    pub fn new(starts_with: Option<String>, ends_with: Vec<String>) -> Self {
        Self { starts_with, ends_with }
    }

    fn matches(&self, path: &str) -> bool {
        if let Some(prefix) = &self.starts_with
            && !path.starts_with(prefix)
        {
            return false;
        }
        if !self.ends_with.is_empty() && !self.ends_with.iter().any(|suffix| path.ends_with(suffix))
        {
            return false;
        }
        true
    }
}

/// Pure planner refusal for project-context retrieval.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectContextRetrievalRefusal {
    /// Stable machine-readable refusal code.
    pub code: ProjectContextRetrievalRefusalCode,
    /// Human-readable redaction-safe refusal detail.
    pub detail: String,
}

/// Stable project-context retrieval refusal code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectContextRetrievalRefusalCode {
    /// Manifest freshness is not strong enough for injection/query retrieval.
    ManifestNotInjectable,
    /// Context pack construction rejected stale evidence.
    StaleEvidenceRejected,
    /// Evidence line range could not be resolved from the manifest.
    EvidenceRangeMissing,
    /// Evidence path failed read-request validation.
    InvalidReadRequestPath,
}

/// Pure retrieval planning result.
#[derive(Clone, Debug, PartialEq)]
pub enum ProjectContextRetrievalPlanningOutcome {
    /// Retrieval is refused before any read or write side effect.
    Refusal(ProjectContextRetrievalRefusal),
    /// Retrieval is planned and safe for IO execution by a service boundary.
    Plan(Box<ProjectContextRetrievalPlan>),
}

/// Redaction-safe pure diagnostic projection for a retrieval planning outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectContextRetrievalPlanDiagnostic {
    /// Whether the planner produced an executable plan.
    pub planned: bool,
    /// Stable machine-readable refusal code when planning was refused.
    pub refusal_code: Option<ProjectContextRetrievalRefusalCode>,
    /// Human-readable redaction-safe refusal detail when planning was refused.
    pub refusal_detail: Option<String>,
    /// Number of retrieval results selected by the planner.
    pub selected_result_count: usize,
    /// Number of validated read requests planned before readback.
    pub read_request_count: usize,
    /// Deterministic write decision when planning succeeded.
    pub write_decision: Option<ProjectContextWriteDecision>,
    /// Bounded metadata-only summaries of selected retrieval results.
    pub selected_summaries: Vec<ProjectContextRetrievalSelectedSummary>,
    /// Bounded metadata-only summaries of planned read requests.
    pub read_request_summaries: Vec<ProjectContextRetrievalReadRequestSummary>,
    /// Whether retrieval selected no evidence.
    pub retrieval_empty: bool,
    /// Whether selected or read-request summaries were truncated.
    pub truncated: bool,
}

impl ProjectContextRetrievalPlanDiagnostic {
    /// Builds a redaction-safe diagnostic projection from a pure planning outcome.
    ///
    /// # Arguments
    ///
    /// * `outcome` - Pure retrieval planning outcome to project into bounded diagnostics.
    pub fn from_outcome(outcome: &ProjectContextRetrievalPlanningOutcome) -> Self {
        match outcome {
            ProjectContextRetrievalPlanningOutcome::Refusal(refusal) => Self {
                planned: false,
                refusal_code: Some(refusal.code.clone()),
                refusal_detail: Some(refusal.detail.clone()),
                selected_result_count: 0,
                read_request_count: 0,
                write_decision: None,
                selected_summaries: Vec::new(),
                read_request_summaries: Vec::new(),
                retrieval_empty: false,
                truncated: false,
            },
            ProjectContextRetrievalPlanningOutcome::Plan(plan) => {
                let selected_result_count = plan.selected_results.len();
                let read_request_count = plan.read_requests.len();
                let selected_summaries = plan
                    .selected_results
                    .iter()
                    .take(MAX_DIAGNOSTIC_SUMMARIES)
                    .map(ProjectContextRetrievalSelectedSummary::from_result)
                    .collect::<Vec<_>>();
                let read_request_summaries = plan
                    .read_requests
                    .iter()
                    .take(MAX_DIAGNOSTIC_SUMMARIES)
                    .map(ProjectContextRetrievalReadRequestSummary::from_request)
                    .collect::<Vec<_>>();
                Self {
                    planned: true,
                    refusal_code: None,
                    refusal_detail: None,
                    selected_result_count,
                    read_request_count,
                    write_decision: Some(plan.write_decision.clone()),
                    selected_summaries,
                    read_request_summaries,
                    retrieval_empty: selected_result_count == 0,
                    truncated: selected_result_count > MAX_DIAGNOSTIC_SUMMARIES
                        || read_request_count > MAX_DIAGNOSTIC_SUMMARIES,
                }
            }
        }
    }
}

/// Metadata-only selected-result summary for retrieval diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectContextRetrievalSelectedSummary {
    /// Evidence identifier selected by retrieval.
    pub evidence_id: String,
    /// Manifest-relative path associated with the selected result.
    pub path: String,
    /// Optional one-based inclusive start line.
    pub start_line: Option<u32>,
    /// Optional one-based inclusive end line.
    pub end_line: Option<u32>,
    /// Planner relevance score.
    pub relevance: f32,
}

impl ProjectContextRetrievalSelectedSummary {
    fn from_result(result: &RetrievalResult) -> Self {
        Self {
            evidence_id: result.id.clone(),
            path: result.path.clone(),
            start_line: result.provenance.start_line,
            end_line: result.provenance.end_line,
            relevance: result.score,
        }
    }
}

/// Metadata-only read-request summary for retrieval diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectContextRetrievalReadRequestSummary {
    /// Evidence identifier planned for readback.
    pub evidence_id: String,
    /// Manifest-relative path planned for readback.
    pub path: String,
    /// One-based inclusive start line.
    pub start_line: u32,
    /// One-based inclusive end line.
    pub end_line: u32,
}

impl ProjectContextRetrievalReadRequestSummary {
    fn from_request(request: &ProjectContextReadRequest) -> Self {
        Self {
            evidence_id: request.evidence_id.clone(),
            path: request.relative_manifest_path().to_string(),
            start_line: request.start_line,
            end_line: request.end_line,
        }
    }
}

/// Pure project-context retrieval execution plan.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectContextRetrievalPlan {
    /// Diagnostics for the retrieval query created by the planner.
    pub query_diagnostics: ProjectContextRetrievalQueryDiagnostics,
    /// Selected retrieval results before context-pack ordering.
    pub selected_results: Vec<RetrievalResult>,
    /// Deterministic context pack intended for persistence after readback.
    pub context_pack: Option<ContextPack>,
    /// Validated read requests to execute before writing the pack.
    pub read_requests: Vec<ProjectContextReadRequest>,
    /// Deterministic write decision.
    pub write_decision: ProjectContextWriteDecision,
    /// Stable return ordering independent of context-pack evidence order.
    pub return_order: Vec<ProjectContextReturnOrderItem>,
}

/// Diagnostics for a project-context retrieval query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectContextRetrievalQueryDiagnostics {
    /// Query text supplied to retrieval.
    pub query_text: Option<String>,
    /// Optional prefix scope supplied to retrieval.
    pub path_prefix: Option<String>,
    /// Suffix scope applied before truncation by the planner.
    pub path_suffixes: Vec<String>,
    /// Effective retrieval limit.
    pub limit: usize,
    /// Whether graph expansion was requested.
    pub include_graph_expansion: bool,
    /// Fixed stale policy used for query path injection.
    pub stale_policy: StaleEvidencePolicy,
    /// Freshness proof level used for injection gating.
    pub freshness_proof_level: FreshnessProofLevel,
}

/// Deterministic write decision for project-context retrieval.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectContextWriteDecision {
    /// No readback or pack write should occur because retrieval selected no
    /// evidence.
    NoWriteEmptyRetrieval,
    /// Persist the context pack after every readback succeeds.
    WriteContextPackAfterReadback,
}

/// Stable metadata item describing service return order.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectContextReturnOrderItem {
    /// Evidence identifier returned to the service as a node id.
    pub evidence_id: String,
    /// Relevance score used for stable ordering.
    pub relevance: f32,
}

/// Validated manifest-relative read request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectContextReadRequest {
    relative_manifest_path: String,
    /// Evidence identifier read from this path.
    pub evidence_id: String,
    /// One-based inclusive start line.
    pub start_line: u32,
    /// One-based inclusive end line.
    pub end_line: u32,
}

impl ProjectContextReadRequest {
    /// Builds a validated manifest-relative read request.
    ///
    /// # Arguments
    ///
    /// * `relative_manifest_path` - Manifest-relative path using safe normal
    ///   path components.
    /// * `evidence_id` - Evidence identifier to associate with the readback.
    /// * `start_line` - One-based inclusive start line.
    /// * `end_line` - One-based inclusive end line.
    ///
    /// # Errors
    ///
    /// Returns an error for absolute paths, empty paths, parent components,
    /// traversal components, platform prefixes, or invalid line ranges.
    pub fn new(
        relative_manifest_path: impl Into<String>,
        evidence_id: impl Into<String>,
        start_line: u32,
        end_line: u32,
    ) -> Result<Self> {
        let relative_manifest_path = relative_manifest_path.into();
        validate_manifest_relative_path(&relative_manifest_path)?;
        if start_line == 0 || end_line < start_line {
            bail!("project context read request line range is invalid");
        }
        Ok(Self {
            relative_manifest_path,
            evidence_id: evidence_id.into(),
            start_line,
            end_line,
        })
    }

    /// Returns the validated manifest-relative path.
    pub fn relative_manifest_path(&self) -> &str {
        &self.relative_manifest_path
    }
}

/// Plans project-context retrieval without performing IO.
///
/// # Arguments
///
/// * `manifest` - Project manifest used for retrieval and evidence resolution.
/// * `freshness` - Freshness evaluation used for injection gating and stale
///   evidence checks.
/// * `request` - Typed retrieval request.
pub fn plan_project_context_retrieval(
    manifest: &ProjectManifest,
    freshness: &ManifestFreshnessEvaluation,
    request: ProjectContextRetrievalRequest,
) -> ProjectContextRetrievalPlanningOutcome {
    let effective_limit = if request.limit == 0 {
        10
    } else {
        request.limit
    };
    let diagnostics = ProjectContextRetrievalQueryDiagnostics {
        query_text: Some(request.query_text.clone()),
        path_prefix: request.path_scope.starts_with.clone(),
        path_suffixes: request.path_scope.ends_with.clone(),
        limit: effective_limit,
        include_graph_expansion: request.include_graph_expansion,
        stale_policy: StaleEvidencePolicy::Reject,
        freshness_proof_level: freshness.proof_level.clone(),
    };
    if !freshness.can_inject() {
        return ProjectContextRetrievalPlanningOutcome::Refusal(ProjectContextRetrievalRefusal {
            code: ProjectContextRetrievalRefusalCode::ManifestNotInjectable,
            detail: "project-model manifest is not fully fresh for injection".to_string(),
        });
    }

    let query = RetrievalQuery {
        text: diagnostics.query_text.clone(),
        path: None,
        path_prefix: diagnostics.path_prefix.clone(),
        symbol: None,
        limit: usize::MAX,
        include_graph_expansion: diagnostics.include_graph_expansion,
    };
    let mut selected_results = retrieve(manifest, &query)
        .into_iter()
        .filter(|result| request.path_scope.matches(&result.path))
        .collect::<Vec<_>>();
    selected_results.sort_by(compare_retrieval_results_for_return);
    selected_results.truncate(effective_limit);

    if selected_results.is_empty() {
        return ProjectContextRetrievalPlanningOutcome::Plan(Box::new(
            ProjectContextRetrievalPlan {
                query_diagnostics: diagnostics,
                selected_results,
                context_pack: None,
                read_requests: Vec::new(),
                write_decision: ProjectContextWriteDecision::NoWriteEmptyRetrieval,
                return_order: Vec::new(),
            },
        ));
    }

    let context_pack = match ContextPack::from_selection(
        manifest,
        ContextPackSelection {
            retrieval_results: selected_results.clone(),
            shards: Vec::new(),
            evidence: Vec::new(),
            freshness: freshness.state.clone(),
            stale_policy: StaleEvidencePolicy::Reject,
        },
    ) {
        Ok(context_pack) => context_pack,
        Err(error) => {
            return ProjectContextRetrievalPlanningOutcome::Refusal(
                ProjectContextRetrievalRefusal {
                    code: ProjectContextRetrievalRefusalCode::StaleEvidenceRejected,
                    detail: error.to_string(),
                },
            );
        }
    };

    let mut read_requests = Vec::new();
    for evidence in &context_pack.evidence {
        let target = match resolve_manifest_evidence_target(manifest, &evidence.id) {
            Some(target) => target,
            None => {
                return ProjectContextRetrievalPlanningOutcome::Refusal(
                    ProjectContextRetrievalRefusal {
                        code: ProjectContextRetrievalRefusalCode::EvidenceRangeMissing,
                        detail: format!(
                            "project-model evidence line range is missing: {}",
                            evidence.id
                        ),
                    },
                );
            }
        };
        let (start_line, end_line) = match target.line_range {
            Some(line_range) => line_range,
            None => {
                return ProjectContextRetrievalPlanningOutcome::Refusal(
                    ProjectContextRetrievalRefusal {
                        code: ProjectContextRetrievalRefusalCode::EvidenceRangeMissing,
                        detail: format!(
                            "project-model evidence line range is missing: {}",
                            evidence.id
                        ),
                    },
                );
            }
        };
        match ProjectContextReadRequest::new(target.path, evidence.id.clone(), start_line, end_line)
        {
            Ok(read_request) => read_requests.push(read_request),
            Err(error) => {
                return ProjectContextRetrievalPlanningOutcome::Refusal(
                    ProjectContextRetrievalRefusal {
                        code: ProjectContextRetrievalRefusalCode::InvalidReadRequestPath,
                        detail: error.to_string(),
                    },
                );
            }
        }
    }
    let return_order = stable_return_order(&context_pack.evidence);

    ProjectContextRetrievalPlanningOutcome::Plan(Box::new(ProjectContextRetrievalPlan {
        query_diagnostics: diagnostics,
        selected_results,
        context_pack: Some(context_pack),
        read_requests,
        write_decision: ProjectContextWriteDecision::WriteContextPackAfterReadback,
        return_order,
    }))
}

fn stable_return_order(evidence: &[ContextPackEvidence]) -> Vec<ProjectContextReturnOrderItem> {
    let mut items = evidence
        .iter()
        .map(|evidence| ProjectContextReturnOrderItem {
            evidence_id: evidence.id.clone(),
            relevance: evidence.score,
        })
        .collect::<Vec<_>>();
    items.sort_by(|left, right| {
        right
            .relevance
            .total_cmp(&left.relevance)
            .then_with(|| left.evidence_id.cmp(&right.evidence_id))
    });
    items
}

fn compare_retrieval_results_for_return(
    left: &RetrievalResult,
    right: &RetrievalResult,
) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.id.cmp(&right.id))
}

fn validate_manifest_relative_path(path: &str) -> Result<()> {
    if path.is_empty() || path.trim().is_empty() {
        bail!("project context read path must not be empty");
    }
    if path.contains('\\') || path.contains('\0') {
        bail!("project context read path contains unsupported separators");
    }
    let parsed = Path::new(path);
    if parsed.is_absolute() {
        bail!("project context read path must be relative");
    }
    for segment in path.split('/') {
        if segment.is_empty() {
            bail!("project context read path contains empty component");
        }
        if segment == "." || segment == ".." {
            bail!("project context read path contains traversal");
        }
    }
    let mut has_component = false;
    for component in parsed.components() {
        match component {
            Component::Normal(value) if !value.is_empty() => {
                has_component = true;
            }
            Component::Normal(_) => {
                bail!("project context read path contains empty component");
            }
            Component::CurDir
            | Component::ParentDir
            | Component::RootDir
            | Component::Prefix(_) => {
                bail!("project context read path contains traversal");
            }
        }
    }
    if !has_component {
        bail!("project context read path must contain a normal component");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::indexer::tests::fixture_project;
    use crate::types::{
        CargoDependencyDeclaration, CargoDependencyKind, CargoPackageDependency,
        CargoPackageMetadata, CargoTargetDeclaration, CargoTargetKind, CargoTargetMetadata,
        FreshnessState, Language, Provenance, SourceFile,
    };
    use crate::{ProjectIndexer, ShardManifest, SymbolKind, SymbolNode, fingerprint};

    #[test]
    fn planner_builds_whole_file_read_request_for_cargo_metadata_evidence() {
        let setup = cargo_plan_manifest();
        let request = ProjectContextRetrievalRequest::new(
            "fixture_app package",
            1,
            ProjectContextPathScope::default(),
            false,
        );

        let actual = plan_project_context_retrieval(&setup, &freshness(&setup), request);
        let expected = Some(("Cargo.toml".to_string(), 1u32, 16u32));
        assert_eq!(
            match actual {
                ProjectContextRetrievalPlanningOutcome::Plan(plan) =>
                    plan.read_requests.first().map(|request| {
                        (
                            request.relative_manifest_path().to_string(),
                            request.start_line,
                            request.end_line,
                        )
                    }),
                ProjectContextRetrievalPlanningOutcome::Refusal(refusal) => {
                    panic!("unexpected refusal: {:?}", refusal)
                }
            },
            expected,
        );
    }

    #[test]
    fn invalid_or_suspicious_cargo_metadata_ids_do_not_produce_read_requests() {
        let setup = cargo_plan_manifest();
        let candidates = [
            "cargo:dependency:../x",
            "cargo:v1:dependency:../x",
            "cargo:v1:unknown:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "cargo:v1:dependency:raw:separator:collision",
        ];
        let actual = candidates
            .into_iter()
            .map(|id| resolve_manifest_evidence_target(&setup, id).is_none())
            .collect::<Vec<_>>();
        let expected = vec![true, true, true, true];
        assert_eq!(actual, expected);
    }

    #[test]
    fn stale_freshness_refusal_is_unchanged_for_cargo_metadata_hits() {
        let setup = cargo_plan_manifest();
        let evaluation = ManifestFreshnessEvaluation {
            state: FreshnessState {
                changed: vec!["Cargo.toml".to_string()],
                fresh: true,
                ..fresh_state(&setup)
            },
            proof_level: FreshnessProofLevel::FullFilesystem,
        };
        let request = ProjectContextRetrievalRequest::new(
            "fixture_app package",
            1,
            ProjectContextPathScope::default(),
            false,
        );

        let actual = plan_project_context_retrieval(&setup, &evaluation, request);
        let expected = Some(ProjectContextRetrievalRefusalCode::StaleEvidenceRejected);
        assert_eq!(refusal_code(actual), expected);
    }

    #[test]
    fn cargo_like_normal_ids_cannot_shadow_valid_metadata_ids() {
        let mut setup = cargo_plan_manifest();
        setup.files.push(SourceFile {
            path: "src/cargo.rs".to_string(),
            language: Language::Rust,
            bytes: 10,
            lines: 2,
            content_hash: fingerprint("cargo-like-source"),
            provenance: provenance("src/cargo.rs", Some(1), Some(2), "test", "file"),
        });
        setup.symbols.push(SymbolNode {
            id: "symbol:src/cargo.rs:Function:cargo:v1:package:fake".to_string(),
            name: "cargo:v1:package:fake".to_string(),
            kind: SymbolKind::Function,
            path: "src/cargo.rs".to_string(),
            parent: None,
            start_line: 1,
            end_line: 2,
            provenance: provenance("src/cargo.rs", Some(1), Some(2), "test", "symbol"),
        });
        setup.shards.push(ShardManifest {
            id: "shard:cargo:v1:package:fake".to_string(),
            path: "src/cargo.rs".to_string(),
            start_line: 1,
            end_line: 2,
            content_hash: fingerprint("cargo-like-shard"),
            symbol_ids: Vec::new(),
            provenance: provenance("src/cargo.rs", Some(1), Some(2), "test", "shard"),
        });

        let actual = (
            resolve_manifest_evidence_target(&setup, "cargo:v1:package:fake").is_none(),
            resolve_manifest_evidence_target(
                &setup,
                "symbol:src/cargo.rs:Function:cargo:v1:package:fake",
            )
            .map(|target| target.path),
            resolve_manifest_evidence_target(&setup, "shard:cargo:v1:package:fake")
                .map(|target| target.path),
        );
        let expected = (
            true,
            Some("src/cargo.rs".to_string()),
            Some("src/cargo.rs".to_string()),
        );
        assert_eq!(actual, expected);
    }
    #[test]
    fn planner_applies_prefix_and_suffix_before_truncation_without_underfill() -> Result<()> {
        let fixture = tempfile::TempDir::new()?;
        let root = fixture.path().join("project");
        std::fs::create_dir_all(root.join("src").join("in"))?;
        std::fs::create_dir_all(root.join("src").join("out"))?;
        std::fs::write(
            root.join("src").join("in").join("target.rs"),
            "pub fn target() {\n    let _ = \"scopedneedle\";\n}\n",
        )?;
        std::fs::write(
            root.join("src").join("out").join("loud.rs"),
            "pub fn loud() {\n    let _ = \"scopedneedle scopedneedle scopedneedle scopedneedle scopedneedle\";\n}\n",
        )?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let request = ProjectContextRetrievalRequest::new(
            "target Function",
            1,
            ProjectContextPathScope::new(Some("src/in/".to_string()), vec![".rs".to_string()]),
            true,
        );

        let actual = plan_project_context_retrieval(&manifest, &freshness(&manifest), request);
        let expected = vec!["src/in/target.rs".to_string()];
        assert_eq!(planned_paths(actual), expected);
        Ok(())
    }

    #[test]
    fn planner_preserves_limit_zero_as_retrieval_default_limit() -> Result<()> {
        let (_fixture, root, manifest) = indexed_fixture()?;
        let request = ProjectContextRetrievalRequest::new(
            "Root",
            0,
            ProjectContextPathScope::default(),
            true,
        );

        let actual = plan_project_context_retrieval(&manifest, &freshness(&manifest), request);
        let expected = (
            ProjectContextWriteDecision::WriteContextPackAfterReadback,
            10usize,
            false,
        );
        let actual = match actual {
            ProjectContextRetrievalPlanningOutcome::Plan(plan) => {
                assert!(plan.selected_results.len() <= 10);
                assert!(!plan.read_requests.is_empty());
                (
                    plan.write_decision,
                    plan.query_diagnostics.limit,
                    plan.context_pack.is_none(),
                )
            }
            ProjectContextRetrievalPlanningOutcome::Refusal(refusal) => {
                panic!("unexpected refusal: {:?}", refusal)
            }
        };
        assert_eq!(actual, expected);
        assert!(root.is_dir());
        Ok(())
    }

    #[test]
    fn planner_refuses_indexed_files_only_freshness_even_when_state_is_fresh() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let evaluation = ManifestFreshnessEvaluation {
            state: fresh_state(&manifest),
            proof_level: FreshnessProofLevel::IndexedFilesOnly,
        };
        let request = ProjectContextRetrievalRequest::new(
            "Root",
            3,
            ProjectContextPathScope::default(),
            true,
        );

        let actual = plan_project_context_retrieval(&manifest, &evaluation, request);
        let expected = Some(ProjectContextRetrievalRefusalCode::ManifestNotInjectable);
        assert_eq!(refusal_code(actual), expected);
        Ok(())
    }

    #[test]
    fn planner_refuses_changed_stale_evidence_under_reject_policy() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let evaluation = ManifestFreshnessEvaluation {
            state: FreshnessState {
                changed: vec!["src/lib.rs".to_string()],
                fresh: true,
                ..fresh_state(&manifest)
            },
            proof_level: FreshnessProofLevel::FullFilesystem,
        };
        let request = ProjectContextRetrievalRequest::new(
            "Root",
            3,
            ProjectContextPathScope::default(),
            true,
        );

        let actual = plan_project_context_retrieval(&manifest, &evaluation, request);
        let expected = Some(ProjectContextRetrievalRefusalCode::StaleEvidenceRejected);
        assert_eq!(refusal_code(actual), expected);
        Ok(())
    }

    #[test]
    fn planner_refuses_deleted_stale_evidence_under_reject_policy() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let evaluation = ManifestFreshnessEvaluation {
            state: FreshnessState {
                deleted: vec!["src/lib.rs".to_string()],
                fresh: true,
                ..fresh_state(&manifest)
            },
            proof_level: FreshnessProofLevel::FullFilesystem,
        };
        let request = ProjectContextRetrievalRequest::new(
            "Root",
            3,
            ProjectContextPathScope::default(),
            true,
        );

        let actual = plan_project_context_retrieval(&manifest, &evaluation, request);
        let expected = Some(ProjectContextRetrievalRefusalCode::StaleEvidenceRejected);
        assert_eq!(refusal_code(actual), expected);
        Ok(())
    }

    #[test]
    fn planner_empty_retrieval_gives_no_write_no_read_plan() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let request = ProjectContextRetrievalRequest::new(
            "absent-token-for-empty-plan",
            5,
            ProjectContextPathScope::new(
                Some("absent/scope/".to_string()),
                vec![".rs".to_string()],
            ),
            true,
        );

        let actual = plan_project_context_retrieval(&manifest, &freshness(&manifest), request);
        let expected = (
            ProjectContextWriteDecision::NoWriteEmptyRetrieval,
            0usize,
            true,
        );
        assert_eq!(
            match actual {
                ProjectContextRetrievalPlanningOutcome::Plan(plan) => (
                    plan.write_decision,
                    plan.read_requests.len(),
                    plan.context_pack.is_none(),
                ),
                ProjectContextRetrievalPlanningOutcome::Refusal(refusal) => {
                    panic!("unexpected refusal: {:?}", refusal)
                }
            },
            expected,
        );
        Ok(())
    }

    #[test]
    fn read_request_plans_symbol_shard_and_file_evidence_ranges() -> Result<()> {
        let mut manifest = manual_manifest();
        manifest.symbols.push(SymbolNode {
            id: "symbol:src/lib.rs:Function:target".to_string(),
            name: "target".to_string(),
            kind: SymbolKind::Function,
            path: "src/lib.rs".to_string(),
            parent: None,
            start_line: 2,
            end_line: 4,
            provenance: provenance("src/lib.rs", Some(2), Some(4), "test", "symbol"),
        });
        manifest.shards.push(ShardManifest {
            id: "shard:src/lib.rs:5-7".to_string(),
            path: "src/lib.rs".to_string(),
            start_line: 5,
            end_line: 7,
            content_hash: fingerprint("shard"),
            symbol_ids: Vec::new(),
            provenance: provenance("src/lib.rs", Some(5), Some(7), "test", "shard"),
        });
        let selection = ContextPackSelection {
            retrieval_results: Vec::new(),
            shards: manifest.shards.clone(),
            evidence: vec![
                ContextPackEvidence {
                    id: "src/lib.rs".to_string(),
                    path: "src/lib.rs".to_string(),
                    symbol: None,
                    source: crate::ContextPackEvidenceSource::DirectEvidence,
                    freshness: crate::EvidenceFreshness::Fresh,
                    provenance: provenance("src/lib.rs", Some(1), Some(9), "test", "file"),
                    score: 1.0,
                },
                ContextPackEvidence {
                    id: "symbol:src/lib.rs:Function:target".to_string(),
                    path: "src/lib.rs".to_string(),
                    symbol: Some("target".to_string()),
                    source: crate::ContextPackEvidenceSource::DirectEvidence,
                    freshness: crate::EvidenceFreshness::Fresh,
                    provenance: provenance("src/lib.rs", Some(2), Some(4), "test", "symbol-direct"),
                    score: 3.0,
                },
            ],
            freshness: fresh_state(&manifest),
            stale_policy: StaleEvidencePolicy::Reject,
        };
        let pack = ContextPack::from_selection(&manifest, selection)?;
        let actual = pack
            .evidence
            .iter()
            .map(|evidence| {
                let target = resolve_manifest_evidence_target(&manifest, &evidence.id).unwrap();
                let (start, end) = target.line_range.unwrap();
                ProjectContextReadRequest::new(target.path, evidence.id.clone(), start, end)
                    .unwrap()
            })
            .map(|request| (request.evidence_id, request.start_line, request.end_line))
            .collect::<Vec<_>>();
        let expected = vec![
            ("shard:src/lib.rs:5-7".to_string(), 5, 7),
            ("src/lib.rs".to_string(), 1, 9),
            ("symbol:src/lib.rs:Function:target".to_string(), 2, 4),
        ];
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn read_request_rejects_path_traversal_and_absolute_or_empty_paths() {
        let setup = [
            "../secret",
            "/tmp/secret",
            "",
            ".",
            "safe/./secret",
            "safe/../secret",
            "safe\\..\\secret",
        ];
        let actual = setup
            .into_iter()
            .map(|path| ProjectContextReadRequest::new(path, "id", 1, 1).is_err())
            .collect::<Vec<_>>();
        let expected = vec![true, true, true, true, true, true, true];
        assert_eq!(actual, expected);
    }

    #[test]
    fn planner_return_order_is_stable_independent_of_pack_order() -> Result<()> {
        let (_fixture, _root, manifest) = indexed_fixture()?;
        let request = ProjectContextRetrievalRequest::new(
            "Root model",
            5,
            ProjectContextPathScope::default(),
            true,
        );

        let actual = plan_project_context_retrieval(&manifest, &freshness(&manifest), request);
        let plan = match actual {
            ProjectContextRetrievalPlanningOutcome::Plan(plan) => plan,
            ProjectContextRetrievalPlanningOutcome::Refusal(refusal) => {
                panic!("unexpected refusal: {:?}", refusal)
            }
        };
        let expected_return = {
            let mut values = plan
                .context_pack
                .as_ref()
                .unwrap()
                .evidence
                .iter()
                .map(|evidence| (evidence.id.clone(), evidence.score))
                .collect::<Vec<_>>();
            values.sort_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            values
        };
        let actual_return = plan
            .return_order
            .iter()
            .map(|item| (item.evidence_id.clone(), item.relevance))
            .collect::<Vec<_>>();
        let pack_order = plan
            .context_pack
            .as_ref()
            .unwrap()
            .evidence
            .iter()
            .map(|evidence| evidence.id.clone())
            .collect::<Vec<_>>();
        let return_ids = plan
            .return_order
            .iter()
            .map(|item| item.evidence_id.clone())
            .collect::<Vec<_>>();
        assert_eq!(actual_return, expected_return);
        assert_ne!(return_ids, pack_order);
        Ok(())
    }

    fn cargo_plan_manifest() -> ProjectManifest {
        ProjectManifest {
            version: 1,
            root: "/workspace".into(),
            files: vec![SourceFile {
                path: "Cargo.toml".to_string(),
                language: Language::Toml,
                bytes: 200,
                lines: 16,
                content_hash: fingerprint("cargo-plan-toml"),
                provenance: provenance("Cargo.toml", Some(1), Some(16), "test", "file"),
            }],
            cargo_packages: vec![CargoPackageMetadata {
                manifest_path: "Cargo.toml".to_string(),
                package_root: "".to_string(),
                name: "fixture_app".to_string(),
                version: Some("0.1.0".to_string()),
                edition: Some("2021".to_string()),
                targets: vec![CargoTargetMetadata {
                    name: "fixture_bin".to_string(),
                    kind: CargoTargetKind::Bin,
                    path: "src/main.rs".to_string(),
                    declaration: CargoTargetDeclaration::Declared,
                    provenance: provenance("Cargo.toml", Some(8), Some(10), "test", "target"),
                }],
                features: Vec::new(),
                provenance: provenance("Cargo.toml", Some(1), Some(5), "test", "package"),
            }],
            cargo_package_dependencies: vec![CargoPackageDependency {
                manifest_path: "Cargo.toml".to_string(),
                declaring_package: Some("fixture_app".to_string()),
                dependency_key: "serde_alias".to_string(),
                package_name: "serde".to_string(),
                kind: CargoDependencyKind::Normal,
                target: None,
                version: Some("1".to_string()),
                path: None,
                optional: false,
                features: Vec::new(),
                declaration: CargoDependencyDeclaration::DeclaredExternal,
                linked_package_manifest_path: None,
                provenance: provenance("Cargo.toml", Some(12), Some(12), "test", "dependency"),
            }],
            manifest_hash: fingerprint("cargo-plan"),
            ..ProjectManifest::default()
        }
    }

    fn indexed_fixture() -> Result<(tempfile::TempDir, std::path::PathBuf, ProjectManifest)> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        Ok((fixture, root, manifest))
    }

    fn planned_paths(outcome: ProjectContextRetrievalPlanningOutcome) -> Vec<String> {
        match outcome {
            ProjectContextRetrievalPlanningOutcome::Plan(plan) => plan
                .read_requests
                .into_iter()
                .map(|request| request.relative_manifest_path().to_string())
                .collect(),
            ProjectContextRetrievalPlanningOutcome::Refusal(refusal) => {
                panic!("unexpected refusal: {:?}", refusal)
            }
        }
    }

    fn refusal_code(
        outcome: ProjectContextRetrievalPlanningOutcome,
    ) -> Option<ProjectContextRetrievalRefusalCode> {
        match outcome {
            ProjectContextRetrievalPlanningOutcome::Refusal(refusal) => Some(refusal.code),
            ProjectContextRetrievalPlanningOutcome::Plan(_) => None,
        }
    }

    fn freshness(manifest: &ProjectManifest) -> ManifestFreshnessEvaluation {
        ManifestFreshnessEvaluation {
            state: fresh_state(manifest),
            proof_level: FreshnessProofLevel::FullFilesystem,
        }
    }

    fn fresh_state(manifest: &ProjectManifest) -> FreshnessState {
        FreshnessState {
            changed: Vec::new(),
            deleted: Vec::new(),
            added: Vec::new(),
            unchanged: manifest
                .files
                .iter()
                .map(|file| file.path.clone())
                .collect(),
            fresh: true,
        }
    }

    fn manual_manifest() -> ProjectManifest {
        ProjectManifest {
            version: 1,
            root: std::path::PathBuf::from("/workspace"),
            files: vec![SourceFile {
                path: "src/lib.rs".to_string(),
                language: crate::Language::Rust,
                bytes: 100,
                lines: 9,
                content_hash: fingerprint("file"),
                provenance: provenance("src/lib.rs", Some(1), Some(9), "test", "file-provenance"),
            }],
            manifest_hash: fingerprint("manifest"),
            ..ProjectManifest::default()
        }
    }

    fn provenance(
        path: &str,
        start_line: Option<u32>,
        end_line: Option<u32>,
        source: &str,
        seed: &str,
    ) -> Provenance {
        Provenance {
            path: path.to_string(),
            start_line,
            end_line,
            source: source.to_string(),
            fingerprint: fingerprint(seed),
        }
    }
}
