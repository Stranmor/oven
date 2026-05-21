use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, Subcommand};
use forge_quality_runtime::{
    ArtifactReport, GateResult, QualityError, QualityProfile, QualityProfileCompileRequest,
    ReleaseDecisionEvaluateRequest, RuntimeStatus, SCHEMA_VERSION, TraceAppendRequest, TraceQuery,
    TraceStore, TraceStoreConfig, compile_quality_profile, evaluate_release,
};
use rmcp::{
    ErrorData as McpError, Json, ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Parser)]
#[command(name = "forge-quality-runtime")]
#[command(about = "MCP-first typed Quality Runtime")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Serve {
        #[arg(long)]
        project_root: PathBuf,
        #[arg(long)]
        trace_dir: PathBuf,
    },
    Smoke {
        #[arg(long)]
        project_root: PathBuf,
        #[arg(long)]
        trace_dir: PathBuf,
    },
}

#[derive(Clone)]
struct QualityRuntimeServer {
    store: TraceStore,
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

impl QualityRuntimeServer {
    fn new(project_root: PathBuf, trace_dir: PathBuf) -> anyhow::Result<Self> {
        Ok(Self {
            store: TraceStore::new(TraceStoreConfig { project_root, trace_dir })?,
            tool_router: Self::tool_router(),
        })
    }
}

#[tool_router]
impl QualityRuntimeServer {
    #[tool(description = "Return Quality Runtime status and supported tool names")]
    fn runtime_status(&self) -> Json<RuntimeStatus> {
        Json(self.store.runtime_status())
    }

    #[tool(description = "Compile a deterministic typed quality profile for an artifact report")]
    fn quality_profile_compile(
        &self,
        Parameters(request): Parameters<QualityProfileCompileRequest>,
    ) -> Result<Json<QualityProfile>, McpError> {
        compile_quality_profile(request.artifact)
            .map(Json)
            .map_err(quality_error)
    }

    #[tool(description = "Append a bounded redaction-safe event to the append-only trace store")]
    fn trace_append(
        &self,
        Parameters(request): Parameters<TraceAppendRequest>,
    ) -> Result<Json<forge_quality_runtime::TraceEvent>, McpError> {
        self.store.append(request).map(Json).map_err(quality_error)
    }

    #[tool(description = "Read the append-only trace store with optional bounded tail limit")]
    fn trace_get(
        &self,
        Parameters(query): Parameters<TraceQuery>,
    ) -> Result<Json<Vec<forge_quality_runtime::TraceEvent>>, McpError> {
        self.store.query(query).map(Json).map_err(quality_error)
    }

    #[tool(
        description = "Alias for trace_get; query trace events with optional bounded tail limit"
    )]
    fn trace_query(
        &self,
        Parameters(query): Parameters<TraceQuery>,
    ) -> Result<Json<Vec<forge_quality_runtime::TraceEvent>>, McpError> {
        self.store.query(query).map(Json).map_err(quality_error)
    }

    #[tool(description = "Record a gate result as a typed trace event")]
    fn gate_record(
        &self,
        Parameters(request): Parameters<GateRecordRequest>,
    ) -> Result<Json<forge_quality_runtime::TraceEvent>, McpError> {
        let payload = serde_json::to_value(&request.gate_result).map_err(|_| {
            McpError::internal_error("quality_runtime.schema_serialization_failed", None)
        })?;
        self.store
            .append(TraceAppendRequest {
                project_root: request.project_root,
                idempotency_key: request.idempotency_key,
                event_kind: forge_quality_runtime::TraceEventKind::GateRecorded,
                payload,
            })
            .map(Json)
            .map_err(quality_error)
    }

    #[tool(
        description = "Evaluate whether a quality profile can be released from vector gate results"
    )]
    fn release_decision_evaluate(
        &self,
        Parameters(request): Parameters<ReleaseDecisionEvaluateRequest>,
    ) -> Result<Json<forge_quality_runtime::ReleaseDecision>, McpError> {
        let now = request.now.unwrap_or_else(chrono::Utc::now);
        evaluate_release(&request.profile, &request.gate_results, now)
            .map(Json)
            .map_err(quality_error)
    }

    #[tool(
        description = "Alias for release_decision_evaluate; compute release decision from inputs"
    )]
    fn release_decision_get(
        &self,
        Parameters(request): Parameters<ReleaseDecisionEvaluateRequest>,
    ) -> Result<Json<forge_quality_runtime::ReleaseDecision>, McpError> {
        let now = request.now.unwrap_or_else(chrono::Utc::now);
        evaluate_release(&request.profile, &request.gate_results, now)
            .map(Json)
            .map_err(quality_error)
    }
}

impl ServerHandler for QualityRuntimeServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some("Typed Quality Runtime: compile profiles, record gates/traces, and evaluate release decisions.".into()),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct GateRecordRequest {
    project_root: PathBuf,
    idempotency_key: String,
    gate_result: GateResult,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Serve { project_root, trace_dir } => {
            let server = QualityRuntimeServer::new(project_root, trace_dir)?;
            server.serve(rmcp::transport::stdio()).await?;
        }
        Command::Smoke { project_root, trace_dir } => {
            let server = QualityRuntimeServer::new(project_root.clone(), trace_dir)?;
            let status = server.store.runtime_status();
            let artifact = ArtifactReport {
                schema_version: SCHEMA_VERSION,
                artifact_id: "smoke-artifact".to_string(),
                artifact_class: forge_quality_runtime::ArtifactClass::CodeMcpToolSurface,
                artifact_version: "0.0.0-smoke".to_string(),
                artifact_hash: forge_quality_runtime::digest_text("smoke"),
                producer_id: "smoke-producer".to_string(),
                owner: Some("quality-runtime".to_string()),
                claim: forge_quality_runtime::ReleaseClaim::LocalSmoke,
                non_claims: vec!["no live MCP client handshake in smoke command".to_string()],
                dimensions_not_checked: Default::default(),
                publish_approval_boundary: None,
            };
            let profile = compile_quality_profile(artifact).context("compile smoke profile")?;
            let event = server.store.append(TraceAppendRequest {
                project_root,
                idempotency_key: "smoke-status".to_string(),
                event_kind: forge_quality_runtime::TraceEventKind::RuntimeStatus,
                payload: serde_json::to_value(&status)?,
            })?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "schema_version": SCHEMA_VERSION,
                    "status": status,
                    "profile_id": profile.profile_id,
                    "trace_event_id": event.event_id,
                }))?
            );
        }
    }
    Ok(())
}

fn quality_error(error: QualityError) -> McpError {
    let code = match error {
        QualityError::UnknownArtifactType => "quality_runtime.unknown_artifact_type",
        QualityError::ProjectRootRejected => "quality_runtime.project_root_rejected",
        QualityError::PayloadRejected => "quality_runtime.payload_rejected",
        QualityError::SecretRejected => "quality_runtime.secret_rejected",
        QualityError::IdempotencyConflict => "quality_runtime.idempotency_conflict",
        QualityError::NonMonotonicSequence => "quality_runtime.non_monotonic_sequence",
        QualityError::DigestChainInvalid => "quality_runtime.digest_chain_invalid",
        QualityError::MalformedReleaseRequest => "quality_runtime.malformed_release_request",
        QualityError::Io(_) => "quality_runtime.io_error",
        QualityError::Json(_) => "quality_runtime.json_error",
    };
    McpError::internal_error(code, None)
}
