use std::path::PathBuf;

use derive_more::Display;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Workspace identifier (UUID) from workspace server.
///
/// Generated locally and sent to server during CreateWorkspace.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Display)]
#[display("{}", _0)]
pub struct WorkspaceId(Uuid);

impl WorkspaceId {
    /// Generate a new random workspace ID
    pub fn generate() -> Self {
        Self(Uuid::new_v4())
    }

    /// Parse a workspace ID from a string
    ///
    /// # Errors
    /// Returns an error if the string is not a valid UUID
    pub fn from_string(s: &str) -> anyhow::Result<Self> {
        Ok(Self(Uuid::parse_str(s)?))
    }

    /// Get the inner UUID
    pub fn inner(&self) -> Uuid {
        self.0
    }
}

/// Freshness state for the local project-model manifest used by automatic
/// context injection and workspace diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkspaceContextFreshness {
    /// The local project-model manifest exists and matches the current
    /// filesystem view.
    Fresh,
    /// The local project-model manifest exists but no longer matches the
    /// current filesystem view.
    Stale {
        /// Files changed since the manifest was written.
        changed: Vec<String>,
        /// Files deleted since the manifest was written.
        deleted: Vec<String>,
        /// Files added since the manifest was written.
        added: Vec<String>,
    },
    /// Freshness could not be proven, so callers must treat the manifest as
    /// unavailable for injection.
    Unknown {
        /// Redaction-safe reason freshness could not be evaluated.
        reason: String,
    },
}

impl WorkspaceContextFreshness {
    /// Returns true only when the manifest is proven fresh.
    pub fn is_fresh(&self) -> bool {
        matches!(self, Self::Fresh)
    }

    /// Returns a stable diagnostic label for this freshness state.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Fresh => "fresh",
            Self::Stale { .. } => "stale",
            Self::Unknown { .. } => "unknown",
        }
    }
}

/// Local project-model manifest diagnostic for a workspace candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceContextManifestDiagnostic {
    /// Workspace root path being evaluated.
    pub workspace_root: PathBuf,
    /// Expected local project-model manifest path.
    pub manifest_path: PathBuf,
    /// Whether the manifest file exists at the expected path.
    pub manifest_found: bool,
    /// Freshness classification for the manifest when present.
    pub freshness: WorkspaceContextFreshness,
}

impl WorkspaceContextManifestDiagnostic {
    /// Returns true only when the manifest is present and proven fresh.
    pub fn can_inject(&self) -> bool {
        self.manifest_found && self.freshness.is_fresh()
    }
}

/// Candidate path considered while explaining project-model context injection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceContextCandidateDiagnostic {
    /// Candidate path before ancestor scanning.
    pub candidate_path: PathBuf,
    /// Selected workspace root for this candidate, when a fresh manifest is
    /// found.
    pub selected_workspace: Option<PathBuf>,
    /// Path filter that would be applied to retrieval for this candidate.
    pub path_filter: Option<String>,
    /// Exact reason this candidate was not selected.
    pub skip_reason: Option<String>,
}

/// User-facing explanation for automatic project-model context injection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceContextExplanation {
    /// Current working directory used for target resolution.
    pub cwd: PathBuf,
    /// Latest user query or explicit diagnostic query being explained.
    pub query: Option<String>,
    /// Candidate paths extracted from cwd and path mentions.
    pub candidates: Vec<WorkspaceContextCandidateDiagnostic>,
    /// Fresh selected targets that would be queried for context.
    pub selected_targets: Vec<WorkspaceContextManifestDiagnostic>,
    /// Selected target roots whose retrieval returned no usable nodes for the
    /// explained query.
    pub retrieval_empty_targets: Vec<PathBuf>,
    /// Whether automatic project-model context would be injected.
    pub would_inject: bool,
    /// Exact top-level reason context would not be injected.
    pub skip_reason: Option<String>,
}
