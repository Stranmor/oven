//! Local project-model storage path conventions and target resolution policy.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Directory name used for local project-model storage under a workspace root.
pub const LOCAL_PROJECT_MODEL_DIR_NAME: &str = ".forge_project_model";

/// Manifest file name used by local project-model storage.
pub const LOCAL_PROJECT_MODEL_MANIFEST_FILE_NAME: &str = "project_manifest.json";

/// External fact ingestion report file name used by local project-model storage.
pub const LOCAL_PROJECT_MODEL_EXTERNAL_FACT_REPORT_FILE_NAME: &str =
    "external_fact_artifact_ingestion_report.json";

/// Returns the canonical local project-model directory for a workspace root.
///
/// # Arguments
///
/// * `workspace_root` - Workspace root that owns local project-model storage.
pub fn local_project_model_dir(workspace_root: &Path) -> PathBuf {
    workspace_root.join(LOCAL_PROJECT_MODEL_DIR_NAME)
}

/// Returns the canonical local project-model manifest path for a workspace root.
///
/// # Arguments
///
/// * `workspace_root` - Workspace root that owns local project-model storage.
pub fn local_project_model_manifest(workspace_root: &Path) -> PathBuf {
    local_project_model_dir(workspace_root).join(LOCAL_PROJECT_MODEL_MANIFEST_FILE_NAME)
}

/// Returns the canonical local project-model external fact ingestion report path.
///
/// # Arguments
///
/// * `workspace_root` - Workspace root that owns local project-model storage.
pub fn local_project_model_external_fact_report(workspace_root: &Path) -> PathBuf {
    local_project_model_dir(workspace_root).join(LOCAL_PROJECT_MODEL_EXTERNAL_FACT_REPORT_FILE_NAME)
}

/// A selected project-model context target.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProjectContextTarget {
    /// Workspace root whose local project-model manifest should be queried.
    pub workspace_root: PathBuf,
    /// Optional relative directory prefix used to narrow retrieval.
    pub path_filter: Option<String>,
}

impl ProjectContextTarget {
    /// Creates a selected project-model context target.
    ///
    /// # Arguments
    ///
    /// * `workspace_root` - Workspace root to query.
    /// * `path_filter` - Optional relative path prefix for narrowed retrieval.
    pub fn new(workspace_root: PathBuf, path_filter: Option<String>) -> Self {
        Self { workspace_root, path_filter }
    }
}

/// Bounded target-resolution budget for prompt-derived paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetResolutionBudget {
    remaining_candidates: usize,
    remaining_index_probes: usize,
}

impl TargetResolutionBudget {
    /// Creates a bounded target-resolution budget.
    ///
    /// # Arguments
    ///
    /// * `explicit_target_candidates` - Maximum prompt-derived path candidates.
    /// * `index_probes` - Maximum ancestor manifest probes.
    pub fn new(explicit_target_candidates: usize, index_probes: usize) -> Self {
        Self {
            remaining_candidates: explicit_target_candidates.saturating_add(1),
            remaining_index_probes: index_probes,
        }
    }

    /// Claims one candidate slot.
    pub fn claim_candidate(&mut self) -> bool {
        let Some(remaining) = self.remaining_candidates.checked_sub(1) else {
            return false;
        };
        self.remaining_candidates = remaining;
        true
    }

    /// Claims one index-probe slot.
    pub fn claim_index_probe(&mut self) -> bool {
        let Some(remaining) = self.remaining_index_probes.checked_sub(1) else {
            return false;
        };
        self.remaining_index_probes = remaining;
        true
    }
}

/// Extracts path candidates mentioned in user text.
///
/// # Arguments
///
/// * `message` - User text that may contain tagged or path-like references.
/// * `cwd` - Current working directory used to resolve relative references.
/// * `home` - Optional home directory used for `~` references.
pub fn mentioned_paths(message: &str, cwd: &Path, home: Option<&Path>) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let mut seen = BTreeSet::new();
    for tag_path in attachment_tag_paths(message) {
        if let Some(path) = resolve_mentioned_path(&tag_path, cwd, home)
            && seen.insert(path.clone())
        {
            paths.push(path);
        }
    }
    for token in path_like_tokens(message) {
        if let Some(path) = resolve_mentioned_path(&token, cwd, home)
            && seen.insert(path.clone())
        {
            paths.push(path);
        }
    }
    paths
}

fn attachment_tag_paths(message: &str) -> Vec<String> {
    let mut paths = Vec::new();
    let mut rest = message;
    while let Some(start) = rest.find("@[") {
        let Some(after_prefix_start) = start.checked_add(2) else {
            break;
        };
        let after_start = &rest[after_prefix_start..];
        let Some(end) = after_start.find(']') else {
            break;
        };
        let path = &after_start[..end];
        if !path.is_empty() {
            paths.push(path.to_string());
        }
        let Some(next_start) = end.checked_add(1) else {
            break;
        };
        rest = &after_start[next_start..];
    }
    paths
}

/// Returns an optional relative directory prefix when a candidate path is an
/// existing directory under a workspace root.
///
/// # Arguments
///
/// * `path` - Candidate path selected from cwd or prompt text.
/// * `workspace_root` - Workspace root selected by manifest probing.
pub fn directory_path_filter(path: &Path, workspace_root: &Path) -> Option<String> {
    let relative = path
        .strip_prefix(workspace_root)
        .ok()
        .filter(|relative| !relative.as_os_str().is_empty())?;
    if !path.is_dir() {
        return None;
    }
    let mut filter = relative.to_string_lossy().replace('\\', "/");
    if !filter.ends_with('/') {
        filter.push('/');
    }
    Some(filter)
}

/// Resolves a raw path reference against cwd and home.
///
/// # Arguments
///
/// * `raw_path` - Raw prompt path reference.
/// * `cwd` - Current working directory for relative paths.
/// * `home` - Optional home directory for `~` paths.
pub fn resolve_mentioned_path(raw_path: &str, cwd: &Path, home: Option<&Path>) -> Option<PathBuf> {
    if raw_path.is_empty() || raw_path.contains("://") {
        return None;
    }
    let path = if raw_path == "~" {
        home?.to_path_buf()
    } else if let Some(stripped) = raw_path.strip_prefix("~/") {
        home?.join(stripped)
    } else {
        let path = PathBuf::from(raw_path);
        if path.is_absolute() {
            path
        } else {
            cwd.join(path)
        }
    };
    Some(path)
}

fn path_like_tokens(message: &str) -> Vec<String> {
    message
        .split_whitespace()
        .filter_map(normalize_path_token)
        .collect()
}

fn normalize_path_token(raw: &str) -> Option<String> {
    let token = raw.trim_matches(|character: char| {
        matches!(
            character,
            '`' | '\'' | '"' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';'
        )
    });
    let token = token.trim_end_matches(['.', ':', '!']);
    if token.is_empty()
        || token.contains("://")
        || token.starts_with('<')
        || token.ends_with('>')
        || token.starts_with("@[")
    {
        return None;
    }
    let token = trim_line_suffix(token);
    if token.starts_with('/')
        || token.starts_with('~')
        || token.starts_with("./")
        || token.starts_with("../")
        || is_relative_path_token(token)
    {
        Some(token.to_string())
    } else {
        None
    }
}

fn is_relative_path_token(token: &str) -> bool {
    if !token.contains('/') {
        return false;
    }
    let components = token
        .split('/')
        .filter(|component| !component.is_empty())
        .collect::<Vec<_>>();
    components.len() > 2
        || components
            .last()
            .is_some_and(|component| component.contains('.'))
}

fn trim_line_suffix(token: &str) -> &str {
    let Some((prefix, suffix)) = token.rsplit_once(':') else {
        return token;
    };
    if suffix.chars().all(|character| character.is_ascii_digit()) {
        trim_line_suffix(prefix)
    } else {
        token
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn resolves_tagged_absolute_home_and_relative_paths() -> anyhow::Result<()> {
        let fixture = TempDir::new()?;
        let cwd = fixture.path().join("workspace");
        let home = fixture.path().join("home");
        let setup = format!(
            "inspect @[src/lib.rs] ~/notes.md {}/abs.rs src/bin/main.rs:12",
            fixture.path().display()
        );
        let actual = mentioned_paths(&setup, &cwd, Some(&home));
        let expected = vec![
            cwd.join("src/lib.rs"),
            home.join("notes.md"),
            fixture.path().join("abs.rs"),
            cwd.join("src/bin/main.rs"),
        ];
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn directory_filter_is_relative_and_slash_terminated() -> anyhow::Result<()> {
        let fixture = TempDir::new()?;
        let setup = fixture.path().join("workspace");
        std::fs::create_dir_all(setup.join("src/bin"))?;
        let actual = directory_path_filter(&setup.join("src/bin"), &setup);
        let expected = Some("src/bin/".to_string());
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn local_manifest_path_uses_single_convention() {
        let setup = PathBuf::from("/workspace");
        let actual = local_project_model_manifest(&setup);
        let expected = PathBuf::from("/workspace/.forge_project_model/project_manifest.json");
        assert_eq!(actual, expected);
    }

    #[test]
    fn local_external_fact_report_path_uses_single_convention() {
        let setup = PathBuf::from("/workspace");
        let actual = local_project_model_external_fact_report(&setup);
        let expected = PathBuf::from(
            "/workspace/.forge_project_model/external_fact_artifact_ingestion_report.json",
        );
        assert_eq!(actual, expected);
    }
}
