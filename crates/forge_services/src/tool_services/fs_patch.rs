use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use bytes::Bytes;
use forge_app::domain::PatchOperation;
use forge_app::{
    ApplyPatchFileOutput, ApplyPatchFileStatus, ApplyPatchOutput, EnvironmentInfra,
    FileWriterInfra, FsPatchService, PatchOutput, compute_hash,
};
use forge_config::ForgeConfig;
use forge_domain::{
    FuzzySearchRepository, SearchMatch, SearchMode, SnapshotRepository, TextPatchBlock,
    TextPatchRepository, ValidationRepository,
};
use thiserror::Error;
use tokio::fs;

use crate::utils::assert_absolute_path;

/// A match found in the source text. Represents a range in the source text that
/// can be used for extraction or replacement operations. Stores the position
/// and length to allow efficient substring operations.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
struct Range {
    /// Starting position of the match in source text
    start: usize,
    /// Length of the matched text
    length: usize,
}

impl Range {
    /// Create a new match from a start position and length
    fn new(start: usize, length: usize) -> Self {
        Self { start, length }
    }

    /// Get the end position (exclusive) of this match
    fn end(&self) -> usize {
        self.start.saturating_add(self.length)
    }

    /// Try to find an exact match in the source text
    fn find_exact(source: &str, search: &str) -> Option<Self> {
        source
            .find(search)
            .map(|start| Self::new(start, search.len()))
    }

    /// Detect the line ending used in the source (CRLF or LF)
    fn detect_line_ending(source: &str) -> &'static str {
        if source.contains("\r\n") {
            "\r\n"
        } else {
            "\n"
        }
    }

    /// Normalize line endings in a search string to match the source
    fn normalize_search_line_endings(source: &str, search: &str) -> String {
        let line_ending = Self::detect_line_ending(source);
        if line_ending == "\r\n" {
            search.replace("\r\n", "\n").replace("\n", "\r\n")
        } else {
            search.replace("\r\n", "\n")
        }
    }

    /// Create a range from a fuzzy search match
    #[allow(dead_code)]
    fn from_search_match(source: &str, search_match: &SearchMatch) -> Self {
        let lines: Vec<&str> = source.lines().collect();

        // Handle empty source
        if lines.is_empty() {
            return Self::new(0, 0);
        }

        // SearchMatch uses 0-based inclusive line numbers
        // Convert to 0-based array indices
        let start_idx = usize::try_from(search_match.start_line)
            .unwrap_or(usize::MAX)
            .min(lines.len());
        // end_line is 0-based inclusive, convert to 0-based exclusive for slicing
        // Add 1 to make it exclusive: line 0 to line 0 means [0..1], one line
        let end_idx = usize::try_from(search_match.end_line)
            .unwrap_or(usize::MAX)
            .saturating_add(1)
            .min(lines.len());

        // Find the byte position of the start line.
        // Split on '\n' so each segment retains its '\r' (if any), giving the
        // correct per-line byte length regardless of mixed line endings.
        let start_pos = source
            .split('\n')
            .take(start_idx)
            .map(|l| l.len().saturating_add(1))
            .sum::<usize>()
            .min(source.len());

        // Calculate the length
        let length = if start_idx == end_idx {
            // Single line match: just the line content, no trailing newline
            if start_idx >= lines.len() {
                0 // Out of bounds match
            } else {
                lines.get(start_idx).map_or(0, |l| l.len())
            }
        } else {
            // Multi-line match: include newlines between lines but NOT after the last line
            // Sum lengths of lines from start_idx to end_idx (exclusive)
            let content_len: usize = if start_idx >= lines.len() || end_idx > lines.len() {
                0 // Out of bounds match
            } else {
                lines
                    .get(start_idx..end_idx)
                    .map_or(0, |slice| slice.iter().map(|l| l.len()).sum())
            };
            let newlines_between = end_idx.saturating_sub(start_idx).saturating_sub(1);
            // Count actual newline bytes (\r\n = 2, \n = 1) to handle mixed endings
            let newline_bytes: usize = source
                .split('\n')
                .skip(start_idx)
                .take(newlines_between)
                .map(|l| if l.ends_with('\r') { 2 } else { 1 })
                .sum();
            content_len.saturating_add(newline_bytes)
        };

        Self::new(start_pos, length)
    }

    // Fuzzy matching removed - we only use exact matching
}

impl From<Range> for std::ops::Range<usize> {
    fn from(m: Range) -> Self {
        m.start..m.end()
    }
}

// MatchSequence struct and implementation removed - we only use exact matching

#[derive(Debug, Error)]
enum Error {
    #[error("Failed to read/write file: {0}")]
    FileOperation(#[from] std::io::Error),
    #[error(
        "Could not find match for search text: '{0}'. File may have changed externally, consider reading the file again."
    )]
    NoMatch(String),
    #[error("Could not find swap target text: {0}")]
    NoSwapTarget(String),
    #[error(
        "Multiple matches found for search text: '{0}'. Either provide a more specific search pattern or use replace_all to replace all occurrences."
    )]
    MultipleMatches(String),
    #[error(
        "Match range [{0}..{1}) is out of bounds for content of length {2}. File may have changed externally, consider reading the file again."
    )]
    RangeOutOfBounds(usize, usize, usize),
    #[error("Failed to build fuzzy patch: {message}")]
    PatchBuild { message: String },
}

/// Compute a range from search text, with operation-aware error handling
///
/// Returns Some(range) if a match is found, None if no search or operation
/// doesn't require a match, or an error if a search was provided but no match
/// was found for operations that require it.
fn compute_range(
    source: &str,
    search: Option<&str>,
    operation: &PatchOperation,
) -> Result<Option<Range>, Error> {
    match search {
        Some(s) if !s.is_empty() => {
            let normalized_search = Range::normalize_search_line_endings(source, s);
            let match_result = Range::find_exact(source, &normalized_search)
                .ok_or_else(|| Error::NoMatch(s.to_string()));
            match match_result {
                Ok(r) => Ok(Some(r)),
                Err(e) => {
                    // Handle no match based on operation type
                    match operation {
                        PatchOperation::Replace
                        | PatchOperation::ReplaceAll
                        | PatchOperation::Swap => Err(e),
                        _ => Ok(None),
                    }
                }
            }
        }
        _ => Ok(None),
    }
}

/// A match found in the source text. Represents a range in the source text that
///
/// # Arguments
/// * `haystack` - The original content to patch
/// * `range` - Optional range indicating the location to apply the patch
/// * `operation` - The patch operation to perform
/// * `content` - The content to use for the patch operation
///
/// # Returns
/// The patched content, or an error if the operation fails
fn apply_replacement(
    haystack: String,
    range: Option<Range>,
    operation: &PatchOperation,
    content: &str,
) -> Result<String, Error> {
    let line_ending = Range::detect_line_ending(&haystack);
    let normalized_content = Range::normalize_search_line_endings(&haystack, content);
    // Handle case where range is provided (match found)
    if let Some(patch) = range {
        // Validate the range is within bounds before indexing
        if patch.end() > haystack.len() {
            return Err(Error::RangeOutOfBounds(
                patch.start,
                patch.end(),
                haystack.len(),
            ));
        }

        // Extract the matched text from haystack
        let needle = haystack
            .get(patch.start..patch.end())
            .ok_or_else(|| Error::RangeOutOfBounds(patch.start, patch.end(), haystack.len()))?;

        // Apply the operation based on its type
        match operation {
            // Prepend content before the matched text
            PatchOperation::Prepend => {
                let before = haystack.get(..patch.start).ok_or(Error::RangeOutOfBounds(
                    0,
                    patch.start,
                    haystack.len(),
                ))?;
                let after = haystack.get(patch.start..).ok_or({
                    Error::RangeOutOfBounds(patch.start, haystack.len(), haystack.len())
                })?;
                Ok(format!("{}{}{}", before, normalized_content, after))
            }

            // Replace all occurrences of the matched text with new content
            PatchOperation::ReplaceAll => Ok(haystack.replace(needle, &normalized_content)),

            // Append content after the matched text
            PatchOperation::Append => {
                let before = haystack
                    .get(..patch.end())
                    .ok_or_else(|| Error::RangeOutOfBounds(0, patch.end(), haystack.len()))?;
                let after = haystack.get(patch.end()..).ok_or_else(|| {
                    Error::RangeOutOfBounds(patch.end(), haystack.len(), haystack.len())
                })?;
                Ok(format!(
                    "{}{}{}{}",
                    before, line_ending, normalized_content, after
                ))
            }

            // Replace matched text with new content
            PatchOperation::Replace => {
                // Check if there are multiple matches
                let mut match_count = 0usize;
                let mut search_start = 0;
                while let Some(pos) = haystack.get(search_start..).and_then(|s| s.find(needle)) {
                    match_count = match_count.saturating_add(1);
                    if match_count > 1 {
                        return Err(Error::MultipleMatches(needle.to_string()));
                    }
                    search_start = search_start
                        .saturating_add(pos)
                        .saturating_add(needle.len());
                }

                let before = haystack.get(..patch.start).ok_or(Error::RangeOutOfBounds(
                    0,
                    patch.start,
                    haystack.len(),
                ))?;
                let after = haystack.get(patch.end()..).ok_or_else(|| {
                    Error::RangeOutOfBounds(patch.end(), haystack.len(), haystack.len())
                })?;
                Ok(format!("{}{}{}", before, normalized_content, after))
            }

            // Swap with another text in the source
            PatchOperation::Swap => {
                // Find the target text to swap with
                let target_patch = Range::find_exact(&haystack, &normalized_content)
                    .ok_or_else(|| Error::NoSwapTarget(content.to_string()))?;

                // Handle the case where patches overlap
                if (patch.start <= target_patch.start && patch.end() > target_patch.start)
                    || (target_patch.start <= patch.start && target_patch.end() > patch.start)
                {
                    // For overlapping ranges, we just do an ordinary replacement
                    let before = haystack.get(..patch.start).ok_or(Error::RangeOutOfBounds(
                        0,
                        patch.start,
                        haystack.len(),
                    ))?;
                    let after = haystack.get(patch.end()..).ok_or_else(|| {
                        Error::RangeOutOfBounds(patch.end(), haystack.len(), haystack.len())
                    })?;
                    return Ok(format!("{}{}{}", before, normalized_content, after));
                }

                // We need to handle different ordering of patches
                if patch.start < target_patch.start {
                    // Original text comes first
                    let part1 = haystack.get(..patch.start).ok_or(Error::RangeOutOfBounds(
                        0,
                        patch.start,
                        haystack.len(),
                    ))?;
                    let part2 = haystack
                        .get(patch.end()..target_patch.start)
                        .ok_or_else(|| {
                            Error::RangeOutOfBounds(patch.end(), target_patch.start, haystack.len())
                        })?;
                    let part3 = haystack.get(patch.start..patch.end()).ok_or_else(|| {
                        Error::RangeOutOfBounds(patch.start, patch.end(), haystack.len())
                    })?;
                    let part4 = haystack.get(target_patch.end()..).ok_or_else(|| {
                        Error::RangeOutOfBounds(target_patch.end(), haystack.len(), haystack.len())
                    })?;
                    Ok(format!(
                        "{}{}{}{}{}",
                        part1, normalized_content, part2, part3, part4
                    ))
                } else {
                    // Target text comes first
                    let part1 = haystack.get(..target_patch.start).ok_or({
                        Error::RangeOutOfBounds(0, target_patch.start, haystack.len())
                    })?;
                    let part2 = haystack.get(patch.start..patch.end()).ok_or_else(|| {
                        Error::RangeOutOfBounds(patch.start, patch.end(), haystack.len())
                    })?;
                    let part3 = haystack
                        .get(target_patch.end()..patch.start)
                        .ok_or_else(|| {
                            Error::RangeOutOfBounds(target_patch.end(), patch.start, haystack.len())
                        })?;
                    let part4 = haystack.get(patch.end()..).ok_or_else(|| {
                        Error::RangeOutOfBounds(patch.end(), haystack.len(), haystack.len())
                    })?;
                    Ok(format!(
                        "{}{}{}{}{}",
                        part1, part2, part3, normalized_content, part4
                    ))
                }
            }
        }
    } else {
        // No match (range is None) - treat as empty search (full file operation)
        match operation {
            // Append to the end of the file
            PatchOperation::Append => Ok(format!("{haystack}{line_ending}{normalized_content}")),
            // Prepend to the beginning of the file
            PatchOperation::Prepend => Ok(format!("{normalized_content}{haystack}")),
            // Replace is equivalent to completely replacing the file
            PatchOperation::Replace | PatchOperation::ReplaceAll => Ok(normalized_content),
            // Swap doesn't make sense with empty search - keep source unchanged
            PatchOperation::Swap => Ok(haystack),
        }
    }
}

// Using PatchOperation from forge_domain

// Using FSPatchInput from forge_domain

fn build_fuzzy_patch(
    current_content: &str,
    search_text: &str,
    content: &str,
    patch: TextPatchBlock,
) -> String {
    let _ = (
        Range::normalize_search_line_endings(current_content, search_text),
        Range::normalize_search_line_endings(current_content, content),
        patch.patch,
    );
    patch.patched_text
}

async fn apply_fuzzy_search_fallback<F: FuzzySearchRepository>(
    infra: &F,
    current_content: String,
    search_text: String,
    content: &str,
    operation: &PatchOperation,
) -> Result<String, Error> {
    let range = match infra
        .fuzzy_search(&search_text, &current_content, SearchMode::FirstMatch)
        .await
    {
        Ok(matches) if !matches.is_empty() => matches
            .first()
            .map(|m| Range::from_search_match(&current_content, m)),
        _ => return Err(Error::NoMatch(search_text)),
    };

    apply_replacement(current_content, range, operation, content)
}

async fn apply_text_patch_fallback<F: TextPatchRepository>(
    infra: &F,
    current_content: String,
    search_text: String,
    content: &str,
) -> Result<String, Error> {
    let normalized_search = Range::normalize_search_line_endings(&current_content, &search_text);
    let normalized_content = Range::normalize_search_line_endings(&current_content, content);
    let patch = infra
        .build_text_patch(&current_content, &normalized_search, &normalized_content)
        .await
        .map_err(|error| Error::PatchBuild { message: error.to_string() })?;
    Ok(build_fuzzy_patch(
        &current_content,
        &search_text,
        content,
        patch,
    ))
}

async fn apply_replace_operation<F: FuzzySearchRepository + TextPatchRepository>(
    infra: &F,
    current_content: String,
    search: &str,
    content: &str,
    operation: &PatchOperation,
    use_text_patch_fallback: bool,
) -> Result<String, Error> {
    match compute_range(&current_content, Some(search), operation) {
        Ok(range) => apply_replacement(current_content, range, operation, content),
        Err(Error::NoMatch(search_text)) if matches!(operation, PatchOperation::Replace) => {
            if use_text_patch_fallback {
                apply_text_patch_fallback(infra, current_content, search_text, content).await
            } else {
                apply_fuzzy_search_fallback(infra, current_content, search_text, content, operation)
                    .await
            }
        }
        Err(e) => Err(e),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ApplyPatchOperation {
    Update,
    Add,
    Delete,
}

#[derive(Debug, Clone)]
struct ApplyPatchSection {
    operation: ApplyPatchOperation,
    path: String,
    lines: Vec<String>,
}

#[derive(Debug, Clone)]
struct PlannedApplyPatchFile {
    section: ApplyPatchSection,
    absolute_path: PathBuf,
    before: Option<String>,
    after: Option<String>,
    hunks_applied: usize,
}

fn parse_apply_patch(input: &str) -> anyhow::Result<Vec<ApplyPatchSection>> {
    let raw_lines: Vec<&str> = input.lines().collect();
    let first = raw_lines
        .iter()
        .position(|line| !line.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("apply_patch input is empty"))?;
    let last = raw_lines
        .iter()
        .rposition(|line| !line.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("apply_patch input is empty"))?;

    let first_line = raw_lines
        .get(first)
        .ok_or_else(|| anyhow::anyhow!("apply_patch first marker is out of bounds"))?;
    if first_line.trim() != "*** Begin Patch" {
        anyhow::bail!("apply_patch first non-empty line must be '*** Begin Patch'");
    }
    let last_line = raw_lines
        .get(last)
        .ok_or_else(|| anyhow::anyhow!("apply_patch last marker is out of bounds"))?;
    if last_line.trim() != "*** End Patch" {
        anyhow::bail!("apply_patch last non-empty line must be '*** End Patch'");
    }

    let mut sections = Vec::new();
    let mut current: Option<ApplyPatchSection> = None;
    let mut seen = HashSet::new();

    let section_start = first
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("apply_patch first marker index overflowed"))?;
    let body_lines = raw_lines
        .get(section_start..last)
        .ok_or_else(|| anyhow::anyhow!("apply_patch body range is invalid"))?;
    for line in body_lines {
        if line.trim().is_empty() && current.is_none() {
            continue;
        }

        if let Some((operation, path)) = parse_apply_patch_section_header(line)? {
            if let Some(section) = current.take() {
                sections.push(section);
            }
            let normalized_key = lexical_normalized_path_key(path)?;
            if !seen.insert(normalized_key.clone()) {
                anyhow::bail!("apply_patch contains duplicate target: {normalized_key}");
            }
            current = Some(ApplyPatchSection {
                operation,
                path: path.trim().to_string(),
                lines: Vec::new(),
            });
            continue;
        }

        if line.trim_start().starts_with("*** ") {
            anyhow::bail!("apply_patch contains unknown section header: {line}");
        }

        let section = current
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("apply_patch content appeared before a file section"))?;
        section.lines.push((*line).to_string());
    }

    if let Some(section) = current.take() {
        sections.push(section);
    }

    if sections.is_empty() {
        anyhow::bail!("apply_patch must contain at least one file section");
    }

    Ok(sections)
}

fn parse_apply_patch_section_header(
    line: &str,
) -> anyhow::Result<Option<(ApplyPatchOperation, &str)>> {
    let trimmed = line.trim();
    let header = trimmed
        .strip_prefix("*** Update File: ")
        .map(|path| (ApplyPatchOperation::Update, path))
        .or_else(|| {
            trimmed
                .strip_prefix("*** Add File: ")
                .map(|path| (ApplyPatchOperation::Add, path))
        })
        .or_else(|| {
            trimmed
                .strip_prefix("*** Delete File: ")
                .map(|path| (ApplyPatchOperation::Delete, path))
        });

    if header.is_none() && trimmed.starts_with("*** ") {
        anyhow::bail!("unsupported apply_patch section: {trimmed}");
    }

    if let Some((operation, path)) = header {
        if path.trim().is_empty() {
            anyhow::bail!("apply_patch file section path cannot be empty");
        }
        Ok(Some((operation, path)))
    } else {
        Ok(None)
    }
}

fn lexical_normalized_path_key(path: &str) -> anyhow::Result<String> {
    let path = Path::new(path.trim());
    let mut components = Vec::new();
    for component in path.components() {
        match component {
            Component::Prefix(_) => anyhow::bail!("Windows path prefixes are not supported"),
            Component::RootDir => components.clear(),
            Component::CurDir => {}
            Component::ParentDir => {
                anyhow::bail!("path traversal is not allowed: {}", path.display())
            }
            Component::Normal(value) => components.push(value.to_string_lossy().to_string()),
        }
    }
    if components.is_empty() {
        anyhow::bail!("apply_patch path must target a file: {}", path.display());
    }
    Ok(components.join("/"))
}

fn normalize_apply_patch_target(path: &str, cwd: &Path) -> anyhow::Result<PathBuf> {
    lexical_normalized_path_key(path)?;
    let raw = Path::new(path.trim());
    let absolute = if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        cwd.join(raw)
    };
    if absolute
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        anyhow::bail!("path traversal is not allowed: {}", absolute.display());
    }
    Ok(absolute)
}

fn ensure_target_inside_workspace(path: &Path, cwd: &Path, must_exist: bool) -> anyhow::Result<()> {
    let canonical_workspace = std::fs::canonicalize(cwd)
        .with_context(|| format!("Failed to canonicalize workspace root: {}", cwd.display()))?;
    let canonical_target = if must_exist {
        std::fs::canonicalize(path)
            .with_context(|| format!("Failed to canonicalize target: {}", path.display()))?
    } else {
        let parent = path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Target has no parent: {}", path.display()))?;
        let canonical_parent = std::fs::canonicalize(parent)
            .with_context(|| format!("Failed to canonicalize parent: {}", parent.display()))?;
        canonical_parent.join(
            path.file_name()
                .ok_or_else(|| anyhow::anyhow!("Target has no file name: {}", path.display()))?,
        )
    };
    if !canonical_target.starts_with(&canonical_workspace) {
        anyhow::bail!(
            "apply_patch target '{}' escapes workspace '{}'",
            canonical_target.display(),
            canonical_workspace.display()
        );
    }
    Ok(())
}

fn apply_update_lines(source: &str, lines: &[String]) -> anyhow::Result<(String, usize)> {
    let mut output = source.to_string();
    let mut index = 0;
    let mut hunks = 0usize;
    while index < lines.len() {
        let mut removed = Vec::new();
        let mut added = Vec::new();

        while let Some(line) = lines.get(index) {
            let Some(stripped) = line.strip_prefix('-') else {
                break;
            };
            removed.push(strip_patch_line_payload(stripped));
            index = index
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("apply_patch line index overflowed"))?;
        }
        while let Some(line) = lines.get(index) {
            let Some(stripped) = line.strip_prefix('+') else {
                break;
            };
            added.push(strip_patch_line_payload(stripped));
            index = index
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("apply_patch line index overflowed"))?;
        }

        if removed.is_empty() && added.is_empty() {
            let invalid_line = lines
                .get(index)
                .map(String::as_str)
                .unwrap_or("<end of patch>");
            anyhow::bail!(
                "Update File hunks support only '-' removed lines followed by '+' replacement lines; invalid line: {}",
                invalid_line
            );
        }
        if removed.is_empty() {
            anyhow::bail!("Update File hunk must include at least one '-' line");
        }

        let old = join_patch_lines(&removed);
        let new = join_patch_lines(&added);
        let normalized_old = Range::normalize_search_line_endings(&output, &old);
        let normalized_new = Range::normalize_search_line_endings(&output, &new);
        let range = Range::find_exact(&output, &normalized_old)
            .ok_or_else(|| anyhow::anyhow!("apply_patch hunk did not match exactly: {old:?}"))?;
        output.replace_range(std::ops::Range::<usize>::from(range), &normalized_new);
        hunks = hunks.saturating_add(1);
    }
    Ok((output, hunks))
}

fn strip_patch_line_payload(payload: &str) -> String {
    payload.strip_prefix(' ').unwrap_or(payload).to_string()
}

fn parse_add_file_content(lines: &[String]) -> anyhow::Result<String> {
    let mut content = Vec::new();
    for line in lines {
        let Some(stripped) = line.strip_prefix('+') else {
            anyhow::bail!("Add File content lines must start with '+': {line}");
        };
        content.push(strip_patch_line_payload(stripped));
    }
    Ok(join_patch_lines(&content))
}

fn join_patch_lines(lines: &[String]) -> String {
    if lines.is_empty() {
        String::new()
    } else {
        format!("{}\n", lines.join("\n"))
    }
}

fn rejected_apply_patch_output(
    sections: &[ApplyPatchSection],
    paths: &[PathBuf],
    message: impl Into<String>,
) -> ApplyPatchOutput {
    let message = message.into();
    let files = sections
        .iter()
        .zip(paths.iter())
        .map(|(section, path)| ApplyPatchFileOutput {
            path: path.display().to_string(),
            status: ApplyPatchFileStatus::Rejected,
            hunks_applied: 0,
            before: None,
            after: None,
            before_hash: None,
            after_hash: None,
            errors: vec![format!("{}: {message}", section.path)],
            validation_errors: Vec::new(),
        })
        .collect();
    ApplyPatchOutput { files }
}
/// Service for patching files with snapshot coordination
///
/// This service coordinates between infrastructure (file I/O) and repository
/// (snapshots) to modify files while preserving the ability to undo changes.
pub struct ForgeFsPatch<F> {
    infra: Arc<F>,
}

impl<F> ForgeFsPatch<F> {
    pub fn new(infra: Arc<F>) -> Self {
        Self { infra }
    }
}

#[async_trait::async_trait]
impl<
    F: EnvironmentInfra<Config = ForgeConfig>
        + FileWriterInfra
        + SnapshotRepository
        + ValidationRepository
        + FuzzySearchRepository
        + TextPatchRepository,
> FsPatchService for ForgeFsPatch<F>
{
    async fn patch(
        &self,
        input_path: String,
        search: String,
        content: String,
        replace_all: bool,
    ) -> anyhow::Result<PatchOutput> {
        let path = Path::new(&input_path);
        assert_absolute_path(path)?;

        // Convert replace_all boolean to PatchOperation
        let operation = if replace_all {
            PatchOperation::ReplaceAll
        } else {
            PatchOperation::Replace
        };

        // Read the original content once
        let mut current_content = fs::read_to_string(path)
            .await
            .map_err(Error::FileOperation)?;

        // Save the old content before modification for diff generation
        let old_content = current_content.clone();
        let use_text_patch_fallback = self.infra.get_config()?.use_text_patch_fallback;

        current_content = apply_replace_operation(
            &*self.infra,
            current_content,
            &search,
            &content,
            &operation,
            use_text_patch_fallback,
        )
        .await?;

        // SNAPSHOT COORDINATION: Always capture snapshot before modifying
        self.infra.insert_snapshot(path).await?;

        // Write final content to file after all patches are applied
        self.infra
            .write(path, Bytes::from(current_content.clone()))
            .await?;

        // Compute hash of the final file content
        let content_hash = compute_hash(&current_content);

        // Validate file syntax using remote validation API (graceful failure)
        let errors = self
            .infra
            .validate_file(path, &current_content)
            .await
            .unwrap_or_default();

        Ok(PatchOutput {
            errors,
            before: old_content,
            after: current_content,
            content_hash,
        })
    }

    async fn multi_patch(
        &self,
        input_path: String,
        edits: Vec<forge_domain::PatchEdit>,
    ) -> anyhow::Result<PatchOutput> {
        let path = Path::new(&input_path);
        assert_absolute_path(path)?;

        // Read the original content once
        let mut current_content = fs::read_to_string(path)
            .await
            .map_err(Error::FileOperation)?;
        // Save the old content before modification for diff generation
        let old_content = current_content.clone();
        let use_text_patch_fallback = self.infra.get_config()?.use_text_patch_fallback;

        // Apply each edit sequentially
        for edit in &edits {
            // Convert replace_all boolean to PatchOperation
            let operation = if edit.replace_all {
                PatchOperation::ReplaceAll
            } else {
                PatchOperation::Replace
            };

            current_content = apply_replace_operation(
                &*self.infra,
                current_content,
                &edit.old_string,
                &edit.new_string,
                &operation,
                use_text_patch_fallback,
            )
            .await?;
        }

        // SNAPSHOT COORDINATION: Always capture snapshot before modifying
        self.infra.insert_snapshot(path).await?;

        // Write final content to file after all patches are applied
        self.infra
            .write(path, Bytes::from(current_content.clone()))
            .await?;

        // Compute hash of the final file content
        let content_hash = compute_hash(&current_content);

        // Validate file syntax using remote validation API (graceful failure)
        let errors = self
            .infra
            .validate_file(path, &current_content)
            .await
            .unwrap_or_default();

        Ok(PatchOutput {
            errors,
            before: old_content,
            after: current_content,
            content_hash,
        })
    }

    async fn apply_patch(&self, patch: String) -> anyhow::Result<ApplyPatchOutput> {
        let sections = parse_apply_patch(&patch)?;
        let cwd = self.infra.get_environment().cwd;
        let mut paths = Vec::new();
        for section in &sections {
            let path = normalize_apply_patch_target(&section.path, cwd.as_path())?;
            assert_absolute_path(path.as_path())?;
            paths.push(path);
        }

        if sections
            .iter()
            .any(|section| matches!(section.operation, ApplyPatchOperation::Delete))
        {
            return Ok(rejected_apply_patch_output(
                &sections,
                &paths,
                "Delete File is parsed but not supported by this safe first slice",
            ));
        }

        let mut planned = Vec::new();
        for (section, path) in sections.iter().cloned().zip(paths.iter().cloned()) {
            match section.operation {
                ApplyPatchOperation::Update => {
                    ensure_target_inside_workspace(path.as_path(), cwd.as_path(), true)?;
                    let before = match fs::read_to_string(path.as_path()).await {
                        Ok(content) => content,
                        Err(error) => {
                            return Ok(rejected_apply_patch_output(
                                &sections,
                                &paths,
                                format!("Failed to read update target: {error}"),
                            ));
                        }
                    };
                    let (after, hunks_applied) = match apply_update_lines(&before, &section.lines) {
                        Ok(result) => result,
                        Err(error) => {
                            return Ok(rejected_apply_patch_output(
                                &sections,
                                &paths,
                                error.to_string(),
                            ));
                        }
                    };
                    planned.push(PlannedApplyPatchFile {
                        section,
                        absolute_path: path,
                        before: Some(before),
                        after: Some(after),
                        hunks_applied,
                    });
                }
                ApplyPatchOperation::Add => {
                    ensure_target_inside_workspace(path.as_path(), cwd.as_path(), false)?;
                    let parent = path.parent().ok_or_else(|| {
                        anyhow::anyhow!("Add File target has no parent: {}", path.display())
                    })?;
                    if !parent.exists() {
                        return Ok(rejected_apply_patch_output(
                            &sections,
                            &paths,
                            format!("Add File parent does not exist: {}", parent.display()),
                        ));
                    }
                    if path.exists() {
                        return Ok(rejected_apply_patch_output(
                            &sections,
                            &paths,
                            format!("Add File target already exists: {}", path.display()),
                        ));
                    }
                    let after = match parse_add_file_content(&section.lines) {
                        Ok(content) => content,
                        Err(error) => {
                            return Ok(rejected_apply_patch_output(
                                &sections,
                                &paths,
                                error.to_string(),
                            ));
                        }
                    };
                    planned.push(PlannedApplyPatchFile {
                        section,
                        absolute_path: path,
                        before: None,
                        after: Some(after),
                        hunks_applied: 0,
                    });
                }
                ApplyPatchOperation::Delete => unreachable!("delete was rejected before planning"),
            }
        }

        for file in &planned {
            if file.before.is_some() {
                self.infra
                    .insert_snapshot(file.absolute_path.as_path())
                    .await?;
            }
        }

        for file in &planned {
            let after = file.after.as_deref().unwrap_or_default();
            self.infra
                .write(file.absolute_path.as_path(), Bytes::from(after.to_string()))
                .await?;
        }

        let mut files = Vec::new();
        for file in planned {
            let after = file.after.unwrap_or_default();
            let validation_errors = self
                .infra
                .validate_file(file.absolute_path.as_path(), &after)
                .await
                .unwrap_or_default();
            let before_hash = file.before.as_ref().map(|content| compute_hash(content));
            let after_hash = compute_hash(&after);
            let status = match file.section.operation {
                ApplyPatchOperation::Update => ApplyPatchFileStatus::Updated,
                ApplyPatchOperation::Add => ApplyPatchFileStatus::Created,
                ApplyPatchOperation::Delete => ApplyPatchFileStatus::Rejected,
            };
            files.push(ApplyPatchFileOutput {
                path: file.absolute_path.display().to_string(),
                status,
                hunks_applied: file.hunks_applied,
                before: file.before,
                after: Some(after),
                before_hash,
                after_hash: Some(after_hash),
                errors: Vec::new(),
                validation_errors,
            });
        }

        Ok(ApplyPatchOutput { files })
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use bytes::Bytes;
    use forge_app::domain::PatchOperation;
    use forge_app::{ApplyPatchFileStatus, EnvironmentInfra, FileWriterInfra, FsPatchService};
    use forge_config::ForgeConfig;
    use forge_domain::{Environment, SearchMatch, Snapshot, TextPatchBlock};
    use pretty_assertions::assert_eq;

    use super::{ForgeFsPatch, lexical_normalized_path_key, parse_apply_patch};

    struct StubTextPatchRepository;

    #[async_trait::async_trait]
    impl forge_domain::FuzzySearchRepository for StubTextPatchRepository {
        async fn fuzzy_search(
            &self,
            _needle: &str,
            _haystack: &str,
            _mode: forge_domain::SearchMode,
        ) -> anyhow::Result<Vec<forge_domain::SearchMatch>> {
            Ok(vec![])
        }
    }

    #[async_trait::async_trait]
    impl forge_domain::TextPatchRepository for StubTextPatchRepository {
        async fn build_text_patch(
            &self,
            _haystack: &str,
            _old_string: &str,
            _new_string: &str,
        ) -> anyhow::Result<TextPatchBlock> {
            Ok(TextPatchBlock {
                patch: "synthetic patch".to_string(),
                patched_text: "patched by repository".to_string(),
            })
        }
    }

    #[tokio::test]
    async fn test_apply_replace_operation_uses_text_patch_for_missing_replace_only() {
        let fixture = StubTextPatchRepository;
        let source = "hello world".to_string();
        let operation = PatchOperation::Replace;

        let actual = super::apply_replace_operation(
            &fixture,
            source,
            "missing text",
            "replacement",
            &operation,
            true,
        )
        .await
        .unwrap();
        let expected = "patched by repository";

        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn test_apply_replace_operation_rejects_missing_replace_all() {
        let fixture = StubTextPatchRepository;
        let source = "hello world".to_string();
        let operation = PatchOperation::ReplaceAll;

        let actual = super::apply_replace_operation(
            &fixture,
            source,
            "missing text",
            "replacement",
            &operation,
            true,
        )
        .await;
        let expected = "Could not find match for search text: 'missing text'. File may have changed externally, consider reading the file again.";

        assert_eq!(actual.unwrap_err().to_string(), expected);
    }

    #[tokio::test]
    async fn test_apply_replace_operation_rejects_missing_swap() {
        let fixture = StubTextPatchRepository;
        let source = "hello world".to_string();
        let operation = PatchOperation::Swap;

        let actual = super::apply_replace_operation(
            &fixture,
            source,
            "missing text",
            "replacement",
            &operation,
            true,
        )
        .await;
        let expected = "Could not find match for search text: 'missing text'. File may have changed externally, consider reading the file again.";

        assert_eq!(actual.unwrap_err().to_string(), expected);
    }

    #[test]
    fn test_apply_replace_operation_uses_fuzzy_search_when_text_patch_fallback_disabled() {
        let fixture = tokio::runtime::Runtime::new().unwrap();

        let actual = fixture.block_on(super::apply_replace_operation(
            &FallbackRepository,
            "alpha\nbeta\ngamma".to_string(),
            "betaa",
            "delta",
            &PatchOperation::Replace,
            false,
        ));

        let expected = "alpha\ndelta\ngamma";
        assert_eq!(actual.unwrap(), expected);
    }

    #[test]
    fn test_apply_replace_operation_uses_text_patch_when_enabled() {
        let fixture = tokio::runtime::Runtime::new().unwrap();

        let actual = fixture.block_on(super::apply_replace_operation(
            &FallbackRepository,
            "alpha\nbeta\ngamma".to_string(),
            "betaa",
            "delta",
            &PatchOperation::Replace,
            true,
        ));

        let expected = "patched via text patch";
        assert_eq!(actual.unwrap(), expected);
    }

    #[derive(Default)]
    struct FallbackRepository;

    #[async_trait::async_trait]
    impl forge_domain::FuzzySearchRepository for FallbackRepository {
        async fn fuzzy_search(
            &self,
            _needle: &str,
            _haystack: &str,
            _mode: forge_domain::SearchMode,
        ) -> anyhow::Result<Vec<forge_domain::SearchMatch>> {
            let actual = vec![forge_domain::SearchMatch { start_line: 1, end_line: 1 }];
            Ok(actual)
        }
    }

    #[async_trait::async_trait]
    impl forge_domain::TextPatchRepository for FallbackRepository {
        async fn build_text_patch(
            &self,
            _haystack: &str,
            _old_string: &str,
            _new_string: &str,
        ) -> anyhow::Result<forge_domain::TextPatchBlock> {
            let actual = forge_domain::TextPatchBlock {
                patch: "@@ -1 +1 @@".to_string(),
                patched_text: "patched via text patch".to_string(),
            };
            Ok(actual)
        }
    }

    #[derive(Clone)]
    struct ApplyPatchTestInfra {
        cwd: PathBuf,
    }

    impl ApplyPatchTestInfra {
        fn new(cwd: PathBuf) -> Self {
            Self { cwd }
        }
    }

    impl EnvironmentInfra for ApplyPatchTestInfra {
        type Config = ForgeConfig;

        fn get_env_var(&self, _key: &str) -> Option<String> {
            None
        }

        fn get_env_vars(&self) -> std::collections::BTreeMap<String, String> {
            std::collections::BTreeMap::new()
        }

        fn get_environment(&self) -> Environment {
            Environment {
                os: "linux".to_string(),
                cwd: self.cwd.clone(),
                home: None,
                shell: "/bin/sh".to_string(),
                base_path: self.cwd.clone(),
            }
        }

        fn get_config(&self) -> anyhow::Result<Self::Config> {
            Ok(ForgeConfig::default())
        }

        async fn update_environment(
            &self,
            _ops: Vec<forge_domain::ConfigOperation>,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl FileWriterInfra for ApplyPatchTestInfra {
        async fn write(&self, path: &Path, contents: Bytes) -> anyhow::Result<()> {
            tokio::fs::write(path, contents).await?;
            Ok(())
        }

        async fn append(&self, path: &Path, contents: Bytes) -> anyhow::Result<()> {
            use tokio::io::AsyncWriteExt;

            let mut file = tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .await?;
            file.write_all(&contents).await?;
            Ok(())
        }

        async fn write_temp(
            &self,
            prefix: &str,
            ext: &str,
            content: &str,
        ) -> anyhow::Result<PathBuf> {
            let path = self.cwd.join(format!("{prefix}{ext}"));
            tokio::fs::write(&path, content).await?;
            Ok(path)
        }
    }

    #[async_trait::async_trait]
    impl forge_domain::SnapshotRepository for ApplyPatchTestInfra {
        async fn insert_snapshot(&self, file_path: &Path) -> anyhow::Result<Snapshot> {
            Snapshot::create(file_path.to_path_buf())
        }

        async fn undo_snapshot(&self, _file_path: &Path) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl forge_domain::ValidationRepository for ApplyPatchTestInfra {
        async fn validate_file(
            &self,
            _path: impl AsRef<Path> + Send,
            _content: &str,
        ) -> anyhow::Result<Vec<forge_domain::SyntaxError>> {
            Ok(Vec::new())
        }
    }

    #[async_trait::async_trait]
    impl forge_domain::FuzzySearchRepository for ApplyPatchTestInfra {
        async fn fuzzy_search(
            &self,
            _needle: &str,
            _haystack: &str,
            _mode: forge_domain::SearchMode,
        ) -> anyhow::Result<Vec<forge_domain::SearchMatch>> {
            Ok(Vec::new())
        }
    }

    #[async_trait::async_trait]
    impl forge_domain::TextPatchRepository for ApplyPatchTestInfra {
        async fn build_text_patch(
            &self,
            _haystack: &str,
            _old_string: &str,
            _new_string: &str,
        ) -> anyhow::Result<forge_domain::TextPatchBlock> {
            anyhow::bail!("unused")
        }
    }

    fn apply_patch_service(cwd: PathBuf) -> ForgeFsPatch<ApplyPatchTestInfra> {
        ForgeFsPatch::new(Arc::new(ApplyPatchTestInfra::new(cwd)))
    }

    fn patch_for(path: &Path, old: &str, new: &str) -> String {
        format!(
            "*** Begin Patch\n*** Update File: {}\n- {}\n+ {}\n*** End Patch\n",
            path.display(),
            old,
            new
        )
    }

    #[test]
    fn test_parse_apply_patch_accepts_two_update_sections() {
        let fixture = "*** Begin Patch\n*** Update File: a.txt\n- a\n+ b\n*** Update File: b.txt\n- c\n+ d\n*** End Patch\n";

        let actual = parse_apply_patch(fixture).unwrap();
        let expected = 2;

        assert_eq!(actual.len(), expected);
    }

    #[test]
    fn test_parse_apply_patch_rejects_malformed_begin_end() {
        let fixture = "*** Begin\n*** Update File: a.txt\n- a\n+ b\n*** End Patch\n";

        let actual = parse_apply_patch(fixture);

        assert!(actual.is_err());
    }

    #[test]
    fn test_parse_apply_patch_rejects_unknown_operation() {
        let fixture = "*** Begin Patch\n*** Move File: a.txt\n*** End Patch\n";

        let actual = parse_apply_patch(fixture);

        assert!(actual.is_err());
    }

    #[test]
    fn test_parse_apply_patch_rejects_duplicate_target() {
        let fixture = "*** Begin Patch\n*** Update File: a.txt\n- a\n+ b\n*** Update File: ./a.txt\n- c\n+ d\n*** End Patch\n";

        let actual = parse_apply_patch(fixture);

        assert!(actual.is_err());
    }

    #[test]
    fn test_apply_patch_rejects_path_traversal() {
        let fixture = "../outside.txt";

        let actual = lexical_normalized_path_key(fixture);

        assert!(actual.is_err());
    }

    #[tokio::test]
    async fn test_apply_patch_second_file_hunk_mismatch_leaves_first_file_unchanged() {
        let fixture = tempfile::tempdir().unwrap();
        let first = fixture.path().join("first.txt");
        let second = fixture.path().join("second.txt");
        tokio::fs::write(&first, "alpha\n").await.unwrap();
        tokio::fs::write(&second, "beta\n").await.unwrap();
        let service = apply_patch_service(fixture.path().to_path_buf());
        let patch = format!(
            "*** Begin Patch\n*** Update File: {}\n- alpha\n+ changed\n*** Update File: {}\n- missing\n+ changed\n*** End Patch\n",
            first.display(),
            second.display()
        );

        let actual = service.apply_patch(patch).await.unwrap();
        let expected_content = "alpha\n";

        assert_eq!(
            tokio::fs::read_to_string(&first).await.unwrap(),
            expected_content
        );
        assert_eq!(actual.files[0].status, ApplyPatchFileStatus::Rejected);
    }

    #[tokio::test]
    async fn test_apply_patch_add_file_success_and_collision_failure() {
        let fixture = tempfile::tempdir().unwrap();
        let target = fixture.path().join("new.txt");
        let service = apply_patch_service(fixture.path().to_path_buf());
        let patch = format!(
            "*** Begin Patch\n*** Add File: {}\n+ hello\n*** End Patch\n",
            target.display()
        );

        let actual = service.apply_patch(patch.clone()).await.unwrap();
        let expected = "hello\n";

        assert_eq!(tokio::fs::read_to_string(&target).await.unwrap(), expected);
        assert_eq!(actual.files[0].status, ApplyPatchFileStatus::Created);

        let actual = service.apply_patch(patch).await.unwrap();
        assert_eq!(actual.files[0].status, ApplyPatchFileStatus::Rejected);
    }

    #[tokio::test]
    async fn test_apply_patch_delete_fails_closed() {
        let fixture = tempfile::tempdir().unwrap();
        let target = fixture.path().join("delete.txt");
        tokio::fs::write(&target, "delete me\n").await.unwrap();
        let service = apply_patch_service(fixture.path().to_path_buf());
        let patch = format!(
            "*** Begin Patch\n*** Delete File: {}\n*** End Patch\n",
            target.display()
        );

        let actual = service.apply_patch(patch).await.unwrap();
        let expected = "delete me\n";

        assert_eq!(actual.files[0].status, ApplyPatchFileStatus::Rejected);
        assert_eq!(tokio::fs::read_to_string(&target).await.unwrap(), expected);
    }

    #[tokio::test]
    async fn test_apply_patch_output_includes_every_touched_file() {
        let fixture = tempfile::tempdir().unwrap();
        let first = fixture.path().join("first.txt");
        let second = fixture.path().join("second.txt");
        tokio::fs::write(&first, "alpha\n").await.unwrap();
        tokio::fs::write(&second, "beta\n").await.unwrap();
        let service = apply_patch_service(fixture.path().to_path_buf());
        let patch = format!(
            "*** Begin Patch\n*** Update File: {}\n- alpha\n+ one\n*** Update File: {}\n- beta\n+ two\n*** End Patch\n",
            first.display(),
            second.display()
        );

        let actual = service.apply_patch(patch).await.unwrap();
        let expected = 2;

        assert_eq!(actual.files.len(), expected);
        assert_eq!(actual.files[0].hunks_applied, 1);
        assert_eq!(actual.files[1].hunks_applied, 1);
        assert!(actual.files[0].before_hash.is_some());
        assert!(actual.files[1].after_hash.is_some());
    }

    #[tokio::test]
    async fn test_apply_patch_has_no_fuzzy_fallback_on_near_match() {
        let fixture = tempfile::tempdir().unwrap();
        let target = fixture.path().join("file.txt");
        tokio::fs::write(&target, "alpha\n").await.unwrap();
        let service = apply_patch_service(fixture.path().to_path_buf());
        let patch = patch_for(&target, "alphaa", "changed");

        let actual = service.apply_patch(patch).await.unwrap();
        let expected = "alpha\n";

        assert_eq!(actual.files[0].status, ApplyPatchFileStatus::Rejected);
        assert_eq!(tokio::fs::read_to_string(&target).await.unwrap(), expected);
    }

    #[test]
    fn test_range_from_search_match_single_line() {
        let source = "line1\nline2\nline3";
        // 0-based: line 1 (the second line, "line2")
        let search_match = SearchMatch { start_line: 1, end_line: 1 };

        let range = super::Range::from_search_match(source, &search_match);
        let actual = &source[range.start..range.end()];
        let expected = "line2";

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_range_from_search_match_multi_line() {
        let source = "line1\nline2\nline3\nline4";
        // 0-based: lines 1-2 (second and third lines, "line2\nline3")
        let search_match = SearchMatch { start_line: 1, end_line: 2 };

        let range = super::Range::from_search_match(source, &search_match);
        let actual = &source[range.start..range.end()];
        let expected = "line2\nline3";

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_range_from_search_match_first_line() {
        let source = "line1\nline2\nline3";
        // 0-based: line 0 (first line, "line1")
        let search_match = SearchMatch { start_line: 0, end_line: 0 };

        let range = super::Range::from_search_match(source, &search_match);
        let actual = &source[range.start..range.end()];
        let expected = "line1";

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_range_from_search_match_last_line() {
        let source = "line1\nline2\nline3";
        // 0-based: line 2 (third line, "line3")
        let search_match = SearchMatch { start_line: 2, end_line: 2 };

        let range = super::Range::from_search_match(source, &search_match);
        let actual = &source[range.start..range.end()];
        let expected = "line3";

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_range_from_search_match_last_line_without_newline() {
        let source = "line1\nline2\nline3"; // No trailing newline
        // 0-based: line 2 (third line, "line3")
        let search_match = SearchMatch { start_line: 2, end_line: 2 };

        let range = super::Range::from_search_match(source, &search_match);
        let actual = &source[range.start..range.end()];
        let expected = "line3";

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_range_from_search_match_all_lines() {
        let source = "line1\nline2\nline3";
        // 0-based: lines 0-2 (all three lines)
        let search_match = SearchMatch { start_line: 0, end_line: 2 };

        let range = super::Range::from_search_match(source, &search_match);
        let actual = &source[range.start..range.end()];
        let expected = "line1\nline2\nline3";

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_range_from_search_match_empty_source() {
        let source = "";
        // 0-based: line 0 (but source is empty)
        let search_match = SearchMatch { start_line: 0, end_line: 0 };

        let range = super::Range::from_search_match(source, &search_match);
        let actual = &source[range.start..range.end()];
        let expected = "";

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_range_from_search_match_single_line_source() {
        let source = "single line";
        // 0-based: line 0 (the only line)
        let search_match = SearchMatch { start_line: 0, end_line: 0 };

        let range = super::Range::from_search_match(source, &search_match);
        let actual = &source[range.start..range.end()];
        let expected = "single line";

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_apply_replacement_replace_multiple_matches_error() {
        let source = "test test test";
        let search = Some("test".to_string());
        let operation = PatchOperation::Replace;
        let content = "replaced";

        // Multiple matches error is detected inside apply_replacement, not in
        // compute_range
        let range = super::compute_range(source, search.as_deref(), &operation).unwrap();
        let result = super::apply_replacement(source.to_string(), range, &operation, content);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Multiple matches found for search text: 'test'. Either provide a more specific search pattern or use replace_all to replace all occurrences."));
    }

    #[test]
    fn test_apply_replacement_replace_single_match_success() {
        let source = "hello world test";
        let search = Some("world".to_string());
        let operation = PatchOperation::Replace;
        let content = "universe";

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            content,
        );
        assert_eq!(result.unwrap(), "hello universe test");
    }

    #[test]
    fn test_apply_replacement_prepend() {
        let source = "b\nc\nd";
        let search = Some("b".to_string());
        let operation = PatchOperation::Prepend;
        let content = "a\n".to_string();

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            &content,
        );
        assert_eq!(result.unwrap(), "a\nb\nc\nd");
    }

    #[test]
    fn test_apply_replacement_prepend_empty() {
        let source = "b\nc\nd";
        let search = Some("".to_string());
        let operation = PatchOperation::Prepend;
        let content = "a\n".to_string();

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            &content,
        );
        assert_eq!(result.unwrap(), "a\nb\nc\nd");
    }

    #[test]
    fn test_apply_replacement_prepend_no_search() {
        let source = "hello world";
        let search: Option<String> = None;
        let operation = PatchOperation::Prepend;
        let content = "prefix ";

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            content,
        );
        assert_eq!(result.unwrap(), "prefix hello world");
    }

    #[test]
    fn test_apply_replacement_append() {
        let source = "hello world";
        let search = Some("hello".to_string());
        let operation = PatchOperation::Append;
        let content = " there";

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            content,
        );
        assert_eq!(result.unwrap(), "hello\n there world");
    }

    #[test]
    fn test_apply_replacement_append_no_search() {
        let source = "hello world";
        let search: Option<String> = None;
        let operation = PatchOperation::Append;
        let content = " suffix";

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            content,
        );
        assert_eq!(result.unwrap(), "hello world\n suffix");
    }

    #[test]
    fn test_apply_replacement_replace() {
        let source = "hello world";
        let search = Some("world".to_string());
        let operation = PatchOperation::Replace;
        let content = "universe";

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            content,
        );
        assert_eq!(result.unwrap(), "hello universe");
    }

    #[test]
    fn test_apply_replacement_replace_no_search() {
        let source = "hello world";
        let search: Option<String> = None;
        let operation = PatchOperation::Replace;
        let content = "new content";

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            content,
        );
        assert_eq!(result.unwrap(), "new content");
    }

    #[test]
    fn test_apply_replacement_swap() {
        let source = "apple banana cherry";
        let search = Some("apple".to_string());
        let operation = PatchOperation::Swap;
        let content = "banana";

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            content,
        );
        assert_eq!(result.unwrap(), "banana apple cherry");
    }

    #[test]
    fn test_apply_replacement_swap_reverse_order() {
        let source = "apple banana cherry";
        let search = Some("banana".to_string());
        let operation = PatchOperation::Swap;
        let content = "apple";

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            content,
        );
        assert_eq!(result.unwrap(), "banana apple cherry");
    }

    #[test]
    fn test_apply_replacement_swap_normalizes_target_line_endings() {
        let source = "before\r\nleft\r\nseparator\r\nright\r\nafter";
        let search = Some("left".to_string());
        let operation = PatchOperation::Swap;
        let content = "right\nafter";

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            content,
        );
        let expected = "before\r\nright\r\nafter\r\nseparator\r\nleft";

        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_apply_replacement_swap_overlapping() {
        let source = "abcdef";
        let search = Some("abc".to_string());
        let operation = PatchOperation::Swap;
        let content = "cde";

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            content,
        );
        assert_eq!(result.unwrap(), "cdedef");
    }

    #[test]
    fn test_apply_replacement_swap_no_search() {
        let source = "hello world";
        let search: Option<String> = None;
        let operation = PatchOperation::Swap;
        let content = "anything";

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            content,
        );
        assert_eq!(result.unwrap(), "hello world");
    }

    #[test]
    fn test_apply_replacement_multiline() {
        let source = "line1\nline2\nline3";
        let search = Some("line2".to_string());
        let operation = PatchOperation::Replace;
        let content = "replaced_line";

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            content,
        );
        assert_eq!(result.unwrap(), "line1\nreplaced_line\nline3");
    }

    #[test]
    fn test_apply_replacement_with_special_chars() {
        let source = "hello $world @test";
        let search = Some("$world".to_string());
        let operation = PatchOperation::Replace;
        let content = "$universe";

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            content,
        );
        assert_eq!(result.unwrap(), "hello $universe @test");
    }

    #[test]
    fn test_apply_replacement_empty_content() {
        let source = "hello world test";
        let search = Some("world ".to_string());
        let operation = PatchOperation::Replace;
        let content = "";

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            content,
        );
        assert_eq!(result.unwrap(), "hello test");
    }

    #[test]
    fn test_apply_replacement_first_occurrence_only() {
        let source = "test test test";
        let search = Some("test".to_string());
        let operation = PatchOperation::Replace;
        let content = "replaced";

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            content,
        );
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Multiple matches found for search text: 'test'")
        );
    }

    // Error cases
    #[test]
    fn test_apply_replacement_no_match() {
        let source = "hello world";
        let search = Some("missing".to_string());
        let operation = PatchOperation::Replace;
        let _content = "replacement";

        let range = super::compute_range(source, search.as_deref(), &operation);
        assert!(range.is_err());
        assert!(
            range
                .unwrap_err()
                .to_string()
                .contains("Could not find match for search text: 'missing'")
        );
    }

    #[test]
    fn test_apply_replacement_swap_no_target() {
        let source = "hello world";
        let search = Some("hello".to_string());
        let operation = PatchOperation::Swap;
        let content = "missing";

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            content,
        );
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Could not find swap target text: missing")
        );
    }

    #[test]
    fn test_apply_replacement_edge_case_same_text() {
        let source = "hello hello";
        let search = Some("hello".to_string());
        let operation = PatchOperation::Swap;
        let content = "hello";

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            content,
        );
        assert_eq!(result.unwrap(), "hello hello");
    }

    #[test]
    fn test_apply_replacement_whitespace_handling() {
        let source = "  hello   world  ";
        let search = Some("hello   world".to_string());
        let operation = PatchOperation::Replace;
        let content = "test";

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            content,
        );
        assert_eq!(result.unwrap(), "  test  ");
    }

    #[test]
    fn test_apply_replacement_unicode() {
        let source = "héllo wørld 🌍";
        let search = Some("wørld".to_string());
        let operation = PatchOperation::Replace;
        let content = "univérse";

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            content,
        );
        assert_eq!(result.unwrap(), "héllo univérse 🌍");
    }

    #[test]
    fn test_apply_replacement_replace_all_multiple_occurrences() {
        let source = "test test test";
        let search = Some("test".to_string());
        let operation = PatchOperation::ReplaceAll;
        let content = "replaced";

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            content,
        );
        assert_eq!(result.unwrap(), "replaced replaced replaced");
    }

    #[test]
    fn test_apply_replacement_replace_all_no_search() {
        let source = "hello world";
        let search: Option<String> = None;
        let operation = PatchOperation::ReplaceAll;
        let content = "new content";

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            content,
        );
        assert_eq!(result.unwrap(), "new content");
    }

    #[test]
    fn test_apply_replacement_replace_all_empty_search() {
        let source = "hello world";
        let search = Some("".to_string());
        let operation = PatchOperation::ReplaceAll;
        let content = "new content";

        let result = super::apply_replacement(
            source.to_string(),
            super::compute_range(source, search.as_deref(), &operation).unwrap(),
            &operation,
            content,
        );
        assert_eq!(result.unwrap(), "new content");
    }

    #[test]
    fn test_apply_replacement_replace_all_no_match() {
        let source = "hello world";
        let search = Some("missing".to_string());
        let operation = PatchOperation::ReplaceAll;
        let _content = "replacement";

        let range = super::compute_range(source, search.as_deref(), &operation);
        assert!(range.is_err());
        assert!(
            range
                .unwrap_err()
                .to_string()
                .contains("Could not find match for search text: 'missing'")
        );
    }

    #[test]
    fn test_range_from_search_match_crlf_single_line() {
        let source = "line1\r\nline2\r\nline3";
        // 0-based: line 1 (the second line, "line2")
        let search_match = SearchMatch { start_line: 1, end_line: 1 };

        let range = super::Range::from_search_match(source, &search_match);
        let actual = &source[range.start..range.end()];
        let expected = "line2";

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_range_from_search_match_crlf_multi_line() {
        let source = "line1\r\nline2\r\nline3\r\nline4";
        // 0-based: lines 1-2 (second and third lines, "line2\r\nline3")
        let search_match = SearchMatch { start_line: 1, end_line: 2 };

        let range = super::Range::from_search_match(source, &search_match);
        let actual = &source[range.start..range.end()];
        let expected = "line2\r\nline3";

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_range_from_search_match_crlf_first_line() {
        let source = "line1\r\nline2\r\nline3";
        // 0-based: line 0 (first line, "line1")
        let search_match = SearchMatch { start_line: 0, end_line: 0 };

        let range = super::Range::from_search_match(source, &search_match);
        let actual = &source[range.start..range.end()];
        let expected = "line1";

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_range_from_search_match_crlf_all_lines() {
        let source = "line1\r\nline2\r\nline3";
        // 0-based: lines 0-2 (all three lines)
        let search_match = SearchMatch { start_line: 0, end_line: 2 };

        let range = super::Range::from_search_match(source, &search_match);
        let actual = &source[range.start..range.end()];
        let expected = "line1\r\nline2\r\nline3";

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_detect_line_ending_crlf() {
        let source = "line1\r\nline2\r\nline3";
        let line_ending = super::Range::detect_line_ending(source);
        assert_eq!(line_ending, "\r\n");
    }

    #[test]
    fn test_detect_line_ending_lf() {
        let source = "line1\nline2\nline3";
        let line_ending = super::Range::detect_line_ending(source);
        assert_eq!(line_ending, "\n");
    }

    #[test]
    fn test_compute_range_normalizes_search_crlf() {
        let source = "line1\r\nline2\r\nline3";
        let search = Some("line2\nline3".to_string());
        let operation = PatchOperation::Replace;

        let range = super::compute_range(source, search.as_deref(), &operation).unwrap();
        let actual = &source[range.unwrap().start..range.unwrap().end()];
        let expected = "line2\r\nline3";

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_compute_range_normalizes_search_lf() {
        let source = "line1\nline2\nline3";
        let search = Some("line2\r\nline3".to_string());
        let operation = PatchOperation::Replace;

        let range = super::compute_range(source, search.as_deref(), &operation).unwrap();
        let actual = &source[range.unwrap().start..range.unwrap().end()];
        let expected = "line2\nline3";

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_compute_range_normalizes_search_crlf_input() {
        let source = "line1\r\nline2\r\nline3";
        let search = Some("line2\r\nline3".to_string());
        let operation = PatchOperation::Replace;

        let range = super::compute_range(source, search.as_deref(), &operation).unwrap();
        let actual = &source[range.unwrap().start..range.unwrap().end()];
        let expected = "line2\r\nline3";

        assert_eq!(actual, expected);
    }

    // --- Out-of-bounds safety tests ---

    #[test]
    fn test_range_from_search_match_out_of_bounds_start_line() {
        let source = "line1\nline2\nline3";
        // start_line way past end of file
        let search_match = SearchMatch { start_line: 100, end_line: 200 };

        let range = super::Range::from_search_match(source, &search_match);
        // Should not panic; range should be clamped so it doesn't exceed source
        assert!(range.end() <= source.len());
    }

    #[test]
    fn test_range_from_search_match_end_line_past_eof() {
        let source = "line1\nline2\nline3";
        // start_line valid, end_line past end
        let search_match = SearchMatch { start_line: 1, end_line: 100 };

        let range = super::Range::from_search_match(source, &search_match);
        assert!(range.end() <= source.len());
        // Should include from line2 to end of source
        let actual = &source[range.start..range.end()];
        assert!(actual.contains("line2"));
        assert!(actual.contains("line3"));
    }

    #[test]
    fn test_range_from_search_match_trailing_newline() {
        let source = "line1\nline2\nline3\n"; // trailing newline
        let search_match = SearchMatch { start_line: 2, end_line: 2 };

        let range = super::Range::from_search_match(source, &search_match);
        assert!(range.end() <= source.len());
        let actual = &source[range.start..range.end()];
        assert_eq!(actual, "line3");
    }

    #[test]
    fn test_range_from_search_match_unicode_content() {
        let source = "héllo\nwørld\n🌍";
        let search_match = SearchMatch { start_line: 1, end_line: 1 };

        let range = super::Range::from_search_match(source, &search_match);
        assert!(range.end() <= source.len());
        let actual = &source[range.start..range.end()];
        assert_eq!(actual, "wørld");
    }

    #[test]
    fn test_range_from_search_match_unicode_multiline() {
        let source = "héllo\nwørld\n🌍";
        let search_match = SearchMatch { start_line: 0, end_line: 2 };

        let range = super::Range::from_search_match(source, &search_match);
        assert!(range.end() <= source.len());
        let actual = &source[range.start..range.end()];
        assert_eq!(actual, source);
    }

    #[test]
    fn test_range_from_search_match_mixed_line_endings() {
        let source = "line1\r\nline2\nline3";
        let search_match = SearchMatch { start_line: 1, end_line: 1 };

        let range = super::Range::from_search_match(source, &search_match);
        assert!(range.end() <= source.len());
        let actual = &source[range.start..range.end()];
        assert_eq!(actual, "line2");
    }

    #[test]
    fn test_apply_replacement_with_out_of_bounds_range_returns_error() {
        let source = "short";
        // Simulate a bad range that exceeds source length
        let bad_range = Some(super::Range::new(0, 1000));
        let operation = PatchOperation::Replace;
        let content = "replacement";

        let result = super::apply_replacement(source.to_string(), bad_range, &operation, content);
        assert!(result.is_err());
    }
}
