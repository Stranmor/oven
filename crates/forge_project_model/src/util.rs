//! Shared deterministic helpers for project-model modules.

use std::path::Path;

use sha2::{Digest, Sha256};

use crate::types::{
    EdgeConfidence, ExternalFactBatchMetadata, GraphEdge, GraphEdgeKind, Language, Provenance,
    SourceFile,
};

/// Builds a redaction-safe SHA-256 fingerprint for arbitrary sensitive text.
///
/// # Arguments
///
/// * `text` - Sensitive or non-sensitive text that must not be persisted raw.
pub fn fingerprint(text: &str) -> String {
    hash_text(text)
}

pub(crate) fn edge(
    from: &str,
    to: &str,
    kind: GraphEdgeKind,
    confidence: f32,
    confidence_kind: EdgeConfidence,
    provenance: Provenance,
) -> GraphEdge {
    GraphEdge {
        from: from.to_string(),
        to: to.to_string(),
        kind,
        confidence,
        confidence_kind,
        provenance,
    }
}

pub(crate) fn provenance(
    path: &str,
    start_line: Option<u32>,
    end_line: Option<u32>,
    source: &str,
    fingerprint_seed: &str,
) -> Provenance {
    Provenance {
        path: path.to_string(),
        start_line,
        end_line,
        source: source.to_string(),
        fingerprint: fingerprint(fingerprint_seed),
    }
}

pub(crate) fn edge_sort_key(edge: &GraphEdge) -> (String, String, GraphEdgeKind) {
    (edge.from.clone(), edge.to.clone(), edge.kind.clone())
}

pub(crate) fn detect_language(path: &str) -> Language {
    match Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
    {
        Some("rs") => Language::Rust,
        Some("toml") => Language::Toml,
        Some("md") => Language::Markdown,
        Some("json") => Language::Json,
        _ => Language::Unknown,
    }
}

pub(crate) fn normalize_path(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

pub(crate) fn line_count(content: &str) -> u32 {
    u32::try_from(content.lines().count()).unwrap_or(u32::MAX)
}

pub(crate) fn ranges_overlap(
    left_start: u32,
    left_end: u32,
    right_start: u32,
    right_end: u32,
) -> bool {
    left_start <= right_end && right_start <= left_end
}

pub(crate) fn line_number_from_index(index: usize) -> Option<u32> {
    index
        .checked_add(1)
        .and_then(|line| u32::try_from(line).ok())
}

pub(crate) fn hash_text(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub(crate) fn manifest_hash(
    files: &[SourceFile],
    external_fact_batches: &[ExternalFactBatchMetadata],
    external_facts_fingerprint: &str,
) -> String {
    let mut content = String::new();
    for file in files {
        content.push_str(&file.path);
        content.push('\0');
        content.push_str(&file.content_hash);
        content.push('\0');
        content.push_str(&file.lines.to_string());
        content.push('\n');
    }
    content.push_str("external_facts_fingerprint");
    content.push('\0');
    content.push_str(external_facts_fingerprint);
    content.push('\n');
    for batch in external_fact_batches {
        content.push_str(&batch.batch_fingerprint);
        content.push('\0');
        content.push_str(&batch.source_label);
        content.push('\0');
        content.push_str(&batch.source_artifact_fingerprint);
        content.push('\n');
    }
    hash_text(&content)
}

pub(crate) fn external_facts_fingerprint(batches: &[ExternalFactBatchMetadata]) -> String {
    let mut content = String::new();
    for batch in batches {
        content.push_str(&batch.batch_fingerprint);
        content.push('\0');
        content.push_str(&batch.source_label);
        content.push('\0');
        content.push_str(batch.tool_version.as_deref().unwrap_or_default());
        content.push('\0');
        if !batch.producer_snapshot_fingerprint.is_empty() {
            content.push_str(&batch.producer_snapshot_fingerprint);
            content.push('\0');
        }
        content.push_str(&batch.workspace_root);
        content.push('\0');
        content.push_str(&batch.source_artifact_fingerprint);
        content.push('\0');
        content.push_str(&batch.manifest_hash_input);
        content.push('\n');
    }
    hash_text(&content)
}
