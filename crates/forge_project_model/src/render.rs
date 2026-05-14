//! Budgeted rendering for project-model context injected into model requests.

use crate::fingerprint;

/// Default maximum number of evidence sources rendered for automatic context
/// injection.
pub const DEFAULT_RENDERED_SOURCE_LIMIT: usize = 3;

/// A typed rendering budget for dynamic project-model context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectModelContextRenderBudget {
    /// Maximum number of evidence sources to render.
    pub max_sources: usize,
    /// Maximum number of content characters rendered for a single source.
    pub max_source_content_chars: usize,
    /// Maximum number of content characters rendered across all sources.
    pub max_total_content_chars: usize,
    /// Maximum number of content lines rendered for a single source.
    pub max_source_lines: usize,
    /// Maximum total rendered XML characters for the complete context payload.
    pub max_rendered_chars: usize,
    /// Maximum characters retained for any metadata attribute value before
    /// preview truncation.
    pub max_metadata_attr_chars: usize,
}

impl Default for ProjectModelContextRenderBudget {
    fn default() -> Self {
        Self {
            max_sources: DEFAULT_RENDERED_SOURCE_LIMIT,
            max_source_content_chars: 320,
            max_total_content_chars: 900,
            max_source_lines: 12,
            max_rendered_chars: 4_000,
            max_metadata_attr_chars: 160,
        }
    }
}

/// A typed project-model context source prepared by an adapter layer.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectModelContextSource {
    /// Relative source path or synthetic evidence path.
    pub path: String,
    /// One-based inclusive start line when known.
    pub start_line: Option<u32>,
    /// One-based inclusive end line when known.
    pub end_line: Option<u32>,
    /// Retrieval or ranking score when known.
    pub score: Option<f32>,
    /// Freshness label for this source.
    pub freshness: String,
    /// Provenance label for this source.
    pub provenance: String,
    /// Stable project-model or retrieval node identifier.
    pub node_id: String,
    /// Optional content hash supplied by the source system.
    pub content_hash: Option<String>,
    /// Optional source content that may be previewed under budget.
    pub content: Option<String>,
    /// Optional reason why content must be metadata-only.
    pub metadata_only_reason: Option<String>,
}

impl ProjectModelContextSource {
    /// Creates a project-model context source with stable identity fields.
    ///
    /// # Arguments
    ///
    /// * `path` - Relative path or synthetic evidence path.
    /// * `freshness` - Freshness label supplied by the caller.
    /// * `provenance` - Provenance label supplied by the caller.
    /// * `node_id` - Stable source node identifier.
    pub fn new(
        path: impl Into<String>,
        freshness: impl Into<String>,
        provenance: impl Into<String>,
        node_id: impl Into<String>,
    ) -> Self {
        Self {
            path: path.into(),
            start_line: None,
            end_line: None,
            score: None,
            freshness: freshness.into(),
            provenance: provenance.into(),
            node_id: node_id.into(),
            content_hash: None,
            content: None,
            metadata_only_reason: None,
        }
    }

    /// Attaches a one-based inclusive line range.
    ///
    /// # Arguments
    ///
    /// * `start_line` - First line covered by this source.
    /// * `end_line` - Last line covered by this source.
    pub fn line_range(mut self, start_line: u32, end_line: u32) -> Self {
        self.start_line = Some(start_line);
        self.end_line = Some(end_line);
        self
    }

    /// Attaches a retrieval score.
    ///
    /// # Arguments
    ///
    /// * `score` - Relevance score supplied by retrieval.
    pub fn score(mut self, score: Option<f32>) -> Self {
        self.score = score;
        self
    }

    /// Attaches a source-system content hash.
    ///
    /// # Arguments
    ///
    /// * `content_hash` - Redaction-safe content hash supplied by the source
    ///   system.
    pub fn content_hash(mut self, content_hash: impl Into<String>) -> Self {
        self.content_hash = Some(content_hash.into());
        self
    }

    /// Attaches source content that may be rendered under budget.
    ///
    /// # Arguments
    ///
    /// * `content` - Source text to preserve exactly when it is cheap enough.
    pub fn content(mut self, content: impl Into<String>) -> Self {
        self.content = Some(content.into());
        self
    }

    /// Marks this source as metadata-only.
    ///
    /// # Arguments
    ///
    /// * `reason` - Stable reason explaining why content was omitted.
    pub fn metadata_only(mut self, reason: impl Into<String>) -> Self {
        self.metadata_only_reason = Some(reason.into());
        self
    }
}

/// Renders dynamic project-model context under a typed budget.
///
/// # Arguments
///
/// * `workspace_root` - Display path for the workspace root.
/// * `manifest_path` - Display path for the local project-model manifest.
/// * `freshness` - Root freshness label.
/// * `provenance` - Root provenance label.
/// * `sources` - Candidate evidence sources in caller-selected ranking order.
/// * `budget` - Rendering budget that bounds sources, lines, and characters.
pub fn render_project_model_context(
    workspace_root: &str,
    manifest_path: &str,
    freshness: &str,
    provenance: &str,
    sources: &[ProjectModelContextSource],
    budget: &ProjectModelContextRenderBudget,
) -> String {
    let rendered = render_project_model_context_inner(
        workspace_root,
        manifest_path,
        freshness,
        provenance,
        sources,
        budget,
        false,
    );
    if rendered.chars().count() <= budget.max_rendered_chars {
        return rendered;
    }
    let metadata_only = render_project_model_context_inner(
        workspace_root,
        manifest_path,
        freshness,
        provenance,
        sources,
        budget,
        true,
    );
    if metadata_only.chars().count() <= budget.max_rendered_chars {
        return metadata_only;
    }
    let minimal =
        render_minimal_project_model_context(workspace_root, manifest_path, freshness, provenance);
    truncate_rendered_context(minimal, budget.max_rendered_chars)
}

fn truncate_rendered_context(rendered: String, max_chars: usize) -> String {
    if rendered.chars().count() <= max_chars {
        return rendered;
    }
    "<project_model_context omitted_reason=\"rendered_context_budget_exceeded\" />".to_string()
}

fn render_project_model_context_inner(
    workspace_root: &str,
    manifest_path: &str,
    freshness: &str,
    provenance: &str,
    sources: &[ProjectModelContextSource],
    budget: &ProjectModelContextRenderBudget,
    force_metadata_only: bool,
) -> String {
    let mut rendered = String::new();
    rendered.push_str(&format!(
        "<project_model_context workspace_root=\"{}\" manifest_path=\"{}\" freshness=\"{}\" provenance=\"{}\" rendered_source_limit=\"{}\" max_total_content_chars=\"{}\">",
        xml_attr(workspace_root),
        xml_attr(manifest_path),
        xml_attr(freshness),
        xml_attr(provenance),
        budget.max_sources,
        budget.max_total_content_chars,
    ));

    let mut remaining_content_chars = budget.max_total_content_chars;
    for source in sources.iter().take(budget.max_sources) {
        let rendered_content =
            render_source_content(source, budget, remaining_content_chars, force_metadata_only);
        remaining_content_chars =
            remaining_content_chars.saturating_sub(rendered_content.used_chars);
        rendered.push_str(&render_source(source, rendered_content, budget));
    }
    rendered.push_str("</project_model_context>");
    rendered
}

fn render_minimal_project_model_context(
    workspace_root: &str,
    manifest_path: &str,
    freshness: &str,
    provenance: &str,
) -> String {
    format!(
        "<project_model_context workspace_root=\"{}\" manifest_path=\"{}\" freshness=\"{}\" provenance=\"{}\" omitted_reason=\"rendered_context_budget_exceeded\" />",
        xml_attr(truncate_attr(workspace_root, 64)),
        xml_attr(truncate_attr(manifest_path, 64)),
        xml_attr(truncate_attr(freshness, 64)),
        xml_attr(truncate_attr(provenance, 64)),
    )
}

struct RenderedSourceContent {
    content_digest: String,
    body: Option<String>,
    used_chars: usize,
    truncated_reason: Option<String>,
    omitted_reason: Option<String>,
}

fn render_source_content(
    source: &ProjectModelContextSource,
    budget: &ProjectModelContextRenderBudget,
    remaining_total_chars: usize,
    force_metadata_only: bool,
) -> RenderedSourceContent {
    let Some(content) = &source.content else {
        return RenderedSourceContent {
            content_digest: source
                .content_hash
                .clone()
                .unwrap_or_else(|| fingerprint(&source.node_id)),
            body: None,
            used_chars: 0,
            truncated_reason: None,
            omitted_reason: Some(
                source
                    .metadata_only_reason
                    .clone()
                    .unwrap_or_else(|| "metadata_only_source".to_string()),
            ),
        };
    };

    let digest = fingerprint(content);
    if force_metadata_only {
        return RenderedSourceContent {
            content_digest: digest,
            body: None,
            used_chars: 0,
            truncated_reason: None,
            omitted_reason: Some("rendered_context_budget_exceeded".to_string()),
        };
    }
    if let Some(reason) = &source.metadata_only_reason {
        return RenderedSourceContent {
            content_digest: digest,
            body: None,
            used_chars: 0,
            truncated_reason: None,
            omitted_reason: Some(reason.clone()),
        };
    }
    if remaining_total_chars == 0 {
        return RenderedSourceContent {
            content_digest: digest,
            body: None,
            used_chars: 0,
            truncated_reason: None,
            omitted_reason: Some("total_content_budget_exhausted".to_string()),
        };
    }

    let max_chars = budget.max_source_content_chars.min(remaining_total_chars);
    let preview = truncate_content(content, max_chars, budget.max_source_lines);
    RenderedSourceContent {
        content_digest: digest,
        used_chars: preview.content.chars().count(),
        truncated_reason: preview.reason,
        body: Some(preview.content),
        omitted_reason: None,
    }
}

struct ContentPreview {
    content: String,
    reason: Option<String>,
}

fn truncate_content(content: &str, max_chars: usize, max_lines: usize) -> ContentPreview {
    let total_chars = content.chars().count();
    let total_lines = content.lines().count();
    if total_chars <= max_chars && total_lines <= max_lines {
        return ContentPreview { content: content.to_string(), reason: None };
    }

    let mut preview = String::new();
    for (line_index, line) in content.lines().enumerate() {
        if line_index >= max_lines {
            break;
        }
        for character in line.chars() {
            if preview.chars().count() >= max_chars {
                return ContentPreview {
                    content: preview,
                    reason: Some("content_char_budget_exceeded".to_string()),
                };
            }
            preview.push(character);
        }
        if line_index.saturating_add(1) < max_lines && preview.chars().count() < max_chars {
            preview.push('\n');
        }
    }

    let reason = if total_lines > max_lines {
        "content_line_budget_exceeded"
    } else {
        "content_char_budget_exceeded"
    };
    ContentPreview { content: preview, reason: Some(reason.to_string()) }
}

fn render_source(
    source: &ProjectModelContextSource,
    rendered_content: RenderedSourceContent,
    budget: &ProjectModelContextRenderBudget,
) -> String {
    let mut attrs = vec![
        (
            "path",
            xml_attr(truncate_attr(&source.path, budget.max_metadata_attr_chars)),
        ),
        (
            "start_line",
            source
                .start_line
                .map_or_else(|| "unknown".to_string(), |line| line.to_string()),
        ),
        (
            "end_line",
            source
                .end_line
                .map_or_else(|| "unknown".to_string(), |line| line.to_string()),
        ),
        (
            "score",
            source
                .score
                .map_or_else(|| "unknown".to_string(), |score| format!("{score:.6}")),
        ),
        (
            "freshness",
            xml_attr(truncate_attr(
                &source.freshness,
                budget.max_metadata_attr_chars,
            )),
        ),
        (
            "provenance",
            xml_attr(truncate_attr(
                &source.provenance,
                budget.max_metadata_attr_chars,
            )),
        ),
        (
            "node_id",
            xml_attr(truncate_attr(
                &source.node_id,
                budget.max_metadata_attr_chars,
            )),
        ),
        ("content_digest", xml_attr(&rendered_content.content_digest)),
    ];
    if let Some(reason) = rendered_content.truncated_reason {
        attrs.push(("truncated_reason", xml_attr(reason)));
    }
    if let Some(reason) = rendered_content.omitted_reason {
        attrs.push(("omitted_reason", xml_attr(reason)));
    }
    if let Some(hash) = &source.content_hash {
        attrs.push((
            "content_hash",
            xml_attr(truncate_attr(hash, budget.max_metadata_attr_chars)),
        ));
    }
    if source.path.chars().count() > budget.max_metadata_attr_chars
        || source.freshness.chars().count() > budget.max_metadata_attr_chars
        || source.provenance.chars().count() > budget.max_metadata_attr_chars
        || source.node_id.chars().count() > budget.max_metadata_attr_chars
        || source
            .content_hash
            .as_ref()
            .is_some_and(|hash| hash.chars().count() > budget.max_metadata_attr_chars)
    {
        attrs.push((
            "metadata_truncated_reason",
            "metadata_attr_budget_exceeded".to_string(),
        ));
    }

    let mut rendered = String::from("<source");
    for (name, value) in attrs {
        rendered.push_str(&format!(" {name}=\"{value}\""));
    }
    match rendered_content.body {
        Some(body) => {
            rendered.push_str("><symbol_or_content><![CDATA[");
            rendered.push_str(&xml_cdata(body));
            rendered.push_str("]]></symbol_or_content></source>");
        }
        None => rendered.push_str(" />"),
    }
    rendered
}

fn truncate_attr(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    value.chars().take(max_chars).collect()
}

fn xml_attr(value: impl ToString) -> String {
    value
        .to_string()
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn xml_cdata(value: impl ToString) -> String {
    value.to_string().replace("]]>", "]]]]><![CDATA[>")
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    fn fixture_source(content: impl Into<String>) -> ProjectModelContextSource {
        ProjectModelContextSource::new(
            "src/lib.rs",
            "manifest_snapshot",
            "local_project_model_manifest",
            "symbol:src/lib.rs:Function:needle",
        )
        .line_range(3, 9)
        .score(Some(0.875))
        .content(content)
    }

    #[test]
    fn renderer_preserves_small_snippets_exactly_with_provenance() {
        let setup = vec![fixture_source("pub fn needle() -> usize { 42 }")];
        let actual = render_project_model_context(
            "/workspace",
            "/workspace/.forge_project_model/project_manifest.json",
            "local_manifest_available",
            "WorkspaceService::query_workspace",
            &setup,
            &ProjectModelContextRenderBudget::default(),
        );
        let expected = (true, true, true, true, true, true, false);

        assert_eq!(
            (
                actual.contains("<project_model_context"),
                actual.contains("path=\"src/lib.rs\""),
                actual.contains("start_line=\"3\""),
                actual.contains("score=\"0.875000\""),
                actual.contains("content_digest=\""),
                actual.contains("pub fn needle() -> usize { 42 }"),
                actual.contains("truncated_reason"),
            ),
            expected,
        );
    }

    #[test]
    fn renderer_limits_sources_and_total_rendered_size() {
        let setup = vec![
            fixture_source("one"),
            fixture_source("two"),
            fixture_source("three"),
            fixture_source("four"),
        ];
        let budget = ProjectModelContextRenderBudget {
            max_sources: 3,
            max_source_content_chars: 10,
            max_total_content_chars: 30,
            max_source_lines: 2,
            max_rendered_chars: 4_000,
            max_metadata_attr_chars: 160,
        };

        let actual = render_project_model_context(
            "/workspace",
            "/manifest",
            "fresh",
            "test",
            &setup,
            &budget,
        );
        let expected = (3usize, true);

        assert_eq!(
            (
                actual.matches("<source").count(),
                actual.chars().count() <= budget.max_rendered_chars
            ),
            expected,
        );
    }

    #[test]
    fn renderer_truncates_long_chunks_and_keeps_digest_line_path_provenance() {
        let setup = vec![fixture_source("line1\nline2\nline3\nline4")];
        let budget = ProjectModelContextRenderBudget {
            max_sources: 3,
            max_source_content_chars: 128,
            max_total_content_chars: 128,
            max_source_lines: 2,
            max_rendered_chars: 4_000,
            max_metadata_attr_chars: 160,
        };

        let actual = render_project_model_context(
            "/workspace",
            "/manifest",
            "fresh",
            "test",
            &setup,
            &budget,
        );
        let expected = (true, true, true, true, false);

        assert_eq!(
            (
                actual.contains("truncated_reason=\"content_line_budget_exceeded\""),
                actual.contains("content_digest=\""),
                actual.contains("start_line=\"3\""),
                actual.contains("provenance=\"local_project_model_manifest\""),
                actual.contains("line4"),
            ),
            expected,
        );
    }

    #[test]
    fn renderer_uses_metadata_only_for_explicit_large_evidence() {
        let setup =
            vec![fixture_source("expensive full file").metadata_only("whole_file_metadata_only")];

        let actual = render_project_model_context(
            "/workspace",
            "/manifest",
            "fresh",
            "test",
            &setup,
            &ProjectModelContextRenderBudget::default(),
        );
        let expected = (true, true, false);

        assert_eq!(
            (
                actual.contains("omitted_reason=\"whole_file_metadata_only\""),
                actual.contains("content_digest=\""),
                actual.contains("expensive full file"),
            ),
            expected,
        );
    }

    #[test]
    fn renderer_uses_valid_minimal_context_when_budget_is_smaller_than_minimal_xml() {
        let setup = vec![
            ProjectModelContextSource::new(
                "src/large.rs",
                "manifest_snapshot",
                "local_project_model_manifest",
                "node:".to_string() + &"x".repeat(2_000),
            )
            .line_range(1, 1)
            .score(Some(1.0))
            .content("small snippet"),
        ];
        let budget = ProjectModelContextRenderBudget {
            max_sources: 3,
            max_source_content_chars: 64,
            max_total_content_chars: 64,
            max_source_lines: 4,
            max_rendered_chars: 8,
            max_metadata_attr_chars: 64,
        };

        let actual = render_project_model_context(
            "/workspace",
            "/manifest",
            "fresh",
            "test",
            &setup,
            &budget,
        );
        let expected = (true, true);

        assert_eq!(
            (
                actual.starts_with("<project_model_context"),
                actual.contains("omitted_reason=\"rendered_context_budget_exceeded\"")
            ),
            expected
        );
    }
}
