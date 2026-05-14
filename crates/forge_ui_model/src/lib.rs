//! Typed UI render model shared by classic and TUI presentation adapters.
//!
//! This crate is the semantic boundary between domain/application events and
//! presentation renderers. It intentionally owns no terminal backend and has no
//! `ratatui` dependency.

use forge_domain::{Category, ChatResponse, ChatResponseContent, InterruptionReason, ToolResult};

/// A complete append-only render document for one UI surface refresh.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct UiModel {
    /// Ordered render blocks produced from typed chat responses.
    pub blocks: Vec<UiBlock>,
}

impl UiModel {
    /// Creates a UI model from precomputed blocks.
    ///
    /// # Arguments
    /// * `blocks` - Ordered blocks that should be rendered by presentation
    ///   adapters.
    pub fn new(blocks: Vec<UiBlock>) -> Self {
        Self { blocks }
    }

    /// Appends a block to the model.
    ///
    /// # Arguments
    /// * `block` - The next typed render block.
    pub fn push(&mut self, block: UiBlock) {
        self.blocks.push(block);
    }

    /// Returns true when the model has no render blocks.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

/// A typed render block that preserves the semantics of a chat response.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UiBlock {
    /// Assistant/user markdown content, including streaming partial state.
    Markdown { text: String, partial: bool },
    /// Reasoning text separated from user-visible markdown.
    Reasoning(String),
    /// Tool input/status title emitted before tool execution.
    ToolInput(UiTitle),
    /// Tool output payload emitted after execution.
    ToolOutput(String),
    /// Tool lifecycle status emitted from typed tool events.
    ToolStatus(UiToolStatus),
    /// Retry status emitted from typed retry events.
    Retry { cause: String, duration_ms: u128 },
    /// Task completion marker.
    Completion,
    /// Interrupt marker with structured reason text.
    Interrupt(String),
}

impl UiBlock {
    /// Returns the primary text payload for renderers that need a plain fallback.
    pub fn plain_text(&self) -> String {
        match self {
            UiBlock::Markdown { text, .. }
            | UiBlock::Reasoning(text)
            | UiBlock::ToolOutput(text)
            | UiBlock::Interrupt(text) => text.clone(),
            UiBlock::ToolInput(title) => title.display_text(),
            UiBlock::ToolStatus(status) => status.display_text(),
            UiBlock::Retry { cause, duration_ms } => {
                format!("retry in {duration_ms}ms: {cause}")
            }
            UiBlock::Completion => "complete".to_string(),
        }
    }
}

/// Presentation-safe title metadata for status lines.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UiTitle {
    /// Main title text.
    pub title: String,
    /// Optional subtitle text.
    pub subtitle: Option<String>,
    /// Semantic category for styling.
    pub category: UiCategory,
}

impl UiTitle {
    /// Formats the title and subtitle as a single deterministic fallback string.
    pub fn display_text(&self) -> String {
        match &self.subtitle {
            Some(subtitle) if !subtitle.is_empty() => format!("{} — {subtitle}", self.title),
            _ => self.title.clone(),
        }
    }
}

/// Presentation-safe category copied from domain title categories.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UiCategory {
    /// User-visible action.
    Action,
    /// Informational status.
    Info,
    /// Debug-only status.
    Debug,
    /// Error status.
    Error,
    /// Completion status.
    Completion,
    /// Warning status.
    Warning,
}

impl From<&Category> for UiCategory {
    fn from(value: &Category) -> Self {
        match value {
            Category::Action => UiCategory::Action,
            Category::Info => UiCategory::Info,
            Category::Debug => UiCategory::Debug,
            Category::Error => UiCategory::Error,
            Category::Completion => UiCategory::Completion,
            Category::Warning => UiCategory::Warning,
        }
    }
}

/// Tool lifecycle phase represented without parsing stdout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UiToolPhase {
    /// Tool execution is about to start.
    Started,
    /// Tool execution completed successfully.
    Finished,
    /// Tool execution completed with an error.
    Failed,
}

/// Presentation-safe tool status block.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UiToolStatus {
    /// Tool name from the typed domain event.
    pub name: String,
    /// Tool lifecycle phase.
    pub phase: UiToolPhase,
    /// Optional typed output summary.
    pub summary: Option<String>,
}

impl UiToolStatus {
    /// Formats the status as a deterministic fallback string.
    pub fn display_text(&self) -> String {
        let phase = match self.phase {
            UiToolPhase::Started => "started",
            UiToolPhase::Finished => "finished",
            UiToolPhase::Failed => "failed",
        };
        match &self.summary {
            Some(summary) if !summary.is_empty() => format!("{} {phase}: {summary}", self.name),
            _ => format!("{} {phase}", self.name),
        }
    }
}

impl From<&ChatResponse> for UiBlock {
    fn from(value: &ChatResponse) -> Self {
        match value {
            ChatResponse::TaskMessage { content } => match content {
                ChatResponseContent::ToolInput(title) => UiBlock::ToolInput(UiTitle {
                    title: title.title.clone(),
                    subtitle: title.sub_title.clone(),
                    category: UiCategory::from(&title.category),
                }),
                ChatResponseContent::ToolOutput(text) => UiBlock::ToolOutput(text.clone()),
                ChatResponseContent::Markdown { text, partial } => {
                    UiBlock::Markdown { text: text.clone(), partial: *partial }
                }
            },
            ChatResponse::TaskReasoning { content } => UiBlock::Reasoning(content.clone()),
            ChatResponse::TaskComplete => UiBlock::Completion,
            ChatResponse::ToolCallStart { tool_call, .. } => UiBlock::ToolStatus(UiToolStatus {
                name: tool_call.name.as_str().to_string(),
                phase: UiToolPhase::Started,
                summary: None,
            }),
            ChatResponse::ToolCallEnd(result) => UiBlock::ToolStatus(tool_result_status(result)),
            ChatResponse::RetryAttempt { cause, duration } => UiBlock::Retry {
                cause: cause.as_str().to_string(),
                duration_ms: duration.as_millis(),
            },
            ChatResponse::Interrupt { reason } => UiBlock::Interrupt(interruption_text(reason)),
        }
    }
}

impl From<&[ChatResponse]> for UiModel {
    fn from(value: &[ChatResponse]) -> Self {
        UiModel::new(
            value
                .iter()
                .filter(|response| !response.is_empty())
                .map(UiBlock::from)
                .collect(),
        )
    }
}

fn tool_result_status(result: &ToolResult) -> UiToolStatus {
    UiToolStatus {
        name: result.name.as_str().to_string(),
        phase: if result.is_error() {
            UiToolPhase::Failed
        } else {
            UiToolPhase::Finished
        },
        summary: result.output.as_str().map(ToString::to_string),
    }
}

fn interruption_text(reason: &InterruptionReason) -> String {
    reason.to_string()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use forge_domain::{ChatResponseContent, ToolCallFull, ToolResult};
    use pretty_assertions::assert_eq;
    use tokio::sync::Notify;

    use super::*;

    #[test]
    fn test_markdown_chat_response_maps_to_ui_model_block() {
        let fixture = ChatResponse::TaskMessage {
            content: ChatResponseContent::Markdown { text: "**Hello**".to_string(), partial: true },
        };

        let actual = UiBlock::from(&fixture);

        let expected = UiBlock::Markdown { text: "**Hello**".to_string(), partial: true };
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_tool_status_events_map_without_stdout_parsing() {
        let start = ChatResponse::ToolCallStart {
            tool_call: ToolCallFull::new("shell"),
            notifier: Arc::new(Notify::new()),
        };
        let end = ChatResponse::ToolCallEnd(ToolResult::new("shell").success("exit 0"));
        let fixture = [start, end];

        let actual = UiModel::from(fixture.as_slice());

        let expected = UiModel::new(vec![
            UiBlock::ToolStatus(UiToolStatus {
                name: "shell".to_string(),
                phase: UiToolPhase::Started,
                summary: None,
            }),
            UiBlock::ToolStatus(UiToolStatus {
                name: "shell".to_string(),
                phase: UiToolPhase::Finished,
                summary: Some("exit 0".to_string()),
            }),
        ]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_retry_attempt_maps_to_typed_retry_block() {
        let error = anyhow::anyhow!("network");
        let fixture = ChatResponse::RetryAttempt {
            cause: (&error).into(),
            duration: Duration::from_millis(250),
        };

        let actual = UiBlock::from(&fixture);

        let expected = UiBlock::Retry { cause: "network".to_string(), duration_ms: 250 };
        assert_eq!(actual, expected);
    }
}
