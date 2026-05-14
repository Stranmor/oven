//! Typed UI render model shared by classic and TUI presentation adapters.
//!
//! This crate is the semantic boundary between domain/application events and
//! presentation renderers. It intentionally owns no terminal backend and has no
//! `ratatui` dependency.

use std::time::Duration;

use forge_domain::{
    Category, ChatResponse, ChatResponseContent, InterruptionReason, ToolCallArguments,
    ToolCallFull, ToolResult,
};

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
    /// Submitted user message for a turn that has been accepted by the UI.
    UserMessage(String),
    /// Typed lifecycle status for an assistant turn before or during provider work.
    TurnStatus(UiTurnStatus),
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
    /// Rich tool detail payload for the side/detail pane.
    ToolDetail(UiToolDetail),
    /// Retry status emitted from typed retry events.
    Retry { cause: String, delay: UiRetryDelay },
    /// Task completion marker.
    Completion,
    /// Interrupt marker with structured reason text.
    Interrupt(String),
}

impl UiBlock {
    /// Returns the primary text payload for renderers that need a plain
    /// fallback.
    pub fn plain_text(&self) -> String {
        match self {
            UiBlock::UserMessage(text) => text.clone(),
            UiBlock::TurnStatus(status) => status.display_text(),
            UiBlock::Markdown { text, .. }
            | UiBlock::Reasoning(text)
            | UiBlock::ToolOutput(text)
            | UiBlock::Interrupt(text) => text.clone(),
            UiBlock::ToolInput(title) => title.display_text(),
            UiBlock::ToolStatus(status) => status.display_text(),
            UiBlock::ToolDetail(detail) => detail.display_text(),
            UiBlock::Retry { cause, delay } => {
                format!("retry in {}: {cause}", delay.display_text())
            }
            UiBlock::Completion => "complete".to_string(),
        }
    }
}

/// Lifecycle state for a submitted interactive turn.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UiTurnPhase {
    /// The user message has been submitted and the provider request is being prepared.
    Pending,
    /// The provider stream has started and is running.
    Running,
}

/// Presentation-safe typed turn status.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UiTurnStatus {
    /// Turn lifecycle phase.
    pub phase: UiTurnPhase,
    /// Human-readable status summary.
    pub summary: Option<String>,
}

impl UiTurnStatus {
    /// Creates a pending turn status for a submitted user message.
    pub fn pending() -> Self {
        Self {
            phase: UiTurnPhase::Pending,
            summary: Some("waiting for provider response".to_string()),
        }
    }

    /// Creates a running turn status for a provider stream.
    pub fn running() -> Self {
        Self {
            phase: UiTurnPhase::Running,
            summary: Some("provider stream running".to_string()),
        }
    }

    /// Formats the turn status as deterministic presentation text.
    pub fn display_text(&self) -> String {
        let phase = match self.phase {
            UiTurnPhase::Pending => "pending",
            UiTurnPhase::Running => "running",
        };
        match &self.summary {
            Some(summary) if !summary.is_empty() => format!("turn {phase}: {summary}"),
            _ => format!("turn {phase}"),
        }
    }
}

/// Creates the typed UI blocks shown immediately after user submission.
///
/// # Arguments
/// * `message` - Submitted user message text.
pub fn submitted_user_turn(message: impl Into<String>) -> UiModel {
    UiModel::new(vec![
        UiBlock::UserMessage(message.into()),
        UiBlock::TurnStatus(UiTurnStatus::pending()),
    ])
}

/// Creates the typed UI block shown when the provider stream starts.
pub fn running_turn() -> UiModel {
    UiModel::new(vec![UiBlock::TurnStatus(UiTurnStatus::running())])
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
    /// Formats the title and subtitle as a single deterministic fallback
    /// string.
    pub fn display_text(&self) -> String {
        match &self.subtitle {
            Some(subtitle) if !subtitle.is_empty() => format!("{} — {subtitle}", self.title),
            _ => self.title.clone(),
        }
    }
}

/// Presentation-safe retry delay that preserves duration semantics across UI
/// boundaries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UiRetryDelay(Duration);

impl UiRetryDelay {
    /// Creates a retry delay from a typed duration.
    ///
    /// # Arguments
    /// * `duration` - Domain retry delay duration to preserve for presentation.
    pub fn from_duration(duration: Duration) -> Self {
        Self(duration)
    }

    /// Returns the retry delay as a typed duration.
    pub fn as_duration(&self) -> Duration {
        self.0
    }

    /// Returns the retry delay in milliseconds for presentation formatting.
    pub fn as_millis(&self) -> u128 {
        self.0.as_millis()
    }

    /// Formats the retry delay as deterministic presentation text.
    pub fn display_text(&self) -> String {
        format!("{}ms", self.as_millis())
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

/// Rich tool detail for renderers with a detail/output pane.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UiToolDetail {
    /// Optional provider/model call ID associated with the tool event.
    pub call_id: Option<String>,
    /// Tool name from the typed domain event.
    pub name: String,
    /// Tool arguments rendered as deterministic JSON/text.
    pub arguments: Option<String>,
    /// Tool output rendered as deterministic text.
    pub output: Option<String>,
    /// True when the tool output represents a failure.
    pub is_error: bool,
}

impl UiToolDetail {
    /// Formats rich tool detail as a deterministic fallback string.
    pub fn display_text(&self) -> String {
        let mut parts = vec![self.name.clone()];
        if let Some(call_id) = &self.call_id {
            parts.push(format!("call_id={call_id}"));
        }
        if let Some(arguments) = &self.arguments {
            parts.push(format!("args={arguments}"));
        }
        if let Some(output) = &self.output {
            parts.push(format!("output={output}"));
        }
        if self.is_error {
            parts.push("error=true".to_string());
        }
        parts.join(" ")
    }
}

impl From<&ToolCallFull> for UiToolDetail {
    fn from(value: &ToolCallFull) -> Self {
        Self {
            call_id: value
                .call_id
                .as_ref()
                .map(|call_id| call_id.as_str().to_string()),
            name: value.name.as_str().to_string(),
            arguments: Some(format_tool_arguments(&value.arguments)),
            output: None,
            is_error: false,
        }
    }
}

impl From<&ToolResult> for UiToolDetail {
    fn from(value: &ToolResult) -> Self {
        Self {
            call_id: value
                .call_id
                .as_ref()
                .map(|call_id| call_id.as_str().to_string()),
            name: value.name.as_str().to_string(),
            arguments: None,
            output: value.output.as_str().map(ToString::to_string),
            is_error: value.is_error(),
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
            ChatResponse::ToolCallStart { tool_call, .. } => {
                UiBlock::ToolDetail(UiToolDetail::from(tool_call))
            }
            ChatResponse::ToolCallEnd(result) => UiBlock::ToolDetail(UiToolDetail::from(result)),
            ChatResponse::RetryAttempt { cause, duration } => UiBlock::Retry {
                cause: cause.as_str().to_string(),
                delay: UiRetryDelay::from_duration(*duration),
            },
            ChatResponse::Interrupt { reason } => UiBlock::Interrupt(interruption_text(reason)),
        }
    }
}

fn blocks_from_response(value: &ChatResponse) -> Vec<UiBlock> {
    match value {
        ChatResponse::ToolCallStart { tool_call, .. } => vec![
            UiBlock::ToolStatus(UiToolStatus {
                name: tool_call.name.as_str().to_string(),
                phase: UiToolPhase::Started,
                summary: None,
            }),
            UiBlock::ToolDetail(UiToolDetail::from(tool_call)),
        ],
        ChatResponse::ToolCallEnd(result) => vec![
            UiBlock::ToolStatus(tool_result_status(result)),
            UiBlock::ToolDetail(UiToolDetail::from(result)),
        ],
        _ => vec![UiBlock::from(value)],
    }
}

impl From<&ChatResponse> for UiModel {
    fn from(value: &ChatResponse) -> Self {
        if value.is_empty() {
            return UiModel::default();
        }
        UiModel::new(blocks_from_response(value))
    }
}

impl From<&[ChatResponse]> for UiModel {
    fn from(value: &[ChatResponse]) -> Self {
        UiModel::new(
            value
                .iter()
                .filter(|response| !response.is_empty())
                .flat_map(blocks_from_response)
                .collect(),
        )
    }
}

fn format_tool_arguments(arguments: &ToolCallArguments) -> String {
    match arguments {
        ToolCallArguments::Unparsed(value) => value.clone(),
        ToolCallArguments::Parsed(value) => value.to_string(),
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

    use forge_domain::{ChatResponseContent, ToolCallFull, ToolCallId, ToolResult};
    use pretty_assertions::assert_eq;
    use tokio::sync::Notify;

    use super::*;

    #[test]
    fn test_submitted_user_turn_model_contains_user_and_pending_blocks() {
        let actual = submitted_user_turn("Hello Forge");

        let expected = UiModel::new(vec![
            UiBlock::UserMessage("Hello Forge".to_string()),
            UiBlock::TurnStatus(UiTurnStatus::pending()),
        ]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_turn_status_formats_deterministic_lifecycle_text() {
        let actual = (
            UiTurnStatus::pending().display_text(),
            UiTurnStatus::running().display_text(),
        );

        let expected = (
            "turn pending: waiting for provider response".to_string(),
            "turn running: provider stream running".to_string(),
        );
        assert_eq!(actual, expected);
    }

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
            UiBlock::ToolDetail(UiToolDetail {
                call_id: None,
                name: "shell".to_string(),
                arguments: Some("{}".to_string()),
                output: None,
                is_error: false,
            }),
            UiBlock::ToolStatus(UiToolStatus {
                name: "shell".to_string(),
                phase: UiToolPhase::Finished,
                summary: Some("exit 0".to_string()),
            }),
            UiBlock::ToolDetail(UiToolDetail {
                call_id: None,
                name: "shell".to_string(),
                arguments: None,
                output: Some("exit 0".to_string()),
                is_error: false,
            }),
        ]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_tool_detail_maps_call_id_arguments_and_output() {
        let start = ChatResponse::ToolCallStart {
            tool_call: ToolCallFull::new("shell")
                .call_id(ToolCallId::new("call-1"))
                .arguments(serde_json::json!({"command":"true"})),
            notifier: Arc::new(Notify::new()),
        };
        let end = ChatResponse::ToolCallEnd(
            ToolResult::new("shell")
                .call_id(Some(ToolCallId::new("call-1")))
                .success("exit 0"),
        );
        let fixture = [start, end];

        let actual = UiModel::from(fixture.as_slice());

        let expected = UiModel::new(vec![
            UiBlock::ToolStatus(UiToolStatus {
                name: "shell".to_string(),
                phase: UiToolPhase::Started,
                summary: None,
            }),
            UiBlock::ToolDetail(UiToolDetail {
                call_id: Some("call-1".to_string()),
                name: "shell".to_string(),
                arguments: Some("{\"command\":\"true\"}".to_string()),
                output: None,
                is_error: false,
            }),
            UiBlock::ToolStatus(UiToolStatus {
                name: "shell".to_string(),
                phase: UiToolPhase::Finished,
                summary: Some("exit 0".to_string()),
            }),
            UiBlock::ToolDetail(UiToolDetail {
                call_id: Some("call-1".to_string()),
                name: "shell".to_string(),
                arguments: None,
                output: Some("exit 0".to_string()),
                is_error: false,
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

        let expected = UiBlock::Retry {
            cause: "network".to_string(),
            delay: UiRetryDelay::from_duration(Duration::from_millis(250)),
        };
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_retry_delay_preserves_typed_duration_and_formats_text() {
        let fixture = Duration::from_millis(1_250);

        let actual = UiRetryDelay::from_duration(fixture);

        assert_eq!(actual.as_duration(), fixture);
        assert_eq!(actual.as_millis(), 1_250);
        assert_eq!(actual.display_text(), "1250ms");
    }
}
