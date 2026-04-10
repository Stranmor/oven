use forge_domain::{ConversationId, InterruptionReason, ToolCallArgumentError, ToolName};

#[derive(Debug, thiserror::Error)]
pub enum PreconditionReason {
    #[error("You must read the file with the read tool before attempting to {0}")]
    UnreadTarget(String),
}

#[derive(Debug, thiserror::Error)]
pub enum OperationPermitReason {
    #[error("Tasks should be intercepted before execution")]
    TaskIntercept,
    #[error("Invalid UTF-8 in path")]
    InvalidUtf8Path,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Invalid tool call arguments: {0}")]
    CallArgument(ToolCallArgumentError),

    #[error("Tool {0} not found")]
    NotFound(ToolName),

    #[error("Tool '{tool_name}' timed out after {timeout} minutes")]
    CallTimeout { tool_name: ToolName, timeout: u64 },

    #[error("Tool \'{name}\' is not available. Please try again with one of these tools: {:?}", supported_tools)]
    NotAllowed {
        name: ToolName,
        supported_tools: Vec<forge_domain::ToolName>,
    },

    #[error("Tool \'{tool_name}\' requires {required_modality:?} modality, but model only supports: {supported_modalities:?}")]
    UnsupportedModality {
        tool_name: ToolName,
        required_modality: forge_domain::InputModality,
        supported_modalities: Vec<forge_domain::InputModality>,
    },

    #[error("Empty tool response")]
    EmptyToolResponse,

    #[error("Agent execution was interrupted: {0:?}")]
    AgentToolInterrupted(InterruptionReason),

    #[error("Authentication still in progress")]
    AuthInProgress,

    #[error("Agent '{0}' not found")]
    AgentNotFound(forge_domain::AgentId),

    #[error("Conversation '{id}' not found")]
    ConversationNotFound { id: ConversationId },

    #[error("No active provider configured")]
    NoActiveProvider,

    #[error("No active model configured")]
    NoActiveModel,

    #[error("Precondition failed: {0}")]
    PreconditionFailed(PreconditionReason),

    #[error("Operation not permitted: {0}")]
    OperationNotPermitted(OperationPermitReason),
}
