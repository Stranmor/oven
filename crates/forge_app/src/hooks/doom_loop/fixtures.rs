use forge_domain::{
    Agent, Context, ContextMessage, Conversation, ConversationId, EventData, MessageEntry, ModelId,
    RequestPayload, Role, TextMessage, ToolCallArguments, ToolCallFull,
};

use super::DoomLoopDetector;

pub(super) fn assistant_message_with_tool_calls(tool_calls: Vec<ToolCallFull>) -> TextMessage {
    TextMessage {
        role: Role::Assistant,
        content: String::new(),
        raw_content: None,
        tool_calls: Some(tool_calls),
        thought_signature: None,
        model: None,
        reasoning_details: None,
        droppable: false,
        phase: None,
    }
}

pub(super) fn assistant_message(tool_call: &ToolCallFull) -> TextMessage {
    assistant_message_with_tool_calls(vec![tool_call.clone()])
}

pub(super) fn text_message(role: Role, content: &str) -> TextMessage {
    TextMessage {
        role,
        content: content.to_string(),
        raw_content: None,
        tool_calls: None,
        thought_signature: None,
        model: None,
        reasoning_details: None,
        droppable: false,
        phase: None,
    }
}

pub(super) fn conversation_with_context_messages(messages: Vec<ContextMessage>) -> Conversation {
    let context_messages: Vec<MessageEntry> =
        messages.into_iter().map(MessageEntry::from).collect();
    let context = Context::default().messages(context_messages);

    Conversation {
        id: ConversationId::generate(),
        parent_id: None,
        title: None,
        context: Some(context),
        initiator: forge_domain::Initiator::User,
        metrics: Default::default(),
        metadata: forge_domain::MetaData::new(chrono::Utc::now()),
    }
}

pub(super) fn conversation_with_messages(messages: Vec<TextMessage>) -> Conversation {
    conversation_with_context_messages(messages.into_iter().map(ContextMessage::Text).collect())
}

pub(super) fn event() -> EventData<RequestPayload> {
    EventData::new(
        Agent::new(
            "test-agent",
            "test-provider".to_string().into(),
            ModelId::new("test-model"),
        ),
        ModelId::new("test-model"),
        RequestPayload::new(3),
    )
}

pub(super) fn tool_call(name: &str, arguments: &str) -> ToolCallFull {
    ToolCallFull::new(name).arguments(ToolCallArguments::from_json(arguments))
}

pub(super) fn repeated_read_conversation() -> Conversation {
    let fixture = tool_call("read", r#"{"path": "file.txt"}"#);
    conversation_with_messages(vec![
        assistant_message(&fixture),
        assistant_message(&fixture),
        assistant_message(&fixture),
    ])
}

pub(super) fn conversation_from_tool_calls(tool_calls: &[ToolCallFull]) -> Conversation {
    conversation_with_messages(tool_calls.iter().map(assistant_message).collect())
}

pub(super) fn detector_with_threshold(threshold: usize) -> DoomLoopDetector {
    DoomLoopDetector::new().threshold(threshold)
}
