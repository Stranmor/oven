use serde_json::json;

use crate::{
    Context, ContextMessage, MessageEntry, ModelId, ToolCallFull, ToolCallId, ToolName, ToolResult,
};

/// Converts a condensed string pattern into a Context with messages.
///
/// This utility type is primarily used in tests to quickly create Context
/// objects with specific message sequences without verbose setup code.
///
/// # Pattern Format
///
/// Each character in the pattern represents a message with a specific role:
/// - `'u'` = User message
/// - `'a'` = Assistant message
/// - `'s'` = System message
/// - `'t'` = Assistant message with tool call
/// - `'r'` = Tool result message
///
/// # Examples
///
/// ```rust,ignore
/// // Creates: User -> Assistant -> User
/// let context = MessagePattern::new("uau").build();
///
/// // Creates: System -> System -> User -> System -> User -> System -> User -> System -> Assistant -> Assistant -> System -> Assistant
/// let context = MessagePattern::new("ssusususaasa").build();
///
/// // Creates: User -> Assistant with tool call -> Tool result -> User
/// let context = MessagePattern::new("utru").build();
/// ```
#[derive(Debug, Clone)]
pub struct MessagePattern {
    pattern: String,
}

impl MessagePattern {
    /// Creates a new MessagePattern from the given pattern string.
    ///
    /// # Arguments
    ///
    /// * `pattern` - A string where each character represents a message role:
    ///   - `'u'` for User
    ///   - `'a'` for Assistant
    ///   - `'s'` for System
    ///   - `'t'` for Assistant with tool call
    ///   - `'r'` for Tool result
    pub fn new(pattern: impl Into<String>) -> Self {
        Self { pattern: pattern.into() }
    }

    /// Builds a Context from the pattern.
    ///
    /// Each message will have content in the format "Message {index}" where
    /// index starts from 1. Tool calls and tool results use predefined test
    /// data.
    ///
    /// # Panics
    ///
    /// Panics if the pattern contains any character other than 'u', 'a', 's',
    /// 't', or 'r'.
    pub fn build(self) -> Context {
        let model_id = ModelId::new("gpt-4");

        let tool_call = ToolCallFull {
            name: ToolName::new("read"),
            call_id: Some(ToolCallId::new("call_123")),
            arguments: json!({"path": "/test/path"}).into(),
            thought_signature: None,
        };

        let tool_result = ToolResult::new(ToolName::new("read"))
            .call_id(ToolCallId::new("call_123"))
            .success(json!({"content": "File content"}).to_string());

        let messages: Vec<MessageEntry> = self
            .pattern
            .chars()
            .enumerate()
            .map(|(i, c)| {
                let content = format!("Message {}", i + 1);
                match c {
                    'u' => ContextMessage::user(&content, Some(model_id.clone())),
                    'a' => ContextMessage::assistant(&content, None, None, None),
                    's' => ContextMessage::system(&content),
                    't' => ContextMessage::assistant(
                        &content,
                        None,
                        None,
                        Some(vec![tool_call.clone()]),
                    ),
                    'r' => ContextMessage::tool_result(tool_result.clone()),
                    _ => {
                        panic!("Invalid character '{c}' in pattern. Use 'u', 'a', 's', 't', or 'r'")
                    }
                }
            })
            .map(MessageEntry::from)
            .collect();
        Context::default().messages(messages)
    }
}

impl From<&str> for MessagePattern {
    fn from(pattern: &str) -> Self {
        Self::new(pattern)
    }
}

impl From<String> for MessagePattern {
    fn from(pattern: String) -> Self {
        Self::new(pattern)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::{ContextMessage, ModelId, Role, TextMessage};

    #[test]
    fn test_message_pattern_single_user() {
        let fixture = MessagePattern::new("u");
        let actual = fixture.build();
        let expected = Context::default().messages(vec![
            ContextMessage::Text(
                TextMessage::new(Role::User, "Message 1").model(ModelId::new("gpt-4")),
            )
            .into(),
        ]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_message_pattern_user_assistant_user() {
        let fixture = MessagePattern::new("uau");
        let actual = fixture.build();
        let expected = Context::default().messages(vec![
            ContextMessage::Text(
                TextMessage::new(Role::User, "Message 1").model(ModelId::new("gpt-4")),
            )
            .into(),
            ContextMessage::Text(TextMessage::new(Role::Assistant, "Message 2")).into(),
            ContextMessage::Text(
                TextMessage::new(Role::User, "Message 3").model(ModelId::new("gpt-4")),
            )
            .into(),
        ]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_message_pattern_complex() {
        let fixture = MessagePattern::new("ssusususaasa");
        let actual = fixture.build();

        let actual_roles: Vec<_> = actual
            .messages
            .iter()
            .filter_map(|message| {
                if message.has_role(Role::System) {
                    Some(Role::System)
                } else if message.has_role(Role::User) {
                    Some(Role::User)
                } else if message.has_role(Role::Assistant) {
                    Some(Role::Assistant)
                } else {
                    None
                }
            })
            .collect();
        let expected_roles = vec![
            Role::System,
            Role::System,
            Role::User,
            Role::System,
            Role::User,
            Role::System,
            Role::User,
            Role::System,
            Role::Assistant,
            Role::Assistant,
            Role::System,
            Role::Assistant,
        ];
        assert_eq!(actual_roles, expected_roles);
    }

    #[test]
    fn test_message_pattern_empty() {
        let fixture = MessagePattern::new("");
        let actual = fixture.build();
        let expected = Context::default();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_message_pattern_all_system() {
        let fixture = MessagePattern::new("sss");
        let actual = fixture.build();

        assert_eq!(actual.messages.len(), 3);
        assert!(actual.messages.iter().all(|m| m.has_role(Role::System)));
    }

    #[test]
    #[should_panic(expected = "Invalid character 'x' in pattern. Use 'u', 'a', 's', 't', or 'r'")]
    fn test_message_pattern_invalid_character() {
        let fixture = MessagePattern::new("uax");
        fixture.build();
    }

    #[test]
    fn test_message_pattern_from_str() {
        let fixture = MessagePattern::from("ua");
        let actual = fixture.build();
        assert_eq!(actual.messages.len(), 2);
    }

    #[test]
    fn test_message_pattern_from_string() {
        let fixture = MessagePattern::from("ua".to_string());
        let actual = fixture.build();
        assert_eq!(actual.messages.len(), 2);
    }

    #[test]
    fn test_message_pattern_content_numbering() {
        let fixture = MessagePattern::new("uau");
        let actual = fixture.build();

        let actual: Vec<_> = actual
            .messages
            .iter()
            .map(|message| {
                message
                    .content()
                    .expect("expected message content")
                    .to_string()
            })
            .collect();
        let expected = vec!["Message 1", "Message 2", "Message 3"];
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_message_pattern_with_tool_call() {
        let fixture = MessagePattern::new("utr");
        let actual = fixture.build();

        let actual_roles = actual
            .messages
            .iter()
            .map(|message| {
                if message.has_role(Role::User) {
                    "user"
                } else if message.has_role(Role::Assistant) {
                    "assistant"
                } else if message.has_tool_result() {
                    "tool_result"
                } else {
                    "other"
                }
            })
            .collect::<Vec<_>>();
        let expected_roles = vec!["user", "assistant", "tool_result"];
        assert_eq!(actual_roles, expected_roles);
        assert!(
            actual
                .messages
                .iter()
                .any(|message| message.has_tool_call())
        );
    }

    #[test]
    fn test_message_pattern_with_multiple_tool_calls() {
        let fixture = MessagePattern::new("utrtr");
        let actual = fixture.build();

        let actual = actual
            .messages
            .iter()
            .map(|message| (message.has_tool_call(), message.has_tool_result()))
            .collect::<Vec<_>>();
        let expected = vec![
            (false, false),
            (true, false),
            (false, true),
            (true, false),
            (false, true),
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_message_pattern_complex_with_tools() {
        let fixture = MessagePattern::new("sutruaua");
        let actual = fixture.build();

        let actual_roles = actual
            .messages
            .iter()
            .map(|message| {
                if message.has_role(Role::System) {
                    "system"
                } else if message.has_role(Role::User) {
                    "user"
                } else if message.has_role(Role::Assistant) {
                    "assistant"
                } else if message.has_tool_result() {
                    "tool_result"
                } else {
                    "other"
                }
            })
            .collect::<Vec<_>>();
        let expected_roles = vec![
            "system",
            "user",
            "assistant",
            "tool_result",
            "user",
            "assistant",
            "user",
            "assistant",
        ];
        assert_eq!(actual_roles, expected_roles);
        assert!(
            actual
                .messages
                .iter()
                .any(|message| message.has_tool_call())
        );
    }
}
