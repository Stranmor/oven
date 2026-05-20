use forge_domain::{Context, ConversationId, MessageId, Role};
use serde::{Deserialize, Serialize};

const MAX_PREVIEW_CHARS: usize = 96;

/// Display and API DTO for a message that can be used as a conversation branch target.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConversationBranchTarget {
    /// Source conversation identifier.
    pub conversation_id: ConversationId,
    /// Stable target message identifier.
    pub message_id: MessageId,
    /// Zero-based source context ordinal.
    pub ordinal: usize,
    /// Selectable message role.
    pub role: Role,
    /// Safe single-line preview for terminal and JSONL surfaces.
    pub preview: String,
}

impl ConversationBranchTarget {
    /// Builds branch target DTOs from the context's typed selectable branch targets.
    ///
    /// # Arguments
    /// * `conversation_id` - Source conversation identifier.
    /// * `context` - Source context with persisted or materialized message identifiers.
    pub fn list_from_context(conversation_id: ConversationId, context: &Context) -> Vec<Self> {
        context
            .selectable_branch_targets()
            .into_iter()
            .map(|target| {
                let preview = context
                    .messages
                    .get(target.ordinal)
                    .and_then(|entry| entry.message.content())
                    .map(format_branch_target_preview)
                    .unwrap_or_default();
                Self {
                    conversation_id,
                    message_id: target.id,
                    ordinal: target.ordinal,
                    role: target.role,
                    preview,
                }
            })
            .collect()
    }
}

/// Formats user-visible branch target preview text safely for terminal and JSONL output.
///
/// # Arguments
/// * `input` - Raw persisted message content.
pub fn format_branch_target_preview(input: &str) -> String {
    let stripped = strip_ansi_sequences(input);
    let normalized = stripped
        .chars()
        .filter_map(|character| match character {
            '\n' | '\r' | '\t' => Some(' '),
            character if character.is_control() => None,
            character => Some(character),
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    unicode_truncate(&normalized, MAX_PREVIEW_CHARS)
}

fn unicode_truncate(input: &str, max_chars: usize) -> String {
    let mut chars = input.chars();
    let mut output = chars.by_ref().take(max_chars).collect::<String>();
    if chars.next().is_some() {
        output.push('…');
    }
    output
}

fn strip_ansi_sequences(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(character) = chars.next() {
        if character != '\u{1b}' {
            output.push(character);
            continue;
        }

        if chars.peek() == Some(&'[') {
            chars.next();
            for next in chars.by_ref() {
                if ('@'..='~').contains(&next) {
                    break;
                }
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use forge_domain::{ContextMessage, TextMessage, TextMessageKind};
    use pretty_assertions::assert_eq;

    use super::*;

    fn fixture_context() -> Context {
        Context::default()
            .add_message(ContextMessage::system("system"))
            .add_message(ContextMessage::user(" user\nmessage ", None))
            .add_message(ContextMessage::assistant(
                "\u{1b}[31massistant\u{1b}[0m\tmessage\u{7}",
                None,
                None,
                None,
            ))
            .add_message(TextMessage::learning_context(Role::User, "internal"))
            .add_message(ContextMessage::assistant(
                "tool-call owner",
                None,
                None,
                Some(vec![
                    forge_domain::ToolCallFull::new("read")
                        .call_id("call_id")
                        .arguments(forge_domain::ToolCallArguments::from_json("{}")),
                ]),
            ))
            .add_message(
                TextMessage::new(Role::User, "droppable")
                    .droppable(true)
                    .kind(TextMessageKind::RuntimeContext),
            )
    }

    #[test]
    fn test_format_branch_target_preview_strips_ansi_controls_and_whitespace() {
        let fixture = " hello\n\u{1b}[31mred\u{1b}[0m\tworld\u{7} ";
        let actual = format_branch_target_preview(fixture);
        let expected = "hello red world".to_string();

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_format_branch_target_preview_truncates_on_unicode_boundary() {
        let fixture = "аб".repeat(60);
        let actual = format_branch_target_preview(&fixture);
        let expected = format!("{}…", "аб".repeat(48));

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_branch_targets_use_selectable_filter_and_preserve_metadata() {
        let conversation_id = ConversationId::generate();
        let mut context = fixture_context().conversation_id(conversation_id);
        context.normalize_message_ids();
        let actual = ConversationBranchTarget::list_from_context(conversation_id, &context)
            .into_iter()
            .map(|target| {
                (
                    target.conversation_id,
                    target.ordinal,
                    target.role,
                    target.preview,
                )
            })
            .collect::<Vec<_>>();
        let expected = vec![
            (
                conversation_id,
                1usize,
                Role::User,
                "user message".to_string(),
            ),
            (
                conversation_id,
                2usize,
                Role::Assistant,
                "assistant message".to_string(),
            ),
        ];

        assert_eq!(actual, expected);
    }
}
