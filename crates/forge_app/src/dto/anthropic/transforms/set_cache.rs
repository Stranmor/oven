use forge_domain::Transformer;

use crate::dto::anthropic::Request;

/// Transformer that keeps Anthropic prompt-cache markers stable:
/// - Always caches every system message so the static system prefix remains
///   reusable
/// - Falls back to caching the first conversation message when there is no
///   system prompt so single-turn requests still establish a reusable prefix
/// - Uses exactly one rolling message-level marker on the newest message
pub struct SetCache;

impl Transformer for SetCache {
    type Value = Request;

    /// Applies the default Anthropic cache strategy:
    /// 1. Cache every system message when present, otherwise cache the first
    ///    conversation message.
    /// 2. Cache only the last message as the rolling message-level marker.
    fn transform(&mut self, mut request: Self::Value) -> Self::Value {
        let len = request.get_messages().len();
        let sys_len = request.system.as_ref().map_or(0, |msgs| msgs.len());

        if let Some(last_tool) = request.tools.last_mut() {
            last_tool.cache_control = Some(crate::dto::anthropic::CacheControl::Ephemeral);
        }

        if len == 0 && sys_len == 0 {
            return request;
        }

        let has_system_prompt = request
            .system
            .as_ref()
            .is_some_and(|messages| !messages.is_empty());

        if let Some(system_messages) = request.system.as_mut() {
            for message in system_messages.iter_mut() {
                *message = std::mem::take(message).cached(true);
            }
        }

        let first_cache_eligible = (0..len).find(|index| request.is_message_cache_eligible(*index));
        let last_cache_eligible = (0..len)
            .rev()
            .find(|index| request.is_message_cache_eligible(*index));

        for message in request.get_messages_mut().iter_mut() {
            *message = std::mem::take(message).cached(false);
        }

        if !has_system_prompt
            && let Some(index) = first_cache_eligible
            && let Some(first_message) = request.get_messages_mut().get_mut(index)
        {
            *first_message = std::mem::take(first_message).cached(true);
        }

        if let Some(index) = last_cache_eligible
            && let Some(message) = request.get_messages_mut().get_mut(index)
        {
            *message = std::mem::take(message).cached(true);
        }

        request
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use forge_domain::{Context, ContextMessage, ModelId, Role, TextMessage};
    use pretty_assertions::assert_eq;

    use super::*;

    fn create_test_context_with_system(
        system_messages: &str,
        conversation_messages: &str,
    ) -> String {
        let mut messages = Vec::new();

        // Add system messages to the regular messages array for Anthropic format
        for c in system_messages.chars() {
            match c {
                's' => messages.push(
                    ContextMessage::Text(TextMessage::new(Role::System, c.to_string())).into(),
                ),
                _ => panic!("Invalid character in system message: {}", c),
            }
        }

        // Add conversation messages
        for c in conversation_messages.chars() {
            match c {
                'u' => messages.push(
                    ContextMessage::Text(
                        TextMessage::new(Role::User, c.to_string())
                            .model(ModelId::new("claude-3-5-sonnet-20241022")),
                    )
                    .into(),
                ),
                'a' => messages.push(
                    ContextMessage::Text(TextMessage::new(Role::Assistant, c.to_string())).into(),
                ),
                'd' => messages.push(
                    ContextMessage::Text(
                        TextMessage::new(Role::User, c.to_string())
                            .model(ModelId::new("claude-3-5-sonnet-20241022"))
                            .droppable(true)
                            .cacheable(false),
                    )
                    .into(),
                ),
                _ => panic!("Invalid character in conversation message: {}", c),
            }
        }

        let context = Context {
            conversation_id: None,
            initiator: None,
            messages,
            tools: vec![],
            tool_choice: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            reasoning: None,
            model_context_length: None,
            stream: None,
            response_format: None,
            frequency_penalty: None,
            presence_penalty: None,
        };

        let request = Request::try_from(context).expect("Failed to convert context to request");
        let mut transformer = SetCache;
        let request = transformer.transform(request);

        let mut output = String::new();

        // Check which system messages are cached
        if let Some(sys) = request.system.as_ref() {
            for (i, msg) in sys.iter().enumerate() {
                if msg.is_cached() {
                    output.push('[');
                }
                output.push(system_messages.chars().nth(i).unwrap());
            }
        }

        // Check which regular messages are cached
        let cached_indices = request
            .get_messages()
            .iter()
            .enumerate()
            .filter(|(_, m)| m.is_cached())
            .map(|(i, _)| i)
            .collect::<HashSet<usize>>();

        for (i, c) in conversation_messages.chars().enumerate() {
            if cached_indices.contains(&i) {
                output.push('[');
            }
            output.push(c);
        }

        output
    }

    fn create_test_context(message: impl ToString) -> String {
        create_test_context_with_system("", &message.to_string())
    }

    #[test]
    fn test_single_message() {
        let actual = create_test_context("u");
        let expected = "[u";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_two_messages() {
        let actual = create_test_context("ua");
        let expected = "[u[a";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_three_messages_cache_first_and_last_only() {
        let actual = create_test_context("uau");
        let expected = "[ua[u";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_four_messages_cache_first_and_last_only() {
        let actual = create_test_context("uaua");
        let expected = "[uau[a";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_five_messages_cache_first_and_last_only() {
        let actual = create_test_context("uauau");
        let expected = "[uaua[u";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_longer_conversation_caches_first_and_last_only() {
        let actual = create_test_context("uauauauaua");
        let expected = "[uauauauau[a";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_with_system_message_single_conversation_message() {
        let actual = create_test_context_with_system("s", "u");
        let expected = "[s[u";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_with_system_message_multiple_conversation_messages() {
        let actual = create_test_context_with_system("ss", "uaua");
        let expected = "[s[suau[a";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_with_system_message_long_conversation() {
        let actual = create_test_context_with_system("s", "uauauauaua");
        let expected = "[suauauauau[a";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_only_system_message() {
        let actual = create_test_context_with_system("s", "");
        let expected = "[s";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_multiple_system_messages_all_cached() {
        let fixture = Context {
            conversation_id: None,
            initiator: None,
            messages: vec![
                ContextMessage::Text(TextMessage::new(Role::System, "first")).into(),
                ContextMessage::Text(TextMessage::new(Role::System, "second")).into(),
                ContextMessage::Text(
                    TextMessage::new(Role::User, "user")
                        .model(ModelId::new("claude-3-5-sonnet-20241022")),
                )
                .into(),
            ],
            tools: vec![],
            tool_choice: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            reasoning: None,
            model_context_length: None,
            stream: None,
            response_format: None,
            frequency_penalty: None,
            presence_penalty: None,
        };

        let request = Request::try_from(fixture).expect("Failed to convert context to request");
        let mut transformer = SetCache;
        let request = transformer.transform(request);

        let expected = vec![true, true];
        let actual = request
            .system
            .as_ref()
            .unwrap()
            .iter()
            .map(|message| message.is_cached())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert!(request.get_messages()[0].is_cached());
    }

    #[test]
    fn test_project_model_context_is_sent_but_cache_ineligible() {
        let fixture = Context::default()
            .add_message(ContextMessage::Text(
                TextMessage::new(Role::User, "real user")
                    .model(ModelId::new("claude-3-5-sonnet-20241022")),
            ))
            .add_message(ContextMessage::Text(
                TextMessage::new(
                    Role::User,
                    "<project_model_context>dynamic</project_model_context>",
                )
                .model(ModelId::new("claude-3-5-sonnet-20241022"))
                .droppable(true)
                .cacheable(false),
            ));
        let mut transformer = SetCache;

        let actual = transformer.transform(Request::try_from(fixture).unwrap());

        let expected = (true, false, true);
        assert_eq!(
            (
                actual.get_messages()[1]
                    .content
                    .iter()
                    .any(|content| matches!(content, crate::dto::anthropic::Content::Text { text, .. } if text.contains("project_model_context"))),
                actual.get_messages()[1].is_cached(),
                actual.get_messages()[0].is_cached(),
            ),
            expected
        );
    }

    #[test]
    fn test_runtime_context_is_sent_but_cache_ineligible() {
        let fixture = Context::default()
            .add_message(ContextMessage::Text(
                TextMessage::new(Role::User, "real user")
                    .model(ModelId::new("claude-3-5-sonnet-20241022")),
            ))
            .add_message(ContextMessage::Text(
                TextMessage::new(
                    Role::User,
                    "<runtime_context freshness=\"live\" cache=\"uncached\">dynamic</runtime_context>",
                )
                .model(ModelId::new("claude-3-5-sonnet-20241022"))
                .cacheable(false),
            ));
        let mut transformer = SetCache;

        let actual = transformer.transform(Request::try_from(fixture).unwrap());

        let expected = (true, false, true);
        assert_eq!(
            (
                actual.get_messages()[1]
                    .content
                    .iter()
                    .any(|content| matches!(content, crate::dto::anthropic::Content::Text { text, .. } if text.contains("runtime_context"))),
                actual.get_messages()[1].is_cached(),
                actual.get_messages()[0].is_cached(),
            ),
            expected
        );
    }

    #[test]
    fn test_real_user_message_remains_rolling_marker_before_runtime_context() {
        let actual = create_test_context_with_system("s", "ud");
        let expected = "[s[ud";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_changed_files_notification_is_cache_ineligible() {
        let fixture = Context::default()
            .add_message(ContextMessage::Text(
                TextMessage::new(Role::User, "real user")
                    .model(ModelId::new("claude-3-5-sonnet-20241022")),
            ))
            .add_message(ContextMessage::Text(
                TextMessage::new(Role::User, "modified externally")
                    .model(ModelId::new("claude-3-5-sonnet-20241022"))
                    .droppable(true)
                    .cacheable(false),
            ));
        let mut transformer = SetCache;

        let actual = transformer.transform(Request::try_from(fixture).unwrap());

        let expected = vec![true, false];
        assert_eq!(
            actual
                .get_messages()
                .iter()
                .map(|message| message.is_cached())
                .collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn test_real_user_message_remains_rolling_marker_before_dynamic_messages() {
        let actual = create_test_context_with_system("s", "ud");
        let expected = "[s[ud";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_last_tool_definition_is_cached() {
        let fixture = Request {
            tools: vec![
                crate::dto::anthropic::ToolDefinition {
                    name: "first".to_string(),
                    description: Some("first tool".to_string()),
                    cache_control: None,
                    input_schema: serde_json::json!({"type": "object"}),
                },
                crate::dto::anthropic::ToolDefinition {
                    name: "last".to_string(),
                    description: Some("last tool".to_string()),
                    cache_control: None,
                    input_schema: serde_json::json!({"type": "object"}),
                },
            ],
            ..Request::default()
        };
        let mut transformer = SetCache;

        let actual = transformer.transform(fixture);
        let expected = vec![false, true];

        assert_eq!(
            actual
                .tools
                .iter()
                .map(|tool| tool.cache_control.is_some())
                .collect::<Vec<_>>(),
            expected
        );
    }
}
