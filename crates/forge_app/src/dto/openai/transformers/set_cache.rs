use forge_domain::{MessageCacheClass, Transformer};

use crate::dto::openai::Request;

/// Transformer that implements a simple two-breakpoint cache strategy:
/// - Always caches the first message in the conversation
/// - Always caches the last message in the conversation
/// - Removes cache control from the second-to-last message
pub struct SetCache;

impl Transformer for SetCache {
    type Value = Request;

    /// Implements a simple two-breakpoint cache strategy:
    /// 1. Cache the first message (index 0)
    /// 2. Cache the last message (index messages.len() - 1)
    /// 3. Remove cache control from second-to-last message (index
    ///    messages.len() - 2)
    fn transform(&mut self, mut request: Self::Value) -> Self::Value {
        let len = request.message_count();
        let stable_cache_indices = (0..len)
            .filter(|index| {
                request.message_cache_class(*index) == MessageCacheClass::StableProjectModel
            })
            .collect::<Vec<_>>();
        let first_cache_eligible = (0..len)
            .filter(|index| request.message_cache_class(*index) == MessageCacheClass::Conversation)
            .find(|index| request.is_message_cache_eligible(*index));
        let last_cache_eligible = (0..len)
            .rev()
            .filter(|index| request.message_cache_class(*index) == MessageCacheClass::Conversation)
            .find(|index| request.is_message_cache_eligible(*index));

        if let Some(messages) = request.messages.as_mut() {
            if len == 0 {
                return request;
            }

            for message in messages.iter_mut() {
                if let Some(ref content) = message.content {
                    message.content = Some(content.clone().cached(false));
                }
            }

            for index in stable_cache_indices {
                if let Some(message) = messages.get_mut(index)
                    && let Some(ref content) = message.content
                {
                    message.content = Some(content.clone().cached(true));
                }
            }

            if let Some(index) = first_cache_eligible
                && let Some(message) = messages.get_mut(index)
                && let Some(ref content) = message.content
            {
                message.content = Some(content.clone().cached(true));
            }

            if let Some(index) = last_cache_eligible
                && let Some(message) = messages.get_mut(index)
                && let Some(ref content) = message.content
            {
                message.content = Some(content.clone().cached(true));
            }
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

    fn create_test_context(message: impl ToString) -> String {
        let context = Context {
            conversation_id: None,
            initiator: None,
            messages: message
                .to_string()
                .chars()
                .map(|c| match c {
                    's' => ContextMessage::Text(TextMessage::new(Role::System, c.to_string())),
                    'u' => ContextMessage::Text(
                        TextMessage::new(Role::User, c.to_string()).model(ModelId::new("gpt-4")),
                    ),
                    'a' => ContextMessage::Text(TextMessage::new(Role::Assistant, c.to_string())),
                    'd' => ContextMessage::Text(
                        TextMessage::new(Role::User, c.to_string())
                            .model(ModelId::new("gpt-4"))
                            .droppable(true)
                            .cacheable(false),
                    ),
                    _ => {
                        panic!("Invalid character in test message");
                    }
                })
                .map(|msg| msg.into())
                .collect(),
            tools: vec![],
            tool_choice: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            reasoning: None,
            context_window_recovery: None,
            model_context_length: None,
            stream: None,
            response_format: None,
            frequency_penalty: None,
            presence_penalty: None,
        };

        let request = Request::from(context);
        let mut transformer = SetCache;
        let request = transformer.transform(request);
        let mut output = String::new();
        let sequences = request
            .messages
            .into_iter()
            .flatten()
            .flat_map(|m| m.content)
            .enumerate()
            .filter(|(_, m)| m.is_cached())
            .map(|(i, _)| i)
            .collect::<HashSet<usize>>();

        for (i, c) in message.to_string().chars().enumerate() {
            if sequences.contains(&i) {
                output.push('[');
            }
            output.push_str(c.to_string().as_str())
        }

        output
    }

    #[test]
    fn test_single_message() {
        let actual = create_test_context("s");
        let expected = "[s";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_two_messages() {
        let actual = create_test_context("su");
        let expected = "[s[u";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_multiple_system_messages() {
        let actual = create_test_context("sssuuu");
        let expected = "[sssuu[u";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_three_messages_first_and_last_cached() {
        let actual = create_test_context("sua");
        let expected = "[su[a";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_four_messages_first_and_last_cached() {
        let actual = create_test_context("suau");
        let expected = "[sua[u";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_five_messages_first_and_last_cached() {
        let actual = create_test_context("suaua");
        let expected = "[suau[a";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_longer_conversation() {
        let actual = create_test_context("suuauuaaau");
        let expected = "[suuauuaaa[u";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_project_model_context_is_sent_but_cache_ineligible() {
        let fixture = Context::default()
            .add_message(ContextMessage::Text(
                TextMessage::new(Role::User, "real user").model(ModelId::new("gpt-4")),
            ))
            .add_message(ContextMessage::Text(
                TextMessage::stable_project_model_context(
                    Role::User,
                    "<project_model_context cache=\"stable\">stable</project_model_context>",
                )
                .model(ModelId::new("gpt-4")),
            ));
        let mut transformer = SetCache;

        let actual = transformer.transform(Request::from(fixture));
        let messages = actual.messages.unwrap();

        let expected = (true, true, true);
        assert_eq!(
            (
                serde_json::to_string(messages[1].content.as_ref().unwrap())
                    .unwrap()
                    .contains("project_model_context"),
                messages[1].content.as_ref().unwrap().is_cached(),
                messages[0].content.as_ref().unwrap().is_cached(),
            ),
            expected
        );
    }

    #[test]
    fn test_project_model_volatile_sidecar_is_not_cached_between_stable_and_user_messages() {
        let fixture = Context::default()
            .add_message(ContextMessage::Text(
                TextMessage::new(Role::User, "real user").model(ModelId::new("gpt-4")),
            ))
            .add_message(ContextMessage::Text(
                TextMessage::stable_project_model_context(
                    Role::User,
                    "<project_model_context cache=\"stable\">stable</project_model_context>",
                )
                .model(ModelId::new("gpt-4")),
            ))
            .add_message(ContextMessage::Text(
                TextMessage::project_model_volatile_sidecar(
                    Role::User,
                    "<project_model_volatile_sidecar cache=\"uncached\">time</project_model_volatile_sidecar>",
                )
                .model(ModelId::new("gpt-4")),
            ));
        let mut transformer = SetCache;

        let actual = transformer.transform(Request::from(fixture));
        let messages = actual.messages.unwrap();

        let expected = vec![true, true, false];
        assert_eq!(
            messages
                .iter()
                .map(|message| message.content.as_ref().unwrap().is_cached())
                .collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn test_cache_class_not_inferred_from_message_position_or_role_alone() {
        let fixture = Context::default()
            .add_message(ContextMessage::Text(
                TextMessage::new(Role::User, "uncached first")
                    .model(ModelId::new("gpt-4"))
                    .cache_ineligible(),
            ))
            .add_message(ContextMessage::Text(
                TextMessage::new(Role::User, "uncached last")
                    .model(ModelId::new("gpt-4"))
                    .cache_ineligible(),
            ));
        let mut transformer = SetCache;

        let actual = transformer.transform(Request::from(fixture));
        let messages = actual.messages.unwrap();

        let expected = vec![false, false];
        assert_eq!(
            messages
                .iter()
                .map(|message| message.content.as_ref().unwrap().is_cached())
                .collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn test_runtime_context_is_sent_but_cache_ineligible() {
        let fixture = Context::default()
            .add_message(ContextMessage::Text(
                TextMessage::new(Role::User, "real user").model(ModelId::new("gpt-4")),
            ))
            .add_message(ContextMessage::Text(
                TextMessage::new(
                    Role::User,
                    "<runtime_context freshness=\"live\" cache=\"uncached\">dynamic</runtime_context>",
                )
                .model(ModelId::new("gpt-4"))
                .cacheable(false),
            ));
        let mut transformer = SetCache;

        let actual = transformer.transform(Request::from(fixture));
        let messages = actual.messages.unwrap();

        let expected = (true, false, true);
        assert_eq!(
            (
                matches!(messages[1].content.as_ref().unwrap(), crate::dto::openai::MessageContent::Text(text) if text.contains("runtime_context")),
                messages[1].content.as_ref().unwrap().is_cached(),
                messages[0].content.as_ref().unwrap().is_cached(),
            ),
            expected
        );
    }

    #[test]
    fn test_real_user_message_remains_rolling_marker_before_runtime_context() {
        let actual = create_test_context("sud");
        let expected = "[s[ud";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_changed_files_notification_is_cache_ineligible() {
        let actual = create_test_context("ud");
        let expected = "[ud";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_real_user_message_remains_rolling_marker_before_dynamic_messages() {
        let actual = create_test_context("sud");
        let expected = "[s[ud";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_cache_removal_from_second_to_last() {
        // Test that second-to-last message doesn't have cache when there are 3+
        // messages
        let actual = create_test_context("suuauuaaauauau");
        let expected = "[suuauuaaauaua[u";
        assert_eq!(actual, expected);
    }
}
