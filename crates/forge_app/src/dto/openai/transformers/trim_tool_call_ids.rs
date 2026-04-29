use forge_domain::Transformer;

use crate::dto::openai::Request;

const MAX_TOOL_CALL_ID_LEN: usize = 40;

fn trim_id(id: &mut Option<forge_domain::ToolCallId>) {
    if let Some(tool_call_id) = id {
        if tool_call_id.as_str().chars().count() > MAX_TOOL_CALL_ID_LEN {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(tool_call_id.as_str().as_bytes());
            let hash = hasher.finalize();
            // Take first 8 bytes and format as 16 hex chars
            let hash_bytes: [u8; 8] = hash
                .get(0..8)
                .unwrap_or(&[0; 8])
                .try_into()
                .unwrap_or([0; 8]);
            let hash_val = u64::from_be_bytes(hash_bytes);
            let prefix_len = MAX_TOOL_CALL_ID_LEN.saturating_sub(17);
            let prefix: String = tool_call_id.as_str().chars().take(prefix_len).collect();
            let trimmed_id = format!("{}_{:016x}", prefix, hash_val);
            *tool_call_id = forge_domain::ToolCallId::new(trimmed_id);
        }
    }
}

/// Trims tool call IDs to a maximum of 40 characters for OpenAI compatibility.
/// OpenAI requires tool call IDs to be max 40 characters.
pub struct TrimToolCallIds;

impl Transformer for TrimToolCallIds {
    type Value = Request;

    fn transform(&mut self, mut request: Self::Value) -> Self::Value {
        if let Some(messages) = request.messages.as_mut() {
            for message in messages.iter_mut() {
                // Trim tool_call_id in tool role messages
                trim_id(&mut message.tool_call_id);

                // Trim tool call IDs in assistant messages
                if let Some(ref mut tool_calls) = message.tool_calls {
                    for tool_call in tool_calls.iter_mut() {
                        match tool_call {
                            crate::dto::openai::response::ToolCall::Function { id, .. } => {
                                trim_id(id);
                            }
                            crate::dto::openai::response::ToolCall::CodeInterpreter { id } => {
                                trim_id(id);
                            }
                        }
                    }
                }
            }
        }
        request
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::dto::openai::response::{FunctionCall, ToolCall as ResponseToolCall};
    use crate::dto::openai::{Message, Role};

    #[test]
    fn test_trim_tool_call_id_in_tool_message() -> anyhow::Result<()> {
        // Create a tool call ID that's longer than 40 characters
        let long_id = "call_12345678901234567890123456789012345678901234567890";
        assert!(long_id.len() > 40);

        let fixture = Request::default().messages(vec![Message {
            role: Role::Tool,
            content: None,
            name: None,
            tool_call_id: Some(forge_domain::ToolCallId::new(long_id)),
            tool_calls: None,
            reasoning_details: None,
            reasoning_text: None,
            reasoning_opaque: None,
            reasoning_content: None,
            extra_content: None,
        }]);

        let actual = TrimToolCallIds.transform(fixture);

        let messages = actual.messages.context("Missing")?;
        let first_msg = messages.first().context("No first msg")?;
        let tool_call_id = first_msg.tool_call_id.as_ref().context("Missing id")?;
        assert_eq!(tool_call_id.as_str().len(), 40);
        assert!(tool_call_id.as_str().starts_with("call_123456789012345678"));
        Ok(())
    }

    #[test]
    fn test_trim_tool_call_id_in_assistant_message() -> anyhow::Result<()> {
        // Create tool calls with IDs longer than 40 characters
        let long_id = "call_12345678901234567890123456789012345678901234567890";
        assert!(long_id.len() > 40);

        let fixture = Request::default().messages(vec![Message {
            role: Role::Assistant,
            content: None,
            name: None,
            tool_call_id: None,
            tool_calls: Some(vec![ResponseToolCall::Function {
                id: Some(forge_domain::ToolCallId::new(long_id)),
                function: FunctionCall {
                    name: Some(forge_domain::ToolName::new("test_tool")),
                    arguments: "{}".to_string(),
                },
                extra_content: None,
            }]),
            reasoning_details: None,
            reasoning_text: None,
            reasoning_opaque: None,
            reasoning_content: None,
            extra_content: None,
        }]);

        let actual = TrimToolCallIds.transform(fixture);

        let messages = actual.messages.context("Missing")?;
        let tool_calls = messages
            .first()
            .context("No msg")?
            .tool_calls
            .as_ref()
            .context("No calls")?;
        if let ResponseToolCall::Function { id, .. } = tool_calls.first().context("No tool")? {
            let id_str = id.as_ref().context("No id")?.as_str();
            assert_eq!(id_str.len(), 40);
            assert!(id_str.starts_with("call_123456789012345678"));
        } else {
            anyhow::bail!("Expected Function tool call");
        }
        Ok(())
    }

    #[test]
    fn test_trim_multiple_tool_calls_in_assistant_message() -> anyhow::Result<()> {
        let long_id_1 = "call_11111111111111111111111111111111111111111111111111";
        let long_id_2 = "call_22222222222222222222222222222222222222222222222222";
        assert!(long_id_1.len() > 40);
        assert!(long_id_2.len() > 40);

        let fixture = Request::default().messages(vec![Message {
            role: Role::Assistant,
            content: None,
            name: None,
            tool_call_id: None,
            tool_calls: Some(vec![
                ResponseToolCall::Function {
                    id: Some(forge_domain::ToolCallId::new(long_id_1)),
                    function: FunctionCall {
                        name: Some(forge_domain::ToolName::new("tool_1")),
                        arguments: "{}".to_string(),
                    },
                    extra_content: None,
                },
                ResponseToolCall::Function {
                    id: Some(forge_domain::ToolCallId::new(long_id_2)),
                    function: FunctionCall {
                        name: Some(forge_domain::ToolName::new("tool_2")),
                        arguments: "{}".to_string(),
                    },
                    extra_content: None,
                },
            ]),
            reasoning_details: None,
            reasoning_text: None,
            reasoning_opaque: None,
            reasoning_content: None,
            extra_content: None,
        }]);

        let actual = TrimToolCallIds.transform(fixture);

        let messages = actual.messages.context("Missing")?;
        let tool_calls = messages
            .first()
            .context("No msg")?
            .tool_calls
            .as_ref()
            .context("No calls")?;

        if let ResponseToolCall::Function { id, .. } = tool_calls.first().context("No 1")? {
            let id_str = id.as_ref().context("id")?.as_str();
            assert_eq!(id_str.len(), 40);
            assert!(id_str.starts_with("call_111111111111111111"));
        }
        if let ResponseToolCall::Function { id, .. } = tool_calls.get(1).context("No 2")? {
            let id_str = id.as_ref().context("id")?.as_str();
            assert_eq!(id_str.len(), 40);
            assert!(id_str.starts_with("call_222222222222222222"));
        }
        Ok(())
    }

    #[test]
    fn test_trim_does_not_affect_short_ids() -> anyhow::Result<()> {
        // Create a tool call ID that's already under 40 characters
        let short_id = "call_123";
        assert!(short_id.len() < 40);

        let fixture = Request::default().messages(vec![Message {
            role: Role::Tool,
            content: None,
            name: None,
            tool_call_id: Some(forge_domain::ToolCallId::new(short_id)),
            tool_calls: None,
            reasoning_details: None,
            reasoning_text: None,
            reasoning_opaque: None,
            reasoning_content: None,
            extra_content: None,
        }]);

        let actual = TrimToolCallIds.transform(fixture);

        let messages = actual.messages.context("Missing")?;
        let tool_call_id = messages
            .first()
            .context("No msg")?
            .tool_call_id
            .as_ref()
            .context("No id")?;
        assert_eq!(tool_call_id.as_str(), short_id);
        Ok(())
    }

    #[test]
    fn test_trim_exactly_40_chars_id() -> anyhow::Result<()> {
        // Create a tool call ID that's exactly 40 characters
        let exact_id = "call_12345678901234567890123456789012345";
        assert_eq!(exact_id.len(), 40);

        let fixture = Request::default().messages(vec![Message {
            role: Role::Tool,
            content: None,
            name: None,
            tool_call_id: Some(forge_domain::ToolCallId::new(exact_id)),
            tool_calls: None,
            reasoning_details: None,
            reasoning_text: None,
            reasoning_opaque: None,
            reasoning_content: None,
            extra_content: None,
        }]);

        let actual = TrimToolCallIds.transform(fixture);

        let messages = actual.messages.context("Missing")?;
        let tool_call_id = messages
            .first()
            .context("No msg")?
            .tool_call_id
            .as_ref()
            .context("No id")?;
        assert_eq!(tool_call_id.as_str(), exact_id);
        Ok(())
    }

    #[test]
    fn test_trim_handles_multiple_messages() -> anyhow::Result<()> {
        let long_id = "call_12345678901234567890123456789012345678901234567890";
        let short_id = "call_abc";

        let fixture = Request::default().messages(vec![
            Message {
                role: Role::Tool,
                content: None,
                name: None,
                tool_call_id: Some(forge_domain::ToolCallId::new(long_id)),
                tool_calls: None,
                reasoning_details: None,
                reasoning_text: None,
                reasoning_opaque: None,
                reasoning_content: None,
                extra_content: None,
            },
            Message {
                role: Role::Tool,
                content: None,
                name: None,
                tool_call_id: Some(forge_domain::ToolCallId::new(short_id)),
                tool_calls: None,
                reasoning_details: None,
                reasoning_text: None,
                reasoning_opaque: None,
                reasoning_content: None,
                extra_content: None,
            },
        ]);

        let actual = TrimToolCallIds.transform(fixture);

        let messages = actual.messages.context("Missing")?;
        let first_msg = messages.first().context("No first msg")?;
        let second_msg = messages.get(1).context("No second msg")?;

        assert_eq!(
            first_msg
                .tool_call_id
                .as_ref()
                .context("Missing")?
                .as_str()
                .len(),
            40
        );
        assert_eq!(
            second_msg
                .tool_call_id
                .as_ref()
                .context("Missing")?
                .as_str()
                .len(),
            short_id.len()
        );
        Ok(())
    }
}
