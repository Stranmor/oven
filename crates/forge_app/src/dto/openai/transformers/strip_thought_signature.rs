use forge_domain::Transformer;

use crate::dto::openai::Request;

/// Strips thought signatures from request messages.
///
/// This transformer removes the `extra_content` field from all messages,
/// which contains Google-specific thought signatures. This should be applied
/// to models that don't support thought signatures (all models except gemini3).
pub struct StripThoughtSignature;

impl Transformer for StripThoughtSignature {
    type Value = Request;

    fn transform(&mut self, mut request: Self::Value) -> Self::Value {
        if let Some(messages) = request.messages.as_mut() {
            for message in messages.iter_mut() {
                // Remove extra_content which contains thought_signature
                message.extra_content = None;

                // Also remove extra_content from tool_calls
                if let Some(tool_calls) = message.tool_calls.as_mut() {
                    for tool_call in tool_calls.iter_mut() {
                        if let crate::dto::openai::ToolCall::Function { extra_content, .. } =
                            tool_call
                        {
                            *extra_content = None;
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
    use forge_domain::{ModelId, Transformer};
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::dto::openai::{
        ExtraContent, FunctionCall, GoogleMetadata, Message, MessageContent, Role, ToolCall,
    };

    #[test]
    fn test_strip_thought_signature_removes_extra_content() -> anyhow::Result<()> {
        let fixture = Request::default().messages(vec![Message {
            role: Role::Assistant,
            content: Some(MessageContent::Text("Hello".to_string())),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            reasoning_details: None,
            reasoning_text: None,
            reasoning_opaque: None,
            reasoning_content: None,
            extra_content: Some(ExtraContent {
                google: Some(GoogleMetadata { thought_signature: Some("sig123".to_string()) }),
            }),
        }]);

        let mut transformer = StripThoughtSignature;
        let actual = transformer.transform(fixture);

        let msgs = actual.messages.context("Missing msgs")?;
        let first_msg = msgs.first().context("No first msg")?;
        assert!(first_msg.extra_content.is_none());
        Ok(())
    }

    #[test]
    fn test_strip_thought_signature_removes_from_tool_calls() -> anyhow::Result<()> {
        let fixture = Request::default().messages(vec![Message {
            role: Role::Assistant,
            content: Some(MessageContent::Text("Using tool".to_string())),
            name: None,
            tool_call_id: None,
            tool_calls: Some(vec![ToolCall::Function {
                id: None,
                function: FunctionCall { name: None, arguments: "{}".to_string() },
                extra_content: Some(ExtraContent {
                    google: Some(GoogleMetadata { thought_signature: Some("sig456".to_string()) }),
                }),
            }]),
            reasoning_details: None,
            reasoning_text: None,
            reasoning_opaque: None,
            reasoning_content: None,
            extra_content: None,
        }]);

        let mut transformer = StripThoughtSignature;
        let actual = transformer.transform(fixture);

        let messages = actual.messages.context("No messages")?;
        let tool_calls = messages
            .first()
            .context("No first message")?
            .tool_calls
            .as_ref()
            .context("No tool calls")?;
        if let ToolCall::Function { extra_content, .. } =
            tool_calls.first().context("No first tool call")?
        {
            assert!(extra_content.is_none());
        } else {
            anyhow::bail!("Expected Function tool call");
        }
        Ok(())
    }

    #[test]
    fn test_strip_thought_signature_no_messages() {
        let fixture = Request::default();

        let mut transformer = StripThoughtSignature;
        let actual = transformer.transform(fixture);

        assert!(actual.messages.is_none());
    }

    #[test]
    fn test_strip_thought_signature_preserves_other_fields() -> anyhow::Result<()> {
        let fixture = Request::default()
            .model(ModelId::new("gpt-4"))
            .messages(vec![Message {
                role: Role::Assistant,
                content: Some(MessageContent::Text("Hello".to_string())),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                reasoning_details: None,
                reasoning_text: Some("reasoning".to_string()),
                reasoning_opaque: None,
                reasoning_content: None,
                extra_content: Some(ExtraContent {
                    google: Some(GoogleMetadata { thought_signature: Some("sig123".to_string()) }),
                }),
            }]);

        let mut transformer = StripThoughtSignature;
        let actual = transformer.transform(fixture);

        let messages = actual.messages.context("Missing")?;
        let first_msg = messages.first().context("No first msg")?;
        assert!(first_msg.extra_content.is_none());
        assert_eq!(first_msg.reasoning_text, Some("reasoning".to_string()));
        assert_eq!(actual.model, Some(ModelId::new("gpt-4")));
        Ok(())
    }
}
