use derive_setters::Setters;
use serde::de::{Error as DeError, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

use crate::{ConversationId, Event};

#[derive(Debug, Serialize, Deserialize, Clone, Setters)]
#[setters(into, strip_option)]
pub struct ChatRequest {
    pub event: Event,
    pub conversation_id: ConversationId,
}

impl ChatRequest {
    pub fn new(content: Event, conversation_id: ConversationId) -> Self {
        Self { event: content, conversation_id }
    }
}

/// Typed control-plane message queued for the primary conversation only.
#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
pub struct SteerMessage {
    content: String,
}

impl SteerMessage {
    /// Creates a steer message from non-empty control-plane content.
    ///
    /// # Arguments
    /// * `content` - The message content to deliver after the current tool
    ///   batch.
    ///
    /// # Errors
    /// Returns an error when the steer message is empty or whitespace-only.
    pub fn new(content: impl Into<String>) -> anyhow::Result<Self> {
        let content = content.into();
        if content.trim().is_empty() {
            return Err(anyhow::anyhow!("Steer message cannot be empty"));
        }
        Ok(Self { content })
    }

    /// Returns the steer message body.
    pub fn content(&self) -> &str {
        &self.content
    }
}

impl<'de> Deserialize<'de> for SteerMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SteerMessageVisitor;

        impl<'de> Visitor<'de> for SteerMessageVisitor {
            type Value = SteerMessage;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a non-empty steer message object")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut content = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "content" => content = Some(map.next_value::<String>()?),
                        _ => {
                            let _ = map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                let content = content.ok_or_else(|| A::Error::missing_field("content"))?;
                SteerMessage::new(content).map_err(A::Error::custom)
            }
        }

        deserializer.deserialize_map(SteerMessageVisitor)
    }
}

/// Typed request to enqueue a steer message for a conversation.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SteerRequest {
    /// Conversation that must still be a primary user conversation.
    pub conversation_id: ConversationId,
    /// Typed steer message to enqueue.
    pub message: SteerMessage,
}

impl SteerRequest {
    /// Creates a typed steer request for a conversation.
    ///
    /// # Arguments
    /// * `conversation_id` - The primary conversation that should receive
    ///   steer.
    /// * `message` - The typed steer message to queue.
    pub fn new(conversation_id: ConversationId, message: SteerMessage) -> Self {
        Self { conversation_id, message }
    }
}

/// In-memory FIFO queue for typed steer messages.
#[derive(Debug, Default, Clone)]
pub struct SteerQueue {
    messages: Vec<SteerMessage>,
}

impl SteerQueue {
    /// Enqueues a typed steer message for delayed delivery.
    ///
    /// # Arguments
    /// * `message` - The validated steer message to enqueue.
    pub fn push(&mut self, message: SteerMessage) {
        self.messages.push(message);
    }

    /// Drains all queued steer messages in insertion order.
    pub fn drain(&mut self) -> impl Iterator<Item = SteerMessage> + '_ {
        self.messages.drain(..)
    }

    /// Returns true when no steer messages are queued.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_steer_message_rejects_whitespace_through_constructor() {
        let actual = SteerMessage::new(" \t\n ").is_err();
        let expected = true;

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_steer_message_rejects_whitespace_through_serde() {
        let actual = serde_json::from_str::<SteerMessage>(r#"{"content":" \t\n "}"#).is_err();
        let expected = true;

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_steer_queue_drain_is_fifo_and_non_duplicating() {
        let mut setup = SteerQueue::default();
        setup.push(SteerMessage::new("first").unwrap());
        setup.push(SteerMessage::new("second").unwrap());

        let actual = (
            setup
                .drain()
                .map(|message| message.content().to_string())
                .collect::<Vec<_>>(),
            setup
                .drain()
                .map(|message| message.content().to_string())
                .collect::<Vec<_>>(),
        );
        let expected = (
            vec!["first".to_string(), "second".to_string()],
            Vec::<String>::new(),
        );

        assert_eq!(actual, expected);
    }
}
