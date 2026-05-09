use std::collections::HashMap;
use std::sync::Arc;

use forge_app::SteerService;
use forge_domain::{ConversationId, SteerMessage, SteerQueue};
use tokio::sync::Mutex;

/// Process-local typed steer queue store.
#[derive(Default)]
pub struct ForgeSteerService {
    queues: Mutex<HashMap<ConversationId, SteerQueue>>,
}

impl ForgeSteerService {
    /// Creates a shared steer service instance.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
}

#[async_trait::async_trait]
impl SteerService for ForgeSteerService {
    async fn enqueue_steer(
        &self,
        conversation_id: &ConversationId,
        message: SteerMessage,
    ) -> anyhow::Result<()> {
        self.queues
            .lock()
            .await
            .entry(*conversation_id)
            .or_default()
            .push(message);
        Ok(())
    }

    async fn clear_steer(&self, conversation_id: &ConversationId) -> anyhow::Result<()> {
        self.queues.lock().await.remove(conversation_id);
        Ok(())
    }

    async fn drain_steer(
        &self,
        conversation_id: &ConversationId,
    ) -> anyhow::Result<Vec<SteerMessage>> {
        Ok(self
            .queues
            .lock()
            .await
            .remove(conversation_id)
            .map(|mut queue| queue.drain().collect())
            .unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use forge_app::SteerService;
    use forge_domain::{ConversationId, SteerMessage};
    use pretty_assertions::assert_eq;

    use super::*;

    #[tokio::test]
    async fn test_drain_steer_is_fifo_and_non_duplicating() {
        let setup = ForgeSteerService::default();
        let conversation_id = ConversationId::generate();
        setup
            .enqueue_steer(&conversation_id, SteerMessage::new("first").unwrap())
            .await
            .unwrap();
        setup
            .enqueue_steer(&conversation_id, SteerMessage::new("second").unwrap())
            .await
            .unwrap();

        let actual = (
            setup
                .drain_steer(&conversation_id)
                .await
                .unwrap()
                .into_iter()
                .map(|message| message.content().to_string())
                .collect::<Vec<_>>(),
            setup
                .drain_steer(&conversation_id)
                .await
                .unwrap()
                .into_iter()
                .map(|message| message.content().to_string())
                .collect::<Vec<_>>(),
        );
        let expected = (
            vec!["first".to_string(), "second".to_string()],
            Vec::<String>::new(),
        );

        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn test_clear_steer_discards_queue_without_delivery() {
        let setup = ForgeSteerService::default();
        let conversation_id = ConversationId::generate();
        setup
            .enqueue_steer(&conversation_id, SteerMessage::new("stale").unwrap())
            .await
            .unwrap();

        setup.clear_steer(&conversation_id).await.unwrap();
        let actual = setup.drain_steer(&conversation_id).await.unwrap();
        let expected = Vec::new();

        assert_eq!(actual, expected);
    }
}
