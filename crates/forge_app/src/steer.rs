use std::sync::Arc;

use forge_domain::SteerRequest;

use crate::ConversationService;
use crate::services::SteerService;

/// Typed handle for accepting main-conversation steer messages.
pub struct SteerHandle<S> {
    services: Arc<S>,
}

impl<S: ConversationService + SteerService> SteerHandle<S> {
    /// Creates a steer handle backed by conversation and steer services.
    ///
    /// # Arguments
    /// * `services` - Service facade used to validate and enqueue steer.
    pub fn new(services: Arc<S>) -> Self {
        Self { services }
    }

    /// Accepts a steer message only for primary user conversations.
    ///
    /// # Arguments
    /// * `request` - Typed steer request containing target conversation and message.
    ///
    /// # Errors
    /// Returns an error when the conversation does not exist or is not primary.
    pub async fn accept(&self, request: SteerRequest) -> anyhow::Result<()> {
        let conversation = self
            .services
            .find_conversation(&request.conversation_id)
            .await?
            .ok_or_else(|| forge_domain::Error::ConversationNotFound(request.conversation_id))?;

        if !conversation.is_primary_user_conversation() {
            return Err(forge_domain::Error::SteerRejectedNonPrimaryConversation.into());
        }

        self.services
            .enqueue_steer(&request.conversation_id, request.message)
            .await?;

        let conversation = self
            .services
            .find_conversation(&request.conversation_id)
            .await?
            .ok_or_else(|| forge_domain::Error::ConversationNotFound(request.conversation_id))?;
        if !conversation.is_primary_user_conversation() {
            self.services.clear_steer(&request.conversation_id).await?;
            return Err(forge_domain::Error::SteerRejectedNonPrimaryConversation.into());
        }

        let is_still_primary = self
            .services
            .modify_conversation(&request.conversation_id, |conversation| {
                conversation.is_primary_user_conversation()
            })
            .await?;
        if !is_still_primary {
            self.services.clear_steer(&request.conversation_id).await?;
            return Err(forge_domain::Error::SteerRejectedNonPrimaryConversation.into());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use forge_domain::{
        Context, Conversation, ConversationId, Initiator, SteerMessage, SteerQueue,
    };
    use pretty_assertions::assert_eq;
    use tokio::sync::Mutex;

    #[derive(Default)]
    struct FixtureServices {
        conversations: Mutex<HashMap<ConversationId, Conversation>>,
        steer: Mutex<HashMap<ConversationId, SteerQueue>>,
        promote_on_enqueue: Mutex<Option<(ConversationId, ConversationId)>>,
        promote_after_find: Mutex<Option<(usize, ConversationId, ConversationId)>>,
        find_count: Mutex<usize>,
    }

    #[async_trait::async_trait]
    impl ConversationService for FixtureServices {
        async fn find_conversation(
            &self,
            id: &ConversationId,
        ) -> anyhow::Result<Option<Conversation>> {
            let conversation = self.conversations.lock().await.get(id).cloned();
            let mut find_count = self.find_count.lock().await;
            *find_count += 1;
            let current_find = *find_count;
            drop(find_count);
            if let Some((target_find, target_id, parent_id)) = {
                let mut promote_after_find = self.promote_after_find.lock().await;
                let next_promotion = promote_after_find.take();
                if let Some((target_find, target_id, parent_id)) = next_promotion {
                    if target_find == current_find && target_id == *id {
                        Some((target_find, target_id, parent_id))
                    } else {
                        *promote_after_find = Some((target_find, target_id, parent_id));
                        None
                    }
                } else {
                    None
                }
            } {
                {
                    let mut conversations = self.conversations.lock().await;
                    let conversation = conversations
                        .get_mut(id)
                        .ok_or_else(|| forge_domain::Error::ConversationNotFound(*id))?;
                    conversation.ensure_delegated(Some(parent_id));
                }
                let _ = target_find;
                let _ = target_id;
            }
            Ok(conversation)
        }

        async fn upsert_conversation(&self, conversation: Conversation) -> anyhow::Result<()> {
            self.conversations
                .lock()
                .await
                .insert(conversation.id, conversation);
            Ok(())
        }

        async fn ensure_delegated_conversation(
            &self,
            id: &ConversationId,
            parent_id: Option<ConversationId>,
        ) -> anyhow::Result<Conversation> {
            let mut conversations = self.conversations.lock().await;
            let conversation = conversations
                .get_mut(id)
                .ok_or_else(|| forge_domain::Error::ConversationNotFound(*id))?;
            conversation.ensure_delegated(parent_id);
            Ok(conversation.clone())
        }

        async fn modify_conversation<F, T>(&self, id: &ConversationId, f: F) -> anyhow::Result<T>
        where
            F: FnOnce(&mut Conversation) -> T + Send,
            T: Send,
        {
            let mut conversations = self.conversations.lock().await;
            let conversation = conversations
                .get_mut(id)
                .ok_or_else(|| forge_domain::Error::ConversationNotFound(*id))?;
            Ok(f(conversation))
        }

        async fn get_conversations(&self) -> anyhow::Result<Vec<Conversation>> {
            Ok(self.conversations.lock().await.values().cloned().collect())
        }

        async fn get_sub_conversations(
            &self,
            parent_id: &ConversationId,
        ) -> anyhow::Result<Vec<Conversation>> {
            Ok(self
                .conversations
                .lock()
                .await
                .values()
                .filter(|conversation| conversation.parent_id == Some(*parent_id))
                .cloned()
                .collect())
        }

        async fn last_conversation(&self) -> anyhow::Result<Option<Conversation>> {
            Ok(self.conversations.lock().await.values().next().cloned())
        }

        async fn delete_conversation(
            &self,
            conversation_id: &ConversationId,
        ) -> anyhow::Result<()> {
            self.conversations.lock().await.remove(conversation_id);
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl SteerService for FixtureServices {
        async fn enqueue_steer(
            &self,
            conversation_id: &ConversationId,
            message: SteerMessage,
        ) -> anyhow::Result<()> {
            if let Some((target_id, parent_id)) = self.promote_on_enqueue.lock().await.take()
                && target_id == *conversation_id
            {
                self.ensure_delegated_conversation(conversation_id, Some(parent_id))
                    .await?;
            }
            self.steer
                .lock()
                .await
                .entry(*conversation_id)
                .or_default()
                .push(message);
            Ok(())
        }

        async fn clear_steer(&self, conversation_id: &ConversationId) -> anyhow::Result<()> {
            self.steer.lock().await.remove(conversation_id);
            Ok(())
        }

        async fn drain_steer(
            &self,
            conversation_id: &ConversationId,
        ) -> anyhow::Result<Vec<SteerMessage>> {
            Ok(self
                .steer
                .lock()
                .await
                .remove(conversation_id)
                .map(|mut queue| queue.drain().collect())
                .unwrap_or_default())
        }
    }

    fn primary_conversation() -> Conversation {
        Conversation::generate().context(Context::default())
    }

    fn delegated_conversation(parent_id: ConversationId) -> Conversation {
        Conversation::generate()
            .parent_id(parent_id)
            .initiator(Initiator::Agent)
            .context(Context::default())
    }

    #[tokio::test]
    async fn test_steer_is_accepted_only_for_primary_conversation() {
        let setup = Arc::new(FixtureServices::default());
        let primary = primary_conversation();
        let delegated = delegated_conversation(primary.id);
        setup.upsert_conversation(primary.clone()).await.unwrap();
        setup.upsert_conversation(delegated.clone()).await.unwrap();

        let handle = SteerHandle::new(setup.clone());
        handle
            .accept(SteerRequest::new(
                primary.id,
                SteerMessage::new("primary steer").unwrap(),
            ))
            .await
            .unwrap();
        let actual = handle
            .accept(SteerRequest::new(
                delegated.id,
                SteerMessage::new("delegated steer").unwrap(),
            ))
            .await
            .is_err();
        let expected = true;

        assert_eq!(actual, expected);
        assert_eq!(setup.drain_steer(&primary.id).await.unwrap().len(), 1);
        assert_eq!(setup.drain_steer(&delegated.id).await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_steer_accept_rejects_promotion_after_final_primary_check() {
        let setup = Arc::new(FixtureServices::default());
        let primary = primary_conversation();
        let parent = primary_conversation();
        setup.upsert_conversation(primary.clone()).await.unwrap();
        setup.upsert_conversation(parent.clone()).await.unwrap();
        *setup.promote_after_find.lock().await = Some((2, primary.id, parent.id));

        let handle = SteerHandle::new(setup.clone());
        let actual = handle
            .accept(SteerRequest::new(
                primary.id,
                SteerMessage::new("post-check racy steer").unwrap(),
            ))
            .await
            .is_err();
        let expected = true;

        assert_eq!(actual, expected);
        assert_eq!(setup.drain_steer(&primary.id).await.unwrap(), Vec::new());
    }
}
