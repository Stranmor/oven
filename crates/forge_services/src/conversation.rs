use std::sync::Arc;

use anyhow::Result;
use forge_app::ConversationService;
use forge_app::domain::{Conversation, ConversationId};
use forge_domain::ConversationRepository;

/// Service for managing conversations, including creation, retrieval, and
/// updates
#[derive(Clone)]
pub struct ForgeConversationService<S> {
    conversation_repository: Arc<S>,
}

impl<S: ConversationRepository> ForgeConversationService<S> {
    /// Creates a new ForgeConversationService with the provided repository
    pub fn new(repo: Arc<S>) -> Self {
        Self { conversation_repository: repo }
    }
}

#[async_trait::async_trait]
impl<S: ConversationRepository> ConversationService for ForgeConversationService<S> {
    async fn modify_conversation<F, T>(&self, id: &ConversationId, f: F) -> Result<T>
    where
        F: FnOnce(&mut Conversation) -> T + Send,
        T: Send,
    {
        let mut conversation = self
            .conversation_repository
            .get_conversation(id)
            .await?
            .ok_or_else(|| forge_app::domain::Error::ConversationNotFound(*id))?;
        let out = f(&mut conversation);
        let _ = self
            .conversation_repository
            .upsert_conversation(conversation)
            .await?;
        Ok(out)
    }

    async fn find_conversation(&self, id: &ConversationId) -> Result<Option<Conversation>> {
        self.conversation_repository.get_conversation(id).await
    }

    async fn upsert_conversation(&self, conversation: Conversation) -> Result<()> {
        let _ = self
            .conversation_repository
            .upsert_conversation(conversation)
            .await?;
        Ok(())
    }

    async fn ensure_delegated_conversation(
        &self,
        id: &ConversationId,
        parent_id: Option<ConversationId>,
    ) -> Result<Conversation> {
        let mut conversation = self
            .conversation_repository
            .get_conversation(id)
            .await?
            .ok_or_else(|| forge_app::domain::Error::ConversationNotFound(*id))?;
        conversation.ensure_delegated(parent_id);
        self.conversation_repository
            .upsert_conversation(conversation.clone())
            .await?;
        Ok(conversation)
    }

    async fn get_conversations(&self) -> Result<Vec<Conversation>> {
        self.conversation_repository.get_all_conversations().await
    }

    async fn get_sub_conversations(&self, parent_id: &ConversationId) -> Result<Vec<Conversation>> {
        self.conversation_repository
            .get_sub_conversations(parent_id)
            .await
    }

    async fn last_conversation(&self) -> Result<Option<Conversation>> {
        self.conversation_repository.get_last_conversation().await
    }

    async fn delete_conversation(&self, conversation_id: &ConversationId) -> Result<()> {
        self.conversation_repository
            .delete_conversation(conversation_id)
            .await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use forge_app::ConversationService;
    use forge_app::domain::{Context, ContextMessage, ConversationId, Initiator};
    use pretty_assertions::assert_eq;

    use super::*;

    #[derive(Default)]
    struct FixtureRepository {
        conversations: Mutex<HashMap<ConversationId, Conversation>>,
    }

    #[async_trait::async_trait]
    impl ConversationRepository for FixtureRepository {
        async fn upsert_conversation(&self, conversation: Conversation) -> anyhow::Result<()> {
            self.conversations
                .lock()
                .unwrap()
                .insert(conversation.id, conversation);
            Ok(())
        }

        async fn get_conversation(
            &self,
            conversation_id: &ConversationId,
        ) -> anyhow::Result<Option<Conversation>> {
            Ok(self
                .conversations
                .lock()
                .unwrap()
                .get(conversation_id)
                .cloned())
        }

        async fn get_all_conversations(&self) -> anyhow::Result<Vec<Conversation>> {
            Ok(self
                .conversations
                .lock()
                .unwrap()
                .values()
                .cloned()
                .collect())
        }

        async fn get_sub_conversations(
            &self,
            parent_id: &ConversationId,
        ) -> anyhow::Result<Vec<Conversation>> {
            Ok(self
                .conversations
                .lock()
                .unwrap()
                .values()
                .filter(|conversation| conversation.parent_id == Some(*parent_id))
                .cloned()
                .collect())
        }

        async fn get_last_conversation(&self) -> anyhow::Result<Option<Conversation>> {
            Ok(self.conversations.lock().unwrap().values().next().cloned())
        }

        async fn delete_conversation(
            &self,
            conversation_id: &ConversationId,
        ) -> anyhow::Result<()> {
            self.conversations.lock().unwrap().remove(conversation_id);
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_ensure_delegated_conversation_promotes_reused_session() -> anyhow::Result<()> {
        let repository = Arc::new(FixtureRepository::default());
        let service = ForgeConversationService::new(repository.clone());
        let parent_id = ConversationId::generate();
        let conversation = Conversation::new(ConversationId::generate()).context(Some(
            Context::default().messages(vec![ContextMessage::user("User chat", None).into()]),
        ));

        repository.upsert_conversation(conversation.clone()).await?;
        let actual = service
            .ensure_delegated_conversation(&conversation.id, Some(parent_id))
            .await?;
        let expected = (Initiator::Agent, Some(parent_id));

        assert_eq!((actual.initiator, actual.parent_id), expected);
        let persisted = repository
            .get_conversation(&conversation.id)
            .await?
            .expect("promoted conversation should be persisted");
        assert_eq!((persisted.initiator, persisted.parent_id), expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_ensure_delegated_conversation_without_parent_keeps_parentless_agent()
    -> anyhow::Result<()> {
        let repository = Arc::new(FixtureRepository::default());
        let service = ForgeConversationService::new(repository.clone());
        let conversation = Conversation::new(ConversationId::generate()).context(Some(
            Context::default().messages(vec![ContextMessage::user("User chat", None).into()]),
        ));

        repository.upsert_conversation(conversation.clone()).await?;
        let actual = service
            .ensure_delegated_conversation(&conversation.id, None)
            .await?;
        let expected = (Initiator::Agent, None);

        assert_eq!((actual.initiator, actual.parent_id), expected);
        let persisted = repository
            .get_conversation(&conversation.id)
            .await?
            .expect("parentless promoted conversation should be persisted");
        assert_eq!((persisted.initiator, persisted.parent_id), expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_ensure_delegated_conversation_reparents_already_agent_session()
    -> anyhow::Result<()> {
        let repository = Arc::new(FixtureRepository::default());
        let service = ForgeConversationService::new(repository.clone());
        let parent_id = ConversationId::generate();
        let conversation = Conversation::new(ConversationId::generate())
            .initiator(Initiator::Agent)
            .context(Some(
                Context::default().messages(vec![ContextMessage::user("Agent chat", None).into()]),
            ));

        repository.upsert_conversation(conversation.clone()).await?;
        let actual = service
            .ensure_delegated_conversation(&conversation.id, Some(parent_id))
            .await?;
        let expected = (Initiator::Agent, Some(parent_id));

        assert_eq!((actual.initiator, actual.parent_id), expected);
        Ok(())
    }
}
