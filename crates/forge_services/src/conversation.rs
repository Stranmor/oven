use std::sync::Arc;

use anyhow::Result;
use forge_app::ConversationService;
use forge_app::domain::{
    Conversation, ConversationId, SubagentTaskId, SubagentTaskSession, SubagentTaskSessionFilter,
};
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
        if conversation.parent_id.is_some() && conversation.parent_id != parent_id {
            anyhow::bail!(
                "Conversation {id} is already owned by parent {:?}; refusing silent reparent to {:?}",
                conversation.parent_id,
                parent_id
            );
        }
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

    async fn upsert_subagent_task_session(&self, session: SubagentTaskSession) -> Result<()> {
        self.conversation_repository
            .upsert_subagent_task_session(session)
            .await
    }

    async fn get_subagent_task_session(
        &self,
        task_id: &SubagentTaskId,
    ) -> Result<Option<SubagentTaskSession>> {
        self.conversation_repository
            .get_subagent_task_session(task_id)
            .await
    }

    async fn get_subagent_task_session_by_conversation(
        &self,
        conversation_id: &ConversationId,
    ) -> Result<Option<SubagentTaskSession>> {
        self.conversation_repository
            .get_subagent_task_session_by_conversation(conversation_id)
            .await
    }

    async fn list_subagent_task_sessions(
        &self,
        filter: SubagentTaskSessionFilter,
    ) -> Result<Vec<SubagentTaskSession>> {
        self.conversation_repository
            .list_subagent_task_sessions(filter)
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

        async fn upsert_subagent_task_session(
            &self,
            _session: SubagentTaskSession,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn get_subagent_task_session(
            &self,
            _task_id: &SubagentTaskId,
        ) -> anyhow::Result<Option<SubagentTaskSession>> {
            Ok(None)
        }

        async fn get_subagent_task_session_by_conversation(
            &self,
            _conversation_id: &ConversationId,
        ) -> anyhow::Result<Option<SubagentTaskSession>> {
            Ok(None)
        }

        async fn list_subagent_task_sessions(
            &self,
            _filter: SubagentTaskSessionFilter,
        ) -> anyhow::Result<Vec<SubagentTaskSession>> {
            Ok(Vec::new())
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
    async fn test_ensure_delegated_conversation_rejects_unrelated_reparent() -> anyhow::Result<()> {
        let repository = Arc::new(FixtureRepository::default());
        let service = ForgeConversationService::new(repository.clone());
        let original_parent_id = ConversationId::generate();
        let unrelated_parent_id = ConversationId::generate();
        let conversation = Conversation::new(ConversationId::generate())
            .initiator(Initiator::Agent)
            .parent_id(original_parent_id)
            .context(Some(
                Context::default().messages(vec![ContextMessage::user("Agent chat", None).into()]),
            ));

        repository.upsert_conversation(conversation.clone()).await?;
        let actual = service
            .ensure_delegated_conversation(&conversation.id, Some(unrelated_parent_id))
            .await;
        let expected = original_parent_id;

        assert!(actual.is_err());
        let persisted = repository
            .get_conversation(&conversation.id)
            .await?
            .expect("conversation should remain persisted");
        assert_eq!(persisted.parent_id, Some(expected));
        Ok(())
    }

    #[tokio::test]
    async fn test_ensure_delegated_conversation_rejects_parented_session_from_parentless_resume()
    -> anyhow::Result<()> {
        let repository = Arc::new(FixtureRepository::default());
        let service = ForgeConversationService::new(repository.clone());
        let original_parent_id = ConversationId::generate();
        let conversation = Conversation::new(ConversationId::generate())
            .initiator(Initiator::Agent)
            .parent_id(original_parent_id)
            .context(Some(
                Context::default().messages(vec![ContextMessage::user("Agent chat", None).into()]),
            ));

        repository.upsert_conversation(conversation.clone()).await?;
        let actual = service
            .ensure_delegated_conversation(&conversation.id, None)
            .await;
        let expected = original_parent_id;

        assert!(actual.is_err());
        let persisted = repository
            .get_conversation(&conversation.id)
            .await?
            .expect("conversation should remain persisted");
        assert_eq!(persisted.parent_id, Some(expected));
        Ok(())
    }
}
