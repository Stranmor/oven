use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;
use forge_app::ConversationService;
use forge_app::domain::{
    Conversation, ConversationId, MessageId, SubagentTaskId, SubagentTaskSession,
    SubagentTaskSessionFilter,
};
use forge_app::dto::ConversationBranchTarget;
use forge_domain::{ConversationRepository, ConversationVisibilityFilter};

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
        self.conversation_repository
            .promote_delegated_conversation(id, parent_id)
            .await
    }

    async fn resolve_root_conversation_id(
        &self,
        parent_id: Option<ConversationId>,
    ) -> Result<Option<ConversationId>> {
        let Some(mut current_id) = parent_id else {
            return Ok(None);
        };
        let mut root_id = current_id;
        let mut seen = HashSet::new();
        while seen.insert(current_id) {
            let Some(parent) = self
                .conversation_repository
                .get_conversation(&current_id)
                .await?
            else {
                break;
            };
            let Some(next_parent_id) = parent.parent_id else {
                break;
            };
            root_id = next_parent_id;
            current_id = next_parent_id;
        }
        Ok(Some(root_id))
    }

    async fn list_branch_targets(
        &self,
        conversation_id: &ConversationId,
    ) -> Result<Vec<ConversationBranchTarget>> {
        let source = self
            .conversation_repository
            .get_conversation(conversation_id)
            .await?
            .ok_or_else(|| forge_app::domain::Error::ConversationNotFound(*conversation_id))?;
        let mut context = source
            .context
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Conversation {conversation_id} has no context"))?;
        context.conversation_id = Some(source.id);
        Ok(ConversationBranchTarget::list_from_context(
            source.id, &context,
        ))
    }

    async fn branch_conversation(
        &self,
        conversation_id: &ConversationId,
        target_id: MessageId,
    ) -> Result<Conversation> {
        let mut source = self
            .conversation_repository
            .get_conversation(conversation_id)
            .await?
            .ok_or_else(|| forge_app::domain::Error::ConversationNotFound(*conversation_id))?;
        let mut context = source
            .context
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Conversation {conversation_id} has no context"))?;
        context.conversation_id = Some(source.id);
        let source_normalized = context.normalize_message_ids();
        if source_normalized {
            source.context = Some(context.clone());
            self.conversation_repository
                .upsert_conversation(source.clone())
                .await?;
        }
        let boundary = context.branch_boundary_for(target_id)?;
        let new_id = ConversationId::generate();
        let mut branch_context = boundary.branch_context(&context);
        branch_context.conversation_id = Some(new_id);
        branch_context.normalize_message_ids();
        let branch = Conversation::new(new_id)
            .title(
                source
                    .title
                    .clone()
                    .map(|title| format!("{title} (branch)")),
            )
            .context(Some(branch_context))
            .initiator(source.initiator);
        self.conversation_repository
            .upsert_conversation(branch.clone())
            .await?;
        Ok(branch)
    }

    async fn get_conversations(&self) -> Result<Vec<Conversation>> {
        self.conversation_repository.get_all_conversations().await
    }

    async fn get_conversations_including_agent(&self) -> Result<Vec<Conversation>> {
        self.conversation_repository
            .get_all_conversations_including_agent()
            .await
    }

    async fn get_conversations_by_visibility(
        &self,
        visibility: ConversationVisibilityFilter,
    ) -> Result<Vec<Conversation>> {
        self.conversation_repository
            .get_all_conversations_by_visibility(visibility)
            .await
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
    use forge_app::domain::{
        Context, ContextMessage, ConversationId, Initiator, MessageEntry, Role, TextMessage,
    };
    use pretty_assertions::assert_eq;

    use super::*;

    #[derive(Default)]
    struct FixtureRepository {
        conversations: Mutex<HashMap<ConversationId, Conversation>>,
        subagent_task_sessions: Mutex<HashMap<ConversationId, SubagentTaskSession>>,
        ledger_insert_after_lookup: Mutex<Option<SubagentTaskSession>>,
        delete_count: Mutex<usize>,
    }

    #[async_trait::async_trait]
    impl ConversationRepository for FixtureRepository {
        async fn promote_delegated_conversation(
            &self,
            conversation_id: &ConversationId,
            parent_id: Option<ConversationId>,
        ) -> anyhow::Result<Conversation> {
            let mut conversations = self.conversations.lock().unwrap();
            let mut subagent_task_sessions = self.subagent_task_sessions.lock().unwrap();
            if let Some(session) = self.ledger_insert_after_lookup.lock().unwrap().take() {
                subagent_task_sessions.insert(session.conversation_id, session);
            }
            let conversation = conversations
                .get_mut(conversation_id)
                .ok_or_else(|| forge_app::domain::Error::ConversationNotFound(*conversation_id))?;
            if conversation.parent_id.is_some() && conversation.parent_id != parent_id {
                anyhow::bail!(
                    "Conversation {conversation_id} is already owned by parent {:?}; refusing silent reparent to {:?}",
                    conversation.parent_id,
                    parent_id
                );
            }
            if let Some(existing) = subagent_task_sessions.get(conversation_id)
                && existing.parent_conversation_id.is_some()
                && existing.parent_conversation_id != parent_id
            {
                anyhow::bail!(
                    "Subagent session {conversation_id} belongs to parent {:?}; refusing silent reparent to {:?}",
                    existing.parent_conversation_id,
                    parent_id
                );
            }
            conversation.ensure_delegated(parent_id);
            Ok(conversation.clone())
        }

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

        async fn get_all_conversations_including_agent(&self) -> anyhow::Result<Vec<Conversation>> {
            self.get_all_conversations().await
        }

        async fn get_all_conversations_by_visibility(
            &self,
            visibility: forge_app::domain::ConversationVisibilityFilter,
        ) -> anyhow::Result<Vec<Conversation>> {
            let conversations = self.get_all_conversations_including_agent().await?;
            Ok(conversations
                .into_iter()
                .filter(|conversation| match visibility {
                    forge_app::domain::ConversationVisibilityFilter::Normal => {
                        conversation.is_normal_visibility()
                    }
                    forge_app::domain::ConversationVisibilityFilter::Background => {
                        conversation.is_background()
                    }
                    forge_app::domain::ConversationVisibilityFilter::All => true,
                })
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
            session: SubagentTaskSession,
        ) -> anyhow::Result<()> {
            self.subagent_task_sessions
                .lock()
                .unwrap()
                .insert(session.conversation_id, session);
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
            conversation_id: &ConversationId,
        ) -> anyhow::Result<Option<SubagentTaskSession>> {
            let actual = self
                .subagent_task_sessions
                .lock()
                .unwrap()
                .get(conversation_id)
                .cloned();
            if let Some(session) = self.ledger_insert_after_lookup.lock().unwrap().take() {
                self.subagent_task_sessions
                    .lock()
                    .unwrap()
                    .insert(session.conversation_id, session);
            }
            Ok(actual)
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
            *self.delete_count.lock().unwrap() += 1;
            self.conversations.lock().unwrap().remove(conversation_id);
            Ok(())
        }
    }

    fn text_contents(conversation: &Conversation) -> Vec<String> {
        conversation
            .context
            .as_ref()
            .map(|context| {
                context
                    .messages
                    .iter()
                    .filter_map(|entry| entry.message.content().map(ToString::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn user_message(content: impl Into<String>) -> MessageEntry {
        ContextMessage::user(content.into(), None).into()
    }

    fn assistant_message(content: impl Into<String>) -> MessageEntry {
        ContextMessage::assistant(content.into(), None, None, None).into()
    }

    #[tokio::test]
    async fn test_list_branch_targets_is_read_only_and_preserves_metadata() -> anyhow::Result<()> {
        let repository = Arc::new(FixtureRepository::default());
        let service = ForgeConversationService::new(repository.clone());
        let source_id = ConversationId::generate();
        let source = Conversation::new(source_id)
            .title(Some("Source".to_string()))
            .context(Some(
                Context::default().conversation_id(source_id).messages(vec![
                    ContextMessage::system("system").into(),
                    user_message("hello\nuser"),
                    assistant_message("hello assistant"),
                ]),
            ));

        repository.upsert_conversation(source.clone()).await?;
        let actual = service.list_branch_targets(&source_id).await?;
        let persisted_source = repository
            .get_conversation(&source_id)
            .await?
            .expect("source conversation should remain persisted");
        let expected = vec![
            (source_id, 1usize, Role::User, "hello user".to_string()),
            (
                source_id,
                2usize,
                Role::Assistant,
                "hello assistant".to_string(),
            ),
        ];
        let actual_metadata = actual
            .iter()
            .map(|target| {
                (
                    target.conversation_id,
                    target.ordinal,
                    target.role,
                    target.preview.clone(),
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(actual_metadata, expected);
        assert_eq!(text_contents(&persisted_source), text_contents(&source));
        assert!(
            actual
                .iter()
                .all(|target| !target.message_id.into_string().is_empty())
        );
        assert!(
            persisted_source
                .context
                .as_ref()
                .expect("source context should exist")
                .messages
                .iter()
                .all(|entry| entry.id.is_none())
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_list_branch_targets_uses_domain_filter_for_negative_cases() -> anyhow::Result<()>
    {
        let repository = Arc::new(FixtureRepository::default());
        let service = ForgeConversationService::new(repository.clone());
        let source_id = ConversationId::generate();
        let source = Conversation::new(source_id).context(Some(
            Context::default().conversation_id(source_id).messages(vec![
                ContextMessage::system("system").into(),
                ContextMessage::Text(TextMessage::learning_context(Role::User, "learning")).into(),
                ContextMessage::Text(TextMessage::new(Role::User, "droppable").droppable(true))
                    .into(),
                ContextMessage::assistant(
                    "assistant with tool",
                    None,
                    None,
                    Some(vec![
                        forge_domain::ToolCallFull::new("read").call_id("call_id"),
                    ]),
                )
                .into(),
                user_message("kept"),
            ]),
        ));

        repository.upsert_conversation(source).await?;
        let actual = service
            .list_branch_targets(&source_id)
            .await?
            .into_iter()
            .map(|target| (target.ordinal, target.role, target.preview))
            .collect::<Vec<_>>();
        let expected = vec![(4usize, Role::User, "kept".to_string())];

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_branch_conversation_persists_new_branch_and_preserves_source_history()
    -> anyhow::Result<()> {
        let repository = Arc::new(FixtureRepository::default());
        let service = ForgeConversationService::new(repository.clone());
        let source_id = ConversationId::generate();
        let source = Conversation::new(source_id)
            .title(Some("Source".to_string()))
            .context(Some(
                Context::default().conversation_id(source_id).messages(vec![
                    user_message("keep"),
                    user_message("bad"),
                    assistant_message("after"),
                ]),
            ));
        let target_id = source
            .context
            .as_ref()
            .map(|context| {
                forge_domain::MessageId::materialized(
                    Some(source_id),
                    1,
                    &context
                        .messages
                        .get(1)
                        .expect("target message exists")
                        .message,
                )
            })
            .expect("source context exists");

        repository.upsert_conversation(source.clone()).await?;
        let actual = service.branch_conversation(&source_id, target_id).await?;
        let persisted_source = repository
            .get_conversation(&source_id)
            .await?
            .expect("source conversation should remain persisted");
        let persisted_branch = repository
            .get_conversation(&actual.id)
            .await?
            .expect("branch conversation should be persisted");
        let expected_source_messages =
            vec!["keep".to_string(), "bad".to_string(), "after".to_string()];
        let expected_branch_messages = vec!["keep".to_string()];

        assert_ne!(actual.id, source_id);
        assert_eq!(text_contents(&persisted_source), expected_source_messages);
        assert_eq!(text_contents(&persisted_branch), expected_branch_messages);
        assert_eq!(persisted_branch.parent_id, None);
        assert_eq!(*repository.delete_count.lock().unwrap(), 0);
        assert!(
            persisted_source
                .context
                .as_ref()
                .expect("source context should exist")
                .messages
                .iter()
                .all(|entry| entry.id.is_some())
        );
        Ok(())
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

    #[tokio::test]
    async fn test_ensure_delegated_conversation_rejects_ledger_owned_parentless_reparent()
    -> anyhow::Result<()> {
        let repository = Arc::new(FixtureRepository::default());
        let service = ForgeConversationService::new(repository.clone());
        let original_parent_id = ConversationId::generate();
        let unrelated_parent_id = ConversationId::generate();
        let conversation = Conversation::new(ConversationId::generate())
            .initiator(Initiator::Agent)
            .context(Some(
                Context::default().messages(vec![ContextMessage::user("Agent chat", None).into()]),
            ));
        let session = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            conversation.id,
            Some(original_parent_id),
            Some(original_parent_id),
            "resume guarded task",
        );

        repository.upsert_conversation(conversation.clone()).await?;
        repository.upsert_subagent_task_session(session).await?;
        let actual = service
            .ensure_delegated_conversation(&conversation.id, Some(unrelated_parent_id))
            .await;
        let expected = None;

        assert!(actual.is_err());
        let persisted = repository
            .get_conversation(&conversation.id)
            .await?
            .expect("conversation should remain persisted");
        assert_eq!(persisted.parent_id, expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_ensure_delegated_conversation_rejects_ledger_inserted_between_guard_and_persist()
    -> anyhow::Result<()> {
        let repository = Arc::new(FixtureRepository::default());
        let service = ForgeConversationService::new(repository.clone());
        let original_parent_id = ConversationId::generate();
        let unrelated_parent_id = ConversationId::generate();
        let conversation = Conversation::new(ConversationId::generate())
            .initiator(Initiator::Agent)
            .context(Some(
                Context::default().messages(vec![ContextMessage::user("Agent chat", None).into()]),
            ));
        let session = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            conversation.id,
            Some(original_parent_id),
            Some(original_parent_id),
            "racing ledger owner",
        );

        repository.upsert_conversation(conversation.clone()).await?;
        *repository.ledger_insert_after_lookup.lock().unwrap() = Some(session);
        let actual = service
            .ensure_delegated_conversation(&conversation.id, Some(unrelated_parent_id))
            .await;
        let expected = None;

        assert!(actual.is_err());
        let persisted = repository
            .get_conversation(&conversation.id)
            .await?
            .expect("conversation should remain persisted");
        assert_eq!(persisted.parent_id, expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_resolve_root_conversation_id_walks_nested_parent_chain() -> anyhow::Result<()> {
        let repository = Arc::new(FixtureRepository::default());
        let service = ForgeConversationService::new(repository.clone());
        let root = Conversation::new(ConversationId::generate()).context(Some(
            Context::default().messages(vec![ContextMessage::user("Root chat", None).into()]),
        ));
        let child = Conversation::new(ConversationId::generate())
            .initiator(Initiator::Agent)
            .parent_id(root.id)
            .context(Some(
                Context::default().messages(vec![ContextMessage::user("Child chat", None).into()]),
            ));
        let grandchild = Conversation::new(ConversationId::generate())
            .initiator(Initiator::Agent)
            .parent_id(child.id)
            .context(Some(Context::default().messages(vec![
                ContextMessage::user("Grandchild chat", None).into(),
            ])));

        repository.upsert_conversation(root.clone()).await?;
        repository.upsert_conversation(child.clone()).await?;
        repository.upsert_conversation(grandchild.clone()).await?;
        let actual = service
            .resolve_root_conversation_id(Some(grandchild.id))
            .await?;
        let expected = Some(root.id);

        assert_eq!(actual, expected);
        Ok(())
    }
}
