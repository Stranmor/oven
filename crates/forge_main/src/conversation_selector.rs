use anyhow::Result;
use forge_api::{Conversation, ConversationListItem};
use forge_domain::ConversationId;
use forge_select::{ForgeWidget, SelectRow};

use crate::display_constants::markers;
use crate::info::Info;
use crate::porcelain::Porcelain;
use crate::utils::humanize_time;

/// Logic for selecting conversations from a list
pub struct ConversationSelector;

impl ConversationSelector {
    /// Select a conversation from the provided list using a custom TUI with
    /// a preview pane showing conversation details.
    ///
    /// The preview command uses `forge conversation info` and
    /// `forge conversation show` to display the selected conversation's
    /// metadata and last message side-by-side with the picker list.
    ///
    /// Returns the selected conversation, or None if the user cancelled.
    ///
    /// # Arguments
    /// * `conversations` - Conversations available for primary conversation
    ///   selection.
    /// * `current_conversation_id` - Optional conversation ID to focus
    ///   initially.
    /// * `query` - Optional initial fuzzy-search query.
    ///
    /// # Errors
    /// Returns an error if selector rendering or terminal interaction fails.
    pub async fn select_conversation(
        conversations: &[Conversation],
        current_conversation_id: Option<ConversationId>,
        query: Option<String>,
    ) -> Result<Option<Conversation>> {
        Self::select_from_conversations(
            conversations,
            current_conversation_id,
            query,
            Self::primary_conversations,
        )
        .await
    }

    /// Select a conversation ID from metadata-only rows without loading full contexts.
    ///
    /// Preview is intentionally disabled for the fast first slice so selector first
    /// render never spawns a nested `forge conversation show` process.
    ///
    /// # Arguments
    /// * `conversations` - Metadata-only conversation rows for primary selection.
    /// * `current_conversation_id` - Optional conversation ID to focus initially.
    /// * `query` - Optional initial fuzzy-search query.
    ///
    /// # Errors
    /// Returns an error if selector rendering or terminal interaction fails.
    pub async fn select_conversation_item(
        conversations: &[ConversationListItem],
        current_conversation_id: Option<ConversationId>,
        query: Option<String>,
    ) -> Result<Option<ConversationId>> {
        Self::select_from_conversation_items(conversations, current_conversation_id, query).await
    }

    /// Select a sub-conversation from an explicit subchat list.
    ///
    /// Unlike the primary selector, this keeps agent-initiated delegated
    /// conversations because explicit subchat browsing is the operator surface
    /// for delegated work.
    ///
    /// # Arguments
    /// * `conversations` - Sub-conversations available for selection.
    /// * `current_conversation_id` - Optional conversation ID to focus
    ///   initially.
    ///
    /// # Errors
    /// Returns an error if selector rendering or terminal interaction fails.
    pub async fn select_sub_conversation(
        conversations: &[Conversation],
        current_conversation_id: Option<ConversationId>,
    ) -> Result<Option<Conversation>> {
        Self::select_from_conversations(
            conversations,
            current_conversation_id,
            None,
            Self::conversations_with_context,
        )
        .await
    }

    async fn select_from_conversation_items(
        conversations: &[ConversationListItem],
        current_conversation_id: Option<ConversationId>,
        query: Option<String>,
    ) -> Result<Option<ConversationId>> {
        if conversations.is_empty() {
            return Ok(None);
        }

        let valid_conversations = Self::primary_conversation_items(conversations);

        if valid_conversations.is_empty() {
            return Ok(None);
        }

        let rows = Self::rows_for_items(&valid_conversations);
        let conv_map: std::collections::HashMap<String, ConversationId> = valid_conversations
            .into_iter()
            .map(|c| (c.id.to_string(), c.id))
            .collect();
        let initial_raw = current_conversation_id.map(|id| id.to_string());

        let selected_uuid = tokio::task::spawn_blocking(move || -> Result<Option<String>> {
            Ok(ForgeWidget::select_rows("Conversation", rows)
                .query(query)
                .header_lines(1_usize)
                .initial_raw(initial_raw)
                .prompt()?
                .map(|row| row.raw))
        })
        .await??;

        Ok(selected_uuid.and_then(|uuid| conv_map.get(&uuid).copied()))
    }

    async fn select_from_conversations(
        conversations: &[Conversation],
        current_conversation_id: Option<ConversationId>,
        query: Option<String>,
        filter: fn(&[Conversation]) -> Vec<&Conversation>,
    ) -> Result<Option<Conversation>> {
        if conversations.is_empty() {
            return Ok(None);
        }

        let valid_conversations = filter(conversations);

        if valid_conversations.is_empty() {
            return Ok(None);
        }

        let rows = Self::rows_for_conversations(&valid_conversations);

        let conv_map: std::collections::HashMap<String, Conversation> = valid_conversations
            .into_iter()
            .map(|c| (c.id.to_string(), c.clone()))
            .collect();
        let initial_raw = current_conversation_id.map(|id| id.to_string());
        let selected_uuid = tokio::task::spawn_blocking(move || -> Result<Option<String>> {
            Ok(ForgeWidget::select_rows("Conversation", rows)
                .query(query)
                .header_lines(1_usize)
                .initial_raw(initial_raw)
                .prompt()?
                .map(|row| row.raw))
        })
        .await??;

        Ok(selected_uuid.and_then(|uuid| conv_map.get(&uuid).cloned()))
    }

    fn rows_for_conversations(conversations: &[&Conversation]) -> Vec<SelectRow> {
        Self::rows_from_parts(conversations.iter().map(|conv| {
            (
                conv.id,
                conv.title.as_deref(),
                conv.metadata.updated_at.unwrap_or(conv.metadata.created_at),
            )
        }))
    }

    fn rows_for_items(conversations: &[&ConversationListItem]) -> Vec<SelectRow> {
        Self::rows_from_parts(
            conversations
                .iter()
                .map(|conv| (conv.id, conv.title.as_deref(), conv.display_updated_at())),
        )
    }

    fn rows_from_parts<'a>(
        conversations: impl Iterator<
            Item = (
                ConversationId,
                Option<&'a str>,
                chrono::DateTime<chrono::Utc>,
            ),
        >,
    ) -> Vec<SelectRow> {
        let mut info = Info::new();
        let mut ids = Vec::new();

        for (id, title, updated_at) in conversations {
            let title = title
                .map(|t| t.to_string())
                .unwrap_or_else(|| markers::EMPTY.to_string());
            info = info
                .add_title(id)
                .add_key_value("Title", title)
                .add_key_value("Updated", humanize_time(updated_at));
            ids.push(id);
        }

        let porcelain_output = Porcelain::from(&info)
            .drop_col(0)
            .truncate(0, 60)
            .uppercase_headers();
        let porcelain_str = porcelain_output.to_string();
        let all_lines: Vec<&str> = porcelain_str.lines().collect();
        let mut rows: Vec<SelectRow> = Vec::with_capacity(all_lines.len());

        if let Some(header) = all_lines.first() {
            rows.push(SelectRow::header(header.to_string()));
        }

        for (i, line) in all_lines.iter().skip(1).enumerate() {
            if let Some(id) = ids.get(i) {
                let uuid = id.to_string();
                rows.push(SelectRow {
                    raw: uuid.clone(),
                    display: line.to_string(),
                    search: line.to_string(),
                    fields: vec![uuid],
                });
            }
        }

        rows
    }

    fn primary_conversation_items(
        conversations: &[ConversationListItem],
    ) -> Vec<&ConversationListItem> {
        conversations
            .iter()
            .filter(|conv| conv.is_primary_user_conversation())
            .collect()
    }

    fn primary_conversations(conversations: &[Conversation]) -> Vec<&Conversation> {
        conversations
            .iter()
            .filter(|conv| conv.is_primary_user_conversation())
            .collect()
    }

    fn conversations_with_context(conversations: &[Conversation]) -> Vec<&Conversation> {
        conversations
            .iter()
            .filter(|conv| conv.context.is_some())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use forge_api::Conversation;
    use forge_domain::{Context, ContextMessage, ConversationId, MetaData, Metrics};
    use pretty_assertions::assert_eq;

    use super::*;

    fn create_test_conversation(id: &str, title: Option<&str>) -> Conversation {
        let now = Utc::now();
        Conversation {
            id: ConversationId::parse(id).expect("fixture conversation ID should be valid"),
            parent_id: None,
            title: title.map(|t| t.to_string()),
            initiator: forge_domain::Initiator::User,
            visibility: forge_domain::ConversationVisibility::Normal,
            context: Some(Context::default()),
            metrics: Metrics::default().started_at(now),
            metadata: MetaData { created_at: now, updated_at: Some(now) },
        }
    }

    #[tokio::test]
    async fn test_select_conversation_empty_list() {
        let conversations = vec![];
        let result = ConversationSelector::select_conversation(&conversations, None, None)
            .await
            .expect("empty conversation list should be selectable");
        assert!(result.is_none());
    }

    #[test]
    fn test_select_conversation_with_titles() {
        let conversations = [
            create_test_conversation(
                "550e8400-e29b-41d4-a716-446655440000",
                Some("First Conversation"),
            ),
            create_test_conversation(
                "550e8400-e29b-41d4-a716-446655440001",
                Some("Second Conversation"),
            ),
        ];

        assert_eq!(conversations.len(), 2);
    }

    #[test]
    fn test_select_conversation_without_titles() {
        let conversations = [
            create_test_conversation("550e8400-e29b-41d4-a716-446655440002", None),
            create_test_conversation("550e8400-e29b-41d4-a716-446655440003", None),
        ];

        assert_eq!(conversations.len(), 2);
    }

    #[test]
    fn test_primary_conversations_keeps_untitled_main_chat() {
        let conversations = [create_test_conversation(
            "550e8400-e29b-41d4-a716-446655440004",
            None,
        )];

        let actual = ConversationSelector::primary_conversations(&conversations);
        let expected = 1;

        assert_eq!(actual.len(), expected);
    }

    #[test]
    fn test_primary_conversations_excludes_agent_chat() {
        let mut conversation =
            create_test_conversation("550e8400-e29b-41d4-a716-446655440005", Some("Agent"));
        conversation.initiator = forge_domain::Initiator::Agent;
        conversation.context =
            Some(Context::default().messages(vec![ContextMessage::user("Task", None).into()]));
        let conversations = [conversation];

        let actual = ConversationSelector::primary_conversations(&conversations);
        let expected = 0;

        assert_eq!(actual.len(), expected);
    }

    #[test]
    fn test_sub_conversations_keep_promoted_reused_agent_chat() {
        let parent_id = ConversationId::generate();
        let mut conversation =
            create_test_conversation("550e8400-e29b-41d4-a716-446655440007", Some("Agent"));
        conversation.ensure_delegated(Some(parent_id));
        conversation.context =
            Some(Context::default().messages(vec![ContextMessage::user("Task", None).into()]));
        let conversations = [conversation];

        let actual = ConversationSelector::conversations_with_context(&conversations);
        let expected = (1, Some(parent_id), forge_domain::Initiator::Agent);

        let actual_conversation = actual
            .first()
            .expect("one promoted agent conversation should be present");
        assert_eq!(
            (
                actual.len(),
                actual_conversation.parent_id,
                actual_conversation.initiator
            ),
            expected
        );
    }
}
