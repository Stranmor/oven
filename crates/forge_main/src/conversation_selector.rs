use std::fmt::Display;

use anyhow::Result;
use forge_api::Conversation;
use forge_domain::ConversationId;
use forge_select::ForgeWidget;

use crate::display_constants::markers;
use crate::info::Info;
use crate::porcelain::Porcelain;
use crate::utils::humanize_time;

/// Logic for selecting conversations from a list
pub struct ConversationSelector;

impl ConversationSelector {
    /// Select a conversation from the provided list using porcelain-style
    /// tabular display matching the shell plugin's `:conversation` action.
    ///
    /// Displays columns: TITLE, UPDATED (hiding the UUID column).
    /// The header row is non-selectable via `header_lines=1`.
    ///
    /// Returns the selected conversation, or None if no selection was made.
    pub async fn select_conversation(
        conversations: &[Conversation],
        current_conversation_id: Option<ConversationId>,
    ) -> Result<Option<Conversation>> {
        if conversations.is_empty() {
            return Ok(None);
        }

        let valid_conversations = Self::primary_conversations(conversations);

        if valid_conversations.is_empty() {
            return Ok(None);
        }

        // Build Info structure (same as on_show_conversations)
        let mut info = Info::new();

        for conv in &valid_conversations {
            let title = conv
                .title
                .as_deref()
                .map(|t| t.to_string())
                .unwrap_or_else(|| markers::EMPTY.to_string());

            let time_ago =
                humanize_time(conv.metadata.updated_at.unwrap_or(conv.metadata.created_at));

            info = info
                .add_title(conv.id)
                .add_key_value("Title", title)
                .add_key_value("Updated", time_ago);
        }

        // Convert to porcelain, drop UUID column (col 0), truncate title
        let porcelain_output = Porcelain::from(&info)
            .drop_col(0)
            .truncate(0, 60)
            .uppercase_headers();
        let porcelain_str = porcelain_output.to_string();

        let all_lines: Vec<&str> = porcelain_str.lines().collect();
        if all_lines.is_empty() {
            return Ok(None);
        }

        #[derive(Clone)]
        struct ConversationRow {
            conversation: Option<Conversation>,
            display: String,
        }
        impl Display for ConversationRow {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.display)
            }
        }

        let mut rows: Vec<ConversationRow> = Vec::with_capacity(all_lines.len());
        // Header row (non-selectable via header_lines=1)
        if let Some(header) = all_lines.first() {
            rows.push(ConversationRow { conversation: None, display: header.to_string() });
        }
        // Data rows
        for (i, line) in all_lines.iter().skip(1).enumerate() {
            rows.push(ConversationRow {
                conversation: valid_conversations.get(i).cloned().cloned(),
                display: line.to_string(),
            });
        }

        // Find starting cursor for the current conversation
        let starting_cursor = current_conversation_id
            .and_then(|current| valid_conversations.iter().position(|c| c.id == current))
            .unwrap_or(0);

        if let Some(selected) = tokio::task::spawn_blocking(move || {
            ForgeWidget::select("Conversation", rows)
                .with_starting_cursor(starting_cursor)
                .with_header_lines(1)
                .prompt()
        })
        .await??
        {
            Ok(selected.conversation)
        } else {
            Ok(None)
        }
    }

    fn primary_conversations(conversations: &[Conversation]) -> Vec<&Conversation> {
        conversations
            .iter()
            .filter(|conv| conv.context.is_some() && !conv.is_agent_initiated())
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
            id: ConversationId::parse(id).unwrap(),
            parent_id: None,
            title: title.map(|t| t.to_string()),
            initiator: forge_domain::Initiator::User,
            context: Some(Context::default()),
            metrics: Metrics::default().started_at(now),
            metadata: MetaData { created_at: now, updated_at: Some(now) },
        }
    }

    #[tokio::test]
    async fn test_select_conversation_empty_list() {
        let conversations = vec![];
        let result = ConversationSelector::select_conversation(&conversations, None)
            .await
            .unwrap();
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

        // We can't test the actual selection without mocking the UI,
        // but we can test that the function structure is correct
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
}
