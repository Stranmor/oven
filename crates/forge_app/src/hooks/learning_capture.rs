use std::sync::Arc;

use forge_domain::{ContextMessage, Conversation, Role};
use sha2::{Digest, Sha256};

use crate::LearningService;

/// Captures deterministic conversation-derived self-learning candidates after
/// successful conversation persistence.
#[derive(Clone)]
pub struct LearningCapture<S> {
    services: Arc<S>,
}

impl<S> LearningCapture<S> {
    /// Creates a learning capture helper with the provided service container.
    ///
    /// # Arguments
    /// * `services` - Service container exposing the learning service.
    pub fn new(services: Arc<S>) -> Self {
        Self { services }
    }

    /// Captures a deterministic candidate for a durably saved conversation.
    ///
    /// # Arguments
    /// * `conversation` - Conversation snapshot already persisted by the caller.
    pub async fn capture_saved_conversation(&self, conversation: &Conversation)
    where
        S: LearningService,
    {
        let Some(summary) = Self::candidate_summary(conversation) else {
            return;
        };
        let source_event_id = Self::source_event_id(conversation);
        if let Err(error) = self
            .services
            .capture_candidate_from_conversation(conversation.id, source_event_id, summary)
            .await
        {
            tracing::debug!(error = ?error, conversation_id = %conversation.id, "Learning candidate capture failed; preserving saved conversation");
        }
    }

    fn candidate_summary(conversation: &Conversation) -> Option<String> {
        let context = conversation.context.as_ref()?;
        let message_count = context.messages.len();
        if message_count == 0 {
            return None;
        }
        let first_user = context
            .messages
            .iter()
            .find_map(|message| match &message.message {
                ContextMessage::Text(text)
                    if text.role == Role::User && !text.is_internal_context() =>
                {
                    Some(text.content.as_str())
                }
                _ => None,
            })
            .unwrap_or("no user message");
        let preview = first_user
            .split_whitespace()
            .take(24)
            .collect::<Vec<_>>()
            .join(" ");
        Some(format!(
            "conversation_saved message_count={} context_fingerprint={} first_user_preview={}",
            message_count,
            Self::context_fingerprint(conversation),
            preview
        ))
    }

    fn source_event_id(conversation: &Conversation) -> String {
        format!(
            "conversation:{}:context:{}",
            conversation.id.into_string(),
            Self::context_fingerprint(conversation)
        )
    }

    fn context_fingerprint(conversation: &Conversation) -> String {
        let mut hasher = Sha256::new();
        hasher.update(conversation.id.into_string());
        if let Some(context) = &conversation.context {
            for message in &context.messages {
                if let ContextMessage::Text(text) = &message.message {
                    hasher.update(text.role.to_string());
                    hasher.update([0]);
                    if let Some(kind) = &text.kind {
                        hasher.update(format!("{kind:?}"));
                    }
                    hasher.update([0]);
                    hasher.update(text.content.as_bytes());
                    hasher.update([0]);
                }
            }
        }
        hex::encode(hasher.finalize())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use forge_domain::{
        Context, ContextMessage, ConversationId, LearningLedgerEvent, LearningLedgerFreshness,
        LearningRecordProjection, LearningReviewState, ModelId,
    };
    use pretty_assertions::assert_eq;

    use super::*;

    #[derive(Default)]
    struct FixtureLearningService {
        events: Mutex<HashMap<String, LearningLedgerEvent>>,
        fail_capture: bool,
    }

    #[async_trait::async_trait]
    impl LearningService for FixtureLearningService {
        async fn capture_candidate_from_conversation(
            &self,
            conversation_id: ConversationId,
            source_event_id: String,
            summary: String,
        ) -> anyhow::Result<LearningLedgerEvent> {
            if self.fail_capture {
                anyhow::bail!("fixture capture failure");
            }
            let event = LearningLedgerEvent::capture_candidate(
                summary,
                forge_domain::LearningProvenance::conversation(
                    conversation_id,
                    source_event_id,
                    "fixture-source-fingerprint",
                ),
                chrono::Utc::now(),
            )?;
            let mut events = self.events.lock().unwrap();
            let entry = events.entry(event.idempotency_key.clone()).or_insert(event);
            Ok(entry.clone())
        }

        async fn insert_learning_event(
            &self,
            _event: LearningLedgerEvent,
        ) -> anyhow::Result<LearningLedgerEvent> {
            anyhow::bail!("unused insert")
        }

        async fn list_learning_records(
            &self,
            _review_state: Option<LearningReviewState>,
            _limit: usize,
        ) -> anyhow::Result<Vec<LearningRecordProjection>> {
            anyhow::bail!("unused list")
        }

        async fn learning_freshness(
            &self,
            _review_state: Option<LearningReviewState>,
        ) -> anyhow::Result<LearningLedgerFreshness> {
            anyhow::bail!("unused freshness")
        }
    }

    fn fixture_conversation() -> Conversation {
        let model_id = ModelId::new("test-model");
        Conversation::generate().context(Context::default().add_message(ContextMessage::user(
            "capture a deterministic learning candidate",
            Some(model_id),
        )))
    }

    #[tokio::test]
    async fn learning_capture_captures_candidate_from_saved_conversation() -> anyhow::Result<()> {
        let fixture = Arc::new(FixtureLearningService::default());
        let capture = LearningCapture::new(fixture.clone());
        let conversation = fixture_conversation();

        capture.capture_saved_conversation(&conversation).await;

        let actual = fixture.events.lock().unwrap().values().next().map(|event| {
            (
                event.summary.contains("conversation_saved"),
                event.provenance.conversation_id,
            )
        });
        let expected = Some((true, Some(conversation.id)));
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_capture_is_idempotent_for_duplicate_saved_conversation() -> anyhow::Result<()>
    {
        let fixture = Arc::new(FixtureLearningService::default());
        let capture = LearningCapture::new(fixture.clone());
        let conversation = fixture_conversation();

        capture.capture_saved_conversation(&conversation).await;
        capture.capture_saved_conversation(&conversation).await;

        let actual = fixture.events.lock().unwrap().len();
        let expected = 1usize;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_capture_source_event_id_is_stable_across_reload_metadata()
    -> anyhow::Result<()> {
        let fixture = Arc::new(FixtureLearningService::default());
        let capture = LearningCapture::new(fixture.clone());
        let conversation = fixture_conversation();
        let mut reloaded = conversation.clone();
        reloaded.metadata.updated_at = Some(chrono::Utc::now() + chrono::Duration::seconds(60));

        capture.capture_saved_conversation(&conversation).await;
        capture.capture_saved_conversation(&reloaded).await;

        let actual = fixture.events.lock().unwrap().len();
        let expected = 1usize;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_capture_failure_does_not_fail_saved_conversation_path() -> anyhow::Result<()>
    {
        let fixture = Arc::new(FixtureLearningService {
            events: Mutex::new(HashMap::new()),
            fail_capture: true,
        });
        let capture = LearningCapture::new(fixture.clone());
        let conversation = fixture_conversation();

        capture.capture_saved_conversation(&conversation).await;

        assert_eq!(fixture.events.lock().unwrap().len(), 0usize);
        Ok(())
    }
}
