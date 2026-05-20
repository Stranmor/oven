use std::sync::Arc;

use forge_domain::{
    CONVERSATION_SAVE_CAPTURE_VERSION, ContextMessage, Conversation,
    DETERMINISTIC_CONVERSATION_SAVE_AUTO_ACCEPT_REASON,
    DETERMINISTIC_CONVERSATION_SAVE_AUTO_REVIEWER_V1, LearningCaptureMetadata, LearningEventKind,
    LearningLedgerEvent, LearningProvenance, LearningRedactionStatus, LearningReviewState,
    LearningSourceKind, Role, learning_digest_hex,
};
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
        let Some(draft) = Self::candidate_draft(conversation) else {
            return;
        };
        match self
            .services
            .capture_candidate_from_conversation(
                conversation.id,
                draft.source_event_id.clone(),
                draft.summary.clone(),
                draft.metadata.clone(),
            )
            .await
        {
            Ok(event) => {
                self.auto_review_current_capture(conversation, &draft, event)
                    .await;
            }
            Err(error) => {
                tracing::debug!(error = ?error, conversation_id = %conversation.id, "Learning candidate capture failed; preserving saved conversation");
            }
        }
    }

    async fn auto_review_current_capture(
        &self,
        conversation: &Conversation,
        draft: &ConversationCaptureDraft,
        event: LearningLedgerEvent,
    ) where
        S: LearningService,
    {
        if !Self::is_current_capture_event(conversation, draft, &event) {
            return;
        }
        let candidate_record = match self.services.get_learning_record(event.record_id).await {
            Ok(Some(record)) => record,
            Ok(None) => return,
            Err(error) => {
                tracing::debug!(error = ?error, conversation_id = %conversation.id, "Learning auto-review skipped because candidate projection lookup failed");
                return;
            }
        };
        if candidate_record.review_state != LearningReviewState::Candidate {
            return;
        }
        let review_note = format!(
            "reviewer={} reason_code={}",
            DETERMINISTIC_CONVERSATION_SAVE_AUTO_REVIEWER_V1,
            DETERMINISTIC_CONVERSATION_SAVE_AUTO_ACCEPT_REASON
        );
        let review_source_event_id = format!(
            "{}:record:{}:capture:{}",
            DETERMINISTIC_CONVERSATION_SAVE_AUTO_REVIEWER_V1,
            event.record_id.into_string(),
            event.idempotency_key
        );
        let review_source_fingerprint = learning_digest_hex(format!(
            "{}:{}:{}",
            DETERMINISTIC_CONVERSATION_SAVE_AUTO_REVIEWER_V1,
            DETERMINISTIC_CONVERSATION_SAVE_AUTO_ACCEPT_REASON,
            event.content_fingerprint
        ));
        let review_event = match LearningLedgerEvent::review(
            event.record_id,
            LearningEventKind::ReviewAccepted,
            review_note,
            LearningProvenance::eval(
                DETERMINISTIC_CONVERSATION_SAVE_AUTO_REVIEWER_V1,
                review_source_event_id,
                review_source_fingerprint,
            ),
            chrono::Utc::now(),
        ) {
            Ok(event) => event,
            Err(error) => {
                tracing::debug!(error = ?error, conversation_id = %conversation.id, "Learning auto-review event construction failed");
                return;
            }
        };
        if let Err(error) = self
            .services
            .review_learning_candidate_event(review_event)
            .await
        {
            tracing::debug!(error = ?error, conversation_id = %conversation.id, "Learning auto-review append failed; candidate remains pending");
        }
    }

    fn is_current_capture_event(
        conversation: &Conversation,
        draft: &ConversationCaptureDraft,
        event: &LearningLedgerEvent,
    ) -> bool {
        let Some(metadata) = event.capture_metadata.as_ref() else {
            return false;
        };
        event.event_kind == LearningEventKind::CandidateCaptured
            && event.schema_version == forge_domain::LEARNING_LEDGER_SCHEMA_VERSION
            && event.redaction_status == LearningRedactionStatus::Clean
            && event.provenance.source_kind == LearningSourceKind::Conversation
            && event.provenance.conversation_id == Some(conversation.id)
            && event.provenance.source_event_id == draft.source_event_id
            && event.summary == draft.summary
            && event.content_fingerprint == draft.metadata.summary_fingerprint
            && metadata == &draft.metadata
            && metadata.validate_current().is_ok()
            && metadata.capture_version == CONVERSATION_SAVE_CAPTURE_VERSION
            && metadata.summary_fingerprint == learning_digest_hex(&event.summary)
            && Self::candidate_summary_from_metadata(metadata) == event.summary
    }

    fn candidate_draft(conversation: &Conversation) -> Option<ConversationCaptureDraft> {
        let context = conversation.context.as_ref()?;
        let message_count = i32::try_from(context.messages.len()).ok()?;
        if message_count == 0 {
            return None;
        }
        let user_message_count = i32::try_from(
            context
                .messages
                .iter()
                .filter(|message| match &message.message {
                    ContextMessage::Text(text) => {
                        text.role == Role::User && !text.is_internal_context()
                    }
                    _ => false,
                })
                .count(),
        )
        .ok()?;
        let context_fingerprint = Self::context_fingerprint(conversation);
        let summary = Self::candidate_summary_from_parts(
            message_count,
            user_message_count,
            &context_fingerprint,
        );
        let metadata = LearningCaptureMetadata::conversation_save(
            message_count,
            user_message_count,
            context_fingerprint,
            learning_digest_hex(&summary),
        );
        Some(ConversationCaptureDraft {
            source_event_id: Self::source_event_id(conversation, &metadata.context_fingerprint),
            summary,
            metadata,
        })
    }

    fn candidate_summary_from_metadata(metadata: &LearningCaptureMetadata) -> String {
        Self::candidate_summary_from_parts(
            metadata.message_count,
            metadata.user_message_count,
            &metadata.context_fingerprint,
        )
    }

    fn candidate_summary_from_parts(
        message_count: i32,
        user_message_count: i32,
        context_fingerprint: &str,
    ) -> String {
        format!(
            "conversation_saved message_count={} user_message_count={} context_fingerprint={}",
            message_count, user_message_count, context_fingerprint
        )
    }

    fn source_event_id(conversation: &Conversation, context_fingerprint: &str) -> String {
        format!(
            "conversation:{}:context:{}",
            conversation.id.into_string(),
            context_fingerprint
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

struct ConversationCaptureDraft {
    source_event_id: String,
    summary: String,
    metadata: LearningCaptureMetadata,
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use forge_domain::{
        Context, ContextMessage, ConversationId, LearningLedgerEvent, LearningLedgerFreshness,
        LearningRecordProjection, LearningReviewOutcome, LearningReviewState, ModelId,
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
            metadata: LearningCaptureMetadata,
        ) -> anyhow::Result<LearningLedgerEvent> {
            if self.fail_capture {
                anyhow::bail!("fixture capture failure");
            }
            let mut event = LearningLedgerEvent::capture_candidate(
                summary,
                forge_domain::LearningProvenance::conversation(
                    conversation_id,
                    source_event_id,
                    metadata.context_fingerprint.clone(),
                ),
                chrono::Utc::now(),
            )?;
            event.capture_metadata = Some(metadata);
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

        async fn review_learning_candidate_event(
            &self,
            event: LearningLedgerEvent,
        ) -> anyhow::Result<LearningReviewOutcome> {
            let mut events = self.events.lock().unwrap();
            let event = events
                .entry(event.idempotency_key.clone())
                .or_insert(event)
                .clone();
            let candidate = events
                .values()
                .find(|candidate| candidate.record_id == event.record_id)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("candidate should exist"))?;
            Ok(LearningReviewOutcome {
                event: event.clone(),
                projection: LearningRecordProjection {
                    record_id: candidate.record_id,
                    summary: candidate.summary,
                    review_state: LearningReviewState::Accepted,
                    redaction_status: candidate.redaction_status,
                    provenance: candidate.provenance,
                    capture_metadata: candidate.capture_metadata,
                    created_at: candidate.created_at,
                    updated_at: event.created_at,
                    schema_version: candidate.schema_version,
                },
            })
        }

        async fn get_learning_record(
            &self,
            record_id: forge_domain::LearningRecordId,
        ) -> anyhow::Result<Option<LearningRecordProjection>> {
            Ok(self
                .events
                .lock()
                .unwrap()
                .values()
                .find(|event| {
                    event.record_id == record_id
                        && event.event_kind == LearningEventKind::CandidateCaptured
                })
                .cloned()
                .map(|event| LearningRecordProjection {
                    record_id: event.record_id,
                    summary: event.summary,
                    review_state: LearningReviewState::Candidate,
                    redaction_status: event.redaction_status,
                    provenance: event.provenance,
                    capture_metadata: event.capture_metadata,
                    created_at: event.created_at,
                    updated_at: event.created_at,
                    schema_version: event.schema_version,
                }))
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

        let actual = fixture
            .events
            .lock()
            .unwrap()
            .values()
            .find(|event| event.event_kind == LearningEventKind::CandidateCaptured)
            .map(|event| {
                (
                    event.summary.contains("conversation_saved"),
                    event.summary.contains("user_message_count=1"),
                    event
                        .summary
                        .contains("capture a deterministic learning candidate"),
                    event.provenance.conversation_id,
                )
            });
        let expected = Some((true, true, false, Some(conversation.id)));
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
        let expected = 2usize;
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
        let expected = 2usize;
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
