use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use forge_app::LearningService;
use forge_app::domain::{
    ConversationId, LearningLedgerEvent, LearningLedgerFreshness, LearningProvenance,
    LearningRecordProjection, LearningRepository, LearningReviewState,
};

/// Domain service for capture/query operations over the append-only learning
/// ledger.
#[derive(Clone)]
pub struct ForgeLearningService<R> {
    repository: Arc<R>,
}

impl<R> ForgeLearningService<R> {
    /// Creates a learning service from a repository implementation.
    ///
    /// # Arguments
    /// * `repository` - Learning repository dependency.
    pub fn new(repository: Arc<R>) -> Self {
        Self { repository }
    }
}

#[async_trait::async_trait]
impl<R: LearningRepository> LearningService for ForgeLearningService<R> {
    async fn capture_candidate_from_conversation(
        &self,
        conversation_id: ConversationId,
        source_event_id: String,
        summary: String,
    ) -> Result<LearningLedgerEvent> {
        let event = LearningLedgerEvent::capture_candidate(
            summary,
            LearningProvenance::conversation(
                conversation_id,
                source_event_id,
                "conversation-source-redacted",
            ),
            Utc::now(),
        )?;
        self.repository.insert_learning_event(event).await
    }

    async fn insert_learning_event(
        &self,
        event: LearningLedgerEvent,
    ) -> Result<LearningLedgerEvent> {
        self.repository.insert_learning_event(event).await
    }

    async fn list_learning_records(
        &self,
        review_state: Option<LearningReviewState>,
        limit: usize,
    ) -> Result<Vec<LearningRecordProjection>> {
        self.repository
            .list_learning_records(review_state, limit)
            .await
    }

    async fn learning_freshness(
        &self,
        review_state: Option<LearningReviewState>,
    ) -> Result<LearningLedgerFreshness> {
        self.repository.learning_freshness(review_state).await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use chrono::Utc;
    use pretty_assertions::assert_eq;

    use super::*;
    use forge_app::domain::{LearningEventId, LearningRecordId, LearningRedactionStatus};

    #[derive(Default)]
    struct FixtureLearningRepository {
        events: Mutex<HashMap<String, LearningLedgerEvent>>,
    }

    #[async_trait::async_trait]
    impl LearningRepository for FixtureLearningRepository {
        async fn insert_learning_event(
            &self,
            event: LearningLedgerEvent,
        ) -> Result<LearningLedgerEvent> {
            let mut events = self.events.lock().unwrap();
            let entry = events.entry(event.idempotency_key.clone()).or_insert(event);
            Ok(entry.clone())
        }

        async fn list_learning_records(
            &self,
            _review_state: Option<LearningReviewState>,
            _limit: usize,
        ) -> Result<Vec<LearningRecordProjection>> {
            Ok(Vec::new())
        }

        async fn learning_freshness(
            &self,
            _review_state: Option<LearningReviewState>,
        ) -> Result<LearningLedgerFreshness> {
            Ok(LearningLedgerFreshness::default())
        }
    }

    #[tokio::test]
    async fn learning_service_capture_is_idempotent_and_redacted() -> Result<()> {
        let fixture = Arc::new(FixtureLearningRepository::default());
        let service = ForgeLearningService::new(fixture);
        let conversation_id = ConversationId::generate();

        let first = service
            .capture_candidate_from_conversation(
                conversation_id,
                "event-1".to_string(),
                "token sk-123456789012345678901234 should be redacted".to_string(),
            )
            .await?;
        let second = service
            .capture_candidate_from_conversation(
                conversation_id,
                "event-1".to_string(),
                "token sk-123456789012345678901234 should be redacted".to_string(),
            )
            .await?;

        let actual = (
            first.event_id == second.event_id,
            first.summary.contains("sk-"),
            first.redaction_status,
        );
        let expected = (true, false, LearningRedactionStatus::Redacted);

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_service_forwards_explicit_events_without_state_mutation() -> Result<()> {
        let fixture = Arc::new(FixtureLearningRepository::default());
        let service = ForgeLearningService::new(fixture);
        let conversation_id = ConversationId::generate();
        let event = LearningLedgerEvent {
            event_id: LearningEventId::generate(),
            record_id: LearningRecordId::generate(),
            idempotency_key: "explicit-key".to_string(),
            event_kind: forge_app::domain::LearningEventKind::CandidateCaptured,
            summary: "typed explicit event".to_string(),
            content_fingerprint: "fingerprint".to_string(),
            redaction_status: LearningRedactionStatus::Clean,
            provenance: LearningProvenance::conversation(
                conversation_id,
                "event-2",
                "source-fingerprint-2",
            ),
            created_at: Utc::now(),
            schema_version: forge_app::domain::LEARNING_LEDGER_SCHEMA_VERSION,
        };

        let actual = service.insert_learning_event(event.clone()).await?;
        let expected = event;

        assert_eq!(actual, expected);
        Ok(())
    }
}
