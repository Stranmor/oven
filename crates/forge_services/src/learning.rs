use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use forge_app::LearningService;
use forge_app::domain::{
    ConversationId, FakeLearningSensorReviewer, LearningCaptureMetadata,
    LearningLedgerAppendOutcome, LearningLedgerEvent, LearningLedgerFreshness, LearningProvenance,
    LearningRecordProjection, LearningRepository, LearningReviewOutcome, LearningReviewState,
    LearningSensorReviewInput, RedactedLearningSummary, SensorLessonPromotionOutcome,
    SensorLessonPromotionRequest,
};
use forge_domain::LearningSensorReviewer;

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

impl<R: LearningRepository> ForgeLearningService<R> {
    /// Runs the deterministic fake Sensor reviewer over sanitized candidate evidence.
    ///
    /// # Arguments
    /// * `input` - Sanitized Sensor review input.
    ///
    /// # Errors
    /// Returns an error when validation, stale-state checks, or persistence fail.
    pub async fn review_with_fake_sensor(
        &self,
        input: LearningSensorReviewInput,
    ) -> Result<LearningLedgerAppendOutcome> {
        let output = FakeLearningSensorReviewer.review(input.clone())?;
        self.append_learning_sensor_review(input, output).await
    }
}

#[async_trait::async_trait]
impl<R: LearningRepository> LearningService for ForgeLearningService<R> {
    async fn capture_candidate_from_conversation(
        &self,
        conversation_id: ConversationId,
        source_event_id: String,
        summary: String,
        metadata: LearningCaptureMetadata,
    ) -> Result<LearningLedgerAppendOutcome> {
        metadata.validate_current()?;
        let redacted = RedactedLearningSummary::from_raw(&summary);
        let mut event = LearningLedgerEvent::capture_candidate(
            summary,
            LearningProvenance::conversation(
                conversation_id,
                source_event_id,
                metadata.context_fingerprint.clone(),
            ),
            Utc::now(),
        )?;
        event.capture_metadata = Some(metadata);
        event.content_fingerprint = redacted.fingerprint;
        self.repository.insert_learning_event(event).await
    }

    async fn insert_learning_event(
        &self,
        event: LearningLedgerEvent,
    ) -> Result<LearningLedgerAppendOutcome> {
        self.repository.insert_learning_event(event).await
    }

    async fn review_learning_candidate_event(
        &self,
        event: LearningLedgerEvent,
    ) -> Result<LearningReviewOutcome> {
        self.repository.review_learning_candidate_event(event).await
    }

    async fn promote_sensor_lesson(
        &self,
        request: SensorLessonPromotionRequest,
    ) -> Result<SensorLessonPromotionOutcome> {
        self.repository.promote_sensor_lesson(request).await
    }

    async fn get_learning_record(
        &self,
        record_id: forge_app::domain::LearningRecordId,
    ) -> Result<Option<LearningRecordProjection>> {
        self.repository.get_learning_record(record_id).await
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
    use forge_app::domain::{
        LEARNING_SENSOR_REVIEW_SCHEMA_VERSION, LearningEventId, LearningEventKind,
        LearningRecordId, LearningRedactionStatus, LearningSensorDecisionKind,
        LearningSensorReviewOutput,
    };

    #[derive(Default)]
    struct FixtureLearningRepository {
        events: Mutex<HashMap<String, LearningLedgerEvent>>,
    }

    #[async_trait::async_trait]
    impl LearningRepository for FixtureLearningRepository {
        async fn insert_learning_event(
            &self,
            event: LearningLedgerEvent,
        ) -> Result<LearningLedgerAppendOutcome> {
            let mut events = self.events.lock().unwrap();
            let key = event.idempotency_key.clone();
            if let Some(existing) = events.get(&key) {
                return Ok(LearningLedgerAppendOutcome::existing(existing.clone()));
            }
            events.insert(key, event.clone());
            Ok(LearningLedgerAppendOutcome::inserted(event))
        }

        async fn review_learning_candidate_event(
            &self,
            event: LearningLedgerEvent,
        ) -> Result<LearningReviewOutcome> {
            let outcome = self.insert_learning_event(event).await?;
            let event = outcome.event;
            Ok(LearningReviewOutcome {
                projection: LearningRecordProjection {
                    record_id: event.record_id,
                    summary: event.summary.clone(),
                    accepted_summary: None,
                    review_state: LearningReviewState::Accepted,
                    redaction_status: event.redaction_status,
                    provenance: event.provenance.clone(),
                    capture_metadata: event.capture_metadata.clone(),
                    created_at: event.created_at,
                    updated_at: event.created_at,
                    schema_version: event.schema_version,
                },
                event,
            })
        }

        async fn get_learning_event_view(
            &self,
            _event_id: LearningEventId,
        ) -> Result<Option<forge_app::domain::LearningLedgerEventView>> {
            anyhow::bail!("unused learning event view")
        }

        async fn promote_sensor_lesson(
            &self,
            _request: SensorLessonPromotionRequest,
        ) -> Result<SensorLessonPromotionOutcome> {
            anyhow::bail!("unused promotion")
        }

        async fn get_learning_record(
            &self,
            record_id: LearningRecordId,
        ) -> Result<Option<LearningRecordProjection>> {
            Ok(self
                .events
                .lock()
                .unwrap()
                .values()
                .find(|event| event.record_id == record_id)
                .map(|event| LearningRecordProjection {
                    record_id: event.record_id,
                    summary: event.summary.clone(),
                    accepted_summary: None,
                    review_state: LearningReviewState::Candidate,
                    redaction_status: event.redaction_status,
                    provenance: event.provenance.clone(),
                    capture_metadata: event.capture_metadata.clone(),
                    created_at: event.created_at,
                    updated_at: event.created_at,
                    schema_version: event.schema_version,
                }))
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
                LearningCaptureMetadata::conversation_save(
                    1,
                    1,
                    "context-fingerprint",
                    "summary-fingerprint",
                ),
            )
            .await?;
        let second = service
            .capture_candidate_from_conversation(
                conversation_id,
                "event-1".to_string(),
                "token sk-123456789012345678901234 should be redacted".to_string(),
                LearningCaptureMetadata::conversation_save(
                    1,
                    1,
                    "context-fingerprint",
                    "summary-fingerprint",
                ),
            )
            .await?;

        let actual = (
            first.event.event_id == second.event.event_id,
            first.event.summary.contains("sk-"),
            first.event.redaction_status,
            first.freshness,
            second.freshness,
        );
        let expected = (
            true,
            false,
            LearningRedactionStatus::Redacted,
            forge_app::domain::LearningLedgerEventFreshness::Inserted,
            forge_app::domain::LearningLedgerEventFreshness::Existing,
        );

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_service_fake_sensor_metadata_only_appends_pending_not_proposal() -> Result<()>
    {
        let fixture = Arc::new(FixtureLearningRepository::default());
        let service = ForgeLearningService::new(fixture);
        let conversation_id = ConversationId::generate();
        let candidate = service
            .capture_candidate_from_conversation(
                conversation_id,
                "event-sensor-1".to_string(),
                "conversation_saved message_count=1 user_message_count=1 context_fingerprint=abc"
                    .to_string(),
                LearningCaptureMetadata::conversation_save(1, 1, "abc", "summary-fingerprint"),
            )
            .await?
            .event;
        let projection = service
            .get_learning_record(candidate.record_id)
            .await?
            .expect("candidate projection should exist");
        let input = LearningSensorReviewInput::from_candidate_projection(&projection);

        let actual = service
            .review_with_fake_sensor(input)
            .await?
            .event
            .event_kind;
        let expected = LearningEventKind::SensorReviewPending;

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_service_fake_sensor_fixture_appends_proposal_but_not_accepted() -> Result<()>
    {
        let fixture = Arc::new(FixtureLearningRepository::default());
        let service = ForgeLearningService::new(fixture);
        let conversation_id = ConversationId::generate();
        let candidate = service
            .capture_candidate_from_conversation(
                conversation_id,
                "event-sensor-2".to_string(),
                "conversation_saved message_count=1 user_message_count=1 context_fingerprint=def"
                    .to_string(),
                LearningCaptureMetadata::conversation_save(1, 1, "def", "summary-fingerprint"),
            )
            .await?
            .event;
        let projection = service
            .get_learning_record(candidate.record_id)
            .await?
            .expect("candidate projection should exist");
        let input = LearningSensorReviewInput::fake_fixture(
            &projection,
            "Durable typed observation",
            "A recurring typed fixture observation exists",
        );

        let outcome = service.review_with_fake_sensor(input).await?;
        let record = service
            .get_learning_record(candidate.record_id)
            .await?
            .expect("candidate projection should remain present");
        let actual = (outcome.event.event_kind, record.review_state);
        let expected = (
            LearningEventKind::SensorLessonProposed,
            LearningReviewState::Candidate,
        );

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_service_rejects_mismatched_sensor_output_without_accepting() -> Result<()> {
        let fixture = Arc::new(FixtureLearningRepository::default());
        let service = ForgeLearningService::new(fixture);
        let conversation_id = ConversationId::generate();
        let candidate = service
            .capture_candidate_from_conversation(
                conversation_id,
                "event-sensor-3".to_string(),
                "conversation_saved message_count=1 user_message_count=1 context_fingerprint=ghi"
                    .to_string(),
                LearningCaptureMetadata::conversation_save(1, 1, "ghi", "summary-fingerprint"),
            )
            .await?
            .event;
        let projection = service
            .get_learning_record(candidate.record_id)
            .await?
            .expect("candidate projection should exist");
        let input = LearningSensorReviewInput::fake_fixture(
            &projection,
            "Durable typed observation",
            "A recurring typed fixture observation exists",
        );
        let output = LearningSensorReviewOutput {
            schema_version: LEARNING_SENSOR_REVIEW_SCHEMA_VERSION,
            reviewer_id: "wrong".to_string(),
            reviewer_version: 1,
            input_fingerprint: input.fingerprint()?,
            decision: LearningSensorDecisionKind::ProposeLesson,
            reason_code: "typed_fixture_substantive_evidence".to_string(),
            proposal_title: Some("Durable typed observation".to_string()),
            proposal_body: Some("A recurring typed fixture observation exists".to_string()),
        };

        let rejected = service
            .append_learning_sensor_review(input, output)
            .await
            .is_err();
        let record = service
            .get_learning_record(candidate.record_id)
            .await?
            .expect("candidate projection should remain present");
        let actual = (rejected, record.review_state);
        let expected = (true, LearningReviewState::Candidate);

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_service_rejects_stale_sensor_projection_hash_without_append() -> Result<()> {
        let fixture = Arc::new(FixtureLearningRepository::default());
        let service = ForgeLearningService::new(fixture);
        let conversation_id = ConversationId::generate();
        let candidate = service
            .capture_candidate_from_conversation(
                conversation_id,
                "event-sensor-4".to_string(),
                "conversation_saved message_count=1 user_message_count=1 context_fingerprint=jkl"
                    .to_string(),
                LearningCaptureMetadata::conversation_save(1, 1, "jkl", "summary-fingerprint"),
            )
            .await?
            .event;
        let projection = service
            .get_learning_record(candidate.record_id)
            .await?
            .expect("candidate projection should exist");
        let mut input = LearningSensorReviewInput::fake_fixture(
            &projection,
            "Durable typed observation",
            "A recurring typed fixture observation exists",
        );
        input.sanitized_projection_hash = "stale".to_string();

        let actual = service.review_with_fake_sensor(input).await.is_err();
        let expected = true;

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
            capture_metadata: None,
            created_at: Utc::now(),
            schema_version: forge_app::domain::LEARNING_LEDGER_SCHEMA_VERSION,
        };

        let actual = service.insert_learning_event(event.clone()).await?;
        let expected = LearningLedgerAppendOutcome::inserted(event);

        assert_eq!(actual, expected);
        Ok(())
    }
}
