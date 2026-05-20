use std::collections::BTreeMap;
use std::str::FromStr;
use std::sync::Arc;

use chrono::{DateTime, NaiveDateTime, Utc};
use diesel::prelude::*;
use forge_domain::{
    AcceptedLearningSummary, ConversationId, LearningCaptureMetadata, LearningEventId,
    LearningEventKind, LearningEventSeq, LearningLedgerAppendOutcome, LearningLedgerCursor,
    LearningLedgerEvent, LearningLedgerEventView, LearningLedgerFreshness, LearningProvenance,
    LearningRecordId, LearningRecordProjection, LearningRedactionStatus, LearningRepository,
    LearningReviewOutcome, LearningReviewState, LearningSourceKind, RedactedLearningSummary,
    SANCTIONED_SANITIZED_OBSERVATION_PROMOTION_REASON,
    SANCTIONED_SANITIZED_OBSERVATION_PROMOTION_REVIEWER_ID, SensorLessonPromotionOutcome,
    SensorLessonPromotionProposal, SensorLessonPromotionRequest, SubagentTaskId, WorkspaceHash,
};
use sha2::{Digest, Sha256};

use crate::database::schema::learning_ledger_events;
use crate::database::{DatabasePool, PooledSqliteConnection};

pub struct LearningRepositoryImpl {
    pool: Arc<DatabasePool>,
    wid: WorkspaceHash,
}

#[derive(Debug, Clone, Queryable, QueryableByName, Selectable)]
#[diesel(table_name = crate::database::schema::learning_ledger_events)]
struct LearningLedgerEventRecord {
    #[diesel(sql_type = diesel::sql_types::BigInt)]
    pub event_seq: i64,
    #[diesel(sql_type = diesel::sql_types::Text)]
    pub event_id: String,
    #[diesel(sql_type = diesel::sql_types::Text)]
    pub record_id: String,
    #[diesel(sql_type = diesel::sql_types::Text)]
    pub idempotency_key: String,
    #[diesel(sql_type = diesel::sql_types::BigInt)]
    pub workspace_id: i64,
    #[diesel(sql_type = diesel::sql_types::Text)]
    pub event_kind: String,
    #[diesel(sql_type = diesel::sql_types::Text)]
    pub summary: String,
    #[diesel(sql_type = diesel::sql_types::Text)]
    pub content_fingerprint: String,
    #[diesel(sql_type = diesel::sql_types::Text)]
    pub redaction_status: String,
    #[diesel(sql_type = diesel::sql_types::Text)]
    pub source_kind: String,
    #[diesel(sql_type = diesel::sql_types::Text)]
    pub source_id: String,
    #[diesel(sql_type = diesel::sql_types::Text)]
    pub source_event_id: String,
    #[diesel(sql_type = diesel::sql_types::Text)]
    pub source_fingerprint: String,
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Text>)]
    pub conversation_id: Option<String>,
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Text>)]
    pub task_id: Option<String>,
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Text>)]
    pub tool_name: Option<String>,
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Text>)]
    pub eval_id: Option<String>,
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Text>)]
    pub capture_metadata: Option<String>,
    #[diesel(sql_type = diesel::sql_types::Timestamp)]
    pub created_at: NaiveDateTime,
    #[diesel(sql_type = diesel::sql_types::Integer)]
    pub schema_version: i32,
}

impl LearningRepositoryImpl {
    pub fn new(pool: Arc<DatabasePool>, workspace_id: WorkspaceHash) -> Self {
        Self { pool, wid: workspace_id }
    }

    async fn run_with_connection<F, T>(&self, operation: F) -> anyhow::Result<T>
    where
        F: FnOnce(&mut PooledSqliteConnection, WorkspaceHash) -> anyhow::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let pool = self.pool.clone();
        let wid = self.wid;
        tokio::task::spawn_blocking(move || {
            let mut connection = pool.get_connection()?;
            operation(&mut connection, wid)
        })
        .await
        .map_err(|error| anyhow::anyhow!("Learning repository task failed: {error}"))?
    }
}

#[async_trait::async_trait]
impl LearningRepository for LearningRepositoryImpl {
    async fn insert_learning_event(
        &self,
        event: LearningLedgerEvent,
    ) -> anyhow::Result<LearningLedgerAppendOutcome> {
        self.run_with_connection(move |connection, wid| {
            event.provenance.validate()?;
            let workspace_id = workspace_db_id(wid);
            let record = LearningLedgerEventRecord::new(event, workspace_id)?;
            connection.immediate_transaction::<_, anyhow::Error, _>(|connection| {
                let existing = learning_ledger_events::table
                    .filter(learning_ledger_events::workspace_id.eq(workspace_id))
                    .filter(learning_ledger_events::idempotency_key.eq(&record.idempotency_key))
                    .first::<LearningLedgerEventRecord>(connection)
                    .optional()?;
                if let Some(existing) = existing {
                    ensure_learning_event_idempotency_replay(&existing, &record)?;
                    return existing
                        .try_into_event()
                        .map(LearningLedgerAppendOutcome::existing);
                }
                diesel::insert_into(learning_ledger_events::table)
                    .values((
                        learning_ledger_events::event_id.eq(&record.event_id),
                        learning_ledger_events::record_id.eq(&record.record_id),
                        learning_ledger_events::idempotency_key.eq(&record.idempotency_key),
                        learning_ledger_events::workspace_id.eq(record.workspace_id),
                        learning_ledger_events::event_kind.eq(&record.event_kind),
                        learning_ledger_events::summary.eq(&record.summary),
                        learning_ledger_events::content_fingerprint.eq(&record.content_fingerprint),
                        learning_ledger_events::redaction_status.eq(&record.redaction_status),
                        learning_ledger_events::source_kind.eq(&record.source_kind),
                        learning_ledger_events::source_id.eq(&record.source_id),
                        learning_ledger_events::source_event_id.eq(&record.source_event_id),
                        learning_ledger_events::source_fingerprint.eq(&record.source_fingerprint),
                        learning_ledger_events::conversation_id.eq(&record.conversation_id),
                        learning_ledger_events::task_id.eq(&record.task_id),
                        learning_ledger_events::tool_name.eq(&record.tool_name),
                        learning_ledger_events::eval_id.eq(&record.eval_id),
                        learning_ledger_events::capture_metadata.eq(&record.capture_metadata),
                        learning_ledger_events::created_at.eq(record.created_at),
                        learning_ledger_events::schema_version.eq(record.schema_version),
                    ))
                    .execute(connection)?;
                learning_ledger_events::table
                    .filter(learning_ledger_events::workspace_id.eq(workspace_id))
                    .filter(learning_ledger_events::idempotency_key.eq(&record.idempotency_key))
                    .first::<LearningLedgerEventRecord>(connection)?
                    .try_into_event()
                    .map(LearningLedgerAppendOutcome::inserted)
            })
        })
        .await
    }

    async fn review_learning_candidate_event(
        &self,
        event: LearningLedgerEvent,
    ) -> anyhow::Result<LearningReviewOutcome> {
        self.run_with_connection(move |connection, wid| {
            event.provenance.validate()?;
            let target_state = review_target_state(event.event_kind)?;
            let workspace_id = workspace_db_id(wid);
            let record = LearningLedgerEventRecord::new(event, workspace_id)?;
            connection.immediate_transaction::<_, anyhow::Error, _>(|connection| {
                let records = learning_ledger_events::table
                    .filter(learning_ledger_events::workspace_id.eq(workspace_id))
                    .order(learning_ledger_events::event_seq.asc())
                    .load::<LearningLedgerEventRecord>(connection)?;
                let projection = project_records(records.clone())?
                    .into_iter()
                    .find(|projection| projection.record_id.into_string() == record.record_id)
                    .ok_or_else(|| anyhow::anyhow!("learning candidate record not found"))?;
                if projection.review_state == target_state {
                    let existing = learning_ledger_events::table
                        .filter(learning_ledger_events::workspace_id.eq(workspace_id))
                        .filter(learning_ledger_events::record_id.eq(&record.record_id))
                        .filter(learning_ledger_events::event_kind.eq(&record.event_kind))
                        .order(learning_ledger_events::event_seq.desc())
                        .first::<LearningLedgerEventRecord>(connection)?;
                    ensure_learning_event_idempotency_replay(&existing, &record)?;
                    let event = existing.try_into_event()?;
                    return Ok(LearningReviewOutcome { event, projection });
                }
                if projection.review_state != LearningReviewState::Candidate {
                    anyhow::bail!(
                        "learning record cannot be reviewed from state {}",
                        projection.review_state
                    );
                }
                if target_state == LearningReviewState::Accepted
                    && records.iter().any(|existing| {
                        existing.record_id == record.record_id
                            && existing.event_kind
                                == LearningEventKind::SensorLessonProposed.to_string()
                    })
                {
                    anyhow::bail!(
                        "sensor-derived learning candidates require promotion proof before acceptance"
                    );
                }
                let existing = learning_ledger_events::table
                    .filter(learning_ledger_events::workspace_id.eq(workspace_id))
                    .filter(learning_ledger_events::idempotency_key.eq(&record.idempotency_key))
                    .first::<LearningLedgerEventRecord>(connection)
                    .optional()?;
                let event = if let Some(existing) = existing {
                    ensure_learning_event_idempotency_replay(&existing, &record)?;
                    anyhow::ensure!(
                        projection.review_state == target_state,
                        "learning review idempotency replay is stale for current projection state {}",
                        projection.review_state
                    );
                    existing.try_into_event()?
                } else {
                    diesel::insert_into(learning_ledger_events::table)
                        .values((
                            learning_ledger_events::event_id.eq(&record.event_id),
                            learning_ledger_events::record_id.eq(&record.record_id),
                            learning_ledger_events::idempotency_key.eq(&record.idempotency_key),
                            learning_ledger_events::workspace_id.eq(record.workspace_id),
                            learning_ledger_events::event_kind.eq(&record.event_kind),
                            learning_ledger_events::summary.eq(&record.summary),
                            learning_ledger_events::content_fingerprint
                                .eq(&record.content_fingerprint),
                            learning_ledger_events::redaction_status.eq(&record.redaction_status),
                            learning_ledger_events::source_kind.eq(&record.source_kind),
                            learning_ledger_events::source_id.eq(&record.source_id),
                            learning_ledger_events::source_event_id.eq(&record.source_event_id),
                            learning_ledger_events::source_fingerprint
                                .eq(&record.source_fingerprint),
                            learning_ledger_events::conversation_id.eq(&record.conversation_id),
                            learning_ledger_events::task_id.eq(&record.task_id),
                            learning_ledger_events::tool_name.eq(&record.tool_name),
                            learning_ledger_events::eval_id.eq(&record.eval_id),
                            learning_ledger_events::capture_metadata.eq(&record.capture_metadata),
                            learning_ledger_events::created_at.eq(record.created_at),
                            learning_ledger_events::schema_version.eq(record.schema_version),
                        ))
                        .execute(connection)?;
                    learning_ledger_events::table
                        .filter(learning_ledger_events::workspace_id.eq(workspace_id))
                        .filter(learning_ledger_events::idempotency_key.eq(&record.idempotency_key))
                        .first::<LearningLedgerEventRecord>(connection)?
                        .try_into_event()?
                };
                let records = learning_ledger_events::table
                    .filter(learning_ledger_events::workspace_id.eq(workspace_id))
                    .order(learning_ledger_events::event_seq.asc())
                    .load::<LearningLedgerEventRecord>(connection)?;
                let projection = project_records(records.clone())?
                    .into_iter()
                    .find(|projection| projection.record_id.into_string() == record.record_id)
                    .ok_or_else(|| {
                        anyhow::anyhow!("learning review projection not found after append")
                    })?;
                anyhow::ensure!(
                    projection.review_state == target_state,
                    "learning review did not transition projection to target state {}",
                    target_state
                );
                Ok(LearningReviewOutcome { event, projection })
            })
        })
        .await
    }

    async fn get_learning_event_view(
        &self,
        event_id: LearningEventId,
    ) -> anyhow::Result<Option<LearningLedgerEventView>> {
        self.run_with_connection(move |connection, wid| {
            let workspace_id = workspace_db_id(wid);
            let ledger_cursor = learning_ledger_events::table
                .filter(learning_ledger_events::workspace_id.eq(workspace_id))
                .select(diesel::dsl::max(learning_ledger_events::event_seq))
                .first::<Option<i64>>(connection)?
                .unwrap_or(0);
            let Some(record) = learning_ledger_events::table
                .filter(learning_ledger_events::workspace_id.eq(workspace_id))
                .filter(learning_ledger_events::event_id.eq(event_id.into_string()))
                .first::<LearningLedgerEventRecord>(connection)
                .optional()?
            else {
                return Ok(None);
            };
            Ok(Some(LearningLedgerEventView {
                event_seq: LearningEventSeq::new(record.event_seq)?,
                ledger_cursor: LearningLedgerCursor::new(ledger_cursor)?,
                event: record.try_into_event()?,
            }))
        })
        .await
    }

    async fn promote_sensor_lesson(
        &self,
        request: SensorLessonPromotionRequest,
    ) -> anyhow::Result<SensorLessonPromotionOutcome> {
        self.run_with_connection(move |connection, wid| {
            request.audit_event().provenance.validate()?;
            request.review_event().provenance.validate()?;
            let workspace_id = workspace_db_id(wid);
            let audit_record =
                LearningLedgerEventRecord::new(request.audit_event().clone(), workspace_id)?;
            let review_record =
                LearningLedgerEventRecord::new(request.review_event().clone(), workspace_id)?;
            connection.immediate_transaction::<_, anyhow::Error, _>(|connection| {
                let records = learning_ledger_events::table
                    .filter(learning_ledger_events::workspace_id.eq(workspace_id))
                    .order(learning_ledger_events::event_seq.asc())
                    .load::<LearningLedgerEventRecord>(connection)?;
                let ledger_cursor = records
                    .iter()
                    .map(|record| record.event_seq)
                    .max()
                    .unwrap_or(0);
                let projection_before = project_records(records.clone())?
                    .into_iter()
                    .find(|projection| projection.record_id == request.proposal().candidate_id());
                if let Some(projection) = projection_before.as_ref()
                    && projection.review_state == LearningReviewState::Accepted
                {
                    let audit_event = learning_ledger_events::table
                        .filter(learning_ledger_events::workspace_id.eq(workspace_id))
                        .filter(
                            learning_ledger_events::idempotency_key
                                .eq(&audit_record.idempotency_key),
                        )
                        .first::<LearningLedgerEventRecord>(connection)?;
                    let review_event = learning_ledger_events::table
                        .filter(learning_ledger_events::workspace_id.eq(workspace_id))
                        .filter(
                            learning_ledger_events::idempotency_key
                                .eq(&review_record.idempotency_key),
                        )
                        .first::<LearningLedgerEventRecord>(connection)?;
                    ensure_learning_event_idempotency_replay(&audit_event, &audit_record)?;
                    ensure_learning_event_idempotency_replay(&review_event, &review_record)?;
                    anyhow::ensure!(
                        projection.accepted_summary.as_deref()
                            == Some(request.proposal().accepted_summary()),
                        "sensor promotion replay accepted summary mismatch"
                    );
                    return Ok(SensorLessonPromotionOutcome {
                        audit_event: audit_event.try_into_event()?,
                        review_event: review_event.try_into_event()?,
                        projection: projection.clone(),
                    });
                }
                anyhow::ensure!(
                    ledger_cursor == request.proposal().observed_ledger_cursor().get(),
                    "sensor promotion ledger cursor changed after proof construction"
                );
                let proposal_record = records
                    .iter()
                    .find(|record| {
                        record.event_seq == request.proposal().proposal_event_seq().get()
                            && record.record_id == request.proposal().candidate_id().into_string()
                            && record.event_kind
                                == LearningEventKind::SensorLessonProposed.to_string()
                    })
                    .ok_or_else(|| anyhow::anyhow!("sensor promotion proposal event not found"))?;
                let projection = project_records(records.clone())?
                    .into_iter()
                    .find(|projection| projection.record_id == request.proposal().candidate_id())
                    .ok_or_else(|| anyhow::anyhow!("sensor promotion candidate not found"))?;
                anyhow::ensure!(
                    projection.review_state == LearningReviewState::Candidate,
                    "sensor promotion candidate is no longer reviewable: {}",
                    projection.review_state
                );
                anyhow::ensure!(
                    forge_domain::learning_projection_hash(&projection)
                        == request.proposal().projection_hash(),
                    "sensor promotion candidate projection hash mismatch"
                );
                let event_view = LearningLedgerEventView {
                    event_seq: LearningEventSeq::new(proposal_record.event_seq)?,
                    ledger_cursor: LearningLedgerCursor::new(ledger_cursor)?,
                    event: proposal_record.clone().try_into_event()?,
                };
                SensorLessonPromotionProposal::new(&event_view, &projection)?;

                let audit_event = insert_or_replay_learning_event(connection, &audit_record)?;
                let review_event = insert_or_replay_learning_event(connection, &review_record)?;

                let records = learning_ledger_events::table
                    .filter(learning_ledger_events::workspace_id.eq(workspace_id))
                    .order(learning_ledger_events::event_seq.asc())
                    .load::<LearningLedgerEventRecord>(connection)?;
                let projection = project_records(records.clone())?
                    .into_iter()
                    .find(|projection| projection.record_id == request.proposal().candidate_id())
                    .ok_or_else(|| anyhow::anyhow!("sensor promotion projection missing"))?;
                anyhow::ensure!(
                    projection.review_state == LearningReviewState::Accepted,
                    "sensor promotion did not accept projection"
                );
                anyhow::ensure!(
                    projection.accepted_summary.as_deref()
                        == Some(request.proposal().accepted_summary()),
                    "sensor promotion accepted summary projection mismatch"
                );
                Ok(SensorLessonPromotionOutcome { audit_event, review_event, projection })
            })
        })
        .await
    }

    async fn get_learning_record(
        &self,
        record_id: LearningRecordId,
    ) -> anyhow::Result<Option<LearningRecordProjection>> {
        self.run_with_connection(move |connection, wid| {
            let workspace_id = workspace_db_id(wid);
            let records = learning_ledger_events::table
                .filter(learning_ledger_events::workspace_id.eq(workspace_id))
                .order(learning_ledger_events::event_seq.asc())
                .load::<LearningLedgerEventRecord>(connection)?;
            Ok(project_records(records.clone())?
                .into_iter()
                .find(|projection| projection.record_id == record_id))
        })
        .await
    }

    async fn list_learning_records(
        &self,
        review_state: Option<LearningReviewState>,
        limit: usize,
    ) -> anyhow::Result<Vec<LearningRecordProjection>> {
        self.run_with_connection(move |connection, wid| {
            let workspace_id = workspace_db_id(wid);
            let records = learning_ledger_events::table
                .filter(learning_ledger_events::workspace_id.eq(workspace_id))
                .order(learning_ledger_events::event_seq.asc())
                .load::<LearningLedgerEventRecord>(connection)?;
            let mut projections = project_records(records.clone())?;
            if let Some(review_state) = review_state {
                projections.retain(|projection| projection.review_state == review_state);
            }
            projections.sort_by(|left, right| {
                right.updated_at.cmp(&left.updated_at).then_with(|| {
                    left.record_id
                        .into_string()
                        .cmp(&right.record_id.into_string())
                })
            });
            projections.truncate(limit);
            Ok(projections)
        })
        .await
    }

    async fn learning_freshness(
        &self,
        review_state: Option<LearningReviewState>,
    ) -> anyhow::Result<LearningLedgerFreshness> {
        self.run_with_connection(move |connection, wid| {
            let workspace_id = workspace_db_id(wid);
            let records = learning_ledger_events::table
                .filter(learning_ledger_events::workspace_id.eq(workspace_id))
                .order(learning_ledger_events::event_seq.asc())
                .load::<LearningLedgerEventRecord>(connection)?;
            let ledger_cursor = records
                .iter()
                .map(|record| record.event_seq)
                .max()
                .unwrap_or(0);
            let mut projections = project_records(records.clone())?;
            if let Some(review_state) = review_state {
                projections.retain(|projection| projection.review_state == review_state);
            }
            let review_state_fingerprint = fingerprint_projection(&projections);
            Ok(LearningLedgerFreshness {
                ledger_cursor,
                projection_version: ledger_cursor,
                review_state_fingerprint,
            })
        })
        .await
    }
}

impl LearningLedgerEventRecord {
    fn new(event: LearningLedgerEvent, workspace_id: i64) -> anyhow::Result<Self> {
        let source_kind = event.provenance.source_kind;
        let raw_source_id = event.provenance.source_id()?;
        let redacted = RedactedLearningSummary::from_raw(&event.summary);
        let redacted_source_id = RedactedLearningSummary::from_raw(&raw_source_id);
        let redacted_source_event_id =
            RedactedLearningSummary::from_raw(&event.provenance.source_event_id);
        let redacted_source_fingerprint =
            RedactedLearningSummary::from_raw(&event.provenance.source_fingerprint);
        let redacted_tool_name = event
            .provenance
            .tool_name
            .as_ref()
            .map(RedactedLearningSummary::from_raw);
        let redacted_eval_id = event
            .provenance
            .eval_id
            .as_ref()
            .map(RedactedLearningSummary::from_raw);
        let source_id = match source_kind {
            LearningSourceKind::Conversation | LearningSourceKind::Task => raw_source_id,
            LearningSourceKind::Tool => redacted_tool_name
                .as_ref()
                .map(|redacted| redacted.summary.clone())
                .unwrap_or_else(|| redacted_source_id.summary.clone()),
            LearningSourceKind::Eval => redacted_eval_id
                .as_ref()
                .map(|redacted| redacted.summary.clone())
                .unwrap_or_else(|| redacted_source_id.summary.clone()),
        };
        let source_id_redaction_status = match source_kind {
            LearningSourceKind::Conversation | LearningSourceKind::Task => {
                LearningRedactionStatus::Clean
            }
            LearningSourceKind::Tool | LearningSourceKind::Eval => redacted_source_id.status,
        };
        let redaction_status = [
            event.redaction_status,
            redacted.status,
            source_id_redaction_status,
            redacted_source_event_id.status,
            redacted_source_fingerprint.status,
            redacted_tool_name
                .as_ref()
                .map(|redacted| redacted.status)
                .unwrap_or(LearningRedactionStatus::Clean),
            redacted_eval_id
                .as_ref()
                .map(|redacted| redacted.status)
                .unwrap_or(LearningRedactionStatus::Clean),
        ]
        .into_iter()
        .fold(LearningRedactionStatus::Clean, |actual, status| {
            if actual == LearningRedactionStatus::Redacted
                || status == LearningRedactionStatus::Redacted
            {
                LearningRedactionStatus::Redacted
            } else {
                LearningRedactionStatus::Clean
            }
        });
        if let Some(metadata) = event.capture_metadata.as_ref() {
            metadata.validate_current()?;
        }
        let capture_metadata = event
            .capture_metadata
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        Ok(Self {
            event_seq: 0,
            event_id: event.event_id.into_string(),
            record_id: event.record_id.into_string(),
            idempotency_key: event.idempotency_key,
            workspace_id,
            event_kind: event.event_kind.to_string(),
            summary: redacted.summary,
            content_fingerprint: redacted.fingerprint,
            redaction_status: redaction_status.to_string(),
            source_kind: event.provenance.source_kind.to_string(),
            source_id,
            source_event_id: redacted_source_event_id.summary,
            source_fingerprint: redacted_source_fingerprint.fingerprint,
            conversation_id: event.provenance.conversation_id.map(|id| id.into_string()),
            task_id: event.provenance.task_id.map(|id| id.into_string()),
            tool_name: redacted_tool_name.map(|redacted| redacted.summary),
            eval_id: redacted_eval_id.map(|redacted| redacted.summary),
            capture_metadata,
            created_at: event.created_at.naive_utc(),
            schema_version: event.schema_version,
        })
    }

    fn try_into_event(self) -> anyhow::Result<LearningLedgerEvent> {
        let event_kind = parse_event_kind(&self.event_kind)?;
        let redaction_status = parse_redaction_status(&self.redaction_status)?;
        let provenance = self.try_into_provenance()?;
        let capture_metadata = self
            .capture_metadata
            .as_deref()
            .map(serde_json::from_str::<LearningCaptureMetadata>)
            .transpose()?;
        if let Some(metadata) = capture_metadata.as_ref() {
            metadata.validate_current()?;
        }
        Ok(LearningLedgerEvent {
            event_id: LearningEventId::parse(self.event_id)?,
            record_id: LearningRecordId::parse(self.record_id)?,
            idempotency_key: self.idempotency_key,
            event_kind,
            summary: self.summary,
            content_fingerprint: self.content_fingerprint,
            redaction_status,
            provenance,
            capture_metadata,
            created_at: from_naive(self.created_at),
            schema_version: self.schema_version,
        })
    }

    fn try_into_provenance(&self) -> anyhow::Result<LearningProvenance> {
        Ok(LearningProvenance {
            source_kind: parse_source_kind(&self.source_kind)?,
            conversation_id: self
                .conversation_id
                .clone()
                .map(ConversationId::parse)
                .transpose()?,
            task_id: self
                .task_id
                .clone()
                .map(SubagentTaskId::parse)
                .transpose()?,
            tool_name: self.tool_name.clone(),
            eval_id: self.eval_id.clone(),
            source_event_id: self.source_event_id.clone(),
            source_fingerprint: self.source_fingerprint.clone(),
        })
    }
}

#[derive(Clone)]
struct ProjectionBuilder {
    record_id: LearningRecordId,
    summary: String,
    accepted_summary: Option<String>,
    review_state: LearningReviewState,
    redaction_status: LearningRedactionStatus,
    provenance: LearningProvenance,
    capture_metadata: Option<LearningCaptureMetadata>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    schema_version: i32,
}

fn insert_or_replay_learning_event(
    connection: &mut diesel::sqlite::SqliteConnection,
    record: &LearningLedgerEventRecord,
) -> anyhow::Result<LearningLedgerEvent> {
    let existing = learning_ledger_events::table
        .filter(learning_ledger_events::workspace_id.eq(record.workspace_id))
        .filter(learning_ledger_events::idempotency_key.eq(&record.idempotency_key))
        .first::<LearningLedgerEventRecord>(connection)
        .optional()?;
    if let Some(existing) = existing {
        ensure_learning_event_idempotency_replay(&existing, record)?;
        return existing.try_into_event();
    }
    diesel::insert_into(learning_ledger_events::table)
        .values((
            learning_ledger_events::event_id.eq(&record.event_id),
            learning_ledger_events::record_id.eq(&record.record_id),
            learning_ledger_events::idempotency_key.eq(&record.idempotency_key),
            learning_ledger_events::workspace_id.eq(record.workspace_id),
            learning_ledger_events::event_kind.eq(&record.event_kind),
            learning_ledger_events::summary.eq(&record.summary),
            learning_ledger_events::content_fingerprint.eq(&record.content_fingerprint),
            learning_ledger_events::redaction_status.eq(&record.redaction_status),
            learning_ledger_events::source_kind.eq(&record.source_kind),
            learning_ledger_events::source_id.eq(&record.source_id),
            learning_ledger_events::source_event_id.eq(&record.source_event_id),
            learning_ledger_events::source_fingerprint.eq(&record.source_fingerprint),
            learning_ledger_events::conversation_id.eq(&record.conversation_id),
            learning_ledger_events::task_id.eq(&record.task_id),
            learning_ledger_events::tool_name.eq(&record.tool_name),
            learning_ledger_events::eval_id.eq(&record.eval_id),
            learning_ledger_events::capture_metadata.eq(&record.capture_metadata),
            learning_ledger_events::created_at.eq(record.created_at),
            learning_ledger_events::schema_version.eq(record.schema_version),
        ))
        .execute(connection)?;
    learning_ledger_events::table
        .filter(learning_ledger_events::workspace_id.eq(record.workspace_id))
        .filter(learning_ledger_events::idempotency_key.eq(&record.idempotency_key))
        .first::<LearningLedgerEventRecord>(connection)?
        .try_into_event()
}

fn ensure_learning_event_idempotency_replay(
    existing: &LearningLedgerEventRecord,
    replay: &LearningLedgerEventRecord,
) -> anyhow::Result<()> {
    if existing.idempotency_key == replay.idempotency_key
        && existing.event_kind == replay.event_kind
        && (existing.event_kind == LearningEventKind::CandidateCaptured.to_string()
            || existing.record_id == replay.record_id)
        && existing.summary == replay.summary
        && existing.content_fingerprint == replay.content_fingerprint
        && existing.redaction_status == replay.redaction_status
        && existing.source_kind == replay.source_kind
        && existing.source_id == replay.source_id
        && existing.source_event_id == replay.source_event_id
        && existing.source_fingerprint == replay.source_fingerprint
        && existing.conversation_id == replay.conversation_id
        && existing.task_id == replay.task_id
        && existing.tool_name == replay.tool_name
        && existing.eval_id == replay.eval_id
        && existing.capture_metadata == replay.capture_metadata
        && existing.schema_version == replay.schema_version
    {
        return Ok(());
    }

    anyhow::bail!("learning event idempotency key collision for a different event semantics")
}

fn project_records(
    records: Vec<LearningLedgerEventRecord>,
) -> anyhow::Result<Vec<LearningRecordProjection>> {
    let mut projections: BTreeMap<String, ProjectionBuilder> = BTreeMap::new();
    for record in records {
        let event = record.try_into_event()?;
        let record_key = event.record_id.into_string();
        match event.event_kind {
            LearningEventKind::CandidateCaptured => {
                projections.entry(record_key).or_insert(ProjectionBuilder {
                    record_id: event.record_id,
                    summary: event.summary,
                    accepted_summary: None,
                    review_state: LearningReviewState::Candidate,
                    redaction_status: event.redaction_status,
                    provenance: event.provenance,
                    capture_metadata: event.capture_metadata,
                    created_at: event.created_at,
                    updated_at: event.created_at,
                    schema_version: event.schema_version,
                });
            }
            LearningEventKind::ReviewAccepted => {
                if let Some(projection) = projections.get_mut(&record_key) {
                    projection.review_state = LearningReviewState::Accepted;
                    projection.accepted_summary = accepted_summary_from_review_event(&event)?;
                    projection.updated_at = event.created_at;
                }
            }
            LearningEventKind::ReviewRejected => {
                if let Some(projection) = projections.get_mut(&record_key) {
                    projection.review_state = LearningReviewState::Rejected;
                    projection.updated_at = event.created_at;
                }
            }
            LearningEventKind::SensorLessonProposed
            | LearningEventKind::SensorReviewPending
            | LearningEventKind::SensorReviewRejected
            | LearningEventKind::PromotionAudit => {
                if let Some(projection) = projections.get_mut(&record_key) {
                    projection.updated_at = event.created_at;
                }
            }
            LearningEventKind::Superseded => {
                if let Some(projection) = projections.get_mut(&record_key) {
                    projection.review_state = LearningReviewState::Superseded;
                    projection.updated_at = event.created_at;
                }
            }
        }
    }
    Ok(projections
        .into_values()
        .map(|projection| LearningRecordProjection {
            record_id: projection.record_id,
            summary: projection.summary,
            accepted_summary: projection.accepted_summary,
            review_state: projection.review_state,
            redaction_status: projection.redaction_status,
            provenance: projection.provenance,
            capture_metadata: projection.capture_metadata,
            created_at: projection.created_at,
            updated_at: projection.updated_at,
            schema_version: projection.schema_version,
        })
        .collect())
}

fn accepted_summary_from_review_event(
    event: &LearningLedgerEvent,
) -> anyhow::Result<Option<String>> {
    if event.provenance.eval_id.as_deref()
        != Some(SANCTIONED_SANITIZED_OBSERVATION_PROMOTION_REVIEWER_ID)
    {
        return Ok(None);
    }
    if !event
        .summary
        .contains(SANCTIONED_SANITIZED_OBSERVATION_PROMOTION_REASON)
        || !event.summary.contains(
            "accepted_summary=sanctioned_sanitized_observation:validated_counters_and_fingerprints",
        )
    {
        anyhow::bail!("sensor promotion review summary is not canonical");
    }
    Ok(Some(
        AcceptedLearningSummary::new(
            "sanctioned_sanitized_observation:validated_counters_and_fingerprints",
        )?
        .into_string(),
    ))
}

fn fingerprint_projection(projections: &[LearningRecordProjection]) -> String {
    let mut hasher = Sha256::new();
    for projection in projections {
        hasher.update(projection.record_id.into_string());
        hasher.update(projection.review_state.to_string());
        hasher.update(projection.updated_at.to_rfc3339());
    }
    hex::encode(hasher.finalize())
}

fn workspace_db_id(wid: WorkspaceHash) -> i64 {
    i64::from_ne_bytes(wid.id().to_ne_bytes())
}

fn review_target_state(event_kind: LearningEventKind) -> anyhow::Result<LearningReviewState> {
    match event_kind {
        LearningEventKind::ReviewAccepted => Ok(LearningReviewState::Accepted),
        LearningEventKind::ReviewRejected => Ok(LearningReviewState::Rejected),
        LearningEventKind::Superseded => Ok(LearningReviewState::Superseded),
        LearningEventKind::CandidateCaptured
        | LearningEventKind::SensorLessonProposed
        | LearningEventKind::SensorReviewPending
        | LearningEventKind::SensorReviewRejected
        | LearningEventKind::PromotionAudit => {
            anyhow::bail!("event kind {} cannot review learning record", event_kind)
        }
    }
}

fn parse_event_kind(value: &str) -> anyhow::Result<LearningEventKind> {
    LearningEventKind::from_str(value)
        .map_err(|_| anyhow::anyhow!("Unknown learning event kind '{value}'"))
}

fn parse_source_kind(value: &str) -> anyhow::Result<LearningSourceKind> {
    LearningSourceKind::from_str(value)
        .map_err(|_| anyhow::anyhow!("Unknown learning source kind '{value}'"))
}

fn parse_redaction_status(value: &str) -> anyhow::Result<LearningRedactionStatus> {
    LearningRedactionStatus::from_str(value)
        .map_err(|_| anyhow::anyhow!("Unknown learning redaction status '{value}'"))
}

fn from_naive(value: NaiveDateTime) -> DateTime<Utc> {
    DateTime::from_naive_utc_and_offset(value, Utc)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use chrono::Duration;
    use forge_domain::{
        FakeLearningSensorReviewer, LEARNING_LEDGER_SCHEMA_VERSION, LearningSensorReviewInput,
        LearningSensorReviewer, RedactedLearningSummary, SanitizedChatLessonObservation,
        SanitizedChatObservationKind, SanitizedObservationCountBucket,
        SanitizedObservationSeverity,
    };
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::database::DatabasePool;

    fn fixture_repo(workspace_id: u64) -> anyhow::Result<LearningRepositoryImpl> {
        let pool = Arc::new(DatabasePool::in_memory()?);
        Ok(LearningRepositoryImpl::new(
            pool,
            WorkspaceHash::new(workspace_id),
        ))
    }

    fn fixture_event(
        conversation_id: ConversationId,
        source_event_id: &str,
        summary: &str,
        created_at: DateTime<Utc>,
    ) -> anyhow::Result<LearningLedgerEvent> {
        LearningLedgerEvent::capture_candidate(
            summary,
            LearningProvenance::conversation(
                conversation_id,
                source_event_id,
                RedactedLearningSummary::from_raw(summary).fingerprint,
            ),
            created_at,
        )
    }

    #[tokio::test]
    async fn learning_insert_is_idempotent_for_duplicate_capture() -> anyhow::Result<()> {
        let fixture = fixture_repo(1)?;
        let conversation_id = ConversationId::generate();
        let created_at = Utc::now();
        let first = fixture_event(conversation_id, "event-1", "typed ledger", created_at)?;
        let second = fixture_event(conversation_id, "event-1", "typed ledger", created_at)?;

        let left = fixture.insert_learning_event(first).await?;
        let right = fixture.insert_learning_event(second).await?;
        let records = fixture.list_learning_records(None, 10).await?;
        let actual = (
            left.event.event_id,
            right.event.event_id,
            left.freshness,
            right.freshness,
            records.len(),
        );
        let expected = (
            left.event.event_id,
            left.event.event_id,
            forge_domain::LearningLedgerEventFreshness::Inserted,
            forge_domain::LearningLedgerEventFreshness::Existing,
            1usize,
        );

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_insert_rejects_idempotency_collision_for_different_candidate()
    -> anyhow::Result<()> {
        let fixture = fixture_repo(18)?;
        let conversation_id = ConversationId::generate();
        let created_at = Utc::now();
        let first = fixture_event(conversation_id, "event-1", "original candidate", created_at)?;
        let first_outcome = fixture.insert_learning_event(first).await?;
        let mut colliding = fixture_event(
            conversation_id,
            "event-2",
            "different candidate with forged idempotency key",
            created_at + Duration::seconds(1),
        )?;
        colliding.idempotency_key = first_outcome.event.idempotency_key;

        let colliding_result = fixture.insert_learning_event(colliding).await;
        let records = fixture.list_learning_records(None, 10).await?;
        let actual = (
            colliding_result.is_err(),
            records.len(),
            records.first().map(|record| record.summary.clone()),
        );
        let expected = (true, 1usize, Some("original candidate".to_string()));

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_insert_is_idempotent_for_duplicate_review_event() -> anyhow::Result<()> {
        let fixture = fixture_repo(19)?;
        let conversation_id = ConversationId::generate();
        let created_at = Utc::now();
        let candidate = fixture
            .insert_learning_event(fixture_event(
                conversation_id,
                "event-1",
                "generic review replay candidate",
                created_at,
            )?)
            .await?
            .event;
        let review = LearningLedgerEvent::review(
            candidate.record_id,
            LearningEventKind::ReviewAccepted,
            "generic insert review replay",
            LearningProvenance::conversation(conversation_id, "review-1", "review-fingerprint-1"),
            created_at + Duration::seconds(1),
        )?;

        let left = fixture.insert_learning_event(review.clone()).await?;
        let right = fixture.insert_learning_event(review).await?;
        let events = fixture
            .run_with_connection(move |connection, wid| {
                learning_ledger_events::table
                    .filter(learning_ledger_events::workspace_id.eq(workspace_db_id(wid)))
                    .load::<LearningLedgerEventRecord>(connection)
                    .map_err(Into::into)
            })
            .await?;
        let actual = (left.freshness, right.freshness, events.len());
        let expected = (
            forge_domain::LearningLedgerEventFreshness::Inserted,
            forge_domain::LearningLedgerEventFreshness::Existing,
            2usize,
        );

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_projection_is_append_only_and_review_state_is_event_derived()
    -> anyhow::Result<()> {
        let fixture = fixture_repo(2)?;
        let conversation_id = ConversationId::generate();
        let created_at = Utc::now();
        let candidate = fixture
            .insert_learning_event(fixture_event(
                conversation_id,
                "event-1",
                "accept reviewed learning only",
                created_at,
            )?)
            .await?
            .event;
        let review = LearningLedgerEvent::review(
            candidate.record_id,
            LearningEventKind::ReviewAccepted,
            "review accepted",
            LearningProvenance::conversation(conversation_id, "review-1", "review-fingerprint-1"),
            created_at + Duration::seconds(1),
        )?;
        fixture.insert_learning_event(review).await?;

        let actual = fixture
            .list_learning_records(Some(LearningReviewState::Accepted), 10)
            .await?
            .first()
            .map(|projection| (projection.summary.clone(), projection.review_state));
        let expected = Some((
            "accept reviewed learning only".to_string(),
            LearningReviewState::Accepted,
        ));

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_review_event_is_terminal_and_repeat_returns_persisted_event()
    -> anyhow::Result<()> {
        let fixture = fixture_repo(11)?;
        let conversation_id = ConversationId::generate();
        let created_at = Utc::now();
        let candidate = fixture
            .insert_learning_event(fixture_event(
                conversation_id,
                "event-1",
                "terminal review transition",
                created_at,
            )?)
            .await?
            .event;
        let accepted = LearningLedgerEvent::review(
            candidate.record_id,
            LearningEventKind::ReviewAccepted,
            "same review note",
            LearningProvenance::conversation(conversation_id, "review-1", "review-fingerprint-1"),
            created_at + Duration::seconds(1),
        )?;
        let repeated = accepted.clone();
        let first = fixture.review_learning_candidate_event(accepted).await?;
        let second = fixture.review_learning_candidate_event(repeated).await?;
        let rejected = LearningLedgerEvent::review(
            candidate.record_id,
            LearningEventKind::ReviewRejected,
            "conflicting review",
            LearningProvenance::conversation(conversation_id, "review-3", "review-fingerprint-3"),
            created_at + Duration::seconds(3),
        )?;
        let conflicting = fixture.review_learning_candidate_event(rejected).await;
        let events = fixture
            .run_with_connection(move |connection, wid| {
                learning_ledger_events::table
                    .filter(learning_ledger_events::workspace_id.eq(workspace_db_id(wid)))
                    .load::<LearningLedgerEventRecord>(connection)
                    .map_err(Into::into)
            })
            .await?;

        let actual = (
            first.projection.review_state,
            second.event.event_id,
            first.event.event_id,
            conflicting.is_err(),
            events.len(),
        );
        let expected = (
            LearningReviewState::Accepted,
            first.event.event_id,
            first.event.event_id,
            true,
            2usize,
        );

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_review_rejects_idempotency_collision_for_different_record()
    -> anyhow::Result<()> {
        let fixture = fixture_repo(12)?;
        let conversation_id = ConversationId::generate();
        let created_at = Utc::now();
        let first_candidate = fixture
            .insert_learning_event(fixture_event(
                conversation_id,
                "event-1",
                "first terminal review transition",
                created_at,
            )?)
            .await?
            .event;
        let second_candidate = fixture
            .insert_learning_event(fixture_event(
                conversation_id,
                "event-2",
                "second terminal review transition",
                created_at + Duration::seconds(1),
            )?)
            .await?
            .event;
        let first_review = LearningLedgerEvent::review(
            first_candidate.record_id,
            LearningEventKind::ReviewAccepted,
            "first review note",
            LearningProvenance::conversation(conversation_id, "review-1", "review-fingerprint-1"),
            created_at + Duration::seconds(2),
        )?;
        let first_outcome = fixture
            .review_learning_candidate_event(first_review)
            .await?;
        let mut colliding_review = LearningLedgerEvent::review(
            second_candidate.record_id,
            LearningEventKind::ReviewAccepted,
            "second review note with malicious idempotency collision",
            LearningProvenance::conversation(conversation_id, "review-2", "review-fingerprint-2"),
            created_at + Duration::seconds(3),
        )?;
        colliding_review.idempotency_key = first_outcome.event.idempotency_key.clone();

        let colliding = fixture
            .review_learning_candidate_event(colliding_review)
            .await;
        let second_projection = fixture
            .list_learning_records(None, 10)
            .await?
            .into_iter()
            .find(|projection| projection.record_id == second_candidate.record_id)
            .expect("second candidate projection should exist");
        let events = fixture
            .run_with_connection(move |connection, wid| {
                learning_ledger_events::table
                    .filter(learning_ledger_events::workspace_id.eq(workspace_db_id(wid)))
                    .load::<LearningLedgerEventRecord>(connection)
                    .map_err(Into::into)
            })
            .await?;

        let actual = (
            colliding.is_err(),
            second_projection.review_state,
            events.len(),
        );
        let expected = (true, LearningReviewState::Candidate, 3usize);

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_review_rejects_idempotency_collision_for_same_record_and_kind()
    -> anyhow::Result<()> {
        let fixture = fixture_repo(20)?;
        let conversation_id = ConversationId::generate();
        let created_at = Utc::now();
        let candidate = fixture
            .insert_learning_event(fixture_event(
                conversation_id,
                "event-1",
                "same record same kind collision candidate",
                created_at,
            )?)
            .await?
            .event;
        let first_review = LearningLedgerEvent::review(
            candidate.record_id,
            LearningEventKind::ReviewAccepted,
            "original accepted review",
            LearningProvenance::conversation(conversation_id, "review-1", "review-fingerprint-1"),
            created_at + Duration::seconds(1),
        )?;
        let first_outcome = fixture
            .review_learning_candidate_event(first_review)
            .await?;
        let mut colliding_review = LearningLedgerEvent::review(
            candidate.record_id,
            LearningEventKind::ReviewAccepted,
            "different accepted review with forged idempotency key",
            LearningProvenance::conversation(conversation_id, "review-2", "review-fingerprint-2"),
            created_at + Duration::seconds(2),
        )?;
        colliding_review.idempotency_key = first_outcome.event.idempotency_key.clone();

        let colliding = fixture
            .review_learning_candidate_event(colliding_review)
            .await;
        let events = fixture
            .run_with_connection(move |connection, wid| {
                learning_ledger_events::table
                    .filter(learning_ledger_events::workspace_id.eq(workspace_db_id(wid)))
                    .load::<LearningLedgerEventRecord>(connection)
                    .map_err(Into::into)
            })
            .await?;
        let actual = (colliding.is_err(), events.len());
        let expected = (true, 2usize);

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_review_rejects_stale_pre_candidate_replay_collision_for_same_record_and_kind()
    -> anyhow::Result<()> {
        let fixture = fixture_repo(21)?;
        let conversation_id = ConversationId::generate();
        let created_at = Utc::now();
        let candidate_event = fixture_event(
            conversation_id,
            "event-1",
            "pre candidate stale replay collision candidate",
            created_at,
        )?;
        let stale_review = LearningLedgerEvent::review(
            candidate_event.record_id,
            LearningEventKind::ReviewAccepted,
            "stale accepted review inserted before candidate",
            LearningProvenance::conversation(conversation_id, "review-1", "review-fingerprint-1"),
            created_at - Duration::seconds(1),
        )?;
        let stale_outcome = fixture.insert_learning_event(stale_review).await?;
        let candidate = fixture.insert_learning_event(candidate_event).await?.event;
        let mut colliding_review = LearningLedgerEvent::review(
            candidate.record_id,
            LearningEventKind::ReviewAccepted,
            "different accepted review with stale forged idempotency key",
            LearningProvenance::conversation(conversation_id, "review-2", "review-fingerprint-2"),
            created_at + Duration::seconds(1),
        )?;
        colliding_review.idempotency_key = stale_outcome.event.idempotency_key.clone();

        let colliding = fixture
            .review_learning_candidate_event(colliding_review)
            .await;
        let projection = fixture
            .get_learning_record(candidate.record_id)
            .await?
            .expect("candidate projection should exist");
        let actual = (colliding.is_err(), projection.review_state);
        let expected = (true, LearningReviewState::Candidate);

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_review_rejects_stale_pre_candidate_exact_replay_without_projection_transition()
    -> anyhow::Result<()> {
        let fixture = fixture_repo(22)?;
        let conversation_id = ConversationId::generate();
        let created_at = Utc::now();
        let candidate_event = fixture_event(
            conversation_id,
            "event-1",
            "pre candidate exact stale replay candidate",
            created_at,
        )?;
        let stale_review = LearningLedgerEvent::review(
            candidate_event.record_id,
            LearningEventKind::ReviewAccepted,
            "stale accepted review inserted before candidate",
            LearningProvenance::conversation(conversation_id, "review-1", "review-fingerprint-1"),
            created_at - Duration::seconds(1),
        )?;
        fixture.insert_learning_event(stale_review.clone()).await?;
        let candidate = fixture.insert_learning_event(candidate_event).await?.event;
        let exact_replay = LearningLedgerEvent { record_id: candidate.record_id, ..stale_review };

        let replay = fixture.review_learning_candidate_event(exact_replay).await;
        let projection = fixture
            .get_learning_record(candidate.record_id)
            .await?
            .expect("candidate projection should exist");
        let actual = (replay.is_err(), projection.review_state);
        let expected = (true, LearningReviewState::Candidate);

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_review_rejects_idempotency_collision_for_different_event_kind()
    -> anyhow::Result<()> {
        let fixture = fixture_repo(13)?;
        let conversation_id = ConversationId::generate();
        let created_at = Utc::now();
        let candidate = fixture
            .insert_learning_event(fixture_event(
                conversation_id,
                "event-1",
                "same record different event kind collision",
                created_at,
            )?)
            .await?
            .event;
        let mut colliding_review = LearningLedgerEvent::review(
            candidate.record_id,
            LearningEventKind::ReviewAccepted,
            "review note with candidate idempotency collision",
            LearningProvenance::conversation(conversation_id, "review-1", "review-fingerprint-1"),
            created_at + Duration::seconds(1),
        )?;
        colliding_review.idempotency_key = candidate.idempotency_key.clone();

        let colliding = fixture
            .review_learning_candidate_event(colliding_review)
            .await;
        let projection = fixture.list_learning_records(None, 10).await?.remove(0);
        let events = fixture
            .run_with_connection(move |connection, wid| {
                learning_ledger_events::table
                    .filter(learning_ledger_events::workspace_id.eq(workspace_db_id(wid)))
                    .load::<LearningLedgerEventRecord>(connection)
                    .map_err(Into::into)
            })
            .await?;

        let actual = (colliding.is_err(), projection.review_state, events.len());
        let expected = (true, LearningReviewState::Candidate, 1usize);

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_queries_are_workspace_isolated() -> anyhow::Result<()> {
        let left = fixture_repo(3)?;
        let right = fixture_repo(4)?;
        let conversation_id = ConversationId::generate();
        let created_at = Utc::now();
        left.insert_learning_event(fixture_event(
            conversation_id,
            "event-1",
            "left workspace",
            created_at,
        )?)
        .await?;
        right
            .insert_learning_event(fixture_event(
                conversation_id,
                "event-2",
                "right workspace",
                created_at,
            )?)
            .await?;

        let actual = (
            left.list_learning_records(None, 10).await?[0]
                .summary
                .clone(),
            right.list_learning_records(None, 10).await?[0]
                .summary
                .clone(),
        );
        let expected = ("left workspace".to_string(), "right workspace".to_string());

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_persistence_roundtrip_preserves_provenance_and_schema() -> anyhow::Result<()>
    {
        let fixture = fixture_repo(5)?;
        let conversation_id = ConversationId::generate();
        let event = fixture_event(conversation_id, "event-1", "roundtrip", Utc::now())?;
        fixture.insert_learning_event(event).await?;

        let actual = fixture
            .list_learning_records(None, 10)
            .await?
            .first()
            .map(|projection| {
                (
                    projection.provenance.conversation_id,
                    projection.provenance.source_event_id.clone(),
                    projection.schema_version,
                )
            });
        let expected = Some((
            Some(conversation_id),
            "event-1".to_string(),
            LEARNING_LEDGER_SCHEMA_VERSION,
        ));

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_redaction_prevents_raw_secret_persistence() -> anyhow::Result<()> {
        let fixture = fixture_repo(6)?;
        let conversation_id = ConversationId::generate();
        fixture
            .insert_learning_event(fixture_event(
                conversation_id,
                "event-1",
                "password hunter2 token sk-123456789012345678901234",
                Utc::now(),
            )?)
            .await?;

        let projection = fixture.list_learning_records(None, 10).await?.remove(0);
        let actual = (
            projection.summary.contains("hunter2"),
            projection.summary.contains("sk-"),
            projection.redaction_status,
        );
        let expected = (false, false, LearningRedactionStatus::Redacted);

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_insert_rejects_or_redacts_explicit_raw_secret_event() -> anyhow::Result<()> {
        let fixture = fixture_repo(8)?;
        let conversation_id = ConversationId::generate();
        let raw_secret = "token sk-123456789012345678901234";
        let event = LearningLedgerEvent {
            event_id: LearningEventId::generate(),
            record_id: LearningRecordId::generate(),
            idempotency_key: "explicit-raw-secret".to_string(),
            event_kind: LearningEventKind::CandidateCaptured,
            summary: raw_secret.to_string(),
            content_fingerprint: "raw-secret-fingerprint".to_string(),
            redaction_status: LearningRedactionStatus::Clean,
            provenance: LearningProvenance::conversation(
                conversation_id,
                "event-raw-secret",
                "source fingerprint contains token sk-123456789012345678901234",
            ),
            capture_metadata: None,
            created_at: Utc::now(),
            schema_version: LEARNING_LEDGER_SCHEMA_VERSION,
        };

        let actual = match fixture.insert_learning_event(event).await {
            Ok(_) => fixture
                .list_learning_records(None, 10)
                .await?
                .first()
                .map(|projection| {
                    (
                        projection.summary.contains("sk-")
                            || projection.provenance.source_fingerprint.contains("sk-"),
                        projection.redaction_status,
                    )
                }),
            Err(_) => None,
        };
        let expected = Some((false, LearningRedactionStatus::Redacted));

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_insert_redacts_secret_bearing_provenance_identity_fields()
    -> anyhow::Result<()> {
        let fixture = fixture_repo(9)?;
        let event = LearningLedgerEvent {
            event_id: LearningEventId::generate(),
            record_id: LearningRecordId::generate(),
            idempotency_key: "explicit-raw-secret-provenance".to_string(),
            event_kind: LearningEventKind::CandidateCaptured,
            summary: "safe summary".to_string(),
            content_fingerprint: "safe-fingerprint".to_string(),
            redaction_status: LearningRedactionStatus::Clean,
            provenance: LearningProvenance {
                source_kind: LearningSourceKind::Tool,
                conversation_id: None,
                task_id: None,
                tool_name: Some("tool token sk-123456789012345678901234".to_string()),
                eval_id: Some("eval token sk-123456789012345678901234".to_string()),
                source_event_id: "event token sk-123456789012345678901234".to_string(),
                source_fingerprint: "safe-source-fingerprint".to_string(),
            },
            capture_metadata: None,
            created_at: Utc::now(),
            schema_version: LEARNING_LEDGER_SCHEMA_VERSION,
        };

        fixture.insert_learning_event(event).await?;

        let projection = fixture.list_learning_records(None, 10).await?.remove(0);
        let persisted_event = fixture
            .run_with_connection(move |connection, wid| {
                learning_ledger_events::table
                    .filter(learning_ledger_events::workspace_id.eq(workspace_db_id(wid)))
                    .first::<LearningLedgerEventRecord>(connection)
                    .map_err(Into::into)
            })
            .await?;
        let actual = (
            projection
                .provenance
                .tool_name
                .unwrap_or_default()
                .contains("sk-")
                || projection.provenance.source_event_id.contains("sk-")
                || projection
                    .provenance
                    .eval_id
                    .unwrap_or_default()
                    .contains("sk-"),
            persisted_event.source_id.contains("sk-")
                || persisted_event.source_event_id.contains("sk-")
                || persisted_event
                    .tool_name
                    .unwrap_or_default()
                    .contains("sk-")
                || persisted_event.eval_id.unwrap_or_default().contains("sk-"),
            projection.redaction_status,
        );
        let expected = (false, false, LearningRedactionStatus::Redacted);

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_insert_preserves_clean_typed_conversation_identity() -> anyhow::Result<()> {
        let fixture = fixture_repo(10)?;
        let conversation_id = ConversationId::generate();
        let event = LearningLedgerEvent::capture_candidate(
            "clean typed identity",
            LearningProvenance::conversation(conversation_id, "event-1", "safe-source"),
            Utc::now(),
        )?;

        fixture.insert_learning_event(event).await?;

        let persisted_event = fixture
            .run_with_connection(move |connection, wid| {
                learning_ledger_events::table
                    .filter(learning_ledger_events::workspace_id.eq(workspace_db_id(wid)))
                    .first::<LearningLedgerEventRecord>(connection)
                    .map_err(Into::into)
            })
            .await?;
        let actual = (persisted_event.source_id, persisted_event.redaction_status);
        let expected = (
            conversation_id.into_string(),
            LearningRedactionStatus::Clean.to_string(),
        );

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_idempotency_is_workspace_scoped_on_shared_database() -> anyhow::Result<()> {
        let pool = Arc::new(DatabasePool::in_memory()?);
        let left = LearningRepositoryImpl::new(pool.clone(), WorkspaceHash::new(9));
        let right = LearningRepositoryImpl::new(pool, WorkspaceHash::new(10));
        let conversation_id = ConversationId::generate();
        let created_at = Utc::now();
        left.insert_learning_event(fixture_event(
            conversation_id,
            "event-1",
            "same source across isolated workspaces",
            created_at,
        )?)
        .await?;

        let right_result = right
            .insert_learning_event(fixture_event(
                conversation_id,
                "event-1",
                "same source across isolated workspaces",
                created_at,
            )?)
            .await;
        let actual = (
            right_result.is_ok(),
            left.list_learning_records(None, 10).await?.len(),
            right.list_learning_records(None, 10).await?.len(),
        );
        let expected = (true, 1usize, 1usize);

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_persistence_roundtrip_preserves_capture_metadata() -> anyhow::Result<()> {
        let fixture = fixture_repo(14)?;
        let conversation_id = ConversationId::generate();
        let mut event =
            fixture_event(conversation_id, "event-1", "metadata roundtrip", Utc::now())?;
        let metadata = LearningCaptureMetadata::conversation_save(
            3,
            1,
            "context-fingerprint-14",
            "summary-fingerprint-14",
        );
        event.capture_metadata = Some(metadata.clone());
        fixture.insert_learning_event(event).await?;

        let actual = fixture
            .list_learning_records(None, 10)
            .await?
            .first()
            .and_then(|projection| projection.capture_metadata.clone());
        let expected = Some(metadata);
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_repository_rejects_invalid_capture_metadata() -> anyhow::Result<()> {
        let fixture = fixture_repo(15)?;
        let conversation_id = ConversationId::generate();
        let mut event = fixture_event(conversation_id, "event-1", "invalid metadata", Utc::now())?;
        event.capture_metadata = Some(LearningCaptureMetadata::conversation_save(
            0,
            1,
            "context-fingerprint-15",
            "summary-fingerprint-15",
        ));

        let actual = fixture.insert_learning_event(event).await.is_err();
        let expected = true;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_repository_rejects_corrupt_current_capture_metadata_on_readback()
    -> anyhow::Result<()> {
        let fixture = fixture_repo(16)?;
        let conversation_id = ConversationId::generate();
        let mut event = fixture_event(
            conversation_id,
            "event-1",
            "corrupt metadata readback",
            Utc::now(),
        )?;
        event.capture_metadata = Some(LearningCaptureMetadata::conversation_save(
            3,
            1,
            "context-fingerprint-16",
            "summary-fingerprint-16",
        ));
        fixture.insert_learning_event(event).await?;
        fixture
            .run_with_connection(move |connection, wid| {
                diesel::update(
                    learning_ledger_events::table
                        .filter(learning_ledger_events::workspace_id.eq(workspace_db_id(wid))),
                )
                .set(learning_ledger_events::capture_metadata.eq(Some(
                    r#"{"source":"conversation_save","capture_version":1,"message_count":0,"user_message_count":1,"context_fingerprint":"context-fingerprint-16","summary_fingerprint":"summary-fingerprint-16"}"#,
                )))
                .execute(connection)?;
                Ok(())
            })
            .await?;

        let actual = fixture.list_learning_records(None, 10).await.is_err();
        let expected = true;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_repository_preserves_absent_capture_metadata() -> anyhow::Result<()> {
        let fixture = fixture_repo(17)?;
        let conversation_id = ConversationId::generate();
        let event = fixture_event(conversation_id, "event-1", "absent metadata", Utc::now())?;
        fixture.insert_learning_event(event).await?;

        let actual = fixture
            .list_learning_records(None, 10)
            .await?
            .first()
            .and_then(|projection| projection.capture_metadata.clone());
        let expected = None;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_sensor_proposal_event_does_not_transition_to_accepted() -> anyhow::Result<()>
    {
        let fixture = fixture_repo(23)?;
        let conversation_id = ConversationId::generate();
        let created_at = Utc::now();
        let candidate = fixture
            .insert_learning_event(fixture_event(
                conversation_id,
                "event-1",
                "sensor proposal stays non accepted",
                created_at,
            )?)
            .await?
            .event;
        let projection = fixture
            .get_learning_record(candidate.record_id)
            .await?
            .expect("candidate projection should exist");
        let input = LearningSensorReviewInput::fake_fixture(
            &projection,
            "Durable typed observation",
            "A recurring typed fixture observation exists",
        );
        let output = FakeLearningSensorReviewer.review(input.clone())?;
        let event = output.into_sensor_event(&input, created_at + Duration::seconds(1))?;

        fixture.insert_learning_event(event).await?;

        let projection = fixture
            .get_learning_record(candidate.record_id)
            .await?
            .expect("candidate projection should exist after sensor event");
        let accepted = fixture
            .list_learning_records(Some(LearningReviewState::Accepted), 10)
            .await?;
        let actual = (projection.review_state, accepted.len());
        let expected = (LearningReviewState::Candidate, 0usize);

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_sensor_event_replay_is_idempotent() -> anyhow::Result<()> {
        let fixture = fixture_repo(24)?;
        let conversation_id = ConversationId::generate();
        let created_at = Utc::now();
        let candidate = fixture
            .insert_learning_event(fixture_event(
                conversation_id,
                "event-1",
                "sensor idempotent replay",
                created_at,
            )?)
            .await?
            .event;
        let projection = fixture
            .get_learning_record(candidate.record_id)
            .await?
            .expect("candidate projection should exist");
        let input = LearningSensorReviewInput::fake_fixture(
            &projection,
            "Durable typed observation",
            "A recurring typed fixture observation exists",
        );
        let output = FakeLearningSensorReviewer.review(input.clone())?;
        let first_event = output.into_sensor_event(&input, created_at + Duration::seconds(1))?;
        let second_event = output.into_sensor_event(&input, created_at + Duration::seconds(2))?;

        let first = fixture.insert_learning_event(first_event).await?;
        let second = fixture.insert_learning_event(second_event).await?;
        let events = fixture
            .run_with_connection(move |connection, wid| {
                learning_ledger_events::table
                    .filter(learning_ledger_events::workspace_id.eq(workspace_db_id(wid)))
                    .load::<LearningLedgerEventRecord>(connection)
                    .map_err(Into::into)
            })
            .await?;
        let actual = (first.freshness, second.freshness, events.len());
        let expected = (
            forge_domain::LearningLedgerEventFreshness::Inserted,
            forge_domain::LearningLedgerEventFreshness::Existing,
            2usize,
        );

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_freshness_for_accepted_ignores_sensor_events() -> anyhow::Result<()> {
        let fixture = fixture_repo(25)?;
        let conversation_id = ConversationId::generate();
        let created_at = Utc::now();
        let candidate = fixture
            .insert_learning_event(fixture_event(
                conversation_id,
                "event-1",
                "sensor events excluded from accepted freshness",
                created_at,
            )?)
            .await?
            .event;
        let before = fixture
            .learning_freshness(Some(LearningReviewState::Accepted))
            .await?;
        let projection = fixture
            .get_learning_record(candidate.record_id)
            .await?
            .expect("candidate projection should exist");
        let input = LearningSensorReviewInput::fake_fixture(
            &projection,
            "Durable typed observation",
            "A recurring typed fixture observation exists",
        );
        let output = FakeLearningSensorReviewer.review(input.clone())?;
        fixture
            .insert_learning_event(
                output.into_sensor_event(&input, created_at + Duration::seconds(1))?,
            )
            .await?;
        let after = fixture
            .learning_freshness(Some(LearningReviewState::Accepted))
            .await?;

        let actual = before.review_state_fingerprint == after.review_state_fingerprint;
        let expected = true;

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn learning_sensor_stale_projection_hash_is_detectable_before_append()
    -> anyhow::Result<()> {
        let fixture = fixture_repo(26)?;
        let conversation_id = ConversationId::generate();
        let created_at = Utc::now();
        let candidate = fixture
            .insert_learning_event(fixture_event(
                conversation_id,
                "event-1",
                "sensor stale candidate",
                created_at,
            )?)
            .await?
            .event;
        let projection = fixture
            .get_learning_record(candidate.record_id)
            .await?
            .expect("candidate projection should exist");
        let mut input = LearningSensorReviewInput::fake_fixture(
            &projection,
            "Durable typed observation",
            "A recurring typed fixture observation exists",
        );
        input.sanitized_projection_hash = "stale".to_string();
        let stale_detected =
            forge_domain::learning_projection_hash(&projection) != input.sanitized_projection_hash;
        let events = fixture
            .run_with_connection(move |connection, wid| {
                learning_ledger_events::table
                    .filter(learning_ledger_events::workspace_id.eq(workspace_db_id(wid)))
                    .load::<LearningLedgerEventRecord>(connection)
                    .map_err(Into::into)
            })
            .await?;
        let actual = (stale_detected, events.len());
        let expected = (true, 1usize);

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn sanitized_observation_sensor_proposal_remains_candidate_non_accepted()
    -> anyhow::Result<()> {
        let fixture = fixture_repo(27)?;
        let conversation_id = ConversationId::generate();
        let created_at = Utc::now();
        let candidate = fixture
            .insert_learning_event(fixture_event(
                conversation_id,
                "event-1",
                "sanitized observation proposal stays audit only",
                created_at,
            )?)
            .await?
            .event;
        let projection = fixture
            .get_learning_record(candidate.record_id)
            .await?
            .expect("candidate projection should exist");
        let input = LearningSensorReviewInput::from_sanitized_chat_observation(
            &projection,
            fixture_sanitized_observation().validate()?,
        );
        let output = FakeLearningSensorReviewer.review(input.clone())?;
        fixture
            .insert_learning_event(
                output.into_sensor_event(&input, created_at + Duration::seconds(1))?,
            )
            .await?;

        let projection = fixture
            .get_learning_record(candidate.record_id)
            .await?
            .expect("candidate projection should exist after sensor event");
        let accepted = fixture
            .list_learning_records(Some(LearningReviewState::Accepted), 10)
            .await?;
        let actual = (projection.review_state, accepted.len());
        let expected = (LearningReviewState::Candidate, 0usize);

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn sanitized_observation_sensor_proposal_promotes_atomically_with_accepted_summary()
    -> anyhow::Result<()> {
        let fixture = fixture_repo(28)?;
        let created_at = Utc::now();
        let request = fixture_promotion_request(&fixture, created_at).await?;

        let outcome = fixture.promote_sensor_lesson(request.clone()).await?;
        let replay = fixture.promote_sensor_lesson(request).await?;
        let events = fixture
            .run_with_connection(move |connection, wid| {
                learning_ledger_events::table
                    .filter(learning_ledger_events::workspace_id.eq(workspace_db_id(wid)))
                    .order(learning_ledger_events::event_seq.asc())
                    .load::<LearningLedgerEventRecord>(connection)
                    .map_err(Into::into)
            })
            .await?;
        let actual = (
            outcome.projection.review_state,
            outcome.projection.accepted_summary.clone(),
            outcome.review_event.event_id == replay.review_event.event_id,
            events
                .iter()
                .filter(|event| event.event_kind == LearningEventKind::PromotionAudit.to_string())
                .count(),
            events
                .iter()
                .filter(|event| event.event_kind == LearningEventKind::ReviewAccepted.to_string())
                .count(),
        );
        let expected = (
            LearningReviewState::Accepted,
            Some(
                "sanctioned_sanitized_observation:validated_counters_and_fingerprints".to_string(),
            ),
            true,
            1usize,
            1usize,
        );

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn generic_review_acceptance_is_blocked_for_sensor_derived_candidate()
    -> anyhow::Result<()> {
        let fixture = fixture_repo(29)?;
        let created_at = Utc::now();
        let request = fixture_promotion_request(&fixture, created_at).await?;
        let generic_review = LearningLedgerEvent::review(
            request.proposal().candidate_id(),
            LearningEventKind::ReviewAccepted,
            "generic unsafe acceptance",
            LearningProvenance::conversation(
                ConversationId::generate(),
                "generic-review",
                "generic-review-fingerprint",
            ),
            created_at + Duration::seconds(2),
        )?;

        let actual = fixture
            .review_learning_candidate_event(generic_review)
            .await
            .is_err();
        let expected = true;

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn sensor_promotion_rejects_stale_cursor_after_later_terminal_event() -> anyhow::Result<()>
    {
        let fixture = fixture_repo(30)?;
        let created_at = Utc::now();
        let request = fixture_promotion_request(&fixture, created_at).await?;
        let rejection = LearningLedgerEvent::review(
            request.proposal().candidate_id(),
            LearningEventKind::ReviewRejected,
            "terminal rejection wins race",
            LearningProvenance::conversation(
                ConversationId::generate(),
                "reject-review",
                "reject-review-fingerprint",
            ),
            created_at + Duration::seconds(2),
        )?;
        fixture.review_learning_candidate_event(rejection).await?;

        let actual = fixture.promote_sensor_lesson(request).await.is_err();
        let expected = true;

        assert_eq!(actual, expected);
        Ok(())
    }

    async fn fixture_promotion_request(
        fixture: &LearningRepositoryImpl,
        created_at: DateTime<Utc>,
    ) -> anyhow::Result<SensorLessonPromotionRequest> {
        let conversation_id = ConversationId::generate();
        let candidate = fixture
            .insert_learning_event(fixture_event(
                conversation_id,
                "event-promotion",
                "raw candidate summary must not be injected after sensor promotion",
                created_at,
            )?)
            .await?
            .event;
        let projection = fixture
            .get_learning_record(candidate.record_id)
            .await?
            .expect("candidate projection should exist");
        let input = LearningSensorReviewInput::from_sanitized_chat_observation(
            &projection,
            fixture_sanitized_observation().validate()?,
        );
        let output = FakeLearningSensorReviewer.review(input.clone())?;
        let sensor_event = fixture
            .insert_learning_event(
                output.into_sensor_event(&input, created_at + Duration::seconds(1))?,
            )
            .await?
            .event;
        let view = fixture
            .get_learning_event_view(sensor_event.event_id)
            .await?
            .expect("sensor event view should exist");
        let projection = fixture
            .get_learning_record(candidate.record_id)
            .await?
            .expect("candidate projection should still exist");
        let proposal = SensorLessonPromotionProposal::new(&view, &projection)?;
        SensorLessonPromotionRequest::new(proposal, created_at + Duration::seconds(2))
    }

    fn fixture_sanitized_observation() -> SanitizedChatLessonObservation {
        SanitizedChatLessonObservation::new(
            SanitizedChatObservationKind::ReviewerIdentifiedGap,
            SanitizedObservationCountBucket::Two,
            SanitizedObservationSeverity::Medium,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        )
        .unwrap()
    }

    #[tokio::test]
    async fn learning_freshness_changes_with_projection_version() -> anyhow::Result<()> {
        let fixture = fixture_repo(7)?;
        let conversation_id = ConversationId::generate();
        let before = fixture.learning_freshness(None).await?;
        fixture
            .insert_learning_event(fixture_event(
                conversation_id,
                "event-1",
                "freshness cursor",
                Utc::now(),
            )?)
            .await?;
        let after = fixture.learning_freshness(None).await?;

        let actual = after.ledger_cursor > before.ledger_cursor
            && after.projection_version > before.projection_version
            && after.review_state_fingerprint != before.review_state_fingerprint;
        let expected = true;

        assert_eq!(actual, expected);
        Ok(())
    }
}
