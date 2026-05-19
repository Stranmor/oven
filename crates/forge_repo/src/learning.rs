use std::collections::BTreeMap;
use std::str::FromStr;
use std::sync::Arc;

use chrono::{DateTime, NaiveDateTime, Utc};
use diesel::prelude::*;
use forge_domain::{
    ConversationId, LearningEventId, LearningEventKind, LearningLedgerEvent,
    LearningLedgerFreshness, LearningProvenance, LearningRecordId, LearningRecordProjection,
    LearningRedactionStatus, LearningRepository, LearningReviewState, LearningSourceKind,
    RedactedLearningSummary, SubagentTaskId, WorkspaceHash,
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
    ) -> anyhow::Result<LearningLedgerEvent> {
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
                        learning_ledger_events::created_at.eq(record.created_at),
                        learning_ledger_events::schema_version.eq(record.schema_version),
                    ))
                    .execute(connection)?;
                learning_ledger_events::table
                    .filter(learning_ledger_events::workspace_id.eq(workspace_id))
                    .filter(learning_ledger_events::idempotency_key.eq(&record.idempotency_key))
                    .first::<LearningLedgerEventRecord>(connection)?
                    .try_into_event()
            })
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
            let mut projections = project_records(records)?;
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
            let mut projections = project_records(records)?;
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
        let source_id = event.provenance.source_id()?;
        let redacted = RedactedLearningSummary::from_raw(&event.summary);
        let redaction_status = match (event.redaction_status, redacted.status) {
            (LearningRedactionStatus::Redacted, _) | (_, LearningRedactionStatus::Redacted) => {
                LearningRedactionStatus::Redacted
            }
            (LearningRedactionStatus::Clean, LearningRedactionStatus::Clean) => {
                LearningRedactionStatus::Clean
            }
        };
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
            source_event_id: event.provenance.source_event_id,
            source_fingerprint: event.provenance.source_fingerprint,
            conversation_id: event.provenance.conversation_id.map(|id| id.into_string()),
            task_id: event.provenance.task_id.map(|id| id.into_string()),
            tool_name: event.provenance.tool_name,
            eval_id: event.provenance.eval_id,
            created_at: event.created_at.naive_utc(),
            schema_version: event.schema_version,
        })
    }

    fn try_into_event(self) -> anyhow::Result<LearningLedgerEvent> {
        let event_kind = parse_event_kind(&self.event_kind)?;
        let redaction_status = parse_redaction_status(&self.redaction_status)?;
        let provenance = self.try_into_provenance()?;
        Ok(LearningLedgerEvent {
            event_id: LearningEventId::parse(self.event_id)?,
            record_id: LearningRecordId::parse(self.record_id)?,
            idempotency_key: self.idempotency_key,
            event_kind,
            summary: self.summary,
            content_fingerprint: self.content_fingerprint,
            redaction_status,
            provenance,
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
    review_state: LearningReviewState,
    redaction_status: LearningRedactionStatus,
    provenance: LearningProvenance,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    schema_version: i32,
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
                    review_state: LearningReviewState::Candidate,
                    redaction_status: event.redaction_status,
                    provenance: event.provenance,
                    created_at: event.created_at,
                    updated_at: event.created_at,
                    schema_version: event.schema_version,
                });
            }
            LearningEventKind::ReviewAccepted => {
                if let Some(projection) = projections.get_mut(&record_key) {
                    projection.review_state = LearningReviewState::Accepted;
                    projection.updated_at = event.created_at;
                }
            }
            LearningEventKind::ReviewRejected => {
                if let Some(projection) = projections.get_mut(&record_key) {
                    projection.review_state = LearningReviewState::Rejected;
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
            review_state: projection.review_state,
            redaction_status: projection.redaction_status,
            provenance: projection.provenance,
            created_at: projection.created_at,
            updated_at: projection.updated_at,
            schema_version: projection.schema_version,
        })
        .collect())
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
    use forge_domain::{LEARNING_LEDGER_SCHEMA_VERSION, RedactedLearningSummary};
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
        let actual = (left.event_id, right.event_id, records.len());
        let expected = (left.event_id, left.event_id, 1usize);

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
            .await?;
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
                "source-fingerprint-raw-secret",
            ),
            created_at: Utc::now(),
            schema_version: LEARNING_LEDGER_SCHEMA_VERSION,
        };

        let actual = match fixture.insert_learning_event(event).await {
            Ok(_) => fixture
                .list_learning_records(None, 10)
                .await?
                .iter()
                .any(|projection| projection.summary.contains("sk-")),
            Err(_) => false,
        };
        let expected = false;

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
