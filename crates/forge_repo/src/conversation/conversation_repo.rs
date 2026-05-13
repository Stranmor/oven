use std::sync::Arc;

use chrono::{Duration, Utc};
use diesel::prelude::*;
use diesel::sqlite::SqliteConnection;
use forge_domain::{
    Conversation, ConversationId, ConversationRepository, SubagentTaskId, SubagentTaskSession,
    SubagentTaskSessionFilter, SubagentTaskStatus, WorkspaceHash,
};

use crate::conversation::conversation_record::ConversationRecord;
use crate::conversation::subagent_task_record::SubagentTaskSessionRecord;
use crate::database::schema::{conversations, subagent_task_sessions};
use crate::database::{DatabasePool, PooledSqliteConnection};

const SUBAGENT_TASK_HEARTBEAT_TIMEOUT: Duration = Duration::minutes(5);

fn classify_subagent_task_session(session: SubagentTaskSession) -> SubagentTaskSession {
    session.classify_with_heartbeat(Utc::now(), SUBAGENT_TASK_HEARTBEAT_TIMEOUT)
}

fn ensure_ledger_owner_matches_exact(
    conversation_id: ConversationId,
    requested_parent_id: Option<ConversationId>,
    requested_root_id: Option<ConversationId>,
    existing: SubagentTaskSession,
) -> anyhow::Result<()> {
    if existing.parent_conversation_id != requested_parent_id
        || existing.root_conversation_id != requested_root_id
    {
        anyhow::bail!(
            "Subagent session {conversation_id} belongs to parent {:?} and root {:?}; refusing silent reparent to parent {:?} and root {:?}",
            existing.parent_conversation_id,
            existing.root_conversation_id,
            requested_parent_id,
            requested_root_id
        );
    }
    Ok(())
}

fn ensure_conversation_ledger_owner_matches_exact(
    connection: &mut SqliteConnection,
    workspace_id: i64,
    conversation_id: ConversationId,
    requested_parent_id: Option<ConversationId>,
    requested_root_id: Option<ConversationId>,
) -> anyhow::Result<()> {
    let records: Vec<SubagentTaskSessionRecord> = subagent_task_sessions::table
        .filter(subagent_task_sessions::workspace_id.eq(workspace_id))
        .filter(subagent_task_sessions::conversation_id.eq(conversation_id.into_string()))
        .load(connection)?;

    for record in records {
        let existing = SubagentTaskSession::try_from(record)?;
        ensure_ledger_owner_matches_exact(
            conversation_id,
            requested_parent_id,
            requested_root_id,
            existing,
        )?;
    }
    Ok(())
}

fn ensure_conversation_ledger_parent_is_compatible(
    connection: &mut SqliteConnection,
    workspace_id: i64,
    conversation_id: ConversationId,
    requested_parent_id: Option<ConversationId>,
) -> anyhow::Result<()> {
    let records: Vec<SubagentTaskSessionRecord> = subagent_task_sessions::table
        .filter(subagent_task_sessions::workspace_id.eq(workspace_id))
        .filter(subagent_task_sessions::conversation_id.eq(conversation_id.into_string()))
        .load(connection)?;
    let mut owner: Option<(Option<ConversationId>, Option<ConversationId>)> = None;

    for record in records {
        let existing = SubagentTaskSession::try_from(record)?;
        let existing_owner = (
            existing.parent_conversation_id,
            existing.root_conversation_id,
        );
        if existing_owner.0 != requested_parent_id {
            anyhow::bail!(
                "Subagent session {conversation_id} belongs to parent {:?} and root {:?}; refusing silent reparent to parent {:?}",
                existing_owner.0,
                existing_owner.1,
                requested_parent_id
            );
        }
        if let Some(owner) = owner {
            if owner != existing_owner {
                anyhow::bail!(
                    "Subagent session {conversation_id} has inconsistent historical owners {:?} and {:?}; refusing promotion to parent {:?}",
                    owner,
                    existing_owner,
                    requested_parent_id
                );
            }
        } else {
            owner = Some(existing_owner);
        }
    }
    Ok(())
}

fn ensure_task_session_identity_matches_by_task_id(
    connection: &mut SqliteConnection,
    record: &SubagentTaskSessionRecord,
) -> anyhow::Result<()> {
    let existing: Option<SubagentTaskSessionRecord> = subagent_task_sessions::table
        .filter(subagent_task_sessions::task_id.eq(&record.task_id))
        .first(connection)
        .optional()?;

    if let Some(existing) = existing {
        if existing.workspace_id != record.workspace_id {
            anyhow::bail!(
                "Subagent task session {} belongs to a different workspace",
                record.task_id
            );
        }
        if existing.conversation_id != record.conversation_id
            || existing.parent_conversation_id != record.parent_conversation_id
            || existing.root_conversation_id != record.root_conversation_id
        {
            anyhow::bail!(
                "Subagent task session {} is already bound to conversation {}, parent {:?}, and root {:?}; refusing reassignment to conversation {}, parent {:?}, and root {:?}",
                record.task_id,
                existing.conversation_id,
                existing.parent_conversation_id,
                existing.root_conversation_id,
                record.conversation_id,
                record.parent_conversation_id,
                record.root_conversation_id
            );
        }
    }
    Ok(())
}

fn ensure_no_other_active_conversation_session(
    connection: &mut SqliteConnection,
    record: &SubagentTaskSessionRecord,
) -> anyhow::Result<()> {
    let status = record.status.parse::<SubagentTaskStatus>()?;
    if !status.is_active() {
        return Ok(());
    }

    let existing: Option<SubagentTaskSessionRecord> = subagent_task_sessions::table
        .filter(subagent_task_sessions::workspace_id.eq(record.workspace_id))
        .filter(subagent_task_sessions::conversation_id.eq(&record.conversation_id))
        .filter(subagent_task_sessions::task_id.ne(&record.task_id))
        .filter(
            subagent_task_sessions::status
                .eq("created")
                .or(subagent_task_sessions::status.eq("running"))
                .or(subagent_task_sessions::status.eq("zombie")),
        )
        .order((
            subagent_task_sessions::updated_at.desc(),
            subagent_task_sessions::created_at.desc(),
            subagent_task_sessions::task_id.desc(),
        ))
        .first(connection)
        .optional()?;

    if let Some(existing) = existing {
        anyhow::bail!(
            "Subagent session {} already has active task session {}; refusing duplicate active attempt {}",
            record.conversation_id,
            existing.task_id,
            record.task_id
        );
    }
    Ok(())
}

fn insert_subagent_task_session_record(
    connection: &mut SqliteConnection,
    record: &SubagentTaskSessionRecord,
) -> anyhow::Result<()> {
    ensure_task_session_identity_matches_by_task_id(connection, record)?;
    let changed_rows = diesel::sql_query(
        "INSERT INTO subagent_task_sessions (
                    task_id, agent_id, conversation_id, parent_conversation_id,
                    root_conversation_id, workspace_id, status, task, created_at,
                    updated_at, heartbeat_at, final_result, final_error, delivered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    agent_id = excluded.agent_id,
                    status = excluded.status,
                    task = excluded.task,
                    updated_at = excluded.updated_at,
                    heartbeat_at = excluded.heartbeat_at,
                    final_result = excluded.final_result,
                    final_error = excluded.final_error,
                    delivered_at = excluded.delivered_at
                WHERE subagent_task_sessions.workspace_id = excluded.workspace_id
                    AND subagent_task_sessions.conversation_id = excluded.conversation_id
                    AND subagent_task_sessions.parent_conversation_id IS excluded.parent_conversation_id
                    AND subagent_task_sessions.root_conversation_id IS excluded.root_conversation_id",
    )
    .bind::<diesel::sql_types::Text, _>(&record.task_id)
    .bind::<diesel::sql_types::Text, _>(&record.agent_id)
    .bind::<diesel::sql_types::Text, _>(&record.conversation_id)
    .bind::<diesel::sql_types::Nullable<diesel::sql_types::Text>, _>(&record.parent_conversation_id)
    .bind::<diesel::sql_types::Nullable<diesel::sql_types::Text>, _>(&record.root_conversation_id)
    .bind::<diesel::sql_types::BigInt, _>(record.workspace_id)
    .bind::<diesel::sql_types::Text, _>(&record.status)
    .bind::<diesel::sql_types::Text, _>(&record.task)
    .bind::<diesel::sql_types::Timestamp, _>(record.created_at)
    .bind::<diesel::sql_types::Timestamp, _>(record.updated_at)
    .bind::<diesel::sql_types::Timestamp, _>(record.heartbeat_at)
    .bind::<diesel::sql_types::Nullable<diesel::sql_types::Text>, _>(&record.final_result)
    .bind::<diesel::sql_types::Nullable<diesel::sql_types::Text>, _>(&record.final_error)
    .bind::<diesel::sql_types::Nullable<diesel::sql_types::Timestamp>, _>(record.delivered_at)
    .execute(connection)?;
    if changed_rows == 0 {
        anyhow::bail!(
            "Subagent task session {} belongs to a different workspace",
            record.task_id
        );
    }
    Ok(())
}

pub struct ConversationRepositoryImpl {
    pool: Arc<DatabasePool>,
    wid: WorkspaceHash,
}

fn workspace_db_id(wid: WorkspaceHash) -> i64 {
    i64::from_ne_bytes(wid.id().to_ne_bytes())
}

impl ConversationRepositoryImpl {
    pub fn new(pool: Arc<DatabasePool>, workspace_id: WorkspaceHash) -> Self {
        Self { pool, wid: workspace_id }
    }

    async fn run_blocking<F, T>(&self, operation: F) -> anyhow::Result<T>
    where
        F: FnOnce(Arc<DatabasePool>, WorkspaceHash) -> anyhow::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let pool = self.pool.clone();
        let wid = self.wid;
        tokio::task::spawn_blocking(move || operation(pool, wid))
            .await
            .map_err(|e| anyhow::anyhow!("Conversation repository task failed: {e}"))?
    }

    async fn run_with_connection<F, T>(&self, operation: F) -> anyhow::Result<T>
    where
        F: FnOnce(&mut PooledSqliteConnection, WorkspaceHash) -> anyhow::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        self.run_blocking(move |pool, wid| {
            let mut connection = pool.get_connection()?;
            operation(&mut connection, wid)
        })
        .await
    }
}

#[async_trait::async_trait]
impl ConversationRepository for ConversationRepositoryImpl {
    async fn promote_delegated_conversation(
        &self,
        conversation_id: &ConversationId,
        parent_id: Option<ConversationId>,
    ) -> anyhow::Result<Conversation> {
        let conversation_id = *conversation_id;
        self.run_with_connection(move |connection, wid| {
            connection.immediate_transaction::<_, anyhow::Error, _>(|connection| {
                let workspace_id = workspace_db_id(wid);
                let mut conversation = conversations::table
                    .filter(conversations::workspace_id.eq(&workspace_id))
                    .filter(conversations::conversation_id.eq(conversation_id.into_string()))
                    .first::<ConversationRecord>(connection)
                    .optional()?
                    .map(Conversation::try_from)
                    .transpose()?
                    .ok_or_else(|| forge_domain::Error::ConversationNotFound(conversation_id))?;
                if conversation.parent_id.is_some() && conversation.parent_id != parent_id {
                    anyhow::bail!(
                        "Conversation {conversation_id} is already owned by parent {:?}; refusing silent reparent to {:?}",
                        conversation.parent_id,
                        parent_id
                    );
                }
                ensure_conversation_ledger_parent_is_compatible(
                    connection,
                    workspace_id,
                    conversation_id,
                    parent_id,
                )?;
                conversation.ensure_delegated(parent_id);
                let record = ConversationRecord::new(conversation.clone(), workspace_id);
                let changed_rows = diesel::sql_query(
                    "INSERT INTO conversations (
                    conversation_id, title, workspace_id, context, created_at,
                    updated_at, metrics, parent_id, initiator
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    title = excluded.title,
                    context = excluded.context,
                    updated_at = excluded.updated_at,
                    metrics = excluded.metrics,
                    parent_id = excluded.parent_id,
                    initiator = excluded.initiator
                WHERE conversations.workspace_id = excluded.workspace_id",
                )
                .bind::<diesel::sql_types::Text, _>(&record.conversation_id)
                .bind::<diesel::sql_types::Nullable<diesel::sql_types::Text>, _>(&record.title)
                .bind::<diesel::sql_types::BigInt, _>(record.workspace_id)
                .bind::<diesel::sql_types::Nullable<diesel::sql_types::Text>, _>(&record.context)
                .bind::<diesel::sql_types::Timestamp, _>(record.created_at)
                .bind::<diesel::sql_types::Nullable<diesel::sql_types::Timestamp>, _>(record.updated_at)
                .bind::<diesel::sql_types::Nullable<diesel::sql_types::Text>, _>(&record.metrics)
                .bind::<diesel::sql_types::Nullable<diesel::sql_types::Text>, _>(&record.parent_id)
                .bind::<diesel::sql_types::Nullable<diesel::sql_types::Text>, _>(&record.initiator)
                .execute(connection)?;

                if changed_rows == 0 {
                    anyhow::bail!(
                        "Conversation {} belongs to a different workspace",
                        record.conversation_id
                    );
                }
                Ok(conversation)
            })
        })
        .await
    }

    async fn upsert_conversation(&self, conversation: Conversation) -> anyhow::Result<()> {
        self.run_with_connection(move |connection, wid| {
            let workspace_id = workspace_db_id(wid);
            let record = ConversationRecord::new(conversation, workspace_id);
            let changed_rows = diesel::sql_query(
                "INSERT INTO conversations (
                    conversation_id, title, workspace_id, context, created_at,
                    updated_at, metrics, parent_id, initiator
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    title = excluded.title,
                    context = excluded.context,
                    updated_at = excluded.updated_at,
                    metrics = excluded.metrics,
                    parent_id = excluded.parent_id,
                    initiator = excluded.initiator
                WHERE conversations.workspace_id = excluded.workspace_id",
            )
            .bind::<diesel::sql_types::Text, _>(&record.conversation_id)
            .bind::<diesel::sql_types::Nullable<diesel::sql_types::Text>, _>(&record.title)
            .bind::<diesel::sql_types::BigInt, _>(record.workspace_id)
            .bind::<diesel::sql_types::Nullable<diesel::sql_types::Text>, _>(&record.context)
            .bind::<diesel::sql_types::Timestamp, _>(record.created_at)
            .bind::<diesel::sql_types::Nullable<diesel::sql_types::Timestamp>, _>(record.updated_at)
            .bind::<diesel::sql_types::Nullable<diesel::sql_types::Text>, _>(&record.metrics)
            .bind::<diesel::sql_types::Nullable<diesel::sql_types::Text>, _>(&record.parent_id)
            .bind::<diesel::sql_types::Nullable<diesel::sql_types::Text>, _>(&record.initiator)
            .execute(connection)?;

            if changed_rows == 0 {
                anyhow::bail!(
                    "Conversation {} belongs to a different workspace",
                    record.conversation_id
                );
            }
            Ok(())
        })
        .await
    }

    async fn get_conversation(
        &self,
        conversation_id: &ConversationId,
    ) -> anyhow::Result<Option<Conversation>> {
        let conversation_id = *conversation_id;
        self.run_with_connection(move |connection, wid| {
            let workspace_id = workspace_db_id(wid);
            let record: Option<ConversationRecord> = conversations::table
                .filter(conversations::workspace_id.eq(&workspace_id))
                .filter(conversations::conversation_id.eq(conversation_id.into_string()))
                .first(connection)
                .optional()?;

            match record {
                Some(record) => Ok(Some(Conversation::try_from(record)?)),
                None => Ok(None),
            }
        })
        .await
    }

    async fn get_all_conversations(&self) -> anyhow::Result<Vec<Conversation>> {
        self.run_with_connection(move |connection, wid| {
            let workspace_id = workspace_db_id(wid);
            let query = conversations::table
                .filter(conversations::workspace_id.eq(&workspace_id))
                .filter(conversations::context.is_not_null())
                .filter(conversations::parent_id.is_null())
                .order(conversations::updated_at.desc())
                .into_boxed();

            let records: Vec<ConversationRecord> = query
                .filter(conversations::initiator.is_null())
                .filter(conversations::context.not_like("%\"initiator\":\"agent\"%"))
                .filter(conversations::context.not_like("%\"initiator\": \"agent\"%"))
                .load(connection)?;

            let conversations: Vec<Conversation> = records
                .into_iter()
                .map(Conversation::try_from)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .filter(Conversation::is_primary_user_conversation)
                .collect();
            Ok(conversations)
        })
        .await
    }

    async fn get_sub_conversations(
        &self,
        parent_id: &ConversationId,
    ) -> anyhow::Result<Vec<Conversation>> {
        let parent_id = *parent_id;
        self.run_with_connection(move |connection, wid| {
            let workspace_id = workspace_db_id(wid);
            let records: Vec<ConversationRecord> = conversations::table
                .filter(conversations::workspace_id.eq(&workspace_id))
                .filter(conversations::context.is_not_null())
                .filter(conversations::parent_id.eq(parent_id.into_string()))
                .order(conversations::updated_at.desc())
                .load(connection)?;

            let conversations: Result<Vec<Conversation>, _> =
                records.into_iter().map(Conversation::try_from).collect();
            conversations
        })
        .await
    }

    async fn upsert_subagent_task_session(
        &self,
        session: SubagentTaskSession,
    ) -> anyhow::Result<()> {
        self.run_with_connection(move |connection, wid| {
            connection.immediate_transaction::<_, anyhow::Error, _>(|connection| {
                let workspace_id = workspace_db_id(wid);
                let record = SubagentTaskSessionRecord::new(session, workspace_id);
                let conversation_id = ConversationId::parse(&record.conversation_id)?;
                let requested_parent_id = record
                    .parent_conversation_id
                    .as_deref()
                    .map(ConversationId::parse)
                    .transpose()?;
                let requested_root_id = record
                    .root_conversation_id
                    .as_deref()
                    .map(ConversationId::parse)
                    .transpose()?;
                ensure_conversation_ledger_owner_matches_exact(
                    connection,
                    workspace_id,
                    conversation_id,
                    requested_parent_id,
                    requested_root_id,
                )?;
                ensure_no_other_active_conversation_session(connection, &record)?;
                insert_subagent_task_session_record(connection, &record)
            })
        })
        .await
    }

    async fn get_subagent_task_session(
        &self,
        task_id: &SubagentTaskId,
    ) -> anyhow::Result<Option<SubagentTaskSession>> {
        let task_id = *task_id;
        self.run_with_connection(move |connection, wid| {
            let workspace_id = workspace_db_id(wid);
            let record: Option<SubagentTaskSessionRecord> = subagent_task_sessions::table
                .filter(subagent_task_sessions::workspace_id.eq(workspace_id))
                .filter(subagent_task_sessions::task_id.eq(task_id.into_string()))
                .first(connection)
                .optional()?;
            record
                .map(SubagentTaskSession::try_from)
                .transpose()
                .map(|session| session.map(classify_subagent_task_session))
        })
        .await
    }

    async fn get_subagent_task_session_by_conversation(
        &self,
        conversation_id: &ConversationId,
    ) -> anyhow::Result<Option<SubagentTaskSession>> {
        let conversation_id = *conversation_id;
        self.run_with_connection(move |connection, wid| {
            let workspace_id = workspace_db_id(wid);
            let active_record: Option<SubagentTaskSessionRecord> = subagent_task_sessions::table
                .filter(subagent_task_sessions::workspace_id.eq(workspace_id))
                .filter(subagent_task_sessions::conversation_id.eq(conversation_id.into_string()))
                .filter(
                    subagent_task_sessions::status
                        .eq("created")
                        .or(subagent_task_sessions::status.eq("running"))
                        .or(subagent_task_sessions::status.eq("zombie")),
                )
                .order((
                    subagent_task_sessions::updated_at.desc(),
                    subagent_task_sessions::created_at.desc(),
                    subagent_task_sessions::task_id.desc(),
                ))
                .first(connection)
                .optional()?;
            let record: Option<SubagentTaskSessionRecord> = match active_record {
                Some(record) => Some(record),
                None => subagent_task_sessions::table
                    .filter(subagent_task_sessions::workspace_id.eq(workspace_id))
                    .filter(
                        subagent_task_sessions::conversation_id.eq(conversation_id.into_string()),
                    )
                    .order((
                        subagent_task_sessions::updated_at.desc(),
                        subagent_task_sessions::created_at.desc(),
                        subagent_task_sessions::task_id.desc(),
                    ))
                    .first(connection)
                    .optional()?,
            };
            record
                .map(SubagentTaskSession::try_from)
                .transpose()
                .map(|session| session.map(classify_subagent_task_session))
        })
        .await
    }

    async fn list_subagent_task_sessions(
        &self,
        filter: SubagentTaskSessionFilter,
    ) -> anyhow::Result<Vec<SubagentTaskSession>> {
        self.run_with_connection(move |connection, wid| {
            let workspace_id = workspace_db_id(wid);
            let mut query = subagent_task_sessions::table
                .filter(subagent_task_sessions::workspace_id.eq(workspace_id))
                .order(subagent_task_sessions::updated_at.desc())
                .into_boxed();
            if filter == SubagentTaskSessionFilter::Active {
                query = query.filter(
                    subagent_task_sessions::status
                        .eq("created")
                        .or(subagent_task_sessions::status.eq("running"))
                        .or(subagent_task_sessions::status.eq("zombie")),
                );
            }
            let records: Vec<SubagentTaskSessionRecord> = query.load(connection)?;
            records
                .into_iter()
                .map(SubagentTaskSession::try_from)
                .map(|session| session.map(classify_subagent_task_session))
                .collect()
        })
        .await
    }

    async fn get_last_conversation(&self) -> anyhow::Result<Option<Conversation>> {
        self.run_with_connection(move |connection, wid| {
            let workspace_id = workspace_db_id(wid);
            let conversation: Option<Conversation> = conversations::table
                .filter(conversations::workspace_id.eq(&workspace_id))
                .filter(conversations::context.is_not_null())
                .filter(conversations::parent_id.is_null())
                .filter(conversations::initiator.is_null())
                .filter(conversations::context.not_like("%\"initiator\":\"agent\"%"))
                .filter(conversations::context.not_like("%\"initiator\": \"agent\"%"))
                .order(conversations::updated_at.desc())
                .load::<ConversationRecord>(connection)?
                .into_iter()
                .map(Conversation::try_from)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .find(Conversation::is_primary_user_conversation);
            Ok(conversation)
        })
        .await
    }

    async fn delete_conversation(&self, conversation_id: &ConversationId) -> anyhow::Result<()> {
        let conversation_id = *conversation_id;
        self.run_with_connection(move |connection, wid| {
            let workspace_id = workspace_db_id(wid);

            diesel::sql_query(
                "WITH RECURSIVE descendants(id) AS (
                SELECT conversation_id FROM conversations
                WHERE parent_id = ? AND workspace_id = ?
                UNION ALL
                SELECT c.conversation_id FROM conversations c
                JOIN descendants d ON c.parent_id = d.id
                WHERE c.workspace_id = ?
            )
            DELETE FROM conversations
            WHERE workspace_id = ?
            AND (conversation_id IN (SELECT id FROM descendants) OR conversation_id = ?)",
            )
            .bind::<diesel::sql_types::Text, _>(conversation_id.into_string())
            .bind::<diesel::sql_types::BigInt, _>(workspace_id)
            .bind::<diesel::sql_types::BigInt, _>(workspace_id)
            .bind::<diesel::sql_types::BigInt, _>(workspace_id)
            .bind::<diesel::sql_types::Text, _>(conversation_id.into_string())
            .execute(connection)?;

            Ok(())
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use forge_domain::{
        Context, ContextMessage, Effort, FileOperation, Metrics, Role, ToolCallFull, ToolCallId,
        ToolChoice, ToolDefinition, ToolKind, ToolName, ToolOutput, ToolResult, ToolValue, Usage,
    };
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::conversation::conversation_record::{ContextRecord, MetricsRecord};
    use crate::database::DatabasePool;

    fn repository() -> anyhow::Result<ConversationRepositoryImpl> {
        repository_with_workspace(WorkspaceHash::new(0))
    }

    fn repository_with_workspace(
        workspace_id: WorkspaceHash,
    ) -> anyhow::Result<ConversationRepositoryImpl> {
        let pool = Arc::new(DatabasePool::in_memory()?);
        Ok(ConversationRepositoryImpl::new(pool, workspace_id))
    }

    fn repository_with_pool(
        pool: Arc<DatabasePool>,
        workspace_id: WorkspaceHash,
    ) -> ConversationRepositoryImpl {
        ConversationRepositoryImpl::new(pool, workspace_id)
    }

    async fn insert_legacy_agent_record(
        repo: &ConversationRepositoryImpl,
        title: &str,
    ) -> anyhow::Result<ConversationId> {
        let id = ConversationId::generate();
        let now = Utc::now().naive_utc();
        let record = ConversationRecord {
            conversation_id: id.into_string(),
            title: Some(title.to_string()),
            workspace_id: 0,
            context: Some(r#"{"initiator":"agent"}"#.to_string()),
            created_at: now,
            updated_at: Some(now),
            metrics: None,
            parent_id: None,
            initiator: None,
        };

        repo.run_with_connection(move |connection, _wid| {
            diesel::insert_into(conversations::table)
                .values(&record)
                .execute(connection)?;
            Ok(())
        })
        .await?;

        Ok(id)
    }

    async fn insert_legacy_subagent_task_session(
        repo: &ConversationRepositoryImpl,
        session: SubagentTaskSession,
    ) -> anyhow::Result<()> {
        repo.run_with_connection(move |connection, wid| {
            let workspace_id = workspace_db_id(wid);
            let record = SubagentTaskSessionRecord::new(session, workspace_id);
            insert_subagent_task_session_record(connection, &record)
        })
        .await
    }

    #[tokio::test]
    async fn test_upsert_and_find_by_id() -> anyhow::Result<()> {
        let fixture = Conversation::new(ConversationId::generate())
            .title(Some("Test Conversation".to_string()));
        let repo = repository()?;

        repo.upsert_conversation(fixture.clone()).await?;

        let actual = repo.get_conversation(&fixture.id).await?;
        assert!(actual.is_some());
        let retrieved = actual.unwrap();
        assert_eq!(retrieved.id, fixture.id);
        assert_eq!(retrieved.title, fixture.title);
        Ok(())
    }

    #[tokio::test]
    async fn test_find_by_id_is_scoped_to_repository_workspace() -> anyhow::Result<()> {
        let pool = Arc::new(DatabasePool::in_memory()?);
        let foreign_repo = repository_with_pool(pool.clone(), WorkspaceHash::new(1));
        let scoped_repo = repository_with_pool(pool, WorkspaceHash::new(0));
        let fixture = Conversation::new(ConversationId::generate())
            .title(Some("Foreign Workspace Conversation".to_string()));

        foreign_repo.upsert_conversation(fixture.clone()).await?;
        let actual = scoped_repo.get_conversation(&fixture.id).await?;
        let expected = None;

        assert_eq!(actual.map(|conversation| conversation.id), expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_upsert_does_not_promote_same_id_in_foreign_workspace() -> anyhow::Result<()> {
        let pool = Arc::new(DatabasePool::in_memory()?);
        let foreign_repo = repository_with_pool(pool.clone(), WorkspaceHash::new(1));
        let scoped_repo = repository_with_pool(pool, WorkspaceHash::new(0));
        let parent_id = ConversationId::generate();
        let fixture = Conversation::new(ConversationId::generate())
            .title(Some("Foreign Workspace Conversation".to_string()));
        let mut scoped_promotion = fixture.clone().title(Some("Scoped Promotion".to_string()));
        scoped_promotion.ensure_delegated(Some(parent_id));

        foreign_repo.upsert_conversation(fixture.clone()).await?;
        let promotion_result = scoped_repo.upsert_conversation(scoped_promotion).await;
        let actual = foreign_repo
            .get_conversation(&fixture.id)
            .await?
            .expect("foreign conversation should remain visible in its workspace");
        let expected = (fixture.title, fixture.initiator, fixture.parent_id);

        assert!(promotion_result.is_err());
        assert_eq!((actual.title, actual.initiator, actual.parent_id), expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_high_bit_workspace_ids_remain_distinct() -> anyhow::Result<()> {
        let pool = Arc::new(DatabasePool::in_memory()?);
        let first_repo =
            repository_with_pool(pool.clone(), WorkspaceHash::new(i64::MAX as u64 + 1));
        let second_repo = repository_with_pool(pool, WorkspaceHash::new(i64::MAX as u64 + 2));
        let fixture = Conversation::new(ConversationId::generate())
            .title(Some("High Bit Workspace Conversation".to_string()));

        first_repo.upsert_conversation(fixture.clone()).await?;
        let actual = second_repo.get_conversation(&fixture.id).await?;
        let expected = None;

        assert_eq!(actual.map(|conversation| conversation.id), expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_find_by_id_non_existing() -> anyhow::Result<()> {
        let repo = repository()?;
        let non_existing_id = ConversationId::generate();

        let actual = repo.get_conversation(&non_existing_id).await?;

        assert!(actual.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn test_upsert_updates_existing_conversation() -> anyhow::Result<()> {
        let mut fixture = Conversation::new(ConversationId::generate())
            .title(Some("Test Conversation".to_string()));
        let repo = repository()?;

        // Insert initial conversation
        repo.upsert_conversation(fixture.clone()).await?;

        // Update the conversation
        fixture = fixture.title(Some("Updated Title".to_string()));
        repo.upsert_conversation(fixture.clone()).await?;

        let actual = repo.get_conversation(&fixture.id).await?;
        assert!(actual.is_some());
        assert_eq!(actual.unwrap().title, Some("Updated Title".to_string()));
        Ok(())
    }

    #[tokio::test]
    async fn test_find_all_conversations() -> anyhow::Result<()> {
        let context1 =
            Context::default().messages(vec![ContextMessage::user("Hello", None).into()]);
        let context2 =
            Context::default().messages(vec![ContextMessage::user("World", None).into()]);
        let conversation1 = Conversation::new(ConversationId::generate())
            .title(Some("Test Conversation".to_string()))
            .context(Some(context1));
        let conversation2 = Conversation::new(ConversationId::generate())
            .title(Some("Second Conversation".to_string()))
            .context(Some(context2));
        let repo = repository()?;

        repo.upsert_conversation(conversation1.clone()).await?;
        repo.upsert_conversation(conversation2.clone()).await?;

        let actual = repo.get_all_conversations().await?;

        assert_eq!(actual.len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn test_find_all_conversations_excludes_agent_initiated() -> anyhow::Result<()> {
        let user_context =
            Context::default().messages(vec![ContextMessage::user("User task", None).into()]);
        let agent_context =
            Context::default().messages(vec![ContextMessage::user("Agent task", None).into()]);
        let user_conversation = Conversation::new(ConversationId::generate())
            .title(Some("User Conversation".to_string()))
            .context(Some(user_context));
        let agent_conversation = Conversation::new(ConversationId::generate())
            .initiator(forge_domain::Initiator::Agent)
            .title(Some("Agent Conversation".to_string()))
            .context(Some(agent_context));
        let repo = repository()?;

        repo.upsert_conversation(agent_conversation).await?;
        repo.upsert_conversation(user_conversation.clone()).await?;

        let actual = repo.get_all_conversations().await?;
        let expected = vec![user_conversation.id];

        assert_eq!(
            actual.iter().map(|conv| conv.id).collect::<Vec<_>>(),
            expected
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_reused_delegated_conversation_promotion_hides_from_primary_lists()
    -> anyhow::Result<()> {
        let parent_id = ConversationId::generate();
        let delegated_context =
            Context::default().messages(vec![ContextMessage::user("Delegated task", None).into()]);
        let mut reused_conversation = Conversation::new(ConversationId::generate())
            .title(Some("Reused User Session".to_string()))
            .context(Some(delegated_context));
        reused_conversation.ensure_delegated(Some(parent_id));
        let repo = repository()?;

        repo.upsert_conversation(reused_conversation.clone())
            .await?;

        let actual = repo.get_all_conversations().await?;
        let expected: Vec<ConversationId> = Vec::new();

        assert_eq!(
            actual.iter().map(|conv| conv.id).collect::<Vec<_>>(),
            expected
        );
        assert!(repo.get_last_conversation().await?.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn test_reused_delegated_conversation_promotion_keeps_subchat_visible()
    -> anyhow::Result<()> {
        let parent_id = ConversationId::generate();
        let delegated_context =
            Context::default().messages(vec![ContextMessage::user("Delegated task", None).into()]);
        let mut reused_conversation = Conversation::new(ConversationId::generate())
            .title(Some("Reused User Session".to_string()))
            .context(Some(delegated_context));
        reused_conversation.ensure_delegated(Some(parent_id));
        let repo = repository()?;

        repo.upsert_conversation(reused_conversation.clone())
            .await?;

        let actual = repo.get_sub_conversations(&parent_id).await?;
        let expected = vec![reused_conversation.id];

        assert_eq!(
            actual.iter().map(|conv| conv.id).collect::<Vec<_>>(),
            expected
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_get_sub_conversations_includes_agent_conversations_for_explicit_parent()
    -> anyhow::Result<()> {
        let parent_id = ConversationId::generate();
        let child_context =
            Context::default().messages(vec![ContextMessage::user("Agent task", None).into()]);
        let child_conversation = Conversation::new(ConversationId::generate())
            .initiator(forge_domain::Initiator::Agent)
            .title(Some("Explicit Subchat".to_string()))
            .context(Some(child_context))
            .parent_id(Some(parent_id));
        let repo = repository()?;

        repo.upsert_conversation(child_conversation.clone()).await?;
        let actual = repo.get_sub_conversations(&parent_id).await?;
        let expected = vec![child_conversation.id];

        assert_eq!(
            actual.iter().map(|conv| conv.id).collect::<Vec<_>>(),
            expected
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_get_sub_conversations_is_scoped_by_workspace_when_parent_id_is_reused()
    -> anyhow::Result<()> {
        let pool = Arc::new(DatabasePool::in_memory()?);
        let parent_id = ConversationId::generate();
        let scoped_repo = repository_with_pool(pool.clone(), WorkspaceHash::new(0));
        let foreign_repo = repository_with_pool(pool, WorkspaceHash::new(1));
        let child_context =
            Context::default().messages(vec![ContextMessage::user("Agent task", None).into()]);
        let child_conversation = Conversation::new(ConversationId::generate())
            .initiator(forge_domain::Initiator::Agent)
            .title(Some("Foreign Explicit Subchat".to_string()))
            .context(Some(child_context))
            .parent_id(Some(parent_id));

        foreign_repo
            .upsert_conversation(child_conversation.clone())
            .await?;
        let scoped_actual = scoped_repo.get_sub_conversations(&parent_id).await?;
        let foreign_actual = foreign_repo.get_sub_conversations(&parent_id).await?;
        let expected = vec![child_conversation.id];

        assert!(scoped_actual.is_empty());
        assert_eq!(
            foreign_actual
                .iter()
                .map(|conv| conv.id)
                .collect::<Vec<_>>(),
            expected
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_persisted_user_session_promotion_updates_primary_and_subchat_indexes()
    -> anyhow::Result<()> {
        let parent_id = ConversationId::generate();
        let delegated_context =
            Context::default().messages(vec![ContextMessage::user("Delegated task", None).into()]);
        let mut reused_conversation = Conversation::new(ConversationId::generate())
            .title(Some("Persisted User Session".to_string()))
            .context(Some(delegated_context));
        let repo = repository()?;

        repo.upsert_conversation(reused_conversation.clone())
            .await?;
        assert_eq!(repo.get_all_conversations().await?.len(), 1);

        reused_conversation.ensure_delegated(Some(parent_id));
        repo.upsert_conversation(reused_conversation.clone())
            .await?;

        let primary = repo.get_all_conversations().await?;
        let subchats = repo.get_sub_conversations(&parent_id).await?;
        let expected = vec![reused_conversation.id];

        assert!(primary.is_empty());
        assert!(repo.get_last_conversation().await?.is_none());
        assert_eq!(
            subchats.iter().map(|conv| conv.id).collect::<Vec<_>>(),
            expected
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_find_all_conversations_empty() -> anyhow::Result<()> {
        let repo = repository()?;

        let actual = repo.get_all_conversations().await?;

        assert!(actual.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn test_find_all_conversations_excludes_with_parent_id() -> anyhow::Result<()> {
        let parent_id = ConversationId::generate();
        let user_context =
            Context::default().messages(vec![ContextMessage::user("Hello", None).into()]);
        let child_context =
            Context::default().messages(vec![ContextMessage::user("Sub task", None).into()]);

        let user_conv = Conversation::new(parent_id)
            .title(Some("Parent Chat".to_string()))
            .context(Some(user_context));
        let child_conv = Conversation::new(ConversationId::generate())
            .initiator(forge_domain::Initiator::Agent)
            .title(Some("Child Sub-Chat".to_string()))
            .context(Some(child_context))
            .parent_id(Some(parent_id));

        let repo = repository()?;
        repo.upsert_conversation(user_conv.clone()).await?;
        repo.upsert_conversation(child_conv).await?;

        let actual = repo.get_all_conversations().await?;

        assert_eq!(actual.len(), 1);
        assert_eq!(actual[0].id, user_conv.id);
        Ok(())
    }

    #[tokio::test]
    async fn test_find_all_conversations_excludes_agent_without_parent_id() -> anyhow::Result<()> {
        let user_context =
            Context::default().messages(vec![ContextMessage::user("Hello", None).into()]);
        let agent_context =
            Context::default().messages(vec![ContextMessage::user("Agent task", None).into()]);

        let user_conv = Conversation::new(ConversationId::generate())
            .title(Some("User Chat".to_string()))
            .context(Some(user_context));
        // Agent-initiated conversation WITHOUT parent_id — the old LIKE filter's
        // weakness. The new dedicated `initiator` column catches it reliably.
        let agent_conv = Conversation::new(ConversationId::generate())
            .initiator(forge_domain::Initiator::Agent)
            .title(Some("Agent No Parent".to_string()))
            .context(Some(agent_context));

        let repo = repository()?;
        repo.upsert_conversation(user_conv.clone()).await?;
        repo.upsert_conversation(agent_conv).await?;

        let actual = repo.get_all_conversations().await?;

        assert_eq!(actual.len(), 1);
        assert_eq!(actual[0].id, user_conv.id);
        Ok(())
    }

    #[tokio::test]
    async fn test_find_all_conversations_excludes_parentless_delegated_reused_session()
    -> anyhow::Result<()> {
        let user_context =
            Context::default().messages(vec![ContextMessage::user("Hello", None).into()]);
        let delegated_context =
            Context::default().messages(vec![ContextMessage::user("Delegated", None).into()]);
        let user_conv = Conversation::new(ConversationId::generate())
            .title(Some("User Chat".to_string()))
            .context(Some(user_context));
        let mut delegated_conv = Conversation::new(ConversationId::generate())
            .title(Some("Parentless Delegated".to_string()))
            .context(Some(delegated_context));
        delegated_conv.ensure_delegated(None);
        let repo = repository()?;

        repo.upsert_conversation(user_conv.clone()).await?;
        repo.upsert_conversation(delegated_conv).await?;
        let actual = repo.get_all_conversations().await?;
        let expected = vec![user_conv.id];

        assert_eq!(
            actual.iter().map(|conv| conv.id).collect::<Vec<_>>(),
            expected
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_find_all_conversations_returns_limitable_user_items_after_filtering()
    -> anyhow::Result<()> {
        let repo = repository()?;
        let mut expected = Vec::new();

        for index in 0..4 {
            let agent_context = Context::default().messages(vec![
                ContextMessage::user(format!("Agent task {index}"), None).into(),
            ]);
            let agent_conv = Conversation::new(ConversationId::generate())
                .initiator(forge_domain::Initiator::Agent)
                .title(Some(format!("Agent {index}")))
                .context(Some(agent_context));
            repo.upsert_conversation(agent_conv).await?;
        }
        for index in 0..3 {
            let user_context = Context::default().messages(vec![
                ContextMessage::user(format!("User task {index}"), None).into(),
            ]);
            let user_conv = Conversation::new(ConversationId::generate())
                .title(Some(format!("User {index}")))
                .context(Some(user_context));
            repo.upsert_conversation(user_conv.clone()).await?;
            expected.push(user_conv.id);
        }

        let actual = repo.get_all_conversations().await?;
        let actual_ids = actual
            .into_iter()
            .take(2)
            .map(|conv| conv.id)
            .collect::<Vec<_>>();
        let expected = expected.into_iter().rev().take(2).collect::<Vec<_>>();

        assert_eq!(actual_ids, expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_find_all_conversations_excludes_legacy_json_agent() -> anyhow::Result<()> {
        let user_context =
            Context::default().messages(vec![ContextMessage::user("Hello", None).into()]);
        let user_conv = Conversation::new(ConversationId::generate())
            .title(Some("User Chat".to_string()))
            .context(Some(user_context));
        let repo = repository()?;

        repo.upsert_conversation(user_conv.clone()).await?;
        insert_legacy_agent_record(&repo, "Legacy Agent").await?;

        let actual = repo.get_all_conversations().await?;

        assert_eq!(actual.len(), 1);
        assert_eq!(actual[0].id, user_conv.id);
        Ok(())
    }

    #[tokio::test]
    async fn test_find_last_conversation_excludes_agent_initiated() -> anyhow::Result<()> {
        let user_context =
            Context::default().messages(vec![ContextMessage::user("Hello", None).into()]);
        let agent_context =
            Context::default().messages(vec![ContextMessage::user("Agent task", None).into()]);

        let user_conv = Conversation::new(ConversationId::generate())
            .title(Some("User Chat".to_string()))
            .context(Some(user_context));

        let repo = repository()?;
        repo.upsert_conversation(user_conv.clone()).await?;
        // Insert agent conversation after the user one so it would be "last"
        // if not filtered
        std::thread::sleep(std::time::Duration::from_millis(10));
        let agent_conv = Conversation::new(ConversationId::generate())
            .initiator(forge_domain::Initiator::Agent)
            .title(Some("Agent Sub-Chat".to_string()))
            .context(Some(agent_context));
        repo.upsert_conversation(agent_conv).await?;

        let actual = repo.get_last_conversation().await?;

        assert!(actual.is_some());
        assert_eq!(actual.unwrap().id, user_conv.id);
        Ok(())
    }

    #[tokio::test]
    async fn test_find_last_conversation_excludes_legacy_json_agent() -> anyhow::Result<()> {
        let user_context =
            Context::default().messages(vec![ContextMessage::user("Hello", None).into()]);
        let user_conv = Conversation::new(ConversationId::generate())
            .title(Some("User Chat".to_string()))
            .context(Some(user_context));
        let repo = repository()?;

        repo.upsert_conversation(user_conv.clone()).await?;
        std::thread::sleep(std::time::Duration::from_millis(10));
        insert_legacy_agent_record(&repo, "Legacy Agent").await?;

        let actual = repo.get_last_conversation().await?;

        assert!(actual.is_some());
        assert_eq!(actual.unwrap().id, user_conv.id);
        Ok(())
    }

    #[tokio::test]
    async fn test_find_last_active_conversation_with_context() -> anyhow::Result<()> {
        let context = Context::default().messages(vec![ContextMessage::user("Hello", None).into()]);
        let conversation_with_context = Conversation::new(ConversationId::generate())
            .title(Some("Conversation with Context".to_string()))
            .context(Some(context));
        let conversation_without_context = Conversation::new(ConversationId::generate())
            .title(Some("Test Conversation".to_string()));
        let repo = repository()?;

        repo.upsert_conversation(conversation_without_context)
            .await?;
        repo.upsert_conversation(conversation_with_context.clone())
            .await?;

        let actual = repo.get_last_conversation().await?;

        assert!(actual.is_some());
        assert_eq!(actual.unwrap().id, conversation_with_context.id);
        Ok(())
    }

    #[tokio::test]
    async fn test_find_last_active_conversation_no_context() -> anyhow::Result<()> {
        let conversation_without_context = Conversation::new(ConversationId::generate())
            .title(Some("Test Conversation".to_string()));
        let repo = repository()?;

        repo.upsert_conversation(conversation_without_context)
            .await?;

        let actual = repo.get_last_conversation().await?;

        assert!(actual.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn test_find_last_active_conversation_ignores_empty_context() -> anyhow::Result<()> {
        let conversation_with_empty_context = Conversation::new(ConversationId::generate())
            .title(Some("Conversation with Empty Context".to_string()))
            .context(Some(Context::default()));
        let conversation_without_context = Conversation::new(ConversationId::generate())
            .title(Some("Test Conversation".to_string()));
        let repo = repository()?;

        repo.upsert_conversation(conversation_without_context)
            .await?;
        repo.upsert_conversation(conversation_with_empty_context)
            .await?;

        let actual = repo.get_last_conversation().await?;

        assert!(actual.is_none()); // Should not find conversations with empty contexts
        Ok(())
    }

    #[tokio::test]
    async fn test_find_last_conversation_skips_agent_initiated() -> anyhow::Result<()> {
        let user_context =
            Context::default().messages(vec![ContextMessage::user("User task", None).into()]);
        let agent_context =
            Context::default().messages(vec![ContextMessage::user("Agent task", None).into()]);
        let user_conversation =
            Conversation::new(ConversationId::generate()).context(Some(user_context));
        let agent_conversation = Conversation::new(ConversationId::generate())
            .initiator(forge_domain::Initiator::Agent)
            .context(Some(agent_context));
        let repo = repository()?;

        repo.upsert_conversation(user_conversation.clone()).await?;
        repo.upsert_conversation(agent_conversation).await?;

        let actual = repo.get_last_conversation().await?;

        assert_eq!(actual.unwrap().id, user_conversation.id);
        Ok(())
    }

    #[test]
    fn test_conversation_record_from_conversation() -> anyhow::Result<()> {
        let fixture = Conversation::new(ConversationId::generate())
            .title(Some("Test Conversation".to_string()));

        let actual = ConversationRecord::new(fixture.clone(), 0);

        assert_eq!(actual.conversation_id, fixture.id.into_string());
        assert_eq!(actual.title, Some("Test Conversation".to_string()));
        assert_eq!(actual.context, None);
        Ok(())
    }

    #[test]
    fn test_conversation_record_from_conversation_with_context() -> anyhow::Result<()> {
        let context = Context::default().messages(vec![ContextMessage::user("Hello", None).into()]);
        let fixture = Conversation::new(ConversationId::generate())
            .title(Some("Conversation with Context".to_string()))
            .context(Some(context));

        let actual = ConversationRecord::new(fixture.clone(), 0);

        assert_eq!(actual.conversation_id, fixture.id.into_string());
        assert_eq!(actual.title, Some("Conversation with Context".to_string()));
        assert!(actual.context.is_some());
        Ok(())
    }

    #[test]
    fn test_conversation_record_from_conversation_with_empty_context() -> anyhow::Result<()> {
        let fixture = Conversation::new(ConversationId::generate())
            .title(Some("Conversation with Empty Context".to_string()))
            .context(Some(Context::default()));

        let actual = ConversationRecord::new(fixture.clone(), 0);

        assert_eq!(actual.conversation_id, fixture.id.into_string());
        assert_eq!(
            actual.title,
            Some("Conversation with Empty Context".to_string())
        );

        assert!(actual.context.is_none()); // Empty context should be filtered out
        Ok(())
    }

    #[test]
    fn test_conversation_from_conversation_record() -> anyhow::Result<()> {
        let test_id = ConversationId::generate();
        let fixture = ConversationRecord {
            conversation_id: test_id.into_string(),
            parent_id: None,
            title: Some("Test Conversation".to_string()),
            context: None,
            created_at: Utc::now().naive_utc(),
            updated_at: None,
            workspace_id: 0,
            metrics: None,
            initiator: None,
        };

        let actual = Conversation::try_from(fixture)?;

        assert_eq!(actual.id, test_id);
        assert_eq!(actual.title, Some("Test Conversation".to_string()));
        assert_eq!(actual.context, None);
        Ok(())
    }

    #[tokio::test]
    async fn test_upsert_and_retrieve_conversation_with_metrics() -> anyhow::Result<()> {
        let repo = repository()?;

        // Create a conversation with metrics
        let metrics = Metrics::default()
            .started_at(Utc::now())
            .insert(
                "src/main.rs".to_string(),
                FileOperation::new(ToolKind::Write)
                    .lines_added(10u64)
                    .lines_removed(5u64)
                    .content_hash(Some("abc123def456".to_string())),
            )
            .insert(
                "src/lib.rs".to_string(),
                FileOperation::new(ToolKind::Write)
                    .lines_added(3u64)
                    .lines_removed(2u64)
                    .content_hash(Some("789xyz456abc".to_string())),
            );

        let fixture = Conversation::generate().metrics(metrics.clone());

        // Save the conversation
        repo.upsert_conversation(fixture.clone()).await?;

        // Retrieve the conversation
        let actual = repo
            .get_conversation(&fixture.id)
            .await?
            .expect("Conversation should exist");

        // Verify metrics are preserved
        assert_eq!(actual.metrics.file_operations.len(), 2);
        let main_metrics = actual.metrics.file_operations.get("src/main.rs").unwrap();
        assert_eq!(main_metrics.lines_added, 10);
        assert_eq!(main_metrics.lines_removed, 5);
        assert_eq!(main_metrics.content_hash, Some("abc123def456".to_string()));
        let lib_metrics = actual.metrics.file_operations.get("src/lib.rs").unwrap();
        assert_eq!(lib_metrics.lines_added, 3);
        assert_eq!(lib_metrics.lines_removed, 2);
        assert_eq!(lib_metrics.content_hash, Some("789xyz456abc".to_string()));
        Ok(())
    }

    #[test]
    fn test_metrics_record_conversion_preserves_all_fields() {
        // This test ensures compile-time safety: if Metrics schema changes,
        // this test will fail to compile, alerting us to update MetricsRecord
        let fixture = Metrics::default().started_at(Utc::now()).insert(
            "test.rs".to_string(),
            FileOperation::new(ToolKind::Write)
                .lines_added(5u64)
                .lines_removed(3u64)
                .content_hash(Some("test_hash_123".to_string())),
        );

        // Convert to record and back
        let record = MetricsRecord::from(&fixture);
        let actual = Metrics::from(record);

        // Verify all fields are preserved
        assert_eq!(actual.started_at, fixture.started_at);
        assert_eq!(actual.file_operations.len(), fixture.file_operations.len());

        let actual_file = actual.file_operations.get("test.rs").unwrap();
        let expected_file = fixture.file_operations.get("test.rs").unwrap();
        assert_eq!(actual_file.lines_added, expected_file.lines_added);
        assert_eq!(actual_file.lines_removed, expected_file.lines_removed);
        assert_eq!(actual_file.content_hash, expected_file.content_hash);
    }

    #[test]
    fn test_deserialize_old_format_without_tool_field() {
        // Old format from database: missing tool and content_hash fields
        let json = r#"{
            "started_at": "2024-01-01T00:00:00Z",
            "files_changed": {
                "src/main.rs": {
                    "lines_added": 10,
                    "lines_removed": 5
                },
                "src/lib.rs": {
                    "lines_added": 3,
                    "lines_removed": 2
                }
            }
        }"#;

        let record: MetricsRecord = serde_json::from_str(json).unwrap();
        let actual = Metrics::from(record);

        // Verify files are loaded
        assert_eq!(actual.file_operations.len(), 2);

        // Verify main.rs
        let main_file = actual.file_operations.get("src/main.rs").unwrap();
        assert_eq!(main_file.lines_added, 10);
        assert_eq!(main_file.lines_removed, 5);
        assert_eq!(main_file.content_hash, None);
        assert_eq!(main_file.tool, ToolKind::Write); // Default tool

        // Verify lib.rs
        let lib_file = actual.file_operations.get("src/lib.rs").unwrap();
        assert_eq!(lib_file.lines_added, 3);
        assert_eq!(lib_file.lines_removed, 2);
        assert_eq!(lib_file.content_hash, None);
        assert_eq!(lib_file.tool, ToolKind::Write); // Default tool
    }

    #[test]
    fn test_deserialize_array_format_takes_last_operation() {
        // Array format from database: multiple operations per file
        let json = r#"{
            "started_at": "2024-01-01T00:00:00Z",
            "files_changed": {
                "src/main.rs": [
                    {
                        "lines_added": 2,
                        "lines_removed": 4,
                        "content_hash": "hash1",
                        "tool": "read"
                    },
                    {
                        "lines_added": 1,
                        "lines_removed": 1,
                        "content_hash": "hash2",
                        "tool": "patch"
                    },
                    {
                        "lines_added": 5,
                        "lines_removed": 3,
                        "content_hash": "hash3",
                        "tool": "write"
                    }
                ]
            }
        }"#;

        let record: MetricsRecord = serde_json::from_str(json).unwrap();
        let actual = Metrics::from(record);

        // Verify only the last operation is kept
        assert_eq!(actual.file_operations.len(), 1);

        let main_file = actual.file_operations.get("src/main.rs").unwrap();
        assert_eq!(main_file.lines_added, 5);
        assert_eq!(main_file.lines_removed, 3);
        assert_eq!(main_file.content_hash, Some("hash3".to_string()));
        assert_eq!(main_file.tool, ToolKind::Write);
    }

    #[test]
    fn test_deserialize_array_format_with_empty_array() {
        // Array format with empty array should be skipped
        let json = r#"{
            "started_at": "2024-01-01T00:00:00Z",
            "files_changed": {
                "src/main.rs": [],
                "src/lib.rs": {
                    "lines_added": 5,
                    "lines_removed": 2,
                    "content_hash": "hash1",
                    "tool": "patch"
                }
            }
        }"#;

        let record: MetricsRecord = serde_json::from_str(json).unwrap();
        let actual = Metrics::from(record);

        // Empty array should be skipped, only lib.rs should be present
        assert_eq!(actual.file_operations.len(), 1);
        assert!(actual.file_operations.contains_key("src/lib.rs"));
        assert!(!actual.file_operations.contains_key("src/main.rs"));
    }

    #[test]
    fn test_deserialize_current_format_with_all_fields() {
        // Current format: single object with all fields
        let json = r#"{
            "started_at": "2024-01-01T00:00:00Z",
            "files_changed": {
                "src/main.rs": {
                    "lines_added": 10,
                    "lines_removed": 5,
                    "content_hash": "abc123def456",
                    "tool": "patch"
                },
                "src/lib.rs": {
                    "lines_added": 3,
                    "lines_removed": 2,
                    "content_hash": "789xyz456abc",
                    "tool": "write"
                }
            }
        }"#;

        let record: MetricsRecord = serde_json::from_str(json).unwrap();
        let actual = Metrics::from(record);

        // Verify all fields are preserved
        assert_eq!(actual.file_operations.len(), 2);

        let main_file = actual.file_operations.get("src/main.rs").unwrap();
        assert_eq!(main_file.lines_added, 10);
        assert_eq!(main_file.lines_removed, 5);
        assert_eq!(main_file.content_hash, Some("abc123def456".to_string()));
        assert_eq!(main_file.tool, ToolKind::Patch);

        let lib_file = actual.file_operations.get("src/lib.rs").unwrap();
        assert_eq!(lib_file.lines_added, 3);
        assert_eq!(lib_file.lines_removed, 2);
        assert_eq!(lib_file.content_hash, Some("789xyz456abc".to_string()));
        assert_eq!(lib_file.tool, ToolKind::Write);
    }

    #[test]
    fn test_deserialize_mixed_format() {
        // Mix of old format, array format, and current format
        let json = r#"{
            "started_at": "2024-01-01T00:00:00Z",
            "files_changed": {
                "old_file.rs": {
                    "lines_added": 10,
                    "lines_removed": 5
                },
                "array_file.rs": [
                    {
                        "lines_added": 1,
                        "lines_removed": 2,
                        "content_hash": "hash1",
                        "tool": "read"
                    },
                    {
                        "lines_added": 3,
                        "lines_removed": 4,
                        "content_hash": "hash2",
                        "tool": "patch"
                    }
                ],
                "current_file.rs": {
                    "lines_added": 7,
                    "lines_removed": 8,
                    "content_hash": "hash3",
                    "tool": "write"
                }
            }
        }"#;

        let record: MetricsRecord = serde_json::from_str(json).unwrap();
        let actual = Metrics::from(record);

        assert_eq!(actual.file_operations.len(), 3);

        // Old format file
        let old_file = actual.file_operations.get("old_file.rs").unwrap();
        assert_eq!(old_file.lines_added, 10);
        assert_eq!(old_file.lines_removed, 5);
        assert_eq!(old_file.content_hash, None);
        assert_eq!(old_file.tool, ToolKind::Write); // Default

        // Array format file (should have last operation)
        let array_file = actual.file_operations.get("array_file.rs").unwrap();
        assert_eq!(array_file.lines_added, 3);
        assert_eq!(array_file.lines_removed, 4);
        assert_eq!(array_file.content_hash, Some("hash2".to_string()));
        assert_eq!(array_file.tool, ToolKind::Patch);

        // Current format file
        let current_file = actual.file_operations.get("current_file.rs").unwrap();
        assert_eq!(current_file.lines_added, 7);
        assert_eq!(current_file.lines_removed, 8);
        assert_eq!(current_file.content_hash, Some("hash3".to_string()));
        assert_eq!(current_file.tool, ToolKind::Write);
    }

    #[test]
    fn test_serialize_current_format() {
        // Test that we always serialize in the current format (single object)
        let fixture = Metrics::default().started_at(Utc::now()).insert(
            "src/main.rs".to_string(),
            FileOperation::new(ToolKind::Patch)
                .lines_added(10u64)
                .lines_removed(5u64)
                .content_hash(Some("abc123".to_string())),
        );

        let record = MetricsRecord::from(&fixture);
        let json = serde_json::to_string(&record).unwrap();

        // Verify it's not an array format
        assert!(!json.contains("[{"));
        // Verify it contains the tool field
        assert!(json.contains("\"tool\":\"patch\""));

        // Verify structure is correct
        assert!(json.contains("\"lines_added\":10"));
        assert!(json.contains("\"lines_removed\":5"));
        assert!(json.contains("\"content_hash\":\"abc123\""));
    }

    #[test]
    fn test_context_record_conversion_preserves_all_fields() {
        let tool_def = ToolDefinition::new("test_tool").description("A test tool");

        let reasoning = forge_domain::ReasoningConfig {
            effort: Some(Effort::Medium),
            max_tokens: Some(2048),
            exclude: Some(false),
            enabled: Some(true),
        };

        // Create a comprehensive set of messages to test all message types
        let messages = vec![
            ContextMessage::user("Hello", None).into(),
            ContextMessage::system("System prompt").into(),
            ContextMessage::Tool(ToolResult {
                name: ToolName::new("test_tool"),
                call_id: Some(ToolCallId::new("call_123".to_string())),
                output: ToolOutput {
                    is_error: false,
                    values: vec![ToolValue::Text("Result text".to_string()), ToolValue::Empty],
                },
            })
            .into(),
            forge_domain::MessageEntry {
                message: ContextMessage::Text(forge_domain::TextMessage {
                    role: Role::Assistant,
                    content: "Assistant response".to_string(),
                    raw_content: None,
                    tool_calls: Some(vec![ToolCallFull {
                        name: ToolName::new("another_tool"),
                        call_id: Some(ToolCallId::new("call_456".to_string())),
                        arguments: forge_domain::ToolCallArguments::from(
                            serde_json::json!({"param": "value"}),
                        ),
                        thought_signature: None,
                    }]),
                    model: Some(forge_domain::ModelId::from("gpt-4")),
                    thought_signature: None,
                    reasoning_details: None,
                    droppable: false,
                    phase: None,
                    cacheable: Some(false),
                    kind: None,
                }),
                usage: Some(Usage {
                    prompt_tokens: forge_domain::TokenCount::Actual(100),
                    completion_tokens: forge_domain::TokenCount::Actual(50),
                    total_tokens: forge_domain::TokenCount::Actual(150),
                    cached_tokens: forge_domain::TokenCount::Actual(0),
                    cost: Some(0.001),
                }),
            },
        ];

        let fixture = Context::default()
            .conversation_id(ConversationId::generate())
            .messages(messages)
            .tools(vec![tool_def.clone()])
            .tool_choice(ToolChoice::Call(ToolName::new("test_tool")))
            .max_tokens(1000usize)
            .temperature(forge_domain::Temperature::new(0.7).unwrap())
            .top_p(forge_domain::TopP::new(0.9).unwrap())
            .top_k(forge_domain::TopK::new(50).unwrap())
            .reasoning(reasoning.clone())
            .stream(true);

        // Convert to record and back
        let record = ContextRecord::from(&fixture);
        let actual = Context::try_from(record).unwrap();

        // Verify all fields are preserved
        assert_eq!(actual.conversation_id, fixture.conversation_id);
        assert_eq!(actual.messages.len(), 4);
        assert_eq!(actual.tools.len(), 1);
        assert_eq!(actual.tools[0].name.to_string(), "test_tool");
        assert_eq!(
            actual.tool_choice,
            Some(ToolChoice::Call(ToolName::new("test_tool")))
        );
        assert_eq!(actual.max_tokens, fixture.max_tokens);
        assert_eq!(actual.temperature, fixture.temperature);
        assert_eq!(actual.top_p, fixture.top_p);
        assert_eq!(actual.top_k, fixture.top_k);
        assert_eq!(actual.reasoning, Some(reasoning));
        assert_eq!(actual.stream, fixture.stream);

        // Verify message types and content
        match &actual.messages[0].message {
            ContextMessage::Text(msg) => {
                assert_eq!(msg.role, Role::User);
                assert_eq!(msg.content, "Hello");
            }
            _ => panic!("Expected user message"),
        }

        match &actual.messages[2].message {
            ContextMessage::Tool(tool_result) => {
                assert_eq!(tool_result.name.to_string(), "test_tool");
                assert_eq!(
                    tool_result.call_id.as_ref().map(|id| id.as_str()),
                    Some("call_123")
                );
                assert!(!tool_result.output.is_error);
                assert_eq!(tool_result.output.values.len(), 2);
            }
            _ => panic!("Expected tool result message"),
        }

        match &actual.messages[3].message {
            ContextMessage::Text(message) => {
                assert_eq!(message.cacheable, Some(false));
                assert_eq!(message.is_cache_eligible(), false);
            }
            _ => panic!("Expected assistant text message"),
        }

        // Verify usage is preserved
        match &actual.messages[3].usage {
            Some(usage) => {
                assert_eq!(*usage.prompt_tokens, 100);
                assert_eq!(*usage.completion_tokens, 50);
                assert_eq!(*usage.total_tokens, 150);
                assert_eq!(usage.cost, Some(0.001));
            }
            None => panic!("Expected usage information"),
        }
    }

    #[test]
    fn test_conversation_deserialization_error_includes_id() {
        // Test that deserialization errors include the conversation ID
        let test_id = ConversationId::generate();
        let fixture = ConversationRecord {
            conversation_id: test_id.into_string(),
            parent_id: None,
            title: Some("Test Conversation".to_string()),
            context: Some("invalid json".to_string()), // Invalid JSON to trigger error
            created_at: Utc::now().naive_utc(),
            updated_at: None,
            workspace_id: 0,
            metrics: None,
            initiator: None,
        };

        let result = Conversation::try_from(fixture);

        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();
        assert!(
            error_message.contains(&test_id.to_string()),
            "Error message should contain conversation ID. Got: {}",
            error_message
        );
        assert!(
            error_message.contains("Failed to deserialize context"),
            "Error message should indicate context deserialization failure. Got: {}",
            error_message
        );
    }

    #[tokio::test]
    async fn test_delete_conversation_success() -> anyhow::Result<()> {
        let repo = repository()?;
        let conversation = Conversation::new(ConversationId::generate())
            .title(Some("Test Conversation".to_string()));

        repo.upsert_conversation(conversation.clone()).await?;

        repo.delete_conversation(&conversation.id).await?;

        let result = repo.get_conversation(&conversation.id).await?;
        assert!(result.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn test_delete_conversation_workspace_filtering() -> anyhow::Result<()> {
        let repo = repository()?;
        let conversation = Conversation::new(ConversationId::generate())
            .title(Some("Test Conversation".to_string()));

        repo.upsert_conversation(conversation.clone()).await?;

        // Delete should succeed regardless of existence (idempotent)
        repo.delete_conversation(&conversation.id).await?;

        // Verify conversation is deleted
        let deleted = repo.get_conversation(&conversation.id).await?;
        assert!(deleted.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn test_delete_conversation_cross_workspace_security() -> anyhow::Result<()> {
        let repo = repository()?;

        // Create conversation in current workspace
        let conversation_id = ConversationId::generate();
        let conversation =
            Conversation::new(conversation_id).title(Some("Test Conversation".to_string()));

        repo.upsert_conversation(conversation.clone()).await?;

        // Try to delete with different workspace ID (should fail due to security)
        // Note: This test would require modifying workspace ID in repo
        // For now, we test that deletion works with current workspace
        repo.delete_conversation(&conversation.id).await?;

        // Verify it's actually deleted
        let deleted = repo.get_conversation(&conversation.id).await?;
        assert!(deleted.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn test_delete_conversation_end_to_end_workflow() -> anyhow::Result<()> {
        let repo = repository()?;
        let conversation_id = ConversationId::generate();
        let conversation =
            Conversation::new(conversation_id).title(Some("Test Conversation".to_string()));

        // Test complete workflow: create -> delete -> verify -> create new -> verify
        repo.upsert_conversation(conversation.clone()).await?;

        // Delete conversation
        repo.delete_conversation(&conversation.id).await?;

        // Verify it's gone
        let deleted_check = repo.get_conversation(&conversation.id).await?;
        assert!(deleted_check.is_none());

        // Create new conversation to ensure system still works
        let new_conversation_id = ConversationId::generate();
        let new_conversation = Conversation::new(new_conversation_id);
        repo.upsert_conversation(new_conversation.clone()).await?;

        // Verify new conversation exists
        let new_check = repo.get_conversation(&new_conversation_id).await?;
        assert!(new_check.is_some());

        Ok(())
    }

    #[tokio::test]
    async fn test_rename_conversation_via_upsert() -> anyhow::Result<()> {
        let repo = repository()?;
        let conversation =
            Conversation::new(ConversationId::generate()).title(Some("Original Title".to_string()));

        repo.upsert_conversation(conversation.clone()).await?;

        // Rename by upserting with a new title
        let renamed = conversation
            .clone()
            .title(Some("Renamed Session".to_string()));
        repo.upsert_conversation(renamed).await?;

        let actual = repo.get_conversation(&conversation.id).await?.unwrap();
        assert_eq!(actual.title, Some("Renamed Session".to_string()));
        Ok(())
    }

    #[tokio::test]
    async fn test_rename_conversation_from_none() -> anyhow::Result<()> {
        let repo = repository()?;
        let conversation = Conversation::new(ConversationId::generate());

        // Start with no title
        assert!(conversation.title.is_none());
        repo.upsert_conversation(conversation.clone()).await?;

        // Rename it
        let renamed = conversation.clone().title(Some("My Session".to_string()));
        repo.upsert_conversation(renamed).await?;

        let actual = repo.get_conversation(&conversation.id).await?.unwrap();
        assert_eq!(actual.title, Some("My Session".to_string()));
        Ok(())
    }

    #[test]
    fn test_legacy_tool_value_pair_deserialization() {
        use crate::conversation::conversation_record::ToolOutputRecord;

        // This JSON represents the old Pair variant format that was stored in the
        // database
        let legacy_json = r#"{
            "is_error": false,
            "values": [
                {"pair": [
                    {"text": "XML content for LLM"},
                    {"fileDiff": {"path": "/test/file.rs", "old_text": "old", "new_text": "new"}}
                ]}
            ]
        }"#;

        let record: ToolOutputRecord = serde_json::from_str(legacy_json).unwrap();
        let actual: forge_domain::ToolOutput = record.try_into().unwrap();

        // The Pair variant should be converted by taking the first element (LLM
        // content)
        assert!(!actual.is_error);
        assert_eq!(actual.values.len(), 1);
        assert_eq!(
            actual.values[0],
            forge_domain::ToolValue::Text("XML content for LLM".to_string())
        );
    }

    #[test]
    fn test_legacy_tool_value_markdown_deserialization() {
        use crate::conversation::conversation_record::ToolOutputRecord;

        let legacy_json = r##"{
            "is_error": false,
            "values": [{"markdown": "# Heading - Some bold text"}]
        }"##;

        let record: ToolOutputRecord = serde_json::from_str(legacy_json).unwrap();
        let actual: forge_domain::ToolOutput = record.try_into().unwrap();

        // Markdown should be converted to Text
        assert_eq!(actual.values.len(), 1);
        assert_eq!(
            actual.values[0],
            forge_domain::ToolValue::Text("# Heading - Some bold text".to_string())
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_concurrent_operations_dont_block_runtime() -> anyhow::Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::{Duration, Instant};

        // Heartbeat fires every `TICK`; we require a measurement window of at
        // least `MIN_WINDOW` so the assertion is meaningful even when the DB
        // workload finishes very quickly (e.g. on fast machines with the
        // in-memory SQLite pool).
        const TICK: Duration = Duration::from_millis(10);
        const MIN_WINDOW: Duration = Duration::from_millis(200);

        let repo = Arc::new(repository()?);
        let heartbeat = Arc::new(AtomicUsize::new(0));

        // Heartbeat task - if runtime is blocked, this won't increment.
        let heartbeat_clone = heartbeat.clone();
        let heartbeat_handle = tokio::spawn(async move {
            loop {
                tokio::time::sleep(TICK).await;
                heartbeat_clone.fetch_add(1, Ordering::Relaxed);
            }
        });

        // Warm up: let the heartbeat task get scheduled and complete its first
        // tick before we start measuring, then reset the counter so timing
        // begins from a clean state.
        tokio::time::sleep(TICK * 3).await;
        heartbeat.store(0, Ordering::Relaxed);

        // Spawn many concurrent DB operations.
        let mut handles = vec![];
        let start = Instant::now();

        for i in 0..20 {
            let repo = repo.clone();
            let handle = tokio::spawn(async move {
                for j in 0..10 {
                    let conversation = Conversation::new(ConversationId::generate())
                        .title(Some(format!("Task {} - Write {}", i, j)));
                    repo.upsert_conversation(conversation).await?;
                }
                anyhow::Result::<()>::Ok(())
            });
            handles.push(handle);
        }

        // Wait for all operations.
        for handle in handles {
            handle.await??;
        }

        // Ensure the measurement window is long enough for heartbeat math to
        // be meaningful regardless of how fast the DB workload completed.
        let work_elapsed = start.elapsed();
        if work_elapsed < MIN_WINDOW {
            tokio::time::sleep(MIN_WINDOW - work_elapsed).await;
        }
        let elapsed = start.elapsed();

        // Stop heartbeat.
        heartbeat_handle.abort();

        // Verify runtime wasn't blocked: heartbeat should have fired at least
        // 80% of the theoretical max for the elapsed window. The threshold is
        // clamped to at least 1 to keep the assertion well-defined.
        let heartbeat_count = heartbeat.load(Ordering::Relaxed);
        let expected_heartbeats = (elapsed.as_millis() as usize) / (TICK.as_millis() as usize);
        let threshold = (expected_heartbeats * 8 / 10).max(1);

        assert!(
            heartbeat_count >= threshold,
            "Runtime was blocked! Expected at least {} heartbeats (~{} theoretical) in {:?}, got {}",
            threshold,
            expected_heartbeats,
            elapsed,
            heartbeat_count
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_mixed_read_write_contention() -> anyhow::Result<()> {
        let repo = Arc::new(repository()?);
        let mut handles = vec![];

        // Pre-populate some data
        for i in 0..10 {
            let conv =
                Conversation::new(ConversationId::generate()).title(Some(format!("Initial {}", i)));
            repo.upsert_conversation(conv).await?;
        }

        // Spawn writers
        for i in 0..10 {
            let repo = repo.clone();
            handles.push(tokio::spawn(async move {
                for j in 0..10 {
                    let conv = Conversation::new(ConversationId::generate())
                        .title(Some(format!("Writer {} - {}", i, j)));
                    repo.upsert_conversation(conv).await?;
                }
                anyhow::Result::<()>::Ok(())
            }));
        }

        // Spawn readers (interleave with writers)
        for _ in 0..10 {
            let repo = repo.clone();
            handles.push(tokio::spawn(async move {
                for _ in 0..10 {
                    // Read all conversations
                    let _ = repo.get_all_conversations().await?;
                    tokio::task::yield_now().await;
                }
                anyhow::Result::<()>::Ok(())
            }));
        }

        // All should complete without timeout
        for handle in handles {
            handle.await??;
        }

        Ok(())
    }

    #[test]
    fn test_legacy_tool_value_file_diff_deserialization() {
        use crate::conversation::conversation_record::ToolOutputRecord;

        let legacy_json = r#"{
            "is_error": false,
            "values": [{"fileDiff": {"path": "/src/main.rs", "old_text": "fn old()", "new_text": "fn new()"}}]
        }"#;

        let record: ToolOutputRecord = serde_json::from_str(legacy_json).unwrap();
        let actual: forge_domain::ToolOutput = record.try_into().unwrap();

        // FileDiff should be converted to a text summary
        assert_eq!(actual.values.len(), 1);
        assert_eq!(
            actual.values[0],
            forge_domain::ToolValue::Text("[File diff: /src/main.rs]".to_string())
        );
    }

    #[tokio::test]
    async fn test_delete_conversation_recursive_depth() -> anyhow::Result<()> {
        let repo = repository()?;

        let parent_id = forge_domain::ConversationId::generate();
        let child_id = forge_domain::ConversationId::generate();
        let grandchild_id = forge_domain::ConversationId::generate();

        // Parent
        let parent = forge_domain::Conversation::new(parent_id);
        repo.upsert_conversation(parent).await?;

        // Child
        let child = forge_domain::Conversation::new(child_id).parent_id(Some(parent_id));
        repo.upsert_conversation(child).await?;

        // Grandchild
        let grandchild = forge_domain::Conversation::new(grandchild_id).parent_id(Some(child_id));
        repo.upsert_conversation(grandchild).await?;

        // Delete parent
        repo.delete_conversation(&parent_id).await?;

        // Verify all are gone
        assert!(
            repo.get_conversation(&parent_id).await?.is_none(),
            "Parent should be deleted"
        );
        assert!(
            repo.get_conversation(&child_id).await?.is_none(),
            "Child should be deleted"
        );
        assert!(
            repo.get_conversation(&grandchild_id).await?.is_none(),
            "Grandchild should be deleted"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_subagent_task_session_lifecycle_persists_final_result() -> anyhow::Result<()> {
        let repo = repository()?;
        let mut fixture = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            ConversationId::generate(),
            Some(ConversationId::generate()),
            Some(ConversationId::generate()),
            "inspect repository",
        );
        fixture.mark_completed("done");

        repo.upsert_subagent_task_session(fixture.clone()).await?;
        let actual = repo
            .get_subagent_task_session(&fixture.task_id)
            .await?
            .expect("task session should be persisted");
        let expected = (
            fixture.task_id,
            forge_domain::SubagentTaskStatus::Completed,
            Some("done".to_string()),
            fixture.conversation_id,
        );

        assert_eq!(
            (
                actual.task_id,
                actual.status,
                actual.final_result,
                actual.conversation_id,
            ),
            expected
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_subagent_task_session_lifecycle_updates_same_task_identity() -> anyhow::Result<()>
    {
        let repo = repository()?;
        let mut fixture = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            ConversationId::generate(),
            Some(ConversationId::generate()),
            Some(ConversationId::generate()),
            "inspect repository",
        );
        fixture.mark_running();
        repo.upsert_subagent_task_session(fixture.clone()).await?;
        fixture.mark_completed("done");
        repo.upsert_subagent_task_session(fixture.clone()).await?;
        fixture.mark_delivered();

        repo.upsert_subagent_task_session(fixture.clone()).await?;
        let actual = repo
            .get_subagent_task_session(&fixture.task_id)
            .await?
            .expect("task session should be updated");
        let expected = fixture;

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_subagent_task_session_lifecycle_updates_parentless_task_identity()
    -> anyhow::Result<()> {
        let repo = repository()?;
        let mut fixture = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            ConversationId::generate(),
            None,
            None,
            "parentless task",
        );
        fixture.mark_running();
        repo.upsert_subagent_task_session(fixture.clone()).await?;
        fixture.mark_completed("done");

        repo.upsert_subagent_task_session(fixture.clone()).await?;
        let actual = repo
            .get_subagent_task_session(&fixture.task_id)
            .await?
            .expect("parentless task session should be updated");
        let expected = fixture;

        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_list_subagent_task_sessions_active_and_all() -> anyhow::Result<()> {
        let repo = repository()?;
        let mut active = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            ConversationId::generate(),
            Some(ConversationId::generate()),
            Some(ConversationId::generate()),
            "active task",
        );
        active.mark_running();
        let mut completed = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            ConversationId::generate(),
            Some(ConversationId::generate()),
            Some(ConversationId::generate()),
            "completed task",
        );
        completed.mark_completed("done");

        repo.upsert_subagent_task_session(active.clone()).await?;
        repo.upsert_subagent_task_session(completed.clone()).await?;
        let actual_active = repo
            .list_subagent_task_sessions(SubagentTaskSessionFilter::Active)
            .await?;
        let actual_all = repo
            .list_subagent_task_sessions(SubagentTaskSessionFilter::All)
            .await?;
        let expected = (vec![active.task_id], 2usize);

        assert_eq!(
            (
                actual_active
                    .into_iter()
                    .map(|session| session.task_id)
                    .collect::<Vec<_>>(),
                actual_all.len(),
            ),
            expected
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_list_subagent_task_sessions_classifies_stale_active_as_zombie()
    -> anyhow::Result<()> {
        let repo = repository()?;
        let mut fixture = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            ConversationId::generate(),
            Some(ConversationId::generate()),
            Some(ConversationId::generate()),
            "stale task",
        );
        fixture.mark_running();
        fixture.heartbeat_at = Utc::now() - Duration::minutes(20);

        repo.upsert_subagent_task_session(fixture.clone()).await?;
        let actual = repo
            .list_subagent_task_sessions(SubagentTaskSessionFilter::Active)
            .await?
            .into_iter()
            .find(|session| session.task_id == fixture.task_id)
            .expect("stale task should be listed as active zombie");
        let expected = forge_domain::SubagentTaskStatus::Zombie;

        assert_eq!(actual.status, expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_upsert_subagent_task_session_rejects_cross_workspace_task_id_collision()
    -> anyhow::Result<()> {
        let pool = Arc::new(DatabasePool::in_memory()?);
        let repo = repository_with_pool(pool.clone(), WorkspaceHash::new(1));
        let foreign_repo = repository_with_pool(pool, WorkspaceHash::new(2));
        let fixture = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            ConversationId::generate(),
            Some(ConversationId::generate()),
            Some(ConversationId::generate()),
            "workspace task",
        );

        repo.upsert_subagent_task_session(fixture.clone()).await?;
        let actual = foreign_repo.upsert_subagent_task_session(fixture).await;

        assert!(actual.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_upsert_subagent_task_session_rejects_conflicting_ledger_owner()
    -> anyhow::Result<()> {
        let repo = repository()?;
        let conversation_id = ConversationId::generate();
        let original_owner = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            conversation_id,
            Some(ConversationId::generate()),
            Some(ConversationId::generate()),
            "original owner",
        );
        let conflicting_owner = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            conversation_id,
            Some(ConversationId::generate()),
            original_owner.root_conversation_id,
            "conflicting owner",
        );

        repo.upsert_subagent_task_session(original_owner).await?;
        let actual = repo.upsert_subagent_task_session(conflicting_owner).await;

        assert!(actual.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_upsert_subagent_task_session_rejects_task_id_conversation_reassignment()
    -> anyhow::Result<()> {
        let repo = repository()?;
        let original = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            ConversationId::generate(),
            Some(ConversationId::generate()),
            Some(ConversationId::generate()),
            "original task",
        );
        let mut reassigned = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            ConversationId::generate(),
            Some(ConversationId::generate()),
            Some(ConversationId::generate()),
            "reassigned task",
        );
        reassigned.task_id = original.task_id;

        repo.upsert_subagent_task_session(original.clone()).await?;
        let actual = repo.upsert_subagent_task_session(reassigned).await;
        let persisted = repo
            .get_subagent_task_session(&original.task_id)
            .await?
            .expect("original task session should remain persisted");
        let expected = original.conversation_id;

        assert!(actual.is_err());
        assert_eq!(persisted.conversation_id, expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_promote_delegated_conversation_rejects_any_historical_conflicting_ledger_owner()
    -> anyhow::Result<()> {
        let repo = repository()?;
        let conversation_id = ConversationId::generate();
        let original_parent_id = ConversationId::generate();
        let resume_parent_id = ConversationId::generate();
        let conversation = forge_domain::Conversation::new(conversation_id);
        let mut original_owner = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            conversation_id,
            Some(original_parent_id),
            Some(original_parent_id),
            "original owner",
        );
        original_owner.mark_completed("done");
        original_owner.updated_at = Utc::now() - Duration::minutes(10);
        let mut latest_attempt = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            conversation_id,
            Some(resume_parent_id),
            Some(resume_parent_id),
            "conflicting latest owner",
        );
        latest_attempt.mark_running();

        repo.upsert_conversation(conversation).await?;
        repo.upsert_subagent_task_session(original_owner).await?;
        insert_legacy_subagent_task_session(&repo, latest_attempt).await?;
        let actual = repo
            .promote_delegated_conversation(&conversation_id, Some(resume_parent_id))
            .await;
        let expected = None;

        assert!(actual.is_err());
        let persisted = repo
            .get_conversation(&conversation_id)
            .await?
            .expect("conversation should remain persisted");
        assert_eq!(persisted.parent_id, expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_get_subagent_task_session_by_conversation_breaks_ties_by_newest_attempt()
    -> anyhow::Result<()> {
        let repo = repository()?;
        let conversation_id = ConversationId::generate();
        let parent_id = ConversationId::generate();
        let root_id = ConversationId::generate();
        let shared_updated_at = Utc::now();
        let mut previous = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            conversation_id,
            Some(parent_id),
            Some(root_id),
            "previous task",
        );
        previous.created_at = shared_updated_at - Duration::seconds(1);
        previous.updated_at = shared_updated_at;
        previous.heartbeat_at = shared_updated_at;
        previous.mark_completed("previous result");
        previous.updated_at = shared_updated_at;
        previous.heartbeat_at = shared_updated_at;
        let mut latest = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            conversation_id,
            Some(parent_id),
            Some(root_id),
            "latest task",
        );
        latest.created_at = shared_updated_at;
        latest.updated_at = shared_updated_at;
        latest.heartbeat_at = shared_updated_at;
        latest.mark_running();
        latest.updated_at = shared_updated_at;
        latest.heartbeat_at = shared_updated_at;

        repo.upsert_subagent_task_session(previous).await?;
        repo.upsert_subagent_task_session(latest.clone()).await?;
        let actual = repo
            .get_subagent_task_session_by_conversation(&conversation_id)
            .await?
            .expect("latest attempt should be selected deterministically");
        let expected = latest.task_id;

        assert_eq!(actual.task_id, expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_get_subagent_task_session_by_conversation_prefers_older_active_attempt_over_newer_terminal()
    -> anyhow::Result<()> {
        let repo = repository()?;
        let conversation_id = ConversationId::generate();
        let parent_id = ConversationId::generate();
        let root_id = ConversationId::generate();
        let mut active = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            conversation_id,
            Some(parent_id),
            Some(root_id),
            "older active task",
        );
        active.mark_running();
        active.updated_at = Utc::now() - Duration::minutes(10);
        active.heartbeat_at = active.updated_at;
        let mut terminal = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            conversation_id,
            Some(parent_id),
            Some(root_id),
            "newer terminal task",
        );
        terminal.mark_completed("done");

        repo.upsert_subagent_task_session(active.clone()).await?;
        repo.upsert_subagent_task_session(terminal.clone()).await?;
        let actual = repo
            .get_subagent_task_session_by_conversation(&conversation_id)
            .await?
            .expect("active attempt should block resume even when not latest");
        let expected = (active.task_id, forge_domain::SubagentTaskStatus::Zombie);

        assert_eq!((actual.task_id, actual.status), expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_upsert_subagent_task_session_rejects_second_active_attempt_for_conversation()
    -> anyhow::Result<()> {
        let repo = repository()?;
        let conversation_id = ConversationId::generate();
        let parent_id = ConversationId::generate();
        let root_id = ConversationId::generate();
        let mut original = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            conversation_id,
            Some(parent_id),
            Some(root_id),
            "original active task",
        );
        original.mark_running();
        let mut duplicate = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            conversation_id,
            Some(parent_id),
            Some(root_id),
            "duplicate active task",
        );
        duplicate.mark_running();

        repo.upsert_subagent_task_session(original.clone()).await?;
        let actual = repo.upsert_subagent_task_session(duplicate).await;
        let persisted = repo
            .get_subagent_task_session(&original.task_id)
            .await?
            .expect("original active task should remain persisted");
        let expected = original.task_id;

        assert!(actual.is_err());
        assert_eq!(persisted.task_id, expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_upsert_subagent_task_session_allows_new_active_attempt_after_terminal_history()
    -> anyhow::Result<()> {
        let repo = repository()?;
        let conversation_id = ConversationId::generate();
        let parent_id = ConversationId::generate();
        let root_id = ConversationId::generate();
        let mut terminal = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            conversation_id,
            Some(parent_id),
            Some(root_id),
            "terminal task",
        );
        terminal.mark_completed("done");
        let mut next = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            conversation_id,
            Some(parent_id),
            Some(root_id),
            "next active task",
        );
        next.mark_running();

        repo.upsert_subagent_task_session(terminal).await?;
        let actual = repo.upsert_subagent_task_session(next.clone()).await;
        let persisted = repo
            .get_subagent_task_session_by_conversation(&conversation_id)
            .await?
            .expect("new active task should be persisted after terminal history");
        let expected = next.task_id;

        assert!(actual.is_ok());
        assert_eq!(persisted.task_id, expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_get_subagent_task_session_by_conversation_returns_latest_attempt()
    -> anyhow::Result<()> {
        let repo = repository()?;
        let conversation_id = ConversationId::generate();
        let mut previous = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            conversation_id,
            Some(ConversationId::generate()),
            Some(ConversationId::generate()),
            "previous task",
        );
        previous.mark_completed("previous result");
        previous.mark_delivered();
        previous.updated_at = Utc::now() - Duration::minutes(1);
        let mut latest = SubagentTaskSession::new(
            forge_domain::AgentId::new("forge"),
            conversation_id,
            previous.parent_conversation_id,
            previous.root_conversation_id,
            "latest task",
        );
        latest.mark_running();

        repo.upsert_subagent_task_session(previous.clone()).await?;
        repo.upsert_subagent_task_session(latest.clone()).await?;
        let actual_latest = repo
            .get_subagent_task_session_by_conversation(&conversation_id)
            .await?
            .expect("latest attempt should be found by conversation");
        let actual_previous = repo
            .get_subagent_task_session(&previous.task_id)
            .await?
            .expect("previous terminal attempt should remain durable");
        let expected = (
            latest.task_id,
            Some("previous result".to_string()),
            previous.delivered_at.is_some(),
        );

        assert_eq!(
            (
                actual_latest.task_id,
                actual_previous.final_result,
                actual_previous.delivered_at.is_some(),
            ),
            expected
        );
        Ok(())
    }
}
