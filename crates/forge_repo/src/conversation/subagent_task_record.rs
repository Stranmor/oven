use std::str::FromStr;

use chrono::{DateTime, NaiveDateTime, Utc};
use diesel::prelude::*;
use forge_domain::{
    AgentId, ConversationId, SubagentTaskId, SubagentTaskSession, SubagentTaskStatus,
};

#[derive(Debug, Clone, Queryable, QueryableByName, Selectable)]
#[diesel(table_name = crate::database::schema::subagent_task_sessions)]
pub(super) struct SubagentTaskSessionRecord {
    #[diesel(sql_type = diesel::sql_types::Text)]
    pub task_id: String,
    #[diesel(sql_type = diesel::sql_types::Text)]
    pub agent_id: String,
    #[diesel(sql_type = diesel::sql_types::Text)]
    pub conversation_id: String,
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Text>)]
    pub parent_conversation_id: Option<String>,
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Text>)]
    pub root_conversation_id: Option<String>,
    #[diesel(sql_type = diesel::sql_types::BigInt)]
    pub workspace_id: i64,
    #[diesel(sql_type = diesel::sql_types::Text)]
    pub status: String,
    #[diesel(sql_type = diesel::sql_types::Text)]
    pub task: String,
    #[diesel(sql_type = diesel::sql_types::Timestamp)]
    pub created_at: NaiveDateTime,
    #[diesel(sql_type = diesel::sql_types::Timestamp)]
    pub updated_at: NaiveDateTime,
    #[diesel(sql_type = diesel::sql_types::Timestamp)]
    pub heartbeat_at: NaiveDateTime,
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Text>)]
    pub final_result: Option<String>,
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Text>)]
    pub final_error: Option<String>,
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Timestamp>)]
    pub delivered_at: Option<NaiveDateTime>,
}

impl SubagentTaskSessionRecord {
    pub(super) fn new(session: SubagentTaskSession, workspace_id: i64) -> Self {
        Self {
            task_id: session.task_id.into_string(),
            agent_id: session.agent_id.to_string(),
            conversation_id: session.conversation_id.into_string(),
            parent_conversation_id: session.parent_conversation_id.map(|id| id.into_string()),
            root_conversation_id: session.root_conversation_id.map(|id| id.into_string()),
            workspace_id,
            status: session.status.to_string(),
            task: session.task,
            created_at: session.created_at.naive_utc(),
            updated_at: session.updated_at.naive_utc(),
            heartbeat_at: session.heartbeat_at.naive_utc(),
            final_result: session.final_result,
            final_error: session.final_error,
            delivered_at: session.delivered_at.map(|at| at.naive_utc()),
        }
    }
}

fn parse_status(value: &str) -> anyhow::Result<SubagentTaskStatus> {
    SubagentTaskStatus::from_str(value)
        .map_err(|_| anyhow::anyhow!("Unknown subagent task status '{value}'"))
}

fn from_naive(value: NaiveDateTime) -> DateTime<Utc> {
    DateTime::from_naive_utc_and_offset(value, Utc)
}

impl TryFrom<SubagentTaskSessionRecord> for SubagentTaskSession {
    type Error = anyhow::Error;

    fn try_from(record: SubagentTaskSessionRecord) -> anyhow::Result<Self> {
        Ok(Self {
            task_id: SubagentTaskId::parse(record.task_id)?,
            agent_id: AgentId::new(record.agent_id),
            conversation_id: ConversationId::parse(record.conversation_id)?,
            parent_conversation_id: record
                .parent_conversation_id
                .map(ConversationId::parse)
                .transpose()?,
            root_conversation_id: record
                .root_conversation_id
                .map(ConversationId::parse)
                .transpose()?,
            status: parse_status(&record.status)?,
            task: record.task,
            created_at: from_naive(record.created_at),
            updated_at: from_naive(record.updated_at),
            heartbeat_at: from_naive(record.heartbeat_at),
            final_result: record.final_result,
            final_error: record.final_error,
            delivered_at: record.delivered_at.map(from_naive),
        })
    }
}
