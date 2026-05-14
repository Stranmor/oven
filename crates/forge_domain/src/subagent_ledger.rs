use std::str::FromStr;

use chrono::{DateTime, Duration, Utc};
use derive_more::derive::Display;
use derive_setters::Setters;
use serde::{Deserialize, Serialize};
use strum_macros::{Display as StrumDisplay, EnumString};
use uuid::Uuid;

use crate::{AgentId, ConversationId, Error, Result};

/// Durable identifier for a delegated subagent task lifecycle record.
#[derive(Debug, Default, Display, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash)]
#[serde(transparent)]
pub struct SubagentTaskId(Uuid);

impl SubagentTaskId {
    /// Generates a new durable subagent task identifier.
    pub fn generate() -> Self {
        Self(Uuid::new_v4())
    }

    /// Converts this task ID to its canonical string representation.
    pub fn into_string(&self) -> String {
        self.0.to_string()
    }

    /// Parses a durable subagent task identifier from text.
    ///
    /// # Arguments
    /// * `value` - The textual UUID value to parse.
    ///
    /// # Errors
    /// Returns an error when `value` is not a valid UUID.
    pub fn parse(value: impl ToString) -> Result<Self> {
        Ok(Self(
            Uuid::parse_str(&value.to_string()).map_err(Error::ConversationId)?,
        ))
    }
}

impl FromStr for SubagentTaskId {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

/// Lifecycle status persisted for delegated subagent work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, StrumDisplay, EnumString)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum SubagentTaskStatus {
    /// The task session has been created but execution has not started.
    Created,
    /// The delegated agent is currently running or was recently heartbeated.
    Running,
    /// The delegated agent produced a final result.
    Completed,
    /// The delegated agent failed with a persisted error.
    Failed,
    /// The delegated agent was interrupted before normal completion.
    Interrupted,
    /// The last heartbeat is stale while the task was not terminal.
    Zombie,
}

impl SubagentTaskStatus {
    /// Returns true when this status represents an active lifecycle state.
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Created | Self::Running | Self::Zombie)
    }

    /// Returns true when this status represents a terminal lifecycle state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Interrupted)
    }
}

/// List filter for durable subagent task sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubagentTaskSessionFilter {
    /// Return only task sessions that are still active or stale-active.
    Active,
    /// Return every known task session.
    All,
}

/// Durable lifecycle record for one delegated subagent task/session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Setters)]
#[setters(into, strip_option)]
pub struct SubagentTaskSession {
    /// Persistent task lifecycle identifier independent from conversation ID.
    pub task_id: SubagentTaskId,
    /// Delegated agent identifier that owns this lifecycle record.
    pub agent_id: AgentId,
    /// The subagent conversation/session ID used for compatibility resume.
    pub conversation_id: ConversationId,
    /// Parent conversation that spawned this task when known.
    pub parent_conversation_id: Option<ConversationId>,
    /// Root conversation for the delegation tree when known.
    pub root_conversation_id: Option<ConversationId>,
    /// Current persisted lifecycle status.
    pub status: SubagentTaskStatus,
    /// Short task title or request summary.
    pub task: String,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Last lifecycle mutation timestamp.
    pub updated_at: DateTime<Utc>,
    /// Last heartbeat timestamp emitted while executing.
    pub heartbeat_at: DateTime<Utc>,
    /// Final result text when the delegated agent completes.
    pub final_result: Option<String>,
    /// Final error text when the delegated agent fails or is interrupted.
    pub final_error: Option<String>,
    /// Timestamp when the result was delivered to the parent tool call.
    pub delivered_at: Option<DateTime<Utc>>,
}

impl SubagentTaskSession {
    /// Creates a new durable subagent task lifecycle record.
    ///
    /// # Arguments
    /// * `agent_id` - The delegated agent identifier.
    /// * `conversation_id` - The delegated conversation/session ID.
    /// * `parent_conversation_id` - The parent conversation that spawned this
    ///   task.
    /// * `root_conversation_id` - The root conversation for the delegation
    ///   tree.
    /// * `task` - The task prompt or summary.
    pub fn new(
        agent_id: AgentId,
        conversation_id: ConversationId,
        parent_conversation_id: Option<ConversationId>,
        root_conversation_id: Option<ConversationId>,
        task: impl Into<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            task_id: SubagentTaskId::generate(),
            agent_id,
            conversation_id,
            parent_conversation_id,
            root_conversation_id,
            status: SubagentTaskStatus::Created,
            task: task.into(),
            created_at: now,
            updated_at: now,
            heartbeat_at: now,
            final_result: None,
            final_error: None,
            delivered_at: None,
        }
    }

    /// Marks this task session as running and refreshes heartbeat timestamps.
    pub fn mark_running(&mut self) {
        let now = Utc::now();
        self.status = SubagentTaskStatus::Running;
        self.updated_at = now;
        self.heartbeat_at = now;
    }

    /// Refreshes the heartbeat timestamp for an active task session.
    pub fn heartbeat(&mut self) {
        let now = Utc::now();
        self.updated_at = now;
        self.heartbeat_at = now;
    }

    /// Persists a completed final result without marking parent delivery.
    ///
    /// # Arguments
    /// * `result` - The delegated agent final response text.
    pub fn mark_completed(&mut self, result: impl Into<String>) {
        let now = Utc::now();
        self.status = SubagentTaskStatus::Completed;
        self.updated_at = now;
        self.heartbeat_at = now;
        self.final_result = Some(result.into());
        self.final_error = None;
    }

    /// Marks this completed task result as delivered to the parent tool call.
    pub fn mark_delivered(&mut self) {
        let now = Utc::now();
        self.updated_at = now;
        self.delivered_at = Some(now);
    }

    /// Persists a failed final error.
    ///
    /// # Arguments
    /// * `error` - The failure text to persist.
    pub fn mark_failed(&mut self, error: impl Into<String>) {
        let now = Utc::now();
        self.status = SubagentTaskStatus::Failed;
        self.updated_at = now;
        self.heartbeat_at = now;
        self.final_result = None;
        self.final_error = Some(error.into());
        self.delivered_at = None;
    }

    /// Persists an interrupted final error.
    ///
    /// # Arguments
    /// * `error` - The interruption reason to persist.
    pub fn mark_interrupted(&mut self, error: impl Into<String>) {
        let now = Utc::now();
        self.status = SubagentTaskStatus::Interrupted;
        self.updated_at = now;
        self.heartbeat_at = now;
        self.final_result = None;
        self.final_error = Some(error.into());
        self.delivered_at = None;
    }

    /// Returns this record classified as zombie when its active heartbeat is
    /// stale.
    ///
    /// # Arguments
    /// * `now` - The timestamp to classify against.
    /// * `timeout` - Maximum allowed active heartbeat age.
    pub fn classify_with_heartbeat(mut self, now: DateTime<Utc>, timeout: Duration) -> Self {
        if !self.status.is_terminal() && now - self.heartbeat_at > timeout {
            self.status = SubagentTaskStatus::Zombie;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_classify_with_heartbeat_marks_stale_running_task_zombie() {
        let mut fixture = SubagentTaskSession::new(
            AgentId::new("forge"),
            ConversationId::generate(),
            Some(ConversationId::generate()),
            Some(ConversationId::generate()),
            "run task",
        );
        fixture.status = SubagentTaskStatus::Running;
        fixture.heartbeat_at = Utc::now() - Duration::minutes(20);

        let actual = fixture.classify_with_heartbeat(Utc::now(), Duration::minutes(5));
        let expected = SubagentTaskStatus::Zombie;

        assert_eq!(actual.status, expected);
    }

    #[test]
    fn test_classify_with_heartbeat_keeps_fresh_running_task_active() {
        let mut fixture = SubagentTaskSession::new(
            AgentId::new("forge"),
            ConversationId::generate(),
            Some(ConversationId::generate()),
            Some(ConversationId::generate()),
            "run task",
        );
        fixture.status = SubagentTaskStatus::Running;
        fixture.heartbeat_at = Utc::now() - Duration::minutes(1);

        let actual = fixture.classify_with_heartbeat(Utc::now(), Duration::minutes(5));
        let expected = SubagentTaskStatus::Running;

        assert_eq!(actual.status, expected);
    }

    #[test]
    fn test_mark_completed_does_not_mark_parent_delivery() {
        let mut fixture = SubagentTaskSession::new(
            AgentId::new("forge"),
            ConversationId::generate(),
            Some(ConversationId::generate()),
            Some(ConversationId::generate()),
            "run task",
        );

        fixture.mark_completed("done");
        let actual = fixture.delivered_at;
        let expected = None;

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_mark_running_preserves_previous_terminal_payload_for_recovery() {
        let mut fixture = SubagentTaskSession::new(
            AgentId::new("forge"),
            ConversationId::generate(),
            Some(ConversationId::generate()),
            Some(ConversationId::generate()),
            "run task",
        );
        fixture.mark_completed("done");
        fixture.mark_delivered();

        fixture.mark_running();
        let actual = (
            fixture.status,
            fixture.final_result,
            fixture.final_error,
            fixture.delivered_at.is_some(),
        );
        let expected = (
            SubagentTaskStatus::Running,
            Some("done".to_string()),
            None,
            true,
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_terminal_transitions_clear_opposite_terminal_payload() {
        let mut fixture = SubagentTaskSession::new(
            AgentId::new("forge"),
            ConversationId::generate(),
            Some(ConversationId::generate()),
            Some(ConversationId::generate()),
            "run task",
        );
        fixture.mark_failed("failed");

        fixture.mark_completed("done");
        let actual = (fixture.final_result, fixture.final_error);
        let expected = (Some("done".to_string()), None);

        assert_eq!(actual, expected);
    }
}
